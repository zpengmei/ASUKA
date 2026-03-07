from __future__ import annotations

"""Runtime-compiled CUDA kernels for XC numerical integration.

These kernels target the dense AO-grid contractions inside :mod:`asuka.xc.numint`.
The design keeps the heavy AO evaluation in the existing CUDA extension and fuses
as much of the downstream grid algebra as possible:

- ``xc_contract_density_fused`` computes ``rho``, ``sigma``, ``tau``, and
  ``nabla rho`` in a single pass over the AO values/gradients and density matrix.
- ``xc_build_vxc_fused`` accumulates the LDA, GGA, and meta-GGA contributions to
  ``V_xc`` in one AO-pair-tiled kernel.

The kernels are compiled lazily via CuPy/NVRTC and cached by tile configuration.
"""

from pathlib import Path
from functools import lru_cache
import glob
import os
import subprocess
from typing import Any

import numpy as np

try:  # optional CUDA stack
    import cupy as cp  # type: ignore

    _CUDA_OK = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _CUDA_OK = False


def _int_env(key: str, default: int) -> int:
    raw = os.environ.get(key, "").strip()
    if raw == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - config error path
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


def _require_cupy():
    if not _CUDA_OK:  # pragma: no cover
        raise RuntimeError(
            "CuPy is required for ASUKA XC fused CUDA kernels. Install the CUDA extra "
            "(for example: pip install -e '.[cuda12]')."
        )
    assert cp is not None
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:  # pragma: no cover
        raise RuntimeError("No CUDA device is visible to CuPy")
    return cp


def _kernel_source_path() -> Path:
    return Path(__file__).with_name("cuda_kernels.cu")


def ensure_kernel_source_available() -> Path:
    src = _kernel_source_path()
    if not src.exists() or not src.is_file():  # pragma: no cover - packaging error path
        raise RuntimeError(
            "Missing XC CUDA kernel source file at "
            f"{src}. Reinstall ASUKA with package data enabled."
        )
    return src


def _kernel_source() -> str:
    return ensure_kernel_source_available().read_text(encoding="utf-8")


def _kernel_config() -> tuple[int, int, int]:
    # Tuned defaults for RTX 4090-class GPUs (float64): the density kernel
    # benefits from a slightly larger block, while the Vxc kernel prefers
    # smaller AO tiles and a larger grid-point panel for reuse.
    density_threads = int(_int_env("ASUKA_XC_DENSITY_THREADS", 192))
    density_threads = max(32, density_threads)
    if density_threads % 32 != 0:
        density_threads = 32 * ((density_threads + 31) // 32)
    if density_threads > 1024:
        raise ValueError("ASUKA_XC_DENSITY_THREADS must be <= 1024")

    ao_tile = int(_int_env("ASUKA_XC_FUSED_AO_TILE", 8))
    if ao_tile <= 0:
        raise ValueError("ASUKA_XC_FUSED_AO_TILE must be > 0")
    if ao_tile * ao_tile > 1024:
        raise ValueError("ASUKA_XC_FUSED_AO_TILE^2 must be <= 1024")

    grid_tile = int(_int_env("ASUKA_XC_FUSED_GRID_TILE", 16))
    if grid_tile <= 0:
        raise ValueError("ASUKA_XC_FUSED_GRID_TILE must be > 0")

    return int(density_threads), int(ao_tile), int(grid_tile)


def _nvrtc_host_include_options() -> tuple[str, ...]:
    """Return extra NVRTC include flags for standard headers (e.g. stddef.h).

    Some CuPy/NVRTC distributions (notably slim CUDA runtime bundles) do not ship
    the CUDA CRT headers, so NVRTC must find standard C/C++ headers via the host
    compiler include directories.  When NVRTC is invoked without an appropriate
    host include path, compilation can fail with errors such as:

        catastrophic error: cannot open source file "stddef.h"

    We attempt to locate the host compiler's include directory and pass it via
    ``-I...``. This is a no-op when the header is already resolvable.
    """

    # Prefer querying GCC when available.
    try:
        out = subprocess.check_output(
            ["gcc", "-print-file-name=include"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out and os.path.isfile(os.path.join(out, "stddef.h")):
            return (f"-I{out}",)
    except Exception:
        pass

    # Fallback: probe common GCC installation layouts.
    for inc_dir in sorted(glob.glob("/usr/lib/gcc/*/*/include")):
        if os.path.isfile(os.path.join(inc_dir, "stddef.h")):
            return (f"-I{inc_dir}",)
    return ()


@lru_cache(maxsize=None)
def _compile_module(density_threads: int, ao_tile: int, grid_tile: int):
    cp_mod = _require_cupy()
    module = cp_mod.RawModule(
        code=_kernel_source(),
        options=(
            "--std=c++11",
            *_nvrtc_host_include_options(),
            f"-DASUKA_XC_DENSITY_THREADS={int(density_threads)}",
            f"-DASUKA_XC_AO_TILE={int(ao_tile)}",
            f"-DASUKA_XC_GRID_TILE={int(grid_tile)}",
        ),
        name_expressions=(
            "xc_contract_density_fused",
            "xc_build_vxc_fused",
            "xc_build_vxc_fused_chunked",
        ),
    )
    return module


def ensure_fused_kernels_compiled() -> None:
    """Compile the fused XC kernels for the active tile configuration."""

    density_threads, ao_tile, grid_tile = _kernel_config()
    _compile_module(density_threads, ao_tile, grid_tile)


def resolve_numint_backend(
    backend: str | None,
    *,
    nao: int,
    batch_size: int,
) -> str:
    """Resolve the XC numerical-integration backend.

    Parameters
    ----------
    backend
        ``"auto"``, ``"fused"``, or ``"gemm"``. ``None`` consults the
        ``ASUKA_XC_NUMINT_BACKEND`` environment variable and defaults to
        ``"auto"``.
    nao
        Number of AO functions in the basis that enters the XC contraction.
    batch_size
        Grid batch size used by :func:`asuka.xc.numint.build_vxc`.
    """

    raw = backend if backend is not None else os.environ.get("ASUKA_XC_NUMINT_BACKEND", "auto")
    mode = str(raw).strip().lower()
    if mode in {"", "default"}:
        mode = "auto"
    if mode not in {"auto", "fused", "gemm"}:
        raise ValueError(f"Unknown XC numint backend {backend!r}; use 'auto', 'fused', or 'gemm'.")
    if mode != "auto":
        return mode

    if int(nao) <= 0 or int(batch_size) <= 0:
        return "gemm"

    fused_nao_min = max(1, int(_int_env("ASUKA_XC_FUSED_NAO_MIN", 1)))
    fused_nao_max = max(1, int(_int_env("ASUKA_XC_FUSED_NAO_MAX", 160)))
    fused_batch_min = max(1, int(_int_env("ASUKA_XC_FUSED_BATCH_MIN", 2048)))
    if fused_nao_min <= int(nao) <= fused_nao_max and int(batch_size) >= fused_batch_min:
        return "fused"
    return "gemm"


def fused_contract_density(
    phi: Any,
    dphi: Any,
    D: Any,
    *,
    out_rho: Any | None = None,
    out_sigma: Any | None = None,
    out_tau: Any | None = None,
    out_nabla: Any | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Compute ``rho``, ``sigma``, ``tau``, and ``nabla rho`` via a fused CUDA kernel."""

    cp_mod = _require_cupy()
    density_threads, ao_tile, grid_tile = _kernel_config()
    module = _compile_module(density_threads, ao_tile, grid_tile)
    kernel = module.get_function("xc_contract_density_fused")

    phi_arr = cp_mod.ascontiguousarray(cp_mod.asarray(phi, dtype=cp_mod.float64))
    dphi_arr = cp_mod.ascontiguousarray(cp_mod.asarray(dphi, dtype=cp_mod.float64))
    D_arr = cp_mod.ascontiguousarray(cp_mod.asarray(D, dtype=cp_mod.float64))

    if phi_arr.ndim != 2:
        raise ValueError("phi must be 2D with shape (npt, nao)")
    if dphi_arr.ndim != 3:
        raise ValueError("dphi must be 3D with shape (npt, nao, 3)")
    npt, nao = map(int, phi_arr.shape)
    if tuple(map(int, dphi_arr.shape)) != (npt, nao, 3):
        raise ValueError(f"dphi must have shape ({npt}, {nao}, 3)")
    if tuple(map(int, D_arr.shape)) != (nao, nao):
        raise ValueError(f"D must have shape ({nao}, {nao})")

    rho = cp_mod.empty((npt,), dtype=cp_mod.float64) if out_rho is None else out_rho
    sigma = cp_mod.empty((npt,), dtype=cp_mod.float64) if out_sigma is None else out_sigma
    tau = cp_mod.empty((npt,), dtype=cp_mod.float64) if out_tau is None else out_tau
    nabla = cp_mod.empty((npt, 3), dtype=cp_mod.float64) if out_nabla is None else out_nabla

    rho = cp_mod.asarray(rho, dtype=cp_mod.float64)
    sigma = cp_mod.asarray(sigma, dtype=cp_mod.float64)
    tau = cp_mod.asarray(tau, dtype=cp_mod.float64)
    nabla = cp_mod.asarray(nabla, dtype=cp_mod.float64)

    if tuple(map(int, rho.shape)) != (npt,):
        raise ValueError(f"out_rho must have shape ({npt},)")
    if tuple(map(int, sigma.shape)) != (npt,):
        raise ValueError(f"out_sigma must have shape ({npt},)")
    if tuple(map(int, tau.shape)) != (npt,):
        raise ValueError(f"out_tau must have shape ({npt},)")
    if tuple(map(int, nabla.shape)) != (npt, 3):
        raise ValueError(f"out_nabla must have shape ({npt}, 3)")

    kernel(
        (int(npt),),
        (int(density_threads),),
        (
            phi_arr,
            dphi_arr,
            D_arr,
            np.int32(npt),
            np.int32(nao),
            rho,
            sigma,
            tau,
            nabla,
        ),
        shared_mem=0,
    )
    return rho, sigma, tau, nabla


def fused_build_vxc(
    phi: Any,
    dphi: Any,
    weights: Any,
    vrho: Any,
    vsigma: Any,
    vtau: Any,
    nabla_rho: Any,
    *,
    out: Any | None = None,
) -> Any:
    """Build one ``V_xc`` batch contribution via a fused AO-pair CUDA kernel."""

    cp_mod = _require_cupy()
    density_threads, ao_tile, grid_tile = _kernel_config()
    module = _compile_module(density_threads, ao_tile, grid_tile)
    del density_threads, grid_tile
    kernel = module.get_function("xc_build_vxc_fused")
    kernel_chunked = module.get_function("xc_build_vxc_fused_chunked")

    phi_arr = cp_mod.ascontiguousarray(cp_mod.asarray(phi, dtype=cp_mod.float64))
    dphi_arr = cp_mod.ascontiguousarray(cp_mod.asarray(dphi, dtype=cp_mod.float64))
    weights_arr = cp_mod.ascontiguousarray(cp_mod.asarray(weights, dtype=cp_mod.float64).ravel())
    vrho_arr = cp_mod.ascontiguousarray(cp_mod.asarray(vrho, dtype=cp_mod.float64).ravel())
    vsigma_arr = cp_mod.ascontiguousarray(cp_mod.asarray(vsigma, dtype=cp_mod.float64).ravel())
    vtau_arr = cp_mod.ascontiguousarray(cp_mod.asarray(vtau, dtype=cp_mod.float64).ravel())
    nabla_arr = cp_mod.ascontiguousarray(cp_mod.asarray(nabla_rho, dtype=cp_mod.float64))

    if phi_arr.ndim != 2:
        raise ValueError("phi must be 2D with shape (npt, nao)")
    if dphi_arr.ndim != 3:
        raise ValueError("dphi must be 3D with shape (npt, nao, 3)")
    npt, nao = map(int, phi_arr.shape)
    if tuple(map(int, dphi_arr.shape)) != (npt, nao, 3):
        raise ValueError(f"dphi must have shape ({npt}, {nao}, 3)")
    if tuple(map(int, weights_arr.shape)) != (npt,):
        raise ValueError(f"weights must have shape ({npt},)")
    if tuple(map(int, vrho_arr.shape)) != (npt,):
        raise ValueError(f"vrho must have shape ({npt},)")
    if tuple(map(int, vsigma_arr.shape)) != (npt,):
        raise ValueError(f"vsigma must have shape ({npt},)")
    if tuple(map(int, vtau_arr.shape)) != (npt,):
        raise ValueError(f"vtau must have shape ({npt},)")
    if tuple(map(int, nabla_arr.shape)) != (npt, 3):
        raise ValueError(f"nabla_rho must have shape ({npt}, 3)")

    V = cp_mod.empty((nao, nao), dtype=cp_mod.float64) if out is None else out
    V = cp_mod.asarray(V, dtype=cp_mod.float64)
    if tuple(map(int, V.shape)) != (nao, nao):
        raise ValueError(f"out must have shape ({nao}, {nao})")

    grid_x = (int(nao) + int(ao_tile) - 1) // int(ao_tile)

    # Optional z-dimension chunking over grid points to improve occupancy for
    # smaller AO sizes. Chunking uses atomicAdd into the output matrix, so we
    # only enable it when there are enough points to amortize the overhead.
    force_chunks = int(_int_env("ASUKA_XC_FUSED_POINT_CHUNKS", 0))
    max_chunks = max(1, int(_int_env("ASUKA_XC_FUSED_POINT_CHUNKS_MAX", 64)))
    chunk_min_pts = max(1, int(_int_env("ASUKA_XC_FUSED_CHUNK_MIN_PTS", 2048)))
    blocks_per_sm = max(1, int(_int_env("ASUKA_XC_FUSED_TARGET_BLOCKS_PER_SM", 4)))
    try:
        sm_count = int(cp_mod.cuda.Device().attributes.get("MultiProcessorCount", 1))
    except Exception:
        sm_count = 1

    ntiles = int(grid_x)
    tri_tiles = int(ntiles * (ntiles + 1) // 2)
    target_blocks = int(blocks_per_sm * max(1, sm_count))
    if force_chunks > 0:
        nchunks = min(int(force_chunks), int(max_chunks))
    else:
        nchunks = (target_blocks + tri_tiles - 1) // tri_tiles
        nchunks = max(1, min(int(nchunks), int(max_chunks)))
        max_by_points = max(1, int(npt) // int(chunk_min_pts))
        nchunks = max(1, min(int(nchunks), int(max_by_points)))

    use_chunked = bool(nchunks > 1)
    if use_chunked:
        V.fill(0.0)
    grid = (int(grid_x), int(grid_x), int(nchunks))
    block = (int(ao_tile), int(ao_tile), 1)

    kernel_use = kernel_chunked if use_chunked else kernel
    kernel_use(
        grid,
        block,
        (
            phi_arr,
            dphi_arr,
            weights_arr,
            vrho_arr,
            vsigma_arr,
            vtau_arr,
            nabla_arr,
            np.int32(npt),
            np.int32(nao),
            V,
        ),
    )
    return V


__all__ = [
    "ensure_fused_kernels_compiled",
    "ensure_kernel_source_available",
    "fused_build_vxc",
    "fused_contract_density",
    "resolve_numint_backend",
]
