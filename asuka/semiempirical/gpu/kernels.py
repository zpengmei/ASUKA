"""Kernel loading and data packing for semiempirical CUDA Fock updates."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from asuka.nddo_core import build_pair_ri_payload, nao_for_Z

from .runtime import _import_cupy

_FOCK_KERNEL_CACHE = None
_GRADIENT_KERNEL_CACHE = None


def _fock_kernel_source_path() -> Path:
    return Path(__file__).with_suffix("").parent / "cuda" / "fock_kernels.cu"


def ensure_kernel_source_available() -> Path:
    """Return CUDA kernel source path and fail with a clear packaging error if missing."""
    src = _fock_kernel_source_path()
    if not src.exists():
        raise RuntimeError(
            "Missing AM1 CUDA kernel source file at "
            f"{src}. Reinstall ASUKA with package data enabled."
        )
    if not src.is_file():
        raise RuntimeError(
            f"AM1 CUDA kernel path exists but is not a file: {src}"
        )
    return src


def _fock_kernel_source() -> str:
    src = ensure_kernel_source_available()
    return src.read_text(encoding="utf-8")


def _gradient_kernel_source_path() -> Path:
    return Path(__file__).with_suffix("").parent / "cuda" / "gradient_kernels.cu"


def _pair_math_header_path() -> Path:
    return Path(__file__).with_suffix("").parent / "cuda" / "pair_math.cuh"


def ensure_gradient_kernel_sources_available() -> Tuple[Path, Path]:
    """Return CUDA gradient source/header paths with actionable packaging errors."""
    src = _gradient_kernel_source_path()
    hdr = _pair_math_header_path()
    if not src.exists() or not src.is_file():
        raise RuntimeError(
            "Missing AM1 CUDA gradient kernel source file at "
            f"{src}. Reinstall ASUKA with package data enabled."
        )
    if not hdr.exists() or not hdr.is_file():
        raise RuntimeError(
            "Missing AM1 CUDA gradient math header file at "
            f"{hdr}. Reinstall ASUKA with package data enabled."
        )
    return src, hdr


def _gradient_kernel_source() -> str:
    src, hdr = ensure_gradient_kernel_sources_available()
    # RawModule compiles a single translation unit; prepend shared math header.
    return hdr.read_text(encoding="utf-8") + "\n\n" + src.read_text(encoding="utf-8")


def _require_cupy_cuda():
    cp = _import_cupy()
    if cp is None:
        raise RuntimeError(
            "CuPy is required for semiempirical CUDA kernels. "
            "Install ASUKA with the CUDA extra (for example: pip install -e '.[cuda]')."
        )
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            "Unable to query CUDA devices via CuPy runtime; CUDA semiempirical kernels "
            "cannot be compiled."
        ) from exc
    if ndev < 1:
        raise RuntimeError("No CUDA device is visible to CuPy")
    return cp


def get_fock_kernels() -> Dict[str, object]:
    """Compile and cache CUDA kernels for one-center and two-center Fock updates."""
    global _FOCK_KERNEL_CACHE
    if _FOCK_KERNEL_CACHE is not None:
        return _FOCK_KERNEL_CACHE

    cp = _require_cupy_cuda()
    ensure_kernel_source_available()

    module = cp.RawModule(
        code=_fock_kernel_source(),
        options=("--std=c++11",),
        name_expressions=(
            "onecenter_fock_kernel",
            "twocenter_fock_kernel",
            "twocenter_fock_ri_kernel",
            "twocenter_fock_ri_11_kernel",
            "twocenter_fock_ri_14_kernel",
            "twocenter_fock_ri_41_kernel",
            "twocenter_fock_ri_44_kernel",
            "build_wblocks_from_ri_kernel",
        ),
    )
    _FOCK_KERNEL_CACHE = {
        "onecenter": module.get_function("onecenter_fock_kernel"),
        "twocenter": module.get_function("twocenter_fock_kernel"),
        "twocenter_ri": module.get_function("twocenter_fock_ri_kernel"),
        "twocenter_ri_11": module.get_function("twocenter_fock_ri_11_kernel"),
        "twocenter_ri_14": module.get_function("twocenter_fock_ri_14_kernel"),
        "twocenter_ri_41": module.get_function("twocenter_fock_ri_41_kernel"),
        "twocenter_ri_44": module.get_function("twocenter_fock_ri_44_kernel"),
        "build_wblocks_from_ri": module.get_function("build_wblocks_from_ri_kernel"),
    }
    return _FOCK_KERNEL_CACHE


def get_gradient_kernels() -> Dict[str, object]:
    """Compile and cache CUDA kernels for AM1 analytical pair gradients."""
    global _GRADIENT_KERNEL_CACHE
    if _GRADIENT_KERNEL_CACHE is not None:
        return _GRADIENT_KERNEL_CACHE

    cp = _require_cupy_cuda()
    ensure_gradient_kernel_sources_available()

    module = cp.RawModule(
        code=_gradient_kernel_source(),
        options=("--std=c++11",),
        name_expressions=(
            "am1_grad_pair_11_kernel",
            "am1_grad_pair_14_kernel",
            "am1_grad_pair_41_kernel",
            "am1_grad_pair_44_kernel",
        ),
    )
    _GRADIENT_KERNEL_CACHE = {
        "11": module.get_function("am1_grad_pair_11_kernel"),
        "14": module.get_function("am1_grad_pair_14_kernel"),
        "41": module.get_function("am1_grad_pair_41_kernel"),
        "44": module.get_function("am1_grad_pair_44_kernel"),
    }
    return _GRADIENT_KERNEL_CACHE


def pack_onecenter_eris(atomic_numbers: Sequence[int], onecenter_eris: List[np.ndarray]) -> np.ndarray:
    """Pack one-center tensors to fixed-size [nat, 256] storage."""
    nat = len(atomic_numbers)
    out = np.zeros((nat, 256), dtype=np.float64)

    for A, Z in enumerate(atomic_numbers):
        nao = nao_for_Z(Z)
        G = onecenter_eris[A]
        for m in range(nao):
            for n in range(nao):
                for l in range(nao):
                    for s in range(nao):
                        idx = ((m * 4 + n) * 4 + l) * 4 + s
                        out[A, idx] = G[m, n, l, s]
    return out


def pack_twocenter_blocks(
    atomic_numbers: Sequence[int],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    W_list: List[np.ndarray],
) -> np.ndarray:
    """Pack two-center tensors to fixed-size [npairs, 256] storage."""
    npairs = len(pair_i)
    out = np.zeros((npairs, 256), dtype=np.float64)

    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        naoA = nao_for_Z(atomic_numbers[iA])
        naoB = nao_for_Z(atomic_numbers[iB])
        W = W_list[k]

        for m in range(naoA):
            for n in range(naoA):
                for l in range(naoB):
                    for s in range(naoB):
                        idx = ((m * 4 + n) * 4 + l) * 4 + s
                        out[k, idx] = W[m, n, l, s]

    return out


def pack_twocenter_ri(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    mp_params,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack RI invariants and transforms for fused RI CUDA Fock kernels.

    Returns
    -------
    ri_pack : (npairs, 22) float64
        Rotational invariants for each pair.
    ta_pack : (npairs, 16) float64
        Flattened 4x4 local->molecular AO transform for atom A.
    tb_pack : (npairs, 16) float64
        Flattened 4x4 local->molecular AO transform for atom B.
    """
    ri_pack, ta_pack, tb_pack, _, _, _ = build_pair_ri_payload(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        pair_i=pair_i,
        pair_j=pair_j,
        mp_params=mp_params,
    )
    return ri_pack, ta_pack, tb_pack


def build_wblocks_from_ri_device(
    cp,
    kernels,
    pair_i_d,
    pair_j_d,
    ao_off_d,
    nao_atom_d,
    ri_d,
    ta_d,
    tb_d,
    npairs: int,
):
    """Materialize packed W blocks on device from RI payload once per run."""
    if npairs <= 0:
        return cp.empty((0,), dtype=cp.float64)
    out = cp.zeros((int(npairs), 256), dtype=cp.float64)
    kernels["build_wblocks_from_ri"](
        (int(npairs),),
        (32,),
        (
            pair_i_d,
            pair_j_d,
            ao_off_d,
            nao_atom_d,
            ri_d,
            ta_d,
            tb_d,
            out,
            np.int32(int(npairs)),
        ),
    )
    return out


def build_pair_buckets(
    atomic_numbers: Sequence[int],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build pair buckets by AO block sizes.

    Buckets are keyed by A/B AO counts: ``\"11\"``, ``\"14\"``, ``\"41\"``, ``\"44\"``.
    """
    b11: List[int] = []
    b14: List[int] = []
    b41: List[int] = []
    b44: List[int] = []
    for k in range(len(pair_i)):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        naoA = nao_for_Z(int(atomic_numbers[iA]))
        naoB = nao_for_Z(int(atomic_numbers[iB]))
        if naoA == 1 and naoB == 1:
            b11.append(k)
        elif naoA == 1 and naoB == 4:
            b14.append(k)
        elif naoA == 4 and naoB == 1:
            b41.append(k)
        elif naoA == 4 and naoB == 4:
            b44.append(k)
        else:
            raise ValueError(f"Unsupported pair block shape ({naoA}, {naoB})")
    return {
        "11": np.asarray(b11, dtype=np.int32),
        "14": np.asarray(b14, dtype=np.int32),
        "41": np.asarray(b41, dtype=np.int32),
        "44": np.asarray(b44, dtype=np.int32),
    }
