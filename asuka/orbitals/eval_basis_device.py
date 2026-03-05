from __future__ import annotations

"""GPU evaluation of (cartesian) basis functions on point clouds.

This is a lightweight helper around the optional CUDA extension
`asuka._orbitals_cuda_ext`. It caches device copies of packed basis arrays so
repeated evaluations (THC / grids) don't re-transfer metadata each call.

Notes
-----
- Only value evaluation is implemented (no gradients/Laplacians).
- Supports l<=10 per shell (matches the CUDA kernels in the extension).
"""

from dataclasses import dataclass
from typing import Any
import weakref

import numpy as np

try:  # optional GPU stack
    import cupy as cp  # type: ignore
    _CUDA_OK = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _CUDA_OK = False


def _require_cuda_stack():
    if not _CUDA_OK:  # pragma: no cover
        raise RuntimeError("CUDA basis-eval backend unavailable (requires cupy and asuka._orbitals_cuda_ext)")
    assert cp is not None
    if int(cp.cuda.runtime.getDeviceCount()) <= 0:  # pragma: no cover
        raise RuntimeError("CUDA basis-eval backend requested but no CUDA devices are visible")

    from asuka.kernels.orbitals import require_ext  # local import: optional extension

    ext = require_ext()
    return cp, ext


def _stream_ptr(stream: Any) -> int:
    if stream is None:
        assert cp is not None
        return int(cp.cuda.get_current_stream().ptr)
    if hasattr(stream, "ptr"):
        return int(stream.ptr)
    return int(stream)


def _nbf_from_cart_basis(basis: Any) -> int:
    if not hasattr(basis, "shell_ao_start") or not hasattr(basis, "shell_l"):
        raise TypeError("basis must provide shell_ao_start and shell_l")
    from asuka.cueri.cart import ncart  # local import to keep module light

    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int64).ravel()
    shell_l = np.asarray(basis.shell_l, dtype=np.int64).ravel()
    if shell_ao_start.shape != shell_l.shape:
        raise ValueError("basis.shell_ao_start and basis.shell_l must have identical shape")
    if shell_l.size == 0:
        return 0
    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
    return int(np.max(shell_ao_start + nfunc))


@dataclass(frozen=True)
class _DeviceBasis:
    shell_cxyz: "cp.ndarray"
    shell_prim_start: "cp.ndarray"
    shell_nprim: "cp.ndarray"
    shell_l: "cp.ndarray"
    shell_ao_start: "cp.ndarray"
    prim_exp: "cp.ndarray"
    prim_coef: "cp.ndarray"
    nbf: int


_BASIS_CACHE: dict[int, tuple[weakref.ref[Any], _DeviceBasis]] = {}


def _get_device_basis(basis: Any) -> _DeviceBasis:
    cp_mod, _ext_mod = _require_cuda_stack()

    key = id(basis)
    hit = _BASIS_CACHE.get(key)
    if hit is not None:
        ref, dev = hit
        if ref() is basis:
            return dev

    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    if shell_l.size == 0:
        raise ValueError("basis has no shells")
    lmax = int(np.max(shell_l))
    if lmax > 10:
        raise ValueError(f"CUDA basis eval supports l<=10 (got lmax={lmax})")

    nbf = _nbf_from_cart_basis(basis)
    if nbf <= 0:
        raise ValueError("basis has no functions")

    dev = _DeviceBasis(
        shell_cxyz=cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(basis.shell_cxyz, dtype=np.float64))),
        shell_prim_start=cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(basis.shell_prim_start, dtype=np.int32))),
        shell_nprim=cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(basis.shell_nprim, dtype=np.int32))),
        shell_l=cp_mod.ascontiguousarray(cp_mod.asarray(shell_l, dtype=np.int32)),
        shell_ao_start=cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(basis.shell_ao_start, dtype=np.int32))),
        prim_exp=cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(basis.prim_exp, dtype=np.float64))),
        prim_coef=cp_mod.ascontiguousarray(cp_mod.asarray(np.asarray(basis.prim_coef, dtype=np.float64))),
        nbf=int(nbf),
    )

    _BASIS_CACHE[key] = (weakref.ref(basis), dev)
    return dev


def eval_aos_cart_value_on_points_device(
    basis: Any,
    points: Any,
    *,
    out: Any | None = None,
    threads: int = 256,
    stream: Any = None,
    sync: bool = True,
) -> "cp.ndarray":
    """Evaluate contracted cartesian GTO basis values on points (GPU).

    Parameters
    ----------
    basis
        Packed cartesian basis (AO basis or auxiliary basis).
    points
        Device points array with shape (npt, 3) in Bohr. Accepts any object
        convertible to a CuPy array.
    out
        Optional preallocated output array with shape (npt, nbf) float64.

    Returns
    -------
    ao : cupy.ndarray
        Array of shape (npt, nbf) float64, C-contiguous.
    """

    cp_mod, ext = _require_cuda_stack()

    pts = cp_mod.ascontiguousarray(cp_mod.asarray(points, dtype=cp_mod.float64).reshape((-1, 3)))
    npt = int(pts.shape[0])

    dev = _get_device_basis(basis)
    nbf = int(dev.nbf)

    if out is None:
        ao = cp_mod.empty((npt, nbf), dtype=cp_mod.float64)
    else:
        ao = out
        if not isinstance(ao, cp_mod.ndarray):
            raise TypeError("out must be a CuPy array when provided")
        if ao.dtype != cp_mod.float64:
            raise TypeError("out must have dtype float64")
        if ao.shape != (npt, nbf):
            raise ValueError(f"out must have shape ({npt},{nbf}); got {tuple(map(int, ao.shape))}")
        if hasattr(ao, "flags") and not bool(ao.flags.c_contiguous):
            raise ValueError("out must be C-contiguous")

    stream_ptr = _stream_ptr(stream)

    ext.eval_aos_cart_value_f64_inplace_device(
        dev.shell_cxyz,
        dev.shell_prim_start,
        dev.shell_nprim,
        dev.shell_l,
        dev.shell_ao_start,
        dev.prim_exp,
        dev.prim_coef,
        pts,
        ao,
        threads=int(threads),
        stream_ptr=int(stream_ptr),
        sync=bool(sync),
    )
    return ao


__all__ = ["eval_aos_cart_value_on_points_device"]
