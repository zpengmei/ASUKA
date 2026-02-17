"""Runtime helpers for optional CuPy/CUDA semiempirical kernels."""

from __future__ import annotations


def _import_cupy():
    try:
        import cupy as cp  # type: ignore
    except Exception:
        return None
    return cp


def has_cupy() -> bool:
    """Return True if CuPy can be imported."""
    return _import_cupy() is not None


def has_cuda_device() -> bool:
    """Return True if a CUDA device is visible to CuPy."""
    cp = _import_cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False
