from __future__ import annotations

from typing import Any


def _import_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for spherical int1e CUDA contraction") from e
    return cp


def _import_ext():
    try:
        from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cuERI CUDA extension is unavailable") from e
    return _ext


def has_int1e_sph_cuda_kernels() -> bool:
    try:
        _ext = _import_ext()
    except Exception:
        return False
    return bool(
        hasattr(_ext, "int1e_dS_deriv_contracted_sph_inplace_device")
        and hasattr(_ext, "int1e_dhcore_deriv_contracted_sph_inplace_device")
    )


def contract_dS_sph_prebuilt_cuda(
    dS_sph: Any,
    M_sph: Any,
    *,
    threads: int = 256,
    return_device: bool = False,
) -> Any:
    """Contract prebuilt spherical dS tensor on CUDA."""

    cp = _import_cupy()
    _ext = _import_ext()
    if not hasattr(_ext, "int1e_dS_deriv_contracted_sph_inplace_device"):
        raise RuntimeError("CUDA extension missing int1e_dS_deriv_contracted_sph_inplace_device")

    dS = cp.asarray(dS_sph, dtype=cp.float64)
    if dS.ndim != 4:
        raise ValueError("dS_sph must have shape (natm,3,nao_sph,nao_sph)")
    natm, n3, nao0, nao1 = map(int, dS.shape)
    if n3 != 3 or nao0 != nao1:
        raise ValueError("dS_sph must have shape (natm,3,nao_sph,nao_sph)")
    dS = cp.ascontiguousarray(dS, dtype=cp.float64)

    M = cp.asarray(M_sph, dtype=cp.float64)
    if M.ndim != 2 or int(M.shape[0]) != int(M.shape[1]) or int(M.shape[0]) != int(nao0):
        raise ValueError("M_sph must have shape (nao_sph,nao_sph) matching dS_sph")
    M = cp.ascontiguousarray(M, dtype=cp.float64)

    out = cp.zeros((natm, 3), dtype=cp.float64)
    _ext.int1e_dS_deriv_contracted_sph_inplace_device(
        dS.reshape(-1),
        M.reshape(-1),
        int(natm),
        int(nao0),
        out.reshape(-1),
        int(threads),
        int(cp.cuda.get_current_stream().ptr),
        False,
    )
    if return_device:
        return out
    return cp.asnumpy(out)


def contract_dhcore_sph_prebuilt_cuda(
    dT_sph: Any,
    dV_sph: Any,
    M_sph: Any,
    *,
    threads: int = 256,
    return_device: bool = False,
) -> Any:
    """Contract prebuilt spherical dT/dV tensors on CUDA."""

    cp = _import_cupy()
    _ext = _import_ext()
    if not hasattr(_ext, "int1e_dhcore_deriv_contracted_sph_inplace_device"):
        raise RuntimeError("CUDA extension missing int1e_dhcore_deriv_contracted_sph_inplace_device")

    dT = cp.asarray(dT_sph, dtype=cp.float64)
    dV = cp.asarray(dV_sph, dtype=cp.float64)
    if dT.ndim != 4 or dV.ndim != 4:
        raise ValueError("dT_sph/dV_sph must have shape (natm,3,nao_sph,nao_sph)")
    if tuple(map(int, dT.shape)) != tuple(map(int, dV.shape)):
        raise ValueError("dT_sph and dV_sph shape mismatch")
    natm, n3, nao0, nao1 = map(int, dT.shape)
    if n3 != 3 or nao0 != nao1:
        raise ValueError("dT_sph/dV_sph must have shape (natm,3,nao_sph,nao_sph)")
    dT = cp.ascontiguousarray(dT, dtype=cp.float64)
    dV = cp.ascontiguousarray(dV, dtype=cp.float64)

    M = cp.asarray(M_sph, dtype=cp.float64)
    if M.ndim != 2 or int(M.shape[0]) != int(M.shape[1]) or int(M.shape[0]) != int(nao0):
        raise ValueError("M_sph must have shape (nao_sph,nao_sph) matching dT_sph/dV_sph")
    M = cp.ascontiguousarray(M, dtype=cp.float64)

    out = cp.zeros((natm, 3), dtype=cp.float64)
    _ext.int1e_dhcore_deriv_contracted_sph_inplace_device(
        dT.reshape(-1),
        dV.reshape(-1),
        M.reshape(-1),
        int(natm),
        int(nao0),
        out.reshape(-1),
        int(threads),
        int(cp.cuda.get_current_stream().ptr),
        False,
    )
    if return_device:
        return out
    return cp.asnumpy(out)


__all__ = [
    "has_int1e_sph_cuda_kernels",
    "contract_dS_sph_prebuilt_cuda",
    "contract_dhcore_sph_prebuilt_cuda",
]
