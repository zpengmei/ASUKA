from __future__ import annotations

import numpy as np

try:  # optional CUDA extension
    from asuka import _guga_cuda_linalg_ext as _ext
except Exception:  # pragma: no cover
    _ext = None


def has_cuda_linalg_ext() -> bool:
    return _ext is not None


def device_info() -> dict[str, object]:
    if _ext is None:
        raise RuntimeError("CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext")
    return dict(_ext.device_info())


def mem_info() -> dict[str, int]:
    if _ext is None:
        raise RuntimeError("CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext")
    return dict(_ext.mem_info())


def cublas_emulation_info() -> dict[str, object]:
    if _ext is None:
        raise RuntimeError("CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext")
    return dict(_ext.cublas_emulation_info())


def eigh_sym(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if _ext is None:
        raise RuntimeError("CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext")
    w, v = _ext.eigh_sym(np.asarray(a, dtype=np.float64, order="C"))
    return np.asarray(w, dtype=np.float64), np.asarray(v, dtype=np.float64)


def gemm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if _ext is None:
        raise RuntimeError("CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext")
    return np.asarray(_ext.gemm(np.asarray(a, dtype=np.float64, order="C"), np.asarray(b, dtype=np.float64, order="C")), dtype=np.float64)


def davidson_dense_sym(
    a: np.ndarray,
    *,
    nroots: int = 1,
    max_cycle: int = 50,
    max_space: int = 12,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    denom_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if _ext is None:
        raise RuntimeError("CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext")
    conv, e, x, niter = _ext.davidson_dense_sym(
        np.asarray(a, dtype=np.float64, order="C"),
        int(nroots),
        int(max_cycle),
        int(max_space),
        float(tol),
        float(lindep),
        float(denom_tol),
    )
    return (
        np.asarray(conv, dtype=np.bool_),
        np.asarray(e, dtype=np.float64),
        np.asarray(x, dtype=np.float64),
        int(niter),
    )


class DenseSymDavidsonWorkspace:
    def __init__(self, n: int, *, nroots: int = 1, max_space: int = 12) -> None:
        if _ext is None:
            raise RuntimeError(
                "CUDA linalg extension not available; build with python -m asuka.build.guga_cuda_linalg_ext"
            )
        self._ws = _ext.DenseSymDavidsonWorkspace(int(n), int(nroots), int(max_space))

    @property
    def n(self) -> int:
        return int(self._ws.n)

    @property
    def nroots(self) -> int:
        return int(self._ws.nroots)

    @property
    def max_space_eff(self) -> int:
        return int(self._ws.max_space_eff)

    def set_matrix(self, a: np.ndarray) -> None:
        self._ws.set_matrix(np.asarray(a, dtype=np.float64, order="C"))

    def cublas_emulation_info(self) -> dict[str, object]:
        return dict(self._ws.cublas_emulation_info())

    def gemm_backend(self) -> str:
        return str(self._ws.gemm_backend())

    def set_gemm_backend(self, backend: str) -> None:
        self._ws.set_gemm_backend(str(backend))

    def gemm_algo(self) -> int:
        return int(self._ws.gemm_algo())

    def set_gemm_algo(self, algo: int) -> None:
        self._ws.set_gemm_algo(int(algo))

    def set_cublas_math_mode(self, mode: str) -> None:
        self._ws.set_cublas_math_mode(str(mode))

    def cublas_workspace_bytes(self) -> int:
        return int(self._ws.cublas_workspace_bytes())

    def set_cublas_workspace_bytes(self, bytes_: int) -> None:
        self._ws.set_cublas_workspace_bytes(int(bytes_))

    def autoset_cublas_workspace_bytes(self, *, cap_mb: int = 2048) -> int:
        """Auto-size cuBLAS workspace for emulated-FP64 GEMMEx (safe bound; capped)."""

        from asuka.cuda.cublas_workspace import recommend_cublas_workspace_bytes_for_emulated_fp64_gemm

        cap_bytes = int(cap_mb) * 1024 * 1024
        ws_info = dict(self.cublas_emulation_info())
        ws_info["gemm_backend"] = str(self.gemm_backend())
        n = int(self.n)
        m = int(self.max_space_eff)
        rec = recommend_cublas_workspace_bytes_for_emulated_fp64_gemm(
            ws_info=ws_info,
            gemm_shapes=[
                (int(n), int(m), int(n)),  # W = A @ V
                (int(m), int(m), int(n)),  # Hsub = V^T @ W
            ],
            batch_count=1,
            is_complex=False,
            cap_bytes=int(cap_bytes),
        )
        self.set_cublas_workspace_bytes(int(rec))
        return int(rec)

    def set_cublas_emulation_strategy(self, strategy: str) -> None:
        self._ws.set_cublas_emulation_strategy(str(strategy))

    def set_cublas_emulation_special_values_support(self, mask: int) -> None:
        self._ws.set_cublas_emulation_special_values_support(int(mask))

    def set_cublas_fixed_point_mantissa_control(self, control: str) -> None:
        self._ws.set_cublas_fixed_point_mantissa_control(str(control))

    def set_cublas_fixed_point_max_mantissa_bits(self, max_bits: int) -> None:
        self._ws.set_cublas_fixed_point_max_mantissa_bits(int(max_bits))

    def set_cublas_fixed_point_mantissa_bit_offset(self, bit_offset: int) -> None:
        self._ws.set_cublas_fixed_point_mantissa_bit_offset(int(bit_offset))

    def solve(
        self,
        *,
        max_cycle: int = 50,
        tol: float = 1e-10,
        lindep: float = 1e-14,
        denom_tol: float = 1e-12,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        conv, e, x, niter = self._ws.solve(int(max_cycle), float(tol), float(lindep), float(denom_tol))
        return (
            np.asarray(conv, dtype=np.bool_),
            np.asarray(e, dtype=np.float64),
            np.asarray(x, dtype=np.float64),
            int(niter),
        )
