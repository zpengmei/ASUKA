from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _try_import_cupy():
    try:  # pragma: no cover - optional
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        return None
    return cp


def _is_cupy_array(x: Any) -> bool:
    cp = _try_import_cupy()
    return bool(cp is not None and isinstance(x, cp.ndarray))


def _as_numpy_f64(x: Any) -> np.ndarray:
    cp = _try_import_cupy()
    if cp is not None and isinstance(x, cp.ndarray):
        return np.asarray(cp.asnumpy(x), dtype=np.float64, order="C")
    return np.asarray(x, dtype=np.float64, order="C")


def _pair_density_from_flat(flat: Any, norb: int, xp: Any):
    flat_v = xp.asarray(flat, dtype=xp.float64).reshape(int(norb), int(norb))
    return 0.5 * (flat_v + flat_v.T)


@dataclass
class DirectRowOracleIntegrals:
    """Exact active-space two-electron operator backed by direct J builds.

    This object provides the minimal row-oracle surface expected by the sparse
    GUGA solver path without materializing active-space ERIs.
    """

    norb: int
    provider: Any
    C_act: Any
    backend: str = "cuda"
    _j_ps_cache: np.ndarray | None = None
    _pair_norm_cache: np.ndarray | None = None
    _eri_ppqq_cache: np.ndarray | None = None
    _eri_pqqp_cache: np.ndarray | None = None
    _j_ps_device_cache: Any | None = None
    _pair_norm_device_cache: Any | None = None
    _eri_ppqq_device_cache: Any | None = None
    _eri_pqqp_device_cache: Any | None = None

    def __post_init__(self) -> None:
        self.norb = int(self.norb)
        if self.norb <= 0:
            raise ValueError("norb must be > 0")
        C_act_np = _as_numpy_f64(self.C_act)
        if C_act_np.ndim != 2 or int(C_act_np.shape[1]) != int(self.norb):
            raise ValueError("C_act must have shape (nao, norb)")

    @property
    def l_full(self) -> None:
        return None

    @property
    def naux(self) -> int:
        return 0

    def _xp(self):
        probe = getattr(self.provider, "probe_array", lambda: None)()
        if _is_cupy_array(probe) or _is_cupy_array(self.C_act):
            cp = _try_import_cupy()
            if cp is not None:
                return cp
        return np

    def _cp(self):
        cp = _try_import_cupy()
        if cp is None:
            raise RuntimeError("CuPy is required for cuda_direct exact operator paths")
        return cp

    def _as_backend(self, x: Any):
        xp = self._xp()
        return xp.asarray(x, dtype=xp.float64)

    def _project_density_to_ao(self, D_act: Any):
        xp = self._xp()
        C_act = xp.asarray(self.C_act, dtype=xp.float64)
        D_act = xp.asarray(D_act, dtype=xp.float64)
        D_sym = 0.5 * (D_act + D_act.T)
        return C_act @ D_sym @ C_act.T

    def _project_j_to_active(self, J_ao: Any):
        xp = self._xp()
        C_act = xp.asarray(self.C_act, dtype=xp.float64)
        J_ao = xp.asarray(J_ao, dtype=xp.float64)
        return C_act.T @ J_ao @ C_act

    def _active_j(self, D_act: Any):
        D_ao = self._project_density_to_ao(D_act)
        J_ao, _ = self.provider.jk(D_ao, want_J=True, want_K=False, profile=None)
        if J_ao is None:  # pragma: no cover
            raise RuntimeError("direct provider returned no Coulomb matrix")
        return self._project_j_to_active(J_ao)

    @property
    def supports_cuda_direct(self) -> bool:
        return bool(
            callable(getattr(self.provider, "jk", None))
            and callable(getattr(self.provider, "jk_multi2", None))
            and _try_import_cupy() is not None
        )

    def _active_j_batched_device(self, D0_act: Any, D1_act: Any | None = None, *, profile: dict | None = None):
        cp = self._cp()
        C_act = cp.asarray(self.C_act, dtype=cp.float64)
        D0 = cp.asarray(D0_act, dtype=cp.float64)
        D0 = 0.5 * (D0 + D0.T)
        D0_ao = C_act @ D0 @ C_act.T
        if D1_act is None:
            J0_ao, _ = self.provider.jk(D0_ao, want_J=True, want_K=False, profile=profile)
            if J0_ao is None:  # pragma: no cover
                raise RuntimeError("direct provider returned no Coulomb matrix")
            J0 = C_act.T @ cp.asarray(J0_ao, dtype=cp.float64) @ C_act
            return J0, None
        D1 = cp.asarray(D1_act, dtype=cp.float64)
        D1 = 0.5 * (D1 + D1.T)
        D1_ao = C_act @ D1 @ C_act.T
        J0_ao, _K0_ao, J1_ao, _K1_ao = self.provider.jk_multi2(
            D0_ao,
            D1_ao,
            want_J=True,
            want_K=False,
            profile=profile,
        )
        if J0_ao is None or J1_ao is None:  # pragma: no cover
            raise RuntimeError("direct provider returned no Coulomb matrix")
        J0 = C_act.T @ cp.asarray(J0_ao, dtype=cp.float64) @ C_act
        J1 = C_act.T @ cp.asarray(J1_ao, dtype=cp.float64) @ C_act
        return J0, J1

    def contract_w_block_device(
        self,
        w_block: Any,
        *,
        half: float = 0.5,
        out: Any | None = None,
        profile: dict | None = None,
    ):
        cp = self._cp()
        w = cp.ascontiguousarray(cp.asarray(w_block, dtype=cp.float64))
        if w.ndim != 2 or int(w.shape[1]) != int(self.norb) * int(self.norb):
            raise ValueError("w_block must have shape (k, norb*norb)")
        nrows = int(w.shape[0])
        norb = int(self.norb)
        if out is None:
            out_dev = cp.empty((nrows, norb * norb), dtype=cp.float64)
        else:
            out_dev = cp.asarray(out, dtype=cp.float64)
            if out_dev.shape != (nrows, norb * norb):
                raise ValueError("out must have shape (k, norb*norb)")
        if nrows == 0:
            return out_dev
        w3 = w.reshape(nrows, norb, norb)
        for row0 in range(0, nrows, 2):
            row1 = min(nrows, row0 + 2)
            D0 = w3[row0]
            if row1 - row0 == 2:
                J0, J1 = self._active_j_batched_device(D0, w3[row0 + 1], profile=profile)
                out_dev[row0] = float(half) * J0.reshape(norb * norb)
                if J1 is None:  # pragma: no cover
                    raise RuntimeError("internal error: missing second J block")
                out_dev[row0 + 1] = float(half) * J1.reshape(norb * norb)
            else:
                J0, _ = self._active_j_batched_device(D0, None, profile=profile)
                out_dev[row0] = float(half) * J0.reshape(norb * norb)
        return out_dev

    def build_diag_slices_device(self):
        cp = self._cp()
        if self._eri_ppqq_device_cache is not None and self._eri_pqqp_device_cache is not None:
            return self._eri_ppqq_device_cache, self._eri_pqqp_device_cache
        norb = int(self.norb)
        eri_ppqq = cp.empty((norb, norb), dtype=cp.float64)
        eri_pqqp = cp.empty((norb, norb), dtype=cp.float64)
        eye = cp.eye(norb, dtype=cp.float64)
        for q0 in range(0, norb, 2):
            q1 = min(norb, q0 + 2)
            D0 = cp.diag(eye[q0])
            if q1 - q0 == 2:
                J0, J1 = self._active_j_batched_device(D0, cp.diag(eye[q0 + 1]))
                eri_ppqq[:, q0] = cp.diag(J0)
                if J1 is None:  # pragma: no cover
                    raise RuntimeError("internal error: missing second diagonal J block")
                eri_ppqq[:, q0 + 1] = cp.diag(J1)
            else:
                J0, _ = self._active_j_batched_device(D0, None)
                eri_ppqq[:, q0] = cp.diag(J0)
        pair_ids = [(p, q) for p in range(norb) for q in range(norb)]
        for idx0 in range(0, len(pair_ids), 2):
            p0, q0 = pair_ids[idx0]
            D0 = cp.zeros((norb, norb), dtype=cp.float64)
            D0[q0, p0] = 1.0
            if idx0 + 1 < len(pair_ids):
                p1, q1 = pair_ids[idx0 + 1]
                D1 = cp.zeros((norb, norb), dtype=cp.float64)
                D1[q1, p1] = 1.0
                J0, J1 = self._active_j_batched_device(D0, D1)
                eri_pqqp[p0, q0] = J0[p0, q0]
                if J1 is None:  # pragma: no cover
                    raise RuntimeError("internal error: missing second exchange J block")
                eri_pqqp[p1, q1] = J1[p1, q1]
            else:
                J0, _ = self._active_j_batched_device(D0, None)
                eri_pqqp[p0, q0] = J0[p0, q0]
        self._eri_ppqq_device_cache = cp.ascontiguousarray(eri_ppqq)
        self._eri_pqqp_device_cache = cp.ascontiguousarray(eri_pqqp)
        return self._eri_ppqq_device_cache, self._eri_pqqp_device_cache

    @property
    def j_ps(self) -> np.ndarray:
        if self._j_ps_cache is None:
            xp = self._xp()
            C_act = xp.asarray(self.C_act, dtype=xp.float64)
            D_ao = C_act @ xp.eye(int(self.norb), dtype=xp.float64) @ C_act.T
            _J_ao, K_ao = self.provider.jk(D_ao, want_J=True, want_K=True, profile=None)
            if K_ao is None:  # pragma: no cover
                raise RuntimeError("direct provider returned no exchange matrix")
            K_act = C_act.T @ xp.asarray(K_ao, dtype=xp.float64) @ C_act
            self._j_ps_cache = _as_numpy_f64(K_act)
        return self._j_ps_cache

    @property
    def j_ps_device(self):
        cp = self._cp()
        if self._j_ps_device_cache is None:
            self._j_ps_device_cache = cp.ascontiguousarray(cp.asarray(self.j_ps, dtype=cp.float64))
        return self._j_ps_device_cache

    @property
    def pair_norm(self) -> np.ndarray:
        if self._pair_norm_cache is None:
            xp = self._xp()
            norb = int(self.norb)
            pair_norm = np.empty((norb * norb,), dtype=np.float64)
            for p in range(norb):
                for q in range(norb):
                    D_act = xp.zeros((norb, norb), dtype=xp.float64)
                    D_act[int(q), int(p)] = 1.0
                    J_act = self._active_j(D_act)
                    J_np = _as_numpy_f64(J_act)
                    val = max(float(J_np[int(p), int(q)]), 0.0)
                    pair_norm[int(p) * norb + int(q)] = float(np.sqrt(val))
            self._pair_norm_cache = np.asarray(pair_norm, dtype=np.float64, order="C")
        return self._pair_norm_cache

    @property
    def pair_norm_device(self):
        cp = self._cp()
        if self._pair_norm_device_cache is None:
            if self._eri_pqqp_device_cache is None:
                _ppqq, pqqp = self.build_diag_slices_device()
            else:
                pqqp = self._eri_pqqp_device_cache
            self._pair_norm_device_cache = cp.sqrt(cp.maximum(pqqp.reshape(int(self.norb) * int(self.norb)), 0.0))
        return self._pair_norm_device_cache

    def _maybe_build_eri_mat(self, eri_mat_max_bytes: int) -> None:
        _ = int(eri_mat_max_bytes)
        return None

    def contract_cols(
        self,
        pair_ids: np.ndarray,
        coeff: np.ndarray,
        *,
        half: float = 0.5,
        eri_mat_max_bytes: int = 0,
    ) -> np.ndarray:
        _ = int(eri_mat_max_bytes)
        pair_ids = np.asarray(pair_ids, dtype=np.int32).ravel()
        coeff = np.asarray(coeff, dtype=np.float64).ravel()
        if pair_ids.size != coeff.size:
            raise ValueError("pair_ids and coeff must have same length")
        norb = int(self.norb)
        if pair_ids.size == 0:
            return np.zeros((norb * norb,), dtype=np.float64)
        xp = self._xp()
        D_act = xp.zeros((norb, norb), dtype=xp.float64)
        D_flat = D_act.reshape(norb * norb)
        coeff_backend = xp.asarray(coeff, dtype=xp.float64)
        if pair_ids.size == 1:
            D_flat[int(pair_ids[0])] = coeff_backend[0]
        else:
            xp.add.at(D_flat, xp.asarray(pair_ids, dtype=xp.int32), coeff_backend)
        J_act = self._active_j(D_act)
        return float(half) * _as_numpy_f64(J_act).reshape(norb * norb)

    def rr_slice_h_eff(self, occ: np.ndarray, *, half: float = 0.5, eri_mat_max_bytes: int = 0) -> np.ndarray:
        _ = int(eri_mat_max_bytes)
        occ = np.asarray(occ, dtype=np.float64).ravel()
        norb = int(self.norb)
        if occ.size != norb:
            raise ValueError("occ has wrong length")
        xp = self._xp()
        D_act = xp.asarray(np.diag(occ), dtype=xp.float64)
        J_act = self._active_j(D_act)
        return float(half) * _as_numpy_f64(J_act)


__all__ = ["DirectRowOracleIntegrals"]


@dataclass
class DeviceDirectMOIntegrals(DirectRowOracleIntegrals):
    """Exact active-space operator for CUDA paths without ERI materialization.

    This keeps the row-oracle surface for fallback/reference paths, but adds
    device-native helpers used by the CUDA matvec workspace.
    """

    _j_ps_cache_device: Any | None = None
    _pair_norm_cache_device: Any | None = None
    _eri_ppqq_cache_device: Any | None = None
    _eri_pqqp_cache_device: Any | None = None
    _pu_qv_cache_device: Any | None = None

    def _active_pu_qv_device(self):
        if self._pu_qv_cache_device is None:
            xp = self._xp()
            norb = int(self.norb)
            C_act = xp.asarray(self.C_act, dtype=xp.float64)
            pu_qv = xp.asarray(
                self.provider.build_pu_qv(C_act, C_act, out=None, profile=None),
                dtype=xp.float64,
            ).reshape(norb, norb, norb, norb)
            self._pu_qv_cache_device = xp.ascontiguousarray(pu_qv)
        return self._pu_qv_cache_device

    def _active_j(self, D_act: Any):
        xp = self._xp()
        pu_qv = self._active_pu_qv_device()
        D_act = xp.asarray(D_act, dtype=xp.float64)
        return xp.einsum("puqv,qv->pu", pu_qv, D_act, optimize=True)

    def _active_j_two(self, D_act_a: Any, D_act_b: Any | None = None):
        xp = self._xp()
        pu_qv = self._active_pu_qv_device()
        D_a = xp.asarray(D_act_a, dtype=xp.float64)
        J_a = xp.einsum("puqv,qv->pu", pu_qv, D_a, optimize=True)
        if D_act_b is None:
            return J_a, None
        D_b = xp.asarray(D_act_b, dtype=xp.float64)
        J_b = xp.einsum("puqv,qv->pu", pu_qv, D_b, optimize=True)
        return J_a, J_b

    @property
    def j_ps(self):
        if self._j_ps_cache is None:
            _ = self.j_ps_device
        return self._j_ps_cache

    @property
    def j_ps_device(self):
        if self._j_ps_cache_device is None:
            xp = self._xp()
            pu_qv = self._active_pu_qv_device()
            self._j_ps_cache_device = xp.einsum("prrs->ps", pu_qv, optimize=True)
            self._j_ps_cache = _as_numpy_f64(self._j_ps_cache_device)
        return self._j_ps_cache_device

    @property
    def pair_norm(self):
        if self._pair_norm_cache is None:
            _ = self.pair_norm_device
        return self._pair_norm_cache

    @property
    def pair_norm_device(self):
        if self._pair_norm_cache_device is None:
            xp = self._xp()
            norb = int(self.norb)
            pu_qv = self._active_pu_qv_device()
            pair_norm = xp.empty((norb * norb,), dtype=xp.float64)
            for p in range(norb):
                for q in range(norb):
                    val = xp.maximum(pu_qv[int(p), int(q), int(q), int(p)], 0.0)
                    pair_norm[int(p) * norb + int(q)] = xp.sqrt(val)
            self._pair_norm_cache_device = pair_norm
            self._pair_norm_cache = _as_numpy_f64(pair_norm)
        return self._pair_norm_cache_device

    def _ensure_hdiag_slices_device(self) -> tuple[Any, Any]:
        if self._eri_ppqq_cache_device is not None and self._eri_pqqp_cache_device is not None:
            return self._eri_ppqq_cache_device, self._eri_pqqp_cache_device

        xp = self._xp()
        norb = int(self.norb)
        pu_qv = self._active_pu_qv_device()
        eri_ppqq = xp.empty((norb, norb), dtype=xp.float64)
        for q in range(norb):
            for p in range(norb):
                eri_ppqq[int(p), int(q)] = pu_qv[int(p), int(p), int(q), int(q)]
        eri_pqqp = xp.empty((norb, norb), dtype=xp.float64)
        for p in range(norb):
            for q in range(norb):
                eri_pqqp[int(p), int(q)] = pu_qv[int(p), int(q), int(q), int(p)]
        self._eri_ppqq_cache_device = xp.ascontiguousarray(eri_ppqq)
        self._eri_pqqp_cache_device = xp.ascontiguousarray(eri_pqqp)
        return self._eri_ppqq_cache_device, self._eri_pqqp_cache_device

    def get_hdiag_slices_device(self):
        return self._ensure_hdiag_slices_device()

    def contract_w_block_device(self, w_block: Any, *, half: float = 0.5, out: Any | None = None):
        xp = self._xp()
        w_block = xp.asarray(w_block, dtype=xp.float64)
        if w_block.ndim != 2:
            raise ValueError("w_block must be 2D")
        nrows, nops = map(int, w_block.shape)
        norb = int(self.norb)
        if int(nops) != int(norb * norb):
            raise ValueError("w_block has wrong pair dimension")
        g_block = out
        if g_block is None:
            g_block = xp.empty((nrows, nops), dtype=xp.float64)
        else:
            g_block = xp.asarray(g_block, dtype=xp.float64)
            if tuple(g_block.shape) != (nrows, nops):
                raise ValueError("out has wrong shape")
        row = 0
        while row < nrows:
            if row + 1 < nrows:
                D_a = _pair_density_from_flat(w_block[row], norb, xp)
                D_b = _pair_density_from_flat(w_block[row + 1], norb, xp)
                J_a, J_b = self._active_j_two(D_a, D_b)
                g_block[row] = float(half) * xp.asarray(J_a, dtype=xp.float64).reshape(nops)
                g_block[row + 1] = float(half) * xp.asarray(J_b, dtype=xp.float64).reshape(nops)
                row += 2
            else:
                D_a = _pair_density_from_flat(w_block[row], norb, xp)
                J_a, _ = self._active_j_two(D_a, None)
                g_block[row] = float(half) * xp.asarray(J_a, dtype=xp.float64).reshape(nops)
                row += 1
        return g_block

    def contract_apply_w_block_device(
        self,
        drt: Any,
        drt_dev: Any,
        state_dev: Any,
        epq_table_t: Any,
        w_block: Any,
        *,
        k_start: int,
        y: Any,
        half: float = 0.5,
        overflow: Any | None = None,
        threads_apply: int = 32,
        add: bool = True,
        stream: Any | None = None,
        sync: bool = False,
        check_overflow: bool = False,
        use_kahan: bool = False,
        profile: dict | None = None,
    ):
        from asuka.cuda.cuda_backend import apply_g_flat_gather_epq_transpose_range_inplace_device  # noqa: PLC0415

        xp = self._xp()
        w_block = xp.asarray(w_block, dtype=xp.float64)
        if w_block.ndim != 2:
            raise ValueError("w_block must be 2D")
        nrows, nops = map(int, w_block.shape)
        norb = int(self.norb)
        if int(nops) != int(norb * norb):
            raise ValueError("w_block has wrong pair dimension")
        if nrows == 0:
            return xp.asarray(y, dtype=xp.float64)

        if stream is None and hasattr(xp, "cuda"):
            stream = xp.cuda.get_current_stream()

        y_dev = xp.asarray(y, dtype=xp.float64)
        t_contract = 0.0
        t_apply = 0.0
        row = 0
        while row < nrows:
            row_count = 2 if row + 1 < nrows else 1
            g_rows = xp.empty((row_count, nops), dtype=xp.float64)
            t0 = None
            if profile is not None:
                import time as _time  # noqa: PLC0415

                t0 = _time.perf_counter()
            if row_count == 2:
                D_a = _pair_density_from_flat(w_block[row], norb, xp)
                D_b = _pair_density_from_flat(w_block[row + 1], norb, xp)
                J_a, J_b = self._active_j_two(D_a, D_b)
                g_rows[0] = float(half) * xp.asarray(J_a, dtype=xp.float64).reshape(nops)
                g_rows[1] = float(half) * xp.asarray(J_b, dtype=xp.float64).reshape(nops)
            else:
                D_a = _pair_density_from_flat(w_block[row], norb, xp)
                J_a, _ = self._active_j_two(D_a, None)
                g_rows[0] = float(half) * xp.asarray(J_a, dtype=xp.float64).reshape(nops)
            if profile is not None and t0 is not None and stream is not None:
                stream.synchronize()
                t_contract += _time.perf_counter() - t0
                t0 = _time.perf_counter()
            apply_g_flat_gather_epq_transpose_range_inplace_device(
                drt,
                drt_dev,
                state_dev,
                epq_table_t,
                g_rows,
                k_start=int(k_start) + int(row),
                k_count=int(row_count),
                y=y_dev,
                overflow=overflow,
                threads=int(threads_apply),
                add=bool(add),
                stream=stream,
                sync=bool(sync),
                check_overflow=bool(check_overflow),
                dtype=xp.float64,
                use_kahan=bool(use_kahan),
            )
            if profile is not None and t0 is not None and stream is not None:
                stream.synchronize()
                t_apply += _time.perf_counter() - t0
            row += row_count
        if profile is not None:
            profile["offdiag_direct_cuda_s"] = profile.get("offdiag_direct_cuda_s", 0.0) + float(t_contract)
            profile["offdiag_direct_apply_s"] = profile.get("offdiag_direct_apply_s", 0.0) + float(t_apply)
            profile["offdiag_direct_fused_s"] = profile.get("offdiag_direct_fused_s", 0.0) + float(t_contract + t_apply)
        return y_dev


__all__ = ["DirectRowOracleIntegrals", "DeviceDirectMOIntegrals"]
