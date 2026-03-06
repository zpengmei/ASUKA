from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuda.active_space_thc.active_space_integrals import (
    build_device_dfmo_integrals_local_thc,
    build_device_dfmo_integrals_thc,
)
from asuka.hf.local_thc_factors import LocalTHCFactors
from asuka.hf.local_thc_jk import local_thc_JK
from asuka.hf.thc_factors import THCFactors
from asuka.hf.thc_jk import THCJKWork, thc_JK
from asuka.mcscf.newton_df import DFNewtonERIs, _as_xp_f64, _asnumpy_f64, _get_xp
from asuka.mcscf.orbital_grad import cayley_update


def _require_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("THC Newton-CASSCF support requires CuPy") from e
    return cp


def build_thc_newton_eris(
    scf_out: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
    q_block: int = 256,
    pair_p_block: int = 8,
) -> DFNewtonERIs:
    """Build Newton-CASSCF ERI intermediates from THC / local-THC factors."""

    cp = _require_cupy()

    thc = getattr(scf_out, "thc_factors", None)
    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    mo = cp.asarray(mo_coeff, dtype=cp.float64)
    if int(getattr(mo, "ndim", 0)) != 2:
        raise ValueError("mo_coeff must have shape (nao,nmo)")
    nao, nmo = map(int, mo.shape)
    ncore = int(ncore)
    ncas = int(ncas)
    nocc = int(ncore + ncas)
    if ncore < 0 or ncas <= 0 or nocc > nmo:
        raise ValueError("invalid ncore/ncas for THC Newton ERIs")

    if isinstance(thc, THCFactors):
        full = build_device_dfmo_integrals_thc(
            thc,
            mo,
            want_eri_mat=False,
            want_pair_norm=False,
            p_block=int(max(1, pair_p_block)),
        )
    else:
        full = build_device_dfmo_integrals_local_thc(
            thc,
            mo,
            want_eri_mat=False,
            p_block=int(max(1, pair_p_block)),
        )

    l_full = getattr(full, "l_full", None)
    if l_full is None:
        raise ValueError("THC Newton ERIs require l_full on the full MO space")
    l_full = cp.asarray(l_full, dtype=cp.float64)
    if int(getattr(l_full, "ndim", 0)) != 2:
        raise ValueError("THC Newton l_full must be 2D")
    naux_eff = int(l_full.shape[1])
    L = cp.ascontiguousarray(l_full.reshape(int(nmo), int(nmo), int(naux_eff)), dtype=cp.float64)

    act = slice(int(ncore), int(nocc))
    L_uv = cp.ascontiguousarray(L[act, act], dtype=cp.float64)
    L_pu = cp.ascontiguousarray(L[:, act], dtype=cp.float64)
    L_pi = cp.ascontiguousarray(L[:, : int(ncore)], dtype=cp.float64) if int(ncore) else None

    ppaa = cp.ascontiguousarray(cp.einsum("pqQ,uvQ->pquv", L, L_uv, optimize=True), dtype=cp.float64)
    papa = cp.ascontiguousarray(cp.einsum("puQ,qvQ->puqv", L_pu, L_pu, optimize=True), dtype=cp.float64)

    L_pp = cp.ascontiguousarray(L[cp.arange(int(nmo)), cp.arange(int(nmo))], dtype=cp.float64)
    if int(ncore):
        L_ii = cp.ascontiguousarray(L_pp[: int(ncore)], dtype=cp.float64)
        j_pc = cp.ascontiguousarray(L_pp @ L_ii.T, dtype=cp.float64)
        if L_pi is None:  # pragma: no cover
            raise RuntimeError("internal error: expected L_pi for ncore > 0")
        k_pc = cp.ascontiguousarray(cp.einsum("piQ,piQ->pi", L_pi, L_pi, optimize=True), dtype=cp.float64)
    else:
        j_pc = cp.zeros((int(nmo), 0), dtype=cp.float64)
        k_pc = cp.zeros((int(nmo), 0), dtype=cp.float64)

    if int(ncore):
        C_core = mo[:, : int(ncore)]
        D_core = 2.0 * (C_core @ C_core.T)
        if isinstance(thc, THCFactors):
            Jc, Kc = thc_JK(D_core, thc.X, thc.Z, work=THCJKWork(q_block=int(max(1, q_block))))
        else:
            Jc, Kc = local_thc_JK(D_core, thc, q_block=int(max(1, q_block)))
        v_ao = cp.asarray(Jc - 0.5 * Kc, dtype=cp.float64)
        vhf_c = cp.ascontiguousarray(mo.T @ v_ao @ mo, dtype=cp.float64)
    else:
        vhf_c = cp.zeros((int(nmo), int(nmo)), dtype=cp.float64)

    return DFNewtonERIs(
        ppaa=ppaa,
        papa=papa,
        vhf_c=vhf_c,
        j_pc=j_pc,
        k_pc=k_pc,
        L_pu=L_pu,
        L_pi=L_pi,
        L_uv=L_uv,
        L_pq=L,
    )


@dataclass
class THCNewtonCASSCFAdapter:
    """Minimal THC-backed CASSCF-like adapter for the internal Newton operator."""

    scf_out: Any
    hcore_ao: Any
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    mo_coeff: Any
    fcisolver: Any

    weights: list[float] | None = None
    frozen: Any | None = None
    internal_rotation: bool = False
    extrasym: Any | None = None
    q_block: int = 256
    pair_p_block: int = 8

    def _get_2e_probe(self) -> Any:
        thc = getattr(self.scf_out, "thc_factors", None)
        if isinstance(thc, THCFactors):
            return getattr(thc, "X", self.hcore_ao)
        if isinstance(thc, LocalTHCFactors) and len(getattr(thc, "blocks", ())) > 0:
            return getattr(thc.blocks[0], "X", self.hcore_ao)
        return self.hcore_ao

    def get_hcore(self) -> Any:
        xp, _is_gpu = _get_xp(self._get_2e_probe(), self.hcore_ao)
        return _as_xp_f64(xp, self.hcore_ao)

    def ao2mo(self, mo_coeff: Any) -> DFNewtonERIs:
        return build_thc_newton_eris(
            self.scf_out,
            mo_coeff,
            ncore=int(self.ncore),
            ncas=int(self.ncas),
            q_block=int(self.q_block),
            pair_p_block=int(self.pair_p_block),
        )

    def uniq_var_indices(self, nmo: int, ncore: int, ncas: int, frozen: Any | None) -> np.ndarray:
        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        mask = np.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True
        if bool(self.internal_rotation):
            mask[ncore:nocc, ncore:nocc][np.tril_indices(ncas, -1)] = True
        if self.extrasym is not None:
            extrasym = np.asarray(self.extrasym)
            extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
            mask = mask & extrasym_allowed
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[: int(frozen)] = False
                mask[:, : int(frozen)] = False
            else:
                frozen_idx = np.asarray(frozen, dtype=np.int32).ravel()
                mask[frozen_idx] = False
                mask[:, frozen_idx] = False
        return mask

    def pack_uniq_var(self, mat: Any) -> np.ndarray:
        xp, _on_gpu = _get_xp(mat)
        mat = xp.asarray(mat, dtype=xp.float64)
        nmo = int(self.mo_coeff.shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        return xp.asarray(mat[idx], dtype=xp.float64)

    def unpack_uniq_var(self, v: Any) -> np.ndarray:
        xp, _on_gpu = _get_xp(v)
        v = xp.asarray(v, dtype=xp.float64).ravel()
        nmo = int(self.mo_coeff.shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        mat = xp.zeros((nmo, nmo), dtype=xp.float64)
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx: Any, u0: Any = 1) -> np.ndarray:
        dr = self.unpack_uniq_var(dx)
        u = cayley_update(np, dr)
        return np.dot(u0, np.asarray(u, dtype=np.float64))

    def update_jk_in_ah(
        self,
        mo: Any,
        r: Any,
        casdm1: Any,
        eris: Any | None = None,
        *,
        return_gpu: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        cp = _require_cupy()
        ncore = int(self.ncore)
        ncas = int(self.ncas)
        nocc = ncore + ncas

        mo = cp.asarray(mo, dtype=cp.float64)
        r = cp.asarray(r, dtype=cp.float64)
        casdm1 = cp.asarray(casdm1, dtype=cp.float64)

        if mo.ndim != 2:
            raise ValueError("mo must be 2D (nao,nmo)")
        nao, nmo = map(int, mo.shape)
        if nocc > nmo:
            raise ValueError("ncore+ncas exceeds nmo")
        if r.shape != (nmo, nmo):
            raise ValueError("r must be (nmo,nmo)")
        if casdm1.shape != (ncas, ncas):
            raise ValueError("casdm1 must be (ncas,ncas)")

        if eris is None:
            eris = self.ao2mo(mo)

        L_pq = getattr(eris, "L_pq", None)
        if L_pq is None:
            raise ValueError("THCNewtonCASSCFAdapter.update_jk_in_ah requires eris.L_pq")
        L_t = cp.asarray(L_pq, dtype=cp.float64).transpose(0, 2, 1)  # (nmo,naux,nmo)
        naux = int(L_t.shape[1])

        dm3 = mo[:, :ncore] @ r[:ncore, ncore:] @ mo[:, ncore:].T
        dm3 = dm3 + dm3.T

        dm4 = mo[:, ncore:nocc] @ casdm1 @ r[ncore:nocc] @ mo.T
        dm4 = dm4 + dm4.T
        dm_total = dm3 * 2.0 + dm4

        x_core_rest = r[:ncore, ncore:]
        x_act = r[ncore:nocc]
        casdm1_jk = casdm1

        L_t_act_flat = L_t[ncore:nocc].reshape(ncas * naux, nmo)
        L_t_core_flat = L_t[:ncore].reshape(ncore * naux, nmo)

        LDM0 = (L_t_act_flat @ dm3).reshape(ncas, naux * nmo)
        LDM1 = (L_t_core_flat @ dm_total).reshape(ncore, naux * nmo)

        qblk = int(max(1, min(int(naux), int(self.q_block))))
        K0_act = cp.zeros((ncas, nmo), dtype=cp.float64)
        K1_core = cp.zeros((ncore, nmo), dtype=cp.float64)
        for q0 in range(0, int(naux), int(qblk)):
            q1 = min(int(naux), int(q0) + int(qblk))
            qb = int(q1 - q0)
            if qb <= 0:
                continue
            off0 = int(q0 * nmo)
            off1 = int(q1 * nmo)
            L_blk_t = L_t[:, q0:q1, :].reshape(nmo, qb * nmo).T
            K0_act += LDM0[:, off0:off1] @ L_blk_t
            K1_core += LDM1[:, off0:off1] @ L_blk_t

        rho0 = 2.0 * cp.einsum(
            "iQa,ia->Q",
            L_t[:ncore, :, ncore:],
            x_core_rest,
            optimize=True,
        )
        J0_act = cp.einsum("pQq,Q->pq", L_t[ncore:nocc], rho0, optimize=True)
        dm4_act = casdm1_jk @ x_act
        rho_dm4 = 2.0 * cp.einsum("pQq,pq->Q", L_t[ncore:nocc], dm4_act, optimize=True)
        rho1 = 2.0 * rho0 + rho_dm4
        J1_core = cp.einsum("pQq,Q->pq", L_t[:ncore], rho1, optimize=True)

        va = casdm1 @ (J0_act * 2.0 - K0_act)
        vc = (J1_core * 2.0 - K1_core)[:, ncore:]

        va_cont = cp.ascontiguousarray(cp.asarray(va, dtype=cp.float64))
        vc_cont = cp.ascontiguousarray(cp.asarray(vc, dtype=cp.float64))

        if return_gpu:
            return va_cont, vc_cont
        return _asnumpy_f64(va_cont), _asnumpy_f64(vc_cont)


__all__ = ["THCNewtonCASSCFAdapter", "build_thc_newton_eris"]
