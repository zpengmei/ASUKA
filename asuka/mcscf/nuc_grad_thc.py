from __future__ import annotations

"""Analytic nuclear gradients for (SA-)CASSCF with THC/LS-THC 2e integrals.

Scope (initial implementation)
------------------------------
- State-averaged CASSCF gradient only (i.e. gradient of the SA objective).
- Global THC factors (`asuka.hf.thc_factors.THCFactors`) and local-THC factors
  (`asuka.hf.local_thc_factors.LocalTHCFactors`) are supported.
- Analytic THC factor gradients currently require:
  - `solve_method in {'inv_metric','fit_metric_gram'}` (i.e. Y from an aux-metric
    triangular solve or a Gram-matrix solve)
  - atom-centered grids with `meta['point_atom']` present (grid_kind in
    {'becke','rdvr'}).

Notes
-----
This module parallels :mod:`asuka.mcscf.nuc_grad_df` but replaces DF 2e
derivative contractions with a THC VJP through the stored THC factors.

Unlike the DF path, the THC factor VJP already differentiates the AO
collocation values `X` with respect to the nuclear coordinates. That AO-basis
response subsumes the usual overlap-Pulay contribution, so THC-CASSCF must not
add a separate `-Tr[W dS]` term on top.

Per-root (state-specific) SA-CASSCF gradients (Z-vector / response) are not
implemented yet.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import time

from asuka.solver import GUGAFCISolver
from asuka.utils.einsum_cache import cached_einsum

from .state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from .nuc_grad_df import DFNucGradResult, _mol_coords_charges_bohr


def _require_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("THC-CASSCF analytic gradients require CuPy") from e
    return cp


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(a, cp.ndarray):
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class THCNucGradComponents:
    nuc: np.ndarray
    hcore: np.ndarray
    pulay: np.ndarray
    thc_2e: np.ndarray
    metric: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return np.asarray(self.nuc + self.hcore + self.pulay + self.thc_2e + self.metric, dtype=np.float64)


def _symmetrize(cp, A):
    return 0.5 * (A + A.T)


def _thc_energy_adjoint_jk_bilinear(
    D_right: Any,
    D_left: Any,
    X: Any,
    Z: Any,
    Y: Any,
    *,
    cJ: float,
    cK: float,
    q_block: int = 256,
) -> tuple[Any, Any]:
    """Return (bar_X, bar_Y) for E = cJ*Tr(D_left*J[D_right]) + cK*Tr(D_left*K[D_right]).

    Shapes
    - D_right, D_left: (nao,nao), symmetric
    - X: (npt,nao)
    - Z: (npt,npt) (symmetric)
    - Y: (npt,naux) with Z = Y @ Y.T
    """

    cp = _require_cupy()
    D = cp.asarray(D_right, dtype=cp.float64)
    B = cp.asarray(D_left, dtype=cp.float64)
    X = cp.asarray(X, dtype=cp.float64)
    Z = cp.asarray(Z, dtype=cp.float64)
    Y = cp.asarray(Y, dtype=cp.float64)

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D_right must be (nao,nao)")
    if B.shape != D.shape:
        raise ValueError("D_left shape mismatch with D_right")
    nao = int(D.shape[0])

    if X.ndim != 2 or int(X.shape[1]) != int(nao):
        raise ValueError("X must have shape (npt,nao)")
    npt = int(X.shape[0])
    if Z.shape != (npt, npt):
        raise ValueError("Z must have shape (npt,npt)")
    if Y.ndim != 2 or int(Y.shape[0]) != int(npt):
        raise ValueError("Y must have shape (npt,naux)")

    q_block = max(1, min(int(q_block), int(npt)))
    cJ_f = float(cJ)
    cK_f = float(cK)

    # A_D[P,nu] = sum_mu X[P,mu] D[mu,nu]
    A_D = X @ D  # (npt,nao)
    # A_B[P,nu] = sum_mu X[P,mu] B[mu,nu]
    A_B = X @ B  # (npt,nao)

    # m[P] = (X D X^T)[P,P]
    m = cp.sum(A_D * X, axis=1)
    # t[P] = (X B X^T)[P,P]
    t = cp.sum(A_B * X, axis=1)

    # g = Z m
    g = Z @ m  # (npt,)

    # ---- Coulomb-like contribution: E_J = cJ * t^T (Z m) ----
    # bar_t = cJ * g
    bar_t = cJ_f * g
    # bar_g = cJ * t
    bar_g = cJ_f * t

    # bar_X from t and m paths.
    # dt/dX: 2 * diag(bar_t) * (X B)
    bar_X = 2.0 * (bar_t[:, None] * A_B)
    # dg/dm: bar_m = Z^T bar_g (Z symmetric)
    bar_m = Z @ bar_g
    # dm/dX: 2 * diag(bar_m) * (X D)
    bar_X += 2.0 * (bar_m[:, None] * A_D)

    # bar_Y via Z = Y Y^T and bar_Z = outer(bar_g, m)
    # bar_Y = (bar_Z + bar_Z^T) @ Y
    if cJ_f != 0.0:
        v_m = m @ Y  # (naux,)
        v_t = t @ Y  # (naux,)
        bar_Y = cJ_f * (t[:, None] * v_m[None, :] + m[:, None] * v_t[None, :])
    else:
        bar_Y = cp.zeros_like(Y)

    # ---- Exchange-like contribution: E_K = cK * sum_{P,Q} Z[P,Q] M[P,Q] N[P,Q]
    # where M = X D X^T, N = X B X^T (both symmetric).
    if cK_f != 0.0:
        for q0 in range(0, npt, q_block):
            q1 = min(npt, int(q0) + int(q_block))
            nb = int(q1 - q0)
            if nb <= 0:
                continue

            Xq = X[int(q0) : int(q1), :]  # (nb,nao)
            Yq = Y[int(q0) : int(q1), :]  # (nb,naux)

            # M[:,Q] and N[:,Q] columns.
            Mq = A_D @ Xq.T  # (npt,nb)
            Nq = A_B @ Xq.T  # (npt,nb)

            Zblk = Z[:, int(q0) : int(q1)]  # (npt,nb)
            T_N = Zblk * Nq  # (npt,nb)
            T_M = Zblk * Mq  # (npt,nb)

            # bar_X += 2*cK * ( (Z⊙N) @ (X D) + (Z⊙M) @ (X B) )
            bar_X += 2.0 * cK_f * (T_N @ A_D[int(q0) : int(q1), :] + T_M @ A_B[int(q0) : int(q1), :])

            # bar_Y += 2*cK * (M⊙N) @ Y
            bar_Y += 2.0 * cK_f * ((Mq * Nq) @ Yq)

            del Xq, Yq, Mq, Nq, Zblk, T_N, T_M

    return cp.ascontiguousarray(bar_X), cp.ascontiguousarray(bar_Y)


def _dm2_sym_flat(cp, dm2_act: Any, *, ncas: int) -> Any:
    dm2 = cp.asarray(dm2_act, dtype=cp.float64)
    if dm2.shape == (ncas, ncas, ncas, ncas):
        dm2 = dm2.reshape(ncas * ncas, ncas * ncas)
    elif dm2.shape != (ncas * ncas, ncas * ncas):
        raise ValueError("dm2_act must have shape (ncas,ncas,ncas,ncas) or (ncas^2,ncas^2)")
    return 0.5 * (dm2 + dm2.T)


def _thc_energy_adjoint_active_global(
    X: Any,
    Y: Any,
    C_act: Any,
    dm2_flat_sym: Any,
    *,
    pair_p_block: int = 8,
) -> tuple[Any, Any]:
    """Return (bar_X, bar_Y) for E_aa = 0.5 * sum_{uvwx} dm2_uvwx (uv|wx) via global THC factors."""

    cp = _require_cupy()
    X = cp.asarray(X, dtype=cp.float64)
    Y = cp.asarray(Y, dtype=cp.float64)
    C_act = cp.asarray(C_act, dtype=cp.float64)
    dm2 = cp.asarray(dm2_flat_sym, dtype=cp.float64)

    if X.ndim != 2:
        raise ValueError("X must be 2D (npt,nao)")
    npt, nao = map(int, X.shape)
    if C_act.ndim != 2 or int(C_act.shape[0]) != int(nao):
        raise ValueError("C_act must have shape (nao,ncas)")
    ncas = int(C_act.shape[1])
    if ncas <= 0:
        raise ValueError("empty active space")
    if dm2.shape != (ncas * ncas, ncas * ncas):
        raise ValueError("dm2_flat_sym shape mismatch")

    naux = int(Y.shape[1])
    if Y.shape[0] != npt:
        raise ValueError("X/Y npt mismatch")

    # X_act[P,u] = sum_mu X[P,mu] C_act[mu,u]
    X_act = cp.ascontiguousarray(X @ C_act)  # (npt,ncas)

    # L2[pq,L] = sum_P (X_act[P,p] X_act[P,q]) Y[P,L] (ordered pairs)
    pair_p_block = int(pair_p_block)
    if pair_p_block <= 0:
        raise ValueError("pair_p_block must be > 0")
    pair_p_block = min(int(pair_p_block), int(ncas))

    L2 = cp.empty((int(ncas) * int(ncas), int(naux)), dtype=cp.float64)
    for p0 in range(0, int(ncas), int(pair_p_block)):
        p1 = min(int(ncas), int(p0) + int(pair_p_block))
        pb = int(p1 - p0)
        if pb <= 0:
            continue
        U = X_act[:, int(p0) : int(p1)]  # (npt,pb)
        pairs = U[:, :, None] * X_act[:, None, :]  # (npt,pb,ncas)
        pairs2 = pairs.reshape(int(npt), int(pb) * int(ncas))  # (npt,pb*ncas)
        block_l = pairs2.T @ Y  # (pb*ncas,naux)
        L2[int(p0) * int(ncas) : int(p1) * int(ncas), :] = block_l
        pairs = None
        pairs2 = None
        block_l = None

    # bar_L2 = dm2_sym @ L2
    bar_L2 = dm2 @ L2  # (ncas^2,naux)

    # bar_Y = pairs2 @ bar_L2 (accumulate in p-blocks to avoid a full pairs2)
    bar_Y = cp.zeros_like(Y)
    for p0 in range(0, int(ncas), int(pair_p_block)):
        p1 = min(int(ncas), int(p0) + int(pair_p_block))
        pb = int(p1 - p0)
        if pb <= 0:
            continue
        U = X_act[:, int(p0) : int(p1)]
        pairs = U[:, :, None] * X_act[:, None, :]
        pairs2 = pairs.reshape(int(npt), int(pb) * int(ncas))
        bar_Y += pairs2 @ bar_L2[int(p0) * int(ncas) : int(p1) * int(ncas), :]
        pairs = None
        pairs2 = None

    # bar_pairs2[P,pq] = sum_L Y[P,L] * bar_L2[pq,L]
    bar_pairs2 = Y @ bar_L2.T  # (npt,ncas^2)
    bar_pairs = bar_pairs2.reshape(int(npt), int(ncas), int(ncas))

    # bar_X_act[P,p] = sum_q (bar_pairs[P,p,q] + bar_pairs[P,q,p]) * X_act[P,q]
    S = bar_pairs + bar_pairs.transpose(0, 2, 1)
    bar_X_act = cached_einsum("Ppq,Pq->Pp", S, X_act, xp=cp)

    bar_X = cp.ascontiguousarray(bar_X_act @ C_act.T)  # (npt,nao)
    return bar_X, cp.ascontiguousarray(bar_Y)


def _thc_energy_adjoint_active_local_block(
    X_blk: Any,
    Z_blk: Any,
    Y_blk: Any,
    *,
    ao_idx_global: np.ndarray,
    n_early: int,
    n_primary: int,
    C_act: Any,
    dm2_flat_sym: Any,
    pair_p_block: int = 8,
    q_block: int = 256,
) -> tuple[Any, Any]:
    """Return (bar_X_blk, bar_Y_blk) for the local-THC active-active energy term.

    This differentiates the repaired local active-space energy that applies the
    same local AO ERI operator used by the LocalTHC CASCI/CASSCF energy path.
    """

    cp = _require_cupy()
    X = cp.asarray(X_blk, dtype=cp.float64)
    Z = cp.asarray(Z_blk, dtype=cp.float64)
    Y = cp.asarray(Y_blk, dtype=cp.float64)
    C_act = cp.asarray(C_act, dtype=cp.float64)
    dm2 = cp.asarray(dm2_flat_sym, dtype=cp.float64)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X_blk/Y_blk must be 2D")
    npt, nloc = map(int, X.shape)
    if Z.shape != (int(npt), int(npt)):
        raise ValueError("Z_blk must have shape (npt,npt)")
    if int(Y.shape[0]) != int(npt):
        raise ValueError("X_blk/Y_blk npt mismatch")

    idx_np = np.asarray(ao_idx_global, dtype=np.int32).ravel()
    if int(idx_np.size) != int(nloc):
        raise ValueError("ao_idx_global size mismatch with X_blk columns")

    if C_act.ndim != 2:
        raise ValueError("C_act must be 2D (nao,ncas)")
    nao, ncas = map(int, C_act.shape)
    if dm2.shape != (ncas * ncas, ncas * ncas):
        raise ValueError("dm2_flat_sym shape mismatch")

    n_early_i = int(n_early)
    n_primary_i = int(n_primary)
    if n_early_i < 0 or n_early_i > nloc:
        raise ValueError("invalid n_early")
    if n_primary_i < 0 or (n_early_i + n_primary_i) > nloc:
        raise ValueError("invalid n_primary")
    if int(nloc) == 0:
        return cp.zeros_like(X), cp.zeros_like(Y)

    idx_xp = cp.asarray(idx_np, dtype=cp.int32)
    C_loc = C_act[idx_xp, :]

    pair_p_block = int(pair_p_block)
    if pair_p_block <= 0:
        raise ValueError("pair_p_block must be > 0")
    pair_p_block = min(int(pair_p_block), int(ncas))
    q_block = max(1, min(int(q_block), int(npt)))

    bar_X = cp.zeros_like(X)
    bar_Y = cp.zeros_like(Y)
    tail = int(n_early_i + n_primary_i)

    for w0 in range(0, int(ncas), int(pair_p_block)):
        w1 = min(int(ncas), int(w0) + int(pair_p_block))
        for w in range(int(w0), int(w1)):
            cw = C_loc[:, int(w)]
            for x in range(int(ncas)):
                cx = C_loc[:, int(x)]
                D_sub = 0.5 * (
                    cw[:, None] * cx[None, :] + cx[:, None] * cw[None, :]
                )

                coeff = 0.5 * dm2[:, int(w) * int(ncas) + int(x)].reshape(int(ncas), int(ncas))
                coeff = 0.5 * (coeff + coeff.T)
                B_sub = C_loc @ coeff @ C_loc.T
                B_eff = B_sub.copy()
                if n_early_i > 0:
                    B_eff[:n_early_i, :] = 0.0
                    B_eff[:, :n_early_i] = 0.0
                if tail < int(nloc):
                    B_eff[int(tail) :, int(tail) :] = 0.0

                bar_X_pair, bar_Y_pair = _thc_energy_adjoint_jk_bilinear(
                    D_sub,
                    B_eff,
                    X,
                    Z,
                    Y,
                    cJ=1.0,
                    cK=0.0,
                    q_block=int(q_block),
                )
                bar_X += bar_X_pair
                bar_Y += bar_Y_pair

    return cp.ascontiguousarray(bar_X), cp.ascontiguousarray(bar_Y)


def _build_gfock_casscf_thc(
    scf_out: Any,
    *,
    C: Any,
    ncore: int,
    ncas: int,
    dm1_act: Any,
    dm2_act: Any,
    q_block: int = 256,
    pair_p_block: int = 8,
    profile: dict | None = None,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return (gfock_mo, D_core_ao, D_act_ao, D_tot_ao, C_act) for THC-CASSCF."""

    cp = _require_cupy()
    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0:
        raise ValueError("invalid ncore/ncas")

    thc = getattr(scf_out, "thc_factors", None)
    if thc is None:
        raise ValueError("scf_out.thc_factors is missing")

    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415
    from asuka.hf.local_thc_factors import LocalTHCFactors  # noqa: PLC0415
    from asuka.hf import df_scf as _df_scf  # noqa: PLC0415

    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    int1e = getattr(scf_out, "int1e", None)
    if int1e is None:
        raise ValueError("scf_out.int1e is missing")
    hcore = getattr(int1e, "hcore", None)
    if hcore is None:
        raise ValueError("scf_out.int1e.hcore is missing")

    if isinstance(thc, THCFactors):
        xp, is_gpu = _df_scf._get_xp(C, thc.X, thc.Z, thc.Y)  # noqa: SLF001
    else:
        if len(thc.blocks) == 0:
            raise ValueError("LocalTHCFactors.blocks is empty")
        xp, is_gpu = _df_scf._get_xp(C, thc.blocks[0].X)  # noqa: SLF001
    if xp is not cp or not bool(is_gpu):
        raise RuntimeError("THC-CASSCF gradients currently require GPU (CuPy) arrays")

    C = cp.asarray(C, dtype=cp.float64)
    h_ao = cp.asarray(hcore, dtype=cp.float64)

    nao, nmo = map(int, C.shape)
    nocc = int(ncore + ncas)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    dm1 = cp.asarray(dm1_act, dtype=cp.float64)
    if dm1.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")
    dm1 = _symmetrize(cp, dm1)

    dm2_flat = _dm2_sym_flat(cp, dm2_act, ncas=int(ncas))

    # Densities
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    if ncore:
        D_core_ao = 2.0 * (C_core @ C_core.T)
    else:
        D_core_ao = cp.zeros((nao, nao), dtype=cp.float64)
    D_act_ao = C_act @ dm1 @ C_act.T
    D_tot_ao = D_core_ao + D_act_ao

    # Mean-field potentials from core and active 1-RDM.
    if isinstance(thc, THCFactors):
        from asuka.hf.thc_jk import THCJKWork, thc_JK  # noqa: PLC0415

        Jc, Kc = thc_JK(D_core_ao, thc.X, thc.Z, work=THCJKWork(q_block=int(q_block)))
        Ja, Ka = thc_JK(D_act_ao, thc.X, thc.Z, work=THCJKWork(q_block=int(q_block)))
    else:
        from asuka.hf.local_thc_jk import local_thc_JK  # noqa: PLC0415

        Jc, Kc = local_thc_JK(D_core_ao, thc, q_block=int(q_block))
        Ja, Ka = local_thc_JK(D_act_ao, thc, q_block=int(q_block))

    vhf_c_ao = Jc - 0.5 * Kc
    vhf_a_ao = Ja - 0.5 * Ka
    vhf_ca_ao = vhf_c_ao + vhf_a_ao

    # Transform to MO basis
    h_mo = C.T @ h_ao @ C
    vhf_c_mo = C.T @ vhf_c_ao @ C
    vhf_ca_mo = C.T @ vhf_ca_ao @ C

    # Active-active 2-RDM term contraction (same as orbital_gradient_thc).
    pair_p_block = int(pair_p_block)
    if pair_p_block <= 0:
        raise ValueError("pair_p_block must be > 0")
    pair_p_block = min(int(pair_p_block), int(ncas))

    if isinstance(thc, THCFactors):
        X_thc = cp.asarray(thc.X, dtype=cp.float64)
        Y_thc = cp.asarray(thc.Y, dtype=cp.float64)
        X_mo = X_thc @ C  # (npt,nmo)
        X_act_thc = X_mo[:, ncore:nocc]  # (npt,ncas)
        npt = int(X_mo.shape[0])
        naux = int(Y_thc.shape[1])

        L2 = cp.empty((int(ncas) * int(ncas), int(naux)), dtype=cp.float64)
        for p0 in range(0, int(ncas), int(pair_p_block)):
            p1 = min(int(ncas), int(p0) + int(pair_p_block))
            pb = int(p1 - p0)
            if pb <= 0:
                continue
            U = X_act_thc[:, int(p0) : int(p1)]
            pairs = U[:, :, None] * X_act_thc[:, None, :]
            pairs2 = pairs.reshape(int(npt), int(pb) * int(ncas))
            block_l = pairs2.T @ Y_thc
            L2[int(p0) * int(ncas) : int(p1) * int(ncas), :] = block_l
            pairs = None
            pairs2 = None
            block_l = None

        T_flat = L2.T @ dm2_flat  # (naux,ncas^2)
        S_flat = Y_thc @ T_flat  # (npt,ncas^2)
        S = S_flat.reshape(int(npt), int(ncas), int(ncas))
        t_pv = cached_einsum("Pu,Puv->Pv", X_act_thc, S, xp=cp)
        g_dm2 = X_mo.T @ t_pv  # (nmo,ncas)
    else:
        from asuka.hf.local_thc_jk import local_thc_eri_apply_batched  # noqa: PLC0415

        dm2_wxuv = cp.asarray(dm2_act, dtype=cp.float64)
        if dm2_wxuv.shape != (ncas, ncas, ncas, ncas):
            dm2_wxuv = dm2_wxuv.reshape(ncas, ncas, ncas, ncas)

        g_dm2 = cp.zeros((nmo, ncas), dtype=cp.float64)
        for w0 in range(0, int(ncas), int(pair_p_block)):
            w1 = min(int(ncas), int(w0) + int(pair_p_block))
            wb = int(w1 - w0)
            if wb <= 0:
                continue

            nbatch = int(wb) * int(ncas)
            D_batch = cp.empty((nbatch, int(nao), int(nao)), dtype=cp.float64)
            dm2_batch = cp.empty((nbatch, int(ncas), int(ncas)), dtype=cp.float64)

            ib = 0
            for w in range(int(w0), int(w1)):
                cw = C_act[:, int(w)]
                for x in range(int(ncas)):
                    cx = C_act[:, int(x)]
                    D_batch[int(ib)] = 0.5 * (
                        cw[:, None] * cx[None, :] + cx[:, None] * cw[None, :]
                    )
                    dm2_batch[int(ib)] = dm2_wxuv[int(w), int(x)]
                    ib += 1

            V_batch = local_thc_eri_apply_batched(D_batch, thc, symmetrize=True)
            pu_batch = cached_einsum("mp,bmn,nu->bpu", C, V_batch, C_act, xp=cp)
            g_dm2 += cached_einsum("bpu,buv->pv", pu_batch, dm2_batch, xp=cp)

            D_batch = None
            dm2_batch = None
            V_batch = None
            pu_batch = None

    # Generalized Fock matrix in MO basis.
    gfock = cp.zeros((nmo, nmo), dtype=cp.float64)
    if ncore:
        gfock[:, :ncore] = 2.0 * (h_mo + vhf_ca_mo)[:, :ncore]
    gfock[:, ncore:nocc] = (h_mo + vhf_c_mo)[:, ncore:nocc] @ dm1 + g_dm2

    if profile is not None:
        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)

    return gfock, D_core_ao, D_act_ao, D_tot_ao, C_act


def _thc_factor_vjp_atomgrad_inv_metric(
    *,
    mol: Any,
    ao_basis_cart: Any,
    aux_basis_cart: Any,
    sph_map: Any | None,
    is_spherical: bool,
    pts: Any,
    w: Any,
    point_atom: Any,
    becke_n: int,
    X: Any,
    Y: Any,
    L_metric: Any,
    bar_X: Any,
    bar_Y: Any,
    df_threads: int = 0,
    threads: int = 256,
) -> tuple[Any, Any]:
    """Return (grad_thc_dev, grad_metric_dev) on device for inv_metric THC factors."""

    cp = _require_cupy()

    from asuka.integrals.df_adjoint import chol_lower_adjoint  # noqa: PLC0415
    from asuka.orbitals.eval_basis_device import (  # noqa: PLC0415
        becke_weight_vjp_atomgrad_device,
        contract_aos_cart_value_grad_vjp_atomgrad_device,
        eval_aos_cart_value_on_points_device,
    )
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415
    from asuka.hf.nuc_grad_thc import _metric_2c2e_deriv_aux_atomgrad_cuda  # noqa: PLC0415

    coords = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    if natm <= 0:
        return cp.zeros((0, 3), dtype=cp.float64), cp.zeros((0, 3), dtype=cp.float64)

    pts = cp.asarray(pts, dtype=cp.float64)
    w = cp.asarray(w, dtype=cp.float64).ravel()
    p_atom = cp.asarray(point_atom, dtype=cp.int32).ravel()

    X = cp.asarray(X, dtype=cp.float64)
    Y = cp.asarray(Y, dtype=cp.float64)
    L = cp.asarray(L_metric, dtype=cp.float64)
    bar_X = cp.asarray(bar_X, dtype=cp.float64)
    bar_Y = cp.asarray(bar_Y, dtype=cp.float64)

    if p_atom.shape != (int(w.shape[0]),):
        raise ValueError("point_atom shape mismatch with grid size")

    # ---- Backprop inv_metric: Y^T = L^{-1} X_aux^T ----
    bar_Xw_T = bar_Y.T  # (naux,npt)
    try:
        import cupyx.scipy.linalg as cpx_linalg  # noqa: PLC0415

        bar_S = cpx_linalg.solve_triangular(L, bar_Xw_T, lower=True, trans="T")
    except Exception:
        bar_S = cp.linalg.solve(L.T, bar_Xw_T)

    # bar_L = -tril( bar_S @ Y )
    bar_L = -(bar_S @ Y)
    bar_L = cp.tril(bar_L)

    bar_V = chol_lower_adjoint(L, bar_L)
    bar_X_aux_p = bar_S.T  # (npt,naux)

    # ---- Accumulate atom gradients from collocation + Becke weights ----
    w_quart = cp.sqrt(cp.sqrt(w))
    w_sqrt = cp.sqrt(w)

    # bar_w from AO collocation: (1/(4w)) * sum_mu bar_X*X
    bar_w = (0.25 / w) * cp.sum(bar_X * X, axis=1)

    # bar_w from aux collocation (evaluate aux basis values).
    aux_cart = eval_aos_cart_value_on_points_device(aux_basis_cart, pts, threads=int(threads), sync=True)
    X_aux_p_val = aux_cart * w_sqrt[:, None]
    bar_w += (0.5 / w) * cp.sum(bar_X_aux_p * X_aux_p_val, axis=1)
    del aux_cart, X_aux_p_val

    grad_thc = cp.zeros((natm, 3), dtype=cp.float64)

    # AO collocation VJP needs cart basis.
    if bool(is_spherical):
        if sph_map is None:
            raise RuntimeError("expected sph_map for mol.cart=False")
        T_c2s = getattr(sph_map, "T_c2s", None)
        if T_c2s is None:
            T_c2s = sph_map[0]
        T_dev = cp.asarray(np.asarray(T_c2s, dtype=np.float64), dtype=cp.float64)
        bar_X_cart = bar_X @ T_dev.T
    else:
        bar_X_cart = bar_X

    shell_atom_cart = shell_to_atom_map(ao_basis_cart, atom_coords_bohr=coords)
    grad_thc = contract_aos_cart_value_grad_vjp_atomgrad_device(
        ao_basis_cart,
        pts,
        point_atom=p_atom,
        w_pow=w_quart,
        bar_ao=bar_X_cart,
        shell_atom=cp.asarray(shell_atom_cart, dtype=cp.int32),
        natm=natm,
        out=grad_thc,
        threads=int(threads),
        sync=False,
    )

    # Aux collocation contributions.
    shell_atom_aux = shell_to_atom_map(aux_basis_cart, atom_coords_bohr=coords)
    grad_thc = contract_aos_cart_value_grad_vjp_atomgrad_device(
        aux_basis_cart,
        pts,
        point_atom=p_atom,
        w_pow=w_sqrt,
        bar_ao=bar_X_aux_p,
        shell_atom=cp.asarray(shell_atom_aux, dtype=cp.int32),
        natm=natm,
        out=grad_thc,
        threads=int(threads),
        sync=False,
    )

    # Becke partition weight derivative contributions.
    atom_coords_dev = cp.ascontiguousarray(cp.asarray(coords, dtype=cp.float64))
    grad_thc = becke_weight_vjp_atomgrad_device(
        pts,
        w,
        bar_w=bar_w,
        point_atom=p_atom,
        atom_coords=atom_coords_dev,
        becke_n=int(becke_n),
        out=grad_thc,
        threads=int(threads),
        sync=False,
    )

    # Metric derivative contraction (cuERI CUDA).
    grad_metric = _metric_2c2e_deriv_aux_atomgrad_cuda(
        aux_basis_cart,
        atom_coords_bohr=coords,
        bar_V=bar_V,
        df_threads=int(df_threads),
    )

    return grad_thc, grad_metric


def _thc_factor_vjp_atomgrad_fit_metric_gram(
    *,
    mol: Any,
    ao_basis_cart: Any,
    aux_basis_cart: Any,
    sph_map: Any | None,
    is_spherical: bool,
    pts: Any,
    w: Any,
    point_atom: Any,
    becke_n: int,
    X: Any,
    Y: Any,
    L_metric: Any,
    bar_X: Any,
    bar_Y: Any,
    solve_rcond: float = 1e-12,
    df_threads: int = 0,
    threads: int = 256,
) -> tuple[Any, Any]:
    """Return (grad_thc_dev, grad_metric_dev) for fit_metric_gram THC factors.

    Forward (global) factor build is:
      X_aux_p = w^(1/2) * chi(r)
      Gm = X_aux_p^T X_aux_p (+ small ridge)
      G = solve(Gm, L) where L = chol(V)
      Y = X_aux_p @ G
      Z = Y @ Y^T
    """

    cp = _require_cupy()

    from asuka.integrals.df_adjoint import chol_lower_adjoint  # noqa: PLC0415
    from asuka.orbitals.eval_basis_device import (  # noqa: PLC0415
        becke_weight_vjp_atomgrad_device,
        contract_aos_cart_value_grad_vjp_atomgrad_device,
        eval_aos_cart_value_on_points_device,
    )
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415
    from asuka.hf.nuc_grad_thc import _metric_2c2e_deriv_aux_atomgrad_cuda  # noqa: PLC0415

    coords = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    if natm <= 0:
        z = cp.zeros((0, 3), dtype=cp.float64)
        return z, z

    pts = cp.asarray(pts, dtype=cp.float64)
    w = cp.asarray(w, dtype=cp.float64).ravel()
    p_atom = cp.asarray(point_atom, dtype=cp.int32).ravel()

    X = cp.asarray(X, dtype=cp.float64)
    Y = cp.asarray(Y, dtype=cp.float64)
    L = cp.asarray(L_metric, dtype=cp.float64)
    bar_X = cp.asarray(bar_X, dtype=cp.float64)
    bar_Y = cp.asarray(bar_Y, dtype=cp.float64)

    if p_atom.shape != (int(w.shape[0]),):
        raise ValueError("point_atom shape mismatch with grid size")

    w_quart = cp.sqrt(cp.sqrt(w))
    w_sqrt = cp.sqrt(w)

    # Aux collocation (cart) for Gram solve.
    aux_cart = eval_aos_cart_value_on_points_device(aux_basis_cart, pts, threads=int(threads), sync=True)
    X_aux_p = cp.ascontiguousarray(aux_cart * w_sqrt[:, None])  # (npt,naux)
    del aux_cart

    # ---- Backprop fit_metric_gram: Y = X_aux_p @ G, (Gm+lam I) G = L ----
    Gm = cp.ascontiguousarray(X_aux_p.T @ X_aux_p)
    rcond = float(solve_rcond)
    try:
        smax = float(cp.linalg.norm(Gm, ord=2).item())
    except Exception:
        smax = 0.0
    lam = (float(rcond) ** 2) * max(float(smax), 1.0)
    if lam != 0.0:
        Gm = Gm + float(lam) * cp.eye(int(Gm.shape[0]), dtype=cp.float64)

    G = cp.linalg.solve(Gm, L)  # (naux,naux)

    bar_X_aux_p = bar_Y @ G.T
    bar_G = X_aux_p.T @ bar_Y

    U = cp.linalg.solve(Gm.T, bar_G)  # A^{-T} bar_G
    bar_L = cp.tril(U)
    bar_V = chol_lower_adjoint(L, bar_L)

    bar_Gm = -(U @ G.T)
    bar_X_aux_p = bar_X_aux_p + X_aux_p @ (bar_Gm + bar_Gm.T)

    del Gm, G, bar_G, U, bar_L, bar_Gm

    # ---- Accumulate atom gradients from collocation + Becke weights ----
    bar_w = (0.25 / w) * cp.sum(bar_X * X, axis=1)
    bar_w += (0.5 / w) * cp.sum(bar_X_aux_p * X_aux_p, axis=1)
    del X_aux_p

    grad_thc = cp.zeros((natm, 3), dtype=cp.float64)

    # AO collocation VJP needs cart basis.
    if bool(is_spherical):
        if sph_map is None:
            raise RuntimeError("expected sph_map for mol.cart=False")
        T_c2s = getattr(sph_map, "T_c2s", None)
        if T_c2s is None:
            T_c2s = sph_map[0]
        T_dev = cp.asarray(np.asarray(T_c2s, dtype=np.float64), dtype=cp.float64)
        bar_X_cart = bar_X @ T_dev.T
    else:
        bar_X_cart = bar_X

    shell_atom_cart = shell_to_atom_map(ao_basis_cart, atom_coords_bohr=coords)
    grad_thc = contract_aos_cart_value_grad_vjp_atomgrad_device(
        ao_basis_cart,
        pts,
        point_atom=p_atom,
        w_pow=w_quart,
        bar_ao=bar_X_cart,
        shell_atom=cp.asarray(shell_atom_cart, dtype=cp.int32),
        natm=natm,
        out=grad_thc,
        threads=int(threads),
        sync=False,
    )

    # Aux collocation contributions.
    shell_atom_aux = shell_to_atom_map(aux_basis_cart, atom_coords_bohr=coords)
    grad_thc = contract_aos_cart_value_grad_vjp_atomgrad_device(
        aux_basis_cart,
        pts,
        point_atom=p_atom,
        w_pow=w_sqrt,
        bar_ao=bar_X_aux_p,
        shell_atom=cp.asarray(shell_atom_aux, dtype=cp.int32),
        natm=natm,
        out=grad_thc,
        threads=int(threads),
        sync=False,
    )

    # Becke partition weight derivative contributions.
    atom_coords_dev = cp.ascontiguousarray(cp.asarray(coords, dtype=cp.float64))
    grad_thc = becke_weight_vjp_atomgrad_device(
        pts,
        w,
        bar_w=bar_w,
        point_atom=p_atom,
        atom_coords=atom_coords_dev,
        becke_n=int(becke_n),
        out=grad_thc,
        threads=int(threads),
        sync=False,
    )

    # Metric derivative contraction (cuERI CUDA).
    grad_metric = _metric_2c2e_deriv_aux_atomgrad_cuda(
        aux_basis_cart,
        atom_coords_bohr=coords,
        bar_V=bar_V,
        df_threads=int(df_threads),
    )

    return grad_thc, grad_metric


def casscf_nuc_grad_thc(
    scf_out: Any,
    casscf: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    atmlst: Sequence[int] | None = None,
    q_block: int = 256,
    pair_p_block: int = 8,
    df_threads: int = 0,
    int1e_contract_backend: Literal["auto", "cpu", "cuda"] = "cpu",
    solver_kwargs: dict[str, Any] | None = None,
    want_components: bool = False,
    profile: dict | None = None,
) -> DFNucGradResult | tuple[DFNucGradResult, THCNucGradComponents]:
    """Analytic nuclear gradient for the SA-CASSCF objective using THC 2e integrals."""

    t0_total = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile.clear()

    cp = _require_cupy()

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    is_spherical = not bool(getattr(mol, "cart", True))

    thc = getattr(scf_out, "thc_factors", None)
    if thc is None:
        raise ValueError("scf_out.thc_factors is missing")

    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415
    from asuka.hf.local_thc_factors import LocalTHCFactors  # noqa: PLC0415

    if not isinstance(thc, (THCFactors, LocalTHCFactors)):
        raise TypeError("scf_out.thc_factors must be THCFactors or LocalTHCFactors")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])
    if natm <= 0:
        res0 = DFNucGradResult(e_tot=float(getattr(casscf, "e_tot", 0.0)), e_nuc=float(mol.energy_nuc()), grad=np.zeros((0, 3)))
        if want_components:
            g0 = np.zeros((0, 3), dtype=np.float64)
            return res0, THCNucGradComponents(nuc=g0, hcore=g0, pulay=g0, thc_2e=g0, metric=g0)
        return res0

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")

    nroots = int(getattr(casscf, "nroots", 1))
    weights = normalize_weights(getattr(casscf, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(getattr(casscf, "ci"), nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    dm1_act, dm2_act = make_state_averaged_rdms(
        fcisolver_use,
        ci_list,
        weights,
        ncas=int(ncas),
        nelecas=nelecas,
        solver_kwargs=solver_kwargs,
    )
    t_rdms = time.perf_counter() if profile is not None else 0.0

    # ---- Build generalized Fock + densities (GPU) ----
    C = cp.asarray(getattr(casscf, "mo_coeff"), dtype=cp.float64)
    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_thc(
        scf_out,
        C=C,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        q_block=int(q_block),
        pair_p_block=int(pair_p_block),
        profile=profile.setdefault("gfock", {}) if profile is not None else None,
    )
    t_gfock = time.perf_counter() if profile is not None else 0.0

    # ---- THC 2e adjoint (GPU) ----
    dm2_flat_sym = _dm2_sym_flat(cp, dm2_act, ncas=int(ncas))
    D_w = D_act_ao + 0.5 * D_core_ao

    grad_thc_dev = cp.zeros((natm, 3), dtype=cp.float64)
    grad_metric_dev = cp.zeros((natm, 3), dtype=cp.float64)

    sph_map = getattr(scf_out, "sph_map", None)
    ao_basis_cart = getattr(scf_out, "ao_basis")
    aux_basis_cart = getattr(scf_out, "aux_basis")

    if isinstance(thc, THCFactors):
        meta = {} if thc.meta is None else dict(thc.meta)
        solve_method = str(meta.get("solve_method", "fit_metric_qr")).strip().lower()
        inv_metric_methods = {"inv_metric", "inv", "metric_inv", "vinv", "v_inv"}
        fit_metric_gram_methods = {"fit_metric_gram", "gram"}
        if solve_method in inv_metric_methods:
            solve_kind = "inv_metric"
        elif solve_method in fit_metric_gram_methods:
            solve_kind = "fit_metric_gram"
        else:
            raise NotImplementedError(
                "THC-CASSCF analytic gradients currently support solve_method in {'inv_metric','fit_metric_gram'} "
                f"(got {solve_method!r})"
            )
        if bool(meta.get("downselected", False)):
            raise NotImplementedError("THC-CASSCF analytic gradients require THC factors built without point downselect")

        point_atom = meta.get("point_atom", None)
        if point_atom is None:
            raise ValueError("THC factors are missing meta['point_atom']; rebuild THC factors with grid_kind='becke' or 'rdvr'")
        grid_kind = str(meta.get("grid_kind", "")).strip().lower()
        if grid_kind not in {"becke", "rdvr"}:
            raise NotImplementedError("THC-CASSCF analytic gradients currently support only grid_kind in {'becke','rdvr'}")
        becke_n = int(meta.get("becke_n", 3))

        bar_X_mean, bar_Y_mean = _thc_energy_adjoint_jk_bilinear(
            D_core_ao,
            D_w,
            thc.X,
            thc.Z,
            thc.Y,
            cJ=1.0,
            cK=-0.5,
            q_block=int(q_block),
        )
        bar_X_aa, bar_Y_aa = _thc_energy_adjoint_active_global(
            thc.X,
            thc.Y,
            C_act,
            dm2_flat_sym,
            pair_p_block=int(pair_p_block),
        )
        bar_X = bar_X_mean + bar_X_aa
        bar_Y = bar_Y_mean + bar_Y_aa

        if solve_kind == "inv_metric":
            g_thc_blk, g_metric_blk = _thc_factor_vjp_atomgrad_inv_metric(
                mol=mol,
                ao_basis_cart=ao_basis_cart,
                aux_basis_cart=aux_basis_cart,
                sph_map=sph_map,
                is_spherical=bool(is_spherical),
                pts=thc.points,
                w=thc.weights,
                point_atom=point_atom,
                becke_n=int(becke_n),
                X=thc.X,
                Y=thc.Y,
                L_metric=thc.L_metric,
                bar_X=bar_X,
                bar_Y=bar_Y,
                df_threads=int(df_threads),
            )
        else:
            g_thc_blk, g_metric_blk = _thc_factor_vjp_atomgrad_fit_metric_gram(
                mol=mol,
                ao_basis_cart=ao_basis_cart,
                aux_basis_cart=aux_basis_cart,
                sph_map=sph_map,
                is_spherical=bool(is_spherical),
                pts=thc.points,
                w=thc.weights,
                point_atom=point_atom,
                becke_n=int(becke_n),
                X=thc.X,
                Y=thc.Y,
                L_metric=thc.L_metric,
                bar_X=bar_X,
                bar_Y=bar_Y,
                solve_rcond=float(meta.get("solve_rcond", 1e-12)),
                df_threads=int(df_threads),
            )

        grad_thc_dev += g_thc_blk
        grad_metric_dev += g_metric_blk
    else:
        # Local THC: sum block contributions.
        from asuka.cueri.basis_subset import subset_cart_basis_by_shells  # noqa: PLC0415

        lmeta = {} if thc.meta is None else dict(thc.meta)
        solve_method = str(lmeta.get("solve_method", "fit_metric_qr")).strip().lower()
        inv_metric_methods = {"inv_metric", "inv", "metric_inv", "vinv", "v_inv"}
        fit_metric_gram_methods = {"fit_metric_gram", "gram"}
        if solve_method in inv_metric_methods:
            solve_kind = "inv_metric"
        elif solve_method in fit_metric_gram_methods:
            solve_kind = "fit_metric_gram"
        else:
            raise NotImplementedError(
                "LocalTHC analytic gradients currently support solve_method in {'inv_metric','fit_metric_gram'} "
                f"(got {solve_method!r})"
            )
        if bool(lmeta.get("downselected", False)):
            raise NotImplementedError("LocalTHC analytic gradients require factors built without point downselect")

        dm2_flat_sym = cp.ascontiguousarray(dm2_flat_sym)

        for blk in thc.blocks:
            bmeta = {} if getattr(blk, "meta", None) is None else dict(getattr(blk, "meta"))
            if bool(bmeta.get("downselected", False)):
                raise NotImplementedError("LocalTHC analytic gradients require blocks built without point downselect")
            point_atom = bmeta.get("point_atom", None)
            if point_atom is None:
                raise ValueError("LocalTHCBlock.meta['point_atom'] is missing; rebuild local THC factors with gradient metadata")
            grid_kind = str(bmeta.get("grid_kind", "")).strip().lower()
            if grid_kind not in {"becke", "rdvr"}:
                raise NotImplementedError("LocalTHC analytic gradients currently support only grid_kind in {'becke','rdvr'}")
            becke_n = int(bmeta.get("becke_n", 3))

            ao_shells = bmeta.get("ao_shells", None)
            aux_shells = bmeta.get("aux_shells", None)
            if ao_shells is None or aux_shells is None:
                raise ValueError("LocalTHCBlock.meta is missing ao_shells/aux_shells; rebuild local THC factors with metadata")

            ao_basis_blk = subset_cart_basis_by_shells(ao_basis_cart, list(map(int, ao_shells)))
            aux_basis_blk = subset_cart_basis_by_shells(aux_basis_cart, list(map(int, aux_shells)))

            idx_np = np.asarray(getattr(blk, "ao_idx_global"), dtype=np.int32).ravel()
            idx = cp.asarray(idx_np, dtype=cp.int32)
            D_core_sub = D_core_ao[idx[:, None], idx[None, :]]
            D_w_sub = D_w[idx[:, None], idx[None, :]]

            # Ownership mask: move the mask from outputs into D_left.
            nloc = int(idx_np.size)
            n_early = int(getattr(blk, "n_early", 0))
            n_primary = int(getattr(blk, "n_primary", 0))
            if n_early < 0 or n_early > nloc:
                raise ValueError("invalid blk.n_early")
            if n_primary < 0 or (n_early + n_primary) > nloc:
                raise ValueError("invalid blk.n_primary")
            tail = int(n_early + n_primary)
            B_eff = D_w_sub.copy()
            if n_early > 0:
                B_eff[:n_early, :] = 0.0
                B_eff[:, :n_early] = 0.0
            if tail < nloc:
                B_eff[tail:, tail:] = 0.0

            bar_X_mean, bar_Y_mean = _thc_energy_adjoint_jk_bilinear(
                D_core_sub,
                B_eff,
                blk.X,
                blk.Z,
                blk.Y,
                cJ=1.0,
                cK=-0.5,
                q_block=int(q_block),
            )

            bar_X_aa, bar_Y_aa = _thc_energy_adjoint_active_local_block(
                blk.X,
                blk.Z,
                blk.Y,
                ao_idx_global=idx_np,
                n_early=n_early,
                n_primary=n_primary,
                C_act=C_act,
                dm2_flat_sym=dm2_flat_sym,
                pair_p_block=int(pair_p_block),
                q_block=int(q_block),
            )

            bar_X_blk = bar_X_mean + bar_X_aa
            bar_Y_blk = bar_Y_mean + bar_Y_aa

            blk_sph_map = None
            if bool(is_spherical):
                # Local blocks store X in spherical AO ordering; reconstruct the
                # cart->sph transform for this subset basis so the AO-value VJP
                # can be evaluated in cart space.
                from asuka.integrals.cart2sph import (  # noqa: PLC0415
                    build_cart2sph_matrix,
                    compute_sph_layout_from_cart_basis,
                )
                from asuka.cueri.cart import ncart  # noqa: PLC0415

                shell_l_blk = np.asarray(getattr(ao_basis_blk, "shell_l"), dtype=np.int32).ravel()
                shell_start_cart_blk = np.asarray(getattr(ao_basis_blk, "shell_ao_start"), dtype=np.int32).ravel()
                shell_start_sph_blk, nao_sph_blk = compute_sph_layout_from_cart_basis(ao_basis_blk)
                if int(shell_l_blk.size):
                    nfn_cart = np.asarray([ncart(int(l)) for l in shell_l_blk.tolist()], dtype=np.int32)
                    nao_cart_blk = int(np.max(shell_start_cart_blk + nfn_cart))
                else:
                    nao_cart_blk = 0
                T = build_cart2sph_matrix(
                    shell_l_blk,
                    shell_start_cart_blk,
                    np.asarray(shell_start_sph_blk, dtype=np.int32).ravel(),
                    int(nao_cart_blk),
                    int(nao_sph_blk),
                )

                class _TmpSphMap:
                    T_c2s = T

                blk_sph_map = _TmpSphMap()

            if solve_kind == "inv_metric":
                g_thc_blk, g_metric_blk = _thc_factor_vjp_atomgrad_inv_metric(
                    mol=mol,
                    ao_basis_cart=ao_basis_blk,
                    aux_basis_cart=aux_basis_blk,
                    sph_map=blk_sph_map,
                    is_spherical=bool(is_spherical),
                    pts=getattr(blk, "points"),
                    w=getattr(blk, "weights"),
                    point_atom=point_atom,
                    becke_n=int(becke_n),
                    X=getattr(blk, "X"),
                    Y=getattr(blk, "Y"),
                    L_metric=getattr(blk, "L_metric"),
                    bar_X=bar_X_blk,
                    bar_Y=bar_Y_blk,
                    df_threads=int(df_threads),
                )
            else:
                g_thc_blk, g_metric_blk = _thc_factor_vjp_atomgrad_fit_metric_gram(
                    mol=mol,
                    ao_basis_cart=ao_basis_blk,
                    aux_basis_cart=aux_basis_blk,
                    sph_map=blk_sph_map,
                    is_spherical=bool(is_spherical),
                    pts=getattr(blk, "points"),
                    w=getattr(blk, "weights"),
                    point_atom=point_atom,
                    becke_n=int(becke_n),
                    X=getattr(blk, "X"),
                    Y=getattr(blk, "Y"),
                    L_metric=getattr(blk, "L_metric"),
                    bar_X=bar_X_blk,
                    bar_Y=bar_Y_blk,
                    solve_rcond=1e-12,
                    df_threads=int(df_threads),
                )

            grad_thc_dev += g_thc_blk
            grad_metric_dev += g_metric_blk

    t_thc = time.perf_counter() if profile is not None else 0.0

    # Synchronize once before copying back (THC + metric kernels are async).
    cp.cuda.get_current_stream().synchronize()

    de_thc_2e = np.asarray(cp.asnumpy(grad_thc_dev), dtype=np.float64)
    de_metric = np.asarray(cp.asnumpy(grad_metric_dev), dtype=np.float64)

    # ---- 1e + nuclear (CPU, with spherical CUDA fast path when available) ----
    from asuka.integrals.int1e_cart import contract_dhcore_cart, shell_to_atom_map  # noqa: PLC0415

    ao_basis = ao_basis_cart
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    if bool(is_spherical):
        # Mirror DF gradient policy: use CUDA fused contractions when available.
        from asuka.mcscf.nuc_grad_df import _select_sph_int1e_backend  # noqa: PLC0415

        sph_backend = _select_sph_int1e_backend(contract_backend=str(int1e_contract_backend), df_backend="cuda")
        if str(sph_backend) == "cuda":
            from asuka.mcscf.nuc_grad_df import _build_sph_int1e_prebuilt  # noqa: PLC0415

            dS_pre, dT_pre, dV_pre = _build_sph_int1e_prebuilt(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                shell_atom=shell_atom,
                need_overlap=False,
                need_hcore=True,
                to_gpu=True,
            )
            from asuka.integrals.int1e_sph_cuda import contract_dhcore_sph_prebuilt_cuda  # noqa: PLC0415

            de_h1 = contract_dhcore_sph_prebuilt_cuda(dT_pre, dV_pre, D_tot_ao)
        else:
            from asuka.integrals.int1e_sph import contract_dhcore_sph  # noqa: PLC0415

            de_h1 = contract_dhcore_sph(
                ao_basis,
                atom_coords_bohr=coords,
                atom_charges=charges,
                M_sph=_asnumpy_f64(D_tot_ao),
                shell_atom=shell_atom,
            )
    else:
        de_h1 = contract_dhcore_cart(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M=_asnumpy_f64(D_tot_ao),
            shell_atom=shell_atom,
            contract_backend=str(int1e_contract_backend),
        )
    de_pulay = np.zeros_like(np.asarray(de_h1, dtype=np.float64))

    de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)

    comps = THCNucGradComponents(
        nuc=np.asarray(de_nuc, dtype=np.float64),
        hcore=np.asarray(de_h1, dtype=np.float64),
        pulay=np.asarray(de_pulay, dtype=np.float64),
        thc_2e=np.asarray(de_thc_2e, dtype=np.float64),
        metric=np.asarray(de_metric, dtype=np.float64),
    )

    grad = comps.total
    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grad = grad[idx]

    if profile is not None:
        profile["t_rdms_s"] = float(t_rdms - t0_total)
        profile["t_gfock_s"] = float(t_gfock - t_rdms)
        profile["t_thc_s"] = float(t_thc - t_gfock)
        profile["t_total_s"] = float(time.perf_counter() - t0_total)

    res = DFNucGradResult(
        e_tot=float(getattr(casscf, "e_tot", 0.0)),
        e_nuc=float(mol.energy_nuc()),
        grad=np.asarray(grad, dtype=np.float64),
    )
    return (res, comps) if want_components else res


__all__ = ["THCNucGradComponents", "casscf_nuc_grad_thc"]
