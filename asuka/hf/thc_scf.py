from __future__ import annotations

"""SCF (RHF/UHF/ROHF) drivers using THC J/K backends.

These routines mirror the control flow in `asuka.hf.df_scf` but swap the 2e
backend to THC via `asuka.hf.thc_jk`.

Notes
-----
- Inputs `S`/`hcore` are in the AO representation of the SCF (cart or sph).
- THC factors `X,Z` must match that AO representation (`X.shape[1] == nao`).
- Density-difference is handled by providing a reference (D_ref, J_ref, K_ref).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .df_scf import SCFResult
from .thc_factors import THCFactors
from .thc_jk import THCJKWork, THCPrecisionPolicy, thc_J, thc_JK, thc_K_blocked

from . import df_scf as _df  # reuse orthogonalization/DIIS helpers


@dataclass(frozen=True)
class THCReferenceRHF:
    D_ref: Any
    J_ref: Any
    K_ref: Any


@dataclass(frozen=True)
class THCReferenceUHF:
    Da_ref: Any
    Db_ref: Any
    J_ref: Any
    Ka_ref: Any
    Kb_ref: Any


def rhf_thc(
    S,
    hcore,
    thc: THCFactors,
    *,
    nelec: int,
    enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 1,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    q_block: int = 256,
    mp_mode: str = "fp64",
    rebase_dD_rel_tol: float = 0.25,
    rebase_min_cycle: int = 2,
    tc_balance: bool = True,
    dm0=None,
    mo_coeff0=None,
    init_fock_cycles: int = 0,
    reference: THCReferenceRHF | None = None,
    profile: dict | None = None,
    xc_spec=None,
    xc_grid_coords=None,
    xc_grid_weights=None,
    xc_ao_basis=None,
    xc_sph_transform=None,
    xc_batch_size: int = 50000,
) -> SCFResult:
    """RHF SCF with THC J/K backend."""

    xp, _is_gpu = _df._get_xp(S, hcore, thc.X, thc.Y, thc.Z)
    S = _df._as_xp(xp, S, dtype=xp.float64)
    h = _df._as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    if int(thc.X.shape[1]) != int(nao):
        raise ValueError("thc.X nao mismatch with S/hcore")

    X = _df._orthogonalizer_from_S(S)

    occ_np, _nocc = _df._occ_rhf(nelec, nao)
    nocc = int(_nocc)
    occ = _df._as_xp(xp, occ_np, dtype=xp.float64)

    eps, C = _df._gen_eigh_with_X(h, X)
    if dm0 is not None:
        D = _df._as_xp(xp, dm0, dtype=xp.float64)
        if D.shape != (nao, nao):
            raise ValueError("dm0 must have shape (nao, nao)")
        D = _df._symmetrize(xp, D)
    elif mo_coeff0 is not None:
        C0 = _df._as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 must have shape (nao, nao)")
        C = C0
        D = _df._symmetrize(xp, _df._density_from_C_occ(C, occ))
    else:
        D = _df._density_from_C_occ(C, occ)

    if reference is None:
        D_ref = xp.zeros((nao, nao), dtype=xp.float64)
        J_ref = xp.zeros((nao, nao), dtype=xp.float64)
        K_ref = xp.zeros((nao, nao), dtype=xp.float64)
    else:
        D_ref = _df._as_xp(xp, reference.D_ref, dtype=xp.float64)
        J_ref = _df._as_xp(xp, reference.J_ref, dtype=xp.float64)
        K_ref = _df._as_xp(xp, reference.K_ref, dtype=xp.float64)

    lam = float(damping) if damping else 0.0
    diis_obj = _df._DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    work = THCJKWork(q_block=int(q_block))

    mp_mode_s = str(mp_mode).strip().lower()
    if mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("mp_mode must be 'fp64' or 'tf32'")
    policy_fp64 = THCPrecisionPolicy(compute_dtype=np.float64, out_dtype=np.float64, use_tf32=False, prefer_Y=True)
    if mp_mode_s == "tf32":
        from .thc_tc import make_thc_tc_cache  # noqa: PLC0415

        tc_cache = make_thc_tc_cache(thc, compute_dtype=np.float32, balance=bool(tc_balance))
        X_tc = tc_cache.X_tc
        Y_tc = tc_cache.Y_tc
        policy_tc = THCPrecisionPolicy(compute_dtype=np.float32, out_dtype=np.float64, use_tf32=True, prefer_Y=True)
    else:
        tc_cache = None
        X_tc = None
        Y_tc = None
        policy_tc = None

    n_rebase = 0

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("init_fock_ms", 0.0)
        prof.setdefault("init_fock_cycles", int(init_fock_cycles))
        prof.setdefault("init_fock_applied", False)
        prof.setdefault("mp_mode", str(mp_mode_s))
        prof.setdefault("tc_balance", bool(tc_balance))
        prof.setdefault("rebase_dD_rel_tol", float(rebase_dD_rel_tol))
        prof.setdefault("rebase_min_cycle", int(rebase_min_cycle))
        prof.setdefault("rebases", 0)
        if tc_cache is not None:
            prof.setdefault("tc_cache", dict(getattr(tc_cache, "meta", {}) or {}))
        prof.setdefault("iters", 0)

    init_fock_cycles = max(0, int(init_fock_cycles))
    run_init = bool(init_fock_cycles > 0 and dm0 is None and mo_coeff0 is None)
    if profile is not None:
        profile.setdefault("scf", {})["init_fock_applied"] = bool(run_init)

    if run_init:
        for _ in range(int(init_fock_cycles)):
            D_prev = D
            t_init = _df._time_ms_start(xp) if profile is not None else None

            dD = D_prev - D_ref
            if mp_mode_s == "tf32":
                rel = float(xp.linalg.norm(dD).item()) / max(float(xp.linalg.norm(D_prev).item()), 1e-30)
                init_cycle = int(_) + 1
                if init_cycle >= int(rebase_min_cycle) and rel > float(rebase_dD_rel_tol):
                    J_thc, K_thc = thc_JK(dD, thc.X, thc.Z, work=work, Y=thc.Y, policy=policy_fp64)
                    J_ref = J_ref + J_thc
                    K_ref = K_ref + K_thc
                    D_ref = D_prev
                    n_rebase += 1
                    if profile is not None:
                        profile.setdefault("scf", {})["rebases"] = int(n_rebase)
                    J = J_ref
                    K = K_ref
                else:
                    assert X_tc is not None and Y_tc is not None and policy_tc is not None
                    J_thc, K_thc = thc_JK(dD, X_tc, None, work=work, Y=Y_tc, policy=policy_tc)
                    J = J_ref + J_thc
                    K = K_ref + K_thc
            else:
                J_thc, K_thc = thc_JK(dD, thc.X, thc.Z, work=work, Y=thc.Y, policy=policy_fp64)
                J = J_ref + J_thc
                K = K_ref + K_thc

            _cx_init = 0.5 * float(xc_spec.cx_hf) if xc_spec is not None else 0.5
            F = h + J - _cx_init * K
            if xc_spec is not None:
                from asuka.xc.numint import build_vxc as _build_vxc
                _Vxc, _Exc = _build_vxc(xc_spec, D_prev, xc_ao_basis, xc_grid_coords,
                                         xc_grid_weights, batch_size=int(xc_batch_size),
                                         sph_transform=xc_sph_transform)
                F = F + _Vxc
            F = _df._symmetrize(xp, F)

            if level_shift:
                shift = float(level_shift)
                if shift != 0.0:
                    Fp = X.T @ F @ X
                    Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                    F = X @ Fp @ X.T

            eps, C = _df._gen_eigh_with_X(F, X)
            D = _df._symmetrize(xp, _df._density_from_C_occ(C, occ))
            if damping:
                D = (1.0 - lam) * D + lam * D_prev

            if profile is not None and t_init is not None:
                profile["scf"]["init_fock_ms"] += _df._time_ms_end(xp, t_init)

    _cx_main = 0.5 * float(xc_spec.cx_hf) if xc_spec is not None else 0.5
    _E_xc = 0.0
    for cycle in range(1, int(max_cycle) + 1):
        t = _df._time_ms_start(xp)
        dD = D - D_ref
        if mp_mode_s == "tf32":
            rel = float(xp.linalg.norm(dD).item()) / max(float(xp.linalg.norm(D).item()), 1e-30)
            if int(cycle) >= int(rebase_min_cycle) and rel > float(rebase_dD_rel_tol):
                J_thc, K_thc = thc_JK(dD, thc.X, thc.Z, work=work, Y=thc.Y, policy=policy_fp64)
                J_ref = J_ref + J_thc
                K_ref = K_ref + K_thc
                D_ref = D
                n_rebase += 1
                if profile is not None:
                    profile.setdefault("scf", {})["rebases"] = int(n_rebase)
                J = J_ref
                K = K_ref
            else:
                assert X_tc is not None and Y_tc is not None and policy_tc is not None
                J_thc, K_thc = thc_JK(dD, X_tc, None, work=work, Y=Y_tc, policy=policy_tc)
                J = J_ref + J_thc
                K = K_ref + K_thc
        else:
            J_thc, K_thc = thc_JK(dD, thc.X, thc.Z, work=work, Y=thc.Y, policy=policy_fp64)
            J = J_ref + J_thc
            K = K_ref + K_thc
        if profile is not None:
            profile["scf"]["jk_ms"] += _df._time_ms_end(xp, t)

        F = h + J - _cx_main * K
        if xc_spec is not None:
            from asuka.xc.numint import build_vxc as _build_vxc
            _Vxc, _E_xc = _build_vxc(xc_spec, D, xc_ao_basis, xc_grid_coords,
                                       xc_grid_weights, batch_size=int(xc_batch_size),
                                       sph_transform=xc_sph_transform)
            F = F + _Vxc
        F = _df._symmetrize(xp, F)

        if level_shift:
            shift = float(level_shift)
            if shift != 0.0:
                Fp = X.T @ F @ X
                Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                F = X @ Fp @ X.T

        if diis_obj is not None and cycle >= int(diis_start_cycle):
            t = _df._time_ms_start(xp)
            e = _df._fock_error_rhf(F, D, S)
            diis_obj.push(F, e)
            F = _df._symmetrize(xp, diis_obj.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _df._time_ms_end(xp, t)

        t = _df._time_ms_start(xp)
        eps, C = _df._gen_eigh_with_X(F, X)
        if profile is not None:
            profile["scf"]["diag_ms"] += _df._time_ms_end(xp, t)
        D_new = _df._density_from_C_occ(C, occ)

        if damping:
            D_new = (1.0 - lam) * D_new + lam * D

        if xc_spec is not None:
            e_one = float(xp.trace(D_new @ h).item())
            e_coul = float(0.5 * xp.trace(D_new @ J).item())
            e_ex = float(_cx_main * 0.5 * xp.trace(D_new @ K).item())
            e_elec = e_one + e_coul - e_ex + _E_xc
        else:
            e_elec = float(0.5 * xp.trace(D_new @ (h + F)).item())
        e_tot = float(e_elec + float(enuc))

        dm_err = float(xp.linalg.norm(D_new - D).item())
        de = float(abs(e_tot - e_last)) if e_last is not None else float("inf")

        if de < float(conv_tol) and dm_err < float(conv_tol_dm):
            converged = True
            D = D_new
            e_last = e_tot
            break

        D = D_new
        e_last = e_tot

    if profile is not None:
        profile.setdefault("scf", {})["iters"] = int(cycle)

    return SCFResult(
        method="RKS-THC" if xc_spec is not None else "RHF-THC",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=eps,
        mo_coeff=C,
        mo_occ=occ,
    )


def uhf_thc(
    S,
    hcore,
    thc: THCFactors,
    *,
    nalpha: int,
    nbeta: int,
    enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    q_block: int = 256,
    mp_mode: str = "fp64",
    rebase_dD_rel_tol: float = 0.25,
    rebase_min_cycle: int = 2,
    tc_balance: bool = True,
    dm0=None,
    mo_coeff0=None,
    reference: THCReferenceUHF | None = None,
    profile: dict | None = None,
    xc_spec=None,
    xc_grid_coords=None,
    xc_grid_weights=None,
    xc_ao_basis=None,
    xc_sph_transform=None,
    xc_batch_size: int = 50000,
) -> SCFResult:
    """UHF SCF with THC J/K backend (or UKS if ``xc_spec`` is provided)."""

    xp, _is_gpu = _df._get_xp(S, hcore, thc.X, thc.Y, thc.Z)
    S = _df._as_xp(xp, S, dtype=xp.float64)
    h = _df._as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    if int(thc.X.shape[1]) != int(nao):
        raise ValueError("thc.X nao mismatch with S/hcore")

    X = _df._orthogonalizer_from_S(S)

    occ_a_np, occ_b_np = _df._occ_uhf(nalpha, nbeta, nao)
    occ_a = _df._as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _df._as_xp(xp, occ_b_np, dtype=xp.float64)

    _e0, C = _df._gen_eigh_with_X(h, X)
    Ca = C
    Cb = C

    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for UHF must be a (Da, Db) tuple")
        Da = _df._as_xp(xp, dm0[0], dtype=xp.float64)
        Db = _df._as_xp(xp, dm0[1], dtype=xp.float64)
        if Da.shape != (nao, nao) or Db.shape != (nao, nao):
            raise ValueError("dm0 for UHF must have shape (nao, nao) for both spins")
        Da = _df._symmetrize(xp, Da)
        Db = _df._symmetrize(xp, Db)
    elif mo_coeff0 is not None:
        if isinstance(mo_coeff0, (tuple, list)):
            if len(mo_coeff0) != 2:
                raise TypeError("mo_coeff0 for UHF must be a (Ca, Cb) tuple")
            Ca0 = _df._as_xp(xp, mo_coeff0[0], dtype=xp.float64)
            Cb0 = _df._as_xp(xp, mo_coeff0[1], dtype=xp.float64)
        else:
            Ca0 = _df._as_xp(xp, mo_coeff0, dtype=xp.float64)
            Cb0 = Ca0
        if Ca0.shape != (nao, nao) or Cb0.shape != (nao, nao):
            raise ValueError("mo_coeff0 for UHF must have shape (nao, nao) for both spins")
        Ca = Ca0
        Cb = Cb0
        Da = _df._symmetrize(xp, _df._density_from_C_occ(Ca0, occ_a))
        Db = _df._symmetrize(xp, _df._density_from_C_occ(Cb0, occ_b))
    else:
        Da = _df._density_from_C_occ(Ca, occ_a)
        Db = _df._density_from_C_occ(Cb, occ_b)

    if reference is None:
        Da_ref = xp.zeros((nao, nao), dtype=xp.float64)
        Db_ref = xp.zeros((nao, nao), dtype=xp.float64)
        J_ref = xp.zeros((nao, nao), dtype=xp.float64)
        Ka_ref = xp.zeros((nao, nao), dtype=xp.float64)
        Kb_ref = xp.zeros((nao, nao), dtype=xp.float64)
    else:
        Da_ref = _df._as_xp(xp, reference.Da_ref, dtype=xp.float64)
        Db_ref = _df._as_xp(xp, reference.Db_ref, dtype=xp.float64)
        J_ref = _df._as_xp(xp, reference.J_ref, dtype=xp.float64)
        Ka_ref = _df._as_xp(xp, reference.Ka_ref, dtype=xp.float64)
        Kb_ref = _df._as_xp(xp, reference.Kb_ref, dtype=xp.float64)

    lam = float(damping) if damping else 0.0
    diis_a = _df._DIIS(max_vec=diis_space) if diis else None
    diis_b = _df._DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    work = THCJKWork(q_block=int(q_block))

    mp_mode_s = str(mp_mode).strip().lower()
    if mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("mp_mode must be 'fp64' or 'tf32'")
    policy_fp64 = THCPrecisionPolicy(compute_dtype=np.float64, out_dtype=np.float64, use_tf32=False, prefer_Y=True)
    if mp_mode_s == "tf32":
        from .thc_tc import make_thc_tc_cache  # noqa: PLC0415

        tc_cache = make_thc_tc_cache(thc, compute_dtype=np.float32, balance=bool(tc_balance))
        X_tc = tc_cache.X_tc
        Y_tc = tc_cache.Y_tc
        policy_tc = THCPrecisionPolicy(compute_dtype=np.float32, out_dtype=np.float64, use_tf32=True, prefer_Y=True)
    else:
        tc_cache = None
        X_tc = None
        Y_tc = None
        policy_tc = None

    n_rebase = 0

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("iters", 0)
        prof.setdefault("mp_mode", str(mp_mode_s))
        prof.setdefault("tc_balance", bool(tc_balance))
        prof.setdefault("rebase_dD_rel_tol", float(rebase_dD_rel_tol))
        prof.setdefault("rebase_min_cycle", int(rebase_min_cycle))
        prof.setdefault("rebases", 0)
        if tc_cache is not None:
            prof.setdefault("tc_cache", dict(getattr(tc_cache, "meta", {}) or {}))

    for cycle in range(1, int(max_cycle) + 1):
        Dtot = Da + Db
        Dtot_ref = Da_ref + Db_ref

        t = _df._time_ms_start(xp)
        dDtot = Dtot - Dtot_ref
        dDa = Da - Da_ref
        dDb = Db - Db_ref
        if mp_mode_s == "tf32":
            rel = float(xp.linalg.norm(dDtot).item()) / max(float(xp.linalg.norm(Dtot).item()), 1e-30)
            if int(cycle) >= int(rebase_min_cycle) and rel > float(rebase_dD_rel_tol):
                dJ = thc_J(dDtot, thc.X, thc.Z, Y=thc.Y, policy=policy_fp64)
                dKa = thc_K_blocked(dDa, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
                dKb = thc_K_blocked(dDb, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
                J_ref = J_ref + dJ
                Ka_ref = Ka_ref + dKa
                Kb_ref = Kb_ref + dKb
                Da_ref = Da
                Db_ref = Db
                n_rebase += 1
                if profile is not None:
                    profile.setdefault("scf", {})["rebases"] = int(n_rebase)
                J = J_ref
                Ka = Ka_ref
                Kb = Kb_ref
            else:
                assert X_tc is not None and Y_tc is not None and policy_tc is not None
                dJ = thc_J(dDtot, X_tc, None, Y=Y_tc, policy=policy_tc)
                dKa = thc_K_blocked(dDa, X_tc, None, q_block=int(work.q_block), Y=Y_tc, policy=policy_tc)
                dKb = thc_K_blocked(dDb, X_tc, None, q_block=int(work.q_block), Y=Y_tc, policy=policy_tc)
                J = J_ref + dJ
                Ka = Ka_ref + dKa
                Kb = Kb_ref + dKb
        else:
            dJ = thc_J(dDtot, thc.X, thc.Z, Y=thc.Y, policy=policy_fp64)
            dKa = thc_K_blocked(dDa, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
            dKb = thc_K_blocked(dDb, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
            J = J_ref + dJ
            Ka = Ka_ref + dKa
            Kb = Kb_ref + dKb
        if profile is not None:
            profile["scf"]["jk_ms"] += _df._time_ms_end(xp, t)

        _E_xc = 0.0
        if xc_spec is not None:
            from asuka.xc.numint import build_vxc_u as _build_vxc_u  # noqa: PLC0415

            _cx = float(xc_spec.cx_hf)
            _Vxc_a, _Vxc_b, _E_xc = _build_vxc_u(
                xc_spec,
                Da,
                Db,
                xc_ao_basis,
                xc_grid_coords,
                xc_grid_weights,
                batch_size=int(xc_batch_size),
                sph_transform=xc_sph_transform,
            )
            Fa = _df._symmetrize(xp, h + J - _cx * Ka + _Vxc_a)
            Fb = _df._symmetrize(xp, h + J - _cx * Kb + _Vxc_b)
        else:
            Fa = _df._symmetrize(xp, h + J - Ka)
            Fb = _df._symmetrize(xp, h + J - Kb)

        if diis_a is not None and cycle >= int(diis_start_cycle):
            t = _df._time_ms_start(xp)
            ea_mat = _df._fock_error_rhf(Fa, Da, S)
            diis_a.push(Fa, ea_mat)
            Fa = _df._symmetrize(xp, diis_a.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _df._time_ms_end(xp, t)
        if diis_b is not None and cycle >= int(diis_start_cycle):
            t = _df._time_ms_start(xp)
            eb_mat = _df._fock_error_rhf(Fb, Db, S)
            diis_b.push(Fb, eb_mat)
            Fb = _df._symmetrize(xp, diis_b.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _df._time_ms_end(xp, t)

        t = _df._time_ms_start(xp)
        ea, Ca = _df._gen_eigh_with_X(Fa, X)
        eb, Cb = _df._gen_eigh_with_X(Fb, X)
        if profile is not None:
            profile["scf"]["diag_ms"] += _df._time_ms_end(xp, t)

        Da_new = _df._density_from_C_occ(Ca, occ_a)
        Db_new = _df._density_from_C_occ(Cb, occ_b)

        if damping:
            Da_new = (1.0 - lam) * Da_new + lam * Da
            Db_new = (1.0 - lam) * Db_new + lam * Db

        Dtot_new = Da_new + Db_new
        e_one = float(xp.trace(Dtot_new @ h).item())
        e_coul = float(0.5 * xp.trace(Dtot_new @ J).item())
        e_ex = float(0.5 * xp.trace(Da_new @ Ka).item() + 0.5 * xp.trace(Db_new @ Kb).item())
        if xc_spec is not None:
            _cx_e = float(xc_spec.cx_hf)
            e_elec = e_one + e_coul - _cx_e * e_ex + float(_E_xc)
        else:
            e_elec = e_one + e_coul - e_ex
        e_tot = float(e_elec + float(enuc))

        dm_err = float(xp.linalg.norm(Da_new - Da).item() + xp.linalg.norm(Db_new - Db).item())
        de = float(abs(e_tot - e_last)) if e_last is not None else float("inf")

        if de < float(conv_tol) and dm_err < float(conv_tol_dm):
            converged = True
            Da, Db = Da_new, Db_new
            e_last = e_tot
            break

        Da, Db = Da_new, Db_new
        e_last = e_tot

    if profile is not None:
        profile.setdefault("scf", {})["iters"] = int(cycle)

    return SCFResult(
        method="UKS-THC" if xc_spec is not None else "UHF-THC",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        mo_occ=(occ_a, occ_b),
    )


def rohf_thc(
    S,
    hcore,
    thc: THCFactors,
    *,
    nalpha: int,
    nbeta: int,
    enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    q_block: int = 256,
    mp_mode: str = "fp64",
    rebase_dD_rel_tol: float = 0.25,
    rebase_min_cycle: int = 2,
    tc_balance: bool = True,
    dm0=None,
    mo_coeff0=None,
    reference: THCReferenceUHF | None = None,
    profile: dict | None = None,
) -> SCFResult:
    """ROHF SCF with THC J/K backend."""

    xp, _is_gpu = _df._get_xp(S, hcore, thc.X, thc.Y, thc.Z)
    S = _df._as_xp(xp, S, dtype=xp.float64)
    h = _df._as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    if int(thc.X.shape[1]) != int(nao):
        raise ValueError("thc.X nao mismatch with S/hcore")

    X = _df._orthogonalizer_from_S(S)

    if int(nalpha) < 0 or int(nbeta) < 0 or int(nalpha) < int(nbeta):
        raise ValueError("ROHF requires nalpha>=nbeta>=0")
    if int(nalpha) > int(nao) or int(nbeta) > int(nao):
        raise ValueError("nalpha/nbeta exceeds number of orbitals")

    occ_a_np, occ_b_np = _df._occ_uhf(nalpha, nbeta, nao)
    occ_a = _df._as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _df._as_xp(xp, occ_b_np, dtype=xp.float64)

    e0, C = _df._gen_eigh_with_X(h, X)
    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for ROHF must be a (Da, Db) tuple")
        Da = _df._as_xp(xp, dm0[0], dtype=xp.float64)
        Db = _df._as_xp(xp, dm0[1], dtype=xp.float64)
        if Da.shape != (nao, nao) or Db.shape != (nao, nao):
            raise ValueError("dm0 for ROHF must have shape (nao, nao) for both spins")
        Da = _df._symmetrize(xp, Da)
        Db = _df._symmetrize(xp, Db)
    elif mo_coeff0 is not None:
        C0 = _df._as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 for ROHF must have shape (nao, nao)")
        C = C0
        Da = _df._symmetrize(xp, _df._density_from_C_occ(C, occ_a))
        Db = _df._symmetrize(xp, _df._density_from_C_occ(C, occ_b))
    else:
        Da = _df._density_from_C_occ(C, occ_a)
        Db = _df._density_from_C_occ(C, occ_b)

    if reference is None:
        Da_ref = xp.zeros((nao, nao), dtype=xp.float64)
        Db_ref = xp.zeros((nao, nao), dtype=xp.float64)
        J_ref = xp.zeros((nao, nao), dtype=xp.float64)
        Ka_ref = xp.zeros((nao, nao), dtype=xp.float64)
        Kb_ref = xp.zeros((nao, nao), dtype=xp.float64)
    else:
        Da_ref = _df._as_xp(xp, reference.Da_ref, dtype=xp.float64)
        Db_ref = _df._as_xp(xp, reference.Db_ref, dtype=xp.float64)
        J_ref = _df._as_xp(xp, reference.J_ref, dtype=xp.float64)
        Ka_ref = _df._as_xp(xp, reference.Ka_ref, dtype=xp.float64)
        Kb_ref = _df._as_xp(xp, reference.Kb_ref, dtype=xp.float64)

    lam = float(damping) if damping else 0.0
    diis_obj = _df._DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    work = THCJKWork(q_block=int(q_block))

    mp_mode_s = str(mp_mode).strip().lower()
    if mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("mp_mode must be 'fp64' or 'tf32'")
    policy_fp64 = THCPrecisionPolicy(compute_dtype=np.float64, out_dtype=np.float64, use_tf32=False, prefer_Y=True)
    if mp_mode_s == "tf32":
        from .thc_tc import make_thc_tc_cache  # noqa: PLC0415

        tc_cache = make_thc_tc_cache(thc, compute_dtype=np.float32, balance=bool(tc_balance))
        X_tc = tc_cache.X_tc
        Y_tc = tc_cache.Y_tc
        policy_tc = THCPrecisionPolicy(compute_dtype=np.float32, out_dtype=np.float64, use_tf32=True, prefer_Y=True)
    else:
        tc_cache = None
        X_tc = None
        Y_tc = None
        policy_tc = None

    n_rebase = 0

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("iters", 0)
        prof.setdefault("mp_mode", str(mp_mode_s))
        prof.setdefault("tc_balance", bool(tc_balance))
        prof.setdefault("rebase_dD_rel_tol", float(rebase_dD_rel_tol))
        prof.setdefault("rebase_min_cycle", int(rebase_min_cycle))
        prof.setdefault("rebases", 0)
        if tc_cache is not None:
            prof.setdefault("tc_cache", dict(getattr(tc_cache, "meta", {}) or {}))

    for cycle in range(1, int(max_cycle) + 1):
        Dtot = Da + Db
        Dtot_ref = Da_ref + Db_ref

        t = _df._time_ms_start(xp)
        dDtot = Dtot - Dtot_ref
        dDa = Da - Da_ref
        dDb = Db - Db_ref
        if mp_mode_s == "tf32":
            rel = float(xp.linalg.norm(dDtot).item()) / max(float(xp.linalg.norm(Dtot).item()), 1e-30)
            if int(cycle) >= int(rebase_min_cycle) and rel > float(rebase_dD_rel_tol):
                dJ = thc_J(dDtot, thc.X, thc.Z, Y=thc.Y, policy=policy_fp64)
                dKa = thc_K_blocked(dDa, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
                dKb = thc_K_blocked(dDb, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
                J_ref = J_ref + dJ
                Ka_ref = Ka_ref + dKa
                Kb_ref = Kb_ref + dKb
                Da_ref = Da
                Db_ref = Db
                n_rebase += 1
                if profile is not None:
                    profile.setdefault("scf", {})["rebases"] = int(n_rebase)
                J = J_ref
                Ka = Ka_ref
                Kb = Kb_ref
            else:
                assert X_tc is not None and Y_tc is not None and policy_tc is not None
                dJ = thc_J(dDtot, X_tc, None, Y=Y_tc, policy=policy_tc)
                dKa = thc_K_blocked(dDa, X_tc, None, q_block=int(work.q_block), Y=Y_tc, policy=policy_tc)
                dKb = thc_K_blocked(dDb, X_tc, None, q_block=int(work.q_block), Y=Y_tc, policy=policy_tc)
                J = J_ref + dJ
                Ka = Ka_ref + dKa
                Kb = Kb_ref + dKb
        else:
            dJ = thc_J(dDtot, thc.X, thc.Z, Y=thc.Y, policy=policy_fp64)
            dKa = thc_K_blocked(dDa, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
            dKb = thc_K_blocked(dDb, thc.X, thc.Z, q_block=int(work.q_block), Y=thc.Y, policy=policy_fp64)
            J = J_ref + dJ
            Ka = Ka_ref + dKa
            Kb = Kb_ref + dKb
        if profile is not None:
            profile["scf"]["jk_ms"] += _df._time_ms_end(xp, t)

        Fa = _df._symmetrize(xp, h + J - Ka)
        Fb = _df._symmetrize(xp, h + J - Kb)
        F = _df._roothaan_fock_rohf(Fa, Fb, Da, Db, S)

        if diis_obj is not None and cycle >= int(diis_start_cycle):
            t = _df._time_ms_start(xp)
            e = _df._fock_error_rhf(F, Dtot, S)
            diis_obj.push(F, e)
            F = _df._symmetrize(xp, diis_obj.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _df._time_ms_end(xp, t)

        t = _df._time_ms_start(xp)
        e0, C = _df._gen_eigh_with_X(F, X)
        if profile is not None:
            profile["scf"]["diag_ms"] += _df._time_ms_end(xp, t)

        Da_new = _df._density_from_C_occ(C, occ_a)
        Db_new = _df._density_from_C_occ(C, occ_b)

        if damping:
            Da_new = (1.0 - lam) * Da_new + lam * Da
            Db_new = (1.0 - lam) * Db_new + lam * Db

        Dtot_new = Da_new + Db_new
        e_one = float(xp.trace(Dtot_new @ h).item())
        e_coul = float(0.5 * xp.trace(Dtot_new @ J).item())
        e_ex = float(0.5 * xp.trace(Da_new @ Ka).item() + 0.5 * xp.trace(Db_new @ Kb).item())
        e_elec = e_one + e_coul - e_ex
        e_tot = float(e_elec + float(enuc))

        dm_err = float(xp.linalg.norm(Da_new - Da).item() + xp.linalg.norm(Db_new - Db).item())
        de = float(abs(e_tot - e_last)) if e_last is not None else float("inf")

        if de < float(conv_tol) and dm_err < float(conv_tol_dm):
            converged = True
            Da, Db = Da_new, Db_new
            e_last = e_tot
            break

        Da, Db = Da_new, Db_new
        e_last = e_tot

    if profile is not None:
        profile.setdefault("scf", {})["iters"] = int(cycle)

    return SCFResult(
        method="ROHF-THC",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=e0,
        mo_coeff=C,
        mo_occ=(occ_a, occ_b),
    )


__all__ = [
    "THCReferenceRHF",
    "THCReferenceUHF",
    "rhf_thc",
    "rohf_thc",
    "uhf_thc",
]
