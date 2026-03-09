from __future__ import annotations

"""Integral-direct SCF (RHF/UHF/ROHF) drivers.

Uses integral-direct J/K contraction (no materialized ERI tensor) via
:mod:`asuka.hf.direct_jk`.  Memory usage is O(nao^2) instead of O(nao^4).
"""

from .direct_jk import DirectJKContext, direct_JK, direct_JK_multi, make_direct_jk_context
from .df_scf import (
    SCFResult,
    _DIIS,
    _as_xp,
    _density_from_C_occ,
    _density_from_C_occ_syrk,
    _fock_error_rhf,
    _gen_eigh_with_X,
    _occ_rhf,
    _occ_uhf,
    _orthogonalizer_from_S,
    _roothaan_fock_rohf,
    _symmetrize,
    _symmetrize_inplace,
    _time_ms_end,
    _time_ms_start,
)


def _is_finite_matrix(xp, A) -> bool:
    try:
        return bool(xp.all(xp.isfinite(A)))
    except Exception:
        return False


def rhf_direct(
    S,
    hcore,
    jk_ctx: DirectJKContext,
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
    dm0=None,
    mo_coeff0=None,
    init_fock_cycles: int = 1,
    profile: dict | None = None,
):
    """RHF SCF with integral-direct J/K evaluation."""

    import cupy as cp  # noqa: PLC0415

    xp = cp  # Direct J/K always uses CUDA
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    if nao != jk_ctx.nao:
        raise ValueError(f"nao mismatch: S has {nao}, DirectJKContext has {jk_ctx.nao}")

    X = _orthogonalizer_from_S(S)
    occ_np, nocc = _occ_rhf(nelec, nao)
    occ = _as_xp(xp, occ_np, dtype=xp.float64)

    eps, C = _gen_eigh_with_X(h, X)
    if dm0 is not None:
        D = _as_xp(xp, dm0, dtype=xp.float64)
        if D.shape != (nao, nao):
            raise ValueError("dm0 must have shape (nao, nao)")
        D = _symmetrize_inplace(xp, D)
    elif mo_coeff0 is not None:
        C0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 must have shape (nao, nao)")
        C = C0
        D = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ, nocc))
    else:
        D = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ, nocc))

    lam = float(damping) if damping else 0.0
    diis_obj = _DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("diis_fallbacks", 0)
        prof.setdefault("init_fock_ms", 0.0)
        prof.setdefault("init_fock_cycles", 0)
        prof.setdefault("init_fock_applied", False)
        prof.setdefault("iters", 0)
        jk_prof = profile.setdefault("jk", {})
        jk_prof.setdefault("calls", 0)

    init_fock_cycles_i = max(0, int(init_fock_cycles))
    run_init_predictor = bool(init_fock_cycles_i > 0 and dm0 is None and mo_coeff0 is None)
    if profile is not None:
        profile["scf"]["init_fock_cycles"] = int(init_fock_cycles_i)
        profile["scf"]["init_fock_applied"] = bool(run_init_predictor)

    if run_init_predictor:
        for _ in range(int(init_fock_cycles_i)):
            D_prev = D
            t_init = _time_ms_start(xp) if profile is not None else None
            J, K = direct_JK(jk_ctx, D_prev, want_J=True, want_K=True, profile=profile)
            F = _symmetrize_inplace(xp, h + J - 0.5 * K)
            if level_shift:
                shift = float(level_shift)
                if shift != 0.0:
                    Fp = X.T @ F @ X
                    Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                    F = X @ Fp @ X.T
            eps, C = _gen_eigh_with_X(F, X)
            D = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ, nocc))
            if damping:
                D = (1.0 - lam) * D + lam * D_prev
            if profile is not None and t_init is not None:
                profile["scf"]["init_fock_ms"] += _time_ms_end(xp, t_init)

    cycle = 0
    for cycle in range(1, int(max_cycle) + 1):
        t = _time_ms_start(xp) if profile is not None else None
        J, K = direct_JK(jk_ctx, D, want_J=True, want_K=True, profile=profile)
        if profile is not None and t is not None:
            profile["jk"]["calls"] = int(profile["jk"].get("calls", 0)) + 1
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        F = _symmetrize_inplace(xp, h + J - 0.5 * K)
        if level_shift:
            shift = float(level_shift)
            if shift != 0.0:
                Fp = X.T @ F @ X
                Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                F = X @ Fp @ X.T

        if diis_obj is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp) if profile is not None else None
            F_base = F
            e = _fock_error_rhf(F, D, S)
            diis_obj.push(F, e)
            F_try = None
            try:
                F_try = diis_obj.extrapolate()
            except Exception:
                F_try = None
            if F_try is None or not _is_finite_matrix(xp, F_try):
                F = F_base
                diis_obj = _DIIS(max_vec=diis_space)
                if profile is not None:
                    profile["scf"]["diis_fallbacks"] = int(profile["scf"].get("diis_fallbacks", 0)) + 1
            else:
                F = _symmetrize_inplace(xp, F_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp) if profile is not None else None
        eps, C = _gen_eigh_with_X(F, X)
        if profile is not None and t is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)

        D_new = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ, nocc))
        if damping:
            D_new = (1.0 - lam) * D_new + lam * D

        e_elec = float(0.5 * xp.sum(D_new * (h + F)).item())
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
        profile["scf"]["iters"] = int(cycle)

    return SCFResult(
        method="RHF",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=eps,
        mo_coeff=C,
        mo_occ=occ,
    )


def uhf_direct(
    S,
    hcore,
    jk_ctx: DirectJKContext,
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
    dm0=None,
    mo_coeff0=None,
    profile: dict | None = None,
):
    """UHF SCF with integral-direct J/K evaluation."""

    import cupy as cp  # noqa: PLC0415

    xp = cp
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    X = _orthogonalizer_from_S(S)
    occ_a_np, occ_b_np = _occ_uhf(nalpha, nbeta, nao)
    occ_a = _as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _as_xp(xp, occ_b_np, dtype=xp.float64)

    _e0, C = _gen_eigh_with_X(h, X)
    Ca = C
    Cb = C
    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for UHF must be a (Da, Db) tuple")
        Da = _symmetrize_inplace(xp, _as_xp(xp, dm0[0], dtype=xp.float64))
        Db = _symmetrize_inplace(xp, _as_xp(xp, dm0[1], dtype=xp.float64))
    elif mo_coeff0 is not None:
        if isinstance(mo_coeff0, (tuple, list)):
            if len(mo_coeff0) != 2:
                raise TypeError("mo_coeff0 for UHF must be a (Ca, Cb) tuple")
            Ca = _as_xp(xp, mo_coeff0[0], dtype=xp.float64)
            Cb = _as_xp(xp, mo_coeff0[1], dtype=xp.float64)
        else:
            Ca = _as_xp(xp, mo_coeff0, dtype=xp.float64)
            Cb = Ca
        Da = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, Ca, occ_a, nalpha))
        Db = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, Cb, occ_b, nbeta))
    else:
        Da = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, Ca, occ_a, nalpha))
        Db = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, Cb, occ_b, nbeta))

    lam = float(damping) if damping else 0.0
    diis_a = _DIIS(max_vec=diis_space) if diis else None
    diis_b = _DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("diis_fallbacks", 0)
        prof.setdefault("iters", 0)
        jk_prof = profile.setdefault("jk", {})
        jk_prof.setdefault("calls", 0)

    cycle = 0
    for cycle in range(1, int(max_cycle) + 1):
        t = _time_ms_start(xp) if profile is not None else None
        Ja, Ka, Jb, Kb = direct_JK_multi(jk_ctx, Da, Db, want_J=True, want_K=True, profile=profile)
        J = Ja + Jb
        if profile is not None and t is not None:
            profile["jk"]["calls"] = int(profile["jk"].get("calls", 0)) + 1
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        Fa = _symmetrize_inplace(xp, h + J - Ka)
        Fb = _symmetrize_inplace(xp, h + J - Kb)

        if diis_a is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp) if profile is not None else None
            Fa_base = Fa
            ea = _fock_error_rhf(Fa, Da, S)
            diis_a.push(Fa, ea)
            Fa_try = None
            try:
                Fa_try = diis_a.extrapolate()
            except Exception:
                Fa_try = None
            if Fa_try is None or not _is_finite_matrix(xp, Fa_try):
                Fa = Fa_base
                diis_a = _DIIS(max_vec=diis_space)
                if profile is not None:
                    profile["scf"]["diis_fallbacks"] = int(profile["scf"].get("diis_fallbacks", 0)) + 1
            else:
                Fa = _symmetrize_inplace(xp, Fa_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)
        if diis_b is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp) if profile is not None else None
            Fb_base = Fb
            eb = _fock_error_rhf(Fb, Db, S)
            diis_b.push(Fb, eb)
            Fb_try = None
            try:
                Fb_try = diis_b.extrapolate()
            except Exception:
                Fb_try = None
            if Fb_try is None or not _is_finite_matrix(xp, Fb_try):
                Fb = Fb_base
                diis_b = _DIIS(max_vec=diis_space)
                if profile is not None:
                    profile["scf"]["diis_fallbacks"] = int(profile["scf"].get("diis_fallbacks", 0)) + 1
            else:
                Fb = _symmetrize_inplace(xp, Fb_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp) if profile is not None else None
        ea, Ca = _gen_eigh_with_X(Fa, X)
        eb, Cb = _gen_eigh_with_X(Fb, X)
        if profile is not None and t is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)

        Da_new = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, Ca, occ_a, nalpha))
        Db_new = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, Cb, occ_b, nbeta))
        if damping:
            Da_new = (1.0 - lam) * Da_new + lam * Da
            Db_new = (1.0 - lam) * Db_new + lam * Db

        Dtot_new = Da_new + Db_new
        e_one = float(xp.sum(Dtot_new * h).item())
        e_coul = float(0.5 * xp.sum(Dtot_new * J).item())
        e_ex = float(0.5 * xp.sum(Da_new * Ka).item() + 0.5 * xp.sum(Db_new * Kb).item())
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
        profile["scf"]["iters"] = int(cycle)

    return SCFResult(
        method="UHF",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        mo_occ=(occ_a, occ_b),
    )


def rohf_direct(
    S,
    hcore,
    jk_ctx: DirectJKContext,
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
    dm0=None,
    mo_coeff0=None,
    profile: dict | None = None,
):
    """ROHF SCF with integral-direct J/K evaluation."""

    import cupy as cp  # noqa: PLC0415

    xp = cp
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    X = _orthogonalizer_from_S(S)
    if int(nalpha) < int(nbeta):
        raise ValueError("ROHF requires nalpha >= nbeta")

    occ_a_np, occ_b_np = _occ_uhf(nalpha, nbeta, nao)
    occ_a = _as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _as_xp(xp, occ_b_np, dtype=xp.float64)

    e0, C = _gen_eigh_with_X(h, X)
    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for ROHF must be a (Da, Db) tuple")
        Da = _symmetrize_inplace(xp, _as_xp(xp, dm0[0], dtype=xp.float64))
        Db = _symmetrize_inplace(xp, _as_xp(xp, dm0[1], dtype=xp.float64))
    elif mo_coeff0 is not None:
        C0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
        C = C0
        Da = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ_a, nalpha))
        Db = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ_b, nbeta))
    else:
        Da = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ_a, nalpha))
        Db = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ_b, nbeta))

    lam = float(damping) if damping else 0.0
    diis_obj = _DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("diis_fallbacks", 0)
        prof.setdefault("iters", 0)
        jk_prof = profile.setdefault("jk", {})
        jk_prof.setdefault("calls", 0)

    cycle = 0
    for cycle in range(1, int(max_cycle) + 1):
        t = _time_ms_start(xp) if profile is not None else None
        Ja, Ka, Jb, Kb = direct_JK_multi(jk_ctx, Da, Db, want_J=True, want_K=True, profile=profile)
        J = Ja + Jb
        if profile is not None and t is not None:
            profile["jk"]["calls"] = int(profile["jk"].get("calls", 0)) + 1
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        Fa = _symmetrize_inplace(xp, h + J - Ka)
        Fb = _symmetrize_inplace(xp, h + J - Kb)
        F = _roothaan_fock_rohf(Fa, Fb, Da, Db, S)
        Dtot = Da + Db

        if diis_obj is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp) if profile is not None else None
            F_base = F
            e = _fock_error_rhf(F, Dtot, S)
            diis_obj.push(F, e)
            F_try = None
            try:
                F_try = diis_obj.extrapolate()
            except Exception:
                F_try = None
            if F_try is None or not _is_finite_matrix(xp, F_try):
                F = F_base
                diis_obj = _DIIS(max_vec=diis_space)
                if profile is not None:
                    profile["scf"]["diis_fallbacks"] = int(profile["scf"].get("diis_fallbacks", 0)) + 1
            else:
                F = _symmetrize_inplace(xp, F_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp) if profile is not None else None
        e0, C = _gen_eigh_with_X(F, X)
        if profile is not None and t is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)

        Da_new = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ_a, nalpha))
        Db_new = _symmetrize_inplace(xp, _density_from_C_occ_syrk(xp, C, occ_b, nbeta))
        if damping:
            Da_new = (1.0 - lam) * Da_new + lam * Da
            Db_new = (1.0 - lam) * Db_new + lam * Db

        Dtot_new = Da_new + Db_new
        e_one = float(xp.sum(Dtot_new * h).item())
        e_coul = float(0.5 * xp.sum(Dtot_new * J).item())
        e_ex = float(0.5 * xp.sum(Da_new * Ka).item() + 0.5 * xp.sum(Db_new * Kb).item())
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
        profile["scf"]["iters"] = int(cycle)

    return SCFResult(
        method="ROHF",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=e0,
        mo_coeff=C,
        mo_occ=(occ_a, occ_b),
    )


__all__ = ["rhf_direct", "rohf_direct", "uhf_direct"]
