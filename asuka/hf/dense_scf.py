from __future__ import annotations

"""Self-contained dense-AO-ERI SCF (RHF/UHF/ROHF) drivers."""

from . import dense_jk
from .df_scf import (
    SCFResult,
    _DIIS,
    _as_xp,
    _density_from_C_occ,
    _fock_error_rhf,
    _gen_eigh_with_X,
    _get_xp,
    _occ_rhf,
    _occ_uhf,
    _orthogonalizer_from_S,
    _roothaan_fock_rohf,
    _symmetrize,
    _time_ms_end,
    _time_ms_start,
)


def _validate_eri_mat_shape(eri_mat, nao: int) -> None:
    n2 = int(nao) * int(nao)
    if getattr(eri_mat, "ndim", None) != 2:
        raise ValueError("eri_mat must be a 2D square matrix")
    if tuple(map(int, eri_mat.shape)) != (n2, n2):
        raise ValueError(f"eri_mat must have shape ({n2},{n2}), got {tuple(map(int, eri_mat.shape))}")


def _is_finite_matrix(xp, A) -> bool:
    try:
        return bool(xp.all(xp.isfinite(A)))
    except Exception:
        return False


def rhf_dense(
    S,
    hcore,
    eri_mat,
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
    """RHF SCF with dense AO ERIs in ordered-pair matrix form."""

    xp, _is_gpu = _get_xp(S, hcore, eri_mat)
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    eri_mat = _as_xp(xp, eri_mat, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    _validate_eri_mat_shape(eri_mat, nao)

    X = _orthogonalizer_from_S(S)
    occ_np, nocc = _occ_rhf(nelec, nao)
    occ = _as_xp(xp, occ_np, dtype=xp.float64)

    eps, C = _gen_eigh_with_X(h, X)
    if dm0 is not None:
        D = _as_xp(xp, dm0, dtype=xp.float64)
        if D.shape != (nao, nao):
            raise ValueError("dm0 must have shape (nao, nao)")
        D = _symmetrize(xp, D)
    elif mo_coeff0 is not None:
        C0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 must have shape (nao, nao)")
        C = C0
        D = _symmetrize(xp, _density_from_C_occ(C, occ))
    else:
        D = _symmetrize(xp, _density_from_C_occ(C, occ))

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
            J, K = dense_jk.dense_JK_from_eri_mat_D(eri_mat, D_prev, want_J=True, want_K=True)
            F = _symmetrize(xp, h + J - 0.5 * K)
            if level_shift:
                shift = float(level_shift)
                if shift != 0.0:
                    Fp = X.T @ F @ X
                    Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                    F = X @ Fp @ X.T
            eps, C = _gen_eigh_with_X(F, X)
            D = _symmetrize(xp, _density_from_C_occ(C, occ))
            if damping:
                D = (1.0 - lam) * D + lam * D_prev
            if profile is not None and t_init is not None:
                profile["scf"]["init_fock_ms"] += _time_ms_end(xp, t_init)

    cycle = 0
    for cycle in range(1, int(max_cycle) + 1):
        t = _time_ms_start(xp) if profile is not None else None
        J, K = dense_jk.dense_JK_from_eri_mat_D(eri_mat, D, want_J=True, want_K=True)
        if profile is not None and t is not None:
            profile["jk"]["calls"] = int(profile["jk"].get("calls", 0)) + 1
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        F = _symmetrize(xp, h + J - 0.5 * K)
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
                F = _symmetrize(xp, F_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp) if profile is not None else None
        eps, C = _gen_eigh_with_X(F, X)
        if profile is not None and t is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)

        D_new = _symmetrize(xp, _density_from_C_occ(C, occ))
        if damping:
            D_new = (1.0 - lam) * D_new + lam * D

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


def uhf_dense(
    S,
    hcore,
    eri_mat,
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
    """UHF SCF with dense AO ERIs in ordered-pair matrix form."""

    xp, _is_gpu = _get_xp(S, hcore, eri_mat)
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    eri_mat = _as_xp(xp, eri_mat, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    _validate_eri_mat_shape(eri_mat, nao)

    X = _orthogonalizer_from_S(S)
    occ_a_np, occ_b_np = _occ_uhf(nalpha, nbeta, nao)
    occ_a = _as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _as_xp(xp, occ_b_np, dtype=xp.float64)

    _e0, C = _gen_eigh_with_X(h, X)
    ea = _e0
    eb = _e0
    Ca = C
    Cb = C
    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for UHF must be a (Da, Db) tuple")
        Da = _symmetrize(xp, _as_xp(xp, dm0[0], dtype=xp.float64))
        Db = _symmetrize(xp, _as_xp(xp, dm0[1], dtype=xp.float64))
        if Da.shape != (nao, nao) or Db.shape != (nao, nao):
            raise ValueError("dm0 for UHF must have shape (nao, nao) for both spins")
    elif mo_coeff0 is not None:
        if isinstance(mo_coeff0, (tuple, list)):
            if len(mo_coeff0) != 2:
                raise TypeError("mo_coeff0 for UHF must be a (Ca, Cb) tuple")
            Ca0 = _as_xp(xp, mo_coeff0[0], dtype=xp.float64)
            Cb0 = _as_xp(xp, mo_coeff0[1], dtype=xp.float64)
        else:
            Ca0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
            Cb0 = Ca0
        if Ca0.shape != (nao, nao) or Cb0.shape != (nao, nao):
            raise ValueError("mo_coeff0 for UHF must have shape (nao, nao) for both spins")
        Ca = Ca0
        Cb = Cb0
        Da = _symmetrize(xp, _density_from_C_occ(Ca, occ_a))
        Db = _symmetrize(xp, _density_from_C_occ(Cb, occ_b))
    else:
        Da = _symmetrize(xp, _density_from_C_occ(Ca, occ_a))
        Db = _symmetrize(xp, _density_from_C_occ(Cb, occ_b))

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
        Dtot = Da + Db
        t = _time_ms_start(xp) if profile is not None else None
        J = dense_jk.dense_J_from_eri_mat_D(eri_mat, Dtot)
        Ka = dense_jk.dense_K_from_eri_mat_D(eri_mat, Da)
        Kb = dense_jk.dense_K_from_eri_mat_D(eri_mat, Db)
        if profile is not None and t is not None:
            profile["jk"]["calls"] = int(profile["jk"].get("calls", 0)) + 1
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        Fa = _symmetrize(xp, h + J - Ka)
        Fb = _symmetrize(xp, h + J - Kb)

        if diis_a is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp) if profile is not None else None
            Fa_base = Fa
            ea_mat = _fock_error_rhf(Fa, Da, S)
            diis_a.push(Fa, ea_mat)
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
                Fa = _symmetrize(xp, Fa_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)
        if diis_b is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp) if profile is not None else None
            Fb_base = Fb
            eb_mat = _fock_error_rhf(Fb, Db, S)
            diis_b.push(Fb, eb_mat)
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
                Fb = _symmetrize(xp, Fb_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp) if profile is not None else None
        ea, Ca = _gen_eigh_with_X(Fa, X)
        eb, Cb = _gen_eigh_with_X(Fb, X)
        if profile is not None and t is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)

        Da_new = _symmetrize(xp, _density_from_C_occ(Ca, occ_a))
        Db_new = _symmetrize(xp, _density_from_C_occ(Cb, occ_b))
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


def rohf_dense(
    S,
    hcore,
    eri_mat,
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
    """ROHF SCF with dense AO ERIs in ordered-pair matrix form."""

    xp, _is_gpu = _get_xp(S, hcore, eri_mat)
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    eri_mat = _as_xp(xp, eri_mat, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    _validate_eri_mat_shape(eri_mat, nao)

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
        Da = _symmetrize(xp, _as_xp(xp, dm0[0], dtype=xp.float64))
        Db = _symmetrize(xp, _as_xp(xp, dm0[1], dtype=xp.float64))
        if Da.shape != (nao, nao) or Db.shape != (nao, nao):
            raise ValueError("dm0 for ROHF must have shape (nao, nao) for both spins")
    elif mo_coeff0 is not None:
        C0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 for ROHF must have shape (nao, nao)")
        C = C0
        Da = _symmetrize(xp, _density_from_C_occ(C, occ_a))
        Db = _symmetrize(xp, _density_from_C_occ(C, occ_b))
    else:
        Da = _symmetrize(xp, _density_from_C_occ(C, occ_a))
        Db = _symmetrize(xp, _density_from_C_occ(C, occ_b))

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
        Dtot = Da + Db
        t = _time_ms_start(xp) if profile is not None else None
        J = dense_jk.dense_J_from_eri_mat_D(eri_mat, Dtot)
        Ka = dense_jk.dense_K_from_eri_mat_D(eri_mat, Da)
        Kb = dense_jk.dense_K_from_eri_mat_D(eri_mat, Db)
        if profile is not None and t is not None:
            profile["jk"]["calls"] = int(profile["jk"].get("calls", 0)) + 1
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        Fa = _symmetrize(xp, h + J - Ka)
        Fb = _symmetrize(xp, h + J - Kb)
        F = _roothaan_fock_rohf(Fa, Fb, Da, Db, S)

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
                F = _symmetrize(xp, F_try)
            if profile is not None and t is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp) if profile is not None else None
        e0, C = _gen_eigh_with_X(F, X)
        if profile is not None and t is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)

        Da_new = _symmetrize(xp, _density_from_C_occ(C, occ_a))
        Db_new = _symmetrize(xp, _density_from_C_occ(C, occ_b))
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


__all__ = ["rhf_dense", "rohf_dense", "uhf_dense"]
