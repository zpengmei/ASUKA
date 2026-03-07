from __future__ import annotations

"""SCF driver using local-THC J/K backend."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .df_scf import SCFResult
from .local_thc_factors import LocalTHCFactors
from .local_thc_jk import local_thc_J, local_thc_JK, local_thc_K_blocked

from . import df_scf as _df  # reuse orthogonalization/DIIS helpers


@dataclass(frozen=True)
class LocalTHCReferenceRHF:
    D_ref: Any
    J_ref: Any
    K_ref: Any


@dataclass(frozen=True)
class LocalTHCReferenceUHF:
    Da_ref: Any
    Db_ref: Any
    J_ref: Any
    Ka_ref: Any
    Kb_ref: Any


def rhf_local_thc(
    S,
    hcore,
    lthc: LocalTHCFactors,
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
    dm0=None,
    mo_coeff0=None,
    init_fock_cycles: int = 0,
    reference: LocalTHCReferenceRHF | None = None,
    profile: dict | None = None,
    xc_spec=None,
    xc_grid_coords=None,
    xc_grid_weights=None,
    xc_ao_basis=None,
    xc_sph_transform=None,
    xc_batch_size: int = 50000,
) -> SCFResult:
    """RHF SCF with local-THC J/K backend."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local-THC SCF requires CuPy") from e

    xp = cp
    S = _df._as_xp(xp, S, dtype=xp.float64)
    h = _df._as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with S/hcore")

    X = _df._orthogonalizer_from_S(S)

    occ_np, _nocc = _df._occ_rhf(nelec, nao)
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

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("init_fock_ms", 0.0)
        prof.setdefault("init_fock_cycles", int(init_fock_cycles))
        prof.setdefault("init_fock_applied", False)
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
            J_thc, K_thc = local_thc_JK(dD, lthc, q_block=int(q_block))
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
        J_thc, K_thc = local_thc_JK(dD, lthc, q_block=int(q_block))
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
        method="RKS-LTHC" if xc_spec is not None else "RHF-LTHC",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=eps,
        mo_coeff=C,
        mo_occ=occ,
    )

def uhf_local_thc(
    S,
    hcore,
    lthc: LocalTHCFactors,
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
    dm0=None,
    mo_coeff0=None,
    reference: LocalTHCReferenceUHF | None = None,
    profile: dict | None = None,
) -> SCFResult:
    """UHF SCF with local-THC J/K backend."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local-THC SCF requires CuPy") from e

    xp = cp
    S = _df._as_xp(xp, S, dtype=xp.float64)
    h = _df._as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with S/hcore")

    X = _df._orthogonalizer_from_S(S)

    occ_a_np, occ_b_np = _df._occ_uhf(int(nalpha), int(nbeta), nao)
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

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("iters", 0)

    for cycle in range(1, int(max_cycle) + 1):
        Dtot = Da + Db
        Dtot_ref = Da_ref + Db_ref

        t = _df._time_ms_start(xp)
        J = J_ref + local_thc_J(Dtot - Dtot_ref, lthc)
        Ka = Ka_ref + local_thc_K_blocked(Da - Da_ref, lthc, q_block=int(q_block))
        Kb = Kb_ref + local_thc_K_blocked(Db - Db_ref, lthc, q_block=int(q_block))
        if profile is not None:
            profile["scf"]["jk_ms"] += _df._time_ms_end(xp, t)

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
        method="UHF-LTHC",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        mo_occ=(occ_a, occ_b),
    )


def rohf_local_thc(
    S,
    hcore,
    lthc: LocalTHCFactors,
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
    dm0=None,
    mo_coeff0=None,
    reference: LocalTHCReferenceUHF | None = None,
    profile: dict | None = None,
) -> SCFResult:
    """ROHF SCF with local-THC J/K backend."""

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("local-THC SCF requires CuPy") from e

    xp = cp
    S = _df._as_xp(xp, S, dtype=xp.float64)
    h = _df._as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")
    if int(lthc.nao) != int(nao):
        raise ValueError("lthc.nao mismatch with S/hcore")

    X = _df._orthogonalizer_from_S(S)

    if int(nalpha) < 0 or int(nbeta) < 0 or int(nalpha) < int(nbeta):
        raise ValueError("ROHF requires nalpha>=nbeta>=0")
    if int(nalpha) > int(nao) or int(nbeta) > int(nao):
        raise ValueError("nalpha/nbeta exceeds number of orbitals")

    occ_a_np, occ_b_np = _df._occ_uhf(int(nalpha), int(nbeta), nao)
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

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("iters", 0)

    for cycle in range(1, int(max_cycle) + 1):
        Dtot = Da + Db
        Dtot_ref = Da_ref + Db_ref

        t = _df._time_ms_start(xp)
        J = J_ref + local_thc_J(Dtot - Dtot_ref, lthc)
        Ka = Ka_ref + local_thc_K_blocked(Da - Da_ref, lthc, q_block=int(q_block))
        Kb = Kb_ref + local_thc_K_blocked(Db - Db_ref, lthc, q_block=int(q_block))
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
        method="ROHF-LTHC",
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
    "LocalTHCReferenceRHF",
    "LocalTHCReferenceUHF",
    "rhf_local_thc",
    "rohf_local_thc",
    "uhf_local_thc",
]
