"""Z-vector (CP-MCSCF) solver for CASPT2 response."""
from __future__ import annotations

from typing import Any

import numpy as np
import os

from asuka.caspt2.gradient.debug_utils import (
    _apply_debug_zorb_block_signs,
    _resolve_response_dpt2_mode,
)


def _solve_zvector(
    scf_out, casscf, lagrangians,
    ncore, ncas, nelecas, twos,
    B_ao, h_ao, C, ci_raw,
    z_tol=1e-10, z_maxiter=200,
    project_ci_tangent: bool = False,
    return_meta: bool = False,
    dump_vectors: bool = False,
    verbose=0,
):
    """Solve CP-MCSCF (Z-vector) equations for CASPT2 response."""
    from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter
    from asuka.mcscf.zvector import (
        build_mcscf_hessian_operator,
        prepare_ci_rhs_for_zvector,
        project_ci_rhs_normalized,
        solve_mcscf_zvector,
    )
    from asuka.mcscf.state_average import ci_as_list
    from asuka.solver import GUGAFCISolver

    nroots = int(getattr(casscf, "nroots", 1))
    ci_list = ci_as_list(ci_raw, nroots=nroots)

    fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))

    mc = DFNewtonCASSCFAdapter(
        df_B=B_ao,
        hcore_ao=h_ao,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=C,
        fcisolver=fcisolver,
        weights=[1.0 / nroots] * nroots,
        frozen=getattr(casscf, "frozen", None),
        internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
    )
    eris = mc.ao2mo(C)

    _hess_impl = str(os.environ.get("ASUKA_CASPT2_ZVEC_HESS_IMPL", "internal")).strip().lower()
    if _hess_impl not in {"internal", "auto"}:
        _hess_impl = "internal"
    if _hess_impl == "internal":
        _force_internal = os.environ.get("CUGUGA_NEWTON_CASSCF", "")
        os.environ["CUGUGA_NEWTON_CASSCF"] = "internal"
        os.environ["CUGUGA_NEWTON_CASSCF_IMPL"] = "internal"
        try:
            hess_op = build_mcscf_hessian_operator(
                mc, mo_coeff=C, ci=ci_list, eris=eris,
                use_newton_hessian=True,
            )
        finally:
            if _force_internal:
                os.environ["CUGUGA_NEWTON_CASSCF"] = _force_internal
            else:
                os.environ.pop("CUGUGA_NEWTON_CASSCF", None)
            os.environ.pop("CUGUGA_NEWTON_CASSCF_IMPL", None)
    else:
        hess_op = build_mcscf_hessian_operator(
            mc, mo_coeff=C, ci=ci_list, eris=eris,
            use_newton_hessian=True,
        )

    n_orb = int(hess_op.n_orb)

    # Build Z-vector RHS from OLag (orbital part)
    # OLag is antisymmetrized: OLag[p,q] = -OLag[q,p]
    # The Z-vector RHS is the OLag packed into independent orbital rotation pairs
    _resp_dpt2_mode_req = os.environ.get("ASUKA_RESPONSE_DPT2_DENSITY", "full")
    _resp_dpt2_mode = _resolve_response_dpt2_mode(_resp_dpt2_mode_req)
    if str(_resp_dpt2_mode) == "bare" and "olag_bare" in lagrangians:
        olag = np.asarray(lagrangians["olag_bare"], dtype=np.float64)
        olag = np.asarray(olag - olag.T, dtype=np.float64)
    else:
        olag = lagrangians["olag"]
    rhs_orb = np.asarray(mc.pack_uniq_var(olag), dtype=np.float64).ravel()

    # CI RHS from CLag.
    # For strict Molcas parity lane we enforce the normalized-CI tangent gauge
    # before the solve; Molcas RESP CIT is tangent-dominated in this gauge.
    clag = lagrangians["clag"]
    rhs_ci_raw = np.asarray(clag, dtype=np.float64).copy()
    if bool(project_ci_tangent):
        if nroots == 1:
            rhs_ci = prepare_ci_rhs_for_zvector(
                ci0=np.asarray(ci_list[0], dtype=np.float64),
                rhs_ci=rhs_ci_raw,
                project_normalized=True,
            )
        else:
            rhs_ci = prepare_ci_rhs_for_zvector(
                ci0=[np.asarray(c, dtype=np.float64) for c in ci_list],
                rhs_ci=[np.asarray(v, dtype=np.float64) for v in rhs_ci_raw],
                project_normalized=True,
            )
    else:
        rhs_ci = rhs_ci_raw

    hess_use = hess_op

    z = solve_mcscf_zvector(
        mc,
        rhs_orb=rhs_orb,
        rhs_ci=rhs_ci,
        hessian_op=hess_use,
        tol=float(z_tol),
        maxiter=int(z_maxiter),
    )

    if verbose >= 1:
        print(f"[CASPT2 grad] Z-vector: converged={z.converged} niter={z.niter} |r|={z.residual_norm:.2e}")

    z_orb = mc.unpack_uniq_var(z.z_packed[:n_orb])
    z_ci = hess_op.ci_unflatten(z.z_packed[n_orb:])
    if bool(project_ci_tangent):
        if isinstance(z_ci, list):
            z_ci = [
                project_ci_rhs_normalized(np.asarray(ci_list[i], dtype=np.float64), np.asarray(z_ci[i], dtype=np.float64))
                for i in range(len(z_ci))
            ]
        else:
            z_ci = project_ci_rhs_normalized(np.asarray(ci_list[0], dtype=np.float64), np.asarray(z_ci, dtype=np.float64))
    z_orb, _zorb_sign_meta = _apply_debug_zorb_block_signs(
        np.asarray(z_orb, dtype=np.float64),
        ncore=int(ncore),
        ncas=int(ncas),
    )

    if bool(return_meta):
        meta = {
            "converged": bool(getattr(z, "converged", False)),
            "niter": int(getattr(z, "niter", -1)),
            "residual_norm": float(getattr(z, "residual_norm", np.nan)),
            "hess_impl": str(_hess_impl),
            "response_dpt2_mode_req": str(_resp_dpt2_mode_req),
            "response_dpt2_mode": str(_resp_dpt2_mode),
            "z_orb_override_applied": False,
            "z_ci_override_applied": False,
            "z_orb_flip_d_sign": bool(_zorb_sign_meta.get("flip_d", False)),
            "z_orb_flip_c_sign": bool(_zorb_sign_meta.get("flip_c", False)),
        }
        if bool(dump_vectors):
            try:
                meta["rhs_orb_packed"] = np.asarray(rhs_orb, dtype=np.float64).ravel()
                meta["rhs_ci_raw_packed"] = np.asarray(rhs_ci_raw, dtype=np.float64).ravel()
                meta["rhs_ci_packed"] = np.asarray(rhs_ci, dtype=np.float64).ravel()
                meta["z_orb_packed"] = np.asarray(mc.pack_uniq_var(np.asarray(z_orb, dtype=np.float64)), dtype=np.float64).ravel()
                if isinstance(z_ci, list):
                    meta["z_ci_packed"] = np.concatenate([np.asarray(v, dtype=np.float64).ravel() for v in z_ci])
                else:
                    meta["z_ci_packed"] = np.asarray(z_ci, dtype=np.float64).ravel()
                if nroots == 1:
                    c0 = np.asarray(ci_list[0], dtype=np.float64).ravel()
                    zc = np.asarray(meta["z_ci_packed"], dtype=np.float64).ravel()
                    rc = np.asarray(meta["rhs_ci_packed"], dtype=np.float64).ravel()
                    den = float(np.dot(c0, c0))
                    if den > 0.0:
                        meta["z_ci_alpha"] = float(np.dot(zc, c0) / den)
                        meta["rhs_ci_alpha"] = float(np.dot(rc, c0) / den)
            except Exception:
                pass
        return z_orb, z_ci, meta
    return z_orb, z_ci
