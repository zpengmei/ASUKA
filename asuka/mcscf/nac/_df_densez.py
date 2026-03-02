from __future__ import annotations

"""Dense-Z debug NAC driver (DF-only).

This mirrors :func:`asuka.mcscf.nac_df.sacasscf_nonadiabatic_couplings_df` with
``response_term="split_orbfd"`` but replaces the iterative Z-vector solve with a
dense SVD pseudo-inverse solve.

Use this for:
  - small systems / debugging convergence issues
  - multi-pair runs where factoring once is helpful
"""

from typing import Any, Literal, Sequence

import numpy as np

from asuka.integrals.int1e_cart import contract_dS_ip_cart, shell_to_atom_map
from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter
from asuka.mcscf import newton_casscf as _newton_casscf
from asuka.mcscf.state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from asuka.mcscf.zvector import build_mcscf_hessian_operator
from asuka.solver import GUGAFCISolver

from ..zvector_dense import DenseSVDLinearSolver

# Reuse the DF gradient/NAC building blocks from _df.py.
from ._df import (  # noqa: PLC0415
    _asnumpy_f64,
    _base_fcisolver_method,
    _eris_patch_active,
    _FixedRDMFcisolver,
    _Lorb_dot_dgorb_dx_df,
    _force_internal_newton,
    _grad_elec_active_df,
    _mol_coords_charges_bohr,
)


def sacasscf_nonadiabatic_couplings_df_densez(
    scf_out: Any,
    casscf: Any,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    atmlst: Sequence[int] | None = None,
    use_etfs: bool = False,
    mult_ediff: bool = False,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    z_rcond: float = 1e-12,
) -> np.ndarray:
    """Compute SA-CASSCF NACVs (<bra|d/dR|ket>) using DF integrals and dense Z-solve.

    Returns ``nac[bra, ket, atom, xyz]``.
    """

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have .mol")
    _is_sph_dz = not bool(getattr(mol, "cart", True))

    coords, _charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])

    if atmlst is None:
        atmlst_use = list(range(natm))
    else:
        atmlst_use = [int(a) for a in atmlst]

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    nocc = ncore + ncas

    ci_raw = getattr(casscf, "ci", None)
    nroots = int(getattr(casscf, "nroots", len(ci_raw) if isinstance(ci_raw, (list, tuple)) else 1))
    ci_list = ci_as_list(ci_raw, nroots=nroots)

    if nroots <= 1:
        return np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    weights_in = getattr(casscf, "root_weights", None)
    if weights_in is None:
        weights_in = getattr(casscf, "weights", None)
    weights = normalize_weights(weights_in, nroots=nroots)

    e_raw = getattr(casscf, "e_states", None)
    if e_raw is None:
        e_raw = getattr(casscf, "e_roots", None)
    if e_raw is None:
        raise ValueError("casscf must provide per-root energies as e_states or e_roots")
    e_states = np.asarray(e_raw, dtype=np.float64).ravel()
    if int(e_states.size) != nroots:
        raise ValueError("energy array length mismatch")

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver

    C_ref = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    if C_ref.ndim != 2:
        raise ValueError("casscf.mo_coeff must be 2D (nao,nmo)")
    nao, nmo = map(int, C_ref.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    B_ref = _asnumpy_f64(getattr(scf_out, "df_B"))
    hcore_ref = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))
    ao_basis_ref = getattr(scf_out, "ao_basis")

    # SA Newton adapter + Hessian operator (built once)
    mc_sa = DFNewtonCASSCFAdapter(
        df_B=B_ref,
        hcore_ao=hcore_ref,
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=C_ref,
        fcisolver=fcisolver_use,
        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
        frozen=getattr(casscf, "frozen", None),
        internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
        extrasym=getattr(casscf, "extrasym", None),
    )
    # Build the SA objective Hessian with *unpatched* ERIs (matches PySCF's SA objective).
    eris_sa = mc_sa.ao2mo(C_ref)
    with _force_internal_newton():
        hess_op = build_mcscf_hessian_operator(
            mc_sa,
            mo_coeff=C_ref,
            ci=ci_list,
            eris=eris_sa,
            use_newton_hessian=True,
        )

    dense_solver = DenseSVDLinearSolver.from_hessian_op(hess_op, rcond=float(z_rcond))

    # SA RDMs for orbital directional derivative term
    dm1_sa, dm2_sa = make_state_averaged_rdms(fcisolver_use, ci_list, weights, ncas=int(ncas), nelecas=nelecas)

    if pairs is None:
        pair_list = [(ket, bra) for ket in range(nroots) for bra in range(nroots) if ket != bra]
    else:
        pair_list = [(int(ket), int(bra)) for (ket, bra) in pairs if int(ket) != int(bra)]

    nac = np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    shell_atom_ref = shell_to_atom_map(ao_basis_ref, atom_coords_bohr=coords)

    for ket, bra in pair_list:
        ket = int(ket)
        bra = int(bra)
        if ket == bra:
            continue

        ediff = float(e_states[bra] - e_states[ket])

        # Transition densities (bra,ket)
        trans_rdm12 = _base_fcisolver_method(fcisolver_use, "trans_rdm12")
        dm1_t, dm2_t = trans_rdm12(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
        dm1_t = 0.5 * (np.asarray(dm1_t, dtype=np.float64) + np.asarray(dm1_t, dtype=np.float64).T)
        dm2_t = 0.5 * (
            np.asarray(dm2_t, dtype=np.float64) + np.asarray(dm2_t, dtype=np.float64).transpose(1, 0, 3, 2)
        )

        # Hamiltonian derivative term (<bra|dH/dR|ket> without nuclear term)
        ham = _grad_elec_active_df(
            scf_out=scf_out,
            mo_coeff=C_ref,
            dm1_act=dm1_t,
            dm2_act=dm2_t,
            ncore=int(ncore),
            ncas=int(ncas),
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
        )

        # AO-overlap (CSF) term in numerator form
        if not bool(use_etfs):
            trans_rdm1 = _base_fcisolver_method(fcisolver_use, "trans_rdm1")
            dm1 = trans_rdm1(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
            castm1 = np.asarray(dm1, dtype=np.float64).T - np.asarray(dm1, dtype=np.float64)
            mo_cas = C_ref[:, ncore:nocc]
            tm1 = mo_cas @ castm1 @ mo_cas.T
            if _is_sph_dz:
                from asuka.integrals.int1e_sph import contract_dS_ip_sph  # noqa: PLC0415
                nac_csf = 0.5 * contract_dS_ip_sph(
                    ao_basis_ref,
                    atom_coords_bohr=coords,
                    M_sph=tm1,
                    shell_atom=shell_atom_ref,
                )
            else:
                nac_csf = 0.5 * contract_dS_ip_cart(
                    ao_basis_ref,
                    atom_coords_bohr=coords,
                    M=tm1,
                    shell_atom=shell_atom_ref,
                )
            ham = np.asarray(ham, dtype=np.float64) + np.asarray(nac_csf * ediff, dtype=np.float64)

        # Pair-specific RHS for Z-vector
        fcisolver_fixed = _FixedRDMFcisolver(fcisolver_use, dm1=dm1_t, dm2=dm2_t)
        mc_trans = DFNewtonCASSCFAdapter(
            df_B=B_ref,
            hcore_ao=hcore_ref,
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            mo_coeff=C_ref,
            fcisolver=fcisolver_fixed,
            frozen=getattr(casscf, "frozen", None),
            internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
            extrasym=getattr(casscf, "extrasym", None),
        )
        # Match PySCF NAC RHS construction: remove core-orbital contributions from the Newton operator.
        eris_act = _eris_patch_active(eris_sa, mo_coeff=C_ref, hcore_ao=hcore_ref, ncore=int(ncore))

        g_ket, *_ = _newton_casscf.gen_g_hop(
            mc_trans, C_ref, ci_list[ket], eris_act, verbose=0, implementation="internal"
        )
        g_bra, *_ = _newton_casscf.gen_g_hop(
            mc_trans, C_ref, ci_list[bra], eris_act, verbose=0, implementation="internal"
        )
        g_ket = np.asarray(g_ket, dtype=np.float64).ravel()
        g_bra = np.asarray(g_bra, dtype=np.float64).ravel()

        n_orb = int(hess_op.n_orb)
        g_orb = g_ket[:n_orb]

        g_ci_bra = 0.5 * g_ket[n_orb:].copy()
        g_ci_ket = 0.5 * g_bra[n_orb:].copy()

        ndet_ket = int(np.asarray(ci_list[ket]).size)
        ndet_bra = int(np.asarray(ci_list[bra]).size)
        if ndet_ket == ndet_bra:
            ket2bra = float(np.dot(np.asarray(ci_list[bra], dtype=np.float64).ravel(), g_ci_ket))
            bra2ket = float(np.dot(np.asarray(ci_list[ket], dtype=np.float64).ravel(), g_ci_bra))
            g_ci_ket = g_ci_ket - ket2bra * np.asarray(ci_list[bra], dtype=np.float64).ravel()
            g_ci_bra = g_ci_bra - bra2ket * np.asarray(ci_list[ket], dtype=np.float64).ravel()

        rhs_ci_list = [np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()) for r in range(nroots)]
        rhs_ci_list[ket] = g_ci_ket[:ndet_ket]
        rhs_ci_list[bra] = g_ci_bra[:ndet_bra]

        # Dense Z solve: A L = -rhs
        Lvec = dense_solver.solve_rhs(rhs_orb=np.asarray(g_orb, dtype=np.float64), rhs_ci_list=rhs_ci_list)

        # Split response: Lci_dot_dgci + Lorb_dot_dgorb (analytic DF orbital Lagrange term)
        Lorb_mat = mc_sa.unpack_uniq_var(Lvec[:n_orb])
        Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])
        if not isinstance(Lci_list, list) or len(Lci_list) != int(nroots):
            raise RuntimeError("unexpected CI unpack structure in dense Z solution")

        # CI response via weighted transition RDMs between Lci[root] and ci[root]
        dm1_lci = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
        dm2_lci = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        for r in range(int(nroots)):
            wr = float(w_arr[r])
            if abs(wr) < 1e-14:
                continue
            dm1_r, dm2_r = trans_rdm12(
                fcisolver_use,
                np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                np.asarray(ci_list[r], dtype=np.float64).ravel(),
                int(ncas),
                nelecas,
            )
            dm1_r = np.asarray(dm1_r, dtype=np.float64)
            dm2_r = np.asarray(dm2_r, dtype=np.float64)
            dm1_lci += wr * (dm1_r + dm1_r.T)
            dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))

        de_lci = _grad_elec_active_df(
            scf_out=scf_out,
            mo_coeff=C_ref,
            dm1_act=dm1_lci,
            dm2_act=dm2_lci,
            ncore=int(ncore),
            ncas=int(ncas),
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
        )

        # Orbital response: analytic DF analogue of PySCF's Lorb_dot_dgorb_dx.
        de_lorb = _Lorb_dot_dgorb_dx_df(
            scf_out=scf_out,
            mo_coeff=C_ref,
            dm1_act=dm1_sa,
            dm2_act=dm2_sa,
            Lorb=np.asarray(Lorb_mat, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            eris=eris_sa,
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
        )

        resp_full = np.asarray(de_lci, dtype=np.float64) + np.asarray(de_lorb, dtype=np.float64)

        nac_num = (
            np.asarray(ham, dtype=np.float64)[np.asarray(atmlst_use, dtype=np.int32)]
            + np.asarray(resp_full, dtype=np.float64)[np.asarray(atmlst_use, dtype=np.int32)]
        )
        if not bool(mult_ediff):
            if abs(ediff) < 1e-12:
                raise ZeroDivisionError("E_bra - E_ket too small; use mult_ediff=True for numerator mode")
            nac_num = nac_num / ediff

        nac[bra, ket] = np.asarray(nac_num, dtype=np.float64)

    return nac


__all__ = ["sacasscf_nonadiabatic_couplings_df_densez"]
