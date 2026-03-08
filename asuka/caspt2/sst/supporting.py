"""Supporting subspace correction for SST-CASPT2.

Computes E^eff_S (Eq. 65 of Song 2024), the supporting subspace energy
correction:

    E^eff_S = ⟨h_S | G_apx⁻¹ | h_S⟩

where h_S is the RHS vector projected to the supporting subspace, and
G_apx⁻¹ is the inverse of the Kronecker sum approximation.

The supporting subspace consists of IC cases {A(1), B±(2-3), C(4), D(5),
F±(8-9)} — those where the Kronecker sum G_apx differs from the exact
Dyall H0 due to active-space inter-site interactions.

The supporting correction cancels the overcounting in E^MP2_dressed for
these cases: the dressed MP2 energy includes contributions that are
incorrect for the supporting subspace functions, and E^eff_S subtracts
this error.
"""
from __future__ import annotations

import numpy as np

from asuka.caspt2.energy import _get_external_energies
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.superindex import SuperindexMap

from asuka.caspt2.sst.coupling import DressedOrbitals
from asuka.caspt2.sst.kronecker import (
    apply_g_apx_inv_dressed,
    build_g_apx_diagonal,
)
from asuka.caspt2.sst.subspaces import SUPPORT_CASES, CASE_NAMES
from asuka.caspt2.sst.types import SSTConfig, SSTInput

__all__ = ["solve_supporting_subspace"]


from asuka.caspt2.sst.reduced_system import _build_full_mo_eris  # noqa: F401


def solve_supporting_subspace(
    inp: SSTInput,
    dressed: DressedOrbitals,
    cfg: SSTConfig,
) -> tuple[float, dict[int, np.ndarray]]:
    """Compute the supporting subspace correction E^eff_S.

    Builds and solves the supporting subspace equations for IC cases
    {A, B±, C, D, F±}. The correction is:

        E^eff_S = Σ_case ⟨v_case | G_apx⁻¹ | v_case⟩

    where v_case is the transformed RHS for each supporting case.

    Parameters
    ----------
    inp : SSTInput
    dressed : DressedOrbitals
    cfg : SSTConfig

    Returns
    -------
    e_eff_s : float
        Supporting subspace energy correction.
    s_amplitudes : dict[int, ndarray]
        Per-case amplitude vectors (s = G_apx⁻¹ h) in the SR basis.
    """
    smap: SuperindexMap = inp.smap
    fock: CASPT2Fock = inp.fock
    dm1 = np.asarray(inp.dm1_act, dtype=np.float64)
    dm2 = np.asarray(inp.dm2_act, dtype=np.float64)
    dm3 = np.asarray(inp.dm3_act, dtype=np.float64)
    ci_context: CASPT2CIContext = inp.ci_context
    verbose = cfg.verbose

    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)

    # Build full MO ERIs for RHS construction
    if inp.eri_mo is not None:
        eri_mo = np.asarray(inp.eri_mo, dtype=np.float64)
    elif inp.B_ao is not None:
        eri_mo = _build_full_mo_eris(inp.B_ao, inp.mo_coeff)
    else:
        raise ValueError("supporting subspace requires either eri_mo or B_ao")

    if verbose >= 1:
        print(f"SST supporting subspace (cases {sorted(SUPPORT_CASES)})")

    e_eff_s = 0.0
    s_amplitudes: dict[int, np.ndarray] = {}

    for case in sorted(SUPPORT_CASES):
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])

        if nasup == 0 or nisup == 0:
            s_amplitudes[case] = np.empty(0, dtype=np.float64)
            if verbose >= 2:
                print(f"  Case {case} ({CASE_NAMES[case]}): skipped (empty)")
            continue

        # Build S and B matrices (active superindex)
        smat = build_smat(case, smap, dm1, dm2, dm3)
        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)

        # Joint S/B diagonalization with linear-dep removal
        decomp = sbdiag(smat, bmat,
                        threshold_norm=cfg.threshold,
                        threshold_s=cfg.threshold_s)

        if decomp.nindep == 0:
            s_amplitudes[case] = np.empty(0, dtype=np.float64)
            if verbose >= 2:
                print(f"  Case {case} ({CASE_NAMES[case]}): all lin-dep removed")
            continue

        # External energies (from standard Fock diagonal)
        ext_energies = _get_external_energies(case, smap, fock)

        # Build the exact H0 diagonal (for reference/comparison)
        if ext_energies.size > 0:
            h0_exact = (decomp.b_diag[:, None] + ext_energies[None, :]).ravel()
        else:
            h0_exact = decomp.b_diag.copy()

        # Build G_apx diagonal for this case
        g_apx_diag = build_g_apx_diagonal(
            case, dressed, nish, nash, nssh,
            decomp.b_diag, ext_energies,
        )

        # Build RHS vector
        rhs_raw = build_rhs(case, smap, fock, eri_mo, dm1, dm2)

        # Transform RHS to SR basis: v_SR = T^T S V
        rhs_mat = rhs_raw.reshape(nasup, nisup)
        rhs_sr = (decomp.transform.T @ smat @ rhs_mat).ravel()

        # Apply G_apx⁻¹ to get supporting amplitude: s = G_apx⁻¹ v
        s_case = apply_g_apx_inv_dressed(rhs_sr, g_apx_diag)
        s_amplitudes[case] = s_case

        # E^eff_S contribution: ⟨v | G_apx⁻¹ | v⟩ = ⟨v | s⟩
        e_case = float(np.dot(rhs_sr, s_case))
        e_eff_s += e_case

        if verbose >= 1:
            print(f"  Case {case} ({CASE_NAMES[case]}): "
                  f"nindep={decomp.nindep}, E^eff_S = {e_case:.10f}")

    if verbose >= 1:
        print(f"  E^eff_S total = {e_eff_s:.10f}")

    return float(e_eff_s), s_amplitudes
