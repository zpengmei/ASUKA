"""SS-CASPT2 energy driver.

Ports OpenMolcas ``eqctl2.f`` equation control logic.
Orchestrates S/B construction, diagonalization, RHS computation,
and amplitude solution for single-state CASPT2.
"""

from __future__ import annotations

import numpy as np

from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat, compute_e0
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.result import CASPT2EnergyResult
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.shifts import (
    apply_imaginary_shift,
    apply_real_shift,
    compute_shift_correction,
)
from asuka.caspt2.sigma import SigmaC1ActiveVirtualCoupling, SigmaC1CaseCoupling
from asuka.caspt2.solver import pcg_solve, pcg_solve_iterative
from asuka.caspt2.superindex import SuperindexMap

# Case names for reporting
_CASE_NAMES = [
    "A(VJTU)", "B+(VJTIP)", "B-(VJTIM)", "C(ATVX)", "D(AIVX)",
    "E+(VJAIP)", "E-(VJAIM)", "F+(BVATP)", "F-(BVATM)",
    "G+(BJATQ)", "G-(BJATM)", "H+(BJAIP)", "H-(BJAIM)",
]


def caspt2_energy_ss(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    e_ref: float,
    ci_context: CASPT2CIContext | None = None,
    *,
    pt2_backend: str = "cpu",
    cuda_device: int | None = None,
    cuda_mode: str = "hybrid",
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    df_blocks: object | None = None,
    store_rhs: bool = False,
    store_row_dots: bool = False,
    ipea_shift: float = 0.0,
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    mixed_precision_rhs: bool = False,
    verbose: int = 0,
) -> CASPT2EnergyResult:
    """Compute SS-CASPT2 energy for a single reference state.

    Orchestrates the full per-case workflow: S/B matrix construction,
    joint diagonalization with linear-dependence removal, RHS vector
    transformation, optional level shifts, and amplitude solution via
    direct divide or iterative PCG.

    When ``pt2_backend="cuda"``, dispatches to ``caspt2_energy_ss_cuda()``
    which uses DF-based RHS construction and a GPU sigma operator.

    Parameters
    ----------
    smap : SuperindexMap
        Precomputed superindex mappings (from ``build_superindex``).
    fock : CASPT2Fock
        MO-basis Fock matrices.
    eri_mo : (nmo, nmo, nmo, nmo)
        Full MO ERIs in chemists' notation.  Unused when ``pt2_backend="cuda"``
        (DF blocks are used instead).
    dm1, dm2, dm3 : np.ndarray
        Active-space RDMs in E-operator convention.
    e_ref : float
        Reference (CASSCF) energy for this state.
    ci_context : CASPT2CIContext | None
        DRT + CI vector needed for F3 contractions (cases A/C).
    df_blocks : CASPT2DFBlocks | None
        DF pair blocks for CUDA backend.
    store_rhs : bool
        If True, store raw RHS vectors in the result breakdown dict.
    store_row_dots : bool
        If True, store per-case row_dots matrices (needed for MS Heff).
        Only supported for ``pt2_backend="cuda"``.
    ipea_shift, imag_shift, real_shift : float
        Level-shift parameters for intruder-state removal.
    tol : float
        Solver convergence tolerance.
    maxiter : int
        Maximum number of PCG iterations.
    threshold : float
        Diagonal-norm threshold (Molcas THRSHN) for S-metric pre-scaling.
    threshold_s : float
        Scaled-S eigenvalue threshold (Molcas THRSHS) for linear-dependence
        removal.
    verbose : int
        Verbosity level (0=silent, 1=summary, 2=per-case detail).

    Returns
    -------
    CASPT2EnergyResult
        Contains ``e_ref``, ``e_pt2``, ``e_tot``, ``amplitudes`` (list of 13
        per-case amplitude vectors in the SR basis), and ``breakdown`` dict
        with per-case E2 contributions and shift corrections.
    """
    pt2_backend_norm = str(pt2_backend).strip().lower()
    if pt2_backend_norm in ("cuda", "cupy", "gpu"):
        from asuka.caspt2.cuda.energy_cuda import caspt2_energy_ss_cuda  # noqa: PLC0415

        if df_blocks is None:
            raise ValueError("pt2_backend='cuda' requires df_blocks to be provided")
        return caspt2_energy_ss_cuda(
            smap,
            fock,
            eri_mo,
            dm1,
            dm2,
            dm3,
            e_ref,
            df_blocks=df_blocks,  # type: ignore[arg-type]
            ci_context=ci_context,
            cuda_mode=str(cuda_mode),
            cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
            cuda_profile=bool(cuda_profile),
            ipea_shift=ipea_shift,
            imag_shift=imag_shift,
            real_shift=real_shift,
            tol=tol,
            maxiter=maxiter,
            threshold=threshold,
            threshold_s=threshold_s,
            verbose=verbose,
            device=cuda_device,
            store_rhs=store_rhs,
            store_row_dots=store_row_dots,
            mixed_precision_rhs=bool(mixed_precision_rhs),
        )

    if bool(store_row_dots):
        raise NotImplementedError("store_row_dots is only supported for pt2_backend='cuda'")

    nish = smap.orbs.nish
    nash = smap.orbs.nash
    nssh = smap.orbs.nssh

    if verbose >= 1:
        print(f"SS-CASPT2 Energy Calculation")
        print(f"  nish={nish}, nash={nash}, nssh={nssh}")
        print(f"  E_ref = {e_ref:.10f}")

    # Molcas EASUM = sum_w epsa[w] * <E_ww>.
    # Note: The B matrices are already constructed for (H0 - E0) in the
    # internally-contracted basis, so we do not subtract EASUM again when
    # assembling denominators.
    easum = compute_e0(fock, dm1, nish, nash)
    if verbose >= 1:
        print(f"  EASUM = {easum:.10f}")

    # Build S and B matrices for all 13 cases, diagonalize, build RHS
    sb_decomps: list[SBDecomposition] = []
    h0_diags: list[np.ndarray] = []
    rhs_transformed: list[np.ndarray] = []
    smats_std: list[np.ndarray] = []
    breakdown: dict[str, float] = {}

    for case in range(1, 14):
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])

        if nasup == 0 or nisup == 0:
            sb_decomps.append(SBDecomposition(
                s_eigvals=np.empty(0), transform=np.empty((0, 0)),
                nindep=0, b_diag=np.empty(0),
            ))
            h0_diags.append(np.empty(0, dtype=np.float64))
            rhs_transformed.append(np.empty(0, dtype=np.float64))
            smats_std.append(np.empty((0, 0), dtype=np.float64))
            if verbose >= 2:
                print(f"  Case {case} ({_CASE_NAMES[case-1]}): skipped (empty)")
            continue

        # Build S matrix (active superindex only)
        smat = build_smat(case, smap, dm1, dm2, dm3)
        smats_std.append(smat)

        # Build B matrix (active superindex only)
        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)

        # Joint diagonalization with linear-dep removal
        decomp = sbdiag(smat, bmat, threshold_norm=threshold, threshold_s=threshold_s)
        sb_decomps.append(decomp)

        if decomp.nindep == 0:
            h0_diags.append(np.empty(0, dtype=np.float64))
            rhs_transformed.append(np.empty(0, dtype=np.float64))
            if verbose >= 2:
                print(f"  Case {case} ({_CASE_NAMES[case-1]}): all lin-dep removed")
            continue

        # Build H0 diagonal in the diagonalized basis
        # For cases with external indices, add the external orbital energies
        h0_diag_active = decomp.b_diag  # (nindep,)

        # External orbital energy contributions
        ext_energies = _get_external_energies(case, smap, fock)

        # Full diagonal of (H0 - E0): (nindep * nisup,)
        # In OpenMolcas this is applied as delta = BD(mu) + ID(ext) (+ shifts),
        # see `resdia.f`. BD(mu) already contains the active-space (H0 - E0)
        # contribution (e.g. -EASUM*S terms are already included in B).
        if ext_energies.size > 0:
            full_diag = (h0_diag_active[:, None] + ext_energies[None, :]).ravel()
        else:
            full_diag = h0_diag_active.copy()

        h0_diags.append(full_diag)

        # Build RHS vector
        rhs_raw = build_rhs(case, smap, fock, eri_mo, dm1, dm2)

        # Transform RHS to diagonalized basis.
        # OpenMolcas PTRTOSR uses ITYPE=1 (S*T) for the RHS transformation:
        #   v_SR = (S*T)^T V = T^T S V
        # where T = transform and T^T S T = I.
        if nisup > 0 and nasup > 0:
            rhs_mat = rhs_raw.reshape(nasup, nisup)
            rhs_diag = decomp.transform.T @ smat @ rhs_mat  # (nindep, nisup)
            rhs_transformed.append(rhs_diag.ravel())
        else:
            rhs_transformed.append(rhs_raw)

        if verbose >= 2:
            print(f"  Case {case} ({_CASE_NAMES[case-1]}): "
                  f"nasup={nasup}, nisup={nisup}, nindep={decomp.nindep}")

    # Apply shifts
    h0_diags_orig = [d.copy() for d in h0_diags]
    if abs(imag_shift) > 1e-15:
        h0_diags = apply_imaginary_shift(h0_diags, imag_shift)
    if abs(real_shift) > 1e-15:
        h0_diags = apply_real_shift(h0_diags, real_shift)

    # Solve equations: (H0 - E0)|T> = -|V>
    def _maxabs(block: np.ndarray) -> float:
        if block.size == 0:
            return 0.0
        return float(np.max(np.abs(block)))

    use_sigma = False
    if int(smap.orbs.nash) > 0:
        ao = int(smap.orbs.nish)
        vo = ao + int(smap.orbs.nash)
        so = vo + int(smap.orbs.nssh)
        max_off = 0.0
        # Off-diagonal FBLOCKs in Molcas `fblock.f`: FTI/FIT/FIA/FAI/FTA/FAT
        if int(smap.orbs.nish) > 0:
            max_off = max(max_off, _maxabs(fock.fifa[ao:vo, :ao]))  # FTI
            max_off = max(max_off, _maxabs(fock.fifa[:ao, ao:vo]))  # FIT
        if int(smap.orbs.nssh) > 0:
            max_off = max(max_off, _maxabs(fock.fifa[ao:vo, vo:so]))  # FTA
            max_off = max(max_off, _maxabs(fock.fifa[vo:so, ao:vo]))  # FAT
        if int(smap.orbs.nish) > 0 and int(smap.orbs.nssh) > 0:
            max_off = max(max_off, _maxabs(fock.fifa[:ao, vo:so]))  # FIA
            max_off = max(max_off, _maxabs(fock.fifa[vo:so, :ao]))  # FAI
        use_sigma = max_off > 1e-12

    if use_sigma:
        nactel = max(1, int(round(float(np.trace(dm1[:nash, :nash])))))
        if int(smap.orbs.nish) > 0:
            sigma_op = SigmaC1CaseCoupling(
                smap=smap,
                fock=fock,
                smats=smats_std,
                sb_decomp=sb_decomps,
                h0_diag=h0_diags,
                nactel=nactel,
            )
        else:
            sigma_op = SigmaC1ActiveVirtualCoupling(
                smap=smap,
                fock=fock,
                smats=smats_std,
                sb_decomp=sb_decomps,
                h0_diag=h0_diags,
                nactel=nactel,
            )
        result = pcg_solve_iterative(
            sigma_op,
            rhs_transformed,
            h0_diags,
            tol=tol,
            maxiter=maxiter,
            verbose=verbose,
        )
    else:
        result = pcg_solve(h0_diags, rhs_transformed, tol=tol, maxiter=maxiter, verbose=verbose)

    if verbose >= 1 and not result.converged:
        print(f"  WARNING: PCG did not converge (|r|={result.residual:.2e})")

    # Compute energy: E2 = <V|T> = sum_P V_P * T_P
    e_pt2 = 0.0
    for case_idx, (v, t) in enumerate(zip(rhs_transformed, result.amplitudes)):
        if v.size == 0:
            continue
        e_case = float(np.dot(v.ravel(), t.ravel()))
        e_pt2 += e_case
        breakdown[f"e2_case{case_idx + 1}"] = e_case
        if verbose >= 1:
            print(f"  Case {case_idx + 1} ({_CASE_NAMES[case_idx]}): E2 = {e_case:.10f}")

    # Shift correction
    if abs(imag_shift) > 1e-15 or abs(real_shift) > 1e-15:
        e_shift_corr = compute_shift_correction(result.amplitudes, h0_diags, h0_diags_orig)
        e_pt2 -= e_shift_corr
        breakdown["e_shift_correction"] = -e_shift_corr
        if verbose >= 1:
            print(f"  Shift correction: {-e_shift_corr:.10f}")

    e_tot = e_ref + e_pt2
    breakdown["e_ref"] = e_ref
    breakdown["e_pt2"] = e_pt2
    breakdown["e_tot"] = e_tot

    if verbose >= 1:
        print(f"  E_PT2  = {e_pt2:.10f}")
        print(f"  E_tot  = {e_tot:.10f}")

    return CASPT2EnergyResult(
        e_ref=e_ref,
        e_pt2=e_pt2,
        e_tot=e_tot,
        amplitudes=result.amplitudes,
        breakdown=breakdown,
    )


def _get_external_energies(case: int, smap: SuperindexMap, fock: CASPT2Fock) -> np.ndarray:
    """Get external (inactive/virtual) orbital energy contributions.

    Returns the non-active diagonal contribution ID(ext) (OpenMolcas
    ``nadiag.f``) for each external superindex. Combined with the
    transformed B-matrix eigenvalues BD(mu) (which already represent the
    active-space part of ``H0 - E0``), this gives the CASPT2 denominator:

        denom[p, ext] = b_diag[p] + ext_energy[ext]

    Sign conventions:
      * Inactive hole  → -eps_i  (removing an electron costs +|eps_i|)
      * Virtual particle → +eps_a (adding an electron costs eps_a > 0)
    """
    nish = smap.orbs.nish
    nash = smap.orbs.nash
    nssh = smap.orbs.nssh

    # Orbital energies from Fock diagonal
    eps_i = np.array([fock.fifa[i, i] for i in range(nish)], dtype=np.float64)
    vo = nish + nash
    eps_a = np.array([fock.fifa[vo + a, vo + a] for a in range(nssh)], dtype=np.float64)

    if case == 1:  # A: 1 inactive hole
        return -eps_i
    elif case == 2:  # B+: 2 inactive holes (sym)
        energies = []
        for q in range(smap.nigej):
            i, j = smap.migej[q]
            energies.append(-(eps_i[i] + eps_i[j]))
        return np.array(energies, dtype=np.float64)
    elif case == 3:  # B-: 2 inactive holes (asym)
        energies = []
        for q in range(smap.nigtj):
            i, j = smap.migtj[q]
            energies.append(-(eps_i[i] + eps_i[j]))
        return np.array(energies, dtype=np.float64)
    elif case == 4:  # C: 1 virtual particle
        return eps_a
    elif case == 5:  # D: 1 virtual particle + 1 inactive hole
        energies = []
        for ai in range(nssh * nish):
            a = ai // nish
            i = ai % nish
            energies.append(eps_a[a] - eps_i[i])
        return np.array(energies, dtype=np.float64)
    elif case == 6:  # E+: 1 virtual + sym inactive pair (igej, a)
        energies = []
        for igej in range(smap.nigej):
            i, j = smap.migej[igej]
            for a in range(nssh):
                energies.append(-eps_i[i] - eps_i[j] + eps_a[a])
        return np.array(energies, dtype=np.float64)
    elif case == 7:  # E-: 1 virtual + asym inactive pair (igtj, a)
        energies = []
        for igtj in range(smap.nigtj):
            i, j = smap.migtj[igtj]
            for a in range(nssh):
                energies.append(-eps_i[i] - eps_i[j] + eps_a[a])
        return np.array(energies, dtype=np.float64)
    elif case == 8:  # F+: 2 virtual particles (sym)
        energies = []
        for q in range(smap.nageb):
            a, b = smap.mageb[q]
            energies.append(eps_a[a] + eps_a[b])
        return np.array(energies, dtype=np.float64)
    elif case == 9:  # F-: 2 virtual particles (asym)
        energies = []
        for q in range(smap.nagtb):
            a, b = smap.magtb[q]
            energies.append(eps_a[a] + eps_a[b])
        return np.array(energies, dtype=np.float64)
    elif case == 10:  # G+: 1 inactive + sym virtual pair (ageb, i)
        energies = []
        for ageb in range(smap.nageb):
            a, b = smap.mageb[ageb]
            for i in range(nish):
                energies.append(-eps_i[i] + eps_a[a] + eps_a[b])
        return np.array(energies, dtype=np.float64)
    elif case == 11:  # G-: 1 inactive + asym virtual pair (agtb, i)
        energies = []
        for agtb in range(smap.nagtb):
            a, b = smap.magtb[agtb]
            for i in range(nish):
                energies.append(-eps_i[i] + eps_a[a] + eps_a[b])
        return np.array(energies, dtype=np.float64)
    elif case == 12:  # H+: 2 inactive holes (virtual part in B matrix)
        energies = []
        for q in range(smap.nigej):
            i, j = smap.migej[q]
            energies.append(-(eps_i[i] + eps_i[j]))
        return np.array(energies, dtype=np.float64)
    elif case == 13:  # H-: 2 inactive holes (virtual part in B matrix)
        energies = []
        for q in range(smap.nigtj):
            i, j = smap.migtj[q]
            energies.append(-(eps_i[i] + eps_i[j]))
        return np.array(energies, dtype=np.float64)
    else:
        return np.empty(0, dtype=np.float64)
