"""IC-equivalent per-case amplitudes for SST-CASPT2 (SS).

This module provides a bridge from the SST backend (which decomposes the PT2
problem) to the existing Molcas-style Lagrangian/gradient machinery, which
expects IC-CASPT2 per-case SR-basis amplitudes plus the corresponding S/B
decomposition metadata.

The current implementation returns a 13-case amplitude list compatible with
``asuka.caspt2.pt2lag._build_case_amps_from_asuka``:

  - Cases 1-11 (A..G±) are solved as a reduced IC system with H± blocks
    omitted (matching the SST IC split).
  - Cases 12-13 (H±) are solved as an MP2-like external sector (diagonal).

The returned ``breakdown`` dict includes ``sb_transform_caseXX``,
``sb_bdiag_caseXX``, ``sb_nindep_caseXX``, and ``rhs_sr_caseXX`` so the
translation pipeline can reuse the exact SR basis and avoid basis drift in
near-degenerate channels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from asuka.caspt2.energy import _get_external_energies
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.shifts import (
    apply_imaginary_shift,
    apply_real_shift,
    compute_shift_correction,
)
from asuka.caspt2.sigma import SigmaC1ActiveVirtualCoupling, SigmaC1CaseCoupling
from asuka.caspt2.solver import pcg_solve, pcg_solve_iterative
from asuka.caspt2.superindex import SuperindexMap

__all__ = ["sst_ic_equivalent_amplitudes_ss"]


def _empty_decomp() -> SBDecomposition:
    return SBDecomposition(
        s_eigvals=np.empty(0, dtype=np.float64),
        transform=np.empty((0, 0), dtype=np.float64),
        nindep=0,
        b_diag=np.empty(0, dtype=np.float64),
    )


class _SigmaReduced11:
    """Sigma wrapper for the reduced 11-case system (pads empty H± blocks)."""

    def __init__(self, inner: SigmaC1CaseCoupling | SigmaC1ActiveVirtualCoupling) -> None:
        self._inner = inner

    def __call__(self, vec_in_11: list[np.ndarray]) -> list[np.ndarray]:
        vec_13 = list(vec_in_11) + [np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)]
        sigma_13 = self._inner(vec_13)
        return sigma_13[:11]


def _needs_sigma(smap: SuperindexMap, fock: CASPT2Fock) -> bool:
    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    if nash <= 0:
        return False
    ao = nish
    vo = nish + nash
    so = vo + nssh
    max_off = 0.0
    if nish > 0:
        max_off = max(max_off, float(np.max(np.abs(fock.fifa[ao:vo, :nish]))))
        max_off = max(max_off, float(np.max(np.abs(fock.fifa[:nish, ao:vo]))))
    if nssh > 0:
        max_off = max(max_off, float(np.max(np.abs(fock.fifa[ao:vo, vo:so]))))
        max_off = max(max_off, float(np.max(np.abs(fock.fifa[vo:so, ao:vo]))))
    if nish > 0 and nssh > 0:
        max_off = max(max_off, float(np.max(np.abs(fock.fifa[:nish, vo:so]))))
        max_off = max(max_off, float(np.max(np.abs(fock.fifa[vo:so, :nish]))))
    return bool(max_off > 1e-12)


def _build_sb_rhs_for_case(
    *,
    case: int,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    ci_context: CASPT2CIContext | None,
    threshold: float,
    threshold_s: float,
) -> tuple[SBDecomposition, np.ndarray, np.ndarray, np.ndarray]:
    """Return (decomp, smat, rhs_sr_mat, h0_diag_full) for one case."""
    nasup = int(smap.nasup[int(case) - 1])
    nisup = int(smap.nisup[int(case) - 1])
    if nasup == 0 or nisup == 0:
        return _empty_decomp(), np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    smat = build_smat(int(case), smap, dm1, dm2, dm3)
    bmat = build_bmat(int(case), smap, fock, dm1, dm2, dm3, ci_context=ci_context)
    decomp = sbdiag(smat, bmat, threshold_norm=float(threshold), threshold_s=float(threshold_s))
    if int(decomp.nindep) == 0:
        return decomp, smat, np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.float64)

    ext_energies = _get_external_energies(int(case), smap, fock)
    if ext_energies.size > 0:
        h0_diag_full = (decomp.b_diag[:, None] + ext_energies[None, :]).ravel()
    else:
        h0_diag_full = np.asarray(decomp.b_diag, dtype=np.float64).copy()

    rhs_raw = build_rhs(int(case), smap, fock, eri_mo, dm1, dm2)
    rhs_mat = rhs_raw.reshape(nasup, nisup)
    rhs_sr = decomp.transform.T @ smat @ rhs_mat  # (nIN, nIS)
    return decomp, smat, np.asarray(rhs_sr, dtype=np.float64), np.asarray(h0_diag_full, dtype=np.float64)


def sst_ic_equivalent_amplitudes_ss(
    *,
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    ci_context: CASPT2CIContext | None,
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    verbose: int = 0,
) -> tuple[float, list[np.ndarray], dict[str, Any], bool, int]:
    """Solve IC-equivalent SST amplitudes for SS-CASPT2.

    Returns
    -------
    e_pt2 : float
        Correlation energy (cases 1-11 + H±).
    amplitudes : list[ndarray]
        Length-13 list of per-case SR-basis amplitudes (flattened).
    breakdown : dict
        Contains SB replay keys (sb_transform_caseXX, sb_bdiag_caseXX,
        sb_nindep_caseXX, rhs_sr_caseXX) plus SST diagnostics.
    converged, niter : bool, int
        PCG status for the reduced 11-case solve (always True/1 when sigma is
        not needed).
    """
    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)
    if verbose >= 1:
        print("SST SS amplitudes (IC-equivalent): reduced cases 1-11 + H±")
        print(f"  nish={nish}, nash={nash}, nssh={nssh}")

    eri_mo = np.asarray(eri_mo, dtype=np.float64)
    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)

    breakdown: dict[str, Any] = {
        "sst_mode": "ic_equivalent",
        "sst_reduced_system": "cases_1_11",
    }

    # --- Cases 1-11 (A..G±): build S/B decomps and RHS in SR basis ---
    sb_decomps_11: list[SBDecomposition] = []
    smats_11: list[np.ndarray] = []
    rhs_sr_11: list[np.ndarray] = []
    h0_diags_11: list[np.ndarray] = []

    for case in range(1, 12):
        decomp, smat, rhs_sr_mat, h0_diag = _build_sb_rhs_for_case(
            case=case,
            smap=smap,
            fock=fock,
            eri_mo=eri_mo,
            dm1=dm1,
            dm2=dm2,
            dm3=dm3,
            ci_context=ci_context,
            threshold=float(threshold),
            threshold_s=float(threshold_s),
        )
        sb_decomps_11.append(decomp)
        smats_11.append(smat)
        rhs_sr_11.append(np.asarray(rhs_sr_mat, dtype=np.float64))
        h0_diags_11.append(np.asarray(h0_diag, dtype=np.float64))

        case_lbl = f"{int(case):02d}"
        if int(decomp.nindep) > 0 and rhs_sr_mat.size > 0:
            breakdown[f"sb_transform_case{case_lbl}"] = np.asarray(decomp.transform, dtype=np.float64)
            breakdown[f"sb_bdiag_case{case_lbl}"] = np.asarray(decomp.b_diag[: int(decomp.nindep)], dtype=np.float64)
            breakdown[f"sb_nindep_case{case_lbl}"] = int(decomp.nindep)
            breakdown[f"rhs_sr_case{case_lbl}"] = np.asarray(rhs_sr_mat, dtype=np.float64)

    # Apply shifts to the reduced-system diagonals.
    h0_diags_11_orig = [d.copy() for d in h0_diags_11]
    if abs(float(imag_shift)) > 1e-15:
        h0_diags_11 = apply_imaginary_shift(h0_diags_11, float(imag_shift))
    if abs(float(real_shift)) > 1e-15:
        h0_diags_11 = apply_real_shift(h0_diags_11, float(real_shift))

    rhs_flat_11 = [r.ravel() for r in rhs_sr_11]

    # Solve reduced system (optionally with sigma couplings among cases 1-11).
    use_sigma = _needs_sigma(smap, fock)
    if use_sigma:
        nactel = max(1, int(round(float(np.trace(dm1[:nash, :nash])))))
        empty = _empty_decomp()
        sb_decomps_13 = list(sb_decomps_11) + [empty, empty]
        h0_diags_13 = list(h0_diags_11) + [np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)]
        smats_13 = list(smats_11) + [np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)]
        if nish > 0:
            inner_sigma = SigmaC1CaseCoupling(
                smap=smap,
                fock=fock,
                smats=smats_13,
                sb_decomp=sb_decomps_13,
                h0_diag=h0_diags_13,
                nactel=int(nactel),
            )
        else:
            inner_sigma = SigmaC1ActiveVirtualCoupling(
                smap=smap,
                fock=fock,
                smats=smats_13,
                sb_decomp=sb_decomps_13,
                h0_diag=h0_diags_13,
                nactel=int(nactel),
            )
        sigma_op = _SigmaReduced11(inner_sigma)
        res = pcg_solve_iterative(
            sigma_op,
            rhs_flat_11,
            h0_diags_11,
            tol=float(tol),
            maxiter=int(maxiter),
            verbose=int(verbose),
        )
    else:
        res = pcg_solve(
            h0_diags_11,
            rhs_flat_11,
            tol=float(tol),
            maxiter=int(maxiter),
            verbose=int(verbose),
        )

    # Reduced-system energy: sum <v|t> for cases 1-11
    e_1_11 = 0.0
    for v, t in zip(rhs_flat_11, res.amplitudes):
        if v.size == 0:
            continue
        e_1_11 += float(np.dot(np.asarray(v, dtype=np.float64).ravel(), np.asarray(t, dtype=np.float64).ravel()))

    if abs(float(imag_shift)) > 1e-15 or abs(float(real_shift)) > 1e-15:
        e_shift_corr = compute_shift_correction(res.amplitudes, h0_diags_11, h0_diags_11_orig)
        e_1_11 -= float(e_shift_corr)
        breakdown["sst_shift_corr_cases_1_11"] = float(-e_shift_corr)

    breakdown["sst_e_cases_1_11"] = float(e_1_11)
    breakdown["sst_reduced_use_sigma"] = bool(use_sigma)
    breakdown["sst_reduced_converged"] = bool(res.converged)
    breakdown["sst_reduced_niter"] = int(res.niter)
    breakdown["sst_reduced_residual_norm"] = float(res.residual)

    amps_1_11 = [np.asarray(a, dtype=np.float64) for a in res.amplitudes]

    # --- Cases 12-13 (H±): diagonal MP2-like sector with SB metadata ---
    amps_12_13: list[np.ndarray] = []
    e_12_13 = 0.0
    for case in (12, 13):
        decomp, _smat, rhs_sr_mat, h0_diag = _build_sb_rhs_for_case(
            case=case,
            smap=smap,
            fock=fock,
            eri_mo=eri_mo,
            dm1=dm1,
            dm2=dm2,
            dm3=dm3,
            ci_context=ci_context,
            threshold=float(threshold),
            threshold_s=float(threshold_s),
        )
        case_lbl = f"{int(case):02d}"
        if int(decomp.nindep) > 0 and rhs_sr_mat.size > 0:
            breakdown[f"sb_transform_case{case_lbl}"] = np.asarray(decomp.transform, dtype=np.float64)
            breakdown[f"sb_bdiag_case{case_lbl}"] = np.asarray(decomp.b_diag[: int(decomp.nindep)], dtype=np.float64)
            breakdown[f"sb_nindep_case{case_lbl}"] = int(decomp.nindep)
            breakdown[f"rhs_sr_case{case_lbl}"] = np.asarray(rhs_sr_mat, dtype=np.float64)

        v = np.asarray(rhs_sr_mat, dtype=np.float64).ravel()
        if v.size == 0 or h0_diag.size == 0:
            amps_12_13.append(np.empty((0,), dtype=np.float64))
            continue

        h0_list = [np.asarray(h0_diag, dtype=np.float64)]
        h0_orig = [h0_list[0].copy()]
        if abs(float(imag_shift)) > 1e-15:
            h0_list = apply_imaginary_shift(h0_list, float(imag_shift))
        if abs(float(real_shift)) > 1e-15:
            h0_list = apply_real_shift(h0_list, float(real_shift))

        res_h = pcg_solve(
            h0_list,
            [v],
            tol=float(tol),
            maxiter=int(maxiter),
            verbose=int(verbose),
        )
        t = np.asarray(res_h.amplitudes[0], dtype=np.float64)
        amps_12_13.append(t)
        e_case = float(np.dot(v, t))
        if abs(float(imag_shift)) > 1e-15 or abs(float(real_shift)) > 1e-15:
            e_shift_corr = compute_shift_correction([t], h0_list, h0_orig)
            e_case -= float(e_shift_corr)
        e_12_13 += e_case

    breakdown["sst_e_hpm"] = float(e_12_13)
    e_pt2 = float(e_1_11 + e_12_13)
    breakdown["sst_e_pt2_total"] = float(e_pt2)

    amplitudes_13 = list(amps_1_11) + list(amps_12_13)
    if len(amplitudes_13) != 13:
        raise RuntimeError(f"internal error: expected 13 cases, got {len(amplitudes_13)}")

    return e_pt2, amplitudes_13, breakdown, bool(res.converged), int(res.niter)

