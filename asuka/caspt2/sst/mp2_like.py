"""MP2-like energy from the SST Schur complement (external doubles).

Stage 0: brute-force implementation using full MO ERIs.
Stage 1: replaces with DF-based contractions.

This computes the energy contribution from the purely external sector
(cases H+ and H- in the IC framework), where all occupied indices are
core and all virtual indices are secondary.  No active RDMs are needed.

The orbital energies used are the **diagonal elements of the full MO Fock
matrix (fifa)**, consistent with the IC-CASPT2 denominator convention.
Off-diagonal Fock couplings (to active space) are handled by the sigma
operator in the IC framework; in the SST framework they appear in the
reduced active system, not here.

Level shifts (imaginary and real) are applied consistently with the IC
framework: the denominator is modified, amplitudes solved with the shifted
denominator, and a shift correction is subtracted from the energy.
"""
from __future__ import annotations

import numpy as np

from asuka.caspt2.sst.types import SSTInput, SSTConfig


def sst_mp2_like_energy(inp: SSTInput, cfg: SSTConfig) -> float:
    """Compute the MP2-like external energy with level shift support.

    This is the energy contribution from the purely external excitation
    sector: double excitations from core->virtual with no active orbital
    indices.  Corresponds to IC-CASPT2 cases H+ and H-.

    Uses the MO Fock diagonal for orbital energies (consistent with IC
    denominator convention).

    Parameters
    ----------
    inp : SSTInput
    cfg : SSTConfig

    Returns
    -------
    E_mp2_like : float
        Shift-corrected MP2-like energy.
    """
    ncore = inp.ncore
    ncas = inp.ncas
    nvirt = inp.nvirt

    if ncore == 0 or nvirt == 0:
        return 0.0

    # Orbital energies: diagonal of fifa (MO Fock)
    fifa = np.asarray(inp.fock.fifa, dtype=np.float64)
    nocc = ncore + ncas
    eps_core = np.array([fifa[i, i] for i in range(ncore)], dtype=np.float64)
    eps_virt = np.array([fifa[nocc + a, nocc + a] for a in range(nvirt)], dtype=np.float64)

    if inp.eri_mo is not None:
        return _mp2_like_from_full_eri(inp.eri_mo, ncore, ncas, nvirt, eps_core, eps_virt, cfg)

    if inp.B_ao is not None:
        return _mp2_like_from_df(inp, eps_core, eps_virt, cfg)

    raise ValueError("sst_mp2_like_energy requires either eri_mo or B_ao")


def _apply_shift_to_denom(denom: float, imag_shift: float, real_shift: float) -> float:
    """Apply imaginary and real shifts to a denominator value.

    Imaginary shift: d -> d + sigma^2/d
    Real shift: d -> d + s
    """
    d = denom
    if abs(imag_shift) > 1e-15 and abs(d) > 1e-14:
        d += imag_shift * imag_shift / d
    if abs(real_shift) > 1e-15:
        d += real_shift
    return d


def _mp2_like_from_full_eri(
    eri_mo: np.ndarray,
    ncore: int,
    ncas: int,
    nvirt: int,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    cfg: SSTConfig,
) -> float:
    """Brute-force MP2-like energy from full MO ERIs (Stage 0).

    Computes the sum of IC-CASPT2 cases H+ and H-:

    E_H+ = sum_{a>=b, i>=j} [(ai|bj)+(aj|bi)]^2 / [(1+d_ab)(1+d_ij)*denom]
    E_H- = sum_{a>b, i>j} 3*[(ai|bj)-(aj|bi)]^2 / denom

    where denom = eps_a + eps_b - eps_i - eps_j, (ai|bj) = eri_mo[a,i,b,j]
    in chemists' notation.

    With level shifts, the energy is E_shifted - E_correction where
    E_shifted uses the modified denominator and E_correction = sum |T|^2 * delta_d.
    """
    nocc = ncore + ncas
    has_shift = abs(cfg.imag_shift) > 1e-15 or abs(cfg.real_shift) > 1e-15

    e_hp = 0.0
    e_hm = 0.0
    shift_corr = 0.0

    # H+ contribution: symmetric pairs a>=b, i>=j
    for a in range(nvirt):
        for b in range(a + 1):  # b <= a
            d_ab = 1.0 if a == b else 0.0
            for i in range(ncore):
                for j in range(i + 1):  # j <= i
                    d_ij = 1.0 if i == j else 0.0
                    denom_orig = eps_virt[a] + eps_virt[b] - eps_core[i] - eps_core[j]
                    norm_factor = (1.0 + d_ab) * (1.0 + d_ij)
                    # (ai|bj) in chemists' notation
                    aibj = eri_mo[nocc + a, i, nocc + b, j]
                    ajbi = eri_mo[nocc + a, j, nocc + b, i]
                    v_plus = aibj + ajbi
                    v2 = v_plus ** 2 / norm_factor

                    if has_shift:
                        denom_shifted = _apply_shift_to_denom(denom_orig, cfg.imag_shift, cfg.real_shift)
                        e_hp -= v2 / denom_shifted
                        t_p = -v_plus / (np.sqrt(norm_factor) * denom_shifted)
                        shift_corr += t_p ** 2 * (denom_shifted - denom_orig)
                    else:
                        e_hp -= v2 / denom_orig

    # H- contribution: antisymmetric pairs a>b, i>j
    for a in range(nvirt):
        for b in range(a):  # b < a
            for i in range(ncore):
                for j in range(i):  # j < i
                    denom_orig = eps_virt[a] + eps_virt[b] - eps_core[i] - eps_core[j]
                    aibj = eri_mo[nocc + a, i, nocc + b, j]
                    ajbi = eri_mo[nocc + a, j, nocc + b, i]
                    v_minus = aibj - ajbi
                    v2 = 3.0 * v_minus ** 2

                    if has_shift:
                        denom_shifted = _apply_shift_to_denom(denom_orig, cfg.imag_shift, cfg.real_shift)
                        e_hm -= v2 / denom_shifted
                        t_p = -np.sqrt(3.0) * v_minus / denom_shifted
                        shift_corr += t_p ** 2 * (denom_shifted - denom_orig)
                    else:
                        e_hm -= v2 / denom_orig

    e_total = e_hp + e_hm
    if has_shift:
        e_total -= shift_corr
    return float(e_total)


def _mp2_like_from_df(
    inp: SSTInput,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    cfg: SSTConfig,
) -> float:
    """DF-based MP2-like energy (Stage 1).

    Builds L_ai from DF factors and computes the H+/H- energy directly.
    Uses the original (non-semicanonical) MO coefficients.
    """
    ncore = inp.ncore
    ncas = inp.ncas
    nvirt = inp.nvirt
    nocc = ncore + ncas
    B_ao = np.asarray(inp.B_ao, dtype=np.float64)
    C = np.asarray(inp.mo_coeff, dtype=np.float64)
    has_shift = abs(cfg.imag_shift) > 1e-15 or abs(cfg.real_shift) > 1e-15

    nao, _, naux = B_ao.shape
    # Build L_ai[a, i, Q] in original MO basis
    C_core = C[:, :ncore]          # (nao, ncore)
    C_virt = C[:, nocc:nocc+nvirt]  # (nao, nvirt)

    # L_ai[a, i, Q] = sum_{mu,nu} C_virt[mu,a] * B_ao[mu,nu,Q] * C_core[nu,i]
    B_flat = B_ao.reshape(nao, nao * naux)
    tmp = C_virt.T @ B_flat  # (nvirt, nao*naux)
    tmp = tmp.reshape(nvirt, nao, naux)
    L_ai = np.einsum("anQ,ni->aiQ", tmp, C_core)  # (nvirt, ncore, naux)

    e_hp = 0.0
    e_hm = 0.0
    shift_corr = 0.0

    # H+: symmetric pairs a>=b, i>=j
    for a in range(nvirt):
        for b in range(a + 1):
            d_ab = 1.0 if a == b else 0.0
            for i in range(ncore):
                for j in range(i + 1):
                    d_ij = 1.0 if i == j else 0.0
                    denom_orig = eps_virt[a] + eps_virt[b] - eps_core[i] - eps_core[j]
                    norm_factor = (1.0 + d_ab) * (1.0 + d_ij)
                    # (ai|bj) = sum_Q L_ai[a,i,Q] * L_ai[b,j,Q]
                    aibj = np.dot(L_ai[a, i, :], L_ai[b, j, :])
                    ajbi = np.dot(L_ai[a, j, :], L_ai[b, i, :])
                    v_plus = aibj + ajbi
                    v2 = v_plus ** 2 / norm_factor

                    if has_shift:
                        denom_shifted = _apply_shift_to_denom(denom_orig, cfg.imag_shift, cfg.real_shift)
                        e_hp -= v2 / denom_shifted
                        t_p = -v_plus / (np.sqrt(norm_factor) * denom_shifted)
                        shift_corr += t_p ** 2 * (denom_shifted - denom_orig)
                    else:
                        e_hp -= v2 / denom_orig

    # H-: antisymmetric pairs a>b, i>j
    for a in range(nvirt):
        for b in range(a):
            for i in range(ncore):
                for j in range(i):
                    denom_orig = eps_virt[a] + eps_virt[b] - eps_core[i] - eps_core[j]
                    aibj = np.dot(L_ai[a, i, :], L_ai[b, j, :])
                    ajbi = np.dot(L_ai[a, j, :], L_ai[b, i, :])
                    v_minus = aibj - ajbi
                    v2 = 3.0 * v_minus ** 2

                    if has_shift:
                        denom_shifted = _apply_shift_to_denom(denom_orig, cfg.imag_shift, cfg.real_shift)
                        e_hm -= v2 / denom_shifted
                        t_p = -np.sqrt(3.0) * v_minus / denom_shifted
                        shift_corr += t_p ** 2 * (denom_shifted - denom_orig)
                    else:
                        e_hm -= v2 / denom_orig

    e_total = e_hp + e_hm
    if has_shift:
        e_total -= shift_corr
    return float(e_total)
