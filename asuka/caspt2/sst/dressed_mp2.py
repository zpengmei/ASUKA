"""Dressed MP2 energy for SST-CASPT2.

Computes E^MP2_dressed (Eq. 33 of Song 2024) — a standard closed-shell MP2
energy formula evaluated in the dressed orbital basis where ALL hole×particle
index pairs contribute (not just core×virtual as in standard MP2).

The dressed orbitals are obtained by diagonalizing the coupling matrices γ
(hole) and γ̄ (particle), giving dressed energies ω_η and ω̄_π.

E^MP2_dressed = -Σ_{η1,η2,π1,π2} |v^+|² / [(1+δ_{η1η2})(1+δ_{π1π2}) d]
              - Σ_{η1>η2, π1>π2} 3|v^-|² / d

where v^+ = (η1π1|η2π2) + (η1π2|η2π1) [symmetric],
      v^- = (η1π1|η2π2) - (η1π2|η2π1) [antisymmetric],
      d = ω̄_π1 + ω̄_π2 - ω_η1 - ω_η2.

This is the "leading subspace" energy (E_L in the paper's notation).
"""
from __future__ import annotations

import numpy as np

from asuka.caspt2.sst.coupling import DressedOrbitals
from asuka.caspt2.sst.types import SSTInput

__all__ = ["compute_dressed_mp2"]


def compute_dressed_mp2(
    inp: SSTInput,
    dressed: DressedOrbitals,
) -> float:
    """Compute E^MP2_dressed in the dressed orbital basis.

    Parameters
    ----------
    inp
        SST input data (provides B_ao, mo_coeff, eri_mo).
    dressed
        Dressed orbitals from coupling matrix diagonalization.

    Returns
    -------
    float
        Dressed MP2 energy E^MP2_dressed (Eq. 33).
    """
    ncore = dressed.ncore
    ncas = dressed.ncas
    nvirt = dressed.nvirt
    nhole = ncore + ncas
    nparticle = ncas + nvirt
    nocc = ncore + ncas

    omega = dressed.omega_hole      # (nhole,)
    omega_bar = dressed.omega_particle  # (nparticle,)

    if nhole == 0 or nparticle == 0:
        return 0.0

    if inp.eri_mo is not None:
        return _dressed_mp2_from_full_eri(
            inp.eri_mo, inp.mo_coeff, dressed, omega, omega_bar,
        )

    if inp.B_ao is not None:
        return _dressed_mp2_from_df(
            inp.B_ao, inp.mo_coeff, dressed, omega, omega_bar,
        )

    raise ValueError("compute_dressed_mp2 requires either eri_mo or B_ao")


def _build_dressed_eris(
    eri_mo: np.ndarray,
    mo_coeff: np.ndarray,
    dressed: DressedOrbitals,
) -> np.ndarray:
    """Build dressed-basis 2e integrals (η1 π1 | η2 π2) from MO ERIs.

    The dressed integrals are obtained by transforming from MO to dressed basis:
        (η π | η' π') = Σ_{pqrs} U_h[p,η] U_p[q,π] (pq|rs) U_h[r,η'] U_p[s,π']

    where p,r are hole MOs (indices 0..nhole-1) and q,s are particle MOs
    (indices ncore..nmo-1 in full MO indexing).

    Returns shape (nhole, nparticle, nhole, nparticle) in chemists' notation.
    """
    eri_mo = np.asarray(eri_mo, dtype=np.float64)
    ncore = dressed.ncore
    ncas = dressed.ncas
    nvirt = dressed.nvirt
    nocc = ncore + ncas
    nmo = ncore + ncas + nvirt
    nhole = ncore + ncas
    nparticle = ncas + nvirt

    u_h = dressed.u_hole       # (nhole, nhole)
    u_p = dressed.u_particle   # (nparticle, nparticle)

    # Extract the relevant ERI block: hole indices = [0..nhole-1],
    # particle indices = [ncore..nmo-1] in the MO basis.
    # eri_block[h1, p1, h2, p2] = eri_mo[h1, ncore+p1, h2, ncore+p2]
    hole_idx = np.arange(nhole)
    part_mo_idx = np.arange(ncore, nmo)  # particle MOs start at ncore

    eri_block = eri_mo[np.ix_(hole_idx, part_mo_idx, hole_idx, part_mo_idx)]
    # shape: (nhole, nparticle, nhole, nparticle)

    # 4-index transform via tensordot (step by step)
    # Step 1: transform 1st index (hole)
    tmp1 = np.tensordot(u_h.T, eri_block, axes=([1], [0]))
    # tmp1[η1, p1_mo, h2_mo, p2_mo] = Σ_h1 U_h[h1,η1] * eri[h1, p1, h2, p2]

    # Step 2: transform 2nd index (particle)
    tmp2 = np.tensordot(tmp1, u_p, axes=([1], [0]))
    # tmp2[η1, h2_mo, p2_mo, π1] → need to transpose
    tmp2 = tmp2.transpose(0, 3, 1, 2)
    # tmp2[η1, π1, h2_mo, p2_mo]

    # Step 3: transform 3rd index (hole)
    tmp3 = np.tensordot(tmp2, u_h, axes=([2], [0]))
    # tmp3[η1, π1, p2_mo, η2] → transpose
    tmp3 = tmp3.transpose(0, 1, 3, 2)
    # tmp3[η1, π1, η2, p2_mo]

    # Step 4: transform 4th index (particle)
    result = np.tensordot(tmp3, u_p, axes=([3], [0]))
    # result[η1, π1, η2, π2]

    return np.asarray(result, dtype=np.float64, order="C")


def _dressed_mp2_from_full_eri(
    eri_mo: np.ndarray,
    mo_coeff: np.ndarray,
    dressed: DressedOrbitals,
    omega: np.ndarray,
    omega_bar: np.ndarray,
) -> float:
    """Compute E^MP2_dressed from full MO ERIs (brute force)."""
    nhole = dressed.ncore + dressed.ncas
    nparticle = dressed.ncas + dressed.nvirt

    # Build dressed-basis integrals
    eri_dressed = _build_dressed_eris(eri_mo, mo_coeff, dressed)
    # eri_dressed[η1, π1, η2, π2] in chemists' notation

    e_plus = 0.0
    e_minus = 0.0

    # H+ contribution: symmetric pairs η1 >= η2, π1 >= π2
    for eta1 in range(nhole):
        for eta2 in range(eta1 + 1):  # eta2 <= eta1
            d_eta = 1.0 if eta1 == eta2 else 0.0
            for pi1 in range(nparticle):
                for pi2 in range(pi1 + 1):  # pi2 <= pi1
                    d_pi = 1.0 if pi1 == pi2 else 0.0
                    denom = (omega_bar[pi1] + omega_bar[pi2]
                             - omega[eta1] - omega[eta2])

                    if abs(denom) < 1e-14:
                        continue

                    # (η1 π1 | η2 π2) + (η1 π2 | η2 π1)
                    v_plus = (eri_dressed[eta1, pi1, eta2, pi2]
                              + eri_dressed[eta1, pi2, eta2, pi1])
                    norm_factor = (1.0 + d_eta) * (1.0 + d_pi)
                    e_plus -= v_plus ** 2 / (norm_factor * denom)

    # H- contribution: antisymmetric pairs η1 > η2, π1 > π2
    for eta1 in range(nhole):
        for eta2 in range(eta1):  # eta2 < eta1
            for pi1 in range(nparticle):
                for pi2 in range(pi1):  # pi2 < pi1
                    denom = (omega_bar[pi1] + omega_bar[pi2]
                             - omega[eta1] - omega[eta2])

                    if abs(denom) < 1e-14:
                        continue

                    # (η1 π1 | η2 π2) - (η1 π2 | η2 π1)
                    v_minus = (eri_dressed[eta1, pi1, eta2, pi2]
                               - eri_dressed[eta1, pi2, eta2, pi1])
                    e_minus -= 3.0 * v_minus ** 2 / denom

    return float(e_plus + e_minus)


def _dressed_mp2_from_df(
    B_ao: np.ndarray,
    mo_coeff: np.ndarray,
    dressed: DressedOrbitals,
    omega: np.ndarray,
    omega_bar: np.ndarray,
) -> float:
    """Compute E^MP2_dressed from DF factors.

    Builds L[η, π, Q] in dressed basis and evaluates the MP2 formula.
    """
    B_ao = np.asarray(B_ao, dtype=np.float64)
    C = np.asarray(mo_coeff, dtype=np.float64)
    ncore = dressed.ncore
    ncas = dressed.ncas
    nvirt = dressed.nvirt
    nocc = ncore + ncas
    nmo = ncore + ncas + nvirt
    nhole = ncore + ncas
    nparticle = ncas + nvirt
    nao = C.shape[0]
    naux = B_ao.shape[2]

    u_h = dressed.u_hole       # (nhole, nhole)
    u_p = dressed.u_particle   # (nparticle, nparticle)

    # Build dressed-basis MO coefficients
    # Hole MOs are columns 0..nhole-1 of C
    C_hole_mo = C[:, :nhole]  # (nao, nhole)
    C_hole_dressed = C_hole_mo @ u_h  # (nao, nhole)

    # Particle MOs are columns ncore..nmo-1 of C
    C_part_mo = C[:, ncore:nmo]  # (nao, nparticle)
    C_part_dressed = C_part_mo @ u_p  # (nao, nparticle)

    # Build L[η, π, Q] = Σ_{μν} C_hole_dressed[μ,η] B_ao[μ,ν,Q] C_part_dressed[ν,π]
    B_flat = B_ao.reshape(nao, nao * naux)
    tmp = C_hole_dressed.T @ B_flat  # (nhole, nao*naux)
    tmp = tmp.reshape(nhole, nao, naux)
    L = np.einsum("hnQ,np->hpQ", tmp, C_part_dressed, optimize=True)
    # L[η, π, Q]

    e_plus = 0.0
    e_minus = 0.0

    # H+ contribution: symmetric pairs η1 >= η2, π1 >= π2
    for eta1 in range(nhole):
        for eta2 in range(eta1 + 1):
            d_eta = 1.0 if eta1 == eta2 else 0.0
            for pi1 in range(nparticle):
                for pi2 in range(pi1 + 1):
                    d_pi = 1.0 if pi1 == pi2 else 0.0
                    denom = (omega_bar[pi1] + omega_bar[pi2]
                             - omega[eta1] - omega[eta2])

                    if abs(denom) < 1e-14:
                        continue

                    # (η1 π1 | η2 π2) = Σ_Q L[η1,π1,Q] L[η2,π2,Q]
                    int1 = np.dot(L[eta1, pi1, :], L[eta2, pi2, :])
                    int2 = np.dot(L[eta1, pi2, :], L[eta2, pi1, :])
                    v_plus = int1 + int2
                    norm_factor = (1.0 + d_eta) * (1.0 + d_pi)
                    e_plus -= v_plus ** 2 / (norm_factor * denom)

    # H- contribution: antisymmetric pairs η1 > η2, π1 > π2
    for eta1 in range(nhole):
        for eta2 in range(eta1):
            for pi1 in range(nparticle):
                for pi2 in range(pi1):
                    denom = (omega_bar[pi1] + omega_bar[pi2]
                             - omega[eta1] - omega[eta2])

                    if abs(denom) < 1e-14:
                        continue

                    int1 = np.dot(L[eta1, pi1, :], L[eta2, pi2, :])
                    int2 = np.dot(L[eta1, pi2, :], L[eta2, pi1, :])
                    v_minus = int1 - int2
                    e_minus -= 3.0 * v_minus ** 2 / denom

    return float(e_plus + e_minus)
