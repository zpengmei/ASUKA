"""NDDO Fock matrix construction for semiempirical methods.

Builds the core Hamiltonian H and the two-electron contribution G(P)
to the Fock matrix F = H + G(P).

Reference: MOPAC h1elec.F90, fock.F90
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .basis import build_ao_offsets, nao_for_Z, valence_electrons
from .params import ElementParams


def build_onecenter_eris(
    norb: int,
    gss: float,
    gsp: float,
    gpp: float,
    gp2: float,
    hsp: float,
) -> np.ndarray:
    """Build one-center ERI tensor from empirical Slater-Condon parameters.

    Returns (norb, norb, norb, norb) tensor with (mu nu | lam sig) indexing.
    Parameters in Hartree.
    """
    G = np.zeros((norb, norb, norb, norb), dtype=float)

    if norb == 1:
        G[0, 0, 0, 0] = gss
        return G

    # s=0, px=1, py=2, pz=3
    G[0, 0, 0, 0] = gss

    for p in range(1, 4):
        G[0, 0, p, p] = gsp
        G[p, p, 0, 0] = gsp
        G[p, p, p, p] = gpp
        G[0, p, 0, p] = hsp
        G[p, 0, p, 0] = hsp
        G[0, p, p, 0] = hsp
        G[p, 0, 0, p] = hsp

        for q in range(1, 4):
            if q == p:
                continue
            G[p, p, q, q] = gp2
            G[p, q, p, q] = 0.5 * (gpp - gp2)
            G[p, q, q, p] = 0.5 * (gpp - gp2)

    return G


def build_core_hamiltonian(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    S: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    W_list: List[np.ndarray],
    elem_params: Dict[int, ElementParams],
) -> np.ndarray:
    """Build the one-electron core Hamiltonian matrix.

    Diagonal (one-center):
        H[mu,mu] = U_mu - sum_{B!=A} Z_B * (mu mu | ss_B)

    Off-diagonal (two-center, A != B):
        H[mu,nu] = 0.5 * S[mu,nu] * (beta_mu + beta_nu)

    Parameters
    ----------
    atomic_numbers : sequence of int
    coords_bohr : (N, 3) array
    S : (nao, nao) overlap matrix
    pair_i, pair_j : (npairs,) int arrays
    W_list : list of two-center integral arrays
    elem_params : dict of ElementParams keyed by Z

    Returns
    -------
    H : (nao, nao) array
    """
    N = len(atomic_numbers)
    offsets = build_ao_offsets(atomic_numbers)
    nao_total = offsets[-1]
    H = np.zeros((nao_total, nao_total), dtype=float)

    # Build beta values for each AO
    beta_ao = np.zeros(nao_total, dtype=float)
    for iatom, Z in enumerate(atomic_numbers):
        ep = elem_params[Z]
        i0 = offsets[iatom]
        nao = nao_for_Z(Z)
        beta_ao[i0] = ep.beta_s
        if nao == 4:
            beta_ao[i0 + 1] = ep.beta_p
            beta_ao[i0 + 2] = ep.beta_p
            beta_ao[i0 + 3] = ep.beta_p

    # Diagonal: one-center one-electron integrals U_mu
    for iatom, Z in enumerate(atomic_numbers):
        ep = elem_params[Z]
        i0 = offsets[iatom]
        nao = nao_for_Z(Z)
        H[i0, i0] = ep.uss
        if nao == 4:
            H[i0 + 1, i0 + 1] = ep.upp
            H[i0 + 2, i0 + 2] = ep.upp
            H[i0 + 3, i0 + 3] = ep.upp

    # Add electron-nuclear attraction: -Z_B * (mu_A nu_A | ss_B)
    # This modifies the diagonal block H_AA
    npairs = len(pair_i)
    for k in range(npairs):
        iA = pair_i[k]
        iB = pair_j[k]
        Z_A = atomic_numbers[iA]
        Z_B = atomic_numbers[iB]
        i0A = offsets[iA]
        i0B = offsets[iB]
        naoA = nao_for_Z(Z_A)
        naoB = nao_for_Z(Z_B)
        zval_A = valence_electrons(Z_A)
        zval_B = valence_electrons(Z_B)

        W = W_list[k]

        # Attraction of A electrons to B core: -Z_B * W[mu_A, nu_A, 0, 0]
        idxA = slice(i0A, i0A + naoA)
        H[idxA, idxA] -= zval_B * W[:naoA, :naoA, 0, 0]

        # Attraction of B electrons to A core: -Z_A * W[0, 0, lam_B, sig_B]
        idxB = slice(i0B, i0B + naoB)
        H[idxB, idxB] -= zval_A * W[0, 0, :naoB, :naoB]

    # Off-diagonal (two-center) resonance integrals
    for k in range(npairs):
        iA = pair_i[k]
        iB = pair_j[k]
        i0A = offsets[iA]
        i0B = offsets[iB]
        naoA = nao_for_Z(atomic_numbers[iA])
        naoB = nao_for_Z(atomic_numbers[iB])

        for mu_loc in range(naoA):
            mu = i0A + mu_loc
            for nu_loc in range(naoB):
                nu = i0B + nu_loc
                H[mu, nu] = 0.5 * S[mu, nu] * (beta_ao[mu] + beta_ao[nu])
                H[nu, mu] = H[mu, nu]

    return H


def build_core_hamiltonian_from_pair_terms(
    atomic_numbers: Sequence[int],
    S: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    vaa_pack: np.ndarray,
    vbb_pack: np.ndarray,
    elem_params: Dict[int, ElementParams],
) -> np.ndarray:
    """Build core Hamiltonian from reduced pair terms.

    Parameters
    ----------
    atomic_numbers : sequence of int
        Atomic numbers.
    S : (nao, nao) overlap matrix.
    pair_i, pair_j : (npairs,) int arrays
        Pair list indices.
    vaa_pack : (npairs, 16) float64
        Packed ``W[:,:,0,0]`` terms per pair in padded 4x4 form.
    vbb_pack : (npairs, 16) float64
        Packed ``W[0,0,:,:]`` terms per pair in padded 4x4 form.
    elem_params : dict of ElementParams keyed by Z
        AM1 element parameters.

    Returns
    -------
    H : (nao, nao) array
        Core Hamiltonian matrix.
    """
    N = len(atomic_numbers)
    offsets = build_ao_offsets(atomic_numbers)
    nao_total = offsets[-1]
    H = np.zeros((nao_total, nao_total), dtype=float)

    beta_ao = np.zeros(nao_total, dtype=float)
    for iatom, Z in enumerate(atomic_numbers):
        ep = elem_params[Z]
        i0 = offsets[iatom]
        nao = nao_for_Z(Z)
        beta_ao[i0] = ep.beta_s
        if nao == 4:
            beta_ao[i0 + 1] = ep.beta_p
            beta_ao[i0 + 2] = ep.beta_p
            beta_ao[i0 + 3] = ep.beta_p

    for iatom, Z in enumerate(atomic_numbers):
        ep = elem_params[Z]
        i0 = offsets[iatom]
        nao = nao_for_Z(Z)
        H[i0, i0] = ep.uss
        if nao == 4:
            H[i0 + 1, i0 + 1] = ep.upp
            H[i0 + 2, i0 + 2] = ep.upp
            H[i0 + 3, i0 + 3] = ep.upp

    npairs = len(pair_i)
    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        Z_A = atomic_numbers[iA]
        Z_B = atomic_numbers[iB]
        i0A = offsets[iA]
        i0B = offsets[iB]
        naoA = nao_for_Z(Z_A)
        naoB = nao_for_Z(Z_B)
        zval_A = valence_electrons(Z_A)
        zval_B = valence_electrons(Z_B)

        idxA = slice(i0A, i0A + naoA)
        idxB = slice(i0B, i0B + naoB)
        Waa = vaa_pack[k].reshape(4, 4)[:naoA, :naoA]
        Wbb = vbb_pack[k].reshape(4, 4)[:naoB, :naoB]
        H[idxA, idxA] -= zval_B * Waa
        H[idxB, idxB] -= zval_A * Wbb

    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        i0A = offsets[iA]
        i0B = offsets[iB]
        naoA = nao_for_Z(atomic_numbers[iA])
        naoB = nao_for_Z(atomic_numbers[iB])

        for mu_loc in range(naoA):
            mu = i0A + mu_loc
            for nu_loc in range(naoB):
                nu = i0B + nu_loc
                H[mu, nu] = 0.5 * S[mu, nu] * (beta_ao[mu] + beta_ao[nu])
                H[nu, mu] = H[mu, nu]

    return H


def build_fock(
    H: np.ndarray,
    P: np.ndarray,
    atomic_numbers: Sequence[int],
    offsets: np.ndarray,
    onecenter_eris: List[np.ndarray],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    W_list: List[np.ndarray],
) -> np.ndarray:
    """Build Fock matrix F = H + G(P).

    Parameters
    ----------
    H : (nao, nao) core Hamiltonian
    P : (nao, nao) density matrix
    atomic_numbers : sequence of int
    offsets : (N+1,) AO offset array
    onecenter_eris : list of (norb, norb, norb, norb) arrays per atom
    pair_i, pair_j : (npairs,) int arrays
    W_list : list of two-center integral arrays

    Returns
    -------
    F : (nao, nao) Fock matrix
    """
    N = len(atomic_numbers)
    F = H.copy()

    # One-center contributions: J - 0.5*K from one-center ERIs
    for A in range(N):
        i0 = offsets[A]
        nao = nao_for_Z(atomic_numbers[A])
        idx = slice(i0, i0 + nao)
        P_AA = P[idx, idx]
        G_AA = onecenter_eris[A]

        # Coulomb: J[mn] = sum_{ls} P[ls] * G[mn,ls]
        J = np.einsum("ls,mnls->mn", P_AA, G_AA, optimize=True)
        # Exchange: K[mn] = sum_{ls} P[ls] * G[ml,ns]
        G_ex = G_AA.transpose(0, 2, 1, 3)
        K = np.einsum("ls,mnls->mn", P_AA, G_ex, optimize=True)
        F[idx, idx] += J - 0.5 * K

    # Two-center contributions
    npairs = len(pair_i)
    for k in range(npairs):
        iA = pair_i[k]
        iB = pair_j[k]
        i0A = offsets[iA]
        i0B = offsets[iB]
        naoA = nao_for_Z(atomic_numbers[iA])
        naoB = nao_for_Z(atomic_numbers[iB])
        idxA = slice(i0A, i0A + naoA)
        idxB = slice(i0B, i0B + naoB)

        W = W_list[k]
        P_BB = P[idxB, idxB]
        P_AA = P[idxA, idxA]

        # Coulomb: electrons on A feel density on B
        # F_AA[mu,nu] += sum_{lam,sig on B} P_BB[lam,sig] * W[mu,nu,lam,sig]
        J_to_A = np.einsum("ls,mnls->mn", P_BB, W, optimize=True)
        F[idxA, idxA] += J_to_A

        # Coulomb: electrons on B feel density on A
        # F_BB[lam,sig] += sum_{mu,nu on A} P_AA[mu,nu] * W[mu,nu,lam,sig]
        J_to_B = np.einsum("mn,mnls->ls", P_AA, W, optimize=True)
        F[idxB, idxB] += J_to_B

        # Exchange: off-diagonal block
        # F_AB[mu,lam] -= 0.5 * sum_{nu,sig} P_AB[nu,sig] * W[mu,nu,lam,sig]
        P_AB = P[idxA, idxB]
        # W[mu,nu,lam,sig] -> contract on nu(A) and sig(B) with P_AB[nu,sig]
        # K_AB[mu,lam] = sum_{nu,sig} P_AB[nu,sig] * W[mu,nu,lam,sig]
        K_AB = np.einsum("ns,mnls->ml", P_AB, W, optimize=True)
        F[idxA, idxB] -= 0.5 * K_AB
        F[idxB, idxA] -= 0.5 * K_AB.T

    return F
