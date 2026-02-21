"""MS-CASPT2: Multi-State CASPT2 effective Hamiltonian.

Ports OpenMolcas ``hcoup.f``, ``hefval.F90``, ``mltctl.f``.
Constructs the effective Hamiltonian Heff from SS-CASPT2 results
for multiple states, then diagonalizes to get MS-CASPT2 energies.

The effective Hamiltonian is defined as:

    Heff[I,J] = δ(I,J) · E_CASPT2(I)  +  (1 - δ(I,J)) · ⟨I|H|Ω_J⟩

where |Ω_J⟩ is the first-order wavefunction for state J.  The off-diagonal
couplings ⟨I|H|Ω_J⟩ are evaluated by contracting the ket-state J's RHS and
solution vectors (in the raw active superindex basis) against transition
densities (TG1/TG2/TG3) between states I and J, using per-case HCOUP kernels.

The workflow is:
  1. Run SS-CASPT2 for each state → amplitudes + per-case decompositions
  2. Back-transform amplitudes to the raw (un-diagonalized) active superindex
     basis: T_raw = transform @ T_SR
  3. Precompute row_dots[ias,jas] = V1[ias,:] · V2[jas,:] (RHS × solution)
  4. For each off-diagonal (I,J) pair, compute transition dm1/dm2/dm3 and
     contract with row_dots via hcoup_case_contribution()
  5. Diagonalize Heff to obtain MS-CASPT2 energies
"""

from __future__ import annotations

import numpy as np
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hcoup import hcoup_case_contribution
from asuka.caspt2.hzero import build_bmat
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.result import CASPT2EnergyResult
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.superindex import SuperindexMap
from asuka.rdm.rdm123 import _trans_rdm123_pyscf


def build_heff(
    nstates: int,
    ss_results: list[CASPT2EnergyResult],
    ci_vectors: list[np.ndarray],
    drt,
    smap: SuperindexMap,
    fock: CASPT2Fock | list[CASPT2Fock],
    eri_mo: np.ndarray,
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    *,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    block_nops: int = 8,
    max_memory_mb: float = 4000.0,
    verbose: int = 0,
) -> np.ndarray:
    """Build MS-CASPT2 effective Hamiltonian.

    Heff[I,J] = δ(I,J) * E_CASPT2(I) + (1-δ(I,J)) * <I|H|Ω_J>

    where Ω_J is the first-order wavefunction for state J.

    Parameters
    ----------
    nstates : int
        Number of states.
    ss_results : list of CASPT2EnergyResult
        SS-CASPT2 results for each state.
    ci_vectors : list of arrays
        CI coefficient vectors for each state.
    drt : DRT
        Active-space DRT (needed for transition-density couplings).
    dm1_list, dm2_list, dm3_list : list of arrays
        RDMs for each state.
    threshold : float
        S-metric diagonal threshold (Molcas THRSHN).
    threshold_s : float
        S-metric scaled-eigenvalue threshold (Molcas THRSHS).
    block_nops : int
        Kept for backward API compatibility (unused in current no-proxy path).
    max_memory_mb : float
        Memory cap forwarded to transition dm123 builder.

    Returns
    -------
    heff : (nstates, nstates)
        Effective Hamiltonian matrix.
    """
    heff = np.zeros((nstates, nstates), dtype=np.float64)

    def _fock_for_state(state: int) -> CASPT2Fock:
        if isinstance(fock, (list, tuple)):
            return fock[state]
        return fock

    # Diagonal: SS-CASPT2 energies
    for i in range(nstates):
        heff[i, i] = ss_results[i].e_tot

    # ``block_nops`` is kept for API compatibility (old dm1-stream path).
    _ = int(block_nops)
    nactel = int(getattr(drt, "nelec"))

    # Build per-state (case-wise) orthonormal transforms and raw vectors.
    # We store:
    #   rhs_raw_by_state[i][case] : H|I> block in active superindex basis
    #   t_raw_by_state[j][case]   : |Omega_J> block back-transformed to active basis
    # so we can apply transition-metric kernels directly (Molcas HCOUP style).
    rhs_raw_by_state: list[list[np.ndarray]] = []
    t_raw_by_state: list[list[np.ndarray]] = []
    for j in range(nstates):
        fock_j = _fock_for_state(j)
        ci_context = CASPT2CIContext(drt=drt, ci_csf=ci_vectors[j])
        decomp_j: list[SBDecomposition] = []
        rhs_j: list[np.ndarray] = []
        t_raw_j: list[np.ndarray] = []
        for case in range(1, 14):
            nasup = int(smap.nasup[case - 1])
            nisup = int(smap.nisup[case - 1])
            if nasup == 0 or nisup == 0:
                decomp_j.append(
                    SBDecomposition(
                        s_eigvals=np.empty(0, dtype=np.float64),
                        transform=np.empty((0, 0), dtype=np.float64),
                        nindep=0,
                        b_diag=np.empty(0, dtype=np.float64),
                    )
                )
                rhs_j.append(np.empty((0, 0), dtype=np.float64))
                t_raw_j.append(np.empty((0, 0), dtype=np.float64))
                continue
            smat = build_smat(case, smap, dm1_list[j], dm2_list[j], dm3_list[j])
            bmat = build_bmat(case, smap, fock_j, dm1_list[j], dm2_list[j], dm3_list[j], ci_context=ci_context)
            decomp_case = sbdiag(
                smat, bmat, threshold_norm=threshold, threshold_s=threshold_s
            )
            decomp_j.append(decomp_case)

            rhs_raw = build_rhs(
                case,
                smap,
                fock_j,
                eri_mo,
                dm1_list[j],
                dm2_list[j],
                nactel=nactel,
            ).reshape(nasup, nisup)
            rhs_j.append(rhs_raw)

            amps_j = ss_results[j].amplitudes[case - 1]
            if decomp_case.nindep == 0 or amps_j.size == 0:
                t_raw_j.append(np.empty((0, 0), dtype=np.float64))
            else:
                expected_size = int(decomp_case.nindep) * int(nisup)
                if int(amps_j.size) != expected_size:
                    raise ValueError(
                        "MS Heff amplitude dimension mismatch for "
                        f"state {j} case {case}: amps {amps_j.size} vs expected {expected_size}"
                    )
                amps_mat = amps_j.reshape(decomp_case.nindep, nisup)
                t_raw_j.append(np.asarray(decomp_case.transform @ amps_mat, dtype=np.float64, order="C"))

        rhs_raw_by_state.append(rhs_j)
        t_raw_by_state.append(t_raw_j)

    for i in range(nstates):
        for j in range(nstates):
            if i == j:
                continue
            ovl = float(np.dot(ci_vectors[i], ci_vectors[j]))
            tdm1_ij, tdm2_ij, tdm3_ij = _trans_rdm123_pyscf(
                drt,
                ci_vectors[i],
                ci_vectors[j],
                max_memory_mb=float(max_memory_mb),
                reorder=True,
                reorder_mode="molcas",
            )
            # Transition and diagonal state densities both use Molcas normal-order
            # conventions (including the transition-specific TG2/TG3 symmetrization).
            coupling = 0.0
            for case in range(1, 14):
                # OpenMolcas `HCOUP` evaluates <I|H|Omega_J> using the ket-state
                # RHS/solution vectors and a transition-density kernel (TG1/2/3)
                # between (I,J). See `hefval.F90` -> `hcoup.f`.
                rhs_j = rhs_raw_by_state[j][case - 1]
                t_j = t_raw_by_state[j][case - 1]
                if rhs_j.size == 0 or t_j.size == 0:
                    continue

                if rhs_j.shape != t_j.shape:
                    raise ValueError(
                        "MS Heff raw block shape mismatch for "
                        f"state-pair ({i},{j}) case {case}: rhs {rhs_j.shape} vs t {t_j.shape}"
                    )

                row_dots = rhs_j @ t_j.T
                coupling += hcoup_case_contribution(
                    case, smap, row_dots, tdm1_ij, tdm2_ij, tdm3_ij, ovl=ovl
                )

            heff[i, j] = coupling

    if verbose >= 1:
        print("MS-CASPT2 Heff:")
        for i in range(nstates):
            print(f"  Heff[{i},{i}] = {heff[i, i]:.10f}")
        if nstates > 1:
            for i in range(nstates):
                for j in range(nstates):
                    if i != j:
                        print(f"  Heff[{i},{j}] = {heff[i, j]:.10f}")

    return heff


def build_heff_coupling(
    ivec: int,
    jvec: int,
    ci_i: np.ndarray,
    ci_j: np.ndarray,
    t_amps_j: list[np.ndarray],
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    trans_dm1: np.ndarray,
    trans_dm2: np.ndarray,
    *,
    drt=None,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    verbose: int = 0,
) -> float:
    """Compute off-diagonal Heff coupling element <I|H|Ω_J>.

    Uses transition RDMs between states I and J.

    Parameters
    ----------
    ivec, jvec : int
        State indices.
    ci_i, ci_j : arrays
        CI vectors for states I and J.
    t_amps_j : list of arrays
        T amplitudes for state J.
    trans_dm1, trans_dm2 : arrays
        Transition 1-RDM and 2-RDM between states I and J.

    Returns
    -------
    coupling : float
        Heff[I,J] coupling element.
    """
    if drt is None:
        raise ValueError("build_heff_coupling requires drt for active-electron count.")

    nactel = int(getattr(drt, "nelec"))
    nash = int(smap.orbs.nash)
    dm2_dummy = np.zeros((nash, nash, nash, nash), dtype=np.float64)
    ci_context = CASPT2CIContext(drt=drt, ci_csf=ci_j)

    coupling = 0.0
    for case in range(1, 14):
        amps = t_amps_j[case - 1]
        if amps.size == 0:
            continue
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])
        if nasup == 0 or nisup == 0:
            continue

        smat = build_smat(case, smap, trans_dm1, dm2_dummy, np.zeros((nash, nash, nash, nash, nash, nash)))
        bmat = build_bmat(case, smap, fock, trans_dm1, dm2_dummy, np.zeros((nash, nash, nash, nash, nash, nash)), ci_context=ci_context)
        decomp = sbdiag(smat, bmat, threshold_norm=threshold, threshold_s=threshold_s)
        if decomp.nindep == 0:
            continue

        rhs_raw = build_rhs(case, smap, fock, eri_mo, trans_dm1, dm2_dummy, nactel=nactel)
        rhs_mat = rhs_raw.reshape(nasup, nisup)
        rhs_vec = (decomp.transform.T @ rhs_mat).ravel()
        if rhs_vec.size != amps.size:
            raise ValueError(
                f"Coupling size mismatch in case {case}: rhs {rhs_vec.size} vs amps {amps.size}"
            )
        coupling += float(np.dot(rhs_vec, amps))

    return coupling


def diagonalize_heff(heff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the effective Hamiltonian.

    Returns
    -------
    eigenvalues : (nstates,) sorted ascending
    eigenvectors : (nstates, nstates) columns are eigenvectors
    """
    heff = 0.5 * (heff + heff.T)
    eigenvalues, eigenvectors = np.linalg.eigh(heff)
    return eigenvalues, eigenvectors
