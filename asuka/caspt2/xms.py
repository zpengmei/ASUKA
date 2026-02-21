"""XMS-CASPT2: Extended Multi-State rotation of reference states.

Ports OpenMolcas ``xdwinit.f``.
Rotates the SA-CASSCF CI vectors to diagonalize the state-averaged
Fock operator in the model space before performing MS-CASPT2.
"""

from __future__ import annotations

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.cuguga.drt import DRT
from asuka.rdm.stream import trans_rdm1_all_streaming


def xms_rotate_states(
    drt: DRT,
    ci_vectors: list[np.ndarray],
    dm1_list: list[np.ndarray],
    fock: CASPT2Fock,
    nish: int,
    nash: int,
    nstates: int,
    *,
    verbose: int = 0,
    block_nops: int = 8,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """XMS state rotation.

    Rotates the SA-CASSCF CI vectors to diagonalize the state-averaged
    Fock operator projected onto the model space.

    Parameters
    ----------
    drt : DRT
        Active-space DRT used by transition-density builders.
    ci_vectors : list of arrays
        Original SA-CASSCF CI vectors (length nstates).
    dm1_list : list of arrays
        1-RDMs for each state.
    fock : CASPT2Fock
        Fock matrices (built with state-averaged density).
    nish, nash : int
        Orbital space dimensions.
    nstates : int
        Number of states.
    block_nops : int
        Block size for transition dm1 construction.

    Returns
    -------
    rotated_ci : list of arrays
        Rotated CI vectors.
    u0 : (nstates, nstates) array
        Rotation matrix.
    h0_model : (nstates, nstates) array
        Zeroth-order Hamiltonian in model space.
    """
    act = slice(nish, nish + nash)
    f_act = fock.fifa[act, act]

    # Build H0 in model space: H0[I,J] = <I|F_SA|J>.
    # Use transition 1-RDMs for all I,J to match xdwinit/FOPAB behavior.
    h0_model = np.zeros((nstates, nstates), dtype=np.float64)
    tdm1_adj = trans_rdm1_all_streaming(drt, ci_vectors, ci_vectors, block_nops=int(block_nops))
    # stream-trans dm1 convention is <bra|E_{qp}|ket>; CASPT2 tensors use <E_{pq}>.
    tdm1 = tdm1_adj.transpose(0, 1, 3, 2)

    for i in range(nstates):
        for j in range(i, nstates):
            # Fock contraction in active model space.
            val = float(np.einsum("pq,pq->", f_act, tdm1[i, j]))
            # Keep state-diagonal dm1 contraction as reference sanity path.
            if i == j:
                val = float(np.trace(f_act @ dm1_list[i]))

            h0_model[i, j] = val
            h0_model[j, i] = val

    # Add inactive orbital energy contribution
    e_inact = 0.0
    for ii in range(nish):
        e_inact += 2.0 * fock.fifa[ii, ii]

    for i in range(nstates):
        h0_model[i, i] += e_inact

    # Diagonalize H0 model space
    eigenvalues, u0 = np.linalg.eigh(h0_model)

    if verbose >= 1:
        print(f"XMS rotation:")
        print(f"  H0 eigenvalues: {eigenvalues}")

    # Rotate CI vectors: C'_J = sum_I U0[I,J] * C_I
    rotated_ci = []
    for j in range(nstates):
        ci_new = np.zeros_like(ci_vectors[0])
        for i in range(nstates):
            ci_new += u0[i, j] * ci_vectors[i]
        rotated_ci.append(ci_new)

    return rotated_ci, u0, h0_model
