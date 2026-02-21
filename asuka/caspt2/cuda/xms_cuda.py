from __future__ import annotations

import time
from typing import Any

import numpy as np

from asuka.caspt2.fock import CASPT2Fock
from asuka.cuguga.drt import DRT


def xms_rotate_states_cuda(
    drt: DRT,
    ci_vectors: list[np.ndarray],
    dm1_list: list[Any],
    fock: CASPT2Fock,
    nish: int,
    nash: int,
    nstates: int,
    *,
    device: int | None = None,
    verbose: int = 0,
    build_threads: int = 256,
    profile: dict[str, float] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """CUDA-backed XMS state rotation.

    This is a drop-in analogue of :func:`asuka.caspt2.xms.xms_rotate_states`, but
    replaces the CPU `trans_rdm1_all_streaming` call with the GPU EPQ-table path
    (:func:`asuka.cuda.rdm_gpu.trans_rdm1_all_cuda`).

    Notes
    -----
    - The expensive transition-dm1 construction runs on GPU.
    - The resulting H0 model-space build and diagonalization are small and run on CPU.
    """

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for xms_rotate_states_cuda") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    from asuka.cuda.rdm_gpu import trans_rdm1_all_cuda  # noqa: PLC0415

    nstates = int(nstates)
    if int(len(ci_vectors)) != nstates:
        raise ValueError("ci_vectors length mismatch with nstates")
    if int(len(dm1_list)) != nstates:
        raise ValueError("dm1_list length mismatch with nstates")

    act = slice(int(nish), int(nish) + int(nash))
    f_act = np.asarray(fock.fifa[act, act], dtype=np.float64, order="C")

    t0 = time.perf_counter()
    # Convention matches trans_rdm1_all_streaming:
    #   dm1_adj[bra,ket,p,q] = <bra|E_{q p}|ket>
    tdm1_adj = trans_rdm1_all_cuda(
        drt,
        ci_vectors,
        build_threads=int(build_threads),
        use_epq_table=True,
    )
    if profile is not None:
        profile["xms_tdm1_all_cuda_s"] = float(time.perf_counter() - t0)

    # stream-trans dm1 convention is <bra|E_{q p}|ket>; CASPT2 tensors use <E_{p q}>.
    tdm1 = np.asarray(tdm1_adj, dtype=np.float64).transpose(0, 1, 3, 2)

    # Build H0 in model space: H0[I,J] = <I|F_SA|J>.
    h0_model = np.einsum("pq,abpq->ab", f_act, tdm1, optimize=True)
    h0_model = np.asarray(h0_model, dtype=np.float64, order="C")

    # Keep state-diagonal dm1 contraction as reference sanity path (matches CPU).
    f_act_d = cp.asarray(f_act, dtype=cp.float64)
    for i in range(nstates):
        dm1_i = dm1_list[i]
        if hasattr(dm1_i, "dtype"):
            # CuPy array: small, but avoid bringing it to host if possible.
            try:
                dm1_i = cp.asarray(dm1_i, dtype=cp.float64)
                h0_model[i, i] = float(cp.trace(f_act_d @ dm1_i).get())
                continue
            except Exception:
                pass
        h0_model[i, i] = float(np.trace(f_act @ np.asarray(dm1_i, dtype=np.float64)))

    # Add inactive orbital energy contribution.
    e_inact = float(2.0 * np.sum(np.diag(np.asarray(fock.fifa[: int(nish), : int(nish)], dtype=np.float64))))
    for i in range(nstates):
        h0_model[i, i] += e_inact

    # Diagonalize H0 model space.
    evals, u0 = np.linalg.eigh(h0_model)
    if verbose >= 1:
        print("XMS rotation (CUDA dm1 backend):")
        print(f"  H0 eigenvalues: {evals}")

    # Rotate CI vectors: C'_J = sum_I U0[I,J] * C_I
    rotated_ci: list[np.ndarray] = []
    for j in range(nstates):
        ci_new = np.zeros_like(ci_vectors[0])
        for i in range(nstates):
            ci_new += float(u0[i, j]) * np.asarray(ci_vectors[i], dtype=np.float64)
        rotated_ci.append(np.asarray(ci_new, dtype=np.float64))

    return rotated_ci, np.asarray(u0, dtype=np.float64), h0_model
