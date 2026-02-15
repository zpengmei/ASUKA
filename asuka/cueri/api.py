from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ActiveDFResult:
    """Result container for active-space Density Fitting (DF) decomposition.

    This immutable data structure holds the decomposition of the electron repulsion integrals (ERIs)
    in the active space, typically in the form `(pq|rs) ≈ sum_L (pq|L) (L|rs)`.

    Parameters
    ----------
    norb : int
        The number of active orbitals.
    naux : int
        The number of auxiliary basis functions (DF rank).
    l_full : np.ndarray | cupy.ndarray
        The 3-center integral tensor `(pq|L)` reshaped as a 2-D array `(norb*norb, naux)`.
        The row index is `pq = p * norb + q`.
    j_ps : np.ndarray | cupy.ndarray | None, optional
        Pre-computed J-matrix or related intermediates, if requested.
    eri_mat : np.ndarray | cupy.ndarray | None, optional
        The full reconstructed ERI matrix, if explicitly requested or built.
    pair_norm : np.ndarray | cupy.ndarray | None, optional
        Norm of the orbital pairs, used for screening or analysis.
    """

    norb: int
    naux: int
    l_full: Any
    j_ps: Any | None = None
    eri_mat: Any | None = None
    pair_norm: Any | None = None


def build_active_df(
    ao_basis,
    aux_basis,
    C_active,
    *,
    stream=None,
    backend: str = "gpu_rys",
    streamed: bool = True,
    **kwargs,
) -> ActiveDFResult:
    """Construct active-space Density Fitting (DF) vectors.

    This function builds the 3-center integrals `(pq|L)` for the specified active space
    using the Resolution of the Identity (RI) approximation. The result is returned as
    vectors in `(pq, L)` layout.

    Parameters
    ----------
    ao_basis : BasisCartSoA | object
        The atomic orbital basis set.
    aux_basis : BasisCartSoA | object
        The auxiliary basis set for density fitting.
    C_active : np.ndarray | cupy.ndarray
        Active space MO coefficients. Shape: `(nao, norb)`.
    stream : cupy.cuda.Stream | None, optional
        CUDA stream for asynchronous execution.
    backend : str, default='gpu_rys'
        The backend implementation to use (e.g., 'gpu_rys', 'gpu_direct').
    streamed : bool, default=True
        If True, uses a memory-efficient streamed construction algorithm.
        If False, uses a full-matrix construction strategy (higher memory usage).
    **kwargs : dict
        Additional arguments passed to the underlying backend functions.

    Returns
    -------
    ActiveDFResult
        A container holding the DF vectors (`l_full`) and metadata.

    Notes
    -----
    - The output vectors `l_full` are typically on the GPU (CuPy arrays).
    - Requires CuPy availability at runtime.
    """

    shape = getattr(C_active, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("C_active must have shape (nao, norb)")
    norb = int(shape[1])
    if norb <= 0:
        raise ValueError("C_active must have norb > 0")

    if streamed:
        from .df import active_Lfull_streamed_basis

        l_full = active_Lfull_streamed_basis(ao_basis, aux_basis, C_active, stream=stream, backend=backend, **kwargs)
    else:
        from .df import active_Lfull_from_B, cholesky_metric, int3c2e_basis, metric_2c2e_basis, whiten_3c2e

        V = metric_2c2e_basis(aux_basis, stream=stream, backend=backend)
        X = int3c2e_basis(ao_basis, aux_basis, stream=stream, backend=backend)
        L = cholesky_metric(V)
        B = whiten_3c2e(X, L)
        l_full = active_Lfull_from_B(B, C_active)

    l_shape = getattr(l_full, "shape", None)
    if l_shape is None or len(l_shape) != 2:
        raise ValueError("Expected l_full to be a 2D array with shape (norb*norb, naux)")
    if int(l_shape[0]) != int(norb) * int(norb):
        raise ValueError(f"Expected l_full.shape[0] == norb*norb ({norb*norb}), got {int(l_shape[0])}")
    naux = int(l_shape[1])

    return ActiveDFResult(norb=norb, naux=naux, l_full=l_full)
