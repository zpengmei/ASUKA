from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT


@dataclass(frozen=True)
class DRTSubspaceMap:
    """Index map between a subspace DRT and a full-space DRT with identical orbitals/quantum numbers.

    Attributes
    ----------
    drt_full : DRT
        Full-space Discrete Reaction Field.
    drt_sub : DRT
        Subspace Discrete Reaction Field.
    sub_to_full : np.ndarray
        Mapping array from subspace index to full-space index. Shape: (nsub,).
    """

    drt_full: DRT
    drt_sub: DRT
    sub_to_full: np.ndarray  # (nsub,), int64

    @property
    def nfull(self) -> int:
        return int(self.drt_full.ncsf)

    @property
    def nsub(self) -> int:
        return int(self.drt_sub.ncsf)


def build_subspace_map(*, drt_full: DRT, drt_sub: DRT) -> DRTSubspaceMap:
    """Build a CSF index map from ``drt_sub`` to ``drt_full`` by matching CSF paths.

    Parameters
    ----------
    drt_full : DRT
        Full-space DRT.
    drt_sub : DRT
        Subspace DRT.

    Returns
    -------
    DRTSubspaceMap
        Mapping object containing the index map.

    Raises
    ------
    ValueError
        If DRT parameters (norb, nelec, twos) do not match.
    """

    if int(drt_full.norb) != int(drt_sub.norb):
        raise ValueError("drt_full and drt_sub must have the same norb")
    if int(drt_full.nelec) != int(drt_sub.nelec):
        raise ValueError("drt_full and drt_sub must have the same nelec")
    if int(drt_full.twos_target) != int(drt_sub.twos_target):
        raise ValueError("drt_full and drt_sub must have the same twos_target")

    nsub = int(drt_sub.ncsf)
    sub_to_full = np.empty(nsub, dtype=np.int64)
    for j in range(nsub):
        steps = drt_sub.index_to_path(int(j))
        sub_to_full[j] = int(drt_full.path_to_index(steps))
    return DRTSubspaceMap(drt_full=drt_full, drt_sub=drt_sub, sub_to_full=sub_to_full)


def projected_contract_h_csf_multi(
    *,
    mapping: DRTSubspaceMap,
    h1e: Any,
    eri: Any,
    xs_sub: list[np.ndarray] | np.ndarray,
    precompute_epq_full: bool = True,
    nthreads: int = 1,
    blas_nthreads: int | None = 1,
    executor: ThreadPoolExecutor | None = None,
    workspace: Any | None = None,
) -> list[np.ndarray]:
    """Apply the exact projected Hamiltonian Y = P H_full P X in a truncated DRT space.

    This is an exact-but-expensive reference implementation:
      - Embed X_sub into the full DRT basis.
      - Apply H in the full space using existing contraction backends.
      - Project back by gathering the subspace CSF indices.

    Parameters
    ----------
    mapping : DRTSubspaceMap
        Subspace-to-fullspace mapping.
    h1e : Any
        One-electron integrals.
    eri : Any
        Two-electron integrals.
    xs_sub : list[np.ndarray] | np.ndarray
        List of subspace vectors or a matrix.
    precompute_epq_full : bool, optional
        Precompute EPQ actions in the full space.
    nthreads : int, optional
        Number of threads for contraction.
    blas_nthreads : int | None, optional
        Number of BLAS threads.
    executor : ThreadPoolExecutor | None, optional
        Executor for parallel contraction.
    workspace : Any | None, optional
        Workspace for contraction.

    Returns
    -------
    list[np.ndarray]
        List of result vectors in the subspace.
    """

    drt_full = mapping.drt_full
    nfull = int(drt_full.ncsf)
    nsub = int(mapping.nsub)

    xmat = np.asarray(xs_sub, dtype=np.float64)
    if xmat.ndim == 1:
        xmat = xmat.reshape(1, -1)
    if xmat.ndim != 2 or int(xmat.shape[1]) != nsub:
        raise ValueError(f"xs_sub has wrong shape: {xmat.shape} (expected (*, {nsub}))")

    nvec = int(xmat.shape[0])
    x_full = np.zeros((nvec, nfull), dtype=np.float64)
    x_full[:, np.asarray(mapping.sub_to_full, dtype=np.int64)] = xmat

    try:
        from asuka.integrals.df_integrals import DFMOIntegrals  # noqa: PLC0415
    except Exception:  # pragma: no cover
        DFMOIntegrals = None  # type: ignore[assignment]

    if DFMOIntegrals is not None and isinstance(eri, DFMOIntegrals):
        from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

        y_full_list = _contract_df(
            drt_full,
            h1e,
            eri,
            x_full,
            precompute_epq=bool(precompute_epq_full),
            nthreads=int(nthreads),
            blas_nthreads=blas_nthreads,
            executor=executor,
            workspace=workspace,
        )
    else:
        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        y_full_list = _contract_dense(
            drt_full,
            h1e,
            eri,
            x_full,
            precompute_epq=bool(precompute_epq_full),
            nthreads=int(nthreads),
            blas_nthreads=blas_nthreads,
            executor=executor,
            workspace=workspace,
        )

    idx = np.asarray(mapping.sub_to_full, dtype=np.int64)
    out: list[np.ndarray] = []
    for y_full in y_full_list:
        y_full = np.asarray(y_full, dtype=np.float64).ravel()
        if int(y_full.size) != nfull:
            raise RuntimeError("contract backend returned a vector with wrong length")
        out.append(np.ascontiguousarray(y_full[idx], dtype=np.float64))
    return out
