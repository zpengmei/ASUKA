from __future__ import annotations

import numpy as np

from asuka.integrals.df_integrals import DFMOIntegrals
from asuka.cuguga.drt import DRT
from asuka.cuguga.oracle.sparse import connected_row_sparse_df
from asuka.cuguga.screening import RowScreening
from asuka.cuguga.state_cache import DRTStateCache


def matvec_df_row_oracle(
    drt: DRT,
    h1e: np.ndarray,
    df_eri: DFMOIntegrals,
    xs: list[np.ndarray],
    *,
    max_out: int = 200_000,
    screening: RowScreening | None = None,
    state_cache: DRTStateCache | None = None,
) -> list[np.ndarray]:
    """Compute y = H x using the DF sparse row oracle (column scan).

    This is an exact (thresholds=0) but Python-looped implementation intended as:
    - a correctness reference for future compiled backends (Numba/C++), and
    - a scalable-memory alternative to dense-intermediate contraction kernels.
    """

    ncsf = int(drt.ncsf)
    if ncsf < 1:
        raise ValueError("drt.ncsf must be >= 1")

    max_out = int(max_out)
    if max_out < 1:
        raise ValueError("max_out must be >= 1")

    screen = RowScreening() if screening is None else screening

    x_list: list[np.ndarray] = []
    for x in xs:
        arr = np.asarray(x, dtype=np.float64).ravel()
        if arr.size != ncsf:
            raise ValueError(f"x has wrong length: {arr.size} (expected {ncsf})")
        x_list.append(np.ascontiguousarray(arr))

    xmat = np.ascontiguousarray(np.stack(x_list, axis=0))  # (nvec, ncsf)
    nvec = int(xmat.shape[0])
    ymat = np.zeros((nvec, ncsf), dtype=np.float64)

    for j in range(ncsf):
        xj = xmat[:, j]
        if not np.any(xj):
            continue

        i_idx, hij = connected_row_sparse_df(
            drt,
            h1e,
            df_eri,
            int(j),
            max_out=max_out,
            screening=screen,
            state_cache=state_cache,
        )

        ymat[:, i_idx] += xj[:, None] * hij[None, :]

    return [np.ascontiguousarray(ymat[i]) for i in range(nvec)]
