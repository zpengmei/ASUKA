from __future__ import annotations

"""Shell-block pivoted Cholesky decomposition for the RICD generator.

Implements the shell-block CD described in the design document §7.4:
select shell blocks from an atomic candidate metric matrix using an
eigenvalue-based scoring criterion.  All operations are pure NumPy on
small, dense, one-center matrices.
"""

import numpy as np

from asuka.cueri.cart import ncart
from asuka.integrals.ricd_types import RICDShell, ShellBlock


# ---------------------------------------------------------------------------
# Shell-block layout
# ---------------------------------------------------------------------------

def build_shell_blocks(shells: list[RICDShell]) -> list[ShellBlock]:
    """Build the shell-block index layout from a list of RICD shells.

    Each shell becomes one block whose dimension is ``ncart(l)``.

    Parameters
    ----------
    shells : list[RICDShell]
        Candidate shells.

    Returns
    -------
    list[ShellBlock]
        Ordered shell blocks with cumulative offsets into the metric matrix.
    """
    blocks: list[ShellBlock] = []
    offset = 0
    for idx, sh in enumerate(shells):
        dim = ncart(sh.l)
        blocks.append(ShellBlock(shell_id=idx, l=sh.l, dim=dim, offset=offset))
        offset += dim
    return blocks


# ---------------------------------------------------------------------------
# Shell-block pivoted Cholesky
# ---------------------------------------------------------------------------

def _block_max_eigenvalue(R_ss: np.ndarray) -> float:
    """Largest eigenvalue of a small symmetric matrix (shell diagonal block).

    For 1×1 blocks this is simply the scalar value.  For larger blocks
    (d, f, …) we use ``np.linalg.eigvalsh`` which is robust for small
    dense matrices.
    """
    n = int(R_ss.shape[0])
    if n == 0:
        return 0.0
    if n == 1:
        return float(R_ss[0, 0])
    evals = np.linalg.eigvalsh(R_ss)
    return float(evals[-1])


def block_pivoted_cholesky(
    metric: np.ndarray,
    blocks: list[ShellBlock],
    threshold: float,
) -> list[int]:
    """Shell-block pivoted Cholesky selection on a candidate metric.

    Parameters
    ----------
    metric : np.ndarray
        Symmetric positive semi-definite candidate metric ``M^(c)``
        with shape ``(N, N)`` where ``N = sum(block.dim)``.
    blocks : list[ShellBlock]
        Shell-block layout (from :func:`build_shell_blocks`).
    threshold : float
        Stop when the largest shell score drops below this value.

    Returns
    -------
    list[int]
        Indices into *blocks* (and thus into the shell list) of selected
        shells, in selection order.
    """
    threshold = float(threshold)
    n_blocks = len(blocks)
    if n_blocks == 0:
        return []

    N = int(metric.shape[0])
    if N == 0:
        return []

    # Work on a copy so we can update the residual in place.
    R = np.array(metric, dtype=np.float64, order="C")
    # Symmetrize to guard against minor floating-point asymmetry.
    R = 0.5 * (R + R.T)

    selected: list[int] = []
    is_selected = [False] * n_blocks

    # Cholesky column storage: each selected block contributes a tall matrix.
    # W_columns[k] has shape (N, dim_k) for the k-th selected block.
    W_columns: list[np.ndarray] = []

    for _iteration in range(n_blocks):
        # --- Score each unselected block ---
        best_score = -1.0
        best_idx = -1
        for i in range(n_blocks):
            if is_selected[i]:
                continue
            bi = blocks[i]
            R_ii = R[bi.offset : bi.offset + bi.dim, bi.offset : bi.offset + bi.dim]
            score = _block_max_eigenvalue(R_ii)
            if score > best_score:
                best_score = score
                best_idx = i

        # --- Convergence check ---
        if best_score < threshold or best_idx < 0:
            break

        # --- Select block p ---
        p = best_idx
        bp = blocks[p]
        is_selected[p] = True
        selected.append(p)

        # --- Factor the diagonal block: R_pp = U_p U_p^T ---
        R_pp = R[bp.offset : bp.offset + bp.dim, bp.offset : bp.offset + bp.dim]
        R_pp = 0.5 * (R_pp + R_pp.T)

        # Add small regularization for numerical stability on tiny blocks.
        eps = max(1.0e-14, float(threshold) * 1.0e-10) * max(1.0, float(np.trace(R_pp)))
        R_pp_reg = R_pp + eps * np.eye(bp.dim, dtype=np.float64)

        try:
            U_p = np.linalg.cholesky(R_pp_reg)
        except np.linalg.LinAlgError:
            # Block is numerically zero — skip.
            is_selected[p] = False
            selected.pop()
            continue

        # --- Form column blocks W_sp = R_sp @ inv(U_p^T) for all s ---
        # W = R[:, p_cols] @ inv(U_p^T) = R[:, p_cols] @ inv(U_p).T
        # We solve U_p^T X^T = R_sp^T  ⟹ X = R_sp @ inv(U_p^T)
        p_slice = slice(bp.offset, bp.offset + bp.dim)
        R_col_p = R[:, p_slice]  # (N, dim_p)
        # Solve: W = R_col_p @ inv(U_p^T)
        #
        # To avoid explicitly forming an inverse, solve:
        #   U_p * X = R_col_p^T   (U_p is lower-triangular)
        # then W = X^T = R_col_p @ inv(U_p^T).
        #
        # NOTE: solving against U_p.T here would yield inv(U_p) (wrong),
        # breaking positive-semidefiniteness of the residual.
        W = np.linalg.solve(U_p, R_col_p.T).T  # (N, dim_p)

        # --- Update residual: R -= W @ W^T ---
        R -= W @ W.T

        # Store column for potential later use (not strictly needed after update).
        W_columns.append(W)

    return selected


__all__ = [
    "block_pivoted_cholesky",
    "build_shell_blocks",
]
