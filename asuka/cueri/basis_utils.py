from __future__ import annotations

import numpy as np

from .basis_cart import BasisCartSoA
from .cart import ncart


def shell_nfunc_cart(basis) -> np.ndarray:
    """Return the number of Cartesian AO functions per shell for a packed basis.

    This function calculates the number of Cartesian functions for each shell, taking into
    account whether the shell is contracted or expanded.

    Parameters
    ----------
    basis : object
        A basis object containing shell information (`shell_l`).
        If the basis is contracted, it must also provide `shell_nctr`.

    Returns
    -------
    np.ndarray
        An array of integers representing the number of Cartesian functions for each shell.
        The shape corresponds to the number of shells in the basis.

    Notes
    -----
    - For expanded shells (nctr == 1), the number of functions is determined solely by angular momentum `l`.
    - For contracted shells, the number of functions is multiplied by `shell_nctr`.
    """

    shell_l = np.asarray(getattr(basis, "shell_l"), dtype=np.int64).ravel()
    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
    if hasattr(basis, "shell_nctr"):
        shell_nctr = np.asarray(getattr(basis, "shell_nctr"), dtype=np.int64).ravel()
        if shell_nctr.shape != shell_l.shape:
            raise ValueError("shell_nctr must have the same shape as shell_l")
        nfunc = nfunc * shell_nctr
    return nfunc


def expand_contracted_cart_basis(basis) -> BasisCartSoA:
    """Expand a contracted Cartesian basis into an expanded (nctr=1) view.

    This utility converts a contracted basis object (e.g., :class:`~cueri.basis_cart_contracted.BasisCartContractedSoA`)
    into an expanded form (:class:`~cueri.basis_cart.BasisCartSoA`) by replicating each shell
    explicitly for every contraction column.

    Parameters
    ----------
    basis : object
        The input basis object, expected to have `shell_nctr`, `shell_coef_start`,
        `prim_coef_flat`, and other standard basis attributes.

    Returns
    -------
    BasisCartSoA
        A new basis object where all shells are expanded (i.e., effective `nctr` is 1).

    Notes
    -----
    The expansion follows the ordering convention:
    - Expanded shells are ordered as (shell0, ctr0..ctr{nctr-1}), (shell1, ctr0..), etc.
    - The AO indices correspond to:
      ``ao_start_expanded = shell_ao_start[shell] + ctr * ncart(l)``
    """

    # Minimal structural validation (keep this helper fast; full checks live in dataclasses).
    if not hasattr(basis, "shell_nctr") or not hasattr(basis, "shell_coef_start") or not hasattr(basis, "prim_coef_flat"):
        raise TypeError("basis must provide shell_nctr/shell_coef_start/prim_coef_flat (contracted basis expected)")

    shell_cxyz = np.asarray(basis.shell_cxyz, dtype=np.float64, order="C")
    shell_prim_start = np.asarray(basis.shell_prim_start, dtype=np.int32).ravel()
    shell_nprim = np.asarray(basis.shell_nprim, dtype=np.int32).ravel()
    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start = np.asarray(basis.shell_ao_start, dtype=np.int32).ravel()
    shell_nctr = np.asarray(basis.shell_nctr, dtype=np.int32).ravel()
    shell_coef_start = np.asarray(basis.shell_coef_start, dtype=np.int32).ravel()

    prim_exp = np.asarray(basis.prim_exp, dtype=np.float64).ravel()
    prim_coef_flat = np.asarray(basis.prim_coef_flat, dtype=np.float64).ravel()

    n_shell = int(shell_l.size)
    if shell_cxyz.shape != (n_shell, 3):
        raise ValueError("shell_cxyz shape mismatch")
    if not (
        shell_prim_start.shape == shell_nprim.shape == shell_l.shape == shell_ao_start.shape == shell_nctr.shape == shell_coef_start.shape == (n_shell,)
    ):
        raise ValueError("contracted shell array shape mismatch")

    if n_shell == 0:
        return BasisCartSoA(
            shell_cxyz=np.empty((0, 3), dtype=np.float64),
            shell_prim_start=np.empty((0,), dtype=np.int32),
            shell_nprim=np.empty((0,), dtype=np.int32),
            shell_l=np.empty((0,), dtype=np.int32),
            shell_ao_start=np.empty((0,), dtype=np.int32),
            prim_exp=np.empty((0,), dtype=np.float64),
            prim_coef=np.empty((0,), dtype=np.float64),
            source_bas_id=np.empty((0,), dtype=np.int32),
            source_ctr_id=np.empty((0,), dtype=np.int32),
        )

    if np.any(shell_nprim < 0) or np.any(shell_nctr < 1) or np.any(shell_l < 0):
        raise ValueError("invalid shell_nprim/shell_nctr/shell_l values")

    n_shell_exp = int(np.sum(shell_nctr.astype(np.int64)))
    n_prim_exp = int(np.sum(shell_nprim.astype(np.int64) * shell_nctr.astype(np.int64)))

    shell_cxyz_exp = np.empty((n_shell_exp, 3), dtype=np.float64)
    shell_prim_start_exp = np.empty((n_shell_exp,), dtype=np.int32)
    shell_nprim_exp = np.empty((n_shell_exp,), dtype=np.int32)
    shell_l_exp = np.empty((n_shell_exp,), dtype=np.int32)
    shell_ao_start_exp = np.empty((n_shell_exp,), dtype=np.int32)

    prim_exp_exp = np.empty((n_prim_exp,), dtype=np.float64)
    prim_coef_exp = np.empty((n_prim_exp,), dtype=np.float64)

    # Optional bookkeeping
    bas_id_in = getattr(basis, "source_bas_id", None)
    bas_id_exp = None
    if bas_id_in is not None:
        bas_id_in = np.asarray(bas_id_in, dtype=np.int32).ravel()
        if bas_id_in.shape != (n_shell,):
            raise ValueError("source_bas_id shape mismatch")
        bas_id_exp = np.empty((n_shell_exp,), dtype=np.int32)
    ctr_id_exp = np.empty((n_shell_exp,), dtype=np.int32)

    out_sh = 0
    out_p = 0
    for sh in range(n_shell):
        l = int(shell_l[sh])
        nprim = int(shell_nprim[sh])
        nctr = int(shell_nctr[sh])

        exp0 = int(shell_prim_start[sh])
        exp1 = exp0 + nprim
        coef0 = int(shell_coef_start[sh])
        coef1 = coef0 + nprim * nctr

        exp_slice = prim_exp[exp0:exp1]
        coef_mat = prim_coef_flat[coef0:coef1].reshape((nprim, nctr), order="C")

        ao0 = int(shell_ao_start[sh])
        cxyz = shell_cxyz[sh]
        for ctr in range(nctr):
            shell_cxyz_exp[out_sh] = cxyz
            shell_prim_start_exp[out_sh] = np.int32(out_p)
            shell_nprim_exp[out_sh] = np.int32(nprim)
            shell_l_exp[out_sh] = np.int32(l)
            shell_ao_start_exp[out_sh] = np.int32(ao0 + int(ctr) * int(ncart(l)))
            if bas_id_exp is not None:
                bas_id_exp[out_sh] = bas_id_in[sh]
            ctr_id_exp[out_sh] = np.int32(ctr)

            prim_exp_exp[out_p : out_p + nprim] = exp_slice
            prim_coef_exp[out_p : out_p + nprim] = coef_mat[:, ctr]
            out_p += nprim
            out_sh += 1

    if out_sh != n_shell_exp or out_p != n_prim_exp:
        raise RuntimeError("internal error while expanding contracted basis")

    return BasisCartSoA(
        shell_cxyz=shell_cxyz_exp,
        shell_prim_start=shell_prim_start_exp,
        shell_nprim=shell_nprim_exp,
        shell_l=shell_l_exp,
        shell_ao_start=shell_ao_start_exp,
        prim_exp=prim_exp_exp,
        prim_coef=prim_coef_exp,
        source_bas_id=bas_id_exp,
        source_ctr_id=ctr_id_exp,
    )


__all__ = ["expand_contracted_cart_basis", "shell_nfunc_cart"]
