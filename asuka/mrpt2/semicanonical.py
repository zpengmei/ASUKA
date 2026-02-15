from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SemicanonicalCoreVirt:
    """Semicanonicalization result for core/virtual orbitals."""

    mo_core: np.ndarray
    mo_virt: np.ndarray
    eps_core: np.ndarray
    eps_virt: np.ndarray
    u_core: np.ndarray
    u_virt: np.ndarray


def _cluster_degenerate_eigenvalues(eps: np.ndarray, *, tol: float) -> list[list[int]]:
    eps = np.asarray(eps, dtype=np.float64).ravel()
    if eps.size == 0:
        return []
    tol = float(tol)
    if tol <= 0.0:
        return [[i] for i in range(int(eps.size))]

    clusters: list[list[int]] = []
    cur: list[int] = [0]
    for i in range(1, int(eps.size)):
        if abs(float(eps[i]) - float(eps[i - 1])) <= tol:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    return clusters


def _rotate_degenerate_eigvecs_toward_identity(eps: np.ndarray, u: np.ndarray, *, tol: float) -> np.ndarray:
    """Fix gauge freedom in (near-)degenerate eigenspaces by choosing a basis close to identity.

    When `eps` contains (near-)degenerate values, the corresponding columns of `u` are not unique.
    For reproducibility (and for cross-code comparisons), rotate each degenerate block to maximize
    its overlap with the original orbital basis (i.e., make the block of `u` as close to identity
    as possible in Frobenius norm).
    """

    eps = np.asarray(eps, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] != eps.size:
        raise ValueError("eigensystem dimension mismatch")

    clusters = _cluster_degenerate_eigenvalues(eps, tol=float(tol))
    if not clusters:
        return u

    u_out = np.asarray(u, dtype=np.float64, order="C").copy()
    for idx in clusters:
        if len(idx) <= 1:
            continue
        sel = np.ix_(idx, idx)
        a = np.asarray(u_out[sel], dtype=np.float64, order="C")
        # Orthogonal Procrustes to best match `a @ r ≈ I` (maximize Tr(a @ r)).
        uu, _s, vt = np.linalg.svd(a, full_matrices=False)
        r = np.asarray(vt.T @ uu.T, dtype=np.float64, order="C")
        u_out[:, idx] = np.asarray(u_out[:, idx] @ r, dtype=np.float64, order="C")
    return u_out


def _molcas_jacob_eigh(a: np.ndarray, *, max_sweeps: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a symmetric matrix using OpenMolcas' Jacobi routine (misc_util/jacob.F90).

    OpenMolcas uses a Jacobi diagonalizer (`JACOB`) for small Fock blocks. The sequence of Jacobi
    rotations fixes the gauge inside (near-)degenerate eigenspaces differently from LAPACK.
    For parity/debugging with OpenMolcas, this can matter.

    Notes
    -----
    - This is intended for **small** matrices (active blocks, small core/virtual blocks).
    - For large blocks, prefer `np.linalg.eigh`.
    """

    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("a must be a square 2D array")
    n = int(a.shape[0])
    if n <= 0:
        raise ValueError("a must be non-empty")
    if n == 1:
        return np.asarray([float(a[0, 0])], dtype=np.float64), np.eye(1, dtype=np.float64)

    # Pack lower triangle in Molcas row-wise order:
    #   ARRAY(k) = A(i,j) for i=1..n, j=1..i (Fortran 1-based).
    tri = lambda i: int(i * (i + 1) // 2)  # noqa: E731
    arr = np.zeros(n * (n + 1) // 2, dtype=np.float64)
    for i in range(n):
        base = tri(i)
        arr[base : base + i + 1] = a[i, : i + 1]

    vecs = np.eye(n, dtype=np.float64)

    # OpenMolcas applies a diagonal shift:
    #   SHIFT = 0.5*(A11 + Ann)
    shift = 0.5 * (float(arr[tri(0) + 0]) + float(arr[tri(n - 1) + (n - 1)]))
    for i in range(n):
        arr[tri(i) + i] -= shift

    eps = 1.0e-16
    eps2 = 1.0e-30
    max_sweeps = int(max_sweeps)
    if max_sweeps <= 0:
        raise ValueError("max_sweeps must be > 0")

    for _nsweep in range(max_sweeps):
        nr = 0
        subdac = 0.0
        nsubd = 0

        # Sweep over subdiagonal elements in Molcas order: i=2..n, j=1..i-1 (Fortran).
        for i in range(1, n):
            ii = tri(i)
            for j in range(0, i):
                jj = tri(j)

                arij = float(arr[ii + j])
                aaij = abs(arij)
                arii = float(arr[ii + i])
                arjj = float(arr[jj + j])

                diff = arii - arjj
                sgn = 1.0
                if diff < 0.0:
                    diff = -diff
                    sgn = -1.0

                subdac += aaij
                nsubd += 1

                # Molcas skip heuristics.
                if float(nsubd) * aaij <= 0.5 * subdac:
                    continue
                if aaij <= eps * diff:
                    continue
                if aaij <= eps2:
                    continue

                nr += 1

                dum = diff + float(np.sqrt(diff * diff + 4.0 * aaij * aaij))
                tn = 2.0 * sgn * arij / dum
                cs = 1.0 / float(np.sqrt(1.0 + tn * tn))
                sn = cs * tn

                # 1) k = 1..j-1 (0-based: 0..j-1): update A(i,k) and A(j,k).
                for k in range(0, j):
                    aik = float(arr[ii + k])
                    ajk = float(arr[jj + k])
                    arr[ii + k] = sn * ajk + cs * aik
                    arr[jj + k] = cs * ajk - sn * aik

                # 2) k = j+1..i-1 (0-based: j+1..i-1): update A(i,k) and A(k,j).
                for k in range(j + 1, i):
                    kk = tri(k)
                    akj = float(arr[kk + j])
                    aik = float(arr[ii + k])
                    arr[kk + j] = cs * akj - sn * aik
                    arr[ii + k] = sn * akj + cs * aik

                # 3) k = i+1..n (0-based: i+1..n-1): update A(k,j) and A(k,i).
                for k in range(i + 1, n):
                    kk = tri(k)
                    akj = float(arr[kk + j])
                    aki = float(arr[kk + i])
                    arr[kk + j] = cs * akj - sn * aki
                    arr[kk + i] = sn * akj + cs * aki

                # Update diagonal and zero A(i,j).
                temp = 2.0 * cs * sn * arij
                cs2 = cs * cs
                sn2 = sn * sn
                arr[jj + j] = sn2 * arii + cs2 * arjj - temp
                arr[ii + j] = 0.0
                arr[ii + i] = cs2 * arii + sn2 * arjj + temp

                # Accumulate eigenvectors: VECS <- VECS * G.
                vj = vecs[:, j].copy()
                vi = vecs[:, i].copy()
                vecs[:, i] = sn * vj + cs * vi
                vecs[:, j] = cs * vj - sn * vi

        if nr == 0:
            break
    else:
        raise RuntimeError(f"Molcas JACOB failed to converge after {max_sweeps} sweeps")

    # Restore the shift on the diagonal.
    for i in range(n):
        arr[tri(i) + i] += shift

    evals = np.asarray([float(arr[tri(i) + i]) for i in range(n)], dtype=np.float64)
    return evals, np.asarray(vecs, dtype=np.float64, order="C")


def _molcas_diafck_reorder(eps: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mimic OpenMolcas `DIAFCK` post-processing of Fock eigenvectors.

    OpenMolcas (`src/caspt2/diafck.f`) diagonalizes each diagonal Fock block, then:
      1) Greedily permutes eigenvectors to maximize |u[i,i]| for i=1..n (row-wise pivoting).
      2) Flips eigenvector signs so that u[i,i] >= 0 for all i.

    This fixes gauge freedom in (near-)degenerate eigenspaces in a Molcas-compatible way and
    makes cross-code comparisons deterministic.
    """

    eps = np.asarray(eps, dtype=np.float64).ravel().copy()
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] != eps.size:
        raise ValueError("eigensystem dimension mismatch")
    u_out = np.asarray(u, dtype=np.float64, order="C").copy()

    n = int(eps.size)
    for i in range(n):
        # Find column j>=i that maximizes |u[i,j]|.
        j_rel = int(np.argmax(np.abs(u_out[i, i:])))
        j = i + j_rel
        if j != i:
            u_out[:, [i, j]] = u_out[:, [j, i]]
            eps[[i, j]] = eps[[j, i]]
        if float(u_out[i, i]) < 0.0:
            u_out[:, i] *= -1.0
    return eps, u_out


def molcas_diafck_eigh(
    f_block: np.ndarray,
    *,
    max_sweeps: int = 10_000,
    jacob_max_n: int = 32,
    deg_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize a Molcas Fock block and apply DIAFCK post-processing.

    For small blocks (default: n<=32), use a Molcas-compatible Jacobi diagonalizer to
    better match OpenMolcas' gauge choice inside degenerate eigenspaces. For larger
    blocks, fall back to `np.linalg.eigh` for performance.
    """

    f_block = np.asarray(f_block, dtype=np.float64)
    if f_block.ndim != 2 or f_block.shape[0] != f_block.shape[1]:
        raise ValueError("f_block must be square")
    f_block = 0.5 * (f_block + f_block.T)

    n = int(f_block.shape[0])
    if n <= int(jacob_max_n):
        eps, u = _molcas_jacob_eigh(f_block, max_sweeps=int(max_sweeps))
    else:
        eps, u = np.linalg.eigh(f_block)
        # LAPACK is free to rotate degenerate eigenspaces arbitrarily; choose a stable gauge
        # before the DIAFCK pivot/sign step to reduce cross-run/cross-code noise.
        u = _rotate_degenerate_eigvecs_toward_identity(eps, u, tol=float(deg_tol))

    eps, u = _molcas_diafck_reorder(eps, u)
    return np.asarray(eps, dtype=np.float64, order="C").ravel(), np.asarray(u, dtype=np.float64, order="C")


def _build_cart2sph_ao_matrix(mol: Any) -> np.ndarray | None:
    """Build the full AO Cartesian→spherical transformation matrix.

    Returns *None* when the molecule already uses Cartesian GTOs (no transform needed).
    """
    if getattr(mol, "cart", False):
        return None

    from asuka.cueri.sph import cart2sph_matrix  # noqa: PLC0415
    from scipy.linalg import block_diag  # noqa: PLC0415

    blocks: list[np.ndarray] = []
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        nc = mol.bas_nctr(ib)
        c2s = cart2sph_matrix(l)  # (ncart, nsph)
        for _ in range(nc):
            blocks.append(c2s)
    return block_diag(*blocks)  # (nao_cart, nao_sph)


def build_hcore_ao(mol: Any) -> np.ndarray:
    """Build core Hamiltonian (T+V) using ASUKA's native 1e integrals.

    Handles both Cartesian and spherical-harmonic basis sets.
    """
    from asuka.integrals.int1e_cart import build_int1e_cart  # noqa: PLC0415
    from asuka.cueri.mol_basis import pack_cart_shells_from_mol  # noqa: PLC0415

    # Build a Cartesian-mode copy of mol if needed
    mol_cart = mol
    if not getattr(mol, "cart", False):
        mol_cart = mol.copy()
        mol_cart.cart = True
        mol_cart.build(dump_input=False, parse_arg=False)

    basis = pack_cart_shells_from_mol(mol_cart)
    coords = np.asarray(mol_cart.atom_coords(unit="Bohr"), dtype=np.float64)
    charges = np.asarray(mol_cart.atom_charges(), dtype=np.float64)
    int1e = build_int1e_cart(basis, atom_coords_bohr=coords, atom_charges=charges)
    hcore_cart = np.asarray(int1e.hcore, dtype=np.float64)

    c2s = _build_cart2sph_ao_matrix(mol)
    if c2s is None:
        return hcore_cart
    return c2s.T @ hcore_cart @ c2s


def build_vhf_df(dm: np.ndarray, *, b_ao: np.ndarray) -> np.ndarray:
    """Build VHF = J - 0.5*K from precomputed DF/Cholesky AO factors."""
    dm = np.asarray(dm, dtype=np.float64)
    b_ao = np.asarray(b_ao, dtype=np.float64)
    rho = np.einsum("mnQ,mn->Q", b_ao, dm, optimize=True)
    j = np.einsum("mnQ,Q->mn", b_ao, rho, optimize=True)
    k = np.einsum("mrQ,rs,nsQ->mn", b_ao, dm, b_ao, optimize=True)
    return j - 0.5 * k


def build_generalized_fock_ao(
    mol: Any,
    *,
    mo_core: np.ndarray,
    mo_act: np.ndarray,
    dm1_act: np.ndarray,
    auxbasis: Any = "weigend+etb",
    max_memory: int = 2000,
    verbose: int = 0,
    hcore: np.ndarray | None = None,
) -> np.ndarray:
    """Build the spin-free generalized Fock in the AO basis from core+active density.

    Uses ASUKA's native DF/Cholesky integrals for Coulomb/exchange.

    Parameters
    ----------
    mol:
        Molecule object (must support `atom_coords`, `atom_charges`, etc.).
    auxbasis:
        Auxiliary basis name/spec for the DF approximation.
    hcore:
        Precomputed core Hamiltonian (T+V). If *None*, built from ASUKA's 1e integrals.
    """

    mo_core = np.asarray(mo_core, dtype=np.float64)
    mo_act = np.asarray(mo_act, dtype=np.float64)
    dm1_act = np.asarray(dm1_act, dtype=np.float64)
    if mo_core.ndim != 2 or mo_act.ndim != 2:
        raise ValueError("mo blocks must be 2D arrays")
    if mo_core.shape[0] != mo_act.shape[0]:
        raise ValueError("mo blocks must share the same AO dimension")

    nact = int(mo_act.shape[1])
    if dm1_act.shape != (nact, nact):
        raise ValueError("dm1_act shape mismatch")

    nao = int(mo_act.shape[0])
    dm_core = 2.0 * (mo_core @ mo_core.T) if mo_core.size else np.zeros((nao, nao), dtype=np.float64)
    dm_act = mo_act @ dm1_act @ mo_act.T
    dm_tot = dm_act + dm_core

    if hcore is None:
        hcore = build_hcore_ao(mol)
    else:
        hcore = np.asarray(hcore, dtype=np.float64)

    from asuka.integrals.df_context import get_df_cholesky_context  # noqa: PLC0415

    ctx = get_df_cholesky_context(
        mol,
        auxbasis=auxbasis,
        max_memory=int(max_memory),
        verbose=int(verbose),
    )
    b_ao = np.asarray(ctx.B_ao, dtype=np.float64, order="C")  # (nao,nao,naux)
    if b_ao.shape[0] != nao or b_ao.shape[1] != nao:
        raise ValueError("DF context AO dimension mismatch")

    vhf = build_vhf_df(dm_tot, b_ao=b_ao)
    return hcore + vhf


def semicanonicalize_core_virt_from_fock_ao(
    f_ao: np.ndarray,
    *,
    mo_core: np.ndarray,
    mo_virt: np.ndarray,
    deg_tol: float = 1e-10,
) -> SemicanonicalCoreVirt:
    """Semicanonicalize core/virtual orbitals by diagonalizing blocks of a given AO Fock."""

    f_ao = np.asarray(f_ao, dtype=np.float64)
    mo_core = np.asarray(mo_core, dtype=np.float64)
    mo_virt = np.asarray(mo_virt, dtype=np.float64)
    if f_ao.ndim != 2 or f_ao.shape[0] != f_ao.shape[1]:
        raise ValueError("f_ao must be a square 2D array")
    nao = int(f_ao.shape[0])
    if mo_core.ndim != 2 or int(mo_core.shape[0]) != nao:
        raise ValueError("mo_core AO dimension mismatch")
    if mo_virt.ndim != 2 or int(mo_virt.shape[0]) != nao:
        raise ValueError("mo_virt AO dimension mismatch")

    ncore = int(mo_core.shape[1])
    nvirt = int(mo_virt.shape[1])

    if ncore > 0:
        f_cc = mo_core.T @ f_ao @ mo_core
        f_cc = 0.5 * (f_cc + f_cc.T)
        eps_core, u_core = molcas_diafck_eigh(f_cc)
        mo_core_sc = mo_core @ u_core
    else:
        eps_core = np.zeros((0,), dtype=np.float64)
        u_core = np.zeros((0, 0), dtype=np.float64)
        mo_core_sc = mo_core

    if nvirt > 0:
        f_vv = mo_virt.T @ f_ao @ mo_virt
        f_vv = 0.5 * (f_vv + f_vv.T)
        eps_virt, u_virt = molcas_diafck_eigh(f_vv)
        mo_virt_sc = mo_virt @ u_virt
    else:
        eps_virt = np.zeros((0,), dtype=np.float64)
        u_virt = np.zeros((0, 0), dtype=np.float64)
        mo_virt_sc = mo_virt

    return SemicanonicalCoreVirt(
        mo_core=np.asarray(mo_core_sc, order="C"),
        mo_virt=np.asarray(mo_virt_sc, order="C"),
        eps_core=np.asarray(eps_core, order="C"),
        eps_virt=np.asarray(eps_virt, order="C"),
        u_core=np.asarray(u_core, order="C"),
        u_virt=np.asarray(u_virt, order="C"),
    )


def semicanonicalize_core_virt_from_generalized_fock(
    mc: Any,
    *,
    mo_core: np.ndarray,
    mo_act: np.ndarray,
    mo_virt: np.ndarray,
    dm1_act: np.ndarray,
) -> SemicanonicalCoreVirt:
    """Semicanonicalize core/virtual orbitals by diagonalizing the generalized Fock blocks.

    The generalized (spin-free) Fock is built from the reference density:
      D = D_core + D_act
    where:
      D_core = 2 * C_core C_core^T
      D_act  = C_act dm1_act C_act^T

    Only the core-core and virtual-virtual blocks are diagonalized; active orbitals
    are left unchanged.

    If either the core or virtual block is empty, only the non-empty block is
    diagonalized and the missing block returns empty energies/rotations.
    """

    mo_core = np.asarray(mo_core, dtype=np.float64)
    mo_act = np.asarray(mo_act, dtype=np.float64)
    mo_virt = np.asarray(mo_virt, dtype=np.float64)
    dm1_act = np.asarray(dm1_act, dtype=np.float64)

    if mo_core.ndim != 2 or mo_act.ndim != 2 or mo_virt.ndim != 2:
        raise ValueError("mo blocks must be 2D arrays")
    nao = int(mo_core.shape[0])
    if mo_act.shape[0] != nao or mo_virt.shape[0] != nao:
        raise ValueError("mo blocks must share the same AO dimension")
    ncore = int(mo_core.shape[1])
    nact = int(mo_act.shape[1])
    nvirt = int(mo_virt.shape[1])
    if dm1_act.shape != (nact, nact):
        raise ValueError("dm1_act shape mismatch")
    f_ao = build_generalized_fock_ao(mc.mol, mo_core=mo_core, mo_act=mo_act, dm1_act=dm1_act)
    return semicanonicalize_core_virt_from_fock_ao(f_ao, mo_core=mo_core, mo_virt=mo_virt)
