from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _as_f64_2d(a: Any) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("expected a 2D array")
    return arr


def _is_cart_basis(mol: Any) -> bool:
    """Return True if `mol` uses Cartesian basis functions.

    ASUKA's cuERI-backed DF context currently only supports Cartesian AO bases.
    PySCF's default is spherical (`cart=False`). For spherical bases, we fall
    back to PySCF's DF/Cholesky builder (see `_build_df_pair_blocks_pyscf`).
    """

    # PySCF Mole: `mol.cart` exists.
    cart = getattr(mol, "cart", None)
    if cart is None:
        # asuka.frontend.molecule.Molecule path: treat as Cartesian by default.
        return True
    return bool(cart)


def _transform_cderi_s1_to_mo_pairs(
    cderi_s1: np.ndarray,
    mo_x: np.ndarray,
    mo_y: np.ndarray,
    *,
    max_memory: int,
) -> np.ndarray:
    """Transform AO DF/Cholesky vectors (aosym='s1') to an MO pair block.

    Parameters
    ----------
    cderi_s1:
        AO factors as shape (naux, nao*nao), where the last dimension is the
        ordered AO pair index μ*nao+ν (PySCF aosym='s1' convention).
    mo_x, mo_y:
        AO->MO coefficient blocks with shapes (nao, nx) and (nao, ny).

    Returns
    -------
    mo_cderi : np.ndarray
        Shape (naux, nx*ny) in PySCF outcore.general style.
    """

    cderi_s1 = np.asarray(cderi_s1, dtype=np.float64, order="C")
    mo_x = np.asarray(mo_x, dtype=np.float64, order="C")
    mo_y = np.asarray(mo_y, dtype=np.float64, order="C")

    naux = int(cderi_s1.shape[0])
    nao = int(mo_x.shape[0])
    if cderi_s1.shape[1] != nao * nao:
        raise ValueError("cderi_s1 shape mismatch with mo_x/mo_y")

    nx = int(mo_x.shape[1])
    ny = int(mo_y.shape[1])
    out = np.empty((naux, nx * ny), dtype=np.float64, order="C")

    # Chunk over aux to control peak intermediate memory (tmp/res).
    mem_mb = int(max_memory)
    if mem_mb <= 0:
        mem_mb = 256
    mem_bytes = int(mem_mb) * (1024**2)
    per_aux = int(8 * (nao * ny + nx * ny))
    if per_aux <= 0:
        per_aux = 1
    block_naux = max(1, min(naux, mem_bytes // per_aux))
    block_naux = int(min(block_naux, 1024))

    for q0 in range(0, naux, block_naux):
        q1 = min(naux, q0 + block_naux)
        qb = int(q1 - q0)
        # (qb, nao, nao)
        b_blk = cderi_s1[q0:q1, :].reshape(qb, nao, nao)

        # tmp[Q, μ, y] = Σ_ν B[Q,μ,ν] C_y[ν,y]
        tmp = np.tensordot(b_blk, mo_y, axes=([2], [0]))  # (qb, nao, ny)
        tmp = np.transpose(tmp, (1, 2, 0))  # (nao, ny, qb)
        # res[x, y, Q] = Σ_μ C_x[μ,x] tmp[μ,y,Q]
        res = np.tensordot(mo_x.T, tmp, axes=([1], [0]))  # (nx, ny, qb)
        out[q0:q1, :] = np.asarray(res.reshape(nx * ny, qb).T, dtype=np.float64, order="C")

    return out


def _build_df_pair_blocks_pyscf(
    mol: Any,
    blocks: list[tuple[np.ndarray, np.ndarray]],
    *,
    auxbasis: Any,
    max_memory: int,
    verbose: int,
    compute_pair_norm: bool,
) -> list["DFPairBlock"]:
    """PySCF fallback builder for spherical AO bases (mol.cart=False)."""

    try:
        from pyscf.df import incore  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "PySCF is required to build DF pair blocks for spherical AO bases (mol.cart=False). "
            "Either install PySCF or run with a cartesian basis (mol.cart=True)."
        ) from e

    # Use aosym='s1' (full ordered AO pairs) to match DFPairBlock conventions.
    cderi = incore.cholesky_eri(
        mol,
        auxbasis=auxbasis,
        aosym="s1",
        max_memory=int(max_memory),
        verbose=int(verbose),
    )
    cderi = np.asarray(cderi, dtype=np.float64, order="C")  # (naux, nao*nao)

    out: list[DFPairBlock] = []
    for mo_x, mo_y in blocks:
        mo_x = _as_f64_2d(mo_x)
        mo_y = _as_f64_2d(mo_y)
        if mo_x.shape[0] != mo_y.shape[0]:
            raise ValueError("mo_x and mo_y must have the same number of AO rows")
        nx = int(mo_x.shape[1])
        ny = int(mo_y.shape[1])
        if nx <= 0 or ny <= 0:
            raise ValueError("empty orbital block")

        mo_cderi = _transform_cderi_s1_to_mo_pairs(cderi, mo_x, mo_y, max_memory=int(max_memory))
        l_full = np.asarray(mo_cderi.T, dtype=np.float64, order="C")
        pair_norm = None
        if compute_pair_norm:
            pair_norm = np.asarray(np.linalg.norm(l_full, axis=1), dtype=np.float64, order="C")
        out.append(DFPairBlock(nx=nx, ny=ny, l_full=l_full, pair_norm=pair_norm))
    return out


@dataclass(frozen=True)
class DFPairBlock:
    """Rectangular DF/Cholesky vectors for a fixed MO orbital block (X,Y).

    Stores vectors d[L,xy] such that:
      (x y| u v) ~= sum_L d[L,xy] d[L,uv] = dot(d[:,xy], d[:,uv])

    Conventions
    -----------
    - Ordered-pair flattening is used:
        xy_id = x * ny + y
      where x in [0,nx) and y in [0,ny).
    - `l_full` is stored as shape (nx*ny, naux), row-major in `xy_id`.
    """

    nx: int
    ny: int
    l_full: np.ndarray  # (nx*ny, naux)
    pair_norm: np.ndarray | None = None  # (nx*ny,)

    @property
    def naux(self) -> int:
        return int(self.l_full.shape[1])

    def xy_id(self, x: int, y: int) -> int:
        x = int(x)
        y = int(y)
        if x < 0 or x >= int(self.nx):
            raise IndexError("x out of range")
        if y < 0 or y >= int(self.ny):
            raise IndexError("y out of range")
        return x * int(self.ny) + y

    def vec_xy(self, x: int, y: int) -> np.ndarray:
        """Return the length-naux DF vector d[:,xy] for a given ordered pair (x,y)."""

        return self.l_full[self.xy_id(x, y)]


def rotate_df_pair_block(
    block: DFPairBlock,
    *,
    u_x: np.ndarray | None = None,
    u_y: np.ndarray | None = None,
) -> DFPairBlock:
    """Rotate a DF pair block by orbital rotations within the X and/or Y spaces.

    Given MO coefficient blocks ``C_x`` and ``C_y`` used to build ``block``,
    the rotated orbitals are:

      C'_x = C_x @ u_x
      C'_y = C_y @ u_y

    The DF vectors transform as:

      d'[P, x', y'] = Σ_{x,y} u_x[x,x'] u_y[y,y'] d[P, x, y]

    Notes
    -----
    - This is a purely algebraic rotation of the already-built MO-basis DF vectors;
      it does not touch any AO DF context.
    - ``pair_norm`` is dropped (set to ``None``) because it is not used in the CASPT2
      DF workflows in this repo (and would need recomputation after rotation).
    """

    nx = int(block.nx)
    ny = int(block.ny)
    naux = int(block.naux)

    l_full = np.asarray(block.l_full, dtype=np.float64)
    if l_full.shape != (nx * ny, naux):
        raise ValueError("block.l_full shape mismatch")

    l3 = np.asarray(l_full.reshape(nx, ny, naux), dtype=np.float64, order="C")

    if u_x is not None:
        u_x = np.asarray(u_x, dtype=np.float64)
        if u_x.shape != (nx, nx):
            raise ValueError("u_x shape mismatch")
        l3 = np.einsum("xa,xyP->ayP", u_x, l3, optimize=True)

    if u_y is not None:
        u_y = np.asarray(u_y, dtype=np.float64)
        if u_y.shape != (ny, ny):
            raise ValueError("u_y shape mismatch")
        l3 = np.einsum("yb,ayP->abP", u_y, l3, optimize=True)

    l_rot = np.asarray(l3.reshape(nx * ny, naux), dtype=np.float64, order="C")
    return DFPairBlock(nx=nx, ny=ny, l_full=l_rot, pair_norm=None)


def build_df_pair_block(
    mol,
    mo_x: np.ndarray,
    mo_y: np.ndarray,
    *,
    auxbasis: Any = "weigend+etb",
    filename: str | Path | None = None,
    dataname: str = "cderi",
    tmpdir: str | None = None,
    max_memory: int = 2000,
    verbose: int = 0,
    compute_pair_norm: bool = True,
) -> DFPairBlock:
    """Build MO-basis DF/Cholesky vectors for the rectangular orbital block (X,Y).

    By default (when `filename is None`), this uses a cached AO DF context and
    performs the AO->MO transform in-memory.

    If `filename` is provided, the transformed MO DF vectors are also written to an
    HDF5 file under the dataset name `dataname` (PySCF-style outcore layout:
    shape (naux, ncol)).

    Parameters
    ----------
    mo_x, mo_y:
        AO->MO coefficient matrices with shapes (nao, nx) and (nao, ny).
    """

    mo_x = _as_f64_2d(mo_x)
    mo_y = _as_f64_2d(mo_y)
    if mo_x.shape[0] != mo_y.shape[0]:
        raise ValueError("mo_x and mo_y must have the same number of AO rows")

    nx = int(mo_x.shape[1])
    ny = int(mo_y.shape[1])
    if nx <= 0 or ny <= 0:
        raise ValueError("empty orbital block")

    if not _is_cart_basis(mol):
        # Spherical AO basis (PySCF default): use PySCF DF/Cholesky integrals.
        blocks = _build_df_pair_blocks_pyscf(
            mol,
            [(mo_x, mo_y)],
            auxbasis=auxbasis,
            max_memory=int(max_memory),
            verbose=int(verbose),
            compute_pair_norm=bool(compute_pair_norm),
        )
        block = blocks[0]
        if filename is not None:
            try:
                import h5py  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError("filename output requires `h5py` to be installed") from e

            filename = str(filename)
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(str(filename), "w") as f:
                # PySCF outcore layout: (naux, ncol)
                f.create_dataset(str(dataname), data=np.asarray(block.l_full.T, dtype=np.float64, order="C"))
        return block

    from asuka.integrals.df_context import get_df_cholesky_context  # noqa: PLC0415

    ctx = get_df_cholesky_context(
        mol,
        auxbasis=auxbasis,
        max_memory=int(max_memory),
        verbose=int(verbose),
    )
    mo_cderi = ctx.transform(mo_x, mo_y, compact=False, cache=True, max_memory=int(max_memory))

    if filename is not None:
        try:
            import h5py  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("filename output requires `h5py` to be installed") from e

        filename = str(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(filename), "w") as f:
            f.create_dataset(str(dataname), data=np.asarray(mo_cderi, dtype=np.float64, order="C"))

    mo_cderi = _as_f64_2d(mo_cderi)
    ncol = nx * ny
    if mo_cderi.shape[1] == ncol:
        # PySCF convention: (naux, ncol) for outcore.general
        l_full = mo_cderi.T
    elif mo_cderi.shape[0] == ncol:
        l_full = mo_cderi
    else:
        raise ValueError(f"unexpected DF shape {mo_cderi.shape} for nx*ny={ncol}")

    l_full_c = np.asarray(l_full, order="C")
    pair_norm = None
    if compute_pair_norm:
        pair_norm = np.asarray(np.linalg.norm(l_full_c, axis=1), order="C")
    return DFPairBlock(nx=nx, ny=ny, l_full=l_full_c, pair_norm=pair_norm)


def build_df_pair_blocks(
    mol,
    blocks: list[tuple[np.ndarray, np.ndarray]],
    *,
    auxbasis: Any = "weigend+etb",
    tmpdir: str | None = None,
    max_memory: int = 2000,
    verbose: int = 0,
    compute_pair_norm: bool = True,
) -> list[DFPairBlock]:
    """Build multiple DFPairBlock objects in a single pass over cached AO DF data."""

    if not blocks:
        return []

    if not _is_cart_basis(mol):
        # Spherical AO basis (PySCF default): use PySCF DF/Cholesky integrals.
        return _build_df_pair_blocks_pyscf(
            mol,
            blocks,
            auxbasis=auxbasis,
            max_memory=int(max_memory),
            verbose=int(verbose),
            compute_pair_norm=bool(compute_pair_norm),
        )

    from asuka.integrals.df_context import get_df_cholesky_context  # noqa: PLC0415

    ctx = get_df_cholesky_context(
        mol,
        auxbasis=auxbasis,
        max_memory=int(max_memory),
        verbose=int(verbose),
    )

    prepared: list[tuple[np.ndarray, np.ndarray, bool]] = []
    shapes: list[tuple[int, int]] = []
    for mo_x, mo_y in blocks:
        mo_x = _as_f64_2d(mo_x)
        mo_y = _as_f64_2d(mo_y)
        if mo_x.shape[0] != mo_y.shape[0]:
            raise ValueError("mo_x and mo_y must have the same number of AO rows")
        nx = int(mo_x.shape[1])
        ny = int(mo_y.shape[1])
        if nx <= 0 or ny <= 0:
            raise ValueError("empty orbital block")
        prepared.append((mo_x, mo_y, False))
        shapes.append((nx, ny))

    mo_cderi_list = ctx.transform_many(prepared, cache=True, max_memory=int(max_memory))
    out: list[DFPairBlock] = []
    for (nx, ny), mo_cderi in zip(shapes, mo_cderi_list):
        mo_cderi = _as_f64_2d(mo_cderi)
        ncol = int(nx) * int(ny)
        if mo_cderi.shape[1] == ncol:
            l_full = mo_cderi.T
        elif mo_cderi.shape[0] == ncol:
            l_full = mo_cderi
        else:
            raise ValueError(f"unexpected DF shape {mo_cderi.shape} for nx*ny={ncol}")

        l_full_c = np.asarray(l_full, order="C")
        pair_norm = None
        if compute_pair_norm:
            pair_norm = np.asarray(np.linalg.norm(l_full_c, axis=1), order="C")
        out.append(DFPairBlock(nx=int(nx), ny=int(ny), l_full=l_full_c, pair_norm=pair_norm))
    return out


def build_df_pair_blocks_from_df_B(
    B_ao: Any,
    blocks: list[tuple[np.ndarray, np.ndarray]],
    *,
    max_memory: int = 2000,
    compute_pair_norm: bool = False,
) -> list[DFPairBlock]:
    """Build DFPairBlock objects from whitened AO DF factors ``B[μ,ν,Q]``.

    This is a bridge for the ASUKA frontend (cuERI) pipeline where SCF produces
    the whitened 3-center factors on GPU:

      (μν|λσ) ≈ Σ_Q B[μ,ν,Q] B[λ,σ,Q]

    Parameters
    ----------
    B_ao
        Whitened AO DF factors with shape (nao, nao, naux). Can be a NumPy or
        CuPy array. The returned DFPairBlocks will live on the same backend.
    blocks
        List of (mo_x, mo_y) coefficient blocks with shapes (nao, nx) and
        (nao, ny). These may be NumPy arrays even when B_ao is on GPU; they
        will be converted as needed.
    max_memory
        Chunking hint (in MB) for the aux dimension.
    compute_pair_norm
        If True, compute and store per-pair norms (used rarely in this repo).
    """

    if not blocks:
        return []

    # Select backend based on B_ao type.
    xp = np
    try:
        import cupy as cp  # noqa: PLC0415

        if hasattr(B_ao, "__cuda_array_interface__") or isinstance(B_ao, cp.ndarray):
            xp = cp
    except Exception:
        cp = None  # type: ignore[assignment]

    B = xp.asarray(B_ao, dtype=xp.float64)
    if B.ndim != 3:
        raise ValueError("B_ao must have shape (nao,nao,naux)")
    nao, nao2, naux = map(int, B.shape)
    if nao <= 0 or nao2 != nao or naux <= 0:
        raise ValueError("B_ao must have shape (nao,nao,naux) with positive sizes")

    # Chunk over aux to control intermediate memory (tmp/res).
    mem_mb = int(max_memory)
    if mem_mb <= 0:
        mem_mb = 256
    mem_bytes = int(mem_mb) * (1024**2)

    out: list[DFPairBlock] = []
    for mo_x, mo_y in blocks:
        Cx_in = xp.asarray(mo_x, dtype=xp.float64)
        Cy_in = xp.asarray(mo_y, dtype=xp.float64)
        if Cx_in.ndim != 2 or Cy_in.ndim != 2:
            raise ValueError("mo_x and mo_y must be 2D arrays")
        if int(Cx_in.shape[0]) != nao or int(Cy_in.shape[0]) != nao:
            raise ValueError("mo_x/mo_y AO dimension mismatch with B_ao")
        nx = int(Cx_in.shape[1])
        ny = int(Cy_in.shape[1])
        if nx <= 0 or ny <= 0:
            raise ValueError("empty orbital block")

        Cx = xp.ascontiguousarray(Cx_in)
        Cy = xp.ascontiguousarray(Cy_in)

        # Estimate per-aux temporary footprint: tmp(nao,ny) + res(nx,ny).
        per_aux = int(8 * (nao * ny + nx * ny))
        if per_aux <= 0:
            per_aux = 1
        block_naux = max(1, min(naux, mem_bytes // per_aux))
        block_naux = int(min(block_naux, 1024))

        mo_cderi = xp.empty((naux, nx * ny), dtype=xp.float64)
        for q0 in range(0, naux, block_naux):
            q1 = min(naux, q0 + block_naux)
            qb = int(q1 - q0)
            b_blk = B[:, :, q0:q1]  # (nao,nao,qb)

            # tmp[μ,y,Q] = Σ_ν B[μ,ν,Q] Cy[ν,y]
            tmp = xp.tensordot(b_blk, Cy, axes=([1], [0]))  # (nao,qb,ny)
            tmp = xp.transpose(tmp, (0, 2, 1))  # (nao,ny,qb)
            # res[x,y,Q] = Σ_μ Cx[μ,x] tmp[μ,y,Q]
            res = xp.tensordot(Cx.T, tmp, axes=([1], [0]))  # (nx,ny,qb)
            mo_cderi[q0:q1, :] = xp.asarray(res.reshape(nx * ny, qb).T, dtype=xp.float64)

        l_full = xp.ascontiguousarray(mo_cderi.T)  # (nx*ny,naux)
        pair_norm = None
        if compute_pair_norm:
            pair_norm = xp.asarray(xp.linalg.norm(l_full, axis=1), dtype=xp.float64)

        out.append(DFPairBlock(nx=nx, ny=ny, l_full=l_full, pair_norm=pair_norm))

    return out
