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
