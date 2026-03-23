"""Packed orbital rotation parameter vector.

Packed storage for orbital rotation parameters (closed-active, virtual-active, virtual-closed blocks).
Three blocks stored contiguously: CA (closed-active), VA (virtual-active),
VC (virtual-closed).

Memory layout (column-major per block):
    data_ = [ CA(nclosed*nact) | VA(nvirt*nact) | VC(nvirt*nclosed) ]

Element accessors:
    ele_ca(j, i):  j = closed index,  i = active index
    ele_va(j, i):  j = virtual index, i = active index
    ele_vc(j, i):  j = virtual index, i = closed index
"""

from __future__ import annotations

import numpy as np


class RotFile:
    """Orbital rotation vector in the non-redundant CA/VA/VC representation."""

    __slots__ = ("_nclosed", "_nact", "_nvirt", "_size", "_data")

    def __init__(self, nclosed: int, nact: int, nvirt: int, *, data: np.ndarray | None = None):
        self._nclosed = int(nclosed)
        self._nact = int(nact)
        self._nvirt = int(nvirt)
        self._size = self._nclosed * self._nact + self._nvirt * self._nact + self._nvirt * self._nclosed
        if data is not None:
            if data.shape != (self._size,):
                raise ValueError(f"data shape {data.shape} != ({self._size},)")
            self._data = np.array(data, dtype=np.float64)
        else:
            self._data = np.zeros(self._size, dtype=np.float64)

    # ------------------------------------------------------------------ sizes
    @property
    def size(self) -> int:
        return self._size

    @property
    def nclosed(self) -> int:
        return self._nclosed

    @property
    def nact(self) -> int:
        return self._nact

    @property
    def nvirt(self) -> int:
        return self._nvirt

    # ------------------------------------------------------------------ data
    @property
    def data(self) -> np.ndarray:
        return self._data

    # ---------------------------------------------------------- block offsets
    def _off_ca(self) -> int:
        return 0

    def _off_va(self) -> int:
        return self._nclosed * self._nact

    def _off_vc(self) -> int:
        return (self._nclosed + self._nvirt) * self._nact

    # ---------------------------------------------------- element accessors
    def ele_ca(self, j: int, i: int) -> float:
        return float(self._data[j + i * self._nclosed])

    def set_ele_ca(self, j: int, i: int, v: float) -> None:
        self._data[j + i * self._nclosed] = v

    def ele_va(self, j: int, i: int) -> float:
        return float(self._data[self._off_va() + j + i * self._nvirt])

    def set_ele_va(self, j: int, i: int, v: float) -> None:
        self._data[self._off_va() + j + i * self._nvirt] = v

    def ele_vc(self, j: int, i: int) -> float:
        return float(self._data[self._off_vc() + j + i * self._nvirt])

    def set_ele_vc(self, j: int, i: int, v: float) -> None:
        self._data[self._off_vc() + j + i * self._nvirt] = v

    # ---------------------------------------------------- block matrix views
    def ca_mat(self) -> np.ndarray:
        """Return a copy of the CA block as (nclosed, nact) column-major matrix."""
        o = self._off_ca()
        return self._data[o : o + self._nclosed * self._nact].reshape(
            self._nclosed, self._nact, order="F"
        ).copy()

    def va_mat(self) -> np.ndarray:
        """Return a copy of the VA block as (nvirt, nact) column-major matrix."""
        o = self._off_va()
        return self._data[o : o + self._nvirt * self._nact].reshape(
            self._nvirt, self._nact, order="F"
        ).copy()

    def vc_mat(self) -> np.ndarray:
        """Return a copy of the VC block as (nvirt, nclosed) column-major matrix."""
        o = self._off_vc()
        return self._data[o : o + self._nvirt * self._nclosed].reshape(
            self._nvirt, self._nclosed, order="F"
        ).copy()

    # ----------------------------------------- block ax_plus_y helpers
    def ax_plus_y_ca(self, a: float, mat: np.ndarray) -> None:
        """Add ``a * mat`` to the CA block.  *mat* is (nclosed, nact)."""
        o = self._off_ca()
        n = self._nclosed * self._nact
        self._data[o : o + n] += a * mat.ravel(order="F")

    def ax_plus_y_va(self, a: float, mat: np.ndarray) -> None:
        """Add ``a * mat`` to the VA block.  *mat* is (nvirt, nact)."""
        o = self._off_va()
        n = self._nvirt * self._nact
        self._data[o : o + n] += a * mat.ravel(order="F")

    def ax_plus_y_vc(self, a: float, mat: np.ndarray) -> None:
        """Add ``a * mat`` to the VC block.  *mat* is (nvirt, nclosed)."""
        o = self._off_vc()
        n = self._nvirt * self._nclosed
        self._data[o : o + n] += a * mat.ravel(order="F")

    # ------------------------------------------------- vector operations
    def dot(self, other: RotFile) -> float:
        return float(np.dot(self._data, other._data))

    def norm(self) -> float:
        return float(np.sqrt(self.dot(self)))

    def rms(self) -> float:
        if self._size == 0:
            return 0.0
        return self.norm() / np.sqrt(float(self._size))

    def normalize(self) -> float:
        """Normalize in-place; return the pre-normalization norm."""
        n = self.norm()
        if n > 0.0:
            self._data *= 1.0 / n
        return n

    def scale(self, factor: float) -> None:
        self._data *= factor

    def ax_plus_y(self, a: float, other: RotFile) -> None:
        """``self += a * other``."""
        self._data += a * other._data

    def copy(self) -> RotFile:
        return RotFile(self._nclosed, self._nact, self._nvirt, data=self._data.copy())

    def clone(self) -> RotFile:
        """Return a zero-filled copy with the same dimensions."""
        return RotFile(self._nclosed, self._nact, self._nvirt)

    def orthog(self, c_list: list[RotFile]) -> float:
        """Orthogonalize self against all vectors in *c_list*; return post-norm."""
        for c in c_list:
            self.ax_plus_y(-self.dot(c), c)
        return self.normalize()

    # ------------------------------------------------------- unpack
    def unpack(self) -> np.ndarray:
        """Unpack to full antisymmetric (nmo, nmo) matrix A.

        Layout (upper triangle stores the raw blocks,
        lower triangle is negated):

            A[active, closed]  =  CA[closed, active]^T   (row=active, col=closed)
            A[virtual, active] =  VA[virtual, active]     (row=virtual, col=active)
            A[virtual, closed] =  VC[virtual, closed]     (row=virtual, col=closed)
            A[j, i] = -A[i, j]  for j <= i
        """
        nocc = self._nclosed + self._nact
        nmo = nocc + self._nvirt
        out = np.zeros((nmo, nmo), dtype=np.float64)

        # Fill upper-triangle blocks (i > j convention in matrix element)
        for i in range(self._nact):
            for j in range(self._nvirt):
                out[j + nocc, i + self._nclosed] = self.ele_va(j, i)
            for j in range(self._nclosed):
                out[i + self._nclosed, j] = self.ele_ca(j, i)
        for i in range(self._nclosed):
            for j in range(self._nvirt):
                out[j + nocc, i] = self.ele_vc(j, i)

        # Antisymmetrize
        for i in range(nmo):
            for j in range(i):
                out[j, i] = -out[i, j]
        return out
