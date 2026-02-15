from __future__ import annotations

"""Harmonic vibrational frequency analysis and normal modes."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .constants import AMU_TO_AU, AU_TO_CM1


@dataclass(frozen=True)
class NormalModes:
    coords_bohr: np.ndarray
    masses_amu: np.ndarray

    hessian_cart: np.ndarray
    hessian_mw: np.ndarray

    tr_basis_mw: np.ndarray
    linear: bool

    eigvals: np.ndarray
    eigvecs_mw: np.ndarray

    freq_cm1: np.ndarray
    modes_cart: np.ndarray

    def n_imag(self) -> int:
        return int(np.count_nonzero(np.asarray(self.freq_cm1) < 0.0))

    def imag_mode_indices(self) -> np.ndarray:
        return np.where(np.asarray(self.freq_cm1) < 0.0)[0]

    def displace_along_mode(
        self,
        mode: int,
        *,
        step: float,
        unit: str = "Angstrom",
        mode_scale: str = "cart_max",
    ) -> tuple[np.ndarray, np.ndarray]:
        from .sampling import displace_along_mode as _displace  # noqa: PLC0415

        mode = int(mode)
        if mode < 0 or mode >= int(self.modes_cart.shape[2]):
            raise ValueError("mode index out of range")

        return _displace(
            self.coords_bohr,
            self.modes_cart[:, :, mode],
            step=float(step),
            unit=str(unit),
            mode_scale=str(mode_scale),
            masses_amu=self.masses_amu,
        )

    def sample(
        self,
        *,
        n_samples: int,
        temperature_k: float = 300.0,
        method: str = "wigner",
        exclude_imag: bool = True,
        scale: float = 1.0,
        seed: int | None = None,
        unit: str = "Bohr",
    ) -> np.ndarray:
        from .sampling import sample_normal_modes as _sample  # noqa: PLC0415

        return _sample(
            self,
            n_samples=int(n_samples),
            temperature_k=float(temperature_k),
            method=str(method),
            exclude_imag=bool(exclude_imag),
            scale=float(scale),
            seed=None if seed is None else int(seed),
            unit=str(unit),
        )


def _coords_from_mol_bohr(mol: Any) -> np.ndarray:
    if callable(getattr(mol, "atom_coord", None)):
        natm = int(getattr(mol, "natm"))
        coords = np.zeros((natm, 3), dtype=np.float64)
        for ia in range(natm):
            coords[ia] = np.asarray(mol.atom_coord(ia, unit="Bohr"), dtype=np.float64)
        return coords
    if hasattr(mol, "atoms_bohr"):
        return np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64)
    raise TypeError("cannot extract coordinates from mol; pass coords_bohr explicitly")


def _masses_from_mol_amu(mol: Any) -> np.ndarray:
    if callable(getattr(mol, "atom_mass_list", None)):
        masses = np.asarray(mol.atom_mass_list(), dtype=np.float64).ravel()
        if masses.size != int(getattr(mol, "natm")):
            raise ValueError("mol.atom_mass_list returned wrong length")
        return masses
    raise TypeError("masses_amu not provided and cannot infer masses from mol")


def _is_linear(coords_bohr: np.ndarray, masses_au: np.ndarray, *, tol: float = 1e-10) -> bool:
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    m = np.asarray(masses_au, dtype=np.float64).ravel()
    if coords.shape[0] != m.size:
        raise ValueError("coords and masses size mismatch")

    mtot = float(np.sum(m))
    if mtot <= 0.0:
        raise ValueError("non-positive total mass")

    com = np.einsum("i,ij->j", m, coords) / mtot
    r = coords - com[None, :]

    I = np.zeros((3, 3), dtype=np.float64)
    for i in range(coords.shape[0]):
        ri = r[i]
        mi = float(m[i])
        I += mi * ((np.dot(ri, ri) * np.eye(3)) - np.outer(ri, ri))

    evals = np.linalg.eigvalsh(I)
    return bool(np.min(evals) < tol * max(1.0, float(np.max(evals))))


def _build_tr_basis_mw(
    coords_bohr: np.ndarray,
    masses_au: np.ndarray,
    *,
    linear: bool | None,
    tr_tol: float,
) -> tuple[np.ndarray, bool]:
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    m = np.asarray(masses_au, dtype=np.float64).ravel()
    if m.size != natm:
        raise ValueError("masses size mismatch")

    n = 3 * natm
    msqrt = np.sqrt(m)

    mtot = float(np.sum(m))
    com = np.einsum("i,ij->j", m, coords) / mtot
    r = coords - com[None, :]

    t = np.zeros((n, 3), dtype=np.float64)
    for ia in range(natm):
        for a in range(3):
            t[3 * ia + a, a] = msqrt[ia]

    rot = np.zeros((n, 3), dtype=np.float64)
    for ia in range(natm):
        x, y, z = r[ia]
        s = msqrt[ia]
        rot[3 * ia + 0, 0] = 0.0
        rot[3 * ia + 1, 0] = -z * s
        rot[3 * ia + 2, 0] = y * s
        rot[3 * ia + 0, 1] = z * s
        rot[3 * ia + 1, 1] = 0.0
        rot[3 * ia + 2, 1] = -x * s
        rot[3 * ia + 0, 2] = -y * s
        rot[3 * ia + 1, 2] = x * s
        rot[3 * ia + 2, 2] = 0.0

    B = np.concatenate([t, rot], axis=1)

    Q, R = np.linalg.qr(B)
    diag = np.abs(np.diag(R))
    keep = diag > float(tr_tol)
    Qtr = Q[:, keep]

    if linear is None:
        linear = _is_linear(coords, m, tol=1e-12)

    return np.asarray(Qtr, dtype=np.float64), bool(linear)


def _orthonormal_complement(Q: np.ndarray, *, seed: int = 0) -> np.ndarray:
    Q = np.asarray(Q, dtype=np.float64)
    n, k = Q.shape
    if k == 0:
        return np.eye(n, dtype=np.float64)
    if k >= n:
        raise ValueError("Q has no orthogonal complement")

    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal((n, n - k))
    X = X - Q @ (Q.T @ X)
    Qp, R = np.linalg.qr(X)
    if np.min(np.abs(np.diag(R))) < 1e-12:  # pragma: no cover
        raise RuntimeError("failed to build a full-rank orthogonal complement; increase seed or check TR basis")
    return np.asarray(Qp, dtype=np.float64)


def frequency_analysis(
    *,
    hessian_cart: np.ndarray,
    mol: Any | None = None,
    coords_bohr: np.ndarray | None = None,
    masses_amu: Sequence[float] | None = None,
    linear: bool | None = None,
    tr_tol: float = 1e-10,
    symmetrize: bool = True,
    seed: int = 0,
) -> NormalModes:
    """Perform harmonic vibrational analysis given a Cartesian Hessian."""

    H = np.asarray(hessian_cart, dtype=np.float64)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("hessian_cart must be a square matrix")

    if mol is None and coords_bohr is None:
        raise ValueError("provide either mol or coords_bohr")

    coords = _coords_from_mol_bohr(mol) if coords_bohr is None else np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    n = 3 * natm
    if H.shape != (n, n):
        raise ValueError(f"hessian_cart has shape {H.shape}, expected {(n, n)}")

    if masses_amu is None:
        if mol is None:
            raise ValueError("masses_amu not provided and mol is None")
        masses = _masses_from_mol_amu(mol)
    else:
        masses = np.asarray(list(masses_amu), dtype=np.float64).ravel()

    if masses.size != natm:
        raise ValueError("masses_amu length mismatch")
    if np.any(masses <= 0.0):
        raise ValueError("masses_amu must be positive")

    if bool(symmetrize):
        H = 0.5 * (H + H.T)

    masses_au = masses * float(AMU_TO_AU)
    mrep = np.repeat(masses_au, 3)
    inv_sqrt_m = 1.0 / np.sqrt(mrep)

    F = (inv_sqrt_m[:, None] * H) * inv_sqrt_m[None, :]
    if bool(symmetrize):
        F = 0.5 * (F + F.T)

    Qtr, linear_flag = _build_tr_basis_mw(coords, masses_au, linear=linear, tr_tol=float(tr_tol))

    Qvib = _orthonormal_complement(Qtr, seed=int(seed))
    Fv = Qvib.T @ F @ Qvib
    if bool(symmetrize):
        Fv = 0.5 * (Fv + Fv.T)

    evals, evecs = np.linalg.eigh(Fv)
    idx = np.argsort(evals)
    evals = np.asarray(evals[idx], dtype=np.float64)
    evecs = np.asarray(evecs[:, idx], dtype=np.float64)

    Vmw = Qvib @ evecs

    freq = np.zeros_like(evals)
    for i, lam in enumerate(evals):
        if lam >= 0.0:
            freq[i] = np.sqrt(lam) * float(AU_TO_CM1)
        else:
            freq[i] = -np.sqrt(-lam) * float(AU_TO_CM1)

    Lflat = Vmw * inv_sqrt_m[:, None]
    modes_cart = Lflat.reshape((natm, 3, -1))

    return NormalModes(
        coords_bohr=np.asarray(coords, dtype=np.float64),
        masses_amu=np.asarray(masses, dtype=np.float64),
        hessian_cart=np.asarray(H, dtype=np.float64),
        hessian_mw=np.asarray(F, dtype=np.float64),
        tr_basis_mw=np.asarray(Qtr, dtype=np.float64),
        linear=bool(linear_flag),
        eigvals=np.asarray(evals, dtype=np.float64),
        eigvecs_mw=np.asarray(Vmw, dtype=np.float64),
        freq_cm1=np.asarray(freq, dtype=np.float64),
        modes_cart=np.asarray(modes_cart, dtype=np.float64),
    )


__all__ = [
    "NormalModes",
    "frequency_analysis",
]
