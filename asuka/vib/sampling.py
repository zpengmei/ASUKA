from __future__ import annotations

"""Normal mode displacements and sampling."""

from typing import TYPE_CHECKING

import numpy as np

from .constants import AMU_TO_AU, ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, KB_HARTREE_PER_K

if TYPE_CHECKING:
    from .frequency import NormalModes


def displace_along_mode(
    coords_bohr: np.ndarray,
    mode_cart: np.ndarray,
    *,
    step: float,
    unit: str = "Angstrom",
    mode_scale: str = "cart_max",
    masses_amu: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return +/- displacements along a single mode."""

    coords0 = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    L = np.asarray(mode_cart, dtype=np.float64).reshape(coords0.shape)

    scale_s = str(mode_scale).strip().lower()
    unit_s = str(unit).strip().lower()

    if scale_s == "cart_max":
        if unit_s in ("bohr", "a0", "au"):
            step_bohr = float(step)
        elif unit_s in ("angstrom", "ang", "a"):
            step_bohr = float(step) * float(ANGSTROM_TO_BOHR)
        else:
            raise ValueError("unit must be 'Bohr' or 'Angstrom'")

        denom = float(np.max(np.abs(L)))
        if denom == 0.0:
            raise ValueError("mode_cart is all zeros")
        q = step_bohr / denom
        dR = q * L

    elif scale_s == "q":
        if masses_amu is None:
            raise ValueError("masses_amu is required when mode_scale='q'")
        m = np.asarray(masses_amu, dtype=np.float64).ravel() * float(AMU_TO_AU)
        if m.size != coords0.shape[0]:
            raise ValueError("masses_amu length mismatch")
        msqrt = np.sqrt(m)[:, None]
        Vmw = (msqrt * L).reshape((-1,))
        Vmw /= np.linalg.norm(Vmw)
        Q = float(step)
        dR = (Vmw.reshape(coords0.shape) / msqrt) * Q

    else:
        raise ValueError("mode_scale must be 'cart_max' or 'q'")

    return coords0 + dR, coords0 - dR


def _coth(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    e2x = np.exp(np.clip(2.0 * x, -700.0, 700.0))
    return (e2x + 1.0) / (e2x - 1.0)


def sample_normal_modes(
    modes: "NormalModes",
    *,
    n_samples: int,
    temperature_k: float = 300.0,
    method: str = "wigner",
    exclude_imag: bool = True,
    scale: float = 1.0,
    seed: int | None = None,
    unit: str = "Bohr",
) -> np.ndarray:
    """Sample geometries from a harmonic normal mode model."""

    natm = int(modes.coords_bohr.shape[0])
    if int(n_samples) <= 0:
        raise ValueError("n_samples must be positive")

    T = float(temperature_k)
    if T < 0.0:
        raise ValueError("temperature_k must be non-negative")

    method_s = str(method).strip().lower()
    if method_s not in ("wigner", "classical"):
        raise ValueError("method must be 'wigner' or 'classical'")

    lam = np.asarray(modes.eigvals, dtype=np.float64)
    Vmw = np.asarray(modes.eigvecs_mw, dtype=np.float64)

    keep = (lam > 0.0) if bool(exclude_imag) else np.ones_like(lam, dtype=bool)
    lam_k = lam[keep]
    Vmw_k = Vmw[:, keep]
    if lam_k.size == 0:
        raise ValueError("no modes selected for sampling")

    omega = np.sqrt(np.abs(lam_k))

    if method_s == "classical":
        if T == 0.0:
            sigma2 = np.zeros_like(omega)
        else:
            sigma2 = float(KB_HARTREE_PER_K) * T / (omega**2)
    else:
        if T == 0.0:
            sigma2 = 1.0 / (2.0 * omega)
        else:
            beta = 1.0 / (float(KB_HARTREE_PER_K) * T)
            x = 0.5 * beta * omega
            sigma2 = (0.5 * _coth(x)) / omega

    rng = np.random.default_rng(None if seed is None else int(seed))
    q = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=(int(n_samples), lam_k.size))
    q *= float(scale)

    dq_flat = q @ Vmw_k.T

    m_au = np.asarray(modes.masses_amu, dtype=np.float64) * float(AMU_TO_AU)
    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(m_au, 3))
    dR_flat = dq_flat * inv_sqrt_m[None, :]
    dR = dR_flat.reshape((int(n_samples), natm, 3))

    coords = modes.coords_bohr[None, :, :] + dR

    unit_s = str(unit).strip().lower()
    if unit_s in ("bohr", "a0", "au"):
        return coords
    if unit_s in ("angstrom", "ang", "a"):
        return coords * float(BOHR_TO_ANGSTROM)
    raise ValueError("unit must be 'Bohr' or 'Angstrom'")


__all__ = [
    "displace_along_mode",
    "sample_normal_modes",
]

