"""Public API for NDDO semiempirical calculations."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .basis import symbol_to_Z
from .gradient import am1_energy_gradient_scf
from .params import ANGSTROM_TO_BOHR, load_params
from .scf import SCFResult, am1_scf


def _normalize_device(device: str) -> str:
    out = str(device).strip().lower()
    if out not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    return out


def _normalize_fock_mode_kwargs(scf_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(scf_kwargs)
    if "fock_mode" in out:
        mode = str(out["fock_mode"]).strip().lower()
        if mode not in ("ri", "w", "auto"):
            raise ValueError("fock_mode must be 'ri', 'w', or 'auto'")
        out["fock_mode"] = mode
    return out


def _pop_gradient_backend_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    out = dict(kwargs)
    backend = str(out.pop("gradient_backend", "auto")).strip().lower()
    if backend not in ("auto", "cuda_analytic", "cpu_frozen"):
        raise ValueError("gradient_backend must be 'auto', 'cuda_analytic', or 'cpu_frozen'")
    return out, backend


class SemiempiricalCalculator:
    """Calculator for semiempirical NDDO energies.

    Parameters
    ----------
    method : str
        Method name ("AM1" or "PM7"; PM7 is scaffolded in this branch).
    charge : int
        Molecular charge.
    """

    def __init__(self, method: str = "AM1", charge: int = 0, device: str = "cpu"):
        self.method = method.upper()
        self.charge = charge
        self.device = _normalize_device(device)
        self.params = load_params(self.method)
        if self.params.is_placeholder:
            raise NotImplementedError(
                f"{self.method} parameters are scaffold-only in this tree and not validated yet"
            )

    def energy(
        self,
        symbols: Sequence[str],
        coords: np.ndarray,
        coords_unit: Literal["bohr", "angstrom"] = "angstrom",
        device: Optional[Literal["cpu", "cuda"]] = None,
        **scf_kwargs,
    ) -> SCFResult:
        """Compute SCF energy.

        Parameters
        ----------
        symbols : sequence of str
            Element symbols (e.g., ["H", "H"]).
        coords : (N, 3) array
            Atomic coordinates.
        coords_unit : str
            Unit of coordinates ("bohr" or "angstrom").
        **scf_kwargs
            Additional SCF options forwarded to :func:`asuka.semiempirical.scf.am1_scf`.
            The optional ``fock_mode`` keyword is validated as ``"ri"``,
            ``"w"``, or ``"auto"``.

        Returns
        -------
        SCFResult
        """
        coords = np.asarray(coords, dtype=float)
        if coords_unit == "angstrom":
            coords_bohr = coords * ANGSTROM_TO_BOHR
        elif coords_unit == "bohr":
            coords_bohr = coords
        else:
            raise ValueError("coords_unit must be 'bohr' or 'angstrom'")

        atomic_numbers = [symbol_to_Z(s) for s in symbols]
        run_device = self.device if device is None else _normalize_device(device)
        scf_kwargs = _normalize_fock_mode_kwargs(scf_kwargs)
        scf_kwargs, _ = _pop_gradient_backend_kwargs(scf_kwargs)
        return am1_scf(
            atomic_numbers, coords_bohr, self.params,
            charge=self.charge, device=run_device, **scf_kwargs,
        )

    def energy_gradient(
        self,
        symbols: Sequence[str],
        coords: np.ndarray,
        coords_unit: Literal["bohr", "angstrom"] = "angstrom",
        device: Optional[Literal["cpu", "cuda"]] = None,
        step_bohr: float = 1e-4,
        return_gradient_metadata: bool = False,
        **scf_kwargs,
    ) -> Union[Tuple[SCFResult, np.ndarray], Tuple[SCFResult, np.ndarray, Dict[str, Any]]]:
        """Compute AM1 SCF energy + Cartesian gradient.

        Notes
        -----
        ``gradient_backend`` may be provided in ``scf_kwargs`` as
        ``"auto"``, ``"cuda_analytic"``, or ``"cpu_frozen"``.

        Returns
        -------
        tuple
            ``(scf_result, grad)`` where ``grad`` has shape ``(natm, 3)``
            and units of Hartree/Bohr.
        """
        coords = np.asarray(coords, dtype=float)
        if coords_unit == "angstrom":
            coords_bohr = coords * ANGSTROM_TO_BOHR
        elif coords_unit == "bohr":
            coords_bohr = coords
        else:
            raise ValueError("coords_unit must be 'bohr' or 'angstrom'")

        atomic_numbers = [symbol_to_Z(s) for s in symbols]
        run_device = self.device if device is None else _normalize_device(device)
        scf_kwargs = _normalize_fock_mode_kwargs(scf_kwargs)
        scf_kwargs, gradient_backend = _pop_gradient_backend_kwargs(scf_kwargs)
        fock_mode = str(scf_kwargs.get("fock_mode", "ri")).strip().lower()
        scf_result = am1_scf(
            atomic_numbers,
            coords_bohr,
            self.params,
            charge=self.charge,
            device=run_device,
            **scf_kwargs,
        )
        out = am1_energy_gradient_scf(
            atomic_numbers=atomic_numbers,
            coords_bohr=coords_bohr,
            params=self.params,
            scf_result=scf_result,
            step_bohr=float(step_bohr),
            device=run_device,
            fock_mode=fock_mode,
            gradient_backend=gradient_backend,
            return_metadata=True,
        )
        _scf, grad, grad_meta = out
        if return_gradient_metadata:
            return _scf, grad, grad_meta
        return _scf, grad

    def gradient(
        self,
        symbols: Sequence[str],
        coords: np.ndarray,
        coords_unit: Literal["bohr", "angstrom"] = "angstrom",
        device: Optional[Literal["cpu", "cuda"]] = None,
        step_bohr: float = 1e-4,
        **scf_kwargs,
    ) -> np.ndarray:
        """Compute AM1 Cartesian gradient in Hartree/Bohr."""
        _scf, grad = self.energy_gradient(
            symbols,
            coords,
            coords_unit=coords_unit,
            device=device,
            step_bohr=step_bohr,
            **scf_kwargs,
        )
        return grad


def am1_energy(
    symbols: Sequence[str],
    coords: np.ndarray,
    charge: int = 0,
    device: Literal["cpu", "cuda"] = "cpu",
    coords_unit: Literal["bohr", "angstrom"] = "angstrom",
    return_details: bool = False,
    **scf_kwargs,
) -> Union[float, Dict[str, Any]]:
    """Convenience function for AM1 single-point energy.

    Parameters
    ----------
    symbols : sequence of str
        Element symbols.
    coords : (N, 3) array
        Atomic coordinates.
    charge : int
        Molecular charge.
    coords_unit : str
        Unit of coordinates.
    return_details : bool
        If True, return a dictionary with full SCF details.
    **scf_kwargs
        Additional SCF options. ``fock_mode`` is supported as ``"ri"``
        (default), ``"w"``, or ``"auto"`` for two-center CUDA Fock assembly.

    Returns
    -------
    float or dict
        Total energy in Hartree, or dict with full details.
    """
    scf_kwargs = _normalize_fock_mode_kwargs(scf_kwargs)
    scf_kwargs, _ = _pop_gradient_backend_kwargs(scf_kwargs)
    calc = SemiempiricalCalculator(method="AM1", charge=charge, device=device)
    result = calc.energy(symbols, coords, coords_unit=coords_unit, **scf_kwargs)

    if not return_details:
        return result.energy_total

    return {
        "method": "AM1",
        "charge": charge,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "energy_electronic": result.energy_electronic,
        "energy_core": result.energy_core,
        "energy_total": result.energy_total,
        "eps": result.eps,
        "C": result.C,
        "P": result.P,
    }


def am1_gradient(
    symbols: Sequence[str],
    coords: np.ndarray,
    charge: int = 0,
    device: Literal["cpu", "cuda"] = "cpu",
    coords_unit: Literal["bohr", "angstrom"] = "angstrom",
    step_bohr: float = 1e-4,
    **scf_kwargs,
) -> np.ndarray:
    """Convenience function for AM1 Cartesian gradient (Hartree/Bohr).

    ``gradient_backend`` is supported as ``\"auto\"`` (default),
    ``\"cuda_analytic\"``, or ``\"cpu_frozen\"``.
    """
    scf_kwargs = _normalize_fock_mode_kwargs(scf_kwargs)
    scf_kwargs, gradient_backend = _pop_gradient_backend_kwargs(scf_kwargs)
    scf_kwargs["gradient_backend"] = gradient_backend
    calc = SemiempiricalCalculator(method="AM1", charge=charge, device=device)
    return calc.gradient(
        symbols,
        coords,
        coords_unit=coords_unit,
        step_bohr=step_bohr,
        **scf_kwargs,
    )


def am1_energy_gradient(
    symbols: Sequence[str],
    coords: np.ndarray,
    charge: int = 0,
    device: Literal["cpu", "cuda"] = "cpu",
    coords_unit: Literal["bohr", "angstrom"] = "angstrom",
    step_bohr: float = 1e-4,
    return_details: bool = False,
    **scf_kwargs,
) -> Union[Tuple[float, np.ndarray], Dict[str, Any]]:
    """Convenience function for AM1 energy + Cartesian gradient.

    ``gradient_backend`` is supported as ``\"auto\"`` (default),
    ``\"cuda_analytic\"``, or ``\"cpu_frozen\"``.

    Returns
    -------
    tuple or dict
        If ``return_details`` is False, return ``(energy_total_ha, gradient)``.
        Otherwise return the standard AM1 details dictionary with an added
        ``"gradient"`` key in Hartree/Bohr and gradient backend timing fields.
    """
    scf_kwargs = _normalize_fock_mode_kwargs(scf_kwargs)
    scf_kwargs, gradient_backend = _pop_gradient_backend_kwargs(scf_kwargs)
    scf_kwargs["gradient_backend"] = gradient_backend
    calc = SemiempiricalCalculator(method="AM1", charge=charge, device=device)
    out = calc.energy_gradient(
        symbols,
        coords,
        coords_unit=coords_unit,
        step_bohr=step_bohr,
        return_gradient_metadata=return_details,
        **scf_kwargs,
    )
    if return_details:
        result, grad, grad_meta = out
    else:
        result, grad = out

    if not return_details:
        return float(result.energy_total), grad

    details = {
        "method": "AM1",
        "charge": charge,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "energy_electronic": result.energy_electronic,
        "energy_core": result.energy_core,
        "energy_total": result.energy_total,
        "eps": result.eps,
        "C": result.C,
        "P": result.P,
        "gradient": grad,
    }
    details["gradient_backend_used"] = grad_meta.get("gradient_backend_used", "cpu_frozen")
    details["gradient_pack_time_s"] = float(grad_meta.get("gradient_pack_time_s", 0.0))
    details["gradient_kernel_time_s"] = float(grad_meta.get("gradient_kernel_time_s", 0.0))
    details["gradient_post_time_s"] = float(grad_meta.get("gradient_post_time_s", 0.0))
    if "gradient_fallback_reason" in grad_meta:
        details["gradient_fallback_reason"] = str(grad_meta["gradient_fallback_reason"])
    return details


def pm7_energy(
    symbols: Sequence[str],
    coords: np.ndarray,
    charge: int = 0,
    device: Literal["cpu", "cuda"] = "cpu",
    coords_unit: Literal["bohr", "angstrom"] = "angstrom",
    return_details: bool = False,
    **scf_kwargs,
) -> Union[float, Dict[str, Any]]:
    """Convenience function for PM7 single-point energy.

    PM7 parameterization/corrections are scaffolded but not finalized in this branch.
    """
    calc = SemiempiricalCalculator(method="PM7", charge=charge, device=device)
    result = calc.energy(symbols, coords, coords_unit=coords_unit, **scf_kwargs)

    if not return_details:
        return result.energy_total

    return {
        "method": "PM7",
        "charge": charge,
        "converged": result.converged,
        "n_iter": result.n_iter,
        "energy_electronic": result.energy_electronic,
        "energy_core": result.energy_core,
        "energy_total": result.energy_total,
        "eps": result.eps,
        "C": result.C,
        "P": result.P,
    }
