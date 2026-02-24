from __future__ import annotations

"""Convenience helpers for geomopt and vibrational analysis.

These helpers are designed for workflows built around `asuka.frontend.Molecule`.
They run the underlying algorithms and store the resulting objects under
`mol.results[...]` for convenient downstream use.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from asuka.frontend.molecule import Molecule


EnergyGradFn = Callable[[np.ndarray], tuple[float, np.ndarray]]
GradFn = Callable[[np.ndarray], np.ndarray | tuple[float, np.ndarray]]


@dataclass(frozen=True)
class DFMethodEvalArtifacts:
    """Optional container for storing intermediate method artifacts per evaluation."""

    scf_out: Any | None
    mc: Any | None
    grad: Any | None


def _clone_molecule_with_coords(mol: Molecule, coords_bohr: np.ndarray) -> Molecule:
    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    if coords.shape[0] != int(mol.natm):
        raise ValueError("coords_bohr has wrong natm")
    atoms = tuple((sym, np.asarray(coords[i], dtype=np.float64).copy()) for i, (sym, _xyz) in enumerate(mol.atoms_bohr))
    return Molecule(atoms_bohr=atoms, charge=int(mol.charge), spin=int(mol.spin), basis=mol.basis, cart=bool(mol.cart))


def make_df_casscf_energy_grad(
    mol: Molecule,
    *,
    hf_kwargs: Mapping[str, Any] | None = None,
    casscf_kwargs: Mapping[str, Any] | None = None,
    grad_kwargs: Mapping[str, Any] | None = None,
    save_key: str = "method_eval",
    save_intermediates: bool = True,
    warm_start: bool = True,
    guess: Any | None = None,
    orbital_tracking: bool = True,
    tracking_method: str = "subspace",
    tracking_ref: Any | None = None,
) -> EnergyGradFn:
    """Build an `energy_grad(coords_bohr)` callback for DF-CASSCF.

    This adapter re-runs SCF + CASSCF at the requested coordinates and returns
    the CASSCF total energy and analytic DF nuclear gradient.

    The evaluation is performed on a geometry-cloned `Molecule` so the input
    `mol` coordinates are not mutated by calls to the returned callback.

    Parameters
    ----------
    orbital_tracking : bool
        Enable orbital tracking to maintain active space continuity across
        geometry changes (default: True). When enabled, uses cross-geometry
        overlap to identify which new orbitals should form the active space,
        preventing active space drift during MD/geomopt/scans.
    tracking_method : str
        Method for orbital assignment: "subspace" (default, robust) or
        "hungarian" (requires scipy, good for small CAS).
    tracking_ref : Any | None
        Reference calculation to track from. Can be:
        - A Molecule object (uses its geometry as reference)
        - A CASSCF result object (extracts mol, mo_coeff, ncore, ncas)
        - A tuple (mol, casscf_result)
        - A DFMethodEvalArtifacts object
        If None, uses automatic warm_start state tracking.

    Examples
    --------
    # Automatic tracking (state stored internally)
    >>> energy_grad = make_df_casscf_energy_grad(mol, ...)

    # Explicit reference from previous calculation
    >>> ref_result = run_casscf(...)
    >>> energy_grad = make_df_casscf_energy_grad(
    ...     mol, tracking_ref=ref_result, ...
    ... )

    # Track from a specific geometry
    >>> ref_mol = Molecule.from_atoms(...)
    >>> energy_grad = make_df_casscf_energy_grad(
    ...     mol, tracking_ref=(ref_mol, ref_casscf), ...
    ... )
    """

    hf_kwargs_use = dict(hf_kwargs or {})
    casscf_kwargs_use = dict(casscf_kwargs or {})
    grad_kwargs_use = dict(grad_kwargs or {})

    # Pick a reasonable HF flavor if not specified.
    if "method" not in hf_kwargs_use:
        hf_kwargs_use["method"] = "rhf" if int(mol.spin) == 0 else "rohf"

    # Extract ncore/ncas for orbital tracking
    ncore = casscf_kwargs_use.get("ncore")
    ncas = casscf_kwargs_use.get("ncas")
    if orbital_tracking and (ncore is None or ncas is None):
        raise ValueError(
            "orbital_tracking=True requires 'ncore' and 'ncas' in casscf_kwargs"
        )

    prev_scf_mo_coeff: Any | None = None
    prev_casscf_mo_coeff: Any | None = None
    prev_ci: Any | None = None
    prev_mol: Molecule | None = None
    prev_ncore: int | None = None
    prev_ncas: int | None = None

    if guess is not None:
        g_scf = None
        g_mc = None
        if isinstance(guess, tuple) and len(guess) == 2:
            g_scf, g_mc = guess
        else:
            g_scf = getattr(guess, "scf_out", None)
            g_mc = getattr(guess, "mc", None)
            if g_scf is None and hasattr(guess, "scf"):
                g_scf = guess
            if g_mc is None and hasattr(guess, "mo_coeff"):
                g_mc = guess

        if g_scf is not None:
            scf_guess = getattr(g_scf, "scf", g_scf)
            prev_scf_mo_coeff = getattr(scf_guess, "mo_coeff", None)
        if g_mc is not None:
            prev_casscf_mo_coeff = getattr(g_mc, "mo_coeff", None)
            prev_ci = getattr(g_mc, "ci", None)
            if prev_scf_mo_coeff is None and prev_casscf_mo_coeff is not None:
                prev_scf_mo_coeff = prev_casscf_mo_coeff

    # Parse tracking_ref for explicit reference geometry/calculation
    if tracking_ref is not None and orbital_tracking:
        ref_mol_parsed = None
        ref_casscf_parsed = None

        # Case 1: tracking_ref is a Molecule
        if isinstance(tracking_ref, Molecule):
            ref_mol_parsed = tracking_ref

        # Case 2: tracking_ref is a tuple (mol, casscf_result)
        elif isinstance(tracking_ref, tuple) and len(tracking_ref) == 2:
            ref_mol_parsed, ref_casscf_parsed = tracking_ref

        # Case 3: tracking_ref is a CASSCF result (has mol attribute)
        elif hasattr(tracking_ref, "mol"):
            ref_mol_parsed = getattr(tracking_ref, "mol", None)
            ref_casscf_parsed = tracking_ref

        # Case 4: tracking_ref is a DFMethodEvalArtifacts
        elif isinstance(tracking_ref, DFMethodEvalArtifacts):
            ref_scf = getattr(tracking_ref, "scf_out", None)
            ref_mc = getattr(tracking_ref, "mc", None)
            if ref_scf is not None and hasattr(ref_scf, "mol"):
                ref_mol_parsed = ref_scf.mol
            elif ref_mc is not None and hasattr(ref_mc, "mol"):
                ref_mol_parsed = ref_mc.mol
            ref_casscf_parsed = ref_mc

        # Extract orbital tracking info from parsed reference
        if ref_mol_parsed is not None:
            prev_mol = ref_mol_parsed

        if ref_casscf_parsed is not None:
            prev_casscf_mo_coeff = getattr(ref_casscf_parsed, "mo_coeff", None)
            prev_ncore = getattr(ref_casscf_parsed, "ncore", ncore)
            prev_ncas = getattr(ref_casscf_parsed, "ncas", ncas)
            # Also use for warm start if not already set
            if prev_ci is None:
                prev_ci = getattr(ref_casscf_parsed, "ci", None)
            if prev_scf_mo_coeff is None and prev_casscf_mo_coeff is not None:
                prev_scf_mo_coeff = prev_casscf_mo_coeff

    def energy_grad(coords_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        from asuka.frontend import run_hf  # noqa: PLC0415
        from asuka.mcscf import run_casscf  # noqa: PLC0415
        from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df  # noqa: PLC0415

        nonlocal prev_scf_mo_coeff, prev_casscf_mo_coeff, prev_ci, prev_mol, prev_ncore, prev_ncas

        if bool(warm_start) and (prev_scf_mo_coeff is None and prev_casscf_mo_coeff is None and prev_ci is None):
            prev = mol.results.get(str(save_key))
            if isinstance(prev, DFMethodEvalArtifacts):
                scf_prev = getattr(prev, "scf_out", None)
                mc_prev = getattr(prev, "mc", None)
                try:
                    prev_scf_mo_coeff = getattr(getattr(scf_prev, "scf", None), "mo_coeff", None)
                except Exception:
                    prev_scf_mo_coeff = None
                try:
                    prev_casscf_mo_coeff = getattr(mc_prev, "mo_coeff", None)
                    prev_ci = getattr(mc_prev, "ci", None)
                except Exception:
                    prev_casscf_mo_coeff = None
                    prev_ci = None

        mol_eval = _clone_molecule_with_coords(mol, coords_bohr)

        hf_call = dict(hf_kwargs_use)
        if bool(warm_start) and ("dm0" not in hf_call and "mo_coeff0" not in hf_call):
            if prev_scf_mo_coeff is not None:
                hf_call["mo_coeff0"] = prev_scf_mo_coeff

        scf_out = run_hf(mol_eval, **hf_call)

        # Orbital tracking: reorder SCF orbitals to match previous active space
        if bool(orbital_tracking) and bool(warm_start):
            if prev_mol is not None and prev_casscf_mo_coeff is not None:
                if prev_ncore is not None and prev_ncas is not None:
                    from asuka.frontend.one_electron import build_ao_basis_cart  # noqa: PLC0415
                    from asuka.integrals.cross_geometry import build_S_cross_cart  # noqa: PLC0415
                    from asuka.mcscf.orbital_tracking import (  # noqa: PLC0415
                        align_orbital_phases,
                        assign_active_orbitals_by_overlap,
                        reorder_mo_to_active_space,
                    )

                    # Build basis for both geometries
                    basis_prev, _ = build_ao_basis_cart(prev_mol)
                    basis_new, _ = build_ao_basis_cart(mol_eval)

                    # Compute cross-geometry overlap
                    S_cross = build_S_cross_cart(basis_prev, basis_new)

                    # Get new SCF orbitals
                    scf_mo_coeff = np.asarray(
                        getattr(getattr(scf_out, "scf", None), "mo_coeff", None),
                        dtype=np.float64,
                    )

                    # Identify which new orbitals match previous active space
                    prev_active_idx = list(range(prev_ncore, prev_ncore + prev_ncas))
                    new_active_idx = assign_active_orbitals_by_overlap(
                        prev_casscf_mo_coeff,
                        scf_mo_coeff,
                        S_cross,
                        prev_active_idx,
                        ncas,
                        method=tracking_method,
                    )

                    # Reorder SCF orbitals to place matched orbitals in active space
                    scf_mo_reordered = reorder_mo_to_active_space(
                        scf_mo_coeff, new_active_idx, ncore
                    )

                    # Align phases for continuity
                    scf_mo_aligned = align_orbital_phases(
                        prev_casscf_mo_coeff,
                        scf_mo_reordered,
                        S_cross,
                        alignment_idx=range(ncore, ncore + ncas),
                    )

                    # Override SCF mo_coeff for CASSCF initial guess
                    if hasattr(scf_out, "scf") and hasattr(scf_out.scf, "mo_coeff"):
                        scf_out.scf.mo_coeff = scf_mo_aligned

        casscf_call = dict(casscf_kwargs_use)
        if bool(warm_start):
            if "mo_coeff0" not in casscf_call and prev_casscf_mo_coeff is not None:
                casscf_call["mo_coeff0"] = prev_casscf_mo_coeff
            if "ci0" not in casscf_call and prev_ci is not None:
                casscf_call["ci0"] = prev_ci

        mc = run_casscf(scf_out, **casscf_call)
        g = casscf_nuc_grad_df(scf_out, mc, **grad_kwargs_use)

        if bool(save_intermediates):
            mol.results[str(save_key)] = DFMethodEvalArtifacts(scf_out=scf_out, mc=mc, grad=g)

        if bool(warm_start) and bool(getattr(getattr(scf_out, "scf", None), "converged", False)):
            prev_scf_mo_coeff = getattr(getattr(scf_out, "scf", None), "mo_coeff", None)
        if bool(warm_start) and bool(getattr(mc, "converged", False)):
            prev_casscf_mo_coeff = getattr(mc, "mo_coeff", None)
            prev_ci = getattr(mc, "ci", None)
            # Store geometry and active space info for orbital tracking
            if bool(orbital_tracking):
                prev_mol = mol_eval
                prev_ncore = ncore
                prev_ncas = ncas

        return float(g.e_tot), np.asarray(g.grad, dtype=np.float64)

    return energy_grad


def make_df_casci_energy_grad(
    mol: Molecule,
    *,
    hf_kwargs: Mapping[str, Any] | None = None,
    casci_kwargs: Mapping[str, Any] | None = None,
    grad_kwargs: Mapping[str, Any] | None = None,
    relaxed: bool = True,
    save_key: str = "method_eval",
    save_intermediates: bool = True,
    warm_start: bool = True,
    guess: Any | None = None,
) -> EnergyGradFn:
    """Build an `energy_grad(coords_bohr)` callback for DF-CASCI.

    The evaluation is performed on a geometry-cloned `Molecule` so the input
    `mol` coordinates are not mutated by calls to the returned callback.
    """

    hf_kwargs_use = dict(hf_kwargs or {})
    casci_kwargs_use = dict(casci_kwargs or {})
    grad_kwargs_use = dict(grad_kwargs or {})

    if "method" not in hf_kwargs_use:
        hf_kwargs_use["method"] = "rhf" if int(mol.spin) == 0 else "rohf"

    prev_scf_mo_coeff: Any | None = None
    prev_ci: Any | None = None

    if guess is not None:
        g_scf = None
        g_mc = None
        if isinstance(guess, tuple) and len(guess) == 2:
            g_scf, g_mc = guess
        else:
            g_scf = getattr(guess, "scf_out", None)
            g_mc = getattr(guess, "mc", None)
            if g_scf is None and hasattr(guess, "scf"):
                g_scf = guess
            if g_mc is None and hasattr(guess, "ci"):
                g_mc = guess

        if g_scf is not None:
            scf_guess = getattr(g_scf, "scf", g_scf)
            prev_scf_mo_coeff = getattr(scf_guess, "mo_coeff", None)
        if g_mc is not None:
            prev_ci = getattr(g_mc, "ci", None)

    def energy_grad(coords_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        from asuka.frontend import run_hf  # noqa: PLC0415
        from asuka.mcscf import run_casci  # noqa: PLC0415
        from asuka.mcscf.nuc_grad_df import casci_nuc_grad_df_relaxed, casci_nuc_grad_df_unrelaxed  # noqa: PLC0415

        nonlocal prev_scf_mo_coeff, prev_ci

        if bool(warm_start) and (prev_scf_mo_coeff is None and prev_ci is None):
            prev = mol.results.get(str(save_key))
            if isinstance(prev, DFMethodEvalArtifacts):
                scf_prev = getattr(prev, "scf_out", None)
                mc_prev = getattr(prev, "mc", None)
                try:
                    prev_scf_mo_coeff = getattr(getattr(scf_prev, "scf", None), "mo_coeff", None)
                except Exception:
                    prev_scf_mo_coeff = None
                try:
                    prev_ci = getattr(mc_prev, "ci", None)
                except Exception:
                    prev_ci = None

        mol_eval = _clone_molecule_with_coords(mol, coords_bohr)

        hf_call = dict(hf_kwargs_use)
        if bool(warm_start) and ("dm0" not in hf_call and "mo_coeff0" not in hf_call):
            if prev_scf_mo_coeff is not None:
                hf_call["mo_coeff0"] = prev_scf_mo_coeff
        scf_out = run_hf(mol_eval, **hf_call)

        casci_call = dict(casci_kwargs_use)
        if bool(warm_start) and "ci0" not in casci_call and prev_ci is not None:
            casci_call["ci0"] = prev_ci
        mc = run_casci(scf_out, **casci_call)
        if bool(relaxed):
            g = casci_nuc_grad_df_relaxed(scf_out, mc, **grad_kwargs_use)
        else:
            g = casci_nuc_grad_df_unrelaxed(scf_out, mc, **grad_kwargs_use)

        if bool(save_intermediates):
            mol.results[str(save_key)] = DFMethodEvalArtifacts(scf_out=scf_out, mc=mc, grad=g)

        if bool(warm_start) and bool(getattr(getattr(scf_out, "scf", None), "converged", False)):
            prev_scf_mo_coeff = getattr(getattr(scf_out, "scf", None), "mo_coeff", None)
        # CASCI result carries CI, and we can use it as a CI solver initial guess next time.
        prev_ci = getattr(mc, "ci", None) if bool(warm_start) else prev_ci

        return float(g.e_tot), np.asarray(g.grad, dtype=np.float64)

    return energy_grad


MultirootEnergyGradFn = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]


def make_df_casscf_multiroot_energy_grad(
    mol: Molecule,
    *,
    hf_kwargs: Mapping[str, Any] | None = None,
    casscf_kwargs: Mapping[str, Any] | None = None,
    grad_kwargs: Mapping[str, Any] | None = None,
    save_key: str = "method_eval_multiroot",
    save_intermediates: bool = True,
    warm_start: bool = True,
    guess: Any | None = None,
) -> MultirootEnergyGradFn:
    """Build a ``(coords_bohr) -> (e_roots, grads)`` callback for per-root SA-CASSCF gradients.

    This adapter re-runs SCF + CASSCF at the requested coordinates and returns
    per-root energies and per-root analytic DF nuclear gradients (with CP-MCSCF
    orbital response).

    Returns
    -------
    MultirootEnergyGradFn
        A callable ``(coords_bohr) -> (e_roots, grads)`` where ``e_roots`` has
        shape ``(nroots,)`` and ``grads`` has shape ``(nroots, natm, 3)``.
    """

    hf_kwargs_use = dict(hf_kwargs or {})
    casscf_kwargs_use = dict(casscf_kwargs or {})
    grad_kwargs_use = dict(grad_kwargs or {})

    if "method" not in hf_kwargs_use:
        hf_kwargs_use["method"] = "rhf" if int(mol.spin) == 0 else "rohf"

    prev_scf_mo_coeff: Any | None = None
    prev_casscf_mo_coeff: Any | None = None
    prev_ci: Any | None = None

    if guess is not None:
        g_scf = None
        g_mc = None
        if isinstance(guess, tuple) and len(guess) == 2:
            g_scf, g_mc = guess
        else:
            g_scf = getattr(guess, "scf_out", None)
            g_mc = getattr(guess, "mc", None)
            if g_scf is None and hasattr(guess, "scf"):
                g_scf = guess
            if g_mc is None and hasattr(guess, "mo_coeff"):
                g_mc = guess

        if g_scf is not None:
            scf_guess = getattr(g_scf, "scf", g_scf)
            prev_scf_mo_coeff = getattr(scf_guess, "mo_coeff", None)
        if g_mc is not None:
            prev_casscf_mo_coeff = getattr(g_mc, "mo_coeff", None)
            prev_ci = getattr(g_mc, "ci", None)
            if prev_scf_mo_coeff is None and prev_casscf_mo_coeff is not None:
                prev_scf_mo_coeff = prev_casscf_mo_coeff

    def energy_grad(coords_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from asuka.frontend import run_hf  # noqa: PLC0415
        from asuka.mcscf import run_casscf  # noqa: PLC0415
        from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df_per_root  # noqa: PLC0415

        nonlocal prev_scf_mo_coeff, prev_casscf_mo_coeff, prev_ci

        if bool(warm_start) and (prev_scf_mo_coeff is None and prev_casscf_mo_coeff is None and prev_ci is None):
            prev = mol.results.get(str(save_key))
            if isinstance(prev, DFMethodEvalArtifacts):
                scf_prev = getattr(prev, "scf_out", None)
                mc_prev = getattr(prev, "mc", None)
                try:
                    prev_scf_mo_coeff = getattr(getattr(scf_prev, "scf", None), "mo_coeff", None)
                except Exception:
                    prev_scf_mo_coeff = None
                try:
                    prev_casscf_mo_coeff = getattr(mc_prev, "mo_coeff", None)
                    prev_ci = getattr(mc_prev, "ci", None)
                except Exception:
                    prev_casscf_mo_coeff = None
                    prev_ci = None

        mol_eval = _clone_molecule_with_coords(mol, coords_bohr)

        hf_call = dict(hf_kwargs_use)
        if bool(warm_start) and ("dm0" not in hf_call and "mo_coeff0" not in hf_call):
            if prev_scf_mo_coeff is not None:
                hf_call["mo_coeff0"] = prev_scf_mo_coeff

        scf_out = run_hf(mol_eval, **hf_call)

        casscf_call = dict(casscf_kwargs_use)
        if bool(warm_start):
            if "mo_coeff0" not in casscf_call and prev_casscf_mo_coeff is not None:
                casscf_call["mo_coeff0"] = prev_casscf_mo_coeff
            if "ci0" not in casscf_call and prev_ci is not None:
                casscf_call["ci0"] = prev_ci

        mc = run_casscf(scf_out, **casscf_call)
        g = casscf_nuc_grad_df_per_root(scf_out, mc, **grad_kwargs_use)

        if bool(save_intermediates):
            mol.results[str(save_key)] = DFMethodEvalArtifacts(scf_out=scf_out, mc=mc, grad=g)

        if bool(warm_start) and bool(getattr(getattr(scf_out, "scf", None), "converged", False)):
            prev_scf_mo_coeff = getattr(getattr(scf_out, "scf", None), "mo_coeff", None)
        if bool(warm_start) and bool(getattr(mc, "converged", False)):
            prev_casscf_mo_coeff = getattr(mc, "mo_coeff", None)
            prev_ci = getattr(mc, "ci", None)

        return np.asarray(g.e_roots, dtype=np.float64), np.asarray(g.grads, dtype=np.float64)

    return energy_grad


def geomopt_molecule(
    mol: Molecule,
    energy_grad: EnergyGradFn,
    *,
    settings: Any | None = None,
    save_key: str = "geomopt",
    update_geometry: bool = True,
):
    """Run Cartesian minimum optimization and attach results to the Molecule."""

    from asuka.geomopt.optimizer import GeomOptSettings, optimize_cartesian  # noqa: PLC0415

    coords0 = np.asarray(mol.coords_bohr, dtype=np.float64)
    st = GeomOptSettings() if settings is None else settings
    res = optimize_cartesian(energy_grad, coords0, settings=st)
    mol.results[str(save_key)] = res
    if bool(update_geometry):
        mol.set_coords_bohr_inplace(res.coords_final_bohr)
    return res


def fd_hessian_molecule(
    mol: Molecule,
    grad_fn: GradFn,
    *,
    step_bohr: float = 1e-3,
    method: str = "central",
    symmetrize: bool = True,
    verbose: int = 0,
    save_key: str = "hessian_fd",
):
    """Build an FD Cartesian Hessian from gradients and store it on the Molecule.

    `grad_fn(coords_bohr)` may return either a gradient array or an `(E, grad)`
    tuple.
    """

    from asuka.vib.hessian_fd import fd_cartesian_hessian  # noqa: PLC0415

    coords0 = np.asarray(mol.coords_bohr, dtype=np.float64)
    hres = fd_cartesian_hessian(
        grad_fn,
        coords0,
        step_bohr=float(step_bohr),
        method=str(method),
        symmetrize=bool(symmetrize),
        verbose=int(verbose),
    )
    mol.results[str(save_key)] = hres
    return hres


def frequency_analysis_molecule(
    mol: Molecule,
    hessian_cart: np.ndarray,
    *,
    masses_amu: Sequence[float] | None = None,
    linear: bool | None = None,
    tr_tol: float = 1e-10,
    symmetrize: bool = True,
    seed: int = 0,
    save_key: str = "normal_modes",
):
    """Harmonic frequency analysis for a Molecule and store it on the Molecule."""

    from asuka.vib.frequency import frequency_analysis  # noqa: PLC0415

    if masses_amu is None:
        masses = mol.atom_mass_list()
    else:
        masses = list(masses_amu)

    nm = frequency_analysis(
        hessian_cart=np.asarray(hessian_cart, dtype=np.float64),
        mol=mol,
        coords_bohr=np.asarray(mol.coords_bohr, dtype=np.float64),
        masses_amu=masses,
        linear=linear,
        tr_tol=float(tr_tol),
        symmetrize=bool(symmetrize),
        seed=int(seed),
    )
    mol.results[str(save_key)] = nm
    return nm


__all__ = [
    "EnergyGradFn",
    "GradFn",
    "DFMethodEvalArtifacts",
    "MethodWorkflow",
    "make_df_casscf_energy_grad",
    "make_df_casci_energy_grad",
    "geomopt_molecule",
    "fd_hessian_molecule",
    "frequency_analysis_molecule",
]


@dataclass(frozen=True)
class MethodWorkflow:
    """High-level workflow wrapper around a Molecule + an `energy_grad` callback.

    This is the ergonomic layer that "internalizes" the common plumbing:
    - Builds method-specific `(E, grad)` evaluators (e.g. DF-CASSCF) with warm-start.
    - Runs geomopt/FD Hessians/frequency analysis while storing results on `mol.results[...]`.
    """

    mol: Molecule
    energy_grad: EnergyGradFn

    def __call__(self, coords_bohr: np.ndarray) -> tuple[float, np.ndarray]:
        return self.energy_grad(coords_bohr)

    @classmethod
    def from_energy_grad(cls, mol: Molecule, energy_grad: EnergyGradFn) -> "MethodWorkflow":
        return cls(mol=mol, energy_grad=energy_grad)

    @classmethod
    def from_method(cls, mol: Molecule, method: str, /, **kwargs) -> "MethodWorkflow":
        """Create a workflow from a method name and keyword arguments.

        Supported method names:
        - "df_casscf"
        - "df_casci"
        """

        m = str(method).strip().lower()
        if m in {"df_casscf", "casscf_df"}:
            return cls.df_casscf_method(mol, **kwargs)
        if m in {"df_casci", "casci_df"}:
            return cls.df_casci_method(mol, **kwargs)
        raise ValueError(f"unsupported method={method!r} (expected one of: 'df_casscf', 'df_casci')")

    @classmethod
    def from_casscf_singlepoint(
        cls,
        scf_out: Any,
        casscf_out: Any,
        *,
        backend: str | None = None,
        df_backend: str | None = None,
        df_threads: int = 0,
        save_key: str = "method_eval",
        save_intermediates: bool = False,
        warm_start: bool = True,
        **overrides,
    ) -> "MethodWorkflow":
        """Create a DF-CASSCF workflow from a converged single-point.

        This is the ergonomic entry point for geometry scans/geomopt/FD Hessians:
        users run a single-point DF-HF + DF-CASSCF once, then pass the resulting
        objects here to seed initial guesses automatically.
        """

        mol = getattr(scf_out, "mol", None) or getattr(casscf_out, "mol", None)
        if not isinstance(mol, Molecule):
            raise TypeError("scf_out/mc_out must carry an asuka.frontend.Molecule under .mol")

        # Infer backend from the DF factors if not specified.
        if backend is None:
            df_B = getattr(scf_out, "df_B", None)
            if df_B is None:
                df_B = getattr(scf_out, "df_b", None)
            if df_B is None:
                df_B = getattr(scf_out, "df", None)
            backend_use = "cpu"
            try:
                import cupy as cp  # type: ignore

                if df_B is not None and isinstance(df_B, cp.ndarray):  # type: ignore[attr-defined]
                    backend_use = "cuda"
            except Exception:
                backend_use = "cpu"
        else:
            backend_use = str(backend).strip().lower()

        df_backend_use = backend_use if df_backend is None else str(df_backend).strip().lower()

        # Basis/auxbasis: prefer mol specs; otherwise fall back to names from the single-point output.
        basis_use = mol.basis
        if basis_use is None:
            bname = getattr(scf_out, "basis_name", None)
            if isinstance(bname, str) and bname not in {"<explicit>", "<unknown>"}:
                basis_use = bname

        aux_use: Any = "autoaux"
        aname = getattr(scf_out, "auxbasis_name", None)
        if isinstance(aname, str) and aname not in {"<explicit>", "<unknown>"}:
            aux_use = aname

        root_weights = getattr(casscf_out, "root_weights", None)

        kwargs = dict(
            backend=backend_use,
            basis=basis_use,
            auxbasis=aux_use,
            ncore=int(getattr(casscf_out, "ncore")),
            ncas=int(getattr(casscf_out, "ncas")),
            nelecas=getattr(casscf_out, "nelecas"),
            nroots=int(getattr(casscf_out, "nroots", 1)),
            root_weights=None if root_weights is None else tuple(float(x) for x in np.asarray(root_weights).ravel().tolist()),
            df_threads=int(df_threads),
            save_key=str(save_key),
            save_intermediates=bool(save_intermediates),
            warm_start=bool(warm_start),
            guess=(scf_out, casscf_out),
        )
        kwargs.update(overrides)
        return cls.df_casscf_method(mol, **kwargs)

    @classmethod
    def df_casscf_method(
        cls,
        mol: Molecule,
        *,
        ncore: int,
        ncas: int,
        nelecas: int | tuple[int, int],
        nroots: int = 1,
        root_weights: Sequence[float] | None = None,
        backend: str = "cpu",
        basis: Any | None = None,
        auxbasis: Any = "autoaux",
        hf_method: str | None = None,
        max_cycle_scf: int = 80,
        conv_tol_scf: float = 1e-12,
        conv_tol_dm_scf: float = 1e-9,
        max_cycle_macro: int = 80,
        tol: float = 1e-8,
        conv_tol_grad: float = 2e-4,
        max_stepsize: float = 0.2,
        df_threads: int = 0,
        save_key: str = "method_eval",
        save_intermediates: bool = True,
        warm_start: bool = True,
        guess: Any | None = None,
        hf_kwargs: Mapping[str, Any] | None = None,
        casscf_kwargs: Mapping[str, Any] | None = None,
        grad_kwargs: Mapping[str, Any] | None = None,
    ) -> "MethodWorkflow":
        """Convenience constructor for DF-CASSCF (energy + analytic DF gradients)."""

        backend_s = str(backend).strip().lower()
        basis_use = mol.basis if basis is None else basis

        hf_call = dict(hf_kwargs or {})
        hf_call.setdefault("method", ("rhf" if int(mol.spin) == 0 else "rohf") if hf_method is None else str(hf_method))
        hf_call.setdefault("backend", backend_s)
        hf_call.setdefault("df", True)
        hf_call.setdefault("basis", basis_use)
        hf_call.setdefault("auxbasis", auxbasis)
        hf_call.setdefault("max_cycle", int(max_cycle_scf))
        hf_call.setdefault("conv_tol", float(conv_tol_scf))
        hf_call.setdefault("conv_tol_dm", float(conv_tol_dm_scf))

        casscf_call = dict(casscf_kwargs or {})
        casscf_call.setdefault("ncore", int(ncore))
        casscf_call.setdefault("ncas", int(ncas))
        casscf_call.setdefault("nelecas", nelecas)
        casscf_call.setdefault("nroots", int(nroots))
        if root_weights is not None:
            casscf_call.setdefault("root_weights", root_weights)
        casscf_call.setdefault("backend", backend_s)
        casscf_call.setdefault("df", True)
        casscf_call.setdefault("max_cycle_macro", int(max_cycle_macro))
        casscf_call.setdefault("tol", float(tol))
        casscf_call.setdefault("conv_tol_grad", float(conv_tol_grad))
        casscf_call.setdefault("max_stepsize", float(max_stepsize))

        grad_call = dict(grad_kwargs or {})
        grad_call.setdefault("df_backend", backend_s)
        grad_call.setdefault("df_threads", int(df_threads))

        return cls.df_casscf(
            mol,
            hf_kwargs=hf_call,
            casscf_kwargs=casscf_call,
            grad_kwargs=grad_call,
            save_key=str(save_key),
            save_intermediates=bool(save_intermediates),
            warm_start=bool(warm_start),
            guess=guess,
        )

    @classmethod
    def df_casci_method(
        cls,
        mol: Molecule,
        *,
        ncore: int,
        ncas: int,
        nelecas: int | tuple[int, int],
        nroots: int = 1,
        backend: str = "cpu",
        basis: Any | None = None,
        auxbasis: Any = "autoaux",
        hf_method: str | None = None,
        relaxed: bool = True,
        max_cycle_scf: int = 80,
        conv_tol_scf: float = 1e-12,
        conv_tol_dm_scf: float = 1e-9,
        df_threads: int = 0,
        save_key: str = "method_eval",
        save_intermediates: bool = True,
        warm_start: bool = True,
        guess: Any | None = None,
        hf_kwargs: Mapping[str, Any] | None = None,
        casci_kwargs: Mapping[str, Any] | None = None,
        grad_kwargs: Mapping[str, Any] | None = None,
    ) -> "MethodWorkflow":
        """Convenience constructor for DF-CASCI (energy + analytic DF gradients)."""

        backend_s = str(backend).strip().lower()
        basis_use = mol.basis if basis is None else basis

        hf_call = dict(hf_kwargs or {})
        hf_call.setdefault("method", ("rhf" if int(mol.spin) == 0 else "rohf") if hf_method is None else str(hf_method))
        hf_call.setdefault("backend", backend_s)
        hf_call.setdefault("df", True)
        hf_call.setdefault("basis", basis_use)
        hf_call.setdefault("auxbasis", auxbasis)
        hf_call.setdefault("max_cycle", int(max_cycle_scf))
        hf_call.setdefault("conv_tol", float(conv_tol_scf))
        hf_call.setdefault("conv_tol_dm", float(conv_tol_dm_scf))

        casci_call = dict(casci_kwargs or {})
        casci_call.setdefault("ncore", int(ncore))
        casci_call.setdefault("ncas", int(ncas))
        casci_call.setdefault("nelecas", nelecas)
        casci_call.setdefault("nroots", int(nroots))
        casci_call.setdefault("backend", backend_s)
        casci_call.setdefault("df", True)

        grad_call = dict(grad_kwargs or {})
        grad_call.setdefault("df_backend", backend_s)
        grad_call.setdefault("df_threads", int(df_threads))

        eg = make_df_casci_energy_grad(
            mol,
            hf_kwargs=hf_call,
            casci_kwargs=casci_call,
            grad_kwargs=grad_call,
            relaxed=bool(relaxed),
            save_key=str(save_key),
            save_intermediates=bool(save_intermediates),
            warm_start=bool(warm_start),
            guess=guess,
        )
        return cls(mol=mol, energy_grad=eg)

    @classmethod
    def df_casscf(
        cls,
        mol: Molecule,
        *,
        hf_kwargs: Mapping[str, Any] | None = None,
        casscf_kwargs: Mapping[str, Any] | None = None,
        grad_kwargs: Mapping[str, Any] | None = None,
        save_key: str = "method_eval",
        save_intermediates: bool = True,
        warm_start: bool = True,
        guess: Any | None = None,
    ) -> "MethodWorkflow":
        eg = make_df_casscf_energy_grad(
            mol,
            hf_kwargs=hf_kwargs,
            casscf_kwargs=casscf_kwargs,
            grad_kwargs=grad_kwargs,
            save_key=save_key,
            save_intermediates=bool(save_intermediates),
            warm_start=bool(warm_start),
            guess=guess,
        )
        return cls(mol=mol, energy_grad=eg)

    @classmethod
    def df_casci(
        cls,
        mol: Molecule,
        *,
        hf_kwargs: Mapping[str, Any] | None = None,
        casci_kwargs: Mapping[str, Any] | None = None,
        grad_kwargs: Mapping[str, Any] | None = None,
        relaxed: bool = True,
        save_key: str = "method_eval",
        save_intermediates: bool = True,
        warm_start: bool = True,
        guess: Any | None = None,
    ) -> "MethodWorkflow":
        eg = make_df_casci_energy_grad(
            mol,
            hf_kwargs=hf_kwargs,
            casci_kwargs=casci_kwargs,
            grad_kwargs=grad_kwargs,
            relaxed=bool(relaxed),
            save_key=save_key,
            save_intermediates=bool(save_intermediates),
            warm_start=bool(warm_start),
            guess=guess,
        )
        return cls(mol=mol, energy_grad=eg)

    def geomopt(
        self,
        *,
        settings: Any | None = None,
        save_key: str = "geomopt",
        update_geometry: bool = True,
    ):
        return geomopt_molecule(self.mol, self.energy_grad, settings=settings, save_key=save_key, update_geometry=update_geometry)

    def hessian_fd(
        self,
        *,
        step_bohr: float = 1e-3,
        fd_method: str = "central",
        symmetrize: bool = True,
        verbose: int = 0,
        save_key: str = "hessian_fd",
    ):
        return fd_hessian_molecule(
            self.mol,
            self.energy_grad,
            step_bohr=float(step_bohr),
            method=str(fd_method),
            symmetrize=bool(symmetrize),
            verbose=int(verbose),
            save_key=save_key,
        )

    def frequencies_fd(
        self,
        *,
        step_bohr: float = 1e-3,
        fd_method: str = "central",
        symmetrize_hessian: bool = True,
        hessian_verbose: int = 0,
        masses_amu: Sequence[float] | None = None,
        linear: bool | None = None,
        tr_tol: float = 1e-10,
        symmetrize_modes: bool = True,
        seed: int = 0,
        save_key_hessian: str = "hessian_fd",
        save_key_modes: str = "normal_modes",
    ):
        hres = self.hessian_fd(
            step_bohr=float(step_bohr),
            fd_method=str(fd_method),
            symmetrize=bool(symmetrize_hessian),
            verbose=int(hessian_verbose),
            save_key=save_key_hessian,
        )
        return frequency_analysis_molecule(
            self.mol,
            hres.hessian,
            masses_amu=masses_amu,
            linear=linear,
            tr_tol=float(tr_tol),
            symmetrize=bool(symmetrize_modes),
            seed=int(seed),
            save_key=save_key_modes,
        )
