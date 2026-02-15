from __future__ import annotations

"""Nuclear gradients (forces) for MRCI.

This module provides:
- **Analytic** gradients for uncontracted MRCISD (`method="mrcisd"`) via a
  CP-CASSCF Z-vector solve (`asuka/mrci/grad_analytic.py`).
- **Finite-difference (FD)** gradients for both uncontracted (`mrcisd`) and
  contracted (`ic_mrcisd`) methods (validation/robust fallback).
- A **heuristic** analytic option for `plus_q=True` by scaling the uncorrected
  correlation gradient with a constant Davidson denominator.

Why FD first?
-------------
Analytic MRCI gradients require a Lagrangian/Z-vector treatment because the
post-CAS CI energy is not variational with respect to orbital rotations and the
frozen-core effective Hamiltonian depends on the SCF response. FD gradients are
the simplest *correct* reference implementation and are valuable both for end
users (small systems) and for validating future analytic implementations.

Conventions
-----------
* Returned gradients are dE/dR in Eh/Bohr.
* Nuclear *forces* are the negative gradient: F = -dE/dR.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.mrci.driver import MRCIFromMCResult, assign_roots_by_overlap, mrci_from_mc, mrci_states_from_mc
from asuka.solver import GUGAFCISolver

_BOHR_TO_ANGSTROM = 0.52917721092


Backend = Literal["fd", "analytic"]
Method = Literal["mrcisd", "ic_mrcisd"]


@dataclass(frozen=True)
class MRCIGradResult:
    """Driver-level result for MRCI nuclear gradients.

    Attributes
    ----------
    method : str
        Method used ("mrcisd" or "ic_mrcisd").
    backend : str
        Gradient backend used ("fd" or "analytic").
    fd_step_bohr : float | None
        Finite-difference step size in Bohr, if used.
    e_ref : float
        Reference energy at the reference geometry.
    e_tot : float
        Total energy at the reference geometry.
    e_corr : float
        Correlation energy at the reference geometry.
    e_tot_used : float
        Total energy used for gradient computation (may include +Q correction).
    grad_ref : np.ndarray
        Reference gradient (Eh/Bohr).
    grad_tot : np.ndarray
        Total gradient (Eh/Bohr).
    grad_corr : np.ndarray
        Correlation gradient (Eh/Bohr).
    diagnostics : dict[str, float] | None
        Optional diagnostics from +Q or the underlying solver.
    """

    method: str
    backend: str

    fd_step_bohr: float | None

    # Energies at reference geometry.
    e_ref: float
    e_tot: float
    e_corr: float

    # Optional corrected energy used for gradients (e.g. MRCISD+Q).
    e_tot_used: float

    # Gradients (Eh/Bohr)
    grad_ref: np.ndarray
    grad_tot: np.ndarray
    grad_corr: np.ndarray

    # Optional diagnostics from +Q or the underlying solver.
    diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class StateGrad:
    state: int
    root: int

    e_ref: float
    e_tot: float
    e_corr: float
    e_tot_used: float

    grad_tot: np.ndarray
    grad_ref: np.ndarray | None
    grad_corr: np.ndarray | None

    diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class MRCIGradStatesResult:
    method: str
    backend: str
    fd_step_bohr: float | None
    states: list[int]
    state_results: list[StateGrad]
    overlap_ref_root: np.ndarray | None = None


def _require_native_guga_reference(mc: Any) -> None:
    fcisolver = getattr(mc, "fcisolver", None)
    if not isinstance(fcisolver, GUGAFCISolver):
        cls = type(fcisolver).__name__ if fcisolver is not None else "None"
        raise RuntimeError(
            "ASUKA native MRCI-grad path requires mc.fcisolver=asuka.solver.GUGAFCISolver; "
            f"got {cls}."
        )


def _infer_states(mc: Any, states: Sequence[int] | None) -> list[int]:
    if states is None:
        ci = getattr(mc, "ci")
        if isinstance(ci, (list, tuple)):
            return list(range(len(ci)))
        return [0]
    out = [int(s) for s in states]
    if not out:
        raise ValueError("states must be non-empty when provided")
    if len(set(out)) != len(out):
        raise ValueError("states must not contain duplicates")
    if any(s < 0 for s in out):
        raise ValueError("states must be non-negative")
    ci = getattr(mc, "ci")
    if isinstance(ci, (list, tuple)):
        n = int(len(ci))
        if any(s >= n for s in out):
            raise ValueError("state index out of range for mc.ci")
    else:
        if any(s != 0 for s in out):
            raise ValueError("state != 0 but mc.ci is not a list/tuple")
    return out


def mrci_grad_states_from_mc(
    mc: Any,
    *,
    method: Method = "mrcisd",
    backend: Backend = "analytic",
    states: Sequence[int] | None = None,
    root_follow: Literal["hungarian", "greedy"] = "hungarian",
    # FD options (used only for backend="fd")
    fd_step_bohr: float = 1e-3,
    fd_reset_to_ref: bool = True,
    # existing knobs
    atmlst=None,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    plus_q: bool = False,
    **mrci_kwargs: Any,
) -> MRCIGradStatesResult:
    """Compute MRCI nuclear gradients for multiple reference states.

    Parameters
    ----------
    mc : Any
        PySCF CASCI/CASSCF-like object.
    method : {"mrcisd", "ic_mrcisd"}, optional
        Method to use. Default is "mrcisd".
    backend : {"analytic", "fd"}, optional
        Gradient backend. Default is "analytic".
    states : Sequence[int] | None, optional
        Indices of states to compute.
    root_follow : {"hungarian", "greedy"}, optional
        Algorithm to assign roots to reference states. Default is "hungarian".
    fd_step_bohr : float, optional
        Finite-difference step size in Bohr. Default is 1e-3.
    fd_reset_to_ref : bool, optional
        Whether to reset scanner to reference geometry. Default is True.
    atmlst : Sequence[int] | None, optional
        Atom indices.
    rdm_backend : {"cuda", "cpu"}, optional
        RDM backend. Default is "cuda".
    plus_q : bool, optional
        Whether to include Davidson +Q correction. Default is False.
    mrci_kwargs : Any
        Additional keyword arguments for the MRCI driver.

    Returns
    -------
    MRCIGradStatesResult
        Result object containing gradients for all requested states.
    """
    _require_native_guga_reference(mc)

    backend_s = str(backend).strip().lower()
    if backend_s not in ("fd", "analytic"):
        raise ValueError("backend must be 'fd' or 'analytic'")

    method_s = str(method).strip().lower()
    if method_s not in ("mrcisd", "ic_mrcisd"):
        raise ValueError("method must be 'mrcisd' or 'ic_mrcisd'")

    root_follow_s = str(root_follow).strip().lower()
    if root_follow_s not in ("hungarian", "greedy"):
        raise ValueError("root_follow must be 'hungarian' or 'greedy'")

    # Optional controls that apply to gradients but are not accepted by mrci_from_mc.
    atmlst = mrci_kwargs.pop("atmlst", atmlst)
    rdm_backend = str(mrci_kwargs.pop("rdm_backend", rdm_backend)).lower()  # type: ignore[assignment]
    if "use_cuda" in mrci_kwargs:
        raise ValueError("use_cuda is removed; use hop_backend/integrals_backend explicitly")

    # Determine requested reference states.
    requested_states = _infer_states(mc, states)

    # For robust excited-state selection, include all lower states as guesses.
    solve_states = list(range(max(requested_states) + 1))

    if backend_s == "analytic":
        if method_s == "mrcisd":
            from asuka.mrci.grad_analytic import mrcisd_energy_and_grad_states_from_mc

            assign_method: Literal["hungarian", "greedy"] = "hungarian" if root_follow_s == "hungarian" else "greedy"

            mrci_states, roots, grads = mrcisd_energy_and_grad_states_from_mc(
                mc,
                states=solve_states,
                nroots=len(solve_states),
                root_follow=assign_method,
                plus_q=False,
                rdm_backend=rdm_backend,  # type: ignore[arg-type]
                atmlst=atmlst,
                max_cycle=int(mrci_kwargs.get("max_cycle", 80)),
                conv_tol=float(mrci_kwargs.get("conv_tol", 1e-10)),
                max_virt_e=int(mrci_kwargs.get("max_virt_e", 2)),
                verbose=mrci_kwargs.get("verbose"),
                hop_backend=mrci_kwargs.get("hop_backend"),
                correlate_inactive=int(mrci_kwargs.get("correlate_inactive", 0)),
            )

            # Reference gradients: best-effort per-state.
            if not hasattr(mc, "nuc_grad_method"):
                raise TypeError(
                    "Analytic backend requires a PySCF CASSCF/SA-CASSCF object with nuc_grad_method()."
                )
            grad_method = mc.nuc_grad_method()

            state_results: list[StateGrad] = []
            for req_state in requested_states:
                k = solve_states.index(int(req_state))
                root = int(roots[k])
                e_ref = float(mrci_states.e_ref[k])
                e_tot = float(mrci_states.mrci.e_mrci[root])
                e_corr = float(e_tot - e_ref)
                e_tot_used = float(e_tot)

                grad_tot = np.asarray(grads[k])

                grad_ref = None
                try:
                    grad_ref = np.asarray(grad_method.kernel(atmlst=atmlst, state=int(req_state)))
                except TypeError:
                    try:
                        grad_ref = np.asarray(grad_method.kernel(atmlst=atmlst))
                    except Exception:
                        grad_ref = None
                except Exception:
                    grad_ref = None

                grad_corr = None
                if grad_ref is not None:
                    grad_corr = grad_tot - np.asarray(grad_ref)

                diagnostics = None
                if bool(plus_q):
                    plus_q_model_use = str(mrci_kwargs.get("plus_q_model", "fixed"))
                    plus_q_min_ref_use = float(mrci_kwargs.get("plus_q_min_ref", 1e-8))
                    # Heuristic +Q gradients: scale the uncorrected correlation gradient by a factor
                    # consistent with the Davidson-type corrected energy model:
                    #
                    #   E(+Q) = E_ref + (E_MRCI - E_ref)/denom
                    #   ∇E(+Q) ≈ ∇E_ref + (1/denom) * (∇E_MRCI - ∇E_ref)
                    #
                    # i.e. treat denom (c2 or w_ref) as constant w.r.t. nuclear displacements.
                    #
                    # Columbus can write similarly scaled "CI+DVx" density matrices (heuristic);
                    # see `/home/zpengmei/columbus/Columbus/source/ciudg/ciudg_main.F90:6362`.
                    if grad_ref is None or grad_corr is None:
                        raise RuntimeError(
                            "Heuristic +Q analytic gradients require an available reference gradient (grad_ref)."
                        )
                    from asuka.mrci.mrcisd import mrcisd_plus_q  # noqa: PLC0415

                    e_q, q_diag = mrcisd_plus_q(
                        e_mrci=float(e_tot),
                        e_ref=float(e_ref),
                        ci_mrci=mrci_states.mrci.ci[root],
                        ci_ref0=mrci_states.mrci.ci_ref0[k],
                        ref_idx=mrci_states.mrci.ref_idx,
                        model=plus_q_model_use,
                        min_ref=plus_q_min_ref_use,
                    )
                    if e_q is None:
                        raise RuntimeError(
                            "Requested plus_q=True but +Q correction returned None. "
                            "Try increasing plus_q_min_ref or disable +Q."
                        )
                    denom = float(q_diag["c2"] if plus_q_model_use.strip().lower() == "fixed" else q_diag["w_ref"])
                    if not np.isfinite(denom) or denom <= 0.0:
                        raise RuntimeError("Invalid +Q denominator (c2/w_ref) encountered")
                    scale = 1.0 / denom

                    e_tot_used = float(e_q)
                    grad_tot = np.asarray(grad_ref) + float(scale) * np.asarray(grad_corr)
                    grad_corr = grad_tot - np.asarray(grad_ref)

                    diagnostics = dict(q_diag)
                    diagnostics["plus_q_scale"] = float(scale)
                    diagnostics["plus_q_model_is_fixed"] = 1.0 if plus_q_model_use.strip().lower() == "fixed" else 0.0

                state_results.append(
                    StateGrad(
                        state=int(req_state),
                        root=root,
                        e_ref=e_ref,
                        e_tot=e_tot,
                        e_corr=e_corr,
                        e_tot_used=e_tot_used,
                        grad_tot=grad_tot,
                        grad_ref=grad_ref,
                        grad_corr=grad_corr,
                        diagnostics=diagnostics,
                    )
                )

            return MRCIGradStatesResult(
                method="mrcisd",
                backend="analytic",
                fd_step_bohr=None,
                states=list(requested_states),
                state_results=state_results,
                overlap_ref_root=np.asarray(mrci_states.mrci.overlap_ref_root),
            )

        # ic_mrcisd analytic backend (validation: reconstruction).
        if method_s != "ic_mrcisd":  # pragma: no cover
            raise AssertionError("unreachable")

        if bool(plus_q):
            raise NotImplementedError("+Q correction is currently supported only for uncontracted mrcisd")

        from asuka.mrci.grad_analytic import ic_mrcisd_energy_and_grad_states_from_mc

        # No multi-root solver yet; just loop requested states.
        mrci_res_list, grads = ic_mrcisd_energy_and_grad_states_from_mc(
            mc,
            states=requested_states,
            contraction=mrci_kwargs.get("contraction", "fic"),
            backend=mrci_kwargs.get("backend", "semi_direct"),
            sc_backend=mrci_kwargs.get("sc_backend", "otf"),
            symmetry=bool(mrci_kwargs.get("symmetry", True)),
            allow_same_external=bool(mrci_kwargs.get("allow_same_external", True)),
            allow_same_internal=bool(mrci_kwargs.get("allow_same_internal", True)),
            norm_min_singles=float(mrci_kwargs.get("norm_min_singles", 0.0)),
            norm_min_doubles=float(mrci_kwargs.get("norm_min_doubles", 0.0)),
            s_tol=float(mrci_kwargs.get("s_tol", 1e-12)),
            solver=mrci_kwargs.get("solver", "davidson"),
            dense_nlab_max=int(mrci_kwargs.get("dense_nlab_max", 250)),
            n_virt=mrci_kwargs.get("n_virt"),
            max_virt_e=int(mrci_kwargs.get("max_virt_e", 2)),
            hop_backend=mrci_kwargs.get("hop_backend"),
            contract_nthreads=int(mrci_kwargs.get("contract_nthreads", 1)),
            contract_blas_nthreads=mrci_kwargs.get("contract_blas_nthreads"),
            precompute_epq=bool(mrci_kwargs.get("precompute_epq", True)),
            plus_q=False,
            rdm_backend=rdm_backend,  # type: ignore[arg-type]
            atmlst=atmlst,
            max_cycle=int(mrci_kwargs.get("max_cycle", 80)),
            conv_tol=float(mrci_kwargs.get("conv_tol", 1e-10)),
            verbose=mrci_kwargs.get("verbose"),
        )

        if not hasattr(mc, "nuc_grad_method"):
            raise TypeError(
                "Analytic backend requires a PySCF CASSCF/SA-CASSCF object with nuc_grad_method()."
            )
        grad_method = mc.nuc_grad_method()

        state_results = []
        for mrci_res, grad_tot, req_state in zip(mrci_res_list, grads, requested_states):
            e_ref = float(mrci_res.e_ref)
            e_tot = float(mrci_res.e_tot)
            e_corr = float(mrci_res.e_corr)
            e_tot_used = float(e_tot)

            grad_ref = None
            try:
                grad_ref = np.asarray(grad_method.kernel(atmlst=atmlst, state=int(req_state)))
            except TypeError:
                try:
                    grad_ref = np.asarray(grad_method.kernel(atmlst=atmlst))
                except Exception:
                    grad_ref = None
            except Exception:
                grad_ref = None

            grad_corr = None
            if grad_ref is not None:
                grad_corr = np.asarray(grad_tot) - np.asarray(grad_ref)

            state_results.append(
                StateGrad(
                    state=int(req_state),
                    root=0,
                    e_ref=e_ref,
                    e_tot=e_tot,
                    e_corr=e_corr,
                    e_tot_used=e_tot_used,
                    grad_tot=np.asarray(grad_tot),
                    grad_ref=grad_ref,
                    grad_corr=grad_corr,
                    diagnostics=None,
                )
            )

        return MRCIGradStatesResult(
            method="ic_mrcisd",
            backend="analytic",
            fd_step_bohr=None,
            states=list(requested_states),
            state_results=state_results,
            overlap_ref_root=None,
        )

    # FD backend
    if bool(plus_q) and method_s != "mrcisd":
        raise NotImplementedError("+Q correction is currently supported only for uncontracted mrcisd")

    fd_step_bohr = float(fd_step_bohr)
    if fd_step_bohr <= 0.0:
        raise ValueError("fd_step_bohr must be positive")

    if requested_states == [0] and solve_states == [0]:
        scan = mc.as_scanner()
        mol = scan.mol
        natm = int(mol.natm)

        # Tighten CASSCF/SCF convergence for FD scans to reduce numerical noise in
        # E_ref (and thus grad_corr), especially for small fd_step_bohr.
        scan_conv_tol0 = getattr(scan, "conv_tol", None)
        scf_conv_tol0 = getattr(getattr(scan, "_scf", None), "conv_tol", None)
        if scan_conv_tol0 is not None:
            try:
                scan.conv_tol = min(float(scan_conv_tol0), 1e-10)
            except Exception:
                pass
        if scf_conv_tol0 is not None:
            try:
                scan._scf.conv_tol = min(float(scf_conv_tol0), 1e-11)
            except Exception:
                pass

        unit = str(getattr(mol, "unit", "Bohr")).strip().lower()
        if unit.startswith("a"):
            bohr_to_scan = float(_BOHR_TO_ANGSTROM)
        else:
            bohr_to_scan = 1.0

        coords0 = np.asarray(mol.atom_coords(), dtype=np.float64, order="C")
        if coords0.shape != (natm, 3):
            raise ValueError("unexpected atom_coords() shape")

        atmlst_list = list(range(natm)) if atmlst is None else list(atmlst)

        def _energy_at(coords_bohr: np.ndarray) -> MRCIFromMCResult:
            scan(np.asarray(coords_bohr, dtype=np.float64) * bohr_to_scan)
            return mrci_from_mc(
                scan,
                method=method_s,  # type: ignore[arg-type]
                plus_q=bool(plus_q),
                **mrci_kwargs,
            )

        ref = _energy_at(coords0)
        e_tot_used0 = float(ref.e_tot_plus_q) if (bool(plus_q) and ref.e_tot_plus_q is not None) else float(ref.e_tot)
        if bool(plus_q) and ref.e_tot_plus_q is None:
            raise RuntimeError("Requested plus_q=True but +Q correction returned None at reference geometry")

        grad_ref = np.zeros((len(atmlst_list), 3), dtype=np.float64)
        grad_tot = np.zeros((len(atmlst_list), 3), dtype=np.float64)
        grad_corr = np.zeros((len(atmlst_list), 3), dtype=np.float64)

        for ib, ia in enumerate(atmlst_list):
            for xyz in range(3):
                coords_p = np.array(coords0, copy=True)
                coords_m = np.array(coords0, copy=True)
                coords_p[ia, xyz] += fd_step_bohr
                coords_m[ia, xyz] -= fd_step_bohr

                if bool(fd_reset_to_ref):
                    _energy_at(coords0)
                ep = _energy_at(coords_p)

                if bool(fd_reset_to_ref):
                    _energy_at(coords0)
                em = _energy_at(coords_m)

                ep_tot_used = (
                    float(ep.e_tot_plus_q) if (bool(plus_q) and ep.e_tot_plus_q is not None) else float(ep.e_tot)
                )
                em_tot_used = (
                    float(em.e_tot_plus_q) if (bool(plus_q) and em.e_tot_plus_q is not None) else float(em.e_tot)
                )
                if bool(plus_q) and (ep.e_tot_plus_q is None or em.e_tot_plus_q is None):
                    raise RuntimeError(
                        "Requested plus_q=True but +Q correction returned None at a displaced geometry. "
                        "Try increasing plus_q_min_ref or disable +Q for this geometry."
                    )

                grad_ref[ib, xyz] = (float(ep.e_ref) - float(em.e_ref)) / (2.0 * fd_step_bohr)
                grad_tot[ib, xyz] = (ep_tot_used - em_tot_used) / (2.0 * fd_step_bohr)
                grad_corr[ib, xyz] = ((ep_tot_used - float(ep.e_ref)) - (em_tot_used - float(em.e_ref))) / (
                    2.0 * fd_step_bohr
                )

        _energy_at(coords0)

        # Replace FD reference gradient with the (more accurate) analytic CASSCF
        # reference gradient. This reduces numerical noise in grad_corr while
        # keeping grad_tot as a true finite-difference quantity.
        try:
            grad_method = mc.nuc_grad_method()
            try:
                grad_ref_ana = np.asarray(grad_method.kernel(atmlst=atmlst, state=0))
            except TypeError:
                grad_ref_ana = np.asarray(grad_method.kernel(atmlst=atmlst))
            grad_ref = grad_ref_ana
            grad_corr = np.asarray(grad_tot) - np.asarray(grad_ref)
        except Exception:
            # Fall back to FD-based grad_ref/grad_corr if analytic reference gradients
            # are not available (e.g. SA-CASSCF with a CSF-based solver).
            pass

        # Restore original convergence thresholds.
        if scan_conv_tol0 is not None:
            try:
                scan.conv_tol = scan_conv_tol0
            except Exception:
                pass
        if scf_conv_tol0 is not None:
            try:
                scan._scf.conv_tol = scf_conv_tol0
            except Exception:
                pass

        diagnostics = None
        if bool(plus_q) and ref.plus_q_diag is not None:
            diagnostics = dict(ref.plus_q_diag)

        st = StateGrad(
            state=0,
            root=0,
            e_ref=float(ref.e_ref),
            e_tot=float(ref.e_tot),
            e_corr=float(ref.e_corr),
            e_tot_used=float(e_tot_used0),
            grad_tot=np.asarray(grad_tot),
            grad_ref=np.asarray(grad_ref),
            grad_corr=np.asarray(grad_corr),
            diagnostics=diagnostics,
        )
        return MRCIGradStatesResult(
            method=str(ref.method),
            backend="fd",
            fd_step_bohr=float(fd_step_bohr),
            states=[0],
            state_results=[st],
            overlap_ref_root=None,
        )

    if method_s not in ("mrcisd", "ic_mrcisd"):
        raise ValueError("method must be 'mrcisd' or 'ic_mrcisd'")

    scan = mc.as_scanner()
    mol = scan.mol
    natm = int(mol.natm)

    # Tighten convergence for FD scans (see single-state path above).
    scan_conv_tol0 = getattr(scan, "conv_tol", None)
    scf_conv_tol0 = getattr(getattr(scan, "_scf", None), "conv_tol", None)
    if scan_conv_tol0 is not None:
        try:
            scan.conv_tol = min(float(scan_conv_tol0), 1e-10)
        except Exception:
            pass
    if scf_conv_tol0 is not None:
        try:
            scan._scf.conv_tol = min(float(scf_conv_tol0), 1e-11)
        except Exception:
            pass

    unit = str(getattr(mol, "unit", "Bohr")).strip().lower()
    if unit.startswith("a"):
        bohr_to_scan = float(_BOHR_TO_ANGSTROM)
    else:
        bohr_to_scan = 1.0

    coords0 = np.asarray(mol.atom_coords(), dtype=np.float64, order="C")
    if coords0.shape != (natm, 3):
        raise ValueError("unexpected atom_coords() shape")

    atmlst_list = list(range(natm)) if atmlst is None else list(atmlst)

    if method_s == "mrcisd":
        plus_q_model = str(mrci_kwargs.pop("plus_q_model", "fixed"))
        plus_q_min_ref = float(mrci_kwargs.pop("plus_q_min_ref", 1e-8))
        assign_method: Literal["hungarian", "greedy"] = "hungarian" if root_follow_s == "hungarian" else "greedy"

        def _energies_at(coords_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            scan(np.asarray(coords_bohr, dtype=np.float64) * bohr_to_scan)
            mrci_states = mrci_states_from_mc(
                scan,
                method="mrcisd",
                states=solve_states,
                nroots=len(solve_states),
                **mrci_kwargs,
            )
            roots = assign_roots_by_overlap(mrci_states.mrci.overlap_ref_root, method=assign_method)
            e_ref = np.asarray(mrci_states.e_ref, dtype=np.float64)
            e_tot = np.asarray([mrci_states.mrci.e_mrci[int(r)] for r in roots], dtype=np.float64)

            if not bool(plus_q):
                e_tot_used = e_tot.copy()
            else:
                from asuka.mrci.mrcisd import mrcisd_plus_q  # noqa: PLC0415

                e_tot_used = np.zeros_like(e_tot)
                for k in range(len(solve_states)):
                    root = int(roots[k])
                    e_q, _diag = mrcisd_plus_q(
                        e_mrci=float(e_tot[k]),
                        e_ref=float(e_ref[k]),
                        ci_mrci=mrci_states.mrci.ci[root],
                        ci_ref0=mrci_states.mrci.ci_ref0[k],
                        ref_idx=mrci_states.mrci.ref_idx,
                        model=plus_q_model,
                        min_ref=plus_q_min_ref,
                    )
                    if e_q is None:
                        raise RuntimeError(
                            "Requested plus_q=True but +Q correction returned None at a geometry. "
                            "Try increasing plus_q_min_ref or disable +Q."
                        )
                    e_tot_used[k] = float(e_q)

            return roots, e_ref, e_tot, e_tot_used

    else:
        # Contracted (ic) MRCI: no multi-root solver yet. Compute energies for each
        # requested reference state via repeated single-root runs.
        def _energies_at(coords_bohr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            scan(np.asarray(coords_bohr, dtype=np.float64) * bohr_to_scan)

            roots = np.zeros(len(solve_states), dtype=np.int64)
            e_ref = np.zeros(len(solve_states), dtype=np.float64)
            e_tot = np.zeros(len(solve_states), dtype=np.float64)
            e_tot_used = np.zeros(len(solve_states), dtype=np.float64)

            for k, st in enumerate(solve_states):
                res = mrci_from_mc(
                    scan,
                    method="ic_mrcisd",
                    state=int(st),
                    plus_q=False,
                    **mrci_kwargs,
                )
                e_ref[k] = float(res.e_ref)
                e_tot[k] = float(res.e_tot)
                e_tot_used[k] = float(res.e_tot)

            return roots, e_ref, e_tot, e_tot_used

    roots0, e_ref0, e_tot0, e_tot_used0 = _energies_at(coords0)
    e_corr0 = e_tot0 - e_ref0

    grad_ref = np.zeros((len(requested_states), len(atmlst_list), 3), dtype=np.float64)
    grad_tot = np.zeros((len(requested_states), len(atmlst_list), 3), dtype=np.float64)
    grad_corr = np.zeros((len(requested_states), len(atmlst_list), 3), dtype=np.float64)

    for ib, ia in enumerate(atmlst_list):
        for xyz in range(3):
            coords_p = np.array(coords0, copy=True)
            coords_m = np.array(coords0, copy=True)
            coords_p[ia, xyz] += fd_step_bohr
            coords_m[ia, xyz] -= fd_step_bohr

            if bool(fd_reset_to_ref):
                _energies_at(coords0)
            _roots_p, e_ref_p, _e_tot_p, e_tot_used_p = _energies_at(coords_p)

            if bool(fd_reset_to_ref):
                _energies_at(coords0)
            _roots_m, e_ref_m, _e_tot_m, e_tot_used_m = _energies_at(coords_m)

            de_ref = (e_ref_p - e_ref_m) / (2.0 * fd_step_bohr)
            de_tot = (e_tot_used_p - e_tot_used_m) / (2.0 * fd_step_bohr)
            de_corr = ((e_tot_used_p - e_ref_p) - (e_tot_used_m - e_ref_m)) / (2.0 * fd_step_bohr)

            for sidx, st in enumerate(requested_states):
                k = solve_states.index(int(st))
                grad_ref[sidx, ib, xyz] = float(de_ref[k])
                grad_tot[sidx, ib, xyz] = float(de_tot[k])
                grad_corr[sidx, ib, xyz] = float(de_corr[k])

    # Reset scanner back to reference coordinates.
    _energies_at(coords0)

    # Replace FD reference gradients with analytic CASSCF reference gradients.
    try:
        grad_method = mc.nuc_grad_method()
        grad_ref_ana = np.zeros_like(grad_ref)
        for sidx, st in enumerate(requested_states):
            try:
                g = np.asarray(grad_method.kernel(atmlst=atmlst, state=int(st)))
            except TypeError:
                g = np.asarray(grad_method.kernel(atmlst=atmlst))
            grad_ref_ana[sidx] = g
        grad_ref = grad_ref_ana
        grad_corr = np.asarray(grad_tot) - np.asarray(grad_ref)
    except Exception:
        # Keep FD-based grad_ref/grad_corr if analytic reference gradients are not available.
        pass

    # Restore original convergence thresholds.
    if scan_conv_tol0 is not None:
        try:
            scan.conv_tol = scan_conv_tol0
        except Exception:
            pass
    if scf_conv_tol0 is not None:
        try:
            scan._scf.conv_tol = scf_conv_tol0
        except Exception:
            pass

    state_results: list[StateGrad] = []
    for sidx, st in enumerate(requested_states):
        k = solve_states.index(int(st))
        state_results.append(
            StateGrad(
                state=int(st),
                root=int(roots0[k]),
                e_ref=float(e_ref0[k]),
                e_tot=float(e_tot0[k]),
                e_corr=float(e_corr0[k]),
                e_tot_used=float(e_tot_used0[k]),
                grad_tot=np.asarray(grad_tot[sidx]),
                grad_ref=np.asarray(grad_ref[sidx]),
                grad_corr=np.asarray(grad_corr[sidx]),
                diagnostics=None,
            )
        )

    return MRCIGradStatesResult(
        method=str(method_s),
        backend="fd",
        fd_step_bohr=float(fd_step_bohr),
        states=list(requested_states),
        state_results=state_results,
        overlap_ref_root=None,
    )


def mrci_grad_from_mc(
    mc: Any,
    *,
    method: Method = "mrcisd",
    backend: Backend = "fd",
    state: int = 0,
    fd_step_bohr: float = 1e-3,
    fd_reset_to_ref: bool = True,
    # --- arguments forwarded to mrci_from_mc ---
    plus_q: bool = False,
    **mrci_kwargs: Any,
) -> MRCIGradResult:
    """Compute MRCI nuclear gradients from a PySCF CASCI/CASSCF object.

    Parameters
    ----------
    mc : Any
        PySCF CASCI/CASSCF-like object.
    method : {"mrcisd", "ic_mrcisd"}, optional
        Method to use ("mrcisd" or "ic_mrcisd"). Default is "mrcisd".
    backend : {"fd", "analytic"}, optional
        Gradient backend ("fd" or "analytic"). Default is "fd".
    state : int, optional
        State index to compute. Default is 0.
    fd_step_bohr : float, optional
        Finite-difference step size in Bohr. Default is 1e-3.
    fd_reset_to_ref : bool, optional
        If True, reset the PySCF scanner back to the reference geometry before
        each displacement. This reduces path dependence from iterative solvers.
        Default is True.
    plus_q : bool, optional
        If True and method="mrcisd":
          - backend="fd": finite-difference gradient of the +Q-corrected energy
            (includes denom(R) variation).
          - backend="analytic": heuristic gradient using the analytic uncorrected
            gradients plus a constant-denominator scaling.
        Default is False.
    mrci_kwargs : Any
        Additional keyword arguments forwarded to :func:`asuka.mrci.driver.mrci_from_mc`.

    Returns
    -------
    MRCIGradResult
        Result object containing the computed gradient and energies.
    """

    # Multi-state wrapper for single-state compatibility.
    res_states = mrci_grad_states_from_mc(
        mc,
        method=method,
        backend=backend,
        states=[int(state)],
        root_follow="hungarian",
        fd_step_bohr=fd_step_bohr,
        fd_reset_to_ref=fd_reset_to_ref,
        plus_q=plus_q,
        **mrci_kwargs,
    )
    out = res_states.state_results[0]
    if out.grad_ref is None or out.grad_corr is None:
        raise RuntimeError("internal error: missing grad_ref/grad_corr for single-state result")
    return MRCIGradResult(
        method=str(res_states.method),
        backend=str(res_states.backend),
        fd_step_bohr=res_states.fd_step_bohr,
        e_ref=float(out.e_ref),
        e_tot=float(out.e_tot),
        e_corr=float(out.e_corr),
        e_tot_used=float(out.e_tot_used),
        grad_ref=np.asarray(out.grad_ref),
        grad_tot=np.asarray(out.grad_tot),
        grad_corr=np.asarray(out.grad_corr),
        diagnostics=out.diagnostics,
    )
