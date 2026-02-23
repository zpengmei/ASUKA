from __future__ import annotations

"""ASUKA-native MRCISD nuclear gradients.

The public entry points here work with ASUKA frontend objects:
- `scf_out`: output of :func:`asuka.frontend.scf.run_hf_df`
- `ref`: output of :func:`asuka.mcscf.run_casscf` (or CASCI)

Phase-1 scope
-------------
- Uncontracted MRCISD only (``method="mrcisd"``).
- Analytic gradients use ASUKA's DF derivative contractions and a CP-CASSCF
  Z-vector solve in the *reference* SA-CASSCF parameter space.
- Finite-difference gradients are provided as a slow validation/fallback backend.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.mrci.common import assign_roots_by_overlap
from asuka.mrci.driver_asuka import mrci_states_from_ref
from asuka.mrci.result import MRCIStatesResult


Backend = Literal["analytic", "fd"]
Method = Literal["mrcisd"]


@dataclass(frozen=True)
class MRCIGradResult:
    """Single-state MRCISD nuclear gradient result (Eh/Bohr)."""

    method: str
    backend: str
    fd_step_bohr: float | None

    state: int
    root: int

    e_ref: float
    e_tot: float
    e_corr: float
    e_tot_used: float

    grad_tot: np.ndarray
    grad_ref: np.ndarray | None = None
    grad_corr: np.ndarray | None = None

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
    grad_ref: np.ndarray | None = None
    grad_corr: np.ndarray | None = None
    diagnostics: dict[str, float] | None = None


@dataclass(frozen=True)
class MRCIGradStatesResult:
    method: str
    backend: str
    fd_step_bohr: float | None
    states: list[int]
    roots: np.ndarray
    state_results: list[StateGrad]
    mrci: MRCIStatesResult


def _resolve_scf_out_from_ref(ref: Any, *, scf_out: Any | None) -> Any:
    scf_out_use = scf_out
    if scf_out_use is None:
        scf_out_use = getattr(ref, "scf_out", None)
    if scf_out_use is None and hasattr(ref, "casci"):
        scf_out_use = getattr(getattr(ref, "casci"), "scf_out", None)
    if scf_out_use is None:
        raise ValueError("scf_out is required (missing on ref; pass scf_out explicitly)")
    return scf_out_use


def _infer_states_from_ref(ref: Any, states: Sequence[int] | None) -> list[int]:
    if states is None:
        ci = getattr(ref, "ci")
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

    ci = getattr(ref, "ci")
    if isinstance(ci, (list, tuple)):
        n = int(len(ci))
        if any(s >= n for s in out):
            raise ValueError("state index out of range for ref.ci")
    else:
        if any(s != 0 for s in out):
            raise ValueError("state != 0 but ref.ci is not a list/tuple")
    return out


def mrci_grad_states_from_ref(
    ref: Any,
    *,
    scf_out: Any | None = None,
    method: Method = "mrcisd",
    backend: Backend = "analytic",
    states: Sequence[int] | None = None,
    root_follow: Literal["hungarian", "greedy"] = "hungarian",
    # FD options
    fd_step_bohr: float = 1e-3,
    fd_which: Sequence[tuple[int, int]] | None = None,
    # Analytic options
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    # Forwarded to mrci_states_from_ref
    **mrci_kwargs: Any,
) -> MRCIGradStatesResult:
    """Compute MRCISD nuclear gradients for multiple reference states."""

    method_s = str(method).strip().lower()
    if method_s != "mrcisd":
        raise NotImplementedError("ASUKA-native gradients currently support only method='mrcisd'")

    backend_s = str(backend).strip().lower()
    if backend_s not in ("analytic", "fd"):
        raise ValueError("backend must be 'analytic' or 'fd'")

    root_follow_s = str(root_follow).strip().lower()
    if root_follow_s not in ("hungarian", "greedy"):
        raise ValueError("root_follow must be 'hungarian' or 'greedy'")

    scf_out_use = _resolve_scf_out_from_ref(ref, scf_out=scf_out)
    states_list = _infer_states_from_ref(ref, states)

    # Keep the limitation explicit: the ASUKA-native driver currently enforces nroots==len(states).
    if "nroots" in mrci_kwargs and mrci_kwargs["nroots"] is not None:
        nroots_user = int(mrci_kwargs["nroots"])
        if nroots_user != len(states_list):
            raise ValueError(
                "mrci_grad_states_from_ref requires nroots == len(states) (ASUKA-native driver limitation); "
                f"got nroots={nroots_user} and len(states)={len(states_list)}"
            )

    mrci_states = mrci_states_from_ref(
        ref,
        scf_out=scf_out_use,
        method="mrcisd",
        states=states_list,
        **mrci_kwargs,
    )

    overlap = np.asarray(mrci_states.mrci.overlap_ref_root, dtype=np.float64)
    roots = assign_roots_by_overlap(overlap, method=root_follow_s)

    if backend_s == "analytic":
        from asuka.mrci.grad_analytic import mrci_grad_states_from_ref_analytic  # noqa: PLC0415

        grads = mrci_grad_states_from_ref_analytic(
            scf_out_use,
            ref,
            mrci_states=mrci_states,
            roots=roots,
            states=states_list,
            max_virt_e=int(mrci_kwargs.get("max_virt_e", 2)),
            rdm_backend=rdm_backend,
            df_backend=df_backend,
            df_config=df_config,
            df_threads=int(df_threads),
            z_tol=float(z_tol),
            z_maxiter=int(z_maxiter),
        )
        fd_step = None
    else:
        from asuka.mrci.grad_fd import mrci_grad_states_from_ref_fd  # noqa: PLC0415

        grads = mrci_grad_states_from_ref_fd(
            scf_out_use,
            ref,
            mrci_states=mrci_states,
            roots=roots,
            states=states_list,
            fd_step_bohr=float(fd_step_bohr),
            which=fd_which,
            max_virt_e=int(mrci_kwargs.get("max_virt_e", 2)),
        )
        fd_step = float(fd_step_bohr)

    state_results: list[StateGrad] = []
    e_mrci = np.asarray(mrci_states.mrci.e_mrci, dtype=np.float64).ravel()
    e_ref_arr = np.asarray(mrci_states.e_ref, dtype=np.float64).ravel()
    for k, st in enumerate(states_list):
        root = int(roots[k])
        e_tot = float(e_mrci[root])
        e_ref = float(e_ref_arr[k])
        e_corr = float(e_tot - e_ref)
        state_results.append(
            StateGrad(
                state=int(st),
                root=int(root),
                e_ref=e_ref,
                e_tot=e_tot,
                e_corr=e_corr,
                e_tot_used=e_tot,
                grad_tot=np.asarray(grads[k], dtype=np.float64),
            )
        )

    return MRCIGradStatesResult(
        method=str(method_s),
        backend=str(backend_s),
        fd_step_bohr=fd_step,
        states=list(states_list),
        roots=np.asarray(roots, dtype=np.int64),
        state_results=state_results,
        mrci=mrci_states,
    )


def mrci_grad_from_ref(
    ref: Any,
    *,
    scf_out: Any | None = None,
    method: Method = "mrcisd",
    backend: Backend = "analytic",
    state: int = 0,
    root_follow: Literal["hungarian", "greedy"] = "hungarian",
    fd_step_bohr: float = 1e-3,
    **kwargs: Any,
) -> MRCIGradResult:
    """Compute an MRCISD nuclear gradient for a single reference state."""

    state_i = int(state)
    if state_i < 0:
        raise ValueError("state must be non-negative")

    # Ensure excited-state tracking works with the current MRCI driver restriction
    # `nroots == len(states)` by solving the lowest (state+1) roots.
    solve_states = list(range(state_i + 1))
    res = mrci_grad_states_from_ref(
        ref,
        scf_out=scf_out,
        method=method,
        backend=backend,
        states=solve_states,
        root_follow=root_follow,
        fd_step_bohr=float(fd_step_bohr),
        **kwargs,
    )
    idx = res.states.index(state_i) if state_i in res.states else None
    if idx is None:
        raise RuntimeError("internal error: requested state not in computed states")
    s = res.state_results[idx]

    return MRCIGradResult(
        method=res.method,
        backend=res.backend,
        fd_step_bohr=res.fd_step_bohr,
        state=int(s.state),
        root=int(s.root),
        e_ref=float(s.e_ref),
        e_tot=float(s.e_tot),
        e_corr=float(s.e_corr),
        e_tot_used=float(s.e_tot_used),
        grad_tot=np.asarray(s.grad_tot, dtype=np.float64),
        grad_ref=s.grad_ref,
        grad_corr=s.grad_corr,
        diagnostics=s.diagnostics,
    )


__all__ = [
    "MRCIGradResult",
    "MRCIGradStatesResult",
    "StateGrad",
    "mrci_grad_from_ref",
    "mrci_grad_states_from_ref",
]
