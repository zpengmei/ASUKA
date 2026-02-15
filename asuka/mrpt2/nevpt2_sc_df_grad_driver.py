from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.mrpt2.nevpt2_sc_df_driver import NEVPT2SCDFResult, nevpt2_sc_df_from_mc


@dataclass(frozen=True)
class NEVPT2SCDFGradResult:
    """Result object for SC-NEVPT2(DF) nuclear gradients.

    Notes
    -----
    - This is a **driver-level** result container. It is intentionally method-agnostic:
      the `backend` field indicates how the gradient was computed.
    - Currently, only `backend="fd"` (finite difference) is implemented. The analytic
      backend will follow the Lagrangian/Z-vector design in `guga_cuda/NEVPT2_GRADIENT_DESIGN.md`.
    """

    backend: str
    fd_step_bohr: float | None

    e_corr: float
    grad_corr: np.ndarray

    e_cas: float | None = None
    grad_cas: np.ndarray | None = None

    e_tot: float | None = None
    grad_tot: np.ndarray | None = None

    breakdown: dict[str, float] | None = None


def nevpt2_sc_df_grad_from_mc(
    mc,
    *,
    auxbasis: Any = "weigend+etb",
    twos: int = 0,
    semicanonicalize: bool = True,
    pt2_backend: str = "cpu",
    cuda_device: int | None = None,
    max_memory_mb: float = 4000.0,
    guga_tol: float = 1e-14,
    guga_max_cycle: int = 400,
    guga_max_space: int = 30,
    guga_pspace_size: int = 0,
    verbose: int = 0,
    grad_backend: str = "fd",
    fd_step_bohr: float = 1e-3,
    fd_reset_to_ref: bool = True,
) -> NEVPT2SCDFGradResult:
    """Compute SC-NEVPT2(DF) nuclear gradients from a PySCF CASCI/CASSCF object.

    Parameters
    ----------
    mc:
        PySCF CASCI/CASSCF-like object.
    grad_backend:
        Currently supports only `"fd"` (finite difference).
    fd_step_bohr:
        Finite-difference step in Bohr.
    fd_reset_to_ref:
        If True, reset the scanner back to the reference geometry before each +h and -h
        displacement to reduce "path dependence" from iterative solvers.
    """

    backend = str(grad_backend).lower()
    if backend == "fd":
        return _nevpt2_sc_df_grad_from_mc_fd(
            mc,
            auxbasis=auxbasis,
            twos=twos,
            semicanonicalize=semicanonicalize,
            pt2_backend=pt2_backend,
            cuda_device=cuda_device,
            max_memory_mb=max_memory_mb,
            guga_tol=guga_tol,
            guga_max_cycle=guga_max_cycle,
            guga_max_space=guga_max_space,
            guga_pspace_size=guga_pspace_size,
            verbose=verbose,
            fd_step_bohr=fd_step_bohr,
            fd_reset_to_ref=fd_reset_to_ref,
        )
    if backend in ("analytic", "zvector", "z-vector"):
        raise NotImplementedError(
            "Analytic SC-NEVPT2 gradients are not implemented yet; "
            "use grad_backend='fd' for finite-difference gradients."
        )
    raise ValueError("grad_backend must be one of: 'fd', 'analytic'")


def _nevpt2_sc_df_grad_from_mc_fd(
    mc,
    *,
    auxbasis: Any,
    twos: int,
    semicanonicalize: bool,
    pt2_backend: str,
    cuda_device: int | None,
    max_memory_mb: float,
    guga_tol: float,
    guga_max_cycle: int,
    guga_max_space: int,
    guga_pspace_size: int,
    verbose: int,
    fd_step_bohr: float,
    fd_reset_to_ref: bool,
) -> NEVPT2SCDFGradResult:
    fd_step_bohr = float(fd_step_bohr)
    if fd_step_bohr <= 0.0:
        raise ValueError("fd_step_bohr must be positive")

    scan = mc.as_scanner()
    scan.verbose = int(verbose)

    mol = scan.mol
    natm = int(mol.natm)
    coords0 = np.asarray(mol.atom_coords(), dtype=np.float64, order="C")
    if coords0.shape != (natm, 3):
        raise ValueError("unexpected atom_coords() shape")

    def _energy_at(coords_bohr: np.ndarray) -> NEVPT2SCDFResult:
        scan(coords_bohr)
        return nevpt2_sc_df_from_mc(
            scan,
            auxbasis=auxbasis,
            twos=twos,
            semicanonicalize=semicanonicalize,
            pt2_backend=pt2_backend,
            cuda_device=cuda_device,
            max_memory_mb=max_memory_mb,
            guga_tol=guga_tol,
            guga_max_cycle=guga_max_cycle,
            guga_max_space=guga_max_space,
            guga_pspace_size=guga_pspace_size,
            verbose=verbose,
        )

    ref = _energy_at(coords0)

    grad_corr = np.zeros((natm, 3), dtype=np.float64)
    grad_cas = np.zeros((natm, 3), dtype=np.float64) if ref.e_cas is not None else None
    grad_tot = np.zeros((natm, 3), dtype=np.float64) if ref.e_tot is not None else None

    for ia in range(natm):
        for xyz in range(3):
            coords_p = np.array(coords0, copy=True)
            coords_m = np.array(coords0, copy=True)
            coords_p[ia, xyz] += fd_step_bohr
            coords_m[ia, xyz] -= fd_step_bohr

            if bool(fd_reset_to_ref):
                _energy_at(coords0)
            e_p = _energy_at(coords_p)

            if bool(fd_reset_to_ref):
                _energy_at(coords0)
            e_m = _energy_at(coords_m)

            grad_corr[ia, xyz] = (e_p.e_corr - e_m.e_corr) / (2.0 * fd_step_bohr)

            if grad_cas is not None:
                if e_p.e_cas is None or e_m.e_cas is None:
                    raise RuntimeError("reference provides e_cas but displaced geometry does not")
                grad_cas[ia, xyz] = (float(e_p.e_cas) - float(e_m.e_cas)) / (2.0 * fd_step_bohr)

            if grad_tot is not None:
                if e_p.e_tot is None or e_m.e_tot is None:
                    raise RuntimeError("reference provides e_tot but displaced geometry does not")
                grad_tot[ia, xyz] = (float(e_p.e_tot) - float(e_m.e_tot)) / (2.0 * fd_step_bohr)

    _energy_at(coords0)

    return NEVPT2SCDFGradResult(
        backend="fd",
        fd_step_bohr=fd_step_bohr,
        e_corr=float(ref.e_corr),
        grad_corr=grad_corr,
        e_cas=None if ref.e_cas is None else float(ref.e_cas),
        grad_cas=grad_cas,
        e_tot=None if ref.e_tot is None else float(ref.e_tot),
        grad_tot=grad_tot,
        breakdown=dict(ref.breakdown),
    )

