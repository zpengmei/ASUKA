from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.frontend.molecule import Molecule
from asuka.mrpt2.nevpt2_sc_df_driver import NEVPT2SCDFResult, nevpt2_sc_df_from_ref


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


def nevpt2_sc_df_grad_from_ref(
    ref0,
    *,
    scf_out: Any | None = None,
    state: int = 0,
    auxbasis: Any | None = None,
    twos: int | None = None,
    semicanonicalize: bool = True,
    pt2_backend: str = "cpu",
    cuda_device: int | None = None,
    max_memory_mb: float = 4000.0,
    verbose: int = 0,
    grad_backend: str = "fd",
    fd_step_bohr: float = 1e-3,
    fd_reset_to_ref: bool = True,
    fd_which: list[tuple[int, int]] | None = None,
) -> NEVPT2SCDFGradResult:
    """Compute SC-NEVPT2(DF) nuclear gradients from an ASUKA CAS reference.

    Parameters
    ----------
    ref0:
        CASSCF/CASCI result object providing (mo_coeff, ci, ncore, ncas, nelecas, mol).
    grad_backend:
        Currently supports only `"fd"` (finite difference).
    fd_step_bohr:
        Finite-difference step in Bohr.
    fd_reset_to_ref:
        If True, use the same SCF/CASSCF guesses at every displacement to reduce
        path dependence from iterative solvers.
    """

    backend = str(grad_backend).lower()
    if backend == "fd":
        return _nevpt2_sc_df_grad_from_ref_fd(
            ref0,
            scf_out=scf_out,
            state=state,
            auxbasis=auxbasis,
            twos=twos,
            semicanonicalize=semicanonicalize,
            pt2_backend=pt2_backend,
            cuda_device=cuda_device,
            max_memory_mb=max_memory_mb,
            verbose=verbose,
            fd_step_bohr=fd_step_bohr,
            fd_reset_to_ref=fd_reset_to_ref,
            fd_which=fd_which,
        )
    if backend in ("analytic", "zvector", "z-vector"):
        raise NotImplementedError(
            "Analytic SC-NEVPT2 gradients are not implemented yet; "
            "use grad_backend='fd' for finite-difference gradients."
        )
    raise ValueError("grad_backend must be one of: 'fd', 'analytic'")


def _infer_backend_from_df_B(df_B: Any) -> str:
    if df_B is None:
        return "cpu"
    if hasattr(df_B, "__cuda_array_interface__"):
        return "cuda"
    return "cpu"


def _nevpt2_sc_df_grad_from_ref_fd(
    ref0,
    *,
    scf_out: Any | None,
    state: int,
    auxbasis: Any | None,
    twos: int | None,
    semicanonicalize: bool,
    pt2_backend: str,
    cuda_device: int | None,
    max_memory_mb: float,
    verbose: int,
    fd_step_bohr: float,
    fd_reset_to_ref: bool,
    fd_which: list[tuple[int, int]] | None,
) -> NEVPT2SCDFGradResult:
    fd_step_bohr = float(fd_step_bohr)
    if fd_step_bohr <= 0.0:
        raise ValueError("fd_step_bohr must be positive")

    if scf_out is None:
        scf_out = getattr(ref0, "scf_out", None)
    if scf_out is None:
        casci = getattr(ref0, "casci", None)
        if casci is not None:
            scf_out = getattr(casci, "scf_out", None)
    if scf_out is None:
        raise ValueError("scf_out must be provided (or available as ref0.scf_out/ref0.casci.scf_out)")

    mol0 = getattr(ref0, "mol", None)
    if mol0 is None:
        raise ValueError("ref0.mol must be available")
    natm = int(getattr(mol0, "natm"))
    coords0 = np.asarray(getattr(mol0, "coords_bohr"), dtype=np.float64, order="C")
    if coords0.shape != (natm, 3):
        raise ValueError("unexpected atom_coords() shape")

    if twos is None:
        twos = getattr(mol0, "spin", None)
    if twos is None:
        raise ValueError("twos must be provided (or available as ref0.mol.spin)")
    twos = int(twos)

    method0 = str(getattr(getattr(scf_out, "scf", None), "method", "RHF")).strip().lower()
    if method0 not in ("rhf", "uhf", "rohf"):
        method0 = "rhf"
    backend0 = _infer_backend_from_df_B(getattr(scf_out, "df_B", None))
    auxbasis0 = auxbasis if auxbasis is not None else getattr(scf_out, "auxbasis_name", "autoaux")

    def _mol_at(coords_bohr: np.ndarray) -> Molecule:
        atoms = [(mol0.atom_symbol(i), coords_bohr[i].copy()) for i in range(natm)]
        return Molecule.from_atoms(
            atoms,
            unit="Bohr",
            charge=int(getattr(mol0, "charge")),
            spin=int(getattr(mol0, "spin")),
            basis=getattr(mol0, "basis", None),
            cart=bool(getattr(mol0, "cart", True)),
        )

    ncore = int(getattr(ref0, "ncore"))
    ncas = int(getattr(ref0, "ncas"))
    nelecas = getattr(ref0, "nelecas")
    nroots = int(getattr(ref0, "nroots", 1))
    root_weights = getattr(ref0, "root_weights", None)

    def _energy_at(coords_bohr: np.ndarray, *, guess_scf: Any | None, guess_ref: Any | None) -> tuple[NEVPT2SCDFResult, Any, Any]:
        from asuka.frontend.scf import run_hf_df  # noqa: PLC0415
        from asuka.mcscf import run_casscf  # noqa: PLC0415

        mol = _mol_at(coords_bohr)
        scf_d = run_hf_df(
            mol,
            method=method0,
            backend=backend0,
            df=True,
            guess=guess_scf,
            auxbasis=auxbasis0,
        )
        ref_d = run_casscf(
            scf_d,
            ncore=ncore,
            ncas=ncas,
            nelecas=nelecas,
            backend=backend0,
            df=True,
            guess=guess_ref,
            nroots=nroots,
            root_weights=root_weights,
        )
        e = nevpt2_sc_df_from_ref(
            ref_d,
            scf_out=scf_d,
            state=int(state),
            auxbasis=auxbasis,
            twos=twos,
            semicanonicalize=semicanonicalize,
            pt2_backend=pt2_backend,
            cuda_device=cuda_device,
            max_memory_mb=max_memory_mb,
            verbose=int(verbose),
        )
        return e, scf_d, ref_d

    guess_scf = scf_out
    guess_ref = ref0
    ref, scf_ref, casscf_ref = _energy_at(coords0, guess_scf=guess_scf, guess_ref=guess_ref)

    grad_corr = np.zeros((natm, 3), dtype=np.float64)
    grad_cas = np.zeros((natm, 3), dtype=np.float64) if ref.e_cas is not None else None
    grad_tot = np.zeros((natm, 3), dtype=np.float64) if ref.e_tot is not None else None

    if fd_which is None:
        which = [(ia, xyz) for ia in range(natm) for xyz in range(3)]
    else:
        which = [(int(ia), int(xyz)) for ia, xyz in fd_which]
        for ia, xyz in which:
            if ia < 0 or ia >= natm or xyz < 0 or xyz >= 3:
                raise ValueError("fd_which contains an out-of-range (atom,xyz) entry")

    for ia, xyz in which:
        coords_p = np.array(coords0, copy=True)
        coords_m = np.array(coords0, copy=True)
        coords_p[ia, xyz] += fd_step_bohr
        coords_m[ia, xyz] -= fd_step_bohr

        if bool(fd_reset_to_ref):
            guess_scf = scf_out
            guess_ref = ref0
        e_p, scf_p, ref_p = _energy_at(coords_p, guess_scf=guess_scf, guess_ref=guess_ref)
        if not bool(fd_reset_to_ref):
            guess_scf, guess_ref = scf_p, ref_p

        if bool(fd_reset_to_ref):
            guess_scf = scf_out
            guess_ref = ref0
        e_m, scf_m, ref_m = _energy_at(coords_m, guess_scf=guess_scf, guess_ref=guess_ref)
        if not bool(fd_reset_to_ref):
            guess_scf, guess_ref = scf_m, ref_m

        grad_corr[ia, xyz] = (e_p.e_corr - e_m.e_corr) / (2.0 * fd_step_bohr)

        if grad_cas is not None:
            if e_p.e_cas is None or e_m.e_cas is None:
                raise RuntimeError("reference provides e_cas but displaced geometry does not")
            grad_cas[ia, xyz] = (float(e_p.e_cas) - float(e_m.e_cas)) / (2.0 * fd_step_bohr)

        if grad_tot is not None:
            if e_p.e_tot is None or e_m.e_tot is None:
                raise RuntimeError("reference provides e_tot but displaced geometry does not")
            grad_tot[ia, xyz] = (float(e_p.e_tot) - float(e_m.e_tot)) / (2.0 * fd_step_bohr)

    _ = _energy_at(coords0, guess_scf=scf_out, guess_ref=ref0)

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
