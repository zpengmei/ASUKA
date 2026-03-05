from __future__ import annotations

"""Finite-difference nuclear gradients for ASUKA-native MRCISD.

This backend is intended for debugging/validation on small systems. It is
expensive because it rebuilds SCF → CASSCF → MRCISD at every displaced geometry.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.mrci.common import assign_roots_by_overlap
from asuka.mrci.driver_asuka import mrci_states_from_ref
from asuka.mrci.result import MRCIStatesResult


@dataclass(frozen=True)
class MRCIFDPoint:
    coords_bohr: np.ndarray
    e_tot: np.ndarray  # (nstate,)


def _infer_hf_method(scf_out: Any) -> str:
    """Map `SCFResult.method` to frontend `run_hf_df(method=...)`."""

    scf = getattr(scf_out, "scf", None)
    method = "" if scf is None else str(getattr(scf, "method", "")).strip().upper()
    if method.startswith("RHF"):
        return "rhf"
    if method.startswith("ROHF"):
        return "rohf"
    if method.startswith("UHF"):
        return "uhf"
    # Best-effort default.
    return "rhf"


def _infer_backend_from_scf_out(scf_out: Any) -> Literal["cpu", "cuda"]:
    B = getattr(scf_out, "df_B", None)
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(B, cp.ndarray):  # type: ignore[attr-defined]
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _mol_with_coords_like(mol0: Any, coords_bohr: np.ndarray):
    from asuka.frontend.molecule import Molecule  # noqa: PLC0415

    coords = np.asarray(coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(coords.shape[0])
    atoms = [(str(mol0.atom_symbol(i)), coords[i].copy()) for i in range(natm)]
    return Molecule.from_atoms(
        atoms,
        unit="Bohr",
        charge=int(getattr(mol0, "charge", 0)),
        spin=int(getattr(mol0, "spin", 0)),
        basis=getattr(mol0, "basis", None),
        cart=bool(getattr(mol0, "cart", True)),
    )


def _run_hf_like_reference(scf_out0: Any, mol: Any):
    hf_method = _infer_hf_method(scf_out0)
    if getattr(scf_out0, "df_B", None) is None and getattr(scf_out0, "thc_factors", None) is not None:
        cfg = getattr(scf_out0, "thc_run_config", None)
        if cfg is None:
            raise ValueError("THC FD gradients require scf_out.thc_run_config")
        if str(hf_method) == "rhf":
            from asuka.frontend.scf import run_rhf_thc  # noqa: PLC0415

            return run_rhf_thc(
                mol,
                basis=cfg.basis,
                auxbasis=cfg.auxbasis,
                df_config=cfg.df_config,
                expand_contractions=bool(cfg.expand_contractions),
                thc_mode=str(cfg.thc_mode),
                thc_local_config=cfg.thc_local_config,
                thc_grid_spec=cfg.thc_grid_spec,
                thc_grid_kind=str(cfg.thc_grid_kind),
                thc_dvr_basis=cfg.thc_dvr_basis,
                thc_grid_options=cfg.thc_grid_options,
                thc_npt=cfg.thc_npt,
                thc_solve_method=str(cfg.thc_solve_method),
                use_density_difference=bool(cfg.use_density_difference),
                df_warmup_cycles=int(cfg.df_warmup_cycles),
                df_warmup_ediff=cfg.df_warmup_ediff,
                df_warmup_max_cycles=int(cfg.df_warmup_max_cycles if cfg.df_warmup_max_cycles is not None else 25),
                df_aux_block_naux=int(cfg.df_aux_block_naux),
                df_k_q_block=int(cfg.df_k_q_block),
                max_cycle=int(cfg.max_cycle),
                conv_tol=float(cfg.conv_tol),
                conv_tol_dm=float(cfg.conv_tol_dm),
                diis=bool(cfg.diis),
                diis_start_cycle=cfg.diis_start_cycle,
                diis_space=int(cfg.diis_space),
                damping=float(cfg.damping),
                level_shift=float(cfg.level_shift),
                q_block=int(cfg.q_block),
                init_guess="auto" if cfg.init_guess is None else str(cfg.init_guess),
                mo_coeff0=getattr(getattr(scf_out0, "scf", None), "mo_coeff", None),
                init_fock_cycles=cfg.init_fock_cycles,
            )
        if str(hf_method) == "rohf":
            from asuka.frontend.scf import run_rohf_thc  # noqa: PLC0415

            return run_rohf_thc(
                mol,
                basis=cfg.basis,
                auxbasis=cfg.auxbasis,
                df_config=cfg.df_config,
                expand_contractions=bool(cfg.expand_contractions),
                thc_mode=str(cfg.thc_mode),
                thc_local_config=cfg.thc_local_config,
                thc_grid_spec=cfg.thc_grid_spec,
                thc_grid_kind=str(cfg.thc_grid_kind),
                thc_dvr_basis=cfg.thc_dvr_basis,
                thc_grid_options=cfg.thc_grid_options,
                thc_npt=cfg.thc_npt,
                thc_solve_method=str(cfg.thc_solve_method),
                use_density_difference=bool(cfg.use_density_difference),
                df_warmup_cycles=int(cfg.df_warmup_cycles),
                df_aux_block_naux=int(cfg.df_aux_block_naux),
                df_k_q_block=int(cfg.df_k_q_block),
                max_cycle=int(cfg.max_cycle),
                conv_tol=float(cfg.conv_tol),
                conv_tol_dm=float(cfg.conv_tol_dm),
                diis=bool(cfg.diis),
                diis_start_cycle=cfg.diis_start_cycle,
                diis_space=int(cfg.diis_space),
                damping=float(cfg.damping),
                q_block=int(cfg.q_block),
                mo_coeff0=getattr(getattr(scf_out0, "scf", None), "mo_coeff", None),
            )
        from asuka.frontend.scf import run_uhf_thc  # noqa: PLC0415

        return run_uhf_thc(
            mol,
            basis=cfg.basis,
            auxbasis=cfg.auxbasis,
            df_config=cfg.df_config,
            expand_contractions=bool(cfg.expand_contractions),
            thc_mode=str(cfg.thc_mode),
            thc_local_config=cfg.thc_local_config,
            thc_grid_spec=cfg.thc_grid_spec,
            thc_grid_kind=str(cfg.thc_grid_kind),
            thc_dvr_basis=cfg.thc_dvr_basis,
            thc_grid_options=cfg.thc_grid_options,
            thc_npt=cfg.thc_npt,
            thc_solve_method=str(cfg.thc_solve_method),
            use_density_difference=bool(cfg.use_density_difference),
            df_warmup_cycles=int(cfg.df_warmup_cycles),
            df_aux_block_naux=int(cfg.df_aux_block_naux),
            df_k_q_block=int(cfg.df_k_q_block),
            max_cycle=int(cfg.max_cycle),
            conv_tol=float(cfg.conv_tol),
            conv_tol_dm=float(cfg.conv_tol_dm),
            diis=bool(cfg.diis),
            diis_start_cycle=cfg.diis_start_cycle,
            diis_space=int(cfg.diis_space),
            damping=float(cfg.damping),
            q_block=int(cfg.q_block),
            mo_coeff0=getattr(getattr(scf_out0, "scf", None), "mo_coeff", None),
        )

    df_cfg = getattr(scf_out0, "df_run_config", None)
    if df_cfg is not None:
        from asuka.frontend.scf import run_hf_df  # noqa: PLC0415

        return run_hf_df(
            mol,
            method=str(df_cfg.hf_method),
            backend=str(df_cfg.backend),
            basis=df_cfg.basis,
            auxbasis=df_cfg.auxbasis,
            df_config=df_cfg.df_config,
            expand_contractions=bool(df_cfg.expand_contractions),
            max_cycle=int(df_cfg.max_cycle),
            conv_tol=float(df_cfg.conv_tol),
            conv_tol_dm=float(df_cfg.conv_tol_dm),
            diis=bool(df_cfg.diis),
            diis_start_cycle=df_cfg.diis_start_cycle,
            diis_space=int(df_cfg.diis_space),
            damping=float(df_cfg.damping),
            level_shift=float(df_cfg.level_shift),
            k_q_block=int(df_cfg.k_q_block),
            cublas_math_mode=df_cfg.cublas_math_mode,
            mo_coeff0=getattr(getattr(scf_out0, "scf", None), "mo_coeff", None),
            init_fock_cycles=df_cfg.init_fock_cycles,
            guess=scf_out0,
        )

    from asuka.frontend.scf import run_hf_df  # noqa: PLC0415

    return run_hf_df(
        mol,
        method=str(hf_method),
        backend=str(_infer_backend_from_scf_out(scf_out0)),
        df=True,
        auxbasis=getattr(scf_out0, "auxbasis_name", "autoaux"),
        guess=scf_out0,
    )


def _run_casscf_like_reference(scf_out: Any, ref0: Any):
    from asuka.mcscf.casscf import run_casscf  # noqa: PLC0415

    cfg = getattr(ref0, "run_config", None)
    if cfg is not None:
        kwargs = dict(getattr(cfg, "kwargs", {}) or {})
        root_weights = getattr(cfg, "root_weights", None)
        return run_casscf(
            scf_out,
            ncore=int(getattr(ref0, "ncore", 0)),
            ncas=int(getattr(ref0, "ncas", 0)),
            nelecas=getattr(ref0, "nelecas"),
            backend=str(getattr(cfg, "backend", _infer_backend_from_scf_out(scf_out))),
            df=bool(getattr(cfg, "df", True)),
            guess=ref0,
            matvec_backend=str(getattr(cfg, "matvec_backend", "cuda_eri_mat")),
            nroots=int(getattr(cfg, "nroots", getattr(ref0, "nroots", 1))),
            root_weights=None if root_weights is None else list(root_weights),
            **kwargs,
        )

    return run_casscf(
        scf_out,
        ncore=int(getattr(ref0, "ncore", 0)),
        ncas=int(getattr(ref0, "ncas", 0)),
        nelecas=getattr(ref0, "nelecas"),
        backend=str(_infer_backend_from_scf_out(scf_out)),
        df=True,
        guess=ref0,
        nroots=int(getattr(ref0, "nroots", 1)),
        root_weights=None
        if getattr(ref0, "root_weights", None) is None
        else np.asarray(getattr(ref0, "root_weights"), dtype=np.float64).ravel().tolist(),
    )


def mrci_grad_states_from_ref_fd(
    scf_out0: Any,
    ref0: Any,
    *,
    mrci_states: MRCIStatesResult,
    roots: np.ndarray,
    states: Sequence[int],
    fd_step_bohr: float = 1e-3,
    which: Sequence[tuple[int, int]] | None = None,
    method: str | None = None,
    mrci_kwargs: dict[str, Any] | None = None,
    max_virt_e: int = 2,
    root_follow: Literal["hungarian", "greedy"] = "hungarian",
) -> list[np.ndarray]:
    """Finite-difference gradients for MRCISD total energies (Eh/Bohr).

    Notes
    -----
    This implementation performs a *cold-start* rebuild of SCF, CASSCF, and MRCISD
    at each displaced geometry. It uses overlap-based root assignment inside each
    MRCISD run to associate roots with the requested reference states.
    """

    states_list = [int(s) for s in states]
    roots = np.asarray(roots, dtype=np.int64).ravel()
    if roots.shape != (len(states_list),):
        raise ValueError("roots must have shape (len(states),)")

    delta = float(fd_step_bohr)
    if delta <= 0.0:
        raise ValueError("fd_step_bohr must be > 0")

    mrci_kwargs_use = {} if mrci_kwargs is None else dict(mrci_kwargs)
    mrci_kwargs_use.pop("states", None)
    mrci_kwargs_use.pop("nroots", None)

    mol0 = getattr(ref0, "mol", None)
    if mol0 is None:
        mol0 = getattr(scf_out0, "mol", None)
    if mol0 is None:
        raise ValueError("ref0.mol or scf_out0.mol is required")

    coords0 = np.asarray(getattr(mol0, "coords_bohr"), dtype=np.float64).reshape((-1, 3))
    natm = int(coords0.shape[0])
    if natm == 0:
        return [np.zeros((0, 3), dtype=np.float64) for _ in states_list]
    ncore = int(getattr(ref0, "ncore", 0))
    ncas = int(getattr(ref0, "ncas", 0))
    nelecas = getattr(ref0, "nelecas")
    nroots_ref = int(getattr(ref0, "nroots", 1))
    root_weights = getattr(ref0, "root_weights", None)

    # Energy evaluator returning energies for the requested state list.
    def _energies_at(coords_bohr: np.ndarray) -> np.ndarray:
        mol = _mol_with_coords_like(mol0, coords_bohr)
        scf_out = _run_hf_like_reference(scf_out0, mol)
        casscf = _run_casscf_like_reference(scf_out, ref0)
        mrci_disp = mrci_states_from_ref(
            casscf,
            scf_out=scf_out,
            method=str(method if method is not None else getattr(mrci_states, "method", "mrcisd")),
            states=states_list,
            max_virt_e=int(max_virt_e),
            **mrci_kwargs_use,
        )
        ov = np.asarray(mrci_disp.mrci.overlap_ref_root, dtype=np.float64)
        roots_disp = assign_roots_by_overlap(ov, method=str(root_follow))
        if hasattr(mrci_disp.mrci, "e_mrci"):
            e = np.asarray(mrci_disp.mrci.e_mrci, dtype=np.float64).ravel()
        else:
            e = np.asarray(mrci_disp.mrci.e_tot, dtype=np.float64).ravel()
        return np.asarray([float(e[int(roots_disp[i])]) for i in range(len(states_list))], dtype=np.float64)

    # Central-difference loop (optionally restricted to a subset of coordinates).
    e0 = _energies_at(coords0.copy())
    grads = [np.zeros((natm, 3), dtype=np.float64) for _ in states_list]

    if which is None:
        which_list = [(ia, ax) for ia in range(natm) for ax in range(3)]
    else:
        which_list = [(int(ia), int(ax)) for ia, ax in which]
        if not which_list:
            raise ValueError("which must be non-empty when provided")
        if len(set(which_list)) != len(which_list):
            raise ValueError("which must not contain duplicates")
        for ia, ax in which_list:
            if ia < 0 or ia >= natm:
                raise ValueError(f"atom index out of range: {ia}")
            if ax < 0 or ax >= 3:
                raise ValueError(f"axis index out of range: {ax}")

    for ia, ax in which_list:
            coords_p = coords0.copy()
            coords_m = coords0.copy()
            coords_p[ia, ax] += delta
            coords_m[ia, ax] -= delta
            e_p = _energies_at(coords_p)
            e_m = _energies_at(coords_m)
            de = (e_p - e_m) / (2.0 * delta)
            for i in range(len(states_list)):
                grads[i][ia, ax] = float(de[i])

    return grads


__all__ = ["MRCIFDPoint", "mrci_grad_states_from_ref_fd"]
