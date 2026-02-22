from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.cuguga import build_drt
from asuka.mcscf.zvector import (
    MCSCFZVectorResult,
    build_mcscf_hessian_operator,
    ensure_real_ci_rhs,
    effective_active_rdms_from_ci_zvector,
    pack_orbital_gradient,
    prepare_ci_rhs_for_zvector,
    solve_mcscf_zvector,
)
from asuka.soc.si import SpinFreeState
from asuka.soc.si import SOCIntegrals, compute_si_adjoint_weights, soc_state_interaction, soc_xyz_to_spherical
from asuka.soc.trdm import build_ci_rhs_soc_full, build_rho_soc_m_streaming


def _nelec_total(nelec: int | tuple[int, int]) -> int:
    if isinstance(nelec, tuple):
        return int(nelec[0]) + int(nelec[1])
    return int(nelec)


def _normalize_soc_backend(backend: str) -> Literal["cpu", "cuda", "auto"]:
    mode = str(backend).strip().lower()
    if mode not in ("cpu", "cuda", "auto"):
        raise ValueError("soc_backend must be one of: 'cpu', 'cuda', 'auto'")
    return mode  # type: ignore[return-value]


def _normalize_soc_cuda_gm_strategy(strategy: str) -> Literal["auto", "apply_gemm", "direct_reduction"]:
    mode = str(strategy).strip().lower()
    if mode not in ("auto", "apply_gemm", "direct_reduction"):
        raise ValueError("soc_cuda_gm_strategy must be one of: 'auto', 'apply_gemm', 'direct_reduction'")
    return mode  # type: ignore[return-value]


def spinfree_states_from_mc(
    mc: Any,
    *,
    ci: Any | None = None,
    energies: Sequence[float] | None = None,
    twos: int | None = None,
) -> list[SpinFreeState]:
    """Build `SpinFreeState` objects from a PySCF-like CASSCF/CASCI object.

    Notes
    -----
    This helper is intended for developer workflows where the active-space CI
    vectors in `mc.ci` are in the same CSF basis as cuGUGA's DRT builder.
    """

    if ci is None:
        ci = getattr(mc, "ci", None)
    if ci is None:
        raise ValueError("ci must be provided or present on mc")

    if twos is None:
        twos = getattr(getattr(mc, "fcisolver", None), "twos", None)
    if twos is None:
        raise ValueError("twos must be provided (or available as mc.fcisolver.twos)")
    twos = int(twos)

    norb = int(getattr(mc, "ncas"))
    nelec = _nelec_total(getattr(mc, "nelecas"))
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)

    if isinstance(ci, (list, tuple)):
        ci_list = [np.asarray(v, dtype=np.float64).ravel() for v in ci]
    else:
        ci_list = [np.asarray(ci, dtype=np.float64).ravel()]

    if energies is None:
        e_states = getattr(mc, "e_states", None)
        if e_states is not None:
            e_arr = np.asarray(e_states, dtype=np.float64).ravel()
            if int(e_arr.size) == len(ci_list):
                energies = [float(x) for x in e_arr]
        if energies is None:
            energies = [float(getattr(mc, "e_tot", 0.0)) for _ in range(len(ci_list))]
    if len(energies) != len(ci_list):
        raise ValueError("energies length mismatch")

    states: list[SpinFreeState] = []
    for e, c in zip(energies, ci_list):
        if int(c.size) != int(drt.ncsf):
            raise ValueError("CI vector length does not match the constructed DRT")
        states.append(SpinFreeState(twos=twos, energy=float(e), drt=drt, ci=c))
    return states


def build_soc_ci_rhs_for_zvector(
    *,
    ci0: Any,
    states: list[SpinFreeState],
    eta: np.ndarray,
    h_m: np.ndarray,
    block_nops: int = 8,
    eps: float = 0.0,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
) -> Any:
    """Build and prepare the SOC CI RHS for `solve_mcscf_zvector` (CI-first milestone)."""

    mode = _normalize_soc_backend(str(soc_backend))
    rhs_list = build_ci_rhs_soc_full(
        states,
        eta,
        h_m,
        block_nops=int(block_nops),
        eps=float(eps),
        backend=str(mode),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
    )

    if isinstance(ci0, (list, tuple)):
        rhs_ci: Any = rhs_list
        if len(rhs_ci) != len(ci0):
            raise ValueError("ci0 list length does not match the number of provided states")
    else:
        if len(rhs_list) != 1:
            raise ValueError("ci0 is a single CI vector but multiple states were provided")
        rhs_ci = rhs_list[0]

    return prepare_ci_rhs_for_zvector(
        ci0=ci0,
        rhs_ci=rhs_ci,
        imag_tol=float(imag_tol),
        project_normalized=bool(project_normalized),
    )


@dataclass(frozen=True)
class SOCCIZVectorResponse:
    """CI-response intermediates from a SOC adjoint RHS in the MCSCF Z-vector space."""

    rhs_ci: Any
    rhs_orb: np.ndarray | None
    z: MCSCFZVectorResult
    rho_m: np.ndarray | None
    dm1_ci: np.ndarray
    dm2_ci: np.ndarray


@dataclass(frozen=True)
class SOCSpinManifoldZVectorResponse:
    """Per-(spin manifold) Z-vector response for the SOC adjoint functional."""

    mc: Any
    twos: int
    state_ids: list[int]
    w_state: np.ndarray
    response: SOCCIZVectorResponse


@dataclass(frozen=True)
class SOCMultiSpinZVectorResponse:
    """SOC-SI adjoint and Z-vector responses for multiple spin manifolds."""

    states: list[SpinFreeState]
    si_energies: np.ndarray
    si_vectors: np.ndarray
    si_basis: list[tuple[int, int]]
    w_state: np.ndarray
    eta: np.ndarray
    rho_m: np.ndarray | None
    manifolds: list[SOCSpinManifoldZVectorResponse]


@dataclass(frozen=True)
class SOCSINuclearGradientsResult:
    """SOC-SI nuclear gradients for multiple SO roots (single spin manifold)."""

    so_roots: list[int]
    so_energies: np.ndarray  # (nso,)
    so_vectors: np.ndarray  # (nss, nss) SI spin-component eigenvectors
    so_basis: list[tuple[int, int]]

    gradients: np.ndarray  # (nso, natm, 3)
    grad_sf: np.ndarray  # (nso, natm, 3)
    grad_soc_resp: np.ndarray  # (nso, natm, 3)
    grad_soc_int: np.ndarray  # (nso, natm, 3)

    w_state: np.ndarray  # (nso, nstates)


@dataclass(frozen=True)
class SOCSISingleNuclearGradientResult:
    """Detailed SOC-SI nuclear gradient components for one selected SO root."""

    gradient: np.ndarray  # (natm, 3)
    grad_sf: np.ndarray  # (natm, 3)
    grad_soc_resp: np.ndarray  # (natm, 3)
    grad_soc_int: np.ndarray  # (natm, 3)
    so_energy: float
    w_state: np.ndarray  # (nstates,)


def soc_lagrange_response_nuc_grad(
    mc: Any,
    z: MCSCFZVectorResult,
    *,
    mo_coeff: np.ndarray | None = None,
    ci: Any | None = None,
    atmlst: Sequence[int] | None = None,
    eris: Any | None = None,
    mf_grad: Any | None = None,
    verbose: int | None = None,
    return_parts: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the Z-vector Lagrange contribution ``z · d(g_ref)/dR`` to nuclear gradients.

    This is a thin wrapper around PySCF's SA-CASSCF Lagrange kernels, which are generic enough
    to work with the CSF-based `GUGAFCISolver` interface as long as `mc.gen_g_hop` is available.

    Notes
    -----
    - This term depends only on the *reference* MCSCF stationarity conditions and the solved
      multipliers `z`; it is agnostic to how `z` was constructed (SOC, MRPT2, etc.).
    - The returned gradient is *electronic only*; add nuclear repulsion separately if needed.
    """

    try:
        from pyscf.grad import sacasscf as _sacasscf  # type: ignore[import-not-found]
    except Exception as err:  # pragma: no cover
        raise ImportError("soc_lagrange_response_nuc_grad requires PySCF to be installed") from err

    if mo_coeff is None:
        mo_coeff = getattr(mc, "mo_coeff", None)
    if ci is None:
        ci = getattr(mc, "ci", None)
    if mo_coeff is None or ci is None:
        raise ValueError("mo_coeff/ci must be provided or present on mc")

    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol must be available")

    if atmlst is None:
        atmlst = list(range(int(mol.natm)))
    else:
        atmlst = [int(x) for x in atmlst]

    if verbose is None:
        verbose = int(getattr(mc, "verbose", 0))

    # Reference SA weights (if any). For state-specific, default to [1.0].
    weights = getattr(mc, "weights", None)
    if weights is None:
        if isinstance(ci, (list, tuple)):
            weights = np.ones(len(ci), dtype=np.float64) / float(len(ci))
        else:
            weights = np.ones(1, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("SA weights must be non-negative and sum to a positive number")
    weights_norm = weights / float(np.sum(weights))

    @contextmanager
    def _maybe_set_attr(obj: Any, name: str, value: Any):
        missing = object()
        if obj is None:
            yield False
            return
        try:
            old = getattr(obj, name, missing)
        except Exception:
            yield False
            return
        try:
            setattr(obj, name, value)
        except Exception:
            yield False
            return
        try:
            yield True
        finally:
            try:
                if old is missing:
                    delattr(obj, name)
                else:
                    setattr(obj, name, old)
            except Exception:
                pass

    z_info = getattr(z, "info", None)
    if not isinstance(z_info, dict):
        z_info = {}

    # `solve_mcscf_zvector` works in the packed-variable convention of the underlying
    # Hessian operator. PySCF's `sacasscf` Lagrange-gradient kernels expect the
    # *matrix-form* multipliers in the unscaled convention, so convert here.
    orb_scale = float(z_info.get("lagrange_orb_scale", 1.0))
    ci_scale = float(z_info.get("lagrange_ci_scale", 1.0))

    Lorb = mc.unpack_uniq_var(np.asarray(z.z_orb, dtype=np.float64).ravel() * orb_scale)
    Lci = z.z_ci
    if isinstance(Lci, list):
        Lci = [np.asarray(v, dtype=np.float64) * ci_scale for v in Lci]
    elif Lci is not None:
        Lci = np.asarray(Lci, dtype=np.float64) * ci_scale

    # Build ERIs cache if not provided (needed by the Lagrange kernels).
    if eris is None:
        if not hasattr(mc, "ao2mo"):
            raise AttributeError("mc object does not provide ao2mo required for Lagrange terms")
        eris = mc.ao2mo(mo_coeff)

    # CI and orbital Lagrange contributions.
    #
    # IMPORTANT: In PySCF, `sacasscf.Lci_dot_dgci_dx` uses `mc.fcisolver.trans_rdm12`, which
    # typically pulls SA weights from `mc.fcisolver.weights` (the explicit `weights` argument
    # is not used in some versions). To avoid convention mismatches, enforce a single source
    # of truth by temporarily normalizing and syncing both `mc.weights` and `mc.fcisolver.weights`.
    fs = getattr(mc, "fcisolver", None)
    w_list = [float(x) for x in weights_norm.tolist()]
    with _maybe_set_attr(mc, "weights", w_list), _maybe_set_attr(fs, "weights", w_list):
        de_Lci = _sacasscf.Lci_dot_dgci_dx(
            Lci,
            weights_norm,
            mc,
            mo_coeff=mo_coeff,
            ci=ci,
            atmlst=atmlst,
            mf_grad=mf_grad,
            eris=eris,
            verbose=int(verbose),
        )
        de_Lorb = _sacasscf.Lorb_dot_dgorb_dx(
            Lorb,
            mc,
            mo_coeff=mo_coeff,
            ci=ci,
            atmlst=atmlst,
            mf_grad=mf_grad,
            eris=eris,
            verbose=int(verbose),
        )
        total = np.asarray(de_Lci + de_Lorb, dtype=np.float64)
        if bool(return_parts):
            return total, np.asarray(de_Lorb, dtype=np.float64), np.asarray(de_Lci, dtype=np.float64)
        return total


def soc_lagrange_response_nuc_grad_df(
    mc: Any,
    z: MCSCFZVectorResult,
    *,
    auxbasis: Any = "weigend+etb",
    mo_coeff: np.ndarray | None = None,
    ci: Any | None = None,
    dm1_act: np.ndarray | None = None,
    dm2_act: np.ndarray | None = None,
    atmlst: Sequence[int] | None = None,
    max_memory_mb: float = 4000.0,
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    verbose: int | None = None,
    return_parts: bool = False,
    return_term_breakdown: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]
):
    """ASUKA-internal DF analogue of `soc_lagrange_response_nuc_grad` (no PySCF gradients).

    This contracts the Z-vector multipliers with nuclear derivatives of the SA-CASSCF
    stationarity conditions using:
      - ASUKA DF factors `B[μ,ν,Q]` from `asuka.integrals.df_context` (cuERI-CPU)
      - ASUKA 1e integrals (analytic cart)
      - ASUKA DF 2e derivative backends (analytic when available; else FD fallback)

    The returned gradient is *electronic only* (same convention as the PySCF backend).
    """

    from types import SimpleNamespace  # noqa: PLC0415

    from asuka.frontend.molecule import Molecule  # noqa: PLC0415
    from asuka.frontend.periodic_table import atomic_number  # noqa: PLC0415
    from asuka.integrals.df_context import get_df_cholesky_context  # noqa: PLC0415
    from asuka.integrals.int1e_cart import build_int1e_cart  # noqa: PLC0415
    from asuka.mcscf.nac._df import _Lorb_dot_dgorb_dx_df as _Lorb_dot_dgorb_dx_df_impl  # noqa: PLC0415
    from asuka.mcscf.nac._df import _grad_elec_active_df as _grad_elec_active_df_impl  # noqa: PLC0415
    from asuka.mcscf.newton_df import build_df_newton_eris  # noqa: PLC0415
    from asuka.mcscf.state_average import make_state_averaged_rdms  # noqa: PLC0415

    def _base_fcisolver_method(fcisolver: Any, name: str):
        base_cls = getattr(fcisolver, "_base_class", None)
        if base_cls is not None and hasattr(base_cls, name):
            return getattr(base_cls, name)
        if not hasattr(fcisolver, name):
            raise AttributeError(f"fcisolver does not implement {name}")
        return getattr(fcisolver, name)

    if mo_coeff is None:
        mo_coeff = getattr(mc, "mo_coeff", None)
    if ci is None:
        ci = getattr(mc, "ci", None)
    if mo_coeff is None or ci is None:
        raise ValueError("mo_coeff/ci must be provided or present on mc")

    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol must be available")

    if atmlst is None:
        atmlst = list(range(int(mol.natm)))
    else:
        atmlst = [int(x) for x in atmlst]

    if verbose is None:
        verbose = int(getattr(mc, "verbose", 0))

    # Reference SA weights (if any). For state-specific, default to [1.0].
    weights = getattr(mc, "weights", None)
    if weights is None:
        if isinstance(ci, (list, tuple)):
            weights = np.ones(len(ci), dtype=np.float64) / float(len(ci))
        else:
            weights = np.ones(1, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if np.any(weights < 0.0) or float(np.sum(weights)) <= 0.0:
        raise ValueError("SA weights must be non-negative and sum to a positive number")
    weights_norm = weights / float(np.sum(weights))

    z_info = getattr(z, "info", None)
    if not isinstance(z_info, dict):
        z_info = {}
    orb_scale = float(z_info.get("lagrange_orb_scale", 1.0))
    ci_scale = float(z_info.get("lagrange_ci_scale", 1.0))

    Lorb = mc.unpack_uniq_var(np.asarray(z.z_orb, dtype=np.float64).ravel() * orb_scale)
    Lci = z.z_ci
    if isinstance(Lci, list):
        Lci = [np.asarray(v, dtype=np.float64) * ci_scale for v in Lci]
    elif Lci is not None:
        Lci = np.asarray(Lci, dtype=np.float64) * ci_scale

    ncore = int(getattr(mc, "ncore", 0))
    ncas = int(getattr(mc, "ncas", 0))
    nelecas = getattr(mc, "nelecas", None)
    if ncas <= 0:
        raise ValueError("mc.ncas must be > 0 for DF Lagrange response")

    # DF context: AO DF factors and packed bases (PySCF-order-preserving for PySCF mols).
    df_ctx = get_df_cholesky_context(
        mol,
        auxbasis=auxbasis,
        max_memory=int(max_memory_mb),
        verbose=int(verbose),
        threads=int(df_threads),
    )

    # Minimal internal molecule container for the DF derivative kernels.
    atoms_bohr: list[tuple[str, np.ndarray]] = []
    for ia in range(int(mol.natm)):
        sym_raw = str(mol.atom_symbol(int(ia)))
        sym = sym_raw.strip().capitalize()
        xyz = np.asarray(mol.atom_coord(int(ia)), dtype=np.float64).reshape((3,))
        atoms_bohr.append((sym, xyz))
    mol_df = Molecule.from_atoms(
        atoms_bohr,
        unit="Bohr",
        charge=int(getattr(mol, "charge", 0)),
        spin=int(getattr(mol, "spin", 0)),
        basis=None,
        cart=True,
    )
    coords = np.asarray(mol_df.coords_bohr, dtype=np.float64)
    charges = np.asarray([float(atomic_number(sym)) for sym, _xyz in mol_df.atoms_bohr], dtype=np.float64)
    int1e = build_int1e_cart(df_ctx.ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    scf_out = SimpleNamespace(
        mol=mol_df,
        df_B=np.asarray(df_ctx.B_ao, dtype=np.float64, order="C"),
        ao_basis=df_ctx.ao_basis,
        aux_basis=df_ctx.aux_basis,
        int1e=int1e,
    )
    eris_df = build_df_newton_eris(scf_out.df_B, np.asarray(mo_coeff, dtype=np.float64), ncore=int(ncore), ncas=int(ncas))

    # SA active-space RDMs for the orbital response term.
    if dm1_act is None or dm2_act is None:
        fcisolver = getattr(mc, "fcisolver", None)
        if fcisolver is None:
            raise ValueError("mc.fcisolver is required to build RDMs for DF Lagrange response")
        if isinstance(ci, (list, tuple)):
            ci_list = [np.asarray(v, dtype=np.float64).ravel() for v in ci]
            dm1_act, dm2_act = make_state_averaged_rdms(
                fcisolver,
                ci_list,
                weights_norm,
                ncas=int(ncas),
                nelecas=nelecas,
            )
        else:
            dm1_act, dm2_act = fcisolver.make_rdm12(np.asarray(ci, dtype=np.float64).ravel(), int(ncas), nelecas)

    # CI response: build weighted transition densities between Lci[root] and ci[root].
    fcisolver = getattr(mc, "fcisolver", None)
    if fcisolver is None:
        raise ValueError("mc.fcisolver is required for the DF Lagrange response term")
    trans_rdm12_raw = _base_fcisolver_method(fcisolver, "trans_rdm12")

    # `trans_rdm12` call convention differs across solvers/wrappers:
    # - PySCF SA wrappers: we may grab an *unbound* base-class function and must call it as
    #     trans_rdm12(fcisolver, cibra, ciket, ncas, nelecas)
    # - ASUKA solvers (e.g. GUGAFCISolver): `getattr(fcisolver, ...)` returns a *bound* method:
    #     trans_rdm12(cibra, ciket, ncas, nelecas)
    if getattr(trans_rdm12_raw, "__self__", None) is not None:
        def _trans_rdm12(cibra: np.ndarray, ciket: np.ndarray):
            return trans_rdm12_raw(cibra, ciket, int(ncas), nelecas)
    else:
        def _trans_rdm12(cibra: np.ndarray, ciket: np.ndarray):
            return trans_rdm12_raw(fcisolver, cibra, ciket, int(ncas), nelecas)

    dm1_lci = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
    dm2_lci = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
    if isinstance(ci, (list, tuple)):
        ci_list = [np.asarray(v, dtype=np.float64).ravel() for v in ci]
        if not isinstance(Lci, list) or len(Lci) != len(ci_list):
            raise ValueError("z.z_ci list length does not match the number of reference CI roots")
        for wr, lci_r, ci_r in zip(weights_norm.tolist(), Lci, ci_list):
            wr = float(wr)
            if abs(wr) < 1e-14:
                continue
            dm1_r, dm2_r = _trans_rdm12(np.asarray(lci_r, dtype=np.float64).ravel(), np.asarray(ci_r, dtype=np.float64).ravel())
            dm1_r = np.asarray(dm1_r, dtype=np.float64)
            dm2_r = np.asarray(dm2_r, dtype=np.float64)
            dm1_lci += wr * (dm1_r + dm1_r.T)
            dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))
    else:
        if Lci is None or isinstance(Lci, list):
            raise ValueError("expected a single-root z.z_ci for SS contraction")
        dm1_r, dm2_r = _trans_rdm12(np.asarray(Lci, dtype=np.float64).ravel(), np.asarray(ci, dtype=np.float64).ravel())
        dm1_r = np.asarray(dm1_r, dtype=np.float64)
        dm2_r = np.asarray(dm2_r, dtype=np.float64)
        dm1_lci += (dm1_r + dm1_r.T)
        dm2_lci += (dm2_r + dm2_r.transpose(1, 0, 3, 2))

    if bool(return_term_breakdown) and not bool(return_parts):
        raise ValueError("return_term_breakdown=True requires return_parts=True")

    lci_terms = None
    lorb_terms = None
    if bool(return_term_breakdown):
        de_Lci, lci_terms = _grad_elec_active_df_impl(
            scf_out=scf_out,
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            dm1_act=np.asarray(dm1_lci, dtype=np.float64),
            dm2_act=np.asarray(dm2_lci, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            return_terms=True,
        )
        de_Lorb, lorb_terms = _Lorb_dot_dgorb_dx_df_impl(
            scf_out=scf_out,
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            dm1_act=np.asarray(dm1_act, dtype=np.float64),
            dm2_act=np.asarray(dm2_act, dtype=np.float64),
            Lorb=np.asarray(Lorb, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            eris=eris_df,
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
            return_terms=True,
        )
    else:
        de_Lci = _grad_elec_active_df_impl(
            scf_out=scf_out,
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            dm1_act=np.asarray(dm1_lci, dtype=np.float64),
            dm2_act=np.asarray(dm2_lci, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
        )
        de_Lorb = _Lorb_dot_dgorb_dx_df_impl(
            scf_out=scf_out,
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            dm1_act=np.asarray(dm1_act, dtype=np.float64),
            dm2_act=np.asarray(dm2_act, dtype=np.float64),
            Lorb=np.asarray(Lorb, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            eris=eris_df,
            df_backend=str(df_backend),
            df_config=df_config,
            df_threads=int(df_threads),
        )

    total = np.asarray(de_Lci + de_Lorb, dtype=np.float64)
    if bool(return_parts):
        if bool(return_term_breakdown):
            if not isinstance(lorb_terms, dict) or not isinstance(lci_terms, dict):  # pragma: no cover
                raise RuntimeError("internal error: missing term breakdown dictionaries")
            return (
                total,
                np.asarray(de_Lorb, dtype=np.float64),
                np.asarray(de_Lci, dtype=np.float64),
                {str(k): np.asarray(v, dtype=np.float64) for k, v in lorb_terms.items()},
                {str(k): np.asarray(v, dtype=np.float64) for k, v in lci_terms.items()},
            )
        return total, np.asarray(de_Lorb, dtype=np.float64), np.asarray(de_Lci, dtype=np.float64)
    return total


def _spinfree_state_nuc_grads(mc: Any) -> list[np.ndarray]:
    """Return per-root *total* (electronic + nuclear) gradients for a CASSCF object.

    For state-average CASSCF objects backed by the CSF-based `GUGAFCISolver`, PySCF's
    built-in `sacasscf.Gradients` assumes determinant CI dimensions.  This helper
    provides a small compatibility layer so we can obtain state-specific gradients
    in the CSF parameterization.
    """

    try:
        from pyscf.grad import lagrange as _lagrange  # type: ignore[import-not-found]
        from pyscf.grad import sacasscf as _sacasscf  # type: ignore[import-not-found]
    except Exception as err:  # pragma: no cover
        raise ImportError("_spinfree_state_nuc_grads requires PySCF to be installed") from err

    ci = getattr(mc, "ci", None)
    if ci is None:
        raise ValueError("mc.ci must be available")

    # State-specific (single root): PySCF's standard gradient implementation works.
    if not isinstance(ci, (list, tuple)):
        grad = mc.nuc_grad_method().kernel()
        return [np.asarray(grad, dtype=np.float64)]

    nroots = int(len(ci))
    if nroots == 0:
        raise ValueError("mc.ci is empty")

    # CSF-compatible SA-CASSCF gradients: override CI block sizes to match CSF vector lengths.
    class _CSFSACASSCFGradients(_sacasscf.Gradients):  # type: ignore[misc]
        def __init__(self, mc_in: Any, state: int | None = None):
            super().__init__(mc_in, state=state)
            sizes = [int(np.asarray(c).size) for c in getattr(mc_in, "ci")]
            if any(s <= 0 for s in sizes):
                raise ValueError("Encountered empty CI root in state-average gradient setup")
            self.na_states = sizes
            self.nb_states = [1] * len(sizes)
            self.nci = int(sum(sizes))
            # Re-initialize the Lagrange base class with the corrected total variable count.
            _lagrange.Gradients.__init__(self, mc_in, int(self.ngorb + self.nci))

        def pack_uniq_var(self, xorb, xci):
            xorb = self.base.pack_uniq_var(xorb)
            xci = np.concatenate([np.asarray(x).ravel() for x in xci])
            return np.append(xorb, xci)

        def unpack_uniq_var(self, x):
            x = np.asarray(x).ravel()
            xorb = self.base.unpack_uniq_var(x[: int(self.ngorb)])
            x = x[int(self.ngorb) :]
            xci = []
            for sz in self.na_states:
                xci.append(x[: int(sz)].copy())
                x = x[int(sz) :]
            return xorb, xci

    grads: list[np.ndarray] = []
    for st in range(nroots):
        g = _CSFSACASSCFGradients(mc, state=st).kernel(state=st)
        grads.append(np.asarray(g, dtype=np.float64))
    return grads


def _base_fcisolver_method(mc: Any, name: str):
    """Return the (possibly unwrapped) fcisolver method `name`.

    PySCF state-average fcisolver wrappers implement `trans_rdm12`/`trans_rdm1` with list semantics.
    For pairwise transition densities we need the underlying single-root method.
    """

    solver = getattr(mc, "fcisolver", None)
    if solver is None:
        raise ValueError("mc.fcisolver must be available")

    base_cls = getattr(solver, "_base_class", None)
    if base_cls is not None and hasattr(base_cls, name):
        return getattr(base_cls, name)
    if not hasattr(solver, name):
        raise AttributeError(f"mc.fcisolver does not implement {name}")
    return getattr(solver, name)


def _compute_spinfree_state_derivative_couplings(
    mc: Any,
    *,
    atmlst: Sequence[int] | None = None,
    use_etfs: bool = False,
    mult_ediff: bool = False,
    verbose: int | None = None,
) -> np.ndarray:
    """Compute SA-CASSCF nonadiabatic couplings between spin-free roots (CSF-friendly).

    Returns
    -------
    nac
        Array with shape (nroots, nroots, natm, 3) where nac[bra, ket] = <bra| d(ket)/dR>.
        Diagonal blocks are zero.

    Notes
    -----
    This is a CSF-compatible adaptation of `pyscf.nac.sacasscf.NonAdiabaticCouplings` for use with
    `GUGAFCISolver`-based SA-CASSCF wavefunctions.
    """

    try:
        from pyscf import lib  # type: ignore[import-not-found]
        from pyscf.grad import casscf as _casscf_grad  # type: ignore[import-not-found]
        from pyscf.grad import lagrange as _lagrange  # type: ignore[import-not-found]
        from pyscf.grad import sacasscf as _sacasscf_grad  # type: ignore[import-not-found]
        from asuka.mcscf import newton_casscf as _newton_casscf  # noqa: PLC0415
    except Exception as err:  # pragma: no cover
        raise ImportError("_compute_spinfree_state_derivative_couplings requires PySCF to be installed") from err

    ci = getattr(mc, "ci", None)
    if not isinstance(ci, (list, tuple)) or len(ci) < 2:
        raise ValueError("mc.ci must be a list/tuple with >= 2 roots to compute derivative couplings")

    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol must be available")

    if verbose is None:
        verbose = int(getattr(mc, "verbose", 0))

    if atmlst is None:
        atmlst = list(range(int(mol.natm)))
    else:
        atmlst = [int(x) for x in atmlst]

    nroots = int(len(ci))
    nac = np.zeros((nroots, nroots, len(atmlst), 3), dtype=np.float64)

    def _unpack_state(state):
        ket, bra = state
        ket = int(ket)
        bra = int(bra)
        if ket < 0 or bra < 0 or ket >= nroots or bra >= nroots:
            raise ValueError("state indices out of range")
        return ket, bra

    def _gen_g_hop_active(casscf, mo, ci0, eris, *, verbose_local=0):
        moH = mo.conj().T
        ncore = int(casscf.ncore)
        vnocore = np.asarray(eris.vhf_c).copy()
        vnocore[:, :ncore] = -moH @ casscf.get_hcore() @ mo[:, :ncore]
        with lib.temporary_env(eris, vhf_c=vnocore):
            return _newton_casscf.gen_g_hop(casscf, mo, ci0, eris, verbose=verbose_local)

    def _grad_elec_core(mc_grad, mo_coeff=None, atmlst_local=None, eris=None, mf_grad=None):
        mc_local = mc_grad.base
        if mo_coeff is None:
            mo_coeff = mc_local.mo_coeff
        if eris is None:
            eris = mc_local.ao2mo(mo_coeff)
        if mf_grad is None:
            mf_grad = mc_local._scf.nuc_grad_method()

        ncore = int(mc_local.ncore)
        moH = mo_coeff.conj().T
        f0 = (moH @ mc_local.get_hcore() @ mo_coeff) + eris.vhf_c
        mo_energy = np.asarray(f0.diagonal()).copy()
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:ncore] = 2.0

        # Patch the energy-weighted density matrix for the doubly-occupied core.
        f0_occ = f0 * mo_occ[None, :]
        dme_sf = mo_coeff @ ((f0_occ + f0_occ.T) * 0.5) @ moH

        uhf_like = str(getattr(getattr(mf_grad, "grad_elec", None), "__module__", "")).endswith(".uhf")

        def _make_rdm1e(_mo_energy=None, _mo_coeff=None, _mo_occ=None):
            if uhf_like:
                half = 0.5 * np.asarray(dme_sf)
                return np.asarray((half, half))
            return np.asarray(dme_sf)

        with lib.temporary_env(mf_grad, make_rdm1e=_make_rdm1e, verbose=0):
            with lib.temporary_env(mf_grad.base, mo_coeff=mo_coeff, mo_occ=mo_occ):
                return mf_grad.grad_elec(
                    mo_coeff=mo_coeff, mo_energy=mo_energy, mo_occ=mo_occ, atmlst=atmlst_local
                )

    def _grad_elec_active(mc_grad, mo_coeff=None, ci_local=None, atmlst_local=None, eris=None, mf_grad=None):
        de = mc_grad.grad_elec(mo_coeff=mo_coeff, ci=ci_local, atmlst=atmlst_local, verbose=0)
        de -= _grad_elec_core(mc_grad, mo_coeff=mo_coeff, atmlst_local=atmlst_local, eris=eris, mf_grad=mf_grad)
        return de

    def _nac_csf_from_tm1(mol_local, mf_grad, tm1, atmlst_local):
        aoslices = mol_local.aoslice_by_atom()
        s1 = mf_grad.get_ovlp(mol_local)
        out = np.zeros((len(atmlst_local), 3), dtype=np.float64)
        for k, ia in enumerate(atmlst_local):
            _shl0, _shl1, p0, p1 = aoslices[int(ia)]
            out[k] += 0.5 * np.einsum("xij,ij->x", s1[:, p0:p1], tm1[p0:p1])
        return out

    def _nac_csf(mc_grad, mo_coeff, ci_all, state, mf_grad, atmlst_local):
        ket, bra = _unpack_state(state)
        mc_local = mc_grad.base
        ncore = int(mc_local.ncore)
        ncas = int(mc_local.ncas)
        nocc = ncore + ncas
        nelecas = mc_local.nelecas

        trans_rdm1 = _base_fcisolver_method(mc_local, "trans_rdm1")
        dm1 = trans_rdm1(mc_local.fcisolver, ci_all[bra], ci_all[ket], ncas, nelecas)
        # dm1[p,q] = <bra|E_{q p}|ket> => dm1.T - dm1 gives <bra|E_{p q} - E_{q p}|ket>
        castm1 = np.asarray(dm1).conj().T - np.asarray(dm1)
        mo_cas = np.asarray(mo_coeff)[:, ncore:nocc]
        tm1 = mo_cas @ castm1 @ mo_cas.conj().T
        nac_csf_val = _nac_csf_from_tm1(mc_local.mol, mf_grad, tm1, atmlst_local)

        # Multiply by energy difference to return the numerator; division (if requested) happens in kernel().
        e_bra = float(np.asarray(getattr(mc_local, "e_states"))[bra])
        e_ket = float(np.asarray(getattr(mc_local, "e_states"))[ket])
        return nac_csf_val * (e_bra - e_ket)

    class _CSFNonAdiabaticCouplings(_sacasscf_grad.Gradients):  # type: ignore[misc]
        def __init__(self, mc_in: Any, *, state: tuple[int, int], mult_ediff: bool, use_etfs: bool):
            self.mult_ediff = bool(mult_ediff)
            self.use_etfs = bool(use_etfs)
            super().__init__(mc_in, state=state)

            sizes = [int(np.asarray(c).size) for c in getattr(mc_in, "ci")]
            if any(s <= 0 for s in sizes):
                raise ValueError("Encountered empty CI root in derivative coupling setup")
            self.na_states = sizes
            self.nb_states = [1] * len(sizes)
            self.nci = int(sum(sizes))
            _lagrange.Gradients.__init__(self, mc_in, int(self.ngorb + self.nci))

        def pack_uniq_var(self, xorb, xci):
            xorb = self.base.pack_uniq_var(xorb)
            xci = np.concatenate([np.asarray(x).ravel() for x in xci])
            return np.append(xorb, xci)

        def unpack_uniq_var(self, x):
            x = np.asarray(x).ravel()
            xorb = self.base.unpack_uniq_var(x[: int(self.ngorb)])
            x = x[int(self.ngorb) :]
            xci = []
            for sz in self.na_states:
                xci.append(x[: int(sz)].copy())
                x = x[int(sz) :]
            return xorb, xci

        def make_fcasscf_nacs(self, state=None, casscf_attr=None, fcisolver_attr=None):
            if state is None:
                state = self.state
            ket, bra = _unpack_state(state)
            if casscf_attr is None:
                casscf_attr = {}
            if fcisolver_attr is None:
                fcisolver_attr = {}

            ci_all = self.base.ci
            ncas = int(self.base.ncas)
            nelecas = self.base.nelecas

            trans_rdm12 = _base_fcisolver_method(self.base, "trans_rdm12")
            tdm1, tdm2 = trans_rdm12(self.base.fcisolver, ci_all[bra], ci_all[ket], ncas, nelecas)
            tdm1 = 0.5 * (np.asarray(tdm1) + np.asarray(tdm1).T)
            tdm2 = 0.5 * (np.asarray(tdm2) + np.asarray(tdm2).transpose(1, 0, 3, 2))

            fcisolver_attr["make_rdm12"] = lambda *_a, **_k: (tdm1, tdm2)
            fcisolver_attr["make_rdm1"] = lambda *_a, **_k: tdm1
            fcisolver_attr["make_rdm2"] = lambda *_a, **_k: tdm2

            return _sacasscf_grad.Gradients.make_fcasscf(  # type: ignore[misc]
                self, state=ket, casscf_attr=casscf_attr, fcisolver_attr=fcisolver_attr
            )

        def get_wfn_response(self, atmlst=None, state=None, verbose=None, mo=None, ci=None, **kwargs):
            if state is None:
                state = self.state
            if atmlst is None:
                atmlst = self.atmlst
            if verbose is None:
                verbose = self.verbose
            if mo is None:
                mo = self.base.mo_coeff
            if ci is None:
                ci = self.base.ci
            ket, bra = _unpack_state(state)

            fcasscf = self.make_fcasscf_nacs(state)
            fcasscf.mo_coeff = mo
            fcasscf.ci = ci[ket]
            eris = fcasscf.ao2mo(mo)

            g_all_ket = _gen_g_hop_active(fcasscf, mo, ci[ket], eris, verbose_local=0)[0]
            g_all = np.zeros(self.nlag, dtype=np.float64)
            g_all[: self.ngorb] = g_all_ket[: self.ngorb]

            g_ci_bra = 0.5 * np.asarray(g_all_ket[self.ngorb :], dtype=np.float64).ravel()
            g_all_bra = _gen_g_hop_active(fcasscf, mo, ci[bra], eris, verbose_local=0)[0]
            g_ci_ket = 0.5 * np.asarray(g_all_bra[self.ngorb :], dtype=np.float64).ravel()

            ndet_ket = int(self.na_states[ket] * self.nb_states[ket])
            ndet_bra = int(self.na_states[bra] * self.nb_states[bra])
            if ndet_ket == ndet_bra:
                ket2bra = float(np.dot(np.asarray(ci[bra], dtype=np.float64).ravel(), g_ci_ket))
                bra2ket = float(np.dot(np.asarray(ci[ket], dtype=np.float64).ravel(), g_ci_bra))
                g_ci_ket = g_ci_ket - ket2bra * np.asarray(ci[bra], dtype=np.float64).ravel()
                g_ci_bra = g_ci_bra - bra2ket * np.asarray(ci[ket], dtype=np.float64).ravel()

            offs_ket = int(sum(self.na_states[:ket])) if ket > 0 else 0
            offs_bra = int(sum(self.na_states[:bra])) if bra > 0 else 0
            g_all[self.ngorb :][offs_ket:][:ndet_ket] = g_ci_ket[:ndet_ket]
            g_all[self.ngorb :][offs_bra:][:ndet_bra] = g_ci_bra[:ndet_bra]
            return g_all

        def get_ham_response(self, state=None, atmlst=None, verbose=None, mo=None, ci=None, eris=None, mf_grad=None, **kwargs):
            if state is None:
                state = self.state
            if atmlst is None:
                atmlst = self.atmlst
            if verbose is None:
                verbose = self.verbose
            if mo is None:
                mo = self.base.mo_coeff
            if ci is None:
                ci = self.base.ci
            if mf_grad is None:
                mf_grad = self.base._scf.nuc_grad_method()
            if eris is None and self.eris is None:
                eris = self.eris = self.base.ao2mo(mo)
            elif eris is None:
                eris = self.eris

            ket, _bra = _unpack_state(state)
            fcasscf_grad = _casscf_grad.Gradients(self.make_fcasscf_nacs(state))
            fcasscf_grad._finalize = lambda: None

            nac_num = _grad_elec_active(
                fcasscf_grad,
                mo_coeff=mo,
                ci_local=ci[ket],
                eris=eris,
                mf_grad=mf_grad,
                atmlst_local=atmlst,
            )
            if not bool(kwargs.get("use_etfs", self.use_etfs)):
                nac_num = nac_num + _nac_csf(self, mo_coeff=mo, ci_all=ci, state=state, mf_grad=mf_grad, atmlst_local=atmlst)
            return np.asarray(nac_num, dtype=np.float64)

        def kernel(self, *args, **kwargs):
            mult_ediff_local = kwargs.get("mult_ediff", self.mult_ediff)
            state = kwargs.get("state", self.state)
            ket, bra = _unpack_state(state)
            if ket == bra:
                mol_local = kwargs.get("mol", self.mol)
                atmlst_local = kwargs.get("atmlst", range(int(mol_local.natm)))
                return np.zeros((len(list(atmlst_local)), 3), dtype=np.float64)

            nac_val = _sacasscf_grad.Gradients.kernel(self, *args, **kwargs)
            if not bool(mult_ediff_local):
                e_bra = float(np.asarray(getattr(self.base, "e_states"))[bra])
                e_ket = float(np.asarray(getattr(self.base, "e_states"))[ket])
                nac_val = np.asarray(nac_val, dtype=np.float64) / (e_bra - e_ket)
            return np.asarray(nac_val, dtype=np.float64)

    for ket in range(nroots):
        for bra in range(nroots):
            if ket == bra:
                continue
            nac_pair = _CSFNonAdiabaticCouplings(
                mc, state=(ket, bra), mult_ediff=bool(mult_ediff), use_etfs=bool(use_etfs)
            ).kernel(state=(ket, bra), atmlst=atmlst, verbose=verbose)
            nac[bra, ket] = np.asarray(nac_pair, dtype=np.float64)

    return nac


def _compute_Gm_from_states_and_soc_integrals(
    states: list[SpinFreeState],
    *,
    h_m: np.ndarray,
    block_nops: int = 8,
) -> np.ndarray:
    """Compute G_m(IJ) matrices from triplet TRDMs and SOC integrals.

    Returns
    -------
    Gm
        Complex array of shape (3, nstates, nstates) in m=(-1,0,+1) order.
    """

    if not states:
        raise ValueError("states is empty")

    norb = int(states[0].drt.norb)
    nelec = int(states[0].drt.nelec)
    for st in states:
        if int(st.drt.norb) != norb:
            raise ValueError("all states must have the same norb")
        if int(st.drt.nelec) != nelec:
            raise ValueError("all states must have the same nelec")

    h_m = np.asarray(h_m, dtype=np.complex128)
    if h_m.shape != (3, norb, norb):
        raise ValueError("h_m must have shape (3, norb, norb)")

    from asuka.soc.trdm import trans_trdm1_triplet_all_streaming  # noqa: PLC0415

    nstates = int(len(states))
    Gm = np.zeros((3, nstates, nstates), dtype=np.complex128)

    by_drt: dict[Any, list[int]] = {}
    for idx, st in enumerate(states):
        by_drt.setdefault(st.drt, []).append(int(idx))

    for drt_bra, bra_ids in by_drt.items():
        bra_cis = [states[i].ci for i in bra_ids]
        bra_idx = np.asarray(bra_ids, dtype=np.int32)
        for drt_ket, ket_ids in by_drt.items():
            ket_cis = [states[j].ci for j in ket_ids]
            ket_idx = np.asarray(ket_ids, dtype=np.int32)
            u_blk = trans_trdm1_triplet_all_streaming(
                drt_bra, drt_ket, bra_cis, ket_cis, block_nops=int(block_nops)
            )  # (nb, nk, norb, norb)
            Gm_blk = np.einsum("mpq,ikpq->mik", h_m, u_blk, optimize=True)  # (3, nb, nk)
            ix = np.ix_(bra_idx, ket_idx)
            Gm[:, ix[0], ix[1]] = Gm_blk

    return Gm


def _soc_state_rotation_lci_from_ci_rhs(
    *,
    ci0: Sequence[np.ndarray],
    rhs_ci_raw: Sequence[np.ndarray],
    energies: Sequence[float],
    scale: float,
    ediff_tol: float = 1e-12,
) -> list[np.ndarray]:
    """Build the missing SA root-rotation Lci component from the unprojected CI RHS.

    This follows the OpenMolcas MS/XMS-CASPT2 gradient pattern (see
    `OpenMolcas/src/caspt2/msgrad.f`, XMS_Grad), where the state-rotation multipliers are
    recovered from dot products of the CI vectors against the full CI Lagrangian.

    Parameters
    ----------
    ci0
        List of reference CI vectors (one per SA root).
    rhs_ci_raw
        Unprojected CI RHS vectors for the target functional, matching the structure of `ci0`.
    energies
        Spin-free state energies (one per root) used for the (E_j - E_i) denominators.
    scale
        Overall scale factor (implementation-dependent); the default chosen by callers matches
        the 2*SLag convention used to build pseudo-densities in OpenMolcas.
    """

    ci0 = [np.asarray(c, dtype=np.float64) for c in ci0]
    rhs_ci_raw = [np.asarray(g, dtype=np.float64) for g in rhs_ci_raw]
    if len(ci0) != len(rhs_ci_raw):
        raise ValueError("ci0 and rhs_ci_raw must have the same number of roots")
    nroots = int(len(ci0))
    if nroots < 2:
        return [np.zeros_like(ci0[0], dtype=np.float64)]

    e = np.asarray(list(energies), dtype=np.float64).ravel()
    if int(e.size) != nroots:
        raise ValueError("energies length mismatch for SA root rotation")

    ci_flat = [c.ravel() for c in ci0]
    rhs_flat = [g.ravel() for g in rhs_ci_raw]
    sizes = {int(v.size) for v in ci_flat}
    if len(sizes) != 1:
        raise ValueError("All CI roots must have the same size for SA root rotation")
    if any(int(v.size) != int(next(iter(sizes))) for v in rhs_flat):
        raise ValueError("CI RHS root sizes do not match CI root sizes")

    # A_ij = <c_i|rhs_j>
    a = np.zeros((nroots, nroots), dtype=np.float64)
    for i in range(nroots):
        for j in range(nroots):
            a[i, j] = float(np.dot(ci_flat[i], rhs_flat[j]))

    # Symmetric "state rotation multiplier" matrix (Molcas stores it as SLag).
    s = np.zeros((nroots, nroots), dtype=np.float64)
    for i in range(nroots):
        for j in range(i + 1, nroots):
            ediff = float(e[j] - e[i])
            if abs(ediff) <= float(ediff_tol):
                continue
            num = float(a[i, j] - a[j, i])
            val = float(scale) * num / ediff
            s[i, j] = val
            s[j, i] = val

    lci: list[np.ndarray] = []
    for i in range(nroots):
        acc = np.zeros_like(ci_flat[0], dtype=np.float64)
        for j in range(nroots):
            if s[i, j] == 0.0:
                continue
            acc += float(s[i, j]) * ci_flat[j]
        lci.append(acc.reshape(ci0[i].shape))
    return lci


def _soc_state_rotation_nuc_grad_correction_single(
    mc: Any,
    *,
    states: list[SpinFreeState],
    eta: np.ndarray,
    h_m: np.ndarray,
    block_nops: int,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    mo_coeff: np.ndarray | None = None,
    eris: Any | None = None,
    mf_grad: Any | None = None,
    imag_tol: float = 1e-10,
    offdiag_tol: float = 1e-12,
) -> np.ndarray:
    """State-rotation correction for a single SA-CASSCF manifold (one `mc`).

    For equal-weight SA-CASSCF, the CI response equations are singular in the root-rotation
    subspace.  When the target functional has substantial off-diagonal couplings (e.g. SOC-SI
    with off-diagonal `eta`), projecting out the CI root-subspace components drops a physically
    important contribution.  OpenMolcas treats this via explicit state-rotation multipliers;
    we emulate that by constructing an extra CI Lagrange component in the CI root subspace and
    contracting it with PySCF's SA-CASSCF Lci·dgci/dR kernel.
    """

    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol must be available")
    natm = int(mol.natm)

    ci = getattr(mc, "ci", None)
    if not isinstance(ci, (list, tuple)) or len(ci) < 2:
        return np.zeros((natm, 3), dtype=np.float64)

    eta = np.asarray(eta, dtype=np.complex128)
    nstates = int(eta.shape[1])
    if eta.shape != (3, nstates, nstates):
        raise ValueError("eta must have shape (3, nstates, nstates)")
    if len(states) != nstates:
        raise ValueError("states length mismatch for eta")

    offdiag = eta[:, ~np.eye(nstates, dtype=bool)]
    if float(np.max(np.abs(offdiag))) <= float(offdiag_tol):
        return np.zeros((natm, 3), dtype=np.float64)

    # Build the *unprojected* SOC CI RHS for each SA root.
    mode = _normalize_soc_backend(str(soc_backend))
    rhs_list = build_ci_rhs_soc_full(
        states,
        eta,
        h_m,
        block_nops=int(block_nops),
        eps=0.0,
        backend=str(mode),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
    )
    rhs_list = [np.asarray(v) for v in rhs_list]
    if len(rhs_list) != nstates:
        raise ValueError("SOC CI RHS root count mismatch")

    rhs_list_real = ensure_real_ci_rhs(rhs_list, imag_tol=float(imag_tol))
    ci0 = getattr(mc, "ci", None)
    if not isinstance(ci0, (list, tuple)) or len(ci0) != nstates:
        raise ValueError("mc.ci must be a list matching the number of SA roots")

    # OpenMolcas-like state-rotation multipliers -> CI-space Lagrange component.
    # For PySCF's equal-weight SA objective, the effective scaling matches using
    # `scale = 0.5` for the antisymmetric numerator/(E_j-E_i) (see msgrad.f where
    # SLag is built with 0.25 and later used with a factor 2 in the pseudo-density).
    energies = [float(s.energy) for s in states]
    lci_rot = _soc_state_rotation_lci_from_ci_rhs(
        ci0=[np.asarray(c, dtype=np.float64) for c in ci0],
        rhs_ci_raw=[np.asarray(g, dtype=np.float64) for g in rhs_list_real],
        energies=energies,
        scale=0.5,
    )

    # Contract against the reference SA-CASSCF dgci/dR kernel (CI Lagrange term only).
    try:
        from pyscf.grad import sacasscf as _sacasscf  # type: ignore[import-not-found]
    except Exception as err:  # pragma: no cover
        raise ImportError("SOC state-rotation correction requires PySCF to be installed") from err

    if mo_coeff is None:
        mo_coeff = getattr(mc, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff must be provided or present on mc")

    weights = getattr(mc, "weights", None)
    if weights is None:
        weights = np.ones(nstates, dtype=np.float64) / float(nstates)
    weights = np.asarray(weights, dtype=np.float64).ravel()

    if eris is None:
        eris = mc.ao2mo(np.asarray(mo_coeff))
    if mf_grad is None:
        mf_grad = mc._scf.nuc_grad_method()
    return np.asarray(
        _sacasscf.Lci_dot_dgci_dx(
            lci_rot,
            weights,
            mc,
            mo_coeff=np.asarray(mo_coeff),
            ci=ci0,
            atmlst=list(range(natm)),
            mf_grad=mf_grad,
            eris=eris,
            verbose=0,
        ),
        dtype=np.float64,
    )


def _soc_state_rotation_nuc_grad_correction(
    resp: SOCMultiSpinZVectorResponse,
    *,
    h_m: np.ndarray,
    block_nops: int,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    imag_tol: float = 1e-10,
    offdiag_tol: float = 1e-12,
) -> np.ndarray:
    """Multi-manifold SA root-rotation correction (OpenMolcas-style).

    This applies the same CI-root-subspace correction as
    `_soc_state_rotation_nuc_grad_correction_single` independently to each SA manifold, and sums
    the resulting gradient contributions.
    """

    eta = np.asarray(resp.eta, dtype=np.complex128)
    states = list(resp.states)
    if eta.ndim != 3 or eta.shape[0] != 3:
        raise ValueError("resp.eta must have shape (3, nstates, nstates)")
    nstates = int(eta.shape[1])
    if eta.shape[2] != nstates:
        raise ValueError("resp.eta shape mismatch")
    if len(states) != nstates:
        raise ValueError("resp.states length mismatch")

    offdiag = eta[:, ~np.eye(nstates, dtype=bool)]
    if float(np.max(np.abs(offdiag))) <= float(offdiag_tol):
        mol = getattr(resp.manifolds[0].mc, "mol", None)
        if mol is None:
            raise ValueError("mc.mol must be available")
        return np.zeros((int(mol.natm), 3), dtype=np.float64)

    grad = None
    h_m = np.asarray(h_m, dtype=np.complex128)
    for mresp in resp.manifolds:
        mc = mresp.mc
        state_ids = list(mresp.state_ids)
        if len(state_ids) < 2:
            continue

        eta_sub = eta[:, state_ids, :][:, :, state_ids]
        offdiag = eta_sub[:, ~np.eye(len(state_ids), dtype=bool)]
        if float(np.max(np.abs(offdiag))) <= float(offdiag_tol):
            continue

        states_sub = [states[i] for i in state_ids]
        corr = _soc_state_rotation_nuc_grad_correction_single(
            mc,
            states=states_sub,
            eta=eta_sub,
            h_m=h_m,
            block_nops=int(block_nops),
            soc_backend=str(soc_backend),
            soc_cuda_threads=int(soc_cuda_threads),
            soc_cuda_sync=bool(soc_cuda_sync),
            soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
            imag_tol=float(imag_tol),
            offdiag_tol=float(offdiag_tol),
        )
        grad = corr if grad is None else (grad + corr)

    if grad is None:
        mol = getattr(resp.manifolds[0].mc, "mol", None)
        if mol is None:
            raise ValueError("mc.mol must be available")
        return np.zeros((int(mol.natm), 3), dtype=np.float64)
    return np.asarray(grad, dtype=np.float64)


def _as_dh_m_ao(dh_m_ao: np.ndarray, *, natm: int, nao: int) -> np.ndarray:
    """Canonicalize AO SOC integral derivatives to shape (3, natm, 3, nao, nao)."""

    dh = np.asarray(dh_m_ao)
    if dh.shape == (3, natm, 3, nao, nao):
        return dh
    if dh.shape == (natm, 3, 3, nao, nao):
        # Interpret as (atom, xyz, m, ao, ao)
        return np.transpose(dh, (2, 0, 1, 3, 4))
    raise ValueError(
        "dh_m_ao must have shape (3,natm,3,nao,nao) (m,atom,xyz,ao,ao) "
        "or (natm,3,3,nao,nao) (atom,xyz,m,ao,ao)"
    )


def _soc_integral_derivative_nuc_grad(
    mc: Any,
    *,
    rho_m_act: np.ndarray,
    dh_m_ao: np.ndarray,
    mo_coeff: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the explicit SOC integral-derivative contribution Σ ρ · (dh/dR)."""

    c = _as_mo_coeff(mc, mo_coeff)
    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol must be available")

    natm = int(mol.natm)
    nao = int(c.shape[0])
    dh = _as_dh_m_ao(np.asarray(dh_m_ao), natm=natm, nao=nao)

    act = _active_mo_slice(mc)
    c_act = np.asarray(c[:, act])
    rho_m_act = np.asarray(rho_m_act, dtype=np.complex128)
    if rho_m_act.shape != (3, int(getattr(mc, "ncas")), int(getattr(mc, "ncas"))):
        raise ValueError("rho_m_act must have shape (3, ncas, ncas)")

    # D_ao^(m)[μ,ν] = Σ_{p,q in act} C*_{μp} rho[p,q] C_{νq}
    d_ao = np.empty((3, nao, nao), dtype=np.complex128)
    c_act_conj = np.asarray(c_act.conj())
    c_act_T = np.asarray(c_act.T)
    for m in range(3):
        d_ao[m] = c_act_conj @ rho_m_act[m] @ c_act_T

    # grad[A,xyz] = Re Σ_{m,μ,ν} (dh_m[m,A,xyz,μ,ν] * d_ao[m,μ,ν])
    grad = np.einsum("maxuv,muv->ax", dh, d_ao, optimize=True)
    return np.asarray(grad.real, dtype=np.float64)


def soc_si_nuclear_gradients_all_roots(
    mc: Any,
    soc_integrals: SOCIntegrals,
    *,
    so_roots: Sequence[int] | None = None,
    h_m_ao: np.ndarray | None = None,
    h_xyz_ao: np.ndarray | None = None,
    dh_m_ao: np.ndarray | None = None,
    dh_xyz_ao: np.ndarray | None = None,
    block_nops: int = 8,
    symmetrize: bool = True,
    eps: float = 0.0,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    z_method: Literal["gmres", "gcrotmk"] | None = None,
    z_restart: int | None = None,
    z_gcrotmk_k: int | None = None,
    z_recycle: bool = True,
    z_warm_start: bool = True,
    use_newton_hessian: bool | None = True,
    degeneracy_tol: float = 1e-10,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: Literal["auto", "apply_gemm", "direct_reduction"] = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
) -> SOCSINuclearGradientsResult:
    """Compute SOC-SI nuclear gradients for all (or selected) SO roots in one spin manifold.

    Notes
    -----
    - Reuses the SI diagonalization and spin-free per-root gradients across all requested SO roots.
    - Solves one MCSCF Z-vector per requested SO root.
    - For (near-)degenerate SO energies (within `degeneracy_tol`), returns the averaged gradients
      within each degenerate block to reduce gauge sensitivity of `eigh` eigenvectors.
    """

    if h_m_ao is not None and h_xyz_ao is not None:
        raise ValueError("provide at most one of h_m_ao or h_xyz_ao")
    if dh_m_ao is not None and dh_xyz_ao is not None:
        raise ValueError("provide at most one of dh_m_ao or dh_xyz_ao")
    if h_xyz_ao is not None:
        h_m_ao = soc_xyz_to_spherical(h_xyz_ao)
    if dh_xyz_ao is not None:
        # dh_xyz_ao is expected to have the same leading indices as dh_m_ao; transform the xyz axis (0).
        dh_xyz_ao = np.asarray(dh_xyz_ao, dtype=np.complex128)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        dhx = dh_xyz_ao[0]
        dhy = dh_xyz_ao[1]
        dhz = dh_xyz_ao[2]
        dh_m1 = (dhx - 1j * dhy) * inv_sqrt2
        dh_0 = dhz
        dh_p1 = -(dhx + 1j * dhy) * inv_sqrt2
        dh_m_ao = np.stack([dh_m1, dh_0, dh_p1], axis=0)

    mode = _normalize_soc_backend(str(soc_backend))
    gm_strategy = _normalize_soc_cuda_gm_strategy(str(soc_cuda_gm_strategy))
    gm_direct_max = int(soc_cuda_gm_direct_max_nb_nk)
    if gm_direct_max < 1:
        raise ValueError("soc_cuda_gm_direct_max_nb_nk must be >= 1")

    ci0 = getattr(mc, "ci", None)
    if ci0 is None:
        raise ValueError("mc.ci must be available")
    mol = getattr(mc, "mol", None)
    if mol is None:
        raise ValueError("mc.mol must be available")
    natm = int(mol.natm)

    mo_coeff = getattr(mc, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mc.mo_coeff must be available")
    eris = mc.ao2mo(np.asarray(mo_coeff))
    mf_grad = mc._scf.nuc_grad_method()
    hess_op = build_mcscf_hessian_operator(
        mc,
        mo_coeff=np.asarray(mo_coeff),
        ci=ci0,
        eris=eris,
        use_newton_hessian=use_newton_hessian,
    )

    states = spinfree_states_from_mc(mc)

    e_si, c_si, basis = soc_state_interaction(
        states,
        soc_integrals,
        include_diag=True,
        block_nops=int(block_nops),
        symmetrize=bool(symmetrize),
        backend=str(mode),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        cuda_gm_strategy=str(gm_strategy),
        cuda_gm_direct_max_nb_nk=int(gm_direct_max),
    )
    nso_total = int(e_si.size)

    if so_roots is None:
        roots = list(range(nso_total))
    else:
        roots = [int(r) for r in so_roots]
        if any(r < 0 or r >= nso_total for r in roots):
            raise ValueError("so_roots contains an out-of-range index")
        if len(set(roots)) != len(roots):
            raise ValueError("so_roots contains duplicates")

    if z_method is None:
        method_z = "gcrotmk" if int(len(roots)) > 1 else "gmres"
    else:
        method_z = str(z_method).strip().lower()
    if method_z not in ("gmres", "gcrotmk"):
        raise ValueError("z_method must be 'gmres' or 'gcrotmk'")
    gcrotmk_k_use = z_gcrotmk_k
    if method_z == "gcrotmk" and gcrotmk_k_use is None:
        gcrotmk_k_use = 10
    recycle_space = [] if (method_z == "gcrotmk" and bool(z_recycle)) else None
    x0_z = None

    grads_sf_per_root = _spinfree_state_nuc_grads(mc)
    if int(len(grads_sf_per_root)) != int(len(states)):
        raise ValueError("spin-free gradient root count does not match spin-free state count")

    h_m = soc_integrals.h_m
    if h_m is None:
        if soc_integrals.h_xyz is None:
            raise ValueError("soc_integrals must provide h_m or h_xyz")
        h_m = soc_xyz_to_spherical(soc_integrals.h_xyz)
    h_m = np.asarray(h_m, dtype=np.complex128)

    nout = int(len(roots))
    gradients = np.zeros((nout, natm, 3), dtype=np.float64)
    grad_sf = np.zeros_like(gradients)
    grad_soc_resp = np.zeros_like(gradients)
    grad_soc_int = np.zeros_like(gradients)
    w_state_all = np.zeros((nout, int(len(states))), dtype=np.float64)

    for out_i, so_root in enumerate(roots):
        w_state, eta = compute_si_adjoint_weights(states, basis, c_si[:, int(so_root)])
        w_state = np.asarray(w_state, dtype=np.float64).ravel()
        w_state_all[out_i] = w_state

        g_sf = np.zeros((natm, 3), dtype=np.float64)
        for w, g in zip(w_state, grads_sf_per_root):
            g_sf += float(w) * np.asarray(g, dtype=np.float64)
        grad_sf[out_i] = g_sf

        rhs_ci = build_soc_ci_rhs_for_zvector(
            ci0=ci0,
            states=states,
            eta=eta,
            h_m=h_m,
            block_nops=int(block_nops),
            eps=float(eps),
            soc_backend=str(mode),
            soc_cuda_threads=int(soc_cuda_threads),
            soc_cuda_sync=bool(soc_cuda_sync),
            soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
            imag_tol=float(imag_tol),
            project_normalized=bool(project_normalized),
        )

        rhs_orb = None
        rho_m = None
        if h_m_ao is not None or dh_m_ao is not None:
            rho_m = build_rho_soc_m_streaming(
                states,
                eta,
                block_nops=int(block_nops),
                eps=float(eps),
                backend=str(mode),
                cuda_threads=int(soc_cuda_threads),
                cuda_sync=bool(soc_cuda_sync),
                cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
            )
        if h_m_ao is not None:
            rhs_orb = build_soc_orbital_rhs_for_zvector(
                mc,
                rho_m_act=np.asarray(rho_m),
                h_m_ao=np.asarray(h_m_ao),
                imag_tol=float(imag_tol),
            )

        z = solve_mcscf_zvector(
            mc,
            rhs_orb=rhs_orb,
            rhs_ci=rhs_ci,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
            method=method_z,
            restart=z_restart,
            gcrotmk_k=gcrotmk_k_use,
            recycle_space=recycle_space,
            x0=x0_z if bool(z_warm_start) else None,
            hessian_op=hess_op,
        )
        if bool(z_warm_start):
            x0_z = np.asarray(z.z_packed, dtype=np.float64).ravel()

        g_resp = soc_lagrange_response_nuc_grad(
            mc,
            z,
            mo_coeff=np.asarray(mo_coeff),
            ci=ci0,
            eris=eris,
            mf_grad=mf_grad,
            verbose=0,
        )
        if bool(project_normalized):
            g_resp += _soc_state_rotation_nuc_grad_correction_single(
                mc,
                states=states,
                eta=eta,
                h_m=h_m,
                block_nops=int(block_nops),
                soc_backend=str(mode),
                soc_cuda_threads=int(soc_cuda_threads),
                soc_cuda_sync=bool(soc_cuda_sync),
                soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
                mo_coeff=np.asarray(mo_coeff),
                eris=eris,
                mf_grad=mf_grad,
            )
        grad_soc_resp[out_i] = np.asarray(g_resp, dtype=np.float64)

        if dh_m_ao is not None:
            if rho_m is None:
                rho_m = build_rho_soc_m_streaming(
                    states,
                    eta,
                    block_nops=int(block_nops),
                    eps=float(eps),
                    backend=str(mode),
                    cuda_threads=int(soc_cuda_threads),
                    cuda_sync=bool(soc_cuda_sync),
                    cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
                )
            grad_soc_int[out_i] = _soc_integral_derivative_nuc_grad(
                mc, rho_m_act=np.asarray(rho_m), dh_m_ao=np.asarray(dh_m_ao)
            )

        gradients[out_i] = grad_sf[out_i] + grad_soc_resp[out_i] + grad_soc_int[out_i]

    degeneracy_tol = float(degeneracy_tol)
    if degeneracy_tol > 0.0 and nout > 1:
        order = np.argsort(np.asarray(e_si[roots], dtype=np.float64))
        blocks: list[list[int]] = []
        cur: list[int] = [int(order[0])]
        e0 = float(e_si[roots[int(order[0])]])
        for idx in order[1:]:
            ei = float(e_si[roots[int(idx)]])
            if abs(ei - e0) <= degeneracy_tol:
                cur.append(int(idx))
            else:
                blocks.append(cur)
                cur = [int(idx)]
                e0 = ei
        blocks.append(cur)

        for blk in blocks:
            if len(blk) < 2:
                continue
            gradients[blk] = np.mean(gradients[blk], axis=0)
            grad_sf[blk] = np.mean(grad_sf[blk], axis=0)
            grad_soc_resp[blk] = np.mean(grad_soc_resp[blk], axis=0)
            grad_soc_int[blk] = np.mean(grad_soc_int[blk], axis=0)
            w_state_all[blk] = np.mean(w_state_all[blk], axis=0)

    return SOCSINuclearGradientsResult(
        so_roots=roots,
        so_energies=np.asarray(e_si[roots], dtype=np.float64),
        so_vectors=np.asarray(c_si, dtype=np.complex128),
        so_basis=basis,
        gradients=gradients,
        grad_sf=grad_sf,
        grad_soc_resp=grad_soc_resp,
        grad_soc_int=grad_soc_int,
        w_state=w_state_all,
    )


def soc_si_nuclear_gradient(
    mc: Any,
    soc_integrals: SOCIntegrals,
    *,
    so_root: int = 0,
    h_m_ao: np.ndarray | None = None,
    h_xyz_ao: np.ndarray | None = None,
    dh_m_ao: np.ndarray | None = None,
    dh_xyz_ao: np.ndarray | None = None,
    block_nops: int = 8,
    symmetrize: bool = True,
    eps: float = 0.0,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    use_newton_hessian: bool | None = True,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: Literal["auto", "apply_gemm", "direct_reduction"] = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
    return_parts: bool = False,
) -> np.ndarray | SOCSISingleNuclearGradientResult:
    """Compute the full SO-mixed SI nuclear gradient for one spin manifold.

    The returned gradient is the total derivative of the selected SO-mixed eigenvalue `E_a`
    (including nuclear repulsion via the spin-free state gradients).

    Notes
    -----
    - SOC integral derivatives are optional. If `dh_m_ao` (or `dh_xyz_ao`) is not provided,
      the explicit SOC integral-derivative term is skipped.
    - If AO SOC integrals `h_m_ao`/`h_xyz_ao` are not provided, the SOC orbital RHS is omitted.
    - Use `soc_backend`/`soc_cuda_*` to force deterministic backend behavior during debugging.
    - Set `return_parts=True` to obtain a component-wise gradient breakdown.
    """

    if h_m_ao is not None and h_xyz_ao is not None:
        raise ValueError("provide at most one of h_m_ao or h_xyz_ao")
    if dh_m_ao is not None and dh_xyz_ao is not None:
        raise ValueError("provide at most one of dh_m_ao or dh_xyz_ao")
    if h_xyz_ao is not None:
        h_m_ao = soc_xyz_to_spherical(h_xyz_ao)
    if dh_xyz_ao is not None:
        # dh_xyz_ao is expected to have the same leading indices as dh_m_ao; transform the xyz axis (0).
        dh_xyz_ao = np.asarray(dh_xyz_ao, dtype=np.complex128)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        dhx = dh_xyz_ao[0]
        dhy = dh_xyz_ao[1]
        dhz = dh_xyz_ao[2]
        dh_m1 = (dhx - 1j * dhy) * inv_sqrt2
        dh_0 = dhz
        dh_p1 = -(dhx + 1j * dhy) * inv_sqrt2
        dh_m_ao = np.stack([dh_m1, dh_0, dh_p1], axis=0)

    mode = _normalize_soc_backend(str(soc_backend))
    gm_strategy = _normalize_soc_cuda_gm_strategy(str(soc_cuda_gm_strategy))
    gm_direct_max = int(soc_cuda_gm_direct_max_nb_nk)
    if gm_direct_max < 1:
        raise ValueError("soc_cuda_gm_direct_max_nb_nk must be >= 1")

    states = spinfree_states_from_mc(mc)

    # SI diagonalization and adjoint weights.
    e_si, c_si, basis = soc_state_interaction(
        states,
        soc_integrals,
        include_diag=True,
        block_nops=int(block_nops),
        symmetrize=bool(symmetrize),
        backend=str(mode),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        cuda_gm_strategy=str(gm_strategy),
        cuda_gm_direct_max_nb_nk=int(gm_direct_max),
    )
    so_root = int(so_root)
    if so_root < 0 or so_root >= int(e_si.size):
        raise ValueError("so_root out of range for SI eigenvectors")

    w_state, eta = compute_si_adjoint_weights(states, basis, c_si[:, so_root])

    # 1) Spin-free energy gradients (per-root), weighted by SI adjoints w_state.
    grads_sf = _spinfree_state_nuc_grads(mc)
    if int(len(grads_sf)) != int(w_state.size):
        raise ValueError("spin-free gradient root count does not match w_state length")
    grad_sf = np.zeros_like(grads_sf[0], dtype=np.float64)
    for w, g in zip(np.asarray(w_state, dtype=np.float64).ravel(), grads_sf):
        grad_sf += float(w) * np.asarray(g, dtype=np.float64)

    # 2) SOC response term via Z-vector.
    h_m = soc_integrals.h_m
    if h_m is None:
        if soc_integrals.h_xyz is None:
            raise ValueError("soc_integrals must provide h_m or h_xyz")
        h_m = soc_xyz_to_spherical(soc_integrals.h_xyz)
    h_m = np.asarray(h_m, dtype=np.complex128)

    resp = solve_soc_ci_zvector_response(
        mc,
        states=states,
        eta=eta,
        h_m=h_m,
        h_m_ao=np.asarray(h_m_ao) if h_m_ao is not None else None,
        block_nops=int(block_nops),
        eps=float(eps),
        soc_backend=str(mode),
        soc_cuda_threads=int(soc_cuda_threads),
        soc_cuda_sync=bool(soc_cuda_sync),
        soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        imag_tol=float(imag_tol),
        project_normalized=bool(project_normalized),
        z_tol=float(z_tol),
        z_maxiter=int(z_maxiter),
        use_newton_hessian=use_newton_hessian,
    )

    grad_soc_resp = soc_lagrange_response_nuc_grad(mc, resp.z)
    if bool(project_normalized):
        grad_soc_resp += _soc_state_rotation_nuc_grad_correction_single(
            mc,
            states=states,
            eta=eta,
            h_m=h_m,
            block_nops=int(block_nops),
            soc_backend=str(mode),
            soc_cuda_threads=int(soc_cuda_threads),
            soc_cuda_sync=bool(soc_cuda_sync),
            soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        )

    # 3) Explicit SOC integral-derivative term (optional).
    grad_soc_int = np.zeros_like(grad_sf, dtype=np.float64)
    if dh_m_ao is not None:
        rho_m = resp.rho_m
        if rho_m is None:
            rho_m = build_rho_soc_m_streaming(
                states,
                eta,
                block_nops=int(block_nops),
                eps=float(eps),
                backend=str(mode),
                cuda_threads=int(soc_cuda_threads),
                cuda_sync=bool(soc_cuda_sync),
                cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
            )
        grad_soc_int = _soc_integral_derivative_nuc_grad(mc, rho_m_act=rho_m, dh_m_ao=np.asarray(dh_m_ao))

    total = np.asarray(grad_sf + grad_soc_resp + grad_soc_int, dtype=np.float64)
    if not bool(return_parts):
        return total
    return SOCSISingleNuclearGradientResult(
        gradient=total,
        grad_sf=np.asarray(grad_sf, dtype=np.float64),
        grad_soc_resp=np.asarray(grad_soc_resp, dtype=np.float64),
        grad_soc_int=np.asarray(grad_soc_int, dtype=np.float64),
        so_energy=float(np.asarray(e_si, dtype=np.float64)[int(so_root)]),
        w_state=np.asarray(w_state, dtype=np.float64),
    )


def soc_si_nuclear_gradient_multi_spin(
    mcs: Sequence[Any],
    soc_integrals: SOCIntegrals,
    *,
    so_root: int = 0,
    h_m_ao: np.ndarray | None = None,
    h_xyz_ao: np.ndarray | None = None,
    dh_m_ao: np.ndarray | None = None,
    dh_xyz_ao: np.ndarray | None = None,
    block_nops: int = 8,
    symmetrize: bool = True,
    eps: float = 0.0,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    use_newton_hessian: bool | None = True,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: Literal["auto", "apply_gemm", "direct_reduction"] = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
) -> np.ndarray:
    """Compute the full SO-mixed SI nuclear gradient for multiple spin manifolds."""

    if h_m_ao is not None and h_xyz_ao is not None:
        raise ValueError("provide at most one of h_m_ao or h_xyz_ao")
    if dh_m_ao is not None and dh_xyz_ao is not None:
        raise ValueError("provide at most one of dh_m_ao or dh_xyz_ao")
    if h_xyz_ao is not None:
        h_m_ao = soc_xyz_to_spherical(h_xyz_ao)
    if dh_xyz_ao is not None:
        dh_xyz_ao = np.asarray(dh_xyz_ao, dtype=np.complex128)
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        dhx = dh_xyz_ao[0]
        dhy = dh_xyz_ao[1]
        dhz = dh_xyz_ao[2]
        dh_m1 = (dhx - 1j * dhy) * inv_sqrt2
        dh_0 = dhz
        dh_p1 = -(dhx + 1j * dhy) * inv_sqrt2
        dh_m_ao = np.stack([dh_m1, dh_0, dh_p1], axis=0)

    mode = _normalize_soc_backend(str(soc_backend))
    gm_strategy = _normalize_soc_cuda_gm_strategy(str(soc_cuda_gm_strategy))
    gm_direct_max = int(soc_cuda_gm_direct_max_nb_nk)
    if gm_direct_max < 1:
        raise ValueError("soc_cuda_gm_direct_max_nb_nk must be >= 1")

    resp = solve_soc_ci_zvector_response_multi_spin(
        mcs,
        soc_integrals,
        so_root=int(so_root),
        h_m_ao=np.asarray(h_m_ao) if h_m_ao is not None else None,
        block_nops=int(block_nops),
        symmetrize=bool(symmetrize),
        eps=float(eps),
        soc_backend=str(mode),
        soc_cuda_threads=int(soc_cuda_threads),
        soc_cuda_sync=bool(soc_cuda_sync),
        soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        soc_cuda_gm_strategy=str(gm_strategy),
        soc_cuda_gm_direct_max_nb_nk=int(gm_direct_max),
        imag_tol=float(imag_tol),
        project_normalized=bool(project_normalized),
        z_tol=float(z_tol),
        z_maxiter=int(z_maxiter),
        use_newton_hessian=use_newton_hessian,
    )

    # Weighted sum of spin-free state gradients.
    grad_sf = None
    for mresp in resp.manifolds:
        grads = _spinfree_state_nuc_grads(mresp.mc)
        w = np.asarray(mresp.w_state, dtype=np.float64).ravel()
        if len(grads) != int(w.size):
            raise ValueError("spin-free gradient root count mismatch in multi-spin driver")
        g = np.zeros_like(grads[0], dtype=np.float64)
        for wi, gi in zip(w, grads):
            g += float(wi) * np.asarray(gi, dtype=np.float64)
        grad_sf = g if grad_sf is None else (grad_sf + g)
    if grad_sf is None:
        raise ValueError("no manifolds provided")

    # SOC response term from each manifold's Z-vector.
    grad_soc_resp = np.zeros_like(grad_sf, dtype=np.float64)
    for mresp in resp.manifolds:
        grad_soc_resp += soc_lagrange_response_nuc_grad(mresp.mc, mresp.response.z)
    if bool(project_normalized):
        h_m = soc_integrals.h_m
        if h_m is None:
            if soc_integrals.h_xyz is None:
                raise ValueError("soc_integrals must provide h_m or h_xyz")
            h_m = soc_xyz_to_spherical(soc_integrals.h_xyz)
        h_m = np.asarray(h_m, dtype=np.complex128)
        grad_soc_resp += _soc_state_rotation_nuc_grad_correction(
            resp,
            h_m=h_m,
            block_nops=int(block_nops),
            soc_backend=str(mode),
            soc_cuda_threads=int(soc_cuda_threads),
            soc_cuda_sync=bool(soc_cuda_sync),
            soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        )

    # Explicit SOC integral-derivative term (optional, uses global rho_m).
    grad_soc_int = np.zeros_like(grad_sf, dtype=np.float64)
    if dh_m_ao is not None:
        if resp.rho_m is None:
            raise ValueError("rho_m is not available to build the SOC integral-derivative term")
        grad_soc_int = _soc_integral_derivative_nuc_grad(
            resp.manifolds[0].mc, rho_m_act=resp.rho_m, dh_m_ao=np.asarray(dh_m_ao)
        )

    return np.asarray(grad_sf + grad_soc_resp + grad_soc_int, dtype=np.float64)


def _active_mo_slice(mc: Any) -> slice:
    ncore = int(getattr(mc, "ncore"))
    ncas = int(getattr(mc, "ncas"))
    return slice(ncore, ncore + ncas)


def _as_mo_coeff(mc: Any, mo_coeff: np.ndarray | None) -> np.ndarray:
    if mo_coeff is None:
        mo_coeff = getattr(mc, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff must be provided or present on mc")
    return np.asarray(mo_coeff)


def _validate_common_active_space(mcs: Sequence[Any]) -> tuple[int, int, int]:
    """Validate that all `mc` objects share the same (ncore, ncas, nelecas_total)."""

    mcs = list(mcs)
    if not mcs:
        raise ValueError("mcs is empty")

    ncore0 = int(getattr(mcs[0], "ncore"))
    ncas0 = int(getattr(mcs[0], "ncas"))
    nelec0 = _nelec_total(getattr(mcs[0], "nelecas"))

    for mc in mcs[1:]:
        if int(getattr(mc, "ncore")) != ncore0:
            raise ValueError("All mc objects must have the same ncore for multi-spin SOC-SI")
        if int(getattr(mc, "ncas")) != ncas0:
            raise ValueError("All mc objects must have the same ncas for multi-spin SOC-SI")
        if _nelec_total(getattr(mc, "nelecas")) != nelec0:
            raise ValueError("All mc objects must have the same nelecas for multi-spin SOC-SI")
    return ncore0, ncas0, nelec0


def _validate_common_active_orbitals(
    mcs: Sequence[Any],
    *,
    mo_coeff_tol: float = 1e-10,
) -> None:
    """Validate that all `mc` objects use the same active MO basis (within tolerance)."""

    mcs = list(mcs)
    if not mcs:
        raise ValueError("mcs is empty")

    _validate_common_active_space(mcs)
    act = _active_mo_slice(mcs[0])
    c0 = _as_mo_coeff(mcs[0], None)
    c0_act = np.asarray(c0[:, act])

    for mc in mcs[1:]:
        c = _as_mo_coeff(mc, None)
        if c.shape != c0.shape:
            raise ValueError("All mc objects must have the same mo_coeff shape for multi-spin SOC-SI")
        c_act = np.asarray(c[:, act])
        if not np.allclose(c_act, c0_act, atol=float(mo_coeff_tol), rtol=0.0):
            raise ValueError(
                "All mc objects must share the same active MO coefficients for multi-spin SOC-SI "
                "(common active basis required)."
            )


def soc_ao_to_mo(h_m_ao: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
    """Transform AO-basis SOC integrals to full MO basis.

    Parameters
    ----------
    h_m_ao
        Array of shape (3, nao, nao) in m=(-1,0,+1) order.
    mo_coeff
        MO coefficient matrix (nao, nmo).
    """

    h_m_ao = np.asarray(h_m_ao)
    if h_m_ao.ndim != 3 or h_m_ao.shape[0] != 3 or h_m_ao.shape[1] != h_m_ao.shape[2]:
        raise ValueError("h_m_ao must have shape (3, nao, nao)")
    c = np.asarray(mo_coeff)
    if c.ndim != 2 or int(c.shape[0]) != int(h_m_ao.shape[1]):
        raise ValueError("mo_coeff has incompatible shape for h_m_ao")
    c_h = c.conj().T
    out = np.empty((3, int(c.shape[1]), int(c.shape[1])), dtype=np.result_type(h_m_ao, c))
    for m in range(3):
        out[m] = c_h @ h_m_ao[m] @ c
    return out


def build_soc_orbital_gradient_matrix(
    mc: Any,
    *,
    rho_m_act: np.ndarray,
    h_m_ao: np.ndarray,
    mo_coeff: np.ndarray | None = None,
    imag_tol: float = 1e-10,
) -> np.ndarray:
    """Build the SOC contribution to the orbital-rotation gradient matrix (MO basis).

    The functional is assumed to be of the Frobenius form in the active space:

        F_SOC = Σ_m Σ_{p,q in act} h_m[p,q] * rho_m[p,q]

    with `rho_m` in the same (p,q) orientation as `h_m`.

    Returns
    -------
    g_mat
        Real antisymmetric matrix (nmo, nmo) suitable for `mc.pack_uniq_var`.
    """

    rho_m_act = np.asarray(rho_m_act)
    if rho_m_act.ndim != 3 or rho_m_act.shape[0] != 3 or rho_m_act.shape[1] != rho_m_act.shape[2]:
        raise ValueError("rho_m_act must have shape (3, ncas, ncas)")

    c = _as_mo_coeff(mc, mo_coeff)
    nmo = int(c.shape[1])
    act = _active_mo_slice(mc)
    ncas = int(getattr(mc, "ncas"))
    if rho_m_act.shape[1] != ncas:
        raise ValueError("rho_m_act ncas does not match mc.ncas")

    h_m_mo = soc_ao_to_mo(h_m_ao, c)  # (3, nmo, nmo)

    # For a Frobenius functional F = Σ_{pq} h[p,q] D[p,q] and real orbital rotations (κ^T=-κ),
    # the antisymmetric orbital gradient matrix is:
    #
    #   g = h^T D + h D^T - D h^T - D^T h
    #
    # This reduces to 2*(hD - (hD)^T) when D is symmetric.
    g = np.zeros((nmo, nmo), dtype=np.complex128)
    for m in range(3):
        h = h_m_mo[m]
        d = rho_m_act[m]
        dt = d.T
        g[:, act] += h.T[:, act] @ d
        g[:, act] += h[:, act] @ dt
        g[act, :] -= d @ h.T[act, :]
        g[act, :] -= dt @ h[act, :]
    if np.iscomplexobj(g):
        max_imag = float(np.max(np.abs(g.imag))) if g.size else 0.0
        if max_imag > float(imag_tol):
            raise ValueError(f"SOC orbital gradient has large imaginary part: max|Im|={max_imag:g} > {imag_tol:g}")
        g = g.real

    g = np.asarray(g, dtype=np.float64)
    # Enforce antisymmetry numerically.
    return 0.5 * (g - g.T)


def build_soc_orbital_rhs_for_zvector(
    mc: Any,
    *,
    rho_m_act: np.ndarray,
    h_m_ao: np.ndarray,
    mo_coeff: np.ndarray | None = None,
    imag_tol: float = 1e-10,
) -> np.ndarray:
    """Build the packed orbital RHS contribution from the SOC term."""

    g_mat = build_soc_orbital_gradient_matrix(
        mc, rho_m_act=rho_m_act, h_m_ao=h_m_ao, mo_coeff=mo_coeff, imag_tol=float(imag_tol)
    )
    return pack_orbital_gradient(mc, g_mat)


def solve_soc_ci_zvector_response(
    mc: Any,
    *,
    states: list[SpinFreeState],
    eta: np.ndarray,
    h_m: np.ndarray,
    h_m_ao: np.ndarray | None = None,
    block_nops: int = 8,
    eps: float = 0.0,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    use_newton_hessian: bool | None = True,
    rdm_weights: Sequence[float] | None = None,
) -> SOCCIZVectorResponse:
    """Solve the MCSCF Z-vector driven by the SOC CI RHS and return CI-response RDMs.

    This is a CI-first milestone helper. It does not yet include orbital-side RHS pieces or
    SOC integral-derivative contractions; it only propagates the SOC CI RHS through the
    existing MCSCF Z-vector machinery to obtain CI-response contributions to the active
    1- and 2-RDMs.
    """

    ci0 = getattr(mc, "ci", None)
    if ci0 is None:
        raise ValueError("mc.ci must be available")
    mode = _normalize_soc_backend(str(soc_backend))

    rhs_ci = build_soc_ci_rhs_for_zvector(
        ci0=ci0,
        states=states,
        eta=eta,
        h_m=h_m,
        block_nops=int(block_nops),
        eps=float(eps),
        soc_backend=str(mode),
        soc_cuda_threads=int(soc_cuda_threads),
        soc_cuda_sync=bool(soc_cuda_sync),
        soc_cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        imag_tol=float(imag_tol),
        project_normalized=bool(project_normalized),
    )

    rhs_orb = None
    rho_m = None
    if h_m_ao is not None:
        rho_m = build_rho_soc_m_streaming(
            states,
            eta,
            block_nops=int(block_nops),
            eps=float(eps),
            backend=str(mode),
            cuda_threads=int(soc_cuda_threads),
            cuda_sync=bool(soc_cuda_sync),
            cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        )
        rhs_orb = build_soc_orbital_rhs_for_zvector(
            mc,
            rho_m_act=rho_m,
            h_m_ao=np.asarray(h_m_ao),
            imag_tol=float(imag_tol),
        )

    z = solve_mcscf_zvector(
        mc,
        rhs_orb=rhs_orb,
        rhs_ci=rhs_ci,
        tol=float(z_tol),
        maxiter=int(z_maxiter),
        use_newton_hessian=use_newton_hessian,
    )

    norb = int(getattr(mc, "ncas"))
    nelec = getattr(mc, "nelecas")
    dm1_ci, dm2_ci = effective_active_rdms_from_ci_zvector(
        mc.fcisolver,
        ci0=ci0,
        z_ci=z.z_ci,
        norb=norb,
        nelec=nelec,
        weights=rdm_weights,
    )

    return SOCCIZVectorResponse(rhs_ci=rhs_ci, rhs_orb=rhs_orb, z=z, rho_m=rho_m, dm1_ci=dm1_ci, dm2_ci=dm2_ci)


def solve_soc_ci_zvector_response_multi_spin(
    mcs: Sequence[Any],
    soc_integrals: SOCIntegrals,
    *,
    so_root: int = 0,
    h_m_ao: np.ndarray | None = None,
    h_xyz_ao: np.ndarray | None = None,
    block_nops: int = 8,
    symmetrize: bool = True,
    eps: float = 0.0,
    soc_backend: Literal["cpu", "cuda", "auto"] = "auto",
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: Literal["auto", "apply_gemm", "direct_reduction"] = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
    imag_tol: float = 1e-10,
    project_normalized: bool = True,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    use_newton_hessian: bool | None = True,
) -> SOCMultiSpinZVectorResponse:
    """General (multi-spin) SOC-SI adjoint → per-manifold MCSCF Z-vector responses.

    This supports SOC state interaction between spin-free states of different total spin (S),
    assuming all states share the same active orbital basis used by `soc_integrals`.
    """

    mcs = list(mcs)
    if not mcs:
        raise ValueError("mcs is empty")
    if h_m_ao is not None and h_xyz_ao is not None:
        raise ValueError("provide at most one of h_m_ao or h_xyz_ao")
    if h_xyz_ao is not None:
        h_m_ao = soc_xyz_to_spherical(h_xyz_ao)
    mode = _normalize_soc_backend(str(soc_backend))
    gm_strategy = _normalize_soc_cuda_gm_strategy(str(soc_cuda_gm_strategy))
    gm_direct_max = int(soc_cuda_gm_direct_max_nb_nk)
    if gm_direct_max < 1:
        raise ValueError("soc_cuda_gm_direct_max_nb_nk must be >= 1")

    _validate_common_active_orbitals(mcs)

    # Build a concatenated state list and remember which states belong to which manifold.
    states: list[SpinFreeState] = []
    manifold_state_ids: list[list[int]] = []
    for mc in mcs:
        start = len(states)
        st = spinfree_states_from_mc(mc)
        states.extend(st)
        manifold_state_ids.append(list(range(start, start + len(st))))

    if not states:
        raise ValueError("no spin-free states found across manifolds")

    # SOC integrals in active MO basis.
    if not isinstance(soc_integrals, SOCIntegrals):
        raise TypeError("soc_integrals must be a asuka.soc.SOCIntegrals")
    h_m = soc_integrals.h_m
    if h_m is None:
        if soc_integrals.h_xyz is None:
            raise ValueError("soc_integrals must provide h_m or h_xyz")
        h_m = soc_xyz_to_spherical(soc_integrals.h_xyz)
    h_m = np.asarray(h_m, dtype=np.complex128)

    # SI diagonalization and adjoint weights.
    e_si, c_si, basis = soc_state_interaction(
        states,
        SOCIntegrals(h_m=h_m),
        include_diag=True,
        block_nops=int(block_nops),
        symmetrize=bool(symmetrize),
        backend=str(mode),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        cuda_gm_strategy=str(gm_strategy),
        cuda_gm_direct_max_nb_nk=int(gm_direct_max),
    )
    so_root = int(so_root)
    if so_root < 0 or so_root >= int(e_si.size):
        raise ValueError("so_root out of range for SI eigenvectors")
    w_state, eta = compute_si_adjoint_weights(states, basis, c_si[:, so_root])

    # Global SOC CI RHS and (optionally) global SOC density for the orbital RHS.
    rhs_all = build_ci_rhs_soc_full(
        states,
        eta,
        h_m,
        block_nops=int(block_nops),
        eps=float(eps),
        backend=str(mode),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
    )
    rho_m = None
    if h_m_ao is not None:
        rho_m = build_rho_soc_m_streaming(
            states,
            eta,
            block_nops=int(block_nops),
            eps=float(eps),
            backend=str(mode),
            cuda_threads=int(soc_cuda_threads),
            cuda_sync=bool(soc_cuda_sync),
            cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        )

    out_manifolds: list[SOCSpinManifoldZVectorResponse] = []
    for mc, state_ids in zip(mcs, manifold_state_ids):
        ci0 = getattr(mc, "ci", None)
        if ci0 is None:
            raise ValueError("each mc must provide mc.ci")

        # Build RHS CI structure matching mc.ci.
        if isinstance(ci0, (list, tuple)):
            rhs_ci_raw: Any = [rhs_all[i] for i in state_ids]
        else:
            if len(state_ids) != 1:
                raise ValueError("mc.ci is a single vector but multiple states were built from this mc")
            rhs_ci_raw = rhs_all[state_ids[0]]

        rhs_ci = prepare_ci_rhs_for_zvector(
            ci0=ci0, rhs_ci=rhs_ci_raw, imag_tol=float(imag_tol), project_normalized=bool(project_normalized)
        )

        rhs_orb = None
        if rho_m is not None and h_m_ao is not None:
            rhs_orb = build_soc_orbital_rhs_for_zvector(
                mc, rho_m_act=rho_m, h_m_ao=np.asarray(h_m_ao), imag_tol=float(imag_tol)
            )

        z = solve_mcscf_zvector(
            mc,
            rhs_orb=rhs_orb,
            rhs_ci=rhs_ci,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
            use_newton_hessian=use_newton_hessian,
        )

        norb = int(getattr(mc, "ncas"))
        nelec = getattr(mc, "nelecas")
        dm1_ci, dm2_ci = effective_active_rdms_from_ci_zvector(
            mc.fcisolver, ci0=ci0, z_ci=z.z_ci, norb=norb, nelec=nelec
        )

        response = SOCCIZVectorResponse(
            rhs_ci=rhs_ci,
            rhs_orb=rhs_orb,
            z=z,
            rho_m=rho_m,
            dm1_ci=dm1_ci,
            dm2_ci=dm2_ci,
        )

        twos = int(getattr(getattr(mc, "fcisolver", None), "twos", states[state_ids[0]].twos))
        out_manifolds.append(
            SOCSpinManifoldZVectorResponse(
                mc=mc,
                twos=twos,
                state_ids=state_ids,
                w_state=np.asarray(w_state[state_ids], dtype=np.float64),
                response=response,
            )
        )

    return SOCMultiSpinZVectorResponse(
        states=states,
        si_energies=e_si,
        si_vectors=c_si,
        si_basis=basis,
        w_state=w_state,
        eta=eta,
        rho_m=rho_m,
        manifolds=out_manifolds,
    )
