from __future__ import annotations

"""SCF front-end wrappers.

This module ties together:
- `frontend.one_electron` (S/T/V)
- `frontend.df` / cuERI DF (B[μ,ν,Q])
- `asuka.hf.df_scf` (SCF driver)

Notes
-----
- The `run_*_df` entrypoints build DF factors via cuERI (GPU).
- The `run_*_df_cpu` entrypoints build DF factors on CPU via cuERI-CPU (requires
  the CPU ERI extension to be built).
"""

from collections import OrderedDict
from dataclasses import dataclass, replace as _dc_replace
import os
from typing import Any, TYPE_CHECKING

import numpy as np

from asuka.hf.dense_eri import build_ao_eri_dense
from asuka.hf.dense_scf import rhf_dense, rohf_dense, uhf_dense
from asuka.hf.df_scf import SCFResult, rhf_df, rohf_df, uhf_df
from asuka.integrals.cueri_df import CuERIDFConfig, build_df_B_from_cueri_packed_bases
from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu
from asuka.integrals.int1e_cart import Int1eResult, build_int1e_cart

from .molecule import Molecule
from .one_electron import build_ao_basis_cart
from ._scf_build import (
    apply_sph_transform as _apply_sph_transform_impl,
    atom_coords_charges_bohr as _atom_coords_charges_bohr_impl,
    build_aux_basis_cart as _build_aux_basis_cart_impl,
    init_guess_dm_atom_hcore_cart as _init_guess_dm_atom_hcore_cart_impl,
    unique_elements as _unique_elements_impl,
)
from ._scf_cache import (
    cache_clear_all as _cache_clear_all,
    cache_get as _cache_get,
    cache_put as _cache_put,
    cuda_device_id_or_neg1 as _cuda_device_id_or_neg1,
    mol_cache_key as _mol_cache_key,
    normalize_basis_key as _normalize_basis_key,
)
from ._scf_config import (
    df_config_key as _df_config_key,
    resolve_cueri_df_config as _resolve_cueri_df_config,
)
from ._scf_df_build import (
    build_df_metric_cholesky as _build_df_metric_cholesky_impl,
    prepare_direct_df_inputs as _prepare_direct_df_inputs_impl,
)
from ._scf_dispatch import run_hf_df_dispatch as _run_hf_df_dispatch
from ._scf_methods import (
    run_rhf_df_impl as _run_rhf_df_impl,
    run_rhf_df_cpu_impl as _run_rhf_df_cpu_impl,
    run_rohf_df_cpu_impl as _run_rohf_df_cpu_impl,
    run_rohf_df_impl as _run_rohf_df_impl,
    run_rks_df_impl as _run_rks_df_impl,
    run_uhf_df_cpu_impl as _run_uhf_df_cpu_impl,
    run_uhf_df_impl as _run_uhf_df_impl,
    run_uks_df_impl as _run_uks_df_impl,
)

if TYPE_CHECKING:
    from asuka.integrals.cart2sph import AOSphericalTransform


class _SCFRunResultView:
    """ASUKA-facing convenience view over the embedded SCFResult.

    Callers should prefer these top-level properties over reaching through
    ``result.scf`` directly.
    """

    @property
    def method(self) -> str:
        return str(getattr(self.scf, "method"))

    @property
    def converged(self) -> bool:
        return bool(getattr(self.scf, "converged"))

    @property
    def niter(self) -> int:
        return int(getattr(self.scf, "niter"))

    @property
    def e_tot(self) -> float:
        return float(getattr(self.scf, "e_tot"))

    @property
    def e_elec(self) -> float:
        return float(getattr(self.scf, "e_elec"))

    @property
    def e_nuc(self) -> float:
        return float(getattr(self.scf, "e_nuc"))

    @property
    def mo_energy(self) -> Any:
        return getattr(self.scf, "mo_energy")

    @property
    def mo_coeff(self) -> Any:
        return getattr(self.scf, "mo_coeff")

    @property
    def mo_occ(self) -> Any:
        return getattr(self.scf, "mo_occ")

    @property
    def scf_result(self) -> SCFResult:
        return getattr(self, "scf")


@dataclass(frozen=True)
class THCRunConfig:
    """Reproducible THC frontend settings for native downstream workflows."""

    hf_method: str
    basis: Any | None
    auxbasis: Any
    df_config: Any | None
    expand_contractions: bool
    thc_mode: str
    thc_local_config: Any | None
    thc_grid_spec: Any | None
    thc_grid_kind: str
    thc_dvr_basis: Any | None
    thc_grid_options: Any | None
    thc_npt: int | None
    thc_solve_method: str
    use_density_difference: bool
    df_warmup_cycles: int
    df_warmup_ediff: float | None = None
    df_warmup_max_cycles: int | None = None
    df_aux_block_naux: int = 256
    df_k_q_block: int = 128
    max_cycle: int = 50
    conv_tol: float = 1e-10
    conv_tol_dm: float = 1e-8
    diis: bool = True
    diis_start_cycle: int | None = None
    diis_space: int = 8
    damping: float = 0.0
    level_shift: float = 0.0
    q_block: int = 256
    init_guess: str | None = None
    init_fock_cycles: int | None = None


@dataclass(frozen=True)
class DFRunConfig:
    """Reproducible DF frontend settings for native downstream workflows."""

    hf_method: str
    basis: Any | None
    auxbasis: Any
    df_config: Any | None
    expand_contractions: bool
    backend: str
    max_cycle: int = 50
    conv_tol: float = 1e-10
    conv_tol_dm: float = 1e-8
    diis: bool = True
    diis_start_cycle: int | None = None
    diis_space: int = 8
    damping: float = 0.0
    level_shift: float = 0.0
    k_q_block: int = 128
    cublas_math_mode: str | None = None
    init_fock_cycles: int | None = None


def _apply_sph_transform(
    mol: Molecule,
    int1e: Int1eResult,
    B,
    ao_basis,
    *,
    df_B_layout: str = "mnQ",
) -> tuple[Int1eResult, Any, AOSphericalTransform | None]:
    return _apply_sph_transform_impl(
        mol,
        int1e,
        B,
        ao_basis,
        df_B_layout=df_B_layout,
    )


@dataclass(frozen=True)
class RHFDFRunResult(_SCFRunResultView):
    mol: Molecule
    basis_name: str
    auxbasis_name: str
    ao_basis: Any
    aux_basis: Any
    int1e: Int1eResult
    df_B: Any
    scf: SCFResult
    profile: dict | None = None
    ao_eri: Any | None = None
    sph_map: AOSphericalTransform | tuple[np.ndarray, int, int] | None = None
    df_L: Any | None = None  # Cholesky factor of aux metric (for deterministic gradients)
    df_run_config: DFRunConfig | None = None
    thc_factors: Any | None = None
    thc_run_config: THCRunConfig | None = None
    two_e_backend: str | None = None
    direct_jk_ctx: Any | None = None
    cueri_shared_ctx: Any | None = None


@dataclass(frozen=True)
class UHFDFRunResult(_SCFRunResultView):
    mol: Molecule
    basis_name: str
    auxbasis_name: str
    ao_basis: Any
    aux_basis: Any
    int1e: Int1eResult
    df_B: Any
    scf: SCFResult
    profile: dict | None = None
    ao_eri: Any | None = None
    sph_map: AOSphericalTransform | tuple[np.ndarray, int, int] | None = None
    df_L: Any | None = None  # Cholesky factor of aux metric (for deterministic gradients)
    df_run_config: DFRunConfig | None = None
    thc_factors: Any | None = None
    thc_run_config: THCRunConfig | None = None
    two_e_backend: str | None = None
    direct_jk_ctx: Any | None = None
    cueri_shared_ctx: Any | None = None


@dataclass(frozen=True)
class ROHFDFRunResult(_SCFRunResultView):
    mol: Molecule
    basis_name: str
    auxbasis_name: str
    ao_basis: Any
    aux_basis: Any
    int1e: Int1eResult
    df_B: Any
    scf: SCFResult
    profile: dict | None = None
    ao_eri: Any | None = None
    sph_map: AOSphericalTransform | tuple[np.ndarray, int, int] | None = None
    df_L: Any | None = None  # Cholesky factor of aux metric (for deterministic gradients)
    df_run_config: DFRunConfig | None = None
    thc_factors: Any | None = None
    thc_run_config: THCRunConfig | None = None
    two_e_backend: str | None = None
    direct_jk_ctx: Any | None = None
    cueri_shared_ctx: Any | None = None


_HF_PREP_CACHE_MAX = max(0, int(os.environ.get("ASUKA_HF_PREP_CACHE_MAX", "0")))
_HF_GUESS_CACHE_MAX = max(0, int(os.environ.get("ASUKA_HF_GUESS_CACHE_MAX", "0")))
_HF_INIT_FOCK_CYCLES = max(0, int(os.environ.get("ASUKA_HF_INIT_FOCK_CYCLES", "1")))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


_HF_DENSE_MEM_BUDGET_GIB = _env_float("ASUKA_HF_DENSE_MEM_BUDGET_GIB", 8.0)
_RHF_PREP_CACHE: "OrderedDict[tuple[Any, ...], tuple[Any, str, Int1eResult, Any, str, Any]]" = OrderedDict()
_RHF_GUESS_CACHE: "OrderedDict[tuple[Any, ...], Any]" = OrderedDict()


def _make_thc_run_config(
    *,
    hf_method: str,
    basis: Any | None,
    auxbasis: Any,
    df_config: Any | None,
    expand_contractions: bool,
    thc_mode: str,
    thc_local_config: Any | None,
    thc_grid_spec: Any | None,
    thc_grid_kind: str,
    thc_dvr_basis: Any | None,
    thc_grid_options: Any | None,
    thc_npt: int | None,
    thc_solve_method: str,
    use_density_difference: bool,
    df_warmup_cycles: int,
    df_warmup_ediff: float | None = None,
    df_warmup_max_cycles: int | None = None,
    df_aux_block_naux: int = 256,
    df_k_q_block: int = 128,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    q_block: int = 256,
    init_guess: str | None = None,
    init_fock_cycles: int | None = None,
) -> THCRunConfig:
    return THCRunConfig(
        hf_method=str(hf_method),
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=bool(expand_contractions),
        thc_mode=str(thc_mode),
        thc_local_config=thc_local_config,
        thc_grid_spec=thc_grid_spec,
        thc_grid_kind=str(thc_grid_kind),
        thc_dvr_basis=thc_dvr_basis,
        thc_grid_options=thc_grid_options,
        thc_npt=None if thc_npt is None else int(thc_npt),
        thc_solve_method=str(thc_solve_method),
        use_density_difference=bool(use_density_difference),
        df_warmup_cycles=int(df_warmup_cycles),
        df_warmup_ediff=None if df_warmup_ediff is None else float(df_warmup_ediff),
        df_warmup_max_cycles=None if df_warmup_max_cycles is None else int(df_warmup_max_cycles),
        df_aux_block_naux=int(df_aux_block_naux),
        df_k_q_block=int(df_k_q_block),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=None if diis_start_cycle is None else int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        level_shift=float(level_shift),
        q_block=int(q_block),
        init_guess=None if init_guess is None else str(init_guess),
        init_fock_cycles=None if init_fock_cycles is None else int(init_fock_cycles),
    )


def _make_df_run_config(
    *,
    hf_method: str,
    basis: Any | None,
    auxbasis: Any,
    df_config: Any | None,
    expand_contractions: bool,
    backend: str,
    max_cycle: int,
    conv_tol: float,
    conv_tol_dm: float,
    diis: bool,
    diis_start_cycle: int | None,
    diis_space: int,
    damping: float,
    level_shift: float,
    k_q_block: int,
    cublas_math_mode: str | None = None,
    init_fock_cycles: int | None = None,
) -> DFRunConfig:
    return DFRunConfig(
        hf_method=str(hf_method),
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=bool(expand_contractions),
        backend=str(backend),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=None if diis_start_cycle is None else int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        level_shift=float(level_shift),
        k_q_block=int(k_q_block),
        cublas_math_mode=None if cublas_math_mode is None else str(cublas_math_mode),
        init_fock_cycles=None if init_fock_cycles is None else int(init_fock_cycles),
    )


def _build_df_metric_cholesky(
    aux_basis,
    *,
    df_config: CuERIDFConfig | None = None,
    profile: dict | None = None,
):
    return _build_df_metric_cholesky_impl(
        aux_basis,
        df_config=df_config,
        profile=profile,
    )


def _prepare_direct_df_inputs(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    df_config: CuERIDFConfig | None,
    df_backend: str | None = None,
    df_mode: str | None = None,
    df_threads: int | None = None,
    L_metric=None,
    profile: dict | None = None,
):
    return _prepare_direct_df_inputs_impl(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
        df_config=df_config,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        L_metric=L_metric,
        profile=profile,
    )


def _init_guess_dm_atom_hcore_cart(
    mol: Molecule,
    *,
    ao_basis,
    int1e_cart: Int1eResult,
) -> np.ndarray:
    return _init_guess_dm_atom_hcore_cart_impl(
        mol,
        ao_basis=ao_basis,
        int1e_cart=int1e_cart,
    )


def _rhf_prep_key(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    df_config: CuERIDFConfig | None,
    df_layout_build: str = "mnQ",
) -> tuple[Any, ...]:
    return (
        _mol_cache_key(mol),
        _normalize_basis_key(basis_in),
        _normalize_basis_key(auxbasis),
        bool(expand_contractions),
        _df_config_key(df_config),
        str(df_layout_build).strip().lower(),
    )


def _rhf_guess_key(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
) -> tuple[Any, ...]:
    return (
        "rhf",
        _mol_cache_key(mol),
        _normalize_basis_key(basis_in),
        _normalize_basis_key(auxbasis),
        bool(expand_contractions),
        int(_cuda_device_id_or_neg1()),
    )


def _copy_mo_coeff_for_cache(mo_coeff: Any):
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(mo_coeff, cp.ndarray):  # type: ignore[attr-defined]
        return cp.ascontiguousarray(cp.asarray(mo_coeff, dtype=cp.float64))
    return np.asarray(mo_coeff, dtype=np.float64, order="C").copy()


def clear_hf_frontend_caches() -> None:
    _cache_clear_all(_RHF_PREP_CACHE, _RHF_GUESS_CACHE)


def _atom_coords_charges_bohr(mol: Molecule) -> tuple[np.ndarray, np.ndarray]:
    return _atom_coords_charges_bohr_impl(mol)


def _unique_elements(mol: Molecule) -> list[str]:
    return _unique_elements_impl(mol)


def _build_aux_basis_cart(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    ao_basis: Any = None,
) -> tuple[Any, str]:
    return _build_aux_basis_cart_impl(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
        ao_basis=ao_basis,
    )


def _nalpha_nbeta_from_mol(mol: Molecule) -> tuple[int, int]:
    nelec = int(mol.nelectron)
    spin = int(mol.spin)
    if nelec <= 0:
        raise ValueError("nelectron must be positive")
    if (nelec + spin) % 2 != 0 or (nelec - spin) % 2 != 0:
        raise ValueError("incompatible nelectron/spin parity (requires nelec±spin even)")
    nalpha = (nelec + spin) // 2
    nbeta = (nelec - spin) // 2
    if nalpha < 0 or nbeta < 0:
        raise ValueError("invalid nelectron/spin combination (negative nalpha/nbeta)")
    return int(nalpha), int(nbeta)


def _dense_default_threads(backend: str) -> int:
    backend_s = str(backend).strip().lower()
    return 0 if backend_s == "cpu" else 256


def _build_dense_ao_eri(
    ao_basis,
    *,
    backend: str,
    dense_threads: int | None,
    dense_max_tile_bytes: int,
    dense_eps_ao: float,
    dense_max_l: int | None,
    dense_mem_budget_gib: float | None,
    profile: dict | None = None,
):
    backend_s = str(backend).strip().lower()
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")
    threads_i = _dense_default_threads(backend_s) if dense_threads is None else int(dense_threads)
    budget_gib = float(_HF_DENSE_MEM_BUDGET_GIB) if dense_mem_budget_gib is None else float(dense_mem_budget_gib)
    dense_prof = profile.setdefault("dense_eri_build", {}) if profile is not None else None
    return build_ao_eri_dense(
        ao_basis,
        backend=str(backend_s),
        threads=int(threads_i),
        max_tile_bytes=int(dense_max_tile_bytes),
        eps_ao=float(dense_eps_ao),
        max_l=dense_max_l,
        mem_budget_gib=float(budget_gib),
        profile=dense_prof,
    )


def run_rhf_dense(
    mol: Molecule,
    *,
    basis: Any | None = None,
    backend: str = "cuda",
    expand_contractions: bool = True,
    dense_threads: int | None = None,
    dense_max_tile_bytes: int = 256 << 20,
    dense_eps_ao: float = 0.0,
    dense_max_l: int | None = None,
    dense_mem_budget_gib: float | None = None,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    profile: dict | None = None,
) -> RHFDFRunResult:
    """Run RHF with dense AO ERIs (non-DF)."""

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_dense currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    dense = _build_dense_ao_eri(
        ao_basis,
        backend=str(backend),
        dense_threads=dense_threads,
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_eps_ao=float(dense_eps_ao),
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        profile=profile,
    )

    sph_map: AOSphericalTransform | None = None
    eri_mat_use = dense.eri_mat
    if not bool(mol.cart):
        int1e, _, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis)
        from asuka.integrals.cart2sph import transform_dense_eri_cart_to_sph  # noqa: PLC0415
        T, nao_cart, nao_sph = sph_map
        eri_mat_use = transform_dense_eri_cart_to_sph(dense.eri_mat, T, nao_cart, nao_sph)

    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf_prof = profile if profile is not None else None
    scf = rhf_dense(
        int1e.S,
        int1e.hcore,
        eri_mat_use,
        nelec=int(nelec),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle_i),
        diis_space=int(diis_space),
        damping=float(damping),
        level_shift=float(level_shift),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        profile=scf_prof,
    )

    return RHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<dense>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e,
        df_B=None,
        scf=scf,
        profile=profile,
        ao_eri=eri_mat_use,
        sph_map=sph_map,
    )


def run_rhf_direct(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
    eps_schwarz: float = 1e-12,
    direct_threads: int = 256,
    direct_max_tile_bytes: int = 256 << 20,
    direct_gpu_task_budget_bytes: int | None = None,
    direct_max_slab_tasks: int | None = None,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    profile: dict | None = None,
) -> RHFDFRunResult:
    """Run RHF with integral-direct 4-center J/K (no DF, no materialized ERI)."""

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_direct currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    if not bool(mol.cart):
        raise NotImplementedError("Integral-direct SCF does not yet support spherical AOs (cart=False)")

    from asuka.hf.direct_jk import make_direct_jk_context  # noqa: PLC0415
    from asuka.hf.direct_scf import rhf_direct  # noqa: PLC0415

    jk_ctx = make_direct_jk_context(
        ao_basis,
        eps_schwarz=float(eps_schwarz),
        threads=int(direct_threads),
        max_tile_bytes=int(direct_max_tile_bytes),
        max_slab_tasks=direct_max_slab_tasks,
        gpu_task_budget_bytes=direct_gpu_task_budget_bytes,
    )

    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf = rhf_direct(
        int1e.S,
        int1e.hcore,
        jk_ctx,
        nelec=int(nelec),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle_i),
        diis_space=int(diis_space),
        damping=float(damping),
        level_shift=float(level_shift),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        profile=profile,
    )

    return RHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<direct>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e,
        df_B=None,
        scf=scf,
        profile=profile,
        two_e_backend="direct",
        direct_jk_ctx=jk_ctx,
    )


def run_uhf_direct(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
    eps_schwarz: float = 1e-12,
    direct_threads: int = 256,
    direct_max_tile_bytes: int = 256 << 20,
    direct_gpu_task_budget_bytes: int | None = None,
    direct_max_slab_tasks: int | None = None,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with integral-direct 4-center J/K."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    if not bool(mol.cart):
        raise NotImplementedError("Integral-direct SCF does not yet support spherical AOs (cart=False)")

    from asuka.hf.direct_jk import make_direct_jk_context  # noqa: PLC0415
    from asuka.hf.direct_scf import uhf_direct  # noqa: PLC0415

    jk_ctx = make_direct_jk_context(
        ao_basis,
        eps_schwarz=float(eps_schwarz),
        threads=int(direct_threads),
        max_tile_bytes=int(direct_max_tile_bytes),
        max_slab_tasks=direct_max_slab_tasks,
        gpu_task_budget_bytes=direct_gpu_task_budget_bytes,
    )

    scf = uhf_direct(
        int1e.S,
        int1e.hcore,
        jk_ctx,
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
    )

    return UHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<direct>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e,
        df_B=None,
        scf=scf,
        profile=profile,
        two_e_backend="direct",
        direct_jk_ctx=jk_ctx,
    )


def run_rohf_direct(
    mol: Molecule,
    *,
    basis: Any | None = None,
    expand_contractions: bool = True,
    eps_schwarz: float = 1e-12,
    direct_threads: int = 256,
    direct_max_tile_bytes: int = 256 << 20,
    direct_gpu_task_budget_bytes: int | None = None,
    direct_max_slab_tasks: int | None = None,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with integral-direct 4-center J/K."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    if not bool(mol.cart):
        raise NotImplementedError("Integral-direct SCF does not yet support spherical AOs (cart=False)")

    from asuka.hf.direct_jk import make_direct_jk_context  # noqa: PLC0415
    from asuka.hf.direct_scf import rohf_direct  # noqa: PLC0415

    jk_ctx = make_direct_jk_context(
        ao_basis,
        eps_schwarz=float(eps_schwarz),
        threads=int(direct_threads),
        max_tile_bytes=int(direct_max_tile_bytes),
        max_slab_tasks=direct_max_slab_tasks,
        gpu_task_budget_bytes=direct_gpu_task_budget_bytes,
    )

    scf = rohf_direct(
        int1e.S,
        int1e.hcore,
        jk_ctx,
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
    )

    return ROHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<direct>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e,
        df_B=None,
        scf=scf,
        profile=profile,
        two_e_backend="direct",
        direct_jk_ctx=jk_ctx,
    )


def run_rhf_direct_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    df_backend: str | None = None,
    df_mode: str | None = None,
    df_threads: int | None = None,
    df_aux_block_naux: int = 256,
    L_metric=None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    profile: dict | None = None,
) -> RHFDFRunResult:
    """Run RHF with streamed DF J/K (direct SCF, no materialized DF tensor)."""

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_direct_df currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis

    if mo_coeff0 is None and dm0 is None:
        guess_key = _rhf_guess_key(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )
        guess_hit = _cache_get(_RHF_GUESS_CACHE, guess_key)
        if guess_hit is not None:
            mo_coeff0 = _copy_mo_coeff_for_cache(guess_hit)
            if profile is not None:
                profile.setdefault("scf_guess", {})["cache_hit"] = True
        elif profile is not None:
            profile.setdefault("scf_guess", {})["cache_hit"] = False

    cfg, ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, sph_map, df_ao_rep, L_chol = _prepare_direct_df_inputs(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_config=df_config,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        L_metric=L_metric,
        profile=profile,
    )

    from asuka.hf.direct_scf import rhf_direct_df as _rhf_direct_df_scf  # noqa: PLC0415

    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf = _rhf_direct_df_scf(
        int1e_scf.S,
        int1e_scf.hcore,
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        nelec=int(nelec),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle_i),
        diis_space=int(diis_space),
        damping=float(damping),
        level_shift=float(level_shift),
        k_q_block=int(k_q_block),
        cublas_math_mode=cublas_math_mode,
        df_backend=str(cfg.backend),
        df_ao_rep=str(df_ao_rep),
        df_threads=int(cfg.threads),
        df_mode=str(cfg.mode),
        df_aux_block_naux=int(df_aux_block_naux),
        L_metric=L_chol,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        profile=profile,
    )

    if bool(getattr(scf, "converged", False)):
        guess_key = _rhf_guess_key(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )
        _cache_put(
            _RHF_GUESS_CACHE,
            guess_key,
            _copy_mo_coeff_for_cache(scf.mo_coeff),
            max_size=int(_HF_GUESS_CACHE_MAX),
        )

    return RHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=_make_df_run_config(
            hf_method="rhf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            backend="cuda",
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=diis_start_cycle_i,
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=float(level_shift),
            k_q_block=int(k_q_block),
            cublas_math_mode=cublas_math_mode,
            init_fock_cycles=init_fock_cycles_i,
        ),
        two_e_backend="direct_df",
    )


def run_uhf_direct_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    df_backend: str | None = None,
    df_mode: str | None = None,
    df_threads: int | None = None,
    df_aux_block_naux: int = 256,
    L_metric=None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with streamed DF J/K (direct SCF, no materialized DF tensor)."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis

    cfg, ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, sph_map, df_ao_rep, L_chol = _prepare_direct_df_inputs(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_config=df_config,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        L_metric=L_metric,
        profile=profile,
    )

    from asuka.hf.direct_scf import uhf_direct_df as _uhf_direct_df_scf  # noqa: PLC0415

    scf = _uhf_direct_df_scf(
        int1e_scf.S,
        int1e_scf.hcore,
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        k_q_block=int(k_q_block),
        cublas_math_mode=cublas_math_mode,
        df_backend=str(cfg.backend),
        df_ao_rep=str(df_ao_rep),
        df_threads=int(cfg.threads),
        df_mode=str(cfg.mode),
        df_aux_block_naux=int(df_aux_block_naux),
        L_metric=L_chol,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
    )

    return UHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=_make_df_run_config(
            hf_method="uhf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            backend="cuda",
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle),
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=0.0,
            k_q_block=int(k_q_block),
            cublas_math_mode=cublas_math_mode,
        ),
        two_e_backend="direct_df",
    )


def run_rohf_direct_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    df_backend: str | None = None,
    df_mode: str | None = None,
    df_threads: int | None = None,
    df_aux_block_naux: int = 256,
    L_metric=None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with streamed DF J/K (direct SCF, no materialized DF tensor)."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_direct_df requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis

    cfg, ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, sph_map, df_ao_rep, L_chol = _prepare_direct_df_inputs(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_config=df_config,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        L_metric=L_metric,
        profile=profile,
    )

    from asuka.hf.direct_scf import rohf_direct_df as _rohf_direct_df_scf  # noqa: PLC0415

    scf = _rohf_direct_df_scf(
        int1e_scf.S,
        int1e_scf.hcore,
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        k_q_block=int(k_q_block),
        cublas_math_mode=cublas_math_mode,
        df_backend=str(cfg.backend),
        df_ao_rep=str(df_ao_rep),
        df_threads=int(cfg.threads),
        df_mode=str(cfg.mode),
        df_aux_block_naux=int(df_aux_block_naux),
        L_metric=L_chol,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
    )

    return ROHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=_make_df_run_config(
            hf_method="rohf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            backend="cuda",
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle),
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=0.0,
            k_q_block=int(k_q_block),
            cublas_math_mode=cublas_math_mode,
        ),
        two_e_backend="direct_df",
    )


def run_uhf_dense(
    mol: Molecule,
    *,
    basis: Any | None = None,
    backend: str = "cuda",
    expand_contractions: bool = True,
    dense_threads: int | None = None,
    dense_max_tile_bytes: int = 256 << 20,
    dense_eps_ao: float = 0.0,
    dense_max_l: int | None = None,
    dense_mem_budget_gib: float | None = None,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with dense AO ERIs (non-DF)."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    dense = _build_dense_ao_eri(
        ao_basis,
        backend=str(backend),
        dense_threads=dense_threads,
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_eps_ao=float(dense_eps_ao),
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        profile=profile,
    )

    sph_map: AOSphericalTransform | None = None
    eri_mat_use = dense.eri_mat
    if not bool(mol.cart):
        int1e, _, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis)
        from asuka.integrals.cart2sph import transform_dense_eri_cart_to_sph  # noqa: PLC0415
        T, nao_cart, nao_sph = sph_map
        eri_mat_use = transform_dense_eri_cart_to_sph(dense.eri_mat, T, nao_cart, nao_sph)

    scf_prof = profile if profile is not None else None
    scf = uhf_dense(
        int1e.S,
        int1e.hcore,
        eri_mat_use,
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return UHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<dense>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e,
        df_B=None,
        scf=scf,
        profile=profile,
        ao_eri=eri_mat_use,
        sph_map=sph_map,
    )


def run_rohf_dense(
    mol: Molecule,
    *,
    basis: Any | None = None,
    backend: str = "cuda",
    expand_contractions: bool = True,
    dense_threads: int | None = None,
    dense_max_tile_bytes: int = 256 << 20,
    dense_eps_ao: float = 0.0,
    dense_max_l: int | None = None,
    dense_mem_budget_gib: float | None = None,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with dense AO ERIs (non-DF)."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_dense requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    dense = _build_dense_ao_eri(
        ao_basis,
        backend=str(backend),
        dense_threads=dense_threads,
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_eps_ao=float(dense_eps_ao),
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        profile=profile,
    )

    sph_map: AOSphericalTransform | None = None
    eri_mat_use = dense.eri_mat
    if not bool(mol.cart):
        int1e, _, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis)
        from asuka.integrals.cart2sph import transform_dense_eri_cart_to_sph  # noqa: PLC0415
        T, nao_cart, nao_sph = sph_map
        eri_mat_use = transform_dense_eri_cart_to_sph(dense.eri_mat, T, nao_cart, nao_sph)

    scf_prof = profile if profile is not None else None
    scf = rohf_dense(
        int1e.S,
        int1e.hcore,
        eri_mat_use,
        nalpha=int(nalpha),
        nbeta=int(nbeta),
        enuc=float(mol.energy_nuc()),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return ROHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<dense>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e,
        df_B=None,
        scf=scf,
        profile=profile,
        ao_eri=eri_mat_use,
        sph_map=sph_map,
    )


def run_rhf_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    df_int3c_plan_policy: str | None = None,
    df_int3c_work_small_max: int | None = None,
    df_int3c_work_large_min: int | None = None,
    df_int3c_blocks_per_task: int | None = None,
    df_k_cache_max_mb: int | None = None,
    df_layout: str = "mnQ",
    expand_contractions: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    profile: dict | None = None,
) -> RHFDFRunResult:
    """Run RHF with DF integrals from cuERI.

    Notes
    -----
    - This path currently requires CUDA (cuERI DF builds `B` on GPU via CuPy).
    - RHF is restricted to closed-shell (mol.spin==0).
    """

    return _run_rhf_df_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
        df_k_cache_max_mb=df_k_cache_max_mb,
        df_layout=df_layout,
        expand_contractions=expand_contractions,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        level_shift=level_shift,
        k_q_block=k_q_block,
        cublas_math_mode=cublas_math_mode,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        profile=profile,
        init_fock_cycles_default=_HF_INIT_FOCK_CYCLES,
        resolve_cueri_df_config=_resolve_cueri_df_config,
        rhf_prep_key=_rhf_prep_key,
        rhf_guess_key=_rhf_guess_key,
        cache_get=_cache_get,
        cache_put=_cache_put,
        rhf_prep_cache=_RHF_PREP_CACHE,
        rhf_guess_cache=_RHF_GUESS_CACHE,
        hf_prep_cache_max=_HF_PREP_CACHE_MAX,
        hf_guess_cache_max=_HF_GUESS_CACHE_MAX,
        copy_mo_coeff_for_cache=_copy_mo_coeff_for_cache,
        make_df_run_config=_make_df_run_config,
        result_cls=RHFDFRunResult,
    )


def run_rhf_thc(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    # THC build
    thc_mode: str = "global",
    thc_local_config: Any | None = None,
    thc_grid_spec: Any | None = None,
    thc_grid_kind: str = "rdvr",
    thc_dvr_basis: Any | None = None,
    thc_grid_options: Any | None = None,
    thc_npt: int | None = None,
    thc_solve_method: str = "fit_metric_qr",
    thc_mp_mode: str = "fp64",
    thc_store_Z: bool | None = None,
    thc_rebase_dD_rel_tol: float = 0.25,
    thc_rebase_min_cycle: int = 2,
    thc_tc_balance: bool = True,
    # Density-difference reference
    use_density_difference: bool = True,
    df_warmup_cycles: int = 2,
    df_warmup_ediff: float | None = None,
    df_warmup_max_cycles: int = 25,
    df_aux_block_naux: int = 256,
    df_k_q_block: int = 128,
    # SCF solve
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    q_block: int = 256,
    init_guess: str = "auto",
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    profile: dict | None = None,
    # DFT (optional)
    functional: str | None = None,
    grid_radial_n: int = 75,
    grid_angular_n: int = 590,
    xc_batch_size: int = 50000,
) -> RHFDFRunResult:
    """Run RHF (or RKS-DFT if functional is given) with THC J/K.

    Notes
    -----
    - THC factor construction requires CUDA (CuPy + `asuka._orbitals_cuda_ext`).
    - Density-difference warmup uses streamed DF (no materialized B tensor).
    """

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_thc supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    cfg = CuERIDFConfig() if df_config is None else df_config

    # AO basis + 1e integrals (cart)
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    # Aux basis (cart)
    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )

    # SCF AO representation: cart or sph
    int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout="mnQ")
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

    # DFT setup (optional)
    xc_spec = None
    xc_grid_coords = None
    xc_grid_weights = None
    xc_sph_transform = None
    if functional is not None:
        from asuka.xc.functional import get_functional
        from asuka.density.grids_device import make_becke_grid_device

        xc_spec = get_functional(functional)
        xc_grid_coords, xc_grid_weights = make_becke_grid_device(
            mol, radial_n=int(grid_radial_n), angular_n=int(grid_angular_n),
            radial_scheme="treutler",
        )
        if not bool(mol.cart) and sph_map is not None:
            import cupy as _cp_xc
            if hasattr(sph_map, "T_c2s"):
                xc_sph_transform = _cp_xc.asarray(sph_map.T_c2s, dtype=_cp_xc.float64)
            elif hasattr(sph_map, "T_matrix"):
                xc_sph_transform = _cp_xc.asarray(sph_map.T_matrix, dtype=_cp_xc.float64)
            elif isinstance(sph_map, tuple) and len(sph_map) >= 1:
                xc_sph_transform = _cp_xc.asarray(sph_map[0], dtype=_cp_xc.float64)

    # Optional guess reuse (RHF cache)
    init_guess_s = str(init_guess).strip().lower()
    if mo_coeff0 is None and dm0 is None and init_guess_s in {"auto", "cache", "cache_or_core"}:
        guess_key = _rhf_guess_key(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )
        guess_hit = _cache_get(_RHF_GUESS_CACHE, guess_key)
        if guess_hit is not None:
            mo_coeff0 = _copy_mo_coeff_for_cache(guess_hit)
            if profile is not None:
                profile.setdefault("scf_guess", {})["cache_hit"] = True
        elif profile is not None:
            profile.setdefault("scf_guess", {})["cache_hit"] = False

    # SAD-like initial guess for density-difference runs (paper-faithful default).
    #
    # This uses an atom-block hcore diagonalization (no 2e integrals) to produce
    # a localized density guess.  It is intentionally optional so callers can
    # still provide dm0/mo_coeff0 or rely on cached guesses.
    if dm0 is None and mo_coeff0 is None and init_guess_s in {"sad", "atom", "atom_hcore", "atom-hcore"}:
        try:
            D_cart = _init_guess_dm_atom_hcore_cart(mol, ao_basis=ao_basis, int1e_cart=int1e)
            if bool(mol.cart):
                dm0 = D_cart
            else:
                if sph_map is None:
                    raise RuntimeError("expected sph_map for mol.cart=False")
                T_c2s = np.asarray(sph_map.T_c2s, dtype=np.float64)
                dm0 = T_c2s.T @ D_cart @ T_c2s
                dm0 = 0.5 * (dm0 + dm0.T)
            if profile is not None:
                profile.setdefault("scf_guess", {})["init_guess"] = "atom_hcore"
        except Exception:
            # Best-effort: fall back to the default core-H guess if something
            # goes wrong (unexpected basis centers, etc).
            if profile is not None:
                profile.setdefault("scf_guess", {})["init_guess"] = "core_fallback"

    # Grid kind selection (Becke vs DVR).
    grid_kind_s = str(thc_grid_kind).strip().lower()
    dvr_basis_cart = None
    if grid_kind_s in {"rdvr", "r-dvr", "r_dvr", "fdvr", "f-dvr", "f_dvr"}:
        if thc_dvr_basis is None:
            dvr_basis_cart = aux_basis
        elif isinstance(thc_dvr_basis, str) and str(thc_dvr_basis).strip().lower() in {"aux", "auxbasis"}:
            dvr_basis_cart = aux_basis
        else:
            dvr_basis_cart, _dvr_name_unused = build_ao_basis_cart(
                mol,
                basis=thc_dvr_basis,
                expand_contractions=bool(expand_contractions),
            )

    # Build THC factors (global THC or local-THC/LS-THC-style blocks).
    thc_mode_s = str(thc_mode).strip().lower()
    thc_prof = profile.setdefault("thc_build", {}) if profile is not None else None

    thc_mp_mode_s = str(thc_mp_mode).strip().lower()
    if thc_mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("thc_mp_mode must be 'fp64' or 'tf32'")
    if thc_store_Z is None:
        store_Z_eff = bool(thc_mp_mode_s != "tf32")
    else:
        store_Z_eff = bool(thc_store_Z)
    if thc_prof is not None:
        thc_prof.setdefault("mp_mode", str(thc_mp_mode_s))
        thc_prof.setdefault("store_Z", bool(store_Z_eff))

    thc_factors = None
    L_metric_full = None

    if thc_mode_s in {"global", "thc", "full"}:
        from asuka.hf.thc_factors import build_thc_factors  # noqa: PLC0415

        thc = build_thc_factors(
            mol,
            ao_basis,
            aux_basis,
            sph_map=sph_map,
            grid_spec=thc_grid_spec,
            grid_kind=str(thc_grid_kind),
            dvr_basis=dvr_basis_cart,
            grid_options=thc_grid_options,
            npt=thc_npt,
            metric_backend=str(cfg.backend),
            metric_mode=str(cfg.mode),
            metric_threads=int(cfg.threads),
            solve_method=str(thc_solve_method),
            store_Z=bool(store_Z_eff),
            stream=cfg.stream,
            profile=thc_prof,
        )
        thc_factors = thc
        L_metric_full = thc.L_metric
    elif thc_mode_s in {"local", "lthc", "local-thc", "local_thc", "ls-thc", "ls_thc"}:
        from asuka.hf.local_thc_config import LocalTHCConfig  # noqa: PLC0415
        from asuka.hf.local_thc_factors import build_local_thc_factors  # noqa: PLC0415

        if thc_local_config is None:
            lcfg = LocalTHCConfig()
        elif isinstance(thc_local_config, LocalTHCConfig):
            lcfg = thc_local_config
        elif isinstance(thc_local_config, dict):
            lcfg = LocalTHCConfig(**thc_local_config)
        else:
            raise TypeError("thc_local_config must be a LocalTHCConfig or a dict")

        if bool(use_density_difference):
            from asuka.cueri import df as cueri_df  # noqa: PLC0415

            V_full = cueri_df.metric_2c2e_basis(
                aux_basis,
                stream=cfg.stream,
                backend=str(cfg.backend),
                mode=str(cfg.mode),
                threads=int(cfg.threads),
            )
            L_metric_full = cueri_df.cholesky_metric(V_full)

        lthc = build_local_thc_factors(
            mol,
            ao_basis,
            aux_basis,
            S_scf=int1e_scf.S,
            grid_kind=str(thc_grid_kind),
            dvr_basis=dvr_basis_cart,
            grid_options=thc_grid_options,
            grid_spec=thc_grid_spec,
            thc_npt=thc_npt,
            config=lcfg,
            metric_backend=str(cfg.backend),
            metric_mode=str(cfg.mode),
            metric_threads=int(cfg.threads),
            solve_method=str(thc_solve_method),
            store_Z=bool(store_Z_eff),
            stream=cfg.stream,
            profile=thc_prof,
        )
        thc_factors = lthc
    else:
        raise ValueError("thc_mode must be one of: 'global', 'local'")

    # Optional density-difference reference via streamed DF warmup
    ref = None
    mo_coeff_thc0 = mo_coeff0
    if bool(use_density_difference):
        from asuka.hf import df_jk_streamed  # noqa: PLC0415
        from asuka.hf import df_scf as hf_df_scf  # noqa: PLC0415
        from asuka.hf.local_thc_scf import LocalTHCReferenceRHF  # noqa: PLC0415
        from asuka.hf.thc_scf import THCReferenceRHF  # noqa: PLC0415

        warm_ediff = None if df_warmup_ediff is None else float(df_warmup_ediff)
        if warm_ediff is not None and (not np.isfinite(warm_ediff) or warm_ediff <= 0.0):
            raise ValueError("df_warmup_ediff must be a finite, positive float or None")
        warm_max = max(1, int(df_warmup_max_cycles))
        n_warm = max(1, int(df_warmup_cycles)) if warm_ediff is None else int(warm_max)
        warm_prof = profile.setdefault("df_warmup", {}) if profile is not None else None
        warm = hf_df_scf.rhf_df(
            int1e_scf.S,
            int1e_scf.hcore,
            None,
            nelec=int(nelec),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(n_warm),
            conv_tol=0.0 if warm_ediff is None else float(warm_ediff),
            conv_tol_dm=0.0 if warm_ediff is None else 1e9,
            # Paper-faithful: these are "conventional iterations" to refine the
            # density before switching to THC on ΔD, so use DIIS.
            diis=True,
            damping=0.0,
            level_shift=0.0,
            jk_mode="streamed",
            k_engine="from_Cocc",
            k_q_block=int(df_k_q_block),
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            df_backend=str(cfg.backend),
            df_ao_rep=str(df_ao_rep),
            df_threads=int(cfg.threads),
            df_mode=str(cfg.mode),
            df_aux_block_naux=int(df_aux_block_naux),
            L_metric=L_metric_full,
            dm0=dm0,
            mo_coeff0=mo_coeff0,
            init_fock_cycles=0,
            profile=warm_prof,
        )
        C_ref = warm.mo_coeff
        occ_ref = warm.mo_occ

        xp, _ = hf_df_scf._get_xp(C_ref, occ_ref)
        D_ref = hf_df_scf._symmetrize(xp, hf_df_scf._density_from_C_occ(C_ref, occ_ref))

        # Build accurate J/K once at D_ref (same streamed DF context as warmup).
        ctx = df_jk_streamed.make_streamed_df_jk_context(
            ao_basis,
            aux_basis,
            L_metric=L_metric_full,
            backend=str(cfg.backend),
            threads=int(cfg.threads),
            mode=str(cfg.mode),
            ao_rep=str(df_ao_rep),
            aux_block_naux=int(df_aux_block_naux),
            profile=None,
        )
        nocc = int(nelec // 2)
        C_occ = xp.ascontiguousarray(C_ref[:, :nocc])
        occ_vals = occ_ref[:nocc]
        J_ref, K_ref = df_jk_streamed.df_JK_streamed(
            ctx,
            D_ref,
            C_occ,
            occ_vals,
            k_q_block=int(df_k_q_block),
            cublas_math_mode=None,
            work={},
            profile=None,
        )
        if thc_mode_s in {"global", "thc", "full"}:
            ref = THCReferenceRHF(D_ref=D_ref, J_ref=J_ref, K_ref=K_ref)
        else:
            ref = LocalTHCReferenceRHF(D_ref=D_ref, J_ref=J_ref, K_ref=K_ref)

        if mo_coeff_thc0 is None:
            mo_coeff_thc0 = C_ref

    # If we performed a warmup/reference build, start the THC iterations from
    # the refined MO coefficients rather than the original dm0 guess.
    dm0_thc = None if bool(use_density_difference) else dm0

    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf_prof = profile if profile is not None else None
    if thc_mode_s in {"global", "thc", "full"}:
        # THC-SCF solve
        from asuka.hf.thc_scf import rhf_thc  # noqa: PLC0415

        scf = rhf_thc(
            int1e_scf.S,
            int1e_scf.hcore,
            thc_factors,
            nelec=int(nelec),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=float(level_shift),
            q_block=int(q_block),
            mp_mode=str(thc_mp_mode_s),
            rebase_dD_rel_tol=float(thc_rebase_dD_rel_tol),
            rebase_min_cycle=int(thc_rebase_min_cycle),
            tc_balance=bool(thc_tc_balance),
            dm0=dm0_thc,
            mo_coeff0=mo_coeff_thc0,
            init_fock_cycles=int(init_fock_cycles_i),
            reference=ref,
            profile=scf_prof,
            xc_spec=xc_spec,
            xc_grid_coords=xc_grid_coords,
            xc_grid_weights=xc_grid_weights,
            xc_ao_basis=ao_basis,
            xc_sph_transform=xc_sph_transform,
            xc_batch_size=int(xc_batch_size),
        )
    else:
        from asuka.hf.local_thc_scf import rhf_local_thc  # noqa: PLC0415

        scf = rhf_local_thc(
            int1e_scf.S,
            int1e_scf.hcore,
            thc_factors,
            nelec=int(nelec),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=float(level_shift),
            q_block=int(q_block),
            mp_mode=str(thc_mp_mode_s),
            rebase_dD_rel_tol=float(thc_rebase_dD_rel_tol),
            rebase_min_cycle=int(thc_rebase_min_cycle),
            tc_balance=bool(thc_tc_balance),
            dm0=dm0_thc,
            mo_coeff0=mo_coeff_thc0,
            init_fock_cycles=int(init_fock_cycles_i),
            reference=ref,
            profile=scf_prof,
            xc_spec=xc_spec,
            xc_grid_coords=xc_grid_coords,
            xc_grid_weights=xc_grid_weights,
            xc_ao_basis=ao_basis,
            xc_sph_transform=xc_sph_transform,
            xc_batch_size=int(xc_batch_size),
        )

    if bool(getattr(scf, "converged", False)):
        guess_key = _rhf_guess_key(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )
        _cache_put(
            _RHF_GUESS_CACHE,
            guess_key,
            _copy_mo_coeff_for_cache(scf.mo_coeff),
            max_size=int(_HF_GUESS_CACHE_MAX),
        )

    return RHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_metric_full,
        thc_factors=thc_factors,
        thc_run_config=_make_thc_run_config(
            hf_method="rhf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            thc_mode=str(thc_mode),
            thc_local_config=thc_local_config,
            thc_grid_spec=thc_grid_spec,
            thc_grid_kind=str(thc_grid_kind),
            thc_dvr_basis=thc_dvr_basis,
            thc_grid_options=thc_grid_options,
            thc_npt=thc_npt,
            thc_solve_method=str(thc_solve_method),
            use_density_difference=bool(use_density_difference),
            df_warmup_cycles=int(df_warmup_cycles),
            df_warmup_ediff=df_warmup_ediff,
            df_warmup_max_cycles=int(df_warmup_max_cycles),
            df_aux_block_naux=int(df_aux_block_naux),
            df_k_q_block=int(df_k_q_block),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=diis_start_cycle_i,
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=float(level_shift),
            q_block=int(q_block),
            init_guess=str(init_guess),
            init_fock_cycles=init_fock_cycles_i,
        ),
    )


def run_uhf_thc(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    # THC build
    thc_grid_spec: Any | None = None,
    thc_grid_kind: str = "rdvr",
    thc_dvr_basis: Any | None = None,
    thc_grid_options: Any | None = None,
    thc_mode: str = "global",
    thc_local_config: Any | None = None,
    thc_npt: int | None = None,
    thc_solve_method: str = "fit_metric_qr",
    thc_mp_mode: str = "fp64",
    thc_store_Z: bool | None = None,
    thc_rebase_dD_rel_tol: float = 0.25,
    thc_rebase_min_cycle: int = 2,
    thc_tc_balance: bool = True,
    # Density-difference reference
    use_density_difference: bool = True,
    df_warmup_cycles: int = 2,
    df_aux_block_naux: int = 256,
    df_k_q_block: int = 128,
    # SCF solve
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    q_block: int = 256,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
    # DFT (optional)
    functional: str | None = None,
    grid_radial_n: int = 75,
    grid_angular_n: int = 590,
    xc_batch_size: int = 50000,
) -> UHFDFRunResult:
    """Run UHF (or UKS-DFT if ``functional`` is given) with THC J/K."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis
    cfg = CuERIDFConfig() if df_config is None else df_config

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )

    int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout="mnQ")
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

    # DFT setup (optional)
    xc_spec = None
    xc_grid_coords = None
    xc_grid_weights = None
    xc_sph_transform = None
    if functional is not None:
        from asuka.xc.functional import get_functional
        from asuka.density.grids_device import make_becke_grid_device

        xc_spec = get_functional(functional)
        xc_grid_coords, xc_grid_weights = make_becke_grid_device(
            mol, radial_n=int(grid_radial_n), angular_n=int(grid_angular_n),
            radial_scheme="treutler",
        )
        if not bool(mol.cart) and sph_map is not None:
            import cupy as _cp_xc  # noqa: PLC0415
            if hasattr(sph_map, "T_c2s"):
                xc_sph_transform = _cp_xc.asarray(sph_map.T_c2s, dtype=_cp_xc.float64)
            elif hasattr(sph_map, "T_matrix"):
                xc_sph_transform = _cp_xc.asarray(sph_map.T_matrix, dtype=_cp_xc.float64)
            elif isinstance(sph_map, tuple) and len(sph_map) >= 1:
                xc_sph_transform = _cp_xc.asarray(sph_map[0], dtype=_cp_xc.float64)

    # Grid kind selection (Becke vs DVR).
    grid_kind_s = str(thc_grid_kind).strip().lower()
    dvr_basis_cart = None
    if grid_kind_s in {"rdvr", "r-dvr", "r_dvr", "fdvr", "f-dvr", "f_dvr"}:
        if thc_dvr_basis is None:
            dvr_basis_cart = aux_basis
        elif isinstance(thc_dvr_basis, str) and str(thc_dvr_basis).strip().lower() in {"aux", "auxbasis"}:
            dvr_basis_cart = aux_basis
        else:
            dvr_basis_cart, _dvr_name_unused = build_ao_basis_cart(
                mol,
                basis=thc_dvr_basis,
                expand_contractions=bool(expand_contractions),
            )

    # Build THC factors (global THC or local-THC/LS-THC-style blocks).
    thc_mode_s = str(thc_mode).strip().lower()
    thc_prof = profile.setdefault("thc_build", {}) if profile is not None else None

    thc_mp_mode_s = str(thc_mp_mode).strip().lower()
    if thc_mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("thc_mp_mode must be 'fp64' or 'tf32'")
    if thc_store_Z is None:
        store_Z_eff = bool(thc_mp_mode_s != "tf32")
    else:
        store_Z_eff = bool(thc_store_Z)
    if thc_prof is not None:
        thc_prof.setdefault("mp_mode", str(thc_mp_mode_s))
        thc_prof.setdefault("store_Z", bool(store_Z_eff))

    thc_factors = None
    L_metric_full = None

    if thc_mode_s in {"global", "thc", "full"}:
        from asuka.hf.thc_factors import build_thc_factors  # noqa: PLC0415

        thc = build_thc_factors(
            mol,
            ao_basis,
            aux_basis,
            sph_map=sph_map,
            grid_spec=thc_grid_spec,
            grid_kind=str(thc_grid_kind),
            dvr_basis=dvr_basis_cart,
            grid_options=thc_grid_options,
            npt=thc_npt,
            metric_backend=str(cfg.backend),
            metric_mode=str(cfg.mode),
            metric_threads=int(cfg.threads),
            solve_method=str(thc_solve_method),
            store_Z=bool(store_Z_eff),
            stream=cfg.stream,
            profile=thc_prof,
        )
        thc_factors = thc
        L_metric_full = thc.L_metric
    elif thc_mode_s in {"local", "lthc", "local-thc", "local_thc", "ls-thc", "ls_thc"}:
        from asuka.hf.local_thc_config import LocalTHCConfig  # noqa: PLC0415
        from asuka.hf.local_thc_factors import build_local_thc_factors  # noqa: PLC0415

        if thc_local_config is None:
            lcfg = LocalTHCConfig()
        elif isinstance(thc_local_config, LocalTHCConfig):
            lcfg = thc_local_config
        elif isinstance(thc_local_config, dict):
            lcfg = LocalTHCConfig(**thc_local_config)
        else:
            raise TypeError("thc_local_config must be a LocalTHCConfig or a dict")

        if bool(use_density_difference):
            from asuka.cueri import df as cueri_df  # noqa: PLC0415

            V_full = cueri_df.metric_2c2e_basis(
                aux_basis,
                stream=cfg.stream,
                backend=str(cfg.backend),
                mode=str(cfg.mode),
                threads=int(cfg.threads),
            )
            L_metric_full = cueri_df.cholesky_metric(V_full)

        lthc = build_local_thc_factors(
            mol,
            ao_basis,
            aux_basis,
            S_scf=int1e_scf.S,
            grid_kind=str(thc_grid_kind),
            dvr_basis=dvr_basis_cart,
            grid_options=thc_grid_options,
            grid_spec=thc_grid_spec,
            thc_npt=thc_npt,
            config=lcfg,
            metric_backend=str(cfg.backend),
            metric_mode=str(cfg.mode),
            metric_threads=int(cfg.threads),
            solve_method=str(thc_solve_method),
            store_Z=bool(store_Z_eff),
            stream=cfg.stream,
            profile=thc_prof,
        )
        thc_factors = lthc
    else:
        raise ValueError("thc_mode must be one of: 'global', 'local'")

    ref = None
    mo_coeff_thc0 = mo_coeff0
    if bool(use_density_difference):
        if dm0 is not None:
            raise NotImplementedError("density-difference warmup currently does not support dm0 (streamed DF requires mo_coeff)")

        from asuka.hf import df_jk_streamed  # noqa: PLC0415
        from asuka.hf import df_scf as hf_df_scf  # noqa: PLC0415
        from asuka.hf.local_thc_scf import LocalTHCReferenceUHF  # noqa: PLC0415
        from asuka.hf.thc_scf import THCReferenceUHF  # noqa: PLC0415

        n_warm = max(1, int(df_warmup_cycles))
        warm_prof = profile.setdefault("df_warmup", {}) if profile is not None else None
        warm = hf_df_scf.uhf_df(
            int1e_scf.S,
            int1e_scf.hcore,
            None,
            nalpha=int(nalpha),
            nbeta=int(nbeta),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(n_warm),
            conv_tol=0.0,
            conv_tol_dm=0.0,
            # Conventional refinement iterations before switching to THC on ΔD.
            diis=True,
            damping=0.0,
            jk_mode="streamed",
            k_engine="from_Cocc",
            k_q_block=int(df_k_q_block),
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            df_backend=str(cfg.backend),
            df_ao_rep=str(df_ao_rep),
            df_threads=int(cfg.threads),
            df_mode=str(cfg.mode),
            df_aux_block_naux=int(df_aux_block_naux),
            L_metric=L_metric_full,
            mo_coeff0=mo_coeff0,
            profile=warm_prof,
            xc_spec=xc_spec,
            xc_grid_coords=xc_grid_coords,
            xc_grid_weights=xc_grid_weights,
            xc_ao_basis=ao_basis,
            xc_sph_transform=xc_sph_transform,
            xc_batch_size=int(xc_batch_size),
        )
        Ca_ref, Cb_ref = warm.mo_coeff
        occ_a_ref, occ_b_ref = warm.mo_occ

        xp, _ = hf_df_scf._get_xp(Ca_ref, occ_a_ref)
        Da_ref = hf_df_scf._symmetrize(xp, hf_df_scf._density_from_C_occ(Ca_ref, occ_a_ref))
        Db_ref = hf_df_scf._symmetrize(xp, hf_df_scf._density_from_C_occ(Cb_ref, occ_b_ref))
        Dtot_ref = Da_ref + Db_ref

        ctx = df_jk_streamed.make_streamed_df_jk_context(
            ao_basis,
            aux_basis,
            L_metric=L_metric_full,
            backend=str(cfg.backend),
            threads=int(cfg.threads),
            mode=str(cfg.mode),
            ao_rep=str(df_ao_rep),
            aux_block_naux=int(df_aux_block_naux),
            profile=None,
        )
        Ca_occ = xp.ascontiguousarray(Ca_ref[:, : int(nalpha)])
        Cb_occ = xp.ascontiguousarray(Cb_ref[:, : int(nbeta)])
        occ_a_vals = occ_a_ref[: int(nalpha)]
        occ_b_vals = occ_b_ref[: int(nbeta)]
        J_ref, Ks = df_jk_streamed.df_JKs_streamed(
            ctx,
            Dtot_ref,
            [Ca_occ, Cb_occ],
            [occ_a_vals, occ_b_vals],
            k_q_block=int(df_k_q_block),
            cublas_math_mode=None,
            work={},
            profile=None,
        )
        Ka_ref, Kb_ref = Ks
        if thc_mode_s in {"global", "thc", "full"}:
            ref = THCReferenceUHF(Da_ref=Da_ref, Db_ref=Db_ref, J_ref=J_ref, Ka_ref=Ka_ref, Kb_ref=Kb_ref)
        else:
            ref = LocalTHCReferenceUHF(Da_ref=Da_ref, Db_ref=Db_ref, J_ref=J_ref, Ka_ref=Ka_ref, Kb_ref=Kb_ref)

        if mo_coeff_thc0 is None:
            mo_coeff_thc0 = (Ca_ref, Cb_ref)

    diis_start_cycle_i = int(diis_start_cycle) if diis_start_cycle is not None else 2
    if thc_mode_s in {"global", "thc", "full"}:
        from asuka.hf.thc_scf import uhf_thc  # noqa: PLC0415

        scf = uhf_thc(
            int1e_scf.S,
            int1e_scf.hcore,
            thc_factors,
            nalpha=int(nalpha),
            nbeta=int(nbeta),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            q_block=int(q_block),
            mp_mode=str(thc_mp_mode_s),
            rebase_dD_rel_tol=float(thc_rebase_dD_rel_tol),
            rebase_min_cycle=int(thc_rebase_min_cycle),
            tc_balance=bool(thc_tc_balance),
            dm0=dm0,
            mo_coeff0=mo_coeff_thc0,
            reference=ref,
            profile=profile,
            xc_spec=xc_spec,
            xc_grid_coords=xc_grid_coords,
            xc_grid_weights=xc_grid_weights,
            xc_ao_basis=ao_basis,
            xc_sph_transform=xc_sph_transform,
            xc_batch_size=int(xc_batch_size),
        )
    else:
        from asuka.hf.local_thc_scf import uhf_local_thc  # noqa: PLC0415

        scf = uhf_local_thc(
            int1e_scf.S,
            int1e_scf.hcore,
            thc_factors,
            nalpha=int(nalpha),
            nbeta=int(nbeta),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            q_block=int(q_block),
            mp_mode=str(thc_mp_mode_s),
            rebase_dD_rel_tol=float(thc_rebase_dD_rel_tol),
            rebase_min_cycle=int(thc_rebase_min_cycle),
            tc_balance=bool(thc_tc_balance),
            dm0=dm0,
            mo_coeff0=mo_coeff_thc0,
            reference=ref,
            profile=profile,
            xc_spec=xc_spec,
            xc_grid_coords=xc_grid_coords,
            xc_grid_weights=xc_grid_weights,
            xc_ao_basis=ao_basis,
            xc_sph_transform=xc_sph_transform,
            xc_batch_size=int(xc_batch_size),
        )

    return UHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_metric_full,
        thc_factors=thc_factors,
        thc_run_config=_make_thc_run_config(
            hf_method="uks" if functional is not None else "uhf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            thc_mode=str(thc_mode),
            thc_local_config=thc_local_config,
            thc_grid_spec=thc_grid_spec,
            thc_grid_kind=str(thc_grid_kind),
            thc_dvr_basis=thc_dvr_basis,
            thc_grid_options=thc_grid_options,
            thc_npt=thc_npt,
            thc_solve_method=str(thc_solve_method),
            use_density_difference=bool(use_density_difference),
            df_warmup_cycles=int(df_warmup_cycles),
            df_aux_block_naux=int(df_aux_block_naux),
            df_k_q_block=int(df_k_q_block),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=diis_start_cycle_i,
            diis_space=int(diis_space),
            damping=float(damping),
            q_block=int(q_block),
        ),
    )


def run_rohf_thc(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    expand_contractions: bool = True,
    # THC build
    thc_grid_spec: Any | None = None,
    thc_grid_kind: str = "rdvr",
    thc_dvr_basis: Any | None = None,
    thc_grid_options: Any | None = None,
    thc_mode: str = "global",
    thc_local_config: Any | None = None,
    thc_npt: int | None = None,
    thc_solve_method: str = "fit_metric_qr",
    thc_mp_mode: str = "fp64",
    thc_store_Z: bool | None = None,
    thc_rebase_dD_rel_tol: float = 0.25,
    thc_rebase_min_cycle: int = 2,
    thc_tc_balance: bool = True,
    # Density-difference reference
    use_density_difference: bool = True,
    df_warmup_cycles: int = 2,
    df_aux_block_naux: int = 256,
    df_k_q_block: int = 128,
    # SCF solve
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    q_block: int = 256,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with THC J/K."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("ROHF requires nalpha>=nbeta")

    basis_in = mol.basis if basis is None else basis
    cfg = CuERIDFConfig() if df_config is None else df_config

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )

    int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout="mnQ")
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

    # Grid kind selection (Becke vs DVR).
    grid_kind_s = str(thc_grid_kind).strip().lower()
    dvr_basis_cart = None
    if grid_kind_s in {"rdvr", "r-dvr", "r_dvr", "fdvr", "f-dvr", "f_dvr"}:
        if thc_dvr_basis is None:
            dvr_basis_cart = aux_basis
        elif isinstance(thc_dvr_basis, str) and str(thc_dvr_basis).strip().lower() in {"aux", "auxbasis"}:
            dvr_basis_cart = aux_basis
        else:
            dvr_basis_cart, _dvr_name_unused = build_ao_basis_cart(
                mol,
                basis=thc_dvr_basis,
                expand_contractions=bool(expand_contractions),
            )

    # Build THC factors (global THC or local-THC/LS-THC-style blocks).
    thc_mode_s = str(thc_mode).strip().lower()
    thc_prof = profile.setdefault("thc_build", {}) if profile is not None else None

    thc_mp_mode_s = str(thc_mp_mode).strip().lower()
    if thc_mp_mode_s not in {"fp64", "tf32"}:
        raise ValueError("thc_mp_mode must be 'fp64' or 'tf32'")
    if thc_store_Z is None:
        store_Z_eff = bool(thc_mp_mode_s != "tf32")
    else:
        store_Z_eff = bool(thc_store_Z)
    if thc_prof is not None:
        thc_prof.setdefault("mp_mode", str(thc_mp_mode_s))
        thc_prof.setdefault("store_Z", bool(store_Z_eff))

    thc_factors = None
    L_metric_full = None

    if thc_mode_s in {"global", "thc", "full"}:
        from asuka.hf.thc_factors import build_thc_factors  # noqa: PLC0415

        thc = build_thc_factors(
            mol,
            ao_basis,
            aux_basis,
            sph_map=sph_map,
            grid_spec=thc_grid_spec,
            grid_kind=str(thc_grid_kind),
            dvr_basis=dvr_basis_cart,
            grid_options=thc_grid_options,
            npt=thc_npt,
            metric_backend=str(cfg.backend),
            metric_mode=str(cfg.mode),
            metric_threads=int(cfg.threads),
            solve_method=str(thc_solve_method),
            store_Z=bool(store_Z_eff),
            stream=cfg.stream,
            profile=thc_prof,
        )
        thc_factors = thc
        L_metric_full = thc.L_metric
    elif thc_mode_s in {"local", "lthc", "local-thc", "local_thc", "ls-thc", "ls_thc"}:
        from asuka.hf.local_thc_config import LocalTHCConfig  # noqa: PLC0415
        from asuka.hf.local_thc_factors import build_local_thc_factors  # noqa: PLC0415

        if thc_local_config is None:
            lcfg = LocalTHCConfig()
        elif isinstance(thc_local_config, LocalTHCConfig):
            lcfg = thc_local_config
        elif isinstance(thc_local_config, dict):
            lcfg = LocalTHCConfig(**thc_local_config)
        else:
            raise TypeError("thc_local_config must be a LocalTHCConfig or a dict")

        if bool(use_density_difference):
            from asuka.cueri import df as cueri_df  # noqa: PLC0415

            V_full = cueri_df.metric_2c2e_basis(
                aux_basis,
                stream=cfg.stream,
                backend=str(cfg.backend),
                mode=str(cfg.mode),
                threads=int(cfg.threads),
            )
            L_metric_full = cueri_df.cholesky_metric(V_full)

        lthc = build_local_thc_factors(
            mol,
            ao_basis,
            aux_basis,
            S_scf=int1e_scf.S,
            grid_kind=str(thc_grid_kind),
            dvr_basis=dvr_basis_cart,
            grid_options=thc_grid_options,
            grid_spec=thc_grid_spec,
            thc_npt=thc_npt,
            config=lcfg,
            metric_backend=str(cfg.backend),
            metric_mode=str(cfg.mode),
            metric_threads=int(cfg.threads),
            solve_method=str(thc_solve_method),
            store_Z=bool(store_Z_eff),
            stream=cfg.stream,
            profile=thc_prof,
        )
        thc_factors = lthc
    else:
        raise ValueError("thc_mode must be one of: 'global', 'local'")

    ref = None
    mo_coeff_thc0 = mo_coeff0
    if bool(use_density_difference):
        if dm0 is not None:
            raise NotImplementedError("density-difference warmup currently does not support dm0 (streamed DF requires mo_coeff)")

        from asuka.hf import df_jk_streamed  # noqa: PLC0415
        from asuka.hf import df_scf as hf_df_scf  # noqa: PLC0415
        from asuka.hf.local_thc_scf import LocalTHCReferenceUHF  # noqa: PLC0415
        from asuka.hf.thc_scf import THCReferenceUHF  # noqa: PLC0415

        n_warm = max(1, int(df_warmup_cycles))
        warm_prof = profile.setdefault("df_warmup", {}) if profile is not None else None
        warm = hf_df_scf.rohf_df(
            int1e_scf.S,
            int1e_scf.hcore,
            None,
            nalpha=int(nalpha),
            nbeta=int(nbeta),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(n_warm),
            conv_tol=0.0,
            conv_tol_dm=0.0,
            # Conventional refinement iterations before switching to THC on ΔD.
            diis=True,
            damping=0.0,
            jk_mode="streamed",
            k_engine="from_Cocc",
            k_q_block=int(df_k_q_block),
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            df_backend=str(cfg.backend),
            df_ao_rep=str(df_ao_rep),
            df_threads=int(cfg.threads),
            df_mode=str(cfg.mode),
            df_aux_block_naux=int(df_aux_block_naux),
            L_metric=L_metric_full,
            mo_coeff0=mo_coeff0,
            profile=warm_prof,
        )
        C_ref = warm.mo_coeff
        occ_a_ref, occ_b_ref = warm.mo_occ

        xp, _ = hf_df_scf._get_xp(C_ref, occ_a_ref)
        Da_ref = hf_df_scf._symmetrize(xp, hf_df_scf._density_from_C_occ(C_ref, occ_a_ref))
        Db_ref = hf_df_scf._symmetrize(xp, hf_df_scf._density_from_C_occ(C_ref, occ_b_ref))
        Dtot_ref = Da_ref + Db_ref

        ctx = df_jk_streamed.make_streamed_df_jk_context(
            ao_basis,
            aux_basis,
            L_metric=L_metric_full,
            backend=str(cfg.backend),
            threads=int(cfg.threads),
            mode=str(cfg.mode),
            ao_rep=str(df_ao_rep),
            aux_block_naux=int(df_aux_block_naux),
            profile=None,
        )
        Ca_occ = xp.ascontiguousarray(C_ref[:, : int(nalpha)])
        Cb_occ = xp.ascontiguousarray(C_ref[:, : int(nbeta)])
        occ_a_vals = occ_a_ref[: int(nalpha)]
        occ_b_vals = occ_b_ref[: int(nbeta)]
        J_ref, Ks = df_jk_streamed.df_JKs_streamed(
            ctx,
            Dtot_ref,
            [Ca_occ, Cb_occ],
            [occ_a_vals, occ_b_vals],
            k_q_block=int(df_k_q_block),
            cublas_math_mode=None,
            work={},
            profile=None,
        )
        Ka_ref, Kb_ref = Ks
        if thc_mode_s in {"global", "thc", "full"}:
            ref = THCReferenceUHF(Da_ref=Da_ref, Db_ref=Db_ref, J_ref=J_ref, Ka_ref=Ka_ref, Kb_ref=Kb_ref)
        else:
            ref = LocalTHCReferenceUHF(Da_ref=Da_ref, Db_ref=Db_ref, J_ref=J_ref, Ka_ref=Ka_ref, Kb_ref=Kb_ref)

        if mo_coeff_thc0 is None:
            mo_coeff_thc0 = C_ref

    diis_start_cycle_i = int(diis_start_cycle) if diis_start_cycle is not None else 2
    if thc_mode_s in {"global", "thc", "full"}:
        from asuka.hf.thc_scf import rohf_thc  # noqa: PLC0415

        scf = rohf_thc(
            int1e_scf.S,
            int1e_scf.hcore,
            thc_factors,
            nalpha=int(nalpha),
            nbeta=int(nbeta),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            q_block=int(q_block),
            mp_mode=str(thc_mp_mode_s),
            rebase_dD_rel_tol=float(thc_rebase_dD_rel_tol),
            rebase_min_cycle=int(thc_rebase_min_cycle),
            tc_balance=bool(thc_tc_balance),
            dm0=dm0,
            mo_coeff0=mo_coeff_thc0,
            reference=ref,
            profile=profile,
        )
    else:
        from asuka.hf.local_thc_scf import rohf_local_thc  # noqa: PLC0415

        scf = rohf_local_thc(
            int1e_scf.S,
            int1e_scf.hcore,
            thc_factors,
            nalpha=int(nalpha),
            nbeta=int(nbeta),
            enuc=float(mol.energy_nuc()),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            q_block=int(q_block),
            mp_mode=str(thc_mp_mode_s),
            rebase_dD_rel_tol=float(thc_rebase_dD_rel_tol),
            rebase_min_cycle=int(thc_rebase_min_cycle),
            tc_balance=bool(thc_tc_balance),
            dm0=dm0,
            mo_coeff0=mo_coeff_thc0,
            reference=ref,
            profile=profile,
        )

    return ROHFDFRunResult(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_metric_full,
        thc_factors=thc_factors,
        thc_run_config=_make_thc_run_config(
            hf_method="rohf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            thc_mode=str(thc_mode),
            thc_local_config=thc_local_config,
            thc_grid_spec=thc_grid_spec,
            thc_grid_kind=str(thc_grid_kind),
            thc_dvr_basis=thc_dvr_basis,
            thc_grid_options=thc_grid_options,
            thc_npt=thc_npt,
            thc_solve_method=str(thc_solve_method),
            use_density_difference=bool(use_density_difference),
            df_warmup_cycles=int(df_warmup_cycles),
            df_aux_block_naux=int(df_aux_block_naux),
            df_k_q_block=int(df_k_q_block),
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=diis_start_cycle_i,
            diis_space=int(diis_space),
            damping=float(damping),
            q_block=int(q_block),
        ),
    )


def _with_two_e_metadata(
    out: RHFDFRunResult | UHFDFRunResult | ROHFDFRunResult,
    *,
    two_e_backend: str,
    direct_jk_ctx: Any | None = None,
) -> RHFDFRunResult | UHFDFRunResult | ROHFDFRunResult:
    """Attach normalized two-electron backend metadata to frontend SCF results."""

    try:
        return _dc_replace(
            out,
            two_e_backend=str(two_e_backend),
            direct_jk_ctx=direct_jk_ctx if direct_jk_ctx is not None else getattr(out, "direct_jk_ctx", None),
        )
    except Exception:
        return out


def run_hf_df(
    mol: Molecule,
    *,
    method: str = "rhf",
    backend: str = "cuda",
    df: bool = True,
    two_e_backend: str | None = None,
    guess: Any | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    **kwargs,
) -> RHFDFRunResult | UHFDFRunResult | ROHFDFRunResult:
    """Unified HF driver (RHF/UHF/ROHF) over (backend, 2e-backend) switches.

    Notes
    -----
    - If `two_e_backend` is provided, it overrides the legacy `df` flag.
    - `two_e_backend="df"`: DF-HF via whitened DF factors `B[μ,ν,Q]`.
    - `two_e_backend="dense"`: dense AO-ERI HF (`ao_eri` is populated in the result; `df_B=None`).
    - `two_e_backend="thc"`: THC-HF (GPU-only).
    - `two_e_backend="direct"`: 4-center integral-direct SCF (GPU-only).
    - `two_e_backend="direct_df"`: streamed DF direct-SCF with no materialized `B` tensor (GPU-only).
    - `backend="cuda"`: build DF factors on GPU (CuPy).
    - `backend="cpu"`: build DF factors on CPU (requires `asuka.cueri._eri_rys_cpu`).
    """
    return _run_hf_df_dispatch(
        mol=mol,
        method=method,
        backend=backend,
        df=df,
        two_e_backend=two_e_backend,
        guess=guess,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        kwargs=kwargs,
        with_two_e_metadata=_with_two_e_metadata,
        ops={
            "run_rks_df": run_rks_df,
            "run_uks_df": run_uks_df,
            "run_rhf_dense": run_rhf_dense,
            "run_uhf_dense": run_uhf_dense,
            "run_rohf_dense": run_rohf_dense,
            "run_rhf_direct": run_rhf_direct,
            "run_uhf_direct": run_uhf_direct,
            "run_rohf_direct": run_rohf_direct,
            "run_rhf_direct_df": run_rhf_direct_df,
            "run_uhf_direct_df": run_uhf_direct_df,
            "run_rohf_direct_df": run_rohf_direct_df,
            "run_rhf_thc": run_rhf_thc,
            "run_uhf_thc": run_uhf_thc,
            "run_rohf_thc": run_rohf_thc,
            "run_rhf_df": run_rhf_df,
            "run_uhf_df": run_uhf_df,
            "run_rohf_df": run_rohf_df,
            "run_rhf_df_cpu": run_rhf_df_cpu,
            "run_uhf_df_cpu": run_uhf_df_cpu,
            "run_rohf_df_cpu": run_rohf_df_cpu,
        },
    )


def run_hf(
    mol: Molecule,
    *,
    method: str = "rhf",
    backend: str = "cuda",
    df: bool = True,
    two_e_backend: str | None = None,
    guess: Any | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    **kwargs,
) -> RHFDFRunResult | UHFDFRunResult | ROHFDFRunResult:
    """Unified HF entrypoint.

    Preferred alias for :func:`run_hf_df` (keeps the `df` switch visible in the API).
    """

    return run_hf_df(
        mol,
        method=str(method),
        backend=str(backend),
        df=bool(df),
        two_e_backend=two_e_backend,
        guess=guess,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        **kwargs,
    )


def run_rhf(
    mol: Molecule,
    *,
    backend: str = "cuda",
    df: bool = True,
    guess: Any | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    **kwargs,
) -> RHFDFRunResult:
    """Unified RHF entrypoint."""

    out = run_hf(
        mol,
        method="rhf",
        backend=str(backend),
        df=bool(df),
        guess=guess,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        **kwargs,
    )
    if not isinstance(out, RHFDFRunResult):  # pragma: no cover
        raise TypeError("run_rhf returned non-RHF result")
    return out


def run_uhf(
    mol: Molecule,
    *,
    backend: str = "cuda",
    df: bool = True,
    guess: Any | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    **kwargs,
) -> UHFDFRunResult:
    """Unified UHF entrypoint."""

    out = run_hf(
        mol,
        method="uhf",
        backend=str(backend),
        df=bool(df),
        guess=guess,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        **kwargs,
    )
    if not isinstance(out, UHFDFRunResult):  # pragma: no cover
        raise TypeError("run_uhf returned non-UHF result")
    return out


def run_rohf(
    mol: Molecule,
    *,
    backend: str = "cuda",
    df: bool = True,
    guess: Any | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    **kwargs,
) -> ROHFDFRunResult:
    """Unified ROHF entrypoint."""

    out = run_hf(
        mol,
        method="rohf",
        backend=str(backend),
        df=bool(df),
        guess=guess,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        **kwargs,
    )
    if not isinstance(out, ROHFDFRunResult):  # pragma: no cover
        raise TypeError("run_rohf returned non-ROHF result")
    return out


def run_rks_df(
    mol: Molecule,
    *,
    functional: str = "mn15",
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: "CuERIDFConfig | None" = None,
    df_int3c_plan_policy: str | None = None,
    df_int3c_work_small_max: int | None = None,
    df_int3c_work_large_min: int | None = None,
    df_int3c_blocks_per_task: int | None = None,
    df_k_cache_max_mb: int | None = None,
    df_layout: str = "mnQ",
    expand_contractions: bool = True,
    max_cycle: int = 100,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    grid_radial_n: int = 75,
    grid_angular_n: int = 590,
    xc_grid_coords: Any | None = None,
    xc_grid_weights: Any | None = None,
    xc_batch_size: int = 50000,
    profile: dict | None = None,
) -> RHFDFRunResult:
    """Run RKS-DFT with DF integrals and a meta-GGA functional (MN15/M06/M06-2X/M06-L).

    This wraps the DF-RHF solver with an XC potential on a Becke grid.
    """
    return _run_rks_df_impl(
        mol,
        functional=functional,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
        df_k_cache_max_mb=df_k_cache_max_mb,
        df_layout=df_layout,
        expand_contractions=expand_contractions,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        level_shift=level_shift,
        k_q_block=k_q_block,
        cublas_math_mode=cublas_math_mode,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        grid_radial_n=grid_radial_n,
        grid_angular_n=grid_angular_n,
        xc_grid_coords=xc_grid_coords,
        xc_grid_weights=xc_grid_weights,
        xc_batch_size=xc_batch_size,
        profile=profile,
        init_fock_cycles_default=_HF_INIT_FOCK_CYCLES,
        result_cls=RHFDFRunResult,
    )


def run_rks(
    mol: Molecule,
    *,
    functional: str = "mn15",
    backend: str = "cuda",
    integral: str = "df",
    **kwargs,
) -> RHFDFRunResult:
    """Unified RKS-DFT entrypoint.

    Parameters
    ----------
    integral : str
        Integral backend: 'df' (density fitting), 'thc' (global THC),
        'local-thc' (local THC).
    """
    if str(backend).strip().lower() != "cuda":
        raise NotImplementedError("RKS currently requires backend='cuda'")
    int_s = str(integral).strip().lower()
    if int_s == "df":
        return run_rks_df(mol, functional=functional, **kwargs)
    if int_s in {"thc", "global-thc", "global_thc"}:
        return run_rhf_thc(mol, functional=functional, thc_mode="global", **kwargs)
    if int_s in {"local-thc", "local_thc", "lthc"}:
        return run_rhf_thc(mol, functional=functional, thc_mode="local", **kwargs)
    raise ValueError(f"Unknown integral backend: {integral!r}. Use 'df', 'thc', or 'local-thc'.")


def run_uks(
    mol: Molecule,
    *,
    functional: str = "mn15",
    backend: str = "cuda",
    integral: str = "df",
    **kwargs,
) -> UHFDFRunResult:
    """Unified UKS-DFT entrypoint.

    Parameters
    ----------
    integral : str
        Integral backend: 'df' (density fitting), 'thc' (global THC),
        'local-thc' (local THC).
    """
    if str(backend).strip().lower() != "cuda":
        raise NotImplementedError("UKS currently requires backend='cuda'")
    int_s = str(integral).strip().lower()
    if int_s == "df":
        return run_uks_df(mol, functional=functional, **kwargs)
    if int_s in {"thc", "global-thc", "global_thc"}:
        return run_uhf_thc(mol, functional=functional, thc_mode="global", **kwargs)
    if int_s in {"local-thc", "local_thc", "lthc"}:
        return run_uhf_thc(mol, functional=functional, thc_mode="local", **kwargs)
    raise ValueError(f"Unknown integral backend: {integral!r}. Use 'df', 'thc', or 'local-thc'.")


def run_rhf_df_cpu(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    expand_contractions: bool = True,
    df_threads: int = 0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int | None = None,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    k_q_block: int = 128,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    init_fock_cycles: int | None = None,
    profile: dict | None = None,
) -> RHFDFRunResult:
    """Run RHF with DF factors built on CPU via cuERI-CPU."""
    return _run_rhf_df_cpu_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
        df_threads=df_threads,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        level_shift=level_shift,
        k_q_block=k_q_block,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        profile=profile,
        init_fock_cycles_default=_HF_INIT_FOCK_CYCLES,
        result_cls=RHFDFRunResult,
    )


def run_uhf_df_cpu(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    expand_contractions: bool = True,
    df_threads: int = 0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with DF factors built on CPU via cuERI-CPU."""
    return _run_uhf_df_cpu_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
        df_threads=df_threads,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        k_q_block=k_q_block,
        cublas_math_mode=cublas_math_mode,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        result_cls=UHFDFRunResult,
    )


def run_rohf_df_cpu(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    expand_contractions: bool = True,
    df_threads: int = 0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with DF factors built on CPU via cuERI-CPU."""
    return _run_rohf_df_cpu_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
        df_threads=df_threads,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        k_q_block=k_q_block,
        cublas_math_mode=cublas_math_mode,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        result_cls=ROHFDFRunResult,
    )


def run_uhf_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    df_int3c_plan_policy: str | None = None,
    df_int3c_work_small_max: int | None = None,
    df_int3c_work_large_min: int | None = None,
    df_int3c_blocks_per_task: int | None = None,
    df_k_cache_max_mb: int | None = None,
    df_layout: str = "mnQ",
    expand_contractions: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with DF integrals from cuERI."""
    return _run_uhf_df_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
        df_k_cache_max_mb=df_k_cache_max_mb,
        df_layout=df_layout,
        expand_contractions=expand_contractions,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        k_q_block=k_q_block,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        result_cls=UHFDFRunResult,
        make_df_run_config=_make_df_run_config,
    )


def run_uks_df(
    mol: Molecule,
    *,
    functional: str = "mn15",
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: "CuERIDFConfig | None" = None,
    df_int3c_plan_policy: str | None = None,
    df_int3c_work_small_max: int | None = None,
    df_int3c_work_large_min: int | None = None,
    df_int3c_blocks_per_task: int | None = None,
    df_k_cache_max_mb: int | None = None,
    df_layout: str = "mnQ",
    expand_contractions: bool = True,
    max_cycle: int = 100,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    grid_radial_n: int = 75,
    grid_angular_n: int = 590,
    xc_grid_coords: Any | None = None,
    xc_grid_weights: Any | None = None,
    xc_batch_size: int = 50000,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UKS-DFT with DF integrals and a meta-GGA functional (MN15/M06/M06-2X/M06-L).

    Open-shell (spin-polarized) analogue of run_rks_df, using the full
    spin-polarized meta-GGA form (rho_a/rho_b, sigma_aa/sigma_ab/sigma_bb,
    tau_a/tau_b) to build V_xc and E_xc.
    """
    return _run_uks_df_impl(
        mol,
        functional=functional,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
        df_k_cache_max_mb=df_k_cache_max_mb,
        df_layout=df_layout,
        expand_contractions=expand_contractions,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        k_q_block=k_q_block,
        grid_radial_n=grid_radial_n,
        grid_angular_n=grid_angular_n,
        xc_grid_coords=xc_grid_coords,
        xc_grid_weights=xc_grid_weights,
        xc_batch_size=xc_batch_size,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        result_cls=UHFDFRunResult,
        make_df_run_config=_make_df_run_config,
    )


def run_rohf_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
    df_int3c_plan_policy: str | None = None,
    df_int3c_work_small_max: int | None = None,
    df_int3c_work_large_min: int | None = None,
    df_int3c_blocks_per_task: int | None = None,
    df_k_cache_max_mb: int | None = None,
    df_layout: str = "mnQ",
    expand_contractions: bool = True,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    k_q_block: int = 128,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with DF integrals from cuERI."""
    return _run_rohf_df_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
        df_k_cache_max_mb=df_k_cache_max_mb,
        df_layout=df_layout,
        expand_contractions=expand_contractions,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        k_q_block=k_q_block,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        result_cls=ROHFDFRunResult,
        make_df_run_config=_make_df_run_config,
    )


__all__ = [
    "clear_hf_frontend_caches",
    "RHFDFRunResult",
    "ROHFDFRunResult",
    "UHFDFRunResult",
    "run_hf",
    "run_hf_df",
    "run_rks_df",
    "run_rks",
    "run_uks_df",
    "run_uks",
    "run_rhf_dense",
    "run_rhf_direct",
    "run_rhf_direct_df",
    "run_rhf_df_cpu",
    "run_rhf_df",
    "run_rhf",
    "run_rohf_dense",
    "run_rohf_direct",
    "run_rohf_direct_df",
    "run_rohf_df_cpu",
    "run_rohf_df",
    "run_rohf",
    "run_uhf_dense",
    "run_uhf_direct",
    "run_uhf_direct_df",
    "run_uhf_df_cpu",
    "run_uhf_df",
    "run_uhf",
]
