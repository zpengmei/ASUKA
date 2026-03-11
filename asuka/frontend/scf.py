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
from dataclasses import dataclass
import os
from typing import Any, TYPE_CHECKING

import numpy as np

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
from ._scf_dense import (
    build_dense_ao_eri as _build_dense_ao_eri_impl,
    dense_default_threads as _dense_default_threads_impl,
)
from ._scf_dispatch import run_hf_df_dispatch as _run_hf_df_dispatch
from ._scf_keys import (
    copy_mo_coeff_for_cache as _copy_mo_coeff_for_cache_impl,
    rhf_guess_key as _rhf_guess_key_impl,
    rhf_prep_key as _rhf_prep_key_impl,
)
from ._scf_metadata import with_two_e_metadata as _with_two_e_metadata
from ._scf_methods import (
    run_rhf_direct_df_impl as _run_rhf_direct_df_impl,
    run_rhf_df_impl as _run_rhf_df_impl,
    run_rhf_df_cpu_impl as _run_rhf_df_cpu_impl,
    run_rohf_direct_df_impl as _run_rohf_direct_df_impl,
    run_rohf_df_cpu_impl as _run_rohf_df_cpu_impl,
    run_rohf_df_impl as _run_rohf_df_impl,
    run_rks_df_impl as _run_rks_df_impl,
    run_uhf_direct_df_impl as _run_uhf_direct_df_impl,
    run_uhf_df_cpu_impl as _run_uhf_df_cpu_impl,
    run_uhf_df_impl as _run_uhf_df_impl,
    run_uks_df_impl as _run_uks_df_impl,
)
from ._scf_dense_direct_methods import (
    run_rhf_dense_impl as _run_rhf_dense_impl,
    run_uhf_dense_impl as _run_uhf_dense_impl,
    run_rohf_dense_impl as _run_rohf_dense_impl,
    run_rhf_direct_impl as _run_rhf_direct_impl,
    run_uhf_direct_impl as _run_uhf_direct_impl,
    run_rohf_direct_impl as _run_rohf_direct_impl,
)
from ._scf_run_config import (
    DFRunConfig,
    THCRunConfig,
    env_float as _env_float,
    make_df_run_config as _make_df_run_config,
    make_thc_run_config as _make_thc_run_config,
)
from ._scf_spin import nalpha_nbeta_from_mol as _nalpha_nbeta_from_mol_impl
from ._scf_thc_common import (
    coerce_local_thc_config as _coerce_local_thc_config,
    resolve_thc_grid_and_dvr_basis as _resolve_thc_grid_and_dvr_basis,
    resolve_thc_mp_store_policy as _resolve_thc_mp_store_policy,
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

_HF_DENSE_MEM_BUDGET_GIB = _env_float("ASUKA_HF_DENSE_MEM_BUDGET_GIB", 8.0)
_RHF_PREP_CACHE: "OrderedDict[tuple[Any, ...], tuple[Any, str, Int1eResult, Any, str, Any]]" = OrderedDict()
_RHF_GUESS_CACHE: "OrderedDict[tuple[Any, ...], Any]" = OrderedDict()


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
    return _rhf_prep_key_impl(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
        df_config=df_config,
        df_layout_build=df_layout_build,
    )


def _rhf_guess_key(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
) -> tuple[Any, ...]:
    return _rhf_guess_key_impl(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=expand_contractions,
    )


def _copy_mo_coeff_for_cache(mo_coeff: Any):
    return _copy_mo_coeff_for_cache_impl(mo_coeff)


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
    return _nalpha_nbeta_from_mol_impl(mol)


def _dense_default_threads(backend: str) -> int:
    return _dense_default_threads_impl(backend)


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
    return _build_dense_ao_eri_impl(
        ao_basis,
        backend=backend,
        dense_threads=dense_threads,
        dense_max_tile_bytes=dense_max_tile_bytes,
        dense_eps_ao=dense_eps_ao,
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        default_mem_budget_gib=float(_HF_DENSE_MEM_BUDGET_GIB),
        profile=profile,
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
    return _run_rhf_dense_impl(
        mol,
        basis=basis,
        backend=backend,
        expand_contractions=expand_contractions,
        dense_threads=dense_threads,
        dense_max_tile_bytes=dense_max_tile_bytes,
        dense_eps_ao=dense_eps_ao,
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        level_shift=level_shift,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        profile=profile,
        init_fock_cycles_default=int(_HF_INIT_FOCK_CYCLES),
        atom_coords_charges_bohr_fn=_atom_coords_charges_bohr,
        build_dense_ao_eri_fn=_build_dense_ao_eri,
        apply_sph_transform_fn=_apply_sph_transform,
        result_cls=RHFDFRunResult,
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
    return _run_rhf_direct_impl(
        mol,
        basis=basis,
        expand_contractions=expand_contractions,
        eps_schwarz=eps_schwarz,
        direct_threads=direct_threads,
        direct_max_tile_bytes=direct_max_tile_bytes,
        direct_gpu_task_budget_bytes=direct_gpu_task_budget_bytes,
        direct_max_slab_tasks=direct_max_slab_tasks,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        level_shift=level_shift,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        profile=profile,
        init_fock_cycles_default=int(_HF_INIT_FOCK_CYCLES),
        atom_coords_charges_bohr_fn=_atom_coords_charges_bohr,
        result_cls=RHFDFRunResult,
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
    return _run_uhf_direct_impl(
        mol,
        basis=basis,
        expand_contractions=expand_contractions,
        eps_schwarz=eps_schwarz,
        direct_threads=direct_threads,
        direct_max_tile_bytes=direct_max_tile_bytes,
        direct_gpu_task_budget_bytes=direct_gpu_task_budget_bytes,
        direct_max_slab_tasks=direct_max_slab_tasks,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        nalpha_nbeta_from_mol_fn=_nalpha_nbeta_from_mol,
        atom_coords_charges_bohr_fn=_atom_coords_charges_bohr,
        result_cls=UHFDFRunResult,
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
    return _run_rohf_direct_impl(
        mol,
        basis=basis,
        expand_contractions=expand_contractions,
        eps_schwarz=eps_schwarz,
        direct_threads=direct_threads,
        direct_max_tile_bytes=direct_max_tile_bytes,
        direct_gpu_task_budget_bytes=direct_gpu_task_budget_bytes,
        direct_max_slab_tasks=direct_max_slab_tasks,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        nalpha_nbeta_from_mol_fn=_nalpha_nbeta_from_mol,
        atom_coords_charges_bohr_fn=_atom_coords_charges_bohr,
        result_cls=ROHFDFRunResult,
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

    return _run_rhf_direct_df_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=bool(expand_contractions),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=diis_start_cycle,
        diis_space=int(diis_space),
        damping=float(damping),
        level_shift=float(level_shift),
        k_q_block=int(k_q_block),
        cublas_math_mode=cublas_math_mode,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        df_aux_block_naux=int(df_aux_block_naux),
        L_metric=L_metric,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        profile=profile,
        init_fock_cycles_default=int(_HF_INIT_FOCK_CYCLES),
        rhf_guess_key=_rhf_guess_key,
        cache_get=_cache_get,
        cache_put=_cache_put,
        rhf_guess_cache=_RHF_GUESS_CACHE,
        hf_guess_cache_max=int(_HF_GUESS_CACHE_MAX),
        copy_mo_coeff_for_cache=_copy_mo_coeff_for_cache,
        make_df_run_config=_make_df_run_config,
        result_cls=RHFDFRunResult,
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

    return _run_uhf_direct_df_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=bool(expand_contractions),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        k_q_block=int(k_q_block),
        cublas_math_mode=cublas_math_mode,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        df_aux_block_naux=int(df_aux_block_naux),
        L_metric=L_metric,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        make_df_run_config=_make_df_run_config,
        result_cls=UHFDFRunResult,
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

    return _run_rohf_direct_df_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=bool(expand_contractions),
        max_cycle=int(max_cycle),
        conv_tol=float(conv_tol),
        conv_tol_dm=float(conv_tol_dm),
        diis=bool(diis),
        diis_start_cycle=int(diis_start_cycle),
        diis_space=int(diis_space),
        damping=float(damping),
        k_q_block=int(k_q_block),
        cublas_math_mode=cublas_math_mode,
        df_backend=df_backend,
        df_mode=df_mode,
        df_threads=df_threads,
        df_aux_block_naux=int(df_aux_block_naux),
        L_metric=L_metric,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        make_df_run_config=_make_df_run_config,
        result_cls=ROHFDFRunResult,
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
    return _run_uhf_dense_impl(
        mol,
        basis=basis,
        backend=backend,
        expand_contractions=expand_contractions,
        dense_threads=dense_threads,
        dense_max_tile_bytes=dense_max_tile_bytes,
        dense_eps_ao=dense_eps_ao,
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        nalpha_nbeta_from_mol_fn=_nalpha_nbeta_from_mol,
        atom_coords_charges_bohr_fn=_atom_coords_charges_bohr,
        build_dense_ao_eri_fn=_build_dense_ao_eri,
        apply_sph_transform_fn=_apply_sph_transform,
        result_cls=UHFDFRunResult,
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
    return _run_rohf_dense_impl(
        mol,
        basis=basis,
        backend=backend,
        expand_contractions=expand_contractions,
        dense_threads=dense_threads,
        dense_max_tile_bytes=dense_max_tile_bytes,
        dense_eps_ao=dense_eps_ao,
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        nalpha_nbeta_from_mol_fn=_nalpha_nbeta_from_mol,
        atom_coords_charges_bohr_fn=_atom_coords_charges_bohr,
        build_dense_ao_eri_fn=_build_dense_ao_eri,
        apply_sph_transform_fn=_apply_sph_transform,
        result_cls=ROHFDFRunResult,
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
    """Run RHF (or RKS-DFT if functional is given) with THC J/K."""

    from ._scf_thc_methods import run_rhf_thc_impl

    return run_rhf_thc_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=expand_contractions,
        thc_mode=thc_mode,
        thc_local_config=thc_local_config,
        thc_grid_spec=thc_grid_spec,
        thc_grid_kind=thc_grid_kind,
        thc_dvr_basis=thc_dvr_basis,
        thc_grid_options=thc_grid_options,
        thc_npt=thc_npt,
        thc_solve_method=thc_solve_method,
        thc_mp_mode=thc_mp_mode,
        thc_store_Z=thc_store_Z,
        thc_rebase_dD_rel_tol=thc_rebase_dD_rel_tol,
        thc_rebase_min_cycle=thc_rebase_min_cycle,
        thc_tc_balance=thc_tc_balance,
        use_density_difference=use_density_difference,
        df_warmup_cycles=df_warmup_cycles,
        df_warmup_ediff=df_warmup_ediff,
        df_warmup_max_cycles=df_warmup_max_cycles,
        df_aux_block_naux=df_aux_block_naux,
        df_k_q_block=df_k_q_block,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        level_shift=level_shift,
        q_block=q_block,
        init_guess=init_guess,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=init_fock_cycles,
        profile=profile,
        functional=functional,
        grid_radial_n=grid_radial_n,
        grid_angular_n=grid_angular_n,
        xc_batch_size=xc_batch_size,
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

    from ._scf_thc_methods import run_uhf_thc_impl

    return run_uhf_thc_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=expand_contractions,
        thc_grid_spec=thc_grid_spec,
        thc_grid_kind=thc_grid_kind,
        thc_dvr_basis=thc_dvr_basis,
        thc_grid_options=thc_grid_options,
        thc_mode=thc_mode,
        thc_local_config=thc_local_config,
        thc_npt=thc_npt,
        thc_solve_method=thc_solve_method,
        thc_mp_mode=thc_mp_mode,
        thc_store_Z=thc_store_Z,
        thc_rebase_dD_rel_tol=thc_rebase_dD_rel_tol,
        thc_rebase_min_cycle=thc_rebase_min_cycle,
        thc_tc_balance=thc_tc_balance,
        use_density_difference=use_density_difference,
        df_warmup_cycles=df_warmup_cycles,
        df_aux_block_naux=df_aux_block_naux,
        df_k_q_block=df_k_q_block,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        q_block=q_block,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        functional=functional,
        grid_radial_n=grid_radial_n,
        grid_angular_n=grid_angular_n,
        xc_batch_size=xc_batch_size,
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

    from ._scf_thc_methods import run_rohf_thc_impl

    return run_rohf_thc_impl(
        mol,
        basis=basis,
        auxbasis=auxbasis,
        df_config=df_config,
        expand_contractions=expand_contractions,
        thc_grid_spec=thc_grid_spec,
        thc_grid_kind=thc_grid_kind,
        thc_dvr_basis=thc_dvr_basis,
        thc_grid_options=thc_grid_options,
        thc_mode=thc_mode,
        thc_local_config=thc_local_config,
        thc_npt=thc_npt,
        thc_solve_method=thc_solve_method,
        thc_mp_mode=thc_mp_mode,
        thc_store_Z=thc_store_Z,
        thc_rebase_dD_rel_tol=thc_rebase_dD_rel_tol,
        thc_rebase_min_cycle=thc_rebase_min_cycle,
        thc_tc_balance=thc_tc_balance,
        use_density_difference=use_density_difference,
        df_warmup_cycles=df_warmup_cycles,
        df_aux_block_naux=df_aux_block_naux,
        df_k_q_block=df_k_q_block,
        max_cycle=max_cycle,
        conv_tol=conv_tol,
        conv_tol_dm=conv_tol_dm,
        diis=diis,
        diis_start_cycle=diis_start_cycle,
        diis_space=diis_space,
        damping=damping,
        q_block=q_block,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
    )

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
