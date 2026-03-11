from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


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


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


def make_thc_run_config(
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


def make_df_run_config(
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


__all__ = [
    "DFRunConfig",
    "THCRunConfig",
    "env_float",
    "make_df_run_config",
    "make_thc_run_config",
]
