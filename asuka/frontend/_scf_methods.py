from __future__ import annotations

from typing import Any, Callable

import numpy as np

from asuka.hf.df_scf import rhf_df, rohf_df, uhf_df
from asuka.integrals.cueri_df import CuERIDFConfig, build_df_B_from_cueri_packed_bases
from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu
from asuka.integrals.int1e_cart import build_int1e_cart

from ._scf_build import (
    apply_sph_transform,
    atom_coords_charges_bohr,
    build_aux_basis_cart,
)
from ._scf_config import resolve_cueri_df_config
from .one_electron import build_ao_basis_cart
from ._scf_spin import nalpha_nbeta_from_mol as _nalpha_nbeta_from_mol_impl


def _nalpha_nbeta_from_mol(mol) -> tuple[int, int]:
    return _nalpha_nbeta_from_mol_impl(mol)


def _maybe_pack_df_B(B_scf, *, df_layout_s: str, int1e_scf):
    if B_scf is None:
        return B_scf
    try:  # pragma: no cover
        import os as _os_dfpack  # noqa: PLC0415
        import cupy as _cp_dfpack  # noqa: PLC0415
        from asuka.integrals.df_packed_s2 import ao_packed_s2_enabled, pack_B_to_Qp  # noqa: PLC0415

        _explicit = "ASUKA_DF_AO_PACKED_S2" in _os_dfpack.environ
        _want_pack = bool(ao_packed_s2_enabled()) if _explicit else isinstance(B_scf, _cp_dfpack.ndarray)
        if bool(_want_pack) and int(getattr(B_scf, "ndim", 0)) == 3:
            layout = "mnQ" if str(df_layout_s) == "mnq" else "Qmn"
            return pack_B_to_Qp(B_scf, layout=layout, nao=int(int1e_scf.S.shape[0]))
    except Exception:
        pass
    return B_scf


def _profile_section(profile: dict | None, key: str):
    if profile is None:
        return None
    return profile.setdefault(key, {})


def _normalize_df_layout(df_layout: str) -> str:
    df_layout_s = str(df_layout).strip().lower()
    if df_layout_s not in {"mnq", "qmn"}:
        raise ValueError("df_layout must be one of: 'mnQ', 'Qmn'")
    return df_layout_s


def _df_ao_rep(mol) -> str:
    return "cart" if bool(mol.cart) else "sph"


def _build_ao_int1e_aux_problem(
    mol,
    *,
    basis_in,
    auxbasis,
    expand_contractions: bool,
):
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)
    aux_basis, auxbasis_name = build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )
    return ao_basis, str(basis_name), int1e, aux_basis, str(auxbasis_name)


def _build_df_factors_gpu(
    mol,
    *,
    ao_basis,
    aux_basis,
    cfg: CuERIDFConfig,
    df_layout_s: str,
    df_profile,
):
    return build_df_B_from_cueri_packed_bases(
        ao_basis,
        aux_basis,
        config=cfg,
        layout=str(df_layout_s),
        ao_rep=str(_df_ao_rep(mol)),
        profile=df_profile,
        return_L=True,
    )


def _build_df_factors_cpu(
    *,
    ao_basis,
    aux_basis,
    df_threads: int,
    df_profile,
):
    return build_df_B_from_cueri_packed_bases_cpu(
        ao_basis,
        aux_basis,
        threads=int(df_threads),
        profile=df_profile,
        return_L=True,
    )


def _transform_df_for_scf(
    mol,
    *,
    int1e,
    B,
    ao_basis,
    df_layout: str = "mnQ",
):
    if bool(mol.cart):
        return apply_sph_transform(mol, int1e, B, ao_basis, df_B_layout=str(df_layout))
    int1e_scf, _B_unused, sph_map = apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout=str(df_layout))
    return int1e_scf, B, sph_map


def _prepare_df_gpu_problem(
    mol,
    *,
    basis_in,
    auxbasis,
    expand_contractions: bool,
    cfg: CuERIDFConfig,
    df_layout: str,
    profile: dict | None,
):
    ao_basis, basis_name, int1e, aux_basis, auxbasis_name = _build_ao_int1e_aux_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )
    df_layout_s = _normalize_df_layout(df_layout)
    df_prof = _profile_section(profile, "df_build")
    B, L_chol = _build_df_factors_gpu(
        mol,
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        cfg=cfg,
        df_layout_s=df_layout_s,
        df_profile=df_prof,
    )
    int1e_scf, B_scf, sph_map = _transform_df_for_scf(
        mol,
        int1e=int1e,
        B=B,
        ao_basis=ao_basis,
        df_layout=str(df_layout),
    )
    return (
        ao_basis,
        basis_name,
        int1e,
        int1e_scf,
        aux_basis,
        auxbasis_name,
        B,
        B_scf,
        sph_map,
        L_chol,
        df_layout_s,
    )


def _prepare_df_cpu_problem(
    mol,
    *,
    basis_in,
    auxbasis,
    expand_contractions: bool,
    df_threads: int,
    profile: dict | None,
):
    ao_basis, basis_name, int1e, aux_basis, auxbasis_name = _build_ao_int1e_aux_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )
    df_prof = _profile_section(profile, "df_build_cpu")
    B, L_chol = _build_df_factors_cpu(
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        df_threads=int(df_threads),
        df_profile=df_prof,
    )
    int1e_scf, B_scf, sph_map = apply_sph_transform(mol, int1e, B, ao_basis)
    return ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, B_scf, sph_map, L_chol


def _coerce_or_build_xc_grid(
    mol,
    *,
    xc_grid_coords,
    xc_grid_weights,
    grid_radial_n: int,
    grid_angular_n: int,
):
    from asuka.density.grids_device import make_becke_grid_device
    import cupy as _cp_grid

    if xc_grid_coords is not None and xc_grid_weights is not None:
        return (
            _cp_grid.asarray(xc_grid_coords, dtype=_cp_grid.float64),
            _cp_grid.asarray(xc_grid_weights, dtype=_cp_grid.float64),
        )
    return make_becke_grid_device(
        mol,
        radial_n=int(grid_radial_n),
        angular_n=int(grid_angular_n),
        radial_scheme="treutler",
    )


def _xc_sph_transform_from_map(mol, sph_map):
    if bool(mol.cart) or sph_map is None:
        return None
    import cupy as _cp_xc

    if hasattr(sph_map, "T_c2s"):
        return _cp_xc.asarray(sph_map.T_c2s, dtype=_cp_xc.float64)
    if hasattr(sph_map, "T_matrix"):
        return _cp_xc.asarray(sph_map.T_matrix, dtype=_cp_xc.float64)
    if isinstance(sph_map, tuple) and len(sph_map) >= 1:
        return _cp_xc.asarray(sph_map[0], dtype=_cp_xc.float64)
    return None


def run_rhf_df_cpu_impl(
    mol,
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
    init_fock_cycles_default: int,
    result_cls: Any,
):
    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_df_cpu currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, B_scf, sph_map, L_chol = _prepare_df_cpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_threads=int(df_threads),
        profile=profile,
    )

    init_fock_cycles_i = int(init_fock_cycles_default) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf_prof = profile if profile is not None else None
    scf = rhf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        profile=scf_prof,
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
    )


def run_rhf_df_impl(
    mol,
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
    init_fock_cycles_default: int,
    resolve_cueri_df_config: Callable[..., CuERIDFConfig],
    rhf_prep_key: Callable[..., Any],
    rhf_guess_key: Callable[..., Any],
    cache_get: Callable[..., Any],
    cache_put: Callable[..., None],
    rhf_prep_cache: Any,
    rhf_guess_cache: Any,
    hf_prep_cache_max: int,
    hf_guess_cache_max: int,
    copy_mo_coeff_for_cache: Callable[[Any], Any],
    make_df_run_config: Callable[..., Any],
    result_cls: Any,
):
    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_df currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    cfg = resolve_cueri_df_config(
        df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
    )

    df_layout_s = _normalize_df_layout(df_layout)

    prep_key = rhf_prep_key(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_config=cfg,
        df_layout_build=str(df_layout_s),
    )
    prep_hit = cache_get(rhf_prep_cache, prep_key)
    if prep_hit is None:
        ao_basis, basis_name, int1e, aux_basis, auxbasis_name = _build_ao_int1e_aux_problem(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )

        df_prof = _profile_section(profile, "df_build")
        if df_prof is not None:
            df_prof["cache_hit"] = False

        B, L_chol = _build_df_factors_gpu(
            mol,
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            cfg=cfg,
            df_layout_s=df_layout_s,
            df_profile=df_prof,
        )
        cache_put(
            rhf_prep_cache,
            prep_key,
            (ao_basis, str(basis_name), int1e, aux_basis, str(auxbasis_name), B, L_chol),
            max_size=int(hf_prep_cache_max),
        )
    else:
        _prep_tuple = prep_hit
        if len(_prep_tuple) == 7:
            ao_basis, basis_name, int1e, aux_basis, auxbasis_name, B, L_chol = _prep_tuple
        else:
            ao_basis, basis_name, int1e, aux_basis, auxbasis_name, B = _prep_tuple
            L_chol = None
        df_prof = _profile_section(profile, "df_build")
        if df_prof is not None:
            df_prof["cache_hit"] = True

    int1e_scf, B_scf, sph_map = _transform_df_for_scf(
        mol,
        int1e=int1e,
        B=B,
        ao_basis=ao_basis,
        df_layout=str(df_layout),
    )

    try:  # pragma: no cover
        if B is not None and B is not B_scf:
            del B
        import cupy as cp  # noqa: PLC0415

        if isinstance(B_scf, cp.ndarray) and int(getattr(B_scf, "nbytes", 0)) >= 2_000_000_000:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

    if mo_coeff0 is None and dm0 is None:
        guess_key = rhf_guess_key(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )
        guess_hit = cache_get(rhf_guess_cache, guess_key)
        if guess_hit is not None:
            mo_coeff0 = copy_mo_coeff_for_cache(guess_hit)
            if profile is not None:
                profile.setdefault("scf_guess", {})["cache_hit"] = True
        elif profile is not None:
            profile.setdefault("scf_guess", {})["cache_hit"] = False

    init_fock_cycles_i = int(init_fock_cycles_default) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf_prof = profile if profile is not None else None
    scf = rhf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        k_cache_max_mb=None if df_k_cache_max_mb is None else int(df_k_cache_max_mb),
        profile=scf_prof,
    )

    B_scf = _maybe_pack_df_B(B_scf, df_layout_s=df_layout_s, int1e_scf=int1e_scf)

    if bool(getattr(scf, "converged", False)):
        guess_key = rhf_guess_key(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )
        cache_put(
            rhf_guess_cache,
            guess_key,
            copy_mo_coeff_for_cache(scf.mo_coeff),
            max_size=int(hf_guess_cache_max),
        )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=make_df_run_config(
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
    )


def run_uhf_df_cpu_impl(
    mol,
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
    result_cls: Any,
):
    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis

    ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, B_scf, sph_map, L_chol = _prepare_df_cpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_threads=int(df_threads),
        profile=profile,
    )

    scf_prof = profile if profile is not None else None
    scf = uhf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
    )


def run_rohf_df_cpu_impl(
    mol,
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
    result_cls: Any,
):
    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_df_cpu requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis

    ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, B_scf, sph_map, L_chol = _prepare_df_cpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_threads=int(df_threads),
        profile=profile,
    )

    scf_prof = profile if profile is not None else None
    scf = rohf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
    )


def run_uhf_df_impl(
    mol,
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
    result_cls: Any,
    make_df_run_config: Callable[..., Any],
):
    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis
    cfg = resolve_cueri_df_config(
        df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
    )

    (
        ao_basis,
        basis_name,
        _int1e_unused,
        int1e_scf,
        aux_basis,
        auxbasis_name,
        _B_unused,
        B_scf,
        sph_map,
        L_chol,
        df_layout_s,
    ) = _prepare_df_gpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        cfg=cfg,
        df_layout=str(df_layout),
        profile=profile,
    )

    scf_prof = profile if profile is not None else None
    scf = uhf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        k_cache_max_mb=None if df_k_cache_max_mb is None else int(df_k_cache_max_mb),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    B_scf = _maybe_pack_df_B(B_scf, df_layout_s=df_layout_s, int1e_scf=int1e_scf)

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=make_df_run_config(
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
        ),
    )


def run_rohf_df_impl(
    mol,
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
    result_cls: Any,
    make_df_run_config: Callable[..., Any],
):
    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_df requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis
    cfg = resolve_cueri_df_config(
        df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
    )

    (
        ao_basis,
        basis_name,
        _int1e_unused,
        int1e_scf,
        aux_basis,
        auxbasis_name,
        _B_unused,
        B_scf,
        sph_map,
        L_chol,
        df_layout_s,
    ) = _prepare_df_gpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        cfg=cfg,
        df_layout=str(df_layout),
        profile=profile,
    )

    scf_prof = profile if profile is not None else None
    scf = rohf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        k_cache_max_mb=None if df_k_cache_max_mb is None else int(df_k_cache_max_mb),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    B_scf = _maybe_pack_df_B(B_scf, df_layout_s=df_layout_s, int1e_scf=int1e_scf)

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=make_df_run_config(
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
        ),
    )


def run_rks_df_impl(
    mol,
    *,
    functional: str = "mn15",
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
    init_fock_cycles_default: int,
    result_cls: Any,
):
    from asuka.xc.functional import get_functional

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rks_df currently supports only closed-shell molecules (spin=0)")

    xc_spec = get_functional(functional)
    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RKS requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    cfg = resolve_cueri_df_config(
        df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
    )

    (
        ao_basis,
        basis_name,
        int1e,
        int1e_scf,
        aux_basis,
        auxbasis_name,
        _B_unused,
        B_scf,
        sph_map,
        L_chol,
        _df_layout_s,
    ) = _prepare_df_gpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        cfg=cfg,
        df_layout=str(df_layout),
        profile=profile,
    )

    grid_coords, grid_weights = _coerce_or_build_xc_grid(
        mol,
        xc_grid_coords=xc_grid_coords,
        xc_grid_weights=xc_grid_weights,
        grid_radial_n=int(grid_radial_n),
        grid_angular_n=int(grid_angular_n),
    )
    xc_sph_transform = _xc_sph_transform_from_map(mol, sph_map)

    init_fock_cycles_i = int(init_fock_cycles_default) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf = rhf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        k_cache_max_mb=None if df_k_cache_max_mb is None else int(df_k_cache_max_mb),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        profile=profile,
        xc_spec=xc_spec,
        xc_grid_coords=grid_coords,
        xc_grid_weights=grid_weights,
        xc_ao_basis=ao_basis,
        xc_sph_transform=xc_sph_transform,
        xc_batch_size=int(xc_batch_size),
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
    )


def run_uks_df_impl(
    mol,
    *,
    functional: str = "mn15",
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
    result_cls: Any,
    make_df_run_config: Callable[..., Any],
):
    from asuka.xc.functional import get_functional

    xc_spec = get_functional(functional)
    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)

    basis_in = mol.basis if basis is None else basis
    cfg = resolve_cueri_df_config(
        df_config,
        df_int3c_plan_policy=df_int3c_plan_policy,
        df_int3c_work_small_max=df_int3c_work_small_max,
        df_int3c_work_large_min=df_int3c_work_large_min,
        df_int3c_blocks_per_task=df_int3c_blocks_per_task,
    )
    (
        ao_basis,
        basis_name,
        _int1e_unused,
        int1e_scf,
        aux_basis,
        auxbasis_name,
        _B_unused,
        B_scf,
        sph_map,
        L_chol,
        _df_layout_s,
    ) = _prepare_df_gpu_problem(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        cfg=cfg,
        df_layout=str(df_layout),
        profile=profile,
    )

    grid_coords, grid_weights = _coerce_or_build_xc_grid(
        mol,
        xc_grid_coords=xc_grid_coords,
        xc_grid_weights=xc_grid_weights,
        grid_radial_n=int(grid_radial_n),
        grid_angular_n=int(grid_angular_n),
    )
    xc_sph_transform = _xc_sph_transform_from_map(mol, sph_map)

    diis_start_cycle_i = int(diis_start_cycle)
    scf = uhf_df(
        int1e_scf.S,
        int1e_scf.hcore,
        B_scf,
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
        k_q_block=int(k_q_block),
        k_cache_max_mb=None if df_k_cache_max_mb is None else int(df_k_cache_max_mb),
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=profile,
        xc_spec=xc_spec,
        xc_grid_coords=grid_coords,
        xc_grid_weights=grid_weights,
        xc_ao_basis=ao_basis,
        xc_sph_transform=xc_sph_transform,
        xc_batch_size=int(xc_batch_size),
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name=str(auxbasis_name),
        ao_basis=ao_basis,
        aux_basis=aux_basis,
        int1e=int1e_scf,
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
        df_run_config=make_df_run_config(
            hf_method="uks",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=cfg,
            expand_contractions=bool(expand_contractions),
            backend="cuda",
            max_cycle=int(max_cycle),
            conv_tol=float(conv_tol),
            conv_tol_dm=float(conv_tol_dm),
            diis=bool(diis),
            diis_start_cycle=int(diis_start_cycle_i),
            diis_space=int(diis_space),
            damping=float(damping),
            level_shift=0.0,
            k_q_block=int(k_q_block),
        ),
    )
