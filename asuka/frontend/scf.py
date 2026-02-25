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
import json
import os
import threading
from typing import Any

import numpy as np

from asuka.hf.dense_eri import build_ao_eri_dense
from asuka.hf.dense_scf import rhf_dense, rohf_dense, uhf_dense
from asuka.hf.df_scf import SCFResult, rhf_df, rohf_df, uhf_df
from asuka.integrals.cueri_df import CuERIDFConfig, build_df_B_from_cueri_packed_bases
from asuka.integrals.cueri_df_cpu import build_df_B_from_cueri_packed_bases_cpu
from asuka.integrals.int1e_cart import Int1eResult, build_int1e_cart

from .basis_bse import load_autoaux_shells, load_basis_shells
from .basis_packer import pack_cart_basis, parse_pyscf_basis_dict
from .molecule import Molecule
from .one_electron import build_ao_basis_cart
from .periodic_table import atomic_number


def _apply_sph_transform(
    mol: Molecule,
    int1e: Int1eResult,
    B,
    ao_basis,
) -> tuple[Int1eResult, Any, tuple[np.ndarray, int, int] | None]:
    """If ``mol.cart=False``, transform int1e and B to spherical AOs.

    Returns ``(int1e, B, sph_map_or_None)``.
    """
    if bool(mol.cart):
        return int1e, B, None

    from asuka.integrals.cart2sph import (  # noqa: PLC0415
        build_cart2sph_matrix,
        compute_sph_layout_from_cart_basis,
        transform_1e_cart_to_sph,
        transform_df_B_cart_to_sph,
    )

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
    nao_cart = int(int1e.S.shape[0])

    T = build_cart2sph_matrix(shell_l, shell_ao_start_cart, shell_ao_start_sph, nao_cart, nao_sph)

    S_sph = transform_1e_cart_to_sph(int1e.S, T)
    T_kin_sph = transform_1e_cart_to_sph(int1e.T, T)
    V_sph = transform_1e_cart_to_sph(int1e.V, T)
    int1e_sph = Int1eResult(S=S_sph, T=T_kin_sph, V=V_sph)

    if B is not None:
        B_sph = transform_df_B_cart_to_sph(B, T)
    else:
        B_sph = None

    return int1e_sph, B_sph, (T, nao_cart, nao_sph)


@dataclass(frozen=True)
class RHFDFRunResult:
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
    sph_map: tuple[np.ndarray, int, int] | None = None  # (T, nao_cart, nao_sph)
    df_L: Any | None = None  # Cholesky factor of aux metric (for deterministic gradients)


@dataclass(frozen=True)
class UHFDFRunResult:
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
    sph_map: tuple[np.ndarray, int, int] | None = None  # (T, nao_cart, nao_sph)


@dataclass(frozen=True)
class ROHFDFRunResult:
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
    sph_map: tuple[np.ndarray, int, int] | None = None  # (T, nao_cart, nao_sph)


_HF_PREP_CACHE_MAX = max(0, int(os.environ.get("ASUKA_HF_PREP_CACHE_MAX", "0")))
_HF_GUESS_CACHE_MAX = max(0, int(os.environ.get("ASUKA_HF_GUESS_CACHE_MAX", "0")))
_HF_INIT_FOCK_CYCLES = max(0, int(os.environ.get("ASUKA_HF_INIT_FOCK_CYCLES", "1")))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)


_HF_DENSE_MEM_BUDGET_GIB = _env_float("ASUKA_HF_DENSE_MEM_BUDGET_GIB", 8.0)
_HF_CACHE_LOCK = threading.Lock()
_RHF_PREP_CACHE: "OrderedDict[tuple[Any, ...], tuple[Any, str, Int1eResult, Any, str, Any]]" = OrderedDict()
_RHF_GUESS_CACHE: "OrderedDict[tuple[Any, ...], Any]" = OrderedDict()


def _cache_get(cache: OrderedDict, key: tuple[Any, ...]):
    with _HF_CACHE_LOCK:
        hit = cache.get(key)
        if hit is None:
            return None
        cache.move_to_end(key)
        return hit


def _cache_put(cache: OrderedDict, key: tuple[Any, ...], val: Any, max_size: int):
    if int(max_size) <= 0:
        return
    with _HF_CACHE_LOCK:
        cache[key] = val
        cache.move_to_end(key)
        while len(cache) > int(max_size):
            cache.popitem(last=False)


def _normalize_basis_key(spec: Any) -> tuple[str, str]:
    if isinstance(spec, str):
        return ("str", str(spec).strip().lower())
    if isinstance(spec, dict):
        try:
            txt = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            txt = repr(spec)
        return ("dict", txt)
    return (type(spec).__name__, repr(spec))


def _mol_cache_key(mol: Molecule) -> tuple[Any, ...]:
    atoms = []
    for sym, xyz in mol.atoms_bohr:
        x, y, z = map(float, np.asarray(xyz, dtype=np.float64).reshape((3,)))
        atoms.append((str(sym), round(x, 12), round(y, 12), round(z, 12)))
    return (tuple(atoms), int(mol.charge), int(mol.spin), bool(mol.cart))


def _cuda_device_id_or_neg1() -> int:
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        return -1
    try:
        return int(cp.cuda.Device().id)
    except Exception:
        return -1


def _df_config_key(config: CuERIDFConfig | None) -> tuple[Any, ...]:
    cfg = CuERIDFConfig() if config is None else config
    return (
        str(cfg.backend).strip().lower(),
        str(cfg.mode).strip().lower(),
        int(cfg.threads),
        int(_cuda_device_id_or_neg1()),
    )


def _rhf_prep_key(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
    df_config: CuERIDFConfig | None,
) -> tuple[Any, ...]:
    return (
        _mol_cache_key(mol),
        _normalize_basis_key(basis_in),
        _normalize_basis_key(auxbasis),
        bool(expand_contractions),
        _df_config_key(df_config),
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
    with _HF_CACHE_LOCK:
        _RHF_PREP_CACHE.clear()
        _RHF_GUESS_CACHE.clear()


def _atom_coords_charges_bohr(mol: Molecule) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray([xyz for _sym, xyz in mol.atoms_bohr], dtype=np.float64).reshape((mol.natm, 3))
    charges = np.asarray([atomic_number(sym) for sym, _xyz in mol.atoms_bohr], dtype=np.float64)
    return coords, charges


def _unique_elements(mol: Molecule) -> list[str]:
    return sorted(set(mol.elements))


def _build_aux_basis_cart(
    mol: Molecule,
    *,
    basis_in: Any,
    auxbasis: Any,
    expand_contractions: bool,
) -> tuple[Any, str]:
    """Build (aux_basis, auxbasis_name) as a cuERI packed cart basis."""

    elements = _unique_elements(mol)
    auxbasis_name = ""

    if isinstance(auxbasis, str) and str(auxbasis).strip().lower() in ("auto", "autoaux"):
        if not isinstance(basis_in, str):
            raise ValueError("auxbasis='autoaux' requires basis to be a string name")
        auxbasis_name, aux_shells = load_autoaux_shells(str(basis_in), elements=elements)
    elif isinstance(auxbasis, str):
        auxbasis_name = str(auxbasis)
        try:
            aux_shells = load_basis_shells(auxbasis_name, elements=elements)
        except Exception:
            # Basis Set Exchange does not necessarily expose fitted aux bases as
            # standalone names (e.g. "<basis>-jkfit"). Treat common JKFIT-like
            # names as aliases for the BSE autoaux basis.
            if isinstance(basis_in, str):
                base = str(auxbasis_name).strip()
                for suf in ("-jkfit", "-jfit", "-rifit", "-ri", "-mp2fit"):
                    if base.lower().endswith(suf):
                        base = base[: -len(suf)]
                        break
                base = base or str(basis_in)
                auxbasis_name, aux_shells = load_autoaux_shells(str(base), elements=elements)
            else:
                raise
    elif isinstance(auxbasis, dict):
        auxbasis_name = "<explicit>"
        aux_shells = parse_pyscf_basis_dict(auxbasis, elements=elements)
    else:
        raise TypeError("auxbasis must be 'autoaux', a string name, or an explicit per-element basis dict")

    aux_basis = pack_cart_basis(list(mol.atoms_bohr), aux_shells, expand_contractions=bool(expand_contractions))
    return aux_basis, auxbasis_name or "<unknown>"


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

    if not bool(mol.cart):
        raise NotImplementedError("Dense ERI path does not yet support spherical AOs (cart=False); use df=True")

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

    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf_prof = profile if profile is not None else None
    scf = rhf_dense(
        int1e.S,
        int1e.hcore,
        dense.eri_mat,
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
        ao_eri=dense.eri_mat,
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

    if not bool(mol.cart):
        raise NotImplementedError("Dense ERI path does not yet support spherical AOs (cart=False); use df=True")

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

    scf_prof = profile if profile is not None else None
    scf = uhf_dense(
        int1e.S,
        int1e.hcore,
        dense.eri_mat,
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
        ao_eri=dense.eri_mat,
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

    if not bool(mol.cart):
        raise NotImplementedError("Dense ERI path does not yet support spherical AOs (cart=False); use df=True")

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

    scf_prof = profile if profile is not None else None
    scf = rohf_dense(
        int1e.S,
        int1e.hcore,
        dense.eri_mat,
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
        ao_eri=dense.eri_mat,
    )


def run_rhf_df(
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

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_df currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    prep_key = _rhf_prep_key(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_config=df_config,
    )
    prep_hit = _cache_get(_RHF_PREP_CACHE, prep_key)
    if prep_hit is None:
        # AO basis + 1e integrals
        ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
        coords, charges = _atom_coords_charges_bohr(mol)
        int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

        # DF auxiliary basis + whitened DF factors B[μ,ν,Q]
        aux_basis, auxbasis_name = _build_aux_basis_cart(
            mol,
            basis_in=basis_in,
            auxbasis=auxbasis,
            expand_contractions=bool(expand_contractions),
        )

        df_prof = None
        if profile is not None:
            df_prof = profile.setdefault("df_build", {})
            df_prof["cache_hit"] = False

        B, L_chol = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, config=df_config, profile=df_prof, return_L=True)
        _cache_put(
            _RHF_PREP_CACHE,
            prep_key,
            (ao_basis, str(basis_name), int1e, aux_basis, str(auxbasis_name), B, L_chol),
            max_size=int(_HF_PREP_CACHE_MAX),
        )
    else:
        _prep_tuple = prep_hit
        if len(_prep_tuple) == 7:
            ao_basis, basis_name, int1e, aux_basis, auxbasis_name, B, L_chol = _prep_tuple
        else:
            # Legacy cache entry without L_chol
            ao_basis, basis_name, int1e, aux_basis, auxbasis_name, B = _prep_tuple
            L_chol = None
        if profile is not None:
            df_prof = profile.setdefault("df_build", {})
            df_prof["cache_hit"] = True

    # Spherical AO transform (if requested)
    int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis)

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

    # SCF solve
    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
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
        df_B=B_scf,
        scf=scf,
        profile=profile,
        sph_map=sph_map,
        df_L=L_chol,
    )


def run_hf_df(
    mol: Molecule,
    *,
    method: str = "rhf",
    backend: str = "cuda",
    df: bool = True,
    guess: Any | None = None,
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    **kwargs,
) -> RHFDFRunResult | UHFDFRunResult | ROHFDFRunResult:
    """Unified HF driver (RHF/UHF/ROHF) over (backend, df) switches.

    Notes
    -----
    - `df=True`: DF-HF via whitened DF factors `B[μ,ν,Q]`.
    - `df=False`: dense AO-ERI HF (`ao_eri` is populated in the result; `df_B=None`).
    - `backend="cuda"`: build DF factors on GPU (CuPy).
    - `backend="cpu"`: build DF factors on CPU (requires `asuka.cueri._eri_rys_cpu`).
    """

    method_s = str(method).strip().lower()
    backend_s = str(backend).strip().lower()
    if method_s not in {"rhf", "uhf", "rohf"}:
        raise ValueError("method must be one of: 'rhf', 'uhf', 'rohf'")
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    if mo_coeff0 is None and dm0 is None and guess is not None:
        scf_guess = getattr(guess, "scf", guess)
        try:
            mo_coeff0 = getattr(scf_guess, "mo_coeff", None)
        except Exception:
            mo_coeff0 = None

    if not bool(df):
        dense_kwargs = dict(kwargs)
        # Ignore DF-only knobs if users pass them through generic wrappers.
        for key in (
            "auxbasis",
            "df_config",
            "df_threads",
            "jk_mode",
            "k_engine",
            "k_q_block",
            "cublas_math_mode",
            "ao_basis",
            "aux_basis",
            "df_backend",
            "df_mode",
            "df_aux_block_naux",
            "L_metric",
        ):
            dense_kwargs.pop(key, None)
        if method_s == "rhf":
            return run_rhf_dense(mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs)
        if method_s == "uhf":
            return run_uhf_dense(mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs)
        return run_rohf_dense(mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs)

    if backend_s == "cuda":
        if method_s == "rhf":
            return run_rhf_df(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)
        if method_s == "uhf":
            return run_uhf_df(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)
        return run_rohf_df(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)

    if method_s == "rhf":
        return run_rhf_df_cpu(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)
    if method_s == "uhf":
        return run_uhf_df_cpu(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)
    return run_rohf_df_cpu(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)


def run_hf(
    mol: Molecule,
    *,
    method: str = "rhf",
    backend: str = "cuda",
    df: bool = True,
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

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_df_cpu currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis

    # AO basis + 1e integrals
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    # DF auxiliary basis + whitened DF factors B[μ,ν,Q]
    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build_cpu", {})
    B = build_df_B_from_cueri_packed_bases_cpu(ao_basis, aux_basis, threads=int(df_threads), profile=df_prof)

    # Spherical AO transform (if requested)
    int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis)

    # SCF solve on CPU.
    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
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

    return RHFDFRunResult(
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
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with DF factors built on CPU via cuERI-CPU."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build_cpu", {})
    B = build_df_B_from_cueri_packed_bases_cpu(ao_basis, aux_basis, threads=int(df_threads), profile=df_prof)

    # Spherical AO transform (if requested)
    int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis)

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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return UHFDFRunResult(
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
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with DF factors built on CPU via cuERI-CPU."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_df_cpu requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build_cpu", {})
    B = build_df_B_from_cueri_packed_bases_cpu(ao_basis, aux_basis, threads=int(df_threads), profile=df_prof)

    # Spherical AO transform (if requested)
    int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis)

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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return ROHFDFRunResult(
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
    )


def run_uhf_df(
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
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> UHFDFRunResult:
    """Run UHF with DF integrals from cuERI."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis

    # AO basis + 1e integrals
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    # DF auxiliary basis + whitened DF factors B[μ,ν,Q]
    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build", {})

    B = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, config=df_config, profile=df_prof)

    # Spherical AO transform (if requested)
    int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis)

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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return UHFDFRunResult(
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
    )


def run_rohf_df(
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
    dm0: Any | None = None,
    mo_coeff0: Any | None = None,
    profile: dict | None = None,
) -> ROHFDFRunResult:
    """Run ROHF with DF integrals from cuERI."""

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_df requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis

    # AO basis + 1e integrals
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    # DF auxiliary basis + whitened DF factors B[μ,ν,Q]
    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build", {})

    B = build_df_B_from_cueri_packed_bases(ao_basis, aux_basis, config=df_config, profile=df_prof)

    # Spherical AO transform (if requested)
    int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis)

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
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        profile=scf_prof,
    )

    return ROHFDFRunResult(
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
    )


__all__ = [
    "clear_hf_frontend_caches",
    "RHFDFRunResult",
    "ROHFDFRunResult",
    "UHFDFRunResult",
    "run_hf",
    "run_hf_df",
    "run_rhf_dense",
    "run_rhf_df_cpu",
    "run_rhf_df",
    "run_rhf",
    "run_rohf_dense",
    "run_rohf_df_cpu",
    "run_rohf_df",
    "run_rohf",
    "run_uhf_dense",
    "run_uhf_df_cpu",
    "run_uhf_df",
    "run_uhf",
]
