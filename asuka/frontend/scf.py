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
import json
import os
import threading
from typing import Any, TYPE_CHECKING

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
    """If ``mol.cart=False``, transform int1e and B to spherical AOs.

    Returns ``(int1e, B, sph_map_or_None)``.
    """
    if bool(mol.cart):
        return int1e, B, None

    from asuka.integrals.cart2sph import (  # noqa: PLC0415
        AOSphericalTransform,
        build_cart2sph_matrix,
        compute_sph_layout_from_cart_basis,
        transform_1e_cart_to_sph,
        transform_df_B_cart_to_sph,
    )

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    if int(shell_l.size) and int(np.max(shell_l)) > 5:
        raise ValueError(
            "Spherical AO transform supports basis shells up to l<=5. "
            f"Got max(shell_l)={int(np.max(shell_l))}. Use cart=True for higher angular momentum."
        )
    shell_ao_start_cart = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    shell_ao_start_sph, nao_sph = compute_sph_layout_from_cart_basis(ao_basis)
    nao_cart = int(int1e.S.shape[0])

    T = build_cart2sph_matrix(shell_l, shell_ao_start_cart, shell_ao_start_sph, nao_cart, nao_sph)

    S_sph = transform_1e_cart_to_sph(int1e.S, T)
    T_kin_sph = transform_1e_cart_to_sph(int1e.T, T)
    V_sph = transform_1e_cart_to_sph(int1e.V, T)
    int1e_sph = Int1eResult(S=S_sph, T=T_kin_sph, V=V_sph)

    if B is not None:
        B_sph = transform_df_B_cart_to_sph(
            B,
            T,
            shell_l=shell_l,
            shell_ao_start_cart=shell_ao_start_cart,
            shell_ao_start_sph=shell_ao_start_sph,
            out_layout=str(df_B_layout),
        )
    else:
        B_sph = None

    return int1e_sph, B_sph, AOSphericalTransform(T_c2s=T, nao_cart=nao_cart, nao_sph=nao_sph)


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


def _init_guess_dm_atom_hcore_cart(
    mol: Molecule,
    *,
    ao_basis,
    int1e_cart: Int1eResult,
) -> np.ndarray:
    """Build a SAD-like initial density from per-atom core-Hamiltonian blocks.

    This is not a full SAD (which typically uses pretabulated atomic HF/DFT
    densities). Instead, we:
    - partition AOs by atom (shell centers)
    - diagonalize the per-atom (hcore, S) blocks
    - fill the lowest-energy atomic orbitals globally to match nelec

    The resulting density is block-diagonal in the AO ordering and provides a
    more localized starting guess than a full-molecule hcore diagonalization.
    """

    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.hf.local_thc_partition import map_shells_to_atoms  # noqa: PLC0415

    S = np.asarray(int1e_cart.S, dtype=np.float64)
    h = np.asarray(int1e_cart.hcore, dtype=np.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("int1e_cart must have (nao,nao) S/hcore")

    coords, _charges = _atom_coords_charges_bohr(mol)
    _sh2a, atom_to_shells = map_shells_to_atoms(np.asarray(ao_basis.shell_cxyz), coords)

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()
    if shell_l.shape != shell_start.shape:
        raise ValueError("ao_basis.shell_l and shell_ao_start must have identical shape")
    if int(shell_l.size) == 0:
        return np.zeros((nao, nao), dtype=np.float64)

    nfn_shell = np.asarray([ncart(int(l)) for l in shell_l.tolist()], dtype=np.int32)

    # Collect per-atom eigenpairs (eps, idx_global, coeff_vector).
    orb_eps: list[float] = []
    orb_idx: list[np.ndarray] = []
    orb_c: list[np.ndarray] = []

    # Symmetrize inputs (defensive; should already be symmetric).
    S = 0.5 * (S + S.T)
    h = 0.5 * (h + h.T)

    for ia in range(int(mol.natm)):
        shells = atom_to_shells[int(ia)]
        if not shells:
            continue
        idx: list[int] = []
        for sh in shells:
            s0 = int(shell_start[int(sh)])
            n = int(nfn_shell[int(sh)])
            idx.extend(range(s0, s0 + n))
        if not idx:
            continue
        idx_np = np.asarray(idx, dtype=np.int32)

        Sblk = np.asarray(S[np.ix_(idx_np, idx_np)], dtype=np.float64)
        hblk = np.asarray(h[np.ix_(idx_np, idx_np)], dtype=np.float64)
        Sblk = 0.5 * (Sblk + Sblk.T)
        hblk = 0.5 * (hblk + hblk.T)

        # Symmetric orthogonalization for this atom block.
        w, U = np.linalg.eigh(Sblk)
        w = np.asarray(w, dtype=np.float64)
        U = np.asarray(U, dtype=np.float64)
        wmax = float(np.max(w)) if int(w.size) else 0.0
        if wmax <= 0.0:
            continue
        keep = w > (1e-12 * wmax)
        if not bool(np.any(keep)):
            continue
        wk = w[keep]
        Uk = U[:, keep]
        Xk = Uk / np.sqrt(wk)[None, :]  # (n, m), with Xk.T S Xk = I
        Fp = Xk.T @ hblk @ Xk
        Fp = 0.5 * (Fp + Fp.T)
        e, Cp = np.linalg.eigh(Fp)
        e = np.asarray(e, dtype=np.float64).ravel()
        Cp = np.asarray(Cp, dtype=np.float64)
        C = Xk @ Cp  # (n, m), S-orthonormal columns

        for j in range(int(C.shape[1])):
            orb_eps.append(float(e[int(j)]))
            orb_idx.append(idx_np)
            orb_c.append(np.ascontiguousarray(C[:, int(j)], dtype=np.float64))

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")
    nocc = int(nelec // 2)
    if int(len(orb_eps)) < int(nocc):
        raise RuntimeError("atom-hcore guess produced too few orbitals to fill nelec")

    order = np.argsort(np.asarray(orb_eps, dtype=np.float64), kind="stable")
    D = np.zeros((nao, nao), dtype=np.float64)
    for k in range(int(nocc)):
        j = int(order[int(k)])
        idx_np = orb_idx[j]
        c = orb_c[j]
        D[np.ix_(idx_np, idx_np)] += 2.0 * np.outer(c, c)
    return 0.5 * (D + D.T)
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
    ao_basis: Any = None,
) -> tuple[Any, str]:
    """Build (aux_basis, auxbasis_name) as a cuERI packed cart basis.

    When *auxbasis* is an RICD alias (``"ricd"``, ``"acd"``, ``"accd"``) or an
    :class:`~asuka.integrals.ricd_types.RICDOptions` instance, the auxiliary
    basis is generated on-the-fly from the orbital basis via the aCD/acCD
    Cholesky decomposition.  In that case *ao_basis* **must** be provided.
    """

    # --- RICD on-the-fly auxiliary basis generation ---
    from asuka.integrals.ricd_types import is_ricd_request  # noqa: PLC0415

    if is_ricd_request(auxbasis):
        if ao_basis is None:
            raise ValueError(
                "ao_basis must be provided when using RICD auxiliary basis generation "
                "(auxbasis='ricd'/'acd'/'accd' or an RICDOptions instance)"
            )
        from asuka.integrals.ricd_types import normalize_ricd_options  # noqa: PLC0415
        from asuka.integrals.ricd_builder import build_ricd_aux_basis  # noqa: PLC0415

        opts = normalize_ricd_options(auxbasis)
        atoms_bohr = list(mol.atoms_bohr)
        gen = build_ricd_aux_basis(ao_basis, atoms_bohr, options=opts)
        return gen.packed_basis, gen.basis_name

    # --- Standard auxiliary basis paths ---
    elements = _unique_elements(mol)
    auxbasis_name = ""

    def _autoaux_basis_name_hint() -> str | None:
        """Return a basis name usable for BSE autoaux.

        `auxbasis='autoaux'` normally requires `basis_in` to be a string name.
        For workflows that pass an explicit basis dict (e.g., imported from
        external codes), callers may stash the original basis name into
        `mol.results['basis_name']` to keep using AutoAux.
        """

        if isinstance(basis_in, str):
            base = str(basis_in).strip()
            return base or None
        try:
            hint = mol.results.get("basis_name")
        except Exception:
            hint = None
        if isinstance(hint, str):
            hint_s = str(hint).strip()
            return hint_s or None
        return None

    if isinstance(auxbasis, str) and str(auxbasis).strip().lower() in ("auto", "autoaux"):
        base = _autoaux_basis_name_hint()
        if base is None:
            raise ValueError(
                "auxbasis='autoaux' requires basis to be a string name "
                "(or set mol.results['basis_name'] to the corresponding basis name)"
            )
        auxbasis_name, aux_shells = load_autoaux_shells(str(base), elements=elements)
    elif isinstance(auxbasis, str):
        auxbasis_name = str(auxbasis)
        try:
            aux_shells = load_basis_shells(auxbasis_name, elements=elements)
        except Exception:
            # Basis Set Exchange does not necessarily expose fitted aux bases as
            # standalone names (e.g. "<basis>-jkfit"). Treat common JKFIT-like
            # names as aliases for the BSE autoaux basis.
            base = str(auxbasis_name).strip()
            for suf in ("-jkfit", "-jfit", "-rifit", "-ri", "-mp2fit"):
                if base.lower().endswith(suf):
                    base = base[: -len(suf)]
                    break
            base = base or (_autoaux_basis_name_hint() or "")
            if base:
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

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_df currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis

    df_layout_s = str(df_layout).strip().lower()
    if df_layout_s not in {"mnq", "qmn"}:
        raise ValueError("df_layout must be one of: 'mnQ', 'Qmn'")
    df_build_layout = df_layout_s
    df_ao_rep = "cart" if bool(mol.cart) else "sph"
    prep_key = _rhf_prep_key(
        mol,
        basis_in=basis_in,
        auxbasis=auxbasis,
        expand_contractions=bool(expand_contractions),
        df_config=df_config,
        df_layout_build=str(df_build_layout),
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
            ao_basis=ao_basis,
        )

        df_prof = None
        if profile is not None:
            df_prof = profile.setdefault("df_build", {})
            df_prof["cache_hit"] = False

        B, L_chol = build_df_B_from_cueri_packed_bases(
            ao_basis,
            aux_basis,
            config=df_config,
            layout=str(df_build_layout),
            ao_rep=str(df_ao_rep),
            profile=df_prof,
            return_L=True,
        )
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
    if bool(mol.cart):
        int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis, df_B_layout=str(df_layout))
    else:
        int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout=str(df_layout))
        B_scf = B
    # For large bases, building and transforming DF factors can leave very large
    # freed blocks cached in CuPy's default memory pool, inflating driver-visible
    # VRAM and triggering avoidable OOM/non-deterministic failures downstream.
    #
    # In the spherical path, we also no longer need the Cartesian B tensor after
    # the transform. Drop the reference early so its blocks are eligible for pool
    # release/reuse.
    try:  # pragma: no cover (best-effort memory hygiene)
        if B is not None and B is not B_scf:
            del B
        import cupy as cp  # noqa: PLC0415

        if isinstance(B_scf, cp.ndarray) and int(getattr(B_scf, "nbytes", 0)) >= 2_000_000_000:
            cp.cuda.Device().synchronize()
            cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass

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
        cublas_math_mode=cublas_math_mode,
        dm0=dm0,
        mo_coeff0=mo_coeff0,
        init_fock_cycles=int(init_fock_cycles_i),
        profile=scf_prof,
    )

    # Optional: pack DF factors to unique AO-pair triangle (Qp layout) to reduce VRAM.
    #
    # Default behavior (CUDA): pack unless explicitly disabled via ASUKA_DF_AO_PACKED_S2=0.
    # CPU behavior: preserve the historical full mnQ/Qmn tensor unless explicitly enabled.
    if B_scf is not None:
        try:  # pragma: no cover (best-effort: keep legacy behavior if something goes wrong)
            import os as _os_dfpack  # noqa: PLC0415
            import cupy as _cp_dfpack  # noqa: PLC0415
            from asuka.integrals.df_packed_s2 import ao_packed_s2_enabled, pack_B_to_Qp  # noqa: PLC0415

            _explicit = "ASUKA_DF_AO_PACKED_S2" in _os_dfpack.environ
            _want_pack = bool(ao_packed_s2_enabled()) if _explicit else isinstance(B_scf, _cp_dfpack.ndarray)
            if bool(_want_pack) and int(getattr(B_scf, "ndim", 0)) == 3:
                layout = "mnQ" if str(df_layout_s) == "mnq" else "Qmn"
                B_scf = pack_B_to_Qp(B_scf, layout=layout, nao=int(int1e_scf.S.shape[0]))
        except Exception:
            pass

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
        df_run_config=_make_df_run_config(
            hf_method="rhf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=df_config,
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
    - `backend="cuda"`: build DF factors on GPU (CuPy).
    - `backend="cpu"`: build DF factors on CPU (requires `asuka.cueri._eri_rys_cpu`).
    """

    method_s = str(method).strip().lower()
    backend_s = str(backend).strip().lower()
    if method_s not in {"rhf", "uhf", "rohf", "rks", "uks"}:
        raise ValueError("method must be one of: 'rhf', 'uhf', 'rohf', 'rks', 'uks'")
    if backend_s not in {"cpu", "cuda"}:
        raise ValueError("backend must be 'cpu' or 'cuda'")

    if mo_coeff0 is None and dm0 is None and guess is not None:
        scf_guess = getattr(guess, "scf", guess)
        try:
            mo_coeff0 = getattr(scf_guess, "mo_coeff", None)
        except Exception:
            mo_coeff0 = None

    if two_e_backend is None:
        two_e = "df" if bool(df) else "dense"
    else:
        two_e = str(two_e_backend).strip().lower()
        if two_e not in {"df", "dense", "thc", "direct"}:
            raise ValueError("two_e_backend must be one of: 'df', 'dense', 'thc', 'direct'")

    if method_s in {"rks", "uks"}:
        if backend_s != "cuda":
            raise NotImplementedError("RKS/UKS currently require backend='cuda'")
        if two_e == "dense":
            raise NotImplementedError("RKS/UKS currently do not support two_e_backend='dense'")

        dft_kwargs = dict(kwargs)
        xc_name = dft_kwargs.pop("functional", "mn15")

        if two_e == "df":
            if method_s == "rks":
                return _with_two_e_metadata(
                    run_rks_df(mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
                    two_e_backend="df",
                )
            return _with_two_e_metadata(
                run_uks_df(mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
                two_e_backend="df",
            )

        # THC (global or local selected via thc_mode in kwargs)
        if method_s == "rks":
            return _with_two_e_metadata(
                run_rhf_thc(mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
                two_e_backend="thc",
            )
        return _with_two_e_metadata(
            run_uhf_thc(mol, functional=str(xc_name), dm0=dm0, mo_coeff0=mo_coeff0, **dft_kwargs),
            two_e_backend="thc",
        )

    if two_e == "dense":
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
            return _with_two_e_metadata(
                run_rhf_dense(mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs),
                two_e_backend="dense",
            )
        if method_s == "uhf":
            return _with_two_e_metadata(
                run_uhf_dense(mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs),
                two_e_backend="dense",
            )
        return _with_two_e_metadata(
            run_rohf_dense(mol, backend=backend_s, dm0=dm0, mo_coeff0=mo_coeff0, **dense_kwargs),
            two_e_backend="dense",
        )

    if two_e == "direct":
        if backend_s != "cuda":
            raise NotImplementedError("Direct integral backend currently requires backend='cuda'")
        direct_kwargs = dict(kwargs)
        # Ignore DF/dense-only knobs.
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
            "dense_threads",
            "dense_max_tile_bytes",
            "dense_eps_ao",
            "dense_max_l",
            "dense_mem_budget_gib",
        ):
            direct_kwargs.pop(key, None)
        if method_s == "rhf":
            out = run_rhf_direct(mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_kwargs)
            return _with_two_e_metadata(out, two_e_backend="direct", direct_jk_ctx=getattr(out, "direct_jk_ctx", None))
        if method_s == "uhf":
            out = run_uhf_direct(mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_kwargs)
            return _with_two_e_metadata(out, two_e_backend="direct", direct_jk_ctx=getattr(out, "direct_jk_ctx", None))
        out = run_rohf_direct(mol, dm0=dm0, mo_coeff0=mo_coeff0, **direct_kwargs)
        return _with_two_e_metadata(out, two_e_backend="direct", direct_jk_ctx=getattr(out, "direct_jk_ctx", None))

    if two_e == "thc":
        if backend_s != "cuda":
            raise NotImplementedError("THC backend currently requires backend='cuda'")
        if method_s == "rhf":
            return _with_two_e_metadata(
                run_rhf_thc(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs),
                two_e_backend="thc",
            )
        if method_s == "uhf":
            return _with_two_e_metadata(
                run_uhf_thc(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs),
                two_e_backend="thc",
            )
        return _with_two_e_metadata(
            run_rohf_thc(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs),
            two_e_backend="thc",
        )

    if backend_s == "cuda":
        if method_s == "rhf":
            out = run_rhf_df(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)
        elif method_s == "uhf":
            out = run_uhf_df(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)
        else:
            out = run_rohf_df(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs)

        # Optional post-SCF cleanup: reduce driver-visible peak VRAM by releasing
        # cached CUDA workspaces and returning unused CuPy pool blocks.
        #
        # Default heuristic: enable when VRAM telemetry is on (benchmarking /
        # debugging) or when a GPU memory cap is set.
        try:
            import os as _os_scf  # noqa: PLC0415

            _flag = _os_scf.environ.get("ASUKA_FLUSH_GPU_POOL_AFTER_SCF")
            if _flag is None:
                def _env_true(name: str) -> bool:
                    v = _os_scf.environ.get(name)
                    if v is None:
                        return False
                    return str(v).strip().lower() not in ("0", "false", "no", "off", "")

                _flag = "1" if (_env_true("ASUKA_VRAM_DEBUG") or _env_true("ASUKA_GPU_MEM_CAP_GB")) else "0"
            _v = str(_flag).strip().lower()
            if _v not in ("0", "false", "no", "off", ""):
                import cupy as _cp_scf  # noqa: PLC0415
                from asuka.hf import df_jk as _df_jk  # noqa: PLC0415

                _cp_scf.cuda.Device().synchronize()
                try:
                    _df_jk.release_cuda_ext_workspace_cache()
                except Exception:
                    pass
                _cp_scf.get_default_memory_pool().free_all_blocks()
                try:
                    _cp_scf.get_default_pinned_memory_pool().free_all_blocks()
                except Exception:
                    pass
        except Exception:
            pass

        return _with_two_e_metadata(out, two_e_backend="df")

    if method_s == "rhf":
        return _with_two_e_metadata(run_rhf_df_cpu(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs), two_e_backend="df")
    if method_s == "uhf":
        return _with_two_e_metadata(run_uhf_df_cpu(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs), two_e_backend="df")
    return _with_two_e_metadata(run_rohf_df_cpu(mol, dm0=dm0, mo_coeff0=mo_coeff0, **kwargs), two_e_backend="df")


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
    from asuka.xc.functional import get_functional
    from asuka.density.grids_device import make_becke_grid_device

    if int(mol.spin) != 0:
        raise NotImplementedError("run_rks_df currently supports only closed-shell molecules (spin=0)")

    xc_spec = get_functional(functional)

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RKS requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis

    df_layout_s = str(df_layout).strip().lower()
    if df_layout_s not in {"mnq", "qmn"}:
        raise ValueError("df_layout must be one of: 'mnQ', 'Qmn'")
    df_build_layout = df_layout_s
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

    # AO basis + 1e integrals (reuse existing infrastructure)
    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    # DF auxiliary basis + whitened DF factors
    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol, basis_in=basis_in, auxbasis=auxbasis, expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build", {})

    B, L_chol = build_df_B_from_cueri_packed_bases(
        ao_basis, aux_basis, config=df_config, layout=str(df_build_layout),
        ao_rep=str(df_ao_rep), profile=df_prof, return_L=True,
    )

    # Spherical AO transform
    if bool(mol.cart):
        int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis, df_B_layout=str(df_layout))
    else:
        int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout=str(df_layout))
        B_scf = B

    # Build DFT grid on device (or use caller-supplied grid)
    import cupy as _cp_grid
    if xc_grid_coords is not None and xc_grid_weights is not None:
        grid_coords = _cp_grid.asarray(xc_grid_coords, dtype=_cp_grid.float64)
        grid_weights = _cp_grid.asarray(xc_grid_weights, dtype=_cp_grid.float64)
    else:
        grid_coords, grid_weights = make_becke_grid_device(
            mol, radial_n=int(grid_radial_n), angular_n=int(grid_angular_n),
            radial_scheme="treutler",
        )

    # Spherical transform for XC builder
    xc_sph_transform = None
    if not bool(mol.cart) and sph_map is not None:
        import cupy as _cp_xc
        if hasattr(sph_map, "T_c2s"):
            xc_sph_transform = _cp_xc.asarray(sph_map.T_c2s, dtype=_cp_xc.float64)
        elif hasattr(sph_map, "T_matrix"):
            xc_sph_transform = _cp_xc.asarray(sph_map.T_matrix, dtype=_cp_xc.float64)
        elif isinstance(sph_map, tuple) and len(sph_map) >= 1:
            xc_sph_transform = _cp_xc.asarray(sph_map[0], dtype=_cp_xc.float64)

    # SCF solve
    init_fock_cycles_i = int(_HF_INIT_FOCK_CYCLES) if init_fock_cycles is None else max(0, int(init_fock_cycles))
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

    return RHFDFRunResult(
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
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build_cpu", {})
    B, L_chol = build_df_B_from_cueri_packed_bases_cpu(
        ao_basis,
        aux_basis,
        threads=int(df_threads),
        profile=df_prof,
        return_L=True,
    )

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
        df_L=L_chol,
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
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build_cpu", {})
    B, L_chol = build_df_B_from_cueri_packed_bases_cpu(
        ao_basis,
        aux_basis,
        threads=int(df_threads),
        profile=df_prof,
        return_L=True,
    )

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
        cublas_math_mode=cublas_math_mode,
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
        df_L=L_chol,
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
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build_cpu", {})
    B, L_chol = build_df_B_from_cueri_packed_bases_cpu(
        ao_basis,
        aux_basis,
        threads=int(df_threads),
        profile=df_prof,
        return_L=True,
    )

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
        cublas_math_mode=cublas_math_mode,
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
        df_L=L_chol,
    )


def run_uhf_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
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

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    basis_in = mol.basis if basis is None else basis

    df_layout_s = str(df_layout).strip().lower()
    if df_layout_s not in {"mnq", "qmn"}:
        raise ValueError("df_layout must be one of: 'mnQ', 'Qmn'")
    df_build_layout = df_layout_s
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

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
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build", {})

    B, L_chol = build_df_B_from_cueri_packed_bases(
        ao_basis,
        aux_basis,
        config=df_config,
        layout=str(df_build_layout),
        ao_rep=str(df_ao_rep),
        profile=df_prof,
        return_L=True,
    )

    # Spherical AO transform (if requested)
    if bool(mol.cart):
        int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis, df_B_layout=str(df_layout))
    else:
        int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout=str(df_layout))
        B_scf = B

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

    if B_scf is not None:
        try:  # pragma: no cover
            import os as _os_dfpack  # noqa: PLC0415
            import cupy as _cp_dfpack  # noqa: PLC0415
            from asuka.integrals.df_packed_s2 import ao_packed_s2_enabled, pack_B_to_Qp  # noqa: PLC0415

            _explicit = "ASUKA_DF_AO_PACKED_S2" in _os_dfpack.environ
            _want_pack = bool(ao_packed_s2_enabled()) if _explicit else isinstance(B_scf, _cp_dfpack.ndarray)
            if bool(_want_pack) and int(getattr(B_scf, "ndim", 0)) == 3:
                layout = "mnQ" if str(df_layout_s) == "mnq" else "Qmn"
                B_scf = pack_B_to_Qp(B_scf, layout=layout, nao=int(int1e_scf.S.shape[0]))
        except Exception:
            pass

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
        df_L=L_chol,
        df_run_config=_make_df_run_config(
            hf_method="uhf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=df_config,
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


def run_uks_df(
    mol: Molecule,
    *,
    functional: str = "mn15",
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: "CuERIDFConfig | None" = None,
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
    from asuka.xc.functional import get_functional
    from asuka.density.grids_device import make_becke_grid_device

    xc_spec = get_functional(functional)
    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)

    basis_in = mol.basis if basis is None else basis
    df_layout_s = str(df_layout).strip().lower()
    if df_layout_s not in {"mnq", "qmn"}:
        raise ValueError("df_layout must be one of: 'mnQ', 'Qmn'")
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

    ao_basis, basis_name = build_ao_basis_cart(mol, basis=basis_in, expand_contractions=bool(expand_contractions))
    coords, charges = _atom_coords_charges_bohr(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)

    aux_basis, auxbasis_name = _build_aux_basis_cart(
        mol, basis_in=basis_in, auxbasis=auxbasis, expand_contractions=bool(expand_contractions),
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build", {})

    B, L_chol = build_df_B_from_cueri_packed_bases(
        ao_basis, aux_basis, config=df_config, layout=str(df_layout_s),
        ao_rep=str(df_ao_rep), profile=df_prof, return_L=True,
    )

    if bool(mol.cart):
        int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis, df_B_layout=str(df_layout))
    else:
        int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout=str(df_layout))
        B_scf = B

    # Build DFT grid (or use caller-supplied)
    import cupy as _cp_grid
    if xc_grid_coords is not None and xc_grid_weights is not None:
        grid_coords = _cp_grid.asarray(xc_grid_coords, dtype=_cp_grid.float64)
        grid_weights = _cp_grid.asarray(xc_grid_weights, dtype=_cp_grid.float64)
    else:
        grid_coords, grid_weights = make_becke_grid_device(
            mol, radial_n=int(grid_radial_n), angular_n=int(grid_angular_n),
            radial_scheme="treutler",
        )

    xc_sph_transform = None
    if not bool(mol.cart) and sph_map is not None:
        import cupy as _cp_xc
        if hasattr(sph_map, "T_c2s"):
            xc_sph_transform = _cp_xc.asarray(sph_map.T_c2s, dtype=_cp_xc.float64)
        elif hasattr(sph_map, "T_matrix"):
            xc_sph_transform = _cp_xc.asarray(sph_map.T_matrix, dtype=_cp_xc.float64)
        elif isinstance(sph_map, tuple) and len(sph_map) >= 1:
            xc_sph_transform = _cp_xc.asarray(sph_map[0], dtype=_cp_xc.float64)

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
        df_L=L_chol,
        df_run_config=_make_df_run_config(
            hf_method="uks",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=df_config,
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


def run_rohf_df(
    mol: Molecule,
    *,
    basis: Any | None = None,
    auxbasis: Any = "autoaux",
    df_config: CuERIDFConfig | None = None,
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

    nalpha, nbeta = _nalpha_nbeta_from_mol(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_df requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis

    df_layout_s = str(df_layout).strip().lower()
    if df_layout_s not in {"mnq", "qmn"}:
        raise ValueError("df_layout must be one of: 'mnQ', 'Qmn'")
    df_build_layout = df_layout_s
    df_ao_rep = "cart" if bool(mol.cart) else "sph"

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
        ao_basis=ao_basis,
    )

    df_prof = None
    if profile is not None:
        df_prof = profile.setdefault("df_build", {})

    B, L_chol = build_df_B_from_cueri_packed_bases(
        ao_basis,
        aux_basis,
        config=df_config,
        layout=str(df_build_layout),
        ao_rep=str(df_ao_rep),
        profile=df_prof,
        return_L=True,
    )

    # Spherical AO transform (if requested)
    if bool(mol.cart):
        int1e_scf, B_scf, sph_map = _apply_sph_transform(mol, int1e, B, ao_basis, df_B_layout=str(df_layout))
    else:
        int1e_scf, _B_unused, sph_map = _apply_sph_transform(mol, int1e, None, ao_basis, df_B_layout=str(df_layout))
        B_scf = B

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

    if B_scf is not None:
        try:  # pragma: no cover
            import os as _os_dfpack  # noqa: PLC0415
            import cupy as _cp_dfpack  # noqa: PLC0415
            from asuka.integrals.df_packed_s2 import ao_packed_s2_enabled, pack_B_to_Qp  # noqa: PLC0415

            _explicit = "ASUKA_DF_AO_PACKED_S2" in _os_dfpack.environ
            _want_pack = bool(ao_packed_s2_enabled()) if _explicit else isinstance(B_scf, _cp_dfpack.ndarray)
            if bool(_want_pack) and int(getattr(B_scf, "ndim", 0)) == 3:
                layout = "mnQ" if str(df_layout_s) == "mnq" else "Qmn"
                B_scf = pack_B_to_Qp(B_scf, layout=layout, nao=int(int1e_scf.S.shape[0]))
        except Exception:
            pass

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
        df_L=L_chol,
        df_run_config=_make_df_run_config(
            hf_method="rohf",
            basis=basis_in,
            auxbasis=auxbasis,
            df_config=df_config,
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
