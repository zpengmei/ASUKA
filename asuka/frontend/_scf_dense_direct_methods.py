from __future__ import annotations

from typing import Any, Callable

from asuka.hf.dense_scf import rhf_dense, rohf_dense, uhf_dense
from asuka.integrals.int1e_cart import build_int1e_cart

from .one_electron import build_ao_basis_cart


def _build_ao_int1e_problem(
    mol,
    *,
    basis_in: Any,
    expand_contractions: bool,
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
):
    ao_basis, basis_name = build_ao_basis_cart(
        mol,
        basis=basis_in,
        expand_contractions=bool(expand_contractions),
    )
    coords, charges = atom_coords_charges_bohr_fn(mol)
    int1e = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges)
    return ao_basis, str(basis_name), int1e


def _maybe_transform_dense_eri(
    mol,
    *,
    int1e: Any,
    eri_mat: Any,
    ao_basis: Any,
    apply_sph_transform_fn: Callable[..., tuple[Any, Any, Any]],
):
    if bool(mol.cart):
        return int1e, eri_mat, None
    int1e_scf, _unused_B, sph_map = apply_sph_transform_fn(mol, int1e, None, ao_basis)
    from asuka.integrals.cart2sph import transform_dense_eri_cart_to_sph  # noqa: PLC0415

    T, nao_cart, nao_sph = sph_map
    eri_scf = transform_dense_eri_cart_to_sph(eri_mat, T, nao_cart, nao_sph)
    return int1e_scf, eri_scf, sph_map


def run_rhf_dense_impl(
    mol,
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
    init_fock_cycles_default: int,
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
    build_dense_ao_eri_fn: Callable[..., Any],
    apply_sph_transform_fn: Callable[..., tuple[Any, Any, Any]],
    result_cls: Any,
):
    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_dense currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e = _build_ao_int1e_problem(
        mol,
        basis_in=basis_in,
        expand_contractions=bool(expand_contractions),
        atom_coords_charges_bohr_fn=atom_coords_charges_bohr_fn,
    )
    dense = build_dense_ao_eri_fn(
        ao_basis,
        backend=str(backend),
        dense_threads=dense_threads,
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_eps_ao=float(dense_eps_ao),
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        profile=profile,
    )
    int1e_scf, eri_mat_use, sph_map = _maybe_transform_dense_eri(
        mol,
        int1e=int1e,
        eri_mat=dense.eri_mat,
        ao_basis=ao_basis,
        apply_sph_transform_fn=apply_sph_transform_fn,
    )

    init_fock_cycles_i = int(init_fock_cycles_default) if init_fock_cycles is None else max(0, int(init_fock_cycles))
    diis_start_cycle_i = (
        int(diis_start_cycle) if diis_start_cycle is not None else (1 if int(init_fock_cycles_i) > 0 else 2)
    )
    scf = rhf_dense(
        int1e_scf.S,
        int1e_scf.hcore,
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
        profile=profile if profile is not None else None,
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<dense>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        ao_eri=eri_mat_use,
        sph_map=sph_map,
    )


def run_uhf_dense_impl(
    mol,
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
    nalpha_nbeta_from_mol_fn: Callable[[Any], tuple[int, int]],
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
    build_dense_ao_eri_fn: Callable[..., Any],
    apply_sph_transform_fn: Callable[..., tuple[Any, Any, Any]],
    result_cls: Any,
):
    nalpha, nbeta = nalpha_nbeta_from_mol_fn(mol)
    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e = _build_ao_int1e_problem(
        mol,
        basis_in=basis_in,
        expand_contractions=bool(expand_contractions),
        atom_coords_charges_bohr_fn=atom_coords_charges_bohr_fn,
    )
    dense = build_dense_ao_eri_fn(
        ao_basis,
        backend=str(backend),
        dense_threads=dense_threads,
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_eps_ao=float(dense_eps_ao),
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        profile=profile,
    )
    int1e_scf, eri_mat_use, sph_map = _maybe_transform_dense_eri(
        mol,
        int1e=int1e,
        eri_mat=dense.eri_mat,
        ao_basis=ao_basis,
        apply_sph_transform_fn=apply_sph_transform_fn,
    )
    scf = uhf_dense(
        int1e_scf.S,
        int1e_scf.hcore,
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
        profile=profile if profile is not None else None,
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<dense>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        ao_eri=eri_mat_use,
        sph_map=sph_map,
    )


def run_rohf_dense_impl(
    mol,
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
    nalpha_nbeta_from_mol_fn: Callable[[Any], tuple[int, int]],
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
    build_dense_ao_eri_fn: Callable[..., Any],
    apply_sph_transform_fn: Callable[..., tuple[Any, Any, Any]],
    result_cls: Any,
):
    nalpha, nbeta = nalpha_nbeta_from_mol_fn(mol)
    if int(nalpha) < int(nbeta):
        raise ValueError("run_rohf_dense requires spin >= 0 (nalpha >= nbeta)")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e = _build_ao_int1e_problem(
        mol,
        basis_in=basis_in,
        expand_contractions=bool(expand_contractions),
        atom_coords_charges_bohr_fn=atom_coords_charges_bohr_fn,
    )
    dense = build_dense_ao_eri_fn(
        ao_basis,
        backend=str(backend),
        dense_threads=dense_threads,
        dense_max_tile_bytes=int(dense_max_tile_bytes),
        dense_eps_ao=float(dense_eps_ao),
        dense_max_l=dense_max_l,
        dense_mem_budget_gib=dense_mem_budget_gib,
        profile=profile,
    )
    int1e_scf, eri_mat_use, sph_map = _maybe_transform_dense_eri(
        mol,
        int1e=int1e,
        eri_mat=dense.eri_mat,
        ao_basis=ao_basis,
        apply_sph_transform_fn=apply_sph_transform_fn,
    )
    scf = rohf_dense(
        int1e_scf.S,
        int1e_scf.hcore,
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
        profile=profile if profile is not None else None,
    )

    return result_cls(
        mol=mol,
        basis_name=str(basis_name),
        auxbasis_name="<dense>",
        ao_basis=ao_basis,
        aux_basis=None,
        int1e=int1e_scf,
        df_B=None,
        scf=scf,
        profile=profile,
        ao_eri=eri_mat_use,
        sph_map=sph_map,
    )


def run_rhf_direct_impl(
    mol,
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
    init_fock_cycles_default: int,
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
    result_cls: Any,
):
    if int(mol.spin) != 0:
        raise NotImplementedError("run_rhf_direct currently supports only closed-shell molecules (spin=0)")

    nelec = int(mol.nelectron)
    if nelec <= 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even, positive electron count")

    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e = _build_ao_int1e_problem(
        mol,
        basis_in=basis_in,
        expand_contractions=bool(expand_contractions),
        atom_coords_charges_bohr_fn=atom_coords_charges_bohr_fn,
    )
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

    init_fock_cycles_i = int(init_fock_cycles_default) if init_fock_cycles is None else max(0, int(init_fock_cycles))
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

    return result_cls(
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


def run_uhf_direct_impl(
    mol,
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
    nalpha_nbeta_from_mol_fn: Callable[[Any], tuple[int, int]],
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
    result_cls: Any,
):
    nalpha, nbeta = nalpha_nbeta_from_mol_fn(mol)
    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e = _build_ao_int1e_problem(
        mol,
        basis_in=basis_in,
        expand_contractions=bool(expand_contractions),
        atom_coords_charges_bohr_fn=atom_coords_charges_bohr_fn,
    )
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

    return result_cls(
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


def run_rohf_direct_impl(
    mol,
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
    nalpha_nbeta_from_mol_fn: Callable[[Any], tuple[int, int]],
    atom_coords_charges_bohr_fn: Callable[[Any], tuple[Any, Any]],
    result_cls: Any,
):
    nalpha, nbeta = nalpha_nbeta_from_mol_fn(mol)
    basis_in = mol.basis if basis is None else basis
    ao_basis, basis_name, int1e = _build_ao_int1e_problem(
        mol,
        basis_in=basis_in,
        expand_contractions=bool(expand_contractions),
        atom_coords_charges_bohr_fn=atom_coords_charges_bohr_fn,
    )
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

    return result_cls(
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
