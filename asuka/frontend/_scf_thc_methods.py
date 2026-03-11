from __future__ import annotations

# THC frontend implementations extracted from frontend/scf.py.
# This module imports `frontend.scf` lazily at module import time and aliases
# shared helpers/results to avoid behavior changes while shrinking scf.py.
from . import scf as _scf_mod
from ._scf_run_config import make_thc_run_config as _make_thc_run_config
from ._scf_xc_runtime import resolve_xc_runtime as _resolve_xc_runtime

np = _scf_mod.np
Molecule = _scf_mod.Molecule
Any = _scf_mod.Any
CuERIDFConfig = _scf_mod.CuERIDFConfig

build_ao_basis_cart = _scf_mod.build_ao_basis_cart
build_int1e_cart = _scf_mod.build_int1e_cart

_apply_sph_transform = _scf_mod._apply_sph_transform
_atom_coords_charges_bohr = _scf_mod._atom_coords_charges_bohr
_build_aux_basis_cart = _scf_mod._build_aux_basis_cart
_init_guess_dm_atom_hcore_cart = _scf_mod._init_guess_dm_atom_hcore_cart

_cache_get = _scf_mod._cache_get
_cache_put = _scf_mod._cache_put
_copy_mo_coeff_for_cache = _scf_mod._copy_mo_coeff_for_cache
_rhf_guess_key = _scf_mod._rhf_guess_key
_nalpha_nbeta_from_mol = _scf_mod._nalpha_nbeta_from_mol

_resolve_thc_grid_and_dvr_basis = _scf_mod._resolve_thc_grid_and_dvr_basis
_resolve_thc_mp_store_policy = _scf_mod._resolve_thc_mp_store_policy
_coerce_local_thc_config = _scf_mod._coerce_local_thc_config

_HF_INIT_FOCK_CYCLES = _scf_mod._HF_INIT_FOCK_CYCLES
_HF_GUESS_CACHE_MAX = _scf_mod._HF_GUESS_CACHE_MAX
_RHF_GUESS_CACHE = _scf_mod._RHF_GUESS_CACHE

RHFDFRunResult = _scf_mod.RHFDFRunResult
UHFDFRunResult = _scf_mod.UHFDFRunResult
ROHFDFRunResult = _scf_mod.ROHFDFRunResult


def run_rhf_thc_impl(
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
    xc_spec, xc_grid_coords, xc_grid_weights, xc_sph_transform = _resolve_xc_runtime(
        functional=functional,
        mol=mol,
        sph_map=sph_map,
        grid_radial_n=int(grid_radial_n),
        grid_angular_n=int(grid_angular_n),
    )

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
    grid_kind_s, dvr_basis_cart = _resolve_thc_grid_and_dvr_basis(
        mol=mol,
        aux_basis=aux_basis,
        thc_grid_kind=str(thc_grid_kind),
        thc_dvr_basis=thc_dvr_basis,
        expand_contractions=bool(expand_contractions),
        build_ao_basis_cart_fn=build_ao_basis_cart,
    )

    # Build THC factors (global THC or local-THC/LS-THC-style blocks).
    thc_mode_s = str(thc_mode).strip().lower()
    thc_prof = profile.setdefault("thc_build", {}) if profile is not None else None

    thc_mp_mode_s, store_Z_eff = _resolve_thc_mp_store_policy(
        thc_mp_mode=str(thc_mp_mode),
        thc_store_Z=thc_store_Z,
        thc_profile=thc_prof,
    )

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

        lcfg = _coerce_local_thc_config(
            thc_local_config=thc_local_config,
            local_config_type=LocalTHCConfig,
        )

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


def run_uhf_thc_impl(
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
    xc_spec, xc_grid_coords, xc_grid_weights, xc_sph_transform = _resolve_xc_runtime(
        functional=functional,
        mol=mol,
        sph_map=sph_map,
        grid_radial_n=int(grid_radial_n),
        grid_angular_n=int(grid_angular_n),
    )

    # Grid kind selection (Becke vs DVR).
    grid_kind_s, dvr_basis_cart = _resolve_thc_grid_and_dvr_basis(
        mol=mol,
        aux_basis=aux_basis,
        thc_grid_kind=str(thc_grid_kind),
        thc_dvr_basis=thc_dvr_basis,
        expand_contractions=bool(expand_contractions),
        build_ao_basis_cart_fn=build_ao_basis_cart,
    )

    # Build THC factors (global THC or local-THC/LS-THC-style blocks).
    thc_mode_s = str(thc_mode).strip().lower()
    thc_prof = profile.setdefault("thc_build", {}) if profile is not None else None

    thc_mp_mode_s, store_Z_eff = _resolve_thc_mp_store_policy(
        thc_mp_mode=str(thc_mp_mode),
        thc_store_Z=thc_store_Z,
        thc_profile=thc_prof,
    )

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

        lcfg = _coerce_local_thc_config(
            thc_local_config=thc_local_config,
            local_config_type=LocalTHCConfig,
        )

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


def run_rohf_thc_impl(
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
    grid_kind_s, dvr_basis_cart = _resolve_thc_grid_and_dvr_basis(
        mol=mol,
        aux_basis=aux_basis,
        thc_grid_kind=str(thc_grid_kind),
        thc_dvr_basis=thc_dvr_basis,
        expand_contractions=bool(expand_contractions),
        build_ao_basis_cart_fn=build_ao_basis_cart,
    )

    # Build THC factors (global THC or local-THC/LS-THC-style blocks).
    thc_mode_s = str(thc_mode).strip().lower()
    thc_prof = profile.setdefault("thc_build", {}) if profile is not None else None

    thc_mp_mode_s, store_Z_eff = _resolve_thc_mp_store_policy(
        thc_mp_mode=str(thc_mp_mode),
        thc_store_Z=thc_store_Z,
        thc_profile=thc_prof,
    )

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

        lcfg = _coerce_local_thc_config(
            thc_local_config=thc_local_config,
            local_config_type=LocalTHCConfig,
        )

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
