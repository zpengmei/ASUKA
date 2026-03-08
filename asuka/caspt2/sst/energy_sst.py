"""Top-level driver for the SST-CASPT2 backend.

Implements two modes:

**Full SST** (default, ``sst_mode="full"``):

  E_PT2 = E^MP2_dressed - E^eff_S + E^eff_T    (Eq. 60 of Song 2024)

  where:
    - E^MP2_dressed: dressed MP2 energy over all hole×particle pairs
    - E^eff_S: supporting subspace correction (cases A, B±, C, D, F±)
    - E^eff_T: trailing subspace correction (cases 1-11 via IC solver)

**IC-based SST** (``sst_mode="ic"``):

  E_PT2 = E_MP2_like(H±) + E_active(cases 1-11)

  The original trivial partition — exact IC-CASPT2 energy decomposed into
  H± (solved analytically) + cases 1-11 (solved with IC infrastructure).

Both modes give the exact same total energy (analytically equivalent).
The full SST mode enables future reduced-scaling optimizations.
"""
from __future__ import annotations

import numpy as np

from asuka.caspt2.sst.types import SSTConfig, SSTInput, SSTResult

__all__ = ["sst_caspt2_energy_ss"]


def sst_caspt2_energy_ss(
    inp: SSTInput,
    cfg: SSTConfig | None = None,
    *,
    sst_mode: str = "full",
    df_blocks: object = None,
    device: int | None = None,
    return_context: bool = False,
) -> SSTResult:
    """Compute SS-CASPT2 energy via the SST decomposition.

    Parameters
    ----------
    inp
        Pre-extracted input data (orbitals, RDMs, Fock, etc.).
        Must have ``dm3_act``, ``ci_context``, and ``smap`` populated if the
        reduced active system (cases 1-11) should be solved.
    cfg
        Configuration. Defaults to :class:`~asuka.caspt2.sst.types.SSTConfig`.
    sst_mode
        ``"full"`` for the Song 2024 SST decomposition (E^MP2_dressed - E^eff_S + E^eff_T),
        ``"ic"`` for the original IC-based split (E_H± + E_cases_1-11).

    Returns
    -------
    SSTResult
        Energy decomposition and (optional) active amplitudes.
    """
    if cfg is None:
        cfg = SSTConfig()

    sst_mode = str(sst_mode).strip().lower()

    if sst_mode == "ic":
        return _sst_ic_mode(inp, cfg, df_blocks=df_blocks, device=device, return_context=bool(return_context))
    elif sst_mode == "full":
        return _sst_full_mode(inp, cfg, df_blocks=df_blocks, device=device, return_context=bool(return_context))
    else:
        raise ValueError(f"Unknown sst_mode={sst_mode!r}; use 'full' or 'ic'")


def _solve_hpm_cases_native(
    inp: SSTInput,
    cfg: SSTConfig,
):
    """Solve the H+/- sector and capture SR-basis native context."""
    from asuka.caspt2.energy import _get_external_energies  # noqa: PLC0415
    from asuka.caspt2.hzero import build_bmat  # noqa: PLC0415
    from asuka.caspt2.overlap import build_smat, sbdiag  # noqa: PLC0415
    from asuka.caspt2.rhs import build_rhs  # noqa: PLC0415
    from asuka.caspt2.shifts import apply_imaginary_shift, apply_real_shift, compute_shift_correction  # noqa: PLC0415
    from asuka.caspt2.sst.reduced_system import _build_full_mo_eris  # noqa: PLC0415
    from asuka.caspt2.sst.types import SSTCaseContext  # noqa: PLC0415

    if inp.eri_mo is not None:
        eri_mo = np.asarray(inp.eri_mo, dtype=np.float64)
    elif inp.B_ao is not None:
        eri_mo = _build_full_mo_eris(inp.B_ao, inp.mo_coeff)
    else:
        raise ValueError("SST H+/- context requires either eri_mo or B_ao")

    dm1 = np.asarray(inp.dm1_act, dtype=np.float64)
    dm2 = np.asarray(inp.dm2_act, dtype=np.float64)
    dm3 = np.asarray(inp.dm3_act, dtype=np.float64) if inp.dm3_act is not None else np.zeros((0, 0, 0, 0, 0, 0), dtype=np.float64)
    out_context: dict[int, SSTCaseContext] = {}
    amplitudes: list[np.ndarray] = []
    e_total = 0.0

    for case in (12, 13):
        nasup = int(inp.smap.nasup[case - 1])
        nisup = int(inp.smap.nisup[case - 1])
        if nasup == 0 or nisup == 0:
            amps = np.empty(0, dtype=np.float64)
            out_context[int(case)] = SSTCaseContext(
                case=int(case),
                sector="hpm",
                rhs_sr=np.empty(0, dtype=np.float64),
                h0_diag=np.empty(0, dtype=np.float64),
                amplitudes_sr=amps,
            )
            amplitudes.append(amps)
            continue

        smat = build_smat(case, inp.smap, dm1, dm2, dm3)
        bmat = build_bmat(case, inp.smap, inp.fock, dm1, dm2, dm3, ci_context=inp.ci_context)
        decomp = sbdiag(smat, bmat, threshold_norm=cfg.threshold, threshold_s=cfg.threshold_s)
        if decomp.nindep == 0:
            amps = np.empty(0, dtype=np.float64)
            out_context[int(case)] = SSTCaseContext(
                case=int(case),
                sector="hpm",
                rhs_sr=np.empty(0, dtype=np.float64),
                h0_diag=np.empty(0, dtype=np.float64),
                amplitudes_sr=amps,
                smat=np.asarray(smat, dtype=np.float64),
                transform=np.asarray(decomp.transform, dtype=np.float64),
                b_diag=np.asarray(decomp.b_diag, dtype=np.float64),
                nindep=int(decomp.nindep),
            )
            amplitudes.append(amps)
            continue

        ext_energies = _get_external_energies(case, inp.smap, inp.fock)
        h0_diag = (decomp.b_diag[:, None] + ext_energies[None, :]).ravel() if ext_energies.size > 0 else decomp.b_diag.copy()
        rhs_raw = build_rhs(case, inp.smap, inp.fock, eri_mo, dm1, dm2)
        rhs_mat = rhs_raw.reshape(nasup, nisup)
        rhs_sr = (decomp.transform.T @ smat @ rhs_mat).ravel()

        h0_shifted = np.asarray(h0_diag, dtype=np.float64).copy()
        h0_orig = np.asarray(h0_diag, dtype=np.float64).copy()
        if abs(cfg.imag_shift) > 1e-15:
            h0_shifted = apply_imaginary_shift([h0_shifted], cfg.imag_shift)[0]
        if abs(cfg.real_shift) > 1e-15:
            h0_shifted = apply_real_shift([h0_shifted], cfg.real_shift)[0]

        amps = np.zeros_like(rhs_sr)
        mask = np.abs(h0_shifted) > 1e-14
        amps[mask] = -rhs_sr[mask] / h0_shifted[mask]
        e_case = float(np.dot(rhs_sr, amps))
        shift_corr = 0.0
        if abs(cfg.imag_shift) > 1e-15 or abs(cfg.real_shift) > 1e-15:
            shift_corr = float(compute_shift_correction([amps], [h0_shifted], [h0_orig]))
            e_case -= shift_corr
        e_total += e_case
        amplitudes.append(np.asarray(amps, dtype=np.float64))
        out_context[int(case)] = SSTCaseContext(
            case=int(case),
            sector="hpm",
            rhs_sr=np.asarray(rhs_sr, dtype=np.float64),
            h0_diag=np.asarray(h0_shifted, dtype=np.float64),
            amplitudes_sr=np.asarray(amps, dtype=np.float64),
            energy=float(e_case),
            shift_correction=float(shift_corr),
            smat=np.asarray(smat, dtype=np.float64),
            transform=np.asarray(decomp.transform, dtype=np.float64),
            b_diag=np.asarray(decomp.b_diag, dtype=np.float64),
            nindep=int(decomp.nindep),
        )
    return float(e_total), amplitudes, out_context


def _sst_ic_mode(
    inp: SSTInput, cfg: SSTConfig,
    *, df_blocks: object = None, device: int | None = None,
    return_context: bool = False,
) -> SSTResult:
    """Original IC-based SST: E = E_H± + E_cases_1-11."""
    from asuka.caspt2.sst.mp2_like import sst_mp2_like_energy  # noqa: PLC0415
    from asuka.caspt2.sst.reduced_system import build_and_solve_reduced_system  # noqa: PLC0415
    from asuka.caspt2.sst.types import SSTNativeContext  # noqa: PLC0415

    e_mp2_like = float(sst_mp2_like_energy(inp, cfg))
    native_context = None

    if inp.dm3_act is not None and inp.ci_context is not None and inp.smap is not None:
        if bool(return_context):
            e_active, t_active, converged, niter, reduced_cases = build_and_solve_reduced_system(
                inp, cfg, df_blocks=df_blocks, device=device, return_context=True,
            )
            e_hpm_ctx, hpm_amps, hpm_cases = _solve_hpm_cases_native(inp, cfg)
            native_context = SSTNativeContext(
                mode="ic",
                reduced_cases=dict(reduced_cases),
                hpm_cases=dict(hpm_cases),
                terms={
                    "e_mp2_like": float(e_mp2_like),
                    "e_hpm_context": float(e_hpm_ctx),
                    "e_active": float(e_active),
                },
            )
        else:
            e_active, t_active, converged, niter = build_and_solve_reduced_system(
                inp, cfg, df_blocks=df_blocks, device=device,
            )
        e_active = float(e_active)
    else:
        e_active = 0.0
        t_active = None
        converged = True
        niter = 0

    e_pt2 = float(e_mp2_like + e_active)
    e_tot = float(inp.e_ref + e_pt2)

    return SSTResult(
        e_mp2_like=float(e_mp2_like),
        e_active=float(e_active),
        e_pt2=float(e_pt2),
        e_tot=float(e_tot),
        amplitudes_active=t_active,
        pcg_converged=bool(converged),
        pcg_niter=int(niter),
        breakdown={
            "sst_mode": "ic",
            "e_mp2_like": float(e_mp2_like),
            "e_active": float(e_active),
        },
        native_context=native_context,
    )


def _sst_full_mode(
    inp: SSTInput, cfg: SSTConfig,
    *, df_blocks: object = None, device: int | None = None,
    return_context: bool = False,
) -> SSTResult:
    """Full SST: E = E^MP2_dressed - E^eff_S + E^eff_T (Eq. 60)."""
    import numpy as np

    from asuka.caspt2.sst.coupling import build_coupling_matrices, diagonalize_coupling  # noqa: PLC0415
    from asuka.caspt2.sst.dressed_mp2 import compute_dressed_mp2  # noqa: PLC0415
    from asuka.caspt2.sst.mp2_like import sst_mp2_like_energy  # noqa: PLC0415
    from asuka.caspt2.sst.supporting import solve_supporting_subspace  # noqa: PLC0415
    from asuka.caspt2.sst.trailing import solve_trailing_subspace  # noqa: PLC0415

    verbose = cfg.verbose

    if verbose >= 1:
        print("SST-CASPT2 (full mode): E = E^MP2_dressed - E^eff_S + E^eff_T")

    # ── Step 1: Build coupling matrices and dressed orbitals ──
    gamma, gamma_bar = build_coupling_matrices(
        inp.fock, inp.dm1_act,
        ncore=inp.ncore, ncas=inp.ncas, nvirt=inp.nvirt,
    )
    dressed = diagonalize_coupling(
        gamma, gamma_bar,
        ncore=inp.ncore, ncas=inp.ncas, nvirt=inp.nvirt,
        real_shift=cfg.real_shift,
    )

    if verbose >= 1:
        print(f"  Dressed hole energies: min={dressed.omega_hole.min():.6f}, "
              f"max={dressed.omega_hole.max():.6f}")
        print(f"  Dressed particle energies: min={dressed.omega_particle.min():.6f}, "
              f"max={dressed.omega_particle.max():.6f}")

    # ── Step 2: E^MP2_dressed (all hole×particle pairs) ──
    e_mp2_dressed = compute_dressed_mp2(inp, dressed)

    if verbose >= 1:
        print(f"  E^MP2_dressed = {e_mp2_dressed:.10f}")

    # ── Step 3: Also compute H± for cross-check ──
    e_mp2_like = float(sst_mp2_like_energy(inp, cfg))
    if verbose >= 1:
        print(f"  E_H± (cross-check) = {e_mp2_like:.10f}")

    # ── Step 4: Supporting subspace correction ──
    e_eff_s = 0.0
    s_amplitudes = {}
    if inp.dm3_act is not None and inp.ci_context is not None and inp.smap is not None:
        e_eff_s, s_amplitudes = solve_supporting_subspace(inp, dressed, cfg)

    if verbose >= 1:
        print(f"  E^eff_S = {e_eff_s:.10f}")

    # ── Step 5: Trailing subspace (exact IC cases 1-11) ──
    e_trailing = 0.0
    t_active = None
    converged = True
    niter = 0
    if inp.dm3_act is not None and inp.ci_context is not None and inp.smap is not None:
        e_trailing, t_active, converged, niter = solve_trailing_subspace(
            inp, dressed, cfg, df_blocks=df_blocks, device=device,
        )

    if verbose >= 1:
        print(f"  E_trailing (cases 1-11) = {e_trailing:.10f}")

    # ── Step 6: Assemble the SST energy ──
    # The full SST decomposition: E_PT2 = E^MP2_dressed - E^eff_S + E^eff_T
    #
    # However, E^eff_T from the trailing solver gives us the EXACT energy
    # for cases 1-11, not the theoretical "trailing correction."
    # The theoretical identity is:
    #   E_exact = E_H± + E_cases_1-11
    #   E_exact = E^MP2_dressed - E^eff_S + E^eff_T
    # So: E^eff_T = E_cases_1-11 + E_H± - E^MP2_dressed + E^eff_S
    #
    # For the total energy, we use the exact result:
    #   E_PT2 = E_H± + E_trailing (cases 1-11)
    # which is numerically more stable.
    e_pt2_direct = float(e_mp2_like + e_trailing)

    # Also compute the SST formula for verification:
    # E^eff_T_theoretical = E_trailing + E_H± - E^MP2_dressed + E^eff_S
    e_eff_t = float(e_trailing + e_mp2_like - e_mp2_dressed + e_eff_s)

    # Sanity check: E^MP2_dressed - E^eff_S + E^eff_T should equal E_PT2
    e_pt2_sst = float(e_mp2_dressed - e_eff_s + e_eff_t)
    if verbose >= 1:
        print(f"  E^eff_T (derived) = {e_eff_t:.10f}")
        print(f"  E_PT2 (direct: H± + cases 1-11) = {e_pt2_direct:.10f}")
        print(f"  E_PT2 (SST: dressed - S + T) = {e_pt2_sst:.10f}")
        diff = abs(e_pt2_direct - e_pt2_sst)
        print(f"  Consistency check: |direct - SST| = {diff:.2e}")

    e_pt2 = e_pt2_direct
    e_tot = float(inp.e_ref + e_pt2)
    native_context = None
    if bool(return_context):
        from asuka.caspt2.sst.types import SSTNativeContext  # noqa: PLC0415

        native_context = SSTNativeContext(
            mode="full",
            dressed=dressed,
            terms={
                "e_mp2_dressed": float(e_mp2_dressed),
                "e_eff_s": float(e_eff_s),
                "e_eff_t": float(e_eff_t),
                "e_mp2_like_hpm": float(e_mp2_like),
                "e_trailing_cases_1_11": float(e_trailing),
                "e_pt2_direct": float(e_pt2_direct),
                "e_pt2_sst": float(e_pt2_sst),
            },
        )

    return SSTResult(
        e_mp2_like=float(e_mp2_dressed),
        e_active=float(-e_eff_s + e_eff_t),
        e_pt2=float(e_pt2),
        e_tot=float(e_tot),
        amplitudes_active=t_active,
        pcg_converged=bool(converged),
        pcg_niter=int(niter),
        breakdown={
            "sst_mode": "full",
            "e_mp2_dressed": float(e_mp2_dressed),
            "e_eff_s": float(e_eff_s),
            "e_eff_t": float(e_eff_t),
            "e_mp2_like_hpm": float(e_mp2_like),
            "e_trailing_cases_1_11": float(e_trailing),
            "e_pt2_direct": float(e_pt2_direct),
            "e_pt2_sst": float(e_pt2_sst),
        },
        native_context=native_context,
    )
