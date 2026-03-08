"""MS/XMS-CASPT2 via the SST backend (CPU and CUDA paths).

Computes the multi-state effective Hamiltonian by:
1. Running per-state SS-CASPT2 using the SST IC-mode solver (cases 1-11)
2. Solving H+/- amplitudes (cases 12-13) via diagonal solve
3. Assembling CASPT2EnergyResult with 13-case amplitudes
4. Delegating to the existing build_heff() / diagonalize_heff() infrastructure

For XMS, the CI vectors are rotated before step 1 and the reference
rotation correction is applied to Heff after step 3.

CUDA path:
- Computes row_dots from CPU amplitudes and GPU RHS (via build_rhs_df_cuda)
- Delegates to build_heff_cuda() for GPU transition RDMs and HCOUP
- Eliminates the O(N^5) full ERI construction
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from asuka.caspt2.energy import _get_external_energies
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat
from asuka.caspt2.multistate import build_heff, diagonalize_heff
from asuka.caspt2.overlap import build_smat, sbdiag
from asuka.caspt2.result import CASPT2EnergyResult, CASPT2Result
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.shifts import (
    apply_imaginary_shift,
    apply_real_shift,
    compute_shift_correction,
)
from asuka.caspt2.sst.energy_sst import sst_caspt2_energy_ss
from asuka.caspt2.sst.reduced_system import _build_full_mo_eris
from asuka.caspt2.sst.types import SSTConfig, SSTInput
from asuka.caspt2.superindex import SuperindexMap

if TYPE_CHECKING:
    from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks


def _solve_hpm_amplitudes(
    inp: SSTInput,
    cfg: SSTConfig,
    eri_mo: np.ndarray | None = None,
    *,
    df_blocks: object = None,
) -> list[np.ndarray]:
    """Solve H+/- amplitude equations (cases 12-13) via diagonal solve.

    Cases 12-13 are purely external (core->virtual doubles) with identity
    overlap in the active superindex space. The solve reduces to:
        T_SR[p, ext] = -RHS_SR[p, ext] / h0_diag[p, ext]

    Returns a 2-element list: [amps_case12, amps_case13].
    """
    smap: SuperindexMap = inp.smap
    fock: CASPT2Fock = inp.fock
    dm1 = np.asarray(inp.dm1_act, dtype=np.float64)
    dm2 = np.asarray(inp.dm2_act, dtype=np.float64)
    dm3 = np.asarray(inp.dm3_act, dtype=np.float64)
    ci_context = inp.ci_context

    hpm_amps: list[np.ndarray] = []

    for case in (12, 13):
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])

        if nasup == 0 or nisup == 0:
            hpm_amps.append(np.empty(0, dtype=np.float64))
            continue

        smat = build_smat(case, smap, dm1, dm2, dm3)
        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)
        decomp = sbdiag(smat, bmat, threshold_norm=cfg.threshold, threshold_s=cfg.threshold_s)

        if decomp.nindep == 0:
            hpm_amps.append(np.empty(0, dtype=np.float64))
            continue

        # Build H0 diagonal
        ext_energies = _get_external_energies(case, smap, fock)
        if ext_energies.size > 0:
            full_diag = (decomp.b_diag[:, None] + ext_energies[None, :]).ravel()
        else:
            full_diag = decomp.b_diag.copy()

        # Build RHS and transform to SR basis
        if df_blocks is not None:
            from asuka.caspt2.rhs_df import build_rhs_df  # noqa: PLC0415
            rhs_raw = build_rhs_df(case, smap, fock, df_blocks, dm1, dm2)
        else:
            rhs_raw = build_rhs(case, smap, fock, eri_mo, dm1, dm2)
        rhs_mat = rhs_raw.reshape(nasup, nisup)
        rhs_sr = (decomp.transform.T @ smat @ rhs_mat).ravel()

        # Apply shifts to denominator
        h0_orig = full_diag.copy()
        h0_shifted = full_diag.copy()
        if abs(cfg.imag_shift) > 1e-15:
            h0_shifted = apply_imaginary_shift([h0_shifted], cfg.imag_shift)[0]
        if abs(cfg.real_shift) > 1e-15:
            h0_shifted = apply_real_shift([h0_shifted], cfg.real_shift)[0]

        # Diagonal solve: T = -RHS / h0
        amps = np.zeros_like(rhs_sr)
        mask = np.abs(h0_shifted) > 1e-14
        amps[mask] = -rhs_sr[mask] / h0_shifted[mask]

        # Shift correction applied at energy level, not stored in amplitudes
        hpm_amps.append(amps)

    return hpm_amps


def _solve_hpm_amplitudes_df(
    inp: SSTInput,
    cfg: SSTConfig,
    df_blocks: Any,
    *,
    device: int | None = None,
) -> list[np.ndarray]:
    """Solve H+/- amplitude equations (cases 12-13) using DF-based RHS on GPU.

    Same as _solve_hpm_amplitudes() but replaces build_rhs(eri_mo) with
    build_rhs_df_cuda(df_blocks), eliminating the O(N^5) full ERI construction.

    Returns a 2-element list: [amps_case12, amps_case13].
    """
    from asuka.caspt2.cuda.rhs_df_cuda import build_rhs_df_cuda  # noqa: PLC0415

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF-based H+/- solve") from e

    smap: SuperindexMap = inp.smap
    fock: CASPT2Fock = inp.fock
    dm1 = np.asarray(inp.dm1_act, dtype=np.float64)
    dm2 = np.asarray(inp.dm2_act, dtype=np.float64)
    dm3 = np.asarray(inp.dm3_act, dtype=np.float64)
    ci_context = inp.ci_context

    hpm_amps: list[np.ndarray] = []

    for case in (12, 13):
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])

        if nasup == 0 or nisup == 0:
            hpm_amps.append(np.empty(0, dtype=np.float64))
            continue

        smat = build_smat(case, smap, dm1, dm2, dm3)
        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)
        decomp = sbdiag(smat, bmat, threshold_norm=cfg.threshold, threshold_s=cfg.threshold_s)

        if decomp.nindep == 0:
            hpm_amps.append(np.empty(0, dtype=np.float64))
            continue

        # Build H0 diagonal
        ext_energies = _get_external_energies(case, smap, fock)
        if ext_energies.size > 0:
            full_diag = (decomp.b_diag[:, None] + ext_energies[None, :]).ravel()
        else:
            full_diag = decomp.b_diag.copy()

        # Build RHS on GPU and transfer to CPU
        rhs_c = build_rhs_df_cuda(case, smap, fock, df_blocks, dm1, dm2, device=device)
        rhs_raw = cp.asnumpy(rhs_c)
        del rhs_c  # release GPU memory immediately
        rhs_mat = rhs_raw.reshape(nasup, nisup)
        rhs_sr = (decomp.transform.T @ smat @ rhs_mat).ravel()

        # Apply shifts to denominator
        h0_shifted = full_diag.copy()
        if abs(cfg.imag_shift) > 1e-15:
            h0_shifted = apply_imaginary_shift([h0_shifted], cfg.imag_shift)[0]
        if abs(cfg.real_shift) > 1e-15:
            h0_shifted = apply_real_shift([h0_shifted], cfg.real_shift)[0]

        # Diagonal solve: T = -RHS / h0
        amps = np.zeros_like(rhs_sr)
        mask = np.abs(h0_shifted) > 1e-14
        amps[mask] = -rhs_sr[mask] / h0_shifted[mask]

        hpm_amps.append(amps)

    return hpm_amps


def _compute_row_dots_sst(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    amplitudes_13: list[np.ndarray],
    df_blocks: Any,
    *,
    ci_context: Any | None = None,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    device: int | None = None,
) -> list:
    """Compute row_dots_by_case_cuda from CPU SR-basis amplitudes + GPU DF RHS.

    For each case c in 1..13:
        1. Build RHS on GPU: rhs_c = build_rhs_df_cuda(c, ...)   → CuPy (nasup, nisup)
        2. Rebuild S/B decomposition (CPU, small active space)
        3. Back-transform: t_raw = decomp.transform @ amps_SR    → NumPy (nasup, nisup)
        4. Upload: t_raw_d = cp.asarray(t_raw)
        5. GPU GEMM: row_dots = rhs_c @ t_raw_d.T                → CuPy (nasup, nasup)

    Returns a 13-element list of CuPy arrays, suitable for build_heff_cuda().
    """
    from asuka.caspt2.cuda.rhs_df_cuda import build_rhs_df_cuda  # noqa: PLC0415

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy required for _compute_row_dots_sst") from e

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)

    row_dots_list: list = []

    for case in range(1, 14):
        case_idx = case - 1
        nasup = int(smap.nasup[case_idx])
        nisup = int(smap.nisup[case_idx])
        amps_sr = amplitudes_13[case_idx]

        if nasup == 0 or nisup == 0 or amps_sr.size == 0:
            row_dots_list.append(cp.empty((0, 0), dtype=cp.float64))
            continue

        # Rebuild S/B decomposition for back-transform
        smat = build_smat(case, smap, dm1, dm2, dm3)
        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)
        decomp = sbdiag(smat, bmat, threshold_norm=threshold, threshold_s=threshold_s)

        nindep = int(decomp.nindep)
        if nindep == 0:
            row_dots_list.append(cp.empty((0, 0), dtype=cp.float64))
            continue

        # Back-transform from SR to raw basis: t_raw = U @ t_SR
        # decomp.transform is (nasup, nindep)
        t_sr = amps_sr.reshape(nindep, nisup)
        t_raw = decomp.transform @ t_sr  # (nasup, nisup)

        # Build RHS on GPU
        rhs_c = build_rhs_df_cuda(case, smap, fock, df_blocks, dm1, dm2, device=device)
        # rhs_c shape: (nasup, nisup) as CuPy

        # Upload t_raw and compute row_dots via GPU GEMM
        t_raw_d = cp.asarray(t_raw, dtype=cp.float64)
        row_dots = rhs_c @ t_raw_d.T  # (nasup, nasup)
        del rhs_c, t_raw_d  # release GPU intermediates
        row_dots_list.append(cp.ascontiguousarray(row_dots))

    return row_dots_list


def _sst_ss_with_full_amplitudes_cuda(
    inp: SSTInput,
    cfg: SSTConfig,
    df_blocks: Any,
    *,
    device: int | None = None,
) -> CASPT2EnergyResult:
    """Run SS-SST (IC mode) and produce CASPT2EnergyResult with GPU row_dots.

    Combines:
    1. SST IC-mode solver → 11-case SR-basis amplitudes
    2. DF-based H+/- solver → cases 12-13 amplitudes (no full ERIs)
    3. _compute_row_dots_sst() → 13 CuPy row_dots arrays

    Stores row_dots in breakdown["row_dots_by_case_cuda"] for build_heff_cuda().
    """
    # Run SS solver in IC mode → 11-case amplitudes (pass DF blocks to avoid full ERIs)
    sst_res = sst_caspt2_energy_ss(inp, cfg, sst_mode="ic", df_blocks=df_blocks, device=device)

    amps_11 = sst_res.amplitudes_active
    if amps_11 is None:
        amps_11 = [np.empty(0, dtype=np.float64) for _ in range(11)]

    # Get H+/- amplitudes via DF (no full ERIs)
    hpm_amps = _solve_hpm_amplitudes_df(inp, cfg, df_blocks, device=device)

    # Assemble 13-case amplitudes
    amplitudes_13 = list(amps_11) + hpm_amps

    # Compute row_dots on GPU
    row_dots = _compute_row_dots_sst(
        smap=inp.smap,
        fock=inp.fock,
        dm1=np.asarray(inp.dm1_act, dtype=np.float64),
        dm2=np.asarray(inp.dm2_act, dtype=np.float64),
        dm3=np.asarray(inp.dm3_act, dtype=np.float64),
        amplitudes_13=amplitudes_13,
        df_blocks=df_blocks,
        ci_context=inp.ci_context,
        threshold=cfg.threshold,
        threshold_s=cfg.threshold_s,
        device=device,
    )

    breakdown = dict(sst_res.breakdown) if isinstance(sst_res.breakdown, dict) else {}
    breakdown["row_dots_by_case_cuda"] = row_dots

    return CASPT2EnergyResult(
        e_ref=float(sst_res.e_tot - sst_res.e_pt2),
        e_pt2=float(sst_res.e_pt2),
        e_tot=float(sst_res.e_tot),
        amplitudes=amplitudes_13,
        breakdown=breakdown,
    )


def sst_caspt2_energy_ms_cuda(
    nstates: int,
    sst_inputs: list[SSTInput],
    ci_vectors: list[np.ndarray],
    drt: Any,
    smap: SuperindexMap,
    fock_list: list[CASPT2Fock] | CASPT2Fock,
    df_blocks: Any,
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    e_ref_list: list[float],
    cfg: SSTConfig,
    *,
    xms_mode: bool = False,
    u0: np.ndarray | None = None,
    device: int | None = None,
    verbose: int = 0,
) -> CASPT2Result:
    """MS/XMS-CASPT2 via the SST backend with CUDA-accelerated Heff.

    Uses build_heff_cuda() for GPU transition RDMs and HCOUP contraction,
    eliminating the O(N^5) full ERI construction entirely.

    Parameters
    ----------
    df_blocks : CASPT2DFBlocks
        DF pair blocks for GPU RHS construction.
    device : int or None
        CUDA device index.

    Other parameters match sst_caspt2_energy_ms().
    """
    from asuka.caspt2.cuda.multistate_cuda import build_heff_cuda  # noqa: PLC0415

    if xms_mode and u0 is None:
        raise ValueError("XMS mode requires u0 rotation matrix")

    if verbose >= 1:
        print(f"SST-{'XMS' if xms_mode else 'MS'}-CASPT2 (CUDA): {nstates} states")

    # Step 1: Per-state SS with GPU row_dots
    ss_results: list[CASPT2EnergyResult] = []
    for i in range(nstates):
        if verbose >= 1:
            print(f"\n--- State {i} SS-CASPT2 (SST-IC + CUDA row_dots) ---")
        ss_res = _sst_ss_with_full_amplitudes_cuda(
            sst_inputs[i], cfg, df_blocks, device=device,
        )
        ss_results.append(ss_res)
        if verbose >= 1:
            print(f"  E_PT2({i}) = {ss_res.e_pt2:.10f}")
            print(f"  E_tot({i}) = {ss_res.e_tot:.10f}")

    # Step 2: Build Heff on GPU
    heff = build_heff_cuda(
        nstates,
        ss_results,
        [np.asarray(c, dtype=np.float64) for c in ci_vectors],
        drt,
        smap,
        device=device,
        verbose=verbose,
    )

    # Step 3: XMS reference rotation correction
    if xms_mode:
        if u0 is None:
            raise ValueError("XMS mode requires u0 rotation matrix")
        from asuka.caspt2.xms_utils import _apply_xms_reference_rotation  # noqa: PLC0415
        heff = _apply_xms_reference_rotation(
            heff=heff, e_ref_list=e_ref_list, u0=u0,
        )

    # Step 4: Diagonalize
    ms_energies, ueff = diagonalize_heff(heff)

    if verbose >= 1:
        print(f"\n{'XMS' if xms_mode else 'MS'}-CASPT2 eigenvalues:")
        for i, e in enumerate(ms_energies):
            print(f"  State {i}: {e:.10f}")

    method = "XMS" if xms_mode else "MS"
    breakdown: dict[str, Any] = {
        "ss_energies": [float(r.e_tot) for r in ss_results],
        "ms_energies": ms_energies.tolist(),
        "heff_backend": "sst-cuda",
    }
    if xms_mode and u0 is not None:
        breakdown["u0"] = np.asarray(u0, dtype=np.float64).tolist()
        breakdown["reference_rotation_applied"] = True

    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(ms_energies[i] - float(e_ref_list[i])) for i in range(nstates)],
        e_tot=ms_energies.tolist(),
        heff=np.asarray(heff, dtype=np.float64),
        ueff=np.asarray(ueff, dtype=np.float64),
        amplitudes=[r.amplitudes for r in ss_results],
        method=method,
        breakdown=breakdown,
    )


def _sst_ss_with_full_amplitudes(
    inp: SSTInput,
    cfg: SSTConfig,
    eri_mo: np.ndarray | None = None,
    *,
    df_blocks: object = None,
) -> CASPT2EnergyResult:
    """Run SS-SST (IC mode) and produce a CASPT2EnergyResult with 13-case amplitudes.

    Cases 1-11 come from the IC-mode SST solver.
    Cases 12-13 come from _solve_hpm_amplitudes().

    When *df_blocks* is provided, DF-native RHS is used for both the
    IC solver (cases 1-11) and the H+/- solver (cases 12-13), avoiding
    the O(N^5) full ERI construction for the SS energy.
    """
    # Run SS solver in IC mode → 11-case amplitudes
    sst_res = sst_caspt2_energy_ss(inp, cfg, sst_mode="ic", df_blocks=df_blocks)

    # Get 11-case amplitudes
    amps_11 = sst_res.amplitudes_active
    if amps_11 is None:
        amps_11 = [np.empty(0, dtype=np.float64) for _ in range(11)]

    # Get H+/- amplitudes (cases 12-13)
    hpm_amps = _solve_hpm_amplitudes(inp, cfg, eri_mo, df_blocks=df_blocks)

    # Assemble 13-case list
    amplitudes_13 = list(amps_11) + hpm_amps

    return CASPT2EnergyResult(
        e_ref=float(sst_res.e_tot - sst_res.e_pt2),
        e_pt2=float(sst_res.e_pt2),
        e_tot=float(sst_res.e_tot),
        amplitudes=amplitudes_13,
        breakdown=sst_res.breakdown,
    )


def sst_caspt2_energy_ms(
    nstates: int,
    sst_inputs: list[SSTInput],
    ci_vectors: list[np.ndarray],
    drt: Any,
    smap: SuperindexMap,
    fock_list: list[CASPT2Fock] | CASPT2Fock,
    eri_mo: np.ndarray | None,
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    e_ref_list: list[float],
    cfg: SSTConfig,
    *,
    xms_mode: bool = False,
    u0: np.ndarray | None = None,
    verbose: int = 0,
) -> CASPT2Result:
    """MS/XMS-CASPT2 via the SST backend.

    Parameters
    ----------
    nstates : int
        Number of states.
    sst_inputs : list[SSTInput]
        Per-state SST input objects.
    ci_vectors : list[ndarray]
        CI vectors for each state (possibly XMS-rotated).
    drt : DRT
        Active-space DRT.
    smap : SuperindexMap
        Superindex mapping (shared across states for same active space).
    fock_list : list[CASPT2Fock] or CASPT2Fock
        Per-state Fock (MS) or shared SA-Fock (XMS).
    eri_mo : ndarray or None
        Full MO ERIs. Built from DF if None.
    dm1_list, dm2_list, dm3_list : list[ndarray]
        Per-state RDMs.
    e_ref_list : list[float]
        Per-state reference (CASSCF) energies.
    cfg : SSTConfig
        SST configuration.
    xms_mode : bool
        If True, apply XMS reference rotation to Heff.
    u0 : ndarray or None
        XMS rotation matrix (required if xms_mode=True).
    verbose : int
        Verbosity level.

    Returns
    -------
    CASPT2Result
    """
    if xms_mode and u0 is None:
        raise ValueError("XMS mode requires u0 rotation matrix")

    # Build CPU DF blocks for SS solver (avoids full ERIs for cases 1-13 RHS).
    # build_heff on CPU still needs full ERIs for off-diagonal HCOUP, so we
    # defer that construction until Step 2.
    df_blocks_cpu = None
    if eri_mo is None and sst_inputs[0].B_ao is not None:
        from asuka.caspt2.rhs_df import build_df_blocks_cpu  # noqa: PLC0415
        df_blocks_cpu = build_df_blocks_cpu(
            sst_inputs[0].B_ao, sst_inputs[0].mo_coeff,
            ncore=sst_inputs[0].ncore, ncas=sst_inputs[0].ncas,
            nvirt=sst_inputs[0].nvirt,
        )

    if verbose >= 1:
        print(f"SST-{'XMS' if xms_mode else 'MS'}-CASPT2: {nstates} states")

    # Step 1: Per-state SS with 13-case amplitudes (using DF blocks when available)
    ss_results: list[CASPT2EnergyResult] = []
    for i in range(nstates):
        if verbose >= 1:
            print(f"\n--- State {i} SS-CASPT2 ---")
        ss_res = _sst_ss_with_full_amplitudes(
            sst_inputs[i], cfg, eri_mo, df_blocks=df_blocks_cpu,
        )
        ss_results.append(ss_res)
        if verbose >= 1:
            print(f"  E_PT2({i}) = {ss_res.e_pt2:.10f}")
            print(f"  E_tot({i}) = {ss_res.e_tot:.10f}")

    # Step 2: Build Heff using existing CPU build_heff.
    # build_heff needs full MO ERIs for off-diagonal HCOUP elements.
    if eri_mo is None:
        eri_mo = _build_full_mo_eris(sst_inputs[0].B_ao, sst_inputs[0].mo_coeff)
    heff = build_heff(
        nstates,
        ss_results,
        [np.asarray(c, dtype=np.float64) for c in ci_vectors],
        drt,
        smap,
        fock_list,
        eri_mo,
        dm1_list,
        dm2_list,
        dm3_list,
        threshold=cfg.threshold,
        threshold_s=cfg.threshold_s,
        verbose=verbose,
    )

    # Step 3: XMS reference rotation correction (if applicable)
    if xms_mode:
        if u0 is None:
            raise ValueError("XMS mode requires u0 rotation matrix")
        from asuka.caspt2.xms_utils import _apply_xms_reference_rotation
        heff = _apply_xms_reference_rotation(
            heff=heff, e_ref_list=e_ref_list, u0=u0,
        )

    # Step 4: Diagonalize
    ms_energies, ueff = diagonalize_heff(heff)

    if verbose >= 1:
        print(f"\n{'XMS' if xms_mode else 'MS'}-CASPT2 eigenvalues:")
        for i, e in enumerate(ms_energies):
            print(f"  State {i}: {e:.10f}")

    method = "XMS" if xms_mode else "MS"
    breakdown: dict[str, Any] = {
        "ss_energies": [float(r.e_tot) for r in ss_results],
        "ms_energies": ms_energies.tolist(),
        "heff_backend": "sst-cpu",
    }
    if xms_mode and u0 is not None:
        breakdown["u0"] = np.asarray(u0, dtype=np.float64).tolist()
        breakdown["reference_rotation_applied"] = True

    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(ms_energies[i] - float(e_ref_list[i])) for i in range(nstates)],
        e_tot=ms_energies.tolist(),
        heff=np.asarray(heff, dtype=np.float64),
        ueff=np.asarray(ueff, dtype=np.float64),
        amplitudes=[r.amplitudes for r in ss_results],
        method=method,
        breakdown=breakdown,
    )
