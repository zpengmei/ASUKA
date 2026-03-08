"""Reduced active-space system for SST-CASPT2: IC cases 1-11.

Solves the 11 non-H± IC-CASPT2 cases using the existing IC infrastructure
(S/B decomposition, RHS, sigma operator, PCG solver) with H± coupling
KODs effectively zeroed out by passing empty H± arrays.

The H± energy contribution is computed separately as the MP2-like term.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from asuka.caspt2.energy import _get_external_energies
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat, compute_e0
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.shifts import (
    apply_imaginary_shift,
    apply_real_shift,
    compute_shift_correction,
)
from asuka.caspt2.sigma import (
    SigmaC1ActiveVirtualCoupling,
    SigmaC1CaseCoupling,
)
from asuka.caspt2.solver import pcg_solve, pcg_solve_iterative
from asuka.caspt2.superindex import SuperindexMap

from asuka.caspt2.sst.types import SSTCaseContext, SSTConfig, SSTInput

def _build_full_mo_eris(B_ao: np.ndarray, mo_coeff: np.ndarray) -> np.ndarray:
    """Build full 4-index MO ERIs from DF factors. O(N^5) cost."""
    B_ao_np = np.asarray(B_ao, dtype=np.float64)
    C_np = np.asarray(mo_coeff, dtype=np.float64)
    nao, nmo = C_np.shape
    naux = B_ao_np.shape[2]
    # B_mo[p,q,Q] = C^T @ B_ao @ C
    B_flat = B_ao_np.reshape(nao, nao * naux)
    tmp = C_np.T @ B_flat  # (nmo, nao*naux)
    tmp = tmp.reshape(nmo, nao, naux)
    B_mo = np.einsum("pnQ,nq->pqQ", tmp, C_np)  # (nmo, nmo, naux)
    # eri_mo[p,q,r,s] = sum_Q B_mo[p,q,Q] * B_mo[r,s,Q]
    eri_mo = np.einsum("pqQ,rsQ->pqrs", B_mo, B_mo, optimize=True)
    return eri_mo


_CASE_NAMES = [
    "A(VJTU)", "B+(VJTIP)", "B-(VJTIM)", "C(ATVX)", "D(AIVX)",
    "E+(VJAIP)", "E-(VJAIM)", "F+(BVATP)", "F-(BVATM)",
    "G+(BJATQ)", "G-(BJATM)",
]


class _SigmaSST:
    """Wrapper that runs the 13-case sigma operator on only cases 1-11.

    Pads the 11-element input to 13 elements (empty arrays for H±),
    calls the inner sigma operator, then returns only the first 11 outputs.

    The inner sigma operator already skips empty blocks via size checks,
    so H± couplings (KODs 17-18, 21-22, 23-24) are effectively zeroed.
    """

    def __init__(self, inner: SigmaC1CaseCoupling | SigmaC1ActiveVirtualCoupling) -> None:
        self._inner = inner

    def __call__(self, vec_in_11: list[np.ndarray]) -> list[np.ndarray]:
        # Pad to 13 elements: add empty arrays for cases 12 (H+) and 13 (H-)
        vec_13 = list(vec_in_11) + [np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)]
        sigma_13 = self._inner(vec_13)
        return sigma_13[:11]


def build_and_solve_reduced_system(
    inp: SSTInput,
    cfg: SSTConfig,
    *,
    df_blocks: Any = None,
    device: int | None = None,
    return_context: bool = False,
) -> tuple[float, list[np.ndarray] | None, bool, int] | tuple[float, list[np.ndarray] | None, bool, int, dict[int, SSTCaseContext]]:
    """Solve IC cases 1-11 using existing IC infrastructure.

    Parameters
    ----------
    inp : SSTInput
        Must have dm3_act, ci_context, smap, and fock populated.
    cfg : SSTConfig
    df_blocks : CASPT2DFBlocks or None
        DF pair blocks (CPU NumPy or CUDA CuPy). When provided, avoids
        building full O(nmo^4) MO ERIs.
    device : int or None
        CUDA device for DF RHS. When set with df_blocks, uses GPU RHS builder.

    Returns
    -------
    e_active : float
        Energy contribution from cases 1-11.
    amplitudes : list[ndarray] or None
        Per-case amplitude vectors (11 entries) in the SR basis.
    converged : bool
    niter : int
    context : dict[int, SSTCaseContext], optional
        Returned only when ``return_context=True``.
    """
    smap: SuperindexMap = inp.smap
    fock: CASPT2Fock = inp.fock
    dm1 = np.asarray(inp.dm1_act, dtype=np.float64)
    dm2 = np.asarray(inp.dm2_act, dtype=np.float64)
    dm3 = np.asarray(inp.dm3_act, dtype=np.float64)
    ci_context: CASPT2CIContext = inp.ci_context
    verbose = cfg.verbose

    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)

    if verbose >= 1:
        print(f"SST reduced system (cases 1-11)")
        print(f"  nish={nish}, nash={nash}, nssh={nssh}")

    # Build S/B, diagonalize, build RHS for cases 1-11 only
    sb_decomps: list[SBDecomposition] = []
    h0_diags: list[np.ndarray] = []
    rhs_transformed: list[np.ndarray] = []
    smats_std: list[np.ndarray] = []
    decomp_by_case: dict[int, SBDecomposition] = {}

    # Build RHS dispatch: prefer DF blocks over full ERIs.
    _rhs_fn = None
    if df_blocks is not None and device is not None:
        # CUDA DF path — RHS built on GPU, transferred to CPU
        from asuka.caspt2.cuda.rhs_df_cuda import build_rhs_df_cuda  # noqa: PLC0415
        import cupy as cp  # noqa: PLC0415
        _rhs_fn = lambda case: cp.asnumpy(
            build_rhs_df_cuda(case, smap, fock, df_blocks, dm1, dm2, device=device)
        )
    elif df_blocks is not None:
        # CPU DF path — no full ERIs needed
        from asuka.caspt2.rhs_df import build_rhs_df  # noqa: PLC0415
        _rhs_fn = lambda case: build_rhs_df(case, smap, fock, df_blocks, dm1, dm2)
    elif inp.B_ao is not None:
        # Auto-build CPU DF blocks from B_ao
        from asuka.caspt2.rhs_df import build_rhs_df, build_df_blocks_cpu  # noqa: PLC0415
        _df = build_df_blocks_cpu(inp.B_ao, inp.mo_coeff, nish, nash, nssh)
        _rhs_fn = lambda case: build_rhs_df(case, smap, fock, _df, dm1, dm2)
    elif inp.eri_mo is not None:
        eri_mo = np.asarray(inp.eri_mo, dtype=np.float64)
        _rhs_fn = lambda case: build_rhs(case, smap, fock, eri_mo, dm1, dm2)
    else:
        raise ValueError("reduced system requires df_blocks, eri_mo, or B_ao")

    for case in range(1, 12):  # cases 1-11 only
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])

        if nasup == 0 or nisup == 0:
            sb_decomps.append(SBDecomposition(
                s_eigvals=np.empty(0), transform=np.empty((0, 0)),
                nindep=0, b_diag=np.empty(0),
            ))
            h0_diags.append(np.empty(0, dtype=np.float64))
            rhs_transformed.append(np.empty(0, dtype=np.float64))
            smats_std.append(np.empty((0, 0), dtype=np.float64))
            if verbose >= 2:
                print(f"  Case {case} ({_CASE_NAMES[case-1]}): skipped (empty)")
            continue

        smat = build_smat(case, smap, dm1, dm2, dm3)
        smats_std.append(smat)

        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)

        decomp = sbdiag(smat, bmat, threshold_norm=cfg.threshold, threshold_s=cfg.threshold_s)
        sb_decomps.append(decomp)
        decomp_by_case[int(case)] = decomp

        if decomp.nindep == 0:
            h0_diags.append(np.empty(0, dtype=np.float64))
            rhs_transformed.append(np.empty(0, dtype=np.float64))
            if verbose >= 2:
                print(f"  Case {case} ({_CASE_NAMES[case-1]}): all lin-dep removed")
            continue

        ext_energies = _get_external_energies(case, smap, fock)

        if ext_energies.size > 0:
            full_diag = (decomp.b_diag[:, None] + ext_energies[None, :]).ravel()
        else:
            full_diag = decomp.b_diag.copy()

        h0_diags.append(full_diag)

        rhs_raw = _rhs_fn(case)

        if nisup > 0 and nasup > 0:
            rhs_mat = rhs_raw.reshape(nasup, nisup)
            rhs_diag = decomp.transform.T @ smat @ rhs_mat
            rhs_transformed.append(rhs_diag.ravel())
        else:
            rhs_transformed.append(rhs_raw)

        if verbose >= 2:
            print(f"  Case {case} ({_CASE_NAMES[case-1]}): "
                  f"nasup={nasup}, nisup={nisup}, nindep={decomp.nindep}")

    # Apply shifts
    h0_diags_orig = [d.copy() for d in h0_diags]
    if abs(cfg.imag_shift) > 1e-15:
        h0_diags = apply_imaginary_shift(h0_diags, cfg.imag_shift)
    if abs(cfg.real_shift) > 1e-15:
        h0_diags = apply_real_shift(h0_diags, cfg.real_shift)

    # Check whether sigma operator is needed
    use_sigma = False
    if nash > 0:
        ao = nish
        vo = nish + nash
        so = vo + nssh
        max_off = 0.0
        if nish > 0:
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[ao:vo, :ao]))))
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[:ao, ao:vo]))))
        if nssh > 0:
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[ao:vo, vo:so]))))
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[vo:so, ao:vo]))))
        if nish > 0 and nssh > 0:
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[:ao, vo:so]))))
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[vo:so, :ao]))))
        use_sigma = max_off > 1e-12

    if use_sigma:
        nactel = max(1, int(round(float(np.trace(dm1[:nash, :nash])))))

        # Build the 13-element lists needed by the sigma operator.
        # Cases 12-13 get empty placeholders so the inner sigma skips them.
        _empty_decomp = SBDecomposition(
            s_eigvals=np.empty(0), transform=np.empty((0, 0)),
            nindep=0, b_diag=np.empty(0),
        )
        sb_decomps_13 = list(sb_decomps) + [_empty_decomp, _empty_decomp]
        h0_diags_13 = list(h0_diags) + [np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)]
        smats_13 = list(smats_std) + [np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)]

        if nish > 0:
            inner_sigma = SigmaC1CaseCoupling(
                smap=smap,
                fock=fock,
                smats=smats_13,
                sb_decomp=sb_decomps_13,
                h0_diag=h0_diags_13,
                nactel=nactel,
            )
        else:
            inner_sigma = SigmaC1ActiveVirtualCoupling(
                smap=smap,
                fock=fock,
                smats=smats_13,
                sb_decomp=sb_decomps_13,
                h0_diag=h0_diags_13,
                nactel=nactel,
            )

        sigma_op = _SigmaSST(inner_sigma)
        result = pcg_solve_iterative(
            sigma_op,
            rhs_transformed,
            h0_diags,
            tol=cfg.tol,
            maxiter=cfg.maxiter,
            verbose=verbose,
        )
    else:
        result = pcg_solve(h0_diags, rhs_transformed, tol=cfg.tol, maxiter=cfg.maxiter, verbose=verbose)

    if not result.converged:
        import warnings
        warnings.warn(
            f"SST PCG did not converge (|r|={result.residual:.2e}, niter={result.niter})",
            stacklevel=2,
        )
        if verbose >= 1:
            print(f"  WARNING: PCG did not converge (|r|={result.residual:.2e})")

    # Compute energy: E2 = <V|T> for cases 1-11
    e_active = 0.0
    for case_idx, (v, t) in enumerate(zip(rhs_transformed, result.amplitudes)):
        if v.size == 0:
            continue
        e_case = float(np.dot(v.ravel(), t.ravel()))
        e_active += e_case
        if verbose >= 1:
            print(f"  Case {case_idx + 1} ({_CASE_NAMES[case_idx]}): E2 = {e_case:.10f}")

    # Shift correction
    if abs(cfg.imag_shift) > 1e-15 or abs(cfg.real_shift) > 1e-15:
        e_shift_corr = compute_shift_correction(result.amplitudes, h0_diags, h0_diags_orig)
        e_active -= e_shift_corr
        if verbose >= 1:
            print(f"  Shift correction (cases 1-11): {-e_shift_corr:.10f}")

    if verbose >= 1:
        print(f"  E_active (cases 1-11) = {e_active:.10f}")

    if not bool(return_context):
        return float(e_active), result.amplitudes, result.converged, result.niter

    context: dict[int, SSTCaseContext] = {}
    empty_decomp = SBDecomposition(
        s_eigvals=np.empty(0),
        transform=np.empty((0, 0)),
        nindep=0,
        b_diag=np.empty(0),
    )
    for case_idx, case in enumerate(range(1, 12)):
        amps = np.asarray(result.amplitudes[case_idx], dtype=np.float64)
        rhs_case = np.asarray(rhs_transformed[case_idx], dtype=np.float64)
        decomp = decomp_by_case.get(int(case), empty_decomp)
        e_case = 0.0 if rhs_case.size == 0 else float(np.dot(rhs_case.ravel(), amps.ravel()))
        context[int(case)] = SSTCaseContext(
            case=int(case),
            sector="reduced",
            rhs_sr=np.asarray(rhs_case, dtype=np.float64),
            h0_diag=np.asarray(h0_diags[case_idx], dtype=np.float64),
            amplitudes_sr=np.asarray(amps, dtype=np.float64),
            energy=float(e_case),
            smat=np.asarray(smats_std[case_idx], dtype=np.float64),
            transform=np.asarray(decomp.transform, dtype=np.float64),
            b_diag=np.asarray(decomp.b_diag, dtype=np.float64),
            nindep=int(decomp.nindep),
        )
    return float(e_active), result.amplitudes, result.converged, result.niter, context
