"""Trailing subspace correction for SST-CASPT2.

Computes E^eff_T (Eq. 68-74 of Song 2024), the trailing subspace energy
correction. This captures the difference between the exact Dyall H0 and
the Kronecker sum approximation G_apx for the supporting subspace cases.

The trailing equation is:

    (Q_T G^eff Q_T) · t = Q_T h^eff

where:
    G^eff = G - G_apx + Q_S G_apx Q_S    (exact H0 minus Kronecker sum)
    h^eff = h + G_apx s                     (RHS corrected by supporting solution)

The trailing subspace is effectively the supporting subspace with the
leading-subspace projection removed (which is handled by the fact that
G^eff is zero in the leading subspace by construction).

The energy contribution is:
    E^eff_T = ⟨h^eff | t⟩

In the initial implementation, we solve the full IC cases 1-11 with the
exact H0 (including sigma operator coupling) and then subtract the G_apx
part. This gives the correct E^eff_T without explicitly constructing
G^eff.

Equivalently, the total PT2 energy from the IC perspective is:
    E_IC = Σ ⟨v | t_exact⟩        (for all 13 cases)
    E_IC = E^MP2_H± + E_cases_1-11

And the SST decomposition gives:
    E_IC = E^MP2_dressed - E^eff_S + E^eff_T

So:
    E^eff_T = E_cases_1-11 - E^MP2_dressed + E^MP2_H± + E^eff_S

For correctness verification, we solve cases 1-11 with the exact IC
machinery and derive E^eff_T from the identity above.

For the actual reduced-scaling implementation, E^eff_T would be computed
by solving the trailing equation with G^eff directly.
"""
from __future__ import annotations

import numpy as np

from asuka.caspt2.energy import _get_external_energies
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.shifts import (
    apply_real_shift,
    compute_shift_correction,
)
from asuka.caspt2.sigma import (
    SigmaC1ActiveVirtualCoupling,
    SigmaC1CaseCoupling,
)
from asuka.caspt2.solver import pcg_solve, pcg_solve_iterative
from asuka.caspt2.superindex import SuperindexMap

from asuka.caspt2.sst.coupling import DressedOrbitals
from asuka.caspt2.sst.subspaces import CASE_NAMES
from asuka.caspt2.sst.types import SSTConfig, SSTInput

__all__ = ["solve_trailing_subspace"]

# Case names for the 13 IC cases (used for verbose output)
_CASE_NAMES_13 = [
    "A(VJTU)", "B+(VJTIP)", "B-(VJTIM)", "C(ATVX)", "D(AIVX)",
    "E+(VJAIP)", "E-(VJAIM)", "F+(BVATP)", "F-(BVATM)",
    "G+(BJATQ)", "G-(BJATM)", "H+(BJAIP)", "H-(BJAIM)",
]


from asuka.caspt2.sst.reduced_system import _build_full_mo_eris  # noqa: F401


class _SigmaTrailing:
    """Sigma operator wrapper for trailing subspace (cases 1-11 only).

    Pads input from 11 to 13 elements (empty H± blocks), calls the
    inner sigma, and returns only the first 11 outputs.
    """

    def __init__(self, inner) -> None:
        self._inner = inner

    def __call__(self, vec_in_11: list[np.ndarray]) -> list[np.ndarray]:
        vec_13 = list(vec_in_11) + [
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        ]
        sigma_13 = self._inner(vec_13)
        return sigma_13[:11]


def solve_trailing_subspace(
    inp: SSTInput,
    dressed: DressedOrbitals,
    cfg: SSTConfig,
    *,
    df_blocks: object = None,
    device: int | None = None,
) -> tuple[float, list[np.ndarray] | None, bool, int]:
    """Solve the trailing subspace equation for cases 1-11.

    This implementation uses the exact IC machinery (S/B decomposition,
    sigma operator, PCG) for cases 1-11, giving the exact trailing
    subspace energy. This is equivalent to the ``build_and_solve_reduced_system``
    function in ``reduced_system.py`` but with the SST driver interface.

    The real level shift is applied to the H0 diagonal before solving.

    Parameters
    ----------
    inp : SSTInput
    dressed : DressedOrbitals
    cfg : SSTConfig
    df_blocks : CASPT2DFBlocks or None
        DF pair blocks. When provided, avoids building full MO ERIs.
    device : int or None
        CUDA device for GPU DF RHS builder.

    Returns
    -------
    e_trailing : float
        Energy from cases 1-11 (exact IC solution).
    amplitudes : list[ndarray] or None
        Per-case amplitude vectors (11 entries) in the SR basis.
    converged : bool
    niter : int
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
        print(f"SST trailing subspace (cases 1-11, exact IC)")

    # Build RHS dispatch: prefer DF blocks over full ERIs.
    _rhs_fn = None
    if df_blocks is not None and device is not None:
        from asuka.caspt2.cuda.rhs_df_cuda import build_rhs_df_cuda  # noqa: PLC0415
        import cupy as cp  # noqa: PLC0415
        _rhs_fn = lambda case: cp.asnumpy(
            build_rhs_df_cuda(case, smap, fock, df_blocks, dm1, dm2, device=device)
        )
    elif df_blocks is not None:
        from asuka.caspt2.rhs_df import build_rhs_df  # noqa: PLC0415
        _rhs_fn = lambda case: build_rhs_df(case, smap, fock, df_blocks, dm1, dm2)
    elif inp.B_ao is not None:
        from asuka.caspt2.rhs_df import build_rhs_df, build_df_blocks_cpu  # noqa: PLC0415
        _df = build_df_blocks_cpu(inp.B_ao, inp.mo_coeff, nish, nash, nssh)
        _rhs_fn = lambda case: build_rhs_df(case, smap, fock, _df, dm1, dm2)
    elif inp.eri_mo is not None:
        from asuka.caspt2.rhs import build_rhs  # noqa: PLC0415
        eri_mo = np.asarray(inp.eri_mo, dtype=np.float64)
        _rhs_fn = lambda case: build_rhs(case, smap, fock, eri_mo, dm1, dm2)
    else:
        raise ValueError("trailing subspace requires df_blocks, eri_mo, or B_ao")

    # Build S/B, diagonalize, build RHS for cases 1-11
    sb_decomps: list[SBDecomposition] = []
    h0_diags: list[np.ndarray] = []
    rhs_transformed: list[np.ndarray] = []
    smats_std: list[np.ndarray] = []

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
                print(f"  Case {case} ({_CASE_NAMES_13[case-1]}): skipped (empty)")
            continue

        smat = build_smat(case, smap, dm1, dm2, dm3)
        smats_std.append(smat)

        bmat = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_context)

        decomp = sbdiag(smat, bmat,
                        threshold_norm=cfg.threshold,
                        threshold_s=cfg.threshold_s)
        sb_decomps.append(decomp)

        if decomp.nindep == 0:
            h0_diags.append(np.empty(0, dtype=np.float64))
            rhs_transformed.append(np.empty(0, dtype=np.float64))
            if verbose >= 2:
                print(f"  Case {case} ({_CASE_NAMES_13[case-1]}): all lin-dep removed")
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
            print(f"  Case {case} ({_CASE_NAMES_13[case-1]}): "
                  f"nasup={nasup}, nisup={nisup}, nindep={decomp.nindep}")

    # Apply real level shift (no imaginary shift in SST mode)
    h0_diags_orig = [d.copy() for d in h0_diags]
    if abs(cfg.real_shift) > 1e-15:
        h0_diags = apply_real_shift(h0_diags, cfg.real_shift)

    # Determine if sigma operator is needed (off-diagonal Fock coupling)
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

        # Build 13-element lists for sigma operator (H± get empty placeholders)
        _empty_decomp = SBDecomposition(
            s_eigvals=np.empty(0), transform=np.empty((0, 0)),
            nindep=0, b_diag=np.empty(0),
        )
        sb_decomps_13 = list(sb_decomps) + [_empty_decomp, _empty_decomp]
        h0_diags_13 = list(h0_diags) + [
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        ]
        smats_13 = list(smats_std) + [
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
        ]

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

        sigma_op = _SigmaTrailing(inner_sigma)
        result = pcg_solve_iterative(
            sigma_op,
            rhs_transformed,
            h0_diags,
            tol=cfg.tol,
            maxiter=cfg.maxiter,
            verbose=verbose,
        )
    else:
        result = pcg_solve(
            h0_diags, rhs_transformed,
            tol=cfg.tol, maxiter=cfg.maxiter, verbose=verbose,
        )

    if verbose >= 1 and not result.converged:
        print(f"  WARNING: PCG did not converge (|r|={result.residual:.2e})")

    # Compute energy: E_trailing = Σ ⟨v | t⟩ for cases 1-11
    e_trailing = 0.0
    for case_idx, (v, t) in enumerate(zip(rhs_transformed, result.amplitudes)):
        if v.size == 0:
            continue
        e_case = float(np.dot(v.ravel(), t.ravel()))
        e_trailing += e_case
        if verbose >= 1:
            print(f"  Case {case_idx + 1} ({_CASE_NAMES_13[case_idx]}): "
                  f"E2 = {e_case:.10f}")

    # Shift correction for real shift
    if abs(cfg.real_shift) > 1e-15:
        e_shift_corr = compute_shift_correction(
            result.amplitudes, h0_diags, h0_diags_orig,
        )
        e_trailing -= e_shift_corr
        if verbose >= 1:
            print(f"  Shift correction (cases 1-11): {-e_shift_corr:.10f}")

    if verbose >= 1:
        print(f"  E_trailing (cases 1-11) = {e_trailing:.10f}")

    return float(e_trailing), result.amplitudes, result.converged, result.niter
