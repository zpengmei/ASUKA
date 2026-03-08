r"""Molcas-style PT2_Lag intermediates for SS-CASPT2.

This module hosts the superindex-ordering bridge (ASUKA C-order → Molcas
Fortran-order) and the validated Molcas translation pipeline used to build the
PT2_Lag intermediates.

Mathematical Background
-----------------------
The CASPT2 nuclear gradient requires solving the Z-vector equation and
constructing several Lagrangian intermediates:

- **CLag** (Configuration Lagrangian): Gradient of the CASPT2 energy with
  respect to the CI coefficients:

  .. math::

      \text{CLag}_I = \frac{\partial E_{\text{PT2}}}{\partial c_I}

- **OLag** (Orbital Lagrangian, antisymmetric): Gradient with respect to
  orbital rotations:

  .. math::

      \text{OLag}_{pq} = \frac{\partial E_{\text{PT2}}}{\partial \kappa_{pq}}

- **WLag**: AO renormalization contribution to the gradient, accounting
  for basis-set non-orthogonality.

- **DPT2 / DPT2C**: MO and AO PT2 density matrices, used for the
  one-electron and two-electron gradient contributions.

Superindex Ordering Bridge
~~~~~~~~~~~~~~~~~~~~~~~~~~
ASUKA builds superindex maps in C (row-major) order while the Molcas
translation code iterates in Fortran (column-major) order. This matters for:

- Cases A and C: triple indices ``(t,u,v)`` —
  ASUKA: ``t*nash²+u*nash+v`` vs Molcas: ``t+nash*u+nash²*v``
- Case D: full pair indices in 2-block structure —
  ASUKA: ``t*nash+u`` vs Molcas: ``t+nash*u``

Symmetric/antisymmetric pairs and single-index cases have identical
ordering in both conventions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# Re-exports from split modules (backward compatibility).
from asuka.caspt2.pt2lag_debug import (  # noqa: F401
    _asnumpy_f64,
    _read_molcas_dump_matrix,
    _infer_active_row_signs_from_dump,
    _best_col_signed_perm_2d,
    _best_col_signed_perm_assign_2d,
    _best_signed_perm_from_overlap_square,
    _infer_inact_virt_orbital_maps_from_dump,
    _case_external_map_from_orbital_perms,
    _collect_case_offdiag_from_breakdown,
    _collect_case_offdiag_from_dump_dir,
)
from asuka.caspt2.superindex_bridge import (  # noqa: F401
    _superindex_c_to_f_perm,
    _pair_sym_c_to_f_perm,
    _pair_asym_c_to_f_perm,
    _expand_block_perm,
    _case_c_to_f_perms,
    _parse_pair_perm_cases_env,
    _select_case_pair_perms,
)


def _build_case_amps_from_asuka(
    smap, fock, dm1, dm2, dm3, ci_ctx,
    amplitudes, orb, eri_mo,
    pt2_breakdown: dict[str, Any] | None = None,
    *,
    df_blocks: object | None = None,
):
    """Convert ASUKA CASPT2 energy result to CaseAmplitudes list.

    The CASPT2 energy solver stores amplitudes in an SR (spectrally-resolved)
    basis obtained from S/B joint diagonalisation in ASUKA's C-order superindex
    layout. The Molcas translation pipeline (CLagDX dispatchers, OLagNS, …)
    iterates superindices in Fortran column-major order.

    For cases whose superindex ordering differs (triples/pairs),
    we must:

    1. Decompose S/B in ASUKA order  →  ``decomp_orig``  (matches the amplitude
       basis produced by the energy solver).
    2. Reconstruct the amplitude in MO-superindex space:
       ``T_mo = trans_orig @ T_sr``.
    3. Permute ``T_mo``, ``smat``, ``bmat``, ``rhs_mo`` to Molcas order.
    4. Re-decompose S/B in Molcas order  →  ``decomp_perm``.
    5. Project ``T_mo`` into the permuted SR basis:
       ``T_sr_perm = trans_perm^T @ smat_perm @ T_mo_perm``.
    """
    from asuka.caspt2.overlap import build_smat, sbdiag  # noqa: PLC0415
    from asuka.caspt2.hzero import build_bmat  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.types import CaseAmplitudes  # noqa: PLC0415

    use_df_rhs = eri_mo is None
    if bool(use_df_rhs):
        if df_blocks is None:
            raise ValueError("df_blocks must be provided when eri_mo is None")
        from asuka.caspt2.rhs_df import build_rhs_df  # noqa: PLC0415
    else:
        from asuka.caspt2.rhs import build_rhs  # noqa: PLC0415

    nish = orb.nish
    nash = orb.nash
    nssh = orb.nssh
    _pair_perm_mode = str(os.environ.get("ASUKA_PT2LAG_PAIR_PERM_MODE", "off")).strip().lower()
    _pair_perm_cases = _parse_pair_perm_cases_env("")
    bd = dict(pt2_breakdown or {})

    # Optional OFFDIAG feeds from breakdown/native (ASUKA ordering).
    case_offdiag_asuka = _collect_case_offdiag_from_breakdown(bd)
    case_offdiag_molcas: dict[int, np.ndarray] = {}

    case_amps = []
    case_perm_meta: dict[int, dict[str, np.ndarray | None]] = {}
    for case_idx in range(13):
        case = case_idx + 1
        nasup = int(smap.nasup[case_idx])
        nisup = int(smap.nisup[case_idx])
        amp = amplitudes[case_idx]

        if nasup == 0 or nisup == 0 or amp.size == 0:
            case_amps.append(None)
            continue

        # Build S in ASUKA C-order superindex layout
        smat_orig = build_smat(case, smap, dm1, dm2, dm3)
        bmat_orig: np.ndarray | None = None

        # Build RHS in ASUKA ordering
        if bool(use_df_rhs):
            rhs_raw = build_rhs_df(case, smap, fock, df_blocks, dm1, dm2)  # type: ignore[arg-type]
        else:
            rhs_raw = build_rhs(case, smap, fock, eri_mo, dm1, dm2)
        rhs_mo_orig = rhs_raw.reshape(nasup, nisup)

        # Prefer the exact decomposition used by the CASPT2 energy solver when
        # available (avoids basis drift in near-linear-dependent channels).
        case_lbl = f"{int(case):02d}"
        trans_orig: np.ndarray | None = None
        eig_orig: np.ndarray | None = None
        rhs_sr_orig: np.ndarray | None = None
        nIN: int | None = None
        used_solver_sb = False
        try:
            k_tr = f"sb_transform_case{case_lbl}"
            k_bd = f"sb_bdiag_case{case_lbl}"
            k_ni = f"sb_nindep_case{case_lbl}"
            if k_tr in bd and k_bd in bd and k_ni in bd:
                tr = np.asarray(bd[k_tr], dtype=np.float64)
                bdiag = np.asarray(bd[k_bd], dtype=np.float64).reshape(-1)
                nindep = int(np.asarray(bd[k_ni]).reshape(-1)[0])
                if (
                    nindep > 0
                    and tr.ndim == 2
                    and tr.shape[0] == nasup
                    and tr.shape[1] == nindep
                    and bdiag.size >= nindep
                    and amp.size == nindep * nisup
                ):
                    trans_orig = np.asarray(tr, dtype=np.float64)
                    eig_orig = np.asarray(bdiag[:nindep], dtype=np.float64)
                    nIN = int(nindep)
                    used_solver_sb = True
                    k_rhs = f"rhs_sr_case{case_lbl}"
                    if k_rhs in bd:
                        rhs_try = np.asarray(bd[k_rhs], dtype=np.float64)
                        if rhs_try.shape == (nIN, nisup):
                            rhs_sr_orig = np.asarray(rhs_try, dtype=np.float64)
        except Exception:
            trans_orig = None
            eig_orig = None
            rhs_sr_orig = None
            nIN = None
            used_solver_sb = False

        # Fallback: rebuild decomposition from S/B directly.
        if nIN is None or trans_orig is None or eig_orig is None:
            bmat_orig = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_ctx)
            decomp_orig = sbdiag(smat_orig, bmat_orig)
            nIN = int(decomp_orig.nindep)
            if nIN == 0:
                case_amps.append(None)
                continue
            trans_orig = np.asarray(decomp_orig.transform, dtype=np.float64)
            eig_orig = np.asarray(decomp_orig.b_diag[:nIN], dtype=np.float64)
            used_solver_sb = False
        T_sr_orig = amp.reshape(nIN, nisup)

        perm_act_raw, perm_ext_raw = _case_c_to_f_perms(case, nish, nash, nssh)
        perm_act, perm_ext = _select_case_pair_perms(
            case=int(case),
            perm_act_raw=perm_act_raw,
            perm_ext_raw=perm_ext_raw,
            pair_perm_mode=str(_pair_perm_mode),
            pair_perm_cases=_pair_perm_cases,
        )
        sign_ext: np.ndarray | None = None

        if perm_act is None and perm_ext is None:
            # No permutation needed — ASUKA and Molcas orderings agree
            T_sr = T_sr_orig
            if rhs_sr_orig is not None and rhs_sr_orig.shape == (nIN, nisup):
                rhs_sr = np.asarray(rhs_sr_orig, dtype=np.float64)
            else:
                rhs_sr = np.asarray(trans_orig.T @ smat_orig @ rhs_mo_orig, dtype=np.float64)
            ca = CaseAmplitudes(
                case=case,
                T=T_sr,
                lbd=np.zeros_like(T_sr),
                rhs_sr=rhs_sr,
                rhs_mo=rhs_mo_orig,
                trans=trans_orig,
                eig=eig_orig,
                nAS=nasup,
                nIN=nIN,
                nIS=nisup,
                smat=smat_orig,
            )
        elif perm_act is None and perm_ext is not None:
            # External-only reorder (rows unchanged).
            rhs_mo_perm = rhs_mo_orig[:, perm_ext]
            T_sr_perm = T_sr_orig[:, perm_ext]
            if rhs_sr_orig is not None and rhs_sr_orig.shape == (nIN, nisup):
                rhs_sr_perm = np.asarray(rhs_sr_orig[:, perm_ext], dtype=np.float64)
            else:
                rhs_sr_perm = np.asarray(trans_orig.T @ smat_orig @ rhs_mo_perm, dtype=np.float64)
            if sign_ext is not None:
                s_ext = np.asarray(sign_ext, dtype=np.float64).reshape(1, nisup)
                rhs_mo_perm = np.asarray(rhs_mo_perm * s_ext, dtype=np.float64)
                T_sr_perm = np.asarray(T_sr_perm * s_ext, dtype=np.float64)
                rhs_sr_perm = np.asarray(rhs_sr_perm * s_ext, dtype=np.float64)
            ca = CaseAmplitudes(
                case=case,
                T=T_sr_perm,
                lbd=np.zeros_like(T_sr_perm),
                rhs_sr=rhs_sr_perm,
                rhs_mo=rhs_mo_perm,
                trans=trans_orig,
                eig=eig_orig,
                nAS=nasup,
                nIN=nIN,
                nIS=nisup,
                smat=smat_orig,
            )
        elif bool(used_solver_sb):
            # Active-index reorder without re-diagonalisation: permute solver SB
            # basis directly to preserve the exact SR basis used by the energy
            # solver (avoids basis drift in near-degenerate channels).
            smat_perm = smat_orig if perm_act is None else smat_orig[np.ix_(perm_act, perm_act)]
            trans_perm = trans_orig if perm_act is None else trans_orig[perm_act, :]

            rhs_mo_perm = rhs_mo_orig if perm_act is None else rhs_mo_orig[perm_act, :]
            T_sr_perm = T_sr_orig
            if rhs_sr_orig is not None and rhs_sr_orig.shape == (nIN, nisup):
                rhs_sr_perm = np.asarray(rhs_sr_orig, dtype=np.float64)
            else:
                rhs_sr_perm = np.asarray(trans_perm.T @ smat_perm @ rhs_mo_perm, dtype=np.float64)

            if perm_ext is not None:
                rhs_mo_perm = rhs_mo_perm[:, perm_ext]
                T_sr_perm = T_sr_perm[:, perm_ext]
                rhs_sr_perm = rhs_sr_perm[:, perm_ext]
                if sign_ext is not None:
                    s_ext = np.asarray(sign_ext, dtype=np.float64).reshape(1, nisup)
                    rhs_mo_perm = np.asarray(rhs_mo_perm * s_ext, dtype=np.float64)
                    T_sr_perm = np.asarray(T_sr_perm * s_ext, dtype=np.float64)
                    rhs_sr_perm = np.asarray(rhs_sr_perm * s_ext, dtype=np.float64)

            ca = CaseAmplitudes(
                case=case,
                T=T_sr_perm,
                lbd=np.zeros_like(T_sr_perm),
                rhs_sr=rhs_sr_perm,
                rhs_mo=rhs_mo_perm,
                trans=trans_perm,
                eig=eig_orig,
                nAS=nasup,
                nIN=nIN,
                nIS=nisup,
                smat=smat_perm,
            )
        else:
            # Step 2: reconstruct amplitude in MO-superindex space
            T_mo_orig = trans_orig @ T_sr_orig  # (nAS, nIS)

            # Step 3: permute to Molcas order
            if bmat_orig is None:
                bmat_orig = build_bmat(case, smap, fock, dm1, dm2, dm3, ci_context=ci_ctx)
            smat_perm = smat_orig if perm_act is None else smat_orig[np.ix_(perm_act, perm_act)]
            bmat_perm = bmat_orig if perm_act is None else bmat_orig[np.ix_(perm_act, perm_act)]
            rhs_mo_perm = rhs_mo_orig if perm_act is None else rhs_mo_orig[perm_act, :]
            T_mo_perm = T_mo_orig if perm_act is None else T_mo_orig[perm_act, :]
            if perm_ext is not None:
                rhs_mo_perm = rhs_mo_perm[:, perm_ext]
                T_mo_perm = T_mo_perm[:, perm_ext]
                if sign_ext is not None:
                    s_ext = np.asarray(sign_ext, dtype=np.float64).reshape(1, nisup)
                    rhs_mo_perm = np.asarray(rhs_mo_perm * s_ext, dtype=np.float64)
                    T_mo_perm = np.asarray(T_mo_perm * s_ext, dtype=np.float64)

            # Step 4: re-decompose in Molcas order
            decomp_perm = sbdiag(smat_perm, bmat_perm)
            nIN_perm = decomp_perm.nindep
            assert nIN_perm == nIN, (
                f"case {case}: nIN mismatch after permutation ({nIN} vs {nIN_perm})"
            )

            # Step 5: project amplitude into permuted SR basis
            T_sr_perm = decomp_perm.transform.T @ smat_perm @ T_mo_perm
            rhs_sr_perm = decomp_perm.transform.T @ smat_perm @ rhs_mo_perm

            ca = CaseAmplitudes(
                case=case,
                T=T_sr_perm,
                lbd=np.zeros_like(T_sr_perm),
                rhs_sr=rhs_sr_perm,
                rhs_mo=rhs_mo_perm,
                trans=decomp_perm.transform,
                eig=decomp_perm.b_diag[:nIN_perm],
                nAS=nasup,
                nIN=nIN_perm,
                nIS=nisup,
                smat=smat_perm,
            )
        case_perm_meta[int(case)] = {
            "perm_act": None if perm_act is None else np.asarray(perm_act, dtype=np.int64),
            "perm_act_raw": None if perm_act_raw is None else np.asarray(perm_act_raw, dtype=np.int64),
        }
        # Preserve raw ASUKA-basis vectors for native SIGDER OFFDIAG construction.
        # CLagDX and related translation lanes consume `ca.*` in Molcas ordering,
        # but SIGDER kernels are implemented in ASUKA ordering.
        try:
            ca._asuka_smat = np.asarray(smat_orig, dtype=np.float64)  # type: ignore[attr-defined]
            ca._asuka_trans = np.asarray(trans_orig, dtype=np.float64)  # type: ignore[attr-defined]
            ca._asuka_t_sr = np.asarray(T_sr_orig, dtype=np.float64)  # type: ignore[attr-defined]
            ca._asuka_perm_act = (
                None if perm_act is None else np.asarray(perm_act, dtype=np.int64)
            )  # type: ignore[attr-defined]
            ca._asuka_perm_ext = (
                None if perm_ext is None else np.asarray(perm_ext, dtype=np.int64)
            )  # type: ignore[attr-defined]
            ca._asuka_sign_ext = (
                None if sign_ext is None else np.asarray(sign_ext, dtype=np.float64)
            )  # type: ignore[attr-defined]
        except Exception:
            pass
        case_amps.append(ca)

    # Optional off-diagonal coupling matrix (STD/SIGDER contribution).
    # Priority:
    #   1) Molcas dump-dir source (already in Molcas ordering)
    #   2) ASUKA breakdown/native source (permute to Molcas ordering if needed)
    for ca in case_amps:
        if ca is None:
            continue
        case = int(ca.case)
        meta = dict(case_perm_meta.get(int(case), {}))
        perm_act = meta.get("perm_act")
        perm_act_raw = meta.get("perm_act_raw")
        offdiag_case = None
        if int(case) in case_offdiag_molcas:
            try:
                offdiag_case = np.asarray(case_offdiag_molcas[int(case)], dtype=np.float64)
                # Molcas dump is always in Molcas ordering. If this case kept
                # ASUKA ordering (perm disabled), map Molcas -> ASUKA.
                if perm_act is None and perm_act_raw is not None:
                    invp = np.empty_like(perm_act_raw)
                    invp[np.asarray(perm_act_raw, dtype=np.int64)] = np.arange(
                        int(perm_act_raw.size), dtype=np.int64
                    )
                    offdiag_case = offdiag_case[np.ix_(invp, invp)]
            except Exception:
                offdiag_case = None
        elif int(case) in case_offdiag_asuka:
            try:
                offdiag_case = np.asarray(case_offdiag_asuka[int(case)], dtype=np.float64)
                if perm_act is not None:
                    offdiag_case = offdiag_case[np.ix_(perm_act, perm_act)]
            except Exception:
                offdiag_case = None

        if offdiag_case is not None:
            try:
                if offdiag_case.shape != (int(ca.nAS), int(ca.nAS)):
                    if int(offdiag_case.size) == int(ca.nAS) * int(ca.nAS):
                        offdiag_case = np.asarray(
                            offdiag_case.reshape((int(ca.nAS), int(ca.nAS))),
                            dtype=np.float64,
                        )
                    else:
                        raise ValueError(
                            f"case {case}: OFFDIAG shape {offdiag_case.shape} "
                            f"!= {(int(ca.nAS), int(ca.nAS))}"
                        )
                ca.offdiag = np.asarray(offdiag_case, dtype=np.float64)
            except Exception:
                pass

    return case_amps


def _attach_native_offdiag_after_lambda(
    *,
    case_amps,
    smap,
    fock,
    dm1: np.ndarray,
    offdiag_strict: bool,
) -> None:
    """Populate missing case OFFDIAG using native SIGDER after lambda solve.

    OpenMolcas SIGDER contracts both iVecX (T) and iVecR (lambda-like) vectors.
    Therefore native OFFDIAG generation must run after `solve_lambda` has filled
    `ca.lbd` in each case.
    """
    _offdiag_native = str(os.environ.get("ASUKA_PT2LAG_OFFDIAG_NATIVE", "0")).strip().lower() not in {
        "", "0", "off", "false", "no",
    }
    if not _offdiag_native:
        return

    try:
        from asuka.caspt2.sigder_native import build_sigder_offdiag_asuka_c1  # noqa: PLC0415

        def _invert_perm(perm: np.ndarray | None) -> np.ndarray | None:
            if perm is None:
                return None
            p = np.asarray(perm, dtype=np.int64).reshape(-1)
            if p.size == 0:
                return None
            inv = np.empty_like(p)
            inv[p] = np.arange(int(p.size), dtype=np.int64)
            return inv

        def _lambda_sr_asuka(
            *,
            ca: Any,
            smat_asuka: np.ndarray,
            trans_asuka: np.ndarray,
        ) -> np.ndarray:
            """Map case lambda from current case-amp ordering to ASUKA ordering."""
            lbd_cur = np.asarray(getattr(ca, "lbd"), dtype=np.float64)
            perm_act = getattr(ca, "_asuka_perm_act", None)
            perm_ext = getattr(ca, "_asuka_perm_ext", None)
            sign_ext = getattr(ca, "_asuka_sign_ext", None)
            if perm_act is None and perm_ext is None:
                return lbd_cur

            trans_cur = np.asarray(getattr(ca, "trans"), dtype=np.float64)
            if (
                lbd_cur.ndim != 2
                or trans_cur.ndim != 2
                or trans_cur.shape[1] != lbd_cur.shape[0]
            ):
                return lbd_cur

            lbd_mo = np.asarray(trans_cur @ lbd_cur, dtype=np.float64)
            inv_act = _invert_perm(None if perm_act is None else np.asarray(perm_act, dtype=np.int64))
            inv_ext = _invert_perm(None if perm_ext is None else np.asarray(perm_ext, dtype=np.int64))

            if inv_act is not None:
                if int(inv_act.size) != int(lbd_mo.shape[0]):
                    return lbd_cur
                lbd_mo = np.asarray(lbd_mo[inv_act, :], dtype=np.float64)
            if inv_ext is not None:
                if int(inv_ext.size) != int(lbd_mo.shape[1]):
                    return lbd_cur
                lbd_mo = np.asarray(lbd_mo[:, inv_ext], dtype=np.float64)
                if sign_ext is not None:
                    s = np.asarray(sign_ext, dtype=np.float64).reshape(-1)
                    if int(s.size) == int(inv_ext.size):
                        lbd_mo = np.asarray(
                            lbd_mo * s[np.asarray(inv_ext, dtype=np.int64)][None, :],
                            dtype=np.float64,
                        )

            sm = np.asarray(smat_asuka, dtype=np.float64)
            tr = np.asarray(trans_asuka, dtype=np.float64)
            if (
                sm.ndim != 2
                or tr.ndim != 2
                or sm.shape[0] != sm.shape[1]
                or tr.shape[0] != sm.shape[0]
                or lbd_mo.shape[0] != sm.shape[0]
            ):
                return lbd_cur
            return np.asarray(tr.T @ sm @ lbd_mo, dtype=np.float64)

        smat_by_case: dict[int, np.ndarray] = {}
        trans_by_case: dict[int, np.ndarray] = {}
        t_sr_by_case: dict[int, np.ndarray] = {}
        lbd_sr_by_case: dict[int, np.ndarray] = {}
        for ca in case_amps:
            if ca is None:
                continue
            case = int(ca.case)
            if int(ca.nAS) <= 0 or int(ca.nIS) <= 0:
                continue
            smat_asuka = np.asarray(getattr(ca, "_asuka_smat", ca.smat), dtype=np.float64)
            trans_asuka = np.asarray(getattr(ca, "_asuka_trans", ca.trans), dtype=np.float64)
            t_sr_asuka = np.asarray(getattr(ca, "_asuka_t_sr", ca.T), dtype=np.float64)
            lbd_sr_asuka = _lambda_sr_asuka(
                ca=ca,
                smat_asuka=smat_asuka,
                trans_asuka=trans_asuka,
            )

            smat_by_case[case] = smat_asuka
            trans_by_case[case] = trans_asuka
            t_sr_by_case[case] = t_sr_asuka
            lbd_sr_by_case[case] = lbd_sr_asuka

        nactel = int(round(np.trace(np.asarray(dm1, dtype=np.float64))))
        native_offdiag = build_sigder_offdiag_asuka_c1(
            smap=smap,
            fock=fock,
            smat_by_case=smat_by_case,
            trans_by_case=trans_by_case,
            t_sr_by_case=t_sr_by_case,
            lbd_sr_by_case=lbd_sr_by_case,
            nactel=max(1, int(nactel)),
            vecrot=1.0,
        )

        for ca in case_amps:
            if ca is None:
                continue
            # Keep existing source precedence: do not override injected dump OFFDIAG.
            if getattr(ca, "offdiag", None) is not None:
                continue
            cv = native_offdiag.get(int(ca.case))
            if cv is None:
                continue
            arr = np.asarray(cv, dtype=np.float64)
            perm_act = getattr(ca, "_asuka_perm_act", None)
            if perm_act is not None:
                p = np.asarray(perm_act, dtype=np.int64).reshape(-1)
                if int(p.size) == int(arr.shape[0]):
                    arr = np.asarray(arr[np.ix_(p, p)], dtype=np.float64)
            shape_ref = (int(ca.nAS), int(ca.nAS))
            if arr.shape != shape_ref:
                if int(arr.size) == shape_ref[0] * shape_ref[1]:
                    arr = np.asarray(arr.reshape(shape_ref), dtype=np.float64)
                else:
                    raise ValueError(
                        f"case {int(ca.case)}: native OFFDIAG shape {arr.shape} != {shape_ref}"
                    )
            ca.offdiag = np.asarray(arr, dtype=np.float64)

    except Exception:
        if bool(offdiag_strict):
            raise


class _DRTSigmaAdapterNative:
    """DRT sigma adapter for ASUKA-convention data (no orbital reversal).

    Unlike DRTSigmaAdapter in cnstclag.py (which reverses orbital indices to
    convert Molcas-convention DG/DF to ASUKA GUGA convention), this adapter uses
    E_tu directly in ASUKA convention. This is needed when the DG/DF arrays were
    built from ASUKA data (which is already in ASUKA convention).

    Uses trans_rdm1 to build exact E_tu generator matrices:
      E_tu[t, u, I, J] = <I|a†_t a_u|J>  (ASUKA orbital ordering)
    """

    def __init__(self, solver, drt, nelec):
        self.solver = solver
        self.drt = drt
        self.norb = drt.norb
        self.nelec = nelec
        self._E_tu = None
        self._ncsf = drt.ncsf

    def _build_E_tu(self) -> np.ndarray:
        norb = self.norb
        ncsf = self._ncsf
        E = np.zeros((norb, norb, ncsf, ncsf))
        for J in range(ncsf):
            e_J = np.zeros(ncsf)
            e_J[J] = 1.0
            for I in range(ncsf):
                e_I = np.zeros(ncsf)
                e_I[I] = 1.0
                # trans_rdm1(bra, ket) returns rdm[p,q] = <bra|E_qp|ket>
                # (transposed convention). Transpose to get E_pq convention.
                rdm = self.solver.trans_rdm1(e_I, e_J, norb, self.nelec)
                E[:, :, I, J] = rdm.T
        # E[t,u,I,J] = <I|E_tu|J> in ASUKA orbital + CSF convention.
        return E

    def _get_E_tu(self) -> np.ndarray:
        if self._E_tu is None:
            self._E_tu = self._build_E_tu()
        return self._E_tu

    def contract_1e(self, h1e: np.ndarray, ci: np.ndarray) -> np.ndarray:
        """Sigma 1e: sum_{tu} h[t,u] * E_tu @ ci (ASUKA convention)."""
        E = self._get_E_tu()
        norb = self.norb
        result = np.zeros_like(ci)
        for t in range(norb):
            for u in range(norb):
                if abs(h1e[t, u]) > 1e-20:
                    result += h1e[t, u] * (E[t, u] @ ci)
        return result

    def contract_2e(self, eri: np.ndarray, ci: np.ndarray) -> np.ndarray:
        """Sigma 2e: sum_{tuvx} eri[t,u,v,x] * e_tuvx @ ci (ASUKA convention).

        e_tuvx = E_tu E_vx - delta_uv E_tx
        """
        E = self._get_E_tu()
        norb = self.norb
        eri_4d = eri.reshape(norb, norb, norb, norb)
        result = np.zeros_like(ci)
        for t in range(norb):
            for u in range(norb):
                for v in range(norb):
                    for x in range(norb):
                        if abs(eri_4d[t, u, v, x]) < 1e-20:
                            continue
                        evx_ci = E[v, x] @ ci
                        etu_evx_ci = E[t, u] @ evx_ci
                        term = etu_evx_ci
                        if u == v:
                            term -= E[t, x] @ ci
                        result += eri_4d[t, u, v, x] * term
        return result


def _build_lagrangians(
    case_amps, smap, orb, fock, rdms,
    ci, ncsf, eri_mo, cmo, s_ao, drt,
    verbose=0,
    *,
    B_ao=None, h_ao=None, e_nuc=None, twos=0,
    df_blocks: object | None = None,
    collect_breakdown: bool = False,
):
    """Build all CASPT2 Lagrangians using the Molcas translation pipeline.

    Follows the exact sequence from full_gradient_pipeline() in
    tools/molcas_caspt2_grad/translation/gradient_assembly.py, which is validated
    against Molcas reference data.

    The DG/DF arrays from accumulate_dg_df are in ASUKA orbital convention
    (same as the smat/bmat/rhs that produced them), so the sigma adapter must
    NOT reverse orbital indices.
    """
    from tools.molcas_caspt2_grad.translation.caspt2_res import solve_lambda  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.trdns2d import trdns2d  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.trdns2o import trdns2o  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.clagd import clag_driver  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.olagns import build_olagns2  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.olagvvvo import olagvvvo  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.eigder import eigder, eigder_clag  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.depsa import add_depsa, symmetrize_depsa  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.density import dpt2_trf  # noqa: PLC0415
    from tools.molcas_caspt2_grad.translation.cnstclag import make_drt_adapter  # noqa: PLC0415
    from asuka.solver import GUGAFCISolver  # noqa: PLC0415

    nash = orb.nash
    nactel = int(round(np.trace(rdms.dm1)))

    # DRT sigma adapter mode:
    # - native: ASUKA ordering throughout (default)
    # - molcas: Molcas-convention adapter (orbital reversal in cnstclag)
    _drt_mode = str(os.environ.get("ASUKA_CLAG_DRT_MODE", "native")).strip().lower()
    if _drt_mode == "molcas":
        drt_adapter = make_drt_adapter(drt, nactel)
    else:
        solver = GUGAFCISolver()
        drt_adapter = _DRTSigmaAdapterNative(solver, drt, nactel)

    stage_checkpoints: dict[str, Any] = {}
    stage_order: list[str] = []

    # Step 1: Lambda equations (trivial for SS without shift)
    lbd_norms = np.zeros((13,), dtype=np.float64)
    lbd_nindep = np.zeros((13,), dtype=np.int32)
    for ca in case_amps:
        if ca is not None and ca.nAS > 0 and ca.nIS > 0:
            ca.lbd[:] = solve_lambda(ca)
            idx = int(ca.case) - 1
            if 0 <= idx < 13:
                lbd_norms[idx] = float(np.linalg.norm(ca.lbd))
                lbd_nindep[idx] = int(ca.nIN)
    if bool(collect_breakdown):
        stage_order.append("s1_lambda")
        stage_checkpoints["s1_lambda"] = {
            "lbd_case_norms": np.asarray(lbd_norms, dtype=np.float64),
            "lbd_case_nindep": np.asarray(lbd_nindep, dtype=np.int32),
        }

    # Optional native SIGDER OFFDIAG path must run after lambda is available.
    _attach_native_offdiag_after_lambda(
        case_amps=case_amps,
        smap=smap,
        fock=fock,
        dm1=np.asarray(rdms.dm1, dtype=np.float64),
        offdiag_strict=False,
    )

    # Step 2-3: Diagonal + off-diagonal PT2 density
    dpt2 = trdns2d(case_amps, orb, rdms.dm1)
    dpt2 = trdns2o(case_amps, orb, rdms.dm1, dpt2)
    dpt2_bare = np.asarray(0.5 * (dpt2 + dpt2.T), dtype=np.float64, order="C").copy()
    dpt2 = np.asarray(dpt2_bare, dtype=np.float64, order="C").copy()
    if bool(collect_breakdown):
        stage_order.append("s3_dpt2_density")
        stage_checkpoints["s3_dpt2_density"] = {
            "dpt2_bare": np.asarray(dpt2_bare, dtype=np.float64),
        }

    # Step 4: CLag via the full translation pipeline (clagdx + cnstclag)
    clag_case_amp_terms: dict[str, dict[str, Any]] = {}
    DG1, DG2, DF1, DF2, depsa, clag, deasum = clag_driver(
        case_amps, orb, fock, rdms,
        ci, ncsf, drt=drt_adapter,
        collect_terms=False,
        collect_case_terms=False,
    )
    clag_debug = getattr(clag_driver, "_last_details", None)
    clag_sigma_terms: dict[str, np.ndarray] = {}
    clag_case_terms: dict[str, dict[str, Any]] = {}
    clagdx_case_terms: dict[str, dict[str, Any]] = {}
    clag_deasum_post_terms: dict[str, Any] = {}
    if isinstance(clag_debug, dict):
        for kk, vv in dict(clag_debug.get("sigma_terms", {}) or {}).items():
            try:
                clag_sigma_terms[str(kk)] = np.asarray(vv, dtype=np.float64)
            except Exception:
                continue
        for ck, cvals in dict(clag_debug.get("case_terms", {}) or {}).items():
            if not isinstance(cvals, dict):
                continue
            rec: dict[str, Any] = {}
            for k2, v2 in dict(cvals).items():
                if v2 is None:
                    continue
                if isinstance(v2, (int, float, np.integer, np.floating)):
                    rec[str(k2)] = float(v2)
                    continue
                try:
                    rec[str(k2)] = np.asarray(v2, dtype=np.float64)
                except Exception:
                    continue
            if rec:
                clag_case_terms[str(ck)] = rec
        for ck, cvals in dict(clag_debug.get("clagdx_case_terms", {}) or {}).items():
            if not isinstance(cvals, dict):
                continue
            rec: dict[str, Any] = {}
            for k2, v2 in dict(cvals).items():
                if v2 is None:
                    continue
                try:
                    rec[str(k2)] = np.asarray(v2, dtype=np.float64)
                except Exception:
                    continue
            if rec:
                clagdx_case_terms[str(ck)] = rec
        _post = dict(clag_debug.get("deasum_post_terms", {}) or {})
        for pk, pv in _post.items():
            if pv is None:
                continue
            if isinstance(pv, (int, float, np.integer, np.floating)):
                clag_deasum_post_terms[str(pk)] = float(pv)
                continue
            try:
                clag_deasum_post_terms[str(pk)] = np.asarray(pv, dtype=np.float64)
            except Exception:
                continue
    clag_raw = np.asarray(clag, dtype=np.float64, order="C").copy()
    if bool(collect_breakdown):
        stage_order.append("s5_clag_derfg3")
        s5 = {
            "DG1": np.asarray(DG1, dtype=np.float64),
            "DG2": np.asarray(DG2, dtype=np.float64),
            "DF1": np.asarray(DF1, dtype=np.float64),
            "DF2": np.asarray(DF2, dtype=np.float64),
            "G1": np.asarray(rdms.dm1, dtype=np.float64),
            "G2": np.asarray(rdms.dm2.reshape(nash * nash, nash * nash), dtype=np.float64),
            "depsa_raw": np.asarray(depsa, dtype=np.float64),
            "deasum": np.asarray(deasum, dtype=np.float64),
            "ci_ref": np.asarray(ci, dtype=np.float64),
            "clag_vec_raw": np.asarray(clag_raw, dtype=np.float64),
            # Keep legacy key for downstream tools that expect `clag_vec`.
            "clag_vec": np.asarray(clag_raw, dtype=np.float64),
        }
        if clag_sigma_terms:
            s5["clag_sigma_terms"] = dict(clag_sigma_terms)
        if clag_case_terms:
            s5["clag_case_terms"] = dict(clag_case_terms)
        if clagdx_case_terms:
            s5["clagdx_case_terms"] = dict(clagdx_case_terms)
        if clag_case_amp_terms:
            s5["clag_case_amp_terms"] = dict(clag_case_amp_terms)
        if clag_deasum_post_terms:
            s5["clag_deasum_post_terms"] = dict(clag_deasum_post_terms)
        stage_checkpoints["s5_clag_derfg3"] = s5

    # Step 5: Symmetrize DEPSA and add to DPT2
    depsa = symmetrize_depsa(depsa)
    dpt2 = add_depsa(dpt2, depsa, rdms.dm1, orb)
    dpt2 = np.asarray(0.5 * (dpt2 + dpt2.T), dtype=np.float64, order="C")
    if bool(collect_breakdown):
        s3 = dict(stage_checkpoints.get("s3_dpt2_density", {}))
        s3["depsa_used"] = np.asarray(depsa, dtype=np.float64)
        s3["dpt2_full"] = np.asarray(dpt2, dtype=np.float64)
        stage_checkpoints["s3_dpt2_density"] = s3

    # Step 6: DPT2 → AO
    dpt2_ao = dpt2_trf(dpt2, cmo)

    # Step 7: OLag from OLagNS2 (also produces DPT2C and VVVO intermediates)
    if eri_mo is None:
        from asuka.caspt2.pt2lag_df2e import build_olagns2_df  # noqa: PLC0415

        if df_blocks is None:
            raise ValueError("DF OLagNS2 requested but df_blocks was not provided")
        olag, dpt2c, tildeT = build_olagns2_df(
            case_amps, orb, fock, rdms, df_blocks, nactel=nactel,
        )
        t2_mo = np.empty((0,), dtype=np.float64)
        olagns2_details = getattr(build_olagns2_df, "_last_details", None)
    else:
        olag, dpt2c, t2_mo = build_olagns2(
            case_amps, orb, fock, rdms, eri_mo, nactel=nactel,
        )
        tildeT = None
        olagns2_details = getattr(build_olagns2, "_last_details", None)
    olag_case_terms: dict[str, np.ndarray] = {}
    dpt2c_case_terms: dict[str, np.ndarray] = {}
    dpt2c_case_terms_pre_scale: dict[str, np.ndarray] = {}
    tc_case_terms: dict[str, np.ndarray] = {}
    olag_case_subterms: dict[str, dict[str, np.ndarray]] = {}
    dpt2c_case_subterms_pre_scale: dict[str, dict[str, np.ndarray]] = {}
    dpt2c_case_subterms: dict[str, dict[str, np.ndarray]] = {}

    olagns2_scale = float(1.0 / max(1, nactel))
    olagns2_nactel = int(nactel)
    olagns2_scale_mode = "nactel"
    if isinstance(olagns2_details, dict):
        for kk, vv in dict(olagns2_details.get("olag_case_terms", {}) or {}).items():
            try:
                olag_case_terms[str(kk)] = np.asarray(vv, dtype=np.float64)
            except Exception:
                continue
        for kk, vv in dict(olagns2_details.get("dpt2c_case_terms", {}) or {}).items():
            try:
                dpt2c_case_terms[str(kk)] = np.asarray(vv, dtype=np.float64)
            except Exception:
                continue
        for kk, vv in dict(olagns2_details.get("dpt2c_case_terms_pre_scale", {}) or {}).items():
            try:
                dpt2c_case_terms_pre_scale[str(kk)] = np.asarray(vv, dtype=np.float64)
            except Exception:
                continue

        # Optional deeper per-subterm breakdown (env-gated in build_olagns2).
        for ck, sub in dict(olagns2_details.get("olag_case_subterms", {}) or {}).items():
            try:
                sub_d = dict(sub) if isinstance(sub, dict) else {}
                olag_case_subterms[str(ck)] = {str(sk): np.asarray(sv, dtype=np.float64) for sk, sv in sub_d.items()}
            except Exception:
                continue
        for ck, sub in dict(olagns2_details.get("dpt2c_case_subterms_pre_scale", {}) or {}).items():
            try:
                sub_d = dict(sub) if isinstance(sub, dict) else {}
                dpt2c_case_subterms_pre_scale[str(ck)] = {str(sk): np.asarray(sv, dtype=np.float64) for sk, sv in sub_d.items()}
            except Exception:
                continue
        for ck, sub in dict(olagns2_details.get("dpt2c_case_subterms", {}) or {}).items():
            try:
                sub_d = dict(sub) if isinstance(sub, dict) else {}
                dpt2c_case_subterms[str(ck)] = {str(sk): np.asarray(sv, dtype=np.float64) for sk, sv in sub_d.items()}
            except Exception:
                continue

        try:
            olagns2_scale = float(olagns2_details.get("dpt2c_scale", olagns2_scale))
        except Exception:
            pass
        try:
            olagns2_scale_mode = str(olagns2_details.get("dpt2c_scale_mode", olagns2_scale_mode))
        except Exception:
            pass
        try:
            olagns2_nactel = int(olagns2_details.get("nactel", olagns2_nactel))
        except Exception:
            pass

    # Covariant amplitudes (TC) used by OLagNS:
    #   TC = S * TRANS * T_sr
    # Keep per-case matrices for direct parity checks against Molcas olagns.f
    # debug dumps (`ASUKA_TC`, `ASUKA_TC+`, `ASUKA_TC-`).
    for ca in case_amps:
        if ca is None:
            continue
        try:
            nAS = int(getattr(ca, "nAS", 0))
            nIS = int(getattr(ca, "nIS", 0))
            if nAS <= 0 or nIS <= 0:
                continue
            sm = np.asarray(getattr(ca, "smat"), dtype=np.float64)
            tr = np.asarray(getattr(ca, "trans"), dtype=np.float64)
            t_sr = np.asarray(getattr(ca, "T"), dtype=np.float64).reshape(int(getattr(ca, "nIN")), nIS)
            tc = np.asarray(sm @ tr @ t_sr, dtype=np.float64)
            tc_case_terms[str(int(getattr(ca, "case")))] = tc
        except Exception:
            continue

    olag_ns2 = np.asarray(olag, dtype=np.float64, order="C").copy()
    dpt2c_mo = np.asarray(dpt2c, dtype=np.float64, order="C").copy()
    dpt2c_ao = dpt2_trf(dpt2c_mo, cmo)
    if bool(collect_breakdown):
        stage_order.append("s2_olagns2")
        s2 = {
            "olag_ns2_total": np.asarray(olag_ns2, dtype=np.float64),
            "dpt2c_total": np.asarray(dpt2c_mo, dtype=np.float64),
            "dpt2c_ao_total": np.asarray(dpt2c_ao, dtype=np.float64),
            "olag_case_terms": dict(olag_case_terms),
            "dpt2c_case_terms": dict(dpt2c_case_terms),
            "dpt2c_case_terms_pre_scale": dict(dpt2c_case_terms_pre_scale),
            "tc_case_terms": dict(tc_case_terms),
            "dpt2c_scale": np.asarray([float(olagns2_scale)], dtype=np.float64),
            "dpt2c_scale_mode": np.asarray(
                [str(olagns2_scale_mode)],
                dtype=f"<U{max(1, len(str(olagns2_scale_mode)))}",
            ),
            "nactel": np.asarray([int(olagns2_nactel)], dtype=np.int32),
        }
        if olag_case_subterms:
            s2["olag_case_subterms"] = dict(olag_case_subterms)
        if dpt2c_case_subterms_pre_scale:
            s2["dpt2c_case_subterms_pre_scale"] = dict(dpt2c_case_subterms_pre_scale)
        if dpt2c_case_subterms:
            s2["dpt2c_case_subterms"] = dict(dpt2c_case_subterms)
        stage_checkpoints["s2_olagns2"] = s2

    # Step 8: OLagVVVO + FPT2
    if eri_mo is None:
        from asuka.caspt2.pt2lag_df2e import olagvvvo_df  # noqa: PLC0415

        if df_blocks is None:
            raise ValueError("DF OLagVVVO requested but df_blocks was not provided")
        if B_ao is None:
            raise ValueError("DF OLagVVVO requested but B_ao was not provided")
        _aux_blk = int(os.environ.get("ASUKA_PT2LAG_VVVO_AUX_BLOCK_NAUX", "256"))
        fpt2_ao, fpt2c_ao, olag = olagvvvo_df(
            olag,
            dpt2,
            dpt2c,
            orb=orb,
            df_blocks=df_blocks,
            tildeT=tildeT,
            cmo=cmo,
            s_ao=s_ao,
            B_ao=B_ao,
            aux_block_naux=int(max(1, _aux_blk)),
        )
    else:
        fpt2_ao, fpt2c_ao, olag = olagvvvo(
            olag, dpt2, dpt2c, fock, orb, eri_mo, cmo,
            t2_mo=t2_mo, s_ao=s_ao,
        )
    olag_before_eigder = np.asarray(olag, dtype=np.float64, order="C").copy()

    # Step 9: EigDer
    nish = int(orb.nish)
    nash = int(orb.nash)
    fpt2_mo = np.asarray(cmo.T @ fpt2_ao @ cmo, dtype=np.float64)
    fpt2c_mo = np.asarray(cmo.T @ fpt2c_ao @ cmo, dtype=np.float64)
    fpt2_loc = np.asarray(2.0 * fpt2_mo, dtype=np.float64)
    fpt2c_loc = np.asarray(2.0 * fpt2c_mo, dtype=np.float64)
    dm1_loc = np.asarray(rdms.dm1, dtype=np.float64)

    eig_t1 = np.zeros_like(olag_before_eigder, dtype=np.float64)
    eig_t1[:, :nish] = 2.0 * fpt2_loc[:, :nish]
    eig_t2 = np.zeros_like(olag_before_eigder, dtype=np.float64)
    if nash > 0:
        eig_t2[:, nish : nish + nash] = fpt2_loc[:, nish : nish + nash] @ dm1_loc
    eig_t3 = np.asarray(2.0 * (fock.fifa @ dpt2.T), dtype=np.float64)
    eig_t4 = np.asarray(fock.fimo @ dpt2c.T + fock.fimo.T @ dpt2c, dtype=np.float64)
    eig_t5 = np.zeros_like(olag_before_eigder, dtype=np.float64)
    eig_t5[:, :nish] = 2.0 * fpt2c_loc[:, :nish]
    eig_total = np.asarray(eig_t1 + eig_t2 + eig_t3 + eig_t4 + eig_t5, dtype=np.float64)

    olag, rdmeig = eigder(
        olag,
        fpt2_ao=fpt2_ao,
        fpt2c_ao=fpt2c_ao,
        cmopt2=cmo,
        dpt2_mo=dpt2,
        dpt2c_mo=dpt2c,
        fock=fock,
        rdms=rdms,
        orb=orb,
    )
    olag_after_eigder = np.asarray(olag, dtype=np.float64, order="C").copy()
    if bool(collect_breakdown):
        stage_order.append("s4_eigder")
        stage_checkpoints["s4_eigder"] = {
            "olag_bare": np.asarray(olag_before_eigder, dtype=np.float64),
            "olag_full": np.asarray(olag_after_eigder, dtype=np.float64),
            "rdmeig": np.asarray(rdmeig, dtype=np.float64),
            "fpt2_ao": np.asarray(fpt2_ao, dtype=np.float64),
            "fpt2c_ao": np.asarray(fpt2c_ao, dtype=np.float64),
            "fpt2_mo": np.asarray(fpt2_mo, dtype=np.float64),
            "fpt2c_mo": np.asarray(fpt2c_mo, dtype=np.float64),
            "terms_full_dpt2": {
                "olag_inact_fpt2": np.asarray(eig_t1, dtype=np.float64),
                "olag_act_fpt2": np.asarray(eig_t2, dtype=np.float64),
                "olag_fifa_dpt2": np.asarray(eig_t3, dtype=np.float64),
                "olag_fimo_dpt2c": np.asarray(eig_t4, dtype=np.float64),
                "olag_inact_fpt2c": np.asarray(eig_t5, dtype=np.float64),
                "olag_add_total": np.asarray(eig_total, dtype=np.float64),
            },
        }

    # Step 9b: CLagEig — add RDMEIG contributions to CLag
    clag_before_eigder = np.asarray(clag, dtype=np.float64, order="C").copy()
    clag = eigder_clag(clag, rdmeig, ci, nash, ncsf, drt=drt_adapter)
    clag_before_projection = np.asarray(clag, dtype=np.float64, order="C").copy()
    clag_eigder_add = np.asarray(clag_before_projection - clag_before_eigder, dtype=np.float64)

    # Step 9c: Project CLag orthogonal to reference CI vector
    clag_projection_parallel = np.asarray(np.dot(clag_before_projection, ci) * ci, dtype=np.float64)
    clag = np.asarray(clag_before_projection - clag_projection_parallel, dtype=np.float64)
    if bool(collect_breakdown):
        s5 = dict(stage_checkpoints.get("s5_clag_derfg3", {}))
        s5["clag_vec_eigder_add"] = np.asarray(clag_eigder_add, dtype=np.float64)
        s5["clag_vec_before_projection"] = np.asarray(clag_before_projection, dtype=np.float64)
        s5["clag_vec_projection_parallel"] = np.asarray(clag_projection_parallel, dtype=np.float64)
        s5["clag_vec"] = np.asarray(clag, dtype=np.float64)
        stage_checkpoints["s5_clag_derfg3"] = s5

    # Step 10: WLag and OLag finalization
    # WLag: W_PT2_AO = C @ (0.5 * OLag) @ C^T  (before antisymmetrization)
    wlag_ao = cmo @ (0.5 * olag) @ cmo.T

    # Antisymmetrize OLag
    olag_antisym = olag - olag.T
    if bool(collect_breakdown):
        stage_order.append("s8_wlag_olagfinal")
        stage_checkpoints["s8_wlag_olagfinal"] = {
            "W_pt2_ao": np.asarray(wlag_ao, dtype=np.float64),
            "olag_antisym": np.asarray(olag_antisym, dtype=np.float64),
        }

    if verbose >= 1:
        print(f"[CASPT2 pt2lag] |CLag|={np.linalg.norm(clag):.6e}")
        print(f"[CASPT2 pt2lag] |OLag|={np.linalg.norm(olag_antisym):.6e}")
        print(f"[CASPT2 pt2lag] |WLag_AO|={np.linalg.norm(wlag_ao):.6e}")
        print(f"[CASPT2 pt2lag] |DPT2|={np.linalg.norm(dpt2):.6e}")

    # Keep exported stage order deterministic and aligned with strict replay naming.
    _canonical_stage_order = [
        "s1_lambda",
        "s2_olagns2",
        "s3_dpt2_density",
        "s4_eigder",
        "s5_clag_derfg3",
        "s8_wlag_olagfinal",
    ]
    if stage_checkpoints:
        stage_order = [s for s in _canonical_stage_order if s in stage_checkpoints]
        stage_order.extend([s for s in stage_checkpoints.keys() if s not in stage_order])
        stage_checkpoints = {s: stage_checkpoints[s] for s in stage_order}

    return {
        "clag": clag,
        "olag": olag_antisym,
        "wlag_ao": wlag_ao,
        "dpt2": dpt2,
        "dpt2_bare": np.asarray(dpt2_bare, dtype=np.float64),
        "dpt2c": dpt2c,
        "olag_ns2": np.asarray(olag_ns2, dtype=np.float64),
        "dpt2c_ao": np.asarray(dpt2c_ao, dtype=np.float64),
        "olagns2_case_terms": dict(olag_case_terms),
        "dpt2c_case_terms": dict(dpt2c_case_terms),
        "dpt2c_case_terms_pre_scale": dict(dpt2c_case_terms_pre_scale),
        "olagns2_tc_case_terms": dict(tc_case_terms),
        "t2_mo": np.asarray(t2_mo, dtype=np.float64),
        "olagns2_case_subterms": dict(olag_case_subterms),
        "dpt2c_case_subterms_pre_scale": dict(dpt2c_case_subterms_pre_scale),
        "dpt2c_case_subterms": dict(dpt2c_case_subterms),
        "olagns2_scale": float(olagns2_scale),
        "olagns2_scale_mode": str(olagns2_scale_mode),
        "olagns2_nactel": int(olagns2_nactel),
        "dpt2_ao": dpt2_ao,
        "depsa": depsa,
        "depsa_used": np.asarray(depsa, dtype=np.float64),
        "clag_sigma_terms": dict(clag_sigma_terms),
        "clag_case_terms": dict(clag_case_terms),
        "clagdx_case_terms": dict(clagdx_case_terms),
        "clag_case_amp_terms": dict(clag_case_amp_terms),
        "clag_deasum_post_terms": dict(clag_deasum_post_terms),
        "clag_raw": np.asarray(clag_raw, dtype=np.float64),
        "clag_eigder_add": np.asarray(clag_eigder_add, dtype=np.float64),
        "clag_before_projection": np.asarray(clag_before_projection, dtype=np.float64),
        "clag_projection_parallel": np.asarray(clag_projection_parallel, dtype=np.float64),
        "olag_bare": np.asarray(olag_before_eigder, dtype=np.float64),
        "olag_full_before_antisym": np.asarray(olag_after_eigder, dtype=np.float64),
        "eigder_depsa_olag_add": np.asarray(olag_after_eigder - olag_before_eigder, dtype=np.float64),
        "eigder_terms_full_dpt2": {
            "olag_inact_fpt2": np.asarray(eig_t1, dtype=np.float64),
            "olag_act_fpt2": np.asarray(eig_t2, dtype=np.float64),
            "olag_fifa_dpt2": np.asarray(eig_t3, dtype=np.float64),
            "olag_fimo_dpt2c": np.asarray(eig_t4, dtype=np.float64),
            "olag_inact_fpt2c": np.asarray(eig_t5, dtype=np.float64),
            "olag_add_total": np.asarray(eig_total, dtype=np.float64),
        },
        "rdmeig": rdmeig,
        "fpt2_ao": fpt2_ao,
        "fpt2c_ao": fpt2c_ao,
        "stage_checkpoints": stage_checkpoints,
        "stage_checkpoint_order": list(stage_order),
    }


def build_pt2lag_from_ss_inputs(
    *,
    smap: Any,
    fock: Any,
    eri_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    ci_ctx: Any,
    amplitudes: list[np.ndarray],
    ci: np.ndarray,
    cmo: np.ndarray,
    s_ao: np.ndarray,
    drt: Any,
    verbose: int = 0,
    pt2lag_profile: str = "molcas_ss_strict",
    collect_breakdown: bool = True,
) -> dict[str, Any]:
    """Build Molcas-style PT2_Lag intermediates from an ASUKA SS-CASPT2 solution."""
    from tools.molcas_caspt2_grad.translation.types import (  # noqa: PLC0415
        OrbitalInfo,
        CASPT2Fock as TransFock,
        RDMs,
    )

    ncore = int(getattr(smap.orbs, "nish"))
    ncas = int(getattr(smap.orbs, "nash"))
    nvirt = int(getattr(smap.orbs, "nssh"))

    orb = OrbitalInfo(nfro=0, nish=ncore, nash=ncas, nssh=nvirt)
    trans_fock = TransFock(
        fimo=_asnumpy_f64(getattr(fock, "fimo")),
        famo=_asnumpy_f64(getattr(fock, "famo")),
        fifa=_asnumpy_f64(getattr(fock, "fifa")),
        epsa=_asnumpy_f64(getattr(fock, "epsa")),
    )
    rdms = RDMs(
        dm1=_asnumpy_f64(dm1),
        dm2=_asnumpy_f64(dm2).reshape(ncas, ncas, ncas, ncas),
        dm3=_asnumpy_f64(dm3),
    )
    ci = _asnumpy_f64(ci).ravel()
    ncsf = int(ci.size)

    case_amps = _build_case_amps_from_asuka(
        smap, fock, rdms.dm1, rdms.dm2, rdms.dm3, ci_ctx,
        amplitudes, orb, eri_mo,
        pt2_breakdown={
            "mo_coeff": np.asarray(cmo, dtype=np.float64),
            "S_ao": np.asarray(s_ao, dtype=np.float64),
        },
    )
    lag = _build_lagrangians(
        case_amps, smap, orb, trans_fock, rdms,
        ci, ncsf, eri_mo, _asnumpy_f64(cmo), _asnumpy_f64(s_ao), drt,
        verbose=int(verbose),
        collect_breakdown=bool(collect_breakdown),
    )
    return lag


def _molcas_pack_tril_from_raw(a_raw: np.ndarray) -> np.ndarray:
    """Pack a square matrix using Molcas packed-triangle convention.

    Molcas packed-triangle convention (for symmetric matrices):
      diag:    A_ii
      offdiag: A_ij + A_ji   (== 2*A_ij if symmetric)
    """
    a = np.asarray(a_raw, dtype=np.float64, order="C")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrix must be square, got {a.shape}")
    n = int(a.shape[0])
    tri = np.empty((n * (n + 1) // 2,), dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1):
            if i == j:
                tri[idx] = a[i, i]
            else:
                tri[idx] = a[i, j] + a[j, i]
            idx += 1
    return tri


def write_pt2lag(
    path: str | Path,
    *,
    clag: np.ndarray,
    olag_mo: np.ndarray,
    slag: np.ndarray,
    wlag_ao_raw: np.ndarray,
    dpt2_mo: np.ndarray,
    dpt2c_mo: np.ndarray,
    dpt2_ao_raw: np.ndarray,
    dpt2c_ao_raw: np.ndarray,
) -> None:
    """Write a Molcas-compatible PT2_Lag text file.

    The layout matches OpenMolcas `GrdCls` (one value per line, free-format):
      1) CLagFull  (nConf x nState)  : loop state then conf
      2) OLagFull  (nBas x nBas)     : Fortran order flatten
      3) SLag      (nState x nState) : Fortran order flatten
      4) WLag      (nBas*(nBas+1)/2) : packed lower-tri (Molcas convention)
      5) DPT2_tot  (nBas x nBas)     : Fortran order flatten
      6) DPT2C_tot (nBas x nBas)     : Fortran order flatten
      7) DPT2_AO_tot  : NBSQT lines, first NBTRI packed, rest zeros
      8) DPT2C_AO_tot : NBSQT lines, first NBTRI packed, rest zeros
    """
    p = Path(path)

    clag = np.asarray(clag, dtype=np.float64)
    if clag.ndim == 1:
        clag = clag.reshape(-1, 1)
    if clag.ndim != 2:
        raise ValueError("clag must be 1D or 2D")
    nconf, nstate = map(int, clag.shape)

    olag_mo = np.asarray(olag_mo, dtype=np.float64)
    if olag_mo.ndim != 2 or olag_mo.shape[0] != olag_mo.shape[1]:
        raise ValueError("olag_mo must be a square 2D matrix")
    nbas = int(olag_mo.shape[0])

    slag = np.asarray(slag, dtype=np.float64)
    if slag.shape != (nstate, nstate):
        raise ValueError(f"slag shape {slag.shape} != ({nstate}, {nstate})")

    dpt2_mo = np.asarray(dpt2_mo, dtype=np.float64)
    dpt2c_mo = np.asarray(dpt2c_mo, dtype=np.float64)
    if dpt2_mo.shape != (nbas, nbas):
        raise ValueError("dpt2_mo shape mismatch")
    if dpt2c_mo.shape != (nbas, nbas):
        raise ValueError("dpt2c_mo shape mismatch")

    wlag_tri = _molcas_pack_tril_from_raw(np.asarray(wlag_ao_raw, dtype=np.float64))
    dpt2_ao_tri = _molcas_pack_tril_from_raw(np.asarray(dpt2_ao_raw, dtype=np.float64))
    dpt2c_ao_tri = _molcas_pack_tril_from_raw(np.asarray(dpt2c_ao_raw, dtype=np.float64))
    ntri = int(wlag_tri.size)
    nbsq = int(nbas * nbas)

    with p.open("w", encoding="utf-8") as fh:
        # 1) CLagFull: state-major, then conf
        for j in range(nstate):
            for i in range(nconf):
                fh.write(f"{float(clag[i, j]):.16e}\n")

        # 2) OLagFull: Fortran-order flatten
        for v in olag_mo.reshape(nbsq, order="F"):
            fh.write(f"{float(v):.16e}\n")

        # 3) SLag: Fortran-order flatten
        for v in slag.reshape(nstate * nstate, order="F"):
            fh.write(f"{float(v):.16e}\n")

        # 4) WLag: packed lower triangle
        for v in wlag_tri:
            fh.write(f"{float(v):.16e}\n")

        # 5) DPT2_tot (MO): Fortran-order flatten
        for v in dpt2_mo.reshape(nbsq, order="F"):
            fh.write(f"{float(v):.16e}\n")

        # 6) DPT2C_tot (MO)
        for v in dpt2c_mo.reshape(nbsq, order="F"):
            fh.write(f"{float(v):.16e}\n")

        # 7) DPT2_AO_tot: NBSQT lines, first NBTRI packed
        for v in dpt2_ao_tri:
            fh.write(f"{float(v):.16e}\n")
        for _ in range(nbsq - ntri):
            fh.write("0.0000000000000000e+00\n")

        # 8) DPT2C_AO_tot
        for v in dpt2c_ao_tri:
            fh.write(f"{float(v):.16e}\n")
        for _ in range(nbsq - ntri):
            fh.write("0.0000000000000000e+00\n")
