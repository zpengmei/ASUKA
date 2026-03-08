"""Gradient assembly from Lagrangians and Z-vector for CASPT2."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from asuka.caspt2.gradient.debug_utils import _asnumpy_f64


def _build_pt2_bar_L(B_ao_x, D_pt2_ao, D_core_ref_ao, D_act_ref_ao, xp, *,
                     hf2e_ref_mode="core_only", hf2e_include_self=True, hf2e_build_mode="jk"):
    """Build bar_L from PT2 density cross-terms with reference density.

    Implements the mean-field two-electron contribution to the CASPT2 nuclear gradient.
    For the unrelaxed PT2 2-body density, the bar_L tensor is built from J/K cross-terms
    between D_PT2 and D_ref, plus self-interaction terms.

    Returns dict with 'bar_L_pt2' and diagnostic sub-components.
    """
    nao = int(B_ao_x.shape[0])
    naux = int(B_ao_x.shape[2])
    B2 = B_ao_x.reshape(nao * nao, naux)

    # DF projections for PT2 and reference blocks.
    rho_core = B2.T @ D_core_ref_ao.reshape(-1)
    rho_act = B2.T @ D_act_ref_ao.reshape(-1)
    rho_pt2 = B2.T @ D_pt2_ao.reshape(-1)
    BQ = xp.transpose(B_ao_x, (2, 0, 1))  # (naux,nao,nao)

    # Build explicit HF/PT2 bar_L subterms for per-channel diagnostics.
    bar_hf_cross_j_core = (
        rho_pt2[:, None, None] * D_core_ref_ao[None, :, :]
        + rho_core[:, None, None] * D_pt2_ao[None, :, :]
    )
    bar_hf_cross_j_act = (
        rho_pt2[:, None, None] * D_act_ref_ao[None, :, :]
        + rho_act[:, None, None] * D_pt2_ao[None, :, :]
    )
    bar_hf_cross_k_core = xp.zeros((naux, nao, nao), dtype=xp.float64)
    bar_hf_cross_k_act = xp.zeros((naux, nao, nao), dtype=xp.float64)
    bar_hf_self_j = rho_pt2[:, None, None] * D_pt2_ao[None, :, :]
    bar_hf_self_k = xp.zeros((naux, nao, nao), dtype=xp.float64)

    # Chunk over aux index for memory.
    _chunk = max(1, naux // 4)
    for q0 in range(0, naux, _chunk):
        q1 = min(q0 + _chunk, naux)
        bq = BQ[q0:q1]

        t_core = xp.matmul(xp.matmul(D_core_ref_ao[None, :, :], bq), D_pt2_ao)
        t_core += xp.matmul(xp.matmul(D_pt2_ao[None, :, :], bq), D_core_ref_ao)
        t_core *= -0.5
        bar_hf_cross_k_core[q0:q1] += t_core

        t_act = xp.matmul(xp.matmul(D_act_ref_ao[None, :, :], bq), D_pt2_ao)
        t_act += xp.matmul(xp.matmul(D_pt2_ao[None, :, :], bq), D_act_ref_ao)
        t_act *= -0.5
        bar_hf_cross_k_act[q0:q1] += t_act

        t_self = xp.matmul(xp.matmul(D_pt2_ao[None, :, :], bq), D_pt2_ao)
        t_self *= -0.5
        bar_hf_self_k[q0:q1] += t_self

        del t_core, t_act, t_self

    def _symm_bar(bar_qmn: Any) -> Any:
        return 0.5 * (bar_qmn + xp.transpose(bar_qmn, (0, 2, 1)))

    bar_hf_cross_j_core = _symm_bar(bar_hf_cross_j_core)
    bar_hf_cross_j_act = _symm_bar(bar_hf_cross_j_act)
    bar_hf_cross_k_core = _symm_bar(bar_hf_cross_k_core)
    bar_hf_cross_k_act = _symm_bar(bar_hf_cross_k_act)
    bar_hf_self_j = _symm_bar(bar_hf_self_j)
    bar_hf_self_k = _symm_bar(bar_hf_self_k)

    bar_hf_cross_core = bar_hf_cross_j_core + bar_hf_cross_k_core
    bar_hf_cross_act = bar_hf_cross_j_act + bar_hf_cross_k_act
    bar_hf_cross_core_j = bar_hf_cross_j_core
    bar_hf_cross_act_j = bar_hf_cross_j_act
    if hf2e_build_mode == "j_only":
        if hf2e_ref_mode == "core_only":
            bar_hf_cross_selected = bar_hf_cross_core_j
        elif hf2e_ref_mode == "core_minus_active":
            bar_hf_cross_selected = bar_hf_cross_core_j - bar_hf_cross_act_j
        else:
            bar_hf_cross_selected = bar_hf_cross_core_j + bar_hf_cross_act_j
        bar_hf_self_selected = xp.zeros_like(bar_hf_self_j)
    elif hf2e_build_mode == "j_only_selfj":
        if hf2e_ref_mode == "core_only":
            bar_hf_cross_selected = bar_hf_cross_core_j
        elif hf2e_ref_mode == "core_minus_active":
            bar_hf_cross_selected = bar_hf_cross_core_j - bar_hf_cross_act_j
        else:
            bar_hf_cross_selected = bar_hf_cross_core_j + bar_hf_cross_act_j
        bar_hf_self_selected = bar_hf_self_j if hf2e_include_self else xp.zeros_like(bar_hf_self_j)
    else:
        if hf2e_ref_mode == "core_only":
            bar_hf_cross_selected = bar_hf_cross_core
        elif hf2e_ref_mode == "core_minus_active":
            bar_hf_cross_selected = bar_hf_cross_core - bar_hf_cross_act
        else:
            bar_hf_cross_selected = bar_hf_cross_core + bar_hf_cross_act
        bar_hf_self_selected = bar_hf_self_j + bar_hf_self_k if hf2e_include_self else xp.zeros_like(bar_hf_self_j)
    bar_L_pt2 = _symm_bar(bar_hf_cross_selected + bar_hf_self_selected)

    return {
        "bar_L_pt2": bar_L_pt2,
        "bar_hf_cross_j_core": bar_hf_cross_j_core,
        "bar_hf_cross_j_act": bar_hf_cross_j_act,
        "bar_hf_cross_k_core": bar_hf_cross_k_core,
        "bar_hf_cross_k_act": bar_hf_cross_k_act,
        "bar_hf_self_j": bar_hf_self_j,
        "bar_hf_self_k": bar_hf_self_k,
    }


def _build_ci_response(
    z_ci, ci_list, weights, nroots,
    ncas, nelecas, twos, ncore, nmo,
    B_ao_x, C_x, xp,
    *,
    zci_mode="full",
    ci_trans_rdm_mode="solver",
    collect_breakdown=False,
):
    """Build CI Z-vector response contributions to bar_L, D, and diagnostics.

    The CI Z-vector modifies the active-space density through transition RDMs:
      dm1_lci = sum_K w_K * (<z_K|E_{pq}|c_K> + <c_K|E_{pq}|z_K>)
      dm2_lci = sum_K w_K * (<z_K|e_{pqrs}|c_K> + <c_K|e_{pqrs}|z_K>)
    These are then contracted with DF integrals to yield bar_L_ci and D_ci.

    Returns dict with production outputs and diagnostic arrays.
    """
    from asuka.caspt2.gradient.debug_utils import (  # noqa: PLC0415
        _apply_ci_basis_map,
        _infer_ci_basis_map_from_dump,
        _infer_ci_basis_map_from_resp,
        _parse_ci_basis_map_from_env,
    )
    from asuka.mcscf.nac._df import _build_bar_L_net_active_df  # noqa: PLC0415
    from asuka.solver import GUGAFCISolver  # noqa: PLC0415

    fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))

    # Initialize diagnostic arrays
    z_ci_overlap = np.zeros((nroots,), dtype=np.float64)
    z_ci_alpha = np.zeros((nroots,), dtype=np.float64)
    z_ci_norm = np.zeros((nroots,), dtype=np.float64)
    z_ci_parallel_norm = np.zeros((nroots,), dtype=np.float64)
    z_ci_tangent_norm = np.zeros((nroots,), dtype=np.float64)
    z_ci_overlap_raw = np.zeros((nroots,), dtype=np.float64)
    z_ci_alpha_raw = np.zeros((nroots,), dtype=np.float64)
    z_ci_norm_raw = np.zeros((nroots,), dtype=np.float64)
    z_ci_parallel_norm_raw = np.zeros((nroots,), dtype=np.float64)
    z_ci_tangent_norm_raw = np.zeros((nroots,), dtype=np.float64)
    z_ci_map_delta_norm = np.zeros((nroots,), dtype=np.float64)
    z_ci_map_delta_max_abs = np.zeros((nroots,), dtype=np.float64)
    ci_ref_map_delta_norm = np.zeros((nroots,), dtype=np.float64)
    ci_ref_map_delta_max_abs = np.zeros((nroots,), dtype=np.float64)
    ci_dm1_asym_ratio = np.zeros((nroots,), dtype=np.float64)
    ci_dm2_pair_asym_ratio = np.zeros((nroots,), dtype=np.float64)
    _ci_z_map_spec: dict[str, Any] | None = None
    _ci_z_map_mode = "none"
    _ci_z_map_perm = np.zeros((0,), dtype=np.int64)
    _ci_z_map_signs = np.zeros((0,), dtype=np.float64)
    _ci_z_map_applied = False

    def _select_ci_mode(z_full: np.ndarray, c_ref: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
        denom = float(np.dot(c_ref, c_ref))
        alpha = float(np.dot(c_ref, z_full) / max(1.0e-30, denom))
        z_par = alpha * c_ref
        z_tan = z_full - z_par
        if mode == "parallel":
            return z_par, alpha
        if mode == "tangent":
            return z_tan, alpha
        return z_full, alpha

    _ci_tdm_mode = str(ci_trans_rdm_mode).strip().lower()
    if _ci_tdm_mode not in {"solver", "molcas"}:
        _ci_tdm_mode = "solver"
    _trans_drt = None
    _trans_rdm_molcas = None
    if _ci_tdm_mode == "molcas":
        from asuka.cuguga.drt import build_drt as _build_drt  # noqa: PLC0415
        from asuka.rdm.rdm123 import _trans_rdm123_pyscf as _make_trans_rdm123  # noqa: PLC0415

        nelec_total = int(nelecas) if isinstance(nelecas, (int, np.integer)) else int(sum(nelecas))
        _trans_drt = _build_drt(norb=int(ncas), nelec=nelec_total, twos_target=int(twos))
        _trans_rdm_molcas = _make_trans_rdm123

    def _trans_dm12(ci_bra_vec: np.ndarray, ci_ket_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bra = np.asarray(ci_bra_vec, dtype=np.float64).ravel()
        ket = np.asarray(ci_ket_vec, dtype=np.float64).ravel()
        if _ci_tdm_mode == "molcas" and _trans_drt is not None and _trans_rdm_molcas is not None:
            dm1_raw, dm2_raw, _dm3 = _trans_rdm_molcas(
                _trans_drt,
                bra,
                ket,
                reorder=True,
                reorder_mode="molcas",
            )
            dm1_use = np.asarray(dm1_raw, dtype=np.float64).T
            dm2_use = np.asarray(dm2_raw, dtype=np.float64)
            return dm1_use, dm2_use
        dm1_use, dm2_use = fcisolver.trans_rdm12(
            bra,
            ket,
            int(ncas), nelecas,
            return_cupy=False,
        )
        return np.asarray(dm1_use, dtype=np.float64), np.asarray(dm2_use, dtype=np.float64)

    # CI basis map detection
    z_ci_list = z_ci if isinstance(z_ci, list) else [z_ci]
    if len(z_ci_list) > 0:
        nocc_full = int(ncore + ncas)
        nvirt_full = int(nmo - nocc_full)
        n_orb_packed = int(ncore * ncas + nvirt_full * nocc_full)
        _ci_z_map_spec = _parse_ci_basis_map_from_env(
            int(np.asarray(z_ci_list[0], dtype=np.float64).size)
        )
        if _ci_z_map_spec is None:
            _ci_z_map_spec = _infer_ci_basis_map_from_resp(
                n_ci=int(np.asarray(z_ci_list[0], dtype=np.float64).size),
                n_orb_packed=int(n_orb_packed),
                z_ci_asuka=np.asarray(z_ci_list[0], dtype=np.float64).ravel(),
            )
        if _ci_z_map_spec is None:
            _ci_z_map_spec = _infer_ci_basis_map_from_dump(
                n_ci=int(np.asarray(z_ci_list[0], dtype=np.float64).size),
                ci_ref_asuka=np.asarray(ci_list[0], dtype=np.float64).ravel(),
            )
        if _ci_z_map_spec is not None:
            _ci_z_map_mode = str(_ci_z_map_spec.get("mode", "direct"))
            _ci_z_map_perm = np.asarray(_ci_z_map_spec.get("perm"), dtype=np.int64).ravel()
            _ci_z_map_signs = np.asarray(_ci_z_map_spec.get("signs"), dtype=np.float64).ravel()
            _ci_z_map_applied = bool(
                (_ci_z_map_perm.size > 0)
                and (
                    not np.array_equal(
                        _ci_z_map_perm,
                        np.arange(_ci_z_map_perm.size, dtype=np.int64),
                    )
                    or bool(np.any(_ci_z_map_signs < 0.0))
                )
            )

    dm1_lci = np.zeros((ncas, ncas), dtype=np.float64)
    dm2_lci = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)
    dm1_lci_par = np.zeros_like(dm1_lci) if bool(collect_breakdown) else None
    dm2_lci_par = np.zeros_like(dm2_lci) if bool(collect_breakdown) else None
    dm1_lci_tan = np.zeros_like(dm1_lci) if bool(collect_breakdown) else None
    dm2_lci_tan = np.zeros_like(dm2_lci) if bool(collect_breakdown) else None
    _ci_dm1_sym_mode = "sum"
    _ci_dm2_sym_mode = "sum"

    def _sym_dm1(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        return xx + xx.T

    def _sym_dm2(x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64)
        xxt = xx.transpose(1, 0, 3, 2)
        return xx + xxt

    for r in range(nroots):
        if r >= len(z_ci_list):
            continue
        wr = float(np.asarray(weights, dtype=np.float64).ravel()[r])
        if abs(wr) < 1e-14:
            continue
        z_full_raw = np.asarray(z_ci_list[r], dtype=np.float64).ravel()
        if _ci_z_map_spec is not None:
            if z_full_raw.size != int(_ci_z_map_perm.size):
                raise ValueError(
                    "CI map/vector size mismatch across roots: "
                    f"root={r} size={z_full_raw.size} map={int(_ci_z_map_perm.size)}"
                )
            z_full = _apply_ci_basis_map(z_full_raw, _ci_z_map_spec)
        else:
            z_full = z_full_raw
        c_ref_raw = np.asarray(ci_list[r], dtype=np.float64).ravel()
        if _ci_z_map_spec is not None:
            c_ref = _apply_ci_basis_map(c_ref_raw, _ci_z_map_spec)
        else:
            c_ref = c_ref_raw
        denom = float(np.dot(c_ref, c_ref))
        denom_raw = float(np.dot(c_ref_raw, c_ref_raw))
        alpha_raw = float(np.dot(c_ref_raw, z_full_raw) / max(1.0e-30, denom_raw))
        z_par_raw = alpha_raw * c_ref_raw
        z_tan_raw = z_full_raw - z_par_raw
        alpha = float(np.dot(c_ref, z_full) / max(1.0e-30, denom))
        z_par = alpha * c_ref
        z_tan = z_full - z_par
        z_ci_overlap_raw[r] = float(np.dot(z_full_raw, c_ref_raw))
        z_ci_alpha_raw[r] = alpha_raw
        z_ci_norm_raw[r] = float(np.linalg.norm(z_full_raw))
        z_ci_parallel_norm_raw[r] = float(np.linalg.norm(z_par_raw))
        z_ci_tangent_norm_raw[r] = float(np.linalg.norm(z_tan_raw))
        z_ci_overlap[r] = float(np.dot(z_full, c_ref))
        z_ci_alpha[r] = alpha
        z_ci_norm[r] = float(np.linalg.norm(z_full))
        z_ci_parallel_norm[r] = float(np.linalg.norm(z_par))
        z_ci_tangent_norm[r] = float(np.linalg.norm(z_tan))
        dz_map = np.asarray(z_full - z_full_raw, dtype=np.float64)
        z_ci_map_delta_norm[r] = float(np.linalg.norm(dz_map))
        z_ci_map_delta_max_abs[r] = float(np.max(np.abs(dz_map)))
        dc_map = np.asarray(c_ref - c_ref_raw, dtype=np.float64)
        ci_ref_map_delta_norm[r] = float(np.linalg.norm(dc_map))
        ci_ref_map_delta_max_abs[r] = float(np.max(np.abs(dc_map)))
        z_resp, _alpha_resp = _select_ci_mode(z_full, c_ref, zci_mode)
        dm1_r, dm2_r = _trans_dm12(
            np.asarray(z_resp, dtype=np.float64).ravel(),
            c_ref,
        )
        _dm1_r_np = np.asarray(dm1_r, dtype=np.float64)
        _dm2_r_np = np.asarray(dm2_r, dtype=np.float64)
        _dm1_n = float(np.linalg.norm(_dm1_r_np))
        ci_dm1_asym_ratio[r] = float(np.linalg.norm(_dm1_r_np - _dm1_r_np.T) / max(1.0e-30, _dm1_n))
        _dm2_n = float(np.linalg.norm(_dm2_r_np))
        _dm2_t = _dm2_r_np.transpose(1, 0, 3, 2)
        ci_dm2_pair_asym_ratio[r] = float(np.linalg.norm(_dm2_r_np - _dm2_t) / max(1.0e-30, _dm2_n))
        dm1_lci += wr * _sym_dm1(dm1_r)
        dm2_lci += wr * _sym_dm2(dm2_r)
        if bool(collect_breakdown) and dm1_lci_par is not None and dm2_lci_par is not None and dm1_lci_tan is not None and dm2_lci_tan is not None:
            dm1_par_r, dm2_par_r = _trans_dm12(z_par, c_ref)
            dm1_tan_r, dm2_tan_r = _trans_dm12(z_tan, c_ref)
            dm1_lci_par += wr * _sym_dm1(dm1_par_r)
            dm2_lci_par += wr * _sym_dm2(dm2_par_r)
            dm1_lci_tan += wr * _sym_dm1(dm1_tan_r)
            dm2_lci_tan += wr * _sym_dm2(dm2_tan_r)

    bar_L_lci, D_act_lci = _build_bar_L_net_active_df(
        B_ao_x, C_x,
        dm1_lci, dm2_lci,
        ncore=int(ncore), ncas=int(ncas),
        xp=xp,
    )
    bar_L_ci = xp.asarray(bar_L_lci, dtype=xp.float64)
    D_ci_1e = _asnumpy_f64(D_act_lci)

    # Optional breakdown sub-components
    bar_L_ci_dm1 = None
    bar_L_ci_dm2 = None
    D_ci_1e_dm1 = None
    D_ci_1e_dm2 = None
    bar_L_ci_par = None
    bar_L_ci_tan = None
    D_ci_1e_par = None
    D_ci_1e_tan = None
    if bool(collect_breakdown):
        bar_L_lci_dm1, _D_act_lci_dm1 = _build_bar_L_net_active_df(
            B_ao_x, C_x,
            dm1_lci,
            np.zeros_like(dm2_lci),
            ncore=int(ncore), ncas=int(ncas),
            xp=xp,
        )
        bar_L_lci_dm2, _D_act_lci_dm2 = _build_bar_L_net_active_df(
            B_ao_x, C_x,
            np.zeros_like(dm1_lci),
            dm2_lci,
            ncore=int(ncore), ncas=int(ncas),
            xp=xp,
        )
        bar_L_ci_dm1 = xp.asarray(bar_L_lci_dm1, dtype=xp.float64)
        bar_L_ci_dm2 = xp.asarray(bar_L_lci_dm2, dtype=xp.float64)
        D_ci_1e_dm1 = _asnumpy_f64(_D_act_lci_dm1)
        D_ci_1e_dm2 = _asnumpy_f64(_D_act_lci_dm2)
        if dm1_lci_par is not None and dm2_lci_par is not None:
            bar_lci_par, d_act_lci_par = _build_bar_L_net_active_df(
                B_ao_x, C_x,
                dm1_lci_par,
                dm2_lci_par,
                ncore=int(ncore), ncas=int(ncas),
                xp=xp,
            )
            bar_L_ci_par = xp.asarray(bar_lci_par, dtype=xp.float64)
            D_ci_1e_par = _asnumpy_f64(d_act_lci_par)
        if dm1_lci_tan is not None and dm2_lci_tan is not None:
            bar_lci_tan, d_act_lci_tan = _build_bar_L_net_active_df(
                B_ao_x, C_x,
                dm1_lci_tan,
                dm2_lci_tan,
                ncore=int(ncore), ncas=int(ncas),
                xp=xp,
            )
            bar_L_ci_tan = xp.asarray(bar_lci_tan, dtype=xp.float64)
            D_ci_1e_tan = _asnumpy_f64(d_act_lci_tan)

    return {
        # Production outputs
        "bar_L_ci": bar_L_ci,
        "D_ci_1e": D_ci_1e,
        "dm1_lci": dm1_lci,
        "dm2_lci": dm2_lci,
        # Breakdown sub-components
        "dm1_lci_par": dm1_lci_par,
        "dm2_lci_par": dm2_lci_par,
        "dm1_lci_tan": dm1_lci_tan,
        "dm2_lci_tan": dm2_lci_tan,
        "bar_L_ci_dm1": bar_L_ci_dm1,
        "bar_L_ci_dm2": bar_L_ci_dm2,
        "D_ci_1e_dm1": D_ci_1e_dm1,
        "D_ci_1e_dm2": D_ci_1e_dm2,
        "bar_L_ci_par": bar_L_ci_par,
        "bar_L_ci_tan": bar_L_ci_tan,
        "D_ci_1e_par": D_ci_1e_par,
        "D_ci_1e_tan": D_ci_1e_tan,
        # Diagnostics
        "z_ci_overlap": z_ci_overlap,
        "z_ci_alpha": z_ci_alpha,
        "z_ci_norm": z_ci_norm,
        "z_ci_parallel_norm": z_ci_parallel_norm,
        "z_ci_tangent_norm": z_ci_tangent_norm,
        "z_ci_overlap_raw": z_ci_overlap_raw,
        "z_ci_alpha_raw": z_ci_alpha_raw,
        "z_ci_norm_raw": z_ci_norm_raw,
        "z_ci_parallel_norm_raw": z_ci_parallel_norm_raw,
        "z_ci_tangent_norm_raw": z_ci_tangent_norm_raw,
        "z_ci_map_delta_norm": z_ci_map_delta_norm,
        "z_ci_map_delta_max_abs": z_ci_map_delta_max_abs,
        "ci_ref_map_delta_norm": ci_ref_map_delta_norm,
        "ci_ref_map_delta_max_abs": ci_ref_map_delta_max_abs,
        "ci_dm1_asym_ratio": ci_dm1_asym_ratio,
        "ci_dm2_pair_asym_ratio": ci_dm2_pair_asym_ratio,
        "ci_z_map_mode": _ci_z_map_mode,
        "ci_z_map_applied": _ci_z_map_applied,
        "ci_z_map_perm": _ci_z_map_perm,
        "ci_z_map_signs": _ci_z_map_signs,
        "ci_dm1_sym_mode": _ci_dm1_sym_mode,
        "ci_dm2_sym_mode": _ci_dm2_sym_mode,
        "ci_tdm_mode": _ci_tdm_mode,
    }


def _collect_gradient_breakdown(
    *,
    contract_df,
    contract_h1,
    contract_pulay,
    natm,
    intermediates,
):
    """Collect per-channel gradient breakdown for diagnostics.

    Re-contracts each bar_L/D/W component separately to produce a detailed
    breakdown of the gradient into reference, PT2, CI-response, and orbital-response
    channels. Only called when collect_breakdown=True.

    Returns the breakdown dict.
    """
    # Unpack intermediates
    bar_L_ref = intermediates["bar_L_ref"]
    bar_L_hf = intermediates["bar_L_hf"]
    bar_L_ci = intermediates["bar_L_ci"]
    bar_L_orb = intermediates["bar_L_orb"]
    bar_hf_cross_j_core = intermediates["bar_hf_cross_j_core"]
    bar_hf_cross_j_act = intermediates["bar_hf_cross_j_act"]
    bar_hf_cross_k_core = intermediates["bar_hf_cross_k_core"]
    bar_hf_cross_k_act = intermediates["bar_hf_cross_k_act"]
    bar_hf_self_j = intermediates["bar_hf_self_j"]
    bar_hf_self_k = intermediates["bar_hf_self_k"]
    bar_L_ci_dm1 = intermediates["bar_L_ci_dm1"]
    bar_L_ci_dm2 = intermediates["bar_L_ci_dm2"]
    bar_L_ci_par = intermediates["bar_L_ci_par"]
    bar_L_ci_tan = intermediates["bar_L_ci_tan"]
    bar_L_orb_mean = intermediates["bar_L_orb_mean"]
    bar_L_orb_dm2 = intermediates["bar_L_orb_dm2"]
    D_ref_1e = intermediates["D_ref_1e"]
    D_hf_1e = intermediates["D_hf_1e"]
    D_hf_1e_selected = intermediates["D_hf_1e_selected"]
    D_hf_1e_dpt2c_sym_quarter = intermediates["D_hf_1e_dpt2c_sym_quarter"]
    D_hf_1e_molcas_dk = intermediates["D_hf_1e_molcas_dk"]
    D_ci_1e = intermediates["D_ci_1e"]
    D_ci_1e_dm1 = intermediates["D_ci_1e_dm1"]
    D_ci_1e_dm2 = intermediates["D_ci_1e_dm2"]
    D_ci_1e_par = intermediates["D_ci_1e_par"]
    D_ci_1e_tan = intermediates["D_ci_1e_tan"]
    D_orb_1e = intermediates["D_orb_1e"]
    W_ref = intermediates["W_ref"]
    W_hf = intermediates["W_hf"]
    W_ci = intermediates["W_ci"]
    W_orb = intermediates["W_orb"]
    W_orb_parts = intermediates["W_orb_parts"]
    W_total = intermediates["W_total"]
    _hf_2e_scale = intermediates["hf_2e_scale"]
    _hf_1e_scale = intermediates["hf_1e_scale"]
    _hf_pulay_scale = intermediates["hf_pulay_scale"]
    _ci_2e_scale = intermediates["ci_2e_scale"]
    _ci_1e_scale = intermediates["ci_1e_scale"]
    _ci_pulay_scale_eff = intermediates["ci_pulay_scale_eff"]
    _orb_2e_scale = intermediates["orb_2e_scale"]
    _orb_1e_scale = intermediates["orb_1e_scale"]
    _orb_pulay_scale = intermediates["orb_pulay_scale"]
    _hf2e_ref_mode = intermediates["hf2e_ref_mode"]
    _hf2e_build_mode = intermediates["hf2e_build_mode"]
    _hf2e_include_self = intermediates["hf2e_include_self"]
    _hf1e_mode = intermediates["hf1e_mode"]
    _zci_mode = intermediates["zci_mode"]
    _ci_tdm_mode = intermediates["ci_tdm_mode"]
    _ci_dm1_sym_mode = intermediates["ci_dm1_sym_mode"]
    _ci_dm2_sym_mode = intermediates["ci_dm2_sym_mode"]
    _lorb_wlag_mode = intermediates["lorb_wlag_mode"]
    _include_lorb_wlag = intermediates["include_lorb_wlag"]
    de_btamp_dense_raw = intermediates["de_btamp_dense_raw"]
    de_btamp_dense_scaled = intermediates["de_btamp_dense_scaled"]
    de_nuc = intermediates["de_nuc"]
    z_ci = intermediates["z_ci"]
    orb_case_payload = intermediates["orb_case_payload"]
    dm1_sa = intermediates["dm1_sa"]
    dm2_sa = intermediates["dm2_sa"]
    dm1_lci = intermediates["dm1_lci"]
    dm2_lci = intermediates["dm2_lci"]
    dm1_lci_par = intermediates["dm1_lci_par"]
    dm2_lci_par = intermediates["dm2_lci_par"]
    dm1_lci_tan = intermediates["dm1_lci_tan"]
    dm2_lci_tan = intermediates["dm2_lci_tan"]
    ncas = intermediates["ncas"]
    _asnp = intermediates["_asnumpy_f64"]
    # CI response diagnostics
    ci_resp = intermediates.get("ci_response_result", {})
    z_ci_overlap = ci_resp.get("z_ci_overlap", np.zeros((0,), dtype=np.float64))
    z_ci_alpha = ci_resp.get("z_ci_alpha", np.zeros((0,), dtype=np.float64))
    z_ci_norm = ci_resp.get("z_ci_norm", np.zeros((0,), dtype=np.float64))
    z_ci_parallel_norm = ci_resp.get("z_ci_parallel_norm", np.zeros((0,), dtype=np.float64))
    z_ci_tangent_norm = ci_resp.get("z_ci_tangent_norm", np.zeros((0,), dtype=np.float64))
    z_ci_overlap_raw = ci_resp.get("z_ci_overlap_raw", np.zeros((0,), dtype=np.float64))
    z_ci_alpha_raw = ci_resp.get("z_ci_alpha_raw", np.zeros((0,), dtype=np.float64))
    z_ci_norm_raw = ci_resp.get("z_ci_norm_raw", np.zeros((0,), dtype=np.float64))
    z_ci_parallel_norm_raw = ci_resp.get("z_ci_parallel_norm_raw", np.zeros((0,), dtype=np.float64))
    z_ci_tangent_norm_raw = ci_resp.get("z_ci_tangent_norm_raw", np.zeros((0,), dtype=np.float64))
    z_ci_map_delta_norm = ci_resp.get("z_ci_map_delta_norm", np.zeros((0,), dtype=np.float64))
    z_ci_map_delta_max_abs = ci_resp.get("z_ci_map_delta_max_abs", np.zeros((0,), dtype=np.float64))
    ci_ref_map_delta_norm = ci_resp.get("ci_ref_map_delta_norm", np.zeros((0,), dtype=np.float64))
    ci_ref_map_delta_max_abs = ci_resp.get("ci_ref_map_delta_max_abs", np.zeros((0,), dtype=np.float64))
    ci_dm1_asym_ratio = ci_resp.get("ci_dm1_asym_ratio", np.zeros((0,), dtype=np.float64))
    ci_dm2_pair_asym_ratio = ci_resp.get("ci_dm2_pair_asym_ratio", np.zeros((0,), dtype=np.float64))
    _ci_z_map_mode = ci_resp.get("ci_z_map_mode", "none")
    _ci_z_map_applied = ci_resp.get("ci_z_map_applied", False)
    _ci_z_map_perm = ci_resp.get("ci_z_map_perm", np.zeros((0,), dtype=np.int64))
    _ci_z_map_signs = ci_resp.get("ci_z_map_signs", np.zeros((0,), dtype=np.float64))

    z3 = np.zeros((natm, 3), dtype=np.float64)

    de_ref_2e = contract_df(bar_L_ref)
    de_hf_2e_base = float(_hf_2e_scale) * contract_df(bar_L_hf)
    de_hf_2e = np.asarray(de_hf_2e_base + de_btamp_dense_scaled, dtype=np.float64)
    de_hf_2e_cross_j_core = float(_hf_2e_scale) * contract_df(bar_hf_cross_j_core)
    de_hf_2e_cross_j_act = float(_hf_2e_scale) * contract_df(bar_hf_cross_j_act)
    de_hf_2e_cross_k_core = float(_hf_2e_scale) * contract_df(bar_hf_cross_k_core)
    de_hf_2e_cross_k_act = float(_hf_2e_scale) * contract_df(bar_hf_cross_k_act)
    de_hf_2e_self_j = float(_hf_2e_scale) * contract_df(bar_hf_self_j)
    de_hf_2e_self_k = float(_hf_2e_scale) * contract_df(bar_hf_self_k)

    de_hf_2e_cross_core = np.asarray(de_hf_2e_cross_j_core + de_hf_2e_cross_k_core, dtype=np.float64)
    de_hf_2e_cross_act = np.asarray(de_hf_2e_cross_j_act + de_hf_2e_cross_k_act, dtype=np.float64)
    de_hf_2e_cross_core_j = np.asarray(de_hf_2e_cross_j_core, dtype=np.float64)
    de_hf_2e_cross_act_j = np.asarray(de_hf_2e_cross_j_act, dtype=np.float64)
    if _hf2e_build_mode == "j_only":
        if _hf2e_ref_mode == "core_only":
            de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core_j, dtype=np.float64)
        elif _hf2e_ref_mode == "core_minus_active":
            de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core_j - de_hf_2e_cross_act_j, dtype=np.float64)
        else:
            de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core_j + de_hf_2e_cross_act_j, dtype=np.float64)
        de_hf_2e_self_selected = z3.copy()
    elif _hf2e_build_mode == "j_only_selfj":
        if _hf2e_ref_mode == "core_only":
            de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core_j, dtype=np.float64)
        elif _hf2e_ref_mode == "core_minus_active":
            de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core_j - de_hf_2e_cross_act_j, dtype=np.float64)
        else:
            de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core_j + de_hf_2e_cross_act_j, dtype=np.float64)
        de_hf_2e_self_selected = np.asarray(de_hf_2e_self_j, dtype=np.float64) if bool(_hf2e_include_self) else z3.copy()
    elif _hf2e_ref_mode == "core_only":
        de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core, dtype=np.float64)
        de_hf_2e_self_selected = (
            np.asarray(de_hf_2e_self_j + de_hf_2e_self_k, dtype=np.float64)
            if bool(_hf2e_include_self) else z3.copy()
        )
    elif _hf2e_ref_mode == "core_minus_active":
        de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core - de_hf_2e_cross_act, dtype=np.float64)
        de_hf_2e_self_selected = (
            np.asarray(de_hf_2e_self_j + de_hf_2e_self_k, dtype=np.float64)
            if bool(_hf2e_include_self) else z3.copy()
        )
    else:
        de_hf_2e_cross_selected = np.asarray(de_hf_2e_cross_core + de_hf_2e_cross_act, dtype=np.float64)
        de_hf_2e_self_selected = (
            np.asarray(de_hf_2e_self_j + de_hf_2e_self_k, dtype=np.float64)
            if bool(_hf2e_include_self) else z3.copy()
        )
    de_hf_2e_rebuilt = np.asarray(de_hf_2e_cross_selected + de_hf_2e_self_selected, dtype=np.float64)
    de_ci_2e = float(_ci_2e_scale) * contract_df(bar_L_ci) if z_ci is not None else z3.copy()
    if z_ci is not None and bar_L_ci_dm1 is not None and bar_L_ci_dm2 is not None:
        de_ci_2e_dm1 = float(_ci_2e_scale) * contract_df(bar_L_ci_dm1)
        de_ci_2e_dm2 = float(_ci_2e_scale) * contract_df(bar_L_ci_dm2)
    else:
        de_ci_2e_dm1 = z3.copy()
        de_ci_2e_dm2 = z3.copy()
    if z_ci is not None and bar_L_ci_par is not None and bar_L_ci_tan is not None:
        de_ci_2e_parallel = float(_ci_2e_scale) * contract_df(bar_L_ci_par)
        de_ci_2e_tangent = float(_ci_2e_scale) * contract_df(bar_L_ci_tan)
    else:
        de_ci_2e_parallel = z3.copy()
        de_ci_2e_tangent = z3.copy()
    de_ci_2e_rebuilt = np.asarray(de_ci_2e_dm1 + de_ci_2e_dm2, dtype=np.float64)

    de_orb_2e = float(_orb_2e_scale) * contract_df(bar_L_orb)
    if bar_L_orb_mean is not None and bar_L_orb_dm2 is not None:
        de_orb_2e_mean = float(_orb_2e_scale) * contract_df(bar_L_orb_mean)
        de_orb_2e_dm2 = float(_orb_2e_scale) * contract_df(bar_L_orb_dm2)
    else:
        de_orb_2e_mean = z3.copy()
        de_orb_2e_dm2 = z3.copy()
    de_orb_2e_rebuilt = np.asarray(de_orb_2e_mean + de_orb_2e_dm2, dtype=np.float64)

    de_ref_1e = contract_h1(D_ref_1e)
    de_hf_1e_dpt2 = float(_hf_1e_scale) * contract_h1(D_hf_1e)
    de_hf_1e_dpt2c_sym_quarter = float(_hf_1e_scale) * contract_h1(
        D_hf_1e_dpt2c_sym_quarter
    )
    de_hf_1e_molcas_dk = float(_hf_1e_scale) * contract_h1(D_hf_1e_molcas_dk)
    de_hf_1e = float(_hf_1e_scale) * contract_h1(D_hf_1e_selected)
    de_ci_1e = float(_ci_1e_scale) * contract_h1(D_ci_1e) if z_ci is not None else z3.copy()
    if z_ci is not None and D_ci_1e_dm1 is not None and D_ci_1e_dm2 is not None:
        de_ci_1e_dm1 = float(_ci_1e_scale) * contract_h1(D_ci_1e_dm1)
        de_ci_1e_dm2 = float(_ci_1e_scale) * contract_h1(D_ci_1e_dm2)
        de_ci_1e_rebuilt_dm = np.asarray(de_ci_1e_dm1 + de_ci_1e_dm2, dtype=np.float64)
        de_ci_1e_rebuilt_dm_resid = np.asarray(de_ci_1e - de_ci_1e_rebuilt_dm, dtype=np.float64)
    else:
        de_ci_1e_dm1 = z3.copy()
        de_ci_1e_dm2 = z3.copy()
        de_ci_1e_rebuilt_dm = z3.copy()
        de_ci_1e_rebuilt_dm_resid = z3.copy()
    if z_ci is not None and D_ci_1e_par is not None and D_ci_1e_tan is not None:
        de_ci_1e_parallel = float(_ci_1e_scale) * contract_h1(D_ci_1e_par)
        de_ci_1e_tangent = float(_ci_1e_scale) * contract_h1(D_ci_1e_tan)
    else:
        de_ci_1e_parallel = z3.copy()
        de_ci_1e_tangent = z3.copy()
    de_orb_1e = float(_orb_1e_scale) * contract_h1(D_orb_1e)

    de_ref_pulay = contract_pulay(W_ref)
    de_hf_pulay = float(_hf_pulay_scale) * contract_pulay(W_hf)
    de_ci_pulay = contract_pulay(W_ci) if z_ci is not None else z3.copy()
    de_orb_pulay = float(_orb_pulay_scale) * contract_pulay(W_orb)
    de_orb_case_A_2e = de_orb_case_A_1e = de_orb_case_A_pulay = de_orb_case_A_total = z3.copy()
    de_orb_case_C_2e = de_orb_case_C_1e = de_orb_case_C_pulay = de_orb_case_C_total = z3.copy()
    de_orb_case_D_2e = de_orb_case_D_1e = de_orb_case_D_pulay = de_orb_case_D_total = z3.copy()
    de_orb_case_A_pulay_mean_vmix_j = de_orb_case_A_pulay_mean_vmix_j_core = de_orb_case_A_pulay_mean_vmix_j_act = z3.copy()
    de_orb_case_C_pulay_mean_vmix_j = de_orb_case_C_pulay_mean_vmix_j_core = de_orb_case_C_pulay_mean_vmix_j_act = z3.copy()
    de_orb_case_D_pulay_mean_vmix_j = de_orb_case_D_pulay_mean_vmix_j_core = de_orb_case_D_pulay_mean_vmix_j_act = z3.copy()
    de_orb_case_A_pulay_mean_vmix_k = de_orb_case_A_pulay_mean_vmix_k_core = de_orb_case_A_pulay_mean_vmix_k_act = z3.copy()
    de_orb_case_C_pulay_mean_vmix_k = de_orb_case_C_pulay_mean_vmix_k_core = de_orb_case_C_pulay_mean_vmix_k_act = z3.copy()
    de_orb_case_D_pulay_mean_vmix_k = de_orb_case_D_pulay_mean_vmix_k_core = de_orb_case_D_pulay_mean_vmix_k_act = z3.copy()
    de_orb_case_A_pulay_mean_vmix_k_right = de_orb_case_A_pulay_mean_vmix_k_core_right = de_orb_case_A_pulay_mean_vmix_k_act_right = z3.copy()
    de_orb_case_C_pulay_mean_vmix_k_right = de_orb_case_C_pulay_mean_vmix_k_core_right = de_orb_case_C_pulay_mean_vmix_k_act_right = z3.copy()
    de_orb_case_D_pulay_mean_vmix_k_right = de_orb_case_D_pulay_mean_vmix_k_core_right = de_orb_case_D_pulay_mean_vmix_k_act_right = z3.copy()
    de_orb_case_A_pulay_mean_vmix_k_lr_avg = de_orb_case_C_pulay_mean_vmix_k_lr_avg = de_orb_case_D_pulay_mean_vmix_k_lr_avg = z3.copy()
    de_orb_case_A_pulay_mean_vmix_k_lr_comm = de_orb_case_C_pulay_mean_vmix_k_lr_comm = de_orb_case_D_pulay_mean_vmix_k_lr_comm = z3.copy()
    bar_L_orb_np = _asnp(bar_L_orb)
    D_orb_1e_np = _asnp(D_orb_1e)
    W_orb_np = _asnp(W_orb)
    if orb_case_payload:
        for _lbl in ("A", "C", "D"):
            _payload = orb_case_payload.get(_lbl, {})
            if not _payload:
                continue
            _de2 = float(_orb_2e_scale) * contract_df(
                _asnp(_payload.get("bar_L", np.zeros_like(bar_L_orb_np)))
            )
            _de1 = float(_orb_1e_scale) * contract_h1(
                _asnp(_payload.get("D_1e", np.zeros_like(D_orb_1e_np)))
            )
            _dep = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_payload.get("W", np.zeros_like(W_orb_np)))
            )
            _w_parts_case = dict(_payload.get("W_parts", {}) or {})
            _depj = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_j", np.zeros_like(W_orb_np)))
            )
            _depj_core = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_j_core", np.zeros_like(W_orb_np)))
            )
            _depj_act = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_j_act", np.zeros_like(W_orb_np)))
            )
            _depk = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k", np.zeros_like(W_orb_np)))
            )
            _depk_core = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_core", np.zeros_like(W_orb_np)))
            )
            _depk_act = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_act", np.zeros_like(W_orb_np)))
            )
            _depk_right = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_right", np.zeros_like(W_orb_np)))
            )
            _depk_core_right = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_core_right", np.zeros_like(W_orb_np)))
            )
            _depk_act_right = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_act_right", np.zeros_like(W_orb_np)))
            )
            _depk_lr_avg = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_lr_avg", np.zeros_like(W_orb_np)))
            )
            _depk_lr_comm = float(_orb_pulay_scale) * contract_pulay(
                _asnp(_w_parts_case.get("dme0_mean_vmix_k_lr_comm", np.zeros_like(W_orb_np)))
            )
            _det = np.asarray(_de2 + _de1 + _dep, dtype=np.float64)
            if _lbl == "A":
                de_orb_case_A_2e, de_orb_case_A_1e, de_orb_case_A_pulay, de_orb_case_A_total = _de2, _de1, _dep, _det
                de_orb_case_A_pulay_mean_vmix_j = _depj
                de_orb_case_A_pulay_mean_vmix_j_core = _depj_core
                de_orb_case_A_pulay_mean_vmix_j_act = _depj_act
                de_orb_case_A_pulay_mean_vmix_k = _depk
                de_orb_case_A_pulay_mean_vmix_k_core = _depk_core
                de_orb_case_A_pulay_mean_vmix_k_act = _depk_act
                de_orb_case_A_pulay_mean_vmix_k_right = _depk_right
                de_orb_case_A_pulay_mean_vmix_k_core_right = _depk_core_right
                de_orb_case_A_pulay_mean_vmix_k_act_right = _depk_act_right
                de_orb_case_A_pulay_mean_vmix_k_lr_avg = _depk_lr_avg
                de_orb_case_A_pulay_mean_vmix_k_lr_comm = _depk_lr_comm
            elif _lbl == "C":
                de_orb_case_C_2e, de_orb_case_C_1e, de_orb_case_C_pulay, de_orb_case_C_total = _de2, _de1, _dep, _det
                de_orb_case_C_pulay_mean_vmix_j = _depj
                de_orb_case_C_pulay_mean_vmix_j_core = _depj_core
                de_orb_case_C_pulay_mean_vmix_j_act = _depj_act
                de_orb_case_C_pulay_mean_vmix_k = _depk
                de_orb_case_C_pulay_mean_vmix_k_core = _depk_core
                de_orb_case_C_pulay_mean_vmix_k_act = _depk_act
                de_orb_case_C_pulay_mean_vmix_k_right = _depk_right
                de_orb_case_C_pulay_mean_vmix_k_core_right = _depk_core_right
                de_orb_case_C_pulay_mean_vmix_k_act_right = _depk_act_right
                de_orb_case_C_pulay_mean_vmix_k_lr_avg = _depk_lr_avg
                de_orb_case_C_pulay_mean_vmix_k_lr_comm = _depk_lr_comm
            elif _lbl == "D":
                de_orb_case_D_2e, de_orb_case_D_1e, de_orb_case_D_pulay, de_orb_case_D_total = _de2, _de1, _dep, _det
                de_orb_case_D_pulay_mean_vmix_j = _depj
                de_orb_case_D_pulay_mean_vmix_j_core = _depj_core
                de_orb_case_D_pulay_mean_vmix_j_act = _depj_act
                de_orb_case_D_pulay_mean_vmix_k = _depk
                de_orb_case_D_pulay_mean_vmix_k_core = _depk_core
                de_orb_case_D_pulay_mean_vmix_k_act = _depk_act
                de_orb_case_D_pulay_mean_vmix_k_right = _depk_right
                de_orb_case_D_pulay_mean_vmix_k_core_right = _depk_core_right
                de_orb_case_D_pulay_mean_vmix_k_act_right = _depk_act_right
                de_orb_case_D_pulay_mean_vmix_k_lr_avg = _depk_lr_avg
                de_orb_case_D_pulay_mean_vmix_k_lr_comm = _depk_lr_comm
    de_orb_case_acd_rebuilt = np.asarray(
        de_orb_case_A_total + de_orb_case_C_total + de_orb_case_D_total,
        dtype=np.float64,
    )
    de_orb_case_A_pulay_mean_vmix_k_rebuilt_ca = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_core + de_orb_case_A_pulay_mean_vmix_k_act,
        dtype=np.float64,
    )
    de_orb_case_C_pulay_mean_vmix_k_rebuilt_ca = np.asarray(
        de_orb_case_C_pulay_mean_vmix_k_core + de_orb_case_C_pulay_mean_vmix_k_act,
        dtype=np.float64,
    )
    de_orb_case_D_pulay_mean_vmix_k_rebuilt_ca = np.asarray(
        de_orb_case_D_pulay_mean_vmix_k_core + de_orb_case_D_pulay_mean_vmix_k_act,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_j_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_j
        + de_orb_case_C_pulay_mean_vmix_j
        + de_orb_case_D_pulay_mean_vmix_j,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_j_core_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_j_core
        + de_orb_case_C_pulay_mean_vmix_j_core
        + de_orb_case_D_pulay_mean_vmix_j_core,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_j_act_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_j_act
        + de_orb_case_C_pulay_mean_vmix_j_act
        + de_orb_case_D_pulay_mean_vmix_j_act,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k
        + de_orb_case_C_pulay_mean_vmix_k
        + de_orb_case_D_pulay_mean_vmix_k,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_core_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_core
        + de_orb_case_C_pulay_mean_vmix_k_core
        + de_orb_case_D_pulay_mean_vmix_k_core,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_act_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_act
        + de_orb_case_C_pulay_mean_vmix_k_act
        + de_orb_case_D_pulay_mean_vmix_k_act,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_right_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_right
        + de_orb_case_C_pulay_mean_vmix_k_right
        + de_orb_case_D_pulay_mean_vmix_k_right,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_core_right_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_core_right
        + de_orb_case_C_pulay_mean_vmix_k_core_right
        + de_orb_case_D_pulay_mean_vmix_k_core_right,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_act_right_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_act_right
        + de_orb_case_C_pulay_mean_vmix_k_act_right
        + de_orb_case_D_pulay_mean_vmix_k_act_right,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_lr_avg_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_lr_avg
        + de_orb_case_C_pulay_mean_vmix_k_lr_avg
        + de_orb_case_D_pulay_mean_vmix_k_lr_avg,
        dtype=np.float64,
    )
    de_orb_case_pulay_mean_vmix_k_lr_comm_rebuilt_acd = np.asarray(
        de_orb_case_A_pulay_mean_vmix_k_lr_comm
        + de_orb_case_C_pulay_mean_vmix_k_lr_comm
        + de_orb_case_D_pulay_mean_vmix_k_lr_comm,
        dtype=np.float64,
    )
    if W_orb_parts is not None:
        de_orb_pulay_h1 = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_h1", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vLmix = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vLmix", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vL_c = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vL_c", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vc_L = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vc_L", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_j = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_j", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_k = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_k", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_j = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_j", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_j_core = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_j_core", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_j_act = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_j_act", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_core = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_core", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_act = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_act", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_right = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_right", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_core_right = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_core_right", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_act_right = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_act_right", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_lr_avg = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_lr_avg", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_core_lr_avg = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_core_lr_avg", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_act_lr_avg = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_act_lr_avg", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_lr_comm = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_lr_comm", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_core_lr_comm = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_core_lr_comm", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vmix_k_act_lr_comm = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vmix_k_act_lr_comm", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vLmix_j = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vLmix_j", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vLmix_k = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vLmix_k", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vL_c_j = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vL_c_j", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vL_c_k = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vL_c_k", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vc_L_j = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vc_L_j", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_vc_L_k = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_mean_vc_L_k", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_mean_rebuilt = np.asarray(
            de_orb_pulay_mean_vmix
            + de_orb_pulay_mean_vLmix
            + de_orb_pulay_mean_vL_c
            + de_orb_pulay_mean_vc_L,
            dtype=np.float64,
        )
        de_orb_pulay_mean_rebuilt_jk = np.asarray(
            de_orb_pulay_mean_j + de_orb_pulay_mean_k,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vmix_rebuilt_jk = np.asarray(
            de_orb_pulay_mean_vmix_j + de_orb_pulay_mean_vmix_k,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vmix_j_rebuilt_ca = np.asarray(
            de_orb_pulay_mean_vmix_j_core + de_orb_pulay_mean_vmix_j_act,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vmix_k_rebuilt_ca = np.asarray(
            de_orb_pulay_mean_vmix_k_core + de_orb_pulay_mean_vmix_k_act,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vmix_k_right_rebuilt_ca = np.asarray(
            de_orb_pulay_mean_vmix_k_core_right + de_orb_pulay_mean_vmix_k_act_right,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vmix_k_lr_avg_rebuilt_ca = np.asarray(
            de_orb_pulay_mean_vmix_k_core_lr_avg + de_orb_pulay_mean_vmix_k_act_lr_avg,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vmix_k_lr_comm_rebuilt_ca = np.asarray(
            de_orb_pulay_mean_vmix_k_core_lr_comm + de_orb_pulay_mean_vmix_k_act_lr_comm,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vLmix_rebuilt_jk = np.asarray(
            de_orb_pulay_mean_vLmix_j + de_orb_pulay_mean_vLmix_k,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vL_c_rebuilt_jk = np.asarray(
            de_orb_pulay_mean_vL_c_j + de_orb_pulay_mean_vL_c_k,
            dtype=np.float64,
        )
        de_orb_pulay_mean_vc_L_rebuilt_jk = np.asarray(
            de_orb_pulay_mean_vc_L_j + de_orb_pulay_mean_vc_L_k,
            dtype=np.float64,
        )
        de_orb_pulay_aa1 = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_aa1", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_aa1_j = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_aa1_j", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_aa1_k = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_aa1_k", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_aa1_rebuilt_jk = np.asarray(
            de_orb_pulay_aa1_j + de_orb_pulay_aa1_k,
            dtype=np.float64,
        )
        de_orb_pulay_aa2 = float(_orb_pulay_scale) * contract_pulay(
            np.asarray(W_orb_parts.get("dme0_aa2", np.zeros_like(W_orb_np)), dtype=np.float64)
        )
        de_orb_pulay_rebuilt = np.asarray(
            de_orb_pulay_h1 + de_orb_pulay_mean + de_orb_pulay_aa1 + de_orb_pulay_aa2,
            dtype=np.float64,
        )
    else:
        de_orb_pulay_h1 = z3.copy()
        de_orb_pulay_mean = z3.copy()
        de_orb_pulay_mean_vmix = z3.copy()
        de_orb_pulay_mean_vLmix = z3.copy()
        de_orb_pulay_mean_vL_c = z3.copy()
        de_orb_pulay_mean_vc_L = z3.copy()
        de_orb_pulay_mean_j = z3.copy()
        de_orb_pulay_mean_k = z3.copy()
        de_orb_pulay_mean_vmix_j = z3.copy()
        de_orb_pulay_mean_vmix_k = z3.copy()
        de_orb_pulay_mean_vmix_j_core = z3.copy()
        de_orb_pulay_mean_vmix_j_act = z3.copy()
        de_orb_pulay_mean_vmix_k_core = z3.copy()
        de_orb_pulay_mean_vmix_k_act = z3.copy()
        de_orb_pulay_mean_vmix_k_right = z3.copy()
        de_orb_pulay_mean_vmix_k_core_right = z3.copy()
        de_orb_pulay_mean_vmix_k_act_right = z3.copy()
        de_orb_pulay_mean_vmix_k_lr_avg = z3.copy()
        de_orb_pulay_mean_vmix_k_core_lr_avg = z3.copy()
        de_orb_pulay_mean_vmix_k_act_lr_avg = z3.copy()
        de_orb_pulay_mean_vmix_k_lr_comm = z3.copy()
        de_orb_pulay_mean_vmix_k_core_lr_comm = z3.copy()
        de_orb_pulay_mean_vmix_k_act_lr_comm = z3.copy()
        de_orb_pulay_mean_vLmix_j = z3.copy()
        de_orb_pulay_mean_vLmix_k = z3.copy()
        de_orb_pulay_mean_vL_c_j = z3.copy()
        de_orb_pulay_mean_vL_c_k = z3.copy()
        de_orb_pulay_mean_vc_L_j = z3.copy()
        de_orb_pulay_mean_vc_L_k = z3.copy()
        de_orb_pulay_mean_rebuilt = z3.copy()
        de_orb_pulay_mean_rebuilt_jk = z3.copy()
        de_orb_pulay_mean_vmix_rebuilt_jk = z3.copy()
        de_orb_pulay_mean_vmix_j_rebuilt_ca = z3.copy()
        de_orb_pulay_mean_vmix_k_rebuilt_ca = z3.copy()
        de_orb_pulay_mean_vmix_k_right_rebuilt_ca = z3.copy()
        de_orb_pulay_mean_vmix_k_lr_avg_rebuilt_ca = z3.copy()
        de_orb_pulay_mean_vmix_k_lr_comm_rebuilt_ca = z3.copy()
        de_orb_pulay_mean_vLmix_rebuilt_jk = z3.copy()
        de_orb_pulay_mean_vL_c_rebuilt_jk = z3.copy()
        de_orb_pulay_mean_vc_L_rebuilt_jk = z3.copy()
        de_orb_pulay_aa1 = z3.copy()
        de_orb_pulay_aa1_j = z3.copy()
        de_orb_pulay_aa1_k = z3.copy()
        de_orb_pulay_aa1_rebuilt_jk = z3.copy()
        de_orb_pulay_aa2 = z3.copy()
        de_orb_pulay_rebuilt = z3.copy()

    de_hf = np.asarray(de_hf_2e + de_hf_1e + de_hf_pulay, dtype=np.float64)
    de_ci = np.asarray(de_ci_2e + de_ci_1e + de_ci_pulay, dtype=np.float64)
    de_orb = np.asarray(de_orb_2e + de_orb_1e + de_orb_pulay, dtype=np.float64)
    grad_pt2 = np.asarray(de_hf + de_ci + de_orb, dtype=np.float64)
    grad_ref = np.asarray(de_nuc + de_ref_2e + de_ref_1e + de_ref_pulay, dtype=np.float64)

    breakdown = {
        # Main gradient split
        "grad_ref": grad_ref,
        "grad_pt2": grad_pt2,
        # Current SS-native channel split
        "de_hf_2e": de_hf_2e,
        "de_hf_2e_base": de_hf_2e_base,
        "de_hf_2e_btamp_dense_raw": de_btamp_dense_raw,
        "de_hf_2e_btamp_dense": de_btamp_dense_scaled,
        "de_hf_2e_cross_j_core": de_hf_2e_cross_j_core,
        "de_hf_2e_cross_j_act": de_hf_2e_cross_j_act,
        "de_hf_2e_cross_k_core": de_hf_2e_cross_k_core,
        "de_hf_2e_cross_k_act": de_hf_2e_cross_k_act,
        "de_hf_2e_self_j": de_hf_2e_self_j,
        "de_hf_2e_self_k": de_hf_2e_self_k,
        "de_hf_2e_cross_core": de_hf_2e_cross_core,
        "de_hf_2e_cross_act": de_hf_2e_cross_act,
        "de_hf_2e_cross_j_selected_core": de_hf_2e_cross_core_j,
        "de_hf_2e_cross_j_selected_act": de_hf_2e_cross_act_j,
        "de_hf_2e_cross_selected": de_hf_2e_cross_selected,
        "de_hf_2e_self_selected": de_hf_2e_self_selected,
        "de_hf_2e_rebuilt": de_hf_2e_rebuilt,
        "de_hf_2e_rebuild_resid": np.asarray(de_hf_2e - de_hf_2e_rebuilt, dtype=np.float64),
        "btamp_dense_term": "total",
        "btamp_dense_scale": 0.0,
        "btamp_dense_diag": False,
        "btamp_dense_error": "",
        "de_hf_1e": de_hf_1e,
        "de_hf_1e_dpt2": de_hf_1e_dpt2,
        "de_hf_1e_dpt2c_sym_quarter": de_hf_1e_dpt2c_sym_quarter,
        "de_hf_1e_molcas_dk": de_hf_1e_molcas_dk,
        "de_hf_1e_molcas_dk_rebuilt": np.asarray(
            de_hf_1e_dpt2 + de_hf_1e_dpt2c_sym_quarter,
            dtype=np.float64,
        ),
        "de_hf_1e_molcas_dk_rebuild_resid": np.asarray(
            de_hf_1e_molcas_dk - (de_hf_1e_dpt2 + de_hf_1e_dpt2c_sym_quarter),
            dtype=np.float64,
        ),
        "hf_1e_mode": str(_hf1e_mode),
        "de_hf_pulay": de_hf_pulay,
        "de_hf": de_hf,
        "de_ci_2e": de_ci_2e,
        "de_ci_2e_dm1": de_ci_2e_dm1,
        "de_ci_2e_dm2": de_ci_2e_dm2,
        "de_ci_2e_parallel": de_ci_2e_parallel,
        "de_ci_2e_tangent": de_ci_2e_tangent,
        "de_ci_2e_rebuilt": de_ci_2e_rebuilt,
        "de_ci_2e_rebuild_resid": np.asarray(de_ci_2e - de_ci_2e_rebuilt, dtype=np.float64),
        "de_ci_1e": de_ci_1e,
        "de_ci_1e_dm1": de_ci_1e_dm1,
        "de_ci_1e_dm2": de_ci_1e_dm2,
        "de_ci_1e_rebuilt_dm": de_ci_1e_rebuilt_dm,
        "de_ci_1e_rebuilt_dm_resid": de_ci_1e_rebuilt_dm_resid,
        "de_ci_1e_parallel": de_ci_1e_parallel,
        "de_ci_1e_tangent": de_ci_1e_tangent,
        "de_ci_pulay": de_ci_pulay,
        "de_ci": de_ci,
        "de_orb_2e": de_orb_2e,
        "de_orb_2e_mean": de_orb_2e_mean,
        "de_orb_2e_dm2": de_orb_2e_dm2,
        "de_orb_2e_rebuilt": de_orb_2e_rebuilt,
        "de_orb_2e_rebuild_resid": np.asarray(de_orb_2e - de_orb_2e_rebuilt, dtype=np.float64),
        "de_orb_1e": de_orb_1e,
        "de_orb_pulay": de_orb_pulay,
        "de_orb_case_A_2e": de_orb_case_A_2e,
        "de_orb_case_A_1e": de_orb_case_A_1e,
        "de_orb_case_A_pulay": de_orb_case_A_pulay,
        "de_orb_case_A_total": de_orb_case_A_total,
        "de_orb_case_C_2e": de_orb_case_C_2e,
        "de_orb_case_C_1e": de_orb_case_C_1e,
        "de_orb_case_C_pulay": de_orb_case_C_pulay,
        "de_orb_case_C_total": de_orb_case_C_total,
        "de_orb_case_D_2e": de_orb_case_D_2e,
        "de_orb_case_D_1e": de_orb_case_D_1e,
        "de_orb_case_D_pulay": de_orb_case_D_pulay,
        "de_orb_case_D_total": de_orb_case_D_total,
        "de_orb_case_acd_rebuilt": de_orb_case_acd_rebuilt,
        "de_orb_case_acd_rebuild_resid": np.asarray(
            de_orb - de_orb_case_acd_rebuilt,
            dtype=np.float64,
        ),
        "de_orb_case_A_pulay_mean_vmix_j": de_orb_case_A_pulay_mean_vmix_j,
        "de_orb_case_A_pulay_mean_vmix_j_core": de_orb_case_A_pulay_mean_vmix_j_core,
        "de_orb_case_A_pulay_mean_vmix_j_act": de_orb_case_A_pulay_mean_vmix_j_act,
        "de_orb_case_A_pulay_mean_vmix_k": de_orb_case_A_pulay_mean_vmix_k,
        "de_orb_case_A_pulay_mean_vmix_k_core": de_orb_case_A_pulay_mean_vmix_k_core,
        "de_orb_case_A_pulay_mean_vmix_k_act": de_orb_case_A_pulay_mean_vmix_k_act,
        "de_orb_case_A_pulay_mean_vmix_k_right": de_orb_case_A_pulay_mean_vmix_k_right,
        "de_orb_case_A_pulay_mean_vmix_k_core_right": de_orb_case_A_pulay_mean_vmix_k_core_right,
        "de_orb_case_A_pulay_mean_vmix_k_act_right": de_orb_case_A_pulay_mean_vmix_k_act_right,
        "de_orb_case_A_pulay_mean_vmix_k_lr_avg": de_orb_case_A_pulay_mean_vmix_k_lr_avg,
        "de_orb_case_A_pulay_mean_vmix_k_lr_comm": de_orb_case_A_pulay_mean_vmix_k_lr_comm,
        "de_orb_case_A_pulay_mean_vmix_k_rebuilt_ca": de_orb_case_A_pulay_mean_vmix_k_rebuilt_ca,
        "de_orb_case_A_pulay_mean_vmix_k_rebuilt_ca_resid": np.asarray(
            de_orb_case_A_pulay_mean_vmix_k - de_orb_case_A_pulay_mean_vmix_k_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_case_C_pulay_mean_vmix_j": de_orb_case_C_pulay_mean_vmix_j,
        "de_orb_case_C_pulay_mean_vmix_j_core": de_orb_case_C_pulay_mean_vmix_j_core,
        "de_orb_case_C_pulay_mean_vmix_j_act": de_orb_case_C_pulay_mean_vmix_j_act,
        "de_orb_case_C_pulay_mean_vmix_k": de_orb_case_C_pulay_mean_vmix_k,
        "de_orb_case_C_pulay_mean_vmix_k_core": de_orb_case_C_pulay_mean_vmix_k_core,
        "de_orb_case_C_pulay_mean_vmix_k_act": de_orb_case_C_pulay_mean_vmix_k_act,
        "de_orb_case_C_pulay_mean_vmix_k_right": de_orb_case_C_pulay_mean_vmix_k_right,
        "de_orb_case_C_pulay_mean_vmix_k_core_right": de_orb_case_C_pulay_mean_vmix_k_core_right,
        "de_orb_case_C_pulay_mean_vmix_k_act_right": de_orb_case_C_pulay_mean_vmix_k_act_right,
        "de_orb_case_C_pulay_mean_vmix_k_lr_avg": de_orb_case_C_pulay_mean_vmix_k_lr_avg,
        "de_orb_case_C_pulay_mean_vmix_k_lr_comm": de_orb_case_C_pulay_mean_vmix_k_lr_comm,
        "de_orb_case_C_pulay_mean_vmix_k_rebuilt_ca": de_orb_case_C_pulay_mean_vmix_k_rebuilt_ca,
        "de_orb_case_C_pulay_mean_vmix_k_rebuilt_ca_resid": np.asarray(
            de_orb_case_C_pulay_mean_vmix_k - de_orb_case_C_pulay_mean_vmix_k_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_case_D_pulay_mean_vmix_j": de_orb_case_D_pulay_mean_vmix_j,
        "de_orb_case_D_pulay_mean_vmix_j_core": de_orb_case_D_pulay_mean_vmix_j_core,
        "de_orb_case_D_pulay_mean_vmix_j_act": de_orb_case_D_pulay_mean_vmix_j_act,
        "de_orb_case_D_pulay_mean_vmix_k": de_orb_case_D_pulay_mean_vmix_k,
        "de_orb_case_D_pulay_mean_vmix_k_core": de_orb_case_D_pulay_mean_vmix_k_core,
        "de_orb_case_D_pulay_mean_vmix_k_act": de_orb_case_D_pulay_mean_vmix_k_act,
        "de_orb_case_D_pulay_mean_vmix_k_right": de_orb_case_D_pulay_mean_vmix_k_right,
        "de_orb_case_D_pulay_mean_vmix_k_core_right": de_orb_case_D_pulay_mean_vmix_k_core_right,
        "de_orb_case_D_pulay_mean_vmix_k_act_right": de_orb_case_D_pulay_mean_vmix_k_act_right,
        "de_orb_case_D_pulay_mean_vmix_k_lr_avg": de_orb_case_D_pulay_mean_vmix_k_lr_avg,
        "de_orb_case_D_pulay_mean_vmix_k_lr_comm": de_orb_case_D_pulay_mean_vmix_k_lr_comm,
        "de_orb_case_D_pulay_mean_vmix_k_rebuilt_ca": de_orb_case_D_pulay_mean_vmix_k_rebuilt_ca,
        "de_orb_case_D_pulay_mean_vmix_k_rebuilt_ca_resid": np.asarray(
            de_orb_case_D_pulay_mean_vmix_k - de_orb_case_D_pulay_mean_vmix_k_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_j_rebuilt_acd": de_orb_case_pulay_mean_vmix_j_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_j_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_j - de_orb_case_pulay_mean_vmix_j_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_j_core_rebuilt_acd": de_orb_case_pulay_mean_vmix_j_core_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_j_core_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_j_core - de_orb_case_pulay_mean_vmix_j_core_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_j_act_rebuilt_acd": de_orb_case_pulay_mean_vmix_j_act_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_j_act_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_j_act - de_orb_case_pulay_mean_vmix_j_act_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k - de_orb_case_pulay_mean_vmix_k_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_core_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_core_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_core_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_core - de_orb_case_pulay_mean_vmix_k_core_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_act_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_act_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_act_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_act - de_orb_case_pulay_mean_vmix_k_act_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_right_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_right_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_right_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_right - de_orb_case_pulay_mean_vmix_k_right_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_core_right_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_core_right_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_core_right_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_core_right - de_orb_case_pulay_mean_vmix_k_core_right_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_act_right_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_act_right_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_act_right_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_act_right - de_orb_case_pulay_mean_vmix_k_act_right_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_lr_avg_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_lr_avg_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_lr_avg_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_lr_avg - de_orb_case_pulay_mean_vmix_k_lr_avg_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_case_pulay_mean_vmix_k_lr_comm_rebuilt_acd": de_orb_case_pulay_mean_vmix_k_lr_comm_rebuilt_acd,
        "de_orb_case_pulay_mean_vmix_k_lr_comm_rebuilt_acd_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_lr_comm - de_orb_case_pulay_mean_vmix_k_lr_comm_rebuilt_acd,
            dtype=np.float64,
        ),
        "de_orb_pulay_h1": de_orb_pulay_h1,
        "de_orb_pulay_mean": de_orb_pulay_mean,
        "de_orb_pulay_mean_vmix": de_orb_pulay_mean_vmix,
        "de_orb_pulay_mean_vLmix": de_orb_pulay_mean_vLmix,
        "de_orb_pulay_mean_vL_c": de_orb_pulay_mean_vL_c,
        "de_orb_pulay_mean_vc_L": de_orb_pulay_mean_vc_L,
        "de_orb_pulay_mean_j": de_orb_pulay_mean_j,
        "de_orb_pulay_mean_k": de_orb_pulay_mean_k,
        "de_orb_pulay_mean_rebuilt_jk": de_orb_pulay_mean_rebuilt_jk,
        "de_orb_pulay_mean_rebuilt_jk_resid": np.asarray(
            de_orb_pulay_mean - de_orb_pulay_mean_rebuilt_jk,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vmix_j": de_orb_pulay_mean_vmix_j,
        "de_orb_pulay_mean_vmix_k": de_orb_pulay_mean_vmix_k,
        "de_orb_pulay_mean_vmix_j_core": de_orb_pulay_mean_vmix_j_core,
        "de_orb_pulay_mean_vmix_j_act": de_orb_pulay_mean_vmix_j_act,
        "de_orb_pulay_mean_vmix_j_rebuilt_ca": de_orb_pulay_mean_vmix_j_rebuilt_ca,
        "de_orb_pulay_mean_vmix_j_rebuilt_ca_resid": np.asarray(
            de_orb_pulay_mean_vmix_j - de_orb_pulay_mean_vmix_j_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vmix_k_core": de_orb_pulay_mean_vmix_k_core,
        "de_orb_pulay_mean_vmix_k_act": de_orb_pulay_mean_vmix_k_act,
        "de_orb_pulay_mean_vmix_k_right": de_orb_pulay_mean_vmix_k_right,
        "de_orb_pulay_mean_vmix_k_core_right": de_orb_pulay_mean_vmix_k_core_right,
        "de_orb_pulay_mean_vmix_k_act_right": de_orb_pulay_mean_vmix_k_act_right,
        "de_orb_pulay_mean_vmix_k_lr_avg": de_orb_pulay_mean_vmix_k_lr_avg,
        "de_orb_pulay_mean_vmix_k_core_lr_avg": de_orb_pulay_mean_vmix_k_core_lr_avg,
        "de_orb_pulay_mean_vmix_k_act_lr_avg": de_orb_pulay_mean_vmix_k_act_lr_avg,
        "de_orb_pulay_mean_vmix_k_lr_comm": de_orb_pulay_mean_vmix_k_lr_comm,
        "de_orb_pulay_mean_vmix_k_core_lr_comm": de_orb_pulay_mean_vmix_k_core_lr_comm,
        "de_orb_pulay_mean_vmix_k_act_lr_comm": de_orb_pulay_mean_vmix_k_act_lr_comm,
        "de_orb_pulay_mean_vmix_k_rebuilt_ca": de_orb_pulay_mean_vmix_k_rebuilt_ca,
        "de_orb_pulay_mean_vmix_k_rebuilt_ca_resid": np.asarray(
            de_orb_pulay_mean_vmix_k - de_orb_pulay_mean_vmix_k_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vmix_k_right_rebuilt_ca": de_orb_pulay_mean_vmix_k_right_rebuilt_ca,
        "de_orb_pulay_mean_vmix_k_right_rebuilt_ca_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_right - de_orb_pulay_mean_vmix_k_right_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vmix_k_lr_avg_rebuilt_ca": de_orb_pulay_mean_vmix_k_lr_avg_rebuilt_ca,
        "de_orb_pulay_mean_vmix_k_lr_avg_rebuilt_ca_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_lr_avg - de_orb_pulay_mean_vmix_k_lr_avg_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vmix_k_lr_comm_rebuilt_ca": de_orb_pulay_mean_vmix_k_lr_comm_rebuilt_ca,
        "de_orb_pulay_mean_vmix_k_lr_comm_rebuilt_ca_resid": np.asarray(
            de_orb_pulay_mean_vmix_k_lr_comm - de_orb_pulay_mean_vmix_k_lr_comm_rebuilt_ca,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vmix_rebuilt_jk": de_orb_pulay_mean_vmix_rebuilt_jk,
        "de_orb_pulay_mean_vmix_rebuilt_jk_resid": np.asarray(
            de_orb_pulay_mean_vmix - de_orb_pulay_mean_vmix_rebuilt_jk,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vLmix_j": de_orb_pulay_mean_vLmix_j,
        "de_orb_pulay_mean_vLmix_k": de_orb_pulay_mean_vLmix_k,
        "de_orb_pulay_mean_vLmix_rebuilt_jk": de_orb_pulay_mean_vLmix_rebuilt_jk,
        "de_orb_pulay_mean_vLmix_rebuilt_jk_resid": np.asarray(
            de_orb_pulay_mean_vLmix - de_orb_pulay_mean_vLmix_rebuilt_jk,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vL_c_j": de_orb_pulay_mean_vL_c_j,
        "de_orb_pulay_mean_vL_c_k": de_orb_pulay_mean_vL_c_k,
        "de_orb_pulay_mean_vL_c_rebuilt_jk": de_orb_pulay_mean_vL_c_rebuilt_jk,
        "de_orb_pulay_mean_vL_c_rebuilt_jk_resid": np.asarray(
            de_orb_pulay_mean_vL_c - de_orb_pulay_mean_vL_c_rebuilt_jk,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_vc_L_j": de_orb_pulay_mean_vc_L_j,
        "de_orb_pulay_mean_vc_L_k": de_orb_pulay_mean_vc_L_k,
        "de_orb_pulay_mean_vc_L_rebuilt_jk": de_orb_pulay_mean_vc_L_rebuilt_jk,
        "de_orb_pulay_mean_vc_L_rebuilt_jk_resid": np.asarray(
            de_orb_pulay_mean_vc_L - de_orb_pulay_mean_vc_L_rebuilt_jk,
            dtype=np.float64,
        ),
        "de_orb_pulay_mean_rebuilt": de_orb_pulay_mean_rebuilt,
        "de_orb_pulay_mean_rebuild_resid": np.asarray(
            de_orb_pulay_mean - de_orb_pulay_mean_rebuilt,
            dtype=np.float64,
        ),
        "de_orb_pulay_aa1": de_orb_pulay_aa1,
        "de_orb_pulay_aa1_j": de_orb_pulay_aa1_j,
        "de_orb_pulay_aa1_k": de_orb_pulay_aa1_k,
        "de_orb_pulay_aa1_rebuilt_jk": de_orb_pulay_aa1_rebuilt_jk,
        "de_orb_pulay_aa1_rebuilt_jk_resid": np.asarray(
            de_orb_pulay_aa1 - de_orb_pulay_aa1_rebuilt_jk,
            dtype=np.float64,
        ),
        "de_orb_pulay_aa2": de_orb_pulay_aa2,
        "de_orb_pulay_rebuilt": de_orb_pulay_rebuilt,
        "de_orb_pulay_rebuild_resid": np.asarray(de_orb_pulay - de_orb_pulay_rebuilt, dtype=np.float64),
        "de_orb": de_orb,
        # Legacy aliases used by existing diagnostics/scripts.
        "de_unrelaxed": de_hf,
        "de_resp": np.asarray(de_ci + de_orb, dtype=np.float64),
        "de_2e_unrelaxed": de_hf_2e,
        "de_h1_fock": de_hf_1e,
        "de_pulay_pt2": de_hf_pulay,
        "de_lci": de_ci,
        "de_lci_df2e": de_ci_2e,
        "de_lci_h1": de_ci_1e,
        "de_lorb": de_orb,
        "de_lorb_df2e": de_orb_2e,
        "de_lorb_h1": de_orb_1e,
        # Diagnostics
        "de_ref_2e": de_ref_2e,
        "de_ref_1e": de_ref_1e,
        "de_ref_pulay": de_ref_pulay,
        "z_ci_mode": str(_zci_mode),
        "z_ci_pulay_mode": str(_zci_mode),
        "z_ci_overlap": np.asarray(z_ci_overlap, dtype=np.float64),
        "z_ci_alpha": np.asarray(z_ci_alpha, dtype=np.float64),
        "z_ci_norm": np.asarray(z_ci_norm, dtype=np.float64),
        "z_ci_parallel_norm": np.asarray(z_ci_parallel_norm, dtype=np.float64),
        "z_ci_tangent_norm": np.asarray(z_ci_tangent_norm, dtype=np.float64),
        "z_ci_overlap_raw": np.asarray(z_ci_overlap_raw, dtype=np.float64),
        "z_ci_alpha_raw": np.asarray(z_ci_alpha_raw, dtype=np.float64),
        "z_ci_norm_raw": np.asarray(z_ci_norm_raw, dtype=np.float64),
        "z_ci_parallel_norm_raw": np.asarray(z_ci_parallel_norm_raw, dtype=np.float64),
        "z_ci_tangent_norm_raw": np.asarray(z_ci_tangent_norm_raw, dtype=np.float64),
        "z_ci_map_mode": str(_ci_z_map_mode),
        "z_ci_map_applied": bool(_ci_z_map_applied),
        "z_ci_map_perm": np.asarray(_ci_z_map_perm, dtype=np.int64),
        "z_ci_map_signs": np.asarray(_ci_z_map_signs, dtype=np.float64),
        "z_ci_map_delta_norm": np.asarray(z_ci_map_delta_norm, dtype=np.float64),
        "z_ci_map_delta_max_abs": np.asarray(z_ci_map_delta_max_abs, dtype=np.float64),
        "ci_ref_map_delta_norm": np.asarray(ci_ref_map_delta_norm, dtype=np.float64),
        "ci_ref_map_delta_max_abs": np.asarray(ci_ref_map_delta_max_abs, dtype=np.float64),
        "ci_dm1_asym_ratio": np.asarray(ci_dm1_asym_ratio, dtype=np.float64),
        "ci_dm2_pair_asym_ratio": np.asarray(ci_dm2_pair_asym_ratio, dtype=np.float64),
        "z_ci_trans_rdm_mode": str(_ci_tdm_mode),
        "dm1_sa": np.asarray(dm1_sa, dtype=np.float64),
        "dm2_sa": np.asarray(dm2_sa, dtype=np.float64),
        "dm1_lci": np.asarray(dm1_lci, dtype=np.float64) if z_ci is not None else np.zeros((int(ncas), int(ncas)), dtype=np.float64),
        "dm2_lci": np.asarray(dm2_lci, dtype=np.float64) if z_ci is not None else np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64),
        "dm1_lci_tangent": np.asarray(dm1_lci_tan, dtype=np.float64) if dm1_lci_tan is not None else np.zeros((int(ncas), int(ncas)), dtype=np.float64),
        "dm2_lci_tangent": np.asarray(dm2_lci_tan, dtype=np.float64) if dm2_lci_tan is not None else np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64),
        "dm1_lci_parallel": np.asarray(dm1_lci_par, dtype=np.float64) if dm1_lci_par is not None else np.zeros((int(ncas), int(ncas)), dtype=np.float64),
        "dm2_lci_parallel": np.asarray(dm2_lci_par, dtype=np.float64) if dm2_lci_par is not None else np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64),
        "z_ci_trans_swap_braket": False,
        "hf2e_ref_mode": str(_hf2e_ref_mode),
        "hf2e_build_mode": str(_hf2e_build_mode),
        "hf2e_include_self": bool(_hf2e_include_self),
        "scale_hf_2e": float(_hf_2e_scale),
        "scale_hf_1e": float(_hf_1e_scale),
        "scale_hf_pulay": float(_hf_pulay_scale),
        "scale_ci_2e": float(_ci_2e_scale),
        "scale_ci_1e": float(_ci_1e_scale),
        "scale_ci_pulay": float(_ci_pulay_scale_eff),
        "scale_orb_2e": float(_orb_2e_scale),
        "scale_orb_1e": float(_orb_1e_scale),
        "scale_orb_pulay": float(_orb_pulay_scale),
        "lorb_wlag_mode": str(_lorb_wlag_mode),
        "lorb_wlag_included": bool(_include_lorb_wlag),
        "lorb_pulay_fd_mode": False,
        "lorb_dml_sym_mode": "full",
        "lorb_vmix_dml_core_mode": "as_dml",
        "lorb_vmix_j_mm_mode": "left",
        "lorb_vmix_k_mm_mode": "left",
        "ci_trans_dm1_sym_mode": str(_ci_dm1_sym_mode) if z_ci is not None else "n/a",
        "ci_trans_dm2_sym_mode": str(_ci_dm2_sym_mode) if z_ci is not None else "n/a",
    }
    # WLag norms/diagnostics (dump-to-disk disabled)
    _wlag_ao = {
        "W_ref": np.asarray(W_ref, dtype=np.float64),
        "W_hf": np.asarray(W_hf, dtype=np.float64),
        "W_ci": np.asarray(W_ci, dtype=np.float64),
        "W_orb": np.asarray(W_orb, dtype=np.float64),
        "W_total": np.asarray(W_total, dtype=np.float64),
    }
    breakdown["wlag_ao_norms"] = {
        str(k): float(np.linalg.norm(v))
        for k, v in _wlag_ao.items()
    }
    breakdown["wlag_ao_max_abs"] = {
        str(k): float(np.max(np.abs(v)))
        for k, v in _wlag_ao.items()
    }
    if orb_case_payload:
        _case_w_ao: dict[str, np.ndarray] = {}
        _case_w_parts_ao: dict[str, dict[str, np.ndarray]] = {}
        for _lbl, _payload in orb_case_payload.items():
            _w_case = np.asarray(
                _payload.get("W", np.zeros_like(W_orb)),
                dtype=np.float64,
            )
            _case_w_ao[str(_lbl)] = _w_case
            _parts_case = {
                str(k): np.asarray(v, dtype=np.float64)
                for k, v in dict(_payload.get("W_parts", {}) or {}).items()
                if isinstance(k, str)
            }
            _case_w_parts_ao[str(_lbl)] = _parts_case
        breakdown["orb_case_wlag_ao_norms"] = {
            str(k): float(np.linalg.norm(v))
            for k, v in _case_w_ao.items()
        }
        breakdown["orb_case_wlag_ao_max_abs"] = {
            str(k): float(np.max(np.abs(v)))
            for k, v in _case_w_ao.items()
        }
        if _case_w_parts_ao:
            breakdown["orb_case_wlag_parts_ao_norms"] = {
                str(lbl): {
                    str(k): float(np.linalg.norm(v))
                    for k, v in parts.items()
                    if v.ndim == 2
                }
                for lbl, parts in _case_w_parts_ao.items()
            }
            breakdown["orb_case_wlag_parts_ao_max_abs"] = {
                str(lbl): {
                    str(k): float(np.max(np.abs(v)))
                    for k, v in parts.items()
                    if v.ndim == 2
                }
                for lbl, parts in _case_w_parts_ao.items()
            }
    if W_orb_parts is not None:
        _w_parts_np = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in dict(W_orb_parts).items()
            if isinstance(k, str)
        }
        breakdown["orb_pulay_parts_ao_norms"] = {
            str(k): float(np.linalg.norm(v))
            for k, v in _w_parts_np.items()
            if v.ndim == 2
        }
        breakdown["orb_pulay_parts_ao_max_abs"] = {
            str(k): float(np.max(np.abs(v)))
            for k, v in _w_parts_np.items()
            if v.ndim == 2
        }

    return breakdown


def _assemble_gradient(
    scf_out, casscf, lagrangians,
    z_orb, z_ci,
    ncore, ncas, nelecas, twos,
    B_ao, h_ao, S_ao, C,
    dm1_act, dm2_act, ci_raw,
    eri_mo=None,
    df_backend="cpu",
    int1e_contract_backend="cpu",
    ci_trans_rdm_mode: str = "solver",
    return_components: bool = False,
    collect_breakdown: bool = False,
    verbose=0,
    dm1_sa_override=None,
    dm2_sa_override=None,
    dm1_casscf_direct_override=None,
    dm2_casscf_direct_override=None,
):
    """Assemble the final nuclear gradient from Lagrangians + Z-vector."""
    from asuka.integrals.grad import (
        _basis_cart_shifted_by_atom,
        _contract_bar_with_B_streamed,
        compute_df_gradient_contributions_analytic_from_bar_x_packed_bases,
        compute_df_gradient_contributions_analytic_packed_bases,
        compute_df_gradient_contributions_fd_packed_bases,
    )
    from asuka.integrals.int1e_cart import (
        contract_dS_cart,
        contract_dhcore_cart,
        shell_to_atom_map,
    )
    from asuka.mcscf.nac._df import (
        _build_bar_L_lorb_df,
        _build_bar_L_net_active_df,
    )
    from asuka.mcscf.nuc_grad_df import (
        _apply_df_pool_policy,
        _as_xp_f64,
        _build_bar_L_casscf_df,
        _build_gfock_casscf_df,
        _mol_coords_charges_bohr,
        _resolve_xp,
    )
    from asuka.mcscf.state_average import (
        ci_as_list,
        make_state_averaged_rdms,
        normalize_weights,
    )
    from asuka.solver import GUGAFCISolver

    mol = getattr(scf_out, "mol")
    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])
    nmo = C.shape[1]

    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    xp, _is_gpu = _resolve_xp(df_backend)
    B_ao_x = _as_xp_f64(xp, B_ao)
    # Expand packed 2D (naux, ntri) B to full 3D (nao, nao, naux) if needed.
    if getattr(B_ao_x, "ndim", 3) == 2:
        from asuka.integrals.df_packed_s2 import unpack_Qp_to_mnQ
        _nao_unpack = int(C.shape[0])
        B_ao_x = _as_xp_f64(xp, unpack_Qp_to_mnQ(B_ao_x, nao=_nao_unpack))
        B_ao = np.asarray(B_ao_x.get() if hasattr(B_ao_x, "get") else B_ao_x, dtype=np.float64)
    h_ao_x = _as_xp_f64(xp, h_ao)
    C_x = _as_xp_f64(xp, C)
    _restore_pool = _apply_df_pool_policy(B_ao_x, label="caspt2_grad")

    nroots = int(getattr(casscf, "nroots", 1))
    weights = normalize_weights(getattr(casscf, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(ci_raw, nroots=nroots)
    fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots))

    dm1_sa, dm2_sa = make_state_averaged_rdms(
        fcisolver, ci_list, weights,
        ncas=int(ncas), nelecas=nelecas,
    )
    dm1_sa = np.asarray(dm1_sa, dtype=np.float64)
    dm2_sa = np.asarray(dm2_sa, dtype=np.float64)
    # Allow caller to override SA density (e.g., MS-CASPT2 with U[I,iroot]^2 weights).
    if dm1_sa_override is not None:
        dm1_sa = np.asarray(dm1_sa_override, dtype=np.float64)
    if dm2_sa_override is not None:
        dm2_sa = np.asarray(dm2_sa_override, dtype=np.float64)

    # For MS gradient: direct CASSCF gradient uses MS-weighted density,
    # but Z-response still uses SA density (SA orbital stationarity constraint).
    dm1_for_direct = dm1_sa if dm1_casscf_direct_override is None else np.asarray(dm1_casscf_direct_override, dtype=np.float64)
    dm2_for_direct = dm2_sa if dm2_casscf_direct_override is None else np.asarray(dm2_casscf_direct_override, dtype=np.float64)

    # --- Unrelaxed target: CASSCF + PT2 density ---
    # Add DPT2 to the active 1-RDM (active block only for now)
    dpt2 = lagrangians["dpt2"]
    nocc = ncore + ncas

    # Build modified dm1/dm2 with PT2 correction for the full-space treatment
    # For the 1-RDM: add DPT2 to the reference
    # For the 2-RDM: approximate with mean-field (D1⊗D1 antisymmetrized)
    # The cumulant correction is captured through the Z-vector response

    # Build target gfock and densities using the CASSCF framework
    # with the reference RDMs (unrelaxed HF part)
    gfock, D_core_ao, D_act_ao, D_tot_ao, C_act = _build_gfock_casscf_df(
        B_ao_x, h_ao_x, C_x,
        ncore=int(ncore), ncas=int(ncas),
        dm1_act=dm1_for_direct, dm2_act=dm2_for_direct,
    )

    # Unrelaxed bar_L from reference CASSCF
    bar_L_ref = _build_bar_L_casscf_df(
        B_ao_x,
        D_core_ao=D_core_ao,
        D_act_ao=D_act_ao,
        C_act=C_act,
        dm2_act=dm2_for_direct,
    )

    # --- PT2 density contribution to bar_L ---
    dpt2_ao = lagrangians["dpt2_ao"]
    D_pt2_ao = _as_xp_f64(xp, dpt2_ao)

    nao = B_ao.shape[0]
    naux = B_ao.shape[2]
    D_core_ref_ao = _as_xp_f64(xp, _asnumpy_f64(D_core_ao))
    D_act_ref_ao = _as_xp_f64(xp, _asnumpy_f64(D_act_ao))

    _hf2e_ref_mode = str(os.environ.get("ASUKA_CASPT2_HF2E_REF_MODE", "core_only")).strip().lower()
    if _hf2e_ref_mode not in {"total", "core_only", "core_minus_active"}:
        _hf2e_ref_mode = "total"
    _hf2e_include_self = str(os.environ.get("ASUKA_CASPT2_HF2E_INCLUDE_SELF", "1")).strip().lower() not in {
        "0", "false", "no", "off",
    }
    _hf2e_build_mode = str(os.environ.get("ASUKA_CASPT2_HF2E_BUILD_MODE", "jk")).strip().lower()
    if _hf2e_build_mode not in {"jk", "j_only", "j_only_selfj"}:
        _hf2e_build_mode = "jk"

    _pt2_bar_result = _build_pt2_bar_L(
        B_ao_x, D_pt2_ao, D_core_ref_ao, D_act_ref_ao, xp,
        hf2e_ref_mode=_hf2e_ref_mode,
        hf2e_include_self=_hf2e_include_self,
        hf2e_build_mode=_hf2e_build_mode,
    )
    bar_L_pt2 = _pt2_bar_result["bar_L_pt2"]
    bar_hf_cross_j_core = _pt2_bar_result["bar_hf_cross_j_core"]
    bar_hf_cross_j_act = _pt2_bar_result["bar_hf_cross_j_act"]
    bar_hf_cross_k_core = _pt2_bar_result["bar_hf_cross_k_core"]
    bar_hf_cross_k_act = _pt2_bar_result["bar_hf_cross_k_act"]
    bar_hf_self_j = _pt2_bar_result["bar_hf_self_j"]
    bar_hf_self_k = _pt2_bar_result["bar_hf_self_k"]

    bar_L_hf = xp.asarray(bar_L_pt2, dtype=xp.float64)
    bar_L_ci = xp.zeros_like(bar_L_hf)
    bar_L_orb = xp.zeros_like(bar_L_hf)
    bar_L_total = xp.asarray(bar_L_ref, dtype=xp.float64)

    # --- WLag Pulay term ---
    # W_ref from gfock (CASSCF energy-weighted density)
    _C_np = _asnumpy_f64(C_x)
    _gfock_np = _asnumpy_f64(gfock)
    _nocc = ncore + ncas
    _C_occ = _C_np[:, :_nocc]
    _tmp_w = _C_np @ _gfock_np[:, :_nocc]
    W_ref = 0.5 * (_tmp_w @ _C_occ.T + _C_occ @ _tmp_w.T)

    # Add WLag from PT2: full AO matrix (NOT packed triangular)
    W_pt2_ao = lagrangians["wlag_ao"]
    W_hf = _asnumpy_f64(W_pt2_ao)
    W_ci = np.zeros_like(W_ref)
    W_orb = np.zeros_like(W_ref)
    W_orb_parts: dict[str, np.ndarray] | None = None
    W_total = np.asarray(W_ref, dtype=np.float64)

    # --- D_total for 1e derivative ---
    D_ref_1e = _asnumpy_f64(D_tot_ao)
    D_hf_1e = _asnumpy_f64(dpt2_ao)
    # Molcas MCLR (out_pt2.F90) builds effective variational PT2 density with:
    #   D_K += DPT2 + 0.25 * (DPT2C + DPT2C^T)
    # Keep these as diagnostics in the SS breakdown to compare conventions
    # term-by-term without changing the current production assembly path.
    dpt2c_ao = lagrangians.get("dpt2c_ao", None)
    if dpt2c_ao is not None:
        _dpt2c_ao = _asnumpy_f64(dpt2c_ao)
        D_hf_1e_dpt2c_sym_quarter = 0.25 * (_dpt2c_ao + _dpt2c_ao.T)
    else:
        D_hf_1e_dpt2c_sym_quarter = np.zeros_like(D_hf_1e)
    D_hf_1e_molcas_dk = np.asarray(D_hf_1e + D_hf_1e_dpt2c_sym_quarter, dtype=np.float64)
    # OpenMolcas MCLR `out_pt2.F90` builds effective PT2 1e density as:
    #   D_K += DPT2 + 0.25 * (DPT2C + DPT2C^T)
    _hf1e_mode = str(os.environ.get("ASUKA_CASPT2_HF_1E_MODE", "dpt2")).strip().lower()
    if _hf1e_mode not in {"dpt2", "molcas_dk"}:
        _hf1e_mode = "dpt2"
    D_hf_1e_selected = D_hf_1e_molcas_dk if _hf1e_mode == "molcas_dk" else D_hf_1e
    D_ci_1e = np.zeros_like(D_ref_1e)
    D_orb_1e = np.zeros_like(D_ref_1e)
    D_tot_1e = np.asarray(D_ref_1e, dtype=np.float64)

    # Experimental CI-response Pulay term (off by default).
    # This is useful for parity diagnostics but currently over-corrects on LiH.
    _zci_mode = str(os.environ.get("ASUKA_CASPT2_ZCI_MODE", "full")).strip().lower()
    if _zci_mode not in {"full", "parallel", "tangent"}:
        _zci_mode = "full"

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    _pt2_2e_scale = _env_float("ASUKA_PT2_2E_SCALE", 1.0)
    _pt2_h1_scale = _env_float("ASUKA_PT2_H1_SCALE", 1.0)
    _pt2_pulay_scale = _env_float("ASUKA_PT2_PULAY_SCALE", 1.0)
    _pt2_lci_scale = _env_float("ASUKA_PT2_LCI_SCALE", 1.0)
    _pt2_lorb_scale = _env_float("ASUKA_PT2_LORB_SCALE", 1.0)
    _pt2_lci_df2e_scale = _env_float("ASUKA_PT2_LCI_DF2E_SCALE", 1.0)
    _pt2_lorb_df2e_scale = _env_float("ASUKA_PT2_LORB_DF2E_SCALE", 1.0)
    _pt2_lci_h1_scale = _env_float("ASUKA_PT2_LCI_H1_SCALE", 1.0)
    _pt2_lorb_h1_scale = _env_float("ASUKA_PT2_LORB_H1_SCALE", 1.0)
    _pt2_lci_pulay_scale = _env_float("ASUKA_PT2_LCI_PULAY_SCALE", 1.0)
    _pt2_lorb_pulay_scale = _env_float("ASUKA_PT2_LORB_PULAY_SCALE", 1.0)

    _hf_2e_scale = float(_pt2_2e_scale)
    _hf_1e_scale = float(_pt2_h1_scale)
    _hf_pulay_scale = float(_pt2_pulay_scale)
    _ci_2e_scale = float(_pt2_lci_scale) * float(_pt2_lci_df2e_scale)
    _ci_1e_scale = float(_pt2_lci_scale) * float(_pt2_lci_h1_scale)
    _ci_pulay_scale_eff = float(_pt2_lci_scale) * float(_pt2_lci_pulay_scale)
    _orb_2e_scale = float(_pt2_lorb_scale) * float(_pt2_lorb_df2e_scale)
    _orb_1e_scale = float(_pt2_lorb_scale) * float(_pt2_lorb_h1_scale)
    _orb_pulay_scale = float(_pt2_lorb_scale) * float(_pt2_lorb_pulay_scale)
    # OpenMolcas out_pt2.F90 consumes WLag read from PT2_Lag directly.
    # Keep this separate toggle so strict parity can disable ASUKA's
    # additional orbital-response WLag add-on.
    # OpenMolcas consumes the PT2 WLag written by PT2_Lag; adding an extra
    # orbital-response Pulay/WLag term here over-corrects live FD checks.
    _lorb_wlag_mode = str(os.environ.get("ASUKA_CASPT2_LORB_WLAG_MODE", "off")).strip().lower()
    if _lorb_wlag_mode not in {"on", "off"}:
        _lorb_wlag_mode = "on"
    _include_lorb_wlag = bool(_lorb_wlag_mode == "on")

    bar_L_total = bar_L_total + float(_hf_2e_scale) * bar_L_hf
    D_tot_1e = D_tot_1e + float(_hf_1e_scale) * D_hf_1e_selected
    W_total = W_total + float(_hf_pulay_scale) * W_hf
    _include_ci_pulay = False

    # --- Z-vector response corrections ---
    bar_L_ci_dm1 = None
    bar_L_ci_dm2 = None
    bar_L_ci_par = None
    bar_L_ci_tan = None
    D_ci_1e_par = None
    D_ci_1e_tan = None
    D_ci_1e_dm1 = None
    D_ci_1e_dm2 = None
    bar_L_orb_mean = None
    bar_L_orb_dm2 = None
    orb_case_payload: dict[str, dict[str, np.ndarray]] = {}

    # Orbital response: Lorb modifies both density and bar_L
    bar_L_lorb, D_lorb = _build_bar_L_lorb_df(
        B_ao_x, C_x,
        np.asarray(z_orb, dtype=np.float64),
        dm1_sa, dm2_sa,
        ncore=int(ncore), ncas=int(ncas),
        xp=xp,
    )
    bar_L_orb = xp.asarray(bar_L_lorb, dtype=xp.float64)
    if bool(collect_breakdown):
        _z_orb_np = np.asarray(z_orb, dtype=np.float64)
        _nocc_dbg = int(ncore + ncas)
        _nmo_dbg = int(C.shape[1])
        _nvir_dbg = int(_nmo_dbg - _nocc_dbg)
        _mask_a = np.zeros_like(_z_orb_np)
        _mask_c = np.zeros_like(_z_orb_np)
        _mask_d = np.zeros_like(_z_orb_np)
        if int(ncore) > 0 and int(ncas) > 0:
            _mask_a[:ncore, ncore:_nocc_dbg] = 1.0
            _mask_a[ncore:_nocc_dbg, :ncore] = 1.0
        if int(ncas) > 0 and int(_nvir_dbg) > 0:
            _mask_c[ncore:_nocc_dbg, _nocc_dbg:_nmo_dbg] = 1.0
            _mask_c[_nocc_dbg:_nmo_dbg, ncore:_nocc_dbg] = 1.0
        if int(ncore) > 0 and int(_nvir_dbg) > 0:
            _mask_d[:ncore, _nocc_dbg:_nmo_dbg] = 1.0
            _mask_d[_nocc_dbg:_nmo_dbg, :ncore] = 1.0
        for _lbl, _mask in (("A", _mask_a), ("C", _mask_c), ("D", _mask_d)):
            _z_case = np.asarray(_z_orb_np * _mask, dtype=np.float64)
            _bar_case, _d_case = _build_bar_L_lorb_df(
                B_ao_x,
                C_x,
                _z_case,
                dm1_sa,
                dm2_sa,
                ncore=int(ncore),
                ncas=int(ncas),
                xp=xp,
            )
            orb_case_payload[_lbl] = {
                "z_orb": _z_case,
                "bar_L": np.asarray(_asnumpy_f64(_bar_case), dtype=np.float64),
                "D_1e": np.asarray(_asnumpy_f64(_d_case), dtype=np.float64),
                "W": np.zeros((nao, nao), dtype=np.float64),
                "W_parts": {},
            }
    if bool(collect_breakdown):
        bar_L_lorb_mean, _D_lorb_mean = _build_bar_L_lorb_df(
            B_ao_x, C_x,
            np.asarray(z_orb, dtype=np.float64),
            dm1_sa,
            np.zeros_like(dm2_sa),
            ncore=int(ncore), ncas=int(ncas),
            xp=xp,
        )
        bar_L_orb_mean = xp.asarray(bar_L_lorb_mean, dtype=xp.float64)
        bar_L_orb_dm2 = xp.asarray(bar_L_orb - bar_L_orb_mean, dtype=xp.float64)
    bar_L_total = bar_L_total + float(_orb_2e_scale) * bar_L_orb
    D_orb_1e = _asnumpy_f64(D_lorb)
    D_tot_1e += float(_orb_1e_scale) * D_orb_1e

    # CI response: transition RDMs from z_ci
    _ci_response_result = None
    dm1_lci = None
    dm2_lci = None
    dm1_lci_par = None
    dm2_lci_par = None
    dm1_lci_tan = None
    dm2_lci_tan = None
    _ci_dm1_sym_mode = "sum"
    _ci_dm2_sym_mode = "sum"
    _ci_tdm_mode = str(ci_trans_rdm_mode).strip().lower()
    if _ci_tdm_mode not in {"solver", "molcas"}:
        _ci_tdm_mode = "solver"
    _ci_z_map_mode = "none"
    _ci_z_map_applied = False
    _ci_z_map_perm = np.zeros((0,), dtype=np.int64)
    _ci_z_map_signs = np.zeros((0,), dtype=np.float64)
    if z_ci is not None:
        _ci_response_result = _build_ci_response(
            z_ci, ci_list, weights, nroots,
            ncas, nelecas, twos, ncore, nmo,
            B_ao_x, C_x, xp,
            zci_mode=_zci_mode,
            ci_trans_rdm_mode=ci_trans_rdm_mode,
            collect_breakdown=collect_breakdown,
        )
        bar_L_ci = _ci_response_result["bar_L_ci"]
        D_ci_1e = _ci_response_result["D_ci_1e"]
        dm1_lci = _ci_response_result["dm1_lci"]
        dm2_lci = _ci_response_result["dm2_lci"]
        dm1_lci_par = _ci_response_result["dm1_lci_par"]
        dm2_lci_par = _ci_response_result["dm2_lci_par"]
        dm1_lci_tan = _ci_response_result["dm1_lci_tan"]
        dm2_lci_tan = _ci_response_result["dm2_lci_tan"]
        bar_L_ci_dm1 = _ci_response_result["bar_L_ci_dm1"]
        bar_L_ci_dm2 = _ci_response_result["bar_L_ci_dm2"]
        D_ci_1e_dm1 = _ci_response_result["D_ci_1e_dm1"]
        D_ci_1e_dm2 = _ci_response_result["D_ci_1e_dm2"]
        bar_L_ci_par = _ci_response_result["bar_L_ci_par"]
        bar_L_ci_tan = _ci_response_result["bar_L_ci_tan"]
        D_ci_1e_par = _ci_response_result["D_ci_1e_par"]
        D_ci_1e_tan = _ci_response_result["D_ci_1e_tan"]
        _ci_dm1_sym_mode = _ci_response_result["ci_dm1_sym_mode"]
        _ci_dm2_sym_mode = _ci_response_result["ci_dm2_sym_mode"]
        _ci_tdm_mode = _ci_response_result["ci_tdm_mode"]
        _ci_z_map_mode = _ci_response_result["ci_z_map_mode"]
        _ci_z_map_applied = _ci_response_result["ci_z_map_applied"]
        _ci_z_map_perm = _ci_response_result["ci_z_map_perm"]
        _ci_z_map_signs = _ci_response_result["ci_z_map_signs"]
        bar_L_total = bar_L_total + float(_ci_2e_scale) * bar_L_ci
        D_tot_1e += float(_ci_1e_scale) * D_ci_1e

    # --- Response Pulay: orbital Z-vector contribution to W ---
    # W_lorb = dme0 = 0.5*(gfock_lorb + gfock_lorb^T) in AO basis
    # This captures how the energy-weighted density changes under orbital rotation.
    from asuka.mcscf.nuc_grad_df import _build_dme0_lorb_response
    nmo = C.shape[1]
    nocc_full = ncore + ncas
    ppaa = eri_mo[:, :, ncore:nocc_full, ncore:nocc_full] if eri_mo is not None else None
    papa = eri_mo[:, ncore:nocc_full, :, ncore:nocc_full] if eri_mo is not None else None
    if ppaa is not None and papa is not None and bool(_include_lorb_wlag):
        W_lorb_ret = _build_dme0_lorb_response(
            B_ao, h_ao, C,
            np.asarray(z_orb, dtype=np.float64),
            dm1_sa, dm2_sa,
            ppaa=ppaa, papa=papa,
            ncore=int(ncore), ncas=int(ncas),
            return_parts=bool(collect_breakdown),
        )
        if bool(collect_breakdown) and isinstance(W_lorb_ret, tuple):
            W_lorb, _w_parts = W_lorb_ret
            W_orb_parts = {
                str(k): np.asarray(_asnumpy_f64(v), dtype=np.float64)
                for k, v in dict(_w_parts).items()
            }
        else:
            W_lorb = W_lorb_ret
        W_orb = _asnumpy_f64(W_lorb)
        if bool(collect_breakdown) and orb_case_payload:
            for _lbl, _payload in orb_case_payload.items():
                _z_case = np.asarray(_payload.get("z_orb", np.zeros_like(np.asarray(z_orb, dtype=np.float64))), dtype=np.float64)
                _w_case_ret = _build_dme0_lorb_response(
                    B_ao,
                    h_ao,
                    C,
                    _z_case,
                    dm1_sa,
                    dm2_sa,
                    ppaa=ppaa,
                    papa=papa,
                    ncore=int(ncore),
                    ncas=int(ncas),
                    return_parts=True,
                )
                if isinstance(_w_case_ret, tuple):
                    _w_case, _w_case_parts = _w_case_ret
                    _payload["W_parts"] = {
                        str(k): np.asarray(_asnumpy_f64(v), dtype=np.float64)
                        for k, v in dict(_w_case_parts).items()
                    }
                else:
                    _w_case = _w_case_ret
                    _payload["W_parts"] = {}
                _payload["W"] = np.asarray(_asnumpy_f64(_w_case), dtype=np.float64)
        W_total = W_total + float(_orb_pulay_scale) * W_orb

    # --- Contract with integral derivatives ---
    def _contract_df_component_dense_exact_fd(bar_l_ao: Any) -> np.ndarray:
        """Dense-integral fallback for DF-derivative contraction (aux_basis is unavailable).

        This path finite-differences the dense AO ERIs, rebuilds exact DF-like
        factors at each displacement, aligns their factor gauge to the reference,
        and contracts against ``bar_L``.
        """
        from asuka.hf.dense_eri import build_ao_eri_dense  # noqa: PLC0415

        bar_l_np = np.asarray(_asnumpy_f64(bar_l_ao), dtype=np.float64)
        b_ref = np.asarray(_asnumpy_f64(B_ao_x), dtype=np.float64)
        if b_ref.ndim != 3:
            raise ValueError("dense fallback requires B_ao with shape (nao,nao,naux)")
        nao, nao1, naux = map(int, b_ref.shape)
        if nao != nao1:
            raise ValueError(f"invalid B_ao shape for dense fallback: {b_ref.shape}")
        if bar_l_np.shape != (naux, nao, nao):
            raise ValueError(
                "bar_L shape mismatch in dense fallback: "
                f"expected {(naux, nao, nao)}, got {bar_l_np.shape}"
            )

        fd_step = 1.0e-4
        eig_tol = 1.0e-12
        dense_backend = str(df_backend).strip().lower()
        if dense_backend not in {"cpu", "cuda"}:
            dense_backend = "cpu"

        g = np.zeros((natm, 3), dtype=np.float64)
        for ia in range(natm):
            for k in range(3):
                disp = np.zeros((3,), dtype=np.float64)
                disp[k] = float(fd_step)
                ao_p = _basis_cart_shifted_by_atom(
                    ao_basis,
                    shell_atom=shell_atom,
                    atom=int(ia),
                    disp_bohr_xyz=disp,
                )
                ao_m = _basis_cart_shifted_by_atom(
                    ao_basis,
                    shell_atom=shell_atom,
                    atom=int(ia),
                    disp_bohr_xyz=-disp,
                )

                eri_p = build_ao_eri_dense(ao_p, backend=str(dense_backend), eps_ao=0.0).eri_mat
                eri_m = build_ao_eri_dense(ao_m, backend=str(dense_backend), eps_ao=0.0).eri_mat

                b_p = _build_exact_df_like_factors_from_ao_eri(_asnumpy_f64(eri_p), eig_tol=eig_tol)
                b_m = _build_exact_df_like_factors_from_ao_eri(_asnumpy_f64(eri_m), eig_tol=eig_tol)
                if b_p.shape != b_ref.shape or b_m.shape != b_ref.shape:
                    raise ValueError(
                        "dense exact-factor rank changed under displacement: "
                        f"ref={b_ref.shape} plus={b_p.shape} minus={b_m.shape}"
                    )

                b_p_aln = _align_df_like_factors_to_reference(b_ref, b_p)
                b_m_aln = _align_df_like_factors_to_reference(b_ref, b_m)

                e_p = _contract_bar_with_B_streamed(
                    bar_l_np,
                    b_p_aln,
                    backend="cpu",
                    aux_block_naux=max(1, int(naux // 8) or 1),
                )
                e_m = _contract_bar_with_B_streamed(
                    bar_l_np,
                    b_m_aln,
                    backend="cpu",
                    aux_block_naux=max(1, int(naux // 8) or 1),
                )
                g[ia, k] = float((e_p - e_m) / (2.0 * fd_step))
        return np.asarray(g, dtype=np.float64)

    def _contract_df_component(bar_l_ao: Any) -> np.ndarray:
        if aux_basis is None:
            return _contract_df_component_dense_exact_fd(bar_l_ao)
        try:
            v = compute_df_gradient_contributions_analytic_packed_bases(
                ao_basis, aux_basis,
                atom_coords_bohr=coords,
                B_ao=B_ao_x,
                bar_L_ao=xp.asarray(bar_l_ao, dtype=xp.float64),
                L_chol=getattr(scf_out, "df_L", None),
                backend=str(df_backend),
            )
        except (NotImplementedError, RuntimeError):
            v = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis, aux_basis,
                atom_coords_bohr=coords,
                bar_L_ao=bar_l_ao,
                backend=str(df_backend),
                delta_bohr=1e-4,
            )
        return np.asarray(_asnumpy_f64(v), dtype=np.float64)

    def _contract_h1_component(dm_ao: np.ndarray) -> np.ndarray:
        return np.asarray(
            contract_dhcore_cart(
                ao_basis, atom_coords_bohr=coords,
                atom_charges=charges,
                M=np.asarray(dm_ao, dtype=np.float64),
                shell_atom=shell_atom,
                contract_backend=str(int1e_contract_backend),
            ),
            dtype=np.float64,
        )

    def _contract_pulay_component(w_ao: np.ndarray) -> np.ndarray:
        return -1.0 * np.asarray(
            contract_dS_cart(
                ao_basis, atom_coords_bohr=coords,
                M=np.asarray(w_ao, dtype=np.float64),
                shell_atom=shell_atom,
                contract_backend=str(int1e_contract_backend),
            ),
            dtype=np.float64,
        )

    de_btamp_dense_raw = np.zeros((natm, 3), dtype=np.float64)
    de_btamp_dense_scaled = np.zeros((natm, 3), dtype=np.float64)

    de_df = _contract_df_component(bar_L_total) + de_btamp_dense_scaled
    de_h1 = _contract_h1_component(np.asarray(D_tot_1e, dtype=np.float64))

    # Nuclear repulsion gradient
    de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)

    # Pulay (overlap-derivative) term: -tr(W @ dS/dR)
    de_pulay = _contract_pulay_component(np.asarray(W_total, dtype=np.float64))

    grad = np.asarray(de_nuc + de_h1 + de_df + de_pulay, dtype=np.float64)

    breakdown: dict[str, Any] = {}
    if bool(collect_breakdown):
        _bkd_intermediates = {
            "bar_L_ref": bar_L_ref,
            "bar_L_hf": bar_L_hf,
            "bar_L_ci": bar_L_ci,
            "bar_L_orb": bar_L_orb,
            "bar_hf_cross_j_core": bar_hf_cross_j_core,
            "bar_hf_cross_j_act": bar_hf_cross_j_act,
            "bar_hf_cross_k_core": bar_hf_cross_k_core,
            "bar_hf_cross_k_act": bar_hf_cross_k_act,
            "bar_hf_self_j": bar_hf_self_j,
            "bar_hf_self_k": bar_hf_self_k,
            "bar_L_ci_dm1": bar_L_ci_dm1,
            "bar_L_ci_dm2": bar_L_ci_dm2,
            "bar_L_ci_par": bar_L_ci_par,
            "bar_L_ci_tan": bar_L_ci_tan,
            "bar_L_orb_mean": bar_L_orb_mean,
            "bar_L_orb_dm2": bar_L_orb_dm2,
            "D_ref_1e": D_ref_1e,
            "D_hf_1e": D_hf_1e,
            "D_hf_1e_selected": D_hf_1e_selected,
            "D_hf_1e_dpt2c_sym_quarter": D_hf_1e_dpt2c_sym_quarter,
            "D_hf_1e_molcas_dk": D_hf_1e_molcas_dk,
            "D_ci_1e": D_ci_1e,
            "D_ci_1e_dm1": D_ci_1e_dm1,
            "D_ci_1e_dm2": D_ci_1e_dm2,
            "D_ci_1e_par": D_ci_1e_par,
            "D_ci_1e_tan": D_ci_1e_tan,
            "D_orb_1e": D_orb_1e,
            "W_ref": W_ref,
            "W_hf": W_hf,
            "W_ci": W_ci,
            "W_orb": W_orb,
            "W_orb_parts": W_orb_parts,
            "W_total": W_total,
            "hf_2e_scale": _hf_2e_scale,
            "hf_1e_scale": _hf_1e_scale,
            "hf_pulay_scale": _hf_pulay_scale,
            "ci_2e_scale": _ci_2e_scale,
            "ci_1e_scale": _ci_1e_scale,
            "ci_pulay_scale_eff": _ci_pulay_scale_eff,
            "orb_2e_scale": _orb_2e_scale,
            "orb_1e_scale": _orb_1e_scale,
            "orb_pulay_scale": _orb_pulay_scale,
            "hf2e_ref_mode": _hf2e_ref_mode,
            "hf2e_build_mode": _hf2e_build_mode,
            "hf2e_include_self": _hf2e_include_self,
            "hf1e_mode": _hf1e_mode,
            "zci_mode": _zci_mode,
            "ci_tdm_mode": _ci_tdm_mode,
            "ci_dm1_sym_mode": _ci_dm1_sym_mode,
            "ci_dm2_sym_mode": _ci_dm2_sym_mode,
            "lorb_wlag_mode": _lorb_wlag_mode,
            "include_lorb_wlag": _include_lorb_wlag,
            "de_btamp_dense_raw": de_btamp_dense_raw,
            "de_btamp_dense_scaled": de_btamp_dense_scaled,
            "de_nuc": de_nuc,
            "z_ci": z_ci,
            "orb_case_payload": orb_case_payload,
            "dm1_sa": dm1_sa,
            "dm2_sa": dm2_sa,
            "dm1_lci": dm1_lci if dm1_lci is not None else np.zeros((int(ncas), int(ncas)), dtype=np.float64),
            "dm2_lci": dm2_lci if dm2_lci is not None else np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64),
            "dm1_lci_par": dm1_lci_par,
            "dm2_lci_par": dm2_lci_par,
            "dm1_lci_tan": dm1_lci_tan,
            "dm2_lci_tan": dm2_lci_tan,
            "ncas": ncas,
            "_asnumpy_f64": _asnumpy_f64,
            "ci_response_result": _ci_response_result if _ci_response_result is not None else {},
        }
        breakdown = _collect_gradient_breakdown(
            contract_df=_contract_df_component,
            contract_h1=_contract_h1_component,
            contract_pulay=_contract_pulay_component,
            natm=natm,
            intermediates=_bkd_intermediates,
        )

    if verbose >= 1:
        print(f"[CASPT2 grad] de_nuc: {de_nuc.ravel()[:3]}")
        print(f"[CASPT2 grad] de_h1:  {de_h1.ravel()[:3]}")
        print(f"[CASPT2 grad] de_df:  {de_df.ravel()[:3]}")
        print(f"[CASPT2 grad] de_pul: {de_pulay.ravel()[:3]}")
        print(f"[CASPT2 grad] total:  {grad.ravel()[:3]}")

    comp = {
        "de_nuc": np.asarray(de_nuc, dtype=np.float64),
        "de_h1": np.asarray(de_h1, dtype=np.float64),
        "de_df": np.asarray(de_df, dtype=np.float64),
        "de_pulay": np.asarray(de_pulay, dtype=np.float64),
        "breakdown": breakdown,
    }
    _restore_pool()
    if bool(return_components):
        return grad, comp
    return grad
