"""SS-CASPT2 analytic gradient driver."""
from __future__ import annotations
from typing import Any
import numpy as np
import os
from asuka.caspt2.gradient.debug_utils import (
    _asnumpy_f64,
    _apply_debug_zorb_block_signs,
    _build_dlao_candidate_ao,
    _parse_ci_basis_map_from_env,
    _apply_ci_basis_map,
    _infer_ci_basis_map_from_dump,
    _infer_ci_basis_map_from_resp,
    _resolve_response_dpt2_mode,
)
from asuka.caspt2.gradient.zvector import _solve_zvector
from asuka.caspt2.gradient.assembly import _assemble_gradient


def caspt2_ss_gradient_native(
    scf_out: Any,
    casscf: Any,
    *,
    iroot: int = 0,
    pt2_backend: str = "ic",
    pt2_tol: float = 1e-8,
    pt2_maxiter: int = 200,
    pt2_threshold: float = 1e-10,
    pt2_threshold_s: float = 1e-8,
    df_backend: str = "cpu",
    int1e_contract_backend: str = "cpu",
    parity_profile: str = "default",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    verbose: int = 0,
) -> Any:
    """Compute SS-CASPT2 analytic nuclear gradient.

    Parameters
    ----------
    scf_out : Any
        ASUKA SCF result.
    casscf : Any
        ASUKA CASSCF result.
    iroot : int
        Target state index (must be 0 for SS).
    pt2_backend : str
        CASPT2 backend selector. Supports ``"sst"``, ``"sst-ic"``, and
        ``"sst-full"`` to source IC-equivalent per-case amplitudes from the SST
        backend for the SS analytic gradient. Other values use the standard IC
        CASPT2 energy solver (CPU path) to obtain amplitudes.
    pt2_tol : float
        PT2 amplitude-solver convergence tolerance forwarded to the CASPT2
        energy driver (both CPU and CUDA backends).
    pt2_maxiter : int
        Maximum number of PT2 solver iterations (PCG).
    pt2_threshold : float
        Diagonal-norm threshold (Molcas THRSHN) for S-metric pre-scaling.
    pt2_threshold_s : float
        Scaled-S eigenvalue threshold (Molcas THRSHS) for linear-dependence
        removal.
    df_backend : str
        Backend for DF derivative contraction.
    int1e_contract_backend : str
        Backend for 1e derivative contractions.
    parity_profile : str
        Gradient convention profile. Accepted for dispatcher compatibility.
    imag_shift, real_shift : float
        Level shift parameters.
    z_tol : float
        Z-vector convergence tolerance.
    z_maxiter : int
        Z-vector max iterations.
    verbose : int
        Verbosity level.

    Returns
    -------
    CASPT2GradResult
    """
    from asuka.caspt2.result import CASPT2GradResult

    # Dispatcher validates the profile and applies profile-specific env overrides.
    _parity_u = str(parity_profile).strip().lower()
    pt2_backend_u = str(pt2_backend).strip().lower()

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    # --- Extract orbital dimensions ---
    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    C = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    ci_raw = getattr(casscf, "ci")
    nao, nmo = C.shape
    nvirt = nmo - ncore - ncas

    if verbose >= 1:
        print(f"[CASPT2 grad] ncore={ncore} ncas={ncas} nvirt={nvirt} nmo={nmo}")

    # --- Build AO ingredients ---
    B_ao = _asnumpy_f64(getattr(scf_out, "df_B"))
    h_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))
    S_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "S"))
    e_nuc = float(mol.energy_nuc())

    # --- Build RDMs ---
    from asuka.cuguga.drt import build_drt
    from asuka.rdm.rdm123 import _make_rdm123_pyscf, _reorder_dm123_molcas

    twos = int(getattr(mol, "spin", 0))
    nelec_total = int(nelecas) if isinstance(nelecas, (int, np.integer)) else int(sum(nelecas))
    drt = build_drt(norb=ncas, nelec=nelec_total, twos_target=twos)
    ci = _asnumpy_f64(ci_raw).ravel()
    ncsf = ci.size
    dm1, dm2, dm3 = _make_rdm123_pyscf(drt, ci, reorder=False)
    dm1, dm2, dm3 = _reorder_dm123_molcas(dm1, dm2, dm3, inplace=True)
    ci_raw_work: Any = ci_raw

    # --- Build CASPT2 Fock / ERI ingredients ---
    from asuka.caspt2.fock import build_caspt2_fock
    from asuka.caspt2.fock_df import build_caspt2_fock_ao
    from asuka.caspt2.f3 import CASPT2CIContext
    from asuka.caspt2.superindex import build_superindex

    # Build MO ERIs (DF default; optional dense ERI for parity/debug).
    #
    # This is gated by env to keep the default fast path unchanged.
    # Dense ERIs are only practical for very small AO bases.
    eri_mode = str(os.environ.get("ASUKA_SS_ERI_MODE", "df")).strip().lower()

    def _build_mo_eri(cmo: np.ndarray) -> np.ndarray:
        cmo = np.asarray(cmo, dtype=np.float64)
        if eri_mode in {"dense", "full", "eri_dense"}:
            from asuka.hf.dense_eri import build_ao_eri_dense  # noqa: PLC0415

            ao_basis = getattr(scf_out, "ao_basis", None)
            if ao_basis is None:
                raise TypeError("scf_out must have .ao_basis to build dense ERIs")

            dense_backend = str(os.environ.get("ASUKA_SS_DENSE_ERI_BACKEND", "cuda")).strip().lower()
            eri_res = build_ao_eri_dense(ao_basis, backend=dense_backend, eps_ao=0.0)
            eri_mat = _asnumpy_f64(getattr(eri_res, "eri_mat"))
            eri_ao = np.asarray(eri_mat, dtype=np.float64).reshape(nao, nao, nao, nao)
            return np.asarray(
                np.einsum("mp,nq,lr,ks,mnlk->pqrs", cmo, cmo, cmo, cmo, eri_ao, optimize=True),
                dtype=np.float64,
            )
        # DF: eri_mo[pq,rs] = sum_P B_mo[pq,P] * B_mo[rs,P]
        b_mo = np.einsum("mi,mnP,nj->ijP", cmo, B_ao, cmo, optimize=True)
        naux = b_mo.shape[2]
        b2 = b_mo.reshape(nmo * nmo, naux)
        return np.asarray((b2 @ b2.T).reshape(nmo, nmo, nmo, nmo), dtype=np.float64)

    def _env_bool(name: str, default: bool) -> bool:
        raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
        return raw not in {"0", "false", "off", "no"}

    semican_meta: dict[str, Any] = {
        "active_semicanon_enabled": False,
        "active_semicanon_applied": False,
        "active_semicanon_offdiag_before": 0.0,
        "active_semicanon_offdiag_after": 0.0,
        "active_semicanon_ci_norm_before": float(np.linalg.norm(ci)),
        "active_semicanon_ci_norm_after": float(np.linalg.norm(ci)),
        "core_virt_semicanon_enabled": False,
        "core_virt_semicanon_applied": False,
        "core_semicanon_offdiag_before": 0.0,
        "core_semicanon_offdiag_after": 0.0,
        "virt_semicanon_offdiag_before": 0.0,
        "virt_semicanon_offdiag_after": 0.0,
    }
    _active_semicanon = _env_bool(
        "ASUKA_CASPT2_ACTIVE_SEMICANON",
        default=False,
    )
    semican_meta["active_semicanon_enabled"] = bool(_active_semicanon)
    if bool(_active_semicanon) and int(ncas) > 1:
        from asuka.mrpt2.semicanonical import molcas_diafck_eigh  # noqa: PLC0415
        from asuka.soc.rassi import transform_csf_ci_for_orbital_transform  # noqa: PLC0415

        nocc = int(ncore + ncas)
        fock_probe = build_caspt2_fock_ao(h_ao, B_ao, C, dm1, ncore, ncas, nvirt, e_nuc=e_nuc)
        faa = np.asarray(fock_probe.fifa[ncore:nocc, ncore:nocc], dtype=np.float64)
        faa = 0.5 * (faa + faa.T)
        off_before = np.asarray(faa - np.diag(np.diag(faa)), dtype=np.float64)
        eps_act, u_act = molcas_diafck_eigh(faa)
        faa_sc = np.asarray(u_act.T @ faa @ u_act, dtype=np.float64)
        off_after = np.asarray(faa_sc - np.diag(np.diag(faa_sc)), dtype=np.float64)

        semican_meta["active_semicanon_epsa_before"] = np.asarray(np.diag(faa), dtype=np.float64)
        semican_meta["active_semicanon_epsa_after"] = np.asarray(eps_act, dtype=np.float64)
        semican_meta["active_semicanon_u_aa"] = np.asarray(u_act, dtype=np.float64)
        semican_meta["active_semicanon_faa_before"] = np.asarray(faa, dtype=np.float64)
        semican_meta["active_semicanon_faa_after"] = np.asarray(faa_sc, dtype=np.float64)
        semican_meta["active_semicanon_offdiag_before"] = float(np.max(np.abs(off_before)))
        semican_meta["active_semicanon_offdiag_after"] = float(np.max(np.abs(off_after)))

        u_full = np.eye(nmo, dtype=np.float64)
        u_full[ncore:nocc, ncore:nocc] = np.asarray(u_act, dtype=np.float64)
        C = np.asarray(C @ u_full, dtype=np.float64, order="C")

        # `transform_csf_ci_for_orbital_transform` follows RASSI-style CI
        # transform conventions where the inverse one-particle map is applied.
        tra = np.asarray(u_act.T, dtype=np.float64, order="C")
        try:
            _ci_trf_tol = float(os.environ.get("ASUKA_CASPT2_ACTIVE_SEMICANON_CI_TOL", "1.0e-14"))
        except Exception:
            _ci_trf_tol = 1.0e-14
        ci_before = np.asarray(ci, dtype=np.float64)
        ci = transform_csf_ci_for_orbital_transform(
            drt,
            ci_before,
            tra,
            tol=float(max(0.0, _ci_trf_tol)),
        )
        ci = np.asarray(ci, dtype=np.float64).ravel()
        ci_norm_after = float(np.linalg.norm(ci))
        if ci_norm_after > 0.0:
            ci = np.asarray(ci / ci_norm_after, dtype=np.float64)

        semican_meta["active_semicanon_ci_norm_before"] = float(np.linalg.norm(ci_before))
        semican_meta["active_semicanon_ci_norm_after"] = float(np.linalg.norm(ci))
        semican_meta["active_semicanon_ci_delta_norm"] = float(np.linalg.norm(ci - ci_before))
        semican_meta["active_semicanon_applied"] = True

        dm1, dm2, dm3 = _make_rdm123_pyscf(drt, ci, reorder=False)
        dm1, dm2, dm3 = _reorder_dm123_molcas(dm1, dm2, dm3, inplace=True)
        ci_raw_work = np.asarray(ci, dtype=np.float64, order="C")

    _core_virt_semicanon = _env_bool(
        "ASUKA_CASPT2_CORE_VIRT_SEMICANON",
        default=False,
    )
    semican_meta["core_virt_semicanon_enabled"] = bool(_core_virt_semicanon)
    if bool(_core_virt_semicanon) and (int(ncore) > 1 or int(nvirt) > 1):
        from asuka.mrpt2.semicanonical import molcas_diafck_eigh  # noqa: PLC0415

        nocc = int(ncore + ncas)
        fock_probe_cv = build_caspt2_fock_ao(h_ao, B_ao, C, dm1, ncore, ncas, nvirt, e_nuc=e_nuc)
        fifa_cv = np.asarray(fock_probe_cv.fifa, dtype=np.float64)
        u_full_cv = np.eye(int(nmo), dtype=np.float64)

        if int(ncore) > 1:
            fcc = np.asarray(0.5 * (fifa_cv[:ncore, :ncore] + fifa_cv[:ncore, :ncore].T), dtype=np.float64)
            fcc_off_before = np.asarray(fcc - np.diag(np.diag(fcc)), dtype=np.float64)
            eps_core, u_core = molcas_diafck_eigh(fcc)
            fcc_sc = np.asarray(u_core.T @ fcc @ u_core, dtype=np.float64)
            fcc_off_after = np.asarray(fcc_sc - np.diag(np.diag(fcc_sc)), dtype=np.float64)
            u_full_cv[:ncore, :ncore] = np.asarray(u_core, dtype=np.float64)
            semican_meta["core_semicanon_eps_before"] = np.asarray(np.diag(fcc), dtype=np.float64)
            semican_meta["core_semicanon_eps_after"] = np.asarray(eps_core, dtype=np.float64)
            semican_meta["core_semicanon_u"] = np.asarray(u_core, dtype=np.float64)
            semican_meta["core_semicanon_offdiag_before"] = float(np.max(np.abs(fcc_off_before)))
            semican_meta["core_semicanon_offdiag_after"] = float(np.max(np.abs(fcc_off_after)))

        if int(nvirt) > 1:
            fvv = np.asarray(0.5 * (fifa_cv[nocc:, nocc:] + fifa_cv[nocc:, nocc:].T), dtype=np.float64)
            fvv_off_before = np.asarray(fvv - np.diag(np.diag(fvv)), dtype=np.float64)
            eps_virt, u_virt = molcas_diafck_eigh(fvv)
            fvv_sc = np.asarray(u_virt.T @ fvv @ u_virt, dtype=np.float64)
            fvv_off_after = np.asarray(fvv_sc - np.diag(np.diag(fvv_sc)), dtype=np.float64)
            u_full_cv[nocc:, nocc:] = np.asarray(u_virt, dtype=np.float64)
            semican_meta["virt_semicanon_eps_before"] = np.asarray(np.diag(fvv), dtype=np.float64)
            semican_meta["virt_semicanon_eps_after"] = np.asarray(eps_virt, dtype=np.float64)
            semican_meta["virt_semicanon_u"] = np.asarray(u_virt, dtype=np.float64)
            semican_meta["virt_semicanon_offdiag_before"] = float(np.max(np.abs(fvv_off_before)))
            semican_meta["virt_semicanon_offdiag_after"] = float(np.max(np.abs(fvv_off_after)))

        C = np.asarray(C @ u_full_cv, dtype=np.float64, order="C")
        semican_meta["core_virt_semicanon_u_full"] = np.asarray(u_full_cv, dtype=np.float64)
        semican_meta["core_virt_semicanon_applied"] = True

    use_cuda_df = pt2_backend_u in {"cuda", "df", "df-cuda", "gpu"}
    df_blocks = None
    if bool(use_cuda_df):
        # DF blocks for CUDA PT2 backend + DF-native PT2_Lag 2e path.
        # Includes l_ii and l_ab which are required for the DF OLagNS2/VVVO path.
        from asuka.mrpt2.df_pair_block import DFPairBlock, build_df_pair_blocks_from_df_B  # noqa: PLC0415
        from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks  # noqa: PLC0415

        B_full = np.asarray(B_ao, dtype=np.float64)
        if B_full.ndim == 2:
            from asuka.integrals.df_packed_s2 import unpack_Qp_to_mnQ  # noqa: PLC0415

            B_full = np.asarray(unpack_Qp_to_mnQ(B_full, nao=int(nao)), dtype=np.float64)
            B_ao = B_full

        C_core = np.asarray(C[:, :ncore], dtype=np.float64, order="C")
        C_act = np.asarray(C[:, ncore : ncore + ncas], dtype=np.float64, order="C")
        C_virt = np.asarray(C[:, ncore + ncas :], dtype=np.float64, order="C")
        naux = int(B_full.shape[2])

        def _empty_block(nx: int, ny: int) -> DFPairBlock:
            return DFPairBlock(
                nx=int(nx),
                ny=int(ny),
                l_full=np.zeros((int(nx) * int(ny), naux), dtype=np.float64),
                pair_norm=None,
            )

        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        labels: list[str] = []
        if ncore > 0:
            pairs.append((C_core, C_core))
            labels.append("ii")
        if ncore > 0 and ncas > 0:
            pairs.append((C_core, C_act))
            labels.append("it")
        if ncore > 0 and nvirt > 0:
            pairs.append((C_core, C_virt))
            labels.append("ia")
        if nvirt > 0 and ncas > 0:
            pairs.append((C_virt, C_act))
            labels.append("at")
        if ncas > 0:
            pairs.append((C_act, C_act))
            labels.append("tu")
        if nvirt > 0:
            pairs.append((C_virt, C_virt))
            labels.append("ab")

        if pairs:
            try:
                _df_mem = float(os.environ.get("ASUKA_CASPT2_DF_PAIRBLOCK_MAX_MEMORY_MB", "512"))
            except Exception:
                _df_mem = 512.0
            built = build_df_pair_blocks_from_df_B(
                B_full,
                pairs,
                max_memory=int(max(1.0, _df_mem)),
                compute_pair_norm=False,
            )
            by_label = dict(zip(labels, built))
        else:
            by_label = {}

        df_blocks = CASPT2DFBlocks(
            l_it=by_label.get("it", _empty_block(ncore, ncas)),
            l_ia=by_label.get("ia", _empty_block(ncore, nvirt)),
            l_at=by_label.get("at", _empty_block(nvirt, ncas)),
            l_tu=by_label.get("tu", _empty_block(ncas, ncas)),
            l_ii=by_label.get("ii", None if ncore == 0 else _empty_block(ncore, ncore)),
            l_ab=by_label.get("ab", None if nvirt == 0 else _empty_block(nvirt, nvirt)),
        )

        eri_mo = None
        fock = build_caspt2_fock_ao(h_ao, B_ao, C, dm1, ncore, ncas, nvirt, e_nuc=e_nuc)
    else:
        eri_mo = _build_mo_eri(C)
        if eri_mode in {"dense", "full", "eri_dense"}:
            h_mo = np.asarray(C.T @ h_ao @ C, dtype=np.float64)
            fock = build_caspt2_fock(h_mo, eri_mo, dm1, ncore, ncas, nvirt, e_nuc=e_nuc)
        else:
            fock = build_caspt2_fock_ao(h_ao, B_ao, C, dm1, ncore, ncas, nvirt, e_nuc=e_nuc)
    smap = build_superindex(ncore, ncas, nvirt)
    ci_ctx = CASPT2CIContext(drt=drt, ci_csf=ci)
    e_ref = float(getattr(casscf, "e_tot"))

    # --- Run CASPT2 energy to obtain per-case SR amplitudes ---
    #
    # For SST backends, solve the IC-equivalent 13-case amplitudes (cases 1-11
    # via the reduced IC system, cases 12-13 via the MP2-like H± sector) and
    # provide SB replay metadata so the Molcas-style translation machinery can
    # reuse the exact SR basis.
    pt2_breakdown: dict[str, Any] = {}
    if pt2_backend_u in {"sst", "sst-ic", "sst-full"}:
        from asuka.caspt2.sst.ic_amplitudes import sst_ic_equivalent_amplitudes_ss  # noqa: PLC0415

        e_pt2, amplitudes, sst_bd, sst_conv, sst_niter = sst_ic_equivalent_amplitudes_ss(
            smap=smap,
            fock=fock,
            eri_mo=np.asarray(eri_mo, dtype=np.float64),
            dm1=np.asarray(dm1, dtype=np.float64),
            dm2=np.asarray(dm2, dtype=np.float64),
            dm3=np.asarray(dm3, dtype=np.float64),
            ci_context=ci_ctx,
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            verbose=int(verbose),
        )
        pt2_breakdown = dict(sst_bd or {})
        pt2_breakdown.setdefault("sst_pcg_converged", bool(sst_conv))
        pt2_breakdown.setdefault("sst_pcg_niter", int(sst_niter))
        e_pt2 = float(e_pt2)
        e_tot = float(e_ref + e_pt2)
    else:
        from asuka.caspt2.energy import caspt2_energy_ss

        if bool(use_cuda_df):
            if df_blocks is None:
                raise RuntimeError("internal error: DF blocks missing for CUDA PT2 backend")
            pt2_result = caspt2_energy_ss(
                smap,
                fock,
                np.empty((0, 0, 0, 0), dtype=np.float64),
                dm1,
                dm2,
                dm3,
                e_ref,
                ci_context=ci_ctx,
                pt2_backend="cuda",
                df_blocks=df_blocks,
                imag_shift=imag_shift,
                real_shift=real_shift,
                tol=float(pt2_tol),
                maxiter=int(pt2_maxiter),
                threshold=float(pt2_threshold),
                threshold_s=float(pt2_threshold_s),
                store_sb_decomp=True,
                verbose=verbose,
            )
        else:
            pt2_result = caspt2_energy_ss(
                smap, fock, eri_mo, dm1, dm2, dm3, e_ref,
                ci_context=ci_ctx,
                pt2_backend="cpu",
                imag_shift=imag_shift,
                real_shift=real_shift,
                tol=float(pt2_tol),
                maxiter=int(pt2_maxiter),
                threshold=float(pt2_threshold),
                threshold_s=float(pt2_threshold_s),
                store_sb_decomp=True,
                verbose=verbose,
            )
        e_pt2 = float(pt2_result.e_pt2)
        e_tot = float(pt2_result.e_tot)
        amplitudes = list(getattr(pt2_result, "amplitudes"))
        pt2_breakdown = dict(getattr(pt2_result, "breakdown", {}) or {})

    if verbose >= 1:
        print(f"[CASPT2 grad] E_ref={e_ref:.10f} E_pt2={e_pt2:.10f} E_tot={e_tot:.10f}")

    # --- Reconstruct per-case data for Lagrangian computation ---
    from tools.molcas_caspt2_grad.translation.types import (
        OrbitalInfo, CASPT2Fock as TransFock, RDMs, CaseAmplitudes,
    )
    from tools.molcas_caspt2_grad.translation.utils import compute_case_dims

    orb = OrbitalInfo(nfro=0, nish=ncore, nash=ncas, nssh=nvirt)
    trans_fock = TransFock(
        fimo=_asnumpy_f64(fock.fimo),
        famo=_asnumpy_f64(fock.famo),
        fifa=_asnumpy_f64(fock.fifa),
        epsa=_asnumpy_f64(fock.epsa),
    )
    rdms = RDMs(dm1=dm1, dm2=dm2.reshape(ncas, ncas, ncas, ncas), dm3=dm3)

    # Rebuild S matrices and S/B decompositions to get transform matrices
    case_amps = _build_case_amps_from_asuka(
        smap, fock, dm1, dm2, dm3, ci_ctx,
        amplitudes, orb, eri_mo,
        pt2_breakdown={
            **dict(pt2_breakdown or {}),
            "mo_coeff": np.asarray(C, dtype=np.float64),
            "S_ao": np.asarray(S_ao, dtype=np.float64),
        },
        df_blocks=df_blocks,
    )

    # --- Build Lagrangians using translation pipeline ---
    lagrangians = _build_lagrangians(
        case_amps, smap, orb, trans_fock, rdms,
        ci, ncsf, eri_mo, C, S_ao, drt,
        verbose=verbose,
        B_ao=B_ao, h_ao=h_ao, e_nuc=e_nuc, twos=twos,
        df_blocks=df_blocks,
        collect_breakdown=True,
    )

    # --- Solve Z-vector ---
    z_orb, z_ci, z_meta = _solve_zvector(
        scf_out, casscf, lagrangians,
        ncore, ncas, nelecas, twos,
        B_ao, h_ao, C, ci_raw_work,
        z_tol=z_tol, z_maxiter=z_maxiter,
        project_ci_tangent=False,
        return_meta=True,
        dump_vectors=_parity_u == "molcas_ss_strict",
        verbose=verbose,
    )

    # --- Assemble gradient ---
    _ci_trans_mode = "molcas" if _parity_u == "molcas_ss_strict" else "solver"
    grad, grad_comp = _assemble_gradient(
        scf_out, casscf, lagrangians,
        z_orb, z_ci,
        ncore, ncas, nelecas, twos,
        B_ao, h_ao, S_ao, C,
        dm1, dm2, ci_raw_work,
        eri_mo=eri_mo,
        df_backend=df_backend,
        int1e_contract_backend=int1e_contract_backend,
        ci_trans_rdm_mode=_ci_trans_mode,
        return_components=True,
        collect_breakdown=True,
        verbose=verbose,
    )
    breakdown = dict(grad_comp.get("breakdown", {}) or {})
    # Expose Molcas-reference intermediate names for crosscheck tooling.
    breakdown.setdefault("clag_vec", np.asarray(lagrangians["clag"], dtype=np.float64))
    if "clag_raw" in lagrangians:
        breakdown.setdefault("clag_vec_raw", np.asarray(lagrangians["clag_raw"], dtype=np.float64))
    if "clag_eigder_add" in lagrangians:
        breakdown.setdefault("clag_vec_eigder_add", np.asarray(lagrangians["clag_eigder_add"], dtype=np.float64))
    if "clag_before_projection" in lagrangians:
        breakdown.setdefault("clag_vec_before_projection", np.asarray(lagrangians["clag_before_projection"], dtype=np.float64))
    if "clag_projection_parallel" in lagrangians:
        breakdown.setdefault("clag_vec_projection_parallel", np.asarray(lagrangians["clag_projection_parallel"], dtype=np.float64))
    if "clag_sigma_terms" in lagrangians:
        _ct_in = lagrangians.get("clag_sigma_terms") or {}
        if isinstance(_ct_in, dict):
            clag_terms: dict[str, np.ndarray] = {}
            for _k, _v in _ct_in.items():
                try:
                    clag_terms[str(_k)] = np.asarray(_v, dtype=np.float64)
                except Exception:
                    continue
            # Backward-compatible aliases expected by existing diagnostics.
            if "clag_sigma_dg_total" in clag_terms:
                clag_terms.setdefault("clag_dg_ci", np.asarray(clag_terms["clag_sigma_dg_total"], dtype=np.float64))
            if "clag_sigma_df_total" in clag_terms:
                clag_terms.setdefault("clag_df_ci_corr", np.asarray(clag_terms["clag_sigma_df_total"], dtype=np.float64))
            if "clag_raw" in lagrangians:
                clag_terms.setdefault("clag_sigma_raw", np.asarray(lagrangians["clag_raw"], dtype=np.float64))
            if "clag_before_projection" in lagrangians:
                clag_terms.setdefault("clag_before_projection", np.asarray(lagrangians["clag_before_projection"], dtype=np.float64))
            if clag_terms:
                breakdown.setdefault("clag_terms", clag_terms)
    if "clag_case_terms" in lagrangians:
        breakdown.setdefault("clag_case_terms", lagrangians["clag_case_terms"])
    if "clagdx_case_terms" in lagrangians:
        breakdown.setdefault("clagdx_case_terms", lagrangians["clagdx_case_terms"])
    if "clag_case_amp_terms" in lagrangians:
        breakdown.setdefault("clag_case_amp_terms", lagrangians["clag_case_amp_terms"])
    if "clag_deasum_post_terms" in lagrangians:
        breakdown.setdefault("clag_deasum_post_terms", lagrangians["clag_deasum_post_terms"])
    breakdown.setdefault("dpt2_bare", np.asarray(lagrangians.get("dpt2_bare", lagrangians["dpt2"]), dtype=np.float64))
    breakdown.setdefault("dpt2_full", np.asarray(lagrangians["dpt2"], dtype=np.float64))
    breakdown.setdefault("dpt2_for_ci_rhs", np.asarray(lagrangians["dpt2"], dtype=np.float64))
    breakdown.setdefault("dpt2c", np.asarray(lagrangians["dpt2c"], dtype=np.float64))
    if "dpt2_ao" in lagrangians:
        breakdown.setdefault("dpt2_ao", np.asarray(lagrangians["dpt2_ao"], dtype=np.float64))
    if "dpt2c_ao" in lagrangians:
        breakdown.setdefault("dpt2c_ao", np.asarray(lagrangians["dpt2c_ao"], dtype=np.float64))
    breakdown.setdefault("olag_ns2", np.asarray(lagrangians.get("olag_ns2", np.zeros_like(lagrangians["dpt2"])), dtype=np.float64))
    breakdown.setdefault("depsa_used", np.asarray(lagrangians.get("depsa_used", lagrangians.get("depsa")), dtype=np.float64))
    breakdown.setdefault("olag_bare", np.asarray(lagrangians.get("olag_bare", lagrangians["olag"]), dtype=np.float64))
    breakdown.setdefault("eigder_depsa_olag_add", np.asarray(lagrangians.get("eigder_depsa_olag_add", np.zeros_like(lagrangians["olag"])), dtype=np.float64))
    if "rdmeig" in lagrangians:
        breakdown.setdefault("rdmeig", np.asarray(lagrangians["rdmeig"], dtype=np.float64))
    if "fpt2_ao" in lagrangians:
        breakdown.setdefault("fpt2_ao", np.asarray(lagrangians["fpt2_ao"], dtype=np.float64))
    if "fpt2c_ao" in lagrangians:
        breakdown.setdefault("fpt2c_ao", np.asarray(lagrangians["fpt2c_ao"], dtype=np.float64))
    if "olagns2_case_terms" in lagrangians:
        breakdown.setdefault("olagns2_case_terms", lagrangians["olagns2_case_terms"])
    if "dpt2c_case_terms" in lagrangians:
        breakdown.setdefault("dpt2c_case_terms", lagrangians["dpt2c_case_terms"])
    if "dpt2c_case_terms_pre_scale" in lagrangians:
        breakdown.setdefault("dpt2c_case_terms_pre_scale", lagrangians["dpt2c_case_terms_pre_scale"])
    if "olagns2_tc_case_terms" in lagrangians:
        breakdown.setdefault("olagns2_tc_case_terms", lagrangians["olagns2_tc_case_terms"])
    if "eigder_terms_full_dpt2" in lagrangians:
        breakdown.setdefault("eigder_terms_full_dpt2", lagrangians["eigder_terms_full_dpt2"])
    olag_full = np.asarray(lagrangians.get("olag_full_before_antisym", lagrangians["olag"]), dtype=np.float64)
    breakdown.setdefault("olag_full", olag_full)
    breakdown.setdefault("olag_full_dpt2", olag_full)
    breakdown.setdefault("molcas_ref_olag_full_from_native", olag_full)
    breakdown.setdefault("molcas_ref_olag_full_mo", np.asarray(lagrangians["olag"], dtype=np.float64))
    # Store raw AO WLag; crosscheck converts to Molcas packed-tri square convention.
    breakdown.setdefault("molcas_ref_wlag_ao_square", np.asarray(lagrangians["wlag_ao"], dtype=np.float64))
    stage_payload = lagrangians.get("stage_checkpoints")
    if isinstance(stage_payload, dict) and len(stage_payload) > 0:
        breakdown.setdefault("stage_checkpoints", stage_payload)
        breakdown.setdefault("stage_checkpoint_order", list(lagrangians.get("stage_checkpoint_order", [])))
    for _k, _v in semican_meta.items():
        breakdown.setdefault(str(_k), _v)

    # Expose AO overlap + MO coefficients needed for Molcas PT2_Lag basis alignment
    # tooling. Keep this on the strict Molcas-parity lane only to avoid bloating
    # generic API callers with large basis arrays.
    if str(parity_profile).strip().lower() == "molcas_ss_strict":
        # Z-vector debug lane: expose packed RHS and solution vectors so we can
        # compare directly to OpenMolcas MCLR `RESP` contents (Kappa/CIT).
        for _k in ("rhs_orb_packed", "rhs_ci_packed", "z_orb_packed", "z_ci_packed"):
            if _k in z_meta and z_meta[_k] is not None:
                try:
                    breakdown.setdefault(_k, np.asarray(z_meta[_k], dtype=np.float64).ravel())
                except Exception:
                    pass
        breakdown.setdefault("S_ao", np.asarray(S_ao, dtype=np.float64))
        breakdown.setdefault("mo_coeff", np.asarray(C, dtype=np.float64))
    for _k in (
        "response_dpt2_mode_req",
        "response_dpt2_mode",
        "z_orb_flip_d_sign",
        "z_orb_flip_c_sign",
    ):
        if _k in z_meta:
            breakdown.setdefault(_k, z_meta[_k])

    if "grad_ref" not in breakdown:
        try:
            from asuka.mcscf.nuc_grad_df import casscf_nuc_grad_df

            gref = casscf_nuc_grad_df(
                scf_out,
                casscf,
                df_backend=str(df_backend),
                int1e_contract_backend=str(int1e_contract_backend),
            )
            breakdown["grad_ref"] = np.asarray(gref.grad, dtype=np.float64)
        except Exception:
            gpt2 = np.asarray(breakdown.get("grad_pt2", np.zeros_like(grad)), dtype=np.float64)
            breakdown["grad_ref"] = np.asarray(grad - gpt2, dtype=np.float64)
    if "grad_pt2" in breakdown and "grad_ref" in breakdown:
        breakdown["grad_total_rebuilt"] = (
            np.asarray(breakdown["grad_ref"], dtype=np.float64)
            + np.asarray(breakdown["grad_pt2"], dtype=np.float64)
        )
    breakdown.setdefault("grad_source", "analytic")

    convergence_flags = {
        "pt2_backend": str(pt2_backend_u),
        "zvector_converged": bool(z_meta.get("converged", False)),
        "zvector_niter": int(z_meta.get("niter", 0)),
        "zvector_residual_norm": float(z_meta.get("residual_norm", np.nan)),
        "zvector_auto_project_tangent_requested": bool(z_meta.get("auto_project_tangent_requested", False)),
        "zvector_auto_project_tangent_applied": bool(z_meta.get("auto_project_tangent_applied", False)),
        "zvector_auto_project_tangent_ratio": float(z_meta.get("auto_project_tangent_ratio", np.nan)),
    }
    if pt2_backend_u in {"sst", "sst-ic", "sst-full"}:
        convergence_flags.setdefault(
            "sst_pcg_converged",
            bool(dict(pt2_breakdown or {}).get("sst_pcg_converged", True)),
        )
        convergence_flags.setdefault(
            "sst_pcg_niter",
            int(dict(pt2_breakdown or {}).get("sst_pcg_niter", 0)),
        )

    return CASPT2GradResult(
        e_tot=e_tot,
        e_ref=e_ref,
        e_pt2=e_pt2,
        grad=grad,
        method="SS",
        iroot=iroot,
        clag=lagrangians["clag"],
        olag=lagrangians["olag"],
        wlag=lagrangians["wlag_ao"],
        dpt2_1rdm=lagrangians["dpt2"],
        convergence_flags=convergence_flags,
        breakdown=breakdown,
    )
from asuka.caspt2.pt2lag import _build_case_amps_from_asuka  # noqa: E402
from asuka.caspt2.pt2lag import _build_lagrangians  # noqa: E402
from asuka.caspt2.pt2lag import _superindex_c_to_f_perm  # noqa: E402


def _compute_clag_fd_with_context(
    ci, ncsf, drt, C, B_ao, h_ao, e_nuc,
    ncore, ncas, nvirt, twos, eri_mo,
    delta=1e-5, verbose=0,
):
    """Compute CLag by FD with full context."""
    from asuka.caspt2.fock_df import build_caspt2_fock_ao
    from asuka.caspt2.superindex import build_superindex
    from asuka.caspt2.f3 import CASPT2CIContext
    from asuka.caspt2.energy import caspt2_energy_ss
    from asuka.rdm.rdm123 import _make_rdm123_pyscf, _reorder_dm123_molcas

    nocc = ncore + ncas

    def _e_pt2(ci_pert):
        ci_n = ci_pert / np.linalg.norm(ci_pert)
        dm1_p, dm2_p, dm3_p = _make_rdm123_pyscf(drt, ci_n, reorder=False)
        dm1_p, dm2_p, dm3_p = _reorder_dm123_molcas(dm1_p, dm2_p, dm3_p, inplace=True)
        fock_p = build_caspt2_fock_ao(h_ao, B_ao, C, dm1_p, ncore, ncas, nvirt, e_nuc=e_nuc)
        smap_p = build_superindex(ncore, ncas, nvirt)
        ci_ctx_p = CASPT2CIContext(drt=drt, ci_csf=ci_n)
        # Reference energy
        h1eff = fock_p.fimo[ncore:nocc, ncore:nocc]
        eri_act = eri_mo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]
        e_ref = fock_p.e_core + np.einsum("tu,tu->", h1eff, dm1_p) + \
                0.5 * np.einsum("tuvx,tuvx->", eri_act, dm2_p.reshape(ncas, ncas, ncas, ncas))
        pt2 = caspt2_energy_ss(smap_p, fock_p, eri_mo, dm1_p, dm2_p, dm3_p, e_ref,
                               ci_context=ci_ctx_p, pt2_backend="cpu")
        return pt2.e_pt2

    clag_fd = np.zeros(ncsf)
    for I in range(ncsf):
        ci_p = ci.copy(); ci_p[I] += delta
        ci_m = ci.copy(); ci_m[I] -= delta
        clag_fd[I] = (_e_pt2(ci_p) - _e_pt2(ci_m)) / (2 * delta)

    # The FD with normalization automatically gives the projected CLag
    if verbose >= 1:
        print(f"[CASPT2 grad] CLag (FD): |CLag|={np.linalg.norm(clag_fd):.6e}")

    return clag_fd


