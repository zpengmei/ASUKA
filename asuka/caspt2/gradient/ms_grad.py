"""MS/XMS-CASPT2 analytic nuclear gradient.

Phase 1 (diagonal weighted accumulation):
  For each state I, build per-state SS Lagrangians; accumulate weighted by the
  diagonal of the State Lagrangian matrix (SLag[I,I] = U[I,iroot]^2 - δ[I,iroot]).
  Solve a single combined Z-vector and assemble the gradient identically to SS.

Phase 2 (off-diagonal XMS correction, XMS only):
  For each (I,J) pair with |slag[I,J]| > threshold, add cross-state contributions:
  - Off-diagonal CLag from SIGDER built using J's amplitudes on I's case_amps (and
    symmetrically I's amplitudes on J's case_amps).
  - Off-diagonal OLag from build_olagns2 with cross-state case_amps and transition RDMs.
  - Off-diagonal WLag derived from the off-diagonal OLag.
  The XMS ∂U0/∂κ correction (model-space Fock derivative) is deferred (Phase 2d).

Reuses the following SS helpers without modification:
  - asuka.caspt2.pt2lag._build_case_amps_from_asuka
  - asuka.caspt2.pt2lag._build_lagrangians
  - asuka.caspt2.gradient.native_grad._solve_zvector
  - asuka.caspt2.gradient.native_grad._assemble_gradient
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(a, dtype=np.float64)


def _delta_clag_from_offdiag_pure(
    case_amps: Any,
    orb: Any,
    fock: Any,
    rdms: Any,
    ci: np.ndarray,
    ncsf: int,
    drt_adapter: Any,
    offdiag_dict: dict,
) -> np.ndarray:
    """Compute CLag contribution PURELY from offdiag_dict, isolating ca.offdiag.

    Runs clag_driver twice — once with offdiag_dict and once with zero arrays for
    the same keys — and returns the difference.  This isolates the linear additive
    contribution from offdiag_dict independent of any per-case ca.offdiag.
    """
    from tools.molcas_caspt2_grad.translation.clagd import clag_driver  # noqa: PLC0415

    if not offdiag_dict:
        return np.zeros(ncsf, dtype=np.float64)

    zeros_dict = {k: np.zeros_like(v) for k, v in offdiag_dict.items()}
    _, _, _, _, _, clag_with, _ = clag_driver(
        case_amps, orb, fock, rdms, ci, ncsf, drt_adapter, offdiag=offdiag_dict,
    )
    _, _, _, _, _, clag_zero, _ = clag_driver(
        case_amps, orb, fock, rdms, ci, ncsf, drt_adapter, offdiag=zeros_dict,
    )
    return np.asarray(clag_with - clag_zero, dtype=np.float64)


def caspt2_ms_gradient_native(
    scf_out: Any,
    casscf: Any,
    *,
    method: str = "MS",
    iroot: int = 0,
    df_backend: str = "cpu",
    int1e_contract_backend: str = "cpu",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
    verbose: int = 0,
) -> Any:
    """Compute MS/XMS-CASPT2 analytic nuclear gradient (Phase 1).

    Parameters
    ----------
    scf_out : Any
        ASUKA SCF result.
    casscf : Any
        ASUKA CASSCF result with multiple roots (nroots >= 2).
    method : {"MS", "XMS"}
        CASPT2 multistate variant.
    iroot : int
        Target MS/XMS state index (0-based).
    df_backend : str
        Backend for DF derivative contractions.
    int1e_contract_backend : str
        Backend for 1e derivative contractions.
    imag_shift, real_shift : float
        Level shift parameters forwarded to SS energy solver.
    z_tol, z_maxiter : float / int
        Z-vector convergence controls.
    verbose : int
        Verbosity level.

    Returns
    -------
    CASPT2GradResult
    """
    from asuka.caspt2.energy import caspt2_energy_ss
    from asuka.caspt2.f3 import CASPT2CIContext
    from asuka.caspt2.fock import build_caspt2_fock
    from asuka.caspt2.fock_df import build_caspt2_fock_ao
    from asuka.caspt2.multistate import build_heff, diagonalize_heff
    from asuka.caspt2.result import CASPT2GradResult
    from asuka.caspt2.superindex import build_superindex
    from asuka.caspt2.gradient.native_grad import _solve_zvector, _assemble_gradient
    from asuka.caspt2.gradient.slag import build_state_lagrangian
    from asuka.caspt2.pt2lag import _build_case_amps_from_asuka, _build_lagrangians
    from asuka.cuguga.drt import build_drt
    from asuka.mcscf.state_average import ci_as_list
    from asuka.rdm.rdm123 import _make_rdm123_pyscf, _reorder_dm123_molcas, _reorder_rdm_pyscf
    from tools.molcas_caspt2_grad.translation.types import (
        OrbitalInfo, CASPT2Fock as TransFock, RDMs,
    )

    method_u = str(method).upper().strip()
    if method_u not in {"MS", "XMS"}:
        raise ValueError(f"caspt2_ms_gradient_native: method must be 'MS' or 'XMS', got '{method}'")

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    # --- Shared orbital dimensions ---
    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    C = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    ci_raw = getattr(casscf, "ci")
    nao, nmo = C.shape
    nvirt = nmo - ncore - ncas
    nocc = ncore + ncas

    nroots = int(getattr(casscf, "nroots", 1))
    if nroots < 2:
        raise ValueError(
            f"caspt2_ms_gradient_native requires nroots >= 2, got nroots={nroots}. "
            "Use caspt2_ss_gradient_native for single-state."
        )
    if int(iroot) < 0 or int(iroot) >= nroots:
        raise ValueError(f"iroot={iroot} out of range for nroots={nroots}")

    if verbose >= 1:
        print(f"[CASPT2 MS grad] method={method_u} iroot={iroot} nroots={nroots}")
        print(f"[CASPT2 MS grad] ncore={ncore} ncas={ncas} nvirt={nvirt} nmo={nmo}")

    # --- AO quantities ---
    B_ao = _asnumpy_f64(getattr(scf_out, "df_B"))
    h_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))
    S_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "S"))
    e_nuc = float(mol.energy_nuc())

    # --- DRT + per-state CI vectors ---
    twos = int(getattr(mol, "spin", 0))
    nelec_total = int(nelecas) if isinstance(nelecas, (int, np.integer)) else int(sum(nelecas))
    drt = build_drt(norb=ncas, nelec=nelec_total, twos_target=twos)
    ci_list_raw = ci_as_list(ci_raw, nroots=nroots)
    ci_vectors = [np.asarray(c, dtype=np.float64).ravel() for c in ci_list_raw]
    ncsf = ci_vectors[0].size

    # --- Build MO ERIs once (shared across states) ---
    eri_mode = str(os.environ.get("ASUKA_SS_ERI_MODE", "df")).strip().lower()
    if eri_mode in {"dense", "full", "eri_dense"}:
        from asuka.hf.dense_eri import build_ao_eri_dense
        ao_basis = getattr(scf_out, "ao_basis", None)
        if ao_basis is None:
            raise TypeError("scf_out must have .ao_basis to build dense ERIs")
        dense_backend = str(os.environ.get("ASUKA_SS_DENSE_ERI_BACKEND", "cuda")).strip().lower()
        eri_res = build_ao_eri_dense(ao_basis, backend=dense_backend, eps_ao=0.0)
        eri_mat = _asnumpy_f64(getattr(eri_res, "eri_mat"))
        eri_ao = np.asarray(eri_mat, dtype=np.float64).reshape(nao, nao, nao, nao)
        eri_mo = np.asarray(
            np.einsum("mp,nq,lr,ks,mnlk->pqrs", C, C, C, C, eri_ao, optimize=True),
            dtype=np.float64,
        )
    else:
        b_mo = np.einsum("mi,mnP,nj->ijP", C, B_ao, C, optimize=True)
        naux = b_mo.shape[2]
        b2 = b_mo.reshape(nmo * nmo, naux)
        eri_mo = np.asarray((b2 @ b2.T).reshape(nmo, nmo, nmo, nmo), dtype=np.float64)

    # --- Shared orbital-info / superindex map ---
    smap = build_superindex(ncore, ncas, nvirt)
    orb = OrbitalInfo(nfro=0, nish=ncore, nash=ncas, nssh=nvirt)

    # --- Per-state loop: build SS results and Lagrangians ---
    ss_results = []
    dm1_list: list[np.ndarray] = []
    dm1_pyscf_list: list[np.ndarray] = []  # PySCF format (before Molcas reorder) for dm1_ms
    dm2_list: list[np.ndarray] = []
    dm3_list: list[np.ndarray] = []
    # PySCF-ordered 4D dm2 (ncas,ncas,ncas,ncas) — before Molcas reorder.
    dm2_pyscf4d_list: list[np.ndarray] = []
    fock_list = []
    lagrangians_list: list[dict[str, Any]] = []
    # Stored for Phase 2 off-diagonal XMS corrections:
    case_amps_list: list[Any] = []
    fock_trans_list: list[Any] = []
    rdms_trans_list: list[Any] = []

    for I in range(nroots):
        ci_I = ci_vectors[I]

        # Build per-state RDMs
        dm1_I, dm2_I, dm3_I = _make_rdm123_pyscf(drt, ci_I, reorder=False)
        # Save PySCF-reordered dm2 (with delta correction) for _build_gfock_casscf_df.
        # Must use the reordered (not raw) format — same as what make_state_averaged_rdms
        # returns and what _build_gfock_casscf_df expects.
        dm1_pyscf_list.append(np.asarray(dm1_I, dtype=np.float64).copy())  # PySCF format for dm1_ms
        _, dm2_pyscf4d_I = _reorder_rdm_pyscf(dm1_I, dm2_I.reshape(ncas, ncas, ncas, ncas), inplace=False)
        dm2_pyscf4d_list.append(np.asarray(dm2_pyscf4d_I, dtype=np.float64))
        dm1_I, dm2_I, dm3_I = _reorder_dm123_molcas(dm1_I, dm2_I, dm3_I, inplace=True)
        dm1_list.append(dm1_I)
        dm2_list.append(dm2_I)
        dm3_list.append(dm3_I)

        # Per-state CASPT2 Fock — mirror native_grad.py: dense path uses MO ERIs
        if eri_mode in {"dense", "full", "eri_dense"}:
            h_mo_I = np.asarray(C.T @ h_ao @ C, dtype=np.float64)
            fock_I = build_caspt2_fock(h_mo_I, eri_mo, dm1_I, ncore, ncas, nvirt, e_nuc=e_nuc)
        else:
            fock_I = build_caspt2_fock_ao(h_ao, B_ao, C, dm1_I, ncore, ncas, nvirt, e_nuc=e_nuc)
        fock_list.append(fock_I)

        # Per-state reference energy from Fock + RDMs
        h1eff_I = fock_I.fimo[ncore:nocc, ncore:nocc]
        eri_act = eri_mo[ncore:nocc, ncore:nocc, ncore:nocc, ncore:nocc]
        e_ref_I = (
            float(fock_I.e_core)
            + float(np.einsum("tu,tu->", h1eff_I, dm1_I))
            + 0.5 * float(np.einsum("tuvx,tuvx->", eri_act, dm2_I.reshape(ncas, ncas, ncas, ncas)))
        )

        # CI context
        ci_ctx_I = CASPT2CIContext(drt=drt, ci_csf=ci_I)

        if verbose >= 1:
            print(f"[CASPT2 MS grad] state {I}: e_ref={e_ref_I:.10f}")

        # SS-CASPT2 energy
        pt2_result_I = caspt2_energy_ss(
            smap, fock_I, eri_mo, dm1_I, dm2_I, dm3_I, e_ref_I,
            ci_context=ci_ctx_I,
            pt2_backend="cpu",
            imag_shift=imag_shift,
            real_shift=real_shift,
            store_sb_decomp=True,
            verbose=verbose,
        )
        ss_results.append(pt2_result_I)

        if verbose >= 1:
            print(f"[CASPT2 MS grad] state {I}: e_pt2={pt2_result_I.e_pt2:.10f} e_tot={pt2_result_I.e_tot:.10f}")

        # Case amplitudes
        trans_fock_I = TransFock(
            fimo=_asnumpy_f64(fock_I.fimo),
            famo=_asnumpy_f64(fock_I.famo),
            fifa=_asnumpy_f64(fock_I.fifa),
            epsa=_asnumpy_f64(fock_I.epsa),
        )
        rdms_I = RDMs(dm1=dm1_I, dm2=dm2_I.reshape(ncas, ncas, ncas, ncas), dm3=dm3_I)

        case_amps_I = _build_case_amps_from_asuka(
            smap, fock_I, dm1_I, dm2_I, dm3_I, ci_ctx_I,
            pt2_result_I.amplitudes, orb, eri_mo,
            pt2_breakdown={
                **dict(getattr(pt2_result_I, "breakdown", {}) or {}),
                "mo_coeff": np.asarray(C, dtype=np.float64),
                "S_ao": np.asarray(S_ao, dtype=np.float64),
            },
        )

        # Per-state Lagrangians
        lag_I = _build_lagrangians(
            case_amps_I, smap, orb, trans_fock_I, rdms_I,
            ci_I, ncsf, eri_mo, C, S_ao, drt,
            verbose=verbose,
            B_ao=B_ao, h_ao=h_ao, e_nuc=e_nuc, twos=twos,
            collect_breakdown=False,
        )
        lagrangians_list.append(lag_I)

        # Store for Phase 2 off-diagonal XMS (case_amps mutated by _build_lagrangians
        # to have lbd and offdiag populated; store after that).
        case_amps_list.append(case_amps_I)
        fock_trans_list.append(trans_fock_I)
        rdms_trans_list.append(rdms_I)

    # --- Build Heff and diagonalize ---
    heff = build_heff(
        nroots, ss_results, ci_vectors, drt, smap,
        fock_list, eri_mo, dm1_list, dm2_list, dm3_list,
        verbose=verbose,
    )
    e_ms, ueff = diagonalize_heff(heff)

    if verbose >= 1:
        print(f"[CASPT2 MS grad] MS energies: {e_ms}")
        print(f"[CASPT2 MS grad] Ueff:\n{ueff}")

    # --- Build SLag ---
    # For XMS we pass is_xms=True, but off-diagonal corrections (Phase 2) are not yet implemented.
    # The SLag diagonal entries are used for Phase 1 regardless.
    u0 = None  # XMS reference rotation not used for Phase 1
    slag = build_state_lagrangian(
        nroots, ueff, int(iroot),
        is_xms=bool(method_u == "XMS"),
        u0=u0,
    )

    if verbose >= 1:
        print(f"[CASPT2 MS grad] SLag diagonal: {np.diag(slag)}")

    # --- Accumulate weighted Lagrangians (Phase 1: diagonal only) ---
    # Weight: w_I = U[I,iroot]^2 (raw MS mixing weights, NO -delta correction).
    # The SLag -delta convention is for Molcas's code structure where the
    # CASSCF gradient is computed separately; here we accumulate full SS Lagrangians.
    # Shapes: olag (nmo,nmo), clag (ncsf,), wlag_ao (nao,nao),
    #         dpt2 (ncas,ncas), dpt2_ao (nao,nao), dpt2c (nmo,nmo)
    lag0 = lagrangians_list[0]
    olag_total = np.zeros_like(np.asarray(lag0["olag"], dtype=np.float64))
    # clag is kept per-root: the SA-CASSCF Z-vector RHS is a block vector of
    # shape (nroots, ncsf); block I = w_I * clag_I (plus Phase 2 off-diag).
    clag_per_root = [np.zeros(ncsf, dtype=np.float64) for _ in range(nroots)]
    wlag_total = np.zeros_like(np.asarray(lag0["wlag_ao"], dtype=np.float64))
    dpt2_total = np.zeros_like(np.asarray(lag0["dpt2"], dtype=np.float64))
    dpt2_ao_total = np.zeros_like(np.asarray(lag0["dpt2_ao"], dtype=np.float64))
    dpt2c_total = np.zeros_like(np.asarray(lag0["dpt2c"], dtype=np.float64))
    # MS-weighted density for the CASSCF part of the gradient.
    dm1_ms = np.zeros_like(dm1_list[0])
    # PySCF-ordered dm2 for _build_gfock_casscf_df (shape: ncas,ncas,ncas,ncas).
    dm2_ms_pyscf4d = np.zeros((ncas, ncas, ncas, ncas), dtype=np.float64)

    for I in range(nroots):
        w_u2 = float(ueff[I, int(iroot)]**2)
        lag_I = lagrangians_list[I]
        if abs(w_u2) > 1e-15:
            olag_total    += w_u2 * np.asarray(lag_I["olag"],    dtype=np.float64)
            clag_per_root[I] += w_u2 * np.asarray(lag_I["clag"],    dtype=np.float64)
            wlag_total    += w_u2 * np.asarray(lag_I["wlag_ao"], dtype=np.float64)
            dpt2_total    += w_u2 * np.asarray(lag_I["dpt2"],    dtype=np.float64)
            dpt2_ao_total += w_u2 * np.asarray(lag_I["dpt2_ao"], dtype=np.float64)
            dpt2c_total   += w_u2 * np.asarray(lag_I["dpt2c"],   dtype=np.float64)
            dm1_ms        += w_u2 * dm1_pyscf_list[I]            # PySCF format
            dm2_ms_pyscf4d += w_u2 * dm2_pyscf4d_list[I]

    # --- CASSCF gfock correction for MS ---
    # At the SA-CASSCF stationary point, the SA-weighted gfock is zero, but the
    # MS-weighted gfock is not: gfock_ms = Σ_I U[I,iroot]^2 * gfock_I ≠ 0.
    # The Z-vector RHS = (∂E_MS/∂κ) = gfock_ms - gfock_ms.T + olag_total.
    # This matches Molcas's rhs_sa.f which passes dm1_ms = Σ_I U^2 * dm1_I
    # to FockGen, adding gfock(dm1_ms) to the Z-vector RHS alongside the PT2 olag.
    from asuka.mcscf.nuc_grad_df import _build_gfock_casscf_df  # noqa: PLC0415

    gfock_ms_mo, _, _, _, _ = _build_gfock_casscf_df(
        B_ao, h_ao, C,
        ncore=int(ncore), ncas=int(ncas),
        dm1_act=dm1_ms, dm2_act=dm2_ms_pyscf4d,
    )
    gfock_ms_mo = np.asarray(gfock_ms_mo, dtype=np.float64)
    olag_total += gfock_ms_mo - gfock_ms_mo.T

    # --- Phase 2: XMS off-diagonal corrections ---
    # For XMS, slag has lower-triangular off-diagonal entries
    # slag[I,J] = 2*u[I,iroot]*u[J,iroot] for I>J (factor-2 accounts for both
    # dHeff[I,J]/dR and dHeff[J,I]/dR directions).
    # Contributions: off-diagonal CLag (from SIGDER with cross-state amplitudes)
    #   and off-diagonal OLag (from build_olagns2 with transition RDMs).
    # XMS dU0/dkappa correction (Phase 2d) is deferred — marked in breakdown.
    _xms_phase2_ran = False
    if method_u == "XMS":
        from asuka.caspt2.pt2lag import _DRTSigmaAdapterNative  # noqa: PLC0415
        from asuka.caspt2.sigder_native import build_sigder_offdiag_asuka_c1  # noqa: PLC0415
        from asuka.rdm.rdm123 import _trans_rdm123_pyscf  # noqa: PLC0415
        from asuka.solver import GUGAFCISolver  # noqa: PLC0415
        from tools.molcas_caspt2_grad.translation.olagns import build_olagns2 as _olagns2  # noqa: PLC0415

        # Build drt_adapter once for all Phase 2 clag_driver calls.
        _nactel_p2 = int(round(np.trace(dm1_list[0])))
        _drt_adapter_p2 = _DRTSigmaAdapterNative(GUGAFCISolver(), drt, _nactel_p2)

        _offdiag_threshold = 1e-12
        for I in range(nroots):
            for J in range(I):  # lower triangular, I > J
                w_ij = float(slag[I, J])
                if abs(w_ij) < _offdiag_threshold:
                    continue

                if verbose >= 1:
                    print(f"[CASPT2 MS grad] Phase 2: pair ({I},{J}) slag={w_ij:.4e}")

                # Transition RDMs <I|...|J> in Molcas ordering.
                tdm1_IJ, tdm2_IJ, tdm3_IJ = _trans_rdm123_pyscf(
                    drt, ci_vectors[I], ci_vectors[J],
                    max_memory_mb=4000.0, reorder=True, reorder_mode="molcas",
                )
                # <J|...|I> by transposition.
                tdm1_JI = np.asarray(tdm1_IJ.T, dtype=np.float64)
                tdm2_JI = np.asarray(tdm2_IJ.transpose(2, 3, 0, 1), dtype=np.float64)
                tdm3_JI = np.asarray(tdm3_IJ.transpose(3, 4, 5, 0, 1, 2), dtype=np.float64)

                # Extract J's amplitude data for SIGDER (→ dHeff[I,J]/dkappa).
                smat_J_c: dict[int, np.ndarray] = {}
                trans_J_c: dict[int, np.ndarray] = {}
                t_sr_J_c: dict[int, np.ndarray] = {}
                lbd_J_c: dict[int, np.ndarray] = {}
                for _c in range(1, 14):
                    _ca = case_amps_list[J][_c - 1]
                    if _ca is not None and _ca.nAS > 0 and _ca.nIS > 0:
                        if _ca.smat is not None:
                            smat_J_c[_c] = np.asarray(_ca.smat, dtype=np.float64)
                        trans_J_c[_c] = np.asarray(_ca.trans, dtype=np.float64)
                        t_sr_J_c[_c] = np.asarray(_ca.T, dtype=np.float64)
                        lbd_J_c[_c] = np.asarray(_ca.lbd, dtype=np.float64)

                # Extract I's amplitude data for SIGDER (→ dHeff[J,I]/dkappa).
                smat_I_c: dict[int, np.ndarray] = {}
                trans_I_c: dict[int, np.ndarray] = {}
                t_sr_I_c: dict[int, np.ndarray] = {}
                lbd_I_c: dict[int, np.ndarray] = {}
                for _c in range(1, 14):
                    _ca = case_amps_list[I][_c - 1]
                    if _ca is not None and _ca.nAS > 0 and _ca.nIS > 0:
                        if _ca.smat is not None:
                            smat_I_c[_c] = np.asarray(_ca.smat, dtype=np.float64)
                        trans_I_c[_c] = np.asarray(_ca.trans, dtype=np.float64)
                        t_sr_I_c[_c] = np.asarray(_ca.T, dtype=np.float64)
                        lbd_I_c[_c] = np.asarray(_ca.lbd, dtype=np.float64)

                nactel_I = int(round(np.trace(dm1_list[I])))
                nactel_J = int(round(np.trace(dm1_list[J])))

                # Build SIGDER from J's amplitudes acting on I's Fock.
                offdiag_IJ = build_sigder_offdiag_asuka_c1(
                    smap=smap, fock=fock_list[I],
                    smat_by_case=smat_J_c, trans_by_case=trans_J_c,
                    t_sr_by_case=t_sr_J_c, lbd_sr_by_case=lbd_J_c,
                    nactel=nactel_I, vecrot=1.0,
                )
                # Build SIGDER from I's amplitudes acting on J's Fock.
                offdiag_JI = build_sigder_offdiag_asuka_c1(
                    smap=smap, fock=fock_list[J],
                    smat_by_case=smat_I_c, trans_by_case=trans_I_c,
                    t_sr_by_case=t_sr_I_c, lbd_sr_by_case=lbd_I_c,
                    nactel=nactel_J, vecrot=1.0,
                )

                # Delta CLag: J's SIGDER acting on I's case_amps (dHeff[I,J]/dkappa → CLag_I).
                dclag_IJ = _delta_clag_from_offdiag_pure(
                    case_amps_list[I], orb, fock_trans_list[I], rdms_trans_list[I],
                    ci_vectors[I], ncsf, _drt_adapter_p2, offdiag_IJ,
                )
                # Delta CLag: I's SIGDER acting on J's case_amps (dHeff[J,I]/dkappa → CLag_J).
                dclag_JI = _delta_clag_from_offdiag_pure(
                    case_amps_list[J], orb, fock_trans_list[J], rdms_trans_list[J],
                    ci_vectors[J], ncsf, _drt_adapter_p2, offdiag_JI,
                )

                # Off-diagonal OLag: J's amplitudes + I-J transition RDMs.
                _rdms_IJ = RDMs(
                    dm1=tdm1_IJ,
                    dm2=tdm2_IJ.reshape(ncas, ncas, ncas, ncas),
                    dm3=tdm3_IJ,
                )
                _olag_IJ_raw, _, _ = _olagns2(
                    case_amps_list[J], orb, fock_trans_list[I], _rdms_IJ, eri_mo,
                    nactel=nactel_I,
                )
                # Off-diagonal OLag: I's amplitudes + J-I transition RDMs.
                _rdms_JI = RDMs(
                    dm1=tdm1_JI,
                    dm2=tdm2_JI.reshape(ncas, ncas, ncas, ncas),
                    dm3=tdm3_JI,
                )
                _olag_JI_raw, _, _ = _olagns2(
                    case_amps_list[I], orb, fock_trans_list[J], _rdms_JI, eri_mo,
                    nactel=nactel_J,
                )

                # Antisymmetrize and combine (both dHeff[I,J] and dHeff[J,I] directions).
                dolag_IJ = np.asarray(_olag_IJ_raw - _olag_IJ_raw.T, dtype=np.float64)
                dolag_JI = np.asarray(_olag_JI_raw - _olag_JI_raw.T, dtype=np.float64)
                dolag_combined = dolag_IJ + dolag_JI

                # Off-diagonal WLag from OLag: W = C @ (0.5*olag) @ C^T.
                dwlag_combined = np.asarray(C @ (0.5 * dolag_combined) @ C.T, dtype=np.float64)

                # Accumulate: slag[I,J] for I>J already has factor-2 built in.
                # Per-root CI blocks: dclag_IJ → block I, dclag_JI → block J.
                clag_per_root[I] += w_ij * dclag_IJ
                clag_per_root[J] += w_ij * dclag_JI
                olag_total += w_ij * dolag_combined
                wlag_total += w_ij * dwlag_combined

                if verbose >= 1:
                    print(
                        f"[CASPT2 MS grad] Phase 2 ({I},{J}): "
                        f"|dclag_IJ|={np.linalg.norm(dclag_IJ):.3e} "
                        f"|dclag_JI|={np.linalg.norm(dclag_JI):.3e} "
                        f"|dolag|={np.linalg.norm(dolag_combined):.3e}"
                    )

        _xms_phase2_ran = True
        # TODO Phase 2d: XMS dU0/dkappa correction (model-space Fock derivative).
        # Adds an OLag term proportional to Σ_{IJ} SLag[I,J]*dH0[I,J]/dkappa where
        # H0 is the XMS zeroth-order Hamiltonian (state-averaged Fock projected to
        # the model space). Not yet implemented.

    # Assemble combined lagrangians dict for Z-vector and gradient assembly.
    # Only the fields actually accessed by _solve_zvector and _assemble_gradient
    # need to be present.
    lagrangians_combined: dict[str, Any] = {
        "olag": olag_total,
        # clag is stacked (nroots, ncsf): _solve_zvector/_flatten_ci handles this.
        "clag": np.stack(clag_per_root, axis=0),
        "wlag_ao": wlag_total,
        "dpt2": dpt2_total,
        "dpt2_ao": dpt2_ao_total,
        "dpt2c": dpt2c_total,
        # Pass olag_bare for optional "bare" DPT2 response mode
        "olag_bare": olag_total.copy(),
    }

    # --- Solve Z-vector ---
    z_orb, z_ci, z_meta = _solve_zvector(
        scf_out, casscf, lagrangians_combined,
        ncore, ncas, nelecas, twos,
        B_ao, h_ao, C, ci_raw,
        z_tol=z_tol, z_maxiter=z_maxiter,
        project_ci_tangent=False,
        return_meta=True,
        dump_vectors=False,
        verbose=verbose,
    )

    if verbose >= 1:
        print(
            f"[CASPT2 MS grad] Z-vector: converged={z_meta.get('converged')} "
            f"niter={z_meta.get('niter')} |r|={z_meta.get('residual_norm', float('nan')):.2e}"
        )

    # --- Assemble gradient ---
    grad, grad_comp = _assemble_gradient(
        scf_out, casscf, lagrangians_combined,
        z_orb, z_ci,
        ncore, ncas, nelecas, twos,
        B_ao, h_ao, S_ao, C,
        dm1_list[iroot], dm2_list[iroot], ci_raw,
        eri_mo=eri_mo,
        df_backend=df_backend,
        int1e_contract_backend=int1e_contract_backend,
        ci_trans_rdm_mode="solver",
        return_components=True,
        collect_breakdown=True,
        verbose=verbose,
        # MS gradient: direct CASSCF gradient uses MS-weighted density (Σ_I U[I,iroot]^2 * D_I),
        # but Z-response remains with SA density (SA orbital stationarity constraint).
        dm1_casscf_direct_override=dm1_ms,
        dm2_casscf_direct_override=dm2_ms_pyscf4d,
    )

    # --- Build result ---
    e_tot_iroot = float(e_ms[int(iroot)])
    # e_ref: SA-CASSCF energy for the target root (from per-state energy)
    e_ref_iroot = float(ss_results[iroot].e_ref) if hasattr(ss_results[iroot], "e_ref") else float(ss_results[iroot].e_tot - ss_results[iroot].e_pt2)
    e_pt2_iroot = float(e_tot_iroot - e_ref_iroot)

    breakdown = dict(grad_comp.get("breakdown", {}) or {})
    breakdown["heff"] = np.asarray(heff, dtype=np.float64)
    breakdown["ueff"] = np.asarray(ueff, dtype=np.float64)
    breakdown["slag"] = np.asarray(slag, dtype=np.float64)
    breakdown["e_ms"] = np.asarray(e_ms, dtype=np.float64)
    breakdown["ss_e_tot"] = np.asarray([float(r.e_tot) for r in ss_results], dtype=np.float64)
    breakdown["ss_e_pt2"] = np.asarray([float(r.e_pt2) for r in ss_results], dtype=np.float64)
    breakdown["grad_source"] = "analytic"
    if method_u == "XMS":
        breakdown["ms_phase"] = "phase2_offdiag_xms" if _xms_phase2_ran else "phase1_diagonal"
        breakdown["xms_phase2_sigder_clag_olag"] = _xms_phase2_ran
        breakdown["xms_phase2d_u0_kappa_correction"] = False  # deferred
    else:
        breakdown["ms_phase"] = "phase1_diagonal"

    return CASPT2GradResult(
        e_tot=e_tot_iroot,
        e_ref=e_ref_iroot,
        e_pt2=e_pt2_iroot,
        grad=np.asarray(grad, dtype=np.float64),
        method=method_u,
        iroot=int(iroot),
        nstates=nroots,
        clag=np.asarray(clag_per_root[int(iroot)], dtype=np.float64),
        olag=np.asarray(olag_total, dtype=np.float64),
        slag=np.asarray(slag, dtype=np.float64),
        wlag=np.asarray(wlag_total, dtype=np.float64),
        dpt2_1rdm=np.asarray(dpt2_total, dtype=np.float64),
        convergence_flags={
            "gradient_backend": "analytic",
            "ms_phase": (
                "phase2_offdiag_xms" if (method_u == "XMS" and _xms_phase2_ran)
                else "phase1_diagonal"
            ),
            "xms_phase2_sigder_clag_olag": bool(method_u == "XMS" and _xms_phase2_ran),
            "xms_phase2d_u0_kappa_correction": False,
            "zvector_converged": bool(z_meta.get("converged", False)),
            "zvector_niter": int(z_meta.get("niter", 0)),
            "zvector_residual_norm": float(z_meta.get("residual_norm", np.nan)),
        },
        breakdown=breakdown,
    )


__all__ = ["caspt2_ms_gradient_native"]
