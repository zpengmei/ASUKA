from __future__ import annotations

"""ASUKA-native CASPT2 drivers (GPU-first, DF, C1, FP64).

This module provides end-to-end CASPT2 entry points that start from ASUKA's
frontend SCF outputs and ASUKA CASCI/CASSCF results.
"""

from typing import Any, Literal, Sequence

import numpy as np

from asuka.caspt2.energy import caspt2_energy_ss
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.multistate import diagonalize_heff
from asuka.caspt2.result import CASPT2EnergyResult, CASPT2Result
from asuka.caspt2.result import CASPT2SOCResult
from asuka.caspt2.superindex import SuperindexMap, build_superindex


def _nelecas_total(nelecas: int | tuple[int, int]) -> int:
    if isinstance(nelecas, (int, np.integer)):
        return int(nelecas)
    if isinstance(nelecas, (tuple, list)) and len(nelecas) == 2:
        return int(nelecas[0]) + int(nelecas[1])
    raise ValueError("nelecas must be an int or a length-2 tuple/list")


def _as_list_f64(x: Any, *, n: int) -> list[float]:
    if n <= 1:
        return [float(np.asarray(x, dtype=np.float64).ravel()[0])]
    if isinstance(x, (list, tuple)):
        if len(x) != n:
            raise ValueError("root energy list length mismatch")
        return [float(v) for v in x]
    arr = np.asarray(x, dtype=np.float64).ravel()
    if int(arr.size) != n:
        raise ValueError("root energy array length mismatch")
    return [float(v) for v in arr]


def _resolve_scf_out_from_ref(ref: Any, *, scf_out: Any | None) -> Any:
    scf_out_use = scf_out
    if scf_out_use is None:
        scf_out_use = getattr(ref, "scf_out", None)
    if scf_out_use is None and hasattr(ref, "casci"):
        scf_out_use = getattr(getattr(ref, "casci"), "scf_out", None)
    if scf_out_use is None:
        raise ValueError("scf_out is required (missing on ref; pass scf_out explicitly)")
    return scf_out_use


def _caspt2_spinfree_states_for_soc(
    ref: Any,
    scf_out: Any,
    caspt2: CASPT2Result,
    *,
    twos_override: int | None = None,
) -> list["SpinFreeState"]:
    from asuka.cuguga.drt import build_drt  # noqa: PLC0415
    from asuka.mcscf.state_average import ci_as_list  # noqa: PLC0415
    from asuka.soc.si import SpinFreeState  # noqa: PLC0415

    method_u = str(getattr(caspt2, "method", "SS")).upper().strip()
    if method_u not in ("SS", "MS", "XMS"):
        raise ValueError("caspt2.method must be 'SS', 'MS', or 'XMS'")

    if method_u == "SS":
        nstates = 1
        e_list = [float(np.asarray(caspt2.e_tot, dtype=np.float64).ravel()[0])]
    else:
        e_tot = caspt2.e_tot
        if not isinstance(e_tot, (list, tuple, np.ndarray)):
            raise TypeError("MS/XMS CASPT2Result.e_tot must be a list/array of root energies")
        e_arr = np.asarray(e_tot, dtype=np.float64).ravel()
        nstates = int(e_arr.size)
        if nstates < 1:
            raise ValueError("invalid number of MS/XMS roots from caspt2.e_tot")
        e_list = [float(x) for x in e_arr.tolist()]

    ncas = int(getattr(ref, "ncas"))
    nelecas_total = _nelecas_total(getattr(ref, "nelecas"))
    if twos_override is not None:
        twos = int(twos_override)
    else:
        twos = getattr(getattr(ref, "mol", None), "spin", None)
        if twos is None:
            twos = getattr(getattr(scf_out, "mol", None), "spin", 0)
        twos = int(twos)
    drt = build_drt(norb=int(ncas), nelec=int(nelecas_total), twos_target=int(twos))

    ci_orig = ci_as_list(getattr(ref, "ci"), nroots=int(nstates))
    ci_mat = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in ci_orig], axis=0)  # (nstates,ncsf)
    if ci_mat.shape[1] != int(drt.ncsf):
        raise ValueError("reference CI vector length mismatch with constructed active-space DRT")

    if method_u == "SS":
        ci_final = ci_mat
    elif method_u == "MS":
        ueff = np.asarray(caspt2.ueff, dtype=np.float64)
        if ueff.shape != (nstates, nstates):
            raise ValueError("CASPT2Result.ueff shape mismatch")
        ci_final = ueff.T @ ci_mat
    else:  # XMS
        breakdown = caspt2.breakdown if isinstance(caspt2.breakdown, dict) else {}
        if "u0" not in breakdown:
            raise ValueError("XMS CASPT2Result.breakdown['u0'] is missing")
        u0 = np.asarray(breakdown.get("u0"), dtype=np.float64)
        ueff = np.asarray(caspt2.ueff, dtype=np.float64)
        if u0.shape != (nstates, nstates):
            raise ValueError("XMS u0 shape mismatch")
        if ueff.shape != (nstates, nstates):
            raise ValueError("XMS ueff shape mismatch")
        utot = u0 @ ueff
        ci_final = utot.T @ ci_mat

    states: list[SpinFreeState] = []
    for i in range(nstates):
        states.append(SpinFreeState(twos=int(twos), energy=float(e_list[i]), drt=drt, ci=ci_final[i]))
    return states


def _apply_xms_reference_rotation(*, heff: np.ndarray, e_ref_list: list[float], u0: np.ndarray) -> np.ndarray:
    h = np.asarray(heff, dtype=np.float64)
    u = np.asarray(u0, dtype=np.float64)
    e_ref = np.asarray(e_ref_list, dtype=np.float64).ravel()
    nstates = int(e_ref.size)
    if h.shape != (nstates, nstates):
        raise ValueError("heff shape mismatch with e_ref_list")
    if u.shape != (nstates, nstates):
        raise ValueError("u0 shape mismatch with e_ref_list")
    d_ref = np.diag(e_ref)
    h_ref_rot = np.asarray(u.T @ d_ref @ u, dtype=np.float64)
    return np.asarray(h - d_ref + h_ref_rot, dtype=np.float64, order="C")


def _build_df_blocks_from_scf_out(
    *,
    B_ao: Any,
    C: Any,
    ncore: int,
    ncas: int,
    nvirt: int,
    max_memory_mb: float,
) -> Any:
    from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks  # noqa: PLC0415
    from asuka.mrpt2.df_pair_block import DFPairBlock, build_df_pair_blocks_from_df_B  # noqa: PLC0415

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF-CASPT2 CUDA drivers") from e

    B_arr = cp.asarray(B_ao, dtype=cp.float64)
    if B_arr.ndim != 3:
        raise ValueError("scf_out.df_B must have shape (nao,nao,naux)")
    naux = int(B_arr.shape[2])
    if naux <= 0:
        raise ValueError("invalid naux from scf_out.df_B")

    C = cp.asarray(C, dtype=cp.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    C_core = C[:, : int(ncore)]
    C_act = C[:, int(ncore) : int(ncore) + int(ncas)]
    C_virt = C[:, int(ncore) + int(ncas) :]

    # NOTE: l_ab (virt-virt DF block) is no longer built here.
    # The CASPT2 Fock is now constructed in AO basis via build_caspt2_fock_ao(),
    # which avoids the O(nvirt^2 * naux) allocation entirely.
    # The RHS builder (rhs_df_cuda.py) never uses l_ab.

    def _empty_block(nx: int, ny: int) -> DFPairBlock:
        l_full = cp.zeros((int(nx) * int(ny), int(naux)), dtype=cp.float64)
        return DFPairBlock(nx=int(nx), ny=int(ny), l_full=l_full, pair_norm=None)

    pairs: list[tuple[Any, Any]] = []
    labels: list[str] = []
    if int(ncore) > 0:
        pairs.append((C_core, C_core))
        labels.append("ii")
    if int(ncore) > 0 and int(ncas) > 0:
        pairs.append((C_core, C_act))
        labels.append("it")
    if int(ncore) > 0 and int(nvirt) > 0:
        pairs.append((C_core, C_virt))
        labels.append("ia")
    if int(nvirt) > 0 and int(ncas) > 0:
        pairs.append((C_virt, C_act))
        labels.append("at")
    if int(ncas) > 0:
        pairs.append((C_act, C_act))
        labels.append("tu")

    built = build_df_pair_blocks_from_df_B(
        B_arr,
        pairs,
        max_memory=int(max(1.0, float(max_memory_mb))),
        compute_pair_norm=False,
    )
    by_label = {k: v for k, v in zip(labels, built)}

    l_it = by_label.get("it", _empty_block(ncore, ncas))
    l_ia = by_label.get("ia", _empty_block(ncore, nvirt))
    l_at = by_label.get("at", _empty_block(nvirt, ncas))
    l_tu = by_label.get("tu", _empty_block(ncas, ncas))
    l_ii = by_label.get("ii", _empty_block(ncore, ncore))
    return CASPT2DFBlocks(l_it=l_it, l_ia=l_ia, l_at=l_at, l_tu=l_tu, l_ii=l_ii, l_ab=None)


def _build_h1e_mo_from_scf_out(*, scf_out: Any, C: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF-CASPT2 CUDA drivers") from e

    h_ao = getattr(getattr(scf_out, "int1e", None), "hcore", None)
    if h_ao is None:
        raise ValueError("scf_out.int1e.hcore is missing")
    C_d = cp.asarray(C, dtype=cp.float64)
    h_d = cp.asarray(h_ao, dtype=cp.float64)
    h_mo = C_d.T @ h_d @ C_d
    return np.asarray(cp.asnumpy(h_mo), dtype=np.float64, order="C")


def _ensure_cuda_device(device: int) -> Any:
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for DF-CASPT2 CUDA drivers") from e
    cp.cuda.Device(int(device)).use()
    return cp


def _caspt2_from_asuka_ref(
    scf_out: Any,
    ref: Any,
    *,
    method: str,
    nstates: int | None,
    iroot: int,
    cuda_device: int,
    cuda_mode: str,
    rdm_backend: str,
    fock_backend: str,
    integrals_backend: str,
    heff_backend: str,
    pt2_backend: str = "ic",
    imag_shift: float,
    real_shift: float,
    tol: float,
    maxiter: int,
    max_memory_mb: float,
    threshold: float,
    threshold_s: float,
    cuda_f3_cache_bytes: int,
    cuda_profile: bool,
    verbose: int,
) -> CASPT2Result:
    method_u = str(method).upper().strip()
    if method_u not in ("SS", "MS", "XMS"):
        raise ValueError("method must be 'SS', 'MS', or 'XMS'")

    cuda_mode_norm = str(cuda_mode).strip().lower()
    if cuda_mode_norm not in ("strict", "hybrid", "full"):
        raise ValueError("cuda_mode must be one of: 'strict', 'hybrid', 'full'")
    if method_u != "SS" and cuda_mode_norm != "strict":
        raise ValueError("MS/XMS CASPT2 currently requires cuda_mode='strict'")

    if str(rdm_backend).strip().lower() != "cuda":
        raise ValueError("ASUKA-native CASPT2 requires rdm_backend='cuda'")
    if str(integrals_backend).strip().lower() != "df":
        raise ValueError("ASUKA-native CASPT2 requires integrals_backend='df'")
    if str(fock_backend).strip().lower() != "df":
        raise ValueError("ASUKA-native CASPT2 requires fock_backend='df'")
    if method_u != "SS" and str(heff_backend).strip().lower() != "cuda":
        raise ValueError("MS/XMS with DF requires heff_backend='cuda'")

    cp = _ensure_cuda_device(cuda_device)

    # Resolve reference object fields (CASCIResult/CASSCFResult).
    ncore = int(getattr(ref, "ncore"))
    ncas = int(getattr(ref, "ncas"))
    nelecas = getattr(ref, "nelecas")
    nelecas_total = _nelecas_total(nelecas)

    C = getattr(ref, "mo_coeff", None)
    if C is None:
        raise ValueError("reference object missing mo_coeff")
    C = cp.asarray(C, dtype=cp.float64)
    nao, nmo = map(int, C.shape)
    nvirt = int(nmo - ncore - ncas)
    if nvirt < 0:
        raise ValueError("invalid orbital partition: ncore+ncas exceeds nmo")

    nroots_ref = int(getattr(ref, "nroots", 1))
    if nstates is None:
        nstates = 1 if method_u == "SS" else nroots_ref
    nstates = int(nstates)
    if method_u == "SS":
        nstates = 1
    if method_u in ("MS", "XMS") and nstates <= 1:
        raise ValueError("MS/XMS requires nstates > 1")
    if nstates > nroots_ref:
        raise ValueError(f"nstates={nstates} exceeds reference nroots={nroots_ref}")

    iroot = int(iroot)
    if iroot < 0 or iroot >= nstates:
        raise ValueError("iroot out of range")

    # Reference energies.
    if hasattr(ref, "e_roots"):
        e_ref_list = _as_list_f64(getattr(ref, "e_roots"), n=nstates)
    else:
        e_ref_list = _as_list_f64(getattr(ref, "e_tot"), n=nstates)

    # Active-space DRT + CI vectors.
    from asuka.cuguga.drt import build_drt  # noqa: PLC0415
    from asuka.mcscf.state_average import ci_as_list  # noqa: PLC0415

    twos = int(getattr(getattr(scf_out, "mol", None), "spin", 0))
    drt = build_drt(norb=int(ncas), nelec=int(nelecas_total), twos_target=int(twos))
    ci_vectors = ci_as_list(getattr(ref, "ci"), nroots=nstates)
    for i in range(nstates):
        if int(np.asarray(ci_vectors[i]).size) != int(drt.ncsf):
            raise ValueError("CI vector length mismatch with DRT ncsf")

    # Diagonal RDMs (Molcas convention). Prefer CUDA when available; fall back to CPU.
    try:
        from asuka.cuda.rdm123_gpu import make_rdm123_molcas_cuda  # noqa: PLC0415
    except Exception:
        make_rdm123_molcas_cuda = None  # type: ignore[assignment]
        from asuka.rdm.rdm123 import _make_rdm123_pyscf as _make_rdm123_raw  # noqa: PLC0415
        from asuka.rdm.rdm123 import _reorder_dm123_molcas as _reorder_dm123_molcas  # noqa: PLC0415

    dm1_list: list[Any] = []
    dm2_list: list[Any] = []
    dm3_list: list[Any] = []
    for i in range(nstates):
        ci_i = np.asarray(ci_vectors[i], dtype=np.float64)
        if make_rdm123_molcas_cuda is not None:
            dm1_i, dm2_i, dm3_i = make_rdm123_molcas_cuda(drt, ci_i, device=int(cuda_device))
        else:
            dm1_raw, dm2_raw, dm3_raw = _make_rdm123_raw(drt, ci_i, reorder=False)
            dm1_i, dm2_i, dm3_i = _reorder_dm123_molcas(dm1_raw, dm2_raw, dm3_raw, inplace=True)
        dm1_list.append(dm1_i)
        dm2_list.append(dm2_i)
        dm3_list.append(dm3_i)

    # Superindex and DF blocks (state-independent).
    smap = build_superindex(int(ncore), int(ncas), int(nvirt))
    B_ao = getattr(scf_out, "df_B", None)
    if B_ao is None:
        raise ValueError("scf_out.df_B is missing (DF factors required)")
    df_blocks = _build_df_blocks_from_scf_out(
        B_ao=B_ao,
        C=C,
        ncore=ncore,
        ncas=ncas,
        nvirt=nvirt,
        max_memory_mb=float(max_memory_mb),
    )

    # DF-Fock(s) — built in AO basis (avoids l_ab allocation).
    from asuka.caspt2.fock_df import build_caspt2_fock_ao  # noqa: PLC0415

    h_ao = getattr(getattr(scf_out, "int1e", None), "hcore", None)
    if h_ao is None:
        raise ValueError("scf_out.int1e.hcore is missing for AO Fock build")
    h1e_mo = _build_h1e_mo_from_scf_out(scf_out=scf_out, C=C)
    e_nuc = float(getattr(getattr(scf_out, "mol", None), "energy_nuc")())

    def _build_fock(dm1_i):
        return build_caspt2_fock_ao(
            h_ao, B_ao, C, dm1_i,
            int(ncore), int(ncas), int(nvirt),
            e_nuc=float(e_nuc),
        )

    if nstates > 1:
        dm1_avg = dm1_list[0] * 0.0
        for d in dm1_list:
            dm1_avg = dm1_avg + d
        dm1_avg = dm1_avg / float(nstates)
    else:
        dm1_avg = dm1_list[0]

    fock_sa = _build_fock(dm1_avg)

    if method_u == "SS":
        fock = _build_fock(dm1_list[iroot])
    elif method_u == "MS":
        fock = [_build_fock(dm1_list[i]) for i in range(nstates)]
    else:  # XMS
        fock = fock_sa

    # ── SST backend early return ──
    pt2_backend_norm = str(pt2_backend).strip().lower()
    if pt2_backend_norm == "sst":
        if method_u != "SS":
            raise NotImplementedError("SST backend supports SS-CASPT2 only")
        from asuka.caspt2.sst import sst_caspt2_energy_ss  # noqa: PLC0415
        from asuka.caspt2.sst.types import SSTConfig, SSTInput  # noqa: PLC0415

        fock_ss: CASPT2Fock = fock  # type: ignore[assignment]
        C_np = np.asarray(cp.asnumpy(C) if hasattr(C, "get") else C, dtype=np.float64)

        # Convert RDMs from CuPy to NumPy if needed
        _to_np = lambda x: np.asarray(cp.asnumpy(x) if hasattr(x, "get") else x, dtype=np.float64)

        # Build CI context for the reduced system (cases A/C need F3)
        ci_ctx = CASPT2CIContext(
            drt=drt,
            ci_csf=np.asarray(ci_vectors[iroot], dtype=np.float64),
            max_memory_mb=float(max_memory_mb),
        )

        sst_inp = SSTInput(
            ncore=int(ncore),
            ncas=int(ncas),
            nvirt=int(nvirt),
            mo_coeff=C_np,
            dm1_act=_to_np(dm1_list[iroot]),
            dm2_act=_to_np(dm2_list[iroot]),
            fock=fock_ss,
            semicanonical=None,
            e_ref=float(e_ref_list[iroot]),
            e_nuc=float(e_nuc),
            dm3_act=_to_np(dm3_list[iroot]),
            ci_context=ci_ctx,
            smap=smap,
            B_ao=_to_np(B_ao) if B_ao is not None else None,
        )
        sst_cfg = SSTConfig(
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            verbose=int(verbose),
        )
        sst_res = sst_caspt2_energy_ss(sst_inp, sst_cfg)
        return CASPT2Result(
            e_ref=float(sst_res.e_tot - sst_res.e_pt2),
            e_pt2=float(sst_res.e_pt2),
            e_tot=float(sst_res.e_tot),
            amplitudes=sst_res.amplitudes_active if sst_res.amplitudes_active is not None else None,
            method="SS",
            breakdown=sst_res.breakdown,
        )

    # Placeholder for full ERIs: DF path does not use eri_mo.
    eri_mo = np.empty((0,), dtype=np.float64)

    def _ss_one(state: int, *, fock_i: CASPT2Fock, dm1, dm2, dm3, e_ref: float, store_row_dots: bool) -> CASPT2EnergyResult:
        ci_context = CASPT2CIContext(drt=drt, ci_csf=np.asarray(ci_vectors[state], dtype=np.float64), max_memory_mb=float(max_memory_mb))
        return caspt2_energy_ss(
            smap,
            fock_i,
            eri_mo,
            dm1,
            dm2,
            dm3,
            float(e_ref),
            ci_context=ci_context,
            pt2_backend="cuda",
            cuda_device=int(cuda_device),
            cuda_mode=str(cuda_mode_norm),
            cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
            cuda_profile=bool(cuda_profile),
            df_blocks=df_blocks,
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            verbose=int(verbose),
            store_row_dots=bool(store_row_dots),
        )

    if method_u == "SS":
        res = _ss_one(
            int(iroot),
            fock_i=fock,  # type: ignore[arg-type]
            dm1=dm1_list[iroot],
            dm2=dm2_list[iroot],
            dm3=dm3_list[iroot],
            e_ref=e_ref_list[iroot],
            store_row_dots=False,
        )
        return CASPT2Result(
            e_ref=float(res.e_ref),
            e_pt2=float(res.e_pt2),
            e_tot=float(res.e_tot),
            amplitudes=res.amplitudes,
            method="SS",
            breakdown=res.breakdown,
        )

    # MS/XMS: run SS per (possibly rotated) state, then build/diagonalize Heff.
    def _run_ms_from_lists(
        *,
        fock_for_state: Sequence[CASPT2Fock] | CASPT2Fock,
        ci_list: list[np.ndarray],
        dm1_l: list[Any],
        dm2_l: list[Any],
        dm3_l: list[Any],
        e_ref_l: list[float],
        breakdown_extra: dict[str, Any] | None = None,
    ) -> CASPT2Result:
        ss_results: list[CASPT2EnergyResult] = []
        for i in range(nstates):
            fock_i = fock_for_state[i] if isinstance(fock_for_state, (list, tuple)) else fock_for_state
            ss_results.append(
                _ss_one(
                    i,
                    fock_i=fock_i,
                    dm1=dm1_l[i],
                    dm2=dm2_l[i],
                    dm3=dm3_l[i],
                    e_ref=e_ref_l[i],
                    store_row_dots=True,
                )
            )

        from asuka.caspt2.cuda.multistate_cuda import build_heff_cuda  # noqa: PLC0415

        heff_profile: dict[str, Any] | None = {} if bool(cuda_profile) else None
        heff = build_heff_cuda(
            nstates,
            ss_results,
            ci_list,
            drt,
            smap,
            device=int(cuda_device),
            profile=heff_profile,
            verbose=int(verbose),
        )
        ms_energies, ueff = diagonalize_heff(heff)

        breakdown = {
            "ss_energies": [float(r.e_tot) for r in ss_results],
            "ms_energies": ms_energies.tolist(),
            "heff_backend": "cuda",
        }
        if heff_profile is not None:
            breakdown["heff_cuda_profile"] = heff_profile
        if breakdown_extra:
            breakdown.update(breakdown_extra)

        return CASPT2Result(
            e_ref=e_ref_l,
            e_pt2=[float(ms_energies[i] - float(e_ref_l[i])) for i in range(nstates)],
            e_tot=ms_energies.tolist(),
            heff=np.asarray(heff, dtype=np.float64),
            ueff=np.asarray(ueff, dtype=np.float64),
            amplitudes=[r.amplitudes for r in ss_results],
            method="MS",
            breakdown=breakdown,
        )

    if method_u == "MS":
        # State-specific fock list.
        return _run_ms_from_lists(
            fock_for_state=fock,  # type: ignore[arg-type]
            ci_list=[np.asarray(c, dtype=np.float64) for c in ci_vectors],
            dm1_l=dm1_list,
            dm2_l=dm2_list,
            dm3_l=dm3_list,
            e_ref_l=e_ref_list,
        )

    # XMS
    from asuka.caspt2.cuda.xms_cuda import xms_rotate_states_cuda  # noqa: PLC0415

    xms_profile: dict[str, float] | None = {} if bool(cuda_profile) else None
    rotated_ci, u0, h0_model = xms_rotate_states_cuda(
        drt,
        [np.asarray(c, dtype=np.float64) for c in ci_vectors],
        dm1_list,
        fock_sa,
        ncore,
        ncas,
        nstates,
        device=int(cuda_device),
        verbose=int(verbose),
        profile=xms_profile,
    )

    rot_dm1_list: list[Any] = []
    rot_dm2_list: list[Any] = []
    rot_dm3_list: list[Any] = []
    for i, c in enumerate(rotated_ci):
        dm1_i, dm2_i, dm3_i = make_rdm123_molcas_cuda(
            drt, np.asarray(c, dtype=np.float64), device=int(cuda_device)
        )
        rot_dm1_list.append(dm1_i)
        rot_dm2_list.append(dm2_i)
        rot_dm3_list.append(dm3_i)

    ms_rot = _run_ms_from_lists(
        fock_for_state=fock_sa,
        ci_list=[np.asarray(c, dtype=np.float64) for c in rotated_ci],
        dm1_l=rot_dm1_list,
        dm2_l=rot_dm2_list,
        dm3_l=rot_dm3_list,
        e_ref_l=e_ref_list,
        breakdown_extra={
            "h0_model": np.asarray(h0_model, dtype=np.float64).tolist(),
            "u0": np.asarray(u0, dtype=np.float64).tolist(),
            "reference_rotation_applied": True,
            "xms_cuda_profile": dict(xms_profile) if xms_profile is not None else None,
        },
    )

    heff_corr = _apply_xms_reference_rotation(heff=np.asarray(ms_rot.heff, dtype=np.float64), e_ref_list=e_ref_list, u0=np.asarray(u0, dtype=np.float64))
    xms_energies, ueff = diagonalize_heff(heff_corr)

    breakdown = dict(ms_rot.breakdown) if isinstance(ms_rot.breakdown, dict) else {}
    breakdown["heff_uncorrected"] = np.asarray(ms_rot.heff, dtype=np.float64).tolist() if ms_rot.heff is not None else None

    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(xms_energies[i] - float(e_ref_list[i])) for i in range(nstates)],
        e_tot=xms_energies.tolist(),
        heff=np.asarray(heff_corr, dtype=np.float64),
        ueff=np.asarray(ueff, dtype=np.float64),
        amplitudes=ms_rot.amplitudes,
        method="XMS",
        breakdown=breakdown,
    )


def caspt2_from_casscf(
    scf_out: Any,
    casscf: Any,
    *,
    method: Literal["SS", "MS", "XMS"],
    nstates: int | None = None,
    iroot: int = 0,
    cuda_device: int = 0,
    cuda_mode: Literal["strict"] = "strict",
    rdm_backend: Literal["cuda"] = "cuda",
    fock_backend: Literal["df"] = "df",
    integrals_backend: Literal["df"] = "df",
    heff_backend: Literal["cuda"] = "cuda",
    pt2_backend: str = "ic",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    max_memory_mb: float = 4000.0,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    verbose: int = 0,
) -> CASPT2Result:
    return _caspt2_from_asuka_ref(
        scf_out,
        casscf,
        method=str(method),
        nstates=nstates,
        iroot=int(iroot),
        cuda_device=int(cuda_device),
        cuda_mode=str(cuda_mode),
        rdm_backend=str(rdm_backend),
        fock_backend=str(fock_backend),
        integrals_backend=str(integrals_backend),
        heff_backend=str(heff_backend),
        pt2_backend=str(pt2_backend),
        imag_shift=float(imag_shift),
        real_shift=float(real_shift),
        tol=float(tol),
        maxiter=int(maxiter),
        max_memory_mb=float(max_memory_mb),
        threshold=float(threshold),
        threshold_s=float(threshold_s),
        cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
        cuda_profile=bool(cuda_profile),
        verbose=int(verbose),
    )


def caspt2_from_casci(
    scf_out: Any,
    casci: Any,
    *,
    method: Literal["SS", "MS", "XMS"],
    nstates: int | None = None,
    iroot: int = 0,
    cuda_device: int = 0,
    cuda_mode: Literal["strict"] = "strict",
    rdm_backend: Literal["cuda"] = "cuda",
    fock_backend: Literal["df"] = "df",
    integrals_backend: Literal["df"] = "df",
    heff_backend: Literal["cuda"] = "cuda",
    pt2_backend: str = "ic",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    max_memory_mb: float = 4000.0,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    verbose: int = 0,
) -> CASPT2Result:
    return _caspt2_from_asuka_ref(
        scf_out,
        casci,
        method=str(method),
        nstates=nstates,
        iroot=int(iroot),
        cuda_device=int(cuda_device),
        cuda_mode=str(cuda_mode),
        rdm_backend=str(rdm_backend),
        fock_backend=str(fock_backend),
        integrals_backend=str(integrals_backend),
        heff_backend=str(heff_backend),
        pt2_backend=str(pt2_backend),
        imag_shift=float(imag_shift),
        real_shift=float(real_shift),
        tol=float(tol),
        maxiter=int(maxiter),
        max_memory_mb=float(max_memory_mb),
        threshold=float(threshold),
        threshold_s=float(threshold_s),
        cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
        cuda_profile=bool(cuda_profile),
        verbose=int(verbose),
    )

def run_caspt2(
    ref: Any,
    *,
    scf_out: Any | None = None,
    method: Literal["SS", "MS", "XMS"],
    nstates: int | None = None,
    iroot: int = 0,
    device: int = 0,
    cuda_mode: Literal["strict"] = "strict",
    rdm_backend: Literal["cuda"] = "cuda",
    fock_backend: Literal["df"] = "df",
    integrals_backend: Literal["df"] = "df",
    heff_backend: Literal["cuda"] = "cuda",
    pt2_backend: str = "ic",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    max_memory_mb: float = 4000.0,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    verbose: int = 0,
) -> CASPT2Result:
    """Run SS/MS/XMS CASPT2 from an ASUKA CASCI/CASSCF result.

    This is the primary convenience entry point: it accepts the output of
    `asuka.mcscf.run_casci*` or `asuka.mcscf.run_casscf` directly.
    """

    scf_out_use = scf_out
    if scf_out_use is None:
        scf_out_use = getattr(ref, "scf_out", None)
    if scf_out_use is None and hasattr(ref, "casci"):
        scf_out_use = getattr(getattr(ref, "casci"), "scf_out", None)
    if scf_out_use is None:
        raise ValueError("scf_out is required (missing on ref; pass scf_out explicitly)")

    return _caspt2_from_asuka_ref(
        scf_out_use,
        ref,
        method=str(method),
        nstates=nstates,
        iroot=int(iroot),
        cuda_device=int(device),
        cuda_mode=str(cuda_mode),
        rdm_backend=str(rdm_backend),
        fock_backend=str(fock_backend),
        integrals_backend=str(integrals_backend),
        heff_backend=str(heff_backend),
        pt2_backend=str(pt2_backend),
        imag_shift=float(imag_shift),
        real_shift=float(real_shift),
        tol=float(tol),
        maxiter=int(maxiter),
        max_memory_mb=float(max_memory_mb),
        threshold=float(threshold),
        threshold_s=float(threshold_s),
        cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
        cuda_profile=bool(cuda_profile),
        verbose=int(verbose),
    )


def run_caspt2_soc(
    ref: Any,
    *,
    scf_out: Any | None = None,
    soc_integrals: "SOCIntegrals | None" = None,
    soc_method: Literal["integrals", "amfi"] = "amfi",
    # --- AMFI (internal SOC integrals) ---
    amfi_scale: float = 1.0,
    amfi_include_mean_field: bool = True,
    amfi_atoms: Sequence[int] | None = None,
    amfi_rme_scale: float = 4.0,
    amfi_phase: complex = 1j,
    # --- CASPT2 args forwarded ---
    method: Literal["SS", "MS", "XMS"],
    nstates: int | None = None,
    iroot: int = 0,
    device: int = 0,
    cuda_mode: Literal["strict"] = "strict",
    rdm_backend: Literal["cuda"] = "cuda",
    fock_backend: Literal["df"] = "df",
    integrals_backend: Literal["df"] = "df",
    heff_backend: Literal["cuda"] = "cuda",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    max_memory_mb: float = 4000.0,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    verbose: int = 0,
    # --- SOC args ---
    soc_backend: str = "auto",
    soc_block_nops: int = 8,
    soc_symmetrize: bool = True,
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: str = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
    return_h_si: bool = False,
) -> CASPT2SOCResult:
    """Run CASPT2, then perform SOC-SI (RASSI-SO style) on the correlated spin-free states.

    Notes
    -----
    - This is a Phase-1 SOC-on-top wrapper: the correlation step is real/spin-free.
    - If `soc_integrals` is provided, it must be a `asuka.soc.SOCIntegrals` object in the
      *active MO basis* corresponding to the CASPT2 DRT orbitals (shape: (3, ncas, ncas)).
    - If `soc_integrals is None`, `soc_method='amfi'` builds one-center OpenMolcas-style
      AMFI integrals internally and slices them to the CAS active space.
    """

    from asuka.soc.si import SOCIntegrals, soc_state_interaction  # noqa: PLC0415

    scf_out_use = _resolve_scf_out_from_ref(ref, scf_out=scf_out)

    caspt2 = run_caspt2(
        ref,
        scf_out=scf_out_use,
        method=method,
        nstates=nstates,
        iroot=int(iroot),
        device=int(device),
        cuda_mode=cuda_mode,
        rdm_backend=rdm_backend,
        fock_backend=fock_backend,
        integrals_backend=integrals_backend,
        heff_backend=heff_backend,
        imag_shift=float(imag_shift),
        real_shift=float(real_shift),
        tol=float(tol),
        maxiter=int(maxiter),
        max_memory_mb=float(max_memory_mb),
        threshold=float(threshold),
        threshold_s=float(threshold_s),
        cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
        cuda_profile=bool(cuda_profile),
        verbose=int(verbose),
    )

    soc_method_s = str(soc_method).strip().lower()
    if soc_integrals is None:
        if soc_method_s != "amfi":
            raise ValueError("soc_integrals is None; set soc_method='amfi' to build internal AMFI SOC integrals")
        from asuka.soc.amfi import build_amfi_soc_integrals_from_scf_out  # noqa: PLC0415

        mo_coeff = getattr(ref, "mo_coeff", None)
        if mo_coeff is None:
            raise ValueError("ref.mo_coeff is required to build AMFI SOC integrals")
        ncore_ref = int(getattr(ref, "ncore", 0))
        ncas_ref = int(getattr(ref, "ncas", 0))
        soc_integrals_use = build_amfi_soc_integrals_from_scf_out(
            scf_out_use,
            mo_coeff=mo_coeff,
            ncore=int(ncore_ref),
            ncas=int(ncas_ref),
            rme_scale=float(amfi_rme_scale),
            phase=complex(amfi_phase),
            scale=float(amfi_scale),
            include_mean_field=bool(amfi_include_mean_field),
            atoms=amfi_atoms,
        )
    else:
        if not isinstance(soc_integrals, SOCIntegrals):
            raise TypeError("soc_integrals must be a asuka.soc.SOCIntegrals")
        soc_integrals_use = soc_integrals

    states_sf = _caspt2_spinfree_states_for_soc(ref, scf_out_use, caspt2)
    e_so, c_so, basis = soc_state_interaction(
        states_sf,
        soc_integrals_use,
        include_diag=True,
        block_nops=int(soc_block_nops),
        symmetrize=bool(soc_symmetrize),
        backend=str(soc_backend),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        cuda_gm_strategy=str(soc_cuda_gm_strategy),
        cuda_gm_direct_max_nb_nk=int(soc_cuda_gm_direct_max_nb_nk),
    )

    h_si = None
    if bool(return_h_si):
        c = np.asarray(c_so, dtype=np.complex128)
        e = np.asarray(e_so, dtype=np.float64).ravel()
        h_si = np.asarray(c @ np.diag(e.astype(np.complex128)) @ c.conj().T, dtype=np.complex128)

    return CASPT2SOCResult(
        caspt2=caspt2,
        spinfree_states=list(states_sf),
        so_energies=np.asarray(e_so, dtype=np.float64),
        so_vectors=np.asarray(c_so, dtype=np.complex128),
        so_basis=list(basis),
        h_si=h_si,
    )


def run_caspt2_soc_multispin(
    refs: Sequence[Any],
    *,
    scf_outs: Sequence[Any | None] | None = None,
    soc_integrals: "SOCIntegrals | None" = None,
    soc_method: Literal["integrals", "amfi"] = "amfi",
    # --- AMFI (internal SOC integrals) ---
    amfi_scale: float = 1.0,
    amfi_include_mean_field: bool = True,
    amfi_atoms: Sequence[int] | None = None,
    amfi_rme_scale: float = 4.0,
    amfi_phase: complex = 1j,
    # --- CASPT2 args forwarded (applied per ref) ---
    method: Literal["SS", "MS", "XMS"],
    nstates: int | Sequence[int | None] | None = None,
    iroot: int | Sequence[int] = 0,
    device: int = 0,
    cuda_mode: Literal["strict"] = "strict",
    rdm_backend: Literal["cuda"] = "cuda",
    fock_backend: Literal["df"] = "df",
    integrals_backend: Literal["df"] = "df",
    heff_backend: Literal["cuda"] = "cuda",
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    max_memory_mb: float = 4000.0,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    verbose: int = 0,
    # --- SOC args ---
    soc_backend: str = "auto",
    soc_block_nops: int = 8,
    soc_symmetrize: bool = True,
    soc_cuda_threads: int = 128,
    soc_cuda_sync: bool = True,
    soc_cuda_fallback_to_cpu: bool = True,
    soc_cuda_gm_strategy: str = "auto",
    soc_cuda_gm_direct_max_nb_nk: int = 256,
    return_h_si: bool = False,
) -> "CASPT2SOCResultMultiSpin":
    """Run multiple CASPT2 jobs (different spin manifolds), then perform a single SOC-SI diagonalization.

    This convenience wrapper is intended for cross-spin SOC mixing (e.g., singlet+triplet) when all
    correlated states share a common active-orbital basis (same `ref.mo_coeff` active block).
    """

    from asuka.soc.si import SOCIntegrals, soc_state_interaction  # noqa: PLC0415

    refs_list = list(refs)
    if not refs_list:
        raise ValueError("refs must be a non-empty sequence")

    if scf_outs is None:
        scf_outs_list: list[Any | None] = [None] * len(refs_list)
    else:
        scf_outs_list = list(scf_outs)
        if len(scf_outs_list) != len(refs_list):
            raise ValueError("scf_outs length must match refs length")

    # Normalize per-ref nstates/iroot.
    if isinstance(nstates, (list, tuple)):
        nstates_list = list(nstates)
        if len(nstates_list) != len(refs_list):
            raise ValueError("nstates length must match refs length")
    else:
        nstates_list = [nstates] * len(refs_list)

    if isinstance(iroot, (list, tuple, np.ndarray)):
        iroot_list = [int(x) for x in list(iroot)]
        if len(iroot_list) != len(refs_list):
            raise ValueError("iroot length must match refs length")
    else:
        iroot_list = [int(iroot)] * len(refs_list)

    def _asnumpy_f64(x: Any) -> np.ndarray:
        try:
            import cupy as cp  # noqa: PLC0415

            if isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
                return cp.asnumpy(x).astype(np.float64, copy=False)
        except Exception:
            pass
        return np.asarray(x, dtype=np.float64)

    # Validate common active space / orbitals for SOC.
    ref0 = refs_list[0]
    mo0 = getattr(ref0, "mo_coeff", None)
    if mo0 is None:
        raise ValueError("refs[0].mo_coeff is required for SOC mixing")
    ncore0 = int(getattr(ref0, "ncore", 0))
    ncas0 = int(getattr(ref0, "ncas", 0))
    nelecas0 = _nelecas_total(getattr(ref0, "nelecas"))
    C0 = _asnumpy_f64(mo0)
    for k, ref in enumerate(refs_list[1:], start=1):
        if int(getattr(ref, "ncore", 0)) != ncore0 or int(getattr(ref, "ncas", 0)) != ncas0:
            raise ValueError("all refs must share the same ncore/ncas for common-orbital SOC mixing")
        if _nelecas_total(getattr(ref, "nelecas")) != nelecas0:
            raise ValueError("all refs must share the same active electron count for SOC mixing")
        Ck = _asnumpy_f64(getattr(ref, "mo_coeff", None))
        if Ck.shape != C0.shape:
            raise ValueError("ref.mo_coeff shape mismatch across refs")
        if float(np.max(np.abs(Ck[:, ncore0 : ncore0 + ncas0] - C0[:, ncore0 : ncore0 + ncas0]))) > 1e-10:
            raise ValueError("active MO coefficients differ across refs; use a biorth/RASSI workflow instead")

    soc_method_s = str(soc_method).strip().lower()
    if soc_integrals is None:
        if soc_method_s != "amfi":
            raise ValueError("soc_integrals is None; set soc_method='amfi' to build internal AMFI SOC integrals")
        from asuka.soc.amfi import build_amfi_soc_integrals_from_scf_out  # noqa: PLC0415

        scf_out0 = _resolve_scf_out_from_ref(ref0, scf_out=scf_outs_list[0])
        soc_integrals_use = build_amfi_soc_integrals_from_scf_out(
            scf_out0,
            mo_coeff=C0,
            ncore=int(ncore0),
            ncas=int(ncas0),
            rme_scale=float(amfi_rme_scale),
            phase=complex(amfi_phase),
            scale=float(amfi_scale),
            include_mean_field=bool(amfi_include_mean_field),
            atoms=amfi_atoms,
        )
    else:
        if not isinstance(soc_integrals, SOCIntegrals):
            raise TypeError("soc_integrals must be a asuka.soc.SOCIntegrals")
        soc_integrals_use = soc_integrals

    caspt2_results: list[CASPT2Result] = []
    states_sf_all: list[Any] = []
    labels_all: list[tuple[int, int]] = []

    for i_ref, (ref, scf_out_i, nst_i, ir_i) in enumerate(zip(refs_list, scf_outs_list, nstates_list, iroot_list)):
        scf_out_use = _resolve_scf_out_from_ref(ref, scf_out=scf_out_i)
        caspt2_i = run_caspt2(
            ref,
            scf_out=scf_out_use,
            method=method,
            nstates=nst_i,
            iroot=int(ir_i),
            device=int(device),
            cuda_mode=cuda_mode,
            rdm_backend=rdm_backend,
            fock_backend=fock_backend,
            integrals_backend=integrals_backend,
            heff_backend=heff_backend,
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            max_memory_mb=float(max_memory_mb),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            cuda_f3_cache_bytes=int(cuda_f3_cache_bytes),
            cuda_profile=bool(cuda_profile),
            verbose=int(verbose),
        )
        caspt2_results.append(caspt2_i)

        twos_i = int(getattr(getattr(ref, "mol", None), "spin", getattr(getattr(scf_out_use, "mol", None), "spin", 0)))
        states_i = _caspt2_spinfree_states_for_soc(ref, scf_out_use, caspt2_i, twos_override=int(twos_i))
        states_sf_all.extend(states_i)
        labels_all.extend([(int(i_ref), int(j)) for j in range(len(states_i))])

    e_so, c_so, basis = soc_state_interaction(
        states_sf_all,
        soc_integrals_use,
        include_diag=True,
        block_nops=int(soc_block_nops),
        symmetrize=bool(soc_symmetrize),
        backend=str(soc_backend),
        cuda_threads=int(soc_cuda_threads),
        cuda_sync=bool(soc_cuda_sync),
        cuda_fallback_to_cpu=bool(soc_cuda_fallback_to_cpu),
        cuda_gm_strategy=str(soc_cuda_gm_strategy),
        cuda_gm_direct_max_nb_nk=int(soc_cuda_gm_direct_max_nb_nk),
    )

    h_si = None
    if bool(return_h_si):
        c = np.asarray(c_so, dtype=np.complex128)
        e = np.asarray(e_so, dtype=np.float64).ravel()
        h_si = np.asarray(c @ np.diag(e.astype(np.complex128)) @ c.conj().T, dtype=np.complex128)

    from asuka.caspt2.result import CASPT2SOCResultMultiSpin  # noqa: PLC0415

    return CASPT2SOCResultMultiSpin(
        caspt2=list(caspt2_results),
        spinfree_states=list(states_sf_all),
        spinfree_labels=list(labels_all),
        so_energies=np.asarray(e_so, dtype=np.float64),
        so_vectors=np.asarray(c_so, dtype=np.complex128),
        so_basis=list(basis),
        h_si=h_si,
    )


__all__ = ["caspt2_from_casci", "caspt2_from_casscf", "run_caspt2", "run_caspt2_soc", "run_caspt2_soc_multispin"]
