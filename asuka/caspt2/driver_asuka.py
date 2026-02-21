from __future__ import annotations

"""ASUKA-native CASPT2 drivers (GPU-first, DF, C1, FP64).

This module provides end-to-end CASPT2 entry points that start from ASUKA's
frontend SCF outputs and ASUKA CASCI/CASSCF results. No PySCF objects are used.
"""

from typing import Any, Literal, Sequence

import numpy as np

from asuka.caspt2.energy import caspt2_energy_ss
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.multistate import diagonalize_heff
from asuka.caspt2.result import CASPT2EnergyResult, CASPT2Result
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

    # Guardrail: the virt-virt DF block can be very large.
    if int(nvirt) > 0 and int(naux) > 0:
        ab_bytes = float(int(nvirt) * int(nvirt)) * float(int(naux)) * 8.0
        if ab_bytes > float(max_memory_mb) * 1e6:
            raise MemoryError(
                "DF Fock/RHS requested full virt-virt DF block (ab), but the implied "
                f"allocation is ~{ab_bytes/1e6:.1f} MB (nvirt={int(nvirt)}, naux={int(naux)}). "
                "Increase max_memory_mb or reduce the basis size."
            )

    def _empty_block(nx: int, ny: int) -> DFPairBlock:
        l_full = cp.zeros((int(nx) * int(ny), int(naux)), dtype=cp.float64)
        return DFPairBlock(nx=int(nx), ny=int(ny), l_full=l_full, pair_norm=None)

    pairs: list[tuple[Any, Any]] = []
    labels: list[str] = []
    if int(ncore) > 0:
        pairs.append((C_core, C_core))
        labels.append("ii")
    if int(nvirt) > 0:
        pairs.append((C_virt, C_virt))
        labels.append("ab")
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
    l_ab = by_label.get("ab", _empty_block(nvirt, nvirt))
    return CASPT2DFBlocks(l_it=l_it, l_ia=l_ia, l_at=l_at, l_tu=l_tu, l_ii=l_ii, l_ab=l_ab)


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
    if cuda_mode_norm != "strict":
        raise ValueError("ASUKA-native CASPT2 driver currently supports cuda_mode='strict' only")

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

    # Diagonal RDMs on GPU (Molcas convention).
    from asuka.cuda.rdm123_gpu import make_rdm123_molcas_cuda  # noqa: PLC0415

    dm1_list: list[Any] = []
    dm2_list: list[Any] = []
    dm3_list: list[Any] = []
    for i in range(nstates):
        dm1_i, dm2_i, dm3_i = make_rdm123_molcas_cuda(
            drt, np.asarray(ci_vectors[i], dtype=np.float64), device=int(cuda_device)
        )
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

    # DF-Fock(s).
    from asuka.caspt2.fock_df import build_caspt2_fock_df  # noqa: PLC0415

    h1e_mo = _build_h1e_mo_from_scf_out(scf_out=scf_out, C=C)
    e_nuc = float(getattr(getattr(scf_out, "mol", None), "energy_nuc")())

    if nstates > 1:
        dm1_avg = dm1_list[0] * 0.0
        for d in dm1_list:
            dm1_avg = dm1_avg + d
        dm1_avg = dm1_avg / float(nstates)
    else:
        dm1_avg = dm1_list[0]

    fock_sa = build_caspt2_fock_df(
        h1e_mo,
        df_blocks,
        dm1_avg,
        int(ncore),
        int(ncas),
        int(nvirt),
        e_nuc=float(e_nuc),
        xp="cupy",
    )

    if method_u == "SS":
        fock = build_caspt2_fock_df(
            h1e_mo,
            df_blocks,
            dm1_list[iroot],
            int(ncore),
            int(ncas),
            int(nvirt),
            e_nuc=float(e_nuc),
            xp="cupy",
        )
    elif method_u == "MS":
        fock = [
            build_caspt2_fock_df(
                h1e_mo,
                df_blocks,
                dm1_list[i],
                int(ncore),
                int(ncas),
                int(nvirt),
                e_nuc=float(e_nuc),
                xp="cupy",
            )
            for i in range(nstates)
        ]
    else:  # XMS
        fock = fock_sa

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
            cuda_mode="strict",
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

__all__ = ["caspt2_from_casci", "caspt2_from_casscf", "run_caspt2"]
