from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.caspt2.cuda import kernels
from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks, build_all_rhs_df_cuda, df_blocks_to_device
from asuka.caspt2.cuda.sigma_cuda import SigmaC1CaseCouplingCuda
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock
from asuka.caspt2.hzero import build_bmat, compute_e0
from asuka.caspt2.overlap import SBDecomposition, build_smat, sbdiag
from asuka.caspt2.result import CASPT2EnergyResult
from asuka.caspt2.energy import _get_external_energies  # keep orbital-energy convention aligned with CPU


@dataclass(frozen=True)
class _CaseGPUData:
    smat: Any
    t: Any
    tt: Any
    bd: Any
    id: Any
    rhs_sr: Any  # (nindep,nisup) flattened


def caspt2_energy_ss_cuda(
    smap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,  # kept for API parity; RHS uses DF blocks
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    e_ref: float,
    *,
    df_blocks: CASPT2DFBlocks,
    ci_context: CASPT2CIContext | None = None,
    cuda_mode: str = "hybrid",
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    ipea_shift: float = 0.0,
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    verbose: int = 0,
    device: int | None = None,
    store_rhs: bool = False,
    store_row_dots: bool = False,
) -> CASPT2EnergyResult:
    """SS-CASPT2 energy on GPU (C1, FP64) using DF RHS and CUDA sigma kernels."""

    # Silence unused (for now): IPEA shift support not implemented in this CASPT2 path.
    _ = float(ipea_shift)
    _ = eri_mo
    cuda_mode_norm = str(cuda_mode).strip().lower()
    if cuda_mode_norm not in ("hybrid", "full", "strict"):
        raise ValueError("cuda_mode must be 'hybrid', 'full', or 'strict'")
    profile: dict[str, float] | None = {} if bool(cuda_profile) else None

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CASPT2 CUDA backend") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    def _sync() -> None:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

    nish = int(smap.orbs.nish)
    nash = int(smap.orbs.nash)
    nssh = int(smap.orbs.nssh)

    if verbose >= 1:
        print("SS-CASPT2 (CUDA backend)")
        print(f"  nish={nish}, nash={nash}, nssh={nssh}")
        print(f"  E_ref = {float(e_ref):.10f}")

    dm1_cpu = dm2_cpu = dm3_cpu = None
    if cuda_mode_norm != "strict":
        # If RDMs are device arrays (rdm_backend="cuda"), the residual CPU builders
        # (cases != 1,4) need host copies. Do this once to avoid implicit transfers
        # inside Python loops.
        dm1_cpu = dm1
        dm2_cpu = dm2
        dm3_cpu = dm3
        if hasattr(dm1, "__cuda_array_interface__"):
            dm1_cpu = np.asarray(cp.asnumpy(dm1), dtype=np.float64, order="C")
            dm2_cpu = np.asarray(cp.asnumpy(dm2), dtype=np.float64, order="C")
            dm3_cpu = np.asarray(cp.asnumpy(dm3), dtype=np.float64, order="C")

        easum = compute_e0(fock, dm1_cpu, nish, nash)
        if verbose >= 1:
            print(f"  EASUM = {float(easum):.10f}")

    # Build DF blocks on device once.
    import time

    t0 = time.perf_counter()
    df_d = df_blocks_to_device(df_blocks, cp)
    _sync()
    if profile is not None:
        profile["df_blocks_to_device_s"] = float(time.perf_counter() - t0)

    # Device copies of dm tensors (strict mode and case-1/4 GPU builders).
    dm1_d = dm2_d = dm3_d = None
    epsa_d = easum_d = fd_d = fp_d = None
    if cuda_mode_norm in ("full", "strict") and nash > 0:
        from asuka.caspt2.cuda.sb_cuda import precompute_fock_quantities_cuda  # noqa: PLC0415

        t0 = time.perf_counter()
        dm1_d = cp.ascontiguousarray(cp.asarray(dm1, dtype=cp.float64))
        dm2_d = cp.ascontiguousarray(cp.asarray(dm2, dtype=cp.float64))
        dm3_d = cp.ascontiguousarray(cp.asarray(dm3, dtype=cp.float64))
        _sync()
        if profile is not None:
            profile["dm123_to_device_s"] = float(time.perf_counter() - t0)

        t0 = time.perf_counter()
        epsa_d, easum_d, fd_d, fp_d = precompute_fock_quantities_cuda(fock, dm1_d, dm2_d, dm3_d, cp=cp)
        _sync()
        if profile is not None:
            profile["fock_quants_s"] = float(time.perf_counter() - t0)

        if verbose >= 1:
            try:
                print(f"  EASUM = {float(cp.asnumpy(easum_d)):.10f}")
            except Exception:
                pass

    # CPU-side S/B decompositions (small dense), GPU-side RHS.
    sb_decomps: list[Any] = []
    smats_std: list[Any] = []
    bd_list: list[Any] = []
    id_list: list[Any] = []

    # Build RHS blocks on GPU (C basis) and transform to SR basis on GPU.
    if cuda_mode_norm == "strict":
        if dm1_d is None:
            raise RuntimeError("internal error: dm1_d missing in strict mode")
        nactel = max(1, int(round(float(cp.asnumpy(cp.trace(dm1_d[:nash, :nash]))))))
        dm1_rhs = np.empty((0, 0), dtype=np.float64)
        dm2_rhs = np.empty((0, 0, 0, 0), dtype=np.float64)
    else:
        assert dm1_cpu is not None
        nactel = max(1, int(round(float(np.trace(np.asarray(dm1_cpu, dtype=np.float64)[:nash, :nash])))))
        dm1_rhs = dm1_cpu
        dm2_rhs = dm2_cpu
    t0 = time.perf_counter()
    rhs_c_list = build_all_rhs_df_cuda(smap, fock, df_d, dm1_rhs, dm2_rhs, nactel=nactel, device=device)
    _sync()
    if profile is not None:
        profile["rhs_df_build_s"] = float(time.perf_counter() - t0)

    rhs_sr_list: list[Any] = []
    # GPU F3 engine (case 1/4) for full/strict modes.
    f3_eng = None
    if cuda_mode_norm in ("full", "strict") and nash > 0 and int(smap.ntuv) > 0:
        from asuka.caspt2.cuda.f3_cuda import F3ContractionEngineCuda  # noqa: PLC0415

        if ci_context is None:
            raise ValueError(f"cuda_mode='{cuda_mode_norm}' requires ci_context for cases 1 and 4 (F3)")

        t0 = time.perf_counter()
        f3_eng = F3ContractionEngineCuda(
            ci_context,
            np.asarray(fock.epsa, dtype=np.float64),
            device=device,
            cache_bytes=int(cuda_f3_cache_bytes),
        )
        _sync()
        if profile is not None:
            profile["f3_engine_init_s"] = float(time.perf_counter() - t0)

    for case in range(1, 14):
        nasup = int(smap.nasup[case - 1])
        nisup = int(smap.nisup[case - 1])

        if nasup == 0 or nisup == 0:
            if cuda_mode_norm == "strict":
                from asuka.caspt2.cuda.sb_cuda import SBDecompositionDevice  # noqa: PLC0415

                sb_decomps.append(
                    SBDecompositionDevice(
                        s_eigvals=cp.empty((0,), dtype=cp.float64),
                        transform=cp.empty((0, 0), dtype=cp.float64),
                        nindep=0,
                        b_diag=cp.empty((0,), dtype=cp.float64),
                    )
                )
                smats_std.append(cp.empty((0, 0), dtype=cp.float64))
                bd_list.append(cp.empty((0,), dtype=cp.float64))
                id_list.append(np.empty((0,), dtype=np.float64))
                rhs_sr_list.append(cp.zeros((0,), dtype=cp.float64))
            else:
                sb_decomps.append(
                    SBDecomposition(
                        s_eigvals=np.empty(0, dtype=np.float64),
                        transform=np.empty((0, 0), dtype=np.float64),
                        nindep=0,
                        b_diag=np.empty(0, dtype=np.float64),
                    )
                )
                smats_std.append(np.empty((0, 0), dtype=np.float64))
                bd_list.append(np.empty((0,), dtype=np.float64))
                id_list.append(np.empty((0,), dtype=np.float64))
                rhs_sr_list.append(cp.zeros((0,), dtype=cp.float64))
            continue

        if cuda_mode_norm == "strict":
            from asuka.caspt2.cuda.sb_cuda import (  # noqa: PLC0415
                build_bmat_case_cuda,
                build_smat_case_cuda,
                sbdiag_cuda,
            )

            assert dm1_d is not None and dm2_d is not None and dm3_d is not None
            assert epsa_d is not None and easum_d is not None and fd_d is not None and fp_d is not None

            t0 = time.perf_counter()
            smat = build_smat_case_cuda(case, smap, dm1_d, dm2_d, dm3_d, cp=cp)
            _sync()
            if profile is not None:
                profile[f"smat_case{case}_s"] = float(time.perf_counter() - t0)
            smats_std.append(smat)

            t0 = time.perf_counter()
            bmat = build_bmat_case_cuda(
                case,
                smap,
                fock,
                dm1_d,
                dm2_d,
                dm3_d,
                f3_engine=f3_eng,
                smat_d=smat,
                epsa_d=epsa_d,
                easum_d=easum_d,
                fd_d=fd_d,
                fp_d=fp_d,
                cp=cp,
            )
            _sync()
            if profile is not None:
                profile[f"bmat_case{case}_s"] = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            decomp = sbdiag_cuda(
                smat,
                bmat,
                threshold_norm=float(threshold),
                threshold_s=float(threshold_s),
                cp=cp,
            )
            _sync()
            if profile is not None:
                profile[f"sbdiag_case{case}_s"] = float(time.perf_counter() - t0)
            sb_decomps.append(decomp)
        elif cuda_mode_norm == "full" and case in (1, 4) and f3_eng is not None:
            from asuka.caspt2.cuda.sb_cuda import (  # noqa: PLC0415
                build_bmat_a_cuda,
                build_bmat_c_cuda,
                build_smat_a_cuda,
                build_smat_c_cuda,
                sbdiag_cuda,
            )

            # Dominant (ntuv x ntuv) cases on GPU.
            assert dm1_d is not None and dm2_d is not None and dm3_d is not None
            assert epsa_d is not None and easum_d is not None and fd_d is not None and fp_d is not None

            t0 = time.perf_counter()
            if case == 1:
                smat = build_smat_a_cuda(smap, dm1_d, dm2_d, dm3_d, cp=cp)
            else:
                smat = build_smat_c_cuda(smap, dm1_d, dm2_d, dm3_d, cp=cp)
            _sync()
            if profile is not None:
                profile[f"smat_case{case}_s"] = float(time.perf_counter() - t0)
            smats_std.append(smat)

            t0 = time.perf_counter()
            if case == 1:
                bmat = build_bmat_a_cuda(
                    smap,
                    fock,
                    dm1_d,
                    dm2_d,
                    dm3_d,
                    f3_engine=f3_eng,
                    smat_d=smat,
                    easum_d=easum_d,
                    fd_d=fd_d,
                    fp_d=fp_d,
                    cp=cp,
                )
            else:
                bmat = build_bmat_c_cuda(
                    smap,
                    fock,
                    dm1_d,
                    dm2_d,
                    dm3_d,
                    f3_engine=f3_eng,
                    smat_d=smat,
                    easum_d=easum_d,
                    fd_d=fd_d,
                    fp_d=fp_d,
                    cp=cp,
                )
            _sync()
            if profile is not None:
                profile[f"bmat_case{case}_s"] = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            decomp = sbdiag_cuda(
                smat,
                bmat,
                threshold_norm=float(threshold),
                threshold_s=float(threshold_s),
                cp=cp,
            )
            _sync()
            if profile is not None:
                profile[f"sbdiag_case{case}_s"] = float(time.perf_counter() - t0)
            sb_decomps.append(decomp)
        else:
            t0 = time.perf_counter()
            assert dm1_cpu is not None and dm2_cpu is not None and dm3_cpu is not None
            smat = build_smat(case, smap, dm1_cpu, dm2_cpu, dm3_cpu)
            if profile is not None and case in (1, 4):
                profile[f"smat_case{case}_s"] = float(time.perf_counter() - t0)
            smats_std.append(smat)

            t0 = time.perf_counter()
            bmat = build_bmat(case, smap, fock, dm1_cpu, dm2_cpu, dm3_cpu, ci_context=ci_context)
            if profile is not None and case in (1, 4):
                profile[f"bmat_case{case}_s"] = float(time.perf_counter() - t0)

            t0 = time.perf_counter()
            decomp = sbdiag(smat, bmat, threshold_norm=threshold, threshold_s=threshold_s)
            if profile is not None and case in (1, 4):
                profile[f"sbdiag_case{case}_s"] = float(time.perf_counter() - t0)
            sb_decomps.append(decomp)

        if decomp.nindep == 0:
            if cuda_mode_norm == "strict":
                bd_list.append(cp.empty((0,), dtype=cp.float64))
            else:
                bd_list.append(
                    cp.empty((0,), dtype=cp.float64)
                    if (cuda_mode_norm == "full" and case in (1, 4))
                    else np.empty((0,), dtype=np.float64)
                )
            id_list.append(np.empty((0,), dtype=np.float64))
            rhs_sr_list.append(cp.zeros((0,), dtype=cp.float64))
            continue

        # Keep BD on device if already device-resident (cuda_mode='full' case 1/4).
        if cuda_mode_norm == "strict":
            bd_list.append(cp.ascontiguousarray(cp.asarray(decomp.b_diag, dtype=cp.float64).ravel()))
        elif cuda_mode_norm == "full" and case in (1, 4):
            bd_list.append(cp.ascontiguousarray(cp.asarray(decomp.b_diag, dtype=cp.float64).ravel()))
        else:
            bd_list.append(np.asarray(decomp.b_diag, dtype=np.float64, order="C"))
        ext = _get_external_energies(case, smap, fock)
        id_list.append(np.asarray(ext, dtype=np.float64, order="C"))

        rhs_c = rhs_c_list[case - 1]
        if tuple(rhs_c.shape) != (nasup, nisup):
            raise RuntimeError(f"DF RHS shape mismatch for case {case}: {rhs_c.shape} vs {(nasup, nisup)}")

        # PTRTOSR for RHS uses ITYPE=1: rhs_SR = T^T * S * rhs_C
        t_d = cp.asarray(decomp.transform, dtype=cp.float64)
        s_d = cp.asarray(smat, dtype=cp.float64)
        tmp = s_d @ rhs_c
        rhs_sr = t_d.T @ tmp
        rhs_sr_list.append(rhs_sr.ravel())

        if verbose >= 2:
            print(f"  Case {case}: nasup={nasup}, nisup={nisup}, nindep={decomp.nindep}")

    # Decide whether to include Molcas off-diagonal Fock couplings (sigma operator).
    use_sigma = False
    if nash > 0:
        ao = nish
        vo = nish + nash
        so = vo + nssh
        max_off = 0.0
        if nish > 0:
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[ao:vo, :ao]))))  # FTI
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[:ao, ao:vo]))))  # FIT
        if nssh > 0:
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[ao:vo, vo:so]))))  # FTA
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[vo:so, ao:vo]))))  # FAT
        if nish > 0 and nssh > 0:
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[:ao, vo:so]))))  # FIA
            max_off = max(max_off, float(np.max(np.abs(fock.fifa[vo:so, :ao]))))  # FAI
        use_sigma = max_off > 1e-12

    # Solve (H0 - E0) T = -RHS in SR basis.
    amps_sr_list: list[Any] = []
    if not use_sigma:
        t0 = time.perf_counter()
        for case_idx, (rhs, decomp) in enumerate(zip(rhs_sr_list, sb_decomps)):
            nisup = int(smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                amps_sr_list.append(cp.zeros_like(rhs))
                continue
            out = cp.empty((nin, nisup), dtype=cp.float64)
            kernels.apply_precond_sr(
                out=out,
                r=rhs.reshape(nin, nisup),
                bd=cp.asarray(bd_list[case_idx], dtype=cp.float64),
                id=cp.asarray(id_list[case_idx], dtype=cp.float64),
                real_shift=float(real_shift),
                imag_shift=float(imag_shift),
                scale=-1.0,
            )
            amps_sr_list.append(out.ravel())
        _sync()
        if profile is not None:
            profile["solve_s"] = float(time.perf_counter() - t0)
    else:
        from asuka.cuda.krylov_gcrotmk import gcrotmk_xp  # noqa: PLC0415

        sizes = [int(x.size) for x in rhs_sr_list]
        offsets = np.cumsum([0] + sizes, dtype=np.int64)
        total = int(offsets[-1])
        if total == 0:
            amps_sr_list = [cp.zeros_like(x) for x in rhs_sr_list]
        else:
            b = cp.empty((total,), dtype=cp.float64)
            for k, rhs in enumerate(rhs_sr_list):
                if sizes[k] == 0:
                    continue
                b[offsets[k] : offsets[k + 1]] = -rhs

            bd_d = [cp.asarray(bd, dtype=cp.float64) for bd in bd_list]
            id_d = [cp.asarray(i, dtype=cp.float64) for i in id_list]

            def _views(x_flat):
                out = []
                for k in range(13):
                    out.append(x_flat[offsets[k] : offsets[k + 1]])
                return out

            def matvec(x_flat):
                vecs = _views(x_flat)
                sig = sigma_op(vecs)
                out = cp.empty_like(x_flat)
                for k in range(13):
                    if sizes[k] == 0:
                        continue
                    out[offsets[k] : offsets[k + 1]] = sig[k].ravel()
                return out

            def precond(x_flat):
                out = cp.empty_like(x_flat)
                for k, decomp in enumerate(sb_decomps):
                    nisup = int(smap.nisup[k])
                    nin = int(decomp.nindep)
                    if nisup == 0 or nin == 0:
                        continue
                    r = x_flat[offsets[k] : offsets[k + 1]].reshape(nin, nisup)
                    z = out[offsets[k] : offsets[k + 1]].reshape(nin, nisup)
                    kernels.apply_precond_sr(
                        out=z,
                        r=r,
                        bd=bd_d[k],
                        id=id_d[k],
                        real_shift=float(real_shift),
                        imag_shift=float(imag_shift),
                        scale=1.0,
                    )
                return out

            t_sigma0 = time.perf_counter()
            sigma_op = SigmaC1CaseCouplingCuda(
                smap=smap,
                fock=fock,
                smats=smats_std,
                sb_decomp=sb_decomps,
                bd=bd_list,
                id=id_list,
                nactel=nactel,
                real_shift=float(real_shift),
                imag_shift=float(imag_shift),
                device=device,
            )

            if profile is not None:
                _sync()
                profile["sigma_setup_s"] = float(time.perf_counter() - t_sigma0)

            x0 = precond(b)
            t0 = time.perf_counter()
            x, info = gcrotmk_xp(matvec, b, x0=x0, rtol=float(tol), maxiter=int(maxiter), M=precond, xp=cp)
            _sync()
            if profile is not None:
                profile["solve_s"] = float(time.perf_counter() - t0)
            if int(info) != 0 and verbose >= 1:
                print(f"  WARNING: GCROT did not converge (info={info})")

            amps_sr_list = _views(x)

    # Energy: E2 = <RHS|T> in SR basis (per-case + total).
    e_pt2 = 0.0
    breakdown: dict[str, Any] = {}
    if profile is not None:
        breakdown["cuda_profile"] = dict(profile)
    for case_idx, (rhs, t) in enumerate(zip(rhs_sr_list, amps_sr_list), start=1):
        if int(rhs.size) == 0:
            continue
        e_case = kernels.ddot(rhs.ravel(), t.ravel())
        e_pt2 += float(e_case)
        breakdown[f"e2_case{case_idx}"] = float(e_case)
        if verbose >= 1:
            print(f"  Case {case_idx}: E2 = {float(e_case):.10f}")

    # Shift correction (CPU-compatible real shift convention).
    if abs(float(imag_shift)) > 1e-15 or abs(float(real_shift)) > 1e-15:
        corr = 0.0
        for case_idx, (t, decomp) in enumerate(zip(amps_sr_list, sb_decomps)):
            nisup = int(smap.nisup[case_idx])
            nin = int(decomp.nindep)
            if nisup == 0 or nin == 0:
                continue
            bd = cp.asarray(bd_list[case_idx], dtype=cp.float64)
            idv = cp.asarray(id_list[case_idx], dtype=cp.float64)
            d0 = bd[:, None] + idv[None, :]
            delta = cp.full_like(d0, float(real_shift))
            if abs(float(imag_shift)) > 1e-15:
                mask = cp.abs(d0) > 1e-14
                delta = delta + cp.where(mask, (float(imag_shift) * float(imag_shift)) / d0, 0.0)
            tt = t.reshape(nin, nisup)
            corr += float(cp.asnumpy(cp.sum((tt * tt) * delta)))
        e_pt2 -= float(corr)
        breakdown["e_shift_correction"] = -float(corr)
        if verbose >= 1:
            print(f"  Shift correction: {-float(corr):.10f}")

    e_tot = float(e_ref) + float(e_pt2)
    breakdown["e_ref"] = float(e_ref)
    breakdown["e_pt2"] = float(e_pt2)
    breakdown["e_tot"] = float(e_tot)

    # Optionally store row-wise dot products for MS/XMS Heff construction:
    #   row_dots[IAS,JAS] = dot(RHS[IAS,:], T_raw[JAS,:])
    if bool(store_row_dots):
        row_dots_by_case: list[Any] = []
        for case_idx in range(13):
            nasup = int(smap.nasup[case_idx])
            nisup = int(smap.nisup[case_idx])
            decomp = sb_decomps[case_idx]
            nin = int(getattr(decomp, "nindep", 0))
            if nasup == 0 or nisup == 0 or nin == 0:
                row_dots_by_case.append(cp.empty((0, 0), dtype=cp.float64))
                continue
            rhs_c = rhs_c_list[case_idx]
            if tuple(getattr(rhs_c, "shape", ())) != (nasup, nisup):
                raise RuntimeError(
                    f"row_dots: RHS shape mismatch for case {case_idx+1}: {rhs_c.shape} vs {(nasup, nisup)}"
                )
            amps = amps_sr_list[case_idx].reshape(nin, nisup)
            t_d = cp.asarray(decomp.transform, dtype=cp.float64)
            t_raw = t_d @ amps  # (nasup,nisup)
            row_dots = rhs_c @ t_raw.T  # (nasup,nasup)
            row_dots_by_case.append(cp.ascontiguousarray(row_dots, dtype=cp.float64))

        breakdown["row_dots_by_case_cuda"] = row_dots_by_case

    # Transfer amplitudes (and optionally RHS) to host for API stability.
    amps_host = [np.asarray(cp.asnumpy(t), dtype=np.float64) for t in amps_sr_list]
    rhs_host = None
    if bool(store_rhs):
        rhs_host = [np.asarray(cp.asnumpy(v), dtype=np.float64) for v in rhs_sr_list]

    if verbose >= 1:
        print(f"  E_PT2  = {float(e_pt2):.10f}")
        print(f"  E_tot  = {float(e_tot):.10f}")

    if rhs_host is not None:
        breakdown["rhs_sr"] = rhs_host

    return CASPT2EnergyResult(
        e_ref=float(e_ref),
        e_pt2=float(e_pt2),
        e_tot=float(e_tot),
        amplitudes=amps_host,
        breakdown=breakdown,
    )
