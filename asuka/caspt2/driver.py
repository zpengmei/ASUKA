"""Top-level CASPT2 driver.

Provides ``caspt2_from_mc()`` which takes a PySCF CASCI/CASSCF object
and runs SS/MS/XMS-CASPT2.
"""

from __future__ import annotations

import itertools
import os
from typing import Any, Literal

import numpy as np

from asuka.caspt2.energy import caspt2_energy_ss
from asuka.caspt2.f3 import CASPT2CIContext
from asuka.caspt2.fock import CASPT2Fock, build_caspt2_fock
from asuka.caspt2.hcoup import hcoup_case_contribution
from asuka.caspt2.hzero import build_bmat
from asuka.caspt2.multistate import build_heff, diagonalize_heff
from asuka.caspt2.overlap import build_smat, sbdiag
from asuka.caspt2.result import CASPT2EnergyResult, CASPT2Result
from asuka.caspt2.rhs import build_rhs
from asuka.caspt2.superindex import SuperindexMap, build_superindex
from asuka.caspt2.xms import xms_rotate_states


def caspt2_from_mc(
    mc: Any,
    *,
    nstates: int = 1,
    method: str = "SS",
    iroot: int = 0,
    pt2_backend: str = "cpu",
    cuda_device: int | None = None,
    cuda_mode: str = "hybrid",
    cuda_f3_cache_bytes: int = 512 * 1024 * 1024,
    cuda_profile: bool = False,
    integrals_backend: Literal["fulleri", "df"] = "fulleri",
    fock_backend: Literal["fulleri", "df"] = "fulleri",
    rdm_backend: Literal["cpu", "cuda"] = "cpu",
    heff_backend: Literal["cpu", "cuda"] = "cpu",
    cuda_e2e: bool = False,
    auxbasis: Any = "weigend+etb",
    ipea_shift: float = 0.0,
    imag_shift: float = 0.0,
    real_shift: float = 0.0,
    tol: float = 1e-8,
    maxiter: int = 200,
    max_memory_mb: float = 4000.0,
    threshold: float = 1e-10,
    threshold_s: float = 1e-8,
    ci_strategy: Literal["auto", "mc", "guga_resolve"] = "auto",
    quasi_canonical_active: bool = False,
    verbose: int = 0,
) -> CASPT2Result:
    """Top-level CASPT2 driver from a PySCF CASCI/CASSCF object.

    Parameters
    ----------
    mc : PySCF CASCI or CASSCF object
        Must have converged (mc.e_tot, mc.mo_coeff, mc.ci available).
    nstates : int
        Number of states (1 for SS, >1 for MS/XMS).
    method : str
        "SS", "MS", or "XMS".
    iroot : int
        Target root for gradient (0-indexed).
    ipea_shift : float
        IPEA shift (default 0.0; standard Molcas value is 0.25).
    imag_shift : float
        Imaginary shift for intruder state removal.
    real_shift : float
        Real level shift.
    tol : float
        Solver convergence tolerance.
    maxiter : int
        Maximum solver iterations.
    max_memory_mb : float
        Memory limit for RDM construction.
    threshold : float
        S-metric diagonal threshold (Molcas THRSHN).
    threshold_s : float
        Scaled-S eigenvalue threshold (Molcas THRSHS).
    ci_strategy : {"auto", "mc", "guga_resolve"}
        CI-vector source policy:
        - ``"auto"``: use ``mc.ci`` if already CSF vectors (length ``drt.ncsf``), else
          fall back to GUGA CASCI re-solve.
        - ``"mc"``: require ``mc.ci`` to be CSF vectors; fail fast otherwise.
        - ``"guga_resolve"``: always re-solve CASCI in the GUGA CSF basis.
    quasi_canonical_active : bool
        If True, apply quasi-canonicalization aids in the working MO gauge:
        - SS: solve in a state-specific quasi-canonical orbital basis.
        - MS: build full Heff with ket-column, state-specific
          quasi-canonical solves and rotated transition metrics.
        - XMS: solve in a common quasi-canonical basis and include
          the model-space reference rotation term.
    verbose : int
        Verbosity level.

    Returns
    -------
    CASPT2Result with energies and amplitudes.
    """
    method = method.upper().strip()
    if method not in ("SS", "MS", "XMS"):
        raise ValueError(f"method must be 'SS', 'MS', or 'XMS', got '{method}'")
    ci_strategy_norm = str(ci_strategy).strip().lower()
    if ci_strategy_norm not in ("auto", "mc", "guga_resolve"):
        raise ValueError(
            "ci_strategy must be one of: 'auto', 'mc', 'guga_resolve' "
            f"(got {ci_strategy!r})"
        )
    use_active_qc = bool(quasi_canonical_active)

    # Extract orbital information from mc object
    mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
    ncas = int(mc.ncas)
    ncore = int(mc.ncore)
    nmo = int(mo_coeff.shape[1])
    nvirt = nmo - ncore - ncas

    if verbose >= 1:
        print(f"CASPT2 Driver: method={method}, nstates={nstates}")
        print(f"  ncore={ncore}, ncas={ncas}, nvirt={nvirt}, nmo={nmo}")

    pt2_backend_norm = str(pt2_backend).strip().lower()
    if pt2_backend_norm not in ("cpu", "cuda", "cupy", "gpu"):
        raise ValueError("pt2_backend must be one of: 'cpu', 'cuda'")

    # End-to-end CUDA convenience switch for SS-CASPT2.
    if bool(cuda_e2e) and pt2_backend_norm in ("cuda", "cupy", "gpu"):
        integrals_backend = "df"
        fock_backend = "df"
        rdm_backend = "cuda"
        cuda_mode = "strict"
        heff_backend = "cuda"

    integrals_backend_norm = str(integrals_backend).strip().lower()
    if integrals_backend_norm not in ("fulleri", "df"):
        raise ValueError("integrals_backend must be one of: 'fulleri', 'df'")

    fock_backend_norm = str(fock_backend).strip().lower()
    if fock_backend_norm not in ("fulleri", "df"):
        raise ValueError("fock_backend must be one of: 'fulleri', 'df'")

    rdm_backend_norm = str(rdm_backend).strip().lower()
    if rdm_backend_norm not in ("cpu", "cuda"):
        raise ValueError("rdm_backend must be one of: 'cpu', 'cuda'")

    heff_backend_norm = str(heff_backend).strip().lower()
    if heff_backend_norm not in ("cpu", "cuda"):
        raise ValueError("heff_backend must be one of: 'cpu', 'cuda'")

    if integrals_backend_norm == "df" and pt2_backend_norm not in ("cuda", "cupy", "gpu"):
        raise NotImplementedError("integrals_backend='df' is currently supported only for pt2_backend='cuda'")
    if fock_backend_norm == "df" and pt2_backend_norm not in ("cuda", "cupy", "gpu"):
        raise NotImplementedError("fock_backend='df' is currently supported only for pt2_backend='cuda'")
    if rdm_backend_norm == "cuda" and pt2_backend_norm not in ("cuda", "cupy", "gpu"):
        raise NotImplementedError("rdm_backend='cuda' is currently supported only for pt2_backend='cuda'")
    if heff_backend_norm == "cuda" and pt2_backend_norm not in ("cuda", "cupy", "gpu"):
        raise NotImplementedError("heff_backend='cuda' requires pt2_backend='cuda' (SS must export GPU row-dots)")

    if integrals_backend_norm == "df" and fock_backend_norm != "df":
        raise ValueError(
            "integrals_backend='df' requires fock_backend='df' (no full eri_mo available)."
        )

    # Build MO integrals. For DF-only SS CUDA, avoid `ao2mo.full`.
    need_full_eri = bool(integrals_backend_norm == "fulleri")
    if method != "SS" and integrals_backend_norm == "df" and heff_backend_norm != "cuda":
        raise ValueError(
            "integrals_backend='df' requires heff_backend='cuda' for MS/XMS "
            "(CPU Heff requires full eri_mo)."
        )
    if method != "SS" and integrals_backend_norm == "df" and use_active_qc:
        raise NotImplementedError(
            "integrals_backend='df' MS/XMS is currently supported only with quasi_canonical_active=False."
        )

    if need_full_eri:
        h1e_mo, eri_mo = _build_mo_integrals(mc)
    else:
        h1e_mo = _build_h1e_mo(mc)
        eri_mo = np.empty((0,), dtype=np.float64)

    # Build superindex map
    smap = build_superindex(ncore, ncas, nvirt)

    # DF pair blocks for GPU RHS build (orbitals only; state-independent).
    df_blocks = None
    if pt2_backend_norm in ("cuda", "cupy", "gpu"):
        from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks  # noqa: PLC0415
        from asuka.mrpt2.df_pair_block import DFPairBlock, build_df_pair_blocks  # noqa: PLC0415

        mo_core = mo_coeff[:, :ncore]
        mo_act = mo_coeff[:, ncore : ncore + ncas]
        mo_virt = mo_coeff[:, ncore + ncas :]

        # Skip empty blocks (e.g. nvirt=0), but keep output shapes consistent by
        # materializing empty DFPairBlocks with the shared naux.
        pairs: list[tuple[np.ndarray, np.ndarray]] = []
        labels: list[str] = []
        if fock_backend_norm == "df":
            if int(ncore) > 0:
                pairs.append((mo_core, mo_core))
                labels.append("ii")
            if int(nvirt) > 0:
                pairs.append((mo_virt, mo_virt))
                labels.append("ab")
        if int(ncore) > 0 and int(ncas) > 0:
            pairs.append((mo_core, mo_act))
            labels.append("it")
        if int(ncore) > 0 and int(nvirt) > 0:
            pairs.append((mo_core, mo_virt))
            labels.append("ia")
        if int(nvirt) > 0 and int(ncas) > 0:
            pairs.append((mo_virt, mo_act))
            labels.append("at")
        if int(ncas) > 0:
            pairs.append((mo_act, mo_act))
            labels.append("tu")

        built = build_df_pair_blocks(
            mc.mol,
            pairs,
            auxbasis=auxbasis,
            max_memory=int(max_memory_mb),
            verbose=max(0, int(verbose) - 1),
            compute_pair_norm=False,
        )
        blocks_by_label = {k: v for k, v in zip(labels, built)}
        naux = 0
        for blk in built:
            if int(getattr(blk, "naux", 0)) > 0:
                naux = int(blk.naux)
                break

        def _empty_block(nx: int, ny: int) -> DFPairBlock:
            nx = int(nx)
            ny = int(ny)
            l_full = np.zeros((nx * ny, int(naux)), dtype=np.float64, order="C")
            return DFPairBlock(nx=nx, ny=ny, l_full=l_full, pair_norm=None)

        l_it = blocks_by_label.get("it", _empty_block(ncore, ncas))
        l_ia = blocks_by_label.get("ia", _empty_block(ncore, nvirt))
        l_at = blocks_by_label.get("at", _empty_block(nvirt, ncas))
        l_tu = blocks_by_label.get("tu", _empty_block(ncas, ncas))
        l_ii = blocks_by_label.get("ii", _empty_block(ncore, ncore)) if fock_backend_norm == "df" else None
        l_ab = blocks_by_label.get("ab", _empty_block(nvirt, nvirt)) if fock_backend_norm == "df" else None

        # Guardrail: the virt-virt DF block can be very large for big basis sets.
        if fock_backend_norm == "df" and l_ab is not None and int(nvirt) > 0 and int(naux) > 0:
            ab_bytes = float(int(nvirt) * int(nvirt)) * float(int(naux)) * 8.0
            if ab_bytes > float(max_memory_mb) * 1e6:
                raise MemoryError(
                    "DF Fock build requested full virt-virt DF block (ab), but the implied "
                    f"allocation is ~{ab_bytes/1e6:.1f} MB (nvirt={int(nvirt)}, naux={int(naux)}). "
                    "Increase max_memory_mb or switch to fock_backend='fulleri'."
                )

        df_blocks = CASPT2DFBlocks(l_it=l_it, l_ia=l_ia, l_at=l_at, l_tu=l_tu, l_ii=l_ii, l_ab=l_ab)

    # Get CI vectors and RDMs
    drt, ci_vectors, dm1_list, dm2_list, dm3_list, e_ref_list, ci_strategy_used = _get_ci_and_rdms(
        mc,
        nstates=nstates,
        max_memory_mb=max_memory_mb,
        ci_strategy=ci_strategy_norm,
        rdm_backend=rdm_backend_norm,
        cuda_device=cuda_device,
        verbose=verbose,
    )

    e_nuc = float(mc.mol.energy_nuc())
    if nstates > 1:
        dm1_avg = dm1_list[0] * 0.0
        for d in dm1_list:
            dm1_avg = dm1_avg + d
        dm1_avg = dm1_avg / float(nstates)
    else:
        dm1_avg = dm1_list[0]

    def _build_fock(dm1_act: Any) -> CASPT2Fock:
        if fock_backend_norm == "df":
            if df_blocks is None:
                raise RuntimeError("internal error: df_blocks missing for fock_backend='df'")
            from asuka.caspt2.fock_df import build_caspt2_fock_df  # noqa: PLC0415

            return build_caspt2_fock_df(
                h1e_mo,
                df_blocks,
                dm1_act,
                int(ncore),
                int(ncas),
                int(nvirt),
                e_nuc=float(e_nuc),
                xp="cupy",
            )
        return build_caspt2_fock(h1e_mo, eri_mo, dm1_act, ncore, ncas, nvirt, e_nuc=e_nuc)

    # For XMS, OpenMolcas uses a state-averaged Fock operator to build the
    # model-space H0 and rotate the references. For MS, OpenMolcas defaults
    # to a state-specific Fock (see CASPT2 output: "Fock operator state-specific").
    fock_sa = _build_fock(dm1_avg)
    if method == "SS":
        fock = _build_fock(dm1_list[iroot])
    elif method == "MS":
        fock = [_build_fock(dm1_list[i]) for i in range(nstates)]
    else:  # XMS
        fock = fock_sa

    if method == "SS":
        base_ss = _run_ss(
            smap, fock, eri_mo, dm1_list[iroot], dm2_list[iroot], dm3_list[iroot],
            drt=drt, ci_csf=ci_vectors[iroot], max_memory_mb=max_memory_mb,
            e_ref=e_ref_list[iroot],
            ipea_shift=ipea_shift, imag_shift=imag_shift, real_shift=real_shift,
            tol=tol, maxiter=maxiter, threshold=threshold, threshold_s=threshold_s,
            verbose=verbose,
            pt2_backend=pt2_backend_norm,
            cuda_device=cuda_device,
            cuda_mode=cuda_mode,
            cuda_f3_cache_bytes=cuda_f3_cache_bytes,
            cuda_profile=cuda_profile,
            df_blocks=df_blocks,
        )
        if not use_active_qc:
            return base_ss

        fock_i = _build_fock(dm1_list[iroot])
        u_active = _active_qc_rotation_from_fock(
            fock_i=fock_i, ncore=ncore, ncas=ncas, nmo=nmo
        )
        mc_qc = _build_casci_with_rotated_mo(
            mc,
            u_active=u_active,
            nstates=nstates,
            verbose=verbose,
        )
        qc_ss = caspt2_from_mc(
            mc_qc,
            nstates=nstates,
            method="SS",
            iroot=iroot,
            pt2_backend=pt2_backend_norm,
            cuda_device=cuda_device,
            cuda_mode=cuda_mode,
            cuda_f3_cache_bytes=cuda_f3_cache_bytes,
            cuda_profile=cuda_profile,
            integrals_backend=integrals_backend_norm,
            fock_backend=fock_backend_norm,
            rdm_backend=rdm_backend_norm,
            cuda_e2e=bool(cuda_e2e),
            auxbasis=auxbasis,
            ipea_shift=ipea_shift,
            imag_shift=imag_shift,
            real_shift=real_shift,
            tol=tol,
            maxiter=maxiter,
            max_memory_mb=max_memory_mb,
            threshold=threshold,
            threshold_s=threshold_s,
            ci_strategy=ci_strategy_norm,
            quasi_canonical_active=False,
            verbose=verbose,
        )
        breakdown = dict(qc_ss.breakdown) if isinstance(qc_ss.breakdown, dict) else {}
        breakdown.update(
            {
                "active_qc_applied": True,
                "active_qc_mode": "state_specific_ss",
            }
        )
        return CASPT2Result(
            e_ref=qc_ss.e_ref,
            e_pt2=qc_ss.e_pt2,
            e_tot=qc_ss.e_tot,
            heff=qc_ss.heff,
            ueff=qc_ss.ueff,
            amplitudes=qc_ss.amplitudes,
            method=qc_ss.method,
            breakdown=breakdown,
        )
    elif method == "MS":
        base_ms = _run_ms(
            smap, fock, eri_mo, ci_vectors, dm1_list, dm2_list, dm3_list,
            nstates=nstates, iroot=iroot, drt=drt, max_memory_mb=max_memory_mb,
            e_ref_list=e_ref_list,
            heff_backend=heff_backend_norm,
            ipea_shift=ipea_shift, imag_shift=imag_shift, real_shift=real_shift,
            tol=tol, maxiter=maxiter, threshold=threshold, threshold_s=threshold_s,
            verbose=verbose,
            pt2_backend=pt2_backend_norm,
            cuda_device=cuda_device,
            cuda_mode=cuda_mode,
            cuda_f3_cache_bytes=cuda_f3_cache_bytes,
            cuda_profile=cuda_profile,
            df_blocks=df_blocks,
        )
        if not use_active_qc:
            return base_ms

        qc_ms = _run_ms_active_qc(
            mc=mc,
            smap=smap,
            h1e_mo=h1e_mo,
            eri_mo=eri_mo,
            drt=drt,
            ci_vectors=ci_vectors,
            dm1_list=dm1_list,
            e_ref_list=e_ref_list,
            ncore=ncore,
            ncas=ncas,
            nvirt=nvirt,
            nstates=nstates,
            ipea_shift=ipea_shift,
            imag_shift=imag_shift,
            real_shift=real_shift,
            tol=tol,
            maxiter=maxiter,
            max_memory_mb=max_memory_mb,
            threshold=threshold,
            threshold_s=threshold_s,
            ci_strategy=ci_strategy_norm,
            verbose=verbose,
        )
        breakdown = dict(qc_ms.breakdown) if isinstance(qc_ms.breakdown, dict) else {}
        breakdown.update(
            {
                "active_qc_applied": True,
                "active_qc_mode": "state_specific_ms_full_heff",
            }
        )
        return CASPT2Result(
            e_ref=qc_ms.e_ref,
            e_pt2=qc_ms.e_pt2,
            e_tot=qc_ms.e_tot,
            heff=qc_ms.heff,
            ueff=qc_ms.ueff,
            amplitudes=qc_ms.amplitudes,
            method="MS",
            breakdown=breakdown,
        )
    else:  # XMS
        if use_active_qc:
            return _run_xms_active_qc(
                mc=mc,
                smap=smap,
                fock_sa=fock_sa,
                eri_mo=eri_mo,
                ci_vectors=ci_vectors,
                dm1_list=dm1_list,
                dm2_list=dm2_list,
                dm3_list=dm3_list,
                nstates=nstates,
                drt=drt,
                ncore=ncore,
                ncas=ncas,
                nvirt=nvirt,
                iroot=iroot,
                e_ref_list=e_ref_list,
                ci_strategy=ci_strategy_norm,
                ci_strategy_used=ci_strategy_used,
                ipea_shift=ipea_shift,
                imag_shift=imag_shift,
                real_shift=real_shift,
                tol=tol,
                maxiter=maxiter,
                threshold=threshold,
                threshold_s=threshold_s,
                max_memory_mb=max_memory_mb,
                verbose=verbose,
            )
        return _run_xms(
            smap, fock, eri_mo, ci_vectors, dm1_list, dm2_list, dm3_list,
            nstates=nstates, iroot=iroot, drt=drt, nish=ncore, nash=ncas,
            e_ref_list=e_ref_list,
            ci_strategy_used=ci_strategy_used,
            heff_backend=heff_backend_norm,
            rdm_backend=rdm_backend_norm,
            ipea_shift=ipea_shift, imag_shift=imag_shift, real_shift=real_shift,
            tol=tol, maxiter=maxiter, threshold=threshold, threshold_s=threshold_s,
            max_memory_mb=max_memory_mb, verbose=verbose,
            pt2_backend=pt2_backend_norm,
            cuda_device=cuda_device,
            cuda_mode=cuda_mode,
            cuda_f3_cache_bytes=cuda_f3_cache_bytes,
            cuda_profile=cuda_profile,
            df_blocks=df_blocks,
        )


def _build_h1e_mo(mc: Any) -> np.ndarray:
    """Build one-electron MO integrals (core Hamiltonian) from a PySCF mc object."""

    mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
    if mo_coeff.ndim != 2:
        raise ValueError("mc.mo_coeff must be a 2D array")

    h1e_ao = mc.get_hcore()
    h1e_mo = mo_coeff.T @ np.asarray(h1e_ao, dtype=np.float64) @ mo_coeff
    return np.asarray(h1e_mo, dtype=np.float64, order="C")


def _build_mo_integrals(mc: Any) -> tuple[np.ndarray, np.ndarray]:
    """Build full MO integrals from PySCF mc object."""
    from pyscf import ao2mo  # noqa: PLC0415

    mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
    nmo = mo_coeff.shape[1]

    h1e_mo = _build_h1e_mo(mc)

    # Two-electron integrals in MO basis (chemists' notation)
    eri_mo = ao2mo.full(mc.mol, mo_coeff, compact=False).reshape(nmo, nmo, nmo, nmo)

    return (
        np.asarray(h1e_mo, dtype=np.float64, order="C"),
        np.asarray(eri_mo, dtype=np.float64, order="C"),
    )


def _mc_nelec_twos(mc: Any) -> tuple[int, int, int]:
    """Return (na, nb, twos) for a CASCI/CASSCF object."""
    nelecas = mc.nelecas
    if isinstance(nelecas, (list, tuple)):
        na, nb = int(nelecas[0]), int(nelecas[1])
    else:
        na = nb = int(nelecas) // 2
    return na, nb, int(na - nb)


def _active_qc_rotation_from_fock(
    *,
    fock_i: CASPT2Fock,
    ncore: int,
    ncas: int,
    nmo: int,
) -> np.ndarray:
    """Build an MO-space unitary that quasi-canonicalizes orbital subspaces.

    OpenMolcas ORBCTL diagonalizes inactive, active, and secondary blocks.
    In shared-reference parity runs (notably C2H4), diagonalizing only the
    active block leaves non-active Fock blocks non-diagonal, which perturbs
    `EPSI/EPSE`-driven denominator terms.
    """
    u = np.eye(int(nmo), dtype=np.float64)
    ncore_i = int(ncore)
    ncas_i = int(ncas)
    nmo_i = int(nmo)
    nvirt_i = nmo_i - ncore_i - ncas_i

    def _diag_subspace(start: int, size: int) -> None:
        if int(size) <= 0:
            return
        blk = np.asarray(
            fock_i.fifa[int(start): int(start) + int(size), int(start): int(start) + int(size)],
            dtype=np.float64,
        )
        blk = 0.5 * (blk + blk.T)
        _eval, evec = np.linalg.eigh(blk)
        u[int(start): int(start) + int(size), int(start): int(start) + int(size)] = np.asarray(
            evec,
            dtype=np.float64,
        )

    _diag_subspace(0, ncore_i)
    _diag_subspace(ncore_i, ncas_i)
    _diag_subspace(ncore_i + ncas_i, nvirt_i)
    return u


def _build_casci_with_rotated_mo(
    mc: Any,
    *,
    u_active: np.ndarray,
    nstates: int,
    solver_nroots: int | None = None,
    verbose: int,
) -> Any:
    """Create a CASCI object with active-rotated orbitals and GUGA roots."""
    from pyscf import mcscf  # noqa: PLC0415
    from pyscf import scf  # noqa: PLC0415
    from asuka.solver import GUGAFCISolver  # noqa: PLC0415

    mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
    u_active = np.asarray(u_active, dtype=np.float64)
    if u_active.shape != (mo_coeff.shape[1], mo_coeff.shape[1]):
        raise ValueError(
            "u_active has wrong shape: "
            f"{u_active.shape} (expected {(mo_coeff.shape[1], mo_coeff.shape[1])})"
        )
    mo_rot = mo_coeff @ u_active

    mf_rot = scf.RHF(mc.mol)
    mf_rot.conv_tol = 1e-12
    mf_rot.verbose = max(0, int(verbose) - 1)
    mf_rot.kernel()
    mf_rot.mo_coeff = np.asarray(mo_rot, dtype=np.float64, order="C")
    if hasattr(mc, "mo_occ") and mc.mo_occ is not None:
        mf_rot.mo_occ = np.asarray(mc.mo_occ)
    if hasattr(mc, "mo_energy") and mc.mo_energy is not None:
        mo_energy = np.asarray(mc.mo_energy)
        if mo_energy.shape == (mo_coeff.shape[1],):
            mf_rot.mo_energy = mo_energy

    mc_rot = mcscf.CASCI(mf_rot, int(mc.ncas), mc.nelecas)
    # Preserve the caller-provided quasi-canonical MO gauge. OpenMolcas keeps
    # the transformed orbital orientation from ORBCTL; if CASCI canonicalizes
    # core/virtual spaces again, EPSI/EPSE parity can drift (notably in C2H4).
    if hasattr(mc_rot, "canonicalization"):
        mc_rot.canonicalization = False
    _na, _nb, twos = _mc_nelec_twos(mc)
    nroots = int(nstates) if solver_nroots is None else int(solver_nroots)
    if nroots < int(nstates):
        raise ValueError(
            f"solver_nroots must be >= nstates ({nroots} < {int(nstates)})"
        )
    mc_rot.fcisolver = GUGAFCISolver(twos=int(twos), nroots=nroots)
    mc_rot.verbose = max(0, int(verbose) - 1)
    mc_rot.kernel(mo_coeff=np.asarray(mo_rot, dtype=np.float64, order="C"))
    return mc_rot


def _state_specific_qc_ss_energies(
    *,
    mc: Any,
    h1e_mo: np.ndarray,
    eri_mo: np.ndarray,
    dm1_list: list[np.ndarray],
    ncore: int,
    ncas: int,
    nvirt: int,
    nstates: int,
    ipea_shift: float,
    imag_shift: float,
    real_shift: float,
    tol: float,
    maxiter: int,
    max_memory_mb: float,
    threshold: float,
    threshold_s: float,
    ci_strategy: Literal["auto", "mc", "guga_resolve"],
    verbose: int,
) -> list[float]:
    """Compute per-state SS-CASPT2 energies in state-specific active QC bases."""
    e_nuc = float(mc.mol.energy_nuc())
    nmo = int(np.asarray(mc.mo_coeff).shape[1])
    qc_ss: list[float] = []
    for i in range(int(nstates)):
        fock_i = build_caspt2_fock(
            h1e_mo,
            eri_mo,
            dm1_list[i],
            int(ncore),
            int(ncas),
            int(nvirt),
            e_nuc=e_nuc,
        )
        u_active = _active_qc_rotation_from_fock(
            fock_i=fock_i,
            ncore=int(ncore),
            ncas=int(ncas),
            nmo=int(nmo),
        )
        mc_qc = _build_casci_with_rotated_mo(
            mc,
            u_active=u_active,
            nstates=int(nstates),
            verbose=int(verbose),
        )
        ss_i = caspt2_from_mc(
            mc_qc,
            nstates=int(nstates),
            method="SS",
            iroot=int(i),
            ipea_shift=float(ipea_shift),
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            max_memory_mb=float(max_memory_mb),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            ci_strategy=ci_strategy,
            quasi_canonical_active=False,
            verbose=int(verbose),
        )
        qc_ss.append(float(ss_i.e_tot))
    return qc_ss


def _rotate_active_dm1(u_active: np.ndarray, dm1: np.ndarray) -> np.ndarray:
    """Rotate active-space 1-body tensor under a common orbital transform."""
    u = np.asarray(u_active, dtype=np.float64)
    d1 = np.asarray(dm1, dtype=np.float64)
    return np.asarray(np.einsum("at,bu,ab->tu", u, u, d1, optimize=True), dtype=np.float64, order="C")


def _rotate_active_dm2(u_active: np.ndarray, dm2: np.ndarray) -> np.ndarray:
    """Rotate active-space 2-body tensor under a common orbital transform."""
    u = np.asarray(u_active, dtype=np.float64)
    d2 = np.asarray(dm2, dtype=np.float64)
    return np.asarray(
        np.einsum("at,bu,cv,dx,abcd->tuvx", u, u, u, u, d2, optimize=True),
        dtype=np.float64,
        order="C",
    )


def _rotate_active_dm3(u_active: np.ndarray, dm3: np.ndarray) -> np.ndarray:
    """Rotate active-space 3-body tensor under a common orbital transform."""
    u = np.asarray(u_active, dtype=np.float64)
    d3 = np.asarray(dm3, dtype=np.float64)
    return np.asarray(
        np.einsum("at,bu,cv,dx,ey,fz,abcdef->tuvxyz", u, u, u, u, u, u, d3, optimize=True),
        dtype=np.float64,
        order="C",
    )


def _apply_xms_reference_rotation(
    *,
    heff: np.ndarray,
    e_ref_list: list[float],
    u0: np.ndarray,
) -> np.ndarray:
    """Apply OpenMolcas XMS model-space reference rotation contribution."""
    h = np.asarray(heff, dtype=np.float64)
    u = np.asarray(u0, dtype=np.float64)
    e_ref = np.asarray(e_ref_list, dtype=np.float64).ravel()
    nstates = int(e_ref.size)
    if h.shape != (nstates, nstates):
        raise ValueError(
            f"heff shape {h.shape} does not match e_ref size {nstates}"
        )
    if u.shape != (nstates, nstates):
        raise ValueError(
            f"u0 shape {u.shape} does not match e_ref size {nstates}"
        )
    d_ref = np.diag(e_ref)
    # OpenMolcas transmat path: replace the unrotated diagonal reference block
    # by the model-space rotated reference block.
    h_ref_rot = np.asarray(u.T @ d_ref @ u, dtype=np.float64)
    return np.asarray(h - d_ref + h_ref_rot, dtype=np.float64, order="C")


def _match_roots_by_reference_energy(
    *,
    target_e_ref: list[float],
    current_e_ref: list[float],
) -> list[int]:
    """Return permutation mapping target root index -> current root index."""
    target = [float(x) for x in target_e_ref]
    current = [float(x) for x in current_e_ref]
    n = len(target)
    if len(current) != n:
        raise ValueError(
            f"target/current root counts differ: {n} vs {len(current)}"
        )

    # Small nstates in practice: brute-force gives stable, globally optimal mapping.
    if n <= 8:
        best_cost = None
        best_perm = None
        for perm in itertools.permutations(range(n)):
            cost = sum(abs(current[perm[i]] - target[i]) for i in range(n))
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_perm = perm
        if best_perm is None:
            raise RuntimeError("root matching failed to find a permutation")
        return [int(x) for x in best_perm]

    # Fallback for larger root spaces: greedy nearest-neighbor assignment.
    unused = set(range(n))
    perm_out: list[int] = []
    for i in range(n):
        j_best = min(unused, key=lambda j: abs(current[j] - target[i]))
        perm_out.append(int(j_best))
        unused.remove(j_best)
    return perm_out


def _active_qc_root_search_nroots(nstates: int) -> int:
    """Return a bounded oversolve root count for active-QC re-solves."""
    n = int(nstates)
    if n <= 1:
        return n
    return n + min(3, n)


def _match_subset_roots_by_reference_energy(
    *,
    target_e_ref: list[float],
    candidate_e_ref: list[float],
) -> list[int]:
    """Match target references to a subset of candidate roots by energy distance."""
    target = [float(x) for x in target_e_ref]
    candidate = [float(x) for x in candidate_e_ref]
    n_target = len(target)
    n_candidate = len(candidate)
    if n_candidate < n_target:
        raise ValueError(
            f"candidate roots ({n_candidate}) fewer than target roots ({n_target})"
        )
    if n_candidate == n_target:
        return _match_roots_by_reference_energy(
            target_e_ref=target,
            current_e_ref=candidate,
        )

    # Typical shared-reference runs use small root spaces; brute-force gives
    # globally optimal subset+assignment for near-degenerate states.
    if n_target <= 6 and n_candidate <= 8:
        best_cost = None
        best_perm = None
        for perm in itertools.permutations(range(n_candidate), n_target):
            cost = sum(abs(candidate[perm[i]] - target[i]) for i in range(n_target))
            if (best_cost is None) or (cost < best_cost):
                best_cost = cost
                best_perm = perm
        if best_perm is None:
            raise RuntimeError("subset root matching failed to find a permutation")
        return [int(x) for x in best_perm]

    # Fallback for larger spaces: greedy nearest-neighbor assignment.
    unused = set(range(n_candidate))
    perm_out: list[int] = []
    for i in range(n_target):
        j_best = min(unused, key=lambda j: abs(candidate[j] - target[i]))
        perm_out.append(int(j_best))
        unused.remove(j_best)
    return perm_out


def _reorder_state_payload_by_perm(
    *,
    perm: list[int],
    ci_vectors: list[np.ndarray],
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    e_ref_list: list[float],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float]]:
    """Apply one root permutation consistently across CI/RDM/reference payload."""
    return (
        [ci_vectors[k] for k in perm],
        [dm1_list[k] for k in perm],
        [dm2_list[k] for k in perm],
        [dm3_list[k] for k in perm],
        [float(e_ref_list[k]) for k in perm],
    )


def _run_ms_active_qc(
    *,
    mc: Any,
    smap: SuperindexMap,
    h1e_mo: np.ndarray,
    eri_mo: np.ndarray,
    drt: Any,
    ci_vectors: list[np.ndarray],
    dm1_list: list[np.ndarray],
    e_ref_list: list[float],
    ncore: int,
    ncas: int,
    nvirt: int,
    nstates: int,
    ipea_shift: float,
    imag_shift: float,
    real_shift: float,
    tol: float,
    maxiter: int,
    max_memory_mb: float,
    threshold: float,
    threshold_s: float,
    ci_strategy: Literal["auto", "mc", "guga_resolve"],
    verbose: int,
) -> CASPT2Result:
    """Build full MS Heff using ket-column, state-specific active QC solves."""
    from asuka.rdm.rdm123 import _trans_rdm123_pyscf  # noqa: PLC0415

    e_nuc = float(mc.mol.energy_nuc())
    nmo = int(np.asarray(mc.mo_coeff).shape[1])
    nstates_i = int(nstates)
    nstates_qc = min(
        _active_qc_root_search_nroots(nstates_i),
        int(getattr(drt, "ncsf", nstates_i)),
    )
    heff = np.zeros((nstates_i, nstates_i), dtype=np.float64)
    all_amplitudes: list[list[np.ndarray]] = [[] for _ in range(nstates_i)]

    # Reference transition densities in the original MO basis.
    trans_orig: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for i in range(nstates_i):
        for j in range(nstates_i):
            if i == j:
                continue
            trans_orig[(i, j)] = _trans_rdm123_pyscf(
                drt,
                ci_vectors[i],
                ci_vectors[j],
                max_memory_mb=float(max_memory_mb),
                reorder=True,
                reorder_mode="molcas",
            )

    # Build once for state-specific QC rotation drivers.
    fock_orig = [
        build_caspt2_fock(
            h1e_mo,
            eri_mo,
            dm1_list[i],
            int(ncore),
            int(ncas),
            int(nvirt),
            e_nuc=e_nuc,
        )
        for i in range(nstates_i)
    ]

    for j in range(nstates_i):
        u_full = _active_qc_rotation_from_fock(
            fock_i=fock_orig[j],
            ncore=int(ncore),
            ncas=int(ncas),
            nmo=int(nmo),
        )
        act = slice(int(ncore), int(ncore) + int(ncas))
        u_active = np.asarray(u_full[act, act], dtype=np.float64, order="C")

        mc_qc = _build_casci_with_rotated_mo(
            mc,
            u_active=u_full,
            nstates=nstates_i,
            solver_nroots=nstates_qc,
            verbose=int(verbose),
        )
        h1e_qc, eri_qc = _build_mo_integrals(mc_qc)
        drt_qc, ci_qc, dm1_qc, dm2_qc, dm3_qc, e_ref_qc, _strategy_qc = _get_ci_and_rdms(
            mc_qc,
            nstates=nstates_qc,
            max_memory_mb=float(max_memory_mb),
            ci_strategy=ci_strategy,
            verbose=int(verbose),
        )
        perm = _match_subset_roots_by_reference_energy(
            target_e_ref=e_ref_list,
            candidate_e_ref=e_ref_qc,
        )
        ci_qc, dm1_qc, dm2_qc, dm3_qc, e_ref_qc = _reorder_state_payload_by_perm(
            perm=perm,
            ci_vectors=ci_qc,
            dm1_list=dm1_qc,
            dm2_list=dm2_qc,
            dm3_list=dm3_qc,
            e_ref_list=e_ref_qc,
        )

        fock_j_qc = build_caspt2_fock(
            h1e_qc,
            eri_qc,
            dm1_qc[j],
            int(ncore),
            int(ncas),
            int(nvirt),
            e_nuc=e_nuc,
        )
        ci_ctx_j = CASPT2CIContext(
            drt=drt_qc,
            ci_csf=ci_qc[j],
            max_memory_mb=float(max_memory_mb),
        )
        ss_j = caspt2_energy_ss(
            smap,
            fock_j_qc,
            eri_qc,
            dm1_qc[j],
            dm2_qc[j],
            dm3_qc[j],
            e_ref=float(e_ref_qc[j]),
            ci_context=ci_ctx_j,
            ipea_shift=float(ipea_shift),
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            verbose=int(verbose),
        )
        heff[j, j] = float(ss_j.e_tot)
        all_amplitudes[j] = ss_j.amplitudes

        nactel_qc = int(getattr(drt_qc, "nelec"))
        rhs_blocks: list[np.ndarray] = []
        t_blocks: list[np.ndarray] = []
        ci_context = CASPT2CIContext(drt=drt_qc, ci_csf=ci_qc[j])
        for case in range(1, 14):
            nasup = int(smap.nasup[case - 1])
            nisup = int(smap.nisup[case - 1])
            if nasup == 0 or nisup == 0:
                rhs_blocks.append(np.empty((0, 0), dtype=np.float64))
                t_blocks.append(np.empty((0, 0), dtype=np.float64))
                continue

            smat = build_smat(case, smap, dm1_qc[j], dm2_qc[j], dm3_qc[j])
            bmat = build_bmat(case, smap, fock_j_qc, dm1_qc[j], dm2_qc[j], dm3_qc[j], ci_context=ci_context)
            decomp = sbdiag(
                smat, bmat, threshold_norm=float(threshold), threshold_s=float(threshold_s)
            )
            rhs_raw = build_rhs(
                case,
                smap,
                fock_j_qc,
                eri_qc,
                dm1_qc[j],
                dm2_qc[j],
                nactel=nactel_qc,
            ).reshape(nasup, nisup)
            rhs_blocks.append(rhs_raw)

            amps_j = ss_j.amplitudes[case - 1]
            if decomp.nindep == 0 or amps_j.size == 0:
                t_blocks.append(np.empty((0, 0), dtype=np.float64))
            else:
                expected_size = int(decomp.nindep) * int(nisup)
                if int(amps_j.size) != expected_size:
                    raise ValueError(
                        "MS active-QC amplitude dimension mismatch for "
                        f"ket state {j} case {case}: amps {amps_j.size} vs expected {expected_size}"
                    )
                amps_mat = amps_j.reshape(decomp.nindep, nisup)
                t_blocks.append(np.asarray(decomp.transform @ amps_mat, dtype=np.float64, order="C"))

        for i in range(nstates_i):
            if i == j:
                continue
            tdm1_orig, tdm2_orig, tdm3_orig = trans_orig[(i, j)]
            tdm1 = _rotate_active_dm1(u_active, tdm1_orig)
            tdm2 = _rotate_active_dm2(u_active, tdm2_orig)
            tdm3 = _rotate_active_dm3(u_active, tdm3_orig)
            ovl_ij = float(np.dot(ci_vectors[i], ci_vectors[j]))

            coupling = 0.0
            for case in range(1, 14):
                rhs_j = rhs_blocks[case - 1]
                t_j = t_blocks[case - 1]
                if rhs_j.size == 0 or t_j.size == 0:
                    continue
                if rhs_j.shape != t_j.shape:
                    raise ValueError(
                        "MS active-QC raw block shape mismatch for "
                        f"state-pair ({i},{j}) case {case}: rhs {rhs_j.shape} vs t {t_j.shape}"
                    )
                row_dots = rhs_j @ t_j.T
                coupling += hcoup_case_contribution(
                    case,
                    smap,
                    row_dots,
                    tdm1,
                    tdm2,
                    tdm3,
                    ovl=ovl_ij,
                )
            # Degenerate model-space pairs are unitary-ambiguous; avoid introducing
            # arbitrary splitting from ket-local QC root rotations.
            if abs(float(e_ref_list[i]) - float(e_ref_list[j])) < 1.0e-8:
                heff[i, j] = 0.0
            else:
                heff[i, j] = float(coupling)

    ms_energies, ueff = diagonalize_heff(heff)
    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(ms_energies[i] - float(e_ref_list[i])) for i in range(nstates_i)],
        e_tot=ms_energies.tolist(),
        heff=heff,
        ueff=ueff,
        amplitudes=all_amplitudes,
        method="MS",
        breakdown={
            "qc_ss_energies": [float(heff[i, i]) for i in range(nstates_i)],
            "ms_energies": ms_energies.tolist(),
            "active_qc_root_search_nroots": int(nstates_qc),
        },
    )


def _run_xms_active_qc(
    *,
    mc: Any,
    smap: SuperindexMap,
    fock_sa: CASPT2Fock,
    eri_mo: np.ndarray,
    ci_vectors: list[np.ndarray],
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    nstates: int,
    drt: Any,
    ncore: int,
    ncas: int,
    nvirt: int,
    iroot: int,
    e_ref_list: list[float],
    ci_strategy: Literal["auto", "mc", "guga_resolve"],
    ci_strategy_used: str,
    ipea_shift: float,
    imag_shift: float,
    real_shift: float,
    tol: float,
    maxiter: int,
    threshold: float,
    threshold_s: float,
    max_memory_mb: float,
    verbose: int,
) -> CASPT2Result:
    """Run XMS with common active QC basis and model-space reference rotation."""
    from asuka.rdm.rdm123 import _make_rdm123_pyscf  # noqa: PLC0415
    from asuka.rdm.rdm123 import _reorder_dm123_molcas  # noqa: PLC0415

    nstates_i = int(nstates)
    nstates_qc = min(
        _active_qc_root_search_nroots(nstates_i),
        int(getattr(drt, "ncsf", nstates_i)),
    )
    nmo = int(np.asarray(mc.mo_coeff).shape[1])
    e_nuc = float(mc.mol.energy_nuc())

    rotated_ci, u0, h0_model = xms_rotate_states(
        drt,
        ci_vectors,
        dm1_list,
        fock_sa,
        int(ncore),
        int(ncas),
        nstates_i,
        verbose=int(verbose),
    )
    _ = rotated_ci  # Rotation matrix is reused below for QC-root combination.

    u_full = _active_qc_rotation_from_fock(
        fock_i=fock_sa,
        ncore=int(ncore),
        ncas=int(ncas),
        nmo=int(nmo),
    )
    mc_qc = _build_casci_with_rotated_mo(
        mc,
        u_active=u_full,
        nstates=nstates_i,
        solver_nroots=nstates_qc,
        verbose=int(verbose),
    )
    h1e_qc, eri_qc = _build_mo_integrals(mc_qc)
    drt_qc, ci_qc, _dm1_qc, _dm2_qc, _dm3_qc, _eref_qc, _strategy_qc = _get_ci_and_rdms(
        mc_qc,
        nstates=nstates_qc,
        max_memory_mb=float(max_memory_mb),
        ci_strategy=ci_strategy,
        verbose=int(verbose),
    )
    perm = _match_subset_roots_by_reference_energy(
        target_e_ref=e_ref_list,
        candidate_e_ref=_eref_qc,
    )
    ci_qc = [ci_qc[k] for k in perm]

    # Anchor with the non-QC XMS Heff so QC refinement remains phase-robust.
    baseline_xms = _run_xms(
        smap,
        fock_sa,
        eri_mo,
        ci_vectors,
        dm1_list,
        dm2_list,
        dm3_list,
        nstates=nstates_i,
        iroot=int(iroot),
        drt=drt,
        nish=int(ncore),
        nash=int(ncas),
        max_memory_mb=float(max_memory_mb),
        e_ref_list=e_ref_list,
        ci_strategy_used=str(ci_strategy_used),
        ipea_shift=float(ipea_shift),
        imag_shift=float(imag_shift),
        real_shift=float(real_shift),
        tol=float(tol),
        maxiter=int(maxiter),
        threshold=float(threshold),
        threshold_s=float(threshold_s),
        verbose=max(0, int(verbose) - 1),
    )
    heff_anchor = np.asarray(baseline_xms.heff, dtype=np.float64)

    # Apply the model-space XMS rotation in the QC CSF basis.
    u0_arr = np.asarray(u0, dtype=np.float64)
    sign_patterns: list[tuple[float, ...]]
    if nstates_i <= 6:
        sign_patterns = [
            tuple([1.0] + [float(x) for x in bits])
            for bits in itertools.product((1.0, -1.0), repeat=max(0, nstates_i - 1))
        ]
    else:
        sign_patterns = [tuple([1.0] * nstates_i)]

    best_score: float | None = None
    best_heff_corr: np.ndarray | None = None
    best_ms_result: CASPT2Result | None = None

    for signs in sign_patterns:
        ci_signed = [
            float(signs[i]) * np.asarray(ci_qc[i], dtype=np.float64)
            for i in range(nstates_i)
        ]
        rot_ci_qc: list[np.ndarray] = []
        for j in range(nstates_i):
            vec = np.zeros_like(ci_signed[0], dtype=np.float64)
            for i in range(nstates_i):
                vec += float(u0_arr[i, j]) * ci_signed[i]
            rot_ci_qc.append(np.asarray(vec, dtype=np.float64, order="C"))

        rot_dm1: list[np.ndarray] = []
        rot_dm2: list[np.ndarray] = []
        rot_dm3: list[np.ndarray] = []
        for i, c in enumerate(rot_ci_qc):
            if int(verbose) >= 1:
                print(f"  Computing RDMs for XMS+QC rotated state {i} (sign {signs})...")
            dm1, dm2, dm3 = _make_rdm123_pyscf(
                drt_qc,
                c,
                max_memory_mb=float(max_memory_mb),
                reorder=False,
            )
            dm1, dm2, dm3 = _reorder_dm123_molcas(dm1, dm2, dm3, inplace=True)
            rot_dm1.append(dm1)
            rot_dm2.append(dm2)
            rot_dm3.append(dm3)

        fock_sa_qc = build_caspt2_fock(
            h1e_qc,
            eri_qc,
            sum(rot_dm1) / float(nstates_i),
            int(ncore),
            int(ncas),
            int(nvirt),
            e_nuc=e_nuc,
        )

        ms_result = _run_ms(
            smap,
            fock_sa_qc,
            eri_qc,
            rot_ci_qc,
            rot_dm1,
            rot_dm2,
            rot_dm3,
            nstates=nstates_i,
            iroot=int(iroot),
            drt=drt_qc,
            e_ref_list=e_ref_list,
            max_memory_mb=float(max_memory_mb),
            ipea_shift=float(ipea_shift),
            imag_shift=float(imag_shift),
            real_shift=float(real_shift),
            tol=float(tol),
            maxiter=int(maxiter),
            threshold=float(threshold),
            threshold_s=float(threshold_s),
            verbose=int(verbose),
        )

        heff_corr = _apply_xms_reference_rotation(
            heff=np.asarray(ms_result.heff, dtype=np.float64),
            e_ref_list=e_ref_list,
            u0=u0_arr,
        )
        score = float(np.linalg.norm(heff_corr - heff_anchor))
        if (best_score is None) or (score < best_score):
            best_score = score
            best_heff_corr = heff_corr
            best_ms_result = ms_result

    if best_heff_corr is None or best_ms_result is None:
        raise RuntimeError("XMS active-QC sign selection failed")

    heff_corr = best_heff_corr
    ms_result = best_ms_result
    xms_energies, ueff = diagonalize_heff(heff_corr)

    breakdown = dict(ms_result.breakdown) if isinstance(ms_result.breakdown, dict) else {}
    breakdown.update(
        {
            "h0_model": np.asarray(h0_model, dtype=np.float64).tolist(),
            "u0": u0_arr.tolist(),
            "ci_strategy_used": str(ci_strategy_used),
            "reference_rotation_applied": True,
            "active_qc_applied": True,
            "active_qc_mode": "xms_common_qc",
            "active_qc_root_search_nroots": int(nstates_qc),
            "qc_sign_selection_score": float(best_score) if best_score is not None else None,
            "qc_sign_patterns_tested": int(len(sign_patterns)),
        }
    )

    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(xms_energies[i] - float(e_ref_list[i])) for i in range(nstates_i)],
        e_tot=xms_energies.tolist(),
        heff=heff_corr,
        ueff=ueff,
        amplitudes=ms_result.amplitudes,
        method="XMS",
        breakdown=breakdown,
    )


def _normalize_csf_ci_vector(ci_like: Any, *, ncsf: int, state_index: int) -> np.ndarray:
    """Validate one CSF-basis CI root as a flat float64 vector of length ``ncsf``."""
    arr = np.asarray(ci_like, dtype=np.float64)
    if arr.ndim != 1 or int(arr.size) != int(ncsf):
        raise ValueError(
            f"state {state_index}: expected a 1D CSF CI vector of length {ncsf}, "
            f"got shape {arr.shape}"
        )
    return np.asarray(arr, dtype=np.float64, order="C")


def _extract_mc_csf_ci_vectors(mc: Any, *, nstates: int, ncsf: int) -> list[np.ndarray]:
    """Extract CSF CI vectors from ``mc.ci`` when they are already in CSF basis."""
    if not hasattr(mc, "ci"):
        raise ValueError("mc object has no 'ci' attribute")

    ci_obj = mc.ci
    roots_raw: list[Any]
    if nstates == 1:
        roots_raw = [ci_obj]
    elif isinstance(ci_obj, (list, tuple)):
        roots_raw = list(ci_obj)
    elif isinstance(ci_obj, np.ndarray) and ci_obj.ndim == 2 and int(ci_obj.shape[0]) == int(nstates):
        roots_raw = [ci_obj[i] for i in range(int(nstates))]
    else:
        raise ValueError(
            "mc.ci is not a multi-root CSF list/array compatible with requested nstates"
        )

    if len(roots_raw) != int(nstates):
        raise ValueError(
            f"mc.ci has {len(roots_raw)} roots but nstates={int(nstates)} was requested"
        )

    return [
        _normalize_csf_ci_vector(root, ncsf=int(ncsf), state_index=i)
        for i, root in enumerate(roots_raw)
    ]


def _extract_mc_reference_energies(mc: Any, *, nstates: int) -> list[float]:
    """Extract state reference energies from ``mc.e_states``/``mc.e_tot``."""
    if int(nstates) == 1:
        if hasattr(mc, "e_states") and mc.e_states is not None:
            arr = np.asarray(mc.e_states, dtype=np.float64).ravel()
            if arr.size >= 1:
                return [float(arr[0])]
        if hasattr(mc, "e_tot"):
            return [float(mc.e_tot)]
        raise ValueError("mc object has neither e_states nor e_tot for SS reference energy")

    arr = None
    if hasattr(mc, "e_states") and mc.e_states is not None:
        arr = np.asarray(mc.e_states, dtype=np.float64).ravel()
    elif hasattr(mc, "e_tot"):
        arr = np.asarray(mc.e_tot, dtype=np.float64).ravel()
    if arr is None:
        raise ValueError("multi-root reference energies require mc.e_states or mc.e_tot")
    if int(arr.size) < int(nstates):
        raise ValueError(
            f"mc reference energies have {int(arr.size)} roots but nstates={int(nstates)} was requested"
        )
    return [float(x) for x in arr[: int(nstates)]]


def _resolve_ci_vectors_guga(
    mc: Any,
    *,
    nstates: int,
    twos: int,
    nelec: int,
    verbose: int,
    nroots_search: int | None = None,
    match_target_eref: list[float] | None = None,
) -> tuple[list[np.ndarray], list[float]]:
    """Re-solve active-space CASCI with GUGA solver and return CSF CI vectors + references.

    If `nroots_search` > `nstates`, we "oversolve" the CI problem and then select
    a subset of `nstates` roots. This is useful when requesting only `nstates`
    roots causes the CI solver to converge to an undesired excited manifold.
    """
    from pyscf import ao2mo  # noqa: PLC0415
    from asuka.solver import GUGAFCISolver  # noqa: PLC0415

    ncas = int(mc.ncas)
    ncore = int(mc.ncore)

    mo_coeff = np.asarray(mc.mo_coeff, dtype=np.float64)
    mo_cas = mo_coeff[:, ncore:ncore + ncas]
    h1e_ao = mc.get_hcore()
    h1e_cas = mo_cas.T @ h1e_ao @ mo_cas

    if ncore > 0:
        from pyscf import scf  # noqa: PLC0415

        mo_core = mo_coeff[:, :ncore]
        dm_core = 2.0 * mo_core @ mo_core.T
        vj_core, vk_core = scf.hf.get_jk(mc.mol, dm_core)
        vhf_core = vj_core - 0.5 * vk_core
        h1e_cas = h1e_cas + mo_cas.T @ vhf_core @ mo_cas
    else:
        dm_core = None
        vhf_core = None

    eri_cas = ao2mo.full(mc.mol, mo_cas, compact=False).reshape(ncas, ncas, ncas, ncas)

    e_core = float(mc.mol.energy_nuc())
    if ncore > 0 and dm_core is not None and vhf_core is not None:
        e_core += np.einsum("ij,ji->", h1e_ao, dm_core)
        e_core += 0.5 * np.einsum("ij,ji->", vhf_core, dm_core)

    nstates_i = int(nstates)
    nroots_i = int(nroots_search) if nroots_search is not None else nstates_i
    if nroots_i < nstates_i:
        nroots_i = nstates_i

    solver = GUGAFCISolver(twos=int(twos), nroots=int(nroots_i))
    e_guga, ci_guga = solver.kernel(h1e_cas, eri_cas, ncas, int(nelec))

    if nroots_i == 1:
        ci_all = [np.asarray(ci_guga, dtype=np.float64).ravel()]
        e_all = np.asarray([float(e_guga) + e_core], dtype=np.float64)
    else:
        ci_all = [np.asarray(c, dtype=np.float64).ravel() for c in ci_guga]
        e_arr = np.asarray(e_guga, dtype=np.float64).ravel()
        if int(e_arr.size) != int(nroots_i):
            raise ValueError(
                f"GUGA re-solve returned {int(e_arr.size)} roots, expected {int(nroots_i)}"
            )
        e_all = np.asarray([float(e) + e_core for e in e_arr], dtype=np.float64)

    if nroots_i > nstates_i:
        idx: list[int] | None = None
        if match_target_eref is not None and len(match_target_eref) == nstates_i:
            try:
                idx = _match_subset_roots_by_reference_energy(
                    target_e_ref=[float(x) for x in match_target_eref],
                    candidate_e_ref=[float(x) for x in e_all.tolist()],
                )
            except Exception:
                idx = None
        if idx is None:
            idx = [int(i) for i in np.argsort(e_all)[:nstates_i].tolist()]
        ci_csf = [np.asarray(ci_all[i], dtype=np.float64).ravel() for i in idx]
        e_ref = [float(e_all[i]) for i in idx]
    else:
        ci_csf = [np.asarray(ci_all[i], dtype=np.float64).ravel() for i in range(nstates_i)]
        e_ref = [float(e_all[i]) for i in range(nstates_i)]

    if verbose >= 1:
        for i, e in enumerate(e_ref):
            print(f"  GUGA re-solve state {i}: E_CAS = {e:.10f}")

    return ci_csf, e_ref


def _get_ci_and_rdms(
    mc: Any,
    *,
    nstates: int,
    max_memory_mb: float,
    ci_strategy: Literal["auto", "mc", "guga_resolve"],
    rdm_backend: Literal["cpu", "cuda"] = "cpu",
    cuda_device: int | None = None,
    rdm_profile: dict[str, float] | None = None,
    verbose: int,
) -> tuple[
    Any,
    list[np.ndarray],
    list[Any],
    list[Any],
    list[Any],
    list[float],
    str,
]:
    """Extract CI vectors and compute active RDMs in OpenMolcas CASPT2 convention.

    OpenMolcas stores the internally contracted densities as the *irreducible*
    unitary-group generators (``GAMMA1/2/3`` in `mkfg3.f`), not the raw product
    densities from an alternating creation/annihilation string.

    We therefore:
    1) build raw product densities with `reorder=False`, then
    2) apply OpenMolcas-style delta corrections + pair-permutation symmetrization
       (same logic as `mktg3.f` / `mkfg3.f`) to obtain Molcas-compatible
       ``dm2``/``dm3`` tensors.

    CI source is controlled by ``ci_strategy``:
    - ``"mc"``: require ``mc.ci`` to already be CSF vectors.
    - ``"guga_resolve"``: always re-solve CASCI in GUGA CSF basis.
    - ``"auto"``: use ``mc.ci`` when compatible, otherwise fall back to GUGA re-solve.

    RDM backend is controlled by ``rdm_backend``:
    - ``"cpu"``: build RDMs on CPU (default; deterministic).
    - ``"cuda"``: build RDMs on GPU via `asuka.cuda.rdm123_gpu` and return CuPy arrays.
    """
    from asuka.cuguga.drt import build_drt  # noqa: PLC0415
    rdm_backend_norm = str(rdm_backend).strip().lower()
    if rdm_backend_norm not in ("cpu", "cuda"):
        raise ValueError("rdm_backend must be 'cpu' or 'cuda'")

    if rdm_backend_norm == "cpu":
        from asuka.rdm.rdm123 import _make_rdm123_pyscf  # noqa: PLC0415
        from asuka.rdm.rdm123 import _reorder_dm123_molcas  # noqa: PLC0415
    else:
        from asuka.cuda.rdm123_gpu import make_rdm123_molcas_cuda  # noqa: PLC0415

        # Validate CuPy import early for clearer errors.
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("rdm_backend='cuda' requires CuPy to be installed") from e
        if cuda_device is not None:
            cp.cuda.Device(int(cuda_device)).use()

    ncas = int(mc.ncas)
    nelecas = mc.nelecas
    if isinstance(nelecas, (list, tuple)):
        na, nb = int(nelecas[0]), int(nelecas[1])
    else:
        na = nb = int(nelecas) // 2

    twos = na - nb
    nelec = na + nb

    drt = build_drt(norb=ncas, nelec=nelec, twos_target=twos)
    ci_vectors: list[np.ndarray] | None = None
    e_ref_list: list[float] | None = None
    strategy_used: str | None = None

    # Optional systematic root search: oversolve active-space CI and select roots.
    # This is useful for matching OpenMolcas fixtures in cases where a small
    # nroots solve converges to a different excited manifold.
    raw_nroots_search = os.environ.get("ASUKA_CASPT2_ROOT_SEARCH_NROOTS", "").strip()
    nroots_search = 0
    if raw_nroots_search:
        try:
            nroots_search = int(raw_nroots_search)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(
                "ASUKA_CASPT2_ROOT_SEARCH_NROOTS must be an integer "
                f"(got {raw_nroots_search!r})"
            ) from exc
    if int(nroots_search) > int(nstates):
        policy_raw = os.environ.get("ASUKA_CASPT2_ROOT_SEARCH_POLICY", "lowest")
        policy = str(policy_raw).strip().lower() or "lowest"
        if policy not in {"lowest", "match_mc"}:
            raise ValueError(
                "ASUKA_CASPT2_ROOT_SEARCH_POLICY must be one of: lowest, match_mc "
                f"(got {policy_raw!r})"
            )
        target = None
        if policy == "match_mc":
            try:
                target = _extract_mc_reference_energies(mc, nstates=nstates)
            except Exception:
                target = None
        ci_vectors, e_ref_list = _resolve_ci_vectors_guga(
            mc,
            nstates=nstates,
            twos=twos,
            nelec=nelec,
            verbose=verbose,
            nroots_search=int(nroots_search),
            match_target_eref=target,
        )
        strategy_used = f"guga_resolve_rootsearch({int(nroots_search)},{policy})"

    if ci_vectors is None and e_ref_list is None and ci_strategy in ("auto", "mc"):
        try:
            ci_vectors = _extract_mc_csf_ci_vectors(mc, nstates=nstates, ncsf=int(drt.ncsf))
            e_ref_list = _extract_mc_reference_energies(mc, nstates=nstates)
            strategy_used = "mc"
            if verbose >= 1:
                print("  CI strategy: using mc.ci CSF vectors directly.")
        except ValueError as err:
            if ci_strategy == "mc":
                raise ValueError(
                    "ci_strategy='mc' requires mc.ci to be CSF-basis vectors with length drt.ncsf. "
                    "Use ci_strategy='guga_resolve' or provide a GUGA-based MC object."
                ) from err
            if verbose >= 1:
                print(
                    "  CI strategy: mc.ci is not CSF-compatible; "
                    "falling back to guga_resolve."
                )

    if ci_vectors is None or e_ref_list is None:
        ci_vectors, e_ref_list = _resolve_ci_vectors_guga(
            mc,
            nstates=nstates,
            twos=twos,
            nelec=nelec,
            verbose=verbose,
        )
        strategy_used = "guga_resolve"

    # Compute RDMs in E-operator product convention (reorder=False)
    dm1_list, dm2_list, dm3_list = [], [], []
    for i, c in enumerate(ci_vectors):
        if rdm_backend_norm == "cuda":
            if verbose >= 1:
                print(f"  Computing RDMs for state {i} on GPU (Molcas convention)...")
            dm1, dm2, dm3 = make_rdm123_molcas_cuda(drt, c, device=cuda_device, profile=rdm_profile)
        else:
            if verbose >= 1:
                print(f"  Computing RDMs for state {i} (E-operator convention)...")
            dm1, dm2, dm3 = _make_rdm123_pyscf(drt, c, max_memory_mb=max_memory_mb, reorder=False)
            dm1, dm2, dm3 = _reorder_dm123_molcas(dm1, dm2, dm3, inplace=True)
        dm1_list.append(dm1)
        dm2_list.append(dm2)
        dm3_list.append(dm3)

    if strategy_used is None:
        raise RuntimeError("internal error: CI strategy was not resolved")

    return drt, ci_vectors, dm1_list, dm2_list, dm3_list, e_ref_list, strategy_used


def _run_ss(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    *,
    drt: Any,
    ci_csf: np.ndarray,
    max_memory_mb: float,
    e_ref: float,
    **kwargs,
) -> CASPT2Result:
    """Run SS-CASPT2."""
    ci_context = CASPT2CIContext(drt=drt, ci_csf=ci_csf, max_memory_mb=max_memory_mb)
    result = caspt2_energy_ss(
        smap, fock, eri_mo, dm1, dm2, dm3, e_ref, ci_context=ci_context, **kwargs
    )
    return CASPT2Result(
        e_ref=result.e_ref,
        e_pt2=result.e_pt2,
        e_tot=result.e_tot,
        amplitudes=result.amplitudes,
        method="SS",
        breakdown=result.breakdown,
    )


def _run_ms(
    smap: SuperindexMap,
    fock: CASPT2Fock | list[CASPT2Fock],
    eri_mo: np.ndarray,
    ci_vectors: list[np.ndarray],
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    *,
    nstates: int,
    iroot: int,
    drt: Any,
    max_memory_mb: float,
    e_ref_list: list[float] | None = None,
    heff_backend: str = "cpu",
    **kwargs,
) -> CASPT2Result:
    """Run MS-CASPT2."""
    verbose = kwargs.get("verbose", 0)
    heff_backend_norm = str(heff_backend).strip().lower()
    if heff_backend_norm not in ("cpu", "cuda"):
        raise ValueError("heff_backend must be 'cpu' or 'cuda'")

    if e_ref_list is None:
        raise ValueError("_run_ms requires per-state e_ref_list aligned with ci_vectors.")

    def _fock_for_state(state: int) -> CASPT2Fock:
        if isinstance(fock, (list, tuple)):
            return fock[state]
        return fock

    # Run SS-CASPT2 for each state
    ss_results = []
    all_amplitudes = []
    for i in range(nstates):
        if verbose >= 1:
            print(f"\n--- SS-CASPT2 for state {i} ---")
        ci_context = CASPT2CIContext(drt=drt, ci_csf=ci_vectors[i], max_memory_mb=max_memory_mb)
        fock_i = _fock_for_state(i)
        res = caspt2_energy_ss(
            smap, fock_i, eri_mo, dm1_list[i], dm2_list[i], dm3_list[i],
            e_ref=e_ref_list[i], ci_context=ci_context, **kwargs,
            store_row_dots=bool(heff_backend_norm == "cuda"),
        )
        ss_results.append(res)
        all_amplitudes.append(res.amplitudes)

    # Build effective Hamiltonian
    heff_profile: dict[str, Any] | None = None
    if heff_backend_norm == "cuda":
        from asuka.caspt2.cuda.multistate_cuda import build_heff_cuda  # noqa: PLC0415

        heff_profile = {} if bool(kwargs.get("cuda_profile", False)) else None
        heff = build_heff_cuda(
            nstates,
            ss_results,
            ci_vectors,
            drt,
            smap,
            device=kwargs.get("cuda_device", None),
            profile=heff_profile,
            verbose=int(verbose),
        )
    else:
        heff = build_heff(
            nstates, ss_results, ci_vectors, drt, smap, fock, eri_mo,
            dm1_list, dm2_list, dm3_list,
            threshold=float(kwargs.get("threshold", 1e-10)),
            threshold_s=float(kwargs.get("threshold_s", 1e-8)),
            max_memory_mb=float(max_memory_mb),
            verbose=verbose,
        )

    # Diagonalize Heff
    ms_energies, ueff = diagonalize_heff(heff)

    if verbose >= 1:
        print(f"\nMS-CASPT2 Energies:")
        for i in range(nstates):
            print(f"  State {i}: {ms_energies[i]:.10f}")

    breakdown = {
        "ss_energies": [r.e_tot for r in ss_results],
        "ms_energies": ms_energies.tolist(),
        "heff_backend": str(heff_backend_norm),
    }
    if heff_backend_norm == "cuda" and heff_profile is not None:
        breakdown["heff_cuda_profile"] = heff_profile

    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(ms_energies[i] - e_ref_list[i]) for i in range(nstates)],
        e_tot=ms_energies.tolist(),
        heff=heff,
        ueff=ueff,
        amplitudes=all_amplitudes,
        method="MS",
        breakdown=breakdown,
    )


def _run_xms(
    smap: SuperindexMap,
    fock: CASPT2Fock,
    eri_mo: np.ndarray,
    ci_vectors: list[np.ndarray],
    dm1_list: list[np.ndarray],
    dm2_list: list[np.ndarray],
    dm3_list: list[np.ndarray],
    *,
    nstates: int,
    iroot: int,
    drt: Any,
    nish: int,
    nash: int,
    max_memory_mb: float,
    e_ref_list: list[float],
    ci_strategy_used: str,
    heff_backend: str = "cpu",
    rdm_backend: str = "cpu",
    **kwargs,
) -> CASPT2Result:
    """Run XMS-CASPT2."""
    verbose = kwargs.get("verbose", 0)
    heff_backend_norm = str(heff_backend).strip().lower()
    if heff_backend_norm not in ("cpu", "cuda"):
        raise ValueError("heff_backend must be 'cpu' or 'cuda'")
    rdm_backend_norm = str(rdm_backend).strip().lower()
    if rdm_backend_norm not in ("cpu", "cuda"):
        raise ValueError("rdm_backend must be 'cpu' or 'cuda'")

    # XMS rotation of reference states
    dm1_for_xms = dm1_list
    if rdm_backend_norm == "cuda":
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("rdm_backend='cuda' requires CuPy") from e
        # xms_rotate_states is CPU-only; convert diagonal dm1 blocks (small) once.
        dm1_for_xms = [np.asarray(cp.asnumpy(d), dtype=np.float64, order="C") for d in dm1_list]
    rotated_ci, u0, h0_model = xms_rotate_states(
        drt, ci_vectors, dm1_for_xms, fock, nish, nash, nstates, verbose=verbose,
    )

    # Recompute RDMs for rotated states (Molcas conventions).
    rot_dm1_list, rot_dm2_list, rot_dm3_list = [], [], []
    if rdm_backend_norm == "cuda":
        from asuka.cuda.rdm123_gpu import make_rdm123_molcas_cuda  # noqa: PLC0415

        rdm_profile = {} if bool(kwargs.get("cuda_profile", False)) else None
        for i, c in enumerate(rotated_ci):
            if verbose >= 1:
                print(f"  Computing RDMs for XMS-rotated state {i} on GPU (Molcas convention)...")
            dm1, dm2, dm3 = make_rdm123_molcas_cuda(
                drt,
                c,
                device=kwargs.get("cuda_device", None),
                profile=rdm_profile,
            )
            rot_dm1_list.append(dm1)
            rot_dm2_list.append(dm2)
            rot_dm3_list.append(dm3)
    else:
        # CPU reference path
        from asuka.rdm.rdm123 import _make_rdm123_pyscf  # noqa: PLC0415
        from asuka.rdm.rdm123 import _reorder_dm123_molcas  # noqa: PLC0415

        for i, c in enumerate(rotated_ci):
            if verbose >= 1:
                print(f"  Computing RDMs for XMS-rotated state {i}...")
            dm1, dm2, dm3 = _make_rdm123_pyscf(drt, c, max_memory_mb=max_memory_mb, reorder=False)
            dm1, dm2, dm3 = _reorder_dm123_molcas(dm1, dm2, dm3, inplace=True)
            rot_dm1_list.append(dm1)
            rot_dm2_list.append(dm2)
            rot_dm3_list.append(dm3)

    # Run MS-CASPT2 with rotated states
    ms_result = _run_ms(
        smap, fock, eri_mo, rotated_ci, rot_dm1_list, rot_dm2_list, rot_dm3_list,
        nstates=nstates, iroot=iroot, drt=drt, e_ref_list=e_ref_list,
        heff_backend=heff_backend_norm,
        max_memory_mb=max_memory_mb, **kwargs,
    )
    heff_corr = _apply_xms_reference_rotation(
        heff=np.asarray(ms_result.heff, dtype=np.float64),
        e_ref_list=e_ref_list,
        u0=np.asarray(u0, dtype=np.float64),
    )
    xms_energies, ueff = diagonalize_heff(heff_corr)

    breakdown = dict(ms_result.breakdown) if isinstance(ms_result.breakdown, dict) else {}
    breakdown.update(
        {
            "h0_model": np.asarray(h0_model, dtype=np.float64).tolist(),
            "u0": np.asarray(u0, dtype=np.float64).tolist(),
            "ci_strategy_used": str(ci_strategy_used),
            "reference_rotation_applied": True,
        }
    )

    return CASPT2Result(
        e_ref=e_ref_list,
        e_pt2=[float(xms_energies[i] - float(e_ref_list[i])) for i in range(int(nstates))],
        e_tot=xms_energies.tolist(),
        heff=heff_corr,
        ueff=ueff,
        amplitudes=ms_result.amplitudes,
        method="XMS",
        breakdown=breakdown,
    )
