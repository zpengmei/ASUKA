from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import time
from typing import Sequence

import numpy as np

from asuka.cuguga.drt import DRT, build_drt
from asuka.cuguga.state_cache import get_state_cache


@dataclass(frozen=True)
class MRCISDResult:
    """Result container for single-state scalar MRCISD calculations.

    Attributes
    ----------
    converged : bool
        Whether the Davidson solver converged.
    e_mrci : float
        Total MRCISD energy (including frozen core and nuclear repulsion).
    ci : np.ndarray
        optimized CI vector in the MRCISD DRT basis.
    drt : DRT
        The Directed Robot Walk Graph (DRT) defining the CI basis.
    ci_ref0 : np.ndarray
        Reference vector in the MRCISD basis (embedded CAS CI vector).
    ref_idx : np.ndarray
        Indices of the reference configuration(s) in the MRCISD basis.
    diagnostics : dict[str, float]
        Additional diagnostics (weights, timings, etc.).
    """

    converged: bool
    e_mrci: float
    ci: np.ndarray
    drt: DRT
    ci_ref0: np.ndarray
    ref_idx: np.ndarray
    diagnostics: dict[str, float]


@dataclass(frozen=True)
class MRCISDResultMulti:
    """Result container for multi-state MRCISD calculations.

    Attributes
    ----------
    converged : np.ndarray
        Boolean array indicating convergence for each root. Shape: (nroots,).
    e_mrci : np.ndarray
        Total MRCISD energies for each root. Shape: (nroots,).
    ci : list[np.ndarray]
        List of optimized CI vectors, one per root.
    drt : DRT
        The Directed Robot Walk Graph (DRT) defining the CI basis.
    ci_ref0 : list[np.ndarray]
        List of reference vectors in the MRCISD basis.
    ref_idx : np.ndarray
        Indices of the reference configuration(s) in the MRCISD basis.
    overlap_ref_root : np.ndarray
        Overlap matrix between reference states and optimized roots.
        Shape: (nref, nroots), where element [k, i] is |<ref_k|root_i>|^2.
    diagnostics : list[dict[str, float]]
        List of per-root diagnostics.
    """

    converged: np.ndarray
    e_mrci: np.ndarray
    ci: list[np.ndarray]
    drt: DRT
    ci_ref0: list[np.ndarray]
    ref_idx: np.ndarray
    overlap_ref_root: np.ndarray
    diagnostics: list[dict[str, float]]


def build_drt_mrcisd(
    *,
    n_act: int,
    n_virt: int,
    nelec: int,
    twos: int,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    max_virt_e: int = 2,
) -> DRT:
    """Build a restricted DRT for uncontracted MRCISD in an [active][virtual] ordering.

    The restriction is enforced as a boundary constraint at k=n_act:
    ne >= nelec - max_virt_e  <=>  (# electrons in virtual) <= max_virt_e.

    Parameters
    ----------
    n_act : int
        Number of active orbitals.
    n_virt : int
        Number of virtual orbitals.
    nelec : int
        Total number of electrons.
    twos : int
        Spin multiplicity (2S).
    orbsym : Sequence[int] | None, optional
        Orbital symmetry labels.
    wfnsym : int | None, optional
        Wavefunction symmetry label.
    max_virt_e : int, optional
        Maximum number of electrons in the virtual space. Default is 2.

    Returns
    -------
    DRT
        The constructed Directed Robot Walk Graph.
    """

    n_act = int(n_act)
    n_virt = int(n_virt)
    nelec = int(nelec)
    twos = int(twos)
    max_virt_e = int(max_virt_e)

    if n_act < 0 or n_virt < 0:
        raise ValueError("n_act and n_virt must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")
    if max_virt_e < 0:
        raise ValueError("max_virt_e must be >= 0")

    norb = n_act + n_virt
    if max_virt_e > nelec:
        max_virt_e = nelec

    ne_min = max(0, nelec - max_virt_e)
    ne_constraints = {n_act: (ne_min, nelec)}

    return build_drt(
        norb=norb,
        nelec=nelec,
        twos_target=twos,
        orbsym=orbsym,
        wfnsym=wfnsym,
        ne_constraints=ne_constraints,
    )


def embed_cas_ci_into_mrcisd(
    *,
    drt_cas: DRT,
    drt_mrci: DRT,
    ci_cas: np.ndarray,
    n_virt: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Embed a CAS CI vector into an MRCISD basis by appending empty virtual steps.

    Parameters
    ----------
    drt_cas : DRT
        DRT for the CAS space.
    drt_mrci : DRT
        DRT for the MRCISD space.
    ci_cas : np.ndarray
        CAS CI vector to embed.
    n_virt : int
        Number of virtual orbitals appended to the CAS space.

    Returns
    -------
    ci0_mrci : np.ndarray
        Initial guess vector in the MRCISD basis (the embedded CAS vector).
    ci_ref0 : np.ndarray
        Fixed reference function in the MRCISD basis (same as ci0_mrci for a CAS reference).
    ref_idx : np.ndarray
        int32 indices of the full CAS block in the MRCISD basis (one per CAS CSF).
    """

    n_virt = int(n_virt)
    if n_virt < 0:
        raise ValueError("n_virt must be >= 0")

    ncsf_cas = int(drt_cas.ncsf)
    ncsf_mrci = int(drt_mrci.ncsf)
    if int(ci_cas.size) != ncsf_cas:
        raise ValueError(f"ci_cas has wrong size {int(ci_cas.size)} (expected {ncsf_cas})")

    if int(drt_mrci.norb) != int(drt_cas.norb) + n_virt:
        raise ValueError("drt_mrci.norb must equal drt_cas.norb + n_virt")

    ci0_mrci = np.zeros(ncsf_mrci, dtype=np.float64)
    ref_idx = np.empty(ncsf_cas, dtype=np.int32)

    n_act = int(drt_cas.norb)
    steps_full = np.empty(n_act + n_virt, dtype=np.int8)
    steps_full[n_act:] = 0  # E steps in the virtual tail

    ci_cas_f64 = np.asarray(ci_cas, dtype=np.float64)
    for j in range(ncsf_cas):
        steps_act = drt_cas.index_to_path(int(j))
        steps_full[:n_act] = steps_act
        J = int(drt_mrci.path_to_index(steps_full))
        ci0_mrci[J] = float(ci_cas_f64[j])
        ref_idx[j] = int(J)

    ci_ref0 = ci0_mrci.copy()
    return ci0_mrci, ci_ref0, ref_idx


def _make_hdiag_det_guess(drt: DRT, h1e, eri) -> np.ndarray:
    norb = int(drt.norb)
    ncsf = int(drt.ncsf)
    nelec_total = int(drt.nelec)
    twos = int(drt.twos_target)

    h1e = np.asarray(h1e, dtype=np.float64)
    if h1e.shape != (norb, norb):
        raise ValueError("h1e has wrong shape")

    try:
        from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
    except Exception:  # pragma: no cover
        DFMOIntegrals = None  # type: ignore[assignment]
        DeviceDFMOIntegrals = None  # type: ignore[assignment]

    if DFMOIntegrals is not None and isinstance(eri, DFMOIntegrals):
        l_full = np.asarray(eri.l_full, dtype=np.float64, order="C")
        pair_norm = np.asarray(eri.pair_norm, dtype=np.float64, order="C")

        diag_ids = (np.arange(norb, dtype=np.int32) * (norb + 1)).astype(np.int32, copy=False)
        l_diag = np.asarray(l_full[diag_ids], dtype=np.float64, order="C")
        eri_ppqq = l_diag @ l_diag.T  # (p p| q q)
        eri_pqqp = np.square(pair_norm.reshape(norb, norb))  # (p q| q p)
    elif DeviceDFMOIntegrals is not None and isinstance(eri, DeviceDFMOIntegrals):
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("DeviceDFMOIntegrals requires CuPy") from e

        nops = int(norb) * int(norb)

        if eri.eri_mat is not None:
            eri_mat_d = cp.asarray(eri.eri_mat, dtype=cp.float64)
            if eri_mat_d.ndim != 2 or tuple(map(int, eri_mat_d.shape)) != (nops, nops):
                raise ValueError("DeviceDFMOIntegrals.eri_mat has wrong shape")

            p = cp.arange(norb, dtype=cp.int32)
            idx_pp = p * (norb + 1)
            eri_ppqq_d = eri_mat_d[idx_pp[:, None], idx_pp[None, :]].copy()

            pp = p[:, None]
            qq = p[None, :]
            idx_pq = (pp * norb + qq).ravel()
            idx_qp = (qq * norb + pp).ravel()
            eri_pqqp_d = eri_mat_d[idx_pq, idx_qp].reshape(norb, norb).copy()

            eri_ppqq = np.asarray(cp.asnumpy(eri_ppqq_d), dtype=np.float64, order="C")
            eri_pqqp = np.asarray(cp.asnumpy(eri_pqqp_d), dtype=np.float64, order="C")
        else:
            if eri.l_full is None:
                raise ValueError("DeviceDFMOIntegrals must provide l_full or eri_mat for determinant-diagonal guess")
            l_full_d = cp.asarray(eri.l_full, dtype=cp.float64)
            if l_full_d.ndim != 2 or int(l_full_d.shape[0]) != nops:
                raise ValueError("DeviceDFMOIntegrals.l_full has wrong shape")

            if eri.pair_norm is not None:
                pair_norm_d = cp.asarray(eri.pair_norm, dtype=cp.float64)
            else:
                pair_norm_d = cp.linalg.norm(l_full_d, axis=1)

            diag_ids = (np.arange(norb, dtype=np.int64) * (norb + 1)).astype(np.int64, copy=False)
            l_diag_d = l_full_d[cp.asarray(diag_ids, dtype=cp.int64)]
            eri_ppqq_d = l_diag_d @ l_diag_d.T
            eri_pqqp_d = cp.square(pair_norm_d.reshape(norb, norb))

            eri_ppqq = np.asarray(cp.asnumpy(eri_ppqq_d), dtype=np.float64, order="C")
            eri_pqqp = np.asarray(cp.asnumpy(eri_pqqp_d), dtype=np.float64, order="C")
    else:
        from asuka.cuguga.oracle import _restore_eri_4d  # noqa: PLC0415

        eri4 = _restore_eri_4d(eri, norb).astype(np.float64, copy=False)
        eri_ppqq = np.einsum("iijj->ij", eri4)
        eri_pqqp = np.einsum("ijji->ij", eri4)

    cache = get_state_cache(drt)
    steps = np.asarray(cache.steps, dtype=np.int8)
    if steps.shape != (ncsf, norb):
        raise RuntimeError("internal error: invalid cached steps table shape")

    step_to_occ = np.asarray([0, 1, 1, 2], dtype=np.int8)
    occ = step_to_occ[steps]

    doubly = occ == 2
    singles = occ == 1

    neleca_det = (nelec_total + twos) // 2
    nelecb_det = nelec_total - neleca_det

    ndoubly = np.sum(doubly, axis=1, dtype=np.int32)
    alpha_need = np.asarray(neleca_det, dtype=np.int32) - ndoubly

    single_prefix = np.cumsum(singles, axis=1, dtype=np.int32)
    alpha_single = singles & (single_prefix <= alpha_need[:, None])
    beta_single = singles & (~alpha_single)

    alpha = (doubly | alpha_single).astype(np.float64)
    beta = (doubly | beta_single).astype(np.float64)
    n = alpha + beta

    h1e_diag = np.diag(h1e)
    hdiag = n @ h1e_diag

    tmp = n @ eri_ppqq
    hdiag += 0.5 * np.sum(tmp * n, axis=1)

    tmp_a = alpha @ eri_pqqp
    hdiag += -0.5 * np.sum(tmp_a * alpha, axis=1)

    tmp_b = beta @ eri_pqqp
    hdiag += -0.5 * np.sum(tmp_b * beta, axis=1)

    if nelecb_det < 0:  # pragma: no cover
        raise RuntimeError("internal error: invalid alpha/beta electron split")
    return np.asarray(hdiag, dtype=np.float64)


def mrcisd_kernel(
    *,
    h1e,
    eri,
    n_act: int,
    n_virt: int,
    nelec: int,
    twos: int,
    ci_cas: np.ndarray | Sequence[np.ndarray],
    nroots: int = 1,
    ecore: float = 0.0,
    orbsym_act: Sequence[int] | None = None,
    orbsym_corr: Sequence[int] | None = None,
    wfnsym: int | None = None,
    max_virt_e: int = 2,
    hop_backend: str | None = None,
    tol: float = 1e-10,
    lindep: float = 1e-14,
    max_cycle: int = 100,
    max_space: int = 12,
    max_memory: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = 1,
    precompute_epq: bool = True,
    verbose: int | None = None,
) -> MRCISDResult | MRCISDResultMulti:
    """Solve uncontracted, state-specific MRCISD in a restricted CSF/DRT basis.

    Parameters
    ----------
    h1e : np.ndarray
        One-electron integral matrix.
    eri : Any
        Two-electron integral payload (dense eri4, DF object, etc.).
    n_act : int
        Number of active orbitals.
    n_virt : int
        Number of virtual orbitals.
    nelec : int
        Total number of electrons.
    twos : int
        Spin multiplicity (2S).
    ci_cas : np.ndarray | Sequence[np.ndarray]
        Initial guess vector(s) from ID-CASCI.
    nroots : int, optional
        Number of roots to solve for. Default is 1.
    ecore : float, optional
        Frozen-core energy shift. Default is 0.0.
    orbsym_act : Sequence[int], optional
        Active orbital symmetry labels.
    orbsym_corr : Sequence[int], optional
        Correlated orbital symmetry labels (active + virtual).
    wfnsym : int, optional
        Wavefunction symmetry label.
    max_virt_e : int, optional
        Maximum number of electrons in the virtual space. Default is 2.
    hop_backend : str, optional
        Backend for Hamiltonian-vector product ("fast", "augmented", "cuda").
    tol : float, optional
        Convergence tolerance. Default is 1e-10.
    lindep : float, optional
        Linear dependency tolerance for Davidson. Default is 1e-14.
    max_cycle : int, optional
        Maximum number of iterations. Default is 100.
    max_space : int, optional
        Maximum subspace dimensionality. Default is 12.
    max_memory : float, optional
        Maximum memory usage in MB. Default is 4000.0.
    contract_nthreads : int, optional
        Number of threads for contraction. Default is 1.
    contract_blas_nthreads : int, optional
        Number of threads for BLAS. Default is 1.
    precompute_epq : bool, optional
        Whether to precompute EPQ actions. Default is True.
    verbose : int, optional
        Verbosity level.

    Returns
    -------
    MRCISDResult | MRCISDResultMulti
        Result object(s) containing energies and wavefunctions.

    Notes
    -----
    - ``hop_backend="fast"``: applies dense-intermediate contraction directly in the restricted DRT.
      (Exact only if the space is closed under needed generator actions).
    - ``hop_backend="augmented"``: applies contraction in an augmented DRT with ``max_virt_e+1``
      and projects back. Exact for "virtual electrons <= max_virt_e" truncation.
    """
    from asuka.cuguga.davidson import davidson1 as davidson1_sym  # noqa: PLC0415
    from asuka.cuguga.oracle import (  # noqa: PLC0415
        _get_epq_action_cache,
        _restore_eri_4d,
        occ_table,
        precompute_epq_actions,
    )

    # Auto-detect hop backend based on whether space is restricted
    if hop_backend is None:
        # A space is restricted only if max_virt_e < min(nelec, 2*n_virt)
        # Counterexample: If n_virt=1, max_virt_e=2 is NOT a restriction (since 2*1=2)
        n_act_int = int(n_act)
        n_virt_int = int(n_virt)
        nelec_int = int(nelec)
        max_virt_e_int = int(max_virt_e)
        restricted = (max_virt_e_int < min(nelec_int, 2 * n_virt_int))
        hop_backend = "augmented" if restricted else "fast"

    hop_backend = str(hop_backend).strip().lower()
    if hop_backend not in ("fast", "augmented", "cuda"):
        raise ValueError("hop_backend must be 'fast', 'augmented', or 'cuda'")

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")

    n_act = int(n_act)
    n_virt = int(n_virt)
    nelec = int(nelec)
    twos = int(twos)
    if n_act < 0 or n_virt < 0:
        raise ValueError("n_act and n_virt must be >= 0")
    if nelec < 0:
        raise ValueError("nelec must be >= 0")

    drt_cas = build_drt(norb=n_act, nelec=nelec, twos_target=twos, orbsym=orbsym_act, wfnsym=wfnsym)
    drt_mrci = build_drt_mrcisd(
        n_act=n_act,
        n_virt=n_virt,
        nelec=nelec,
        twos=twos,
        orbsym=orbsym_corr,
        wfnsym=wfnsym,
        max_virt_e=max_virt_e,
    )

    drt_hop = drt_mrci
    hop_map = None
    if hop_backend == "augmented":
        from asuka.mrci.projected_hop import build_subspace_map  # noqa: PLC0415

        drt_hop = build_drt_mrcisd(
            n_act=n_act,
            n_virt=n_virt,
            nelec=nelec,
            twos=twos,
            orbsym=orbsym_corr,
            wfnsym=wfnsym,
            max_virt_e=int(max_virt_e) + 1,
        )
        hop_map = build_subspace_map(drt_full=drt_hop, drt_sub=drt_mrci)

    if isinstance(ci_cas, (list, tuple)):
        ci_cas_list = [np.asarray(v, dtype=np.float64) for v in ci_cas]
    else:
        ci_cas_list = [np.asarray(ci_cas, dtype=np.float64)]
    if not ci_cas_list:
        raise ValueError("ci_cas must be non-empty")
    if len(ci_cas_list) < nroots:
        raise ValueError(
            f"Need at least nroots={nroots} initial guess vector(s) in ci_cas (got {len(ci_cas_list)}). "
            "Pass ci_cas as a list/tuple of CAS CI vectors."
        )

    # Embed reference guesses once: compute ref_idx mapping from the first vector,
    # then reuse it for the remaining guesses to avoid repeated path lookups.
    ci0_mrci0, ci_ref0_0, ref_idx = embed_cas_ci_into_mrcisd(
        drt_cas=drt_cas, drt_mrci=drt_mrci, ci_cas=ci_cas_list[0], n_virt=n_virt
    )
    ref_idx_i64 = np.asarray(ref_idx, dtype=np.int64)

    ci_ref0_list: list[np.ndarray] = [np.asarray(ci_ref0_0, dtype=np.float64)]
    x0: list[np.ndarray] = []

    norm0 = float(np.linalg.norm(ci0_mrci0))
    if norm0 <= 0.0:
        raise ValueError("embedded CAS vector has zero norm")
    x0.append(ci0_mrci0 / norm0)

    for guess in ci_cas_list[1:]:
        guess_f64 = np.asarray(guess, dtype=np.float64).ravel()
        if int(guess_f64.size) != int(drt_cas.ncsf):
            raise ValueError(
                f"ci_cas has wrong size {int(guess_f64.size)} (expected {int(drt_cas.ncsf)})"
            )
        ci0 = np.zeros(int(drt_mrci.ncsf), dtype=np.float64)
        ci0[ref_idx_i64] = guess_f64
        ci_ref0_list.append(ci0.copy())
        nrm = float(np.linalg.norm(ci0))
        if nrm <= 0.0:
            raise ValueError("embedded CAS vector has zero norm")
        x0.append(ci0 / nrm)

    if bool(precompute_epq):
        # The CUDA hop builds a combined E_pq action table directly on the device
        # and does not require the CPU-side per-(p,q) CSR cache. Skip this costly
        # setup by default for hop_backend="cuda".
        if hop_backend != "cuda" or bool(int(os.getenv("CUGUGA_MRCI_CUDA_PRECOMPUTE_EPQ", "0"))):
            from asuka.cuguga.oracle import precompute_epq_actions  # noqa: PLC0415

            precompute_epq_actions(drt_hop)

    hdiag = None

    def precond(dx, e, _x0):
        if hdiag is None:  # pragma: no cover
            raise RuntimeError("internal error: hdiag is not initialized")
        denom = hdiag - float(e)
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        return dx / denom

    contract_nthreads = int(contract_nthreads)
    if contract_nthreads < 1:
        raise ValueError("contract_nthreads must be >= 1")

    contract_executor: ThreadPoolExecutor | None = None
    if contract_nthreads > 1:
        contract_executor = ThreadPoolExecutor(max_workers=contract_nthreads)

    contract_ws = None
    try:
        from asuka.contract import ContractWorkspace  # noqa: PLC0415

        contract_ws = ContractWorkspace()
    except Exception:  # pragma: no cover
        contract_ws = None

    def hop(xs: list[np.ndarray]) -> list[np.ndarray]:
        if hop_backend == "augmented":
            if hop_map is None:  # pragma: no cover
                raise RuntimeError("internal error: missing augmented hop map")
            from asuka.mrci.projected_hop import projected_contract_h_csf_multi  # noqa: PLC0415

            return projected_contract_h_csf_multi(
                mapping=hop_map,
                h1e=h1e,
                eri=eri,
                xs_sub=xs,
                precompute_epq_full=False,
                nthreads=contract_nthreads,
                blas_nthreads=contract_blas_nthreads,
                executor=contract_executor,
                workspace=contract_ws,
            )

        try:
            from asuka.integrals.df_integrals import DFMOIntegrals  # noqa: PLC0415
        except Exception:  # pragma: no cover
            DFMOIntegrals = None  # type: ignore[assignment]

        if DFMOIntegrals is not None and isinstance(eri, DFMOIntegrals):
            from asuka.integrals.contract_df import contract_h_csf_multi_df as _contract_df  # noqa: PLC0415

            return _contract_df(
                drt_mrci,
                h1e,
                eri,
                xs,
                precompute_epq=False,
                nthreads=contract_nthreads,
                blas_nthreads=contract_blas_nthreads,
                executor=contract_executor,
                workspace=contract_ws,
            )

        from asuka.contract import contract_h_csf_multi as _contract_dense  # noqa: PLC0415

        return _contract_dense(
            drt_mrci,
            h1e,
            eri,
            xs,
            precompute_epq=False,
            nthreads=contract_nthreads,
            blas_nthreads=contract_blas_nthreads,
            executor=contract_executor,
            workspace=contract_ws,
        )

    if hop_backend == "cuda":
        import cupy as cp  # noqa: PLC0415
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            build_epq_action_table_combined_device,
            build_hdiag_det_guess_from_steps_inplace_device,
            make_device_drt,
            make_device_state_cache,
        )
        from asuka.cuda.cuda_davidson import davidson_sym_gpu  # noqa: PLC0415
        from asuka.cuda.mrci_hop import (  # noqa: PLC0415
            CudaMrciHopWorkspace,
            hop_cuda_epq_table,
            hop_cuda_projected,
        )

        hop_stage_profile = bool(int(os.getenv("CUGUGA_MRCI_CUDA_HOP_PROFILE", "0")))
        profile_cuda = bool(int(os.getenv("CUGUGA_MRCI_CUDA_PROFILE", "0"))) or hop_stage_profile
        profile_cuda_sync = bool(int(os.getenv("CUGUGA_MRCI_CUDA_PROFILE_SYNC", "0")))
        sym_pair_env = str(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "auto")).strip().lower()
        if sym_pair_env in ("1", "true", "yes", "on"):
            sym_pair = True
        elif sym_pair_env in ("0", "false", "no", "off"):
            sym_pair = False
        else:
            sym_pair = None
        subspace_eigh_cpu_env = str(os.getenv("CUGUGA_MRCI_CUDA_SUBSPACE_EIGH_CPU", "auto")).strip().lower()
        subspace_eigh_cpu_max_m = int(os.getenv("CUGUGA_MRCI_CUDA_SUBSPACE_EIGH_CPU_MAX_M", "64"))
        if subspace_eigh_cpu_max_m < 0:
            subspace_eigh_cpu_max_m = 0
        if subspace_eigh_cpu_env in ("1", "true", "yes", "on"):
            subspace_eigh_cpu = True
        elif subspace_eigh_cpu_env in ("0", "false", "no", "off"):
            subspace_eigh_cpu = False
        else:
            subspace_cutoff = int(os.getenv("CUGUGA_MRCI_CUDA_SUBSPACE_EIGH_CPU_NCSF_CUTOFF", "100000000"))
            if subspace_cutoff < 0:
                subspace_cutoff = 0
            subspace_eigh_cpu = bool(int(drt_mrci.ncsf) <= int(subspace_cutoff))
        make_hdiag_cpu_env = str(os.getenv("CUGUGA_MRCI_CUDA_MAKE_HDIAG_CPU", "auto")).strip().lower()
        make_hdiag_cutoff = int(os.getenv("CUGUGA_MRCI_CUDA_MAKE_HDIAG_CPU_NCSF_CUTOFF", "10000"))
        if make_hdiag_cutoff < 0:
            make_hdiag_cutoff = 0
        if make_hdiag_cpu_env in ("1", "true", "yes", "on"):
            make_hdiag_cpu = True
        elif make_hdiag_cpu_env in ("0", "false", "no", "off"):
            make_hdiag_cpu = False
        else:
            make_hdiag_cpu = bool(int(drt_mrci.ncsf) <= int(make_hdiag_cutoff))

        # GPU setup
        n_act_int = int(n_act)
        n_virt_int = int(n_virt)
        nelec_int = int(nelec)
        max_virt_e_int = int(max_virt_e)
        restricted = max_virt_e_int < min(nelec_int, 2 * n_virt_int)

        drt_dev_mrci = make_device_drt(drt_mrci)
        state_dev_mrci = make_device_state_cache(drt=drt_mrci, drt_dev=drt_dev_mrci)

        # Prepare 1e/2e integrals on GPU (dense or DF/Cholesky).
        norb = int(h1e.shape[0])
        nops = int(norb) * int(norb)

        try:
            from asuka.integrals.df_integrals import DFMOIntegrals, DeviceDFMOIntegrals
        except Exception:  # pragma: no cover
            DFMOIntegrals = None  # type: ignore[assignment]
            DeviceDFMOIntegrals = None  # type: ignore[assignment]

        l_full_d = None
        eri_mat_d = None
        eri_mat_t_d = None

        if DFMOIntegrals is not None and isinstance(eri, DFMOIntegrals):
            if int(eri.norb) != int(norb):
                raise ValueError("DFMOIntegrals.norb must match h1e.shape[0]")
            h_eff = np.asarray(h1e, dtype=np.float64) - 0.5 * np.asarray(eri.j_ps, dtype=np.float64)
            h_eff_d = cp.asarray(h_eff, dtype=cp.float64)
            l_full_d = cp.asarray(np.asarray(eri.l_full, dtype=np.float64, order="C"), dtype=cp.float64)
        elif DeviceDFMOIntegrals is not None and isinstance(eri, DeviceDFMOIntegrals):
            if int(eri.norb) != int(norb):
                raise ValueError("DeviceDFMOIntegrals.norb must match h1e.shape[0]")
            h_eff_d = cp.asarray(h1e, dtype=cp.float64) - 0.5 * cp.asarray(eri.j_ps, dtype=cp.float64)
            if eri.l_full is not None:
                l_full_d = cp.asarray(eri.l_full, dtype=cp.float64)
            elif eri.eri_mat is not None:
                eri_mat_d = cp.asarray(eri.eri_mat, dtype=cp.float64)
                eri_mat_t_d = 0.5 * eri_mat_d.T
            else:
                raise ValueError("DeviceDFMOIntegrals must provide l_full or eri_mat")
        else:
            eri4 = _restore_eri_4d(eri, norb)
            h_eff = np.asarray(h1e, dtype=np.float64) - 0.5 * np.einsum("pqqs->ps", eri4)
            h_eff_d = cp.asarray(h_eff, dtype=cp.float64)
            eri_mat_d = cp.asarray(np.asarray(eri4.reshape(nops, nops), dtype=np.float64, order="C"), dtype=cp.float64)
            eri_mat_t_d = 0.5 * eri_mat_d.T

        naux = None
        if l_full_d is not None:
            if getattr(l_full_d, "ndim", None) != 2 or int(l_full_d.shape[0]) != int(nops):
                raise ValueError("l_full must have shape (norb*norb, naux)")
            naux = int(l_full_d.shape[1])

        if sym_pair is None:
            # Auto: symmetric-pair contraction is most beneficial for larger CI spaces,
            # and for DF/Cholesky (`l_full`) it directly halves the dominant pair-space GEMM.
            # Keep a conservative norb guard to avoid regressions on smaller CAS spaces.
            use_df = bool(l_full_d is not None)
            sym_pair = bool(14 <= int(norb) <= 16 and (use_df or int(drt_mrci.ncsf) >= 200_000))

        ws_profile = None
        if restricted:
            drt_full = build_drt_mrcisd(
                n_act=n_act,
                n_virt=n_virt,
                nelec=nelec,
                twos=twos,
                orbsym=orbsym_corr,
                wfnsym=wfnsym,
                max_virt_e=int(max_virt_e) + 1,
            )
            from asuka.mrci.projected_hop import build_subspace_map  # noqa: PLC0415

            hop_map = build_subspace_map(drt_full=drt_full, drt_sub=drt_mrci)
            sub_to_full_d = cp.asarray(hop_map.sub_to_full, dtype=cp.int64)

            # Note: The CUDA hop uses a device-built combined E_pq table and does
            # not require the CPU-side precomputed per-(p,q) CSR operators.
            # Keeping this off avoids a large one-time setup cost.
            if bool(int(os.getenv("CUGUGA_MRCI_CUDA_PRECOMPUTE_EPQ", "0"))):
                precompute_epq_actions(drt_full)
            drt_dev_full = make_device_drt(drt_full)
            state_dev_full = make_device_state_cache(drt=drt_full, drt_dev=drt_dev_full)

            epq_table_full = build_epq_action_table_combined_device(
                drt=drt_full, drt_dev=drt_dev_full, state_dev=state_dev_full
            )

            # Re-build eri_mat_t for full space if needed (norb is the same though)
            # We assume h_eff and eri_mat_t (norb x norb) are same for full and sub
            # since they only depend on norb.

            ws_full = CudaMrciHopWorkspace.auto(ncsf=int(drt_full.ncsf), nops=int(nops), naux=naux, sym_pair=sym_pair)
            x_full_buf = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            y_full_buf = cp.empty((int(drt_full.ncsf),), dtype=cp.float64)
            ws_profile = ws_full

            def hop_gpu(x_d):
                return hop_cuda_projected(
                    drt_full=drt_full,
                    drt_dev_full=drt_dev_full,
                    state_dev_full=state_dev_full,
                    epq_table_full=epq_table_full,
                    h_eff=h_eff_d,
                    eri_mat_t_full=eri_mat_t_d,
                    l_full_full=l_full_d,
                    x_sub=x_d,
                    sub_to_full=sub_to_full_d,
                    x_full_buf=x_full_buf,
                    y_full_buf=y_full_buf,
                    workspace_full=ws_full,
                    sym_pair=bool(sym_pair),
                )

        else:
            # See note above on CUGUGA_MRCI_CUDA_PRECOMPUTE_EPQ.
            if bool(int(os.getenv("CUGUGA_MRCI_CUDA_PRECOMPUTE_EPQ", "0"))):
                precompute_epq_actions(drt_mrci)
            epq_table = build_epq_action_table_combined_device(
                drt=drt_mrci, drt_dev=drt_dev_mrci, state_dev=state_dev_mrci
            )

            ws = CudaMrciHopWorkspace.auto(ncsf=int(drt_mrci.ncsf), nops=int(nops), naux=naux, sym_pair=sym_pair)
            ws_profile = ws

            def hop_gpu(x_d):
                return hop_cuda_epq_table(
                    drt=drt_mrci,
                    drt_dev=drt_dev_mrci,
                    state_dev=state_dev_mrci,
                    epq_table=epq_table,
                    h_eff=h_eff_d,
                    eri_mat_t=eri_mat_t_d,
                    l_full=l_full_d,
                    x=x_d,
                    workspace=ws,
                    sym_pair=bool(sym_pair),
                )

        # Build hdiag for GPU Davidson (CPU/GPU auto policy with safe fallback).
        hdiag_backend = "cpu"
        hdiag_error = None
        t_hdiag = time.perf_counter()
        if bool(make_hdiag_cpu):
            hdiag = _make_hdiag_det_guess(drt_mrci, h1e, eri)
            hdiag_d = cp.asarray(hdiag, dtype=cp.float64)
        else:
            try:
                neleca_det = (int(drt_mrci.nelec) + int(drt_mrci.twos_target)) // 2
                p = cp.arange(norb, dtype=cp.int32)
                idx_pp = p * int(norb + 1)
                h1e_diag_d = cp.asarray(np.diag(np.asarray(h1e, dtype=np.float64)), dtype=cp.float64)
                if l_full_d is not None:
                    l_pp = cp.ascontiguousarray(l_full_d[idx_pp])
                    eri_ppqq_d = cp.ascontiguousarray(l_pp @ l_pp.T)
                    pair_norm_src = getattr(eri, "pair_norm", None)
                    if pair_norm_src is not None:
                        eri_pqqp_d = cp.ascontiguousarray(
                            cp.square(cp.asarray(pair_norm_src, dtype=cp.float64).reshape(norb, norb))
                        )
                    else:
                        eri_pqqp_d = cp.ascontiguousarray(cp.square(cp.linalg.norm(l_full_d, axis=1).reshape(norb, norb)))
                    hdiag_backend = "cuda_det_guess_df_l_full"
                elif eri_mat_d is not None:
                    eri_ppqq_d = cp.ascontiguousarray(eri_mat_d[idx_pp[:, None], idx_pp[None, :]])
                    pp = p[:, None]
                    qq = p[None, :]
                    idx_pq = (pp * norb + qq).ravel()
                    idx_qp = (qq * norb + pp).ravel()
                    eri_pqqp_d = cp.ascontiguousarray(eri_mat_d[idx_pq, idx_qp].reshape(norb, norb))
                    hdiag_backend = "cuda_det_guess"
                else:  # pragma: no cover
                    raise RuntimeError("internal error: missing integral representation for CUDA hdiag build")
                hdiag_d = build_hdiag_det_guess_from_steps_inplace_device(
                    state_dev_mrci,
                    neleca_det=neleca_det,
                    h1e_diag=h1e_diag_d,
                    eri_ppqq=eri_ppqq_d,
                    eri_pqqp=eri_pqqp_d,
                    threads=256,
                    stream=cp.cuda.get_current_stream(),
                    sync=bool(profile_cuda_sync),
                )
            except Exception as e:
                hdiag = _make_hdiag_det_guess(drt_mrci, h1e, eri)
                hdiag_d = cp.asarray(hdiag, dtype=cp.float64)
                hdiag_backend = "cpu_fallback"
                hdiag_error = f"{type(e).__name__}: {e}"
        hdiag_s = time.perf_counter() - t_hdiag
        res = davidson_sym_gpu(
            hop_gpu,
            x0=x0,
            hdiag=hdiag_d,
            nroots=nroots,
            max_cycle=max_cycle,
            max_space=max_space,
            tol=tol,
            profile=profile_cuda,
            profile_cuda_sync=profile_cuda_sync,
            subspace_eigh_cpu=bool(subspace_eigh_cpu),
            subspace_eigh_cpu_max_m=int(subspace_eigh_cpu_max_m),
        )
        converged, e, xs = res.converged, res.e, res.x
        cuda_stats = None if res.stats is None else dict(res.stats)
        if cuda_stats is not None and ws_profile is not None:
            if getattr(ws_profile, "profile_total", None):
                cuda_stats.update({f"hop_stage_{k}": float(v) for k, v in ws_profile.profile_total.items()})
            cuda_stats["hop_stage_calls"] = float(getattr(ws_profile, "profile_calls", 0))
            cuda_stats["sym_pair"] = float(int(bool(sym_pair)))
            cuda_stats["hdiag_gpu"] = float(int(hdiag_backend.startswith("cuda_det_guess")))
            cuda_stats["hdiag_fallback"] = float(int(hdiag_backend == "cpu_fallback"))
            cuda_stats["hdiag_policy_cpu"] = float(int(bool(make_hdiag_cpu)))
            cuda_stats["hdiag_cutoff"] = float(int(make_hdiag_cutoff))
            cuda_stats["subspace_eigh_cpu_max_m"] = float(int(subspace_eigh_cpu_max_m))
            cuda_stats["make_hdiag_s"] = float(hdiag_s)
            if hdiag_error is not None:
                cuda_stats["hdiag_error"] = 1.0
    else:
        cuda_stats = None
        hdiag = _make_hdiag_det_guess(drt_mrci, h1e, eri)
        try:
            converged, e, xs = davidson1_sym(
                hop,
                x0,
                precond,
                tol=float(tol),
                lindep=float(lindep),
                max_cycle=int(max_cycle),
                max_space=int(max_space),
                max_memory=float(max_memory),
                nroots=nroots,
                verbose=0 if verbose is None else int(verbose),
            )
        finally:
            if contract_executor is not None:
                contract_executor.shutdown(wait=True)

    e_roots = np.asarray(e, dtype=np.float64).ravel()
    if int(e_roots.size) != nroots:
        raise RuntimeError("internal error: unexpected eigenvalue array size from Davidson")
    e_roots = e_roots + float(ecore)

    ci_roots = [np.ascontiguousarray(v, dtype=np.float64) for v in xs[:nroots]]
    if len(ci_roots) != nroots:
        raise RuntimeError("internal error: unexpected eigenvector list size from Davidson")

    overlap_ref_root = np.zeros((len(ci_ref0_list), nroots), dtype=np.float64)
    for k, ref in enumerate(ci_ref0_list):
        for i in range(nroots):
            ov = np.vdot(ref, ci_roots[i])
            overlap_ref_root[k, i] = float(np.real(ov * np.conjugate(ov)))

    cuda_diag_num: dict[str, float] = {}
    if cuda_stats is not None:
        for k, v in cuda_stats.items():
            try:
                cuda_diag_num[f"cuda_{k}"] = float(v)
            except Exception:
                # Keep diagnostics robust even when backend reports symbolic/string values.
                continue

    # Backward-compatible single-root return type.
    if nroots == 1 and not isinstance(ci_cas, (list, tuple)):
        ci_out = ci_roots[0]
        overlap = np.vdot(ci_ref0_list[0], ci_out)
        c2 = float(np.real(overlap * np.conjugate(overlap)))
        w_ref = float(np.sum(np.square(np.abs(ci_out[np.asarray(ref_idx, dtype=np.int64)]))))
        diag = {"c2": c2, "w_ref": w_ref}
        if cuda_diag_num:
            diag.update(cuda_diag_num)
        diag.update(mrcisd_virtual_weights(drt=drt_mrci, ci=ci_out, n_act=n_act))
        return MRCISDResult(
            converged=bool(np.asarray(converged, dtype=bool).ravel()[0]),
            e_mrci=float(e_roots[0]),
            ci=ci_out,
            drt=drt_mrci,
            ci_ref0=ci_ref0_list[0],
            ref_idx=ref_idx,
            diagnostics=diag,
        )

    diagnostics: list[dict[str, float]] = []
    ref_idx_i64 = np.asarray(ref_idx, dtype=np.int64)
    for i in range(nroots):
        ci_i = ci_roots[i]
        w_ref = float(np.sum(np.square(np.abs(ci_i[ref_idx_i64]))))
        diag = {"w_ref": w_ref, "c2_max": float(np.max(overlap_ref_root[:, i]))}
        if cuda_diag_num:
            diag.update(cuda_diag_num)
        diag.update(mrcisd_virtual_weights(drt=drt_mrci, ci=ci_i, n_act=n_act))
        diagnostics.append(diag)

    return MRCISDResultMulti(
        converged=np.asarray(converged, dtype=bool).ravel()[:nroots].copy(),
        e_mrci=np.asarray(e_roots, dtype=np.float64).ravel()[:nroots].copy(),
        ci=ci_roots,
        drt=drt_mrci,
        ci_ref0=[np.ascontiguousarray(v, dtype=np.float64) for v in ci_ref0_list],
        ref_idx=np.asarray(ref_idx, dtype=np.int32).copy(),
        overlap_ref_root=overlap_ref_root,
        diagnostics=diagnostics,
    )


def mrcisd_plus_q(
    *,
    e_mrci: float,
    e_ref: float,
    ci_mrci: np.ndarray,
    ci_ref0: np.ndarray,
    ref_idx: np.ndarray,
    model: str = "fixed",
    min_ref: float = 1e-8,
) -> tuple[float | None, dict[str, float]]:
    """Compute a Davidson-type +Q correction for MRCISD.

    Parameters
    ----------
    model:
        - "fixed": use c^2 = |<Ψ_ref|Ψ_MRCI>|^2
        - "weight": use w_ref = Σ_{I∈ref} |C_I|^2
    """

    model = str(model).strip().lower()
    if model not in ("fixed", "weight"):
        raise ValueError("model must be 'fixed' or 'weight'")

    c_mrci = np.asarray(ci_mrci)
    c_ref = np.asarray(ci_ref0)
    if c_mrci.shape != c_ref.shape:
        raise ValueError("ci_mrci and ci_ref0 must have the same shape")

    ref_idx_i = np.asarray(ref_idx, dtype=np.int64).ravel()
    if ref_idx_i.size == 0:
        raise ValueError("ref_idx must be non-empty")
    ncsf = int(c_mrci.size)
    if np.any(ref_idx_i < 0) or np.any(ref_idx_i >= ncsf):
        raise ValueError("ref_idx contains out-of-range indices")

    overlap = np.vdot(c_ref, c_mrci)
    c2 = float(np.real(overlap * np.conjugate(overlap)))
    w_ref = float(np.sum(np.square(np.abs(c_mrci[ref_idx_i]))))
    e_corr = float(e_mrci) - float(e_ref)

    denom = c2 if model == "fixed" else w_ref
    if float(denom) < float(min_ref):
        return None, {"c2": c2, "w_ref": w_ref, "e_corr": e_corr, "de_q": float("nan")}

    de_q = e_corr * (1.0 - denom) / denom
    return float(e_mrci) + float(de_q), {"c2": c2, "w_ref": w_ref, "e_corr": e_corr, "de_q": float(de_q)}


def mrcisd_virtual_weights(*, drt: DRT, ci: np.ndarray, n_act: int) -> dict[str, float]:
    """Return weights by number of electrons in the virtual block (0/1/2/...)."""

    n_act = int(n_act)
    if n_act < 0 or n_act > int(drt.norb):
        raise ValueError("n_act must satisfy 0 <= n_act <= drt.norb")

    ci_arr = np.asarray(ci)
    if ci_arr.ndim != 1 or int(ci_arr.size) != int(drt.ncsf):
        raise ValueError("ci must be a 1D vector of length drt.ncsf")

    cache = get_state_cache(drt)
    steps = np.asarray(cache.steps, dtype=np.int8)
    if steps.shape != (int(drt.ncsf), int(drt.norb)):
        raise RuntimeError("internal error: invalid cached steps table shape")

    step_to_occ = np.asarray([0, 1, 1, 2], dtype=np.int16)
    virt_e = np.sum(step_to_occ[steps[:, n_act:]], axis=1, dtype=np.int16)

    w = np.square(np.abs(ci_arr))
    out: dict[str, float] = {}
    for n in np.unique(virt_e).tolist():
        mask = virt_e == int(n)
        out[f"w_virt_e_{int(n)}"] = float(np.sum(w[mask]))
    return out
