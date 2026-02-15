from __future__ import annotations

"""PySCF-facing driver helpers for MRCI.

The low-level MRCI kernels in :mod:`asuka.mrci.mrcisd` and
:mod:`asuka.mrci.ic_mrcisd` are written in terms of *MO-basis* integrals.
In practice, most users will want to run MRCI on top of a PySCF CASCI/CASSCF
calculation.

This module provides high-level drivers to:
  - Build frozen-core shifted MO integrals for a correlated subspace.
  - Call the uncontracted (MRCISD) or contracted (ic-MRCISD) kernels.
  - (Optionally) compute a Davidson-type +Q correction.

Notes
-----
* These helpers import PySCF lazily so that CPU-only installs without PySCF
  can still import :mod:`cuguga`.
* Orbital ordering convention used by the kernels:
    [internal/active][external]
  i.e. active orbitals first, then external (virtual) orbitals.
"""

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.cuguga.drt import build_drt
from asuka.integrals.df_integrals import DeviceDFMOIntegrals
from asuka.mrci.frozen_core import _build_frozen_core_mo_integrals_pyscf
from asuka.mrci.frozen_core import _frozen_core_h1e_ecore_pyscf
from asuka.mrci.frozen_core import FrozenCoreMOIntegrals
from asuka.mrci.ic_mrcisd import ICMRCISDResult, ic_mrcisd_kernel
from asuka.mrci.mrcisd import MRCISDResult, MRCISDResultMulti, mrcisd_kernel, mrcisd_plus_q
from asuka.solver import GUGAFCISolver


Method = Literal["mrcisd", "ic_mrcisd"]
IntegralsBackend = Literal["pyscf_eri4", "cueri_df"]


@dataclass(frozen=True)
class MRCIFromMCResult:
    """Result container for MRCI calculations on top of PySCF objects.

    Attributes
    ----------
    method : str
        Method used ("mrcisd" or "ic_mrcisd").
    e_ref : float
        Reference (CAS) total energy in the frozen-core convention.
    e_tot : float
        MRCI total energy (includes the frozen-core + E_nuc shift).
    e_corr : float
        Correlation energy relative to the reference.
    e_tot_plus_q : float | None
        Total energy including Davidson-type +Q correction, if requested.
    plus_q_diag : dict[str, float] | None
        Diagnostics from the +Q correction, if requested.
    result : MRCISDResult | ICMRCISDResult
        Raw kernel result object.
    integrals : FrozenCoreMOIntegrals | None
        Cached MO integrals, if requested.
    df_integrals : DeviceDFMOIntegrals | None
        Cached density-fitted integrals for CUDA hop, if requested.
    """

    method: str
    e_ref: float
    e_tot: float
    e_corr: float
    e_tot_plus_q: float | None
    plus_q_diag: dict[str, float] | None
    result: MRCISDResult | ICMRCISDResult
    integrals: FrozenCoreMOIntegrals | None = None
    df_integrals: DeviceDFMOIntegrals | None = None


@dataclass(frozen=True)
class MRCIFromMCStatesResult:
    """Result container for multi-state MRCI calculations on top of PySCF objects.

    Attributes
    ----------
    method : str
        Method used ("mrcisd" or "ic_mrcisd").
    states : list[int]
        List of reference state indices.
    nroots : int
        Number of roots computed.
    e_ref : np.ndarray
        Per-reference-state energies in the frozen-core convention. Shape: (nstates,).
    mrci : MRCISDResultMulti
        Raw multi-root MRCISD payload (roots are *not* assigned to reference states).
    ecore : float
        Frozen-core energy component (including nuclear repulsion).
    ncore : int
        Number of core orbitals (frozen + correlated).
    n_act : int
        Number of active orbitals.
    n_virt : int
        Number of virtual orbitals.
    nelec : int
        Total number of electrons (frozen + correlated).
    twos : int
        Spin multiplicity (2S).
    integrals : FrozenCoreMOIntegrals | None
        Cached MO integrals, if requested.
    df_integrals : DeviceDFMOIntegrals | None
        Cached density-fitted integrals for CUDA hop, if requested.
    """

    method: str
    states: list[int]
    nroots: int
    e_ref: np.ndarray
    mrci: MRCISDResultMulti
    ecore: float
    ncore: int
    n_act: int
    n_virt: int
    nelec: int
    twos: int
    integrals: FrozenCoreMOIntegrals | None = None
    df_integrals: DeviceDFMOIntegrals | None = None


def _as_float(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64).ravel()[0])


def _require_native_guga_reference(mc: Any) -> None:
    fcisolver = getattr(mc, "fcisolver", None)
    if not isinstance(fcisolver, GUGAFCISolver):
        cls = type(fcisolver).__name__ if fcisolver is not None else "None"
        raise RuntimeError(
            "ASUKA native MRCI path requires mc.fcisolver=asuka.solver.GUGAFCISolver; "
            f"got {cls}."
        )


def _get_states_ci(
    mc: Any,
    *,
    states: Sequence[int],
    n_act: int,
    nelec_act: int,
    twos: int,
    orbsym_act: Sequence[int] | None,
    wfnsym: int | None,
) -> list[np.ndarray]:
    """Return requested CAS CI vectors in ASUKA DRT-CSF order.

    Native-only path: selected `mc.ci` vectors must already be in ASUKA CSF
    basis with size equal to DRT ``ncsf``.
    """

    _require_native_guga_reference(mc)

    states_i = [int(s) for s in states]
    if not states_i:
        raise ValueError("states must be non-empty")
    if any(s < 0 for s in states_i):
        raise ValueError("state index must be non-negative")

    drt = build_drt(
        norb=int(n_act),
        nelec=int(nelec_act),
        twos_target=int(twos),
        orbsym=orbsym_act,
        wfnsym=wfnsym,
    )
    ncsf = int(drt.ncsf)

    ci = getattr(mc, "ci")
    if isinstance(ci, (list, tuple)):
        ci_list = list(ci)
    else:
        ci_list = [ci]
    nci = int(len(ci_list))

    if any(s >= nci for s in states_i):
        raise ValueError("state index out of range for mc.ci")

    selected = [np.asarray(ci_list[s], dtype=np.float64) for s in states_i]
    bad = [int(states_i[k]) for k, arr in enumerate(selected) if int(arr.size) != ncsf]
    if bad:
        raise RuntimeError(
            "ASUKA native path requires mc.ci in CSF/DRT layout. "
            f"Expected vector size ncsf={ncsf}; non-CSF states={bad}."
        )
    return [arr.ravel() for arr in selected]


def _get_state_ci(
    mc: Any,
    *,
    state: int = 0,
    n_act: int,
    nelec_act: int,
    twos: int,
    orbsym_act: Sequence[int] | None,
    wfnsym: int | None,
) -> np.ndarray:
    return _get_states_ci(
        mc,
        states=[int(state)],
        n_act=int(n_act),
        nelec_act=int(nelec_act),
        twos=int(twos),
        orbsym_act=orbsym_act,
        wfnsym=wfnsym,
    )[0]


def _infer_twos(mc: Any, *, twos: int | None) -> int:
    if twos is not None:
        return int(twos)
    fcisolver = getattr(mc, "fcisolver", None)
    if fcisolver is not None and getattr(fcisolver, "twos", None) is not None:
        return int(getattr(fcisolver, "twos"))
    # PySCF-native solvers often expose only spin multiplicity/2S+1 indirectly.
    # Default to singlet if not provided.
    return 0


def _infer_orbsym_corr_and_wfnsym(
    mc: Any,
    *,
    ncore_mc: int,
    n_act: int,
    n_virt: int,
    ncore_frozen: int | None = None,
    orbsym: Sequence[int] | None,
    wfnsym: int | None,
) -> tuple[tuple[int, ...] | None, int | None]:
    """Infer symmetry labels for correlated orbitals from a PySCF object.

    When PySCF symmetry is enabled, `mf.orbsym` labels every MO. cuGUGA's MRCI
    kernels expect symmetry labels for the correlated orbital ordering
    `[active][virtual]` (excluding frozen core orbitals).
    """

    ncore_i = int(ncore_mc)
    n_act_i = int(n_act)
    n_virt_i = int(n_virt)
    ncore_frozen_i = ncore_i if ncore_frozen is None else int(ncore_frozen)
    if ncore_frozen_i < 0 or ncore_frozen_i > ncore_i:
        raise ValueError("ncore_frozen must satisfy 0 <= ncore_frozen <= ncore_mc")

    if orbsym is not None:
        orbsym_corr = tuple(int(x) for x in np.asarray(orbsym, dtype=np.int32).ravel().tolist())
    else:
        mf_or_mc = getattr(mc, "_scf", mc)
        orbsym_full = getattr(mf_or_mc, "orbsym", None)
        if orbsym_full is None:
            orbsym_corr = None
        else:
            arr = np.asarray(orbsym_full, dtype=np.int32).ravel()
            j0 = ncore_frozen_i
            j1 = ncore_i + n_act_i + n_virt_i
            orbsym_corr = tuple(int(x) for x in arr[j0:j1].tolist())

    wfnsym_i = wfnsym
    if wfnsym_i is None:
        fcisolver = getattr(mc, "fcisolver", None)
        if fcisolver is not None and getattr(fcisolver, "wfnsym", None) is not None:
            wfnsym_i = int(getattr(fcisolver, "wfnsym"))
    if wfnsym_i is not None:
        wfnsym_i = int(wfnsym_i)

    return orbsym_corr, wfnsym_i


def _embed_ci_with_docc_prefix(
    *,
    ci_act: np.ndarray,
    n_docc: int,
    n_act: int,
    nelec_act: int,
    twos: int,
    orbsym_act: Sequence[int] | None,
    orbsym_full: Sequence[int] | None,
    wfnsym: int | None,
) -> np.ndarray:
    """Embed an active-space CI vector into a larger internal space with a fixed DOCC prefix."""

    n_docc_i = int(n_docc)
    if n_docc_i <= 0:
        return np.asarray(ci_act, dtype=np.float64)

    n_act_i = int(n_act)
    nelec_act_i = int(nelec_act)
    twos_i = int(twos)
    norb_small = n_act_i
    norb_large = n_docc_i + n_act_i
    nelec_large = nelec_act_i + 2 * n_docc_i

    drt_small = build_drt(
        norb=norb_small,
        nelec=nelec_act_i,
        twos_target=twos_i,
        orbsym=orbsym_act,
        wfnsym=wfnsym,
    )
    drt_large = build_drt(
        norb=norb_large,
        nelec=nelec_large,
        twos_target=twos_i,
        orbsym=orbsym_full,
        wfnsym=wfnsym,
    )

    ci_act_f64 = np.asarray(ci_act, dtype=np.float64).ravel()
    if int(ci_act_f64.size) != int(drt_small.ncsf):
        raise ValueError(f"ci_act has wrong size {int(ci_act_f64.size)} (expected {int(drt_small.ncsf)})")

    out = np.zeros(int(drt_large.ncsf), dtype=np.float64)
    steps_full = np.empty(norb_large, dtype=np.int8)
    steps_full[:n_docc_i] = 3  # D steps (doubly occupied)
    for j in range(int(drt_small.ncsf)):
        steps_act = drt_small.index_to_path(int(j))
        steps_full[n_docc_i:] = steps_act
        J = int(drt_large.path_to_index(steps_full))
        out[J] = float(ci_act_f64[j])
    return out


def _infer_states(mc: Any, *, states: Sequence[int] | None) -> list[int]:
    if states is None:
        ci = getattr(mc, "ci")
        if isinstance(ci, (list, tuple)):
            return list(range(len(ci)))
        return [0]

    out = [int(s) for s in states]
    if not out:
        raise ValueError("states must be non-empty when provided")
    if len(set(out)) != len(out):
        raise ValueError("states must not contain duplicates")
    if any(s < 0 for s in out):
        raise ValueError("states must be non-negative")

    ci = getattr(mc, "ci")
    if isinstance(ci, (list, tuple)):
        n = int(len(ci))
        if any(s >= n for s in out):
            raise ValueError("state index out of range for mc.ci")
    else:
        if any(s != 0 for s in out):
            raise ValueError("state != 0 but mc.ci is not a list/tuple")
    return out


def assign_roots_by_overlap(
    overlap: np.ndarray,
    *,
    method: Literal["hungarian", "greedy"] = "hungarian",
) -> np.ndarray:
    """Assign MRCI roots to reference states by maximum overlap.

    Parameters
    ----------
    overlap : np.ndarray
        Overlap matrix with shape (nref, nroots), where ``overlap[k, i] = |<ref_k|root_i>|^2``.
    method : {"hungarian", "greedy"}, optional
        Assignment algorithm. "hungarian" solves the linear sum assignment problem
        (optimal unique mapping), while "greedy" picks the largest overlap iteratively.
        Default is "hungarian".

    Returns
    -------
    roots : np.ndarray
        Integer array of length nref where ``roots[k]`` is the assigned root index for
        reference state k. Indices are unique.
    """

    overlap = np.asarray(overlap, dtype=np.float64)
    if overlap.ndim != 2:
        raise ValueError("overlap must be a 2D array")
    nref, nroots = map(int, overlap.shape)
    if nref == 0:
        return np.zeros((0,), dtype=np.int64)
    if nroots < nref:
        raise ValueError(f"need nroots >= nref for assignment (got nroots={nroots}, nref={nref})")

    method_s = str(method).strip().lower()
    if method_s not in ("hungarian", "greedy"):
        raise ValueError("method must be 'hungarian' or 'greedy'")

    if method_s == "greedy":
        # Greedy assignment with uniqueness constraints.
        roots = -np.ones(nref, dtype=np.int64)
        used = np.zeros(nroots, dtype=bool)
        row_order = np.argsort(np.max(overlap, axis=1))[::-1]
        for k in row_order.tolist():
            cols = np.argsort(overlap[k])[::-1]
            for i in cols.tolist():
                if not bool(used[i]):
                    roots[k] = int(i)
                    used[i] = True
                    break
            if int(roots[k]) < 0:
                raise RuntimeError("failed to assign unique roots (greedy)")
        return roots

    try:
        from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

        row, col = linear_sum_assignment(-overlap)
        if int(row.size) != nref:
            raise RuntimeError("unexpected assignment output size")
        order = np.argsort(row)
        roots = np.asarray(col[order], dtype=np.int64)
        if roots.shape != (nref,):
            raise RuntimeError("unexpected assignment output shape")
        if len(set(int(i) for i in roots.tolist())) != nref:
            raise RuntimeError("assignment produced duplicate roots")
        return roots
    except Exception:
        # Greedy fallback with uniqueness constraints.
        roots = -np.ones(nref, dtype=np.int64)
        used = np.zeros(nroots, dtype=bool)
        row_order = np.argsort(np.max(overlap, axis=1))[::-1]
        for k in row_order.tolist():
            cols = np.argsort(overlap[k])[::-1]
            for i in cols.tolist():
                if not bool(used[i]):
                    roots[k] = int(i)
                    used[i] = True
                    break
            if int(roots[k]) < 0:
                raise RuntimeError("failed to assign unique roots (fallback)")
        return roots


def _compute_cas_reference_energy(
    *,
    h1e_corr: np.ndarray,
    eri4_corr: np.ndarray,
    ecore: float,
    ci_cas: np.ndarray,
    n_act: int,
    nelec: int,
    twos: int,
    orbsym_act: Sequence[int] | None = None,
    wfnsym: int | None = None,
) -> float:
    """Compute E_ref (total) in the same convention as the MRCI kernels."""

    n_act = int(n_act)
    if n_act < 0:
        raise ValueError("n_act must be >= 0")

    cas = GUGAFCISolver(twos=int(twos), orbsym=orbsym_act, wfnsym=wfnsym)
    gamma, dm2 = cas.make_rdm12(ci_cas, norb=n_act, nelec=int(nelec))
    gamma = np.asarray(gamma, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)

    h_int = np.asarray(h1e_corr[:n_act, :n_act], dtype=np.float64)
    eri_int = np.asarray(eri4_corr[:n_act, :n_act, :n_act, :n_act], dtype=np.float64)
    e1 = float(np.einsum("pq,pq->", h_int, gamma, optimize=True))
    e2 = 0.5 * float(np.einsum("pqrs,pqrs->", eri_int, dm2, optimize=True))
    return float(ecore + e1 + e2)


def _compute_cas_reference_energy_df(
    *,
    h1e_corr: np.ndarray,
    l_full: Any,
    ecore: float,
    ci_cas: np.ndarray,
    n_act: int,
    nelec: int,
    twos: int,
    orbsym_act: Sequence[int] | None = None,
    wfnsym: int | None = None,
) -> float:
    """Compute E_ref (total) using DF/Cholesky vectors `l_full` instead of a dense `eri4`.

    Notes
    -----
    `l_full` is expected to store ordered-pair vectors in shape (norb*norb, naux),
    consistent with :class:`asuka.integrals.df_integrals.DFMOIntegrals` and
    :class:`asuka.integrals.df_integrals.DeviceDFMOIntegrals`.
    """

    n_act = int(n_act)
    if n_act < 0:
        raise ValueError("n_act must be >= 0")

    h1e_corr = np.asarray(h1e_corr, dtype=np.float64)
    norb = int(h1e_corr.shape[0])
    if h1e_corr.shape != (norb, norb):
        raise ValueError("h1e_corr must be square")

    cas = GUGAFCISolver(twos=int(twos), orbsym=orbsym_act, wfnsym=wfnsym)
    gamma, dm2 = cas.make_rdm12(ci_cas, norb=n_act, nelec=int(nelec))
    gamma = np.asarray(gamma, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)

    h_int = np.asarray(h1e_corr[:n_act, :n_act], dtype=np.float64)
    e1 = float(np.einsum("pq,pq->", h_int, gamma, optimize=True))

    if n_act == 0:
        return float(ecore + e1)

    # Build ordered-pair IDs for the internal block (p,q in [0,n_act)).
    p = np.arange(n_act, dtype=np.int64)
    pq_ids = (p[:, None] * int(norb) + p[None, :]).reshape(n_act * n_act)

    dm2_mat = dm2.reshape(n_act * n_act, n_act * n_act)

    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(l_full, cp.ndarray):
            l_int = cp.take(cp.asarray(l_full, dtype=cp.float64), cp.asarray(pq_ids, dtype=cp.int64), axis=0)
            d = cp.asarray(dm2_mat, dtype=cp.float64)
            tmp = d @ l_int
            e2 = 0.5 * cp.sum(l_int * tmp)
            return float(ecore + e1 + float(cp.asnumpy(e2)))
    except Exception:
        pass

    l_full_np = np.asarray(l_full, dtype=np.float64, order="C")
    if l_full_np.ndim != 2 or int(l_full_np.shape[0]) != int(norb) * int(norb):
        raise ValueError("l_full must have shape (norb*norb, naux)")
    l_int_np = np.asarray(l_full_np[pq_ids], dtype=np.float64, order="C")
    tmp_np = dm2_mat @ l_int_np
    e2 = 0.5 * float(np.einsum("pL,pL->", l_int_np, tmp_np, optimize=True))
    return float(ecore + e1 + e2)


def mrci_from_mc(
    mc: Any,
    *,
    method: Method = "mrcisd",
    state: int = 0,
    n_virt: int | None = None,
    twos: int | None = None,
    max_virt_e: int = 2,
    correlate_inactive: int = 0,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    # --- uncontracted MRCISD knobs ---
    hop_backend: str | None = None,
    tol: float = 1e-10,
    max_cycle: int = 400,
    max_space: int = 30,
    max_memory_mb: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = None,
    precompute_epq: bool = True,
    # --- contracted (ic) knobs ---
    contraction: str = "fic",
    backend: str = "semi_direct",
    sc_backend: str = "otf",
    symmetry: bool = True,
    allow_same_external: bool = True,
    allow_same_internal: bool = True,
    norm_min_singles: float = 0.0,
    norm_min_doubles: float = 0.0,
    s_tol: float = 1e-12,
    solver: str = "davidson",
    dense_nlab_max: int = 250,
    # --- optional +Q ---
    plus_q: bool = False,
    plus_q_model: str = "fixed",
    plus_q_min_ref: float = 1e-8,
    # --- integrals backend ---
    integrals_backend: IntegralsBackend = "pyscf_eri4",
    df_auxbasis: Any = "auto",
    cueri_backend: str = "gpu_rys",
    cueri_threads: int = 256,
    cueri_aux_block_naux: int = 256,
    cueri_max_tile_bytes: int = 256 * 1024 * 1024,
    return_integrals: bool = False,
) -> MRCIFromMCResult:
    """Run (ic-)MRCISD on top of a PySCF CASCI/CASSCF object.

    Parameters
    ----------
    mc : Any
        PySCF CASCI/CASSCF-like object (or a scanner produced by ``mc.as_scanner()``).
    method : {"mrcisd", "ic_mrcisd"}, optional
        Method to run: "mrcisd" (uncontracted) or "ic_mrcisd" (internally contracted).
        Default is "mrcisd".
    state : int, optional
        Root index for multi-root CASSCF objects where ``mc.ci`` is a list. Default is 0.
    n_virt : int, optional
        Number of external (virtual) orbitals to include. Default is None (all orbitals
        after the CAS active space).
    twos : int, optional
        Spin multiplicity (2S). If None, inferred from `mc.fcisolver.twos` or defaults to 0.
    max_virt_e : int, optional
        Maximum number of electrons in the virtual space. Default is 2 (CISD).
    correlate_inactive : int, optional
        Number of inactive (core) orbitals in the reference to correlate by
        including them in the MRCI internal space. This reduces the frozen-core
        count from ``mc.ncore`` to ``mc.ncore - correlate_inactive``. Default is 0.
    orbsym : Sequence[int], optional
        Orbital symmetry labels. If None, inferred from `mc`.
    wfnsym : int, optional
        Wavefunction symmetry label. If None, inferred from `mc`.
    hop_backend : str, optional
        Backend for the Hamiltonian-vector product.
    tol : float, optional
        Convergence tolerance for the Davidson solver. Default is 1e-10.
    max_cycle : int, optional
        Maximum number of Davidson iterations. Default is 400.
    max_space : int, optional
        Maximum subspace size for Davidson. Default is 30.
    max_memory_mb : float, optional
        Maximum memory usage in MB. Default is 4000.0.
    contract_nthreads : int, optional
        Number of threads for contraction. Default is 1.
    contract_blas_nthreads : int, optional
        Number of threads for BLAS operations during contraction.
    precompute_epq : bool, optional
        Whether to precompute EPQ actions. Default is True.
    contraction : str, optional
        Contraction scheme for ic-MRCISD. Default is "fic".
    backend : str, optional
        Backend for ic-MRCISD. Default is "semi_direct".
    sc_backend : str, optional
        Self-consistent backend for ic-MRCISD. Default is "otf".
    symmetry : bool, optional
        Whether to use symmetry in ic-MRCISD. Default is True.
    allow_same_external : bool, optional
        Whether to allow same external orbitals in ic-MRCISD. Default is True.
    allow_same_internal : bool, optional
        Whether to allow same internal orbitals in ic-MRCISD. Default is True.
    norm_min_singles : float, optional
        Minimum norm for singles. Default is 0.0.
    norm_min_doubles : float, optional
        Minimum norm for doubles. Default is 0.0.
    s_tol : float, optional
        Tolerance for singularity check in ic-MRCISD. Default is 1e-12.
    solver : str, optional
        Solver for ic-MRCISD. Default is "davidson".
    dense_nlab_max : int, optional
        Maximum number of labels for dense operations in ic-MRCISD. Default is 250.
    plus_q : bool, optional
        If True, compute a Davidson-type +Q correction and return it in
        ``e_tot_plus_q``. Default is False.
    plus_q_model : str, optional
        Model for +Q correction. Default is "fixed".
    plus_q_min_ref : float, optional
        Minimum reference weight for +Q correction. Default is 1e-8.
    integrals_backend : {"pyscf_eri4", "cueri_df"}, optional
        MO-integral backend:
          - "pyscf_eri4" (default): build dense MO-basis ``eri4`` via PySCF AO->MO.
            This backend is intended for parity tests/benchmarks only.
          - "cueri_df": build GPU-resident DF/Cholesky vectors ``l_full[pq,L]`` via cuERI streamed DF.
            This avoids materializing ``eri4`` and currently requires ``hop_backend="cuda"`` and ``mol.cart=True``.
    df_auxbasis : Any, optional
        Auxiliary basis for DF. Default is "auto".
    cueri_backend : str, optional
        Backend for cuERI. Default is "gpu_rys".
    cueri_threads : int, optional
        Number of threads for cuERI. Default is 256.
    cueri_aux_block_naux : int, optional
        Block size for auxiliary basis in cuERI. Default is 256.
    cueri_max_tile_bytes : int, optional
        Maximum tile size in bytes for cuERI. Default is 256*1024*1024.
    return_integrals : bool, optional
        Whether to return cached integrals. Default is False.

    Returns
    -------
    MRCIFromMCResult
        Result object containing energies, wavefunction, and diagnostics.
    """

    method_s = str(method).strip().lower()
    if method_s not in ("mrcisd", "ic_mrcisd"):
        raise ValueError("method must be 'mrcisd' or 'ic_mrcisd'")

    integrals_backend_s = str(integrals_backend).strip().lower()
    if integrals_backend_s not in ("pyscf_eri4", "cueri_df"):
        raise ValueError("integrals_backend must be 'pyscf_eri4' or 'cueri_df'")

    mol = mc.mol
    mo = np.asarray(mc.mo_coeff, dtype=np.float64)
    ncore_mc = int(getattr(mc, "ncore", 0))
    n_act_mc = int(getattr(mc, "ncas", 0))
    if n_act_mc <= 0:
        raise ValueError("mc.ncas must be positive")

    correlate_inactive_i = int(correlate_inactive)
    if correlate_inactive_i < 0 or correlate_inactive_i > ncore_mc:
        raise ValueError("correlate_inactive must satisfy 0 <= correlate_inactive <= mc.ncore")
    if method_s != "mrcisd" and correlate_inactive_i != 0:
        raise NotImplementedError("correlate_inactive is currently supported only for method='mrcisd'")

    ncore_frozen = ncore_mc - correlate_inactive_i
    n_act_int = n_act_mc + correlate_inactive_i

    nmo = int(mo.shape[1])
    nvirt_all = nmo - ncore_mc - n_act_mc
    if nvirt_all < 0:
        raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

    if n_virt is None:
        n_virt = nvirt_all
    n_virt = int(n_virt)
    if n_virt < 0 or n_virt > nvirt_all:
        raise ValueError("n_virt must satisfy 0 <= n_virt <= (nmo-ncore-ncas)")

    mo_core_frozen = mo[:, :ncore_frozen]
    mo_core_corr = mo[:, ncore_frozen:ncore_mc]
    mo_act = mo[:, ncore_mc : ncore_mc + n_act_mc]
    mo_virt = mo[:, ncore_mc + n_act_mc : ncore_mc + n_act_mc + n_virt]
    mo_corr = np.hstack([mo_core_corr, mo_act, mo_virt])

    mf_or_mc = getattr(mc, "_scf", mc)

    eri_payload: Any
    ints_ret: FrozenCoreMOIntegrals | None = None
    df_ints_ret: DeviceDFMOIntegrals | None = None

    if integrals_backend_s == "pyscf_eri4":
        # Frozen-core shift in the *same* convention used by CASPT2 drivers:
        #   ecore = E_nuc + E_core(one+two)
        #   h1e_corr = <p|hcore+V_core|q>
        #   eri4_corr = (pq|rs) over correlated orbitals
        ints = _build_frozen_core_mo_integrals_pyscf(
            mol=mol, mf_or_mc=mf_or_mc, mo_core=mo_core_frozen, mo_corr=mo_corr
        )
        h1e_corr = ints.h1e
        eri_payload = ints.eri4
        ecore = float(ints.ecore)
        ints_ret = ints if bool(return_integrals) else None
    else:
        # cuERI streamed DF on GPU: avoid materializing dense eri4.
        if method_s != "mrcisd":
            raise NotImplementedError("integrals_backend='cueri_df' currently supports only method='mrcisd'")
        if str(hop_backend).strip().lower() != "cuda":
            raise ValueError("integrals_backend='cueri_df' currently requires hop_backend='cuda'")

        h1e_corr, ecore = _frozen_core_h1e_ecore_pyscf(mol=mol, mf_or_mc=mf_or_mc, mo_core=mo_core_frozen, mo_corr=mo_corr)

        try:
            from asuka.cuda.active_space_df.cueri_builder import (  # noqa: PLC0415
                CuERIActiveSpaceDFBuilder,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("integrals_backend='cueri_df' requires cuERI") from e

        df_builder = CuERIActiveSpaceDFBuilder(
            mol,
            auxbasis=df_auxbasis,
            backend=str(cueri_backend),
            threads=int(cueri_threads),
            aux_block_naux=int(cueri_aux_block_naux),
            max_tile_bytes=int(cueri_max_tile_bytes),
        )
        df = df_builder.build(
            mo_corr,
            want_eri_mat=False,
            want_j_ps=True,
            want_pair_norm=True,
        )
        df_ints = DeviceDFMOIntegrals(
            norb=int(mo_corr.shape[1]),
            l_full=df.l_full,
            j_ps=df.j_ps,
            pair_norm=df.pair_norm,
            eri_mat=df.eri_mat,
        )
        eri_payload = df_ints
        df_ints_ret = df_ints if bool(return_integrals) else None

    # Active-space electron count (CAS electrons only).
    nelecas = getattr(mc, "nelecas", None)
    if nelecas is None:
        raise ValueError("mc.nelecas is required")
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        nelec_act = int(nelecas[0]) + int(nelecas[1])
    else:
        nelec_act = int(nelecas)
    nelec_corr = nelec_act + 2 * correlate_inactive_i
    twos_i = _infer_twos(mc, twos=twos)

    orbsym_corr, wfnsym_i = _infer_orbsym_corr_and_wfnsym(
        mc,
        ncore_mc=ncore_mc,
        n_act=n_act_mc,
        n_virt=n_virt,
        ncore_frozen=ncore_frozen,
        orbsym=orbsym,
        wfnsym=wfnsym,
    )
    orbsym_act_int = None if orbsym_corr is None else orbsym_corr[:n_act_int]
    orbsym_act_small = None if orbsym_act_int is None else orbsym_act_int[correlate_inactive_i:]

    ci_cas_act = _get_state_ci(
        mc,
        state=state,
        n_act=n_act_mc,
        nelec_act=nelec_act,
        twos=twos_i,
        orbsym_act=orbsym_act_small,
        wfnsym=wfnsym_i,
    )

    ci_cas = ci_cas_act
    if correlate_inactive_i > 0:
        ci_cas = _embed_ci_with_docc_prefix(
            ci_act=ci_cas_act,
            n_docc=correlate_inactive_i,
            n_act=n_act_mc,
            nelec_act=nelec_act,
            twos=twos_i,
            orbsym_act=orbsym_act_small,
            orbsym_full=orbsym_act_int,
            wfnsym=wfnsym_i,
        )

    # Reference total energy in the frozen-core convention.
    if integrals_backend_s == "pyscf_eri4":
        e_ref = _compute_cas_reference_energy(
            h1e_corr=h1e_corr,
            eri4_corr=np.asarray(eri_payload, dtype=np.float64),
            ecore=ecore,
            ci_cas=ci_cas,
            n_act=n_act_int,
            nelec=nelec_corr,
            twos=twos_i,
            orbsym_act=orbsym_act_int,
            wfnsym=wfnsym_i,
        )
    else:
        e_ref = _compute_cas_reference_energy_df(
            h1e_corr=h1e_corr,
            l_full=eri_payload.l_full,
            ecore=ecore,
            ci_cas=ci_cas,
            n_act=n_act_int,
            nelec=nelec_corr,
            twos=twos_i,
            orbsym_act=orbsym_act_int,
            wfnsym=wfnsym_i,
        )

    if method_s == "mrcisd":
        nthreads = int(contract_nthreads)
        if nthreads < 1:
            nthreads = 1
        res = mrcisd_kernel(
            h1e=h1e_corr,
            eri=eri_payload,
            n_act=n_act_int,
            n_virt=n_virt,
            nelec=nelec_corr,
            twos=twos_i,
            ci_cas=ci_cas,
            ecore=ecore,
            orbsym_act=orbsym_act_int,
            orbsym_corr=orbsym_corr,
            wfnsym=wfnsym_i,
            max_virt_e=max_virt_e,
            hop_backend=hop_backend,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            max_memory=max_memory_mb,
            contract_nthreads=nthreads,
            contract_blas_nthreads=contract_blas_nthreads,
            precompute_epq=precompute_epq,
        )
        e_tot = float(res.e_mrci)
        e_corr = float(e_tot - e_ref)

        e_plus_q = None
        q_diag = None
        if bool(plus_q):
            e_plus_q, q_diag = mrcisd_plus_q(
                e_mrci=e_tot,
                e_ref=e_ref,
                ci_mrci=res.ci,
                ci_ref0=res.ci_ref0,
                ref_idx=res.ref_idx,
                model=str(plus_q_model),
                min_ref=float(plus_q_min_ref),
            )
            if e_plus_q is not None:
                e_plus_q = float(e_plus_q)

        return MRCIFromMCResult(
            method="mrcisd",
            e_ref=float(e_ref),
            e_tot=float(e_tot),
            e_corr=float(e_corr),
            e_tot_plus_q=e_plus_q,
            plus_q_diag=q_diag,
            result=res,
            integrals=ints_ret,
            df_integrals=df_ints_ret,
        )

    # ic-MRCISD
    hop_backend_ic = "augmented" if hop_backend is None else str(hop_backend)
    res2 = ic_mrcisd_kernel(
        h1e=h1e_corr,
        eri=eri_payload,
        n_act=n_act_int,
        n_virt=n_virt,
        nelec=nelec_corr,
        twos=twos_i,
        ci_cas=ci_cas,
        ecore=ecore,
        orbsym=orbsym_corr,
        wfnsym=wfnsym_i,
        max_virt_e=max_virt_e,
        hop_backend=hop_backend_ic,
        contraction=contraction,
        backend=backend,
        sc_backend=sc_backend,
        symmetry=symmetry,
        allow_same_external=allow_same_external,
        allow_same_internal=allow_same_internal,
        norm_min_singles=norm_min_singles,
        norm_min_doubles=norm_min_doubles,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        s_tol=s_tol,
        solver=solver,
        dense_nlab_max=dense_nlab_max,
        contract_nthreads=contract_nthreads if contract_nthreads > 0 else 1,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
    )
    e_tot = float(res2.e_tot)
    e_corr = float(e_tot - e_ref)

    # +Q for contracted variants is not well-defined in the non-orthogonal
    # contracted basis; users should apply +Q only to the uncontracted CI.
    if bool(plus_q):
        raise NotImplementedError("+Q correction is currently supported only for uncontracted mrcisd")

    return MRCIFromMCResult(
        method="ic_mrcisd",
        e_ref=float(e_ref),
        e_tot=float(e_tot),
        e_corr=float(e_corr),
        e_tot_plus_q=None,
        plus_q_diag=None,
        result=res2,
        integrals=ints_ret,
        df_integrals=df_ints_ret,
    )


def mrci_states_from_mc(
    mc: Any,
    *,
    method: Method = "mrcisd",
    states: Sequence[int] | None = None,
    nroots: int | None = None,
    n_virt: int | None = None,
    twos: int | None = None,
    max_virt_e: int = 2,
    correlate_inactive: int = 0,
    orbsym: Sequence[int] | None = None,
    wfnsym: int | None = None,
    # --- integrals backend ---
    integrals_backend: IntegralsBackend = "pyscf_eri4",
    df_auxbasis: Any = "auto",
    cueri_backend: str = "gpu_rys",
    cueri_threads: int = 256,
    cueri_aux_block_naux: int = 256,
    cueri_max_tile_bytes: int = 256 * 1024 * 1024,
    # --- uncontracted MRCISD knobs ---
    hop_backend: str | None = None,
    tol: float = 1e-10,
    max_cycle: int = 400,
    max_space: int = 30,
    max_memory_mb: float = 4000.0,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = None,
    precompute_epq: bool = True,
    return_integrals: bool = False,
) -> MRCIFromMCStatesResult:
    """Run uncontracted multi-root MRCISD for multiple reference states (integrals built once).

    Notes
    -----
    - This API returns *roots* from a single multi-root MRCISD solve; it does not
      assign roots to reference states. Use the returned overlap matrix
      ``mrci.overlap_ref_root`` for root assignment.
    - For now, ``nroots`` must equal ``len(states)`` so we can provide one CAS
      guess per desired root.
    - ``integrals_backend="cueri_df"`` builds GPU-resident DF vectors via cuERI and currently requires
      ``hop_backend="cuda"`` and ``mol.cart=True``.
    - ``integrals_backend="pyscf_eri4"`` is intended for parity tests/benchmarks only.
    """

    method_s = str(method).strip().lower()
    if method_s != "mrcisd":
        raise NotImplementedError("mrci_states_from_mc currently supports only method='mrcisd'")

    integrals_backend_s = str(integrals_backend).strip().lower()
    if integrals_backend_s not in ("pyscf_eri4", "cueri_df"):
        raise ValueError("integrals_backend must be 'pyscf_eri4' or 'cueri_df'")

    mol = mc.mol
    mo = np.asarray(mc.mo_coeff, dtype=np.float64)
    ncore_mc = int(getattr(mc, "ncore", 0))
    n_act_mc = int(getattr(mc, "ncas", 0))
    if n_act_mc <= 0:
        raise ValueError("mc.ncas must be positive")

    correlate_inactive_i = int(correlate_inactive)
    if correlate_inactive_i < 0 or correlate_inactive_i > ncore_mc:
        raise ValueError("correlate_inactive must satisfy 0 <= correlate_inactive <= mc.ncore")
    ncore_frozen = ncore_mc - correlate_inactive_i
    n_act_int = n_act_mc + correlate_inactive_i

    nmo = int(mo.shape[1])
    nvirt_all = nmo - ncore_mc - n_act_mc
    if nvirt_all < 0:
        raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

    if n_virt is None:
        n_virt = nvirt_all
    n_virt = int(n_virt)
    if n_virt < 0 or n_virt > nvirt_all:
        raise ValueError("n_virt must satisfy 0 <= n_virt <= (nmo-ncore-ncas)")

    states_list = _infer_states(mc, states=states)
    nroots_i = len(states_list) if nroots is None else int(nroots)
    if nroots_i != len(states_list):
        raise ValueError(
            f"mrci_states_from_mc currently requires nroots == len(states); got nroots={nroots_i} and "
            f"len(states)={len(states_list)}"
        )

    mo_core_frozen = mo[:, :ncore_frozen]
    mo_core_corr = mo[:, ncore_frozen:ncore_mc]
    mo_act = mo[:, ncore_mc : ncore_mc + n_act_mc]
    mo_virt = mo[:, ncore_mc + n_act_mc : ncore_mc + n_act_mc + n_virt]
    mo_corr = np.hstack([mo_core_corr, mo_act, mo_virt])

    mf_or_mc = getattr(mc, "_scf", mc)

    eri_payload: Any
    ints_ret: FrozenCoreMOIntegrals | None = None
    df_ints_ret: DeviceDFMOIntegrals | None = None

    if integrals_backend_s == "pyscf_eri4":
        ints = _build_frozen_core_mo_integrals_pyscf(
            mol=mol, mf_or_mc=mf_or_mc, mo_core=mo_core_frozen, mo_corr=mo_corr
        )
        h1e_corr = ints.h1e
        eri_payload = ints.eri4
        ecore = float(ints.ecore)
        ints_ret = ints if bool(return_integrals) else None
    else:
        if str(hop_backend).strip().lower() != "cuda":
            raise ValueError("integrals_backend='cueri_df' currently requires hop_backend='cuda'")

        h1e_corr, ecore = _frozen_core_h1e_ecore_pyscf(mol=mol, mf_or_mc=mf_or_mc, mo_core=mo_core_frozen, mo_corr=mo_corr)

        try:
            from asuka.cuda.active_space_df.cueri_builder import (  # noqa: PLC0415
                CuERIActiveSpaceDFBuilder,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError("integrals_backend='cueri_df' requires cuERI") from e

        df_builder = CuERIActiveSpaceDFBuilder(
            mol,
            auxbasis=df_auxbasis,
            backend=str(cueri_backend),
            threads=int(cueri_threads),
            aux_block_naux=int(cueri_aux_block_naux),
            max_tile_bytes=int(cueri_max_tile_bytes),
        )
        df = df_builder.build(
            mo_corr,
            want_eri_mat=False,
            want_j_ps=True,
            want_pair_norm=True,
        )
        df_ints = DeviceDFMOIntegrals(
            norb=int(mo_corr.shape[1]),
            l_full=df.l_full,
            j_ps=df.j_ps,
            pair_norm=df.pair_norm,
            eri_mat=df.eri_mat,
        )
        eri_payload = df_ints
        df_ints_ret = df_ints if bool(return_integrals) else None

    nelecas = getattr(mc, "nelecas", None)
    if nelecas is None:
        raise ValueError("mc.nelecas is required")
    if isinstance(nelecas, (tuple, list, np.ndarray)):
        nelec_act = int(nelecas[0]) + int(nelecas[1])
    else:
        nelec_act = int(nelecas)
    nelec_corr = nelec_act + 2 * correlate_inactive_i
    twos_i = _infer_twos(mc, twos=twos)
    orbsym_corr, wfnsym_i = _infer_orbsym_corr_and_wfnsym(
        mc,
        ncore_mc=ncore_mc,
        n_act=n_act_mc,
        n_virt=n_virt,
        ncore_frozen=ncore_frozen,
        orbsym=orbsym,
        wfnsym=wfnsym,
    )
    orbsym_act_int = None if orbsym_corr is None else orbsym_corr[:n_act_int]
    orbsym_act_small = None if orbsym_act_int is None else orbsym_act_int[correlate_inactive_i:]
    ci_cas_act_list = _get_states_ci(
        mc,
        states=states_list,
        n_act=n_act_mc,
        nelec_act=nelec_act,
        twos=twos_i,
        orbsym_act=orbsym_act_small,
        wfnsym=wfnsym_i,
    )

    ci_cas_list = []
    for ci_act in ci_cas_act_list:
        if correlate_inactive_i > 0:
            ci_cas_list.append(
                _embed_ci_with_docc_prefix(
                    ci_act=ci_act,
                    n_docc=correlate_inactive_i,
                    n_act=n_act_mc,
                    nelec_act=nelec_act,
                    twos=twos_i,
                    orbsym_act=orbsym_act_small,
                    orbsym_full=orbsym_act_int,
                    wfnsym=wfnsym_i,
                )
            )
        else:
            ci_cas_list.append(np.asarray(ci_act, dtype=np.float64))
    if integrals_backend_s == "pyscf_eri4":
        e_ref_list = [
            _compute_cas_reference_energy(
                h1e_corr=h1e_corr,
                eri4_corr=np.asarray(eri_payload, dtype=np.float64),
                ecore=ecore,
                ci_cas=ci_cas,
                n_act=n_act_int,
                nelec=nelec_corr,
                twos=twos_i,
                orbsym_act=orbsym_act_int,
                wfnsym=wfnsym_i,
            )
            for ci_cas in ci_cas_list
        ]
    else:
        e_ref_list = [
            _compute_cas_reference_energy_df(
                h1e_corr=h1e_corr,
                l_full=eri_payload.l_full,
                ecore=ecore,
                ci_cas=ci_cas,
                n_act=n_act_int,
                nelec=nelec_corr,
                twos=twos_i,
                orbsym_act=orbsym_act_int,
                wfnsym=wfnsym_i,
            )
            for ci_cas in ci_cas_list
        ]

    nthreads = int(contract_nthreads)
    if nthreads < 1:
        nthreads = 1

    mrci = mrcisd_kernel(
        h1e=h1e_corr,
        eri=eri_payload,
        n_act=n_act_int,
        n_virt=n_virt,
        nelec=nelec_corr,
        twos=twos_i,
        ci_cas=ci_cas_list,
        nroots=nroots_i,
        ecore=ecore,
        orbsym_act=orbsym_act_int,
        orbsym_corr=orbsym_corr,
        wfnsym=wfnsym_i,
        max_virt_e=max_virt_e,
        hop_backend=hop_backend,
        tol=tol,
        max_cycle=max_cycle,
        max_space=max_space,
        max_memory=max_memory_mb,
        contract_nthreads=nthreads,
        contract_blas_nthreads=contract_blas_nthreads,
        precompute_epq=precompute_epq,
    )
    if not isinstance(mrci, MRCISDResultMulti):
        raise RuntimeError("internal error: expected a multi-root MRCISD result")

    return MRCIFromMCStatesResult(
        method="mrcisd",
        states=states_list,
        nroots=int(nroots_i),
        e_ref=np.asarray(e_ref_list, dtype=np.float64),
        mrci=mrci,
        ecore=float(ecore),
        ncore=int(ncore_frozen),
        n_act=int(n_act_int),
        n_virt=int(n_virt),
        nelec=int(nelec_corr),
        twos=int(twos_i),
        integrals=ints_ret,
        df_integrals=df_ints_ret,
    )
