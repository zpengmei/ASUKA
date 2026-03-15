from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import get_state_cache
from asuka.cuguga.oracle.sparse import connected_row_sparse
from .labels import (
    coalesce_sparse_state,
    normalize_state_rep,
    requires_int64_labels,
    resolve_label_array,
    resolve_optional_label_index,
)
from .estimators import choose_reference_index, projected_energy_ref, rayleigh_energy_ref
from .sparse import (
    coalesce_coo_auto_f64,
    coalesce_coo_i32_f64,
    coalesce_coo_i64_f64,
    sparse_abs_l1_on_support,
    sparse_dot_sorted,
)


@dataclass(frozen=True)
class FCIFRIRun:
    idx: np.ndarray
    val: np.ndarray
    key_u64: np.ndarray | None
    label_kind: str
    energies: np.ndarray
    ref_idx: np.ndarray
    energy_estimator: str
    trial_cosine: np.ndarray | None = None
    trial_support_l1_frac: np.ndarray | None = None


@dataclass(frozen=True)
class FCIFRISubspaceRun:
    idx: list[np.ndarray]
    val: list[np.ndarray]
    energies: np.ndarray
    iters: np.ndarray
    backend: str


def _validate_large_space_ground_estimator(*, need_i64_labels: bool, energy_estimator: str) -> None:
    if bool(need_i64_labels) and str(energy_estimator).lower() == "rayleigh":
        raise ValueError("large-space FCI-FRI runs require energy_estimator='projected'; rayleigh is small-space only")


def _coalesce_block_column(
    drt: DRT,
    idx: np.ndarray,
    val: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    idx_u, val_u = coalesce_sparse_state(drt, idx=idx, key=None, val=val, name=name)
    if idx_u is None or val_u is None:
        raise ValueError(f"{name} is missing labels or values")
    if idx_u.size == 0:
        raise ValueError(f"{name} is empty")
    return idx_u, val_u


def _normalize_sparse_generic(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    idx_arr = np.asarray(idx).ravel()
    if idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError("sparse vector indices must have an integer dtype")
    val_arr = np.asarray(val, dtype=np.float64).ravel()
    if idx_arr.size != val_arr.size:
        raise ValueError("idx and val must have the same size")
    nrm = float(np.linalg.norm(val_arr))
    if nrm == 0.0:
        raise ValueError("vector norm is zero")
    return np.ascontiguousarray(idx_arr), np.asarray(val_arr / nrm, dtype=np.float64, order="C"), nrm


def _compress_phi_pivot_resample_generic(
    idx: np.ndarray,
    val: np.ndarray,
    *,
    m: int,
    pivot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    m = int(m)
    pivot = int(pivot)
    if m < 0:
        raise ValueError("m must be >= 0")
    if pivot < 0:
        raise ValueError("pivot must be >= 0")

    idx_arr = np.asarray(idx).ravel()
    if idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError("idx must have an integer dtype")
    if idx_arr.dtype.kind == "u":
        if idx_arr.size > 0 and int(np.max(idx_arr)) > np.iinfo(np.int64).max:
            raise ValueError("uint64 sparse labels must fit in int64")
        idx_arr = np.asarray(idx_arr, dtype=np.int64, order="C")
    idx_u, val_u = coalesce_coo_auto_f64(idx_arr, val, prefer_unsigned=False)
    out_dtype = idx_u.dtype if idx_u.size else (np.int64 if idx_arr.dtype.itemsize > 4 else np.int32)
    nent = int(idx_u.size)
    if m == 0 or nent == 0:
        return np.zeros(0, dtype=out_dtype), np.zeros(0, dtype=np.float64)
    if nent <= m:
        return idx_u, val_u

    abs_u = np.abs(val_u)
    nkeep = min(int(pivot), int(m), nent)
    if nkeep <= 0:
        piv_mask = np.zeros(nent, dtype=np.bool_)
        idx_p = np.zeros(0, dtype=out_dtype)
        val_p = np.zeros(0, dtype=np.float64)
    else:
        piv_pos = np.argpartition(abs_u, -nkeep)[-nkeep:]
        piv_mask = np.zeros(nent, dtype=np.bool_)
        piv_mask[piv_pos] = True
        idx_p = np.asarray(idx_u[piv_pos], dtype=out_dtype, order="C")
        val_p = np.asarray(val_u[piv_pos], dtype=np.float64, order="C")

    nsample = int(m - nkeep)
    idx_r = np.asarray(idx_u[~piv_mask], dtype=out_dtype, order="C")
    val_r = np.asarray(val_u[~piv_mask], dtype=np.float64, order="C")
    abs_r = np.abs(val_r)
    weight = float(np.sum(abs_r))
    if nsample <= 0 or weight == 0.0 or idx_r.size == 0:
        return coalesce_coo_auto_f64(idx_p, val_p, prefer_unsigned=False)

    step = weight / float(nsample)
    u0 = float(rng.random()) * step
    u = u0 + step * np.arange(nsample, dtype=np.float64)
    cdf = np.cumsum(abs_r, dtype=np.float64)
    pos = np.searchsorted(cdf, u, side="left")
    pos = np.asarray(pos, dtype=np.int64)
    pos = np.clip(pos, 0, idx_r.size - 1)

    idx_samp = np.asarray(idx_r[pos], dtype=out_dtype, order="C")
    val_samp = np.sign(val_r[pos]).astype(np.float64, copy=False) * step
    return coalesce_coo_auto_f64(
        np.concatenate((idx_p, idx_samp)),
        np.concatenate((val_p, val_samp)),
        prefer_unsigned=False,
    )


def _orthonormalize_mgs_generic(
    cols: list[tuple[np.ndarray, np.ndarray]],
    *,
    m: int,
    pivot: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k, (idx_k, val_k) in enumerate(cols):
        idx_cur, val_cur = coalesce_coo_auto_f64(idx_k, val_k, prefer_unsigned=False)
        if idx_cur.size == 0:
            raise ValueError(f"column {k} is empty")
        for i, (idx_i, val_i) in enumerate(out):
            ov = sparse_dot_sorted(idx_i, val_i, idx_cur, val_cur)
            if ov == 0.0:
                continue
            idx_cur, val_cur = coalesce_coo_auto_f64(
                np.concatenate((idx_cur, idx_i)),
                np.concatenate((val_cur, (-float(ov)) * np.asarray(val_i, dtype=np.float64))),
                prefer_unsigned=False,
            )
            idx_cur, val_cur = _compress_phi_pivot_resample_generic(
                idx_cur,
                val_cur,
                m=int(m),
                pivot=int(pivot),
                rng=rng,
            )
            if idx_cur.size == 0:
                raise ValueError(f"column {k} annihilated during orthonormalization (projection on {i})")
        idx_cur, val_cur, _ = _normalize_sparse_generic(idx_cur, val_cur)
        out.append((idx_cur, val_cur))
    return out


def _apply_right_matrix_generic(
    cols: list[tuple[np.ndarray, np.ndarray]],
    mat: np.ndarray,
    *,
    m: int,
    pivot: int,
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    mat_arr = np.asarray(mat, dtype=np.float64)
    nroots = int(len(cols))
    if mat_arr.shape != (nroots, nroots):
        raise ValueError(f"mat has wrong shape: {mat_arr.shape} (expected {(nroots, nroots)})")

    out: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(nroots):
        idx_chunks: list[np.ndarray] = []
        val_chunks: list[np.ndarray] = []
        for j in range(nroots):
            coeff = float(mat_arr[j, k])
            if coeff == 0.0:
                continue
            idx_j, val_j = cols[j]
            if np.asarray(idx_j).size == 0:
                continue
            idx_chunks.append(np.asarray(idx_j))
            val_chunks.append(coeff * np.asarray(val_j, dtype=np.float64))
        if not idx_chunks:
            raise ValueError("matrix application produced an empty column")
        idx_out, val_out = coalesce_coo_auto_f64(
            np.concatenate(idx_chunks),
            np.concatenate(val_chunks),
            prefer_unsigned=False,
        )
        idx_out, val_out = _compress_phi_pivot_resample_generic(
            idx_out,
            val_out,
            m=int(m),
            pivot=int(pivot),
            rng=rng,
        )
        idx_out, val_out, _ = _normalize_sparse_generic(idx_out, val_out)
        out.append((idx_out, val_out))
    return out


def _apply_hamiltonian_sparse_column(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    idx: np.ndarray,
    val: np.ndarray,
    *,
    state_cache: Any | None,
    max_out: int = 10_000_000,
) -> tuple[np.ndarray, np.ndarray]:
    idx_arr = np.asarray(idx).ravel()
    val_arr = np.asarray(val, dtype=np.float64).ravel()
    if idx_arr.size != val_arr.size:
        raise ValueError("idx and val must have the same size")

    idx_dtype = np.int64 if requires_int64_labels(drt, idx_arr) else np.int32
    idx_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for j, xj in zip(idx_arr.tolist(), val_arr.tolist(), strict=False):
        if float(xj) == 0.0:
            continue
        i_idx, hij = connected_row_sparse(
            drt,
            np.asarray(h1e, dtype=np.float64),
            eri,
            int(j),
            max_out=int(max_out),
            state_cache=state_cache,
        )
        if i_idx.size == 0:
            continue
        idx_parts.append(np.asarray(i_idx, dtype=idx_dtype, order="C"))
        val_parts.append(float(xj) * np.asarray(hij, dtype=np.float64, order="C"))

    if not idx_parts:
        return np.zeros(0, dtype=idx_dtype), np.zeros(0, dtype=np.float64)
    idx_all = np.concatenate(idx_parts)
    val_all = np.concatenate(val_parts)
    if idx_dtype == np.int64:
        return coalesce_coo_i64_f64(idx_all, val_all)
    return coalesce_coo_i32_f64(idx_all, val_all)


def _build_block_tmat_exact(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    cols: list[tuple[np.ndarray, np.ndarray]],
    *,
    state_cache: Any | None,
) -> np.ndarray:
    nroots = int(len(cols))
    hx_cols = [
        _apply_hamiltonian_sparse_column(
            drt,
            h1e,
            eri,
            idx_k,
            val_k,
            state_cache=state_cache,
        )
        for idx_k, val_k in cols
    ]
    tmat = np.zeros((nroots, nroots), dtype=np.float64)
    for i in range(nroots):
        idx_i, val_i = cols[i]
        for k in range(nroots):
            tmat[i, k] = sparse_dot_sorted(idx_i, val_i, hx_cols[k][0], hx_cols[k][1])
    return 0.5 * (tmat + tmat.T)

def run_fcifri_ground(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray | None,
    x_val: np.ndarray,
    *,
    m: int,
    eps: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    omp_threads: int | None = None,
    pivot: int = 256,
    initiator_na: float = 0.0,
    energy_stride: int = 1,
    energy_estimator: str = "projected",
    preferred_ref_idx: int | None = None,
    preferred_ref_key: int | np.integer | None = None,
    use_state_cache: bool = True,
    backend: str = "auto",
    state_rep: str = "auto",
    x_key: np.ndarray | None = None,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    spawner: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    spawner_kwargs: Mapping[str, object] | None = None,
    key64_pair_norm: np.ndarray | None = None,
    key64_pair_sampling_mode: int = 0,
    trial: object | None = None,
    trial_idx: np.ndarray | None = None,
    trial_key: np.ndarray | None = None,
    trial_val: np.ndarray | None = None,
    det_idx: np.ndarray | None = None,
    det_max_out: int = 10_000_000,
) -> FCIFRIRun:
    """Single-root FCI-FRI projector iteration (scalable CUDA uint64-label path).

    The ``trial`` parameter accepts any object with a ``to_qmc_x0(root=0)``
    method (e.g. :class:`CIPSITrialSpaceResult`).  When provided, it
    auto-populates ``trial_idx`` and ``trial_val`` (which must be ``None``).

    The ``det_idx`` parameter specifies CSF indices for the semi-stochastic
    deterministic subspace.  When provided, H·v contributions within this
    subspace are computed exactly each step, dramatically reducing noise.

    Trial diagnostics (``trial_cosine``, ``trial_support_l1_frac``) are
    computed at each energy checkpoint when a trial is supplied.
    """

    if trial is not None:
        if trial_idx is not None or trial_val is not None:
            raise ValueError("cannot specify both 'trial' and 'trial_idx'/'trial_val'")
        if not hasattr(trial, "to_qmc_x0"):
            raise TypeError("trial object must have a to_qmc_x0() method (e.g. CIPSITrialSpaceResult)")
        _t_idx, _t_val = trial.to_qmc_x0(root=0)
        trial_idx = np.asarray(_t_idx)
        trial_val = np.asarray(_t_val, dtype=np.float64)

    m = int(m)
    niter = int(niter)
    if m < 1:
        raise ValueError("m must be >= 1")
    if float(initiator_na) != 0.0 and m <= 1:
        raise ValueError("m must be > 1 when initiator_na > 0")
    if niter < 0:
        raise ValueError("niter must be >= 0")
    energy_stride = int(energy_stride)
    if energy_stride < 1:
        raise ValueError("energy_stride must be >= 1")

    energy_estimator = str(energy_estimator).lower()
    if energy_estimator not in ("projected", "rayleigh"):
        raise ValueError("energy_estimator must be 'projected' or 'rayleigh'")

    backend = str(backend).lower().strip()
    state_rep_s = normalize_state_rep(state_rep)
    if backend not in ("auto", "cuda_key64", "cuda_idx64"):
        raise ValueError("backend must be 'auto', 'cuda_key64', or 'cuda_idx64'")
    if state_rep_s == "i32":
        raise ValueError(
            "state_rep='i32' has been removed from the scalable FCI-FRI path; use state_rep='auto', 'i64'/'idx64', or 'key64'"
        )
    if int(drt.norb) > 64:
        raise ValueError("scalable CUDA FCI-FRI requires drt.norb <= 64")
    if omp_threads is not None:
        raise ValueError("omp_threads is unsupported for scalable CUDA FCI-FRI (CUDA path only)")
    if compressor is not None or spawner is not None or spawner_kwargs is not None:
        raise ValueError("custom compressor/spawner hooks are removed from scalable FCI-FRI; use default CUDA kernels")

    if backend == "auto":
        if state_rep_s == "key64":
            backend_effective = "cuda_key64"
        elif state_rep_s == "i64":
            backend_effective = "cuda_idx64"
        else:
            backend_effective = "cuda_key64" if int(drt.norb) <= 32 else "cuda_idx64"
    else:
        backend_effective = backend

    if backend_effective == "cuda_key64":
        if int(drt.norb) > 32:
            raise ValueError("backend='cuda_key64' requires drt.norb <= 32")
        if state_rep_s == "i64":
            raise ValueError("backend='cuda_key64' is incompatible with state_rep='i64'/'idx64'")
        label_mode = "key64"
    else:
        if state_rep_s == "key64":
            raise ValueError("backend='cuda_idx64' is incompatible with state_rep='key64'")
        if int(drt.ncsf) > np.iinfo(np.int64).max:
            raise ValueError("backend='cuda_idx64' requires drt.ncsf <= int64 max")
        label_mode = "idx64"

    x_idx_u, x_val_u = coalesce_sparse_state(drt, idx=x_idx, key=x_key, val=x_val, name="initial state")
    if x_idx_u is None or x_val_u is None:
        raise ValueError("initial state must be provided via x_idx/x_val or x_key/x_val")
    need_i64_labels = requires_int64_labels(drt, x_idx_u)
    _validate_large_space_ground_estimator(
        need_i64_labels=need_i64_labels,
        energy_estimator=energy_estimator,
    )
    preferred_ref_idx = resolve_optional_label_index(
        drt,
        idx=preferred_ref_idx,
        key=preferred_ref_key,
        name="preferred_ref",
    )
    state_cache = get_state_cache(drt) if (bool(use_state_cache) and not need_i64_labels) else None

    # --- trial wavefunction setup ---
    trial_idx_resolved = resolve_label_array(
        drt, idx=trial_idx, key=trial_key, name="trial",
    )
    has_trial = trial_idx_resolved is not None and trial_val is not None
    if has_trial:
        if need_i64_labels:
            trial_idx_u, trial_val_u = coalesce_coo_i64_f64(trial_idx_resolved, trial_val)
        else:
            trial_idx_u, trial_val_u = coalesce_coo_i32_f64(trial_idx_resolved, trial_val)
        if trial_idx_u.size == 0:
            raise ValueError("trial vector is empty")
        trial_l2 = float(np.linalg.norm(trial_val_u))
        if trial_l2 == 0.0:
            raise ValueError("trial vector has zero norm")
    else:
        trial_idx_u = None
        trial_val_u = None
        trial_l2 = 0.0

    if x_idx_u.size == 0:
        raise ValueError("initial x is empty")
    idx_dtype_out = np.int64 if need_i64_labels else np.int32

    n2 = float(np.linalg.norm(x_val_u))
    if n2 == 0.0:
        raise ValueError("initial x has zero norm")
    x_val_u = np.asarray(x_val_u / n2, dtype=np.float64, order="C")

    n_energy = (niter // energy_stride) + 1
    energies = np.empty(n_energy, dtype=np.float64)
    ref_hist = np.empty(n_energy, dtype=np.int64)
    trial_cosine_hist = np.full(n_energy, np.nan, dtype=np.float64) if has_trial else None
    trial_support_hist = np.full(n_energy, np.nan, dtype=np.float64) if has_trial else None
    ref0 = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
    if energy_estimator == "rayleigh":
        e0, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, state_cache=state_cache)
    else:
        e0, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, ref_idx=ref0, max_out=10_000_000, state_cache=state_cache)
    energies[0] = float(e0)
    ref_hist[0] = np.int64(ref0)
    if has_trial:
        x_l2 = float(np.linalg.norm(x_val_u))
        trial_cosine_hist[0] = (
            sparse_dot_sorted(x_idx_u, x_val_u, trial_idx_u, trial_val_u) / (x_l2 * trial_l2)
            if x_l2 > 0.0
            else 0.0
        )
        trial_support_hist[0] = (
            sparse_abs_l1_on_support(x_idx_u, x_val_u, trial_idx_u)
            / float(np.sum(np.abs(x_val_u)))
            if float(np.sum(np.abs(x_val_u))) > 0.0
            else 0.0
        )
    e_pos = 1

    from .cuda_backend import (  # noqa: PLC0415
        csf_idx_to_key64_host,
        cuda_projector_step_hamiltonian_u64_ws,
        key64_to_csf_idx_host,
        make_cuda_projector_context_idx64,
        make_cuda_projector_context_key64,
    )

    if backend_effective == "cuda_key64":
        key0 = csf_idx_to_key64_host(drt, x_idx_u, state_cache=state_cache)
    else:
        idx_i64 = np.asarray(x_idx_u, dtype=np.int64).ravel()
        if idx_i64.size:
            if int(np.min(idx_i64)) < 0:
                raise ValueError("initial state indices must be non-negative for idx64 backend")
            if int(np.max(idx_i64)) >= int(drt.ncsf):
                raise ValueError("initial state indices must be < drt.ncsf for idx64 backend")
        key0 = np.asarray(idx_i64, dtype=np.uint64, order="C")
    order0 = np.argsort(key0)
    key0 = np.asarray(key0[order0], dtype=np.uint64, order="C")
    val0 = np.asarray(x_val_u[order0], dtype=np.float64, order="C")

    if niter == 0:
        if label_mode == "key64":
            x_idx_out = key64_to_csf_idx_host(drt, key0, strict=True)
        else:
            x_idx_out = np.asarray(key0, dtype=np.int64, order="C")
        order_idx = np.argsort(x_idx_out)
        return FCIFRIRun(
            idx=np.asarray(x_idx_out[order_idx], dtype=idx_dtype_out, order="C"),
            val=np.asarray(val0[order_idx], dtype=np.float64, order="C"),
            key_u64=np.asarray(key0[order_idx], dtype=np.uint64, order="C"),
            label_kind="key64" if label_mode == "key64" else "idx64",
            energies=np.asarray(energies, dtype=np.float64, order="C"),
            ref_idx=np.asarray(ref_hist, dtype=np.int64, order="C"),
            energy_estimator=energy_estimator,
            trial_cosine=trial_cosine_hist,
            trial_support_l1_frac=trial_support_hist,
        )

    try:  # optional
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"scalable CUDA FCI-FRI backend requires cupy: {e}") from e

    pair_sampling_mode = int(key64_pair_sampling_mode)
    pair_norm = None
    pair_alias_prob = None
    pair_alias_idx = None
    pair_norm_sum = 0.0
    if pair_sampling_mode != 0:
        from .alias import build_alias_table_from_weights  # noqa: PLC0415

        if key64_pair_norm is None:
            raise ValueError("key64_pair_norm must be provided when key64_pair_sampling_mode!=0")
        alias = build_alias_table_from_weights(np.asarray(key64_pair_norm, dtype=np.float64).ravel())
        pair_norm = np.asarray(key64_pair_norm, dtype=np.float64).ravel()
        pair_alias_prob = alias.prob
        pair_alias_idx = alias.alias
        pair_norm_sum = float(alias.weight_sum)

    # Auto-adjust pivot to preserve deterministic subspace through Φ compression.
    pivot_eff = int(pivot)
    if det_idx is not None:
        ndet = int(np.unique(np.asarray(det_idx, dtype=np.int64).ravel()).size)
        pivot_eff = max(pivot_eff, ndet)

    if backend_effective == "cuda_key64":
        ctx = make_cuda_projector_context_key64(
            drt,
            h1e,
            eri,
            m=int(m),
            pivot=pivot_eff,
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            pair_alias_prob=pair_alias_prob,
            pair_alias_idx=pair_alias_idx,
            pair_norm=pair_norm,
            pair_norm_sum=float(pair_norm_sum),
            pair_sampling_mode=int(pair_sampling_mode),
            det_idx=det_idx,
            det_max_out=int(det_max_out),
        )
    else:
        ctx = make_cuda_projector_context_idx64(
            drt,
            h1e,
            eri,
            m=int(m),
            pivot=pivot_eff,
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            pair_alias_prob=pair_alias_prob,
            pair_alias_idx=pair_alias_idx,
            pair_norm=pair_norm,
            pair_norm_sum=float(pair_norm_sum),
            pair_sampling_mode=int(pair_sampling_mode),
            ncsf_u64=int(drt.ncsf),
            det_idx=det_idx,
            det_max_out=int(det_max_out),
        )
    rng = np.random.default_rng(int(seed))
    try:
        nnz0 = int(key0.size)
        if nnz0 > int(m):
            raise ValueError("initial x nnz exceeds m")
        ctx.x_key[:nnz0] = cp.asarray(key0, dtype=cp.uint64)
        ctx.x_val[:nnz0] = cp.asarray(val0, dtype=cp.float64)
        ctx.nnz = nnz0

        # GPU-resident trial vector (upload once, sorted by key)
        if has_trial:
            _trial_idx_np = np.asarray(trial_idx_u, dtype=np.int64)
            if label_mode == "key64":
                _trial_key_np = csf_idx_to_key64_host(drt, _trial_idx_np, state_cache=state_cache)
            else:
                _trial_key_np = np.asarray(_trial_idx_np, dtype=np.uint64)
            _t_order = np.argsort(_trial_key_np)
            trial_key_dev = cp.asarray(_trial_key_np[_t_order], dtype=cp.uint64)
            trial_val_dev = cp.asarray(
                np.asarray(trial_val_u, dtype=np.float64)[_t_order], dtype=cp.float64,
            )
            trial_l2_dev = float(cp.linalg.norm(trial_val_dev).get())
            del _trial_idx_np, _trial_key_np, _t_order

        def _gpu_sparse_dot_sorted(k_a, v_a, k_b, v_b):
            """⟨a|b⟩ for two GPU-resident sorted-key sparse vectors."""
            n_b = int(k_b.size)
            if n_b == 0 or int(k_a.size) == 0:
                return 0.0
            pos = cp.searchsorted(k_b, k_a)
            pos = cp.clip(pos, 0, max(n_b - 1, 0))
            match = k_b[pos] == k_a
            return float(cp.sum(v_a[match] * v_b[pos[match]]).get())

        # ---- Deterministic projected energy at checkpoints ----
        # At each energy_stride checkpoint: pull x from GPU, convert labels to CSF indices,
        # call projected_energy_ref (one row oracle, ~35ms). Zero per-step variance.

        for it in range(1, niter + 1):
            nnz_pre = int(ctx.nnz)
            initiator_t_dev = None
            if float(initiator_na) != 0.0:
                l1_dev = cp.sum(cp.abs(ctx.x_val[:nnz_pre]))
                initiator_t_dev = float(initiator_na) * l1_dev / float(m - 1)
            seed_spawn = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            seed_phi = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
            cuda_projector_step_hamiltonian_u64_ws(
                ctx,
                eps=float(eps),
                initiator_t=0.0,
                initiator_t_dev=initiator_t_dev,
                seed_spawn=seed_spawn,
                seed_phi=seed_phi,
                scale_identity=1.0,
                sync=True,
            )

            # Normalize (keeps amplitudes bounded across iterations).
            n2_dev = cp.linalg.norm(ctx.x_val[:ctx.nnz])
            ctx.x_val[:ctx.nnz] /= n2_dev

            if it % energy_stride == 0:
                # Download compressed x and compute exact projected energy on CPU.
                nnz_now = int(ctx.nnz)
                x_key_h = cp.asnumpy(ctx.x_key[:nnz_now]).view(np.uint64)
                x_val_h = cp.asnumpy(ctx.x_val[:nnz_now]).astype(np.float64, copy=False)
                if label_mode == "key64":
                    x_idx_h = key64_to_csf_idx_host(drt, x_key_h, strict=False)
                else:
                    x_idx_h = x_key_h.astype(np.int64)
                # projected_energy_ref needs x sorted by CSF index.
                sort_ord = np.argsort(x_idx_h)
                x_idx_sorted = x_idx_h[sort_ord]
                x_val_sorted = x_val_h[sort_ord]
                try:
                    e_val, _, _ = projected_energy_ref(
                        drt, h1e, eri,
                        x_idx_sorted, x_val_sorted,
                        ref_idx=ref0, max_out=10_000_000,
                        state_cache=state_cache,
                    )
                    energies[e_pos] = float(e_val)
                except Exception:
                    energies[e_pos] = np.nan
                ref_hist[e_pos] = np.int64(ref0)

                if has_trial:
                    x_key_new = ctx.x_key[:nnz_now]
                    x_val_new = ctx.x_val[:nnz_now]
                    x_l2 = float(cp.linalg.norm(x_val_new).get())
                    tc = _gpu_sparse_dot_sorted(
                        trial_key_dev, trial_val_dev, x_key_new, x_val_new,
                    ) / (x_l2 * trial_l2_dev) if x_l2 > 0 else 0.0
                    trial_cosine_hist[e_pos] = tc

                    t_pos = cp.searchsorted(x_key_new, trial_key_dev)
                    t_pos = cp.clip(t_pos, 0, max(nnz_now - 1, 0))
                    t_match = x_key_new[t_pos] == trial_key_dev
                    x_abs_on_trial = float(cp.sum(cp.abs(x_val_new[t_pos[t_match]])).get())
                    x_l1 = float(cp.sum(cp.abs(x_val_new)).get())
                    trial_support_hist[e_pos] = x_abs_on_trial / x_l1 if x_l1 > 0 else 0.0

                e_pos += 1

        x_key_u = cp.asnumpy(ctx.x_key[: ctx.nnz]).astype(np.uint64, copy=False)
        x_val_u = cp.asnumpy(ctx.x_val[: ctx.nnz]).astype(np.float64, copy=False)
    finally:
        ctx.release()

    if label_mode == "key64":
        x_idx_u = key64_to_csf_idx_host(drt, x_key_u, strict=True)
    else:
        x_idx_u = np.asarray(x_key_u, dtype=np.int64, order="C")
    order = np.argsort(x_idx_u)
    return FCIFRIRun(
        idx=np.asarray(x_idx_u[order], dtype=idx_dtype_out, order="C"),
        val=np.asarray(x_val_u[order], dtype=np.float64, order="C"),
        key_u64=np.asarray(x_key_u[order], dtype=np.uint64, order="C"),
        label_kind="key64" if label_mode == "key64" else "idx64",
        energies=np.asarray(energies, dtype=np.float64, order="C"),
        ref_idx=np.asarray(ref_hist, dtype=np.int64, order="C"),
        energy_estimator=energy_estimator,
        trial_cosine=trial_cosine_hist,
        trial_support_l1_frac=trial_support_hist,
    )


def run_fcifri_block(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    *,
    nroots: int,
    m: int,
    eps: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    omp_threads: int | None = None,
    pivot: int = 256,
    initiator_na: float = 0.0,
    ortho_stride: int = 1,
    ritz_stride: int = 5,
    rsi_alpha: float = 0.5,
    rsi_burn_in: int = 50,
    rsi_min_nsample: int = 10,
    rsi_compression: str = "pivotal",
    backend: str = "auto",
    use_state_cache: bool = True,
    x0: np.ndarray | list[tuple[np.ndarray, np.ndarray]] | None = None,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
    spawner: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
    spawner_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
) -> FCIFRISubspaceRun:
    """Scalable multi-root FCI-FRI block iteration.

    Projector steps run on the scalable CUDA `key64`/`idx64` single-root backend,
    one root at a time. Multi-root Ritz rotation and orthonormalization are done
    on sparse host vectors so the public block API remains available for large
    spaces without depending on the removed int32-only block kernels.
    """

    nroots = int(nroots)
    if nroots < 1:
        raise ValueError("nroots must be >= 1")
    m = int(m)
    if m < 1:
        raise ValueError("m must be >= 1")
    niter = int(niter)
    if niter < 0:
        raise ValueError("niter must be >= 0")
    ortho_stride = int(ortho_stride)
    if ortho_stride < 1:
        raise ValueError("ortho_stride must be >= 1")
    ritz_stride = int(ritz_stride)
    if ritz_stride < 1:
        raise ValueError("ritz_stride must be >= 1")
    if float(initiator_na) != 0.0 and m <= 1:
        raise ValueError("m must be > 1 when initiator_na > 0")

    backend_s = str(backend).lower().strip()
    if backend_s not in ("auto", "cuda_key64", "cuda_idx64"):
        raise ValueError("backend must be 'auto', 'cuda_key64', or 'cuda_idx64'")
    if int(drt.norb) > 64:
        raise ValueError("scalable CUDA FCI-FRI requires drt.norb <= 64")
    if omp_threads is not None:
        raise ValueError("omp_threads is unsupported for scalable run_fcifri_block (CUDA path only)")
    if compressor is not None or spawner is not None or spawner_kwargs is not None:
        raise ValueError("custom compressor/spawner hooks are removed from scalable run_fcifri_block")

    _ = (ortho_stride, rsi_alpha, rsi_burn_in, rsi_min_nsample, rsi_compression)

    if backend_s == "auto":
        backend_effective = "cuda_key64" if int(drt.norb) <= 32 else "cuda_idx64"
    else:
        backend_effective = backend_s
    if backend_effective == "cuda_key64":
        if int(drt.norb) > 32:
            raise ValueError("backend='cuda_key64' requires drt.norb <= 32")
        label_mode = "key64"
    else:
        if int(drt.ncsf) > np.iinfo(np.int64).max:
            raise ValueError("backend='cuda_idx64' requires drt.ncsf <= int64 max")
        label_mode = "idx64"

    if x0 is not None and hasattr(x0, "to_qmc_x0"):
        x0 = x0.to_qmc_x0()

    if x0 is None:
        if nroots > int(drt.ncsf):
            raise ValueError(f"default block initialization requires nroots <= drt.ncsf ({int(drt.ncsf)})")
        init_dtype = np.int64 if requires_int64_labels(drt) else np.int32
        cols = [
            (
                np.asarray([k], dtype=init_dtype),
                np.asarray([1.0], dtype=np.float64),
            )
            for k in range(nroots)
        ]
    elif isinstance(x0, list):
        if len(x0) != nroots:
            raise ValueError(f"x0 list must have length nroots={nroots}")
        cols = [
            _coalesce_block_column(drt, idx_k, val_k, name=f"x0[{k}]")
            for k, (idx_k, val_k) in enumerate(x0)
        ]
    else:
        raise NotImplementedError(
            "dense x0 arrays are removed from scalable run_fcifri_block; pass sparse x0=[(idx,val), ...] or a trial object with to_qmc_x0()"
        )

    need_i64_labels = requires_int64_labels(drt)
    idx_dtype_out = np.int64 if need_i64_labels else np.int32
    state_cache = get_state_cache(drt) if (bool(use_state_cache) and not need_i64_labels) else None
    rng = np.random.default_rng(int(seed))

    cols = [
        (
            np.asarray(idx_k, dtype=idx_dtype_out, order="C"),
            np.asarray(val_k, dtype=np.float64, order="C"),
        )
        for idx_k, val_k in cols
    ]
    cols = _orthonormalize_mgs_generic(cols, m=int(m), pivot=int(pivot), rng=rng)

    ncheck = (niter // ritz_stride) + 1
    energies = np.empty((ncheck, nroots), dtype=np.float64)
    iters = np.empty(ncheck, dtype=np.int32)
    chk = 0

    iters[chk] = np.int32(0)
    tmat = _build_block_tmat_exact(drt, h1e, eri, cols, state_cache=state_cache)
    w, c = np.linalg.eigh(tmat)
    energies[chk] = np.asarray(w, dtype=np.float64)
    chk += 1
    cols = _apply_right_matrix_generic(cols, c, m=int(m), pivot=int(pivot), rng=rng)
    cols = _orthonormalize_mgs_generic(cols, m=int(m), pivot=int(pivot), rng=rng)

    if niter == 0:
        return FCIFRISubspaceRun(
            idx=[np.asarray(idx_k, dtype=idx_dtype_out, order="C") for idx_k, _ in cols],
            val=[np.asarray(val_k, dtype=np.float64, order="C") for _, val_k in cols],
            energies=energies[:chk],
            iters=iters[:chk],
            backend=backend_effective,
        )

    try:  # optional
        import cupy as cp  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"scalable CUDA FCI-FRI block backend requires cupy: {e}") from e

    from .cuda_backend import (  # noqa: PLC0415
        csf_idx_to_key64_host,
        cuda_projector_step_hamiltonian_u64_ws,
        key64_to_csf_idx_host,
        make_cuda_projector_context_idx64,
        make_cuda_projector_context_key64,
    )

    def _encode_labels(idx_arr: np.ndarray) -> np.ndarray:
        if label_mode == "key64":
            return csf_idx_to_key64_host(drt, idx_arr, state_cache=state_cache)
        return np.asarray(idx_arr, dtype=np.uint64, order="C")

    def _make_ctx():
        if backend_effective == "cuda_key64":
            return make_cuda_projector_context_key64(
                drt,
                h1e,
                eri,
                m=int(m),
                pivot=int(pivot),
                nspawn_one=int(nspawn_one),
                nspawn_two=int(nspawn_two),
            )
        return make_cuda_projector_context_idx64(
            drt,
            h1e,
            eri,
            m=int(m),
            pivot=int(pivot),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
        )

    ctxs = [_make_ctx() for _ in range(nroots)]

    def _upload_cols(cols_in: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for ctx, (idx_k, val_k) in zip(ctxs, cols_in, strict=False):
            key_k = _encode_labels(idx_k)
            order_k = np.argsort(key_k)
            key_k = np.asarray(key_k[order_k], dtype=np.uint64, order="C")
            val_ord = np.asarray(val_k[order_k], dtype=np.float64, order="C")
            nnz_k = int(key_k.size)
            if nnz_k <= 0:
                raise ValueError("block column is empty")
            if nnz_k > int(m):
                raise ValueError(f"block column nnz exceeds m={int(m)}")
            ctx.x_key[:nnz_k] = cp.asarray(key_k, dtype=cp.uint64)
            ctx.x_val[:nnz_k] = cp.asarray(val_ord, dtype=cp.float64)
            ctx.nnz = nnz_k

    def _download_cols() -> list[tuple[np.ndarray, np.ndarray]]:
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for ctx in ctxs:
            nnz_k = int(ctx.nnz)
            key_k = cp.asnumpy(ctx.x_key[:nnz_k]).astype(np.uint64, copy=False)
            val_k = cp.asnumpy(ctx.x_val[:nnz_k]).astype(np.float64, copy=False)
            if label_mode == "key64":
                idx_k = key64_to_csf_idx_host(drt, key_k, strict=True)
            else:
                idx_k = np.asarray(key_k, dtype=np.int64, order="C")
            order_k = np.argsort(idx_k)
            out.append(
                (
                    np.asarray(idx_k[order_k], dtype=idx_dtype_out, order="C"),
                    np.asarray(val_k[order_k], dtype=np.float64, order="C"),
                )
            )
        return out

    _upload_cols(cols)
    try:
        for it in range(1, niter + 1):
            for ctx in ctxs:
                initiator_t = 0.0
                if float(initiator_na) != 0.0:
                    l1 = float(cp.sum(cp.abs(ctx.x_val[: ctx.nnz])).get())
                    initiator_t = float(initiator_na) * l1 / float(int(m) - 1)
                nnz_out = cuda_projector_step_hamiltonian_u64_ws(
                    ctx,
                    eps=float(eps),
                    initiator_t=float(initiator_t),
                    seed_spawn=int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)),
                    seed_phi=int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64)),
                    scale_identity=1.0,
                    sync=True,
                )
                if int(nnz_out) <= 0:
                    raise RuntimeError("block column collapsed to empty (try smaller eps or larger m)")

            cols_host: list[tuple[np.ndarray, np.ndarray]] | None = None
            if it % ortho_stride == 0:
                cols_host = _download_cols()
                cols_host = _orthonormalize_mgs_generic(cols_host, m=int(m), pivot=int(pivot), rng=rng)
                _upload_cols(cols_host)

            if it % ritz_stride == 0:
                if cols_host is None:
                    cols_host = _download_cols()
                iters[chk] = np.int32(it)
                tmat = _build_block_tmat_exact(drt, h1e, eri, cols_host, state_cache=state_cache)
                w, c = np.linalg.eigh(tmat)
                energies[chk] = np.asarray(w, dtype=np.float64)
                chk += 1
                cols_host = _apply_right_matrix_generic(cols_host, c, m=int(m), pivot=int(pivot), rng=rng)
                cols_host = _orthonormalize_mgs_generic(cols_host, m=int(m), pivot=int(pivot), rng=rng)
                _upload_cols(cols_host)

        cols = _download_cols()
    finally:
        for ctx in ctxs:
            ctx.release()

    return FCIFRISubspaceRun(
        idx=[np.asarray(idx_k, dtype=idx_dtype_out, order="C") for idx_k, _ in cols],
        val=[np.asarray(val_k, dtype=np.float64, order="C") for _, val_k in cols],
        energies=energies[:chk],
        iters=iters[:chk],
        backend=backend_effective,
    )

