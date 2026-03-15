from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import get_state_cache
from .labels import (
    coalesce_sparse_state,
    normalize_state_rep,
    requires_int64_labels,
    resolve_label_array,
    resolve_optional_label_index,
)
from .estimators import (
    choose_reference_index,
    projected_energy_ref,
    projected_energy_ref_status,
    rayleigh_energy_ref,
    sparse_abs_l1_on_support,
    sparse_dot_sorted,
)
from .sparse import coalesce_coo_i32_f64, coalesce_coo_i64_f64


_TRIAL_DYNAMIC_REF_POLICIES = {"dynamic_max_abs", "fixed_idx_strict", "fixed_trial_max_abs"}
_SHIFT_MIN_POP_FRAC = 0.8


@dataclass(frozen=True)
class FCIQMCRun:
    idx: np.ndarray
    val: np.ndarray
    key_u64: np.ndarray | None
    label_kind: str
    energies: np.ndarray
    shifts: np.ndarray
    populations: np.ndarray
    ref_idx: np.ndarray
    energy_estimator: str
    sample_iters: np.ndarray
    energies_projected_fixed: np.ndarray
    energies_projected_dynamic: np.ndarray
    energies_rayleigh: np.ndarray
    fixed_ref_idx: int
    dynamic_ref_idx: np.ndarray
    fixed_ref_alive: np.ndarray
    trial_cosine: np.ndarray
    trial_support_l1_frac: np.ndarray
    det_subspace_l1_frac: np.ndarray


@dataclass
class _FCIQMCTrace:
    sample_iters: np.ndarray
    energies_projected_fixed: np.ndarray
    energies_projected_dynamic: np.ndarray
    energies_rayleigh: np.ndarray
    dynamic_ref_idx: np.ndarray
    fixed_ref_alive: np.ndarray
    trial_cosine: np.ndarray
    trial_support_l1_frac: np.ndarray
    det_subspace_l1_frac: np.ndarray


@dataclass(frozen=True)
class _CheckpointMetrics:
    projected_fixed: float
    projected_dynamic: float
    rayleigh: float
    dynamic_ref_idx: int
    fixed_ref_alive: bool
    trial_cosine: float
    trial_support_l1_frac: float
    det_subspace_l1_frac: float


def update_shift(
    shift: float,
    population: float,
    *,
    target_population: float,
    dt: float,
    damping: float,
    log_clip: float | None = None,
    prev_population: float | None = None,
    shift_stride: int = 1,
) -> float:
    """Population-control shift update (NECI-style derivative controller).

    Uses the growth rate ``ln(N(t) / N(t-A))`` rather than the deviation from
    target ``ln(N / N_target)``.  This avoids the catastrophic overshoot that
    proportional-to-target control suffers in large active spaces where spawn
    rates are high and population can swing by 10×+ between shift updates.

    The standard NECI formula is::

        S(t) = S(t-A) - ζ / (A·δτ) · ln(N(t) / N(t-A))

    Parameters
    ----------
    shift : float
        Current shift value.
    population : float
        Current L1 walker population N(t).
    target_population : float
        Target population (used only for the min-pop gate in ``_maybe_update_shift``).
    dt : float
        Imaginary-time step δτ.
    damping : float
        Damping parameter ζ.  Typical range 0.01–0.10.
    log_clip : float or None
        Maximum absolute value of the log-ratio (prevents extreme jumps).
    prev_population : float or None
        Population at the previous shift-update point N(t-A).
        If None, falls back to proportional control ``ln(N / N_target)``.
    shift_stride : int
        Number of iterations between shift updates (A).
    """

    shift = float(shift)
    pop = float(population)
    tgt = float(target_population)
    dt = float(dt)
    damp = float(damping)
    stride = max(1, int(shift_stride))
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if damp <= 0.0:
        return shift
    if pop <= 0.0 or tgt <= 0.0:
        raise ValueError("population and target_population must be > 0")
    if prev_population is not None and float(prev_population) > 0.0:
        log_ratio = float(np.log(pop / float(prev_population)))
    else:
        log_ratio = float(np.log(pop / tgt))
    if log_clip is not None:
        clip = abs(float(log_clip))
        if clip == 0.0:
            log_ratio = 0.0
        else:
            log_ratio = float(np.clip(log_ratio, -clip, clip))
    return shift - (damp / (stride * dt)) * log_ratio


def _validate_trial_inputs(
    *,
    x_idx_u: np.ndarray,
    x_val_u: np.ndarray,
    trial_idx: np.ndarray | None,
    trial_val: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    if (trial_idx is None) != (trial_val is None):
        raise ValueError("trial_idx and trial_val must be provided together")
    if trial_idx is None:
        trial_idx_u = np.asarray(x_idx_u, dtype=np.asarray(x_idx_u).dtype, order="C")
        trial_val_u = np.asarray(x_val_u, dtype=np.float64, order="C")
    else:
        if np.asarray(x_idx_u).dtype == np.int64:
            trial_idx_u, trial_val_u = coalesce_coo_i64_f64(trial_idx, trial_val)
        else:
            trial_idx_u, trial_val_u = coalesce_coo_i32_f64(trial_idx, trial_val)
    if trial_idx_u.size == 0:
        raise ValueError("trial vector is empty")
    trial_l2 = float(np.linalg.norm(trial_val_u))
    if trial_l2 == 0.0:
        raise ValueError("trial vector has zero norm")
    return trial_idx_u, trial_val_u, trial_l2


def _resolve_reference_policy(
    *,
    reference_policy: str,
    preferred_ref_idx: int | None,
    trial_idx_u: np.ndarray,
    trial_val_u: np.ndarray,
) -> int:
    policy = str(reference_policy).lower()
    if policy not in _TRIAL_DYNAMIC_REF_POLICIES:
        raise ValueError(
            "reference_policy must be 'dynamic_max_abs', 'fixed_idx_strict', or 'fixed_trial_max_abs'"
        )
    if policy == "fixed_idx_strict":
        if preferred_ref_idx is None:
            raise ValueError("preferred_ref_idx must be provided when reference_policy='fixed_idx_strict'")
        return int(preferred_ref_idx)
    k = int(np.argmax(np.abs(np.asarray(trial_val_u, dtype=np.float64).ravel())))
    return int(np.asarray(trial_idx_u).ravel()[k])


def _prepare_det_subspace(
    deterministic_subspace_idx: np.ndarray | None,
    *,
    fixed_ref_idx: int,
) -> np.ndarray:
    if deterministic_subspace_idx is None:
        return np.zeros(0, dtype=np.int64)
    det_idx = np.asarray(deterministic_subspace_idx, dtype=np.int64).ravel()
    if det_idx.size == 0:
        return np.zeros(0, dtype=np.int64)
    det_idx = np.unique(np.concatenate((det_idx, np.asarray([int(fixed_ref_idx)], dtype=np.int64))))
    det_idx.sort()
    return np.asarray(det_idx, dtype=np.int64, order="C")


def _make_trace(niter: int, energy_stride: int) -> _FCIQMCTrace:
    n_sample = (int(niter) // int(energy_stride)) + 1
    return _FCIQMCTrace(
        sample_iters=np.empty(n_sample, dtype=np.int32),
        energies_projected_fixed=np.full(n_sample, np.nan, dtype=np.float64),
        energies_projected_dynamic=np.full(n_sample, np.nan, dtype=np.float64),
        energies_rayleigh=np.full(n_sample, np.nan, dtype=np.float64),
        dynamic_ref_idx=np.empty(n_sample, dtype=np.int64),
        fixed_ref_alive=np.zeros(n_sample, dtype=np.bool_),
        trial_cosine=np.full(n_sample, np.nan, dtype=np.float64),
        trial_support_l1_frac=np.full(n_sample, np.nan, dtype=np.float64),
        det_subspace_l1_frac=np.zeros(n_sample, dtype=np.float64),
    )


def _need_rayleigh(it: int, *, rayleigh_stride: int) -> bool:
    return int(rayleigh_stride) > 0 and (int(it) % int(rayleigh_stride) == 0)


def _validate_large_space_estimator(
    *,
    need_i64_labels: bool,
    energy_estimator: str,
    rayleigh_stride: int,
) -> None:
    if not bool(need_i64_labels):
        return
    if str(energy_estimator).lower() == "rayleigh":
        raise ValueError("large-space FCIQMC runs require energy_estimator='projected'; rayleigh is small-space only")
    if int(rayleigh_stride) > 0:
        raise ValueError("large-space FCIQMC runs do not support rayleigh_stride diagnostics; set rayleigh_stride=0")


def _evaluate_checkpoint(
    *,
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx_u: np.ndarray,
    x_val_u: np.ndarray,
    fixed_ref_idx: int,
    preferred_ref_idx: int | None,
    trial_idx_u: np.ndarray,
    trial_val_u: np.ndarray,
    trial_l2: float,
    det_idx_u: np.ndarray,
    compute_rayleigh: bool,
    state_cache,
    max_out: int = 10_000_000,
) -> _CheckpointMetrics:
    dynamic_ref_idx = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
    e_dynamic, _, _ = projected_energy_ref(
        drt,
        h1e,
        eri,
        x_idx_u,
        x_val_u,
        ref_idx=int(dynamic_ref_idx),
        max_out=int(max_out),
        state_cache=state_cache,
    )
    e_fixed, _, _, fixed_ref_alive = projected_energy_ref_status(
        drt,
        h1e,
        eri,
        x_idx_u,
        x_val_u,
        ref_idx=int(fixed_ref_idx),
        max_out=int(max_out),
        state_cache=state_cache,
    )
    e_rayleigh = np.nan
    if bool(compute_rayleigh):
        e_rayleigh, _, _ = rayleigh_energy_ref(
            drt,
            h1e,
            eri,
            x_idx_u,
            x_val_u,
            state_cache=state_cache,
        )

    x_l2 = float(np.linalg.norm(x_val_u))
    trial_cosine = np.nan
    if x_l2 > 0.0:
        trial_cosine = float(
            sparse_dot_sorted(x_idx_u, x_val_u, trial_idx_u, trial_val_u) / (x_l2 * float(trial_l2))
        )
    pop_l1 = float(np.sum(np.abs(x_val_u), dtype=np.float64))
    trial_support_l1_frac = np.nan
    det_subspace_l1_frac = 0.0
    if pop_l1 > 0.0:
        trial_support_l1_frac = float(sparse_abs_l1_on_support(x_idx_u, x_val_u, trial_idx_u) / pop_l1)
        if int(det_idx_u.size) > 0:
            det_subspace_l1_frac = float(sparse_abs_l1_on_support(x_idx_u, x_val_u, det_idx_u) / pop_l1)

    return _CheckpointMetrics(
        projected_fixed=float(e_fixed),
        projected_dynamic=float(e_dynamic),
        rayleigh=float(e_rayleigh),
        dynamic_ref_idx=int(dynamic_ref_idx),
        fixed_ref_alive=bool(fixed_ref_alive),
        trial_cosine=float(trial_cosine),
        trial_support_l1_frac=float(trial_support_l1_frac),
        det_subspace_l1_frac=float(det_subspace_l1_frac),
    )


def _write_checkpoint(trace: _FCIQMCTrace, pos: int, it: int, metrics: _CheckpointMetrics) -> None:
    trace.sample_iters[pos] = np.int32(it)
    trace.energies_projected_fixed[pos] = float(metrics.projected_fixed)
    trace.energies_projected_dynamic[pos] = float(metrics.projected_dynamic)
    trace.energies_rayleigh[pos] = float(metrics.rayleigh)
    trace.dynamic_ref_idx[pos] = np.int64(metrics.dynamic_ref_idx)
    trace.fixed_ref_alive[pos] = bool(metrics.fixed_ref_alive)
    trace.trial_cosine[pos] = float(metrics.trial_cosine)
    trace.trial_support_l1_frac[pos] = float(metrics.trial_support_l1_frac)
    trace.det_subspace_l1_frac[pos] = float(metrics.det_subspace_l1_frac)


def _select_primary_history(
    *,
    trace: _FCIQMCTrace,
    energy_estimator: str,
    reference_policy: str,
    fixed_ref_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    estimator = str(energy_estimator).lower()
    policy = str(reference_policy).lower()
    if estimator == "rayleigh":
        return (
            np.asarray(trace.energies_rayleigh, dtype=np.float64, order="C"),
            np.asarray(trace.dynamic_ref_idx, dtype=np.int64, order="C"),
        )
    if policy == "dynamic_max_abs":
        return (
            np.asarray(trace.energies_projected_dynamic, dtype=np.float64, order="C"),
            np.asarray(trace.dynamic_ref_idx, dtype=np.int64, order="C"),
        )
    return (
        np.asarray(trace.energies_projected_fixed, dtype=np.float64, order="C"),
        np.full(trace.sample_iters.shape, np.int64(fixed_ref_idx), dtype=np.int64),
    )


def _build_run(
    *,
    idx: np.ndarray,
    val: np.ndarray,
    key_u64: np.ndarray | None,
    label_kind: str,
    shifts: np.ndarray,
    populations: np.ndarray,
    trace: _FCIQMCTrace,
    energy_estimator: str,
    reference_policy: str,
    fixed_ref_idx: int,
) -> FCIQMCRun:
    energies, ref_idx = _select_primary_history(
        trace=trace,
        energy_estimator=energy_estimator,
        reference_policy=reference_policy,
        fixed_ref_idx=fixed_ref_idx,
    )
    idx_arr = np.asarray(idx).ravel()
    if idx_arr.dtype.kind not in ("i", "u"):
        raise ValueError("idx must have an integer dtype")
    return FCIQMCRun(
        idx=np.asarray(idx_arr, dtype=idx_arr.dtype, order="C"),
        val=np.asarray(val, dtype=np.float64, order="C"),
        key_u64=None if key_u64 is None else np.asarray(key_u64, dtype=np.uint64, order="C"),
        label_kind=str(label_kind),
        energies=energies,
        shifts=np.asarray(shifts, dtype=np.float64, order="C"),
        populations=np.asarray(populations, dtype=np.float64, order="C"),
        ref_idx=ref_idx,
        energy_estimator=str(energy_estimator),
        sample_iters=np.asarray(trace.sample_iters, dtype=np.int32, order="C"),
        energies_projected_fixed=np.asarray(trace.energies_projected_fixed, dtype=np.float64, order="C"),
        energies_projected_dynamic=np.asarray(trace.energies_projected_dynamic, dtype=np.float64, order="C"),
        energies_rayleigh=np.asarray(trace.energies_rayleigh, dtype=np.float64, order="C"),
        fixed_ref_idx=int(fixed_ref_idx),
        dynamic_ref_idx=np.asarray(trace.dynamic_ref_idx, dtype=np.int64, order="C"),
        fixed_ref_alive=np.asarray(trace.fixed_ref_alive, dtype=np.bool_, order="C"),
        trial_cosine=np.asarray(trace.trial_cosine, dtype=np.float64, order="C"),
        trial_support_l1_frac=np.asarray(trace.trial_support_l1_frac, dtype=np.float64, order="C"),
        det_subspace_l1_frac=np.asarray(trace.det_subspace_l1_frac, dtype=np.float64, order="C"),
    )


def _maybe_update_shift(
    *,
    shift: float,
    pop: float,
    it: int,
    dt: float,
    tgt_pop: float,
    shift_damping: float,
    shift_stride: int,
    shift_start: int,
    shift_warmup_iters: int,
    shift_log_clip: float | None,
    prev_population: float | None = None,
) -> float:
    if float(shift_damping) <= 0.0:
        return float(shift)
    if int(it) < max(int(shift_start), int(shift_warmup_iters)):
        return float(shift)
    if int(it) % int(shift_stride) != 0:
        return float(shift)
    if float(pop) < (_SHIFT_MIN_POP_FRAC * float(tgt_pop)):
        return float(shift)
    return update_shift(
        shift,
        pop,
        target_population=float(tgt_pop),
        dt=float(dt),
        damping=float(shift_damping),
        log_clip=shift_log_clip,
        prev_population=prev_population,
        shift_stride=int(shift_stride),
    )


def run_fciqmc(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray | None,
    x_val: np.ndarray,
    *,
    dt: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    backend: str = "auto",
    state_rep: str = "auto",
    x_key: np.ndarray | None = None,
    max_walker: int | None = None,
    omp_threads: int | None = None,
    shift_init: float = 0.0,
    target_population: float | None = None,
    shift_damping: float = 0.0,
    shift_stride: int = 1,
    shift_start: int = 0,
    energy_stride: int = 1,
    energy_estimator: str = "projected",
    preferred_ref_idx: int | None = None,
    preferred_ref_key: int | np.integer | None = None,
    reference_policy: str = "dynamic_max_abs",
    trial: object | None = None,
    trial_idx: np.ndarray | None = None,
    trial_key: np.ndarray | None = None,
    trial_val: np.ndarray | None = None,
    deterministic_subspace_idx: np.ndarray | None = None,
    deterministic_subspace_key: np.ndarray | None = None,
    shift_warmup_iters: int = 0,
    shift_log_clip: float | None = None,
    rayleigh_stride: int = 0,
    initiator_t: float = 0.0,
    use_state_cache: bool = True,
    key64_pair_norm: np.ndarray | None = None,
    key64_pair_sampling_mode: int = 0,
) -> FCIQMCRun:
    """Single-root CSF-native FCIQMC loop (scalable CUDA uint64-label path).

    The ``trial`` parameter accepts any object with a ``to_qmc_x0(root=0)``
    method (e.g. :class:`CIPSITrialSpaceResult`).  When provided, it
    auto-populates ``trial_idx`` and ``trial_val`` (which must be ``None``).
    """

    if trial is not None:
        if trial_idx is not None or trial_val is not None:
            raise ValueError("cannot specify both 'trial' and 'trial_idx'/'trial_val'")
        if not hasattr(trial, "to_qmc_x0"):
            raise TypeError("trial object must have a to_qmc_x0() method (e.g. CIPSITrialSpaceResult)")
        _t_idx, _t_val = trial.to_qmc_x0(root=0)
        trial_idx = np.asarray(_t_idx)
        trial_val = np.asarray(_t_val, dtype=np.float64)

    dt = float(dt)
    niter = int(niter)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if niter < 0:
        raise ValueError("niter must be >= 0")
    shift_stride = int(shift_stride)
    shift_start = int(shift_start)
    energy_stride = int(energy_stride)
    shift_warmup_iters = int(shift_warmup_iters)
    rayleigh_stride = int(rayleigh_stride)
    if shift_stride < 1:
        raise ValueError("shift_stride must be >= 1")
    if shift_start < 0:
        raise ValueError("shift_start must be >= 0")
    if shift_warmup_iters < 0:
        raise ValueError("shift_warmup_iters must be >= 0")
    if energy_stride < 1:
        raise ValueError("energy_stride must be >= 1")
    if rayleigh_stride < 0:
        raise ValueError("rayleigh_stride must be >= 0")

    energy_estimator = str(energy_estimator).lower()
    if energy_estimator not in ("projected", "rayleigh"):
        raise ValueError("energy_estimator must be 'projected' or 'rayleigh'")
    if energy_estimator == "rayleigh" and rayleigh_stride == 0:
        rayleigh_stride = energy_stride

    backend_s = str(backend).lower().strip()
    state_rep_s = normalize_state_rep(state_rep)
    if backend_s not in ("auto", "cuda_key64", "cuda_idx64"):
        raise ValueError("backend must be 'auto', 'cuda_key64', or 'cuda_idx64'")
    if state_rep_s == "i32":
        raise ValueError(
            "state_rep='i32' has been removed from the scalable FCIQMC path; use state_rep='auto', 'i64'/'idx64', or 'key64'"
        )
    if int(drt.norb) > 64:
        raise ValueError("scalable CUDA FCIQMC requires drt.norb <= 64")
    if omp_threads is not None:
        raise ValueError("omp_threads is unsupported for scalable CUDA FCIQMC (CUDA path only)")

    if backend_s == "auto":
        if state_rep_s == "key64":
            backend_effective = "cuda_key64"
        elif state_rep_s == "i64":
            backend_effective = "cuda_idx64"
        else:
            backend_effective = "cuda_key64" if int(drt.norb) <= 32 else "cuda_idx64"
    else:
        backend_effective = backend_s

    if backend_effective == "cuda_key64":
        if int(drt.norb) > 32:
            raise ValueError("backend='cuda_key64' requires drt.norb <= 32")
        if state_rep_s == "i64":
            raise ValueError("backend='cuda_key64' is incompatible with state_rep='i64'/'idx64'")
    else:
        if state_rep_s == "key64":
            raise ValueError("backend='cuda_idx64' is incompatible with state_rep='key64'")
        if int(drt.ncsf) > np.iinfo(np.int64).max:
            raise ValueError("backend='cuda_idx64' requires drt.ncsf <= int64 max")

    x_idx_u, x_val_u = coalesce_sparse_state(drt, idx=x_idx, key=x_key, val=x_val, name="initial state")
    if x_idx_u is None or x_val_u is None:
        raise ValueError("initial state must be provided via x_idx/x_val or x_key/x_val")
    need_i64_labels = requires_int64_labels(drt, x_idx_u)
    _validate_large_space_estimator(
        need_i64_labels=need_i64_labels,
        energy_estimator=energy_estimator,
        rayleigh_stride=rayleigh_stride,
    )
    if x_idx_u.size == 0:
        raise ValueError("initial x is empty")

    preferred_ref_idx = resolve_optional_label_index(
        drt,
        idx=preferred_ref_idx,
        key=preferred_ref_key,
        name="preferred_ref",
    )
    trial_idx_resolved = resolve_label_array(drt, idx=trial_idx, key=trial_key, name="trial state")
    det_idx_resolved = resolve_label_array(
        drt,
        idx=deterministic_subspace_idx,
        key=deterministic_subspace_key,
        name="deterministic_subspace",
    )
    trial_idx_u, trial_val_u, trial_l2 = _validate_trial_inputs(
        x_idx_u=x_idx_u,
        x_val_u=x_val_u,
        trial_idx=trial_idx_resolved,
        trial_val=trial_val,
    )
    fixed_ref_idx = _resolve_reference_policy(
        reference_policy=reference_policy,
        preferred_ref_idx=preferred_ref_idx,
        trial_idx_u=trial_idx_u,
        trial_val_u=trial_val_u,
    )
    det_idx_u = _prepare_det_subspace(det_idx_resolved, fixed_ref_idx=fixed_ref_idx)

    shift = float(shift_init)
    pops = np.empty(niter + 1, dtype=np.float64)
    shifts = np.empty(niter + 1, dtype=np.float64)
    trace = _make_trace(niter=niter, energy_stride=energy_stride)

    pop0 = float(np.sum(np.abs(x_val_u), dtype=np.float64))
    if pop0 == 0.0:
        raise ValueError("initial population is zero")
    tgt_pop = float(pop0 if target_population is None else float(target_population))
    if tgt_pop <= 0.0:
        raise ValueError("target_population must be > 0")

    use_state_cache_eff = bool(use_state_cache) and not bool(need_i64_labels)
    state_cache = get_state_cache(drt) if use_state_cache_eff else None

    metrics0 = _evaluate_checkpoint(
        drt=drt,
        h1e=h1e,
        eri=eri,
        x_idx_u=x_idx_u,
        x_val_u=x_val_u,
        fixed_ref_idx=int(fixed_ref_idx),
        preferred_ref_idx=preferred_ref_idx,
        trial_idx_u=trial_idx_u,
        trial_val_u=trial_val_u,
        trial_l2=float(trial_l2),
        det_idx_u=det_idx_u,
        compute_rayleigh=_need_rayleigh(0, rayleigh_stride=rayleigh_stride),
        state_cache=state_cache,
    )
    _write_checkpoint(trace, 0, 0, metrics0)

    pops[0] = pop0
    shifts[0] = shift

    if niter == 0:
        if backend_effective == "cuda_key64":
            from .cuda_backend import csf_idx_to_key64_host  # noqa: PLC0415

            key_u64 = csf_idx_to_key64_host(drt, x_idx_u, state_cache=state_cache)
        else:
            idx_i64 = np.asarray(x_idx_u, dtype=np.int64).ravel()
            if idx_i64.size:
                if int(np.min(idx_i64)) < 0:
                    raise ValueError("initial state indices must be non-negative for idx64 backend")
                if int(np.max(idx_i64)) >= int(drt.ncsf):
                    raise ValueError("initial state indices must be < drt.ncsf for idx64 backend")
            key_u64 = np.asarray(idx_i64, dtype=np.uint64, order="C")
        return _build_run(
            idx=x_idx_u,
            val=x_val_u,
            key_u64=np.asarray(key_u64, dtype=np.uint64, order="C"),
            label_kind="key64" if backend_effective == "cuda_key64" else "idx64",
            shifts=shifts,
            populations=pops,
            trace=trace,
            energy_estimator=energy_estimator,
            reference_policy=reference_policy,
            fixed_ref_idx=fixed_ref_idx,
        )

    common_kwargs = dict(
        drt=drt,
        h1e=h1e,
        eri=eri,
        x_idx_u=x_idx_u,
        x_val_u=x_val_u,
        dt=dt,
        niter=niter,
        nspawn_one=nspawn_one,
        nspawn_two=nspawn_two,
        seed=seed,
        max_walker=max_walker,
        shift=shift,
        tgt_pop=tgt_pop,
        shift_damping=float(shift_damping),
        shift_stride=int(shift_stride),
        shift_start=int(shift_start),
        shift_warmup_iters=int(shift_warmup_iters),
        shift_log_clip=shift_log_clip,
        energy_stride=int(energy_stride),
        preferred_ref_idx=preferred_ref_idx,
        fixed_ref_idx=fixed_ref_idx,
        initiator_t=float(initiator_t),
        pops=pops,
        shifts=shifts,
        trace=trace,
        trial_idx_u=trial_idx_u,
        trial_val_u=trial_val_u,
        trial_l2=float(trial_l2),
        det_idx_u=det_idx_u,
        rayleigh_stride=int(rayleigh_stride),
        state_cache=state_cache,
        energy_estimator=str(energy_estimator),
        reference_policy=str(reference_policy),
        key64_pair_norm=key64_pair_norm,
        key64_pair_sampling_mode=int(key64_pair_sampling_mode),
    )
    if backend_effective == "cuda_key64":
        return _run_fciqmc_cuda_key64(**common_kwargs)
    return _run_fciqmc_cuda_idx64(**common_kwargs)


def _run_fciqmc_cuda_key64(
    *,
    drt,
    h1e,
    eri,
    x_idx_u: np.ndarray,
    x_val_u: np.ndarray,
    dt: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    max_walker: int | None,
    shift: float,
    tgt_pop: float,
    shift_damping: float,
    shift_stride: int,
    shift_start: int,
    shift_warmup_iters: int,
    shift_log_clip: float | None,
    energy_stride: int,
    preferred_ref_idx: int | None,
    fixed_ref_idx: int,
    initiator_t: float,
    pops: np.ndarray,
    shifts: np.ndarray,
    trace: _FCIQMCTrace,
    trial_idx_u: np.ndarray,
    trial_val_u: np.ndarray,
    trial_l2: float,
    det_idx_u: np.ndarray,
    rayleigh_stride: int,
    state_cache,
    energy_estimator: str,
    reference_policy: str,
    key64_pair_norm: np.ndarray | None,
    key64_pair_sampling_mode: int,
) -> FCIQMCRun:
    from .cuda_backend import (  # noqa: PLC0415
        csf_idx_to_key64_host,
        cuda_fciqmc_step_hamiltonian_u64_ws,
        key64_to_csf_idx64_host,
        make_cuda_fciqmc_context_key64,
    )

    norb = int(drt.norb)
    if norb > 32:
        raise ValueError("backend='cuda_key64' requires drt.norb <= 32")

    if max_walker is None:
        max_walker = max(1024, 4 * int(x_idx_u.size))
    max_walker = int(max_walker)

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

    ctx = make_cuda_fciqmc_context_key64(
        drt,
        h1e,
        eri,
        max_walker=max_walker,
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        det_idx=det_idx_u if int(det_idx_u.size) > 0 else None,
        pair_alias_prob=pair_alias_prob,
        pair_alias_idx=pair_alias_idx,
        pair_norm=pair_norm,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
    )

    nnz0 = int(x_idx_u.size)
    import cupy as cp  # noqa: PLC0415

    idx_dtype_out = np.asarray(x_idx_u).dtype
    key0 = csf_idx_to_key64_host(drt, x_idx_u, state_cache=state_cache)
    order0 = np.argsort(key0)
    key0 = np.asarray(key0[order0], dtype=np.uint64, order="C")
    val0 = np.asarray(x_val_u[order0], dtype=np.float64, order="C")
    ctx.x_key[:nnz0] = cp.asarray(key0, dtype=cp.uint64)
    ctx.x_val[:nnz0] = cp.asarray(val0, dtype=cp.float64)
    ctx.nnz = nnz0

    sample_pos = 1
    prev_pop_at_shift_update = float(pops[0])
    try:
        pops_dev = cp.empty(int(niter) + 1, dtype=cp.float64)
        pops_dev[0] = float(pops[0])
        for it in range(1, niter + 1):
            cuda_fciqmc_step_hamiltonian_u64_ws(
                ctx,
                dt=dt,
                shift=shift,
                initiator_t=float(initiator_t),
                seed_spawn=int(seed) + it,
                sync=True,
            )

            nnz_now = int(ctx.nnz)
            pop_dev = cp.sum(cp.abs(ctx.x_val[:nnz_now]))
            pops_dev[it] = pop_dev
            shifts[it] = shift

            if float(shift_damping) > 0.0 and it >= max(int(shift_start), int(shift_warmup_iters)) and (it % int(shift_stride) == 0):
                pop = float(pop_dev.get())
                shift = _maybe_update_shift(
                    shift=shift,
                    pop=pop,
                    it=it,
                    dt=dt,
                    tgt_pop=tgt_pop,
                    shift_damping=shift_damping,
                    shift_stride=shift_stride,
                    shift_start=shift_start,
                    shift_warmup_iters=shift_warmup_iters,
                    shift_log_clip=shift_log_clip,
                    prev_population=prev_pop_at_shift_update,
                )
                prev_pop_at_shift_update = pop

            if it % int(energy_stride) == 0:
                x_key_h = cp.asnumpy(ctx.x_key[:nnz_now]).astype(np.uint64, copy=False)
                x_val_h = cp.asnumpy(ctx.x_val[:nnz_now]).astype(np.float64, copy=False)
                x_idx_h = key64_to_csf_idx64_host(drt, x_key_h, strict=True)
                order = np.argsort(x_idx_h)
                x_idx_h = np.asarray(x_idx_h[order], dtype=idx_dtype_out, order="C")
                x_val_h = np.asarray(x_val_h[order], dtype=np.float64, order="C")
                metrics = _evaluate_checkpoint(
                    drt=drt,
                    h1e=h1e,
                    eri=eri,
                    x_idx_u=x_idx_h,
                    x_val_u=x_val_h,
                    fixed_ref_idx=int(fixed_ref_idx),
                    preferred_ref_idx=preferred_ref_idx,
                    trial_idx_u=trial_idx_u,
                    trial_val_u=trial_val_u,
                    trial_l2=float(trial_l2),
                    det_idx_u=det_idx_u,
                    compute_rayleigh=_need_rayleigh(it, rayleigh_stride=rayleigh_stride),
                    state_cache=state_cache,
                )
                _write_checkpoint(trace, sample_pos, it, metrics)
                sample_pos += 1

        pops[1:] = cp.asnumpy(pops_dev[1:]).astype(np.float64, copy=False)

        nnz_final = int(ctx.nnz)
        x_key_out = cp.asnumpy(ctx.x_key[:nnz_final]).astype(np.uint64, copy=False)
        x_val_out = cp.asnumpy(ctx.x_val[:nnz_final]).astype(np.float64, copy=False)
        x_idx_out = key64_to_csf_idx64_host(drt, x_key_out, strict=True)
        order = np.argsort(x_idx_out)
        x_idx_out = np.asarray(x_idx_out[order], dtype=idx_dtype_out, order="C")
        x_val_out = np.asarray(x_val_out[order], dtype=np.float64, order="C")
    finally:
        ctx.release()

    return _build_run(
        idx=x_idx_out,
        val=x_val_out,
        key_u64=np.asarray(x_key_out[order], dtype=np.uint64, order="C"),
        label_kind="key64",
        shifts=shifts,
        populations=pops,
        trace=trace,
        energy_estimator=energy_estimator,
        reference_policy=reference_policy,
        fixed_ref_idx=fixed_ref_idx,
    )


def _run_fciqmc_cuda_idx64(
    *,
    drt,
    h1e,
    eri,
    x_idx_u: np.ndarray,
    x_val_u: np.ndarray,
    dt: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    max_walker: int | None,
    shift: float,
    tgt_pop: float,
    shift_damping: float,
    shift_stride: int,
    shift_start: int,
    shift_warmup_iters: int,
    shift_log_clip: float | None,
    energy_stride: int,
    preferred_ref_idx: int | None,
    fixed_ref_idx: int,
    initiator_t: float,
    pops: np.ndarray,
    shifts: np.ndarray,
    trace: _FCIQMCTrace,
    trial_idx_u: np.ndarray,
    trial_val_u: np.ndarray,
    trial_l2: float,
    det_idx_u: np.ndarray,
    rayleigh_stride: int,
    state_cache,
    energy_estimator: str,
    reference_policy: str,
    key64_pair_norm: np.ndarray | None,
    key64_pair_sampling_mode: int,
) -> FCIQMCRun:
    from .cuda_backend import (  # noqa: PLC0415
        cuda_fciqmc_step_hamiltonian_u64_ws,
        make_cuda_fciqmc_context_idx64,
    )

    norb = int(drt.norb)
    if norb > 64:
        raise ValueError("backend='cuda_idx64' requires drt.norb <= 64")
    if int(drt.ncsf) > np.iinfo(np.int64).max:
        raise ValueError("backend='cuda_idx64' requires drt.ncsf <= int64 max")

    if max_walker is None:
        max_walker = max(1024, 4 * int(x_idx_u.size))
    max_walker = int(max_walker)

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

    ctx = make_cuda_fciqmc_context_idx64(
        drt,
        h1e,
        eri,
        max_walker=max_walker,
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
        det_idx=det_idx_u if int(det_idx_u.size) > 0 else None,
        pair_alias_prob=pair_alias_prob,
        pair_alias_idx=pair_alias_idx,
        pair_norm=pair_norm,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
        ncsf_u64=int(drt.ncsf),
    )

    nnz0 = int(x_idx_u.size)
    import cupy as cp  # noqa: PLC0415

    idx_dtype_out = np.asarray(x_idx_u).dtype
    idx0_i64 = np.asarray(x_idx_u, dtype=np.int64).ravel()
    if idx0_i64.size:
        if int(np.min(idx0_i64)) < 0:
            raise ValueError("initial state indices must be non-negative for idx64 backend")
        if int(np.max(idx0_i64)) >= int(drt.ncsf):
            raise ValueError("initial state indices must be < drt.ncsf for idx64 backend")
    key0 = np.asarray(idx0_i64, dtype=np.uint64, order="C")
    order0 = np.argsort(key0)
    key0 = np.asarray(key0[order0], dtype=np.uint64, order="C")
    val0 = np.asarray(x_val_u[order0], dtype=np.float64, order="C")
    ctx.x_key[:nnz0] = cp.asarray(key0, dtype=cp.uint64)
    ctx.x_val[:nnz0] = cp.asarray(val0, dtype=cp.float64)
    ctx.nnz = nnz0

    sample_pos = 1
    prev_pop_at_shift_update = float(pops[0])
    try:
        pops_dev = cp.empty(int(niter) + 1, dtype=cp.float64)
        pops_dev[0] = float(pops[0])
        for it in range(1, niter + 1):
            cuda_fciqmc_step_hamiltonian_u64_ws(
                ctx,
                dt=dt,
                shift=shift,
                initiator_t=float(initiator_t),
                seed_spawn=int(seed) + it,
                sync=True,
            )

            nnz_now = int(ctx.nnz)
            pop_dev = cp.sum(cp.abs(ctx.x_val[:nnz_now]))
            pops_dev[it] = pop_dev
            shifts[it] = shift

            if float(shift_damping) > 0.0 and it >= max(int(shift_start), int(shift_warmup_iters)) and (it % int(shift_stride) == 0):
                pop = float(pop_dev.get())
                shift = _maybe_update_shift(
                    shift=shift,
                    pop=pop,
                    it=it,
                    dt=dt,
                    tgt_pop=tgt_pop,
                    shift_damping=shift_damping,
                    shift_stride=shift_stride,
                    shift_start=shift_start,
                    shift_warmup_iters=shift_warmup_iters,
                    shift_log_clip=shift_log_clip,
                    prev_population=prev_pop_at_shift_update,
                )
                prev_pop_at_shift_update = pop

            if it % int(energy_stride) == 0:
                x_key_h = cp.asnumpy(ctx.x_key[:nnz_now]).astype(np.uint64, copy=False)
                x_val_h = cp.asnumpy(ctx.x_val[:nnz_now]).astype(np.float64, copy=False)
                x_idx_h = np.asarray(x_key_h, dtype=np.int64, order="C")
                order = np.argsort(x_idx_h)
                x_idx_h = np.asarray(x_idx_h[order], dtype=idx_dtype_out, order="C")
                x_val_h = np.asarray(x_val_h[order], dtype=np.float64, order="C")
                metrics = _evaluate_checkpoint(
                    drt=drt,
                    h1e=h1e,
                    eri=eri,
                    x_idx_u=x_idx_h,
                    x_val_u=x_val_h,
                    fixed_ref_idx=int(fixed_ref_idx),
                    preferred_ref_idx=preferred_ref_idx,
                    trial_idx_u=trial_idx_u,
                    trial_val_u=trial_val_u,
                    trial_l2=float(trial_l2),
                    det_idx_u=det_idx_u,
                    compute_rayleigh=_need_rayleigh(it, rayleigh_stride=rayleigh_stride),
                    state_cache=state_cache,
                )
                _write_checkpoint(trace, sample_pos, it, metrics)
                sample_pos += 1

        pops[1:] = cp.asnumpy(pops_dev[1:]).astype(np.float64, copy=False)

        nnz_final = int(ctx.nnz)
        x_key_out = cp.asnumpy(ctx.x_key[:nnz_final]).astype(np.uint64, copy=False)
        x_val_out = cp.asnumpy(ctx.x_val[:nnz_final]).astype(np.float64, copy=False)
        x_idx_out = np.asarray(x_key_out, dtype=np.int64, order="C")
        order = np.argsort(x_idx_out)
        x_idx_out = np.asarray(x_idx_out[order], dtype=idx_dtype_out, order="C")
        x_val_out = np.asarray(x_val_out[order], dtype=np.float64, order="C")
    finally:
        ctx.release()

    return _build_run(
        idx=x_idx_out,
        val=x_val_out,
        key_u64=np.asarray(x_key_out[order], dtype=np.uint64, order="C"),
        label_kind="idx64",
        shifts=shifts,
        populations=pops,
        trace=trace,
        energy_estimator=energy_estimator,
        reference_policy=reference_policy,
        fixed_ref_idx=fixed_ref_idx,
    )
