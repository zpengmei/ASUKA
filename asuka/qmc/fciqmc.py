from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import get_state_cache
from .estimators import choose_reference_index, projected_energy_ref, rayleigh_energy_ref
from .omp import maybe_set_openmp_threads
from .spawn import spawn_hamiltonian_events
from .sparse import coalesce_coo_i32_f64


def update_shift(
    shift: float,
    population: float,
    *,
    target_population: float,
    dt: float,
    damping: float,
) -> float:
    """Simple population-control shift update.

    Convention:
      x_{t+1} = x_t - dt (H - S I) x_t

    With this sign convention, decreasing S increases death/cloning and tends to
    reduce the walker population. The update below therefore decreases S when
    population is above target.
    """

    shift = float(shift)
    pop = float(population)
    tgt = float(target_population)
    dt = float(dt)
    damp = float(damping)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if damp <= 0.0:
        return shift
    if pop <= 0.0 or tgt <= 0.0:
        raise ValueError("population and target_population must be > 0")
    return shift - (damp / dt) * float(np.log(pop / tgt))


@dataclass(frozen=True)
class FCIQMCRun:
    idx: np.ndarray
    val: np.ndarray
    energies: np.ndarray
    shifts: np.ndarray
    populations: np.ndarray
    ref_idx: np.ndarray
    energy_estimator: str


def run_fciqmc(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
    x_val: np.ndarray,
    *,
    dt: float,
    niter: int,
    nspawn_one: int,
    nspawn_two: int,
    seed: int,
    backend: str = "cpu",
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
    initiator_t: float = 0.0,
    use_state_cache: bool = True,
    key64_pair_norm: np.ndarray | None = None,
    key64_pair_sampling_mode: int = 0,
) -> FCIQMCRun:
    """Single-root CSF-native FCIQMC loop.

    This uses a continuous-walker update with annihilation via coalescing:

      x <- x - dt * (H - S I) x

    Notes
    -----
    - This driver uses a deterministic row oracle for the projected-energy
      estimator.
    - Integer walkers / stochastic rounding is a later milestone.
    """

    dt = float(dt)
    niter = int(niter)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if niter < 0:
        raise ValueError("niter must be >= 0")
    shift_stride = int(shift_stride)
    shift_start = int(shift_start)
    energy_stride = int(energy_stride)
    if shift_stride < 1:
        raise ValueError("shift_stride must be >= 1")
    if shift_start < 0:
        raise ValueError("shift_start must be >= 0")
    if energy_stride < 1:
        raise ValueError("energy_stride must be >= 1")

    energy_estimator = str(energy_estimator).lower()
    if energy_estimator not in ("projected", "rayleigh"):
        raise ValueError("energy_estimator must be 'projected' or 'rayleigh'")

    backend_s = str(backend).lower()
    if backend_s not in ("cpu", "cuda", "cuda_key64"):
        raise ValueError("backend must be 'cpu', 'cuda', or 'cuda_key64'")

    x_idx_u, x_val_u = coalesce_coo_i32_f64(x_idx, x_val)
    if x_idx_u.size == 0:
        raise ValueError("initial x is empty")

    shift = float(shift_init)
    pops = np.empty(niter + 1, dtype=np.float64)
    shifts = np.empty(niter + 1, dtype=np.float64)

    n_energy = (niter // energy_stride) + 1
    energies = np.empty(n_energy, dtype=np.float64)
    ref_hist = np.empty(n_energy, dtype=np.int32)
    e_pos = 0

    pop0 = float(np.sum(np.abs(x_val_u)))
    if pop0 == 0.0:
        raise ValueError("initial population is zero")
    tgt_pop = float(pop0 if target_population is None else float(target_population))
    if tgt_pop <= 0.0:
        raise ValueError("target_population must be > 0")

    state_cache = get_state_cache(drt) if (bool(use_state_cache) and backend_s == "cpu") else None

    ref = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
    if energy_estimator == "rayleigh":
        e0, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, state_cache=state_cache)
    else:
        e0, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, ref_idx=ref, state_cache=state_cache)
    energies[e_pos] = float(e0)
    ref_hist[e_pos] = np.int32(ref)
    e_pos += 1

    pops[0] = pop0
    shifts[0] = shift

    if backend_s == "cuda":
        return _run_fciqmc_cuda(
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
            shift_damping=shift_damping,
            shift_stride=shift_stride,
            shift_start=shift_start,
            energy_stride=energy_stride,
            energy_estimator=energy_estimator,
            preferred_ref_idx=preferred_ref_idx,
            initiator_t=initiator_t,
            pops=pops,
            shifts=shifts,
            energies=energies,
            ref_hist=ref_hist,
            e_pos=e_pos,
        )
    if backend_s == "cuda_key64":
        return _run_fciqmc_cuda_key64(
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
            energy_stride=int(energy_stride),
            energy_estimator=str(energy_estimator),
            preferred_ref_idx=preferred_ref_idx,
            initiator_t=float(initiator_t),
            pops=pops,
            shifts=shifts,
            energies=energies,
            ref_hist=ref_hist,
            e_pos=e_pos,
            key64_pair_norm=key64_pair_norm,
            key64_pair_sampling_mode=int(key64_pair_sampling_mode),
        )

    # CPU path
    maybe_set_openmp_threads(omp_threads)
    rng = np.random.default_rng(int(seed))

    for it in range(1, niter + 1):
        evt_i, evt_v = spawn_hamiltonian_events(
            drt,
            h1e,
            eri,
            x_idx_u,
            x_val_u,
            eps=dt,
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            rng=rng,
            initiator_t=float(initiator_t),
            state_cache=state_cache,
        )

        scale = 1.0 + dt * float(shift)
        if evt_i.size:
            idx_all = np.concatenate((x_idx_u, evt_i))
            val_all = np.concatenate((scale * x_val_u, evt_v))
        else:
            idx_all = x_idx_u
            val_all = scale * x_val_u

        x_idx_u, x_val_u = coalesce_coo_i32_f64(idx_all, val_all)

        pop = float(np.sum(np.abs(x_val_u)))
        pops[it] = pop
        shifts[it] = shift

        if it >= shift_start and (it % shift_stride == 0):
            shift = update_shift(
                shift,
                pop,
                target_population=tgt_pop,
                dt=dt,
                damping=float(shift_damping),
            )

        if it % energy_stride == 0:
            ref = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
            if energy_estimator == "rayleigh":
                e, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, state_cache=state_cache)
            else:
                e, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, ref_idx=ref, state_cache=state_cache)
            energies[e_pos] = float(e)
            ref_hist[e_pos] = np.int32(ref)
            e_pos += 1

    return FCIQMCRun(
        idx=x_idx_u,
        val=x_val_u,
        energies=energies,
        shifts=shifts,
        populations=pops,
        ref_idx=ref_hist,
        energy_estimator=energy_estimator,
    )


def _run_fciqmc_cuda(
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
    energy_stride: int,
    energy_estimator: str,
    preferred_ref_idx: int | None,
    initiator_t: float,
    pops: np.ndarray,
    shifts: np.ndarray,
    energies: np.ndarray,
    ref_hist: np.ndarray,
    e_pos: int,
) -> FCIQMCRun:
    from .cuda_backend import make_cuda_fciqmc_context, cuda_fciqmc_step_hamiltonian_ws  # noqa: PLC0415

    if max_walker is None:
        # Default: 4x initial nnz, clamped to at least 1024
        max_walker = max(1024, 4 * int(x_idx_u.size))
    max_walker = int(max_walker)

    ctx = make_cuda_fciqmc_context(
        drt,
        h1e,
        eri,
        max_walker=max_walker,
        nspawn_one=int(nspawn_one),
        nspawn_two=int(nspawn_two),
    )

    # Upload initial vector.
    nnz0 = int(x_idx_u.size)
    import cupy as cp  # noqa: PLC0415
    ctx.x_idx[:nnz0] = cp.asarray(x_idx_u, dtype=cp.int32)
    ctx.x_val[:nnz0] = cp.asarray(x_val_u, dtype=cp.float64)
    ctx.nnz = nnz0

    try:
        pops_dev = cp.empty(int(niter) + 1, dtype=cp.float64)
        for it in range(1, niter + 1):
            cuda_fciqmc_step_hamiltonian_ws(
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

            if it >= shift_start and (it % shift_stride == 0):
                pop = float(pop_dev.get())
                shift = update_shift(
                    shift,
                    pop,
                    target_population=tgt_pop,
                    dt=dt,
                    damping=float(shift_damping),
                )

            if it % energy_stride == 0:
                # Full vector download only on energy evaluation steps.
                x_idx_h = cp.asnumpy(ctx.x_idx[:nnz_now]).astype(np.int32, copy=False)
                x_val_h = cp.asnumpy(ctx.x_val[:nnz_now]).astype(np.float64, copy=False)
                ref = choose_reference_index(x_idx_h, x_val_h, preferred=preferred_ref_idx)
                if energy_estimator == "rayleigh":
                    e, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_h, x_val_h)
                else:
                    e, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_h, x_val_h, ref_idx=ref)
                energies[e_pos] = float(e)
                ref_hist[e_pos] = np.int32(ref)
                e_pos += 1

        # Final download.
        if niter >= 1:
            pops[1:] = cp.asnumpy(pops_dev[1:]).astype(np.float64, copy=False)
        nnz_final = int(ctx.nnz)
        x_idx_out = cp.asnumpy(ctx.x_idx[:nnz_final]).astype(np.int32, copy=False)
        x_val_out = cp.asnumpy(ctx.x_val[:nnz_final]).astype(np.float64, copy=False)
    finally:
        ctx.release()

    return FCIQMCRun(
        idx=x_idx_out,
        val=x_val_out,
        energies=energies,
        shifts=shifts,
        populations=pops,
        ref_idx=ref_hist,
        energy_estimator=energy_estimator,
    )


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
    energy_stride: int,
    energy_estimator: str,
    preferred_ref_idx: int | None,
    initiator_t: float,
    pops: np.ndarray,
    shifts: np.ndarray,
    energies: np.ndarray,
    ref_hist: np.ndarray,
    e_pos: int,
    key64_pair_norm: np.ndarray | None,
    key64_pair_sampling_mode: int,
) -> FCIQMCRun:
    from .cuda_backend import (  # noqa: PLC0415
        csf_idx_to_key64_host,
        cuda_fciqmc_step_hamiltonian_u64_ws,
        key64_to_csf_idx_host,
        make_cuda_fciqmc_context_key64,
    )

    norb = int(drt.norb)
    if norb > 32:
        raise ValueError("backend='cuda_key64' requires drt.norb <= 32")

    if max_walker is None:
        # Default: 4x initial nnz, clamped to at least 1024
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
        pair_alias_prob=pair_alias_prob,
        pair_alias_idx=pair_alias_idx,
        pair_norm=pair_norm,
        pair_norm_sum=float(pair_norm_sum),
        pair_sampling_mode=int(pair_sampling_mode),
    )

    # Upload initial vector (convert CSF indices -> Key64).
    nnz0 = int(x_idx_u.size)
    import cupy as cp  # noqa: PLC0415

    key0 = csf_idx_to_key64_host(drt, x_idx_u, state_cache=None)
    order0 = np.argsort(key0)
    key0 = np.asarray(key0[order0], dtype=np.uint64, order="C")
    val0 = np.asarray(x_val_u[order0], dtype=np.float64, order="C")
    ctx.x_key[:nnz0] = cp.asarray(key0, dtype=cp.uint64)
    ctx.x_val[:nnz0] = cp.asarray(val0, dtype=cp.float64)
    ctx.nnz = nnz0

    try:
        pops_dev = cp.empty(int(niter) + 1, dtype=cp.float64)
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

            if it >= shift_start and (it % shift_stride == 0):
                pop = float(pop_dev.get())
                shift = update_shift(
                    shift,
                    pop,
                    target_population=tgt_pop,
                    dt=dt,
                    damping=float(shift_damping),
                )

            if it % energy_stride == 0:
                # Full vector download only on energy evaluation steps.
                x_key_h = cp.asnumpy(ctx.x_key[:nnz_now]).astype(np.uint64, copy=False)
                x_val_h = cp.asnumpy(ctx.x_val[:nnz_now]).astype(np.float64, copy=False)
                x_idx_h = key64_to_csf_idx_host(drt, x_key_h, strict=True)
                order = np.argsort(x_idx_h)
                x_idx_h = np.asarray(x_idx_h[order], dtype=np.int32, order="C")
                x_val_h = np.asarray(x_val_h[order], dtype=np.float64, order="C")

                ref = choose_reference_index(x_idx_h, x_val_h, preferred=preferred_ref_idx)
                if energy_estimator == "rayleigh":
                    e, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_h, x_val_h)
                else:
                    e, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_h, x_val_h, ref_idx=ref)
                energies[e_pos] = float(e)
                ref_hist[e_pos] = np.int32(ref)
                e_pos += 1

        # Final download.
        if niter >= 1:
            pops[1:] = cp.asnumpy(pops_dev[1:]).astype(np.float64, copy=False)

        nnz_final = int(ctx.nnz)
        x_key_out = cp.asnumpy(ctx.x_key[:nnz_final]).astype(np.uint64, copy=False)
        x_val_out = cp.asnumpy(ctx.x_val[:nnz_final]).astype(np.float64, copy=False)
        x_idx_out = key64_to_csf_idx_host(drt, x_key_out, strict=True)
        order = np.argsort(x_idx_out)
        x_idx_out = np.asarray(x_idx_out[order], dtype=np.int32, order="C")
        x_val_out = np.asarray(x_val_out[order], dtype=np.float64, order="C")
    finally:
        ctx.release()

    return FCIQMCRun(
        idx=x_idx_out,
        val=x_val_out,
        energies=energies,
        shifts=shifts,
        populations=pops,
        ref_idx=ref_hist,
        energy_estimator=energy_estimator,
    )
