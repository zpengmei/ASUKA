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
) -> FCIQMCRun:
    """Single-root CSF-native FCIQMC-like loop (CPU implementation).

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

    rng = np.random.default_rng(int(seed))
    maybe_set_openmp_threads(omp_threads)
    state_cache = get_state_cache(drt) if bool(use_state_cache) else None

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
