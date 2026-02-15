from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import warnings

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.cuguga.state_cache import get_state_cache
from asuka.cuguga.oracle.sparse import connected_row_sparse
from .estimators import choose_reference_index, projected_energy_ref, rayleigh_energy_ref
from .omp import maybe_set_openmp_threads
from .subspace import apply_right_matrix, dot_sparse, orthonormalize_mgs
from .projector import projector_step
from .sparse import coalesce_coo_i32_f64


@dataclass(frozen=True)
class FCIFRIRun:
    idx: np.ndarray
    val: np.ndarray
    energies: np.ndarray
    ref_idx: np.ndarray
    energy_estimator: str


@dataclass(frozen=True)
class FCIFRISubspaceRun:
    idx: list[np.ndarray]
    val: list[np.ndarray]
    energies: np.ndarray
    iters: np.ndarray
    backend: str


def run_fcifri_ground(
    drt: DRT,
    h1e: np.ndarray,
    eri,
    x_idx: np.ndarray,
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
    use_state_cache: bool = True,
    backend: str = "stochastic",
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    spawner: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
    spawner_kwargs: Mapping[str, object] | None = None,
) -> FCIFRIRun:
    """Single-root FCI-FRI-like projector iteration (CPU implementation).

    This is the Neigen=1 specialization of the FCI-FRI iteration:
      x <- Φ((I - eps*H) x) , followed by a scalar normalization.

    Notes
    -----
    - This routine uses a deterministic row oracle to compute projected
      energies every `energy_stride` iterations.
    - Multi-root (excited-state) iteration + QR-style stabilization is a later
      milestone.
    """

    m = int(m)
    niter = int(niter)
    if m < 1:
        raise ValueError("m must be >= 1")
    if niter < 0:
        raise ValueError("niter must be >= 0")
    energy_stride = int(energy_stride)
    if energy_stride < 1:
        raise ValueError("energy_stride must be >= 1")

    energy_estimator = str(energy_estimator).lower()
    if energy_estimator not in ("projected", "rayleigh"):
        raise ValueError("energy_estimator must be 'projected' or 'rayleigh'")

    backend = str(backend).lower()
    if backend not in ("stochastic", "cuda"):
        raise ValueError("backend must be 'stochastic' or 'cuda'")

    use_cuda = backend == "cuda"
    if use_cuda and (spawner is not None or spawner_kwargs is not None):
        warnings.warn(
            "backend='cuda' does not support custom spawner/spawner_kwargs yet; falling back to CPU backend.",
            stacklevel=2,
        )
        use_cuda = False

    rng = np.random.default_rng(int(seed))
    maybe_set_openmp_threads(omp_threads)
    state_cache = get_state_cache(drt) if bool(use_state_cache) else None

    x_idx_u, x_val_u = coalesce_coo_i32_f64(x_idx, x_val)
    if x_idx_u.size == 0:
        raise ValueError("initial x is empty")

    # L2 normalization (Neigen=1: G(tau) is just a scalar).
    n2 = float(np.linalg.norm(x_val_u))
    if n2 == 0.0:
        raise ValueError("initial x has zero norm")
    x_val_u = np.asarray(x_val_u / n2, dtype=np.float64, order="C")

    n_energy = (niter // energy_stride) + 1
    energies = np.empty(n_energy, dtype=np.float64)
    ref_hist = np.empty(n_energy, dtype=np.int32)

    e_pos = 0
    ref = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
    if energy_estimator == "rayleigh":
        e0, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, state_cache=state_cache)
    else:
        e0, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, ref_idx=ref, state_cache=state_cache)
    energies[e_pos] = float(e0)
    ref_hist[e_pos] = np.int32(ref)
    e_pos += 1

    if use_cuda:
        try:  # optional
            import cupy as cp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"backend='cuda' requires cupy: {e}") from e

        try:  # optional
            from .cuda_backend import cuda_projector_step_hamiltonian_ws, make_cuda_projector_context  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"failed to import CUDA QMC backend: {e}") from e

        if state_cache is None:
            state_cache = get_state_cache(drt)

        ctx = make_cuda_projector_context(
            drt,
            h1e,
            eri,
            m=int(m),
            pivot=int(pivot),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
        )
        try:
            # Upload initial x into ctx buffers.
            nnz0 = int(x_idx_u.size)
            if nnz0 > int(m):
                raise ValueError("initial x nnz exceeds m")
            ctx.x_idx[:nnz0] = cp.asarray(x_idx_u, dtype=cp.int32)
            ctx.x_val[:nnz0] = cp.asarray(x_val_u, dtype=cp.float64)
            ctx.nnz = nnz0

            for it in range(1, niter + 1):
                # Initiator threshold uses the current column l1 norm.
                if float(initiator_na) != 0.0:
                    l1 = float(cp.sum(cp.abs(ctx.x_val[: ctx.nnz])).get())
                    initiator_t = float(initiator_na) * l1 / float(m - 1)
                else:
                    initiator_t = 0.0

                seed_spawn = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
                seed_phi = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
                cuda_projector_step_hamiltonian_ws(
                    ctx,
                    eps=float(eps),
                    initiator_t=float(initiator_t),
                    seed_spawn=seed_spawn,
                    seed_phi=seed_phi,
                    scale_identity=1.0,
                    sync=True,
                    compressor=compressor,
                )

                n2 = float(cp.linalg.norm(ctx.x_val[: ctx.nnz]).get())
                if n2 == 0.0:
                    raise RuntimeError("vector collapsed to zero (try smaller eps or more spawn)")
                ctx.x_val[: ctx.nnz] /= n2

                if it % energy_stride == 0:
                    x_idx_u = cp.asnumpy(ctx.x_idx[: ctx.nnz]).astype(np.int32, copy=False)
                    x_val_u = cp.asnumpy(ctx.x_val[: ctx.nnz]).astype(np.float64, copy=False)
                    ref = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
                    if energy_estimator == "rayleigh":
                        e, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, state_cache=state_cache)
                    else:
                        e, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, ref_idx=ref, state_cache=state_cache)
                    energies[e_pos] = float(e)
                    ref_hist[e_pos] = np.int32(ref)
                    e_pos += 1

            # Final host output.
            x_idx_u = cp.asnumpy(ctx.x_idx[: ctx.nnz]).astype(np.int32, copy=False)
            x_val_u = cp.asnumpy(ctx.x_val[: ctx.nnz]).astype(np.float64, copy=False)
        finally:
            ctx.release()
    else:
        for it in range(1, niter + 1):
            x_idx_u, x_val_u = projector_step(
                drt,
                h1e,
                eri,
                x_idx_u,
                x_val_u,
                eps=float(eps),
                nspawn_one=int(nspawn_one),
                nspawn_two=int(nspawn_two),
                rng=rng,
                m=int(m),
                pivot=int(pivot),
                initiator_na=float(initiator_na),
                state_cache=state_cache,
                spawner=spawner,
                spawner_kwargs=spawner_kwargs,
                compressor=compressor,
            )
            n2 = float(np.linalg.norm(x_val_u))
            if n2 == 0.0:
                raise RuntimeError("vector collapsed to zero (try smaller eps or more spawn)")
            x_val_u = np.asarray(x_val_u / n2, dtype=np.float64, order="C")

            if it % energy_stride == 0:
                ref = choose_reference_index(x_idx_u, x_val_u, preferred=preferred_ref_idx)
                if energy_estimator == "rayleigh":
                    e, _, _ = rayleigh_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, state_cache=state_cache)
                else:
                    e, _, _ = projected_energy_ref(drt, h1e, eri, x_idx_u, x_val_u, ref_idx=ref, state_cache=state_cache)
                energies[e_pos] = float(e)
                ref_hist[e_pos] = np.int32(ref)
                e_pos += 1

    return FCIFRIRun(idx=x_idx_u, val=x_val_u, energies=energies, ref_idx=ref_hist, energy_estimator=energy_estimator)


def run_fcifri_subspace(
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
    method: str = "legacy",
    rsi_alpha: float = 0.5,
    rsi_burn_in: int = 50,
    rsi_min_nsample: int = 10,
    rsi_compression: str = "pivotal",
    backend: str = "stochastic",
    use_state_cache: bool = True,
    x0: np.ndarray | list[tuple[np.ndarray, np.ndarray]] | None = None,
    compressor: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
    spawner: Callable[..., tuple[np.ndarray, np.ndarray]] | Sequence[Callable[..., tuple[np.ndarray, np.ndarray]]] | None = None,
    spawner_kwargs: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
) -> FCIFRISubspaceRun:
    """Multi-root FCI-FRI-style subspace iteration (CPU reference).

    Algorithm (debug/staged version):
    - Apply the (stochastic or exact) projector step column-wise:
        Y_k <- Φ((I - eps*H) X_k)
    - Orthonormalize columns by modified Gram-Schmidt (MGS).
    - Periodically compute Ritz energies from the orthonormal subspace:
        T = X^T H X   (deterministic row-oracle evaluation, for validation)
      then rotate X <- X C where C diagonalizes T.

    Notes
    -----
    - `backend='exact'` uses deterministic row oracles in the projector step and
      is intended for robust validation on tiny CI spaces.
    - `backend='cuda'` runs the projector step on GPU (dense ERI only) but keeps
      orthonormalization and Ritz extraction on CPU for now.
    - Production FCI-FRI will replace the deterministic Ritz evaluation with
      stochastic estimators (U^T H X / U^T X) and use the paper's stabilization
      matrix schedule.
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
    ritz_stride = int(ritz_stride)
    if ortho_stride < 1:
        raise ValueError("ortho_stride must be >= 1")
    if ritz_stride < 1:
        raise ValueError("ritz_stride must be >= 1")

    method = str(method).lower()
    if method not in ("legacy", "rsi"):
        raise ValueError("method must be 'legacy' or 'rsi'")

    backend = str(backend).lower()
    if backend not in ("stochastic", "exact", "cuda"):
        raise ValueError("backend must be 'stochastic', 'exact', or 'cuda'")

    if backend == "cuda" and (compressor is not None or spawner is not None or spawner_kwargs is not None):
        warnings.warn(
            "backend='cuda' does not support custom compressor/spawner yet; falling back to CPU backend.",
            stacklevel=2,
        )
        backend = "stochastic"

    rng = np.random.default_rng(int(seed))
    maybe_set_openmp_threads(omp_threads)
    state_cache = get_state_cache(drt) if bool(use_state_cache) else None

    def _init_sparse_trial_vectors_for_rsi(
        x0_in: np.ndarray | list[tuple[np.ndarray, np.ndarray]] | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        ncsf_i = int(drt.ncsf)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        if x0_in is None:
            # Default: sparse random trial vectors (avoid dense (nroots,ncsf) allocation).
            nnz = min(int(m), ncsf_i)
            if nnz <= 0:
                raise ValueError("m must be >= 1")
            for _k in range(int(nroots)):
                idx = np.asarray(rng.choice(ncsf_i, size=nnz, replace=False), dtype=np.int32)
                idx.sort()
                val = rng.normal(size=nnz).astype(np.float64, copy=False)
                out.append(coalesce_coo_i32_f64(idx, val))
            return out

        if isinstance(x0_in, list):
            if len(x0_in) != int(nroots):
                raise ValueError(f"x0 list must have length nroots={int(nroots)}")
            for idx_k, val_k in x0_in:
                out.append(coalesce_coo_i32_f64(idx_k, val_k))
            return out

        x0_arr = np.asarray(x0_in, dtype=np.float64)
        if x0_arr.shape != (int(nroots), ncsf_i):
            raise ValueError(f"x0 has wrong shape: {x0_arr.shape} (expected {(int(nroots), ncsf_i)})")

        # Convert each dense row to a sparse vector by keeping up to m largest-|x| entries.
        nnz = min(int(m), ncsf_i)
        for k in range(int(nroots)):
            row = np.asarray(x0_arr[k], dtype=np.float64)
            if nnz >= ncsf_i:
                idx = np.arange(ncsf_i, dtype=np.int32)
                val = row
            else:
                pos = np.argpartition(np.abs(row), -nnz)[-nnz:]
                pos = np.asarray(pos, dtype=np.int32)
                pos.sort()
                idx = pos
                val = np.asarray(row[pos], dtype=np.float64)
            out.append(coalesce_coo_i32_f64(idx, val))
        return out

    if method == "rsi":
        if x0 is None:
            warnings.warn(
                "method='rsi' called with x0=None; initializing *random* trial vectors U0. "
                "This is useful for smoke tests but typically yields poor agreement with FCI. "
                "For meaningful results, pass high-quality trial vectors (e.g. from "
                "asuka.sci.selected_ci.selected_ci(...).ci_full) via x0=[(idx,val),...].",
                stacklevel=2,
            )
        # Paper-faithful RSI excited-state path.
        # Reuse ortho_stride as delta_ortho and ritz_stride as eval_stride.
        from .rsi import run_fcifri_rsi  # noqa: PLC0415

        U0 = _init_sparse_trial_vectors_for_rsi(x0)

        compression = str(rsi_compression)
        if backend == "cuda" and compression.lower() != "pivot_resample":
            compression = "pivot_resample"

        res = run_fcifri_rsi(
            drt,
            h1e,
            eri,
            U0=U0,
            m=int(m),
            eps=float(eps),
            niter=int(niter),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
            seed=int(seed),
            omp_threads=omp_threads,
            pivot=int(pivot),
            initiator_na=float(initiator_na),
            alpha=float(rsi_alpha),
            delta_ortho=int(ortho_stride),
            eval_stride=int(ritz_stride),
            burn_in=int(rsi_burn_in),
            compression=compression,
            projector_compressor=compressor,
            projector_spawner=spawner,
            projector_spawner_kwargs=spawner_kwargs,
            backend=backend,
            use_state_cache=bool(use_state_cache),
        )

        rsi_min_nsample = int(rsi_min_nsample)
        if rsi_min_nsample < 0:
            raise ValueError("rsi_min_nsample must be >= 0")
        if int(res.nsample) >= int(rsi_min_nsample) and int(res.nsample) > 0:
            energies = res.energies_avg
        else:
            if int(res.nsample) > 0 and int(rsi_min_nsample) > 0:
                warnings.warn(
                    f"RSI averaged estimator has nsample={int(res.nsample)} (<{int(rsi_min_nsample)}); "
                    "using instantaneous generalized-eigen energies instead. "
                    "Consider increasing niter and/or decreasing ritz_stride and/or reducing burn-in.",
                    stacklevel=2,
                )
            energies = res.energies_inst
        return FCIFRISubspaceRun(
            idx=res.idx,
            val=res.val,
            energies=np.asarray(energies, dtype=np.float64),
            iters=np.asarray(res.iters, dtype=np.int32),
            backend=backend,
        )

    ncsf = int(drt.ncsf)
    cols: list[tuple[np.ndarray, np.ndarray]] = []
    if x0 is None:
        x0 = rng.normal(size=(nroots, ncsf)).astype(np.float64, copy=False)

    if isinstance(x0, list):
        if len(x0) != nroots:
            raise ValueError(f"x0 list must have length nroots={nroots}")
        for idx_k, val_k in x0:
            cols.append(coalesce_coo_i32_f64(idx_k, val_k))
    else:
        x0 = np.asarray(x0, dtype=np.float64)
        if x0.shape != (nroots, ncsf):
            raise ValueError(f"x0 has wrong shape: {x0.shape} (expected {(nroots, ncsf)})")
        full_idx = np.arange(ncsf, dtype=np.int32)
        for k in range(nroots):
            cols.append(coalesce_coo_i32_f64(full_idx, np.asarray(x0[k], dtype=np.float64)))

    # Orthonormalize the whole block.
    cols = orthonormalize_mgs(cols, m=m, pivot=pivot, rng=rng, compressor=compressor)

    # Ritz bookkeeping.
    n_check = (niter // ritz_stride) + 1
    energies = np.empty((n_check, nroots), dtype=np.float64)
    iters = np.empty(n_check, dtype=np.int32)
    chk = 0

    def ritz_and_rotate() -> np.ndarray:
        nonlocal cols
        # Build Hx_k as dense vectors via row oracles (validation path).
        hx = np.zeros((nroots, ncsf), dtype=np.float64)
        for k in range(nroots):
            idx_k, val_k = cols[k]
            for j, xj in zip(idx_k.tolist(), val_k.tolist()):
                if xj == 0.0:
                    continue
                i_idx, hij = connected_row_sparse(
                    drt,
                    h1e,
                    eri,
                    int(j),
                    max_out=200_000,
                    state_cache=state_cache,
                )
                if i_idx.size:
                    hx[k, i_idx] += float(xj) * np.asarray(hij, dtype=np.float64)

        tmat = np.zeros((nroots, nroots), dtype=np.float64)
        for i in range(nroots):
            idx_i, val_i = cols[i]
            for k in range(nroots):
                tmat[i, k] = float(np.dot(hx[k, idx_i], val_i))

        tmat = 0.5 * (tmat + tmat.T)
        w, c = np.linalg.eigh(tmat)

        # Rotate the subspace: X <- X C (columns remain orthonormal if not compressed).
        cols = apply_right_matrix(cols, c, m=m, pivot=pivot, rng=rng, compressor=compressor)
        cols = orthonormalize_mgs(cols, m=m, pivot=pivot, rng=rng, compressor=compressor)
        return np.asarray(w, dtype=np.float64)

    if backend == "cuda":
        try:  # optional
            import cupy as cp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"backend='cuda' requires cupy: {e}") from e

        try:  # optional
            from .cuda_backend import (  # noqa: PLC0415
                cuda_block_apply_right_matrix_phi_ws,
                cuda_block_build_tmat_hamiltonian_stochastic_ws,
                cuda_block_projector_step_hamiltonian_ws,
                cuda_block_orthonormalize_mgs_ws,
                make_cuda_block_projector_context,
            )
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"failed to import CUDA QMC backend: {e}") from e

        m_i = int(m)
        if float(initiator_na) != 0.0 and m_i <= 1:
            raise ValueError("m must be > 1 when initiator_na > 0")

        ctx = make_cuda_block_projector_context(
            drt,
            h1e,
            eri,
            nroots=int(nroots),
            m=m_i,
            pivot=int(pivot),
            nspawn_one=int(nspawn_one),
            nspawn_two=int(nspawn_two),
        )
        try:
            # Upload initial columns (after initial CPU MGS).
            idx_pad = np.zeros((nroots, m_i), dtype=np.int32)
            val_pad = np.zeros((nroots, m_i), dtype=np.float64)
            nnz_pad = np.zeros(nroots, dtype=np.int32)
            for k in range(nroots):
                idx_k, val_k = cols[k]
                nnz_k = int(idx_k.size)
                if nnz_k <= 0:
                    raise ValueError(f"column {k} is empty")
                if nnz_k > m_i:
                    raise ValueError(f"column {k} nnz exceeds m")
                idx_pad[k, :nnz_k] = np.asarray(idx_k, dtype=np.int32, order="C")
                val_pad[k, :nnz_k] = np.asarray(val_k, dtype=np.float64, order="C")
                nnz_pad[k] = np.int32(nnz_k)
            ctx.set_cols_packed(idx_pad, val_pad, nnz_pad)

            # Initial Ritz on GPU (stochastic T build), then rotate on GPU.
            iters[chk] = np.int32(0)
            t_seed = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
            tmat = cuda_block_build_tmat_hamiltonian_stochastic_ws(ctx, seeds_spawn=t_seed, eps=1.0, initiator_t=0.0, sync=True)
            w, c = np.linalg.eigh(tmat)
            energies[chk] = np.asarray(w, dtype=np.float64)
            chk += 1

            rot_seeds = rng.integers(0, np.iinfo(np.int64).max, size=nroots * (nroots - 1), dtype=np.int64)
            cuda_block_apply_right_matrix_phi_ws(ctx, mat=c, seeds_phi=rot_seeds, sync=True)
            mgs_seeds = rng.integers(0, np.iinfo(np.int64).max, size=nroots * (nroots - 1) // 2, dtype=np.int64)
            cuda_block_orthonormalize_mgs_ws(ctx, seeds_phi=mgs_seeds, sync=True)

            for it in range(1, niter + 1):
                initiator_t = np.zeros(nroots, dtype=np.float64)
                for k in range(nroots):
                    nnz_k = int(ctx.nnz[k])
                    if float(initiator_na) != 0.0:
                        l1 = float(cp.sum(cp.abs(ctx.x_val[k, :nnz_k])).get())
                        initiator_t[k] = float(initiator_na) * l1 / float(m_i - 1)

                seed_spawn = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
                seed_phi = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
                nnz_out = cuda_block_projector_step_hamiltonian_ws(
                    ctx,
                    eps=float(eps),
                    initiator_t=initiator_t,
                    seed_spawn=seed_spawn,
                    seed_phi=seed_phi,
                    scale_identity=1.0,
                    sync=True,
                )
                if int(np.min(nnz_out)) <= 0:
                    k0 = int(np.argmin(nnz_out))
                    raise RuntimeError(f"column {k0} collapsed to empty (try smaller eps or more spawn)")

                if it % ortho_stride == 0:
                    # GPU MGS stabilization (no host transfer).
                    nseed = nroots * (nroots - 1) // 2
                    seeds_phi_mgs = rng.integers(0, np.iinfo(np.int64).max, size=nseed, dtype=np.int64)
                    cuda_block_orthonormalize_mgs_ws(ctx, seeds_phi=seeds_phi_mgs, sync=True)

                if it % ritz_stride == 0:
                    iters[chk] = np.int32(it)
                    t_seed = rng.integers(0, np.iinfo(np.int64).max, size=nroots, dtype=np.int64)
                    tmat = cuda_block_build_tmat_hamiltonian_stochastic_ws(ctx, seeds_spawn=t_seed, eps=1.0, initiator_t=0.0, sync=True)
                    w, c = np.linalg.eigh(tmat)
                    energies[chk] = np.asarray(w, dtype=np.float64)
                    chk += 1

                    rot_seeds = rng.integers(0, np.iinfo(np.int64).max, size=nroots * (nroots - 1), dtype=np.int64)
                    cuda_block_apply_right_matrix_phi_ws(ctx, mat=c, seeds_phi=rot_seeds, sync=True)
                    mgs_seeds = rng.integers(0, np.iinfo(np.int64).max, size=nroots * (nroots - 1) // 2, dtype=np.int64)
                    cuda_block_orthonormalize_mgs_ws(ctx, seeds_phi=mgs_seeds, sync=True)

            # Final host output.
            idx_pad, val_pad, nnz_pad = ctx.get_cols_packed()
            cols = []
            for k in range(nroots):
                nnz_k = int(nnz_pad[k])
                cols.append(
                    (
                        np.ascontiguousarray(idx_pad[k, :nnz_k], dtype=np.int32),
                        np.ascontiguousarray(val_pad[k, :nnz_k], dtype=np.float64),
                    )
                )
        finally:
            ctx.release()
    else:
        # Initial Ritz (CPU validation path).
        iters[chk] = np.int32(0)
        energies[chk] = ritz_and_rotate()
        chk += 1

        for it in range(1, niter + 1):
            new_cols: list[tuple[np.ndarray, np.ndarray]] = []
            for k in range(nroots):
                idx_k, val_k = cols[k]

                comp = compressor
                if isinstance(compressor, (list, tuple)):
                    if k >= len(compressor):
                        raise ValueError("compressor sequence is shorter than number of roots")
                    comp = compressor[k]

                sp = spawner
                if isinstance(spawner, (list, tuple)):
                    if k >= len(spawner):
                        raise ValueError("spawner sequence is shorter than number of roots")
                    sp = spawner[k]

                sp_kw = spawner_kwargs
                if isinstance(spawner_kwargs, (list, tuple)):
                    if k >= len(spawner_kwargs):
                        raise ValueError("spawner_kwargs sequence is shorter than number of roots")
                    sp_kw = spawner_kwargs[k]

                idx_y, val_y = projector_step(
                    drt,
                    h1e,
                    eri,
                    idx_k,
                    val_k,
                    eps=float(eps),
                    nspawn_one=int(nspawn_one),
                    nspawn_two=int(nspawn_two),
                    rng=rng,
                    m=int(m),
                    pivot=int(pivot),
                    initiator_na=float(initiator_na),
                    state_cache=state_cache,
                    backend=backend,
                    spawner=sp,
                    spawner_kwargs=sp_kw,
                    compressor=comp,
                )
                new_cols.append((idx_y, val_y))
            cols = new_cols

            if it % ortho_stride == 0:
                cols = orthonormalize_mgs(cols, m=m, pivot=pivot, rng=rng, compressor=compressor)

            if it % ritz_stride == 0:
                iters[chk] = np.int32(it)
                energies[chk] = ritz_and_rotate()
                chk += 1

    return FCIFRISubspaceRun(
        idx=[c[0] for c in cols],
        val=[c[1] for c in cols],
        energies=energies[:chk],
        iters=iters[:chk],
        backend=backend,
    )
