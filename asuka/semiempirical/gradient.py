"""AM1 Cartesian gradients.

This module exposes two gradient backends:
- ``cpu_frozen``: frozen-density finite-difference reference path.
- ``cuda_analytic``: CUDA pairwise analytical kernel path.

``auto`` selects ``cuda_analytic`` for ``device='cuda'`` and falls back to
``cpu_frozen`` when CUDA compilation/runtime is unavailable.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np

from asuka.nddo_core import (
    build_ao_offsets,
    build_pair_list,
    compute_all_multipole_params,
    nao_for_Z,
    valence_electrons,
)

from .core_repulsion import _pair_core_repulsion
from .nddo_integrals import build_pair_two_center_tensor
from .overlap import build_pair_overlap_block
from .params import ElementParams, MethodParams
from .scf import SCFResult


@dataclass(frozen=True)
class _PairCache:
    """Frozen density/cache payload for one atom pair."""

    iA: int
    iB: int
    Z_A: int
    Z_B: int
    zval_A: int
    zval_B: int
    ep_A: ElementParams
    ep_B: ElementParams
    beta_A: np.ndarray
    beta_B: np.ndarray
    P_AA: np.ndarray
    P_BB: np.ndarray
    P_AB: np.ndarray


def _validate_fock_mode(fock_mode: str) -> str:
    mode = str(fock_mode).strip().lower()
    if mode not in ("ri", "w", "auto"):
        raise ValueError("fock_mode must be 'ri', 'w', or 'auto'")
    return mode


def _normalize_gradient_backend(gradient_backend: str) -> str:
    mode = str(gradient_backend).strip().lower()
    if mode not in ("auto", "cuda_analytic", "cpu_frozen"):
        raise ValueError("gradient_backend must be 'auto', 'cuda_analytic', or 'cpu_frozen'")
    return mode


def _build_beta_ao(atomic_numbers: Sequence[int], elem_params: Dict[int, ElementParams]) -> np.ndarray:
    offsets = build_ao_offsets(atomic_numbers)
    out = np.zeros((int(offsets[-1]),), dtype=float)
    for iatom, Z in enumerate(atomic_numbers):
        ep = elem_params[int(Z)]
        i0 = int(offsets[iatom])
        nao = nao_for_Z(int(Z))
        out[i0] = ep.beta_s
        if nao == 4:
            out[i0 + 1 : i0 + 4] = ep.beta_p
    return out


def _pair_energy_from_positions(
    pos_A: np.ndarray,
    pos_B: np.ndarray,
    cache: _PairCache,
    elem_params: Dict[int, ElementParams],
    mp_params,
) -> float:
    """Compute one pair frozen-density energy contribution."""
    atomic_pair = [cache.Z_A, cache.Z_B]
    coords_pair = np.vstack((pos_A, pos_B))

    W, gamma_ss = build_pair_two_center_tensor(
        atomic_numbers=atomic_pair,
        coords_bohr=coords_pair,
        iA=0,
        iB=1,
        mp_params=mp_params,
    )
    S_AB = build_pair_overlap_block(
        atomic_numbers=atomic_pair,
        coords_bohr=coords_pair,
        iA=0,
        iB=1,
        elem_params=elem_params,
    )

    H_AA_pair = -float(cache.zval_B) * W[:, :, 0, 0]
    H_BB_pair = -float(cache.zval_A) * W[0, 0, :, :]
    H_AB_pair = 0.5 * S_AB * (cache.beta_A[:, None] + cache.beta_B[None, :])
    e_one = (
        float(np.sum(cache.P_AA * H_AA_pair))
        + float(np.sum(cache.P_BB * H_BB_pair))
        + float(2.0 * np.sum(cache.P_AB * H_AB_pair))
    )

    J_to_A = np.einsum("ls,mnls->mn", cache.P_BB, W, optimize=True)
    J_to_B = np.einsum("mn,mnls->ls", cache.P_AA, W, optimize=True)
    K_AB = np.einsum("ns,mnls->ml", cache.P_AB, W, optimize=True)
    e_two = (
        0.5 * float(np.sum(cache.P_AA * J_to_A))
        + 0.5 * float(np.sum(cache.P_BB * J_to_B))
        - 0.5 * float(np.sum(cache.P_AB * K_AB))
    )

    R = float(np.linalg.norm(pos_B - pos_A))
    e_core = _pair_core_repulsion(
        Z_A=cache.Z_A,
        Z_B=cache.Z_B,
        R=R,
        zval_A=cache.zval_A,
        zval_B=cache.zval_B,
        gamma_ss=float(gamma_ss),
        ep_A=cache.ep_A,
        ep_B=cache.ep_B,
    )
    return e_one + e_two + float(e_core)


def _am1_gradient_from_scf_cpu_frozen(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    scf_result: SCFResult,
    *,
    step_bohr: float = 1e-4,
) -> np.ndarray:
    """Reference frozen-density finite-difference AM1 gradient."""
    step = float(step_bohr)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step_bohr must be a positive finite float")

    atomic_numbers = [int(z) for z in atomic_numbers]
    coords = np.asarray(coords_bohr, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N, 3)")
    if len(atomic_numbers) != coords.shape[0]:
        raise ValueError("atomic_numbers length must match coords_bohr")

    elem_params = params.elements
    for Z in atomic_numbers:
        if Z not in elem_params:
            raise ValueError(f"No parameters for element Z={Z}")

    offsets = build_ao_offsets(atomic_numbers)
    nao_total = int(offsets[-1])
    P = np.asarray(scf_result.P, dtype=float)
    if P.shape != (nao_total, nao_total):
        raise ValueError(
            f"SCF density shape mismatch: got {P.shape}, expected ({nao_total}, {nao_total})"
        )

    beta_ao = _build_beta_ao(atomic_numbers, elem_params)
    pair_i, pair_j, _, _ = build_pair_list(coords)
    npairs = int(len(pair_i))
    if npairs == 0:
        return np.zeros((coords.shape[0], 3), dtype=float)

    mp_params = compute_all_multipole_params(elem_params)
    caches = []
    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        Z_A = int(atomic_numbers[iA])
        Z_B = int(atomic_numbers[iB])
        nao_A = nao_for_Z(Z_A)
        nao_B = nao_for_Z(Z_B)
        i0A = int(offsets[iA])
        i0B = int(offsets[iB])
        idxA = slice(i0A, i0A + nao_A)
        idxB = slice(i0B, i0B + nao_B)
        caches.append(
            _PairCache(
                iA=iA,
                iB=iB,
                Z_A=Z_A,
                Z_B=Z_B,
                zval_A=valence_electrons(Z_A),
                zval_B=valence_electrons(Z_B),
                ep_A=elem_params[Z_A],
                ep_B=elem_params[Z_B],
                beta_A=beta_ao[idxA].copy(),
                beta_B=beta_ao[idxB].copy(),
                P_AA=P[idxA, idxA].copy(),
                P_BB=P[idxB, idxB].copy(),
                P_AB=P[idxA, idxB].copy(),
            )
        )

    grad = np.zeros((coords.shape[0], 3), dtype=float)
    for cache in caches:
        base_A = coords[cache.iA].copy()
        base_B = coords[cache.iB].copy()

        for axis in range(3):
            # dE/dR_A,axis
            pos_A_p = base_A.copy()
            pos_A_m = base_A.copy()
            pos_A_p[axis] += step
            pos_A_m[axis] -= step
            e_p = _pair_energy_from_positions(pos_A_p, base_B, cache, elem_params, mp_params)
            e_m = _pair_energy_from_positions(pos_A_m, base_B, cache, elem_params, mp_params)
            grad[cache.iA, axis] += (e_p - e_m) / (2.0 * step)

            # dE/dR_B,axis
            pos_B_p = base_B.copy()
            pos_B_m = base_B.copy()
            pos_B_p[axis] += step
            pos_B_m[axis] -= step
            e_p = _pair_energy_from_positions(base_A, pos_B_p, cache, elem_params, mp_params)
            e_m = _pair_energy_from_positions(base_A, pos_B_m, cache, elem_params, mp_params)
            grad[cache.iB, axis] += (e_p - e_m) / (2.0 * step)

    return grad


def _run_cuda_analytic(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    scf_result: SCFResult,
    *,
    fock_mode: str,
):
    from .gpu.gradient_gpu import am1_gradient_cuda_analytic

    return am1_gradient_cuda_analytic(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        params=params,
        scf_result=scf_result,
        fock_mode=fock_mode,
    )


def am1_gradient_from_scf(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    scf_result: SCFResult,
    *,
    step_bohr: float = 1e-4,
    device: str = "cpu",
    fock_mode: str = "ri",
    gradient_backend: str = "auto",
    return_metadata: bool = False,
):
    """Build AM1 Cartesian gradient in Hartree/Bohr from converged SCF state.

    Parameters
    ----------
    step_bohr
        Central-difference step for ``gradient_backend='cpu_frozen'``.
        Ignored by ``cuda_analytic``.
    device
        ``'cpu'`` or ``'cuda'`` execution context.
    fock_mode
        Accepted and validated for interface consistency (``ri|w|auto``).
    gradient_backend
        ``auto`` (default), ``cuda_analytic``, or ``cpu_frozen``.
    return_metadata
        If True, return ``(gradient, metadata)``.
    """
    if not bool(scf_result.converged):
        raise RuntimeError(
            "AM1 gradient requires a converged SCF state; run with tighter SCF settings "
            "or check geometry/initial guess."
        )

    run_device = str(device).strip().lower()
    if run_device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")

    mode = _validate_fock_mode(fock_mode)
    backend = _normalize_gradient_backend(gradient_backend)

    def _cpu_run():
        t0 = time.perf_counter()
        g = _am1_gradient_from_scf_cpu_frozen(
            atomic_numbers=atomic_numbers,
            coords_bohr=coords_bohr,
            params=params,
            scf_result=scf_result,
            step_bohr=step_bohr,
        )
        t1 = time.perf_counter()
        meta = {
            "gradient_backend_used": "cpu_frozen",
            "gradient_pack_time_s": 0.0,
            "gradient_kernel_time_s": float(t1 - t0),
            "gradient_post_time_s": 0.0,
        }
        return g, meta

    if backend == "cpu_frozen":
        grad, meta = _cpu_run()
        return (grad, meta) if return_metadata else grad

    if backend == "cuda_analytic":
        if run_device != "cuda":
            raise ValueError("gradient_backend='cuda_analytic' requires device='cuda'")
        grad, meta = _run_cuda_analytic(
            atomic_numbers=atomic_numbers,
            coords_bohr=coords_bohr,
            params=params,
            scf_result=scf_result,
            fock_mode=mode,
        )
        return (grad, meta) if return_metadata else grad

    # auto backend
    if run_device == "cuda":
        try:
            grad, meta = _run_cuda_analytic(
                atomic_numbers=atomic_numbers,
                coords_bohr=coords_bohr,
                params=params,
                scf_result=scf_result,
                fock_mode=mode,
            )
            return (grad, meta) if return_metadata else grad
        except Exception as exc:
            warnings.warn(
                "AM1 CUDA analytical gradient backend failed; falling back to "
                f"cpu_frozen. Reason: {type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            grad, meta = _cpu_run()
            meta["gradient_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return (grad, meta) if return_metadata else grad

    grad, meta = _cpu_run()
    return (grad, meta) if return_metadata else grad


def am1_energy_gradient_scf(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    scf_result: SCFResult,
    *,
    step_bohr: float = 1e-4,
    device: str = "cpu",
    fock_mode: str = "ri",
    gradient_backend: str = "auto",
    return_metadata: bool = False,
):
    """Return converged SCF result with Cartesian gradient (Hartree/Bohr)."""
    out = am1_gradient_from_scf(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        params=params,
        scf_result=scf_result,
        step_bohr=step_bohr,
        device=device,
        fock_mode=fock_mode,
        gradient_backend=gradient_backend,
        return_metadata=True,
    )
    grad, meta = out
    if return_metadata:
        return scf_result, grad, meta
    return scf_result, grad


__all__ = ["am1_gradient_from_scf", "am1_energy_gradient_scf"]
