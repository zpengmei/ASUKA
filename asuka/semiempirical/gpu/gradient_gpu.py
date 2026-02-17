"""CUDA analytical AM1 gradient launcher (pairwise dual-number kernels)."""

from __future__ import annotations

import time
from typing import Dict, Sequence, Tuple

import numpy as np

from asuka.nddo_core import (
    build_ao_offsets,
    build_pair_list,
    compute_all_multipole_params,
    nao_for_Z,
    valence_electrons,
)
from asuka.semiempirical.params import MethodParams
from asuka.semiempirical.scf import SCFResult

from .kernels import (
    build_pair_buckets,
    ensure_gradient_kernel_sources_available,
    get_gradient_kernels,
)
from .runtime import _import_cupy


def _require_cuda_runtime():
    cp = _import_cupy()
    if cp is None:
        raise RuntimeError(
            "CuPy is required for CUDA analytical AM1 gradients. "
            "Install ASUKA with CUDA extras (for example: pip install -e '.[cuda]')."
        )
    try:
        ndev = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError("Unable to query CUDA devices via CuPy runtime") from exc
    if ndev < 1:
        raise RuntimeError("No CUDA device is visible to CuPy (CUDA analytical gradients unavailable)")
    return cp


def _build_atom_param_pack(
    atomic_numbers: Sequence[int],
    params: MethodParams,
) -> np.ndarray:
    """Pack atom parameters used by CUDA gradient kernels.

    Layout per atom (length 32):
      0 Z, 1 zval, 2 zeta_s, 3 zeta_p, 4 beta_s, 5 beta_p, 6 alpha, 7 ngauss,
      8:12 gk, 12:16 gl, 16:20 gm,
      20 dd, 21 qq, 22 am, 23 ad, 24 aq
    Remaining slots are reserved.
    """
    nat = len(atomic_numbers)
    out = np.zeros((nat, 32), dtype=np.float64)

    mp_params = compute_all_multipole_params(params.elements)
    for iatom, Z in enumerate(atomic_numbers):
        ep = params.elements[int(Z)]
        mp = mp_params[int(Z)]
        out[iatom, 0] = float(int(Z))
        out[iatom, 1] = float(valence_electrons(int(Z)))
        out[iatom, 2] = float(ep.zeta_s)
        out[iatom, 3] = float(ep.zeta_p)
        out[iatom, 4] = float(ep.beta_s)
        out[iatom, 5] = float(ep.beta_p)
        out[iatom, 6] = float(ep.alpha)
        out[iatom, 7] = float(len(ep.gaussians))
        for ig, g in enumerate(ep.gaussians[:4]):
            out[iatom, 8 + ig] = float(g.k)
            out[iatom, 12 + ig] = float(g.l)
            out[iatom, 16 + ig] = float(g.m)
        out[iatom, 20] = float(mp.dd)
        out[iatom, 21] = float(mp.qq)
        out[iatom, 22] = float(mp.am)
        out[iatom, 23] = float(mp.ad)
        out[iatom, 24] = float(mp.aq)

    return out


def _pack_pair_density_blocks(
    atomic_numbers: Sequence[int],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    P: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack per-pair density blocks into padded 4x4 payloads."""
    offsets = build_ao_offsets(atomic_numbers)
    npairs = int(len(pair_i))

    paa = np.zeros((npairs, 16), dtype=np.float64)
    pbb = np.zeros((npairs, 16), dtype=np.float64)
    pab = np.zeros((npairs, 16), dtype=np.float64)

    for k in range(npairs):
        iA = int(pair_i[k])
        iB = int(pair_j[k])
        ZA = int(atomic_numbers[iA])
        ZB = int(atomic_numbers[iB])
        naoA = nao_for_Z(ZA)
        naoB = nao_for_Z(ZB)
        i0A = int(offsets[iA])
        i0B = int(offsets[iB])
        idxA = slice(i0A, i0A + naoA)
        idxB = slice(i0B, i0B + naoB)

        PAA = np.zeros((4, 4), dtype=np.float64)
        PBB = np.zeros((4, 4), dtype=np.float64)
        PAB = np.zeros((4, 4), dtype=np.float64)
        PAA[:naoA, :naoA] = P[idxA, idxA]
        PBB[:naoB, :naoB] = P[idxB, idxB]
        PAB[:naoA, :naoB] = P[idxA, idxB]

        paa[k, :] = PAA.ravel()
        pbb[k, :] = PBB.ravel()
        pab[k, :] = PAB.ravel()

    return paa, pbb, pab


def am1_gradient_cuda_analytic(
    atomic_numbers: Sequence[int],
    coords_bohr: np.ndarray,
    params: MethodParams,
    scf_result: SCFResult,
    *,
    fock_mode: str = "ri",
) -> Tuple[np.ndarray, Dict[str, float | str]]:
    """Compute AM1 Cartesian gradients with CUDA dual-number pair kernels.

    Parameters
    ----------
    fock_mode
        Accepted for API parity with SCF/gradient entrypoints. The current
        analytical gradient path does not branch on this value.
    """
    cp = _require_cuda_runtime()
    ensure_gradient_kernel_sources_available()

    mode = str(fock_mode).strip().lower()
    if mode not in ("ri", "w", "auto"):
        raise ValueError("fock_mode must be 'ri', 'w', or 'auto'")

    atomic_numbers = [int(z) for z in atomic_numbers]
    coords = np.asarray(coords_bohr, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N, 3)")
    nat = int(coords.shape[0])

    P = np.asarray(scf_result.P, dtype=np.float64)
    offsets = build_ao_offsets(atomic_numbers)
    nao_total = int(offsets[-1])
    if P.shape != (nao_total, nao_total):
        raise ValueError(
            f"SCF density shape mismatch: got {P.shape}, expected ({nao_total}, {nao_total})"
        )

    t_pack0 = time.perf_counter()
    pair_i, pair_j, _, _ = build_pair_list(coords)
    npairs = int(len(pair_i))
    if npairs == 0:
        grad = np.zeros((nat, 3), dtype=np.float64)
        meta = {
            "gradient_backend_used": "cuda_analytic",
            "gradient_pack_time_s": 0.0,
            "gradient_kernel_time_s": 0.0,
            "gradient_post_time_s": 0.0,
        }
        return grad, meta

    atom_pack = _build_atom_param_pack(atomic_numbers, params)
    paa, pbb, pab = _pack_pair_density_blocks(atomic_numbers, pair_i, pair_j, P)
    buckets = build_pair_buckets(atomic_numbers, pair_i, pair_j)
    t_pack1 = time.perf_counter()

    kernels = get_gradient_kernels()

    coords_d = cp.asarray(coords.reshape(-1), dtype=cp.float64)
    atom_pack_d = cp.asarray(atom_pack, dtype=cp.float64)
    grad_d = cp.zeros((nat, 3), dtype=cp.float64)

    evt_start = cp.cuda.Event()
    evt_stop = cp.cuda.Event()
    evt_start.record()

    block_size = 128
    for key in ("11", "14", "41", "44"):
        idx = buckets[key]
        n = int(len(idx))
        if n == 0:
            continue

        pair_i_d = cp.asarray(pair_i[idx].astype(np.int32))
        pair_j_d = cp.asarray(pair_j[idx].astype(np.int32))
        paa_d = cp.asarray(paa[idx, :], dtype=cp.float64)
        pbb_d = cp.asarray(pbb[idx, :], dtype=cp.float64)
        pab_d = cp.asarray(pab[idx, :], dtype=cp.float64)

        grid_size = (n + block_size - 1) // block_size
        kernels[key](
            (grid_size,),
            (block_size,),
            (
                pair_i_d,
                pair_j_d,
                coords_d,
                atom_pack_d,
                paa_d,
                pbb_d,
                pab_d,
                grad_d,
                np.int32(n),
            ),
        )

    evt_stop.record()
    evt_stop.synchronize()
    kernel_time_s = float(cp.cuda.get_elapsed_time(evt_start, evt_stop)) * 1e-3

    t_post0 = time.perf_counter()
    grad = cp.asnumpy(grad_d)
    t_post1 = time.perf_counter()

    meta: Dict[str, float | str] = {
        "gradient_backend_used": "cuda_analytic",
        "gradient_pack_time_s": float(t_pack1 - t_pack0),
        "gradient_kernel_time_s": kernel_time_s,
        "gradient_post_time_s": float(t_post1 - t_post0),
    }
    return grad, meta


__all__ = ["am1_gradient_cuda_analytic"]
