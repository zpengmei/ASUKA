from __future__ import annotations

"""cuERI-powered DF (density fitting) on CPU.

This module provides a CPU fallback for building whitened AO DF
factors `B[μ,ν,Q]` such that:

  (μν|λσ) ~= Σ_Q B[μν,Q] B[λσ,Q]

Approach
-----------------------
Reuse the cuERI Step-2 (general-l) CPU Rys ERI evaluator by representing:
- metric: (P|Q) == (P*1 | Q*1)
- 3c2e:   (μν|P) == (μν | P*1)

where "1" is modeled as a dummy s-shell with one primitive (exp=0, coef=1).
"""

import time
import warnings

import numpy as np
from scipy.linalg import solve_triangular

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.cueri.cart import ncart
from asuka.cueri.dense_cpu import CPU_MAX_L
from asuka.cueri.shell_pairs import ShellPairs, build_shell_pairs_l_order
from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu
from asuka.integrals.int1e_cart import nao_cart_from_basis


def _require_eri_cpu_ext():
    try:
        from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU ERI extension is not built. Build it with:\n"
            "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
        ) from e
    return _ext


def _build_df_combined_basis_and_shell_pairs(
    ao_basis: BasisCartSoA,
    aux_basis: BasisCartSoA,
) -> tuple[BasisCartSoA, ShellPairs, int, int, int]:
    """Return (basis_all, sp_all, nsp_ao, n_shell_ao, n_shell_aux)."""

    max_l = 0
    if int(ao_basis.shell_l.size):
        max_l = max(max_l, int(np.max(np.asarray(ao_basis.shell_l, dtype=np.int32))))
    if int(aux_basis.shell_l.size):
        max_l = max(max_l, int(np.max(np.asarray(aux_basis.shell_l, dtype=np.int32))))
    if max_l > CPU_MAX_L:
        raise NotImplementedError(f"cuERI CPU DF currently supports only l<={CPU_MAX_L}")

    n_shell_ao = int(np.asarray(ao_basis.shell_cxyz, dtype=np.float64).shape[0])
    n_shell_aux = int(np.asarray(aux_basis.shell_cxyz, dtype=np.float64).shape[0])
    dummy_shell = int(n_shell_ao + n_shell_aux)

    ao_prim_n = int(np.asarray(ao_basis.prim_exp, dtype=np.float64).shape[0])
    aux_prim_n = int(np.asarray(aux_basis.prim_exp, dtype=np.float64).shape[0])

    nao = nao_cart_from_basis(ao_basis)
    naux = nao_cart_from_basis(aux_basis)

    shell_cxyz = np.concatenate(
        [
            np.asarray(ao_basis.shell_cxyz, dtype=np.float64, order="C"),
            np.asarray(aux_basis.shell_cxyz, dtype=np.float64, order="C"),
            np.zeros((1, 3), dtype=np.float64),
        ],
        axis=0,
    )
    shell_prim_start = np.concatenate(
        [
            np.asarray(ao_basis.shell_prim_start, dtype=np.int32, order="C"),
            np.asarray(aux_basis.shell_prim_start, dtype=np.int32, order="C") + np.int32(ao_prim_n),
            np.asarray([ao_prim_n + aux_prim_n], dtype=np.int32),
        ],
        axis=0,
    )
    shell_nprim = np.concatenate(
        [
            np.asarray(ao_basis.shell_nprim, dtype=np.int32, order="C"),
            np.asarray(aux_basis.shell_nprim, dtype=np.int32, order="C"),
            np.asarray([1], dtype=np.int32),
        ],
        axis=0,
    )
    shell_l = np.concatenate(
        [
            np.asarray(ao_basis.shell_l, dtype=np.int32, order="C"),
            np.asarray(aux_basis.shell_l, dtype=np.int32, order="C"),
            np.asarray([0], dtype=np.int32),
        ],
        axis=0,
    )
    shell_ao_start = np.concatenate(
        [
            np.asarray(ao_basis.shell_ao_start, dtype=np.int32, order="C"),
            (np.asarray(aux_basis.shell_ao_start, dtype=np.int32, order="C") + np.int32(nao)),
            np.asarray([nao + naux], dtype=np.int32),
        ],
        axis=0,
    )
    prim_exp = np.concatenate(
        [
            np.asarray(ao_basis.prim_exp, dtype=np.float64, order="C"),
            np.asarray(aux_basis.prim_exp, dtype=np.float64, order="C"),
            np.asarray([0.0], dtype=np.float64),
        ],
        axis=0,
    )
    prim_coef = np.concatenate(
        [
            np.asarray(ao_basis.prim_coef, dtype=np.float64, order="C"),
            np.asarray(aux_basis.prim_coef, dtype=np.float64, order="C"),
            np.asarray([1.0], dtype=np.float64),
        ],
        axis=0,
    )

    basis_all = BasisCartSoA(
        shell_cxyz=shell_cxyz,
        shell_prim_start=shell_prim_start,
        shell_nprim=shell_nprim,
        shell_l=shell_l,
        shell_ao_start=shell_ao_start,
        prim_exp=prim_exp,
        prim_coef=prim_coef,
    )

    # ShellPairs:
    # - AO side: all unique unordered AO shell pairs, oriented with l(A) >= l(B).
    sp_ao = build_shell_pairs_l_order(ao_basis)
    nsp_ao = int(sp_ao.sp_A.shape[0])

    # - Aux side: (P, dummy) for each aux shell P.
    aux_shell_idx = np.arange(n_shell_aux, dtype=np.int32)
    sp_aux_A = (aux_shell_idx + np.int32(n_shell_ao)).astype(np.int32, copy=False)
    sp_aux_B = np.full((int(aux_shell_idx.size),), int(dummy_shell), dtype=np.int32)
    sp_aux_npair = np.asarray(aux_basis.shell_nprim, dtype=np.int32, order="C").ravel().astype(np.int32, copy=False)

    sp_A = np.concatenate([np.asarray(sp_ao.sp_A, dtype=np.int32, order="C"), sp_aux_A], axis=0)
    sp_B = np.concatenate([np.asarray(sp_ao.sp_B, dtype=np.int32, order="C"), sp_aux_B], axis=0)
    sp_npair = np.concatenate([np.asarray(sp_ao.sp_npair, dtype=np.int32, order="C"), sp_aux_npair], axis=0)
    sp_pair_start = np.empty((int(sp_npair.shape[0]) + 1,), dtype=np.int32)
    sp_pair_start[0] = 0
    sp_pair_start[1:] = np.cumsum(sp_npair, dtype=np.int32)

    sp_all = ShellPairs(sp_A=sp_A, sp_B=sp_B, sp_npair=sp_npair, sp_pair_start=sp_pair_start)
    return basis_all, sp_all, int(nsp_ao), int(n_shell_ao), int(n_shell_aux)


def build_df_B_from_cueri_packed_bases_cpu(
    ao_basis: BasisCartSoA,
    aux_basis: BasisCartSoA,
    *,
    threads: int = 0,
    profile: dict | None = None,
) -> np.ndarray:
    """Build whitened AO DF factors B[μ,ν,Q] on CPU using cuERI Step-2 tiles."""

    if profile is not None:
        profile.clear()

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    _ext = _require_eri_cpu_ext()
    if threads_i > 1 and hasattr(_ext, "openmp_enabled") and not bool(_ext.openmp_enabled()):
        warnings.warn(
            "threads>1 requested but asuka.cueri._eri_rys_cpu was built without OpenMP; "
            "rebuild with CUERI_USE_OPENMP=1 to enable parallelism",
            RuntimeWarning,
            stacklevel=2,
        )

    nao = nao_cart_from_basis(ao_basis)
    naux = nao_cart_from_basis(aux_basis)
    if nao == 0 or naux == 0:
        return np.empty((nao, nao, naux), dtype=np.float64)

    basis_all, sp_all, nsp_ao, _n_shell_ao, n_shell_aux = _build_df_combined_basis_and_shell_pairs(ao_basis, aux_basis)
    aux_sp0 = int(nsp_ao)

    # Pair tables for all shell pairs (AO pairs + aux*dummy pairs).
    pt_prof = None
    if profile is not None:
        pt_prof = profile.setdefault("pair_tables", {})
    pair_tables = build_pair_tables_cpu(basis_all, sp_all, threads=threads_i, profile=pt_prof)

    shell_cxyz = np.asarray(basis_all.shell_cxyz, dtype=np.float64, order="C")
    shell_l = np.asarray(basis_all.shell_l, dtype=np.int32, order="C")

    sp_A = np.asarray(sp_all.sp_A, dtype=np.int32, order="C")
    sp_B = np.asarray(sp_all.sp_B, dtype=np.int32, order="C")
    sp_pair_start = np.asarray(sp_all.sp_pair_start, dtype=np.int32, order="C")
    sp_npair = np.asarray(sp_all.sp_npair, dtype=np.int32, order="C")

    pair_eta = np.asarray(pair_tables.pair_eta, dtype=np.float64, order="C")
    pair_Px = np.asarray(pair_tables.pair_Px, dtype=np.float64, order="C")
    pair_Py = np.asarray(pair_tables.pair_Py, dtype=np.float64, order="C")
    pair_Pz = np.asarray(pair_tables.pair_Pz, dtype=np.float64, order="C")
    pair_cK = np.asarray(pair_tables.pair_cK, dtype=np.float64, order="C")

    eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
    if eri_batch is None:  # pragma: no cover
        raise RuntimeError("CPU ERI extension is missing batch tile entry points; rebuild the extension")

    # Group aux shells by l to satisfy the batch evaluator requirement.
    aux_shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32, order="C").ravel()
    aux_shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32, order="C").ravel()

    by_l: dict[int, list[int]] = {}
    for sh in range(n_shell_aux):
        by_l.setdefault(int(aux_shell_l[sh]), []).append(int(sh))

    # ---- Metric V(P,Q) = (P|Q) ----
    t0 = time.perf_counter() if profile is not None else 0.0
    V = np.zeros((naux, naux), dtype=np.float64)
    for psh in range(n_shell_aux):
        lp = int(aux_shell_l[psh])
        nP = int(ncart(lp))
        p0 = int(aux_shell_ao_start[psh])
        spAB = int(aux_sp0 + psh)

        for lq, q_shells in by_l.items():
            nQ = int(ncart(int(lq)))
            q_list = [int(q) for q in q_shells if int(q) <= int(psh)]
            if not q_list:
                continue

            spCD = (aux_sp0 + np.asarray(q_list, dtype=np.int32)).astype(np.int32, copy=False)
            tiles = eri_batch(
                shell_cxyz,
                shell_l,
                sp_A,
                sp_B,
                sp_pair_start,
                sp_npair,
                pair_eta,
                pair_Px,
                pair_Py,
                pair_Pz,
                pair_cK,
                int(spAB),
                spCD,
                int(threads_i),
            )

            for t, qsh in enumerate(q_list):
                q0 = int(aux_shell_ao_start[qsh])
                block = np.asarray(tiles[int(t)], dtype=np.float64, order="C").reshape((nP, nQ))
                V[p0 : p0 + nP, q0 : q0 + nQ] = block
                if qsh != psh:
                    V[q0 : q0 + nQ, p0 : p0 + nP] = block.T

    if profile is not None:
        profile["t_metric_s"] = float(time.perf_counter() - t0)

    V = 0.5 * (V + V.T)

    # ---- 3c2e X(μ,ν,P) = (μν|P) ----
    t0 = time.perf_counter() if profile is not None else 0.0
    X_flat = np.zeros((int(nao) * int(nao), int(naux)), dtype=np.float64)

    ao_shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32, order="C").ravel()
    ao_shell_ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32, order="C").ravel()

    sp_ao_A = np.asarray(sp_all.sp_A[:nsp_ao], dtype=np.int32, order="C")
    sp_ao_B = np.asarray(sp_all.sp_B[:nsp_ao], dtype=np.int32, order="C")

    # Pre-build spCD arrays per aux-l group to reduce Python overhead in the main loop.
    spCD_by_l: dict[int, np.ndarray] = {}
    shells_by_l: dict[int, list[int]] = {}
    for lq, q_shells in by_l.items():
        shells_by_l[int(lq)] = list(map(int, q_shells))
        spCD_by_l[int(lq)] = (aux_sp0 + np.asarray(q_shells, dtype=np.int32)).astype(np.int32, copy=False)

    for spAB in range(nsp_ao):
        Ash = int(sp_ao_A[spAB])
        Bsh = int(sp_ao_B[spAB])
        la = int(ao_shell_l[Ash])
        lb = int(ao_shell_l[Bsh])
        nA = int(ncart(la))
        nB = int(ncart(lb))
        a0 = int(ao_shell_ao_start[Ash])
        b0 = int(ao_shell_ao_start[Bsh])

        idxA = (a0 + np.arange(nA, dtype=np.int32))[:, None]
        idxB = (b0 + np.arange(nB, dtype=np.int32))[None, :]
        rows = (idxA * np.int64(nao) + idxB).reshape((nA * nB,))

        if Ash != Bsh:
            idxA2 = (b0 + np.arange(nB, dtype=np.int32))[:, None]
            idxB2 = (a0 + np.arange(nA, dtype=np.int32))[None, :]
            rows_T = (idxA2 * np.int64(nao) + idxB2).reshape((nB * nA,))
        else:
            rows_T = None

        for lq, spCD in spCD_by_l.items():
            nQ = int(ncart(int(lq)))
            tiles = eri_batch(
                shell_cxyz,
                shell_l,
                sp_A,
                sp_B,
                sp_pair_start,
                sp_npair,
                pair_eta,
                pair_Px,
                pair_Py,
                pair_Pz,
                pair_cK,
                int(spAB),
                spCD,
                int(threads_i),
            )
            q_shells = shells_by_l[int(lq)]
            for t, qsh in enumerate(q_shells):
                q0 = int(aux_shell_ao_start[qsh])
                tile = np.asarray(tiles[int(t)], dtype=np.float64, order="C")
                if tile.shape != (nA * nB, nQ):
                    tile = tile.reshape((nA * nB, nQ))
                X_flat[rows, q0 : q0 + nQ] = tile
                if rows_T is not None:
                    tile_T = tile.reshape((nA, nB, nQ)).transpose((1, 0, 2)).reshape((nB * nA, nQ))
                    X_flat[rows_T, q0 : q0 + nQ] = tile_T

    if profile is not None:
        profile["t_int3c2e_s"] = float(time.perf_counter() - t0)

    # ---- Cholesky + whitening ----
    t0 = time.perf_counter() if profile is not None else 0.0
    L = np.linalg.cholesky(V)
    if profile is not None:
        profile["t_metric_cholesky_s"] = float(time.perf_counter() - t0)

    t0 = time.perf_counter() if profile is not None else 0.0
    BT = solve_triangular(L, X_flat.T, lower=True, trans="N", unit_diagonal=False, overwrite_b=False)
    B_flat = np.asarray(BT.T, dtype=np.float64, order="C")
    B = B_flat.reshape((int(nao), int(nao), int(naux)))
    if profile is not None:
        profile["t_whiten_s"] = float(time.perf_counter() - t0)
        profile["nao"] = int(nao)
        profile["naux"] = int(naux)
        profile["threads"] = int(threads_i)

    return B


__all__ = ["build_df_B_from_cueri_packed_bases_cpu"]
