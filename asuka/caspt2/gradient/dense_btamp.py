"""Dense debug helpers for the Molcas CASPT2_BTAMP contribution.

These helpers are diagnostic only. They reconstruct dense AO-quartet
back-transformed-amplitude candidates from the OLagNS ``t2_mo`` tensor using
the same stored half-transformed amplitude consumed by OpenMolcas
``CASPT2_BTAMP``/``PGet3``.

OpenMolcas does not contract the raw ``T2AO`` tensor directly in the
conventional gradient path. ``OLagVVVO`` first writes a symmetrized
half-transformed record

    T_hbf(i,j,mu,nu) = T2AO(j,nu,i,mu) + T2AO(i,mu,j,nu)

to the ``GAMMA`` file, and ``CASPT2_BTAMP`` then forms the four-term ``Q``
decomposition

    G_toc(nu,sigma,mu,lambda)
      = 1/8 * [
          Q(nu,sigma;mu,lambda)
        + Q(mu,lambda;nu,sigma)
        + Q(nu,lambda;mu,sigma)
        + Q(mu,sigma;nu,lambda)
      ]

where

    Q(x,y;u,v) = sum_{i,j} C_{x j} C_{y i} T_hbf(i,j,u,v)

and ``T_hbf(i,j,u,v)`` is represented here by the dense AO tensor
``t2ao[j, v, i, u] + t2ao[i, u, j, v]`` built from ``t2_mo[j, s, i, r]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _asnumpy_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def build_t2ao_dense(
    *,
    mo_coeff: np.ndarray,
    t2_mo: np.ndarray,
) -> np.ndarray:
    """Transform ``t2_mo[j,s,i,r]`` to AO form ``T2AO[j,ao_s,i,ao_r]``."""
    c = _asnumpy_f64(mo_coeff)
    t = _asnumpy_f64(t2_mo)
    if c.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    if t.ndim != 4:
        raise ValueError("t2_mo must be 4D")
    nocc = int(t.shape[0])
    if int(t.shape[2]) != nocc:
        raise ValueError("t2_mo occupied axes must match")
    if int(t.shape[1]) != int(c.shape[1]) or int(t.shape[3]) != int(c.shape[1]):
        raise ValueError("t2_mo MO axes must match mo_coeff")
    return np.asarray(
        np.einsum("jsir,as,br->jaib", t, c, c, optimize=True),
        dtype=np.float64,
    )


def build_gtoc_dense_term_arrays(
    *,
    mo_coeff: np.ndarray,
    t2_mo: np.ndarray,
    scale: float = 0.125,
) -> dict[str, np.ndarray]:
    """Reconstruct dense AO-quartet BTAMP candidate tensors.

    Returns tensors in the stored Molcas ``G_toc(j,l,i,k)`` ordering used by
    ``PGet3``.
    """
    c = _asnumpy_f64(mo_coeff)
    nocc = int(t2_mo.shape[0])
    c_occ = np.asarray(c[:, :nocc], dtype=np.float64)
    t2ao = build_t2ao_dense(mo_coeff=c, t2_mo=t2_mo)
    nao = int(c.shape[0])
    scl = float(scale)

    def _transform_occ_pair(w_ji: np.ndarray) -> np.ndarray:
        """Form Q(x,y;u,v) from a dense occupied-pair slice.

        ``w_ji[j,i]`` corresponds to the Molcas stored-record ordering in
        ``Q(x,y;u,v) = sum_{i,j} C_{xj} C_{yi} w_ji[j,i]``.
        """
        return np.asarray(scl * (c_occ @ w_ji @ c_occ.T), dtype=np.float64)

    g_ik_to_jl = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    for ia in range(nao):
        for ka in range(nao):
            wrk = np.asarray(t2ao[:, ka, :, ia] + t2ao[:, ia, :, ka].T, dtype=np.float64)
            g_ik_to_jl[:, :, ia, ka] = _transform_occ_pair(wrk)

    g_il_to_jk = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    for ia in range(nao):
        for la in range(nao):
            wrk = np.asarray(t2ao[:, la, :, ia] + t2ao[:, ia, :, la].T, dtype=np.float64)
            blk = _transform_occ_pair(wrk)
            for ja in range(nao):
                g_il_to_jk[ja, la, ia, :] = blk[ja, :]

    g_jl_to_ik = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    for ja in range(nao):
        for la in range(nao):
            wrk = np.asarray(t2ao[:, la, :, ja] + t2ao[:, ja, :, la].T, dtype=np.float64)
            blk = _transform_occ_pair(wrk)
            for ia in range(nao):
                g_jl_to_ik[ja, la, ia, :] = blk[ia, :]

    g_jk_to_il = np.zeros((nao, nao, nao, nao), dtype=np.float64)
    for ja in range(nao):
        for ka in range(nao):
            wrk = np.asarray(t2ao[:, ka, :, ja] + t2ao[:, ja, :, ka].T, dtype=np.float64)
            blk = _transform_occ_pair(wrk)
            for ia in range(nao):
                for la in range(nao):
                    g_jk_to_il[ja, la, ia, ka] = blk[ia, la]

    total = np.asarray(g_ik_to_jl + g_il_to_jk + g_jl_to_ik + g_jk_to_il, dtype=np.float64)
    pair_only = np.asarray(g_il_to_jk + g_jk_to_il, dtype=np.float64)
    return {
        "ik_to_jl": g_ik_to_jl,
        "il_to_jk": g_il_to_jk,
        "jl_to_ik": g_jl_to_ik,
        "jk_to_il": g_jk_to_il,
        "pair_only": pair_only,
        "total": total,
    }


def _fold_ordered_ao4_to_shellpair_tile(
    *,
    bar_ao4: np.ndarray,
    shell_ao_start: np.ndarray,
    sp_a: np.ndarray,
    sp_b: np.ndarray,
    shell_l: np.ndarray,
    sp_ab: int,
    sp_cd: int,
    ncart_fn,
) -> np.ndarray:
    """Fold an ordered AO 4-tensor into a canonical shell-pair tile.

    The dense contracted 4c derivative kernel iterates over canonical shell
    pairs only. For off-diagonal shell pairs this means the ordered AO tensor
    must be folded as ``AB + BA`` on rows and ``CD + DC`` on columns before the
    tile is passed to the kernel.
    """

    A = int(sp_a[int(sp_ab)])
    B = int(sp_b[int(sp_ab)])
    C = int(sp_a[int(sp_cd)])
    D = int(sp_b[int(sp_cd)])
    aoA = int(shell_ao_start[A])
    aoB = int(shell_ao_start[B])
    aoC = int(shell_ao_start[C])
    aoD = int(shell_ao_start[D])
    nA = int(ncart_fn(int(shell_l[A])))
    nB = int(ncart_fn(int(shell_l[B])))
    nC = int(ncart_fn(int(shell_l[C])))
    nD = int(ncart_fn(int(shell_l[D])))

    tile = np.asarray(
        bar_ao4[aoA : aoA + nA, aoB : aoB + nB, aoC : aoC + nC, aoD : aoD + nD],
        dtype=np.float64,
    )
    if A != B:
        tile = tile + np.asarray(
            bar_ao4[aoB : aoB + nB, aoA : aoA + nA, aoC : aoC + nC, aoD : aoD + nD],
            dtype=np.float64,
        ).transpose(1, 0, 2, 3)
    if C != D:
        tile = tile + np.asarray(
            bar_ao4[aoA : aoA + nA, aoB : aoB + nB, aoD : aoD + nD, aoC : aoC + nC],
            dtype=np.float64,
        ).transpose(0, 1, 3, 2)
    if A != B and C != D:
        tile = tile + np.asarray(
            bar_ao4[aoB : aoB + nB, aoA : aoA + nA, aoD : aoD + nD, aoC : aoC + nC],
            dtype=np.float64,
        ).transpose(1, 0, 3, 2)
    return np.asarray(tile.reshape(nA * nB, nC * nD), dtype=np.float64)


def contract_ordered_ao4_dense_deriv(
    *,
    ao_basis: Any,
    atom_coords_bohr: np.ndarray,
    bar_ao4: np.ndarray,
    cache_cpu: Any | None = None,
    max_tile_bytes: int = 64 << 20,
    threads: int = 0,
) -> np.ndarray:
    """Contract an ordered AO 4-tensor against dense 4c ERI derivatives.

    This is a diagnostic helper for tiny systems. It folds the ordered AO
    tensor into the canonical shell-pair convention expected by the dense
    contracted derivative kernel, so it reproduces the full ordered
    ``sum_pqrs bar[p,q,r,s] * d(pq|rs)/dR`` contraction.
    """

    from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
    from asuka.cueri.cart import ncart  # noqa: PLC0415
    from asuka.integrals.eri4c_deriv_contracted import (  # noqa: PLC0415
        accumulate_eri4c_task_derivs_to_atoms,
    )
    from asuka.mcscf.dense_eri_cpu import (  # noqa: PLC0415
        build_dense_eri4c_deriv_contraction_cache_cpu,
    )

    bar4 = np.asarray(bar_ao4, dtype=np.float64)
    if bar4.ndim != 4:
        raise ValueError("bar_ao4 must be a rank-4 ordered AO tensor")
    nao = int(bar4.shape[0])
    if bar4.shape != (nao, nao, nao, nao):
        raise ValueError(f"bar_ao4 must have shape (nao,nao,nao,nao), got {bar4.shape}")

    coords = np.asarray(atom_coords_bohr, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=coords,
            pair_table_threads=0,
        )

    shell_l = np.asarray(cache_cpu.shell_l, dtype=np.int32)
    shell_ao_start = np.asarray(cache_cpu.shell_ao_start, dtype=np.int32)
    sp_a = np.asarray(cache_cpu.sp_A, dtype=np.int32)
    sp_b = np.asarray(cache_cpu.sp_B, dtype=np.int32)
    natm = int(coords.shape[0])
    grad = np.zeros((natm, 3), dtype=np.float64)
    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    for sp_ab in range(int(sp_a.size)):
        A = int(sp_a[sp_ab])
        B = int(sp_b[sp_ab])
        nA = int(ncart(int(shell_l[A])))
        nB = int(ncart(int(shell_l[B])))
        nAB = int(nA * nB)
        for sp_cd_all in cache_cpu.spCD_by_key.values():
            sp_cd_all = np.asarray(sp_cd_all, dtype=np.int32)
            if int(sp_cd_all.size) == 0:
                continue
            C0 = int(sp_a[int(sp_cd_all[0])])
            D0 = int(sp_b[int(sp_cd_all[0])])
            nC = int(ncart(int(shell_l[C0])))
            nD = int(ncart(int(shell_l[D0])))
            nCD = int(nC * nD)
            chunk_nt = int(max(1, max_tile_bytes_i // max(8 * nAB * nCD, 1)))
            for i0 in range(0, int(sp_cd_all.size), chunk_nt):
                sp_cd = np.asarray(sp_cd_all[i0 : i0 + chunk_nt], dtype=np.int32)
                nt = int(sp_cd.size)
                tiles = np.empty((nt, nAB, nCD), dtype=np.float64)
                for t, sp_cd_t in enumerate(sp_cd.tolist()):
                    tiles[t] = _fold_ordered_ao4_to_shellpair_tile(
                        bar_ao4=bar4,
                        shell_ao_start=shell_ao_start,
                        sp_a=sp_a,
                        sp_b=sp_b,
                        shell_l=shell_l,
                        sp_ab=int(sp_ab),
                        sp_cd=int(sp_cd_t),
                        ncart_fn=ncart,
                    )
                out = _ext.eri_rys_deriv_contracted_cart_sp_batch_cy(
                    cache_cpu.ctx.shell_cxyz,
                    cache_cpu.ctx.shell_l,
                    cache_cpu.ctx.shell_prim_start,
                    cache_cpu.ctx.shell_nprim,
                    cache_cpu.ctx.prim_exp,
                    cache_cpu.ctx.sp_A,
                    cache_cpu.ctx.sp_B,
                    cache_cpu.ctx.sp_pair_start,
                    cache_cpu.ctx.sp_npair,
                    cache_cpu.ctx.pair_eta,
                    cache_cpu.ctx.pair_Px,
                    cache_cpu.ctx.pair_Py,
                    cache_cpu.ctx.pair_Pz,
                    cache_cpu.ctx.pair_cK,
                    int(sp_ab),
                    sp_cd,
                    np.asarray(tiles, dtype=np.float64, order="C"),
                    int(threads),
                )
                grad += accumulate_eri4c_task_derivs_to_atoms(
                    task_spAB=np.full((nt,), int(sp_ab), dtype=np.int32),
                    task_spCD=sp_cd,
                    shell_pairs=cache_cpu.sp,
                    shell_to_atom=cache_cpu.shell_atom,
                    task_derivs=np.asarray(out, dtype=np.float64),
                    natm=natm,
                )
    return np.asarray(grad, dtype=np.float64)


__all__ = [
    "build_t2ao_dense",
    "build_gtoc_dense_term_arrays",
    "contract_ordered_ao4_dense_deriv",
]
