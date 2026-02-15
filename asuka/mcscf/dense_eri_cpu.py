from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cueri.cart import ncart
from asuka.cueri.eri_utils import build_pair_coeff_ordered, build_pair_coeff_ordered_mixed
from asuka.cueri.pair_tables_cpu import PairTablesCPU, build_pair_tables_cpu
from asuka.cueri.shell_pairs import ShellPairs, build_shell_pairs_l_order
from asuka.integrals.eri4c_deriv_contracted import (
    ERI4cDerivContractionCPUContext,
    make_eri4c_deriv_contraction_cpu_context,
)
from asuka.integrals.int1e_cart import shell_to_atom_map


def _asnumpy_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _infer_nao_from_shells(*, shell_l: np.ndarray, shell_ao_start: np.ndarray) -> int:
    """Infer the total number of AO functions from shell angular momenta and start indices."""
    shell_l = np.asarray(shell_l, dtype=np.int32).ravel()
    shell_ao_start = np.asarray(shell_ao_start, dtype=np.int32).ravel()
    if shell_l.shape != shell_ao_start.shape:
        raise ValueError("shell_l and shell_ao_start must have identical shape")

    nao = 0
    for sh in range(int(shell_l.size)):
        nao = max(int(nao), int(shell_ao_start[sh]) + int(ncart(int(shell_l[sh]))))
    return int(nao)


def _require_eri_cpu_ext():
    try:
        from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU ERI extension is not built. Build it with:\n"
            "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
        ) from e
    return _ext


@dataclass(frozen=True)
class DenseERI4cDerivContractionCPUCache:
    """Reusable CPU cache for dense ERI tile evaluation and derivative contractions.

    Attributes
    ----------
    sp : ShellPairs
        Canonical shell pairs (l-ordered).
    pair_tables : PairTablesCPU
        Pair tables for screening.
    ctx : ERI4cDerivContractionCPUContext
        Context for derivative contraction/evaluation.
    shell_atom : np.ndarray
        Map from shell index to atom index.
    shell_l : np.ndarray
        Angular momentum for each shell.
    shell_ao_start : np.ndarray
        AO start index for each shell.
    sp_A : np.ndarray
        Shell A index for each shell pair.
    sp_B : np.ndarray
        Shell B index for each shell pair.
    spCD_by_key : dict[int, np.ndarray]
        Grouping of shell pairs by (lc, ld) key.

    Notes
    -----
    - Uses canonical shell pairs (l-ordered orientation) and CPU Rys tiles via
      `asuka.cueri._eri_rys_cpu`.
    - The `spCD_by_key` grouping matches the requirement of
      `eri_rys_tile_cart_sp_batch_cy`: all spCD tasks in a batch must share the
      same (lc,ld) class.
    """

    sp: ShellPairs
    pair_tables: PairTablesCPU
    ctx: ERI4cDerivContractionCPUContext
    shell_atom: np.ndarray  # (nShell,)

    shell_l: np.ndarray  # (nShell,)
    shell_ao_start: np.ndarray  # (nShell,)

    sp_A: np.ndarray  # (nSP,)
    sp_B: np.ndarray  # (nSP,)

    spCD_by_key: dict[int, np.ndarray]  # key=(lc<<8)|ld -> spCD indices


def build_dense_eri4c_deriv_contraction_cache_cpu(
    ao_basis: Any,
    *,
    atom_coords_bohr: np.ndarray,
    pair_table_threads: int = 0,
) -> DenseERI4cDerivContractionCPUCache:
    """Build shell-pair tables and groupings reused across many dense contractions (CPU).

    Parameters
    ----------
    ao_basis : Any
        The AO basis set object.
    atom_coords_bohr : np.ndarray
        Atomic coordinates in Bohr, shape (natm, 3).
    pair_table_threads : int, optional
        Number of threads for pair table construction. Defaults to 0.

    Returns
    -------
    DenseERI4cDerivContractionCPUCache
        The constructed cache object.
    """

    atom_coords_bohr = _asnumpy_f64(atom_coords_bohr)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")

    sp = build_shell_pairs_l_order(ao_basis)
    pt = build_pair_tables_cpu(ao_basis, sp, threads=int(pair_table_threads), profile=None)
    ctx = make_eri4c_deriv_contraction_cpu_context(ao_basis, sp, pt)
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords_bohr)

    shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
    shell_ao_start = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int32).ravel()

    sp_A = np.asarray(sp.sp_A, dtype=np.int32).ravel()
    sp_B = np.asarray(sp.sp_B, dtype=np.int32).ravel()

    key_cd = (shell_l[sp_A].astype(np.int32) << 8) | (shell_l[sp_B].astype(np.int32) & 0xFF)
    spCD_by_key: dict[int, np.ndarray] = {}
    for k in np.unique(key_cd).tolist():
        ki = int(k)
        spCD_by_key[ki] = np.nonzero(key_cd == ki)[0].astype(np.int32, copy=False)

    return DenseERI4cDerivContractionCPUCache(
        sp=sp,
        pair_tables=pt,
        ctx=ctx,
        shell_atom=np.asarray(shell_atom, dtype=np.int32),
        shell_l=shell_l,
        shell_ao_start=shell_ao_start,
        sp_A=sp_A,
        sp_B=sp_B,
        spCD_by_key=spCD_by_key,
    )


def dense_vhf_ao_from_tiles_cpu(
    ao_basis: Any,
    D_ao: np.ndarray,
    *,
    atom_coords_bohr: np.ndarray | None = None,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
) -> np.ndarray:
    """Return dense AO mean-field potential V = J(D) - 0.5 K(D) from CPU ERI tiles.

    This is the materialization-free analogue of `einsum` contraction against
    a full AO ERI tensor. Intended for tiny systems / validation; does not apply
    screening.

    Parameters
    ----------
    ao_basis : Any
        The AO basis set object.
    D_ao : np.ndarray
        AO density matrix, shape (nao, nao).
    atom_coords_bohr : np.ndarray | None, optional
        Atomic coordinates in Bohr. Required if cache_cpu is None.
    cache_cpu : DenseERI4cDerivContractionCPUCache | None, optional
        Precomputed cache object. If None, it will be built (requires atom_coords_bohr).
    pair_table_threads : int, optional
        Number of threads for pair table construction.
    max_tile_bytes : int, optional
        Max bytes per ERI tile.
    threads : int, optional
        Number of threads for ERI evaluation.

    Returns
    -------
    np.ndarray
        The computed potential matrix V, shape (nao, nao).
    """

    D = _asnumpy_f64(D_ao)
    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D_ao must be a square 2D matrix")
    nao = int(D.shape[0])
    D = 0.5 * (D + D.T)

    if cache_cpu is None:
        if atom_coords_bohr is None:
            raise ValueError("atom_coords_bohr is required when cache_cpu is None")
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
            pair_table_threads=int(pair_table_threads),
        )

    shell_l = cache_cpu.shell_l
    shell_ao_start = cache_cpu.shell_ao_start
    nao_cache = _infer_nao_from_shells(shell_l=shell_l, shell_ao_start=shell_ao_start)
    if int(nao_cache) != nao:
        raise ValueError(f"D_ao nao={nao} mismatch with ao_basis nao={nao_cache}")

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    _ext = _require_eri_cpu_ext()
    eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
    if eri_batch is None:  # pragma: no cover
        raise RuntimeError("asuka.cueri._eri_rys_cpu is missing eri_rys_tile_cart_sp_batch_cy; rebuild it")

    ctx = cache_cpu.ctx
    sp_A = cache_cpu.sp_A
    sp_B = cache_cpu.sp_B
    nsp = int(sp_A.size)

    # ---- Coulomb J(D) (s8 loop, symmetric updates like the original implementation) ----
    J = np.zeros((nao, nao), dtype=np.float64)

    def _add_block_sym(M: np.ndarray, rs: slice, cs: slice, block: np.ndarray) -> None:
        M[rs, cs] += block
        if (rs.start, rs.stop) != (cs.start, cs.stop):
            M[cs, rs] += block.T

    for spAB in range(nsp):
        A = int(sp_A[spAB])
        B = int(sp_B[spAB])
        la = int(shell_l[A])
        lb = int(shell_l[B])
        nA = int(ncart(la))
        nB = int(ncart(lb))
        nAB = int(nA * nB)
        aoA = int(shell_ao_start[A])
        aoB = int(shell_ao_start[B])

        # Fold D for the canonical AB pair for the (CD|AB) Coulomb update.
        D_AB = np.asarray(D[aoA : aoA + nA, aoB : aoB + nB], dtype=np.float64, order="C")
        if A != B:
            D_AB = D_AB + np.asarray(D[aoB : aoB + nB, aoA : aoA + nA], dtype=np.float64).T
        D_AB_flat = np.asarray(D_AB.reshape(nAB), dtype=np.float64, order="C")

        for spCD_all in cache_cpu.spCD_by_key.values():
            spCD_all = np.asarray(spCD_all, dtype=np.int32, order="C")
            if int(spCD_all.size) == 0:
                continue

            # Evaluate only spCD <= spAB to avoid double counting (AB|CD)=(CD|AB).
            spCD_all = spCD_all[spCD_all <= int(spAB)]
            if int(spCD_all.size) == 0:
                continue

            spCD0 = int(spCD_all[0])
            C0 = int(sp_A[spCD0])
            D0 = int(sp_B[spCD0])
            lc = int(shell_l[C0])
            ld = int(shell_l[D0])
            nC = int(ncart(lc))
            nD = int(ncart(ld))
            nCD = int(nC * nD)

            bytes_per_task = int(8 * nAB * nCD)
            chunk_nt = int(max(1, max_tile_bytes_i // max(bytes_per_task, 1)))

            for i0 in range(0, int(spCD_all.size), chunk_nt):
                i1 = min(int(spCD_all.size), i0 + chunk_nt)
                spCD = np.asarray(spCD_all[i0:i1], dtype=np.int32, order="C")
                nt = int(spCD.size)
                if nt == 0:
                    continue

                tile_batch = eri_batch(
                    ctx.shell_cxyz,
                    ctx.shell_l,
                    ctx.sp_A,
                    ctx.sp_B,
                    ctx.sp_pair_start,
                    ctx.sp_npair,
                    ctx.pair_eta,
                    ctx.pair_Px,
                    ctx.pair_Py,
                    ctx.pair_Pz,
                    ctx.pair_cK,
                    int(spAB),
                    spCD,
                    threads_i,
                )
                tile_batch = np.asarray(tile_batch, dtype=np.float64)
                if tile_batch.shape != (nt, nAB, nCD):  # pragma: no cover
                    raise RuntimeError("unexpected shape from eri_rys_tile_cart_sp_batch_cy")

                for t, spCD_t in enumerate(spCD.tolist()):
                    C = int(sp_A[int(spCD_t)])
                    Dsh = int(sp_B[int(spCD_t)])
                    aoC = int(shell_ao_start[C])
                    aoD = int(shell_ao_start[Dsh])

                    D_CD = np.asarray(D[aoC : aoC + nC, aoD : aoD + nD], dtype=np.float64, order="C")
                    if C != Dsh:
                        D_CD = D_CD + np.asarray(D[aoD : aoD + nD, aoC : aoC + nC], dtype=np.float64).T
                    D_CD_flat = np.asarray(D_CD.reshape(nCD), dtype=np.float64, order="C")

                    tile = np.asarray(tile_batch[t], dtype=np.float64, order="C")  # (nAB,nCD)

                    sA = slice(aoA, aoA + nA)
                    sB = slice(aoB, aoB + nB)
                    sC = slice(aoC, aoC + nC)
                    sD = slice(aoD, aoD + nD)

                    _add_block_sym(J, sA, sB, (tile @ D_CD_flat).reshape(nA, nB))
                    if int(spCD_t) != int(spAB):
                        _add_block_sym(J, sC, sD, (tile.T @ D_AB_flat).reshape(nC, nD))

    # ---- Exchange K(D) (s4 loop over all spAB,spCD) ----
    K = np.zeros((nao, nao), dtype=np.float64)
    for spAB in range(nsp):
        A = int(sp_A[spAB])
        B = int(sp_B[spAB])
        la = int(shell_l[A])
        lb = int(shell_l[B])
        nA = int(ncart(la))
        nB = int(ncart(lb))
        nAB = int(nA * nB)
        aoA = int(shell_ao_start[A])
        aoB = int(shell_ao_start[B])
        same_ab = bool(A == B)

        for spCD_all in cache_cpu.spCD_by_key.values():
            spCD_all = np.asarray(spCD_all, dtype=np.int32, order="C")
            if int(spCD_all.size) == 0:
                continue

            spCD0 = int(spCD_all[0])
            C0 = int(sp_A[spCD0])
            D0 = int(sp_B[spCD0])
            lc = int(shell_l[C0])
            ld = int(shell_l[D0])
            nC = int(ncart(lc))
            nD = int(ncart(ld))
            nCD = int(nC * nD)

            bytes_per_task = int(8 * nAB * nCD)
            chunk_nt = int(max(1, max_tile_bytes_i // max(bytes_per_task, 1)))

            for i0 in range(0, int(spCD_all.size), chunk_nt):
                i1 = min(int(spCD_all.size), i0 + chunk_nt)
                spCD = np.asarray(spCD_all[i0:i1], dtype=np.int32, order="C")
                nt = int(spCD.size)
                if nt == 0:
                    continue

                tile_batch = eri_batch(
                    ctx.shell_cxyz,
                    ctx.shell_l,
                    ctx.sp_A,
                    ctx.sp_B,
                    ctx.sp_pair_start,
                    ctx.sp_npair,
                    ctx.pair_eta,
                    ctx.pair_Px,
                    ctx.pair_Py,
                    ctx.pair_Pz,
                    ctx.pair_cK,
                    int(spAB),
                    spCD,
                    threads_i,
                )
                tile_batch = np.asarray(tile_batch, dtype=np.float64)
                if tile_batch.shape != (nt, nAB, nCD):  # pragma: no cover
                    raise RuntimeError("unexpected shape from eri_rys_tile_cart_sp_batch_cy")

                sA = slice(aoA, aoA + nA)
                sB = slice(aoB, aoB + nB)

                for t, spCD_t in enumerate(spCD.tolist()):
                    C = int(sp_A[int(spCD_t)])
                    Dsh = int(sp_B[int(spCD_t)])
                    aoC = int(shell_ao_start[C])
                    aoD = int(shell_ao_start[Dsh])
                    same_cd = bool(C == Dsh)

                    sC = slice(aoC, aoC + nC)
                    sD = slice(aoD, aoD + nD)

                    tile = np.asarray(tile_batch[t], dtype=np.float64, order="C")  # (nAB,nCD)
                    tile4 = tile.reshape(nA, nB, nC, nD)  # (μ,λ,ν,σ) in (A,B|C,D)

                    D_BD = np.asarray(D[aoB : aoB + nB, aoD : aoD + nD], dtype=np.float64, order="C")
                    K[sA, sC] += np.einsum("abcd,bd->ac", tile4, D_BD, optimize=True)

                    if not same_cd:
                        D_BC = np.asarray(D[aoB : aoB + nB, aoC : aoC + nC], dtype=np.float64, order="C")
                        K[sA, sD] += np.einsum("abcd,bc->ad", tile4, D_BC, optimize=True)

                    if not same_ab:
                        D_AD = np.asarray(D[aoA : aoA + nA, aoD : aoD + nD], dtype=np.float64, order="C")
                        K[sB, sC] += np.einsum("abcd,ad->bc", tile4, D_AD, optimize=True)
                        if not same_cd:
                            D_AC = np.asarray(D[aoA : aoA + nA, aoC : aoC + nC], dtype=np.float64, order="C")
                            K[sB, sD] += np.einsum("abcd,ac->bd", tile4, D_AC, optimize=True)

    K = 0.5 * (K + K.T)
    V = np.asarray(J - 0.5 * K, dtype=np.float64)
    return 0.5 * (V + V.T)


def dense_ppaa_papa_from_tiles_cpu(
    ao_basis: Any,
    mo_coeff: np.ndarray,
    *,
    ncore: int,
    ncas: int,
    atom_coords_bohr: np.ndarray | None = None,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense `ppaa` and `papa` tensors from CPU ERI tiles.

    Parameters
    ----------
    ao_basis : Any
        The AO basis set object.
    mo_coeff : np.ndarray
        Molecular orbital coefficients, shape (nao, nmo).
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    atom_coords_bohr : np.ndarray | None, optional
        Atomic coordinates.
    cache_cpu : DenseERI4cDerivContractionCPUCache | None, optional
        Precomputed cache.
    pair_table_threads : int, optional
        Number of threads for pair table construction.
    max_tile_bytes : int, optional
        Max tile size.
    threads : int, optional
        Number of threads for contraction.

    Returns
    -------
    ppaa : np.ndarray
        Array with shape `(nmo,nmo,ncas,ncas)`, with `ppaa[p,q,u,v] = (p q|u v)`.
    papa : np.ndarray
        Array with shape `(nmo,ncas,nmo,ncas)`, with `papa[p,u,q,v] = (p u|q v)`.
    """

    C = _asnumpy_f64(mo_coeff)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D (nao,nmo)")
    nao, nmo = map(int, C.shape)

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0:
        raise ValueError("invalid ncore/ncas")
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    if cache_cpu is None:
        if atom_coords_bohr is None:
            raise ValueError("atom_coords_bohr is required when cache_cpu is None")
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
            pair_table_threads=int(pair_table_threads),
        )

    shell_l = cache_cpu.shell_l
    shell_ao_start = cache_cpu.shell_ao_start
    nao_cache = _infer_nao_from_shells(shell_l=shell_l, shell_ao_start=shell_ao_start)
    if int(nao_cache) != nao:
        raise ValueError(f"mo_coeff nao={nao} mismatch with ao_basis nao={nao_cache}")

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    _ext = _require_eri_cpu_ext()
    eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
    if eri_batch is None:  # pragma: no cover
        raise RuntimeError("asuka.cueri._eri_rys_cpu is missing eri_rys_tile_cart_sp_batch_cy; rebuild it")

    ctx = cache_cpu.ctx
    sp_A = cache_cpu.sp_A
    sp_B = cache_cpu.sp_B
    nsp = int(sp_A.size)

    C = np.asarray(C, dtype=np.float64, order="C")
    C_act = np.asarray(C[:, ncore:nocc], dtype=np.float64, order="C")

    ppaa = np.zeros((nmo, nmo, ncas, ncas), dtype=np.float64)
    papa = np.zeros((nmo, ncas, nmo, ncas), dtype=np.float64)

    for spAB in range(nsp):
        A = int(sp_A[spAB])
        B = int(sp_B[spAB])
        la = int(shell_l[A])
        lb = int(shell_l[B])
        nA = int(ncart(la))
        nB = int(ncart(lb))
        nAB = int(nA * nB)
        aoA = int(shell_ao_start[A])
        aoB = int(shell_ao_start[B])

        same_ab = bool(A == B)

        CA_full = np.asarray(C[aoA : aoA + nA, :], dtype=np.float64, order="C")
        CB_full = np.asarray(C[aoB : aoB + nB, :], dtype=np.float64, order="C")
        CA_act = np.asarray(C_act[aoA : aoA + nA, :], dtype=np.float64, order="C")
        CB_act = np.asarray(C_act[aoB : aoB + nB, :], dtype=np.float64, order="C")

        K_AB_pq = build_pair_coeff_ordered(CA_full, CB_full, same_shell=same_ab)  # (nAB,nmo^2)
        K_AB_pu = build_pair_coeff_ordered_mixed(CA_full, CB_full, CA_act, CB_act, same_shell=same_ab)  # (nAB,nmo*ncas)

        for spCD_all in cache_cpu.spCD_by_key.values():
            spCD_all = np.asarray(spCD_all, dtype=np.int32, order="C")
            if int(spCD_all.size) == 0:
                continue

            spCD0 = int(spCD_all[0])
            C0 = int(sp_A[spCD0])
            D0 = int(sp_B[spCD0])
            lc = int(shell_l[C0])
            ld = int(shell_l[D0])
            nC = int(ncart(lc))
            nD = int(ncart(ld))
            nCD = int(nC * nD)

            bytes_per_task = int(8 * nAB * nCD)
            chunk_nt = int(max(1, max_tile_bytes_i // max(bytes_per_task, 1)))

            for i0 in range(0, int(spCD_all.size), chunk_nt):
                i1 = min(int(spCD_all.size), i0 + chunk_nt)
                spCD = np.asarray(spCD_all[i0:i1], dtype=np.int32, order="C")
                nt = int(spCD.size)
                if nt == 0:
                    continue

                tile_batch = eri_batch(
                    ctx.shell_cxyz,
                    ctx.shell_l,
                    ctx.sp_A,
                    ctx.sp_B,
                    ctx.sp_pair_start,
                    ctx.sp_npair,
                    ctx.pair_eta,
                    ctx.pair_Px,
                    ctx.pair_Py,
                    ctx.pair_Pz,
                    ctx.pair_cK,
                    int(spAB),
                    spCD,
                    threads_i,
                )
                tile_batch = np.asarray(tile_batch, dtype=np.float64)
                if tile_batch.shape != (nt, nAB, nCD):  # pragma: no cover
                    raise RuntimeError("unexpected shape from eri_rys_tile_cart_sp_batch_cy")

                for t, spCD_t in enumerate(spCD.tolist()):
                    Csh = int(sp_A[int(spCD_t)])
                    Dsh = int(sp_B[int(spCD_t)])
                    aoC = int(shell_ao_start[Csh])
                    aoD = int(shell_ao_start[Dsh])

                    same_cd = bool(Csh == Dsh)
                    CC_full = np.asarray(C[aoC : aoC + nC, :], dtype=np.float64, order="C")
                    CD_full = np.asarray(C[aoD : aoD + nD, :], dtype=np.float64, order="C")
                    CC_act = np.asarray(C_act[aoC : aoC + nC, :], dtype=np.float64, order="C")
                    CD_act = np.asarray(C_act[aoD : aoD + nD, :], dtype=np.float64, order="C")

                    K_CD_uv = build_pair_coeff_ordered(CC_act, CD_act, same_shell=same_cd)  # (nCD,ncas^2)
                    K_CD_qv = build_pair_coeff_ordered_mixed(
                        CC_full, CD_full, CC_act, CD_act, same_shell=same_cd
                    )  # (nCD,nmo*ncas)

                    tile = np.asarray(tile_batch[t], dtype=np.float64, order="C")  # (nAB,nCD)

                    tmp_ppaa = tile @ K_CD_uv  # (nAB,ncas^2)
                    ppaa += (K_AB_pq.T @ tmp_ppaa).reshape(nmo, nmo, ncas, ncas)

                    tmp_papa = tile @ K_CD_qv  # (nAB,nmo*ncas)
                    papa += (K_AB_pu.T @ tmp_papa).reshape(nmo, ncas, nmo, ncas)

    return np.asarray(ppaa, dtype=np.float64, order="C"), np.asarray(papa, dtype=np.float64, order="C")


__all__ = [
    "DenseERI4cDerivContractionCPUCache",
    "build_dense_eri4c_deriv_contraction_cache_cpu",
    "dense_ppaa_papa_from_tiles_cpu",
    "dense_vhf_ao_from_tiles_cpu",
]
