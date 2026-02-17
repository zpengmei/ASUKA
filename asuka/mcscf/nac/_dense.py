from __future__ import annotations

"""Dense-integral SA-CASSCF NAC (contracted 4c ERI derivatives).

This module mirrors the DF NAC flow but replaces DF 2e derivative pieces with
dense 4-center contracted derivative kernels.

Current scope:
- Reusable CPU cache for cuERI 4c derivative contractions.
- Hamiltonian-response terms without materializing derivative ERIs.
- Split-response (Lci + Lorb) with internal Newton-CASSCF operators.

This path is primarily intended for small/medium validation cases.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from asuka.cueri.cart import ncart
from asuka.cueri.eri_utils import build_pair_coeff_ordered
from asuka.integrals.eri4c_deriv_contracted import (
    ERI4cDerivContractionCUDAContext,
    make_eri4c_deriv_contraction_cuda_context,
)
from asuka.integrals.int1e_cart import (
    build_int1e_cart,
    contract_dS_ip_cart,
    contract_dhcore_cart,
)
from asuka.frontend.periodic_table import atomic_number
from asuka.mcscf.dense_eri_cpu import (
    DenseERI4cDerivContractionCPUCache,
    build_dense_eri4c_deriv_contraction_cache_cpu,
    dense_ppaa_papa_from_tiles_cpu,
    dense_vhf_ao_from_tiles_cpu,
)
from asuka.mcscf.state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from asuka.solver import GUGAFCISolver


def _asnumpy_f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _base_fcisolver_method(fcisolver: Any, name: str):
    """Return the (possibly unwrapped) fcisolver method ``name``.

    Matches the helper in :mod:`asuka.mcscf.nac_df`.
    """

    base_cls = getattr(fcisolver, "_base_class", None)
    if base_cls is not None and hasattr(base_cls, name):
        return getattr(base_cls, name)
    if not hasattr(fcisolver, name):
        raise AttributeError(f"fcisolver does not implement {name}")
    return getattr(fcisolver, name)


def _mol_coords_charges_bohr(mol: Any) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("mol.coords_bohr must have shape (natm,3)")
    charges = np.asarray([float(atomic_number(sym)) for sym in getattr(mol, "elements")], dtype=np.float64)
    if charges.shape != (int(coords.shape[0]),):
        raise ValueError("invalid mol.elements for charge construction")
    return coords, charges


@contextmanager
def _force_internal_newton():
    import os

    k_prefer = "CUGUGA_NEWTON_CASSCF"
    k_impl = "CUGUGA_NEWTON_CASSCF_IMPL"
    old_prefer = os.environ.get(k_prefer)
    old_impl = os.environ.get(k_impl)
    os.environ[k_prefer] = "internal"
    os.environ[k_impl] = "internal"
    try:
        yield
    finally:
        if old_prefer is None:
            os.environ.pop(k_prefer, None)
        else:
            os.environ[k_prefer] = old_prefer
        if old_impl is None:
            os.environ.pop(k_impl, None)
        else:
            os.environ[k_impl] = old_impl


class _FixedRDMFcisolver:
    """Delegate wrapper that overrides make_rdm* to return fixed transition densities."""

    def __init__(self, base: Any, dm1: np.ndarray, dm2: np.ndarray):
        self._base = base
        self._dm1 = np.asarray(dm1, dtype=np.float64)
        self._dm2 = np.asarray(dm2, dtype=np.float64)

    def make_rdm12(self, *_a: Any, **_k: Any):
        return self._dm1, self._dm2

    def make_rdm1(self, *_a: Any, **_k: Any):
        return self._dm1

    def make_rdm2(self, *_a: Any, **_k: Any):
        return self._dm2

    # Important: `newton_casscf` prefers `states_make_rdm12` when available. Many solvers
    # (including ASUKA's) implement it, so we must override it here to prevent the
    # fixed transition densities from being bypassed.
    def states_make_rdm12(self, ci_list: Sequence[Any], *_a: Any, **_k: Any):
        n = int(len(ci_list))
        dm1 = np.asarray(self._dm1, dtype=np.float64)
        dm2 = np.asarray(self._dm2, dtype=np.float64)
        return np.broadcast_to(dm1, (n,) + dm1.shape).copy(), np.broadcast_to(dm2, (n,) + dm2.shape).copy()

    def states_make_rdm1(self, ci_list: Sequence[Any], *_a: Any, **_k: Any):
        n = int(len(ci_list))
        dm1 = np.asarray(self._dm1, dtype=np.float64)
        return np.broadcast_to(dm1, (n,) + dm1.shape).copy()

    def states_make_rdm2(self, ci_list: Sequence[Any], *_a: Any, **_k: Any):
        n = int(len(ci_list))
        dm2 = np.asarray(self._dm2, dtype=np.float64)
        return np.broadcast_to(dm2, (n,) + dm2.shape).copy()

    def __getattr__(self, name: str):
        return getattr(self._base, name)


def _eris_patch_active(eris: Any, *, mo_coeff: np.ndarray, hcore_ao: np.ndarray, ncore: int):
    """Patch eris.vhf_c to remove core-orbital contributions (PySCF NAC convention)."""

    ncore = int(ncore)
    if ncore <= 0:
        return eris
    mo = np.asarray(mo_coeff, dtype=np.float64)
    moH = mo.T
    vnocore = np.asarray(getattr(eris, "vhf_c"), dtype=np.float64).copy()
    vnocore[:, :ncore] = -moH @ np.asarray(hcore_ao, dtype=np.float64) @ mo[:, :ncore]
    return type(eris)(ppaa=eris.ppaa, papa=eris.papa, vhf_c=vnocore, j_pc=eris.j_pc, k_pc=eris.k_pc)


def _symm_dm1(dm1: np.ndarray) -> np.ndarray:
    dm1 = np.asarray(dm1, dtype=np.float64)
    if dm1.ndim != 2 or dm1.shape[0] != dm1.shape[1]:
        raise ValueError("dm1 must be square (n,n)")
    return 0.5 * (dm1 + dm1.T)


def _symm_dm2_for_eri_contraction(dm2: np.ndarray) -> np.ndarray:
    """Symmetrize dm2 in the permutations that do not affect contraction with (uv|wx).

    This is safe for both expectation and transition 2-RDMs because (uv|wx) is
    symmetric under:
      u<->v, w<->x, and (uv)<->(wx).
    """

    dm2 = np.asarray(dm2, dtype=np.float64)
    if dm2.ndim != 4 or dm2.shape[0] != dm2.shape[1] or dm2.shape[2] != dm2.shape[3] or dm2.shape[0] != dm2.shape[2]:
        raise ValueError("dm2 must have shape (n,n,n,n)")

    dm2 = 0.25 * (
        dm2
        + dm2.transpose(1, 0, 2, 3)
        + dm2.transpose(0, 1, 3, 2)
        + dm2.transpose(1, 0, 3, 2)
    )
    dm2 = 0.5 * (dm2 + dm2.transpose(2, 3, 0, 1))
    return dm2


def _bar_tile_meanfield_jk(
    D_w: np.ndarray,
    D_core: np.ndarray,
    *,
    aoA: int,
    aoB: int,
    aoC: int,
    aoD: int,
    nA: int,
    nB: int,
    nC: int,
    nD: int,
    same_ab: bool,
    same_cd: bool,
) -> np.ndarray:
    """Return bar tile for Tr(D_w J(Dc)) - 0.5 Tr(D_w K(Dc)) in cuERI (AB|CD) layout."""

    # ---- Coulomb-like part: D_w[μν] * D_core[λσ] ----
    Dw_AB = np.asarray(D_w[aoA : aoA + nA, aoB : aoB + nB], dtype=np.float64, order="C")
    if not bool(same_ab):
        Dw_AB = Dw_AB + np.asarray(D_w[aoB : aoB + nB, aoA : aoA + nA], dtype=np.float64).T

    Dc_CD = np.asarray(D_core[aoC : aoC + nC, aoD : aoD + nD], dtype=np.float64, order="C")
    if not bool(same_cd):
        Dc_CD = Dc_CD + np.asarray(D_core[aoD : aoD + nD, aoC : aoC + nC], dtype=np.float64).T

    nAB = int(nA * nB)
    nCD = int(nC * nD)
    bar_J = Dw_AB.reshape(nAB, 1) * Dc_CD.reshape(1, nCD)

    # ---- Exchange-like part: -0.5 * D_w[μλ] * D_core[νσ] with pair folding ----
    Dw_AC = np.asarray(D_w[aoA : aoA + nA, aoC : aoC + nC], dtype=np.float64, order="C")
    Dc_BD = np.asarray(D_core[aoB : aoB + nB, aoD : aoD + nD], dtype=np.float64, order="C")
    t = Dw_AC[:, None, :, None] * Dc_BD[None, :, None, :]  # (nA,nB,nC,nD)

    if not bool(same_ab):
        Dw_BC = np.asarray(D_w[aoB : aoB + nB, aoC : aoC + nC], dtype=np.float64, order="C")
        Dc_AD = np.asarray(D_core[aoA : aoA + nA, aoD : aoD + nD], dtype=np.float64, order="C")
        t2 = Dw_BC[:, None, :, None] * Dc_AD[None, :, None, :]  # (nB,nA,nC,nD)
        t = t + t2.transpose(1, 0, 2, 3)

    if not bool(same_cd):
        Dw_AD = np.asarray(D_w[aoA : aoA + nA, aoD : aoD + nD], dtype=np.float64, order="C")
        Dc_BC = np.asarray(D_core[aoB : aoB + nB, aoC : aoC + nC], dtype=np.float64, order="C")
        t3 = Dw_AD[:, None, :, None] * Dc_BC[None, :, None, :]  # (nA,nB,nD,nC)
        t = t + t3.transpose(0, 1, 3, 2)

    if (not bool(same_ab)) and (not bool(same_cd)):
        Dw_BD = np.asarray(D_w[aoB : aoB + nB, aoD : aoD + nD], dtype=np.float64, order="C")
        Dc_AC = np.asarray(D_core[aoA : aoA + nA, aoC : aoC + nC], dtype=np.float64, order="C")
        t4 = Dw_BD[:, None, :, None] * Dc_AC[None, :, None, :]  # (nB,nA,nD,nC)
        t = t + t4.transpose(1, 0, 3, 2)

    bar_K = (-0.5) * t.reshape(nAB, nCD)

    return np.asarray(bar_J + bar_K, dtype=np.float64, order="C")


def _accum_eri4c_out_to_grad_dev(
    *,
    out_dev,
    task_spCD_dev,
    atomA: int,
    atomB: int,
    grad_dev,
    shell_atom_dev,
    sp_A_dev,
    sp_B_dev,
) -> None:
    """Accumulate contracted 4c derivative outputs into per-atom gradients on GPU."""

    import cupy as cp  # noqa: PLC0415

    grad_dev[int(atomA), :] += cp.sum(out_dev[:, 0, :], axis=0)
    grad_dev[int(atomB), :] += cp.sum(out_dev[:, 1, :], axis=0)

    shC = sp_A_dev[task_spCD_dev]
    shD = sp_B_dev[task_spCD_dev]
    atomC = shell_atom_dev[shC]
    atomD = shell_atom_dev[shD]

    valsC = out_dev[:, 2, :]
    valsD = out_dev[:, 3, :]

    cp.add.at(grad_dev[:, 0], atomC, valsC[:, 0])
    cp.add.at(grad_dev[:, 1], atomC, valsC[:, 1])
    cp.add.at(grad_dev[:, 2], atomC, valsC[:, 2])

    cp.add.at(grad_dev[:, 0], atomD, valsD[:, 0])
    cp.add.at(grad_dev[:, 1], atomD, valsD[:, 1])
    cp.add.at(grad_dev[:, 2], atomD, valsD[:, 2])


def _fill_eri_tile_symm_s1(
    eri_ao: np.ndarray,
    tile_abcd: np.ndarray,
    *,
    aoA: int,
    aoB: int,
    aoC: int,
    aoD: int,
) -> None:
    """Fill AO ERI tensor with all 8 s1 symmetries from one (A,B,C,D) tile.

    Parameters
    ----------
    eri_ao
        AO ERI tensor with shape (nao,nao,nao,nao).
    tile_abcd
        Tile shaped (nA,nB,nC,nD) for (A,B|C,D).
    aoA, aoB, aoC, aoD
        AO start indices for shells A,B,C,D.
    """

    tile = np.asarray(tile_abcd, dtype=np.float64, order="C")
    if tile.ndim != 4:
        raise ValueError("tile_abcd must have shape (nA,nB,nC,nD)")
    nA, nB, nC, nD = map(int, tile.shape)

    sA = slice(int(aoA), int(aoA) + nA)
    sB = slice(int(aoB), int(aoB) + nB)
    sC = slice(int(aoC), int(aoC) + nC)
    sD = slice(int(aoD), int(aoD) + nD)

    # (A,B|C,D) and swaps within each pair
    eri_ao[sA, sB, sC, sD] = tile
    eri_ao[sB, sA, sC, sD] = tile.transpose(1, 0, 2, 3)
    eri_ao[sA, sB, sD, sC] = tile.transpose(0, 1, 3, 2)
    eri_ao[sB, sA, sD, sC] = tile.transpose(1, 0, 3, 2)

    # Pair interchange: (C,D|A,B) and swaps
    t = tile.transpose(2, 3, 0, 1)
    eri_ao[sC, sD, sA, sB] = t
    eri_ao[sD, sC, sA, sB] = t.transpose(1, 0, 2, 3)
    eri_ao[sC, sD, sB, sA] = t.transpose(0, 1, 3, 2)
    eri_ao[sD, sC, sB, sA] = t.transpose(1, 0, 3, 2)


def _build_eri_ao_cart_dense_cpu(
    ao_basis,
    *,
    atom_coords_bohr: np.ndarray,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    max_eri_bytes: int = 512 << 20,
    threads: int = 0,
) -> np.ndarray:
    """Build full AO ERIs (cart) on CPU via cuERI tiles (s1 symmetry), for tiny systems.

    This helper builds dense-consistent mean-field JK potentials for NAC
    validation. It is **not** intended for production-scale
    workloads.
    """

    atom_coords_bohr = _asnumpy_f64(atom_coords_bohr)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            pair_table_threads=int(pair_table_threads),
        )

    try:
        from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU ERI extension is not built. Build it with:\n"
            "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
        ) from e

    eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
    if eri_batch is None:  # pragma: no cover
        raise RuntimeError("asuka.cueri._eri_rys_cpu is missing eri_rys_tile_cart_sp_batch_cy; rebuild it")

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")
    max_eri_bytes_i = int(max_eri_bytes)
    if max_eri_bytes_i <= 0:
        raise ValueError("max_eri_bytes must be > 0")
    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    shell_l = cache_cpu.shell_l
    shell_ao_start = cache_cpu.shell_ao_start
    n_shell = int(shell_l.size)
    if n_shell <= 0:
        return np.zeros((0, 0, 0, 0), dtype=np.float64)

    nao = 0
    for S in range(n_shell):
        nao = max(int(nao), int(shell_ao_start[S]) + int(ncart(int(shell_l[S]))))
    if nao <= 0:
        return np.zeros((0, 0, 0, 0), dtype=np.float64)

    need_bytes = int(nao) * int(nao) * int(nao) * int(nao) * 8
    if need_bytes > max_eri_bytes_i:
        raise NotImplementedError(
            f"AO ERI tensor too large for dense NAC path: nao={nao} requires {need_bytes/1024/1024:.1f} MiB, "
            f"max_eri_bytes={max_eri_bytes_i/1024/1024:.1f} MiB"
        )

    ctx = cache_cpu.ctx
    sp_A = cache_cpu.sp_A
    sp_B = cache_cpu.sp_B
    nsp = int(sp_A.size)

    eri_ao = np.zeros((nao, nao, nao, nao), dtype=np.float64)

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

        # Canonical between shell-pair indices: build only spCD <= spAB and fill the (CD|AB) half via symmetry.
        for spCD_all in cache_cpu.spCD_by_key.values():
            spCD_all = np.asarray(spCD_all, dtype=np.int32, order="C")
            if int(spCD_all.size) == 0:
                continue
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
                    D = int(sp_B[int(spCD_t)])
                    aoC = int(shell_ao_start[C])
                    aoD = int(shell_ao_start[D])
                    tile = tile_batch[t].reshape(nA, nB, nC, nD)
                    _fill_eri_tile_symm_s1(eri_ao, tile, aoA=aoA, aoB=aoB, aoC=aoC, aoD=aoD)

    return np.asarray(eri_ao, dtype=np.float64)


def _dense_vhf_ao_from_eri(eri_ao: np.ndarray, D_ao: np.ndarray) -> np.ndarray:
    """Return dense AO mean-field potential V = J(D) - 0.5 K(D) from full AO ERIs."""

    eri_ao = _asnumpy_f64(eri_ao)
    if eri_ao.ndim != 4 or eri_ao.shape[0] != eri_ao.shape[1] or eri_ao.shape[0] != eri_ao.shape[2] or eri_ao.shape[0] != eri_ao.shape[3]:
        raise ValueError("eri_ao must have shape (nao,nao,nao,nao)")
    nao = int(eri_ao.shape[0])

    D = _asnumpy_f64(D_ao)
    if D.shape != (nao, nao):
        raise ValueError("D_ao shape mismatch with eri_ao")
    D = 0.5 * (D + D.T)

    J = np.einsum("ls,mnls->mn", D, eri_ao, optimize=True)
    K = np.einsum("ls,mlns->mn", D, eri_ao, optimize=True)
    V = np.asarray(J - 0.5 * K, dtype=np.float64)
    return 0.5 * (V + V.T)


def _build_gfock_casscf_dense_cpu(
    ao_basis,
    *,
    atom_coords_bohr: np.ndarray,
    h_ao: np.ndarray,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ncore: int,
    ncas: int,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    g_dm2_eps_ao: float = 0.0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (gfock_mo, D_core_ao, D_act_ao, D_tot_ao, C_act, vhf_c_ao, vhf_ca_ao) for dense CASSCF."""

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0:
        raise ValueError("invalid ncore/ncas")

    h_ao = _asnumpy_f64(h_ao)
    if h_ao.ndim != 2 or h_ao.shape[0] != h_ao.shape[1]:
        raise ValueError("h_ao must be a square 2D matrix")
    nao = int(h_ao.shape[0])

    C = _asnumpy_f64(mo_coeff)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D (nao,nmo)")
    if int(C.shape[0]) != nao:
        raise ValueError("mo_coeff nao mismatch with h_ao")
    nmo = int(C.shape[1])
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    dm1_act = _symm_dm1(dm1_act)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")
    dm2_act = _symm_dm2_for_eri_contraction(dm2_act)
    if dm2_act.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    if ncore:
        D_core_ao = 2.0 * (C_core @ C_core.T)
    else:
        D_core_ao = np.zeros((nao, nao), dtype=np.float64)
    D_act_ao = C_act @ dm1_act @ C_act.T
    D_tot_ao = D_core_ao + D_act_ao

    vhf_c_ao = dense_vhf_ao_from_tiles_cpu(
        ao_basis,
        D_core_ao,
        atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )
    vhf_ca_ao = dense_vhf_ao_from_tiles_cpu(
        ao_basis,
        D_tot_ao,
        atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    h_mo = C.T @ h_ao @ C
    vhf_c_mo = C.T @ vhf_c_ao @ C
    vhf_ca_mo = C.T @ vhf_ca_ao @ C

    dm2_wxuv_flat = dm2_act.transpose(2, 3, 0, 1).reshape(ncas * ncas, ncas * ncas)
    dm2_wxuv_flat = 0.5 * (dm2_wxuv_flat + dm2_wxuv_flat.T)

    # Dense g_dm2 (exact 4c integrals) for the active 2-RDM contribution.
    from asuka.mcscf.orbital_grad import _g_dm2_dense_cpu  # noqa: PLC0415

    g_dm2 = _g_dm2_dense_cpu(
        ao_basis,
        C_mo=C,
        C_act=C_act,
        dm2_wxuv_flat=dm2_wxuv_flat,
        eps_ao=float(g_dm2_eps_ao),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
        blas_nthreads=None,
        p_block_nmo=64,
        profile=None,
    )

    gfock = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore:
        gfock[:, :ncore] = 2.0 * (h_mo + vhf_ca_mo)[:, :ncore]
    gfock[:, ncore:nocc] = (h_mo + vhf_c_mo)[:, ncore:nocc] @ dm1_act + np.asarray(g_dm2, dtype=np.float64)

    return (
        np.asarray(gfock, dtype=np.float64),
        np.asarray(D_core_ao, dtype=np.float64),
        np.asarray(D_act_ao, dtype=np.float64),
        np.asarray(D_tot_ao, dtype=np.float64),
        np.asarray(C_act, dtype=np.float64),
        np.asarray(vhf_c_ao, dtype=np.float64),
        np.asarray(vhf_ca_ao, dtype=np.float64),
    )


def _core_energy_weighted_density_dense(
    *,
    mo_coeff: np.ndarray,
    h_ao: np.ndarray,
    vhf_c_ao: np.ndarray,
    ncore: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (D_core_ao, dme_sf_core) for the core-only RHF-like term (dense-consistent)."""

    mo = _asnumpy_f64(mo_coeff)
    if mo.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nao, nmo = map(int, mo.shape)
    ncore = int(ncore)
    if ncore < 0 or ncore > nmo:
        raise ValueError("invalid ncore")

    h_ao = _asnumpy_f64(h_ao)
    vhf_c_ao = _asnumpy_f64(vhf_c_ao)
    if h_ao.shape != (nao, nao) or vhf_c_ao.shape != (nao, nao):
        raise ValueError("h_ao/vhf_c_ao shape mismatch with mo_coeff")

    if ncore:
        mo_core = mo[:, :ncore]
        D_core_ao = 2.0 * (mo_core @ mo_core.T)
        v_ao = vhf_c_ao
    else:
        D_core_ao = np.zeros((nao, nao), dtype=np.float64)
        v_ao = np.zeros((nao, nao), dtype=np.float64)

    f0 = mo.T @ h_ao @ mo
    if ncore:
        f0 = f0 + (mo.T @ v_ao @ mo)

    mo_occ = np.zeros((nmo,), dtype=np.float64)
    mo_occ[:ncore] = 2.0
    f0_occ = f0 * mo_occ[None, :]
    dme_sf = mo @ ((f0_occ + f0_occ.T) * 0.5) @ mo.T
    return np.asarray(D_core_ao, dtype=np.float64), np.asarray(dme_sf, dtype=np.float64)


def _grad_elec_active_dense(
    *,
    ao_basis,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    h_ao: np.ndarray | None = None,
    eri_ao: np.ndarray | None = None,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ncore: int,
    ncas: int,
    backend: Literal["cpu", "cuda"] = "cpu",
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    max_eri_bytes: int = 512 << 20,
    threads: int = 0,
) -> np.ndarray:
    """Return active-electron part of <dH/dR> using dense contracted 4c derivatives."""

    atom_coords_bohr = _asnumpy_f64(atom_coords_bohr)
    atom_charges = _asnumpy_f64(atom_charges).ravel()
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")
    natm = int(atom_coords_bohr.shape[0])
    if atom_charges.shape != (natm,):
        raise ValueError("atom_charges must have shape (natm,)")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            pair_table_threads=int(pair_table_threads),
        )

    if h_ao is None:
        int1e = build_int1e_cart(ao_basis, atom_coords_bohr=atom_coords_bohr, atom_charges=atom_charges)
        h_ao_use = np.asarray(int1e.hcore, dtype=np.float64)
    else:
        h_ao_use = np.asarray(h_ao, dtype=np.float64)

    # `eri_ao`/`max_eri_bytes` are kept for backward compatibility with an older
    # implementation that materialized the full AO ERI tensor. The current dense
    # NAC backend builds mean-field terms directly from ERI tiles.
    _ = eri_ao
    _ = max_eri_bytes

    gfock, D_core_ao, _D_act_ao, D_tot_ao, _C_act, vhf_c_ao, _vhf_ca_ao = _build_gfock_casscf_dense_cpu(
        ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        h_ao=h_ao_use,
        mo_coeff=mo_coeff,
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        ncore=int(ncore),
        ncas=int(ncas),
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        g_dm2_eps_ao=0.0,
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    shell_atom = cache_cpu.shell_atom
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        atom_charges=atom_charges,
        M=D_tot_ao,
        shell_atom=shell_atom,
    )

    de_2e = grad_2e_ham_dense_eri4c_contracted(
        ao_basis=ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        mo_coeff=mo_coeff,
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        ncore=int(ncore),
        ncas=int(ncas),
        backend=backend,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    # Core-only RHF-like subtraction.
    D_core_only, _dme_core = _core_energy_weighted_density_dense(
        mo_coeff=mo_coeff,
        h_ao=h_ao_use,
        vhf_c_ao=vhf_c_ao,
        ncore=int(ncore),
    )
    de_h1_core = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        atom_charges=atom_charges,
        M=D_core_only,
        shell_atom=shell_atom,
    )
    zero_dm1 = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
    zero_dm2 = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
    de_2e_core = grad_2e_ham_dense_eri4c_contracted(
        ao_basis=ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        mo_coeff=mo_coeff,
        dm1_act=zero_dm1,
        dm2_act=zero_dm2,
        ncore=int(ncore),
        ncas=int(ncas),
        backend=backend,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    return np.asarray(de_h1 + de_2e - de_h1_core - de_2e_core, dtype=np.float64)


def _dense_ppaa_papa_from_eri_ao(
    eri_ao: np.ndarray,
    mo_coeff: np.ndarray,
    *,
    ncore: int,
    ncas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build dense ppaa/papa tensors from full AO ERIs (tiny-system helper).

    Returns
    -------
    ppaa
        Array with shape ``(nmo,nmo,ncas,ncas)``, with ``ppaa[p,q,u,v] = (p q|u v)``.
    papa
        Array with shape ``(nmo,ncas,nmo,ncas)``, with ``papa[p,u,q,v] = (p u|q v)``.
    """

    eri_ao = _asnumpy_f64(eri_ao)
    if eri_ao.ndim != 4 or eri_ao.shape[0] != eri_ao.shape[1] or eri_ao.shape[0] != eri_ao.shape[2] or eri_ao.shape[0] != eri_ao.shape[3]:
        raise ValueError("eri_ao must have shape (nao,nao,nao,nao)")
    nao = int(eri_ao.shape[0])

    C = _asnumpy_f64(mo_coeff)
    if C.ndim != 2 or int(C.shape[0]) != nao:
        raise ValueError("mo_coeff shape mismatch with eri_ao")
    nmo = int(C.shape[1])

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0 or ncore + ncas > nmo:
        raise ValueError("invalid ncore/ncas for mo_coeff")
    nocc = ncore + ncas

    C_act = np.asarray(C[:, ncore:nocc], dtype=np.float64, order="C")

    # ppaa[p,q,u,v] = (p q|u v) where u,v are active indices.
    tmp = np.einsum("mnls,sv->mnlv", eri_ao, C_act, optimize=True)
    tmp2 = np.einsum("mnlv,lu->mnuv", tmp, C_act, optimize=True)
    tmp3 = np.einsum("mnuv,nq->mquv", tmp2, C, optimize=True)
    ppaa = np.einsum("mquv,mp->pquv", tmp3, C, optimize=True)
    ppaa = np.asarray(ppaa, dtype=np.float64, order="C")

    # papa[p,u,q,v] = (p u|q v) where u,v are active indices.
    tmp = np.einsum("mnls,sv->mnlv", eri_ao, C_act, optimize=True)
    tmp2 = np.einsum("mnlv,lq->mnqv", tmp, C, optimize=True)
    tmp3 = np.einsum("mnqv,nu->muqv", tmp2, C_act, optimize=True)
    papa = np.einsum("muqv,mp->puqv", tmp3, C, optimize=True)
    papa = np.asarray(papa, dtype=np.float64, order="C")

    return ppaa, papa


def _Lorb_dot_dgorb_dx_dense(
    *,
    ao_basis,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    Lorb: np.ndarray,
    ncore: int,
    ncas: int,
    backend: Literal["cpu", "cuda"] = "cpu",
    h_ao: np.ndarray | None = None,
    eri_ao: np.ndarray | None = None,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    max_eri_bytes: int = 512 << 20,
    threads: int = 0,
) -> np.ndarray:
    """Dense orbital Lagrange term nuclear derivative (PySCF `Lorb_dot_dgorb_dx` analogue)."""

    atom_coords_bohr = _asnumpy_f64(atom_coords_bohr)
    atom_charges = _asnumpy_f64(atom_charges).ravel()
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")
    natm = int(atom_coords_bohr.shape[0])
    if atom_charges.shape != (natm,):
        raise ValueError("atom_charges must have shape (natm,)")

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0:
        raise ValueError("invalid ncore/ncas")

    C = _asnumpy_f64(mo_coeff)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D (nao,nmo)")
    nao, nmo = map(int, C.shape)
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    L = _asnumpy_f64(Lorb)
    if L.shape != (nmo, nmo):
        raise ValueError("Lorb shape mismatch with mo_coeff")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            pair_table_threads=int(pair_table_threads),
        )

    if h_ao is None:
        int1e = build_int1e_cart(ao_basis, atom_coords_bohr=atom_coords_bohr, atom_charges=atom_charges)
        h_ao_use = np.asarray(int1e.hcore, dtype=np.float64)
    else:
        h_ao_use = np.asarray(h_ao, dtype=np.float64)
    if h_ao_use.shape != (nao, nao):
        raise ValueError("h_ao shape mismatch with mo_coeff")

    # `eri_ao`/`max_eri_bytes` are kept for backward compatibility with an older
    # implementation that materialized AO ERIs. The current implementation is
    # materialization-free and builds all mean-field / ppaa/papa terms from tiles.
    _ = eri_ao
    _ = max_eri_bytes

    dm1_act = _symm_dm1(dm1_act)
    dm2_act = _symm_dm2_for_eri_contraction(dm2_act)

    C_core = np.asarray(C[:, :ncore], dtype=np.float64, order="C")
    C_act = np.asarray(C[:, ncore:nocc], dtype=np.float64, order="C")

    C_L = np.asarray(C @ L, dtype=np.float64, order="C")
    C_L_core = np.asarray(C_L[:, :ncore], dtype=np.float64, order="C")
    C_L_act = np.asarray(C_L[:, ncore:nocc], dtype=np.float64, order="C")

    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
        D_L_core = 2.0 * (C_L_core @ C_core.T)
        D_L_core = D_L_core + D_L_core.T
    else:
        D_core = np.zeros((nao, nao), dtype=np.float64)
        D_L_core = np.zeros((nao, nao), dtype=np.float64)

    D_act = C_act @ dm1_act @ C_act.T
    D_L_act = C_L_act @ dm1_act @ C_act.T
    D_L_act = D_L_act + D_L_act.T

    D_tot = D_core + D_act
    D_L = D_L_core + D_L_act

    # Mean-field potentials from dense ERI tiles.
    if ncore:
        vhf_c = dense_vhf_ao_from_tiles_cpu(
            ao_basis,
            D_core,
            atom_coords_bohr=atom_coords_bohr,
            cache_cpu=cache_cpu,
            pair_table_threads=int(pair_table_threads),
            max_tile_bytes=int(max_tile_bytes),
            threads=int(threads),
        )
        vhfL_c = dense_vhf_ao_from_tiles_cpu(
            ao_basis,
            D_L_core,
            atom_coords_bohr=atom_coords_bohr,
            cache_cpu=cache_cpu,
            pair_table_threads=int(pair_table_threads),
            max_tile_bytes=int(max_tile_bytes),
            threads=int(threads),
        )
    else:
        vhf_c = np.zeros((nao, nao), dtype=np.float64)
        vhfL_c = np.zeros((nao, nao), dtype=np.float64)

    vhf_a = dense_vhf_ao_from_tiles_cpu(
        ao_basis,
        D_act,
        atom_coords_bohr=atom_coords_bohr,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )
    vhfL_a = dense_vhf_ao_from_tiles_cpu(
        ao_basis,
        D_L_act,
        atom_coords_bohr=atom_coords_bohr,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    gfock = np.asarray(h_ao_use @ D_L, dtype=np.float64)
    gfock += np.asarray((vhf_c + vhf_a) @ D_L_core, dtype=np.float64)
    gfock += np.asarray((vhfL_c + vhfL_a) @ D_core, dtype=np.float64)
    gfock += np.asarray(vhfL_c @ D_act, dtype=np.float64)
    gfock += np.asarray(vhf_c @ D_L_act, dtype=np.float64)

    # Convert AO->(MO definition) by left-multiplying S^{-1} ≈ C C^T (PySCF convention).
    s0_inv = np.asarray(C @ C.T, dtype=np.float64)
    gfock = np.asarray(s0_inv @ gfock, dtype=np.float64)

    # Active-active part mirroring PySCF (aapa/aapaL contraction).
    ppaa, papa = dense_ppaa_papa_from_tiles_cpu(
        ao_basis,
        C,
        ncore=int(ncore),
        ncas=int(ncas),
        atom_coords_bohr=atom_coords_bohr,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )
    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=np.float64)
    aapaL = np.zeros_like(aapa)
    for i in range(nmo):
        jbuf = np.asarray(ppaa[i], dtype=np.float64)  # (nmo,ncas,ncas)
        kbuf = np.asarray(papa[i], dtype=np.float64)  # (ncas,nmo,ncas)
        aapa[:, :, i, :] = np.asarray(jbuf[ncore:nocc, :, :], dtype=np.float64).transpose(1, 2, 0)
        aapaL[:, :, i, :] += np.tensordot(jbuf, L[:, ncore:nocc], axes=((0), (0)))
        kk = np.tensordot(kbuf, L[:, ncore:nocc], axes=((1), (0))).transpose(1, 2, 0)
        aapaL[:, :, i, :] += kk + kk.transpose(1, 0, 2)

    t1 = np.einsum("uviw,uvtw->it", aapaL, dm2_act, optimize=True)
    t2 = np.einsum("uviw,vuwt->it", aapa, dm2_act, optimize=True)
    gfock += (C @ np.asarray(t1, dtype=np.float64) @ C_act.T)
    gfock += (C @ np.asarray(t2, dtype=np.float64) @ C_L_act.T)

    shell_atom = cache_cpu.shell_atom
    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        atom_charges=atom_charges,
        M=D_L,
        shell_atom=shell_atom,
    )

    de_2e = grad_2e_lorb_dense_eri4c_contracted(
        ao_basis=ao_basis,
        atom_coords_bohr=atom_coords_bohr,
        mo_coeff=C,
        dm1_act=dm1_act,
        dm2_act=dm2_act,
        Lorb=L,
        ncore=int(ncore),
        ncas=int(ncas),
        backend=backend,
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    return np.asarray(de_h1 + de_2e, dtype=np.float64)

@dataclass(frozen=True)
class DenseNACCPUCache:
    """Reusable preprocessing cache for dense NAC (CPU).

    Notes
    -----
    - This cache does **not** materialize AO ERIs. Mean-field and Newton-CASSCF
      intermediates are built directly from cuERI tiles.
    """

    ao_basis: Any
    atom_coords_bohr: np.ndarray
    atom_charges: np.ndarray
    h_ao: np.ndarray
    eri4c_cache: DenseERI4cDerivContractionCPUCache


def build_dense_nac_cache_cpu(
    scf_out: Any,
    *,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    max_eri_bytes: int = 512 << 20,
    threads: int = 0,
) -> DenseNACCPUCache:
    """Build a reusable cache for dense NAC on CPU."""

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("dense NAC currently requires cart=True")

    coords, charges = _mol_coords_charges_bohr(mol)
    ao_basis = getattr(scf_out, "ao_basis", None)
    if ao_basis is None:
        raise TypeError("scf_out must have .ao_basis")

    cache_eri4c = build_dense_eri4c_deriv_contraction_cache_cpu(
        ao_basis,
        atom_coords_bohr=coords,
        pair_table_threads=int(pair_table_threads),
    )

    h_ao = None
    int1e = getattr(scf_out, "int1e", None)
    if int1e is not None:
        h_ao = getattr(int1e, "hcore", None)
    if h_ao is None:
        h_ao = build_int1e_cart(ao_basis, atom_coords_bohr=coords, atom_charges=charges).hcore

    _ = max_tile_bytes
    _ = max_eri_bytes
    _ = threads

    return DenseNACCPUCache(
        ao_basis=ao_basis,
        atom_coords_bohr=np.asarray(coords, dtype=np.float64),
        atom_charges=np.asarray(charges, dtype=np.float64),
        h_ao=np.asarray(h_ao, dtype=np.float64),
        eri4c_cache=cache_eri4c,
    )


def sacasscf_nonadiabatic_couplings_dense(
    scf_out: Any,
    casscf: Any,
    *,
    pairs: Sequence[tuple[int, int]] | None = None,
    atmlst: Sequence[int] | None = None,
    use_etfs: bool = False,
    mult_ediff: bool = False,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    backend: Literal["cpu", "cuda"] = "cpu",
    response_term: Literal["none", "split_orbfd"] = "none",
    cache_cpu: DenseNACCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    max_eri_bytes: int = 512 << 20,
    threads: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
) -> np.ndarray:
    """Compute SA-CASSCF NACVs using dense contracted 4c derivatives.

    Includes:
    - Hamiltonian-response term (<bra|dH/dR|ket>) using dense contracted 4c derivatives
    - Explicit CSF overlap term (when ``use_etfs=False``)
    - Split-response (Z-vector) term (when ``response_term='split_orbfd'``)
    """

    backend_s = str(backend).strip().lower()
    if backend_s not in ("cpu", "cuda"):
        raise NotImplementedError("dense NAC currently supports backend='cpu' or backend='cuda' only")

    response = str(response_term).strip().lower()
    if response not in ("none", "split_orbfd"):
        raise NotImplementedError("dense NAC response_term must be 'none' or 'split_orbfd'")

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have .mol")
    if not bool(getattr(mol, "cart", False)):
        raise NotImplementedError("dense NAC currently requires cart=True")

    if cache_cpu is None:
        cache_cpu = build_dense_nac_cache_cpu(
            scf_out,
            pair_table_threads=int(pair_table_threads),
            max_tile_bytes=int(max_tile_bytes),
            max_eri_bytes=int(max_eri_bytes),
            threads=int(threads),
        )

    coords = np.asarray(cache_cpu.atom_coords_bohr, dtype=np.float64)
    charges = np.asarray(cache_cpu.atom_charges, dtype=np.float64)
    natm = int(coords.shape[0])

    if atmlst is None:
        atmlst_use = list(range(natm))
    else:
        atmlst_use = [int(a) for a in atmlst]

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    nocc = ncore + ncas

    ci_raw = getattr(casscf, "ci", None)
    nroots = int(getattr(casscf, "nroots", len(ci_raw) if isinstance(ci_raw, (list, tuple)) else 1))
    ci_list = ci_as_list(ci_raw, nroots=nroots)
    if nroots <= 1:
        return np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    e_raw = getattr(casscf, "e_states", None)
    if e_raw is None:
        e_raw = getattr(casscf, "e_roots", None)
    if e_raw is None:
        raise ValueError("casscf must provide per-root energies as e_states or e_roots")
    e_states = np.asarray(e_raw, dtype=np.float64).ravel()
    if int(e_states.size) != nroots:
        raise ValueError("energy array length mismatch")

    weights_in = getattr(casscf, "root_weights", None)
    if weights_in is None:
        weights_in = getattr(casscf, "weights", None)
    weights = normalize_weights(weights_in, nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    C_ref = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    if C_ref.ndim != 2:
        raise ValueError("casscf.mo_coeff must be 2D (nao,nmo)")
    nao, nmo = map(int, C_ref.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    # Output tensor
    nac = np.zeros((nroots, nroots, len(atmlst_use), 3), dtype=np.float64)

    def _unpack_state(state: tuple[int, int]) -> tuple[int, int]:
        ket, bra = state
        ket = int(ket)
        bra = int(bra)
        if ket < 0 or bra < 0 or ket >= nroots or bra >= nroots:
            raise ValueError("state indices out of range")
        return ket, bra

    pair_list: list[tuple[int, int]]
    if pairs is None:
        pair_list = [(ket, bra) for ket in range(nroots) for bra in range(nroots) if ket != bra]
    else:
        pair_list = [(int(ket), int(bra)) for (ket, bra) in pairs if int(ket) != int(bra)]

    ao_basis_ref = cache_cpu.ao_basis
    shell_atom = np.asarray(cache_cpu.eri4c_cache.shell_atom, dtype=np.int32)

    trans_rdm12 = _base_fcisolver_method(fcisolver_use, "trans_rdm12")
    trans_rdm1 = _base_fcisolver_method(fcisolver_use, "trans_rdm1")

    # Z-vector machinery (split response).
    mc_sa = None
    eris_sa = None
    hess_op = None
    dm1_sa = None
    dm2_sa = None
    if response == "split_orbfd":
        from asuka.mcscf.newton_dense import DenseNewtonCASSCFAdapter  # noqa: PLC0415
        from asuka.mcscf import newton_casscf as _newton_casscf  # noqa: PLC0415
        from asuka.mcscf.zvector import build_mcscf_hessian_operator  # noqa: PLC0415

        mc_sa = DenseNewtonCASSCFAdapter(
            ao_basis=ao_basis_ref,
            atom_coords_bohr=np.asarray(coords, dtype=np.float64),
            hcore_ao=np.asarray(cache_cpu.h_ao, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            nelecas=nelecas,
            mo_coeff=C_ref,
            fcisolver=fcisolver_use,
            eri_cache_cpu=cache_cpu.eri4c_cache,
            pair_table_threads=int(pair_table_threads),
            max_tile_bytes=int(max_tile_bytes),
            eri_threads=int(threads),
            weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
            frozen=getattr(casscf, "frozen", None),
            internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
            extrasym=getattr(casscf, "extrasym", None),
        )
        eris_sa = mc_sa.ao2mo(C_ref)
        with _force_internal_newton():
            hess_op = build_mcscf_hessian_operator(
                mc_sa,
                mo_coeff=C_ref,
                ci=ci_list,
                eris=eris_sa,
                use_newton_hessian=True,
            )

        dm1_sa, dm2_sa = make_state_averaged_rdms(
            fcisolver_use,
            ci_list,
            weights,
            ncas=int(ncas),
            nelecas=nelecas,
        )

    for ket, bra in pair_list:
        ket, bra = _unpack_state((ket, bra))
        if ket == bra:
            continue

        ediff = float(e_states[bra] - e_states[ket])

        dm1_t, dm2_t = trans_rdm12(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
        dm1_t = 0.5 * (np.asarray(dm1_t, dtype=np.float64) + np.asarray(dm1_t, dtype=np.float64).T)
        dm2_t = np.asarray(dm2_t, dtype=np.float64)

        ham = _grad_elec_active_dense(
            ao_basis=ao_basis_ref,
            atom_coords_bohr=coords,
            atom_charges=charges,
            h_ao=cache_cpu.h_ao,
            mo_coeff=C_ref,
            dm1_act=dm1_t,
            dm2_act=dm2_t,
            ncore=int(ncore),
            ncas=int(ncas),
            backend=backend_s,
            cache_cpu=cache_cpu.eri4c_cache,
            pair_table_threads=int(pair_table_threads),
            max_tile_bytes=int(max_tile_bytes),
            max_eri_bytes=int(max_eri_bytes),
            threads=int(threads),
        )

        if not bool(use_etfs):
            dm1 = trans_rdm1(fcisolver_use, ci_list[bra], ci_list[ket], int(ncas), nelecas)
            castm1 = np.asarray(dm1, dtype=np.float64).T - np.asarray(dm1, dtype=np.float64)
            mo_cas = C_ref[:, ncore:nocc]
            tm1 = mo_cas @ castm1 @ mo_cas.T
            nac_csf = 0.5 * contract_dS_ip_cart(
                ao_basis_ref,
                atom_coords_bohr=coords,
                M=tm1,
                shell_atom=shell_atom,
            )
            ham = np.asarray(ham, dtype=np.float64) + np.asarray(nac_csf * ediff, dtype=np.float64)

        resp_full = np.zeros((natm, 3), dtype=np.float64)
        if response == "split_orbfd":
            if mc_sa is None or eris_sa is None or hess_op is None:  # pragma: no cover
                raise RuntimeError("internal error: missing SA Newton/Hessian objects for response")

            from asuka.mcscf.newton_dense import DenseNewtonCASSCFAdapter  # noqa: PLC0415
            from asuka.mcscf import newton_casscf as _newton_casscf  # noqa: PLC0415
            from asuka.mcscf.zvector import solve_mcscf_zvector  # noqa: PLC0415

            # Pair-specific Z-vector RHS in SA parameter space.
            fcisolver_fixed = _FixedRDMFcisolver(fcisolver_use, dm1=dm1_t, dm2=dm2_t)
            mc_trans = DenseNewtonCASSCFAdapter(
                ao_basis=ao_basis_ref,
                atom_coords_bohr=np.asarray(coords, dtype=np.float64),
                hcore_ao=np.asarray(cache_cpu.h_ao, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
                nelecas=nelecas,
                mo_coeff=C_ref,
                fcisolver=fcisolver_fixed,
                eri_cache_cpu=cache_cpu.eri4c_cache,
                pair_table_threads=int(pair_table_threads),
                max_tile_bytes=int(max_tile_bytes),
                eri_threads=int(threads),
                frozen=getattr(casscf, "frozen", None),
                internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
                extrasym=getattr(casscf, "extrasym", None),
            )
            eris_act = _eris_patch_active(eris_sa, mo_coeff=C_ref, hcore_ao=cache_cpu.h_ao, ncore=int(ncore))

            g_ket, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                mc_trans,
                C_ref,
                ci_list[ket],
                eris_act,
                verbose=0,
                implementation="internal",
            )
            g_bra, _gupd, _hop, _hdiag = _newton_casscf.gen_g_hop(
                mc_trans,
                C_ref,
                ci_list[bra],
                eris_act,
                verbose=0,
                implementation="internal",
            )
            g_ket = np.asarray(g_ket, dtype=np.float64).ravel()
            g_bra = np.asarray(g_bra, dtype=np.float64).ravel()

            n_orb = int(hess_op.n_orb)
            g_orb = g_ket[:n_orb]

            g_ci_bra = 0.5 * g_ket[n_orb:].copy()
            g_ci_ket = 0.5 * g_bra[n_orb:].copy()

            ndet_ket = int(np.asarray(ci_list[ket]).size)
            ndet_bra = int(np.asarray(ci_list[bra]).size)
            if ndet_ket == ndet_bra:
                ket2bra = float(np.dot(np.asarray(ci_list[bra], dtype=np.float64).ravel(), g_ci_ket))
                bra2ket = float(np.dot(np.asarray(ci_list[ket], dtype=np.float64).ravel(), g_ci_bra))
                g_ci_ket = g_ci_ket - ket2bra * np.asarray(ci_list[bra], dtype=np.float64).ravel()
                g_ci_bra = g_ci_bra - bra2ket * np.asarray(ci_list[ket], dtype=np.float64).ravel()

            rhs_ci_list: list[np.ndarray] = []
            for r in range(nroots):
                rhs_ci_list.append(np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()))
            rhs_ci_list[ket] = g_ci_ket[:ndet_ket]
            rhs_ci_list[bra] = g_ci_bra[:ndet_bra]

            z = solve_mcscf_zvector(
                mc_sa,
                rhs_orb=np.asarray(g_orb, dtype=np.float64),
                rhs_ci=rhs_ci_list,
                hessian_op=hess_op,
                tol=float(z_tol),
                maxiter=int(z_maxiter),
            )
            Lvec = np.asarray(z.z_packed, dtype=np.float64).ravel()
            if int(Lvec.size) != int(hess_op.n_tot):  # pragma: no cover
                raise RuntimeError("unexpected Z-vector packed length")

            Lorb_mat = mc_sa.unpack_uniq_var(Lvec[:n_orb])
            Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])
            if not isinstance(Lci_list, list) or len(Lci_list) != int(nroots):  # pragma: no cover
                raise RuntimeError("unexpected CI unpack structure in Z-vector solution")

            # CI response: build weighted transition RDMs between Lci[root] and ci[root], then reuse dense gradient.
            dm1_lci = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
            dm2_lci = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
            w_arr = np.asarray(weights, dtype=np.float64).ravel()
            for r in range(int(nroots)):
                wr = float(w_arr[r])
                if abs(wr) < 1e-14:
                    continue
                dm1_r, dm2_r = trans_rdm12(
                    fcisolver_use,
                    np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                    np.asarray(ci_list[r], dtype=np.float64).ravel(),
                    int(ncas),
                    nelecas,
                )
                dm1_r = np.asarray(dm1_r, dtype=np.float64)
                dm2_r = np.asarray(dm2_r, dtype=np.float64)
                dm1_lci += wr * (dm1_r + dm1_r.T)
                dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))

            de_lci = _grad_elec_active_dense(
                ao_basis=ao_basis_ref,
                atom_coords_bohr=coords,
                atom_charges=charges,
                h_ao=cache_cpu.h_ao,
                mo_coeff=C_ref,
                dm1_act=dm1_lci,
                dm2_act=dm2_lci,
                ncore=int(ncore),
                ncas=int(ncas),
                backend=backend_s,
                cache_cpu=cache_cpu.eri4c_cache,
                pair_table_threads=int(pair_table_threads),
                max_tile_bytes=int(max_tile_bytes),
                max_eri_bytes=int(max_eri_bytes),
                threads=int(threads),
            )

            if dm1_sa is None or dm2_sa is None:  # pragma: no cover
                raise RuntimeError("internal error: missing SA RDMs for split_orbfd response")

            de_lorb = _Lorb_dot_dgorb_dx_dense(
                ao_basis=ao_basis_ref,
                atom_coords_bohr=coords,
                atom_charges=charges,
                h_ao=cache_cpu.h_ao,
                mo_coeff=C_ref,
                dm1_act=dm1_sa,
                dm2_act=dm2_sa,
                Lorb=np.asarray(Lorb_mat, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
                backend=backend_s,
                cache_cpu=cache_cpu.eri4c_cache,
                pair_table_threads=int(pair_table_threads),
                max_tile_bytes=int(max_tile_bytes),
                max_eri_bytes=int(max_eri_bytes),
                threads=int(threads),
            )

            resp_full = np.asarray(de_lci, dtype=np.float64) + np.asarray(de_lorb, dtype=np.float64)

        nac_num = np.asarray(ham, dtype=np.float64)[np.asarray(atmlst_use, dtype=np.int32)]
        if response == "split_orbfd":
            nac_num = nac_num + np.asarray(resp_full, dtype=np.float64)[np.asarray(atmlst_use, dtype=np.int32)]
        if not bool(mult_ediff):
            if abs(ediff) < 1e-12:
                raise ZeroDivisionError("E_bra - E_ket is too small; use mult_ediff=True for numerator mode")
            nac_num = nac_num / ediff
        nac[bra, ket] = np.asarray(nac_num, dtype=np.float64)

    return nac

def grad_2e_ham_dense_eri4c_contracted(
    *,
    ao_basis,
    atom_coords_bohr: np.ndarray,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ncore: int,
    ncas: int,
    backend: Literal["cpu", "cuda"] = "cpu",
    cuda_ctx: ERI4cDerivContractionCUDAContext | None = None,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
) -> np.ndarray:
    """2e part of <bra| dH_elec/dR |ket> using dense contracted 4c derivatives.

    Notes
    -----
    - This is the **explicit** (Hellmann–Feynman-like) derivative term w.r.t AO
      integral centers. Orbital/CI response terms are handled elsewhere.
    - Returns a full (natm,3) array in Eh/Bohr.
    """

    backend_s = str(backend).strip().lower()
    if backend_s not in ("cpu", "cuda"):
        raise NotImplementedError("dense 4c contracted derivatives currently implemented for backend='cpu' or backend='cuda' only")

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    mo_coeff = np.asarray(mo_coeff, dtype=np.float64, order="C")
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be 2D (nao,nmo)")
    nao, nmo = map(int, mo_coeff.shape)

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0 or ncore + ncas > nmo:
        raise ValueError("invalid ncore/ncas for mo_coeff")
    nocc = ncore + ncas

    dm1_act = _symm_dm1(dm1_act)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")
    dm2_act = _symm_dm2_for_eri_contraction(dm2_act)
    if dm2_act.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            pair_table_threads=int(pair_table_threads),
        )

    sp = cache_cpu.sp
    ctx = cache_cpu.ctx
    shell_atom = cache_cpu.shell_atom
    shell_l = cache_cpu.shell_l
    shell_ao_start = cache_cpu.shell_ao_start
    sp_A = cache_cpu.sp_A
    sp_B = cache_cpu.sp_B

    # Densities in AO basis.
    C_core = mo_coeff[:, :ncore]
    C_act = mo_coeff[:, ncore:nocc]
    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
    else:
        D_core = np.zeros((nao, nao), dtype=np.float64)
    D_act = C_act @ dm1_act @ C_act.T
    D_w = D_act + 0.5 * D_core

    dm2_flat = dm2_act.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    fn_batch = None
    threads_cpu = threads_i
    threads_cuda = int(threads_i) if int(threads_i) > 0 else 256

    if backend_s == "cpu":
        try:
            from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "CPU ERI extension is not built. Build it with:\n"
                "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
            ) from e

        fn_batch = getattr(_ext, "eri_rys_deriv_contracted_cart_sp_batch_cy", None)
        if fn_batch is None:  # pragma: no cover
            raise RuntimeError(
                "asuka.cueri._eri_rys_cpu is missing eri_rys_deriv_contracted_cart_sp_batch_cy; rebuild it"
            )
    else:
        if int(threads_cuda) % 32 != 0 or int(threads_cuda) > 256:
            raise ValueError("CUDA contracted derivatives require threads to be a multiple of 32 and <= 256")
        if cuda_ctx is None:
            cuda_ctx = make_eri4c_deriv_contraction_cuda_context(ao_basis, sp, threads=int(threads_cuda))
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e
        grad_dev = cp.zeros((natm, 3), dtype=cp.float64)
        shell_atom_dev = cp.ascontiguousarray(cp.asarray(shell_atom, dtype=cp.int32))
        spCD_groups: list[tuple[np.ndarray, Any]] = []
        for spCD_all in cache_cpu.spCD_by_key.values():
            spCD_all_h = np.asarray(spCD_all, dtype=np.int32, order="C")
            if int(spCD_all_h.size) == 0:
                continue
            spCD_groups.append((spCD_all_h, cp.ascontiguousarray(cp.asarray(spCD_all_h, dtype=cp.int32))))
        from asuka.cueri import gpu as cueri_gpu  # noqa: PLC0415
        use_fused_atom_scatter = bool(getattr(cueri_gpu, "eri_rys_generic_deriv_contracted_atom_grad_inplace_device", None))
        work: dict[tuple[int, int, int], dict[str, Any]] = {}

    nsp = int(sp_A.size)
    grad = np.zeros((natm, 3), dtype=np.float64)

    for spAB in range(nsp):
        A = int(sp_A[spAB])
        B = int(sp_B[spAB])
        aoA = int(shell_ao_start[A])
        aoB = int(shell_ao_start[B])
        la = int(shell_l[A])
        lb = int(shell_l[B])
        nA = int(ncart(la))
        nB = int(ncart(lb))
        nAB = int(nA * nB)

        atomA = int(shell_atom[A])
        atomB = int(shell_atom[B])

        # Active pair coeffs for AB.
        CA = np.asarray(C_act[aoA : aoA + nA, :], dtype=np.float64, order="C")
        CB = np.asarray(C_act[aoB : aoB + nB, :], dtype=np.float64, order="C")
        K_AB = build_pair_coeff_ordered(CA, CB, same_shell=bool(A == B))  # (nAB,ncas^2)
        M_AB = K_AB @ dm2_flat  # (nAB,ncas^2)

        if backend_s == "cpu":
            spCD_iter = cache_cpu.spCD_by_key.values()
        else:
            spCD_iter = spCD_groups
        for spCD_item in spCD_iter:
            # Filter by lc/ld class via the precomputed grouping.
            spCD_all_dev = None
            if backend_s == "cpu":
                spCD_all = np.asarray(spCD_item, dtype=np.int32, order="C")
                if int(spCD_all.size) == 0:
                    continue
            else:
                spCD_all, spCD_all_dev = spCD_item

            # Determine tile sizes from the first task in the class.
            spCD0 = int(spCD_all[0])
            C0 = int(sp_A[spCD0])
            D0 = int(sp_B[spCD0])
            lc = int(shell_l[C0])
            ld = int(shell_l[D0])
            nC = int(ncart(lc))
            nD = int(ncart(ld))
            nCD = int(nC * nD)

            # Chunk tasks to bound peak memory for bar tiles.
            bytes_per_task = int(8 * nAB * nCD)
            chunk_nt = int(max(1, max_tile_bytes_i // max(bytes_per_task, 1)))

            for i0 in range(0, int(spCD_all.size), chunk_nt):
                i1 = min(int(spCD_all.size), i0 + chunk_nt)
                spCD = np.asarray(spCD_all[i0:i1], dtype=np.int32, order="C")
                nt = int(spCD.size)
                if nt == 0:
                    continue

                bar = np.empty((nt, nAB, nCD), dtype=np.float64)

                # Build per-task bar tiles for this (spAB, spCD_chunk).
                for t, spCD_t in enumerate(spCD.tolist()):
                    C = int(sp_A[int(spCD_t)])
                    D = int(sp_B[int(spCD_t)])
                    aoC = int(shell_ao_start[C])
                    aoD = int(shell_ao_start[D])
                    same_cd = bool(C == D)

                    bar_mean = _bar_tile_meanfield_jk(
                        D_w,
                        D_core,
                        aoA=aoA,
                        aoB=aoB,
                        aoC=aoC,
                        aoD=aoD,
                        nA=nA,
                        nB=nB,
                        nC=nC,
                        nD=nD,
                        same_ab=bool(A == B),
                        same_cd=same_cd,
                    )

                    CC = np.asarray(C_act[aoC : aoC + nC, :], dtype=np.float64, order="C")
                    CD = np.asarray(C_act[aoD : aoD + nD, :], dtype=np.float64, order="C")
                    K_CD = build_pair_coeff_ordered(CC, CD, same_shell=same_cd)  # (nCD,ncas^2)
                    bar_dm2 = 0.5 * (M_AB @ K_CD.T)  # (nAB,nCD)

                    bar[t] = bar_mean + bar_dm2

                if backend_s == "cpu":
                    out = fn_batch(
                        ctx.shell_cxyz,
                        ctx.shell_l,
                        ctx.shell_prim_start,
                        ctx.shell_nprim,
                        ctx.prim_exp,
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
                        np.asarray(bar, dtype=np.float64, order="C"),
                        int(threads_cpu),
                    )
                    out = np.asarray(out, dtype=np.float64)
                else:
                    if spCD_all_dev is None:  # pragma: no cover
                        raise RuntimeError("internal error: missing device spCD class")

                    wkey = (int(nAB), int(nCD), int(chunk_nt))
                    w = work.get(wkey)
                    if w is None:
                        w = {
                            "task_spAB": cp.empty((int(chunk_nt),), dtype=cp.int32),
                            "task_spCD": cp.empty((int(chunk_nt),), dtype=cp.int32),
                            "bar": cp.empty((int(chunk_nt), int(nAB), int(nCD)), dtype=cp.float64),
                        }
                        if not use_fused_atom_scatter:
                            w["out"] = cp.empty((int(chunk_nt), 12), dtype=cp.float64)
                        work[wkey] = w

                    task_spAB_dev = w["task_spAB"][:nt]
                    task_spCD_dev = w["task_spCD"][:nt]
                    bar_dev = w["bar"][:nt]

                    task_spAB_dev.fill(int(spAB))
                    task_spCD_dev[...] = spCD_all_dev[i0:i1]
                    bar_dev.set(bar)

                    if use_fused_atom_scatter:
                        cueri_gpu.eri_rys_generic_deriv_contracted_atom_grad_inplace_device(
                            task_spAB_dev,
                            task_spCD_dev,
                            cuda_ctx.dsp,
                            cuda_ctx.dbasis,
                            cuda_ctx.dpt,
                            int(la),
                            int(lb),
                            int(lc),
                            int(ld),
                            bar_dev,
                            shell_atom_dev,
                            grad_dev,
                            threads=int(threads_cuda),
                            sync=False,
                        )
                    else:
                        out_buf = w["out"][:nt]
                        out_view = cueri_gpu.eri_rys_generic_deriv_contracted_device(
                            task_spAB_dev,
                            task_spCD_dev,
                            cuda_ctx.dsp,
                            cuda_ctx.dbasis,
                            cuda_ctx.dpt,
                            int(la),
                            int(lb),
                            int(lc),
                            int(ld),
                            bar_dev,
                            out=out_buf,
                            threads=int(threads_cuda),
                            sync=False,
                        )
                        _accum_eri4c_out_to_grad_dev(
                            out_dev=out_view,
                            task_spCD_dev=task_spCD_dev,
                            atomA=atomA,
                            atomB=atomB,
                            grad_dev=grad_dev,
                            shell_atom_dev=shell_atom_dev,
                            sp_A_dev=cuda_ctx.dsp.sp_A,
                            sp_B_dev=cuda_ctx.dsp.sp_B,
                        )
                    continue
                if out.shape != (nt, 4, 3):  # pragma: no cover
                    raise RuntimeError("unexpected output shape from contracted derivative kernel")

                # Accumulate to atoms. A/B are constant in this outer loop.
                grad[atomA] += out[:, 0, :].sum(axis=0)
                grad[atomB] += out[:, 1, :].sum(axis=0)

                shellC = sp_A[spCD]
                shellD = sp_B[spCD]
                atomC = shell_atom[shellC]
                atomD = shell_atom[shellD]
                np.add.at(grad, atomC, out[:, 2, :])
                np.add.at(grad, atomD, out[:, 3, :])

    if backend_s == "cuda":
        return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)
    return np.asarray(grad, dtype=np.float64)


def grad_2e_lorb_dense_eri4c_contracted(
    *,
    ao_basis,
    atom_coords_bohr: np.ndarray,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    Lorb: np.ndarray,
    ncore: int,
    ncas: int,
    backend: Literal["cpu", "cuda"] = "cpu",
    cuda_ctx: ERI4cDerivContractionCUDAContext | None = None,
    cache_cpu: DenseERI4cDerivContractionCPUCache | None = None,
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
) -> np.ndarray:
    """2e part of ``Lorb_dot_dgorb_dx`` using dense contracted 4c derivatives."""

    backend_s = str(backend).strip().lower()
    if backend_s not in ("cpu", "cuda"):
        raise NotImplementedError("dense 4c contracted derivatives currently implemented for backend='cpu' or backend='cuda' only")

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64)
    if atom_coords_bohr.ndim != 2 or atom_coords_bohr.shape[1] != 3:
        raise ValueError("atom_coords_bohr must have shape (natm,3)")
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return np.zeros((0, 3), dtype=np.float64)

    mo_coeff = np.asarray(mo_coeff, dtype=np.float64, order="C")
    if mo_coeff.ndim != 2:
        raise ValueError("mo_coeff must be 2D (nao,nmo)")
    nao, nmo = map(int, mo_coeff.shape)

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0 or ncas <= 0 or ncore + ncas > nmo:
        raise ValueError("invalid ncore/ncas for mo_coeff")
    nocc = ncore + ncas

    L = np.asarray(Lorb, dtype=np.float64)
    if L.shape != (nmo, nmo):
        raise ValueError("Lorb shape mismatch with mo_coeff")

    dm1_act = _symm_dm1(dm1_act)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError("dm1_act shape mismatch")
    dm2_act = _symm_dm2_for_eri_contraction(dm2_act)
    if dm2_act.shape != (ncas, ncas, ncas, ncas):
        raise ValueError("dm2_act shape mismatch")

    if cache_cpu is None:
        cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
            ao_basis,
            atom_coords_bohr=atom_coords_bohr,
            pair_table_threads=int(pair_table_threads),
        )

    ctx = cache_cpu.ctx
    shell_atom = cache_cpu.shell_atom
    shell_l = cache_cpu.shell_l
    shell_ao_start = cache_cpu.shell_ao_start
    sp_A = cache_cpu.sp_A
    sp_B = cache_cpu.sp_B

    # L-effective AO coefficients.
    C = mo_coeff
    C_L = np.asarray(C @ L, dtype=np.float64, order="C")
    C_core = np.asarray(C[:, :ncore], dtype=np.float64, order="C")
    C_act = np.asarray(C[:, ncore:nocc], dtype=np.float64, order="C")
    C_L_core = np.asarray(C_L[:, :ncore], dtype=np.float64, order="C")
    C_L_act = np.asarray(C_L[:, ncore:nocc], dtype=np.float64, order="C")

    # AO densities.
    if ncore:
        D_core = 2.0 * (C_core @ C_core.T)
        D_L_core = 2.0 * (C_L_core @ C_core.T)
        D_L_core = D_L_core + D_L_core.T
    else:
        D_core = np.zeros((nao, nao), dtype=np.float64)
        D_L_core = np.zeros((nao, nao), dtype=np.float64)

    D_act = C_act @ dm1_act @ C_act.T
    D_L_act = C_L_act @ dm1_act @ C_act.T
    D_L_act = D_L_act + D_L_act.T

    D_w = D_act + 0.5 * D_core
    D_wL = D_L_act + 0.5 * D_L_core

    dm2_flat = dm2_act.reshape(ncas * ncas, ncas * ncas)
    dm2_flat = 0.5 * (dm2_flat + dm2_flat.T)

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    fn_batch = None
    threads_cpu = threads_i
    threads_cuda = int(threads_i) if int(threads_i) > 0 else 256

    if backend_s == "cpu":
        try:
            from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "CPU ERI extension is not built. Build it with:\n"
                "  python -m asuka.cueri.build_cpu_ext build_ext --inplace"
            ) from e

        fn_batch = getattr(_ext, "eri_rys_deriv_contracted_cart_sp_batch_cy", None)
        if fn_batch is None:  # pragma: no cover
            raise RuntimeError(
                "asuka.cueri._eri_rys_cpu is missing eri_rys_deriv_contracted_cart_sp_batch_cy; rebuild it"
            )
    else:
        if int(threads_cuda) % 32 != 0 or int(threads_cuda) > 256:
            raise ValueError("CUDA contracted derivatives require threads to be a multiple of 32 and <= 256")
        if cuda_ctx is None:
            cuda_ctx = make_eri4c_deriv_contraction_cuda_context(ao_basis, cache_cpu.sp, threads=int(threads_cuda))
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("backend='cuda' requires CuPy") from e
        grad_dev = cp.zeros((natm, 3), dtype=cp.float64)
        shell_atom_dev = cp.ascontiguousarray(cp.asarray(shell_atom, dtype=cp.int32))
        spCD_groups: list[tuple[np.ndarray, Any]] = []
        for spCD_all in cache_cpu.spCD_by_key.values():
            spCD_all_h = np.asarray(spCD_all, dtype=np.int32, order="C")
            if int(spCD_all_h.size) == 0:
                continue
            spCD_groups.append((spCD_all_h, cp.ascontiguousarray(cp.asarray(spCD_all_h, dtype=cp.int32))))
        from asuka.cueri import gpu as cueri_gpu  # noqa: PLC0415
        use_fused_atom_scatter = bool(getattr(cueri_gpu, "eri_rys_generic_deriv_contracted_atom_grad_inplace_device", None))
        work: dict[tuple[int, int, int], dict[str, Any]] = {}

    nsp = int(sp_A.size)
    grad = np.zeros((natm, 3), dtype=np.float64)

    for spAB in range(nsp):
        A = int(sp_A[spAB])
        B = int(sp_B[spAB])
        aoA = int(shell_ao_start[A])
        aoB = int(shell_ao_start[B])
        la = int(shell_l[A])
        lb = int(shell_l[B])
        nA = int(ncart(la))
        nB = int(ncart(lb))
        nAB = int(nA * nB)

        atomA = int(shell_atom[A])
        atomB = int(shell_atom[B])

        same_ab = bool(A == B)

        # Pair coefficients and their L-variations for AB.
        CA = np.asarray(C_act[aoA : aoA + nA, :], dtype=np.float64, order="C")
        CB = np.asarray(C_act[aoB : aoB + nB, :], dtype=np.float64, order="C")
        K_AB = build_pair_coeff_ordered(CA, CB, same_shell=same_ab)  # (nAB,ncas^2)

        CA_L = np.asarray(C_L_act[aoA : aoA + nA, :], dtype=np.float64, order="C")
        CB_L = np.asarray(C_L_act[aoB : aoB + nB, :], dtype=np.float64, order="C")
        dK_AB = build_pair_coeff_ordered(CA_L, CB, same_shell=same_ab) + build_pair_coeff_ordered(
            CA, CB_L, same_shell=same_ab
        )

        M_AB = K_AB @ dm2_flat  # (nAB,ncas^2)
        dM_AB = dK_AB @ dm2_flat

        if backend_s == "cpu":
            spCD_iter = cache_cpu.spCD_by_key.values()
        else:
            spCD_iter = spCD_groups
        for spCD_item in spCD_iter:
            spCD_all_dev = None
            if backend_s == "cpu":
                spCD_all = np.asarray(spCD_item, dtype=np.int32, order="C")
                if int(spCD_all.size) == 0:
                    continue
            else:
                spCD_all, spCD_all_dev = spCD_item

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

                bar = np.empty((nt, nAB, nCD), dtype=np.float64)

                for t, spCD_t in enumerate(spCD.tolist()):
                    Csh = int(sp_A[int(spCD_t)])
                    Dsh = int(sp_B[int(spCD_t)])
                    aoC = int(shell_ao_start[Csh])
                    aoD = int(shell_ao_start[Dsh])
                    same_cd = bool(Csh == Dsh)

                    bar_mean = _bar_tile_meanfield_jk(
                        D_wL,
                        D_core,
                        aoA=aoA,
                        aoB=aoB,
                        aoC=aoC,
                        aoD=aoD,
                        nA=nA,
                        nB=nB,
                        nC=nC,
                        nD=nD,
                        same_ab=same_ab,
                        same_cd=same_cd,
                    )
                    bar_mean += _bar_tile_meanfield_jk(
                        D_w,
                        D_L_core,
                        aoA=aoA,
                        aoB=aoB,
                        aoC=aoC,
                        aoD=aoD,
                        nA=nA,
                        nB=nB,
                        nC=nC,
                        nD=nD,
                        same_ab=same_ab,
                        same_cd=same_cd,
                    )

                    CC = np.asarray(C_act[aoC : aoC + nC, :], dtype=np.float64, order="C")
                    CD = np.asarray(C_act[aoD : aoD + nD, :], dtype=np.float64, order="C")
                    K_CD = build_pair_coeff_ordered(CC, CD, same_shell=same_cd)  # (nCD,ncas^2)

                    CC_L = np.asarray(C_L_act[aoC : aoC + nC, :], dtype=np.float64, order="C")
                    CD_L = np.asarray(C_L_act[aoD : aoD + nD, :], dtype=np.float64, order="C")
                    dK_CD = build_pair_coeff_ordered(CC_L, CD, same_shell=same_cd) + build_pair_coeff_ordered(
                        CC, CD_L, same_shell=same_cd
                    )

                    bar_dm2 = 0.5 * (dM_AB @ K_CD.T + M_AB @ dK_CD.T)
                    bar[t] = bar_mean + bar_dm2

                if backend_s == "cpu":
                    out = fn_batch(
                        ctx.shell_cxyz,
                        ctx.shell_l,
                        ctx.shell_prim_start,
                        ctx.shell_nprim,
                        ctx.prim_exp,
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
                        np.asarray(bar, dtype=np.float64, order="C"),
                        int(threads_cpu),
                    )
                    out = np.asarray(out, dtype=np.float64)
                else:
                    if spCD_all_dev is None:  # pragma: no cover
                        raise RuntimeError("internal error: missing device spCD class")

                    wkey = (int(nAB), int(nCD), int(chunk_nt))
                    w = work.get(wkey)
                    if w is None:
                        w = {
                            "task_spAB": cp.empty((int(chunk_nt),), dtype=cp.int32),
                            "task_spCD": cp.empty((int(chunk_nt),), dtype=cp.int32),
                            "bar": cp.empty((int(chunk_nt), int(nAB), int(nCD)), dtype=cp.float64),
                        }
                        if not use_fused_atom_scatter:
                            w["out"] = cp.empty((int(chunk_nt), 12), dtype=cp.float64)
                        work[wkey] = w

                    task_spAB_dev = w["task_spAB"][:nt]
                    task_spCD_dev = w["task_spCD"][:nt]
                    bar_dev = w["bar"][:nt]

                    task_spAB_dev.fill(int(spAB))
                    task_spCD_dev[...] = spCD_all_dev[i0:i1]
                    bar_dev.set(bar)

                    if use_fused_atom_scatter:
                        cueri_gpu.eri_rys_generic_deriv_contracted_atom_grad_inplace_device(
                            task_spAB_dev,
                            task_spCD_dev,
                            cuda_ctx.dsp,
                            cuda_ctx.dbasis,
                            cuda_ctx.dpt,
                            int(la),
                            int(lb),
                            int(lc),
                            int(ld),
                            bar_dev,
                            shell_atom_dev,
                            grad_dev,
                            threads=int(threads_cuda),
                            sync=False,
                        )
                    else:
                        out_buf = w["out"][:nt]
                        out_view = cueri_gpu.eri_rys_generic_deriv_contracted_device(
                            task_spAB_dev,
                            task_spCD_dev,
                            cuda_ctx.dsp,
                            cuda_ctx.dbasis,
                            cuda_ctx.dpt,
                            int(la),
                            int(lb),
                            int(lc),
                            int(ld),
                            bar_dev,
                            out=out_buf,
                            threads=int(threads_cuda),
                            sync=False,
                        )
                        _accum_eri4c_out_to_grad_dev(
                            out_dev=out_view,
                            task_spCD_dev=task_spCD_dev,
                            atomA=atomA,
                            atomB=atomB,
                            grad_dev=grad_dev,
                            shell_atom_dev=shell_atom_dev,
                            sp_A_dev=cuda_ctx.dsp.sp_A,
                            sp_B_dev=cuda_ctx.dsp.sp_B,
                        )
                    continue
                if out.shape != (nt, 4, 3):  # pragma: no cover
                    raise RuntimeError("unexpected output shape from contracted derivative kernel")

                grad[atomA] += out[:, 0, :].sum(axis=0)
                grad[atomB] += out[:, 1, :].sum(axis=0)

                shellC = sp_A[spCD]
                shellD = sp_B[spCD]
                atomC = shell_atom[shellC]
                atomD = shell_atom[shellD]
                np.add.at(grad, atomC, out[:, 2, :])
                np.add.at(grad, atomD, out[:, 3, :])

    if backend_s == "cuda":
        return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)
    return np.asarray(grad, dtype=np.float64)


__all__ = [
    "DenseERI4cDerivContractionCPUCache",
    "DenseNACCPUCache",
    "build_dense_eri4c_deriv_contraction_cache_cpu",
    "build_dense_nac_cache_cpu",
    "grad_2e_ham_dense_eri4c_contracted",
    "sacasscf_nonadiabatic_couplings_dense",
    "_build_eri_ao_cart_dense_cpu",
    "_dense_vhf_ao_from_eri",
    "_grad_elec_active_dense",
]
