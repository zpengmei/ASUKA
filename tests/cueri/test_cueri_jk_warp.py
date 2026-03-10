"""Tests for warp-reduce J/K contraction kernels (D9).

Verifies that KernelContractJKTilesOrderedWarp and Multi2Warp produce
identical results to the original per-element kernels on small molecules.
"""

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="CuPy required for GPU tests")
pytestmark = pytest.mark.cuda

from asuka.cueri import _cueri_cuda_ext as _ext


def _make_random_tile_d(ntasks: int, nA: int, nB: int, nC: int, nD: int, seed: int = 0):
    """Random double tile of shape (ntasks, nA*nB, nC*nD) as a flat CuPy array."""
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(ntasks * nA * nB * nC * nD).astype(np.float64)
    return cupy.asarray(data)


def _make_valid_shell_layout(nshells: int, ncart_per_shell: int):
    """Create valid shell_ao_start, sp_A, sp_B with proper AO bounds.

    Each shell has exactly ncart_per_shell AOs.
    shell_ao_start[sh] = sh * ncart_per_shell.
    nao = nshells * ncart_per_shell.
    """
    nao = nshells * ncart_per_shell
    shell_ao_start = np.arange(nshells, dtype=np.int32) * ncart_per_shell
    return nao, shell_ao_start


@pytest.mark.parametrize("la,lb,lc,ld", [
    (0, 0, 0, 0),   # ssss: ncomp=1
    (1, 0, 0, 0),   # psss: ncomp=3
    (1, 1, 0, 0),   # ppss: ncomp=9
    (1, 1, 1, 1),   # pppp: ncomp=81
])
def test_warp_jk_vs_original_single(la, lb, lc, ld):
    """Warp kernel matches original per-element kernel for various angular momenta."""
    def ncart(l):
        return (l + 1) * (l + 2) // 2

    nA, nB, nC, nD = ncart(la), ncart(lb), ncart(lc), ncart(ld)
    # Use ncart_per_shell = max component count so all AO accesses stay in bounds
    ncart_max = max(nA, nB, nC, nD)
    nshells = 8
    ntasks = 50
    nao, shell_ao_start_np = _make_valid_shell_layout(nshells, ncart_max)

    shell_ao_start_dev = cupy.asarray(shell_ao_start_np)
    sp_A_dev = cupy.asarray(
        np.random.default_rng(1).integers(0, nshells, ntasks, dtype=np.int32)
    )
    sp_B_dev = cupy.asarray(
        np.random.default_rng(2).integers(0, nshells, ntasks, dtype=np.int32)
    )
    task_spAB = cupy.asarray(
        np.random.default_rng(3).integers(0, ntasks, ntasks, dtype=np.int32)
    )
    task_spCD = cupy.asarray(
        np.random.default_rng(4).integers(0, ntasks, ntasks, dtype=np.int32)
    )

    tile = _make_random_tile_d(ntasks, nA, nB, nC, nD, seed=42)
    D = cupy.asarray(np.random.default_rng(5).standard_normal((nao, nao)))
    D_flat = (D + D.T).ravel()  # symmetrize

    # Allocate all outputs before any kernel call to avoid pool-reuse OOB corruption
    J_orig = cupy.zeros(nao * nao, dtype=cupy.float64)
    K_orig = cupy.zeros(nao * nao, dtype=cupy.float64)
    J_warp = cupy.zeros(nao * nao, dtype=cupy.float64)
    K_warp = cupy.zeros(nao * nao, dtype=cupy.float64)

    _ext.contract_jk_tiles_ordered_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, D_flat, J_orig, K_orig,
        threads=256, stream=0, sync=True,
    )

    _ext.contract_jk_tiles_ordered_warp_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, D_flat, J_warp, K_warp,
        threads=32, stream=0, sync=True,
    )

    np.testing.assert_allclose(
        cupy.asnumpy(J_warp), cupy.asnumpy(J_orig), rtol=1e-12, atol=1e-14,
        err_msg=f"J mismatch for ({la}{lb}|{lc}{ld})",
    )
    np.testing.assert_allclose(
        cupy.asnumpy(K_warp), cupy.asnumpy(K_orig), rtol=1e-12, atol=1e-14,
        err_msg=f"K mismatch for ({la}{lb}|{lc}{ld})",
    )


def test_warp_jk_multi2_vs_original():
    """Multi2 warp kernel matches 2x single-density warp kernel (ground truth)."""
    nA, nB, nC, nD = 3, 3, 3, 3  # pppp
    # 5 pp-shells × 3 AOs/shell = 15 AOs; shell_ao_start = [0, 3, 6, 9, 12]
    nshells = 5
    ncart_per_shell = 3
    ntasks = 30
    nao, shell_ao_start_np = _make_valid_shell_layout(nshells, ncart_per_shell)  # nao=15

    shell_ao_start_dev = cupy.asarray(shell_ao_start_np)
    sp_A_dev = cupy.asarray(
        np.random.default_rng(1).integers(0, nshells, ntasks, dtype=np.int32)
    )
    sp_B_dev = cupy.asarray(
        np.random.default_rng(2).integers(0, nshells, ntasks, dtype=np.int32)
    )
    task_spAB = cupy.asarray(
        np.random.default_rng(3).integers(0, ntasks, ntasks, dtype=np.int32)
    )
    task_spCD = cupy.asarray(
        np.random.default_rng(4).integers(0, ntasks, ntasks, dtype=np.int32)
    )

    tile = _make_random_tile_d(ntasks, nA, nB, nC, nD, seed=99)
    rng = np.random.default_rng(7)
    Da = cupy.asarray(rng.standard_normal((nao, nao))).ravel()
    Db = cupy.asarray(rng.standard_normal((nao, nao))).ravel()

    # Allocate ALL output arrays before any kernel call to prevent pool-reuse OOB issues
    Ja_ref = cupy.zeros(nao * nao, dtype=cupy.float64)
    Ka_ref = cupy.zeros(nao * nao, dtype=cupy.float64)
    Jb_ref = cupy.zeros(nao * nao, dtype=cupy.float64)
    Kb_ref = cupy.zeros(nao * nao, dtype=cupy.float64)
    Ja_w = cupy.zeros(nao * nao, dtype=cupy.float64)
    Ka_w = cupy.zeros(nao * nao, dtype=cupy.float64)
    Jb_w = cupy.zeros(nao * nao, dtype=cupy.float64)
    Kb_w = cupy.zeros(nao * nao, dtype=cupy.float64)

    # Ground truth: 2x single-density warp kernel (one call per density)
    _ext.contract_jk_tiles_ordered_warp_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, Da, Ja_ref, Ka_ref,
        threads=32, stream=0, sync=True,
    )
    _ext.contract_jk_tiles_ordered_warp_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, Db, Jb_ref, Kb_ref,
        threads=32, stream=0, sync=True,
    )

    # Warp multi2
    _ext.contract_jk_tiles_ordered_warp_multi2_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, Da, Db, Ja_w, Ka_w, Jb_w, Kb_w,
        threads=32, stream=0, sync=True,
    )

    for name, ref, warp in [
        ("Ja", Ja_ref, Ja_w), ("Ka", Ka_ref, Ka_w),
        ("Jb", Jb_ref, Jb_w), ("Kb", Kb_ref, Kb_w),
    ]:
        np.testing.assert_allclose(
            cupy.asnumpy(warp), cupy.asnumpy(ref), rtol=1e-12, atol=1e-14,
            err_msg=f"{name} multi2 mismatch",
        )


def test_warp_jk_bkswap_symmetry():
    """Bra-ket swap (spab != spcd) handled identically by warp and original kernels."""
    nA = nB = nC = nD = 1  # ssss — 1 AO per shell, always in bounds
    nao = 5
    ntasks = 10

    shell_ao_start_dev = cupy.arange(nao, dtype=cupy.int32)
    sp_A_dev = cupy.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=cupy.int32)
    sp_B_dev = cupy.asarray([0, 1, 2, 3, 4, 1, 2, 3, 4, 0], dtype=cupy.int32)
    # spAB != spCD for last 5 tasks → triggers bk_swap
    task_spAB = cupy.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=cupy.int32)
    task_spCD = cupy.asarray([5, 6, 7, 8, 9, 0, 1, 2, 3, 4], dtype=cupy.int32)

    tile = cupy.asarray(
        np.random.default_rng(10).standard_normal(ntasks * nA * nB * nC * nD)
    )
    D_flat = cupy.asarray(
        np.random.default_rng(11).standard_normal(nao * nao)
    )

    J_orig = cupy.zeros(nao * nao, dtype=cupy.float64)
    K_orig = cupy.zeros(nao * nao, dtype=cupy.float64)
    J_warp = cupy.zeros(nao * nao, dtype=cupy.float64)
    K_warp = cupy.zeros(nao * nao, dtype=cupy.float64)

    _ext.contract_jk_tiles_ordered_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, D_flat, J_orig, K_orig,
        threads=256, stream=0, sync=True,
    )

    _ext.contract_jk_tiles_ordered_warp_inplace_device(
        task_spAB, task_spCD,
        sp_A_dev, sp_B_dev, shell_ao_start_dev,
        nao, nA, nB, nC, nD,
        tile, D_flat, J_warp, K_warp,
        threads=32, stream=0, sync=True,
    )

    np.testing.assert_allclose(
        cupy.asnumpy(J_warp), cupy.asnumpy(J_orig), rtol=1e-12, atol=1e-14,
        err_msg="J bkswap mismatch",
    )
    np.testing.assert_allclose(
        cupy.asnumpy(K_warp), cupy.asnumpy(K_orig), rtol=1e-12, atol=1e-14,
        err_msg="K bkswap mismatch",
    )
