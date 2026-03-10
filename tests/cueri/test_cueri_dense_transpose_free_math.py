from __future__ import annotations

import numpy as np


def test_transpose_free_primary_contraction_matches_reference():
    rng = np.random.default_rng(1234)
    m, nAB, nCD, npair = 5, 7, 9, 11

    tile = rng.standard_normal((m, nAB, nCD), dtype=np.float64)
    tile_t = tile.transpose(0, 2, 1)
    k_ab = rng.standard_normal((m, nAB, npair), dtype=np.float64)
    k_cd = rng.standard_normal((m, nCD, npair), dtype=np.float64)

    # Reference path (original orientation): tmp = tile @ K_CD
    tmp_ref = np.matmul(tile, k_cd)
    out_ref = k_ab.reshape(m * nAB, npair).T @ tmp_ref.reshape(m * nAB, npair)

    # Transpose-free path for kernel-orientation tiles: einsum over CD index.
    tmp_new = np.einsum("mca,mcp->map", tile_t, k_cd, optimize=True)
    out_new = k_ab.reshape(m * nAB, npair).T @ tmp_new.reshape(m * nAB, npair)

    np.testing.assert_allclose(out_new, out_ref, rtol=1e-12, atol=1e-12)


def test_transpose_free_off_contraction_matches_reference():
    rng = np.random.default_rng(5678)
    m, nAB, nCD, npair = 6, 8, 5, 10

    tile = rng.standard_normal((m, nAB, nCD), dtype=np.float64)
    tile_t = tile.transpose(0, 2, 1)
    k_ab = rng.standard_normal((m, nAB, npair), dtype=np.float64)
    k_cd = rng.standard_normal((m, nCD, npair), dtype=np.float64)

    off = np.asarray([True, False, True, True, False, True], dtype=bool)
    tile_off = tile[off]
    tile_t_off = tile_t[off]
    k_ab_off = k_ab[off]
    k_cd_off = k_cd[off]
    noff = int(tile_off.shape[0])

    # Reference off path: tmp2 = tile_off.transpose(0,2,1) @ K_AB_off
    tmp2_ref = np.matmul(tile_off.transpose(0, 2, 1), k_ab_off)
    out_ref = k_cd_off.reshape(noff * nCD, npair).T @ tmp2_ref.reshape(noff * nCD, npair)

    # Transpose-free off path with kernel-orientation tiles already in (CD,AB).
    tmp2_new = np.matmul(tile_t_off, k_ab_off)
    out_new = k_cd_off.reshape(noff * nCD, npair).T @ tmp2_new.reshape(noff * nCD, npair)

    np.testing.assert_allclose(out_new, out_ref, rtol=1e-12, atol=1e-12)


def test_transpose_free_ab_group_tmp_matches_reference():
    rng = np.random.default_rng(9012)
    m, nAB, nCD, npair = 4, 6, 7, 9

    tile = rng.standard_normal((m, nAB, nCD), dtype=np.float64)
    tile_t = tile.transpose(0, 2, 1)
    k_cd = rng.standard_normal((m, nCD, npair), dtype=np.float64)

    tmp_ref = np.matmul(tile, k_cd)
    tmp_new = np.einsum("mca,mcp->map", tile_t, k_cd, optimize=True)

    np.testing.assert_allclose(tmp_new, tmp_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tmp_new.sum(axis=0), tmp_ref.sum(axis=0), rtol=1e-12, atol=1e-12)


def test_transpose_free_mixed_index_contractions_match_reference():
    rng = np.random.default_rng(3456)
    m, nAB, nCD = 5, 6, 4
    npair_left, npair_right = 7, 9

    tile = rng.standard_normal((m, nAB, nCD), dtype=np.float64)
    tile_t = tile.transpose(0, 2, 1)
    k_ab_mixed = rng.standard_normal((m, nAB, npair_left), dtype=np.float64)
    k_cd_act = rng.standard_normal((m, nCD, npair_right), dtype=np.float64)

    # Primary contraction parity (mixed-index dense builders).
    tmp_ref = np.matmul(tile, k_cd_act)  # (m,nAB,npair_right)
    out_ref = k_ab_mixed.reshape(m * nAB, npair_left).T @ tmp_ref.reshape(m * nAB, npair_right)

    tmp_new = np.einsum("mca,mcp->map", tile_t, k_cd_act, optimize=True)
    out_new = k_ab_mixed.reshape(m * nAB, npair_left).T @ tmp_new.reshape(m * nAB, npair_right)
    np.testing.assert_allclose(out_new, out_ref, rtol=1e-12, atol=1e-12)

    # Off-path parity (uses tile^T @ K_AB in reference).
    off = np.asarray([True, False, True, True, False], dtype=bool)
    tile_off = tile[off]
    tile_t_off = tile_t[off]
    k_ab_act_off = rng.standard_normal((int(off.sum()), nAB, npair_right), dtype=np.float64)
    k_cd_mixed_off = rng.standard_normal((int(off.sum()), nCD, npair_left), dtype=np.float64)

    tmp2_ref = np.matmul(tile_off.transpose(0, 2, 1), k_ab_act_off)
    out2_ref = k_cd_mixed_off.reshape(int(off.sum()) * nCD, npair_left).T @ tmp2_ref.reshape(
        int(off.sum()) * nCD, npair_right
    )
    tmp2_new = np.matmul(tile_t_off, k_ab_act_off)
    out2_new = k_cd_mixed_off.reshape(int(off.sum()) * nCD, npair_left).T @ tmp2_new.reshape(
        int(off.sum()) * nCD, npair_right
    )
    np.testing.assert_allclose(out2_new, out2_ref, rtol=1e-12, atol=1e-12)
