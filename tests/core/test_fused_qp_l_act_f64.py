"""Parity tests for fused_qp_l_act_f64 vs unpack_Qp + matmul + matmul."""

from __future__ import annotations

import numpy as np
import pytest

from asuka.integrals.df_packed_s2 import (
    fused_qp_l_act_f64,
    unpack_Qp_to_Qmn_block,
)
from asuka.integrals.tri_packed import ntri_from_nao


def _make_random_Qp(rng, naux, nao):
    ntri = int(ntri_from_nao(nao))
    return rng.standard_normal((naux, ntri))


def _ref_l_act(B_Qp, C_act, *, nao, q0, q_count):
    """Reference: unpack Qp → (q,nao,nao), then two matmuls."""
    B_qmn = unpack_Qp_to_Qmn_block(B_Qp, nao=nao, q0=q0, q_count=q_count)
    X = np.matmul(B_qmn, C_act)           # (q,nao,ncas)
    L = np.matmul(C_act.T[None, :, :], X) # (q,ncas,ncas)
    return L


@pytest.mark.parametrize(
    "nao,ncas,naux,q0",
    [
        (6, 2, 20, 0),
        (10, 4, 32, 0),
        (20, 6, 64, 4),
        (50, 10, 100, 10),
        (80, 16, 128, 0),
    ],
)
def test_fused_qp_l_act_cpu_parity(nao, ncas, naux, q0):
    rng = np.random.default_rng(42 + nao * 100 + ncas)
    q_count = min(16, naux - q0)
    B_Qp = _make_random_Qp(rng, naux, nao)
    C_act = rng.standard_normal((nao, ncas))

    ref = _ref_l_act(B_Qp, C_act, nao=nao, q0=q0, q_count=q_count)
    got = fused_qp_l_act_f64(B_Qp, C_act, nao=nao, q0=q0, q_count=q_count)

    assert got.shape == (q_count, ncas, ncas), f"shape mismatch: {got.shape}"
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.cuda
def test_fused_qp_l_act_gpu_parity():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(7)
    nao, ncas, naux, q0, q_count = 50, 8, 128, 8, 32

    B_np = _make_random_Qp(rng, naux, nao)
    C_np = rng.standard_normal((nao, ncas))

    ref = _ref_l_act(B_np, C_np, nao=nao, q0=q0, q_count=q_count)

    B_gpu = cp.asarray(B_np)
    C_gpu = cp.asarray(C_np)
    got_gpu = fused_qp_l_act_f64(B_gpu, C_gpu, nao=nao, q0=q0, q_count=q_count)
    got = cp.asnumpy(got_gpu)

    assert got.shape == (q_count, ncas, ncas)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-10)


def test_fused_qp_l_act_zero_q_count():
    rng = np.random.default_rng(0)
    B_Qp = _make_random_Qp(rng, 10, 5)
    C_act = rng.standard_normal((5, 3))
    result = fused_qp_l_act_f64(B_Qp, C_act, nao=5, q0=0, q_count=0)
    assert result.shape == (0, 3, 3)
