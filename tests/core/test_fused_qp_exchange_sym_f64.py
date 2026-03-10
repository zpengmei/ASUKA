"""Parity tests for fused_qp_exchange_sym_f64 vs unpack + matmul chain + pack."""

from __future__ import annotations

import numpy as np
import pytest

from asuka.integrals.df_packed_s2 import (
    fused_qp_exchange_sym_f64,
    unpack_Qp_to_Qmn_block,
)
from asuka.integrals.tri_packed import ntri_from_nao


def _make_sym(rng, n):
    """Random symmetric (n,n) matrix."""
    A = rng.standard_normal((n, n))
    return 0.5 * (A + A.T)


def _make_random_Qp(rng, naux, nao):
    ntri = int(ntri_from_nao(nao))
    return rng.standard_normal((naux, ntri))


def _ref_exchange_sym(B_Qp, D1, D2, out_Qp, *, nao, q0, q_count, alpha):
    """Reference: unpack + matmul chain + lower-tri pack."""
    bq = unpack_Qp_to_Qmn_block(B_Qp, nao=nao, q0=q0, q_count=q_count)
    t = np.matmul(np.matmul(D1[None, :, :], bq), D2)
    t += np.matmul(np.matmul(D2[None, :, :], bq), D1)
    t *= alpha
    tri_i, tri_j = np.tril_indices(nao)
    out_Qp[q0 : q0 + q_count] += t[:, tri_i, tri_j]


@pytest.mark.parametrize(
    "nao,naux,q0,alpha",
    [
        (6, 20, 0, -0.5),
        (10, 32, 4, 1.0),
        (20, 64, 8, -0.5),
        (40, 100, 10, 0.25),
    ],
)
def test_fused_qp_exchange_sym_cpu_parity(nao, naux, q0, alpha):
    rng = np.random.default_rng(13 + nao)
    q_count = min(16, naux - q0)
    ntri = int(ntri_from_nao(nao))

    B_Qp = _make_random_Qp(rng, naux, nao)
    D1 = _make_sym(rng, nao)
    D2 = _make_sym(rng, nao)

    out_ref = np.zeros((naux, ntri))
    _ref_exchange_sym(B_Qp, D1, D2, out_ref, nao=nao, q0=q0, q_count=q_count, alpha=alpha)

    out_got = np.zeros((naux, ntri))
    fused_qp_exchange_sym_f64(B_Qp, D1, D2, out_got, nao=nao, q0=q0, q_count=q_count, alpha=alpha)

    np.testing.assert_allclose(out_got, out_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.cuda
def test_fused_qp_exchange_sym_gpu_parity():
    cp = pytest.importorskip("cupy")
    rng = np.random.default_rng(99)
    nao, naux, q0, q_count, alpha = 30, 80, 5, 20, -0.5
    ntri = int(ntri_from_nao(nao))

    B_np = _make_random_Qp(rng, naux, nao)
    D1_np = _make_sym(rng, nao)
    D2_np = _make_sym(rng, nao)

    out_ref = np.zeros((naux, ntri))
    _ref_exchange_sym(B_np, D1_np, D2_np, out_ref, nao=nao, q0=q0, q_count=q_count, alpha=alpha)

    out_gpu = cp.zeros((naux, ntri), dtype=cp.float64)
    fused_qp_exchange_sym_f64(
        cp.asarray(B_np), cp.asarray(D1_np), cp.asarray(D2_np), out_gpu,
        nao=nao, q0=q0, q_count=q_count, alpha=alpha,
    )
    out_got = cp.asnumpy(out_gpu)

    np.testing.assert_allclose(out_got, out_ref, rtol=1e-10, atol=1e-10)


def test_fused_qp_exchange_sym_zero_q_count():
    rng = np.random.default_rng(0)
    nao, naux = 5, 10
    ntri = int(ntri_from_nao(nao))
    B_Qp = _make_random_Qp(rng, naux, nao)
    D1, D2 = _make_sym(rng, nao), _make_sym(rng, nao)
    out = np.zeros((naux, ntri))
    fused_qp_exchange_sym_f64(B_Qp, D1, D2, out, nao=nao, q0=0, q_count=0)
    np.testing.assert_array_equal(out, 0.0)
