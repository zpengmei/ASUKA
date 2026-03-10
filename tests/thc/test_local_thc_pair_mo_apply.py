import numpy as np
import pytest


@pytest.mark.cuda
def test_local_thc_eri_apply_pairs_mo_batched_matches_dense():
    cp = pytest.importorskip("cupy")
    try:
        a = cp.arange(10, dtype=cp.float64)
        _ = float(a.sum().item())
        cp.cuda.runtime.deviceSynchronize()
    except Exception as e:
        pytest.skip(f"CUDA/CuPy not usable in this environment: {type(e).__name__}: {e}")

    from asuka.hf.local_thc_factors import LocalTHCBlock, LocalTHCFactors
    from asuka.hf.local_thc_jk import local_thc_eri_apply_batched, local_thc_eri_apply_pairs_mo_batched

    # Small synthetic setup: one local block with ownership masking.
    rs = np.random.default_rng(0)
    nao = 12
    nloc = 8
    npt = 16
    naux = 6
    nbatch = 5
    nleft = 7
    nright = 4

    # Local AO ordering: [early secondary][primary][late secondary]
    n_early = 2
    n_primary = 3
    assert n_early + n_primary < nloc

    # Use a simple contiguous AO index list into the global AO basis.
    ao_idx = np.arange(nloc, dtype=np.int32)

    X = cp.asarray(rs.standard_normal((npt, nloc)), dtype=cp.float64)
    Y = cp.asarray(rs.standard_normal((npt, naux)), dtype=cp.float64)
    Z = cp.ascontiguousarray(Y @ Y.T, dtype=cp.float64)

    blk = LocalTHCBlock(
        block_id=0,
        ao_idx_global=np.ascontiguousarray(ao_idx),
        n_early=int(n_early),
        n_primary=int(n_primary),
        atoms_primary=tuple(),
        atoms_secondary_early=tuple(),
        atoms_secondary_late=tuple(),
        atoms_aux=tuple(),
        X=cp.ascontiguousarray(X),
        Y=cp.ascontiguousarray(Y),
        Z=cp.ascontiguousarray(Z),
        points=cp.zeros((npt, 3), dtype=cp.float64),
        weights=cp.ones((npt,), dtype=cp.float64),
        L_metric=cp.eye(naux, dtype=cp.float64),
        meta=None,
    )
    lthc = LocalTHCFactors(blocks=(blk,), nao=int(nao), ao_rep="cart", L_metric_full=None, meta=None)

    c_left = cp.asarray(rs.standard_normal((nbatch, nao)), dtype=cp.float64)
    c_right = cp.asarray(rs.standard_normal((nbatch, nao)), dtype=cp.float64)
    C_left = cp.asarray(rs.standard_normal((nao, nleft)), dtype=cp.float64)
    C_right = cp.asarray(rs.standard_normal((nao, nright)), dtype=cp.float64)

    # Reference: build dense D_batch and apply local-THC operator in AO, then contract.
    D_batch = 0.5 * (
        c_left[:, :, None] * c_right[:, None, :] + c_right[:, :, None] * c_left[:, None, :]
    )
    V_batch = local_thc_eri_apply_batched(D_batch, lthc, symmetrize=True)
    out_ref = cp.einsum("mp,bmn,nq->bpq", C_left, V_batch, C_right, optimize=True)

    out_new = local_thc_eri_apply_pairs_mo_batched(
        c_left,
        c_right,
        lthc,
        C_left,
        C_right,
        symmetrize=True,
    )

    max_abs = float(cp.max(cp.abs(out_ref - out_new)).item())
    assert max_abs < 1e-9


@pytest.mark.cuda
def test_local_thc_J_factored_mo_batched_matches_dense():
    cp = pytest.importorskip("cupy")
    try:
        a = cp.arange(10, dtype=cp.float64)
        _ = float(a.sum().item())
        cp.cuda.runtime.deviceSynchronize()
    except Exception as e:
        pytest.skip(f"CUDA/CuPy not usable in this environment: {type(e).__name__}: {e}")

    from asuka.hf.local_thc_factors import LocalTHCBlock, LocalTHCFactors
    from asuka.hf.local_thc_jk import local_thc_J_factored, local_thc_J_factored_mo_batched

    rs = np.random.default_rng(1)
    nao = 12
    nloc = 8
    npt = 16
    naux = 6
    nbatch = 4
    r = 3
    nleft = 5
    nright = 4

    n_early = 2
    n_primary = 3
    assert n_early + n_primary < nloc
    ao_idx = np.arange(nloc, dtype=np.int32)

    X = cp.asarray(rs.standard_normal((npt, nloc)), dtype=cp.float64)
    Y = cp.asarray(rs.standard_normal((npt, naux)), dtype=cp.float64)
    Z = cp.ascontiguousarray(Y @ Y.T, dtype=cp.float64)

    blk = LocalTHCBlock(
        block_id=0,
        ao_idx_global=np.ascontiguousarray(ao_idx),
        n_early=int(n_early),
        n_primary=int(n_primary),
        atoms_primary=tuple(),
        atoms_secondary_early=tuple(),
        atoms_secondary_late=tuple(),
        atoms_aux=tuple(),
        X=cp.ascontiguousarray(X),
        Y=cp.ascontiguousarray(Y),
        Z=cp.ascontiguousarray(Z),
        points=cp.zeros((npt, 3), dtype=cp.float64),
        weights=cp.ones((npt,), dtype=cp.float64),
        L_metric=cp.eye(naux, dtype=cp.float64),
        meta=None,
    )
    lthc = LocalTHCFactors(blocks=(blk,), nao=int(nao), ao_rep="cart", L_metric_full=None, meta=None)

    U_batch = cp.asarray(rs.standard_normal((nbatch, nao, r)), dtype=cp.float64)
    V_batch = U_batch  # symmetric densities

    C_left = cp.asarray(rs.standard_normal((nao, nleft)), dtype=cp.float64)
    C_right = cp.asarray(rs.standard_normal((nao, nright)), dtype=cp.float64)

    out_ref = cp.zeros((nbatch, nleft, nright), dtype=cp.float64)
    for b in range(int(nbatch)):
        Jb = local_thc_J_factored(U_batch[b], V_batch[b], lthc)
        out_ref[b] = C_left.T @ Jb @ C_right

    out_new = local_thc_J_factored_mo_batched(U_batch, V_batch, lthc, C_left, C_right)
    max_abs = float(cp.max(cp.abs(out_ref - out_new)).item())
    assert max_abs < 1e-9


@pytest.mark.cuda
def test_local_thc_K_factored_mo_batched_matches_dense():
    cp = pytest.importorskip("cupy")
    try:
        a = cp.arange(10, dtype=cp.float64)
        _ = float(a.sum().item())
        cp.cuda.runtime.deviceSynchronize()
    except Exception as e:
        pytest.skip(f"CUDA/CuPy not usable in this environment: {type(e).__name__}: {e}")

    from asuka.hf.local_thc_factors import LocalTHCBlock, LocalTHCFactors
    from asuka.hf.local_thc_jk import local_thc_K_blocked_factored, local_thc_K_factored_mo_batched

    rs = np.random.default_rng(2)
    nao = 12
    nloc = 8
    npt = 16
    naux = 6
    nbatch = 3
    r = 4
    nleft = 6
    nright = 3

    n_early = 2
    n_primary = 3
    assert n_early + n_primary < nloc
    ao_idx = np.arange(nloc, dtype=np.int32)

    X = cp.asarray(rs.standard_normal((npt, nloc)), dtype=cp.float64)
    Y = cp.asarray(rs.standard_normal((npt, naux)), dtype=cp.float64)
    Z = cp.ascontiguousarray(Y @ Y.T, dtype=cp.float64)

    blk = LocalTHCBlock(
        block_id=0,
        ao_idx_global=np.ascontiguousarray(ao_idx),
        n_early=int(n_early),
        n_primary=int(n_primary),
        atoms_primary=tuple(),
        atoms_secondary_early=tuple(),
        atoms_secondary_late=tuple(),
        atoms_aux=tuple(),
        X=cp.ascontiguousarray(X),
        Y=cp.ascontiguousarray(Y),
        Z=cp.ascontiguousarray(Z),
        points=cp.zeros((npt, 3), dtype=cp.float64),
        weights=cp.ones((npt,), dtype=cp.float64),
        L_metric=cp.eye(naux, dtype=cp.float64),
        meta=None,
    )
    lthc = LocalTHCFactors(blocks=(blk,), nao=int(nao), ao_rep="cart", L_metric_full=None, meta=None)

    U_batch = cp.asarray(rs.standard_normal((nbatch, nao, r)), dtype=cp.float64)
    V_batch = U_batch  # symmetric densities

    C_left = cp.asarray(rs.standard_normal((nao, nleft)), dtype=cp.float64)
    C_right = cp.asarray(rs.standard_normal((nao, nright)), dtype=cp.float64)

    out_ref = cp.zeros((nbatch, nleft, nright), dtype=cp.float64)
    for b in range(int(nbatch)):
        Kb = local_thc_K_blocked_factored(U_batch[b], V_batch[b], lthc, q_block=8)
        out_ref[b] = C_left.T @ Kb @ C_right

    out_new = local_thc_K_factored_mo_batched(U_batch, V_batch, lthc, C_left, C_right, q_block=8, batch_block=nbatch)
    max_abs = float(cp.max(cp.abs(out_ref - out_new)).item())
    assert max_abs < 1e-9


@pytest.mark.cuda
def test_local_thc_K_pairs_mo_batched_matches_dense():
    cp = pytest.importorskip("cupy")
    try:
        a = cp.arange(10, dtype=cp.float64)
        _ = float(a.sum().item())
        cp.cuda.runtime.deviceSynchronize()
    except Exception as e:
        pytest.skip(f"CUDA/CuPy not usable in this environment: {type(e).__name__}: {e}")

    from asuka.hf.local_thc_factors import LocalTHCBlock, LocalTHCFactors
    from asuka.hf.local_thc_jk import local_thc_K_blocked_factored, local_thc_K_pairs_mo_batched

    rs = np.random.default_rng(3)
    nao = 12
    nloc = 8
    npt = 16
    naux = 6
    nbatch = 5
    nleft = 7
    nright = 4

    n_early = 2
    n_primary = 3
    assert n_early + n_primary < nloc
    ao_idx = np.arange(nloc, dtype=np.int32)

    X = cp.asarray(rs.standard_normal((npt, nloc)), dtype=cp.float64)
    Y = cp.asarray(rs.standard_normal((npt, naux)), dtype=cp.float64)
    Z = cp.ascontiguousarray(Y @ Y.T, dtype=cp.float64)

    blk = LocalTHCBlock(
        block_id=0,
        ao_idx_global=np.ascontiguousarray(ao_idx),
        n_early=int(n_early),
        n_primary=int(n_primary),
        atoms_primary=tuple(),
        atoms_secondary_early=tuple(),
        atoms_secondary_late=tuple(),
        atoms_aux=tuple(),
        X=cp.ascontiguousarray(X),
        Y=cp.ascontiguousarray(Y),
        Z=cp.ascontiguousarray(Z),
        points=cp.zeros((npt, 3), dtype=cp.float64),
        weights=cp.ones((npt,), dtype=cp.float64),
        L_metric=cp.eye(naux, dtype=cp.float64),
        meta=None,
    )
    lthc = LocalTHCFactors(blocks=(blk,), nao=int(nao), ao_rep="cart", L_metric_full=None, meta=None)

    c_left = cp.asarray(rs.standard_normal((nbatch, nao)), dtype=cp.float64)
    c_right = cp.asarray(rs.standard_normal((nbatch, nao)), dtype=cp.float64)
    C_left = cp.asarray(rs.standard_normal((nao, nleft)), dtype=cp.float64)
    C_right = cp.asarray(rs.standard_normal((nao, nright)), dtype=cp.float64)

    out_ref = cp.zeros((nbatch, nleft, nright), dtype=cp.float64)
    for b in range(int(nbatch)):
        U = cp.stack((c_left[b], c_right[b]), axis=1)
        V = cp.stack((0.5 * c_right[b], 0.5 * c_left[b]), axis=1)
        Kb = local_thc_K_blocked_factored(U, V, lthc, q_block=8)
        out_ref[b] = C_left.T @ Kb @ C_right

    out_new = local_thc_K_pairs_mo_batched(c_left, c_right, lthc, C_left, C_right, q_block=8, batch_block=nbatch)
    max_abs = float(cp.max(cp.abs(out_ref - out_new)).item())
    assert max_abs < 1e-9
