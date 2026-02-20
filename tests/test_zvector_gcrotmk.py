import numpy as np
import pytest
import scipy.sparse.linalg as spla

from asuka.cuda.krylov_gcrotmk import gcrotmk_xp


def _random_system(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n)).astype(np.float64)
    # Keep the matrix safely nonsingular/well-conditioned.
    a += np.eye(n, dtype=np.float64) * float(n + 4)
    b = rng.normal(size=(n,)).astype(np.float64)
    return a, b


def test_gcrotmk_xp_numpy_matches_scipy():
    if not hasattr(spla, "gcrotmk"):
        pytest.skip("SciPy gcrotmk is unavailable")

    a, b = _random_system(36, seed=12)

    def matvec(x):
        xx = np.asarray(x, dtype=np.float64).ravel()
        return a @ xx

    x_ref, info_ref = spla.gcrotmk(a, b, rtol=1e-10, atol=0.0, maxiter=80, m=14, k=8)
    assert int(info_ref) == 0

    x, info = gcrotmk_xp(
        matvec,
        b,
        rtol=1e-10,
        atol=0.0,
        maxiter=80,
        m=14,
        k=8,
        CU=[],
        xp=np,
    )
    assert int(info) == 0

    err = np.linalg.norm(np.asarray(x) - np.asarray(x_ref)) / max(1.0, np.linalg.norm(np.asarray(x_ref)))
    assert float(err) < 1e-4

    resid = np.linalg.norm(a @ np.asarray(x, dtype=np.float64) - b)
    assert float(resid) < (1e-8 * np.linalg.norm(b) + 1e-12)


def test_gcrotmk_xp_recycle_space_oldest_truncation():
    rng = np.random.default_rng(23)
    n = 30
    a = rng.normal(size=(n, n)).astype(np.float64)
    a += np.eye(n, dtype=np.float64) * float(n + 3)

    def matvec(x):
        xx = np.asarray(x, dtype=np.float64).ravel()
        return a @ xx

    cu: list[tuple[np.ndarray | None, np.ndarray]] = []
    for _ in range(5):
        b = rng.normal(size=(n,)).astype(np.float64)
        x, info = gcrotmk_xp(
            matvec,
            b,
            rtol=1e-9,
            atol=0.0,
            maxiter=80,
            m=12,
            k=3,
            CU=cu,
            truncate="oldest",
            xp=np,
        )
        assert int(info) == 0
        assert np.linalg.norm(a @ np.asarray(x, dtype=np.float64) - b) < (1e-7 * np.linalg.norm(b) + 1e-12)
        assert len(cu) <= 3

    assert len(cu) > 0
    for c, u in cu:
        assert isinstance(u, np.ndarray)
        assert (c is None) or isinstance(c, np.ndarray)


@pytest.mark.cuda
def test_gcrotmk_xp_cupy_converges_and_recycles():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    a_np, b_np = _random_system(34, seed=31)
    a_cp = cp.asarray(a_np, dtype=cp.float64)
    b_cp = cp.asarray(b_np, dtype=cp.float64)

    def matvec(x):
        xx = cp.asarray(x, dtype=cp.float64).ravel()
        return a_cp @ xx

    cu: list[tuple[object | None, object]] = []
    x_cp, info = gcrotmk_xp(
        matvec,
        b_cp,
        rtol=1e-10,
        atol=0.0,
        maxiter=80,
        m=14,
        k=4,
        CU=cu,
        truncate="oldest",
        xp=cp,
    )
    assert int(info) == 0
    assert isinstance(x_cp, cp.ndarray)
    assert 0 < len(cu) <= 4
    for c, u in cu:
        assert isinstance(u, cp.ndarray)
        assert (c is None) or isinstance(c, cp.ndarray)

    resid = float(cp.linalg.norm(matvec(x_cp) - b_cp))
    assert resid < (1e-8 * np.linalg.norm(b_np) + 1e-12)


@pytest.mark.cuda
def test_solve_mcscf_zvector_gcrotmk_gpu_dispatch():
    cp = pytest.importorskip("cupy")
    try:
        _ = cp.zeros((1,), dtype=cp.float64)
    except Exception:
        pytest.skip("CuPy is present but a CUDA device is unavailable")

    from asuka.mcscf.zvector import MCSCFHessianOp, solve_mcscf_zvector

    rng = np.random.default_rng(77)
    n = 26
    a_np = rng.normal(size=(n, n)).astype(np.float64)
    a_np += np.eye(n, dtype=np.float64) * float(n + 6)
    a_cp = cp.asarray(a_np, dtype=cp.float64)

    def mv(x):
        if isinstance(x, cp.ndarray):
            return a_cp @ x
        xx = np.asarray(x, dtype=np.float64).ravel()
        return a_np @ xx

    def _ci_unflatten(v: np.ndarray):
        return np.asarray(v, dtype=np.float64).ravel()

    op = MCSCFHessianOp(
        mv=mv,
        diag=np.diag(a_np).copy(),
        n_orb=n,
        n_ci=0,
        ci_template=np.zeros((0,), dtype=np.float64),
        ci_unflatten=_ci_unflatten,
        orb_only=True,
        is_sa=False,
        ci_ref_list=None,
        sa_gram_inv=None,
        gpu_mode=True,
    )

    x_true = rng.normal(size=(n,)).astype(np.float64)
    rhs_orb = -(a_np @ x_true)
    recycle_space: list[tuple[np.ndarray | None, np.ndarray]] = []

    class _DummyMC:
        pass

    res = solve_mcscf_zvector(
        _DummyMC(),
        hessian_op=op,
        rhs_orb=rhs_orb,
        method="gcrotmk",
        tol=1e-10,
        maxiter=80,
        restart=14,
        gcrotmk_k=6,
        recycle_space=recycle_space,
        auto_rdm_backend_cuda=False,
    )

    assert bool(res.converged)
    assert str(res.info.get("solver", "")) == "gcrotmk_gpu"
    assert str(res.info.get("backend", "")) == "cuda"
    np.testing.assert_allclose(res.z_orb, x_true, rtol=1e-6, atol=1e-7)

    assert len(recycle_space) > 0
    assert any(isinstance(u, cp.ndarray) for _c, u in recycle_space)
