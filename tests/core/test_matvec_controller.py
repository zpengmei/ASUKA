"""Tests for the new GugaMatvecController + CudaGugaExecutor split."""

import pytest
import numpy as np

pytestmark = pytest.mark.cuda

try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    from asuka.cuda import cuda_backend as cb

    _ext = cb._ext
    HAS_EXT = _ext is not None
except Exception:
    HAS_EXT = False

try:
    from asuka.cuguga.drt import DRT, build_drt

    HAS_GUGA = True
except ImportError:
    HAS_GUGA = False

skipif_no_cuda = pytest.mark.skipif(
    not (HAS_CUPY and HAS_EXT and HAS_GUGA),
    reason="CUDA extension, CuPy, or GUGA module not available",
)


def _make_test_inputs(norb, nelec, twos_target=0, use_df=False):
    """Build DRT + random integrals for testing."""
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos_target)
    rng = np.random.default_rng(42)
    nops = norb * norb

    h_eff_h = rng.standard_normal((norb, norb))
    h_eff_h = 0.5 * (h_eff_h + h_eff_h.T)

    naux = max(norb * 3, 20)
    l_raw = rng.standard_normal((norb, norb, naux))
    l_sym = 0.5 * (l_raw + l_raw.transpose(1, 0, 2))
    l_full_h = l_sym.reshape(nops, naux)
    eri_h = l_full_h @ l_full_h.T

    return drt, eri_h, l_full_h, h_eff_h


@skipif_no_cuda
@pytest.mark.parametrize("norb,nelec", [(4, 4), (6, 4), (5, 6)])
def test_controller_matches_workspace_dense(norb, nelec):
    """Controller + CudaGugaExecutor must give same result as GugaMatvecEriMatWorkspace."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor

    drt, eri_h, l_full_h, h_eff_h = _make_test_inputs(norb, nelec)
    ncsf = drt.ncsf

    eri_d = cp.asarray(eri_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    rng = np.random.default_rng(123)
    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    # Reference: old workspace (epq_blocked aggregate path)
    ws = cb.GugaMatvecEriMatWorkspace(
        drt,
        eri_mat=eri_d.copy(),
        h_eff=h_eff_d.copy(),
        aggregate_offdiag_k=True,
        use_epq_table=True,
        epq_build_device=True,
    )
    y_ref = ws.hop(x_d.copy(), sync=True, check_overflow=True)
    y_ref_h = cp.asnumpy(y_ref)

    # New: controller + executor
    executor = CudaGugaExecutor(
        drt,
        eri_mat=eri_d.copy(),
        h_eff=h_eff_d.copy(),
        epq_build_device=True,
    )
    controller = GugaMatvecController(executor, include_diagonal_rs=True)
    y_new = controller.hop(x_d.copy(), sync=True, check_overflow=True)
    y_new_h = cp.asnumpy(y_new)

    max_err = np.max(np.abs(y_ref_h - y_new_h))
    rel_err = max_err / (np.max(np.abs(y_ref_h)) + 1e-30)
    print(f"norb={norb} nelec={nelec} ncsf={ncsf}: max_err={max_err:.2e} rel_err={rel_err:.2e}")
    assert max_err < 1e-10, f"Dense matvec mismatch: max_err={max_err:.2e}"


@skipif_no_cuda
@pytest.mark.parametrize("norb,nelec", [(4, 4), (6, 4)])
def test_controller_matches_workspace_df(norb, nelec):
    """Controller + CudaGugaExecutor (DF path) must match workspace."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor

    drt, eri_h, l_full_h, h_eff_h = _make_test_inputs(norb, nelec, use_df=True)
    ncsf = drt.ncsf

    l_d = cp.asarray(l_full_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    rng = np.random.default_rng(123)
    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    # Reference: old workspace (DF aggregate)
    ws = cb.GugaMatvecEriMatWorkspace(
        drt,
        l_full=l_d.copy(),
        h_eff=h_eff_d.copy(),
        aggregate_offdiag_k=True,
        use_epq_table=True,
        epq_build_device=True,
    )
    y_ref = ws.hop(x_d.copy(), sync=True, check_overflow=True)
    y_ref_h = cp.asnumpy(y_ref)

    # New: controller + executor (DF)
    executor = CudaGugaExecutor(
        drt,
        l_full=l_d.copy(),
        h_eff=h_eff_d.copy(),
        epq_build_device=True,
    )
    controller = GugaMatvecController(executor, include_diagonal_rs=True)
    y_new = controller.hop(x_d.copy(), sync=True, check_overflow=True)
    y_new_h = cp.asnumpy(y_new)

    max_err = np.max(np.abs(y_ref_h - y_new_h))
    rel_err = max_err / (np.max(np.abs(y_ref_h)) + 1e-30)
    print(f"DF norb={norb} nelec={nelec} ncsf={ncsf}: max_err={max_err:.2e} rel_err={rel_err:.2e}")
    assert max_err < 1e-10, f"DF matvec mismatch: max_err={max_err:.2e}"


@skipif_no_cuda
def test_controller_hermitian():
    """Verify <u, Hv> == <Hu, v> through the controller path."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor

    norb, nelec = 5, 4
    drt, eri_h, l_full_h, h_eff_h = _make_test_inputs(norb, nelec)

    eri_d = cp.asarray(eri_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    executor = CudaGugaExecutor(
        drt, eri_mat=eri_d, h_eff=h_eff_d, epq_build_device=True,
    )
    controller = GugaMatvecController(executor)

    rng = np.random.default_rng(77)
    u = cp.asarray(rng.standard_normal(drt.ncsf), dtype=cp.float64)
    v = cp.asarray(rng.standard_normal(drt.ncsf), dtype=cp.float64)

    Hu = controller.hop(u, sync=True)
    Hv = controller.hop(v, sync=True)

    lhs = float(cp.dot(u, Hv))
    rhs = float(cp.dot(Hu, v))
    diff = abs(lhs - rhs)
    print(f"Hermitian check: <u,Hv>={lhs:.12f}  <Hu,v>={rhs:.12f}  diff={diff:.2e}")
    assert diff < 1e-10, f"Hermiticity violated: diff={diff:.2e}"


@skipif_no_cuda
def test_protocol_isinstance():
    """CudaGugaExecutor satisfies GugaExecutorProtocol at runtime."""
    from asuka.cuguga.matvec.protocol import GugaExecutorProtocol
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor

    norb, nelec = 4, 4
    drt, eri_h, _, h_eff_h = _make_test_inputs(norb, nelec)

    executor = CudaGugaExecutor(
        drt,
        eri_mat=cp.asarray(eri_h, dtype=cp.float64),
        h_eff=cp.asarray(h_eff_h, dtype=cp.float64),
        epq_build_device=True,
    )
    assert isinstance(executor, GugaExecutorProtocol)


@skipif_no_cuda
@pytest.mark.parametrize("use_df", [False, True], ids=["dense", "df"])
def test_workspace_delegation_via_env(use_df, monkeypatch):
    """GugaMatvecEriMatWorkspace delegates to controller when ASUKA_MATVEC_USE_CONTROLLER=1."""
    monkeypatch.setenv("ASUKA_MATVEC_USE_CONTROLLER", "1")

    norb, nelec = 5, 4
    drt, eri_h, l_full_h, h_eff_h = _make_test_inputs(norb, nelec, use_df=use_df)
    ncsf = drt.ncsf

    rng = np.random.default_rng(99)
    x_h = rng.standard_normal(ncsf)
    x_d = cp.asarray(x_h, dtype=cp.float64)

    kwargs = dict(
        h_eff=cp.asarray(h_eff_h, dtype=cp.float64),
        aggregate_offdiag_k=True,
        use_epq_table=True,
        epq_build_device=True,
    )
    if use_df:
        kwargs["l_full"] = cp.asarray(l_full_h, dtype=cp.float64)
    else:
        kwargs["eri_mat"] = cp.asarray(eri_h, dtype=cp.float64)

    ws = cb.GugaMatvecEriMatWorkspace(drt, **kwargs)

    # First call: creates controller lazily
    y1 = ws.hop(x_d.copy(), sync=True, check_overflow=True)
    assert hasattr(ws, "_controller")

    # Second call: reuses controller
    y2 = ws.hop(x_d.copy(), sync=True, check_overflow=True)

    max_err = float(cp.max(cp.abs(y1 - y2)))
    assert max_err < 1e-13, f"Repeated calls differ: {max_err:.2e}"

    # Compare against non-delegation path
    monkeypatch.delenv("ASUKA_MATVEC_USE_CONTROLLER")
    ws2 = cb.GugaMatvecEriMatWorkspace(drt, **kwargs)
    y_ref = ws2.hop(x_d.copy(), sync=True, check_overflow=True)

    y1_h = cp.asnumpy(y1)
    y_ref_h = cp.asnumpy(y_ref)
    max_err = float(np.max(np.abs(y1_h - y_ref_h)))
    mode = "DF" if use_df else "Dense"
    print(f"Delegation {mode}: max_err={max_err:.2e}")
    assert max_err < 1e-10, f"Delegation mismatch: {max_err:.2e}"


# ======================================================================
# Step 3: Optimization layer tests
# ======================================================================


@skipif_no_cuda
@pytest.mark.parametrize("norb,nelec", [(4, 4), (6, 4)])
def test_sym_pair_executor_matches_base(norb, nelec):
    """SymPairExecutor must produce the same result as the base executor."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor
    from asuka.cuguga.matvec.cuda_optimizations import SymPairExecutor

    drt, eri_h, _, h_eff_h = _make_test_inputs(norb, nelec)
    eri_d = cp.asarray(eri_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    rng = np.random.default_rng(77)
    x_d = cp.asarray(rng.standard_normal(drt.ncsf), dtype=cp.float64)

    # Base path
    base = CudaGugaExecutor(drt, eri_mat=eri_d.copy(), h_eff=h_eff_d.copy(), epq_build_device=True)
    ctrl_base = GugaMatvecController(base)
    y_base = ctrl_base.hop(x_d.copy(), sync=True)

    # Sym-pair path
    base2 = CudaGugaExecutor(drt, eri_mat=eri_d.copy(), h_eff=h_eff_d.copy(), epq_build_device=True)
    opt = SymPairExecutor(base2)
    ctrl_opt = GugaMatvecController(opt)
    y_opt = ctrl_opt.hop(x_d.copy(), sync=True)

    y_base_h = cp.asnumpy(y_base)
    y_opt_h = cp.asnumpy(y_opt)
    max_err = np.max(np.abs(y_base_h - y_opt_h))
    print(f"SymPair norb={norb} nelec={nelec}: max_err={max_err:.2e}")
    assert max_err < 1e-10, f"SymPair mismatch: {max_err:.2e}"


@skipif_no_cuda
def test_transpose_guard_executor_works():
    """TransposeGuardExecutor builds transpose and warns on low memory."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor
    from asuka.cuguga.matvec.cuda_optimizations import TransposeGuardExecutor

    norb, nelec = 4, 4
    drt, eri_h, _, h_eff_h = _make_test_inputs(norb, nelec)
    eri_d = cp.asarray(eri_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    base = CudaGugaExecutor(drt, eri_mat=eri_d, h_eff=h_eff_d, epq_build_device=True)
    guarded = TransposeGuardExecutor(base, reserve_mib=512)
    ctrl = GugaMatvecController(guarded)

    x_d = cp.asarray(np.random.default_rng(42).standard_normal(drt.ncsf), dtype=cp.float64)
    y = ctrl.hop(x_d, sync=True)
    assert y.shape == (drt.ncsf,)


@skipif_no_cuda
def test_make_optimized_executor_composes():
    """make_optimized_executor composes sym-pair + transpose guard."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor
    from asuka.cuguga.matvec.cuda_optimizations import make_optimized_executor

    norb, nelec = 5, 4
    drt, eri_h, _, h_eff_h = _make_test_inputs(norb, nelec)
    eri_d = cp.asarray(eri_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    rng = np.random.default_rng(55)
    x_d = cp.asarray(rng.standard_normal(drt.ncsf), dtype=cp.float64)

    # Base reference
    base_ref = CudaGugaExecutor(drt, eri_mat=eri_d.copy(), h_eff=h_eff_d.copy(), epq_build_device=True)
    y_ref = GugaMatvecController(base_ref).hop(x_d.copy(), sync=True)

    # Composed optimizations
    base_opt = CudaGugaExecutor(drt, eri_mat=eri_d.copy(), h_eff=h_eff_d.copy(), epq_build_device=True)
    opt = make_optimized_executor(base_opt, sym_pair=True, transpose_guard=True)
    y_opt = GugaMatvecController(opt).hop(x_d.copy(), sync=True)

    max_err = float(np.max(np.abs(cp.asnumpy(y_ref) - cp.asnumpy(y_opt))))
    print(f"Composed optimizations: max_err={max_err:.2e}")
    assert max_err < 1e-10, f"Composed optimization mismatch: {max_err:.2e}"


@skipif_no_cuda
def test_sym_pair_hermitian():
    """SymPairExecutor preserves Hermiticity."""
    from asuka.cuguga.matvec.controller import GugaMatvecController
    from asuka.cuguga.matvec.cuda_executor import CudaGugaExecutor
    from asuka.cuguga.matvec.cuda_optimizations import SymPairExecutor

    norb, nelec = 5, 4
    drt, eri_h, _, h_eff_h = _make_test_inputs(norb, nelec)
    eri_d = cp.asarray(eri_h, dtype=cp.float64)
    h_eff_d = cp.asarray(h_eff_h, dtype=cp.float64)

    base = CudaGugaExecutor(drt, eri_mat=eri_d, h_eff=h_eff_d, epq_build_device=True)
    opt = SymPairExecutor(base)
    ctrl = GugaMatvecController(opt)

    rng = np.random.default_rng(77)
    u = cp.asarray(rng.standard_normal(drt.ncsf), dtype=cp.float64)
    v = cp.asarray(rng.standard_normal(drt.ncsf), dtype=cp.float64)

    Hu = ctrl.hop(u, sync=True)
    Hv = ctrl.hop(v, sync=True)

    lhs = float(cp.dot(u, Hv))
    rhs = float(cp.dot(Hu, v))
    diff = abs(lhs - rhs)
    print(f"SymPair Hermitian: diff={diff:.2e}")
    assert diff < 1e-10, f"SymPair Hermiticity violated: diff={diff:.2e}"
