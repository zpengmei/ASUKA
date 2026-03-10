from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

try:
    _HAS_CUDA = int(cp.cuda.runtime.getDeviceCount()) > 0
except Exception:  # pragma: no cover
    _HAS_CUDA = False

pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required"),
]


def test_fused_contract_density_matches_gemm() -> None:
    from asuka.xc.cuda_kernels import ensure_fused_kernels_compiled, fused_contract_density
    from asuka.xc.numint import _contract_rho_on_grid

    ensure_fused_kernels_compiled()

    rng = np.random.default_rng(0)
    npt = 256
    nao = 37
    phi = cp.asarray(rng.normal(size=(npt, nao)), dtype=cp.float64)
    dphi = cp.asarray(rng.normal(size=(npt, nao, 3)), dtype=cp.float64)
    A = cp.asarray(rng.normal(size=(nao, nao)), dtype=cp.float64)
    D = 0.5 * (A + A.T)

    rho_f, sigma_f, tau_f, nabla_f = fused_contract_density(phi, dphi, D)
    rho_g, sigma_g, tau_g, nabla_g = _contract_rho_on_grid(cp, phi, dphi, D)
    cp.cuda.runtime.deviceSynchronize()

    np.testing.assert_allclose(cp.asnumpy(rho_f), cp.asnumpy(rho_g), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(sigma_f), cp.asnumpy(sigma_g), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(cp.asnumpy(tau_f), cp.asnumpy(tau_g), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(cp.asnumpy(nabla_f), cp.asnumpy(nabla_g), rtol=1e-12, atol=1e-12)


def test_fused_build_vxc_matches_gemm() -> None:
    from asuka.xc.cuda_kernels import ensure_fused_kernels_compiled, fused_build_vxc
    from asuka.xc.numint import _build_vxc_batch

    ensure_fused_kernels_compiled()

    rng = np.random.default_rng(1)
    npt = 192
    nao = 41
    phi = cp.asarray(rng.normal(size=(npt, nao)), dtype=cp.float64)
    dphi = cp.asarray(rng.normal(size=(npt, nao, 3)), dtype=cp.float64)
    weights = cp.asarray(rng.random(size=(npt,)), dtype=cp.float64)
    vrho = cp.asarray(rng.normal(size=(npt,)), dtype=cp.float64)
    vsigma = cp.asarray(rng.normal(size=(npt,)), dtype=cp.float64)
    vtau = cp.asarray(rng.normal(size=(npt,)), dtype=cp.float64)
    nabla_rho = cp.asarray(rng.normal(size=(npt, 3)), dtype=cp.float64)

    V_f = fused_build_vxc(phi, dphi, weights, vrho, vsigma, vtau, nabla_rho)
    V_g = _build_vxc_batch(cp, phi, dphi, weights, vrho, vsigma, vtau, nabla_rho)
    cp.cuda.runtime.deviceSynchronize()

    np.testing.assert_allclose(cp.asnumpy(V_f), cp.asnumpy(V_g), rtol=1e-11, atol=1e-11)
