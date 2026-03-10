from __future__ import annotations

import numpy as np

from asuka.xc.cuda_kernels import resolve_numint_backend
from asuka.xc.numint import _build_vxc_batch, _contract_rho_on_grid


def _naive_contract(phi: np.ndarray, dphi: np.ndarray, D: np.ndarray):
    npt, nao = phi.shape
    rho = np.zeros((npt,), dtype=np.float64)
    sigma = np.zeros((npt,), dtype=np.float64)
    tau = np.zeros((npt,), dtype=np.float64)
    nabla = np.zeros((npt, 3), dtype=np.float64)

    for g in range(npt):
        for mu in range(nao):
            for nu in range(nao):
                Dij = D[mu, nu]
                rho[g] += phi[g, mu] * Dij * phi[g, nu]
                for xyz in range(3):
                    nabla[g, xyz] += 2.0 * phi[g, mu] * Dij * dphi[g, nu, xyz]
                    tau[g] += 0.5 * dphi[g, mu, xyz] * Dij * dphi[g, nu, xyz]
        sigma[g] = float(np.dot(nabla[g], nabla[g]))

    return rho, sigma, tau, nabla


def _naive_vxc(
    phi: np.ndarray,
    dphi: np.ndarray,
    weights: np.ndarray,
    vrho: np.ndarray,
    vsigma: np.ndarray,
    vtau: np.ndarray,
    nabla_rho: np.ndarray,
) -> np.ndarray:
    npt, nao = phi.shape
    V = np.zeros((nao, nao), dtype=np.float64)
    for mu in range(nao):
        for nu in range(nao):
            acc = 0.0
            for g in range(npt):
                phi_mu = phi[g, mu]
                phi_nu = phi[g, nu]
                dmu = dphi[g, mu]
                dnu = dphi[g, nu]
                grad = nabla_rho[g]
                acc += weights[g] * vrho[g] * phi_mu * phi_nu
                acc += 2.0 * weights[g] * vsigma[g] * (
                    float(np.dot(dmu, grad)) * phi_nu + phi_mu * float(np.dot(dnu, grad))
                )
                acc += 0.5 * weights[g] * vtau[g] * float(np.dot(dmu, dnu))
            V[mu, nu] = acc
    return V


def test_contract_rho_on_grid_matches_naive() -> None:
    rng = np.random.default_rng(123)
    npt = 5
    nao = 4
    phi = rng.normal(size=(npt, nao))
    dphi = rng.normal(size=(npt, nao, 3))
    D = rng.normal(size=(nao, nao))
    D = 0.5 * (D + D.T)

    rho, sigma, tau, nabla = _contract_rho_on_grid(np, phi, dphi, D)
    rho_ref, sigma_ref, tau_ref, nabla_ref = _naive_contract(phi, dphi, D)

    np.testing.assert_allclose(rho, rho_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(sigma, sigma_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(tau, tau_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(nabla, nabla_ref, rtol=1e-12, atol=1e-12)


def test_build_vxc_batch_matches_naive() -> None:
    rng = np.random.default_rng(456)
    npt = 6
    nao = 5
    phi = rng.normal(size=(npt, nao))
    dphi = rng.normal(size=(npt, nao, 3))
    weights = rng.random(size=(npt,))
    vrho = rng.normal(size=(npt,))
    vsigma = rng.normal(size=(npt,))
    vtau = rng.normal(size=(npt,))
    nabla_rho = rng.normal(size=(npt, 3))

    V = _build_vxc_batch(np, phi, dphi, weights, vrho, vsigma, vtau, nabla_rho)
    V_ref = _naive_vxc(phi, dphi, weights, vrho, vsigma, vtau, nabla_rho)

    np.testing.assert_allclose(V, V_ref, rtol=1e-12, atol=1e-12)


def test_resolve_numint_backend_auto_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("ASUKA_XC_NUMINT_BACKEND", "auto")
    monkeypatch.setenv("ASUKA_XC_FUSED_NAO_MAX", "32")
    monkeypatch.setenv("ASUKA_XC_FUSED_BATCH_MIN", "1024")

    assert resolve_numint_backend(None, nao=16, batch_size=4096) == "fused"
    assert resolve_numint_backend(None, nao=64, batch_size=4096) == "gemm"
    assert resolve_numint_backend(None, nao=16, batch_size=128) == "gemm"
    assert resolve_numint_backend("gemm", nao=16, batch_size=4096) == "gemm"
    assert resolve_numint_backend("fused", nao=128, batch_size=1) == "fused"
