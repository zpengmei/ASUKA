from __future__ import annotations

"""V_xc matrix builder via GPU numerical integration on Becke grid.

Supports meta-GGA functionals (tau-dependent). Builds V_xc in the AO basis
from grid-based XC evaluation.
"""

from typing import Any

import numpy as np

from .eval_xc import eval_xc
from .functional import FunctionalSpec

try:
    import cupy as cp

    _CUDA_OK = True
except Exception:
    cp = None
    _CUDA_OK = False


def _require_cupy():
    if not _CUDA_OK:
        raise RuntimeError("CuPy required for GPU numerical integration")
    return cp


def _contract_rho_on_grid(xp, phi, dphi, D):
    """Compute rho, nabla_rho, tau from AOs and density matrix.

    Parameters
    ----------
    phi : (npt, nao)
    dphi : (npt, nao, 3)
    D : (nao, nao) density matrix (symmetric)

    Returns
    -------
    rho : (npt,)
    sigma : (npt,)   |nabla rho|^2
    tau : (npt,)     0.5 * sum_i |nabla phi_i|^2 * f_i (KS kinetic energy density)
    """
    # phi_D[g, nu] = sum_mu phi[g, mu] * D[mu, nu]
    phi_D = phi @ D  # (npt, nao)

    # rho[g] = sum_nu phi_D[g, nu] * phi[g, nu]
    rho = xp.sum(phi_D * phi, axis=1)  # (npt,)

    # nabla_rho[g, xyz] = 2 * sum_nu phi_D[g, nu] * dphi[g, nu, xyz]
    # Using einsum for clarity
    nabla_rho = 2.0 * xp.einsum("gn,gnx->gx", phi_D, dphi)  # (npt, 3)

    # sigma = |nabla_rho|^2
    sigma = xp.sum(nabla_rho * nabla_rho, axis=1)  # (npt,)

    # tau = 0.5 * sum_mu,nu D[mu,nu] * sum_xyz dphi[g,mu,xyz] * dphi[g,nu,xyz]
    # = 0.5 * sum_xyz sum_nu (sum_mu dphi[g,mu,xyz]*D[mu,nu]) * dphi[g,nu,xyz]
    # dphi_D[g, nu, xyz] = sum_mu dphi[g, mu, xyz] * D[mu, nu]
    # But D is symmetric so dphi @ D works per xyz slice
    # More efficient: tau = 0.5 * sum_xyz sum_nu dphi_D[g,nu,xyz] * dphi[g,nu,xyz]
    # dphi is (npt, nao, 3), need dphi_D[g,nu,xyz] = sum_mu D[mu,nu] * dphi[g,mu,xyz]
    # Reshape for batch matmul
    npt, nao, _ = dphi.shape
    # dphi_2d: (npt*3, nao) -> matmul with D -> (npt*3, nao) -> reshape back
    dphi_2d = dphi.transpose(0, 2, 1).reshape(npt * 3, nao)  # (npt*3, nao)
    dphi_D_2d = dphi_2d @ D  # (npt*3, nao)
    dphi_D = dphi_D_2d.reshape(npt, 3, nao).transpose(0, 2, 1)  # (npt, nao, 3)

    tau = 0.5 * xp.sum(dphi_D * dphi, axis=(1, 2))  # (npt,)

    return rho, sigma, tau, nabla_rho, phi_D


def _build_vxc_batch(xp, phi, dphi, weights, vrho, vsigma, vtau, nabla_rho, phi_D):
    """Build V_xc contribution from a batch of grid points.

    Parameters
    ----------
    phi : (npt, nao)
    dphi : (npt, nao, 3)
    weights : (npt,)
    vrho, vsigma, vtau : (npt,) XC potential components
    nabla_rho : (npt, 3)
    phi_D : (npt, nao) = phi @ D (precomputed)

    Returns
    -------
    V_xc : (nao, nao) symmetric contribution to V_xc matrix
    """
    npt, nao = phi.shape

    # Weight the potentials
    w_vrho = weights * vrho  # (npt,)

    # --- LDA contribution ---
    # V_lda[mu,nu] = sum_g w*vrho * phi[g,mu] * phi[g,nu]
    # = (phi * w_vrho[:, None])^T @ phi
    wphi = phi * w_vrho[:, None]  # (npt, nao)
    V = wphi.T @ phi  # (nao, nao)

    # --- GGA contribution ---
    # V_gga[mu,nu] = sum_g 2*w*vsigma * [dphi[g,mu,:].nabla_rho[g,:] * phi[g,nu]
    #                                    + phi[g,mu] * dphi[g,nu,:].nabla_rho[g,:]]
    # = 2 * sum_g w*vsigma * chi[g,mu] * phi[g,nu]  +  transpose
    # where chi[g,mu] = sum_xyz dphi[g,mu,xyz] * nabla_rho[g,xyz]
    w_vsigma = weights * vsigma  # (npt,)
    # chi[g, mu] = sum_xyz dphi[g,mu,xyz] * nabla_rho[g,xyz]
    chi = xp.einsum("gnx,gx->gn", dphi, nabla_rho)  # (npt, nao)
    wchi = chi * (2.0 * w_vsigma)[:, None]  # (npt, nao)
    V_gga = wchi.T @ phi  # (nao, nao)
    V += V_gga + V_gga.T

    # --- meta-GGA (tau) contribution ---
    # V_tau[mu,nu] = sum_g 0.5*w*vtau * sum_xyz dphi[g,mu,xyz]*dphi[g,nu,xyz]
    w_vtau = weights * vtau * 0.5  # (npt,)
    # For each xyz: dphi_xyz^T @ diag(w_vtau) @ dphi_xyz
    for d in range(3):
        dphi_d = dphi[:, :, d]  # (npt, nao)
        wdphi = dphi_d * w_vtau[:, None]  # (npt, nao)
        V += wdphi.T @ dphi_d  # (nao, nao)

    return V


def build_vxc(
    spec: FunctionalSpec,
    D: Any,
    ao_basis: Any,
    grid_coords: Any,
    grid_weights: Any,
    *,
    batch_size: int = 50000,
    sph_transform: Any | None = None,
) -> tuple[Any, float]:
    """Build V_xc matrix and E_xc from density matrix on Becke grid.

    Parameters
    ----------
    spec : FunctionalSpec
        XC functional.
    D : (nao, nao) array
        Density matrix (in AO basis, Cartesian or spherical depending on sph_transform).
    ao_basis : object
        Packed AO basis (Cartesian) with shell_l, shell_ao_start, etc.
    grid_coords : (npt_total, 3) array
        Grid coordinates in Bohr.
    grid_weights : (npt_total,) array
        Grid weights (including Becke partition).
    batch_size : int
        Grid points per batch (controls VRAM).
    sph_transform : (nao_cart, nao_sph) array or None
        If provided, D is in spherical basis and must be transformed.

    Returns
    -------
    V_xc : (nao, nao) array
        XC potential matrix in the same basis as D.
    E_xc : float
        Total XC energy.
    """
    xp = _require_cupy()
    from asuka.orbitals.eval_basis_device import eval_aos_cart_value_grad_on_points_device

    npt_total = int(grid_coords.shape[0])
    if sph_transform is not None:
        T = xp.asarray(sph_transform, dtype=xp.float64)
        nao_sph = int(T.shape[1])
        # Transform D from spherical to Cartesian for AO evaluation
        D_cart = T @ D @ T.T
        nao = nao_sph
    else:
        D_cart = D
        nao = int(D.shape[0])

    D_cart = xp.ascontiguousarray(xp.asarray(D_cart, dtype=xp.float64))
    V_xc_cart = xp.zeros((int(D_cart.shape[0]), int(D_cart.shape[0])), dtype=xp.float64)
    E_xc = 0.0

    for p0 in range(0, npt_total, batch_size):
        p1 = min(npt_total, p0 + batch_size)
        pts = xp.ascontiguousarray(grid_coords[p0:p1])
        wts = grid_weights[p0:p1]

        # Evaluate AOs and gradients on this batch
        phi, dphi = eval_aos_cart_value_grad_on_points_device(ao_basis, pts)
        # phi: (nbatch, nao_cart), dphi: (nbatch, nao_cart, 3)

        if sph_transform is not None:
            # Transform AOs to spherical: phi_sph = phi_cart @ T
            phi_use = phi @ T  # (nbatch, nao_sph)
            dphi_use = xp.einsum("gnx,ns->gsx", dphi, T).transpose(0, 2, 1)
            # Actually: dphi_sph[g, s, xyz] = sum_n dphi[g,n,xyz] * T[n,s]
            # dphi is (npt, ncart, 3), T is (ncart, nsph)
            # dphi_use[g, s, xyz] = sum_n dphi[g, n, xyz] * T[n, s]
            dphi_use = xp.einsum("gnx,ns->gsx", dphi, T)  # (npt, nsph, 3)
            D_use = D
        else:
            phi_use = phi
            dphi_use = dphi
            D_use = D_cart

        D_use = xp.asarray(D_use, dtype=xp.float64)

        # Contract density quantities
        rho, sigma, tau_val, nabla_rho, phi_D = _contract_rho_on_grid(
            xp, phi_use, dphi_use, D_use,
        )

        # Evaluate XC
        exc, vrho, vsigma, vtau = eval_xc(spec, rho, sigma, tau_val)

        # Accumulate E_xc
        E_xc += float(xp.sum(wts * exc * rho).item())

        # Build V_xc contribution
        V_batch = _build_vxc_batch(
            xp, phi_use, dphi_use, wts, vrho, vsigma, vtau, nabla_rho, phi_D,
        )

        if sph_transform is not None:
            V_xc_sph = V_batch if not hasattr(V_batch, "get") else V_batch
            if p0 == 0:
                V_xc_out = V_batch.copy()
            else:
                V_xc_out += V_batch
        else:
            V_xc_cart += V_batch

    if sph_transform is not None:
        V_xc = V_xc_out
    else:
        V_xc = V_xc_cart

    # Symmetrize
    V_xc = 0.5 * (V_xc + V_xc.T)

    return V_xc, E_xc


__all__ = ["build_vxc"]
