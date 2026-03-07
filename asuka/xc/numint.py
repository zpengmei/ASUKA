from __future__ import annotations

"""V_xc matrix builder via GPU numerical integration on Becke grids.

This module now supports two CUDA execution modes for the AO-grid contraction
stage:

- ``gemm``: the original CuPy dense-linear-algebra path built from GEMMs and
  pointwise tensor algebra.
- ``fused``: runtime-compiled CUDA kernels that fuse the grid-density
  contraction and the LDA/GGA/meta-GGA ``V_xc`` accumulation.

The default ``auto`` policy chooses the fused path for moderate AO dimensions,
where the reduced temporary traffic tends to dominate. The legacy GEMM path is
retained for larger AO spaces and as a robust fallback.
"""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np

from .eval_xc import eval_xc, eval_xc_sp, eval_xc_u
from .functional import FunctionalSpec

try:
    import cupy as cp

    _CUDA_OK = True
except Exception:  # pragma: no cover
    cp = None
    _CUDA_OK = False


@dataclass(slots=True)
class _NumIntScratch:
    phi_cart: Any | None = None
    dphi_cart: Any | None = None
    rho: Any | None = None
    sigma: Any | None = None
    tau: Any | None = None
    nabla_rho: Any | None = None
    v_batch: Any | None = None


def _require_cupy():
    if not _CUDA_OK:
        raise RuntimeError("CuPy required for GPU numerical integration")
    return cp


def _nao_cart_from_basis(ao_basis: Any) -> int:
    if not hasattr(ao_basis, "shell_ao_start") or not hasattr(ao_basis, "shell_l"):
        raise TypeError("ao_basis must provide shell_ao_start and shell_l")

    from asuka.cueri.cart import ncart

    shell_ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int64).ravel()
    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int64).ravel()
    if shell_ao_start.shape != shell_l.shape:
        raise ValueError("ao_basis.shell_ao_start and ao_basis.shell_l must have identical shape")
    if shell_l.size == 0:
        raise ValueError("ao_basis has no shells")

    nfunc = np.asarray([ncart(int(l)) for l in shell_l], dtype=np.int64)
    return int(np.max(shell_ao_start + nfunc))


def _contract_rho_on_grid(xp, phi, dphi, D):
    """Compute ``rho``, ``nabla rho``, ``sigma``, and ``tau`` on a grid.

    Parameters
    ----------
    phi
        ``(npt, nao)`` AO values.
    dphi
        ``(npt, nao, 3)`` AO gradients.
    D
        ``(nao, nao)`` density matrix.
    """
    phi_D = phi @ D  # (npt, nao)
    rho = xp.sum(phi_D * phi, axis=1)  # (npt,)
    nabla_rho = 2.0 * xp.einsum("gn,gnx->gx", phi_D, dphi)  # (npt, 3)
    sigma = xp.sum(nabla_rho * nabla_rho, axis=1)  # (npt,)

    npt, nao, _ = dphi.shape
    dphi_2d = dphi.transpose(0, 2, 1).reshape(npt * 3, nao)
    dphi_D_2d = dphi_2d @ D
    dphi_D = dphi_D_2d.reshape(npt, 3, nao).transpose(0, 2, 1)
    tau = 0.5 * xp.sum(dphi_D * dphi, axis=(1, 2))

    return rho, sigma, tau, nabla_rho


def _build_vxc_batch(xp, phi, dphi, weights, vrho, vsigma, vtau, nabla_rho):
    """Build one batch contribution to the XC potential matrix.

    This is the legacy dense-linear-algebra path retained as a robust fallback
    and as the preferred backend for larger AO spaces.
    """
    w_vrho = weights * vrho
    V = (phi * w_vrho[:, None]).T @ phi

    w_vsigma = weights * vsigma
    chi = xp.einsum("gnx,gx->gn", dphi, nabla_rho)
    V_gga = (chi * (2.0 * w_vsigma)[:, None]).T @ phi
    V += V_gga + V_gga.T

    w_vtau = weights * vtau * 0.5
    for d in range(3):
        dphi_d = dphi[:, :, d]
        V += (dphi_d * w_vtau[:, None]).T @ dphi_d

    return V


def _build_vxc_batch_u(xp, phi, dphi, weights, vrho, gvec, vtau):
    """Build one batch contribution to a *single-spin* XC potential matrix.

    The input ``gvec`` is the gradient potential vector dE/d(∇rho_sigma) for
    the targeted spin channel, including the appropriate sigma cross-term
    coupling (via ``vsigma_ab``).
    """
    w_vrho = weights * vrho
    V = (phi * w_vrho[:, None]).T @ phi

    wgvec = gvec * weights[:, None]
    chi = xp.einsum("gnx,gx->gn", dphi, wgvec)
    V_gga = chi.T @ phi
    V += V_gga + V_gga.T

    w_vtau = weights * vtau * 0.5
    for d in range(3):
        dphi_d = dphi[:, :, d]
        V += (dphi_d * w_vtau[:, None]).T @ dphi_d

    return V


def _resolve_backend(backend: str | None, *, nao: int, batch_size: int) -> str:
    try:
        from .cuda_kernels import resolve_numint_backend
    except Exception:
        if backend is not None and str(backend).strip().lower() == "fused":
            raise
        return "gemm"

    return resolve_numint_backend(backend, nao=int(nao), batch_size=int(batch_size))


def _resolve_fused_ops(backend: str) -> tuple[bool, Any | None, Any | None]:
    if backend != "fused":
        return False, None, None
    try:
        from .cuda_kernels import (
            ensure_fused_kernels_compiled,
            fused_build_vxc,
            fused_contract_density,
        )

        ensure_fused_kernels_compiled()
        return True, fused_contract_density, fused_build_vxc
    except Exception:
        return False, None, None


def build_vxc(
    spec: FunctionalSpec,
    D: Any,
    ao_basis: Any,
    grid_coords: Any,
    grid_weights: Any,
    *,
    batch_size: int = 50000,
    sph_transform: Any | None = None,
    backend: str | None = None,
) -> tuple[Any, float]:
    """Build ``V_xc`` and ``E_xc`` from a density matrix on a Becke grid.

    Parameters
    ----------
    spec
        XC functional.
    D
        Density matrix in the same AO representation as the returned ``V_xc``.
    ao_basis
        Packed Cartesian AO basis used by the GPU AO evaluator.
    grid_coords
        ``(npt_total, 3)`` grid coordinates in Bohr.
    grid_weights
        ``(npt_total,)`` integration weights.
    batch_size
        Grid points processed per batch.
    sph_transform
        Optional Cartesian-to-spherical AO transform. When provided, the AO
        values/gradients are transformed on the device before the density and
        XC contractions.
    backend
        ``"auto"``, ``"fused"``, or ``"gemm"``. ``None`` defers to the
        ``ASUKA_XC_NUMINT_BACKEND`` environment variable and otherwise defaults
        to ``"auto"``.
    """
    xp = _require_cupy()
    from asuka.orbitals.eval_basis_device import eval_aos_cart_value_grad_on_points_device

    if sph_transform is not None:
        T = xp.ascontiguousarray(xp.asarray(sph_transform, dtype=xp.float64))
        if T.ndim != 2:
            raise ValueError("sph_transform must be a 2D array")
        D_use = xp.ascontiguousarray(xp.asarray(D, dtype=xp.float64))
        if D_use.ndim != 2 or int(D_use.shape[0]) != int(D_use.shape[1]):
            raise ValueError("D must be a square matrix")
        nao = int(D_use.shape[0])
        if int(T.shape[1]) != nao:
            raise ValueError(
                f"sph_transform second dimension ({int(T.shape[1])}) must match D ({nao})"
            )
        nao_cart = int(T.shape[0])
    else:
        T = None
        D_use = xp.ascontiguousarray(xp.asarray(D, dtype=xp.float64))
        if D_use.ndim != 2 or int(D_use.shape[0]) != int(D_use.shape[1]):
            raise ValueError("D must be a square matrix")
        nao = int(D_use.shape[0])
        nao_cart = int(_nao_cart_from_basis(ao_basis))
        if nao_cart != nao:
            raise ValueError(
                f"AO basis Cartesian dimension ({nao_cart}) does not match density matrix ({nao})"
            )

    batch_size = max(1, int(batch_size))
    pts_all = xp.ascontiguousarray(xp.asarray(grid_coords, dtype=xp.float64).reshape((-1, 3)))
    w_all = xp.ascontiguousarray(xp.asarray(grid_weights, dtype=xp.float64).ravel())

    npt_total = int(pts_all.shape[0])
    if tuple(map(int, pts_all.shape)) != (npt_total, 3):
        raise ValueError("grid_coords must have shape (npt_total, 3)")
    if tuple(map(int, w_all.shape)) != (npt_total,):
        raise ValueError("grid_weights must have shape (npt_total,)")

    if nao <= 0:
        raise ValueError("Density matrix has zero dimension")
    if npt_total == 0:
        return xp.zeros((nao, nao), dtype=xp.float64), 0.0

    resolved_backend = _resolve_backend(backend, nao=nao, batch_size=batch_size)
    use_fused, fused_contract_density, fused_build_vxc = _resolve_fused_ops(resolved_backend)
    explicit_backend = backend if backend is not None else os.environ.get("ASUKA_XC_NUMINT_BACKEND", "auto")
    explicit_mode = str(explicit_backend).strip().lower()
    if explicit_mode in {"", "default"}:
        explicit_mode = "auto"
    if resolved_backend == "fused" and not use_fused:
        if explicit_mode == "fused":
            raise RuntimeError(
                "Fused XC backend was requested but the CUDA kernels could not be compiled or loaded."
            )
        resolved_backend = "gemm"
        use_fused = False
        fused_contract_density = None
        fused_build_vxc = None

    max_batch = min(batch_size, npt_total)
    scratch = _NumIntScratch(
        phi_cart=xp.empty((max_batch, nao_cart), dtype=xp.float64),
        dphi_cart=xp.empty((max_batch, nao_cart, 3), dtype=xp.float64),
        rho=xp.empty((max_batch,), dtype=xp.float64) if use_fused else None,
        sigma=xp.empty((max_batch,), dtype=xp.float64) if use_fused else None,
        tau=xp.empty((max_batch,), dtype=xp.float64) if use_fused else None,
        nabla_rho=xp.empty((max_batch, 3), dtype=xp.float64) if use_fused else None,
        v_batch=xp.empty((nao, nao), dtype=xp.float64) if use_fused else None,
    )

    V_xc = xp.zeros((nao, nao), dtype=xp.float64)
    E_xc = xp.zeros((), dtype=xp.float64)

    for p0 in range(0, npt_total, batch_size):
        p1 = min(npt_total, p0 + batch_size)
        nbatch = int(p1 - p0)
        pts = pts_all[p0:p1]
        wts = w_all[p0:p1]

        phi_cart, dphi_cart = eval_aos_cart_value_grad_on_points_device(
            ao_basis,
            pts,
            out=scratch.phi_cart[:nbatch],
            out_grad=scratch.dphi_cart[:nbatch],
            sync=False,
        )

        if T is not None:
            phi_use = xp.ascontiguousarray(phi_cart @ T)
            dphi_use = xp.ascontiguousarray(xp.einsum("gnx,ns->gsx", dphi_cart, T))
        else:
            phi_use = phi_cart
            dphi_use = dphi_cart

        if use_fused:
            rho, sigma, tau_val, nabla_rho = fused_contract_density(
                phi_use,
                dphi_use,
                D_use,
                out_rho=scratch.rho[:nbatch],
                out_sigma=scratch.sigma[:nbatch],
                out_tau=scratch.tau[:nbatch],
                out_nabla=scratch.nabla_rho[:nbatch],
            )
        else:
            rho, sigma, tau_val, nabla_rho = _contract_rho_on_grid(xp, phi_use, dphi_use, D_use)

        exc, vrho, vsigma, vtau = eval_xc(spec, rho, sigma, tau_val)
        E_xc[...] += xp.sum(wts * exc * rho)

        if use_fused:
            V_batch = fused_build_vxc(
                phi_use,
                dphi_use,
                wts,
                vrho,
                vsigma,
                vtau,
                nabla_rho,
                out=scratch.v_batch,
            )
        else:
            V_batch = _build_vxc_batch(
                xp,
                phi_use,
                dphi_use,
                wts,
                vrho,
                vsigma,
                vtau,
                nabla_rho,
            )
        xp.add(V_xc, V_batch, out=V_xc)

    V_xc = 0.5 * (V_xc + V_xc.T)

    return V_xc, float(E_xc.item())


def build_vxc_u(
    spec: FunctionalSpec,
    Da: Any,
    Db: Any,
    ao_basis: Any,
    grid_coords: Any,
    grid_weights: Any,
    *,
    batch_size: int = 50000,
    sph_transform: Any | None = None,
) -> tuple[Any, Any, float]:
    """Spin-polarized V_xc matrices and E_xc for UKS (full spin form).

    Parameters
    ----------
    Da, Db : (nao, nao) alpha- and beta-spin density matrices
    ao_basis, grid_coords, grid_weights, batch_size, sph_transform : same as build_vxc

    Returns
    -------
    V_xc_a : (nao, nao) alpha-spin XC potential
    V_xc_b : (nao, nao) beta-spin XC potential
    E_xc : float total XC energy
    """
    xp = _require_cupy()
    from asuka.orbitals.eval_basis_device import eval_aos_cart_value_grad_on_points_device

    npt_total = int(grid_coords.shape[0])
    if sph_transform is not None:
        T = xp.asarray(sph_transform, dtype=xp.float64)
        Da_cart = T @ xp.asarray(Da, dtype=xp.float64) @ T.T
        Db_cart = T @ xp.asarray(Db, dtype=xp.float64) @ T.T
    else:
        Da_cart = xp.asarray(Da, dtype=xp.float64)
        Db_cart = xp.asarray(Db, dtype=xp.float64)

    Da_cart = xp.ascontiguousarray(Da_cart)
    Db_cart = xp.ascontiguousarray(Db_cart)
    nao_cart = int(Da_cart.shape[0])

    V_a_cart = xp.zeros((nao_cart, nao_cart), dtype=xp.float64)
    V_b_cart = xp.zeros((nao_cart, nao_cart), dtype=xp.float64)
    E_xc = 0.0

    for p0 in range(0, npt_total, batch_size):
        p1 = min(npt_total, p0 + batch_size)
        pts = xp.ascontiguousarray(grid_coords[p0:p1])
        wts = grid_weights[p0:p1]

        phi, dphi = eval_aos_cart_value_grad_on_points_device(ao_basis, pts)

        if sph_transform is not None:
            T_b = xp.asarray(sph_transform, dtype=xp.float64)
            phi_use = phi @ T_b
            dphi_use = xp.einsum("gnx,ns->gsx", dphi, T_b)
            Da_use = xp.asarray(Da, dtype=xp.float64)
            Db_use = xp.asarray(Db, dtype=xp.float64)
        else:
            phi_use = phi
            dphi_use = dphi
            Da_use = Da_cart
            Db_use = Db_cart

        # Contract alpha and beta densities separately
        rho_a, sigma_aa, tau_a, nabla_rho_a = _contract_rho_on_grid(
            xp, phi_use, dphi_use, Da_use,
        )
        rho_b, sigma_bb, tau_b, nabla_rho_b = _contract_rho_on_grid(
            xp, phi_use, dphi_use, Db_use,
        )

        sigma_ab = xp.sum(nabla_rho_a * nabla_rho_b, axis=1)

        exc, vrho_a, vrho_b, vsigma_aa, vsigma_ab, vsigma_bb, vtau_a, vtau_b = eval_xc_sp(
            spec, rho_a, rho_b, sigma_aa, sigma_ab, sigma_bb, tau_a, tau_b,
        )

        rho_tot = rho_a + rho_b
        E_xc += float(xp.sum(wts * exc * rho_tot).item())

        gvec_a = 2.0 * vsigma_aa[:, None] * nabla_rho_a + vsigma_ab[:, None] * nabla_rho_b
        gvec_b = 2.0 * vsigma_bb[:, None] * nabla_rho_b + vsigma_ab[:, None] * nabla_rho_a

        V_a_batch = _build_vxc_batch_u(xp, phi_use, dphi_use, wts, vrho_a, gvec_a, vtau_a)
        V_b_batch = _build_vxc_batch_u(xp, phi_use, dphi_use, wts, vrho_b, gvec_b, vtau_b)

        if sph_transform is not None:
            if p0 == 0:
                V_a_out = V_a_batch.copy()
                V_b_out = V_b_batch.copy()
            else:
                V_a_out += V_a_batch
                V_b_out += V_b_batch
        else:
            V_a_cart += V_a_batch
            V_b_cart += V_b_batch

    if sph_transform is not None:
        V_a = V_a_out
        V_b = V_b_out
    else:
        V_a = V_a_cart
        V_b = V_b_cart

    V_a = 0.5 * (V_a + V_a.T)
    V_b = 0.5 * (V_b + V_b.T)

    return V_a, V_b, E_xc


__all__ = ["build_vxc", "build_vxc_u"]
