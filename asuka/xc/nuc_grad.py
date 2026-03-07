from __future__ import annotations

"""Analytical XC nuclear gradients for the numerical-integration module.

This module implements analytical exchange-correlation (XC) nuclear gradients
for :func:`asuka.xc.numint.build_vxc` at a fixed AO density matrix.

Scope
-----
- Semilocal LDA/GGA/meta-GGA energy gradients for the Minnesota functionals
  supported by :mod:`asuka.xc.eval_xc`.
- Fixed-density-matrix gradients (i.e. the explicit XC contribution only).
- Fixed-grid (frozen quadrature) and moving-grid atom-centered quadratures.
- Optional reference ``V_xc`` evaluation on the CPU helper path.

Notes
-----
The implementation follows the standard semilocal analytic-gradient structure:
only first derivatives of the XC energy with respect to ``(rho, sigma, tau)``
are needed, but AO second derivatives are required for the GGA/meta-GGA chain
rule. The moving-grid Becke response is handled analytically as well.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.cueri.cart import ncart
from asuka.density.grids import becke_partition_weights, make_becke_grid
from asuka.integrals.int1e_cart import shell_to_atom_map
from asuka.orbitals.eval_cart import eval_basis_cart_value_grad_hess_on_points

from .eval_xc import eval_xc
from .functional import FunctionalSpec, get_functional
from .numint import _build_vxc_batch, _contract_rho_on_grid


@dataclass(frozen=True)
class XCNucGradResult:
    """Result container for analytical XC gradients.

    Parameters
    ----------
    e_xc
        Reference-grid XC energy.
    grad_xc
        Cartesian XC gradient contribution in Hartree / Bohr with shape
        ``(natm, 3)``.
    v_xc
        Reference-grid XC potential matrix in the same AO representation as the
        input density matrix. Present only when ``return_vxc=True``.
    grad_vxc
        Reserved for future use. The analytical implementation currently does
        not return explicit derivatives of the XC potential matrix itself.
    moving_grid
        Whether the point cloud and Becke partition were differentiated with
        the atom-centered moving-grid convention.
    backend
        Gradient backend identifier (always ``"analytic"`` here).
    """

    e_xc: float
    grad_xc: np.ndarray
    v_xc: np.ndarray | None = None
    grad_vxc: np.ndarray | None = None
    moving_grid: bool = True
    backend: str = "analytic"


def _coords_bohr_like(mol_or_coords: Any) -> np.ndarray:
    if hasattr(mol_or_coords, "coords_bohr"):
        arr = getattr(mol_or_coords, "coords_bohr")
    else:
        arr = mol_or_coords
    coords = np.asarray(arr, dtype=np.float64).reshape((-1, 3))
    if coords.size == 0:
        raise ValueError("atom_coords must be non-empty")
    return np.ascontiguousarray(coords)


def _to_numpy(x: Any, *, dtype: Any | None = None) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            import cupy as cp  # noqa: PLC0415

            if isinstance(x, cp.ndarray):  # type: ignore[attr-defined]
                arr = cp.asnumpy(x)
            elif isinstance(x, cp.generic):  # type: ignore[attr-defined]
                arr = np.asarray(x.item())
            else:
                arr = np.asarray(x)
        except Exception:
            arr = np.asarray(x)
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return np.ascontiguousarray(arr)


def _normalize_spec(spec: FunctionalSpec | str) -> FunctionalSpec:
    if isinstance(spec, FunctionalSpec):
        return spec
    return get_functional(str(spec))


def _recover_local_becke_weights(
    grid_weights: np.ndarray,
    owner_partition: np.ndarray,
    *,
    tol: float = 1.0e-14,
) -> np.ndarray:
    w = np.asarray(grid_weights, dtype=np.float64).ravel()
    owner = np.asarray(owner_partition, dtype=np.float64).ravel()
    if w.shape != owner.shape:
        raise ValueError("grid_weights and owner_partition must have identical shape")

    local = np.zeros_like(w)
    mask = np.abs(owner) > float(tol)
    if np.any(mask):
        local[mask] = w[mask] / owner[mask]
    bad = (~mask) & (np.abs(w) > float(tol))
    if np.any(bad):
        raise ValueError(
            "failed to recover atom-local quadrature weights because the owning Becke partition "
            "weight is numerically zero while the molecular weight is non-zero"
        )
    return local


def _ao_owner_from_shell_owner(ao_basis: Any, shell_owner: np.ndarray) -> np.ndarray:
    shell_owner = np.asarray(shell_owner, dtype=np.int32).ravel()
    shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
    shell_ao_start = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int32).ravel()
    if shell_owner.shape != shell_l.shape:
        raise ValueError("shell_owner must have one entry per shell")

    nao = 0
    for sh, l in enumerate(shell_l.tolist()):
        ao0 = int(shell_ao_start[sh])
        nao = max(nao, ao0 + int(ncart(int(l))))

    ao_owner = np.empty((nao,), dtype=np.int32)
    for sh, l in enumerate(shell_l.tolist()):
        ao0 = int(shell_ao_start[sh])
        ao1 = ao0 + int(ncart(int(l)))
        ao_owner[ao0:ao1] = int(shell_owner[sh])
    return ao_owner


def _prepare_cart_dm(
    D: Any,
    ao_basis: Any,
    *,
    sph_transform: Any | None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Return a Cartesian density matrix and optional cart->sph transform."""

    D_np = _to_numpy(D, dtype=np.float64)
    if D_np.ndim != 2 or int(D_np.shape[0]) != int(D_np.shape[1]):
        raise ValueError("D must be a square matrix")
    D_np = 0.5 * (D_np + D_np.T)

    shell_l = np.asarray(getattr(ao_basis, "shell_l"), dtype=np.int32).ravel()
    shell_ao_start = np.asarray(getattr(ao_basis, "shell_ao_start"), dtype=np.int32).ravel()
    nao_cart = 0
    for sh, l in enumerate(shell_l.tolist()):
        ao0 = int(shell_ao_start[sh])
        nao_cart = max(nao_cart, ao0 + int(ncart(int(l))))

    if sph_transform is None:
        if int(D_np.shape[0]) != int(nao_cart):
            raise ValueError(
                f"AO basis Cartesian dimension ({nao_cart}) does not match density matrix ({int(D_np.shape[0])})"
            )
        return np.ascontiguousarray(D_np), None, int(nao_cart)

    T = _to_numpy(sph_transform, dtype=np.float64)
    if T.ndim != 2:
        raise ValueError("sph_transform must be a 2D array")
    if tuple(map(int, T.shape)) != (int(nao_cart), int(D_np.shape[0])):
        raise ValueError(
            f"sph_transform must have shape ({nao_cart},{int(D_np.shape[0])}); got {tuple(map(int, T.shape))}"
        )
    D_cart = T @ D_np @ T.T
    D_cart = 0.5 * (D_cart + D_cart.T)
    return np.ascontiguousarray(D_cart), np.ascontiguousarray(T), int(nao_cart)


def _build_vxc_numpy(
    spec: FunctionalSpec | str,
    D: Any,
    ao_basis: Any,
    grid_coords: Any,
    grid_weights: Any,
    *,
    batch_size: int = 4096,
    sph_transform: Any | None = None,
) -> tuple[np.ndarray, float]:
    """CPU helper mirroring :func:`asuka.xc.numint.build_vxc`."""

    spec_use = _normalize_spec(spec)
    pts_all = _to_numpy(grid_coords, dtype=np.float64).reshape((-1, 3))
    w_all = _to_numpy(grid_weights, dtype=np.float64).ravel()
    if int(pts_all.shape[0]) != int(w_all.shape[0]):
        raise ValueError("grid_coords and grid_weights must contain the same number of points")

    batch_size = max(1, int(batch_size))

    if sph_transform is None:
        T = None
        D_use = _to_numpy(D, dtype=np.float64)
        if D_use.ndim != 2 or int(D_use.shape[0]) != int(D_use.shape[1]):
            raise ValueError("D must be a square matrix")
        D_use = 0.5 * (D_use + D_use.T)
        nao_out = int(D_use.shape[0])
    else:
        T = _to_numpy(sph_transform, dtype=np.float64)
        D_use = _to_numpy(D, dtype=np.float64)
        if D_use.ndim != 2 or int(D_use.shape[0]) != int(D_use.shape[1]):
            raise ValueError("D must be a square matrix")
        D_use = 0.5 * (D_use + D_use.T)
        if T.ndim != 2 or int(T.shape[1]) != int(D_use.shape[0]):
            raise ValueError("sph_transform second dimension must match D")
        nao_out = int(D_use.shape[0])

    V_out = np.zeros((nao_out, nao_out), dtype=np.float64)
    E_xc = 0.0

    for p0 in range(0, int(pts_all.shape[0]), batch_size):
        p1 = min(int(pts_all.shape[0]), p0 + batch_size)
        pts = pts_all[p0:p1]
        wts = w_all[p0:p1]

        phi_cart, dphi_cart, _ = eval_basis_cart_value_grad_hess_on_points(
            ao_basis,
            pts,
            want_hess=False,
        )
        if T is None:
            phi_use = phi_cart
            dphi_use = dphi_cart
        else:
            phi_use = phi_cart @ T
            dphi_use = np.einsum("gnk,ns->gsk", dphi_cart, T, optimize=True)

        rho, sigma, tau_val, nabla_rho = _contract_rho_on_grid(np, phi_use, dphi_use, D_use)
        exc, vrho, vsigma, vtau = eval_xc(spec_use, rho, sigma, tau_val)
        E_xc += float(np.sum(wts * exc * rho))
        V_out += _build_vxc_batch(
            np,
            phi_use,
            dphi_use,
            wts,
            vrho,
            vsigma,
            vtau,
            nabla_rho,
        )

    V_out = 0.5 * (V_out + V_out.T)
    return np.ascontiguousarray(V_out), float(E_xc)


def _form_ao_adjoint(
    *,
    D_cart: np.ndarray,
    weights: np.ndarray,
    vrho: np.ndarray,
    vsigma: np.ndarray,
    vtau: np.ndarray,
    phi: np.ndarray,
    dphi: np.ndarray,
    nabla_rho: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return adjoints with respect to AO values and AO gradients."""

    u = phi @ D_cart
    dphi_x = dphi[:, :, 0]
    dphi_y = dphi[:, :, 1]
    dphi_z = dphi[:, :, 2]

    v_x = dphi_x @ D_cart
    v_y = dphi_y @ D_cart
    v_z = dphi_z @ D_cart

    g_x = nabla_rho[:, 0]
    g_y = nabla_rho[:, 1]
    g_z = nabla_rho[:, 2]

    w = np.asarray(weights, dtype=np.float64).reshape((-1, 1))
    vr = np.asarray(vrho, dtype=np.float64).reshape((-1, 1))
    vs = np.asarray(vsigma, dtype=np.float64).reshape((-1, 1))
    vt = np.asarray(vtau, dtype=np.float64).reshape((-1, 1))

    bar_phi = 2.0 * vr * u
    bar_phi += 4.0 * vs * (g_x[:, None] * v_x + g_y[:, None] * v_y + g_z[:, None] * v_z)
    bar_phi *= w

    bar_dphi = np.empty_like(dphi)
    bar_dphi[:, :, 0] = (4.0 * vs * g_x[:, None] * u + vt * v_x) * w
    bar_dphi[:, :, 1] = (4.0 * vs * g_y[:, None] * u + vt * v_y) * w
    bar_dphi[:, :, 2] = (4.0 * vs * g_z[:, None] * u + vt * v_z) * w
    return np.ascontiguousarray(bar_phi), np.ascontiguousarray(bar_dphi)


def _accumulate_ao_response(
    grad_out: np.ndarray,
    *,
    ao_owner: np.ndarray,
    point_atom: np.ndarray | None,
    bar_phi: np.ndarray,
    bar_dphi: np.ndarray,
    dphi: np.ndarray,
    hphi: np.ndarray,
    moving_grid: bool,
) -> None:
    """Accumulate AO-value/AO-gradient response into per-atom gradients."""

    natm = int(grad_out.shape[0])
    bar_dx = bar_dphi[:, :, 0]
    bar_dy = bar_dphi[:, :, 1]
    bar_dz = bar_dphi[:, :, 2]

    qx = bar_phi * dphi[:, :, 0]
    qx += bar_dx * hphi[:, :, 0] + bar_dy * hphi[:, :, 1] + bar_dz * hphi[:, :, 2]

    qy = bar_phi * dphi[:, :, 1]
    qy += bar_dx * hphi[:, :, 1] + bar_dy * hphi[:, :, 3] + bar_dz * hphi[:, :, 4]

    qz = bar_phi * dphi[:, :, 2]
    qz += bar_dx * hphi[:, :, 2] + bar_dy * hphi[:, :, 4] + bar_dz * hphi[:, :, 5]

    if bool(moving_grid):
        if point_atom is None:
            raise ValueError("point_atom is required when moving_grid=True")
        point_term = np.stack((np.sum(qx, axis=1), np.sum(qy, axis=1), np.sum(qz, axis=1)), axis=1)
        np.add.at(grad_out, np.asarray(point_atom, dtype=np.int32).ravel(), point_term)

    shell_term = np.stack((np.sum(qx, axis=0), np.sum(qy, axis=0), np.sum(qz, axis=0)), axis=1)
    shell_acc = np.zeros((natm, 3), dtype=np.float64)
    np.add.at(shell_acc, np.asarray(ao_owner, dtype=np.int32).ravel(), shell_term)
    grad_out -= shell_acc


def _becke_pair_shape_and_deriv(mu: np.ndarray, *, becke_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(sA, sB, dsA/dmu)`` for Becke's recursive smooth switch."""

    x = np.clip(np.asarray(mu, dtype=np.float64), -1.0, 1.0)
    dp = np.ones_like(x)
    p = x.copy()
    for _ in range(int(becke_n)):
        dp = 1.5 * (1.0 - p * p) * dp
        p = 0.5 * (3.0 * p - p * p * p)
    sA = 0.5 * (1.0 - p)
    sB = 1.0 - sA
    dsA = -0.5 * dp
    return sA, sB, dsA


def _becke_owner_weight_vjp_atomgrad_cpu(
    *,
    points: np.ndarray,
    bar_owner_weight: np.ndarray,
    point_atom: np.ndarray,
    atom_coords: np.ndarray,
    becke_n: int,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Accumulate VJP of owner Becke partition weights into atomic gradients.

    ``bar_owner_weight[p]`` is the adjoint with respect to the *owner* Becke
    partition factor ``g_p = w_owner(p)`` for point ``p``.
    """

    pts = np.asarray(points, dtype=np.float64).reshape((-1, 3))
    bar = np.asarray(bar_owner_weight, dtype=np.float64).ravel()
    owner = np.asarray(point_atom, dtype=np.int32).ravel()
    R = np.asarray(atom_coords, dtype=np.float64).reshape((-1, 3))

    npt = int(pts.shape[0])
    natm = int(R.shape[0])
    if bar.shape != (npt,):
        raise ValueError("bar_owner_weight must have shape (npt,)")
    if owner.shape != (npt,):
        raise ValueError("point_atom must have shape (npt,)")
    if natm <= 0:
        raise ValueError("atom_coords must be non-empty")

    gout = np.zeros((natm, 3), dtype=np.float64) if out is None else out
    if gout.shape != (natm, 3):
        raise ValueError(f"out must have shape ({natm}, 3)")

    d = pts[:, None, :] - R[None, :, :]
    r = np.linalg.norm(d, axis=2)
    u = np.zeros_like(d)
    mask_r = r > 1.0e-16
    u[mask_r] = d[mask_r] / r[mask_r, None]

    dAB = R[:, None, :] - R[None, :, :]
    RAB = np.linalg.norm(dAB, axis=2)

    # Forward raw Becke products w_raw[A] for every point.
    raw = np.ones((npt, natm), dtype=np.float64)
    for A in range(natm):
        for B in range(A + 1, natm):
            Rab = float(RAB[A, B])
            if Rab <= 1.0e-14:
                continue
            mu = (r[:, A] - r[:, B]) / Rab
            sA, sB, _ = _becke_pair_shape_and_deriv(mu, becke_n=int(becke_n))
            raw[:, A] *= sA
            raw[:, B] *= sB

    wsum = np.sum(raw, axis=1)
    wsum = np.where(np.abs(wsum) > 1.0e-18, wsum, 1.0)
    wi = raw[np.arange(npt), owner]

    if not np.any(np.abs(bar) > 0.0):
        return gout

    inv_wsum = 1.0 / wsum
    inv_wsum2 = inv_wsum * inv_wsum

    for A in range(natm):
        for B in range(A + 1, natm):
            Rab = float(RAB[A, B])
            if Rab <= 1.0e-14:
                continue

            rAi = r[:, A]
            rBi = r[:, B]

            mu_raw = (rAi - rBi) / Rab
            mu = np.clip(mu_raw, -1.0, 1.0)
            _, _, dsA = _becke_pair_shape_and_deriv(mu, becke_n=int(becke_n))
            dpp_dmu = -2.0 * dsA
            dpp_dmu = np.where(np.abs(mu_raw - mu) <= 1.0e-15, dpp_dmu, 0.0)

            # Recompute the current pair switch values explicitly; these are the
            # factors whose logarithmic derivatives generate the raw-product VJP.
            sA, sB, _ = _becke_pair_shape_and_deriv(mu, becke_n=int(becke_n))

            wA = raw[:, A]
            wB = raw[:, B]
            isA = owner == int(A)
            isB = owner == int(B)

            bar_wA = -bar * wi * inv_wsum2
            bar_wB = bar_wA.copy()
            bar_wA = np.where(isA, bar_wA + bar * inv_wsum, bar_wA)
            bar_wB = np.where(isB, bar_wB + bar * inv_wsum, bar_wB)

            inv_sA = np.divide(1.0, sA, out=np.zeros_like(sA), where=np.abs(sA) > 1.0e-18)
            inv_sB = np.divide(1.0, sB, out=np.zeros_like(sB), where=np.abs(sB) > 1.0e-18)
            bar_sA = bar_wA * wA * inv_sA
            bar_sB = bar_wB * wB * inv_sB

            bar_mu = 0.5 * dpp_dmu * (bar_sB - bar_sA)
            active = np.abs(bar_mu) > 0.0
            if not np.any(active):
                continue

            vAB = dAB[A, B] / Rab
            coef = bar_mu / Rab
            coef_rab = bar_mu * (rAi - rBi) / (Rab * Rab)

            # Point translation term: the point follows its owning atom.
            point_term = coef[:, None] * (u[:, A, :] - u[:, B, :])
            np.add.at(gout, owner[active], point_term[active])

            # Explicit atom-position terms for the two atoms in the pair.
            gout[int(A)] += np.sum((-coef)[:, None] * u[:, A, :] - coef_rab[:, None] * vAB[None, :], axis=0)
            gout[int(B)] += np.sum((coef)[:, None] * u[:, B, :] + coef_rab[:, None] * vAB[None, :], axis=0)

    return gout


def build_vxc_nuc_grad(
    spec: FunctionalSpec | str,
    D: Any,
    ao_basis: Any,
    grid_coords: Any,
    grid_weights: Any,
    *,
    atom_coords: Any,
    point_atom: Any | None = None,
    shell_atom: Any | None = None,
    becke_n: int = 3,
    moving_grid: bool = True,
    batch_size: int = 4096,
    sph_transform: Any | None = None,
    return_vxc: bool = False,
    return_vxc_grad: bool = False,
) -> XCNucGradResult:
    """Analytical XC nuclear gradient from an explicit quadrature grid.

    Parameters
    ----------
    spec
        :class:`~asuka.xc.functional.FunctionalSpec` or functional name.
    D
        AO density matrix in the same AO representation used by the forward
        builder. When ``sph_transform`` is provided, ``D`` is assumed to be in
        the transformed (typically spherical) AO basis.
    ao_basis
        Packed Cartesian AO basis.
    grid_coords, grid_weights
        Reference quadrature grid. The weights are the *molecular* weights,
        i.e. already include the Becke partition factor.
    atom_coords
        Nuclear coordinates in Bohr.
    point_atom
        Owning atom index for each point. Required when ``moving_grid=True``.
    shell_atom
        Optional shell-to-atom map. When omitted it is inferred from
        ``ao_basis.shell_cxyz`` and ``atom_coords``.
    becke_n
        Number of Becke smoothing iterations used when differentiating the
        moving-grid partition weights.
    moving_grid
        Whether to include the analytical moving-grid response.
    batch_size
        Number of points processed per CPU batch. The analytical path needs AO
        Hessians, so this default is intentionally smaller than the forward
        GPU builder's batch size.
    sph_transform
        Optional Cartesian-to-target-AO transform matrix with shape
        ``(nao_cart, nao_out)``.
    return_vxc
        Include a reference ``V_xc`` matrix, evaluated on the CPU helper path.
    return_vxc_grad
        Reserved for future use. The analytical implementation does not yet
        return explicit derivatives of ``V_xc``.
    """

    if bool(return_vxc_grad):
        raise NotImplementedError(
            "Analytical grad_vxc is not implemented yet. The analytical module currently returns the XC energy gradient only."
        )

    spec_use = _normalize_spec(spec)
    atom_coords_np = _coords_bohr_like(atom_coords)
    pts_np = _to_numpy(grid_coords, dtype=np.float64).reshape((-1, 3))
    w_np = _to_numpy(grid_weights, dtype=np.float64).ravel()
    if int(pts_np.shape[0]) != int(w_np.shape[0]):
        raise ValueError("grid_coords and grid_weights must contain the same number of points")

    moving_grid = bool(moving_grid)
    batch_size = max(1, int(batch_size))

    if shell_atom is None:
        shell_atom_np = shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords_np)
    else:
        shell_atom_np = _to_numpy(shell_atom, dtype=np.int32).ravel()

    if point_atom is None:
        point_atom_np: np.ndarray | None = None
    else:
        point_atom_np = _to_numpy(point_atom, dtype=np.int32).ravel()
        if point_atom_np.shape != (int(pts_np.shape[0]),):
            raise ValueError("point_atom must have one entry per grid point")

    if moving_grid and point_atom_np is None:
        raise ValueError("point_atom is required when moving_grid=True")

    D_cart, _T, nao_cart = _prepare_cart_dm(D, ao_basis, sph_transform=sph_transform)
    ao_owner = _ao_owner_from_shell_owner(ao_basis, shell_atom_np)
    if int(ao_owner.shape[0]) != int(nao_cart):
        raise ValueError("AO owner map size does not match the Cartesian AO dimension")

    local_weights: np.ndarray | None = None
    if moving_grid:
        assert point_atom_np is not None
        part0 = becke_partition_weights(pts_np, atom_coords_np, becke_n=int(becke_n))
        owner_part0 = part0[np.arange(int(point_atom_np.size)), point_atom_np]
        local_weights = _recover_local_becke_weights(w_np, owner_part0)

    natm = int(atom_coords_np.shape[0])
    grad_xc = np.zeros((natm, 3), dtype=np.float64)
    E_xc = 0.0
    V_ref_cart = np.zeros((nao_cart, nao_cart), dtype=np.float64) if bool(return_vxc) else None

    for p0 in range(0, int(pts_np.shape[0]), batch_size):
        p1 = min(int(pts_np.shape[0]), p0 + batch_size)
        pts = pts_np[p0:p1]
        wts = w_np[p0:p1]
        p_owner_blk = None if point_atom_np is None else point_atom_np[p0:p1]
        local_w_blk = None if local_weights is None else local_weights[p0:p1]

        phi, dphi, hphi = eval_basis_cart_value_grad_hess_on_points(
            ao_basis,
            pts,
            want_hess=True,
        )
        assert hphi is not None

        rho, sigma, tau_val, nabla_rho = _contract_rho_on_grid(np, phi, dphi, D_cart)
        exc, vrho, vsigma, vtau = eval_xc(spec_use, rho, sigma, tau_val)
        eps_vol = exc * rho
        E_xc += float(np.sum(wts * eps_vol))

        if V_ref_cart is not None:
            V_ref_cart += _build_vxc_batch(
                np,
                phi,
                dphi,
                wts,
                vrho,
                vsigma,
                vtau,
                nabla_rho,
            )

        bar_phi, bar_dphi = _form_ao_adjoint(
            D_cart=D_cart,
            weights=wts,
            vrho=vrho,
            vsigma=vsigma,
            vtau=vtau,
            phi=phi,
            dphi=dphi,
            nabla_rho=nabla_rho,
        )

        _accumulate_ao_response(
            grad_xc,
            ao_owner=ao_owner,
            point_atom=p_owner_blk,
            bar_phi=bar_phi,
            bar_dphi=bar_dphi,
            dphi=dphi,
            hphi=hphi,
            moving_grid=moving_grid,
        )

        if moving_grid:
            assert p_owner_blk is not None
            assert local_w_blk is not None
            _becke_owner_weight_vjp_atomgrad_cpu(
                points=pts,
                bar_owner_weight=np.asarray(local_w_blk, dtype=np.float64) * np.asarray(eps_vol, dtype=np.float64),
                point_atom=p_owner_blk,
                atom_coords=atom_coords_np,
                becke_n=int(becke_n),
                out=grad_xc,
            )

    V_out: np.ndarray | None = None
    if V_ref_cart is not None:
        V_ref_cart = 0.5 * (V_ref_cart + V_ref_cart.T)
        if _T is None:
            V_out = np.ascontiguousarray(V_ref_cart)
        else:
            V_out = np.ascontiguousarray(_T.T @ V_ref_cart @ _T)
            V_out = 0.5 * (V_out + V_out.T)

    return XCNucGradResult(
        e_xc=float(E_xc),
        grad_xc=np.ascontiguousarray(grad_xc),
        v_xc=V_out,
        grad_vxc=None,
        moving_grid=moving_grid,
        backend="analytic",
    )


def build_vxc_nuc_grad_from_mol(
    spec: FunctionalSpec | str,
    D: Any,
    ao_basis: Any,
    mol_or_coords: Any,
    *,
    radial_n: int = 75,
    angular_n: int = 590,
    angular_kind: str = "auto",
    rmax: float = 20.0,
    becke_n: int = 3,
    prune_tol: float = 1.0e-16,
    radial_scheme: str = "treutler",
    atom_Z: Any | None = None,
    moving_grid: bool = True,
    batch_size: int = 4096,
    sph_transform: Any | None = None,
    return_vxc: bool = False,
    return_vxc_grad: bool = False,
) -> XCNucGradResult:
    """Analytical XC nuclear gradient with an internally built Becke grid."""

    atom_coords_np = _coords_bohr_like(mol_or_coords)
    grid_coords, grid_weights, point_atom = make_becke_grid(
        mol_or_coords,
        radial_n=int(radial_n),
        angular_n=int(angular_n),
        angular_kind=str(angular_kind),
        rmax=float(rmax),
        becke_n=int(becke_n),
        prune_tol=float(prune_tol),
        radial_scheme=str(radial_scheme),
        atom_Z=atom_Z,
        return_point_atom=True,
    )
    return build_vxc_nuc_grad(
        spec,
        D,
        ao_basis,
        grid_coords,
        grid_weights,
        atom_coords=atom_coords_np,
        point_atom=point_atom,
        becke_n=int(becke_n),
        moving_grid=bool(moving_grid),
        batch_size=int(batch_size),
        sph_transform=sph_transform,
        return_vxc=bool(return_vxc),
        return_vxc_grad=bool(return_vxc_grad),
    )


__all__ = [
    "XCNucGradResult",
    "build_vxc_nuc_grad",
    "build_vxc_nuc_grad_from_mol",
    "_build_vxc_numpy",
    "_becke_owner_weight_vjp_atomgrad_cpu",
]

