from __future__ import annotations

import numpy as np

from asuka.cueri.basis_cart import BasisCartSoA
from asuka.density.grids import becke_partition_weights, make_becke_grid
from asuka.integrals.int1e_cart import shell_to_atom_map
from asuka.orbitals.eval_cart import eval_shell_cart_value_grad_hess
from asuka.xc.nuc_grad import (
    _becke_owner_weight_vjp_atomgrad_cpu,
    _build_vxc_numpy,
    build_vxc_nuc_grad,
)
from asuka.xc.nuc_grad_fd import _fd_geom_gradient_from_forward


def _toy_basis(shell_cxyz: np.ndarray, shell_l: np.ndarray, exps: np.ndarray, coefs: np.ndarray) -> BasisCartSoA:
    shell_cxyz = np.ascontiguousarray(np.asarray(shell_cxyz, dtype=np.float64).reshape((-1, 3)))
    shell_l = np.ascontiguousarray(np.asarray(shell_l, dtype=np.int32).ravel())
    exps = np.ascontiguousarray(np.asarray(exps, dtype=np.float64).ravel())
    coefs = np.ascontiguousarray(np.asarray(coefs, dtype=np.float64).ravel())
    nshell = int(shell_cxyz.shape[0])
    if shell_l.shape != (nshell,):
        raise ValueError("shell_l must have shape (nshell,)")
    if exps.shape != (nshell,) or coefs.shape != (nshell,):
        raise ValueError("exps/coefs must have shape (nshell,)")

    shell_ao_start = np.empty((nshell,), dtype=np.int32)
    ao0 = 0
    for sh, l in enumerate(shell_l.tolist()):
        shell_ao_start[sh] = int(ao0)
        ao0 += (int(l) + 1) * (int(l) + 2) // 2

    return BasisCartSoA(
        shell_cxyz=shell_cxyz,
        shell_prim_start=np.arange(nshell, dtype=np.int32),
        shell_nprim=np.ones((nshell,), dtype=np.int32),
        shell_l=shell_l,
        shell_ao_start=shell_ao_start,
        prim_exp=exps,
        prim_coef=coefs,
    )


def test_eval_shell_cart_value_grad_hess_matches_fd() -> None:
    basis = _toy_basis(
        np.asarray([[0.2, -0.1, 0.3]], dtype=np.float64),
        np.asarray([2], dtype=np.int32),
        np.asarray([0.7], dtype=np.float64),
        np.asarray([1.1], dtype=np.float64),
    )
    pts = np.asarray(
        [
            [0.4, 0.2, -0.3],
            [-0.5, 0.1, 0.7],
            [0.3, -0.6, 0.2],
        ],
        dtype=np.float64,
    )
    val, grad, hess = eval_shell_cart_value_grad_hess(basis, 0, pts, want_hess=True)
    assert hess is not None

    eps = 1.0e-6
    eye = np.eye(3, dtype=np.float64)
    for axis in range(3):
        val_p, grad_p, _ = eval_shell_cart_value_grad_hess(basis, 0, pts + eps * eye[axis], want_hess=False)
        val_m, grad_m, _ = eval_shell_cart_value_grad_hess(basis, 0, pts - eps * eye[axis], want_hess=False)

        fd_grad = (val_p - val_m) / (2.0 * eps)
        np.testing.assert_allclose(grad[:, :, axis], fd_grad, rtol=1e-6, atol=1e-8)

        fd_hcol = (grad_p - grad_m) / (2.0 * eps)
        if axis == 0:
            ref = hess[:, :, [0, 1, 2]]
        elif axis == 1:
            ref = hess[:, :, [1, 3, 4]]
        else:
            ref = hess[:, :, [2, 4, 5]]
        np.testing.assert_allclose(ref, fd_hcol, rtol=5e-5, atol=5e-7)


def test_becke_owner_weight_vjp_atomgrad_cpu_matches_fd() -> None:
    atom_coords = np.asarray([[0.0, 0.0, 0.0], [1.1, -0.2, 0.3]], dtype=np.float64)
    pts, _w, point_atom = make_becke_grid(
        atom_coords,
        radial_n=2,
        angular_n=6,
        becke_n=3,
        return_point_atom=True,
    )
    rng = np.random.default_rng(123)
    bar = rng.normal(size=(int(pts.shape[0]),))

    grad = _becke_owner_weight_vjp_atomgrad_cpu(
        points=pts,
        bar_owner_weight=bar,
        point_atom=point_atom,
        atom_coords=atom_coords,
        becke_n=3,
    )

    eps = 2.0e-6
    grad_fd = np.zeros_like(grad)

    def owner_partition(coords: np.ndarray, moved_points: np.ndarray) -> np.ndarray:
        part = becke_partition_weights(moved_points, coords, becke_n=3)
        return part[np.arange(int(point_atom.size)), point_atom]

    for ia in range(int(atom_coords.shape[0])):
        for xyz in range(3):
            coords_p = atom_coords.copy()
            coords_m = atom_coords.copy()
            coords_p[ia, xyz] += eps
            coords_m[ia, xyz] -= eps

            pts_p = pts.copy()
            pts_m = pts.copy()
            mask = point_atom == ia
            pts_p[mask, xyz] += eps
            pts_m[mask, xyz] -= eps

            Lp = float(np.dot(bar, owner_partition(coords_p, pts_p)))
            Lm = float(np.dot(bar, owner_partition(coords_m, pts_m)))
            grad_fd[ia, xyz] = (Lp - Lm) / (2.0 * eps)

    np.testing.assert_allclose(grad, grad_fd, rtol=2e-5, atol=2e-6)


def test_build_vxc_nuc_grad_matches_fd_moving_grid() -> None:
    atom_coords = np.asarray([[0.0, 0.0, 0.0], [1.3, 0.2, -0.1]], dtype=np.float64)
    basis = _toy_basis(
        shell_cxyz=atom_coords,
        shell_l=np.asarray([0, 1], dtype=np.int32),
        exps=np.asarray([0.9, 0.7], dtype=np.float64),
        coefs=np.asarray([1.0, 0.8], dtype=np.float64),
    )

    rng = np.random.default_rng(456)
    A = rng.normal(size=(4, 4))
    D = A @ A.T
    D /= float(np.linalg.norm(D))

    pts, wts, point_atom = make_becke_grid(
        atom_coords,
        radial_n=3,
        angular_n=6,
        becke_n=3,
        return_point_atom=True,
    )
    shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords)

    res = build_vxc_nuc_grad(
        "m06-l",
        D,
        basis,
        pts,
        wts,
        atom_coords=atom_coords,
        point_atom=point_atom,
        shell_atom=shell_atom,
        becke_n=3,
        moving_grid=True,
        batch_size=16,
        return_vxc=True,
    )

    V_ref, E_ref = _build_vxc_numpy("m06-l", D, basis, pts, wts, batch_size=16)
    np.testing.assert_allclose(res.v_xc, V_ref, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(res.e_xc, E_ref, rtol=1e-12, atol=1e-12)

    def forward(shifted_basis, shifted_points: np.ndarray, shifted_weights: np.ndarray):
        return _build_vxc_numpy(
            "m06-l",
            D,
            shifted_basis,
            shifted_points,
            shifted_weights,
            batch_size=16,
        )

    grad_fd, _ = _fd_geom_gradient_from_forward(
        ao_basis=basis,
        atom_coords=atom_coords,
        grid_points=pts,
        grid_weights=wts,
        shell_atom=shell_atom,
        point_atom=point_atom,
        displacement=1.0e-5,
        moving_grid=True,
        becke_n=3,
        forward=forward,
        return_vxc_grad=False,
        partition_weights_fn=becke_partition_weights,
    )

    np.testing.assert_allclose(res.grad_xc, grad_fd, rtol=2e-4, atol=2e-5)


def test_build_vxc_nuc_grad_matches_fd_fixed_grid() -> None:
    atom_coords = np.asarray([[0.0, 0.0, 0.0], [1.0, -0.3, 0.4]], dtype=np.float64)
    basis = _toy_basis(
        shell_cxyz=atom_coords,
        shell_l=np.asarray([0, 0], dtype=np.int32),
        exps=np.asarray([0.8, 1.1], dtype=np.float64),
        coefs=np.asarray([1.0, 0.9], dtype=np.float64),
    )
    D = np.asarray([[1.0, 0.2], [0.2, 0.7]], dtype=np.float64)

    pts, wts = make_becke_grid(
        atom_coords,
        radial_n=3,
        angular_n=6,
        becke_n=3,
        return_point_atom=False,
    )
    shell_atom = shell_to_atom_map(basis, atom_coords_bohr=atom_coords)

    res = build_vxc_nuc_grad(
        "mn15",
        D,
        basis,
        pts,
        wts,
        atom_coords=atom_coords,
        shell_atom=shell_atom,
        moving_grid=False,
        batch_size=12,
        return_vxc=False,
    )

    def forward(shifted_basis, shifted_points: np.ndarray, shifted_weights: np.ndarray):
        return _build_vxc_numpy(
            "mn15",
            D,
            shifted_basis,
            shifted_points,
            shifted_weights,
            batch_size=12,
        )

    grad_fd, _ = _fd_geom_gradient_from_forward(
        ao_basis=basis,
        atom_coords=atom_coords,
        grid_points=pts,
        grid_weights=wts,
        shell_atom=shell_atom,
        point_atom=None,
        displacement=1.0e-5,
        moving_grid=False,
        becke_n=3,
        forward=forward,
        return_vxc_grad=False,
        partition_weights_fn=becke_partition_weights,
    )

    np.testing.assert_allclose(res.grad_xc, grad_fd, rtol=1e-4, atol=1e-5)

