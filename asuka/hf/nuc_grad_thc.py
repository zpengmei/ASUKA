from __future__ import annotations

"""Analytic nuclear gradients for RHF-THC (global THC / THF).

Scope (initial implementation)
------------------------------
- Global THC (`asuka.hf.thc_factors.THCFactors`) only.
- Analytic THC factor gradients currently support:
  - `solve_method='inv_metric'` (Y = X_aux @ L^{-T})
  - `solve_method='fit_metric_gram'` (Y = X_aux @ solve(X_aux^T X_aux, L))
- GPU-first: requires CuPy + orbitals CUDA extension + cuERI CUDA extension.

Notes
-----
This implements an analytic nuclear gradient for the *pure* THC-RHF energy
(i.e., `use_density_difference=False` in `frontend.scf.run_rhf_thc`).

Density-difference gradients are not implemented yet because they require
including the reference DF term derivatives consistently.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_cupy():
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("RHF-THC analytic gradients require CuPy") from e
    return cp


def _asnumpy(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(a, cp.ndarray):
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(a, dtype=np.float64)


@dataclass(frozen=True)
class RHFTHCGradComponents:
    nuc: np.ndarray
    hcore: np.ndarray
    pulay: np.ndarray
    thc_2e: np.ndarray
    metric: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return np.asarray(self.nuc + self.hcore + self.pulay + self.thc_2e + self.metric, dtype=np.float64)


def _thc_energy_adjoint_rhf(
    D: Any,
    X: Any,
    Z: Any | None,
    Y: Any,
    *,
    q_block: int = 256,
) -> tuple[Any, Any]:
    """Return (bar_X, bar_Y) for E_2e(RHF) with THC factors.

    Shapes
    - D: (nao,nao)
    - X: (npt,nao)
    - Z: optional (npt,npt)
    - Y: (npt,naux) with Z = Y Y^T
    """

    cp = _require_cupy()
    D = cp.asarray(D, dtype=cp.float64)
    X = cp.asarray(X, dtype=cp.float64)
    Y = cp.asarray(Y, dtype=cp.float64)

    if D.ndim != 2 or int(D.shape[0]) != int(D.shape[1]):
        raise ValueError("D must be (nao,nao)")
    nao = int(D.shape[0])
    if X.ndim != 2 or int(X.shape[1]) != nao:
        raise ValueError("X must have shape (npt,nao)")
    npt = int(X.shape[0])
    if Z is not None:
        Z = cp.asarray(Z, dtype=cp.float64)
        if Z.shape != (npt, npt):
            raise ValueError("Z must have shape (npt,npt)")
    if Y.ndim != 2 or int(Y.shape[0]) != npt:
        raise ValueError("Y must have shape (npt,naux)")

    q_block = max(1, min(int(q_block), int(npt)))

    # A = X D
    A = X @ D  # (npt,nao)

    # m[p] = (X D X^T)[p,p]
    m = cp.sum(A * X, axis=1)  # (npt,)

    # g = Z m (prefer factor Y where Z = Y Y^T)
    if Z is not None:
        g = Z @ m  # (npt,)
    else:
        g = Y @ (Y.T @ m)

    # bar_X = 2*(diag(g) X) D  -  (Z ⊙ (X D X^T)) X D
    # Use A = X D to avoid an extra GEMM: (diag(g) X) D = diag(g) (X D) = g[:,None]*A
    bar_X = 2.0 * (g[:, None] * A)

    # bar_Y = (outer(m,m) - 0.5 * (M ⊙ M)) @ Y, where M = X D X^T
    v = m @ Y  # (naux,)
    bar_Y = m[:, None] * v[None, :]

    for q0 in range(0, npt, q_block):
        q1 = min(npt, int(q0) + int(q_block))
        nb = int(q1 - q0)
        if nb <= 0:
            continue

        Xq = X[int(q0) : int(q1), :]  # (nb,nao)
        Yq = Y[int(q0) : int(q1), :]  # (nb,naux)

        # M[:,Q] = (X D) X_Q^T = A Xq^T
        Mq = A @ Xq.T  # (npt,nb)

        if Z is not None:
            Zblk = Z[:, int(q0) : int(q1)]  # (npt,nb)
        else:
            Zblk = Y @ Yq.T  # (npt,nb)
        Tq = Zblk * Mq  # (npt,nb)

        # Xq D == (X D)[Q,:] == A[Q,:]
        bar_X -= Tq @ A[int(q0) : int(q1), :]

        # bar_Y -= 0.5 * (Mq^2) @ Yq
        bar_Y -= 0.5 * ((Mq * Mq) @ Yq)

        del Xq, Yq, Mq, Zblk, Tq

    return cp.ascontiguousarray(bar_X), cp.ascontiguousarray(bar_Y)


def _metric_2c2e_deriv_aux_atomgrad_cuda(
    aux_basis: Any,
    *,
    atom_coords_bohr: np.ndarray,
    bar_V: Any,
    df_threads: int = 0,
) -> Any:
    """Contract dV/dR with bar_V on CUDA for the aux Coulomb metric V=(P|Q).

    Returns grad_dev (natm,3) on device.
    """

    cp = _require_cupy()
    try:
        from asuka.cueri import _cueri_cuda_ext as _ext_cuda  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("cuERI CUDA extension is required for THC metric derivative contraction") from e

    from asuka.cueri.basis_cart import BasisCartSoA  # noqa: PLC0415
    from asuka.cueri.shell_pairs import ShellPairs  # noqa: PLC0415
    from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu  # noqa: PLC0415
    from asuka.integrals.int1e_cart import shell_to_atom_map, nao_cart_from_basis  # noqa: PLC0415

    atom_coords_bohr = np.asarray(atom_coords_bohr, dtype=np.float64).reshape((-1, 3))
    natm = int(atom_coords_bohr.shape[0])
    if natm <= 0:
        return cp.zeros((0, 3), dtype=cp.float64)

    n_shell_aux = int(np.asarray(aux_basis.shell_cxyz, dtype=np.float64).shape[0])
    if n_shell_aux <= 0:
        return cp.zeros((natm, 3), dtype=cp.float64)

    aux_shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32).ravel()
    lmax = int(np.max(aux_shell_l)) if int(aux_shell_l.size) else 0
    if lmax > 5:
        raise NotImplementedError("cuERI CUDA metric derivative kernels support only l<=5 for aux shells")

    aux_shell_atom = shell_to_atom_map(aux_basis, atom_coords_bohr=atom_coords_bohr)

    naux = int(nao_cart_from_basis(aux_basis))

    # Build aux + dummy combined basis.
    dummy_shell = int(n_shell_aux)
    aux_prim_n = int(np.asarray(aux_basis.prim_exp, dtype=np.float64).shape[0])

    shell_cxyz = np.concatenate(
        [
            np.asarray(aux_basis.shell_cxyz, dtype=np.float64, order="C"),
            np.zeros((1, 3), dtype=np.float64),
        ],
        axis=0,
    )
    shell_prim_start = np.concatenate(
        [
            np.asarray(aux_basis.shell_prim_start, dtype=np.int32, order="C"),
            np.asarray([aux_prim_n], dtype=np.int32),
        ],
        axis=0,
    )
    shell_nprim = np.concatenate(
        [
            np.asarray(aux_basis.shell_nprim, dtype=np.int32, order="C"),
            np.asarray([1], dtype=np.int32),
        ],
        axis=0,
    )
    shell_l = np.concatenate(
        [
            np.asarray(aux_shell_l, dtype=np.int32, order="C"),
            np.asarray([0], dtype=np.int32),
        ],
        axis=0,
    )
    shell_ao_start = np.concatenate(
        [
            np.asarray(aux_basis.shell_ao_start, dtype=np.int32, order="C"),
            np.asarray([naux], dtype=np.int32),
        ],
        axis=0,
    )
    prim_exp = np.concatenate(
        [
            np.asarray(aux_basis.prim_exp, dtype=np.float64, order="C"),
            np.asarray([0.0], dtype=np.float64),
        ],
        axis=0,
    )
    prim_coef = np.concatenate(
        [
            np.asarray(aux_basis.prim_coef, dtype=np.float64, order="C"),
            np.asarray([1.0], dtype=np.float64),
        ],
        axis=0,
    )

    basis_all = BasisCartSoA(
        shell_cxyz=shell_cxyz,
        shell_prim_start=shell_prim_start,
        shell_nprim=shell_nprim,
        shell_l=shell_l,
        shell_ao_start=shell_ao_start,
        prim_exp=prim_exp,
        prim_coef=prim_coef,
    )

    # ShellPairs for metric: (P, dummy) for each aux shell P.
    aux_shell_idx = np.arange(n_shell_aux, dtype=np.int32)
    sp_A = aux_shell_idx.astype(np.int32, copy=False)
    sp_B = np.full((int(aux_shell_idx.size),), int(dummy_shell), dtype=np.int32)
    sp_npair = np.asarray(aux_basis.shell_nprim, dtype=np.int32, order="C").ravel().astype(np.int32, copy=False)
    sp_pair_start = np.empty((int(sp_npair.shape[0]) + 1,), dtype=np.int32)
    sp_pair_start[0] = 0
    np.cumsum(sp_npair, dtype=np.int32, out=sp_pair_start[1:])
    sp_all = ShellPairs(
        sp_A=np.asarray(sp_A, dtype=np.int32, order="C"),
        sp_B=np.asarray(sp_B, dtype=np.int32, order="C"),
        sp_npair=np.asarray(sp_npair, dtype=np.int32, order="C"),
        sp_pair_start=np.asarray(sp_pair_start, dtype=np.int32, order="C"),
    )

    pt = build_pair_tables_cpu(basis_all, sp_all, threads=int(df_threads), profile=None)

    # Upload static tables once.
    shell_cx_dev = cp.ascontiguousarray(cp.asarray(shell_cxyz[:, 0], dtype=cp.float64))
    shell_cy_dev = cp.ascontiguousarray(cp.asarray(shell_cxyz[:, 1], dtype=cp.float64))
    shell_cz_dev = cp.ascontiguousarray(cp.asarray(shell_cxyz[:, 2], dtype=cp.float64))
    shell_prim_start_dev = cp.ascontiguousarray(cp.asarray(shell_prim_start, dtype=cp.int32))
    shell_nprim_dev = cp.ascontiguousarray(cp.asarray(shell_nprim, dtype=cp.int32))
    shell_ao_start_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start, dtype=cp.int32))
    prim_exp_dev = cp.ascontiguousarray(cp.asarray(prim_exp, dtype=cp.float64))

    sp_A_dev = cp.ascontiguousarray(cp.asarray(sp_all.sp_A, dtype=cp.int32))
    sp_B_dev = cp.ascontiguousarray(cp.asarray(sp_all.sp_B, dtype=cp.int32))
    sp_pair_start_dev = cp.ascontiguousarray(cp.asarray(sp_all.sp_pair_start, dtype=cp.int32))
    sp_npair_dev = cp.ascontiguousarray(cp.asarray(sp_all.sp_npair, dtype=cp.int32))

    pair_eta_dev = cp.ascontiguousarray(cp.asarray(np.asarray(pt.pair_eta, dtype=np.float64, order="C"), dtype=cp.float64))
    pair_Px_dev = cp.ascontiguousarray(cp.asarray(np.asarray(pt.pair_Px, dtype=np.float64, order="C"), dtype=cp.float64))
    pair_Py_dev = cp.ascontiguousarray(cp.asarray(np.asarray(pt.pair_Py, dtype=np.float64, order="C"), dtype=cp.float64))
    pair_Pz_dev = cp.ascontiguousarray(cp.asarray(np.asarray(pt.pair_Pz, dtype=np.float64, order="C"), dtype=cp.float64))
    pair_cK_dev = cp.ascontiguousarray(cp.asarray(np.asarray(pt.pair_cK, dtype=np.float64, order="C"), dtype=cp.float64))

    bar_V_dev = cp.ascontiguousarray(cp.asarray(cp.asarray(bar_V, dtype=cp.float64).reshape(-1), dtype=cp.float64))

    shell_atom_dev = cp.ascontiguousarray(cp.asarray(np.asarray(aux_shell_atom, dtype=np.int32), dtype=cp.int32))

    grad_dev = cp.zeros((natm, 3), dtype=cp.float64)
    grad_dev_flat = grad_dev.reshape(-1)

    # Group aux shell indices by angular momentum; sp index == shell index here.
    by_l: dict[int, Any] = {}
    for sh in range(n_shell_aux):
        by_l.setdefault(int(aux_shell_l[sh]), []).append(int(sh))
    sp_by_l = {l: cp.ascontiguousarray(cp.asarray(np.asarray(v, dtype=np.int32), dtype=cp.int32)) for l, v in by_l.items()}

    threads = 256
    stream_ptr = int(cp.cuda.get_current_stream().ptr)

    for lp, spAB_class_dev in sp_by_l.items():
        for lq, spCD_class_dev in sp_by_l.items():
            if int(spAB_class_dev.size) == 0 or int(spCD_class_dev.size) == 0:
                continue
            _ext_cuda.df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device(
                spAB_class_dev,
                spCD_class_dev,
                sp_A_dev,
                sp_B_dev,
                sp_pair_start_dev,
                sp_npair_dev,
                shell_cx_dev,
                shell_cy_dev,
                shell_cz_dev,
                shell_prim_start_dev,
                shell_nprim_dev,
                shell_ao_start_dev,
                prim_exp_dev,
                pair_eta_dev,
                pair_Px_dev,
                pair_Py_dev,
                pair_Pz_dev,
                pair_cK_dev,
                int(0),  # nao0 offset (aux basis starts at 0)
                int(naux),
                int(lp),
                int(lq),
                bar_V_dev,
                shell_atom_dev,
                grad_dev_flat,
                int(threads),
                int(stream_ptr),
                False,
            )

    return grad_dev


def rhf_nuc_grad_thc(
    scf_out: Any,
    *,
    q_block: int = 256,
    df_threads: int = 0,
    want_components: bool = False,
) -> np.ndarray | RHFTHCGradComponents:
    """Compute analytic nuclear gradient for *pure* RHF-THC (global THC).

    Parameters
    ----------
    scf_out
        Output of `asuka.frontend.scf.run_rhf_thc(...)`.
        Must have cached `thc_factors` and must have been run with
        `use_density_difference=False` for correctness.
    q_block
        Point-block size used for blocked contractions in the adjoint.
    df_threads
        CPU threads for building cuERI pair tables used by the CUDA metric
        derivative contraction.
    want_components
        If True, return an `RHFTHCGradComponents` struct instead of a single
        (natm,3) array.
    """

    cp = _require_cupy()

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol")

    thc = getattr(scf_out, "thc_factors", None)
    if thc is None:
        raise ValueError("scf_out.thc_factors is missing (need cached THC factors)")

    from asuka.hf.thc_factors import THCFactors  # noqa: PLC0415

    if not isinstance(thc, THCFactors):
        raise TypeError("rhf_nuc_grad_thc currently supports only global THCFactors")

    meta = {} if thc.meta is None else dict(thc.meta)
    solve_method = str(meta.get("solve_method", "fit_metric_qr")).strip().lower()
    inv_metric_methods = {"inv_metric", "inv", "metric_inv", "vinv", "v_inv"}
    fit_metric_gram_methods = {"fit_metric_gram", "gram"}
    if solve_method in inv_metric_methods:
        solve_kind = "inv_metric"
    elif solve_method in fit_metric_gram_methods:
        solve_kind = "fit_metric_gram"
    else:
        raise NotImplementedError(
            "analytic gradients currently support solve_method in {'inv_metric','fit_metric_gram'} "
            f"(got {solve_method!r})"
        )

    point_atom = meta.get("point_atom", None)
    if point_atom is None:
        raise ValueError("THC factors are missing meta['point_atom']; rebuild THC factors with grid_kind='becke' or 'rdvr'")

    grid_kind = str(meta.get("grid_kind", "")).strip().lower()
    if grid_kind not in {"becke", "rdvr"}:
        raise NotImplementedError("analytic gradients currently support only grid_kind in {'becke','rdvr'}")

    becke_n = int(meta.get("becke_n", 3))

    scf = getattr(scf_out, "scf")
    if not bool(getattr(scf, "converged", False)):
        raise ValueError("SCF is not converged")

    if bool(getattr(scf_out, "profile", None)):
        # not used; placeholder for future profiling hooks
        pass

    natm = int(getattr(mol, "natm", 0))
    if natm <= 0:
        g0 = np.zeros((0, 3), dtype=np.float64)
        return RHFTHCGradComponents(nuc=g0, hcore=g0, pulay=g0, thc_2e=g0, metric=g0) if want_components else g0

    # ---- D and W (AO rep of the SCF, cart or sph) ----
    from asuka.hf import df_scf as _df  # noqa: PLC0415

    C = scf.mo_coeff
    occ = scf.mo_occ
    eps = scf.mo_energy

    xp, _ = _df._get_xp(C, occ, eps)  # noqa: SLF001
    if xp is not cp:
        raise RuntimeError("RHF-THC gradients currently require GPU (CuPy) SCF outputs")

    C = cp.asarray(C, dtype=cp.float64)
    occ = cp.asarray(occ, dtype=cp.float64).ravel()
    eps = cp.asarray(eps, dtype=cp.float64).ravel()

    D = _df._symmetrize(cp, _df._density_from_C_occ(C, occ))  # noqa: SLF001

    occ_mask = occ > 0.0
    if not bool(cp.any(occ_mask).item()):
        raise ValueError("no occupied orbitals")
    C_occ = cp.ascontiguousarray(C[:, occ_mask])
    occ_occ = cp.ascontiguousarray(occ[occ_mask])
    eps_occ = cp.ascontiguousarray(eps[occ_mask])

    # Energy-weighted density W for Pulay term: W = C_occ diag(occ*eps) C_occ^T.
    W = (C_occ * (occ_occ * eps_occ)[None, :]) @ C_occ.T

    # ---- 1e + Pulay (CPU for now; uses existing analytic int1e contractions) ----
    coords = np.asarray(mol.coords_bohr, dtype=np.float64).reshape((natm, 3))
    from asuka.frontend.periodic_table import atomic_number  # noqa: PLC0415

    charges = np.asarray([float(atomic_number(sym)) for sym, _xyz in mol.atoms_bohr], dtype=np.float64)

    ao_basis = getattr(scf_out, "ao_basis")
    shell_atom_ao = None
    try:
        from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415

        shell_atom_ao = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    except Exception:
        shell_atom_ao = None

    is_spherical = not bool(getattr(mol, "cart", True))

    if is_spherical:
        from asuka.integrals.int1e_sph import contract_dS_sph, contract_dhcore_sph  # noqa: PLC0415

        de_hcore = contract_dhcore_sph(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M_sph=_asnumpy(D),
            shell_atom=shell_atom_ao,
        )
        de_pulay = -1.0 * contract_dS_sph(
            ao_basis,
            atom_coords_bohr=coords,
            M_sph=_asnumpy(W),
            shell_atom=shell_atom_ao,
        )
    else:
        from asuka.integrals.int1e_cart import contract_dS_cart, contract_dhcore_cart  # noqa: PLC0415

        de_hcore = contract_dhcore_cart(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M=_asnumpy(D),
            shell_atom=shell_atom_ao,
        )
        de_pulay = -1.0 * contract_dS_cart(
            ao_basis,
            atom_coords_bohr=coords,
            M=_asnumpy(W),
            shell_atom=shell_atom_ao,
        )

    de_hcore = np.asarray(de_hcore, dtype=np.float64)
    de_pulay = np.asarray(de_pulay, dtype=np.float64)

    # ---- THC 2e explicit + metric terms (GPU) ----
    # bar_X, bar_Y from the THC 2e energy.
    bar_X, bar_Y = _thc_energy_adjoint_rhf(D, thc.X, thc.Z, thc.Y, q_block=int(q_block))

    L = cp.asarray(thc.L_metric, dtype=cp.float64)

    # Accumulate atom gradients from AO/aux collocation and Becke weights.
    from asuka.orbitals.eval_basis_device import (
        becke_weight_vjp_atomgrad_device,
        contract_aos_cart_value_grad_vjp_atomgrad_device,
        eval_aos_cart_value_on_points_device,
    )  # noqa: PLC0415
    from asuka.integrals.int1e_cart import shell_to_atom_map  # noqa: PLC0415
    from asuka.integrals.df_adjoint import chol_lower_adjoint  # noqa: PLC0415

    pts = cp.asarray(thc.points, dtype=cp.float64)
    w = cp.asarray(thc.weights, dtype=cp.float64).ravel()
    p_atom = cp.asarray(point_atom, dtype=cp.int32).ravel()

    if p_atom.shape != (int(w.shape[0]),):
        raise ValueError("meta['point_atom'] shape mismatch with THC grid size")

    w_quart = cp.sqrt(cp.sqrt(w))
    w_sqrt = cp.sqrt(w)

    # Aux collocation X_aux_p = w^(1/2) * chi(r).
    aux_basis_cart = getattr(scf_out, "aux_basis")
    aux_cart = eval_aos_cart_value_on_points_device(aux_basis_cart, pts, threads=256, sync=True)
    X_aux_p_val = cp.ascontiguousarray(aux_cart * w_sqrt[:, None])  # (npt,naux)
    del aux_cart

    if solve_kind == "inv_metric":
        # Backprop inv_metric: Y^T = L^{-1} X_aux^T.
        bar_Xw_T = bar_Y.T  # (naux,npt)
        try:
            import cupyx.scipy.linalg as cpx_linalg  # noqa: PLC0415

            bar_S = cpx_linalg.solve_triangular(L, bar_Xw_T, lower=True, trans="T")
        except Exception:
            bar_S = cp.linalg.solve(L.T, bar_Xw_T)

        # bar_L = -tril( bar_S @ Y )
        bar_L = -(bar_S @ cp.asarray(thc.Y, dtype=cp.float64))
        bar_L = cp.tril(bar_L)

        bar_V = chol_lower_adjoint(L, bar_L)
        bar_X_aux_p = bar_S.T  # (npt,naux)
        del bar_S
    else:
        # fit_metric_gram: Y = X_aux_p @ G, where (X_aux_p^T X_aux_p + lam I) G = L.
        Gm = cp.ascontiguousarray(X_aux_p_val.T @ X_aux_p_val)
        rcond = float(meta.get("solve_rcond", 1e-12))
        try:
            smax = float(cp.linalg.norm(Gm, ord=2).item())
        except Exception:
            smax = 0.0
        lam = (float(rcond) ** 2) * max(float(smax), 1.0)
        if lam != 0.0:
            Gm = Gm + float(lam) * cp.eye(int(Gm.shape[0]), dtype=cp.float64)

        G = cp.linalg.solve(Gm, L)  # (naux,naux)

        bar_X_aux_p = bar_Y @ G.T
        bar_G = X_aux_p_val.T @ bar_Y

        U = cp.linalg.solve(Gm.T, bar_G)  # A^{-T} bar_G
        bar_L = cp.tril(U)
        bar_V = chol_lower_adjoint(L, bar_L)

        bar_Gm = -(U @ G.T)
        bar_X_aux_p = bar_X_aux_p + X_aux_p_val @ (bar_Gm + bar_Gm.T)

        del Gm, G, bar_G, U, bar_L, bar_Gm

    # bar_w from AO collocation: (1/(4w)) * sum_mu bar_X*X
    bar_w = (0.25 / w) * cp.sum(bar_X * cp.asarray(thc.X, dtype=cp.float64), axis=1)

    # bar_w from aux collocation requires X_aux_p.
    bar_w += (0.5 / w) * cp.sum(bar_X_aux_p * X_aux_p_val, axis=1)
    del X_aux_p_val

    grad_thc = cp.zeros((natm, 3), dtype=cp.float64)

    # AO collocation contributions (cart basis derivatives).
    sph_map = getattr(scf_out, "sph_map", None)
    if is_spherical:
        if sph_map is None:
            raise RuntimeError("expected scf_out.sph_map for mol.cart=False")
        T_c2s = getattr(sph_map, "T_c2s", None)
        if T_c2s is None:
            T_c2s = sph_map[0]
        T = np.asarray(T_c2s, dtype=np.float64)
        T_dev = cp.asarray(T, dtype=cp.float64)
        bar_X_cart = bar_X @ T_dev.T
    else:
        bar_X_cart = bar_X

    shell_atom_cart = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)
    grad_thc = contract_aos_cart_value_grad_vjp_atomgrad_device(
        ao_basis,
        pts,
        point_atom=p_atom,
        w_pow=w_quart,
        bar_ao=bar_X_cart,
        shell_atom=cp.asarray(shell_atom_cart, dtype=cp.int32),
        natm=natm,
        out=grad_thc,
        threads=256,
        sync=False,
    )

    # Aux collocation contributions.
    shell_atom_aux = shell_to_atom_map(aux_basis_cart, atom_coords_bohr=coords)
    grad_thc = contract_aos_cart_value_grad_vjp_atomgrad_device(
        aux_basis_cart,
        pts,
        point_atom=p_atom,
        w_pow=w_sqrt,
        bar_ao=bar_X_aux_p,
        shell_atom=cp.asarray(shell_atom_aux, dtype=cp.int32),
        natm=natm,
        out=grad_thc,
        threads=256,
        sync=False,
    )

    # Becke partition weight derivative contributions.
    atom_coords_dev = cp.ascontiguousarray(cp.asarray(coords, dtype=cp.float64))
    grad_thc = becke_weight_vjp_atomgrad_device(
        pts,
        w,
        bar_w=bar_w,
        point_atom=p_atom,
        atom_coords=atom_coords_dev,
        becke_n=int(becke_n),
        out=grad_thc,
        threads=256,
        sync=False,
    )

    # Metric derivative contraction.
    grad_metric_dev = _metric_2c2e_deriv_aux_atomgrad_cuda(
        aux_basis_cart,
        atom_coords_bohr=coords,
        bar_V=bar_V,
        df_threads=int(df_threads),
    )

    # Synchronize once before copying back.
    cp.cuda.get_current_stream().synchronize()

    de_thc_2e = np.asarray(cp.asnumpy(grad_thc), dtype=np.float64)
    de_metric = np.asarray(cp.asnumpy(grad_metric_dev), dtype=np.float64)

    de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)

    comps = RHFTHCGradComponents(
        nuc=de_nuc,
        hcore=de_hcore,
        pulay=de_pulay,
        thc_2e=de_thc_2e,
        metric=de_metric,
    )
    return comps if want_components else comps.total


__all__ = ["RHFTHCGradComponents", "rhf_nuc_grad_thc"]
