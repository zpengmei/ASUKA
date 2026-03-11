from __future__ import annotations

"""Exact direct-backend nuclear gradients for (SA-)CASSCF.

This module provides a direct-SCF nuclear gradient path that keeps exact
direct/J-K semantics (no DF bridge). Two-electron derivative contractions use
the dense 4c contracted derivative kernels from :mod:`asuka.mcscf.nac._dense`.
"""

from typing import Any, Literal, Sequence
import warnings

import numpy as np

from asuka.chem.periodic_table import atomic_number
from asuka.integrals.int1e_cart import contract_dS_ip_cart, contract_dhcore_cart
from asuka.solver import GUGAFCISolver

from .nac._dense import (
    _Lorb_dot_dgorb_dx_dense,
    _build_gfock_casscf_dense_cpu,
    _grad_elec_active_dense,
    build_dense_eri4c_deriv_contraction_cache_cpu,
    grad_2e_ham_dense_eri4c_contracted,
)
from .nac._df import _FixedRDMFcisolver
from .newton_df import DFNewtonCASSCFAdapter
from .nuc_grad_df import DFNucGradMultirootResult, DFNucGradResult
from .state_average import ci_as_list, make_state_averaged_rdms, normalize_weights
from .two_e_provider import DirectHybridProvider, resolve_two_e_provider
from .zvector import build_mcscf_hessian_operator, project_ci_rhs_normalized, solve_mcscf_zvector


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        cp = None  # type: ignore[assignment]
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        return np.asarray(cp.asnumpy(a), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)


def _resolve_direct_provider(scf_out: Any) -> DirectHybridProvider:
    provider = resolve_two_e_provider(scf_out)
    if not isinstance(provider, DirectHybridProvider):
        raise ValueError(
            "direct nuclear gradients require a direct-SCF result "
            "(scf_out.direct_jk_ctx must be present and two_e_backend='direct')."
        )
    return provider


def _resolve_dense_deriv_backend(mode: str) -> Literal["cpu", "cuda"]:
    mode_n = str(mode).strip().lower()
    if mode_n not in {"auto", "cpu", "cuda"}:
        raise ValueError("direct_eri_deriv_backend must be one of {'auto','cpu','cuda'}")
    if mode_n == "cpu":
        return "cpu"

    if mode_n in {"auto", "cuda"}:
        try:
            import cupy as cp  # noqa: PLC0415

            from asuka.cueri.gpu import has_cuda_ext  # noqa: PLC0415

            if int(cp.cuda.runtime.getDeviceCount()) > 0 and bool(has_cuda_ext()):
                return "cuda"
        except Exception:
            if mode_n == "cuda":
                raise RuntimeError("direct_eri_deriv_backend='cuda' requires CuPy and cuERI CUDA extension")

    return "cpu"


def _make_rdm12_safe(
    fcisolver: Any,
    ci: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
    *,
    solver_kwargs: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    kw = dict(solver_kwargs or {})
    try:
        dm1, dm2 = fcisolver.make_rdm12(ci, int(ncas), nelecas, **kw)
    except TypeError:
        dm1, dm2 = fcisolver.make_rdm12(ci, int(ncas), nelecas)
    return np.asarray(dm1, dtype=np.float64), np.asarray(dm2, dtype=np.float64)


def _trans_rdm12_safe(
    fcisolver: Any,
    bra: Any,
    ket: Any,
    ncas: int,
    nelecas: int | tuple[int, int],
    *,
    solver_kwargs: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    kw = dict(solver_kwargs or {})
    try:
        dm1, dm2 = fcisolver.trans_rdm12(bra, ket, int(ncas), nelecas, **kw)
    except TypeError:
        dm1, dm2 = fcisolver.trans_rdm12(bra, ket, int(ncas), nelecas)
    return np.asarray(dm1, dtype=np.float64), np.asarray(dm2, dtype=np.float64)


def _sa_energy(casscf: Any, *, weights: Sequence[float], nroots: int) -> float:
    e_roots = np.asarray(getattr(casscf, "e_roots", []), dtype=np.float64).ravel()
    if int(e_roots.size) == int(nroots):
        return float(np.dot(np.asarray(weights, dtype=np.float64).ravel(), e_roots))
    return float(getattr(casscf, "e_tot", 0.0))


def _compute_unrelaxed_casscf_gradient_dense(
    *,
    ao_basis: Any,
    atom_coords_bohr: np.ndarray,
    atom_charges: np.ndarray,
    h_ao: np.ndarray,
    mo_coeff: np.ndarray,
    dm1_act: np.ndarray,
    dm2_act: np.ndarray,
    ncore: int,
    ncas: int,
    de_nuc: np.ndarray,
    shell_atom: np.ndarray,
    cache_cpu: Any,
    dense_deriv_backend: Literal["cpu", "cuda"],
    pair_table_threads: int,
    max_tile_bytes: int,
    threads: int,
) -> np.ndarray:
    gfock, _D_core, _D_act, D_tot, _C_act, _vhf_c, _vhf_ca = _build_gfock_casscf_dense_cpu(
        ao_basis,
        atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
        h_ao=np.asarray(h_ao, dtype=np.float64),
        mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
        dm1_act=np.asarray(dm1_act, dtype=np.float64),
        dm2_act=np.asarray(dm2_act, dtype=np.float64),
        ncore=int(ncore),
        ncas=int(ncas),
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        g_dm2_eps_ao=0.0,
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    de_h1 = contract_dhcore_cart(
        ao_basis,
        atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
        atom_charges=np.asarray(atom_charges, dtype=np.float64),
        M=np.asarray(D_tot, dtype=np.float64),
        shell_atom=np.asarray(shell_atom, dtype=np.int32),
    )

    de_2e = grad_2e_ham_dense_eri4c_contracted(
        ao_basis=ao_basis,
        atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
        mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
        dm1_act=np.asarray(dm1_act, dtype=np.float64),
        dm2_act=np.asarray(dm2_act, dtype=np.float64),
        ncore=int(ncore),
        ncas=int(ncas),
        backend=str(dense_deriv_backend),
        cache_cpu=cache_cpu,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    C = np.asarray(mo_coeff, dtype=np.float64)
    dme0 = C @ (0.5 * (np.asarray(gfock, dtype=np.float64) + np.asarray(gfock, dtype=np.float64).T)) @ C.T
    de_pulay = -2.0 * contract_dS_ip_cart(
        ao_basis,
        atom_coords_bohr=np.asarray(atom_coords_bohr, dtype=np.float64),
        M=np.asarray(dme0, dtype=np.float64),
        shell_atom=np.asarray(shell_atom, dtype=np.int32),
    )

    return np.asarray(de_h1 + de_2e + de_pulay + np.asarray(de_nuc, dtype=np.float64), dtype=np.float64)


def casscf_nuc_grad_direct(
    scf_out: Any,
    casscf: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    atmlst: Sequence[int] | None = None,
    direct_eri_deriv_backend: Literal["auto", "cpu", "cuda"] = "auto",
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
    solver_kwargs: dict[str, Any] | None = None,
) -> DFNucGradResult:
    """Exact direct-backend SA-CASSCF nuclear gradient."""

    provider = _resolve_direct_provider(scf_out)
    _ = provider  # validated for semantic guard

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    if not bool(getattr(mol, "cart", True)):
        raise NotImplementedError("direct nuclear gradients currently require mol.cart=True")

    coords = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64)
    atom_charges = np.asarray([float(atomic_number(sym)) for sym in getattr(mol, "elements")], dtype=np.float64)
    natm = int(coords.shape[0])
    if natm <= 0:
        return DFNucGradResult(
            e_tot=float(getattr(casscf, "e_tot", 0.0)),
            e_nuc=float(mol.energy_nuc()),
            grad=np.zeros((0, 3), dtype=np.float64),
        )

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    nroots = int(getattr(casscf, "nroots", 1))
    weights = normalize_weights(getattr(casscf, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(getattr(casscf, "ci"), nroots=nroots)

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    dm1_sa, dm2_sa = make_state_averaged_rdms(
        fcisolver_use,
        ci_list,
        weights,
        ncas=int(ncas),
        nelecas=nelecas,
        solver_kwargs=solver_kwargs,
    )

    dense_deriv_backend = _resolve_dense_deriv_backend(str(direct_eri_deriv_backend))
    cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
        getattr(scf_out, "ao_basis"),
        atom_coords_bohr=np.asarray(coords, dtype=np.float64),
        pair_table_threads=int(pair_table_threads),
    )
    shell_atom = np.asarray(cache_cpu.shell_atom, dtype=np.int32)
    h_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))
    mo_coeff = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)

    grad = _compute_unrelaxed_casscf_gradient_dense(
        ao_basis=getattr(scf_out, "ao_basis"),
        atom_coords_bohr=np.asarray(coords, dtype=np.float64),
        atom_charges=np.asarray(atom_charges, dtype=np.float64),
        h_ao=np.asarray(h_ao, dtype=np.float64),
        mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
        dm1_act=np.asarray(dm1_sa, dtype=np.float64),
        dm2_act=np.asarray(dm2_sa, dtype=np.float64),
        ncore=int(ncore),
        ncas=int(ncas),
        de_nuc=np.asarray(de_nuc, dtype=np.float64),
        shell_atom=shell_atom,
        cache_cpu=cache_cpu,
        dense_deriv_backend=dense_deriv_backend,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grad = grad[idx]

    return DFNucGradResult(
        e_tot=float(_sa_energy(casscf, weights=weights, nroots=int(nroots))),
        e_nuc=float(mol.energy_nuc()),
        grad=np.asarray(grad, dtype=np.float64),
    )


def casscf_nuc_grad_direct_per_root(
    scf_out: Any,
    casscf: Any,
    *,
    fcisolver: GUGAFCISolver | None = None,
    twos: int | None = None,
    atmlst: Sequence[int] | None = None,
    direct_eri_deriv_backend: Literal["auto", "cpu", "cuda"] = "auto",
    pair_table_threads: int = 0,
    max_tile_bytes: int = 256 << 20,
    threads: int = 0,
    solver_kwargs: dict[str, Any] | None = None,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
) -> DFNucGradMultirootResult:
    """Per-root exact direct-backend SA-CASSCF nuclear gradients."""

    from . import newton_casscf as _newton_casscf  # noqa: PLC0415

    provider = _resolve_direct_provider(scf_out)

    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")
    if not bool(getattr(mol, "cart", True)):
        raise NotImplementedError("direct per-root nuclear gradients currently require mol.cart=True")

    coords = np.asarray(getattr(mol, "coords_bohr"), dtype=np.float64)
    atom_charges = np.asarray([float(atomic_number(sym)) for sym in getattr(mol, "elements")], dtype=np.float64)
    natm = int(coords.shape[0])

    ncore = int(getattr(casscf, "ncore"))
    ncas = int(getattr(casscf, "ncas"))
    nelecas = getattr(casscf, "nelecas")
    nroots = int(getattr(casscf, "nroots", 1))
    weights = normalize_weights(getattr(casscf, "root_weights", None), nroots=nroots)
    ci_list = ci_as_list(getattr(casscf, "ci"), nroots=nroots)
    e_roots = np.asarray(getattr(casscf, "e_roots", np.asarray([float(getattr(casscf, "e_tot", 0.0))])), dtype=np.float64).ravel()

    if int(nroots) > 1:
        w_arr = np.asarray(weights, dtype=np.float64).ravel()
        if w_arr.size == int(nroots) and not np.allclose(w_arr, w_arr[0]):
            warnings.warn(
                "Per-root SA-CASSCF gradients with unequal SA weights may be ill-defined in projected SA gauge. "
                f"Current weights: {w_arr.tolist()}",
                stacklevel=2,
            )

    if natm <= 0:
        g0 = np.zeros((0, 3), dtype=np.float64)
        return DFNucGradMultirootResult(
            e_roots=np.asarray(e_roots, dtype=np.float64).ravel(),
            e_sa=float(_sa_energy(casscf, weights=weights, nroots=nroots)),
            e_nuc=float(mol.energy_nuc()),
            grads=np.zeros((int(nroots), 0, 3), dtype=np.float64),
            grad_sa=np.asarray(g0, dtype=np.float64),
            root_weights=np.asarray(weights, dtype=np.float64).ravel(),
        )

    if fcisolver is None:
        if twos is None:
            twos = int(getattr(mol, "spin", 0))
        fcisolver_use = GUGAFCISolver(twos=int(twos), nroots=int(nroots))
    else:
        fcisolver_use = fcisolver
        if getattr(fcisolver_use, "nroots", None) != int(nroots):
            try:
                fcisolver_use.nroots = int(nroots)
            except Exception:
                pass

    dm1_sa, dm2_sa = make_state_averaged_rdms(
        fcisolver_use,
        ci_list,
        weights,
        ncas=int(ncas),
        nelecas=nelecas,
        solver_kwargs=solver_kwargs,
    )
    per_root_rdms: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(int(nroots)):
        dm1_k, dm2_k = _make_rdm12_safe(
            fcisolver_use,
            ci_list[k],
            int(ncas),
            nelecas,
            solver_kwargs=solver_kwargs,
        )
        per_root_rdms.append((dm1_k, dm2_k))

    dense_deriv_backend = _resolve_dense_deriv_backend(str(direct_eri_deriv_backend))
    cache_cpu = build_dense_eri4c_deriv_contraction_cache_cpu(
        getattr(scf_out, "ao_basis"),
        atom_coords_bohr=np.asarray(coords, dtype=np.float64),
        pair_table_threads=int(pair_table_threads),
    )
    shell_atom = np.asarray(cache_cpu.shell_atom, dtype=np.int32)

    h_ao = _asnumpy_f64(getattr(getattr(scf_out, "int1e"), "hcore"))
    mo_coeff_np = _asnumpy_f64(getattr(casscf, "mo_coeff"))
    mo_coeff = getattr(casscf, "mo_coeff")
    de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64)

    grad_sa_base = _compute_unrelaxed_casscf_gradient_dense(
        ao_basis=getattr(scf_out, "ao_basis"),
        atom_coords_bohr=np.asarray(coords, dtype=np.float64),
        atom_charges=np.asarray(atom_charges, dtype=np.float64),
        h_ao=np.asarray(h_ao, dtype=np.float64),
        mo_coeff=np.asarray(mo_coeff_np, dtype=np.float64),
        dm1_act=np.asarray(dm1_sa, dtype=np.float64),
        dm2_act=np.asarray(dm2_sa, dtype=np.float64),
        ncore=int(ncore),
        ncas=int(ncas),
        de_nuc=np.asarray(de_nuc, dtype=np.float64),
        shell_atom=shell_atom,
        cache_cpu=cache_cpu,
        dense_deriv_backend=dense_deriv_backend,
        pair_table_threads=int(pair_table_threads),
        max_tile_bytes=int(max_tile_bytes),
        threads=int(threads),
    )

    mc_sa = DFNewtonCASSCFAdapter(
        df_B=None,
        hcore_ao=getattr(getattr(scf_out, "int1e"), "hcore"),
        ncore=int(ncore),
        ncas=int(ncas),
        nelecas=nelecas,
        mo_coeff=mo_coeff,
        fcisolver=fcisolver_use,
        jk_provider=provider,
        eri_provider=provider,
        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel().tolist()],
        frozen=getattr(casscf, "frozen", None),
        internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
        extrasym=getattr(casscf, "extrasym", None),
    )
    eris_sa = mc_sa.ao2mo(mo_coeff)
    hess_op = build_mcscf_hessian_operator(
        mc_sa,
        mo_coeff=mo_coeff,
        ci=ci_list,
        eris=eris_sa,
        use_newton_hessian=True,
    )
    n_orb = int(hess_op.n_orb)

    grads_out: list[np.ndarray] = []
    if int(nroots) == 1:
        grads_out.append(np.asarray(grad_sa_base, dtype=np.float64))
    else:
        for k in range(int(nroots)):
            dm1_k, dm2_k = per_root_rdms[k]
            grad_static_k = _compute_unrelaxed_casscf_gradient_dense(
                ao_basis=getattr(scf_out, "ao_basis"),
                atom_coords_bohr=np.asarray(coords, dtype=np.float64),
                atom_charges=np.asarray(atom_charges, dtype=np.float64),
                h_ao=np.asarray(h_ao, dtype=np.float64),
                mo_coeff=np.asarray(mo_coeff_np, dtype=np.float64),
                dm1_act=np.asarray(dm1_k, dtype=np.float64),
                dm2_act=np.asarray(dm2_k, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
                de_nuc=np.asarray(de_nuc, dtype=np.float64),
                shell_atom=shell_atom,
                cache_cpu=cache_cpu,
                dense_deriv_backend=dense_deriv_backend,
                pair_table_threads=int(pair_table_threads),
                max_tile_bytes=int(max_tile_bytes),
                threads=int(threads),
            )

            fcisolver_fixed = _FixedRDMFcisolver(fcisolver_use, dm1=np.asarray(dm1_k, dtype=np.float64), dm2=np.asarray(dm2_k, dtype=np.float64))
            mc_k = DFNewtonCASSCFAdapter(
                df_B=None,
                hcore_ao=getattr(getattr(scf_out, "int1e"), "hcore"),
                ncore=int(ncore),
                ncas=int(ncas),
                nelecas=nelecas,
                mo_coeff=mo_coeff,
                fcisolver=fcisolver_fixed,
                jk_provider=provider,
                eri_provider=provider,
                frozen=getattr(casscf, "frozen", None),
                internal_rotation=bool(getattr(casscf, "internal_rotation", False)),
                extrasym=getattr(casscf, "extrasym", None),
            )
            g_k = _newton_casscf.compute_mcscf_gradient_vector(
                mc_k,
                mo_coeff,
                ci_list[k],
                eris_sa,
                gauge="none",
                strict_weights=False,
                enforce_absorb_h1e_direct=True,
            )
            g_k = np.asarray(g_k, dtype=np.float64).ravel()
            rhs_orb = g_k[:n_orb]
            rhs_ci_k = g_k[n_orb:]
            rhs_ci = [np.zeros_like(np.asarray(ci_list[r], dtype=np.float64).ravel()) for r in range(int(nroots))]
            ndet_k = int(np.asarray(ci_list[k]).size)
            rhs_ci[k] = rhs_ci_k[:ndet_k]

            z_k = solve_mcscf_zvector(
                mc_sa,
                rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
                rhs_ci=rhs_ci,
                hessian_op=hess_op,
                tol=float(z_tol),
                maxiter=int(z_maxiter),
                method="gmres",
            )
            if not np.all(np.isfinite(np.asarray(z_k.z_packed, dtype=np.float64))):
                z_k = solve_mcscf_zvector(
                    mc_sa,
                    rhs_orb=np.asarray(rhs_orb, dtype=np.float64),
                    rhs_ci=rhs_ci,
                    hessian_op=hess_op,
                    tol=float(z_tol),
                    maxiter=int(z_maxiter),
                    method="gmres",
                    x0=None,
                )
            if not np.all(np.isfinite(np.asarray(z_k.z_packed, dtype=np.float64))):
                raise RuntimeError("non-finite direct per-root Z-vector solution")
            Lvec = np.asarray(z_k.z_packed, dtype=np.float64).ravel()
            Lorb_mat = mc_sa.unpack_uniq_var(Lvec[:n_orb])
            Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])
            Lci_list = project_ci_rhs_normalized(ci_list, Lci_list)
            if not isinstance(Lci_list, list) or len(Lci_list) != int(nroots):
                raise RuntimeError("internal error: projected Lci_list has unexpected structure")

            dm1_lci = np.zeros((int(ncas), int(ncas)), dtype=np.float64)
            dm2_lci = np.zeros((int(ncas), int(ncas), int(ncas), int(ncas)), dtype=np.float64)
            w_arr = np.asarray(weights, dtype=np.float64).ravel()
            for r in range(int(nroots)):
                wr = float(w_arr[r])
                if abs(wr) < 1e-14:
                    continue
                dm1_r, dm2_r = _trans_rdm12_safe(
                    fcisolver_use,
                    np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                    np.asarray(ci_list[r], dtype=np.float64).ravel(),
                    int(ncas),
                    nelecas,
                    solver_kwargs=solver_kwargs,
                )
                dm1_lci += wr * (dm1_r + dm1_r.T)
                dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))

            de_lci = _grad_elec_active_dense(
                ao_basis=getattr(scf_out, "ao_basis"),
                atom_coords_bohr=np.asarray(coords, dtype=np.float64),
                atom_charges=np.asarray(atom_charges, dtype=np.float64),
                h_ao=np.asarray(h_ao, dtype=np.float64),
                mo_coeff=np.asarray(mo_coeff_np, dtype=np.float64),
                dm1_act=np.asarray(dm1_lci, dtype=np.float64),
                dm2_act=np.asarray(dm2_lci, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
                backend=str(dense_deriv_backend),
                cache_cpu=cache_cpu,
                pair_table_threads=int(pair_table_threads),
                max_tile_bytes=int(max_tile_bytes),
                threads=int(threads),
            )
            de_lorb = _Lorb_dot_dgorb_dx_dense(
                ao_basis=getattr(scf_out, "ao_basis"),
                atom_coords_bohr=np.asarray(coords, dtype=np.float64),
                atom_charges=np.asarray(atom_charges, dtype=np.float64),
                h_ao=np.asarray(h_ao, dtype=np.float64),
                mo_coeff=np.asarray(mo_coeff_np, dtype=np.float64),
                dm1_act=np.asarray(dm1_sa, dtype=np.float64),
                dm2_act=np.asarray(dm2_sa, dtype=np.float64),
                Lorb=np.asarray(Lorb_mat, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
                backend=str(dense_deriv_backend),
                cache_cpu=cache_cpu,
                pair_table_threads=int(pair_table_threads),
                max_tile_bytes=int(max_tile_bytes),
                threads=int(threads),
            )
            grad_k = np.asarray(
                np.asarray(grad_static_k, dtype=np.float64)
                + np.asarray(de_lci, dtype=np.float64)
                + np.asarray(de_lorb, dtype=np.float64),
                dtype=np.float64,
            )
            grads_out.append(grad_k)

    grads = np.stack([np.asarray(g, dtype=np.float64) for g in grads_out], axis=0)
    grad_sa = np.asarray(grad_sa_base, dtype=np.float64)
    if atmlst is not None:
        idx = np.asarray(list(atmlst), dtype=np.int32).ravel()
        grads = grads[:, idx]
        grad_sa = grad_sa[idx]

    return DFNucGradMultirootResult(
        e_roots=np.asarray(e_roots, dtype=np.float64).ravel(),
        e_sa=float(_sa_energy(casscf, weights=weights, nroots=nroots)),
        e_nuc=float(mol.energy_nuc()),
        grads=np.asarray(grads, dtype=np.float64),
        grad_sa=np.asarray(grad_sa, dtype=np.float64),
        root_weights=np.asarray(weights, dtype=np.float64).ravel(),
    )


__all__ = ["casscf_nuc_grad_direct", "casscf_nuc_grad_direct_per_root"]
