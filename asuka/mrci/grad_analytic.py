from __future__ import annotations

"""ASUKA-native analytic nuclear gradients for uncontracted MRCISD.

This module implements a "SOC-free" (spin-free) analytic nuclear gradient for
uncontracted MRCISD on top of an ASUKA SA-CASSCF reference using:

- ASUKA DF integral derivative contractions
- CP-CASSCF Z-vector response in the *reference* SA-CASSCF parameter space

Limitations (initial)
---------------------
- `method="mrcisd"` only.
- Analytic gradients require:
  * `correlate_inactive == 0` (no split of the inactive space), and
  * `n_virt is None` (all virtual orbitals correlated).

Both restrictions avoid gauge/selection dependence on redundant core/core or
virt/virt rotations, which are not part of the SA-CASSCF optimization variables.
For workflows that violate these conditions, use the FD backend.
"""

import contextlib
import os
from typing import Any, Literal, Sequence

import numpy as np

from asuka.mrci.result import MRCIStatesResult


@contextlib.contextmanager
def _force_internal_newton():
    k_prefer = "CUGUGA_NEWTON_CASSCF"
    k_impl = "CUGUGA_NEWTON_CASSCF_IMPL"
    old_prefer = os.environ.get(k_prefer)
    old_impl = os.environ.get(k_impl)
    os.environ[k_prefer] = "internal"
    os.environ[k_impl] = "internal"
    try:
        yield
    finally:
        if old_prefer is None:
            os.environ.pop(k_prefer, None)
        else:
            os.environ[k_prefer] = old_prefer
        if old_impl is None:
            os.environ.pop(k_impl, None)
        else:
            os.environ[k_impl] = old_impl


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # noqa: PLC0415

        if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    if hasattr(a, "get"):
        try:
            return np.asarray(a.get(), dtype=np.float64)
        except Exception:
            pass
    return np.asarray(a, dtype=np.float64)


def mrci_grad_states_from_ref_analytic(
    scf_out: Any,
    ref: Any,
    *,
    mrci_states: MRCIStatesResult,
    roots: np.ndarray,
    states: Sequence[int],
    max_virt_e: int = 2,
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    df_backend: Literal["cpu", "cuda"] = "cpu",
    df_config: Any | None = None,
    df_threads: int = 0,
    z_tol: float = 1e-10,
    z_maxiter: int = 200,
) -> list[np.ndarray]:
    """Compute analytic MRCISD gradients for the requested reference states.

    Parameters
    ----------
    scf_out
        ASUKA HF/DF output (provides DF factors, AO bases, hcore).
    ref
        ASUKA CASSCF/CASCI result (mo_coeff, ci, ncore, ncas, nelecas, etc.).
    mrci_states
        Spin-free MRCISD result from :func:`asuka.mrci.driver_asuka.mrci_states_from_ref`.
    roots
        Assigned MRCI root indices for each requested reference state (same order as `states`).
    states
        Reference state indices (for bookkeeping only; order must match `roots` and mrci_states.states).
    max_virt_e
        MRCISD truncation parameter used to build the DRT.
    rdm_backend
        Backend for RDM evaluation ("cpu" or "cuda").
    df_backend
        Backend for DF derivative contraction ("cpu" or "cuda").
    df_config, df_threads
        DF contraction knobs (passed to DF derivative kernels / FD fallback).
    z_tol, z_maxiter
        Z-vector solver controls.
    """

    # ── Imports (local to keep module import lightweight) ─────────────────────
    from asuka.integrals.grad import (  # noqa: PLC0415
        compute_df_gradient_contributions_analytic_packed_bases,
        compute_df_gradient_contributions_fd_packed_bases,
    )
    from asuka.integrals.int1e_cart import (  # noqa: PLC0415
        contract_dS_ip_cart,
        contract_dhcore_cart,
        shell_to_atom_map,
    )
    from asuka.integrals.df_grad_context import DFGradContractionContext  # noqa: PLC0415
    from asuka.mcscf.nac._df import (  # noqa: PLC0415
        _build_bar_L_lorb_df,
        _build_bar_L_net_active_df,
    )
    from asuka.mcscf.newton_df import DFNewtonCASSCFAdapter  # noqa: PLC0415
    from asuka.mcscf.nuc_grad_df import (  # noqa: PLC0415
        _build_bar_L_casscf_df,
        _build_gfock_casscf_df,
        _mol_coords_charges_bohr,
    )
    from asuka.mcscf.state_average import (  # noqa: PLC0415
        ci_as_list,
        make_state_averaged_rdms,
        normalize_weights,
    )
    from asuka.mcscf.zvector import (  # noqa: PLC0415
        build_mcscf_hessian_operator,
        solve_mcscf_zvector,
    )
    from asuka.mrci.rdm_mrcisd import (  # noqa: PLC0415
        make_rdm12_mrcisd,
        prepare_mrcisd_rdm_workspace,
    )
    from asuka.solver import GUGAFCISolver  # noqa: PLC0415

    # ── Validate basic shapes / invariance assumptions ───────────────────────
    states_list = [int(s) for s in states]
    roots = np.asarray(roots, dtype=np.int64).ravel()
    if roots.shape != (len(states_list),):
        raise ValueError("roots must have shape (len(states),)")

    ncore_ref = int(getattr(ref, "ncore", 0))
    ncas_ref = int(getattr(ref, "ncas", 0))
    if ncas_ref <= 0:
        raise ValueError("ref.ncas must be positive")

    n_act_int = int(mrci_states.n_act)
    correlate_inactive = int(n_act_int - ncas_ref)
    if correlate_inactive != 0:
        raise NotImplementedError(
            "Analytic MRCISD gradients currently require correlate_inactive==0. "
            "Split inactive spaces make the energy depend on redundant core-core rotations."
        )

    C_full = _asnumpy_f64(getattr(ref, "mo_coeff"))
    nmo = int(C_full.shape[1])
    nvirt_all = nmo - ncore_ref - ncas_ref
    if nvirt_all < 0:
        raise RuntimeError("invalid orbital partition: ncore+ncas > nmo")

    nvirt_corr = int(mrci_states.n_virt)
    if nvirt_corr != int(nvirt_all):
        raise NotImplementedError(
            "Analytic MRCISD gradients currently require n_virt=None (all virtual orbitals correlated). "
            f"Got n_virt={nvirt_corr} but total virtuals={nvirt_all}."
        )

    # ── Common SCF/CASSCF objects ───────────────────────────────────────────
    mol = getattr(scf_out, "mol", None)
    if mol is None:
        raise TypeError("scf_out must have a .mol attribute")

    coords, charges = _mol_coords_charges_bohr(mol)
    natm = int(coords.shape[0])

    ao_basis = getattr(scf_out, "ao_basis")
    aux_basis = getattr(scf_out, "aux_basis")
    shell_atom = shell_to_atom_map(ao_basis, atom_coords_bohr=coords)

    B_ao = getattr(scf_out, "df_B", None)
    if B_ao is None:
        raise ValueError("scf_out.df_B is required")
    h_ao = getattr(getattr(scf_out, "int1e"), "hcore", None)
    if h_ao is None:
        raise ValueError("scf_out.int1e.hcore is required")

    # Reference SA-CASSCF ingredients (Hessian operator for Z-vector solve).
    nroots_ref = int(getattr(ref, "nroots", 1))
    weights = normalize_weights(getattr(ref, "root_weights", None), nroots=nroots_ref)
    ci_list = ci_as_list(getattr(ref, "ci"), nroots=nroots_ref)
    nelecas = getattr(ref, "nelecas")
    twos = int(getattr(getattr(scf_out, "mol", None), "spin", 0))

    fcisolver = GUGAFCISolver(twos=int(twos), nroots=int(nroots_ref))
    dm1_sa, dm2_sa = make_state_averaged_rdms(
        fcisolver,
        ci_list,
        weights,
        ncas=int(ncas_ref),
        nelecas=nelecas,
    )
    dm1_sa = np.asarray(dm1_sa, dtype=np.float64)
    dm2_sa = np.asarray(dm2_sa, dtype=np.float64)

    mc_sa = DFNewtonCASSCFAdapter(
        df_B=_asnumpy_f64(B_ao),
        hcore_ao=_asnumpy_f64(h_ao),
        ncore=int(ncore_ref),
        ncas=int(ncas_ref),
        nelecas=nelecas,
        mo_coeff=C_full,
        fcisolver=fcisolver,
        weights=[float(w) for w in np.asarray(weights, dtype=np.float64).ravel()],
        frozen=getattr(ref, "frozen", None),
        internal_rotation=bool(getattr(ref, "internal_rotation", False)),
        extrasym=getattr(ref, "extrasym", None),
    )
    eris_sa = mc_sa.ao2mo(C_full)
    with _force_internal_newton():
        hess_op = build_mcscf_hessian_operator(
            mc_sa,
            mo_coeff=C_full,
            ci=ci_list,
            eris=eris_sa,
            use_newton_hessian=True,
        )
    n_orb = int(hess_op.n_orb)

    # Optional cached DF contraction context.
    df_grad_ctx = None
    try:
        df_grad_ctx = DFGradContractionContext.build(
            ao_basis,
            aux_basis,
            atom_coords_bohr=coords,
            backend=str(df_backend),
            df_threads=int(df_threads),
            L_chol=getattr(scf_out, "df_L", None),
        )
    except Exception:
        df_grad_ctx = None

    # MRCISD RDM workspace (correlated-space DRT).
    rdm_ws = prepare_mrcisd_rdm_workspace(
        mrci_states.mrci.drt,
        n_act=int(mrci_states.n_act),
        n_virt=int(mrci_states.n_virt),
        nelec=int(mrci_states.nelec),
        twos=int(mrci_states.twos),
        max_virt_e=int(max_virt_e),
    )

    # ── Per-state gradient loop ─────────────────────────────────────────────
    grads: list[np.ndarray] = []

    ncore_frozen = int(mrci_states.ncore)
    ncor = int(mrci_states.n_act) + int(mrci_states.n_virt)

    for k, _st in enumerate(states_list):
        root = int(roots[k])
        if root < 0 or root >= int(mrci_states.nroots):
            raise ValueError(f"root index out of range: {root}")

        ci_mrci = mrci_states.mrci.ci[root]
        dm1_corr, dm2_corr = make_rdm12_mrcisd(rdm_ws, ci_mrci, rdm_backend=rdm_backend)
        dm1_corr = np.asarray(dm1_corr, dtype=np.float64)
        dm2_corr = np.asarray(dm2_corr, dtype=np.float64)

        # Build target generalized Fock + densities treating correlated orbitals as "active".
        gfock, D_core_ao, D_corr_ao, D_tot_ao, C_corr = _build_gfock_casscf_df(
            _asnumpy_f64(B_ao),
            _asnumpy_f64(h_ao),
            C_full,
            ncore=int(ncore_frozen),
            ncas=int(ncor),
            dm1_act=dm1_corr,
            dm2_act=dm2_corr,
        )

        # Unrelaxed DF 2e derivative term via bar_L (core mean-field + corr 2-RDM).
        bar_L_target = _build_bar_L_casscf_df(
            _asnumpy_f64(B_ao),
            D_core_ao=_asnumpy_f64(D_core_ao),
            D_act_ao=_asnumpy_f64(D_corr_ao),
            C_act=_asnumpy_f64(C_corr),
            dm2_act=dm2_corr,
        )

        # Pulay term uses energy-weighted density W from gfock.
        nocc = int(ncore_frozen + ncor)
        C_occ = C_full[:, :nocc]
        gfock_np = _asnumpy_f64(gfock)
        tmp_w = C_full @ gfock_np[:, :nocc]  # (nao,nocc)
        W = 0.5 * (tmp_w @ C_occ.T + C_occ @ tmp_w.T)
        de_pulay = -2.0 * contract_dS_ip_cart(ao_basis, atom_coords_bohr=coords, M=W, shell_atom=shell_atom)

        # Z-vector RHS from orbital gradient matrix 2*(gfock - gfock^T), packed in SA-CASSCF variables.
        g_orb = gfock_np - gfock_np.T
        rhs_orb = 2.0 * np.asarray(mc_sa.pack_uniq_var(g_orb), dtype=np.float64).ravel()

        z = solve_mcscf_zvector(
            mc_sa,
            rhs_orb=rhs_orb,
            rhs_ci=None,
            hessian_op=hess_op,
            tol=float(z_tol),
            maxiter=int(z_maxiter),
        )
        Lvec = np.asarray(z.z_packed, dtype=np.float64).ravel()
        Lorb = mc_sa.unpack_uniq_var(Lvec[:n_orb])
        Lci_list = hess_op.ci_unflatten(Lvec[n_orb:])

        # CI-response transition RDMs (state-averaged weights).
        dm1_lci = np.zeros((ncas_ref, ncas_ref), dtype=np.float64)
        dm2_lci = np.zeros((ncas_ref, ncas_ref, ncas_ref, ncas_ref), dtype=np.float64)
        for r in range(nroots_ref):
            wr = float(np.asarray(weights, dtype=np.float64).ravel()[r])
            if abs(wr) < 1e-14:
                continue
            dm1_r, dm2_r = fcisolver.trans_rdm12(
                np.asarray(Lci_list[r], dtype=np.float64).ravel(),
                np.asarray(ci_list[r], dtype=np.float64).ravel(),
                int(ncas_ref),
                nelecas,
                rdm_backend=str(rdm_backend),
                return_cupy=False,
            )
            dm1_r = np.asarray(dm1_r, dtype=np.float64)
            dm2_r = np.asarray(dm2_r, dtype=np.float64)
            dm1_lci += wr * (dm1_r + dm1_r.T)
            dm2_lci += wr * (dm2_r + dm2_r.transpose(1, 0, 3, 2))

        # Response DF pieces (no Pulay / no nuclear term).
        bar_L_lci_net, D_act_lci = _build_bar_L_net_active_df(
            _asnumpy_f64(B_ao),
            C_full,
            dm1_lci,
            dm2_lci,
            ncore=int(ncore_ref),
            ncas=int(ncas_ref),
            xp=np,
        )
        bar_L_lorb, D_L_lorb = _build_bar_L_lorb_df(
            _asnumpy_f64(B_ao),
            C_full,
            np.asarray(Lorb, dtype=np.float64),
            dm1_sa,
            dm2_sa,
            ncore=int(ncore_ref),
            ncas=int(ncas_ref),
            xp=np,
        )

        bar_L_total = np.asarray(bar_L_target, dtype=np.float64) + _asnumpy_f64(bar_L_lci_net) + _asnumpy_f64(bar_L_lorb)
        D_h1_total = _asnumpy_f64(D_tot_ao) + _asnumpy_f64(D_act_lci) + _asnumpy_f64(D_L_lorb)

        # DF 2e gradient contraction (analytic preferred; fallback to FD-on-B).
        try:
            if df_grad_ctx is not None:
                de_df = df_grad_ctx.contract(B_ao=_asnumpy_f64(B_ao), bar_L_ao=bar_L_total)
            else:
                de_df = compute_df_gradient_contributions_analytic_packed_bases(
                    ao_basis,
                    aux_basis,
                    atom_coords_bohr=coords,
                    B_ao=_asnumpy_f64(B_ao),
                    bar_L_ao=bar_L_total,
                    L_chol=getattr(scf_out, "df_L", None),
                    backend=str(df_backend),
                    df_threads=int(df_threads),
                    profile=None,
                )
        except (NotImplementedError, RuntimeError, ValueError):
            de_df = compute_df_gradient_contributions_fd_packed_bases(
                ao_basis,
                aux_basis,
                atom_coords_bohr=coords,
                bar_L_ao=bar_L_total,
                backend=str(df_backend),
                df_config=df_config,
                df_threads=int(df_threads),
                delta_bohr=1e-4,
                profile=None,
            )

        de_h1 = contract_dhcore_cart(
            ao_basis,
            atom_coords_bohr=coords,
            atom_charges=charges,
            M=_asnumpy_f64(D_h1_total),
            shell_atom=shell_atom,
        )
        de_nuc = np.asarray(mol.energy_nuc_grad(), dtype=np.float64) if natm else np.zeros((0, 3), dtype=np.float64)

        grad = np.asarray(de_nuc + np.asarray(de_h1, dtype=np.float64) + _asnumpy_f64(de_df) + np.asarray(de_pulay, dtype=np.float64), dtype=np.float64)
        grads.append(grad)

    return grads


__all__ = ["mrci_grad_states_from_ref_analytic"]
