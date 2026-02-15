from __future__ import annotations

"""Orbital-rotation gradient helpers for CASCI/CASSCF (DF-based).

The routines here build the CASSCF/CASCI orbital gradient matrix from:
  - DF factors B[μ,ν,Q] (whitened)
  - AO core Hamiltonian hcore
  - MO coefficients C (AO->MO)
  - active-space 1- and 2-RDMs (in MO active indices)

The returned gradient is the antisymmetric matrix (PySCF convention):

    G = g - g.T

Note: With the common Cayley/exponential rotation parameterization, the
finite-difference derivative of the *optimized-CI* CASCI energy w.r.t. a single
rotation parameter κ[p,q] is typically ~ ``2*G[p,q]``. PySCF defines the packed
orbital-gradient vector as the unique elements of ``G``.

with `g` being the generalized "Fock-like" matrix described in
`asuka/mcscf/grad_implementation.md`.
"""

from typing import Any

import time
import numpy as np

from asuka.hf import df_scf as _df_scf


def _as_xp_f64(xp: Any, a: Any) -> Any:
    return xp.asarray(a, dtype=xp.float64)


def _asnumpy_f64(a: Any) -> np.ndarray:
    """Ensure array is numpy.float64 (moves from GPU if needed)."""
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        return np.asarray(cp.asnumpy(a), dtype=np.float64, order="C")
    return np.asarray(a, dtype=np.float64, order="C")


def allowed_rotation_mask(nmo: int, ncore: int, ncas: int) -> np.ndarray:
    """Return boolean mask for non-redundant CASSCF rotations (lower triangle).

    Parameters
    ----------
    nmo : int
        Number of MOs.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.

    Returns
    -------
    np.ndarray
        Boolean mask (nmo,nmo) where True indicates an allowed rotation parameter.
    """

    nmo = int(nmo)
    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas
    if ncore < 0 or ncas <= 0 or nocc > nmo:
        raise ValueError("invalid ncore/ncas for rotation mask")

    core = np.zeros((nmo,), dtype=bool)
    act = np.zeros((nmo,), dtype=bool)
    virt = np.zeros((nmo,), dtype=bool)
    core[:ncore] = True
    act[ncore:nocc] = True
    virt[nocc:] = True

    allowed = np.zeros((nmo, nmo), dtype=bool)
    # core-active, core-virtual, active-virtual
    allowed[np.ix_(act, core)] = True
    allowed[np.ix_(virt, core)] = True
    allowed[np.ix_(virt, act)] = True
    # Unique variables: strict lower triangle
    allowed &= np.tril(np.ones((nmo, nmo), dtype=bool), k=-1)
    return allowed


def cayley_update(xp: Any, A: Any) -> Any:
    """Return U = (I - A/2)^-1 (I + A/2) for antisymmetric A.

    Parameters
    ----------
    xp : Any
        Array module (numpy or cupy).
    A : Any
        Antisymmetric matrix.

    Returns
    -------
    Any
        Unitary rotation matrix U.
    """

    A = _as_xp_f64(xp, A)
    n = int(A.shape[0])
    I = xp.eye(n, dtype=xp.float64)
    lhs = I - 0.5 * A
    rhs = I + 0.5 * A
    return xp.linalg.solve(lhs, rhs)


def _default_gdm2_blocks(xp: Any, *, naux: int, nmo: int) -> tuple[int, int]:
    """Return conservative (aux_block, mo_block) defaults for blocked g_dm2.

    Parameters
    ----------
    xp : Any
        Array module.
    naux : int
        Number of aux functions.
    nmo : int
        Number of MOs.

    Returns
    -------
    tuple[int, int]
        (aux_block_naux, mo_block_nmo)
    """

    naux = int(naux)
    nmo = int(nmo)
    if naux < 1 or nmo < 1:
        raise ValueError("invalid naux/nmo for block defaults")

    if xp is np:
        aux_block = min(naux, 128)
        mo_block = min(nmo, 64)
    else:
        aux_block = min(naux, 256)
        mo_block = min(nmo, 128)
    return max(1, int(aux_block)), max(1, int(mo_block))


def _g_dm2_df_blocked(
    xp: Any,
    *,
    B: Any,
    C: Any,
    C_act: Any,
    dm2_flat: Any,
    aux_block_naux: int,
    mo_block_nmo: int,
    use_batched_matmul: bool,
    profile: dict | None = None,
) -> Any:
    """Streamed build of g_dm2 without materializing L_pact (nmo,ncas,naux).

    Computes active-active contribution to the orbital gradient.

    Parameters
    ----------
    xp : Any
        Array module.
    B : Any
        DF B tensor.
    C : Any
        MO coefficients.
    C_act : Any
        Active MO coefficients.
    dm2_flat : Any
        Flattened 2-RDM.
    aux_block_naux : int
        Block size for aux dimension.
    mo_block_nmo : int
        Block size for MO dimension.
    use_batched_matmul : bool
        Use batched matmul if available.
    profile : dict | None
        Profiling dict.

    Returns
    -------
    Any
        g_dm2 matrix (nmo, ncas).
    """

    t0 = time.perf_counter() if profile is not None else 0.0
    if profile is not None:
        profile.clear()

    B = _as_xp_f64(xp, B)
    C = _as_xp_f64(xp, C)
    C_act = _as_xp_f64(xp, C_act)
    dm2_flat = _as_xp_f64(xp, dm2_flat)

    nao, nmo = map(int, C.shape)
    ncas = int(C_act.shape[1])
    naux = int(B.shape[2])
    if tuple(B.shape[:2]) != (nao, nao):
        raise ValueError("B/C nao mismatch")
    if dm2_flat.shape != (ncas * ncas, ncas * ncas):
        raise ValueError("dm2_flat shape mismatch")

    aux_block_naux = max(1, int(aux_block_naux))
    mo_block_nmo = max(1, int(mo_block_nmo))
    use_batched_matmul = bool(use_batched_matmul)

    g_dm2 = xp.zeros((nmo, ncas), dtype=xp.float64)

    t_x = 0.0
    t_lact = 0.0
    t_t = 0.0
    t_mo = 0.0

    for q0 in range(0, naux, aux_block_naux):
        q1 = min(naux, int(q0) + int(aux_block_naux))
        qb = int(q1 - q0)

        Bq = B[:, :, int(q0) : int(q1)]  # (nao,nao,qb)

        # X[μ,v,Q] = Σ_ν B[μ,ν,Q] C_act[ν,v]  (nao,ncas,qb)
        tt = time.perf_counter() if profile is not None else 0.0
        tmp = xp.tensordot(Bq, C_act, axes=([1], [0]))  # (nao,qb,ncas)
        X = tmp.transpose(0, 2, 1)  # (nao,ncas,qb)
        if profile is not None:
            t_x += time.perf_counter() - tt

        # L_act[w,x,Q] = Σ_μ C_act[μ,w] X[μ,x,Q]  (ncas,ncas,qb)
        tt = time.perf_counter() if profile is not None else 0.0
        L_act = xp.tensordot(C_act.T, X, axes=([1], [0]))  # (ncas,ncas,qb)
        if profile is not None:
            t_lact += time.perf_counter() - tt

        # T[Q,u,v] = Σ_{w,x} L_act[w,x,Q] * dm2[w x, u v]
        tt = time.perf_counter() if profile is not None else 0.0
        L2t = L_act.reshape(ncas * ncas, qb).T  # (qb,ncas^2)
        T2t = L2t @ dm2_flat  # (qb,ncas^2)
        T = T2t.reshape(qb, ncas, ncas)  # (qb,u,v)
        if profile is not None:
            t_t += time.perf_counter() - tt

        tt_mo = time.perf_counter() if profile is not None else 0.0
        for p0 in range(0, nmo, mo_block_nmo):
            p1 = min(nmo, int(p0) + int(mo_block_nmo))
            Cp = C[:, int(p0) : int(p1)]  # (nao,pb)

            # Lp[p,u,Q] = Σ_μ Cp[μ,p] X[μ,u,Q]  (pb,ncas,qb)
            Lp = xp.tensordot(Cp.T, X, axes=([1], [0]))  # (pb,ncas,qb)

            if use_batched_matmul:
                # (qb,pb,ncas) @ (qb,ncas,ncas) -> (qb,pb,ncas), then sum over qb.
                Lp_q = Lp.transpose(2, 0, 1)  # (qb,pb,ncas)
                prod = xp.matmul(Lp_q, T)  # (qb,pb,ncas)
                g_dm2[int(p0) : int(p1)] += xp.sum(prod, axis=0)
            else:
                g_dm2[int(p0) : int(p1)] += xp.einsum("puQ,Quv->pv", Lp, T, optimize=True)
        if profile is not None:
            t_mo += time.perf_counter() - tt_mo

        # Help reuse the array pool (CuPy) / reduce peak live tensors.
        tmp = None
        X = None
        L_act = None
        L2t = None
        T2t = None
        T = None

    if profile is not None:
        profile["is_gpu"] = bool(xp is not np)
        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)
        profile["naux"] = int(naux)
        profile["aux_block_naux"] = int(aux_block_naux)
        profile["mo_block_nmo"] = int(mo_block_nmo)
        profile["use_batched_matmul"] = bool(use_batched_matmul)
        profile["t_build_X_s"] = float(t_x)
        profile["t_build_L_act_s"] = float(t_lact)
        profile["t_build_T_s"] = float(t_t)
        profile["t_mo_blocks_s"] = float(t_mo)
        profile["t_total_s"] = float(time.perf_counter() - t0)

    return g_dm2


def orbital_gradient_df(
    scf_out: Any,
    *,
    C: Any,
    ncore: int,
    ncas: int,
    dm1_act: Any,
    dm2_act: Any,
    allowed: np.ndarray | None = None,
    aux_block_naux: int | None = None,
    mo_block_nmo: int | None = None,
    use_batched_matmul: bool | None = None,
    force_blocked: bool = False,
    profile: dict | None = None,
) -> tuple[Any, float, Any]:
    """Compute the DF-based orbital gradient for a CASCI/CASSCF wavefunction.

    Parameters
    ----------
    scf_out : Any
        SCF/DF container providing at least:
          - ``df_B``: whitened DF factors (nao, nao, naux)
          - ``int1e.hcore``: AO core Hamiltonian (nao, nao)
    C : Any
        MO coefficients (nao, nmo) on either NumPy or CuPy.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    dm1_act : Any
        Active-space 1-RDM (ncas, ncas).
    dm2_act : Any
        Active-space 2-RDM (ncas, ncas, ncas, ncas) or flattened (ncas^2, ncas^2).
    allowed : np.ndarray | None, optional
        Optional non-redundant rotation mask (as returned by
        :func:`allowed_rotation_mask`). If omitted, it is constructed from
        ``(nmo,ncore,ncas)`` and used only for the returned `grad_norm`.
    aux_block_naux : int | None, optional
        Block sizes for the streamed DF contraction of the 2-RDM term. If
        omitted, conservative defaults are chosen based on backend (NumPy vs
        CuPy).
    mo_block_nmo : int | None, optional
        Block size for MO loops.
    use_batched_matmul : bool | None, optional
        If True, uses batched matrix multiplications for the innermost
        contraction when supported by the backend. Default: True on GPU, False
        on CPU.
    force_blocked : bool, optional
        If True, always use the blocked g_dm2 build (useful for testing).
    profile : dict | None, optional
        Profiling data storage.

    Returns
    -------
    G : Any
        Antisymmetric orbital gradient matrix (nmo, nmo) on the DF backend.
    grad_norm : float
        Norm of the non-redundant gradient variables (as selected by `allowed`).
    eps : Any
        Mean-field orbital energies from the diagonal of the DF-Fock matrix,
        shape (nmo,), on the DF backend.
    """

    if profile is not None:
        profile.clear()
        t0_total = time.perf_counter()
    else:
        t0_total = 0.0

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    B = getattr(scf_out, "df_B", None)
    if B is None:
        raise ValueError("scf_out.df_B is missing")
    int1e = getattr(scf_out, "int1e", None)
    if int1e is None:
        raise ValueError("scf_out.int1e is missing")
    hcore = getattr(int1e, "hcore", None)
    if hcore is None:
        raise ValueError("scf_out.int1e.hcore is missing")

    xp, _is_gpu = _df_scf._get_xp(B, C)  # noqa: SLF001
    C = _as_xp_f64(xp, C)
    B = _as_xp_f64(xp, B)
    h_ao = _as_xp_f64(xp, hcore)

    if C.ndim != 2:
        raise ValueError("C must be 2D (nao,nmo)")
    nao, nmo = map(int, C.shape)
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    dm1_act = np.asarray(dm1_act, dtype=np.float64)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError(f"dm1_act must have shape {(ncas, ncas)}, got {tuple(dm1_act.shape)}")

    dm2_arr = np.asarray(dm2_act, dtype=np.float64)
    if dm2_arr.shape == (ncas, ncas, ncas, ncas):
        dm2_flat = dm2_arr.reshape(ncas * ncas, ncas * ncas)
    elif dm2_arr.shape == (ncas * ncas, ncas * ncas):
        dm2_flat = dm2_arr
    else:
        raise ValueError(
            "dm2_act must have shape (ncas,ncas,ncas,ncas) or (ncas^2,ncas^2), "
            f"got {tuple(dm2_arr.shape)}"
        )

    dm1_act_xp = _as_xp_f64(xp, dm1_act)
    dm2_act_xp = _as_xp_f64(xp, dm2_flat)

    # Core + active AO densities
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    if ncore:
        D_core_ao = 2.0 * (C_core @ C_core.T)
    else:
        D_core_ao = xp.zeros((nao, nao), dtype=xp.float64)
    D_act_ao = C_act @ dm1_act_xp @ C_act.T
    D_tot_ao = D_core_ao + D_act_ao

    t_density = time.perf_counter() if profile is not None else 0.0

    # GPU event timing setup (only when profiling on GPU)
    _gpu_jk_events: list[tuple] | None = None
    _gpu_gdm2_ev_start = None
    _gpu_gdm2_ev_end = None
    if profile is not None and xp is not np:
        try:
            import cupy as _cp
            _gpu_jk_events = []
            _jk_ev_s = _cp.cuda.Event(); _jk_ev_e = _cp.cuda.Event()
            _jk_ev_s.record()
        except Exception:
            _gpu_jk_events = None

    # AO potentials: core and active JK (for generalized Fock pieces and preconditioner)
    Jc, Kc = _df_scf._df_JK(B, D_core_ao, want_J=True, want_K=True)  # noqa: SLF001
    Ja, Ka = _df_scf._df_JK(B, D_act_ao, want_J=True, want_K=True)  # noqa: SLF001
    vhf_c_ao = Jc - 0.5 * Kc
    vhf_a_ao = Ja - 0.5 * Ka
    vhf_ca_ao = vhf_c_ao + vhf_a_ao

    # Mean-field Fock (for diagonal energy preconditioner).
    # J/K are linear in density, so reuse the core/active builds instead of a 3rd JK call.
    Jt = Jc + Ja
    Kt = Kc + Ka
    f_ao = h_ao + (Jt - 0.5 * Kt)

    t_jk = time.perf_counter() if profile is not None else 0.0
    if _gpu_jk_events is not None:
        try:
            _jk_ev_e.record()
            _gpu_jk_events.append((_jk_ev_s, _jk_ev_e))
        except Exception:
            pass

    # Transform AO matrices to MO
    h_mo = C.T @ h_ao @ C
    vhf_c_mo = C.T @ vhf_c_ao @ C
    vhf_ca_mo = C.T @ vhf_ca_ao @ C
    f_mo = C.T @ f_ao @ C
    eps = xp.diag(f_mo)

    t_transform = time.perf_counter() if profile is not None else 0.0

    # GPU event for g_dm2 section
    if profile is not None and xp is not np:
        try:
            import cupy as _cp
            _gpu_gdm2_ev_start = _cp.cuda.Event()
            _gpu_gdm2_ev_end = _cp.cuda.Event()
            _gpu_gdm2_ev_start.record()
        except Exception:
            pass

    # g_dm2: DF contraction of the active 2-RDM term.
    # For large systems, materializing X(nao,ncas,naux) and L_pact(nmo,ncas,naux)
    # can OOM. Use a blocked build in that case.
    naux = int(B.shape[2])

    if aux_block_naux is None or mo_block_nmo is None:
        d_aux, d_mo = _default_gdm2_blocks(xp, naux=naux, nmo=nmo)
        if aux_block_naux is None:
            aux_block_naux = d_aux
        if mo_block_nmo is None:
            mo_block_nmo = d_mo

    if use_batched_matmul is None:
        use_batched_matmul = xp is not np

    bytes_l_pact = int(nmo) * int(ncas) * int(naux) * 8
    bytes_x = int(nao) * int(ncas) * int(naux) * 8
    use_blocked = bool(force_blocked) or (bytes_l_pact >= 128 * 1024 * 1024) or (bytes_x >= 128 * 1024 * 1024)

    if not use_blocked:
        X = xp.einsum("mnQ,nv->mvQ", B, C_act, optimize=True)
        L_pact = xp.einsum("mp,mvQ->pvQ", C, X, optimize=True)  # (nmo,ncas,naux)
        L_act = L_pact[ncore:nocc]  # (ncas,ncas,naux)

        L2 = L_act.reshape(ncas * ncas, naux)  # (ncas^2,naux)
        T_flat = L2.T @ dm2_act_xp  # (naux,ncas^2)
        T = T_flat.reshape(naux, ncas, ncas)  # (Q,u,v)
        g_dm2 = xp.einsum("puQ,Quv->pv", L_pact, T, optimize=True)  # (nmo,ncas)
    else:
        gdm2_prof = profile.setdefault("g_dm2_blocked", {}) if profile is not None else None
        g_dm2 = _g_dm2_df_blocked(
            xp,
            B=B,
            C=C,
            C_act=C_act,
            dm2_flat=dm2_act_xp,
            aux_block_naux=int(aux_block_naux),
            mo_block_nmo=int(mo_block_nmo),
            use_batched_matmul=bool(use_batched_matmul),
            profile=gdm2_prof,
        )

    t_gdm2 = time.perf_counter() if profile is not None else 0.0
    if _gpu_gdm2_ev_end is not None:
        try:
            _gpu_gdm2_ev_end.record()
        except Exception:
            pass

    # gpq columns for core + active only (virtual columns are 0)
    gpq = xp.zeros((nmo, nmo), dtype=xp.float64)
    if ncore:
        gpq[:, :ncore] = 2.0 * (h_mo + vhf_ca_mo)[:, :ncore]
    gpq[:, ncore:nocc] = (h_mo + vhf_c_mo)[:, ncore:nocc] @ dm1_act_xp + g_dm2

    # PySCF convention: pack_uniq_var(g - g.T)
    G = gpq - gpq.T

    if allowed is None:
        allowed = allowed_rotation_mask(nmo, ncore, ncas)
    allowed_xp = xp.asarray(allowed)
    g_vec = G[allowed_xp].ravel()
    grad_norm = float(xp.linalg.norm(g_vec).item())

    if profile is not None:
        t_end = time.perf_counter()
        profile["is_gpu"] = bool(xp is not np)
        profile["use_blocked_gdm2"] = bool(use_blocked)
        profile["aux_block_naux"] = int(aux_block_naux)
        profile["mo_block_nmo"] = int(mo_block_nmo)
        profile["use_batched_matmul"] = bool(use_batched_matmul)
        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)
        profile["naux"] = int(naux)
        profile["t_density_s"] = float(t_density - t0_total)
        profile["t_jk_s"] = float(t_jk - t_density)
        profile["t_transform_s"] = float(t_transform - t_jk)
        profile["t_gdm2_s"] = float(t_gdm2 - t_transform)
        profile["t_gpq_s"] = float(t_end - t_gdm2)
        profile["t_total_s"] = float(t_end - t0_total)
        # GPU event timing for JK and g_dm2 if available
        if _gpu_jk_events is not None:
            try:
                import cupy as _cp
                _cp.cuda.get_current_stream().synchronize()
                jk_ms = sum(
                    _cp.cuda.get_elapsed_time(s, e)
                    for s, e in _gpu_jk_events
                )
                profile["gpu_jk_time_ms"] = float(jk_ms)
            except Exception:
                pass
        if _gpu_gdm2_ev_start is not None and _gpu_gdm2_ev_end is not None:
            try:
                import cupy as _cp
                _gpu_gdm2_ev_end.synchronize()
                profile["gpu_gdm2_time_ms"] = float(
                    _cp.cuda.get_elapsed_time(_gpu_gdm2_ev_start, _gpu_gdm2_ev_end)
                )
            except Exception:
                pass

    return G, grad_norm, eps


def _infer_max_l_from_ao_basis(ao_basis: Any) -> int:
    """Infer the maximum angular momentum L in the AO basis."""
    shell_l = getattr(ao_basis, "shell_l", None)
    if shell_l is None:
        return 5
    arr = np.asarray(shell_l, dtype=np.int32).ravel()
    if int(arr.size) == 0:
        return 0
    return int(np.max(arr))


def _g_dm2_dense_cpu(
    ao_basis: Any,
    *,
    C_mo: np.ndarray,
    C_act: np.ndarray,
    dm2_wxuv_flat: np.ndarray,
    eps_ao: float = 0.0,
    max_tile_bytes: int = 256 * 1024 * 1024,
    threads: int = 0,
    blas_nthreads: int | None = None,
    p_block_nmo: int = 64,
    profile: dict | None = None,
) -> np.ndarray:
    """CPU exact (dense) build of g_dm2 using cuERI AO tiles (cart AOs).

    Computes active-active contribution to the orbital gradient.

    Parameters
    ----------
    ao_basis : Any
        AO basis object.
    C_mo : np.ndarray
        MO coefficients.
    C_act : np.ndarray
        Active MO coefficients.
    dm2_wxuv_flat : np.ndarray
        Flattened 2-RDM (row=(w,x), col=(u,v)).
    eps_ao : float, optional
        Schwarz screening threshold.
    max_tile_bytes : int, optional
        Max bytes per tile.
    threads : int, optional
        Number of threads.
    blas_nthreads : int | None, optional
        BLAS threads.
    p_block_nmo : int, optional
        Block size for p index.
    profile : dict | None, optional
        Profiling dict.

    Returns
    -------
    np.ndarray
        g_dm2 matrix (nmo, ncas).

    Notes
    -----
    - This is the dense-consistent analogue of the DF contraction in
      :func:`orbital_gradient_df`.
    - Uses the CPU Rys tile evaluator (requires `asuka.cueri._eri_rys_cpu`).
    - Supports cart bases up to compiled cuERI CPU angular limit (`CPU_MAX_L`).
    """

    if profile is not None:
        profile.clear()

    from time import perf_counter

    from asuka.cueri.dense_cpu import CPU_MAX_L, schwarz_shellpairs_cpu  # noqa: PLC0415
    from asuka.cueri.eri_utils import build_pair_coeff_ordered  # noqa: PLC0415
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order  # noqa: PLC0415

    try:
        from asuka.cueri import _eri_rys_cpu as _eri_cpu  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "CPU dense orbital-gradient path requires asuka.cueri._eri_rys_cpu. "
            "Build it via `python -m asuka.cueri.build_cpu_ext build_ext --inplace`."
        ) from e

    from asuka.cueri.eri_utils import build_pair_coeff_ordered_mixed  # noqa: PLC0415
    from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu  # noqa: PLC0415

    C_mo = np.asarray(C_mo, dtype=np.float64, order="C")
    C_act = np.asarray(C_act, dtype=np.float64, order="C")

    if C_mo.ndim != 2 or C_act.ndim != 2:
        raise ValueError("C_mo/C_act must be 2D arrays")
    nao, nmo = map(int, C_mo.shape)
    nao2, ncas = map(int, C_act.shape)
    if nao2 != nao:
        raise ValueError("C_mo/C_act nao mismatch")

    dm2_wxuv_flat = np.asarray(dm2_wxuv_flat, dtype=np.float64, order="C")
    if dm2_wxuv_flat.shape != (ncas * ncas, ncas * ncas):
        raise ValueError("dm2_wxuv_flat shape mismatch")
    dm2_wxuv = dm2_wxuv_flat.reshape(ncas * ncas, ncas, ncas)  # (wx,u,v)

    max_l = _infer_max_l_from_ao_basis(ao_basis)
    if max_l > CPU_MAX_L:
        raise NotImplementedError(
            f"dense CPU g_dm2 currently supports only l<={CPU_MAX_L} per shell (cuERI CPU limit)"
        )

    eps_ao_f = float(eps_ao)
    if eps_ao_f < 0.0:
        raise ValueError("eps_ao must be >= 0")

    max_tile_bytes_i = int(max_tile_bytes)
    if max_tile_bytes_i <= 0:
        raise ValueError("max_tile_bytes must be > 0")

    threads_i = int(threads)
    if threads_i < 0:
        raise ValueError("threads must be >= 0")

    p_block_nmo_i = int(p_block_nmo)
    if p_block_nmo_i < 1:
        raise ValueError("p_block_nmo must be >= 1")

    blas_nthreads_i: int | None
    if blas_nthreads is None:
        blas_nthreads_i = None
    else:
        blas_nthreads_i = int(blas_nthreads)
        if blas_nthreads_i < 1:
            raise ValueError("blas_nthreads must be >= 1")

    if blas_nthreads_i is not None:
        from asuka.cuguga.blas_threads import blas_thread_limit  # noqa: PLC0415

        blas_cm = blas_thread_limit(int(blas_nthreads_i))
    else:
        import contextlib

        blas_cm = contextlib.nullcontext()

    # Basis preprocessing.
    t0_total = perf_counter() if profile is not None else 0.0
    sp = build_shell_pairs_l_order(ao_basis)
    nsp = int(sp.sp_A.shape[0])
    if nsp == 0 or nmo == 0 or ncas == 0:
        return np.zeros((nmo, ncas), dtype=np.float64)

    pt_prof = {} if profile is not None else None
    pair_tables = build_pair_tables_cpu(ao_basis, sp, threads=threads_i, profile=pt_prof)

    # Schwarz bounds for screening.
    if eps_ao_f > 0.0:
        sp_Q = schwarz_shellpairs_cpu(ao_basis, sp, pair_tables=pair_tables, max_l=max_l, threads=threads_i)
    else:
        sp_Q = np.ones((nsp,), dtype=np.float64)

    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32).ravel()
    shell_ao_start = np.asarray(ao_basis.shell_ao_start, dtype=np.int32).ravel()

    eri_tile_sp = getattr(_eri_cpu, "eri_rys_tile_cart_sp_cy", None)
    eri_tile_sp_batch = getattr(_eri_cpu, "eri_rys_tile_cart_sp_batch_cy", None)
    if eri_tile_sp is None or eri_tile_sp_batch is None:  # pragma: no cover
        raise RuntimeError("asuka.cueri._eri_rys_cpu is missing expected sp tile entry points; rebuild the extension")

    # Pair tables (CPU Rys backend).
    pair_eta = np.asarray(pair_tables.pair_eta, dtype=np.float64, order="C")
    pair_Px = np.asarray(pair_tables.pair_Px, dtype=np.float64, order="C")
    pair_Py = np.asarray(pair_tables.pair_Py, dtype=np.float64, order="C")
    pair_Pz = np.asarray(pair_tables.pair_Pz, dtype=np.float64, order="C")
    pair_cK = np.asarray(pair_tables.pair_cK, dtype=np.float64, order="C")

    sp_A = np.asarray(sp.sp_A, dtype=np.int32, order="C")
    sp_B = np.asarray(sp.sp_B, dtype=np.int32, order="C")
    sp_pair_start = np.asarray(sp.sp_pair_start, dtype=np.int32, order="C")
    sp_npair = np.asarray(sp.sp_npair, dtype=np.int32, order="C")
    shell_cxyz = np.asarray(ao_basis.shell_cxyz, dtype=np.float64, order="C")
    shell_l_c = np.asarray(ao_basis.shell_l, dtype=np.int32, order="C")

    g_dm2 = np.zeros((nmo, ncas), dtype=np.float64)

    # Screened CD list for each AB: include all CD (ordered) so the result is
    # dense-consistent without needing explicit (AB|CD) symmetry handling.
    with blas_cm:
        for spAB in range(nsp):
            q_ab = float(sp_Q[int(spAB)])
            if eps_ao_f > 0.0 and q_ab <= 0.0:
                continue
            if eps_ao_f > 0.0:
                spcd_keep = np.nonzero(q_ab * sp_Q >= eps_ao_f)[0].astype(np.int32, copy=False)
            else:
                spcd_keep = np.arange(nsp, dtype=np.int32)

            if int(spcd_keep.size) == 0:
                continue

            A = int(sp_A[int(spAB)])
            B = int(sp_B[int(spAB)])
            la = int(shell_l[A])
            lb = int(shell_l[B])
            nA = int((la + 1) * (la + 2) // 2)
            nB = int((lb + 1) * (lb + 2) // 2)
            nAB = int(nA * nB)

            aoA = int(shell_ao_start[A])
            aoB = int(shell_ao_start[B])
            CA_p_full = C_mo[aoA : aoA + nA, :]  # (nA,nmo)
            CB_p_full = C_mo[aoB : aoB + nB, :]  # (nB,nmo)
            CA_u = C_act[aoA : aoA + nA, :]  # (nA,ncas)
            CB_u = C_act[aoB : aoB + nB, :]  # (nB,ncas)
            K_act_AB = build_pair_coeff_ordered(CA_u, CB_u, same_shell=(A == B))  # (nAB,ncas^2)

            # B_sum_AB[μν, wx] = Σ_{CD} (AB|CD) ⋅ K_act_CD
            B_sum = np.zeros((nAB, ncas * ncas), dtype=np.float64)

            # Diagonal tile (AB|AB) if requested.
            if bool(np.any(spcd_keep == int(spAB))):
                tile_diag = eri_tile_sp(
                    shell_cxyz,
                    shell_l_c,
                    sp_A,
                    sp_B,
                    sp_pair_start,
                    sp_npair,
                    pair_eta,
                    pair_Px,
                    pair_Py,
                    pair_Pz,
                    pair_cK,
                    int(spAB),
                    int(spAB),
                )  # (nAB,nAB)
                B_sum += tile_diag @ K_act_AB

            spcd_off = spcd_keep[spcd_keep != int(spAB)]
            if int(spcd_off.size) > 0:
                shellC_all = np.asarray(sp_A[spcd_off], dtype=np.int32)
                shellD_all = np.asarray(sp_B[spcd_off], dtype=np.int32)
                lc_all = shell_l[shellC_all]
                ld_all = shell_l[shellD_all]

                key_all = lc_all.astype(np.int32, copy=False) * 16 + ld_all.astype(np.int32, copy=False)
                for key in np.unique(key_all):
                    key_i = int(key)
                    lc0 = key_i // 16
                    ld0 = key_i - lc0 * 16
                    nC = int((lc0 + 1) * (lc0 + 2) // 2)
                    nD = int((ld0 + 1) * (ld0 + 2) // 2)
                    nCD = int(nC * nD)

                    mask = key_all == key_i
                    spcd_grp = np.asarray(spcd_off[mask], dtype=np.int32, order="C")
                    if int(spcd_grp.size) == 0:
                        continue

                    bytes_per_task = int(8 * (nAB * nCD + nAB * (ncas * ncas)))
                    chunk_nt = int(max(1, max_tile_bytes_i // max(bytes_per_task, 1)))

                    for i0 in range(0, int(spcd_grp.size), chunk_nt):
                        i1 = min(int(spcd_grp.size), i0 + chunk_nt)
                        spcd_chunk = np.asarray(spcd_grp[i0:i1], dtype=np.int32, order="C")

                        tile = eri_tile_sp_batch(
                            shell_cxyz,
                            shell_l_c,
                            sp_A,
                            sp_B,
                            sp_pair_start,
                            sp_npair,
                            pair_eta,
                            pair_Px,
                            pair_Py,
                            pair_Pz,
                            pair_cK,
                            int(spAB),
                            spcd_chunk,
                            threads_i,
                        )  # (m, nAB, nCD)

                        # Build K_act_CD for each CD in the chunk, then contract.
                        K_act_CD = np.empty((int(spcd_chunk.size), nCD, ncas * ncas), dtype=np.float64)
                        for j, spCD in enumerate(spcd_chunk.tolist()):
                            C_sh = int(sp_A[int(spCD)])
                            D_sh = int(sp_B[int(spCD)])
                            c0 = int(shell_ao_start[C_sh])
                            d0 = int(shell_ao_start[D_sh])
                            CC_u = C_act[c0 : c0 + nC, :]
                            CD_u = C_act[d0 : d0 + nD, :]
                            K_act_CD[j] = build_pair_coeff_ordered(CC_u, CD_u, same_shell=(C_sh == D_sh))

                        tmp = tile @ K_act_CD  # (m, nAB, ncas^2)
                        B_sum += tmp.sum(axis=0)

            # Now contract bra-side mixed coefficients in p blocks and fold in dm2.
            for p0 in range(0, nmo, p_block_nmo_i):
                p1 = min(nmo, p0 + p_block_nmo_i)
                pb = int(p1 - p0)
                CA_p = CA_p_full[:, int(p0) : int(p1)]
                CB_p = CB_p_full[:, int(p0) : int(p1)]
                K_mixed = build_pair_coeff_ordered_mixed(CA_p, CB_p, CA_u, CB_u, same_shell=(A == B))  # (nAB,pb*ncas)
                pu_wx = K_mixed.T @ B_sum  # (pb*ncas, ncas^2)
                pu_wx = pu_wx.reshape(pb, ncas, ncas * ncas)  # (pb,u,wx)
                g_dm2[int(p0) : int(p1)] += np.einsum("puw,wuv->pv", pu_wx, dm2_wxuv, optimize=True)

    if profile is not None:
        profile["t_total_s"] = float(perf_counter() - float(t0_total))
        profile["eps_ao"] = float(eps_ao_f)
        profile["threads"] = int(threads_i)
        profile["blas_nthreads"] = None if blas_nthreads_i is None else int(blas_nthreads_i)
        profile["max_tile_bytes"] = int(max_tile_bytes_i)
        profile["p_block_nmo"] = int(p_block_nmo_i)
        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)
        profile["nsp"] = int(nsp)
        if pt_prof is not None:
            profile["pair_tables"] = pt_prof

    return g_dm2


def orbital_gradient_dense(
    scf_out: Any,
    *,
    C: Any,
    ncore: int,
    ncas: int,
    dm1_act: Any,
    dm2_act: Any,
    allowed: np.ndarray | None = None,
    dense_eps_ao: float = 0.0,
    dense_max_tile_bytes: int = 256 * 1024 * 1024,
    dense_cpu_threads: int = 0,
    dense_cpu_blas_nthreads: int | None = None,
    dense_cpu_p_block_nmo: int = 64,
    dense_gpu_threads: int = 256,
    dense_gpu_builder: Any | None = None,
    dense_exact_jk: bool = False,
    profile: dict | None = None,
) -> tuple[Any, float, Any]:
    """Dense-consistent orbital gradient for CASCI/CASSCF (exact active-space ERIs).

    This differs from :func:`orbital_gradient_df` only in the 2-RDM contraction term
    `g_dm2`, which is computed using *exact* (non-DF) mixed-index integrals.

    Parameters
    ----------
    scf_out : Any
        SCF result object.
    C : Any
        MO coefficients.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    dm1_act : Any
        Active 1-RDM.
    dm2_act : Any
        Active 2-RDM.
    allowed : np.ndarray | None, optional
        Allowed rotation mask.
    dense_eps_ao : float, optional
        Screening threshold.
    dense_max_tile_bytes : int, optional
        Max tile bytes.
    dense_cpu_threads : int, optional
        CPU threads.
    dense_cpu_blas_nthreads : int | None, optional
        BLAS threads.
    dense_cpu_p_block_nmo : int, optional
        Block size.
    dense_gpu_threads : int, optional
        GPU threads.
    dense_gpu_builder : Any | None, optional
        GPU builder object.
    dense_exact_jk : bool, optional
        Use exact JK.
    profile : dict | None, optional
        Profiling dict.

    Returns
    -------
    G : Any
        Gradient matrix.
    grad_norm : float
        Gradient norm.
    eps : Any
        Orbital energies.
    """

    if profile is not None:
        profile.clear()
        t0_total = time.perf_counter()
    else:
        t0_total = 0.0

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")

    B = getattr(scf_out, "df_B", None)
    if B is None:
        raise ValueError("scf_out.df_B is missing")
    int1e = getattr(scf_out, "int1e", None)
    if int1e is None:
        raise ValueError("scf_out.int1e is missing")
    hcore = getattr(int1e, "hcore", None)
    if hcore is None:
        raise ValueError("scf_out.int1e.hcore is missing")

    ao_basis = getattr(scf_out, "ao_basis", None)
    if ao_basis is None:
        raise ValueError("scf_out.ao_basis is missing (required for dense orbital gradients)")

    xp, _is_gpu = _df_scf._get_xp(B, C)  # noqa: SLF001
    C = _as_xp_f64(xp, C)
    B = _as_xp_f64(xp, B)
    h_ao = _as_xp_f64(xp, hcore)

    if C.ndim != 2:
        raise ValueError("C must be 2D (nao,nmo)")
    nao, nmo = map(int, C.shape)
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds number of MOs")

    dm1_act = np.asarray(dm1_act, dtype=np.float64)
    if dm1_act.shape != (ncas, ncas):
        raise ValueError(f"dm1_act must have shape {(ncas, ncas)}, got {tuple(dm1_act.shape)}")

    dm2_arr = np.asarray(dm2_act, dtype=np.float64)
    if dm2_arr.shape == (ncas, ncas, ncas, ncas):
        dm2_uvwx = dm2_arr
    elif dm2_arr.shape == (ncas * ncas, ncas * ncas):
        dm2_uvwx = dm2_arr.reshape(ncas, ncas, ncas, ncas)
    else:
        raise ValueError(
            "dm2_act must have shape (ncas,ncas,ncas,ncas) or (ncas^2,ncas^2), "
            f"got {tuple(dm2_arr.shape)}"
        )

    dm1_act_xp = _as_xp_f64(xp, dm1_act)
    dm2_wxuv_flat = dm2_uvwx.transpose(2, 3, 0, 1).reshape(ncas * ncas, ncas * ncas)
    dm2_wxuv_flat = 0.5 * (dm2_wxuv_flat + dm2_wxuv_flat.T)
    dm2_wxuv_xp = _as_xp_f64(xp, dm2_wxuv_flat)

    # Core + active AO densities
    C_core = C[:, :ncore]
    C_act = C[:, ncore:nocc]
    if ncore:
        D_core_ao = 2.0 * (C_core @ C_core.T)
    else:
        D_core_ao = xp.zeros((nao, nao), dtype=xp.float64)
    D_act_ao = C_act @ dm1_act_xp @ C_act.T
    D_tot_ao = D_core_ao + D_act_ao
    t_density = time.perf_counter() if profile is not None else 0.0

    # AO potentials: core and active JK (for generalized Fock pieces and preconditioner).
    dense_exact_jk = bool(dense_exact_jk)
    if dense_exact_jk:
        mol_exact = getattr(scf_out, "mol", None)
        scf_exact = getattr(scf_out, "scf", None)
        get_jk = getattr(scf_exact, "get_jk", None)
        if mol_exact is None or not callable(get_jk):
            raise ValueError("dense_exact_jk=True requires scf_out.scf.get_jk")

        def _jk_exact(D_in):
            d_h = _asnumpy_f64(D_in)
            try:
                J_h, K_h = get_jk(mol_exact, d_h, hermi=1)
            except TypeError:
                J_h, K_h = get_jk(d_h, hermi=1)
            return _as_xp_f64(xp, J_h), _as_xp_f64(xp, K_h)

        Jc, Kc = _jk_exact(D_core_ao)
        Ja, Ka = _jk_exact(D_act_ao)
    else:
        Jc, Kc = _df_scf._df_JK(B, D_core_ao, want_J=True, want_K=True)  # noqa: SLF001
        Ja, Ka = _df_scf._df_JK(B, D_act_ao, want_J=True, want_K=True)  # noqa: SLF001
    vhf_c_ao = Jc - 0.5 * Kc
    vhf_a_ao = Ja - 0.5 * Ka
    vhf_ca_ao = vhf_c_ao + vhf_a_ao

    # Mean-field Fock (for diagonal energy preconditioner).
    # J/K are linear in density, so reuse the core/active builds instead of a 3rd JK call.
    Jt = Jc + Ja
    Kt = Kc + Ka
    f_ao = h_ao + (Jt - 0.5 * Kt)
    t_jk = time.perf_counter() if profile is not None else 0.0

    # Transform AO matrices to MO
    h_mo = C.T @ h_ao @ C
    vhf_c_mo = C.T @ vhf_c_ao @ C
    vhf_ca_mo = C.T @ vhf_ca_ao @ C
    f_mo = C.T @ f_ao @ C
    eps = xp.diag(f_mo)
    t_transform = time.perf_counter() if profile is not None else 0.0

    # Dense exact g_dm2.
    if xp is np:
        dense_prof = profile.setdefault("dense_g_dm2_cpu", {}) if profile is not None else None
        g_dm2 = _g_dm2_dense_cpu(
            ao_basis,
            C_mo=np.asarray(C, dtype=np.float64),
            C_act=np.asarray(C_act, dtype=np.float64),
            dm2_wxuv_flat=np.asarray(dm2_wxuv_xp, dtype=np.float64),
            eps_ao=float(dense_eps_ao),
            max_tile_bytes=int(dense_max_tile_bytes),
            threads=int(dense_cpu_threads),
            blas_nthreads=None if dense_cpu_blas_nthreads is None else int(dense_cpu_blas_nthreads),
            p_block_nmo=int(dense_cpu_p_block_nmo),
            profile=dense_prof,
        )
        g_dm2_xp = _as_xp_f64(xp, g_dm2)
    else:
        try:
            import cupy as cp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("dense GPU orbital gradients require CuPy") from e

        dense_prof = profile.setdefault("dense_pu_wx_gpu", {}) if profile is not None else None
        if dense_gpu_builder is not None:
            pu_wx = dense_gpu_builder.build_pu_wx_eri_mat(C, C_act, profile=dense_prof)
        else:
            from asuka.cueri import dense as cueri_dense  # noqa: PLC0415

            pu_wx = cueri_dense.build_pu_wx_eri_mat_dense_rys(
                ao_basis,
                C,
                C_act,
                threads=int(dense_gpu_threads),
                max_tile_bytes=int(dense_max_tile_bytes),
                eps_ao=float(dense_eps_ao),
                profile=dense_prof,
            )  # (nmo*ncas, ncas^2)
        pu_wx = cp.ascontiguousarray(pu_wx, dtype=cp.float64)
        pu_wx = pu_wx.reshape(nmo, ncas, ncas, ncas)  # (p,u,w,x)
        dm2_wxuv = dm2_wxuv_xp.reshape(ncas, ncas, ncas, ncas)  # (w,x,u,v)
        g_dm2_xp = cp.einsum("puwx,wxuv->pv", pu_wx, dm2_wxuv, optimize=True)
    t_gdm2 = time.perf_counter() if profile is not None else 0.0

    # gpq columns for core + active only (virtual columns are 0)
    gpq = xp.zeros((nmo, nmo), dtype=xp.float64)
    if ncore:
        gpq[:, :ncore] = 2.0 * (h_mo + vhf_ca_mo)[:, :ncore]
    gpq[:, ncore:nocc] = (h_mo + vhf_c_mo)[:, ncore:nocc] @ dm1_act_xp + g_dm2_xp

    G = gpq - gpq.T

    if allowed is None:
        allowed = allowed_rotation_mask(nmo, ncore, ncas)
    allowed_xp = xp.asarray(allowed)
    g_vec = G[allowed_xp].ravel()
    grad_norm = float(xp.linalg.norm(g_vec).item())
    if profile is not None:
        t_end = time.perf_counter()
        profile["is_gpu"] = bool(xp is not np)
        profile["nao"] = int(nao)
        profile["nmo"] = int(nmo)
        profile["ncas"] = int(ncas)
        profile["t_density_s"] = float(t_density - t0_total)
        profile["t_jk_s"] = float(t_jk - t_density)
        profile["t_transform_s"] = float(t_transform - t_jk)
        profile["t_gdm2_s"] = float(t_gdm2 - t_transform)
        profile["t_gpq_s"] = float(t_end - t_gdm2)
        profile["t_total_s"] = float(t_end - t0_total)
    return G, grad_norm, eps


__all__ = ["allowed_rotation_mask", "cayley_update", "orbital_gradient_df", "orbital_gradient_dense"]
