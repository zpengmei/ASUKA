"""DF/Cholesky-vector Fock matrix construction for CASPT2.

This module provides two backends for constructing CASPT2 Fock matrices:

1. ``build_caspt2_fock_ao`` (preferred): builds J/K in AO basis via the SCF
   ``_df_JK`` routine and transforms to MO.  This never materialises MO pair
   blocks and avoids the expensive virtual–virtual DF allocation entirely.

2. ``build_caspt2_fock_df`` (legacy): contracts MO-basis DF pair blocks
   (``CASPT2DFBlocks``).  Requires all six pair blocks including ``l_ab``.

Both produce a ``CASPT2Fock`` that is semantically identical to the full-ERI
version in :func:`asuka.caspt2.fock.build_caspt2_fock`.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from asuka.caspt2.cuda.rhs_df_cuda import CASPT2DFBlocks
from asuka.caspt2.fock import CASPT2Fock


def _xp_from_name(xp: str):
    mode = str(xp).strip().lower()
    if mode in ("numpy", "np"):
        return np
    if mode in ("cupy", "cp"):
        import cupy as cp  # noqa: PLC0415

        return cp
    raise ValueError("xp must be 'numpy' or 'cupy'")


def _as_contig(xp_mod, a, *, dtype):
    arr = xp_mod.asarray(a, dtype=dtype)
    return xp_mod.ascontiguousarray(arr)


def build_caspt2_fock_df(
    h1e_mo: np.ndarray,
    df_blocks: CASPT2DFBlocks,
    dm1_act: np.ndarray,
    nish: int,
    nash: int,
    nssh: int,
    *,
    e_nuc: float = 0.0,
    xp: Literal["numpy", "cupy"] = "cupy",
) -> CASPT2Fock:
    """Build CASPT2 Fock matrices using DF/Cholesky MO pair factors.

    Parameters
    ----------
    h1e_mo
        (nmo,nmo) core Hamiltonian in MO basis.
    df_blocks
        DF pair blocks containing at least (ii,it,ia,tu,at,ab) for the requested
        orbital partition. Empty blocks are allowed when their corresponding
        subspace size is zero (e.g. nish==0).
    dm1_act
        (nash,nash) active-space 1-RDM (Molcas convention is fine; only symmetry
        and trace matter for Fock build).
    nish, nash, nssh
        Number of inactive, active, and secondary orbitals.
    e_nuc
        Nuclear repulsion energy.
    xp
        "cupy" (compute on GPU, then return NumPy arrays) or "numpy".
    """

    nish = int(nish)
    nash = int(nash)
    nssh = int(nssh)
    nmo = nish + nash + nssh
    if nmo <= 0:
        raise ValueError("invalid orbital partition: nmo <= 0")

    h1e_mo = np.asarray(h1e_mo, dtype=np.float64)
    if h1e_mo.shape != (nmo, nmo):
        raise ValueError(f"h1e_mo shape {h1e_mo.shape} != ({nmo}, {nmo})")

    xp_mod = _xp_from_name(str(xp))
    f64 = xp_mod.float64

    dm1 = _as_contig(xp_mod, dm1_act, dtype=f64)
    if tuple(dm1.shape) != (nash, nash):
        raise ValueError(f"dm1_act shape {tuple(dm1.shape)} != ({nash}, {nash})")

    # Required DF blocks.
    # - Core blocks are optional when nish==0.
    # - Virtual blocks are optional when nssh==0.
    if nish > 0 and df_blocks.l_ii is None:
        raise ValueError("DF Fock build requires df_blocks.l_ii when nish > 0")
    # l_ab is no longer required — use build_caspt2_fock_ao() instead for
    # the preferred path that avoids the O(nvirt^2 * naux) allocation.

    l_it = _as_contig(xp_mod, df_blocks.l_it.l_full, dtype=f64)
    l_ia = _as_contig(xp_mod, df_blocks.l_ia.l_full, dtype=f64)
    l_at = _as_contig(xp_mod, df_blocks.l_at.l_full, dtype=f64)
    l_tu = _as_contig(xp_mod, df_blocks.l_tu.l_full, dtype=f64)

    # naux from any available block (prefer tu).
    naux = int(getattr(df_blocks.l_tu, "naux", 0))
    if naux <= 0:
        # Fallback: probe other blocks.
        for blk in (df_blocks.l_it, df_blocks.l_ia, df_blocks.l_at):
            naux = int(getattr(blk, "naux", 0))
            if naux > 0:
                break
    if naux <= 0:
        raise ValueError("DF blocks have invalid naux (<= 0)")

    l_ii = None
    l_ab = None
    if nish > 0 and df_blocks.l_ii is not None:
        l_ii = _as_contig(xp_mod, df_blocks.l_ii.l_full, dtype=f64)
    if nssh > 0 and df_blocks.l_ab is not None:
        l_ab = _as_contig(xp_mod, df_blocks.l_ab.l_full, dtype=f64)

    # Reshape to (x,y,P) blocks where needed.
    l_it3 = l_it.reshape(nish, nash, naux) if nish > 0 and nash > 0 else xp_mod.empty((nish, nash, naux), dtype=f64)
    l_ia3 = l_ia.reshape(nish, nssh, naux) if nish > 0 and nssh > 0 else xp_mod.empty((nish, nssh, naux), dtype=f64)
    l_tu3 = l_tu.reshape(nash, nash, naux) if nash > 0 else xp_mod.empty((nash, nash, naux), dtype=f64)
    l_at3 = l_at.reshape(nssh, nash, naux) if nssh > 0 and nash > 0 else xp_mod.empty((nssh, nash, naux), dtype=f64)
    l_ii3 = l_ii.reshape(nish, nish, naux) if (nish > 0 and l_ii is not None) else xp_mod.empty((nish, nish, naux), dtype=f64)
    l_ab3 = l_ab.reshape(nssh, nssh, naux) if (nssh > 0 and l_ab is not None) else xp_mod.empty((nssh, nssh, naux), dtype=f64)

    # ---------------------------------------------------------------------
    # Core Coulomb (J_core) and exchange (K_core)
    # ---------------------------------------------------------------------
    j_core = xp_mod.zeros((nmo, nmo), dtype=f64)
    k_core = xp_mod.zeros((nmo, nmo), dtype=f64)
    if nish > 0:
        # g_core[P] = sum_i L_ii(i,i,P)
        g_core = xp_mod.einsum("iiP->P", l_ii3, optimize=True)

        # J_core blocks: J(p,q) = sum_P L_pq(P) * g_core(P)
        j_core[:nish, :nish] = (l_ii @ g_core).reshape(nish, nish)
        if nash > 0:
            j_it = (l_it @ g_core).reshape(nish, nash)
            j_core[:nish, nish : nish + nash] = j_it
            j_core[nish : nish + nash, :nish] = j_it.T
            j_core[nish : nish + nash, nish : nish + nash] = (l_tu @ g_core).reshape(nash, nash)
        if nssh > 0:
            j_ia = (l_ia @ g_core).reshape(nish, nssh)
            j_core[:nish, nish + nash :] = j_ia
            j_core[nish + nash :, :nish] = j_ia.T
            j_at = (l_at @ g_core).reshape(nssh, nash)
            j_core[nish + nash :, nish : nish + nash] = j_at
            j_core[nish : nish + nash, nish + nash :] = j_at.T
            if l_ab is not None:
                j_core[nish + nash :, nish + nash :] = (l_ab @ g_core).reshape(nssh, nssh)

        # K_core(p,q) = sum_i (p i| q i) ~= sum_{i,P} L_{p i}(P) L_{q i}(P)
        # Build L_{p i}(P) for all p in MO and i in core.
        l_ti3 = l_it3.transpose(1, 0, 2) if (nash > 0) else xp_mod.empty((0, nish, naux), dtype=f64)
        l_ai3 = l_ia3.transpose(1, 0, 2) if (nssh > 0) else xp_mod.empty((0, nish, naux), dtype=f64)
        l_pi3 = xp_mod.concatenate([l_ii3, l_ti3, l_ai3], axis=0)  # (nmo, nish, naux)
        a_mat = l_pi3.reshape(nmo, nish * naux)
        k_core = a_mat @ a_mat.T

    # ---------------------------------------------------------------------
    # Active Coulomb (J_act) and exchange (K_act)
    # ---------------------------------------------------------------------
    j_act = xp_mod.zeros((nmo, nmo), dtype=f64)
    k_act = xp_mod.zeros((nmo, nmo), dtype=f64)
    if nash > 0:
        # g_act[P] = sum_{t,u} dm1[t,u] * L_tu(t,u,P)
        g_act = xp_mod.einsum("tu,tuP->P", dm1, l_tu3, optimize=True)

        # J_act blocks
        if nish > 0:
            j_ii_act = (l_ii @ g_act).reshape(nish, nish)
            j_act[:nish, :nish] = j_ii_act
            j_it_act = (l_it @ g_act).reshape(nish, nash)
            j_act[:nish, nish : nish + nash] = j_it_act
            j_act[nish : nish + nash, :nish] = j_it_act.T
        j_act[nish : nish + nash, nish : nish + nash] = (l_tu @ g_act).reshape(nash, nash)
        if nssh > 0:
            if nish > 0:
                j_ia_act = (l_ia @ g_act).reshape(nish, nssh)
                j_act[:nish, nish + nash :] = j_ia_act
                j_act[nish + nash :, :nish] = j_ia_act.T
            j_at_act = (l_at @ g_act).reshape(nssh, nash)
            j_act[nish + nash :, nish : nish + nash] = j_at_act
            j_act[nish : nish + nash, nish + nash :] = j_at_act.T
            if l_ab is not None:
                j_act[nish + nash :, nish + nash :] = (l_ab @ g_act).reshape(nssh, nssh)

        # K_act(p,q) = sum_{t,u} dm1[t,u] (p t| q u)
        # Use eigen-decomposition of symmetric dm1: dm1 = U diag(w) U^T
        dm1_sym = 0.5 * (dm1 + dm1.T)
        w, u = xp_mod.linalg.eigh(dm1_sym)

        # B[p, l, P] = sum_t u[t,l] * L_{p t}(P)
        # Core: L_{i t}
        b_core = xp_mod.empty((nish, nash, naux), dtype=f64)
        if nish > 0:
            tmp = xp_mod.tensordot(l_it3, u, axes=([1], [0]))  # (i,P,l)
            b_core = tmp.transpose(0, 2, 1)  # (i,l,P)
        else:
            b_core = xp_mod.empty((0, nash, naux), dtype=f64)

        # Active: L_{t u} with first index as "p" in active space.
        tmp = xp_mod.tensordot(l_tu3, u, axes=([1], [0]))  # (p,P,l)
        b_act = tmp.transpose(0, 2, 1)  # (p,l,P)

        # Virtual: L_{a t}
        if nssh > 0:
            tmp = xp_mod.tensordot(l_at3, u, axes=([1], [0]))  # (a,P,l)
            b_virt = tmp.transpose(0, 2, 1)  # (a,l,P)
        else:
            b_virt = xp_mod.empty((0, nash, naux), dtype=f64)

        b_all = xp_mod.concatenate([b_core, b_act, b_virt], axis=0)  # (nmo, nash, naux)

        # K_act(p,q) = sum_l w_l * (B_l @ B_l^T), where w_l may have small
        # negative values from numerical noise. Split by sign to avoid complex
        # square-roots while preserving exact symmetry for any symmetric dm1.
        pos = xp_mod.where(w > 0.0)[0]
        neg = xp_mod.where(w < 0.0)[0]
        if int(pos.size) > 0:
            sqrt_w_pos = xp_mod.sqrt(w[pos])
            b_pos = b_all[:, pos, :] * sqrt_w_pos[None, :, None]
            c_pos = b_pos.reshape(nmo, int(pos.size) * naux)
            k_act = k_act + (c_pos @ c_pos.T)
        if int(neg.size) > 0:
            sqrt_w_neg = xp_mod.sqrt(-w[neg])
            b_neg = b_all[:, neg, :] * sqrt_w_neg[None, :, None]
            c_neg = b_neg.reshape(nmo, int(neg.size) * naux)
            k_act = k_act - (c_neg @ c_neg.T)

    # ---------------------------------------------------------------------
    # Assemble CASPT2 Fock blocks
    # ---------------------------------------------------------------------
    h1 = _as_contig(xp_mod, h1e_mo, dtype=f64)
    fimo = h1 + 2.0 * j_core - k_core
    famo = j_act - 0.5 * k_act
    fifa = fimo + famo

    act = slice(nish, nish + nash)
    epsa = xp_mod.diag(fifa[act, act]).copy() if nash > 0 else xp_mod.zeros((0,), dtype=f64)

    # Core energy: E_core = sum_i [h_ii + fimo_ii] + E_nuc
    e_core = float(e_nuc)
    if nish > 0:
        diag_h = xp_mod.diag(h1[:nish, :nish])
        diag_f = xp_mod.diag(fimo[:nish, :nish])
        e_core += float(xp_mod.sum(diag_h + diag_f))

    # Return NumPy arrays for compatibility with existing CASPT2 code.
    if xp_mod is np:
        return CASPT2Fock(
            fimo=np.asarray(fimo, dtype=np.float64, order="C"),
            famo=np.asarray(famo, dtype=np.float64, order="C"),
            fifa=np.asarray(fifa, dtype=np.float64, order="C"),
            epsa=np.asarray(epsa, dtype=np.float64),
            e_core=float(e_core),
        )

    # CuPy -> NumPy transfer.
    import cupy as cp  # noqa: PLC0415

    return CASPT2Fock(
        fimo=np.asarray(cp.asnumpy(fimo), dtype=np.float64, order="C"),
        famo=np.asarray(cp.asnumpy(famo), dtype=np.float64, order="C"),
        fifa=np.asarray(cp.asnumpy(fifa), dtype=np.float64, order="C"),
        epsa=np.asarray(cp.asnumpy(epsa), dtype=np.float64),
        e_core=float(e_core),
    )


# ---------------------------------------------------------------------------
# AO-based Fock builder (preferred — avoids all MO pair block allocations)
# ---------------------------------------------------------------------------


def build_caspt2_fock_ao(
    h1e_ao: np.ndarray,
    B_ao,
    C,
    dm1_act: np.ndarray,
    nish: int,
    nash: int,
    nssh: int,
    *,
    e_nuc: float = 0.0,
) -> CASPT2Fock:
    """Build CASPT2 Fock matrices from AO-basis DF factors.

    Instead of constructing MO-basis DF pair blocks (including the expensive
    virtual–virtual block ``l_ab``), this function builds J and K directly
    in AO basis using :func:`asuka.hf.df_scf._df_JK` and transforms the
    result to the MO basis.

    The split Fock ``F = h + 2J_core - K_core + J_act - 0.5 K_act`` is
    equivalent to two standard J/K builds with AO densities:

    - ``fimo = h + J[D_core] - 0.5 K[D_core]``  where ``D_core = 2 C_i C_i^T``
    - ``famo = J[D_act] - 0.5 K[D_act]``         where ``D_act  = C_t γ C_t^T``

    Parameters
    ----------
    h1e_ao : (nao, nao)
        Core Hamiltonian in AO basis.
    B_ao
        DF factors ``(nao, nao, naux)``.  May be NumPy or CuPy.
    C
        MO coefficients ``(nao, nmo)``.  May be NumPy or CuPy.
    dm1_act : (nash, nash)
        Active-space 1-RDM.
    nish, nash, nssh : int
        Number of inactive, active, and secondary orbitals.
    e_nuc : float
        Nuclear repulsion energy.
    """
    from asuka.hf.df_scf import _df_JK  # noqa: PLC0415

    nish = int(nish)
    nash = int(nash)
    nssh = int(nssh)
    nmo = nish + nash + nssh
    if nmo <= 0:
        raise ValueError("invalid orbital partition: nmo <= 0")

    # Detect backend (NumPy vs CuPy) from B_ao.
    xp = np
    _is_cupy = False
    if hasattr(B_ao, "__cuda_array_interface__") or (
        hasattr(B_ao, "__class__") and "cupy" in type(B_ao).__module__
    ):
        import cupy as cp  # noqa: PLC0415

        xp = cp
        _is_cupy = True

    f64 = xp.float64
    B = xp.asarray(B_ao, dtype=f64)
    C_full = xp.asarray(C, dtype=f64)
    h_ao = xp.asarray(h1e_ao, dtype=f64)
    dm1 = xp.asarray(dm1_act, dtype=f64)

    if B.ndim != 3:
        raise ValueError("B_ao must have shape (nao, nao, naux)")
    if C_full.ndim != 2:
        raise ValueError("C must have shape (nao, nmo)")
    nao = int(C_full.shape[0])
    if int(C_full.shape[1]) != nmo:
        raise ValueError(f"C has {int(C_full.shape[1])} MOs, expected {nmo}")
    if h_ao.shape != (nao, nao):
        raise ValueError(f"h1e_ao shape {tuple(h_ao.shape)} != ({nao}, {nao})")
    if dm1.shape != (nash, nash):
        raise ValueError(f"dm1_act shape {tuple(dm1.shape)} != ({nash}, {nash})")

    C_core = C_full[:, :nish]
    C_act = C_full[:, nish : nish + nash]

    # -- AO densities --
    if nish > 0:
        D_core = 2.0 * (C_core @ C_core.T)  # (nao, nao)
    else:
        D_core = xp.zeros((nao, nao), dtype=f64)

    if nash > 0:
        D_act = C_act @ dm1 @ C_act.T  # (nao, nao)
    else:
        D_act = xp.zeros((nao, nao), dtype=f64)

    # -- J/K via existing SCF DF routine --
    J_core, K_core = _df_JK(B, D_core, want_J=True, want_K=True)
    J_act, K_act = _df_JK(B, D_act, want_J=True, want_K=True)

    # -- AO Fock matrices --
    F_imo_ao = h_ao + J_core - 0.5 * K_core
    F_amo_ao = J_act - 0.5 * K_act

    # -- Transform to MO --
    fimo = C_full.T @ F_imo_ao @ C_full
    famo = C_full.T @ F_amo_ao @ C_full
    fifa = fimo + famo

    act = slice(nish, nish + nash)
    epsa = xp.diag(fifa[act, act]).copy() if nash > 0 else xp.zeros((0,), dtype=f64)

    # -- Core energy: E_core = sum_i [h_ii + fimo_ii] + E_nuc --
    h_mo = C_full.T @ h_ao @ C_full
    e_core = float(e_nuc)
    if nish > 0:
        diag_h = xp.diag(h_mo[:nish, :nish])
        diag_f = xp.diag(fimo[:nish, :nish])
        e_core += float(xp.sum(diag_h + diag_f))

    # -- Return NumPy arrays --
    def _to_np(a):
        if _is_cupy:
            import cupy as cp  # noqa: PLC0415

            return np.asarray(cp.asnumpy(a), dtype=np.float64, order="C")
        return np.asarray(a, dtype=np.float64, order="C")

    return CASPT2Fock(
        fimo=_to_np(fimo),
        famo=_to_np(famo),
        fifa=_to_np(fifa),
        epsa=np.asarray(_to_np(epsa), dtype=np.float64),
        e_core=float(e_core),
    )
