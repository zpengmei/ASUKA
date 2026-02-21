from __future__ import annotations

"""DF helpers for the Newton-CASSCF operator (gen_g_hop).

This module provides:
  1) A minimal eris-like container with the attributes required by
     :func:`asuka.mcscf.newton_casscf.gen_g_hop_internal`.
  2) A minimal CASSCF-like adapter object that exposes the subset of the PySCF
     CASSCF API used by the internal Newton operator.

Design goal
-----------
Tests may compare against PySCF.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.hf import df_scf as _df_scf
from asuka.mcscf.orbital_grad import cayley_update


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None and isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
        a = cp.asnumpy(a)
    return np.asarray(a, dtype=np.float64)


def _get_xp(*arrays: Any) -> tuple[Any, bool]:
    """Return (xp, is_gpu) where xp is numpy or cupy based on array types.

    Parameters
    ----------
    *arrays : Any
        Arrays to inspect.

    Returns
    -------
    xp : module
        The array module (numpy or cupy).
    is_gpu : bool
        Whether the arrays are on GPU.
    """
    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore
    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


def _as_xp_f64(xp: Any, a: Any) -> Any:
    """Convert array to float64 using the specified backend.

    Parameters
    ----------
    xp : Any
        The array module (numpy or cupy).
    a : Any
        Input array.

    Returns
    -------
    Any
        Converted array.
    """
    return xp.asarray(a, dtype=xp.float64)


@dataclass(frozen=True)
class DFNewtonERIs:
    """Minimal eris-like container for `newton_casscf.gen_g_hop_internal`.

    Attributes
    ----------
    ppaa : Any
        Integrals (nmo,nmo,ncas,ncas).
    papa : Any
        Integrals (nmo,ncas,nmo,ncas).
    vhf_c : Any
        Core HF potential (nmo,nmo).
    j_pc : Any
        Core Coulomb potential (nmo,ncore).
    k_pc : Any
        Core Exchange potential (nmo,ncore).

    Notes
    -----
    Attribute shapes match PySCF's `mc.ao2mo(mo)` ERIS object:
      - ppaa[p,q,u,v] = (p q|u v)
      - papa[p,u,q,v] = (p u|q v)
      - vhf_c[p,q]     = Veff_core[p,q] = J[D_core] - 0.5 K[D_core] in MO basis
      - j_pc[p,i]      = (p p|i i)
      - k_pc[p,i]      = (p i|i p)

    Arrays may be NumPy or CuPy depending on the build path.
    """

    ppaa: Any
    papa: Any
    vhf_c: Any
    j_pc: Any
    k_pc: Any


def build_df_newton_eris(
    B_ao: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
) -> DFNewtonERIs:
    """Build DF ERI intermediates required by the Newton-CASSCF operator.

    Parameters
    ----------
    B_ao : Any
        Density fitting tensor (nao, nao, naux).
    mo_coeff : Any
        Molecular orbital coefficients.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.

    Returns
    -------
    DFNewtonERIs
        The constructed ERI container.

    Notes
    -----
    GPU-aware: if B_ao is a CuPy array, all computation stays on GPU and
    the returned DFNewtonERIs contains CuPy arrays. Otherwise uses NumPy.
    """

    xp, _is_gpu = _get_xp(B_ao, mo_coeff)
    B = _as_xp_f64(xp, B_ao)
    mo = _as_xp_f64(xp, mo_coeff)

    ncore = int(ncore)
    ncas = int(ncas)
    if ncore < 0:
        raise ValueError("ncore must be >= 0")
    if ncas <= 0:
        raise ValueError("ncas must be > 0")
    if B.ndim != 3:
        raise ValueError("B_ao must have shape (nao,nao,naux)")
    if mo.ndim != 2:
        raise ValueError("mo_coeff must have shape (nao,nmo)")
    nao0, nao1, naux = map(int, B.shape)
    if nao0 != nao1:
        raise ValueError("B_ao must have shape (nao,nao,naux)")
    nao, nmo = map(int, mo.shape)
    if nao != nao0:
        raise ValueError("B_ao and mo_coeff nao mismatch")
    nocc = ncore + ncas
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    # L[p,q,Q] = sum_{mu,nu} C[mu,p] * B[mu,nu,Q] * C[nu,q]
    # Use tensordot for GPU GEMM acceleration
    tmp = xp.tensordot(B, mo, axes=([1], [0]))  # (nao, naux, nmo)
    X = xp.transpose(tmp, (0, 2, 1))  # (nao, nmo, naux)
    L = xp.tensordot(mo.T, X, axes=([1], [0]))  # (nmo, nmo, naux)
    L = xp.ascontiguousarray(xp.asarray(L, dtype=xp.float64))

    act = slice(ncore, nocc)
    L_act = xp.ascontiguousarray(xp.asarray(L[act, act], dtype=xp.float64))
    L_pu = xp.ascontiguousarray(xp.asarray(L[:, act], dtype=xp.float64))

    # (p q|u v) = sum_Q L[p,q,Q] L[u,v,Q]
    ppaa = xp.einsum("pqQ,uvQ->pquv", L, L_act, optimize=True)
    # (p u|q v) = sum_Q L[p,u,Q] L[q,v,Q]
    papa = xp.einsum("puQ,qvQ->puqv", L_pu, L_pu, optimize=True)

    ppaa = xp.ascontiguousarray(xp.asarray(ppaa, dtype=xp.float64))
    papa = xp.ascontiguousarray(xp.asarray(papa, dtype=xp.float64))

    # j_pc[p,i] = (p p|i i), k_pc[p,i] = (p i|i p)
    L_pp = xp.ascontiguousarray(L[xp.arange(nmo), xp.arange(nmo)])  # (nmo,naux)
    if ncore:
        L_ii = xp.ascontiguousarray(L_pp[:ncore])  # (ncore,naux)
        j_pc = xp.ascontiguousarray(L_pp @ L_ii.T)
        k_pc = xp.ascontiguousarray(
            xp.einsum("piQ,piQ->pi", L[:, :ncore], L[:, :ncore], optimize=True)
        )
    else:
        j_pc = xp.zeros((nmo, 0), dtype=xp.float64)
        k_pc = xp.zeros((nmo, 0), dtype=xp.float64)

    # vhf_c in MO basis from core density.
    if ncore:
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        Jc, Kc = _df_scf._df_JK(B, D_core, want_J=True, want_K=True)  # noqa: SLF001
        v_ao = xp.asarray(Jc - 0.5 * Kc, dtype=xp.float64)
        vhf_c = xp.ascontiguousarray(mo.T @ v_ao @ mo)
    else:
        vhf_c = xp.zeros((nmo, nmo), dtype=xp.float64)

    return DFNewtonERIs(ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc)


def build_dense_newton_eris(
    dense_gpu_builder: Any,
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
    B_ao_for_vhf: Any | None = None,
    ao_eri_for_vhf: Any | None = None,
) -> DFNewtonERIs:
    """Build Newton ERI intermediates from exact (dense) ERIs via GPU builder.

    Uses ``dense_gpu_builder.build_pq_uv_eri_mat`` for ppaa and
    ``dense_gpu_builder.build_pu_wx_eri_mat`` for papa.

    Parameters
    ----------
    dense_gpu_builder : CuERIActiveSpaceDenseGPUBuilder
        Reusable GPU builder with cached preprocessing.
    mo_coeff : Any
        Full MO coefficient matrix (nao, nmo).
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    B_ao_for_vhf : Any | None, optional
        DF B-tensor for building vhf_c. If None, falls back to ao_eri_for_vhf.
    ao_eri_for_vhf : Any | None, optional
        Materialized AO ERI tensor (nao*nao, nao*nao) for building vhf_c
        when B_ao_for_vhf is None (dense, no-DF mode).

    Returns
    -------
    DFNewtonERIs
        The constructed ERI container.
    """

    try:
        import cupy as cp
    except Exception as e:
        raise RuntimeError("CuPy required for dense Newton ERIs") from e

    ncore = int(ncore)
    ncas = int(ncas)
    nocc = ncore + ncas

    mo = cp.ascontiguousarray(cp.asarray(mo_coeff, dtype=cp.float64))
    nao, nmo = map(int, mo.shape)
    if nocc > nmo:
        raise ValueError("ncore+ncas exceeds nmo")

    C_act = cp.ascontiguousarray(mo[:, ncore:nocc])

    # ppaa[p,q,u,v] = (pq|uv) — shape (nmo*nmo, ncas*ncas)
    ppaa_flat = dense_gpu_builder.build_pq_uv_eri_mat(mo, C_act)
    ppaa = ppaa_flat.reshape(nmo, nmo, ncas, ncas)

    # papa[p,u,q,v] = (pu|qv) — shape (nmo*ncas, nmo*ncas)
    # build_pu_wx_eri_mat gives (pu|wx) with w,x active
    # For papa we need (pu|qv) where q is general MO
    # Use build_pq_uv_eri_mat with swapped pairs: (pu|qv) = ppaa with
    # first pair = (p,u) mixed, second pair = (q,v) mixed
    # Actually, we can get papa from ppaa by transposition:
    # papa[p,u,q,v] = ppaa[p,q,u,v] transposed as (0,2,1,3)
    # NO — (pu|qv) != (pq|uv) in general!
    # We need a separate build. For now, use the pu_wx builder with C_mo as
    # the "active" set for the CD pair. This requires a new builder call.
    # Alternative: build papa from ppaa using the Coulomb integral symmetry:
    # (pu|qv) = integral, (pq|uv) = integral — these are different integrals.
    #
    # Practical approach: build papa via a second pq_uv call with reordered
    # coefficients. (pu|qv) = sum_{mu nu la si} C_mu^p C_nu^u (mu nu|la si) C_la^q C_si^v
    # This is the same as (pq|uv) but with the coefficient assignment:
    #   AB pair: C_mo col p, C_act col u → mixed
    #   CD pair: C_mo col q, C_act col v → mixed
    # We can build this using build_pu_wx_eri_mat with C_mo for both pairs:
    # Actually build_pu_wx_eri_mat gives (pu|wx) where w,x are ACTIVE.
    # For papa we need (pu|qv) where q is GENERAL MO.
    # This is a genuinely different integral.
    #
    # Simplest correct approach: compute papa from the AO integrals directly.
    # Since we already have ppaa, we can compute papa via:
    # papa[p,u,q,v] = (pu|qv) which requires a separate 4-index transform.
    #
    # For now, approximate papa from ppaa using the DF relation as fallback:
    # papa[p,u,q,v] ≈ sum_Q L[p,u,Q] L[q,v,Q]
    # This is only used when B_ao_for_vhf is provided.
    if B_ao_for_vhf is not None:
        # Build papa from DF: L[p,q,Q] -> papa[p,u,q,v] = sum_Q L[p,u,Q] L[q,v,Q]
        B = cp.asarray(B_ao_for_vhf, dtype=cp.float64)
        tmp = cp.tensordot(B, mo, axes=([1], [0]))
        X = cp.transpose(tmp, (0, 2, 1))
        L = cp.tensordot(mo.T, X, axes=([1], [0]))
        L = cp.ascontiguousarray(cp.asarray(L, dtype=cp.float64))
        L_pu = cp.ascontiguousarray(L[:, ncore:nocc])
        papa = cp.einsum("puQ,qvQ->puqv", L_pu, L_pu, optimize=True)
        papa = cp.ascontiguousarray(cp.asarray(papa, dtype=cp.float64))
    elif ao_eri_for_vhf is not None:
        # Dense path: 4-index transform from AO ERIs for papa
        # papa[p,u,q,v] = (pu|qv) = sum_{μνλσ} C_μp C_νu eri[μνλσ] C_λq C_σv
        _ao_eri_4d = cp.asarray(ao_eri_for_vhf, dtype=cp.float64).reshape(nao, nao, nao, nao)
        # σ -> v
        _T1 = cp.tensordot(_ao_eri_4d, C_act, axes=([3], [0]))  # (nao,nao,nao,ncas)
        del _ao_eri_4d
        # λ -> q : result (nao,nao,ncas,nmo) with indices (μ,ν,v,q)
        _T2 = cp.tensordot(_T1, mo, axes=([2], [0]))
        del _T1
        # ν -> u : contract axis 1 of _T2 with C_act
        # _T2 shape (nao_μ, nao_ν, ncas_v, nmo_q)
        _T3 = cp.tensordot(_T2, C_act, axes=([1], [0]))  # (nao_μ, ncas_v, nmo_q, ncas_u)
        del _T2
        # Reorder to (μ, u, q, v)
        _T3 = _T3.transpose(0, 3, 2, 1)  # (nao, ncas, nmo, ncas)
        # μ -> p
        papa = cp.tensordot(mo.T, _T3, axes=([1], [0]))  # (nmo, ncas, nmo, ncas)
        del _T3
        papa = cp.ascontiguousarray(cp.asarray(papa, dtype=cp.float64))
    else:
        # No 2e source — zero papa (should not happen in practice)
        papa = cp.zeros((nmo, ncas, nmo, ncas), dtype=cp.float64)

    # j_pc[p,i] = (pp|ii), k_pc[p,i] = (pi|pi)  where i is a CORE orbital
    if ncore:
        if B_ao_for_vhf is not None:
            # DF path: compute from L[p,q,Q]
            L_pp = cp.ascontiguousarray(L[cp.arange(nmo), cp.arange(nmo)])  # (nmo,naux)
            L_ii = cp.ascontiguousarray(L_pp[:ncore])  # (ncore,naux)
            j_pc = cp.ascontiguousarray(L_pp @ L_ii.T)  # (nmo,ncore)
            k_pc = cp.ascontiguousarray(
                cp.einsum("piQ,piQ->pi", L[:, :ncore], L[:, :ncore], optimize=True)
            )
        elif ao_eri_for_vhf is not None:
            # Dense path: compute from AO ERIs
            # j_pc[p,i] = D_p^T @ eri_2d @ D_i  where D_x[μν] = C[μ,x]*C[ν,x]
            _ao_eri_2d = cp.asarray(ao_eri_for_vhf, dtype=cp.float64)
            if _ao_eri_2d.ndim == 4:
                _ao_eri_2d = _ao_eri_2d.reshape(nao * nao, nao * nao)
            # D_diag[x, μν] = C[μ,x] * C[ν,x]
            _D_all = (mo[:, :, None] * mo[:, None, :])  # (nao, nao, nmo) — outer products
            _D_all = _D_all.transpose(2, 0, 1).reshape(nmo, nao * nao)  # (nmo, nao^2)
            _D_core = _D_all[:ncore]  # (ncore, nao^2)
            _T = _ao_eri_2d @ _D_core.T  # (nao^2, ncore)
            j_pc = cp.ascontiguousarray(_D_all @ _T)  # (nmo, ncore)
            # k_pc[p,i] = K_pi^T @ eri_2d @ K_pi  where K_pi[μν] = C[μ,p]*C[ν,i]
            mo_core = mo[:, :ncore]  # (nao, ncore)
            # K_pi[μ,ν] = C[μ,p]*C[ν,i] → K[p,i,μ,ν] = C[μ,p]*C[ν,i]
            # Vectorized: K_flat[p*ncore+i, μ*nao+ν] = C[μ,p]*C[ν,i]
            # = (C ⊗ C_core)[μν, pi]
            # Reshape for batch matmul: for each i,
            # k_pc[p,i] = sum_{μλ} C[μ,p] * W_i[μ,λ] * C[λ,p]
            # where W_i[μ,λ] = sum_{νσ} C[ν,i] * eri[μν,λσ] * C[σ,i]
            k_pc = cp.zeros((nmo, ncore), dtype=cp.float64)
            _eri_4d = _ao_eri_2d.reshape(nao, nao, nao, nao)
            for i in range(ncore):
                _ci = mo_core[:, i]  # (nao,)
                # W_i[μ,λ] = sum_{νσ} C[ν,i] * eri[μ,ν,λ,σ] * C[σ,i]
                _W = cp.einsum("n,mnls,s->ml", _ci, _eri_4d, _ci, optimize=True)
                # k_pc[p,i] = C[:,p]^T @ W @ C[:,p] for all p
                _CW = mo.T @ _W  # (nmo, nao)
                k_pc[:, i] = cp.einsum("pn,pn->p", _CW, mo.T)
            del _eri_4d, _ao_eri_2d
            k_pc = cp.ascontiguousarray(k_pc)
        else:
            j_pc = cp.zeros((nmo, ncore), dtype=cp.float64)
            k_pc = cp.zeros((nmo, ncore), dtype=cp.float64)
    else:
        j_pc = cp.zeros((nmo, 0), dtype=cp.float64)
        k_pc = cp.zeros((nmo, 0), dtype=cp.float64)

    # vhf_c: core Fock potential in MO basis
    if B_ao_for_vhf is not None and ncore:
        # DF path
        B = cp.asarray(B_ao_for_vhf, dtype=cp.float64)
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        Jc, Kc = _df_scf._df_JK(B, D_core, want_J=True, want_K=True)  # noqa: SLF001
        v_ao = cp.asarray(Jc - 0.5 * Kc, dtype=cp.float64)
        vhf_c = cp.ascontiguousarray(mo.T @ v_ao @ mo)
    elif ao_eri_for_vhf is not None and ncore:
        # Dense path: use materialized AO ERIs for J/K
        from asuka.hf.dense_jk import dense_JK_from_eri_mat_D  # noqa: PLC0415

        _ao_eri = cp.asarray(ao_eri_for_vhf, dtype=cp.float64)
        mo_core = mo[:, :ncore]
        D_core = 2.0 * (mo_core @ mo_core.T)
        Jc, Kc = dense_JK_from_eri_mat_D(_ao_eri, D_core, want_J=True, want_K=True)
        v_ao = cp.asarray(Jc - 0.5 * Kc, dtype=cp.float64)
        vhf_c = cp.ascontiguousarray(mo.T @ v_ao @ mo)
    else:
        vhf_c = cp.zeros((nmo, nmo), dtype=cp.float64)

    return DFNewtonERIs(ppaa=ppaa, papa=papa, vhf_c=vhf_c, j_pc=j_pc, k_pc=k_pc)


@dataclass
class DFNewtonCASSCFAdapter:
    """Minimal CASSCF-like adapter for `newton_casscf.gen_g_hop_internal`.

    This is intentionally small: it only implements what the internal operator
    needs. It can wrap ASUKA's DF SCF output + CI solver without importing PySCF.

    Attributes
    ----------
    df_B : Any
        Density fitting tensor (numpy or cupy).
    hcore_ao : Any
        Core Hamiltonian in AO basis.
    ncore : int
        Number of core orbitals.
    ncas : int
        Number of active orbitals.
    nelecas : int | tuple[int, int]
        Number of active electrons.
    mo_coeff : Any
        MO coefficients.
    fcisolver : Any
        FCI solver object.
    dense_gpu_builder : Any, optional
        CuERIActiveSpaceDenseGPUBuilder for dense ERIs.
    weights : list[float] | None
        State weights for SA-CASSCF.
    frozen : Any | None
        Frozen orbitals.
    internal_rotation : bool
        Whether internal rotation (active-active) is redundant.
    extrasym : Any | None
        Symmetry constraints.

    Notes
    -----
    df_B may be a CuPy array for GPU-accelerated AH. All downstream operations
    (ao2mo, update_jk_in_ah) will auto-detect and stay on GPU.

    When df_B is None, ao_eri (materialized AO ERI tensor) is used for J/K
    computation instead. This enables fully dense (no-DF) CASSCF with CUDA.
    """

    df_B: Any  # (nao,nao,naux) — numpy or cupy, can be None for dense mode
    hcore_ao: Any
    ncore: int
    ncas: int
    nelecas: int | tuple[int, int]
    mo_coeff: Any
    fcisolver: Any
    dense_gpu_builder: Any = None  # CuERIActiveSpaceDenseGPUBuilder, optional
    ao_eri: Any = None  # (nao*nao, nao*nao) — for dense J/K when df_B is None

    # Optional knobs (PySCF-compatible names)
    weights: list[float] | None = None
    frozen: Any | None = None
    internal_rotation: bool = False
    extrasym: Any | None = None

    def _get_2e_probe(self) -> Any:
        """Return the first non-None 2e integral source for xp detection."""
        if self.df_B is not None:
            return self.df_B
        if self.ao_eri is not None:
            return self.ao_eri
        return self.hcore_ao

    def get_hcore(self) -> Any:
        """Return the core Hamiltonian in AO basis (on correct device)."""
        xp, _is_gpu = _get_xp(self._get_2e_probe(), self.hcore_ao)
        return _as_xp_f64(xp, self.hcore_ao)

    def ao2mo(self, mo_coeff: Any) -> DFNewtonERIs:
        """Construct the ERI object for the given MOs.

        Parameters
        ----------
        mo_coeff : Any
            Molecular orbital coefficients.

        Returns
        -------
        DFNewtonERIs
            The ERI container.
        """
        if self.dense_gpu_builder is not None:
            return build_dense_newton_eris(
                self.dense_gpu_builder,
                mo_coeff,
                ncore=int(self.ncore),
                ncas=int(self.ncas),
                B_ao_for_vhf=self.df_B,
                ao_eri_for_vhf=self.ao_eri,
            )
        if self.df_B is None:
            raise ValueError("ao2mo requires df_B or dense_gpu_builder")
        return build_df_newton_eris(self.df_B, mo_coeff, ncore=int(self.ncore), ncas=int(self.ncas))

    def uniq_var_indices(self, nmo: int, ncore: int, ncas: int, frozen: Any | None) -> np.ndarray:
        """Return boolean mask of independent orbital rotation parameters.

        Parameters
        ----------
        nmo : int
            Number of molecular orbitals.
        ncore : int
            Number of core orbitals.
        ncas : int
            Number of active orbitals.
        frozen : Any | None
            Frozen orbitals.

        Returns
        -------
        np.ndarray
            Boolean mask (nmo, nmo) where True elements are independent parameters.
        """
        nmo = int(nmo)
        ncore = int(ncore)
        ncas = int(ncas)
        nocc = ncore + ncas
        mask = np.zeros((nmo, nmo), dtype=bool)
        mask[ncore:nocc, :ncore] = True
        mask[nocc:, :nocc] = True
        if bool(self.internal_rotation):
            mask[ncore:nocc, ncore:nocc][np.tril_indices(ncas, -1)] = True
        if self.extrasym is not None:
            extrasym = np.asarray(self.extrasym)
            extrasym_allowed = extrasym.reshape(-1, 1) == extrasym
            mask = mask & extrasym_allowed
        if frozen is not None:
            if isinstance(frozen, (int, np.integer)):
                mask[: int(frozen)] = False
                mask[:, : int(frozen)] = False
            else:
                frozen_idx = np.asarray(frozen, dtype=np.int32).ravel()
                mask[frozen_idx] = False
                mask[:, frozen_idx] = False
        return mask

    def pack_uniq_var(self, mat: Any) -> np.ndarray:
        """Pack a full anti-symmetric matrix into a flat independent-parameter vector.

        Parameters
        ----------
        mat : Any
            The full matrix (numpy or CuPy).

        Returns
        -------
        numpy or CuPy array
            Flattened vector of independent parameters (same backend as input).
        """
        xp, _on_gpu = _get_xp(mat)
        mat = xp.asarray(mat, dtype=xp.float64)
        nmo = int(self.mo_coeff.shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        return xp.asarray(mat[idx], dtype=xp.float64)

    def unpack_uniq_var(self, v: Any) -> np.ndarray:
        """Unpack a flat independent-parameter vector into a full anti-symmetric matrix.

        Parameters
        ----------
        v : Any
            The flattened vector (numpy or CuPy).

        Returns
        -------
        numpy or CuPy array
            The full anti-symmetric matrix (nmo, nmo), same backend as input.
        """
        xp, _on_gpu = _get_xp(v)
        v = xp.asarray(v, dtype=xp.float64).ravel()
        nmo = int(self.mo_coeff.shape[1])
        idx = self.uniq_var_indices(nmo, int(self.ncore), int(self.ncas), self.frozen)
        mat = xp.zeros((nmo, nmo), dtype=xp.float64)
        mat[idx] = v
        return mat - mat.T

    def update_rotate_matrix(self, dx: Any, u0: Any = 1) -> np.ndarray:
        """Apply orbital rotation `dx` to `u0`.

        Parameters
        ----------
        dx : Any
            Parameter update vector (packed).
        u0 : Any, optional
            Current rotation matrix. Defaults to 1.

        Returns
        -------
        np.ndarray
            Updated rotation matrix.
        """
        dr = self.unpack_uniq_var(dx)
        u = cayley_update(np, dr)
        return np.dot(u0, np.asarray(u, dtype=np.float64))

    def update_jk_in_ah(
        self,
        mo: Any,
        r: Any,
        casdm1: Any,
        eris: Any | None = None,
        *,
        return_gpu: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """DF analogue of PySCF `mc1step.CASSCF.update_jk_in_ah`.

        Parameters
        ----------
        mo : Any
            Molecular orbitals.
        r : Any
            Orbital rotation matrix (anti-symmetric).
        casdm1 : Any
            Active space density matrix.
        eris : Any | None, optional
            Integral object (unused, for compatibility).

        Returns
        -------
        tuple
            (va, vc) where va is active-space potential and vc is core potential update.

        Notes
        -----
        GPU-aware: uses the same array backend as self.df_B (or self.ao_eri).
        """

        _ = eris  # unused (kept for PySCF signature compatibility)

        ncore = int(self.ncore)
        ncas = int(self.ncas)
        nocc = ncore + ncas

        xp, _is_gpu = _get_xp(self._get_2e_probe())
        mo = _as_xp_f64(xp, mo)
        r = _as_xp_f64(xp, r)
        casdm1 = _as_xp_f64(xp, casdm1)

        if mo.ndim != 2:
            raise ValueError("mo must be 2D (nao,nmo)")
        nao, nmo = map(int, mo.shape)
        if nocc > nmo:
            raise ValueError("ncore+ncas exceeds nmo")
        if r.shape != (nmo, nmo):
            raise ValueError("r must be (nmo,nmo)")
        if casdm1.shape != (ncas, ncas):
            raise ValueError("casdm1 must be (ncas,ncas)")

        # dm3 = mo_core @ r_core,rest @ mo_rest^T  (+ sym)
        dm3 = mo[:, :ncore] @ r[:ncore, ncore:] @ mo[:, ncore:].T
        dm3 = dm3 + dm3.T

        # dm4 = mo_act @ casdm1 @ r_act,all @ mo^T (+ sym)
        dm4 = mo[:, ncore:nocc] @ casdm1 @ r[ncore:nocc] @ mo.T
        dm4 = dm4 + dm4.T

        from asuka.mcscf.jk_util import jk_from_2e_source  # noqa: PLC0415

        _df_B_xp = _as_xp_f64(xp, self.df_B) if self.df_B is not None else None
        _ao_eri_xp = _as_xp_f64(xp, self.ao_eri) if self.ao_eri is not None else None
        J0, K0 = jk_from_2e_source(_df_B_xp, _ao_eri_xp, dm3, want_J=True, want_K=True)
        J1, K1 = jk_from_2e_source(_df_B_xp, _ao_eri_xp, dm3 * 2.0 + dm4, want_J=True, want_K=True)

        v0 = xp.asarray(J0 * 2.0 - K0, dtype=xp.float64)
        v1 = xp.asarray(J1 * 2.0 - K1, dtype=xp.float64)

        mo_act = mo[:, ncore:nocc]
        mo_core = mo[:, :ncore]

        va = casdm1 @ mo_act.T @ v0 @ mo
        vc = mo_core.T @ v1 @ mo[:, ncore:]

        va_cont = xp.ascontiguousarray(xp.asarray(va, dtype=xp.float64))
        vc_cont = xp.ascontiguousarray(xp.asarray(vc, dtype=xp.float64))

        if return_gpu:
            return va_cont, vc_cont

        return _asnumpy_f64(va_cont), _asnumpy_f64(vc_cont)
