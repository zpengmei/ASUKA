from __future__ import annotations

"""Contracted ic-MRCISD density builders.

This module provides 1- and 2-RDM builders for internal contracted MRCISD (FIC and SC).

Notes
-----
*   **Exact contracted densities** for the full [ref + singles + doubles] space generally
    require on-the-fly higher-order internal contractions (up to 3-/4-body internal
    information), especially for doubles contributions.
*   **Analytic gradients** require per-state correlated-space dm1/dm2 and the
    generalized Fock matrix built from them.
*   The module provides a stable API for density builders, with "reconstruct"
    backends for validation (expanding to uncontracted CSF CI vectors) and
    "direct" backends for production/large systems (avoiding full reconstruction).
"""

from typing import Any, Literal, Tuple

import numpy as np

from asuka.cuguga.drt import DRT
from asuka.mrci.ic_basis import ICDoubles, ICSingles, SCDoubles, SCSingles


Contraction = Literal["fic", "sc"]
DensityBackend = Literal["reconstruct", "direct"]


def _build_cas_drt_for_ic_res(ic_res: Any, *, n_act: int) -> DRT:
    """Build a CAS DRT consistent with `ci_cas` ordering for Phase-3 higher-order contractions."""

    drt_work = getattr(ic_res, "drt_work", None)
    if drt_work is None:
        raise NotImplementedError("Phase-3 dm2 blocks require ic_res.drt_work (semi-direct backend)")
    if not isinstance(drt_work, DRT):
        raise TypeError("ic_res.drt_work must be a DRT instance")

    nelec = int(getattr(drt_work, "nelec"))
    twos = int(getattr(drt_work, "twos_target"))

    spaces = getattr(ic_res, "spaces", None)
    orbsym_act = None
    if spaces is not None:
        orbsym = getattr(spaces, "orbsym", None)
        if orbsym is not None:
            orbsym_act = np.asarray(orbsym, dtype=np.int32).ravel()[: int(n_act)].tolist()

    wfnsym = None
    try:
        wfnsym = int(np.asarray(getattr(drt_work, "node_sym"))[int(drt_work.leaf)])
    except Exception:  # pragma: no cover
        wfnsym = None

    from asuka.cuguga import build_drt  # noqa: PLC0415

    return build_drt(norb=int(n_act), nelec=nelec, twos_target=twos, orbsym=orbsym_act, wfnsym=wfnsym)


class _CasDm4ContractCtx:
    """On-the-fly contracted CAS dm4 helper (no dm4 materialization).

    This context computes contractions of the *raw* 4-body density
      <E_pq E_rs E_tu E_wv>
    and applies PySCF's `reorder_dm1234` delta corrections at the contraction level,
    returning results consistent with `GUGAFCISolver.make_rdm123(reorder=True)` conventions.

    Notes
    -----
    This is a reference-oriented CPU implementation intended for small active spaces.
    """

    def __init__(self, ic_res: Any, *, ci_cas: np.ndarray, n_act: int) -> None:
        drt = _build_cas_drt_for_ic_res(ic_res, n_act=int(n_act))
        self._drt = drt
        self._norb = int(drt.norb)
        self._ncsf = int(drt.ncsf)

        ci_cas = np.asarray(ci_cas, dtype=np.float64).ravel()
        if int(ci_cas.size) != int(drt.ncsf):
            raise ValueError("ci_cas has wrong length for CAS DRT")
        nrm = float(np.linalg.norm(ci_cas))
        if not np.isfinite(nrm) or nrm <= 0.0:
            raise ValueError("ci_cas must have nonzero finite norm")
        self._ci = np.asarray(ci_cas / nrm, dtype=np.float64)

        from asuka.cuguga.oracle import _get_epq_action_cache  # noqa: PLC0415
        from asuka.rdm.rdm123 import _STEP_TO_OCC_F64  # noqa: PLC0415

        cache = _get_epq_action_cache(drt)
        self._cache = cache
        self._occ = np.asarray(_STEP_TO_OCC_F64[cache.steps], dtype=np.float64, order="C")

        self._mats = None
        self._csc_matmul_dense_inplace_cy = None
        try:  # optional SciPy-backed sparse E_pq matrices
            from asuka.contract import _epq_spmat_list as _epq_spmat_list  # noqa: PLC0415
            from asuka.contract import _sp as _sp  # noqa: PLC0415

            if _sp is not None and _epq_spmat_list is not None:
                self._mats = _epq_spmat_list(drt, cache)
        except Exception:  # pragma: no cover
            self._mats = None

        try:  # optional Cython in-place CSC @ dense kernel
            from asuka._epq_cy import csc_matmul_dense_inplace_cy as _csc_matmul_dense_inplace_cy  # type: ignore[import-not-found]  # noqa: PLC0415

            self._csc_matmul_dense_inplace_cy = _csc_matmul_dense_inplace_cy
        except Exception:  # pragma: no cover
            self._csc_matmul_dense_inplace_cy = None

    def _apply_epq_vec(self, *, p: int, q: int, x: np.ndarray, out: np.ndarray) -> None:
        p = int(p)
        q = int(q)
        if out.shape != x.shape:
            raise ValueError("x/out shape mismatch")
        if p == q:
            np.multiply(self._occ[:, p], x, out=out)
            return

        mats = self._mats
        if mats is not None:
            mat = mats[p * int(self._norb) + q]
            if mat is None:
                raise AssertionError("missing E_pq sparse matrix")
            if self._csc_matmul_dense_inplace_cy is not None:
                out_col = out.reshape(self._ncsf, 1)
                x_col = x.reshape(self._ncsf, 1)
                out_col[:, 0].fill(0.0)
                self._csc_matmul_dense_inplace_cy(  # type: ignore[misc]
                    mat.indptr, mat.indices, mat.data, x_col, out_col
                )
            else:
                out[:] = mat.dot(x)  # type: ignore[operator]
            return

        from asuka.rdm.rdm123 import _fill_epq_vec  # noqa: PLC0415

        _fill_epq_vec(self._drt, self._cache, x, p=p, q=q, out=out)

    def contract_dm4_reordered(
        self,
        *,
        c_bra: np.ndarray,
        c_ket: np.ndarray,
        dm1: np.ndarray,
        dm2: np.ndarray,
        dm3: np.ndarray,
    ) -> np.ndarray:
        """Contract reordered CAS dm4 into a 4-index tensor (i,j,k,l).

        Returns
        -------
        out
            Tensor with shape (norb,norb,norb,norb) corresponding to:
              out[i,j,k,l] = Σ_{r,s,t,u} c_bra[r,s] c_ket[t,u] dm4_reordered[r,t,s,u,i,j,k,l]
        """

        n = int(self._norb)
        if c_bra.shape != (n, n) or c_ket.shape != (n, n):
            raise ValueError("c_bra/c_ket must have shape (n_act,n_act)")
        if dm1.shape != (n, n) or dm2.shape != (n, n, n, n) or dm3.shape != (n, n, n, n, n, n):
            raise ValueError("dm1/dm2/dm3 shapes must match n_act")

        # Compute bra vector u = (Σ_{r,s,t,u} c_bra[r,s] c_ket[t,u] E_us E_tr) |ci>.
        ci = self._ci
        tmp1 = np.empty_like(ci)
        tmp2 = np.empty_like(ci)
        acc = np.zeros_like(ci)

        nz_bra = np.asarray(np.nonzero(c_bra)).T
        nz_ket = np.asarray(np.nonzero(c_ket)).T
        if int(nz_bra.size) == 0 or int(nz_ket.size) == 0:
            return np.zeros((n, n, n, n), dtype=np.float64)

        # Cache E_tr |ci> for all needed (t,r) pairs.
        need_tr: set[tuple[int, int]] = set()
        r_set = set(int(rr) for rr, _ss in nz_bra.tolist())
        t_set = set(int(tt) for tt, _uu in nz_ket.tolist())
        for t in t_set:
            for r in r_set:
                need_tr.add((t, r))

        tr_cache: dict[tuple[int, int], np.ndarray] = {}
        for t, r in need_tr:
            self._apply_epq_vec(p=t, q=r, x=ci, out=tmp1)
            tr_cache[(t, r)] = tmp1.copy()

        for r, s in nz_bra.tolist():
            crs = float(c_bra[int(r), int(s)])
            for t, u in nz_ket.tolist():
                ctu = float(c_ket[int(t), int(u)])
                w = crs * ctu
                if w == 0.0:
                    continue
                v_tr = tr_cache[(int(t), int(r))]
                self._apply_epq_vec(p=int(u), q=int(s), x=v_tr, out=tmp2)
                acc += w * tmp2

        u_vec = np.asarray(acc, dtype=np.float64, order="C")

        # Raw contraction: out_raw[i,j,k,l] = <ci| E_rt E_su E_ij E_kl |ci> contracted over (r,s,t,u).
        # This equals dot(u_vec, E_ij E_kl |ci>).
        out_raw = np.zeros((n, n, n, n), dtype=np.float64)
        v_kl = np.empty_like(ci)
        v_ij = np.empty_like(ci)
        for k in range(n):
            for l in range(n):
                self._apply_epq_vec(p=k, q=l, x=ci, out=v_kl)
                for i in range(n):
                    for j in range(n):
                        self._apply_epq_vec(p=i, q=j, x=v_kl, out=v_ij)
                        out_raw[i, j, k, l] = float(np.dot(u_vec, v_ij))

        # Apply PySCF `reorder_dm1234` delta corrections at the contraction level for the
        # (r,t,s,u,i,j,k,l) index mapping:
        #   dm4_raw[r,t,s,u,i,j,k,l] = <E_rt E_su E_ij E_kl>.
        corr = np.zeros_like(out_raw)

        # (1) axis1==axis6 (t==k): dm4[:,k,:,:,:,:,k,:] -= dm3.transpose(0,2,3,4,5,1)
        for k in range(n):
            row = c_ket[k]
            if not np.any(row):
                continue
            tmp = np.einsum("rs,u,rlsuij->lij", c_bra, row, dm3, optimize=True)
            corr[:, :, k, :] += tmp.transpose(1, 2, 0)

        # (2) axis3==axis6 (u==k): dm4[:,:,:,k,:,:,k,:] -= dm3.transpose(0,1,2,4,5,3)
        for k in range(n):
            col = c_ket[:, k]
            if not np.any(col):
                continue
            tmp = np.einsum("rs,t,rtslij->lij", c_bra, col, dm3, optimize=True)
            corr[:, :, k, :] += tmp.transpose(1, 2, 0)

        # (3) axis5==axis6 (j==k): dm4[:,:,:,:,:,k,k,:] -= dm3
        f_il = np.einsum("rs,tu,rtsuil->il", c_bra, c_ket, dm3, optimize=True)
        for j in range(n):
            corr[:, j, j, :] += f_il

        # (4) axis1==axis4 (t==i): dm4[:,i,:,:,i,:,:,:] -= dm3.transpose(0,2,3,1,4,5)
        corr += np.einsum("rs,iu,rjsukl->ijkl", c_bra, c_ket, dm3, optimize=True)

        # (5) axis3==axis4 (u==i): dm4[:,:,:,i,i,:,:,:] -= dm3
        corr += np.einsum("rs,ti,rtsjkl->ijkl", c_bra, c_ket, dm3, optimize=True)

        # (6) axis1==axis2 (t==s): dm4[:,q,q,:,:,:,:,:] -= dm3
        ab = c_bra @ c_ket  # (r,u) = Σ_t c_bra[r,t] c_ket[t,u]
        corr += np.einsum("ru,ruijkl->ijkl", ab, dm3, optimize=True)

        dm1t = dm1.T

        # Term 7: axis1==axis2 (t==s) and axis3==axis6 (u==k)
        for k in range(n):
            v = c_bra @ c_ket[:, k]
            if not np.any(v):
                continue
            m = np.einsum("r,rlij->lij", v, dm2, optimize=True)  # (l,i,j)
            corr[:, :, k, :] += m.transpose(1, 2, 0)

        # Term 8: axis1==axis2 (t==s) and axis5==axis6 (j==k)
        f8 = np.einsum("ru,ruil->il", ab, dm2, optimize=True)
        for j in range(n):
            corr[:, j, j, :] += f8

        # Term 9: axis1==axis4 (t==i) and axis5==axis6 (j==k)
        for i in range(n):
            if not np.any(c_ket[i]):
                continue
            f9 = np.einsum("rs,u,rlsu->l", c_bra, c_ket[i], dm2[:, :, :, :], optimize=True)  # (l,)
            for j in range(n):
                corr[i, j, j, :] += f9

        # Term 10: axis1==axis4 (t==i) and axis3==axis6 (u==k)
        g10 = np.einsum("rs,rjsl->jl", c_bra, dm2, optimize=True)
        for i in range(n):
            for k in range(n):
                b = float(c_ket[i, k])
                if b != 0.0:
                    corr[i, :, k, :] += b * g10

        # Term 11: axis1==axis6 (t==k) and axis3==axis4 (u==i)
        g11_t = np.einsum("rs,rlsj->lj", c_bra, dm2, optimize=True).T  # (j,l)
        for i in range(n):
            for k in range(n):
                b = float(c_ket[k, i])
                if b != 0.0:
                    corr[i, :, k, :] += b * g11_t

        # Term 12: axis3==axis4 (u==i) and axis5==axis6 (j==k)
        for i in range(n):
            if not np.any(c_ket[:, i]):
                continue
            f12 = np.einsum("rs,t,rtsl->l", c_bra, c_ket[:, i], dm2, optimize=True)
            for j in range(n):
                corr[i, j, j, :] += f12

        # Term 13: axis3==axis4 (u==i) and axis1==axis2 (t==s)
        for i in range(n):
            v = c_bra @ c_ket[:, i]
            if not np.any(v):
                continue
            corr[i] += np.einsum("r,rjkl->jkl", v, dm2, optimize=True)

        # Term 14: axis3==axis4 (u==i), axis1==axis2 (t==s), and axis5==axis6 (j==k)
        for i in range(n):
            v = c_bra @ c_ket[:, i]
            if not np.any(v):
                continue
            f14 = np.einsum("r,rl->l", v, dm1t, optimize=True)
            for j in range(n):
                corr[i, j, j, :] += f14

        out = out_raw - corr
        return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm12(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    contraction: Contraction,
    backend: DensityBackend = "reconstruct",
    rdm_backend: Literal["cuda", "cpu"] = "cuda",
    reconstructed: Tuple[DRT, np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dm1_corr, dm2_corr) for a contracted ic-MRCISD wavefunction.

    Parameters
    ----------
    ic_res : Any
        :class:`~asuka.mrci.ic_mrcisd.ICMRCISDResult` instance.
    ci_cas : np.ndarray
        Reference CAS CI vector in the CAS DRT ordering (same as ``mc.ci[state]``).
    contraction : {"fic", "sc"}
        Contraction scheme ("fic" or "sc").
    backend : {"reconstruct", "direct"}, optional
        Algorithm backend:
          - "reconstruct": expand to an uncontracted CSF CI vector and use the
            existing uncontracted RDM routines.
          - "direct": staged reconstruction backend.
        Default is "reconstruct".
    rdm_backend : {"cuda", "cpu"}, optional
        Backend for RDMs when using reconstruction. Default is "cuda".
    reconstructed : tuple[DRT, np.ndarray] | None, optional
        Pre-computed (drt, ci) tuple from reconstruction, if available.
        Default is None.

    Returns
    -------
    dm1 : np.ndarray
        1-RDM in the correlated MO basis. Shape: (norb, norb).
    dm2 : np.ndarray
        2-RDM in the correlated MO basis. Shape: (norb, norb, norb, norb).
        Convention: ``d2[p,q,r,s] = <E_{pq} E_{rs}> - delta_{qr} <E_{ps}>``.
    """

    contraction_s = str(contraction).strip().lower()
    if contraction_s not in ("fic", "sc"):
        raise ValueError("contraction must be 'fic' or 'sc'")
    backend_s = str(backend).strip().lower()
    if backend_s not in ("reconstruct", "direct"):
        raise ValueError("backend must be 'reconstruct' or 'direct'")

    # Phase 2/3 reconstruction backends.
    from asuka.mrci.ic_reconstruct import (  # noqa: PLC0415
        reconstruct_uncontracted_ci_from_ic_mrcisd,
        reconstruct_uncontracted_ci_from_ic_mrcisd_staged,
    )
    from asuka.mrci.rdm_mrcisd import make_rdm12_mrcisd, prepare_mrcisd_rdm_workspace  # noqa: PLC0415

    # Sanity-check contraction type against label classes.
    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    if contraction_s == "fic":
        if not isinstance(singles, ICSingles) or not isinstance(doubles, ICDoubles):
            raise TypeError("contraction='fic' expects ICSingles/ICDoubles labels")
    else:
        if not isinstance(singles, SCSingles) or not isinstance(doubles, SCDoubles):
            raise TypeError("contraction='sc' expects SCSingles/SCDoubles labels")

    if reconstructed is None:
        if backend_s == "direct":
            drt_mrci, ci_mrci = reconstruct_uncontracted_ci_from_ic_mrcisd_staged(
                ic_res, ci_cas=np.asarray(ci_cas, dtype=np.float64)
            )
        else:
            drt_mrci, ci_mrci = reconstruct_uncontracted_ci_from_ic_mrcisd(
                ic_res, ci_cas=np.asarray(ci_cas, dtype=np.float64)
            )
    else:
        drt_mrci, ci_mrci = reconstructed
        if not isinstance(drt_mrci, DRT):
            raise TypeError("reconstructed[0] must be a DRT")
        ci_mrci = np.asarray(ci_mrci, dtype=np.float64).ravel()
        if ci_mrci.size != int(drt_mrci.ncsf):
            raise ValueError("reconstructed[1] has wrong length for reconstructed DRT")

    spaces = getattr(ic_res, "spaces")
    n_act = int(getattr(spaces, "n_internal"))
    n_virt = int(drt_mrci.norb) - int(n_act)
    if n_virt < 0:
        raise RuntimeError("invalid reconstructed DRT: drt_mrci.norb < n_act")

    diag = getattr(ic_res, "diagnostics", {}) or {}
    max_virt_e = int(float(diag.get("max_virt_e", 2.0)))
    if max_virt_e < 0:
        raise ValueError("invalid max_virt_e in ic_res.diagnostics")

    rdm_ws = prepare_mrcisd_rdm_workspace(
        drt_mrci,
        n_act=n_act,
        n_virt=n_virt,
        nelec=int(drt_mrci.nelec),
        twos=int(drt_mrci.twos_target),
        max_virt_e=max_virt_e,
    )
    dm1_corr, dm2_corr = make_rdm12_mrcisd(rdm_ws, ci_mrci, rdm_backend=rdm_backend)

    return np.asarray(dm1_corr), np.asarray(dm2_corr)


def _infer_n_act_n_virt(ic_res: Any) -> tuple[int, int]:
    spaces = getattr(ic_res, "spaces", None)
    if spaces is None:
        raise TypeError("ic_res missing OrbitalSpaces")
    n_act = int(getattr(spaces, "n_internal"))
    n_virt = int(getattr(spaces, "n_external"))
    if n_act < 0 or n_virt < 0:
        raise ValueError("invalid orbital spaces (negative sizes)")
    return n_act, n_virt


def ic_mrcisd_make_rdm1_internal_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
) -> np.ndarray:
    """Return the internal-internal block of correlated-space dm1 for ic-MRCISD.

    Parameters
    ----------
    ic_res : Any
        ICMRCISDResult object.
    ci_cas : np.ndarray
        Reference CAS CI vector.
    backend : {"direct", "reconstruct"}, optional
        Backend to use. If not "direct", falls back to reconstruction.
    rdm_backend : {"cuda", "cpu"}, optional
        RDM backend for fallback.
    dm1_int : np.ndarray | None, optional
        Pre-computed internal 1-RDM.
    dm2_int : np.ndarray | None, optional
        Pre-computed internal 2-RDM.
    dm3_int : np.ndarray | None, optional
        Pre-computed internal 3-RDM.

    Returns
    -------
    dm1_ii : np.ndarray
        Internal-internal 1-RDM block. Shape: (n_act, n_act).
    """

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        contraction_s = "fic" if is_fic else "sc"
        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        dm1, _dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm1[:n_act, :n_act], dtype=np.float64)

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None or dm3_int is None:
        dm1_int, dm2_int, dm3_int = _cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm1_int = np.asarray(dm1_int, dtype=np.float64)
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)

    out = (float(c0) * float(c0)) * np.asarray(dm1_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        # Singles–singles: Σ_a c_a^2 * Σ_{r,s} dm2_int[r,s,j,i].
        w_s = float(np.dot(c_s, c_s))
        if w_s != 0.0:
            ss_mat = np.einsum("rsji->ij", dm2_int, optimize=True)
            out = out + w_s * ss_mat

        if int(n_doubles) == 0 or not np.any(c_d):
            return np.asarray(out, dtype=np.float64, order="C")

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        w_off = 0.0
        w_diag = 0.0
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")
            if a == b:
                w_diag += cd * cd
            else:
                w_off += cd * cd

        if w_off != 0.0:
            c_off = np.ones((int(n_act), int(n_act)), dtype=np.float64)
            if not allow_same_internal:
                np.fill_diagonal(c_off, 0.0)
            m_off = np.einsum("rs,tu,rtsuji->ij", c_off, c_off, dm3_int, optimize=True)
            out = out + w_off * m_off

        if w_diag != 0.0:
            c_diag = np.triu(np.ones((int(n_act), int(n_act)), dtype=np.float64), k=0 if allow_same_internal else 1)
            m1 = np.einsum("rs,tu,rtsuji->ij", c_diag, c_diag, dm3_int, optimize=True)
            m2 = np.einsum("rs,tu,struji->ij", c_diag, c_diag, dm3_int, optimize=True)
            out = out + w_diag * (m1 + m2)

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    if np.any(cs):
        out = out + np.einsum("ar,as,rsji->ij", cs, cs, dm2_int, optimize=True)

    if int(n_doubles) == 0 or not np.any(c_d):
        return np.asarray(out, dtype=np.float64, order="C")

    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        idx = order[start:stop].astype(np.int64, copy=False)
        cd = c_d[idx]
        if not np.any(cd):
            continue

        r_idx = r_all[idx]
        s_idx = s_all[idx]

        cmat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
        np.add.at(cmat, (r_idx, s_idx), cd)

        a_glob = int(keys[g, 0])
        b_glob = int(keys[g, 1])
        if a_glob == b_glob:
            m1 = np.einsum("rs,tu,rtsuji->ij", cmat, cmat, dm3_int, optimize=True)
            m2 = np.einsum("rs,tu,struji->ij", cmat, cmat, dm3_int, optimize=True)
            out = out + (m1 + m2)
        else:
            out = out + np.einsum("rs,tu,rtsuji->ij", cmat, cmat, dm3_int, optimize=True)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm1_ext_int_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
) -> np.ndarray:
    """Return the external-internal block dm1[ext, int] for ic-MRCISD.

    Parameters
    ----------
    ic_res : Any
        ICMRCISDResult object.
    ci_cas : np.ndarray
        Reference CAS CI vector.
    backend : {"direct", "reconstruct"}, optional
        Backend to use.
    rdm_backend : {"cuda", "cpu"}, optional
        RDM backend for fallback.
    dm1_int : np.ndarray | None, optional
        Pre-computed internal 1-RDM.
    dm2_int : np.ndarray | None, optional
        Pre-computed internal 2-RDM.
    dm3_int : np.ndarray | None, optional
        Pre-computed internal 3-RDM.

    Returns
    -------
    dm1_ei : np.ndarray
        External-internal 1-RDM block. Shape: (n_virt, n_act).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        dm1, _dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm1[n_act:, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None:
        dm1_int, dm2_int, _dm3_int = _cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm1_int = np.asarray(dm1_int, dtype=np.float64)
        dm2_int = np.asarray(dm2_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Reference–singles: c0 * c_a * Σ_r <E_{i r}>.
        g_int = dm1_int.sum(axis=0)
        out = float(c0) * cs[:, None] * g_int[None, :]

        # Singles–doubles: c_{ab} c_b * Σ_{r,s,t} dm2_int[i,r,t,s], with r!=s excluded if requested.
        t_full = dm2_int.sum(axis=(1, 2, 3))
        t_diag = np.einsum("irtr->i", dm2_int, optimize=True)
        if allow_same_internal:
            t_diff = t_full
            t_same = t_full + t_diag
        else:
            t_diff = t_full - t_diag
            t_same = t_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles labels out of range for contiguous orbital convention")

            if a == b:
                if cs[a] != 0.0:
                    out[a] += cd * cs[a] * t_same
            else:
                if cs[b] != 0.0:
                    out[a] += cd * cs[b] * t_diff
                if cs[a] != 0.0:
                    out[b] += cd * cs[a] * t_diff

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Reference–singles: c0 * Σ_r c_{a r} <E_{i r}>.
    out = float(c0) * (cs @ dm1_int)

    # Singles–doubles: Σ_{(a,b;r,s),t} c_{b t} c_{ab;rs} <E_{i r} E_{t s} - δ_{r t} E_{i s}>.
    # Under the external-vacuum assumption this reduces to dm2_int[i,r,t,s] (see derivation in Phase-3 notes).
    da = np.asarray(doubles.a, dtype=np.int64).ravel()
    db = np.asarray(doubles.b, dtype=np.int64).ravel()
    dr = np.asarray(doubles.r, dtype=np.int64).ravel()
    ds = np.asarray(doubles.s, dtype=np.int64).ravel()
    for idx in range(n_doubles):
        cd = float(c_d[idx])
        if cd == 0.0:
            continue
        a = int(da[idx]) - int(n_act)
        b = int(db[idx]) - int(n_act)
        r = int(dr[idx])
        s = int(ds[idx])
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt) and 0 <= r < int(n_act) and 0 <= s < int(n_act)):
            raise ValueError("doubles labels out of range for contiguous orbital convention")

        if a == b:
            cs_a = cs[a]
            if np.any(cs_a):
                out[a] += cd * np.einsum("t,it->i", cs_a, dm2_int[:, r, :, s] + dm2_int[:, s, :, r], optimize=True)
        else:
            cs_b = cs[b]
            if np.any(cs_b):
                out[a] += cd * np.einsum("t,it->i", cs_b, dm2_int[:, r, :, s], optimize=True)

            cs_a = cs[a]
            if np.any(cs_a):
                out[b] += cd * np.einsum("t,it->i", cs_a, dm2_int[:, s, :, r], optimize=True)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm1_ext_ext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
) -> np.ndarray:
    """Return the external-external block dm1[ext, ext] for ic-MRCISD.

    Parameters
    ----------
    ic_res : Any
        ICMRCISDResult object.
    ci_cas : np.ndarray
        Reference CAS CI vector.
    backend : {"direct", "reconstruct"}, optional
        Backend to use.
    rdm_backend : {"cuda", "cpu"}, optional
        RDM backend for fallback.
    dm1_int : np.ndarray | None, optional
        Pre-computed internal 1-RDM.
    dm2_int : np.ndarray | None, optional
        Pre-computed internal 2-RDM.
    dm3_int : np.ndarray | None, optional
        Pre-computed internal 3-RDM.

    Returns
    -------
    dm1_ee : np.ndarray
        External-external 1-RDM block. Shape: (n_virt, n_virt).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        dm1, _dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm1[n_act:, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None:
        dm1_int, dm2_int, _dm3_int = _cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm1_int = np.asarray(dm1_int, dtype=np.float64)
        dm2_int = np.asarray(dm2_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Singles–singles: scalar internal contraction times outer product of SC singles coefficients.
        s1 = float(np.sum(dm1_int))
        out = (cs[:, None] * cs[None, :]) * s1

        c_d = np.asarray(c_d, dtype=np.float64)
        if not np.any(c_d):
            return np.asarray(out, dtype=np.float64, order="C")

        # SC doubles–doubles contributions require computing <ab|E_{b' a'}|cd> between
        # *strongly contracted* doubles basis vectors; this is not a pure "pair relabeling"
        # because the diagonal (a==b) SC doubles use an (r<=s) internal sum.
        #
        # Precompute the few internal contraction scalars needed to evaluate these
        # matrix elements in the SC doubles subspace.
        if allow_same_internal:
            norm_off = float(np.sum(dm2_int))
        else:
            tot = float(np.sum(dm2_int))
            diag_rs = float(np.einsum("rtru->", dm2_int, optimize=True))
            diag_tu = float(np.einsum("rtst->", dm2_int, optimize=True))
            diag_both = float(np.einsum("rtrt->", dm2_int, optimize=True))
            norm_off = tot - diag_rs - diag_tu + diag_both

        norm_diag = 0.0
        for r in range(int(n_act)):
            start_s = r if allow_same_internal else r + 1
            for s in range(start_s, int(n_act)):
                for t in range(int(n_act)):
                    start_u = t if allow_same_internal else t + 1
                    for u in range(start_u, int(n_act)):
                        norm_diag += float(dm2_int[r, t, s, u] + dm2_int[s, t, r, u])

        # offdiag -> diag (e.g. <bb|E_{b a}|ab>) and diag -> offdiag (e.g. <ab|E_{b a}|aa>)
        m_off_to_diag = 0.0
        for r in range(int(n_act)):
            for s in range(int(n_act)):
                if (not allow_same_internal) and r == s:
                    continue
                for t in range(int(n_act)):
                    start_u = t if allow_same_internal else t + 1
                    for u in range(start_u, int(n_act)):
                        m_off_to_diag += float(dm2_int[t, r, u, s] + dm2_int[u, r, t, s])

        m_diag_to_off = 0.0
        for r in range(int(n_act)):
            start_s = r if allow_same_internal else r + 1
            for s in range(start_s, int(n_act)):
                for t in range(int(n_act)):
                    for u in range(int(n_act)):
                        if (not allow_same_internal) and t == u:
                            continue
                        m_diag_to_off += float(dm2_int[t, r, u, s] + dm2_int[t, s, u, r])

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()

        label_to_idx: dict[tuple[int, int], int] = {}
        for idx in range(n_doubles):
            key = (int(da[idx]), int(db[idx]))
            label_to_idx[key] = int(idx)

        for a_rel in range(int(n_virt)):
            a_glob = int(n_act) + int(a_rel)
            for b_rel in range(int(n_virt)):
                b_glob = int(n_act) + int(b_rel)

                dd = 0.0
                for q in range(n_doubles):
                    cq = float(c_d[q])
                    if cq == 0.0:
                        continue
                    qa = int(da[q])
                    qb = int(db[q])

                    if a_glob != qa and a_glob != qb:
                        continue

                    # Diagonal source pair (a_glob,a_glob).
                    if qa == qb:
                        if b_glob == a_glob:
                            dd += cq * cq * 2.0 * float(norm_diag)
                            continue
                        na = min(int(a_glob), int(b_glob))
                        nb = max(int(a_glob), int(b_glob))
                        p = label_to_idx.get((na, nb))
                        if p is None:
                            continue
                        cp = float(c_d[p])
                        dd += cp * cq * float(m_diag_to_off)
                        continue

                    # Off-diagonal source pair with exactly one occurrence of a_glob.
                    other = qb if qa == a_glob else qa
                    if b_glob == other:
                        p = label_to_idx.get((int(b_glob), int(b_glob)))
                        if p is None:
                            continue
                        cp = float(c_d[p])
                        dd += cp * cq * float(m_off_to_diag)
                    else:
                        na = min(int(b_glob), int(other))
                        nb = max(int(b_glob), int(other))
                        p = label_to_idx.get((na, nb))
                        if p is None:
                            continue
                        cp = float(c_d[p])
                        dd += cp * cq * float(norm_off)

                out[a_rel, b_rel] += float(dd)

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Singles–singles: out[a,b] = Σ_{r,s} c_{a s} c_{b r} <E_{r s}>
    out = cs @ dm1_int @ cs.T

    # Doubles–doubles: interpret E_{b a} as mapping doubles (source external a) to doubles (target external b),
    # then take the overlap with the doubles sector.
    from asuka.mrci.ic_overlap import apply_overlap_doubles  # noqa: PLC0415

    da = np.asarray(doubles.a, dtype=np.int64).ravel()
    db = np.asarray(doubles.b, dtype=np.int64).ravel()
    dr = np.asarray(doubles.r, dtype=np.int64).ravel()
    ds = np.asarray(doubles.s, dtype=np.int64).ravel()

    label_to_idx: dict[tuple[int, int, int, int], int] = {}
    for idx in range(n_doubles):
        key = (int(da[idx]), int(db[idx]), int(dr[idx]), int(ds[idx]))
        label_to_idx[key] = int(idx)

    c_d = np.asarray(c_d, dtype=np.float64)
    if not np.any(c_d):
        return np.asarray(out, dtype=np.float64, order="C")

    for a_rel in range(int(n_virt)):
        a_glob = int(n_act) + int(a_rel)
        for b_rel in range(int(n_virt)):
            b_glob = int(n_act) + int(b_rel)

            c_map = np.zeros_like(c_d)
            for k in range(n_doubles):
                w = float(c_d[k])
                if w == 0.0:
                    continue
                aa = int(da[k])
                bb = int(db[k])
                rr = int(dr[k])
                ss = int(ds[k])

                # Replace occurrences of the source external index `a_glob` by `b_glob`.
                if aa == a_glob:
                    na, nb, nr, ns = b_glob, bb, rr, ss
                    if (na > nb) or (na == nb and nr > ns):
                        na, nb, nr, ns = nb, na, ns, nr
                    idx2 = label_to_idx.get((na, nb, nr, ns))
                    if idx2 is not None:
                        c_map[idx2] += w

                if bb == a_glob:
                    na, nb, nr, ns = aa, b_glob, rr, ss
                    if (na > nb) or (na == nb and nr > ns):
                        na, nb, nr, ns = nb, na, ns, nr
                    idx2 = label_to_idx.get((na, nb, nr, ns))
                    if idx2 is not None:
                        c_map[idx2] += w

            if np.any(c_map):
                rho_map = apply_overlap_doubles(c_doubles=c_map, doubles=doubles, dm2=dm2_int)
                out[a_rel, b_rel] += float(np.dot(c_d, rho_map))

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm1_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
) -> np.ndarray:
    """Return the full correlated-space dm1 for ic-MRCISD (Phase-3 API).

    Notes
    -----
    For `backend="direct"`, assemble dm1 from the Phase-3 blocks:
      - internal–internal
      - external–internal
      - external–external

    For `backend!="direct"`, fall back to the reconstruction backend.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

        dm1, _dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm1, dtype=np.float64)

    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    dm1_int = ic_mrcisd_make_rdm1_internal_phase3(ic_res, ci_cas=ci_cas, backend="direct", rdm_backend=rdm_backend)
    dm1_ext_int = ic_mrcisd_make_rdm1_ext_int_phase3(ic_res, ci_cas=ci_cas, backend="direct", rdm_backend=rdm_backend)
    dm1_ext_ext = ic_mrcisd_make_rdm1_ext_ext_phase3(ic_res, ci_cas=ci_cas, backend="direct", rdm_backend=rdm_backend)

    out = np.zeros((int(n_act + n_virt), int(n_act + n_virt)), dtype=np.float64)
    out[:n_act, :n_act] = dm1_int
    out[n_act:, :n_act] = dm1_ext_int
    out[:n_act, n_act:] = dm1_ext_int.T
    out[n_act:, n_act:] = dm1_ext_ext
    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm12_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return (dm1_corr, dm2_corr) for ic-MRCISD in correlated-space MO ordering (Phase-3 API).

    For `backend="direct"`, assemble full correlated-space dm1/dm2 from the Phase-3
    direct density blocks (no CSF-space reconstruction). When `backend!="direct"`,
    fall back to the reconstruction backend via :func:`ic_mrcisd_make_rdm12`.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1/dm2 assembly")

        dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm1, dtype=np.float64), np.asarray(dm2, dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1/dm2 assembly")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    dm1_int, dm2_int, dm3_int = _cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    dm4_ctx = None
    if int(n_doubles) != 0 and np.any(c_d):
        dm4_ctx = _CasDm4ContractCtx(ic_res, ci_cas=ci_cas, n_act=n_act)

    # dm1 assembly (3 blocks).
    dm1_ii = ic_mrcisd_make_rdm1_internal_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
    )
    dm1_vi = ic_mrcisd_make_rdm1_ext_int_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
    )
    dm1_vv = ic_mrcisd_make_rdm1_ext_ext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
    )

    n_cor = int(n_act + n_virt)
    dm1 = np.zeros((n_cor, n_cor), dtype=np.float64)
    dm1[:n_act, :n_act] = dm1_ii
    dm1[n_act:, :n_act] = dm1_vi
    dm1[:n_act, n_act:] = dm1_vi.T
    dm1[n_act:, n_act:] = dm1_vv

    # dm2 assembly (16 blocks).
    dm2 = np.zeros((n_cor, n_cor, n_cor, n_cor), dtype=np.float64)

    dm2[:n_act, :n_act, :n_act, :n_act] = ic_mrcisd_make_rdm2_intint_intint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[n_act:, :n_act, n_act:, :n_act] = ic_mrcisd_make_rdm2_extint_extint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[:n_act, n_act:, :n_act, n_act:] = ic_mrcisd_make_rdm2_intext_intext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )

    dm2[n_act:, :n_act, :n_act, :n_act] = ic_mrcisd_make_rdm2_extint_intint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[:n_act, n_act:, :n_act, :n_act] = ic_mrcisd_make_rdm2_intext_intint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[:n_act, :n_act, n_act:, :n_act] = ic_mrcisd_make_rdm2_intint_extint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[:n_act, :n_act, :n_act, n_act:] = ic_mrcisd_make_rdm2_intint_intext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )

    blk_viiv = ic_mrcisd_make_rdm2_extint_intext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[n_act:, :n_act, :n_act, n_act:] = blk_viiv
    dm2[:n_act, n_act:, n_act:, :n_act] = blk_viiv.transpose(1, 0, 3, 2)  # IVVI

    blk_vvii = ic_mrcisd_make_rdm2_extext_intint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[n_act:, n_act:, :n_act, :n_act] = blk_vvii
    dm2[:n_act, :n_act, n_act:, n_act:] = blk_vvii.transpose(2, 3, 0, 1)  # IIVV

    dm2[n_act:, :n_act, n_act:, n_act:] = ic_mrcisd_make_rdm2_extint_extext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[:n_act, n_act:, n_act:, n_act:] = ic_mrcisd_make_rdm2_intext_extext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[n_act:, n_act:, :n_act, n_act:] = ic_mrcisd_make_rdm2_extext_intext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[n_act:, n_act:, n_act:, :n_act] = ic_mrcisd_make_rdm2_extext_extint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    dm2[n_act:, n_act:, n_act:, n_act:] = ic_mrcisd_make_rdm2_extext_extext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )

    return np.asarray(dm1, dtype=np.float64, order="C"), np.asarray(dm2, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
) -> np.ndarray:
    """Return the full correlated-space dm2 for ic-MRCISD (Phase-3 API)."""

    _dm1, dm2 = ic_mrcisd_make_rdm12_phase3(ic_res, ci_cas=ci_cas, backend=backend, rdm_backend=rdm_backend)
    return np.asarray(dm2, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extint_extint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,int,ext,int] for ic-MRCISD (ΔN_ext=+2 block).

    This corresponds to the correlated-space block:
      dm2[a,i,b,j]  (a,b external; i,j internal)
    in cuGUGA's delta-corrected spin-free convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s} - δ_{q r} E_{p s}>.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, :n_act, n_act:, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)

    out = np.zeros((int(n_virt), int(n_act), int(n_virt), int(n_act)), dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        # Ordered internal pair sum: Σ_{r,s} dm2_int[r,i,s,j]
        blk_full = dm2_int.sum(axis=(0, 2))
        blk_diag = np.einsum("rirj->ij", dm2_int, optimize=True)
        if allow_same_internal:
            blk_diff = blk_full
            blk_same = blk_full + blk_diag  # diagonal counted twice for r<=s symmetric sum
        else:
            blk_diff = blk_full - blk_diag  # exclude r==s
            blk_same = blk_diff

        a_all = np.asarray(doubles.a, dtype=np.int64).ravel()
        b_all = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(a_all[idx]) - int(n_act)
            b = int(b_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")

            blk = blk_same if a == b else blk_diff
            out[a, :, b, :] += float(c0) * cd * blk
            if a != b:
                out[b, :, a, :] += float(c0) * cd * blk.T

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path: doubles labels carry explicit (r,s) internal indices.
    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        a_glob = int(keys[g, 0])
        b_glob = int(keys[g, 1])
        a = a_glob - int(n_act)
        b = b_glob - int(n_act)
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
            raise ValueError("doubles external labels out of range for contiguous orbital convention")

        idx = order[start:stop].astype(np.int64, copy=False)
        r_idx = r_all[idx]
        s_idx = s_all[idx]
        cd = c_d[idx]

        if a_glob == b_glob:
            # For a==b, doubles labels are canonical in the internal pair (r<=s); the overlap
            # includes both internal pairings.
            blk = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
            for k in range(int(idx.size)):
                w = float(cd[k])
                if w == 0.0:
                    continue
                r = int(r_idx[k])
                s = int(s_idx[k])
                blk += w * (dm2_int[r, :, s, :] + dm2_int[s, :, r, :])
        else:
            cmat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
            np.add.at(cmat, (r_idx, s_idx), cd)
            blk = np.einsum("rs,risj->ij", cmat, dm2_int, optimize=True)

        out[a, :, b, :] += float(c0) * blk
        if a != b:
            out[b, :, a, :] += float(c0) * blk.T

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intext_intext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,ext,int,ext] for ic-MRCISD (ΔN_ext=-2 block).

    This corresponds to the correlated-space block:
      dm2[i,a,j,b]  (a,b external; i,j internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, n_act:, :n_act, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)

    out = np.zeros((int(n_act), int(n_virt), int(n_act), int(n_virt)), dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        # Ordered internal pair sum: Σ_{r,s} dm2_int[i,r,j,s]
        blk_full = dm2_int.sum(axis=(1, 3))
        blk_diag = np.einsum("irjr->ij", dm2_int, optimize=True)
        if allow_same_internal:
            blk_diff = blk_full
            blk_same = blk_full + blk_diag
        else:
            blk_diff = blk_full - blk_diag
            blk_same = blk_diff

        a_all = np.asarray(doubles.a, dtype=np.int64).ravel()
        b_all = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(a_all[idx]) - int(n_act)
            b = int(b_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")

            blk = blk_same if a == b else blk_diff
            out[:, a, :, b] += float(c0) * cd * blk
            if a != b:
                out[:, b, :, a] += float(c0) * cd * blk.T

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        a_glob = int(keys[g, 0])
        b_glob = int(keys[g, 1])
        a = a_glob - int(n_act)
        b = b_glob - int(n_act)
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
            raise ValueError("doubles external labels out of range for contiguous orbital convention")

        idx = order[start:stop].astype(np.int64, copy=False)
        r_idx = r_all[idx]
        s_idx = s_all[idx]
        cd = c_d[idx]

        if a_glob == b_glob:
            # For a==b, doubles labels are canonical in the internal pair (r<=s); the overlap
            # includes both internal pairings (transpose in the free internal indices).
            blk = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
            for k in range(int(idx.size)):
                w = float(cd[k])
                if w == 0.0:
                    continue
                r = int(r_idx[k])
                s = int(s_idx[k])
                m = dm2_int[:, r, :, s]
                blk += w * (m + m.T)
        else:
            cmat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
            np.add.at(cmat, (r_idx, s_idx), cd)
            blk = np.einsum("rs,irjs->ij", cmat, dm2_int, optimize=True)

        out[:, a, :, b] += float(c0) * blk
        if a != b:
            out[:, b, :, a] += float(c0) * blk.T

    return np.asarray(out, dtype=np.float64, order="C")


def _require_internal_external_contiguous(spaces: Any, *, n_act: int, n_virt: int) -> None:
    internal = np.asarray(getattr(spaces, "internal"), dtype=np.int32).ravel()
    external = np.asarray(getattr(spaces, "external"), dtype=np.int32).ravel()
    if internal.size != int(n_act) or external.size != int(n_virt):
        raise ValueError("orbital spaces do not match n_act/n_virt")

    want_internal = np.arange(int(n_act), dtype=np.int32)
    want_external = np.arange(int(n_act), int(n_act) + int(n_virt), dtype=np.int32)
    if not bool(np.all(internal == want_internal)) or not bool(np.all(external == want_external)):
        raise NotImplementedError(
            "Phase-3 dm2 blocks currently require contiguous correlated ordering: internal=0..n_act-1, external=n_act.."
        )


def _cas_dm23_for_ic_res(ic_res: Any, *, ci_cas: np.ndarray, n_act: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (dm2_int, dm3_int) for the reference CAS wavefunction.

    Conventions match `GUGAFCISolver.make_rdm123(reorder=True)`, which is consistent with
    cuGUGA's spin-free generator-form algebra (`E_pq`).
    """

    _dm1, dm2, dm3 = _cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    return dm2, dm3


def _cas_dm123_for_ic_res(
    ic_res: Any, *, ci_cas: np.ndarray, n_act: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dm1_int, dm2_int, dm3_int) for the reference CAS wavefunction.

    Conventions match `GUGAFCISolver.make_rdm123(reorder=True)`:
      - dm1_int[p,q] = <E_{q p}>
      - dm2_int uses the delta-corrected / reordered convention
      - dm3_int matches `asuka.rdm.rdm123._make_rdm123_pyscf(reorder=True)`
    """

    drt_work = getattr(ic_res, "drt_work", None)
    if drt_work is None:
        raise NotImplementedError("Phase-3 dm2 blocks require ic_res.drt_work (semi-direct backend)")
    if not isinstance(drt_work, DRT):
        raise TypeError("ic_res.drt_work must be a DRT instance")

    nelec = int(getattr(drt_work, "nelec"))
    twos = int(getattr(drt_work, "twos_target"))

    ci_cas = np.asarray(ci_cas, dtype=np.float64).ravel()
    nrm = float(np.linalg.norm(ci_cas))
    if not np.isfinite(nrm) or nrm <= 0.0:
        raise ValueError("ci_cas must have nonzero finite norm")
    ci_cas = np.asarray(ci_cas / nrm, dtype=np.float64)

    from asuka import GUGAFCISolver  # noqa: PLC0415

    cas = GUGAFCISolver(twos=twos)
    dm1, dm2, dm3 = cas.make_rdm123(ci_cas, norb=int(n_act), nelec=nelec, reorder=True)
    dm1 = np.asarray(dm1, dtype=np.float64, order="C")
    dm2 = np.asarray(dm2, dtype=np.float64, order="C")
    dm3 = np.asarray(dm3, dtype=np.float64, order="C")
    return dm1, dm2, dm3


def ic_mrcisd_make_rdm2_intint_intint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,int,int,int] for ic-MRCISD (ΔN_ext=0 internal block).

    This corresponds to the correlated-space block:
      dm2[i,j,k,l]  (all indices internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, :n_act, :n_act, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None or dm3_int is None:
        dm1_int, dm2_int, dm3_int = _cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm1_int = np.asarray(dm1_int, dtype=np.float64)
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)
    out = (float(c0) * float(c0)) * np.asarray(dm2_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        # Singles–singles: Σ_a c_a^2 * Σ_{r,s} dm3_int[r,s,i,j,k,l].
        w_s = float(np.dot(c_s, c_s))
        if w_s != 0.0:
            out = out + w_s * dm3_int.sum(axis=(0, 1))

        if int(n_doubles) == 0 or not np.any(c_d):
            return np.asarray(out, dtype=np.float64, order="C")

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        w_off = 0.0
        w_diag = 0.0
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")
            if a == b:
                w_diag += cd * cd
            else:
                w_off += cd * cd

        if w_off == 0.0 and w_diag == 0.0:
            return np.asarray(out, dtype=np.float64, order="C")

        ctx = dm4_ctx if dm4_ctx is not None else _CasDm4ContractCtx(ic_res, ci_cas=ci_cas, n_act=n_act)

        if w_off != 0.0:
            c_off = np.ones((int(n_act), int(n_act)), dtype=np.float64)
            if not allow_same_internal:
                np.fill_diagonal(c_off, 0.0)
            out = out + w_off * ctx.contract_dm4_reordered(
                c_bra=c_off, c_ket=c_off, dm1=dm1_int, dm2=dm2_int, dm3=dm3_int
            )

        if w_diag != 0.0:
            c_diag = np.triu(np.ones((int(n_act), int(n_act)), dtype=np.float64), k=0 if allow_same_internal else 1)
            dd1 = ctx.contract_dm4_reordered(c_bra=c_diag, c_ket=c_diag, dm1=dm1_int, dm2=dm2_int, dm3=dm3_int)
            dd2 = ctx.contract_dm4_reordered(c_bra=c_diag.T, c_ket=c_diag, dm1=dm1_int, dm2=dm2_int, dm3=dm3_int)
            out = out + w_diag * (dd1 + dd2)

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    if np.any(cs):
        out = out + np.einsum("ar,as,rsijkl->ijkl", cs, cs, dm3_int, optimize=True)

    if int(n_doubles) == 0 or not np.any(c_d):
        return np.asarray(out, dtype=np.float64, order="C")

    ctx = dm4_ctx if dm4_ctx is not None else _CasDm4ContractCtx(ic_res, ci_cas=ci_cas, n_act=n_act)

    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        idx = order[start:stop].astype(np.int64, copy=False)
        cd = c_d[idx]
        if not np.any(cd):
            continue

        r_idx = r_all[idx]
        s_idx = s_all[idx]

        cmat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
        np.add.at(cmat, (r_idx, s_idx), cd)

        a_glob = int(keys[g, 0])
        b_glob = int(keys[g, 1])
        if a_glob == b_glob:
            dd1 = ctx.contract_dm4_reordered(c_bra=cmat, c_ket=cmat, dm1=dm1_int, dm2=dm2_int, dm3=dm3_int)
            dd2 = ctx.contract_dm4_reordered(c_bra=cmat.T, c_ket=cmat, dm1=dm1_int, dm2=dm2_int, dm3=dm3_int)
            out = out + (dd1 + dd2)
        else:
            out = out + ctx.contract_dm4_reordered(c_bra=cmat, c_ket=cmat, dm1=dm1_int, dm2=dm2_int, dm3=dm3_int)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extint_intext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,int,int,ext] for ic-MRCISD (ΔN_ext=0 block).

    This corresponds to the correlated-space block:
      dm2[a,i,j,b]  (a,b external; i,j internal)
    in cuGUGA's delta-corrected spin-free convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s} - δ_{q r} E_{p s}>.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, :n_act, :n_act, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm2_int is None or dm3_int is None:
        dm2_int, dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Singles–singles: c_a c_b * Σ_{r,s} dm2_int[r,i,j,s]
        blk_ss = dm2_int.sum(axis=(0, 3))
        out = blk_ss[None, :, :, None] * cs[:, None, None, None] * cs[None, None, None, :]

        if int(n_doubles) == 0 or not np.any(c_d):
            return np.asarray(out, dtype=np.float64, order="C")

        base_off = np.ones((int(n_act), int(n_act)), dtype=np.float64)
        if not allow_same_internal:
            np.fill_diagonal(base_off, 0.0)
        base_diag = np.triu(
            np.ones((int(n_act), int(n_act)), dtype=np.float64), k=0 if allow_same_internal else 1
        )
        base_diag = base_diag + base_diag.T

        cd_map: dict[tuple[int, int], float] = {}
        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            w = float(c_d[idx])
            if w == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")
            cd_map[(a, b)] = w

        i_all = np.arange(int(n_act), dtype=np.int64)
        for x in range(int(n_virt)):
            partners: list[int] = []
            mats: list[np.ndarray] = []
            for p in range(int(n_virt)):
                key = (p, x) if p <= x else (x, p)
                w = float(cd_map.get(key, 0.0))
                if w == 0.0:
                    continue
                mat = base_diag if p == x else base_off
                partners.append(p)
                mats.append(w * mat)

            if not partners:
                continue
            partners_arr = np.asarray(partners, dtype=np.int64)
            c_stack = np.stack(mats, axis=0)
            dd_x = np.einsum("prs,qtu,rijtsu->pijq", c_stack, c_stack, dm3_int, optimize=True)
            out[np.ix_(partners_arr, i_all, i_all, partners_arr)] += dd_x

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Singles–singles contribution.
    out = np.einsum("ar,bs,rijs->aijb", cs, cs, dm2_int, optimize=True)

    if int(n_doubles) == 0 or not np.any(c_d):
        return np.asarray(out, dtype=np.float64, order="C")

    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    pair_to_mat: dict[tuple[int, int], np.ndarray] = {}
    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        idx = order[start:stop].astype(np.int64, copy=False)
        cd = c_d[idx]
        if not np.any(cd):
            continue

        a_glob = int(keys[g, 0]) - int(n_act)
        b_glob = int(keys[g, 1]) - int(n_act)
        if not (0 <= a_glob < int(n_virt) and 0 <= b_glob < int(n_virt)):
            raise ValueError("doubles external labels out of range for contiguous orbital convention")

        r_idx = r_all[idx]
        s_idx = s_all[idx]
        mat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
        np.add.at(mat, (r_idx, s_idx), cd)
        pair_to_mat[(a_glob, b_glob)] = mat

    def _get_pair_mat(p: int, q: int) -> np.ndarray | None:
        p = int(p)
        q = int(q)
        if p < q:
            return pair_to_mat.get((p, q))
        if p > q:
            m = pair_to_mat.get((q, p))
            return None if m is None else m.T
        m = pair_to_mat.get((p, p))
        return None if m is None else (m + m.T)

    i_all = np.arange(int(n_act), dtype=np.int64)
    for x in range(int(n_virt)):
        partners: list[int] = []
        mats: list[np.ndarray] = []
        for p in range(int(n_virt)):
            m = _get_pair_mat(p, x)
            if m is None or not np.any(m):
                continue
            partners.append(p)
            mats.append(m)

        if not partners:
            continue
        partners_arr = np.asarray(partners, dtype=np.int64)
        c_stack = np.stack(mats, axis=0)
        dd_x = np.einsum("prs,qtu,rijtsu->pijq", c_stack, c_stack, dm3_int, optimize=True)
        out[np.ix_(partners_arr, i_all, i_all, partners_arr)] += dd_x

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intext_extint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,ext,ext,int] for ic-MRCISD (ΔN_ext=0 block).

    This corresponds to the correlated-space block:
      dm2[i,a,b,j]  (a,b external; i,j internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, n_act:, n_act:, :n_act], dtype=np.float64)

    blk = ic_mrcisd_make_rdm2_extint_intext_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    return np.asarray(blk.transpose(1, 0, 3, 2), dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extext_intint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,ext,int,int] for ic-MRCISD (ΔN_ext=0 block).

    This corresponds to the correlated-space block:
      dm2[a,b,i,j]  (a,b external; i,j internal)
    in cuGUGA's delta-corrected spin-free convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s} - δ_{q r} E_{p s}>.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, n_act:, :n_act, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm2_int is None or dm3_int is None:
        dm2_int, dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Singles–singles: c_a c_b * Σ_{r,s} dm2_int[r,s,i,j]
        blk_ss = dm2_int.sum(axis=(0, 1))
        out = blk_ss[None, None, :, :] * cs[:, None, None, None] * cs[None, :, None, None]

        if int(n_doubles) == 0 or not np.any(c_d):
            return np.asarray(out, dtype=np.float64, order="C")

        base_off = np.ones((int(n_act), int(n_act)), dtype=np.float64)
        if not allow_same_internal:
            np.fill_diagonal(base_off, 0.0)
        base_diag = np.triu(
            np.ones((int(n_act), int(n_act)), dtype=np.float64), k=0 if allow_same_internal else 1
        )
        base_diag = base_diag + base_diag.T

        cd_map: dict[tuple[int, int], float] = {}
        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            w = float(c_d[idx])
            if w == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")
            cd_map[(a, b)] = w

        i_all = np.arange(int(n_act), dtype=np.int64)
        for x in range(int(n_virt)):
            partners: list[int] = []
            mats: list[np.ndarray] = []
            for p in range(int(n_virt)):
                key = (p, x) if p <= x else (x, p)
                w = float(cd_map.get(key, 0.0))
                if w == 0.0:
                    continue
                mat = base_diag if p == x else base_off
                partners.append(p)
                mats.append(w * mat)

            if not partners:
                continue
            partners_arr = np.asarray(partners, dtype=np.int64)
            c_stack = np.stack(mats, axis=0)
            dd_x = np.einsum("prs,qtu,rtijsu->pqij", c_stack, c_stack, dm3_int, optimize=True)
            out[np.ix_(partners_arr, partners_arr, i_all, i_all)] += dd_x

        return np.asarray(out, dtype=np.float64, order="C")

    # FIC path.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    out = np.einsum("ar,bs,rsij->abij", cs, cs, dm2_int, optimize=True)

    if int(n_doubles) == 0 or not np.any(c_d):
        return np.asarray(out, dtype=np.float64, order="C")

    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    pair_to_mat: dict[tuple[int, int], np.ndarray] = {}
    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        idx = order[start:stop].astype(np.int64, copy=False)
        cd = c_d[idx]
        if not np.any(cd):
            continue

        a_glob = int(keys[g, 0]) - int(n_act)
        b_glob = int(keys[g, 1]) - int(n_act)
        if not (0 <= a_glob < int(n_virt) and 0 <= b_glob < int(n_virt)):
            raise ValueError("doubles external labels out of range for contiguous orbital convention")

        r_idx = r_all[idx]
        s_idx = s_all[idx]
        mat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
        np.add.at(mat, (r_idx, s_idx), cd)
        pair_to_mat[(a_glob, b_glob)] = mat

    def _get_pair_mat(p: int, q: int) -> np.ndarray | None:
        p = int(p)
        q = int(q)
        if p < q:
            return pair_to_mat.get((p, q))
        if p > q:
            m = pair_to_mat.get((q, p))
            return None if m is None else m.T
        m = pair_to_mat.get((p, p))
        return None if m is None else (m + m.T)

    i_all = np.arange(int(n_act), dtype=np.int64)
    for x in range(int(n_virt)):
        partners: list[int] = []
        mats: list[np.ndarray] = []
        for p in range(int(n_virt)):
            m = _get_pair_mat(p, x)
            if m is None or not np.any(m):
                continue
            partners.append(p)
            mats.append(m)

        if not partners:
            continue
        partners_arr = np.asarray(partners, dtype=np.int64)
        c_stack = np.stack(mats, axis=0)
        dd_x = np.einsum("prs,qtu,rtijsu->pqij", c_stack, c_stack, dm3_int, optimize=True)
        out[np.ix_(partners_arr, partners_arr, i_all, i_all)] += dd_x

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intint_extext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,int,ext,ext] for ic-MRCISD (ΔN_ext=0 block).

    This corresponds to the correlated-space block:
      dm2[i,j,a,b]  (a,b external; i,j internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, :n_act, n_act:, n_act:], dtype=np.float64)

    blk = ic_mrcisd_make_rdm2_extext_intint_phase3(
        ic_res,
        ci_cas=ci_cas,
        backend="direct",
        rdm_backend=rdm_backend,
        dm1_int=dm1_int,
        dm2_int=dm2_int,
        dm3_int=dm3_int,
        dm4_ctx=dm4_ctx,
    )
    return np.asarray(blk.transpose(2, 3, 0, 1), dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extint_intint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,int,int,int] for ic-MRCISD (ΔN_ext=+1 block).

    This corresponds to the correlated-space block:
      dm2[a,i,j,k]  (a external; i,j,k internal)
    in cuGUGA's delta-corrected spin-free convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s} - δ_{q r} E_{p s}>.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, :n_act, :n_act, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    dm2_int, dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        # Pack strongly contracted singles coefficients: cs[a] = c_a.
        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Reference–singles: c0 * c_a * Σ_r dm2_int[r,i,j,k].
        t_ref = dm2_int.sum(axis=0)
        out = float(c0) * cs[:, None, None, None] * t_ref[None, :, :, :]

        # Doubles–singles: c_{ab} c_b * Σ_{r,s,t} dm3_int[r,i,j,k,s,t], with r!=s excluded if requested.
        t_full = dm3_int.sum(axis=(0, 4, 5))
        t_diag = np.einsum("rijkrt->ijk", dm3_int, optimize=True)
        if allow_same_internal:
            t_diff = t_full
            t_same = t_full + t_diag
        else:
            t_diff = t_full - t_diag
            t_same = t_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles labels out of range for contiguous orbital convention")

            if a == b:
                if cs[a] != 0.0:
                    out[a] += cd * cs[a] * t_same
            else:
                if cs[b] != 0.0:
                    out[a] += cd * cs[b] * t_diff
                if cs[a] != 0.0:
                    out[b] += cd * cs[a] * t_diff

        return np.asarray(out, dtype=np.float64, order="C")

    # Pack FIC singles coefficients into a dense (nV, nI) array: cs[a,i] = c_{a i}.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Reference–singles contribution: c0 * Σ_r c_{a r} dm2_int[r,i,j,k].
    out = float(c0) * np.einsum("ar,rijk->aijk", cs, dm2_int, optimize=True)

    # Doubles–singles contribution: Σ_{(a,b;r,s),t} c_{ab;rs} c_{b t} dm3_int[r,i,j,k,s,t]
    # plus the symmetric contribution that swaps (a,r) and (b,s), which naturally
    # accumulates into out[b] as well.
    da = np.asarray(doubles.a, dtype=np.int64).ravel()
    db = np.asarray(doubles.b, dtype=np.int64).ravel()
    dr = np.asarray(doubles.r, dtype=np.int64).ravel()
    ds = np.asarray(doubles.s, dtype=np.int64).ravel()
    for idx in range(n_doubles):
        cd = float(c_d[idx])
        if cd == 0.0:
            continue
        a = int(da[idx]) - int(n_act)
        b = int(db[idx]) - int(n_act)
        r = int(dr[idx])
        s = int(ds[idx])
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt) and 0 <= r < int(n_act) and 0 <= s < int(n_act)):
            raise ValueError("doubles labels out of range for contiguous orbital convention")

        cs_b = cs[b]
        if np.any(cs_b):
            out[a] += cd * np.einsum("t,ijkt->ijk", cs_b, dm3_int[r, :, :, :, s, :], optimize=True)

        cs_a = cs[a]
        if np.any(cs_a):
            out[b] += cd * np.einsum("t,ijkt->ijk", cs_a, dm3_int[s, :, :, :, r, :], optimize=True)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intint_intext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,int,int,ext] for ic-MRCISD (ΔN_ext=-1 block).

    This corresponds to the correlated-space block:
      dm2[i,j,k,a]  (a external; i,j,k internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, :n_act, :n_act, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    dm2_int, dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Reference–singles: c0 * c_a * Σ_t dm2_int[i,j,k,t].
        t_ref = dm2_int.sum(axis=3)
        out = float(c0) * t_ref[:, :, :, None] * cs[None, None, None, :]

        # Doubles–singles: c_{ab} c_b * Σ_{r,s,t} dm3_int[i,j,k,r,t,s], with r!=s excluded if requested.
        t_full = dm3_int.sum(axis=(3, 4, 5))
        t_diag = np.einsum("ijkrtr->ijk", dm3_int, optimize=True)
        if allow_same_internal:
            t_diff = t_full
            t_same = t_full + t_diag
        else:
            t_diff = t_full - t_diag
            t_same = t_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles labels out of range for contiguous orbital convention")

            if a == b:
                if cs[a] != 0.0:
                    out[:, :, :, a] += cd * cs[a] * t_same
            else:
                if cs[b] != 0.0:
                    out[:, :, :, a] += cd * cs[b] * t_diff
                if cs[a] != 0.0:
                    out[:, :, :, b] += cd * cs[a] * t_diff

        return np.asarray(out, dtype=np.float64, order="C")

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Reference–singles contribution: c0 * Σ_t c_{a t} dm2_int[i,j,k,t].
    out = float(c0) * np.einsum("at,ijkt->ijka", cs, dm2_int, optimize=True)

    # Singles–doubles contribution: Σ_{(a,b;r,s),t} c_{b t} c_{ab;rs} dm3_int[i,j,k,r,t,s]
    # implemented by accumulating into both out[...,a] and out[...,b].
    da = np.asarray(doubles.a, dtype=np.int64).ravel()
    db = np.asarray(doubles.b, dtype=np.int64).ravel()
    dr = np.asarray(doubles.r, dtype=np.int64).ravel()
    ds = np.asarray(doubles.s, dtype=np.int64).ravel()
    for idx in range(n_doubles):
        cd = float(c_d[idx])
        if cd == 0.0:
            continue
        a = int(da[idx]) - int(n_act)
        b = int(db[idx]) - int(n_act)
        r = int(dr[idx])
        s = int(ds[idx])
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt) and 0 <= r < int(n_act) and 0 <= s < int(n_act)):
            raise ValueError("doubles labels out of range for contiguous orbital convention")

        cs_b = cs[b]
        if np.any(cs_b):
            out[:, :, :, a] += cd * np.einsum("t,ijkt->ijk", cs_b, dm3_int[:, :, :, r, :, s], optimize=True)

        cs_a = cs[a]
        if np.any(cs_a):
            out[:, :, :, b] += cd * np.einsum("t,ijkt->ijk", cs_a, dm3_int[:, :, :, s, :, r], optimize=True)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intext_intint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,ext,int,int] for ic-MRCISD (ΔN_ext=+1 block).

    This corresponds to the correlated-space block:
      dm2[i,a,j,k]  (a external; i,j,k internal)
    in cuGUGA's delta-corrected spin-free convention:
      dm2[p,q,r,s] = <E_{p q} E_{r s} - δ_{q r} E_{p s}>.
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, n_act:, :n_act, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm2_int is None or dm3_int is None:
        dm2_int, dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Reference–singles: c0 * c_a * Σ_t dm2_int[j,k,i,t].
        t_ref = dm2_int.sum(axis=3).transpose(2, 0, 1)
        out = float(c0) * t_ref[:, None, :, :] * cs[None, :, None, None]

        # Doubles–singles: c_{ab} c_b * Σ_{r,s,t} dm3_int[i,r,t,s,j,k], with r!=s excluded if requested.
        t_full = dm3_int.sum(axis=(1, 2, 3))
        t_diag = np.einsum("irtrjk->ijk", dm3_int, optimize=True)
        if allow_same_internal:
            t_diff = t_full
            t_same = t_full + t_diag
        else:
            t_diff = t_full - t_diag
            t_same = t_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles labels out of range for contiguous orbital convention")

            if a == b:
                if cs[a] != 0.0:
                    out[:, a] += cd * cs[a] * t_same
            else:
                if cs[b] != 0.0:
                    out[:, a] += cd * cs[b] * t_diff
                if cs[a] != 0.0:
                    out[:, b] += cd * cs[a] * t_diff

        return np.asarray(out, dtype=np.float64, order="C")

    # Pack FIC singles coefficients into a dense (nV, nI) array: cs[a,i] = c_{a i}.
    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Reference–singles contribution: c0 * Σ_t c_{a t} dm2_int[j,k,i,t].
    out = float(c0) * np.einsum("at,jkit->iajk", cs, dm2_int, optimize=True)

    # Singles–doubles contribution: Σ_{(a,b;r,s),t} c_{b t} c_{ab;rs} dm3_int[i,r,t,s,j,k]
    # implemented by accumulating into both out[:,a,:,:] and out[:,b,:,:].
    da = np.asarray(doubles.a, dtype=np.int64).ravel()
    db = np.asarray(doubles.b, dtype=np.int64).ravel()
    dr = np.asarray(doubles.r, dtype=np.int64).ravel()
    ds = np.asarray(doubles.s, dtype=np.int64).ravel()
    for idx in range(n_doubles):
        cd = float(c_d[idx])
        if cd == 0.0:
            continue
        a = int(da[idx]) - int(n_act)
        b = int(db[idx]) - int(n_act)
        r = int(dr[idx])
        s = int(ds[idx])
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt) and 0 <= r < int(n_act) and 0 <= s < int(n_act)):
            raise ValueError("doubles labels out of range for contiguous orbital convention")

        cs_b = cs[b]
        if np.any(cs_b):
            out[:, a] += cd * np.einsum("t,itjk->ijk", cs_b, dm3_int[:, r, :, s, :, :], optimize=True)

        cs_a = cs[a]
        if np.any(cs_a):
            out[:, b] += cd * np.einsum("t,itjk->ijk", cs_a, dm3_int[:, s, :, r, :, :], optimize=True)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intint_extint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,int,ext,int] for ic-MRCISD (ΔN_ext=-1 block).

    This corresponds to the correlated-space block:
      dm2[i,j,a,k]  (a external; i,j,k internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, :n_act, n_act:, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm2_int is None or dm3_int is None:
        dm2_int, dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        # Reference–singles: c0 * c_a * Σ_t dm2_int[i,j,t,k].
        t_ref = dm2_int.sum(axis=2)
        out = float(c0) * t_ref[:, :, None, :] * cs[None, None, :, None]

        # Doubles–singles: c_{ab} c_b * Σ_{r,s,t} dm3_int[i,j,s,t,r,k], with r!=s excluded if requested.
        t_full = dm3_int.sum(axis=(2, 3, 4))
        t_diag = np.einsum("ijrtrk->ijk", dm3_int, optimize=True)
        if allow_same_internal:
            t_diff = t_full
            t_same = t_full + t_diag
        else:
            t_diff = t_full - t_diag
            t_same = t_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles labels out of range for contiguous orbital convention")

            if a == b:
                if cs[a] != 0.0:
                    out[:, :, a] += cd * cs[a] * t_same
            else:
                if cs[b] != 0.0:
                    out[:, :, a] += cd * cs[b] * t_diff
                if cs[a] != 0.0:
                    out[:, :, b] += cd * cs[a] * t_diff

        return np.asarray(out, dtype=np.float64, order="C")

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    # Reference–singles contribution: c0 * Σ_t c_{a t} dm2_int[i,j,t,k].
    out = float(c0) * np.einsum("at,ijtk->ijak", cs, dm2_int, optimize=True)

    # Singles–doubles contribution: Σ_{(a,b;r,s),t} c_{b t} c_{ab;rs} dm3_int[i,j,s,t,r,k]
    # implemented by accumulating into out[:,:,a,:] and out[:,:,b,:].
    da = np.asarray(doubles.a, dtype=np.int64).ravel()
    db = np.asarray(doubles.b, dtype=np.int64).ravel()
    dr = np.asarray(doubles.r, dtype=np.int64).ravel()
    ds = np.asarray(doubles.s, dtype=np.int64).ravel()
    for idx in range(n_doubles):
        cd = float(c_d[idx])
        if cd == 0.0:
            continue
        a = int(da[idx]) - int(n_act)
        b = int(db[idx]) - int(n_act)
        r = int(dr[idx])
        s = int(ds[idx])
        if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt) and 0 <= r < int(n_act) and 0 <= s < int(n_act)):
            raise ValueError("doubles labels out of range for contiguous orbital convention")

        cs_b = cs[b]
        if np.any(cs_b):
            out[:, :, a] += cd * np.einsum("t,ijtk->ijk", cs_b, dm3_int[:, :, s, :, r, :], optimize=True)

        cs_a = cs[a]
        if np.any(cs_a):
            out[:, :, b] += cd * np.einsum("t,ijtk->ijk", cs_a, dm3_int[:, :, r, :, s, :], optimize=True)

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extint_extext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,int,ext,ext] for ic-MRCISD (3-external block).

    This corresponds to the correlated-space block:
      dm2[a,i,b,c]  (a,b,c external; i internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, :n_act, n_act:, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if int(n_singles) == 0 or not np.any(c_s) or int(n_doubles) == 0 or not np.any(c_d):
        return np.zeros((int(n_virt), int(n_act), int(n_virt), int(n_virt)), dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
    out = np.zeros((int(n_virt), int(n_act), int(n_virt), int(n_virt)), dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        blk_full = dm2_int.sum(axis=(0, 2, 3))
        blk_diag = np.einsum("rirt->i", dm2_int, optimize=True)
        if allow_same_internal:
            blk_diff = blk_full
            blk_same = blk_full + blk_diag
        else:
            blk_diff = blk_full - blk_diag
            blk_same = blk_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")

            blk = blk_same if a == b else blk_diff
            if np.any(cs):
                out[a, :, b, :] += cd * blk[:, None] * cs[None, :]
                if a != b:
                    out[b, :, a, :] += cd * blk[:, None] * cs[None, :]

        return np.asarray(out, dtype=np.float64, order="C")

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    pair_to_mat = _fic_build_doubles_pair_to_mat(doubles, c_d, n_act=n_act, n_virt=n_virt)
    if not pair_to_mat:
        return np.asarray(out, dtype=np.float64, order="C")

    cs_t = cs.T
    for (a, b), mat0 in pair_to_mat.items():
        mat_ab = mat0 + mat0.T if a == b else mat0
        tmp = np.einsum("rs,rist->it", mat_ab, dm2_int, optimize=True)
        out[a, :, b, :] += tmp @ cs_t
        if a != b:
            tmp_ba = np.einsum("rs,rist->it", mat0.T, dm2_int, optimize=True)
            out[b, :, a, :] += tmp_ba @ cs_t

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_intext_extext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[int,ext,ext,ext] for ic-MRCISD (3-external block).

    This corresponds to the correlated-space block:
      dm2[i,a,b,c]  (a,b,c external; i internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[:n_act, n_act:, n_act:, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if int(n_singles) == 0 or not np.any(c_s) or int(n_doubles) == 0 or not np.any(c_d):
        return np.zeros((int(n_act), int(n_virt), int(n_virt), int(n_virt)), dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
    out = np.zeros((int(n_act), int(n_virt), int(n_virt), int(n_virt)), dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        blk_full = dm2_int.sum(axis=(0, 2, 3))
        blk_diag = np.einsum("rirt->i", dm2_int, optimize=True)
        if allow_same_internal:
            blk_diff = blk_full
            blk_same = blk_full + blk_diag
        else:
            blk_diff = blk_full - blk_diag
            blk_same = blk_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")

            blk = blk_same if a == b else blk_diff
            if np.any(cs):
                out[:, a, :, b] += cd * blk[:, None] * cs[None, :]
                if a != b:
                    out[:, b, :, a] += cd * blk[:, None] * cs[None, :]

        return np.asarray(out, dtype=np.float64, order="C")

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    pair_to_mat = _fic_build_doubles_pair_to_mat(doubles, c_d, n_act=n_act, n_virt=n_virt)
    if not pair_to_mat:
        return np.asarray(out, dtype=np.float64, order="C")

    cs_t = cs.T
    for (a, b), mat0 in pair_to_mat.items():
        mat_ab = mat0 + mat0.T if a == b else mat0
        tmp = np.einsum("rs,rist->it", mat_ab, dm2_int, optimize=True)
        out[:, a, :, b] += tmp @ cs_t
        if a != b:
            tmp_ba = np.einsum("rs,rist->it", mat0.T, dm2_int, optimize=True)
            out[:, b, :, a] += tmp_ba @ cs_t

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extext_intext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,ext,int,ext] for ic-MRCISD (3-external block).

    This corresponds to the correlated-space block:
      dm2[a,b,i,c]  (a,b,c external; i internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, n_act:, :n_act, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if int(n_singles) == 0 or not np.any(c_s) or int(n_doubles) == 0 or not np.any(c_d):
        return np.zeros((int(n_virt), int(n_virt), int(n_act), int(n_virt)), dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
    out = np.zeros((int(n_virt), int(n_virt), int(n_act), int(n_virt)), dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        blk_full = dm2_int.sum(axis=(0, 1, 3))
        blk_diag = np.einsum("trir->i", dm2_int, optimize=True)
        if allow_same_internal:
            blk_diff = blk_full
            blk_same = blk_full + blk_diag
        else:
            blk_diff = blk_full - blk_diag
            blk_same = blk_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")

            blk = blk_same if a == b else blk_diff
            if np.any(cs):
                out[:, a, :, b] += cd * cs[:, None] * blk[None, :]
                if a != b:
                    out[:, b, :, a] += cd * cs[:, None] * blk[None, :]

        return np.asarray(out, dtype=np.float64, order="C")

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    pair_to_mat = _fic_build_doubles_pair_to_mat(doubles, c_d, n_act=n_act, n_virt=n_virt)
    if not pair_to_mat:
        return np.asarray(out, dtype=np.float64, order="C")

    for (a, b), mat0 in pair_to_mat.items():
        mat_ab = mat0 + mat0.T if a == b else mat0
        tmp = np.einsum("rs,tris->ti", mat_ab, dm2_int, optimize=True)
        out[:, a, :, b] += cs @ tmp
        if a != b:
            tmp_ba = np.einsum("rs,tris->ti", mat0.T, dm2_int, optimize=True)
            out[:, b, :, a] += cs @ tmp_ba

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extext_extint_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,ext,ext,int] for ic-MRCISD (3-external block).

    This corresponds to the correlated-space block:
      dm2[a,b,c,i]  (a,b,c external; i internal).
    """

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, n_act:, n_act:, :n_act], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if int(n_singles) == 0 or not np.any(c_s) or int(n_doubles) == 0 or not np.any(c_d):
        return np.zeros((int(n_virt), int(n_virt), int(n_virt), int(n_act)), dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
    out = np.zeros((int(n_virt), int(n_virt), int(n_virt), int(n_act)), dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        cs = np.zeros(int(n_virt), dtype=np.float64)
        a_all = np.asarray(singles.a, dtype=np.int64).ravel()
        for idx in range(n_singles):
            a = int(a_all[idx]) - int(n_act)
            if not (0 <= a < int(n_virt)):
                raise ValueError("singles labels out of range for contiguous orbital convention")
            cs[a] = float(c_s[idx])

        blk_full = dm2_int.sum(axis=(0, 1, 3))
        blk_diag = np.einsum("trir->i", dm2_int, optimize=True)
        if allow_same_internal:
            blk_diff = blk_full
            blk_same = blk_full + blk_diag
        else:
            blk_diff = blk_full - blk_diag
            blk_same = blk_diff

        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles external labels out of range for contiguous orbital convention")

            blk = blk_same if a == b else blk_diff
            if np.any(cs):
                out[a, :, b, :] += cd * cs[:, None] * blk[None, :]
                if a != b:
                    out[b, :, a, :] += cd * cs[:, None] * blk[None, :]

        return np.asarray(out, dtype=np.float64, order="C")

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    pair_to_mat = _fic_build_doubles_pair_to_mat(doubles, c_d, n_act=n_act, n_virt=n_virt)
    if not pair_to_mat:
        return np.asarray(out, dtype=np.float64, order="C")

    for (a, b), mat0 in pair_to_mat.items():
        mat_ab = mat0 + mat0.T if a == b else mat0
        tmp = np.einsum("rs,tris->ti", mat_ab, dm2_int, optimize=True)
        out[a, :, b, :] += cs @ tmp
        if a != b:
            tmp_ba = np.einsum("rs,tris->ti", mat0.T, dm2_int, optimize=True)
            out[b, :, a, :] += cs @ tmp_ba

    return np.asarray(out, dtype=np.float64, order="C")


def ic_mrcisd_make_rdm2_extext_extext_phase3(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: _CasDm4ContractCtx | None = None,
) -> np.ndarray:
    """Return dm2[ext,ext,ext,ext] for ic-MRCISD (4-external block)."""

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = _infer_n_act_n_virt(ic_res)
        _dm1, dm2 = ic_mrcisd_make_rdm12(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm2[n_act:, n_act:, n_act:, n_act:], dtype=np.float64)

    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = _infer_n_act_n_virt(ic_res)
    _require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if int(n_doubles) == 0 or not np.any(c_d):
        return np.zeros((int(n_virt), int(n_virt), int(n_virt), int(n_virt)), dtype=np.float64)

    if dm2_int is None:
        dm2_int, _dm3_int = _cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm2_int = np.asarray(dm2_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

        base_off = np.ones((int(n_act), int(n_act)), dtype=np.float64)
        if not allow_same_internal:
            np.fill_diagonal(base_off, 0.0)
        base_diag = np.triu(
            np.ones((int(n_act), int(n_act)), dtype=np.float64), k=0 if allow_same_internal else 1
        )
        base_diag = base_diag + base_diag.T

        C_all = np.zeros((int(n_virt), int(n_virt), int(n_act), int(n_act)), dtype=np.float64)
        da = np.asarray(doubles.a, dtype=np.int64).ravel()
        db = np.asarray(doubles.b, dtype=np.int64).ravel()
        for idx in range(n_doubles):
            cd = float(c_d[idx])
            if cd == 0.0:
                continue
            a = int(da[idx]) - int(n_act)
            b = int(db[idx]) - int(n_act)
            if not (0 <= a < int(n_virt) and 0 <= b < int(n_virt)):
                raise ValueError("doubles labels out of range for contiguous orbital convention")

            base = base_diag if a == b else base_off
            C_all[a, b] += cd * base
            if a != b:
                C_all[b, a] += cd * base.T

        out = np.einsum("acrs,bdtu,rtsu->abcd", C_all, C_all, dm2_int, optimize=True)
        return np.asarray(out, dtype=np.float64, order="C")

    pair_to_mat = _fic_build_doubles_pair_to_mat(doubles, c_d, n_act=n_act, n_virt=n_virt)
    if not pair_to_mat:
        return np.zeros((int(n_virt), int(n_virt), int(n_virt), int(n_virt)), dtype=np.float64)

    C_all = np.zeros((int(n_virt), int(n_virt), int(n_act), int(n_act)), dtype=np.float64)
    for (a, b), mat0 in pair_to_mat.items():
        if a == b:
            C_all[a, a] = mat0 + mat0.T
        else:
            C_all[a, b] = mat0
            C_all[b, a] = mat0.T

    out = np.einsum("acrs,bdtu,rtsu->abcd", C_all, C_all, dm2_int, optimize=True)
    return np.asarray(out, dtype=np.float64, order="C")


def _fic_build_doubles_pair_to_mat(
    doubles: ICDoubles, c_d: np.ndarray, *, n_act: int, n_virt: int
) -> dict[tuple[int, int], np.ndarray]:
    """Return mapping (a,b)->C[a,b](r,s) for FIC doubles labels.

    The returned keys use *relative* external indices (0..n_virt-1). The stored
    matrices correspond to the canonical label ordering of `doubles` and must be
    symmetrized/transposed by the caller depending on the external pair order.
    """

    pair_to_mat: dict[tuple[int, int], np.ndarray] = {}
    order = np.asarray(doubles.ab_group_order, dtype=np.int32)
    offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int32)
    keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)
    r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
    s_all = np.asarray(doubles.s, dtype=np.int64).ravel()

    for g in range(int(doubles.n_groups)):
        start = int(offsets[g])
        stop = int(offsets[g + 1])
        if start == stop:
            continue

        idx = order[start:stop].astype(np.int64, copy=False)
        cd = np.asarray(c_d[idx], dtype=np.float64)
        if not np.any(cd):
            continue

        a_rel = int(keys[g, 0]) - int(n_act)
        b_rel = int(keys[g, 1]) - int(n_act)
        if not (0 <= a_rel < int(n_virt) and 0 <= b_rel < int(n_virt)):
            raise ValueError("doubles external labels out of range for contiguous orbital convention")

        r_idx = r_all[idx]
        s_idx = s_all[idx]
        mat = np.zeros((int(n_act), int(n_act)), dtype=np.float64)
        np.add.at(mat, (r_idx, s_idx), cd)
        pair_to_mat[(a_rel, b_rel)] = mat

    return pair_to_mat
