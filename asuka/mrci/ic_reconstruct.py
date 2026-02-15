from __future__ import annotations

"""Reconstruction utilities for contracted ic-MRCISD.

This module provides helpers to reconstruct a contracted ic-MRCISD wavefunction
in the uncontracted MRCISD CSF basis by explicitly applying spin-free generators
(E_pq) to the embedded reference vector.

Target:
- Small-system validation of contracted energies/residuals vs uncontracted code.
- Building correlated-space RDMs for analytic gradients (before Phase 3 direct
  builders are fully available).
"""

from typing import Any

import numpy as np

from asuka.cuguga.drt import DRT, build_drt
from asuka.mrci.ic_basis import ICDoubles, ICSingles, SCDoubles, SCSingles
from asuka.mrci.mrcisd import embed_cas_ci_into_mrcisd


_STEP_TO_OCC_F64 = np.asarray([0.0, 1.0, 1.0, 2.0], dtype=np.float64)  # E,U,L,D


def _normalize_ci(ci: np.ndarray) -> np.ndarray:
    ci = np.asarray(ci, dtype=np.float64).ravel()
    n = float(np.linalg.norm(ci))
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("ci vector must have nonzero finite norm")
    return np.asarray(ci / n, dtype=np.float64)


def _apply_epq(drt: DRT, *, p: int, q: int, x: np.ndarray, cache: Any) -> np.ndarray:
    """Return y = E_pq |x> in the CSF basis (p!=q only)."""

    from asuka.cuguga.oracle import _csr_for_epq  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64).ravel()
    if int(x.size) != int(drt.ncsf):
        raise ValueError("x has wrong length for this DRT")
    p = int(p)
    q = int(q)
    if p == q:
        # E_pp is diagonal in the occupation representation; we do not need it for
        # the contracted basis vectors used here (all have p!=q).
        return np.zeros_like(x)

    csr = _csr_for_epq(cache, drt, p, q)
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    data = np.asarray(csr.data, dtype=np.float64)

    y = np.zeros_like(x)
    nz = np.nonzero(x)[0]
    for j in nz.tolist():
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if start == end:
            continue
        y[indices[start:end]] += data[start:end] * float(x[j])
    return y


def _apply_epq_inplace(drt: DRT, cache: Any, x: np.ndarray, *, p: int, q: int, out: np.ndarray) -> None:
    """Fill ``out[:] = E_pq |x>`` in the CSF basis."""

    from asuka.cuguga.oracle import _csr_for_epq  # noqa: PLC0415

    x = np.asarray(x, dtype=np.float64).ravel()
    if int(x.size) != int(drt.ncsf):
        raise ValueError("x has wrong length for this DRT")
    out = np.asarray(out, dtype=np.float64).ravel()
    if int(out.size) != int(drt.ncsf):
        raise ValueError("out has wrong length for this DRT")

    p = int(p)
    q = int(q)
    if p == q:
        steps = getattr(cache, "steps", None)
        if steps is None:
            raise TypeError("cache must be an EPQ action cache with .steps")
        out[:] = _STEP_TO_OCC_F64[np.asarray(steps, dtype=np.int8)[:, p]] * x
        return

    csr = _csr_for_epq(cache, drt, p, q)
    indptr = np.asarray(csr.indptr, dtype=np.int32)
    indices = np.asarray(csr.indices, dtype=np.int32)
    data = np.asarray(csr.data, dtype=np.float64)

    out.fill(0.0)
    for j in range(int(x.size)):
        xj = float(x[j])
        if xj == 0.0:
            continue
        start = int(indptr[j])
        end = int(indptr[j + 1])
        if start == end:
            continue
        out[indices[start:end]] += data[start:end] * xj


def reconstruct_uncontracted_ci_from_ic_mrcisd(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    max_ncsf: int = 200_000,
    max_nlab: int = 20_000,
) -> tuple[DRT, np.ndarray]:
    """Reconstruct the contracted ic-MRCISD wavefunction as an uncontracted CI vector.

    Parameters
    ----------
    ic_res : Any
        :class:`~asuka.mrci.mrcisd.ICMRCISDResult` instance.
    ci_cas : np.ndarray
        Reference CAS CI vector in the CAS DRT ordering (same as ``mc.ci``).
    max_ncsf : int, optional
        Maximum number of CSFs allowed. Default is 200,000.
    max_nlab : int, optional
        Maximum number of contracted labels allowed. Default is 20,000.

    Returns
    -------
    drt_mrci : DRT
        The uncontracted MRCISD DRT.
    ci_mrci : np.ndarray
        The reconstructed CI vector in the uncontracted basis.
    """

    drt_mrci = getattr(ic_res, "drt_work", None)
    if drt_mrci is None:
        raise NotImplementedError(
            "Reconstruction requires ic-MRCISD backend='semi_direct' (missing drt_work)."
        )
    if not isinstance(drt_mrci, DRT):
        raise TypeError("ic_res.drt_work must be a DRT instance")

    ncsf = int(drt_mrci.ncsf)
    if ncsf < 1:
        raise ValueError("empty DRT")
    if ncsf > int(max_ncsf):
        raise NotImplementedError(f"reconstruction disabled for ncsf={ncsf} > max_ncsf={int(max_ncsf)}")

    spaces = getattr(ic_res, "spaces", None)
    if spaces is None:
        raise TypeError("ic_res missing OrbitalSpaces")

    if getattr(spaces, "orbsym", None) is not None:
        raise NotImplementedError("reconstruction currently supports only symmetry=None (spaces.orbsym is None)")

    n_act = int(getattr(spaces, "n_internal"))
    n_virt = int(getattr(spaces, "n_external"))
    if n_act < 0 or n_virt < 0:
        raise ValueError("invalid orbital spaces")

    # Build the CAS DRT in the same convention as ic_mrcisd_kernel when symmetry is disabled.
    drt_cas = build_drt(norb=n_act, nelec=int(drt_mrci.nelec), twos_target=int(drt_mrci.twos_target))
    _, psi0, _ref_idx = embed_cas_ci_into_mrcisd(drt_cas=drt_cas, drt_mrci=drt_mrci, ci_cas=ci_cas, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    n_singles = int(getattr(singles, "nlab"))
    n_doubles = int(getattr(doubles, "nlab"))
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    if int(1 + n_singles + n_doubles) > int(max_nlab):
        raise NotImplementedError(
            f"reconstruction disabled for nlab={int(1+n_singles+n_doubles)} > max_nlab={int(max_nlab)}"
        )

    from asuka.cuguga.oracle import _get_epq_action_cache  # noqa: PLC0415

    cache = _get_epq_action_cache(drt_mrci)

    ci_out = np.asarray(psi0, dtype=np.float64) * float(c[0])

    diag = getattr(ic_res, "diagnostics", {}) or {}
    try:
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.5)
    except Exception:
        allow_same_internal = True

    # Singles block.
    if isinstance(singles, ICSingles):
        for i in range(n_singles):
            a = int(singles.a[i])
            r = int(singles.r[i])
            vec = _apply_epq(drt_mrci, p=a, q=r, x=psi0, cache=cache)
            ci_out += float(c[1 + i]) * vec
    elif isinstance(singles, SCSingles):
        internal = np.asarray(spaces.internal, dtype=np.int32).ravel()
        for i in range(n_singles):
            a = int(singles.a[i])
            vec = np.zeros_like(ci_out)
            for r in internal.tolist():
                vec += _apply_epq(drt_mrci, p=a, q=int(r), x=psi0, cache=cache)
            ci_out += float(c[1 + i]) * vec
    else:
        raise TypeError("unknown singles label type")

    # Doubles block.
    off = 1 + n_singles
    if isinstance(doubles, ICDoubles):
        for i in range(n_doubles):
            a = int(doubles.a[i])
            r = int(doubles.r[i])
            b = int(doubles.b[i])
            s = int(doubles.s[i])
            tmp = _apply_epq(drt_mrci, p=a, q=r, x=psi0, cache=cache)
            vec = _apply_epq(drt_mrci, p=b, q=s, x=tmp, cache=cache)
            ci_out += float(c[off + i]) * vec
    elif isinstance(doubles, SCDoubles):
        internal = np.asarray(spaces.internal, dtype=np.int32).ravel()
        for i in range(n_doubles):
            a = int(doubles.a[i])
            b = int(doubles.b[i])
            vec = np.zeros_like(ci_out)
            internal_list = internal.tolist()
            if a == b:
                for ir, r in enumerate(internal_list):
                    start_s = ir if allow_same_internal else ir + 1
                    for s in internal_list[start_s:]:
                        tmp1 = _apply_epq(drt_mrci, p=a, q=int(s), x=psi0, cache=cache)
                        vec += _apply_epq(drt_mrci, p=a, q=int(r), x=tmp1, cache=cache)
            else:
                for r in internal_list:
                    tmp_r = _apply_epq(drt_mrci, p=a, q=int(r), x=psi0, cache=cache)
                    for s in internal_list:
                        if (not allow_same_internal) and int(r) == int(s):
                            continue
                        vec += _apply_epq(drt_mrci, p=b, q=int(s), x=tmp_r, cache=cache)
            ci_out += float(c[off + i]) * vec
    else:
        raise TypeError("unknown doubles label type")

    return drt_mrci, np.ascontiguousarray(ci_out, dtype=np.float64)


def reconstruct_uncontracted_ci_from_ic_mrcisd_staged(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    max_ncsf: int = 200_000,
    max_nlab: int = 20_000,
    max_ncsf_times_nint: int = 5_000_000,
) -> tuple[DRT, np.ndarray]:
    """Reconstruct the contracted ic-MRCISD wavefunction with staged operations.

    Reduces the number of ``E_pq`` applications for doubles by staging over internal
    indices and using BLAS to form linear combinations.

    Parameters
    ----------
    ic_res : Any
        :class:`~asuka.mrci.mrcisd.ICMRCISDResult` instance.
    ci_cas : np.ndarray
        Reference CAS CI vector.
    max_ncsf : int, optional
        Maximum number of CSFs allowed.
    max_nlab : int, optional
        Maximum number of contracted labels allowed.
    max_ncsf_times_nint : int, optional
        Memory guard for the auxiliary buffer (ncsf * n_internal). Default is 5,000,000.

    Returns
    -------
    drt_mrci : DRT
        The uncontracted MRCISD DRT.
    ci_mrci : np.ndarray
        The reconstructed CI vector.
    """

    drt_mrci = getattr(ic_res, "drt_work", None)
    if drt_mrci is None:
        raise NotImplementedError(
            "Staged reconstruction requires ic-MRCISD backend='semi_direct' (missing drt_work)."
        )
    if not isinstance(drt_mrci, DRT):
        raise TypeError("ic_res.drt_work must be a DRT instance")

    ncsf = int(drt_mrci.ncsf)
    if ncsf < 1:
        raise ValueError("empty DRT")
    if ncsf > int(max_ncsf):
        raise NotImplementedError(
            f"staged reconstruction disabled for ncsf={ncsf} > max_ncsf={int(max_ncsf)}"
        )

    spaces = getattr(ic_res, "spaces", None)
    if spaces is None:
        raise TypeError("ic_res missing OrbitalSpaces")
    if getattr(spaces, "orbsym", None) is not None:
        raise NotImplementedError("staged reconstruction currently supports only symmetry=None (spaces.orbsym is None)")

    n_act = int(getattr(spaces, "n_internal"))
    n_virt = int(getattr(spaces, "n_external"))
    if n_act < 0 or n_virt < 0:
        raise ValueError("invalid orbital spaces")

    if int(ncsf * n_act) > int(max_ncsf_times_nint):
        # Fall back to the low-memory reference implementation.
        return reconstruct_uncontracted_ci_from_ic_mrcisd(
            ic_res,
            ci_cas=ci_cas,
            max_ncsf=int(max_ncsf),
            max_nlab=int(max_nlab),
        )

    drt_cas = build_drt(norb=n_act, nelec=int(drt_mrci.nelec), twos_target=int(drt_mrci.twos_target))
    _, psi0, _ref_idx = embed_cas_ci_into_mrcisd(drt_cas=drt_cas, drt_mrci=drt_mrci, ci_cas=ci_cas, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    n_singles = int(getattr(singles, "nlab"))
    n_doubles = int(getattr(doubles, "nlab"))
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    if int(1 + n_singles + n_doubles) > int(max_nlab):
        raise NotImplementedError(
            f"staged reconstruction disabled for nlab={int(1+n_singles+n_doubles)} > max_nlab={int(max_nlab)}"
        )

    from asuka.cuguga.oracle import _get_epq_action_cache  # noqa: PLC0415

    cache = _get_epq_action_cache(drt_mrci)

    ci_out = np.asarray(psi0, dtype=np.float64) * float(c[0])

    diag = getattr(ic_res, "diagnostics", {}) or {}
    try:
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.5)
    except Exception:
        allow_same_internal = True

    nI = int(n_act)
    if nI <= 0:
        return drt_mrci, np.ascontiguousarray(ci_out, dtype=np.float64)

    # Workspace buffers (reused across external groups).
    t_buf = np.empty((ncsf, nI), dtype=np.float64, order="F")  # columns = E_{a r} |psi0>
    w_buf = np.empty((ncsf, nI), dtype=np.float64, order="F")  # columns = Σ_r c_rs E_{a r} |psi0>
    tmp_epq = np.empty(ncsf, dtype=np.float64)
    tmp_vec = np.empty(ncsf, dtype=np.float64)
    ones_internal = np.ones(nI, dtype=np.float64)
    c_mat = np.zeros((nI, nI), dtype=np.float64)
    coeff_internal = np.zeros(nI, dtype=np.float64)

    def _build_t_for_a(a_ext: int) -> None:
        a_ext = int(a_ext)
        for r_int in range(nI):
            _apply_epq_inplace(drt_mrci, cache, psi0, p=a_ext, q=r_int, out=t_buf[:, r_int])

    # Singles block.
    if int(n_singles):
        cs = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
        if isinstance(singles, ICSingles):
            r_all = np.asarray(singles.r, dtype=np.int64).ravel()
            if np.any(r_all < 0) or np.any(r_all >= nI):
                raise ValueError("IC singles.r out of range for internal dimension")

            order = np.asarray(singles.a_group_order, dtype=np.int64)
            offsets = np.asarray(singles.a_group_offsets, dtype=np.int64)
            keys = np.asarray(singles.a_group_keys, dtype=np.int64)
            for g, a in enumerate(keys.tolist()):
                start = int(offsets[g])
                stop = int(offsets[g + 1])
                if start == stop:
                    continue
                idx = order[start:stop]
                _build_t_for_a(int(a))

                coeff_internal.fill(0.0)
                coeff_internal[r_all[idx]] = cs[idx]
                np.matmul(t_buf, coeff_internal, out=tmp_vec)
                ci_out += tmp_vec
        elif isinstance(singles, SCSingles):
            for i in range(n_singles):
                a = int(singles.a[i])
                _build_t_for_a(a)
                np.matmul(t_buf, ones_internal, out=tmp_vec)
                ci_out += float(cs[i]) * tmp_vec
        else:
            raise TypeError("unknown singles label type")

    # Doubles block.
    if int(n_doubles):
        cd = np.asarray(c[1 + n_singles :], dtype=np.float64)
        if isinstance(doubles, ICDoubles):
            r_all = np.asarray(doubles.r, dtype=np.int64).ravel()
            s_all = np.asarray(doubles.s, dtype=np.int64).ravel()
            if np.any(r_all < 0) or np.any(r_all >= nI) or np.any(s_all < 0) or np.any(s_all >= nI):
                raise ValueError("IC doubles internal indices out of range for internal dimension")

            order = np.asarray(doubles.ab_group_order, dtype=np.int64)
            offsets = np.asarray(doubles.ab_group_offsets, dtype=np.int64)
            keys = np.asarray(doubles.ab_group_keys, dtype=np.int64)

            cur_a = None
            for g in range(int(keys.shape[0])):
                a = int(keys[g, 0])
                b = int(keys[g, 1])
                start = int(offsets[g])
                stop = int(offsets[g + 1])
                if start == stop:
                    continue
                if cur_a != a:
                    _build_t_for_a(a)
                    cur_a = a

                idx = order[start:stop]
                c_mat.fill(0.0)
                c_mat[r_all[idx], s_all[idx]] = cd[idx]

                np.matmul(t_buf, c_mat, out=w_buf)
                cols = np.nonzero(np.abs(c_mat).sum(axis=0) > 0.0)[0].tolist()
                for s_int in cols:
                    _apply_epq_inplace(drt_mrci, cache, w_buf[:, s_int], p=b, q=int(s_int), out=tmp_epq)
                    ci_out += tmp_epq
        elif isinstance(doubles, SCDoubles):
            cur_a = None
            sum_a = np.zeros(ncsf, dtype=np.float64)
            prefix = np.zeros(ncsf, dtype=np.float64)
            w_in = np.empty(ncsf, dtype=np.float64)

            for i in range(n_doubles):
                a = int(doubles.a[i])
                b = int(doubles.b[i])
                coef = float(cd[i])
                if coef == 0.0:
                    continue

                if cur_a != a:
                    _build_t_for_a(a)
                    np.matmul(t_buf, ones_internal, out=sum_a)
                    cur_a = a

                acc = np.zeros(ncsf, dtype=np.float64)
                if a == b:
                    prefix.fill(0.0)
                    for s_int in range(nI):
                        prefix += t_buf[:, s_int]
                        if allow_same_internal:
                            _apply_epq_inplace(drt_mrci, cache, prefix, p=a, q=s_int, out=tmp_epq)
                        else:
                            np.subtract(prefix, t_buf[:, s_int], out=w_in)
                            _apply_epq_inplace(drt_mrci, cache, w_in, p=a, q=s_int, out=tmp_epq)
                        acc += tmp_epq
                else:
                    for s_int in range(nI):
                        if allow_same_internal:
                            _apply_epq_inplace(drt_mrci, cache, sum_a, p=b, q=s_int, out=tmp_epq)
                        else:
                            np.subtract(sum_a, t_buf[:, s_int], out=w_in)
                            _apply_epq_inplace(drt_mrci, cache, w_in, p=b, q=s_int, out=tmp_epq)
                        acc += tmp_epq

                ci_out += coef * acc
        else:
            raise TypeError("unknown doubles label type")

    return drt_mrci, np.ascontiguousarray(ci_out, dtype=np.float64)


def ic_mrcisd_reference_ci_rhs_from_residual(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
    reconstructed: tuple[DRT, np.ndarray] | None = None,
    max_ncsf: int = 200_000,
    max_nlab: int = 20_000,
    contract_nthreads: int = 1,
    contract_blas_nthreads: int | None = 1,
) -> np.ndarray:
    """Compute the CAS CI RHS (dE/dci) for contracted ic-MRCISD via reconstruction.

    Parameters
    ----------
    ic_res : Any
        ICMRCISDResult object.
    ci_cas : np.ndarray
        Reference CAS CI vector.
    h1e : np.ndarray
        One-electron integrals.
    eri : np.ndarray
        Two-electron integrals.
    reconstructed : tuple[DRT, np.ndarray] | None, optional
        Pre-computed uncontracted data.
    max_ncsf : int, optional
        Maximum CSFs (guard).
    max_nlab : int, optional
        Maximum labels (guard).
    contract_nthreads : int, optional
        Number of threads.
    contract_blas_nthreads : int | None, optional
        Number of BLAS threads.

    Returns
    -------
    rhs : np.ndarray
        Gradient w.r.t. CAS CI coefficients.
    """

    from asuka.contract import contract_h_csf_multi  # noqa: PLC0415
    from asuka.cuguga.oracle import _get_epq_action_cache  # noqa: PLC0415

    ci_cas_n = _normalize_ci(ci_cas)
    if reconstructed is None:
        drt_mrci, ci_mrci = reconstruct_uncontracted_ci_from_ic_mrcisd(
            ic_res,
            ci_cas=ci_cas_n,
            max_ncsf=int(max_ncsf),
            max_nlab=int(max_nlab),
        )
    else:
        drt_mrci, ci_mrci = reconstructed
        if not isinstance(drt_mrci, DRT):
            raise TypeError("reconstructed[0] must be a DRT")
        ci_mrci = np.asarray(ci_mrci, dtype=np.float64).ravel()
        if ci_mrci.size != int(drt_mrci.ncsf):
            raise ValueError("reconstructed[1] has wrong length for reconstructed DRT")

    h1e = np.asarray(h1e, dtype=np.float64)
    eri = np.asarray(eri, dtype=np.float64)
    if h1e.shape != (int(drt_mrci.norb), int(drt_mrci.norb)):
        raise ValueError("h1e has wrong shape for drt_mrci.norb")
    if eri.shape != (int(drt_mrci.norb), int(drt_mrci.norb), int(drt_mrci.norb), int(drt_mrci.norb)):
        raise ValueError("eri has wrong shape for drt_mrci.norb")

    # Residual in the uncontracted (restricted DRT) space: r = (H - E)|Psi>.
    e = float(getattr(ic_res, "e"))
    hy = contract_h_csf_multi(
        drt_mrci,
        h1e,
        eri,
        [ci_mrci],
        precompute_epq=False,
        nthreads=max(1, int(contract_nthreads)),
        blas_nthreads=contract_blas_nthreads,
    )[0]
    r = np.asarray(hy, dtype=np.float64).ravel() - e * np.asarray(ci_mrci, dtype=np.float64).ravel()

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    n_singles = int(getattr(singles, "nlab"))
    n_doubles = int(getattr(doubles, "nlab"))
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")

    cache = _get_epq_action_cache(drt_mrci)

    # g_full = O(c)† r in the uncontracted DRT basis.
    g_full = np.asarray(r, dtype=np.float64) * float(c[0])

    diag = getattr(ic_res, "diagnostics", {}) or {}
    try:
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.5)
    except Exception:
        allow_same_internal = True

    # Singles: O_mu = E_ar (FIC) or Σ_r E_ar (SC).  O_mu† uses de-excitations E_ra.
    off = 1
    if isinstance(singles, ICSingles):
        for i in range(n_singles):
            a = int(singles.a[i])
            r_int = int(singles.r[i])
            g_full += float(c[off + i]) * _apply_epq(drt_mrci, p=r_int, q=a, x=r, cache=cache)
    elif isinstance(singles, SCSingles):
        internal = np.asarray(getattr(ic_res, "spaces").internal, dtype=np.int32).ravel()
        for i in range(n_singles):
            a = int(singles.a[i])
            acc = np.zeros_like(g_full)
            for r_int in internal.tolist():
                acc += _apply_epq(drt_mrci, p=int(r_int), q=a, x=r, cache=cache)
            g_full += float(c[off + i]) * acc
    else:
        raise TypeError("unknown singles label type")

    # Doubles: O_mu = E_ar E_bs (FIC) or Σ_{r,s} E_ar E_bs (SC).
    off = 1 + n_singles
    if isinstance(doubles, ICDoubles):
        for i in range(n_doubles):
            a = int(doubles.a[i])
            r_int = int(doubles.r[i])
            b = int(doubles.b[i])
            s_int = int(doubles.s[i])
            tmp = _apply_epq(drt_mrci, p=r_int, q=a, x=r, cache=cache)
            vec = _apply_epq(drt_mrci, p=s_int, q=b, x=tmp, cache=cache)
            g_full += float(c[off + i]) * vec
    elif isinstance(doubles, SCDoubles):
        internal = np.asarray(getattr(ic_res, "spaces").internal, dtype=np.int32).ravel()
        internal_list = internal.tolist()
        for i in range(n_doubles):
            a = int(doubles.a[i])
            b = int(doubles.b[i])
            acc = np.zeros_like(g_full)
            if a == b:
                for ir, r_int in enumerate(internal_list):
                    start_s = ir if allow_same_internal else ir + 1
                    for s_int in internal_list[start_s:]:
                        tmp = _apply_epq(drt_mrci, p=int(r_int), q=a, x=r, cache=cache)
                        acc += _apply_epq(drt_mrci, p=int(s_int), q=b, x=tmp, cache=cache)
            else:
                for r_int in internal_list:
                    tmp = _apply_epq(drt_mrci, p=int(r_int), q=a, x=r, cache=cache)
                    for s_int in internal_list:
                        if (not allow_same_internal) and int(r_int) == int(s_int):
                            continue
                        acc += _apply_epq(drt_mrci, p=int(s_int), q=b, x=tmp, cache=cache)
            g_full += float(c[off + i]) * acc
    else:
        raise TypeError("unknown doubles label type")

    g_full = np.asarray(g_full * 2.0, dtype=np.float64)

    # Project the uncontracted gradient onto the CAS block and return it in CAS ordering.
    spaces = getattr(ic_res, "spaces")
    n_act = int(getattr(spaces, "n_internal"))
    n_virt = int(getattr(spaces, "n_external"))
    drt_cas = build_drt(norb=n_act, nelec=int(drt_mrci.nelec), twos_target=int(drt_mrci.twos_target))
    _ci0, _psi0, ref_idx = embed_cas_ci_into_mrcisd(drt_cas=drt_cas, drt_mrci=drt_mrci, ci_cas=ci_cas_n, n_virt=n_virt)
    return np.ascontiguousarray(g_full[np.asarray(ref_idx, dtype=np.int64)], dtype=np.float64)
