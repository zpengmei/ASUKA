from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from asuka.mrci.ic_basis import ICDoubles, ICSingles, SCDoubles, SCSingles
from asuka.mrci.ic_rdm_common import (
    cas_dm123_for_ic_res,
    infer_n_act_n_virt,
    require_internal_external_contiguous,
)


DensityBackend = Literal["reconstruct", "direct"]


def build_rdm1_internal_block_impl(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    make_rdm12_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    singles = getattr(ic_res, "singles")
    doubles = getattr(ic_res, "doubles")
    is_fic = isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)
    is_sc = isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles)
    if not is_fic and not is_sc:
        raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        contraction_s = "fic" if is_fic else "sc"
        n_act, _n_virt = infer_n_act_n_virt(ic_res)
        dm1, _dm2 = make_rdm12_fn(
            ic_res,
            ci_cas=ci_cas,
            contraction=contraction_s,  # type: ignore[arg-type]
            backend=backend,
            rdm_backend=rdm_backend,
        )
        return np.asarray(dm1[:n_act, :n_act], dtype=np.float64)

    spaces = getattr(ic_res, "spaces")
    n_act, n_virt = infer_n_act_n_virt(ic_res)
    require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None or dm3_int is None:
        dm1_int, dm2_int, dm3_int = cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
    else:
        dm1_int = np.asarray(dm1_int, dtype=np.float64)
        dm2_int = np.asarray(dm2_int, dtype=np.float64)
        dm3_int = np.asarray(dm3_int, dtype=np.float64)

    out = (float(c0) * float(c0)) * np.asarray(dm1_int, dtype=np.float64)

    if is_sc:
        diag = getattr(ic_res, "diagnostics", {}) or {}
        allow_same_internal = bool(float(diag.get("allow_same_internal", 1.0)) > 0.0)

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


def build_rdm1_external_internal_block_impl(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    make_rdm12_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

        n_act, _n_virt = infer_n_act_n_virt(ic_res)
        dm1, _dm2 = make_rdm12_fn(
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
    n_act, n_virt = infer_n_act_n_virt(ic_res)
    require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c0 = float(c[0])
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None:
        dm1_int, dm2_int, _dm3_int = cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
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

        g_int = dm1_int.sum(axis=0)
        out = float(c0) * cs[:, None] * g_int[None, :]

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

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    out = float(c0) * (cs @ dm1_int)

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


def build_rdm1_external_external_block_impl(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    make_rdm12_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm1 blocks")

        n_act, _n_virt = infer_n_act_n_virt(ic_res)
        dm1, _dm2 = make_rdm12_fn(
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
    n_act, n_virt = infer_n_act_n_virt(ic_res)
    require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

    c = np.asarray(getattr(ic_res, "c"), dtype=np.float64).ravel()
    n_singles = int(singles.nlab)
    n_doubles = int(doubles.nlab)
    if int(c.size) != 1 + n_singles + n_doubles:
        raise ValueError("ic_res.c has wrong length for (ref, singles, doubles)")
    c_s = np.asarray(c[1 : 1 + n_singles], dtype=np.float64)
    c_d = np.asarray(c[1 + n_singles :], dtype=np.float64)

    if dm1_int is None or dm2_int is None:
        dm1_int, dm2_int, _dm3_int = cas_dm123_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
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

        s1 = float(np.sum(dm1_int))
        out = (cs[:, None] * cs[None, :]) * s1

        c_d = np.asarray(c_d, dtype=np.float64)
        if not np.any(c_d):
            return np.asarray(out, dtype=np.float64, order="C")

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

    cs = np.zeros((int(n_virt), int(n_act)), dtype=np.float64)
    a_all = np.asarray(singles.a, dtype=np.int64).ravel()
    r_all = np.asarray(singles.r, dtype=np.int64).ravel()
    for idx in range(n_singles):
        a = int(a_all[idx]) - int(n_act)
        r = int(r_all[idx])
        if not (0 <= a < int(n_virt)) or not (0 <= r < int(n_act)):
            raise ValueError("singles labels out of range for contiguous orbital convention")
        cs[a, r] = float(c_s[idx])

    out = cs @ dm1_int @ cs.T

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


__all__ = [
    "build_rdm1_internal_block_impl",
    "build_rdm1_external_internal_block_impl",
    "build_rdm1_external_external_block_impl",
]
