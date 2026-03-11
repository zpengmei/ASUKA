from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np

from asuka.mrci.ic_basis import ICDoubles, ICSingles, SCDoubles, SCSingles
from asuka.mrci.ic_rdm_common import cas_dm23_for_ic_res, infer_n_act_n_virt, require_internal_external_contiguous


DensityBackend = Literal["reconstruct", "direct"]


def build_rdm2_extint_extext_block_impl(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: Any | None = None,
    make_rdm12_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    pair_to_mat_builder: Callable[..., dict[tuple[int, int], np.ndarray]],
) -> np.ndarray:
    del dm1_int, dm3_int, dm4_ctx

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = infer_n_act_n_virt(ic_res)
        _dm1, dm2 = make_rdm12_fn(
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
    n_act, n_virt = infer_n_act_n_virt(ic_res)
    require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

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
        dm2_int, _dm3_int = cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
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

    pair_to_mat = pair_to_mat_builder(doubles, c_d, n_act=int(n_act), n_virt=int(n_virt))
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


def build_rdm2_intext_extext_block_impl(
    ic_res: Any,
    *,
    ci_cas: np.ndarray,
    backend: DensityBackend = "direct",
    rdm_backend: Literal["cuda", "cpu"] = "cpu",
    dm1_int: np.ndarray | None = None,
    dm2_int: np.ndarray | None = None,
    dm3_int: np.ndarray | None = None,
    dm4_ctx: Any | None = None,
    make_rdm12_fn: Callable[..., tuple[np.ndarray, np.ndarray]],
    pair_to_mat_builder: Callable[..., dict[tuple[int, int], np.ndarray]],
) -> np.ndarray:
    del dm1_int, dm3_int, dm4_ctx

    backend_s = str(backend).strip().lower()
    if backend_s != "direct":
        singles = getattr(ic_res, "singles")
        doubles = getattr(ic_res, "doubles")
        contraction_s = "fic"
        if isinstance(singles, SCSingles) and isinstance(doubles, SCDoubles):
            contraction_s = "sc"
        elif not (isinstance(singles, ICSingles) and isinstance(doubles, ICDoubles)):
            raise TypeError("unsupported ic-MRCISD label types for Phase-3 dm2 blocks")

        n_act, _n_virt = infer_n_act_n_virt(ic_res)
        _dm1, dm2 = make_rdm12_fn(
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
    n_act, n_virt = infer_n_act_n_virt(ic_res)
    require_internal_external_contiguous(spaces, n_act=n_act, n_virt=n_virt)

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
        dm2_int, _dm3_int = cas_dm23_for_ic_res(ic_res, ci_cas=ci_cas, n_act=n_act)
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

    pair_to_mat = pair_to_mat_builder(doubles, c_d, n_act=int(n_act), n_virt=int(n_virt))
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


__all__ = [
    "build_rdm2_extint_extext_block_impl",
    "build_rdm2_intext_extext_block_impl",
]
