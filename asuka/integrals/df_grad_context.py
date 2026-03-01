from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from asuka.cueri.cart import ncart
from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu
from asuka.integrals.cart2sph import compute_sph_layout_from_cart_basis
from asuka.integrals.cueri_df_cpu import _build_df_combined_basis_and_shell_pairs
from asuka.integrals.df_adjoint import chol_lower_adjoint, df_whiten_adjoint, df_whiten_adjoint_Qmn
from asuka.integrals.int1e_cart import nao_cart_from_basis, shell_to_atom_map

Backend = Literal["cpu", "cuda"]


def _normalize_fused_precision(value: str | None) -> str:
    mode = str(value or "fp64").strip().lower()
    if mode in ("fp32_acc64", "fp32-acc64", "fp32acc64", "mixed_fp32_acc64"):
        return "fp32_acc64"
    if mode in ("tf32",):
        return "tf32"
    return "fp64"


def _normalize_sym_kernel_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode in ("1", "true", "yes", "on", "enable", "enabled", "force"):
        return "on"
    if mode in ("0", "false", "no", "off", "disable", "disabled"):
        return "off"
    return "auto"


def _prefer_sym_kernel(arr_nbytes: int, *, nchunks: int = 4) -> bool:
    mode = _normalize_sym_kernel_mode(os.environ.get("ASUKA_DF_SYM_KERNEL"))
    if mode == "on":
        return True
    if mode == "off":
        return False
    try:
        import cupy as cp  # noqa: PLC0415

        free_b, _ = cp.cuda.runtime.memGetInfo()
    except Exception:
        return False
    tmp_b = max(1, int(arr_nbytes) // max(1, int(nchunks)))
    margin_b = max(int(768 * 1024**2), tmp_b // 4)
    return int(free_b) < int(tmp_b + margin_b)


def _normalize_df_3c_sched_mode(value: str | None) -> str:
    mode = str(value or "on").strip().lower()
    if mode in ("off", "0", "false", "no", "disable", "disabled", "legacy"):
        return "off"
    if mode in ("on", "1", "true", "yes", "enable", "enabled", "force"):
        return "on"
    return "auto"


def _normalize_df_3c_ab_tile(value: str | None) -> int:
    try:
        tile = int(str(value or "4").strip())
    except Exception:
        tile = 4
    return max(1, min(16, int(tile)))


def _parse_bucket_edges(value: str | None, *, default: Sequence[int]) -> tuple[int, ...]:
    raw = str(value or "").strip()
    if not raw:
        return tuple(int(v) for v in default)
    vals: list[int] = []
    for tok in raw.split(","):
        tok_s = tok.strip()
        if not tok_s:
            continue
        try:
            x = int(tok_s)
        except Exception:
            continue
        if x > 0:
            vals.append(int(x))
    if not vals:
        return tuple(int(v) for v in default)
    return tuple(sorted(set(vals)))


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    val = os.environ.get(str(name))
    if val is None:
        return bool(default)
    sval = str(val).strip().lower()
    if sval in ("1", "true", "yes", "on", "enable", "enabled"):
        return True
    if sval in ("0", "false", "no", "off", "disable", "disabled"):
        return False
    return bool(default)


def _npair_cv(arr: np.ndarray) -> float:
    x = np.asarray(arr, dtype=np.float64).ravel()
    if int(x.size) <= 1:
        return 0.0
    mean = float(np.mean(x))
    if mean <= 0.0:
        return 0.0
    return float(np.std(x) / mean)


def _choose_df_3c_threads(score: float) -> int:
    s = float(score)
    if s <= 32.0:
        return 128
    if s <= 96.0:
        return 192
    return 256


def _bucketize_indices_by_npair(indices: np.ndarray, npair: np.ndarray, edges: Sequence[int]) -> list[tuple[int, np.ndarray, np.ndarray]]:
    idx = np.asarray(indices, dtype=np.int32).ravel()
    npair_i = np.asarray(npair, dtype=np.int32).ravel()
    if int(idx.size) != int(npair_i.size):
        raise ValueError("indices/npair size mismatch")
    if int(idx.size) == 0:
        return []
    if len(edges) == 0:
        return [(0, idx, npair_i)]
    bucket = np.searchsorted(np.asarray(edges, dtype=np.int32), npair_i, side="right")
    groups: list[tuple[int, np.ndarray, np.ndarray]] = []
    for bid in sorted(set(bucket.tolist())):
        mask = bucket == bid
        groups.append((int(bid), idx[mask].astype(np.int32, copy=False), npair_i[mask].astype(np.int32, copy=False)))
    return groups


def _build_df_3c_launch_jobs(
    *,
    spAB_by_lab: dict[tuple[int, int], np.ndarray],
    spCD_by_l: dict[int, np.ndarray],
    sp_npair: np.ndarray,
    sched_mode: str,
    ab_edges: Sequence[int],
    cd_edges: Sequence[int],
    ab_tile: int,
    small_job_blocks: int = 4096,
) -> list[dict[str, Any]]:
    mode = _normalize_df_3c_sched_mode(sched_mode)
    ab_edges_n = tuple(int(v) for v in ab_edges)
    cd_edges_n = tuple(int(v) for v in cd_edges)
    ab_tile_n = _normalize_df_3c_ab_tile(str(ab_tile))
    npair_all = np.asarray(sp_npair, dtype=np.int32).ravel()

    def _finalize_job(job: dict[str, Any]) -> None:
        ab_np = np.asarray(job["_npair_ab"], dtype=np.int32).ravel()
        cd_np = np.asarray(job["_npair_cd"], dtype=np.int32).ravel()
        ab_idx = np.asarray(job["spAB_host"], dtype=np.int32).ravel()
        cd_idx = np.asarray(job["spCD_host"], dtype=np.int32).ravel()
        n_ab = int(ab_idx.size)
        n_cd = int(cd_idx.size)
        blocks = int(n_ab * n_cd)
        ab_sum = int(np.sum(ab_np, dtype=np.int64))
        cd_sum = int(np.sum(cd_np, dtype=np.int64))
        work_est = int(ab_sum * cd_sum)
        score = float(np.mean(ab_np, dtype=np.float64)) * float(np.mean(cd_np, dtype=np.float64))
        if mode == "off":
            threads = 256
            tile = 1
            use_abtile = False
        else:
            threads = _choose_df_3c_threads(score)
            tile = max(1, min(int(ab_tile_n), n_cd))
            if score >= 160.0:
                tile = max(1, min(tile, 4))
            elif score >= 96.0:
                tile = max(1, min(tile, 6))
            use_abtile = bool(tile > 1 and blocks >= 2048 and n_ab >= 8 and n_cd >= 2)
        job["n_spAB"] = int(n_ab)
        job["n_spCD"] = int(n_cd)
        job["blocks"] = int(blocks)
        job["work_est"] = int(work_est)
        job["score"] = float(score)
        job["threads"] = int(threads)
        job["cd_tile"] = int(tile)
        job["use_abtile"] = bool(use_abtile)

    def _merge_small_jobs(class_jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if int(small_job_blocks) <= 0 or len(class_jobs) <= 1:
            return class_jobs
        for job in class_jobs:
            _finalize_job(job)
        guard = 0
        while len(class_jobs) > 1:
            guard += 1
            if guard > 8192:
                break
            i_small = -1
            best_blocks = 10**18
            for i, job in enumerate(class_jobs):
                blk = int(job.get("blocks", 0))
                if blk < int(small_job_blocks) and blk < best_blocks:
                    i_small = i
                    best_blocks = blk
            if i_small < 0:
                break

            src = class_jobs[i_small]
            best_j = -1
            best_dist = float("inf")
            for j, dst in enumerate(class_jobs):
                if j == i_small:
                    continue
                same_ab = int(src.get("ab_bucket", -1)) == int(dst.get("ab_bucket", -2))
                same_cd = int(src.get("cd_bucket", -1)) == int(dst.get("cd_bucket", -2))
                if not (same_ab or same_cd):
                    continue
                dist = abs(np.log1p(float(src.get("work_est", 1))) - np.log1p(float(dst.get("work_est", 1))))
                if dist < best_dist:
                    best_dist = dist
                    best_j = j
            if best_j < 0:
                break

            dst = class_jobs[best_j]
            same_ab = int(src.get("ab_bucket", -1)) == int(dst.get("ab_bucket", -2))
            if same_ab:
                cd_idx = np.concatenate(
                    [
                        np.asarray(dst["spCD_host"], dtype=np.int32).ravel(),
                        np.asarray(src["spCD_host"], dtype=np.int32).ravel(),
                    ]
                ).astype(np.int32, copy=False)
                cd_np = np.concatenate(
                    [
                        np.asarray(dst["_npair_cd"], dtype=np.int32).ravel(),
                        np.asarray(src["_npair_cd"], dtype=np.int32).ravel(),
                    ]
                ).astype(np.int32, copy=False)
                order = np.argsort(cd_idx, kind="stable")
                dst["spCD_host"] = cd_idx[order]
                dst["_npair_cd"] = cd_np[order]
                dst["cd_bucket"] = -1
            else:
                ab_idx = np.concatenate(
                    [
                        np.asarray(dst["spAB_host"], dtype=np.int32).ravel(),
                        np.asarray(src["spAB_host"], dtype=np.int32).ravel(),
                    ]
                ).astype(np.int32, copy=False)
                ab_np = np.concatenate(
                    [
                        np.asarray(dst["_npair_ab"], dtype=np.int32).ravel(),
                        np.asarray(src["_npair_ab"], dtype=np.int32).ravel(),
                    ]
                ).astype(np.int32, copy=False)
                order = np.argsort(ab_idx, kind="stable")
                dst["spAB_host"] = ab_idx[order]
                dst["_npair_ab"] = ab_np[order]
                dst["ab_bucket"] = -1

            _finalize_job(dst)
            class_jobs.pop(i_small)
        return class_jobs

    jobs: list[dict[str, Any]] = []
    for (la, lb), ab_idx_raw in sorted(spAB_by_lab.items(), key=lambda kv: (int(kv[0][0]), int(kv[0][1]))):
        ab_idx = np.asarray(ab_idx_raw, dtype=np.int32).ravel()
        if int(ab_idx.size) == 0:
            continue
        ab_np = npair_all[ab_idx].astype(np.int32, copy=False)
        cv_ab = _npair_cv(ab_np)

        for lq, cd_idx_raw in sorted(spCD_by_l.items(), key=lambda kv: int(kv[0])):
            cd_idx = np.asarray(cd_idx_raw, dtype=np.int32).ravel()
            if int(cd_idx.size) == 0:
                continue
            cd_np = npair_all[cd_idx].astype(np.int32, copy=False)
            cv_cd = _npair_cv(cd_np)
            class_blocks = int(ab_idx.size) * int(cd_idx.size)
            do_split = False
            if mode == "on":
                do_split = True
            elif mode == "auto":
                do_split = bool(class_blocks >= 65536 and (cv_ab >= 0.30 or cv_cd >= 0.30))

            if do_split:
                ab_groups = _bucketize_indices_by_npair(ab_idx, ab_np, ab_edges_n)
                cd_groups = _bucketize_indices_by_npair(cd_idx, cd_np, cd_edges_n)
            else:
                ab_groups = [(0, ab_idx, ab_np)]
                cd_groups = [(0, cd_idx, cd_np)]

            class_jobs: list[dict[str, Any]] = []
            for ab_bid, ab_g, ab_np_g in ab_groups:
                if int(ab_g.size) == 0:
                    continue
                for cd_bid, cd_g, cd_np_g in cd_groups:
                    if int(cd_g.size) == 0:
                        continue
                    job = {
                        "la": int(la),
                        "lb": int(lb),
                        "lq": int(lq),
                        "spAB_host": np.asarray(ab_g, dtype=np.int32),
                        "spCD_host": np.asarray(cd_g, dtype=np.int32),
                        "_npair_ab": np.asarray(ab_np_g, dtype=np.int32),
                        "_npair_cd": np.asarray(cd_np_g, dtype=np.int32),
                        "ab_bucket": int(ab_bid),
                        "cd_bucket": int(cd_bid),
                        "split": bool(do_split),
                        "cv_ab": float(cv_ab),
                        "cv_cd": float(cv_cd),
                    }
                    _finalize_job(job)
                    class_jobs.append(job)

            class_jobs = _merge_small_jobs(class_jobs)
            jobs.extend(class_jobs)

    jobs.sort(
        key=lambda job: (
            -int(job.get("work_est", 0)),
            -int(job.get("blocks", 0)),
            int(job.get("la", 0)),
            int(job.get("lb", 0)),
            int(job.get("lq", 0)),
        )
    )
    for i, job in enumerate(jobs):
        job["job_id"] = int(i)
        job.pop("_npair_ab", None)
        job.pop("_npair_cd", None)
    return jobs


_AXPY_KERNELS: dict[str, Any] = {}


def _axpy_inplace_device(src: Any, dst: Any) -> None:
    """Fused CUDA axpy kernel: dst += src (elementwise)."""
    import cupy as cp  # noqa: PLC0415

    src_arr = cp.asarray(src)
    dst_arr = cp.asarray(dst)
    if src_arr.size != dst_arr.size:
        raise ValueError("axpy requires src/dst with matching number of elements")
    if cp.dtype(src_arr.dtype) != cp.dtype(dst_arr.dtype):
        raise ValueError("axpy requires matching src/dst dtypes")
    dt = cp.dtype(dst_arr.dtype)
    key = str(dt)
    k = _AXPY_KERNELS.get(key)
    if k is None:
        if dt == cp.dtype(cp.float32):
            k = cp.ElementwiseKernel(
                "float32 x",
                "float32 y",
                "y += x",
                "asuka_df_axpy_f32",
            )
        elif dt == cp.dtype(cp.float64):
            k = cp.ElementwiseKernel(
                "float64 x",
                "float64 y",
                "y += x",
                "asuka_df_axpy_f64",
            )
        else:
            raise ValueError("axpy supports only float32/float64")
        _AXPY_KERNELS[key] = k
    k(src_arr.ravel(), dst_arr.ravel())


def _symmetrize_mnQ_inplace(arr: Any, xp: Any, *, nchunks: int = 4) -> Any:
    """In-place symmetrization over AO indices for ``arr[m,n,Q]``.

    Computes ``arr = 0.5 * (arr + arr^T)`` where the transpose swaps only the
    AO axes (m,n). Chunking along Q limits temporary memory to roughly
    ``sizeof(arr)/nchunks``.
    """
    if int(arr.ndim) != 3:
        raise ValueError("arr must have shape (nao, nao, naux)")
    nao0, nao1, naux = map(int, arr.shape)
    if nao0 != nao1:
        raise ValueError("arr must have shape (nao, nao, naux)")
    _chunk = max(1, int(naux) // max(1, int(nchunks)))
    for _q0 in range(0, int(naux), _chunk):
        _q1 = min(_q0 + _chunk, int(naux))
        _blk = arr[:, :, _q0:_q1].copy()
        arr[:, :, _q0:_q1] += xp.transpose(_blk, (1, 0, 2))
        del _blk
    arr *= 0.5
    return arr


@dataclass
class DFGradContractionContext:
    backend: Backend
    ao_basis: Any
    aux_basis: Any
    atom_coords_bohr: np.ndarray
    df_threads: int

    natm: int
    nao: int
    naux: int

    ao_shell_atom: np.ndarray
    aux_shell_atom: np.ndarray
    aux_shell_l: np.ndarray

    basis_all: Any
    sp_all: Any
    pt_all: Any
    nsp_ao: int
    n_shell_aux: int
    aux_sp0: int

    shell_cxyz_all: np.ndarray
    shell_l_all: np.ndarray
    shell_prim_start_all: np.ndarray
    shell_nprim_all: np.ndarray
    shell_ao_start_all: np.ndarray
    prim_exp_all: np.ndarray

    sp_A_all: np.ndarray
    sp_B_all: np.ndarray
    sp_pair_start_all: np.ndarray
    sp_npair_all: np.ndarray

    pair_eta_all: np.ndarray
    pair_Px_all: np.ndarray
    pair_Py_all: np.ndarray
    pair_Pz_all: np.ndarray
    pair_cK_all: np.ndarray

    shells_by_l: dict[int, np.ndarray]
    spCD_by_l: dict[int, np.ndarray]
    metric_batches: list[tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]]

    L_metric: Any
    cpu: dict[str, Any] | None = None
    cuda: dict[str, Any] | None = None

    @staticmethod
    def build(
        ao_basis: Any,
        aux_basis: Any,
        *,
        atom_coords_bohr: np.ndarray,
        backend: Backend,
        df_threads: int = 0,
        L_chol: Any | None = None,
    ) -> "DFGradContractionContext":
        backend_s = str(backend).strip().lower()
        if backend_s not in ("cpu", "cuda"):
            raise ValueError("backend must be 'cpu' or 'cuda'")

        atom_coords = np.asarray(atom_coords_bohr, dtype=np.float64)
        if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
            raise ValueError("atom_coords_bohr must have shape (natm, 3)")
        natm = int(atom_coords.shape[0])
        if natm <= 0:
            raise ValueError("atom_coords_bohr must have natm > 0")

        ao_shell_atom = np.asarray(shell_to_atom_map(ao_basis, atom_coords_bohr=atom_coords), dtype=np.int32)
        aux_shell_atom = np.asarray(shell_to_atom_map(aux_basis, atom_coords_bohr=atom_coords), dtype=np.int32)

        basis_all, sp_all, nsp_ao, _n_shell_ao, n_shell_aux = _build_df_combined_basis_and_shell_pairs(ao_basis, aux_basis)
        pt_all = build_pair_tables_cpu(basis_all, sp_all, threads=int(df_threads), profile=None)
        aux_sp0 = int(nsp_ao)

        nao = int(nao_cart_from_basis(ao_basis))
        naux = int(nao_cart_from_basis(aux_basis))

        shell_cxyz_all = np.asarray(basis_all.shell_cxyz, dtype=np.float64, order="C")
        shell_l_all = np.asarray(basis_all.shell_l, dtype=np.int32, order="C")
        shell_prim_start_all = np.asarray(basis_all.shell_prim_start, dtype=np.int32, order="C")
        shell_nprim_all = np.asarray(basis_all.shell_nprim, dtype=np.int32, order="C")
        shell_ao_start_all = np.asarray(basis_all.shell_ao_start, dtype=np.int32, order="C")
        prim_exp_all = np.asarray(basis_all.prim_exp, dtype=np.float64, order="C")

        sp_A_all = np.asarray(sp_all.sp_A, dtype=np.int32, order="C")
        sp_B_all = np.asarray(sp_all.sp_B, dtype=np.int32, order="C")
        sp_pair_start_all = np.asarray(sp_all.sp_pair_start, dtype=np.int32, order="C")
        sp_npair_all = np.asarray(sp_all.sp_npair, dtype=np.int32, order="C")

        pair_eta_all = np.asarray(pt_all.pair_eta, dtype=np.float64, order="C")
        pair_Px_all = np.asarray(pt_all.pair_Px, dtype=np.float64, order="C")
        pair_Py_all = np.asarray(pt_all.pair_Py, dtype=np.float64, order="C")
        pair_Pz_all = np.asarray(pt_all.pair_Pz, dtype=np.float64, order="C")
        pair_cK_all = np.asarray(pt_all.pair_cK, dtype=np.float64, order="C")

        aux_shell_l = np.asarray(aux_basis.shell_l, dtype=np.int32, order="C").ravel()
        by_l: dict[int, list[int]] = {}
        for sh in range(int(n_shell_aux)):
            by_l.setdefault(int(aux_shell_l[sh]), []).append(int(sh))

        shells_by_l: dict[int, np.ndarray] = {}
        spCD_by_l: dict[int, np.ndarray] = {}
        for lq, q_shells in by_l.items():
            q_arr = np.asarray(q_shells, dtype=np.int32)
            shells_by_l[int(lq)] = q_arr
            spCD_by_l[int(lq)] = (aux_sp0 + q_arr).astype(np.int32, copy=False)

        metric_batches: list[tuple[int, int, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
        for psh in range(int(n_shell_aux)):
            lp = int(aux_shell_l[int(psh)])
            atomP = int(aux_shell_atom[int(psh)])
            spAB = int(aux_sp0 + int(psh))
            for lq, q_shells in shells_by_l.items():
                q_arr = np.asarray(q_shells, dtype=np.int32)
                q_list = q_arr[q_arr <= np.int32(int(psh))]
                if int(q_list.size) == 0:
                    continue
                spCD_batch = (aux_sp0 + q_list).astype(np.int32, copy=False)
                atomQ = np.asarray(aux_shell_atom[q_list], dtype=np.int32, order="C")
                fac = np.full((int(q_list.size),), 2.0, dtype=np.float64)
                fac[q_list == np.int32(int(psh))] = 1.0
                metric_batches.append((int(spAB), int(lp), int(atomP), int(lq), spCD_batch, atomQ, fac))

        cpu_ctx: dict[str, Any] | None = None
        if backend_s == "cpu":
            try:
                from asuka.cueri import _eri_rys_cpu as _ext  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError(
                    "CPU ERI extension is required for analytic DF gradient contraction on backend='cpu'"
                ) from e

            eri_batch = getattr(_ext, "eri_rys_tile_cart_sp_batch_cy", None)
            fn_3c = getattr(_ext, "df_int3c2e_deriv_contracted_cart_sp_batch_cy", None)
            fn_2c = getattr(_ext, "df_metric_2c2e_deriv_contracted_cart_sp_batch_cy", None)
            if eri_batch is None or fn_3c is None or fn_2c is None:  # pragma: no cover
                raise RuntimeError("CPU ERI extension is missing DF derivative entry points; rebuild the extension")

            aux_shell_ao_start = np.asarray(aux_basis.shell_ao_start, dtype=np.int32, order="C").ravel()
            V = np.zeros((naux, naux), dtype=np.float64)
            for psh in range(int(n_shell_aux)):
                lp = int(aux_shell_l[int(psh)])
                nP = int(ncart(lp))
                p0 = int(aux_shell_ao_start[int(psh)])
                spAB = int(aux_sp0 + psh)

                for lq, q_shells in shells_by_l.items():
                    nQ = int(ncart(int(lq)))
                    q_list = [int(q) for q in q_shells if int(q) <= int(psh)]
                    if not q_list:
                        continue
                    spCD_sub = (aux_sp0 + np.asarray(q_list, dtype=np.int32)).astype(np.int32, copy=False)
                    tiles = eri_batch(
                        shell_cxyz_all,
                        shell_l_all,
                        sp_A_all,
                        sp_B_all,
                        sp_pair_start_all,
                        sp_npair_all,
                        pair_eta_all,
                        pair_Px_all,
                        pair_Py_all,
                        pair_Pz_all,
                        pair_cK_all,
                        int(spAB),
                        spCD_sub,
                        int(df_threads),
                    )
                    for t, qsh in enumerate(q_list):
                        q0 = int(aux_shell_ao_start[int(qsh)])
                        block = np.asarray(tiles[int(t)], dtype=np.float64, order="C").reshape((nP, nQ))
                        V[p0 : p0 + nP, q0 : q0 + nQ] = block
                        if int(qsh) != int(psh):
                            V[q0 : q0 + nQ, p0 : p0 + nP] = block.T

            V = 0.5 * (V + V.T)
            L_metric = np.linalg.cholesky(V)
            cpu_ctx = {"fn_3c": fn_3c, "fn_2c": fn_2c}
        else:
            try:
                import cupy as cp  # noqa: PLC0415
            except Exception as e:  # pragma: no cover
                raise RuntimeError("backend='cuda' requires CuPy") from e

            from asuka.cueri import df as cueri_df  # noqa: PLC0415

            if L_chol is not None:
                L_metric = cp.ascontiguousarray(cp.asarray(L_chol, dtype=cp.float64))
            else:
                V = cueri_df.metric_2c2e_basis(aux_basis, stream=None, backend="gpu_rys", mode="warp", threads=256)
                _v_diag = cp.diag(V)
                _v_shift = max(float(cp.max(cp.abs(_v_diag))) * 1e-14, 1e-12)
                V[cp.diag_indices_from(V)] += _v_shift
                L_metric = cp.linalg.cholesky(V)

        ctx = DFGradContractionContext(
            backend=backend_s,  # type: ignore[arg-type]
            ao_basis=ao_basis,
            aux_basis=aux_basis,
            atom_coords_bohr=atom_coords,
            df_threads=int(df_threads),
            natm=natm,
            nao=int(nao),
            naux=int(naux),
            ao_shell_atom=ao_shell_atom,
            aux_shell_atom=aux_shell_atom,
            aux_shell_l=aux_shell_l,
            basis_all=basis_all,
            sp_all=sp_all,
            pt_all=pt_all,
            nsp_ao=int(nsp_ao),
            n_shell_aux=int(n_shell_aux),
            aux_sp0=int(aux_sp0),
            shell_cxyz_all=shell_cxyz_all,
            shell_l_all=shell_l_all,
            shell_prim_start_all=shell_prim_start_all,
            shell_nprim_all=shell_nprim_all,
            shell_ao_start_all=shell_ao_start_all,
            prim_exp_all=prim_exp_all,
            sp_A_all=sp_A_all,
            sp_B_all=sp_B_all,
            sp_pair_start_all=sp_pair_start_all,
            sp_npair_all=sp_npair_all,
            pair_eta_all=pair_eta_all,
            pair_Px_all=pair_Px_all,
            pair_Py_all=pair_Py_all,
            pair_Pz_all=pair_Pz_all,
            pair_cK_all=pair_cK_all,
            shells_by_l=shells_by_l,
            spCD_by_l=spCD_by_l,
            metric_batches=metric_batches,
            L_metric=L_metric,
            cpu=cpu_ctx,
            cuda=None,
        )
        if backend_s == "cuda":
            ctx._init_cuda()
        return ctx

    def _init_cuda(self) -> None:
        import cupy as cp  # noqa: PLC0415

        try:
            from asuka.cueri import _cueri_cuda_ext as _ext_cuda  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "cuERI CUDA extension is required for analytic DF gradient contraction; "
                "build via `python -m asuka.cueri.build_cuda_ext`"
            ) from e

        spCD_by_l_host = {
            int(lq): np.asarray(spCD_batch, dtype=np.int32).ravel()
            for lq, spCD_batch in self.spCD_by_l.items()
        }
        spCD_by_l_dev = {
            int(lq): cp.ascontiguousarray(cp.asarray(spCD_batch, dtype=cp.int32))
            for lq, spCD_batch in spCD_by_l_host.items()
        }
        atomC_by_l_dev = {
            int(lq): cp.ascontiguousarray(cp.asarray(self.aux_shell_atom[np.asarray(q_shells, dtype=np.int32)], dtype=cp.int32))
            for lq, q_shells in self.shells_by_l.items()
        }
        metric_batches_dev: list[tuple[int, int, int, int, Any, Any, Any]] = []
        for spAB, lp, atomP, lq, spCD_batch, atomQ, fac in self.metric_batches:
            metric_batches_dev.append(
                (
                    int(spAB),
                    int(lp),
                    int(atomP),
                    int(lq),
                    cp.ascontiguousarray(cp.asarray(spCD_batch, dtype=cp.int32)),
                    cp.ascontiguousarray(cp.asarray(atomQ, dtype=cp.int32)),
                    cp.ascontiguousarray(cp.asarray(fac, dtype=cp.float64)),
                )
            )

        # Combined AO+aux shell→atom map (AO shells first, then aux shells).
        # shellA/B from sp_A/sp_B for AO pairs index into [0, n_ao_shells).
        # shellC from sp_A for aux pairs indexes into [n_ao_shells, n_ao_shells+n_aux_shells).
        shell_atom_all = np.concatenate([
            np.asarray(self.ao_shell_atom, dtype=np.int32),
            np.asarray(self.aux_shell_atom, dtype=np.int32),
        ])
        shell_atom_dev = cp.ascontiguousarray(cp.asarray(shell_atom_all, dtype=cp.int32))

        # Pre-group AO shell pairs by (la, lb) angular momentum class for batched kernel.
        shell_l_np = np.asarray(self.shell_l_all, dtype=np.int32).ravel()
        sp_A_np = np.asarray(self.sp_A_all, dtype=np.int32).ravel()
        sp_B_np = np.asarray(self.sp_B_all, dtype=np.int32).ravel()
        spAB_by_lab: dict[tuple[int, int], list[int]] = {}
        for spAB_i in range(int(self.nsp_ao)):
            shA = int(sp_A_np[spAB_i])
            shB = int(sp_B_np[spAB_i])
            key = (int(shell_l_np[shA]), int(shell_l_np[shB]))
            if key not in spAB_by_lab:
                spAB_by_lab[key] = []
            spAB_by_lab[key].append(spAB_i)
        spAB_by_lab_host = {
            (int(lab[0]), int(lab[1])): np.asarray(indices, dtype=np.int32).ravel()
            for lab, indices in spAB_by_lab.items()
        }
        spAB_by_lab_dev = {
            lab: cp.ascontiguousarray(cp.asarray(indices, dtype=cp.int32))
            for lab, indices in spAB_by_lab_host.items()
        }

        sched_mode = _normalize_df_3c_sched_mode(os.environ.get("ASUKA_DF_3C_SCHED_MODE"))
        ab_edges = _parse_bucket_edges(os.environ.get("ASUKA_DF_3C_BUCKET_AB"), default=(4, 8, 16, 24))
        cd_edges = _parse_bucket_edges(os.environ.get("ASUKA_DF_3C_BUCKET_CD"), default=(4, 8, 16, 24))
        ab_tile = _normalize_df_3c_ab_tile(os.environ.get("ASUKA_DF_3C_AB_TILE"))
        jobs_3c_host = _build_df_3c_launch_jobs(
            spAB_by_lab=spAB_by_lab_host,
            spCD_by_l=spCD_by_l_host,
            sp_npair=self.sp_npair_all,
            sched_mode=sched_mode,
            ab_edges=ab_edges,
            cd_edges=cd_edges,
            ab_tile=ab_tile,
        )
        jobs_3c_dev: list[dict[str, Any]] = []
        for job in jobs_3c_host:
            spab_host = np.asarray(job["spAB_host"], dtype=np.int32).ravel()
            spcd_host = np.asarray(job["spCD_host"], dtype=np.int32).ravel()
            jobs_3c_dev.append(
                {
                    "job_id": int(job.get("job_id", len(jobs_3c_dev))),
                    "la": int(job["la"]),
                    "lb": int(job["lb"]),
                    "lq": int(job["lq"]),
                    "n_spAB": int(job["n_spAB"]),
                    "n_spCD": int(job["n_spCD"]),
                    "blocks": int(job["blocks"]),
                    "work_est": int(job["work_est"]),
                    "score": float(job["score"]),
                    "threads": int(job["threads"]),
                    "split": bool(job["split"]),
                    "ab_bucket": int(job.get("ab_bucket", 0)),
                    "cd_bucket": int(job.get("cd_bucket", 0)),
                    "cv_ab": float(job.get("cv_ab", 0.0)),
                    "cv_cd": float(job.get("cv_cd", 0.0)),
                    "use_abtile": bool(job.get("use_abtile", False)),
                    "cd_tile": int(job.get("cd_tile", 1)),
                    "spAB": cp.ascontiguousarray(cp.asarray(spab_host, dtype=cp.int32)),
                    "spCD": cp.ascontiguousarray(cp.asarray(spcd_host, dtype=cp.int32)),
                }
            )
        if not jobs_3c_dev:
            for (la, lb), spab_dev in spAB_by_lab_dev.items():
                n_spab = int(spab_dev.shape[0])
                if n_spab == 0:
                    continue
                for lq, spcd_dev in spCD_by_l_dev.items():
                    n_spcd = int(spcd_dev.shape[0])
                    if n_spcd == 0:
                        continue
                    jobs_3c_dev.append(
                        {
                            "job_id": int(len(jobs_3c_dev)),
                            "la": int(la),
                            "lb": int(lb),
                            "lq": int(lq),
                            "n_spAB": int(n_spab),
                            "n_spCD": int(n_spcd),
                            "blocks": int(n_spab * n_spcd),
                            "work_est": int(n_spab * n_spcd),
                            "score": 0.0,
                            "threads": 256,
                            "split": False,
                            "ab_bucket": 0,
                            "cd_bucket": 0,
                            "cv_ab": 0.0,
                            "cv_cd": 0.0,
                            "use_abtile": False,
                            "cd_tile": 1,
                            "spAB": spab_dev,
                            "spCD": spcd_dev,
                        }
                    )

        # Spherical AO layout for the AO basis. This is needed to transform
        # spherical bar_X(m_sph,n_sph,Q) back to Cartesian for the derivative kernels.
        shell_ao_start_sph_host, nao_sph_host = compute_sph_layout_from_cart_basis(self.ao_basis)
        # Build a combined shell-offset array (same length as shell_ao_start_all)
        # so CUDA kernels can index it with the same combined shell ids used for
        # cart offsets. Only the AO-shell prefix is meaningful; aux/dummy entries
        # are left as 0 and never used.
        n_shell_total = int(np.asarray(self.shell_ao_start_all, dtype=np.int32).size)
        shell_ao_start_sph_all = np.zeros((n_shell_total,), dtype=np.int32)
        n_shell_ao = int(np.asarray(shell_ao_start_sph_host, dtype=np.int32).size)
        if n_shell_ao > n_shell_total:  # pragma: no cover
            raise ValueError("internal error: spherical shell layout longer than combined shell list")
        shell_ao_start_sph_all[:n_shell_ao] = np.asarray(shell_ao_start_sph_host, dtype=np.int32)
        shell_ao_start_sph_dev = cp.ascontiguousarray(cp.asarray(shell_ao_start_sph_all, dtype=cp.int32))

        self.cuda = {
            "_ext": _ext_cuda,
            "shell_cx": cp.ascontiguousarray(cp.asarray(self.shell_cxyz_all[:, 0], dtype=cp.float64)),
            "shell_cy": cp.ascontiguousarray(cp.asarray(self.shell_cxyz_all[:, 1], dtype=cp.float64)),
            "shell_cz": cp.ascontiguousarray(cp.asarray(self.shell_cxyz_all[:, 2], dtype=cp.float64)),
            "shell_prim_start": cp.ascontiguousarray(cp.asarray(self.shell_prim_start_all, dtype=cp.int32)),
            "shell_nprim": cp.ascontiguousarray(cp.asarray(self.shell_nprim_all, dtype=cp.int32)),
            "shell_ao_start": cp.ascontiguousarray(cp.asarray(self.shell_ao_start_all, dtype=cp.int32)),
            "prim_exp": cp.ascontiguousarray(cp.asarray(self.prim_exp_all, dtype=cp.float64)),
            "sp_A": cp.ascontiguousarray(cp.asarray(self.sp_A_all, dtype=cp.int32)),
            "sp_B": cp.ascontiguousarray(cp.asarray(self.sp_B_all, dtype=cp.int32)),
            "sp_pair_start": cp.ascontiguousarray(cp.asarray(self.sp_pair_start_all, dtype=cp.int32)),
            "sp_npair": cp.ascontiguousarray(cp.asarray(self.sp_npair_all, dtype=cp.int32)),
            "pair_eta": cp.ascontiguousarray(cp.asarray(self.pair_eta_all, dtype=cp.float64)),
            "pair_Px": cp.ascontiguousarray(cp.asarray(self.pair_Px_all, dtype=cp.float64)),
            "pair_Py": cp.ascontiguousarray(cp.asarray(self.pair_Py_all, dtype=cp.float64)),
            "pair_Pz": cp.ascontiguousarray(cp.asarray(self.pair_Pz_all, dtype=cp.float64)),
            "pair_cK": cp.ascontiguousarray(cp.asarray(self.pair_cK_all, dtype=cp.float64)),
            "spCD_by_l": spCD_by_l_dev,
            "spCD_by_l_host": spCD_by_l_host,
            "atomC_by_l": atomC_by_l_dev,
            "metric_batches": metric_batches_dev,
            "shell_l_host": shell_l_np,
            "shell_atom": shell_atom_dev,
            "spAB_by_lab": spAB_by_lab_dev,
            "spAB_by_lab_host": spAB_by_lab_host,
            "jobs_3c": jobs_3c_dev,
            "jobs_3c_sched_mode": sched_mode,
            "jobs_3c_ab_tile": int(ab_tile),
            "jobs_3c_bucket_ab": tuple(int(v) for v in ab_edges),
            "jobs_3c_bucket_cd": tuple(int(v) for v in cd_edges),
            "shell_ao_start_sph": shell_ao_start_sph_dev,
            "nao_sph_host": int(nao_sph_host),
            "work_2c_cache": {},
        }

    # ------------------------------------------------------------------
    # Internal: CPU kernel loops extracted for reuse
    # ------------------------------------------------------------------
    def _contract_cpu_from_adjoints(self, bar_X_flat: np.ndarray, bar_V: np.ndarray) -> np.ndarray:
        """Run 3c + 2c CPU kernel loops given pre-computed Cartesian adjoints.

        Parameters
        ----------
        bar_X_flat : np.ndarray, shape (nao_cart * nao_cart, naux)
        bar_V : np.ndarray, shape (naux, naux)
        """
        grad = np.zeros((self.natm, 3), dtype=np.float64)
        if self.cpu is None:  # pragma: no cover
            raise RuntimeError("internal error: CPU function table missing")
        fn_3c = self.cpu["fn_3c"]
        fn_2c = self.cpu["fn_2c"]

        for spAB in range(int(self.nsp_ao)):
            shA = int(self.sp_A_all[int(spAB)])
            shB = int(self.sp_B_all[int(spAB)])
            fac = 2.0 if shA != shB else 1.0
            atomA = int(self.ao_shell_atom[int(shA)])
            atomB = int(self.ao_shell_atom[int(shB)])

            for lq, spCD_batch in self.spCD_by_l.items():
                q_shells = self.shells_by_l[int(lq)]
                out_batch = fn_3c(
                    self.shell_cxyz_all,
                    self.shell_prim_start_all,
                    self.shell_nprim_all,
                    self.shell_l_all,
                    self.shell_ao_start_all,
                    self.prim_exp_all,
                    self.sp_A_all,
                    self.sp_B_all,
                    self.sp_pair_start_all,
                    self.sp_npair_all,
                    self.pair_eta_all,
                    self.pair_Px_all,
                    self.pair_Py_all,
                    self.pair_Pz_all,
                    self.pair_cK_all,
                    int(spAB),
                    spCD_batch,
                    int(self.nao),
                    bar_X_flat,
                )
                out_batch = np.asarray(out_batch, dtype=np.float64)
                for t, qsh in enumerate(q_shells.tolist()):
                    atomC = int(self.aux_shell_atom[int(qsh)])
                    grad[atomA] += fac * out_batch[int(t), 0, :]
                    grad[atomB] += fac * out_batch[int(t), 1, :]
                    grad[atomC] += fac * out_batch[int(t), 2, :]

        for spAB, lp, atomP, lq, spCD_batch, atomQ, fac in self.metric_batches:
            out_batch = fn_2c(
                self.shell_cxyz_all,
                self.shell_prim_start_all,
                self.shell_nprim_all,
                self.shell_l_all,
                self.shell_ao_start_all,
                self.prim_exp_all,
                self.sp_A_all,
                self.sp_B_all,
                self.sp_pair_start_all,
                self.sp_npair_all,
                self.pair_eta_all,
                self.pair_Px_all,
                self.pair_Py_all,
                self.pair_Pz_all,
                self.pair_cK_all,
                int(spAB),
                spCD_batch,
                int(self.nao),
                bar_V,
            )
            out_batch = np.asarray(out_batch, dtype=np.float64)
            grad[int(atomP)] += np.sum(out_batch[:, 0, :] * fac[:, None], axis=0)
            np.add.at(grad, atomQ, out_batch[:, 1, :] * fac[:, None])

        return np.asarray(grad, dtype=np.float64)

    def contract(self, *, B_ao: Any, bar_L_ao: Any) -> np.ndarray:
        if self.backend == "cuda":
            import cupy as cp  # noqa: PLC0415

            grad_dev = self.contract_device(B_ao=B_ao, bar_L_ao=bar_L_ao)
            return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

        B = np.asarray(B_ao, dtype=np.float64, order="C")
        bar_L = np.asarray(bar_L_ao, dtype=np.float64, order="C")

        if B.ndim != 3:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        nao0, nao1, naux = map(int, B.shape)
        if nao0 != nao1:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        if nao0 != int(self.nao) or naux != int(self.naux):
            raise ValueError("B_ao shape mismatch with context")

        if tuple(map(int, bar_L.shape)) != (int(self.naux), int(self.nao), int(self.nao)):
            raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

        bar_L_c = np.asarray(bar_L, dtype=np.float64, order="C")
        bar_X, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L_c, self.L_metric)
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        bar_X = 0.5 * (bar_X + bar_X.transpose((1, 0, 2)))
        bar_X = np.asarray(bar_X, dtype=np.float64, order="C")
        bar_V = np.asarray(bar_V, dtype=np.float64, order="C")
        bar_X_flat = bar_X.reshape((self.nao * self.nao, self.naux))

        return self._contract_cpu_from_adjoints(bar_X_flat, bar_V)

    def contract_sph(self, *, B_sph: Any, bar_L_sph: Any, T_c2s: Any) -> np.ndarray:
        """Contract DF gradient with spherical-basis B and bar_L.

        Computes the DF adjoint (bar_X, bar_V) in the smaller spherical basis,
        then transforms bar_X to Cartesian for the 3c derivative kernel.
        The 2c (metric) contribution is basis-independent (no AO indices).

        Parameters
        ----------
        B_sph : array, shape (nao_sph, nao_sph, naux)
            Whitened DF 3-index integrals in spherical AO basis.
        bar_L_sph : array, shape (naux, nao_sph, nao_sph)
            Adjoint of whitened DF factors in spherical AO basis.
        T_c2s : array, shape (nao_cart, nao_sph)
            Cart-to-spherical transformation matrix.

        Returns
        -------
        np.ndarray, shape (natm, 3)
            Nuclear gradient contribution from DF 2e integrals.
        """
        if self.backend == "cuda":
            import cupy as cp  # noqa: PLC0415

            grad_dev = self.contract_device_sph(B_sph=B_sph, bar_L_sph=bar_L_sph, T_c2s=T_c2s)
            return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

        T = np.asarray(T_c2s, dtype=np.float64)
        B = np.asarray(B_sph, dtype=np.float64, order="C")
        bar_L = np.asarray(bar_L_sph, dtype=np.float64, order="C")

        # 1. Compute adjoints in spherical basis (smaller, faster)
        bar_X_sph, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric)
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)

        # 2. Transform bar_X to Cartesian for 3c kernel: bar_X_cart[mu,nu,Q] = T @ bar_X_sph @ T^T
        bar_X_cart = np.einsum("mi,ijQ,nj->mnQ", T, bar_X_sph, T, optimize=True)
        bar_X_cart = 0.5 * (bar_X_cart + bar_X_cart.transpose((1, 0, 2)))
        bar_X_flat = np.asarray(bar_X_cart.reshape((self.nao * self.nao, self.naux)), dtype=np.float64, order="C")
        bar_V = np.asarray(bar_V, dtype=np.float64, order="C")

        # 3. Reuse existing kernel loops
        return self._contract_cpu_from_adjoints(bar_X_flat, bar_V)

    # ------------------------------------------------------------------
    # Internal: CUDA kernel loops extracted for reuse
    # ------------------------------------------------------------------
    def _contract_device_from_adjoints(self, bar_X_dev: Any, bar_V_dev: Any, *, grad_dev: Any | None = None) -> Any:
        """Run 3c + 2c CUDA kernel loops given pre-computed Cartesian adjoints on device.

        Parameters
        ----------
        bar_X_dev : cupy.ndarray, shape (nao_cart * nao_cart * naux,), flat
        bar_V_dev : cupy.ndarray, shape (naux * naux,), flat
        """
        import cupy as cp  # noqa: PLC0415

        if self.cuda is None:
            raise RuntimeError("CUDA static context is not initialized")
        cuda = self.cuda
        _ext = cuda["_ext"]

        if grad_dev is None:
            grad_dev = cp.zeros((self.natm, 3), dtype=cp.float64)
        else:
            grad_dev = cp.asarray(grad_dev, dtype=cp.float64)
            if tuple(map(int, grad_dev.shape)) != (int(self.natm), 3):
                raise ValueError("grad_dev must have shape (natm, 3)")
            if not bool(getattr(grad_dev, "flags", None).c_contiguous):
                grad_dev = cp.ascontiguousarray(grad_dev, dtype=cp.float64)
        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)

        grad_dev_flat = grad_dev.reshape(-1)
        jobs_3c = cuda.get("jobs_3c")
        if not isinstance(jobs_3c, list) or len(jobs_3c) == 0:
            jobs_3c = []
            for (la_, lb_), spAB_class_dev in cuda["spAB_by_lab"].items():
                n_spAB = int(spAB_class_dev.shape[0])
                if n_spAB == 0:
                    continue
                for lq, spCD_dev in cuda["spCD_by_l"].items():
                    nt = int(spCD_dev.shape[0])
                    if nt == 0:
                        continue
                    jobs_3c.append(
                        {
                            "job_id": int(len(jobs_3c)),
                            "la": int(la_),
                            "lb": int(lb_),
                            "lq": int(lq),
                            "n_spAB": int(n_spAB),
                            "n_spCD": int(nt),
                            "blocks": int(n_spAB * nt),
                            "work_est": int(n_spAB * nt),
                            "threads": int(threads),
                            "split": False,
                            "ab_bucket": 0,
                            "cd_bucket": 0,
                            "use_abtile": False,
                            "cd_tile": 1,
                            "spAB": spAB_class_dev,
                            "spCD": spCD_dev,
                        }
                    )

        _prof_kern = _env_flag_enabled("ASUKA_PROFILE_DF_KERNELS", default=False)
        _prof_jobs = _env_flag_enabled("ASUKA_PROFILE_DF_3C_JOBS", default=False)
        if _prof_jobs:
            mode_s = str(cuda.get("jobs_3c_sched_mode", "legacy"))
            print(f"[DF_3C_JOBS] mode={mode_s} njobs={len(jobs_3c)}")
        if _prof_kern:
            import time as _time  # noqa: PLC0415

            cp.cuda.Device().synchronize()
            _t3c_start = _time.perf_counter()
            _n3c = 0
            _n3c_blocks = 0
            _per_lab: dict[tuple[int, int, int], tuple[float, int]] = {}
            _job_times: list[tuple[float, int, int, int, int, int, int, str, int, int, int]] = []

        has_abtile = hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_inplace_device")
        bar_is_f32 = cp.dtype(getattr(bar_X_dev, "dtype", cp.float64)) == cp.dtype(cp.float32)
        for job in jobs_3c:
            la_ = int(job["la"])
            lb_ = int(job["lb"])
            lq = int(job["lq"])
            spAB_class_dev = job["spAB"]
            spCD_dev = job["spCD"]
            n_spAB = int(job.get("n_spAB", int(spAB_class_dev.shape[0])))
            nt = int(job.get("n_spCD", int(spCD_dev.shape[0])))
            if nt == 0 or n_spAB == 0:
                continue
            threads_job = int(job.get("threads", threads))
            if threads_job <= 0:
                threads_job = int(threads)
            threads_job = min(256, max(32, int(threads_job)))
            threads_job = int((threads_job // 32) * 32)
            cd_tile = max(1, min(int(job.get("cd_tile", 1)), nt))
            use_abtile = bool(job.get("use_abtile", False)) and has_abtile and not bar_is_f32 and cd_tile > 1

            if _prof_kern:
                cp.cuda.Device().synchronize()
                _tk0 = _time.perf_counter()

            if use_abtile:
                _ext.df_int3c2e_deriv_contracted_cart_allsp_atomgrad_abtile_inplace_device(
                    spAB_class_dev,
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(la_),
                    int(lb_),
                    int(lq),
                    bar_X_dev,
                    cuda["shell_atom"],
                    grad_dev_flat,
                    int(cd_tile),
                    int(threads_job),
                    int(stream_ptr),
                    False,
                )
            else:
                _ext.df_int3c2e_deriv_contracted_cart_allsp_atomgrad_inplace_device(
                    spAB_class_dev,
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(la_),
                    int(lb_),
                    int(lq),
                    bar_X_dev,
                    cuda["shell_atom"],
                    grad_dev_flat,
                    int(threads_job),
                    int(stream_ptr),
                    False,
                )

            if _prof_kern:
                cp.cuda.Device().synchronize()
                _dt = _time.perf_counter() - _tk0
                _n3c += 1
                _blocks = int(nt * n_spAB)
                _n3c_blocks += _blocks
                key = (int(la_), int(lb_), int(lq))
                prev_t, prev_b = _per_lab.get(key, (0.0, 0))
                _per_lab[key] = (float(prev_t + _dt), int(prev_b + _blocks))
                if _prof_jobs:
                    _job_times.append(
                        (
                            float(_dt),
                            int(job.get("job_id", -1)),
                            int(la_),
                            int(lb_),
                            int(lq),
                            int(n_spAB),
                            int(nt),
                            int(threads_job),
                            "abtile" if use_abtile else "allsp",
                            int(cd_tile),
                            int(job.get("work_est", _blocks)),
                        )
                    )

        if _prof_kern:
            cp.cuda.Device().synchronize()
            _t3c_end = _time.perf_counter()
            _t3c_total = _t3c_end - _t3c_start
            print(f"[DF_KERNELS] 3c: launches={_n3c} blocks={_n3c_blocks} time={_t3c_total:.3f}s")
            _sorted = sorted(_per_lab.items(), key=lambda x: -x[1][0])
            for (la, lb, lq), (dt, nb) in _sorted[:12]:
                print(f"  (la={la},lb={lb},lq={lq}): {dt:.4f}s  blocks={nb}")
            if _prof_jobs and _job_times:
                print("[DF_3C_JOBS] top kernels by wall-time:")
                for dt, jid, la, lb, lq, nab, ncd, thr, mode, tile, west in sorted(_job_times, key=lambda x: -x[0])[:24]:
                    print(
                        f"  job={jid} (la={la},lb={lb},lq={lq}) mode={mode} tile={tile} "
                        f"nAB={nab} nCD={ncd} threads={thr} work={west} dt={dt:.4f}s"
                    )

        metric_mode = str(os.environ.get("ASUKA_DF_METRIC_2C_KERNEL", "auto")).strip().lower()
        if metric_mode in ("legacy", "batch", "sp_batch", "off", "0", "false", "no", "disable", "disabled"):
            fn_2c_tril = None
        else:
            fn_2c_tril = getattr(
                _ext,
                "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device",
                None,
            )
        if fn_2c_tril is not None:
            for lp, spAB_class_dev in cuda["spCD_by_l"].items():
                n_spAB = int(spAB_class_dev.shape[0])
                if n_spAB == 0:
                    continue
                for lq, spCD_class_dev in cuda["spCD_by_l"].items():
                    ntasks = int(spCD_class_dev.shape[0])
                    if ntasks == 0:
                        continue
                    fn_2c_tril(
                        spAB_class_dev,
                        spCD_class_dev,
                        cuda["sp_A"],
                        cuda["sp_B"],
                        cuda["sp_pair_start"],
                        cuda["sp_npair"],
                        cuda["shell_cx"],
                        cuda["shell_cy"],
                        cuda["shell_cz"],
                        cuda["shell_prim_start"],
                        cuda["shell_nprim"],
                        cuda["shell_ao_start"],
                        cuda["prim_exp"],
                        cuda["pair_eta"],
                        cuda["pair_Px"],
                        cuda["pair_Py"],
                        cuda["pair_Pz"],
                        cuda["pair_cK"],
                        int(self.nao),
                        int(self.naux),
                        int(lp),
                        int(lq),
                        bar_V_dev,
                        cuda["shell_atom"],
                        grad_dev_flat,
                        int(threads),
                        int(stream_ptr),
                        False,
                    )
        else:
            work_2c = cuda.get("work_2c_cache")
            if not isinstance(work_2c, dict):
                work_2c = {}
                cuda["work_2c_cache"] = work_2c
            for spAB, lp, atomP, lq, spCD_dev, atomQ_dev, fac_dev in cuda["metric_batches"]:
                nt = int(spCD_dev.shape[0])
                if nt == 0:
                    continue
                out_dev = work_2c.get(nt)
                if out_dev is None or int(getattr(out_dev, "size", 0)) < int(nt * 6):
                    out_dev = cp.empty((nt * 6,), dtype=cp.float64)
                    work_2c[nt] = out_dev
                _ext.df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device(
                    int(spAB),
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(lp),
                    int(lq),
                    bar_V_dev,
                    out_dev,
                    int(threads),
                    int(stream_ptr),
                    False,
                )
                out_batch_dev = out_dev.reshape((nt, 2, 3))
                grad_dev[int(atomP)] += cp.sum(out_batch_dev[:, 0, :] * fac_dev[:, None], axis=0)
                valsQ = out_batch_dev[:, 1, :] * fac_dev[:, None]
                cp.add.at(grad_dev[:, 0], atomQ_dev, valsQ[:, 0])
                cp.add.at(grad_dev[:, 1], atomQ_dev, valsQ[:, 1])
                cp.add.at(grad_dev[:, 2], atomQ_dev, valsQ[:, 2])

        return grad_dev

    def _contract_device_from_adjoints_sph_qmn(
        self,
        bar_X_sph_Qmn_dev: Any,
        bar_V_dev: Any,
        *,
        nao_sph: int,
        grad_dev: Any | None = None,
    ) -> Any:
        """Run 3c + 2c CUDA kernel loops given spherical bar_X in Qmn layout.

        This avoids allocating the full Cartesian bar_X tensor (nao_cart^2*naux).

        Parameters
        ----------
        bar_X_sph_Qmn_dev : cupy.ndarray, shape (naux*nao_sph*nao_sph,), flat, float64
        bar_V_dev : cupy.ndarray, shape (naux*naux,), flat, float64
        nao_sph : int
            Total number of spherical AOs.
        """
        import cupy as cp  # noqa: PLC0415

        if self.cuda is None:
            raise RuntimeError("CUDA static context is not initialized")
        cuda = self.cuda
        _ext = cuda["_ext"]
        if not hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_inplace_device"):
            raise RuntimeError("CUDA extension missing sphbar-qmn DF 3c derivative contraction kernel")

        if grad_dev is None:
            grad_dev = cp.zeros((self.natm, 3), dtype=cp.float64)
        else:
            grad_dev = cp.asarray(grad_dev, dtype=cp.float64)
            if tuple(map(int, grad_dev.shape)) != (int(self.natm), 3):
                raise ValueError("grad_dev must have shape (natm, 3)")
            if not bool(getattr(grad_dev, "flags", None).c_contiguous):
                grad_dev = cp.ascontiguousarray(grad_dev, dtype=cp.float64)

        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        grad_dev_flat = grad_dev.reshape(-1)

        jobs_3c = cuda.get("jobs_3c")
        if not isinstance(jobs_3c, list) or len(jobs_3c) == 0:
            jobs_3c = []
            for (la_, lb_), spAB_class_dev in cuda["spAB_by_lab"].items():
                n_spAB = int(spAB_class_dev.shape[0])
                if n_spAB == 0:
                    continue
                for lq, spCD_dev in cuda["spCD_by_l"].items():
                    nt = int(spCD_dev.shape[0])
                    if nt == 0:
                        continue
                    jobs_3c.append(
                        {
                            "job_id": int(len(jobs_3c)),
                            "la": int(la_),
                            "lb": int(lb_),
                            "lq": int(lq),
                            "n_spAB": int(n_spAB),
                            "n_spCD": int(nt),
                            "blocks": int(n_spAB * nt),
                            "work_est": int(n_spAB * nt),
                            "threads": int(threads),
                            "split": False,
                            "ab_bucket": 0,
                            "cd_bucket": 0,
                            "use_abtile": False,
                            "cd_tile": 1,
                            "spAB": spAB_class_dev,
                            "spCD": spCD_dev,
                        }
                    )

        _prof_kern = _env_flag_enabled("ASUKA_PROFILE_DF_KERNELS", default=False)
        _prof_jobs = _env_flag_enabled("ASUKA_PROFILE_DF_3C_JOBS", default=False)
        if _prof_jobs:
            mode_s = str(cuda.get("jobs_3c_sched_mode", "legacy"))
            print(f"[DF_3C_JOBS] mode={mode_s} njobs={len(jobs_3c)}")
        if _prof_kern:
            import time as _time  # noqa: PLC0415

            cp.cuda.Device().synchronize()
            _t3c_start = _time.perf_counter()
            _n3c = 0
            _n3c_blocks = 0
            _per_lab: dict[tuple[int, int, int], tuple[float, int]] = {}
            _job_times: list[tuple[float, int, int, int, int, int, int, int, str, int, int]] = []

        has_abtile = hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device")
        for job in jobs_3c:
            la_ = int(job["la"])
            lb_ = int(job["lb"])
            lq = int(job["lq"])
            spAB_class_dev = job["spAB"]
            spCD_dev = job["spCD"]
            n_spAB = int(job.get("n_spAB", int(spAB_class_dev.shape[0])))
            nt = int(job.get("n_spCD", int(spCD_dev.shape[0])))
            if nt == 0 or n_spAB == 0:
                continue
            threads_job = int(job.get("threads", threads))
            if threads_job <= 0:
                threads_job = int(threads)
            threads_job = min(256, max(32, int(threads_job)))
            threads_job = int((threads_job // 32) * 32)
            cd_tile = max(1, min(int(job.get("cd_tile", 1)), nt))
            use_abtile = bool(job.get("use_abtile", False)) and has_abtile and cd_tile > 1

            if _prof_kern:
                cp.cuda.Device().synchronize()
                _tk0 = _time.perf_counter()

            if use_abtile:
                _ext.df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_abtile_inplace_device(
                    spAB_class_dev,
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(nao_sph),
                    int(la_),
                    int(lb_),
                    int(lq),
                    bar_X_sph_Qmn_dev,
                    cuda["shell_ao_start_sph"],
                    cuda["shell_atom"],
                    grad_dev_flat,
                    int(cd_tile),
                    int(threads_job),
                    int(stream_ptr),
                    False,
                )
            else:
                _ext.df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_inplace_device(
                    spAB_class_dev,
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(nao_sph),
                    int(la_),
                    int(lb_),
                    int(lq),
                    bar_X_sph_Qmn_dev,
                    cuda["shell_ao_start_sph"],
                    cuda["shell_atom"],
                    grad_dev_flat,
                    int(threads_job),
                    int(stream_ptr),
                    False,
                )

            if _prof_kern:
                cp.cuda.Device().synchronize()
                _dt = _time.perf_counter() - _tk0
                _n3c += 1
                _blocks = int(nt * n_spAB)
                _n3c_blocks += _blocks
                key = (int(la_), int(lb_), int(lq))
                prev_t, prev_b = _per_lab.get(key, (0.0, 0))
                _per_lab[key] = (float(prev_t + _dt), int(prev_b + _blocks))
                if _prof_jobs:
                    _job_times.append(
                        (
                            float(_dt),
                            int(job.get("job_id", -1)),
                            int(la_),
                            int(lb_),
                            int(lq),
                            int(n_spAB),
                            int(nt),
                            int(threads_job),
                            "abtile" if use_abtile else "allsp",
                            int(cd_tile),
                            int(job.get("work_est", _blocks)),
                        )
                    )

        if _prof_kern:
            cp.cuda.Device().synchronize()
            _t3c_end = _time.perf_counter()
            _t3c_total = _t3c_end - _t3c_start
            print(f"[DF_KERNELS] 3c: launches={_n3c} blocks={_n3c_blocks} time={_t3c_total:.3f}s")
            _sorted = sorted(_per_lab.items(), key=lambda x: -x[1][0])
            for (la, lb, lq), (dt, nb) in _sorted[:12]:
                print(f"  (la={la},lb={lb},lq={lq}): {dt:.4f}s  blocks={nb}")
            if _prof_jobs and _job_times:
                print("[DF_3C_JOBS] top kernels by wall-time:")
                for dt, jid, la, lb, lq, nab, ncd, thr, mode, tile, west in sorted(_job_times, key=lambda x: -x[0])[:24]:
                    print(
                        f"  job={jid} (la={la},lb={lb},lq={lq}) mode={mode} tile={tile} "
                        f"nAB={nab} nCD={ncd} threads={thr} work={west} dt={dt:.4f}s"
                    )

        # Metric 2c contribution (unchanged).
        metric_mode = str(os.environ.get("ASUKA_DF_METRIC_2C_KERNEL", "auto")).strip().lower()
        if metric_mode in ("legacy", "batch", "sp_batch", "off", "0", "false", "no", "disable", "disabled"):
            fn_2c_tril = None
        else:
            fn_2c_tril = getattr(
                _ext,
                "df_metric_2c2e_deriv_contracted_cart_allsp_atomgrad_tril_inplace_device",
                None,
            )
        if _prof_kern:
            cp.cuda.Device().synchronize()
            _t2c_start = _time.perf_counter()
            _n2c = 0
        if fn_2c_tril is not None:
            for lp, spAB_class_dev in cuda["spCD_by_l"].items():
                n_spAB = int(spAB_class_dev.shape[0])
                if n_spAB == 0:
                    continue
                for lq, spCD_class_dev in cuda["spCD_by_l"].items():
                    ntasks = int(spCD_class_dev.shape[0])
                    if ntasks == 0:
                        continue
                    if _prof_kern:
                        _n2c += 1
                    fn_2c_tril(
                        spAB_class_dev,
                        spCD_class_dev,
                        cuda["sp_A"],
                        cuda["sp_B"],
                        cuda["sp_pair_start"],
                        cuda["sp_npair"],
                        cuda["shell_cx"],
                        cuda["shell_cy"],
                        cuda["shell_cz"],
                        cuda["shell_prim_start"],
                        cuda["shell_nprim"],
                        cuda["shell_ao_start"],
                        cuda["prim_exp"],
                        cuda["pair_eta"],
                        cuda["pair_Px"],
                        cuda["pair_Py"],
                        cuda["pair_Pz"],
                        cuda["pair_cK"],
                        int(self.nao),
                        int(self.naux),
                        int(lp),
                        int(lq),
                        bar_V_dev,
                        cuda["shell_atom"],
                        grad_dev_flat,
                        int(threads),
                        int(stream_ptr),
                        False,
                    )
        else:
            work_2c = cuda.get("work_2c_cache")
            if not isinstance(work_2c, dict):
                work_2c = {}
                cuda["work_2c_cache"] = work_2c
            for spAB, lp, atomP, lq, spCD_dev, atomQ_dev, fac_dev in cuda["metric_batches"]:
                nt = int(spCD_dev.shape[0])
                if nt == 0:
                    continue
                out_dev = work_2c.get(nt)
                if out_dev is None or int(getattr(out_dev, "size", 0)) < int(nt * 6):
                    out_dev = cp.empty((nt * 6,), dtype=cp.float64)
                    work_2c[nt] = out_dev
                _ext.df_metric_2c2e_deriv_contracted_cart_sp_batch_inplace_device(
                    int(spAB),
                    spCD_dev,
                    cuda["sp_A"],
                    cuda["sp_B"],
                    cuda["sp_pair_start"],
                    cuda["sp_npair"],
                    cuda["shell_cx"],
                    cuda["shell_cy"],
                    cuda["shell_cz"],
                    cuda["shell_prim_start"],
                    cuda["shell_nprim"],
                    cuda["shell_ao_start"],
                    cuda["prim_exp"],
                    cuda["pair_eta"],
                    cuda["pair_Px"],
                    cuda["pair_Py"],
                    cuda["pair_Pz"],
                    cuda["pair_cK"],
                    int(self.nao),
                    int(self.naux),
                    int(lp),
                    int(lq),
                    bar_V_dev,
                    out_dev,
                    int(threads),
                    int(stream_ptr),
                    False,
                )
                out_batch_dev = out_dev.reshape((nt, 2, 3))
                grad_dev[int(atomP)] += cp.sum(out_batch_dev[:, 0, :] * fac_dev[:, None], axis=0)
                valsQ = out_batch_dev[:, 1, :] * fac_dev[:, None]
                cp.add.at(grad_dev[:, 0], atomQ_dev, valsQ[:, 0])
                cp.add.at(grad_dev[:, 1], atomQ_dev, valsQ[:, 1])
                cp.add.at(grad_dev[:, 2], atomQ_dev, valsQ[:, 2])

        if _prof_kern:
            cp.cuda.Device().synchronize()
            _t2c_end = _time.perf_counter()
            print(f"[DF_KERNELS] 2c: launches={_n2c} time={_t2c_end - _t2c_start:.3f}s")

        return grad_dev

    def _symmetrize_mnQ_device_inplace(self, arr: Any) -> None:
        """Symmetrize arr[m,n,Q] in-place on device.

        Uses the CUDA extension kernel under VRAM pressure (or when
        ``ASUKA_DF_SYM_KERNEL=on``) to avoid transpose-copy temporaries, and
        otherwise keeps the faster CuPy chunked path.
        """
        import cupy as cp  # noqa: PLC0415

        use_kernel = _prefer_sym_kernel(int(getattr(arr, "nbytes", 0)), nchunks=4)
        if not use_kernel:
            _symmetrize_mnQ_inplace(arr, cp)
            return

        if self.cuda is None:
            _symmetrize_mnQ_inplace(arr, cp)
            return

        _ext = self.cuda.get("_ext")
        if _ext is not None:
            try:
                _ext.df_symmetrize_mnq_inplace_device(
                    arr.reshape(-1),
                    int(self.nao),
                    int(self.naux),
                    256,
                    int(cp.cuda.get_current_stream().ptr),
                    False,
                )
                return
            except Exception:
                pass

        _symmetrize_mnQ_inplace(arr, cp)

    def _adjoints_device_from_cart(self, *, B_ao: Any, bar_L_ao: Any, precision: str = "fp64") -> tuple[Any, Any]:
        """Build Cartesian adjoints on device as flat buffers for derivative kernels.

        precision:
            - "fp64": keep bar_X as float64
            - "fp32_acc64"/"tf32": cast bar_X to float32 after symmetrization, but keep grad accumulation in float64
        """
        import cupy as cp  # noqa: PLC0415

        prec = _normalize_fused_precision(precision)

        B = cp.asarray(B_ao, dtype=cp.float64)
        if not B.flags.c_contiguous:
            B = cp.ascontiguousarray(B)
        bar_L = cp.asarray(bar_L_ao, dtype=cp.float64)
        if not bar_L.flags.c_contiguous:
            bar_L = cp.ascontiguousarray(bar_L)

        if B.ndim != 3:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        nao0, nao1, naux = map(int, B.shape)
        if nao0 != nao1:
            raise ValueError("B_ao must have shape (nao, nao, naux)")
        if nao0 != int(self.nao) or naux != int(self.naux):
            raise ValueError("B_ao shape mismatch with context")
        if tuple(map(int, bar_L.shape)) != (int(self.naux), int(self.nao), int(self.nao)):
            raise ValueError("bar_L_ao must have shape (naux, nao, nao)")

        bar_X, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric, overwrite_bar_L=True)
        del bar_L
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        del bar_Lchol

        _ext = self.cuda.get("_ext") if self.cuda is not None else None
        if _ext is not None and hasattr(_ext, "df_symmetrize_qmn_to_mnq_device"):
            # bar_X returned by df_whiten_adjoint_Qmn is a strided view of the
            # underlying (Q,m,n) buffer. Transposing to (Q,m,n) gives a
            # contiguous view we can fuse-transpose+symmetrize into mnQ layout.
            bar_X_qmn = bar_X.transpose((2, 0, 1))
            if not bool(getattr(bar_X_qmn, "flags", None).c_contiguous):
                bar_X_qmn = cp.ascontiguousarray(bar_X_qmn, dtype=cp.float64)

            threads = 256
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
            if prec in ("fp32_acc64", "tf32"):
                bar_X_dev = cp.empty((int(self.nao) * int(self.nao) * int(self.naux),), dtype=cp.float32)
            else:
                bar_X_dev = cp.empty((int(self.nao) * int(self.nao) * int(self.naux),), dtype=cp.float64)
            _ext.df_symmetrize_qmn_to_mnq_device(
                bar_X_qmn.reshape(-1),
                bar_X_dev.reshape(-1),
                int(self.naux),
                int(self.nao),
                int(threads),
                int(stream_ptr),
                False,
            )
            del bar_X_qmn
        else:
            # Fallback: make bar_X contiguous then symmetrize in mnQ layout.
            if not bool(getattr(bar_X, "flags", None).c_contiguous):
                bar_X = cp.ascontiguousarray(bar_X, dtype=cp.float64)

            if prec in ("fp32_acc64", "tf32"):
                # Prefer the fused out-of-place symmetrize+cast kernel if available.
                bar_X_f4 = cp.empty(bar_X.shape, dtype=cp.float32)
                if _ext is not None:
                    try:
                        _ext.df_symmetrize_mnq_to_f32_device(
                            bar_X.reshape(-1),
                            bar_X_f4.reshape(-1),
                            int(self.nao),
                            int(self.naux),
                            256,
                            int(cp.cuda.get_current_stream().ptr),
                            False,
                        )
                    except Exception:
                        self._symmetrize_mnQ_device_inplace(bar_X)
                        bar_X_f4 = cp.asarray(bar_X, dtype=cp.float32)
                else:
                    self._symmetrize_mnQ_device_inplace(bar_X)
                    bar_X_f4 = cp.asarray(bar_X, dtype=cp.float32)
                bar_X_dev = bar_X_f4.reshape(-1)
                if not bool(getattr(bar_X_dev, "flags", None).c_contiguous):
                    bar_X_dev = cp.ascontiguousarray(bar_X_dev, dtype=cp.float32)
                del bar_X_f4
            else:
                self._symmetrize_mnQ_device_inplace(bar_X)
                bar_X_dev = bar_X.reshape(-1)
                if not bool(getattr(bar_X_dev, "flags", None).c_contiguous):
                    bar_X_dev = cp.ascontiguousarray(bar_X_dev, dtype=cp.float64)

        bar_V_dev = bar_V.reshape(-1)
        if not bool(getattr(bar_V_dev, "flags", None).c_contiguous):
            bar_V_dev = cp.ascontiguousarray(bar_V_dev, dtype=cp.float64)
        del bar_X, bar_V

        return bar_X_dev, bar_V_dev

    def _adjoints_device_from_sph(self, *, B_sph: Any, bar_L_sph: Any, T_c2s: Any, precision: str = "fp64") -> tuple[Any, Any]:
        """Build spherical adjoints on device and transform bar_X to Cartesian."""
        import cupy as cp  # noqa: PLC0415

        prec = _normalize_fused_precision(precision)

        B = cp.asarray(B_sph, dtype=cp.float64)
        if not B.flags.c_contiguous:
            B = cp.ascontiguousarray(B)
        bar_L = cp.asarray(bar_L_sph, dtype=cp.float64)
        if not bar_L.flags.c_contiguous:
            bar_L = cp.ascontiguousarray(bar_L)

        if B.ndim != 3:
            raise ValueError("B_sph must have shape (nao_sph, nao_sph, naux)")
        nao_sph0, nao_sph1, naux = map(int, B.shape)
        if nao_sph0 != nao_sph1:
            raise ValueError("B_sph must have shape (nao_sph, nao_sph, naux)")
        if naux != int(self.naux):
            raise ValueError("B_sph shape mismatch with context naux")
        if tuple(map(int, bar_L.shape)) != (int(self.naux), int(nao_sph0), int(nao_sph0)):
            raise ValueError("bar_L_sph must have shape (naux, nao_sph, nao_sph)")

        T = cp.asarray(T_c2s, dtype=cp.float64)
        if tuple(map(int, getattr(T, "shape", ()))) != (int(self.nao), int(nao_sph0)):
            raise ValueError("T_c2s must have shape (nao_cart, nao_sph)")

        bar_X_sph, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric, overwrite_bar_L=True)
        del bar_L
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        del bar_Lchol

        # Fused spherical symmetrize + spherical->Cartesian transform when available.
        bar_X_dev = None
        bar_X_sph_sym = None
        _ext = self.cuda.get("_ext") if self.cuda is not None else None
        if (
            _ext is not None
            and (
                hasattr(_ext, "df_bar_x_sph_qmn_to_cart_sym_device")
                or (hasattr(_ext, "df_symmetrize_qmn_to_mnq_device") and hasattr(_ext, "df_bar_x_sph_to_cart_sym_device"))
            )
            and self.cuda is not None
            and "spAB_by_lab" in self.cuda
            and "shell_ao_start_sph" in self.cuda
        ):
            try:
                # Work in Qmn layout: this is often already contiguous because bar_X is produced
                # from a (Q, mn) triangular solve. Keeping Q leading also makes the fused
                # symmetrize+transform kernel easy.
                bar_X_qmn = bar_X_sph.transpose((2, 0, 1))
                if not bool(getattr(bar_X_qmn, "flags", None).c_contiguous):
                    bar_X_qmn = cp.ascontiguousarray(bar_X_qmn, dtype=cp.float64)

                threads = 256
                stream_ptr = int(cp.cuda.get_current_stream().ptr)
                out_dtype = cp.float32 if prec in ("fp32_acc64", "tf32") else cp.float64
                bar_X_dev = cp.empty((int(self.nao) * int(self.nao) * int(self.naux),), dtype=out_dtype)
                if hasattr(_ext, "df_bar_x_sph_qmn_to_cart_sym_device"):
                    # Fully fused path: symmetrize in spherical basis on-the-fly and transform to Cartesian.
                    for (la, lb), spAB_arr_dev in self.cuda["spAB_by_lab"].items():
                        _ext.df_bar_x_sph_qmn_to_cart_sym_device(
                            spAB_arr_dev,
                            self.cuda["sp_A"],
                            self.cuda["sp_B"],
                            self.cuda["shell_ao_start"],
                            self.cuda["shell_ao_start_sph"],
                            int(self.nao),
                            int(nao_sph0),
                            int(self.naux),
                            int(la),
                            int(lb),
                            bar_X_qmn.reshape(-1),
                            bar_X_dev,
                            int(threads),
                            int(stream_ptr),
                            False,
                        )
                else:
                    # 2-step fallback: symmetrize in spherical basis to mnQ buffer then transform.
                    bar_X_sph_sym = cp.empty((nao_sph0 * nao_sph0 * int(self.naux),), dtype=cp.float64)
                    _ext.df_symmetrize_qmn_to_mnq_device(
                        bar_X_qmn.reshape(-1),
                        bar_X_sph_sym,
                        int(self.naux),
                        int(nao_sph0),
                        int(threads),
                        int(stream_ptr),
                        False,
                    )
                    for (la, lb), spAB_arr_dev in self.cuda["spAB_by_lab"].items():
                        _ext.df_bar_x_sph_to_cart_sym_device(
                            spAB_arr_dev,
                            self.cuda["sp_A"],
                            self.cuda["sp_B"],
                            self.cuda["shell_ao_start"],
                            self.cuda["shell_ao_start_sph"],
                            int(self.nao),
                            int(nao_sph0),
                            int(self.naux),
                            int(la),
                            int(lb),
                            bar_X_sph_sym,
                            bar_X_dev,
                            int(threads),
                            int(stream_ptr),
                            False,
                        )
                del bar_X_qmn, bar_X_sph
            except Exception:
                bar_X_dev = None
                bar_X_sph_sym = None

        if bar_X_dev is None:
            # Fallback: transform with provided T_c2s, then symmetrize/cast in Cartesian.
            if bar_X_sph_sym is not None:
                bar_X_sph = bar_X_sph_sym.reshape((int(nao_sph0), int(nao_sph0), int(self.naux)))
                bar_X_sph_sym = None
            bar_X_cart = cp.einsum("mi,ijQ,nj->mnQ", T, bar_X_sph, T, optimize=True)
            del bar_X_sph
            if not bool(getattr(bar_X_cart, "flags", None).c_contiguous):
                bar_X_cart = cp.ascontiguousarray(bar_X_cart, dtype=cp.float64)

            if prec in ("fp32_acc64", "tf32"):
                bar_X_f4 = cp.empty(bar_X_cart.shape, dtype=cp.float32)
                if _ext is not None:
                    try:
                        _ext.df_symmetrize_mnq_to_f32_device(
                            bar_X_cart.reshape(-1),
                            bar_X_f4.reshape(-1),
                            int(self.nao),
                            int(self.naux),
                            256,
                            int(cp.cuda.get_current_stream().ptr),
                            False,
                        )
                    except Exception:
                        self._symmetrize_mnQ_device_inplace(bar_X_cart)
                        bar_X_f4 = cp.asarray(bar_X_cart, dtype=cp.float32)
                else:
                    self._symmetrize_mnQ_device_inplace(bar_X_cart)
                    bar_X_f4 = cp.asarray(bar_X_cart, dtype=cp.float32)
                bar_X_dev = bar_X_f4.reshape(-1)
                if not bool(getattr(bar_X_dev, "flags", None).c_contiguous):
                    bar_X_dev = cp.ascontiguousarray(bar_X_dev, dtype=cp.float32)
                del bar_X_f4
            else:
                self._symmetrize_mnQ_device_inplace(bar_X_cart)
                bar_X_dev = bar_X_cart.reshape(-1)
                if not bool(getattr(bar_X_dev, "flags", None).c_contiguous):
                    bar_X_dev = cp.ascontiguousarray(bar_X_dev, dtype=cp.float64)
            del bar_X_cart
        else:
            # Success: free spherical temporary.
            if bar_X_sph_sym is not None:
                del bar_X_sph_sym

        bar_V_dev = bar_V.reshape(-1)
        if not bool(getattr(bar_V_dev, "flags", None).c_contiguous):
            bar_V_dev = cp.ascontiguousarray(bar_V_dev, dtype=cp.float64)
        del bar_V

        return bar_X_dev, bar_V_dev

    def _adjoints_device_from_sph_qmn(
        self,
        *,
        B_sph: Any,
        bar_L_sph: Any,
    ) -> tuple[Any, Any, int]:
        """Build spherical adjoints on device and return bar_X in Qmn layout (in-place) for fused 3c contraction."""
        import cupy as cp  # noqa: PLC0415

        if self.cuda is None:
            raise RuntimeError("CUDA static context is not initialized")
        _ext = self.cuda.get("_ext")
        if _ext is None or not hasattr(_ext, "df_symmetrize_qmn_inplace_device"):
            raise RuntimeError("cuERI CUDA extension is required for sph-Qmn adjoint path")

        B = cp.asarray(B_sph, dtype=cp.float64)
        if not B.flags.c_contiguous:
            B = cp.ascontiguousarray(B)
        bar_L = cp.asarray(bar_L_sph, dtype=cp.float64)
        if not bar_L.flags.c_contiguous:
            bar_L = cp.ascontiguousarray(bar_L)

        if B.ndim != 3:
            raise ValueError("B_sph must have shape (nao_sph, nao_sph, naux)")
        nao_sph0, nao_sph1, naux = map(int, B.shape)
        if nao_sph0 != nao_sph1:
            raise ValueError("B_sph must have shape (nao_sph, nao_sph, naux)")
        if naux != int(self.naux):
            raise ValueError("B_sph shape mismatch with context naux")
        if tuple(map(int, bar_L.shape)) != (int(self.naux), int(nao_sph0), int(nao_sph0)):
            raise ValueError("bar_L_sph must have shape (naux, nao_sph, nao_sph)")

        # Whitening adjoint: overwrite bar_L in-place so its buffer becomes the contiguous Qmn solution.
        _, bar_Lchol = df_whiten_adjoint_Qmn(B, bar_L, self.L_metric, overwrite_bar_L=True)
        bar_V = chol_lower_adjoint(self.L_metric, bar_Lchol)
        del bar_Lchol

        # bar_L now holds tmp == bar_X^T in Qmn layout (naux, nao_sph, nao_sph), contiguous.
        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        _ext.df_symmetrize_qmn_inplace_device(
            bar_L.reshape(-1),
            int(self.naux),
            int(nao_sph0),
            int(threads),
            int(stream_ptr),
            False,
        )

        bar_X_sph_Qmn_dev = bar_L.reshape(-1)
        if not bool(getattr(bar_X_sph_Qmn_dev, "flags", None).c_contiguous):
            bar_X_sph_Qmn_dev = cp.ascontiguousarray(bar_X_sph_Qmn_dev, dtype=cp.float64)

        bar_V_dev = bar_V.reshape(-1)
        if not bool(getattr(bar_V_dev, "flags", None).c_contiguous):
            bar_V_dev = cp.ascontiguousarray(bar_V_dev, dtype=cp.float64)
        del bar_V

        return bar_X_sph_Qmn_dev, bar_V_dev, int(nao_sph0)

    def contract_device(self, *, B_ao: Any, bar_L_ao: Any) -> Any:
        """CUDA-only version of :meth:`contract` that returns the gradient on device.

        Returns
        -------
        cupy.ndarray
            Gradient array on device with shape (natm, 3) and dtype float64.
        """
        if self.backend != "cuda":
            raise NotImplementedError("contract_device is only available for backend='cuda'")
        import cupy as cp  # noqa: PLC0415
        _prof = os.environ.get("ASUKA_PROFILE_DF_CONTRACT", "0") == "1"
        if _prof:
            import time as _time
            cp.cuda.Device().synchronize()
            _t0 = _time.perf_counter()
        bar_X_dev, bar_V_dev = self._adjoints_device_from_cart(B_ao=B_ao, bar_L_ao=bar_L_ao)
        if _prof:
            cp.cuda.Device().synchronize()
            _t1 = _time.perf_counter()
        result = self._contract_device_from_adjoints(bar_X_dev, bar_V_dev)
        if _prof:
            cp.cuda.Device().synchronize()
            _t2 = _time.perf_counter()
            print(f"[DF_CONTRACT] adjoint={_t1-_t0:.3f}s kernels={_t2-_t1:.3f}s total={_t2-_t0:.3f}s")
        return result

    def contract_device_sph(self, *, B_sph: Any, bar_L_sph: Any, T_c2s: Any, precision: str = "fp64") -> Any:
        """CUDA spherical variant of :meth:`contract_device`.

        Parameters
        ----------
        B_sph : cupy.ndarray, shape (nao_sph, nao_sph, naux)
        bar_L_sph : cupy.ndarray, shape (naux, nao_sph, nao_sph)
        T_c2s : array, shape (nao_cart, nao_sph)

        Returns
        -------
        cupy.ndarray, shape (natm, 3)
        """
        if self.backend != "cuda":
            raise NotImplementedError("contract_device_sph is only available for backend='cuda'")
        import cupy as cp  # noqa: PLC0415
        _prof = os.environ.get("ASUKA_PROFILE_DF_CONTRACT", "0") == "1"
        mode = str(os.environ.get("ASUKA_DF_3C_SPH_KERNEL", "auto")).strip().lower()
        use_new = mode not in ("off", "0", "false", "no", "disable", "disabled")
        _ext = self.cuda.get("_ext") if self.cuda is not None else None
        has_new = _ext is not None and hasattr(_ext, "df_int3c2e_deriv_contracted_cart_allsp_atomgrad_sphbar_qmn_inplace_device")
        if mode in ("on", "1", "true", "yes") and not has_new:  # pragma: no cover
            raise RuntimeError("ASUKA_DF_3C_SPH_KERNEL=on but sphbar-qmn kernel is unavailable; rebuild CUDA extension")

        if use_new and has_new:
            if _prof:
                import time as _time
                cp.cuda.Device().synchronize()
                _t0 = _time.perf_counter()
            bar_X_sph_Qmn_dev, bar_V_dev, nao_sph = self._adjoints_device_from_sph_qmn(B_sph=B_sph, bar_L_sph=bar_L_sph)
            if _prof:
                cp.cuda.Device().synchronize()
                _t1 = _time.perf_counter()
            grad_dev = self._contract_device_from_adjoints_sph_qmn(bar_X_sph_Qmn_dev, bar_V_dev, nao_sph=nao_sph)
            if _prof:
                cp.cuda.Device().synchronize()
                _t2 = _time.perf_counter()
                print(f"[DF_CONTRACT_SPH] adjoint={_t1-_t0:.3f}s kernels={_t2-_t1:.3f}s total={_t2-_t0:.3f}s")
            del bar_X_sph_Qmn_dev, bar_V_dev
            return grad_dev

        # Fallback: build full Cartesian bar_X (may OOM for large bases).
        bar_X_dev, bar_V_dev = self._adjoints_device_from_sph(
            B_sph=B_sph,
            bar_L_sph=bar_L_sph,
            T_c2s=T_c2s,
            precision=_normalize_fused_precision(precision),
        )
        return self._contract_device_from_adjoints(bar_X_dev, bar_V_dev)

    def contract_device_fused_terms(self, *, B_ao: Any, bar_L_terms: Iterable[Any], precision: str = "fp64") -> Any:
        """CUDA fused-term contraction: sum gradients from multiple bar_L terms without forming bar_L_total.

        Notes
        -----
        We intentionally avoid materializing a full-sized ``bar_L_sum`` buffer. For large
        bases, allocating ``bar_L_sum`` can dominate peak VRAM and lead to OOM.
        """
        if self.backend != "cuda":
            raise NotImplementedError("contract_device_fused_terms is only available for backend='cuda'")

        prec = _normalize_fused_precision(precision)
        terms = [term for term in bar_L_terms if term is not None]
        if len(terms) == 0:
            raise ValueError("bar_L_terms must contain at least one non-None term")
        grad_acc = None
        for term in terms:
            # _adjoints_device_from_cart validates shapes and handles the requested precision.
            bar_X_dev, bar_V_dev = self._adjoints_device_from_cart(B_ao=B_ao, bar_L_ao=term, precision=prec)
            grad_term = self._contract_device_from_adjoints(bar_X_dev, bar_V_dev)
            del bar_X_dev, bar_V_dev
            if grad_acc is None:
                grad_acc = grad_term
            else:
                grad_acc += grad_term
                del grad_term
        if grad_acc is None:  # pragma: no cover
            raise ValueError("bar_L_terms must contain at least one non-None term")
        return grad_acc

    def contract_device_fused_terms_sph(
        self,
        *,
        B_sph: Any,
        bar_L_terms_sph: Iterable[Any],
        T_c2s: Any,
        precision: str = "fp64",
    ) -> Any:
        """CUDA fused-term spherical contraction variant."""
        if self.backend != "cuda":
            raise NotImplementedError("contract_device_fused_terms_sph is only available for backend='cuda'")

        prec = _normalize_fused_precision(precision)
        terms = [term for term in bar_L_terms_sph if term is not None]
        if len(terms) == 0:
            raise ValueError("bar_L_terms_sph must contain at least one non-None term")
        grad_acc = None
        for term in terms:
            grad_term = self.contract_device_sph(B_sph=B_sph, bar_L_sph=term, T_c2s=T_c2s, precision=prec)
            if grad_acc is None:
                grad_acc = grad_term
            else:
                grad_acc += grad_term
                del grad_term
        if grad_acc is None:  # pragma: no cover
            raise ValueError("bar_L_terms_sph must contain at least one non-None term")
        return grad_acc

    def contract_fused_terms(self, *, B_ao: Any, bar_L_terms: Iterable[Any], precision: str = "fp64") -> np.ndarray:
        """Host-return wrapper for fused-term contraction."""
        if self.backend == "cuda":
            import cupy as cp  # noqa: PLC0415

            grad_dev = self.contract_device_fused_terms(B_ao=B_ao, bar_L_terms=bar_L_terms, precision=precision)
            return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

        grad = np.zeros((self.natm, 3), dtype=np.float64)
        nterms = 0
        for term in bar_L_terms:
            if term is None:
                continue
            nterms += 1
            grad += self.contract(B_ao=B_ao, bar_L_ao=term)
        if nterms == 0:
            raise ValueError("bar_L_terms must contain at least one non-None term")
        return np.asarray(grad, dtype=np.float64)

    def contract_fused_terms_sph(
        self,
        *,
        B_sph: Any,
        bar_L_terms_sph: Iterable[Any],
        T_c2s: Any,
        precision: str = "fp64",
    ) -> np.ndarray:
        """Host-return spherical wrapper for fused-term contraction."""
        if self.backend == "cuda":
            import cupy as cp  # noqa: PLC0415

            grad_dev = self.contract_device_fused_terms_sph(
                B_sph=B_sph,
                bar_L_terms_sph=bar_L_terms_sph,
                T_c2s=T_c2s,
                precision=precision,
            )
            return np.asarray(cp.asnumpy(grad_dev), dtype=np.float64)

        grad = np.zeros((self.natm, 3), dtype=np.float64)
        nterms = 0
        for term in bar_L_terms_sph:
            if term is None:
                continue
            nterms += 1
            grad += self.contract_sph(B_sph=B_sph, bar_L_sph=term, T_c2s=T_c2s)
        if nterms == 0:
            raise ValueError("bar_L_terms_sph must contain at least one non-None term")
        return np.asarray(grad, dtype=np.float64)
