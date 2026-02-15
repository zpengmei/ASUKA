from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, None)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name, None)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


class CudaMrciHopWorkspace:
    """Reusable buffers for tiled `hop_cuda_epq_table`."""

    def __init__(self, *, ncsf: int, nops: int, max_tile_csf: int, naux: int | None = None, dtype=None):
        import cupy as cp

        ncsf = int(ncsf)
        nops = int(nops)
        max_tile_csf = int(max_tile_csf)
        if ncsf <= 0 or nops <= 0 or max_tile_csf <= 0:
            raise ValueError("invalid ncsf/nops/max_tile_csf")
        fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
        if fp_dtype not in (cp.float32, cp.float64):
            raise ValueError("dtype must be float32 or float64")

        self.ncsf = ncsf
        self.nops = nops
        self.max_tile_csf = max_tile_csf
        self.dtype = fp_dtype
        self.itemsize = int(cp.dtype(fp_dtype).itemsize)

        self.W = cp.empty((max_tile_csf, nops), dtype=fp_dtype)
        self.G = cp.empty((max_tile_csf, nops), dtype=fp_dtype)
        self.tmp = cp.empty((max_tile_csf,), dtype=fp_dtype)

        self.task_csf = cp.arange(ncsf, dtype=cp.int32)
        self.overflow = cp.empty((1,), dtype=cp.int32)

        self.naux = None if naux is None else int(naux)
        self.Z = None
        if self.naux is not None and self.naux > 0:
            self.Z = cp.empty((max_tile_csf, int(self.naux)), dtype=fp_dtype)

        # Symmetric-pair contraction buffers/caches. Built lazily when requested.
        #
        # The symmetric-pair mode keeps all E_pq operator actions (W build / apply)
        # in full ordered-pair space (norb^2), but performs the expensive two-body
        # contraction in reduced unordered-pair space (norb*(norb+1)//2) when the
        # 1e/2e integrals are symmetric.
        self.W_pair = None
        self.G_pair = None
        self._sym_pair_norb = None
        self._sym_pair_pair_pq = None
        self._sym_pair_pair_qp = None
        self._sym_pair_diag_u = None
        self._sym_pair_full_to_pair = None
        self._sym_pair_eri_src_id = None
        self._sym_pair_eri_pair_t = None
        self._sym_pair_h_eff_src_id = None
        self._sym_pair_h_eff_pair = None
        self._sym_pair_l_src_id = None
        self._sym_pair_l_pair = None

        self.profile_calls = 0
        self.profile_last: dict[str, float] | None = None
        self.profile_total: dict[str, float] = {}

    @classmethod
    def auto(
        cls,
        *,
        ncsf: int,
        nops: int,
        naux: int | None = None,
        sym_pair: bool = False,
        tile_csf: int | None = None,
        mem_fraction: float | None = None,
        dtype=None,
    ) -> "CudaMrciHopWorkspace":
        import cupy as cp

        ncsf = int(ncsf)
        nops = int(nops)
        if ncsf <= 0 or nops <= 0:
            raise ValueError("invalid ncsf/nops")
        fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
        if fp_dtype not in (cp.float32, cp.float64):
            raise ValueError("dtype must be float32 or float64")
        itemsize = int(cp.dtype(fp_dtype).itemsize)

        if tile_csf is None:
            tile_csf = _env_int("CUGUGA_MRCI_CUDA_TILE_CSF", 0)
        tile_csf = int(tile_csf)

        if tile_csf <= 0:
            if mem_fraction is None:
                mem_fraction = _env_float("CUGUGA_MRCI_CUDA_TILE_MEM_FRACTION", 0.6)
            mem_fraction = float(mem_fraction)
            if not (0.0 < mem_fraction <= 1.0):
                mem_fraction = 0.6

            free_bytes, _total_bytes = cp.cuda.runtime.memGetInfo()
            budget = int(float(free_bytes) * float(mem_fraction))
            naux_i = 0 if naux is None else max(0, int(naux))
            # W + G tiles are always needed; add Z tile for DF/Cholesky factor mode.
            denom = int(2 * itemsize) * int(nops) + int(itemsize) * int(naux_i)
            if bool(sym_pair):
                norb = int(round(float(nops) ** 0.5))
                if int(norb) * int(norb) == int(nops):
                    npair = int(norb) * (int(norb) + 1) // 2
                    denom += int(2 * itemsize) * int(npair)  # W_pair + G_pair
            tile_csf = int(budget // denom) if denom > 0 else 0

        tile_csf = max(1, min(int(tile_csf), int(ncsf)))
        return cls(ncsf=ncsf, nops=nops, max_tile_csf=tile_csf, naux=naux, dtype=fp_dtype)

    def _ensure_sym_pair_maps(self, *, norb: int):
        import cupy as cp
        import numpy as np

        norb = int(norb)
        if norb <= 0:
            raise ValueError("norb must be > 0")
        if int(norb) * int(norb) != int(self.nops):
            raise ValueError("sym_pair requires nops == norb*norb")

        if self._sym_pair_norb == norb:
            return

        nops = int(self.nops)
        pair_pq = []
        pair_qp = []
        diag_u = []
        full_to_pair = np.empty((nops,), dtype=np.int32)

        u = 0
        for p in range(norb):
            for q in range(p, norb):
                pq = p * norb + q
                qp = q * norb + p
                pair_pq.append(pq)
                pair_qp.append(qp)
                if p == q:
                    diag_u.append(u)
                full_to_pair[pq] = u
                full_to_pair[qp] = u
                u += 1

        npair = int(u)
        self._sym_pair_norb = int(norb)
        self._sym_pair_pair_pq = cp.asarray(np.asarray(pair_pq, dtype=np.int32), dtype=cp.int32)
        self._sym_pair_pair_qp = cp.asarray(np.asarray(pair_qp, dtype=np.int32), dtype=cp.int32)
        self._sym_pair_diag_u = cp.asarray(np.asarray(diag_u, dtype=np.int32), dtype=cp.int32)
        self._sym_pair_full_to_pair = cp.asarray(full_to_pair, dtype=cp.int32)

        self.W_pair = cp.empty((int(self.max_tile_csf), int(npair)), dtype=self.dtype)
        self.G_pair = cp.empty((int(self.max_tile_csf), int(npair)), dtype=self.dtype)

        # Invalidate caches tied to old shapes/inputs.
        self._sym_pair_eri_src_id = None
        self._sym_pair_eri_pair_t = None
        self._sym_pair_h_eff_src_id = None
        self._sym_pair_h_eff_pair = None
        self._sym_pair_l_src_id = None
        self._sym_pair_l_pair = None

    def _sym_pair_get_eri_pair_t(self, eri_mat_t):
        import cupy as cp

        src_id = id(eri_mat_t)
        if self._sym_pair_eri_pair_t is not None and self._sym_pair_eri_src_id == src_id:
            return self._sym_pair_eri_pair_t

        pair_pq = self._sym_pair_pair_pq
        pair_qp = self._sym_pair_pair_qp
        if pair_pq is None or pair_qp is None:
            raise RuntimeError("sym-pair maps not initialized")

        eri_mat_t = cp.asarray(eri_mat_t, dtype=self.dtype)
        eri_mat_t = cp.ascontiguousarray(eri_mat_t)
        if eri_mat_t.ndim != 2 or tuple(eri_mat_t.shape) != (int(self.nops), int(self.nops)):
            raise ValueError("eri_mat_t must have shape (nops,nops)")

        # Robustify against small pq<->qp asymmetry in numerically generated ERI matrices.
        # For exact molecular ERIs this average is a no-op.
        rows = 0.5 * (cp.take(eri_mat_t, pair_pq, axis=0) + cp.take(eri_mat_t, pair_qp, axis=0))
        eri_pair_t = 0.5 * (cp.take(rows, pair_pq, axis=1) + cp.take(rows, pair_qp, axis=1))
        eri_pair_t = cp.ascontiguousarray(eri_pair_t)

        self._sym_pair_eri_src_id = src_id
        self._sym_pair_eri_pair_t = eri_pair_t
        return eri_pair_t

    def _sym_pair_get_h_eff_pair(self, h_eff):
        import cupy as cp

        src_id = id(h_eff)
        if self._sym_pair_h_eff_pair is not None and self._sym_pair_h_eff_src_id == src_id:
            return self._sym_pair_h_eff_pair

        pair_pq = self._sym_pair_pair_pq
        pair_qp = self._sym_pair_pair_qp
        if pair_pq is None or pair_qp is None or self._sym_pair_norb is None:
            raise RuntimeError("sym-pair maps not initialized")

        h_eff = cp.asarray(h_eff, dtype=self.dtype)
        h_eff = cp.ascontiguousarray(h_eff)
        if h_eff.ndim != 2 or tuple(h_eff.shape) != (int(self._sym_pair_norb), int(self._sym_pair_norb)):
            raise ValueError("h_eff must have shape (norb,norb)")

        h_eff_flat = h_eff.ravel()
        h_eff_pair = 0.5 * (cp.take(h_eff_flat, pair_pq, axis=0) + cp.take(h_eff_flat, pair_qp, axis=0))
        h_eff_pair = cp.ascontiguousarray(h_eff_pair)

        self._sym_pair_h_eff_src_id = src_id
        self._sym_pair_h_eff_pair = h_eff_pair
        return h_eff_pair

    def _sym_pair_get_l_pair(self, l_full_d):
        import cupy as cp

        src_id = id(l_full_d)
        if self._sym_pair_l_pair is not None and self._sym_pair_l_src_id == src_id:
            return self._sym_pair_l_pair

        pair_pq = self._sym_pair_pair_pq
        pair_qp = self._sym_pair_pair_qp
        if pair_pq is None or pair_qp is None:
            raise RuntimeError("sym-pair maps not initialized")

        l_full_d = cp.asarray(l_full_d, dtype=self.dtype)
        l_full_d = cp.ascontiguousarray(l_full_d)
        if l_full_d.ndim != 2 or int(l_full_d.shape[0]) != int(self.nops):
            raise ValueError("l_full must have shape (nops,naux)")

        l_pair = 0.5 * (cp.take(l_full_d, pair_pq, axis=0) + cp.take(l_full_d, pair_qp, axis=0))
        l_pair = cp.ascontiguousarray(l_pair)

        self._sym_pair_l_src_id = src_id
        self._sym_pair_l_pair = l_pair
        return l_pair


def _hop_cuda_epq_table_1d_tiled(
    *,
    drt,
    drt_dev,
    state_dev,
    epq_table,
    h_eff,
    eri_mat_t,
    l_full,
    x,
    y,
    workspace: CudaMrciHopWorkspace | None,
    tile_csf: int | None,
    fp_dtype,
    sync: bool,
    check_overflow: bool,
    profile_stages: bool,
    profile_stage_sync: bool,
    sym_pair: bool,
    build_threads: int,
    diag_threads: int,
    apply_threads: int,
):
    import cupy as cp
    from asuka.cuda.cuda_backend import (
        apply_g_flat_scatter_atomic_inplace_device,
        build_occ_block_from_steps_inplace_device,
        build_w_diag_from_steps_inplace_device,
        build_w_from_epq_table_inplace_device,
    )

    fp_dtype = cp.dtype(fp_dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("fp_dtype must be float32 or float64")

    x = cp.asarray(x, dtype=fp_dtype).ravel()
    x = cp.ascontiguousarray(x)

    ncsf = int(x.size)
    norb = int(h_eff.shape[0])
    nops = norb * norb
    use_df = l_full is not None

    l_full_d = None
    naux = 0
    if use_df:
        l_full_d = cp.asarray(l_full, dtype=fp_dtype)
        if l_full_d.ndim != 2 or int(l_full_d.shape[0]) != int(nops):
            raise ValueError("l_full must have shape (nops,naux)")
        naux = int(l_full_d.shape[1])
    else:
        if eri_mat_t is None:
            raise ValueError("either eri_mat_t or l_full must be provided")
        eri_mat_t = cp.asarray(eri_mat_t, dtype=fp_dtype)

    sym_pair = bool(sym_pair)

    if workspace is None:
        workspace = CudaMrciHopWorkspace.auto(
            ncsf=ncsf,
            nops=nops,
            naux=(None if not use_df else int(naux)),
            sym_pair=bool(sym_pair),
            tile_csf=tile_csf,
            dtype=fp_dtype,
        )
    else:
        if int(workspace.ncsf) != int(ncsf) or int(workspace.nops) != int(nops):
            raise ValueError("workspace has incompatible ncsf/nops")
        if cp.dtype(getattr(workspace, "dtype", cp.float64)) != fp_dtype:
            raise ValueError("workspace dtype does not match fp_dtype")
        if use_df:
            if workspace.Z is None or int(workspace.naux or 0) != int(naux):
                workspace.naux = int(naux)
                workspace.Z = cp.empty((int(workspace.max_tile_csf), int(naux)), dtype=fp_dtype)

    if tile_csf is None:
        tile_csf = int(workspace.max_tile_csf)
    tile_csf = max(1, min(int(tile_csf), int(workspace.max_tile_csf), int(ncsf)))

    y = cp.asarray(y, dtype=fp_dtype).ravel()
    if not getattr(y, "flags", None) or not y.flags.c_contiguous:
        raise ValueError("y must be a C-contiguous device array of shape (ncsf,)")
    if y.shape != (ncsf,):
        raise ValueError("y must have shape (ncsf,)")
    y.fill(0)

    if not sym_pair:
        h_eff_flat = cp.asarray(h_eff, dtype=fp_dtype).ravel()
    else:
        workspace._ensure_sym_pair_maps(norb=norb)
        pair_pq = workspace._sym_pair_pair_pq
        pair_qp = workspace._sym_pair_pair_qp
        diag_u = workspace._sym_pair_diag_u
        full_to_pair = workspace._sym_pair_full_to_pair
        if (
            pair_pq is None
            or pair_qp is None
            or diag_u is None
            or full_to_pair is None
            or workspace.W_pair is None
            or workspace.G_pair is None
        ):
            raise RuntimeError("sym-pair buffers not initialized")
        h_eff_pair = workspace._sym_pair_get_h_eff_pair(h_eff)
        if use_df:
            l_pair = workspace._sym_pair_get_l_pair(l_full_d)
        else:
            eri_pair_t = workspace._sym_pair_get_eri_pair_t(eri_mat_t)

    stream = cp.cuda.get_current_stream()
    stage_events = []

    for k_start in range(0, ncsf, tile_csf):
        k_count = min(tile_csf, ncsf - k_start)
        k_end = k_start + k_count

        W = workspace.W[:k_count]
        G = workspace.G[:k_count]
        W.fill(0)

        if profile_stages:
            ev0 = cp.cuda.Event()
            ev1 = cp.cuda.Event()
            ev2 = cp.cuda.Event()
            ev3 = cp.cuda.Event()
            ev4 = cp.cuda.Event()
            ev5 = cp.cuda.Event()
            ev0.record(stream)

        build_w_from_epq_table_inplace_device(
            drt,
            state_dev,
            epq_table,
            x,
            w_out=W,
            overflow=workspace.overflow,
            k_start=k_start,
            k_count=k_count,
            dtype=fp_dtype,
            threads=int(build_threads),
            sync=sync,
            check_overflow=check_overflow,
        )

        if profile_stages:
            ev1.record(stream)

        if fp_dtype == cp.float64:
            build_w_diag_from_steps_inplace_device(
                state_dev,
                j_start=k_start,
                j_count=k_count,
                x=x,
                w_out=W,
                threads=int(diag_threads),
                sync=sync,
                relative_w=True,
            )
        else:
            occ = cp.empty((k_count, norb), dtype=cp.float64)
            build_occ_block_from_steps_inplace_device(
                state_dev,
                j_start=k_start,
                j_count=k_count,
                occ_out=occ,
                threads=int(diag_threads),
                sync=sync,
            )
            diag_idx = cp.arange(norb, dtype=cp.int32)
            rr_idx = diag_idx * int(norb) + diag_idx
            W[:, rr_idx] += (occ.astype(fp_dtype, copy=False) * x[k_start:k_end, None])

        if profile_stages:
            ev2.record(stream)

        if not sym_pair:
            if use_df:
                Z = workspace.Z[:k_count]
                try:
                    cp.matmul(W, l_full_d, out=Z)
                except TypeError:
                    Z[...] = cp.matmul(W, l_full_d)
                try:
                    cp.matmul(Z, l_full_d.T, out=G)
                except TypeError:
                    G[...] = cp.matmul(Z, l_full_d.T)
                G *= 0.5
            else:
                try:
                    cp.matmul(W, eri_mat_t, out=G)
                except TypeError:
                    G[...] = cp.matmul(W, eri_mat_t)
        else:
            W_pair = workspace.W_pair[:k_count]
            G_pair = workspace.G_pair[:k_count]

            cp.take(W, pair_pq, axis=1, out=W_pair)
            cp.take(W, pair_qp, axis=1, out=G_pair)  # scratch
            W_pair += G_pair
            if int(diag_u.size) > 0:
                W_pair[:, diag_u] *= 0.5

            if use_df:
                Z = workspace.Z[:k_count]
                try:
                    cp.matmul(W_pair, l_pair, out=Z)
                except TypeError:
                    Z[...] = cp.matmul(W_pair, l_pair)
                try:
                    cp.matmul(Z, l_pair.T, out=G_pair)
                except TypeError:
                    G_pair[...] = cp.matmul(Z, l_pair.T)
                G_pair *= 0.5
            else:
                try:
                    cp.matmul(W_pair, eri_pair_t, out=G_pair)
                except TypeError:
                    G_pair[...] = cp.matmul(W_pair, eri_pair_t)

            cp.take(G_pair, full_to_pair, axis=1, out=G)

        if profile_stages:
            ev3.record(stream)

        apply_g_flat_scatter_atomic_inplace_device(
            drt=drt,
            drt_dev=drt_dev,
            state_dev=state_dev,
            task_csf=workspace.task_csf[k_start:k_end],
            task_g=G,
            task_scale=None,
            epq_table=epq_table,
            y=y,
            overflow=workspace.overflow,
            threads=int(apply_threads),
            zero_y=False,
            dtype=fp_dtype,
            sync=sync,
            check_overflow=check_overflow,
        )

        if profile_stages:
            ev4.record(stream)

        tmp = workspace.tmp[:k_count]
        if not sym_pair:
            cp.dot(W, h_eff_flat, out=tmp)
        else:
            cp.dot(workspace.W_pair[:k_count], h_eff_pair, out=tmp)
        y[k_start:k_end] += tmp

        if profile_stages:
            ev5.record(stream)
            stage_events.append((ev0, ev1, ev2, ev3, ev4, ev5))

    if profile_stages and stage_events:
        if profile_stage_sync:
            stream.synchronize()

        offdiag_ms = 0.0
        diag_ms = 0.0
        gemm_ms = 0.0
        apply_ms = 0.0
        onee_ms = 0.0
        total_ms = 0.0

        for ev0, ev1, ev2, ev3, ev4, ev5 in stage_events:
            offdiag_ms += float(cp.cuda.get_elapsed_time(ev0, ev1))
            diag_ms += float(cp.cuda.get_elapsed_time(ev1, ev2))
            gemm_ms += float(cp.cuda.get_elapsed_time(ev2, ev3))
            apply_ms += float(cp.cuda.get_elapsed_time(ev3, ev4))
            onee_ms += float(cp.cuda.get_elapsed_time(ev4, ev5))
            total_ms += float(cp.cuda.get_elapsed_time(ev0, ev5))

        prof = {
            "tiles": float(len(stage_events)),
            "offdiag_w_s": 1e-3 * offdiag_ms,
            "diag_w_s": 1e-3 * diag_ms,
            "gemm_s": 1e-3 * gemm_ms,
            "apply_s": 1e-3 * apply_ms,
            "onee_s": 1e-3 * onee_ms,
            "total_s": 1e-3 * total_ms,
        }

        workspace.profile_calls += 1
        workspace.profile_last = prof
        for k, v in prof.items():
            workspace.profile_total[k] = float(workspace.profile_total.get(k, 0.0)) + float(v)

    return y


def _hop_cuda_epq_table_2d_block_tiled(
    *,
    drt,
    drt_dev,
    state_dev,
    epq_table,
    h_eff,
    eri_mat_t,
    l_full,
    x,
    y,
    workspace: CudaMrciHopWorkspace | None,
    tile_csf: int | None,
    fp_dtype,
    sync: bool,
    check_overflow: bool,
    profile_stages: bool,
    profile_stage_sync: bool,
    nvec_group: int,
    sym_pair: bool,
    build_threads: int,
    diag_threads: int,
    apply_threads: int,
):
    import cupy as cp
    from asuka.cuda.cuda_backend import (
        apply_g_flat_scatter_atomic_inplace_device,
        build_occ_block_from_steps_inplace_device,
        build_w_diag_from_steps_inplace_device,
        build_w_from_epq_table_inplace_device,
    )

    fp_dtype = cp.dtype(fp_dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("fp_dtype must be float32 or float64")

    x = cp.asarray(x, dtype=fp_dtype)
    if x.ndim != 2:
        raise ValueError("x must be 2D (ncsf,nvec)")

    if not getattr(x, "flags", None) or not x.flags.f_contiguous:
        x = cp.asfortranarray(x)

    ncsf, nvec = map(int, x.shape)
    if ncsf <= 0 or nvec <= 0:
        raise ValueError("invalid x shape")

    norb = int(h_eff.shape[0])
    nops = norb * norb
    use_df = l_full is not None

    l_full_d = None
    naux = 0
    if use_df:
        l_full_d = cp.asarray(l_full, dtype=fp_dtype)
        if l_full_d.ndim != 2 or int(l_full_d.shape[0]) != int(nops):
            raise ValueError("l_full must have shape (nops,naux)")
        naux = int(l_full_d.shape[1])
    else:
        if eri_mat_t is None:
            raise ValueError("either eri_mat_t or l_full must be provided")
        eri_mat_t = cp.asarray(eri_mat_t, dtype=fp_dtype)

    sym_pair = bool(sym_pair)

    if workspace is None:
        workspace = CudaMrciHopWorkspace.auto(
            ncsf=ncsf,
            nops=nops,
            naux=(None if not use_df else int(naux)),
            sym_pair=bool(sym_pair),
            tile_csf=tile_csf,
            dtype=fp_dtype,
        )
    else:
        if int(workspace.ncsf) != int(ncsf) or int(workspace.nops) != int(nops):
            raise ValueError("workspace has incompatible ncsf/nops")
        if cp.dtype(getattr(workspace, "dtype", cp.float64)) != fp_dtype:
            raise ValueError("workspace dtype does not match fp_dtype")
        if use_df:
            if workspace.Z is None or int(workspace.naux or 0) != int(naux):
                workspace.naux = int(naux)
                workspace.Z = cp.empty((int(workspace.max_tile_csf), int(naux)), dtype=fp_dtype)

    if tile_csf is None:
        tile_csf = int(workspace.max_tile_csf)
    tile_csf = max(1, min(int(tile_csf), int(workspace.max_tile_csf), int(ncsf)))

    copy_back = False
    if y is None:
        y = cp.empty((ncsf, nvec), dtype=fp_dtype, order="F")
        y_f = y
    else:
        y = cp.asarray(y, dtype=fp_dtype)
        if y.shape != (ncsf, nvec):
            raise ValueError("y must have the same shape as x")
        if getattr(y, "flags", None) and y.flags.f_contiguous:
            y_f = y
        else:
            y_f = cp.empty((ncsf, nvec), dtype=fp_dtype, order="F")
            copy_back = True

    y_f.fill(0)

    if not sym_pair:
        h_eff_flat = cp.asarray(h_eff, dtype=fp_dtype).ravel()
    else:
        workspace._ensure_sym_pair_maps(norb=norb)
        pair_pq = workspace._sym_pair_pair_pq
        pair_qp = workspace._sym_pair_pair_qp
        diag_u = workspace._sym_pair_diag_u
        full_to_pair = workspace._sym_pair_full_to_pair
        if (
            pair_pq is None
            or pair_qp is None
            or diag_u is None
            or full_to_pair is None
            or workspace.W_pair is None
            or workspace.G_pair is None
        ):
            raise RuntimeError("sym-pair buffers not initialized")
        h_eff_pair = workspace._sym_pair_get_h_eff_pair(h_eff)
        if use_df:
            l_pair = workspace._sym_pair_get_l_pair(l_full_d)
        else:
            eri_pair_t = workspace._sym_pair_get_eri_pair_t(eri_mat_t)
    stream = cp.cuda.get_current_stream()
    stage_events = []

    nvec_group = int(nvec_group)
    if nvec_group <= 0:
        nvec_group = 1
    nvec_group = min(nvec_group, nvec)

    for v_start in range(0, nvec, nvec_group):
        g = min(nvec_group, nvec - v_start)
        g = min(g, int(workspace.max_tile_csf))
        if g <= 0:
            raise RuntimeError("invalid g for block matvec")

        tile_csf_g = int(workspace.max_tile_csf) // int(g)
        tile_csf_g = max(1, min(int(tile_csf_g), int(tile_csf)))

        for k_start in range(0, ncsf, tile_csf_g):
            k_count = min(tile_csf_g, ncsf - k_start)
            k_end = k_start + k_count
            total_rows = int(k_count) * int(g)

            W_stack = workspace.W[:total_rows]
            G_stack = workspace.G[:total_rows]
            W_stack.fill(0)

            if profile_stages:
                ev0 = cp.cuda.Event()
                ev1 = cp.cuda.Event()
                ev2 = cp.cuda.Event()
                ev3 = cp.cuda.Event()
                ev4 = cp.cuda.Event()
                ev5 = cp.cuda.Event()
                ev0.record(stream)

            for t in range(g):
                x_vec = x[:, v_start + t]
                w_sub = W_stack[t * k_count : (t + 1) * k_count]
                build_w_from_epq_table_inplace_device(
                    drt,
                    state_dev,
                    epq_table,
                    x_vec,
                    w_out=w_sub,
                    overflow=workspace.overflow,
                    k_start=k_start,
                    k_count=k_count,
                    dtype=fp_dtype,
                    threads=int(build_threads),
                    sync=sync,
                    check_overflow=check_overflow,
                )

            if profile_stages:
                ev1.record(stream)

            for t in range(g):
                x_vec = x[:, v_start + t]
                w_sub = W_stack[t * k_count : (t + 1) * k_count]
                if fp_dtype == cp.float64:
                    build_w_diag_from_steps_inplace_device(
                        state_dev,
                        j_start=k_start,
                        j_count=k_count,
                        x=x_vec,
                        w_out=w_sub,
                        threads=int(diag_threads),
                        sync=sync,
                        relative_w=True,
                    )
                else:
                    occ = cp.empty((k_count, norb), dtype=cp.float64)
                    build_occ_block_from_steps_inplace_device(
                        state_dev,
                        j_start=k_start,
                        j_count=k_count,
                        occ_out=occ,
                        threads=int(diag_threads),
                        sync=sync,
                    )
                    diag_idx = cp.arange(norb, dtype=cp.int32)
                    rr_idx = diag_idx * int(norb) + diag_idx
                    w_sub[:, rr_idx] += (occ.astype(fp_dtype, copy=False) * x_vec[k_start:k_end, None])

            if profile_stages:
                ev2.record(stream)

            if not sym_pair:
                if use_df:
                    Z_stack = workspace.Z[:total_rows]
                    try:
                        cp.matmul(W_stack, l_full_d, out=Z_stack)
                    except TypeError:
                        Z_stack[...] = cp.matmul(W_stack, l_full_d)
                    try:
                        cp.matmul(Z_stack, l_full_d.T, out=G_stack)
                    except TypeError:
                        G_stack[...] = cp.matmul(Z_stack, l_full_d.T)
                    G_stack *= 0.5
                else:
                    try:
                        cp.matmul(W_stack, eri_mat_t, out=G_stack)
                    except TypeError:
                        G_stack[...] = cp.matmul(W_stack, eri_mat_t)
            else:
                W_pair_stack = workspace.W_pair[:total_rows]
                G_pair_stack = workspace.G_pair[:total_rows]

                cp.take(W_stack, pair_pq, axis=1, out=W_pair_stack)
                cp.take(W_stack, pair_qp, axis=1, out=G_pair_stack)  # scratch
                W_pair_stack += G_pair_stack
                if int(diag_u.size) > 0:
                    W_pair_stack[:, diag_u] *= 0.5

                if use_df:
                    Z_stack = workspace.Z[:total_rows]
                    try:
                        cp.matmul(W_pair_stack, l_pair, out=Z_stack)
                    except TypeError:
                        Z_stack[...] = cp.matmul(W_pair_stack, l_pair)
                    try:
                        cp.matmul(Z_stack, l_pair.T, out=G_pair_stack)
                    except TypeError:
                        G_pair_stack[...] = cp.matmul(Z_stack, l_pair.T)
                    G_pair_stack *= 0.5
                else:
                    try:
                        cp.matmul(W_pair_stack, eri_pair_t, out=G_pair_stack)
                    except TypeError:
                        G_pair_stack[...] = cp.matmul(W_pair_stack, eri_pair_t)

                cp.take(G_pair_stack, full_to_pair, axis=1, out=G_stack)

            if profile_stages:
                ev3.record(stream)

            task_csf_tile = workspace.task_csf[k_start:k_end]
            for t in range(g):
                y_vec = y_f[:, v_start + t]
                g_sub = G_stack[t * k_count : (t + 1) * k_count]
                apply_g_flat_scatter_atomic_inplace_device(
                    drt=drt,
                    drt_dev=drt_dev,
                    state_dev=state_dev,
                    task_csf=task_csf_tile,
                    task_g=g_sub,
                    task_scale=None,
                    epq_table=epq_table,
                    y=y_vec,
                    overflow=workspace.overflow,
                    threads=int(apply_threads),
                    zero_y=False,
                    dtype=fp_dtype,
                    sync=sync,
                    check_overflow=check_overflow,
                )

            if profile_stages:
                ev4.record(stream)

            tmp = workspace.tmp[:total_rows]
            if not sym_pair:
                cp.dot(W_stack, h_eff_flat, out=tmp)
            else:
                cp.dot(workspace.W_pair[:total_rows], h_eff_pair, out=tmp)
            for t in range(g):
                y_f[k_start:k_end, v_start + t] += tmp[t * k_count : (t + 1) * k_count]

            if profile_stages:
                ev5.record(stream)
                stage_events.append((ev0, ev1, ev2, ev3, ev4, ev5))

    if profile_stages and stage_events:
        if profile_stage_sync:
            stream.synchronize()

        offdiag_ms = 0.0
        diag_ms = 0.0
        gemm_ms = 0.0
        apply_ms = 0.0
        onee_ms = 0.0
        total_ms = 0.0

        for ev0, ev1, ev2, ev3, ev4, ev5 in stage_events:
            offdiag_ms += float(cp.cuda.get_elapsed_time(ev0, ev1))
            diag_ms += float(cp.cuda.get_elapsed_time(ev1, ev2))
            gemm_ms += float(cp.cuda.get_elapsed_time(ev2, ev3))
            apply_ms += float(cp.cuda.get_elapsed_time(ev3, ev4))
            onee_ms += float(cp.cuda.get_elapsed_time(ev4, ev5))
            total_ms += float(cp.cuda.get_elapsed_time(ev0, ev5))

        prof = {
            "tiles": float(len(stage_events)),
            "offdiag_w_s": 1e-3 * offdiag_ms,
            "diag_w_s": 1e-3 * diag_ms,
            "gemm_s": 1e-3 * gemm_ms,
            "apply_s": 1e-3 * apply_ms,
            "onee_s": 1e-3 * onee_ms,
            "total_s": 1e-3 * total_ms,
        }

        workspace.profile_calls += int(nvec)
        workspace.profile_last = prof
        for k, v in prof.items():
            workspace.profile_total[k] = float(workspace.profile_total.get(k, 0.0)) + float(v)

    if copy_back:
        y[...] = y_f
        return y
    return y_f


def hop_cuda_epq_table(
    drt,
    drt_dev,
    state_dev,
    epq_table,
    h_eff,
    eri_mat_t,
    x,
    y=None,
    *,
    l_full=None,
    workspace: CudaMrciHopWorkspace | None = None,
    tile_csf: int | None = None,
    dtype=None,
    sync: bool | None = None,
    check_overflow: bool | None = None,
    sym_pair: bool | None = None,
):
    """Uncontracted MRCI sigma build on GPU using the dense-intermediate method.

    Parameters
    ----------
    x : cp.ndarray[ncsf] or cp.ndarray[ncsf,nvec]
        Input vector(s). The 2D path is currently a per-vector loop (but reuses the tiled workspace).
    l_full : cp.ndarray[nops,naux] | None
        DF/Cholesky factor for the 2e term. When provided, the hop avoids materializing `eri_mat_t` and computes:
          Z = W @ l_full
          G = 0.5 * Z @ l_full.T
    workspace : CudaMrciHopWorkspace | None
        If provided, reused across hops to avoid allocating W/G buffers per call.
    tile_csf : int | None
        Max CSFs per tile. Defaults to `workspace.max_tile_csf` or an auto heuristic when `workspace is None`.
    dtype : cp.dtype | str | None
        Floating-point type for hop buffers and contractions (`float64` default, `float32` optional).
    sync, check_overflow : bool | None
        Debug options. Defaults are driven by `CUGUGA_MRCI_CUDA_CHECK_OVERFLOW`.
    """
    import cupy as cp

    if check_overflow is None:
        check_overflow = bool(_env_int("CUGUGA_MRCI_CUDA_CHECK_OVERFLOW", 0))
    if sync is None:
        sync = bool(check_overflow)
    if bool(check_overflow) and not bool(sync):
        raise ValueError("check_overflow=True requires sync=True")

    profile_stages = bool(_env_int("CUGUGA_MRCI_CUDA_HOP_PROFILE", 0))
    profile_stage_sync = bool(_env_int("CUGUGA_MRCI_CUDA_HOP_PROFILE_SYNC", 1))
    nvec_group = int(_env_int("CUGUGA_MRCI_CUDA_NVEC_GROUP", 4))
    if sym_pair is None:
        sym_pair_env = str(os.getenv("CUGUGA_MRCI_CUDA_SYM_PAIR", "auto")).strip().lower()
        if sym_pair_env in ("1", "true", "yes", "on"):
            sym_pair = True
        elif sym_pair_env in ("0", "false", "no", "off"):
            sym_pair = False
        else:
            use_df = bool(l_full is not None)
            sym_pair = bool(14 <= int(drt.norb) <= 16 and (use_df or int(drt.ncsf) >= 200_000))
    sym_pair = bool(sym_pair)
    build_threads = max(1, min(1024, int(_env_int("CUGUGA_MRCI_CUDA_BUILD_THREADS", 256))))
    diag_threads = max(1, min(1024, int(_env_int("CUGUGA_MRCI_CUDA_DIAG_THREADS", 256))))
    apply_threads = max(1, min(1024, int(_env_int("CUGUGA_MRCI_CUDA_APPLY_THREADS", 32))))

    fp_dtype = cp.float64 if dtype is None else cp.dtype(dtype)
    if fp_dtype not in (cp.float32, cp.float64):
        raise ValueError("dtype must be float32 or float64")

    x = cp.asarray(x, dtype=fp_dtype)
    if x.ndim == 1:
        if y is None:
            y = cp.empty_like(x)
        return _hop_cuda_epq_table_1d_tiled(
            drt=drt,
            drt_dev=drt_dev,
            state_dev=state_dev,
            epq_table=epq_table,
            h_eff=h_eff,
            eri_mat_t=eri_mat_t,
            l_full=l_full,
            x=x,
            y=y,
            workspace=workspace,
            tile_csf=tile_csf,
            fp_dtype=fp_dtype,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            profile_stages=profile_stages,
            profile_stage_sync=profile_stage_sync,
            sym_pair=bool(sym_pair),
            build_threads=build_threads,
            diag_threads=diag_threads,
            apply_threads=apply_threads,
        )

    if x.ndim != 2:
        raise ValueError("x must be 1D (ncsf,) or 2D (ncsf,nvec)")

    return _hop_cuda_epq_table_2d_block_tiled(
        drt=drt,
        drt_dev=drt_dev,
        state_dev=state_dev,
        epq_table=epq_table,
        h_eff=h_eff,
        eri_mat_t=eri_mat_t,
        l_full=l_full,
        x=x,
        y=y,
        workspace=workspace,
        tile_csf=tile_csf,
        fp_dtype=fp_dtype,
        sync=bool(sync),
        check_overflow=bool(check_overflow),
        profile_stages=profile_stages,
        profile_stage_sync=profile_stage_sync,
        nvec_group=nvec_group,
        sym_pair=bool(sym_pair),
        build_threads=build_threads,
        diag_threads=diag_threads,
        apply_threads=apply_threads,
    )


def hop_cuda_projected(
    drt_full,
    drt_dev_full,
    state_dev_full,
    epq_table_full,
    h_eff,
    eri_mat_t_full,
    x_sub,
    sub_to_full,
    y_sub=None,
    x_full_buf=None,
    y_full_buf=None,
    *,
    l_full_full=None,
    workspace_full: CudaMrciHopWorkspace | None = None,
    tile_csf: int | None = None,
    sync: bool | None = None,
    check_overflow: bool | None = None,
    sym_pair: bool | None = None,
):
    """Projected MRCI sigma build on GPU: y = (P H_full P) x."""

    import cupy as cp
    from asuka.cuda.cuda_backend import (
        gather_project_inplace_device,
        scatter_embed_inplace_device,
    )

    if check_overflow is None:
        check_overflow = bool(_env_int("CUGUGA_MRCI_CUDA_CHECK_OVERFLOW", 0))
    if sync is None:
        sync = bool(check_overflow)
    if bool(check_overflow) and not bool(sync):
        raise ValueError("check_overflow=True requires sync=True")

    x_sub = cp.asarray(x_sub, dtype=cp.float64)
    ncsf_full = int(drt_full.ncsf)
    ncsf_sub = int(x_sub.shape[0])

    if x_full_buf is None:
        x_full_buf = cp.empty((ncsf_full,), dtype=cp.float64)
    else:
        x_full_buf = cp.asarray(x_full_buf, dtype=cp.float64).ravel()
        if x_full_buf.shape != (ncsf_full,):
            raise ValueError("x_full_buf must have shape (ncsf_full,)")

    if y_full_buf is None:
        y_full_buf = cp.empty((ncsf_full,), dtype=cp.float64)
    else:
        y_full_buf = cp.asarray(y_full_buf, dtype=cp.float64).ravel()
        if y_full_buf.shape != (ncsf_full,):
            raise ValueError("y_full_buf must have shape (ncsf_full,)")

    if x_sub.ndim == 1:
        if y_sub is None:
            y_sub = cp.empty((ncsf_sub,), dtype=cp.float64)
        else:
            y_sub = cp.asarray(y_sub, dtype=cp.float64).ravel()
            if y_sub.shape != (ncsf_sub,):
                raise ValueError("y_sub must have shape (ncsf_sub,)")

        x_full_buf.fill(0)
        scatter_embed_inplace_device(x_sub, sub_to_full, x_full_buf)

        hop_cuda_epq_table(
            drt=drt_full,
            drt_dev=drt_dev_full,
            state_dev=state_dev_full,
            epq_table=epq_table_full,
            h_eff=h_eff,
            eri_mat_t=eri_mat_t_full,
            l_full=l_full_full,
            x=x_full_buf,
            y=y_full_buf,
            workspace=workspace_full,
            tile_csf=tile_csf,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            sym_pair=sym_pair,
        )

        gather_project_inplace_device(y_full_buf, sub_to_full, y_sub)
        return y_sub

    if x_sub.ndim != 2:
        raise ValueError("x_sub must be 1D (ncsf_sub,) or 2D (ncsf_sub,nvec)")

    _ncsf_sub2, nvec = map(int, x_sub.shape)
    if _ncsf_sub2 != ncsf_sub:
        raise ValueError("invalid x_sub shape")

    if y_sub is None:
        y_sub = cp.empty_like(x_sub)
    else:
        y_sub = cp.asarray(y_sub, dtype=cp.float64)
        if y_sub.shape != x_sub.shape:
            raise ValueError("y_sub must have the same shape as x_sub")

    if getattr(x_sub, "flags", None) and x_sub.flags.f_contiguous:
        vec = lambda v: x_sub[:, v]
    else:
        x_vecs = cp.ascontiguousarray(x_sub.T)  # (nvec,ncsf_sub)
        vec = lambda v: x_vecs[v]

    y_tmp = cp.empty((ncsf_sub,), dtype=cp.float64)
    for v in range(nvec):
        x_full_buf.fill(0)
        scatter_embed_inplace_device(vec(v), sub_to_full, x_full_buf)

        hop_cuda_epq_table(
            drt=drt_full,
            drt_dev=drt_dev_full,
            state_dev=state_dev_full,
            epq_table=epq_table_full,
            h_eff=h_eff,
            eri_mat_t=eri_mat_t_full,
            l_full=l_full_full,
            x=x_full_buf,
            y=y_full_buf,
            workspace=workspace_full,
            tile_csf=tile_csf,
            sync=bool(sync),
            check_overflow=bool(check_overflow),
            sym_pair=sym_pair,
        )

        gather_project_inplace_device(y_full_buf, sub_to_full, y_tmp)
        y_sub[:, v] = y_tmp

    return y_sub
