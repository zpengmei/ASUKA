from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # optional
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore


def _contiguous_block_ranges(n: int, block_size: int) -> list[tuple[int, int]]:
    n = int(n)
    block_size = max(1, int(block_size))
    return [(lo, min(lo + block_size, n)) for lo in range(0, n, block_size)]


def _bucket_aligned_block_ranges(
    bucket_starts_h: np.ndarray,
    bucket_sizes_h: np.ndarray,
    block_size: int,
) -> list[tuple[int, int]]:
    starts_h = np.asarray(bucket_starts_h, dtype=np.int64).ravel()
    sizes_h = np.asarray(bucket_sizes_h, dtype=np.int64).ravel()
    block_size = max(1, int(block_size))
    if int(starts_h.size) == 0:
        return []
    out: list[tuple[int, int]] = []
    cur_lo = int(starts_h[0])
    cur_hi = cur_lo
    for start_i, size_i in zip(starts_h.tolist(), sizes_h.tolist(), strict=False):
        start_i = int(start_i)
        size_i = int(size_i)
        if size_i <= 0:
            continue
        if size_i >= block_size:
            if cur_hi > cur_lo:
                out.append((cur_lo, cur_hi))
            for lo in range(start_i, start_i + size_i, block_size):
                out.append((int(lo), int(min(lo + block_size, start_i + size_i))))
            cur_lo = start_i + size_i
            cur_hi = cur_lo
            continue
        if cur_hi == cur_lo:
            cur_lo = start_i
            cur_hi = start_i + size_i
            continue
        if (start_i + size_i - cur_lo) <= block_size:
            cur_hi = start_i + size_i
        else:
            out.append((cur_lo, cur_hi))
            cur_lo = start_i
            cur_hi = start_i + size_i
    if cur_hi > cur_lo:
        out.append((cur_lo, cur_hi))
    return out


def _build_dense_blocks_from_local_coo(
    *,
    n: int,
    block_ranges: list[tuple[int, int]],
    row_h: np.ndarray,
    col_h: np.ndarray,
    hij_h: np.ndarray,
    hdiag_h: np.ndarray,
    symmetric_unique: bool,
) -> list[np.ndarray]:
    hdiag_h = np.asarray(hdiag_h, dtype=np.float64).ravel()
    if int(hdiag_h.size) != int(n):
        raise ValueError("hdiag length must match block space size")
    block_of = np.full((int(n),), -1, dtype=np.int32)
    dense_blocks = []
    for block_id, (lo, hi) in enumerate(block_ranges):
        lo = int(lo)
        hi = int(hi)
        block_of[lo:hi] = int(block_id)
        dense_blocks.append(np.diag(np.asarray(hdiag_h[lo:hi], dtype=np.float64, order="C")))
    row_h = np.asarray(row_h, dtype=np.int32).ravel()
    col_h = np.asarray(col_h, dtype=np.int32).ravel()
    hij_h = np.asarray(hij_h, dtype=np.float64).ravel()
    for row_i, col_i, hij_i in zip(row_h.tolist(), col_h.tolist(), hij_h.tolist(), strict=False):
        row_i = int(row_i)
        col_i = int(col_i)
        block_i = int(block_of[row_i]) if 0 <= row_i < int(n) else -1
        if block_i < 0 or block_i != int(block_of[col_i]):
            continue
        lo, _ = block_ranges[block_i]
        rr = int(row_i - lo)
        cc = int(col_i - lo)
        dense_blocks[block_i][rr, cc] += float(hij_i)
        if bool(symmetric_unique) and row_i != col_i:
            dense_blocks[block_i][cc, rr] += float(hij_i)
    if not bool(symmetric_unique):
        for block_i, block_h in enumerate(dense_blocks):
            dense_blocks[block_i] = np.asarray(0.5 * (block_h + block_h.T), dtype=np.float64, order="C")
    return dense_blocks


@dataclass
class ShiftedSpectralBlockPreconditioner:
    n: int
    block_ranges: tuple[tuple[int, int], ...]
    eigvecs_d: tuple[Any, ...]
    eigvals_d: tuple[Any, ...]
    perm_d: Any | None = None
    inv_perm_d: Any | None = None
    denom_tol: float = 1e-8
    label: str = "spectral_block"

    def apply(self, theta: float, r_d):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ShiftedSpectralBlockPreconditioner") from e

        r_arr = cp.asarray(r_d, dtype=cp.float64).ravel()
        if int(r_arr.size) != int(self.n):
            raise ValueError("preconditioner input has wrong length")
        if self.perm_d is not None:
            work_d = cp.ascontiguousarray(r_arr[self.perm_d].ravel())
        else:
            work_d = cp.ascontiguousarray(r_arr.ravel())
        out_d = cp.zeros_like(work_d)
        theta_f = float(theta)
        floor_f = float(self.denom_tol)
        for (lo, hi), eigvec_d, eigval_d in zip(self.block_ranges, self.eigvecs_d, self.eigvals_d, strict=False):
            rhs_block_d = work_d[int(lo):int(hi)]
            coeff_d = eigvec_d.T @ rhs_block_d
            denom_d = cp.asarray(eigval_d, dtype=cp.float64).ravel() - theta_f
            sign_d = cp.where(denom_d >= 0.0, 1.0, -1.0)
            denom_d = cp.where(cp.abs(denom_d) < floor_f, sign_d * floor_f, denom_d)
            out_d[int(lo):int(hi)] = eigvec_d @ cp.asarray(coeff_d / denom_d, dtype=cp.float64)
        if self.inv_perm_d is not None:
            return cp.ascontiguousarray(out_d[self.inv_perm_d].ravel())
        return out_d


def _build_shifted_spectral_preconditioner(
    *,
    n: int,
    block_ranges: list[tuple[int, int]],
    dense_blocks_h: list[np.ndarray],
    cp,
    perm_d=None,
    inv_perm_d=None,
    denom_tol: float = 1e-8,
    label: str,
) -> ShiftedSpectralBlockPreconditioner:
    eigvecs_d: list[Any] = []
    eigvals_d: list[Any] = []
    for block_h in dense_blocks_h:
        block_h = np.asarray(block_h, dtype=np.float64, order="C")
        if int(block_h.shape[0]) != int(block_h.shape[1]):
            raise ValueError("dense preconditioner blocks must be square")
        evals_h, evecs_h = np.linalg.eigh(block_h)
        eigvals_d.append(cp.ascontiguousarray(cp.asarray(evals_h, dtype=cp.float64).ravel()))
        eigvecs_d.append(cp.ascontiguousarray(cp.asarray(evecs_h, dtype=cp.float64)))
    return ShiftedSpectralBlockPreconditioner(
        n=int(n),
        block_ranges=tuple((int(lo), int(hi)) for lo, hi in block_ranges),
        eigvecs_d=tuple(eigvecs_d),
        eigvals_d=tuple(eigvals_d),
        perm_d=None if perm_d is None else cp.ascontiguousarray(cp.asarray(perm_d, dtype=cp.int32).ravel()),
        inv_perm_d=None if inv_perm_d is None else cp.ascontiguousarray(cp.asarray(inv_perm_d, dtype=cp.int32).ravel()),
        denom_tol=float(denom_tol),
        label=str(label),
    )


@dataclass
class ExactSelectedProjectedHop:
    """Exact selected-space projected Hamiltonian apply.

    This is the Batch-1 foundation for scalable SCI projected solves:
    it stores the exact selected-space operator `P_S H P_S` and provides
    a uniform host/GPU hop interface.

    The first implementation is backed by the exact selected-space CSR matrix.
    That keeps the operator exact while giving the SCI driver one stable
    projected-hop abstraction to target before the later on-the-fly kernel work.
    """

    sel_idx: np.ndarray
    h_csr: Any
    hdiag: np.ndarray
    _gpu_sell_ws: Any | None = None

    @classmethod
    def from_csr(cls, sel_idx: np.ndarray, h_csr: Any) -> "ExactSelectedProjectedHop":
        if sp is None:  # pragma: no cover
            raise RuntimeError("scipy is required for ExactSelectedProjectedHop")
        h_csr = h_csr.tocsr()
        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        if int(h_csr.shape[0]) != int(h_csr.shape[1]):
            raise ValueError("selected-space operator must be square")
        if int(h_csr.shape[0]) != int(sel_idx.size):
            raise ValueError("selected-space operator shape must match sel_idx length")
        hdiag = np.asarray(h_csr.diagonal(), dtype=np.float64).ravel()
        return cls(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            h_csr=h_csr,
            hdiag=np.asarray(hdiag, dtype=np.float64, order="C"),
        )

    def hop_host(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64, order="C")
        if int(getattr(x, "ndim", 1)) == 1:
            if int(x.size) != int(self.sel_idx.size):
                raise ValueError("host projected-hop input has wrong length")
            y = self.h_csr @ x
            return np.asarray(y, dtype=np.float64, order="C")
        if int(x.ndim) != 2:
            raise ValueError("host projected-hop input must be 1D or 2D")
        if int(x.shape[0]) != int(self.sel_idx.size):
            raise ValueError("host projected-hop input has wrong leading dimension")
        y = self.h_csr @ x
        return np.asarray(y, dtype=np.float64, order="C")

    def _ensure_gpu_sell_ws(self):
        if self._gpu_sell_ws is not None:
            return self._gpu_sell_ws
        from asuka.cuda.cuda_backend import GugaMatvecFixedSellWorkspace, _csr_to_sell_host  # noqa: PLC0415

        indptr_h = np.asarray(self.h_csr.indptr, dtype=np.int64)
        indices_h = np.asarray(self.h_csr.indices, dtype=np.int32)
        data_h = np.asarray(self.h_csr.data, dtype=np.float64)
        slice_ptr_h, slice_width_h, col_idx_h, val_h = _csr_to_sell_host(
            indptr_h,
            indices_h,
            data_h,
            nrows=int(self.h_csr.shape[0]),
            slice_height=32,
        )
        self._gpu_sell_ws = GugaMatvecFixedSellWorkspace(
            slice_ptr_h,
            slice_width_h,
            col_idx_h,
            val_h,
            nrows=int(self.h_csr.shape[0]),
            slice_height=32,
            threads_spmv=128,
            threads_spmm=128,
        )
        return self._gpu_sell_ws

    def hop_gpu(self, x_d):
        ws = self._ensure_gpu_sell_ws()
        ndim = int(getattr(x_d, "ndim", 1))
        if ndim == 1:
            return ws.hop(x_d, sync=False)
        if ndim == 2:
            return ws.hop_many(x_d, sync=False)
        raise ValueError("gpu projected-hop input must be 1D or 2D")

    def build_jd_preconditioner(self, *, block_size: int = 64, denom_tol: float = 1e-8):
        if sp is None:  # pragma: no cover
            raise RuntimeError("scipy is required for ExactSelectedProjectedHop")
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedProjectedHop preconditioner") from e

        nsel = int(self.sel_idx.size)
        block_ranges = _contiguous_block_ranges(nsel, int(block_size))
        dense_blocks_h = [
            np.asarray(self.h_csr[int(lo):int(hi), int(lo):int(hi)].toarray(), dtype=np.float64, order="C")
            for lo, hi in block_ranges
        ]
        return _build_shifted_spectral_preconditioner(
            n=nsel,
            block_ranges=block_ranges,
            dense_blocks_h=dense_blocks_h,
            cp=cp,
            denom_tol=float(denom_tol),
            label="csr_contiguous_blocks",
        )


@dataclass
class ExactSelectedTupleProjectedHop:
    """Exact selected-space projected hop backed by cached device edge tuples.

    This operator stores the exact selected-only edges for the current selected
    space on device and applies `P_S H P_S` without materializing the explicit
    selected-space matrix. The edge list is static across Davidson iterations
    for a fixed selected space, so the hop state remains device-resident during
    the solve.
    """

    sel_idx: np.ndarray
    target_local_d: Any
    starts_d: Any
    src_sorted_d: Any
    hij_sorted_d: Any
    hdiag_d: Any | None = None
    coo_target_local_d: Any | None = None
    coo_src_pos_d: Any | None = None
    coo_hij_d: Any | None = None

    @classmethod
    def from_local_tuples(
        cls,
        *,
        sel_idx: np.ndarray,
        target_local,
        src_pos,
        hij,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedTupleProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        target_local_d = cp.ascontiguousarray(cp.asarray(target_local, dtype=cp.int32).ravel())
        src_pos_d = cp.ascontiguousarray(cp.asarray(src_pos, dtype=cp.int32).ravel())
        hij_d = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())

        if int(target_local_d.size) == 0:
            return cls(
                sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                target_local_d=cp.zeros((0,), dtype=cp.int32),
                starts_d=cp.zeros((0,), dtype=cp.int64),
                src_sorted_d=cp.zeros((0,), dtype=cp.int32),
                hij_sorted_d=cp.zeros((0,), dtype=cp.float64),
                hdiag_d=None if hdiag is None else cp.ascontiguousarray(cp.asarray(hdiag, dtype=cp.float64).ravel()),
                coo_target_local_d=cp.zeros((0,), dtype=cp.int32),
                coo_src_pos_d=cp.zeros((0,), dtype=cp.int32),
                coo_hij_d=cp.zeros((0,), dtype=cp.float64),
            )

        # The dense exact emitter produces many duplicate (target, source) pairs.
        # Canonicalize them here so the cached projected hop stores reduced COO data.
        pair_key_d = (
            (target_local_d.astype(cp.uint64, copy=False) << cp.uint64(32))
            | src_pos_d.astype(cp.uint32, copy=False).astype(cp.uint64, copy=False)
        )
        order_d = cp.asarray(cp.argsort(pair_key_d), dtype=cp.int64).ravel()
        pair_key_sorted_d = cp.ascontiguousarray(pair_key_d[order_d].ravel())
        hij_pair_sorted_d = cp.ascontiguousarray(hij_d[order_d].ravel())

        if int(pair_key_sorted_d.size) == 1:
            pair_starts_d = cp.asarray([0], dtype=cp.int64)
        else:
            pair_boundaries = cp.nonzero(pair_key_sorted_d[1:] != pair_key_sorted_d[:-1])[0] + 1
            pair_starts_d = cp.ascontiguousarray(
                cp.concatenate(
                    (
                        cp.asarray([0], dtype=cp.int64),
                        cp.asarray(pair_boundaries, dtype=cp.int64).ravel(),
                    )
                ).ravel()
            )
        pair_unique_d = cp.ascontiguousarray(pair_key_sorted_d[pair_starts_d].ravel())
        hij_reduced_d = cp.ascontiguousarray(cp.add.reduceat(hij_pair_sorted_d, pair_starts_d))
        target_sorted_d = cp.ascontiguousarray((pair_unique_d >> cp.uint64(32)).astype(cp.int32, copy=False).ravel())
        src_sorted_d = cp.ascontiguousarray((pair_unique_d & cp.uint64(0xFFFFFFFF)).astype(cp.int32, copy=False).ravel())
        hij_sorted_d = hij_reduced_d

        if int(target_sorted_d.size) == 1:
            starts_d = cp.asarray([0], dtype=cp.int64)
            target_unique_d = target_sorted_d
        else:
            boundaries = cp.nonzero(target_sorted_d[1:] != target_sorted_d[:-1])[0] + 1
            starts_d = cp.ascontiguousarray(
                cp.concatenate(
                    (
                        cp.asarray([0], dtype=cp.int64),
                        cp.asarray(boundaries, dtype=cp.int64).ravel(),
                    )
                ).ravel()
            )
            target_unique_d = cp.ascontiguousarray(target_sorted_d[starts_d].ravel())

        return cls(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            target_local_d=target_unique_d,
            starts_d=starts_d,
            src_sorted_d=src_sorted_d,
            hij_sorted_d=hij_sorted_d,
            hdiag_d=None if hdiag is None else cp.ascontiguousarray(cp.asarray(hdiag, dtype=cp.float64).ravel()),
            coo_target_local_d=target_sorted_d,
            coo_src_pos_d=src_sorted_d,
            coo_hij_d=hij_sorted_d,
        )

    @classmethod
    def from_tuples(
        cls,
        *,
        sel_idx: np.ndarray,
        labels,
        src_pos,
        hij,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedTupleProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        labels_d = cp.ascontiguousarray(cp.asarray(labels, dtype=cp.uint64).ravel())
        src_pos_d = cp.ascontiguousarray(cp.asarray(src_pos, dtype=cp.int32).ravel())
        hij_d = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())

        if int(labels_d.size) == 0:
            return cls.from_local_tuples(
                sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
                target_local=cp.zeros((0,), dtype=cp.int32),
                src_pos=cp.zeros((0,), dtype=cp.int32),
                hij=cp.zeros((0,), dtype=cp.float64),
                hdiag=hdiag,
            )

        sel_u64_h = np.asarray(sel_idx.astype(np.uint64, copy=False), dtype=np.uint64)
        sort_order_h = np.argsort(sel_u64_h, kind="stable").astype(np.int32, copy=False)
        sel_sorted_d = cp.ascontiguousarray(cp.asarray(sel_u64_h[sort_order_h], dtype=cp.uint64).ravel())
        sel_sorted_to_local_d = cp.ascontiguousarray(cp.asarray(sort_order_h, dtype=cp.int32).ravel())
        pos_d = cp.asarray(cp.searchsorted(sel_sorted_d, labels_d), dtype=cp.int32).ravel()
        matched = (
            (pos_d >= 0)
            & (pos_d < int(sel_sorted_d.size))
            & (sel_sorted_d[pos_d] == labels_d)
        )
        if not bool(cp.all(matched).item()):
            raise RuntimeError("selected tuple emitter returned labels outside the selected space")
        return cls.from_local_tuples(
            sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
            target_local=cp.ascontiguousarray(sel_sorted_to_local_d[pos_d].ravel()),
            src_pos=src_pos_d,
            hij=hij_d,
            hdiag=hdiag,
        )

    @classmethod
    def from_tuple_emitter(
        cls,
        *,
        sel_idx: np.ndarray,
        tuple_emitter: Any,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedTupleProjectedHop":
        labels_d, src_pos_d, hij_d = tuple_emitter(sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel())
        return cls.from_tuples(
            sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
            labels=labels_d,
            src_pos=src_pos_d,
            hij=hij_d,
            hdiag=hdiag,
        )

    def hop_gpu(self, x_d):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        nsel = int(self.sel_idx.size)
        x_d = cp.asarray(x_d, dtype=cp.float64)
        if int(getattr(x_d, "ndim", 1)) == 1:
            if int(x_d.size) != nsel:
                raise ValueError("gpu projected-hop input has wrong length")
            y_d = cp.zeros((nsel,), dtype=cp.float64)
            if self.hdiag_d is not None:
                y_d += cp.asarray(self.hdiag_d, dtype=cp.float64) * x_d
            if int(self.hij_sorted_d.size) == 0:
                return y_d
            contrib_d = cp.ascontiguousarray(self.hij_sorted_d * x_d[self.src_sorted_d])
            reduced_d = cp.ascontiguousarray(cp.add.reduceat(contrib_d, self.starts_d))
            y_d[self.target_local_d] += reduced_d
            return y_d
        if int(x_d.ndim) != 2:
            raise ValueError("gpu projected-hop input must be 1D or 2D")
        if int(x_d.shape[0]) != nsel:
            raise ValueError("gpu projected-hop input has wrong leading dimension")
        nvec = int(x_d.shape[1])
        y_d = cp.zeros((nsel, nvec), dtype=cp.float64)
        if self.hdiag_d is not None:
            y_d += cp.asarray(self.hdiag_d, dtype=cp.float64)[:, None] * x_d
        if int(self.hij_sorted_d.size) == 0:
            return y_d
        contrib_d = cp.ascontiguousarray(self.hij_sorted_d[:, None] * x_d[self.src_sorted_d, :])
        reduced_d = cp.ascontiguousarray(cp.add.reduceat(contrib_d, self.starts_d, axis=0))
        y_d[self.target_local_d, :] += reduced_d
        return y_d

    def to_local_tuples(self):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        nnz = int(self.hij_sorted_d.size)
        if nnz == 0:
            return (
                cp.zeros((0,), dtype=cp.int32),
                cp.zeros((0,), dtype=cp.int32),
                cp.zeros((0,), dtype=cp.float64),
            )
        if self.coo_target_local_d is not None and self.coo_src_pos_d is not None and self.coo_hij_d is not None:
            return (
                cp.ascontiguousarray(cp.asarray(self.coo_target_local_d, dtype=cp.int32).ravel()),
                cp.ascontiguousarray(cp.asarray(self.coo_src_pos_d, dtype=cp.int32).ravel()),
                cp.ascontiguousarray(cp.asarray(self.coo_hij_d, dtype=cp.float64).ravel()),
            )
        bounds = cp.concatenate(
            (
                cp.ascontiguousarray(self.starts_d.ravel()),
                cp.asarray([int(nnz)], dtype=cp.int64),
            )
        )
        counts = cp.diff(bounds)
        target_sorted_d = cp.repeat(
            cp.ascontiguousarray(self.target_local_d.ravel()),
            cp.asnumpy(counts.astype(cp.int32, copy=False)),
        )
        return (
            cp.ascontiguousarray(target_sorted_d.ravel()),
            cp.ascontiguousarray(self.src_sorted_d.ravel()),
            cp.ascontiguousarray(self.hij_sorted_d.ravel()),
        )

    def with_appended_local_tuples(
        self,
        *,
        sel_idx: np.ndarray,
        target_local,
        src_pos,
        hij,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedTupleProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        old_target_d, old_src_d, old_hij_d = self.to_local_tuples()
        add_target_d = cp.ascontiguousarray(cp.asarray(target_local, dtype=cp.int32).ravel())
        add_src_d = cp.ascontiguousarray(cp.asarray(src_pos, dtype=cp.int32).ravel())
        add_hij_d = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())
        if int(add_target_d.size) == 0:
            return type(self).from_local_tuples(
                sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
                target_local=old_target_d,
                src_pos=old_src_d,
                hij=old_hij_d,
                hdiag=hdiag,
            )
        return type(self).from_local_tuples(
            sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
            target_local=cp.concatenate((old_target_d, add_target_d)),
            src_pos=cp.concatenate((old_src_d, add_src_d)),
            hij=cp.concatenate((old_hij_d, add_hij_d)),
            hdiag=hdiag,
        )

    def with_hdiag(self, *, hdiag: np.ndarray | None = None) -> "ExactSelectedTupleProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        sel_idx = np.asarray(self.sel_idx, dtype=np.int64, order="C")
        return type(self)(
            sel_idx=sel_idx,
            target_local_d=cp.ascontiguousarray(self.target_local_d.ravel()),
            starts_d=cp.ascontiguousarray(self.starts_d.ravel()),
            src_sorted_d=cp.ascontiguousarray(self.src_sorted_d.ravel()),
            hij_sorted_d=cp.ascontiguousarray(self.hij_sorted_d.ravel()),
            hdiag_d=None if hdiag is None else cp.ascontiguousarray(cp.asarray(hdiag, dtype=cp.float64).ravel()),
            coo_target_local_d=None if self.coo_target_local_d is None else cp.ascontiguousarray(cp.asarray(self.coo_target_local_d, dtype=cp.int32).ravel()),
            coo_src_pos_d=None if self.coo_src_pos_d is None else cp.ascontiguousarray(cp.asarray(self.coo_src_pos_d, dtype=cp.int32).ravel()),
            coo_hij_d=None if self.coo_hij_d is None else cp.ascontiguousarray(cp.asarray(self.coo_hij_d, dtype=cp.float64).ravel()),
        )

    def with_merged_hop(
        self,
        *,
        other: "ExactSelectedTupleProjectedHop",
        sel_idx: np.ndarray,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedTupleProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop") from e

        if int(self.sel_idx.size) == 0:
            return other.with_hdiag(hdiag=hdiag)
        if int(other.sel_idx.size) == 0 or int(other.hij_sorted_d.size) == 0:
            return self.with_hdiag(hdiag=hdiag)

        old_targets_h = np.asarray(cp.asnumpy(self.target_local_d), dtype=np.int32)
        old_starts_h = np.asarray(cp.asnumpy(self.starts_d), dtype=np.int64)
        new_targets_h = np.asarray(cp.asnumpy(other.target_local_d), dtype=np.int32)
        new_starts_h = np.asarray(cp.asnumpy(other.starts_d), dtype=np.int64)
        old_bounds_h = np.concatenate((old_starts_h, np.asarray([int(self.hij_sorted_d.size)], dtype=np.int64)))
        new_bounds_h = np.concatenate((new_starts_h, np.asarray([int(other.hij_sorted_d.size)], dtype=np.int64)))

        merged_targets: list[int] = []
        src_parts: list[Any] = []
        hij_parts: list[Any] = []
        starts: list[int] = [0]
        io = 0
        inew = 0
        nnz = 0
        while io < int(old_targets_h.size) or inew < int(new_targets_h.size):
            old_t = int(old_targets_h[io]) if io < int(old_targets_h.size) else None
            new_t = int(new_targets_h[inew]) if inew < int(new_targets_h.size) else None
            if new_t is None or (old_t is not None and old_t < new_t):
                tgt = int(old_t)
            elif old_t is None or int(new_t) < int(old_t):
                tgt = int(new_t)
            else:
                tgt = int(old_t)
            merged_targets.append(int(tgt))
            if old_t is not None and int(old_t) == int(tgt):
                lo = int(old_bounds_h[io])
                hi = int(old_bounds_h[io + 1])
                if hi > lo:
                    src_parts.append(cp.ascontiguousarray(self.src_sorted_d[lo:hi].ravel()))
                    hij_parts.append(cp.ascontiguousarray(self.hij_sorted_d[lo:hi].ravel()))
                    nnz += int(hi - lo)
                io += 1
            if new_t is not None and int(new_t) == int(tgt):
                lo = int(new_bounds_h[inew])
                hi = int(new_bounds_h[inew + 1])
                if hi > lo:
                    src_parts.append(cp.ascontiguousarray(other.src_sorted_d[lo:hi].ravel()))
                    hij_parts.append(cp.ascontiguousarray(other.hij_sorted_d[lo:hi].ravel()))
                    nnz += int(hi - lo)
                inew += 1
            starts.append(int(nnz))

        src_sorted_d = cp.zeros((0,), dtype=cp.int32) if not src_parts else cp.ascontiguousarray(cp.concatenate(src_parts).ravel())
        hij_sorted_d = cp.zeros((0,), dtype=cp.float64) if not hij_parts else cp.ascontiguousarray(cp.concatenate(hij_parts).ravel())
        raw_target_d, raw_src_d, raw_hij_d = self.to_local_tuples()
        other_target_d, other_src_d, other_hij_d = other.to_local_tuples()
        return type(self)(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            target_local_d=cp.ascontiguousarray(cp.asarray(np.asarray(merged_targets, dtype=np.int32), dtype=cp.int32).ravel()),
            starts_d=cp.ascontiguousarray(cp.asarray(np.asarray(starts[:-1], dtype=np.int64), dtype=cp.int64).ravel()),
            src_sorted_d=src_sorted_d,
            hij_sorted_d=hij_sorted_d,
            hdiag_d=None if hdiag is None else cp.ascontiguousarray(cp.asarray(hdiag, dtype=cp.float64).ravel()),
            coo_target_local_d=cp.ascontiguousarray(cp.concatenate((raw_target_d, other_target_d)).ravel()),
            coo_src_pos_d=cp.ascontiguousarray(cp.concatenate((raw_src_d, other_src_d)).ravel()),
            coo_hij_d=cp.ascontiguousarray(cp.concatenate((raw_hij_d, other_hij_d)).ravel()),
        )

    def build_jd_preconditioner(self, *, block_size: int = 64, denom_tol: float = 1e-8):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedTupleProjectedHop preconditioner") from e

        nsel = int(self.sel_idx.size)
        block_ranges = _contiguous_block_ranges(nsel, int(block_size))
        row_h, col_h, hij_h = self.to_local_tuples()
        dense_blocks_h = _build_dense_blocks_from_local_coo(
            n=nsel,
            block_ranges=block_ranges,
            row_h=np.asarray(cp.asnumpy(row_h), dtype=np.int32),
            col_h=np.asarray(cp.asnumpy(col_h), dtype=np.int32),
            hij_h=np.asarray(cp.asnumpy(hij_h), dtype=np.float64),
            hdiag_h=np.asarray(
                cp.asnumpy(
                    cp.asarray(
                        cp.zeros((nsel,), dtype=cp.float64) if self.hdiag_d is None else self.hdiag_d,
                        dtype=cp.float64,
                    ).ravel()
                ),
                dtype=np.float64,
            ),
            symmetric_unique=False,
        )
        return _build_shifted_spectral_preconditioner(
            n=nsel,
            block_ranges=block_ranges,
            dense_blocks_h=dense_blocks_h,
            cp=cp,
            denom_tol=float(denom_tol),
            label="tuple_contiguous_blocks",
        )


@dataclass
class DenseMatrixProjectedHop:
    """Dense matrix projected hop backed by a full H[nsel, nsel] matrix on device.

    This operator stores the full selected-space Hamiltonian as a dense matrix
    and applies `P_S H P_S` via cuBLAS dense matvec. Drop-in replacement for
    ExactSelectedTupleProjectedHop when the selected space is small enough to
    fit in GPU memory as a dense matrix (nsel <= ~25K on 24GB GPU).

    Built by the pair-wise H[i,j] evaluation kernel which directly computes
    matrix elements between selected CSFs without DFS-walking the full space.
    """

    sel_idx: np.ndarray
    H_d: Any           # cupy [nsel, nsel] float64
    hdiag_d: Any        # cupy [nsel] float64

    def hop_gpu(self, x_d):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for DenseMatrixProjectedHop") from e

        nsel = int(self.sel_idx.size)
        x_d = cp.asarray(x_d, dtype=cp.float64)
        if int(getattr(x_d, "ndim", 1)) == 1:
            if int(x_d.size) != nsel:
                raise ValueError("gpu projected-hop input has wrong length")
            return cp.ascontiguousarray(self.H_d @ x_d)
        if int(x_d.ndim) != 2:
            raise ValueError("gpu projected-hop input must be 1D or 2D")
        if int(x_d.shape[0]) != nsel:
            raise ValueError("gpu projected-hop input has wrong leading dimension")
        return cp.ascontiguousarray(self.H_d @ x_d)

    def build_jd_preconditioner(self, *, block_size: int = 64, denom_tol: float = 1e-8):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for DenseMatrixProjectedHop preconditioner") from e

        nsel = int(self.sel_idx.size)
        block_ranges = _contiguous_block_ranges(nsel, int(block_size))
        dense_blocks_h = [
            np.asarray(cp.asnumpy(self.H_d[int(lo):int(hi), int(lo):int(hi)]), dtype=np.float64, order="C")
            for lo, hi in block_ranges
        ]
        return _build_shifted_spectral_preconditioner(
            n=nsel,
            block_ranges=block_ranges,
            dense_blocks_h=dense_blocks_h,
            cp=cp,
            denom_tol=float(denom_tol),
            label="dense_contiguous_blocks",
        )


@dataclass
class ExactSelectedPairwiseSigmaProjectedHop:
    """Exact selected-space projected hop backed by the bucketed pairwise sigma kernel.

    This keeps the selected operator matrix-free: the selected labels are
    materialized and bucketed once, and every Davidson hop applies the exact
    bucketed pairwise traversal directly without building dense H or emitting
    selected tuples.
    """

    sel_idx: np.ndarray
    drt: Any
    drt_dev: Any
    sel_sorted_u64_d: Any
    sort_perm_d: Any
    inv_perm_d: Any
    h_base_d: Any
    eri4_d: Any
    materialized_sorted: tuple[Any, Any, Any, Any]
    bucket_data: dict[str, Any]
    hdiag_d: Any | None = None
    threads: int = 256

    @classmethod
    def from_selected_space(
        cls,
        *,
        drt,
        drt_dev,
        sel_idx: np.ndarray,
        h_base_d,
        eri4_d,
        cp,
        build_exact_diag: bool = False,
        threads: int = 256,
    ) -> "ExactSelectedPairwiseSigmaProjectedHop":
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            pairwise_diag_bucketed_u64_device,
            pairwise_build_bucket_data,
            pairwise_materialize_u64_device,
        )

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
        materialized = pairwise_materialize_u64_device(
            drt,
            drt_dev,
            sel_u64_d,
            int(sel_idx.size),
            cp,
            threads=int(threads),
            sync=False,
        )
        steps_all, nodes_all, occ_all, b_all = materialized
        bucket_data_full = pairwise_build_bucket_data(occ_all, int(drt.norb), cp)
        sort_perm_d = cp.ascontiguousarray(bucket_data_full["sort_perm"].astype(cp.int32, copy=False))
        inv_perm_d = cp.ascontiguousarray(bucket_data_full["inv_perm"].astype(cp.int32, copy=False))
        materialized_sorted = (
            cp.ascontiguousarray(steps_all[sort_perm_d]),
            cp.ascontiguousarray(nodes_all[sort_perm_d]),
            cp.ascontiguousarray(occ_all[sort_perm_d]),
            cp.ascontiguousarray(b_all[sort_perm_d]),
        )
        sel_sorted_u64_d = cp.ascontiguousarray(sel_u64_d[sort_perm_d])
        bucket_data = {
            "occ_keys_sorted": cp.ascontiguousarray(bucket_data_full["occ_keys_sorted"]),
            "bucket_keys": cp.ascontiguousarray(bucket_data_full["bucket_keys"]),
            "csf_to_bucket": cp.ascontiguousarray(bucket_data_full["csf_to_bucket"]),
            "bucket_starts": cp.ascontiguousarray(bucket_data_full["bucket_starts"]),
            "bucket_sizes": cp.ascontiguousarray(bucket_data_full["bucket_sizes"]),
            "neighbor_offsets": cp.ascontiguousarray(bucket_data_full["neighbor_offsets"]),
            "neighbor_list": cp.ascontiguousarray(bucket_data_full["neighbor_list"]),
            "target_offsets": cp.ascontiguousarray(bucket_data_full["target_offsets"]),
            "target_list": cp.ascontiguousarray(bucket_data_full["target_list"]),
            "target_offsets_1b": cp.ascontiguousarray(bucket_data_full["target_offsets_1b"]),
            "target_list_1b": cp.ascontiguousarray(bucket_data_full["target_list_1b"]),
        }
        hdiag_d = None
        if bool(build_exact_diag):
            hdiag_sorted_d = pairwise_diag_bucketed_u64_device(
                drt,
                drt_dev,
                sel_sorted_u64_d,
                int(sel_idx.size),
                h_base_d,
                eri4_d,
                materialized_sorted,
                bucket_data,
                cp,
                threads=int(threads),
                sync=False,
            )
            hdiag_d = cp.ascontiguousarray(hdiag_sorted_d[inv_perm_d].ravel())
            del hdiag_sorted_d
        del materialized, steps_all, nodes_all, occ_all, b_all, bucket_data_full, sel_u64_d
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return cls(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            drt=drt,
            drt_dev=drt_dev,
            sel_sorted_u64_d=sel_sorted_u64_d,
            sort_perm_d=sort_perm_d,
            inv_perm_d=inv_perm_d,
            h_base_d=cp.ascontiguousarray(cp.asarray(h_base_d, dtype=cp.float64).ravel()),
            eri4_d=cp.ascontiguousarray(cp.asarray(eri4_d, dtype=cp.float64).ravel()),
            materialized_sorted=materialized_sorted,
            bucket_data=bucket_data,
            hdiag_d=hdiag_d,
            threads=int(threads),
        )

    def hop_gpu(self, x_d):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedPairwiseSigmaProjectedHop") from e

        from asuka.cuda.cuda_backend import pairwise_sigma_bucketed_u64_device  # noqa: PLC0415

        nsel = int(self.sel_idx.size)
        x_arr = cp.asarray(x_d, dtype=cp.float64)
        if int(getattr(x_arr, "ndim", 1)) == 1:
            if int(x_arr.size) != nsel:
                raise ValueError("gpu projected-hop input has wrong length")
            x_sorted = cp.ascontiguousarray(x_arr[self.sort_perm_d].ravel())
            y_sorted = pairwise_sigma_bucketed_u64_device(
                self.drt,
                self.drt_dev,
                self.sel_sorted_u64_d,
                nsel,
                self.h_base_d,
                self.eri4_d,
                self.materialized_sorted,
                self.bucket_data,
                x_sorted,
                cp,
                threads=int(self.threads),
                sync=False,
            )
            return cp.ascontiguousarray(y_sorted[self.inv_perm_d].ravel())
        if int(x_arr.ndim) != 2:
            raise ValueError("gpu projected-hop input must be 1D or 2D")
        if int(x_arr.shape[0]) != nsel:
            raise ValueError("gpu projected-hop input has wrong leading dimension")
        x_sorted = cp.ascontiguousarray(x_arr[self.sort_perm_d, :])
        y_sorted = pairwise_sigma_bucketed_u64_device(
            self.drt,
            self.drt_dev,
            self.sel_sorted_u64_d,
            nsel,
            self.h_base_d,
            self.eri4_d,
            self.materialized_sorted,
            self.bucket_data,
            x_sorted,
            cp,
            threads=int(self.threads),
            sync=False,
        )
        return cp.ascontiguousarray(y_sorted[self.inv_perm_d, :])

    def build_jd_preconditioner(self, *, block_size: int = 64, denom_tol: float = 1e-8):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedPairwiseSigmaProjectedHop preconditioner") from e

        from asuka.cuda.cuda_backend import pairwise_build_selected_graph_bucketed_u64_device  # noqa: PLC0415

        nsel = int(self.sel_idx.size)
        bucket_starts = self.bucket_data.get("bucket_starts")
        bucket_sizes = self.bucket_data.get("bucket_sizes")
        if bucket_starts is not None and bucket_sizes is not None:
            block_ranges = _bucket_aligned_block_ranges(
                np.asarray(cp.asnumpy(bucket_starts), dtype=np.int64),
                np.asarray(cp.asnumpy(bucket_sizes), dtype=np.int64),
                int(block_size),
            )
        else:
            block_ranges = _contiguous_block_ranges(nsel, int(block_size))
        row_h, col_h, hij_h, graph_diag_d, _row_counts_d = pairwise_build_selected_graph_bucketed_u64_device(
            self.drt,
            self.drt_dev,
            self.sel_sorted_u64_d,
            int(nsel),
            self.h_base_d,
            self.eri4_d,
            self.materialized_sorted,
            self.bucket_data,
            cp,
            threads=int(self.threads),
            sync=True,
        )
        hdiag_src_d = graph_diag_d if self.hdiag_d is None else self.hdiag_d
        dense_blocks_h = _build_dense_blocks_from_local_coo(
            n=nsel,
            block_ranges=block_ranges,
            row_h=np.asarray(cp.asnumpy(row_h), dtype=np.int32),
            col_h=np.asarray(cp.asnumpy(col_h), dtype=np.int32),
            hij_h=np.asarray(cp.asnumpy(hij_h), dtype=np.float64),
            hdiag_h=np.asarray(cp.asnumpy(hdiag_src_d), dtype=np.float64),
            symmetric_unique=True,
        )
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return _build_shifted_spectral_preconditioner(
            n=nsel,
            block_ranges=block_ranges,
            dense_blocks_h=dense_blocks_h,
            cp=cp,
            perm_d=self.sort_perm_d,
            inv_perm_d=self.inv_perm_d,
            denom_tol=float(denom_tol),
            label="pairwise_bucket_blocks",
        )


@dataclass
class ExactSelectedSymRowGraphProjectedHop:
    """Exact selected-space projected hop backed by a strict-lower symmetric graph."""

    sel_idx: np.ndarray
    row_ptr_d: Any
    col_idx_d: Any
    hij_d: Any
    hdiag_d: Any
    edge_row_d: Any | None = None
    edge_col_d: Any | None = None
    edge_hij_d: Any | None = None

    @classmethod
    def from_unique_local_edges(
        cls,
        *,
        sel_idx: np.ndarray,
        row_local,
        col_local,
        hij,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        nsel = int(sel_idx.size)
        row_d = cp.ascontiguousarray(cp.asarray(row_local, dtype=cp.int32).ravel())
        col_d = cp.ascontiguousarray(cp.asarray(col_local, dtype=cp.int32).ravel())
        hij_d = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())
        if int(row_d.size) != int(col_d.size) or int(row_d.size) != int(hij_d.size):
            raise ValueError("row_local, col_local, and hij must have matching lengths")
        if int(row_d.size) == 0:
            return cls(
                sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                row_ptr_d=cp.zeros((nsel + 1,), dtype=cp.int64),
                col_idx_d=cp.zeros((0,), dtype=cp.int32),
                hij_d=cp.zeros((0,), dtype=cp.float64),
                hdiag_d=cp.ascontiguousarray(
                    cp.asarray(np.zeros((nsel,), dtype=np.float64) if hdiag is None else hdiag, dtype=cp.float64).ravel()
                ),
                edge_row_d=cp.zeros((0,), dtype=cp.int32),
                edge_col_d=cp.zeros((0,), dtype=cp.int32),
                edge_hij_d=cp.zeros((0,), dtype=cp.float64),
            )

        offdiag_mask = row_d != col_d
        if bool(cp.any(offdiag_mask).item()):
            row_d = cp.ascontiguousarray(row_d[offdiag_mask].ravel())
            col_d = cp.ascontiguousarray(col_d[offdiag_mask].ravel())
            hij_d = cp.ascontiguousarray(hij_d[offdiag_mask].ravel())
        else:
            row_d = cp.zeros((0,), dtype=cp.int32)
            col_d = cp.zeros((0,), dtype=cp.int32)
            hij_d = cp.zeros((0,), dtype=cp.float64)

        if int(row_d.size) > 0:
            row_lo_d = cp.maximum(row_d, col_d).astype(cp.int32, copy=False)
            col_lo_d = cp.minimum(row_d, col_d).astype(cp.int32, copy=False)
            pair_key_d = (
                (row_lo_d.astype(cp.uint64, copy=False) << cp.uint64(32))
                | col_lo_d.astype(cp.uint32, copy=False).astype(cp.uint64, copy=False)
            )
            order_d = cp.asarray(cp.argsort(pair_key_d), dtype=cp.int64).ravel()
            pair_key_sorted_d = cp.ascontiguousarray(pair_key_d[order_d].ravel())
            hij_sorted_d = cp.ascontiguousarray(hij_d[order_d].ravel())
            if int(pair_key_sorted_d.size) == 1:
                starts_d = cp.asarray([0], dtype=cp.int64)
            else:
                starts_d = cp.ascontiguousarray(
                    cp.concatenate(
                        (
                            cp.asarray([0], dtype=cp.int64),
                            cp.asarray(
                                cp.nonzero(pair_key_sorted_d[1:] != pair_key_sorted_d[:-1])[0] + 1,
                                dtype=cp.int64,
                            ).ravel(),
                        )
                    ).ravel()
                )
            pair_unique_d = cp.ascontiguousarray(pair_key_sorted_d[starts_d].ravel())
            hij_unique_d = cp.ascontiguousarray(cp.add.reduceat(hij_sorted_d, starts_d))
            nz_mask = hij_unique_d != 0.0
            row_unique_d = cp.ascontiguousarray((pair_unique_d >> cp.uint64(32)).astype(cp.int32, copy=False).ravel())
            col_unique_d = cp.ascontiguousarray((pair_unique_d & cp.uint64(0xFFFFFFFF)).astype(cp.int32, copy=False).ravel())
            if bool(cp.any(nz_mask).item()):
                row_unique_d = cp.ascontiguousarray(row_unique_d[nz_mask].ravel())
                col_unique_d = cp.ascontiguousarray(col_unique_d[nz_mask].ravel())
                hij_unique_d = cp.ascontiguousarray(hij_unique_d[nz_mask].ravel())
            else:
                row_unique_d = cp.zeros((0,), dtype=cp.int32)
                col_unique_d = cp.zeros((0,), dtype=cp.int32)
                hij_unique_d = cp.zeros((0,), dtype=cp.float64)
        else:
            row_unique_d = cp.zeros((0,), dtype=cp.int32)
            col_unique_d = cp.zeros((0,), dtype=cp.int32)
            hij_unique_d = cp.zeros((0,), dtype=cp.float64)

        row_counts_d = cp.bincount(row_unique_d, minlength=nsel).astype(cp.int64, copy=False)
        row_ptr_d = cp.zeros((nsel + 1,), dtype=cp.int64)
        if nsel > 0:
            row_ptr_d[1:] = cp.cumsum(row_counts_d, dtype=cp.int64)
        return cls(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            row_ptr_d=cp.ascontiguousarray(row_ptr_d.ravel()),
            col_idx_d=cp.ascontiguousarray(col_unique_d.ravel()),
            hij_d=cp.ascontiguousarray(hij_unique_d.ravel()),
            hdiag_d=cp.ascontiguousarray(
                cp.asarray(np.zeros((nsel,), dtype=np.float64) if hdiag is None else hdiag, dtype=cp.float64).ravel()
            ),
            edge_row_d=cp.ascontiguousarray(row_unique_d.ravel()),
            edge_col_d=cp.ascontiguousarray(col_unique_d.ravel()),
            edge_hij_d=cp.ascontiguousarray(hij_unique_d.ravel()),
        )

    @classmethod
    def from_local_tuples(
        cls,
        *,
        sel_idx: np.ndarray,
        target_local,
        src_pos,
        hij,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        nsel = int(sel_idx.size)
        target_local_d = cp.ascontiguousarray(cp.asarray(target_local, dtype=cp.int32).ravel())
        src_pos_d = cp.ascontiguousarray(cp.asarray(src_pos, dtype=cp.int32).ravel())
        hij_d = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())

        if int(target_local_d.size) != int(src_pos_d.size) or int(target_local_d.size) != int(hij_d.size):
            raise ValueError("target_local, src_pos, and hij must have matching lengths")

        if int(target_local_d.size) == 0:
            return cls(
                sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                row_ptr_d=cp.zeros((nsel + 1,), dtype=cp.int64),
                col_idx_d=cp.zeros((0,), dtype=cp.int32),
                hij_d=cp.zeros((0,), dtype=cp.float64),
                hdiag_d=cp.ascontiguousarray(
                    cp.asarray(np.zeros((nsel,), dtype=np.float64) if hdiag is None else hdiag, dtype=cp.float64).ravel()
                ),
                edge_row_d=cp.zeros((0,), dtype=cp.int32),
                edge_col_d=cp.zeros((0,), dtype=cp.int32),
                edge_hij_d=cp.zeros((0,), dtype=cp.float64),
            )

        offdiag_mask = target_local_d != src_pos_d
        if bool(cp.any(offdiag_mask).item()):
            target_local_d = cp.ascontiguousarray(target_local_d[offdiag_mask].ravel())
            src_pos_d = cp.ascontiguousarray(src_pos_d[offdiag_mask].ravel())
            hij_d = cp.ascontiguousarray(hij_d[offdiag_mask].ravel())
        else:
            target_local_d = cp.zeros((0,), dtype=cp.int32)
            src_pos_d = cp.zeros((0,), dtype=cp.int32)
            hij_d = cp.zeros((0,), dtype=cp.float64)

        if int(target_local_d.size) > 0:
            row_d = cp.maximum(target_local_d, src_pos_d).astype(cp.int32, copy=False)
            col_d = cp.minimum(target_local_d, src_pos_d).astype(cp.int32, copy=False)
            pair_key_d = (
                (row_d.astype(cp.uint64, copy=False) << cp.uint64(32))
                | col_d.astype(cp.uint32, copy=False).astype(cp.uint64, copy=False)
            )
            order_d = cp.asarray(cp.argsort(pair_key_d), dtype=cp.int64).ravel()
            pair_key_sorted_d = cp.ascontiguousarray(pair_key_d[order_d].ravel())
            hij_sorted_d = cp.ascontiguousarray(hij_d[order_d].ravel())

            if int(pair_key_sorted_d.size) == 1:
                starts_d = cp.asarray([0], dtype=cp.int64)
            else:
                starts_d = cp.ascontiguousarray(
                    cp.concatenate(
                        (
                            cp.asarray([0], dtype=cp.int64),
                            cp.asarray(cp.nonzero(pair_key_sorted_d[1:] != pair_key_sorted_d[:-1])[0] + 1, dtype=cp.int64).ravel(),
                        )
                    ).ravel()
                )
            pair_unique_d = cp.ascontiguousarray(pair_key_sorted_d[starts_d].ravel())
            # The exact selected emitter produces directed (i, j) and (j, i) tuples
            # for symmetric off-diagonal selected edges. Collapse to one undirected
            # edge and divide by two so the symmetric matvec applies H_ij once.
            hij_unique_d = cp.ascontiguousarray(0.5 * cp.add.reduceat(hij_sorted_d, starts_d))
            nz_mask = hij_unique_d != 0.0
            row_unique_d = cp.ascontiguousarray((pair_unique_d >> cp.uint64(32)).astype(cp.int32, copy=False).ravel())
            col_unique_d = cp.ascontiguousarray((pair_unique_d & cp.uint64(0xFFFFFFFF)).astype(cp.int32, copy=False).ravel())
            if bool(cp.any(nz_mask).item()):
                row_unique_d = cp.ascontiguousarray(row_unique_d[nz_mask].ravel())
                col_unique_d = cp.ascontiguousarray(col_unique_d[nz_mask].ravel())
                hij_unique_d = cp.ascontiguousarray(hij_unique_d[nz_mask].ravel())
            else:
                row_unique_d = cp.zeros((0,), dtype=cp.int32)
                col_unique_d = cp.zeros((0,), dtype=cp.int32)
                hij_unique_d = cp.zeros((0,), dtype=cp.float64)
        else:
            row_unique_d = cp.zeros((0,), dtype=cp.int32)
            col_unique_d = cp.zeros((0,), dtype=cp.int32)
            hij_unique_d = cp.zeros((0,), dtype=cp.float64)

        row_counts_d = cp.bincount(row_unique_d, minlength=nsel).astype(cp.int64, copy=False)
        row_ptr_d = cp.zeros((nsel + 1,), dtype=cp.int64)
        if nsel > 0:
            row_ptr_d[1:] = cp.cumsum(row_counts_d, dtype=cp.int64)

        return cls(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            row_ptr_d=cp.ascontiguousarray(row_ptr_d.ravel()),
            col_idx_d=cp.ascontiguousarray(col_unique_d.ravel()),
            hij_d=cp.ascontiguousarray(hij_unique_d.ravel()),
            hdiag_d=cp.ascontiguousarray(
                cp.asarray(np.zeros((nsel,), dtype=np.float64) if hdiag is None else hdiag, dtype=cp.float64).ravel()
            ),
            edge_row_d=cp.ascontiguousarray(row_unique_d.ravel()),
            edge_col_d=cp.ascontiguousarray(col_unique_d.ravel()),
            edge_hij_d=cp.ascontiguousarray(hij_unique_d.ravel()),
        )

    @classmethod
    def from_tuples(
        cls,
        *,
        sel_idx: np.ndarray,
        labels,
        src_pos,
        hij,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        labels_d = cp.ascontiguousarray(cp.asarray(labels, dtype=cp.uint64).ravel())
        src_pos_d = cp.ascontiguousarray(cp.asarray(src_pos, dtype=cp.int32).ravel())
        hij_d = cp.ascontiguousarray(cp.asarray(hij, dtype=cp.float64).ravel())

        if int(labels_d.size) == 0:
            return cls.from_local_tuples(
                sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
                target_local=cp.zeros((0,), dtype=cp.int32),
                src_pos=cp.zeros((0,), dtype=cp.int32),
                hij=cp.zeros((0,), dtype=cp.float64),
                hdiag=hdiag,
            )

        sel_u64_h = np.asarray(sel_idx.astype(np.uint64, copy=False), dtype=np.uint64)
        sort_order_h = np.argsort(sel_u64_h, kind="stable").astype(np.int32, copy=False)
        sel_sorted_d = cp.ascontiguousarray(cp.asarray(sel_u64_h[sort_order_h], dtype=cp.uint64).ravel())
        sel_sorted_to_local_d = cp.ascontiguousarray(cp.asarray(sort_order_h, dtype=cp.int32).ravel())
        pos_d = cp.asarray(cp.searchsorted(sel_sorted_d, labels_d), dtype=cp.int32).ravel()
        matched = (pos_d >= 0) & (pos_d < int(sel_sorted_d.size)) & (sel_sorted_d[pos_d] == labels_d)
        if not bool(cp.all(matched).item()):
            raise RuntimeError("selected graph tuple emitter returned labels outside the selected space")
        return cls.from_local_tuples(
            sel_idx=np.asarray(sel_idx, dtype=np.int64).ravel(),
            target_local=cp.ascontiguousarray(sel_sorted_to_local_d[pos_d].ravel()),
            src_pos=src_pos_d,
            hij=hij_d,
            hdiag=hdiag,
        )

    @classmethod
    def from_pairwise_selected_space(
        cls,
        *,
        drt,
        drt_dev,
        sel_idx: np.ndarray,
        h_base_d,
        eri4_d,
        cp,
        threads: int = 256,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            pairwise_build_bucket_data,
            pairwise_build_selected_graph_bucketed_u64_device,
            pairwise_materialize_u64_device,
        )

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        nsel = int(sel_idx.size)
        sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
        materialized = pairwise_materialize_u64_device(
            drt,
            drt_dev,
            sel_u64_d,
            nsel,
            cp,
            threads=int(threads),
            sync=False,
        )
        steps_all, nodes_all, occ_all, b_all = materialized
        bucket_data_full = pairwise_build_bucket_data(occ_all, int(drt.norb), cp)
        sort_perm_d = cp.ascontiguousarray(bucket_data_full["sort_perm"].astype(cp.int32, copy=False))
        inv_perm_d = cp.ascontiguousarray(bucket_data_full["inv_perm"].astype(cp.int32, copy=False))
        materialized_sorted = (
            cp.ascontiguousarray(steps_all[sort_perm_d]),
            cp.ascontiguousarray(nodes_all[sort_perm_d]),
            cp.ascontiguousarray(occ_all[sort_perm_d]),
            cp.ascontiguousarray(b_all[sort_perm_d]),
        )
        sel_sorted_u64_d = cp.ascontiguousarray(sel_u64_d[sort_perm_d])
        bucket_data = {
            "occ_keys_sorted": cp.ascontiguousarray(bucket_data_full["occ_keys_sorted"]),
            "bucket_keys": cp.ascontiguousarray(bucket_data_full["bucket_keys"]),
            "bucket_starts": cp.ascontiguousarray(bucket_data_full["bucket_starts"]),
            "bucket_sizes": cp.ascontiguousarray(bucket_data_full["bucket_sizes"]),
            "neighbor_offsets": cp.ascontiguousarray(bucket_data_full["neighbor_offsets"]),
            "neighbor_list": cp.ascontiguousarray(bucket_data_full["neighbor_list"]),
            "csf_to_bucket": cp.ascontiguousarray(bucket_data_full["csf_to_bucket"]),
            "target_offsets": cp.ascontiguousarray(bucket_data_full["target_offsets"]),
            "target_list": cp.ascontiguousarray(bucket_data_full["target_list"]),
            "target_offsets_1b": cp.ascontiguousarray(bucket_data_full["target_offsets_1b"]),
            "target_list_1b": cp.ascontiguousarray(bucket_data_full["target_list_1b"]),
        }
        target_sorted_d, src_sorted_d, hij_d, diag_sorted_d, _row_counts_d = pairwise_build_selected_graph_bucketed_u64_device(
            drt,
            drt_dev,
            sel_sorted_u64_d,
            nsel,
            h_base_d,
            eri4_d,
            materialized_sorted,
            bucket_data,
            cp,
            threads=int(threads),
            sync=True,
        )
        graph_hop = cls.from_unique_local_edges(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            row_local=cp.ascontiguousarray(sort_perm_d[target_sorted_d].astype(cp.int32, copy=False)),
            col_local=cp.ascontiguousarray(sort_perm_d[src_sorted_d].astype(cp.int32, copy=False)),
            hij=hij_d,
            hdiag=cp.ascontiguousarray(diag_sorted_d[inv_perm_d].ravel()),
        )
        del materialized, materialized_sorted, bucket_data_full
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        return graph_hop

    @classmethod
    def from_compact_df_selected_space(
        cls,
        *,
        drt,
        drt_dev,
        sel_idx: np.ndarray,
        h_base_d,
        l_full_d,
        cp,
        hdiag: np.ndarray | None = None,
        state_dev=None,
        state_cache=None,
        block_nroots: int = 64,
        csr_threads: int = 128,
        apply_threads: int = 256,
        kernel3_threads: int = 256,
        csr_capacity_scale: int = 8,
        max_selected_mask_ncsf: int = 2_000_000,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            Kernel3BuildGDFWorkspace,
            apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device,
            kernel25_build_csr_from_tasks_deterministic_inplace_device,
            make_device_state_cache,
        )
        from asuka.cuguga.state_cache import get_state_cache  # noqa: PLC0415

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        nsel = int(sel_idx.size)
        if nsel == 0:
            return cls.from_unique_local_edges(
                sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                row_local=cp.zeros((0,), dtype=cp.int32),
                col_local=cp.zeros((0,), dtype=cp.int32),
                hij=cp.zeros((0,), dtype=cp.float64),
                hdiag=hdiag,
            )

        ncsf = int(getattr(drt, "ncsf"))
        if ncsf > int(max_selected_mask_ncsf):
            raise RuntimeError(
                f"compact DF selected-graph builder requires ncsf <= {int(max_selected_mask_ncsf)} "
                f"(got {ncsf})"
            )
        if ncsf > int(np.iinfo(np.int32).max):
            raise RuntimeError("compact DF selected-graph builder requires int32-addressable CSF labels")

        sel_i32_h = np.asarray(sel_idx, dtype=np.int64)
        if np.any(sel_i32_h < 0) or np.any(sel_i32_h >= ncsf):
            raise ValueError("sel_idx contains labels outside the CSF space")
        sel_i32_d = cp.ascontiguousarray(cp.asarray(sel_i32_h.astype(np.int32, copy=False), dtype=cp.int32).ravel())
        sel_u64_h = np.asarray(sel_idx.astype(np.uint64, copy=False), dtype=np.uint64)
        sort_order_h = np.argsort(sel_u64_h, kind="stable").astype(np.int32, copy=False)
        sel_sorted_u64_d = cp.ascontiguousarray(cp.asarray(sel_u64_h[sort_order_h], dtype=cp.uint64).ravel())
        sel_sorted_to_local_d = cp.ascontiguousarray(cp.asarray(sort_order_h, dtype=cp.int32).ravel())

        if state_dev is None:
            if state_cache is None:
                state_cache = get_state_cache(drt)
            state_dev = make_device_state_cache(drt, drt_dev, state_cache)

        norb = int(getattr(drt, "norb"))
        nops = int(norb * norb)
        h_base_flat_d = cp.ascontiguousarray(cp.asarray(h_base_d, dtype=cp.float64).ravel())
        if int(h_base_flat_d.size) != int(nops):
            raise ValueError("h_base_d must have shape (norb*norb,) or (norb,norb)")
        l_full_d = cp.ascontiguousarray(cp.asarray(l_full_d, dtype=cp.float64))
        if int(l_full_d.ndim) != 2 or int(l_full_d.shape[0]) != int(nops):
            raise ValueError("l_full_d must have shape (norb*norb, naux)")
        naux = int(l_full_d.shape[1])

        selected_mask_d = cp.ones((ncsf,), dtype=cp.uint8)
        selected_mask_d[sel_i32_d] = np.uint8(0)

        offdiag_pairs_h = np.asarray(
            [(p, q) for p in range(norb) for q in range(norb) if p != q],
            dtype=np.int32,
        )
        task_p_base_d = cp.ascontiguousarray(cp.asarray(offdiag_pairs_h[:, 0], dtype=cp.int32).ravel())
        task_q_base_d = cp.ascontiguousarray(cp.asarray(offdiag_pairs_h[:, 1], dtype=cp.int32).ravel())
        npairs = int(task_p_base_d.size)

        block_nroots = max(1, int(block_nroots))
        csr_threads = int(csr_threads)
        apply_threads = int(apply_threads)
        kernel3_threads = int(kernel3_threads)
        csr_capacity_scale = max(2, int(csr_capacity_scale))

        hash_cap = 1
        while hash_cap < max(16, 2 * nsel):
            hash_cap <<= 1
        hash_keys_d = cp.empty((hash_cap,), dtype=cp.int32)
        hash_vals_d = cp.empty((block_nroots, hash_cap), dtype=cp.float64)
        hash_overflow_d = cp.zeros((1,), dtype=cp.int32)

        gdf_workspace = Kernel3BuildGDFWorkspace(int(nops), int(naux), max_nrows=max(1, block_nroots * npairs))
        g_buf_d = None

        edge_row_parts: list[Any] = []
        edge_col_parts: list[Any] = []
        edge_hij_parts: list[Any] = []

        for src_lo in range(0, nsel, block_nroots):
            src_hi = min(src_lo + block_nroots, nsel)
            nb = int(src_hi - src_lo)
            src_block_d = cp.ascontiguousarray(sel_i32_d[src_lo:src_hi].ravel())

            hash_keys_d.fill(np.int32(-1))
            hash_vals_d[:nb, :].fill(0.0)
            hash_overflow_d.fill(0)

            eye_block_d = cp.eye(nb, dtype=cp.float64)
            apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                drt,
                drt_dev,
                state_dev,
                src_block_d,
                h_base_flat_d,
                task_scale_task_major=eye_block_d,
                hash_keys=hash_keys_d,
                hash_vals=hash_vals_d[:nb, :],
                selected_mask=selected_mask_d,
                overflow=hash_overflow_d,
                clear_overflow=False,
                threads=int(apply_threads),
                sync=True,
                check_overflow=True,
            )

            task_csf_d = cp.ascontiguousarray(cp.repeat(src_block_d, npairs).astype(cp.int32, copy=False))
            task_p_d = cp.ascontiguousarray(cp.tile(task_p_base_d, nb).astype(cp.int32, copy=False))
            task_q_d = cp.ascontiguousarray(cp.tile(task_q_base_d, nb).astype(cp.int32, copy=False))

            capacity = max(4096, int(task_csf_d.size) * int(csr_capacity_scale))
            while True:
                row_j_d = cp.empty((capacity,), dtype=cp.int32)
                row_k_d = cp.empty((capacity,), dtype=cp.int32)
                indptr_d = cp.empty((capacity + 1,), dtype=cp.int64)
                indices_d = cp.empty((capacity,), dtype=cp.int32)
                data_d = cp.empty((capacity,), dtype=cp.float64)
                csr_overflow_d = cp.zeros((1,), dtype=cp.int32)
                (
                    row_j_d,
                    row_k_d,
                    indptr_d,
                    indices_d,
                    data_d,
                    csr_overflow_d,
                    nrows_csr,
                    nnz_csr,
                    _nnz_in,
                ) = kernel25_build_csr_from_tasks_deterministic_inplace_device(
                    drt,
                    drt_dev,
                    state_dev,
                    task_csf_d,
                    task_p_d,
                    task_q_d,
                    capacity=capacity,
                    row_j=row_j_d,
                    row_k=row_k_d,
                    indptr=indptr_d,
                    indices=indices_d,
                    data=data_d,
                    overflow=csr_overflow_d,
                    threads=int(csr_threads),
                    coalesce=True,
                    sync=True,
                    check_overflow=False,
                )
                if int(cp.asnumpy(csr_overflow_d)[0]) == 0:
                    break
                capacity <<= 1

            nrows_csr = int(nrows_csr)
            if nrows_csr > 0:
                if g_buf_d is None or int(g_buf_d.shape[0]) < nrows_csr:
                    g_buf_d = cp.empty((nrows_csr, nops), dtype=cp.float64)
                src_block_u64_h = np.asarray(cp.asnumpy(src_block_d), dtype=np.uint64)
                src_sort_order_h = np.argsort(src_block_u64_h, kind="stable").astype(np.int32, copy=False)
                src_sorted_d = cp.ascontiguousarray(cp.asarray(src_block_u64_h[src_sort_order_h], dtype=cp.uint64).ravel())
                src_sorted_to_root_d = cp.ascontiguousarray(cp.asarray(src_sort_order_h, dtype=cp.int32).ravel())
                row_pos_d = cp.asarray(cp.searchsorted(src_sorted_d, row_j_d.astype(cp.uint64, copy=False)), dtype=cp.int32).ravel()
                row_root_d = cp.ascontiguousarray(src_sorted_to_root_d[row_pos_d].ravel())
                task_scale_rows_d = cp.zeros((nrows_csr, nb), dtype=cp.float64)
                task_scale_rows_d[cp.arange(nrows_csr, dtype=cp.int32), row_root_d] = 1.0
                gdf_workspace.build_g_from_csr_l_full_range_inplace_device(
                    indptr_d,
                    indices_d,
                    data_d,
                    row_start=0,
                    nrows=nrows_csr,
                    l_full=l_full_d,
                    g_out=g_buf_d[:nrows_csr, :],
                    threads=int(kernel3_threads),
                    stream=None,
                    sync=False,
                )
                apply_g_flat_scatter_atomic_frontier_hash_many_roots_inplace_device(
                    drt,
                    drt_dev,
                    state_dev,
                    row_k_d,
                    g_buf_d[:nrows_csr, :],
                    task_scale_task_major=task_scale_rows_d,
                    hash_keys=hash_keys_d,
                    hash_vals=hash_vals_d[:nb, :],
                    selected_mask=selected_mask_d,
                    overflow=hash_overflow_d,
                    clear_overflow=False,
                    threads=int(apply_threads),
                    sync=True,
                    check_overflow=True,
                )

            occupied_d = hash_keys_d != np.int32(-1)
            if bool(cp.any(occupied_d).item()):
                target_global_d = cp.ascontiguousarray(hash_keys_d[occupied_d].ravel())
                values_d = cp.ascontiguousarray(hash_vals_d[:nb, occupied_d])
                target_pos_d = cp.asarray(cp.searchsorted(sel_sorted_u64_d, target_global_d.astype(cp.uint64, copy=False)), dtype=cp.int32).ravel()
                target_local_d = cp.ascontiguousarray(sel_sorted_to_local_d[target_pos_d].ravel())
                src_local_d = cp.arange(src_lo, src_hi, dtype=cp.int32)
                nz_mask_d = values_d != 0.0
                lower_mask_d = nz_mask_d & (target_local_d[None, :] < src_local_d[:, None])
                if bool(cp.any(lower_mask_d).item()):
                    row_idx_d, occ_idx_d = cp.nonzero(lower_mask_d)
                    edge_row_parts.append(cp.ascontiguousarray(src_local_d[row_idx_d].ravel()))
                    edge_col_parts.append(cp.ascontiguousarray(target_local_d[occ_idx_d].ravel()))
                    edge_hij_parts.append(cp.ascontiguousarray(values_d[row_idx_d, occ_idx_d].ravel()))

        row_all_d = cp.zeros((0,), dtype=cp.int32) if not edge_row_parts else cp.ascontiguousarray(cp.concatenate(edge_row_parts).ravel())
        col_all_d = cp.zeros((0,), dtype=cp.int32) if not edge_col_parts else cp.ascontiguousarray(cp.concatenate(edge_col_parts).ravel())
        hij_all_d = cp.zeros((0,), dtype=cp.float64) if not edge_hij_parts else cp.ascontiguousarray(cp.concatenate(edge_hij_parts).ravel())
        return cls.from_unique_local_edges(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            row_local=row_all_d,
            col_local=col_all_d,
            hij=hij_all_d,
            hdiag=hdiag,
        )

    @classmethod
    def from_selected_rowhash_dense(
        cls,
        *,
        drt,
        drt_dev,
        sel_idx: np.ndarray,
        h_base_d,
        eri4_d,
        cp,
        row_cap: int = 1024,
        threads: int = 256,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            build_selected_membership_hash,
            cas36_exact_selected_build_rowhash_dense_u64_inplace_device,
        )

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        nsel = int(sel_idx.size)
        sel_u64_h = np.asarray(sel_idx.astype(np.uint64, copy=False), dtype=np.uint64)
        sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_u64_h, dtype=cp.uint64).ravel())
        c_bound_d = cp.ones((nsel,), dtype=cp.float64)
        sort_order_h = np.argsort(sel_u64_h, kind="stable").astype(np.int32, copy=False)
        sel_sorted_d = cp.ascontiguousarray(cp.asarray(sel_u64_h[sort_order_h], dtype=cp.uint64).ravel())
        sel_sorted_to_local_d = cp.ascontiguousarray(cp.asarray(sort_order_h, dtype=cp.int32).ravel())
        membership_hash_keys_d, membership_hash_cap = build_selected_membership_hash(sel_sorted_d, cp)
        if membership_hash_keys_d is None or int(membership_hash_cap) <= 0:
            raise RuntimeError("selected membership hash is required for dense selected rowhash emission")

        row_cap = max(64, int(row_cap))
        row_cap = 1 << max(0, row_cap - 1).bit_length()
        retries = 0
        empty_u64 = np.uint64(0xFFFFFFFFFFFFFFFF)
        while True:
            total_cap = int(nsel) * int(row_cap)
            row_hash_keys_d = cp.empty((total_cap,), dtype=cp.uint64)
            row_hash_vals_d = cp.empty((total_cap,), dtype=cp.float64)
            diag_d = cp.zeros((nsel,), dtype=cp.float64)
            overflow_d = cp.zeros((1,), dtype=cp.int32)
            cas36_exact_selected_build_rowhash_dense_u64_inplace_device(
                drt,
                drt_dev,
                sel_u64_d,
                c_bound_d,
                nsel=int(nsel),
                h_base=h_base_d,
                eri4=eri4_d,
                row_hash_keys=row_hash_keys_d,
                row_hash_vals=row_hash_vals_d,
                row_cap=int(row_cap),
                membership_hash_keys=membership_hash_keys_d,
                membership_hash_cap=int(membership_hash_cap),
                out_diag=diag_d,
                overflow=overflow_d,
                threads=int(threads),
                sync=False,
            )
            if int(cp.asnumpy(overflow_d)[0]) == 0:
                break
            retries += 1
            if retries > 4:
                raise RuntimeError(f"dense selected rowhash emitter overflowed after {retries} retries")
            row_cap <<= 1

        occupied_d = row_hash_keys_d != empty_u64
        if not bool(cp.any(occupied_d).item()):
            return cls.from_local_tuples(
                sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                target_local=cp.zeros((0,), dtype=cp.int32),
                src_pos=cp.zeros((0,), dtype=cp.int32),
                hij=cp.zeros((0,), dtype=cp.float64),
                hdiag=np.asarray(cp.asnumpy(diag_d), dtype=np.float64, order="C"),
            )

        occ_slot_d = cp.asarray(cp.nonzero(occupied_d)[0], dtype=cp.int64).ravel()
        row_local_d = cp.ascontiguousarray((occ_slot_d // int(row_cap)).astype(cp.int32, copy=False).ravel())
        labels_d = cp.ascontiguousarray(row_hash_keys_d[occupied_d].ravel())
        hij_d = cp.ascontiguousarray(row_hash_vals_d[occupied_d].ravel())
        pos_d = cp.asarray(cp.searchsorted(sel_sorted_d, labels_d), dtype=cp.int32).ravel()
        matched = (pos_d >= 0) & (pos_d < int(sel_sorted_d.size)) & (sel_sorted_d[pos_d] == labels_d)
        if not bool(cp.all(matched).item()):
            raise RuntimeError("dense selected rowhash emitter returned labels outside the selected space")
        target_local_d = cp.ascontiguousarray(sel_sorted_to_local_d[pos_d].ravel())
        return cls.from_local_tuples(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            target_local=target_local_d,
            src_pos=row_local_d,
            hij=hij_d,
            hdiag=np.asarray(cp.asnumpy(diag_d), dtype=np.float64, order="C"),
        )

    @classmethod
    def from_dense_matrix(
        cls,
        *,
        sel_idx: np.ndarray,
        H_d,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        nsel = int(sel_idx.size)
        H_d = cp.ascontiguousarray(cp.asarray(H_d, dtype=cp.float64))
        if H_d.ndim != 2 or int(H_d.shape[0]) != nsel or int(H_d.shape[1]) != nsel:
            raise ValueError("H_d must have shape (nsel, nsel)")

        hdiag_d = cp.ascontiguousarray(
            cp.asarray(cp.diag(H_d) if hdiag is None else hdiag, dtype=cp.float64).ravel()
        )
        if nsel == 0:
            return cls(
                sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                row_ptr_d=cp.zeros((1,), dtype=cp.int64),
                col_idx_d=cp.zeros((0,), dtype=cp.int32),
                hij_d=cp.zeros((0,), dtype=cp.float64),
                hdiag_d=hdiag_d,
                edge_row_d=cp.zeros((0,), dtype=cp.int32),
                edge_col_d=cp.zeros((0,), dtype=cp.int32),
                edge_hij_d=cp.zeros((0,), dtype=cp.float64),
            )

        lower_mask_d = cp.tril(H_d != 0.0, k=-1)
        row_counts_d = cp.count_nonzero(lower_mask_d, axis=1).astype(cp.int64, copy=False)
        row_ptr_d = cp.zeros((nsel + 1,), dtype=cp.int64)
        row_ptr_d[1:] = cp.cumsum(row_counts_d, dtype=cp.int64)
        edge_row_d, edge_col_d = cp.nonzero(lower_mask_d)
        edge_row_d = cp.ascontiguousarray(edge_row_d.astype(cp.int32, copy=False).ravel())
        edge_col_d = cp.ascontiguousarray(edge_col_d.astype(cp.int32, copy=False).ravel())
        edge_hij_d = cp.ascontiguousarray(H_d[edge_row_d, edge_col_d].ravel())
        return cls(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            row_ptr_d=cp.ascontiguousarray(row_ptr_d.ravel()),
            col_idx_d=cp.ascontiguousarray(edge_col_d.ravel()),
            hij_d=cp.ascontiguousarray(edge_hij_d.ravel()),
            hdiag_d=hdiag_d,
            edge_row_d=edge_row_d,
            edge_col_d=edge_col_d,
            edge_hij_d=edge_hij_d,
        )

    def hop_gpu(self, x_d):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        from asuka.cuda.cuda_backend import cas36_sym_row_graph_spmm_device  # noqa: PLC0415

        nsel = int(self.sel_idx.size)
        x_d = cp.asarray(x_d, dtype=cp.float64)
        if int(getattr(x_d, "ndim", 1)) == 1:
            if int(x_d.size) != nsel:
                raise ValueError("gpu projected-hop input has wrong length")
        elif int(x_d.ndim) == 2:
            if int(x_d.shape[0]) != nsel:
                raise ValueError("gpu projected-hop input has wrong leading dimension")
        else:
            raise ValueError("gpu projected-hop input must be 1D or 2D")
        return cas36_sym_row_graph_spmm_device(
            self.row_ptr_d,
            self.col_idx_d,
            self.hij_d,
            self.hdiag_d,
            x_d,
            cp,
            threads=256,
            sync=False,
        )

    def with_hdiag(self, *, hdiag: np.ndarray | None = None) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        return type(self)(
            sel_idx=np.asarray(self.sel_idx, dtype=np.int64, order="C"),
            row_ptr_d=cp.ascontiguousarray(cp.asarray(self.row_ptr_d, dtype=cp.int64).ravel()),
            col_idx_d=cp.ascontiguousarray(cp.asarray(self.col_idx_d, dtype=cp.int32).ravel()),
            hij_d=cp.ascontiguousarray(cp.asarray(self.hij_d, dtype=cp.float64).ravel()),
            hdiag_d=cp.ascontiguousarray(cp.asarray(self.hdiag_d if hdiag is None else hdiag, dtype=cp.float64).ravel()),
            edge_row_d=None if self.edge_row_d is None else cp.ascontiguousarray(cp.asarray(self.edge_row_d, dtype=cp.int32).ravel()),
            edge_col_d=None if self.edge_col_d is None else cp.ascontiguousarray(cp.asarray(self.edge_col_d, dtype=cp.int32).ravel()),
            edge_hij_d=None if self.edge_hij_d is None else cp.ascontiguousarray(cp.asarray(self.edge_hij_d, dtype=cp.float64).ravel()),
        )

    def with_merged_hop(
        self,
        *,
        other: "ExactSelectedSymRowGraphProjectedHop",
        sel_idx: np.ndarray,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        if int(self.sel_idx.size) == 0:
            return other.with_hdiag(hdiag=hdiag)
        if int(other.sel_idx.size) == 0 or other.edge_hij_d is None or int(other.edge_hij_d.size) == 0:
            return self.with_hdiag(hdiag=hdiag)
        if self.edge_row_d is None or self.edge_col_d is None or self.edge_hij_d is None or int(self.edge_hij_d.size) == 0:
            return other.with_hdiag(hdiag=hdiag)

        return type(self).from_unique_local_edges(
            sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
            row_local=cp.concatenate(
                (
                    cp.ascontiguousarray(cp.asarray(self.edge_row_d, dtype=cp.int32).ravel()),
                    cp.ascontiguousarray(cp.asarray(other.edge_row_d, dtype=cp.int32).ravel()),
                )
            ),
            col_local=cp.concatenate(
                (
                    cp.ascontiguousarray(cp.asarray(self.edge_col_d, dtype=cp.int32).ravel()),
                    cp.ascontiguousarray(cp.asarray(other.edge_col_d, dtype=cp.int32).ravel()),
                )
            ),
            hij=cp.concatenate(
                (
                    cp.ascontiguousarray(cp.asarray(self.edge_hij_d, dtype=cp.float64).ravel()),
                    cp.ascontiguousarray(cp.asarray(other.edge_hij_d, dtype=cp.float64).ravel()),
                )
            ),
            hdiag=hdiag,
        )

    def with_appended_rows(
        self,
        *,
        other: "ExactSelectedSymRowGraphProjectedHop",
        old_n: int,
        sel_idx: np.ndarray,
        hdiag: np.ndarray | None = None,
    ) -> "ExactSelectedSymRowGraphProjectedHop":
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop") from e

        old_n = int(old_n)
        sel_idx = np.asarray(sel_idx, dtype=np.int64, order="C")
        new_n = int(sel_idx.size)
        if old_n < 0 or old_n > int(self.sel_idx.size) or old_n > new_n:
            raise ValueError("invalid old_n for appended graph merge")
        if int(self.sel_idx.size) != old_n:
            raise ValueError("append-only graph merge requires self to match the old selected prefix")
        if int(other.sel_idx.size) != new_n:
            raise ValueError("delta graph sel_idx must match the new selected space")

        old_row_ptr_d = cp.ascontiguousarray(cp.asarray(self.row_ptr_d, dtype=cp.int64).ravel())
        old_col_idx_d = cp.ascontiguousarray(cp.asarray(self.col_idx_d, dtype=cp.int32).ravel())
        old_hij_d = cp.ascontiguousarray(cp.asarray(self.hij_d, dtype=cp.float64).ravel())
        old_edge_row_d = (
            cp.zeros((0,), dtype=cp.int32)
            if self.edge_row_d is None
            else cp.ascontiguousarray(cp.asarray(self.edge_row_d, dtype=cp.int32).ravel())
        )
        old_edge_col_d = (
            cp.zeros((0,), dtype=cp.int32)
            if self.edge_col_d is None
            else cp.ascontiguousarray(cp.asarray(self.edge_col_d, dtype=cp.int32).ravel())
        )
        old_edge_hij_d = (
            cp.zeros((0,), dtype=cp.float64)
            if self.edge_hij_d is None
            else cp.ascontiguousarray(cp.asarray(self.edge_hij_d, dtype=cp.float64).ravel())
        )

        delta_row_ptr_d = cp.ascontiguousarray(cp.asarray(other.row_ptr_d, dtype=cp.int64).ravel())
        delta_col_idx_d = cp.ascontiguousarray(cp.asarray(other.col_idx_d, dtype=cp.int32).ravel())
        delta_hij_d = cp.ascontiguousarray(cp.asarray(other.hij_d, dtype=cp.float64).ravel())
        delta_edge_row_d = (
            cp.zeros((0,), dtype=cp.int32)
            if other.edge_row_d is None
            else cp.ascontiguousarray(cp.asarray(other.edge_row_d, dtype=cp.int32).ravel())
        )
        delta_edge_col_d = (
            cp.zeros((0,), dtype=cp.int32)
            if other.edge_col_d is None
            else cp.ascontiguousarray(cp.asarray(other.edge_col_d, dtype=cp.int32).ravel())
        )
        delta_edge_hij_d = (
            cp.zeros((0,), dtype=cp.float64)
            if other.edge_hij_d is None
            else cp.ascontiguousarray(cp.asarray(other.edge_hij_d, dtype=cp.float64).ravel())
        )

        if new_n == old_n:
            return self.with_hdiag(hdiag=hdiag)

        old_nnz = int(old_hij_d.size)
        delta_base = delta_row_ptr_d[old_n]
        row_tail_d = cp.ascontiguousarray(
            old_nnz + (delta_row_ptr_d[old_n + 1 :] - delta_base)
        )
        row_ptr_d = cp.ascontiguousarray(cp.concatenate((old_row_ptr_d, row_tail_d)).ravel())
        col_idx_d = cp.ascontiguousarray(cp.concatenate((old_col_idx_d, delta_col_idx_d)).ravel())
        hij_d = cp.ascontiguousarray(cp.concatenate((old_hij_d, delta_hij_d)).ravel())
        edge_row_d = cp.ascontiguousarray(cp.concatenate((old_edge_row_d, delta_edge_row_d)).ravel())
        edge_col_d = cp.ascontiguousarray(cp.concatenate((old_edge_col_d, delta_edge_col_d)).ravel())
        edge_hij_d = cp.ascontiguousarray(cp.concatenate((old_edge_hij_d, delta_edge_hij_d)).ravel())
        return type(self)(
            sel_idx=sel_idx,
            row_ptr_d=row_ptr_d,
            col_idx_d=col_idx_d,
            hij_d=hij_d,
            hdiag_d=cp.ascontiguousarray(
                cp.asarray(self.hdiag_d if hdiag is None else hdiag, dtype=cp.float64).ravel()
            ),
            edge_row_d=edge_row_d,
            edge_col_d=edge_col_d,
            edge_hij_d=edge_hij_d,
        )

    def build_jd_preconditioner(self, *, block_size: int = 64, denom_tol: float = 1e-8):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactSelectedSymRowGraphProjectedHop preconditioner") from e

        nsel = int(self.sel_idx.size)
        block_ranges = _contiguous_block_ranges(nsel, int(block_size))
        dense_blocks_h = _build_dense_blocks_from_local_coo(
            n=nsel,
            block_ranges=block_ranges,
            row_h=np.asarray(
                cp.asnumpy(cp.zeros((0,), dtype=cp.int32) if self.edge_row_d is None else self.edge_row_d),
                dtype=np.int32,
            ),
            col_h=np.asarray(
                cp.asnumpy(cp.zeros((0,), dtype=cp.int32) if self.edge_col_d is None else self.edge_col_d),
                dtype=np.int32,
            ),
            hij_h=np.asarray(
                cp.asnumpy(cp.zeros((0,), dtype=cp.float64) if self.edge_hij_d is None else self.edge_hij_d),
                dtype=np.float64,
            ),
            hdiag_h=np.asarray(cp.asnumpy(self.hdiag_d), dtype=np.float64),
            symmetric_unique=True,
        )
        return _build_shifted_spectral_preconditioner(
            n=nsel,
            block_ranges=block_ranges,
            dense_blocks_h=dense_blocks_h,
            cp=cp,
            denom_tol=float(denom_tol),
            label="sym_graph_contiguous_blocks",
        )


@dataclass
class ExactExternalProjectedApply:
    """Exact external projected apply `P_X H P_S C`.

    This is the Batch-2 foundation interface. The first implementation is a host
    reference using the exact connected-row oracle and persistent row cache.
    """

    drt: Any
    h1e: np.ndarray
    eri: Any
    max_out: int
    screening: Any | None = None
    state_cache: Any | None = None
    row_cache: Any | None = None

    def _emit_host_tuples(
        self,
        *,
        sel_idx: np.ndarray,
        c_sel: np.ndarray,
        label_lo: int = 0,
        label_hi: int | None = None,
        screen_contrib: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from asuka.sci.sparse_support import _connected_row_cached  # noqa: PLC0415

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        if c_sel.ndim != 2:
            raise ValueError("c_sel must have shape (nsel, nroots)")
        if int(c_sel.shape[0]) != int(sel_idx.size):
            raise ValueError("c_sel leading dimension must match sel_idx length")
        if label_hi is None:
            label_hi = int(getattr(self.drt, "ncsf"))
        label_lo = int(label_lo)
        label_hi = int(label_hi)
        if label_lo < 0 or label_hi < label_lo:
            raise ValueError("invalid label window")

        max_abs = np.max(np.abs(c_sel), axis=1) if int(sel_idx.size) > 0 else np.zeros((0,), dtype=np.float64)
        label_parts: list[np.ndarray] = []
        src_parts: list[np.ndarray] = []
        hij_parts: list[np.ndarray] = []
        for col, j in enumerate(sel_idx.tolist()):
            max_cj = float(max_abs[col]) if int(max_abs.size) > int(col) else 0.0
            if max_cj == 0.0:
                continue
            i_idx, hij = _connected_row_cached(
                self.drt,
                self.h1e,
                self.eri,
                int(j),
                max_out=int(self.max_out),
                screening=self.screening,
                state_cache=self.state_cache,
                row_cache=self.row_cache,
            )
            i_idx = np.asarray(i_idx, dtype=np.int64).ravel()
            hij = np.asarray(hij, dtype=np.float64).ravel()
            if int(i_idx.size) == 0:
                continue
            mask = (i_idx >= int(label_lo)) & (i_idx < int(label_hi))
            if float(screen_contrib) > 0.0:
                mask &= np.abs(hij) * float(max_cj) >= float(screen_contrib)
            if not bool(np.any(mask)):
                continue
            label_parts.append(np.asarray(i_idx[mask], dtype=np.int64, order="C"))
            src_parts.append(np.full((int(np.count_nonzero(mask)),), int(col), dtype=np.int32))
            hij_parts.append(np.asarray(hij[mask], dtype=np.float64, order="C"))

        if not label_parts:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=np.float64),
            )
        return (
            np.asarray(np.concatenate(label_parts), dtype=np.int64, order="C"),
            np.asarray(np.concatenate(src_parts), dtype=np.int32, order="C"),
            np.asarray(np.concatenate(hij_parts), dtype=np.float64, order="C"),
        )

    def accumulate_host(
        self,
        *,
        sel_idx: np.ndarray,
        c_sel: np.ndarray,
        selected_set: set[int] | None = None,
        label_lo: int = 0,
        label_hi: int | None = None,
        screen_contrib: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        from asuka.sci.sparse_support import _connected_row_cached  # noqa: PLC0415

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        if c_sel.ndim != 2:
            raise ValueError("c_sel must have shape (nsel, nroots)")
        if int(c_sel.shape[0]) != int(sel_idx.size):
            raise ValueError("c_sel leading dimension must match sel_idx length")
        nroots = int(c_sel.shape[1])
        if label_hi is None:
            label_hi = int(getattr(self.drt, "ncsf"))
        label_lo = int(label_lo)
        label_hi = int(label_hi)
        if label_lo < 0 or label_hi < label_lo:
            raise ValueError("invalid label window")
        selected = set(int(x) for x in sel_idx.tolist()) if selected_set is None else {int(x) for x in selected_set}

        ext: dict[int, np.ndarray] = {}
        for col, j in enumerate(sel_idx.tolist()):
            cj = np.asarray(c_sel[col, :], dtype=np.float64)
            max_cj = float(np.max(np.abs(cj)))
            if max_cj == 0.0:
                continue
            i_idx, hij = _connected_row_cached(
                self.drt,
                self.h1e,
                self.eri,
                int(j),
                max_out=int(self.max_out),
                screening=self.screening,
                state_cache=self.state_cache,
                row_cache=self.row_cache,
            )
            for i, v in zip(i_idx.tolist(), hij.tolist(), strict=False):
                ii = int(i)
                if ii < label_lo or ii >= label_hi or ii in selected:
                    continue
                vv = float(v)
                if float(screen_contrib) > 0.0 and abs(vv) * max_cj < float(screen_contrib):
                    continue
                acc = ext.get(ii)
                if acc is None:
                    ext[ii] = vv * cj
                else:
                    acc += vv * cj

        if not ext:
            return np.zeros((0,), dtype=np.int64), np.zeros((0, nroots), dtype=np.float64)

        idx = np.asarray(sorted(ext.keys()), dtype=np.int64)
        vals = np.vstack([np.asarray(ext[int(ii)], dtype=np.float64) for ii in idx.tolist()])
        return np.asarray(idx, dtype=np.int64, order="C"), np.asarray(vals, dtype=np.float64, order="C")

    def accumulate_gpu(
        self,
        *,
        sel_idx: np.ndarray,
        c_sel: np.ndarray,
        label_lo: int = 0,
        label_hi: int | None = None,
        screen_contrib: float = 0.0,
        tuple_emitter: Any | None = None,
    ):
        try:
            import cupy as cp  # type: ignore[import-not-found]
        except Exception as e:  # pragma: no cover
            raise RuntimeError("cupy is required for ExactExternalProjectedApply.accumulate_gpu") from e

        sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
        c_sel = np.asarray(c_sel, dtype=np.float64, order="C")
        if c_sel.ndim != 2:
            raise ValueError("c_sel must have shape (nsel, nroots)")
        if int(c_sel.shape[0]) != int(sel_idx.size):
            raise ValueError("c_sel leading dimension must match sel_idx length")
        nroots = int(c_sel.shape[1])
        sel_sorted_d = None

        def _ensure_sel_sorted_d():
            nonlocal sel_sorted_d
            if sel_sorted_d is None:
                sel_sorted_d = cp.ascontiguousarray(
                    cp.sort(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
                )
            return sel_sorted_d

        def _reduce_scalar_tuples(labels_d, vals_d):
            labels_d = cp.ascontiguousarray(cp.asarray(labels_d, dtype=cp.uint64).ravel())
            vals_d = cp.ascontiguousarray(cp.asarray(vals_d, dtype=cp.float64).ravel())
            if int(labels_d.size) == 0:
                return cp.zeros((0,), dtype=cp.uint64), cp.zeros((0,), dtype=cp.float64)
            order_d = cp.argsort(labels_d)
            labels_sorted_d = cp.ascontiguousarray(labels_d[order_d].ravel())
            vals_sorted_d = cp.ascontiguousarray(vals_d[order_d].ravel())
            if int(labels_sorted_d.size) == 1:
                unique_labels_d = labels_sorted_d
                reduced_vals_d = vals_sorted_d
            else:
                boundaries = cp.nonzero(labels_sorted_d[1:] != labels_sorted_d[:-1])[0] + 1
                starts_d = cp.ascontiguousarray(
                    cp.concatenate((cp.asarray([0], dtype=cp.int64), cp.asarray(boundaries, dtype=cp.int64).ravel())).ravel()
                )
                unique_labels_d = cp.ascontiguousarray(labels_sorted_d[starts_d].ravel())
                reduced_vals_d = cp.ascontiguousarray(cp.add.reduceat(vals_sorted_d, starts_d))
            sel_sorted_local_d = _ensure_sel_sorted_d()
            if int(sel_sorted_local_d.size) > 0 and int(unique_labels_d.size) > 0:
                pos_d = cp.asarray(cp.searchsorted(sel_sorted_local_d, unique_labels_d), dtype=cp.int64).ravel()
                matched = (
                    (pos_d >= 0)
                    & (pos_d < int(sel_sorted_local_d.size))
                    & (sel_sorted_local_d[pos_d] == unique_labels_d)
                )
                if bool(cp.any(matched).item()):
                    keep_mask = ~matched
                    unique_labels_d = cp.ascontiguousarray(unique_labels_d[keep_mask].ravel())
                    reduced_vals_d = cp.ascontiguousarray(reduced_vals_d[keep_mask].ravel())
            return unique_labels_d, reduced_vals_d

        if tuple_emitter is not None:
            labels_d, src_pos_d, hij_d = tuple_emitter(
                sel_idx=sel_idx,
                c_sel=c_sel,
                label_lo=int(label_lo),
                label_hi=label_hi,
                screen_contrib=float(screen_contrib),
            )
            labels_d = cp.ascontiguousarray(cp.asarray(labels_d, dtype=cp.uint64).ravel())
            src_pos_d = cp.ascontiguousarray(cp.asarray(src_pos_d, dtype=cp.int32).ravel())
            hij_d = cp.ascontiguousarray(cp.asarray(hij_d, dtype=cp.float64).ravel())
            if int(labels_d.size) == 0:
                return cp.zeros((0,), dtype=cp.uint64), cp.zeros((0, nroots), dtype=cp.float64)
            c_sel_d = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
            # Tuple emitters may decompose one exact H[i,j] into multiple raw terms.
            # Reduce those terms to exact (label, src) matrix elements before applying
            # any |H_ij| * max|c_j| screening, otherwise thresholding depends on the
            # decomposition rather than the exact row value.
            if int(labels_d.size) == 1:
                pair_labels_d = labels_d
                pair_src_d = src_pos_d
                pair_hij_d = hij_d
            else:
                src_bits = max(1, int(max(0, int(sel_idx.size) - 1)).bit_length())
                label_bits = max(1, int(int(getattr(self.drt, "ncsf", 0))).bit_length())
                if int(label_bits + src_bits) <= 64:
                    pair_key_d = cp.ascontiguousarray(
                        (labels_d << cp.uint64(src_bits))
                        | src_pos_d.astype(cp.uint64, copy=False)
                    )
                    order_d = cp.asarray(cp.argsort(pair_key_d), dtype=cp.int64).ravel()
                    pair_key_sorted_d = cp.ascontiguousarray(pair_key_d[order_d].ravel())
                    hij_pair_sorted_d = cp.ascontiguousarray(hij_d[order_d].ravel())
                    if int(pair_key_sorted_d.size) == 1:
                        starts_d = cp.asarray([0], dtype=cp.int64)
                    else:
                        boundaries = cp.nonzero(pair_key_sorted_d[1:] != pair_key_sorted_d[:-1])[0] + 1
                        starts_d = cp.ascontiguousarray(
                            cp.concatenate((cp.asarray([0], dtype=cp.int64), cp.asarray(boundaries, dtype=cp.int64).ravel())).ravel()
                        )
                    pair_unique_d = cp.ascontiguousarray(pair_key_sorted_d[starts_d].ravel())
                    pair_labels_d = cp.ascontiguousarray((pair_unique_d >> cp.uint64(src_bits)).ravel())
                    src_mask = cp.uint64((1 << src_bits) - 1)
                    pair_src_d = cp.ascontiguousarray((pair_unique_d & src_mask).astype(cp.int32, copy=False).ravel())
                    pair_hij_d = cp.ascontiguousarray(cp.add.reduceat(hij_pair_sorted_d, starts_d))
                else:
                    pair_keys_d = cp.stack(
                        (
                            cp.ascontiguousarray(src_pos_d.astype(cp.uint64, copy=False).ravel()),
                            labels_d,
                        ),
                        axis=0,
                    )
                    order_d = cp.lexsort(pair_keys_d)
                    labels_pair_sorted_d = cp.ascontiguousarray(labels_d[order_d].ravel())
                    src_pair_sorted_d = cp.ascontiguousarray(src_pos_d[order_d].ravel())
                    hij_pair_sorted_d = cp.ascontiguousarray(hij_d[order_d].ravel())
                    boundaries = cp.nonzero(
                        (labels_pair_sorted_d[1:] != labels_pair_sorted_d[:-1])
                        | (src_pair_sorted_d[1:] != src_pair_sorted_d[:-1])
                    )[0] + 1
                    starts_d = cp.ascontiguousarray(
                        cp.concatenate((cp.asarray([0], dtype=cp.int64), cp.asarray(boundaries, dtype=cp.int64).ravel())).ravel()
                    )
                    pair_labels_d = cp.ascontiguousarray(labels_pair_sorted_d[starts_d].ravel())
                    pair_src_d = cp.ascontiguousarray(src_pair_sorted_d[starts_d].ravel())
                    pair_hij_d = cp.ascontiguousarray(cp.add.reduceat(hij_pair_sorted_d, starts_d))
            if float(screen_contrib) > 0.0 and int(pair_hij_d.size) > 0:
                max_abs_c_d = cp.ascontiguousarray(cp.max(cp.abs(c_sel_d), axis=1).ravel())
                keep_pair_d = cp.abs(pair_hij_d) * max_abs_c_d[pair_src_d] >= float(screen_contrib)
                if bool(cp.any(keep_pair_d).item()):
                    pair_labels_d = cp.ascontiguousarray(pair_labels_d[keep_pair_d].ravel())
                    pair_src_d = cp.ascontiguousarray(pair_src_d[keep_pair_d].ravel())
                    pair_hij_d = cp.ascontiguousarray(pair_hij_d[keep_pair_d].ravel())
                else:
                    return cp.zeros((0,), dtype=cp.uint64), cp.zeros((0, nroots), dtype=cp.float64)
            # pair_labels_d remains label-major after the earlier lexsort(src, label)
            # pair reduction, and boolean masking preserves that order.
            if int(nroots) == 1:
                contrib_scalar_d = cp.ascontiguousarray(pair_hij_d * c_sel_d[pair_src_d, 0])
                if int(pair_labels_d.size) == 1:
                    uniq_all_d = cp.ascontiguousarray(pair_labels_d.ravel())
                    vals_all_d = cp.ascontiguousarray(contrib_scalar_d[:, None])
                else:
                    boundaries = cp.nonzero(pair_labels_d[1:] != pair_labels_d[:-1])[0] + 1
                    starts_d = cp.ascontiguousarray(
                        cp.concatenate((cp.asarray([0], dtype=cp.int64), cp.asarray(boundaries, dtype=cp.int64).ravel())).ravel()
                    )
                    uniq_all_d = cp.ascontiguousarray(pair_labels_d[starts_d].ravel())
                    vals_all_d = cp.ascontiguousarray(cp.add.reduceat(contrib_scalar_d, starts_d)[:, None])
            else:
                contrib_d = cp.ascontiguousarray(pair_hij_d[:, None] * c_sel_d[pair_src_d, :])
                if int(pair_labels_d.size) == 1:
                    uniq_all_d = cp.ascontiguousarray(pair_labels_d.ravel())
                    vals_all_d = contrib_d
                else:
                    boundaries = cp.nonzero(pair_labels_d[1:] != pair_labels_d[:-1])[0] + 1
                    starts_d = cp.ascontiguousarray(
                        cp.concatenate((cp.asarray([0], dtype=cp.int64), cp.asarray(boundaries, dtype=cp.int64).ravel())).ravel()
                    )
                    uniq_all_d = cp.ascontiguousarray(pair_labels_d[starts_d].ravel())
                    vals_all_d = cp.ascontiguousarray(cp.add.reduceat(contrib_d, starts_d, axis=0))
            sel_sorted_local_d = _ensure_sel_sorted_d()
            if int(sel_sorted_local_d.size) > 0 and int(uniq_all_d.size) > 0:
                pos_d = cp.asarray(cp.searchsorted(sel_sorted_local_d, uniq_all_d), dtype=cp.int64).ravel()
                matched = (
                    (pos_d >= 0)
                    & (pos_d < int(sel_sorted_local_d.size))
                    & (sel_sorted_local_d[pos_d] == uniq_all_d)
                )
                if bool(cp.any(matched).item()):
                    keep_mask = ~matched
                    uniq_all_d = cp.ascontiguousarray(uniq_all_d[keep_mask].ravel())
                    vals_all_d = cp.ascontiguousarray(vals_all_d[keep_mask, :])
            return uniq_all_d, vals_all_d

        labels_h, src_pos_h, hij_h = self._emit_host_tuples(
            sel_idx=sel_idx,
            c_sel=c_sel,
            label_lo=int(label_lo),
            label_hi=label_hi,
            screen_contrib=float(screen_contrib),
        )
        if int(labels_h.size) == 0:
            return cp.zeros((0,), dtype=cp.uint64), cp.zeros((0, nroots), dtype=cp.float64)

        labels_d = cp.ascontiguousarray(cp.asarray(labels_h.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
        src_pos_d = cp.ascontiguousarray(cp.asarray(src_pos_h, dtype=cp.int32).ravel())
        hij_d = cp.ascontiguousarray(cp.asarray(hij_h, dtype=cp.float64).ravel())
        c_sel_d = cp.ascontiguousarray(cp.asarray(c_sel, dtype=cp.float64))
        order_d = cp.argsort(labels_d)
        labels_sorted_d = cp.ascontiguousarray(labels_d[order_d].ravel())
        if int(labels_sorted_d.size) == 0:
            return cp.zeros((0,), dtype=cp.uint64), cp.zeros((0, nroots), dtype=cp.float64)

        if int(nroots) == 1:
            contrib_scalar_d = cp.ascontiguousarray(hij_d * c_sel_d[src_pos_d, 0])
            contrib_sorted_d = cp.ascontiguousarray(contrib_scalar_d[order_d].ravel())
            if int(labels_sorted_d.size) == 1:
                unique_labels_d = labels_sorted_d
                reduced_vals_d = cp.ascontiguousarray(contrib_sorted_d[:, None])
            else:
                boundaries = cp.nonzero(labels_sorted_d[1:] != labels_sorted_d[:-1])[0] + 1
                starts_d = cp.ascontiguousarray(
                    cp.concatenate(
                        (
                            cp.asarray([0], dtype=cp.int64),
                            cp.asarray(boundaries, dtype=cp.int64).ravel(),
                        )
                    ).ravel()
                )
                unique_labels_d = cp.ascontiguousarray(labels_sorted_d[starts_d].ravel())
                reduced_vals_d = cp.ascontiguousarray(cp.add.reduceat(contrib_sorted_d, starts_d)[:, None])
        else:
            contrib_d = cp.ascontiguousarray(hij_d[:, None] * c_sel_d[src_pos_d, :])
            contrib_sorted_d = cp.ascontiguousarray(contrib_d[order_d, :])
            if int(labels_sorted_d.size) == 1:
                unique_labels_d = labels_sorted_d
                reduced_vals_d = contrib_sorted_d
            else:
                boundaries = cp.nonzero(labels_sorted_d[1:] != labels_sorted_d[:-1])[0] + 1
                starts_d = cp.ascontiguousarray(
                    cp.concatenate(
                        (
                            cp.asarray([0], dtype=cp.int64),
                            cp.asarray(boundaries, dtype=cp.int64).ravel(),
                        )
                    ).ravel()
                )
                unique_labels_d = cp.ascontiguousarray(labels_sorted_d[starts_d].ravel())
                reduced_vals_d = cp.ascontiguousarray(cp.add.reduceat(contrib_sorted_d, starts_d, axis=0))

        sel_sorted_local_d = _ensure_sel_sorted_d()
        if int(sel_sorted_local_d.size) > 0 and int(unique_labels_d.size) > 0:
            pos_d = cp.asarray(cp.searchsorted(sel_sorted_local_d, unique_labels_d), dtype=cp.int64).ravel()
            matched = (pos_d >= 0) & (pos_d < int(sel_sorted_local_d.size)) & (sel_sorted_local_d[pos_d] == unique_labels_d)
            if bool(cp.any(matched).item()):
                keep_mask = ~matched
                unique_labels_d = cp.ascontiguousarray(unique_labels_d[keep_mask].ravel())
                reduced_vals_d = cp.ascontiguousarray(reduced_vals_d[keep_mask, :])

        return unique_labels_d, reduced_vals_d

    @staticmethod
    def score_host(
        *,
        idx: np.ndarray,
        vals_root_major: np.ndarray,
        e_var: np.ndarray,
        hdiag_lookup: Any,
        denom_floor: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.asarray(idx, dtype=np.int64).ravel()
        vals_root_major = np.asarray(vals_root_major, dtype=np.float64, order="C")
        e_var = np.asarray(e_var, dtype=np.float64).ravel()
        if vals_root_major.ndim != 2:
            raise ValueError("vals_root_major must have shape (ncand, nroots)")
        if int(vals_root_major.shape[0]) != int(idx.size):
            raise ValueError("vals_root_major leading dimension must match idx length")
        if int(vals_root_major.shape[1]) != int(e_var.size):
            raise ValueError("vals_root_major second dimension must match e_var length")
        denom_floor = float(denom_floor)
        if denom_floor < 0.0:
            raise ValueError("denom_floor must be >= 0")

        nroots = int(e_var.size)
        ncand = int(idx.size)
        c1 = np.zeros((ncand, nroots), dtype=np.float64)
        w = np.zeros((ncand,), dtype=np.float64)
        e_pt2 = np.zeros((nroots,), dtype=np.float64)
        for pos, ii in enumerate(idx.tolist()):
            denom = np.asarray(e_var - float(hdiag_lookup.get(int(ii))), dtype=np.float64)
            if denom_floor > 0.0:
                small = np.abs(denom) < denom_floor
                if np.any(small):
                    denom = denom.copy()
                    denom[small] = np.where(denom[small] >= 0.0, denom_floor, -denom_floor)
            c1_i = np.asarray(vals_root_major[pos, :] / denom, dtype=np.float64)
            c1[pos, :] = c1_i
            w[pos] = float(np.max(np.abs(c1_i)))
            e_pt2 += (np.asarray(vals_root_major[pos, :], dtype=np.float64) ** 2) / denom
        return np.asarray(c1, dtype=np.float64, order="C"), np.asarray(w, dtype=np.float64), np.asarray(e_pt2, dtype=np.float64)
