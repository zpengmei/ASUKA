from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:  # optional
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    sp = None  # type: ignore


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
        sel_sorted_d = cp.ascontiguousarray(cp.sort(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel()))

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
            if int(sel_sorted_d.size) > 0 and int(unique_labels_d.size) > 0:
                pos_d = cp.asarray(cp.searchsorted(sel_sorted_d, unique_labels_d), dtype=cp.int64).ravel()
                matched = (pos_d >= 0) & (pos_d < int(sel_sorted_d.size)) & (sel_sorted_d[pos_d] == unique_labels_d)
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
            contrib_d = cp.ascontiguousarray(hij_d[:, None] * c_sel_d[src_pos_d, :])
            order_d = cp.argsort(labels_d)
            labels_sorted_d = cp.ascontiguousarray(labels_d[order_d].ravel())
            contrib_sorted_d = cp.ascontiguousarray(contrib_d[order_d, :])
            if int(labels_sorted_d.size) == 1:
                uniq_all_d = labels_sorted_d
                vals_all_d = contrib_sorted_d
            else:
                boundaries = cp.nonzero(labels_sorted_d[1:] != labels_sorted_d[:-1])[0] + 1
                starts_d = cp.ascontiguousarray(
                    cp.concatenate((cp.asarray([0], dtype=cp.int64), cp.asarray(boundaries, dtype=cp.int64).ravel())).ravel()
                )
                uniq_all_d = cp.ascontiguousarray(labels_sorted_d[starts_d].ravel())
                vals_all_d = cp.ascontiguousarray(cp.add.reduceat(contrib_sorted_d, starts_d, axis=0))
            if int(sel_sorted_d.size) > 0 and int(uniq_all_d.size) > 0:
                pos_d = cp.asarray(cp.searchsorted(sel_sorted_d, uniq_all_d), dtype=cp.int64).ravel()
                matched = (pos_d >= 0) & (pos_d < int(sel_sorted_d.size)) & (sel_sorted_d[pos_d] == uniq_all_d)
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
        contrib_d = hij_d[:, None] * c_sel_d[src_pos_d, :]

        order_d = cp.argsort(labels_d)
        labels_sorted_d = cp.ascontiguousarray(labels_d[order_d].ravel())
        contrib_sorted_d = cp.ascontiguousarray(contrib_d[order_d, :])
        if int(labels_sorted_d.size) == 0:
            return cp.zeros((0,), dtype=cp.uint64), cp.zeros((0, nroots), dtype=cp.float64)

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

        if int(sel_sorted_d.size) > 0 and int(unique_labels_d.size) > 0:
            pos_d = cp.asarray(cp.searchsorted(sel_sorted_d, unique_labels_d), dtype=cp.int64).ravel()
            matched = (pos_d >= 0) & (pos_d < int(sel_sorted_d.size)) & (sel_sorted_d[pos_d] == unique_labels_d)
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
