from __future__ import annotations

"""Packed AO-pair ("s2") utilities for DF 3-center tensors.

We store symmetric AO-pair tensors using the lower-triangular packed order:
  p(m,n) = m*(m+1)//2 + n, with m>=n.

Packed DF tensor layout (canonical):
  Qp: shape (naux, ntri), C-order, where ntri=nao*(nao+1)//2.

This module provides fast CUDA-backed pack/unpack helpers via the cuERI CUDA
extension, with NumPy fallbacks for CPU paths.
"""

from typing import Any, Literal
import os

import numpy as np

from .tri_packed import ntri_from_nao, nao_from_ntri

_QP_APPLY_C_KERNEL_F64 = None
_QP_EXTRACT_RC_KERNEL_F64 = None


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(str(name))
    if v is None:
        return bool(default)
    return str(v).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}


def ao_packed_s2_enabled() -> bool:
    """Global switch for packed DF tensors (ASUKA_DF_AO_PACKED_S2=1)."""

    return _env_flag("ASUKA_DF_AO_PACKED_S2", default=False)


def is_df_qp(B: Any) -> bool:
    """Return True if B looks like a packed Qp tensor (naux, ntri)."""

    sh = getattr(B, "shape", None)
    return sh is not None and len(tuple(sh)) == 2


def infer_nao_from_Qp(B_Qp: Any) -> int:
    sh = getattr(B_Qp, "shape", None)
    if sh is None or len(tuple(sh)) != 2:
        raise ValueError("B_Qp must have shape (naux, ntri)")
    return int(nao_from_ntri(int(sh[1])))


def _require_cueri_ext():
    from asuka.kernels.cueri import require_ext  # local import: optional extension

    return require_ext()


def pack_B_to_Qp(B: Any, *, layout: Literal["mnQ", "Qmn"], nao: int | None = None) -> Any:
    """Pack a symmetrized DF tensor into Qp.

    Parameters
    ----------
    B
        DF factors in either mnQ (nao,nao,naux) or Qmn (naux,nao,nao) layout.
    layout
        Input layout, "mnQ" or "Qmn".
    nao
        Required for CUDA paths to validate shapes; inferred for NumPy.
    """

    layout_s = str(layout).strip()
    if layout_s not in ("mnQ", "Qmn"):
        raise ValueError("layout must be 'mnQ' or 'Qmn'")

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    if cp is not None and isinstance(B, cp.ndarray):  # type: ignore[attr-defined]
        B_dev = cp.asarray(B, dtype=cp.float64)
        if not bool(getattr(B_dev, "flags", None).c_contiguous):
            B_dev = cp.ascontiguousarray(B_dev)
        if B_dev.ndim != 3:
            raise ValueError("B must be 3D for pack_B_to_Qp")
        if layout_s == "mnQ":
            nao0, nao1, naux = map(int, B_dev.shape)
            if nao0 != nao1:
                raise ValueError("mnQ input must be (nao,nao,naux)")
            if nao is not None and int(nao) != int(nao0):
                raise ValueError("nao mismatch")
            nao_i = int(nao0)
        else:
            naux, nao0, nao1 = map(int, B_dev.shape)
            if nao0 != nao1:
                raise ValueError("Qmn input must be (naux,nao,nao)")
            if nao is not None and int(nao) != int(nao0):
                raise ValueError("nao mismatch")
            nao_i = int(nao0)

        ntri = ntri_from_nao(int(nao_i))
        out = cp.empty((int(naux), int(ntri)), dtype=cp.float64)
        _ext = _require_cueri_ext()

        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        if layout_s == "mnQ":
            _ext.df_pack_mnq_to_qp_device(
                B_dev.reshape(-1),
                out.reshape(-1),
                int(nao_i),
                int(naux),
                int(threads),
                int(stream_ptr),
                False,
            )
        else:
            _ext.df_pack_qmn_to_qp_device(
                B_dev.reshape(-1),
                out.reshape(-1),
                int(naux),
                int(nao_i),
                int(threads),
                int(stream_ptr),
                False,
            )
        return out

    # NumPy fallback
    B_np = np.asarray(B, dtype=np.float64, order="C")
    if B_np.ndim != 3:
        raise ValueError("B must be 3D for pack_B_to_Qp")
    if layout_s == "mnQ":
        nao0, nao1, naux = map(int, B_np.shape)
        if nao0 != nao1:
            raise ValueError("mnQ input must be (nao,nao,naux)")
        nao_i = int(nao0)
        tri_i, tri_j = np.tril_indices(nao_i)
        # (ntri,naux) -> (naux,ntri)
        return np.asarray(B_np[tri_i, tri_j, :].T, dtype=np.float64, order="C")
    naux, nao0, nao1 = map(int, B_np.shape)
    if nao0 != nao1:
        raise ValueError("Qmn input must be (naux,nao,nao)")
    nao_i = int(nao0)
    tri_i, tri_j = np.tril_indices(nao_i)
    return np.asarray(B_np[:, tri_i, tri_j], dtype=np.float64, order="C")


def pack_Lf_block_to_Qp(
    Lf_block: Any,
    out_Qp: Any,
    *,
    naux: int,
    nao: int,
    q0: int,
    q_count: int,
    threads: int = 256,
    sync: bool = False,
) -> Any:
    """Pack an L_f block into a Qp slice.

    Parameters
    ----------
    Lf_block
        Array with shape (nao, q_count*nao), holding concatenated MO blocks:
          Lf_block[m, q*nao + n] = L[q, m, n]
    out_Qp
        Destination packed tensor with shape (naux, nao*(nao+1)//2).
        This function writes rows q0..q0+q_count-1.
    """

    nao_i = int(nao)
    naux_i = int(naux)
    q0_i = int(q0)
    q_count_i = int(q_count)
    if nao_i < 0 or naux_i < 0 or q0_i < 0 or q_count_i < 0:
        raise ValueError("invalid nao/naux/q0/q_count")
    if q0_i > naux_i or q_count_i > (naux_i - q0_i):
        raise ValueError("q0+q_count must be <= naux")
    ntri = int(ntri_from_nao(int(nao_i)))

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    if cp is not None and isinstance(Lf_block, cp.ndarray):  # type: ignore[attr-defined]
        Lf = cp.asarray(Lf_block, dtype=cp.float64)
        if Lf.ndim != 2:
            raise ValueError("Lf_block must be 2D (nao, q_count*nao)")
        if tuple(map(int, Lf.shape)) != (nao_i, q_count_i * nao_i):
            raise ValueError("Lf_block shape mismatch")
        if not bool(getattr(Lf, "flags", None).c_contiguous):
            Lf = cp.ascontiguousarray(Lf)

        out = cp.asarray(out_Qp, dtype=cp.float64)
        if out.ndim != 2:
            raise ValueError("out_Qp must be 2D (naux,ntri)")
        if tuple(map(int, out.shape)) != (naux_i, ntri):
            raise ValueError("out_Qp shape mismatch")
        if not bool(getattr(out, "flags", None).c_contiguous):
            out = cp.ascontiguousarray(out)

        _ext = _require_cueri_ext()

        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        _ext.df_pack_lf_block_to_qp_device(
            Lf.reshape(-1),
            out.reshape(-1),
            int(naux_i),
            int(nao_i),
            int(q0_i),
            int(q_count_i),
            int(threads),
            int(stream_ptr),
            bool(sync),
        )
        return out

    # NumPy fallback
    Lf_np = np.asarray(Lf_block, dtype=np.float64, order="C")
    out_np = np.asarray(out_Qp, dtype=np.float64, order="C")
    if Lf_np.ndim != 2:
        raise ValueError("Lf_block must be 2D (nao, q_count*nao)")
    if tuple(map(int, Lf_np.shape)) != (nao_i, q_count_i * nao_i):
        raise ValueError("Lf_block shape mismatch")
    if out_np.ndim != 2:
        raise ValueError("out_Qp must be 2D (naux,ntri)")
    if tuple(map(int, out_np.shape)) != (naux_i, ntri):
        raise ValueError("out_Qp shape mismatch")
    tri_i, tri_j = np.tril_indices(nao_i)
    for q_local in range(q_count_i):
        q_abs = int(q0_i + q_local)
        out_np[q_abs, :] = Lf_np[tri_i, int(q_local * nao_i) + tri_j]
    return out_np


def unpack_Qp_to_mnQ(B_Qp: Any, *, nao: int) -> Any:
    """Unpack Qp -> full symmetric mnQ (nao,nao,naux)."""

    nao_i = int(nao)
    if nao_i < 0:
        raise ValueError("nao must be >= 0")
    ntri = ntri_from_nao(nao_i)

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    if cp is not None and isinstance(B_Qp, cp.ndarray):  # type: ignore[attr-defined]
        inp = cp.asarray(B_Qp, dtype=cp.float64)
        if not bool(getattr(inp, "flags", None).c_contiguous):
            inp = cp.ascontiguousarray(inp)
        if inp.ndim != 2:
            raise ValueError("B_Qp must be 2D (naux,ntri)")
        naux, ntri_in = map(int, inp.shape)
        if int(ntri_in) != int(ntri):
            raise ValueError("B_Qp ntri mismatch")
        out = cp.empty((nao_i, nao_i, naux), dtype=cp.float64)
        _ext = _require_cueri_ext()

        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        _ext.df_unpack_qp_to_mnq_device(
            inp.reshape(-1),
            out.reshape(-1),
            int(naux),
            int(nao_i),
            int(threads),
            int(stream_ptr),
            False,
        )
        return out

    # NumPy fallback
    inp = np.asarray(B_Qp, dtype=np.float64, order="C")
    if inp.ndim != 2:
        raise ValueError("B_Qp must be 2D (naux,ntri)")
    naux, ntri_in = map(int, inp.shape)
    if int(ntri_in) != int(ntri):
        raise ValueError("B_Qp ntri mismatch")
    out = np.empty((nao_i, nao_i, naux), dtype=np.float64, order="C")
    tri_i, tri_j = np.tril_indices(nao_i)
    out[tri_i, tri_j, :] = inp.T
    out[tri_j, tri_i, :] = inp.T
    return out


def unpack_Qp_to_Qmn_block(B_Qp: Any, *, nao: int, q0: int, q_count: int, out: Any | None = None) -> Any:
    """Unpack a (q0,q_count) aux block from Qp into Qmn block (q_count,nao,nao)."""

    nao_i = int(nao)
    q0_i = int(q0)
    q_count_i = int(q_count)
    if nao_i < 0 or q0_i < 0 or q_count_i < 0:
        raise ValueError("invalid nao/q0/q_count")
    ntri = ntri_from_nao(nao_i)

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    if cp is not None and isinstance(B_Qp, cp.ndarray):  # type: ignore[attr-defined]
        inp = cp.asarray(B_Qp, dtype=cp.float64)
        if not bool(getattr(inp, "flags", None).c_contiguous):
            inp = cp.ascontiguousarray(inp)
        if inp.ndim != 2:
            raise ValueError("B_Qp must be 2D (naux,ntri)")
        naux, ntri_in = map(int, inp.shape)
        if int(ntri_in) != int(ntri):
            raise ValueError("B_Qp ntri mismatch")
        if q0_i > naux or q_count_i > (naux - q0_i):
            raise ValueError("q0+q_count must be <= naux")

        if out is None:
            out_dev = cp.empty((q_count_i, nao_i, nao_i), dtype=cp.float64)
        else:
            out_dev = cp.asarray(out, dtype=cp.float64)
            if tuple(map(int, out_dev.shape)) != (q_count_i, nao_i, nao_i):
                raise ValueError("out shape mismatch")
            if not bool(getattr(out_dev, "flags", None).c_contiguous):
                out_dev = cp.ascontiguousarray(out_dev)

        _ext = _require_cueri_ext()

        threads = 256
        stream_ptr = int(cp.cuda.get_current_stream().ptr)
        _ext.df_unpack_qp_to_qmn_block_device(
            inp.reshape(-1),
            out_dev.reshape(-1),
            int(naux),
            int(nao_i),
            int(q0_i),
            int(q_count_i),
            int(threads),
            int(stream_ptr),
            False,
        )
        return out_dev

    # NumPy fallback
    inp = np.asarray(B_Qp, dtype=np.float64, order="C")
    if inp.ndim != 2:
        raise ValueError("B_Qp must be 2D (naux,ntri)")
    naux, ntri_in = map(int, inp.shape)
    if int(ntri_in) != int(ntri):
        raise ValueError("B_Qp ntri mismatch")
    if q0_i > naux or q_count_i > (naux - q0_i):
        raise ValueError("q0+q_count must be <= naux")
    if out is None:
        out_np = np.empty((q_count_i, nao_i, nao_i), dtype=np.float64, order="C")
    else:
        out_np = np.asarray(out, dtype=np.float64, order="C")
        if tuple(map(int, out_np.shape)) != (q_count_i, nao_i, nao_i):
            raise ValueError("out shape mismatch")
    tri_i, tri_j = np.tril_indices(nao_i)
    blk = inp[q0_i : q0_i + q_count_i, :]
    out_np[:] = 0.0
    out_np[:, tri_i, tri_j] = blk
    out_np[:, tri_j, tri_i] = blk
    return out_np


def apply_Qp_to_C_block(
    B_Qp: Any,
    C: Any,
    *,
    nao: int,
    q0: int,
    q_count: int,
    out: Any | None = None,
) -> Any:
    """Compute X[q,mu,i] = sum_nu B[q,mu,nu] * C[nu,i] from packed Qp.

    Inputs
    - B_Qp: (naux, ntri)
    - C:    (nao, nmo)
    Output
    - X:    (q_count, nao, nmo)
    """

    nao_i = int(nao)
    q0_i = int(q0)
    q_count_i = int(q_count)
    if nao_i < 0 or q0_i < 0 or q_count_i < 0:
        raise ValueError("invalid nao/q0/q_count")

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    ntri_expected = int(ntri_from_nao(int(nao_i)))
    if cp is not None and isinstance(B_Qp, cp.ndarray):  # type: ignore[attr-defined]
        global _QP_APPLY_C_KERNEL_F64

        B = cp.asarray(B_Qp, dtype=cp.float64)
        C_dev = cp.asarray(C, dtype=cp.float64)
        if B.ndim != 2:
            raise ValueError("B_Qp must be 2D (naux,ntri)")
        if C_dev.ndim != 2:
            raise ValueError("C must be 2D (nao,nmo)")
        naux, ntri = map(int, B.shape)
        nao_c, nmo = map(int, C_dev.shape)
        if int(nao_c) != int(nao_i):
            raise ValueError("C nao mismatch")
        if int(ntri) != int(ntri_expected):
            raise ValueError("B_Qp ntri mismatch")
        if q0_i > naux or q_count_i > (naux - q0_i):
            raise ValueError("q0+q_count must be <= naux")

        if not bool(getattr(B, "flags", None).c_contiguous):
            B = cp.ascontiguousarray(B)
        if not bool(getattr(C_dev, "flags", None).c_contiguous):
            C_dev = cp.ascontiguousarray(C_dev)

        if out is None:
            X = cp.empty((q_count_i, nao_i, nmo), dtype=cp.float64)
        else:
            X = cp.asarray(out, dtype=cp.float64)
            if tuple(map(int, X.shape)) != (q_count_i, nao_i, nmo):
                raise ValueError("out shape mismatch")
            if not bool(getattr(X, "flags", None).c_contiguous):
                X = cp.ascontiguousarray(X)

        if _QP_APPLY_C_KERNEL_F64 is None:
            _QP_APPLY_C_KERNEL_F64 = cp.RawKernel(
                r"""
                extern "C" __global__
                void qp_apply_c_f64(
                    const double* __restrict__ b_qp,
                    const double* __restrict__ c,
                    double* __restrict__ x,
                    int naux, int nao, int nmo, int ntri, int q0, int q_count
                ) {
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    int total = q_count * nao * nmo;
                    if (idx >= total) return;
                    int t = idx;
                    int i = t % nmo; t /= nmo;
                    int mu = t % nao; t /= nao;
                    int q = t;
                    int q_abs = q0 + q;
                    if (q_abs < 0 || q_abs >= naux) return;
                    const double* b_row = b_qp + (size_t)q_abs * (size_t)ntri;
                    double acc = 0.0;
                    for (int nu = 0; nu < nao; ++nu) {
                        int a = mu;
                        int b = nu;
                        if (b > a) { int tmp = a; a = b; b = tmp; }
                        int p = (a * (a + 1)) / 2 + b;
                        acc += b_row[p] * c[(size_t)nu * (size_t)nmo + (size_t)i];
                    }
                    x[idx] = acc;
                }
                """,
                "qp_apply_c_f64",
            )

        total = int(q_count_i * nao_i * nmo)
        threads = 256
        blocks = (total + threads - 1) // threads
        _QP_APPLY_C_KERNEL_F64(
            (int(blocks),),
            (int(threads),),
            (
                B,
                C_dev,
                X,
                np.int32(naux),
                np.int32(nao_i),
                np.int32(nmo),
                np.int32(ntri_expected),
                np.int32(q0_i),
                np.int32(q_count_i),
            ),
        )
        return X

    # NumPy fallback
    B_np = np.asarray(B_Qp, dtype=np.float64, order="C")
    C_np = np.asarray(C, dtype=np.float64, order="C")
    if B_np.ndim != 2:
        raise ValueError("B_Qp must be 2D (naux,ntri)")
    if C_np.ndim != 2:
        raise ValueError("C must be 2D (nao,nmo)")
    naux, ntri = map(int, B_np.shape)
    nao_c, nmo = map(int, C_np.shape)
    if int(nao_c) != int(nao_i):
        raise ValueError("C nao mismatch")
    if int(ntri) != int(ntri_expected):
        raise ValueError("B_Qp ntri mismatch")
    if q0_i > naux or q_count_i > (naux - q0_i):
        raise ValueError("q0+q_count must be <= naux")
    B_qmn = unpack_Qp_to_Qmn_block(B_np, nao=int(nao_i), q0=int(q0_i), q_count=int(q_count_i))
    X_np = np.matmul(B_qmn, C_np)
    if out is not None:
        out_np = np.asarray(out, dtype=np.float64, order="C")
        if tuple(map(int, out_np.shape)) != tuple(map(int, X_np.shape)):
            raise ValueError("out shape mismatch")
        out_np[...] = X_np
        return out_np
    return np.asarray(X_np, dtype=np.float64, order="C")


def extract_Qp_rows_cols_block(
    B_Qp: Any,
    *,
    nao: int,
    q0: int,
    q_count: int,
    row0: int,
    row_count: int,
    col0: int,
    col_count: int,
    out: Any | None = None,
) -> Any:
    """Extract a dense sub-block B[q, rows, cols] from packed Qp storage.

    Inputs
    - B_Qp: (naux, ntri)
    Output
    - out:  (q_count, row_count, col_count)
    """

    nao_i = int(nao)
    q0_i = int(q0)
    q_count_i = int(q_count)
    row0_i = int(row0)
    row_count_i = int(row_count)
    col0_i = int(col0)
    col_count_i = int(col_count)
    if nao_i < 0 or q0_i < 0 or q_count_i < 0 or row0_i < 0 or row_count_i < 0 or col0_i < 0 or col_count_i < 0:
        raise ValueError("invalid nao/q/row/col arguments")
    if row0_i > nao_i or row_count_i > (nao_i - row0_i):
        raise ValueError("row range out of bounds")
    if col0_i > nao_i or col_count_i > (nao_i - col0_i):
        raise ValueError("col range out of bounds")

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    ntri_expected = int(ntri_from_nao(int(nao_i)))
    if cp is not None and isinstance(B_Qp, cp.ndarray):  # type: ignore[attr-defined]
        global _QP_EXTRACT_RC_KERNEL_F64

        B = cp.asarray(B_Qp, dtype=cp.float64)
        if B.ndim != 2:
            raise ValueError("B_Qp must be 2D (naux,ntri)")
        naux, ntri = map(int, B.shape)
        if int(ntri) != int(ntri_expected):
            raise ValueError("B_Qp ntri mismatch")
        if q0_i > naux or q_count_i > (naux - q0_i):
            raise ValueError("q0+q_count must be <= naux")

        if not bool(getattr(B, "flags", None).c_contiguous):
            B = cp.ascontiguousarray(B)

        if out is None:
            out_dev = cp.empty((q_count_i, row_count_i, col_count_i), dtype=cp.float64)
        else:
            out_dev = cp.asarray(out, dtype=cp.float64)
            if tuple(map(int, out_dev.shape)) != (q_count_i, row_count_i, col_count_i):
                raise ValueError("out shape mismatch")
            if not bool(getattr(out_dev, "flags", None).c_contiguous):
                out_dev = cp.ascontiguousarray(out_dev)

        if _QP_EXTRACT_RC_KERNEL_F64 is None:
            _QP_EXTRACT_RC_KERNEL_F64 = cp.RawKernel(
                r"""
                extern "C" __global__
                void qp_extract_rows_cols_f64(
                    const double* __restrict__ b_qp,
                    double* __restrict__ out,
                    int naux, int nao, int ntri, int q0, int q_count,
                    int row0, int row_count, int col0, int col_count
                ) {
                    int idx = blockDim.x * blockIdx.x + threadIdx.x;
                    int total = q_count * row_count * col_count;
                    if (idx >= total) return;

                    int t = idx;
                    int c = t % col_count; t /= col_count;
                    int r = t % row_count; t /= row_count;
                    int q = t;

                    int q_abs = q0 + q;
                    int mu = row0 + r;
                    int nu = col0 + c;
                    if (q_abs < 0 || q_abs >= naux || mu < 0 || mu >= nao || nu < 0 || nu >= nao) {
                        return;
                    }
                    int a = mu;
                    int b = nu;
                    if (b > a) { int tmp = a; a = b; b = tmp; }
                    int p = (a * (a + 1)) / 2 + b;
                    out[idx] = b_qp[(size_t)q_abs * (size_t)ntri + (size_t)p];
                }
                """,
                "qp_extract_rows_cols_f64",
            )

        total = int(q_count_i * row_count_i * col_count_i)
        threads = 256
        blocks = (total + threads - 1) // threads
        _QP_EXTRACT_RC_KERNEL_F64(
            (int(blocks),),
            (int(threads),),
            (
                B,
                out_dev,
                np.int32(naux),
                np.int32(nao_i),
                np.int32(ntri_expected),
                np.int32(q0_i),
                np.int32(q_count_i),
                np.int32(row0_i),
                np.int32(row_count_i),
                np.int32(col0_i),
                np.int32(col_count_i),
            ),
        )
        return out_dev

    # NumPy fallback
    B_np = np.asarray(B_Qp, dtype=np.float64, order="C")
    if B_np.ndim != 2:
        raise ValueError("B_Qp must be 2D (naux,ntri)")
    naux, ntri = map(int, B_np.shape)
    if int(ntri) != int(ntri_expected):
        raise ValueError("B_Qp ntri mismatch")
    if q0_i > naux or q_count_i > (naux - q0_i):
        raise ValueError("q0+q_count must be <= naux")

    if out is None:
        out_np = np.empty((q_count_i, row_count_i, col_count_i), dtype=np.float64, order="C")
    else:
        out_np = np.asarray(out, dtype=np.float64, order="C")
        if tuple(map(int, out_np.shape)) != (q_count_i, row_count_i, col_count_i):
            raise ValueError("out shape mismatch")

    rows = np.arange(row0_i, row0_i + row_count_i, dtype=np.int64).reshape(-1, 1)
    cols = np.arange(col0_i, col0_i + col_count_i, dtype=np.int64).reshape(1, -1)
    aa = np.maximum(rows, cols)
    bb = np.minimum(rows, cols)
    p_idx = (aa * (aa + 1) // 2 + bb).reshape(-1)
    blk = B_np[q0_i : q0_i + q_count_i, :]
    out_np[...] = blk[:, p_idx].reshape(q_count_i, row_count_i, col_count_i)
    return out_np


def fused_qp_l_act_f64(
    B_Qp: Any,
    C_act: Any,
    *,
    nao: int,
    q0: int,
    q_count: int,
) -> Any:
    """Compute L_act[q,u,v] = C_act.T @ unpack(B_Qp[q0:q0+q_count]) @ C_act.

    No (q,nao,nao) or (q,nao,ncas) intermediate is allocated on GPU.
    Falls back to apply_Qp_to_C_block + matmul on CPU/when cuERI ext unavailable.

    Inputs
    - B_Qp:  (naux, ntri)
    - C_act: (nao, ncas)
    Output
    - L_act: (q_count, ncas, ncas)
    """
    nao_i = int(nao)
    q0_i = int(q0)
    q_count_i = int(q_count)
    if nao_i < 0 or q0_i < 0 or q_count_i < 0:
        raise ValueError("invalid nao/q0/q_count")
    if q_count_i == 0:
        ncas = int(C_act.shape[1])
        return np.zeros((0, ncas, ncas), dtype=np.float64)

    ntri_expected = int(ntri_from_nao(nao_i))
    ncas = int(C_act.shape[1])

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    if cp is not None and isinstance(B_Qp, cp.ndarray):  # type: ignore[attr-defined]
        _ext = _require_cueri_ext()
        B = cp.asarray(B_Qp, dtype=cp.float64)
        C = cp.asarray(C_act, dtype=cp.float64)
        if not bool(getattr(B, "flags", None).c_contiguous):
            B = cp.ascontiguousarray(B)
        if not bool(getattr(C, "flags", None).c_contiguous):
            C = cp.ascontiguousarray(C)
        naux = int(B.shape[0])
        if q0_i > naux or q_count_i > naux - q0_i:
            raise ValueError("q0+q_count must be <= naux")
        L = cp.zeros((q_count_i, ncas, ncas), dtype=cp.float64)
        _ext.df_fused_qp_l_act_device(
            B.ravel(),
            C.ravel(),
            L.ravel(),
            int(naux),
            int(nao_i),
            int(ncas),
            int(ntri_expected),
            int(q0_i),
            int(q_count_i),
            16,  # tile
            cp.cuda.get_current_stream().ptr,
            False,
        )
        return L

    # CPU fallback: apply_Qp_to_C_block + batched matmul
    X = apply_Qp_to_C_block(B_Qp, C_act, nao=nao_i, q0=q0_i, q_count=q_count_i)
    C_np = np.asarray(C_act, dtype=np.float64)
    return np.matmul(C_np.T[None, :, :], X)


def fused_qp_exchange_sym_f64(
    B_Qp: Any,
    D1: Any,
    D2: Any,
    out_Qp: Any,
    *,
    nao: int,
    q0: int,
    q_count: int,
    alpha: float = 1.0,
) -> None:
    """Accumulate alpha*(D1@B@D2 + D2@B@D1) into out_Qp in packed Qp format.

    The result is symmetric (D1, D2, B all symmetric), so it maps directly
    to Qp without a separate pack step.  out_Qp is updated in-place.

    Inputs
    - B_Qp:   (naux, ntri) packed DF tensor (read-only for rows q0..q0+q_count-1)
    - D1, D2: (nao, nao) density matrices
    - out_Qp: (naux, ntri) accumulated output — modified in place
    """
    nao_i = int(nao)
    q0_i = int(q0)
    q_count_i = int(q_count)
    alpha_f = float(alpha)
    if nao_i < 0 or q0_i < 0 or q_count_i < 0:
        raise ValueError("invalid nao/q0/q_count")
    if q_count_i == 0:
        return

    ntri_expected = int(ntri_from_nao(nao_i))

    try:
        import cupy as cp  # type: ignore
    except Exception:
        cp = None  # type: ignore

    if cp is not None and isinstance(B_Qp, cp.ndarray):  # type: ignore[attr-defined]
        _ext = _require_cueri_ext()
        B = cp.asarray(B_Qp, dtype=cp.float64)
        d1 = cp.asarray(D1, dtype=cp.float64)
        d2 = cp.asarray(D2, dtype=cp.float64)
        out = cp.asarray(out_Qp, dtype=cp.float64)
        if not bool(getattr(B, "flags", None).c_contiguous):
            B = cp.ascontiguousarray(B)
        if not bool(getattr(d1, "flags", None).c_contiguous):
            d1 = cp.ascontiguousarray(d1)
        if not bool(getattr(d2, "flags", None).c_contiguous):
            d2 = cp.ascontiguousarray(d2)
        if not bool(getattr(out, "flags", None).c_contiguous):
            out = cp.ascontiguousarray(out)
        naux = int(B.shape[0])
        if q0_i > naux or q_count_i > naux - q0_i:
            raise ValueError("q0+q_count must be <= naux")
        _ext.df_fused_qp_exchange_sym_device(
            B.ravel(),
            d1.ravel(),
            d2.ravel(),
            out.ravel(),
            int(naux),
            int(nao_i),
            int(ntri_expected),
            int(q0_i),
            int(q_count_i),
            alpha_f,
            256,  # threads
            cp.cuda.get_current_stream().ptr,
            False,
        )
        # Write back if out is a copy (non-contiguous input case)
        if out is not out_Qp:
            cp.copyto(out_Qp, out)
        return

    # CPU fallback: unpack + matmul chain + pack lower triangle
    B_np = np.asarray(B_Qp, dtype=np.float64, order="C")
    D1_np = np.asarray(D1, dtype=np.float64, order="C")
    D2_np = np.asarray(D2, dtype=np.float64, order="C")
    bq = unpack_Qp_to_Qmn_block(B_np, nao=nao_i, q0=q0_i, q_count=q_count_i)
    t = np.matmul(np.matmul(D1_np[None, :, :], bq), D2_np)
    t += np.matmul(np.matmul(D2_np[None, :, :], bq), D1_np)
    t *= alpha_f
    tri_i, tri_j = np.tril_indices(nao_i)
    out_Qp_np = np.asarray(out_Qp)
    out_Qp_np[q0_i : q0_i + q_count_i] += t[:, tri_i, tri_j]


__all__ = [
    "ao_packed_s2_enabled",
    "is_df_qp",
    "infer_nao_from_Qp",
    "apply_Qp_to_C_block",
    "extract_Qp_rows_cols_block",
    "fused_qp_l_act_f64",
    "fused_qp_exchange_sym_f64",
    "pack_B_to_Qp",
    "pack_Lf_block_to_Qp",
    "unpack_Qp_to_mnQ",
    "unpack_Qp_to_Qmn_block",
]
