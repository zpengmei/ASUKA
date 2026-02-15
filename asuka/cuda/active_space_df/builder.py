from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DeviceActiveDF:
    """GPU-resident active-space DF integrals.

    Arrays are expected to be CuPy ndarrays.
    """

    norb: int
    naux: int
    l_full: Any  # (norb*norb, naux)
    j_ps: Any | None  # (norb, norb)
    pair_norm: Any | None  # (norb*norb,)
    eri_mat: Any | None  # (norb*norb, norb*norb)


class ActiveSpaceDFBuilder:
    """Build active-space DF integrals on GPU via `gpu4pyscf` AO DF blocks.

    This builder focuses on producing `L_full[pq,L]` in ordered-pair indexing, and
    optionally `ERI_mat` and `J_ps`. It is intended to be called once per CASSCF
    macro-iteration (as active MOs change), while caching the AO DF object across
    iterations.

    Parameters
    ----------
    dfobj:
        Optional pre-built `gpu4pyscf.df.DF` object to reuse (avoids paying the AO DF
        build cost again). If not provided, the builder constructs one on demand.
    """

    def __init__(
        self,
        mol,
        *,
        dfobj=None,
        auxbasis: Any = "weigend+etb",
        direct_scf_tol: float = 1e-12,
        df_blksize: int | None = None,
    ) -> None:
        self.mol = mol
        self.auxbasis = auxbasis
        self.direct_scf_tol = float(direct_scf_tol)
        self.df_blksize = None if df_blksize is None else int(df_blksize)

        self._dfobj = dfobj

    @property
    def dfobj(self):
        if self._dfobj is None:
            from gpu4pyscf import df as gdf

            self._dfobj = gdf.DF(self.mol, auxbasis=self.auxbasis).build(direct_scf_tol=self.direct_scf_tol)
        return self._dfobj

    def build(
        self,
        c_cas,
        *,
        method: str = "auto",
        cublas_math_mode: str | None = None,
        want_eri_mat: bool = True,
        want_j_ps: bool = True,
        want_pair_norm: bool = True,
        out: DeviceActiveDF | None = None,
    ) -> DeviceActiveDF:
        import cupy as cp
        from cupy_backends.cuda.libs import cublas as cublas_lib

        c_cas = cp.asarray(c_cas)
        if c_cas.ndim != 2:
            raise ValueError("c_cas must be 2D (nao,norb)")
        if out is not None:
            exp_dtype = out.l_full.dtype
            if exp_dtype not in (cp.float64, cp.complex128):
                raise ValueError(f"out.l_full has unsupported dtype {exp_dtype} (expected float64/complex128)")
            if c_cas.dtype != exp_dtype:
                c_cas = c_cas.astype(exp_dtype)
        else:
            if c_cas.dtype not in (cp.float64, cp.complex128):
                if cp.iscomplexobj(c_cas):
                    c_cas = c_cas.astype(cp.complex128)
                else:
                    c_cas = c_cas.astype(cp.float64)

        nao, norb = map(int, c_cas.shape)
        if nao != int(self.mol.nao_nr()):
            raise ValueError(f"c_cas has nao={nao}, expected mol.nao_nr()={int(self.mol.nao_nr())}")
        if norb <= 0:
            raise ValueError("norb must be > 0")

        nops = norb * norb
        dfobj = self.dfobj
        naux = int(getattr(dfobj, "naux", 0) or dfobj.auxmol.nao_nr())

        # gpu4pyscf internally reorders AO functions for performance. The streamed
        # AO CDERI blocks (`Lij`) are in this sorted AO order, so we must apply the
        # same permutation to `C_cas` before contracting.
        ao_idx = getattr(getattr(dfobj, "intopt", None), "_ao_idx", None)
        if ao_idx is not None:
            c_cas = c_cas[ao_idx]

        method = str(method).lower()
        if method not in ("auto", "dense", "pair_gemm"):
            raise ValueError("method must be one of: 'auto', 'dense', 'pair_gemm'")

        if out is not None:
            if int(out.norb) != norb:
                raise ValueError(f"out.norb={int(out.norb)} does not match norb={norb}")
            if int(out.naux) != naux:
                raise ValueError(f"out.naux={int(out.naux)} does not match naux={naux}")
            if tuple(out.l_full.shape) != (nops, naux):
                raise ValueError(f"out.l_full has shape {tuple(out.l_full.shape)}, expected {(nops, naux)}")
            if out.l_full.dtype != c_cas.dtype:
                raise ValueError(f"out.l_full dtype {out.l_full.dtype} does not match c_cas dtype {c_cas.dtype}")
            if not bool(out.l_full.flags.c_contiguous):
                raise ValueError("out.l_full must be C-contiguous")
            if bool(want_eri_mat) != (out.eri_mat is not None):
                raise ValueError("out.eri_mat presence does not match want_eri_mat")
            if bool(want_j_ps) != (out.j_ps is not None):
                raise ValueError("out.j_ps presence does not match want_j_ps")
            if bool(want_pair_norm) != (out.pair_norm is not None):
                raise ValueError("out.pair_norm presence does not match want_pair_norm")
            if out.eri_mat is not None and not bool(out.eri_mat.flags.c_contiguous):
                raise ValueError("out.eri_mat must be C-contiguous")
            if out.j_ps is not None and not bool(out.j_ps.flags.c_contiguous):
                raise ValueError("out.j_ps must be C-contiguous")
            if out.pair_norm is not None:
                if out.pair_norm.dtype != cp.float64:
                    raise ValueError(f"out.pair_norm dtype {out.pair_norm.dtype} must be float64")
                if not bool(out.pair_norm.flags.c_contiguous):
                    raise ValueError("out.pair_norm must be C-contiguous")
            l_full = out.l_full
        else:
            # Active-space DF vectors in ordered-pair layout: L_full[pq, L].
            l_full = cp.empty((nops, naux), dtype=c_cas.dtype)

        # For complex orbitals, only the left contraction uses conjugation:
        #   d[L,pq] = sum_ij conj(C[i,p]) * Lij[L,i,j] * C[j,q]
        c_right = c_cas
        c_left = c_cas.conj() if c_cas.dtype == cp.complex128 else c_cas

        intopt = getattr(dfobj, "intopt", None)
        if method == "auto" and intopt is not None and hasattr(intopt, "cderi_row") and hasattr(intopt, "cderi_col"):
            npairs = int(len(intopt.cderi_row))
            # Heuristic: the pair-space GEMM approach scales ~O(npairs*norb^2) per aux,
            # while the dense-Lij approach scales ~O(nao^2*norb). Use the pair method
            # only when the AO pair list is sufficiently sparse.
            full_s2 = nao * (nao + 1) // 2
            if full_s2 > 0 and npairs < full_s2 // max(1, norb):
                method = "pair_gemm"
            else:
                method = "dense"
        elif method == "auto":
            method = "dense"

        handle = None
        old_math_mode = None
        new_math_mode = None
        if cublas_math_mode is not None:
            mode = str(cublas_math_mode).lower()
            handle = int(cp.cuda.get_cublas_handle())
            if mode == "default":
                new_math_mode = int(cublas_lib.CUBLAS_DEFAULT_MATH)
            elif mode == "fp64_emulated_fixedpoint":
                # This math mode exists only in cuBLAS 13.x (CUDA 13.0+).
                ver = int(cublas_lib.getVersion(handle))
                if ver < 130000:
                    raise RuntimeError(
                        "cublas_math_mode='fp64_emulated_fixedpoint' requires cuBLAS 13.x (CUDA 13.0+); "
                        f"detected cublas version={ver}"
                    )
                new_math_mode = 8  # cublasMath_t: CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (CUDA 13.0+)
            else:
                raise ValueError("cublas_math_mode must be one of: None, 'default', 'fp64_emulated_fixedpoint'")

            old_math_mode = int(cublas_lib.getMathMode(handle))
            if new_math_mode != old_math_mode:
                cublas_lib.setMathMode(handle, int(new_math_mode))

        try:
            if method == "pair_gemm":
                if intopt is None or not hasattr(intopt, "cderi_row") or not hasattr(intopt, "cderi_col"):
                    raise RuntimeError(
                        "gpu4pyscf DF object does not expose cderi_row/cderi_col; cannot use method='pair_gemm'"
                    )

                rows = intopt.cderi_row
                cols = intopt.cderi_col
                if len(rows) != len(cols):
                    raise RuntimeError("intopt.cderi_row and intopt.cderi_col have inconsistent lengths")
                npairs = int(len(rows))

                # Precompute K[pair, pq] so that for each aux tile:
                #   Lpq_flat[L, pq] = sum_pair buf[L,pair] * K[pair,pq]
                # where buf stores the AO CDERI value for the AO pair (i,j) with symmetry.
                diag_scale = cp.asarray((rows == cols).astype(np.float64) * 0.5 + (rows != cols).astype(np.float64))
                diag_scale = diag_scale.astype(c_cas.dtype, copy=False)

                c_left_rows = c_left[rows]  # (npairs, norb)
                c_right_cols = c_right[cols]  # (npairs, norb)
                c_left_cols = c_left[cols]
                c_right_rows = c_right[rows]

                pair_pq_coeff = (c_left_rows[:, :, None] * c_right_cols[:, None, :]) + (
                    c_left_cols[:, :, None] * c_right_rows[:, None, :]
                )
                pair_pq_coeff *= diag_scale[:, None, None]
                pair_pq_coeff = pair_pq_coeff.reshape(npairs, nops)

                # Pull raw CDERI storage from gpu4pyscf to avoid allocating dense (nL,nao,nao)
                # buffers inside `dfobj.loop` when we only need pair-space values.
                device_id = int(cp.cuda.Device().id)
                cderi_sparse = getattr(dfobj, "_cderi", {}).get(device_id)
                if cderi_sparse is None:
                    raise RuntimeError("gpu4pyscf DF object does not have device-local _cderi data")

                if int(cderi_sparse.shape[1]) != npairs:
                    raise RuntimeError(f"unexpected cderi_sparse shape {cderi_sparse.shape} for npairs={npairs}")
                naux_slice = int(cderi_sparse.shape[0])
                if naux_slice != naux:
                    # Multi-GPU DF distributes auxiliary slices across devices; gather is not implemented here.
                    raise RuntimeError(
                        f"method='pair_gemm' currently expects all aux on the current device (naux_slice={naux_slice}, naux={naux})"
                    )

                blksize = self.df_blksize
                if blksize is None:
                    # A conservative default: reuse gpu4pyscf's blksize heuristic (even though
                    # we don't allocate dense (nao,nao) buffers in this mode).
                    blksize = int(getattr(dfobj, "get_blksize")() if hasattr(dfobj, "get_blksize") else 256)

                for p0 in range(0, naux, blksize):
                    p1 = min(naux, p0 + blksize)
                    buf = cderi_sparse[p0:p1]
                    if not isinstance(buf, cp.ndarray):
                        buf = cp.asarray(buf)
                    # buf: (nL, npairs)
                    l_pq_flat = buf @ pair_pq_coeff  # (nL, nops)
                    l_full[:, p0:p1] = l_pq_flat.T
            else:
                # Stream AO DF blocks over auxiliary index.
                col0 = 0
                for block in dfobj.loop(blksize=self.df_blksize, unpack=True):
                    if isinstance(block, tuple):
                        lij = block[0]
                    else:  # pragma: no cover (gpu4pyscf API change)
                        lij = block

                    if lij.ndim != 3:
                        raise RuntimeError(f"unexpected Lij block ndim={lij.ndim}")
                    nL = int(lij.shape[0])
                    if int(lij.shape[1]) != nao or int(lij.shape[2]) != nao:
                        raise RuntimeError(f"unexpected Lij block shape {lij.shape} for nao={nao}")

                    # First contraction: tmp[L,i,q] = sum_j Lij[L,i,j] * C[j,q]
                    tmp = lij @ c_right  # (nL, nao, norb)

                    # Second contraction: Lpq[L,p,q] = sum_i conj(C[i,p]) * tmp[L,i,q].
                    # Use a batched GEMM (via matmul broadcasting) to avoid materializing a large
                    # contiguous (nL*norb, nao) temporary from a transpose+reshape.
                    l_pq = c_left.T @ tmp  # (nL, norb, norb)

                    # Pack into ordered pairs and write into the appropriate auxiliary slice.
                    l_tile = l_pq.reshape(nL, nops).T  # (nops, nL)
                    col1 = col0 + nL
                    l_full[:, col0:col1] = l_tile
                    col0 = col1

                if col0 != naux:
                    raise RuntimeError(f"DF loop produced {col0} aux columns, expected {naux}")

            eri_mat = None
            if want_eri_mat:
                # Dense pair-space ERI matrix: (pq|rs) = sum_L d[L,pq] d[L,rs]
                if out is not None:
                    eri_mat = out.eri_mat
                    cp.dot(l_full, l_full.T, out=eri_mat)
                else:
                    eri_mat = l_full @ l_full.T
                    eri_mat = cp.ascontiguousarray(eri_mat)

            j_ps = None
            if want_j_ps:
                if eri_mat is not None:
                    eri4 = eri_mat.reshape(norb, norb, norb, norb)
                    j_ps = eri4.diagonal(axis1=1, axis2=2).sum(axis=2)
                else:
                    l3 = l_full.reshape(norb, norb, naux)
                    j_ps = cp.einsum("pql,qsl->ps", l3, l3, optimize=True)
                if out is not None:
                    out.j_ps[...] = j_ps
                    j_ps = out.j_ps
                else:
                    j_ps = cp.ascontiguousarray(j_ps)

            pair_norm = None
            if want_pair_norm:
                pair_norm = cp.linalg.norm(l_full, axis=1)
                if out is not None:
                    out.pair_norm[...] = pair_norm
                    pair_norm = out.pair_norm
                else:
                    pair_norm = cp.ascontiguousarray(pair_norm)

            if out is not None:
                return out

            return DeviceActiveDF(
                norb=norb,
                naux=naux,
                l_full=cp.ascontiguousarray(l_full),
                j_ps=j_ps,
                pair_norm=pair_norm,
                eri_mat=eri_mat,
            )
        finally:
            if new_math_mode is not None and old_math_mode is not None and new_math_mode != old_math_mode:
                assert handle is not None
                cublas_lib.setMathMode(int(handle), int(old_math_mode))

    def allocate(
        self,
        norb: int,
        *,
        dtype: Any = np.float64,
        want_eri_mat: bool = True,
        want_j_ps: bool = True,
        want_pair_norm: bool = True,
    ) -> DeviceActiveDF:
        """Allocate reusable GPU output buffers for `build(..., out=...)`.

        Notes
        -----
        - `pair_norm` is always allocated as float64 (even for complex `dtype`).
        - The returned arrays are uninitialized; `build()` overwrites them.
        """

        import cupy as cp

        norb = int(norb)
        if norb <= 0:
            raise ValueError("norb must be > 0")

        dfobj = self.dfobj
        naux = int(getattr(dfobj, "naux", 0) or dfobj.auxmol.nao_nr())
        nops = norb * norb

        dtype = cp.dtype(dtype)
        if dtype not in (cp.float64, cp.complex128):
            raise ValueError(f"allocate(dtype=...) must be float64 or complex128, got {dtype}")

        l_full = cp.empty((nops, naux), dtype=dtype)
        eri_mat = cp.empty((nops, nops), dtype=dtype) if bool(want_eri_mat) else None
        j_ps = cp.empty((norb, norb), dtype=dtype) if bool(want_j_ps) else None
        pair_norm = cp.empty((nops,), dtype=cp.float64) if bool(want_pair_norm) else None

        return DeviceActiveDF(norb=norb, naux=naux, l_full=l_full, j_ps=j_ps, pair_norm=pair_norm, eri_mat=eri_mat)
