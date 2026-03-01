from __future__ import annotations

"""Cartesian-to-spherical AO transformation utilities.

This module builds the block-diagonal matrix ``T (nao_cart, nao_sph)`` that
transforms AO quantities from Cartesian Gaussians to real spherical harmonics.

Design principle: all integrals are computed in Cartesian on GPU, then
transformed at the front-end boundary.  For gradients, density matrices are
back-transformed to Cartesian so existing derivative code works unchanged.
"""

import numpy as np

from asuka.cueri.cart import ncart
from asuka.cueri.sph import cart2sph_matrix, nsph


def compute_sph_layout_from_cart_basis(
    basis,
) -> tuple[np.ndarray, int]:
    """Derive spherical AO offsets from a packed Cartesian basis (no PySCF mol needed).

    Parameters
    ----------
    basis : BasisCartSoA
        Packed Cartesian basis.

    Returns
    -------
    shell_ao_start_sph : ndarray[int32]
        Spherical AO start offset per shell.
    nao_sph : int
        Total number of spherical AOs.
    """
    shell_l = np.asarray(basis.shell_l, dtype=np.int32).ravel()
    nshell = int(shell_l.size)
    if nshell == 0:
        return np.zeros((0,), dtype=np.int32), 0

    shell_ao_start_sph = np.empty((nshell,), dtype=np.int32)
    cursor = 0
    for i in range(nshell):
        shell_ao_start_sph[i] = cursor
        cursor += nsph(int(shell_l[i]))
    return shell_ao_start_sph, int(cursor)


def build_cart2sph_matrix(
    shell_l: np.ndarray,
    shell_ao_start_cart: np.ndarray,
    shell_ao_start_sph: np.ndarray,
    nao_cart: int,
    nao_sph: int,
) -> np.ndarray:
    """Build the full block-diagonal (nao_cart, nao_sph) transformation matrix.

    Each block is ``cart2sph_matrix(l)`` of shape ``(ncart(l), nsph(l))``.
    """
    nao_cart = int(nao_cart)
    nao_sph = int(nao_sph)
    T = np.zeros((nao_cart, nao_sph), dtype=np.float64)

    shell_l = np.asarray(shell_l, dtype=np.int32).ravel()
    starts_c = np.asarray(shell_ao_start_cart, dtype=np.int32).ravel()
    starts_s = np.asarray(shell_ao_start_sph, dtype=np.int32).ravel()

    for i in range(int(shell_l.size)):
        l = int(shell_l[i])
        nc = ncart(l)
        ns = nsph(l)
        ac = int(starts_c[i])
        as_ = int(starts_s[i])
        T[ac : ac + nc, as_ : as_ + ns] = cart2sph_matrix(l)

    return T


def transform_1e_cart_to_sph(M_cart: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Transform a 1e matrix: ``M_sph = T^T @ M_cart @ T``."""
    return T.T @ M_cart @ T


def transform_df_B_cart_to_sph(
    B_cart,
    T: np.ndarray,
    q_block: int = 64,
    *,
    shell_l: np.ndarray | None = None,
    shell_ao_start_cart: np.ndarray | None = None,
    shell_ao_start_sph: np.ndarray | None = None,
    out_layout: str = "mnQ",
):
    """Transform DF factors ``B(nao_cart, nao_cart, nQ)`` to spherical.

    ``B_sph[i,j,Q] = sum_{mu,nu} T[mu,i] * B_cart[mu,nu,Q] * T[nu,j]``

    Works with numpy or cupy arrays.  Batches over Q for memory efficiency.
    """
    xp = _get_array_module(B_cart)
    layout_s = str(out_layout).strip().lower()
    if layout_s not in {"mnq", "qmn"}:
        raise ValueError("out_layout must be one of: 'mnQ', 'Qmn'")

    if B_cart.ndim == 2:
        # B is (nao_cart*(nao_cart+1)//2, nQ) or (nao_cart**2, nQ) — assume packed triangular or flat
        # Actually the common format is (nao_cart, nao_cart, nQ) stored as 3D
        raise ValueError("B_cart must be 3D (nao_cart, nao_cart, nQ)")

    nao_c, nao_c2, nQ = B_cart.shape
    if int(nao_c) != int(nao_c2):
        raise ValueError("B_cart must have shape (nao_cart, nao_cart, nQ)")
    nao_s = int(T.shape[1])

    # Fast path: custom CUDA kernel (avoids large einsum temporaries).
    if xp is not np and shell_l is not None and shell_ao_start_cart is not None and shell_ao_start_sph is not None:
        try:
            import cupy as cp  # noqa: PLC0415

            from asuka.cueri import _cueri_cuda_ext as _ext  # noqa: PLC0415
        except Exception:
            _ext = None
        has_mnq = _ext is not None and hasattr(_ext, "df_B_cart_to_sph_sym_device")
        has_qmn = _ext is not None and hasattr(_ext, "df_B_cart_to_sph_qmn_sym_device")
        if (layout_s == "mnq" and has_mnq) or (layout_s == "qmn" and has_qmn):
            B_dev = cp.asarray(B_cart, dtype=cp.float64)
            if not bool(getattr(B_dev, "flags", None).c_contiguous):
                B_dev = cp.ascontiguousarray(B_dev, dtype=cp.float64)

            shell_l_np = np.asarray(shell_l, dtype=np.int32).ravel()
            sh0_cart_np = np.asarray(shell_ao_start_cart, dtype=np.int32).ravel()
            sh0_sph_np = np.asarray(shell_ao_start_sph, dtype=np.int32).ravel()
            nshell = int(shell_l_np.size)
            if nshell <= 0:
                return cp.empty((0, 0, int(nQ)), dtype=cp.float64)
            if int(sh0_cart_np.size) != nshell or int(sh0_sph_np.size) != nshell:
                raise ValueError("shell_l/shell_ao_start_cart/shell_ao_start_sph must have identical length")

            # Build unique shell pairs (A>=B) and group by (la,lb).
            sp_A = np.empty((nshell * (nshell + 1) // 2,), dtype=np.int32)
            sp_B = np.empty_like(sp_A)
            spAB_by_lab: dict[tuple[int, int], list[int]] = {}
            sp = 0
            for shA in range(nshell):
                la = int(shell_l_np[shA])
                for shB in range(shA + 1):
                    sp_A[sp] = np.int32(shA)
                    sp_B[sp] = np.int32(shB)
                    lb = int(shell_l_np[shB])
                    spAB_by_lab.setdefault((la, lb), []).append(int(sp))
                    sp += 1

            sp_A_dev = cp.ascontiguousarray(cp.asarray(sp_A, dtype=cp.int32))
            sp_B_dev = cp.ascontiguousarray(cp.asarray(sp_B, dtype=cp.int32))
            sh0_cart_dev = cp.ascontiguousarray(cp.asarray(sh0_cart_np, dtype=cp.int32))
            sh0_sph_dev = cp.ascontiguousarray(cp.asarray(sh0_sph_np, dtype=cp.int32))

            if layout_s == "mnq":
                out = cp.empty((int(nao_s), int(nao_s), int(nQ)), dtype=cp.float64)
            else:
                out = cp.empty((int(nQ), int(nao_s), int(nao_s)), dtype=cp.float64)
            threads = 256
            stream_ptr = int(cp.cuda.get_current_stream().ptr)
            in_flat = B_dev.reshape(-1)
            out_flat = out.reshape(-1)
            for (la, lb), spAB_list in spAB_by_lab.items():
                spAB_arr_dev = cp.ascontiguousarray(cp.asarray(np.asarray(spAB_list, dtype=np.int32), dtype=cp.int32))
                if layout_s == "mnq":
                    _ext.df_B_cart_to_sph_sym_device(
                        spAB_arr_dev,
                        sp_A_dev,
                        sp_B_dev,
                        sh0_cart_dev,
                        sh0_sph_dev,
                        int(nao_c),
                        int(nao_s),
                        int(nQ),
                        int(la),
                        int(lb),
                        in_flat,
                        out_flat,
                        int(threads),
                        int(stream_ptr),
                        False,
                    )
                else:
                    _ext.df_B_cart_to_sph_qmn_sym_device(
                        spAB_arr_dev,
                        sp_A_dev,
                        sp_B_dev,
                        sh0_cart_dev,
                        sh0_sph_dev,
                        int(nao_c),
                        int(nao_s),
                        int(nQ),
                        int(la),
                        int(lb),
                        in_flat,
                        out_flat,
                        int(threads),
                        int(stream_ptr),
                        False,
                    )
            return out

    # Fallback: two-pass einsum with optional Q batching.
    T_dev = xp.asarray(T, dtype=xp.float64)
    B_sph = xp.empty((nao_s, nao_s, nQ), dtype=xp.float64)
    q_block = max(1, int(q_block))
    for q0 in range(0, nQ, q_block):
        q1 = min(q0 + q_block, nQ)
        chunk = B_cart[:, :, q0:q1]  # (nao_c, nao_c, qb)
        # tmp = T^T @ chunk  ->  (nao_s, nao_c, qb)
        tmp = xp.einsum("mc,cnq->mnq", T_dev.T, chunk)
        # B_sph[:,:,q0:q1] = tmp @ T  per Q  ->  einsum
        B_sph[:, :, q0:q1] = xp.einsum("mnq,nj->mjq", tmp, T_dev)

    if layout_s == "mnq":
        return B_sph
    return xp.ascontiguousarray(B_sph.transpose((2, 0, 1)), dtype=xp.float64)


def transform_density_sph_to_cart(D_sph, T: np.ndarray):
    """Back-transform density: ``D_cart = T @ D_sph @ T^T``."""
    xp = _get_array_module(D_sph)
    T_dev = xp.asarray(T, dtype=xp.float64)
    return T_dev @ D_sph @ T_dev.T


def transform_3idx_sph_to_cart(L_sph, T: np.ndarray, q_block: int = 64):
    """Back-transform 3-index tensor: ``L_cart[mu,nu,Q] = T @ L_sph @ T^T`` per Q.

    Works with numpy or cupy arrays.
    """
    xp = _get_array_module(L_sph)
    T_dev = xp.asarray(T, dtype=xp.float64)

    nao_s, nao_s2, nQ = L_sph.shape
    nao_c = int(T.shape[0])
    L_cart = xp.empty((nao_c, nao_c, nQ), dtype=xp.float64)

    q_block = max(1, int(q_block))
    for q0 in range(0, nQ, q_block):
        q1 = min(q0 + q_block, nQ)
        chunk = L_sph[:, :, q0:q1]
        tmp = xp.einsum("mc,cnq->mnq", T_dev, chunk)
        L_cart[:, :, q0:q1] = xp.einsum("mnq,jn->mjq", tmp, T_dev)

    return L_cart


def _get_array_module(a):
    """Return numpy or cupy depending on array type."""
    try:
        import cupy as cp  # noqa: PLC0415
    except Exception:
        return np
    if isinstance(a, cp.ndarray):
        return cp
    return np


__all__ = [
    "build_cart2sph_matrix",
    "compute_sph_layout_from_cart_basis",
    "transform_1e_cart_to_sph",
    "transform_3idx_sph_to_cart",
    "transform_density_sph_to_cart",
    "transform_df_B_cart_to_sph",
]
