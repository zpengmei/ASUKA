from __future__ import annotations

import numpy as np

from asuka.mrpt2.df_pair_block import DFPairBlock


def _df_pair_block_to_device(block: DFPairBlock, cp, *, dtype=None) -> DFPairBlock:
    """Return a DFPairBlock with `l_full` moved to the current CuPy device.

    Notes
    -----
    - This is primarily used by the CUDA total driver to avoid repeated H2D
      transfers across the 8 subspace kernels.
    - `pair_norm` is intentionally dropped (set to None) because current CUDA
      kernels do not consume it and keeping it would double device memory.
    """

    dtype = cp.float64 if dtype is None else dtype
    l_full = cp.asarray(block.l_full)
    if l_full.dtype != dtype:
        l_full = l_full.astype(dtype, copy=False)
    l_full = cp.ascontiguousarray(l_full)
    return DFPairBlock(nx=int(block.nx), ny=int(block.ny), l_full=l_full, pair_norm=None)


def sijrs0_energy_df_cuda(
    l_cv: DFPairBlock,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    *,
    virt_tile: int = 32,
    core_tile: int = 8,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sijrs(0) (MP2-like core->virt doubles) on GPU with CuPy.

    This is a CUDA-accelerated analogue of:
      `asuka.mrpt2.nevpt2_sc_df_tiled.sijrs0_energy_df_tiled`

    Notes
    -----
    - Input DF vectors `l_cv.l_full` may live on CPU (NumPy) or GPU (CuPy). The
      function uses `cupy.asarray` so it will avoid extra copies when already on
      device.
    - `virt_tile` is retained for API compatibility with older validation
      scripts; this implementation primarily uses `core_tile` to block over the
      second core index and reduce many small GEMM launches.
    - This kernel is intended as the first GPU milestone: no active tensors are
      required and the workload is dominated by DF GEMMs.
    """

    if int(virt_tile) <= 0:
        raise ValueError("virt_tile must be positive")
    if int(core_tile) <= 0:
        raise ValueError("core_tile must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    ncore = int(l_cv.nx)
    nvirt = int(l_cv.ny)
    naux = int(l_cv.naux)

    eps_core_h = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt_h = np.asarray(eps_virt, dtype=np.float64).ravel()
    if int(eps_core_h.size) != ncore or int(eps_virt_h.size) != nvirt:
        raise ValueError("orbital energy shape mismatch")

    # Transfer DF vectors and energies to device (no-op if already CuPy).
    l_full = cp.asarray(l_cv.l_full)
    if l_full.dtype != cp.float64:
        l_full = l_full.astype(cp.float64, copy=False)
    if tuple(l_full.shape) != (ncore * nvirt, naux):
        raise ValueError("l_cv.l_full shape mismatch")

    eps_core_d = cp.asarray(eps_core_h, dtype=cp.float64)
    eps_virt_d = cp.asarray(eps_virt_h, dtype=cp.float64)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    # Reshape (i,a) packed DF vectors to (i,a,P) for blocked contractions over j.
    l_3 = l_full.reshape(ncore, nvirt, naux)

    for i in range(ncore):
        b_i = l_3[i]  # (nvirt, naux)
        eps_i = eps_core_d[i]

        for j0 in range(0, ncore, int(core_tile)):
            j1 = min(ncore, j0 + int(core_tile))
            nj = int(j1 - j0)

            # Stack (j,b) pairs as rows.
            b_j = l_3[j0:j1].reshape(nj * nvirt, naux)  # (nj*nvirt, naux)

            # v[a,(j,b)] = Σ_P d[i,a,P] d[j,b,P] ≈ (i a| j b)
            v = b_i @ b_j.T  # (nvirt, nj*nvirt)
            v3 = v.reshape(nvirt, nj, nvirt)  # (a,j,b)

            theta = 2.0 * v3 - v3.transpose(2, 1, 0)  # (a,j,b)
            eps_ij = eps_i + eps_core_d[j0:j1]  # (nj,)
            denom = eps_ij[None, :, None] - eps_virt_d[:, None, None] - eps_virt_d[None, None, :]  # (a,j,b)

            norm_sum = norm_sum + cp.sum(v3 * theta)
            e2 = e2 + cp.sum((v3 / denom) * theta)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def srs_m2_energy_df_cuda(
    l_va: DFPairBlock,
    *,
    m_norm: np.ndarray,
    m_h: np.ndarray,
    eps_virt: np.ndarray,
    r_tile: int = 8,
    s_tile: int = 32,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Srs(-2) on GPU from DF pair blocks and precomputed active matrices.

    This is a CUDA-accelerated analogue of:
      `asuka.mrpt2.nevpt2_sc_df_tiled.srs_m2_energy_df_tiled`

    Parameters
    ----------
    l_va:
        DF pair block for ordered (virt,act) pairs (r,a) with packing ra_id=r*nact+a.
    m_norm, m_h:
        Active-space matrices of shape (nact^2, nact^2) defining:
          norm_rs = 0.5 * x_rs^T m_norm x_rs
          h_rs    = 0.5 * x_rs^T m_h    x_rs
        where x_rs[pq] = (r p| s q) in ordered-pair packing pq_id=p*nact+q.
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if int(s_tile) <= 0:
        raise ValueError("s_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nvirt = int(l_va.nx)
    nact = int(l_va.ny)
    naux = int(l_va.naux)
    n2 = nact * nact

    eps_virt_h = np.asarray(eps_virt, dtype=np.float64).ravel()
    if int(eps_virt_h.size) != nvirt:
        raise ValueError("eps_virt shape mismatch")

    m_norm_h = np.asarray(m_norm, dtype=np.float64)
    m_h_h = np.asarray(m_h, dtype=np.float64)
    if m_norm_h.shape != (n2, n2) or m_h_h.shape != (n2, n2):
        raise ValueError("m_norm/m_h must have shape (nact^2, nact^2)")

    l_full = cp.asarray(l_va.l_full)
    if l_full.dtype != cp.float64:
        l_full = l_full.astype(cp.float64, copy=False)
    if tuple(l_full.shape) != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")

    eps_virt_d = cp.asarray(eps_virt_h, dtype=cp.float64)
    m_norm_d = cp.asarray(m_norm_h, dtype=cp.float64)
    m_h_d = cp.asarray(m_h_h, dtype=cp.float64)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = int(r1 - r0)

        l_rp = l_full[r0 * nact : r1 * nact]  # (nr*nact, naux) for (r,p)
        eps_r = eps_virt_d[r0:r1]  # (nr,)

        for s0 in range(0, nvirt, int(s_tile)):
            s1 = min(nvirt, s0 + int(s_tile))
            ns = int(s1 - s0)

            l_sq = l_full[s0 * nact : s1 * nact]  # (ns*nact, naux) for (s,q)
            eps_s = eps_virt_d[s0:s1]  # (ns,)

            # g[(r,p),(s,q)] = (r p| s q)
            g = l_rp @ l_sq.T  # (nr*nact, ns*nact)

            # Pack each (r,s) block into x_rs[pq] with pq_id = p*nact + q.
            x = g.reshape(nr, nact, ns, nact).transpose(0, 2, 1, 3).reshape(nr * ns, n2)

            xn = x @ m_norm_d
            norm = 0.5 * cp.sum(x * xn, axis=1)

            xh = x @ m_h_d
            h = 0.5 * cp.sum(x * xh, axis=1)

            diff = (eps_r[:, None] + eps_s[None, :]).reshape(nr * ns)

            mask = cp.abs(norm) > float(numerical_zero)
            safe_norm = cp.where(mask, norm, 1.0)
            contrib = cp.where(mask, norm / (diff + h / safe_norm), 0.0)

            norm_sum = norm_sum + cp.sum(norm)
            e2 = e2 - cp.sum(contrib)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def sij_p2_energy_df_cuda(
    l_ac: DFPairBlock,
    *,
    m_norm: np.ndarray,
    m_h: np.ndarray,
    eps_core: np.ndarray,
    i_tile: int = 4,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sij(+2) on GPU from DF pair blocks and precomputed active matrices.

    This is a CUDA-accelerated analogue of:
      `asuka.mrpt2.nevpt2_sc_df_tiled.sij_p2_energy_df_tiled`
    """

    if int(i_tile) <= 0:
        raise ValueError("i_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nact = int(l_ac.nx)
    ncore = int(l_ac.ny)
    naux = int(l_ac.naux)
    n2 = nact * nact

    eps_core_h = np.asarray(eps_core, dtype=np.float64).ravel()
    if int(eps_core_h.size) != ncore:
        raise ValueError("eps_core shape mismatch")

    m_norm_h = np.asarray(m_norm, dtype=np.float64)
    m_h_h = np.asarray(m_h, dtype=np.float64)
    if m_norm_h.shape != (n2, n2) or m_h_h.shape != (n2, n2):
        raise ValueError("m_norm/m_h must have shape (nact^2, nact^2)")

    l_ac_full = cp.asarray(l_ac.l_full)
    if l_ac_full.dtype != cp.float64:
        l_ac_full = l_ac_full.astype(cp.float64, copy=False)
    if tuple(l_ac_full.shape) != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")

    # Repack (p,i) to (i,p) row order as in the CPU tiled kernel.
    l_ip = l_ac_full.reshape(nact, ncore, naux).transpose(1, 0, 2).reshape(ncore * nact, naux)
    l_ip_t = l_ip.T

    eps_core_d = cp.asarray(eps_core_h, dtype=cp.float64)
    m_norm_d = cp.asarray(m_norm_h, dtype=cp.float64)
    m_h_d = cp.asarray(m_h_h, dtype=cp.float64)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for i0 in range(0, ncore, int(i_tile)):
        i1 = min(ncore, i0 + int(i_tile))
        ni = int(i1 - i0)
        b = int(ni * ncore)

        l_left = l_ip[i0 * nact : i1 * nact]  # (ni*nact, naux)
        g = l_left @ l_ip_t  # (ni*nact, ncore*nact), cols packed as (j,q)
        g4 = g.reshape(ni, nact, ncore, nact).transpose(0, 2, 1, 3)  # (i,j,p,q)
        v = g4.reshape(b, n2)

        tN = v @ m_norm_d
        norm = 0.5 * cp.sum(v * tN, axis=1)

        tH = v @ m_h_d
        h = 0.5 * cp.sum(v * tH, axis=1)

        diff = (-(eps_core_d[i0:i1, None] + eps_core_d[None, :])).reshape(b)

        mask = cp.abs(norm) > float(numerical_zero)
        safe_norm = cp.where(mask, norm, 1.0)
        contrib = cp.where(mask, norm / (diff + h / safe_norm), 0.0)

        norm_sum = norm_sum + cp.sum(norm)
        e2 = e2 - cp.sum(contrib)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def sr_m1_prime_energy_df_cuda(
    l_va: DFPairBlock,
    l_aa: DFPairBlock,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    a16: np.ndarray,
    a17: np.ndarray,
    a19: np.ndarray,
    eps_virt: np.ndarray,
    r_tile: int = 8,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sr(-1)' on GPU from DF pair blocks and precomputed active tensors.

    This is a CUDA-accelerated analogue of:
      `asuka.mrpt2.nevpt2_sc_df_tiled.sr_m1_prime_energy_df_tiled`

    Notes
    -----
    This implementation is **reference-oriented** and assumes the full
    `dm3[a,b,c,d,e,f]` / `a16[a,b,c,d,e,f]` tensors are available.
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_va/l_aa active dimensions mismatch")
    if int(l_va.naux) != int(l_aa.naux):
        raise ValueError("l_va and l_aa naux mismatch")
    naux = int(l_va.naux)

    h1e_v_h = np.asarray(h1e_v, dtype=np.float64)
    if h1e_v_h.shape != (nvirt, nact):
        raise ValueError("h1e_v shape mismatch (expect virt x act)")

    eps_virt_h = np.asarray(eps_virt, dtype=np.float64).ravel()
    if int(eps_virt_h.size) != nvirt:
        raise ValueError("eps_virt shape mismatch")

    dm1_h = np.asarray(dm1, dtype=np.float64)
    dm2_h = np.asarray(dm2, dtype=np.float64)
    dm3_h = np.asarray(dm3, dtype=np.float64)
    a16_h = np.asarray(a16, dtype=np.float64)
    a17_h = np.asarray(a17, dtype=np.float64)
    a19_h = np.asarray(a19, dtype=np.float64)

    if dm1_h.shape != (nact, nact) or dm2_h.shape != (nact, nact, nact, nact) or dm3_h.shape != (
        nact,
        nact,
        nact,
        nact,
        nact,
        nact,
    ):
        raise ValueError("dm shape mismatch")
    if a16_h.shape != dm3_h.shape:
        raise ValueError("a16 shape mismatch")
    if a17_h.shape != (nact, nact, nact, nact):
        raise ValueError("a17 shape mismatch")
    if a19_h.shape != (nact, nact):
        raise ValueError("a19 shape mismatch")

    l_va_full = cp.asarray(l_va.l_full)
    l_aa_full = cp.asarray(l_aa.l_full)
    if l_va_full.dtype != cp.float64:
        l_va_full = l_va_full.astype(cp.float64, copy=False)
    if l_aa_full.dtype != cp.float64:
        l_aa_full = l_aa_full.astype(cp.float64, copy=False)
    if tuple(l_va_full.shape) != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")
    if tuple(l_aa_full.shape) != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")
    l_aa_t = l_aa_full.T

    h1e_v_d = cp.asarray(h1e_v_h, dtype=cp.float64)
    eps_virt_d = cp.asarray(eps_virt_h, dtype=cp.float64)

    dm1_d = cp.asarray(dm1_h, dtype=cp.float64)
    dm2_d = cp.asarray(dm2_h, dtype=cp.float64)
    dm3_d = cp.asarray(dm3_h, dtype=cp.float64)
    a16_d = cp.asarray(a16_h, dtype=cp.float64)
    a17_d = cp.asarray(a17_h, dtype=cp.float64)
    a19_d = cp.asarray(a19_h, dtype=cp.float64)

    # One-electron-only terms.
    norm_1e = cp.einsum("ip,pa,ia->i", h1e_v_d, dm1_d, h1e_v_d, optimize=True)
    h_1e = cp.einsum("ip,pa,ia->i", h1e_v_d, a19_d, h1e_v_d, optimize=True)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = int(r1 - r0)

        l_rq = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
        v = l_rq @ l_aa_t  # (nr*nact, nact*nact)
        v4 = v.reshape(nr, nact, nact, nact)  # (r,q,p,s)
        h2e_v = cp.ascontiguousarray(v4.transpose(0, 2, 1, 3))  # (r,p,q,s)

        h = (
            cp.einsum("ipqr,pqrabc,iabc->i", h2e_v, a16_d, h2e_v, optimize=True)
            + 2.0 * cp.einsum("ipqr,pqra,ia->i", h2e_v, a17_d, h1e_v_d[r0:r1], optimize=True)
            + h_1e[r0:r1]
        )

        norm = (
            cp.einsum("ipqr,rpqbac,iabc->i", h2e_v, dm3_d, h2e_v, optimize=True)
            + 2.0 * cp.einsum("ipqr,rpqa,ia->i", h2e_v, dm2_d, h1e_v_d[r0:r1], optimize=True)
            + norm_1e[r0:r1]
        )

        norm_sum = norm_sum + cp.sum(norm)

        diff = eps_virt_d[r0:r1]
        mask = cp.abs(norm) > float(numerical_zero)
        safe_norm = cp.where(mask, norm, 1.0)
        contrib = cp.where(mask, norm / (diff + h / safe_norm), 0.0)
        e2 = e2 - cp.sum(contrib)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def si_p1_prime_energy_df_cuda(
    l_ac: DFPairBlock,
    l_aa: DFPairBlock,
    h1e_v: np.ndarray,
    *,
    dm1_h: np.ndarray,
    dm2_h: np.ndarray,
    dm3_h: np.ndarray,
    a22: np.ndarray,
    a23: np.ndarray,
    a25: np.ndarray,
    eps_core: np.ndarray,
    i_tile: int = 8,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Si(+1)' on GPU from DF pair blocks and precomputed active tensors."""

    if int(i_tile) <= 0:
        raise ValueError("i_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nact, ncore = int(l_ac.nx), int(l_ac.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_ac/l_aa active dimensions mismatch")
    if int(l_ac.naux) != int(l_aa.naux):
        raise ValueError("l_ac and l_aa naux mismatch")
    naux = int(l_ac.naux)

    h1e_v_h = np.asarray(h1e_v, dtype=np.float64)
    if h1e_v_h.shape != (nact, ncore):
        raise ValueError("h1e_v shape mismatch (expect act x core)")

    eps_core_h = np.asarray(eps_core, dtype=np.float64).ravel()
    if int(eps_core_h.size) != ncore:
        raise ValueError("eps_core shape mismatch")

    dm1_h_h = np.asarray(dm1_h, dtype=np.float64)
    dm2_h_h = np.asarray(dm2_h, dtype=np.float64)
    dm3_h_h = np.asarray(dm3_h, dtype=np.float64)
    a22_h = np.asarray(a22, dtype=np.float64)
    a23_h = np.asarray(a23, dtype=np.float64)
    a25_h = np.asarray(a25, dtype=np.float64)

    if dm1_h_h.shape != (nact, nact) or dm2_h_h.shape != (nact, nact, nact, nact) or dm3_h_h.shape != (
        nact,
        nact,
        nact,
        nact,
        nact,
        nact,
    ):
        raise ValueError("dm_h shape mismatch")
    if a22_h.shape != dm3_h_h.shape:
        raise ValueError("a22 shape mismatch")
    if a23_h.shape != (nact, nact, nact, nact):
        raise ValueError("a23 shape mismatch")
    if a25_h.shape != (nact, nact):
        raise ValueError("a25 shape mismatch")

    l_ac_full = cp.asarray(l_ac.l_full)
    l_aa_full = cp.asarray(l_aa.l_full)
    if l_ac_full.dtype != cp.float64:
        l_ac_full = l_ac_full.astype(cp.float64, copy=False)
    if l_aa_full.dtype != cp.float64:
        l_aa_full = l_aa_full.astype(cp.float64, copy=False)
    if tuple(l_ac_full.shape) != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")
    if tuple(l_aa_full.shape) != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")
    l_aa_t = l_aa_full.T

    l_ac_3 = l_ac_full.reshape(nact, ncore, naux)

    h1e_v_d = cp.asarray(h1e_v_h, dtype=cp.float64)
    eps_core_d = cp.asarray(eps_core_h, dtype=cp.float64)

    dm1_d = cp.asarray(dm1_h_h, dtype=cp.float64)
    dm2_d = cp.asarray(dm2_h_h, dtype=cp.float64)
    dm3_d = cp.asarray(dm3_h_h, dtype=cp.float64)
    a22_d = cp.asarray(a22_h, dtype=cp.float64)
    a23_d = cp.asarray(a23_h, dtype=cp.float64)
    a25_d = cp.asarray(a25_h, dtype=cp.float64)

    norm_1e = cp.einsum("pi,pa,ai->i", h1e_v_d, dm1_d, h1e_v_d, optimize=True)
    h_1e = cp.einsum("pi,pa,ai->i", h1e_v_d, a25_d, h1e_v_d, optimize=True)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for i0 in range(0, ncore, int(i_tile)):
        i1 = min(ncore, i0 + int(i_tile))
        ni = int(i1 - i0)

        l_qi = cp.ascontiguousarray(l_ac_3[:, i0:i1, :].reshape(nact * ni, naux))
        v = l_qi @ l_aa_t  # (nact*ni, nact*nact)
        v4 = cp.ascontiguousarray(v.reshape(nact, ni, nact, nact).transpose(0, 2, 1, 3))  # (q,p,i,r)

        h = (
            cp.einsum("qpir,pqrabc,baic->i", v4, a22_d, v4, optimize=True)
            + 2.0 * cp.einsum("qpir,pqra,ai->i", v4, a23_d, h1e_v_d[:, i0:i1], optimize=True)
            + h_1e[i0:i1]
        )
        norm = (
            cp.einsum("qpir,rpqbac,baic->i", v4, dm3_d, v4, optimize=True)
            + 2.0 * cp.einsum("qpir,rpqa,ai->i", v4, dm2_d, h1e_v_d[:, i0:i1], optimize=True)
            + norm_1e[i0:i1]
        )

        norm_sum = norm_sum + cp.sum(norm)

        diff = -eps_core_d[i0:i1]
        mask = cp.abs(norm) > float(numerical_zero)
        safe_norm = cp.where(mask, norm, 1.0)
        contrib = cp.where(mask, norm / (diff + h / safe_norm), 0.0)
        e2 = e2 - cp.sum(contrib)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def sir_0_energy_df_cuda(
    l_vc: DFPairBlock,
    l_aa: DFPairBlock,
    l_va: DFPairBlock,
    l_ac: DFPairBlock,
    h1e_v: np.ndarray,
    *,
    dm1: np.ndarray,
    dm2: np.ndarray,
    a12: np.ndarray,
    a13: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    r_tile: int = 4,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sir(0) on GPU from DF pair blocks and precomputed active tensors.

    This is a CUDA-accelerated analogue of:
      `asuka.mrpt2.nevpt2_sc_df_tiled.sir_0_energy_df_tiled`
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1:
        raise ValueError("l_aa must be square (act,act)")
    nact = int(nact0)

    if int(l_va.nx) != nvirt or int(l_va.ny) != nact:
        raise ValueError("l_va shape mismatch (expect virt x act)")
    if int(l_ac.nx) != nact or int(l_ac.ny) != ncore:
        raise ValueError("l_ac shape mismatch (expect act x core)")
    if int(l_vc.naux) != int(l_aa.naux) or int(l_vc.naux) != int(l_va.naux) or int(l_vc.naux) != int(l_ac.naux):
        raise ValueError("DF naux mismatch")
    naux = int(l_vc.naux)

    h1e_v_h = np.asarray(h1e_v, dtype=np.float64)
    if h1e_v_h.shape != (nvirt, ncore):
        raise ValueError("h1e_v shape mismatch (expect virt x core)")

    dm1_h = np.asarray(dm1, dtype=np.float64)
    dm2_h = np.asarray(dm2, dtype=np.float64)
    a12_h = np.asarray(a12, dtype=np.float64)
    a13_h = np.asarray(a13, dtype=np.float64)
    if dm1_h.shape != (nact, nact):
        raise ValueError("dm1 shape mismatch")
    if dm2_h.shape != (nact, nact, nact, nact):
        raise ValueError("dm2 shape mismatch")
    if a12_h.shape != (nact, nact, nact, nact) or a13_h.shape != (nact, nact, nact, nact):
        raise ValueError("a12/a13 shape mismatch")

    eps_core_h = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt_h = np.asarray(eps_virt, dtype=np.float64).ravel()
    if int(eps_core_h.size) != ncore or int(eps_virt_h.size) != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_vc_full = cp.asarray(l_vc.l_full)
    l_aa_full = cp.asarray(l_aa.l_full)
    l_va_full = cp.asarray(l_va.l_full)
    l_ac_full = cp.asarray(l_ac.l_full)
    if l_vc_full.dtype != cp.float64:
        l_vc_full = l_vc_full.astype(cp.float64, copy=False)
    if l_aa_full.dtype != cp.float64:
        l_aa_full = l_aa_full.astype(cp.float64, copy=False)
    if l_va_full.dtype != cp.float64:
        l_va_full = l_va_full.astype(cp.float64, copy=False)
    if l_ac_full.dtype != cp.float64:
        l_ac_full = l_ac_full.astype(cp.float64, copy=False)

    if tuple(l_vc_full.shape) != (nvirt * ncore, naux):
        raise ValueError("l_vc.l_full shape mismatch")
    if tuple(l_aa_full.shape) != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")
    if tuple(l_va_full.shape) != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")
    if tuple(l_ac_full.shape) != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")

    l_aa_t = l_aa_full.T
    l_ac_t = l_ac_full.T

    h1e_v_d = cp.asarray(h1e_v_h, dtype=cp.float64)
    eps_core_d = cp.asarray(eps_core_h, dtype=cp.float64)
    eps_virt_d = cp.asarray(eps_virt_h, dtype=cp.float64)

    dm1_d = cp.asarray(dm1_h, dtype=cp.float64)
    dm2_d = cp.asarray(dm2_h, dtype=cp.float64)
    a12_d = cp.asarray(a12_h, dtype=cp.float64)
    a13_d = cp.asarray(a13_h, dtype=cp.float64)

    n2 = int(nact * nact)
    m2 = dm2_d.transpose(1, 0, 2, 3).reshape(n2, n2)
    m2e = dm2_d.transpose(3, 0, 2, 1).reshape(n2, n2)
    m12 = a12_d.reshape(n2, n2)
    m13 = a13_d.reshape(n2, n2)
    dm1_t = dm1_d.T

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = int(r1 - r0)
        b = int(nr * ncore)

        l_ri = l_vc_full[r0 * ncore : r1 * ncore]  # (nr*ncore, naux)
        v1 = l_ri @ l_aa_t  # (b, n2)

        l_rp = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
        g = l_rp @ l_ac_t  # (nr*nact, nact*ncore)
        v2_4 = g.reshape(nr, nact, nact, ncore)  # (r,p,q,i)
        v2 = v2_4.transpose(0, 3, 1, 2).reshape(b, n2)  # (r,i,p,q) -> (b,pq)

        v1_3 = v1.reshape(b, nact, nact)
        v2_3 = v2.reshape(b, nact, nact)

        h1 = h1e_v_d[r0:r1].reshape(b)
        diff = (eps_virt_d[r0:r1, None] - eps_core_d[None, :]).reshape(b)

        t1 = v1 @ m2
        t2 = v2 @ m2
        tE = v2 @ m2e

        termA = 2.0 * cp.sum(v1 * t1, axis=1)
        termB = -cp.sum(t1 * v2, axis=1)
        termC = -cp.sum(t2 * v1, axis=1)
        termE = -cp.sum(v2 * tE, axis=1)

        v2_dm = v2_3 @ dm1_d
        termD = 2.0 * cp.sum(v2_dm * v2_3, axis=(1, 2))

        inner_v1 = cp.sum(v1_3 * dm1_t, axis=(1, 2))
        inner_v2 = cp.sum(v2_3 * dm1_t, axis=(1, 2))
        trace_v2 = cp.einsum("bii->b", v2_3, optimize=True)
        termF = trace_v2 * inner_v2

        termG = 4.0 * h1 * inner_v1
        termH = -2.0 * h1 * inner_v2
        termI = 2.0 * h1 * h1

        norm = termA + termB + termC + termD + termE + termF + termG + termH + termI

        t12_v1 = v1 @ m12
        t12_v2 = v2 @ m12
        t13_v2 = v2 @ m13
        h = (
            2.0 * cp.sum(v1 * t12_v1, axis=1)
            - cp.sum(t12_v1 * v2, axis=1)
            - cp.sum(t12_v2 * v1, axis=1)
            + cp.sum(v2 * t13_v2, axis=1)
        )

        norm_sum = norm_sum + cp.sum(norm)

        mask = cp.abs(norm) > float(numerical_zero)
        safe_norm = cp.where(mask, norm, 1.0)
        contrib = cp.where(mask, norm / (diff + h / safe_norm), 0.0)
        e2 = e2 - cp.sum(contrib)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def srsi_m1_energy_df_cuda(
    l_vc: DFPairBlock,
    l_va: DFPairBlock,
    *,
    dm1: np.ndarray,
    k27: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    virt_tile: int = 64,
    core_tile: int = 32,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Srsi(-1) on GPU from DF pair blocks and precomputed active matrices.

    This is a CUDA-accelerated analogue of:
      `asuka.mrpt2.nevpt2_sc_df_tiled.srsi_m1_energy_df_tiled`
    """

    if int(virt_tile) <= 0:
        raise ValueError("virt_tile must be positive")
    if int(core_tile) <= 0:
        raise ValueError("core_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nvirt2, nact = int(l_va.nx), int(l_va.ny)
    if nvirt2 != nvirt:
        raise ValueError("l_vc and l_va virt dimensions mismatch")
    if int(l_vc.naux) != int(l_va.naux):
        raise ValueError("l_vc and l_va naux mismatch")
    naux = int(l_vc.naux)

    dm1_h = np.asarray(dm1, dtype=np.float64)
    k27_h = np.asarray(k27, dtype=np.float64)
    if dm1_h.shape != (nact, nact) or k27_h.shape != (nact, nact):
        raise ValueError("dm1/k27 shape mismatch")

    eps_core_h = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt_h = np.asarray(eps_virt, dtype=np.float64).ravel()
    if int(eps_core_h.size) != ncore or int(eps_virt_h.size) != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_vc_full = cp.asarray(l_vc.l_full)
    l_va_full = cp.asarray(l_va.l_full)
    if l_vc_full.dtype != cp.float64:
        l_vc_full = l_vc_full.astype(cp.float64, copy=False)
    if l_va_full.dtype != cp.float64:
        l_va_full = l_va_full.astype(cp.float64, copy=False)
    l_vc_full = cp.ascontiguousarray(l_vc_full)
    l_va_full = cp.ascontiguousarray(l_va_full)
    if tuple(l_vc_full.shape) != (nvirt * ncore, naux):
        raise ValueError("l_vc.l_full shape mismatch")
    if tuple(l_va_full.shape) != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")

    l_vc_3 = l_vc_full.reshape(nvirt, ncore, naux)

    dm1_d = cp.asarray(dm1_h, dtype=cp.float64)
    k27_d = cp.asarray(k27_h, dtype=cp.float64)
    eps_core_d = cp.asarray(eps_core_h, dtype=cp.float64)
    eps_virt_d = cp.asarray(eps_virt_h, dtype=cp.float64)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for i0 in range(0, ncore, int(core_tile)):
        i1 = min(ncore, i0 + int(core_tile))
        ni = int(i1 - i0)
        eps_i = eps_core_d[i0:i1]  # (ni,)

        for r0 in range(0, nvirt, int(virt_tile)):
            r1 = min(nvirt, r0 + int(virt_tile))
            nr = int(r1 - r0)

            l_rA = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
            l_ri_T = cp.ascontiguousarray(l_vc_3[r0:r1, i0:i1, :].transpose(2, 0, 1).reshape(naux, nr * ni))

            for s0 in range(r0, nvirt, int(virt_tile)):
                s1 = min(nvirt, s0 + int(virt_tile))
                ns = int(s1 - s0)

                l_sA = l_va_full[s0 * nact : s1 * nact]  # (ns*nact, naux)
                l_si_T = cp.ascontiguousarray(l_vc_3[s0:s1, i0:i1, :].transpose(2, 0, 1).reshape(naux, ns * ni))

                g_rs = l_sA @ l_ri_T  # (ns*nact, nr*ni)
                v_rs = g_rs.reshape(ns, nact, nr, ni).transpose(3, 2, 0, 1).reshape(ni, nr * ns, nact)

                g_sr = l_rA @ l_si_T  # (nr*nact, ns*ni)
                v_sr = g_sr.reshape(nr, nact, ns, ni).transpose(3, 0, 2, 1).reshape(ni, nr * ns, nact)

                v_rs2 = v_rs.reshape(ni * nr * ns, nact)
                v_sr2 = v_sr.reshape(ni * nr * ns, nact)

                v_rs_dm = v_rs2 @ dm1_d
                v_sr_dm = v_sr2 @ dm1_d
                a_rs = cp.sum(v_rs_dm * v_rs2, axis=1)
                a_sr = cp.sum(v_sr_dm * v_sr2, axis=1)
                b_rs = cp.sum(v_rs_dm * v_sr2, axis=1)
                b_sr = cp.sum(v_sr_dm * v_rs2, axis=1)
                norm = (2.0 * (a_rs + a_sr) - (b_rs + b_sr)).reshape(ni, nr * ns)

                v_rs_k = v_rs2 @ k27_d
                v_sr_k = v_sr2 @ k27_d
                h_rs = cp.sum(v_rs_k * v_rs2, axis=1)
                h_sr = cp.sum(v_sr_k * v_sr2, axis=1)
                hk_rs = cp.sum(v_rs_k * v_sr2, axis=1)
                hk_sr = cp.sum(v_sr_k * v_rs2, axis=1)
                h = (2.0 * (h_rs + h_sr) - (hk_rs + hk_sr)).reshape(ni, nr * ns)

                diff_rs = (eps_virt_d[r0:r1, None] + eps_virt_d[s0:s1][None, :]).reshape(nr * ns)
                diff = diff_rs[None, :] - eps_i[:, None]

                if s0 == r0:
                    if nr != ns:
                        raise RuntimeError("internal error: diagonal tile must be square")
                    norm_mat = norm.reshape(ni, nr, nr)
                    h_mat = h.reshape(ni, nr, nr)
                    diff_mat = diff.reshape(ni, nr, nr)

                    idx = cp.arange(nr, dtype=cp.int32)
                    norm_mat[:, idx, idx] *= 0.5
                    h_mat[:, idx, idx] *= 0.5

                    triu = cp.triu_indices(nr)
                    norm_use = norm_mat[:, triu[0], triu[1]]
                    h_use = h_mat[:, triu[0], triu[1]]
                    diff_use = diff_mat[:, triu[0], triu[1]]
                else:
                    norm_use = norm
                    h_use = h
                    diff_use = diff

                mask = cp.abs(norm_use) > float(numerical_zero)
                safe_norm = cp.where(mask, norm_use, 1.0)
                contrib = cp.where(mask, norm_use / (diff_use + h_use / safe_norm), 0.0)

                e2 = e2 - cp.sum(contrib)
                norm_sum = norm_sum + cp.sum(norm_use)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def sijr_p1_energy_df_cuda(
    l_vc: DFPairBlock,
    l_ac: DFPairBlock,
    *,
    hdm1: np.ndarray,
    a3: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    r_tile: int = 4,
    numerical_zero: float = 1e-14,
    device: int | None = None,
    return_device: bool = False,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sijr(+1) on GPU from DF pair blocks and precomputed active matrices."""

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    try:
        import cupy as cp
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nact, ncore2 = int(l_ac.nx), int(l_ac.ny)
    if ncore2 != ncore:
        raise ValueError("l_vc/l_ac core dimension mismatch")
    if int(l_vc.naux) != int(l_ac.naux):
        raise ValueError("l_vc/l_ac naux mismatch")
    naux = int(l_vc.naux)

    hdm1_h = np.asarray(hdm1, dtype=np.float64)
    a3_h = np.asarray(a3, dtype=np.float64)
    if hdm1_h.shape != (nact, nact) or a3_h.shape != (nact, nact):
        raise ValueError("hdm1/a3 shape mismatch")

    eps_core_h = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt_h = np.asarray(eps_virt, dtype=np.float64).ravel()
    if int(eps_core_h.size) != ncore or int(eps_virt_h.size) != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_vc_full = cp.asarray(l_vc.l_full)
    l_ac_full = cp.asarray(l_ac.l_full)
    if l_vc_full.dtype != cp.float64:
        l_vc_full = l_vc_full.astype(cp.float64, copy=False)
    if l_ac_full.dtype != cp.float64:
        l_ac_full = l_ac_full.astype(cp.float64, copy=False)
    if tuple(l_vc_full.shape) != (nvirt * ncore, naux):
        raise ValueError("l_vc.l_full shape mismatch")
    if tuple(l_ac_full.shape) != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")

    l_ac_t = l_ac_full.T

    hdm1_d = cp.asarray(hdm1_h, dtype=cp.float64)
    a3_d = cp.asarray(a3_h, dtype=cp.float64)
    eps_core_d = cp.asarray(eps_core_h, dtype=cp.float64)
    eps_virt_d = cp.asarray(eps_virt_h, dtype=cp.float64)

    triu = cp.triu_indices(ncore)
    idx = cp.arange(ncore, dtype=cp.int32)

    norm_sum = cp.zeros((), dtype=cp.float64)
    e2 = cp.zeros((), dtype=cp.float64)

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = int(r1 - r0)

        l_ri = l_vc_full[r0 * ncore : r1 * ncore]  # (nr*ncore, naux)
        v = l_ri @ l_ac_t  # (nr*ncore, nact*ncore)

        # v[(r,i),(p,j)] -> w[r,j,i,p]
        w = v.reshape(nr, ncore, nact, ncore).transpose(0, 3, 1, 2)  # (r,j,i,p)
        w_flat = w.reshape(nr * ncore * ncore, nact)

        wx_n = w_flat @ hdm1_d
        wx_n4 = wx_n.reshape(nr, ncore, ncore, nact)
        a = cp.sum(wx_n4 * w, axis=3)
        b = cp.sum(wx_n4 * w.transpose(0, 2, 1, 3), axis=3)
        norm = 2.0 * a - b
        norm = norm + norm.transpose(0, 2, 1)
        norm[:, idx, idx] *= 0.5

        wx_h = w_flat @ a3_d
        wx_h4 = wx_h.reshape(nr, ncore, ncore, nact)
        ah = cp.sum(wx_h4 * w, axis=3)
        bh = cp.sum(wx_h4 * w.transpose(0, 2, 1, 3), axis=3)
        h = 2.0 * ah - bh
        h = h + h.transpose(0, 2, 1)
        h[:, idx, idx] *= 0.5

        diff = (
            eps_virt_d[r0:r1, None, None]
            - eps_core_d[None, :, None]
            - eps_core_d[None, None, :]
        )
        diff_use = diff[:, triu[0], triu[1]].reshape(nr * triu[0].size)
        norm_use = norm[:, triu[0], triu[1]].reshape(nr * triu[0].size)
        h_use = h[:, triu[0], triu[1]].reshape(nr * triu[0].size)

        mask = cp.abs(norm_use) > float(numerical_zero)
        safe_norm = cp.where(mask, norm_use, 1.0)
        contrib = cp.where(mask, norm_use / (diff_use + h_use / safe_norm), 0.0)

        e2 = e2 - cp.sum(contrib)
        norm_sum = norm_sum + cp.sum(norm_use)

    if bool(return_device):
        return norm_sum, e2
    return float(cp.asnumpy(norm_sum)), float(cp.asnumpy(e2))


def nevpt2_sc_total_energy_df_cuda(
    *,
    l_cv: DFPairBlock,
    l_vc: DFPairBlock,
    l_va: DFPairBlock,
    l_ac: DFPairBlock,
    l_aa: DFPairBlock,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    h1e_v_sir: np.ndarray,
    h1e_v_sr: np.ndarray,
    h1e_v_si: np.ndarray,
    dm1: np.ndarray,
    dm2: np.ndarray,
    dm3: np.ndarray,
    h1e: np.ndarray,
    h2e: np.ndarray,
    f3ca: np.ndarray,
    f3ac: np.ndarray,
    device: int | None = None,
) -> dict[str, float]:
    """CUDA (CuPy) total SC-NEVPT2(DF) energy by summing the 8 subspace kernels.

    This mirrors `asuka.mrpt2.nevpt2_sc.nevpt2_sc_total_energy_df`, but routes the
    heavy DF contractions through CuPy GPU kernels in this module.

    Notes
    -----
    - This is a reference-oriented driver: active intermediates are built on CPU
      (NumPy) and uploaded as needed. To avoid repeated H2D transfers, callers
      may pass CuPy arrays for the DF blocks and intermediates; `cp.asarray` in
      the kernels will then be a no-op.
    - `h1e_v_sr` is assumed to already include the standard DF correction
      `-Σ_b (r b| b a)` (see `sr_h1e_v_correction_df_tiled`).
    """

    try:
        import cupy as cp  # noqa: PLC0415
    except Exception as e:  # pragma: no cover
        raise RuntimeError("CuPy is required for CUDA NEVPT2 kernels") from e

    if device is not None:
        cp.cuda.Device(int(device)).use()

    from asuka.mrpt2.nevpt2_sc import (  # noqa: PLC0415
        make_a12,
        make_a13,
        make_a16,
        make_a17,
        make_a19,
        make_a22,
        make_a23,
        make_a25,
        make_a3,
        make_a7,
        make_a9,
        make_hdm1,
        make_hdm2,
        make_hdm3,
        make_k27,
    )

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    h1e = np.asarray(h1e, dtype=np.float64)
    h2e = np.asarray(h2e, dtype=np.float64)
    f3ca = np.asarray(f3ca, dtype=np.float64)
    f3ac = np.asarray(f3ac, dtype=np.float64)

    nact = int(dm1.shape[0])
    n2 = int(nact * nact)

    # Active intermediates for each subspace (NumPy).
    hdm1 = make_hdm1(dm1)
    a3 = make_a3(h1e, h2e, dm1, dm2, hdm1)
    k27 = make_k27(h1e, h2e, dm1, dm2)

    rm2, a7 = make_a7(h1e, h2e, dm1, dm2, dm3)
    m_norm_srs = np.asarray(rm2.transpose(0, 1, 3, 2).reshape(n2, n2), order="C")
    m_h_srs = np.asarray(a7.reshape(n2, n2), order="C")

    hdm2 = make_hdm2(dm1, dm2)
    hdm3 = make_hdm3(dm1, dm2, dm3, hdm1, hdm2)
    a9 = make_a9(h1e, h2e, hdm1, hdm2, hdm3)
    m_norm_sij = np.asarray(hdm2.reshape(n2, n2), order="C")
    m_h_sij = np.asarray(a9.reshape(n2, n2), order="C")

    a12 = make_a12(h1e, h2e, dm1, dm2, dm3)
    a13 = make_a13(h1e, h2e, dm1, dm2, dm3)

    a16 = make_a16(h1e, h2e, dm3, f3ca=f3ca, f3ac=f3ac)
    a17 = make_a17(h1e, h2e, dm2, dm3)
    a19 = make_a19(h1e, h2e, dm1, dm2)

    a22 = make_a22(h1e, h2e, dm2, dm3, f3ca=f3ca, f3ac=f3ac)
    a23 = make_a23(h1e, h2e, dm1, dm2, dm3)
    a25 = make_a25(h1e, h2e, dm1, dm2)

    delta = np.eye(nact, dtype=np.float64)
    dm3_h = 2.0 * np.einsum("abef,cd->abcdef", dm2, delta, optimize=True) - dm3.transpose(0, 1, 3, 2, 4, 5)
    dm2_h = 2.0 * np.einsum("ab,cd->abcd", dm1, delta, optimize=True) - dm2.transpose(0, 1, 3, 2)
    dm1_h = 2.0 * delta - dm1.T

    # Keep DF blocks resident on device across subspace kernels.
    l_cv_d = _df_pair_block_to_device(l_cv, cp)
    l_vc_d = _df_pair_block_to_device(l_vc, cp)
    l_va_d = _df_pair_block_to_device(l_va, cp)
    l_ac_d = _df_pair_block_to_device(l_ac, cp)
    l_aa_d = _df_pair_block_to_device(l_aa, cp)

    # Subspace energies on GPU.
    norm_sijrs0, e_sijrs0 = sijrs0_energy_df_cuda(l_cv_d, eps_core, eps_virt, device=None, return_device=True)
    norm_sijr, e_sijr = sijr_p1_energy_df_cuda(
        l_vc_d, l_ac_d, hdm1=hdm1, a3=a3, eps_core=eps_core, eps_virt=eps_virt, device=None, return_device=True
    )
    norm_srsi, e_srsi = srsi_m1_energy_df_cuda(
        l_vc_d, l_va_d, dm1=dm1, k27=k27, eps_core=eps_core, eps_virt=eps_virt, device=None, return_device=True
    )
    norm_srs, e_srs = srs_m2_energy_df_cuda(
        l_va_d, m_norm=m_norm_srs, m_h=m_h_srs, eps_virt=eps_virt, device=None, return_device=True
    )
    norm_sij, e_sij = sij_p2_energy_df_cuda(
        l_ac_d, m_norm=m_norm_sij, m_h=m_h_sij, eps_core=eps_core, device=None, return_device=True
    )
    norm_sir, e_sir = sir_0_energy_df_cuda(
        l_vc_d,
        l_aa_d,
        l_va_d,
        l_ac_d,
        h1e_v_sir,
        dm1=dm1,
        dm2=dm2,
        a12=a12,
        a13=a13,
        eps_core=eps_core,
        eps_virt=eps_virt,
        device=None,
        return_device=True,
    )
    norm_sr, e_sr = sr_m1_prime_energy_df_cuda(
        l_va_d,
        l_aa_d,
        h1e_v_sr,
        dm1=dm1,
        dm2=dm2,
        dm3=dm3,
        a16=a16,
        a17=a17,
        a19=a19,
        eps_virt=eps_virt,
        device=None,
        return_device=True,
    )
    norm_si, e_si = si_p1_prime_energy_df_cuda(
        l_ac_d,
        l_aa_d,
        h1e_v_si,
        dm1_h=dm1_h,
        dm2_h=dm2_h,
        dm3_h=dm3_h,
        a22=a22,
        a23=a23,
        a25=a25,
        eps_core=eps_core,
        device=None,
        return_device=True,
    )

    e_total = e_sijrs0 + e_sijr + e_srsi + e_srs + e_sij + e_sir + e_sr + e_si
    norm_total = norm_sijrs0 + norm_sijr + norm_srsi + norm_srs + norm_sij + norm_sir + norm_sr + norm_si

    keys = [
        "norm_sijrs0",
        "e_sijrs0",
        "norm_sijr_p1",
        "e_sijr_p1",
        "norm_srsi_m1",
        "e_srsi_m1",
        "norm_srs_m2",
        "e_srs_m2",
        "norm_sij_p2",
        "e_sij_p2",
        "norm_sir_0",
        "e_sir_0",
        "norm_sr_m1_prime",
        "e_sr_m1_prime",
        "norm_si_p1_prime",
        "e_si_p1_prime",
        "norm_total",
        "e_total",
    ]
    vals_d = [
        norm_sijrs0,
        e_sijrs0,
        norm_sijr,
        e_sijr,
        norm_srsi,
        e_srsi,
        norm_srs,
        e_srs,
        norm_sij,
        e_sij,
        norm_sir,
        e_sir,
        norm_sr,
        e_sr,
        norm_si,
        e_si,
        norm_total,
        e_total,
    ]
    vals_h = cp.asnumpy(cp.stack(vals_d, axis=0))
    return {k: float(vals_h[i]) for i, k in enumerate(keys)}
