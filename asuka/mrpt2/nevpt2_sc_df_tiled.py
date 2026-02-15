from __future__ import annotations

import numpy as np

from asuka.mrpt2.df_pair_block import DFPairBlock


def sijrs0_energy_df_tiled(
    l_cv: DFPairBlock,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    *,
    virt_tile: int = 32,
    numerical_zero: float = 1e-14,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sijrs(0) (MP2-like core->virt doubles) with DF tiling.

    This matches `asuka.mrpt2.nevpt2_sc.sijrs0_energy_df` but avoids allocating
    the dense per-core tensor `g_i[a, j*b]` with shape `(nvirt, ncore*nvirt)`.

    Notes
    -----
    The contraction is performed per (i,j) over virtual tiles A/B:
      V_AB[a,b] = (i a| j b)
      theta_AB  = 2*V_AB - V_BA.T, where V_BA[b,a] = (i b| j a)
    """

    if int(virt_tile) <= 0:
        raise ValueError("virt_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    ncore = int(l_cv.nx)
    nvirt = int(l_cv.ny)
    naux = int(l_cv.naux)

    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_full = np.asarray(l_cv.l_full, dtype=np.float64)
    if l_full.shape != (ncore * nvirt, naux):
        raise ValueError("l_cv.l_full shape mismatch")

    tiles = list(range(0, nvirt, int(virt_tile)))

    norm_sum = 0.0
    e2 = 0.0

    for i in range(ncore):
        b_i = l_full[i * nvirt : (i + 1) * nvirt]  # (nvirt, naux)
        eps_i = float(eps_core[i])

        for j in range(ncore):
            b_j = l_full[j * nvirt : (j + 1) * nvirt]  # (nvirt, naux)
            eps_ij = eps_i + float(eps_core[j])

            for ta, a0 in enumerate(tiles):
                a1 = min(nvirt, a0 + int(virt_tile))
                ba = b_i[a0:a1]
                ea = eps_virt[a0:a1]

                for b0 in tiles[ta:]:
                    b1 = min(nvirt, b0 + int(virt_tile))
                    bb = b_j[b0:b1]
                    eb = eps_virt[b0:b1]

                    v_ab = ba @ bb.T  # (a,b)
                    denom_ab = eps_ij - ea[:, None] - eb[None, :]

                    if a0 == b0:
                        theta = 2.0 * v_ab - v_ab.T
                        norm_sum += float(np.sum(v_ab * theta))
                        e2 += float(np.sum((v_ab / denom_ab) * theta))
                        continue

                    v_ba = b_i[b0:b1] @ b_j[a0:a1].T  # (b,a)
                    theta_ab = 2.0 * v_ab - v_ba.T
                    theta_ba = 2.0 * v_ba - v_ab.T

                    denom_ba = denom_ab.T

                    norm_sum += float(np.sum(v_ab * theta_ab)) + float(np.sum(v_ba * theta_ba))
                    e2 += float(np.sum((v_ab / denom_ab) * theta_ab)) + float(np.sum((v_ba / denom_ba) * theta_ba))

    return float(norm_sum), float(e2)


def srs_m2_energy_df_tiled(
    l_va: DFPairBlock,
    *,
    m_norm: np.ndarray,
    m_h: np.ndarray,
    eps_virt: np.ndarray,
    s_tile: int = 32,
    numerical_zero: float = 1e-14,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Srs(-2) in a tiled DF form.

    This matches the dense `einsum` formulation used by PySCF, but avoids
    materializing the dense `(nvirt*nact, nvirt*nact)` intermediate that arises
    from `l_va.l_full @ l_va.l_full.T`.

    Parameters
    ----------
    l_va:
        DF pair block for (virt,act) ordered pairs, packed as ra_id = r*nact + a.
    m_norm, m_h:
        Active-space matrices with shapes `(nact^2, nact^2)` such that for each
        virtual pair (r,s) with packed integrals:

            x_rs[pq] = (r p | s q)  (pq_id = p*nact + q),

        we evaluate:

            norm_rs = 0.5 * x_rs^T m_norm x_rs
            h_rs    = 0.5 * x_rs^T m_h    x_rs

        and reduce energy via Hylleraas:

            E = -Σ_rs norm_rs / ( (eps_r + eps_s) + h_rs/norm_rs )
    eps_virt:
        Virtual orbital energies, length nvirt.
    s_tile:
        Tile size for the inner `s` loop (controls memory and GEMM size).
    numerical_zero:
        Norm threshold for skipping near-zero contributions.
    """

    if int(s_tile) <= 0:
        raise ValueError("s_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    naux = int(l_va.naux)
    n2 = nact * nact

    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    if eps_virt.size != nvirt:
        raise ValueError("eps_virt shape mismatch")

    m_norm = np.asarray(m_norm, dtype=np.float64)
    m_h = np.asarray(m_h, dtype=np.float64)
    if m_norm.shape != (n2, n2) or m_h.shape != (n2, n2):
        raise ValueError("m_norm/m_h must have shape (nact^2, nact^2)")

    l_full = np.asarray(l_va.l_full, dtype=np.float64)
    if l_full.shape != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")

    norm_sum = 0.0
    e2 = 0.0

    for r in range(nvirt):
        l_r = l_full[r * nact : (r + 1) * nact]  # (nact, naux) for (r,p)
        eps_r = float(eps_virt[r])

        for s0 in range(0, nvirt, int(s_tile)):
            s1 = min(nvirt, s0 + int(s_tile))
            ns = s1 - s0

            l_s = l_full[s0 * nact : s1 * nact]  # (ns*nact, naux) for (s,q)
            # g[p, s*q] = (r p| s q)
            g = l_r @ l_s.T  # (nact, ns*nact)

            # Pack each (r,s) block into x_rs[pq] with pq_id = p*nact + q.
            x = g.reshape(nact, ns, nact).transpose(1, 0, 2).reshape(ns, n2)

            xn = x @ m_norm
            norm = 0.5 * np.sum(x * xn, axis=1)

            xh = x @ m_h
            h = 0.5 * np.sum(x * xh, axis=1)

            diff = eps_r + eps_virt[s0:s1]

            mask = np.abs(norm) > float(numerical_zero)
            if np.any(mask):
                e2 -= float(np.sum(norm[mask] / (diff[mask] + h[mask] / norm[mask])))

            norm_sum += float(np.sum(norm))

    return float(norm_sum), float(e2)


def srsi_m1_energy_df_tiled(
    l_vc: DFPairBlock,
    l_va: DFPairBlock,
    *,
    dm1: np.ndarray,
    k27: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    virt_tile: int = 16,
    numerical_zero: float = 1e-14,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Srsi(-1) in a tiled DF form.

    The dense reference implementation builds `h2e_v[r,s,i,p] = (r i| s p)` and evaluates:

      norm_raw[r,s,i] = 2 * v_rs^T dm1 v_rs  - v_rs^T dm1 v_sr
      h_raw[r,s,i]    = 2 * v_rs^T k27 v_rs  - v_rs^T k27 v_sr

    where `v_rs[p] = (r i| s p)` and `v_sr[p] = (s i| r p)`, then symmetrizes in (r,s)
    and reduces over the upper triangle (r<=s).

    This tiled implementation computes the final (r<=s) contributions directly by
    streaming over (r,s) tiles and forming `v_rs`/`v_sr` via DF GEMMs.
    """

    if int(virt_tile) <= 0:
        raise ValueError("virt_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nvirt2, nact = int(l_va.nx), int(l_va.ny)
    if nvirt2 != nvirt:
        raise ValueError("l_vc and l_va virt dimensions mismatch")
    if int(l_vc.naux) != int(l_va.naux):
        raise ValueError("l_vc and l_va naux mismatch")
    naux = int(l_vc.naux)

    dm1 = np.asarray(dm1, dtype=np.float64)
    k27 = np.asarray(k27, dtype=np.float64)
    if dm1.shape != (nact, nact) or k27.shape != (nact, nact):
        raise ValueError("dm1/k27 shape mismatch")

    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_vc_full = np.asarray(l_vc.l_full, dtype=np.float64)
    l_va_full = np.asarray(l_va.l_full, dtype=np.float64)
    if l_vc_full.shape != (nvirt * ncore, naux):
        raise ValueError("l_vc.l_full shape mismatch")
    if l_va_full.shape != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")

    l_vc_3 = l_vc_full.reshape(nvirt, ncore, naux)

    norm_sum = 0.0
    e2 = 0.0

    for i in range(ncore):
        eps_i = float(eps_core[i])

        for r0 in range(0, nvirt, int(virt_tile)):
            r1 = min(nvirt, r0 + int(virt_tile))
            nr = r1 - r0

            # (r,p) DF vectors, packed row-major in (virt,act)
            l_rA = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
            # (r,i) DF vectors for fixed core index i
            l_ri = np.asarray(l_vc_3[r0:r1, i, :], order="C")  # (nr, naux)

            for s0 in range(r0, nvirt, int(virt_tile)):
                s1 = min(nvirt, s0 + int(virt_tile))
                ns = s1 - s0

                l_sA = l_va_full[s0 * nact : s1 * nact]  # (ns*nact, naux)
                l_si = np.asarray(l_vc_3[s0:s1, i, :], order="C")  # (ns, naux)

                # v_rs[r,s,p] = (r i| s p)  (computed as (s p| r i))
                g_rs = l_sA @ l_ri.T  # (ns*nact, nr)
                v_rs = g_rs.reshape(ns, nact, nr).transpose(2, 0, 1).reshape(nr * ns, nact)

                # v_sr[r,s,p] = (s i| r p)  (computed as (r p| s i))
                g_sr = l_rA @ l_si.T  # (nr*nact, ns)
                v_sr = g_sr.reshape(nr, nact, ns).transpose(0, 2, 1).reshape(nr * ns, nact)

                v_rs_dm = v_rs @ dm1
                v_sr_dm = v_sr @ dm1
                a_rs = np.sum(v_rs_dm * v_rs, axis=1)
                a_sr = np.sum(v_sr_dm * v_sr, axis=1)
                b_rs = np.sum(v_rs_dm * v_sr, axis=1)
                b_sr = np.sum(v_sr_dm * v_rs, axis=1)
                norm = 2.0 * (a_rs + a_sr) - (b_rs + b_sr)

                v_rs_k = v_rs @ k27
                v_sr_k = v_sr @ k27
                h_rs = np.sum(v_rs_k * v_rs, axis=1)
                h_sr = np.sum(v_sr_k * v_sr, axis=1)
                hk_rs = np.sum(v_rs_k * v_sr, axis=1)
                hk_sr = np.sum(v_sr_k * v_rs, axis=1)
                h = 2.0 * (h_rs + h_sr) - (hk_rs + hk_sr)

                diff = (eps_virt[r0:r1, None] + eps_virt[s0:s1][None, :] - eps_i).reshape(nr * ns)

                if s0 == r0:
                    if nr != ns:
                        raise RuntimeError("internal error: diagonal tile must be square")
                    norm_mat = norm.reshape(nr, nr)
                    h_mat = h.reshape(nr, nr)
                    diff_mat = diff.reshape(nr, nr)

                    diag = np.diag_indices(nr)
                    norm_mat[diag] *= 0.5
                    h_mat[diag] *= 0.5

                    triu = np.triu_indices(nr)
                    norm_use = norm_mat[triu]
                    h_use = h_mat[triu]
                    diff_use = diff_mat[triu]
                else:
                    norm_use = norm
                    h_use = h
                    diff_use = diff

                mask = np.abs(norm_use) > float(numerical_zero)
                if np.any(mask):
                    e2 -= float(np.sum(norm_use[mask] / (diff_use[mask] + h_use[mask] / norm_use[mask])))

                norm_sum += float(np.sum(norm_use))

    return float(norm_sum), float(e2)


def sijr_p1_energy_df_tiled(
    l_vc: DFPairBlock,
    l_ac: DFPairBlock,
    *,
    hdm1: np.ndarray,
    a3: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    r_tile: int = 4,
    numerical_zero: float = 1e-14,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sijr(+1) in a tiled DF form.

    The dense reference implementation builds `h2e_v[r,p,j,i] = (r i| p j)` and evaluates:

      norm_raw[r,j,i] = 2 * v_{rji}^T hdm1 v_{rji} - v_{rji}^T hdm1 v_{rij}
      h_raw[r,j,i]    = 2 * v_{rji}^T a3   v_{rji} - v_{rji}^T a3   v_{rij}

    then symmetrizes in (i,j), applies a 1/2 factor on the diagonal, and reduces
    over the upper triangle (j<=i).

    This tiled implementation streams over virtual orbitals in blocks and avoids
    materializing the full `(nvirt*nact*ncore*ncore)` mixed-integral tensor.
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nact, ncore2 = int(l_ac.nx), int(l_ac.ny)
    if ncore2 != ncore:
        raise ValueError("l_vc/l_ac core dimension mismatch")
    if int(l_vc.naux) != int(l_ac.naux):
        raise ValueError("l_vc/l_ac naux mismatch")
    naux = int(l_vc.naux)

    hdm1 = np.asarray(hdm1, dtype=np.float64)
    a3 = np.asarray(a3, dtype=np.float64)
    if hdm1.shape != (nact, nact) or a3.shape != (nact, nact):
        raise ValueError("hdm1/a3 shape mismatch")

    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_vc_full = np.asarray(l_vc.l_full, dtype=np.float64)
    l_ac_full = np.asarray(l_ac.l_full, dtype=np.float64)
    if l_vc_full.shape != (nvirt * ncore, naux):
        raise ValueError("l_vc.l_full shape mismatch")
    if l_ac_full.shape != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")

    ci_triu = np.triu_indices(ncore)

    norm_sum = 0.0
    e2 = 0.0

    l_ac_t = l_ac_full.T  # (naux, nact*ncore) (view)
    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = r1 - r0

        # Rows correspond to ordered pairs (r,i) with i varying fastest.
        l_ri = l_vc_full[r0 * ncore : r1 * ncore]  # (nr*ncore, naux)
        v = l_ri @ l_ac_t  # (nr*ncore, nact*ncore)

        for rr in range(nr):
            r = r0 + rr
            eps_r = float(eps_virt[r])

            v_r = v[rr * ncore : (rr + 1) * ncore]  # (ncore, nact*ncore)
            v_ipj = v_r.reshape(ncore, nact, ncore)  # (i,p,j)
            # w[j,i,p] = (r i| p j) matching the `rpji` label convention.
            w = v_ipj.transpose(2, 0, 1)  # (j,i,p)

            w_flat = w.reshape(ncore * ncore, nact)

            wx_n = w_flat @ hdm1
            a = np.sum(wx_n.reshape(ncore, ncore, nact) * w, axis=2)  # (j,i)
            b = np.einsum("jia,ija->ji", wx_n.reshape(ncore, ncore, nact), w, optimize=True)
            norm = 2.0 * a - b
            norm = norm + norm.T
            diag = np.diag_indices(ncore)
            norm[diag] *= 0.5

            wx_h = w_flat @ a3
            ah = np.sum(wx_h.reshape(ncore, ncore, nact) * w, axis=2)
            bh = np.einsum("jia,ija->ji", wx_h.reshape(ncore, ncore, nact), w, optimize=True)
            h = 2.0 * ah - bh
            h = h + h.T
            h[diag] *= 0.5

            diff = eps_r - eps_core[:, None] - eps_core[None, :]
            diff_use = diff[ci_triu]
            norm_use = norm[ci_triu]
            h_use = h[ci_triu]

            mask = np.abs(norm_use) > float(numerical_zero)
            if np.any(mask):
                e2 -= float(np.sum(norm_use[mask] / (diff_use[mask] + h_use[mask] / norm_use[mask])))

            norm_sum += float(np.sum(norm_use))

    return float(norm_sum), float(e2)


def sir_0_energy_df_tiled(
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
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sir(0) in a tiled DF form.

    This matches the dense PySCF-style `einsum` formulation in
    `asuka.mrpt2.nevpt2_sc.sir_0_energy_from_integrals`, but avoids building
    the full mixed-integral tensors:
      - `h2e_v1[r,p,i,q] = (r i| p q)` (from `l_vc @ l_aa^T`)
      - `h2e_v2[r,p,q,i] = (r p| q i)` (from `l_va @ l_ac^T`)
    which scale as `O(nvirt*ncore*nact^2)` in storage.
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nvirt, ncore = int(l_vc.nx), int(l_vc.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1:
        raise ValueError("l_aa must be square (act,act)")
    nact = nact0
    if int(l_va.nx) != nvirt or int(l_va.ny) != nact:
        raise ValueError("l_va shape mismatch (expect virt x act)")
    if int(l_ac.nx) != nact or int(l_ac.ny) != ncore:
        raise ValueError("l_ac shape mismatch (expect act x core)")
    if int(l_vc.naux) != int(l_aa.naux) or int(l_vc.naux) != int(l_va.naux) or int(l_vc.naux) != int(l_ac.naux):
        raise ValueError("DF naux mismatch")
    naux = int(l_vc.naux)

    h1e_v = np.asarray(h1e_v, dtype=np.float64)
    if h1e_v.shape != (nvirt, ncore):
        raise ValueError("h1e_v shape mismatch (expect virt x core)")

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    a12 = np.asarray(a12, dtype=np.float64)
    a13 = np.asarray(a13, dtype=np.float64)
    if dm1.shape != (nact, nact):
        raise ValueError("dm1 shape mismatch")
    if dm2.shape != (nact, nact, nact, nact):
        raise ValueError("dm2 shape mismatch")
    if a12.shape != (nact, nact, nact, nact) or a13.shape != (nact, nact, nact, nact):
        raise ValueError("a12/a13 shape mismatch")

    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    if eps_core.size != ncore or eps_virt.size != nvirt:
        raise ValueError("orbital energy shape mismatch")

    l_vc_full = np.asarray(l_vc.l_full, dtype=np.float64)
    l_aa_full = np.asarray(l_aa.l_full, dtype=np.float64)
    l_va_full = np.asarray(l_va.l_full, dtype=np.float64)
    l_ac_full = np.asarray(l_ac.l_full, dtype=np.float64)
    if l_vc_full.shape != (nvirt * ncore, naux):
        raise ValueError("l_vc.l_full shape mismatch")
    if l_aa_full.shape != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")
    if l_va_full.shape != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")
    if l_ac_full.shape != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")

    n2 = nact * nact
    m2 = dm2.transpose(1, 0, 2, 3).reshape(n2, n2)
    m2e = dm2.transpose(3, 0, 2, 1).reshape(n2, n2)
    m12 = a12.reshape(n2, n2)
    m13 = a13.reshape(n2, n2)
    dm1_t = dm1.T

    l_aa_t = l_aa_full.T  # (naux, nact*nact) (view)
    l_ac_t = l_ac_full.T  # (naux, nact*ncore) (view)

    norm_sum = 0.0
    e2 = 0.0

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = r1 - r0
        b = nr * ncore

        # v1_vec[(r,i),pq] = (r i| p q)
        l_ri = l_vc_full[r0 * ncore : r1 * ncore]  # (nr*ncore, naux) with i varying fastest
        v1 = l_ri @ l_aa_t  # (b, n2)

        # v2[p,q] = (r p| q i) for all (r,i) in this tile
        l_rp = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux) with p varying fastest
        g = l_rp @ l_ac_t  # (nr*nact, nact*ncore) (columns packed as q*ncore + i)
        v2_4 = g.reshape(nr, nact, nact, ncore)  # (r,p,q,i)
        v2 = v2_4.transpose(0, 3, 1, 2).reshape(b, n2)  # (r,i,p,q) -> (b,pq)

        v1_3 = v1.reshape(b, nact, nact)
        v2_3 = v2.reshape(b, nact, nact)

        h1 = h1e_v[r0:r1].reshape(b)
        diff = (eps_virt[r0:r1, None] - eps_core[None, :]).reshape(b)

        t1 = v1 @ m2
        t2 = v2 @ m2
        tE = v2 @ m2e

        termA = 2.0 * np.sum(v1 * t1, axis=1)
        termB = -np.sum(t1 * v2, axis=1)
        termC = -np.sum(t2 * v1, axis=1)
        termE = -np.sum(v2 * tE, axis=1)

        v2_dm = v2_3 @ dm1
        termD = 2.0 * np.sum(v2_dm * v2_3, axis=(1, 2))

        inner_v1 = np.sum(v1_3 * dm1_t, axis=(1, 2))
        inner_v2 = np.sum(v2_3 * dm1_t, axis=(1, 2))
        trace_v2 = np.einsum("bii->b", v2_3, optimize=True)
        termF = trace_v2 * inner_v2

        termG = 4.0 * h1 * inner_v1
        termH = -2.0 * h1 * inner_v2
        termI = 2.0 * h1 * h1

        norm = termA + termB + termC + termD + termE + termF + termG + termH + termI

        t12_v1 = v1 @ m12
        t12_v2 = v2 @ m12
        t13_v2 = v2 @ m13
        h = (
            2.0 * np.sum(v1 * t12_v1, axis=1)
            - np.sum(t12_v1 * v2, axis=1)
            - np.sum(t12_v2 * v1, axis=1)
            + np.sum(v2 * t13_v2, axis=1)
        )

        norm_sum += float(np.sum(norm))

        mask = np.abs(norm) > float(numerical_zero)
        if np.any(mask):
            e2 -= float(np.sum(norm[mask] / (diff[mask] + h[mask] / norm[mask])))

    return float(norm_sum), float(e2)


def sij_p2_energy_df_tiled(
    l_ac: DFPairBlock,
    *,
    m_norm: np.ndarray,
    m_h: np.ndarray,
    eps_core: np.ndarray,
    i_tile: int = 4,
    numerical_zero: float = 1e-14,
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sij(+2) in a tiled DF form.

    This matches the dense PySCF-style `einsum` formulation in
    `asuka.mrpt2.nevpt2_sc.sij_p2_energy_from_h2e_v`, but avoids materializing
    the dense `(nact*ncore, nact*ncore)` intermediate from `l_ac @ l_ac^T`.

    Parameters
    ----------
    l_ac:
        DF pair block for (act,core) ordered pairs.
    m_norm, m_h:
        Active-space matrices with shape `(nact^2, nact^2)` corresponding to
        `hdm2[p,q,a,b]` and `a9[p,q,a,b]` in ordered pair packing.
    eps_core:
        Core orbital energies, length ncore.
    """

    if int(i_tile) <= 0:
        raise ValueError("i_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nact, ncore = int(l_ac.nx), int(l_ac.ny)
    naux = int(l_ac.naux)
    n2 = nact * nact

    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    if eps_core.size != ncore:
        raise ValueError("eps_core shape mismatch")

    m_norm = np.asarray(m_norm, dtype=np.float64)
    m_h = np.asarray(m_h, dtype=np.float64)
    if m_norm.shape != (n2, n2) or m_h.shape != (n2, n2):
        raise ValueError("m_norm/m_h must have shape (nact^2, nact^2)")

    l_ac_full = np.asarray(l_ac.l_full, dtype=np.float64)
    if l_ac_full.shape != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")

    # Repack (p,i) vectors into a row order where i is the major index:
    # l_ip[i,p] corresponds to the same DF vector as l_ac[q=p, i].
    l_ip = np.asarray(l_ac_full.reshape(nact, ncore, naux).transpose(1, 0, 2).reshape(ncore * nact, naux), order="C")
    l_ip_t = l_ip.T  # (naux, ncore*nact) (view)

    norm_sum = 0.0
    e2 = 0.0

    for i0 in range(0, ncore, int(i_tile)):
        i1 = min(ncore, i0 + int(i_tile))
        ni = i1 - i0
        b = ni * ncore

        l_left = l_ip[i0 * nact : i1 * nact]  # (ni*nact, naux), rows packed as (i,p)
        g = l_left @ l_ip_t  # (ni*nact, ncore*nact), cols packed as (j,q)
        g4 = g.reshape(ni, nact, ncore, nact).transpose(0, 2, 1, 3)  # (i,j,p,q)
        v = g4.reshape(b, n2)

        tN = v @ m_norm
        norm = 0.5 * np.sum(v * tN, axis=1)
        tH = v @ m_h
        h = 0.5 * np.sum(v * tH, axis=1)

        diff = (-(eps_core[i0:i1, None] + eps_core[None, :])).reshape(b)

        norm_sum += float(np.sum(norm))

        mask = np.abs(norm) > float(numerical_zero)
        if np.any(mask):
            e2 -= float(np.sum(norm[mask] / (diff[mask] + h[mask] / norm[mask])))

    return float(norm_sum), float(e2)


def sr_h1e_v_correction_df_tiled(
    l_va: DFPairBlock,
    l_aa: DFPairBlock,
    *,
    r_tile: int = 16,
) -> np.ndarray:
    """Compute the DF correction term `J_sr[r,a] = Σ_b (r b| b a)` for Sr(-1)'.

    This is used to form the effective one-electron block for the Sr(-1)' subspace:

        h1e_v_sr[r,a] = (r|h_core+V_core|a) - Σ_b (r b| b a)

    The naive implementation materializes `h2e_v_sr[r,p,q,s]` and contracts the
    diagonal active indices. This helper computes the contraction directly from
    DF pair blocks without building the 4-index tensor.
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_va/l_aa active dimensions mismatch")
    if int(l_va.naux) != int(l_aa.naux):
        raise ValueError("l_va and l_aa naux mismatch")
    naux = int(l_va.naux)

    l_va_full = np.asarray(l_va.l_full, dtype=np.float64)
    l_aa_full = np.asarray(l_aa.l_full, dtype=np.float64)
    if l_va_full.shape != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")
    if l_aa_full.shape != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")

    # L_rb[r,b,L]
    l_rb = l_va_full.reshape(nvirt, nact, naux)
    # L_ba[b,a,L] (ordered active pairs)
    l_ba = l_aa_full.reshape(nact, nact, naux)

    out = np.zeros((nvirt, nact), dtype=np.float64)
    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        # out[r,a] = Σ_{b,L} L_rb[r,b,L] * L_ba[b,a,L]
        out[r0:r1] = np.einsum("rbL,baL->ra", l_rb[r0:r1], l_ba, optimize=True)
    return np.asarray(out, order="C")


def sr_m1_prime_energy_df_tiled(
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
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Sr(-1)' with a tiled DF contraction for `h2e_v`."""

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nvirt, nact = int(l_va.nx), int(l_va.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_va/l_aa active dimensions mismatch")
    if int(l_va.naux) != int(l_aa.naux):
        raise ValueError("l_va and l_aa naux mismatch")
    naux = int(l_va.naux)

    h1e_v = np.asarray(h1e_v, dtype=np.float64)
    if h1e_v.shape != (nvirt, nact):
        raise ValueError("h1e_v shape mismatch (expect virt x act)")

    eps_virt = np.asarray(eps_virt, dtype=np.float64).ravel()
    if eps_virt.size != nvirt:
        raise ValueError("eps_virt shape mismatch")

    dm1 = np.asarray(dm1, dtype=np.float64)
    dm2 = np.asarray(dm2, dtype=np.float64)
    dm3 = np.asarray(dm3, dtype=np.float64)
    a16 = np.asarray(a16, dtype=np.float64)
    a17 = np.asarray(a17, dtype=np.float64)
    a19 = np.asarray(a19, dtype=np.float64)
    if dm1.shape != (nact, nact) or dm2.shape != (nact, nact, nact, nact) or dm3.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm shape mismatch")
    if a16.shape != dm3.shape:
        raise ValueError("a16 shape mismatch")
    if a17.shape != (nact, nact, nact, nact):
        raise ValueError("a17 shape mismatch")
    if a19.shape != (nact, nact):
        raise ValueError("a19 shape mismatch")

    l_va_full = np.asarray(l_va.l_full, dtype=np.float64)
    l_aa_full = np.asarray(l_aa.l_full, dtype=np.float64)
    if l_va_full.shape != (nvirt * nact, naux):
        raise ValueError("l_va.l_full shape mismatch")
    if l_aa_full.shape != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")
    l_aa_t = l_aa_full.T  # (naux, nact*nact) (view)

    norm_sum = 0.0
    e2 = 0.0

    # h1e_v terms are independent of DF tiling.
    norm_1e = np.einsum("ip,pa,ia->i", h1e_v, dm1, h1e_v, optimize=True)
    h_1e = np.einsum("ip,pa,ia->i", h1e_v, a19, h1e_v, optimize=True)

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = r1 - r0

        l_rq = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux) rows packed as (r,q)
        v = l_rq @ l_aa_t  # (nr*nact, nact*nact) as (r,q|p,s)
        v4 = v.reshape(nr, nact, nact, nact)  # (r,q,p,s)
        h2e_v = np.asarray(v4.transpose(0, 2, 1, 3), order="C")  # (r,p,q,s)

        h = (
            np.einsum("ipqr,pqrabc,iabc->i", h2e_v, a16, h2e_v, optimize=True)
            + 2.0 * np.einsum("ipqr,pqra,ia->i", h2e_v, a17, h1e_v[r0:r1], optimize=True)
            + h_1e[r0:r1]
        )

        norm = (
            np.einsum("ipqr,rpqbac,iabc->i", h2e_v, dm3, h2e_v, optimize=True)
            + 2.0 * np.einsum("ipqr,rpqa,ia->i", h2e_v, dm2, h1e_v[r0:r1], optimize=True)
            + norm_1e[r0:r1]
        )

        norm_sum += float(np.sum(norm))

        diff = eps_virt[r0:r1]
        mask = np.abs(norm) > float(numerical_zero)
        if np.any(mask):
            e2 -= float(np.sum(norm[mask] / (diff[mask] + h[mask] / norm[mask])))

    return float(norm_sum), float(e2)


def si_p1_prime_energy_df_tiled(
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
) -> tuple[float, float]:
    """Compute SC-NEVPT2 Si(+1)' with a tiled DF contraction for `h2e_v`."""

    if int(i_tile) <= 0:
        raise ValueError("i_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nact, ncore = int(l_ac.nx), int(l_ac.ny)
    nact0, nact1 = int(l_aa.nx), int(l_aa.ny)
    if nact0 != nact1 or nact0 != nact:
        raise ValueError("l_ac/l_aa active dimensions mismatch")
    if int(l_ac.naux) != int(l_aa.naux):
        raise ValueError("l_ac and l_aa naux mismatch")
    naux = int(l_ac.naux)

    h1e_v = np.asarray(h1e_v, dtype=np.float64)
    if h1e_v.shape != (nact, ncore):
        raise ValueError("h1e_v shape mismatch (expect act x core)")

    eps_core = np.asarray(eps_core, dtype=np.float64).ravel()
    if eps_core.size != ncore:
        raise ValueError("eps_core shape mismatch")

    dm1_h = np.asarray(dm1_h, dtype=np.float64)
    dm2_h = np.asarray(dm2_h, dtype=np.float64)
    dm3_h = np.asarray(dm3_h, dtype=np.float64)
    a22 = np.asarray(a22, dtype=np.float64)
    a23 = np.asarray(a23, dtype=np.float64)
    a25 = np.asarray(a25, dtype=np.float64)
    if dm1_h.shape != (nact, nact) or dm2_h.shape != (nact, nact, nact, nact) or dm3_h.shape != (nact, nact, nact, nact, nact, nact):
        raise ValueError("dm_h shape mismatch")
    if a22.shape != dm3_h.shape:
        raise ValueError("a22 shape mismatch")
    if a23.shape != (nact, nact, nact, nact):
        raise ValueError("a23 shape mismatch")
    if a25.shape != (nact, nact):
        raise ValueError("a25 shape mismatch")

    l_ac_full = np.asarray(l_ac.l_full, dtype=np.float64)
    l_aa_full = np.asarray(l_aa.l_full, dtype=np.float64)
    if l_ac_full.shape != (nact * ncore, naux):
        raise ValueError("l_ac.l_full shape mismatch")
    if l_aa_full.shape != (nact * nact, naux):
        raise ValueError("l_aa.l_full shape mismatch")
    l_aa_t = l_aa_full.T  # (naux, nact*nact) (view)

    norm_sum = 0.0
    e2 = 0.0

    # One-electron-only terms.
    norm_1e = np.einsum("pi,pa,ai->i", h1e_v, dm1_h, h1e_v, optimize=True)
    h_1e = np.einsum("pi,pa,ai->i", h1e_v, a25, h1e_v, optimize=True)

    # Gather helper for rows (q,i) in l_ac: row_id = q*ncore + i.
    for i0 in range(0, ncore, int(i_tile)):
        i1 = min(ncore, i0 + int(i_tile))
        ni = i1 - i0

        rows = []
        for q in range(nact):
            base = q * ncore
            rows.extend(range(base + i0, base + i1))
        l_qi = np.asarray(l_ac_full[np.asarray(rows, dtype=np.int64)], order="C")  # (nact*ni, naux)

        v = l_qi @ l_aa_t  # (nact*ni, nact*nact) -> (q,i|p,r)
        v4 = v.reshape(nact, ni, nact, nact).transpose(0, 2, 1, 3)  # (q,p,i,r)

        h = (
            np.einsum("qpir,pqrabc,baic->i", v4, a22, v4, optimize=True)
            + 2.0 * np.einsum("qpir,pqra,ai->i", v4, a23, h1e_v[:, i0:i1], optimize=True)
            + h_1e[i0:i1]
        )

        norm = (
            np.einsum("qpir,rpqbac,baic->i", v4, dm3_h, v4, optimize=True)
            + 2.0 * np.einsum("qpir,rpqa,ai->i", v4, dm2_h, h1e_v[:, i0:i1], optimize=True)
            + norm_1e[i0:i1]
        )

        norm_sum += float(np.sum(norm))
        diff = -eps_core[i0:i1]
        mask = np.abs(norm) > float(numerical_zero)
        if np.any(mask):
            e2 -= float(np.sum(norm[mask] / (diff[mask] + h[mask] / norm[mask])))

    return float(norm_sum), float(e2)
