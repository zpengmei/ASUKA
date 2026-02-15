from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from asuka.mrpt2.df_pair_block import DFPairBlock


@dataclass(frozen=True)
class Sijrs0AdjointResult:
    """Adjoint result for the Sijrs(0) SC-NEVPT2 subspace (MP2-like core->virt doubles).

    This provides sensitivities of the *correlation energy contribution* `e2`
    with respect to the DF pair block and orbital energies used in denominators.
    """

    norm_sum: float
    e2: float

    dl_cv: np.ndarray  # shape (ncore*nvirt, naux)
    deps_core: np.ndarray  # shape (ncore,)
    deps_virt: np.ndarray  # shape (nvirt,)


def sijrs0_energy_df_tiled_adjoint(
    l_cv: DFPairBlock,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    *,
    virt_tile: int = 32,
) -> Sijrs0AdjointResult:
    """Compute Sijrs(0) energy and adjoints w.r.t. `l_cv` and orbital energies.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.sijrs0_energy_df_tiled`.
    - The returned derivatives are w.r.t the *inputs of that kernel*:
      - `l_cv.l_full` (the rectangular DF vectors for (core,virt) ordered pairs),
      - `eps_core`, `eps_virt` (used in the external denominators).
    """

    if int(virt_tile) <= 0:
        raise ValueError("virt_tile must be positive")

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

    dl_full = np.zeros_like(l_full)
    deps_c = np.zeros_like(eps_core)
    deps_v = np.zeros_like(eps_virt)

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
                ba = b_i[a0:a1]  # (na, naux)
                ea = eps_virt[a0:a1]  # (na,)

                for b0 in tiles[ta:]:
                    b1 = min(nvirt, b0 + int(virt_tile))
                    bb = b_j[b0:b1]  # (nb, naux)
                    eb = eps_virt[b0:b1]  # (nb,)

                    v_ab = ba @ bb.T  # (na, nb)
                    denom_ab = eps_ij - ea[:, None] - eb[None, :]  # (na, nb)

                    if a0 == b0:
                        # Forward (matches sijrs0_energy_df_tiled):
                        theta = 2.0 * v_ab - v_ab.T
                        norm_sum += float(np.sum(v_ab * theta))
                        e2 += float(np.sum((v_ab / denom_ab) * theta))

                        # Adjoint w.r.t V and denom.
                        # E = Σ (V/D) * (2V - V^T)
                        dV = (4.0 * v_ab - v_ab.T) / denom_ab - (v_ab / denom_ab).T
                        dD = -(v_ab * theta) / (denom_ab * denom_ab)

                        # Propagate V = ba @ bb^T.
                        d_ba = dV @ bb  # (na, naux)
                        d_bb = dV.T @ ba  # (nb, naux)
                        dl_full[i * nvirt + a0 : i * nvirt + a1] += d_ba
                        dl_full[j * nvirt + b0 : j * nvirt + b1] += d_bb

                        # Propagate denom = eps_i + eps_j - ea - eb.
                        s = float(np.sum(dD))
                        deps_c[i] += s
                        deps_c[j] += s

                        row = np.sum(dD, axis=1)
                        col = np.sum(dD, axis=0)
                        deps_v[a0:a1] -= row + col
                        continue

                    v_ba = b_i[b0:b1] @ b_j[a0:a1].T  # (nb, na)
                    denom_ba = denom_ab.T  # (nb, na)

                    theta_ab = 2.0 * v_ab - v_ba.T
                    theta_ba = 2.0 * v_ba - v_ab.T

                    # Forward (matches sijrs0_energy_df_tiled):
                    norm_sum += float(np.sum(v_ab * theta_ab)) + float(np.sum(v_ba * theta_ba))
                    e2 += float(np.sum((v_ab / denom_ab) * theta_ab)) + float(np.sum((v_ba / denom_ba) * theta_ba))

                    # Adjoint w.r.t V blocks.
                    dVab = (4.0 * v_ab - v_ba.T) / denom_ab - (v_ba / denom_ba).T
                    dVba = (4.0 * v_ba - v_ab.T) / denom_ba - (v_ab / denom_ab).T

                    # Adjoint w.r.t denominators.
                    dDab = -(v_ab * theta_ab) / (denom_ab * denom_ab)
                    dDba = -(v_ba * theta_ba) / (denom_ba * denom_ba)

                    # Propagate Vab = ba @ bb^T.
                    d_ba = dVab @ bb
                    d_bb = dVab.T @ ba
                    dl_full[i * nvirt + a0 : i * nvirt + a1] += d_ba
                    dl_full[j * nvirt + b0 : j * nvirt + b1] += d_bb

                    # Propagate Vba = b_i[b] @ b_j[a]^T.
                    bi_b = b_i[b0:b1]
                    bj_a = b_j[a0:a1]
                    d_bi_b = dVba @ bj_a
                    d_bj_a = dVba.T @ bi_b
                    dl_full[i * nvirt + b0 : i * nvirt + b1] += d_bi_b
                    dl_full[j * nvirt + a0 : j * nvirt + a1] += d_bj_a

                    # Propagate denominators.
                    s = float(np.sum(dDab) + np.sum(dDba))
                    deps_c[i] += s
                    deps_c[j] += s

                    deps_v[a0:a1] -= np.sum(dDab, axis=1) + np.sum(dDba, axis=0)
                    deps_v[b0:b1] -= np.sum(dDab, axis=0) + np.sum(dDba, axis=1)

    return Sijrs0AdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_cv=np.asarray(dl_full, dtype=np.float64, order="C"),
        deps_core=np.asarray(deps_c, dtype=np.float64, order="C"),
        deps_virt=np.asarray(deps_v, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SrsM2AdjointResult:
    """Adjoint result for the Srs(-2) SC-NEVPT2 subspace (2-particle external sector)."""

    norm_sum: float
    e2: float

    dl_va: np.ndarray  # shape (nvirt*nact, naux)
    deps_virt: np.ndarray  # shape (nvirt,)


def srs_m2_energy_df_tiled_adjoint(
    l_va: DFPairBlock,
    *,
    m_norm: np.ndarray,
    m_h: np.ndarray,
    eps_virt: np.ndarray,
    s_tile: int = 32,
    numerical_zero: float = 1e-14,
) -> SrsM2AdjointResult:
    """Compute Srs(-2) energy and adjoints w.r.t. `l_va` and `eps_virt`.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.srs_m2_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_va.l_full` (DF vectors for (virt,act) ordered pairs),
      - `eps_virt` (used in denominators `diff = eps_r + eps_s`).
    """

    if int(s_tile) <= 0:
        raise ValueError("s_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nvirt = int(l_va.nx)
    nact = int(l_va.ny)
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

    dl_full = np.zeros_like(l_full)
    deps_v = np.zeros_like(eps_virt)

    norm_sum = 0.0
    e2 = 0.0

    sym_mn = m_norm + m_norm.T
    sym_mh = m_h + m_h.T

    for r in range(nvirt):
        l_r = l_full[r * nact : (r + 1) * nact]  # (nact, naux) for (r,p)
        eps_r = float(eps_virt[r])

        for s0 in range(0, nvirt, int(s_tile)):
            s1 = min(nvirt, s0 + int(s_tile))
            ns = s1 - s0

            l_s = l_full[s0 * nact : s1 * nact]  # (ns*nact, naux) for (s,q)
            g = l_r @ l_s.T  # (nact, ns*nact) with columns packed as (s,q)

            x = g.reshape(nact, ns, nact).transpose(1, 0, 2).reshape(ns, n2)  # (ns, n2)

            xn = x @ m_norm
            norm = 0.5 * np.sum(x * xn, axis=1)
            xh = x @ m_h
            h = 0.5 * np.sum(x * xh, axis=1)
            diff = eps_r + eps_virt[s0:s1]

            norm_sum += float(np.sum(norm))

            mask = np.abs(norm) > float(numerical_zero)
            if not np.any(mask):
                continue

            n_m = norm[mask]
            h_m = h[mask]
            d_m = diff[mask]
            denom = d_m + h_m / n_m
            e2 -= float(np.sum(n_m / denom))

            # Adjoint w.r.t norm/h/diff for the masked rows.
            wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
            wH = 1.0 / (denom * denom)
            wD = n_m / (denom * denom)

            dx_m = 0.5 * (wN[:, None] * (x[mask] @ sym_mn) + wH[:, None] * (x[mask] @ sym_mh))

            # Scatter dx into the full x tile.
            dx = np.zeros_like(x)
            dx[mask] = dx_m

            dG = dx.reshape(ns, nact, nact).transpose(1, 0, 2).reshape(nact, ns * nact)

            # Backprop through g = l_r @ l_s^T.
            d_l_r = dG @ l_s  # (nact, naux)
            d_l_s = dG.T @ l_r  # (ns*nact, naux)
            dl_full[r * nact : (r + 1) * nact] += d_l_r
            dl_full[s0 * nact : s1 * nact] += d_l_s

            deps_v[r] += float(np.sum(wD))
            deps_v[s0 + np.nonzero(mask)[0]] += wD

    return SrsM2AdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_va=np.asarray(dl_full, dtype=np.float64, order="C"),
        deps_virt=np.asarray(deps_v, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SijP2AdjointResult:
    """Adjoint result for the Sij(+2) SC-NEVPT2 subspace (2-hole external sector)."""

    norm_sum: float
    e2: float

    dl_ac: np.ndarray  # shape (nact*ncore, naux)
    deps_core: np.ndarray  # shape (ncore,)


def sij_p2_energy_df_tiled_adjoint(
    l_ac: DFPairBlock,
    *,
    m_norm: np.ndarray,
    m_h: np.ndarray,
    eps_core: np.ndarray,
    i_tile: int = 4,
    numerical_zero: float = 1e-14,
) -> SijP2AdjointResult:
    """Compute Sij(+2) energy and adjoints w.r.t. `l_ac` and `eps_core`.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.sij_p2_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_ac.l_full` (DF vectors for (act,core) ordered pairs),
      - `eps_core` (used in denominators `diff = -(eps_i + eps_j)`).
    """

    if int(i_tile) <= 0:
        raise ValueError("i_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

    nact = int(l_ac.nx)
    ncore = int(l_ac.ny)
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

    dl_ip = np.zeros_like(l_ip)
    deps_c = np.zeros_like(eps_core)

    norm_sum = 0.0
    e2 = 0.0

    sym_mn = m_norm + m_norm.T
    sym_mh = m_h + m_h.T

    for i0 in range(0, ncore, int(i_tile)):
        i1 = min(ncore, i0 + int(i_tile))
        ni = i1 - i0
        b = ni * ncore

        l_left = l_ip[i0 * nact : i1 * nact]  # (ni*nact, naux), rows packed as (i,p)
        g = l_left @ l_ip_t  # (ni*nact, ncore*nact), cols packed as (j,q)
        g4 = g.reshape(ni, nact, ncore, nact).transpose(0, 2, 1, 3)  # (i,j,p,q)
        v = g4.reshape(b, n2)  # (i,j) rows, packed pq

        tN = v @ m_norm
        norm = 0.5 * np.sum(v * tN, axis=1)
        tH = v @ m_h
        h = 0.5 * np.sum(v * tH, axis=1)
        diff = (-(eps_core[i0:i1, None] + eps_core[None, :])).reshape(b)

        norm_sum += float(np.sum(norm))

        mask = np.abs(norm) > float(numerical_zero)
        if not np.any(mask):
            continue

        n_m = norm[mask]
        h_m = h[mask]
        d_m = diff[mask]
        denom = d_m + h_m / n_m
        e2 -= float(np.sum(n_m / denom))

        wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
        wH = 1.0 / (denom * denom)
        wD = n_m / (denom * denom)

        dv_m = 0.5 * (wN[:, None] * (v[mask] @ sym_mn) + wH[:, None] * (v[mask] @ sym_mh))

        dv = np.zeros_like(v)
        dv[mask] = dv_m

        dg4 = dv.reshape(ni, ncore, nact, nact)
        dg = dg4.transpose(0, 2, 1, 3).reshape(ni * nact, ncore * nact)

        # Backprop through g = l_left @ l_ip^T.
        dl_left = dg @ l_ip
        dl_right = (l_left.T @ dg).T
        dl_ip[i0 * nact : i1 * nact] += dl_left
        dl_ip += dl_right

        # Backprop through diff = -(eps_i + eps_j).
        # Row order is (i,j) with i in [i0,i1) and j in [0,ncore).
        wD_full = np.zeros(b, dtype=np.float64)
        wD_full[mask] = wD
        wD_ij = wD_full.reshape(ni, ncore)
        deps_c[i0:i1] -= np.sum(wD_ij, axis=1)
        deps_c -= np.sum(wD_ij, axis=0)

    # Map dl_ip back to the original l_ac row order (p major, i minor).
    dl_ac_full = dl_ip.reshape(ncore, nact, naux).transpose(1, 0, 2).reshape(nact * ncore, naux)

    return SijP2AdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_ac=np.asarray(dl_ac_full, dtype=np.float64, order="C"),
        deps_core=np.asarray(deps_c, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SrsiM1AdjointResult:
    """Adjoint result for the Srsi(-1) SC-NEVPT2 subspace."""

    norm_sum: float
    e2: float

    dl_vc: np.ndarray  # shape (nvirt*ncore, naux)
    dl_va: np.ndarray  # shape (nvirt*nact, naux)
    deps_core: np.ndarray  # shape (ncore,)
    deps_virt: np.ndarray  # shape (nvirt,)


def srsi_m1_energy_df_tiled_adjoint(
    l_vc: DFPairBlock,
    l_va: DFPairBlock,
    *,
    dm1: np.ndarray,
    k27: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    virt_tile: int = 16,
    numerical_zero: float = 1e-14,
) -> SrsiM1AdjointResult:
    """Compute Srsi(-1) energy and adjoints w.r.t. `l_vc`, `l_va`, and orbital energies.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.srsi_m1_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_vc.l_full` and `l_va.l_full`,
      - `eps_core` (only enters via `eps_i`),
      - `eps_virt` (enters as `eps_r + eps_s`).
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

    dl_vc_full = np.zeros_like(l_vc_full)
    dl_va_full = np.zeros_like(l_va_full)
    deps_c = np.zeros_like(eps_core)
    deps_v = np.zeros_like(eps_virt)

    dm1_sym = dm1 + dm1.T
    k27_sym = k27 + k27.T

    norm_sum = 0.0
    e2 = 0.0

    for i in range(ncore):
        eps_i = float(eps_core[i])

        for r0 in range(0, nvirt, int(virt_tile)):
            r1 = min(nvirt, r0 + int(virt_tile))
            nr = r1 - r0

            l_rA = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
            l_ri = np.asarray(l_vc_3[r0:r1, i, :], order="C")  # (nr, naux)

            for s0 in range(r0, nvirt, int(virt_tile)):
                s1 = min(nvirt, s0 + int(virt_tile))
                ns = s1 - s0

                l_sA = l_va_full[s0 * nact : s1 * nact]  # (ns*nact, naux)
                l_si = np.asarray(l_vc_3[s0:s1, i, :], order="C")  # (ns, naux)

                g_rs = l_sA @ l_ri.T  # (ns*nact, nr)
                v_rs = g_rs.reshape(ns, nact, nr).transpose(2, 0, 1).reshape(nr * ns, nact)

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
                    norm_scaled = np.array(norm_mat, copy=True)
                    h_scaled = np.array(h_mat, copy=True)
                    norm_scaled[diag] *= 0.5
                    h_scaled[diag] *= 0.5

                    triu = np.triu_indices(nr)
                    norm_use = norm_scaled[triu]
                    h_use = h_scaled[triu]
                    diff_use = diff_mat[triu]
                else:
                    norm_use = norm
                    h_use = h
                    diff_use = diff

                norm_sum += float(np.sum(norm_use))

                mask = np.abs(norm_use) > float(numerical_zero)
                if not np.any(mask):
                    continue

                n_m = norm_use[mask]
                h_m = h_use[mask]
                d_m = diff_use[mask]
                denom = d_m + h_m / n_m
                e2 -= float(np.sum(n_m / denom))

                wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
                wH = 1.0 / (denom * denom)
                wD = n_m / (denom * denom)

                # Scatter wN/wH/wD back to full (r,s) tile arrays.
                if s0 == r0:
                    gN_mat = np.zeros((nr, nr), dtype=np.float64)
                    gH_mat = np.zeros((nr, nr), dtype=np.float64)
                    gD_mat = np.zeros((nr, nr), dtype=np.float64)
                    tr0, tr1 = triu
                    gN_mat[tr0[mask], tr1[mask]] = wN
                    gH_mat[tr0[mask], tr1[mask]] = wH
                    gD_mat[tr0[mask], tr1[mask]] = wD
                    gN_mat[diag] *= 0.5
                    gH_mat[diag] *= 0.5
                    gN = gN_mat.reshape(nr * ns)
                    gH = gH_mat.reshape(nr * ns)
                    gD = gD_mat.reshape(nr * ns)
                else:
                    gN = np.zeros_like(norm)
                    gH = np.zeros_like(h)
                    gD = np.zeros_like(diff)
                    gN[mask] = wN
                    gH[mask] = wH
                    gD[mask] = wD

                # Backprop through diff = eps_r + eps_s - eps_i.
                gD_rs = gD.reshape(nr, ns)
                deps_c[i] -= float(np.sum(gD_rs))
                deps_v[r0:r1] += np.sum(gD_rs, axis=1)
                deps_v[s0:s1] += np.sum(gD_rs, axis=0)

                # Backprop through norm/h into v_rs and v_sr.
                gN_v = gN[:, None]
                gH_v = gH[:, None]

                vrs_dm1s = v_rs @ dm1_sym
                vsr_dm1s = v_sr @ dm1_sym
                vrs_k27s = v_rs @ k27_sym
                vsr_k27s = v_sr @ k27_sym

                dv_rs = 2.0 * gN_v * vrs_dm1s - gN_v * vsr_dm1s + 2.0 * gH_v * vrs_k27s - gH_v * vsr_k27s
                dv_sr = 2.0 * gN_v * vsr_dm1s - gN_v * vrs_dm1s + 2.0 * gH_v * vsr_k27s - gH_v * vrs_k27s

                dg_rs = dv_rs.reshape(nr, ns, nact).transpose(1, 2, 0).reshape(ns * nact, nr)
                dg_sr = dv_sr.reshape(nr, ns, nact).transpose(0, 2, 1).reshape(nr * nact, ns)

                # Backprop through g_rs = l_sA @ l_ri^T.
                dl_sA = dg_rs @ l_ri
                dl_ri = dg_rs.T @ l_sA
                dl_va_full[s0 * nact : s1 * nact] += dl_sA
                dl_vc_full.reshape(nvirt, ncore, naux)[r0:r1, i, :] += dl_ri

                # Backprop through g_sr = l_rA @ l_si^T.
                dl_rA = dg_sr @ l_si
                dl_si = dg_sr.T @ l_rA
                dl_va_full[r0 * nact : r1 * nact] += dl_rA
                dl_vc_full.reshape(nvirt, ncore, naux)[s0:s1, i, :] += dl_si

    return SrsiM1AdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_vc=np.asarray(dl_vc_full, dtype=np.float64, order="C"),
        dl_va=np.asarray(dl_va_full, dtype=np.float64, order="C"),
        deps_core=np.asarray(deps_c, dtype=np.float64, order="C"),
        deps_virt=np.asarray(deps_v, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SijrP1AdjointResult:
    """Adjoint result for the Sijr(+1) SC-NEVPT2 subspace."""

    norm_sum: float
    e2: float

    dl_vc: np.ndarray  # shape (nvirt*ncore, naux)
    dl_ac: np.ndarray  # shape (nact*ncore, naux)
    deps_core: np.ndarray  # shape (ncore,)
    deps_virt: np.ndarray  # shape (nvirt,)


def sijr_p1_energy_df_tiled_adjoint(
    l_vc: DFPairBlock,
    l_ac: DFPairBlock,
    *,
    hdm1: np.ndarray,
    a3: np.ndarray,
    eps_core: np.ndarray,
    eps_virt: np.ndarray,
    r_tile: int = 4,
    numerical_zero: float = 1e-14,
) -> SijrP1AdjointResult:
    """Compute Sijr(+1) energy and adjoints w.r.t. `l_vc`, `l_ac`, and orbital energies.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.sijr_p1_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_vc.l_full` and `l_ac.l_full`,
      - `eps_core` (enters as `-eps_i - eps_j`),
      - `eps_virt` (enters as `+eps_r`).
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
    tri0, tri1 = ci_triu

    dl_vc_full = np.zeros_like(l_vc_full)
    dl_ac_full = np.zeros_like(l_ac_full)
    deps_c = np.zeros_like(eps_core)
    deps_v = np.zeros_like(eps_virt)

    l_ac_t = l_ac_full.T  # (naux, nact*ncore) (view)

    sym_hdm1 = hdm1 + hdm1.T
    sym_a3 = a3 + a3.T

    norm_sum = 0.0
    e2 = 0.0

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = r1 - r0

        l_ri = l_vc_full[r0 * ncore : r1 * ncore]  # (nr*ncore, naux)
        v = l_ri @ l_ac_t  # (nr*ncore, nact*ncore)

        dv = np.zeros_like(v)

        for rr in range(nr):
            r = r0 + rr
            eps_r = float(eps_virt[r])

            v_r = v[rr * ncore : (rr + 1) * ncore]  # (ncore, nact*ncore)
            v_ipj = v_r.reshape(ncore, nact, ncore)  # (i,p,j)
            w = v_ipj.transpose(2, 0, 1)  # (j,i,p)

            w_flat = w.reshape(ncore * ncore, nact)

            wx_n = w_flat @ hdm1
            a = np.sum(wx_n.reshape(ncore, ncore, nact) * w, axis=2)
            b = np.einsum("jia,ija->ji", wx_n.reshape(ncore, ncore, nact), w, optimize=True)
            n_raw = 2.0 * a - b
            norm = n_raw + n_raw.T
            diag = np.diag_indices(ncore)
            norm[diag] *= 0.5

            wx_h = w_flat @ a3
            ah = np.sum(wx_h.reshape(ncore, ncore, nact) * w, axis=2)
            bh = np.einsum("jia,ija->ji", wx_h.reshape(ncore, ncore, nact), w, optimize=True)
            h_raw = 2.0 * ah - bh
            h = h_raw + h_raw.T
            h[diag] *= 0.5

            diff = eps_r - eps_core[:, None] - eps_core[None, :]
            norm_use = norm[ci_triu]
            h_use = h[ci_triu]
            diff_use = diff[ci_triu]

            norm_sum += float(np.sum(norm_use))

            mask = np.abs(norm_use) > float(numerical_zero)
            if not np.any(mask):
                continue

            n_m = norm_use[mask]
            h_m = h_use[mask]
            d_m = diff_use[mask]
            denom = d_m + h_m / n_m
            e2 -= float(np.sum(n_m / denom))

            wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
            wH = 1.0 / (denom * denom)
            wD = n_m / (denom * denom)

            gN_use = np.zeros_like(norm_use)
            gH_use = np.zeros_like(h_use)
            gN_use[mask] = wN
            gH_use[mask] = wH

            # Backprop diff = eps_r - eps_i - eps_j (upper triangle only).
            deps_v[r] += float(np.sum(wD))
            idx = np.nonzero(mask)[0]
            np.add.at(deps_c, tri0[idx], -wD)
            np.add.at(deps_c, tri1[idx], -wD)

            # Backprop norm/h symmetry + diagonal scaling:
            dB_n = np.zeros((ncore, ncore), dtype=np.float64)
            dB_h = np.zeros((ncore, ncore), dtype=np.float64)
            dB_n[ci_triu] = gN_use
            dB_h[ci_triu] = gH_use

            dA_n = np.array(dB_n, copy=True)
            dA_h = np.array(dB_h, copy=True)
            dA_n[diag] *= 0.5
            dA_h[diag] *= 0.5

            dN = dA_n + dA_n.T
            dH = dA_h + dA_h.T

            # Backprop to w (shape (j,i,p)).
            w_swap = w.transpose(1, 0, 2)

            dw = (2.0 * dN)[:, :, None] * (w @ sym_hdm1)
            dw -= dN[:, :, None] * (w_swap @ hdm1.T)
            dw += (-(dN[:, :, None] * (w @ hdm1))).transpose(1, 0, 2)

            dw += (2.0 * dH)[:, :, None] * (w @ sym_a3)
            dw -= dH[:, :, None] * (w_swap @ a3.T)
            dw += (-(dH[:, :, None] * (w @ a3))).transpose(1, 0, 2)

            dv_ipj = dw.transpose(1, 2, 0)  # (i,p,j)
            dv_r = dv_ipj.reshape(ncore, nact * ncore)
            dv[rr * ncore : (rr + 1) * ncore] += dv_r

        # Backprop through v = l_ri @ l_ac^T.
        dl_vc_full[r0 * ncore : r1 * ncore] += dv @ l_ac_full
        dl_ac_full += (l_ri.T @ dv).T

    return SijrP1AdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_vc=np.asarray(dl_vc_full, dtype=np.float64, order="C"),
        dl_ac=np.asarray(dl_ac_full, dtype=np.float64, order="C"),
        deps_core=np.asarray(deps_c, dtype=np.float64, order="C"),
        deps_virt=np.asarray(deps_v, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class Sir0AdjointResult:
    """Adjoint result for the Sir(0) SC-NEVPT2 subspace."""

    norm_sum: float
    e2: float

    dl_vc: np.ndarray  # shape (nvirt*ncore, naux)
    dl_aa: np.ndarray  # shape (nact*nact, naux)
    dl_va: np.ndarray  # shape (nvirt*nact, naux)
    dl_ac: np.ndarray  # shape (nact*ncore, naux)
    dh1e_v: np.ndarray  # shape (nvirt, ncore)
    deps_core: np.ndarray  # shape (ncore,)
    deps_virt: np.ndarray  # shape (nvirt,)


def sir_0_energy_df_tiled_adjoint(
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
) -> Sir0AdjointResult:
    """Compute Sir(0) energy and adjoints w.r.t. DF blocks, h1e_v, and orbital energies.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.sir_0_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_vc.l_full`, `l_aa.l_full`, `l_va.l_full`, `l_ac.l_full`,
      - `h1e_v` (virt x core),
      - `eps_core`, `eps_virt` (via `diff = eps_virt - eps_core`).
    """

    if int(r_tile) <= 0:
        raise ValueError("r_tile must be positive")
    if float(numerical_zero) <= 0.0:
        raise ValueError("numerical_zero must be positive")

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

    sym_m2 = m2 + m2.T
    sym_m12 = m12 + m12.T
    sym_m13 = m13 + m13.T
    sym_m2e = m2e + m2e.T
    dm1_sym = dm1 + dm1.T

    l_aa_t = l_aa_full.T  # (naux, nact*nact) (view)
    l_ac_t = l_ac_full.T  # (naux, nact*ncore) (view)

    dl_vc_full = np.zeros_like(l_vc_full)
    dl_aa_full = np.zeros_like(l_aa_full)
    dl_va_full = np.zeros_like(l_va_full)
    dl_ac_full = np.zeros_like(l_ac_full)
    dh1e_v = np.zeros_like(h1e_v)
    deps_c = np.zeros_like(eps_core)
    deps_v = np.zeros_like(eps_virt)

    norm_sum = 0.0
    e2 = 0.0

    eye_act = np.eye(nact, dtype=np.float64)
    dm1_t_flat = dm1_t.reshape(n2)

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = r1 - r0
        b = nr * ncore

        l_ri = l_vc_full[r0 * ncore : r1 * ncore]  # (b, naux)
        v1 = l_ri @ l_aa_t  # (b, n2)

        l_rp = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
        g = l_rp @ l_ac_t  # (nr*nact, nact*ncore)
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
        if not np.any(mask):
            continue

        n_m = norm[mask]
        h_m = h[mask]
        d_m = diff[mask]
        denom = d_m + h_m / n_m
        e2 -= float(np.sum(n_m / denom))

        wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
        wH = 1.0 / (denom * denom)
        wD = n_m / (denom * denom)

        gN = np.zeros_like(norm)
        gH = np.zeros_like(h)
        gD = np.zeros_like(diff)
        gN[mask] = wN
        gH[mask] = wH
        gD[mask] = wD

        # Backprop diff = eps_virt - eps_core.
        gD_mat = gD.reshape(nr, ncore)
        deps_v[r0:r1] += np.sum(gD_mat, axis=1)
        deps_c -= np.sum(gD_mat, axis=0)

        # Backprop h1 terms in norm.
        dh1 = gN * (4.0 * inner_v1 - 2.0 * inner_v2 + 4.0 * h1)
        dh1e_v[r0:r1] += dh1.reshape(nr, ncore)

        dv1 = np.zeros_like(v1)
        dv2 = np.zeros_like(v2)

        # Vectorized quadratic terms (norm): A/B/C/E.
        dv1 += (2.0 * gN)[:, None] * (v1 @ sym_m2) - gN[:, None] * (v2 @ sym_m2)
        dv2 += -gN[:, None] * (v1 @ sym_m2) - gN[:, None] * (v2 @ sym_m2e)

        # Vectorized quadratic terms (h): v1/v2.
        dv1 += (2.0 * gH)[:, None] * (v1 @ sym_m12) - gH[:, None] * (v2 @ sym_m12)
        dv2 += -gH[:, None] * (v1 @ sym_m12) + gH[:, None] * (v2 @ sym_m13)

        # Norm: termG contribution to v1 via inner_v1.
        dv1 += (4.0 * gN * h1)[:, None] * dm1_t_flat[None, :]

        # Norm: v2 contributions from termD/termF/termH (matrix form).
        gradY = (2.0 * gN)[:, None, None] * (v2_3 @ dm1_sym)
        gradY += (gN * inner_v2)[:, None, None] * eye_act[None, :, :]
        gradY += (gN * trace_v2)[:, None, None] * dm1_t[None, :, :]
        gradY += (-2.0 * gN * h1)[:, None, None] * dm1_t[None, :, :]
        dv2 += gradY.reshape(b, n2)

        # Backprop to DF blocks.
        dl_vc_full[r0 * ncore : r1 * ncore] += dv1 @ l_aa_full
        dl_aa_full += (l_ri.T @ dv1).T

        dg2 = dv2.reshape(nr, ncore, nact, nact).transpose(0, 2, 3, 1).reshape(nr * nact, nact * ncore)
        dl_va_full[r0 * nact : r1 * nact] += dg2 @ l_ac_full
        dl_ac_full += (l_rp.T @ dg2).T

    return Sir0AdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_vc=np.asarray(dl_vc_full, dtype=np.float64, order="C"),
        dl_aa=np.asarray(dl_aa_full, dtype=np.float64, order="C"),
        dl_va=np.asarray(dl_va_full, dtype=np.float64, order="C"),
        dl_ac=np.asarray(dl_ac_full, dtype=np.float64, order="C"),
        dh1e_v=np.asarray(dh1e_v, dtype=np.float64, order="C"),
        deps_core=np.asarray(deps_c, dtype=np.float64, order="C"),
        deps_virt=np.asarray(deps_v, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SrM1PrimeAdjointResult:
    """Adjoint result for the Sr(-1)' SC-NEVPT2 subspace."""

    norm_sum: float
    e2: float

    dl_va: np.ndarray  # shape (nvirt*nact, naux)
    dl_aa: np.ndarray  # shape (nact*nact, naux)
    dh1e_v: np.ndarray  # shape (nvirt, nact)
    deps_virt: np.ndarray  # shape (nvirt,)


def sr_m1_prime_energy_df_tiled_adjoint(
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
) -> SrM1PrimeAdjointResult:
    """Compute Sr(-1)' energy and adjoints w.r.t. DF blocks, h1e_v, and eps_virt.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.sr_m1_prime_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_va.l_full`, `l_aa.l_full`,
      - `h1e_v` (virt x act),
      - `eps_virt` (via `diff = eps_virt`).
    """

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

    dl_va_full = np.zeros_like(l_va_full)
    dl_aa_full = np.zeros_like(l_aa_full)
    dh1e_v = np.zeros_like(h1e_v)
    deps_v = np.zeros_like(eps_virt)

    norm_sum = 0.0
    e2 = 0.0

    sym_dm1 = dm1 + dm1.T
    sym_a19 = a19 + a19.T

    for r0 in range(0, nvirt, int(r_tile)):
        r1 = min(nvirt, r0 + int(r_tile))
        nr = r1 - r0

        h1_tile = h1e_v[r0:r1]  # (nr, nact)
        norm_1e = np.einsum("ip,pa,ia->i", h1_tile, dm1, h1_tile, optimize=True)
        h_1e = np.einsum("ip,pa,ia->i", h1_tile, a19, h1_tile, optimize=True)

        l_rq = l_va_full[r0 * nact : r1 * nact]  # (nr*nact, naux)
        v = l_rq @ l_aa_t  # (nr*nact, nact*nact) as (r,q|p,s)
        v4 = v.reshape(nr, nact, nact, nact)  # (r,q,p,s)
        h2e_v = np.asarray(v4.transpose(0, 2, 1, 3), order="C")  # (r,p,q,s)

        h = (
            np.einsum("ipqr,pqrabc,iabc->i", h2e_v, a16, h2e_v, optimize=True)
            + 2.0 * np.einsum("ipqr,pqra,ia->i", h2e_v, a17, h1_tile, optimize=True)
            + h_1e
        )

        norm = (
            np.einsum("ipqr,rpqbac,iabc->i", h2e_v, dm3, h2e_v, optimize=True)
            + 2.0 * np.einsum("ipqr,rpqa,ia->i", h2e_v, dm2, h1_tile, optimize=True)
            + norm_1e
        )

        norm_sum += float(np.sum(norm))

        diff = eps_virt[r0:r1]
        mask = np.abs(norm) > float(numerical_zero)
        if not np.any(mask):
            continue

        n_m = norm[mask]
        h_m = h[mask]
        d_m = diff[mask]
        denom = d_m + h_m / n_m
        e2 -= float(np.sum(n_m / denom))

        wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
        wH = 1.0 / (denom * denom)
        wD = n_m / (denom * denom)

        gN = np.zeros_like(norm)
        gH = np.zeros_like(h)
        gD = np.zeros_like(diff)
        gN[mask] = wN
        gH[mask] = wH
        gD[mask] = wD

        deps_v[r0:r1] += gD

        # Backprop into h2e_v.
        dh2 = np.einsum("pqrabc,iabc,i->ipqr", a16, h2e_v, gH, optimize=True)
        dh2 += np.einsum("ipqr,pqrabc,i->iabc", h2e_v, a16, gH, optimize=True)
        dh2 += np.einsum("rpqbac,iabc,i->ipqr", dm3, h2e_v, gN, optimize=True)
        dh2 += np.einsum("ipqr,rpqbac,i->iabc", h2e_v, dm3, gN, optimize=True)

        dh2 += 2.0 * np.einsum("pqra,ia,i->ipqr", a17, h1_tile, gH, optimize=True)
        dh2 += 2.0 * np.einsum("rpqa,ia,i->ipqr", dm2, h1_tile, gN, optimize=True)

        # Backprop into h1e_v.
        dh1 = 2.0 * np.einsum("ipqr,pqra,i->ia", h2e_v, a17, gH, optimize=True)
        dh1 += 2.0 * np.einsum("ipqr,rpqa,i->ia", h2e_v, dm2, gN, optimize=True)
        dh1 += gH[:, None] * (h1_tile @ sym_a19)
        dh1 += gN[:, None] * (h1_tile @ sym_dm1)

        dh1e_v[r0:r1] += dh1

        # Backprop h2e_v -> v -> DF blocks.
        dv4 = dh2.transpose(0, 2, 1, 3)  # (r,q,p,s)
        dv = dv4.reshape(nr * nact, nact * nact)
        dl_va_full[r0 * nact : r1 * nact] += dv @ l_aa_full
        dl_aa_full += (l_rq.T @ dv).T

    return SrM1PrimeAdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_va=np.asarray(dl_va_full, dtype=np.float64, order="C"),
        dl_aa=np.asarray(dl_aa_full, dtype=np.float64, order="C"),
        dh1e_v=np.asarray(dh1e_v, dtype=np.float64, order="C"),
        deps_virt=np.asarray(deps_v, dtype=np.float64, order="C"),
    )


@dataclass(frozen=True)
class SiP1PrimeAdjointResult:
    """Adjoint result for the Si(+1)' SC-NEVPT2 subspace."""

    norm_sum: float
    e2: float

    dl_ac: np.ndarray  # shape (nact*ncore, naux)
    dl_aa: np.ndarray  # shape (nact*nact, naux)
    dh1e_v: np.ndarray  # shape (nact, ncore)
    deps_core: np.ndarray  # shape (ncore,)


def si_p1_prime_energy_df_tiled_adjoint(
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
) -> SiP1PrimeAdjointResult:
    """Compute Si(+1)' energy and adjoints w.r.t. DF blocks, h1e_v, and eps_core.

    Notes
    -----
    - Matches the forward computation in `asuka.mrpt2.nevpt2_sc_df_tiled.si_p1_prime_energy_df_tiled`.
    - Derivatives are returned for the kernel inputs:
      - `l_ac.l_full`, `l_aa.l_full`,
      - `h1e_v` (act x core),
      - `eps_core` (via `diff = -eps_core`).
    """

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

    dl_ac_full = np.zeros_like(l_ac_full)
    dl_aa_full = np.zeros_like(l_aa_full)
    dh1e_v = np.zeros_like(h1e_v)
    deps_c = np.zeros_like(eps_core)

    norm_sum = 0.0
    e2 = 0.0

    sym_dm1h = dm1_h + dm1_h.T
    sym_a25 = a25 + a25.T

    # One-electron-only terms (per core index).
    norm_1e_all = np.einsum("pi,pa,ai->i", h1e_v, dm1_h, h1e_v, optimize=True)
    h_1e_all = np.einsum("pi,pa,ai->i", h1e_v, a25, h1e_v, optimize=True)

    for i0 in range(0, ncore, int(i_tile)):
        i1 = min(ncore, i0 + int(i_tile))
        ni = i1 - i0

        rows: list[int] = []
        for q in range(nact):
            base = q * ncore
            rows.extend(range(base + i0, base + i1))
        rows_arr = np.asarray(rows, dtype=np.int64)

        l_qi = np.asarray(l_ac_full[rows_arr], order="C")  # (nact*ni, naux)
        v = l_qi @ l_aa_t  # (nact*ni, nact*nact) -> (q,i|p,r)
        v4 = v.reshape(nact, ni, nact, nact).transpose(0, 2, 1, 3)  # (q,p,i,r)

        h = (
            np.einsum("qpir,pqrabc,baic->i", v4, a22, v4, optimize=True)
            + 2.0 * np.einsum("qpir,pqra,ai->i", v4, a23, h1e_v[:, i0:i1], optimize=True)
            + h_1e_all[i0:i1]
        )

        norm = (
            np.einsum("qpir,rpqbac,baic->i", v4, dm3_h, v4, optimize=True)
            + 2.0 * np.einsum("qpir,rpqa,ai->i", v4, dm2_h, h1e_v[:, i0:i1], optimize=True)
            + norm_1e_all[i0:i1]
        )

        norm_sum += float(np.sum(norm))

        diff = -eps_core[i0:i1]
        mask = np.abs(norm) > float(numerical_zero)
        if not np.any(mask):
            continue

        n_m = norm[mask]
        h_m = h[mask]
        d_m = diff[mask]
        denom = d_m + h_m / n_m
        e2 -= float(np.sum(n_m / denom))

        wN = -(d_m + 2.0 * h_m / n_m) / (denom * denom)
        wH = 1.0 / (denom * denom)
        wD = n_m / (denom * denom)

        gN = np.zeros_like(norm)
        gH = np.zeros_like(h)
        gD = np.zeros_like(diff)
        gN[mask] = wN
        gH[mask] = wH
        gD[mask] = wD

        deps_c[i0:i1] -= gD

        # Backprop into v4.
        dv4 = np.einsum("pqrabc,baic,i->qpir", a22, v4, gH, optimize=True)
        dv4 += np.einsum("qpir,pqrabc,i->baic", v4, a22, gH, optimize=True)
        dv4 += np.einsum("rpqbac,baic,i->qpir", dm3_h, v4, gN, optimize=True)
        dv4 += np.einsum("qpir,rpqbac,i->baic", v4, dm3_h, gN, optimize=True)

        dv4 += 2.0 * np.einsum("pqra,ai,i->qpir", a23, h1e_v[:, i0:i1], gH, optimize=True)
        dv4 += 2.0 * np.einsum("rpqa,ai,i->qpir", dm2_h, h1e_v[:, i0:i1], gN, optimize=True)

        # Backprop into h1e_v.
        dh = 2.0 * np.einsum("qpir,pqra,i->ai", v4, a23, gH, optimize=True)
        dh += 2.0 * np.einsum("qpir,rpqa,i->ai", v4, dm2_h, gN, optimize=True)
        dh += gH[None, :] * (sym_a25 @ h1e_v[:, i0:i1])
        dh += gN[None, :] * (sym_dm1h @ h1e_v[:, i0:i1])
        dh1e_v[:, i0:i1] += dh

        # Backprop v4 -> v -> DF blocks.
        dv_reshape = dv4.transpose(0, 2, 1, 3)  # (q,i,p,r)
        dv = dv_reshape.reshape(nact * ni, nact * nact)
        dl_qi = dv @ l_aa_full
        dl_aa_full += (l_qi.T @ dv).T

        # Scatter dl_qi back to the original l_ac rows.
        dl_ac_full[rows_arr] += dl_qi

    return SiP1PrimeAdjointResult(
        norm_sum=float(norm_sum),
        e2=float(e2),
        dl_ac=np.asarray(dl_ac_full, dtype=np.float64, order="C"),
        dl_aa=np.asarray(dl_aa_full, dtype=np.float64, order="C"),
        dh1e_v=np.asarray(dh1e_v, dtype=np.float64, order="C"),
        deps_core=np.asarray(deps_c, dtype=np.float64, order="C"),
    )
