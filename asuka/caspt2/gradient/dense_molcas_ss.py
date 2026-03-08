"""Dense Molcas-shaped SS-CASPT2 PT2 assembly helpers.

This module is intentionally dense-first and diagnostic-oriented. It mirrors
the object split used by OpenMolcas `Out_Pt2` / `PrepP` for the SS PT2
correction:

  - `D0(:,1)` : inactive reference density
  - `D0(:,2)` : PT2 variational density correction (`D1aoVar - 0.5*D0(:,1)`)
  - `D0(:,3)` : active reference density
  - `D0(:,4)` : `DLAO`
  - `FockOcc` : PT2 overlap/Pulay object

The exact SA/CASPT2 exchange (`BklK`) and active (`Thpkl`) lanes in Molcas are
not fully reconstructed here yet. The current implementation provides the exact
Molcas-shaped Coulomb and BTAMP pieces plus an explicit cross-exchange
candidate, so the live gradient path can compare the legacy heuristic lane and
the dense Molcas candidate lane term by term.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DenseMolcasTermInfo:
    formula: str
    molcas_source: str
    storage: str
    builder: str
    exact: bool


def _term_info(
    *,
    formula: str,
    molcas_source: str,
    storage: str,
    builder: str,
    exact: bool,
) -> dict[str, Any]:
    return {
        "formula": str(formula),
        "molcas_source": str(molcas_source),
        "storage": str(storage),
        "builder": str(builder),
        "exact": bool(exact),
    }


def _asnumpy_f64(a: Any) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _zeros_like_square(a: np.ndarray) -> np.ndarray:
    arr = _asnumpy_f64(a)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square matrix, got shape={arr.shape}")
    return np.zeros_like(arr, dtype=np.float64)


def _bar_cross_j(da: np.ndarray, db: np.ndarray) -> np.ndarray:
    Da = _asnumpy_f64(da)
    Db = _asnumpy_f64(db)
    return np.asarray(
        np.einsum("pq,rs->pqrs", Da, Db, optimize=True)
        + np.einsum("pq,rs->pqrs", Db, Da, optimize=True),
        dtype=np.float64,
    )


def _bar_cross_k(da: np.ndarray, db: np.ndarray) -> np.ndarray:
    Da = _asnumpy_f64(da)
    Db = _asnumpy_f64(db)
    return np.asarray(
        np.einsum("pr,qs->pqrs", Da, Db, optimize=True)
        + np.einsum("pr,qs->pqrs", Db, Da, optimize=True),
        dtype=np.float64,
    )


def _active_pair_columns(nact: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    n = int(nact)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.float64), []
    pairs: list[tuple[int, int]] = []
    cols: list[np.ndarray] = []
    for t in range(n):
        for u in range(t + 1):
            col = np.zeros((n * n,), dtype=np.float64)
            col[t * n + u] = 1.0
            if t != u:
                col[u * n + t] = 1.0
            cols.append(col)
            pairs.append((t, u))
    return np.asarray(np.stack(cols, axis=1), dtype=np.float64), pairs


def _pack_lower_triangle(mat: np.ndarray) -> np.ndarray:
    m = _asnumpy_f64(mat)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"expected square matrix, got shape={m.shape}")
    n = int(m.shape[0])
    out = np.zeros((n * (n + 1) // 2,), dtype=np.float64)
    k = 0
    for i in range(n):
        for j in range(i + 1):
            out[k] = float(m[i, j])
            k += 1
    return out


def _unpack_lower_triangle_to_symmetric(vec: np.ndarray, *, n: int) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).ravel()
    out = np.zeros((int(n), int(n)), dtype=np.float64)
    k = 0
    for i in range(int(n)):
        for j in range(i + 1):
            out[i, j] = float(v[k])
            out[j, i] = float(v[k])
            k += 1
    return out


def _tri0(i: int, j: int) -> int:
    ii = int(i)
    jj = int(j)
    if ii < jj:
        ii, jj = jj, ii
    return ii * (ii + 1) // 2 + jj


def _unwhiten_df_tensor(
    B_ao: np.ndarray,
    *,
    df_metric_chol: np.ndarray | None,
) -> np.ndarray:
    """Recover unwhitened 3c integrals ``X`` from whitened DF factors ``B``.

    ASUKA stores ``df_B`` as whitened factors with
    ``B = X @ L^{-T}``, where ``L`` is the lower Cholesky factor of the
    auxiliary metric. Molcas active lanes are naturally expressed in the
    unwhitened ``X(mu,nu,Q)`` space.
    """

    B = _asnumpy_f64(B_ao)
    if B.ndim != 3:
        raise ValueError(f"B_ao must have shape (nao,nao,naux), got {B.shape}")
    if df_metric_chol is None:
        return np.asarray(B, dtype=np.float64)
    L = _asnumpy_f64(df_metric_chol)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"df_metric_chol must be square, got {L.shape}")
    if int(L.shape[0]) != int(B.shape[2]):
        raise ValueError(
            f"df_metric_chol shape mismatch: expected ({int(B.shape[2])}, {int(B.shape[2])}), got {L.shape}"
        )
    return np.asarray(np.einsum("mnP,QP->mnQ", B, L, optimize=True), dtype=np.float64)


def build_molcas_exchange_barx_cross_candidate(
    *,
    B_ao: np.ndarray,
    df_metric_chol: np.ndarray | None,
    d0_inactive_ao: np.ndarray,
    d0_var_pt2_ao: np.ndarray,
    d0_active_ao: np.ndarray,
    d0_dlao_ao: np.ndarray,
) -> np.ndarray:
    """Dense DF-space candidate for the Molcas SA exchange lane.

    This is a closer analogue of the RI exchange adjoint than the legacy dense
    AO4 cross-K placeholder. For the SA pairs `(D0_1,D0_2)` and `(D0_3,D0_4)`,
    build the unwhitened DF adjoint

      bar_X_Q = -1/2 * (D1 X_Q D2 + D2 X_Q D1 + D3 X_Q D4 + D4 X_Q D3)

    where `X_Q` is the unwhitened 3c slice `(mu nu | Q)`.
    """

    X = _unwhiten_df_tensor(B_ao, df_metric_chol=df_metric_chol)
    D1 = _asnumpy_f64(d0_inactive_ao)
    D2 = _asnumpy_f64(d0_var_pt2_ao)
    D3 = _asnumpy_f64(d0_active_ao)
    D4 = _asnumpy_f64(d0_dlao_ao)
    if X.ndim != 3:
        raise ValueError(f"unexpected DF tensor shape: {X.shape}")
    bar = np.zeros_like(X, dtype=np.float64)
    for q in range(int(X.shape[2])):
        Xq = np.asarray(X[:, :, q], dtype=np.float64)
        bar[:, :, q] = -0.5 * (
            D1 @ Xq @ D2
            + D2 @ Xq @ D1
            + D3 @ Xq @ D4
            + D4 @ Xq @ D3
        )
    return np.asarray(0.5 * (bar + bar.transpose(1, 0, 2)), dtype=np.float64)


def _molcas_pack_pairpair_from_full_dm2(dm2: np.ndarray) -> np.ndarray:
    """Pack a full active-space 2-RDM into the Molcas pair-pair lower triangle.

    This mirrors the pair symmetrization used for `P2MO` / `PLMO` in
    `mclr/out_pt2.F90` and the state-averaged `D2av` storage consumed by
    `integral_util/prepp.F90`.
    """

    d = _asnumpy_f64(dm2)
    if d.ndim != 4 or len({int(x) for x in d.shape}) != 1:
        raise ValueError(f"dm2 must have shape (n,n,n,n), got {d.shape}")
    n = int(d.shape[0])
    npair = n * (n + 1) // 2
    out = np.zeros((npair * (npair + 1) // 2,), dtype=np.float64)

    for i in range(n):
        for j in range(i + 1):
            ij = _tri0(i, j)
            for k in range(i + 1):
                lmax = k if k < i else j
                for l in range(lmax + 1):
                    kl = _tri0(k, l)
                    out[_tri0(ij, kl)] = 0.25 * (
                        d[i, j, k, l]
                        + d[j, i, k, l]
                        + d[i, j, l, k]
                        + d[j, i, l, k]
                    )

    # Molcas halves selected diagonal-pair rows after the quartet average.
    for k in range(n):
        for l in range(k + 1):
            kl = _tri0(k, l)
            imax = k if l == k else (k - 1)
            for i in range(imax + 1):
                ii = _tri0(i, i)
                out[_tri0(kl, ii)] *= 0.5
    return np.asarray(out, dtype=np.float64)


def _molcas_pairpair_matrix_from_full_dm2(dm2: np.ndarray) -> np.ndarray:
    """Return the symmetric Molcas pair-pair matrix for a full active 2-RDM."""

    packed = _molcas_pack_pairpair_from_full_dm2(dm2)
    n = int(dm2.shape[0])
    npair = n * (n + 1) // 2
    out = np.zeros((npair, npair), dtype=np.float64)
    idx = 0
    for i in range(npair):
        for j in range(i + 1):
            out[i, j] = packed[idx]
            out[j, i] = packed[idx]
            idx += 1
    return np.asarray(out, dtype=np.float64)


def build_c1_sa_active_thpkl_candidate_bar(
    *,
    B_ao: np.ndarray,
    df_metric_chol: np.ndarray | None,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    dm2_d2av: np.ndarray,
    dm2_plmo: np.ndarray,
    ncore: int,
    ncas: int,
) -> np.ndarray:
    """Build a C1 SA active-lane `bar_X_ao(Q,mu,nu)` candidate.

    This follows the Molcas C1 SA channel structure:

      Z1 = G2(:,1) with Lxy(C,C)      where G2(:,1) = P2MO + PLMO
      Z2 = G2(:,2) with Lxy(C,C)      where G2(:,2) = D2av
      Z3 = G2(:,2) with Lxy(LCMO,C)

    For the C1 dense path this collapses to the exact full-tensor form of
    `Compute_txy -> CHO_GET_GRAD -> contract_Zpk_Tpxy -> PGet1_RI3`:

      M1 = U^T (P2MO + PLMO) U
      M2 = U^T D2av U
      z1 = 0.5 * M1 * pack(C^T X_Q C)
      z2 = 0.5 * M2 * pack(C^T X_Q C)
      z3 = 0.5 * M2 * pack(sym(LCMO^T X_Q C))

    The result is assembled into an AO-space `bar_X_ao`
    candidate:

      C Z1 C^T + L Z2 C^T + 2 C Z3 C^T + C Z2 L^T

    where `C` is the active block of the reference orbitals and `L` is the
    active block of `LCMO = C @ Kappa`.
    """

    X = _unwhiten_df_tensor(B_ao, df_metric_chol=df_metric_chol)
    C = _asnumpy_f64(mo_coeff)
    K = _asnumpy_f64(z_orb)
    dm2_avg = _asnumpy_f64(dm2_d2av)
    dm2_pl = _asnumpy_f64(dm2_plmo)
    if X.ndim != 3:
        raise ValueError(f"B_ao must have shape (nao,nao,naux), got {X.shape}")
    nao, nao2, naux = map(int, X.shape)
    if nao != nao2:
        raise ValueError(f"B_ao must have square AO dimensions, got {X.shape}")
    if C.ndim != 2 or int(C.shape[0]) != int(nao):
        raise ValueError(f"mo_coeff shape mismatch: got {C.shape}, expected (nao,nmo)")
    nocc = int(ncore + ncas)
    if nocc > int(C.shape[1]):
        raise ValueError("invalid ncore/ncas for mo_coeff")
    if dm2_avg.shape != (int(ncas), int(ncas), int(ncas), int(ncas)):
        raise ValueError(f"dm2_d2av shape mismatch: got {dm2_avg.shape}")
    if dm2_pl.shape != (int(ncas), int(ncas), int(ncas), int(ncas)):
        raise ValueError(f"dm2_plmo shape mismatch: got {dm2_pl.shape}")

    C_act = np.asarray(C[:, int(ncore) : int(nocc)], dtype=np.float64)
    C_lcmo_act = np.asarray((C @ K)[:, int(ncore) : int(nocc)], dtype=np.float64)

    U_pair, _pairs = _active_pair_columns(int(ncas))
    if U_pair.size == 0:
        return np.zeros((int(naux), int(nao), int(nao)), dtype=np.float64)
    g1 = np.asarray(
        U_pair.T
        @ np.asarray((dm2_avg + dm2_pl).reshape(int(ncas) * int(ncas), int(ncas) * int(ncas)), dtype=np.float64)
        @ U_pair,
        dtype=np.float64,
    )
    g2 = np.asarray(
        U_pair.T
        @ np.asarray(dm2_avg.reshape(int(ncas) * int(ncas), int(ncas) * int(ncas)), dtype=np.float64)
        @ U_pair,
        dtype=np.float64,
    )

    bar = np.zeros((int(naux), int(nao), int(nao)), dtype=np.float64)
    for q in range(int(naux)):
        l_cc = np.asarray(C_act.T @ X[:, :, q] @ C_act, dtype=np.float64)
        l_lc = np.asarray(C_lcmo_act.T @ X[:, :, q] @ C_act, dtype=np.float64)
        l_lc_sym = 0.5 * (l_lc + l_lc.T)

        z1 = np.asarray(0.5 * (g1 @ _pack_lower_triangle(l_cc)), dtype=np.float64)
        z2 = np.asarray(0.5 * (g2 @ _pack_lower_triangle(l_cc)), dtype=np.float64)
        z3 = np.asarray(0.5 * (g2 @ _pack_lower_triangle(l_lc_sym)), dtype=np.float64)

        Z1 = _unpack_lower_triangle_to_symmetric(z1, n=int(ncas))
        Z2 = _unpack_lower_triangle_to_symmetric(z2, n=int(ncas))
        Z3 = _unpack_lower_triangle_to_symmetric(z3, n=int(ncas))

        bar[q] = (
            C_act @ Z1 @ C_act.T
            + C_lcmo_act @ Z2 @ C_act.T
            + 2.0 * (C_act @ Z3 @ C_act.T)
            + C_act @ Z2 @ C_lcmo_act.T
        )
    return np.asarray(0.5 * (bar + bar.transpose(0, 2, 1)), dtype=np.float64)


def build_molcas_d1aovar_pt2_parts_ao(
    *,
    dpt2_ao: np.ndarray,
    dpt2c_ao: np.ndarray | None = None,
    d_ci_1e_ao: np.ndarray | None = None,
    d_orb_1e_ao: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build decomposed PT2 contributions to Molcas `D1aoVar` in AO basis.

    `Out_Pt2` forms the effective MO density correction as:

      D_K += DPT2 + 0.25 * (DPT2C + DPT2C^T) + response

    which is then transformed into `D1aoVar`.
    """
    dpt2 = np.asarray(_asnumpy_f64(dpt2_ao), dtype=np.float64)
    dpt2c_quarter = np.zeros_like(dpt2, dtype=np.float64)
    if dpt2c_ao is not None:
        dpt2c = _asnumpy_f64(dpt2c_ao)
        dpt2c_quarter = np.asarray(0.25 * (dpt2c + dpt2c.T), dtype=np.float64)
    d_ci = np.zeros_like(dpt2, dtype=np.float64)
    if d_ci_1e_ao is not None:
        d_ci = np.asarray(_asnumpy_f64(d_ci_1e_ao), dtype=np.float64)
    d_orb = np.zeros_like(dpt2, dtype=np.float64)
    if d_orb_1e_ao is not None:
        d_orb = np.asarray(_asnumpy_f64(d_orb_1e_ao), dtype=np.float64)
    total = np.asarray(dpt2 + dpt2c_quarter + d_ci + d_orb, dtype=np.float64)
    return {
        "dpt2": dpt2,
        "dpt2c_sym_quarter": dpt2c_quarter,
        "ci": d_ci,
        "orb": d_orb,
        "total": total,
    }


def build_molcas_reference_density_mo(
    *,
    nmo: int,
    ncore: int,
    ncas: int,
    dm1_ref_active_mo: np.ndarray,
) -> np.ndarray:
    """Build the reference MO density used by Molcas `Out_Pt2`."""

    nmo_i = int(nmo)
    ncore_i = int(ncore)
    ncas_i = int(ncas)
    dm1_act = np.asarray(dm1_ref_active_mo, dtype=np.float64)
    if dm1_act.shape != (ncas_i, ncas_i):
        raise ValueError(
            f"dm1_ref_active_mo shape mismatch: expected {(ncas_i, ncas_i)}, got {dm1_act.shape}"
        )
    out = np.zeros((nmo_i, nmo_i), dtype=np.float64)
    if ncore_i > 0:
        out[:ncore_i, :ncore_i] = 2.0 * np.eye(ncore_i, dtype=np.float64)
    nocc = int(ncore_i + ncas_i)
    out[ncore_i:nocc, ncore_i:nocc] = 0.5 * (dm1_act + dm1_act.T)
    return np.asarray(out, dtype=np.float64)


def build_molcas_oitd_mo(
    *,
    z_orb: np.ndarray,
    ncore: int,
    ncas: int,
    dm1_ref_active_mo: np.ndarray,
    act: bool,
) -> np.ndarray:
    """Replay Molcas `OITD` in MO space for C1 dense debugging."""

    K = np.asarray(z_orb, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"z_orb must be square, got {K.shape}")
    dtmp = build_molcas_reference_density_mo(
        nmo=int(K.shape[0]),
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_ref_active_mo=np.asarray(dm1_ref_active_mo, dtype=np.float64) if bool(act) else np.zeros((int(ncas), int(ncas)), dtype=np.float64),
    )
    out = np.asarray(dtmp @ K.T - K.T @ dtmp, dtype=np.float64)
    return np.asarray(0.5 * (out + out.T), dtype=np.float64)


def transform_molcas_mo_square_to_ao(
    *,
    mo_coeff: np.ndarray,
    mo_square: np.ndarray,
) -> np.ndarray:
    """Transform a Molcas-style C1 MO square matrix to AO square storage."""

    C = np.asarray(mo_coeff, dtype=np.float64)
    D = np.asarray(mo_square, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    if D.shape != (int(C.shape[1]), int(C.shape[1])):
        raise ValueError(
            f"mo_square shape mismatch: expected {(int(C.shape[1]), int(C.shape[1]))}, got {D.shape}"
        )
    return np.asarray(C @ D @ C.T, dtype=np.float64)


def symmetrize_molcas_packed_square(
    *,
    ao_square: np.ndarray,
) -> np.ndarray:
    """Return the unpacked AO-square analogue of Molcas `TCMO(-2)` + pack.

    `Make_Conn` stores a general AO square matrix by writing the diagonal as-is
    and each off-diagonal packed element as `A_ij + A_ji`. When that packed
    vector is unpacked back to a dense square for debugging, the natural matrix
    analogue is the symmetric part `0.5 * (A + A^T)`.
    """

    A = np.asarray(ao_square, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"ao_square must be square, got {A.shape}")
    return np.asarray(0.5 * (A + A.T), dtype=np.float64)


def build_molcas_d1aovar_replay_parts(
    *,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    dm1_ref_active_mo: np.ndarray,
    dm1_ci_active_mo: np.ndarray | None,
    dpt2_mo: np.ndarray,
    dpt2c_mo: np.ndarray | None,
    ncore: int,
    ncas: int,
) -> dict[str, np.ndarray]:
    """Replay the Molcas `Out_Pt2` D1aoVar density construction in C1."""

    C = np.asarray(mo_coeff, dtype=np.float64)
    K = np.asarray(z_orb, dtype=np.float64)
    Dpt2 = np.asarray(dpt2_mo, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nmo = int(C.shape[1])
    if K.shape != (nmo, nmo):
        raise ValueError(f"z_orb shape mismatch: expected {(nmo, nmo)}, got {K.shape}")
    if Dpt2.shape != (nmo, nmo):
        raise ValueError(f"dpt2_mo shape mismatch: expected {(nmo, nmo)}, got {Dpt2.shape}")

    d_ref_mo = build_molcas_reference_density_mo(
        nmo=int(nmo),
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_ref_active_mo=np.asarray(dm1_ref_active_mo, dtype=np.float64),
    )
    d_oitd_mo = build_molcas_oitd_mo(
        z_orb=K,
        ncore=int(ncore),
        ncas=int(ncas),
        dm1_ref_active_mo=np.asarray(dm1_ref_active_mo, dtype=np.float64),
        act=True,
    )
    d_ci_mo = np.zeros((nmo, nmo), dtype=np.float64)
    if dm1_ci_active_mo is not None:
        dm1_ci = np.asarray(dm1_ci_active_mo, dtype=np.float64)
        if dm1_ci.shape != (int(ncas), int(ncas)):
            raise ValueError(
                f"dm1_ci_active_mo shape mismatch: expected {(int(ncas), int(ncas))}, got {dm1_ci.shape}"
            )
        nocc = int(ncore + ncas)
        d_ci_mo[int(ncore):nocc, int(ncore):nocc] = 0.5 * (dm1_ci + dm1_ci.T)
    dpt2c_quarter_mo = np.zeros_like(Dpt2, dtype=np.float64)
    if dpt2c_mo is not None:
        Dpt2c = np.asarray(dpt2c_mo, dtype=np.float64)
        if Dpt2c.shape != (nmo, nmo):
            raise ValueError(
                f"dpt2c_mo shape mismatch: expected {(nmo, nmo)}, got {Dpt2c.shape}"
            )
        dpt2c_quarter_mo = np.asarray(0.25 * (Dpt2c + Dpt2c.T), dtype=np.float64)

    d_total_mo = np.asarray(d_ref_mo + d_oitd_mo + d_ci_mo + Dpt2 + dpt2c_quarter_mo, dtype=np.float64)
    d_ref_ao = transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=d_ref_mo)
    d_oitd_ao = transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=d_oitd_mo)
    d_ci_ao = transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=d_ci_mo)
    dpt2_ao = transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=Dpt2)
    dpt2c_quarter_ao = transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=dpt2c_quarter_mo)
    d_total_ao = transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=d_total_mo)
    d_pt2_ao = np.asarray(d_total_ao - d_ref_ao, dtype=np.float64)
    return {
        "ref_mo": np.asarray(d_ref_mo, dtype=np.float64),
        "oitd_mo": np.asarray(d_oitd_mo, dtype=np.float64),
        "ci_mo": np.asarray(d_ci_mo, dtype=np.float64),
        "dpt2_mo": np.asarray(Dpt2, dtype=np.float64),
        "dpt2c_sym_quarter_mo": np.asarray(dpt2c_quarter_mo, dtype=np.float64),
        "total_mo": np.asarray(d_total_mo, dtype=np.float64),
        "ref_ao": np.asarray(d_ref_ao, dtype=np.float64),
        "oitd_ao": np.asarray(d_oitd_ao, dtype=np.float64),
        "ci_ao": np.asarray(d_ci_ao, dtype=np.float64),
        "dpt2_ao": np.asarray(dpt2_ao, dtype=np.float64),
        "dpt2c_sym_quarter_ao": np.asarray(dpt2c_quarter_ao, dtype=np.float64),
        "total_ao": np.asarray(d_total_ao, dtype=np.float64),
        "pt2_ao": np.asarray(d_pt2_ao, dtype=np.float64),
    }


def build_molcas_d1aovar_pt2_ao(
    *,
    dpt2_ao: np.ndarray,
    dpt2c_ao: np.ndarray | None = None,
    d_ci_1e_ao: np.ndarray | None = None,
    d_orb_1e_ao: np.ndarray | None = None,
) -> np.ndarray:
    """Build the PT2 correction to Molcas `D1aoVar` in AO basis."""
    return np.asarray(
        build_molcas_d1aovar_pt2_parts_ao(
            dpt2_ao=dpt2_ao,
            dpt2c_ao=dpt2c_ao,
            d_ci_1e_ao=d_ci_1e_ao,
            d_orb_1e_ao=d_orb_1e_ao,
        )["total"],
        dtype=np.float64,
    )


def build_molcas_dlao_candidate_ao(
    *,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    dpt2_mo: np.ndarray,
    ncore: int,
    ncas: int = 0,
) -> np.ndarray:
    """Reconstruct the Molcas `DLAO` candidate in AO basis."""

    C = _asnumpy_f64(mo_coeff)
    Dpt2 = _asnumpy_f64(dpt2_mo)
    dao_mo = np.asarray(
        build_molcas_oitd_mo(
            z_orb=np.asarray(z_orb, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
            dm1_ref_active_mo=np.zeros((int(ncas), int(ncas)), dtype=np.float64),
            act=False,
        )
        + Dpt2,
        dtype=np.float64,
    )
    return transform_molcas_mo_square_to_ao(mo_coeff=C, mo_square=dao_mo)


def build_molcas_fockocc_pt2_parts_ao(
    *,
    w_hf_ao: np.ndarray,
    w_ci_ao: np.ndarray | None = None,
    w_orb_ao: np.ndarray | None = None,
    w_comm_ao: np.ndarray | None = None,
    w_fockgen_ao: np.ndarray | None = None,
    w_addgrad_ao: np.ndarray | None = None,
    w_rint_candidate_ao: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Build decomposed PT2 contributions to Molcas `FockOcc` in AO basis."""
    w_hf = symmetrize_molcas_packed_square(
        ao_square=np.asarray(_asnumpy_f64(w_hf_ao), dtype=np.float64)
    )
    w_ci = np.zeros_like(w_hf, dtype=np.float64)
    if w_ci_ao is not None:
        w_ci = np.asarray(_asnumpy_f64(w_ci_ao), dtype=np.float64)
    w_orb = np.zeros_like(w_hf, dtype=np.float64)
    if w_orb_ao is not None:
        w_orb = np.asarray(_asnumpy_f64(w_orb_ao), dtype=np.float64)
    w_comm = np.zeros_like(w_hf, dtype=np.float64)
    if w_comm_ao is not None:
        w_comm = np.asarray(_asnumpy_f64(w_comm_ao), dtype=np.float64)
    w_fockgen = np.zeros_like(w_hf, dtype=np.float64)
    if w_fockgen_ao is not None:
        w_fockgen = np.asarray(_asnumpy_f64(w_fockgen_ao), dtype=np.float64)
    w_addgrad = np.zeros_like(w_hf, dtype=np.float64)
    if w_addgrad_ao is not None:
        w_addgrad = np.asarray(_asnumpy_f64(w_addgrad_ao), dtype=np.float64)
    w_rint_candidate = np.zeros_like(w_hf, dtype=np.float64)
    if w_rint_candidate_ao is not None:
        w_rint_candidate = np.asarray(_asnumpy_f64(w_rint_candidate_ao), dtype=np.float64)
    # Molcas Make_Conn consumes 0.5 * FockA (pre-AddGrad), then adds
    # FockGen(Zero,D,P), [Kappa,F0], and finally PT2 WLag in Out_Pt2.
    # `AddGrad` is part of the antisymmetrized `RInt_Generic` Fock object,
    # but not of the `Make_Conn` object actually used for FockOcc.
    total = np.asarray(w_hf + w_comm + w_fockgen + w_rint_candidate + w_ci + w_orb, dtype=np.float64)
    return {
        "hf": w_hf,
        "wlag_pt2": w_hf,
        "make_conn_comm": w_comm,
        "make_conn_rint_candidate": w_rint_candidate,
        "make_conn_fockgen": w_fockgen,
        "rint_addgrad": w_addgrad,
        "ci": w_ci,
        "orb": w_orb,
        "make_conn_surrogate_ci": w_ci,
        "make_conn_surrogate_orb": w_orb,
        "total": total,
    }


def build_molcas_fockocc_pt2_ao(
    *,
    w_hf_ao: np.ndarray,
    w_ci_ao: np.ndarray | None = None,
    w_orb_ao: np.ndarray | None = None,
    w_comm_ao: np.ndarray | None = None,
    w_fockgen_ao: np.ndarray | None = None,
    w_addgrad_ao: np.ndarray | None = None,
    w_rint_candidate_ao: np.ndarray | None = None,
) -> np.ndarray:
    """Build the PT2 correction to Molcas `FockOcc` in AO basis."""
    return np.asarray(
        build_molcas_fockocc_pt2_parts_ao(
            w_hf_ao=w_hf_ao,
            w_ci_ao=w_ci_ao,
            w_orb_ao=w_orb_ao,
            w_comm_ao=w_comm_ao,
            w_fockgen_ao=w_fockgen_ao,
            w_addgrad_ao=w_addgrad_ao,
            w_rint_candidate_ao=w_rint_candidate_ao,
        )["total"],
        dtype=np.float64,
    )


def build_molcas_fockocc_commutator_ao(
    *,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    fock_inactive_mo: np.ndarray | None = None,
    fock_ref_mo: np.ndarray | None = None,
) -> np.ndarray:
    """Build the explicit `[Kappa, F0]` contribution in Molcas `Make_Conn`.

    Molcas adds `Kappa * F0SQMO - F0SQMO * Kappa` before the AO transform and
    then stores the packed symmetric AO form. The unpacked AO analogue is the
    symmetric part of `C @ [Kappa,F0] @ C^T`.
    """

    C = np.asarray(mo_coeff, dtype=np.float64)
    K = np.asarray(z_orb, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nmo = int(C.shape[1])
    if K.shape != (nmo, nmo):
        raise ValueError(f"z_orb shape mismatch: expected {(nmo, nmo)}, got {K.shape}")
    if fock_ref_mo is not None:
        F0 = np.asarray(fock_ref_mo, dtype=np.float64)
    elif fock_inactive_mo is not None:
        F0 = np.asarray(fock_inactive_mo, dtype=np.float64)
    else:
        raise ValueError("either fock_ref_mo or fock_inactive_mo must be provided")
    if F0.shape != (nmo, nmo):
        raise ValueError(f"reference fock shape mismatch: expected {(nmo, nmo)}, got {F0.shape}")
    comm_mo = np.asarray(K @ F0 - F0 @ K, dtype=np.float64)
    comm_ao = np.asarray(C @ comm_mo @ C.T, dtype=np.float64)
    return symmetrize_molcas_packed_square(ao_square=comm_ao)


def build_molcas_fockgen_dense_c1_raw_mo(
    *,
    eri_mo: np.ndarray,
    fock_inactive_mo: np.ndarray,
    dm1_ci_active_mo: np.ndarray,
    dm2_ci_active_mo: np.ndarray,
    ncore: int,
    ncas: int,
) -> np.ndarray:
    """Dense C1 replay of Molcas `FockGen(Zero,D,P)` raw `Fock`.

    This follows the C1/Cho-MO algebra in `mclr/fockgen.F90` for the object
    returned in the first output argument `Fock`, not `FockOut`.
    """

    g = np.asarray(eri_mo, dtype=np.float64)
    FIMO = np.asarray(fock_inactive_mo, dtype=np.float64)
    dm1 = np.asarray(dm1_ci_active_mo, dtype=np.float64)
    dm2 = np.asarray(dm2_ci_active_mo, dtype=np.float64)
    if g.ndim != 4 or len({int(x) for x in g.shape}) != 1:
        raise ValueError(f"eri_mo must have shape (nmo,nmo,nmo,nmo), got {g.shape}")
    nmo = int(g.shape[0])
    if FIMO.shape != (nmo, nmo):
        raise ValueError(f"fock_inactive_mo shape mismatch: expected {(nmo, nmo)}, got {FIMO.shape}")
    ncore_i = int(ncore)
    ncas_i = int(ncas)
    act = slice(ncore_i, ncore_i + ncas_i)
    if dm1.shape != (ncas_i, ncas_i):
        raise ValueError(f"dm1_ci_active_mo shape mismatch: expected {(ncas_i, ncas_i)}, got {dm1.shape}")
    if dm2.shape != (ncas_i, ncas_i, ncas_i, ncas_i):
        raise ValueError(
            f"dm2_ci_active_mo shape mismatch: expected {(ncas_i, ncas_i, ncas_i, ncas_i)}, got {dm2.shape}"
        )

    out = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore_i > 0:
        # Coulomb core-column branch: 2 * (pk|ji) D_ij
        out[:, :ncore_i] += np.asarray(
            2.0 * np.einsum("pkij,ij->pk", g[:, :ncore_i, act, act], dm1, optimize=True),
            dtype=np.float64,
        )
        # Exchange core-column branch: -(pk|ij) D_kj with active k,j.
        out[:, :ncore_i] -= np.asarray(
            np.einsum("pkij,kj->pi", g[:, act, :ncore_i, act], dm1, optimize=True),
            dtype=np.float64,
        )

    # Q = (pj|kl) P, with the Molcas `CreQADD` index order P(j,i,k,l).
    q_act = np.asarray(
        np.einsum("pjkl,jikl->pi", g[:, act, act, act], dm2, optimize=True),
        dtype=np.float64,
    )
    out[:, act] += q_act

    # Common active-column dressing: F(:,a) += sum_b FIMO(:,b) D_{ba}.
    out[:, act] += np.asarray(FIMO[:, act] @ dm1, dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def build_molcas_fockocc_reference_dense_c1_raw_mo(
    *,
    eri_mo: np.ndarray,
    fock_inactive_mo: np.ndarray,
    fock_active_mo: np.ndarray,
    dm1_ref_active_mo: np.ndarray,
    dm2_ref_active_mo: np.ndarray,
    ncore: int,
    ncas: int,
) -> np.ndarray:
    """Dense C1 replay of Molcas reference `FOCK`/`FOCKOC` raw MO matrix."""

    g = np.asarray(eri_mo, dtype=np.float64)
    FI = np.asarray(fock_inactive_mo, dtype=np.float64)
    FA = np.asarray(fock_active_mo, dtype=np.float64)
    dm1 = np.asarray(dm1_ref_active_mo, dtype=np.float64)
    dm2 = np.asarray(dm2_ref_active_mo, dtype=np.float64)
    if g.ndim != 4 or len({int(x) for x in g.shape}) != 1:
        raise ValueError(f"eri_mo must have shape (nmo,nmo,nmo,nmo), got {g.shape}")
    nmo = int(g.shape[0])
    if FI.shape != (nmo, nmo):
        raise ValueError(f"fock_inactive_mo shape mismatch: expected {(nmo, nmo)}, got {FI.shape}")
    if FA.shape != (nmo, nmo):
        raise ValueError(f"fock_active_mo shape mismatch: expected {(nmo, nmo)}, got {FA.shape}")
    ncore_i = int(ncore)
    ncas_i = int(ncas)
    act = slice(ncore_i, ncore_i + ncas_i)
    if dm1.shape != (ncas_i, ncas_i):
        raise ValueError(f"dm1_ref_active_mo shape mismatch: expected {(ncas_i, ncas_i)}, got {dm1.shape}")
    if dm2.shape != (ncas_i, ncas_i, ncas_i, ncas_i):
        raise ValueError(
            f"dm2_ref_active_mo shape mismatch: expected {(ncas_i, ncas_i, ncas_i, ncas_i)}, got {dm2.shape}"
        )

    out = np.zeros((nmo, nmo), dtype=np.float64)
    FP = np.asarray(FI + FA, dtype=np.float64)
    if ncore_i > 0:
        out[:, :ncore_i] = np.asarray(2.0 * FP[:, :ncore_i], dtype=np.float64)
    q_act = np.asarray(
        np.einsum("mjkl,ijkl->mi", g[:, act, act, act], dm2, optimize=True),
        dtype=np.float64,
    )
    out[:, act] = np.asarray(q_act + FI[:, act] @ dm1, dtype=np.float64)
    return np.asarray(out, dtype=np.float64)


def build_molcas_fockocc_reference_ao(
    *,
    mo_coeff: np.ndarray,
    fock_ref_mo: np.ndarray,
) -> np.ndarray:
    """AO packed-square analogue of Molcas reference `FOCKOC`."""

    C = np.asarray(mo_coeff, dtype=np.float64)
    F = np.asarray(fock_ref_mo, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nmo = int(C.shape[1])
    if F.shape != (nmo, nmo):
        raise ValueError(f"fock_ref_mo shape mismatch: expected {(nmo, nmo)}, got {F.shape}")
    ao = np.asarray(C @ F @ C.T, dtype=np.float64)
    return symmetrize_molcas_packed_square(ao_square=ao)


def build_molcas_fockocc_fockgen_ao(
    *,
    mo_coeff: np.ndarray,
    gfock_mo: np.ndarray | None = None,
    eri_mo: np.ndarray | None = None,
    fock_inactive_mo: np.ndarray | None = None,
    dm1_ci_active_mo: np.ndarray | None = None,
    dm2_ci_active_mo: np.ndarray | None = None,
    ncore: int | None = None,
    ncas: int | None = None,
) -> np.ndarray:
    """Build the Molcas `FockGen(Zero,D,P)` contribution in AO square form."""

    C = np.asarray(mo_coeff, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nmo = int(C.shape[1])
    G: np.ndarray
    if (
        eri_mo is not None
        and fock_inactive_mo is not None
        and dm1_ci_active_mo is not None
        and dm2_ci_active_mo is not None
        and ncore is not None
        and ncas is not None
    ):
        G = build_molcas_fockgen_dense_c1_raw_mo(
            eri_mo=np.asarray(eri_mo, dtype=np.float64),
            fock_inactive_mo=np.asarray(fock_inactive_mo, dtype=np.float64),
            dm1_ci_active_mo=np.asarray(dm1_ci_active_mo, dtype=np.float64),
            dm2_ci_active_mo=np.asarray(dm2_ci_active_mo, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
        )
    else:
        if gfock_mo is None:
            raise ValueError("either gfock_mo or dense C1 FockGen inputs must be provided")
        G = np.asarray(gfock_mo, dtype=np.float64)
        if G.shape != (nmo, nmo):
            raise ValueError(f"gfock_mo shape mismatch: expected {(nmo, nmo)}, got {G.shape}")
    out = np.asarray(C @ G @ C.T, dtype=np.float64)
    return symmetrize_molcas_packed_square(ao_square=out)


def build_molcas_fockocc_addgrad_ao(
    *,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    gfock_ref_mo: np.ndarray,
) -> np.ndarray:
    """Build the Molcas `AddGrad` contribution used inside `RInt_Generic`."""

    C = np.asarray(mo_coeff, dtype=np.float64)
    K = np.asarray(z_orb, dtype=np.float64)
    G = np.asarray(gfock_ref_mo, dtype=np.float64)
    if C.ndim != 2:
        raise ValueError("mo_coeff must be 2D")
    nmo = int(C.shape[1])
    if K.shape != (nmo, nmo):
        raise ValueError(f"z_orb shape mismatch: expected {(nmo, nmo)}, got {K.shape}")
    if G.shape != (nmo, nmo):
        raise ValueError(f"gfock_ref_mo shape mismatch: expected {(nmo, nmo)}, got {G.shape}")
    t = np.asarray(G - G.T, dtype=np.float64)
    addgrad_mo = np.asarray(-(K.T @ t - t @ K.T), dtype=np.float64)
    addgrad_ao = np.asarray(C @ addgrad_mo @ C.T, dtype=np.float64)
    return np.asarray(0.5 * (addgrad_ao + addgrad_ao.T), dtype=np.float64)


def build_molcas_fockocc_target_diagnostics(
    *,
    fockocc_molcas_ao: np.ndarray,
    fockocc_ref_ao: np.ndarray,
    fockocc_pt2_ao: np.ndarray,
    fockocc_parts: dict[str, np.ndarray],
) -> dict[str, np.ndarray | float]:
    """Split Molcas `FockOcc` into exact known pieces and unresolved remainder."""

    mol_focc = np.asarray(fockocc_molcas_ao, dtype=np.float64)
    ref = np.asarray(fockocc_ref_ao, dtype=np.float64)
    pt2 = np.asarray(fockocc_pt2_ao, dtype=np.float64)
    if mol_focc.shape != ref.shape or pt2.shape != ref.shape:
        raise ValueError(
            f"FockOcc shape mismatch: mol={mol_focc.shape} ref={ref.shape} pt2={pt2.shape}"
        )
    zero = np.zeros_like(ref, dtype=np.float64)
    known_exact = np.asarray(
        np.asarray(fockocc_parts.get("wlag_pt2", zero), dtype=np.float64)
        + np.asarray(fockocc_parts.get("make_conn_comm", zero), dtype=np.float64)
        + np.asarray(fockocc_parts.get("make_conn_fockgen", zero), dtype=np.float64),
        dtype=np.float64,
    )
    rint_candidate = np.asarray(fockocc_parts.get("make_conn_rint_candidate", zero), dtype=np.float64)
    pt2_target = np.asarray(mol_focc - ref, dtype=np.float64)
    rint_target = np.asarray(pt2_target - known_exact, dtype=np.float64)
    unresolved_asuka = np.asarray(pt2 - known_exact, dtype=np.float64)
    rint_delta = np.asarray(unresolved_asuka - rint_target, dtype=np.float64)
    rint_candidate_delta = np.asarray(rint_candidate - rint_target, dtype=np.float64)
    return {
        "pt2_target_ao": np.asarray(pt2_target, dtype=np.float64),
        "known_exact_ao": np.asarray(known_exact, dtype=np.float64),
        "rint_target_ao": np.asarray(rint_target, dtype=np.float64),
        "rint_candidate_ao": np.asarray(rint_candidate, dtype=np.float64),
        "unresolved_asuka_ao": np.asarray(unresolved_asuka, dtype=np.float64),
        "rint_delta_ao": np.asarray(rint_delta, dtype=np.float64),
        "rint_candidate_delta_ao": np.asarray(rint_candidate_delta, dtype=np.float64),
        "rint_delta_norm": float(np.linalg.norm(rint_delta)),
        "rint_delta_max_abs": float(np.max(np.abs(rint_delta))),
        "rint_candidate_delta_norm": float(np.linalg.norm(rint_candidate_delta)),
        "rint_candidate_delta_max_abs": float(np.max(np.abs(rint_candidate_delta))),
    }


def build_molcas_r2elint_density_rotations(
    *,
    z_orb: np.ndarray,
    dm1_ref_active_mo: np.ndarray,
    ncore: int,
    ncas: int,
    signa: float = -1.0,
) -> dict[str, np.ndarray]:
    """Build the C1 `DI13/24` and `DA13/24` matrices from Molcas `Read2_2`."""

    K = np.asarray(z_orb, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"z_orb must be square, got {K.shape}")
    nmo = int(K.shape[0])
    ncore_i = int(ncore)
    ncas_i = int(ncas)
    nocc = int(ncore_i + ncas_i)
    dm1 = np.asarray(dm1_ref_active_mo, dtype=np.float64)
    if dm1.shape != (ncas_i, ncas_i):
        raise ValueError(
            f"dm1_ref_active_mo shape mismatch: expected {(ncas_i, ncas_i)}, got {dm1.shape}"
        )
    DI = np.zeros((nmo, nmo), dtype=np.float64)
    if ncore_i > 0:
        DI[:ncore_i, :ncore_i] = 2.0 * np.eye(ncore_i, dtype=np.float64)
    DA = np.zeros((nmo, nmo), dtype=np.float64)
    DA[ncore_i:nocc, ncore_i:nocc] = np.asarray(dm1, dtype=np.float64)
    return {
        "DI": np.asarray(DI, dtype=np.float64),
        "DA": np.asarray(DA, dtype=np.float64),
        "DI13": np.asarray(K.T @ DI, dtype=np.float64),
        "DI24": np.asarray(float(signa) * (K @ DI), dtype=np.float64),
        "DA13": np.asarray(K.T @ DA, dtype=np.float64),
        "DA24": np.asarray(float(signa) * (K @ DA), dtype=np.float64),
    }


def build_molcas_r2elint_dense_c1_terms(
    *,
    eri_mo: np.ndarray,
    z_orb: np.ndarray,
    dm1_ref_active_mo: np.ndarray,
    ncore: int,
    ncas: int,
    fock_inactive_mo: np.ndarray,
    fock_active_mo: np.ndarray,
    signa: float = -1.0,
    fact: float = -1.0,
    singlet: bool = True,
) -> dict[str, np.ndarray]:
    """Direct dense C1 replay of the Molcas `R2ElInt` FockI/FockA loops.

    This ports the `Read2_2` / `R2ElInt` algebra for the C1 case only.
    It does not include the later `Q` assembly from `RInt_Generic`.
    """

    g = np.asarray(eri_mo, dtype=np.float64)
    K = np.asarray(z_orb, dtype=np.float64)
    FIMO = np.asarray(fock_inactive_mo, dtype=np.float64)
    FAMO = np.asarray(fock_active_mo, dtype=np.float64)
    if g.ndim != 4 or len({int(x) for x in g.shape}) != 1:
        raise ValueError(f"eri_mo must have shape (nmo,nmo,nmo,nmo), got {g.shape}")
    nmo = int(g.shape[0])
    if K.shape != (nmo, nmo):
        raise ValueError(f"z_orb shape mismatch: expected {(nmo, nmo)}, got {K.shape}")
    if FIMO.shape != (nmo, nmo):
        raise ValueError(f"fock_inactive_mo shape mismatch: expected {(nmo, nmo)}, got {FIMO.shape}")
    if FAMO.shape != (nmo, nmo):
        raise ValueError(f"fock_active_mo shape mismatch: expected {(nmo, nmo)}, got {FAMO.shape}")

    ncore_i = int(ncore)
    ncas_i = int(ncas)
    nocc = int(ncore_i + ncas_i)
    dens = build_molcas_r2elint_density_rotations(
        z_orb=K,
        dm1_ref_active_mo=np.asarray(dm1_ref_active_mo, dtype=np.float64),
        ncore=ncore_i,
        ncas=ncas_i,
        signa=float(signa),
    )
    DI13 = np.asarray(dens["DI13"], dtype=np.float64)
    DI24 = np.asarray(dens["DI24"], dtype=np.float64)
    DA13 = np.asarray(dens["DA13"], dtype=np.float64)
    DA24 = np.asarray(dens["DA24"], dtype=np.float64)

    FockI_coul = np.zeros((nmo, nocc), dtype=np.float64)
    FockA_coul = np.zeros((nmo, nocc), dtype=np.float64)
    # Molcas Read2_2 uses a LOCAL variable `Sgn = One` (always +1) here,
    # NOT the `Signa` parameter.  The coefficient is -Fact*Sgn*Half.
    sgn = 1.0
    alpha_coul = float(-fact * sgn * 0.5)
    for i in range(nocc):
        for j in range(i + 1):
            coul_ij = np.asarray(g[:, :, i, j], dtype=np.float64)  # (l,k)
            FockI_coul[:, j] += alpha_coul * (coul_ij.T @ DI24[:, i])
            FockA_coul[:, j] += alpha_coul * (coul_ij.T @ DA24[:, i])
            if i != j:
                FockI_coul[:, i] += alpha_coul * (coul_ij.T @ DI24[:, j])
                FockA_coul[:, i] += alpha_coul * (coul_ij.T @ DA24[:, j])

    FockI_exch = np.zeros((nmo, nocc), dtype=np.float64)
    FockA_exch = np.zeros((nmo, nocc), dtype=np.float64)
    for j in range(nocc):
        for l in range(nocc):
            exch_jl = np.asarray(g[:, j, :, l], dtype=np.float64)  # (i,k) = (i j | k l)
            if bool(singlet):
                FockI_exch[:, j] += float(fact) * (exch_jl @ DI13[:, l])
                # Molcas uses Fact*Sgn here (Sgn=1), NOT Fact*Signa
                FockI_exch[:, j] += float(fact * sgn) * (exch_jl @ DI24[:, l])
                FockA_exch[:, j] += float(fact) * (exch_jl @ DA13[:, l])
                FockA_exch[:, j] += float(fact * sgn) * (exch_jl @ DA24[:, l])
            FockI_exch[:, l] += float(-fact * 0.5) * (exch_jl @ DI13[:, j])
            FockA_exch[:, l] += float(-fact * 0.5) * (exch_jl @ DA13[:, j])

    FockI_uncontracted = np.asarray(float(signa * fact) * (FIMO @ K) + float(fact) * (K @ FIMO), dtype=np.float64)
    FockA_uncontracted = np.asarray(float(signa * fact) * (FAMO @ K) + float(fact) * (K @ FAMO), dtype=np.float64)

    FockI = np.asarray(FockI_coul + FockI_exch + FockI_uncontracted[:, :nocc], dtype=np.float64)
    FockA = np.asarray(FockA_coul + FockA_exch + FockA_uncontracted[:, :nocc], dtype=np.float64)
    return {
        "FockI_coul": np.asarray(FockI_coul, dtype=np.float64),
        "FockA_coul": np.asarray(FockA_coul, dtype=np.float64),
        "FockI_exch": np.asarray(FockI_exch, dtype=np.float64),
        "FockA_exch": np.asarray(FockA_exch, dtype=np.float64),
        "FockI_uncontracted": np.asarray(FockI_uncontracted[:, :nocc], dtype=np.float64),
        "FockA_uncontracted": np.asarray(FockA_uncontracted[:, :nocc], dtype=np.float64),
        "FockI": np.asarray(FockI, dtype=np.float64),
        "FockA": np.asarray(FockA, dtype=np.float64),
    }


def build_molcas_creq_dense_c1(
    *,
    rint_mo: np.ndarray,
    dm2_ref_active_mo: np.ndarray,
) -> np.ndarray:
    """Dense C1 analogue of Molcas `CreQ`.

    Parameters
    ----------
    rint_mo
        One-index transformed integral tensor `(p,j,k,l)` with `p` general and
        `(j,k,l)` in the active space.
    dm2_ref_active_mo
        Full active-space 2-RDM `(i,j,k,l)` used by `RInt_Generic`.
    """

    r = np.asarray(rint_mo, dtype=np.float64)
    d2 = np.asarray(dm2_ref_active_mo, dtype=np.float64)
    if r.ndim != 4:
        raise ValueError(f"rint_mo must have shape (nmo,ncas,ncas,ncas), got {r.shape}")
    if d2.ndim != 4 or len({int(x) for x in d2.shape}) != 1:
        raise ValueError(f"dm2_ref_active_mo must have shape (ncas,ncas,ncas,ncas), got {d2.shape}")
    ncas = int(d2.shape[0])
    if r.shape[1:] != (ncas, ncas, ncas):
        raise ValueError(
            f"rint_mo active shape mismatch: expected {(ncas, ncas, ncas)}, got {r.shape[1:]}"
        )
    return np.asarray(np.einsum("pjkl,ijkl->pi", r, d2, optimize=True), dtype=np.float64)


def build_molcas_creq2_dense_c1(
    *,
    eri_mo: np.ndarray,
    dm2_ref_active_mo: np.ndarray,
    ncore: int,
    ncas: int,
) -> np.ndarray:
    """Dense C1 analogue of Molcas `CreQ2` from unrotated MO ERIs."""

    g = np.asarray(eri_mo, dtype=np.float64)
    if g.ndim != 4 or len({int(x) for x in g.shape}) != 1:
        raise ValueError(f"eri_mo must have shape (nmo,nmo,nmo,nmo), got {g.shape}")
    nmo = int(g.shape[0])
    act = slice(int(ncore), int(ncore + ncas))
    rint = np.asarray(g[:, act, act, act], dtype=np.float64)
    if rint.shape != (nmo, int(ncas), int(ncas), int(ncas)):
        raise ValueError(f"unexpected CreQ2 tensor shape: {rint.shape}")
    return build_molcas_creq_dense_c1(
        rint_mo=rint,
        dm2_ref_active_mo=np.asarray(dm2_ref_active_mo, dtype=np.float64),
    )


def build_molcas_rint_generic_tensor_dense_c1(
    *,
    eri_mo: np.ndarray,
    z_orb: np.ndarray,
    ncore: int,
    ncas: int,
    signa: float = -1.0,
) -> np.ndarray:
    """Dense C1 closed-form analogue of the `RInt_Generic` one-index transform.

    This is the direct tensor form of the Molcas `Read2_2`/`CreQ` input:

      R(p,j,k,l) =
          sum_o K(p,o) (o,j|k,l)
        + sgn * sum_o K(j,o) (p,o|k,l)
        +       sum_o K(k,o) (p,j|o,l)
        + sgn * sum_o K(l,o) (p,j|k,o)
    """

    g = np.asarray(eri_mo, dtype=np.float64)
    K = np.asarray(z_orb, dtype=np.float64)
    if g.ndim != 4 or len({int(x) for x in g.shape}) != 1:
        raise ValueError(f"eri_mo must have shape (nmo,nmo,nmo,nmo), got {g.shape}")
    nmo = int(g.shape[0])
    if K.shape != (nmo, nmo):
        raise ValueError(f"z_orb shape mismatch: expected {(nmo, nmo)}, got {K.shape}")
    act = slice(int(ncore), int(ncore + ncas))
    Kact = np.asarray(K[act, :], dtype=np.float64)
    term_i = np.asarray(np.einsum("po,ojkl->pjkl", K, g[:, act, act, act], optimize=True), dtype=np.float64)
    term_j = np.asarray(float(signa) * np.einsum("jo,pokl->pjkl", Kact, g[:, :, act, act], optimize=True), dtype=np.float64)
    term_k = np.asarray(np.einsum("ko,pjol->pjkl", Kact, g[:, act, :, act], optimize=True), dtype=np.float64)
    term_l = np.asarray(float(signa) * np.einsum("lo,pjko->pjkl", Kact, g[:, act, act, :], optimize=True), dtype=np.float64)
    return np.asarray(term_i + term_j + term_k + term_l, dtype=np.float64)


def build_molcas_rint_generic_preaddgrad_candidate_ao(
    *,
    mo_coeff: np.ndarray,
    eri_mo: np.ndarray,
    z_orb: np.ndarray,
    dm1_ref_active_mo: np.ndarray,
    dm2_ref_active_mo: np.ndarray,
    ncore: int,
    ncas: int,
    fock_inactive_mo: np.ndarray,
    fock_active_mo: np.ndarray,
) -> dict[str, np.ndarray]:
    """Direct dense C1 candidate for the Molcas `RInt_Generic` pre-AddGrad branch."""

    C = np.asarray(mo_coeff, dtype=np.float64)
    dm1 = np.asarray(dm1_ref_active_mo, dtype=np.float64)
    nmo = int(C.shape[1])
    nocc = int(ncore + ncas)
    r2 = build_molcas_r2elint_dense_c1_terms(
        eri_mo=np.asarray(eri_mo, dtype=np.float64),
        z_orb=np.asarray(z_orb, dtype=np.float64),
        dm1_ref_active_mo=dm1,
        ncore=int(ncore),
        ncas=int(ncas),
        fock_inactive_mo=np.asarray(fock_inactive_mo, dtype=np.float64),
        fock_active_mo=np.asarray(fock_active_mo, dtype=np.float64),
        signa=-1.0,
        fact=-1.0,
        singlet=True,
    )
    rint_mo = build_molcas_rint_generic_tensor_dense_c1(
        eri_mo=np.asarray(eri_mo, dtype=np.float64),
        z_orb=np.asarray(z_orb, dtype=np.float64),
        ncore=int(ncore),
        ncas=int(ncas),
        signa=-1.0,
    )
    q_mo = build_molcas_creq_dense_c1(
        rint_mo=rint_mo,
        dm2_ref_active_mo=np.asarray(dm2_ref_active_mo, dtype=np.float64),
    )

    Fock_base = np.zeros((nmo, nmo), dtype=np.float64)
    Fock_base[:, :ncore] = 2.0 * (r2["FockI"][:, :ncore] + r2["FockA"][:, :ncore])
    Fock_base[:, ncore:nocc] = np.asarray(
        q_mo + r2["FockI"][:, ncore:nocc] @ dm1.T,
        dtype=np.float64,
    )
    # In Molcas `Make_Conn`, the output argument `F` passed into
    # `RInt_Generic` receives `FockA = 2 * Fock_pre` before the later
    # `DGESUB`/`AddGrad` work on the separate `Fock` argument. `Make_Conn`
    # then keeps `0.5 * F` as its RInt contribution. So the object folded into
    # `FockOcc` is the pre-AddGrad `Fock_pre`, not its antisymmetric residual.
    make_conn_rint_mo = np.asarray(Fock_base, dtype=np.float64)
    FockA_pre = np.asarray(2.0 * make_conn_rint_mo, dtype=np.float64)
    make_conn_rint_ao = transform_molcas_mo_square_to_ao(
        mo_coeff=C,
        mo_square=make_conn_rint_mo,
    )
    make_conn_rint_packed_ao = symmetrize_molcas_packed_square(
        ao_square=make_conn_rint_ao,
    )
    FockA_pre_ao = transform_molcas_mo_square_to_ao(
        mo_coeff=C,
        mo_square=FockA_pre,
    )
    return {
        **r2,
        "rint_mo": np.asarray(rint_mo, dtype=np.float64),
        "Q_mo": np.asarray(q_mo, dtype=np.float64),
        "F_base_mo": np.asarray(Fock_base, dtype=np.float64),
        "F_pre_mo": np.asarray(make_conn_rint_mo, dtype=np.float64),
        "MakeConn_rint_mo": np.asarray(make_conn_rint_mo, dtype=np.float64),
        "MakeConn_rint_ao": np.asarray(make_conn_rint_ao, dtype=np.float64),
        "MakeConn_rint_packed_ao": np.asarray(make_conn_rint_packed_ao, dtype=np.float64),
        "FockA_pre_mo": np.asarray(FockA_pre, dtype=np.float64),
        "FockA_pre_ao": np.asarray(FockA_pre_ao, dtype=np.float64),
    }


def build_molcas_coulomb_ao4(
    *,
    d0_inactive_ao: np.ndarray,
    d0_var_pt2_ao: np.ndarray,
    d0_active_ao: np.ndarray,
    d0_dlao_ao: np.ndarray,
) -> np.ndarray:
    """Exact dense analogue of the Molcas SA/CASPT2 Coulomb lane.

    In `PGet1_RI3` / `PGet2_RI3` the SA branch forms:

      J(D0_1, D0_2) + J(D0_3, D0_4)
    """

    return np.asarray(
        _bar_cross_j(d0_inactive_ao, d0_var_pt2_ao)
        + _bar_cross_j(d0_active_ao, d0_dlao_ao),
        dtype=np.float64,
    )


def build_molcas_exchange_cross_candidate_ao4(
    *,
    d0_inactive_ao: np.ndarray,
    d0_var_pt2_ao: np.ndarray,
    d0_active_ao: np.ndarray,
    d0_dlao_ao: np.ndarray,
) -> np.ndarray:
    """Cross-exchange candidate for the Molcas SA/CASPT2 `BklK` lane.

    The true Molcas `BklK` uses signed square-root orbital factorizations of
    `(D0_1, D0_2)` and `(D0_3, D0_4)`. This helper keeps the current dense
    implementation honest by exposing the raw matrix-space cross-K candidate
    separately from the exact Coulomb and BTAMP lanes.
    """

    return np.asarray(
        -0.5
        * (
            _bar_cross_k(d0_inactive_ao, d0_var_pt2_ao)
            + _bar_cross_k(d0_active_ao, d0_dlao_ao)
        ),
        dtype=np.float64,
    )


def build_dense_molcas_term_manifest(
    *,
    exchange_mode: str,
    thpkl_mode: str,
    btamp_mode: str,
    one_body_mode: str,
    fockgen_exact: bool,
    addgrad_exact: bool,
) -> dict[str, dict[str, Any]]:
    """Return a term-by-term provenance ledger for the dense Molcas lane."""

    exch_mode = str(exchange_mode).strip().lower()
    th_mode = str(thpkl_mode).strip().lower()
    bt_mode = str(btamp_mode).strip().lower()
    one_body_mode_n = str(one_body_mode).strip().lower()
    one_body_exact = bool(one_body_mode_n == "out_pt2_density_replay")
    exchange_formula = (
        "-1/2 * (K(D0_1,D0_2) + K(D0_3,D0_4)) dense AO4 placeholder"
        if exch_mode == "raw_cross_k_candidate"
        else "-1/2 * (D0_1 X_Q D0_2 + D0_2 X_Q D0_1 + D0_3 X_Q D0_4 + D0_4 X_Q D0_3)"
    )
    exchange_builder = (
        "build_molcas_exchange_cross_candidate_ao4"
        if exch_mode == "raw_cross_k_candidate"
        else "build_molcas_exchange_barx_cross_candidate"
    )
    exchange_storage = "ordered AO4" if exch_mode == "raw_cross_k_candidate" else "DF bar_X(mu,nu,Q)"
    th_formula = {
        "provided": "caller-supplied Thpkl contraction in ordered AO4 storage",
        "c1_sa_candidate": "C Z1 C^T + L Z2 C^T + 2 C Z3 C^T + C Z2 L^T in DF bar_X storage",
        "absent": "missing Thpkl active lane",
    }.get(th_mode, "missing Thpkl active lane")
    th_builder = {
        "provided": "external",
        "c1_sa_candidate": "build_c1_sa_active_thpkl_candidate_bar",
        "absent": "none",
    }.get(th_mode, "none")
    th_storage = {
        "provided": "ordered AO4",
        "c1_sa_candidate": "DF bar_X(mu,nu,Q)",
        "absent": "none",
    }.get(th_mode, "none")
    th_exact = bool(th_mode == "provided")
    bt_formula = (
        "G_toc(j,l,i,k) from CASPT2_BTAMP / PGet3 dense reconstruction"
        if bt_mode != "absent"
        else "missing BTAMP lane"
    )
    bt_builder = "build_gtoc_dense_term_arrays" if bt_mode != "absent" else "none"
    bt_storage = "ordered AO4" if bt_mode != "absent" else "none"
    bt_exact = bool(bt_mode != "absent")
    return {
        "d0_inactive": _term_info(
            formula="2 * I_core in MO, transformed to AO inactive density",
            molcas_source="integral_util/prepp.F90:Get_D1I -> D0(:,1)",
            storage="AO square",
            builder="build_dense_molcas_ss_assembly",
            exact=True,
        ),
        "d0_var_ref": _term_info(
            formula="D1ao(ref) - 1/2 * D0(:,1)",
            molcas_source="integral_util/prepp.F90:D0(:,2)",
            storage="AO square",
            builder="build_dense_molcas_ss_assembly",
            exact=True,
        ),
        "d0_var_pt2": _term_info(
            formula="D1aoVar(pt2) from Out_Pt2 density replay" if one_body_exact else "D1aoVar(pt2) surrogate from additive AO pieces",
            molcas_source="mclr/out_pt2.F90 + integral_util/prepp.F90:DVar(:,1)",
            storage="AO square",
            builder="build_molcas_d1aovar_replay_parts" if one_body_exact else "build_molcas_d1aovar_pt2_parts_ao",
            exact=one_body_exact,
        ),
        "d0_active": _term_info(
            formula="active reference density from D1av/G1q",
            molcas_source="integral_util/prepp.F90:Get_D1A -> D0(:,3)",
            storage="AO square",
            builder="build_dense_molcas_ss_assembly",
            exact=True,
        ),
        "d0_dlao": _term_info(
            formula="C * (D_core K^T - K^T D_core + D_PT2) * C^T",
            molcas_source="mclr/out_pt2.F90:OITD(...,act=.false.) -> DLAO",
            storage="AO square",
            builder="build_molcas_dlao_candidate_ao",
            exact=True,
        ),
        "d1aovar_pt2_total": _term_info(
            formula="Out_Pt2 D_K replay followed by AO transform" if one_body_exact else "additive AO surrogate for D1aoVar(pt2); exact Out_Pt2 replay still missing",
            molcas_source="mclr/out_pt2.F90:D_K PT2 correction before NatOrb_MCLR/dmat_MCLR",
            storage="AO square",
            builder="build_molcas_d1aovar_replay_parts" if one_body_exact else "build_molcas_d1aovar_pt2_parts_ao",
            exact=one_body_exact,
        ),
        "fockocc_pt2_total": _term_info(
            formula="additive AO surrogate for FockOcc(pt2); exact Make_Conn replay still missing",
            molcas_source="mclr/out_pt2.F90:Make_Conn + WLag -> FockOcc correction",
            storage="AO square",
            builder="build_molcas_fockocc_pt2_parts_ao",
            exact=False,
        ),
        "fockocc_wlag_pt2": _term_info(
            formula="PT2 WLag AO object added in Out_Pt2 after Make_Conn",
            molcas_source="mclr/out_pt2.F90:WLag read from PT2_Lag",
            storage="AO square",
            builder="lagrangians['wlag_ao'] / build_molcas_fockocc_pt2_parts_ao",
            exact=True,
        ),
        "fockocc_make_conn_comm": _term_info(
            formula="sym(C * (Kappa*F0 - F0*Kappa) * C^T)",
            molcas_source="mclr/make_conn.F90:[kappa,F0SQMO] term",
            storage="AO square",
            builder="build_molcas_fockocc_commutator_ao",
            exact=True,
        ),
        "fockocc_make_conn_rint": _term_info(
            formula="0.5 * Rint_generic(Kappa)",
            molcas_source="mclr/make_conn.F90:Rint_generic branch",
            storage="AO square",
            builder="missing",
            exact=False,
        ),
        "fockocc_make_conn_rint_candidate": _term_info(
            formula="direct dense C1 replay of Read2_2 / R2ElInt / RInt_Generic pre-AddGrad branch",
            molcas_source="mclr/read2_2.F90 + mclr/r2elint.F90 + mclr/rint_generic.F90 (C1 dense candidate)",
            storage="AO square",
            builder="build_molcas_rint_generic_preaddgrad_candidate_ao",
            exact=False,
        ),
        "fockocc_make_conn_fockgen": _term_info(
            formula="FockGen(0, D_CI, P_CI)",
            molcas_source="mclr/make_conn.F90:FockGen(Zero,D,P)",
            storage="AO square",
            builder="build_molcas_fockocc_fockgen_ao" if bool(fockgen_exact) else "missing",
            exact=bool(fockgen_exact),
        ),
        "fockocc_rint_addgrad": _term_info(
            formula="-(K^T T - T K^T), T = F0SQMO - F0SQMO^T",
            molcas_source="mclr/addgrad.F90 inside mclr/RInt_Generic",
            storage="AO square",
            builder="build_molcas_fockocc_addgrad_ao" if bool(addgrad_exact) else "missing",
            exact=bool(addgrad_exact),
        ),
        "ao4_coulomb": _term_info(
            formula="J(D0_1,D0_2) + J(D0_3,D0_4) using exact D0(:,1:4)" if one_body_exact else "J(D0_1,D0_2) + J(D0_3,D0_4) using current D0(:,2/4) inputs",
            molcas_source="ri_util/pget1_ri3.F90:V_k/DSO Coulomb lane",
            storage="ordered AO4",
            builder="build_molcas_coulomb_ao4",
            exact=one_body_exact,
        ),
        "ao4_exchange": _term_info(
            formula=exchange_formula,
            molcas_source="ri_util/pget1_ri3.F90:BklK lane",
            storage=exchange_storage,
            builder=exchange_builder,
            exact=False,
        ),
        "ao4_thpkl": _term_info(
            formula=th_formula,
            molcas_source="ri_util/contract_zpk_tpxy.F90 + ri_util/pget1_ri3.F90:Thpkl lane",
            storage=th_storage,
            builder=th_builder,
            exact=th_exact,
        ),
        "ao4_btamp": _term_info(
            formula=bt_formula,
            molcas_source="alaska_util/caspt2_btamp.F90 / PGet3",
            storage=bt_storage,
            builder=bt_builder,
            exact=bt_exact,
        ),
    }


@dataclass(frozen=True)
class DenseMolcasSSAssembly:
    d0_inactive_ao: np.ndarray
    d0_var_ref_ao: np.ndarray
    d0_var_total_ao: np.ndarray
    d0_var_pt2_ao: np.ndarray
    d0_active_ao: np.ndarray
    d0_dlao_ao: np.ndarray
    d1aovar_pt2_ao: np.ndarray
    d1aovar_total_ao: np.ndarray
    fockocc_pt2_ao: np.ndarray
    fockocc_ref_ao: np.ndarray
    fockocc_total_ao: np.ndarray
    d1aovar_parts: dict[str, np.ndarray]
    fockocc_parts: dict[str, np.ndarray]
    ao4_ref_candidate: np.ndarray
    ao4_total_raw_candidate: np.ndarray
    ao4_coulomb: np.ndarray
    ao4_exchange_cross_candidate: np.ndarray
    ao4_thpkl: np.ndarray
    ao4_btamp: np.ndarray
    ao4_total_candidate: np.ndarray
    term_manifest: dict[str, dict[str, Any]]
    all_terms_exact: bool
    non_exact_terms: tuple[str, ...]
    meta: dict[str, Any]


def build_dense_molcas_ss_assembly(
    *,
    d0_inactive_ao: np.ndarray,
    d0_active_ao: np.ndarray,
    d_ref_1e_ao: np.ndarray,
    dpt2_ao: np.ndarray,
    dpt2c_ao: np.ndarray | None,
    d_ci_1e_ao: np.ndarray | None,
    d_orb_1e_ao: np.ndarray | None,
    w_ref_ao: np.ndarray,
    w_hf_ao: np.ndarray,
    w_ci_ao: np.ndarray | None,
    w_orb_ao: np.ndarray | None,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    dpt2_mo: np.ndarray,
    eri_mo: np.ndarray | None = None,
    fock_inactive_mo: np.ndarray | None = None,
    fock_active_mo: np.ndarray | None = None,
    gfock_ci_mo: np.ndarray | None = None,
    gfock_ref_mo: np.ndarray | None = None,
    fockocc_rint_candidate_ao: np.ndarray | None = None,
    ncore: int,
    dpt2c_mo: np.ndarray | None = None,
    B_ao: np.ndarray | None = None,
    df_metric_chol: np.ndarray | None = None,
    ncas: int | None = None,
    dm1_ref_active_mo: np.ndarray | None = None,
    dm1_ci_active_mo: np.ndarray | None = None,
    t2_mo: np.ndarray | None = None,
    btamp_term: str = "total",
    dm2_p2mo: np.ndarray | None = None,
    dm2_d2av: np.ndarray | None = None,
    dm2_plmo: np.ndarray | None = None,
    thpkl_mode: str = "off",
    exchange_mode: str = "barx_cross_candidate",
    ao4_thpkl: np.ndarray | None = None,
) -> DenseMolcasSSAssembly:
    """Build dense Molcas-shaped PT2 objects for the SS gradient."""

    d0_1 = _asnumpy_f64(d0_inactive_ao)
    d0_3 = _asnumpy_f64(d0_active_ao)
    d_ref_1e = _asnumpy_f64(d_ref_1e_ao)
    one_body_mode = "additive_ao_surrogate"
    d1_parts: dict[str, np.ndarray]
    if dm1_ref_active_mo is not None and ncas is not None:
        replay = build_molcas_d1aovar_replay_parts(
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            z_orb=np.asarray(z_orb, dtype=np.float64),
            dm1_ref_active_mo=np.asarray(dm1_ref_active_mo, dtype=np.float64),
            dm1_ci_active_mo=np.asarray(dm1_ci_active_mo, dtype=np.float64) if dm1_ci_active_mo is not None else None,
            dpt2_mo=np.asarray(dpt2_mo, dtype=np.float64),
            dpt2c_mo=np.asarray(dpt2c_mo, dtype=np.float64) if dpt2c_mo is not None else None,
            ncore=int(ncore),
            ncas=int(ncas),
        )
        d_ref_1e = np.asarray(replay["ref_ao"], dtype=np.float64)
        d1aovar_pt2 = np.asarray(replay["pt2_ao"], dtype=np.float64)
        d1aovar_total = np.asarray(replay["total_ao"], dtype=np.float64)
        d1_parts = {
            "dpt2": np.asarray(replay["dpt2_ao"], dtype=np.float64),
            "dpt2c_sym_quarter": np.asarray(replay["dpt2c_sym_quarter_ao"], dtype=np.float64),
            "ci": np.asarray(replay["ci_ao"], dtype=np.float64),
            "orb": np.asarray(replay["oitd_ao"], dtype=np.float64),
            "total": np.asarray(replay["pt2_ao"], dtype=np.float64),
        }
        one_body_mode = "out_pt2_density_replay"
    else:
        d1_parts = build_molcas_d1aovar_pt2_parts_ao(
            dpt2_ao=dpt2_ao,
            dpt2c_ao=dpt2c_ao,
            d_ci_1e_ao=d_ci_1e_ao,
            d_orb_1e_ao=d_orb_1e_ao,
        )
        d1aovar_pt2 = np.asarray(d1_parts["total"], dtype=np.float64)
        d1aovar_total = np.asarray(d_ref_1e + d1aovar_pt2, dtype=np.float64)
    d0_2_ref = np.asarray(d_ref_1e - 0.5 * d0_1, dtype=np.float64)
    d0_2_total = np.asarray(d1aovar_total - 0.5 * d0_1, dtype=np.float64)
    d0_2 = np.asarray(d0_2_total - d0_2_ref, dtype=np.float64)
    d0_4 = build_molcas_dlao_candidate_ao(
        mo_coeff=mo_coeff,
        z_orb=z_orb,
        dpt2_mo=dpt2_mo,
        ncore=int(ncore),
        ncas=int(ncas or 0),
    )
    fock_ref_mo_exact = None
    fockocc_ref = np.asarray(_asnumpy_f64(w_ref_ao), dtype=np.float64)
    if (
        eri_mo is not None
        and fock_inactive_mo is not None
        and fock_active_mo is not None
        and dm1_ref_active_mo is not None
        and dm2_p2mo is not None
        and ncas is not None
    ):
        fock_ref_mo_exact = build_molcas_fockocc_reference_dense_c1_raw_mo(
            eri_mo=np.asarray(eri_mo, dtype=np.float64),
            fock_inactive_mo=np.asarray(fock_inactive_mo, dtype=np.float64),
            fock_active_mo=np.asarray(fock_active_mo, dtype=np.float64),
            dm1_ref_active_mo=np.asarray(dm1_ref_active_mo, dtype=np.float64),
            dm2_ref_active_mo=np.asarray(dm2_p2mo, dtype=np.float64),
            ncore=int(ncore),
            ncas=int(ncas),
        )
        fockocc_ref = build_molcas_fockocc_reference_ao(
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            fock_ref_mo=np.asarray(fock_ref_mo_exact, dtype=np.float64),
        )
    w_comm = (
        build_molcas_fockocc_commutator_ao(
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            z_orb=np.asarray(z_orb, dtype=np.float64),
            fock_inactive_mo=np.asarray(fock_inactive_mo, dtype=np.float64) if fock_inactive_mo is not None else None,
            fock_ref_mo=np.asarray(fock_ref_mo_exact, dtype=np.float64) if fock_ref_mo_exact is not None else None,
        )
        if (fock_inactive_mo is not None or fock_ref_mo_exact is not None)
        else np.zeros_like(fockocc_ref, dtype=np.float64)
    )
    w_fockgen = (
        build_molcas_fockocc_fockgen_ao(
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            gfock_mo=np.asarray(gfock_ci_mo, dtype=np.float64) if gfock_ci_mo is not None else None,
            eri_mo=np.asarray(eri_mo, dtype=np.float64) if eri_mo is not None else None,
            fock_inactive_mo=np.asarray(fock_inactive_mo, dtype=np.float64) if fock_inactive_mo is not None else None,
            dm1_ci_active_mo=np.asarray(dm1_ci_active_mo, dtype=np.float64) if dm1_ci_active_mo is not None else None,
            dm2_ci_active_mo=np.asarray(dm2_plmo, dtype=np.float64) if dm2_plmo is not None else None,
            ncore=int(ncore),
            ncas=int(ncas) if ncas is not None else None,
        )
        if (
            gfock_ci_mo is not None
            or (
                eri_mo is not None
                and fock_inactive_mo is not None
                and dm1_ci_active_mo is not None
                and dm2_plmo is not None
                and ncas is not None
            )
        )
        else np.zeros_like(fockocc_ref, dtype=np.float64)
    )
    w_addgrad = (
        build_molcas_fockocc_addgrad_ao(
            mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
            z_orb=np.asarray(z_orb, dtype=np.float64),
            gfock_ref_mo=np.asarray(gfock_ref_mo, dtype=np.float64),
        )
        if gfock_ref_mo is not None
        else np.zeros_like(fockocc_ref, dtype=np.float64)
    )
    fock_parts = build_molcas_fockocc_pt2_parts_ao(
        w_hf_ao=w_hf_ao,
        w_ci_ao=w_ci_ao,
        w_orb_ao=w_orb_ao,
        w_comm_ao=w_comm,
        w_fockgen_ao=w_fockgen,
        w_addgrad_ao=w_addgrad,
        w_rint_candidate_ao=np.asarray(fockocc_rint_candidate_ao, dtype=np.float64) if fockocc_rint_candidate_ao is not None else None,
    )
    fockocc_pt2 = np.asarray(fock_parts["total"], dtype=np.float64)
    fockocc_total = np.asarray(fockocc_ref + fockocc_pt2, dtype=np.float64)
    _zero_dlao = np.zeros_like(d0_4, dtype=np.float64)
    ao4_ref = np.asarray(
        build_molcas_coulomb_ao4(
            d0_inactive_ao=d0_1,
            d0_var_pt2_ao=d0_2_ref,
            d0_active_ao=d0_3,
            d0_dlao_ao=_zero_dlao,
        )
        + build_molcas_exchange_cross_candidate_ao4(
            d0_inactive_ao=d0_1,
            d0_var_pt2_ao=d0_2_ref,
            d0_active_ao=d0_3,
            d0_dlao_ao=_zero_dlao,
        ),
        dtype=np.float64,
    )
    ao4_total_raw = np.asarray(
        build_molcas_coulomb_ao4(
            d0_inactive_ao=d0_1,
            d0_var_pt2_ao=d0_2_total,
            d0_active_ao=d0_3,
            d0_dlao_ao=d0_4,
        )
        + build_molcas_exchange_cross_candidate_ao4(
            d0_inactive_ao=d0_1,
            d0_var_pt2_ao=d0_2_total,
            d0_active_ao=d0_3,
            d0_dlao_ao=d0_4,
        ),
        dtype=np.float64,
    )
    ao4_coulomb = build_molcas_coulomb_ao4(
        d0_inactive_ao=d0_1,
        d0_var_pt2_ao=d0_2,
        d0_active_ao=d0_3,
        d0_dlao_ao=d0_4,
    )
    exch_mode_n = str(exchange_mode).strip().lower()
    if exch_mode_n not in {"raw_cross_k_candidate", "barx_cross_candidate"}:
        exch_mode_n = "raw_cross_k_candidate"
    if exch_mode_n == "barx_cross_candidate" and B_ao is None:
        exch_mode_n = "raw_cross_k_candidate"
    if exch_mode_n == "raw_cross_k_candidate":
        ao4_exchange = build_molcas_exchange_cross_candidate_ao4(
            d0_inactive_ao=d0_1,
            d0_var_pt2_ao=d0_2,
            d0_active_ao=d0_3,
            d0_dlao_ao=d0_4,
        )
        exchange_barx = None
    else:
        ao4_exchange = np.zeros_like(ao4_coulomb, dtype=np.float64)
        exchange_barx = build_molcas_exchange_barx_cross_candidate(
            B_ao=np.asarray(B_ao, dtype=np.float64),
            df_metric_chol=np.asarray(df_metric_chol, dtype=np.float64) if df_metric_chol is not None else None,
            d0_inactive_ao=np.asarray(d0_1, dtype=np.float64),
            d0_var_pt2_ao=np.asarray(d0_2, dtype=np.float64),
            d0_active_ao=np.asarray(d0_3, dtype=np.float64),
            d0_dlao_ao=np.asarray(d0_4, dtype=np.float64),
        )
    if ao4_thpkl is None:
        th_mode_n = str(thpkl_mode).strip().lower()
        if th_mode_n == "c1_sa_candidate":
            if B_ao is None or ncas is None or dm2_p2mo is None or dm2_d2av is None or dm2_plmo is None:
                raise ValueError(
                    "c1_sa_candidate requires B_ao, ncas, dm2_p2mo, dm2_d2av, and dm2_plmo"
                )
            # The Molcas SA active term is RI-native. Keep the AO4 slot zero and
            # emit the active candidate via metadata for the caller to contract
            # through the DF derivative path.
            ao4_th = np.zeros_like(ao4_coulomb, dtype=np.float64)
            th_mode = "c1_sa_candidate"
        else:
            ao4_th = np.zeros_like(ao4_coulomb, dtype=np.float64)
            th_mode = "absent"
    else:
        ao4_th = np.asarray(ao4_thpkl, dtype=np.float64)
        if ao4_th.shape != ao4_coulomb.shape:
            raise ValueError(
                f"ao4_thpkl shape mismatch: expected {ao4_coulomb.shape}, got {ao4_th.shape}"
            )
        th_mode = "provided"

    btamp_term_n = str(btamp_term).strip().lower()
    if btamp_term_n not in {"ik_to_jl", "il_to_jk", "jl_to_ik", "jk_to_il", "pair_only", "total"}:
        btamp_term_n = "total"

    if t2_mo is None:
        ao4_btamp = np.zeros_like(ao4_coulomb, dtype=np.float64)
        btamp_mode = "absent"
    else:
        from asuka.caspt2.gradient.dense_btamp import build_gtoc_dense_term_arrays  # noqa: PLC0415

        ao4_btamp = np.asarray(
            build_gtoc_dense_term_arrays(
                mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
                t2_mo=np.asarray(t2_mo, dtype=np.float64),
            )[str(btamp_term_n)],
            dtype=np.float64,
        )
        btamp_mode = str(btamp_term_n)

    ao4_total = np.asarray(
        ao4_coulomb + ao4_exchange + ao4_th + ao4_btamp,
        dtype=np.float64,
    )
    term_manifest = build_dense_molcas_term_manifest(
        exchange_mode=str(exch_mode_n),
        thpkl_mode=str(th_mode),
        btamp_mode=str(btamp_mode),
        one_body_mode=str(one_body_mode),
        fockgen_exact=bool(
            gfock_ci_mo is not None
            or (
                eri_mo is not None
                and fock_inactive_mo is not None
                and dm1_ci_active_mo is not None
                and dm2_plmo is not None
                and ncas is not None
            )
        ),
        addgrad_exact=bool(gfock_ref_mo is not None),
    )
    non_exact_terms = tuple(
        str(name)
        for name, info in term_manifest.items()
        if not bool(dict(info).get("exact", False))
    )
    return DenseMolcasSSAssembly(
        d0_inactive_ao=np.asarray(d0_1, dtype=np.float64),
        d0_var_ref_ao=np.asarray(d0_2_ref, dtype=np.float64),
        d0_var_total_ao=np.asarray(d0_2_total, dtype=np.float64),
        d0_var_pt2_ao=np.asarray(d0_2, dtype=np.float64),
        d0_active_ao=np.asarray(d0_3, dtype=np.float64),
        d0_dlao_ao=np.asarray(d0_4, dtype=np.float64),
        d1aovar_pt2_ao=np.asarray(d1aovar_pt2, dtype=np.float64),
        d1aovar_total_ao=np.asarray(d1aovar_total, dtype=np.float64),
        fockocc_pt2_ao=np.asarray(fockocc_pt2, dtype=np.float64),
        fockocc_ref_ao=np.asarray(fockocc_ref, dtype=np.float64),
        fockocc_total_ao=np.asarray(fockocc_total, dtype=np.float64),
        d1aovar_parts={str(k): np.asarray(v, dtype=np.float64) for k, v in d1_parts.items()},
        fockocc_parts={str(k): np.asarray(v, dtype=np.float64) for k, v in fock_parts.items()},
        ao4_ref_candidate=np.asarray(ao4_ref, dtype=np.float64),
        ao4_total_raw_candidate=np.asarray(ao4_total_raw, dtype=np.float64),
        ao4_coulomb=np.asarray(ao4_coulomb, dtype=np.float64),
        ao4_exchange_cross_candidate=np.asarray(ao4_exchange, dtype=np.float64),
        ao4_thpkl=np.asarray(ao4_th, dtype=np.float64),
        ao4_btamp=np.asarray(ao4_btamp, dtype=np.float64),
        ao4_total_candidate=np.asarray(ao4_total, dtype=np.float64),
        term_manifest={str(k): dict(v) for k, v in term_manifest.items()},
        all_terms_exact=bool(len(non_exact_terms) == 0),
        non_exact_terms=tuple(non_exact_terms),
        meta={
            "thpkl_mode": str(th_mode),
            "btamp_mode": str(btamp_mode),
            "exchange_mode": str(exch_mode_n),
            "one_body_mode": str(one_body_mode),
            "exchange_barx_candidate": exchange_barx,
            "term_manifest": {str(k): dict(v) for k, v in term_manifest.items()},
            "all_terms_exact": bool(len(non_exact_terms) == 0),
            "non_exact_terms": tuple(non_exact_terms),
            "thpkl_bar_candidate": build_c1_sa_active_thpkl_candidate_bar(
                B_ao=np.asarray(B_ao, dtype=np.float64),
                df_metric_chol=np.asarray(df_metric_chol, dtype=np.float64) if df_metric_chol is not None else None,
                mo_coeff=np.asarray(mo_coeff, dtype=np.float64),
                z_orb=np.asarray(z_orb, dtype=np.float64),
                dm2_d2av=np.asarray(dm2_d2av, dtype=np.float64),
                dm2_plmo=np.asarray(dm2_plmo, dtype=np.float64),
                ncore=int(ncore),
                ncas=int(ncas),
            ) if str(thpkl_mode).strip().lower() == "c1_sa_candidate" else None,
            "thpkl_df_space": "unwhitened_x" if str(thpkl_mode).strip().lower() == "c1_sa_candidate" else "absent",
        },
    )


def contract_dense_molcas_ss_assembly(
    *,
    assembly: DenseMolcasSSAssembly,
    ao_basis: Any,
    atom_coords_bohr: np.ndarray,
) -> dict[str, np.ndarray]:
    """Contract dense Molcas-shaped AO4 tensors against exact dense 4c derivatives."""

    from asuka.caspt2.gradient.dense_btamp import contract_ordered_ao4_dense_deriv  # noqa: PLC0415

    coords = np.asarray(atom_coords_bohr, dtype=np.float64)
    de_coul = contract_ordered_ao4_dense_deriv(
        ao_basis=ao_basis,
        atom_coords_bohr=coords,
        bar_ao4=np.asarray(assembly.ao4_coulomb, dtype=np.float64),
    )
    de_exch = contract_ordered_ao4_dense_deriv(
        ao_basis=ao_basis,
        atom_coords_bohr=coords,
        bar_ao4=np.asarray(assembly.ao4_exchange_cross_candidate, dtype=np.float64),
    )
    de_th = contract_ordered_ao4_dense_deriv(
        ao_basis=ao_basis,
        atom_coords_bohr=coords,
        bar_ao4=np.asarray(assembly.ao4_thpkl, dtype=np.float64),
    )
    de_bt = contract_ordered_ao4_dense_deriv(
        ao_basis=ao_basis,
        atom_coords_bohr=coords,
        bar_ao4=np.asarray(assembly.ao4_btamp, dtype=np.float64),
    )
    de_tot = np.asarray(de_coul + de_exch + de_th + de_bt, dtype=np.float64)
    return {
        "de_2e_coulomb": np.asarray(de_coul, dtype=np.float64),
        "de_2e_exchange_cross_candidate": np.asarray(de_exch, dtype=np.float64),
        "de_2e_thpkl": np.asarray(de_th, dtype=np.float64),
        "de_2e_btamp": np.asarray(de_bt, dtype=np.float64),
        "de_2e_total_candidate": np.asarray(de_tot, dtype=np.float64),
    }
