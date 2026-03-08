"""Debug and alignment utilities for CASPT2 gradient development."""

from __future__ import annotations

from typing import Any
from pathlib import Path
from itertools import permutations

import numpy as np


def _asnumpy_f64(a: Any) -> np.ndarray:
    try:
        import cupy as cp

        if isinstance(a, cp.ndarray):
            return np.asarray(cp.asnumpy(a), dtype=np.float64)
    except Exception:
        pass
    return np.asarray(a, dtype=np.float64)


def _parse_csv_ints(text: str) -> list[int]:
    vals: list[int] = []
    for tok in str(text).replace(";", ",").split(","):
        t = str(tok).strip()
        if not t:
            continue
        vals.append(int(t))
    return vals


def _parse_csv_floats(text: str) -> list[float]:
    vals: list[float] = []
    for tok in str(text).replace(";", ",").split(","):
        t = str(tok).strip()
        if not t:
            continue
        vals.append(float(t))
    return vals


def _parse_ci_basis_map_from_env(n_ci: int) -> dict[str, Any] | None:
    """Previously parsed CI basis map from env vars; now always returns None."""
    return None


def _apply_ci_basis_map(vec: np.ndarray, map_spec: dict[str, Any] | None) -> np.ndarray:
    """Previously applied CI permutation/sign map; now returns input unchanged."""
    return np.asarray(vec, dtype=np.float64).ravel()


def _read_molcas_dump_matrix(path: Path) -> np.ndarray:
    """Read a Molcas GRAD_DUMP 2D matrix file."""
    toks = str(path.read_text(encoding="utf-8")).split()
    if len(toks) < 2:
        raise ValueError(f"{path}: invalid dump header")
    nrow = int(toks[0])
    ncol = int(toks[1])
    nval = int(nrow * ncol)
    if len(toks) < 2 + nval:
        raise ValueError(f"{path}: insufficient values ({len(toks) - 2} < {nval})")
    vals = [
        float(str(x).replace("D", "E").replace("d", "E"))
        for x in toks[2 : 2 + nval]
    ]
    return np.asarray(vals, dtype=np.float64).reshape((nrow, ncol), order="F")


def _best_signed_perm_vec(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_n: int = 8,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Best signed permutation mapping 1D vector ``a`` to ``b``.

    Returns ``(perm, signs)`` such that ``a[perm] * signs`` best matches ``b``.
    Exhaustive over permutations for small ``n``.
    """
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    if aa.size != bb.size:
        return None
    n = int(aa.size)
    if n <= 0 or n > int(max_n):
        return None

    best_perm: np.ndarray | None = None
    best_signs: np.ndarray | None = None
    best_metric: float | None = None
    for p in permutations(range(n)):
        perm = np.asarray(p, dtype=np.int64)
        ap = np.asarray(aa[perm], dtype=np.float64)
        signs = np.where(np.abs(-ap - bb) < np.abs(ap - bb), -1.0, 1.0)
        d = np.asarray(ap * signs - bb, dtype=np.float64)
        m = float(np.max(np.abs(d)))
        if best_metric is None or m < best_metric:
            best_metric = m
            best_perm = perm.copy()
            best_signs = np.asarray(signs, dtype=np.float64)
    if best_perm is None or best_signs is None:
        return None
    return best_perm, best_signs


def _build_exact_df_like_factors_from_ao_eri(
    ao_eri: np.ndarray,
    *,
    eig_tol: float = 1.0e-12,
) -> np.ndarray:
    """Build dense-exact DF-like factors ``B[mu,nu,Q]`` from AO ERIs.

    This factorization is used only for dense-integral debug/parity lanes where
    no physical auxiliary basis exists.
    """
    eri = np.asarray(ao_eri, dtype=np.float64)
    if eri.ndim == 4:
        nao = int(eri.shape[0])
        if eri.shape != (nao, nao, nao, nao):
            raise ValueError(f"invalid AO ERI tensor shape: {eri.shape}")
        m = np.asarray(eri.reshape(nao * nao, nao * nao), dtype=np.float64)
    elif eri.ndim == 2:
        n2 = int(eri.shape[0])
        if eri.shape != (n2, n2):
            raise ValueError(f"invalid AO ERI matrix shape: {eri.shape}")
        nao = int(round(np.sqrt(float(n2))))
        if nao * nao != n2:
            raise ValueError(f"cannot infer nao from AO ERI matrix shape {eri.shape}")
        m = np.asarray(eri, dtype=np.float64)
    else:
        raise ValueError(f"ao_eri must be 2D or 4D, got ndim={eri.ndim}")

    m = np.asarray(0.5 * (m + m.T), dtype=np.float64)
    w, u = np.linalg.eigh(m)
    wmax = float(np.max(np.abs(w))) if int(w.size) else 0.0
    thr = max(float(eig_tol), float(eig_tol) * wmax)
    keep = np.asarray(w > thr, dtype=bool)
    if int(np.count_nonzero(keep)) < 1:
        raise ValueError("exact AO-ERI factorization kept zero positive eigenmodes")
    b_flat = np.asarray(u[:, keep] * np.sqrt(w[keep])[None, :], dtype=np.float64)
    return np.asarray(b_flat.reshape(nao, nao, int(b_flat.shape[1])), dtype=np.float64)


def _align_df_like_factors_to_reference(
    b_ref: np.ndarray,
    b_trial: np.ndarray,
) -> np.ndarray:
    """Align dense-exact DF-like factor columns via orthogonal Procrustes."""
    br = np.asarray(b_ref, dtype=np.float64)
    bt = np.asarray(b_trial, dtype=np.float64)
    if br.shape != bt.shape or br.ndim != 3:
        raise ValueError(f"B shape mismatch for alignment: ref={br.shape} trial={bt.shape}")
    nao, nao1, naux = map(int, br.shape)
    if nao != nao1:
        raise ValueError(f"B must have shape (nao,nao,naux), got {br.shape}")

    y = np.asarray(br.reshape(nao * nao, naux), dtype=np.float64)
    x = np.asarray(bt.reshape(nao * nao, naux), dtype=np.float64)
    u, _s, vt = np.linalg.svd(x.T @ y, full_matrices=False)
    rot = np.asarray(u @ vt, dtype=np.float64)
    xa = np.asarray(x @ rot, dtype=np.float64)

    # Keep deterministic column signs relative to the reference basis.
    dots = np.asarray(np.einsum("iq,iq->q", xa, y, optimize=True), dtype=np.float64)
    sgn = np.where(dots < 0.0, -1.0, 1.0).reshape(1, naux)
    xa = xa * sgn
    return np.asarray(xa.reshape(nao, nao, naux), dtype=np.float64)


def _infer_ci_basis_map_from_dump(
    *,
    n_ci: int,
    ci_ref_asuka: np.ndarray,
) -> dict[str, Any] | None:
    """Previously inferred CI basis map from Molcas dump; now always returns None."""
    return None


def _infer_ci_basis_map_from_resp(
    *,
    n_ci: int,
    n_orb_packed: int,
    z_ci_asuka: np.ndarray,
) -> dict[str, Any] | None:
    """Previously inferred CI basis map from Molcas RESP file; now always returns None."""
    return None


def _resolve_response_dpt2_mode(mode: str | None) -> str:
    m = str(mode or "full").strip().lower()
    if m in {"", "default", "auto"}:
        return "full"
    if m not in {"full", "bare"}:
        raise ValueError(
            "ASUKA_RESPONSE_DPT2_DENSITY must be one of {'full','bare','auto'}"
        )
    return m


def _build_dlao_candidate_ao(
    *,
    mo_coeff: np.ndarray,
    z_orb: np.ndarray,
    dpt2_mo: np.ndarray,
    ncore: int,
) -> np.ndarray:
    """Reconstruct the Molcas `DLAO` candidate in AO basis.

    Molcas `Out_Pt2` forms the MO-space object

      DAO = OITD(K2, act=.false.) + DPT2_mo

    with

      OITD(K2, act=.false.) = D_core * K2^T - K2^T * D_core,

    where `D_core` is the inactive doubly-occupied density in MO basis.
    """

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
    d_core = np.zeros((nmo, nmo), dtype=np.float64)
    if int(ncore) > 0:
        d_core[: int(ncore), : int(ncore)] = 2.0 * np.eye(int(ncore), dtype=np.float64)
    dao_mo = np.asarray(d_core @ K.T - K.T @ d_core + Dpt2, dtype=np.float64)
    return np.asarray(C @ dao_mo @ C.T, dtype=np.float64)


def _apply_debug_zorb_block_signs(
    z_orb: np.ndarray,
    *,
    ncore: int,
    ncas: int,
) -> tuple[np.ndarray, dict[str, bool]]:
    """Previously applied debug sign flips to z_orb blocks; now returns input unchanged."""
    return np.asarray(z_orb, dtype=np.float64).copy(), {
        "flip_d": False,
        "flip_c": False,
    }
