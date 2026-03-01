from __future__ import annotations

"""Self-contained SCF (RHF/UHF/ROHF) drivers over precomputed DF integrals.

Design goal
-----------
Keep the SCF *driver* self-contained. These routines take
already-built one-/two-electron data (typically in AO basis):

- Overlap S (nao, nao)
- Core Hamiltonian hcore (nao, nao)
- Density-fitting factors B (nao, nao, naux) such that:
    (μν|λσ) ~= Σ_Q B[μν,Q] B[λσ,Q]

The DF path is the only supported 2e backend in this module.
"""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
import time

from . import df_jk

_HF_CPU_EIGH_MAX_NAO = max(0, int(os.environ.get("ASUKA_HF_CPU_EIGH_MAX_NAO", "96")))


def _get_xp(*arrays: Any):
    """Return (xp, is_gpu) where xp is numpy or cupy based on array types."""

    try:
        import cupy as cp  # type: ignore
    except Exception:  # pragma: no cover
        cp = None  # type: ignore

    if cp is not None:
        for a in arrays:
            if isinstance(a, cp.ndarray):  # type: ignore[attr-defined]
                return cp, True
    return np, False


def _as_xp(xp, a, *, dtype):
    out = xp.asarray(a, dtype=dtype)
    return out


def _symmetrize(xp, A):
    return 0.5 * (A + A.T)


def _orthogonalizer_from_S(S, *, eps: float = 1e-12):
    """Return X such that X.T @ S @ X = I (symmetric orthogonalization)."""

    xp, _ = _get_xp(S)
    S = _as_xp(xp, S, dtype=xp.float64)
    S = _symmetrize(xp, S)
    s, U = xp.linalg.eigh(S)
    if xp.any(s <= eps):
        raise ValueError("S is not positive definite (small/negative eigenvalues)")
    X = U @ xp.diag(s ** (-0.5)) @ U.T
    return X


def _gen_eigh_with_X(F, X):
    """Solve F C = S C eps given an orthogonalizer X from S; returns (eps, C)."""

    xp, _ = _get_xp(F, X)
    F = _as_xp(xp, F, dtype=xp.float64)
    X = _as_xp(xp, X, dtype=xp.float64)
    F = _symmetrize(xp, F)
    Fp = X.T @ F @ X

    # For small GPU problems, host LAPACK often beats repeated tiny-device-eigh calls.
    use_cpu_eigh = bool(
        xp is not np and int(_HF_CPU_EIGH_MAX_NAO) > 0 and int(Fp.shape[0]) <= int(_HF_CPU_EIGH_MAX_NAO)
    )
    if use_cpu_eigh:
        Fp_h = np.asarray(xp.asnumpy(_symmetrize(xp, Fp)), dtype=np.float64)
        e_h, Cp_h = np.linalg.eigh(Fp_h)
        e = xp.asarray(e_h, dtype=xp.float64)
        Cp = xp.asarray(Cp_h, dtype=xp.float64)
    else:
        e, Cp = xp.linalg.eigh(_symmetrize(xp, Fp))

    C = X @ Cp
    return e, C


def _gen_eigh(F, S, *, eps: float = 1e-12):
    """Solve F C = S C eps; returns (eps, C) with S-orthonormal columns."""

    xp, _ = _get_xp(F, S)
    F = _as_xp(xp, F, dtype=xp.float64)
    S = _as_xp(xp, S, dtype=xp.float64)
    F = _symmetrize(xp, F)
    S = _symmetrize(xp, S)
    X = _orthogonalizer_from_S(S, eps=eps)
    return _gen_eigh_with_X(F, X)


def _occ_rhf(nelec: int, nao: int):
    nelec = int(nelec)
    if nelec < 0 or nelec % 2 != 0:
        raise ValueError("RHF requires an even nelec >= 0")
    nocc = nelec // 2
    if nocc > int(nao):
        raise ValueError("nelec/2 exceeds number of orbitals")
    occ = np.zeros((nao,), dtype=np.float64)
    occ[:nocc] = 2.0
    return occ, nocc


def _occ_uhf(nalpha: int, nbeta: int, nao: int):
    nalpha = int(nalpha)
    nbeta = int(nbeta)
    if nalpha < 0 or nbeta < 0:
        raise ValueError("nalpha/nbeta must be >= 0")
    if nalpha > int(nao) or nbeta > int(nao):
        raise ValueError("nalpha/nbeta exceeds number of orbitals")
    occ_a = np.zeros((nao,), dtype=np.float64)
    occ_b = np.zeros((nao,), dtype=np.float64)
    occ_a[:nalpha] = 1.0
    occ_b[:nbeta] = 1.0
    return occ_a, occ_b


def _density_from_C_occ(C, occ):
    xp, _ = _get_xp(C, occ)
    C = _as_xp(xp, C, dtype=xp.float64)
    occ = _as_xp(xp, occ, dtype=xp.float64).ravel()
    if C.ndim != 2:
        raise ValueError("C must be 2D")
    nao, nmo = map(int, C.shape)
    if occ.shape != (nmo,):
        raise ValueError(f"occ must have shape ({nmo},), got {tuple(occ.shape)}")
    # D = C diag(occ) C^T
    return (C * occ[None, :]) @ C.T


def _df_JK(B, D, *, want_J: bool = True, want_K: bool = True, B2=None, BQ=None, profile: dict | None = None):
    """Compute Coulomb J and (optionally) exchange K from DF factors and density.

    Inputs
    - B: (nao, nao, naux), float64
    - D: (nao, nao), float64
    """

    xp, _ = _get_xp(B, D)
    B = _as_xp(xp, B, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if B.ndim != 3:
        raise ValueError("B must have shape (nao, nao, naux)")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if tuple(B.shape[:2]) != (nao, nao):
        raise ValueError("B and D nao mismatch")

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

        # Lightweight layout metadata (useful to catch implicit copies / strided access).
        try:
            jk_prof.setdefault("B_shape", list(map(int, B.shape)))
            jk_prof.setdefault("B_strides", list(map(int, getattr(B, "strides", ()))))
            jk_prof.setdefault("B_c_contig", bool(getattr(B, "flags", {}).c_contiguous) if hasattr(B, "flags") else None)
        except Exception:
            pass

    J = None
    if want_J:
        tJ = _time_ms_start(xp) if profile is not None else None
        # J via vectorization:
        #   J_{μν} = Σ_Q B_{μν}^Q * Σ_{λσ} D_{λσ} B_{λσ}^Q
        naux = int(B.shape[2])
        if B2 is None:
            B2 = B.reshape((nao * nao, naux))
        dvec = D.reshape((nao * nao,))
        v = B2.T @ dvec
        J = (B2 @ v).reshape((nao, nao))
        if profile is not None and tJ is not None:
            jk_prof = profile.setdefault("jk", {})
            jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

    if not want_K:
        return J, None

    # K via batched GEMMs: K = Σ_Q B_Q @ D @ B_Q^T
    if BQ is None:
        BQ = B.transpose((2, 0, 1))  # (naux, nao, nao)
    if profile is not None:
        try:
            jk_prof = profile.setdefault("jk", {})
            jk_prof.setdefault("BQ_shape", list(map(int, BQ.shape)))
            jk_prof.setdefault("BQ_strides", list(map(int, getattr(BQ, "strides", ()))))
            jk_prof.setdefault("BQ_c_contig", bool(getattr(BQ, "flags", {}).c_contiguous) if hasattr(BQ, "flags") else None)
        except Exception:
            pass

    naux = int(BQ.shape[0])

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["k_impl"] = "gemm_gpu" if xp is not np else "batched_matmul"

    tK = _time_ms_start(xp) if profile is not None else None
    BD = None
    KQ = None
    if xp is not np:
        # GPU path: single GEMM  K = M1 @ B2d.T
        #   B2d[k, n*naux+Q] = B[k,n,Q]  (free reshape, C-contiguous)
        #   BQD[m,Q,n]       = Σ_p B[m,p,Q]*D[p,n]  via tensordot
        #   M1[m, n*naux+Q]  = BQD[m,Q,n]  (contiguous copy of BQD.T021)
        #   K[m,k]           = Σ_{n,Q} M1[m,n*naux+Q] * B2d[k,n*naux+Q]
        B2d = B.reshape(nao, nao * naux)          # view, no copy
        BQD = xp.tensordot(B, D, axes=([1], [0]))  # (nao, naux, nao)
        M1 = xp.ascontiguousarray(BQD.transpose(0, 2, 1)).reshape(nao, nao * naux)
        del BQD
        K = M1 @ B2d.T
        del M1
    else:
        BD = xp.matmul(BQ, D)  # (naux, nao, nao)
        KQ = xp.matmul(BD, BQ.transpose((0, 2, 1)))  # (naux, nao, nao)
        K = xp.sum(KQ, axis=0)
    if profile is not None and tK is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
        try:
            jk_prof.setdefault("BD_nbytes", int(getattr(BD, "nbytes", 0) if BD is not None else 0))
            jk_prof.setdefault("KQ_nbytes", int(getattr(KQ, "nbytes", 0) if KQ is not None else 0))
        except Exception:
            pass
    return J, K


def _fock_error_rhf(F, D, S):
    xp, _ = _get_xp(F, D, S)
    return F @ D @ S - S @ D @ F


def _roothaan_fock_rohf(Fa, Fb, Da, Db, S):
    """Return Roothaan's effective Fock matrix for ROHF.

    Matches PySCF's `pyscf.scf.rohf.get_roothaan_fock` construction but works
    with either NumPy or CuPy arrays.
    """

    xp, _ = _get_xp(Fa, Fb, Da, Db, S)
    Fa = _as_xp(xp, Fa, dtype=xp.float64)
    Fb = _as_xp(xp, Fb, dtype=xp.float64)
    Da = _as_xp(xp, Da, dtype=xp.float64)
    Db = _as_xp(xp, Db, dtype=xp.float64)
    S = _as_xp(xp, S, dtype=xp.float64)

    nao = int(S.shape[0])
    fc = 0.5 * (Fa + Fb)

    # Projectors for core/closed (pc), open (po), and virtual (pv) spaces.
    pc = Db @ S
    po = (Da - Db) @ S
    pv = xp.eye(nao, dtype=xp.float64) - (Da @ S)

    F = 0.5 * (pc.T @ fc @ pc)
    F += 0.5 * (po.T @ fc @ po)
    F += 0.5 * (pv.T @ fc @ pv)
    F += po.T @ Fb @ pc
    F += po.T @ Fa @ pv
    F += pv.T @ fc @ pc
    return _symmetrize(xp, F + F.T)


class _DIIS:
    def __init__(self, max_vec: int = 8):
        self.max_vec = int(max_vec)
        self._F: list[Any] = []
        self._e: list[Any] = []

    def push(self, F, e):
        self._F.append(F)
        self._e.append(e)
        if len(self._F) > self.max_vec:
            self._F.pop(0)
            self._e.pop(0)

    def extrapolate(self):
        if len(self._F) < 2:
            return self._F[-1]
        xp, _ = _get_xp(self._F[-1])
        n = len(self._F)

        # Build the DIIS B matrix via a single Gram matrix GEMM:
        #   G[i,j] = <e_i | e_j>
        E = xp.stack([xp.ravel(e).astype(xp.float64, copy=False) for e in self._e], axis=0)  # (n, m)
        G = E @ E.T  # (n, n)

        B = xp.empty((n + 1, n + 1), dtype=xp.float64)
        B[:n, :n] = G
        B[:n, n] = -1.0
        B[n, :n] = -1.0
        B[n, n] = 0.0

        rhs = xp.zeros((n + 1,), dtype=xp.float64)
        rhs[n] = -1.0
        coeff = xp.linalg.solve(B, rhs)[:n]  # (n,)

        Fstk = xp.stack([xp.asarray(F, dtype=xp.float64) for F in self._F], axis=0)  # (n, nao, nao)
        return xp.tensordot(coeff, Fstk, axes=(0, 0))


@dataclass(frozen=True)
class SCFResult:
    method: str
    converged: bool
    niter: int
    e_tot: float
    e_elec: float
    e_nuc: float
    mo_energy: Any
    mo_coeff: Any
    mo_occ: Any


def _time_ms_start(xp):
    """Return an opaque timer handle for xp (numpy or cupy)."""

    if xp is np:
        return time.perf_counter()
    start = xp.cuda.Event()
    stream = xp.cuda.get_current_stream()
    start.record(stream)
    return (start, stream)


def _time_ms_end(xp, handle) -> float:
    if xp is np:
        return float((time.perf_counter() - float(handle)) * 1000.0)
    start, stream = handle
    end = xp.cuda.Event()
    end.record(stream)
    end.synchronize()
    return float(xp.cuda.get_elapsed_time(start, end))


def rhf_df(
    S,
    hcore,
    B,
    *,
    nelec: int,
    enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 1,
    diis_space: int = 8,
    damping: float = 0.0,
    level_shift: float = 0.0,
    jk_mode: str = "materialized",
    k_engine: str = "auto",
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    ao_basis=None,
    aux_basis=None,
    df_backend: str = "gpu_rys",
    df_threads: int = 256,
    df_mode: str = "auto",
    df_aux_block_naux: int = 256,
    L_metric=None,
    dm0=None,
    mo_coeff0=None,
    init_fock_cycles: int = 1,
    profile: dict | None = None,
):
    """RHF SCF with DF ERIs in AO basis.

    Parameters
    - S, hcore: (nao, nao)
    - B: DF factors tensor (materialized mode) with shape (nao, nao, naux) or (naux, nao, nao),
         or None (streamed mode).
    - nelec: total electrons (even)
    """

    jk_mode_s = str(jk_mode).strip().lower()
    if jk_mode_s not in {"materialized", "streamed", "auto"}:
        raise ValueError("jk_mode must be one of: 'materialized', 'streamed', 'auto'")
    use_streamed = bool(jk_mode_s == "streamed" or (jk_mode_s == "auto" and B is None))

    if use_streamed:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("jk_mode='streamed' requires CuPy") from e
        xp, is_gpu = cp, True
    else:
        xp, is_gpu = _get_xp(S, hcore, B)
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    X = _orthogonalizer_from_S(S)
    if not use_streamed:
        B_in = _as_xp(xp, B, dtype=xp.float64)
        if B_in.ndim != 3:
            raise ValueError("B must be a 3D array with shape (nao, nao, naux) or (naux, nao, nao)")

        # Accept either B[μ,ν,Q] or BQ[Q,μ,ν] to avoid holding both layouts.
        B_mnQ = None
        B2 = None
        if int(B_in.shape[0]) == int(nao) and int(B_in.shape[1]) == int(nao):
            B_mnQ = B_in
            naux = int(B_mnQ.shape[2])
            B2 = B_mnQ.reshape((nao * nao, naux))
            BQ = B_mnQ.transpose((2, 0, 1))
        elif int(B_in.shape[1]) == int(nao) and int(B_in.shape[2]) == int(nao):
            BQ = B_in
            naux = int(BQ.shape[0])
        else:
            raise ValueError("B must have shape (nao, nao, naux) or (naux, nao, nao)")
    else:
        B_mnQ = None
        B2 = None
        BQ = None
        naux = None

    occ_np, _nocc = _occ_rhf(nelec, nao)
    nocc = int(_nocc)
    occ = _as_xp(xp, occ_np, dtype=xp.float64)

    eps, C = _gen_eigh_with_X(h, X)
    if dm0 is not None:
        D = _as_xp(xp, dm0, dtype=xp.float64)
        if D.shape != (nao, nao):
            raise ValueError("dm0 must have shape (nao, nao)")
        D = _symmetrize(xp, D)
    elif mo_coeff0 is not None:
        C0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 must have shape (nao, nao)")
        C = C0
        D = _symmetrize(xp, _density_from_C_occ(C, occ))
    else:
        D = _density_from_C_occ(C, occ)

    k_engine_s = str(k_engine).strip().lower()
    if k_engine_s not in {"auto", "from_cocc", "from_d"}:
        raise ValueError("k_engine must be one of: 'auto', 'from_Cocc', 'from_D'")
    use_k_cocc = bool(k_engine_s == "from_cocc" or (k_engine_s == "auto" and bool(is_gpu)))
    if use_streamed and not use_k_cocc:
        raise NotImplementedError("jk_mode='streamed' currently requires k_engine='from_Cocc' (or 'auto' on GPU)")
    if use_streamed and dm0 is not None:
        raise NotImplementedError("jk_mode='streamed' does not support dm0 (needs a C_occ-consistent density)")

    k_q_block = int(k_q_block)
    if k_q_block <= 0:
        raise ValueError("k_q_block must be > 0")
    lam = float(damping) if damping else 0.0
    K_prev = None

    # Ensure BQ layout is contiguous on GPU (K is bandwidth/throughput sensitive).
    #
    # For large bases in mnQ layout, making a full contiguous copy of BQ
    # doubles the footprint of the DF tensor (often >10GB) and can OOM a
    # 24GB GPU.  When we can compute K directly from mnQ with q-blocking,
    # skip the full BQ copy and keep BQ as a strided view.
    b_layout = "streamed" if use_streamed else ("mnQ" if B_mnQ is not None else "Qmn")
    bq_copied = False
    if (
        (not use_streamed)
        and bool(is_gpu)
        and (BQ is not None)
        and hasattr(BQ, "flags")
        and (not bool(BQ.flags.c_contiguous))
        and (not (use_k_cocc and B_mnQ is not None))
    ):
        BQ = xp.ascontiguousarray(BQ)
        bq_copied = True

    diis_obj = _DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["B_layout"] = str(b_layout)
        jk_prof["BQ_copied"] = bool(bq_copied)
        if bool(is_gpu):
            try:
                pool = xp.get_default_memory_pool()
                jk_prof["mem_pool_used_bytes"] = int(pool.used_bytes())
                jk_prof["mem_pool_total_bytes"] = int(pool.total_bytes())
            except Exception:
                pass

    if profile is not None and use_k_cocc:
        # Record BQ layout metadata even when we don't call the D-driven K path.
        jk_prof = profile.setdefault("jk", {})
        try:
            if BQ is not None:
                jk_prof.setdefault("BQ_shape", list(map(int, BQ.shape)))
                jk_prof.setdefault("BQ_strides", list(map(int, getattr(BQ, "strides", ()))))
                jk_prof.setdefault(
                    "BQ_c_contig", bool(getattr(BQ, "flags", {}).c_contiguous) if hasattr(BQ, "flags") else None
                )
        except Exception:
            pass

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("init_fock_ms", 0.0)
        prof.setdefault("init_fock_cycles", 0)
        prof.setdefault("init_fock_applied", False)
        prof.setdefault("iters", 0)

    if use_streamed:
        if ao_basis is None or aux_basis is None:
            raise ValueError("jk_mode='streamed' requires ao_basis and aux_basis")
        from . import df_jk_streamed  # local import to avoid cuERI deps in materialized mode

        ctx = df_jk_streamed.make_streamed_df_jk_context(
            ao_basis,
            aux_basis,
            L_metric=L_metric,
            backend=str(df_backend),
            threads=int(df_threads),
            mode=str(df_mode),
            aux_block_naux=int(df_aux_block_naux),
            profile=profile,
        )
        jk_work: dict = {}

    init_fock_cycles = max(0, int(init_fock_cycles))
    run_init_predictor = bool(init_fock_cycles > 0 and dm0 is None and mo_coeff0 is None)
    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof["init_fock_cycles"] = int(init_fock_cycles)
        prof["init_fock_applied"] = bool(run_init_predictor)

    if run_init_predictor:
        for _ in range(int(init_fock_cycles)):
            D_prev = D
            t_init = _time_ms_start(xp) if profile is not None else None

            if use_streamed:
                C_occ = xp.ascontiguousarray(C[:, :nocc])
                occ_vals = occ[:nocc]
                J, K = df_jk_streamed.df_JK_streamed(
                    ctx,
                    D_prev,
                    C_occ,
                    occ_vals,
                    k_q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    work=jk_work,
                    profile=None,
                )
                K_prev = K
            elif use_k_cocc:
                if B2 is not None:
                    J = df_jk.df_J_from_B2_D(B2, D_prev)
                else:
                    J = df_jk.df_J_from_BQ_D(BQ, D_prev)
                J = _symmetrize(xp, J)

                C_occ = xp.ascontiguousarray(C[:, :nocc])
                occ_vals = occ[:nocc]
                if B_mnQ is not None:
                    K = df_jk.df_K_from_BmnQ_Cocc(
                        B_mnQ,
                        C_occ,
                        occ_vals,
                        q_block=int(k_q_block),
                        cublas_math_mode=cublas_math_mode,
                    )
                else:
                    K = df_jk.df_K_from_BQ_Cocc(
                        BQ,
                        C_occ,
                        occ_vals,
                        q_block=int(k_q_block),
                        cublas_math_mode=cublas_math_mode,
                    )
                K = _symmetrize(xp, K)
                K_prev = K
            else:
                if B_mnQ is not None:
                    J, K = _df_JK(B_mnQ, D_prev, want_J=True, want_K=True, B2=B2, BQ=BQ, profile=None)
                else:
                    J = df_jk.df_J_from_BQ_D(BQ, D_prev)
                    J = _symmetrize(xp, J)
                    K = df_jk.df_K_from_BQ_D(BQ, D_prev, profile=None)
                    K = _symmetrize(xp, K)
                if use_k_cocc:
                    K_prev = K

            F = h + J - 0.5 * K
            F = _symmetrize(xp, F)

            if level_shift:
                shift = float(level_shift)
                if shift != 0.0:
                    Fp = X.T @ F @ X
                    Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                    F = X @ Fp @ X.T

            eps, C = _gen_eigh_with_X(F, X)
            D = _symmetrize(xp, _density_from_C_occ(C, occ))
            if damping:
                D = (1.0 - lam) * D + lam * D_prev

            if profile is not None and t_init is not None:
                profile["scf"]["init_fock_ms"] += _time_ms_end(xp, t_init)

    for cycle in range(1, int(max_cycle) + 1):
        t = _time_ms_start(xp)
        if use_streamed:
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1
                jk_prof["k_impl"] = "streamed_cocc_trsm"
                jk_prof["k_q_block"] = int(k_q_block)
                jk_prof["k_nocc"] = int(nocc)

            C_occ = xp.ascontiguousarray(C[:, :nocc])
            occ_vals = occ[:nocc]
            J, K_pure = df_jk_streamed.df_JK_streamed(
                ctx,
                D,
                C_occ,
                occ_vals,
                k_q_block=int(k_q_block),
                cublas_math_mode=cublas_math_mode,
                work=jk_work,
                profile=profile,
            )
            if damping and K_prev is not None:
                K = (1.0 - lam) * K_pure + lam * K_prev
            else:
                K = K_pure
            K_prev = K
        elif use_k_cocc and not (cycle == 1 and dm0 is not None):
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

            tJ = _time_ms_start(xp) if profile is not None else None
            if B2 is not None:
                J = df_jk.df_J_from_B2_D(B2, D)
            else:
                J = df_jk.df_J_from_BQ_D(BQ, D)
            J = _symmetrize(xp, J)
            if profile is not None and tJ is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

            tK = _time_ms_start(xp) if profile is not None else None
            C_occ = xp.ascontiguousarray(C[:, :nocc])
            occ_vals = occ[:nocc]
            if B_mnQ is not None:
                K_pure = df_jk.df_K_from_BmnQ_Cocc(
                    B_mnQ,
                    C_occ,
                    occ_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
            else:
                K_pure = df_jk.df_K_from_BQ_Cocc(
                    BQ,
                    C_occ,
                    occ_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
            if damping and K_prev is not None:
                K = (1.0 - lam) * K_pure + lam * K_prev
            else:
                K = K_pure
            K = _symmetrize(xp, K)

            if profile is not None and tK is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
                jk_prof.setdefault("k_q_block", int(k_q_block))
                jk_prof.setdefault("k_nocc", int(nocc))

            K_prev = K
        else:
            if B_mnQ is not None:
                J, K = _df_JK(B_mnQ, D, want_J=True, want_K=True, B2=B2, BQ=BQ, profile=profile)
            else:
                if profile is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

                tJ = _time_ms_start(xp) if profile is not None else None
                J = df_jk.df_J_from_BQ_D(BQ, D)
                J = _symmetrize(xp, J)
                if profile is not None and tJ is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

                tK = _time_ms_start(xp) if profile is not None else None
                K = df_jk.df_K_from_BQ_D(BQ, D, profile=profile)
                K = _symmetrize(xp, K)
                if profile is not None and tK is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
            if use_k_cocc:
                K_prev = K
        if profile is not None:
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)
        F = h + J - 0.5 * K
        F = _symmetrize(xp, F)

        if level_shift:
            shift = float(level_shift)
            if shift != 0.0:
                Fp = X.T @ F @ X
                Fp = Fp + shift * xp.eye(nao, dtype=xp.float64)
                F = X @ Fp @ X.T

        if diis_obj is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp)
            e = _fock_error_rhf(F, D, S)
            diis_obj.push(F, e)
            F = diis_obj.extrapolate()
            F = _symmetrize(xp, F)
            if profile is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp)
        eps, C = _gen_eigh_with_X(F, X)
        if profile is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)
        D_new = _density_from_C_occ(C, occ)

        if damping:
            D_new = (1.0 - lam) * D_new + lam * D

        e_elec = float(0.5 * xp.trace(D_new @ (h + F)).item())
        e_tot = float(e_elec + float(enuc))

        dm_err = float(xp.linalg.norm(D_new - D).item())
        de = float(abs(e_tot - e_last)) if e_last is not None else float("inf")

        if de < float(conv_tol) and dm_err < float(conv_tol_dm):
            converged = True
            D = D_new
            e_last = e_tot
            break

        D = D_new
        e_last = e_tot

    if profile is not None:
        profile["scf"]["iters"] = int(cycle)
    return SCFResult(
        method="RHF",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=eps,
        mo_coeff=C,
        mo_occ=occ,
    )


def uhf_df(
    S,
    hcore,
    B,
    *,
    nalpha: int,
    nbeta: int,
    enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    jk_mode: str = "materialized",
    k_engine: str = "auto",
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    ao_basis=None,
    aux_basis=None,
    df_backend: str = "gpu_rys",
    df_threads: int = 256,
    df_mode: str = "auto",
    df_aux_block_naux: int = 256,
    L_metric=None,
    dm0=None,
    mo_coeff0=None,
    profile: dict | None = None,
):
    """UHF SCF with DF ERIs in AO basis."""

    jk_mode_s = str(jk_mode).strip().lower()
    if jk_mode_s not in {"materialized", "streamed", "auto"}:
        raise ValueError("jk_mode must be one of: 'materialized', 'streamed', 'auto'")
    use_streamed = bool(jk_mode_s == "streamed" or (jk_mode_s == "auto" and B is None))

    if use_streamed:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("jk_mode='streamed' requires CuPy") from e
        xp, is_gpu = cp, True
    else:
        xp, is_gpu = _get_xp(S, hcore, B)
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    X = _orthogonalizer_from_S(S)
    if not use_streamed:
        B_in = _as_xp(xp, B, dtype=xp.float64)
        if B_in.ndim != 3:
            raise ValueError("B must be a 3D array with shape (nao, nao, naux) or (naux, nao, nao)")

        B_mnQ = None
        B2 = None
        if int(B_in.shape[0]) == int(nao) and int(B_in.shape[1]) == int(nao):
            B_mnQ = B_in
            naux = int(B_mnQ.shape[2])
            B2 = B_mnQ.reshape((nao * nao, naux))
            BQ = B_mnQ.transpose((2, 0, 1))
        elif int(B_in.shape[1]) == int(nao) and int(B_in.shape[2]) == int(nao):
            BQ = B_in
            naux = int(BQ.shape[0])
        else:
            raise ValueError("B must have shape (nao, nao, naux) or (naux, nao, nao)")
    else:
        B_mnQ = None
        B2 = None
        BQ = None
        naux = None

    occ_a_np, occ_b_np = _occ_uhf(nalpha, nbeta, nao)
    occ_a = _as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _as_xp(xp, occ_b_np, dtype=xp.float64)

    e0, C = _gen_eigh_with_X(h, X)
    Ca = C
    Cb = C
    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for UHF must be a (Da, Db) tuple")
        Da = _as_xp(xp, dm0[0], dtype=xp.float64)
        Db = _as_xp(xp, dm0[1], dtype=xp.float64)
        if Da.shape != (nao, nao) or Db.shape != (nao, nao):
            raise ValueError("dm0 for UHF must have shape (nao, nao) for both spins")
        Da = _symmetrize(xp, Da)
        Db = _symmetrize(xp, Db)
    elif mo_coeff0 is not None:
        if isinstance(mo_coeff0, (tuple, list)):
            if len(mo_coeff0) != 2:
                raise TypeError("mo_coeff0 for UHF must be a (Ca, Cb) tuple")
            Ca0 = _as_xp(xp, mo_coeff0[0], dtype=xp.float64)
            Cb0 = _as_xp(xp, mo_coeff0[1], dtype=xp.float64)
        else:
            Ca0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
            Cb0 = Ca0
        if Ca0.shape != (nao, nao) or Cb0.shape != (nao, nao):
            raise ValueError("mo_coeff0 for UHF must have shape (nao, nao) for both spins")
        Ca = Ca0
        Cb = Cb0
        Da = _symmetrize(xp, _density_from_C_occ(Ca0, occ_a))
        Db = _symmetrize(xp, _density_from_C_occ(Cb0, occ_b))
    else:
        Da = _density_from_C_occ(Ca, occ_a)
        Db = _density_from_C_occ(Cb, occ_b)

    k_engine_s = str(k_engine).strip().lower()
    if k_engine_s not in {"auto", "from_cocc", "from_d"}:
        raise ValueError("k_engine must be one of: 'auto', 'from_Cocc', 'from_D'")
    use_k_cocc = bool(k_engine_s == "from_cocc" or (k_engine_s == "auto" and bool(is_gpu)))
    if use_streamed and not use_k_cocc:
        raise NotImplementedError("jk_mode='streamed' currently requires k_engine='from_Cocc' (or 'auto' on GPU)")
    if use_streamed and dm0 is not None:
        raise NotImplementedError("jk_mode='streamed' does not support dm0 (needs a C_occ-consistent density)")

    k_q_block = int(k_q_block)
    if k_q_block <= 0:
        raise ValueError("k_q_block must be > 0")
    lam = float(damping) if damping else 0.0
    Ka_prev = None
    Kb_prev = None

    b_layout = "streamed" if use_streamed else ("mnQ" if B_mnQ is not None else "Qmn")
    bq_copied = False
    if (
        (not use_streamed)
        and bool(is_gpu)
        and (BQ is not None)
        and hasattr(BQ, "flags")
        and (not bool(BQ.flags.c_contiguous))
        and (not (use_k_cocc and B_mnQ is not None))
    ):
        BQ = xp.ascontiguousarray(BQ)
        bq_copied = True

    diis_a = _DIIS(max_vec=diis_space) if diis else None
    diis_b = _DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["B_layout"] = str(b_layout)
        jk_prof["BQ_copied"] = bool(bq_copied)
        if bool(is_gpu):
            try:
                pool = xp.get_default_memory_pool()
                jk_prof["mem_pool_used_bytes"] = int(pool.used_bytes())
                jk_prof["mem_pool_total_bytes"] = int(pool.total_bytes())
            except Exception:
                pass

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("iters", 0)

    if profile is not None and use_k_cocc:
        jk_prof = profile.setdefault("jk", {})
        try:
            if BQ is not None:
                jk_prof.setdefault("BQ_shape", list(map(int, BQ.shape)))
                jk_prof.setdefault("BQ_strides", list(map(int, getattr(BQ, "strides", ()))))
                jk_prof.setdefault(
                    "BQ_c_contig", bool(getattr(BQ, "flags", {}).c_contiguous) if hasattr(BQ, "flags") else None
                )
        except Exception:
            pass

    if use_streamed:
        if ao_basis is None or aux_basis is None:
            raise ValueError("jk_mode='streamed' requires ao_basis and aux_basis")
        from . import df_jk_streamed  # local import to avoid cuERI deps in materialized mode

        ctx = df_jk_streamed.make_streamed_df_jk_context(
            ao_basis,
            aux_basis,
            L_metric=L_metric,
            backend=str(df_backend),
            threads=int(df_threads),
            mode=str(df_mode),
            aux_block_naux=int(df_aux_block_naux),
            profile=profile,
        )
        jk_work: dict = {}

    for cycle in range(1, int(max_cycle) + 1):
        Dtot = Da + Db
        t = _time_ms_start(xp)
        if use_streamed:
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1
                jk_prof["k_impl"] = "streamed_cocc_trsm"
                jk_prof["k_q_block"] = int(k_q_block)
                jk_prof["k_nocc"] = [int(nalpha), int(nbeta)]

            from . import df_jk_streamed  # local import

            Ca_occ = xp.ascontiguousarray(Ca[:, : int(nalpha)])
            Cb_occ = xp.ascontiguousarray(Cb[:, : int(nbeta)])
            occ_a_vals = occ_a[: int(nalpha)]
            occ_b_vals = occ_b[: int(nbeta)]
            J, Ks = df_jk_streamed.df_JKs_streamed(
                ctx,
                Dtot,
                [Ca_occ, Cb_occ],
                [occ_a_vals, occ_b_vals],
                k_q_block=int(k_q_block),
                cublas_math_mode=cublas_math_mode,
                work=jk_work,
                profile=profile,
            )
            Ka_pure, Kb_pure = Ks
            if damping and Ka_prev is not None:
                Ka = (1.0 - lam) * Ka_pure + lam * Ka_prev
            else:
                Ka = Ka_pure
            if damping and Kb_prev is not None:
                Kb = (1.0 - lam) * Kb_pure + lam * Kb_prev
            else:
                Kb = Kb_pure
            Ka_prev = Ka
            Kb_prev = Kb
        elif use_k_cocc and not (cycle == 1 and dm0 is not None):
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

            tJ = _time_ms_start(xp) if profile is not None else None
            if B2 is not None:
                J = df_jk.df_J_from_B2_D(B2, Dtot)
            else:
                J = df_jk.df_J_from_BQ_D(BQ, Dtot)
            J = _symmetrize(xp, J)
            if profile is not None and tJ is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

            tK = _time_ms_start(xp) if profile is not None else None
            Ca_occ = xp.ascontiguousarray(Ca[:, : int(nalpha)])
            Cb_occ = xp.ascontiguousarray(Cb[:, : int(nbeta)])
            occ_a_vals = occ_a[: int(nalpha)]
            occ_b_vals = occ_b[: int(nbeta)]
            if B_mnQ is not None:
                Ka_pure = df_jk.df_K_from_BmnQ_Cocc(
                    B_mnQ,
                    Ca_occ,
                    occ_a_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
                Kb_pure = df_jk.df_K_from_BmnQ_Cocc(
                    B_mnQ,
                    Cb_occ,
                    occ_b_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
            else:
                Ka_pure = df_jk.df_K_from_BQ_Cocc(
                    BQ,
                    Ca_occ,
                    occ_a_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
                Kb_pure = df_jk.df_K_from_BQ_Cocc(
                    BQ,
                    Cb_occ,
                    occ_b_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
            if damping and Ka_prev is not None:
                Ka = (1.0 - lam) * Ka_pure + lam * Ka_prev
            else:
                Ka = Ka_pure
            if damping and Kb_prev is not None:
                Kb = (1.0 - lam) * Kb_pure + lam * Kb_prev
            else:
                Kb = Kb_pure
            Ka = _symmetrize(xp, Ka)
            Kb = _symmetrize(xp, Kb)

            if profile is not None and tK is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
                jk_prof["k_q_block"] = int(k_q_block)
                jk_prof["k_nocc"] = [int(nalpha), int(nbeta)]

            Ka_prev = Ka
            Kb_prev = Kb
        else:
            if B_mnQ is not None:
                J, _ = _df_JK(B_mnQ, Dtot, want_J=True, want_K=False, B2=B2, profile=profile)
                _, Ka = _df_JK(B_mnQ, Da, want_J=False, want_K=True, BQ=BQ, profile=profile)
                _, Kb = _df_JK(B_mnQ, Db, want_J=False, want_K=True, BQ=BQ, profile=profile)
            else:
                if profile is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

                tJ = _time_ms_start(xp) if profile is not None else None
                J = df_jk.df_J_from_BQ_D(BQ, Dtot)
                J = _symmetrize(xp, J)
                if profile is not None and tJ is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

                tK = _time_ms_start(xp) if profile is not None else None
                Ka = df_jk.df_K_from_BQ_D(BQ, Da, profile=profile)
                Kb = df_jk.df_K_from_BQ_D(BQ, Db, profile=profile)
                Ka = _symmetrize(xp, Ka)
                Kb = _symmetrize(xp, Kb)
                if profile is not None and tK is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
            if use_k_cocc:
                Ka_prev = Ka
                Kb_prev = Kb
        if profile is not None:
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        Fa = _symmetrize(xp, h + J - Ka)
        Fb = _symmetrize(xp, h + J - Kb)

        if diis_a is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp)
            ea_mat = _fock_error_rhf(Fa, Da, S)
            diis_a.push(Fa, ea_mat)
            Fa = _symmetrize(xp, diis_a.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)
        if diis_b is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp)
            eb_mat = _fock_error_rhf(Fb, Db, S)
            diis_b.push(Fb, eb_mat)
            Fb = _symmetrize(xp, diis_b.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp)
        ea, Ca = _gen_eigh_with_X(Fa, X)
        eb, Cb = _gen_eigh_with_X(Fb, X)
        if profile is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)
        Da_new = _density_from_C_occ(Ca, occ_a)
        Db_new = _density_from_C_occ(Cb, occ_b)

        if damping:
            Da_new = (1.0 - lam) * Da_new + lam * Da
            Db_new = (1.0 - lam) * Db_new + lam * Db

        Dtot_new = Da_new + Db_new
        e_one = float(xp.trace(Dtot_new @ h).item())
        e_coul = float(0.5 * xp.trace(Dtot_new @ J).item())
        e_ex = float(0.5 * xp.trace(Da_new @ Ka).item() + 0.5 * xp.trace(Db_new @ Kb).item())
        e_elec = e_one + e_coul - e_ex
        e_tot = float(e_elec + float(enuc))

        dm_err = float(xp.linalg.norm(Da_new - Da).item() + xp.linalg.norm(Db_new - Db).item())
        de = float(abs(e_tot - e_last)) if e_last is not None else float("inf")

        if de < float(conv_tol) and dm_err < float(conv_tol_dm):
            converged = True
            Da, Db = Da_new, Db_new
            e_last = e_tot
            break

        Da, Db = Da_new, Db_new
        e_last = e_tot

    if profile is not None:
        profile["scf"]["iters"] = int(cycle)
    return SCFResult(
        method="UHF",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=(ea, eb),
        mo_coeff=(Ca, Cb),
        mo_occ=(occ_a, occ_b),
    )


def rohf_df(
    S,
    hcore,
    B,
    *,
    nalpha: int,
    nbeta: int,
    enuc: float = 0.0,
    max_cycle: int = 50,
    conv_tol: float = 1e-10,
    conv_tol_dm: float = 1e-8,
    diis: bool = True,
    diis_start_cycle: int = 2,
    diis_space: int = 8,
    damping: float = 0.0,
    jk_mode: str = "materialized",
    k_engine: str = "auto",
    k_q_block: int = 128,
    cublas_math_mode: str | None = None,
    ao_basis=None,
    aux_basis=None,
    df_backend: str = "gpu_rys",
    df_threads: int = 256,
    df_mode: str = "auto",
    df_aux_block_naux: int = 256,
    L_metric=None,
    dm0=None,
    mo_coeff0=None,
    profile: dict | None = None,
):
    """ROHF SCF with DF ERIs and a single shared MO coefficient matrix."""

    jk_mode_s = str(jk_mode).strip().lower()
    if jk_mode_s not in {"materialized", "streamed", "auto"}:
        raise ValueError("jk_mode must be one of: 'materialized', 'streamed', 'auto'")
    use_streamed = bool(jk_mode_s == "streamed" or (jk_mode_s == "auto" and B is None))

    if use_streamed:
        try:
            import cupy as cp  # noqa: PLC0415
        except Exception as e:  # pragma: no cover
            raise RuntimeError("jk_mode='streamed' requires CuPy") from e
        xp, is_gpu = cp, True
    else:
        xp, is_gpu = _get_xp(S, hcore, B)
    S = _as_xp(xp, S, dtype=xp.float64)
    h = _as_xp(xp, hcore, dtype=xp.float64)
    nao = int(S.shape[0])
    if S.shape != (nao, nao) or h.shape != (nao, nao):
        raise ValueError("S/hcore must be (nao, nao)")

    X = _orthogonalizer_from_S(S)
    if not use_streamed:
        B_in = _as_xp(xp, B, dtype=xp.float64)
        if B_in.ndim != 3:
            raise ValueError("B must be a 3D array with shape (nao, nao, naux) or (naux, nao, nao)")

        B_mnQ = None
        B2 = None
        if int(B_in.shape[0]) == int(nao) and int(B_in.shape[1]) == int(nao):
            B_mnQ = B_in
            naux = int(B_mnQ.shape[2])
            B2 = B_mnQ.reshape((nao * nao, naux))
            BQ = B_mnQ.transpose((2, 0, 1))
        elif int(B_in.shape[1]) == int(nao) and int(B_in.shape[2]) == int(nao):
            BQ = B_in
            naux = int(BQ.shape[0])
        else:
            raise ValueError("B must have shape (nao, nao, naux) or (naux, nao, nao)")
    else:
        B_mnQ = None
        B2 = None
        BQ = None
        naux = None

    if int(nalpha) < int(nbeta):
        raise ValueError("ROHF requires nalpha >= nbeta")
    occ_a_np, occ_b_np = _occ_uhf(nalpha, nbeta, nao)
    occ_a = _as_xp(xp, occ_a_np, dtype=xp.float64)
    occ_b = _as_xp(xp, occ_b_np, dtype=xp.float64)

    e0, C = _gen_eigh_with_X(h, X)
    if dm0 is not None:
        if not isinstance(dm0, (tuple, list)) or len(dm0) != 2:
            raise TypeError("dm0 for ROHF must be a (Da, Db) tuple")
        Da = _as_xp(xp, dm0[0], dtype=xp.float64)
        Db = _as_xp(xp, dm0[1], dtype=xp.float64)
        if Da.shape != (nao, nao) or Db.shape != (nao, nao):
            raise ValueError("dm0 for ROHF must have shape (nao, nao) for both spins")
        Da = _symmetrize(xp, Da)
        Db = _symmetrize(xp, Db)
    elif mo_coeff0 is not None:
        C0 = _as_xp(xp, mo_coeff0, dtype=xp.float64)
        if C0.shape != (nao, nao):
            raise ValueError("mo_coeff0 for ROHF must have shape (nao, nao)")
        C = C0
        Da = _symmetrize(xp, _density_from_C_occ(C, occ_a))
        Db = _symmetrize(xp, _density_from_C_occ(C, occ_b))
    else:
        Da = _density_from_C_occ(C, occ_a)
        Db = _density_from_C_occ(C, occ_b)

    k_engine_s = str(k_engine).strip().lower()
    if k_engine_s not in {"auto", "from_cocc", "from_d"}:
        raise ValueError("k_engine must be one of: 'auto', 'from_Cocc', 'from_D'")
    use_k_cocc = bool(k_engine_s == "from_cocc" or (k_engine_s == "auto" and bool(is_gpu)))
    if use_streamed and not use_k_cocc:
        raise NotImplementedError("jk_mode='streamed' currently requires k_engine='from_Cocc' (or 'auto' on GPU)")
    if use_streamed and dm0 is not None:
        raise NotImplementedError("jk_mode='streamed' does not support dm0 (needs a C_occ-consistent density)")

    k_q_block = int(k_q_block)
    if k_q_block <= 0:
        raise ValueError("k_q_block must be > 0")
    lam = float(damping) if damping else 0.0
    Ka_prev = None
    Kb_prev = None

    b_layout = "streamed" if use_streamed else ("mnQ" if B_mnQ is not None else "Qmn")
    bq_copied = False
    if (
        (not use_streamed)
        and bool(is_gpu)
        and (BQ is not None)
        and hasattr(BQ, "flags")
        and (not bool(BQ.flags.c_contiguous))
        and (not (use_k_cocc and B_mnQ is not None))
    ):
        BQ = xp.ascontiguousarray(BQ)
        bq_copied = True

    diis_obj = _DIIS(max_vec=diis_space) if diis else None
    e_last = None
    converged = False

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["B_layout"] = str(b_layout)
        jk_prof["BQ_copied"] = bool(bq_copied)
        if bool(is_gpu):
            try:
                pool = xp.get_default_memory_pool()
                jk_prof["mem_pool_used_bytes"] = int(pool.used_bytes())
                jk_prof["mem_pool_total_bytes"] = int(pool.total_bytes())
            except Exception:
                pass

    if profile is not None:
        prof = profile.setdefault("scf", {})
        prof.setdefault("jk_ms", 0.0)
        prof.setdefault("diag_ms", 0.0)
        prof.setdefault("diis_ms", 0.0)
        prof.setdefault("iters", 0)

    if profile is not None and use_k_cocc:
        jk_prof = profile.setdefault("jk", {})
        try:
            if BQ is not None:
                jk_prof.setdefault("BQ_shape", list(map(int, BQ.shape)))
                jk_prof.setdefault("BQ_strides", list(map(int, getattr(BQ, "strides", ()))))
                jk_prof.setdefault(
                    "BQ_c_contig", bool(getattr(BQ, "flags", {}).c_contiguous) if hasattr(BQ, "flags") else None
                )
        except Exception:
            pass

    if use_streamed:
        if ao_basis is None or aux_basis is None:
            raise ValueError("jk_mode='streamed' requires ao_basis and aux_basis")
        from . import df_jk_streamed  # local import to avoid cuERI deps in materialized mode

        ctx = df_jk_streamed.make_streamed_df_jk_context(
            ao_basis,
            aux_basis,
            L_metric=L_metric,
            backend=str(df_backend),
            threads=int(df_threads),
            mode=str(df_mode),
            aux_block_naux=int(df_aux_block_naux),
            profile=profile,
        )
        jk_work: dict = {}

    for cycle in range(1, int(max_cycle) + 1):
        Dtot = Da + Db
        t = _time_ms_start(xp)
        if use_streamed:
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1
                jk_prof["k_impl"] = "streamed_cocc_trsm"
                jk_prof["k_q_block"] = int(k_q_block)
                jk_prof["k_nocc"] = [int(nalpha), int(nbeta)]

            from . import df_jk_streamed  # local import

            # ROHF: beta orbitals are a subset of alpha. Avoid duplicating them by splitting
            # alpha into (beta subset) + (open-shell) and computing two K parts.
            C_beta = xp.ascontiguousarray(C[:, : int(nbeta)])
            C_open = xp.ascontiguousarray(C[:, int(nbeta) : int(nalpha)])
            occ_beta = occ_b[: int(nbeta)]
            occ_open = occ_a[int(nbeta) : int(nalpha)]
            J, Ks = df_jk_streamed.df_JKs_streamed(
                ctx,
                Dtot,
                [C_beta, C_open],
                [occ_beta, occ_open],
                k_q_block=int(k_q_block),
                cublas_math_mode=cublas_math_mode,
                work=jk_work,
                profile=profile,
            )
            Kb_pure = Ks[0]
            Kopen_pure = Ks[1]
            Ka_pure = Kb_pure + Kopen_pure

            if damping and Ka_prev is not None:
                Ka = (1.0 - lam) * Ka_pure + lam * Ka_prev
            else:
                Ka = Ka_pure
            if damping and Kb_prev is not None:
                Kb = (1.0 - lam) * Kb_pure + lam * Kb_prev
            else:
                Kb = Kb_pure
            Ka_prev = Ka
            Kb_prev = Kb
        elif use_k_cocc and not (cycle == 1 and dm0 is not None):
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

            tJ = _time_ms_start(xp) if profile is not None else None
            if B2 is not None:
                J = df_jk.df_J_from_B2_D(B2, Dtot)
            else:
                J = df_jk.df_J_from_BQ_D(BQ, Dtot)
            J = _symmetrize(xp, J)
            if profile is not None and tJ is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

            tK = _time_ms_start(xp) if profile is not None else None
            C_occ_a = xp.ascontiguousarray(C[:, : int(nalpha)])
            C_occ_b = xp.ascontiguousarray(C[:, : int(nbeta)])
            occ_a_vals = occ_a[: int(nalpha)]
            occ_b_vals = occ_b[: int(nbeta)]
            if B_mnQ is not None:
                Ka_pure = df_jk.df_K_from_BmnQ_Cocc(
                    B_mnQ,
                    C_occ_a,
                    occ_a_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
                Kb_pure = df_jk.df_K_from_BmnQ_Cocc(
                    B_mnQ,
                    C_occ_b,
                    occ_b_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
            else:
                Ka_pure = df_jk.df_K_from_BQ_Cocc(
                    BQ,
                    C_occ_a,
                    occ_a_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
                Kb_pure = df_jk.df_K_from_BQ_Cocc(
                    BQ,
                    C_occ_b,
                    occ_b_vals,
                    q_block=int(k_q_block),
                    cublas_math_mode=cublas_math_mode,
                    profile=profile,
                )
            if damping and Ka_prev is not None:
                Ka = (1.0 - lam) * Ka_pure + lam * Ka_prev
            else:
                Ka = Ka_pure
            if damping and Kb_prev is not None:
                Kb = (1.0 - lam) * Kb_pure + lam * Kb_prev
            else:
                Kb = Kb_pure
            Ka = _symmetrize(xp, Ka)
            Kb = _symmetrize(xp, Kb)

            if profile is not None and tK is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
                jk_prof["k_q_block"] = int(k_q_block)
                jk_prof["k_nocc"] = [int(nalpha), int(nbeta)]

            Ka_prev = Ka
            Kb_prev = Kb
        else:
            if B_mnQ is not None:
                J, _ = _df_JK(B_mnQ, Dtot, want_J=True, want_K=False, B2=B2, profile=profile)
                _, Ka = _df_JK(B_mnQ, Da, want_J=False, want_K=True, BQ=BQ, profile=profile)
                _, Kb = _df_JK(B_mnQ, Db, want_J=False, want_K=True, BQ=BQ, profile=profile)
            else:
                if profile is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["calls"] = int(jk_prof.get("calls", 0)) + 1

                tJ = _time_ms_start(xp) if profile is not None else None
                J = df_jk.df_J_from_BQ_D(BQ, Dtot)
                J = _symmetrize(xp, J)
                if profile is not None and tJ is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["j_ms"] = float(jk_prof.get("j_ms", 0.0)) + _time_ms_end(xp, tJ)

                tK = _time_ms_start(xp) if profile is not None else None
                Ka = df_jk.df_K_from_BQ_D(BQ, Da, profile=profile)
                Kb = df_jk.df_K_from_BQ_D(BQ, Db, profile=profile)
                Ka = _symmetrize(xp, Ka)
                Kb = _symmetrize(xp, Kb)
                if profile is not None and tK is not None:
                    jk_prof = profile.setdefault("jk", {})
                    jk_prof["k_ms"] = float(jk_prof.get("k_ms", 0.0)) + _time_ms_end(xp, tK)
            if use_k_cocc:
                Ka_prev = Ka
                Kb_prev = Kb
        if profile is not None:
            profile["scf"]["jk_ms"] += _time_ms_end(xp, t)

        Fa = _symmetrize(xp, h + J - Ka)
        Fb = _symmetrize(xp, h + J - Kb)
        F = _roothaan_fock_rohf(Fa, Fb, Da, Db, S)

        if diis_obj is not None and cycle >= int(diis_start_cycle):
            t = _time_ms_start(xp)
            e = _fock_error_rhf(F, Dtot, S)
            diis_obj.push(F, e)
            F = _symmetrize(xp, diis_obj.extrapolate())
            if profile is not None:
                profile["scf"]["diis_ms"] += _time_ms_end(xp, t)

        t = _time_ms_start(xp)
        e0, C = _gen_eigh_with_X(F, X)
        if profile is not None:
            profile["scf"]["diag_ms"] += _time_ms_end(xp, t)
        Da_new = _density_from_C_occ(C, occ_a)
        Db_new = _density_from_C_occ(C, occ_b)

        if damping:
            Da_new = (1.0 - lam) * Da_new + lam * Da
            Db_new = (1.0 - lam) * Db_new + lam * Db

        Dtot_new = Da_new + Db_new

        e_one = float(xp.trace(Dtot_new @ h).item())
        e_coul = float(0.5 * xp.trace(Dtot_new @ J).item())
        e_ex = float(0.5 * xp.trace(Da_new @ Ka).item() + 0.5 * xp.trace(Db_new @ Kb).item())
        e_elec = e_one + e_coul - e_ex
        e_tot = float(e_elec + float(enuc))

        dm_err = float(xp.linalg.norm(Da_new - Da).item() + xp.linalg.norm(Db_new - Db).item())
        de = float(abs(e_tot - e_last)) if e_last is not None else float("inf")

        if de < float(conv_tol) and dm_err < float(conv_tol_dm):
            converged = True
            Da, Db = Da_new, Db_new
            e_last = e_tot
            break

        Da, Db = Da_new, Db_new
        e_last = e_tot

    if profile is not None:
        profile["scf"]["iters"] = int(cycle)
    return SCFResult(
        method="ROHF",
        converged=bool(converged),
        niter=int(cycle),
        e_tot=float(e_last if e_last is not None else float("nan")),
        e_elec=float(e_last - float(enuc)) if e_last is not None else float("nan"),
        e_nuc=float(enuc),
        mo_energy=e0,
        mo_coeff=C,
        mo_occ=(occ_a, occ_b),
    )


__all__ = ["SCFResult", "rhf_df", "rohf_df", "uhf_df"]
