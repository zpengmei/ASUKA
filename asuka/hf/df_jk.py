from __future__ import annotations

"""DF J/K helpers.

This module provides fast, SCF-friendly DF J/K contractions that can be reused
by AO-basis SCF drivers without pulling SCF control-flow into the math kernels.
"""

from contextlib import contextmanager
import os
from typing import Any, Iterator

import numpy as np


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
    return xp.asarray(a, dtype=dtype)


def _symmetrize(xp, A):
    return 0.5 * (A + A.T)


_HF_DF_JK_CUDA_EXT = None


def _load_hf_df_jk_cuda_ext():
    global _HF_DF_JK_CUDA_EXT
    if _HF_DF_JK_CUDA_EXT is False:
        return None
    if _HF_DF_JK_CUDA_EXT is not None:
        return _HF_DF_JK_CUDA_EXT
    try:
        from asuka import _hf_df_jk_cuda_ext  # type: ignore[import-not-found]
    except Exception:
        _HF_DF_JK_CUDA_EXT = False
        return None
    _HF_DF_JK_CUDA_EXT = _hf_df_jk_cuda_ext
    return _HF_DF_JK_CUDA_EXT


_HF_DF_JK_WS_BY_DEVICE: dict[int, Any] = {}
_HF_DF_JK_BQ_CACHE_BY_DEVICE: dict[int, tuple[tuple[int, int, int, int], Any]] = {}
_HF_DF_JK_QBLOCK_TUNE_BY_DEVICE: dict[int, dict[tuple[Any, ...], int]] = {}


def _get_hf_df_jk_workspace(cp):
    dev = int(cp.cuda.runtime.getDevice())
    ws = _HF_DF_JK_WS_BY_DEVICE.get(dev)
    if ws is None:
        ext = _load_hf_df_jk_cuda_ext()
        if ext is None:  # pragma: no cover
            raise RuntimeError("HF DF-JK CUDA extension is not available")
        ws = ext.DFJKWorkspace()
        _HF_DF_JK_WS_BY_DEVICE[dev] = ws
    return ws


def _hf_k_impl_pref() -> str:
    return str(os.environ.get("ASUKA_HF_K_IMPL", "auto")).strip().lower()


def _hf_k_ext_tune_enabled() -> bool:
    flag = str(os.environ.get("ASUKA_HF_K_EXT_TUNE_QBLOCK", "1")).strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _hf_k_ext_autotune_enabled() -> bool:
    flag = str(os.environ.get("ASUKA_HF_K_EXT_AUTOTUNE_QBLOCK", "1")).strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _hf_k_cache_mnq_enabled() -> bool:
    flag = str(os.environ.get("ASUKA_HF_K_CACHE_MNQ_TO_BQ", "1")).strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _hf_k_cache_max_bytes() -> int:
    try:
        mb = int(os.environ.get("ASUKA_HF_K_CACHE_MAX_MB", "1024"))
    except Exception:
        mb = 1024
    return max(64, int(mb)) * 1024 * 1024


def _hf_k_ext_work_max_bytes() -> int:
    try:
        mb = int(os.environ.get("ASUKA_HF_K_EXT_WORK_MAX_MB", "512"))
    except Exception:
        mb = 512
    return max(64, int(mb)) * 1024 * 1024


def _pick_ext_q_block(cp, *, layout: str, naux: int, nao: int, nocc: int, requested: int) -> int:
    q_req = max(1, int(requested))
    if int(naux) <= 0:
        return q_req
    if int(naux) <= q_req:
        return int(naux)

    if str(layout) == "bq":
        if int(nao) <= 140:
            target = 224
        elif int(nao) <= 170:
            target = 224
        else:
            target = 224
    else:
        if int(nao) <= 170:
            target = 216
        else:
            target = 192

    q = max(q_req, min(int(naux), int(target)))

    try:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        budget = min(_hf_k_ext_work_max_bytes(), max(64 * 1024 * 1024, int(0.35 * int(free_bytes))))
    except Exception:
        budget = _hf_k_ext_work_max_bytes()

    if str(layout) == "bq":
        bytes_per_q = 2 * int(nao) * int(nocc) * 8
    else:
        bytes_per_q = int(nao) * (int(nocc) + max(int(nao), int(nocc))) * 8
    if bytes_per_q <= 0:
        return q

    q_mem = max(1, int(budget // bytes_per_q))
    return max(q_req, min(int(naux), int(q), int(q_mem)))


def _qblock_candidates(*, layout: str, naux: int, requested: int, guess: int) -> list[int]:
    g = int(guess)
    cand: list[int] = [int(requested), g, g - 32, g - 16, g + 16, g + 32]
    if str(layout) == "bq":
        cand.extend([160, 176, 192, 208, 216, 224, 240, 256, 320, 384, 448])
    else:
        cand.extend([128, 160, 176, 192, 200, 208, 216, 224, 240, 256, 320])
    out: list[int] = []
    seen: set[int] = set()
    for q in cand:
        qq = max(1, min(int(naux), int(q)))
        if qq not in seen:
            seen.add(qq)
            out.append(qq)
    return out


def _autotune_q_block_ext(
    xp,
    ws,
    *,
    mode: str,
    in_arr,
    Cw,
    q_req: int,
    q_guess: int,
    naux: int,
    nao: int,
    nocc: int,
    math_mode: int,
) -> int:
    if xp is np:
        return int(q_guess)
    if not _hf_k_ext_autotune_enabled():
        return int(q_guess)

    dev = int(xp.cuda.runtime.getDevice())
    tune_cache = _HF_DF_JK_QBLOCK_TUNE_BY_DEVICE.setdefault(dev, {})
    key = (
        str(mode),
        int(naux),
        int(nao),
        int(nocc),
        int(q_req),
        int(math_mode),
    )
    cached = tune_cache.get(key)
    if cached is not None:
        return int(cached)

    cand = _qblock_candidates(layout=("bq" if str(mode).startswith("bq") else "mnq"), naux=int(naux), requested=int(q_req), guess=int(q_guess))
    if len(cand) <= 1:
        tune_cache[key] = int(cand[0]) if cand else int(q_guess)
        return int(tune_cache[key])

    stream_ptr = int(xp.cuda.get_current_stream().ptr)
    Ktmp = xp.empty((int(nao), int(nao)), dtype=xp.float64)
    ev0 = xp.cuda.Event()
    ev1 = xp.cuda.Event()

    def run_once(q: int):
        if str(mode) == "bq":
            ws.k_from_bq_cw(
                in_arr,
                Cw,
                Ktmp,
                q_block=int(q),
                stream=int(stream_ptr),
                math_mode=int(math_mode),
                sync=False,
            )
        elif str(mode) == "bq_cached":
            ws.k_from_bq_cw(
                in_arr,
                Cw,
                Ktmp,
                q_block=int(q),
                stream=int(stream_ptr),
                math_mode=int(math_mode),
                sync=False,
            )
        else:
            ws.k_from_bmnq_cw(
                in_arr,
                Cw,
                Ktmp,
                q_block=int(q),
                stream=int(stream_ptr),
                math_mode=int(math_mode),
                sync=False,
            )

    scored: list[tuple[float, int]] = []
    for q in cand:
        try:
            # Warm candidate and run a short first pass.
            run_once(int(q))
            samples: list[float] = []
            for _ in range(2):
                ev0.record()
                run_once(int(q))
                ev1.record()
                ev1.synchronize()
                samples.append(float(xp.cuda.get_elapsed_time(ev0, ev1)))
            samples.sort()
            ms = float(samples[len(samples) // 2])
        except Exception:
            continue
        scored.append((ms, int(q)))

    if not scored:
        tune_cache[key] = int(q_guess)
        return int(q_guess)

    scored.sort(key=lambda x: x[0])
    finalists = [qv for _, qv in scored[: min(3, len(scored))]]

    best_q = int(finalists[0])
    best_ms = float("inf")
    for q in finalists:
        try:
            run_once(int(q))
            samples: list[float] = []
            for _ in range(5):
                ev0.record()
                run_once(int(q))
                ev1.record()
                ev1.synchronize()
                samples.append(float(xp.cuda.get_elapsed_time(ev0, ev1)))
            samples.sort()
            ms = float(samples[len(samples) // 2])
        except Exception:
            continue
        if ms < best_ms:
            best_ms = ms
            best_q = int(q)

    tune_cache[key] = int(best_q)
    return int(best_q)


def _cached_bq_from_mnq(cp, B_mnQ, *, nao: int, naux: int):
    if not _hf_k_cache_mnq_enabled():
        return None

    bq_bytes = int(nao) * int(nao) * int(naux) * 8
    if bq_bytes > _hf_k_cache_max_bytes():
        return None

    dev = int(cp.cuda.runtime.getDevice())
    ptr = int(getattr(getattr(B_mnQ, "data", None), "ptr", 0))
    if ptr == 0:
        return None

    key = (int(ptr), int(nao), int(nao), int(naux))
    cached = _HF_DF_JK_BQ_CACHE_BY_DEVICE.get(dev)
    if cached is not None and cached[0] == key:
        return cached[1]

    BQ = cp.ascontiguousarray(B_mnQ.transpose((2, 0, 1)))
    _HF_DF_JK_BQ_CACHE_BY_DEVICE[dev] = (key, BQ)
    return BQ


def _cublas_math_mode_to_int(xp, cublas_math_mode: str | None) -> int:
    if cublas_math_mode is None or xp is np:
        return -1
    mode = str(cublas_math_mode).lower().strip()
    if mode == "default":
        return 0
    if mode == "fp64_emulated_fixedpoint":
        # cuBLAS 13.x: CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
        return 8
    raise ValueError("cublas_math_mode must be one of: None, 'default', 'fp64_emulated_fixedpoint'")


@contextmanager
def _cublas_math_mode_ctx(xp, cublas_math_mode: str | None) -> Iterator[None]:
    """Temporarily set cuBLAS math mode on the current CuPy handle.

    - None: no-op
    - "default": CUBLAS_DEFAULT_MATH
    - "fp64_emulated_fixedpoint": CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (requires CUDA 13 / cuBLAS 13.x)
    """

    if cublas_math_mode is None or xp is np:
        yield
        return

    import cupy as cp  # noqa: PLC0415
    from cupy_backends.cuda.libs import cublas as cublas_lib  # noqa: PLC0415

    mode = str(cublas_math_mode).lower()
    handle = int(cp.cuda.get_cublas_handle())

    if mode == "default":
        new_math_mode = int(cublas_lib.CUBLAS_DEFAULT_MATH)
    elif mode == "fp64_emulated_fixedpoint":
        ver = int(cublas_lib.getVersion(handle))
        if ver < 130000:
            raise RuntimeError(
                "cublas_math_mode='fp64_emulated_fixedpoint' requires cuBLAS 13.x (CUDA 13.0+); "
                f"detected cublas version={ver}"
            )
        # cublasMath_t: CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH (CUDA 13.0+)
        new_math_mode = 8
    else:
        raise ValueError("cublas_math_mode must be one of: None, 'default', 'fp64_emulated_fixedpoint'")

    old_math_mode = int(cublas_lib.getMathMode(handle))
    if new_math_mode == old_math_mode:
        yield
        return

    cublas_lib.setMathMode(handle, int(new_math_mode))
    try:
        yield
    finally:
        cublas_lib.setMathMode(handle, int(old_math_mode))


def df_J_from_B2_D(B2, D):
    """Dense DF-J from a (nao*nao, naux) view B2 and density D."""

    xp, _ = _get_xp(B2, D)
    B2 = _as_xp(xp, B2, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if B2.ndim != 2:
        raise ValueError("B2 must be a 2D array")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    nao = int(D.shape[0])
    if int(B2.shape[0]) != int(nao * nao):
        raise ValueError("B2 and D nao mismatch")

    naux = int(B2.shape[1])
    dvec = D.reshape((nao * nao,))
    v = B2.T @ dvec  # (naux,)
    J = (B2 @ v).reshape((nao, nao))
    return J


def df_J_from_BQ_D(BQ, D):
    """DF-J from BQ layout without forming B_mnQ.

    Inputs
    - BQ: (naux, nao, nao), float64, C-contiguous preferred
    - D: (nao, nao), float64
    """

    xp, _ = _get_xp(BQ, D)
    BQ = _as_xp(xp, BQ, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if BQ.ndim != 3:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    naux, nao0, nao1 = map(int, BQ.shape)
    if nao0 != nao1:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    nao = int(D.shape[0])
    if nao != nao0:
        raise ValueError("BQ and D nao mismatch")

    BQ2 = BQ.reshape((naux, nao * nao))  # view if BQ is contiguous
    d = BQ2 @ D.reshape((nao * nao,))  # (naux,)
    J = (BQ2.T @ d).reshape((nao, nao))
    return J


def df_K_from_BQ_D(BQ, D, *, profile: dict | None = None):
    """DF-K from BQ layout and density D (no B_mnQ required).

    Inputs
    - BQ: (naux, nao, nao), float64
    - D: (nao, nao), float64
    """

    xp, _ = _get_xp(BQ, D)
    BQ = _as_xp(xp, BQ, dtype=xp.float64)
    D = _as_xp(xp, D, dtype=xp.float64)
    if BQ.ndim != 3:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square 2D matrix")
    naux, nao0, nao1 = map(int, BQ.shape)
    if nao0 != nao1:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    nao = int(D.shape[0])
    if nao != nao0:
        raise ValueError("BQ and D nao mismatch")

    use_chunked = (xp is not np) and (int(naux) * int(nao) * int(nao) >= 10_000_000)
    chunk = 128
    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["k_impl"] = "chunked_einsum" if use_chunked else "batched_matmul"
        if use_chunked:
            jk_prof["k_chunk_naux"] = int(chunk)

    if use_chunked:
        K = xp.zeros((nao, nao), dtype=xp.float64)
        for q0 in range(0, naux, int(chunk)):
            q1 = min(int(naux), int(q0) + int(chunk))
            BQc = BQ[int(q0) : int(q1)]
            BDc = xp.matmul(BQc, D)  # (q, nao, nao)
            K += xp.einsum("qik,qjk->ij", BDc, BQc)
        BD = None
        KQ = None
    else:
        BD = xp.matmul(BQ, D)  # (naux, nao, nao)
        KQ = xp.matmul(BD, BQ.transpose((0, 2, 1)))  # (naux, nao, nao)
        K = xp.sum(KQ, axis=0)

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        try:
            jk_prof.setdefault("BD_nbytes", int(getattr(BD, "nbytes", 0) if BD is not None else 0))
            jk_prof.setdefault("KQ_nbytes", int(getattr(KQ, "nbytes", 0) if KQ is not None else 0))
        except Exception:
            pass

    return _symmetrize(xp, K)


def df_K_from_BQ_Cocc(
    BQ,
    C_occ,
    occ_vals,
    *,
    q_block: int = 128,
    cublas_math_mode: str | None = None,
    profile: dict | None = None,
):
    """Occupied-driven DF exchange (RI-K) without a dense D.

    Computes:
      K_{μν} = Σ_i occ_i Σ_Q (μi|Q) (νi|Q)
    with:
      (μi|Q) = Σ_λ B_{μλ,Q} C_{λi}

    Inputs
    - BQ: (naux, nao, nao) contiguous, float64
    - C_occ: (nao, nocc) float64 (need not be contiguous; copied if needed)
    - occ_vals: (nocc,) float64
    """

    xp, _ = _get_xp(BQ, C_occ, occ_vals)
    BQ = _as_xp(xp, BQ, dtype=xp.float64)
    C_occ = _as_xp(xp, C_occ, dtype=xp.float64)
    occ_vals = _as_xp(xp, occ_vals, dtype=xp.float64).ravel()

    if BQ.ndim != 3:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    if C_occ.ndim != 2:
        raise ValueError("C_occ must be 2D")
    if occ_vals.ndim != 1:
        raise ValueError("occ_vals must be 1D")

    naux, nao0, nao1 = map(int, BQ.shape)
    if nao0 != nao1:
        raise ValueError("BQ must have shape (naux, nao, nao)")
    nao = int(nao0)
    if int(C_occ.shape[0]) != nao:
        raise ValueError("C_occ nao mismatch with BQ")
    nocc = int(C_occ.shape[1])
    if int(occ_vals.shape[0]) != nocc:
        raise ValueError(f"occ_vals must have shape ({nocc},), got {tuple(map(int, occ_vals.shape))}")
    if nocc <= 0:
        return xp.zeros((nao, nao), dtype=xp.float64)

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")

    # Weight occupied orbitals by sqrt(occ) so K becomes a simple sum of outer products.
    sqrt_occ = xp.sqrt(occ_vals)
    Cw = C_occ * sqrt_occ[None, :]
    if hasattr(Cw, "flags") and not bool(Cw.flags.c_contiguous):
        Cw = xp.ascontiguousarray(Cw)

    pref = _hf_k_impl_pref()
    if pref not in {"auto", "cupy", "cuda_ext"}:
        raise ValueError("ASUKA_HF_K_IMPL must be one of: 'auto', 'cupy', 'cuda_ext'")

    use_ext = False
    ext = None
    if xp is not np and pref in {"auto", "cuda_ext"}:
        ext = _load_hf_df_jk_cuda_ext()
        use_ext = bool(ext is not None)
        if pref == "cuda_ext" and not use_ext:
            raise RuntimeError(
                "ASUKA_HF_K_IMPL='cuda_ext' was set but asuka._hf_df_jk_cuda_ext is not available. "
                "Reinstall ASUKA with a CUDA toolkit available (nvcc)."
            )

    if use_ext:
        if hasattr(BQ, "flags") and not bool(BQ.flags.c_contiguous):
            BQ = xp.ascontiguousarray(BQ)
        ws = _get_hf_df_jk_workspace(xp)
        stream_ptr = int(xp.cuda.get_current_stream().ptr)
        math_mode = _cublas_math_mode_to_int(xp, cublas_math_mode)
        q_guess = int(q_block)
        if _hf_k_ext_tune_enabled():
            q_guess = _pick_ext_q_block(
                xp, layout="bq", naux=int(naux), nao=int(nao), nocc=int(nocc), requested=int(q_block)
            )

        q_block_ext = int(q_guess)
        if _hf_k_ext_tune_enabled():
            q_block_ext = _autotune_q_block_ext(
                xp,
                ws,
                mode="bq",
                in_arr=BQ,
                Cw=Cw,
                q_req=int(q_block),
                q_guess=int(q_block_ext),
                naux=int(naux),
                nao=int(nao),
                nocc=int(nocc),
                math_mode=int(math_mode),
            )
        K = xp.empty((nao, nao), dtype=xp.float64)
        ws.k_from_bq_cw(
            BQ,
            Cw,
            K,
            q_block=int(q_block_ext),
            stream=int(stream_ptr),
            math_mode=int(math_mode),
            sync=False,
        )
        if profile is not None:
            jk_prof = profile.setdefault("jk", {})
            jk_prof["k_impl"] = "cocc_cuda_ext_gemm_transpose_gemm"
            jk_prof["k_q_block"] = int(q_block_ext)
            jk_prof["k_q_block_req"] = int(q_block)
            jk_prof["k_nocc"] = int(nocc)
        return K

    # CuPy fallback: einsum + GEMM (allocates intermediates).
    K = xp.zeros((nao, nao), dtype=xp.float64)
    with _cublas_math_mode_ctx(xp, cublas_math_mode):
        for q0 in range(0, int(naux), int(q_block)):
            q1 = min(int(naux), int(q0) + int(q_block))
            BQc = BQ[int(q0) : int(q1)]  # (q, nao, nao)
            q = int(q1 - q0)

            # (q, μ, ν) x (ν, i) -> (μ, q, i)
            tmp = xp.einsum("qmn,ni->mqi", BQc, Cw, optimize=True)
            if hasattr(tmp, "flags") and not bool(tmp.flags.c_contiguous):
                tmp = xp.ascontiguousarray(tmp)

            U = tmp.reshape((nao, q * nocc))
            K += U @ U.T

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["k_impl"] = "cocc_cupy_einsum"
        jk_prof["k_q_block"] = int(q_block)
        jk_prof["k_nocc"] = int(nocc)

    return _symmetrize(xp, K)


def df_K_from_BmnQ_Cocc(
    B_mnQ,
    C_occ,
    occ_vals,
    *,
    q_block: int = 128,
    cublas_math_mode: str | None = None,
    profile: dict | None = None,
):
    """Occupied-driven DF exchange (RI-K) from mnQ layout without forming BQ.

    Computes:
      K_{μν} = Σ_i occ_i Σ_Q (μi|Q) (νi|Q)
    with:
      (μi|Q) = Σ_λ B_{μλ,Q} C_{λi}

    Inputs
    - B_mnQ: (nao, nao, naux) contiguous, float64 preferred
    - C_occ: (nao, nocc) float64 (need not be contiguous; copied if needed)
    - occ_vals: (nocc,) float64
    """

    xp, _ = _get_xp(B_mnQ, C_occ, occ_vals)
    B_mnQ = _as_xp(xp, B_mnQ, dtype=xp.float64)
    C_occ = _as_xp(xp, C_occ, dtype=xp.float64)
    occ_vals = _as_xp(xp, occ_vals, dtype=xp.float64).ravel()

    if B_mnQ.ndim != 3:
        raise ValueError("B_mnQ must have shape (nao, nao, naux)")
    if C_occ.ndim != 2:
        raise ValueError("C_occ must be 2D")
    if occ_vals.ndim != 1:
        raise ValueError("occ_vals must be 1D")

    nao0, nao1, naux = map(int, B_mnQ.shape)
    if nao0 != nao1:
        raise ValueError("B_mnQ must have shape (nao, nao, naux)")
    nao = int(nao0)
    if int(C_occ.shape[0]) != nao:
        raise ValueError("C_occ nao mismatch with B_mnQ")
    nocc = int(C_occ.shape[1])
    if int(occ_vals.shape[0]) != nocc:
        raise ValueError(f"occ_vals must have shape ({nocc},), got {tuple(map(int, occ_vals.shape))}")
    if nocc <= 0:
        return xp.zeros((nao, nao), dtype=xp.float64)

    q_block = int(q_block)
    if q_block <= 0:
        raise ValueError("q_block must be > 0")

    # Weight occupied orbitals by sqrt(occ) so K becomes a simple sum of outer products.
    sqrt_occ = xp.sqrt(occ_vals)
    Cw = C_occ * sqrt_occ[None, :]
    if hasattr(Cw, "flags") and not bool(Cw.flags.c_contiguous):
        Cw = xp.ascontiguousarray(Cw)

    pref = _hf_k_impl_pref()
    if pref not in {"auto", "cupy", "cuda_ext"}:
        raise ValueError("ASUKA_HF_K_IMPL must be one of: 'auto', 'cupy', 'cuda_ext'")

    use_ext = False
    if xp is not np and pref in {"auto", "cuda_ext"}:
        use_ext = bool(_load_hf_df_jk_cuda_ext() is not None)
        if pref == "cuda_ext" and not use_ext:
            raise RuntimeError(
                "ASUKA_HF_K_IMPL='cuda_ext' was set but asuka._hf_df_jk_cuda_ext is not available. "
                "Reinstall ASUKA with a CUDA toolkit available (nvcc)."
            )

    if use_ext:
        ws = _get_hf_df_jk_workspace(xp)
        stream_ptr = int(xp.cuda.get_current_stream().ptr)
        math_mode = _cublas_math_mode_to_int(xp, cublas_math_mode)

        # Reuse a one-time mnQ->BQ transform when affordable; this removes repeated
        # per-block pack kernels across SCF iterations and improves extension throughput.
        BQ_cached = _cached_bq_from_mnq(xp, B_mnQ, nao=int(nao), naux=int(naux))
        if BQ_cached is not None:
            q_block_cached = int(q_block)
            q_guess_cached = int(q_block_cached)
            if _hf_k_ext_tune_enabled():
                q_guess_cached = _pick_ext_q_block(
                    xp, layout="bq", naux=int(naux), nao=int(nao), nocc=int(nocc), requested=int(q_block)
                )
                q_block_cached = int(q_guess_cached)

            if _hf_k_ext_tune_enabled():
                q_block_cached = _autotune_q_block_ext(
                    xp,
                    ws,
                    mode="bq_cached",
                    in_arr=BQ_cached,
                    Cw=Cw,
                    q_req=int(q_block),
                    q_guess=int(q_block_cached),
                    naux=int(naux),
                    nao=int(nao),
                    nocc=int(nocc),
                    math_mode=int(math_mode),
                )

            K = xp.empty((nao, nao), dtype=xp.float64)
            ws.k_from_bq_cw(
                BQ_cached,
                Cw,
                K,
                q_block=int(q_block_cached),
                stream=int(stream_ptr),
                math_mode=int(math_mode),
                sync=False,
            )
            if profile is not None:
                jk_prof = profile.setdefault("jk", {})
                jk_prof["k_impl"] = "cocc_cuda_ext_cached_bq_gemm_transpose_gemm"
                jk_prof["k_q_block"] = int(q_block_cached)
                jk_prof["k_q_block_req"] = int(q_block)
                jk_prof["k_nocc"] = int(nocc)
                jk_prof["k_used_cached_bq"] = True
            return K

        if hasattr(B_mnQ, "flags") and not bool(B_mnQ.flags.c_contiguous):
            B_mnQ = xp.ascontiguousarray(B_mnQ)
        q_block_ext = int(q_block)
        q_guess_mnq = int(q_block_ext)
        if _hf_k_ext_tune_enabled():
            q_guess_mnq = _pick_ext_q_block(
                xp, layout="mnq", naux=int(naux), nao=int(nao), nocc=int(nocc), requested=int(q_block)
            )
            q_block_ext = int(q_guess_mnq)

        if _hf_k_ext_tune_enabled():
            q_block_ext = _autotune_q_block_ext(
                xp,
                ws,
                mode="mnq",
                in_arr=B_mnQ,
                Cw=Cw,
                q_req=int(q_block),
                q_guess=int(q_block_ext),
                naux=int(naux),
                nao=int(nao),
                nocc=int(nocc),
                math_mode=int(math_mode),
            )
        K = xp.empty((nao, nao), dtype=xp.float64)
        ws.k_from_bmnq_cw(
            B_mnQ,
            Cw,
            K,
            q_block=int(q_block_ext),
            stream=int(stream_ptr),
            math_mode=int(math_mode),
            sync=False,
        )
        if profile is not None:
            jk_prof = profile.setdefault("jk", {})
            jk_prof["k_impl"] = "cocc_cuda_ext_pack_bmnq_gemm_transpose_gemm"
            jk_prof["k_q_block"] = int(q_block_ext)
            jk_prof["k_q_block_req"] = int(q_block)
            jk_prof["k_nocc"] = int(nocc)
            jk_prof["k_used_cached_bq"] = False
        return K

    # CuPy fallback: einsum + GEMM (allocates intermediates).
    K = xp.zeros((nao, nao), dtype=xp.float64)
    with _cublas_math_mode_ctx(xp, cublas_math_mode):
        for q0 in range(0, int(naux), int(q_block)):
            q1 = min(int(naux), int(q0) + int(q_block))
            Bc = B_mnQ[:, :, int(q0) : int(q1)]  # (nao, nao, q)
            q = int(q1 - q0)

            # (μ, ν, q) x (ν, i) -> (μ, q, i)
            tmp = xp.einsum("mnq,ni->mqi", Bc, Cw, optimize=True)
            if hasattr(tmp, "flags") and not bool(tmp.flags.c_contiguous):
                tmp = xp.ascontiguousarray(tmp)

            U = tmp.reshape((nao, q * nocc))
            K += U @ U.T

    if profile is not None:
        jk_prof = profile.setdefault("jk", {})
        jk_prof["k_impl"] = "cocc_cupy_einsum"
        jk_prof["k_q_block"] = int(q_block)
        jk_prof["k_nocc"] = int(nocc)

    return _symmetrize(xp, K)


__all__ = [
    "df_J_from_B2_D",
    "df_J_from_BQ_D",
    "df_K_from_BQ_D",
    "df_K_from_BQ_Cocc",
    "df_K_from_BmnQ_Cocc",
]
