from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time
from typing import Any


_CACHE_VERSION = 1
_MEM_CACHE: dict[str, Any] | None = None


def _bool_env(key: str, *, default: bool) -> bool:
    val = os.getenv(key)
    if val is None:
        return bool(default)
    s = str(val).strip().lower()
    if s in ("", "1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    return bool(default)


def _cache_path() -> Path:
    override = os.getenv("CUGUGA_CUDA_AUTOTUNE_CACHE")
    if override:
        return Path(override).expanduser()
    base = os.getenv("XDG_CACHE_HOME")
    if base:
        root = Path(base).expanduser()
    else:
        root = Path.home() / ".cache"
    return root / "asuka" / "autotune_cuda.json"


def _load_cache() -> dict[str, Any]:
    global _MEM_CACHE  # noqa: PLW0603
    if _MEM_CACHE is not None:
        return _MEM_CACHE

    path = _cache_path()
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if not isinstance(data, dict):
            data = {}
    except FileNotFoundError:
        data = {}
    except Exception:
        data = {}

    if int(data.get("version", 0) or 0) != _CACHE_VERSION:
        data = {"version": _CACHE_VERSION}

    _MEM_CACHE = data
    return data


def _save_cache(cache: dict[str, Any]) -> None:
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        # Best-effort cache; never fail the solver due to file IO.
        return


def _cuda_device_key(cp) -> str:
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(int(dev.id))
    name_val = props.get("name", b"")
    name = (
        name_val.decode(errors="ignore")
        if isinstance(name_val, (bytes, bytearray))
        else str(name_val)
    )
    major = props.get("major", None)
    minor = props.get("minor", None)
    cc = f"{major}.{minor}" if major is not None and minor is not None else "unknown"
    try:
        drv = int(cp.cuda.runtime.driverGetVersion())
    except Exception:
        drv = -1
    try:
        rt = int(cp.cuda.runtime.runtimeGetVersion())
    except Exception:
        rt = -1
    return f"cuda:{int(dev.id)}|{name}|cc={cc}|drv={drv}|rt={rt}"


def autotune_matvec_cuda_threads_enum(
    drt,
    drt_dev,
    state_dev,
    *,
    default: int = 128,
    profile: dict[str, Any] | None = None,
) -> int:
    """Pick a good default for `matvec_cuda_threads_enum` and cache it on disk.

    This is intended to eliminate user-facing tuning for the CUDA matvec/CSR build
    enumeration kernels. The tuner is intentionally conservative: it returns
    `default` if anything looks suspicious (missing deps, timing noise, etc.).

    Environment variables
    ---------------------
    - `CUGUGA_CUDA_AUTOTUNE=0|1`: disable/enable autotune (default: enabled).
    - `CUGUGA_CUDA_AUTOTUNE_FORCE=0|1`: ignore cache and retune.
    - `CUGUGA_CUDA_AUTOTUNE_CACHE=/path/to/file.json`: override cache file path.
    """
    if not _bool_env("CUGUGA_CUDA_AUTOTUNE", default=True):
        if profile is not None:
            profile["autotune_enabled"] = False
        return int(default)

    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        if profile is not None:
            profile["autotune_error"] = "cupy_missing"
        return int(default)

    try:
        from asuka.cuda.cuda_backend import (  # noqa: PLC0415
            epq_contribs_many_count_allpairs_inplace_device,
            has_epq_table_device_build,
        )
    except Exception:
        if profile is not None:
            profile["autotune_error"] = "cuda_backend_import_failed"
        return int(default)

    if not bool(has_epq_table_device_build()):
        if profile is not None:
            profile["autotune_error"] = "missing_epq_device_kernels"
        return int(default)

    try:
        device_key = _cuda_device_key(cp)
    except Exception:
        if profile is not None:
            profile["autotune_error"] = "device_query_failed"
        return int(default)

    norb = int(getattr(drt, "norb", 0) or 0)
    ncsf = int(getattr(drt, "ncsf", 0) or 0)
    if norb <= 1 or ncsf <= 0:
        return int(default)

    problem_key = f"norb={norb}"

    cache = _load_cache()
    tbl = cache.setdefault("matvec_cuda_threads_enum", {})
    if not isinstance(tbl, dict):
        tbl = {}
        cache["matvec_cuda_threads_enum"] = tbl
    dev_tbl = tbl.setdefault(device_key, {})
    if not isinstance(dev_tbl, dict):
        dev_tbl = {}
        tbl[device_key] = dev_tbl

    force = _bool_env("CUGUGA_CUDA_AUTOTUNE_FORCE", default=False)
    if not force:
        hit = dev_tbl.get(problem_key)
        if isinstance(hit, dict) and isinstance(hit.get("best"), int):
            best = int(hit["best"])
            if profile is not None:
                profile["autotune_hit"] = True
                profile["autotune_device_key"] = device_key
                profile["autotune_problem_key"] = problem_key
                profile["autotune_best"] = best
            return best

    candidates = [128, 192, 256]
    candidates = [int(x) for x in candidates if int(x) > 0]
    # Ensure default is considered (and prefer it when within noise).
    if int(default) not in candidates:
        candidates.insert(0, int(default))

    # Bound the temporary counts buffer size.
    max_counts_mib = 8.0
    try:
        max_counts_mib = float(os.getenv("CUGUGA_CUDA_AUTOTUNE_COUNTS_MIB", str(max_counts_mib)))
    except Exception:
        max_counts_mib = 8.0
    max_counts_bytes = int(max(1.0, max_counts_mib) * 1024 * 1024)

    n_pairs = norb * (norb - 1)
    if n_pairs <= 0:
        return int(default)

    max_ntasks = max(1, max_counts_bytes // 4)
    j_count = max(64, min(ncsf, max_ntasks // n_pairs))
    j_count = min(ncsf, int(j_count))
    if j_count <= 0:
        return int(default)

    if profile is not None:
        profile["autotune_hit"] = False
        profile["autotune_device_key"] = device_key
        profile["autotune_problem_key"] = problem_key
        profile["autotune_candidates"] = list(candidates)
        profile["autotune_j_count"] = int(j_count)
        profile["autotune_counts_mib"] = float(int(j_count) * int(n_pairs) * 4 / (1024 * 1024))

    # Allocate once and reuse to avoid timing alloc/free noise.
    counts = cp.empty((int(j_count) * int(n_pairs),), dtype=cp.int32)
    overflow = cp.empty((1,), dtype=cp.int32)
    stream = cp.cuda.get_current_stream()
    stream.synchronize()

    def _time_threads(threads: int) -> float:
        # Warmup.
        epq_contribs_many_count_allpairs_inplace_device(
            drt,
            drt_dev,
            state_dev,
            j_start=0,
            j_count=int(j_count),
            counts=counts,
            overflow=overflow,
            threads=int(threads),
            stream=int(stream.ptr),
            sync=False,
            check_overflow=False,
        )
        stream.synchronize()

        reps = 3
        times_ms: list[float] = []
        for _ in range(reps):
            start = cp.cuda.Event()
            end = cp.cuda.Event()
            start.record(stream)
            epq_contribs_many_count_allpairs_inplace_device(
                drt,
                drt_dev,
                state_dev,
                j_start=0,
                j_count=int(j_count),
                counts=counts,
                overflow=overflow,
                threads=int(threads),
                stream=int(stream.ptr),
                sync=False,
                check_overflow=False,
            )
            end.record(stream)
            end.synchronize()
            times_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
        return float(statistics.median(times_ms))

    t_by_threads: dict[int, float] = {}
    try:
        for t in candidates:
            if t <= 0:
                continue
            if t % 32 != 0:
                continue
            # Conservative guard: typical CUDA max threads per block.
            if t > 1024:
                continue
            t_by_threads[int(t)] = _time_threads(int(t))
    except Exception as e:
        if profile is not None:
            profile["autotune_error"] = f"bench_failed: {type(e).__name__}: {e}"
        return int(default)

    best = int(default)
    if t_by_threads:
        best = min(t_by_threads.items(), key=lambda kv: kv[1])[0]
        # Avoid flip-flopping due to timing noise: keep `default` unless clearly worse.
        if int(default) in t_by_threads:
            t_def = float(t_by_threads[int(default)])
            t_best = float(t_by_threads[int(best)])
            if t_best >= 0.97 * t_def:
                best = int(default)

    dev_tbl[problem_key] = {
        "best": int(best),
        "default": int(default),
        "candidates": [int(x) for x in candidates],
        "median_ms": {str(k): float(v) for k, v in sorted(t_by_threads.items(), key=lambda kv: kv[0])},
        "j_count": int(j_count),
        "n_pairs": int(n_pairs),
        "timestamp": float(time.time()),
    }
    _save_cache(cache)

    if profile is not None:
        profile["autotune_best"] = int(best)
        profile["autotune_median_ms"] = {int(k): float(v) for k, v in t_by_threads.items()}
    return int(best)
