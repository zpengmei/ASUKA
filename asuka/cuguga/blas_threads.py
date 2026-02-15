from __future__ import annotations

import contextlib
import ctypes
import os
import sys
from pathlib import Path
from typing import Iterator

try:  # optional dependency (installed with SciPy in most environments)
    from threadpoolctl import threadpool_limits as _threadpool_limits  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _threadpool_limits = None


_OPENBLAS = None
_OPENBLAS_PATH: str | None = None
_OPENBLAS_ALL: list[tuple[str, ctypes.CDLL]] | None = None


def _discover_openblas_paths() -> list[str]:
    paths: list[str] = []

    # Prefer already-loaded OpenBLAS shared libraries (common when PySCF bundles
    # its own OpenBLAS alongside NumPy/SciPy OpenBLAS).
    try:
        maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
        for line in maps.splitlines():
            parts = line.split()
            if not parts:
                continue
            path = parts[-1]
            if not path.startswith("/"):
                continue
            low = path.lower()
            if "openblas" not in low or ".so" not in low:
                continue
            paths.append(path)
    except Exception:
        pass

    # Also search sys.prefix/lib (useful for conda envs).
    try:
        libdir = Path(sys.prefix) / "lib"
        if libdir.is_dir():
            for p in libdir.glob("libopenblas*.so*"):
                if p.is_file() and p.name != "libopenblas.a":
                    paths.append(str(p))
            for p in libdir.glob("libopenblasp*.so*"):
                if p.is_file() and p.name != "libopenblasp.a":
                    paths.append(str(p))
    except Exception:
        pass

    # Stable unique, prefer earlier entries (maps-first).
    seen: set[str] = set()
    uniq: list[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _load_openblas_all() -> list[tuple[str, ctypes.CDLL]]:
    global _OPENBLAS_ALL
    if _OPENBLAS_ALL is None:
        _OPENBLAS_ALL = []

    # Libraries can be loaded after the first call (e.g. NumPy/SciPy importing
    # OpenBLAS after PySCF has already imported its bundled OpenBLAS). Keep a
    # stable list but discover and append new paths.
    seen = {p for p, _lib in _OPENBLAS_ALL}
    for path in _discover_openblas_paths():
        if path in seen:
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except Exception:
            continue
        if not hasattr(lib, "openblas_get_num_threads") or not hasattr(lib, "openblas_set_num_threads"):
            continue
        try:
            lib.openblas_get_num_threads.restype = ctypes.c_int
            lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
        except Exception:
            continue
        _OPENBLAS_ALL.append((str(path), lib))
        seen.add(str(path))

    return _OPENBLAS_ALL


def _load_openblas():
    global _OPENBLAS
    global _OPENBLAS_PATH
    if _OPENBLAS is not None:
        return _OPENBLAS

    # Pick the first usable OpenBLAS library as the "primary", but keep the full
    # list for "set all" operations.
    all_libs = _load_openblas_all()
    if all_libs:
        _OPENBLAS_PATH, _OPENBLAS = all_libs[0]
        return _OPENBLAS

    _OPENBLAS = False
    _OPENBLAS_PATH = None
    return _OPENBLAS


def openblas_get_num_threads() -> int | None:
    """Best-effort OpenBLAS thread count for this process.

    Notes
    -----
    Some environments can have *multiple* OpenBLAS shared libraries loaded
    concurrently (e.g. PySCF-bundled OpenBLAS plus NumPy/SciPy OpenBLAS). In
    that case, we return the maximum thread count across discovered libraries.
    """

    libs = _load_openblas_all()
    if not libs:
        return None
    best: int | None = None
    for _path, lib in libs:
        try:
            n = int(lib.openblas_get_num_threads())
        except Exception:
            continue
        if best is None or n > best:
            best = n
    return best


def openblas_set_num_threads(n: int) -> bool:
    n = int(n)
    if n < 1:
        raise ValueError("OpenBLAS thread limit must be >= 1")

    ok = False
    for _path, lib in _load_openblas_all():
        try:
            lib.openblas_set_num_threads(int(n))
            ok = True
        except Exception:
            pass
    return ok


def openblas_thread_info() -> dict[str, int]:
    """Best-effort OpenBLAS thread counts for all discovered OpenBLAS libs."""

    out: dict[str, int] = {}
    for path, lib in _load_openblas_all():
        try:
            out[str(path)] = int(lib.openblas_get_num_threads())
        except Exception:
            pass
    return out


def openblas_lib_path() -> str | None:
    """Best-effort path to a representative OpenBLAS shared library.

    When multiple OpenBLAS libraries are loaded, this returns the path for the
    one reporting the highest current thread count.
    """

    libs = _load_openblas_all()
    if not libs:
        return None
    best_path: str | None = None
    best_n: int | None = None
    for path, lib in libs:
        try:
            n = int(lib.openblas_get_num_threads())
        except Exception:
            continue
        if best_n is None or n > best_n:
            best_n = n
            best_path = str(path)
    return best_path


@contextlib.contextmanager
def openblas_thread_limit(n: int) -> Iterator[None]:
    """Temporarily set OpenBLAS threads for this process."""

    libs = _load_openblas_all()
    if not libs:
        yield
        return

    n = int(n)
    if n < 1:
        raise ValueError("OpenBLAS thread limit must be >= 1")

    prev = [(path, lib, int(lib.openblas_get_num_threads())) for path, lib in libs]
    if all(prev_n == n for _path, _lib, prev_n in prev):
        yield
        return

    for _path, lib, _prev_n in prev:
        try:
            lib.openblas_set_num_threads(int(n))
        except Exception:
            pass
    try:
        yield
    finally:
        for _path, lib, prev_n in prev:
            try:
                lib.openblas_set_num_threads(int(prev_n))
            except Exception:
                pass


@contextlib.contextmanager
def blas_thread_limit(n: int) -> Iterator[None]:
    """Temporarily limit BLAS threads for this process.

    Prefers `threadpoolctl` (covers MKL/OpenBLAS/BLIS) when available; otherwise
    falls back to OpenBLAS control.
    """

    n = int(n)
    if n < 1:
        raise ValueError("BLAS thread limit must be >= 1")

    if _threadpool_limits is not None:
        # Restrict only BLAS-style threadpools. This avoids accidentally
        # throttling OpenMP-parallel PySCF components such as AO2MO.
        with _threadpool_limits(limits=n, user_api="blas"):
            yield
        return

    with openblas_thread_limit(n):
        yield


@contextlib.contextmanager
def openmp_thread_limit(n: int) -> Iterator[None]:
    """Temporarily limit OpenMP threads for Cython OpenMP kernels.

    Notes
    -----
    This currently relies on :mod:`asuka._epq_cy` providing an OpenMP
    thread setter. If the extension is not built (or built without OpenMP),
    this context manager becomes a no-op.
    """

    n = int(n)
    if n < 1:
        raise ValueError("OpenMP thread limit must be >= 1")

    try:
        from asuka._epq_cy import (  # type: ignore[import-not-found]
            have_openmp as _have_openmp,
            openmp_max_threads as _openmp_max_threads,
            openmp_set_num_threads as _openmp_set_num_threads,
        )
    except Exception:
        yield
        return

    if not bool(_have_openmp()):
        yield
        return

    prev = int(_openmp_max_threads())
    if prev < 1:
        prev = 1
    if n == prev:
        yield
        return

    _openmp_set_num_threads(n)
    try:
        yield
    finally:
        _openmp_set_num_threads(prev)


@contextlib.contextmanager
def threadpool_thread_limit(n: int) -> Iterator[None]:
    """Temporarily limit BLAS/OpenMP threadpools for this process.

    This is broader than :func:`blas_thread_limit` and can be useful when
    diagnosing oversubscription, but it may reduce performance for OpenMP-heavy
    routines (e.g. AO2MO transforms).
    """

    n = int(n)
    if n < 1:
        raise ValueError("Threadpool limit must be >= 1")

    if _threadpool_limits is not None:
        with _threadpool_limits(limits=n):
            yield
        return

    with openblas_thread_limit(n):
        yield


@contextlib.contextmanager
def env_thread_limit(n: int, *, keys: tuple[str, ...] | None = None) -> Iterator[None]:
    """Temporarily set common thread-count environment variables.

    Notes
    -----
    This is best-effort. Some libraries only read these variables at import time,
    so prefer :func:`threadpool_thread_limit` when possible.
    """

    n = int(n)
    if n < 1:
        raise ValueError("env thread limit must be >= 1")

    if keys is None:
        keys = (
            "CUGUGA_NUM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )

    prev: dict[str, str | None] = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = str(n)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def asuka_thread_limit(
    n: int,
    *,
    scope: str = "blas",
    env: bool = False,
    openmp: bool = True,
) -> Iterator[None]:
    """Uniform thread limiter for ASUKA (Python threads + BLAS/OpenMP pools).

    This context manager is intended as the single place to constrain CPU
    oversubscription when ASUKA uses Python-level parallelism.

    Parameters
    ----------
    n:
        Target maximum thread count (>=1).
    scope:
        Which runtime thread pools to limit. ``\"blas\"`` limits only BLAS
        threadpools. ``\"threadpool\"`` limits both BLAS and OpenMP threadpools via
        :func:`threadpool_thread_limit` (useful for debugging oversubscription,
        but can reduce performance).
    env:
        Also set common thread-count environment variables (best-effort).
    openmp:
        Also limit OpenMP threads for ASUKA's Cython OpenMP kernels when the
        extension is available.
    """

    n = int(n)
    if n < 1:
        raise ValueError("asuka_thread_limit requires n >= 1")

    scope_opt = str(scope).strip().lower()
    if scope_opt not in ("blas", "threadpool"):
        raise ValueError("scope must be 'blas' or 'threadpool'")

    env_cm = env_thread_limit(n) if bool(env) else contextlib.nullcontext()

    pool_cm = blas_thread_limit(n) if scope_opt == "blas" else threadpool_thread_limit(n)
    openmp_cm = openmp_thread_limit(n) if bool(openmp) else contextlib.nullcontext()
    with env_cm, pool_cm, openmp_cm:
        yield
