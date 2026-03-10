from __future__ import annotations

import argparse
import csv
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterable

import numpy as np

from asuka.kernels import kernel_report


SCHEMA_VERSION = "2026-03-05"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKLOAD_NAMES = (
    "workflow_stilbene",
    "hf_df_jk_bq",
    "hf_df_jk_mnq",
    "hf_thc",
    "orbitals",
    "guga_sparse",
    "guga_linalg",
    "caspt2",
    "sci_cipsi_frontier_hash",
)
QUICK_WORKLOADS = (
    "hf_df_jk_bq",
    "hf_df_jk_mnq",
    "hf_thc",
    "orbitals",
    "guga_sparse",
    "guga_linalg",
    "caspt2",
)
FULL_WORKLOADS = ("workflow_stilbene",) + QUICK_WORKLOADS
WORKLOAD_BENCHMARKS: dict[str, tuple[str, ...]] = {
    "workflow_stilbene": ("workflow_stilbene",),
    "hf_df_jk_bq": ("hf_df_jk_bq_cuda_ext", "hf_df_jk_bq_cupy"),
    "hf_df_jk_mnq": ("hf_df_jk_mnq_cuda_ext", "hf_df_jk_mnq_cupy"),
    "hf_thc": ("hf_thc_rowwise_dot", "hf_thc_scale_rows", "hf_thc_hadamard_inplace"),
    "orbitals": ("orbitals_eval_value", "orbitals_eval_value_grad"),
    "guga_sparse": ("guga_ell_spmv", "guga_ell_spmm"),
    "guga_linalg": ("guga_linalg_gemm",),
    "caspt2": ("caspt2_apply_h0diag_sr", "caspt2_mltmv", "caspt2_mltr1", "caspt2_mltdxp"),
    "sci_cipsi_frontier_hash": (
        "sci_cipsi_frontier_hash_synth_pt2",
        "sci_cipsi_frontier_hash_synth_select",
        "sci_cipsi_frontier_hash_naqs_N2_select",
        "sci_cipsi_frontier_hash_naqs_Li2O_select",
    ),
}
BENCHMARK_NAMES = tuple(name for names in WORKLOAD_BENCHMARKS.values() for name in names)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _payload_summary_block(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {"results": results, "summary": summarize_results(results)}


def _safe_float(x: Any) -> float | None:
    try:
        out = float(x)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def _sample_stats(samples: list[float]) -> dict[str, Any]:
    if not samples:
        return {"count": 0}
    med = float(statistics.median(samples))
    return {
        "count": int(len(samples)),
        "median": med,
        "mean": float(statistics.mean(samples)),
        "min": float(min(samples)),
        "max": float(max(samples)),
        "stdev": float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
        "jitter_pct": float(0.0 if med == 0.0 else (max(samples) - min(samples)) / med * 100.0),
        "samples": [float(x) for x in samples],
    }


def _csv_rows_from_text(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    header_idx = None
    for i, line in enumerate(lines):
        if "," not in line:
            continue
        if line.startswith(("Generating ", "Processing ", "NOTICE:", "Input file", "RC=")):
            continue
        header_idx = i
        break
    if header_idx is None:
        return []
    reader = csv.DictReader(lines[header_idx:])
    return [dict(row) for row in reader if row]


def _int_from_row(row: dict[str, str], key: str) -> int | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).replace(",", "")))
    except Exception:
        return None


def _float_from_row(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        return None
    return _safe_float(str(value).replace(",", ""))


def _top_rows(rows: list[dict[str, str]], *, sort_key: str, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: -float(_float_from_row(row, sort_key) or 0.0))
    out: list[dict[str, Any]] = []
    for row in ranked[: max(0, int(limit))]:
        cooked: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                continue
            ival = _int_from_row(row, key)
            if ival is not None and any(token in key for token in ("Time", "Calls", "Instances", "Duration", "Reg/Trd", "Grd", "Blk", "Ctx", "Strm")):
                cooked[key] = ival
                continue
            fval = _float_from_row(row, key)
            if fval is not None and any(token in key for token in ("Time", "Avg", "Med", "Min", "Max", "StdDev", "Throughput", "Bytes", "MB")):
                cooked[key] = fval
            else:
                cooked[key] = value
        out.append(cooked)
    return out


def _summarize_nsys_kernel_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    total_time_ns = 0
    total_instances = 0
    for row in rows:
        total_time_ns += int(_float_from_row(row, "Total Time (ns)") or 0.0)
        total_instances += int(_float_from_row(row, "Instances") or 0.0)
    return {
        "kernel_count": int(len(rows)),
        "kernel_launches": int(total_instances),
        "total_kernel_time_ns": int(total_time_ns),
        "top_kernels": _top_rows(rows, sort_key="Total Time (ns)", limit=10),
    }


def _summarize_nsys_trace_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    kernel_rows = [row for row in rows if "[CUDA memcpy" not in str(row.get("Name", ""))]
    memcpy_rows = [row for row in rows if "[CUDA memcpy" in str(row.get("Name", ""))]
    streams = sorted(
        {
            int(val)
            for row in kernel_rows
            for val in [row.get("Strm")]
            if val is not None and str(val).strip() != ""
        }
    )
    max_regs = max((_int_from_row(row, "Reg/Trd") or 0) for row in kernel_rows) if kernel_rows else 0
    return {
        "kernel_launches": int(len(kernel_rows)),
        "memcpy_events": int(len(memcpy_rows)),
        "streams": streams,
        "max_registers_per_thread": int(max_regs),
        "top_trace_rows": _top_rows(rows, sort_key="Duration (ns)", limit=10),
    }


def _summarize_nsys_api_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    total_ns = sum(int(_float_from_row(row, "Total Time (ns)") or 0.0) for row in rows)
    return {
        "api_call_count": int(sum(int(_float_from_row(row, "Num Calls") or 0.0) for row in rows)),
        "total_api_time_ns": int(total_ns),
        "top_api_calls": _top_rows(rows, sort_key="Total Time (ns)", limit=10),
    }


def _result_stub(name: str, family: str, *, status: str = "ok") -> dict[str, Any]:
    return {
        "name": str(name),
        "family": str(family),
        "status": str(status),
        "created_at_utc": _utc_now(),
        "timing": {},
        "memory": {},
        "details": {},
    }


def _exception_result(name: str, family: str, exc: BaseException) -> dict[str, Any]:
    out = _result_stub(name, family, status="error")
    out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _skipped_result(name: str, family: str, reason: str) -> dict[str, Any]:
    out = _result_stub(name, family, status="skipped")
    out["reason"] = str(reason)
    return out


def _cupy() -> Any:
    import cupy as cp  # type: ignore

    return cp


def _cuda_init(cp: Any, *, retries: int = 5, delay_s: float = 0.2) -> int:
    _ = cp.cuda.runtime.runtimeGetVersion()
    last_err: BaseException | None = None
    for _ in range(max(1, int(retries))):
        try:
            ndev = int(cp.cuda.runtime.getDeviceCount())
            _ = float(cp.arange(8, dtype=cp.float32).sum())
            cp.cuda.runtime.deviceSynchronize()
            return ndev
        except Exception as exc:  # pragma: no cover
            last_err = exc
            time.sleep(float(delay_s))
    if last_err is not None:
        raise last_err
    return int(cp.cuda.runtime.getDeviceCount())


def _mem_snapshot(cp: Any) -> dict[str, int]:
    cp.cuda.runtime.deviceSynchronize()
    pool = cp.get_default_memory_pool()
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    return {
        "driver_used_bytes": int(total_b - free_b),
        "pool_used_bytes": int(pool.used_bytes()),
        "pool_total_bytes": int(pool.total_bytes()),
    }


def _mem_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in ("driver_used_bytes", "pool_used_bytes", "pool_total_bytes"):
        out[f"{key}_delta"] = int(after.get(key, 0) - before.get(key, 0))
    return out


def _flush_pools(cp: Any) -> None:
    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    try:
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _event_bench(cp: Any, fn: Callable[[], Any], *, warmup: int = 3, iters: int = 10) -> tuple[dict[str, Any], Any]:
    last_out: Any = None
    for _ in range(max(0, int(warmup))):
        last_out = fn()
    cp.cuda.runtime.deviceSynchronize()

    event_ms: list[float] = []
    wall_ms: list[float] = []
    for _ in range(max(1, int(iters))):
        ev0 = cp.cuda.Event()
        ev1 = cp.cuda.Event()
        t0 = time.perf_counter()
        ev0.record()
        last_out = fn()
        ev1.record()
        ev1.synchronize()
        wall_ms.append(float((time.perf_counter() - t0) * 1000.0))
        event_ms.append(float(cp.cuda.get_elapsed_time(ev0, ev1)))
    return {
        "warmup": int(warmup),
        "iters": int(iters),
        "event_ms": _sample_stats(event_ms),
        "wall_ms": _sample_stats(wall_ms),
    }, last_out


@contextmanager
def _temp_env(**updates: str | None):
    old: dict[str, str | None] = {}
    for key, value in updates.items():
        old[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _benchmark_hf_df_jk_variant(
    *,
    layout: str,
    impl: str,
    seed: int = 0,
    nao: int = 150,
    naux: int = 1340,
    nocc: int = 75,
    q_block: int = 128,
    warmup: int = 3,
    iters: int = 10,
) -> dict[str, Any]:
    name = f"hf_df_jk_{layout}_{impl}"
    try:
        cp = _cupy()
        _cuda_init(cp)
        from asuka.hf import df_jk

        rng = np.random.default_rng(int(seed))
        bq_host = rng.normal(size=(int(naux), int(nao), int(nao))).astype(np.float64, copy=False)
        bq_host = 0.5 * (bq_host + bq_host.transpose(0, 2, 1))
        BQ = cp.asarray(bq_host, dtype=cp.float64)
        B_mnQ = cp.ascontiguousarray(BQ.transpose(1, 2, 0))
        C_occ = cp.asarray(rng.normal(size=(int(nao), int(nocc))).astype(np.float64, copy=False), dtype=cp.float64)
        occ_vals = cp.asarray((rng.random(size=(int(nocc),)) + 0.1).astype(np.float64, copy=False), dtype=cp.float64)

        if layout == "bq":
            fn = lambda prof=None: df_jk.df_K_from_BQ_Cocc(BQ, C_occ, occ_vals, q_block=int(q_block), profile=prof)
            in_shape = list(map(int, BQ.shape))
        else:
            fn = lambda prof=None: df_jk.df_K_from_BmnQ_Cocc(B_mnQ, C_occ, occ_vals, q_block=int(q_block), profile=prof)
            in_shape = list(map(int, B_mnQ.shape))

        _flush_pools(cp)
        before = _mem_snapshot(cp)
        prof: dict[str, Any] = {}
        with _temp_env(ASUKA_HF_K_IMPL=str(impl)):
            timing, out = _event_bench(cp, lambda: fn(prof=prof), warmup=int(warmup), iters=int(iters))
        after = _mem_snapshot(cp)
        result = _result_stub(name, "hf_df_jk")
        result["timing"] = timing
        result["memory"] = {
            "before": before,
            "after": after,
            "delta": _mem_delta(before, after),
        }
        result["details"] = {
            "layout": str(layout),
            "impl_requested": str(impl),
            "nao": int(nao),
            "naux": int(naux),
            "nocc": int(nocc),
            "q_block": int(q_block),
            "input_shape": in_shape,
            "output_shape": list(map(int, out.shape)),
            "profile": prof,
        }
        return result
    except Exception as exc:
        return _exception_result(name, "hf_df_jk", exc)


def _benchmark_hf_thc(*, seed: int = 0, warmup: int = 5, iters: int = 20) -> list[dict[str, Any]]:
    try:
        cp = _cupy()
        _cuda_init(cp)
        from asuka import _hf_thc_cuda_ext as ext  # type: ignore[import-not-found]
    except Exception as exc:
        return [_exception_result("hf_thc", "hf_thc", exc)]

    rng = cp.random.default_rng(int(seed))
    stream_ptr = int(cp.cuda.get_current_stream().ptr)
    results: list[dict[str, Any]] = []

    benches: list[tuple[str, Callable[[], Any], dict[str, Any]]] = []

    A = rng.standard_normal((257, 193), dtype=cp.float64)
    X = rng.standard_normal((257, 193), dtype=cp.float64)
    out_dot = cp.empty((257,), dtype=cp.float64)
    benches.append(
        (
            "hf_thc_rowwise_dot",
            lambda: ext.rowwise_dot_f64(A, X, out_dot, threads=256, stream_ptr=stream_ptr, sync=False),
            {"shape_A": [257, 193], "shape_X": [257, 193], "shape_out": [257]},
        )
    )

    X_rows = rng.standard_normal((123, 77), dtype=cp.float64)
    n = rng.standard_normal((123,), dtype=cp.float64)
    out_rows = cp.empty_like(X_rows)
    benches.append(
        (
            "hf_thc_scale_rows",
            lambda: ext.scale_rows_f64(X_rows, n, out_rows, threads=256, stream_ptr=stream_ptr, sync=False),
            {"shape_X": [123, 77], "shape_n": [123], "shape_out": [123, 77]},
        )
    )

    M = rng.standard_normal((211, 33), dtype=cp.float64)
    Zbig = rng.standard_normal((211, 40), dtype=cp.float64)
    Zblk = Zbig[:, 3 : 3 + 33]
    benches.append(
        (
            "hf_thc_hadamard_inplace",
            lambda: ext.hadamard_inplace_f64(M, Zblk, threads=256, stream_ptr=stream_ptr, sync=False),
            {"shape_M": [211, 33], "shape_Z": [211, 33]},
        )
    )

    for name, fn, details in benches:
        try:
            _flush_pools(cp)
            before = _mem_snapshot(cp)
            timing, _out = _event_bench(cp, fn, warmup=int(warmup), iters=int(iters))
            after = _mem_snapshot(cp)
            result = _result_stub(name, "hf_thc")
            result["timing"] = timing
            result["memory"] = {"before": before, "after": after, "delta": _mem_delta(before, after)}
            result["details"] = details
            results.append(result)
        except Exception as exc:
            results.append(_exception_result(name, "hf_thc", exc))
    return results


def _benchmark_orbitals(*, seed: int = 0, warmup: int = 3, iters: int = 10) -> list[dict[str, Any]]:
    try:
        cp = _cupy()
        _cuda_init(cp)
        from asuka.frontend.molecule import Molecule
        from asuka.frontend.one_electron import build_ao_basis_cart
        from asuka.orbitals.eval_basis_device import (
            eval_aos_cart_value_grad_on_points_device,
            eval_aos_cart_value_on_points_device,
        )
    except Exception as exc:
        return [_exception_result("orbitals", "orbitals", exc)]

    atoms = [
        ("O", (0.000000, 0.000000, 0.117300)),
        ("H", (0.000000, 0.757200, -0.469200)),
        ("H", (0.000000, -0.757200, -0.469200)),
    ]
    mol = Molecule.from_atoms(atoms, unit="angstrom", basis="cc-pvdz", cart=True)
    ao_basis, basis_name = build_ao_basis_cart(mol)
    rng = np.random.default_rng(int(seed))
    points = cp.asarray(rng.normal(loc=0.0, scale=1.0, size=(2048, 3)).astype(np.float64, copy=False), dtype=cp.float64)

    results: list[dict[str, Any]] = []
    benches = [
        (
            "orbitals_eval_value",
            lambda: eval_aos_cart_value_on_points_device(ao_basis, points, sync=False),
            {"basis": str(basis_name), "npt": 2048},
        ),
        (
            "orbitals_eval_value_grad",
            lambda: eval_aos_cart_value_grad_on_points_device(ao_basis, points, sync=False),
            {"basis": str(basis_name), "npt": 2048},
        ),
    ]

    for name, fn, details in benches:
        try:
            _flush_pools(cp)
            before = _mem_snapshot(cp)
            timing, out = _event_bench(cp, fn, warmup=int(warmup), iters=int(iters))
            after = _mem_snapshot(cp)
            if isinstance(out, tuple):
                out_shapes = [list(map(int, arr.shape)) for arr in out]
            else:
                out_shapes = [list(map(int, out.shape))]
            result = _result_stub(name, "orbitals")
            result["timing"] = timing
            result["memory"] = {"before": before, "after": after, "delta": _mem_delta(before, after)}
            result["details"] = {**details, "output_shapes": out_shapes}
            results.append(result)
        except Exception as exc:
            results.append(_exception_result(name, "orbitals", exc))
    return results


def _benchmark_guga_sparse(*, seed: int = 0, warmup: int = 3, iters: int = 10) -> list[dict[str, Any]]:
    try:
        cp = _cupy()
        _cuda_init(cp)
        import asuka.cuda.cuda_backend as guga_backend
        from asuka import _guga_cuda_ext as guga_ext  # type: ignore[import-not-found]
    except Exception as exc:
        return [_exception_result("guga_sparse", "guga", exc)]

    rng = np.random.default_rng(int(seed))
    nrows = 4096
    width = 16
    col_idx_np = np.tile(np.arange(width, dtype=np.int32), (nrows, 1))
    col_idx_np = (col_idx_np + np.arange(nrows, dtype=np.int32)[:, None]) % nrows
    val_np = rng.normal(size=(nrows, width)).astype(np.float64, copy=False)
    x_np = rng.normal(size=(nrows,)).astype(np.float64, copy=False)
    x2_np = rng.normal(size=(nrows, 8)).astype(np.float64, copy=False)

    col_idx = cp.asarray(col_idx_np, dtype=cp.int32)
    val = cp.asarray(val_np, dtype=cp.float64)
    x = cp.asarray(x_np, dtype=cp.float64)
    x2 = cp.asarray(x2_np, dtype=cp.float64)

    benches = [
        (
            "guga_ell_spmv",
            lambda: guga_backend.ell_spmv_f64_inplace_device(col_idx, val, x, threads=128, sync=False),
            {"nrows": int(nrows), "width": int(width)},
        ),
        (
            "guga_ell_spmm",
            lambda: guga_backend.ell_spmm_f64_inplace_device(col_idx, val, x2, threads=128, sync=False),
            {"nrows": int(nrows), "width": int(width), "nvec": 8},
        ),
    ]
    results: list[dict[str, Any]] = []
    for name, fn, details in benches:
        try:
            _flush_pools(cp)
            before = _mem_snapshot(cp)
            ext_before = dict(guga_ext.mem_info())
            timing, out = _event_bench(cp, fn, warmup=int(warmup), iters=int(iters))
            after = _mem_snapshot(cp)
            ext_after = dict(guga_ext.mem_info())
            result = _result_stub(name, "guga")
            result["timing"] = timing
            result["memory"] = {
                "before": before,
                "after": after,
                "delta": _mem_delta(before, after),
                "ext_before": ext_before,
                "ext_after": ext_after,
            }
            result["details"] = {
                **details,
                "output_shape": list(map(int, out.shape)),
            }
            results.append(result)
        except Exception as exc:
            results.append(_exception_result(name, "guga", exc))
    return results


def _benchmark_guga_linalg(*, seed: int = 0, warmup: int = 1, iters: int = 3) -> dict[str, Any]:
    name = "guga_linalg_gemm"
    try:
        import asuka.cuda.cuda_linalg_backend as linalg_backend

        rng = np.random.default_rng(int(seed))
        a = rng.normal(size=(512, 256)).astype(np.float64, copy=False)
        b = rng.normal(size=(256, 256)).astype(np.float64, copy=False)
        for _ in range(max(0, int(warmup))):
            _ = linalg_backend.gemm(a, b)
        wall_ms: list[float] = []
        mem_before = dict(linalg_backend.mem_info())
        out = None
        for _ in range(max(1, int(iters))):
            t0 = time.perf_counter()
            out = linalg_backend.gemm(a, b)
            wall_ms.append(float((time.perf_counter() - t0) * 1000.0))
        mem_after = dict(linalg_backend.mem_info())
        result = _result_stub(name, "guga")
        result["timing"] = {
            "warmup": int(warmup),
            "iters": int(iters),
            "wall_ms": _sample_stats(wall_ms),
        }
        result["memory"] = {"ext_before": mem_before, "ext_after": mem_after}
        result["details"] = {
            "shape_a": [512, 256],
            "shape_b": [256, 256],
            "output_shape": list(map(int, out.shape)) if out is not None else [512, 256],
        }
        return result
    except Exception as exc:
        return _exception_result(name, "guga", exc)


def _random_list_soa(rng: np.random.Generator, *, n: int, l1: int, l2: int, l3: int) -> np.ndarray:
    data = np.zeros((4, int(n)), dtype=np.int32)
    data[0, :] = rng.integers(0, max(1, int(l1)), size=int(n), endpoint=False, dtype=np.int32)
    data[1, :] = rng.integers(0, max(1, int(l2)), size=int(n), endpoint=False, dtype=np.int32)
    data[2, :] = rng.integers(0, max(1, int(l3)), size=int(n), endpoint=False, dtype=np.int32)
    data[3, :] = rng.integers(0, 2, size=int(n), endpoint=False, dtype=np.int32)
    return data


def _benchmark_caspt2(*, seed: int = 0, warmup: int = 3, iters: int = 10) -> list[dict[str, Any]]:
    try:
        cp = _cupy()
        _cuda_init(cp)
        from asuka.caspt2.cuda import kernels
    except Exception as exc:
        return [_exception_result("caspt2", "caspt2", exc)]

    rng = np.random.default_rng(int(seed))
    cp_rng = cp.random.default_rng(int(seed))
    results: list[dict[str, Any]] = []

    nin = 256
    nis = 64
    x = cp_rng.standard_normal((nin, nis), dtype=cp.float64)
    y = cp.empty_like(x)
    bd = cp.asarray(np.abs(rng.normal(size=(nin,))).astype(np.float64, copy=False) + 1.0, dtype=cp.float64)
    idv = cp.asarray(np.abs(rng.normal(size=(nis,))).astype(np.float64, copy=False) + 1.0, dtype=cp.float64)

    l1 = cp.asarray(_random_list_soa(rng, n=1024, l1=64, l2=48, l3=32), dtype=cp.int32)
    l2 = cp.asarray(_random_list_soa(rng, n=1024, l1=72, l2=48, l3=32), dtype=cp.int32)
    x_mv = cp_rng.standard_normal((64, 128), dtype=cp.float64)
    f_mv = cp_rng.standard_normal((48, 96), dtype=cp.float64)
    y_mv = cp_rng.standard_normal((32, 128, 96), dtype=cp.float64)
    x_r1 = cp_rng.standard_normal((64, 96, 48), dtype=cp.float64)
    f_r1 = cp_rng.standard_normal((48, 96), dtype=cp.float64)
    y_r1 = cp_rng.standard_normal((32, 48), dtype=cp.float64)

    benches = [
        (
            "caspt2_apply_h0diag_sr",
            lambda: kernels.apply_h0diag_sr(y=y, x=x, bd=bd, id=idv, real_shift=0.2, imag_shift=0.0),
            {"shape_x": [nin, nis], "shape_bd": [nin], "shape_id": [nis]},
        ),
        (
            "caspt2_mltmv",
            lambda: kernels.mltmv(1, l1, x_mv, f_mv, y_mv, val1=(1.0, 1.0)),
            {"shape_x": [64, 128], "shape_f": [48, 96], "shape_y": [32, 128, 96], "list_n": 1024},
        ),
        (
            "caspt2_mltr1",
            lambda: kernels.mltr1(1, l1, x_r1, f_r1, y_r1, val1=(1.0, -1.0)),
            {"shape_x": [64, 96, 48], "shape_f": [48, 96], "shape_y": [32, 48], "list_n": 1024},
        ),
        (
            "caspt2_mltdxp",
            lambda: kernels.mltdxp(
                1,
                l1,
                l2,
                cp_rng.standard_normal((64, 72, 48), dtype=cp.float64),
                cp_rng.standard_normal((48, 48), dtype=cp.float64),
                cp_rng.standard_normal((32, 32, 48), dtype=cp.float64),
                val1=(1.0, -1.0),
                val2=(0.5, 1.5),
            ),
            {"shape_x": [64, 72, 48], "shape_f": [48, 48], "shape_y": [32, 32, 48], "list_n1": 1024, "list_n2": 1024},
        ),
    ]

    for name, fn, details in benches:
        try:
            _flush_pools(cp)
            before = _mem_snapshot(cp)
            timing, _out = _event_bench(cp, fn, warmup=int(warmup), iters=int(iters))
            after = _mem_snapshot(cp)
            result = _result_stub(name, "caspt2")
            result["timing"] = timing
            result["memory"] = {"before": before, "after": after, "delta": _mem_delta(before, after)}
            result["details"] = details
            results.append(result)
        except Exception as exc:
            results.append(_exception_result(name, "caspt2", exc))
    return results


def _benchmark_workflow_stilbene(
    *,
    output_dir: Path,
    basis: str = "cc-pvdz",
    repeats: int = 1,
    profile_hotspots: bool = False,
) -> dict[str, Any]:
    name = "workflow_stilbene"
    script = REPO_ROOT / "profile_vram_631g.py"
    if not script.exists():
        return _skipped_result(name, "workflow", f"missing script: {script}")

    raw_json = output_dir / (
        "workflow_stilbene_diagnostic_raw.json" if bool(profile_hotspots) else "workflow_stilbene_timed_raw.json"
    )
    cmd = [
        sys.executable,
        str(script),
        "--basis",
        str(basis),
        "--repeats",
        str(int(repeats)),
        "--out-json",
        str(raw_json),
    ]
    if bool(profile_hotspots):
        cmd.extend(["--profile-hotspots", "on"])

    env = os.environ.copy()
    workflow_env = {
        # Time-oriented audit should measure the production defaults, not the
        # benchmark script's debug/tight-VRAM fallback path triggered by
        # ASUKA_VRAM_DEBUG=1.
        "ASUKA_VRAM_DEBUG": "0",
    }
    env.update(workflow_env)

    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
    except Exception as exc:
        return _exception_result(name, "workflow", exc)
    if proc.returncode != 0:
        out = _result_stub(name, "workflow", status="error")
        out["error"] = f"profile_vram_631g.py exited with code {proc.returncode}"
        out["details"] = {"stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "cmd": cmd}
        return out
    try:
        payload = json.loads(raw_json.read_text(encoding="utf-8"))
    except Exception as exc:
        out = _exception_result(name, "workflow", exc)
        out["details"] = {"stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:], "cmd": cmd}
        return out

    run0 = payload.get("runs", [{}])[0] if payload.get("runs") else {}
    result = _result_stub(name, "workflow")
    result["timing"] = {
        "wall_s": {
            "t_scf": _safe_float(run0.get("t_scf")),
            "t_casscf": _safe_float(run0.get("t_casscf")),
            "t_grad": _safe_float(run0.get("t_grad")),
            "t_total": _safe_float(run0.get("t_total")),
        }
    }
    result["memory"] = {
        "peak_driver_used_bytes": int(float(run0.get("peak_nvidia_gb", 0.0)) * 1e9),
        "peak_pool_used_bytes": int(float(run0.get("peak_pool_active_gb", 0.0)) * 1e9),
        "peak_pool_total_bytes": int(float(run0.get("peak_pool_total_gb", 0.0)) * 1e9),
    }
    result["details"] = {
        "basis": str(basis),
        "repeats": int(repeats),
        "profile_hotspots": bool(profile_hotspots),
        "env_overrides": workflow_env,
        "raw_artifact": str(raw_json),
        "profiler_keys": sorted(run0.get("profiler", {}).keys()) if isinstance(run0.get("profiler"), dict) else [],
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    return result


def _load_naqs_hdf5_molecule(name: str, *, root: Path | None = None) -> dict[str, Any]:
    """Load dense MO integrals + basic metadata from the NAQS paper HDF5 artifacts."""

    root_eff = Path(
        os.environ.get("NAQS_MOLECULE_ROOT", "/home/zpengmei/Projects/naqs-for-quantum-chemistry/molecules")
        if root is None
        else root
    )
    mol = str(name)
    h5_path = root_eff / mol / f"{mol}.hdf5"
    if not h5_path.exists():
        raise FileNotFoundError(f"NAQS HDF5 not found: {h5_path}")

    try:
        import h5py  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for NAQS HDF5 benchmarks") from exc

    with h5py.File(str(h5_path), "r") as f:
        norb = int(f["n_orbitals"][()])
        nelec = int(f["n_electrons"][()])
        multiplicity = int(f["multiplicity"][()])
        ecore = float(f["nuclear_repulsion"][()])
        h1e = np.asarray(f["one_body_integrals"][()], dtype=np.float64, order="C")
        eri4 = np.asarray(f["two_body_integrals"][()], dtype=np.float64, order="C")
        hf_energy = float(f.get("hf_energy", np.asarray(0.0))[()])
        fci_energy = float(f.get("fci_energy", np.asarray(0.0))[()])
        basis = str(np.asarray(f.get("basis", np.asarray(b"unknown"))).item())

    return {
        "name": mol,
        "path": str(h5_path),
        "basis": basis,
        "norb": norb,
        "nelec": nelec,
        "multiplicity": multiplicity,
        "ecore": ecore,
        "hf_energy": hf_energy,
        "fci_energy": fci_energy,
        "h1e": h1e,
        "eri4": eri4,
    }


def _benchmark_sci_cipsi_frontier_hash(name: str, *, diagnostic: bool) -> dict[str, Any]:
    """Benchmark the current scalable CIPSI selection/PT2 path.

    The public CIPSI driver no longer uses the legacy dense CUDA frontier-hash
    workspace. These benchmark names are preserved, but they now measure the
    sparse row-oracle selector that backs ``selection_mode='frontier_hash'``.
    """

    try:
        from asuka.cuguga.drt import build_drt
        from asuka.sci.frontier_hash import SparseFrontierSelector
        from asuka.sci.selected_ci import DiagonalGuessLookup
    except Exception as exc:
        return _exception_result(str(name), "sci", exc)

    bench = str(name)
    mode = None
    mol = None
    if bench.endswith("_pt2"):
        mode = "pt2"
    elif bench.endswith("_select"):
        mode = "select"
    else:
        return _skipped_result(bench, "sci", "unknown frontier-hash benchmark variant")

    if "naqs_N2" in bench:
        mol = "N2"
    elif "naqs_Li2O" in bench:
        mol = "Li2O"
    elif "synth" in bench:
        mol = None
    else:
        mol = None

    rng = np.random.default_rng(11)
    nroots = 1
    denom_floor = 1e-12
    tile = 1024
    rs_block = 128
    max_add = 0 if mode == "pt2" else 1024

    if mol is None:
        # Synthetic large-CAS-ish shape: match Li2O (norb=15, nelec=14, singlet) for ncsf ~ 9.2M.
        norb = 15
        nelec = 14
        twos = 0
        drt = build_drt(norb=int(norb), nelec=int(nelec), twos_target=int(twos))

        h1e = rng.normal(size=(norb, norb))
        h1e = 0.5 * (h1e + h1e.T)
        eri = rng.normal(size=(norb, norb, norb, norb))
        eri = 0.25 * (
            eri
            + eri.transpose(1, 0, 2, 3)
            + eri.transpose(0, 1, 3, 2)
            + eri.transpose(2, 3, 0, 1)
        )
        h1e = np.asarray(h1e, dtype=np.float64, order="C")
        eri = np.asarray(eri, dtype=np.float64, order="C")
        e_var = np.asarray([-1.0], dtype=np.float64)
        details: dict[str, Any] = {"case": "synth", "norb": int(norb), "nelec": int(nelec), "twos": int(twos)}
    else:
        data = _load_naqs_hdf5_molecule(mol)
        norb = int(data["norb"])
        nelec = int(data["nelec"])
        twos = int(data["multiplicity"]) - 1
        drt = build_drt(norb=int(norb), nelec=int(nelec), twos_target=int(twos))
        h1e = np.asarray(data["h1e"], dtype=np.float64, order="C")
        eri = np.asarray(data["eri4"], dtype=np.float64, order="C")
        e_var = np.asarray([float(data.get("hf_energy", -1.0))], dtype=np.float64)
        details = {
            "case": f"naqs:{mol}",
            "naqs_path": str(data.get("path")),
            "basis": str(data.get("basis", "unknown")),
            "norb": int(norb),
            "nelec": int(nelec),
            "twos": int(twos),
            "ecore": float(data.get("ecore", 0.0)),
            "hf_energy": float(data.get("hf_energy", 0.0)),
            "fci_energy": float(data.get("fci_energy", 0.0)),
        }

    # Selection size: keep tile groups limited for stable profiling.
    ncsf = int(drt.ncsf)
    nsel = int(min(2048, max(64, ncsf // 1000)))
    sel_pool = int(min(ncsf, tile * 8))
    sel_idx = rng.choice(sel_pool, size=nsel, replace=False).astype(np.int64, copy=False)
    sel_idx.sort()
    c_sel = rng.normal(size=(nsel, nroots)).astype(np.float64, copy=False)
    c_sel /= max(1e-12, float(np.linalg.norm(c_sel)))

    hdiag_lookup = DiagonalGuessLookup(drt, h1e, eri, hdiag=None)
    selector = SparseFrontierSelector(
        drt,
        h1e,
        eri,
        hdiag_lookup=hdiag_lookup,
        denom_floor=float(denom_floor),
        max_out=200_000,
        screening=None,
        state_cache=None,
        select_screen_contrib=0.0,
    )
    selector.reset_selected_mask(sel_idx)

    profiled = None
    if bool(diagnostic):
        t0 = time.perf_counter()
        _, _, st = selector.build_and_score(sel_idx=sel_idx, c_sel=c_sel, e_var=e_var, max_add=int(max_add), profile=True)
        elapsed_ms = float((time.perf_counter() - t0) * 1000.0)
        profiled = {
            "driver": "sparse_row_oracle",
            "elapsed_ms": elapsed_ms,
            "hash_cap": int(st.hash_cap),
            "nnz_out": int(st.nnz_out),
            "overflow_retries": int(st.overflow_retries),
            "timings_ms": dict(st.timings_ms),
            "memory": dict(st.memory),
        }

    def fn():
        _, out_pt2, _st = selector.build_and_score(sel_idx=sel_idx, c_sel=c_sel, e_var=e_var, max_add=int(max_add), profile=False)
        return out_pt2

    try:
        out = None
        wall_ms: list[float] = []
        for _ in range(1):
            out = fn()
        for _ in range(2):
            t0 = time.perf_counter()
            out = fn()
            wall_ms.append(float((time.perf_counter() - t0) * 1000.0))
        timing = {
            "warmup": 1,
            "iters": 2,
            "wall_ms": _sample_stats(wall_ms),
        }

        result = _result_stub(bench, "sci")
        result["timing"] = timing
        result["memory"] = {
            "driver": "sparse_row_oracle",
        }
        result["details"] = {
            **details,
            "driver": "sparse_row_oracle",
            "ncsf": int(ncsf),
            "nsel": int(nsel),
            "sel_pool": int(sel_pool),
            "max_add": int(max_add),
            "tile_requested": int(tile),
            "rs_block_requested": int(rs_block),
            "output_shape": list(map(int, out.shape)) if hasattr(out, "shape") else [int(nroots)],
            "profiled": profiled,
        }
        return result
    except Exception as exc:
        return _exception_result(bench, "sci", exc)


def _run_benchmark(name: str, *, output_dir: Path, diagnostic: bool = False) -> dict[str, Any]:
    if name == "workflow_stilbene":
        return _benchmark_workflow_stilbene(output_dir=output_dir, profile_hotspots=bool(diagnostic))
    if name == "hf_df_jk_bq_cuda_ext":
        return _benchmark_hf_df_jk_variant(layout="bq", impl="cuda_ext")
    if name == "hf_df_jk_bq_cupy":
        return _benchmark_hf_df_jk_variant(layout="bq", impl="cupy")
    if name == "hf_df_jk_mnq_cuda_ext":
        return _benchmark_hf_df_jk_variant(layout="mnq", impl="cuda_ext")
    if name == "hf_df_jk_mnq_cupy":
        return _benchmark_hf_df_jk_variant(layout="mnq", impl="cupy")
    if name in {"hf_thc_rowwise_dot", "hf_thc_scale_rows", "hf_thc_hadamard_inplace"}:
        for item in _benchmark_hf_thc():
            if item.get("name") == name:
                return item
        return _skipped_result(name, "hf_thc", "benchmark result not produced")
    if name in {"orbitals_eval_value", "orbitals_eval_value_grad"}:
        for item in _benchmark_orbitals():
            if item.get("name") == name:
                return item
        return _skipped_result(name, "orbitals", "benchmark result not produced")
    if name in {"guga_ell_spmv", "guga_ell_spmm"}:
        for item in _benchmark_guga_sparse():
            if item.get("name") == name:
                return item
        return _skipped_result(name, "guga", "benchmark result not produced")
    if name == "guga_linalg_gemm":
        return _benchmark_guga_linalg()
    if name in {"caspt2_apply_h0diag_sr", "caspt2_mltmv", "caspt2_mltr1", "caspt2_mltdxp"}:
        for item in _benchmark_caspt2():
            if item.get("name") == name:
                return item
        return _skipped_result(name, "caspt2", "benchmark result not produced")
    if name in {
        "sci_cipsi_frontier_hash_synth_pt2",
        "sci_cipsi_frontier_hash_synth_select",
        "sci_cipsi_frontier_hash_naqs_N2_select",
        "sci_cipsi_frontier_hash_naqs_Li2O_select",
    }:
        return _benchmark_sci_cipsi_frontier_hash(str(name), diagnostic=bool(diagnostic))
    raise ValueError(f"unsupported benchmark {name!r}")


def _resolve_workloads(profile: str, includes: Iterable[str]) -> list[str]:
    include_list = [str(x).strip() for x in includes if str(x).strip()]
    if include_list:
        bad = [x for x in include_list if x not in WORKLOAD_NAMES]
        if bad:
            raise ValueError(f"unknown workloads: {', '.join(sorted(bad))}")
        return include_list
    if str(profile) == "full":
        return list(FULL_WORKLOADS)
    return list(QUICK_WORKLOADS)


def _run_workload(name: str, *, output_dir: Path, diagnostic: bool = False) -> list[dict[str, Any]]:
    if name in WORKLOAD_BENCHMARKS:
        return [_run_benchmark(bench, output_dir=output_dir, diagnostic=bool(diagnostic)) for bench in WORKLOAD_BENCHMARKS[name]]
    raise ValueError(f"unsupported workload {name!r}")


def _ncu_smoke(timeout_s: float) -> dict[str, Any]:
    ncu = shutil.which("ncu")
    if not ncu:
        return {"available": False, "reason": "ncu not found"}
    cmd = [
        ncu,
        "--target-processes",
        "all",
        "--metrics",
        "gpu__time_duration.sum",
        "--csv",
        sys.executable,
        "-c",
        "import cupy as cp; x=cp.arange(4096,dtype=cp.float32); y=x*2; cp.cuda.runtime.deviceSynchronize(); print(float(y.sum()))",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=float(timeout_s), check=False)
        combined = "\n".join(part for part in [proc.stdout, proc.stderr] if part).strip()
        reason = None
        if proc.returncode != 0:
            if "ERR_NVGPUCTRPERM" in combined:
                reason = "ERR_NVGPUCTRPERM: GPU performance counters are not accessible for this user/device"
            elif combined:
                reason = combined[-500:]
        return {
            "available": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "reason": reason,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": f"ncu smoke timed out after {timeout_s:.1f}s"}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def _run_nsys_stats(report_path: Path, *, report: str) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    if not nsys:
        return {"status": "skipped", "reason": "nsys not found"}
    cmd = [nsys, "stats", "--force-export=true", "--report", str(report), "--format", "csv", str(report_path)]
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, check=False)
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {
            "status": "error",
            "returncode": int(proc.returncode),
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }
    rows = _csv_rows_from_text(proc.stdout)
    return {"status": "ok", "rows": rows, "stdout_tail": proc.stdout[-2000:], "stderr_tail": proc.stderr[-2000:]}


def _profile_benchmark_with_nsys(benchmark: str, *, output_dir: Path, timeout_s: float) -> dict[str, Any]:
    nsys = shutil.which("nsys")
    if not nsys:
        return {"benchmark": str(benchmark), "status": "skipped", "reason": "nsys not found"}

    nsight_dir = output_dir / "nsight" / str(benchmark)
    nsight_dir.mkdir(parents=True, exist_ok=True)
    rep_prefix = nsight_dir / "profile"
    child_json = nsight_dir / "benchmark.json"
    cmd = [
        nsys,
        "profile",
        "-o",
        str(rep_prefix),
        "--sample=none",
        "--trace=cuda,nvtx",
        sys.executable,
        "-m",
        "asuka.cli.cuda_audit",
        "--single-benchmark",
        str(benchmark),
        "--json-output",
        str(child_json),
        "--no-summary",
    ]
    env = os.environ.copy()
    env.setdefault("ASUKA_NVTX", "1")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"benchmark": str(benchmark), "status": "error", "error": f"nsys timed out after {timeout_s:.1f}s"}
    except Exception as exc:
        return {"benchmark": str(benchmark), "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    report_path = rep_prefix.with_suffix(".nsys-rep")
    if proc.returncode != 0 or not report_path.exists():
        return {
            "benchmark": str(benchmark),
            "status": "error",
            "error": f"nsys exited with code {proc.returncode}",
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }

    kernel_stats = _run_nsys_stats(report_path, report="cuda_gpu_kern_sum")
    trace_stats = _run_nsys_stats(report_path, report="cuda_gpu_trace")
    api_stats = _run_nsys_stats(report_path, report="cuda_api_sum")

    child_result = None
    try:
        child_payload = json.loads(child_json.read_text(encoding="utf-8"))
        for item in child_payload.get("results", []):
            if str(item.get("name")) == str(benchmark):
                child_result = item
                break
    except Exception:
        child_result = None

    out: dict[str, Any] = {
        "benchmark": str(benchmark),
        "status": "ok",
        "profiler": "nsys",
        "report_path": str(report_path),
        "child_json": str(child_json),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if child_result is not None:
        out["benchmark_result"] = child_result
    if kernel_stats.get("status") == "ok":
        out["kernel_summary"] = _summarize_nsys_kernel_rows(kernel_stats.get("rows", []))
    else:
        out["kernel_summary_error"] = kernel_stats
    if trace_stats.get("status") == "ok":
        out["gpu_trace_summary"] = _summarize_nsys_trace_rows(trace_stats.get("rows", []))
    else:
        out["gpu_trace_summary_error"] = trace_stats
    if api_stats.get("status") == "ok":
        out["cuda_api_summary"] = _summarize_nsys_api_rows(api_stats.get("rows", []))
    else:
        out["cuda_api_summary_error"] = api_stats
    return out


def _nsight_targets_from_summary(summary: dict[str, Any], *, top_n: int) -> list[str]:
    names: list[str] = []
    for group in ("time_ranked", "memory_ranked"):
        rows = summary.get(group, [])
        if not isinstance(rows, list):
            continue
        for row in rows[: max(0, int(top_n))]:
            name = str(row.get("name", "")).strip()
            if name and name in BENCHMARK_NAMES and name not in names:
                names.append(name)
    return names


def _collect_nsight_deep_dive(
    *,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    preflight: dict[str, Any],
    output_dir: Path,
    top_n: int,
    timeout_s: float,
) -> dict[str, Any]:
    ncu_smoke = ((preflight.get("nsight") or {}).get("ncu_smoke") or {}) if isinstance(preflight, dict) else {}
    targets = _nsight_targets_from_summary(summary, top_n=int(top_n))
    payload: dict[str, Any] = {
        "enabled": True,
        "targets": targets,
        "ncu_available": bool(ncu_smoke.get("available", False)),
        "ncu_reason": ncu_smoke.get("reason") or ncu_smoke.get("stderr_tail"),
        "results": [],
    }
    by_name = {str(item.get("name")): item for item in results}
    for target in targets:
        entry = _profile_benchmark_with_nsys(target, output_dir=output_dir, timeout_s=float(timeout_s))
        payload["results"].append(entry)
        if entry.get("status") == "ok" and target in by_name:
            by_name[target]["nsight"] = entry
    return payload


PROMOTION_GATE_DEFAULTS = {
    "min_total_time_improvement_pct": 5.0,
    "min_gradient_time_improvement_pct": 7.0,
    "max_peak_driver_vram_increase_pct": 10.0,
    "max_spotcheck_regression_pct": 5.0,
}


def _build_promotion_gate_snapshot(results: list[dict[str, Any]]) -> dict[str, Any]:
    snapshot: list[dict[str, Any]] = []
    for item in results:
        if item.get("status") != "ok":
            continue
        timing = item.get("timing", {})
        memory = item.get("memory", {})
        row: dict[str, Any] = {
            "name": str(item.get("name")),
            "family": str(item.get("family")),
        }
        if isinstance(timing.get("wall_s"), dict):
            row["t_total_s"] = _safe_float(timing["wall_s"].get("t_total"))
            row["t_grad_s"] = _safe_float(timing["wall_s"].get("t_grad"))
        if isinstance(timing.get("event_ms"), dict):
            row["event_ms_median"] = _safe_float(timing["event_ms"].get("median"))
            row["event_ms_jitter_pct"] = _safe_float(timing["event_ms"].get("jitter_pct"))
        if isinstance(timing.get("wall_ms"), dict):
            row["wall_ms_median"] = _safe_float(timing["wall_ms"].get("median"))
            row["wall_ms_jitter_pct"] = _safe_float(timing["wall_ms"].get("jitter_pct"))
        row["peak_driver_used_bytes"] = _safe_float(memory.get("peak_driver_used_bytes"))
        row["peak_pool_total_bytes"] = _safe_float(memory.get("peak_pool_total_bytes"))
        snapshot.append(row)
    return {
        "policy": dict(PROMOTION_GATE_DEFAULTS),
        "snapshot": snapshot,
    }


def _tool_version(binary: str) -> dict[str, Any]:
    path = shutil.which(binary)
    if not path:
        return {"found": False}
    try:
        proc = subprocess.run([path, "--version"], capture_output=True, text=True, check=False)
        text = (proc.stdout or proc.stderr).strip()
    except Exception as exc:
        return {"found": True, "path": path, "error": f"{type(exc).__name__}: {exc}"}
    return {"found": True, "path": path, "version": text}


def run_preflight(*, nsight_smoke_timeout_s: float = 20.0) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "kernel_inventory": kernel_report(),
        "nsight": {
            "ncu": _tool_version("ncu"),
            "nsys": _tool_version("nsys"),
        },
    }
    try:
        cp = _cupy()
        device_count = _cuda_init(cp)
        props = cp.cuda.runtime.getDeviceProperties(0) if device_count > 0 else {}
        name = props.get("name", b"") if isinstance(props, dict) else b""
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        out["cuda"] = {
            "cupy_version": str(cp.__version__),
            "runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_count": int(device_count),
            "device_name": str(name),
            "memory": _mem_snapshot(cp),
        }
    except Exception as exc:
        out["cuda_error"] = f"{type(exc).__name__}: {exc}"

    out["nsight"]["ncu_smoke"] = _ncu_smoke(float(nsight_smoke_timeout_s))
    return out


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    time_ranked: list[dict[str, Any]] = []
    mem_ranked: list[dict[str, Any]] = []

    for item in results:
        if item.get("status") != "ok":
            continue
        name = str(item.get("name"))
        family = str(item.get("family"))
        timing = item.get("timing", {})
        memory = item.get("memory", {})
        event_med = _safe_float(timing.get("event_ms", {}).get("median")) if isinstance(timing.get("event_ms"), dict) else None
        wall_med = _safe_float(timing.get("wall_ms", {}).get("median")) if isinstance(timing.get("wall_ms"), dict) else None
        wall_s_total = None
        if isinstance(timing.get("wall_s"), dict):
            wall_s_total = _safe_float(timing["wall_s"].get("t_total"))
        time_score = event_med if event_med is not None else wall_med
        if time_score is None and wall_s_total is not None:
            time_score = float(wall_s_total * 1000.0)

        mem_score = None
        if isinstance(memory.get("delta"), dict):
            mem_score = _safe_float(memory["delta"].get("pool_total_bytes_delta"))
            if mem_score is None:
                mem_score = _safe_float(memory["delta"].get("driver_used_bytes_delta"))
        if mem_score is None:
            mem_score = _safe_float(memory.get("peak_pool_total_bytes"))
        if mem_score is None:
            mem_score = _safe_float(memory.get("peak_driver_used_bytes"))

        if time_score is not None:
            time_ranked.append({"name": name, "family": family, "time_score_ms": float(time_score)})
        if mem_score is not None:
            mem_ranked.append({"name": name, "family": family, "memory_score_bytes": int(mem_score)})

    time_ranked.sort(key=lambda x: (-float(x["time_score_ms"]), str(x["name"])))
    mem_ranked.sort(key=lambda x: (-int(x["memory_score_bytes"]), str(x["name"])))
    return {
        "time_ranked": time_ranked,
        "memory_ranked": mem_ranked,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("Top workloads by time")
    for row in summary.get("time_ranked", [])[:10]:
        print(f"  {row['name']:<28s} {row['time_score_ms']:10.3f} ms  [{row['family']}]")
    print("Top workloads by space")
    for row in summary.get("memory_ranked", [])[:10]:
        gib = float(row["memory_score_bytes"]) / float(1024**3)
        print(f"  {row['name']:<28s} {gib:10.3f} GiB  [{row['family']}]")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit CUDA kernel time/space efficiency in ASUKA.")
    p.add_argument("--profile", choices=["quick", "full"], default="quick", help="Workload profile to run.")
    p.add_argument(
        "--run-kind",
        choices=["timed", "diagnostic", "both"],
        default="both",
        help="Run timed baselines, diagnostic profiles, or both.",
    )
    p.add_argument("--include", action="append", default=[], help="Specific workload(s) to run.")
    p.add_argument("--output-dir", type=Path, default=None, help="Artifact directory. Defaults to a timestamped directory under benchmark_results/cuda_audit.")
    p.add_argument("--json-output", type=Path, default=None, help="Optional path for the top-level JSON artifact.")
    p.add_argument("--preflight-only", action="store_true", help="Only run environment/kernel inventory checks.")
    p.add_argument("--single-workload", choices=WORKLOAD_NAMES, default=None, help="Run exactly one workload and write its JSON result.")
    p.add_argument("--single-benchmark", choices=BENCHMARK_NAMES, default=None, help="Run exactly one benchmark and write its JSON result.")
    p.add_argument("--nsight-smoke-timeout", type=float, default=20.0, help="Timeout in seconds for the ncu smoke test.")
    p.add_argument("--nsight-deep-dive", action="store_true", help="Profile top-ranked benchmarks with Nsight Systems and attach structured summaries.")
    p.add_argument("--nsight-top-n", type=int, default=3, help="Top N time-ranked and memory-ranked benchmarks to deep-profile.")
    p.add_argument("--nsight-timeout", type=float, default=120.0, help="Timeout in seconds per Nsight Systems deep-dive capture.")
    p.add_argument("--no-summary", action="store_true", help="Suppress human-readable summary output.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (REPO_ROOT / "benchmark_results" / "cuda_audit" / ts)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.single_benchmark is not None:
        results = [_run_benchmark(str(args.single_benchmark), output_dir=output_dir, diagnostic=(str(args.run_kind) == "diagnostic"))]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "mode": "single_benchmark",
            "benchmark": str(args.single_benchmark),
            "run_kind": str(args.run_kind),
            "results": results,
        }
        out_path = args.json_output or (output_dir / f"{args.single_benchmark}.json")
        _write_json(out_path, payload)
        if not args.no_summary:
            print(f"Wrote {out_path}")
        return 0

    if args.single_workload is not None:
        results = _run_workload(
            str(args.single_workload),
            output_dir=output_dir,
            diagnostic=(str(args.run_kind) == "diagnostic"),
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "mode": "single_workload",
            "workload": str(args.single_workload),
            "run_kind": str(args.run_kind),
            "results": results,
        }
        out_path = args.json_output or (output_dir / f"{args.single_workload}.json")
        _write_json(out_path, payload)
        if not args.no_summary:
            print(f"Wrote {out_path}")
        return 0

    preflight = run_preflight(nsight_smoke_timeout_s=float(args.nsight_smoke_timeout))
    if args.preflight_only:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": _utc_now(),
            "mode": "preflight_only",
            "preflight": preflight,
            "results": [],
            "summary": {"time_ranked": [], "memory_ranked": []},
        }
        out_path = args.json_output or (output_dir / "audit.json")
        _write_json(out_path, payload)
        if not args.no_summary:
            print(f"Wrote {out_path}")
        return 0

    workloads = _resolve_workloads(str(args.profile), args.include)
    timed_results: list[dict[str, Any]] = []
    diagnostic_results: list[dict[str, Any]] = []
    if str(args.run_kind) in {"timed", "both"}:
        for workload in workloads:
            timed_results.extend(_run_workload(workload, output_dir=output_dir, diagnostic=False))
    if str(args.run_kind) in {"diagnostic", "both"}:
        for workload in workloads:
            diagnostic_results.extend(_run_workload(workload, output_dir=output_dir, diagnostic=True))

    timed_block = _payload_summary_block(timed_results)
    diagnostic_block = _payload_summary_block(diagnostic_results)
    summary = timed_block["summary"] if timed_results else diagnostic_block["summary"]
    nsight_deep_dive = None
    if bool(args.nsight_deep_dive) and diagnostic_results:
        nsight_deep_dive = _collect_nsight_deep_dive(
            results=diagnostic_results,
            summary=(timed_block["summary"] if timed_results else diagnostic_block["summary"]),
            preflight=preflight,
            output_dir=output_dir,
            top_n=int(args.nsight_top_n),
            timeout_s=float(args.nsight_timeout),
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": _utc_now(),
        "mode": "audit",
        "profile": str(args.profile),
        "run_kind": str(args.run_kind),
        "workloads": workloads,
        "preflight": preflight,
        "results": timed_results if timed_results else diagnostic_results,
        "summary": summary,
        "timed_baseline": timed_block,
        "diagnostic_profile": {
            **diagnostic_block,
            "nsight_deep_dive": nsight_deep_dive,
        },
        "promotion_gate": _build_promotion_gate_snapshot(timed_results if timed_results else diagnostic_results),
        "nsight_deep_dive": nsight_deep_dive,
    }
    out_path = args.json_output or (output_dir / "audit.json")
    _write_json(out_path, payload)
    if not args.no_summary:
        print(f"Wrote {out_path}")
        _print_summary(summary)
    return 0


__all__ = [
    "BENCHMARK_NAMES",
    "FULL_WORKLOADS",
    "QUICK_WORKLOADS",
    "SCHEMA_VERSION",
    "WORKLOAD_NAMES",
    "main",
    "run_preflight",
    "summarize_results",
]
