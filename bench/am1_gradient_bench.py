#!/usr/bin/env python3
"""Benchmark AM1 energy+gradient runtime for semiempirical SCF."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from asuka.semiempirical import SemiempiricalCalculator
from asuka.semiempirical.basis import symbol_to_Z
from asuka.semiempirical.gradient import am1_energy_gradient_scf
from asuka.semiempirical.gpu import has_cupy, has_cuda_device
from asuka.semiempirical.params import ANGSTROM_TO_BOHR


def _water_cluster_case(name: str, n_waters: int, spacing: float = 3.0) -> dict:
    base = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.757, 0.586],
            [0.0, -0.757, 0.586],
        ],
        dtype=float,
    )
    symbols = []
    coords = []
    for i in range(int(n_waters)):
        x = (i % 8) * spacing
        y = ((i // 8) % 8) * spacing
        z = (i // 64) * spacing
        shift = np.array([x, y, z], dtype=float)
        symbols.extend(["O", "H", "H"])
        coords.extend((base + shift).tolist())
    return {"name": name, "symbols": symbols, "coords_angstrom": coords}


CASE_CATALOG = {
    "h2": {
        "name": "h2",
        "symbols": ["H", "H"],
        "coords_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    },
    "hcn": {
        "name": "hcn",
        "symbols": ["H", "C", "N"],
        "coords_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.06], [0.0, 0.0, 2.22]],
    },
    "ch4": {
        "name": "ch4",
        "symbols": ["C", "H", "H", "H", "H"],
        "coords_angstrom": [
            [0.0000, 0.0000, 0.0000],
            [0.6291, 0.6291, 0.6291],
            [0.6291, -0.6291, -0.6291],
            [-0.6291, 0.6291, -0.6291],
            [-0.6291, -0.6291, 0.6291],
        ],
    },
    "h2o": {
        "name": "h2o",
        "symbols": ["O", "H", "H"],
        "coords_angstrom": [[0.0, 0.0, 0.0], [0.0, 0.757, 0.586], [0.0, -0.757, 0.586]],
    },
    "water_16": _water_cluster_case("water_16", 16),
    "water_24": _water_cluster_case("water_24", 24),
    "water_32": _water_cluster_case("water_32", 32),
}

def _run_single(
    symbols,
    coords_angstrom,
    *,
    device: str,
    fock_mode: str,
    gradient_backend: str,
    max_iter: int,
    conv_tol: float,
    step_bohr: float,
):
    coords_ang = np.asarray(coords_angstrom, dtype=float)
    coords_bohr = coords_ang * ANGSTROM_TO_BOHR
    atomic_numbers = [symbol_to_Z(s) for s in symbols]
    calc = SemiempiricalCalculator(method="AM1", charge=0, device=device)

    t0 = time.perf_counter()
    scf = calc.energy(
        symbols,
        coords_ang,
        coords_unit="angstrom",
        max_iter=max_iter,
        conv_tol=conv_tol,
        fock_mode=fock_mode,
    )
    t1 = time.perf_counter()
    _scf, grad, grad_meta = am1_energy_gradient_scf(
        atomic_numbers=atomic_numbers,
        coords_bohr=coords_bohr,
        params=calc.params,
        scf_result=scf,
        step_bohr=step_bohr,
        device=device,
        fock_mode=fock_mode,
        gradient_backend=gradient_backend,
        return_metadata=True,
    )
    t2 = time.perf_counter()

    return {
        "scf_time_s": float(t1 - t0),
        "gradient_time_s": float(t2 - t1),
        "total_time_s": float(t2 - t0),
        "n_iter": int(scf.n_iter),
        "converged": bool(scf.converged),
        "energy_total_ha": float(scf.energy_total),
        "gradient_norm_ha_bohr": float(np.linalg.norm(grad)),
        "gradient_backend_used": str(grad_meta.get("gradient_backend_used", "cpu_frozen")),
        "gradient_pack_time_s": float(grad_meta.get("gradient_pack_time_s", 0.0)),
        "gradient_kernel_time_s": float(grad_meta.get("gradient_kernel_time_s", 0.0)),
        "gradient_post_time_s": float(grad_meta.get("gradient_post_time_s", 0.0)),
    }


def _summarize(samples):
    scf = np.asarray([x["scf_time_s"] for x in samples], dtype=float)
    grad = np.asarray([x["gradient_time_s"] for x in samples], dtype=float)
    total = np.asarray([x["total_time_s"] for x in samples], dtype=float)
    n_iter = np.asarray([x["n_iter"] for x in samples], dtype=float)
    pack = np.asarray([x["gradient_pack_time_s"] for x in samples], dtype=float)
    kernel = np.asarray([x["gradient_kernel_time_s"] for x in samples], dtype=float)
    post = np.asarray([x["gradient_post_time_s"] for x in samples], dtype=float)
    return {
        "repeat": int(len(samples)),
        "scf_time_s_mean": float(np.mean(scf)),
        "gradient_time_s_mean": float(np.mean(grad)),
        "gradient_time_s_median": float(np.median(grad)),
        "total_time_s_mean": float(np.mean(total)),
        "total_time_s_median": float(np.median(total)),
        "total_time_s_min": float(np.min(total)),
        "total_time_s_max": float(np.max(total)),
        "n_iter_mean": float(np.mean(n_iter)),
        "n_iter_max": int(np.max(n_iter)),
        "converged_all": bool(all(s["converged"] for s in samples)),
        "gradient_pack_time_s_mean": float(np.mean(pack)),
        "gradient_kernel_time_s_mean": float(np.mean(kernel)),
        "gradient_post_time_s_mean": float(np.mean(post)),
    }


def _resolve_cases(case_csv: str):
    selected = []
    for name in [x.strip() for x in str(case_csv).split(",") if x.strip()]:
        if name not in CASE_CATALOG:
            choices = ", ".join(sorted(CASE_CATALOG.keys()))
            raise SystemExit(f"Unknown case '{name}'. Available cases: {choices}")
        selected.append(CASE_CATALOG[name])
    if not selected:
        raise SystemExit("--cases resolved to an empty case list")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark AM1 energy+gradient runtimes")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/bench/am1_gradient_baseline.json"),
        help="Output JSON file path",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Warm repeats per case/backend/mode")
    parser.add_argument("--warmup", type=int, default=1, help="Discarded warmup runs after cold run")
    parser.add_argument("--max-iter", type=int, default=120, help="SCF max iterations")
    parser.add_argument("--conv-tol", type=float, default=1e-9, help="SCF convergence tolerance")
    parser.add_argument("--step-bohr", type=float, default=1e-4, help="Gradient central-difference step (Bohr)")
    parser.add_argument(
        "--gradient-backend",
        type=str,
        default="auto",
        choices=["auto", "cuda_analytic", "cpu_frozen"],
        help="Gradient backend mode",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="h2,hcn,ch4,h2o,water_16",
        help="Comma-separated benchmark cases (e.g., water_16,water_24,water_32)",
    )
    parser.add_argument("--strict", action="store_true", help="Fail on errors or missing CUDA")
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    cuda_ready = has_cupy() and has_cuda_device()
    cases = _resolve_cases(args.cases)
    backends = [("cpu", "ri"), ("cuda", "auto"), ("cuda", "w"), ("cuda", "ri")]

    rows = []
    had_error = False
    for case in cases:
        for backend, mode in backends:
            row = {
                "case": case["name"],
                "natm": int(len(case["symbols"])),
                "backend": backend,
                "fock_mode": mode,
                "gradient_backend": str(args.gradient_backend),
                "status": "ok",
            }
            if backend == "cuda" and not cuda_ready:
                row["status"] = "skipped"
                row["reason"] = "CUDA unavailable (CuPy import/device check failed)"
                rows.append(row)
                continue
            try:
                cold_sample = _run_single(
                    case["symbols"],
                    case["coords_angstrom"],
                    device=backend,
                    fock_mode=mode,
                    gradient_backend=args.gradient_backend,
                    max_iter=args.max_iter,
                    conv_tol=args.conv_tol,
                    step_bohr=args.step_bohr,
                )
                for _ in range(args.warmup):
                    _run_single(
                        case["symbols"],
                        case["coords_angstrom"],
                        device=backend,
                        fock_mode=mode,
                        gradient_backend=args.gradient_backend,
                        max_iter=args.max_iter,
                        conv_tol=args.conv_tol,
                        step_bohr=args.step_bohr,
                    )
                warm_samples = []
                for _ in range(args.repeat):
                    warm_samples.append(
                        _run_single(
                            case["symbols"],
                            case["coords_angstrom"],
                            device=backend,
                            fock_mode=mode,
                            gradient_backend=args.gradient_backend,
                            max_iter=args.max_iter,
                            conv_tol=args.conv_tol,
                            step_bohr=args.step_bohr,
                        )
                    )
            except Exception as exc:
                row["status"] = "error"
                row["error"] = f"{type(exc).__name__}: {exc}"
                had_error = True

            if row["status"] == "ok":
                row["cold_sample"] = cold_sample
                row["warmup_discarded"] = int(args.warmup)
                row["warm_summary"] = _summarize(warm_samples)
                row["summary"] = row["warm_summary"]
                row["samples"] = warm_samples
            rows.append(row)

    out = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "max_iter": int(args.max_iter),
        "conv_tol": float(args.conv_tol),
        "step_bohr": float(args.step_bohr),
        "gradient_backend": str(args.gradient_backend),
        "cuda_ready": bool(cuda_ready),
        "cases": [c["name"] for c in cases],
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote benchmark report to {args.output}")

    if args.strict and (had_error or (not cuda_ready)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
