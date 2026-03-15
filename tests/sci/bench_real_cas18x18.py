#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from asuka.frontend import Molecule, run_rhf_df
from asuka.cuguga.drt import build_drt
from asuka.sci.gpu_cipsi import build_cipsi_trials_from_scf


@dataclass(frozen=True)
class RealCASCase:
    name: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    basis: str
    charge: int
    spin: int
    ncore: int
    ncas: int
    nelecas: int


CASES: dict[str, RealCASCase] = {
    "co2_631g_cas1818": RealCASCase(
        name="co2_631g_cas1818",
        atoms=(
            ("O", (0.0, 0.0, -1.16)),
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, 1.16)),
        ),
        basis="6-31g",
        charge=0,
        spin=0,
        ncore=2,
        ncas=18,
        nelecas=18,
    ),
    "co2_ccpvdz_cas1818": RealCASCase(
        name="co2_ccpvdz_cas1818",
        atoms=(
            ("O", (0.0, 0.0, -1.16)),
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, 1.16)),
        ),
        basis="cc-pvdz",
        charge=0,
        spin=0,
        ncore=2,
        ncas=18,
        nelecas=18,
    ),
    "co2_631g_cas2222": RealCASCase(
        name="co2_631g_cas2222",
        atoms=(
            ("O", (0.0, 0.0, -1.16)),
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, 1.16)),
        ),
        basis="6-31g",
        charge=0,
        spin=0,
        ncore=0,
        ncas=22,
        nelecas=22,
    ),
    "o3_631g_cas1818": RealCASCase(
        name="o3_631g_cas1818",
        atoms=(
            ("O", (0.0, 0.0, 0.000000)),
            ("O", (0.0, 0.0, 1.278000)),
            ("O", (1.218000, 0.0, -0.527000)),
        ),
        basis="6-31g",
        charge=0,
        spin=0,
        ncore=3,
        ncas=18,
        nelecas=18,
    ),
}


def _cupy_or_none():
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except Exception:
        return None
    return cp


def _mem_snapshot(cp: Any) -> dict[str, float] | None:
    if cp is None:
        return None
    try:
        cp.cuda.runtime.deviceSynchronize()
        pool = cp.get_default_memory_pool()
        free_b, total_b = cp.cuda.runtime.memGetInfo()
    except Exception:
        return None
    return {
        "driver_used_gb": float(total_b - free_b) / (1024.0 ** 3),
        "driver_total_gb": float(total_b) / (1024.0 ** 3),
        "pool_used_gb": float(pool.used_bytes()) / (1024.0 ** 3),
        "pool_total_gb": float(pool.total_bytes()) / (1024.0 ** 3),
    }


def _asnumpy_f64(cp: Any, arr: Any) -> np.ndarray:
    if cp is not None and isinstance(arr, cp.ndarray):
        return np.asarray(cp.asnumpy(arr), dtype=np.float64)
    return np.asarray(arr, dtype=np.float64)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a real-molecule CAS(18,18) SCI setup.")
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="co2_631g_cas1818",
        help="Real molecule / basis / active-space preset.",
    )
    parser.add_argument("--nsel", type=int, default=2048, help="Fixed selected-space size.")
    parser.add_argument("--init-ncsf", type=int, default=None, help="Initial selected-space size. Defaults to --nsel.")
    parser.add_argument("--max-ncsf", type=int, default=None, help="Maximum selected-space size. Defaults to --nsel.")
    parser.add_argument("--grow-by", type=int, default=0, help="Selected states added per macro iteration.")
    parser.add_argument("--max-iter", type=int, default=0, help="Number of macro iterations to run.")
    parser.add_argument("--selection-mode", choices=("heat_bath", "frontier_hash"), default="heat_bath")
    parser.add_argument("--backend", default="cuda_key64")
    parser.add_argument("--hb-epsilon", type=float, default=1e-4)
    parser.add_argument("--hb-eps-schedule", choices=("fixed", "adaptive"), default="fixed")
    parser.add_argument("--hb-eps-init", type=float, default=1e-3)
    parser.add_argument("--hb-eps-final", type=float, default=1e-4)
    parser.add_argument(
        "--hf-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Seed the initial selected space with the RHF reference CSF.",
    )
    parser.add_argument(
        "--hb-cuda-selector-min-nsel",
        type=int,
        default=None,
        help="Override ASUKA_HB_CUDA_SELECTOR_MIN_NSEL for this run.",
    )
    parser.add_argument(
        "--macro-growth-steps",
        type=int,
        default=None,
        help="Override workspace macro growth steps.",
    )
    parser.add_argument("--davidson-max-cycle", type=int, default=40)
    parser.add_argument("--davidson-max-space", type=int, default=16)
    parser.add_argument("--davidson-tol", type=float, default=1e-7)
    parser.add_argument("--graph-min-nsel", type=int, default=1024)
    parser.add_argument(
        "--projected-solver-matrix-free",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--auxbasis", default="autoaux")
    parser.add_argument("--scf-max-cycle", type=int, default=100)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-8)
    parser.add_argument("--scf-conv-tol-dm", type=float, default=1e-6)
    parser.add_argument("--scf-damping", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = _build_args()
    case = CASES[str(args.case)]
    cp = _cupy_or_none()
    init_ncsf = int(args.nsel if args.init_ncsf is None else args.init_ncsf)
    max_ncsf = int(args.nsel if args.max_ncsf is None else args.max_ncsf)

    os.environ.setdefault("ASUKA_HB_SELECTED_GRAPH_BUILDER", "tuple_emit")
    os.environ.setdefault("ASUKA_GPU_PROJECTED_EIGENSOLVER", "davidson")
    os.environ["ASUKA_HB_SELECTED_GRAPH_NSEL_MIN"] = str(int(args.graph_min_nsel))
    if args.hb_cuda_selector_min_nsel is not None:
        os.environ["ASUKA_HB_CUDA_SELECTOR_MIN_NSEL"] = str(int(args.hb_cuda_selector_min_nsel))

    mol = Molecule.from_atoms(
        case.atoms,
        unit="Angstrom",
        charge=int(case.charge),
        spin=int(case.spin),
        basis=str(case.basis),
        cart=True,
    )

    mem_before = _mem_snapshot(cp)

    t0 = time.perf_counter()
    scf = run_rhf_df(
        mol,
        auxbasis=str(args.auxbasis),
        max_cycle=int(args.scf_max_cycle),
        conv_tol=float(args.scf_conv_tol),
        conv_tol_dm=float(args.scf_conv_tol_dm),
        damping=float(args.scf_damping),
    )
    scf_wall = float(time.perf_counter() - t0)

    ci0 = None
    if bool(args.hf_seed):
        mo_occ = _asnumpy_f64(cp, getattr(scf.scf, "mo_occ")).ravel()
        occ_act = mo_occ[int(case.ncore) : int(case.ncore) + int(case.ncas)]
        steps: list[str] = []
        for occ in occ_act.tolist():
            if occ > 1.5:
                steps.append("D")
            elif occ > 0.5:
                steps.append("S")
            else:
                steps.append("E")
        drt = build_drt(norb=int(case.ncas), nelec=int(case.nelecas), twos_target=int(case.spin))
        hf_idx = int(drt.path_to_index(steps))
        ci0 = [(np.asarray([hf_idx], dtype=np.int64), np.asarray([1.0], dtype=np.float64))]

    workspace_kwargs = {
        "projected_solver_gpu": True,
        "projected_solver_matrix_free": bool(args.projected_solver_matrix_free),
    }
    if args.macro_growth_steps is not None:
        workspace_kwargs["macro_growth_steps"] = int(args.macro_growth_steps)

    t1 = time.perf_counter()
    res = build_cipsi_trials_from_scf(
        scf,
        ncore=int(case.ncore),
        ncas=int(case.ncas),
        nelecas=int(case.nelecas),
        nroots=1,
        df=True,
        backend=str(args.backend),
        epq_mode="no_epq_support_aware",
        ci0=ci0,
        init_ncsf=int(init_ncsf),
        max_ncsf=int(max_ncsf),
        grow_by=int(args.grow_by),
        max_iter=int(args.max_iter),
        selection_mode=str(args.selection_mode),
        hb_epsilon=float(args.hb_epsilon),
        hb_eps_schedule=str(args.hb_eps_schedule),
        hb_eps_init=float(args.hb_eps_init),
        hb_eps_final=float(args.hb_eps_final),
        davidson_max_cycle=int(args.davidson_max_cycle),
        davidson_max_space=int(args.davidson_max_space),
        davidson_tol=float(args.davidson_tol),
        workspace_kwargs=workspace_kwargs,
    )
    sci_wall = float(time.perf_counter() - t1)

    mem_after = _mem_snapshot(cp)
    summary = {
        "case": str(case.name),
        "basis": str(case.basis),
        "ncore": int(case.ncore),
        "ncas": int(case.ncas),
        "nelecas": int(case.nelecas),
        "hf_e_tot": float(getattr(scf, "e_tot")),
        "init_ncsf": int(init_ncsf),
        "max_ncsf": int(max_ncsf),
        "grow_by": int(args.grow_by),
        "max_iter": int(args.max_iter),
        "nsel": int(len(res.sel_idx)),
        "history_len": int(len(res.history)),
        "hf_seed": bool(args.hf_seed),
        "rhf_converged": bool(getattr(scf.scf, "converged", False)),
        "scf_wall_s": float(scf_wall),
        "sci_wall_s": float(sci_wall),
        "e_var": float(np.asarray(res.e_var, dtype=np.float64).ravel()[0]),
        "e_pt2": float(np.asarray(res.e_pt2, dtype=np.float64).ravel()[0]),
        "e_tot": float(np.asarray(res.e_tot, dtype=np.float64).ravel()[0]),
        "backend_effective": res.profile.get("backend_effective"),
        "projected_solver_backend": res.profile.get("projected_solver_backend"),
        "projected_solver_route_taken": res.profile.get("projected_solver_route_taken"),
        "projected_solver_dense_input_source": res.profile.get("projected_solver_dense_input_source"),
        "projected_solver_parity_check_enabled": res.profile.get("projected_solver_parity_check_enabled"),
        "projected_solver_parity_check_reason": res.profile.get("projected_solver_parity_check_reason"),
        "hb_index_build_backend": res.profile.get("hb_index_build_backend"),
        "df_integrals_device_promoted": res.profile.get("df_integrals_device_promoted"),
        "selector_backend_history": res.profile.get("selector_backend_history"),
        "history": res.history,
        "timings_s": res.profile.get("timings_s"),
        "memory_before_gb": mem_before,
        "memory_after_gb": mem_after,
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
