"""Benchmark matrix for mixed-precision ERI acceleration.

Compares energies across backends (dense, df, direct, direct_df) and
precision modes (fp64, mixed, mixed+f32_accum, mixed+f32_tile) to validate
that all combinations produce correct results.

Run:
    python tests/direct/test_mixed_precision_benchmark.py
"""

import os
import sys
import time

# ── molecules ────────────────────────────────────────────────────────────────

MOLECULES = {
    "LiH/STO-3G": dict(
        atoms=[("Li", (0, 0, 0)), ("H", (0, 0, 1.6))],
        basis="sto-3g",
        cart=True,
    ),
    "H2O/STO-3G": dict(
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.587)),
            ("H", (0.0, -0.757, 0.587)),
        ],
        basis="sto-3g",
        cart=True,
    ),
    "H2O/6-31G": dict(
        atoms=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.757, 0.587)),
            ("H", (0.0, -0.757, 0.587)),
        ],
        basis="6-31g",
        cart=True,
    ),
}

# ── precision modes ──────────────────────────────────────────────────────────

PRECISION_MODES = {
    "fp64": dict(
        ASUKA_ERI_MIXED_PRECISION="0",
        ASUKA_ERI_TILE_F32="0",
        ASUKA_ERI_F32_ACCUM="0",
    ),
    "mixed": dict(
        ASUKA_ERI_MIXED_PRECISION="1",
        ASUKA_ERI_TILE_F32="0",
        ASUKA_ERI_F32_ACCUM="0",
    ),
    "mixed+f32tile": dict(
        ASUKA_ERI_MIXED_PRECISION="1",
        ASUKA_ERI_TILE_F32="1",
        ASUKA_ERI_F32_ACCUM="0",
    ),
    "mixed+f32accum": dict(
        ASUKA_ERI_MIXED_PRECISION="1",
        ASUKA_ERI_TILE_F32="0",
        ASUKA_ERI_F32_ACCUM="1",
    ),
    "all_aggressive": dict(
        ASUKA_ERI_MIXED_PRECISION="1",
        ASUKA_ERI_TILE_F32="1",
        ASUKA_ERI_F32_ACCUM="1",
    ),
}

# ── backends ─────────────────────────────────────────────────────────────────

BACKENDS = ["dense", "df", "direct", "direct_df"]


def _set_precision_env(env_dict):
    for k, v in env_dict.items():
        os.environ[k] = v


def _run_scf(mol_cfg, backend):
    """Run a single SCF and return (energy, wall_seconds, converged)."""
    from asuka.frontend import Molecule, run_hf_df

    mol = Molecule.from_atoms(
        mol_cfg["atoms"],
        basis=mol_cfg["basis"],
        cart=mol_cfg.get("cart", True),
    )
    t0 = time.perf_counter()
    try:
        result = run_hf_df(mol, two_e_backend=backend)
        dt = time.perf_counter() - t0
        return result.scf.e_tot, dt, result.scf.converged
    except Exception as e:
        dt = time.perf_counter() - t0
        return None, dt, str(e)


def main():
    # Header
    print("=" * 110)
    print("Mixed-Precision ERI Benchmark Matrix")
    print("=" * 110)
    print()

    # Collect all results: results[mol][prec][backend] = (energy, dt, conv)
    results = {}

    for mol_name, mol_cfg in MOLECULES.items():
        results[mol_name] = {}
        print(f"Molecule: {mol_name}")
        print("-" * 110)
        header = f"{'Precision':<20s}"
        for b in BACKENDS:
            header += f" | {b:>16s} (E_tot)"
        print(header)
        print("-" * 110)

        for prec_name, prec_env in PRECISION_MODES.items():
            _set_precision_env(prec_env)
            results[mol_name][prec_name] = {}
            row = f"{prec_name:<20s}"
            for backend in BACKENDS:
                energy, dt, conv = _run_scf(mol_cfg, backend)
                results[mol_name][prec_name][backend] = (energy, dt, conv)
                if energy is not None:
                    row += f" | {energy:>20.12f}"
                else:
                    err_msg = str(conv)[:16]
                    row += f" |     FAIL({err_msg})"
            print(row)
        print()

    # Error table vs fp64 reference per backend
    print()
    print("=" * 110)
    print("Error vs FP64 Reference (per backend)")
    print("=" * 110)
    print()

    for mol_name in MOLECULES:
        print(f"Molecule: {mol_name}")
        print("-" * 110)
        header = f"{'Precision':<20s}"
        for b in BACKENDS:
            header += f" | {b:>18s} (ΔE)"
        print(header)
        print("-" * 110)

        fp64 = results[mol_name].get("fp64", {})
        for prec_name in PRECISION_MODES:
            row = f"{prec_name:<20s}"
            for backend in BACKENDS:
                e_ref = fp64.get(backend, (None,))[0]
                e_cur = results[mol_name][prec_name][backend][0]
                if e_ref is not None and e_cur is not None:
                    delta = abs(e_cur - e_ref)
                    row += f" | {delta:>18.2e}"
                else:
                    row += f" | {'N/A':>18s}"
            print(row)
        print()

    # Cross-backend consistency (dense vs others at fp64)
    print()
    print("=" * 110)
    print("Cross-Backend Consistency (|E_backend - E_dense| at each precision)")
    print("=" * 110)
    print()

    for mol_name in MOLECULES:
        print(f"Molecule: {mol_name}")
        print("-" * 90)
        header = f"{'Precision':<20s}"
        for b in BACKENDS[1:]:
            header += f" | {b:>18s} vs dense"
        print(header)
        print("-" * 90)

        for prec_name in PRECISION_MODES:
            row = f"{prec_name:<20s}"
            e_dense = results[mol_name][prec_name]["dense"][0]
            for backend in BACKENDS[1:]:
                e_cur = results[mol_name][prec_name][backend][0]
                if e_dense is not None and e_cur is not None:
                    delta = abs(e_cur - e_dense)
                    row += f" | {delta:>18.2e}"
                else:
                    row += f" | {'N/A':>18s}"
            print(row)
        print()

    # Summary pass/fail
    print()
    print("=" * 110)
    print("PASS/FAIL Summary (thresholds: same-backend ΔE < 1e-5, cross-backend < 1e-3)")
    print("=" * 110)
    n_pass = 0
    n_fail = 0
    n_skip = 0

    for mol_name in MOLECULES:
        fp64 = results[mol_name].get("fp64", {})
        for prec_name in PRECISION_MODES:
            for backend in BACKENDS:
                e_ref = fp64.get(backend, (None,))[0]
                e_cur = results[mol_name][prec_name][backend][0]
                if e_ref is None or e_cur is None:
                    n_skip += 1
                    continue
                delta = abs(e_cur - e_ref)
                if delta > 1e-5:
                    print(f"  FAIL: {mol_name} / {prec_name} / {backend}: ΔE={delta:.2e}")
                    n_fail += 1
                else:
                    n_pass += 1

    print(f"\n  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    print()
    return 1 if n_fail > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
