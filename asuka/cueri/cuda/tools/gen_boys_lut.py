#!/usr/bin/env python
"""Generate a Chebyshev interpolation lookup table for the Boys function F_m(T).

The Boys function is defined as:
    F_m(T) = ∫_0^1 t^{2m} exp(-T t^2) dt

This script generates a C header with __constant__ memory arrays containing
Chebyshev coefficients for fast GPU evaluation. The table covers m = 0..MMAX
and T ∈ [0, T_MAX] using NCHEB-order Chebyshev polynomials on NINTERVALS
subintervals. For T > T_MAX, an asymptotic formula is used.

Usage:
    python -m asuka.cueri.cuda.tools.gen_boys_lut [--output PATH]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# Table parameters
MMAX = 21        # covers up to g-shell NROOTS=11 (needs moments up to 2*11-1=21)
NCHEB = 7        # Chebyshev polynomial order (8 coefficients per interval)
NINTERVALS = 200 # number of subintervals over [0, T_MAX]
T_MAX = 50.0     # beyond this, use asymptotic formula


def boys_all(mmax: int, T: float) -> list[float]:
    """Compute F_0(T) through F_mmax(T) using a numerically stable method.

    Uses Miller's algorithm: backward recursion from a high starting index,
    normalized against the exact F_0 from erf. This avoids the precision loss
    of upward recursion for high m and the series convergence issues for
    moderate T.
    """
    if T < 1e-15:
        return [1.0 / (2 * m + 1) for m in range(mmax + 1)]

    e = math.exp(-T)
    # Use pure backward (downward) recursion from a high starting index.
    # F_{k-1} = (2T F_k + e) / (2k - 1) is numerically stable downward.
    # Starting from F_M = 0 at sufficiently large M, the starting error
    # contracts exponentially and becomes negligible.
    # Normalize against the exact F_0 computed from erf.
    F0_exact = 0.5 * math.sqrt(math.pi / T) * math.erf(math.sqrt(T))

    M = max(mmax + 100, int(T) + mmax + 100)
    Fm = 0.0
    all_F = [0.0] * (mmax + 1)
    for mm in range(M, 0, -1):
        Fm = (2.0 * T * Fm + e) / (2 * mm - 1)
        if mm - 1 <= mmax:
            all_F[mm - 1] = Fm

    F0_approx = all_F[0]
    if abs(F0_approx) > 1e-300:
        scale = F0_exact / F0_approx
    else:
        scale = 1.0
    return [all_F[m] * scale for m in range(mmax + 1)]


def boys_reference(m: int, T: float) -> float:
    """Reference Boys function F_m(T), high precision."""
    return boys_all(max(m, MMAX), T)[m]


def chebyshev_coefficients(func, a: float, b: float, n: int) -> np.ndarray:
    """Compute Chebyshev coefficients for func on [a, b] using n+1 terms.

    Uses the discrete cosine transform at Chebyshev nodes.
    """
    # Chebyshev nodes on [-1, 1]
    k = np.arange(n + 1)
    nodes = np.cos(math.pi * (k + 0.5) / (n + 1))
    # Map to [a, b]
    x = 0.5 * (a + b) + 0.5 * (b - a) * nodes
    # Evaluate function
    fvals = np.array([func(xi) for xi in x])
    # DCT-based coefficient computation
    coeffs = np.zeros(n + 1)
    for j in range(n + 1):
        coeffs[j] = (2.0 / (n + 1)) * np.sum(
            fvals * np.cos(math.pi * j * (k + 0.5) / (n + 1))
        )
    coeffs[0] *= 0.5  # c_0 gets factor 1/2
    return coeffs


def generate_table() -> dict:
    """Generate the full Chebyshev coefficient table.

    Returns a dict with:
        'coeffs': np.ndarray of shape (MMAX+1, NINTERVALS, NCHEB+1)
        'T_max': float
        'nintervals': int
        'ncheb': int
        'mmax': int
        'dT': float (interval width)
    """
    dT = T_MAX / NINTERVALS
    coeffs = np.zeros((MMAX + 1, NINTERVALS, NCHEB + 1))

    for m in range(MMAX + 1):
        for i in range(NINTERVALS):
            a = i * dT
            b = (i + 1) * dT
            coeffs[m, i, :] = chebyshev_coefficients(
                lambda T, _m=m: boys_reference(_m, T), a, b, NCHEB
            )

    return {
        'coeffs': coeffs,
        'T_max': T_MAX,
        'nintervals': NINTERVALS,
        'ncheb': NCHEB,
        'mmax': MMAX,
        'dT': dT,
    }


def validate_table(table: dict, *, ntests: int = 10000) -> float:
    """Validate the table against reference values. Returns max relative error."""
    coeffs = table['coeffs']
    dT = table['dT']
    ncheb = table['ncheb']
    nintervals = table['nintervals']
    mmax = table['mmax']
    T_max = table['T_max']

    max_rel_err = 0.0
    rng = np.random.default_rng(42)

    for _ in range(ntests):
        T = rng.uniform(0, T_max)
        m = rng.integers(0, mmax + 1)

        # Evaluate via table
        interval = min(int(T / dT), nintervals - 1)
        a = interval * dT
        b = (interval + 1) * dT
        # Map T to [-1, 1]
        u = 2.0 * (T - a) / (b - a) - 1.0
        # Clenshaw evaluation
        c = coeffs[m, interval, :]
        bk1 = 0.0
        bk2 = 0.0
        for j in range(ncheb, 0, -1):
            bk0 = c[j] + 2.0 * u * bk1 - bk2
            bk2 = bk1
            bk1 = bk0
        val = c[0] + u * bk1 - bk2

        ref = boys_reference(int(m), T)
        if abs(ref) > 1e-30:
            rel = abs(val - ref) / abs(ref)
            max_rel_err = max(max_rel_err, rel)

    return max_rel_err


def emit_header(table: dict) -> str:
    """Generate the C/CUDA header file content."""
    coeffs = table['coeffs']
    ncheb = table['ncheb']
    nintervals = table['nintervals']
    mmax = table['mmax']
    dT = table['dT']
    T_max = table['T_max']

    lines: list[str] = []
    lines.append("// Auto-generated by gen_boys_lut.py — DO NOT EDIT")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <cuda_runtime.h>")
    lines.append("")
    lines.append("namespace cueri_rys {")
    lines.append("")
    lines.append(f"constexpr int kBoysLutMmax = {mmax};")
    lines.append(f"constexpr int kBoysLutNintervals = {nintervals};")
    lines.append(f"constexpr int kBoysLutNcheb = {ncheb};  // order (ncheb+1 coefficients per interval)")
    lines.append(f"constexpr double kBoysLutTmax = {T_max:.1f};")
    lines.append(f"constexpr double kBoysLutDT = {dT:.17e};")
    lines.append(f"constexpr double kBoysLutInvDT = {1.0/dT:.17e};")
    lines.append("")

    # Emit the coefficient table as a flat __constant__ array.
    # Layout: coeffs[m][interval][k] where m=0..MMAX, interval=0..NINTERVALS-1, k=0..NCHEB
    total_size = (mmax + 1) * nintervals * (ncheb + 1)
    lines.append(f"// Total size: {total_size} doubles = {total_size * 8} bytes (~{total_size * 8 / 1024:.1f} KB)")
    lines.append(f"// Layout: [{mmax+1}][{nintervals}][{ncheb+1}] (m, interval, cheb_coeff)")
    lines.append(f"// Stored in __device__ memory (L2 cached via __ldg) — exceeds 64 KB __constant__ limit.")
    lines.append(f"static __device__ const double kBoysLutCoeffs[{total_size}] = {{")

    for m in range(mmax + 1):
        lines.append(f"  // m = {m}")
        for i in range(nintervals):
            vals = ", ".join(f"{coeffs[m, i, k]:.17e}" for k in range(ncheb + 1))
            comma = "," if (m < mmax or i < nintervals - 1) else ""
            lines.append(f"  {vals}{comma}")
    lines.append("};")
    lines.append("")

    # Emit the inline evaluation function
    lines.append("// Evaluate F_m(T) for m <= kBoysLutMmax and T in [0, kBoysLutTmax]")
    lines.append("// using Clenshaw recurrence on the pre-computed Chebyshev coefficients.")
    lines.append("template <int MMAX>")
    lines.append("__device__ inline void boys_fm_lut(double T, double* F) {")
    lines.append("  // For T > T_max, fall back to asymptotic + upward recurrence.")
    lines.append("  if (T > kBoysLutTmax) {")
    lines.append("    const double invT = 1.0 / T;")
    lines.append("    const double e = ::exp(-T);")
    lines.append("    F[0] = 0.5 * ::sqrt(kPi * invT) * ::erf(::sqrt(T));")
    lines.append("    #pragma unroll")
    lines.append("    for (int m = 1; m <= MMAX; ++m) {")
    lines.append("      F[m] = ((2 * m - 1) * F[m - 1] - e) * (0.5 * invT);")
    lines.append("    }")
    lines.append("    return;")
    lines.append("  }")
    lines.append("")
    lines.append("  // Determine interval index")
    lines.append("  int interval = static_cast<int>(T * kBoysLutInvDT);")
    lines.append(f"  if (interval >= kBoysLutNintervals) interval = kBoysLutNintervals - 1;")
    lines.append("")
    lines.append("  // Map T to u in [-1, 1] within the interval")
    lines.append("  const double a = static_cast<double>(interval) * kBoysLutDT;")
    lines.append("  const double u = 2.0 * (T - a) * kBoysLutInvDT - 1.0;")
    lines.append("")
    lines.append("  // Evaluate each F_m via Clenshaw recurrence")
    lines.append("  #pragma unroll")
    lines.append("  for (int m = 0; m <= MMAX; ++m) {")
    lines.append(f"    const int base = m * ({nintervals} * {ncheb + 1}) + interval * {ncheb + 1};")
    lines.append("    double bk1 = 0.0, bk2 = 0.0;")
    lines.append("    #pragma unroll")
    lines.append(f"    for (int j = {ncheb}; j >= 1; --j) {{")
    lines.append("      const double bk0 = __ldg(&kBoysLutCoeffs[base + j]) + 2.0 * u * bk1 - bk2;")
    lines.append("      bk2 = bk1;")
    lines.append("      bk1 = bk0;")
    lines.append("    }")
    lines.append("    F[m] = __ldg(&kBoysLutCoeffs[base]) + u * bk1 - bk2;")
    lines.append("  }")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace cueri_rys")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Generate Boys function Chebyshev LUT header")
    ap.add_argument(
        "--output",
        default=None,
        help="Output header path (default: src/cueri_cuda_rys_lut.cuh relative to ext/)",
    )
    ap.add_argument("--validate", action="store_true", help="Run validation after generation")
    args = ap.parse_args(argv)

    print(f"Generating Boys LUT: MMAX={MMAX}, NCHEB={NCHEB}, NINTERVALS={NINTERVALS}, T_MAX={T_MAX}")
    table = generate_table()

    if args.validate or True:  # always validate
        max_err = validate_table(table)
        print(f"Max relative error: {max_err:.2e}")
        if max_err > 1e-14:
            print("WARNING: relative error exceeds 1e-14!", file=sys.stderr)

    header = emit_header(table)

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = Path(__file__).resolve().parent.parent / "ext" / "src" / "cueri_cuda_rys_lut.cuh"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header, encoding="utf-8")
    total_bytes = (MMAX + 1) * NINTERVALS * (NCHEB + 1) * 8
    print(f"Written {out_path} ({total_bytes} bytes of coefficients, {total_bytes/1024:.1f} KB)")


if __name__ == "__main__":
    main()
