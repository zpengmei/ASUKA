#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


def ncart(l: int) -> int:
    return (l + 1) * (l + 2) // 2


def cart_components(l: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for x in range(l, -1, -1):
        rest = l - x
        for y in range(rest, -1, -1):
            z = rest - y
            out.append((x, y, z))
    return out


def shift_terms(i: int, j: int, k: int, l: int) -> list[tuple[int, int, int, int, int]]:
    # term: coeff, pow_ij, pow_kl, nidx, midx
    # I(i,j,k,l) = sum_{m=0..l} sum_{n=0..j} C(j,n) C(l,m) xij^(j-n) xkl^(l-m) G[n+i, m+k]
    from math import comb

    out: list[tuple[int, int, int, int, int]] = []
    for m in range(l + 1):
        for n in range(j + 1):
            coeff = comb(j, n) * comb(l, m)
            pow_ij = j - n
            pow_kl = l - m
            out.append((coeff, pow_ij, pow_kl, n + i, m + k))
    return out


@dataclass(frozen=True)
class QuartetClass:
    name: str
    la: int
    lb: int
    lc: int
    ld: int

    @property
    def nroots(self) -> int:
        return ((self.la + self.lb + self.lc + self.ld) // 2) + 1

    @property
    def ncomp(self) -> int:
        return ncart(self.la) * ncart(self.lb) * ncart(self.lc) * ncart(self.ld)

    @property
    def nmax(self) -> int:
        return self.la + self.lb

    @property
    def mmax(self) -> int:
        return self.lc + self.ld


WAVE1 = [
    QuartetClass("psds", 1, 0, 2, 0),
    QuartetClass("ppds", 1, 1, 2, 0),
    QuartetClass("dsds", 2, 0, 2, 0),
    QuartetClass("dsdp", 2, 0, 2, 1),
    QuartetClass("dsdd", 2, 0, 2, 2),
    QuartetClass("fpss", 3, 1, 0, 0),
    QuartetClass("fdss", 3, 2, 0, 0),
    QuartetClass("ffss", 3, 3, 0, 0),
    QuartetClass("fpps", 3, 1, 1, 0),
    QuartetClass("fdps", 3, 2, 1, 0),
    QuartetClass("ffps", 3, 3, 1, 0),
    QuartetClass("fpds", 3, 1, 2, 0),
    QuartetClass("fdds", 3, 2, 2, 0),
    QuartetClass("ffds", 3, 3, 2, 0),
    QuartetClass("ssfs", 0, 0, 3, 0),
    QuartetClass("psfs", 1, 0, 3, 0),
    QuartetClass("ppfs", 1, 1, 3, 0),
    QuartetClass("dsfs", 2, 0, 3, 0),
    QuartetClass("fsfs", 3, 0, 3, 0),
    QuartetClass("dpfs", 2, 1, 3, 0),
    QuartetClass("fpfs", 3, 1, 3, 0),
    QuartetClass("ddfs", 2, 2, 3, 0),
    QuartetClass("fdfs", 3, 2, 3, 0),
    QuartetClass("fffs", 3, 3, 3, 0),
    QuartetClass("ssgs", 0, 0, 4, 0),
    QuartetClass("psgs", 1, 0, 4, 0),
    QuartetClass("ppgs", 1, 1, 4, 0),
    QuartetClass("dsgs", 2, 0, 4, 0),
    QuartetClass("fsgs", 3, 0, 4, 0),
    QuartetClass("dpgs", 2, 1, 4, 0),
    QuartetClass("fpgs", 3, 1, 4, 0),
    QuartetClass("ddgs", 2, 2, 4, 0),
    QuartetClass("fdgs", 3, 2, 4, 0),
    QuartetClass("ffgs", 3, 3, 4, 0),
]

WAVE2 = [
    QuartetClass("ssdp", 0, 0, 2, 1),
    QuartetClass("psdp", 1, 0, 2, 1),
    QuartetClass("psdd", 1, 0, 2, 2),
    QuartetClass("ppdp", 1, 1, 2, 1),
    QuartetClass("ppdd", 1, 1, 2, 2),
    QuartetClass("ddss", 2, 2, 0, 0),
    QuartetClass("dpdp", 2, 1, 2, 1),
    QuartetClass("dpdd", 2, 1, 2, 2),
    QuartetClass("dddd", 2, 2, 2, 2),
]

WAVE_MAP: dict[str, list[QuartetClass]] = {
    "wave1": WAVE1,
    "wave2": WAVE2,
    "all": WAVE1 + WAVE2,
}


# Tuned default launch threads per specialized quartet.
# These are used when the caller does not provide an explicit thread count.
LAUNCH_THREADS_OVERRIDES: dict[str, int] = {
    "ssdp": 64,
    "psdp": 128,
    "psdd": 160,
    "ppdp": 192,
    "ppdd": 224,
    "ddss": 96,
    "dpdp": 224,
    "dpdd": 256,
    "dddd": 256,
    "psds": 64,
    "ppds": 64,
    "dsds": 64,
    "dsdp": 128,
    "dsdd": 224,
    "fpss": 64,
    "fdss": 64,
    "ffss": 128,
    "fpps": 96,
    "fdps": 128,
    "ffps": 128,
    "fpds": 128,
    "fdds": 128,
    "ffds": 128,
    "ssfs": 64,
    "psfs": 96,
    "ppfs": 128,
    "dsfs": 128,
    "fsfs": 128,
    "dpfs": 128,
    "fpfs": 160,
    "ddfs": 192,
    "fdfs": 224,
    "fffs": 128,
    "ssgs": 64,
    "psgs": 96,
    "ppgs": 128,
    "dsgs": 96,
    "fsgs": 128,
    "dpgs": 160,
    "fpgs": 224,
    "ddgs": 224,
    "fdgs": 256,
    "ffgs": 128,
}


# Maximum ncomp for which a warp-parallel-pair kernel is generated.
# For ncomp > this limit, the warp launcher falls back to the block kernel.
# Constraint: ncomp * NROOTS register accumulators per lane must fit in
# the SM register file.  162 comps × 3 roots = 486 doubles is still fine
# on SM89 (RTX 4090).  This enables warp kernels for ppdp (ncomp=162)
# and dppp (ncomp=135, via transpose).
WARP_PAIR_NCOMP_MAX = 162


FUSED_FOCK_CLASSES: set[str] = {
    "ddss",
    "ssdp",
    "psds",
    "psdp",
    "psdd",
    "ppds",
    "dsds",
    "dsdp",
    # new s/p/d classes (pppp handled separately in step2)
    "ppdp",
    "ppdd",
    "dsdd",
    "dpdp",
    "dpdd",
    "dddd",
}

FUSED_FOCK_THREADS_OVERRIDES: dict[str, int] = {
    "ddss": 64,
    "ssdp": 64,
    "psds": 64,
    "psdp": 64,
    "psdd": 64,
    "ppds": 64,
    "dsds": 64,
    "dsdp": 64,
    # new s/p/d classes (pppp handled separately in step2)
    "ppdp": 128,
    "ppdd": 160,
    "dsdd": 128,
    "dpdp": 192,
    "dpdd": 224,
    "dddd": 256,
}

FUSED_JK_CLASSES: set[str] = set(FUSED_FOCK_CLASSES)

FUSED_JK_THREADS_OVERRIDES: dict[str, int] = dict(FUSED_FOCK_THREADS_OVERRIDES)


def axis_expr(terms: list[tuple[int, int, int, int, int]], *, stride: int, gsym: str, dij: str, dkl: str) -> str:
    def pow_sym(base: str, p: int) -> str:
        if p == 0:
            return "1.0"
        if p == 1:
            return base
        if p == 2:
            return f"{base}2"
        return f"pow({base}, {p})"

    parts: list[str] = []
    for coeff, pij, pkl, nidx, midx in terms:
        g = f"{gsym}[{nidx * stride + midx}]"
        f_ij = pow_sym(dij, pij)
        f_kl = pow_sym(dkl, pkl)
        factors = [g]
        if f_ij != "1.0":
            factors.append(f_ij)
        if f_kl != "1.0":
            factors.append(f_kl)
        term = " * ".join(factors)
        if coeff != 1:
            term = f"{float(coeff):.1f} * ({term})"
        parts.append(term)
    if not parts:
        return "0.0"
    return " + ".join(parts)


@dataclass(frozen=True)
class AxisDedup:
    uniq: list[str]
    map_idx: list[int]


def dedup_axis_exprs(exprs: list[str]) -> AxisDedup:
    uniq: list[str] = []
    map_idx: list[int] = []
    seen: dict[str, int] = {}
    for expr in exprs:
        idx = seen.get(expr)
        if idx is None:
            idx = len(uniq)
            uniq.append(expr)
            seen[expr] = idx
        map_idx.append(idx)
    return AxisDedup(uniq=uniq, map_idx=map_idx)


def emit_eval_fn(q: QuartetClass, axis: str, exprs: list[str]) -> str:
    fn = [
        f"__device__ __forceinline__ double eval_{q.name}_{axis}(",
        "    int e,",
        "    const double* G,",
        f"    double {axis}ij,",
        f"    double {axis}ij2,",
        f"    double {axis}kl,",
        f"    double {axis}kl2) {{",
        "  switch (e) {",
    ]
    for i, ex in enumerate(exprs):
        fn.append(f"    case {i}: return {ex};")
    fn.extend(
        [
            "    default: return 0.0;",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(fn)


def emit_multiblock_helpers() -> str:
    return "\n".join(
        [
            "template <int NCOMP, int BLOCKS_PER_TASK>",
            "__global__ void KernelMultiblockReduceFixed(const double* partial_sums, double* eri_out) {",
            '  static_assert(NCOMP > 0, "NCOMP must be > 0");',
            '  static_assert(BLOCKS_PER_TASK > 0, "BLOCKS_PER_TASK must be > 0");',
            "  const int t = static_cast<int>(blockIdx.x);",
            "  for (int e = static_cast<int>(threadIdx.x); e < NCOMP; e += static_cast<int>(blockDim.x)) {",
            "    double s = 0.0;",
            "    const int64_t base = static_cast<int64_t>(t) * static_cast<int64_t>(BLOCKS_PER_TASK) * static_cast<int64_t>(NCOMP)",
            "                         + static_cast<int64_t>(e);",
            "    #pragma unroll",
            "    for (int b = 0; b < BLOCKS_PER_TASK; ++b) {",
            "      s += partial_sums[base + static_cast<int64_t>(b) * static_cast<int64_t>(NCOMP)];",
            "    }",
            "    eri_out[static_cast<int64_t>(t) * static_cast<int64_t>(NCOMP) + static_cast<int64_t>(e)] = s;",
            "  }",
            "}",
            "",
            "template <int NCOMP>",
            "__global__ void KernelMultiblockReduceDynamic(const double* partial_sums, int blocks_per_task, double* eri_out) {",
            "  const int t = static_cast<int>(blockIdx.x);",
            "  for (int e = static_cast<int>(threadIdx.x); e < NCOMP; e += static_cast<int>(blockDim.x)) {",
            "    double s = 0.0;",
            "    const int64_t base = static_cast<int64_t>(t) * static_cast<int64_t>(blocks_per_task) * static_cast<int64_t>(NCOMP)",
            "                         + static_cast<int64_t>(e);",
            "    for (int b = 0; b < blocks_per_task; ++b) {",
            "      s += partial_sums[base + static_cast<int64_t>(b) * static_cast<int64_t>(NCOMP)];",
            "    }",
            "    eri_out[static_cast<int64_t>(t) * static_cast<int64_t>(NCOMP) + static_cast<int64_t>(e)] = s;",
            "  }",
            "}",
            "",
        ]
    )


def emit_kernel(q: QuartetClass, stride: int) -> str:
    nA = ncart(q.la)
    nB = ncart(q.lb)
    nC = ncart(q.lc)
    nD = ncart(q.ld)
    nAB = nA * nB
    nCD = nC * nD
    ncomp = nAB * nCD

    A = cart_components(q.la)
    B = cart_components(q.lb)
    C = cart_components(q.lc)
    D = cart_components(q.ld)

    ex = []
    ey = []
    ez = []
    for ab in range(nAB):
        ia = ab // nB
        ib = ab - ia * nB
        ax, ay, az = A[ia]
        bx, by, bz = B[ib]
        for cd in range(nCD):
            ic = cd // nD
            id_ = cd - ic * nD
            cx, cy, cz = C[ic]
            dx, dy, dz = D[id_]
            tx = shift_terms(ax, bx, cx, dx)
            ty = shift_terms(ay, by, cy, dy)
            tz = shift_terms(az, bz, cz, dz)
            ex.append(axis_expr(tx, stride=stride, gsym="G", dij="xij", dkl="xkl"))
            ey.append(axis_expr(ty, stride=stride, gsym="G", dij="yij", dkl="ykl"))
            ez.append(axis_expr(tz, stride=stride, gsym="G", dij="zij", dkl="zkl"))
    assert len(ex) == ncomp

    out: list[str] = []

    # Eval functions for fused classes are emitted in the preamble by generate_text().
    # Flat kernel is the primary ERI kernel — all launchers dispatch to it.
    # Skip fixed/multiblock/warp kernels to reduce TU size and avoid
    # cross-kernel shared dependencies (KernelMultiblockReduceFixed)
    # that prevent the splitter from splitting wave files into multiple parts.
    out.append(emit_flat_kernel(q, stride))
    if q.ncomp <= WARP_PAIR_NCOMP_MAX:
        out.append(emit_warp_kernel(q, stride))
    if q.name in FUSED_FOCK_CLASSES:
        out.append(emit_fused_fock_kernel(q, stride))
    if q.name in FUSED_JK_CLASSES:
        out.append(emit_fused_jk_kernel(q, stride))
    return "\n".join(out)

    # --- Legacy block/multiblock/warp kernels (dead code, kept for reference) ---
    out.append(emit_eval_fn(q, "x", ex))
    out.append(emit_eval_fn(q, "y", ey))
    out.append(emit_eval_fn(q, "z", ez))

    out.extend(
        [
            f"template <int NROOTS>",
            f"__global__ void KernelERI_{q.name}_fixed(",
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    double* eri_out) {",
            "  const int t = static_cast<int>(blockIdx.x);",
            "  if (t >= ntasks) return;",
            "",
            "  const int spAB = static_cast<int>(task_spAB[t]);",
            "  const int spCD = static_cast<int>(task_spCD[t]);",
            "  const int A = static_cast<int>(sp_A[spAB]);",
            "  const int B = static_cast<int>(sp_B[spAB]);",
            "  const int C = static_cast<int>(sp_A[spCD]);",
            "  const int D = static_cast<int>(sp_B[spCD]);",
            "",
            "  const double Ax = shell_cx[A];",
            "  const double Ay = shell_cy[A];",
            "  const double Az = shell_cz[A];",
            "  const double Bx = shell_cx[B];",
            "  const double By = shell_cy[B];",
            "  const double Bz = shell_cz[B];",
            "  const double Cx = shell_cx[C];",
            "  const double Cy = shell_cy[C];",
            "  const double Cz = shell_cz[C];",
            "  const double Dx = shell_cx[D];",
            "  const double Dy = shell_cy[D];",
            "  const double Dz = shell_cz[D];",
            "",
            "  const double xij = Ax - Bx;",
            "  const double yij = Ay - By;",
            "  const double zij = Az - Bz;",
            "  const double xkl = Cx - Dx;",
            "  const double ykl = Cy - Dy;",
            "  const double zkl = Cz - Dz;",
            "  const double xij2 = xij * xij;",
            "  const double yij2 = yij * yij;",
            "  const double zij2 = zij * zij;",
            "  const double xkl2 = xkl * xkl;",
            "  const double ykl2 = ykl * ykl;",
            "  const double zkl2 = zkl * zkl;",
            "",
            "  const int baseAB = static_cast<int>(sp_pair_start[spAB]);",
            "  const int baseCD = static_cast<int>(sp_pair_start[spCD]);",
            "  const int nPairAB = static_cast<int>(sp_npair[spAB]);",
            "  const int nPairCD = static_cast<int>(sp_npair[spCD]);",
            "",
            f"  constexpr int kStride = {stride};",
            f"  constexpr int kNComp = {ncomp};",
            f"  constexpr int kNMax = {q.nmax};",
            f"  constexpr int kMMax = {q.mmax};",
            "",
            "  __shared__ double Gx[kStride * kStride];",
            "  __shared__ double Gy[kStride * kStride];",
            "  __shared__ double Gz[kStride * kStride];",
            "  __shared__ double sh_scale;",
            "  __shared__ double sh_roots[NROOTS];",
            "  __shared__ double sh_weights[NROOTS];",
            "  __shared__ double sh_p;",
            "  __shared__ double sh_q;",
            "  __shared__ double sh_Px;",
            "  __shared__ double sh_Py;",
            "  __shared__ double sh_Pz;",
            "  __shared__ double sh_Qx;",
            "  __shared__ double sh_Qy;",
            "  __shared__ double sh_Qz;",
            "  __shared__ double sh_denom;",
            "  __shared__ double sh_base;",
            "",
            "  for (int ebase = 0; ebase < kNComp; ebase += static_cast<int>(blockDim.x)) {",
            "    const int e = ebase + static_cast<int>(threadIdx.x);",
            "    const bool active = (e < kNComp);",
            "    double val = 0.0;",
            "    for (int ip = 0; ip < nPairAB; ++ip) {",
            "      const int ki = baseAB + ip;",
            "      for (int jp = 0; jp < nPairCD; ++jp) {",
            "        const int kj = baseCD + jp;",
            "        if (threadIdx.x == 0) {",
            "          sh_p = pair_eta[ki];",
            "          sh_q = pair_eta[kj];",
            "          sh_Px = pair_Px[ki];",
            "          sh_Py = pair_Py[ki];",
            "          sh_Pz = pair_Pz[ki];",
            "          sh_Qx = pair_Px[kj];",
            "          sh_Qy = pair_Py[kj];",
            "          sh_Qz = pair_Pz[kj];",
            "          const double dx = sh_Px - sh_Qx;",
            "          const double dy = sh_Py - sh_Qy;",
            "          const double dz = sh_Pz - sh_Qz;",
            "          const double PQ2 = dx * dx + dy * dy + dz * dz;",
            "          sh_denom = sh_p + sh_q;",
            "          const double omega = sh_p * sh_q / sh_denom;",
            "          const double T = omega * PQ2;",
            "          sh_base = kTwoPiToFiveHalves / (sh_p * sh_q * ::sqrt(sh_denom)) * pair_cK[ki] * pair_cK[kj];",
            "          cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots, sh_weights);",
            "        }",
            "        __syncthreads();",
            "        for (int u = 0; u < NROOTS; ++u) {",
            "          if (threadIdx.x == 0) {",
            "            const double x = sh_roots[u];",
            "            const double w = sh_weights[u];",
            "            const double inv_denom = 1.0 / sh_denom;",
            "            const double B0 = x * 0.5 * inv_denom;",
            "            const double B1 = (1.0 - x) * 0.5 / sh_p + B0;",
            "            const double B1p = (1.0 - x) * 0.5 / sh_q + B0;",
            "",
            "            const double Cx_ = (sh_Px - Ax) + (sh_q * inv_denom) * x * (sh_Qx - sh_Px);",
            "            const double Cy_ = (sh_Py - Ay) + (sh_q * inv_denom) * x * (sh_Qy - sh_Py);",
            "            const double Cz_ = (sh_Pz - Az) + (sh_q * inv_denom) * x * (sh_Qz - sh_Pz);",
            "            const double Cpx_ = (sh_Qx - Cx) + (sh_p * inv_denom) * x * (sh_Px - sh_Qx);",
            "            const double Cpy_ = (sh_Qy - Cy) + (sh_p * inv_denom) * x * (sh_Py - sh_Qy);",
            "            const double Cpz_ = (sh_Qz - Cz) + (sh_p * inv_denom) * x * (sh_Pz - sh_Qz);",
            "",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gx, Cx_, Cpx_, B0, B1, B1p);",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gy, Cy_, Cpy_, B0, B1, B1p);",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gz, Cz_, Cpz_, B0, B1, B1p);",
            "            sh_scale = sh_base * w;",
            "          }",
            "          __syncthreads();",
            "          if (active) {",
            "            const double Ix = eval_%s_x(e, Gx, xij, xij2, xkl, xkl2);" % q.name,
            "            const double Iy = eval_%s_y(e, Gy, yij, yij2, ykl, ykl2);" % q.name,
            "            const double Iz = eval_%s_z(e, Gz, zij, zij2, zkl, zkl2);" % q.name,
            "            val += sh_scale * (Ix * Iy * Iz);",
            "          }",
            "          __syncthreads();",
            "        }",
            "      }",
            "    }",
            "    if (active) {",
            "      eri_out[static_cast<int64_t>(t) * static_cast<int64_t>(kNComp) + static_cast<int64_t>(e)] = val;",
            "    }",
            "  }",
            "}",
            "",
        ]
    )
    out.extend(
        [
            "template <int NROOTS>",
            f"__global__ void KernelERI_{q.name}_multiblock_partial(",
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    int blocks_per_task,",
            "    double* partial_sums) {",
            "  const int t = static_cast<int>(blockIdx.x);",
            "  const int b = static_cast<int>(blockIdx.y);",
            "  if (t >= ntasks || b >= blocks_per_task) return;",
            "",
            "  const int spAB = static_cast<int>(task_spAB[t]);",
            "  const int spCD = static_cast<int>(task_spCD[t]);",
            "  const int A = static_cast<int>(sp_A[spAB]);",
            "  const int B = static_cast<int>(sp_B[spAB]);",
            "  const int C = static_cast<int>(sp_A[spCD]);",
            "  const int D = static_cast<int>(sp_B[spCD]);",
            "",
            "  const double Ax = shell_cx[A];",
            "  const double Ay = shell_cy[A];",
            "  const double Az = shell_cz[A];",
            "  const double Bx = shell_cx[B];",
            "  const double By = shell_cy[B];",
            "  const double Bz = shell_cz[B];",
            "  const double Cx = shell_cx[C];",
            "  const double Cy = shell_cy[C];",
            "  const double Cz = shell_cz[C];",
            "  const double Dx = shell_cx[D];",
            "  const double Dy = shell_cy[D];",
            "  const double Dz = shell_cz[D];",
            "",
            "  const double xij = Ax - Bx;",
            "  const double yij = Ay - By;",
            "  const double zij = Az - Bz;",
            "  const double xkl = Cx - Dx;",
            "  const double ykl = Cy - Dy;",
            "  const double zkl = Cz - Dz;",
            "  const double xij2 = xij * xij;",
            "  const double yij2 = yij * yij;",
            "  const double zij2 = zij * zij;",
            "  const double xkl2 = xkl * xkl;",
            "  const double ykl2 = ykl * ykl;",
            "  const double zkl2 = zkl * zkl;",
            "",
            "  const int baseAB = static_cast<int>(sp_pair_start[spAB]);",
            "  const int baseCD = static_cast<int>(sp_pair_start[spCD]);",
            "  const int nPairAB = static_cast<int>(sp_npair[spAB]);",
            "  const int nPairCD = static_cast<int>(sp_npair[spCD]);",
            "  const int64_t nPairsTot = static_cast<int64_t>(nPairAB) * static_cast<int64_t>(nPairCD);",
            "",
            f"  constexpr int kStride = {stride};",
            f"  constexpr int kNComp = {ncomp};",
            f"  constexpr int kNMax = {q.nmax};",
            f"  constexpr int kMMax = {q.mmax};",
            "",
            "  __shared__ double Gx[kStride * kStride];",
            "  __shared__ double Gy[kStride * kStride];",
            "  __shared__ double Gz[kStride * kStride];",
            "  __shared__ double sh_scale;",
            "  __shared__ double sh_roots[NROOTS];",
            "  __shared__ double sh_weights[NROOTS];",
            "  __shared__ double sh_p;",
            "  __shared__ double sh_q;",
            "  __shared__ double sh_Px;",
            "  __shared__ double sh_Py;",
            "  __shared__ double sh_Pz;",
            "  __shared__ double sh_Qx;",
            "  __shared__ double sh_Qy;",
            "  __shared__ double sh_Qz;",
            "  __shared__ double sh_denom;",
            "  __shared__ double sh_base;",
            "",
            "  for (int ebase = 0; ebase < kNComp; ebase += static_cast<int>(blockDim.x)) {",
            "    const int e = ebase + static_cast<int>(threadIdx.x);",
            "    const bool active = (e < kNComp);",
            "    double val = 0.0;",
            "    for (int64_t upair = static_cast<int64_t>(b); upair < nPairsTot; upair += static_cast<int64_t>(blocks_per_task)) {",
            "      if (threadIdx.x == 0) {",
            "        const int ip = static_cast<int>(upair / static_cast<int64_t>(nPairCD));",
            "        const int jp = static_cast<int>(upair - static_cast<int64_t>(ip) * static_cast<int64_t>(nPairCD));",
            "        const int ki = baseAB + ip;",
            "        const int kj = baseCD + jp;",
            "        sh_p = pair_eta[ki];",
            "        sh_q = pair_eta[kj];",
            "        sh_Px = pair_Px[ki];",
            "        sh_Py = pair_Py[ki];",
            "        sh_Pz = pair_Pz[ki];",
            "        sh_Qx = pair_Px[kj];",
            "        sh_Qy = pair_Py[kj];",
            "        sh_Qz = pair_Pz[kj];",
            "        const double dx = sh_Px - sh_Qx;",
            "        const double dy = sh_Py - sh_Qy;",
            "        const double dz = sh_Pz - sh_Qz;",
            "        const double PQ2 = dx * dx + dy * dy + dz * dz;",
            "        sh_denom = sh_p + sh_q;",
            "        const double omega = sh_p * sh_q / sh_denom;",
            "        const double T = omega * PQ2;",
            "        sh_base = kTwoPiToFiveHalves / (sh_p * sh_q * ::sqrt(sh_denom)) * pair_cK[ki] * pair_cK[kj];",
            "        cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots, sh_weights);",
            "      }",
            "      __syncthreads();",
            "      for (int u = 0; u < NROOTS; ++u) {",
            "        if (threadIdx.x == 0) {",
            "          const double x = sh_roots[u];",
            "          const double w = sh_weights[u];",
            "          const double inv_denom = 1.0 / sh_denom;",
            "          const double B0 = x * 0.5 * inv_denom;",
            "          const double B1 = (1.0 - x) * 0.5 / sh_p + B0;",
            "          const double B1p = (1.0 - x) * 0.5 / sh_q + B0;",
            "",
            "          const double Cx_ = (sh_Px - Ax) + (sh_q * inv_denom) * x * (sh_Qx - sh_Px);",
            "          const double Cy_ = (sh_Py - Ay) + (sh_q * inv_denom) * x * (sh_Qy - sh_Py);",
            "          const double Cz_ = (sh_Pz - Az) + (sh_q * inv_denom) * x * (sh_Qz - sh_Pz);",
            "          const double Cpx_ = (sh_Qx - Cx) + (sh_p * inv_denom) * x * (sh_Px - sh_Qx);",
            "          const double Cpy_ = (sh_Qy - Cy) + (sh_p * inv_denom) * x * (sh_Py - sh_Qy);",
            "          const double Cpz_ = (sh_Qz - Cz) + (sh_p * inv_denom) * x * (sh_Pz - sh_Qz);",
            "",
            "          compute_G_stride_fixed<kStride, kNMax, kMMax>(Gx, Cx_, Cpx_, B0, B1, B1p);",
            "          compute_G_stride_fixed<kStride, kNMax, kMMax>(Gy, Cy_, Cpy_, B0, B1, B1p);",
            "          compute_G_stride_fixed<kStride, kNMax, kMMax>(Gz, Cz_, Cpz_, B0, B1, B1p);",
            "          sh_scale = sh_base * w;",
            "        }",
            "        __syncthreads();",
            "        if (active) {",
            "          const double Ix = eval_%s_x(e, Gx, xij, xij2, xkl, xkl2);" % q.name,
            "          const double Iy = eval_%s_y(e, Gy, yij, yij2, ykl, ykl2);" % q.name,
            "          const double Iz = eval_%s_z(e, Gz, zij, zij2, zkl, zkl2);" % q.name,
            "          val += sh_scale * (Ix * Iy * Iz);",
            "        }",
            "        __syncthreads();",
            "      }",
            "    }",
            "    if (active) {",
            "      const int64_t out = (static_cast<int64_t>(t) * static_cast<int64_t>(blocks_per_task) + static_cast<int64_t>(b))",
            "                        * static_cast<int64_t>(kNComp) + static_cast<int64_t>(e);",
            "      partial_sums[out] = val;",
            "    }",
            "  }",
            "}",
            "",
        ]
    )
    out.append(emit_flat_kernel(q, stride))
    if q.ncomp <= WARP_PAIR_NCOMP_MAX:
        out.append(emit_warp_kernel(q, stride))
    if q.name in FUSED_FOCK_CLASSES:
        out.append(emit_fused_fock_kernel(q, stride))
    if q.name in FUSED_JK_CLASSES:
        out.append(emit_fused_jk_kernel(q, stride))
    return "\n".join(out)


def emit_warp_kernel(q: QuartetClass, stride: int) -> str:
    """Generate a warp-parallel-pair ERI kernel.

    Each warp processes one shell-quartet task.  Primitive-pair combinations
    (ip, jp) are distributed across the 32 lanes.  Each lane independently
    computes Rys roots, builds G arrays in *local* (register) memory, and
    accumulates all ``ncomp`` ERI components.  A warp shuffle reduction
    collects the final result.

    This avoids the thread-0 serial bottleneck of the block kernel, where
    only lane 0 computes Rys roots and G arrays while all other threads
    wait at ``__syncthreads()``.
    """

    nA = ncart(q.la)
    nB = ncart(q.lb)
    nC = ncart(q.lc)
    nD = ncart(q.ld)
    nAB = nA * nB
    nCD = nC * nD
    ncomp = nAB * nCD
    assert ncomp <= WARP_PAIR_NCOMP_MAX

    # Build the component evaluation expressions (same as in emit_kernel).
    A = cart_components(q.la)
    B = cart_components(q.lb)
    C = cart_components(q.lc)
    D = cart_components(q.ld)

    # For the warp kernel, instead of using a switch-based eval function, we
    # emit the full unrolled evaluation of all ncomp components inline.
    # Each component is:  acc[e] += scale * Ix * Iy * Iz
    # where Ix = shift expression reading from Gx[...], etc.

    comp_lines: list[str] = []
    for ab in range(nAB):
        ia = ab // nB
        ib = ab - ia * nB
        ax, ay, az = A[ia]
        bx, by, bz = B[ib]
        for cd in range(nCD):
            ic = cd // nD
            id_ = cd - ic * nD
            cx, cy, cz = C[ic]
            dx, dy, dz = D[id_]

            e = ab * nCD + cd
            tx = shift_terms(ax, bx, cx, dx)
            ty = shift_terms(ay, by, cy, dy)
            tz = shift_terms(az, bz, cz, dz)
            ex = axis_expr(tx, stride=stride, gsym="Gx", dij="xij", dkl="xkl")
            ey = axis_expr(ty, stride=stride, gsym="Gy", dij="yij", dkl="ykl")
            ez = axis_expr(tz, stride=stride, gsym="Gz", dij="zij", dkl="zkl")
            comp_lines.append(f"            acc[{e}] += scale * ({ex}) * ({ey}) * ({ez});")

    gsize = stride * stride

    fn: list[str] = []
    fn.extend([
        f"template <int NROOTS>",
        f"__global__ void KernelERI_{q.name}_warp_true(",
        "    const int32_t* __restrict__ task_spAB,",
        "    const int32_t* __restrict__ task_spCD,",
        "    int ntasks,",
        "    const int32_t* __restrict__ sp_A,",
        "    const int32_t* __restrict__ sp_B,",
        "    const int32_t* __restrict__ sp_pair_start,",
        "    const int32_t* __restrict__ sp_npair,",
        "    const double* __restrict__ shell_cx,",
        "    const double* __restrict__ shell_cy,",
        "    const double* __restrict__ shell_cz,",
        "    const double* __restrict__ pair_eta,",
        "    const double* __restrict__ pair_Px,",
        "    const double* __restrict__ pair_Py,",
        "    const double* __restrict__ pair_Pz,",
        "    const double* __restrict__ pair_cK,",
        "    double* __restrict__ eri_out) {",
        "  const int lane = static_cast<int>(threadIdx.x) & 31;",
        "  const int warp_id = static_cast<int>(threadIdx.x) >> 5;",
        "  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;",
        "  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;",
        "  if (t >= ntasks) return;",
        "",
        "  const int spAB = static_cast<int>(task_spAB[t]);",
        "  const int spCD = static_cast<int>(task_spCD[t]);",
        "  const int iA = static_cast<int>(sp_A[spAB]);",
        "  const int iB = static_cast<int>(sp_B[spAB]);",
        "  const int iC = static_cast<int>(sp_A[spCD]);",
        "  const int iD = static_cast<int>(sp_B[spCD]);",
        "",
        "  const double Ax = shell_cx[iA];",
        "  const double Ay = shell_cy[iA];",
        "  const double Az = shell_cz[iA];",
        "  const double Bx = shell_cx[iB];",
        "  const double By = shell_cy[iB];",
        "  const double Bz = shell_cz[iB];",
        "  const double Cx = shell_cx[iC];",
        "  const double Cy = shell_cy[iC];",
        "  const double Cz = shell_cz[iC];",
        "  const double Dx = shell_cx[iD];",
        "  const double Dy = shell_cy[iD];",
        "  const double Dz = shell_cz[iD];",
        "",
        "  const double xij = Ax - Bx;",
        "  const double yij = Ay - By;",
        "  const double zij = Az - Bz;",
        "  const double xkl = Cx - Dx;",
        "  const double ykl = Cy - Dy;",
        "  const double zkl = Cz - Dz;",
    ])
    fn.extend([
        "  const double xij2 = xij * xij;",
        "  const double yij2 = yij * yij;",
        "  const double zij2 = zij * zij;",
        "  const double xkl2 = xkl * xkl;",
        "  const double ykl2 = ykl * ykl;",
        "  const double zkl2 = zkl * zkl;",
    ])
    fn.extend([
        "",
        "  const int baseAB = static_cast<int>(sp_pair_start[spAB]);",
        "  const int baseCD = static_cast<int>(sp_pair_start[spCD]);",
        "  const int nPairAB = static_cast<int>(sp_npair[spAB]);",
        "  const int nPairCD = static_cast<int>(sp_npair[spCD]);",
        "  const int totalPairs = nPairAB * nPairCD;",
        "",
        f"  constexpr int kStride = {stride};",
        f"  constexpr int kNComp = {ncomp};",
        f"  constexpr int kNMax = {q.nmax};",
        f"  constexpr int kMMax = {q.mmax};",
        f"  constexpr int kGSize = {gsize};",
        "",
        "  // Per-lane accumulators in registers.",
        "  double acc[kNComp];",
        "  #pragma unroll",
        "  for (int i = 0; i < kNComp; ++i) acc[i] = 0.0;",
        "",
        "  // Lane-parallel primitive pair loop.",
        "  for (int pair = lane; pair < totalPairs; pair += 32) {",
        "    const int ip = pair / nPairCD;",
        "    const int jp = pair - ip * nPairCD;",
        "    const int ki = baseAB + ip;",
        "    const int kj = baseCD + jp;",
        "",
        "    const double p = pair_eta[ki];",
        "    const double q = pair_eta[kj];",
        "    const double Px = pair_Px[ki];",
        "    const double Py = pair_Py[ki];",
        "    const double Pz = pair_Pz[ki];",
        "    const double Qx = pair_Px[kj];",
        "    const double Qy = pair_Py[kj];",
        "    const double Qz = pair_Pz[kj];",
        "",
        "    const double dxPQ = Px - Qx;",
        "    const double dyPQ = Py - Qy;",
        "    const double dzPQ = Pz - Qz;",
        "    const double PQ2 = dxPQ * dxPQ + dyPQ * dyPQ + dzPQ * dzPQ;",
        "",
        "    const double denom = p + q;",
        "    const double omega = p * q / denom;",
        "    const double T = omega * PQ2;",
        "    const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * pair_cK[ki] * pair_cK[kj];",
        "",
        "    // Rys roots/weights in registers — each lane computes independently.",
        f"    double roots[NROOTS];",
        f"    double weights[NROOTS];",
        "    cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);",
        "",
        "    for (int u = 0; u < NROOTS; ++u) {",
        "      const double x = roots[u];",
        "      const double w = weights[u];",
        "      const double inv_denom = 1.0 / denom;",
        "      const double B0 = x * 0.5 * inv_denom;",
        "      const double B1 = (1.0 - x) * 0.5 / p + B0;",
        "      const double B1p = (1.0 - x) * 0.5 / q + B0;",
        "",
        "      const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);",
        "      const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);",
        "      const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);",
        "      const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);",
        "      const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);",
        "      const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);",
        "",
        "      // G arrays in local memory (registers/L1).",
        "      double Gx[kGSize];",
        "      double Gy[kGSize];",
        "      double Gz[kGSize];",
        "      compute_G_stride_fixed<kStride, kNMax, kMMax>(Gx, Cx_, Cpx_, B0, B1, B1p);",
        "      compute_G_stride_fixed<kStride, kNMax, kMMax>(Gy, Cy_, Cpy_, B0, B1, B1p);",
        "      compute_G_stride_fixed<kStride, kNMax, kMMax>(Gz, Cz_, Cpz_, B0, B1, B1p);",
        "",
        "      const double scale = base * w;",
        "",
        "      // Accumulate all components.",
    ])
    for line in comp_lines:
        fn.append(line)
    fn.extend([
        "    }  // for u (Rys roots)",
        "  }  // for pair",
        "",
        "  // Warp reduction: sum across lanes.",
        "  #pragma unroll",
        "  for (int i = 0; i < kNComp; ++i) {",
        "    #pragma unroll",
        "    for (int offset = 16; offset > 0; offset >>= 1) {",
        "      acc[i] += __shfl_down_sync(0xFFFFFFFF, acc[i], offset);",
        "    }",
        "  }",
        "",
        "  // Lane 0 writes output.",
        "  if (lane == 0) {",
        "    double* out = eri_out + static_cast<int64_t>(t) * static_cast<int64_t>(kNComp);",
        "    #pragma unroll",
        "    for (int i = 0; i < kNComp; ++i) out[i] = acc[i];",
        "  }",
        "}",
        "",
    ])
    return "\n".join(fn)


def emit_flat_kernel(q: QuartetClass, stride: int) -> str:
    """Generate a flat ERI kernel: one thread per task, no shared memory.

    Each thread independently processes one shell-quartet task:
    loads pair data, computes Rys roots/weights, builds G arrays
    in local (register/L1) memory, evaluates all ncomp components,
    and writes the output tile.  No __syncthreads() needed.

    This achieves 100% thread utilization vs the block kernel which
    has only ncomp/blockDim.x utilization (e.g. 18/256 = 7% for psds).

    All component expressions are inlined so the kernel is self-contained
    and can be placed in any TU by the splitter without dependency issues.

    Template parameters:
      NROOTS: Boys function root count (compile-time).
      kTileF32: When true, write output tile in FP32 (halves bandwidth).
      kMixedPrec: When true, evaluate Ux/Uy/Uz components in FP32
                  (32x faster on RTX 4090) while keeping accumulation in FP64.
    """
    nA = ncart(q.la)
    nB = ncart(q.lb)
    nC = ncart(q.lc)
    nD = ncart(q.ld)
    nAB = nA * nB
    nCD = nC * nD
    ncomp = nAB * nCD

    # Use compact G-array layout: stride = mmax+1, size = (nmax+1)*(mmax+1).
    # This dramatically reduces register pressure vs the full kStride*kStride layout.
    flat_stride = q.mmax + 1
    flat_gsize = (q.nmax + 1) * flat_stride

    # Build component expressions with compact stride.
    A_comps = cart_components(q.la)
    B_comps = cart_components(q.lb)
    C_comps = cart_components(q.lc)
    D_comps = cart_components(q.ld)

    comp_exprs_x: list[str] = []
    comp_exprs_y: list[str] = []
    comp_exprs_z: list[str] = []
    for ab in range(nAB):
        ia = ab // nB
        ib = ab - ia * nB
        ax, ay, az = A_comps[ia]
        bx, by, bz = B_comps[ib]
        for cd in range(nCD):
            ic = cd // nD
            id_ = cd - ic * nD
            cx, cy, cz = C_comps[ic]
            dx, dy, dz = D_comps[id_]
            tx = shift_terms(ax, bx, cx, dx)
            ty = shift_terms(ay, by, cy, dy)
            tz = shift_terms(az, bz, cz, dz)
            comp_exprs_x.append(axis_expr(tx, stride=flat_stride, gsym="Gx", dij="xij", dkl="xkl"))
            comp_exprs_y.append(axis_expr(ty, stride=flat_stride, gsym="Gy", dij="yij", dkl="ykl"))
            comp_exprs_z.append(axis_expr(tz, stride=flat_stride, gsym="Gz", dij="zij", dkl="zkl"))

    dedup_x = dedup_axis_exprs(comp_exprs_x)
    dedup_y = dedup_axis_exprs(comp_exprs_y)
    dedup_z = dedup_axis_exprs(comp_exprs_z)

    flat_launch_threads = 128  # must match kFlatThreads in emit_launchers
    fn: list[str] = []
    fn.extend([
        f"template <int NROOTS, bool kTileF32 = false, bool kMixedPrec = false>",
        f"__global__ void __launch_bounds__({flat_launch_threads}) KernelERI_{q.name}_flat(",
        "    const int32_t* __restrict__ task_spAB,",
        "    const int32_t* __restrict__ task_spCD,",
        "    int ntasks,",
        "    const int32_t* __restrict__ sp_A,",
        "    const int32_t* __restrict__ sp_B,",
        "    const int32_t* __restrict__ sp_pair_start,",
        "    const int32_t* __restrict__ sp_npair,",
        "    const double* __restrict__ shell_cx,",
        "    const double* __restrict__ shell_cy,",
        "    const double* __restrict__ shell_cz,",
        "    const double* __restrict__ pair_eta,",
        "    const double* __restrict__ pair_Px,",
        "    const double* __restrict__ pair_Py,",
        "    const double* __restrict__ pair_Pz,",
        "    const double* __restrict__ pair_cK,",
        "    double* __restrict__ eri_out_f64,",
        "    float*  __restrict__ eri_out_f32) {",
        "  const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);",
        "  if (t >= ntasks) return;",
        "",
        "  const int spAB = static_cast<int>(task_spAB[t]);",
        "  const int spCD = static_cast<int>(task_spCD[t]);",
        "  const int A = static_cast<int>(sp_A[spAB]);",
        "  const int B = static_cast<int>(sp_B[spAB]);",
        "  const int C = static_cast<int>(sp_A[spCD]);",
        "  const int D = static_cast<int>(sp_B[spCD]);",
        "",
        "  const double Ax = shell_cx[A];",
        "  const double Ay = shell_cy[A];",
        "  const double Az = shell_cz[A];",
        "  const double Bx = shell_cx[B];",
        "  const double By = shell_cy[B];",
        "  const double Bz = shell_cz[B];",
        "  const double Cx = shell_cx[C];",
        "  const double Cy = shell_cy[C];",
        "  const double Cz = shell_cz[C];",
        "  const double Dx = shell_cx[D];",
        "  const double Dy = shell_cy[D];",
        "  const double Dz = shell_cz[D];",
        "",
        "  const double xij = Ax - Bx;",
        "  const double yij = Ay - By;",
        "  const double zij = Az - Bz;",
        "  const double xkl = Cx - Dx;",
        "  const double ykl = Cy - Dy;",
        "  const double zkl = Cz - Dz;",
        "  const double xij2 = xij * xij;",
        "  const double yij2 = yij * yij;",
        "  const double zij2 = zij * zij;",
        "  const double xkl2 = xkl * xkl;",
        "  const double ykl2 = ykl * ykl;",
        "  const double zkl2 = zkl * zkl;",
        "",
        "  const int baseAB = static_cast<int>(sp_pair_start[spAB]);",
        "  const int baseCD = static_cast<int>(sp_pair_start[spCD]);",
        "  const int nPairAB = static_cast<int>(sp_npair[spAB]);",
        "  const int nPairCD = static_cast<int>(sp_npair[spCD]);",
        "",
        f"  constexpr int kFlatStride = {flat_stride};",
        f"  constexpr int kNComp = {ncomp};",
        f"  constexpr int kNMax = {q.nmax};",
        f"  constexpr int kMMax = {q.mmax};",
        "",
        "  // G arrays in compact layout: (nmax+1)*(mmax+1) entries per axis.",
        f"  double Gx[{flat_gsize}];",
        f"  double Gy[{flat_gsize}];",
        f"  double Gz[{flat_gsize}];",
        "  // Component type: FP32 when kMixedPrec, else FP64.",
        "  using comp_t = typename std::conditional<kMixedPrec, float, double>::type;",
        f"  comp_t Ux[{len(dedup_x.uniq)}];",
        f"  comp_t Uy[{len(dedup_y.uniq)}];",
        f"  comp_t Uz[{len(dedup_z.uniq)}];",
        f"  double tile[{ncomp}];",
    ])
    for i in range(ncomp):
        fn.append(f"  tile[{i}] = 0.0;")
    fn.extend([
        "",
        "  for (int ip = 0; ip < nPairAB; ++ip) {",
        "    const int ki = baseAB + ip;",
        "    const double p = pair_eta[ki];",
        "    const double Px = pair_Px[ki];",
        "    const double Py = pair_Py[ki];",
        "    const double Pz = pair_Pz[ki];",
        "    const double cKi = pair_cK[ki];",
        "    for (int jp = 0; jp < nPairCD; ++jp) {",
        "      const int kj = baseCD + jp;",
        "      const double q = pair_eta[kj];",
        "      const double Qx = pair_Px[kj];",
        "      const double Qy = pair_Py[kj];",
        "      const double Qz = pair_Pz[kj];",
        "      const double dx = Px - Qx;",
        "      const double dy = Py - Qy;",
        "      const double dz = Pz - Qz;",
        "      const double PQ2 = dx * dx + dy * dy + dz * dz;",
        "      const double denom = p + q;",
        "      const double omega = p * q / denom;",
        "      const double T = omega * PQ2;",
        "      const double base = kTwoPiToFiveHalves / (p * q * ::sqrt(denom)) * cKi * pair_cK[kj];",
        "      double roots[NROOTS], weights[NROOTS];",
        "      cueri_rys::rys_roots_weights<NROOTS>(T, roots, weights);",
        "      for (int u = 0; u < NROOTS; ++u) {",
        "        const double x = roots[u];",
        "        const double w = weights[u];",
        "        const double inv_denom = 1.0 / denom;",
        "        const double B0 = x * 0.5 * inv_denom;",
        "        const double B1 = (1.0 - x) * 0.5 / p + B0;",
        "        const double B1p = (1.0 - x) * 0.5 / q + B0;",
        "",
        "        const double Cx_ = (Px - Ax) + (q * inv_denom) * x * (Qx - Px);",
        "        const double Cy_ = (Py - Ay) + (q * inv_denom) * x * (Qy - Py);",
        "        const double Cz_ = (Pz - Az) + (q * inv_denom) * x * (Qz - Pz);",
        "        const double Cpx_ = (Qx - Cx) + (p * inv_denom) * x * (Px - Qx);",
        "        const double Cpy_ = (Qy - Cy) + (p * inv_denom) * x * (Py - Qy);",
        "        const double Cpz_ = (Qz - Cz) + (p * inv_denom) * x * (Pz - Qz);",
        "",
        "        compute_G_stride_fixed<kFlatStride, kNMax, kMMax>(Gx, Cx_, Cpx_, B0, B1, B1p);",
        "        compute_G_stride_fixed<kFlatStride, kNMax, kMMax>(Gy, Cy_, Cpy_, B0, B1, B1p);",
        "        compute_G_stride_fixed<kFlatStride, kNMax, kMMax>(Gz, Cz_, Cpz_, B0, B1, B1p);",
        "        const double sc = base * w;",
    ])
    for i, expr in enumerate(dedup_x.uniq):
        fn.append(f"        Ux[{i}] = static_cast<comp_t>({expr});")
    for i, expr in enumerate(dedup_y.uniq):
        fn.append(f"        Uy[{i}] = static_cast<comp_t>({expr});")
    for i, expr in enumerate(dedup_z.uniq):
        fn.append(f"        Uz[{i}] = static_cast<comp_t>({expr});")
    for i in range(ncomp):
        xi = dedup_x.map_idx[i]
        yi = dedup_y.map_idx[i]
        zi = dedup_z.map_idx[i]
        # When kMixedPrec: Ux*Uy in FP32, promote to FP64 for accumulation.
        fn.append(
            f"        tile[{i}] += sc * static_cast<double>(Ux[{xi}] * Uy[{yi}]) * static_cast<double>(Uz[{zi}]);"
        )
    fn.extend([
        "      }",
        "    }",
        "  }",
        "",
        "  // Write output tile: FP32 or FP64 depending on kTileF32.",
        "  if constexpr (kTileF32) {",
        f"    float* out = eri_out_f32 + static_cast<int64_t>(t) * static_cast<int64_t>({ncomp});",
    ])
    for i in range(ncomp):
        fn.append(f"    out[{i}] = __double2float_rn(tile[{i}]);")
    fn.extend([
        "  } else {",
        f"    double* out = eri_out_f64 + static_cast<int64_t>(t) * static_cast<int64_t>({ncomp});",
    ])
    for i in range(ncomp):
        fn.append(f"    out[{i}] = tile[{i}];")
    fn.extend([
        "  }",
        "}",
        "",
    ])
    return "\n".join(fn)


def emit_fused_fock_kernel(q: QuartetClass, stride: int) -> str:
    nA = ncart(q.la)
    nB = ncart(q.lb)
    nC = ncart(q.lc)
    nD = ncart(q.ld)
    nAB = nA * nB
    nCD = nC * nD

    out: list[str] = []
    out.extend(
        [
            "template <int NROOTS, bool kMixedPrec = false>",
            f"__global__ void __launch_bounds__({int(FUSED_FOCK_THREADS_OVERRIDES.get(q.name, 64))}) KernelFusedFock_{q.name}_fixed(",
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    const int32_t* shell_ao_start,",
            "    int nao,",
            "    const double* D_mat,",
            "    double* F_mat,",
            "    int n_bufs) {",
            "  const int lane = static_cast<int>(threadIdx.x) & 31;",
            "  const int warp_id = static_cast<int>(threadIdx.x) >> 5;",
            "  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;",
            "  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;",
            "  if (t >= ntasks) return;",
            "",
            f"  constexpr int kStride = {stride};",
            f"  constexpr int kNComp = {q.ncomp};",
            f"  constexpr int kNMax = {q.nmax};",
            f"  constexpr int kMMax = {q.mmax};",
            "  constexpr int kGSize = kStride * kStride;",
            "  constexpr int kWarpDoubles = 3 * kGSize + 2 * NROOTS + 11 + kNComp;",
            "",
            "  extern __shared__ char sh_raw[];",
            "  double* sh_warp = reinterpret_cast<double*>(sh_raw) + static_cast<int64_t>(warp_id) * kWarpDoubles;",
            "  double* Gx = sh_warp;",
            "  double* Gy = sh_warp + kGSize;",
            "  double* Gz = sh_warp + 2 * kGSize;",
            "  double* sh_roots = sh_warp + 3 * kGSize;",
            "  double* sh_weights = sh_roots + NROOTS;",
            "  double* sh_sc = sh_weights + NROOTS;",
            "  double* tile = sh_sc + 11;",
            "",
            "  const int spAB = static_cast<int>(task_spAB[t]);",
            "  const int spCD = static_cast<int>(task_spCD[t]);",
            "  const int A = static_cast<int>(sp_A[spAB]);",
            "  const int B = static_cast<int>(sp_B[spAB]);",
            "  const int C = static_cast<int>(sp_A[spCD]);",
            "  const int D = static_cast<int>(sp_B[spCD]);",
            "",
            "  const double Ax = shell_cx[A];",
            "  const double Ay = shell_cy[A];",
            "  const double Az = shell_cz[A];",
            "  const double Bx = shell_cx[B];",
            "  const double By = shell_cy[B];",
            "  const double Bz = shell_cz[B];",
            "  const double Cx = shell_cx[C];",
            "  const double Cy = shell_cy[C];",
            "  const double Cz = shell_cz[C];",
            "  const double Dx = shell_cx[D];",
            "  const double Dy = shell_cy[D];",
            "  const double Dz = shell_cz[D];",
            "",
            "  const double xij = Ax - Bx;",
            "  const double yij = Ay - By;",
            "  const double zij = Az - Bz;",
            "  const double xkl = Cx - Dx;",
            "  const double ykl = Cy - Dy;",
            "  const double zkl = Cz - Dz;",
            "  const double xij2 = xij * xij;",
            "  const double yij2 = yij * yij;",
            "  const double zij2 = zij * zij;",
            "  const double xkl2 = xkl * xkl;",
            "  const double ykl2 = ykl * ykl;",
            "  const double zkl2 = zkl * zkl;",
            "",
            "  const int baseAB = static_cast<int>(sp_pair_start[spAB]);",
            "  const int baseCD = static_cast<int>(sp_pair_start[spCD]);",
            "  const int nPairAB = static_cast<int>(sp_npair[spAB]);",
            "  const int nPairCD = static_cast<int>(sp_npair[spCD]);",
            "",
            "  using comp_t = typename std::conditional<kMixedPrec, float, double>::type;",
            "",
            "  for (int ebase = 0; ebase < kNComp; ebase += 32) {",
            "    const int e = ebase + lane;",
            "    const bool active = (e < kNComp);",
            "    double val = 0.0;",
            "    for (int ip = 0; ip < nPairAB; ++ip) {",
            "      const int ki = baseAB + ip;",
            "      for (int jp = 0; jp < nPairCD; ++jp) {",
            "        const int kj = baseCD + jp;",
            "        if (lane == 0) {",
            "          sh_sc[1] = pair_eta[ki];",
            "          sh_sc[2] = pair_eta[kj];",
            "          sh_sc[3] = pair_Px[ki];",
            "          sh_sc[4] = pair_Py[ki];",
            "          sh_sc[5] = pair_Pz[ki];",
            "          sh_sc[6] = pair_Px[kj];",
            "          sh_sc[7] = pair_Py[kj];",
            "          sh_sc[8] = pair_Pz[kj];",
            "          const double dx = sh_sc[3] - sh_sc[6];",
            "          const double dy = sh_sc[4] - sh_sc[7];",
            "          const double dz = sh_sc[5] - sh_sc[8];",
            "          const double PQ2 = dx * dx + dy * dy + dz * dz;",
            "          sh_sc[9] = sh_sc[1] + sh_sc[2];",
            "          const double omega = sh_sc[1] * sh_sc[2] / sh_sc[9];",
            "          const double T = omega * PQ2;",
            "          sh_sc[10] = kTwoPiToFiveHalves / (sh_sc[1] * sh_sc[2] * ::sqrt(sh_sc[9])) * pair_cK[ki] * pair_cK[kj];",
            "          cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots, sh_weights);",
            "        }",
            "        __syncwarp();",
            "        for (int u = 0; u < NROOTS; ++u) {",
            "          if (lane == 0) {",
            "            const double x = sh_roots[u];",
            "            const double w = sh_weights[u];",
            "            const double inv_denom = 1.0 / sh_sc[9];",
            "            const double B0 = x * 0.5 * inv_denom;",
            "            const double B1 = (1.0 - x) * 0.5 / sh_sc[1] + B0;",
            "            const double B1p = (1.0 - x) * 0.5 / sh_sc[2] + B0;",
            "",
            "            const double Cx_ = (sh_sc[3] - Ax) + (sh_sc[2] * inv_denom) * x * (sh_sc[6] - sh_sc[3]);",
            "            const double Cy_ = (sh_sc[4] - Ay) + (sh_sc[2] * inv_denom) * x * (sh_sc[7] - sh_sc[4]);",
            "            const double Cz_ = (sh_sc[5] - Az) + (sh_sc[2] * inv_denom) * x * (sh_sc[8] - sh_sc[5]);",
            "            const double Cpx_ = (sh_sc[6] - Cx) + (sh_sc[1] * inv_denom) * x * (sh_sc[3] - sh_sc[6]);",
            "            const double Cpy_ = (sh_sc[7] - Cy) + (sh_sc[1] * inv_denom) * x * (sh_sc[4] - sh_sc[7]);",
            "            const double Cpz_ = (sh_sc[8] - Cz) + (sh_sc[1] * inv_denom) * x * (sh_sc[5] - sh_sc[8]);",
            "",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gx, Cx_, Cpx_, B0, B1, B1p);",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gy, Cy_, Cpy_, B0, B1, B1p);",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gz, Cz_, Cpz_, B0, B1, B1p);",
            "            sh_sc[0] = sh_sc[10] * w;",
            "          }",
            "          __syncwarp();",
            "          if (active) {",
            "            const comp_t Ix = static_cast<comp_t>(eval_%s_x(e, Gx, xij, xij2, xkl, xkl2));" % q.name,
            "            const comp_t Iy = static_cast<comp_t>(eval_%s_y(e, Gy, yij, yij2, ykl, ykl2));" % q.name,
            "            const comp_t Iz = static_cast<comp_t>(eval_%s_z(e, Gz, zij, zij2, zkl, zkl2));" % q.name,
            "            val += sh_sc[0] * static_cast<double>(Ix * Iy) * static_cast<double>(Iz);",
            "          }",
            "          __syncwarp();",
            "        }",
            "      }",
            "    }",
            "    if (active) tile[e] = val;",
            "  }",
            "",
            "  __syncwarp();",
            "  {",
            f"    constexpr int nA = {nA}, nB = {nB}, nC = {nC}, nD = {nD};",
            f"    constexpr int nAB = {nAB};",
            f"    constexpr int nCD = {nCD};",
            "    const int a0 = static_cast<int>(shell_ao_start[A]);",
            "    const int b0 = static_cast<int>(shell_ao_start[B]);",
            "    const int c0 = static_cast<int>(shell_ao_start[C]);",
            "    const int d0 = static_cast<int>(shell_ao_start[D]);",
            "    const bool ab_neq = (A != B);",
            "    const bool cd_neq = (C != D);",
            "    const bool bk_swap = (spAB != spCD);",
            "    const double f_ab = ab_neq ? 2.0 : 1.0;",
            "    const double f_cd = cd_neq ? 2.0 : 1.0;",
            "    const int64_t N = static_cast<int64_t>(nao);",
            "    const int buf_id = static_cast<int>(blockIdx.x) % n_bufs;",
            "    cueri_contract_fock_warp_single(",
            "        tile, D_mat, F_mat, lane,",
            "        nAB, nCD, nA, nB, nC, nD,",
            "        a0, b0, c0, d0,",
            "        ab_neq, cd_neq, bk_swap, f_ab, f_cd, N, n_bufs, buf_id);",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(out)


def emit_fused_fock_launcher(q: QuartetClass, stride: int) -> str:
    launch_threads = int(FUSED_FOCK_THREADS_OVERRIDES.get(q.name, 64))
    nroots = q.nroots
    return "\n".join(
        [
            f'extern "C" cudaError_t cueri_fused_fock_{q.name}_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    const int32_t* shell_ao_start,",
            "    int nao,",
            "    const double* D_mat,",
            "    double* F_mat,",
            "    cudaStream_t stream,",
            "    int threads,",
            "    int n_bufs,",
            "    bool mixed_prec) {",
            "  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;",
            "  if (ntasks == 0) return cudaSuccess;",
            f"  constexpr int kDefaultThreads = {launch_threads};",
            "  int launch_threads = 0;",
            "  int blocks = 0;",
            f"  constexpr int kGSize_{q.name} = {stride} * {stride};",
            f"  constexpr int kWarpDoubles_{q.name} = 3 * kGSize_{q.name} + 2 * {nroots} + 11 + {q.ncomp};",
            f"  size_t shmem_{q.name} = 0;",
            "  if (mixed_prec) {",
            f"    const cudaError_t prep_{q.name} = cueri_prepare_fused_fock_warp_launch(",
            f"        KernelFusedFock_{q.name}_fixed<{nroots}, true>,",
            "        threads, kDefaultThreads, ntasks,",
            f"        kWarpDoubles_{q.name}, &launch_threads, &blocks, &shmem_{q.name});",
            f"    if (prep_{q.name} != cudaSuccess) return prep_{q.name};",
            f"    KernelFusedFock_{q.name}_fixed<{nroots}, true><<<blocks, launch_threads, shmem_{q.name}, stream>>>(",
            "        task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,",
            "        shell_cx, shell_cy, shell_cz,",
            "        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,",
            "        shell_ao_start, nao, D_mat, F_mat, n_bufs);",
            "  } else {",
            f"    const cudaError_t prep_{q.name} = cueri_prepare_fused_fock_warp_launch(",
            f"        KernelFusedFock_{q.name}_fixed<{nroots}, false>,",
            "        threads, kDefaultThreads, ntasks,",
            f"        kWarpDoubles_{q.name}, &launch_threads, &blocks, &shmem_{q.name});",
            f"    if (prep_{q.name} != cudaSuccess) return prep_{q.name};",
            f"    KernelFusedFock_{q.name}_fixed<{nroots}, false><<<blocks, launch_threads, shmem_{q.name}, stream>>>(",
            "        task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,",
            "        shell_cx, shell_cy, shell_cz,",
            "        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,",
            "        shell_ao_start, nao, D_mat, F_mat, n_bufs);",
            "  }",
            "  return cudaGetLastError();",
            "}",
            "",
        ]
    )


def emit_fused_jk_kernel(q: QuartetClass, stride: int) -> str:
    nA = ncart(q.la)
    nB = ncart(q.lb)
    nC = ncart(q.lc)
    nD = ncart(q.ld)
    nAB = nA * nB
    nCD = nC * nD

    out: list[str] = []
    out.extend(
        [
            "template <int NROOTS, bool kMixedPrec = false>",
            f"__global__ void __launch_bounds__({int(FUSED_JK_THREADS_OVERRIDES.get(q.name, 64))}) KernelFusedJK_{q.name}_fixed(",
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    const int32_t* shell_ao_start,",
            "    int nao,",
            "    const double* D_mat,",
            "    double* J_mat,",
            "    double* K_mat,",
            "    int n_bufs) {",
            "  const int lane = static_cast<int>(threadIdx.x) & 31;",
            "  const int warp_id = static_cast<int>(threadIdx.x) >> 5;",
            "  const int warps_per_block = static_cast<int>(blockDim.x) >> 5;",
            "  const int t = static_cast<int>(blockIdx.x) * warps_per_block + warp_id;",
            "  if (t >= ntasks) return;",
            "",
            f"  constexpr int kStride = {stride};",
            f"  constexpr int kNComp = {q.ncomp};",
            f"  constexpr int kNMax = {q.nmax};",
            f"  constexpr int kMMax = {q.mmax};",
            "  constexpr int kGSize = kStride * kStride;",
            "  constexpr int kWarpDoubles = 3 * kGSize + 2 * NROOTS + 11 + kNComp;",
            "",
            "  extern __shared__ char sh_raw[];",
            "  double* sh_warp = reinterpret_cast<double*>(sh_raw) + static_cast<int64_t>(warp_id) * kWarpDoubles;",
            "  double* Gx = sh_warp;",
            "  double* Gy = sh_warp + kGSize;",
            "  double* Gz = sh_warp + 2 * kGSize;",
            "  double* sh_roots = sh_warp + 3 * kGSize;",
            "  double* sh_weights = sh_roots + NROOTS;",
            "  double* sh_sc = sh_weights + NROOTS;",
            "  double* tile = sh_sc + 11;",
            "",
            "  const int spAB = static_cast<int>(task_spAB[t]);",
            "  const int spCD = static_cast<int>(task_spCD[t]);",
            "  const int A = static_cast<int>(sp_A[spAB]);",
            "  const int B = static_cast<int>(sp_B[spAB]);",
            "  const int C = static_cast<int>(sp_A[spCD]);",
            "  const int D = static_cast<int>(sp_B[spCD]);",
            "",
            "  const double Ax = shell_cx[A];",
            "  const double Ay = shell_cy[A];",
            "  const double Az = shell_cz[A];",
            "  const double Bx = shell_cx[B];",
            "  const double By = shell_cy[B];",
            "  const double Bz = shell_cz[B];",
            "  const double Cx = shell_cx[C];",
            "  const double Cy = shell_cy[C];",
            "  const double Cz = shell_cz[C];",
            "  const double Dx = shell_cx[D];",
            "  const double Dy = shell_cy[D];",
            "  const double Dz = shell_cz[D];",
            "",
            "  const double xij = Ax - Bx;",
            "  const double yij = Ay - By;",
            "  const double zij = Az - Bz;",
            "  const double xkl = Cx - Dx;",
            "  const double ykl = Cy - Dy;",
            "  const double zkl = Cz - Dz;",
            "  const double xij2 = xij * xij;",
            "  const double yij2 = yij * yij;",
            "  const double zij2 = zij * zij;",
            "  const double xkl2 = xkl * xkl;",
            "  const double ykl2 = ykl * ykl;",
            "  const double zkl2 = zkl * zkl;",
            "",
            "  const int baseAB = static_cast<int>(sp_pair_start[spAB]);",
            "  const int baseCD = static_cast<int>(sp_pair_start[spCD]);",
            "  const int nPairAB = static_cast<int>(sp_npair[spAB]);",
            "  const int nPairCD = static_cast<int>(sp_npair[spCD]);",
            "",
            "  using comp_t = typename std::conditional<kMixedPrec, float, double>::type;",
            "",
            "  for (int ebase = 0; ebase < kNComp; ebase += 32) {",
            "    const int e = ebase + lane;",
            "    const bool active = (e < kNComp);",
            "    double val = 0.0;",
            "    for (int ip = 0; ip < nPairAB; ++ip) {",
            "      const int ki = baseAB + ip;",
            "      for (int jp = 0; jp < nPairCD; ++jp) {",
            "        const int kj = baseCD + jp;",
            "        if (lane == 0) {",
            "          sh_sc[1] = pair_eta[ki];",
            "          sh_sc[2] = pair_eta[kj];",
            "          sh_sc[3] = pair_Px[ki];",
            "          sh_sc[4] = pair_Py[ki];",
            "          sh_sc[5] = pair_Pz[ki];",
            "          sh_sc[6] = pair_Px[kj];",
            "          sh_sc[7] = pair_Py[kj];",
            "          sh_sc[8] = pair_Pz[kj];",
            "          const double dx = sh_sc[3] - sh_sc[6];",
            "          const double dy = sh_sc[4] - sh_sc[7];",
            "          const double dz = sh_sc[5] - sh_sc[8];",
            "          const double PQ2 = dx * dx + dy * dy + dz * dz;",
            "          sh_sc[9] = sh_sc[1] + sh_sc[2];",
            "          const double omega = sh_sc[1] * sh_sc[2] / sh_sc[9];",
            "          const double T = omega * PQ2;",
            "          sh_sc[10] = kTwoPiToFiveHalves / (sh_sc[1] * sh_sc[2] * ::sqrt(sh_sc[9])) * pair_cK[ki] * pair_cK[kj];",
            "          cueri_rys::rys_roots_weights<NROOTS>(T, sh_roots, sh_weights);",
            "        }",
            "        __syncwarp();",
            "        for (int u = 0; u < NROOTS; ++u) {",
            "          if (lane == 0) {",
            "            const double x = sh_roots[u];",
            "            const double w = sh_weights[u];",
            "            const double inv_denom = 1.0 / sh_sc[9];",
            "            const double B0 = x * 0.5 * inv_denom;",
            "            const double B1 = (1.0 - x) * 0.5 / sh_sc[1] + B0;",
            "            const double B1p = (1.0 - x) * 0.5 / sh_sc[2] + B0;",
            "",
            "            const double Cx_ = (sh_sc[3] - Ax) + (sh_sc[2] * inv_denom) * x * (sh_sc[6] - sh_sc[3]);",
            "            const double Cy_ = (sh_sc[4] - Ay) + (sh_sc[2] * inv_denom) * x * (sh_sc[7] - sh_sc[4]);",
            "            const double Cz_ = (sh_sc[5] - Az) + (sh_sc[2] * inv_denom) * x * (sh_sc[8] - sh_sc[5]);",
            "            const double Cpx_ = (sh_sc[6] - Cx) + (sh_sc[1] * inv_denom) * x * (sh_sc[3] - sh_sc[6]);",
            "            const double Cpy_ = (sh_sc[7] - Cy) + (sh_sc[1] * inv_denom) * x * (sh_sc[4] - sh_sc[7]);",
            "            const double Cpz_ = (sh_sc[8] - Cz) + (sh_sc[1] * inv_denom) * x * (sh_sc[5] - sh_sc[8]);",
            "",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gx, Cx_, Cpx_, B0, B1, B1p);",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gy, Cy_, Cpy_, B0, B1, B1p);",
            "            compute_G_stride_fixed<kStride, kNMax, kMMax>(Gz, Cz_, Cpz_, B0, B1, B1p);",
            "            sh_sc[0] = sh_sc[10] * w;",
            "          }",
            "          __syncwarp();",
            "          if (active) {",
            "            const comp_t Ix = static_cast<comp_t>(eval_%s_x(e, Gx, xij, xij2, xkl, xkl2));" % q.name,
            "            const comp_t Iy = static_cast<comp_t>(eval_%s_y(e, Gy, yij, yij2, ykl, ykl2));" % q.name,
            "            const comp_t Iz = static_cast<comp_t>(eval_%s_z(e, Gz, zij, zij2, zkl, zkl2));" % q.name,
            "            val += sh_sc[0] * static_cast<double>(Ix * Iy) * static_cast<double>(Iz);",
            "          }",
            "          __syncwarp();",
            "        }",
            "      }",
            "    }",
            "    if (active) tile[e] = val;",
            "  }",
            "",
            "  __syncwarp();",
            "  {",
            f"    constexpr int nA = {nA}, nB = {nB}, nC = {nC}, nD = {nD};",
            f"    constexpr int nAB = {nAB};",
            f"    constexpr int nCD = {nCD};",
            "    const int a0 = static_cast<int>(shell_ao_start[A]);",
            "    const int b0 = static_cast<int>(shell_ao_start[B]);",
            "    const int c0 = static_cast<int>(shell_ao_start[C]);",
            "    const int d0 = static_cast<int>(shell_ao_start[D]);",
            "    const bool ab_neq = (A != B);",
            "    const bool cd_neq = (C != D);",
            "    const bool bk_swap = (spAB != spCD);",
            "    const double f_ab = ab_neq ? 2.0 : 1.0;",
            "    const double f_cd = cd_neq ? 2.0 : 1.0;",
            "    const int64_t N = static_cast<int64_t>(nao);",
            "    const int buf_id = static_cast<int>(blockIdx.x) % n_bufs;",
            "    cueri_contract_jk_warp_single(",
            "        tile, D_mat, J_mat, K_mat, lane,",
            "        nAB, nCD, nA, nB, nC, nD,",
            "        a0, b0, c0, d0,",
            "        ab_neq, cd_neq, bk_swap, f_ab, f_cd, N, n_bufs, buf_id);",
            "  }",
            "}",
            "",
        ]
    )
    return "\n".join(out)


def emit_fused_jk_launcher(q: QuartetClass, stride: int) -> str:
    launch_threads = int(FUSED_JK_THREADS_OVERRIDES.get(q.name, 64))
    nroots = q.nroots
    return "\n".join(
        [
            f'extern "C" cudaError_t cueri_fused_jk_{q.name}_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    const int32_t* shell_ao_start,",
            "    int nao,",
            "    const double* D_mat,",
            "    double* J_mat,",
            "    double* K_mat,",
            "    cudaStream_t stream,",
            "    int threads,",
            "    int n_bufs,",
            "    bool mixed_prec) {",
            "  if (ntasks < 0 || nao <= 0) return cudaErrorInvalidValue;",
            "  if (ntasks == 0) return cudaSuccess;",
            f"  constexpr int kDefaultThreads = {launch_threads};",
            "  int launch_threads = 0;",
            "  int blocks = 0;",
            f"  constexpr int kGSize_{q.name} = {stride} * {stride};",
            f"  constexpr int kWarpDoubles_{q.name} = 3 * kGSize_{q.name} + 2 * {nroots} + 11 + {q.ncomp};",
            f"  size_t shmem_{q.name} = 0;",
            "  if (mixed_prec) {",
            f"    const cudaError_t prep_{q.name} = cueri_prepare_fused_fock_warp_launch(",
            f"        KernelFusedJK_{q.name}_fixed<{nroots}, true>,",
            "        threads, kDefaultThreads, ntasks,",
            f"        kWarpDoubles_{q.name}, &launch_threads, &blocks, &shmem_{q.name});",
            f"    if (prep_{q.name} != cudaSuccess) return prep_{q.name};",
            f"    KernelFusedJK_{q.name}_fixed<{nroots}, true><<<blocks, launch_threads, shmem_{q.name}, stream>>>(",
            "        task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,",
            "        shell_cx, shell_cy, shell_cz,",
            "        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,",
            "        shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);",
            "  } else {",
            f"    const cudaError_t prep_{q.name} = cueri_prepare_fused_fock_warp_launch(",
            f"        KernelFusedJK_{q.name}_fixed<{nroots}, false>,",
            "        threads, kDefaultThreads, ntasks,",
            f"        kWarpDoubles_{q.name}, &launch_threads, &blocks, &shmem_{q.name});",
            f"    if (prep_{q.name} != cudaSuccess) return prep_{q.name};",
            f"    KernelFusedJK_{q.name}_fixed<{nroots}, false><<<blocks, launch_threads, shmem_{q.name}, stream>>>(",
            "        task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair,",
            "        shell_cx, shell_cy, shell_cz,",
            "        pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK,",
            "        shell_ao_start, nao, D_mat, J_mat, K_mat, n_bufs);",
            "  }",
            "  return cudaGetLastError();",
            "}",
            "",
        ]
    )


def emit_launchers(q: QuartetClass, stride: int) -> str:
    nroots = q.nroots
    launch_threads = int(LAUNCH_THREADS_OVERRIDES.get(q.name, q.ncomp))
    fn = []
    fn.extend(
        [
            f'extern "C" cudaError_t cueri_eri_{q.name}_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    double* eri_out,",
            "    cudaStream_t stream,",
            "    int threads) {",
            "  if (ntasks <= 0) return (ntasks == 0) ? cudaSuccess : cudaErrorInvalidValue;",
            "  // Flat kernel: one thread per task, 100% utilization.",
            "  constexpr int kFlatThreads = 128;",
            "  const int blocks = (ntasks + kFlatThreads - 1) / kFlatThreads;",
            f"  KernelERI_{q.name}_flat<{nroots}, false, false><<<blocks, kFlatThreads, 0, stream>>>(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, nullptr);",
            "  return cudaGetLastError();",
            "}",
            "",
            # --- FP32 tile output launcher ---
            f'extern "C" cudaError_t cueri_eri_{q.name}_f32_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    float* eri_out_f32,",
            "    cudaStream_t stream,",
            "    int threads) {",
            "  if (ntasks <= 0) return (ntasks == 0) ? cudaSuccess : cudaErrorInvalidValue;",
            "  constexpr int kFlatThreads = 128;",
            "  const int blocks = (ntasks + kFlatThreads - 1) / kFlatThreads;",
            f"  KernelERI_{q.name}_flat<{nroots}, true, false><<<blocks, kFlatThreads, 0, stream>>>(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, nullptr, eri_out_f32);",
            "  return cudaGetLastError();",
            "}",
            "",
            # --- Mixed-precision launcher (FP32 components + FP64 tile) ---
            f'extern "C" cudaError_t cueri_eri_{q.name}_mixed_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    double* eri_out,",
            "    cudaStream_t stream,",
            "    int threads) {",
            "  if (ntasks <= 0) return (ntasks == 0) ? cudaSuccess : cudaErrorInvalidValue;",
            "  constexpr int kFlatThreads = 128;",
            "  const int blocks = (ntasks + kFlatThreads - 1) / kFlatThreads;",
            f"  KernelERI_{q.name}_flat<{nroots}, false, true><<<blocks, kFlatThreads, 0, stream>>>(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, nullptr);",
            "  return cudaGetLastError();",
            "}",
            "",
            # --- Mixed-precision + FP32 tile output launcher ---
            f'extern "C" cudaError_t cueri_eri_{q.name}_mixed_f32_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    float* eri_out_f32,",
            "    cudaStream_t stream,",
            "    int threads) {",
            "  if (ntasks <= 0) return (ntasks == 0) ? cudaSuccess : cudaErrorInvalidValue;",
            "  constexpr int kFlatThreads = 128;",
            "  const int blocks = (ntasks + kFlatThreads - 1) / kFlatThreads;",
            f"  KernelERI_{q.name}_flat<{nroots}, true, true><<<blocks, kFlatThreads, 0, stream>>>(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, nullptr, eri_out_f32);",
            "  return cudaGetLastError();",
            "}",
            "",
            f'extern "C" cudaError_t cueri_eri_{q.name}_warp_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    double* eri_out,",
            "    cudaStream_t stream,",
            "    int threads) {",]
    )
    # Warp launcher delegates to the flat kernel.  The generated Rys-based
    # warp kernels (KernelERI_*_warp_true) are available but benchmarking
    # shows they are slower than flat for Rys-quadrature classes because each
    # lane independently computes expensive Rys roots/weights.  The true warp
    # kernels remain compiled and can be dispatched by setting an env var or
    # by a future auto-tuner.
    fn.extend([
        f"  return cueri_eri_{q.name}_launch_stream(",
        "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
        "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);",
        "}",
    ])
    # Multiblock launcher: delegate to flat kernel (ignoring partial_sums/blocks_per_task).
    fn.extend([
            "",
            f'extern "C" cudaError_t cueri_eri_{q.name}_multiblock_launch_stream(',
            "    const int32_t* task_spAB,",
            "    const int32_t* task_spCD,",
            "    int ntasks,",
            "    const int32_t* sp_A,",
            "    const int32_t* sp_B,",
            "    const int32_t* sp_pair_start,",
            "    const int32_t* sp_npair,",
            "    const double* shell_cx,",
            "    const double* shell_cy,",
            "    const double* shell_cz,",
            "    const double* pair_eta,",
            "    const double* pair_Px,",
            "    const double* pair_Py,",
            "    const double* pair_Pz,",
            "    const double* pair_cK,",
            "    double* partial_sums,",
            "    int blocks_per_task,",
            "    double* eri_out,",
            "    cudaStream_t stream,",
            "    int threads) {",
            "  (void)partial_sums; (void)blocks_per_task;",
            f"  return cueri_eri_{q.name}_launch_stream(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);",
            "}",
            "",
        ]
    )
    if q.name in FUSED_FOCK_CLASSES:
        fn.append(emit_fused_fock_launcher(q, stride))
    if q.name in FUSED_JK_CLASSES:
        fn.append(emit_fused_jk_launcher(q, stride))
    return "\n".join(fn)


def generate_text(*, wave_name: str, classes: list[QuartetClass]) -> str:
    if not classes:
        raise ValueError("classes must be non-empty")

    has_fused_fock = any(q.name in FUSED_FOCK_CLASSES for q in classes)
    has_fused_jk = any(q.name in FUSED_JK_CLASSES for q in classes)
    out: list[str] = [
        "// Generated by asuka/cueri/cuda/tools/gen_cuda_kernels.py",
        f"// {wave_name} specialized quartet kernels.",
        "",
        "#include <cuda_runtime.h>",
        "",
        "#include <cmath>",
        "#include <cstdint>",
        "#include <type_traits>",
        "",
        '#include "cueri_cuda_kernels_api.h"',
    ]
    if has_fused_fock:
        out.append('#include "cueri_cuda_contract_fock_warp.cuh"')
    if has_fused_jk:
        out.append('#include "cueri_cuda_contract_jk_warp.cuh"')
    out.extend(
        [
            '#include "cueri_cuda_rys_device.cuh"',
            "",
            "namespace {",
            "",
            "constexpr double kPi = 3.141592653589793238462643383279502884;",
            "constexpr double kTwoPiToFiveHalves = 2.0 * kPi * kPi * 1.772453850905516027298167483341145182;  // 2*pi^(5/2)",
            "",
            "template <int STRIDE, int NMAX, int MMAX>",
            "__device__ __forceinline__ void compute_G_stride_fixed(",
            "    double* G,",
            "    double C,",
            "    double Cp,",
            "    double B0,",
            "    double B1,",
            "    double B1p) {",
            "  G[0] = 1.0;",
            "  if constexpr (NMAX > 0) G[1 * STRIDE + 0] = C;",
            "  if constexpr (MMAX > 0) G[0 * STRIDE + 1] = Cp;",
            "  if constexpr (NMAX >= 2) {",
            "    #pragma unroll",
            "    for (int a = 2; a <= NMAX; ++a) {",
            "      G[a * STRIDE + 0] = B1 * static_cast<double>(a - 1) * G[(a - 2) * STRIDE + 0] + C * G[(a - 1) * STRIDE + 0];",
            "    }",
            "  }",
            "  if constexpr (MMAX >= 2) {",
            "    #pragma unroll",
            "    for (int b = 2; b <= MMAX; ++b) {",
            "      G[0 * STRIDE + b] = B1p * static_cast<double>(b - 1) * G[0 * STRIDE + (b - 2)] + Cp * G[0 * STRIDE + (b - 1)];",
            "    }",
            "  }",
            "  if constexpr (NMAX > 0 && MMAX > 0) {",
            "    #pragma unroll",
            "    for (int a = 1; a <= NMAX; ++a) {",
            "      G[a * STRIDE + 1] = static_cast<double>(a) * B0 * G[(a - 1) * STRIDE + 0] + Cp * G[a * STRIDE + 0];",
            "      if constexpr (MMAX >= 2) {",
            "        #pragma unroll",
            "        for (int b = 2; b <= MMAX; ++b) {",
            "          G[a * STRIDE + b] =",
            "              B1p * static_cast<double>(b - 1) * G[a * STRIDE + (b - 2)] +",
            "              static_cast<double>(a) * B0 * G[(a - 1) * STRIDE + (b - 1)] +",
            "              Cp * G[a * STRIDE + (b - 1)];",
            "        }",
            "      }",
            "    }",
            "  }",
            "}",
            "",
            # Multiblock helpers no longer needed: all launchers delegate to flat kernel.
        ]
    )

    stride = max(max(q.nmax, q.mmax) for q in classes) + 1

    # Emit eval functions for fused classes in the preamble so they're
    # available in every part after splitting.
    for q in classes:
        has_fused = q.name in FUSED_FOCK_CLASSES or q.name in FUSED_JK_CLASSES
        if has_fused:
            nA = ncart(q.la)
            nB = ncart(q.lb)
            nC = ncart(q.lc)
            nD = ncart(q.ld)
            nAB = nA * nB
            nCD = nC * nD
            A = cart_components(q.la)
            B = cart_components(q.lb)
            C = cart_components(q.lc)
            D = cart_components(q.ld)
            ex, ey, ez = [], [], []
            for ab in range(nAB):
                ia = ab // nB
                ib = ab - ia * nB
                for cd in range(nCD):
                    ic = cd // nD
                    id_ = cd - ic * nD
                    tx = shift_terms(A[ia][0], B[ib][0], C[ic][0], D[id_][0])
                    ty = shift_terms(A[ia][1], B[ib][1], C[ic][1], D[id_][1])
                    tz = shift_terms(A[ia][2], B[ib][2], C[ic][2], D[id_][2])
                    ex.append(axis_expr(tx, stride=stride, gsym="G", dij="xij", dkl="xkl"))
                    ey.append(axis_expr(ty, stride=stride, gsym="G", dij="yij", dkl="ykl"))
                    ez.append(axis_expr(tz, stride=stride, gsym="G", dij="zij", dkl="zkl"))
            out.append(emit_eval_fn(q, "x", ex))
            out.append(emit_eval_fn(q, "y", ey))
            out.append(emit_eval_fn(q, "z", ez))

    for q in classes:
        out.append(emit_kernel(q, stride))
    out.append("}  // namespace")
    out.append("")
    for q in classes:
        out.append(emit_launchers(q, stride))

    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate specialized quartet CUDA kernels")
    ap.add_argument(
        "--wave",
        type=str,
        choices=sorted(WAVE_MAP.keys()),
        default="wave1",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
    )
    args = ap.parse_args()
    wave = str(args.wave)
    out_path = args.out
    if out_path is None:
        out_path = Path(f"asuka/cueri/cuda/ext/src/generated/cueri_cuda_kernels_{wave}_generated.cu")

    text = generate_text(wave_name=wave, classes=WAVE_MAP[wave])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
