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
    return "\n".join(out)


def emit_launchers(q: QuartetClass) -> str:
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
            "  if (ntasks < 0) return cudaErrorInvalidValue;",
            f"  constexpr int kDefaultThreads = {launch_threads};",
            "  int launch_threads = (threads > 0) ? threads : kDefaultThreads;",
            "  if (launch_threads > 1024) launch_threads = 1024;",
            "  if (launch_threads < 32) launch_threads = 32;",
            "  launch_threads = (launch_threads / 32) * 32;",
            "  if (launch_threads < 32) launch_threads = 32;",
            "  const int blocks = ntasks;",
            f"  KernelERI_{q.name}_fixed<{nroots}><<<blocks, launch_threads, 0, stream>>>(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out);",
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
            "    int threads) {",
            f"  return cueri_eri_{q.name}_launch_stream(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);",
            "}",
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
            "  (void)partial_sums;",
            "  (void)blocks_per_task;",
            f"  return cueri_eri_{q.name}_launch_stream(",
            "      task_spAB, task_spCD, ntasks, sp_A, sp_B, sp_pair_start, sp_npair, shell_cx, shell_cy, shell_cz,",
            "      pair_eta, pair_Px, pair_Py, pair_Pz, pair_cK, eri_out, stream, threads);",
            "}",
            "",
        ]
    )
    return "\n".join(fn)


def generate_text(*, wave_name: str, classes: list[QuartetClass]) -> str:
    if not classes:
        raise ValueError("classes must be non-empty")

    out: list[str] = []
    out.extend(
        [
            "// Generated by asuka/cueri/cuda/tools/gen_cuda_kernels.py",
            f"// {wave_name} specialized quartet kernels.",
            "",
            "#include <cuda_runtime.h>",
            "",
            "#include <cmath>",
            "#include <cstdint>",
            "",
            '#include "cueri_cuda_kernels_api.h"',
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
        ]
    )

    stride = max(max(q.nmax, q.mmax) for q in classes) + 1
    for q in classes:
        out.append(emit_kernel(q, stride))
    out.append("}  // namespace")
    out.append("")
    for q in classes:
        out.append(emit_launchers(q))

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
