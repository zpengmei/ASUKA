#include <stddef.h>

#ifndef ASUKA_XC_DENSITY_THREADS
#define ASUKA_XC_DENSITY_THREADS 128
#endif

#ifndef ASUKA_XC_AO_TILE
#define ASUKA_XC_AO_TILE 16
#endif

#ifndef ASUKA_XC_GRID_TILE
#define ASUKA_XC_GRID_TILE 8
#endif

__device__ __forceinline__ double asuka_warp_sum(double v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

extern "C" __global__
void xc_contract_density_fused(
    const double* __restrict__ phi,
    const double* __restrict__ dphi,
    const double* __restrict__ D,
    int npt,
    int nao,
    double* __restrict__ rho,
    double* __restrict__ sigma,
    double* __restrict__ tau,
    double* __restrict__ nabla
) {
    const int g = blockIdx.x;
    const int tid = threadIdx.x;
    if (g >= npt) {
        return;
    }

    const double* phi_g = phi + (size_t)g * (size_t)nao;
    const double* dphi_g = dphi + (size_t)g * (size_t)nao * 3u;

    double rho_acc = 0.0;
    double gx_acc = 0.0;
    double gy_acc = 0.0;
    double gz_acc = 0.0;
    double tau_acc = 0.0;

    for (int nu = tid; nu < nao; nu += blockDim.x) {
        const double* D_row = D + (size_t)nu * (size_t)nao;
        double pDn = 0.0;
        double dpxDn = 0.0;
        double dpyDn = 0.0;
        double dpzDn = 0.0;

        for (int mu = 0; mu < nao; ++mu) {
            const double Dij = D_row[mu];
            const double phi_mu = phi_g[mu];
            const double* dphi_mu = dphi_g + (size_t)mu * 3u;
            pDn += Dij * phi_mu;
            dpxDn += Dij * dphi_mu[0];
            dpyDn += Dij * dphi_mu[1];
            dpzDn += Dij * dphi_mu[2];
        }

        const double phi_nu = phi_g[nu];
        const double* dphi_nu = dphi_g + (size_t)nu * 3u;
        rho_acc += pDn * phi_nu;
        gx_acc += 2.0 * pDn * dphi_nu[0];
        gy_acc += 2.0 * pDn * dphi_nu[1];
        gz_acc += 2.0 * pDn * dphi_nu[2];
        tau_acc += 0.5 * (
            dpxDn * dphi_nu[0]
            + dpyDn * dphi_nu[1]
            + dpzDn * dphi_nu[2]
        );
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nwarp = (blockDim.x + 31) >> 5;

    rho_acc = asuka_warp_sum(rho_acc);
    gx_acc = asuka_warp_sum(gx_acc);
    gy_acc = asuka_warp_sum(gy_acc);
    gz_acc = asuka_warp_sum(gz_acc);
    tau_acc = asuka_warp_sum(tau_acc);

    __shared__ double warp_rho[32];
    __shared__ double warp_gx[32];
    __shared__ double warp_gy[32];
    __shared__ double warp_gz[32];
    __shared__ double warp_tau[32];

    if (lane == 0) {
        warp_rho[warp] = rho_acc;
        warp_gx[warp] = gx_acc;
        warp_gy[warp] = gy_acc;
        warp_gz[warp] = gz_acc;
        warp_tau[warp] = tau_acc;
    }
    __syncthreads();

    if (warp == 0) {
        double r = lane < nwarp ? warp_rho[lane] : 0.0;
        double gx = lane < nwarp ? warp_gx[lane] : 0.0;
        double gy = lane < nwarp ? warp_gy[lane] : 0.0;
        double gz = lane < nwarp ? warp_gz[lane] : 0.0;
        double t = lane < nwarp ? warp_tau[lane] : 0.0;

        r = asuka_warp_sum(r);
        gx = asuka_warp_sum(gx);
        gy = asuka_warp_sum(gy);
        gz = asuka_warp_sum(gz);
        t = asuka_warp_sum(t);

        if (lane == 0) {
            rho[g] = r;
            sigma[g] = gx * gx + gy * gy + gz * gz;
            tau[g] = t;
            nabla[(size_t)g * 3u + 0u] = gx;
            nabla[(size_t)g * 3u + 1u] = gy;
            nabla[(size_t)g * 3u + 2u] = gz;
        }
    }
}

extern "C" __global__
void xc_build_vxc_fused(
    const double* __restrict__ phi,
    const double* __restrict__ dphi,
    const double* __restrict__ weights,
    const double* __restrict__ vrho,
    const double* __restrict__ vsigma,
    const double* __restrict__ vtau,
    const double* __restrict__ nabla,
    int npt,
    int nao,
    double* __restrict__ V
) {
    const int local_nu = threadIdx.x;
    const int local_mu = threadIdx.y;
    if (blockIdx.y > blockIdx.x) {
        return;
    }
    const int nu = blockIdx.x * ASUKA_XC_AO_TILE + local_nu;
    const int mu = blockIdx.y * ASUKA_XC_AO_TILE + local_mu;
    const int linear_tid = local_mu * blockDim.x + local_nu;
    const int nthreads = blockDim.x * blockDim.y;

    __shared__ double s_phi_mu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_phi_nu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dmx[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dmy[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dmz[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dnx[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dny[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dnz[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_tmu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_tnu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_wvrho[ASUKA_XC_GRID_TILE];
    __shared__ double s_wvsigma2[ASUKA_XC_GRID_TILE];
    __shared__ double s_wvtau[ASUKA_XC_GRID_TILE];
    __shared__ double s_nx[ASUKA_XC_GRID_TILE];
    __shared__ double s_ny[ASUKA_XC_GRID_TILE];
    __shared__ double s_nz[ASUKA_XC_GRID_TILE];

    double acc = 0.0;

    for (int g0 = 0; g0 < npt; g0 += ASUKA_XC_GRID_TILE) {
        const int panel_elems = ASUKA_XC_GRID_TILE * ASUKA_XC_AO_TILE;

        for (int idx = linear_tid; idx < panel_elems; idx += nthreads) {
            const int tg = idx / ASUKA_XC_AO_TILE;
            const int ta = idx - tg * ASUKA_XC_AO_TILE;
            const int g = g0 + tg;
            const int mu_idx = blockIdx.y * ASUKA_XC_AO_TILE + ta;
            const int nu_idx = blockIdx.x * ASUKA_XC_AO_TILE + ta;

            if (g < npt && mu_idx < nao) {
                const size_t mu_base = (size_t)g * (size_t)nao * 3u + (size_t)mu_idx * 3u;
                s_phi_mu[tg][ta] = phi[(size_t)g * (size_t)nao + (size_t)mu_idx];
                s_dmx[tg][ta] = dphi[mu_base + 0u];
                s_dmy[tg][ta] = dphi[mu_base + 1u];
                s_dmz[tg][ta] = dphi[mu_base + 2u];
            } else {
                s_phi_mu[tg][ta] = 0.0;
                s_dmx[tg][ta] = 0.0;
                s_dmy[tg][ta] = 0.0;
                s_dmz[tg][ta] = 0.0;
            }

            if (g < npt && nu_idx < nao) {
                const size_t nu_base = (size_t)g * (size_t)nao * 3u + (size_t)nu_idx * 3u;
                s_phi_nu[tg][ta] = phi[(size_t)g * (size_t)nao + (size_t)nu_idx];
                s_dnx[tg][ta] = dphi[nu_base + 0u];
                s_dny[tg][ta] = dphi[nu_base + 1u];
                s_dnz[tg][ta] = dphi[nu_base + 2u];
            } else {
                s_phi_nu[tg][ta] = 0.0;
                s_dnx[tg][ta] = 0.0;
                s_dny[tg][ta] = 0.0;
                s_dnz[tg][ta] = 0.0;
            }
        }

        for (int idx = linear_tid; idx < ASUKA_XC_GRID_TILE; idx += nthreads) {
            const int g = g0 + idx;
            if (g < npt) {
                const double w = weights[g];
                s_wvrho[idx] = w * vrho[g];
                s_wvsigma2[idx] = 2.0 * w * vsigma[g];
                s_wvtau[idx] = 0.5 * w * vtau[g];
                s_nx[idx] = nabla[(size_t)g * 3u + 0u];
                s_ny[idx] = nabla[(size_t)g * 3u + 1u];
                s_nz[idx] = nabla[(size_t)g * 3u + 2u];
            } else {
                s_wvrho[idx] = 0.0;
                s_wvsigma2[idx] = 0.0;
                s_wvtau[idx] = 0.0;
                s_nx[idx] = 0.0;
                s_ny[idx] = 0.0;
                s_nz[idx] = 0.0;
            }
        }

        __syncthreads();

        // Precompute dphi · nabla_rho for all AOs in each tile to avoid doing
        // 2 dot products per AO-pair in the accumulation loop.
        for (int idx = linear_tid; idx < panel_elems; idx += nthreads) {
            const int tg = idx / ASUKA_XC_AO_TILE;
            const int ta = idx - tg * ASUKA_XC_AO_TILE;
            const double gradx = s_nx[tg];
            const double grady = s_ny[tg];
            const double gradz = s_nz[tg];
            s_tmu[tg][ta] = s_dmx[tg][ta] * gradx + s_dmy[tg][ta] * grady + s_dmz[tg][ta] * gradz;
            s_tnu[tg][ta] = s_dnx[tg][ta] * gradx + s_dny[tg][ta] * grady + s_dnz[tg][ta] * gradz;
        }

        __syncthreads();

        if (mu < nao && nu < nao) {
            #pragma unroll
            for (int tg = 0; tg < ASUKA_XC_GRID_TILE; ++tg) {
                const double phi_mu = s_phi_mu[tg][local_mu];
                const double phi_nu = s_phi_nu[tg][local_nu];
                const double dmx = s_dmx[tg][local_mu];
                const double dmy = s_dmy[tg][local_mu];
                const double dmz = s_dmz[tg][local_mu];
                const double dnx = s_dnx[tg][local_nu];
                const double dny = s_dny[tg][local_nu];
                const double dnz = s_dnz[tg][local_nu];
                const double dm_dot_grad = s_tmu[tg][local_mu];
                const double dn_dot_grad = s_tnu[tg][local_nu];
                const double grad_pair = dmx * dnx + dmy * dny + dmz * dnz;

                acc += s_wvrho[tg] * phi_mu * phi_nu
                    + s_wvsigma2[tg] * (dm_dot_grad * phi_nu + phi_mu * dn_dot_grad)
                    + s_wvtau[tg] * grad_pair;
            }
        }

        __syncthreads();
    }

    if (mu < nao && nu < nao) {
        if (blockIdx.x == blockIdx.y && local_mu > local_nu) {
            return;
        }
        V[(size_t)mu * (size_t)nao + (size_t)nu] = acc;
        if (mu != nu) {
            V[(size_t)nu * (size_t)nao + (size_t)mu] = acc;
        }
    }
}

extern "C" __global__
void xc_build_vxc_fused_chunked(
    const double* __restrict__ phi,
    const double* __restrict__ dphi,
    const double* __restrict__ weights,
    const double* __restrict__ vrho,
    const double* __restrict__ vsigma,
    const double* __restrict__ vtau,
    const double* __restrict__ nabla,
    int npt,
    int nao,
    double* __restrict__ V
) {
    const int local_nu = threadIdx.x;
    const int local_mu = threadIdx.y;
    if (blockIdx.y > blockIdx.x) {
        return;
    }
    const int nu = blockIdx.x * ASUKA_XC_AO_TILE + local_nu;
    const int mu = blockIdx.y * ASUKA_XC_AO_TILE + local_mu;
    const int linear_tid = local_mu * blockDim.x + local_nu;
    const int nthreads = blockDim.x * blockDim.y;

    const int nchunks = (int)gridDim.z;
    const int chunk_id = (int)blockIdx.z;
    const int chunk_size = (npt + nchunks - 1) / nchunks;
    const int g_start = chunk_id * chunk_size;
    const int g_end = g_start + chunk_size < npt ? g_start + chunk_size : npt;

    __shared__ double s_phi_mu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_phi_nu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dmx[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dmy[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dmz[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dnx[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dny[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_dnz[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_tmu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_tnu[ASUKA_XC_GRID_TILE][ASUKA_XC_AO_TILE];
    __shared__ double s_wvrho[ASUKA_XC_GRID_TILE];
    __shared__ double s_wvsigma2[ASUKA_XC_GRID_TILE];
    __shared__ double s_wvtau[ASUKA_XC_GRID_TILE];
    __shared__ double s_nx[ASUKA_XC_GRID_TILE];
    __shared__ double s_ny[ASUKA_XC_GRID_TILE];
    __shared__ double s_nz[ASUKA_XC_GRID_TILE];

    double acc = 0.0;

    for (int g0 = g_start; g0 < g_end; g0 += ASUKA_XC_GRID_TILE) {
        const int panel_elems = ASUKA_XC_GRID_TILE * ASUKA_XC_AO_TILE;

        for (int idx = linear_tid; idx < panel_elems; idx += nthreads) {
            const int tg = idx / ASUKA_XC_AO_TILE;
            const int ta = idx - tg * ASUKA_XC_AO_TILE;
            const int g = g0 + tg;
            const int mu_idx = blockIdx.y * ASUKA_XC_AO_TILE + ta;
            const int nu_idx = blockIdx.x * ASUKA_XC_AO_TILE + ta;

            if (g < g_end && mu_idx < nao) {
                const size_t mu_base = (size_t)g * (size_t)nao * 3u + (size_t)mu_idx * 3u;
                s_phi_mu[tg][ta] = phi[(size_t)g * (size_t)nao + (size_t)mu_idx];
                s_dmx[tg][ta] = dphi[mu_base + 0u];
                s_dmy[tg][ta] = dphi[mu_base + 1u];
                s_dmz[tg][ta] = dphi[mu_base + 2u];
            } else {
                s_phi_mu[tg][ta] = 0.0;
                s_dmx[tg][ta] = 0.0;
                s_dmy[tg][ta] = 0.0;
                s_dmz[tg][ta] = 0.0;
            }

            if (g < g_end && nu_idx < nao) {
                const size_t nu_base = (size_t)g * (size_t)nao * 3u + (size_t)nu_idx * 3u;
                s_phi_nu[tg][ta] = phi[(size_t)g * (size_t)nao + (size_t)nu_idx];
                s_dnx[tg][ta] = dphi[nu_base + 0u];
                s_dny[tg][ta] = dphi[nu_base + 1u];
                s_dnz[tg][ta] = dphi[nu_base + 2u];
            } else {
                s_phi_nu[tg][ta] = 0.0;
                s_dnx[tg][ta] = 0.0;
                s_dny[tg][ta] = 0.0;
                s_dnz[tg][ta] = 0.0;
            }
        }

        for (int idx = linear_tid; idx < ASUKA_XC_GRID_TILE; idx += nthreads) {
            const int g = g0 + idx;
            if (g < g_end) {
                const double w = weights[g];
                s_wvrho[idx] = w * vrho[g];
                s_wvsigma2[idx] = 2.0 * w * vsigma[g];
                s_wvtau[idx] = 0.5 * w * vtau[g];
                s_nx[idx] = nabla[(size_t)g * 3u + 0u];
                s_ny[idx] = nabla[(size_t)g * 3u + 1u];
                s_nz[idx] = nabla[(size_t)g * 3u + 2u];
            } else {
                s_wvrho[idx] = 0.0;
                s_wvsigma2[idx] = 0.0;
                s_wvtau[idx] = 0.0;
                s_nx[idx] = 0.0;
                s_ny[idx] = 0.0;
                s_nz[idx] = 0.0;
            }
        }

        __syncthreads();

        // Precompute dphi · nabla_rho for all AOs in each tile to avoid doing
        // 2 dot products per AO-pair in the accumulation loop.
        for (int idx = linear_tid; idx < panel_elems; idx += nthreads) {
            const int tg = idx / ASUKA_XC_AO_TILE;
            const int ta = idx - tg * ASUKA_XC_AO_TILE;
            const double gradx = s_nx[tg];
            const double grady = s_ny[tg];
            const double gradz = s_nz[tg];
            s_tmu[tg][ta] = s_dmx[tg][ta] * gradx + s_dmy[tg][ta] * grady + s_dmz[tg][ta] * gradz;
            s_tnu[tg][ta] = s_dnx[tg][ta] * gradx + s_dny[tg][ta] * grady + s_dnz[tg][ta] * gradz;
        }

        __syncthreads();

        if (mu < nao && nu < nao) {
            #pragma unroll
            for (int tg = 0; tg < ASUKA_XC_GRID_TILE; ++tg) {
                const double phi_mu = s_phi_mu[tg][local_mu];
                const double phi_nu = s_phi_nu[tg][local_nu];
                const double dmx = s_dmx[tg][local_mu];
                const double dmy = s_dmy[tg][local_mu];
                const double dmz = s_dmz[tg][local_mu];
                const double dnx = s_dnx[tg][local_nu];
                const double dny = s_dny[tg][local_nu];
                const double dnz = s_dnz[tg][local_nu];
                const double dm_dot_grad = s_tmu[tg][local_mu];
                const double dn_dot_grad = s_tnu[tg][local_nu];
                const double grad_pair = dmx * dnx + dmy * dny + dmz * dnz;

                acc += s_wvrho[tg] * phi_mu * phi_nu
                    + s_wvsigma2[tg] * (dm_dot_grad * phi_nu + phi_mu * dn_dot_grad)
                    + s_wvtau[tg] * grad_pair;
            }
        }

        __syncthreads();
    }

    if (mu < nao && nu < nao) {
        if (blockIdx.x == blockIdx.y && local_mu > local_nu) {
            return;
        }
        atomicAdd(V + (size_t)mu * (size_t)nao + (size_t)nu, acc);
        if (mu != nu) {
            atomicAdd(V + (size_t)nu * (size_t)nao + (size_t)mu, acc);
        }
    }
}
