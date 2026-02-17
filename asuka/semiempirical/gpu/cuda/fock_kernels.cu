extern "C" __device__ __forceinline__ int idx2(int r, int c, int lda) {
    return r * lda + c;
}

extern "C" __device__ __forceinline__ int idx4(int a, int b, int c, int d) {
    return ((a * 4 + b) * 4 + c) * 4 + d;
}

extern "C" __device__ void fill_local_tensor4(
    const double* __restrict__ ri,
    int naoA,
    int naoB,
    double* __restrict__ L
) {
    for (int i = 0; i < 256; ++i) {
        L[i] = 0.0;
    }

    L[idx4(0, 0, 0, 0)] = ri[0];

    if (naoA >= 4) {
        L[idx4(0, 1, 0, 0)] = ri[1];
        L[idx4(1, 0, 0, 0)] = ri[1];
        L[idx4(1, 1, 0, 0)] = ri[2];
        L[idx4(2, 2, 0, 0)] = ri[3];
        L[idx4(3, 3, 0, 0)] = ri[3];
    }

    if (naoB >= 4) {
        L[idx4(0, 0, 0, 1)] = ri[4];
        L[idx4(0, 0, 1, 0)] = ri[4];
        L[idx4(0, 0, 1, 1)] = ri[10];
        L[idx4(0, 0, 2, 2)] = ri[11];
        L[idx4(0, 0, 3, 3)] = ri[11];
    }

    if (naoA >= 4 && naoB >= 4) {
        L[idx4(0, 1, 0, 1)] = ri[5];
        L[idx4(1, 0, 0, 1)] = ri[5];
        L[idx4(0, 1, 1, 0)] = ri[5];
        L[idx4(1, 0, 1, 0)] = ri[5];

        for (int p = 2; p <= 3; ++p) {
            L[idx4(0, p, 0, p)] = ri[6];
            L[idx4(p, 0, 0, p)] = ri[6];
            L[idx4(0, p, p, 0)] = ri[6];
            L[idx4(p, 0, p, 0)] = ri[6];
        }

        L[idx4(1, 1, 0, 1)] = ri[7];
        L[idx4(1, 1, 1, 0)] = ri[7];

        for (int p = 2; p <= 3; ++p) {
            L[idx4(p, p, 0, 1)] = ri[8];
            L[idx4(p, p, 1, 0)] = ri[8];
        }

        for (int p = 2; p <= 3; ++p) {
            L[idx4(1, p, 0, p)] = ri[9];
            L[idx4(p, 1, 0, p)] = ri[9];
            L[idx4(1, p, p, 0)] = ri[9];
            L[idx4(p, 1, p, 0)] = ri[9];
        }

        L[idx4(0, 1, 1, 1)] = ri[12];
        L[idx4(1, 0, 1, 1)] = ri[12];

        for (int p = 2; p <= 3; ++p) {
            L[idx4(0, 1, p, p)] = ri[13];
            L[idx4(1, 0, p, p)] = ri[13];
        }

        for (int p = 2; p <= 3; ++p) {
            L[idx4(0, p, 1, p)] = ri[14];
            L[idx4(p, 0, 1, p)] = ri[14];
            L[idx4(0, p, p, 1)] = ri[14];
            L[idx4(p, 0, p, 1)] = ri[14];
        }

        L[idx4(1, 1, 1, 1)] = ri[15];

        for (int p = 2; p <= 3; ++p) {
            L[idx4(p, p, 1, 1)] = ri[16];
            L[idx4(1, 1, p, p)] = ri[17];
        }

        L[idx4(2, 2, 2, 2)] = ri[18];
        L[idx4(3, 3, 3, 3)] = ri[18];

        for (int p = 2; p <= 3; ++p) {
            L[idx4(1, p, 1, p)] = ri[19];
            L[idx4(p, 1, 1, p)] = ri[19];
            L[idx4(1, p, p, 1)] = ri[19];
            L[idx4(p, 1, p, 1)] = ri[19];
        }

        L[idx4(2, 2, 3, 3)] = ri[20];
        L[idx4(3, 3, 2, 2)] = ri[20];

        L[idx4(2, 3, 2, 3)] = ri[21];
        L[idx4(3, 2, 3, 2)] = ri[21];
        L[idx4(2, 3, 3, 2)] = ri[21];
        L[idx4(3, 2, 2, 3)] = ri[21];
    }
}

extern "C" __device__ void build_w_from_ri(
    const double* __restrict__ ri,
    const double* __restrict__ TA,
    const double* __restrict__ TB,
    int naoA,
    int naoB,
    double* __restrict__ W
) {
    double L[256];
    fill_local_tensor4(ri, naoA, naoB, L);

    for (int i = 0; i < 256; ++i) {
        W[i] = 0.0;
    }

    for (int m = 0; m < naoA; ++m) {
        for (int n = 0; n < naoA; ++n) {
            for (int l = 0; l < naoB; ++l) {
                for (int s = 0; s < naoB; ++s) {
                    double acc = 0.0;
                    for (int a = 0; a < naoA; ++a) {
                        double tma = TA[m * 4 + a];
                        for (int b = 0; b < naoA; ++b) {
                            double tnb = TA[n * 4 + b];
                            for (int c = 0; c < naoB; ++c) {
                                double tlc = TB[l * 4 + c];
                                for (int d = 0; d < naoB; ++d) {
                                    double tsd = TB[s * 4 + d];
                                    acc += tma * tnb * L[idx4(a, b, c, d)] * tlc * tsd;
                                }
                            }
                        }
                    }
                    W[idx4(m, n, l, s)] = acc;
                }
            }
        }
    }
}

extern "C" __device__ void fock_from_w_thread(
    int t,
    const double* __restrict__ W,
    const double* __restrict__ P,
    double* __restrict__ F,
    int a0,
    int b0,
    int naoA,
    int naoB,
    int nao_total
) {
    if (t >= 48) {
        return;
    }

    if (t < 16) {
        int m = t / 4;
        int n = t % 4;
        if (m < naoA && n < naoA) {
            double accJ = 0.0;
            for (int l = 0; l < naoB; ++l) {
                for (int s = 0; s < naoB; ++s) {
                    double p_ls = P[idx2(b0 + l, b0 + s, nao_total)];
                    accJ += p_ls * W[idx4(m, n, l, s)];
                }
            }
            atomicAdd(&F[idx2(a0 + m, a0 + n, nao_total)], accJ);
        }
        return;
    }

    if (t < 32) {
        int tt = t - 16;
        int m = tt / 4;
        int l = tt % 4;
        if (m < naoA && l < naoB) {
            double accK = 0.0;
            for (int n = 0; n < naoA; ++n) {
                for (int s = 0; s < naoB; ++s) {
                    double p_ns = P[idx2(a0 + n, b0 + s, nao_total)];
                    accK += p_ns * W[idx4(m, n, l, s)];
                }
            }
            double val = -0.5 * accK;
            F[idx2(a0 + m, b0 + l, nao_total)] += val;
            F[idx2(b0 + l, a0 + m, nao_total)] += val;
        }
        return;
    }

    int tt = t - 32;
    int l = tt / 4;
    int s = tt % 4;
    if (l < naoB && s < naoB) {
        double accJ = 0.0;
        for (int m = 0; m < naoA; ++m) {
            for (int n = 0; n < naoA; ++n) {
                double p_mn = P[idx2(a0 + m, a0 + n, nao_total)];
                accJ += p_mn * W[idx4(m, n, l, s)];
            }
        }
        atomicAdd(&F[idx2(b0 + l, b0 + s, nao_total)], accJ);
    }
}

extern "C" __global__ void onecenter_fock_kernel(
    const int* __restrict__ ao_off,
    const unsigned char* __restrict__ nao_atom,
    const double* __restrict__ onecenter,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int nat
) {
    int A = blockIdx.x;
    if (A >= nat) {
        return;
    }

    int t = threadIdx.x;
    int nao = (int)nao_atom[A];
    int a0 = ao_off[A];
    const double* G = onecenter + ((size_t)A) * 256;

    if (t >= 16) {
        return;
    }

    int m = t / 4;
    int n = t % 4;
    if (m >= nao || n >= nao) {
        return;
    }

    double accJ = 0.0;
    double accK = 0.0;
    for (int l = 0; l < nao; ++l) {
        for (int s = 0; s < nao; ++s) {
            double p_ls = P[idx2(a0 + l, a0 + s, nao_total)];
            accJ += p_ls * G[idx4(m, n, l, s)];
            accK += p_ls * G[idx4(m, l, n, s)];
        }
    }

    F[idx2(a0 + m, a0 + n, nao_total)] += accJ - 0.5 * accK;
}

extern "C" __global__ void twocenter_fock_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const unsigned char* __restrict__ nao_atom,
    const double* __restrict__ Wblocks,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) {
        return;
    }

    int t = threadIdx.x;
    if (t >= 48) {
        return;
    }

    int A = pair_i[pid];
    int B = pair_j[pid];
    int a0 = ao_off[A];
    int b0 = ao_off[B];
    int naoA = (int)nao_atom[A];
    int naoB = (int)nao_atom[B];

    const double* W = Wblocks + ((size_t)pid) * 256;

    if (t < 16) {
        int m = t / 4;
        int n = t % 4;
        if (m < naoA && n < naoA) {
            double accJ = 0.0;
            for (int l = 0; l < naoB; ++l) {
                for (int s = 0; s < naoB; ++s) {
                    double p_ls = P[idx2(b0 + l, b0 + s, nao_total)];
                    accJ += p_ls * W[idx4(m, n, l, s)];
                }
            }
            atomicAdd(&F[idx2(a0 + m, a0 + n, nao_total)], accJ);
        }
        return;
    }

    if (t < 32) {
        int tt = t - 16;
        int m = tt / 4;
        int l = tt % 4;
        if (m < naoA && l < naoB) {
            double accK = 0.0;
            for (int n = 0; n < naoA; ++n) {
                for (int s = 0; s < naoB; ++s) {
                    double p_ns = P[idx2(a0 + n, b0 + s, nao_total)];
                    accK += p_ns * W[idx4(m, n, l, s)];
                }
            }
            double val = -0.5 * accK;
            F[idx2(a0 + m, b0 + l, nao_total)] += val;
            F[idx2(b0 + l, a0 + m, nao_total)] += val;
        }
        return;
    }

    int tt = t - 32;
    int l = tt / 4;
    int s = tt % 4;
    if (l < naoB && s < naoB) {
        double accJ = 0.0;
        for (int m = 0; m < naoA; ++m) {
            for (int n = 0; n < naoA; ++n) {
                double p_mn = P[idx2(a0 + m, a0 + n, nao_total)];
                accJ += p_mn * W[idx4(m, n, l, s)];
            }
        }
        atomicAdd(&F[idx2(b0 + l, b0 + s, nao_total)], accJ);
    }
}

extern "C" __global__ void twocenter_fock_ri_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const unsigned char* __restrict__ nao_atom,
    const double* __restrict__ ri_pairs,
    const double* __restrict__ ta_pairs,
    const double* __restrict__ tb_pairs,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) {
        return;
    }

    int t = threadIdx.x;

    int A = pair_i[pid];
    int B = pair_j[pid];
    int a0 = ao_off[A];
    int b0 = ao_off[B];
    int naoA = (int)nao_atom[A];
    int naoB = (int)nao_atom[B];

    const double* ri = ri_pairs + ((size_t)pid) * 22;
    const double* TA = ta_pairs + ((size_t)pid) * 16;
    const double* TB = tb_pairs + ((size_t)pid) * 16;

    __shared__ double shW[256];
    if (t == 0) {
        build_w_from_ri(ri, TA, TB, naoA, naoB, shW);
    }
    __syncthreads();

    fock_from_w_thread(t, shW, P, F, a0, b0, naoA, naoB, nao_total);
}

extern "C" __global__ void twocenter_fock_ri_11_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const double* __restrict__ ri_pairs,
    const double* __restrict__ ta_pairs,
    const double* __restrict__ tb_pairs,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) return;
    int t = threadIdx.x;

    int A = pair_i[pid];
    int B = pair_j[pid];
    int a0 = ao_off[A];
    int b0 = ao_off[B];

    const double* ri = ri_pairs + ((size_t)pid) * 22;
    const double* TA = ta_pairs + ((size_t)pid) * 16;
    const double* TB = tb_pairs + ((size_t)pid) * 16;

    __shared__ double shW[256];
    if (t == 0) {
        build_w_from_ri(ri, TA, TB, 1, 1, shW);
    }
    __syncthreads();
    fock_from_w_thread(t, shW, P, F, a0, b0, 1, 1, nao_total);
}

extern "C" __global__ void twocenter_fock_ri_14_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const double* __restrict__ ri_pairs,
    const double* __restrict__ ta_pairs,
    const double* __restrict__ tb_pairs,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) return;
    int t = threadIdx.x;

    int A = pair_i[pid];
    int B = pair_j[pid];
    int a0 = ao_off[A];
    int b0 = ao_off[B];

    const double* ri = ri_pairs + ((size_t)pid) * 22;
    const double* TA = ta_pairs + ((size_t)pid) * 16;
    const double* TB = tb_pairs + ((size_t)pid) * 16;

    __shared__ double shW[256];
    if (t == 0) {
        build_w_from_ri(ri, TA, TB, 1, 4, shW);
    }
    __syncthreads();
    fock_from_w_thread(t, shW, P, F, a0, b0, 1, 4, nao_total);
}

extern "C" __global__ void twocenter_fock_ri_41_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const double* __restrict__ ri_pairs,
    const double* __restrict__ ta_pairs,
    const double* __restrict__ tb_pairs,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) return;
    int t = threadIdx.x;

    int A = pair_i[pid];
    int B = pair_j[pid];
    int a0 = ao_off[A];
    int b0 = ao_off[B];

    const double* ri = ri_pairs + ((size_t)pid) * 22;
    const double* TA = ta_pairs + ((size_t)pid) * 16;
    const double* TB = tb_pairs + ((size_t)pid) * 16;

    __shared__ double shW[256];
    if (t == 0) {
        build_w_from_ri(ri, TA, TB, 4, 1, shW);
    }
    __syncthreads();
    fock_from_w_thread(t, shW, P, F, a0, b0, 4, 1, nao_total);
}

extern "C" __global__ void twocenter_fock_ri_44_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const double* __restrict__ ri_pairs,
    const double* __restrict__ ta_pairs,
    const double* __restrict__ tb_pairs,
    const double* __restrict__ P,
    double* __restrict__ F,
    int nao_total,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) return;
    int t = threadIdx.x;

    int A = pair_i[pid];
    int B = pair_j[pid];
    int a0 = ao_off[A];
    int b0 = ao_off[B];

    const double* ri = ri_pairs + ((size_t)pid) * 22;
    const double* TA = ta_pairs + ((size_t)pid) * 16;
    const double* TB = tb_pairs + ((size_t)pid) * 16;

    __shared__ double shW[256];
    if (t == 0) {
        build_w_from_ri(ri, TA, TB, 4, 4, shW);
    }
    __syncthreads();
    fock_from_w_thread(t, shW, P, F, a0, b0, 4, 4, nao_total);
}

extern "C" __global__ void build_wblocks_from_ri_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const int* __restrict__ ao_off,
    const unsigned char* __restrict__ nao_atom,
    const double* __restrict__ ri_pairs,
    const double* __restrict__ ta_pairs,
    const double* __restrict__ tb_pairs,
    double* __restrict__ wblocks,
    int npairs
) {
    int pid = blockIdx.x;
    if (pid >= npairs) return;
    int t = threadIdx.x;
    if (t != 0) return;

    (void)ao_off;

    int A = pair_i[pid];
    int B = pair_j[pid];
    int naoA = (int)nao_atom[A];
    int naoB = (int)nao_atom[B];
    const double* ri = ri_pairs + ((size_t)pid) * 22;
    const double* TA = ta_pairs + ((size_t)pid) * 16;
    const double* TB = tb_pairs + ((size_t)pid) * 16;
    double* W = wblocks + ((size_t)pid) * 256;
    build_w_from_ri(ri, TA, TB, naoA, naoB, W);
}
