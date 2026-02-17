extern "C" __global__ void am1_grad_pair_11_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const double* __restrict__ coords,
    const double* __restrict__ atom_params,
    const double* __restrict__ P_AA,
    const double* __restrict__ P_BB,
    const double* __restrict__ P_AB,
    double* __restrict__ grad,
    int npairs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int pid = tid; pid < npairs; pid += stride) {
        const int A = pair_i[pid];
        const int B = pair_j[pid];
        const double* PAA = P_AA + ((size_t)pid) * 16;
        const double* PBB = P_BB + ((size_t)pid) * 16;
        const double* PAB = P_AB + ((size_t)pid) * 16;

        const Dual3 e = pm_pair_energy_dual<1, 1>(A, B, coords, atom_params, PAA, PBB, PAB);

        atomicAdd(&grad[pm_idx2(A, 0, 3)], -e.dx);
        atomicAdd(&grad[pm_idx2(A, 1, 3)], -e.dy);
        atomicAdd(&grad[pm_idx2(A, 2, 3)], -e.dz);

        atomicAdd(&grad[pm_idx2(B, 0, 3)], e.dx);
        atomicAdd(&grad[pm_idx2(B, 1, 3)], e.dy);
        atomicAdd(&grad[pm_idx2(B, 2, 3)], e.dz);
    }
}

extern "C" __global__ void am1_grad_pair_14_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const double* __restrict__ coords,
    const double* __restrict__ atom_params,
    const double* __restrict__ P_AA,
    const double* __restrict__ P_BB,
    const double* __restrict__ P_AB,
    double* __restrict__ grad,
    int npairs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int pid = tid; pid < npairs; pid += stride) {
        const int A = pair_i[pid];
        const int B = pair_j[pid];
        const double* PAA = P_AA + ((size_t)pid) * 16;
        const double* PBB = P_BB + ((size_t)pid) * 16;
        const double* PAB = P_AB + ((size_t)pid) * 16;

        const Dual3 e = pm_pair_energy_dual<1, 4>(A, B, coords, atom_params, PAA, PBB, PAB);

        atomicAdd(&grad[pm_idx2(A, 0, 3)], -e.dx);
        atomicAdd(&grad[pm_idx2(A, 1, 3)], -e.dy);
        atomicAdd(&grad[pm_idx2(A, 2, 3)], -e.dz);

        atomicAdd(&grad[pm_idx2(B, 0, 3)], e.dx);
        atomicAdd(&grad[pm_idx2(B, 1, 3)], e.dy);
        atomicAdd(&grad[pm_idx2(B, 2, 3)], e.dz);
    }
}

extern "C" __global__ void am1_grad_pair_41_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const double* __restrict__ coords,
    const double* __restrict__ atom_params,
    const double* __restrict__ P_AA,
    const double* __restrict__ P_BB,
    const double* __restrict__ P_AB,
    double* __restrict__ grad,
    int npairs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int pid = tid; pid < npairs; pid += stride) {
        const int A = pair_i[pid];
        const int B = pair_j[pid];
        const double* PAA = P_AA + ((size_t)pid) * 16;
        const double* PBB = P_BB + ((size_t)pid) * 16;
        const double* PAB = P_AB + ((size_t)pid) * 16;

        const Dual3 e = pm_pair_energy_dual<4, 1>(A, B, coords, atom_params, PAA, PBB, PAB);

        atomicAdd(&grad[pm_idx2(A, 0, 3)], -e.dx);
        atomicAdd(&grad[pm_idx2(A, 1, 3)], -e.dy);
        atomicAdd(&grad[pm_idx2(A, 2, 3)], -e.dz);

        atomicAdd(&grad[pm_idx2(B, 0, 3)], e.dx);
        atomicAdd(&grad[pm_idx2(B, 1, 3)], e.dy);
        atomicAdd(&grad[pm_idx2(B, 2, 3)], e.dz);
    }
}

extern "C" __global__ void am1_grad_pair_44_kernel(
    const int* __restrict__ pair_i,
    const int* __restrict__ pair_j,
    const double* __restrict__ coords,
    const double* __restrict__ atom_params,
    const double* __restrict__ P_AA,
    const double* __restrict__ P_BB,
    const double* __restrict__ P_AB,
    double* __restrict__ grad,
    int npairs
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int pid = tid; pid < npairs; pid += stride) {
        const int A = pair_i[pid];
        const int B = pair_j[pid];
        const double* PAA = P_AA + ((size_t)pid) * 16;
        const double* PBB = P_BB + ((size_t)pid) * 16;
        const double* PAB = P_AB + ((size_t)pid) * 16;

        const Dual3 e = pm_pair_energy_dual<4, 4>(A, B, coords, atom_params, PAA, PBB, PAB);

        atomicAdd(&grad[pm_idx2(A, 0, 3)], -e.dx);
        atomicAdd(&grad[pm_idx2(A, 1, 3)], -e.dy);
        atomicAdd(&grad[pm_idx2(A, 2, 3)], -e.dz);

        atomicAdd(&grad[pm_idx2(B, 0, 3)], e.dx);
        atomicAdd(&grad[pm_idx2(B, 1, 3)], e.dy);
        atomicAdd(&grad[pm_idx2(B, 2, 3)], e.dz);
    }
}
