#pragma once
// Kahan-Babushka-Neumaier compensated summation helpers for CUDA device code.
//
// These provide O(1)-ULP accumulation error regardless of the number of additions,
// vs. O(N*ULP) for naive summation.  The primary use case is FP32 sigma-vector
// accumulation where hundreds of terms are summed per thread.

// Neumaier compensated addition (generalised Kahan).
// Accumulates `val` into `sum` with running compensation `comp`.
template <typename T>
__device__ __forceinline__ void kahan_add(T& sum, T& comp, T val) {
    T t = sum + val;
    // Neumaier variant: always picks the larger magnitude for the error term.
    if (fabs(sum) >= fabs(val))
        comp += (sum - t) + val;
    else
        comp += (val - t) + sum;
    sum = t;
}

// Compensated warp-shuffle reduction.
// Reduces (sum, comp) across the warp using __shfl_down_sync.
template <typename T>
__device__ __forceinline__ void kahan_warp_reduce(unsigned mask, T& sum, T& comp) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        T other_sum  = __shfl_down_sync(mask, sum, off);
        T other_comp = __shfl_down_sync(mask, comp, off);
        kahan_add(sum, comp, other_sum);
        kahan_add(sum, comp, other_comp);
    }
}
