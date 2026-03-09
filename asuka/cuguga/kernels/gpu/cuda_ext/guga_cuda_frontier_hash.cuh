#pragma once

#include <cuda_runtime.h>

#include <cstdint>

// =============================================================================
// Frontier Hash: open-addressing int32 -> float64 accumulator
//
// - keys[cap] uses -1 as empty sentinel.
// - vals is stored root-major: vals[root * cap + slot]
// - cap must be a power of two for mask-based probing.
// =============================================================================

namespace {

__device__ __forceinline__ uint32_t guga_hash_u32(uint32_t x) {
  // 32-bit mix (Murmur3 finalizer style).
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  x ^= x >> 16;
  return x;
}

template <int MAX_PROBES_T = 256>
__device__ __forceinline__ void guga_frontier_hash_insert_add_f64(
    int32_t* __restrict__ keys,
    double* __restrict__ vals_root_major,
    int cap,
    int root,
    int32_t idx,
    double v,
    int* __restrict__ overflow_flag) {
  if (v == 0.0) return;
  if (!keys || !vals_root_major || cap <= 0) {
    if (overflow_flag) atomicExch(overflow_flag, 1);
    return;
  }
  // Require power-of-two capacity (fast mask probing).
  // Caller enforces this; keep the device path branchless.
  uint32_t mask = (uint32_t)(cap - 1);
  uint32_t h = guga_hash_u32((uint32_t)idx);
  uint32_t slot = h & mask;

  for (int probe = 0; probe < MAX_PROBES_T; probe++) {
    // Fast-path: avoid atomics when the slot is occupied by a different key.
    // AtomicCAS on every probe is extremely costly at high occupancy.
    int32_t cur = keys[slot];
    if (cur == idx) {
      atomicAdd(&vals_root_major[(int64_t)root * (int64_t)cap + (int64_t)slot], v);
      return;
    }
    if (cur == (int32_t)-1) {
      int32_t prev = atomicCAS(&keys[slot], (int32_t)-1, idx);
      if (prev == (int32_t)-1 || prev == idx) {
        atomicAdd(&vals_root_major[(int64_t)root * (int64_t)cap + (int64_t)slot], v);
        return;
      }
    }
    slot = (slot + 1) & mask;
  }
  if (overflow_flag) atomicExch(overflow_flag, 1);
}

}  // namespace
