"""Benchmark Step 1+2 of sparse bucketed pairwise kernel:
   occupation key computation and bucket building."""

import time
import numpy as np
import sys


def _skip_no_gpu():
    try:
        import cupy as cp  # noqa: F401
        from asuka.cuda.cuda_backend import has_pairwise_hij_u64_device
        if not has_pairwise_hij_u64_device():
            print("SKIP: pairwise HIJ kernel not available")
            sys.exit(0)
    except Exception:
        print("SKIP: CuPy or CUDA extension not available")
        sys.exit(0)


def bench_occ_keys_and_buckets(norb, nelec, twos, nsel, n_warmup=3, n_iter=10):
    """Benchmark occ key computation + bucketing for given CAS parameters."""
    import cupy as cp
    from asuka.cuguga.drt import build_drt
    from asuka.cuda.cuda_backend import (
        make_device_drt,
        pairwise_materialize_u64_device,
        pairwise_compute_occ_keys,
        pairwise_build_occ_buckets,
    )

    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    if nsel > ncsf:
        print(f"  nsel={nsel} > ncsf={ncsf}, clamping")
        nsel = ncsf

    print(f"\nCAS({nelec},{norb}) S={twos/2}, nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)

    # Select contiguous CSFs (realistic for CIPSI)
    rng = np.random.default_rng(42)
    start = rng.integers(0, max(1, ncsf - nsel))
    sel_idx = np.arange(start, start + nsel, dtype=np.int64)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64), dtype=cp.uint64))

    # Materialize (run once, this is cached in real use)
    materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, nsel, cp)
    steps_all, nodes_all, occ_all, b_all = materialized
    cp.cuda.runtime.deviceSynchronize()

    # --- Benchmark occ key computation ---
    for _ in range(n_warmup):
        keys = pairwise_compute_occ_keys(occ_all, norb, cp)
        cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        keys = pairwise_compute_occ_keys(occ_all, norb, cp)
        cp.cuda.runtime.deviceSynchronize()
    t_keys = (time.perf_counter() - t0) / n_iter

    print(f"  Occ key computation: {t_keys*1000:.3f} ms (nsel={nsel}, norb={norb})")

    # --- Benchmark bucket building ---
    for _ in range(n_warmup):
        sort_perm, sorted_keys, bucket_starts, bucket_keys = pairwise_build_occ_buckets(keys, cp)
        cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    for _ in range(n_iter):
        sort_perm, sorted_keys, bucket_starts, bucket_keys = pairwise_build_occ_buckets(keys, cp)
        cp.cuda.runtime.deviceSynchronize()
    t_buckets = (time.perf_counter() - t0) / n_iter

    nbuckets = len(bucket_keys)
    print(f"  Bucket building: {t_buckets*1000:.3f} ms ({nbuckets} buckets)")
    print(f"  Total (keys + buckets): {(t_keys + t_buckets)*1000:.3f} ms")

    # --- Bucket statistics ---
    bucket_starts_h = cp.asnumpy(bucket_starts)
    sizes = np.diff(np.concatenate([bucket_starts_h, [nsel]]))
    print(f"  Bucket sizes: min={sizes.min()}, max={sizes.max()}, "
          f"mean={sizes.mean():.1f}, median={np.median(sizes):.0f}")

    # --- Verify key correctness ---
    occ_h = cp.asnumpy(occ_all)
    keys_h = cp.asnumpy(keys)
    for i in range(min(10, nsel)):
        key_manual = 0
        for k in range(norb):
            key_manual |= int(occ_h[i, k]) << (2 * k)
        assert keys_h[i] == key_manual, f"Key mismatch at i={i}: {keys_h[i]} vs {key_manual}"
    print(f"  Key correctness: PASS (checked {min(10, nsel)} entries)")

    # --- Neighbor bucket counting (one-body: differ at exactly 2 positions) ---
    # For each bucket, count how many other buckets differ at ≤ 2 occ positions
    bucket_keys_h = cp.asnumpy(bucket_keys)
    if nbuckets <= 5000:
        n1b_neighbors = []
        n2b_neighbors = []
        for i in range(min(100, nbuckets)):
            key_i = bucket_keys_h[i]
            occ_i = np.array([(key_i >> (2*k)) & 3 for k in range(norb)], dtype=np.int8)
            count_1b = 0
            count_2b = 0
            for j in range(nbuckets):
                key_j = bucket_keys_h[j]
                occ_j = np.array([(key_j >> (2*k)) & 3 for k in range(norb)], dtype=np.int8)
                ndiff = np.sum(occ_i != occ_j)
                if ndiff <= 2:
                    count_1b += 1
                if ndiff <= 4:
                    count_2b += 1
            n1b_neighbors.append(count_1b)
            n2b_neighbors.append(count_2b)
        print(f"  1-body neighbor buckets (≤2 pos diff): mean={np.mean(n1b_neighbors):.1f}, "
              f"max={np.max(n1b_neighbors)}")
        print(f"  2-body neighbor buckets (≤4 pos diff): mean={np.mean(n2b_neighbors):.1f}, "
              f"max={np.max(n2b_neighbors)}")

    return t_keys, t_buckets, nbuckets


if __name__ == "__main__":
    _skip_no_gpu()

    print("=" * 70)
    print("Step 1+2 Benchmark: Occupation Key Computation + Bucketing")
    print("=" * 70)

    cases = [
        # (norb, nelec, twos, nsel)
        (10, 10, 0, 1000),
        (10, 10, 0, 5000),
        (18, 18, 0, 5000),
        (18, 18, 0, 15000),
        (24, 24, 0, 5000),
        (24, 24, 0, 15000),
    ]

    for norb, nelec, twos, nsel in cases:
        try:
            bench_occ_keys_and_buckets(norb, nelec, twos, nsel)
        except Exception as e:
            print(f"\n  ERROR: {e}")
