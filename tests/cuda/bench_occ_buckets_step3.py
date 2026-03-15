"""Step 3 benchmark: Full bucket pipeline (keys + sort + neighbor CSR).
   Measures total preprocessing time and validates the structure."""

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


def bench_full_pipeline(norb, nelec, twos, nsel, n_iter=3):
    import cupy as cp
    from asuka.cuguga.drt import build_drt
    from asuka.cuda.cuda_backend import (
        make_device_drt,
        pairwise_materialize_u64_device,
        pairwise_build_bucket_data,
    )

    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    if nsel > ncsf:
        nsel = ncsf

    print(f"\nCAS({nelec},{norb}) S={twos/2}, nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(42)
    start = rng.integers(0, max(1, ncsf - nsel))
    sel_idx = np.arange(start, start + nsel, dtype=np.int64)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64), dtype=cp.uint64))

    # Materialize
    materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, nsel, cp)
    _, _, occ_all, _ = materialized
    cp.cuda.runtime.deviceSynchronize()

    # Benchmark full pipeline
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        bucket_data = pairwise_build_bucket_data(occ_all, norb, cp)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)

    t_avg = np.mean(times)
    print(f"  Full pipeline time: {t_avg*1000:.1f} ms (avg of {n_iter})")

    # Validate structure
    bd = bucket_data
    nbuckets = bd["nbuckets"]
    sort_perm = cp.asnumpy(bd["sort_perm"])
    inv_perm = cp.asnumpy(bd["inv_perm"])
    bucket_starts = cp.asnumpy(bd["bucket_starts"])
    bucket_sizes = cp.asnumpy(bd["bucket_sizes"])
    neighbor_offsets = cp.asnumpy(bd["neighbor_offsets"])
    neighbor_list = cp.asnumpy(bd["neighbor_list"])
    csf_to_bucket = cp.asnumpy(bd["csf_to_bucket"])

    print(f"  Buckets: {nbuckets}")
    print(f"  Sort perm: {sort_perm.shape}, inv_perm: {inv_perm.shape}")
    print(f"  Neighbor CSR: offsets={neighbor_offsets.shape}, list={neighbor_list.shape}")

    # Validate permutation round-trip
    assert np.all(sort_perm[inv_perm] == np.arange(nsel, dtype=np.int32)), "perm round-trip failed"
    print(f"  Permutation round-trip: PASS")

    # Validate bucket assignments
    for b in range(min(20, nbuckets)):
        s = bucket_starts[b]
        sz = bucket_sizes[b]
        for idx in range(s, s + sz):
            assert csf_to_bucket[idx] == b, f"csf_to_bucket[{idx}]={csf_to_bucket[idx]} != {b}"
    print(f"  Bucket assignments: PASS")

    # Validate neighbor CSR
    total_neighbors = neighbor_offsets[-1]
    avg_neighbors = total_neighbors / nbuckets if nbuckets > 0 else 0
    print(f"  Total neighbor entries: {total_neighbors:,}")
    print(f"  Avg neighbors per bucket: {avg_neighbors:.1f}")

    # Memory estimate for bucket data structures
    mem_sort_perm = nsel * 4  # int32
    mem_inv_perm = nsel * 4
    mem_bucket_starts = nbuckets * 4
    mem_bucket_sizes = nbuckets * 4
    mem_neighbor_offsets = (nbuckets + 1) * 4
    mem_neighbor_list = total_neighbors * 4
    mem_csf_to_bucket = nsel * 4
    total_mem = (mem_sort_perm + mem_inv_perm + mem_bucket_starts + mem_bucket_sizes +
                 mem_neighbor_offsets + mem_neighbor_list + mem_csf_to_bucket)
    print(f"  Total memory for bucket structures: {total_mem/1024:.1f} KB")
    print(f"  Dense H matrix would be: {nsel*nsel*8/1024/1024:.1f} MB")

    # CSF pairs to check with bucketing vs dense
    total_pairs_bucketed = 0
    for b in range(nbuckets):
        sz_b = bucket_sizes[b]
        nb_start = neighbor_offsets[b]
        nb_end = neighbor_offsets[b + 1]
        for ni in range(nb_start, nb_end):
            nb_idx = neighbor_list[ni]
            sz_nb = bucket_sizes[nb_idx]
            total_pairs_bucketed += sz_b * sz_nb

    print(f"  CSF pairs (dense): {nsel*nsel:,}")
    print(f"  CSF pairs (bucketed): {total_pairs_bucketed:,} ({100*total_pairs_bucketed/(nsel*nsel):.1f}%)")
    print(f"  Reduction factor: {nsel*nsel/total_pairs_bucketed:.1f}x")


if __name__ == "__main__":
    _skip_no_gpu()

    print("=" * 70)
    print("Step 3 Benchmark: Full Bucket Pipeline")
    print("=" * 70)

    cases = [
        (10, 10, 0, 1000),
        (10, 10, 0, 5000),
        (18, 18, 0, 5000),
        (18, 18, 0, 15000),
        (24, 24, 0, 15000),
    ]

    for norb, nelec, twos, nsel in cases:
        try:
            bench_full_pipeline(norb, nelec, twos, nsel)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
