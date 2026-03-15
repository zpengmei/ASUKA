"""Step 2 deep analysis: measure actual sparsity pattern with buckets.
   How many (source_bucket, target_bucket) pairs produce nonzero H entries?
   What's the reduction in target iteration vs dense full-nsel scan?"""

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


def analyze_bucket_neighbor_structure(norb, nelec, twos, nsel):
    """Analyze how many bucket-pairs need to be checked."""
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
        nsel = ncsf

    print(f"\nCAS({nelec},{norb}) S={twos/2}, nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)

    rng = np.random.default_rng(42)
    start = rng.integers(0, max(1, ncsf - nsel))
    sel_idx = np.arange(start, start + nsel, dtype=np.int64)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64), dtype=cp.uint64))

    materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, nsel, cp)
    _, _, occ_all, _ = materialized
    cp.cuda.runtime.deviceSynchronize()

    keys = pairwise_compute_occ_keys(occ_all, norb, cp)
    sort_perm, sorted_keys, bucket_starts, bucket_keys = pairwise_build_occ_buckets(keys, cp)
    cp.cuda.runtime.deviceSynchronize()

    bucket_keys_h = cp.asnumpy(bucket_keys)
    bucket_starts_h = cp.asnumpy(bucket_starts)
    nbuckets = len(bucket_keys_h)

    # Compute bucket sizes
    bucket_ends = np.concatenate([bucket_starts_h[1:], [nsel]])
    bucket_sizes = bucket_ends - bucket_starts_h

    print(f"  {nbuckets} buckets, sizes: min={bucket_sizes.min()}, max={bucket_sizes.max()}, "
          f"mean={bucket_sizes.mean():.1f}")

    # Decode all bucket keys into occupation vectors
    bucket_occs = np.zeros((nbuckets, norb), dtype=np.int8)
    for b in range(nbuckets):
        key = bucket_keys_h[b]
        for k in range(norb):
            bucket_occs[b, k] = (key >> (2 * k)) & 3

    # For each pair of buckets, compute occupation difference count
    # This tells us if one-body or two-body coupling is possible
    print(f"  Computing pairwise bucket occupation differences...")
    t0 = time.perf_counter()

    # Vectorized: compute all pairwise differences
    # diff[i,j] = number of positions where bucket_occs[i] != bucket_occs[j]
    if nbuckets <= 8000:
        diff_matrix = np.sum(
            bucket_occs[:, None, :] != bucket_occs[None, :, :], axis=2
        )  # [nbuckets, nbuckets]

        t_diff = time.perf_counter() - t0
        print(f"  Diff matrix computed in {t_diff:.3f}s")

        # Statistics
        n_1b_pairs = np.sum(diff_matrix <= 2)  # one-body connected
        n_2b_pairs = np.sum(diff_matrix <= 4)  # two-body connected
        n_total_pairs = nbuckets * nbuckets

        print(f"  Bucket pairs (total): {n_total_pairs:,}")
        print(f"  1-body connected (≤2 diff): {n_1b_pairs:,} ({100*n_1b_pairs/n_total_pairs:.1f}%)")
        print(f"  2-body connected (≤4 diff): {n_2b_pairs:,} ({100*n_2b_pairs/n_total_pairs:.1f}%)")

        # Weighted by bucket sizes: actual CSF pairs to check vs nsel^2
        # For each bucket pair (a, b), the CSF pairs = size_a * size_b
        size_product = bucket_sizes[:, None] * bucket_sizes[None, :]
        total_csf_pairs_1b = np.sum(size_product[diff_matrix <= 2])
        total_csf_pairs_2b = np.sum(size_product[diff_matrix <= 4])

        print(f"\n  CSF pairs to check (weighted by bucket sizes):")
        print(f"    Dense scan: {nsel}^2 = {nsel*nsel:,}")
        print(f"    1-body bucketed: {total_csf_pairs_1b:,} ({100*total_csf_pairs_1b/(nsel*nsel):.1f}%)")
        print(f"    2-body bucketed: {total_csf_pairs_2b:,} ({100*total_csf_pairs_2b/(nsel*nsel):.1f}%)")

        # Per-source average target count
        avg_targets_1b = []
        avg_targets_2b = []
        for b in range(nbuckets):
            # targets for this bucket = sum of sizes of neighbor buckets
            mask_1b = diff_matrix[b] <= 2
            mask_2b = diff_matrix[b] <= 4
            avg_targets_1b.append(np.sum(bucket_sizes[mask_1b]))
            avg_targets_2b.append(np.sum(bucket_sizes[mask_2b]))

        avg_1b = np.mean(avg_targets_1b)
        avg_2b = np.mean(avg_targets_2b)
        print(f"\n  Average targets per source CSF:")
        print(f"    Dense: {nsel}")
        print(f"    1-body bucketed: {avg_1b:.0f} ({100*avg_1b/nsel:.1f}% of nsel)")
        print(f"    2-body bucketed: {avg_2b:.0f} ({100*avg_2b/nsel:.1f}% of nsel)")
        print(f"    Estimated Phase 2 speedup from bucketing: {nsel/avg_2b:.1f}x")
    else:
        # Sample-based analysis for very large bucket counts
        print(f"  Too many buckets ({nbuckets}) for full analysis, sampling...")
        sample_idx = rng.choice(nbuckets, size=min(500, nbuckets), replace=False)
        targets_2b = []
        for b in sample_idx:
            occ_b = bucket_occs[b]
            diffs = np.sum(bucket_occs != occ_b[None, :], axis=1)
            n2b = np.sum(bucket_sizes[diffs <= 4])
            targets_2b.append(n2b)
        avg_2b = np.mean(targets_2b)
        print(f"  Average 2-body targets per source: {avg_2b:.0f} ({100*avg_2b/nsel:.1f}% of nsel)")
        print(f"  Estimated Phase 2 speedup from bucketing: {nsel/avg_2b:.1f}x")


if __name__ == "__main__":
    _skip_no_gpu()

    print("=" * 70)
    print("Step 2 Deep Analysis: Bucket Neighbor Structure")
    print("=" * 70)

    cases = [
        (10, 10, 0, 1000),
        (10, 10, 0, 5000),
        (18, 18, 0, 5000),
        (18, 18, 0, 15000),
    ]

    for norb, nelec, twos, nsel in cases:
        try:
            analyze_bucket_neighbor_structure(norb, nelec, twos, nsel)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
