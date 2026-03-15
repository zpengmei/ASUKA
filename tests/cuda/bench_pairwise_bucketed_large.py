"""Benchmark bucketed vs non-bucketed pairwise kernel on large cases."""

import numpy as np
import time
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


def _make_random_integrals(norb, rng):
    h1e = rng.standard_normal((norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    eri4 = rng.standard_normal((norb, norb, norb, norb)) * 0.1
    eri4 = eri4 + eri4.transpose(1, 0, 2, 3)
    eri4 = eri4 + eri4.transpose(0, 1, 3, 2)
    eri4 = eri4 + eri4.transpose(2, 3, 0, 1)
    eri4 = eri4 / 8.0
    h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4)
    return h_base, eri4


def bench_large(norb, nelec, twos, nsel, n_warmup=1, n_iter=3, check_correctness=True):
    import cupy as cp
    from asuka.cuguga.drt import build_drt
    from asuka.cuda.cuda_backend import (
        make_device_drt,
        pairwise_materialize_u64_device,
        pairwise_hij_u64_device,
        pairwise_hij_bucketed_u64_device,
        pairwise_build_bucket_data,
    )

    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    if nsel > ncsf:
        nsel = ncsf

    print(f"\nCAS({nelec},{norb}) S={twos/2}, nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(42)
    h_base, eri4 = _make_random_integrals(norb, rng)
    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(eri4.ravel(), dtype=cp.float64))

    # Contiguous selection (CIPSI-realistic)
    start = rng.integers(0, max(1, ncsf - nsel))
    sel_idx = np.arange(start, start + nsel, dtype=np.int64)
    sel_u64 = sel_idx.astype(np.uint64)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_u64, dtype=cp.uint64))

    # Materialize
    materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, nsel, cp)
    steps_all, nodes_all, occ_all, b_all = materialized
    cp.cuda.runtime.deviceSynchronize()

    # Build bucket data
    t0 = time.perf_counter()
    bucket_data = pairwise_build_bucket_data(occ_all, norb, cp)
    cp.cuda.runtime.deviceSynchronize()
    t_bucket_build = time.perf_counter() - t0
    print(f"  Bucket build time: {t_bucket_build:.3f}s "
          f"({bucket_data['nbuckets']} buckets, "
          f"target_list: {len(bucket_data['target_list'])} entries)")

    sort_perm = bucket_data["sort_perm"].astype(cp.int64)
    steps_sorted = cp.ascontiguousarray(steps_all[sort_perm])
    nodes_sorted = cp.ascontiguousarray(nodes_all[sort_perm])
    occ_sorted = cp.ascontiguousarray(occ_all[sort_perm])
    b_sorted = cp.ascontiguousarray(b_all[sort_perm])
    sel_sorted = cp.ascontiguousarray(sel_u64_d[sort_perm])
    materialized_sorted = (steps_sorted, nodes_sorted, occ_sorted, b_sorted)

    # --- Benchmark non-bucketed ---
    for _ in range(n_warmup):
        H_ref, _ = pairwise_hij_u64_device(
            drt, drt_dev, sel_u64_d, nsel, h_base_d, eri4_d, materialized, cp
        )
        cp.cuda.runtime.deviceSynchronize()

    times_ref = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        H_ref, _ = pairwise_hij_u64_device(
            drt, drt_dev, sel_u64_d, nsel, h_base_d, eri4_d, materialized, cp
        )
        cp.cuda.runtime.deviceSynchronize()
        times_ref.append(time.perf_counter() - t0)

    # --- Benchmark bucketed ---
    for _ in range(n_warmup):
        H_buck_sorted, _ = pairwise_hij_bucketed_u64_device(
            drt, drt_dev, sel_sorted, nsel, h_base_d, eri4_d,
            materialized_sorted, bucket_data, cp
        )
        cp.cuda.runtime.deviceSynchronize()

    times_buck = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        H_buck_sorted, _ = pairwise_hij_bucketed_u64_device(
            drt, drt_dev, sel_sorted, nsel, h_base_d, eri4_d,
            materialized_sorted, bucket_data, cp
        )
        cp.cuda.runtime.deviceSynchronize()
        times_buck.append(time.perf_counter() - t0)

    t_ref = np.mean(times_ref)
    t_buck = np.mean(times_buck)
    print(f"  Non-bucketed: {t_ref:.3f}s")
    print(f"  Bucketed:     {t_buck:.3f}s")
    print(f"  Speedup:      {t_ref/t_buck:.2f}x")
    print(f"  Dense H mem:  {nsel*nsel*8/1024/1024:.1f} MB")

    if check_correctness:
        inv_perm = bucket_data["inv_perm"].astype(cp.int64)
        H_buck = H_buck_sorted[inv_perm][:, inv_perm]
        max_diff = float(cp.max(cp.abs(H_ref - H_buck)))
        print(f"  Max diff:     {max_diff:.2e}")
        assert max_diff < 1e-10, f"Matrix mismatch: {max_diff}"
        print(f"  Correctness:  PASS")


if __name__ == "__main__":
    _skip_no_gpu()

    print("=" * 70)
    print("Bucketed vs Non-bucketed: Large Case Benchmarks")
    print("=" * 70)

    cases = [
        # (norb, nelec, twos, nsel)
        (10, 10, 0, 5000),
        (14, 14, 0, 5000),
        (18, 18, 0, 5000),
        (18, 18, 0, 15000),
    ]

    for norb, nelec, twos, nsel in cases:
        try:
            bench_large(norb, nelec, twos, nsel)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
