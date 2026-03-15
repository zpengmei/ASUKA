"""Test bucketed pair-wise H[i,j] kernel against the non-bucketed version."""

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


def test_bucketed_vs_dense(norb, nelec, twos, nsel=None, use_contiguous=False):
    """Compare bucketed kernel against non-bucketed (dense) kernel."""
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
    if nsel is None:
        nsel = ncsf
    if nsel > ncsf:
        nsel = ncsf

    print(f"\nCAS({nelec},{norb}) S={twos/2}, nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(42 + norb * 100 + nelec * 10 + twos)
    h_base, eri4 = _make_random_integrals(norb, rng)
    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(eri4.ravel(), dtype=cp.float64))

    if use_contiguous:
        start = rng.integers(0, max(1, ncsf - nsel))
        sel_idx = np.arange(start, start + nsel, dtype=np.int64)
    elif nsel < ncsf:
        sel_idx = np.sort(rng.choice(ncsf, size=nsel, replace=False).astype(np.int64))
    else:
        sel_idx = np.arange(ncsf, dtype=np.int64)

    sel_u64 = sel_idx.astype(np.uint64)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_u64, dtype=cp.uint64))

    # Non-bucketed (reference)
    materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, nsel, cp)
    cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    H_ref, diag_ref = pairwise_hij_u64_device(
        drt, drt_dev, sel_u64_d, nsel, h_base_d, eri4_d, materialized, cp
    )
    cp.cuda.runtime.deviceSynchronize()
    t_ref = time.perf_counter() - t0

    # Bucketed
    steps_all, nodes_all, occ_all, b_all = materialized
    bucket_data = pairwise_build_bucket_data(occ_all, norb, cp)
    cp.cuda.runtime.deviceSynchronize()

    sort_perm = bucket_data["sort_perm"].astype(cp.int64)

    # Sort materialized arrays and sel_idx by occupation key
    steps_sorted = cp.ascontiguousarray(steps_all[sort_perm])
    nodes_sorted = cp.ascontiguousarray(nodes_all[sort_perm])
    occ_sorted = cp.ascontiguousarray(occ_all[sort_perm])
    b_sorted = cp.ascontiguousarray(b_all[sort_perm])
    sel_sorted = cp.ascontiguousarray(sel_u64_d[sort_perm])

    materialized_sorted = (steps_sorted, nodes_sorted, occ_sorted, b_sorted)

    t0 = time.perf_counter()
    H_buck_sorted, diag_buck_sorted = pairwise_hij_bucketed_u64_device(
        drt, drt_dev, sel_sorted, nsel, h_base_d, eri4_d,
        materialized_sorted, bucket_data, cp
    )
    cp.cuda.runtime.deviceSynchronize()
    t_buck = time.perf_counter() - t0

    # Unpermute bucketed H to original order
    inv_perm = bucket_data["inv_perm"].astype(cp.int64)
    H_buck = H_buck_sorted[inv_perm][:, inv_perm]

    H_ref_h = cp.asnumpy(H_ref)
    H_buck_h = cp.asnumpy(H_buck)

    max_diff = np.max(np.abs(H_ref_h - H_buck_h))
    diag_diff = np.max(np.abs(np.diag(H_ref_h) - np.diag(H_buck_h)))

    # Check symmetry
    asym_buck = np.max(np.abs(H_buck_h - H_buck_h.T))

    print(f"  Ref time: {t_ref:.3f}s, Bucketed time: {t_buck:.3f}s")
    print(f"  Speedup: {t_ref/t_buck:.2f}x")
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Diag diff: {diag_diff:.2e}")
    print(f"  Asym (bucketed): {asym_buck:.2e}")
    print(f"  Buckets: {bucket_data['nbuckets']}")

    assert max_diff < 1e-10, f"Matrix mismatch: {max_diff}"
    assert diag_diff < 1e-10, f"Diagonal mismatch: {diag_diff}"
    print(f"  PASS")

    return t_ref, t_buck, max_diff


if __name__ == "__main__":
    _skip_no_gpu()

    print("=" * 70)
    print("Bucketed vs Dense Pairwise Kernel: Correctness + Timing")
    print("=" * 70)

    # Small full-space tests (exact match expected)
    cases_full = [
        (4, 4, 0),
        (4, 6, 0),
        (6, 4, 0),
        (6, 6, 0),
        (4, 4, 2),
        (3, 2, 0),
        (2, 2, 0),
    ]

    print("\n--- Full CI space tests ---")
    for norb, nelec, twos in cases_full:
        test_bucketed_vs_dense(norb, nelec, twos)

    # Subset tests with timing
    print("\n--- Subset tests with timing ---")
    cases_subset = [
        (8, 8, 0, 100),
        (8, 8, 0, 500),
        (10, 10, 0, 50),
        (10, 10, 0, 1000),
    ]
    for norb, nelec, twos, nsel in cases_subset:
        test_bucketed_vs_dense(norb, nelec, twos, nsel=nsel)

    # Contiguous subset (CIPSI-realistic)
    print("\n--- Contiguous subset tests ---")
    cases_contig = [
        (10, 10, 0, 1000),
        (10, 10, 0, 5000),
    ]
    for norb, nelec, twos, nsel in cases_contig:
        test_bucketed_vs_dense(norb, nelec, twos, nsel=nsel, use_contiguous=True)
