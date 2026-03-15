"""Test pair-wise H[i,j] kernel against the existing dense emitter."""

import numpy as np
import pytest

from asuka.cuguga.drt import build_drt


def _skip_no_gpu():
    """Skip if CuPy or CUDA extension is unavailable."""
    try:
        import cupy as cp  # noqa: F401

        from asuka.cuda.cuda_backend import has_pairwise_hij_u64_device

        if not has_pairwise_hij_u64_device():
            pytest.skip("pairwise HIJ kernel not available in CUDA extension")
    except Exception:
        pytest.skip("CuPy or CUDA extension not available")


def _build_dense_H_from_emitter(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp):
    """Build dense H matrix using the existing COO dense emitter, for reference."""
    from asuka.cuda.cuda_backend import (
        build_selected_membership_hash,
        cas36_exact_selected_emit_tuples_dense_u64_inplace_device,
    )

    nsel = len(sel_idx)
    sel_u64 = np.asarray(sel_idx, dtype=np.uint64)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_u64, dtype=cp.uint64).ravel())
    sel_sorted_d = cp.sort(sel_u64_d.copy())

    hash_keys_d, hash_cap = build_selected_membership_hash(sel_sorted_d, cp)
    c_bound_d = cp.ones((nsel,), dtype=cp.float64)

    # Retry with growing cap (emitter can produce multiple COO entries per pair)
    cap = max(nsel * nsel * 20, 4096)
    for _retry in range(5):
        out_keys = cp.zeros((cap,), dtype=cp.uint64)
        out_src = cp.zeros((cap,), dtype=cp.int32)
        out_hij = cp.zeros((cap,), dtype=cp.float64)
        out_diag = cp.zeros((nsel,), dtype=cp.float64)
        out_n = cp.zeros((1,), dtype=cp.int32)
        overflow = cp.zeros((1,), dtype=cp.int32)

        cas36_exact_selected_emit_tuples_dense_u64_inplace_device(
            drt,
            drt_dev,
            sel_u64_d,
            c_bound_d,
            nsel=nsel,
            h_base=h_base_d,
            eri4=eri4_d,
            out_keys_u64=out_keys,
            out_src=out_src,
            out_hij=out_hij,
            cap=cap,
            membership_hash_keys=hash_keys_d,
            membership_hash_cap=hash_cap,
            out_diag=out_diag,
            out_n=out_n,
            overflow=overflow,
        )
        if int(overflow.item()) == 0:
            break
        cap *= 4

    nnz = int(out_n.item())
    assert int(overflow.item()) == 0, f"dense emitter overflowed even with cap={cap}"

    # Build index mapping: global CSF idx -> local position
    idx_to_pos = {}
    for pos, idx in enumerate(sel_idx):
        idx_to_pos[int(idx)] = pos

    # Assemble dense H from COO
    H = np.zeros((nsel, nsel), dtype=np.float64)
    keys_h = cp.asnumpy(out_keys[:nnz])
    src_h = cp.asnumpy(out_src[:nnz])
    hij_h = cp.asnumpy(out_hij[:nnz])

    for t in range(nnz):
        j_local = int(src_h[t])
        i_global = int(keys_h[t])
        if i_global in idx_to_pos:
            i_local = idx_to_pos[i_global]
            H[j_local, i_local] += hij_h[t]

    # Set diagonal
    diag_h = cp.asnumpy(out_diag)
    for k in range(nsel):
        H[k, k] = diag_h[k]

    # Symmetrize
    H = 0.5 * (H + H.T)
    return H


def _build_dense_H_pairwise(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp):
    """Build dense H matrix using the new pair-wise kernel."""
    from asuka.cuda.cuda_backend import (
        pairwise_hij_u64_device,
        pairwise_materialize_u64_device,
    )

    nsel = len(sel_idx)
    sel_u64_d = cp.ascontiguousarray(
        cp.asarray(np.asarray(sel_idx, dtype=np.uint64), dtype=cp.uint64).ravel()
    )

    materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, nsel, cp)
    H_d, diag_d = pairwise_hij_u64_device(
        drt, drt_dev, sel_u64_d, nsel, h_base_d, eri4_d, materialized, cp
    )
    return cp.asnumpy(H_d)


def _make_random_integrals(norb, rng):
    """Generate random but symmetric h1e and eri4 integrals."""
    h1e = rng.standard_normal((norb, norb))
    h1e = 0.5 * (h1e + h1e.T)

    # Make 4-index ERI with all 8-fold symmetry
    eri4 = rng.standard_normal((norb, norb, norb, norb)) * 0.1
    # (pq|rs) = (qp|rs) = (pq|sr) = (qp|sr) = (rs|pq) = (sr|pq) = (rs|qp) = (sr|qp)
    eri4 = eri4 + eri4.transpose(1, 0, 2, 3)
    eri4 = eri4 + eri4.transpose(0, 1, 3, 2)
    eri4 = eri4 + eri4.transpose(2, 3, 0, 1)
    eri4 = eri4 / 8.0

    h_base = h1e - 0.5 * np.einsum("pqqs->ps", eri4)
    return h_base, eri4


@pytest.mark.parametrize(
    "norb,nelec,twos",
    [
        (4, 4, 0),   # CAS(4,4) singlet - small, manageable
        (4, 6, 0),   # CAS(6,4) singlet
        (6, 4, 0),   # CAS(4,6) singlet
        (6, 6, 0),   # CAS(6,6) singlet
        (4, 4, 2),   # CAS(4,4) triplet
        (3, 2, 0),   # CAS(2,3) singlet - minimal
        (2, 2, 0),   # CAS(2,2) singlet - edge case
    ],
)
def test_pairwise_hij_full_space(norb, nelec, twos):
    """Compare pair-wise kernel against dense emitter on the full CI space."""
    _skip_no_gpu()
    import cupy as cp

    from asuka.cuda.cuda_backend import make_device_drt

    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    if ncsf == 0:
        pytest.skip(f"Empty CI space for norb={norb}, nelec={nelec}, twos={twos}")

    print(f"\nCAS({nelec},{norb}) S={twos/2}, ncsf={ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(42 + norb * 100 + nelec * 10 + twos)
    h_base, eri4 = _make_random_integrals(norb, rng)

    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(eri4.ravel(), dtype=cp.float64))

    # Full CI space: all CSFs selected
    sel_idx = np.arange(ncsf, dtype=np.int64)

    H_ref = _build_dense_H_from_emitter(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)
    H_pw = _build_dense_H_pairwise(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)

    # Check symmetry
    asym_ref = np.max(np.abs(H_ref - H_ref.T))
    asym_pw = np.max(np.abs(H_pw - H_pw.T))
    print(f"  Asymmetry: ref={asym_ref:.2e}, pairwise={asym_pw:.2e}")

    # Check diagonal match
    diag_diff = np.max(np.abs(np.diag(H_ref) - np.diag(H_pw)))
    print(f"  Diagonal max diff: {diag_diff:.2e}")

    # Check full matrix match
    max_diff = np.max(np.abs(H_ref - H_pw))
    rms_diff = np.sqrt(np.mean((H_ref - H_pw) ** 2))
    print(f"  Full matrix max diff: {max_diff:.2e}, RMS: {rms_diff:.2e}")
    print(f"  H_ref range: [{H_ref.min():.4f}, {H_ref.max():.4f}]")
    print(f"  H_pw range: [{H_pw.min():.4f}, {H_pw.max():.4f}]")

    # Tolerances: should match to near machine precision
    assert diag_diff < 1e-10, f"Diagonal mismatch: {diag_diff}"
    assert max_diff < 1e-10, f"Matrix mismatch: {max_diff}"


@pytest.mark.parametrize("nsel", [1, 2, 5, 10])
def test_pairwise_hij_subset(nsel):
    """Test with a random subset of the CI space."""
    _skip_no_gpu()
    import cupy as cp

    from asuka.cuda.cuda_backend import make_device_drt

    norb, nelec, twos = 6, 6, 0
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    if nsel > ncsf:
        pytest.skip(f"nsel={nsel} > ncsf={ncsf}")

    print(f"\nCAS(6,6) subset nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(12345 + nsel)
    h_base, eri4 = _make_random_integrals(norb, rng)

    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(eri4.ravel(), dtype=cp.float64))

    # Random subset
    sel_idx = np.sort(rng.choice(ncsf, size=nsel, replace=False).astype(np.int64))

    H_ref = _build_dense_H_from_emitter(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)
    H_pw = _build_dense_H_pairwise(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)

    max_diff = np.max(np.abs(H_ref - H_pw))
    print(f"  Max diff: {max_diff:.2e}")

    assert max_diff < 1e-10, f"Matrix mismatch: {max_diff}"


def test_pairwise_hij_eigenvalues():
    """Check that eigenvalues from pair-wise kernel match the reference."""
    _skip_no_gpu()
    import cupy as cp

    from asuka.cuda.cuda_backend import make_device_drt

    norb, nelec, twos = 6, 6, 0
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    print(f"\nCAS(6,6) full space eigenvalue test, ncsf={ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(99)
    h_base, eri4 = _make_random_integrals(norb, rng)

    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(eri4.ravel(), dtype=cp.float64))

    sel_idx = np.arange(ncsf, dtype=np.int64)

    H_ref = _build_dense_H_from_emitter(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)
    H_pw = _build_dense_H_pairwise(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)

    evals_ref = np.linalg.eigvalsh(H_ref)
    evals_pw = np.linalg.eigvalsh(H_pw)

    max_eval_diff = np.max(np.abs(evals_ref - evals_pw))
    print(f"  Max eigenvalue diff: {max_eval_diff:.2e}")
    print(f"  Lowest 3 eigenvalues (ref): {evals_ref[:3]}")
    print(f"  Lowest 3 eigenvalues (pw):  {evals_pw[:3]}")

    assert max_eval_diff < 1e-10, f"Eigenvalue mismatch: {max_eval_diff}"


@pytest.mark.parametrize(
    "norb,nelec,twos,nsel",
    [
        (8, 8, 0, 100),   # CAS(8,8) subset
        (8, 8, 0, 500),   # CAS(8,8) larger subset
        (10, 10, 0, 50),  # CAS(10,10) subset - half-filled
    ],
)
def test_pairwise_hij_larger_cases(norb, nelec, twos, nsel):
    """Test on larger half-filled active spaces with selected subsets."""
    _skip_no_gpu()
    import cupy as cp

    from asuka.cuda.cuda_backend import make_device_drt

    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    if nsel > ncsf:
        pytest.skip(f"nsel={nsel} > ncsf={ncsf}")

    print(f"\nCAS({nelec},{norb}) S={twos/2}, nsel={nsel}/{ncsf}")

    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(777 + norb * 100 + nsel)
    h_base, eri4 = _make_random_integrals(norb, rng)

    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(eri4.ravel(), dtype=cp.float64))

    sel_idx = np.sort(rng.choice(ncsf, size=nsel, replace=False).astype(np.int64))

    import time

    t0 = time.perf_counter()
    H_ref = _build_dense_H_from_emitter(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    H_pw = _build_dense_H_pairwise(drt, drt_dev, sel_idx, h_base_d, eri4_d, cp)
    t_pw = time.perf_counter() - t0

    max_diff = np.max(np.abs(H_ref - H_pw))
    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Time: ref={t_ref:.3f}s, pairwise={t_pw:.3f}s")

    assert max_diff < 1e-10, f"Matrix mismatch: {max_diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
