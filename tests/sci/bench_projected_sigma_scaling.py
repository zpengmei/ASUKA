"""Benchmark exact selected-space sigma backends on fixed nested CAS windows.

This script is the replacement scorecard for the old dense-H focus:
it measures exact projected apply and Davidson solve costs for
selected-space backends on the same fixed selected windows.

Backends compared:
  - ExactSelectedPairwiseSigmaProjectedHop
  - ExactSelectedSymRowGraphProjectedHop via direct pairwise sparse-graph build
  - ExactSelectedSymRowGraphProjectedHop via exact pairwise dense->graph compression
  - ExactSelectedTupleProjectedHop
  - ExactSelectedSymRowGraphProjectedHop

The pairwise backend is fully matrix-free and uses the same diagonal-guess
preconditioner that the GPU projected solver uses for large selected spaces.
The tuple/graph backends are built from the dense exact-selected tuple emitter,
so the common "selected operator emission" cost is reported separately from
their backend-specific hop materialization cost.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np


def _skip_no_gpu(backend_mode: str) -> None:
    try:
        import cupy as cp  # noqa: F401

        from asuka.cuda.cuda_backend import (
            has_cas36_exact_selected_emit_tuples_dense_u64_device,
            has_pairwise_sigma_bucketed_u64_device,
            has_cas36_sym_row_graph_spmm_device,
        )

        if backend_mode in ("pairwise", "pairwise_graph", "pairwise_graph_dense"):
            if not bool(has_pairwise_sigma_bucketed_u64_device()):
                print("SKIP: bucketed pairwise sigma kernel is not available")
                sys.exit(0)
        else:
            if not bool(has_cas36_exact_selected_emit_tuples_dense_u64_device()):
                print("SKIP: exact selected dense tuple emitter is not available")
                sys.exit(0)
            if not bool(has_cas36_sym_row_graph_spmm_device()):
                print("SKIP: symmetric row-graph SpMM kernel is not available")
                sys.exit(0)
    except Exception:
        print("SKIP: CuPy or CUDA extension is not available")
        sys.exit(0)


def _next_pow2(n: int) -> int:
    n = max(1, int(n))
    return 1 << (int(n - 1).bit_length())


def _make_random_integrals(norb: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h1e = rng.standard_normal((norb, norb))
    h1e = 0.5 * (h1e + h1e.T)
    eri4 = rng.standard_normal((norb, norb, norb, norb)) * 0.1
    eri4 = eri4 + eri4.transpose(1, 0, 2, 3)
    eri4 = eri4 + eri4.transpose(0, 1, 3, 2)
    eri4 = eri4 + eri4.transpose(2, 3, 0, 1)
    eri4 = eri4 / 8.0
    return h1e, eri4


def _emit_exact_selected_tuples_device_dense(
    *,
    drt,
    drt_dev,
    sel_idx: np.ndarray,
    h_base_d,
    eri4_d,
    cp,
    threads: int = 256,
):
    from asuka.cuda.cuda_backend import (
        build_selected_membership_hash,
        cas36_exact_selected_emit_tuples_dense_u64_inplace_device,
    )

    sel_idx = np.asarray(sel_idx, dtype=np.int64).ravel()
    nsel = int(sel_idx.size)
    sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
    sel_sorted_d = cp.ascontiguousarray(cp.sort(sel_u64_d.copy()).ravel())
    membership_hash_keys_d, membership_hash_cap = build_selected_membership_hash(sel_sorted_d, cp)
    c_bound_d = cp.ones((nsel,), dtype=cp.float64)

    # This emitter is selected-only and sparse; start from a moderate cap and retry.
    cap = _next_pow2(max(1 << 20, 256 * max(1, nsel)))
    retries = 0
    while True:
        out_keys = cp.empty((cap,), dtype=cp.uint64)
        out_src = cp.empty((cap,), dtype=cp.int32)
        out_hij = cp.empty((cap,), dtype=cp.float64)
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
            membership_hash_keys=membership_hash_keys_d,
            membership_hash_cap=membership_hash_cap,
            out_diag=out_diag,
            out_n=out_n,
            overflow=overflow,
            threads=int(threads),
            sync=False,
        )
        cp.cuda.runtime.deviceSynchronize()
        overflow_h = int(overflow.item())
        nnz_h = int(out_n.item())
        if overflow_h == 0 and nnz_h <= cap:
            return (
                cp.ascontiguousarray(sel_u64_d),
                cp.ascontiguousarray(out_keys[:nnz_h].ravel()),
                cp.ascontiguousarray(out_src[:nnz_h].ravel()),
                cp.ascontiguousarray(out_hij[:nnz_h].ravel()),
                cp.ascontiguousarray(out_diag.ravel()),
                nnz_h,
            )
        retries += 1
        if retries > 6:
            raise RuntimeError(f"dense exact selected tuple emitter overflowed after {retries} retries (cap={cap})")
        cap = _next_pow2(max(cap * 2, max(nnz_h * 2, cap + 1)))


def _time_hop(hop, x_d, cp, *, warmup: int, repeat: int) -> float:
    for _ in range(int(warmup)):
        _ = hop.hop_gpu(x_d)
        cp.cuda.runtime.deviceSynchronize()
    times: list[float] = []
    for _ in range(int(repeat)):
        t0 = time.perf_counter()
        _ = hop.hop_gpu(x_d)
        cp.cuda.runtime.deviceSynchronize()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times))


def _time_eigensolver(
    hop,
    hdiag_h: np.ndarray,
    x0_h: list[np.ndarray],
    *,
    nroots: int,
    solver: str,
) -> tuple[float, np.ndarray]:
    from asuka.cuda.cuda_davidson import davidson_sym_gpu, jacobi_davidson_sym_gpu

    t0 = time.perf_counter()
    if str(solver).strip().lower() == "jd":
        precond = hop.build_jd_preconditioner(block_size=64, denom_tol=1e-8) if hasattr(hop, "build_jd_preconditioner") else None
        res = jacobi_davidson_sym_gpu(
            lambda v_d: hop.hop_gpu(v_d),
            x0=x0_h,
            hdiag=np.asarray(hdiag_h, dtype=np.float64, order="C"),
            precond=precond,
            nroots=int(nroots),
            max_cycle=24,
            max_space=max(8, int(nroots) + 4),
            tol=1e-8,
            subspace_eigh_cpu=False,
            batch_convergence_transfer=True,
            jd_inner_max_cycle=1,
            jd_inner_tol_rel=0.25,
            jd_keep_corrections=4,
        )
    else:
        res = davidson_sym_gpu(
            lambda v_d: hop.hop_gpu(v_d),
            x0=x0_h,
            hdiag=np.asarray(hdiag_h, dtype=np.float64, order="C"),
            nroots=int(nroots),
            max_cycle=24,
            max_space=max(8, int(nroots) + 4),
            tol=1e-8,
            subspace_eigh_cpu=False,
            batch_convergence_transfer=True,
        )
    return float(time.perf_counter() - t0), np.asarray(res.e, dtype=np.float64)


def bench_case(
    *,
    norb: int,
    nelec: int,
    twos: int,
    nsels: list[int],
    nvec: int,
    warmup: int,
    repeat: int,
    backend_mode: str,
    with_davidson: bool,
    solver: str,
) -> None:
    import cupy as cp

    from asuka.cuguga.drt import build_drt
    from asuka.cuda.cuda_backend import make_device_drt
    from asuka.sci.projected_apply import (
        ExactSelectedPairwiseSigmaProjectedHop,
        ExactSelectedSymRowGraphProjectedHop,
        ExactSelectedTupleProjectedHop,
    )

    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    drt_dev = make_device_drt(drt)
    rng = np.random.default_rng(42)
    h1e, eri4 = _make_random_integrals(norb, rng)
    h_base = np.asarray(
        np.asarray(h1e, dtype=np.float64, order="C") - 0.5 * np.einsum("pqqs->ps", eri4, optimize=True),
        dtype=np.float64,
        order="C",
    )
    h_base_d = cp.ascontiguousarray(cp.asarray(h_base.ravel(), dtype=cp.float64))
    eri4_d = cp.ascontiguousarray(cp.asarray(np.asarray(eri4, dtype=np.float64, order="C").ravel(), dtype=cp.float64))

    max_nsel = int(max(nsels))
    start = max(0, int(drt.ncsf) // 11 - max_nsel // 3)
    base_sel = np.arange(start, start + max_nsel, dtype=np.int64)

    print("=" * 78, flush=True)
    print(f"Projected Sigma Scaling: CAS({nelec},{norb}) S={twos/2} start={start}", flush=True)
    print("=" * 78, flush=True)
    print("All subsets are nested prefixes of the same contiguous selected window.", flush=True)

    for nsel in nsels:
        sel_idx = np.asarray(base_sel[: int(nsel)], dtype=np.int64, order="C")
        print(f"\nnsel={int(nsel)}/{int(drt.ncsf)}", flush=True)

        if backend_mode == "pairwise":
            from asuka.cuda.cuda_backend import (
                pairwise_build_bucket_data,
                pairwise_materialize_u64_device,
                pairwise_sigma_bucketed_u64_device,
            )
            sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
            t0 = time.perf_counter()
            materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, int(nsel), cp, sync=True)
            materialize_s = float(time.perf_counter() - t0)
            steps_all, nodes_all, occ_all, b_all = materialized

            t0 = time.perf_counter()
            bucket_data_full = pairwise_build_bucket_data(occ_all, int(norb), cp)
            bucket_s = float(time.perf_counter() - t0)
            sort_perm_d = cp.ascontiguousarray(bucket_data_full["sort_perm"].astype(cp.int32, copy=False))
            inv_perm_d = cp.ascontiguousarray(bucket_data_full["inv_perm"].astype(cp.int32, copy=False))
            materialized_sorted = (
                cp.ascontiguousarray(steps_all[sort_perm_d]),
                cp.ascontiguousarray(nodes_all[sort_perm_d]),
                cp.ascontiguousarray(occ_all[sort_perm_d]),
                cp.ascontiguousarray(b_all[sort_perm_d]),
            )
            sel_sorted_d = cp.ascontiguousarray(sel_u64_d[sort_perm_d])
            bucket_data = {
                "occ_keys_sorted": cp.ascontiguousarray(bucket_data_full["occ_keys_sorted"]),
                "bucket_keys": cp.ascontiguousarray(bucket_data_full["bucket_keys"]),
                "csf_to_bucket": cp.ascontiguousarray(bucket_data_full["csf_to_bucket"]),
                "bucket_starts": cp.ascontiguousarray(bucket_data_full["bucket_starts"]),
                "bucket_sizes": cp.ascontiguousarray(bucket_data_full["bucket_sizes"]),
                "neighbor_offsets": cp.ascontiguousarray(bucket_data_full["neighbor_offsets"]),
                "neighbor_list": cp.ascontiguousarray(bucket_data_full["neighbor_list"]),
                "target_offsets": cp.ascontiguousarray(bucket_data_full["target_offsets"]),
                "target_list": cp.ascontiguousarray(bucket_data_full["target_list"]),
                "target_offsets_1b": cp.ascontiguousarray(bucket_data_full["target_offsets_1b"]),
                "target_list_1b": cp.ascontiguousarray(bucket_data_full["target_list_1b"]),
            }
            del materialized, steps_all, nodes_all, occ_all, b_all, bucket_data_full, sel_u64_d
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            pairwise_hop = None

            x_h = np.asarray(
                np.random.default_rng(1000 + int(nsel)).standard_normal((int(nsel), int(nvec))),
                dtype=np.float64,
                order="C",
            )
            x_d = cp.ascontiguousarray(cp.asarray(x_h, dtype=cp.float64))
            x_sorted_d = cp.ascontiguousarray(x_d[sort_perm_d])

            def _pairwise_sigma():
                return pairwise_sigma_bucketed_u64_device(
                    drt,
                    drt_dev,
                    sel_sorted_d,
                    int(nsel),
                    h_base_d,
                    eri4_d,
                    materialized_sorted,
                    bucket_data,
                    x_sorted_d,
                    cp,
                    sync=False,
                )

            for _ in range(int(warmup)):
                _ = _pairwise_sigma()
                cp.cuda.runtime.deviceSynchronize()
            times = []
            y_sorted = None
            for _ in range(int(repeat)):
                t0 = time.perf_counter()
                y_sorted = _pairwise_sigma()
                cp.cuda.runtime.deviceSynchronize()
                times.append(time.perf_counter() - t0)
            sigma_s = float(np.mean(times))

            print(f"  Materialize build:    {materialize_s:.3f}s", flush=True)
            print(f"  Bucket build:         {bucket_s:.3f}s", flush=True)
            print(f"  Pairwise sigma avg:   {sigma_s:.6f}s", flush=True)
            print(f"  Total setup:          {materialize_s + bucket_s:.3f}s", flush=True)

            if with_davidson:
                from asuka.cuda.cuda_backend import pairwise_diag_bucketed_u64_device

                pairwise_hop = ExactSelectedPairwiseSigmaProjectedHop(
                    sel_idx=np.asarray(sel_idx, dtype=np.int64, order="C"),
                    drt=drt,
                    drt_dev=drt_dev,
                    sel_sorted_u64_d=sel_sorted_d,
                    sort_perm_d=sort_perm_d,
                    inv_perm_d=inv_perm_d,
                    h_base_d=h_base_d,
                    eri4_d=eri4_d,
                    materialized_sorted=materialized_sorted,
                    bucket_data=bucket_data,
                    hdiag_d=cp.ascontiguousarray(
                        pairwise_diag_bucketed_u64_device(
                            drt,
                            drt_dev,
                            sel_sorted_d,
                            int(nsel),
                            h_base_d,
                            eri4_d,
                            materialized_sorted,
                            bucket_data,
                            cp,
                            sync=True,
                        )[inv_perm_d]
                    ),
                )
                hdiag_h = np.asarray(cp.asnumpy(pairwise_hop.hdiag_d), dtype=np.float64, order="C")
                pairwise_hop.hdiag_d = None
                x0_h = [np.eye(int(nsel), 1, dtype=np.float64)[:, 0]]
                pairwise_davidson_s, _ = _time_eigensolver(pairwise_hop, hdiag_h, x0_h, nroots=1, solver=str(solver))
                print(f"  Pairwise {str(solver).upper()}:       {pairwise_davidson_s:.3f}s", flush=True)
                print("  Pairwise hdiag:       exact bucketed diagonal", flush=True)

            del pairwise_hop, y_sorted, x_sorted_d, x_d, sel_sorted_d, sort_perm_d, inv_perm_d, materialized_sorted
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            continue

        if backend_mode == "pairwise_graph":
            t0 = time.perf_counter()
            graph_hop = ExactSelectedSymRowGraphProjectedHop.from_pairwise_selected_space(
                drt=drt,
                drt_dev=drt_dev,
                sel_idx=sel_idx,
                h_base_d=h_base_d,
                eri4_d=eri4_d,
                cp=cp,
            )
            graph_build_s = float(time.perf_counter() - t0)
            graph_edges = int(graph_hop.hij_d.size)
            density = float(2.0 * graph_edges / max(1, int(nsel) * max(int(nsel) - 1, 1)))
            x_h = np.asarray(
                np.random.default_rng(1000 + int(nsel)).standard_normal((int(nsel), int(nvec))),
                dtype=np.float64,
                order="C",
            )
            x_d = cp.ascontiguousarray(cp.asarray(x_h, dtype=cp.float64))
            hdiag_h = np.asarray(cp.asnumpy(graph_hop.hdiag_d), dtype=np.float64, order="C")
            graph_apply_s = _time_hop(graph_hop, x_d, cp, warmup=warmup, repeat=repeat)
            print(f"  Graph hop build:      {graph_build_s:.3f}s, edges={graph_edges}, density={density:.6f}", flush=True)
            print(f"  Graph sigma avg:      {graph_apply_s:.6f}s", flush=True)
            if with_davidson:
                x0_h = [np.eye(int(nsel), 1, dtype=np.float64)[:, 0]]
                graph_davidson_s, _e_graph = _time_eigensolver(graph_hop, hdiag_h, x0_h, nroots=1, solver=str(solver))
                print(f"  Graph {str(solver).upper()}:          {graph_davidson_s:.3f}s", flush=True)
            del graph_hop, x_d
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            continue

        if backend_mode == "pairwise_graph_dense":
            from asuka.cuda.cuda_backend import (
                pairwise_build_bucket_data,
                pairwise_hij_bucketed_u64_device,
                pairwise_materialize_u64_device,
            )

            sel_u64_d = cp.ascontiguousarray(cp.asarray(sel_idx.astype(np.uint64, copy=False), dtype=cp.uint64).ravel())
            t0 = time.perf_counter()
            materialized = pairwise_materialize_u64_device(drt, drt_dev, sel_u64_d, int(nsel), cp, sync=True)
            materialize_s = float(time.perf_counter() - t0)
            steps_all, nodes_all, occ_all, b_all = materialized
            t0 = time.perf_counter()
            bucket_data = pairwise_build_bucket_data(occ_all, int(norb), cp)
            bucket_s = float(time.perf_counter() - t0)
            sort_perm_d = cp.ascontiguousarray(bucket_data["sort_perm"].astype(cp.int64))
            inv_perm_d = cp.ascontiguousarray(bucket_data["inv_perm"].astype(cp.int64))
            materialized_sorted = (
                cp.ascontiguousarray(steps_all[sort_perm_d]),
                cp.ascontiguousarray(nodes_all[sort_perm_d]),
                cp.ascontiguousarray(occ_all[sort_perm_d]),
                cp.ascontiguousarray(b_all[sort_perm_d]),
            )
            sel_sorted_d = cp.ascontiguousarray(sel_u64_d[sort_perm_d])
            t0 = time.perf_counter()
            H_sorted_d, diag_sorted_d = pairwise_hij_bucketed_u64_device(
                drt,
                drt_dev,
                sel_sorted_d,
                int(nsel),
                h_base_d,
                eri4_d,
                materialized_sorted,
                bucket_data,
                cp,
                sync=True,
            )
            dense_build_s = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            H_d = cp.ascontiguousarray(H_sorted_d[inv_perm_d][:, inv_perm_d])
            cp.cuda.runtime.deviceSynchronize()
            dense_unpermute_s = float(time.perf_counter() - t0)
            t0 = time.perf_counter()
            graph_hop = ExactSelectedSymRowGraphProjectedHop.from_dense_matrix(
                sel_idx=sel_idx,
                H_d=H_d,
                hdiag=np.asarray(cp.asnumpy(diag_sorted_d[inv_perm_d]), dtype=np.float64, order="C"),
            )
            graph_build_s = float(time.perf_counter() - t0)
            graph_edges = int(graph_hop.hij_d.size)
            density = float(2.0 * graph_edges / max(1, int(nsel) * max(int(nsel) - 1, 1)))
            x_h = np.asarray(
                np.random.default_rng(1000 + int(nsel)).standard_normal((int(nsel), int(nvec))),
                dtype=np.float64,
                order="C",
            )
            x_d = cp.ascontiguousarray(cp.asarray(x_h, dtype=cp.float64))
            hdiag_h = np.asarray(cp.asnumpy(graph_hop.hdiag_d), dtype=np.float64, order="C")
            graph_apply_s = _time_hop(graph_hop, x_d, cp, warmup=warmup, repeat=repeat)
            print(f"  Materialize build:    {materialize_s:.3f}s", flush=True)
            print(f"  Bucket build:         {bucket_s:.3f}s", flush=True)
            print(f"  Dense pairwise build: {dense_build_s:.3f}s", flush=True)
            print(f"  Dense unpermute:      {dense_unpermute_s:.3f}s", flush=True)
            print(f"  Graph hop build:      {graph_build_s:.3f}s, edges={graph_edges}, density={density:.6f}", flush=True)
            print(f"  Graph sigma avg:      {graph_apply_s:.6f}s", flush=True)
            if with_davidson:
                x0_h = [np.eye(int(nsel), 1, dtype=np.float64)[:, 0]]
                graph_davidson_s, _e_graph = _time_eigensolver(graph_hop, hdiag_h, x0_h, nroots=1, solver=str(solver))
                print(f"  Graph {str(solver).upper()}:          {graph_davidson_s:.3f}s", flush=True)
            del graph_hop, x_d, H_d, H_sorted_d, diag_sorted_d, materialized_sorted, sel_sorted_d, sort_perm_d, inv_perm_d
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()
            continue

        t0 = time.perf_counter()
        _sel_u64_d, labels_d, src_d, hij_d, diag_d, nnz = _emit_exact_selected_tuples_device_dense(
            drt=drt,
            drt_dev=drt_dev,
            sel_idx=sel_idx,
            h_base_d=h_base_d,
            eri4_d=eri4_d,
            cp=cp,
        )
        emit_s = float(time.perf_counter() - t0)

        x_h = np.asarray(np.random.default_rng(1000 + int(nsel)).standard_normal((int(nsel), int(nvec))), dtype=np.float64, order="C")
        x_d = cp.ascontiguousarray(cp.asarray(x_h, dtype=cp.float64))
        hdiag_h = np.asarray(cp.asnumpy(diag_d), dtype=np.float64, order="C")
        print(f"  Exact tuple emission: {emit_s:.3f}s, nnz={nnz}", flush=True)

        want_tuple = backend_mode in ("tuple", "both")
        want_graph = backend_mode in ("graph", "both")
        tuple_apply_s = math.nan
        graph_apply_s = math.nan
        tuple_davidson_s = math.nan
        graph_davidson_s = math.nan
        tuple_edges = 0
        graph_edges = 0
        density = math.nan
        y_tuple = None
        e_tuple = None

        if want_tuple:
            t0 = time.perf_counter()
            tuple_hop = ExactSelectedTupleProjectedHop.from_tuples(
                sel_idx=sel_idx,
                labels=labels_d,
                src_pos=src_d,
                hij=hij_d,
                hdiag=hdiag_h,
            )
            tuple_build_s = float(time.perf_counter() - t0)
            tuple_edges = int(tuple_hop.coo_hij_d.size) if tuple_hop.coo_hij_d is not None else 0
            tuple_apply_s = _time_hop(tuple_hop, x_d, cp, warmup=warmup, repeat=repeat)
            y_tuple = np.asarray(cp.asnumpy(tuple_hop.hop_gpu(x_d)), dtype=np.float64, order="C")
            print(f"  Tuple hop build:      {tuple_build_s:.3f}s, edges={tuple_edges}", flush=True)
            print(f"  Tuple sigma avg:      {tuple_apply_s:.6f}s", flush=True)
            if with_davidson:
                x0_h = [np.eye(int(nsel), 1, dtype=np.float64)[:, 0]]
                tuple_davidson_s, e_tuple = _time_eigensolver(tuple_hop, hdiag_h, x0_h, nroots=1, solver=str(solver))
                print(f"  Tuple {str(solver).upper()}:          {tuple_davidson_s:.3f}s", flush=True)
            del tuple_hop
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()

        if want_graph:
            t0 = time.perf_counter()
            graph_hop = ExactSelectedSymRowGraphProjectedHop.from_tuples(
                sel_idx=sel_idx,
                labels=labels_d,
                src_pos=src_d,
                hij=hij_d,
                hdiag=hdiag_h,
            )
            graph_build_s = float(time.perf_counter() - t0)
            graph_edges = int(graph_hop.hij_d.size)
            density = float(2.0 * graph_edges / max(1, int(nsel) * max(int(nsel) - 1, 1)))
            graph_apply_s = _time_hop(graph_hop, x_d, cp, warmup=warmup, repeat=repeat)
            y_graph = np.asarray(cp.asnumpy(graph_hop.hop_gpu(x_d)), dtype=np.float64, order="C")
            print(f"  Graph hop build:      {graph_build_s:.3f}s, edges={graph_edges}, density={density:.6f}", flush=True)
            print(f"  Graph sigma avg:      {graph_apply_s:.6f}s", flush=True)
            if y_tuple is not None:
                apply_maxdiff = float(np.max(np.abs(y_tuple - y_graph))) if y_graph.size else 0.0
                print(f"  Apply max diff:       {apply_maxdiff:.3e}", flush=True)
                print(f"  Sigma speedup:        {tuple_apply_s / max(graph_apply_s, 1e-30):.2f}x", flush=True)
            if with_davidson:
                x0_h = [np.eye(int(nsel), 1, dtype=np.float64)[:, 0]]
                graph_davidson_s, e_graph = _time_eigensolver(graph_hop, hdiag_h, x0_h, nroots=1, solver=str(solver))
                print(f"  Graph {str(solver).upper()}:          {graph_davidson_s:.3f}s", flush=True)
                if e_tuple is not None:
                    eig_maxdiff = float(np.max(np.abs(e_tuple - e_graph))) if e_graph.size else 0.0
                    print(f"  Davidson speedup:     {tuple_davidson_s / max(graph_davidson_s, 1e-30):.2f}x", flush=True)
                    print(f"  Eig max diff:         {eig_maxdiff:.3e}", flush=True)
            del graph_hop, y_graph
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.runtime.deviceSynchronize()

        # Reduce peak memory between sizes.
        del x_d, labels_d, src_d, hij_d, diag_d, _sel_u64_d, y_tuple
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.runtime.deviceSynchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--norb", type=int, default=18)
    ap.add_argument("--nelec", type=int, default=18)
    ap.add_argument("--twos", type=int, default=0)
    ap.add_argument("--nsels", default="5000")
    ap.add_argument("--nvec", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--backend-mode", choices=("pairwise", "pairwise_graph", "pairwise_graph_dense", "tuple", "graph", "both"), default="pairwise")
    ap.add_argument("--with-davidson", action="store_true")
    ap.add_argument("--solver", choices=("davidson", "jd"), default="davidson")
    args = ap.parse_args()
    _skip_no_gpu(str(args.backend_mode))

    nsels = [int(x) for x in str(args.nsels).split(",") if str(x).strip()]
    if not nsels:
        raise ValueError("nsels must not be empty")
    bench_case(
        norb=int(args.norb),
        nelec=int(args.nelec),
        twos=int(args.twos),
        nsels=nsels,
        nvec=int(args.nvec),
        warmup=int(args.warmup),
        repeat=int(args.repeat),
        backend_mode=str(args.backend_mode),
        with_davidson=bool(args.with_davidson),
        solver=str(args.solver),
    )


if __name__ == "__main__":
    main()
