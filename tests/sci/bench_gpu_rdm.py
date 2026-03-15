#!/usr/bin/env python
"""Benchmark GPU vs CPU RDM computation for GUGASCISolver.

Uses synthetic selected-space data (random sel_idx + ci coefficients) so
there is no SCF/SCI overhead — pure RDM kernel timing only.

Tested system: CAS(18,18), norb=18, nelec=18, twos=0.
ncsf = 449M (requires key64 label space).
"""
from __future__ import annotations

import time
import numpy as np


def _sync(cp):
    cp.cuda.runtime.deviceSynchronize()


def _time_fn(fn, *, nreps):
    fn()  # warmup
    _sync = getattr(fn, "_sync", None)
    t0 = time.perf_counter()
    for _ in range(nreps):
        fn()
    return (time.perf_counter() - t0) / nreps


def main():
    import cupy as cp
    from asuka.cuguga.drt import build_drt
    from asuka.cuda.cuda_backend import make_device_drt
    from asuka.sci.gpu_rdm import make_rdm12_gpu
    from asuka.sci.sparse_rdm import make_rdm12_selected

    norb, nelec, twos = 18, 18, 0
    print(f"Building DRT: norb={norb}, nelec={nelec}, twos={twos} ...")
    drt = build_drt(norb=norb, nelec=nelec, twos_target=twos)
    ncsf = int(drt.ncsf)
    print(f"  ncsf = {ncsf:,}")
    drt_dev = make_device_drt(drt)

    rng = np.random.default_rng(42)

    nsel_targets = [256, 512, 1024, 2048, 4096, 8192]

    print(f"\n{'nsel':>6}  {'GPU (ms)':>10}  {'CPU (ms)':>10}  {'speedup':>8}  {'dm1_err':>12}")
    print("-" * 58)

    for nsel in nsel_targets:
        # Random synthetic selected space (valid CSF indices in [0, ncsf))
        sel_idx = np.sort(rng.choice(ncsf, size=nsel, replace=False)).astype(np.int64)
        ci_raw  = rng.standard_normal(nsel)
        ci_1d   = (ci_raw / np.linalg.norm(ci_raw)).astype(np.float64)

        nreps_gpu = max(5, 40 // max(1, nsel // 512))
        nreps_cpu = max(2,  8 // max(1, nsel // 512))

        # GPU timing
        cp.cuda.runtime.deviceSynchronize()
        _sync(cp)
        make_rdm12_gpu(drt, drt_dev, sel_idx, ci_1d, cp)  # warmup
        _sync(cp)
        t0 = time.perf_counter()
        for _ in range(nreps_gpu):
            make_rdm12_gpu(drt, drt_dev, sel_idx, ci_1d, cp)
        _sync(cp)
        t_gpu = (time.perf_counter() - t0) / nreps_gpu

        # CPU timing
        make_rdm12_selected(drt, sel_idx, ci_1d)  # warmup
        t0 = time.perf_counter()
        for _ in range(nreps_cpu):
            make_rdm12_selected(drt, sel_idx, ci_1d)
        t_cpu = (time.perf_counter() - t0) / nreps_cpu

        # correctness
        dm1_gpu = cp.asnumpy(make_rdm12_gpu(drt, drt_dev, sel_idx, ci_1d, cp)[0])
        dm1_cpu, _ = make_rdm12_selected(drt, sel_idx, ci_1d)
        err = float(np.max(np.abs(dm1_gpu - dm1_cpu)))

        speedup = t_cpu / t_gpu
        print(f"{nsel:>6}  {t_gpu*1e3:>10.2f}  {t_cpu*1e3:>10.1f}  {speedup:>8.1f}x  {err:>12.2e}")

    print()


if __name__ == "__main__":
    main()
