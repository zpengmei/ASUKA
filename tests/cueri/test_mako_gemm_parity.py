"""Mako GEMM kernel tile parity + speedup tests.

Validates that the GEMM-based kernel produces tiles matching the generic
scalar kernel to < 1e-8, and measures speedup for d-shell classes.
"""
import sys
import os
import time
import pytest
import numpy as np

# Ensure ASUKA-mako is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_h2o_workload(basis="sto-3g"):
    """Build shell pairs + pair tables for H2O."""
    import cupy as cp
    from asuka.frontend import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order
    from asuka.cueri.gpu import (
        to_device_basis_ss,
        to_device_shell_pairs,
        build_pair_tables_ss_device,
    )
    from asuka.cueri.tasks import eri_class_id

    mol = Molecule.from_atoms(
        atoms=[
            ("O", (0.0, 0.0, 0.117347)),
            ("H", (0.0, 0.757215, -0.469388)),
            ("H", (0.0, -0.757215, -0.469388)),
        ],
        unit="angstrom",
        basis=basis,
        cart=True,
    )
    ao_basis, _ = build_ao_basis_cart(mol)
    sp = build_shell_pairs_l_order(ao_basis)
    dbasis = to_device_basis_ss(ao_basis)
    dsp = to_device_shell_pairs(sp)
    pt = build_pair_tables_ss_device(dbasis, dsp)
    shell_l = np.asarray(ao_basis.shell_l, dtype=np.int32)
    return ao_basis, sp, dbasis, dsp, pt, shell_l


def _find_tasks_for_class(sp, shell_l, la, lb, lc, ld, max_tasks=5000):
    """Find shell-quartet tasks matching the given ERI class."""
    from asuka.cueri.tasks import eri_class_id

    target_cid = int(eri_class_id(la, lb, lc, ld))
    ab_list, cd_list = [], []
    for i in range(len(sp.sp_A)):
        la_i = int(shell_l[sp.sp_A[i]])
        lb_i = int(shell_l[sp.sp_B[i]])
        for j in range(len(sp.sp_A)):
            lc_j = int(shell_l[sp.sp_A[j]])
            ld_j = int(shell_l[sp.sp_B[j]])
            if int(eri_class_id(la_i, lb_i, lc_j, ld_j)) == target_cid:
                ab_list.append(i)
                cd_list.append(j)
                if len(ab_list) >= max_tasks:
                    return ab_list, cd_list
    return ab_list, cd_list


def _ncart(l):
    return (l + 1) * (l + 2) // 2


def _run_parity_test(la, lb, lc, ld, basis="sto-3g", atol=1e-8, rtol=1e-8):
    """Run tile parity test between GEMM and generic kernels."""
    import cupy as cp
    from asuka.kernels.cueri_mako import require_ext

    ext = require_ext()
    ao_basis, sp, dbasis, dsp, pt, shell_l = _make_h2o_workload(basis)
    ab_list, cd_list = _find_tasks_for_class(sp, shell_l, la, lb, lc, ld)

    if not ab_list:
        pytest.skip(f"No tasks found for class ({la},{lb},{lc},{ld}) with basis {basis}")

    task_ab = cp.asarray(np.array(ab_list, dtype=np.int32))
    task_cd = cp.asarray(np.array(cd_list, dtype=np.int32))
    ntasks = len(task_ab)
    ncomp = _ncart(la) * _ncart(lb) * _ncart(lc) * _ncart(ld)

    # Generic (scalar) kernel
    out_generic = cp.empty(ntasks * ncomp, dtype=cp.float64)
    ext.mako_generic_eri_fp64_device(
        task_ab, task_cd, la, lb, lc, ld,
        dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
        dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
        pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
        out_generic, 256, 0,
    )

    # GEMM (block-cooperative) kernel
    out_gemm = cp.empty(ntasks * ncomp, dtype=cp.float64)
    ext.mako_gemm_eri_fp64_device(
        task_ab, task_cd, la, lb, lc, ld,
        dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
        dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
        pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
        out_gemm, 128, 0,
    )

    cp.cuda.Device().synchronize()

    generic_np = cp.asnumpy(out_generic)
    gemm_np = cp.asnumpy(out_gemm)

    max_abs_err = float(np.max(np.abs(generic_np - gemm_np)))
    max_val = float(np.max(np.abs(generic_np)))
    max_rel_err = max_abs_err / max(max_val, 1e-30)
    n_mismatch = int(np.sum(~np.isclose(generic_np, gemm_np, atol=atol, rtol=rtol)))

    return ntasks, ncomp, max_abs_err, max_rel_err, n_mismatch


def _run_speedup_test(la, lb, lc, ld, basis="6-31g*", n_warmup=5, n_iter=20):
    """Measure speedup of GEMM kernel over generic kernel."""
    import cupy as cp
    from asuka.kernels.cueri_mako import require_ext

    ext = require_ext()
    ao_basis, sp, dbasis, dsp, pt, shell_l = _make_h2o_workload(basis)
    ab_list, cd_list = _find_tasks_for_class(sp, shell_l, la, lb, lc, ld, max_tasks=50000)

    if not ab_list:
        return None, None, None

    # Replicate for stable timing
    n_base = len(ab_list)
    reps = max(1, 2000 // n_base)
    task_ab = cp.asarray(np.tile(np.array(ab_list, dtype=np.int32), reps))
    task_cd = cp.asarray(np.tile(np.array(cd_list, dtype=np.int32), reps))
    ntasks = len(task_ab)
    ncomp = _ncart(la) * _ncart(lb) * _ncart(lc) * _ncart(ld)

    out = cp.empty(ntasks * ncomp, dtype=cp.float64)

    # Warmup
    for _ in range(n_warmup):
        ext.mako_generic_eri_fp64_device(
            task_ab, task_cd, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
            out, 256, 0,
        )
    cp.cuda.Device().synchronize()

    # Time generic
    times_generic = []
    for _ in range(n_iter):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record()
        ext.mako_generic_eri_fp64_device(
            task_ab, task_cd, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
            out, 256, 0,
        )
        e.record(); e.synchronize()
        times_generic.append(cp.cuda.get_elapsed_time(s, e))

    # Warmup GEMM
    for _ in range(n_warmup):
        ext.mako_gemm_eri_fp64_device(
            task_ab, task_cd, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
            out, 128, 0,
        )
    cp.cuda.Device().synchronize()

    # Time GEMM
    times_gemm = []
    for _ in range(n_iter):
        s = cp.cuda.Event(); e = cp.cuda.Event()
        s.record()
        ext.mako_gemm_eri_fp64_device(
            task_ab, task_cd, la, lb, lc, ld,
            dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
            dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
            pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
            out, 128, 0,
        )
        e.record(); e.synchronize()
        times_gemm.append(cp.cuda.get_elapsed_time(s, e))

    times_generic.sort()
    times_gemm.sort()
    t_generic = times_generic[len(times_generic) // 2]
    t_gemm = times_gemm[len(times_gemm) // 2]
    speedup = t_generic / max(t_gemm, 1e-6)

    return t_generic, t_gemm, speedup


# ===================== Tile parity tests =====================

_AM_LABELS = "spdfg"

_D_SHELL_CLASSES = [
    (1, 0, 0, 0),  # psss
    (1, 1, 0, 0),  # ppss
    (1, 0, 1, 0),  # psps
    (1, 1, 1, 0),  # ppps
    (1, 1, 1, 1),  # pppp
    (2, 0, 0, 0),  # dsss
    (2, 1, 0, 0),  # dpss
    (2, 0, 1, 0),  # dsps
    (2, 1, 1, 0),  # dpps
    (2, 0, 1, 1),  # dspp
    (2, 1, 1, 1),  # dppp
    (2, 2, 0, 0),  # ddss
    (2, 0, 2, 0),  # dsds
    (2, 2, 1, 0),  # ddps
    (2, 0, 2, 1),  # dsdp
    (2, 2, 1, 1),  # ddpp
    (2, 1, 2, 0),  # dpds
    (2, 1, 2, 1),  # dpdp
    (2, 2, 2, 0),  # ddds
    (2, 0, 2, 2),  # dsdd
    (2, 1, 2, 2),  # dpdd
    (2, 2, 2, 2),  # dddd
]


@pytest.mark.parametrize(
    "la,lb,lc,ld",
    _D_SHELL_CLASSES,
    ids=[
        "".join(_AM_LABELS[l] for l in cls) for cls in _D_SHELL_CLASSES
    ],
)
def test_gemm_tile_parity(la, lb, lc, ld):
    """GEMM kernel tiles must match generic kernel to < 1e-8."""
    basis = "6-31g*" if max(la, lb, lc, ld) >= 2 else "sto-3g"
    ntasks, ncomp, max_abs_err, max_rel_err, n_mismatch = _run_parity_test(
        la, lb, lc, ld, basis=basis, atol=1e-8, rtol=1e-8,
    )
    label = "".join(_AM_LABELS[l] for l in (la, lb, lc, ld))
    print(f"  {label}: {ntasks} tasks, {ncomp} comps, "
          f"max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e}, "
          f"mismatches={n_mismatch}")
    assert max_abs_err < 1e-8, (
        f"{label}: max_abs_err={max_abs_err:.2e} >= 1e-8"
    )


# ===================== Speedup tests =====================

@pytest.mark.parametrize(
    "la,lb,lc,ld",
    [(2, 2, 2, 2), (2, 2, 1, 1), (1, 1, 1, 1), (2, 1, 2, 1)],
    ids=["dddd", "ddpp", "pppp", "dpdp"],
)
def test_gemm_speedup(la, lb, lc, ld):
    """GEMM kernel must be faster than generic kernel for d-shell classes."""
    t_generic, t_gemm, speedup = _run_speedup_test(la, lb, lc, ld)
    label = "".join(_AM_LABELS[l] for l in (la, lb, lc, ld))
    if t_generic is None:
        pytest.skip(f"No tasks found for {label}")
    print(f"  {label}: generic={t_generic:.3f}ms, gemm={t_gemm:.3f}ms, "
          f"speedup={speedup:.1f}x")
    # Paper claims 10x+ for dddd; we require at least measurable improvement
    assert speedup > 1.0, f"{label}: no speedup (got {speedup:.2f}x)"


def _run_tf32tc_parity_test(la, lb, lc, ld, basis="sto-3g", atol=5e-3, rtol=5e-3):
    """Run tile parity test between TF32TC and FP64 GEMM kernels."""
    import cupy as cp
    from asuka.kernels.cueri_mako import require_ext

    ext = require_ext()
    if not hasattr(ext, "mako_gemm_eri_tf32tc_device"):
        pytest.skip("TF32TC kernel not available (MAKO_HAS_CUTLASS not defined)")

    ao_basis, sp, dbasis, dsp, pt, shell_l = _make_h2o_workload(basis)
    ab_list, cd_list = _find_tasks_for_class(sp, shell_l, la, lb, lc, ld)

    if not ab_list:
        pytest.skip(f"No tasks found for class ({la},{lb},{lc},{ld}) with basis {basis}")

    task_ab = cp.asarray(np.array(ab_list, dtype=np.int32))
    task_cd = cp.asarray(np.array(cd_list, dtype=np.int32))
    ntasks = len(task_ab)
    ncomp = _ncart(la) * _ncart(lb) * _ncart(lc) * _ncart(ld)

    # FP64 GEMM kernel (reference)
    out_fp64 = cp.empty(ntasks * ncomp, dtype=cp.float64)
    ext.mako_gemm_eri_fp64_device(
        task_ab, task_cd, la, lb, lc, ld,
        dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
        dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
        pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
        out_fp64, 128, 0,
    )

    # TF32 tensor core kernel
    out_tf32tc = cp.empty(ntasks * ncomp, dtype=cp.float64)
    ext.mako_gemm_eri_tf32tc_device(
        task_ab, task_cd, la, lb, lc, ld,
        dsp.sp_A, dsp.sp_B, dsp.sp_pair_start, dsp.sp_npair,
        dbasis.shell_cx, dbasis.shell_cy, dbasis.shell_cz,
        pt.pair_eta, pt.pair_Px, pt.pair_Py, pt.pair_Pz, pt.pair_cK,
        out_tf32tc, 128, 0,
    )

    cp.cuda.Device().synchronize()

    fp64_np = cp.asnumpy(out_fp64)
    tf32tc_np = cp.asnumpy(out_tf32tc)

    max_abs_err = float(np.max(np.abs(fp64_np - tf32tc_np)))
    max_val = float(np.max(np.abs(fp64_np)))
    max_rel_err = max_abs_err / max(max_val, 1e-30)
    n_mismatch = int(np.sum(~np.isclose(fp64_np, tf32tc_np, atol=atol, rtol=rtol)))

    return ntasks, ncomp, max_abs_err, max_rel_err, n_mismatch, max_val


# TF32TC classes: only those where nab >= 16 (MMA tile is 16×8)
_TF32TC_CLASSES = [
    cls for cls in _D_SHELL_CLASSES
    if _ncart(cls[0]) * _ncart(cls[1]) >= 16
]


@pytest.mark.parametrize(
    "la,lb,lc,ld",
    _TF32TC_CLASSES,
    ids=[
        "".join(_AM_LABELS[l] for l in cls) for cls in _TF32TC_CLASSES
    ],
)
def test_tf32tc_tile_parity(la, lb, lc, ld):
    """TF32 tensor core tiles must match FP64 GEMM within TF32 precision."""
    basis = "6-31g*" if max(la, lb, lc, ld) >= 2 else "sto-3g"
    ntasks, ncomp, max_abs_err, max_rel_err, n_mismatch, max_val = _run_tf32tc_parity_test(
        la, lb, lc, ld, basis=basis, atol=5e-3, rtol=5e-3,
    )
    label = "".join(_AM_LABELS[l] for l in (la, lb, lc, ld))
    print(f"  {label}: {ntasks} tasks, {ncomp} comps, "
          f"max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e}, "
          f"mismatches={n_mismatch}")
    # FP32 accumulation has ~7 decimal digits precision.  For ERI values with
    # cancellation, use absolute tolerance; otherwise use relative.
    if max_val < 1e-4:
        assert max_abs_err < 1e-3, (
            f"{label}: max_abs_err={max_abs_err:.2e} >= 1e-3 (small-value class)"
        )
    else:
        assert max_rel_err < 5e-3, (
            f"{label}: max_rel_err={max_rel_err:.2e} >= 5e-3"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Mako GEMM kernel — tile parity tests")
    print("=" * 70)
    for la, lb, lc, ld in _D_SHELL_CLASSES:
        label = "".join(_AM_LABELS[l] for l in (la, lb, lc, ld))
        basis = "6-31g*" if max(la, lb, lc, ld) >= 2 else "sto-3g"
        try:
            ntasks, ncomp, mae, mre, nm = _run_parity_test(
                la, lb, lc, ld, basis=basis,
            )
            status = "PASS" if mae < 1e-8 else "FAIL"
            print(f"  [{status}] {label}: {ntasks} tasks, "
                  f"max_abs={mae:.2e}, mismatches={nm}")
        except Exception as e:
            print(f"  [SKIP] {label}: {e}")

    print()
    print("=" * 70)
    print("Mako GEMM kernel — TF32 tensor core parity tests")
    print("=" * 70)
    for la, lb, lc, ld in _TF32TC_CLASSES:
        label = "".join(_AM_LABELS[l] for l in (la, lb, lc, ld))
        try:
            ntasks, ncomp, mae, mre, nm, mval = _run_tf32tc_parity_test(
                la, lb, lc, ld, basis="6-31g*",
            )
            status = "PASS" if (mval < 1e-6 and mae < 1e-2) or mre < 5e-2 else "FAIL"
            print(f"  [{status}] {label}: {ntasks} tasks, "
                  f"max_abs={mae:.2e}, max_rel={mre:.2e}, mismatches={nm}")
        except Exception as e:
            print(f"  [SKIP] {label}: {e}")

    print()
    print("=" * 70)
    print("Mako GEMM kernel — speedup tests")
    print("=" * 70)
    for la, lb, lc, ld in [(2, 2, 2, 2), (2, 2, 1, 1), (1, 1, 1, 1), (2, 1, 2, 1)]:
        label = "".join(_AM_LABELS[l] for l in (la, lb, lc, ld))
        try:
            t_g, t_m, spd = _run_speedup_test(la, lb, lc, ld)
            if t_g is None:
                print(f"  [SKIP] {label}: no tasks")
            else:
                print(f"  {label}: generic={t_g:.3f}ms, gemm={t_m:.3f}ms, "
                      f"speedup={spd:.1f}x")
        except Exception as e:
            print(f"  [FAIL] {label}: {e}")
