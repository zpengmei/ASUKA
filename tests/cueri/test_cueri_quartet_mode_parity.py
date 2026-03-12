"""Targeted ERI quartet mode parity coverage for handwritten and generated kernels."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

cupy = pytest.importorskip("cupy", reason="CuPy required for GPU tests")
pytestmark = pytest.mark.cuda

_MODE_CASES = [
    ("block", {}),
    ("warp", {}),
    ("multiblock", {"blocks_per_task": 4}),
    ("auto", {"work_small_max": 0, "work_large_min": 1, "blocks_per_task": 4}),
]


def _require_cuda_runtime():
    if int(cupy.cuda.runtime.getDeviceCount()) <= 0:
        pytest.skip("no CUDA device")
    try:
        from asuka.cueri.gpu import has_cuda_ext
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cuERI GPU module unavailable ({type(exc).__name__}: {exc})")
    if not bool(has_cuda_ext()):
        pytest.skip("cuERI CUDA extension unavailable")


def _build_o_631g_star_basis():
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart

    mol = Molecule.from_atoms(
        [("O", (0.0, 0.0, 0.0))],
        unit="Bohr",
        basis="6-31g*",
        cart=True,
    )
    basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    return basis


def _build_o_ccpvtz_basis():
    from asuka.frontend.molecule import Molecule
    from asuka.frontend.one_electron import build_ao_basis_cart

    mol = Molecule.from_atoms(
        [("O", (0.0, 0.0, 0.0))],
        unit="Bohr",
        basis="cc-pvtz",
        cart=True,
    )
    basis, _ = build_ao_basis_cart(mol, basis=mol.basis, expand_contractions=True)
    return basis


def _find_shell_pair_indices(shell_pairs, shell_l: np.ndarray, la: int, lb: int, *, limit: int) -> np.ndarray:
    hits: list[int] = []
    for idx, (A, B) in enumerate(zip(shell_pairs.sp_A, shell_pairs.sp_B)):
        if int(shell_l[int(A)]) == int(la) and int(shell_l[int(B)]) == int(lb):
            hits.append(int(idx))
            if len(hits) >= int(limit):
                break
    if len(hits) < int(limit):
        pytest.skip(f"basis does not provide {limit} shell pairs of class ({la},{lb})")
    return np.asarray(hits, dtype=np.int32)


def _find_shell_pair_with_npair(shell_pairs, shell_l: np.ndarray, la: int, lb: int, *, npair: int) -> np.int32:
    for idx, (A, B, n) in enumerate(zip(shell_pairs.sp_A, shell_pairs.sp_B, shell_pairs.sp_npair)):
        if int(shell_l[int(A)]) == int(la) and int(shell_l[int(B)]) == int(lb) and int(n) == int(npair):
            return np.int32(idx)
    pytest.skip(f"basis does not provide shell pair ({la},{lb}) with npair={npair}")


def _pair_grid(task_ab: np.ndarray, task_cd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ab = np.repeat(np.asarray(task_ab, dtype=np.int32), int(task_cd.shape[0]))
    cd = np.tile(np.asarray(task_cd, dtype=np.int32), int(task_ab.shape[0]))
    return ab, cd


def _explicit_tile_reference(basis, shell_pairs, task_ab: np.ndarray, task_cd: np.ndarray) -> np.ndarray:
    from asuka.cueri.reference_eri_cart import eri_int2e_cart_tile

    ref_tiles = []
    for spAB, spCD in zip(task_ab, task_cd):
        A = int(shell_pairs.sp_A[int(spAB)])
        B = int(shell_pairs.sp_B[int(spAB)])
        C = int(shell_pairs.sp_A[int(spCD)])
        D = int(shell_pairs.sp_B[int(spCD)])
        ref_tiles.append(np.asarray(eri_int2e_cart_tile(basis, A, B, C, D), dtype=np.float64).reshape(-1))
    return np.stack(ref_tiles, axis=0)


def _assert_explicit_mode_parity(device_fn, quartet_test_system, task_ab, task_cd, *, mode, extra_kwargs, width):
    from asuka.cueri.tasks import TaskList

    tasks = TaskList(task_spAB=np.asarray(task_ab, dtype=np.int32), task_spCD=np.asarray(task_cd, dtype=np.int32))
    ref = _explicit_tile_reference(quartet_test_system.basis, quartet_test_system.shell_pairs, task_ab, task_cd)
    got = cupy.asnumpy(
        device_fn(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode=mode,
            **extra_kwargs,
        ).reshape((-1, int(width)))
    )
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=2e-11)


@pytest.fixture(scope="module")
def quartet_test_system():
    _require_cuda_runtime()

    from asuka.cueri.gpu import build_pair_tables_ss_device, to_device_basis_ss, to_device_shell_pairs
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order

    basis = _build_o_631g_star_basis()
    shell_pairs = build_shell_pairs_l_order(basis)
    dbasis = to_device_basis_ss(basis)
    dsp = to_device_shell_pairs(shell_pairs)
    pair_tables = build_pair_tables_ss_device(dbasis, dsp, threads=256)
    shell_l = np.asarray(basis.shell_l, dtype=np.int32)

    return SimpleNamespace(
        basis=basis,
        shell_pairs=shell_pairs,
        dbasis=dbasis,
        dsp=dsp,
        pair_tables=pair_tables,
        shell_l=shell_l,
    )


@pytest.fixture(scope="module")
def generic_quartet_test_system():
    _require_cuda_runtime()

    from asuka.cueri.gpu import build_pair_tables_ss_device, to_device_basis_ss, to_device_shell_pairs
    from asuka.cueri.shell_pairs import build_shell_pairs_l_order

    basis = _build_o_ccpvtz_basis()
    shell_pairs = build_shell_pairs_l_order(basis)
    dbasis = to_device_basis_ss(basis)
    dsp = to_device_shell_pairs(shell_pairs)
    pair_tables = build_pair_tables_ss_device(dbasis, dsp, threads=256)
    shell_l = np.asarray(basis.shell_l, dtype=np.int32)

    return SimpleNamespace(
        basis=basis,
        shell_pairs=shell_pairs,
        dbasis=dbasis,
        dsp=dsp,
        pair_tables=pair_tables,
        shell_l=shell_l,
    )


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_psss_mode_parity_against_cpu_reference(quartet_test_system, mode, extra_kwargs):
    from asuka.cueri.gpu import eri_psss_device

    ps_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, limit=2)
    ss_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 0, 0, limit=2)
    task_ab, task_cd = _pair_grid(ps_idx, ss_idx)
    _assert_explicit_mode_parity(
        eri_psss_device,
        quartet_test_system,
        task_ab,
        task_cd,
        mode=mode,
        extra_kwargs=extra_kwargs,
        width=3,
    )


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_ppss_mode_parity_against_cpu_reference(quartet_test_system, mode, extra_kwargs):
    from asuka.cueri.gpu import eri_ppss_device

    pp_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 1, limit=2)
    ss_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 0, 0, limit=2)
    task_ab, task_cd = _pair_grid(pp_idx, ss_idx)
    _assert_explicit_mode_parity(
        eri_ppss_device,
        quartet_test_system,
        task_ab,
        task_cd,
        mode=mode,
        extra_kwargs=extra_kwargs,
        width=9,
    )


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_psps_mode_parity_against_cpu_reference(quartet_test_system, mode, extra_kwargs):
    from asuka.cueri.gpu import eri_psps_device

    ps_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, limit=2)
    task_ab, task_cd = _pair_grid(ps_idx, ps_idx)
    _assert_explicit_mode_parity(
        eri_psps_device,
        quartet_test_system,
        task_ab,
        task_cd,
        mode=mode,
        extra_kwargs=extra_kwargs,
        width=9,
    )


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_dsss_mode_parity_against_cpu_reference(quartet_test_system, mode, extra_kwargs):
    from asuka.cueri.gpu import eri_dsss_device

    ds_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 2, 0, limit=2)
    ss_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 0, 0, limit=2)
    task_ab, task_cd = _pair_grid(ds_idx, ss_idx)
    _assert_explicit_mode_parity(
        eri_dsss_device,
        quartet_test_system,
        task_ab,
        task_cd,
        mode=mode,
        extra_kwargs=extra_kwargs,
        width=6,
    )


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_ppps_mode_parity_against_cpu_reference(quartet_test_system, mode, extra_kwargs):
    from asuka.cueri.gpu import eri_ppps_device
    from asuka.cueri.tasks import TaskList

    pp_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 1, limit=2)
    ps_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, limit=2)
    task_ab, task_cd = _pair_grid(pp_idx, ps_idx)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    ref = _explicit_tile_reference(quartet_test_system.basis, quartet_test_system.shell_pairs, task_ab, task_cd)
    got = cupy.asnumpy(
        eri_ppps_device(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode=mode,
            **extra_kwargs,
        ).reshape((-1, 27))
    )

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-11, err_msg=f"ppps mismatch in mode={mode}")


def test_ppps_auto_mode_tiny_bucket_against_cpu_reference(quartet_test_system):
    from asuka.cueri.gpu import eri_ppps_device
    from asuka.cueri.tasks import TaskList

    pp_idx = _find_shell_pair_with_npair(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 1, npair=9)
    ps_idx = _find_shell_pair_with_npair(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, npair=9)
    task_ab = np.asarray([pp_idx], dtype=np.int32)
    task_cd = np.asarray([ps_idx], dtype=np.int32)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    ref = _explicit_tile_reference(quartet_test_system.basis, quartet_test_system.shell_pairs, task_ab, task_cd)
    got = cupy.asnumpy(
        eri_ppps_device(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode="auto",
            work_small_max=512,
            work_large_min=200_000,
            blocks_per_task=4,
        ).reshape((-1, 27))
    )

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-11)


def test_ppps_auto_mode_regular_small_bucket_against_cpu_reference(quartet_test_system):
    from asuka.cueri.gpu import eri_ppps_device
    from asuka.cueri.tasks import TaskList

    pp_idx = _find_shell_pair_with_npair(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 1, npair=9)
    ps_idx = _find_shell_pair_with_npair(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, npair=18)
    task_ab = np.asarray([pp_idx], dtype=np.int32)
    task_cd = np.asarray([ps_idx], dtype=np.int32)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    ref = _explicit_tile_reference(quartet_test_system.basis, quartet_test_system.shell_pairs, task_ab, task_cd)
    got = cupy.asnumpy(
        eri_ppps_device(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode="auto",
            work_small_max=512,
            work_large_min=200_000,
            blocks_per_task=4,
        ).reshape((-1, 27))
    )

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    ("npair_ab", "npair_cd"),
    [
        (1, 1),
        (1, 3),
        (3, 1),
        (3, 3),
        (9, 3),
        (9, 1),
        (3, 9),
        (1, 9),
    ],
)
def test_ppps_auto_mode_packed_exact_shapes_against_cpu_reference(quartet_test_system, npair_ab: int, npair_cd: int):
    from asuka.cueri.gpu import eri_ppps_device
    from asuka.cueri.tasks import TaskList

    pp_idx = _find_shell_pair_with_npair(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 1, npair=int(npair_ab))
    ps_idx = _find_shell_pair_with_npair(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, npair=int(npair_cd))
    task_ab = np.asarray([pp_idx], dtype=np.int32)
    task_cd = np.asarray([ps_idx], dtype=np.int32)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    ref = _explicit_tile_reference(quartet_test_system.basis, quartet_test_system.shell_pairs, task_ab, task_cd)
    got = cupy.asnumpy(
        eri_ppps_device(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode="auto",
            work_small_max=512,
            work_large_min=200_000,
            blocks_per_task=4,
        ).reshape((-1, 27))
    )

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_pppp_mode_parity_against_cpu_reference(quartet_test_system, mode, extra_kwargs):
    from asuka.cueri.gpu import eri_pppp_device
    from asuka.cueri.tasks import TaskList

    pp_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 1, limit=2)
    task_ab, task_cd = _pair_grid(pp_idx, pp_idx)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    ref = _explicit_tile_reference(quartet_test_system.basis, quartet_test_system.shell_pairs, task_ab, task_cd)
    got = cupy.asnumpy(
        eri_pppp_device(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode=mode,
            **extra_kwargs,
        ).reshape((-1, 81))
    )

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=2e-11, err_msg=f"pppp mismatch in mode={mode}")


@pytest.mark.parametrize(("mode", "extra_kwargs"), _MODE_CASES)
def test_psds_mode_parity_against_cpu_rys_reference(quartet_test_system, mode, extra_kwargs):
    try:
        from asuka.cueri._eri_rys_cpu import eri_rys_tile_cart_sp_batch_cy
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"CPU Rys extension unavailable ({type(exc).__name__}: {exc})")

    from asuka.cueri.gpu import eri_psds_device
    from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu
    from asuka.cueri.tasks import TaskList

    ps_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 1, 0, limit=1)
    ds_idx = _find_shell_pair_indices(quartet_test_system.shell_pairs, quartet_test_system.shell_l, 2, 0, limit=2)
    task_ab = np.full((int(ds_idx.shape[0]),), int(ps_idx[0]), dtype=np.int32)
    task_cd = np.asarray(ds_idx, dtype=np.int32)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    pt_cpu = build_pair_tables_cpu(quartet_test_system.basis, quartet_test_system.shell_pairs)
    ref = np.asarray(
        eri_rys_tile_cart_sp_batch_cy(
            np.asarray(quartet_test_system.basis.shell_cxyz, dtype=np.float64, order="C"),
            np.asarray(quartet_test_system.basis.shell_l, dtype=np.int32, order="C"),
            np.asarray(quartet_test_system.shell_pairs.sp_A, dtype=np.int32, order="C"),
            np.asarray(quartet_test_system.shell_pairs.sp_B, dtype=np.int32, order="C"),
            np.asarray(quartet_test_system.shell_pairs.sp_pair_start, dtype=np.int32, order="C"),
            np.asarray(quartet_test_system.shell_pairs.sp_npair, dtype=np.int32, order="C"),
            np.asarray(pt_cpu.pair_eta, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_Px, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_Py, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_Pz, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_cK, dtype=np.float64, order="C"),
            int(task_ab[0]),
            np.asarray(task_cd, dtype=np.int32, order="C"),
            threads=0,
        ),
        dtype=np.float64,
    ).reshape((-1, 18))
    got = cupy.asnumpy(
        eri_psds_device(
            tasks,
            quartet_test_system.dbasis,
            quartet_test_system.dsp,
            quartet_test_system.pair_tables,
            threads=256,
            mode=mode,
            **extra_kwargs,
        ).reshape((-1, 18))
    )

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=2e-11, err_msg=f"psds mismatch in mode={mode}")


def test_fsss_generic_fallback_matches_cpu_rys_reference(generic_quartet_test_system):
    try:
        from asuka.cueri._eri_rys_cpu import eri_rys_tile_cart_sp_batch_cy
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"CPU Rys extension unavailable ({type(exc).__name__}: {exc})")

    from asuka.cueri.gpu import eri_rys_generic_device
    from asuka.cueri.native_class_sets import DISPATCH_NATIVE_CLASS_IDS
    from asuka.cueri.pair_tables_cpu import build_pair_tables_cpu
    from asuka.cueri.tasks import TaskList, eri_class_id

    fs_idx = _find_shell_pair_indices(generic_quartet_test_system.shell_pairs, generic_quartet_test_system.shell_l, 3, 0, limit=1)
    ss_idx = _find_shell_pair_indices(generic_quartet_test_system.shell_pairs, generic_quartet_test_system.shell_l, 0, 0, limit=2)

    class_id = int(eri_class_id(3, 0, 0, 0))
    assert class_id not in DISPATCH_NATIVE_CLASS_IDS

    task_ab = np.full((int(ss_idx.shape[0]),), int(fs_idx[0]), dtype=np.int32)
    task_cd = np.asarray(ss_idx, dtype=np.int32)
    tasks = TaskList(task_spAB=task_ab, task_spCD=task_cd)

    pt_cpu = build_pair_tables_cpu(generic_quartet_test_system.basis, generic_quartet_test_system.shell_pairs)
    ref = np.asarray(
        eri_rys_tile_cart_sp_batch_cy(
            np.asarray(generic_quartet_test_system.basis.shell_cxyz, dtype=np.float64, order="C"),
            np.asarray(generic_quartet_test_system.basis.shell_l, dtype=np.int32, order="C"),
            np.asarray(generic_quartet_test_system.shell_pairs.sp_A, dtype=np.int32, order="C"),
            np.asarray(generic_quartet_test_system.shell_pairs.sp_B, dtype=np.int32, order="C"),
            np.asarray(generic_quartet_test_system.shell_pairs.sp_pair_start, dtype=np.int32, order="C"),
            np.asarray(generic_quartet_test_system.shell_pairs.sp_npair, dtype=np.int32, order="C"),
            np.asarray(pt_cpu.pair_eta, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_Px, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_Py, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_Pz, dtype=np.float64, order="C"),
            np.asarray(pt_cpu.pair_cK, dtype=np.float64, order="C"),
            int(task_ab[0]),
            np.asarray(task_cd, dtype=np.int32, order="C"),
            threads=0,
        ),
        dtype=np.float64,
    ).reshape((int(task_cd.shape[0]), -1))
    got = cupy.asnumpy(
        eri_rys_generic_device(
            tasks,
            generic_quartet_test_system.dbasis,
            generic_quartet_test_system.dsp,
            generic_quartet_test_system.pair_tables,
            la=3,
            lb=0,
            lc=0,
            ld=0,
            threads=256,
        )
    ).reshape((int(task_cd.shape[0]), -1))

    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=2e-11, err_msg="fsss generic fallback mismatch")
