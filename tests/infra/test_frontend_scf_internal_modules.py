from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from asuka.frontend._scf_cache import (
    cache_clear_all,
    cache_get,
    cache_put,
    cuda_device_id_or_neg1,
    mol_cache_key,
    normalize_basis_key,
)
from asuka.frontend._scf_config import (
    cfg_get,
    df_config_key,
    resolve_cueri_df_config,
    resolve_df_config_overrides,
)
from asuka.frontend._scf_build import (
    apply_sph_transform,
    atom_coords_charges_bohr,
    unique_elements,
)
from asuka.frontend._scf_df_build import prepare_direct_df_inputs
from asuka.frontend._scf_dispatch import run_hf_df_dispatch
from asuka.frontend._scf_dense import build_dense_ao_eri, dense_default_threads
import asuka.frontend._scf_keys as scf_keys_mod
from asuka.frontend._scf_keys import copy_mo_coeff_for_cache, rhf_guess_key, rhf_prep_key
import asuka.frontend._scf_methods as scf_methods
from asuka.frontend._scf_methods import _maybe_pack_df_B
from asuka.frontend._scf_spin import nalpha_nbeta_from_mol
from asuka.frontend.molecule import Molecule
import asuka.frontend.scf as scf_mod
from asuka.integrals.cueri_df import CuERIDFConfig
from asuka.integrals.int1e_cart import Int1eResult
import numpy as np


def test_scf_cache_put_get_lru_behavior():
    cache: OrderedDict[tuple[str, ...], int] = OrderedDict()
    cache_put(cache, ("a",), 1, max_size=2)
    cache_put(cache, ("b",), 2, max_size=2)
    assert list(cache.keys()) == [("a",), ("b",)]

    assert cache_get(cache, ("a",)) == 1
    assert list(cache.keys()) == [("b",), ("a",)]

    cache_put(cache, ("c",), 3, max_size=2)
    assert list(cache.keys()) == [("a",), ("c",)]
    assert cache_get(cache, ("b",)) is None


def test_scf_cache_clear_and_disabled_put():
    c1: OrderedDict[tuple[str, ...], int] = OrderedDict()
    c2: OrderedDict[tuple[str, ...], int] = OrderedDict()

    cache_put(c1, ("x",), 1, max_size=0)
    assert len(c1) == 0

    cache_put(c1, ("x",), 1, max_size=4)
    cache_put(c2, ("y",), 2, max_size=4)
    cache_clear_all(c1, c2)
    assert len(c1) == 0
    assert len(c2) == 0


def test_scf_cache_key_normalizers():
    assert normalize_basis_key("STO-3G") == ("str", "sto-3g")
    d1 = {"H": "sto-3g", "O": "6-31g"}
    d2 = {"O": "6-31g", "H": "sto-3g"}
    assert normalize_basis_key(d1) == normalize_basis_key(d2)

    key = normalize_basis_key(["custom", 1])
    assert key[0] == "list"
    assert isinstance(key[1], str)


def test_scf_mol_cache_key_rounding_and_device_id():
    mol = SimpleNamespace(
        atoms_bohr=[
            ("H", (0.0, 0.0, 1.0000000000004)),
            ("O", (0.0, -0.5, 0.25)),
        ],
        charge=0,
        spin=1,
        cart=False,
    )
    key = mol_cache_key(mol)
    assert key[1:] == (0, 1, False)
    assert key[0][0][3] == round(1.0000000000004, 12)

    dev = cuda_device_id_or_neg1()
    assert isinstance(dev, int)


def test_scf_config_helpers_cfg_get_and_overrides():
    assert cfg_get(None, "x", 7) == 7
    assert cfg_get({"x": 3}, "x", 7) == 3
    assert cfg_get(SimpleNamespace(x=5), "x", 7) == 5

    cfg = CuERIDFConfig(backend="gpu_rys", mode="warp", threads=64, stream=123)
    out = resolve_df_config_overrides(cfg, backend="cpu_ref", threads=8)
    assert out.backend == "cpu_ref"
    assert out.mode == "warp"
    assert out.threads == 8
    assert out.stream == 123


def test_scf_config_resolve_cueri_df_config_from_env(monkeypatch):
    monkeypatch.setenv("ASUKA_DF_INT3C_PLAN_POLICY", "manual")
    monkeypatch.setenv("ASUKA_DF_INT3C_WORK_SMALL_MAX", "17")
    monkeypatch.setenv("ASUKA_DF_INT3C_WORK_LARGE_MIN", "23")
    monkeypatch.setenv("ASUKA_DF_INT3C_BLOCKS_PER_TASK", "3")

    cfg = resolve_cueri_df_config(None)
    assert cfg.int3c_plan_policy == "manual"
    assert cfg.int3c_work_small_max == 17
    assert cfg.int3c_work_large_min == 23
    assert cfg.int3c_blocks_per_task == 3


def test_scf_config_df_config_key_shape():
    cfg = CuERIDFConfig(
        backend="GPU_RYS",
        mode="WARP",
        threads=128,
        stream=0,
        int3c_work_small_max=111,
        int3c_work_large_min=222,
        int3c_blocks_per_task=4,
        int3c_plan_policy="AUTO",
    )
    key = df_config_key(cfg)
    assert len(key) == 8
    assert key[:7] == ("gpu_rys", "warp", 128, 111, 222, 4, "auto")
    assert isinstance(key[7], int)


def test_scf_build_atom_and_element_helpers():
    mol = Molecule.from_atoms(
        [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.97))],
        unit="angstrom",
        basis="sto-3g",
        cart=True,
    )
    coords, charges = atom_coords_charges_bohr(mol)
    assert coords.shape == (2, 3)
    assert charges.tolist() == [8.0, 1.0]
    assert unique_elements(mol) == ["H", "O"]


def test_scf_build_apply_sph_transform_cart_passthrough():
    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
        unit="angstrom",
        basis="sto-3g",
        cart=True,
    )
    int1e = Int1eResult(S=np.eye(2), T=np.eye(2), V=np.eye(2))
    B = np.ones((2, 2, 1))
    out_int1e, out_B, sph_map = apply_sph_transform(mol, int1e, B, ao_basis=None)
    assert out_int1e is int1e
    assert out_B is B
    assert sph_map is None


def test_scf_dense_default_threads_policy():
    assert dense_default_threads("cpu") == 0
    assert dense_default_threads("cuda") == 256
    assert dense_default_threads(" CuDa ") == 256


def test_scf_dense_build_backend_validation():
    with np.testing.assert_raises(ValueError):
        build_dense_ao_eri(
            ao_basis=object(),
            backend="bogus",
            dense_threads=None,
            dense_max_tile_bytes=1024,
            dense_eps_ao=1e-12,
            dense_max_l=None,
            dense_mem_budget_gib=None,
            default_mem_budget_gib=8.0,
            profile=None,
        )


def test_scf_spin_helper_nalpha_nbeta_validation():
    assert nalpha_nbeta_from_mol(SimpleNamespace(nelectron=10, spin=2)) == (6, 4)
    with np.testing.assert_raises(ValueError):
        nalpha_nbeta_from_mol(SimpleNamespace(nelectron=5, spin=0))
    with np.testing.assert_raises(ValueError):
        nalpha_nbeta_from_mol(SimpleNamespace(nelectron=0, spin=0))


def test_scf_keys_helpers_key_shape_and_device_binding(monkeypatch):
    monkeypatch.setattr(scf_keys_mod, "cuda_device_id_or_neg1", lambda: 7)
    mol = SimpleNamespace(
        atoms_bohr=[("H", (0.0, 0.0, 0.0))],
        charge=0,
        spin=0,
        cart=True,
    )
    cfg = CuERIDFConfig(backend="gpu_rys", mode="warp", threads=64)
    prep = rhf_prep_key(
        mol,
        basis_in="sto-3g",
        auxbasis="autoaux",
        expand_contractions=True,
        df_config=cfg,
        df_layout_build="Qmn",
    )
    guess = rhf_guess_key(
        mol,
        basis_in="sto-3g",
        auxbasis="autoaux",
        expand_contractions=True,
    )
    assert prep[-1] == "qmn"
    assert guess[0] == "rhf"
    assert guess[-1] == 7


def test_scf_keys_copy_mo_coeff_numpy_copy():
    src = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32, order="F")
    out = copy_mo_coeff_for_cache(src)
    assert out.dtype == np.float64
    assert bool(out.flags["C_CONTIGUOUS"])
    src[0, 0] = 99.0
    assert float(out[0, 0]) == 1.0


def test_scf_df_build_prepare_direct_df_inputs_with_prebuilt_metric():
    mol = Molecule.from_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))],
        unit="angstrom",
        basis="sto-3g",
        cart=True,
    )
    marker = np.asarray([[1.0]])
    cfg, ao_basis, basis_name, int1e_scf, aux_basis, auxbasis_name, sph_map, df_ao_rep, L_metric = prepare_direct_df_inputs(
        mol,
        basis_in="sto-3g",
        auxbasis="autoaux",
        expand_contractions=True,
        df_config=None,
        L_metric=marker,
    )
    assert cfg is not None
    assert int1e_scf.S.shape[0] == int1e_scf.hcore.shape[0]
    assert "sto" in basis_name.lower()
    assert isinstance(auxbasis_name, str) and len(auxbasis_name) > 0
    assert sph_map is None
    assert df_ao_rep == "cart"
    assert L_metric is marker
    assert ao_basis is not None and aux_basis is not None


def _dispatch_ops(call_log):
    names = (
        "run_rks_df",
        "run_uks_df",
        "run_rhf_dense",
        "run_uhf_dense",
        "run_rohf_dense",
        "run_rhf_direct",
        "run_uhf_direct",
        "run_rohf_direct",
        "run_rhf_direct_df",
        "run_uhf_direct_df",
        "run_rohf_direct_df",
        "run_rhf_thc",
        "run_uhf_thc",
        "run_rohf_thc",
        "run_rhf_df",
        "run_uhf_df",
        "run_rohf_df",
        "run_rhf_df_cpu",
        "run_uhf_df_cpu",
        "run_rohf_df_cpu",
    )

    def _mk(name):
        def _runner(*args, **kwargs):
            call_log.append((name, args, kwargs))
            return SimpleNamespace(name=name, direct_jk_ctx="ctx")

        return _runner

    return {name: _mk(name) for name in names}


def test_scf_dispatch_dense_branch_filters_df_knobs():
    calls = []

    def _meta(out, *, two_e_backend, direct_jk_ctx=None):
        return (out.name, two_e_backend, direct_jk_ctx)

    out = run_hf_df_dispatch(
        mol=SimpleNamespace(),
        method="rhf",
        backend="cuda",
        df=False,
        two_e_backend=None,
        guess=None,
        dm0=None,
        mo_coeff0=None,
        kwargs={
            "df_config": object(),
            "auxbasis": "autoaux",
            "custom": 7,
            "dense_max_l": 3,
        },
        ops=_dispatch_ops(calls),
        with_two_e_metadata=_meta,
    )
    assert out[0] == "run_rhf_dense"
    assert out[1] == "dense"
    assert out[2] is None
    _name, _args, kw = calls[0]
    assert "custom" in kw and kw["custom"] == 7
    assert "df_config" not in kw
    assert "auxbasis" not in kw
    assert "dense_max_l" in kw


def test_scf_dispatch_cpu_df_branch_filters_cuda_df_knobs():
    calls = []

    def _meta(out, *, two_e_backend, direct_jk_ctx=None):
        return (out.name, two_e_backend, direct_jk_ctx)

    out = run_hf_df_dispatch(
        mol=SimpleNamespace(),
        method="uhf",
        backend="cpu",
        df=True,
        two_e_backend=None,
        guess=None,
        dm0=None,
        mo_coeff0=None,
        kwargs={
            "df_int3c_plan_policy": "manual",
            "df_int3c_work_small_max": 11,
            "df_k_cache_max_mb": 64,
            "keep": "ok",
        },
        ops=_dispatch_ops(calls),
        with_two_e_metadata=_meta,
    )
    assert out[0] == "run_uhf_df_cpu"
    assert out[1] == "df"
    _name, _args, kw = calls[0]
    assert "keep" in kw and kw["keep"] == "ok"
    assert "df_int3c_plan_policy" not in kw
    assert "df_int3c_work_small_max" not in kw
    assert "df_k_cache_max_mb" not in kw


def test_scf_dispatch_direct_requires_cuda():
    with np.testing.assert_raises(NotImplementedError):
        run_hf_df_dispatch(
            mol=SimpleNamespace(),
            method="rhf",
            backend="cpu",
            df=True,
            two_e_backend="direct",
            guess=None,
            dm0=None,
            mo_coeff0=None,
            kwargs={},
            ops=_dispatch_ops([]),
            with_two_e_metadata=lambda out, *, two_e_backend, direct_jk_ctx=None: out,
        )


def test_scf_methods_pack_df_B_passthrough_without_gpu():
    b = np.ones((2, 2, 1))
    int1e = SimpleNamespace(S=np.eye(2))
    out = _maybe_pack_df_B(b, df_layout_s="mnq", int1e_scf=int1e)
    assert out is b


def test_scf_run_rhf_df_delegates_to_methods_impl(monkeypatch):
    captured = {}

    def _fake_impl(mol, **kwargs):
        captured["mol"] = mol
        captured["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(scf_mod, "_run_rhf_df_impl", _fake_impl)
    mol = SimpleNamespace()
    out = scf_mod.run_rhf_df(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        max_cycle=9,
        conv_tol=1e-9,
    )

    assert out == "ok"
    assert captured["mol"] is mol
    assert captured["kwargs"]["basis"] == "sto-3g"
    assert captured["kwargs"]["max_cycle"] == 9
    assert captured["kwargs"]["init_fock_cycles_default"] == scf_mod._HF_INIT_FOCK_CYCLES
    assert captured["kwargs"]["resolve_cueri_df_config"] is scf_mod._resolve_cueri_df_config
    assert captured["kwargs"]["result_cls"] is scf_mod.RHFDFRunResult


def test_scf_methods_rhf_df_impl_cache_hit_df_config_and_guess_flow(monkeypatch):
    mol = SimpleNamespace(spin=0, nelectron=2, cart=True, energy_nuc=lambda: 1.25)
    cfg = SimpleNamespace(name="cfg")
    prep_cache = object()
    guess_cache = object()
    prep_tuple = (
        "ao_basis",
        "sto-3g",
        SimpleNamespace(S=np.eye(2), hcore=np.eye(2)),
        "aux_basis",
        "autoaux",
        np.ones((2, 2, 1)),
        "Lchol",
    )
    calls = {"resolve": [], "prep_key": [], "guess_key": [], "cache_get": [], "cache_put": [], "run_cfg": []}

    def _resolve(df_config, **kwargs):
        calls["resolve"].append((df_config, kwargs))
        return cfg

    def _prep_key(*args, **kwargs):
        calls["prep_key"].append((args, kwargs))
        return ("prep",)

    def _guess_key(*args, **kwargs):
        calls["guess_key"].append((args, kwargs))
        return ("guess",)

    def _cache_get(cache, key):
        calls["cache_get"].append((cache, key))
        if cache is prep_cache:
            return prep_tuple
        if cache is guess_cache:
            return np.eye(2)
        return None

    def _cache_put(cache, key, value, *, max_size):
        calls["cache_put"].append((cache, key, value, max_size))

    def _rhf_df(*args, **kwargs):
        return SimpleNamespace(
            converged=True,
            mo_coeff=np.eye(2),
            method="rhf",
            niter=1,
            e_tot=0.0,
            e_elec=0.0,
            e_nuc=1.25,
            mo_energy=np.zeros(2),
            mo_occ=np.array([2.0, 0.0]),
        )

    def _mk_cfg(**kwargs):
        calls["run_cfg"].append(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(scf_methods, "rhf_df", _rhf_df)

    out = scf_methods.run_rhf_df_impl(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        df_config="in_cfg",
        df_layout="mnQ",
        max_cycle=4,
        conv_tol=1e-9,
        profile={},
        init_fock_cycles_default=1,
        resolve_cueri_df_config=_resolve,
        rhf_prep_key=_prep_key,
        rhf_guess_key=_guess_key,
        cache_get=_cache_get,
        cache_put=_cache_put,
        rhf_prep_cache=prep_cache,
        rhf_guess_cache=guess_cache,
        hf_prep_cache_max=7,
        hf_guess_cache_max=9,
        copy_mo_coeff_for_cache=lambda x: x,
        make_df_run_config=_mk_cfg,
        result_cls=lambda **kw: SimpleNamespace(**kw),
    )

    assert calls["resolve"][0][0] == "in_cfg"
    assert calls["prep_key"][0][1]["df_config"] is cfg
    assert calls["prep_key"][0][1]["df_layout_build"] == "mnq"
    assert out.profile["df_build"]["cache_hit"] is True
    assert out.profile["scf_guess"]["cache_hit"] is True
    assert calls["run_cfg"][0]["df_config"] is cfg
    assert out.df_run_config.df_config is cfg
    assert any(c[0] is guess_cache and c[1] == ("guess",) for c in calls["cache_put"])


def test_scf_methods_rhf_df_impl_cache_miss_builds_and_records_cache(monkeypatch):
    mol = SimpleNamespace(spin=0, nelectron=2, cart=True, energy_nuc=lambda: 0.5)
    cfg = SimpleNamespace(name="cfg_miss")
    prep_cache = object()
    guess_cache = object()
    calls = {"build": [], "cache_get": [], "cache_put": []}

    def _resolve(*args, **kwargs):
        return cfg

    def _cache_get(cache, key):
        calls["cache_get"].append((cache, key))
        return None

    def _cache_put(cache, key, value, *, max_size):
        calls["cache_put"].append((cache, key, max_size))

    def _build_problem(*args, **kwargs):
        return (
            "ao",
            "sto-3g",
            SimpleNamespace(S=np.eye(2), hcore=np.eye(2)),
            "aux",
            "autoaux",
        )

    def _build_df(mol_in, *, cfg, df_layout_s, **kwargs):
        assert mol_in is mol
        calls["build"].append((cfg, df_layout_s))
        return np.ones((2, 2, 1)), "L"

    def _transform(*args, **kwargs):
        int1e = kwargs["int1e"]
        B = kwargs["B"]
        return int1e, B, None

    def _rhf_df(*args, **kwargs):
        return SimpleNamespace(
            converged=False,
            mo_coeff=np.eye(2),
            method="rhf",
            niter=1,
            e_tot=0.0,
            e_elec=0.0,
            e_nuc=0.5,
            mo_energy=np.zeros(2),
            mo_occ=np.array([2.0, 0.0]),
        )

    monkeypatch.setattr(scf_methods, "_build_ao_int1e_aux_problem", _build_problem)
    monkeypatch.setattr(scf_methods, "_build_df_factors_gpu", _build_df)
    monkeypatch.setattr(scf_methods, "_transform_df_for_scf", _transform)
    monkeypatch.setattr(scf_methods, "rhf_df", _rhf_df)

    out = scf_methods.run_rhf_df_impl(
        mol,
        basis="sto-3g",
        auxbasis="autoaux",
        df_config="cfg_in",
        df_layout="Qmn",
        profile={},
        init_fock_cycles_default=1,
        resolve_cueri_df_config=_resolve,
        rhf_prep_key=lambda *a, **k: ("prep_miss",),
        rhf_guess_key=lambda *a, **k: ("guess_miss",),
        cache_get=_cache_get,
        cache_put=_cache_put,
        rhf_prep_cache=prep_cache,
        rhf_guess_cache=guess_cache,
        hf_prep_cache_max=3,
        hf_guess_cache_max=5,
        copy_mo_coeff_for_cache=lambda x: x,
        make_df_run_config=lambda **kw: SimpleNamespace(**kw),
        result_cls=lambda **kw: SimpleNamespace(**kw),
    )

    assert calls["build"] == [(cfg, "qmn")]
    assert out.profile["df_build"]["cache_hit"] is False
    assert out.profile["scf_guess"]["cache_hit"] is False
    assert (prep_cache, ("prep_miss",), 3) in calls["cache_put"]
    assert not any(c[0] is guess_cache for c in calls["cache_put"])
