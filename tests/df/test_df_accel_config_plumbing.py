from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from asuka.frontend import scf as hf_scf
from asuka.integrals.cueri_df import CuERIDFConfig


def test_resolve_cueri_df_config_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("ASUKA_DF_INT3C_PLAN_POLICY", "legacy")
    monkeypatch.setenv("ASUKA_DF_INT3C_WORK_SMALL_MAX", "777")
    monkeypatch.setenv("ASUKA_DF_INT3C_WORK_LARGE_MIN", "888888")
    monkeypatch.setenv("ASUKA_DF_INT3C_BLOCKS_PER_TASK", "9")

    cfg = hf_scf._resolve_cueri_df_config(None)
    assert isinstance(cfg, CuERIDFConfig)
    assert str(cfg.int3c_plan_policy) == "legacy"
    assert int(cfg.int3c_work_small_max) == 777
    assert int(cfg.int3c_work_large_min) == 888888
    assert int(cfg.int3c_blocks_per_task) == 9


def test_df_config_key_includes_accel_fields():
    cfg0 = CuERIDFConfig(int3c_plan_policy="auto", int3c_work_small_max=512, int3c_work_large_min=200000, int3c_blocks_per_task=4)
    cfg1 = replace(cfg0, int3c_plan_policy="legacy")
    cfg2 = replace(cfg0, int3c_work_small_max=1024)
    assert hf_scf._df_config_key(cfg0) != hf_scf._df_config_key(cfg1)
    assert hf_scf._df_config_key(cfg0) != hf_scf._df_config_key(cfg2)


def test_run_hf_df_dense_strips_df_accel_kwargs(monkeypatch):
    seen: dict = {}

    def _fake_run_rhf_dense(_mol, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(hf_scf, "run_rhf_dense", _fake_run_rhf_dense)

    hf_scf.run_hf_df(
        object(),
        method="rhf",
        backend="cuda",
        two_e_backend="dense",
        df_int3c_plan_policy="fast",
        df_int3c_work_small_max=111,
        df_int3c_work_large_min=222,
        df_int3c_blocks_per_task=3,
        df_k_cache_max_mb=4096,
    )

    assert "df_int3c_plan_policy" not in seen
    assert "df_int3c_work_small_max" not in seen
    assert "df_int3c_work_large_min" not in seen
    assert "df_int3c_blocks_per_task" not in seen
    assert "df_k_cache_max_mb" not in seen


def test_run_hf_df_cpu_strips_gpu_df_accel_kwargs(monkeypatch):
    seen: dict = {}

    def _fake_run_rhf_df_cpu(_mol, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(hf_scf, "run_rhf_df_cpu", _fake_run_rhf_df_cpu)

    hf_scf.run_hf_df(
        object(),
        method="rhf",
        backend="cpu",
        two_e_backend="df",
        df_int3c_plan_policy="fast",
        df_int3c_work_small_max=111,
        df_int3c_work_large_min=222,
        df_int3c_blocks_per_task=3,
        df_k_cache_max_mb=4096,
    )

    assert "df_int3c_plan_policy" not in seen
    assert "df_int3c_work_small_max" not in seen
    assert "df_int3c_work_large_min" not in seen
    assert "df_int3c_blocks_per_task" not in seen
    assert "df_k_cache_max_mb" not in seen
