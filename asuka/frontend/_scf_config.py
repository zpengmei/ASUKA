from __future__ import annotations

import os
from typing import Any

from asuka.integrals.cueri_df import CuERIDFConfig

from ._scf_cache import cuda_device_id_or_neg1


def cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def env_int_default(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(str(name), str(default))).strip())
    except Exception:
        return int(default)


def env_str_default(name: str, default: str) -> str:
    return str(os.environ.get(str(name), str(default))).strip()


def df_config_key(config: CuERIDFConfig | None) -> tuple[Any, ...]:
    cfg = CuERIDFConfig() if config is None else config
    return (
        str(cfg_get(cfg, "backend", "gpu_rys")).strip().lower(),
        str(cfg_get(cfg, "mode", "warp")).strip().lower(),
        int(cfg_get(cfg, "threads", 256)),
        int(cfg_get(cfg, "int3c_work_small_max", 512)),
        int(cfg_get(cfg, "int3c_work_large_min", 200_000)),
        int(cfg_get(cfg, "int3c_blocks_per_task", 4)),
        str(cfg_get(cfg, "int3c_plan_policy", "auto")).strip().lower(),
        int(cuda_device_id_or_neg1()),
    )


def resolve_df_config_overrides(
    df_config: CuERIDFConfig | None,
    *,
    backend: str | None = None,
    mode: str | None = None,
    threads: int | None = None,
) -> CuERIDFConfig:
    cfg0 = CuERIDFConfig() if df_config is None else df_config
    return CuERIDFConfig(
        backend=str(cfg0.backend if backend is None else backend),
        mode=str(cfg0.mode if mode is None else mode),
        threads=int(cfg0.threads if threads is None else threads),
        stream=cfg0.stream,
    )


def resolve_cueri_df_config(
    df_config: CuERIDFConfig | None,
    *,
    df_int3c_plan_policy: str | None = None,
    df_int3c_work_small_max: int | None = None,
    df_int3c_work_large_min: int | None = None,
    df_int3c_blocks_per_task: int | None = None,
) -> CuERIDFConfig:
    defaults = CuERIDFConfig()
    cfg_in = defaults if df_config is None else df_config

    if df_config is None:
        plan_policy = (
            str(df_int3c_plan_policy).strip().lower()
            if df_int3c_plan_policy is not None
            else env_str_default("ASUKA_DF_INT3C_PLAN_POLICY", str(getattr(defaults, "int3c_plan_policy", "auto")))
        )
        work_small_max = (
            int(df_int3c_work_small_max)
            if df_int3c_work_small_max is not None
            else env_int_default("ASUKA_DF_INT3C_WORK_SMALL_MAX", int(getattr(defaults, "int3c_work_small_max", 512)))
        )
        work_large_min = (
            int(df_int3c_work_large_min)
            if df_int3c_work_large_min is not None
            else env_int_default("ASUKA_DF_INT3C_WORK_LARGE_MIN", int(getattr(defaults, "int3c_work_large_min", 200_000)))
        )
        blocks_per_task = (
            int(df_int3c_blocks_per_task)
            if df_int3c_blocks_per_task is not None
            else env_int_default("ASUKA_DF_INT3C_BLOCKS_PER_TASK", int(getattr(defaults, "int3c_blocks_per_task", 4)))
        )
    else:
        plan_policy = (
            str(df_int3c_plan_policy).strip().lower()
            if df_int3c_plan_policy is not None
            else str(cfg_get(cfg_in, "int3c_plan_policy", getattr(defaults, "int3c_plan_policy", "auto"))).strip().lower()
        )
        work_small_max = (
            int(df_int3c_work_small_max)
            if df_int3c_work_small_max is not None
            else int(cfg_get(cfg_in, "int3c_work_small_max", getattr(defaults, "int3c_work_small_max", 512)))
        )
        work_large_min = (
            int(df_int3c_work_large_min)
            if df_int3c_work_large_min is not None
            else int(cfg_get(cfg_in, "int3c_work_large_min", getattr(defaults, "int3c_work_large_min", 200_000)))
        )
        blocks_per_task = (
            int(df_int3c_blocks_per_task)
            if df_int3c_blocks_per_task is not None
            else int(cfg_get(cfg_in, "int3c_blocks_per_task", getattr(defaults, "int3c_blocks_per_task", 4)))
        )

    return CuERIDFConfig(
        backend=str(cfg_get(cfg_in, "backend", defaults.backend)),
        mode=str(cfg_get(cfg_in, "mode", defaults.mode)),
        threads=int(cfg_get(cfg_in, "threads", defaults.threads)),
        stream=cfg_get(cfg_in, "stream", defaults.stream),
        int3c_work_small_max=max(1, int(work_small_max)),
        int3c_work_large_min=max(2, int(work_large_min)),
        int3c_blocks_per_task=max(1, int(blocks_per_task)),
        int3c_plan_policy=str(plan_policy),
    )
