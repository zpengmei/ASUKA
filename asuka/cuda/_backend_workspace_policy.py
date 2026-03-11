from __future__ import annotations


def resolve_cache_csr_tiles(
    cache_csr_tiles,
    *,
    j_tile: int,
    ncsf: int,
    norb: int,
    csr_capacity_mult: float,
) -> bool:
    if isinstance(cache_csr_tiles, str) and cache_csr_tiles.lower() == "auto":
        jt = int(j_tile)
        ncsf_i = int(ncsf)
        is_multi_tile = jt < ncsf_i
        if is_multi_tile:
            ntiles = (ncsf_i + jt - 1) // jt
            n_pairs = int(norb) * (int(norb) - 1)
            avg_nnz_per_tile = int(float(jt) * float(n_pairs) * max(1.0, float(csr_capacity_mult)))
            est_cache_bytes = ntiles * avg_nnz_per_tile * 28
            return bool(ntiles <= 32 and est_cache_bytes <= 2 * 1024 * 1024 * 1024)
        return False
    return bool(cache_csr_tiles)


def resolve_fp32_csr_cache(fp32_csr_cache, *, dtype_is_float32: bool) -> bool:
    if isinstance(fp32_csr_cache, str) and fp32_csr_cache.lower() == "auto":
        return bool(dtype_is_float32)
    return bool(fp32_csr_cache)


def resolve_csr_host_cache_mode(csr_host_cache) -> str:
    if isinstance(csr_host_cache, str):
        mode = csr_host_cache.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "host", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "none", "disabled"):
            return "off"
        raise ValueError("csr_host_cache must be bool or one of: auto/on/off")
    return "on" if bool(csr_host_cache) else "off"


def resolve_csr_host_cache_enabled(
    *,
    mode: str,
    j_tile: int,
    ncsf: int,
    budget_bytes: int,
    min_ncsf: int,
    cache_csr_tiles: bool,
    aggregate_offdiag_k: bool,
    use_epq_table: bool,
) -> bool:
    if mode == "on":
        return bool(
            int(j_tile) < int(ncsf)
            and int(budget_bytes) > 0
            and not (bool(aggregate_offdiag_k) and bool(use_epq_table))
        )
    if mode == "auto":
        return bool(
            int(j_tile) < int(ncsf)
            and int(ncsf) >= int(min_ncsf)
            and not bool(cache_csr_tiles)
            and int(budget_bytes) > 0
            and not (bool(aggregate_offdiag_k) and bool(use_epq_table))
        )
    return False


def resolve_epq_apply_cache_budget_gib(epq_apply_cache_budget_gib, env_epq_apply_cache_budget_gib: str) -> float:
    env = str(env_epq_apply_cache_budget_gib or "").strip()
    out = epq_apply_cache_budget_gib
    if env:
        try:
            _in = 4.0 if out is None else float(out)
        except Exception as e:
            raise ValueError("epq_apply_cache_budget_gib must be a float") from e
        if out is None or _in <= 0.0 or abs(_in - 4.0) < 1e-12:
            try:
                out = float(env)
            except Exception as e:
                raise ValueError("ASUKA_CUGUGA_EPQ_APPLY_CACHE_BUDGET_GIB must be a float") from e
    return max(0.0, float(out))


def resolve_epq_apply_cache_mode(epq_apply_cache, env_epq_apply_cache: str) -> str:
    env = str(env_epq_apply_cache or "").strip()
    raw = epq_apply_cache
    if env:
        raw_s = "" if raw is None else str(raw).strip().lower()
        if raw_s in ("", "auto"):
            raw = env
    if isinstance(raw, str):
        mode = raw.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "none", "disabled"):
            return "off"
        raise ValueError("epq_apply_cache must be bool or one of: auto/on/off")
    return "on" if bool(raw) else "off"


def resolve_epq_apply_cache_enabled(
    *,
    mode: str,
    aggregate_offdiag_k: bool,
    use_epq_table: bool,
    budget_bytes: int,
    has_epq_table_device_build: bool,
    csr_host_cache_mode: str,
) -> bool:
    out = False
    if mode == "on":
        out = bool(
            bool(aggregate_offdiag_k)
            and int(budget_bytes) > 0
            and bool(has_epq_table_device_build)
        )
    elif mode == "auto":
        out = bool(
            bool(aggregate_offdiag_k)
            and not bool(use_epq_table)
            and int(budget_bytes) > 0
            and bool(has_epq_table_device_build)
        )
        if str(csr_host_cache_mode) == "on":
            out = False
    return bool(out)


def resolve_csr_pipeline_streams(csr_pipeline_streams) -> tuple[str, int]:
    if isinstance(csr_pipeline_streams, str):
        mode = csr_pipeline_streams.strip().lower()
        if mode in ("", "auto"):
            req_streams = 2
            out_mode = "auto"
        elif mode in ("off", "none", "0", "false", "no"):
            req_streams = 0
            out_mode = "off"
        elif mode in ("on", "1", "true", "yes"):
            req_streams = 2
            out_mode = "on"
        else:
            req_streams = int(mode)
            out_mode = "manual"
    else:
        req_streams = int(csr_pipeline_streams)
        out_mode = "manual"

    req_streams = max(0, int(req_streams))
    if req_streams in (0, 1):
        out_streams = 0
    else:
        out_streams = min(3, int(req_streams))
    return out_mode, int(out_streams)


def resolve_csr_pipeline_enabled(
    *,
    streams_mode: str,
    streams: int,
    j_tile: int,
    ncsf: int,
    min_ncsf: int,
    cache_csr_tiles: bool,
    csr_host_cache_enabled: bool,
    aggregate_offdiag_k: bool,
    use_epq_table: bool,
) -> bool:
    if int(streams) < 2:
        return False
    if streams_mode == "auto":
        return bool(
            int(j_tile) < int(ncsf)
            and int(ncsf) >= int(min_ncsf)
            and not bool(cache_csr_tiles)
            and not bool(csr_host_cache_enabled)
            and not (bool(aggregate_offdiag_k) and bool(use_epq_table))
        )
    return bool(
        int(j_tile) < int(ncsf)
        and not bool(cache_csr_tiles)
        and not bool(csr_host_cache_enabled)
        and not (bool(aggregate_offdiag_k) and bool(use_epq_table))
    )


def resolve_prefilter_trivial_tasks_mode(prefilter_trivial_tasks) -> str:
    if isinstance(prefilter_trivial_tasks, str):
        mode = prefilter_trivial_tasks.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "disabled", "none"):
            return "off"
        raise ValueError("prefilter_trivial_tasks must be bool or one of: auto/on/off")
    return "on" if bool(prefilter_trivial_tasks) else "off"


def resolve_prefilter_trivial_tasks_enabled(
    *,
    mode: str,
    j_tile: int,
    ncsf: int,
    min_ncsf: int,
    use_epq_table: bool,
    aggregate_offdiag_k: bool,
) -> bool:
    if mode == "auto":
        disable_prefilter_for_agg_noepq = bool(aggregate_offdiag_k) and (not bool(use_epq_table))
        return bool(
            int(j_tile) < int(ncsf)
            and int(ncsf) >= int(min_ncsf)
            and not bool(use_epq_table)
            and not bool(disable_prefilter_for_agg_noepq)
        )
    if mode == "on":
        return bool(int(j_tile) < int(ncsf) and not bool(use_epq_table))
    return False


def resolve_skip_zero_x_tiles_mode(skip_zero_x_tiles) -> str:
    if isinstance(skip_zero_x_tiles, str):
        mode = skip_zero_x_tiles.strip().lower()
        if mode in ("", "auto"):
            return "auto"
        if mode in ("1", "true", "yes", "on", "enabled"):
            return "on"
        if mode in ("0", "false", "no", "off", "disabled", "none"):
            return "off"
        raise ValueError("skip_zero_x_tiles must be bool or one of: auto/on/off")
    return "on" if bool(skip_zero_x_tiles) else "off"


def resolve_skip_zero_x_tiles_enabled(
    *,
    mode: str,
    j_tile: int,
    ncsf: int,
    min_ncsf: int,
) -> bool:
    if mode == "auto":
        return bool(int(j_tile) < int(ncsf) and int(ncsf) >= int(min_ncsf))
    if mode == "on":
        return bool(int(j_tile) < int(ncsf))
    return False
