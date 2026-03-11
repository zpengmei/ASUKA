from __future__ import annotations


def resolve_epq_stream_double_buffer_mode(epq_stream_double_buffer, env_epq_stream_db: str) -> str:
    """Resolve EPQ stream double-buffer mode to one of: auto|on|off."""
    if epq_stream_double_buffer is None:
        env_v = str(env_epq_stream_db or "").strip().lower()
        if env_v in ("", "auto"):
            return "auto"
        if env_v in ("1", "true", "yes", "on"):
            return "on"
        if env_v in ("0", "false", "no", "off"):
            return "off"
        raise ValueError("ASUKA_CUGUGA_EPQ_STREAM_DOUBLE_BUFFER must be auto/0/1")
    if isinstance(epq_stream_double_buffer, str):
        mode = epq_stream_double_buffer.strip().lower()
        if mode not in ("auto", "on", "off"):
            raise ValueError("epq_stream_double_buffer must be bool or one of: auto/on/off")
        return mode
    return "on" if bool(epq_stream_double_buffer) else "off"


def resolve_epq_stream_panic_mode(epq_stream_panic_mode, env_epq_stream_panic: str) -> str:
    """Resolve EPQ stream panic mode to one of: auto|on|off."""
    if epq_stream_panic_mode is None:
        env_v = str(env_epq_stream_panic or "").strip().lower()
        if env_v in ("", "auto"):
            return "auto"
        if env_v in ("1", "true", "yes", "on"):
            return "on"
        if env_v in ("0", "false", "no", "off"):
            return "off"
        raise ValueError("ASUKA_CUGUGA_EPQ_STREAM_PANIC_MODE must be auto/0/1")
    if isinstance(epq_stream_panic_mode, str):
        mode = epq_stream_panic_mode.strip().lower()
        if mode not in ("auto", "on", "off"):
            raise ValueError("epq_stream_panic_mode must be bool or one of: auto/on/off")
        return mode
    return "on" if bool(epq_stream_panic_mode) else "off"


def resolve_epq_stream_use_recompute(epq_stream_use_recompute, env_recompute: str):
    """Resolve EPQ streaming recompute policy to bool or 'auto'."""
    if epq_stream_use_recompute is None:
        env_v = str(env_recompute or "").strip().lower()
        if env_v in ("", "auto"):
            return "auto"
        if env_v in ("1", "true", "yes", "on"):
            return True
        if env_v in ("0", "false", "no", "off"):
            return False
        raise ValueError("ASUKA_CUGUGA_EPQ_STREAM_RECOMPUTE must be auto/0/1")
    if isinstance(epq_stream_use_recompute, str):
        mode = epq_stream_use_recompute.strip().lower()
        if mode != "auto":
            raise ValueError("epq_stream_use_recompute must be bool or 'auto'")
        return "auto"
    return bool(epq_stream_use_recompute)


def resolve_apply_warp_coop(apply_warp_coop, env_apply_warp_coop: str):
    """Resolve apply warp-coop mode to bool or 'auto'."""
    if apply_warp_coop is None:
        env_v = str(env_apply_warp_coop or "").strip().lower()
        if env_v in ("", "auto"):
            return "auto"
        if env_v in ("1", "true", "yes", "on"):
            return True
        if env_v in ("0", "false", "no", "off"):
            return False
        raise ValueError("ASUKA_CUGUGA_APPLY_WARP_COOP must be auto/0/1")
    if isinstance(apply_warp_coop, str):
        mode = apply_warp_coop.strip().lower()
        if mode not in ("auto",):
            raise ValueError("apply_warp_coop must be bool or 'auto'")
        return "auto"
    return bool(apply_warp_coop)


def resolve_check_overflow_mode(check_overflow_mode) -> int:
    """Resolve check_overflow_mode to: 0 (none), 1 (deferred), 2 (per-stage)."""
    if isinstance(check_overflow_mode, str):
        mode = check_overflow_mode.strip().lower()
        if mode in ("none", "off", "0", "false", "no"):
            return 0
        if mode in ("deferred", "1", "true", "yes", "on"):
            return 1
        if mode in ("per-stage", "per_stage", "stage", "staged", "2"):
            return 2
        raise ValueError("check_overflow_mode must be one of: none, deferred, per-stage")
    out = int(check_overflow_mode)
    if out not in (0, 1, 2):
        raise ValueError("check_overflow_mode must be 0 (none), 1 (deferred), or 2 (per-stage)")
    return out
