from __future__ import annotations

from typing import Any, Sequence


def matvec_cuda_ws_cache_touch(solver: Any, key: Any) -> None:
    if key in solver._matvec_cuda_ws_cache_lru:
        solver._matvec_cuda_ws_cache_lru.move_to_end(key, last=True)
    else:
        solver._matvec_cuda_ws_cache_lru[key] = None


def matvec_cuda_ws_cache_get(solver: Any, key: Any) -> Any:
    ws = solver._matvec_cuda_ws_cache.get(key)
    if ws is None:
        solver._matvec_cuda_ws_cache_misses += 1
        return None
    solver._matvec_cuda_ws_cache_hits += 1
    matvec_cuda_ws_cache_touch(solver, key)
    return ws


def matvec_cuda_ws_cache_drop(solver: Any, key: Any) -> None:
    ws = solver._matvec_cuda_ws_cache.pop(key, None)
    old_size = int(solver._matvec_cuda_ws_cache_sizes.pop(key, 0))
    solver._matvec_cuda_ws_cache_lru.pop(key, None)
    if old_size > 0:
        solver._matvec_cuda_ws_cache_bytes = max(
            0,
            int(solver._matvec_cuda_ws_cache_bytes) - int(old_size),
        )
    if ws is not None:
        solver._release_matvec_cuda_workspace(ws)


def matvec_cuda_ws_cache_enforce_budget(
    solver: Any,
    *,
    keep_keys: Sequence[Any] = (),
) -> None:
    budget = int(solver._matvec_cuda_ws_cache_budget_bytes)
    if budget <= 0:
        return
    keep = set(keep_keys)
    if int(solver._matvec_cuda_ws_cache_bytes) <= budget:
        return
    while int(solver._matvec_cuda_ws_cache_bytes) > budget and len(solver._matvec_cuda_ws_cache_lru) > 0:
        if all(k in keep for k in solver._matvec_cuda_ws_cache_lru.keys()):
            break
        victim = next(iter(solver._matvec_cuda_ws_cache_lru.keys()))
        if victim in keep:
            solver._matvec_cuda_ws_cache_lru.move_to_end(victim, last=True)
            continue
        matvec_cuda_ws_cache_drop(solver, victim)
        solver._matvec_cuda_ws_cache_evictions += 1


def matvec_cuda_ws_cache_put(
    solver: Any,
    key: Any,
    ws: Any,
    *,
    keep_keys: Sequence[Any] = (),
) -> None:
    if ws is None:
        return
    old_ws = solver._matvec_cuda_ws_cache.get(key)
    old_size = int(solver._matvec_cuda_ws_cache_sizes.get(key, 0))
    if old_ws is not None and old_ws is not ws:
        matvec_cuda_ws_cache_drop(solver, key)
        old_size = 0

    solver._matvec_cuda_ws_cache[key] = ws
    est_size = solver._estimate_matvec_cuda_workspace_bytes(ws)
    solver._matvec_cuda_ws_cache_sizes[key] = int(est_size)
    delta = int(est_size) - int(old_size)
    if delta != 0:
        solver._matvec_cuda_ws_cache_bytes = max(0, int(solver._matvec_cuda_ws_cache_bytes) + int(delta))
    matvec_cuda_ws_cache_touch(solver, key)
    matvec_cuda_ws_cache_enforce_budget(solver, keep_keys=tuple(keep_keys))


def release_matvec_cuda_ws_cache(solver: Any) -> int:
    total = 0
    try:
        for ws in list(getattr(solver, "_matvec_cuda_ws_cache", {}).values()):
            total += int(solver._estimate_matvec_cuda_workspace_bytes(ws))
    except Exception:
        total = 0
    try:
        for key in list(getattr(solver, "_matvec_cuda_ws_cache", {}).keys()):
            matvec_cuda_ws_cache_drop(solver, key)
    except Exception:
        # Best-effort: if the cache structure isn't as expected, fall back
        # to clearing references without raising.
        try:
            cache = getattr(solver, "_matvec_cuda_ws_cache", None)
            if isinstance(cache, dict):
                for ws in list(cache.values()):
                    solver._release_matvec_cuda_workspace(ws)
                cache.clear()
        except Exception:
            pass
    return int(max(0, total))


def configure_matvec_cuda_ws_cache(
    solver: Any,
    *,
    cp_mod: Any | None,
    hard_cap_gib: float,
    fraction: float | None,
    normalize_ws_cache_fraction_fn: Any,
    resolve_budget_bytes_fn: Any,
) -> int:
    if fraction is not None:
        solver.matvec_cuda_ws_cache_fraction = normalize_ws_cache_fraction_fn(fraction)
    solver._matvec_cuda_ws_cache_budget_bytes = resolve_budget_bytes_fn(
        cp_mod=cp_mod,
        hard_cap_gib=float(hard_cap_gib),
        fraction=solver.matvec_cuda_ws_cache_fraction,
    )
    matvec_cuda_ws_cache_enforce_budget(solver)
    return int(solver._matvec_cuda_ws_cache_budget_bytes)


def matvec_cuda_ws_cache_profile(solver: Any) -> dict[str, Any]:
    return {
        "matvec_cuda_ws_cache_entries": int(len(solver._matvec_cuda_ws_cache)),
        "matvec_cuda_ws_cache_bytes": int(solver._matvec_cuda_ws_cache_bytes),
        "matvec_cuda_ws_cache_budget_bytes": int(solver._matvec_cuda_ws_cache_budget_bytes),
        "matvec_cuda_ws_cache_fraction": float(solver.matvec_cuda_ws_cache_fraction),
        "matvec_cuda_ws_cache_hits": int(solver._matvec_cuda_ws_cache_hits),
        "matvec_cuda_ws_cache_misses": int(solver._matvec_cuda_ws_cache_misses),
        "matvec_cuda_ws_cache_evictions": int(solver._matvec_cuda_ws_cache_evictions),
    }
