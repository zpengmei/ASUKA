from __future__ import annotations


def init_csr_pipeline_slots(
    *,
    cp,
    ext,
    enabled: bool,
    n_slots: int,
    initial_capacity: int,
    j_tile: int,
    rs_n_pairs: int,
    csr_data_dtype,
) -> tuple[list[dict[str, object]], object | None]:
    """Initialize optional CSR pipeline slot state for workspace instances."""
    if not bool(enabled):
        return [], None

    n_slots_i = max(0, int(n_slots))
    if n_slots_i < 2:
        return [], None

    base_cap = int(max(1, int(initial_capacity)))
    max_tasks = int(j_tile) * int(rs_n_pairs)
    slots: list[dict[str, object]] = []
    for _ in range(int(n_slots_i)):
        slot: dict[str, object] = {
            "cap": int(base_cap),
            "row_j": cp.empty((base_cap,), dtype=cp.int32),
            "row_k": cp.empty((base_cap,), dtype=cp.int32),
            "indptr": cp.empty((base_cap + 1,), dtype=cp.int64),
            "indices": cp.empty((base_cap,), dtype=cp.int32),
            "data": cp.empty((base_cap,), dtype=csr_data_dtype),
            "overflow": cp.empty((1,), dtype=cp.int32),
            "stream": cp.cuda.Stream(non_blocking=True),
            "inflight_event": None,
            "ws": None,
        }
        if ext is not None and hasattr(ext, "Kernel25Workspace"):
            try:
                slot["ws"] = ext.Kernel25Workspace(int(max_tasks), int(base_cap))
            except Exception:
                slot["ws"] = None
        slots.append(slot)

    if len(slots) < 2:
        return [], None
    return slots, cp.cuda.Stream(non_blocking=True)


def grow_csr_pipeline_slot(
    *,
    cp,
    ext,
    slots: list[dict[str, object]],
    slot_idx: int,
    new_cap: int,
    j_tile: int,
    rs_n_pairs: int,
    csr_data_dtype,
) -> None:
    """Grow one CSR pipeline slot in-place."""
    if int(slot_idx) < 0 or int(slot_idx) >= len(slots):
        raise IndexError("csr pipeline slot index out of range")

    slot = slots[int(slot_idx)]
    cap = int(max(1, int(new_cap)))
    old_cap = int(slot.get("cap", 0))
    if old_cap >= cap:
        return

    slot["cap"] = int(cap)
    slot["row_j"] = cp.empty((cap,), dtype=cp.int32)
    slot["row_k"] = cp.empty((cap,), dtype=cp.int32)
    slot["indptr"] = cp.empty((cap + 1,), dtype=cp.int64)
    slot["indices"] = cp.empty((cap,), dtype=cp.int32)
    slot["data"] = cp.empty((cap,), dtype=csr_data_dtype)
    slot["overflow"] = cp.empty((1,), dtype=cp.int32)

    ws_old = slot.get("ws", None)
    if ws_old is not None:
        try:
            ws_old.release()
        except Exception:
            pass

    ws_new = None
    if ext is not None and hasattr(ext, "Kernel25Workspace"):
        max_tasks = int(j_tile) * int(rs_n_pairs)
        try:
            ws_new = ext.Kernel25Workspace(int(max_tasks), int(cap))
        except Exception:
            ws_new = None
    slot["ws"] = ws_new
