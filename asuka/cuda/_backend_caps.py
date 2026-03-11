from __future__ import annotations

from typing import Any


def has_cuda_ext(ext: Any) -> bool:
    return ext is not None


def has_ext_symbol(ext: Any, symbol: str) -> bool:
    return ext is not None and hasattr(ext, str(symbol))


def has_epq_table_device_build(ext: Any) -> bool:
    """Return True if the extension exposes epq-table device-build entrypoints."""
    return has_ext_symbol(ext, "epq_contribs_many_count_allpairs_inplace_device")


def has_epq_table_device_build_recompute(ext: Any) -> bool:
    """Return True if the extension exposes EPQ recompute device-build entrypoints."""
    return has_ext_symbol(ext, "epq_contribs_many_count_allpairs_recompute_inplace_device")


def has_t_from_epq_table_device_build(ext: Any) -> bool:
    """Return True if the extension exposes T-from-epq-table entrypoint."""
    return has_ext_symbol(ext, "build_t_from_epq_table_inplace_device")


def has_epq_table_gather_apply_device(ext: Any) -> bool:
    """Return True if the extension exposes destination-gather EPQ apply entrypoint."""
    return has_ext_symbol(ext, "apply_g_flat_gather_epq_table_inplace_device")


def has_build_w_from_epq_transpose_range_mm_scaled(ext: Any) -> bool:
    """Return True if the extension exposes mm-scaled EPQ transpose-range W builder."""
    return has_ext_symbol(ext, "build_w_from_epq_transpose_range_mm_scaled_inplace_device")


def has_build_w_from_epq_transpose_range_mm(ext: Any) -> bool:
    """Return True if the extension exposes mm EPQ transpose-range W builder."""
    return has_ext_symbol(ext, "build_w_from_epq_transpose_range_mm_inplace_device")


def device_info(ext: Any) -> dict[str, object]:
    if ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    return dict(ext.device_info())


def mem_info(ext: Any) -> dict[str, int]:
    if ext is None:
        raise RuntimeError("CUDA extension not available; build with python -m asuka.build.guga_cuda_ext")
    return dict(ext.mem_info())
