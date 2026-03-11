from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any


def with_two_e_metadata(
    out: Any,
    *,
    two_e_backend: str,
    direct_jk_ctx: Any | None = None,
) -> Any:
    """Attach normalized two-electron backend metadata to frontend SCF results."""

    try:
        return _dc_replace(
            out,
            two_e_backend=str(two_e_backend),
            direct_jk_ctx=direct_jk_ctx if direct_jk_ctx is not None else getattr(out, "direct_jk_ctx", None),
        )
    except Exception:
        return out


__all__ = ["with_two_e_metadata"]
