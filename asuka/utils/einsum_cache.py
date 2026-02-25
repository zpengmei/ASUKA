"""Einsum with contraction-path caching.

``numpy.einsum`` and ``cupy.einsum`` with ``optimize=True`` recompute the
optimal contraction path on every call via ``einsum_path()``.  When the same
subscript string is used repeatedly with arrays of identical shapes and dtypes
(e.g., the Newton ERI build called every CASSCF macro iteration), this path
computation is redundant overhead.

``cached_einsum`` caches the path keyed on ``(subscripts, shapes, dtypes)``
and reuses it for subsequent calls, eliminating the per-call path search.
"""

from __future__ import annotations

import numpy as np

_path_cache: dict[tuple, list] = {}


def cached_einsum(subscripts: str, *operands, xp=None, **kwargs):
    """Einsum with automatic contraction-path caching.

    Parameters
    ----------
    subscripts : str
        Einsum subscript string (e.g., ``"pqQ,uvQ->pquv"``).
    *operands
        Input arrays (NumPy or CuPy).
    xp : module, optional
        Array module (``numpy`` or ``cupy``).  Auto-detected from operands
        if not provided.
    **kwargs
        Passed through to ``xp.einsum`` (excluding ``optimize``).

    Returns
    -------
    Result of ``xp.einsum(subscripts, *operands, optimize=<cached_path>)``.
    """
    if xp is None:
        for op in operands:
            if hasattr(op, "__cuda_array_interface__"):
                try:
                    import cupy as cp
                    xp = cp
                except ImportError:
                    pass
                break
        if xp is None:
            xp = np

    key = (subscripts, tuple((op.shape, op.dtype) for op in operands))
    if key not in _path_cache:
        # Compute and cache the optimal contraction path.
        path_info = np.einsum_path(
            subscripts,
            *(np.empty(op.shape, dtype=op.dtype) for op in operands),
            optimize=True,
        )
        _path_cache[key] = path_info[0]

    # Pop 'optimize' from kwargs in case caller accidentally passes it.
    kwargs.pop("optimize", None)
    return xp.einsum(subscripts, *operands, optimize=_path_cache[key], **kwargs)


def clear_cache() -> None:
    """Clear the cached contraction paths."""
    _path_cache.clear()
