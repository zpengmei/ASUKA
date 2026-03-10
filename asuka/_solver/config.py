from __future__ import annotations

import os


def auto_num_threads() -> int:
    """Return a process-wide thread hint.

    Prefers explicit environment variables, then falls back to the hardware
    core count.
    """

    for key in ("CUGUGA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        val = os.environ.get(key)
        if not val:
            continue
        try:
            n = int(val)
        except Exception:
            continue
        if n > 0:
            return n

    return max(1, int(os.cpu_count() or 1))
