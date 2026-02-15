from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RowStats:
    """Lightweight time/counter aggregator for row-oracle profiling.

    Intended for coarse breakdowns driven by `bench_oracle --breakdown`.
    Keep this module dependency-free and low overhead.
    """

    times: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def add_time(self, key: str, dt: float) -> None:
        self.times[key] = float(self.times.get(key, 0.0)) + float(dt)

    def inc(self, key: str, n: int = 1) -> None:
        self.counts[key] = int(self.counts.get(key, 0)) + int(n)

    @contextmanager
    def timer(self, key: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add_time(key, time.perf_counter() - t0)

