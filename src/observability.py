"""Lightweight in-memory observability primitives."""
from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, int] = {}
        self._timers: dict[str, dict[str, float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, name: str, duration_ms: float) -> None:
        with self._lock:
            item = self._timers.setdefault(
                name,
                {
                    "count": 0,
                    "total_ms": 0.0,
                    "avg_ms": 0.0,
                    "max_ms": 0.0,
                },
            )
            item["count"] += 1
            item["total_ms"] += duration_ms
            item["avg_ms"] = item["total_ms"] / item["count"]
            item["max_ms"] = max(item["max_ms"], duration_ms)

    @contextmanager
    def time_block(self, name: str):
        start = perf_counter()
        try:
            yield
        finally:
            duration_ms = round((perf_counter() - start) * 1000, 2)
            self.observe(name, duration_ms)

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters": dict(self._counters),
                "timers_ms": {k: dict(v) for k, v in self._timers.items()},
            }


metrics = MetricsRegistry()
