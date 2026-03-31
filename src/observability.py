"""Lightweight in-memory observability primitives + Prometheus export."""
from __future__ import annotations

from contextlib import contextmanager
from threading import Lock
from time import perf_counter

try:
    from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
except ModuleNotFoundError:
    class CollectorRegistry:
        def __init__(self) -> None:
            self._metrics: list[_MetricBase] = []

        def register(self, metric: "_MetricBase") -> None:
            self._metrics.append(metric)

    class _MetricChild:
        def __init__(self, metric: "_MetricBase", key: tuple[str, ...]) -> None:
            self._metric = metric
            self._key = key

        def inc(self, amount: float = 1.0) -> None:
            self._metric._inc(self._key, amount)

        def observe(self, value: float) -> None:
            self._metric._observe(self._key, value)

    class _MetricBase:
        metric_type = "gauge"

        def __init__(
            self,
            name: str,
            documentation: str,
            labelnames: list[str] | tuple[str, ...] | None = None,
            registry: CollectorRegistry | None = None,
        ) -> None:
            self.name = name
            self.documentation = documentation
            self.labelnames = list(labelnames or [])
            self._samples: dict[tuple[str, ...], float] = {}
            if registry is not None:
                registry.register(self)

        def labels(self, **labels: str) -> _MetricChild:
            key = tuple(str(labels.get(label, "")) for label in self.labelnames)
            self._samples.setdefault(key, 0.0)
            return _MetricChild(self, key)

        def _format_labels(self, key: tuple[str, ...]) -> str:
            if not self.labelnames:
                return ""
            parts = [f'{label}="{value}"' for label, value in zip(self.labelnames, key)]
            return "{" + ",".join(parts) + "}"

        def _render_metric(self) -> list[str]:
            lines = [
                f"# HELP {self.name} {self.documentation}",
                f"# TYPE {self.name} {self.metric_type}",
            ]
            for key, value in self._samples.items():
                lines.append(f"{self.name}{self._format_labels(key)} {value}")
            return lines

        def _inc(self, key: tuple[str, ...], amount: float) -> None:
            raise NotImplementedError

        def _observe(self, key: tuple[str, ...], value: float) -> None:
            raise NotImplementedError

    class Counter(_MetricBase):
        metric_type = "counter"

        def _inc(self, key: tuple[str, ...], amount: float) -> None:
            self._samples[key] = self._samples.get(key, 0.0) + amount

        def _observe(self, key: tuple[str, ...], value: float) -> None:
            self._inc(key, value)

    class Histogram(_MetricBase):
        metric_type = "histogram"

        def _inc(self, key: tuple[str, ...], amount: float) -> None:
            self._samples[key] = self._samples.get(key, 0.0) + amount

        def _observe(self, key: tuple[str, ...], value: float) -> None:
            self._samples[key] = value

    def generate_latest(registry: CollectorRegistry) -> bytes:
        lines: list[str] = []
        for metric in registry._metrics:
            lines.extend(metric._render_metric())
        if not lines:
            lines.append("# no_metrics 0")
        return ("\n".join(lines) + "\n").encode("utf-8")


# Prometheus registry (isolated, does not pollute default)
PROM_REGISTRY = CollectorRegistry()

REQUESTS_TOTAL = Counter(
    "agentfactory_requests_total",
    "Total HTTP requests",
    ["endpoint", "method"],
    registry=PROM_REGISTRY,
)
REQUEST_DURATION = Histogram(
    "agentfactory_request_duration_seconds",
    "HTTP request duration in seconds",
    ["endpoint"],
    registry=PROM_REGISTRY,
)
LLM_CALLS_TOTAL = Counter(
    "agentfactory_llm_calls_total",
    "Total LLM API calls",
    ["provider", "model"],
    registry=PROM_REGISTRY,
)
LLM_DURATION = Histogram(
    "agentfactory_llm_duration_seconds",
    "LLM call duration in seconds",
    ["provider"],
    registry=PROM_REGISTRY,
)
EMBEDDING_DURATION = Histogram(
    "agentfactory_embedding_duration_seconds",
    "Embedding duration in seconds",
    ["model"],
    registry=PROM_REGISTRY,
)
RETRIEVAL_DURATION = Histogram(
    "agentfactory_retrieval_duration_seconds",
    "Retrieval duration in seconds",
    ["collection"],
    registry=PROM_REGISTRY,
)
INGESTION_TOTAL = Counter(
    "agentfactory_ingestion_total",
    "Documents ingested",
    ["status"],
    registry=PROM_REGISTRY,
)


def prometheus_metrics() -> bytes:
    """Export all Prometheus metrics in text exposition format."""
    return generate_latest(PROM_REGISTRY)


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
