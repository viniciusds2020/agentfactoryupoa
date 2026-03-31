"""Tests for observability module: in-memory MetricsRegistry + Prometheus export."""
from __future__ import annotations

from src.observability import (
    MetricsRegistry,
    PROM_REGISTRY,
    REQUESTS_TOTAL,
    LLM_CALLS_TOTAL,
    INGESTION_TOTAL,
    EMBEDDING_DURATION,
    prometheus_metrics,
)


class TestMetricsRegistry:
    def test_increment_counter(self):
        m = MetricsRegistry()
        m.increment("test.counter")
        m.increment("test.counter", 5)
        snap = m.snapshot()
        assert snap["counters"]["test.counter"] == 6

    def test_observe_timer(self):
        m = MetricsRegistry()
        m.observe("test.timer", 100.0)
        m.observe("test.timer", 200.0)
        snap = m.snapshot()
        timer = snap["timers_ms"]["test.timer"]
        assert timer["count"] == 2
        assert timer["total_ms"] == 300.0
        assert timer["avg_ms"] == 150.0
        assert timer["max_ms"] == 200.0

    def test_time_block_records_duration(self):
        m = MetricsRegistry()
        with m.time_block("test.block"):
            _ = sum(range(100))
        snap = m.snapshot()
        assert "test.block" in snap["timers_ms"]
        assert snap["timers_ms"]["test.block"]["count"] == 1

    def test_snapshot_is_isolated(self):
        m = MetricsRegistry()
        m.increment("a")
        snap = m.snapshot()
        m.increment("a")
        assert snap["counters"]["a"] == 1  # snapshot is a copy


class TestPrometheusExport:
    def test_prometheus_metrics_returns_bytes(self):
        output = prometheus_metrics()
        assert isinstance(output, bytes)

    def test_prometheus_metrics_contains_prefix(self):
        # Trigger at least one metric so there's output
        INGESTION_TOTAL.labels(status="test").inc()
        output = prometheus_metrics().decode()
        assert "agentfactory_" in output

    def test_counter_labels_work(self):
        REQUESTS_TOTAL.labels(endpoint="/test", method="GET").inc()
        output = prometheus_metrics().decode()
        assert "agentfactory_requests_total" in output

    def test_histogram_labels_work(self):
        EMBEDDING_DURATION.labels(model="test-model").observe(0.5)
        output = prometheus_metrics().decode()
        assert "agentfactory_embedding_duration_seconds" in output


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_200(self):
        from fastapi.testclient import TestClient
        import app as app_module

        client = TestClient(app_module.app)
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        assert "agentfactory_" in resp.text
