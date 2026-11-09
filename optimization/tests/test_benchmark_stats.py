import math

import numpy as np
import pytest

from optimization import benchmark


def test_latency_stats_median_and_p95():
    values = list(range(1, 101))
    stats = benchmark.latency_stats(values)
    assert stats["n"] == 100
    assert stats["median"] == pytest.approx(50.5)
    assert stats["p95"] == pytest.approx(np.percentile(values, 95))
    assert stats["min"] == 1 and stats["max"] == 100


def test_latency_stats_empty_is_nan():
    stats = benchmark.latency_stats([])
    assert stats["n"] == 0 and math.isnan(stats["median"])


def test_relative_change_pct():
    assert benchmark.relative_change_pct(100.0, 70.0) == pytest.approx(-30.0)
    assert benchmark.relative_change_pct(100.0, 130.0) == pytest.approx(30.0)
    assert math.isnan(benchmark.relative_change_pct(0.0, 5.0))


def test_peak_rss_is_reported_or_none():
    peak = benchmark.peak_rss_bytes()
    assert peak is None or peak > 0
