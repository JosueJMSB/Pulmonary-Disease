"""Silencio, tono puro y senal constante no deben producir NaN/Inf; las 188
columnas deben ser deterministas en nombre, orden y cantidad."""

import numpy as np
import pytest

from .. import data as dmod
from .. import features as feat


@pytest.fixture(scope="module")
def cfg():
    return dmod.load_config()


def test_feature_names_are_188_and_unique(cfg):
    assert len(feat.FEATURE_NAMES) == 188
    assert len(set(feat.FEATURE_NAMES)) == 188
    assert len(feat.BASE_SERIES_NAMES) == 47  # 39 MFCC/delta/delta2 + 8 descriptores
    for name in feat.BASE_SERIES_NAMES:
        for suffix in feat.STAT_SUFFIXES:
            assert f"{name}_{suffix}" in feat.FEATURE_NAMES


@pytest.mark.parametrize(
    "make_segment",
    [
        lambda n: np.zeros(n, dtype=np.float64),  # silencio
        lambda n: 0.3 * np.sin(2 * np.pi * 220.0 * np.arange(n) / 4000.0),  # tono puro
        lambda n: np.full(n, 0.4, dtype=np.float64),  # senal constante
    ],
    ids=["silencio", "tono_puro", "constante"],
)
def test_no_nan_or_inf_on_degenerate_signals(cfg, make_segment):
    segment_length = int(cfg["acoustic"]["segment_length"])
    segment = make_segment(segment_length)
    vector = feat.extract_segment_features(segment, cfg)
    assert vector.shape == (188,)
    assert np.isfinite(vector).all()


def test_deterministic_and_finite_on_a_real_shaped_tone(cfg):
    segment_length = int(cfg["acoustic"]["segment_length"])
    t = np.arange(segment_length) / cfg["acoustic"]["sample_rate"]
    segment = 0.2 * np.sin(2 * np.pi * 300.0 * t) + 0.05 * np.sin(2 * np.pi * 900.0 * t)

    first = feat.extract_segment_features(segment, cfg)
    second = feat.extract_segment_features(segment, cfg)

    assert np.array_equal(first, second)
    assert np.isfinite(first).all()


def test_wrong_length_raises(cfg):
    with pytest.raises(ValueError):
        feat.extract_segment_features(np.zeros(100), cfg)


def test_batch_matches_single(cfg):
    segment_length = int(cfg["acoustic"]["segment_length"])
    rng = np.random.RandomState(0)
    segments = rng.randn(3, segment_length) * 0.1
    batch = feat.extract_batch_features(segments, cfg)
    for i in range(3):
        single = feat.extract_segment_features(segments[i], cfg)
        assert np.array_equal(batch[i], single)
