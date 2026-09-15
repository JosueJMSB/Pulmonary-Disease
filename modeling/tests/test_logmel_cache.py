"""Cache Log-Mel de la CNN: forma, finitud, alineacion con segments.csv,
reutilizacion e invalidacion por hash. Solo usa numpy/librosa: no necesita
PyTorch."""

import numpy as np
import pandas as pd
import pytest

from .. import data as dmod

DATASET = "TOY"


@pytest.fixture(scope="module")
def cfg():
    return dmod.load_config(dmod.CNN_CONFIG_PATH)


def _write_toy_dataset(root, n=3, seed=0):
    rng = np.random.RandomState(seed)
    base = root / DATASET
    base.mkdir(parents=True)
    rows = [
        {
            "task_array_index": i, "source_array_index": 100 + i,
            "segment_id": f"S{i}", "audio_id": f"A{i}", "patient_uid": f"P{i}",
            "diagnosis": "COPD" if i % 2 else "Healthy", "target_label": i % 2,
            "target_name": "COPD" if i % 2 else "Control",
            "calibration_patient": False, "dn_reliable": i != 1,
        }
        for i in range(n)
    ]
    pd.DataFrame(rows).to_csv(base / "segments.csv", index=False)
    for branch in dmod.BRANCHES:
        np.save(base / f"segments_{branch}.npy", (0.05 * rng.randn(n, 20000)).astype(np.float32))
    return root


def test_cnn_config_reuses_svm_stft_and_logmel(cfg):
    svm_cfg = dmod.load_config(dmod.DEFAULT_CONFIG_PATH)
    for section in dmod.LOGMEL_CONFIG_SECTIONS:
        assert cfg[section] == svm_cfg[section]


def test_extraction_shape_finiteness_and_alignment(tmp_path, cfg):
    data_root = _write_toy_dataset(tmp_path / "data")
    cache_root = tmp_path / "cache"

    logmel, rows = dmod.extract_or_load_logmel(data_root, cache_root, DATASET, "no_dn", cfg)

    assert logmel.shape == (3, 1, 64, 309)
    assert logmel.dtype == np.float32
    assert np.isfinite(logmel).all()
    assert rows["segment_id"].tolist() == ["S0", "S1", "S2"]
    status, _ = dmod.logmel_cache_status(data_root, cache_root, DATASET, "no_dn", cfg)
    assert status == "valida"


def test_cache_matches_direct_logmel_computation(tmp_path, cfg):
    data_root = _write_toy_dataset(tmp_path / "data")
    logmel, _ = dmod.extract_or_load_logmel(data_root, tmp_path / "cache", DATASET, "no_dn", cfg)

    segment = np.load(data_root / DATASET / "segments_no_dn.npy")[1].astype(np.float64)
    magnitude, _ = dmod.feat.logmel.compute_stft_magnitude(segment, cfg["acoustic"])
    expected = dmod.feat.logmel.compute_logmel_db(magnitude, cfg).astype(np.float32)

    np.testing.assert_array_equal(logmel[1, 0], expected)


def test_valid_cache_is_reused_without_recomputing(tmp_path, cfg, monkeypatch):
    data_root = _write_toy_dataset(tmp_path / "data")
    cache_root = tmp_path / "cache"
    first, _ = dmod.extract_or_load_logmel(data_root, cache_root, DATASET, "no_dn", cfg)

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("la cache valida no deberia recalcular el Log-Mel")

    monkeypatch.setattr(dmod.feat.logmel, "compute_logmel_db", must_not_run)
    second, _ = dmod.extract_or_load_logmel(data_root, cache_root, DATASET, "no_dn", cfg)

    np.testing.assert_array_equal(first, second)


def test_modified_input_invalidates_and_regenerates_cache(tmp_path, cfg):
    data_root = _write_toy_dataset(tmp_path / "data")
    cache_root = tmp_path / "cache"
    first, _ = dmod.extract_or_load_logmel(data_root, cache_root, DATASET, "no_dn", cfg)

    np.save(
        data_root / DATASET / "segments_no_dn.npy",
        (0.1 * np.random.RandomState(1).randn(3, 20000)).astype(np.float32),
    )
    status, detail = dmod.logmel_cache_status(data_root, cache_root, DATASET, "no_dn", cfg)
    assert status == "desactualizada"
    assert "segments_npy_sha256" in detail

    second, _ = dmod.extract_or_load_logmel(data_root, cache_root, DATASET, "no_dn", cfg)
    assert not np.array_equal(first, second)
    assert dmod.logmel_cache_status(data_root, cache_root, DATASET, "no_dn", cfg)[0] == "valida"


def test_condition_logmel_filters_rows_and_keeps_cache_positions(tmp_path, cfg):
    data_root = _write_toy_dataset(tmp_path / "data")
    logmel, rows = dmod.extract_or_load_logmel(data_root, tmp_path / "cache", DATASET, "no_dn", cfg)
    spec = dmod.ConditionSpec(dataset=DATASET, condition="no_dn_reliable", branch="no_dn", dn_reliable_only=True)

    condition = dmod.build_condition_logmel(data_root, spec, logmel, rows)

    assert condition.segments["segment_id"].tolist() == ["S0", "S2"]
    assert condition.segments["cache_row"].tolist() == [0, 2]
    assert condition.logmel is logmel
