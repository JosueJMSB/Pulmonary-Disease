"""Validacion de un archivo de predicciones INT8 recuperado (datos sinteticos)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from optimization import access_registry as ar  # noqa: E402
from optimization import paths  # noqa: E402
from optimization.evaluate_test import load_validated_predictions  # noqa: E402
from optimization.scoring import sigmoid  # noqa: E402


def _test_rows():
    return pd.DataFrame({
        "segment_id": ["s1", "s2", "s3"],
        "audio_id": ["a1", "a1", "a2"],
        "patient_uid": ["p1", "p1", "p2"],
        "target_label": [1, 1, 0],
    })


def _write(path, frame):
    paths.write_csv_atomic(frame, path)
    return path


def _predictions(test, logits):
    return test.assign(logit_int8=logits, score=sigmoid(logits))


def test_valid_file_returns_exact_logits(tmp_path):
    test = _test_rows()
    logits = np.array([0.123456789012345, -2.5, 3.0])
    path = _write(tmp_path / "p.csv", _predictions(test, logits))
    np.testing.assert_array_equal(load_validated_predictions(path, test, "X"), logits)


def test_other_segments_are_rejected(tmp_path):
    test = _test_rows()
    other = test.assign(segment_id=["s1", "s2", "s9"])
    path = _write(tmp_path / "p.csv", _predictions(other, np.zeros(3)))
    with pytest.raises(ar.TestAccessInconsistent, match="segment_id"):
        load_validated_predictions(path, test, "X")


def test_missing_rows_are_rejected(tmp_path):
    test = _test_rows()
    path = _write(tmp_path / "p.csv", _predictions(test.iloc[:2], np.zeros(2)))
    with pytest.raises(ar.TestAccessInconsistent, match="filas"):
        load_validated_predictions(path, test, "X")


def test_inconsistent_score_is_rejected(tmp_path):
    test = _test_rows()
    frame = _predictions(test, np.zeros(3))
    frame.loc[0, "score"] = 0.9
    path = _write(tmp_path / "p.csv", frame)
    with pytest.raises(ar.TestAccessInconsistent, match="sigmoid"):
        load_validated_predictions(path, test, "X")


def test_missing_columns_are_rejected(tmp_path):
    test = _test_rows()
    path = _write(tmp_path / "p.csv", test.assign(logit_int8=0.0))
    with pytest.raises(ar.TestAccessInconsistent, match="faltan columnas"):
        load_validated_predictions(path, test, "X")
