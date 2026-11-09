import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from optimization import scoring  # noqa: E402


def test_sigmoid_is_stable_and_correct():
    z = np.array([-800.0, -2.0, 0.0, 2.0, 800.0])
    out = scoring.sigmoid(z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out[1:4], 1.0 / (1.0 + np.exp(-z[1:4])))
    assert out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(1.0)


def test_decision_uses_greater_or_equal():
    assert scoring.decisions([0.5, 0.4999999], 0.5).tolist() == [1, 0]


def test_two_stage_aggregation_weights_recordings_equally():
    segments = pd.DataFrame({
        "segment_id": ["a0", "a1", "a2", "b0"],
        "audio_id": ["A", "A", "A", "B"],
        "patient_uid": ["P", "P", "P", "P"],
        "target_label": [1, 1, 1, 1],
    })
    probs = [0.9, 0.9, 0.9, 0.1]
    _, recordings, patients = scoring.aggregate(segments, probs)
    assert recordings.set_index("audio_id").loc["A", "score"] == pytest.approx(0.9)
    # Media de grabaciones (0.9 y 0.1), no de segmentos (0.7).
    assert patients.loc[0, "score"] == pytest.approx(0.5)


def test_patient_metrics_reports_negative_recall_alias():
    patients = pd.DataFrame({"patient_uid": ["1", "2", "3", "4"], "target_label": [1, 1, 0, 0],
                             "score": [0.9, 0.2, 0.1, 0.7]})
    metrics = scoring.patient_metrics(patients, "Normal", 0.5)
    assert metrics["recall_copd"] == pytest.approx(0.5)
    assert metrics["recall_negative"] == metrics["recall_normal"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
