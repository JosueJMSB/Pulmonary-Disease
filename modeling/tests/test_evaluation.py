"""La agregacion en dos etapas coincide con un calculo manual, y recall
COPD/Control se verifican contra una matriz de confusion conocida."""

import numpy as np
import pandas as pd
import pytest

from .. import evaluation as ev


def test_two_stage_aggregation_matches_manual_calculation():
    # Paciente A: rec1 (scores 1, 3 -> media 2), rec2 (score 4 -> media 4)
    #   -> paciente A = media(2, 4) = 3.0  (NO es la media directa de los
    #      4 segmentos, que seria 2.6667: cada grabacion pesa igual sin
    #      importar cuantos segmentos aporte).
    # Paciente B: rec3 (scores 0, 0, 6 -> media 2) -> paciente B = 2.0
    segment_scores = pd.DataFrame([
        {"audio_id": "rec1", "patient_uid": "A", "target_label": 1, "score": 1.0},
        {"audio_id": "rec1", "patient_uid": "A", "target_label": 1, "score": 3.0},
        {"audio_id": "rec2", "patient_uid": "A", "target_label": 1, "score": 4.0},
        {"audio_id": "rec3", "patient_uid": "B", "target_label": 0, "score": 0.0},
        {"audio_id": "rec3", "patient_uid": "B", "target_label": 0, "score": 0.0},
        {"audio_id": "rec3", "patient_uid": "B", "target_label": 0, "score": 6.0},
    ])

    recording = ev.aggregate_segment_to_recording(segment_scores)
    recording_score = dict(zip(recording["audio_id"], recording["score"]))
    assert recording_score["rec1"] == pytest.approx(2.0)
    assert recording_score["rec2"] == pytest.approx(4.0)
    assert recording_score["rec3"] == pytest.approx(2.0)

    patient = ev.aggregate_recording_to_patient(recording)
    patient_score = dict(zip(patient["patient_uid"], patient["score"]))
    assert patient_score["A"] == pytest.approx(3.0)
    assert patient_score["B"] == pytest.approx(2.0)

    direct = ev.aggregate_segment_to_patient(segment_scores)
    direct_score = dict(zip(direct["patient_uid"], direct["score"]))
    assert direct_score == pytest.approx(patient_score)


def test_aggregation_rejects_mixed_labels_within_group():
    segment_scores = pd.DataFrame([
        {"audio_id": "rec1", "patient_uid": "A", "target_label": 1, "score": 1.0},
        {"audio_id": "rec1", "patient_uid": "A", "target_label": 0, "score": 2.0},
    ])
    with pytest.raises(ValueError):
        ev.aggregate_segment_to_recording(segment_scores)


def test_patient_metrics_against_known_confusion_matrix():
    # tp=1 (paciente1), fn=1 (paciente2), tn=1 (paciente3), fp=1 (paciente4)
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([2.0, -1.0, -3.0, 0.5])

    metrics = ev.compute_patient_metrics(y_true, y_score, negative_label_name="Healthy", threshold=0.0)

    assert metrics["tp"] == 1 and metrics["fn"] == 1
    assert metrics["tn"] == 1 and metrics["fp"] == 1
    assert metrics["recall_copd"] == pytest.approx(0.5)
    assert metrics["recall_healthy"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    # Concordantes=3, discordantes=1 entre los 2x2 pares positivo-negativo.
    assert metrics["auroc"] == pytest.approx(0.75)


def test_perfect_classifier_has_recall_one():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([5.0, 3.0, -3.0, -5.0])
    metrics = ev.compute_patient_metrics(y_true, y_score, negative_label_name="Normal")
    assert metrics["recall_copd"] == pytest.approx(1.0)
    assert metrics["recall_normal"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc_copd"] == pytest.approx(1.0)


def test_scores_to_predictions_threshold_is_inclusive():
    scores = np.array([-0.1, 0.0, 0.1])
    preds = ev.scores_to_predictions(scores, threshold=0.0)
    assert preds.tolist() == [0, 1, 1]


def test_min_class_recall_picks_the_worse_class():
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 0, 0])  # recall COPD = 2/3, recall Control = 1/1
    assert ev.min_class_recall(y_true, y_pred) == pytest.approx(2 / 3)


def test_bootstrap_confidence_interval_structure_and_point_matches_direct():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([2.0, 1.0, -0.5, -1.0, 0.2, -2.0])
    metric_fn = ev.metric_fn_balanced_accuracy(threshold=0.0)

    result = ev.bootstrap_confidence_interval(
        y_true, y_score, metric_fn, n_resamples=200, confidence=0.95, random_state=0
    )
    assert set(result) == {"point", "lower", "upper", "n_resamples", "confidence"}
    assert result["point"] == pytest.approx(metric_fn(y_true, y_score))
    assert result["lower"] <= result["upper"]
    assert result["n_resamples"] == 200


def test_bootstrap_requires_both_classes():
    y_true = np.array([1, 1, 1])
    y_score = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        ev.bootstrap_confidence_interval(y_true, y_score, ev.metric_fn_auroc())
