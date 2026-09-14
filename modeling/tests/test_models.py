"""Seleccion de hiperparametros (desempate) y un fold completo de extremo a
extremo sobre datos sinteticos separables -nunca sobre el corpus real-."""

import numpy as np
import pandas as pd
import pytest

from .. import data as dmod
from ..models import svm_rbf as svm_model

GAMMA_ORDER = ["scale", 0.001, 0.01, 0.1]
CRITERIA = ["balanced_accuracy", "macro_f1", "min_class_recall", "neg_C", "gamma_order"]


def test_select_hyperparameters_prefers_higher_balanced_accuracy_first():
    grid = pd.DataFrame([
        {"C": 100, "gamma": 0.1, "balanced_accuracy": 0.95, "macro_f1": 0.90, "min_class_recall": 0.90},
        {"C": 0.1, "gamma": "scale", "balanced_accuracy": 0.70, "macro_f1": 0.70, "min_class_recall": 0.70},
    ])
    selected = svm_model.select_hyperparameters(grid, CRITERIA, GAMMA_ORDER)
    assert selected["C"] == 100


def test_select_hyperparameters_tie_break_prefers_lower_c_then_gamma_order():
    grid = pd.DataFrame([
        {"C": 10, "gamma": 0.01, "balanced_accuracy": 0.8, "macro_f1": 0.8, "min_class_recall": 0.8},
        {"C": 1, "gamma": 0.01, "balanced_accuracy": 0.8, "macro_f1": 0.8, "min_class_recall": 0.8},
        {"C": 1, "gamma": "scale", "balanced_accuracy": 0.8, "macro_f1": 0.8, "min_class_recall": 0.8},
    ])
    selected = svm_model.select_hyperparameters(grid, CRITERIA, GAMMA_ORDER)
    assert selected["C"] == 1
    assert selected["gamma"] == "scale"


def _synthetic_condition(n_per_class: int = 6, n_features: int = 5, seed: int = 0):
    """Dos nubes bien separadas (+2 / -2, ruido 0.3): cualquier SVM-RBF
    correctamente implementada deberia separarlas sin ambiguedad."""
    rng = np.random.RandomState(seed)
    rows, vectors = [], []
    for label in (0, 1):
        base = 2.0 if label == 1 else -2.0
        prefix = "POS" if label == 1 else "NEG"
        for i in range(n_per_class):
            patient_uid = f"{prefix}{i}"
            audio_id = f"{patient_uid}_rec0"
            for seg_idx in range(4):
                rows.append({
                    "patient_uid": patient_uid, "audio_id": audio_id,
                    "segment_id": f"{audio_id}_{seg_idx}", "target_label": label,
                })
                vectors.append(base + rng.randn(n_features) * 0.3)
    return pd.DataFrame(rows), np.asarray(vectors, dtype=np.float64)


def test_run_fold_end_to_end_on_separable_synthetic_data():
    segments, X = _synthetic_condition(n_per_class=6)
    cfg = dmod.load_config()

    train_patients = [f"POS{i}" for i in range(4)] + [f"NEG{i}" for i in range(4)]
    val_patients = ["POS4", "NEG4"]
    test_patients = ["POS5", "NEG5"]

    result = svm_model.run_fold(
        fold_id=0, segments=segments, X=X,
        train_patients=train_patients, val_patients=val_patients, test_patients=test_patients,
        cfg=cfg, negative_label_name="Control", n_jobs=1,
    )

    n_c_grid = len(cfg["svm"]["c_grid"])
    n_gamma_grid = len(cfg["svm"]["gamma_grid"])
    assert len(result.grid_table) == n_c_grid * n_gamma_grid
    assert result.selected_C in cfg["svm"]["c_grid"]
    assert result.selected_gamma in cfg["svm"]["gamma_grid"]

    assert sorted(result.patient_predictions["patient_uid"]) == ["NEG5", "POS5"]
    scores = dict(zip(result.patient_predictions["patient_uid"], result.patient_predictions["score"]))
    # Nubes tan separadas no deberian dejar ambiguedad de signo.
    assert scores["POS5"] > 0
    assert scores["NEG5"] < 0

    assert "recall_copd" in result.test_metrics
    assert "recall_control" in result.test_metrics
    assert result.test_metrics["balanced_accuracy"] == pytest.approx(1.0)

    # Linea base de cordura: una fila por estrategia, evaluada sobre el mismo
    # test que la SVM.
    assert sorted(result.baseline_table["strategy"]) == ["most_frequent", "stratified"]
    assert (result.baseline_table["fold"] == 0).all()
    assert "balanced_accuracy" in result.baseline_table.columns


def test_compute_dummy_baseline_matches_manual_expectation():
    # train+val: 3 COPD, 1 Control -> most_frequent siempre predice COPD.
    train_seg = pd.DataFrame({
        "patient_uid": ["A", "B", "C", "D"],
        "audio_id": ["A_r", "B_r", "C_r", "D_r"],
        "segment_id": ["A_s", "B_s", "C_s", "D_s"],
        "target_label": [1, 1, 1, 0],
    })
    test_seg = pd.DataFrame({
        "patient_uid": ["E", "F"],
        "audio_id": ["E_r", "F_r"],
        "segment_id": ["E_s", "F_s"],
        "target_label": [1, 0],
    })
    baseline = svm_model.compute_dummy_baseline(train_seg, test_seg, "Control", fold_id=0, random_state=0)
    most_frequent = baseline.loc[baseline["strategy"] == "most_frequent"].iloc[0]
    # Predice COPD para ambos pacientes de test: acierta E (COPD), falla F (Control).
    assert most_frequent["recall_copd"] == pytest.approx(1.0)
    assert most_frequent["recall_control"] == pytest.approx(0.0)


def test_fit_final_model_uses_all_patients():
    segments, X = _synthetic_condition(n_per_class=4)
    cfg = dmod.load_config()
    scaler, svm, weights = svm_model.fit_final_model(segments, X, C=1.0, gamma="scale", cfg=cfg)
    assert len(weights) == len(segments)
    predictions = svm.predict(scaler.transform(X))
    assert set(predictions.tolist()) <= {0, 1}
