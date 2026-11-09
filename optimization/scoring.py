"""Puntuaciones, agregacion y metricas por paciente, sin PyTorch.

Reutiliza ``modeling/evaluation.py`` (numpy, pandas y scikit-learn): media por
grabacion, media por paciente y decision ``score >= umbral``, exactamente
como ``modeling.models.cnn.evaluate_patients``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modeling import evaluation as ev

# Las mismas ocho metricas (y en el mismo orden) que publica la evaluacion
# final del modelo base (``modeling/final_training/core.py::SUMMARY_METRICS``).
SUMMARY_METRICS = (
    "accuracy", "balanced_accuracy", "recall_copd", "recall_negative",
    "macro_f1", "min_class_recall", "auroc", "auprc_copd",
)


def sigmoid(z) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def decisions(scores, threshold: float) -> np.ndarray:
    return ev.scores_to_predictions(np.asarray(scores, dtype=np.float64), threshold)


def aggregate(segments: pd.DataFrame, probabilities) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Segmento -> grabacion -> paciente. ``segments`` debe tener
    ``audio_id``, ``patient_uid`` y ``target_label``."""
    scored = segments.assign(score=np.asarray(probabilities, dtype=np.float64))
    recordings = ev.aggregate_segment_to_recording(scored)
    patients = ev.aggregate_recording_to_patient(recordings)
    return scored, recordings, patients


def patient_metrics(patients: pd.DataFrame, negative_label_name: str, threshold: float) -> dict:
    metrics = ev.compute_patient_metrics(
        patients["target_label"].to_numpy(), patients["score"].to_numpy(),
        negative_label_name=negative_label_name, threshold=threshold,
    )
    neg_key = f"recall_{negative_label_name.strip().lower()}"
    metrics["recall_negative"] = metrics[neg_key]
    metrics["min_class_recall"] = float(min(metrics["recall_copd"], metrics[neg_key]))
    return metrics


def bootstrap_table(patients: pd.DataFrame, threshold: float, n_resamples: int,
                    confidence: float, random_state: int) -> pd.DataFrame:
    """IC por bootstrap estratificado de las ocho metricas del resultado base,
    con la misma configuracion que la evaluacion final
    (``selection_snapshot.json``)."""
    y_true = patients["target_label"].to_numpy()
    y_score = patients["score"].to_numpy()
    metric_fns = {
        "accuracy": ev.metric_fn_accuracy(threshold),
        "balanced_accuracy": ev.metric_fn_balanced_accuracy(threshold),
        "recall_copd": ev.metric_fn_recall_copd(threshold),
        "recall_negative": ev.metric_fn_recall_negative(threshold),
        "macro_f1": ev.metric_fn_macro_f1(threshold),
        "min_class_recall": ev.metric_fn_min_class_recall(threshold),
        "auroc": ev.metric_fn_auroc(),
        "auprc_copd": ev.metric_fn_auprc(),
    }
    assert tuple(metric_fns) == SUMMARY_METRICS
    rows = []
    for name, fn in metric_fns.items():
        try:
            ci = ev.bootstrap_confidence_interval(
                y_true, y_score, fn, n_resamples=n_resamples, confidence=confidence, random_state=random_state,
            )
            rows.append({"metric": name, **ci})
        except ValueError as exc:
            rows.append({"metric": name, "point": float("nan"), "lower": float("nan"), "upper": float("nan"),
                         "n_resamples": int(n_resamples), "confidence": float(confidence), "error": str(exc)})
    return pd.DataFrame(rows)


def confusion_table(patients: pd.DataFrame, negative_label_name: str, threshold: float) -> pd.DataFrame:
    return ev.confusion_matrix_df(
        patients["target_label"].to_numpy(), patients["score"].to_numpy(),
        (negative_label_name, "COPD"), threshold,
    )
