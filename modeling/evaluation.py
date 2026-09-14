"""Agregacion segmento->grabacion->paciente, metricas e intervalos bootstrap.

Todo lo que hay aqui es independiente del modelo: recibe puntajes
(``decision_function`` u otro score continuo) y etiquetas, nunca un
estimador. Lo reutilizan tanto la SVM como, mas adelante, la CNN/CRNN.

Convencion de etiqueta: ``1 = COPD`` (positivo), ``0 = Control`` (Healthy en
ICBHI, Normal en Fraiwan). El umbral natural de una SVM sin calibrar es 0:
``decision_function >= 0 -> COPD``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

POSITIVE_LABEL = 1  # COPD
NEGATIVE_LABEL = 0  # Control (Healthy / Normal)
LABEL_ORDER = [NEGATIVE_LABEL, POSITIVE_LABEL]


# ---------------------------------------------------------------------------
# Agregacion segmento -> grabacion -> paciente
# ---------------------------------------------------------------------------

def _aggregate_mean(df: pd.DataFrame, group_col: str, label_col: str, score_col: str) -> pd.DataFrame:
    """Media de ``score_col`` por ``group_col``, exigiendo etiqueta uniforme."""
    labels_per_group = df.groupby(group_col)[label_col].nunique()
    bad = labels_per_group[labels_per_group != 1]
    if len(bad):
        raise ValueError(f"{group_col} con etiquetas mixtas dentro del grupo: {bad.index.tolist()}")

    out = df.groupby(group_col, as_index=False).agg(
        **{label_col: (label_col, "first"), score_col: (score_col, "mean")}
    )
    return out


def aggregate_segment_to_recording(segment_scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Media de segmentos -> un puntaje por ``audio_id``.

    Requiere las columnas ``audio_id``, ``patient_uid``, ``target_label`` y
    ``score_col``. Cada grabacion pesa lo mismo despues de esta media, sin
    importar cuantos segmentos aporto.
    """
    required = {"audio_id", "patient_uid", "target_label", score_col}
    missing = required - set(segment_scores.columns)
    if missing:
        raise ValueError(f"faltan columnas: {sorted(missing)}")

    recording = _aggregate_mean(segment_scores, "audio_id", "target_label", score_col)
    patient_map = segment_scores.drop_duplicates("audio_id").set_index("audio_id")["patient_uid"]
    recording["patient_uid"] = recording["audio_id"].map(patient_map)
    return recording


def aggregate_recording_to_patient(recording_scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Media de grabaciones -> un puntaje por ``patient_uid``.

    Cada paciente pesa lo mismo despues de esta segunda media, sin importar
    cuantas grabaciones aporto: por eso la agregacion es en dos etapas y no
    una media directa de segmento a paciente.
    """
    required = {"patient_uid", "target_label", score_col}
    missing = required - set(recording_scores.columns)
    if missing:
        raise ValueError(f"faltan columnas: {sorted(missing)}")
    return _aggregate_mean(recording_scores, "patient_uid", "target_label", score_col)


def aggregate_segment_to_patient(segment_scores: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Composicion de las dos etapas: segmento -> grabacion -> paciente."""
    recording = aggregate_segment_to_recording(segment_scores, score_col=score_col)
    return aggregate_recording_to_patient(recording, score_col=score_col)


# ---------------------------------------------------------------------------
# Umbral y metricas de seleccion (usadas tambien dentro del fold, en validacion)
# ---------------------------------------------------------------------------

def scores_to_predictions(scores: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """``decision_function >= threshold -> COPD (1)``, si no, Control (0)."""
    scores = np.asarray(scores, dtype=np.float64)
    return (scores >= threshold).astype(np.int64)


def min_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """El menor entre recall COPD y recall Control, para no premiar un
    modelo que acierta una clase a costa de ignorar la otra."""
    recalls = recall_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    return float(np.min(recalls))


def selection_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Los tres primeros criterios de seleccion de hiperparametros."""
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "min_class_recall": min_class_recall(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Metricas completas por paciente
# ---------------------------------------------------------------------------

def compute_patient_metrics(
    y_true: np.ndarray, y_score: np.ndarray, negative_label_name: str, threshold: float = 0.0
) -> dict:
    """Todas las metricas por paciente que pide el plan, en un solo dict.

    ``negative_label_name`` nombra la clave del recall/precision/F1 de la
    clase negativa ("Healthy" en ICBHI, "Normal" en Fraiwan). AUROC y AUPRC
    quedan en NaN -no en error- si el conjunto solo tiene una clase (puede
    pasar en subconjuntos muy pequenos de una figura exploratoria).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if y_true.shape != y_score.shape:
        raise ValueError("y_true y y_score deben tener la misma forma")

    y_pred = scores_to_predictions(y_score, threshold)
    neg_key = negative_label_name.strip().lower()

    tn, fp, fn, tp = sk_confusion_matrix(y_true, y_pred, labels=LABEL_ORDER).ravel()
    recalls = recall_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)

    two_classes = len(np.unique(y_true)) == 2
    auroc = float(roc_auc_score(y_true, y_score)) if two_classes else float("nan")
    auprc = (
        float(average_precision_score(y_true, y_score, pos_label=POSITIVE_LABEL))
        if two_classes
        else float("nan")
    )

    return {
        "n_patients": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_copd": float(recalls[1]),
        f"recall_{neg_key}": float(recalls[0]),
        "precision_copd": float(precisions[1]),
        f"precision_{neg_key}": float(precisions[0]),
        "f1_copd": float(f1s[1]),
        f"f1_{neg_key}": float(f1s[0]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": auroc,
        "auprc_copd": auprc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def classification_report_df(
    y_true: np.ndarray, y_score: np.ndarray, target_names: tuple[str, str], threshold: float = 0.0
) -> pd.DataFrame:
    """``sklearn.metrics.classification_report`` como tabla, no como texto."""
    y_pred = scores_to_predictions(y_score, threshold)
    report = classification_report(
        y_true, y_pred, labels=LABEL_ORDER, target_names=list(target_names),
        output_dict=True, zero_division=0,
    )
    df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    return df


def confusion_matrix_df(y_true: np.ndarray, y_score: np.ndarray, target_names: tuple[str, str], threshold: float = 0.0) -> pd.DataFrame:
    """Matriz de confusion como tabla etiquetada (filas=real, columnas=predicho)."""
    y_pred = scores_to_predictions(y_score, threshold)
    matrix = sk_confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    return pd.DataFrame(
        matrix,
        index=[f"real_{name}" for name in target_names],
        columns=[f"predicho_{name}" for name in target_names],
    )


# ---------------------------------------------------------------------------
# Intervalos de confianza por bootstrap estratificado
# ---------------------------------------------------------------------------

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> dict:
    """IC bootstrap de ``metric_fn(y_true, y_score) -> float``.

    Estratificado por clase: cada remuestreo toma, con reemplazo, tantos
    pacientes positivos como habia originalmente y tantos negativos como
    habia originalmente, y los junta. Evita que una remuestra pierda una
    clase por azar cuando una de las dos es pequena (por ejemplo, los 7 COPD
    de Fraiwan).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_idx = np.flatnonzero(y_true == POSITIVE_LABEL)
    neg_idx = np.flatnonzero(y_true == NEGATIVE_LABEL)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("el bootstrap estratificado requiere ambas clases presentes")

    point = float(metric_fn(y_true, y_score))
    rng = np.random.RandomState(random_state)
    values = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sampled_pos = rng.choice(pos_idx, size=pos_idx.size, replace=True)
        sampled_neg = rng.choice(neg_idx, size=neg_idx.size, replace=True)
        idx = np.concatenate([sampled_pos, sampled_neg])
        values[i] = metric_fn(y_true[idx], y_score[idx])

    alpha = 1.0 - confidence
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return {
        "point": point, "lower": lower, "upper": upper,
        "n_resamples": int(n_resamples), "confidence": float(confidence),
    }


def metric_fn_balanced_accuracy(threshold: float = 0.0):
    def _fn(y_true, y_score):
        return balanced_accuracy_score(y_true, scores_to_predictions(y_score, threshold))
    return _fn


def metric_fn_auroc():
    def _fn(y_true, y_score):
        return roc_auc_score(y_true, y_score)
    return _fn


def metric_fn_auprc():
    def _fn(y_true, y_score):
        return average_precision_score(y_true, y_score, pos_label=POSITIVE_LABEL)
    return _fn


def metric_fn_recall_copd(threshold: float = 0.0):
    def _fn(y_true, y_score):
        y_pred = scores_to_predictions(y_score, threshold)
        return recall_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)[1]
    return _fn


def metric_fn_recall_negative(threshold: float = 0.0):
    def _fn(y_true, y_score):
        y_pred = scores_to_predictions(y_score, threshold)
        return recall_score(y_true, y_pred, labels=LABEL_ORDER, average=None, zero_division=0)[0]
    return _fn
