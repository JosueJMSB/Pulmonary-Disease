"""Criterios de comparacion entre versiones del modelo.

Hay dos criterios DISTINTOS:

- ``fp32_parity``: PyTorch frente a ONNX FP32 (y el flujo WAV frente a los
  segmentos almacenados). Tolerancia ESTRICTA y bloqueante sobre logits y
  probabilidades. Los cambios de decision se reportan aparte; con la
  tolerancia estricta solo pueden darse en segmentos con puntuacion
  practicamente igual al umbral.
- ``int8_report``: ONNX FP32 frente a ONNX INT8. NO tiene tolerancia de
  aprobacion: registra las diferencias de puntuacion y los cambios de
  decision. La aceptacion del INT8 se decide despues, por paciente, con el
  criterio de ``[acceptance]`` (ver ``acceptance_check``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def difference_stats(a, b) -> dict:
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if diff.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "n": int(diff.size),
        "mean": float(diff.mean()),
        "median": float(np.median(diff)),
        "p95": float(np.percentile(diff, 95)),
        "max": float(diff.max()),
    }


def decision_changes(ids, scores_a, scores_b, threshold: float, label_a: str, label_b: str) -> pd.DataFrame:
    """Casos en que ``score >= umbral`` cambia entre dos versiones."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    changed = (a >= threshold) != (b >= threshold)
    frame = pd.DataFrame({
        "id": np.asarray(ids)[changed],
        f"score_{label_a}": a[changed],
        f"score_{label_b}": b[changed],
        f"decision_{label_a}": (a[changed] >= threshold).astype(int),
        f"decision_{label_b}": (b[changed] >= threshold).astype(int),
        f"distance_to_threshold_{label_a}": np.abs(a[changed] - threshold),
    })
    return frame.reset_index(drop=True)


def fp32_parity(logits_ref, logits_new, probs_ref, probs_new, ids, threshold: float,
                max_abs_logit: float, max_abs_prob: float, label_ref: str = "torch",
                label_new: str = "onnx_fp32") -> tuple[dict, pd.DataFrame]:
    logit_stats = difference_stats(logits_ref, logits_new)
    prob_stats = difference_stats(probs_ref, probs_new)
    changes = decision_changes(ids, probs_ref, probs_new, threshold, label_ref, label_new)
    passed = (
        logit_stats["n"] > 0
        and logit_stats["max"] <= max_abs_logit
        and prob_stats["max"] <= max_abs_prob
    )
    report = {
        "criterion": "tolerancia estricta (bloqueante)",
        "reference": label_ref,
        "candidate": label_new,
        "tolerance": {"max_abs_logit": max_abs_logit, "max_abs_prob": max_abs_prob},
        "abs_logit_diff": logit_stats,
        "abs_prob_diff": prob_stats,
        "n_decision_changes": int(len(changes)),
        "status": "PASS" if passed else "FAIL",
    }
    return report, changes


def int8_report(probs_fp32, probs_int8, logits_fp32, logits_int8, ids, threshold: float) -> tuple[dict, pd.DataFrame]:
    changes = decision_changes(ids, probs_fp32, probs_int8, threshold, "fp32", "int8")
    n = int(np.asarray(probs_fp32).size)
    report = {
        "criterion": "registro de diferencias (sin tolerancia de aprobacion)",
        "abs_prob_diff": difference_stats(probs_fp32, probs_int8),
        "abs_logit_diff": difference_stats(logits_fp32, logits_int8) if logits_fp32 is not None else None,
        "n_items": n,
        "n_decision_changes": int(len(changes)),
        "concordance": float(1.0 - len(changes) / n) if n else float("nan"),
    }
    return report, changes


def acceptance_check(metrics_fp32: dict, metrics_int8: dict, acceptance: dict) -> dict:
    """Criterio de aceptacion en DESARROLLO, fijado de antemano en
    ``config.toml [acceptance]``. Perdida = FP32 - INT8 (positiva si INT8 empeora)."""
    checks = []
    for metric, key in (("balanced_accuracy", "max_drop_balanced_accuracy"),
                        ("recall_copd", "max_drop_recall_copd"),
                        ("recall_negative", "max_drop_recall_negative")):
        drop = float(metrics_fp32[metric]) - float(metrics_int8[metric])
        limit = float(acceptance[key])
        checks.append({
            "metric": metric,
            "fp32": float(metrics_fp32[metric]),
            "int8": float(metrics_int8[metric]),
            "drop": drop,
            "max_drop": limit,
            "passed": bool(drop <= limit + 1e-12),
        })
    return {
        "criterion": "perdida maxima INT8 frente a FP32 en pacientes de desarrollo",
        "note": ("Estas metricas solo comparan versiones: los pacientes de desarrollo entrenaron el modelo "
                 "final, asi que no estiman el desempeno."),
        "checks": checks,
        "accepted": all(c["passed"] for c in checks),
    }
