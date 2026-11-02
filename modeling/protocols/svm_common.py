"""Piezas de la SVM compartidas por los protocolos oof_v1 y fold_aware_v2."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd

from .. import artifacts as art


def _smoke_test_config(cfg: dict) -> dict:
    """Copia de ``cfg`` con la rejilla recortada a un solo (C, gamma)."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["svm"]["c_grid"] = cfg["svm"]["c_grid"][:1]
    cfg2["svm"]["gamma_grid"] = cfg["svm"]["gamma_grid"][:1]
    return cfg2


def _coerce_gamma_column(series: pd.Series) -> pd.Series:
    """Recupera el tipo real de la columna ``gamma`` tras un viaje por CSV.

    La columna mezcla el texto "scale"/"auto" con floats (0.001, 0.01, 0.1).
    Un CSV no distingue tipos por celda: si UNA fila no es numerica, pandas
    guarda la columna ENTERA como texto, y "0.001" vuelve como el string
    ``"0.001"`` en vez del float ``0.001``. Pasado tal cual a ``SVC(gamma=...)``
    eso revienta con ``InvalidParameterError``. Aqui se convierte a float todo
    lo que no sea literalmente "scale" o "auto".
    """
    def _coerce(value):
        if isinstance(value, str) and value not in ("scale", "auto"):
            return float(value)
        return value

    return series.map(_coerce)


def _load_fold_outputs(run_root: Path, dataset: str, condition: str, fold_id: int) -> dict:
    fdir = art.fold_dir(run_root, dataset, condition, fold_id)
    grid_search = pd.read_csv(fdir / "grid_search.csv")
    grid_search["gamma"] = _coerce_gamma_column(grid_search["gamma"])
    return {
        "fold": fold_id,
        "grid_search": grid_search,
        "test_metrics": json.loads((fdir / "test_metrics.json").read_text(encoding="utf-8")),
        "segment_predictions": pd.read_csv(
            fdir / "test_segment_predictions.csv",
            dtype={"audio_id": str, "patient_uid": str, "segment_id": str},
        ),
        "recording_predictions": pd.read_csv(
            fdir / "test_recording_predictions.csv", dtype={"audio_id": str, "patient_uid": str}
        ),
        "patient_predictions": pd.read_csv(
            fdir / "test_patient_predictions.csv", dtype={"patient_uid": str}
        ),
        "baseline_metrics": pd.read_csv(fdir / "baseline_metrics.csv"),
        "selected": json.loads((fdir / "selected_hyperparameters.json").read_text(encoding="utf-8")),
    }
