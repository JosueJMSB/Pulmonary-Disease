"""Carga verificada de la corrida final y de los segmentos de la etapa final.

- Los checkpoints, manifiestos y predicciones publicadas se leen de
  ``<runs_root>/final_training/<run_id>/datasets/<dataset>/<condition>/``.
- Los segmentos se leen de ``<data_root>/<dataset>/final/`` y, ANTES de
  usarlos, se comprueba que ``segments.csv``, ``segments_<rama>.npy``,
  ``manifest.json`` y ``preprocessing_params.json`` tienen exactamente los
  hashes registrados en ``model_manifest.json`` del modelo final.
- El log-mel se calcula SOLO para las filas pedidas (``role == "train"`` en
  desarrollo; ``role == "test"`` unicamente desde ``evaluate_test.py``). No se
  usa la cache de log-mel del entrenamiento.

Este modulo no importa PyTorch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio_pipeline as ap
from .bundle import sha256_file

STAGE = "final"
ROLE_DEVELOPMENT = "train"
ROLE_TEST = "test"


class FinalDataError(RuntimeError):
    """Los datos o artefactos no coinciden con los del entrenamiento final."""


@dataclass
class FinalModel:
    dataset: str
    condition: str
    branch: str
    model_dir: Path
    manifest: dict
    selection_snapshot: dict

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "final_model.pt"

    @property
    def manifest_path(self) -> Path:
        return self.model_dir / "model_manifest.json"

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])


def load_final_model(final_run_dir: Path, entry: dict) -> FinalModel:
    model_dir = Path(final_run_dir) / "datasets" / entry["name"] / entry["condition"]
    manifest_path = model_dir / "model_manifest.json"
    snapshot_path = model_dir / "selection_snapshot.json"
    for path in (model_dir / "final_model.pt", manifest_path, snapshot_path, model_dir / "_SUCCESS"):
        if not path.is_file():
            raise FinalDataError(f"falta {path}: la corrida final no esta completa")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != entry["name"] or manifest.get("condition") != entry["condition"]:
        raise FinalDataError(f"{manifest_path} no corresponde a {entry['name']}/{entry['condition']}")
    if manifest.get("branch") != entry["branch"]:
        raise FinalDataError(f"{manifest_path}: rama {manifest.get('branch')!r}, se esperaba {entry['branch']!r}")
    return FinalModel(entry["name"], entry["condition"], entry["branch"], model_dir, manifest, snapshot)


def verify_checkpoint(final: FinalModel) -> str:
    expected = final.manifest["hashes"]["final_model_sha256"]
    actual = sha256_file(final.checkpoint_path)
    if actual != expected:
        raise FinalDataError(
            f"{final.checkpoint_path}: hash {actual[:12]}... distinto del registrado ({expected[:12]}...)"
        )
    return actual


def final_data_dir(data_root: Path, dataset: str) -> Path:
    return Path(data_root) / dataset / STAGE


def verify_final_inputs(data_root: Path, final: FinalModel) -> dict:
    """Compara los hashes de la etapa final con ``model_manifest.json``."""
    root = final_data_dir(data_root, final.dataset)
    prefix = f"{final.dataset}/{STAGE}"
    expected = final.manifest["hashes"]["final_data"]
    files = {
        f"{prefix}/segments.csv": root / "segments.csv",
        f"{prefix}/segments_{final.branch}.npy": root / f"segments_{final.branch}.npy",
        f"{prefix}/manifest.json": root / "manifest.json",
        f"{prefix}/preprocessing_params.json": root / "preprocessing_params.json",
    }
    actual = {}
    for key, path in files.items():
        if not path.is_file():
            raise FinalDataError(f"no existe {path}")
        if key not in expected:
            raise FinalDataError(f"model_manifest.json no registra el hash de {key}")
        actual[key] = sha256_file(path)
        if actual[key] != expected[key]:
            raise FinalDataError(
                f"{path}: hash {actual[key][:12]}... distinto del usado en el entrenamiento final "
                f"({expected[key][:12]}...)"
            )
    return actual


def load_preprocessing_params(data_root: Path, dataset: str) -> dict:
    path = final_data_dir(data_root, dataset) / "preprocessing_params.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_segments(data_root: Path, dataset: str) -> pd.DataFrame:
    path = final_data_dir(data_root, dataset) / "segments.csv"
    segments = pd.read_csv(path, dtype={"audio_id": str, "patient_uid": str, "segment_id": str})
    required = {"task_array_index", "segment_id", "audio_id", "patient_uid", "target_label", "role", "start_sample"}
    missing = required - set(segments.columns)
    if missing:
        raise FinalDataError(f"{path}: faltan columnas {sorted(missing)}")
    segments = segments.sort_values("task_array_index").reset_index(drop=True)
    if not np.array_equal(segments["task_array_index"].to_numpy(), np.arange(len(segments))):
        raise FinalDataError(f"{path}: task_array_index no es 0..n-1 consecutivo")
    segments["target_label"] = segments["target_label"].astype(np.int64)
    return segments


def select_role(segments: pd.DataFrame, role: str) -> pd.DataFrame:
    if role not in (ROLE_DEVELOPMENT, ROLE_TEST):
        raise ValueError(f"rol desconocido: {role!r}")
    selected = segments.loc[segments["role"] == role].reset_index(drop=True)
    if selected.empty:
        raise FinalDataError(f"no hay segmentos con role == {role!r}")
    return selected


def branch_array(data_root: Path, dataset: str, branch: str) -> np.ndarray:
    return np.load(final_data_dir(data_root, dataset) / f"segments_{branch}.npy", mmap_mode="r")


def logmel_for_rows(data_root: Path, dataset: str, branch: str, rows: pd.DataFrame,
                    acoustic_cfg: dict, logmel_cfg: dict) -> np.ndarray:
    """Log-mel solo de ``rows`` (por ``task_array_index``)."""
    array = branch_array(data_root, dataset, branch)
    indices = rows["task_array_index"].to_numpy(dtype=np.int64)
    if indices.size and (indices.max() >= array.shape[0] or indices.min() < 0):
        raise FinalDataError("task_array_index fuera del rango del .npy")
    segments = np.asarray(array[indices], dtype=np.float32)
    return ap.logmel_from_segments(segments, acoustic_cfg, logmel_cfg)


def stored_segments_for_rows(data_root: Path, dataset: str, branch: str, rows: pd.DataFrame) -> np.ndarray:
    array = branch_array(data_root, dataset, branch)
    return np.asarray(array[rows["task_array_index"].to_numpy(dtype=np.int64)], dtype=np.float32)


def sample_patients_stratified(rows: pd.DataFrame, n_segments: int, seed: int) -> pd.DataFrame:
    """Subconjunto reproducible repartido entre pacientes: se baraja cada
    paciente con la semilla y se toman segmentos por turnos, paciente a
    paciente, hasta llegar a ``n_segments``."""
    rng = np.random.RandomState(seed)
    per_patient = {}
    for patient, group in rows.groupby("patient_uid", sort=True):
        order = rng.permutation(len(group))
        per_patient[patient] = list(group.index.to_numpy()[order])
    chosen = []
    while len(chosen) < n_segments and any(per_patient.values()):
        for patient in sorted(per_patient):
            if per_patient[patient] and len(chosen) < n_segments:
                chosen.append(per_patient[patient].pop(0))
    return rows.loc[sorted(chosen)].reset_index(drop=True)


def sample_recordings(rows: pd.DataFrame, n_recordings: int, seed: int) -> list[str]:
    """``audio_id`` reproducibles, repartidos entre clases."""
    rng = np.random.RandomState(seed)
    recordings = rows.drop_duplicates("audio_id")[["audio_id", "target_label"]]
    chosen = []
    by_label = {label: list(group["audio_id"].to_numpy()[rng.permutation(len(group))])
                for label, group in recordings.groupby("target_label", sort=True)}
    while len(chosen) < n_recordings and any(by_label.values()):
        for label in sorted(by_label):
            if by_label[label] and len(chosen) < n_recordings:
                chosen.append(by_label[label].pop(0))
    return sorted(chosen)
