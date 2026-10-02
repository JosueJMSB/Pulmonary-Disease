"""Nucleo del protocolo holdout_final_v1: entrenamiento definitivo (100% del
80% de desarrollo) y evaluacion externa UNICA (20% de prueba, bloqueada hasta
que ``modeling/final_training/selection.py`` aprueba el pipeline), para una
CNN/CRNN ya congelada en ``selected_pipelines.toml``.

Reutiliza, sin duplicar nada: la arquitectura y el bucle de entrenamiento de
``models/cnn.py`` (``train_fixed_epochs``, ``checkpoint_payload``), la carga de
datos y cache de ``data.py`` (``extract_or_load_logmel``/``build_condition_logmel``,
con ``fold_id="final"`` como subruta -el mismo mecanismo que usa
``holdout_cv.cv_fold_ref`` para ``"cv/fold_00"``-), las metricas/agregaciones de
``evaluation.py`` y las figuras de ``artifacts.py``. Solo el estado de la
secuencia (``status.json``), la huella de ``--resume`` y la publicacion atomica
de cada ``(dataset, pipeline)`` son nuevos.

No entrena ni selecciona nada por su cuenta: todo hiperparametro, epoca,
condicion y permiso de acceso al test vienen de ``selection.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import artifacts as art
from .. import data as dmod
from .. import evaluation as ev
from .. import splits as sp
from ..holdout_cv import code_fingerprint
from ..models import cnn as cnn_model
from . import selection as sel

STAGE_SUBDIR = "final"  # preprocessing/data/holdout_calibrated/<dataset>/final/

STATUS_PENDING = "PENDING"
STATUS_BLOCKED_SELECTION = "BLOCKED_SELECTION"
STATUS_TRAINING = "TRAINING"
STATUS_MODEL_TRAINED = "MODEL_TRAINED"
STATUS_TEST_EVALUATING = "TEST_EVALUATING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_INTERRUPTED = "INTERRUPTED"

SUMMARY_METRICS = (
    "accuracy", "balanced_accuracy", "recall_copd", "recall_negative",
    "macro_f1", "min_class_recall", "auroc", "auprc_copd",
)
SEGMENT_PREDICTION_COLUMNS = ["segment_id", "audio_id", "patient_uid", "target_label", "source_dataset", "device"]

# Marcador que solo existe una vez que final_model.pt + normalization.json +
# training_history.csv estan los TRES escritos: "hay un modelo con el que se
# puede evaluar sin reentrenar" (ver _execute_pipeline). is_model_ready() nunca
# mira final_model.pt por si solo -podria ser un .pt de un intento anterior sin
# su normalizacion/historial, o viceversa-.
MODEL_READY_MARKER = "_MODEL_READY"

# Lista de aceptacion AUTORITATIVA de lo que ``write_reports``/``_execute_pipeline``
# publican por (dataset, condicion) -esta es la referencia real de nombres, no
# cualquier lista informal de un documento de planificacion externo-:
#   Modelo/checkpoint : final_model.pt, normalization.json, training_history.csv,
#                        model_manifest.json
#   Predicciones      : test_segment_predictions.csv, test_recording_predictions.csv,
#                        test_patient_predictions.csv
#   Metricas/reportes : test_metrics.json, test_metrics_summary.csv, test_bootstrap_ci.csv,
#                        test_classification_report.csv, test_confusion_matrix.csv,
#                        test_metrics_by_device.csv, test_confusion_matrix_by_device.csv,
#                        test_metrics_by_source.csv (CONDICIONAL: solo si >=2 source_dataset)
#   Figuras           : figures/confusion_matrix.{png,pdf}, figures/roc.{png,pdf,csv},
#                        figures/pr.{png,pdf,csv} -las curvas ROC/PR YA incluyen su propio
#                        CSV (fpr/tpr y precision/recall por umbral) junto a la figura,
#                        sin necesidad de un "roc_curve.csv"/"pr_curve.csv" aparte-.
#
# Lo que debe existir en un directorio publicado (con _SUCCESS) para poder
# reutilizarlo (ver verify_published_completeness): el modelo/checkpoint, las
# predicciones, las metricas y los reportes/matrices/figuras esenciales -todo
# menos ``test_metrics_by_source.csv`` (condicional) y los ``.pdf`` (formato
# redundante del mismo contenido que el ``.png``, ambos fijados por
# ``[figures] formats`` en los TOML de referencia de este protocolo).
REQUIRED_PUBLISHED_FILES = (
    "final_model.pt", "normalization.json", "training_history.csv", "model_manifest.json",
    "test_segment_predictions.csv", "test_recording_predictions.csv", "test_patient_predictions.csv",
    "test_metrics.json", "test_metrics_summary.csv", "test_bootstrap_ci.csv",
    "test_classification_report.csv", "test_confusion_matrix.csv",
    "test_metrics_by_device.csv", "test_confusion_matrix_by_device.csv",
    "figures/confusion_matrix.png", "figures/roc.png", "figures/roc.csv",
    "figures/pr.png", "figures/pr.csv",
)

REFERENCE_FINGERPRINT_SECTIONS = (
    "acoustic", "logmel", "cnn", "crnn", "weights", "normalization", "augmentation", "seeds", "determinism",
)


class FinalTrainingError(RuntimeError):
    """Un pipeline no se pudo entrenar o evaluar (datos, huella o configuracion)."""


# ---------------------------------------------------------------------------
# Rutas de un (dataset, condicion): staging -> publicacion atomica con _SUCCESS
# (mismo patron que ``artifacts.publish_fold``/``holdout_cv.publish_unit``).
# ---------------------------------------------------------------------------

def pipeline_dir(run_root: Path, dataset: str, condition: str) -> Path:
    return Path(run_root) / "datasets" / dataset / condition


def pipeline_staging_dir(run_root: Path, dataset: str, condition: str) -> Path:
    target = pipeline_dir(run_root, dataset, condition)
    return target.with_name(target.name + "_staging")


def is_pipeline_complete(run_root: Path, dataset: str, condition: str) -> bool:
    return (pipeline_dir(run_root, dataset, condition) / "_SUCCESS").is_file()


def is_model_ready(staging: Path) -> bool:
    return (staging / MODEL_READY_MARKER).is_file()


def verify_published_completeness(target: Path) -> None:
    """``target`` (publicado con ``_SUCCESS``, o un staging a punto de
    publicarse) debe tener TODOS los artefactos requeridos. Se exige en dos
    momentos: justo ANTES del unico ``publish_pipeline()`` -nunca se publica
    ``_SUCCESS`` sobre un resultado incompleto-, y antes de reutilizar un
    resultado ya publicado (--resume o el atajo "ya completado") -``_SUCCESS``
    por si solo no demuestra que el contenido sigue completo-."""
    missing = [name for name in REQUIRED_PUBLISHED_FILES if not (target / name).is_file()]
    if missing:
        raise FinalTrainingError(
            f"{target}: faltan archivos requeridos ({', '.join(missing)}); "
            "no se publica ni se reutiliza sin revision manual"
        )


def publish_pipeline(staging: Path) -> Path:
    """Publicacion atomica: ``_SUCCESS`` se crea DENTRO del staging, ANTES del
    unico ``os.replace()`` que lo mueve a su destino -asi ``target`` aparece ya
    completo, con su propio ``_SUCCESS``, en una sola operacion atomica del
    sistema de archivos; nunca se crea ``_SUCCESS`` despues del reemplazo-.

    Si ``target`` ya existe aqui es un estado inesperado: ``is_pipeline_complete``
    ya habria detenido la ejecucion antes si estuviera completo (ver
    ``run_pipeline_for_dataset``), asi que un ``target`` incompleto no se borra
    automaticamente -eso perderia evidencia de que el test pudo haberse
    abierto-; se exige revisión y una reanudacion controlada."""
    (staging / "_SUCCESS").touch()
    target = staging.with_name(staging.name.replace("_staging", ""))
    if target.exists():
        raise FinalTrainingError(
            f"{target} ya existe; no se sobrescribe automaticamente. Verifiquelo manualmente "
            "(puede corresponder a una ejecucion interrumpida que ya toco el test) antes de reintentar."
        )
    os.replace(staging, target)
    return target


def cleanup_abandoned_staging(run_root: Path) -> list[Path]:
    """Solo borra staging que NO tenga el modelo listo (``_MODEL_READY``) NI
    predicciones: eso es lo unico que garantiza que el test nunca se toco para
    esa unidad. Si el modelo esta listo (con o sin predicciones), se conserva
    -ver ``run_pipeline_for_dataset``, que continua la evaluacion desde ESE
    checkpoint exacto, sin reentrenar-; perderlo arriesgaria reentrenar con el
    test potencialmente ya abierto. Un ``final_model.pt`` SIN ``_MODEL_READY``
    (un intento que se interrumpio a medias, antes de completar el checkpoint)
    SI se borra: no hay garantia de que ese archivo este integro."""
    removed: list[Path] = []
    datasets_dir = Path(run_root) / "datasets"
    if not datasets_dir.is_dir():
        return removed
    for staging in sorted(datasets_dir.glob("*/*_staging")):
        has_predictions = (staging / "test_segment_predictions.csv").is_file()
        if not is_model_ready(staging) and not has_predictions:
            shutil.rmtree(staging, ignore_errors=True)
            removed.append(staging)
    return removed


# ---------------------------------------------------------------------------
# Huella de un pipeline: lo que --resume debe verificar antes de reutilizar
# cualquier prediccion o modelo ya escrito en su staging.
# ---------------------------------------------------------------------------

def reference_config_fingerprint(cfg: dict) -> str:
    relevant = {k: cfg[k] for k in REFERENCE_FINGERPRINT_SECTIONS if k in cfg}
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_final_input_hashes(data_root: Path, dataset: str, branch: str) -> dict[str, str]:
    root = dmod.dataset_root(data_root, dataset, STAGE_SUBDIR)
    prefix = f"{dataset}/{STAGE_SUBDIR}"
    return {
        f"{prefix}/segments.csv": dmod.sha256_file(root / "segments.csv"),
        f"{prefix}/segments_{branch}.npy": dmod.sha256_file(root / f"segments_{branch}.npy"),
        f"{prefix}/manifest.json": dmod.sha256_file(root / "manifest.json"),
        f"{prefix}/preprocessing_params.json": dmod.sha256_file(root / "preprocessing_params.json"),
    }


def build_pipeline_fingerprint(
    selection_cfg: sel.SelectionConfig, pipeline: sel.SelectedPipeline, cfg: dict, data_root: Path,
) -> dict:
    """Huella de ``--resume`` de ESTE dataset. El componente ``pipeline_fingerprint``
    es la huella canonica de ``selection.py`` -pipeline, hiperparametros, umbral,
    semilla, hashes del split, artefacto de procedencia, aprobacion y revision
    por dispositivo de ESTE dataset-, NUNCA el hash de ``selected_pipelines.toml``
    completo: aprobar ICBHI no debe invalidar un staging de FRAIWAN o COMBINED
    que nunca cambio."""
    fingerprint = {
        "dataset": pipeline.dataset,
        "pipeline_fingerprint": sel.canonical_pipeline_fingerprint(
            pipeline, selection_cfg, reference_config_fingerprint(cfg),
        ),
        "input_hashes": {
            "holdout_splits.csv": selection_cfg.split_csv_sha256,
            "holdout_splits_manifest.json": selection_cfg.split_manifest_sha256,
            **dataset_final_input_hashes(data_root, pipeline.dataset, pipeline.branch),
        },
        **code_fingerprint(),  # code_sha256, code_files: reutilizado de holdout_cv.py, no se duplica
    }
    return fingerprint


def write_pipeline_fingerprint(dir_path: Path, fingerprint: dict) -> None:
    (dir_path / "fingerprint.json").write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8",
    )


def verify_pipeline_fingerprint(dir_path: Path, fingerprint: dict) -> None:
    """Se niega a reutilizar un staging si la configuracion congelada de ESTE
    dataset, el codigo o los datos de entrada cambiaron desde que se escribio:
    reutilizar predicciones calculadas con otra entrada mezclaria resultados
    incoherentes sin ningun aviso."""
    path = dir_path / "fingerprint.json"
    if not path.is_file():
        raise FinalTrainingError(f"no se puede reanudar {dir_path}: falta fingerprint.json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if stored.get("pipeline_fingerprint") != fingerprint["pipeline_fingerprint"]:
        mismatches.append(
            "la configuracion congelada de este dataset (pipeline/hiperparametros/epocas/umbral/semilla/"
            "procedencia/aprobacion/revision por dispositivo) cambio"
        )
    if stored.get("code_sha256") != fingerprint.get("code_sha256"):
        old_files, new_files = stored.get("code_files", {}), fingerprint.get("code_files", {})
        changed = sorted(k for k in set(old_files) | set(new_files) if old_files.get(k) != new_files.get(k))
        detail = f" (archivos: {', '.join(changed[:10])}{' ...' if len(changed) > 10 else ''})" if changed else ""
        mismatches.append(f"el codigo de modeling/ cambio desde la ejecucion original{detail}")
    stored_hashes = stored.get("input_hashes", {})
    for key, value in fingerprint["input_hashes"].items():
        if stored_hashes.get(key) != value:
            mismatches.append(f"entrada modificada desde la ejecucion original: {key}")
    if mismatches:
        raise FinalTrainingError(
            f"--resume rechazado para {dir_path}: no coincide con la ejecucion original.\n  - "
            + "\n  - ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Registro persistente de acceso al test: FUERA del staging (sobrevive aunque
# el staging se pierda o se limpie). Nunca se sobrescribe con otra huella u
# otro modelo: eso indicaria que el test se abrio dos veces con
# configuraciones distintas, algo permanentemente prohibido.
# ---------------------------------------------------------------------------

TEST_ACCESS_STARTED = "STARTED"
TEST_ACCESS_PREDICTIONS_WRITTEN = "PREDICTIONS_WRITTEN"
TEST_ACCESS_COMPLETED = "COMPLETED"


def test_access_dir(run_root: Path) -> Path:
    return Path(run_root) / "test_access"


def test_access_path(run_root: Path, dataset: str, condition: str) -> Path:
    return test_access_dir(run_root) / f"{dataset}_{condition}.json"


def read_test_access_record(run_root: Path, dataset: str, condition: str) -> dict | None:
    path = test_access_path(run_root, dataset, condition)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_test_access_record(
    run_root: Path, dataset: str, condition: str, *,
    pipeline_fingerprint: str, model_sha256: str, status: str,
    split_csv_sha256: str, split_manifest_sha256: str,
) -> None:
    """Registra (o avanza el estado de) el acceso al test de ``(dataset,
    condition)``. Si ya existe un registro con otra huella u otro modelo, se
    rechaza: el test ya quedo abierto con esa configuracion y no se sobrescribe
    con otra en silencio."""
    path = test_access_path(run_root, dataset, condition)
    existing = read_test_access_record(run_root, dataset, condition)
    if existing is not None and (
        existing.get("pipeline_fingerprint") != pipeline_fingerprint or existing.get("model_sha256") != model_sha256
    ):
        raise FinalTrainingError(
            f"{dataset}/{condition}: ya existe un registro de acceso al test con otra huella u otro modelo "
            f"({path}); el test permanece abierto con esa configuracion y no se sobrescribe"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    now = art._now_iso()
    payload = {
        "dataset": dataset, "condition": condition,
        "pipeline_fingerprint": pipeline_fingerprint, "model_sha256": model_sha256, "status": status,
        "split_csv_sha256": split_csv_sha256, "split_manifest_sha256": split_manifest_sha256,
        "created_at_utc": (existing or {}).get("created_at_utc", now),
        "updated_at_utc": now,
    }
    # Archivo temporal + os.replace: el registro nunca queda a medio escribir,
    # ni siquiera si el proceso se interrumpe exactamente en este punto.
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Datos de la etapa final: segments.csv/segments_<rama>.npy de
# <dataset>/final/ tienen AMBOS roles (train=desarrollo completo, test=prueba
# externa). Se verifican contra el split oficial antes de separarlos.
# ---------------------------------------------------------------------------

def verify_final_segments_against_split(segments: pd.DataFrame, split: sp.HoldoutSplit, dataset: str) -> None:
    if "role" not in segments.columns:
        raise FinalTrainingError(f"{dataset}/{STAGE_SUBDIR}: segments.csv no tiene columna 'role'")
    unknown = set(segments["role"]) - {sp.ROLE_TRAIN, sp.ROLE_TEST}
    if unknown:
        raise FinalTrainingError(f"{dataset}/{STAGE_SUBDIR}: roles no permitidos en la etapa final: {sorted(unknown)}")
    development = set(split.development_patients())
    test = set(split.blocked_test_patients())
    present_train = set(segments.loc[segments["role"] == sp.ROLE_TRAIN, "patient_uid"])
    present_test = set(segments.loc[segments["role"] == sp.ROLE_TEST, "patient_uid"])
    if present_train != development:
        raise FinalTrainingError(
            f"{dataset}/{STAGE_SUBDIR}: desarrollo no coincide con el split "
            f"(sobran {sorted(present_train - development)[:5]}, faltan {sorted(development - present_train)[:5]})"
        )
    if present_test != test:
        raise FinalTrainingError(
            f"{dataset}/{STAGE_SUBDIR}: la prueba externa no coincide con el split "
            f"(sobran {sorted(present_test - test)[:5]}, faltan {sorted(test - present_test)[:5]})"
        )


@dataclass
class PipelineData:
    spec: dmod.ConditionSpec
    logmel: np.ndarray
    dev_seg: pd.DataFrame
    test_seg: pd.DataFrame


def load_pipeline_data(
    data_root: Path, cache_root: Path, pipeline: sel.SelectedPipeline, cfg: dict, split: sp.HoldoutSplit,
    force_features: bool, logger,
) -> PipelineData:
    spec = dmod.ConditionSpec(
        dataset=pipeline.dataset, condition=pipeline.condition, branch=pipeline.branch, dn_reliable_only=False,
    )
    logmel, rows = dmod.extract_or_load_logmel(
        data_root, cache_root, pipeline.dataset, pipeline.branch, cfg, force=force_features, fold_id=STAGE_SUBDIR,
        progress=lambda done, total: logger.info(f"log-mel {pipeline.dataset}/{STAGE_SUBDIR}/{pipeline.branch}: {done}/{total}"),
    )
    condition = dmod.build_condition_logmel(data_root, spec, logmel, rows, fold_id=STAGE_SUBDIR)
    verify_final_segments_against_split(condition.segments, split, pipeline.dataset)

    dev_seg = condition.segments.loc[condition.segments["role"] == sp.ROLE_TRAIN].reset_index(drop=True)
    test_seg = condition.segments.loc[condition.segments["role"] == sp.ROLE_TEST].reset_index(drop=True)
    for name, part in ((sp.ROLE_TRAIN, dev_seg), (sp.ROLE_TEST, test_seg)):
        if part.empty:
            raise FinalTrainingError(f"{pipeline.dataset}: el conjunto {name!r} quedo vacio")
    if set(dev_seg["patient_uid"]) & set(test_seg["patient_uid"]):
        raise FinalTrainingError(f"{pipeline.dataset}: hay pacientes compartidos entre desarrollo y prueba")
    return PipelineData(spec=spec, logmel=condition.logmel, dev_seg=dev_seg, test_seg=test_seg)


# ---------------------------------------------------------------------------
# Entrenamiento definitivo: train_fixed_epochs con TODO el desarrollo, sin
# validation ni early stopping. Semilla: seed_base + dataset + condicion +
# "final" (ver models/cnn.py::derive_seed).
# ---------------------------------------------------------------------------

def build_settings(cfg: dict, selection_cfg: sel.SelectionConfig, pipeline: sel.SelectedPipeline, num_workers: int) -> cnn_model.TrainingSettings:
    base = cnn_model.TrainingSettings.from_config(cfg, num_workers)
    return dataclasses.replace(
        base,
        batch_size=int(pipeline.hyperparameters["batch_size"]),
        weight_decay=float(pipeline.hyperparameters["weight_decay"]),
        threshold=float(selection_cfg.threshold),
        seed_base=int(selection_cfg.seed_base),
    )


def train_pipeline(
    data: PipelineData, cfg: dict, selection_cfg: sel.SelectionConfig, pipeline: sel.SelectedPipeline,
    settings: cnn_model.TrainingSettings, device: torch.device, log,
) -> cnn_model.FixedRun:
    config = cnn_model.SearchConfiguration(
        index=int(pipeline.source_config_index),
        lr=float(pipeline.hyperparameters["lr"]),
        dropout=float(pipeline.hyperparameters["dropout"]),
    )
    augment_cfg = cfg["augmentation"] if data.spec.augment else None
    return cnn_model.train_fixed_epochs(
        data.logmel, data.dev_seg, int(pipeline.epochs), cfg, config, settings, device,
        seed_parts=(pipeline.dataset, pipeline.condition, "final"), augment_cfg=augment_cfg, log=log,
    )


# ---------------------------------------------------------------------------
# Evaluacion externa UNICA: inferencia sin gradiente, agregacion
# segmento->grabacion->paciente (``cnn_model.evaluate_patients``, la misma
# funcion que usa el protocolo anterior para su test de fold), metricas,
# reportes, IC bootstrap, curvas y desgloses por dispositivo/fuente.
# ---------------------------------------------------------------------------

@dataclass
class _ModelAndNormalization:
    """Adaptador minimo para ``evaluate_pipeline``: solo necesita ``.model`` y
    ``.normalization``, igual que un ``cnn_model.FixedRun`` real. Se usa al
    reanudar desde un ``final_model.pt`` ya publicado (sin ``.history`` ni
    ``.seconds``, que no se reconstruyen de un checkpoint -ver
    ``_normalization_from_checkpoint``-)."""

    model: torch.nn.Module
    normalization: "cnn_model.NormalizationStats"


def _normalization_from_checkpoint(checkpoint: dict) -> "cnn_model.NormalizationStats":
    norm = checkpoint["normalization"]
    return cnn_model.NormalizationStats(
        mean=norm["mean"].numpy(), std=norm["std"].numpy(),
        n_segments=int(norm["n_segments"]), weighted=bool(norm["weighted"]),
    )


def evaluate_pipeline(
    data: PipelineData, run: cnn_model.FixedRun, pipeline: sel.SelectedPipeline,
    settings: cnn_model.TrainingSettings, device: torch.device, num_workers: int, negative_label_name: str,
) -> dict:
    model = run.model
    model.eval()
    test_ds = cnn_model.LogmelDataset(data.logmel, data.test_seg["cache_row"], data.test_seg["target_label"], None, run.normalization)
    test_loader = cnn_model.make_loader(
        test_ds, settings.eval_batch_size, False, num_workers,
        cnn_model.derive_seed(settings.seed_base, pipeline.dataset, pipeline.condition, "final", "test"), device,
    )
    logits = cnn_model.predict_logits(model, test_loader, device, settings.use_amp(device))
    del test_loader
    probabilities = cnn_model.sigmoid(logits)
    metrics, scored, recordings, patients = cnn_model.evaluate_patients(
        data.test_seg, probabilities, negative_label_name, settings.threshold,
    )
    neg_key = f"recall_{negative_label_name.strip().lower()}"
    metrics["recall_negative"] = metrics[neg_key]

    by_audio = scored.drop_duplicates("audio_id").set_index("audio_id")
    for column in ("source_dataset", "device"):
        if column in by_audio.columns:
            recordings[column] = recordings["audio_id"].map(by_audio[column])
    by_patient = scored.drop_duplicates("patient_uid").set_index("patient_uid")
    if "source_dataset" in by_patient.columns:
        patients["source_dataset"] = patients["patient_uid"].map(by_patient["source_dataset"])

    return {"metrics": metrics, "segments": scored, "recordings": recordings, "patients": patients}


def _reload_evaluation_from_predictions(staging: Path, negative_label_name: str, threshold: float) -> dict:
    """Reconstruye el mismo dict que devuelve ``evaluate_pipeline`` a partir de
    ``test_segment_predictions.csv`` ya escrito en un intento anterior: permite
    reconstruir SOLO los informes faltantes (``write_reports``) sin repetir la
    inferencia ni tocar el modelo de nuevo."""
    scored = pd.read_csv(
        staging / "test_segment_predictions.csv",
        dtype={"segment_id": str, "audio_id": str, "patient_uid": str, "source_dataset": str, "device": str},
        keep_default_na=False,
    )
    recordings = ev.aggregate_segment_to_recording(scored)
    patients = ev.aggregate_recording_to_patient(recordings)
    by_audio = scored.drop_duplicates("audio_id").set_index("audio_id")
    for column in ("source_dataset", "device"):
        if column in by_audio.columns:
            recordings[column] = recordings["audio_id"].map(by_audio[column])
    by_patient = scored.drop_duplicates("patient_uid").set_index("patient_uid")
    if "source_dataset" in by_patient.columns:
        patients["source_dataset"] = patients["patient_uid"].map(by_patient["source_dataset"])
    metrics = ev.compute_patient_metrics(
        patients["target_label"].to_numpy(), patients["score"].to_numpy(), negative_label_name, threshold,
    )
    neg_key = f"recall_{negative_label_name.strip().lower()}"
    metrics["recall_negative"] = metrics[neg_key]
    metrics["min_class_recall"] = float(min(metrics["recall_copd"], metrics["recall_negative"]))
    return {"metrics": metrics, "segments": scored, "recordings": recordings, "patients": patients}


def _bootstrap_metric_fns(threshold: float) -> dict:
    return {
        "accuracy": ev.metric_fn_accuracy(threshold),
        "balanced_accuracy": ev.metric_fn_balanced_accuracy(threshold),
        "recall_copd": ev.metric_fn_recall_copd(threshold),
        "recall_negative": ev.metric_fn_recall_negative(threshold),
        "macro_f1": ev.metric_fn_macro_f1(threshold),
        "min_class_recall": ev.metric_fn_min_class_recall(threshold),
        "auroc": ev.metric_fn_auroc(),
        "auprc_copd": ev.metric_fn_auprc(),
    }


def bootstrap_table(y_true: np.ndarray, y_score: np.ndarray, selection_cfg: sel.SelectionConfig) -> pd.DataFrame:
    rows = []
    for name, fn in _bootstrap_metric_fns(selection_cfg.threshold).items():
        result = ev.bootstrap_confidence_interval(
            y_true, y_score, fn,
            n_resamples=selection_cfg.bootstrap_n_resamples, confidence=selection_cfg.bootstrap_confidence,
            random_state=selection_cfg.bootstrap_random_state,
        )
        rows.append({"metric": name, **result})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Artefactos: todo lo que el plan exige por dataset, en
# <run_root>/datasets/<dataset>/<condicion>/.
# ---------------------------------------------------------------------------

def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(art._json_safe(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_figures(staging: Path, cfg: dict, y_true: np.ndarray, y_score: np.ndarray, target_names: tuple[str, str], threshold: float, logger) -> None:
    figures = staging / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("confusion_matrix", lambda: art.plot_confusion_matrix(y_true, y_score, target_names, figures / "confusion_matrix", cfg, threshold)),
        ("roc", lambda: art.plot_roc_curve(y_true, y_score, figures / "roc", cfg)),
        ("pr", lambda: art.plot_pr_curve(y_true, y_score, figures / "pr", cfg)),
    )
    for name, draw in jobs:
        try:
            draw()
        except Exception:  # noqa: BLE001 - una figura no debe tumbar resultados ya calculados
            logger.exception(f"no se pudo generar la figura {name}")


def write_reports(
    staging: Path, cfg: dict, selection_cfg: sel.SelectionConfig, pipeline: sel.SelectedPipeline,
    negative_label_name: str, evaluation: dict, logger,
) -> dict:
    """Todos los reportes derivados de las predicciones de test ya calculadas
    (metricas, IC bootstrap, reportes, matrices, desgloses, figuras). Se
    puede volver a llamar sobre predicciones reutilizadas de un staging
    anterior (--resume), sin repetir la inferencia."""
    patients = evaluation["patients"]
    y_true = patients["target_label"].to_numpy()
    y_score = patients["score"].to_numpy()
    target_names = (negative_label_name, "COPD")
    threshold = selection_cfg.threshold

    metrics_summary = pd.DataFrame(
        [{"metric": m, "value": evaluation["metrics"][m]} for m in SUMMARY_METRICS]
    )
    bootstrap_ci = bootstrap_table(y_true, y_score, selection_cfg)
    classification_report = ev.classification_report_df(y_true, y_score, target_names, threshold)
    confusion_matrix = ev.confusion_matrix_df(y_true, y_score, target_names, threshold).reset_index().rename(columns={"index": "real"})

    patient_device = ev.aggregate_patient_device(evaluation["segments"])
    metrics_by_device = ev.compute_metrics_by_device(patient_device, negative_label_name, threshold)
    confusion_by_device = ev.confusion_matrix_by_source_df(patient_device, "device", target_names, threshold)

    metrics_by_source = None
    if "source_dataset" in patients.columns and patients["source_dataset"].nunique() >= 2:
        metrics_by_source = ev.compute_metrics_by_source(patients, "source_dataset", negative_label_name, threshold)

    _write_json(evaluation["metrics"], staging / "test_metrics.json")
    _write_csv(metrics_summary, staging / "test_metrics_summary.csv")
    _write_csv(bootstrap_ci, staging / "test_bootstrap_ci.csv")
    _write_csv(classification_report, staging / "test_classification_report.csv")
    _write_csv(confusion_matrix, staging / "test_confusion_matrix.csv")
    _write_csv(metrics_by_device, staging / "test_metrics_by_device.csv")
    _write_csv(confusion_by_device, staging / "test_confusion_matrix_by_device.csv")
    if metrics_by_source is not None:
        _write_csv(metrics_by_source, staging / "test_metrics_by_source.csv")

    write_figures(staging, cfg, y_true, y_score, target_names, threshold, logger)
    return {"metrics_summary": metrics_summary, "bootstrap_ci": bootstrap_ci}


# ---------------------------------------------------------------------------
# Orquestacion de UN (dataset, pipeline): el equivalente, para este
# protocolo, de holdout_cnn.run_holdout_cnn pero sin busqueda ni folds.
# ---------------------------------------------------------------------------

def run_pipeline_for_dataset(
    *,
    run_root: Path,
    dataset: str,
    selection_cfg: sel.SelectionConfig,
    data_root: Path,
    cache_root: Path,
    source_runs_root: Path,
    device: torch.device,
    num_workers: int,
    force_features: bool,
    dry_run: bool,
    logger,
) -> dict:
    try:
        pipeline = sel.validate_pipeline_for_execution(selection_cfg, dataset, source_runs_root)
    except sel.SelectionError as exc:
        logger.warning(f"{dataset}: BLOCKED_SELECTION - {exc}")
        return {"dataset": dataset, "status": STATUS_BLOCKED_SELECTION, "detail": str(exc)}

    cfg = sel.validate_reference_config(pipeline)
    target = pipeline_dir(run_root, dataset, pipeline.condition)
    fingerprint = build_pipeline_fingerprint(selection_cfg, pipeline, cfg, data_root)

    if is_pipeline_complete(run_root, dataset, pipeline.condition):
        # Nunca se confia en un _SUCCESS por si solo: la huella publicada debe
        # seguir coincidiendo con la seleccion/codigo/split/datos ACTUALES, y
        # deben seguir estando todos los artefactos requeridos.
        verify_pipeline_fingerprint(target, fingerprint)
        verify_published_completeness(target)
        logger.info(f"{dataset}/{pipeline.condition}: ya esta COMPLETED, se omite (un run completado no reabre el test)")
        status = art.read_status(target)
        metrics = json.loads((target / "test_metrics.json").read_text(encoding="utf-8"))
        return {
            "dataset": dataset, "condition": pipeline.condition,
            "status": status.get("status", STATUS_COMPLETED), "detail": "ya completado",
            "metrics": metrics, "run_dir": target,
        }

    # Inconsistencias graves del registro persistente (sobrevive aunque se
    # pierda el staging): nunca se reintentan solas, siempre exigen revision manual.
    existing_access = read_test_access_record(run_root, dataset, pipeline.condition)
    if existing_access is not None:
        access_status = existing_access.get("status")
        if access_status == TEST_ACCESS_COMPLETED:
            raise FinalTrainingError(
                f"{dataset}/{pipeline.condition}: {test_access_path(run_root, dataset, pipeline.condition)} "
                f"indica que el test ya se completo, pero no existe la publicacion en {target}; esto es una "
                "inconsistencia grave (el test ya se abrio) y requiere revision manual, nunca un reintento automatico"
            )
        if access_status in (TEST_ACCESS_STARTED, TEST_ACCESS_PREDICTIONS_WRITTEN):
            staging_for_check = pipeline_staging_dir(run_root, dataset, pipeline.condition)
            if not staging_for_check.exists():
                raise FinalTrainingError(
                    f"{dataset}/{pipeline.condition}: {test_access_path(run_root, dataset, pipeline.condition)} "
                    f"indica {access_status!r}, pero no existe ningun staging en {staging_for_check}; se perdio "
                    "la evidencia de ese acceso al test. Requiere revision manual, nunca un reintento automatico."
                )

    if dry_run:
        logger.info(f"{dataset}/{pipeline.condition}: --dry-run OK (seleccion aprobada, split y datos verificados)")
        return {
            "dataset": dataset, "status": "DRY_RUN_OK",
            "detail": "No se cargaron senales, etiquetas ni predicciones del test.",
        }

    staging = pipeline_staging_dir(run_root, dataset, pipeline.condition)
    staging.mkdir(parents=True, exist_ok=True)
    if (staging / "fingerprint.json").is_file():
        verify_pipeline_fingerprint(staging, fingerprint)
    else:
        write_pipeline_fingerprint(staging, fingerprint)

    try:
        return _execute_pipeline(
            staging=staging, cfg=cfg, pipeline=pipeline, selection_cfg=selection_cfg,
            data_root=data_root, cache_root=cache_root, device=device, num_workers=num_workers,
            force_features=force_features, fingerprint=fingerprint, dataset=dataset,
            run_root=run_root, logger=logger,
        )
    except KeyboardInterrupt:
        art.write_status(staging, STATUS_INTERRUPTED)
        raise
    except Exception as exc:  # noqa: BLE001 - se registra en el staging, nunca se omite en silencio
        art.write_status(staging, STATUS_FAILED, {"error": str(exc), "error_type": type(exc).__name__})
        raise


def _execute_pipeline(
    *, staging: Path, cfg: dict, pipeline: sel.SelectedPipeline, selection_cfg: sel.SelectionConfig,
    data_root: Path, cache_root: Path, device: torch.device, num_workers: int,
    force_features: bool, fingerprint: dict, dataset: str, run_root: Path, logger,
) -> dict:
    """Cuerpo real de ``run_pipeline_for_dataset`` una vez resuelta la
    seleccion y preparado el staging. Tres casos, nunca mas de uno, decididos
    por ``_MODEL_READY`` (NUNCA por la mera existencia de ``final_model.pt``:
    ese marcador solo existe una vez que el checkpoint, la normalizacion y el
    historial de entrenamiento estan los TRES escritos):

    - Predicciones YA calculadas (modelo listo + predicciones): se reutilizan
      sin tocar el modelo ni el test de nuevo (solo se reconstruyen los
      informes).
    - Modelo listo pero sin predicciones (una ejecucion anterior se
      interrumpio justo despues de publicarlo): se reanuda la evaluacion
      cargando EXACTAMENTE ese checkpoint, sin reentrenar.
    - Ninguno de los dos: entrenamiento fresco completo.
    - Predicciones SIN modelo listo (estado imposible en operacion normal) se
      rechaza de forma explicita: nunca se reentrena para "resolverlo" solo.

    Separado de ``run_pipeline_for_dataset`` para que esta pueda envolver TODO
    esto en un solo try/except que marque FAILED/INTERRUPTED en el staging
    antes de relanzar la excepcion."""
    negative_label_name = cfg["datasets"][dataset]["negative_label_name"]

    # selection_snapshot.json no depende de entrenar ni evaluar: se escribe
    # siempre, de forma idempotente, para que exista incluso si el resto se
    # interrumpe inmediatamente despues.
    pipeline_snapshot = dataclasses.asdict(pipeline)
    pipeline_snapshot["reference_config"] = str(pipeline.reference_config)
    _write_json({
        "selection_config_path": str(selection_cfg.path),
        "selection_config_sha256": dmod.sha256_file(selection_cfg.path),
        "threshold": selection_cfg.threshold, "seed_base": selection_cfg.seed_base,
        "bootstrap": {
            "n_resamples": selection_cfg.bootstrap_n_resamples,
            "confidence": selection_cfg.bootstrap_confidence,
            "random_state": selection_cfg.bootstrap_random_state,
        },
        "pipeline": pipeline_snapshot,
    }, staging / "selection_snapshot.json")

    has_predictions = (staging / "test_segment_predictions.csv").is_file()
    has_model_ready = is_model_ready(staging)

    if has_predictions and not has_model_ready:
        raise FinalTrainingError(
            f"{dataset}/{pipeline.condition}: existen predicciones en {staging} pero falta el modelo listo "
            f"({MODEL_READY_MARKER}); esto es inconsistente y requiere revision manual, nunca un "
            "reentrenamiento automatico para 'resolverlo'"
        )

    if has_predictions and has_model_ready:
        logger.info(f"{dataset}/{pipeline.condition}: reutilizando predicciones ya calculadas en {staging}")
        evaluation = _reload_evaluation_from_predictions(staging, negative_label_name, selection_cfg.threshold)
    else:
        split = sp.load_holdout_split(selection_cfg.split_csv, selection_cfg.split_manifest, dataset)

        if not has_model_ready:
            # Caso C: entrenamiento fresco completo. Un final_model.pt sin
            # _MODEL_READY (intento anterior interrumpido a medio camino) no
            # esta garantizado integro: se descarta en vez de reutilizarlo.
            for leftover in ("final_model.pt", MODEL_READY_MARKER, "model_manifest.json", "normalization.json",
                              "training_history.csv", "test_access_log.json", "status.json"):
                (staging / leftover).unlink(missing_ok=True)

            art.write_status(staging, STATUS_TRAINING)
            logger.info(f"{dataset}/{pipeline.condition}: cargando datos de la etapa final (100% desarrollo + prueba bloqueada)")
            data = load_pipeline_data(data_root, cache_root, pipeline, cfg, split, force_features, logger)
            settings = build_settings(cfg, selection_cfg, pipeline, num_workers)

            logger.info(
                f"{dataset}/{pipeline.condition}: entrenando {pipeline.epochs} epocas con "
                f"{data.dev_seg['patient_uid'].nunique()} pacientes de desarrollo "
                f"(lr={pipeline.hyperparameters['lr']:g}, dropout={pipeline.hyperparameters['dropout']:g})"
            )
            run = train_pipeline(
                data, cfg, selection_cfg, pipeline, settings, device,
                log=lambda message: logger.info(f"{dataset}/{pipeline.condition}: {message}"),
            )

            state_dict = {k: v.detach().cpu().clone() for k, v in run.model.state_dict().items()}
            hyperparameters = {
                "config_index": pipeline.source_config_index, "lr": pipeline.hyperparameters["lr"],
                "dropout": pipeline.hyperparameters["dropout"], "epochs": pipeline.epochs,
            }
            checkpoint = cnn_model.checkpoint_payload(
                state_dict, run.normalization, cfg, data.spec, negative_label_name, hyperparameters,
                extra={
                    "protocol": sel.PROTOCOL_NAME,
                    "weight_decay": pipeline.hyperparameters["weight_decay"],
                    "batch_size": pipeline.hyperparameters["batch_size"],
                    "seed_base": selection_cfg.seed_base,
                    "seed_parts": [dataset, pipeline.condition, "final"],
                    "n_train_patients": int(data.dev_seg["patient_uid"].nunique()),
                    "n_train_segments": int(len(data.dev_seg)),
                    "source_run_id": pipeline.source_run_id,
                    "source_config_index": pipeline.source_config_index,
                    "source_artifact": pipeline.source_artifact,
                    "source_artifact_sha256": pipeline.source_artifact_sha256,
                    "robustness_run_id": pipeline.robustness_run_id or None,
                    "pipeline_fingerprint": fingerprint["pipeline_fingerprint"],
                    "code_sha256": fingerprint.get("code_sha256"),
                },
            )
            cnn_model.save_checkpoint(staging / "final_model.pt", checkpoint)
            # normalization.json/training_history.csv son del ENTRENAMIENTO, no
            # de la evaluacion: se escriben ya, para que sigan disponibles si la
            # evaluacion que sigue se interrumpe (ver el caso "modelo sin
            # predicciones" mas abajo, que nunca necesita reconstruirlos).
            (staging / "normalization.json").write_text(
                json.dumps(run.normalization.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
            )
            _write_csv(
                run.history.assign(weight_decay=pipeline.hyperparameters["weight_decay"], batch_size=pipeline.hyperparameters["batch_size"]),
                staging / "training_history.csv",
            )
            # _MODEL_READY se crea DESPUES de los tres (checkpoint, normalizacion,
            # historial): es la unica senal que --resume usa para decidir si hay un
            # modelo con el que evaluar sin reentrenar (ver is_model_ready arriba).
            (staging / MODEL_READY_MARKER).touch()
            logger.info(f"{dataset}/{pipeline.condition}: final_model.pt publicado, iniciando la UNICA evaluacion del test")
            art.write_status(staging, STATUS_MODEL_TRAINED)
            model, normalization = run.model, run.normalization
        else:
            # Caso B: una ejecucion anterior ya publico el modelo (_MODEL_READY)
            # pero se interrumpio antes de escribir predicciones. Se reanuda la
            # evaluacion desde ESE checkpoint exacto; nunca se reentrena.
            logger.info(
                f"{dataset}/{pipeline.condition}: final_model.pt ya existe en {staging}; "
                "se reanuda la evaluacion del test SIN reentrenar"
            )
            checkpoint = cnn_model.load_checkpoint(staging / "final_model.pt")
            model = cnn_model.model_from_checkpoint(checkpoint).to(device)
            normalization = _normalization_from_checkpoint(checkpoint)
            settings = build_settings(cfg, selection_cfg, pipeline, num_workers)
            data = load_pipeline_data(data_root, cache_root, pipeline, cfg, split, force_features, logger)

        model_sha256 = dmod.sha256_file(staging / "final_model.pt")
        write_test_access_record(
            run_root, dataset, pipeline.condition,
            pipeline_fingerprint=fingerprint["pipeline_fingerprint"], model_sha256=model_sha256,
            status=TEST_ACCESS_STARTED, split_csv_sha256=selection_cfg.split_csv_sha256,
            split_manifest_sha256=selection_cfg.split_manifest_sha256,
        )
        n_test_patients = int(data.test_seg["patient_uid"].nunique())
        n_test_copd = int(data.test_seg.loc[data.test_seg["target_label"] == 1, "patient_uid"].nunique())
        _write_json({
            "opened_at_utc": art._now_iso(),
            "dataset": dataset, "condition": pipeline.condition, "architecture": pipeline.architecture,
            "selection_status": pipeline.status, "pipeline_fingerprint": fingerprint["pipeline_fingerprint"],
            "split_csv_sha256": selection_cfg.split_csv_sha256,
            "n_test_patients": n_test_patients, "n_test_segments": int(len(data.test_seg)),
        }, staging / "test_access_log.json")

        art.write_status(staging, STATUS_TEST_EVALUATING)
        eval_run = _ModelAndNormalization(model=model, normalization=normalization)
        evaluation = evaluate_pipeline(data, eval_run, pipeline, settings, device, num_workers, negative_label_name)

        warnings_list = []
        if dataset == "FRAIWAN_Extended":
            warnings_list.append(
                f"FRAIWAN_Extended: la prueba externa tiene solo {n_test_patients} pacientes "
                f"({n_test_copd} COPD); los IC por bootstrap y los desgloses por dispositivo/fuente "
                "son poco informativos con tan pocos positivos."
            )

        segment_predictions = evaluation["segments"][
            [c for c in SEGMENT_PREDICTION_COLUMNS if c in evaluation["segments"].columns] + ["score"]
        ]
        _write_csv(segment_predictions, staging / "test_segment_predictions.csv")
        _write_csv(evaluation["recordings"], staging / "test_recording_predictions.csv")
        _write_csv(evaluation["patients"], staging / "test_patient_predictions.csv")

        # model_manifest.json se escribe ANTES de marcar PREDICTIONS_WRITTEN en el
        # registro externo: asi, si el proceso se interrumpe justo despues de ese
        # registro, una reanudacion que reutilice las predicciones (caso A, que
        # nunca vuelve a tocar model_manifest.json) encuentra el manifiesto ya
        # completo, en vez de un _SUCCESS publicado sin el.
        model_manifest = {
            "protocol": sel.PROTOCOL_NAME, "dataset": dataset, "condition": pipeline.condition,
            "architecture": pipeline.architecture, "branch": pipeline.branch,
            "hyperparameters": pipeline.hyperparameters, "epochs": pipeline.epochs,
            "threshold": selection_cfg.threshold, "seed_base": selection_cfg.seed_base,
            "seed_parts": [dataset, pipeline.condition, "final"],
            "n_train_patients": int(data.dev_seg["patient_uid"].nunique()),
            "n_train_segments": int(len(data.dev_seg)),
            "n_test_patients": n_test_patients, "n_test_segments": int(len(data.test_seg)),
            "source_run_id": pipeline.source_run_id, "source_config_index": pipeline.source_config_index,
            "source_artifact": pipeline.source_artifact, "robustness_run_id": pipeline.robustness_run_id or None,
            "warnings": warnings_list,
            "hashes": {
                "pipeline_fingerprint": fingerprint["pipeline_fingerprint"],
                "selection_config_sha256": dmod.sha256_file(selection_cfg.path),
                "split_csv_sha256": selection_cfg.split_csv_sha256,
                "split_manifest_sha256": selection_cfg.split_manifest_sha256,
                "source_artifact_sha256": pipeline.source_artifact_sha256,
                "reference_config_sha256": dmod.sha256_file(pipeline.reference_config),
                "code_sha256": fingerprint.get("code_sha256"),
                "final_model_sha256": model_sha256,
                "final_data": dataset_final_input_hashes(data_root, dataset, pipeline.branch),
            },
            "git": art.git_state(),
            "created_at_utc": art._now_iso(),
        }
        _write_json(model_manifest, staging / "model_manifest.json")

        write_test_access_record(
            run_root, dataset, pipeline.condition,
            pipeline_fingerprint=fingerprint["pipeline_fingerprint"], model_sha256=model_sha256,
            status=TEST_ACCESS_PREDICTIONS_WRITTEN, split_csv_sha256=selection_cfg.split_csv_sha256,
            split_manifest_sha256=selection_cfg.split_manifest_sha256,
        )

        del model, data, eval_run
        cnn_model._release(device)

    write_reports(staging, cfg, selection_cfg, pipeline, negative_label_name, evaluation, logger)
    art.write_status(staging, STATUS_COMPLETED, {
        "balanced_accuracy": evaluation["metrics"]["balanced_accuracy"],
        "n_test_patients": int(evaluation["patients"]["patient_uid"].nunique()),
    })
    model_sha256 = dmod.sha256_file(staging / "final_model.pt")
    # Nunca se publica _SUCCESS sobre un staging incompleto: la misma lista de
    # aceptacion que protege la REUTILIZACION de un directorio ya publicado
    # (REQUIRED_PUBLISHED_FILES) se exige tambien aqui, ANTES del unico
    # publish_pipeline() que lo hace visible como completado.
    verify_published_completeness(staging)
    # Publicar PRIMERO, marcar COMPLETED en el registro externo DESPUES: asi ese
    # estado es una garantia real de que la publicacion ya ocurrio, no una
    # promesa escrita antes de intentarla (ver el guard de "COMPLETED sin
    # publicacion" en run_pipeline_for_dataset, que depende de este orden).
    published_target = publish_pipeline(staging)
    write_test_access_record(
        run_root, dataset, pipeline.condition,
        pipeline_fingerprint=fingerprint["pipeline_fingerprint"], model_sha256=model_sha256,
        status=TEST_ACCESS_COMPLETED, split_csv_sha256=selection_cfg.split_csv_sha256,
        split_manifest_sha256=selection_cfg.split_manifest_sha256,
    )
    logger.info(
        f"{dataset}/{pipeline.condition}: COMPLETED, BA test = {evaluation['metrics']['balanced_accuracy']:.4f} "
        f"-> {published_target}"
    )
    return {
        "dataset": dataset, "condition": pipeline.condition, "status": STATUS_COMPLETED,
        "metrics": evaluation["metrics"], "run_dir": published_target,
    }


def dry_run_check(
    selection_cfg: sel.SelectionConfig, datasets: list[str], data_root: Path, source_runs_root: Path,
) -> dict:
    """Valida, para cada dataset pedido, que el pipeline este aprobado, que el
    split no cambio, que sus hiperparametros coincidan con su procedencia y
    que existan los datos de la etapa final -sin cargar senales ni etiquetas
    del test."""
    rows: list[dict] = []
    ok = True

    def add(dataset: str, check: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and bool(passed)
        rows.append({"dataset": dataset, "check": check, "ok": bool(passed), "detail": detail})

    for dataset in datasets:
        try:
            pipeline = sel.validate_pipeline_for_execution(selection_cfg, dataset, source_runs_root)
            add(dataset, "seleccion_aprobada", True, f"{pipeline.condition} (epochs={pipeline.epochs})")
        except sel.SelectionError as exc:
            add(dataset, "seleccion_aprobada", False, str(exc))
            continue

        try:
            cfg = sel.validate_reference_config(pipeline)
            add(dataset, "reference_config", True, str(pipeline.reference_config))
        except sel.SelectionError as exc:
            add(dataset, "reference_config", False, str(exc))
            continue

        try:
            manifest = dmod.load_task_manifest(data_root, dataset, STAGE_SUBDIR)
        except (FileNotFoundError, ValueError) as exc:
            add(dataset, "manifiesto_etapa_final", False, str(exc))
            continue

        for key, expected in (
            ("protocol", "holdout-v3"), ("stage", STAGE_SUBDIR), ("dataset_scope", dataset), ("verdict", "PASS"),
        ):
            add(dataset, f"manifest_{key}", manifest.get(key) == expected, f"{manifest.get(key)!r} (esperado {expected!r})")
        add(dataset, "selection_status_at_generation",
            manifest.get("selection_status_at_generation") == "approved",
            str(manifest.get("selection_status_at_generation")))
        add(dataset, "counts_by_role", bool(manifest.get("counts_by_role")), str(manifest.get("counts_by_role")))

        # El preprocesamiento final debe haberse generado contra EXACTAMENTE la
        # seleccion congelada actual, no una version anterior (CORRECIONES.md
        # seccion 5): misma huella cruda que preprocessing/fold_denoising.py
        # calculo sobre [pipelines.<dataset>] al publicar esa salida.
        try:
            current_fp = sel.final_pipeline_fingerprint(selection_cfg.path, dataset)
            recorded_fp = manifest.get("selection_pipeline_fingerprint")
            linked = recorded_fp is not None and recorded_fp == current_fp
            add(dataset, "selection_pipeline_fingerprint", linked,
                "coincide con la seleccion actual" if linked else
                "el preprocesamiento final se genero con OTRA seleccion; regenerelo con "
                "preprocessing/fold_denoising.py --protocol holdout-v3 --stage final")
        except sel.SelectionError as exc:
            add(dataset, "selection_pipeline_fingerprint", False, str(exc))

        for key, current in (
            ("holdout_splits_csv_sha256", selection_cfg.split_csv_sha256),
            ("holdout_splits_manifest_sha256", selection_cfg.split_manifest_sha256),
        ):
            linked = manifest.get(key) is not None and manifest.get(key) == current
            add(dataset, key, linked,
                "coincide con la seleccion congelada" if linked else
                "el preprocesamiento final quedo desincronizado del split congelado; regenerelo con "
                "preprocessing/fold_denoising.py --protocol holdout-v3 --stage final")

        # Integridad por HASH de archivo: nunca se parsea segments.csv (etiquetas
        # del test) ni se hace np.load() de las ramas .npy -ver CORRECIONES.md
        # seccion 3-. dmod.sha256_file solo lee bytes y los resume; no interpreta
        # ningun contenido.
        root = dmod.dataset_root(data_root, dataset, STAGE_SUBDIR)
        manifest_hashes = manifest.get("output_hashes", {})
        for name in ("segments.csv", f"segments_{pipeline.branch}.npy", "preprocessing_params.json"):
            path = root / name
            if not path.is_file():
                add(dataset, f"sha256_{name}", False, f"no existe {path}")
                continue
            actual = dmod.sha256_file(path)
            expected = manifest_hashes.get(name)
            match = expected is not None and actual == expected
            add(dataset, f"sha256_{name}", match,
                "coincide con manifest.json" if match else "no coincide con manifest.json")

    return {"ok": ok, "checks": pd.DataFrame(rows)}
