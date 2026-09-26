"""Nucleo compartido, sin PyTorch, del protocolo holdout + validacion cruzada v3
(``PLAN-EXPERIMENTO FINAL.md``), usado por la SVM-RBF, la CNN y la CRNN.

Procedimiento, por pipeline (modelo x dataset x condicion):

1. Solo existe el 80 % de desarrollo (``preprocessing/fold_denoising.py
   --protocol holdout-v3``): la prueba externa esta bloqueada y ninguna
   funcion de este modulo la lee ni la necesita.
2. Cada CONFIGURACION EXACTA se evalua en los 5 folds internos: se entrena
   desde cero con cuatro folds y se valida en el quinto. La unidad de trabajo
   es ``(configuracion, fold)``; se publica de forma atomica (staging +
   ``_SUCCESS``) y es lo que ``--resume`` omite.
3. Los resultados de UNA misma configuracion se agrupan y se elige UNA sola
   configuracion global por pipeline (nunca un ganador distinto por fold ni
   predicciones mezcladas de ganadores distintos). Una configuracion
   incompleta o fallida en algun fold no puede ser seleccionada.
4. No se guarda ningun modelo (``final_model.enabled = false``): los modelos
   internos de cada unidad son temporales. Tampoco se toca la prueba externa
   (``outer_test.enabled = false``).

Este modulo contiene todo lo que no depende del modelo: generacion determinista
de candidatos, almacenamiento de unidades, seleccion, agregacion de
predicciones (segmento -> grabacion -> paciente y paciente-dispositivo),
metricas globales/por fuente/por dispositivo, artefactos, huella para
``--resume`` y ``--dry-run``. El entrenamiento en si vive en
``holdout_svm.py`` y ``holdout_cnn.py``.

Numeracion: en este protocolo TODO es 0-indexado -carpeta de datos
``cv/fold_00..fold_04``, carpeta de resultados ``cv/config_NNN/fold_NN`` y la
columna ``fold``-.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import artifacts as art
from . import data as dmod
from . import evaluation as ev
from . import run_experiment as rexp
from . import splits as sp
from .models import svm_rbf as svm_model

PROTOCOL_NAME = dmod.HOLDOUT_PROTOCOL
CV_SUBDIR = "cv"
N_INNER_FOLDS = 5

# Criterios de seleccion, en este orden exacto (PLAN-EXPERIMENTO FINAL.md 3):
# mayor balanced accuracy media, mayor macro F1 medio, mayor media del menor
# recall entre ambas clases, menor desviacion estandar de balanced accuracy y,
# como desempate final, el orden determinista de la configuracion.
SELECTION_CRITERIA = (
    "balanced_accuracy_mean",
    "macro_f1_mean",
    "min_class_recall_mean",
    "balanced_accuracy_std",
    "config_order",
)
# criterio -> (columna de config_summary a comparar, orden ascendente?)
_CRITERION_SORT = {
    "balanced_accuracy_mean": ("balanced_accuracy_mean", False),
    "macro_f1_mean": ("macro_f1_mean", False),
    "min_class_recall_mean": ("min_class_recall_mean", False),
    "balanced_accuracy_std": ("balanced_accuracy_std_rank", True),
    "config_order": ("config_index", True),
}

# Metricas por paciente que se agregan entre folds (media y desviacion).
SUMMARY_METRICS = (
    "accuracy", "balanced_accuracy", "recall_copd", "recall_negative",
    "macro_f1", "min_class_recall", "auroc", "auprc_copd",
)
# Metricas guardadas por unidad (una fila de cv_search_results.csv).
UNIT_METRIC_COLUMNS = (
    "n_patients", "accuracy", "balanced_accuracy", "recall_copd", "recall_negative",
    "precision_copd", "f1_copd", "macro_f1", "min_class_recall", "auroc", "auprc_copd",
    "tn", "fp", "fn", "tp", "threshold",
)
SCORE_COLUMNS = ["segment_id", "audio_id", "patient_uid", "target_label", "source_dataset", "device", "score"]

STATUS_UNIT_COMPLETED = "COMPLETED"
STATUS_UNIT_FAILED = "FAILED"
STATUS_UNIT_MISSING = "MISSING"


class SelectionError(RuntimeError):
    """Ninguna configuracion completa: no hay nada que seleccionar."""


# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

def cv_fold_ref(fold_id: int) -> str:
    """Subruta ``cv/fold_XX`` bajo cada dataset, que ``data.dataset_root``
    acepta como ``fold_id`` de tipo ``str``."""
    return f"{CV_SUBDIR}/fold_{fold_id:02d}"


def validate_holdout_config(cfg: dict) -> None:
    """Falla ANTES de tocar datos si el TOML no describe exactamente este
    protocolo: prueba externa y modelo final desactivados de forma explicita
    (no por omision), 5 folds internos y criterios de seleccion del plan."""
    if not dmod.is_holdout_protocol(cfg):
        raise RuntimeError(f"el TOML debe declarar protocol = {PROTOCOL_NAME!r}")
    if "folds" in cfg:
        raise RuntimeError("un TOML holdout_cv_v3 no debe declarar [folds] (eso activa el protocolo fold-aware v2)")
    for section in ("outer_test", "final_model"):
        enabled = cfg.get(section, {}).get("enabled")
        if enabled is not False:
            raise RuntimeError(
                f"el protocolo {PROTOCOL_NAME} exige '[{section}]\\nenabled = false' explicito en el TOML "
                f"(valor actual: {enabled!r}); en esta etapa no se usa la prueba externa ni se crea un modelo definitivo"
            )
    holdout = cfg.get("holdout", {})
    for key in ("split_csv", "split_manifest", "n_splits"):
        if key not in holdout:
            raise RuntimeError(f"falta [holdout] {key} en el TOML")
    if int(holdout["n_splits"]) != N_INNER_FOLDS:
        raise RuntimeError(f"[holdout] n_splits debe ser {N_INNER_FOLDS} (valor actual: {holdout['n_splits']})")
    criteria = tuple(cfg.get("selection", {}).get("criteria", ()))
    if criteria != SELECTION_CRITERIA:
        raise RuntimeError(
            f"[selection] criteria debe ser exactamente {list(SELECTION_CRITERIA)} (valor actual: {list(criteria)})"
        )
    if cfg.get("metrics", {}).get("primary") != "balanced_accuracy":
        raise RuntimeError("[metrics] primary debe ser 'balanced_accuracy' (metrica principal del protocolo)")


def holdout_split_paths(cfg: dict) -> tuple[Path, Path]:
    holdout = cfg["holdout"]
    return dmod.REPO_ROOT / holdout["split_csv"], dmod.REPO_ROOT / holdout["split_manifest"]


def load_splits(cfg: dict, datasets: Iterable[str]) -> dict[str, sp.HoldoutSplit]:
    csv_path, manifest_path = holdout_split_paths(cfg)
    return {ds: sp.load_holdout_split(csv_path, manifest_path, ds) for ds in sorted(set(datasets))}


# ---------------------------------------------------------------------------
# Candidatos de configuracion (deterministas)
# ---------------------------------------------------------------------------

SVM_HYPERPARAMETER_KEYS = ("C", "gamma")
NN_HYPERPARAMETER_KEYS = ("lr", "dropout", "weight_decay", "batch_size")


def svm_candidates(cfg: dict) -> list[dict]:
    """Las 30 combinaciones (C, gamma), en orden C-mayor luego gamma: el
    ``config_index`` es la posicion en esa lista y es el desempate final."""
    grid = svm_model.hyperparameter_grid(cfg)
    if len(set(grid)) != len(grid):
        raise ValueError("la cuadricula de la SVM tiene combinaciones repetidas")
    return [{"config_index": i, "C": float(c), "gamma": gamma} for i, (c, gamma) in enumerate(grid)]


def nn_candidates(cfg: dict) -> list[dict]:
    """Configuraciones unicas de la CNN/CRNN: muestreo determinista SIN
    reemplazo del producto cartesiano de ``[search]`` (108 combinaciones ->
    ``n_configurations``), con ``[search] random_state``. Se devuelven en el
    orden del producto cartesiano, de modo que ``config_index`` es estable y
    legible. CNN y CRNN leen la misma seccion, asi que reciben la misma lista.
    """
    search = cfg["search"]
    if search.get("mode", "random_without_replacement") != "random_without_replacement":
        raise ValueError(f"[search] mode desconocido: {search.get('mode')!r}")
    space = list(itertools.product(
        search["learning_rate"], search["dropout"], search["weight_decay"], search["batch_size"],
    ))
    if len(set(space)) != len(space):
        raise ValueError("el espacio de busqueda tiene combinaciones repetidas")
    n_configurations = int(search["n_configurations"])
    if not 1 <= n_configurations <= len(space):
        raise ValueError(f"n_configurations={n_configurations} fuera de 1..{len(space)}")

    rng = np.random.RandomState(int(search["random_state"]))
    chosen = sorted(int(i) for i in rng.choice(len(space), size=n_configurations, replace=False))
    return [
        {
            "config_index": k,
            "lr": float(space[i][0]), "dropout": float(space[i][1]),
            "weight_decay": float(space[i][2]), "batch_size": int(space[i][3]),
        }
        for k, i in enumerate(chosen)
    ]


def hyperparameter_keys_for(model_name: str) -> tuple[str, ...]:
    return SVM_HYPERPARAMETER_KEYS if model_name == "svm_rbf" else NN_HYPERPARAMETER_KEYS


def _hp_cell(key: str, value):
    """Valor de hiperparametro para una celda de CSV: ``gamma`` mezcla
    "scale" y flotantes, asi que se guarda siempre como texto."""
    return str(value) if key == "gamma" else value


def median_best_epoch(values) -> int | None:
    """Mediana redondeada al entero mas cercano (minimo 1) de las mejores
    epocas por fold; ``None`` si no hay ninguna (SVM)."""
    array = np.asarray([v for v in values if v is not None and not pd.isna(v)], dtype=np.float64)
    if array.size == 0:
        return None
    return max(1, int(math.floor(float(np.median(array)) + 0.5)))


# ---------------------------------------------------------------------------
# Comprobaciones de fuga
# ---------------------------------------------------------------------------

def verify_fold_segments_against_split(segments: pd.DataFrame, split: sp.HoldoutSplit, fold_id: int) -> None:
    """Los segmentos de un fold preprocesado deben contener exactamente a los
    pacientes de train y validation de ``holdout_splits.csv`` para ese fold, y
    a ninguno de la prueba externa bloqueada."""
    tag = f"{split.dataset_scope}/{cv_fold_ref(fold_id)}"
    train_patients, val_patients = split.cv_split(fold_id)
    present = set(segments["patient_uid"])
    split.assert_no_blocked_patients(present, tag)

    expected = set(train_patients) | set(val_patients)
    if present != expected:
        raise RuntimeError(
            f"{tag}: los pacientes de segments.csv no coinciden con el split "
            f"(sobran {sorted(present - expected)[:5]}, faltan {sorted(expected - present)[:5]})"
        )
    if "role" in segments.columns:
        unknown = set(segments["role"]) - {sp.ROLE_TRAIN, sp.ROLE_VALIDATION}
        if unknown:
            raise RuntimeError(f"{tag}: roles no permitidos en un fold de validacion cruzada: {sorted(unknown)}")
        role_by_patient = segments.drop_duplicates("patient_uid").set_index("patient_uid")["role"]
        for role, ids in ((sp.ROLE_TRAIN, train_patients), (sp.ROLE_VALIDATION, val_patients)):
            wrong = [p for p in ids if role_by_patient.get(p) != role]
            if wrong:
                raise RuntimeError(f"{tag}: pacientes con rol distinto de {role!r} respecto del split: {wrong[:5]}")


# ---------------------------------------------------------------------------
# Unidades de trabajo (configuracion, fold): almacenamiento atomico
# ---------------------------------------------------------------------------

@dataclass
class UnitOutput:
    """Resultado de entrenar UNA configuracion en UN fold (solo validacion)."""

    val_scores: pd.DataFrame          # columnas SCORE_COLUMNS, un renglon por segmento de validacion
    best_epoch: int | None = None     # solo redes
    epochs_run: int | None = None
    stopped_early: bool | None = None
    history: pd.DataFrame | None = None
    seconds: float = 0.0


@dataclass
class UnitFailure:
    error: str
    error_type: str


def build_val_scores(val_seg: pd.DataFrame, scores) -> pd.DataFrame:
    """Puntajes de validacion por segmento, con las columnas de SCORE_COLUMNS."""
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(val_seg),):
        raise ValueError(f"{len(values)} puntajes para {len(val_seg)} segmentos de validacion")
    if not np.isfinite(values).all():
        raise FloatingPointError("puntajes de validacion no finitos")
    source_col = next((c for c in ("source_dataset", "dataset") if c in val_seg.columns), None)
    device = val_seg["device"].fillna("unknown").astype(str).to_numpy() if "device" in val_seg.columns else "unknown"
    return pd.DataFrame({
        "segment_id": val_seg["segment_id"].to_numpy(),
        "audio_id": val_seg["audio_id"].to_numpy(),
        "patient_uid": val_seg["patient_uid"].to_numpy(),
        "target_label": val_seg["target_label"].to_numpy(dtype=np.int64),
        "source_dataset": val_seg[source_col].astype(str).to_numpy() if source_col else "",
        "device": device,
        "score": values,
    })[SCORE_COLUMNS]


def unit_dir(run_root: Path, dataset: str, condition: str, config_index: int, fold_id: int) -> Path:
    return (
        art.condition_dir(run_root, dataset, condition) / CV_SUBDIR
        / f"config_{config_index:03d}" / f"fold_{fold_id:02d}"
    )


def unit_staging_dir(run_root: Path, dataset: str, condition: str, config_index: int, fold_id: int) -> Path:
    target = unit_dir(run_root, dataset, condition, config_index, fold_id)
    return target.with_name(target.name + "_staging")


def unit_failure_marker(run_root: Path, dataset: str, condition: str, config_index: int, fold_id: int) -> Path:
    target = unit_dir(run_root, dataset, condition, config_index, fold_id)
    return target.with_name(target.name + "_FAILED.json")


def is_unit_complete(run_root: Path, dataset: str, condition: str, config_index: int, fold_id: int) -> bool:
    return (unit_dir(run_root, dataset, condition, config_index, fold_id) / "_SUCCESS").is_file()


def cleanup_abandoned_unit_staging(run_root: Path) -> list[Path]:
    """Borra las unidades ``fold_NN_staging`` sin ``_SUCCESS`` de una
    ejecucion interrumpida: un ``--resume`` nunca las cuenta como resultado.
    (``artifacts.cleanup_abandoned_staging`` solo ve los folds de los
    protocolos anteriores, a otra profundidad.)"""
    removed: list[Path] = []
    datasets_dir = Path(run_root) / "datasets"
    if not datasets_dir.is_dir():
        return removed
    for staging in sorted(datasets_dir.glob(f"*/*/{CV_SUBDIR}/config_*/fold_*_staging")):
        shutil.rmtree(staging, ignore_errors=True)
        removed.append(staging)
    return removed


def unit_metrics(val_scores: pd.DataFrame, negative_label_name: str, threshold: float) -> tuple[dict, pd.DataFrame]:
    """Metricas por paciente de una unidad (segmento -> grabacion -> paciente)
    mas el menor recall entre clases y un alias generico del recall negativo."""
    patients = ev.aggregate_segment_to_patient(val_scores)
    metrics = ev.compute_patient_metrics(
        patients["target_label"].to_numpy(), patients["score"].to_numpy(), negative_label_name, threshold,
    )
    neg_key = negative_label_name.strip().lower()
    metrics["recall_negative"] = metrics[f"recall_{neg_key}"]
    metrics["min_class_recall"] = float(min(metrics["recall_copd"], metrics["recall_negative"]))
    return metrics, patients


def write_unit_failure(
    run_root: Path, spec: dmod.ConditionSpec, config_index: int, fold_id: int, failure: UnitFailure,
) -> None:
    marker = unit_failure_marker(run_root, spec.dataset, spec.condition, config_index, fold_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({
            "config_index": config_index, "fold": fold_id, "error": failure.error,
            "type": failure.error_type, "failed_at_utc": art._now_iso(),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def publish_unit(
    run_root: Path,
    spec: dmod.ConditionSpec,
    model_name: str,
    config: dict,
    fold_id: int,
    output: UnitOutput,
    negative_label_name: str,
    threshold: float,
    n_train_patients: int,
    n_val_patients: int,
    hyperparameter_keys: tuple[str, ...],
) -> dict:
    """Escribe una unidad en staging y la publica con ``_SUCCESS``. Devuelve
    su registro (el mismo ``unit_result.json``)."""
    config_index = int(config["config_index"])
    staging = unit_staging_dir(run_root, spec.dataset, spec.condition, config_index, fold_id)
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    metrics, _patients = unit_metrics(output.val_scores, negative_label_name, threshold)
    record = {
        "model": model_name, "dataset": spec.dataset, "condition": spec.condition,
        "config_index": config_index, "fold": int(fold_id),
        "hyperparameters": {k: config[k] for k in hyperparameter_keys},
        "metrics": metrics,
        "best_epoch": output.best_epoch, "epochs_run": output.epochs_run,
        "stopped_early": output.stopped_early, "seconds": float(output.seconds),
        "n_train_patients": int(n_train_patients), "n_val_patients": int(n_val_patients),
        "n_val_segments": int(len(output.val_scores)),
    }
    output.val_scores.to_csv(staging / "val_segment_scores.csv", index=False, lineterminator="\n")
    if output.history is not None:
        output.history.to_csv(staging / "history.csv", index=False, lineterminator="\n")
    (staging / "unit_result.json").write_text(
        json.dumps(art._json_safe(record), indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    art.publish_fold(staging)

    marker = unit_failure_marker(run_root, spec.dataset, spec.condition, config_index, fold_id)
    if marker.is_file():
        marker.unlink()
    return record


# ---------------------------------------------------------------------------
# Ejecucion de una condicion: 5 folds x configuraciones, con reanudacion
# ---------------------------------------------------------------------------

def run_condition_cv(
    *,
    run_root: Path,
    spec: dmod.ConditionSpec,
    model_name: str,
    split: sp.HoldoutSplit,
    configs: list[dict],
    hyperparameter_keys: tuple[str, ...],
    negative_label_name: str,
    threshold: float,
    fold_ids: list[int],
    prepare_fold: Callable,
    evaluate_fold: Callable,
    release_fold: Callable | None,
    logger,
) -> None:
    """Entrena/valida todas las unidades (configuracion, fold) que aun no
    estan publicadas. Los fallos de una unidad se registran (marcador
    ``fold_NN_FAILED.json``) y NO detienen a las demas: esa configuracion
    queda incompleta y no podra ser seleccionada.

    ``prepare_fold(fold_id, train_patients, val_patients)`` carga los datos de
    un fold y devuelve un contexto; ``evaluate_fold(context, pending_configs)``
    produce ``(config, UnitOutput | UnitFailure)`` de forma perezosa -cada
    unidad se publica apenas termina, asi una interrupcion pierde a lo sumo la
    unidad en curso-; ``release_fold(context)`` libera memoria.
    """
    tag = f"{spec.dataset}/{spec.condition}"
    for fold_id in fold_ids:
        pending = [
            c for c in configs
            if not is_unit_complete(run_root, spec.dataset, spec.condition, c["config_index"], fold_id)
        ]
        if not pending:
            logger.info(f"{tag} fold {fold_id}: las {len(configs)} configuracion(es) ya estaban publicadas")
            continue

        train_patients, val_patients = split.cv_split(fold_id)
        try:
            context = prepare_fold(fold_id, train_patients, val_patients)
        except Exception as exc:  # noqa: BLE001 - se registra, nunca se omite en silencio
            logger.exception(f"{tag} fold {fold_id}: no se pudieron preparar los datos")
            for config in pending:
                write_unit_failure(
                    run_root, spec, config["config_index"], fold_id,
                    UnitFailure(error=str(exc), error_type=type(exc).__name__),
                )
            continue

        logger.info(
            f"{tag} fold {fold_id}: {len(train_patients)}/{len(val_patients)} pacientes train/validation, "
            f"{len(pending)} de {len(configs)} configuracion(es) por evaluar"
        )
        done: set[int] = set()
        try:
            for config, result in evaluate_fold(context, pending):
                index = int(config["config_index"])
                done.add(index)
                if isinstance(result, UnitFailure):
                    write_unit_failure(run_root, spec, index, fold_id, result)
                    logger.error(f"{tag} fold {fold_id} config {index}: FALLO {result.error_type}: {result.error}")
                    continue
                try:
                    record = publish_unit(
                        run_root, spec, model_name, config, fold_id, result, negative_label_name, threshold,
                        len(train_patients), len(val_patients), hyperparameter_keys,
                    )
                except Exception as exc:  # noqa: BLE001
                    write_unit_failure(run_root, spec, index, fold_id, UnitFailure(str(exc), type(exc).__name__))
                    logger.exception(f"{tag} fold {fold_id} config {index}: no se pudo publicar la unidad")
                    continue
                epoch_note = f" mejor_epoca={record['best_epoch']}" if record["best_epoch"] is not None else ""
                logger.info(
                    f"{tag} fold {fold_id} config {index}: OK val_BA={record['metrics']['balanced_accuracy']:.3f}"
                    f"{epoch_note} ({record['seconds'] / 60:.1f} min)"
                )
        except Exception as exc:  # noqa: BLE001 - fallo fuera de una unidad concreta (p. ej. memoria)
            logger.exception(f"{tag} fold {fold_id}: la evaluacion se interrumpio")
            for config in pending:
                if int(config["config_index"]) not in done:
                    write_unit_failure(
                        run_root, spec, config["config_index"], fold_id,
                        UnitFailure(error=str(exc), error_type=type(exc).__name__),
                    )
        finally:
            if release_fold is not None:
                release_fold(context)


# ---------------------------------------------------------------------------
# Tablas de busqueda, resumen por configuracion y seleccion global
# ---------------------------------------------------------------------------

def load_unit_records(
    run_root: Path, spec: dmod.ConditionSpec, configs: list[dict], fold_ids: list[int],
) -> list[dict]:
    """Un registro por (configuracion, fold): el ``unit_result.json`` si esta
    publicada, o su estado (FAILED con el error del marcador / MISSING)."""
    records = []
    for config in configs:
        for fold_id in fold_ids:
            index = int(config["config_index"])
            directory = unit_dir(run_root, spec.dataset, spec.condition, index, fold_id)
            if (directory / "_SUCCESS").is_file():
                record = json.loads((directory / "unit_result.json").read_text(encoding="utf-8"))
                record["status"] = STATUS_UNIT_COMPLETED
            else:
                marker = unit_failure_marker(run_root, spec.dataset, spec.condition, index, fold_id)
                error = None
                status = STATUS_UNIT_MISSING
                if marker.is_file():
                    status = STATUS_UNIT_FAILED
                    error = json.loads(marker.read_text(encoding="utf-8")).get("error")
                record = {
                    "config_index": index, "fold": int(fold_id), "status": status, "error": error,
                    "hyperparameters": {k: config[k] for k in config if k != "config_index"},
                }
            records.append(record)
    return records


def search_results_frame(
    records: list[dict], spec: dmod.ConditionSpec, model_name: str, negative_label_name: str,
    hyperparameter_keys: tuple[str, ...],
) -> pd.DataFrame:
    """``cv_search_results.csv``: TODAS las configuraciones y folds, tambien
    los fallidos o ausentes (con sus metricas en NaN)."""
    rows = []
    for r in records:
        row = {
            "model": model_name, "dataset": spec.dataset, "condition": spec.condition,
            "negative_class": negative_label_name,
            "config_index": int(r["config_index"]), "fold": int(r["fold"]), "status": r["status"],
        }
        for key in hyperparameter_keys:
            row[key] = _hp_cell(key, r["hyperparameters"].get(key))
        metrics = r.get("metrics", {})
        for column in UNIT_METRIC_COLUMNS:
            row[column] = metrics.get(column, np.nan)
        row["best_epoch"] = r.get("best_epoch", np.nan)
        row["epochs_run"] = r.get("epochs_run", np.nan)
        row["stopped_early"] = r.get("stopped_early", np.nan)
        row["seconds"] = r.get("seconds", np.nan)
        row["error"] = r.get("error")
        rows.append(row)
    frame = pd.DataFrame(rows)
    for column in ("best_epoch", "epochs_run", "seconds"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["config_index", "fold"]).reset_index(drop=True)


def config_summary_frame(
    search: pd.DataFrame, expected_folds: int, hyperparameter_keys: tuple[str, ...],
) -> pd.DataFrame:
    """``cv_config_summary.csv``: media y desviacion estandar (ddof=1) entre
    folds de cada configuracion. ``complete`` exige los ``expected_folds``
    folds completados: solo esas pueden seleccionarse."""
    completed = search.loc[search["status"] == STATUS_UNIT_COMPLETED]
    base = search.drop_duplicates("config_index")[["config_index", *hyperparameter_keys]]
    failed = (
        search.loc[search["status"] != STATUS_UNIT_COMPLETED]
        .groupby("config_index").size().rename("n_folds_incomplete").reset_index()
    )
    aggregations = {f"{m}_mean": (m, "mean") for m in SUMMARY_METRICS}
    aggregations.update({f"{m}_std": (m, "std") for m in SUMMARY_METRICS})
    aggregations["n_folds_completed"] = ("fold", "nunique")
    aggregations["median_best_epoch"] = ("best_epoch", "median")
    grouped = completed.groupby("config_index").agg(**aggregations).reset_index()

    summary = base.merge(grouped, on="config_index", how="left").merge(failed, on="config_index", how="left")
    summary["n_folds_completed"] = summary["n_folds_completed"].fillna(0).astype(int)
    summary["n_folds_incomplete"] = summary["n_folds_incomplete"].fillna(0).astype(int)
    summary["complete"] = summary["n_folds_completed"] == int(expected_folds)
    return summary.sort_values("config_index").reset_index(drop=True)


def select_best_configuration(summary: pd.DataFrame) -> tuple[pd.Series, dict]:
    """UNA configuracion global: los criterios de ``SELECTION_CRITERIA`` en su
    orden, solo entre las completas. Devuelve la fila ganadora y un dict con
    los criterios aplicados y cual la decidio."""
    candidates = summary.loc[summary["complete"]]
    if candidates.empty:
        raise SelectionError("ninguna configuracion completo todos los folds: no se puede seleccionar")

    ranked = candidates.assign(
        balanced_accuracy_std_rank=candidates["balanced_accuracy_std"].fillna(0.0),
    )
    columns = [_CRITERION_SORT[c][0] for c in SELECTION_CRITERIA]
    ascending = [_CRITERION_SORT[c][1] for c in SELECTION_CRITERIA]
    ranked = ranked.sort_values(columns, ascending=ascending, kind="mergesort")

    top = ranked.iloc[0]
    decided_by = "unico_candidato"
    applied = [SELECTION_CRITERIA[0]]
    if len(ranked) > 1:
        second = ranked.iloc[1]
        decided_by = SELECTION_CRITERIA[-1]
        for position, criterion in enumerate(SELECTION_CRITERIA):
            column = _CRITERION_SORT[criterion][0]
            if not math.isclose(float(top[column]), float(second[column]), rel_tol=0.0, abs_tol=1e-12):
                decided_by = criterion
                applied = list(SELECTION_CRITERIA[: position + 1])
                break
        else:
            applied = list(SELECTION_CRITERIA)
    info = {
        "criteria": list(SELECTION_CRITERIA),
        "criteria_applied": applied,
        "decided_by": decided_by,
        "n_candidates": int(len(summary)),
        "n_complete_candidates": int(len(candidates)),
        "n_incomplete_candidates": int(len(summary) - len(candidates)),
    }
    return top, info


def select_reused_configuration(summary: pd.DataFrame, config_index: int, source_condition: str) -> tuple[pd.Series, dict]:
    """``no_dn_aug``: no hay busqueda. Se reutiliza la configuracion global ya
    elegida para ``no_dn`` (mismo modelo y dataset); solo se exige que este
    completa en sus folds."""
    rows = summary.loc[summary["config_index"] == int(config_index)]
    if rows.empty or not bool(rows.iloc[0]["complete"]):
        raise SelectionError(
            f"la configuracion {config_index} (reutilizada de {source_condition}) no completo todos los folds"
        )
    info = {
        "criteria": [],
        "criteria_applied": [],
        "decided_by": f"reused_from:{source_condition}",
        "n_candidates": 1,
        "n_complete_candidates": 1,
        "n_incomplete_candidates": 0,
    }
    return rows.iloc[0], info


# ---------------------------------------------------------------------------
# Predicciones agrupadas y artefactos de la configuracion ganadora
# ---------------------------------------------------------------------------

def winner_segment_predictions(
    run_root: Path, spec: dmod.ConditionSpec, config_index: int, fold_ids: list[int],
) -> pd.DataFrame:
    """Predicciones de validacion por segmento de la configuracion ganadora,
    de los ``fold_ids``: cada paciente de desarrollo aparece en UN solo fold."""
    frames = []
    for fold_id in fold_ids:
        path = unit_dir(run_root, spec.dataset, spec.condition, config_index, fold_id) / "val_segment_scores.csv"
        frame = pd.read_csv(
            path, dtype={"segment_id": str, "audio_id": str, "patient_uid": str, "source_dataset": str, "device": str},
            keep_default_na=False,
        )
        frames.append(frame.assign(fold=int(fold_id)))
    return pd.concat(frames, ignore_index=True)


def aggregate_winner_predictions(segments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(grabaciones, pacientes) a partir de las predicciones por segmento.
    Exige una prediccion de validacion por paciente."""
    folds_per_patient = segments.groupby("patient_uid")["fold"].nunique()
    if (folds_per_patient != 1).any():
        raise RuntimeError(
            f"pacientes validados en mas de un fold: {folds_per_patient[folds_per_patient != 1].index.tolist()[:5]}"
        )
    recordings = ev.aggregate_segment_to_recording(segments)
    by_audio = segments.drop_duplicates("audio_id").set_index("audio_id")
    for column in ("fold", "source_dataset", "device"):
        recordings[column] = recordings["audio_id"].map(by_audio[column])
    patients = ev.aggregate_recording_to_patient(recordings)
    by_patient = segments.drop_duplicates("patient_uid").set_index("patient_uid")
    for column in ("fold", "source_dataset"):
        patients[column] = patients["patient_uid"].map(by_patient[column])
    return recordings, patients


def curve_frames(y_true: np.ndarray, y_score: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curvas ROC y precision-recall (COPD positivo) de los pacientes agrupados."""
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score, pos_label=ev.POSITIVE_LABEL)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score, pos_label=ev.POSITIVE_LABEL)
    roc = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds})
    pr = pd.DataFrame({"precision": precision, "recall": recall, "threshold": np.append(pr_thresholds, np.nan)})
    return roc, pr


def fold_metrics_frame(records: list[dict], hyperparameter_keys: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"fold": int(r["fold"])}
        row.update({k: r["metrics"].get(k, np.nan) for k in UNIT_METRIC_COLUMNS})
        row["best_epoch"] = r.get("best_epoch")
        row["n_val_patients"] = r["n_val_patients"]
        row.update({k: _hp_cell(k, r["hyperparameters"][k]) for k in hyperparameter_keys})
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)


def metrics_summary_frame(fold_metrics: pd.DataFrame, pooled: dict) -> pd.DataFrame:
    """``cv_metrics_summary.csv``: media y desviacion entre folds, y las
    metricas agrupadas con UNA prediccion de validacion por paciente."""
    numeric = fold_metrics[list(SUMMARY_METRICS)]
    return pd.DataFrame([
        {"statistic": "fold_mean", "n": int(len(fold_metrics)), **numeric.mean().to_dict()},
        {"statistic": "fold_std", "n": int(len(fold_metrics)), **numeric.std(ddof=1).to_dict()},
        {"statistic": "pooled", "n": int(pooled["n_patients"]), **{m: pooled[m] for m in SUMMARY_METRICS}},
    ])


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(art._json_safe(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_group_artifacts(
    cdir: Path, segments: pd.DataFrame, patients: pd.DataFrame, negative_label_name: str, threshold: float,
) -> None:
    """Analisis descriptivo, NUNCA usado para seleccionar: por fuente (solo si
    hay mas de una, p. ej. COMBINED) y por dispositivo."""
    target_names = (negative_label_name, "COPD")

    sources = patients["source_dataset"].replace("", np.nan).dropna()
    if sources.nunique() >= 2:
        _write_csv(ev.compute_metrics_by_source(patients, "source_dataset", negative_label_name, threshold),
                   cdir / "cv_metrics_by_source.csv")
        _write_csv(ev.classification_report_by_source_df(patients, "source_dataset", target_names, threshold),
                   cdir / "cv_classification_report_by_source.csv")
        _write_csv(ev.confusion_matrix_by_source_df(patients, "source_dataset", target_names, threshold),
                   cdir / "cv_confusion_matrix_by_source.csv")

    patient_device = ev.aggregate_patient_device(segments)
    _write_csv(ev.compute_metrics_by_device(patient_device, negative_label_name, threshold),
               cdir / "cv_metrics_by_device.csv")
    _write_csv(ev.counts_by_device(segments, negative_label_name), cdir / "cv_counts_by_device.csv")
    _write_csv(ev.confusion_matrix_by_source_df(patient_device, "device", target_names, threshold),
               cdir / "cv_confusion_matrix_by_device.csv")


def write_figures(
    cdir: Path, cfg_used: dict, y_true: np.ndarray, y_score: np.ndarray, target_names: tuple[str, str],
    threshold: float, fold_metrics: pd.DataFrame, negative_label_name: str, logger, tag: str,
) -> None:
    """Figuras (formatos y dpi de ``[figures]``) de la configuracion ganadora,
    en ``<condicion>/figures/``: matriz de confusion, curvas ROC y
    precision-recall (pacientes agrupados) y metricas por fold. Reutiliza las
    funciones de ``artifacts.py`` que ya usaban los experimentos anteriores.
    Un fallo al dibujar se registra pero no invalida los resultados, que ya
    estan guardados como CSV."""
    figures = cdir / "figures"
    # Las funciones de artifacts.py escriben primero un CSV junto a la figura: la carpeta debe existir.
    figures.mkdir(parents=True, exist_ok=True)
    # plot_metrics_by_fold espera la columna recall_<clase negativa> de cada dataset.
    neg_key = negative_label_name.strip().lower()
    by_fold = fold_metrics.assign(**{f"recall_{neg_key}": fold_metrics["recall_negative"]})
    jobs = (
        ("confusion_matrix", lambda: art.plot_confusion_matrix(
            y_true, y_score, target_names, figures / "confusion_matrix", cfg_used, threshold)),
        ("roc", lambda: art.plot_roc_curve(y_true, y_score, figures / "roc", cfg_used)),
        ("pr", lambda: art.plot_pr_curve(y_true, y_score, figures / "pr", cfg_used)),
        ("metrics_by_fold", lambda: art.plot_metrics_by_fold(
            by_fold, negative_label_name, figures / "metrics_by_fold", cfg_used)),
    )
    for name, draw in jobs:
        try:
            draw()
        except Exception:  # noqa: BLE001 - la figura no debe tumbar una condicion ya calculada
            logger.exception(f"{tag}: no se pudo generar la figura {name}")


def summarize_condition(
    *,
    run_root: Path,
    spec: dmod.ConditionSpec,
    cfg_used: dict,
    model_name: str,
    configs: list[dict],
    hyperparameter_keys: tuple[str, ...],
    fold_ids: list[int],
    negative_label_name: str,
    threshold: float,
    hyperparameter_source: str,
    reused_config_index: int | None,
    provenance: dict,
    logger,
) -> dict:
    """Agrupa las unidades de la condicion, elige la configuracion global y
    escribe todos los artefactos ``cv_*``, ``best_hyperparameters.json``,
    ``resolved_config.toml``, ``run_manifest.json`` y ``status.json`` en
    ``datasets/<dataset>/<condicion>/``. Nunca lanza por falta de unidades: en
    ese caso la condicion queda FAILED y los archivos que si se pueden
    escribir (busqueda y resumen por configuracion) se escriben igual."""
    tag = f"{spec.dataset}/{spec.condition}"
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    cdir.mkdir(parents=True, exist_ok=True)
    expected_folds = len(fold_ids)

    records = load_unit_records(run_root, spec, configs, fold_ids)
    search = search_results_frame(records, spec, model_name, negative_label_name, hyperparameter_keys)
    summary = config_summary_frame(search, expected_folds, hyperparameter_keys)
    _write_csv(search, cdir / "cv_search_results.csv")

    all_units_ok = bool((search["status"] == STATUS_UNIT_COMPLETED).all())
    try:
        if reused_config_index is None:
            best, selection = select_best_configuration(summary)
        else:
            best, selection = select_reused_configuration(summary, reused_config_index, spec.hyperparameters_from)
    except SelectionError as exc:
        _write_csv(summary.assign(selected=False), cdir / "cv_config_summary.csv")
        logger.error(f"{tag}: {exc}")
        return _finish_condition(
            cdir, spec, cfg_used, model_name, "FAILED", provenance, {"error": str(exc)}, None,
        )

    best_index = int(best["config_index"])
    _write_csv(summary.assign(selected=summary["config_index"] == best_index), cdir / "cv_config_summary.csv")

    winner_records = [r for r in records if r["config_index"] == best_index and r["status"] == STATUS_UNIT_COMPLETED]
    fold_metrics = fold_metrics_frame(winner_records, hyperparameter_keys)
    segments = winner_segment_predictions(run_root, spec, best_index, fold_ids)
    recordings, patients = aggregate_winner_predictions(segments)

    y_true = patients["target_label"].to_numpy()
    y_score = patients["score"].to_numpy()
    pooled = ev.compute_patient_metrics(y_true, y_score, negative_label_name, threshold)
    neg_key = negative_label_name.strip().lower()
    pooled["recall_negative"] = pooled[f"recall_{neg_key}"]
    pooled["min_class_recall"] = float(min(pooled["recall_copd"], pooled["recall_negative"]))
    metrics_summary = metrics_summary_frame(fold_metrics, pooled)
    roc, pr = curve_frames(y_true, y_score)
    target_names = (negative_label_name, "COPD")

    _write_csv(fold_metrics, cdir / "cv_fold_metrics.csv")
    _write_csv(metrics_summary.assign(model=model_name, dataset=spec.dataset, condition=spec.condition),
               cdir / "cv_metrics_summary.csv")
    _write_csv(patients, cdir / "cv_patient_predictions.csv")
    _write_csv(recordings, cdir / "cv_recording_predictions.csv")
    _write_csv(segments, cdir / "cv_segment_predictions.csv")
    ev.confusion_matrix_df(y_true, y_score, target_names, threshold).reset_index().rename(
        columns={"index": "real"}).to_csv(cdir / "cv_confusion_matrix.csv", index=False, lineterminator="\n")
    _write_csv(ev.classification_report_df(y_true, y_score, target_names, threshold), cdir / "cv_classification_report.csv")
    _write_csv(roc, cdir / "cv_roc_curve.csv")
    _write_csv(pr, cdir / "cv_pr_curve.csv")
    write_group_artifacts(cdir, segments, patients, negative_label_name, threshold)
    write_figures(cdir, cfg_used, y_true, y_score, target_names, threshold, fold_metrics, negative_label_name, logger, tag)

    best_epochs = {
        str(int(r["fold"])): r.get("best_epoch") for r in sorted(winner_records, key=lambda r: r["fold"])
    }
    epoch_median = median_best_epoch(best_epochs.values())
    best_config = next(c for c in configs if int(c["config_index"]) == best_index)
    _write_json({
        "protocol": PROTOCOL_NAME,
        "model": model_name, "dataset": spec.dataset, "condition": spec.condition,
        "branch": spec.branch, "augment": bool(spec.augment),
        "config_index": best_index,
        "hyperparameters": {k: best_config[k] for k in hyperparameter_keys},
        "hyperparameters_source": hyperparameter_source,
        "expected_folds": expected_folds, "fold_ids": list(fold_ids),
        "metrics": {
            "fold_mean": {m: metrics_summary.loc[0, m] for m in SUMMARY_METRICS},
            "fold_std": {m: metrics_summary.loc[1, m] for m in SUMMARY_METRICS},
            "pooled": {m: pooled[m] for m in SUMMARY_METRICS},
        },
        "best_epochs_by_fold": best_epochs if epoch_median is not None else None,
        "median_best_epoch": epoch_median,
        "selection": selection,
        "hashes": provenance.get("hashes", {}),
    }, cdir / "best_hyperparameters.json")

    status = "COMPLETED" if all_units_ok else "PARTIAL"
    # Fila de cv_run_summary.csv: TODAS las metricas de SUMMARY_METRICS con su
    # media y desviacion entre folds y su valor agrupado (un paciente, una prediccion).
    summary_row = {
        "model": model_name, "dataset": spec.dataset, "condition": spec.condition, "status": status,
        "best_config_index": best_index,
        "hyperparameters": json.dumps(art._json_safe({k: best_config[k] for k in hyperparameter_keys}), sort_keys=True),
        "median_best_epoch": epoch_median,
        "n_folds": expected_folds,
        **{f"fold_mean_{m}": metrics_summary.loc[0, m] for m in SUMMARY_METRICS},
        **{f"fold_std_{m}": metrics_summary.loc[1, m] for m in SUMMARY_METRICS},
        **{f"pooled_{m}": pooled[m] for m in SUMMARY_METRICS},
        "n_units_completed": int((search["status"] == STATUS_UNIT_COMPLETED).sum()),
        "n_units_expected": int(len(search)),
    }
    logger.info(
        f"{tag}: configuracion global {best_index} (decidida por {selection['decided_by']}), "
        f"BA folds={summary_row['fold_mean_balanced_accuracy']:.3f}±{summary_row['fold_std_balanced_accuracy']:.3f} "
        f"agrupada={summary_row['pooled_balanced_accuracy']:.3f} [{status}]"
    )
    return _finish_condition(cdir, spec, cfg_used, model_name, status, provenance, {"best_config_index": best_index}, summary_row)


def _finish_condition(
    cdir: Path, spec: dmod.ConditionSpec, cfg_used: dict, model_name: str, status: str,
    provenance: dict, status_extra: dict, summary_row: dict | None,
) -> dict:
    """``resolved_config.toml``, ``run_manifest.json`` y ``status.json`` de la
    condicion, y el dict que consume el lanzador."""
    resolved = dict(cfg_used)
    resolved["condition_run"] = {
        "model": model_name, "dataset": spec.dataset, "condition": spec.condition, "branch": spec.branch,
        "augment": bool(spec.augment), "hyperparameters_from": spec.hyperparameters_from or "",
    }
    (cdir / "resolved_config.toml").write_text(art.toml_dump(resolved), encoding="utf-8")
    _write_json({
        "protocol": PROTOCOL_NAME, "model": model_name, "dataset": spec.dataset, "condition": spec.condition,
        "branch": spec.branch, "augment": bool(spec.augment), "status": status,
        "outer_test": {"enabled": False, "accessed": False},
        "final_model": {"enabled": False},
        "git": art.git_state(), "written_at_utc": art._now_iso(),
        **provenance,
    }, cdir / "run_manifest.json")
    art.write_status(cdir, status, status_extra)
    return {"spec": spec, "status": status, "ok": status == "COMPLETED", "summary_row": summary_row}


def write_run_tables(run_root: Path, results: list[dict]) -> None:
    rows = [r["summary_row"] for r in results if r.get("summary_row")]
    if rows:
        _write_csv(pd.DataFrame(rows), Path(run_root) / "cv_run_summary.csv")


def final_run_status(results: list[dict]) -> str:
    if results and all(r["status"] == "COMPLETED" for r in results):
        return art.STATUS_COMPLETED
    if any(r["status"] in ("COMPLETED", "PARTIAL") for r in results):
        return art.STATUS_PARTIAL
    return art.STATUS_FAILED


def assert_no_final_model(run_root: Path, results: list[dict]) -> None:
    """Verificacion defensiva: en esta etapa no existe ningun modelo definitivo
    (``final_model.enabled = false``). Si apareciera un directorio ``final/``
    en alguna condicion, es un error duro, no una advertencia."""
    for result in results:
        spec = result["spec"]
        final_dir = art.condition_dir(run_root, spec.dataset, spec.condition) / "final"
        if final_dir.exists():
            raise RuntimeError(
                f"{spec.dataset}/{spec.condition}: existe {final_dir}, pero el protocolo {PROTOCOL_NAME} "
                "no debe generar ningun modelo definitivo"
            )


def condition_provenance(
    cfg: dict, fingerprint: dict, spec: dmod.ConditionSpec, run_id: str, smoke_test: bool, resumed: bool,
    split: sp.HoldoutSplit, fold_ids: list[int],
) -> dict:
    """Trazabilidad que ``run_manifest.json`` y ``best_hyperparameters.json``
    guardan por condicion: hashes de la configuracion, del split y de los
    datos de entrada de ESA condicion."""
    hashes = fingerprint["input_hashes"]
    data_hashes = {
        key: value for key, value in hashes.items()
        if key.startswith(f"{spec.dataset}/")
        and (key.endswith("/segments.csv") or key.endswith(f"/segments_{spec.branch}.npy"))
    }
    return {
        "run_id": run_id, "smoke_test": bool(smoke_test), "resumed": bool(resumed),
        "fold_ids": [int(k) for k in fold_ids],
        "split": {
            "n_splits": int(split.n_splits),
            "n_development_patients": int(len(split.development_patients())),
            "seed": cfg.get("holdout", {}).get("random_state"),
        },
        "hashes": {
            "config_sha256": fingerprint["config_fingerprint"],
            "code_sha256": fingerprint.get("code_sha256"),
            "split_csv_sha256": hashes["holdout_splits.csv"],
            "split_manifest_sha256": hashes["holdout_splits_manifest.json"],
            "data": data_hashes,
        },
    }


# ---------------------------------------------------------------------------
# Huella (--resume) y --dry-run
# ---------------------------------------------------------------------------

SVM_FINGERPRINT_SECTIONS = (
    "protocol", "acoustic", "logmel", "mfcc", "summary", "weights", "holdout", "outer_test",
    "final_model", "svm", "selection", "metrics", "seeds", "determinism", "datasets", "experiments",
)
NN_FINGERPRINT_SECTIONS = (
    "protocol", "acoustic", "logmel", "weights", "normalization", "training", "search", "selection",
    "metrics", "evaluation", "augmentation", "seeds", "determinism", "holdout", "outer_test",
    "final_model", "datasets", "experiments",
)


# Codigo que define lo que hace una unidad de entrenamiento: todo modeling/**/*.py
# salvo las pruebas y los lanzadores secuenciales (que solo orquestan
# subprocesos y no cambian lo que se entrena). Un --resume no debe reutilizar
# unidades entrenadas con otra implementacion.
CODE_HASH_EXCLUDED_DIRS = ("tests", "__pycache__")
CODE_HASH_EXCLUDED_FILES = ("run_holdout_sequence.py", "run_folded_sequence.py", "run_combined_sequence.py")


def code_fingerprint(root: Path | None = None) -> dict:
    """Huella reproducible del codigo de ``modeling/``.

    ``code_files`` guarda el sha256 de cada archivo relevante (ruta relativa a
    ``root``, en formato posix) y ``code_sha256`` es el sha256 de la lista
    ordenada ``ruta:hash``; asi la huella solo depende del contenido y de los
    nombres, no del orden del sistema de archivos ni del momento. Los saltos de
    linea se normalizan a ``\\n`` para que un mismo commit de lo mismo en
    Windows (CRLF) y en Linux (LF). Guardar solo si git esta "dirty" no basta:
    esto identifica exactamente que codigo produjo cada unidad.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if any(part in CODE_HASH_EXCLUDED_DIRS for part in relative.parts[:-1]):
            continue
        if relative.name in CODE_HASH_EXCLUDED_FILES:
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    digest = hashlib.sha256("\n".join(f"{name}:{h}" for name, h in sorted(files.items())).encode("utf-8")).hexdigest()
    return {"code_sha256": digest, "code_files": files}


def holdout_input_hashes(
    data_root: Path, specs: list[dmod.ConditionSpec], splits: dict[str, sp.HoldoutSplit],
    split_csv: Path, split_manifest: Path, smoke_test: bool = False,
) -> dict[str, str]:
    """sha256 del split y, por cada (dataset, fold interno), de su
    ``segments.csv`` y de la rama ``.npy`` de la condicion. ``smoke_test``
    restringe a ``fold_00``, el unico que tocara."""
    hashes = {
        "holdout_splits.csv": dmod.sha256_file(split_csv),
        "holdout_splits_manifest.json": dmod.sha256_file(split_manifest),
    }
    for spec in specs:
        fold_ids = [0] if smoke_test else range(splits[spec.dataset].n_splits)
        for fold_id in fold_ids:
            root = dmod.dataset_root(data_root, spec.dataset, cv_fold_ref(fold_id))
            prefix = f"{spec.dataset}/{cv_fold_ref(fold_id)}"
            hashes[f"{prefix}/segments.csv"] = dmod.sha256_file(root / "segments.csv")
            hashes[f"{prefix}/segments_{spec.branch}.npy"] = dmod.sha256_file(root / f"segments_{spec.branch}.npy")
    return hashes


def build_holdout_fingerprint(
    cfg: dict,
    data_root: Path,
    specs: list[dmod.ConditionSpec],
    dataset_arg: str,
    experiment_arg: str,
    splits: dict[str, sp.HoldoutSplit],
    sections: tuple[str, ...],
    extra: dict | None = None,
    smoke_test: bool = False,
) -> dict:
    split_csv, split_manifest = holdout_split_paths(cfg)
    fingerprint = {
        "dataset_arg": dataset_arg,
        "experiment_arg": experiment_arg,
        "config_fingerprint": rexp._config_run_fingerprint(cfg, sections),
        "input_hashes": holdout_input_hashes(data_root, specs, splits, split_csv, split_manifest, smoke_test),
        # Version del codigo: run_experiment.verify_run_fingerprint rechaza --resume si cambio.
        **code_fingerprint(),
    }
    if extra is not None:
        fingerprint["extra"] = extra
    return fingerprint


def dry_run_check_holdout(
    data_root: Path, cfg: dict, specs: list[dmod.ConditionSpec], split_csv: Path, split_manifest: Path,
) -> dict:
    """Valida el split y, por dataset, sus 5 carpetas ``cv/fold_00..04``: hashes
    de los datos, del split (el preprocesamiento debe haberse hecho contra el
    ``holdout_splits.csv`` ACTUAL), formas, roles y ausencia de pacientes de la
    prueba externa. No entrena ni escribe nada."""
    rows: list[dict] = []
    ok = True

    def add(dataset: str, check: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and bool(passed)
        rows.append({"dataset": dataset, "check": check, "ok": bool(passed), "detail": detail})

    try:
        current_csv_hash = dmod.sha256_file(Path(split_csv))
        current_manifest_hash = dmod.sha256_file(Path(split_manifest))
    except FileNotFoundError as exc:
        add("-", "split", False, str(exc))
        return {"ok": ok, "checks": pd.DataFrame(rows)}

    for dataset in sorted({s.dataset for s in specs}):
        try:
            split = sp.load_holdout_split(split_csv, split_manifest, dataset)
        except Exception as exc:  # noqa: BLE001 - se reporta, nunca se oculta
            add(dataset, "holdout_split", False, str(exc))
            continue
        development = split.development_table()
        add(dataset, "holdout_split", True,
            f"{len(development)} pacientes de desarrollo, {split.n_splits} folds internos, "
            f"{len(split.blocked_test_patients())} de prueba externa bloqueados")

        for fold_id in range(split.n_splits):
            tag = cv_fold_ref(fold_id)
            fold_ref = tag
            try:
                segments = dmod.load_task_segments(data_root, dataset, fold_ref)
                manifest = dmod.load_task_manifest(data_root, dataset, fold_ref)
            except (FileNotFoundError, ValueError) as exc:
                add(dataset, f"carga_segments[{tag}]", False, str(exc))
                continue

            for key, expected in (
                ("protocol", sp.HOLDOUT_PROTOCOL_NAME), ("stage", CV_SUBDIR),
                ("dataset_scope", dataset), ("fold_id", fold_id),
            ):
                add(dataset, f"manifest_{key}[{tag}]", manifest.get(key) == expected,
                    f"{manifest.get(key)!r} (esperado {expected!r})")
            for key, current in (
                ("holdout_splits_csv_sha256", current_csv_hash),
                ("holdout_splits_manifest_sha256", current_manifest_hash),
            ):
                linked = manifest.get(key) is not None and manifest.get(key) == current
                add(dataset, f"{key}[{tag}]", linked,
                    "coincide con el split actual" if linked else
                    f"{tag}/manifest.json quedo desincronizado del split actual; regenere este fold con "
                    "preprocessing/fold_denoising.py --protocol holdout-v3")

            try:
                verify_fold_segments_against_split(segments, split, fold_id)
                add(dataset, f"pacientes_vs_split[{tag}]", True, "train/validation coinciden con el split; sin prueba externa")
            except Exception as exc:  # noqa: BLE001
                add(dataset, f"pacientes_vs_split[{tag}]", False, str(exc))

            root = dmod.dataset_root(data_root, dataset, fold_ref)
            manifest_hashes = manifest.get("output_hashes", {})
            csv_hash = dmod.sha256_file(root / "segments.csv")
            add(dataset, f"sha256_segments_csv[{tag}]", manifest_hashes.get("segments.csv") == csv_hash,
                "coincide con manifest.json" if manifest_hashes.get("segments.csv") == csv_hash
                else "no coincide con manifest.json")

            for branch in dmod.BRANCHES:
                try:
                    array = dmod.load_branch_array(data_root, dataset, branch, fold_ref)
                except FileNotFoundError as exc:
                    add(dataset, f"npy_{branch}[{tag}]", False, str(exc))
                    continue
                shape_ok = array.shape == (len(segments), int(cfg["acoustic"]["segment_length"]))
                add(dataset, f"forma_{branch}[{tag}]", shape_ok,
                    f"{array.shape} vs ({len(segments)}, {cfg['acoustic']['segment_length']})")
                npy_hash = dmod.sha256_file(root / f"segments_{branch}.npy")
                expected_hash = manifest_hashes.get(f"segments_{branch}.npy")
                add(dataset, f"sha256_{branch}[{tag}]", expected_hash == npy_hash,
                    "coincide con manifest.json" if expected_hash == npy_hash else "no coincide con manifest.json")

    return {"ok": ok, "checks": pd.DataFrame(rows)}
