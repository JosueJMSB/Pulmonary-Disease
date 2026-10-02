"""Carga y validacion de ``modeling/configs/final_test/selected_pipelines.toml``.

Este modulo es el UNICO punto de verdad sobre que pipeline esta congelado para
cada dataset y si tiene permiso para abrir la prueba externa. Ni
``preprocessing/fold_denoising.py --stage final`` ni
``modeling/final_training/run.py`` deciden nada de esto por su cuenta: ambos
llaman a ``validate_pipeline_for_execution`` (o, en el caso del
preprocesamiento, a la lectura minima equivalente que mantiene ese script
autosuficiente) antes de tocar cualquier dato.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .. import data as dmod

PROTOCOL_NAME = "holdout_final_v1"
SOURCE_PROTOCOL_NAME = dmod.HOLDOUT_PROTOCOL  # "holdout_cv_v3": protocolo que debe declarar el artefacto de procedencia

STATUS_APPROVED = "approved"
STATUS_PENDING_DEVICE_REVIEW = "pending_device_review"
STATUS_BLOCKED = "blocked"
VALID_STATUSES = (STATUS_APPROVED, STATUS_PENDING_DEVICE_REVIEW, STATUS_BLOCKED)
KNOWN_ARCHITECTURES = ("cnn", "crnn")
KNOWN_CONDITIONS = ("no_dn", "dn", "no_dn_aug")

DEVICE_REVIEW_REQUIRED_DATASETS = ("ICBHI",)

DEFAULT_SELECTION_CONFIG = dmod.REPO_ROOT / "modeling" / "configs" / "final_test" / "selected_pipelines.toml"
HYPERPARAMETER_KEYS = ("lr", "dropout", "weight_decay", "batch_size")


class SelectionError(RuntimeError):
    """La seleccion congelada esta mal formada, o un pipeline no tiene permiso
    para abrir la prueba externa (estado distinto de 'approved', split
    cambiado, hiperparametros desincronizados de su procedencia, o revision
    por dispositivo faltante)."""


@dataclass(frozen=True)
class SelectedPipeline:
    dataset: str
    architecture: str
    condition: str
    branch: str
    reference_config: Path
    hyperparameters: dict
    epochs: int
    status: str
    source_run_id: str
    source_config_index: int
    source_artifact: str
    source_artifact_sha256: str
    robustness_run_id: str
    device_review_path: str
    device_review_sha256: str

    @property
    def approved(self) -> bool:
        return self.status == STATUS_APPROVED


@dataclass(frozen=True)
class SelectionConfig:
    path: Path
    threshold: float
    seed_base: int
    split_csv: Path
    split_manifest: Path
    split_csv_sha256: str
    split_manifest_sha256: str
    bootstrap_n_resamples: int
    bootstrap_confidence: float
    bootstrap_random_state: int
    pipelines: dict[str, SelectedPipeline]

    def pipeline_for(self, dataset: str) -> SelectedPipeline:
        try:
            return self.pipelines[dataset]
        except KeyError:
            raise SelectionError(
                f"{self.path}: no hay [pipelines.{dataset}] en la seleccion congelada"
            ) from None


def _require(table: dict, key: str, where: str):
    if key not in table:
        raise SelectionError(f"{where}: falta la clave {key!r}")
    return table[key]


def load_selection_config(path: Path | None = None) -> SelectionConfig:
    """Lee y valida la FORMA de ``selected_pipelines.toml`` (no el acceso al
    test: eso lo decide ``validate_pipeline_for_execution`` por pipeline)."""
    path = Path(path) if path is not None else DEFAULT_SELECTION_CONFIG
    if not path.is_file():
        raise SelectionError(f"no existe la seleccion congelada: {path}")
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)

    if raw.get("protocol") != PROTOCOL_NAME:
        raise SelectionError(f"{path}: protocol={raw.get('protocol')!r}, se esperaba {PROTOCOL_NAME!r}")

    selection = raw.get("selection", {})
    split = raw.get("split", {})
    bootstrap = raw.get("bootstrap", {})
    pipelines_raw = raw.get("pipelines", {})
    if not pipelines_raw:
        raise SelectionError(f"{path}: no hay ninguna seccion [pipelines.<dataset>]")

    threshold = float(_require(selection, "threshold", f"{path}: [selection]"))
    if not 0.0 < threshold < 1.0:
        raise SelectionError(f"{path}: [selection] threshold={threshold} fuera de (0, 1)")
    bootstrap_n_resamples = int(_require(bootstrap, "n_resamples", f"{path}: [bootstrap]"))
    if bootstrap_n_resamples <= 0:
        raise SelectionError(f"{path}: [bootstrap] n_resamples={bootstrap_n_resamples} debe ser positivo")
    bootstrap_confidence = float(_require(bootstrap, "confidence", f"{path}: [bootstrap]"))
    if not 0.0 < bootstrap_confidence < 1.0:
        raise SelectionError(f"{path}: [bootstrap] confidence={bootstrap_confidence} fuera de (0, 1)")

    pipelines: dict[str, SelectedPipeline] = {}
    for dataset, entry in pipelines_raw.items():
        where = f"{path}: [pipelines.{dataset}]"
        status = str(_require(entry, "status", where))
        if status not in VALID_STATUSES:
            raise SelectionError(f"{where}: status={status!r} desconocido (use {VALID_STATUSES})")
        architecture = str(_require(entry, "architecture", where))
        if architecture not in KNOWN_ARCHITECTURES:
            raise SelectionError(f"{where}: architecture={architecture!r} desconocida (use {KNOWN_ARCHITECTURES})")
        condition = str(_require(entry, "condition", where))
        if condition not in KNOWN_CONDITIONS:
            raise SelectionError(f"{where}: condition={condition!r} desconocida (use {KNOWN_CONDITIONS})")
        epochs = int(_require(entry, "epochs", where))
        if epochs <= 0:
            raise SelectionError(f"{where}: epochs={epochs} debe ser positivo")
        hp_raw = _require(entry, "hyperparameters", where)
        missing_hp = set(HYPERPARAMETER_KEYS) - set(hp_raw)
        if missing_hp:
            raise SelectionError(f"{where}.hyperparameters: faltan claves {sorted(missing_hp)}")
        batch_size = int(hp_raw["batch_size"])
        if batch_size <= 0:
            raise SelectionError(f"{where}.hyperparameters: batch_size={batch_size} debe ser positivo")
        pipelines[dataset] = SelectedPipeline(
            dataset=dataset,
            architecture=architecture,
            condition=condition,
            branch=str(_require(entry, "branch", where)),
            reference_config=dmod.REPO_ROOT / str(_require(entry, "reference_config", where)),
            hyperparameters={k: hp_raw[k] for k in HYPERPARAMETER_KEYS},
            epochs=epochs,
            status=status,
            source_run_id=str(_require(entry, "source_run_id", where)),
            source_config_index=int(_require(entry, "source_config_index", where)),
            source_artifact=str(_require(entry, "source_artifact", where)),
            source_artifact_sha256=str(_require(entry, "source_artifact_sha256", where)),
            robustness_run_id=str(entry.get("robustness_run_id", "")),
            device_review_path=str(entry.get("device_review_path", "")),
            device_review_sha256=str(entry.get("device_review_sha256", "")),
        )

    return SelectionConfig(
        path=path,
        threshold=threshold,
        seed_base=int(_require(selection, "seed_base", f"{path}: [selection]")),
        split_csv=dmod.REPO_ROOT / str(_require(split, "csv", f"{path}: [split]")),
        split_manifest=dmod.REPO_ROOT / str(_require(split, "manifest", f"{path}: [split]")),
        split_csv_sha256=str(_require(split, "csv_sha256", f"{path}: [split]")),
        split_manifest_sha256=str(_require(split, "manifest_sha256", f"{path}: [split]")),
        bootstrap_n_resamples=bootstrap_n_resamples,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_random_state=int(_require(bootstrap, "random_state", f"{path}: [bootstrap]")),
        pipelines=pipelines,
    )


# ---------------------------------------------------------------------------
# Validaciones de acceso al test (PLAN-ENTRENAMIENTO-FINAL.md, seccion 1)
# ---------------------------------------------------------------------------

def validate_pipeline_approved(pipeline: SelectedPipeline) -> None:
    if not pipeline.approved:
        raise SelectionError(
            f"{pipeline.dataset}: status={pipeline.status!r}, no 'approved'; esta etapa ya constituye "
            "acceso a la prueba externa y permanece bloqueada hasta que el pipeline quede aprobado "
            f"en {DEFAULT_SELECTION_CONFIG}"
        )


def validate_split_unchanged(selection_cfg: SelectionConfig) -> None:
    """El split 80/20 oficial no cambio desde que se congelo esta seleccion."""
    for path, expected, label in (
        (selection_cfg.split_csv, selection_cfg.split_csv_sha256, "holdout_splits.csv"),
        (selection_cfg.split_manifest, selection_cfg.split_manifest_sha256, "holdout_splits_manifest.json"),
    ):
        if not Path(path).is_file():
            raise SelectionError(f"no existe {label}: {path}")
        actual = dmod.sha256_file(Path(path))
        if actual != expected:
            raise SelectionError(
                f"{label}: sha256 {actual[:12]}... no coincide con el congelado en "
                f"{selection_cfg.path} ({expected[:12]}...); el split 80/20 oficial cambio, o esta "
                "seleccion quedo desincronizada de el"
            )


def validate_source_artifact(pipeline: SelectedPipeline, source_runs_root: Path) -> dict:
    """Resuelve EXACTAMENTE ``source_runs_root / pipeline.source_artifact``
    -nunca se reconstruye desde arquitectura/source_run_id/dataset/condicion:
    el campo declarado en la seleccion congelada es la unica fuente de
    verdad-, verifica su sha256 contra ``source_artifact_sha256`` y valida que
    su contenido describe EXACTAMENTE este pipeline antes de confiar en sus
    hiperparametros. Devuelve el JSON ya cargado."""
    artifact_path = Path(source_runs_root) / pipeline.source_artifact
    if not artifact_path.is_file():
        raise SelectionError(
            f"{pipeline.dataset}: no se encuentra el artefacto de procedencia declarado "
            f"({artifact_path}); verifique source_artifact/--source-runs-root"
        )
    actual_hash = dmod.sha256_file(artifact_path)
    if actual_hash != pipeline.source_artifact_sha256:
        raise SelectionError(
            f"{pipeline.dataset}: sha256 {actual_hash[:12]}... de {artifact_path} no coincide con "
            f"source_artifact_sha256 ({pipeline.source_artifact_sha256[:12]}...); el artefacto de procedencia "
            "cambio, o esta seleccion quedo desincronizada de el"
        )

    best = json.loads(artifact_path.read_text(encoding="utf-8"))
    if best.get("protocol") != SOURCE_PROTOCOL_NAME:
        raise SelectionError(
            f"{pipeline.dataset}: {artifact_path} declara protocol={best.get('protocol')!r}, "
            f"se esperaba {SOURCE_PROTOCOL_NAME!r}"
        )
    for key, expected in (
        ("model", pipeline.architecture), ("dataset", pipeline.dataset),
        ("condition", pipeline.condition), ("branch", pipeline.branch),
    ):
        if best.get(key) != expected:
            raise SelectionError(
                f"{pipeline.dataset}: {artifact_path} tiene {key}={best.get(key)!r}, se esperaba {expected!r}"
            )
    if int(best.get("config_index", -1)) != int(pipeline.source_config_index):
        raise SelectionError(
            f"{pipeline.dataset}: source_config_index={pipeline.source_config_index} no coincide con "
            f"config_index={best.get('config_index')} de {artifact_path}"
        )
    stored_hp = best.get("hyperparameters", {})
    for key in HYPERPARAMETER_KEYS:
        if float(stored_hp.get(key, float("nan"))) != float(pipeline.hyperparameters[key]):
            raise SelectionError(
                f"{pipeline.dataset}: hyperparameters.{key}={pipeline.hyperparameters[key]} no coincide con "
                f"{stored_hp.get(key)} de {artifact_path}"
            )
    median_epoch = best.get("median_best_epoch")
    if median_epoch is not None and int(median_epoch) != int(pipeline.epochs):
        raise SelectionError(
            f"{pipeline.dataset}: epochs={pipeline.epochs} no coincide con median_best_epoch="
            f"{median_epoch} de {artifact_path}"
        )
    return best


def validate_device_review(pipeline: SelectedPipeline) -> None:
    """Para ICBHI, 'approved' exige haber registrado la ruta y el sha256 del
    analisis por el subgrupo Meditron (el unico dispositivo de ICBHI con
    ambas clases). Otros datasets no tienen esta exigencia adicional."""
    if pipeline.dataset not in DEVICE_REVIEW_REQUIRED_DATASETS or not pipeline.approved:
        return
    if not pipeline.device_review_path or not pipeline.device_review_sha256:
        raise SelectionError(
            f"{pipeline.dataset}: status='approved' exige device_review_path y device_review_sha256 "
            "registrados (el analisis por el subgrupo Meditron)"
        )
    review_path = dmod.REPO_ROOT / pipeline.device_review_path
    if not review_path.is_file():
        raise SelectionError(f"{pipeline.dataset}: no existe device_review_path: {review_path}")
    actual = dmod.sha256_file(review_path)
    if actual != pipeline.device_review_sha256:
        raise SelectionError(
            f"{pipeline.dataset}: sha256 {actual[:12]}... de {review_path} no coincide con "
            f"device_review_sha256 ({pipeline.device_review_sha256[:12]}...)"
        )


def validate_reference_config(pipeline: SelectedPipeline) -> dict:
    """El TOML de arquitectura referenciado existe y declara el protocolo
    holdout_cv_v3 del que este pipeline tomo su arquitectura/acustica."""
    if not pipeline.reference_config.is_file():
        raise SelectionError(f"{pipeline.dataset}: no existe reference_config: {pipeline.reference_config}")
    cfg = dmod.load_config(pipeline.reference_config)
    if not dmod.is_holdout_protocol(cfg):
        raise SelectionError(
            f"{pipeline.dataset}: {pipeline.reference_config} no declara protocol = {dmod.HOLDOUT_PROTOCOL!r}"
        )
    if dmod.model_architecture(cfg) != pipeline.architecture:
        raise SelectionError(
            f"{pipeline.dataset}: {pipeline.reference_config} describe la arquitectura "
            f"{dmod.model_architecture(cfg)!r}, se esperaba {pipeline.architecture!r}"
        )
    return cfg


def validate_pipeline_for_execution(
    selection_cfg: SelectionConfig, dataset: str, source_runs_root: Path,
) -> SelectedPipeline:
    """Las comprobaciones completas del plan antes del primer acceso al test:
    pipeline aprobado, split sin cambios, artefacto de procedencia (hash y
    contenido), revision por dispositivo (si aplica) y TOML de referencia
    valido. Lanza ``SelectionError`` con un mensaje claro ante la primera que
    falle; no corrige nada en silencio."""
    pipeline = selection_cfg.pipeline_for(dataset)
    validate_pipeline_approved(pipeline)
    validate_split_unchanged(selection_cfg)
    validate_source_artifact(pipeline, source_runs_root)
    validate_device_review(pipeline)
    validate_reference_config(pipeline)
    return pipeline


# ---------------------------------------------------------------------------
# Huella canonica POR DATASET: deliberadamente NO el hash de
# selected_pipelines.toml completo (aprobar ICBHI no debe invalidar un
# staging de FRAIWAN o COMBINED que nunca cambio). Solo lo que el plan exige:
# pipeline seleccionado, hiperparametros y epocas, umbral y semilla, hashes
# del split, artefacto de procedencia, aprobacion y revision por dispositivo
# de ESE dataset.
# ---------------------------------------------------------------------------

def canonical_pipeline_fingerprint(
    pipeline: SelectedPipeline, selection_cfg: SelectionConfig, reference_config_fingerprint: str,
) -> str:
    payload = {
        "dataset": pipeline.dataset, "architecture": pipeline.architecture, "condition": pipeline.condition,
        "branch": pipeline.branch, "hyperparameters": pipeline.hyperparameters, "epochs": pipeline.epochs,
        "threshold": selection_cfg.threshold, "seed_base": selection_cfg.seed_base,
        "source_artifact": pipeline.source_artifact, "source_artifact_sha256": pipeline.source_artifact_sha256,
        "status": pipeline.status, "device_review_path": pipeline.device_review_path,
        "device_review_sha256": pipeline.device_review_sha256,
        "split_csv_sha256": selection_cfg.split_csv_sha256, "split_manifest_sha256": selection_cfg.split_manifest_sha256,
        "reference_config_fingerprint": reference_config_fingerprint,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Huella del diccionario CRUDO [pipelines.<dataset>] (tal cual lo lee
# tomllib, sin pasar por SelectedPipeline): debe coincidir EXACTAMENTE con
# preprocessing/fold_denoising.py::_final_pipeline_fingerprint, que calcula
# esta misma huella sobre el mismo diccionario -preprocessing/ no importa
# modeling/, asi que ambos lados duplican este calculo puro e identico en vez
# de compartirlo-. Permite al dry-run de entrenamiento detectar si el
# preprocesamiento final se genero contra una seleccion distinta de la
# actual (CORRECIONES.md: "comparar en el dry-run el fingerprint del
# preprocesamiento final con el pipeline seleccionado actualmente").
# ---------------------------------------------------------------------------

def raw_pipeline_section(path: Path, dataset: str) -> dict:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    try:
        return raw["pipelines"][dataset]
    except KeyError:
        raise SelectionError(f"{path}: no hay [pipelines.{dataset}]") from None


def final_pipeline_fingerprint(path: Path, dataset: str) -> str:
    pipeline = raw_pipeline_section(path, dataset)
    payload = json.dumps(pipeline, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
