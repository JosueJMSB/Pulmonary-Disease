"""Configuracion, trazabilidad y contratos del diagnostico de ICBHI."""

from __future__ import annotations

import json
import os
import tomllib
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ... import artifacts as art
from ... import data as dmod

from . import PROTOCOL_NAME

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
DEFAULT_PROTOCOL_PATH = PACKAGE_DIR / "protocol.toml"
DATASET = "ICBHI"
EXPECTED_PIPELINE_IDS = ("cnn_no_dn", "cnn_dn", "crnn_dn", "crnn_no_dn_aug")
HYPERPARAMETER_KEYS = ("lr", "dropout", "weight_decay", "batch_size")
NN_CONFIG_FINGERPRINT_SECTIONS = (
    "protocol", "acoustic", "logmel", "weights", "normalization", "training",
    "search", "selection", "metrics", "evaluation", "augmentation", "seeds",
    "determinism", "holdout", "outer_test", "final_model", "datasets",
    "experiments",
)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def resolve_runtime_path(value: str | Path | None, env_name: str, default: str) -> Path:
    if value is not None:
        return Path(value).resolve()
    env = os.environ.get(env_name)
    if env:
        return Path(env).resolve()
    return (REPO_ROOT / default).resolve()


def seed_tag(seed: int) -> str:
    return f"seed_{int(seed)}"


@dataclass(frozen=True)
class Pipeline:
    id: str
    architecture: str
    condition: str
    branch: str
    augment: bool
    source_run_id: str
    source_config_index: int
    source_median_best_epoch: int
    source_config_sha256: str
    lr: float
    dropout: float
    weight_decay: float
    batch_size: int

    @classmethod
    def from_dict(cls, row: dict) -> "Pipeline":
        return cls(
            id=str(row["id"]),
            architecture=str(row["architecture"]),
            condition=str(row["condition"]),
            branch=str(row["branch"]),
            augment=bool(row["augment"]),
            source_run_id=str(row["source_run_id"]),
            source_config_index=int(row["source_config_index"]),
            source_median_best_epoch=int(row["source_median_best_epoch"]),
            source_config_sha256=str(row["source_config_sha256"]),
            lr=float(row["lr"]),
            dropout=float(row["dropout"]),
            weight_decay=float(row["weight_decay"]),
            batch_size=int(row["batch_size"]),
        )

    def candidate(self) -> dict:
        return {
            "config_index": self.source_config_index,
            "lr": self.lr,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
        }

    def condition_spec(self, condition: str | None = None) -> dmod.ConditionSpec:
        return dmod.ConditionSpec(
            dataset=DATASET,
            condition=condition or self.condition,
            branch=self.branch,
            dn_reliable_only=False,
            augment=self.augment,
            hyperparameters_from=None,
        )

    def source_best_path(self, source_runs_root: Path) -> Path:
        return (
            Path(source_runs_root)
            / self.architecture
            / self.source_run_id
            / "datasets"
            / DATASET
            / self.condition
            / "best_hyperparameters.json"
        )


@dataclass(frozen=True)
class Protocol:
    path: Path
    raw: dict
    seeds: tuple[int, ...]
    pipelines: tuple[Pipeline, ...]

    @property
    def n_splits(self) -> int:
        return int(self.raw["n_splits"])

    @property
    def original_split_paths(self) -> tuple[Path, Path]:
        paths = self.raw["paths"]
        return (
            resolve_repo_path(paths["original_split_csv"]),
            resolve_repo_path(paths["original_split_manifest"]),
        )

    def architecture_config_path(self, architecture: str) -> Path:
        if architecture not in ("cnn", "crnn"):
            raise ValueError(f"arquitectura no soportada: {architecture!r}")
        return resolve_repo_path(self.raw["paths"][f"{architecture}_config"])

    def pipeline(self, pipeline_id: str) -> Pipeline:
        for pipeline in self.pipelines:
            if pipeline.id == pipeline_id:
                return pipeline
        raise KeyError(f"pipeline desconocido: {pipeline_id}")


def load_protocol(path: Path | str | None = None) -> Protocol:
    protocol_path = Path(path or DEFAULT_PROTOCOL_PATH).resolve()
    raw = tomllib.loads(protocol_path.read_text(encoding="utf-8"))
    if raw.get("protocol") != PROTOCOL_NAME:
        raise ValueError(
            f"{protocol_path}: protocol={raw.get('protocol')!r}; se esperaba {PROTOCOL_NAME!r}"
        )
    if raw.get("dataset") != DATASET:
        raise ValueError(f"este diagnostico solo admite {DATASET}")
    seeds = tuple(int(value) for value in raw["repetition_seeds"])
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("repetition_seeds debe contener al menos dos semillas unicas")
    pipelines = tuple(Pipeline.from_dict(row) for row in raw["pipelines"])
    ids = tuple(pipeline.id for pipeline in pipelines)
    if ids != EXPECTED_PIPELINE_IDS:
        raise ValueError(f"pipelines debe ser exactamente {EXPECTED_PIPELINE_IDS}; se encontro {ids}")
    if any(pipeline.architecture not in ("cnn", "crnn") for pipeline in pipelines):
        raise ValueError("solo se admiten CNN y CRNN en esta prueba")
    if int(raw["n_splits"]) != 5:
        raise ValueError("el diagnostico fue preespecificado para 5 folds")
    return Protocol(path=protocol_path, raw=raw, seeds=seeds, pipelines=pipelines)


def validate_source_selection(pipeline: Pipeline, source_runs_root: Path) -> dict:
    """Exige que la configuracion congelada coincida con el artefacto v3."""
    path = pipeline.source_best_path(source_runs_root)
    if not path.is_file():
        raise FileNotFoundError(f"falta el artefacto de seleccion: {path}")
    best = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": "holdout_cv_v3",
        "model": pipeline.architecture,
        "dataset": DATASET,
        "condition": pipeline.condition,
        "branch": pipeline.branch,
        "augment": pipeline.augment,
        "config_index": pipeline.source_config_index,
        "median_best_epoch": pipeline.source_median_best_epoch,
    }
    mismatches = {
        key: (best.get(key), wanted)
        for key, wanted in expected.items()
        if best.get(key) != wanted
    }
    stored_hp = best.get("hyperparameters", {})
    for key in HYPERPARAMETER_KEYS:
        wanted = getattr(pipeline, key)
        got = stored_hp.get(key)
        if got is None or float(got) != float(wanted):
            mismatches[f"hyperparameters.{key}"] = (got, wanted)
    got_hash = best.get("hashes", {}).get("config_sha256")
    if got_hash != pipeline.source_config_sha256:
        mismatches["hashes.config_sha256"] = (got_hash, pipeline.source_config_sha256)
    if mismatches:
        raise RuntimeError(
            f"{pipeline.id}: el artefacto {path} no coincide con el protocolo: {mismatches}"
        )
    return {
        "path": str(path),
        "sha256": art.sha256_file(path),
        "source_run_id": pipeline.source_run_id,
        "source_config_sha256": got_hash,
    }


def load_architecture_config(protocol: Protocol, pipeline: Pipeline) -> dict:
    config_path = protocol.architecture_config_path(pipeline.architecture)
    cfg = dmod.load_config(config_path)
    if dmod.model_architecture(cfg) != pipeline.architecture:
        raise RuntimeError(f"{pipeline.id}: el TOML no describe {pipeline.architecture}")
    if cfg.get("outer_test", {}).get("enabled") is not False:
        raise RuntimeError(f"{pipeline.id}: outer_test debe permanecer false")
    if cfg.get("final_model", {}).get("enabled") is not False:
        raise RuntimeError(f"{pipeline.id}: final_model debe permanecer false")
    architecture_sections = (
        ("cnn",) if pipeline.architecture == "cnn" else ("model", "crnn")
    )
    fingerprint_sections = (
        NN_CONFIG_FINGERPRINT_SECTIONS + architecture_sections
    )
    relevant = {
        key: cfg[key] for key in fingerprint_sections if key in cfg
    }
    payload = json.dumps(
        relevant, sort_keys=True, separators=(",", ":"), default=str
    )
    current_fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if current_fingerprint != pipeline.source_config_sha256:
        raise RuntimeError(
            f"{pipeline.id}: la configuracion actual {config_path} tiene huella "
            f"{current_fingerprint}, pero el cofinalista fue seleccionado con "
            f"{pipeline.source_config_sha256}; no se permite cambiar el protocolo "
            "durante la prueba de robustez"
        )
    return cfg


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(art._json_safe(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)

