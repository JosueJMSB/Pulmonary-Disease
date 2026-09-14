"""Gestion de ejecuciones: directorios, staging atomico por fold, figuras,
serializacion de modelos y reanudacion.

Sigue el mismo patron que ``preprocessing/utils.py``: cada fold se escribe
primero en ``fold_NN_staging`` y solo se publica (rename a ``fold_NN`` + un
archivo ``_SUCCESS``) al completarse sin errores. Un ``--resume`` que
encuentra una carpeta de staging abandonada la borra: nunca la cuenta como
resultado valido.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # nunca requiere pantalla: corre en un servidor remoto
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from . import evaluation as ev

REPO_ROOT = Path(__file__).resolve().parent.parent

STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_PARTIAL = "PARTIAL"
STATUS_FAILED = "FAILED"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    """Convierte tipos de numpy/pandas a tipos nativos antes de serializar."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


# ---------------------------------------------------------------------------
# Serializador TOML minimo (solo para volcar la configuracion resuelta).
#
# No es un escritor TOML general: cubre exactamente las formas presentes en
# configs/svm_rbf.toml (tablas anidadas, arrays de tablas de un nivel,
# escalares y listas de escalares), que es todo lo que resolved_config.toml
# necesita representar.
# ---------------------------------------------------------------------------

def _toml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if value is None:
        return '""'
    raise TypeError(f"tipo TOML no soportado: {type(value)!r}")


def _toml_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    return _toml_scalar(value)


def _is_table(value) -> bool:
    return isinstance(value, dict)


def _is_array_of_tables(value) -> bool:
    return isinstance(value, list) and len(value) > 0 and all(isinstance(v, dict) for v in value)


def toml_dump(data: dict) -> str:
    lines: list[str] = []

    def emit(table: dict, path: list[str]) -> None:
        scalars = {k: v for k, v in table.items() if not _is_table(v) and not _is_array_of_tables(v)}
        subtables = {k: v for k, v in table.items() if _is_table(v)}
        array_tables = {k: v for k, v in table.items() if _is_array_of_tables(v)}

        if path:
            lines.append(f"[{'.'.join(path)}]")
        for key, value in scalars.items():
            lines.append(f"{key} = {_toml_value(value)}")
        if scalars:
            lines.append("")
        for key, value in subtables.items():
            emit(value, path + [key])
        for key, value in array_tables.items():
            for entry in value:
                lines.append(f"[[{'.'.join(path + [key])}]]")
                for k, v in entry.items():
                    if not _is_table(v):
                        lines.append(f"{k} = {_toml_value(v)}")
                lines.append("")

    emit(data, [])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Ciclo de vida de una ejecucion
# ---------------------------------------------------------------------------

def new_run_id() -> str:
    now = datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"


def resolve_run_root(runs_root: Path, model: str, run_id: str) -> Path:
    return Path(runs_root) / model / run_id


def find_latest_run(runs_root: Path, model: str) -> str | None:
    model_dir = Path(runs_root) / model
    if not model_dir.is_dir():
        return None
    candidates = sorted(p.name for p in model_dir.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def cleanup_abandoned_staging(run_root: Path) -> list[Path]:
    """Borra `fold_NN_staging` sin `_SUCCESS`: un resume nunca hereda una
    escritura a medias de una ejecucion interrumpida."""
    removed = []
    datasets_dir = run_root / "datasets"
    if not datasets_dir.is_dir():
        return removed
    for staging in sorted(datasets_dir.glob("*/*/fold_*_staging")):
        shutil.rmtree(staging, ignore_errors=True)
        removed.append(staging)
    return removed


def init_run(
    runs_root: Path, model: str, run_id: str | None, resume: bool
) -> tuple[Path, str]:
    """Crea una ejecucion nueva, o localiza una existente para --resume.

    En --resume esta funcion SOLO localiza y valida que ``run_root`` exista:
    no escribe ``status.json`` ni borra staging abandonado. Esos dos efectos
    son irreversibles (perderian el estado de la ejecucion original y
    cualquier evidencia del staging a medias) y deben esperar a que
    ``run_experiment.verify_run_fingerprint`` confirme que la ejecucion a
    reanudar de verdad corresponde a los mismos datos, configuracion y
    alcance (--dataset/--experiment) que la actual invocacion -ver
    ``finalize_resume``-. Para una ejecucion nueva no hay nada que proteger,
    asi que crea los directorios y marca RUNNING de inmediato.
    """
    if resume:
        target_id = run_id
        if target_id in (None, "latest"):
            target_id = find_latest_run(runs_root, model)
            if target_id is None:
                raise FileNotFoundError(f"no hay ejecuciones previas de {model} en {runs_root}")
        run_root = resolve_run_root(runs_root, model, target_id)
        if not run_root.is_dir():
            raise FileNotFoundError(f"no existe la ejecucion a reanudar: {run_root}")
        return run_root, target_id

    target_id = run_id or new_run_id()
    run_root = resolve_run_root(runs_root, model, target_id)
    if run_root.exists():
        raise FileExistsError(
            f"ya existe una ejecucion en {run_root}; use --resume {target_id} para continuar"
        )
    (run_root / "figures").mkdir(parents=True)
    (run_root / "datasets").mkdir(parents=True)
    write_status(run_root, STATUS_RUNNING, {"created_at_utc": _now_iso()})
    return run_root, target_id


def finalize_resume(run_root: Path) -> list[Path]:
    """Efectos de reanudar que solo deben ocurrir DESPUES de verificar la
    huella de la ejecucion (``run_experiment.verify_run_fingerprint``):
    limpia el staging abandonado de una interrupcion previa y vuelve a
    marcar ``status.json`` como RUNNING. Llamarla antes de esa verificacion
    arriesgaria borrar staging o el estado de una ejecucion que en realidad
    no se puede reanudar."""
    removed = cleanup_abandoned_staging(run_root)
    write_status(run_root, STATUS_RUNNING, {"resumed_at_utc": _now_iso()})
    return removed


def write_status(run_root: Path, status: str, extra: dict | None = None) -> None:
    payload = {"status": status, "updated_at_utc": _now_iso()}
    if extra:
        payload.update(_json_safe(extra))
    (run_root / "status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def read_status(run_root: Path) -> dict:
    path = run_root / "status.json"
    if not path.is_file():
        return {"status": None}
    return json.loads(path.read_text(encoding="utf-8"))


def write_environment(run_root: Path) -> None:
    lines = [f"python: {sys.version.split()[0]}", f"platform: {platform.platform()}"]
    for module_name in ("numpy", "pandas", "scipy", "sklearn", "librosa", "joblib", "matplotlib", "soundfile"):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "desconocida")
        except ImportError:
            version = "no instalado"
        lines.append(f"{module_name}: {version}")
    state = git_state()
    lines.append(f"git_commit: {state['commit']}")
    lines.append(f"git_dirty: {state['dirty']}")
    (run_root / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_resolved_config(run_root: Path, cfg: dict, cli_args: dict) -> None:
    resolved = dict(cfg)
    resolved["run"] = {k: v for k, v in _json_safe(cli_args).items() if v is not None}
    (run_root / "resolved_config.toml").write_text(toml_dump(resolved), encoding="utf-8")


def setup_run_logger(run_root: Path, name: str = "modeling") -> logging.Logger:
    logger = logging.getLogger(f"{name}.{run_root.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(run_root / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


# ---------------------------------------------------------------------------
# Directorios de fold: staging -> publicacion atomica con _SUCCESS
# ---------------------------------------------------------------------------

def condition_dir(run_root: Path, dataset: str, condition: str) -> Path:
    return run_root / "datasets" / dataset / condition


def fold_dir(run_root: Path, dataset: str, condition: str, fold_id: int) -> Path:
    return condition_dir(run_root, dataset, condition) / f"fold_{fold_id + 1:02d}"


def fold_staging_dir(run_root: Path, dataset: str, condition: str, fold_id: int) -> Path:
    target = fold_dir(run_root, dataset, condition, fold_id)
    return target.with_name(target.name + "_staging")


def is_fold_complete(run_root: Path, dataset: str, condition: str, fold_id: int) -> bool:
    target = fold_dir(run_root, dataset, condition, fold_id)
    return (target / "_SUCCESS").is_file()


def publish_fold(staging_dir: Path) -> Path:
    target = staging_dir.with_name(staging_dir.name.replace("_staging", ""))
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging_dir, target)
    (target / "_SUCCESS").touch()
    return target


def write_fold_artifacts(staging_dir: Path, fold_result) -> None:
    """Vuelca los artefactos de un fold ya calculado.

    Cada tabla de predicciones lleva una columna ``fold`` explicita: es lo
    que permite despues concatenar las cinco tablas de test en una sola
    ``oof_patient_predictions.csv`` sin perder de que fold vino cada fila.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    fold_result.grid_table.to_csv(staging_dir / "grid_search.csv", index=False, lineterminator="\n")
    fold_result.segment_predictions.assign(fold=fold_result.fold).to_csv(
        staging_dir / "test_segment_predictions.csv", index=False, lineterminator="\n"
    )
    fold_result.recording_predictions.assign(fold=fold_result.fold).to_csv(
        staging_dir / "test_recording_predictions.csv", index=False, lineterminator="\n"
    )
    fold_result.patient_predictions.assign(fold=fold_result.fold).to_csv(
        staging_dir / "test_patient_predictions.csv", index=False, lineterminator="\n"
    )
    fold_result.baseline_table.to_csv(staging_dir / "baseline_metrics.csv", index=False, lineterminator="\n")
    (staging_dir / "test_metrics.json").write_text(
        json.dumps(_json_safe(fold_result.test_metrics), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (staging_dir / "selected_hyperparameters.json").write_text(
        json.dumps(
            _json_safe({
                "C": fold_result.selected_C,
                "gamma": fold_result.selected_gamma,
                "n_support": fold_result.n_support,
            }),
            indent=2, ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    if fold_result.scaler is not None and fold_result.svm is not None:
        save_fold_model(staging_dir, fold_result)


def save_fold_model(staging_dir: Path, fold_result) -> Path:
    """Scaler + SVM de UN fold (ajustados sobre train+validation de ese
    fold), para poder reproducir exactamente sus predicciones de test sin
    volver a correr la busqueda de hiperparametros.

    No lleva la metadata completa del modelo final (patient/recording
    counts, dependencias, etc.): eso solo tiene sentido una vez, para el
    modelo que de verdad se entrega.
    """
    payload = {
        "scaler": fold_result.scaler,
        "svm": fold_result.svm,
        "fold": fold_result.fold,
        "C": fold_result.selected_C,
        "gamma": fold_result.selected_gamma,
    }
    path = staging_dir / "model_fold.joblib"
    tmp_path = path.with_name(path.name + ".part")
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, path)
    return path


# ---------------------------------------------------------------------------
# Modelo final: joblib + metadata + hash
# ---------------------------------------------------------------------------

def save_final_model(
    run_root: Path,
    dataset: str,
    condition: str,
    scaler,
    svm,
    feature_names: list[str],
    cfg: dict,
    C: float,
    gamma,
    negative_label_name: str,
    metadata_extra: dict,
) -> Path:
    final_dir = condition_dir(run_root, dataset, condition) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "scaler": scaler,
        "svm": svm,
        "feature_names": list(feature_names),
        "acoustic_config": cfg["acoustic"],
        "logmel_config": cfg["logmel"],
        "mfcc_config": cfg["mfcc"],
        # La clase negativa se nombra como corresponde al dataset (Healthy en
        # ICBHI, Normal en Fraiwan), no con la etiqueta generica "Control":
        # quien cargue el modelo debe saber contra que fue evaluado.
        "label_mapping": {negative_label_name: 0, "COPD": 1},
        "decision_threshold": float(cfg["svm"].get("decision_threshold", 0.0)),
        "dataset": dataset,
        "condition": condition,
        "C": float(C),
        "gamma": gamma,
        "sample_rate": cfg["acoustic"]["sample_rate"],
        "segment_length": cfg["acoustic"]["segment_length"],
    }
    model_path = final_dir / "final_model.joblib"
    tmp_path = model_path.with_name(model_path.name + ".part")
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, model_path)

    model_hash = sha256_file(model_path)
    (final_dir / "model_sha256.txt").write_text(model_hash + "\n", encoding="utf-8")

    n_support = {}
    if hasattr(svm, "classes_") and hasattr(svm, "n_support_"):
        n_support = {int(c): int(n) for c, n in zip(svm.classes_, svm.n_support_)}

    dependencies = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "librosa", "joblib"):
        try:
            dependencies[name] = getattr(__import__(name), "__version__", "desconocida")
        except ImportError:
            dependencies[name] = "no instalado"

    metadata = {
        "dataset": dataset,
        "condition": condition,
        "C": float(C),
        "gamma": gamma,
        "n_support_vectors": n_support,
        "model_sha256": model_hash,
        "created_at_utc": _now_iso(),
        "git": git_state(),
        "dependencies": dependencies,
        **_json_safe(metadata_extra),
    }
    (final_dir / "model_metadata.json").write_text(
        json.dumps(_json_safe(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return model_path


# ---------------------------------------------------------------------------
# Figuras: PNG a 300 DPI + PDF, con el CSV que las origina junto a cada una.
# Titulos y etiquetas en espanol, como exige el plan.
# ---------------------------------------------------------------------------

def _save_figure(fig, out_prefix: Path, cfg: dict) -> None:
    fig_cfg = cfg.get("figures", {})
    dpi = int(fig_cfg.get("dpi", 300))
    formats = fig_cfg.get("formats", ["png", "pdf"])
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_prefix.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray, y_score: np.ndarray, target_names: tuple[str, str],
    out_prefix: Path, cfg: dict, threshold: float = 0.0,
) -> None:
    """Matriz de confusion absoluta y normalizada, en dos paneles."""
    matrix_df = ev.confusion_matrix_df(y_true, y_score, target_names, threshold)
    matrix_df.to_csv(out_prefix.with_suffix(".csv"))
    matrix = matrix_df.to_numpy()
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, data, title, fmt in (
        (axes[0], matrix, "Matriz de confusion (conteos)", "d"),
        (axes[1], normalized, "Matriz de confusion (normalizada por fila)", ".2f"),
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        ax.set_xticks(range(len(target_names)))
        ax.set_yticks(range(len(target_names)))
        ax.set_xticklabels(target_names)
        ax.set_yticklabels(target_names)
        ax.set_xlabel("Prediccion")
        ax.set_ylabel("Real")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                text = f"{value:{fmt}}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if value > data.max() / 2 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_prefix: Path, cfg: dict) -> None:
    from sklearn.metrics import roc_auc_score

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=ev.POSITIVE_LABEL)
    auc = roc_auc_score(y_true, y_score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "umbral": thresholds}).to_csv(
        out_prefix.with_suffix(".csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_prefix: Path, cfg: dict) -> None:
    from sklearn.metrics import average_precision_score

    precision, recall, thresholds = precision_recall_curve(y_true, y_score, pos_label=ev.POSITIVE_LABEL)
    ap = average_precision_score(y_true, y_score, pos_label=ev.POSITIVE_LABEL)
    pd.DataFrame({
        "precision": precision, "recall": recall,
        "umbral": np.append(thresholds, np.nan),
    }).to_csv(out_prefix.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(recall, precision, label=f"Precision-Recall (AP = {ap:.3f})")
    ax.set_xlabel("Recall (COPD)")
    ax.set_ylabel("Precision (COPD)")
    ax.set_title("Curva Precision-Recall")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)


def plot_metrics_by_fold(metrics_by_fold: pd.DataFrame, negative_label_name: str, out_prefix: Path, cfg: dict) -> None:
    """Balanced accuracy, macro-F1 y ambos recalls, uno por fold."""
    neg_key = f"recall_{negative_label_name.strip().lower()}"
    columns = ["balanced_accuracy", "macro_f1", "recall_copd", neg_key]
    labels = ["Balanced accuracy", "Macro-F1", "Recall COPD", f"Recall {negative_label_name}"]
    metrics_by_fold.to_csv(out_prefix.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = metrics_by_fold["fold"].to_numpy()
    width = 0.8 / len(columns)
    for i, (col, label) in enumerate(zip(columns, labels)):
        ax.bar(x + i * width, metrics_by_fold[col], width=width, label=label)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Valor")
    ax.set_title("Metricas por fold")
    ax.set_xticks(x + width * (len(columns) - 1) / 2)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)


def plot_hyperparameter_heatmap(aggregated_grid: pd.DataFrame, metric_col: str, out_prefix: Path, cfg: dict) -> None:
    """Mapa de calor C x gamma de una metrica de validacion agregada."""
    aggregated_grid.to_csv(out_prefix.with_suffix(".csv"), index=False)
    pivot = aggregated_grid.pivot(index="C", columns="gamma", values=metric_col)
    pivot = pivot.sort_index(ascending=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(c) for c in pivot.index])
    ax.set_xlabel("gamma")
    ax.set_ylabel("C")
    ax.set_title(f"Mapa de calor: {metric_col} en validacion")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iat[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)


def plot_denoising_comparison(
    no_dn_patient_scores: pd.DataFrame, dn_patient_scores: pd.DataFrame, out_prefix: Path, cfg: dict,
) -> None:
    """Comparacion emparejada por paciente: puntaje no_dn frente a dn.

    Emparejar por ``patient_uid`` exige que ambas condiciones cubran la misma
    poblacion (la ablacion filtra ambas ramas por ``dn_reliable`` para
    garantizarlo); un paciente presente en una sola de las dos tablas se
    excluye del emparejamiento y se reporta aparte.
    """
    merged = no_dn_patient_scores.merge(
        dn_patient_scores, on="patient_uid", suffixes=("_no_dn", "_dn"), how="inner",
    )
    only_no_dn = set(no_dn_patient_scores["patient_uid"]) - set(dn_patient_scores["patient_uid"])
    only_dn = set(dn_patient_scores["patient_uid"]) - set(no_dn_patient_scores["patient_uid"])
    merged.to_csv(out_prefix.with_suffix(".csv"), index=False)
    if only_no_dn or only_dn:
        pd.DataFrame({
            "solo_en_no_dn": sorted(only_no_dn) + [""] * max(0, len(only_dn) - len(only_no_dn)),
            "solo_en_dn": sorted(only_dn) + [""] * max(0, len(only_no_dn) - len(only_dn)),
        }).to_csv(out_prefix.with_name(out_prefix.name + "_no_emparejados.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    labels = merged["target_label_no_dn"]
    colors = np.where(labels == ev.POSITIVE_LABEL, "tab:red", "tab:blue")
    ax.scatter(merged["score_no_dn"], merged["score_dn"], c=colors, alpha=0.8)
    limit = float(np.nanmax(np.abs(merged[["score_no_dn", "score_dn"]].to_numpy()))) if len(merged) else 1.0
    limit = max(limit, 1.0)
    ax.plot([-limit, limit], [-limit, limit], linestyle="--", color="gray")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Puntaje por paciente (no_dn)")
    ax.set_ylabel("Puntaje por paciente (dn)")
    ax.set_title("Comparacion emparejada: no_dn frente a dn")
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)


def plot_learning_curve(curve_df: pd.DataFrame, out_prefix: Path, cfg: dict) -> None:
    """Curva de aprendizaje (balanced accuracy y macro-F1 vs. fraccion de
    entrenamiento), calculada enteramente sobre train+validation del fold."""
    curve_df.to_csv(out_prefix.with_suffix(".csv"), index=False)
    if curve_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(
        curve_df["fraction"], curve_df["balanced_accuracy_mean"],
        yerr=curve_df["balanced_accuracy_std"].fillna(0.0), marker="o", label="Balanced accuracy",
    )
    ax.errorbar(
        curve_df["fraction"], curve_df["macro_f1_mean"],
        yerr=curve_df["macro_f1_std"].fillna(0.0), marker="s", label="Macro-F1",
    )
    ax.set_xlabel("Fraccion del conjunto de entrenamiento interno")
    ax.set_ylabel("Valor")
    ax.set_title("Curva de aprendizaje (sin usar el conjunto de prueba)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, out_prefix, cfg)
