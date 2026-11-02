"""Piezas de CNN/CRNN compartidas por los protocolos oof_v1 y fold_aware_v2."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import torch

from .. import artifacts as art
from .. import data as dmod
from ..models import cnn as cnn_model

MODEL_NAME = "cnn"

CNN_FINGERPRINT_SECTIONS = (
    "acoustic", "logmel", "splits", "weights", "normalization", "cnn", "training",
    "search", "selection", "evaluation", "augmentation", "seeds", "determinism",
    "bootstrap", "datasets", "experiments",
)

# La CRNN (``[model] architecture = "crnn"``) usa este mismo modulo: solo
# cambian la seccion de arquitectura en la huella y el directorio de resultados.
CRNN_FINGERPRINT_SECTIONS = tuple(s for s in CNN_FINGERPRINT_SECTIONS if s != "cnn") + ("model", "crnn")

# Secciones que crnn.toml debe copiar sin cambios de cnn.toml. [acoustic] y
# [logmel] garantizan ademas que la cache Log-Mel compartida sea la correcta.
SHARED_PROTOCOL_SECTIONS = (
    "acoustic", "logmel", "splits", "weights", "normalization", "training", "search",
    "selection", "evaluation", "augmentation", "seeds", "determinism", "bootstrap",
    "datasets", "experiments",
)


def fingerprint_sections(cfg: dict) -> tuple[str, ...]:
    return CRNN_FINGERPRINT_SECTIONS if dmod.model_architecture(cfg) == "crnn" else CNN_FINGERPRINT_SECTIONS


def config_consistency_checks(cfg: dict) -> list[dict]:
    """Comparaciones de configuracion con los otros modelos.

    ``blocking=True`` impide ejecutar: la CRNN no arranca si difiere de
    cnn.toml en alguna seccion del protocolo, porque perderia la
    comparabilidad con la CNN y podria regenerar (sobrescribir) la cache
    Log-Mel compartida. Para la CNN solo se informa si [acoustic]/[logmel]
    difieren de la SVM, como hasta ahora.
    """
    svm_cfg = dmod.load_config(dmod.DEFAULT_CONFIG_PATH)
    same_logmel = all(cfg.get(s) == svm_cfg.get(s) for s in dmod.LOGMEL_CONFIG_SECTIONS)
    checks = [{
        "check": "logmel_igual_svm", "ok": same_logmel, "blocking": False,
        "detail": "[acoustic] y [logmel] identicos a svm_rbf.toml" if same_logmel
        else "[acoustic]/[logmel] difieren de svm_rbf.toml",
    }]
    if dmod.model_architecture(cfg) == "crnn":
        reference_path = dmod.reference_cnn_config_path(cfg)
        reference_name = reference_path.name
        cnn_cfg = dmod.load_config(reference_path)
        for section in SHARED_PROTOCOL_SECTIONS:
            same = cfg.get(section) == cnn_cfg.get(section)
            checks.append({
                "check": f"{section}_igual_cnn", "ok": same, "blocking": True,
                "detail": f"identico a {reference_name}" if same
                else f"difiere de {reference_name}: se pierde la comparabilidad con la CNN",
            })
    return checks


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def smoke_test_config(cfg: dict) -> dict:
    """Una configuracion y dos epocas. El fold unico lo decide run_cnn_condition."""
    smoke = copy.deepcopy(cfg)
    smoke["search"]["configurations"] = cfg["search"]["configurations"][:1]
    smoke["training"]["max_epochs"] = min(2, int(cfg["training"]["max_epochs"]))
    smoke["training"]["min_epochs"] = min(int(cfg["training"]["min_epochs"]), smoke["training"]["max_epochs"])
    return smoke


def order_specs(specs: list[dmod.ConditionSpec]) -> list[dmod.ConditionSpec]:
    """Las condiciones que reutilizan hiperparametros van despues de su base, y
    la base tiene que estar en la misma ejecucion."""
    present = {(s.dataset, s.condition) for s in specs}
    for spec in specs:
        if spec.hyperparameters_from and (spec.dataset, spec.hyperparameters_from) not in present:
            raise ValueError(
                f"{spec.dataset}/{spec.condition} reutiliza hiperparametros de "
                f"{spec.hyperparameters_from}, que no esta en esta ejecucion"
            )
    return [s for s in specs if not s.hyperparameters_from] + [s for s in specs if s.hyperparameters_from]


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


def _write_json(obj, path: Path) -> None:
    path.write_text(json.dumps(art._json_safe(obj), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_torch_environment(run_root: Path, device: torch.device) -> None:
    path = run_root / "environment.txt"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = [f"{key}: {value}" for key, value in cnn_model.torch_environment(device).items()]
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def _discard_staging(staging: Path, device: torch.device) -> None:
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

def _fold_configurations(
    run_root: Path, spec: dmod.ConditionSpec, fold_id: int, cfg: dict,
) -> tuple[list[cnn_model.SearchConfiguration], str]:
    """Las 4 configuraciones de busqueda, o la que eligio la condicion base en
    ese mismo fold (condiciones con ``hyperparameters_from``)."""
    if spec.hyperparameters_from is None:
        return cnn_model.search_configurations(cfg), "search"
    source_dir = art.fold_dir(run_root, spec.dataset, spec.hyperparameters_from, fold_id)
    if not (source_dir / "_SUCCESS").is_file():
        raise RuntimeError(
            f"{spec.condition} reutiliza lr/dropout de {spec.hyperparameters_from}, "
            f"pero su fold {fold_id + 1} no esta completo"
        )
    selected = json.loads((source_dir / "selected_hyperparameters.json").read_text(encoding="utf-8"))
    config = cnn_model.SearchConfiguration(
        index=int(selected["config_index"]), lr=float(selected["lr"]), dropout=float(selected["dropout"]),
    )
    return [config], f"reused_from:{spec.hyperparameters_from}"


def write_cnn_fold_artifacts(
    staging: Path,
    result: cnn_model.CnnFoldResult,
    spec: dmod.ConditionSpec,
    cfg: dict,
    negative_label_name: str,
    input_hashes: dict[str, str],
) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    fold = result.fold
    _write_csv(result.search_summary, staging / "search_summary.csv")
    _write_csv(result.search_history, staging / "history.csv")
    _write_csv(result.refit_history, staging / "refit_history.csv")
    _write_csv(result.segment_predictions.assign(fold=fold), staging / "test_segment_predictions.csv")
    _write_csv(result.recording_predictions.assign(fold=fold), staging / "test_recording_predictions.csv")
    _write_csv(result.patient_predictions.assign(fold=fold), staging / "test_patient_predictions.csv")
    _write_csv(result.baseline_table, staging / "baseline_metrics.csv")
    _write_json(result.test_metrics, staging / "test_metrics.json")
    _write_json(result.selected, staging / "selected_hyperparameters.json")
    _write_json(
        {
            "busqueda_train": result.normalization_search.to_dict(),
            "reajuste_train_validation": result.normalization_refit.to_dict(),
        },
        staging / "normalization.json",
    )
    _write_json({"warnings": result.warnings, "seconds": result.seconds}, staging / "warnings.json")

    payload = cnn_model.checkpoint_payload(
        result.model_state, result.normalization_refit, cfg, spec, negative_label_name,
        hyperparameters={
            "config_index": result.selected["config_index"],
            "lr": result.selected["lr"],
            "dropout": result.selected["dropout"],
            "epochs": result.selected["best_epoch"],
        },
        extra={"fold": int(fold), "input_hashes": dict(input_hashes)},
    )
    cnn_model.save_checkpoint(staging / "model_fold.pt", payload)
    art.plot_training_curves(result.search_history, staging / "training_curves", cfg)


def _load_cnn_fold_outputs(run_root: Path, dataset: str, condition: str, fold_id: int) -> dict:
    fdir = art.fold_dir(run_root, dataset, condition, fold_id)
    return {
        "fold": fold_id,
        "search_summary": pd.read_csv(fdir / "search_summary.csv"),
        "test_metrics": json.loads((fdir / "test_metrics.json").read_text(encoding="utf-8")),
        "patient_predictions": pd.read_csv(fdir / "test_patient_predictions.csv", dtype={"patient_uid": str}),
        "baseline_metrics": pd.read_csv(fdir / "baseline_metrics.csv"),
        "selected": json.loads((fdir / "selected_hyperparameters.json").read_text(encoding="utf-8")),
    }


def _final_selection(
    run_root: Path, spec: dmod.ConditionSpec, fold_ids: list[int], search_summaries: pd.DataFrame,
) -> tuple[cnn_model.SearchConfiguration, int, pd.DataFrame, str]:
    """Configuracion del modelo final usando solo validation.

    Condicion normal: promedio de validation por configuracion entre folds y
    mediana de las mejores epocas de la elegida. Condicion con
    ``hyperparameters_from``: el lr/dropout final de la condicion base y la
    mediana de las mejores epocas de esta condicion.
    """
    if spec.hyperparameters_from is None:
        config, epochs, aggregated = cnn_model.select_final_configuration(search_summaries)
        return config, epochs, aggregated, "promedio de validation entre folds"
    base_tables = [
        pd.read_csv(art.fold_dir(run_root, spec.dataset, spec.hyperparameters_from, f) / "search_summary.csv")
        for f in fold_ids
    ]
    config, _, aggregated = cnn_model.select_final_configuration(pd.concat(base_tables, ignore_index=True))
    epochs = cnn_model.median_epochs(search_summaries["best_epoch"])
    source = (
        f"lr/dropout del modelo final de {spec.hyperparameters_from}; "
        f"epocas = mediana de las mejores epocas de {spec.condition}"
    )
    return config, epochs, aggregated, source


def _find_ok(results: list[dict], dataset: str, condition: str) -> dict | None:
    return next(
        (r for r in results if r["spec"].dataset == dataset and r["spec"].condition == condition and r["ok"]),
        None,
    )
