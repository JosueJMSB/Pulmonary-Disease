"""CLI unico del experimento SVM-RBF.

    python -u -m modeling.run_experiment \
        --model svm_rbf --dataset all --experiment all --n-jobs 4 \
        --data-root <ruta> --runs-root <ruta>

No hay rutas personales en este archivo: ``--data-root``/``--runs-root``/
``--cache-root``, o las variables ``PULMONARY_DATA_ROOT`` /
``PULMONARY_RUNS_ROOT`` / ``PULMONARY_CACHE_ROOT``, resuelven donde viven los
datos y los resultados en cada maquina (ver ``modeling.data.resolve_path``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import artifacts as art
from . import data as dmod
from . import evaluation as ev
from . import splits as sp
from .models import svm_rbf as svm_model

MODEL_CHOICES = ("svm_rbf",)
DATASET_CHOICES = ("ICBHI", "FRAIWAN_Extended", "all")
EXPERIMENT_CHOICES = ("main", "denoising_ablation", "all")

# Pareja (lado no_dn, lado dn) de cada dataset en la ablacion, para la figura
# de comparacion emparejada. En Fraiwan el lado no_dn es la misma condicion
# "main_no_dn" que ya corre en el experimento principal (todas sus
# grabaciones son dn_reliable=True, asi que no hace falta una condicion
# "no_dn_reliable" separada).
ABLATION_PAIRS = {
    "ICBHI": ("no_dn_reliable", "dn_reliable"),
    "FRAIWAN_Extended": ("main_no_dn", "dn"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y evalua SVM-RBF para COPD vs Control, dataset por dataset."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--experiment", choices=EXPERIMENT_CHOICES, required=True)
    parser.add_argument("--n-jobs", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument(
        "--cache-root", type=Path, default=None,
        help="Cache de caracteristicas (no listada como obligatoria en el plan, "
             "pero necesaria; por defecto PULMONARY_CACHE_ROOT o modeling/cache/features).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Ruta alternativa a svm_rbf.toml")
    parser.add_argument("--dry-run", action="store_true", help="Valida formas, hashes, conteos y folds; no entrena.")
    parser.add_argument("--smoke-test", action="store_true", help="Solo el fold 1 y la primera combinacion (C, gamma).")
    parser.add_argument("--force-features", action="store_true", help="Regenera la cache de caracteristicas.")
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
        help="Continua una ejecucion existente (RUN_ID o, sin valor, la mas reciente).",
    )
    return parser.parse_args(argv)


def resolve_datasets(cfg: dict, dataset_arg: str) -> list[str]:
    return dmod.dataset_names(cfg) if dataset_arg == "all" else [dataset_arg]


def resolve_experiments(experiment_arg: str) -> list[str]:
    return ["main", "denoising_ablation"] if experiment_arg == "all" else [experiment_arg]


def plan_conditions(cfg: dict, datasets: list[str], experiments: list[str]) -> list[dmod.ConditionSpec]:
    """Condiciones (dataset, condicion) a ejecutar, sin duplicar cuando una
    misma condicion aparece en mas de un experimento (el caso de Fraiwan
    main_no_dn, reutilizada en la ablacion)."""
    seen: dict[tuple[str, str], dmod.ConditionSpec] = {}
    for dataset in datasets:
        for experiment in experiments:
            for spec in dmod.experiment_condition_specs(cfg, experiment, dataset):
                seen[(spec.dataset, spec.condition)] = spec
    return list(seen.values())


def _smoke_test_config(cfg: dict) -> dict:
    """Copia de ``cfg`` con la rejilla recortada a un solo (C, gamma)."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["svm"]["c_grid"] = cfg["svm"]["c_grid"][:1]
    cfg2["svm"]["gamma_grid"] = cfg["svm"]["gamma_grid"][:1]
    return cfg2


# ---------------------------------------------------------------------------
# Huella de la ejecucion: lo que --resume debe verificar antes de reutilizar
# nada. Sin esto, un --resume podria continuar folds calculados sobre datos
# o configuracion distintos sin ningun aviso.
# ---------------------------------------------------------------------------

RUN_FINGERPRINT_CONFIG_SECTIONS = (
    "acoustic", "logmel", "mfcc", "summary", "splits", "weights",
    "svm", "selection", "bootstrap", "datasets", "experiments",
)


def _config_run_fingerprint(cfg: dict) -> str:
    relevant = {k: cfg[k] for k in RUN_FINGERPRINT_CONFIG_SECTIONS if k in cfg}
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _input_hashes_for_specs(data_root: Path, specs: list[dmod.ConditionSpec]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for dataset in sorted({s.dataset for s in specs}):
        csv_path = Path(data_root) / dataset / "segments.csv"
        hashes[f"{dataset}/segments.csv"] = dmod.sha256_file(csv_path)
    for spec in specs:
        npy_path = Path(data_root) / spec.dataset / f"segments_{spec.branch}.npy"
        hashes[f"{spec.dataset}/segments_{spec.branch}.npy"] = dmod.sha256_file(npy_path)
    return hashes


def build_run_fingerprint(
    cfg: dict, data_root: Path, specs: list[dmod.ConditionSpec], dataset_arg: str, experiment_arg: str,
) -> dict:
    """Todo lo que define si una ejecucion es 'la misma' para --resume.

    Los folds no se listan aparte: son una funcion determinista de la
    poblacion (que depende de estos mismos hashes de entrada) y de
    ``splits.random_state``/``splits.n_splits`` (dentro de la configuracion
    fingerprint-ada), asi que verificar datos + configuracion basta para
    garantizar que los folds serian identicos si se reconstruyeran.
    """
    return {
        "dataset_arg": dataset_arg,
        "experiment_arg": experiment_arg,
        "config_fingerprint": _config_run_fingerprint(cfg),
        "input_hashes": _input_hashes_for_specs(data_root, specs),
    }


def _run_fingerprint_path(run_root: Path) -> Path:
    return run_root / "run_fingerprint.json"


def write_run_fingerprint(run_root: Path, fingerprint: dict) -> None:
    _run_fingerprint_path(run_root).write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_run_fingerprint(run_root: Path, fingerprint: dict) -> None:
    """Se niega a reanudar si los datos, la configuracion o el alcance
    (--dataset/--experiment) de la ejecucion original cambiaron: reutilizar
    folds, features cacheadas o modelos ya calculados sobre una entrada
    distinta mezclaria resultados incoherentes sin ningun aviso.
    """
    path = _run_fingerprint_path(run_root)
    if not path.is_file():
        raise RuntimeError(
            f"no se puede reanudar {run_root}: falta run_fingerprint.json "
            "(ejecucion de una version del codigo anterior a esta comprobacion; "
            "empiece una ejecucion nueva)"
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if stored.get("dataset_arg") != fingerprint["dataset_arg"]:
        mismatches.append(f"--dataset cambio: {stored.get('dataset_arg')} -> {fingerprint['dataset_arg']}")
    if stored.get("experiment_arg") != fingerprint["experiment_arg"]:
        mismatches.append(f"--experiment cambio: {stored.get('experiment_arg')} -> {fingerprint['experiment_arg']}")
    if stored.get("config_fingerprint") != fingerprint["config_fingerprint"]:
        mismatches.append(
            "la configuracion cambio (acoustic/logmel/mfcc/summary/splits/weights/svm/"
            "selection/bootstrap/datasets/experiments)"
        )
    stored_hashes = stored.get("input_hashes", {})
    for key, value in fingerprint["input_hashes"].items():
        if stored_hashes.get(key) != value:
            mismatches.append(f"entrada modificada desde la ejecucion original: {key}")
    for key in stored_hashes:
        if key not in fingerprint["input_hashes"]:
            mismatches.append(f"la ejecucion original dependia de una entrada que ya no aplica: {key}")
    if mismatches:
        raise RuntimeError(
            f"--resume rechazado para {run_root}: no coincide con la ejecucion original.\n  - "
            + "\n  - ".join(mismatches)
        )


# ---------------------------------------------------------------------------
# --dry-run: valida formas, hashes, conteos y folds sin entrenar nada
# ---------------------------------------------------------------------------

def dry_run_check(data_root: Path, cfg: dict, specs: list[dmod.ConditionSpec]) -> dict:
    rows: list[dict] = []
    ok = True

    for dataset in sorted({s.dataset for s in specs}):
        try:
            segments = dmod.load_task_segments(data_root, dataset)
            manifest = dmod.load_task_manifest(data_root, dataset)
        except (FileNotFoundError, ValueError) as exc:
            rows.append({"dataset": dataset, "check": "carga_segments", "ok": False, "detail": str(exc)})
            ok = False
            continue
        rows.append({
            "dataset": dataset, "check": "carga_segments", "ok": True,
            "detail": f"{len(segments)} segmentos, {segments['patient_uid'].nunique()} pacientes",
        })

        manifest_hashes = manifest.get("output_hashes", {})
        csv_path = Path(data_root) / dataset / "segments.csv"
        csv_hash = dmod.sha256_file(csv_path)
        expected_csv_hash = manifest_hashes.get("segments.csv")
        csv_hash_ok = expected_csv_hash is not None and csv_hash == expected_csv_hash
        ok = ok and csv_hash_ok
        rows.append({
            "dataset": dataset, "check": "sha256_segments_csv", "ok": csv_hash_ok,
            "detail": (
                "coincide con manifest.json"
                if csv_hash_ok
                else f"{csv_hash[:12]}... vs manifest {str(expected_csv_hash)[:12]}..."
            ),
        })

        for branch in dmod.BRANCHES:
            try:
                array = dmod.load_branch_array(data_root, dataset, branch)
            except FileNotFoundError as exc:
                rows.append({"dataset": dataset, "check": f"npy_{branch}", "ok": False, "detail": str(exc)})
                ok = False
                continue
            shape_ok = array.shape[0] == len(segments) and array.shape[1] == int(cfg["acoustic"]["segment_length"])
            ok = ok and shape_ok
            rows.append({
                "dataset": dataset, "check": f"forma_{branch}", "ok": shape_ok,
                "detail": f"{array.shape} vs ({len(segments)}, {cfg['acoustic']['segment_length']})",
            })

            npy_path = Path(data_root) / dataset / f"segments_{branch}.npy"
            npy_hash = dmod.sha256_file(npy_path)
            expected_npy_hash = manifest_hashes.get(f"segments_{branch}.npy")
            npy_hash_ok = expected_npy_hash is not None and npy_hash == expected_npy_hash
            ok = ok and npy_hash_ok
            rows.append({
                "dataset": dataset, "check": f"sha256_{branch}", "ok": npy_hash_ok,
                "detail": (
                    "coincide con manifest.json"
                    if npy_hash_ok
                    else f"{npy_hash[:12]}... vs manifest {str(expected_npy_hash)[:12]}..."
                ),
            })

    fold_cache: dict[tuple[str, bool], sp.PatientFolds] = {}
    for spec in specs:
        try:
            segments = dmod.load_task_segments(data_root, spec.dataset)
            selected = dmod.select_condition_segments(segments, spec)
            key = (spec.dataset, spec.dn_reliable_only)
            if key not in fold_cache:
                fold_cache[key] = sp.build_patient_folds(
                    selected, n_splits=int(cfg["splits"]["n_splits"]),
                    random_state=int(cfg["splits"]["random_state"]),
                )
            n_patients = len(fold_cache[key].patient_table)
            rows.append({
                "dataset": spec.dataset, "check": f"folds[{spec.condition}]", "ok": True,
                "detail": f"{n_patients} pacientes, {fold_cache[key].n_splits} folds, verificado sin fuga",
            })
        except Exception as exc:  # noqa: BLE001 - se reporta, nunca se oculta
            ok = False
            rows.append({"dataset": spec.dataset, "check": f"folds[{spec.condition}]", "ok": False, "detail": str(exc)})

    return {"ok": ok, "checks": pd.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Un fold, con reintentos y marcas de fallo persistentes
# ---------------------------------------------------------------------------

def _fold_failure_marker(run_root: Path, dataset: str, condition: str, fold_id: int) -> Path:
    return art.condition_dir(run_root, dataset, condition) / f"fold_{fold_id + 1:02d}_FAILED.json"


def _write_fold_failure(run_root: Path, dataset: str, condition: str, fold_id: int, exc: Exception) -> None:
    marker = _fold_failure_marker(run_root, dataset, condition, fold_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({
            "fold": fold_id, "error": str(exc), "type": type(exc).__name__,
            "failed_at_utc": art._now_iso(),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _clear_fold_failure_marker(run_root: Path, dataset: str, condition: str, fold_id: int) -> None:
    marker = _fold_failure_marker(run_root, dataset, condition, fold_id)
    if marker.is_file():
        marker.unlink()


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


def run_condition(
    run_root: Path,
    condition_data: dmod.ConditionData,
    folds: sp.PatientFolds,
    cfg: dict,
    negative_label_name: str,
    n_jobs: int,
    smoke_test: bool,
    logger: logging.Logger,
) -> dict:
    """Todos los folds de una condicion (o solo el primero en --smoke-test).

    Cada fold es independiente de los demas: si uno falla, se marca FAILED,
    se registra en un archivo que sobrevive a la limpieza de staging, y se
    continua con el siguiente fold de la misma condicion. La condicion solo
    se resume (OOF, figuras, modelo final) si TODOS sus folds requeridos
    terminaron completos.
    """
    spec = condition_data.spec
    segments, X = condition_data.segments, condition_data.X
    fold_ids = [0] if smoke_test else list(range(folds.n_splits))
    cfg_used = _smoke_test_config(cfg) if smoke_test else cfg

    fold_status: dict[int, str] = {}
    for fold_id in fold_ids:
        if art.is_fold_complete(run_root, spec.dataset, spec.condition, fold_id):
            fold_status[fold_id] = "COMPLETED"
            logger.info(f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: ya publicado, se reutiliza")
            continue

        staging = art.fold_staging_dir(run_root, spec.dataset, spec.condition, fold_id)
        train_p, val_p, test_p = folds.get_split(fold_id)
        try:
            result = svm_model.run_fold(
                fold_id, segments, X, train_p, val_p, test_p, cfg_used, negative_label_name, n_jobs=n_jobs,
            )
            art.write_fold_artifacts(staging, result)
            art.publish_fold(staging)
            _clear_fold_failure_marker(run_root, spec.dataset, spec.condition, fold_id)
            fold_status[fold_id] = "COMPLETED"
            logger.info(
                f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: OK "
                f"C={result.selected_C} gamma={result.selected_gamma} "
                f"balanced_accuracy={result.test_metrics['balanced_accuracy']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - se registra, nunca se omite en silencio
            fold_status[fold_id] = "FAILED"
            _write_fold_failure(run_root, spec.dataset, spec.condition, fold_id, exc)
            logger.exception(f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: FALLO")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    completed_ids = [fid for fid, status in fold_status.items() if status == "COMPLETED"]
    condition_ok = len(completed_ids) == len(fold_ids)
    fold_outputs = [_load_fold_outputs(run_root, spec.dataset, spec.condition, fid) for fid in completed_ids]

    summary = None
    if condition_ok and fold_outputs:
        summary = summarize_condition(run_root, condition_data, folds, fold_outputs, cfg, negative_label_name, logger)

    return {"spec": spec, "fold_status": fold_status, "ok": condition_ok, "summary": summary}


# ---------------------------------------------------------------------------
# Resumen de una condicion: OOF, figuras, modelo final
# ---------------------------------------------------------------------------

def summarize_condition(
    run_root: Path,
    condition_data: dmod.ConditionData,
    folds: sp.PatientFolds,
    fold_outputs: list[dict],
    cfg: dict,
    negative_label_name: str,
    logger: logging.Logger,
) -> dict:
    spec = condition_data.spec
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    target_names = (negative_label_name, "COPD")
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))
    neg_key = negative_label_name.strip().lower()

    oof_patients = pd.concat([fo["patient_predictions"] for fo in fold_outputs], ignore_index=True)
    metrics_by_fold = pd.DataFrame([fo["test_metrics"] for fo in fold_outputs]).sort_values("fold").reset_index(drop=True)
    grid_tables = [fo["grid_search"] for fo in fold_outputs]
    baseline_metrics = pd.concat([fo["baseline_metrics"] for fo in fold_outputs], ignore_index=True)

    oof_y_true = oof_patients["target_label"].to_numpy()
    oof_y_score = oof_patients["score"].to_numpy()
    oof_metrics = ev.compute_patient_metrics(oof_y_true, oof_y_score, negative_label_name, threshold)

    bootstrap_cfg = cfg["bootstrap"]
    ci_specs = {
        "balanced_accuracy": ev.metric_fn_balanced_accuracy(threshold),
        "auroc": ev.metric_fn_auroc(),
        "auprc_copd": ev.metric_fn_auprc(),
        "recall_copd": ev.metric_fn_recall_copd(threshold),
        f"recall_{neg_key}": ev.metric_fn_recall_negative(threshold),
    }
    bootstrap_results = {
        name: ev.bootstrap_confidence_interval(
            oof_y_true, oof_y_score, fn,
            n_resamples=int(bootstrap_cfg["n_resamples"]),
            confidence=float(bootstrap_cfg["confidence"]),
            random_state=int(bootstrap_cfg["random_state"]),
        )
        for name, fn in ci_specs.items()
    }

    numeric_cols = ["accuracy", "balanced_accuracy", "recall_copd", f"recall_{neg_key}", "macro_f1", "auroc", "auprc_copd"]
    fold_mean = metrics_by_fold[numeric_cols].mean().to_dict()
    fold_std = metrics_by_fold[numeric_cols].std(ddof=1).to_dict()

    # Linea base de cordura (DummyClassifier), promediada entre folds por
    # estrategia: si la SVM no le saca ventaja clara en balanced_accuracy,
    # el resultado no es defendible aunque su cifra aislada luzca bien.
    baseline_mean = (
        baseline_metrics.groupby("strategy")[["balanced_accuracy", "recall_copd", f"recall_{neg_key}"]]
        .mean()
        .to_dict(orient="index")
    )

    summary_row = {
        "dataset": spec.dataset, "condition": spec.condition, "branch": spec.branch,
        "n_folds": len(fold_outputs),
        **{f"oof_{k}": v for k, v in oof_metrics.items()},
        **{f"fold_mean_{k}": v for k, v in fold_mean.items()},
        **{f"fold_std_{k}": v for k, v in fold_std.items()},
        **{
            f"ci95_{name}_{bound}": bootstrap_results[name][bound]
            for name in bootstrap_results for bound in ("point", "lower", "upper")
        },
        **{
            f"baseline_{strategy}_{metric}": value
            for strategy, metrics in baseline_mean.items()
            for metric, value in metrics.items()
        },
    }

    classification_report = ev.classification_report_df(oof_y_true, oof_y_score, target_names, threshold)
    confusion_matrix = ev.confusion_matrix_df(oof_y_true, oof_y_score, target_names, threshold)

    cdir.mkdir(parents=True, exist_ok=True)
    oof_patients.to_csv(cdir / "oof_patient_predictions.csv", index=False, lineterminator="\n")
    metrics_by_fold.to_csv(cdir / "metrics_by_fold.csv", index=False, lineterminator="\n")
    baseline_metrics.to_csv(cdir / "baseline_metrics.csv", index=False, lineterminator="\n")
    pd.concat(grid_tables, ignore_index=True).to_csv(cdir / "hyperparameter_search.csv", index=False, lineterminator="\n")
    classification_report.to_csv(cdir / "classification_report.csv", index=False, lineterminator="\n")
    confusion_matrix.to_csv(cdir / "confusion_matrix.csv")
    (cdir / "metrics_summary.json").write_text(
        json.dumps(art._json_safe(summary_row), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    prefix = run_root / "figures" / f"{spec.dataset}__{spec.condition}"
    art.plot_confusion_matrix(oof_y_true, oof_y_score, target_names,
                               prefix.with_name(prefix.name + "__confusion_matrix"), cfg, threshold)
    art.plot_roc_curve(oof_y_true, oof_y_score, prefix.with_name(prefix.name + "__roc"), cfg)
    art.plot_pr_curve(oof_y_true, oof_y_score, prefix.with_name(prefix.name + "__pr"), cfg)
    art.plot_metrics_by_fold(metrics_by_fold, negative_label_name,
                              prefix.with_name(prefix.name + "__metrics_by_fold"), cfg)

    final_C, final_gamma, aggregated_grid = svm_model.select_final_hyperparameters(grid_tables, cfg)
    art.plot_hyperparameter_heatmap(aggregated_grid, "balanced_accuracy",
                                     prefix.with_name(prefix.name + "__hyperparam_heatmap"), cfg)

    learning_curve_df = pd.DataFrame()
    fold0 = next((fo for fo in fold_outputs if fo["fold"] == 0), None)
    if fold0 is not None:
        train_p, val_p, _test_p = folds.get_split(0)
        pool_seg, pool_X = dmod.select_rows_by_patients(condition_data.segments, condition_data.X, train_p + val_p)
        try:
            learning_curve_df = svm_model.compute_learning_curve(
                pool_seg, pool_X, float(fold0["selected"]["C"]), fold0["selected"]["gamma"], cfg,
            )
        except ValueError as exc:
            logger.warning(f"{spec.dataset}/{spec.condition}: curva de aprendizaje omitida ({exc})")
    art.plot_learning_curve(learning_curve_df, prefix.with_name(prefix.name + "__learning_curve"), cfg)

    scaler, svm, _weights = svm_model.fit_final_model(
        condition_data.segments, condition_data.X, final_C, final_gamma, cfg
    )
    art.save_final_model(
        run_root, spec.dataset, spec.condition, scaler, svm, condition_data.feature_names, cfg,
        final_C, final_gamma, negative_label_name,
        metadata_extra={
            "n_patients": int(condition_data.segments["patient_uid"].nunique()),
            "n_recordings": int(condition_data.segments["audio_id"].nunique()),
            "n_segments": int(len(condition_data.segments)),
            "oof_metrics": oof_metrics,
            "baseline_mean": baseline_mean,
            "aggregated_validation_grid": aggregated_grid.to_dict(orient="records"),
        },
    )

    logger.info(
        f"{spec.dataset}/{spec.condition}: OOF balanced_accuracy={oof_metrics['balanced_accuracy']:.3f} "
        f"auroc={oof_metrics['auroc']:.3f}  modelo final C={final_C} gamma={final_gamma}"
    )

    return {
        "summary_row": summary_row,
        "oof_patients": oof_patients,
        "metrics_by_fold": metrics_by_fold,
        "baseline_metrics": baseline_metrics,
        "hyperparameter_search": pd.concat(grid_tables, ignore_index=True),
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = dmod.load_config(args.config)

    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = resolve_datasets(cfg, args.dataset)
    experiments = resolve_experiments(args.experiment)
    specs = plan_conditions(cfg, datasets, experiments)

    if args.dry_run:
        report = dry_run_check(data_root, cfg, specs)
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(report["checks"].to_string(index=False))
        print(f"\nveredicto: {'OK' if report['ok'] else 'REVISAR'}")
        return 0 if report["ok"] else 1

    run_root, run_id = art.init_run(runs_root, args.model, args.resume, bool(args.resume))
    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} dataset={args.dataset} experiment={args.experiment} "
        f"n_jobs={args.n_jobs} smoke_test={args.smoke_test} resume={bool(args.resume)}"
    )

    # --resume: TODO lo que sigue en este bloque debe ocurrir en este orden
    # exacto. verify_run_fingerprint() es lo primero -antes de tocar
    # status.json, environment.txt, resolved_config.toml o borrar staging
    # abandonado-, porque cualquiera de esas escrituras es irreversible y
    # aqui todavia no se sabe si esta ejecucion es compatible con la
    # original. Si la huella no coincide, se devuelve sin haber mutado nada
    # de eso (solo se registra el rechazo en status.json, ya como resultado
    # de la verificacion, no como paso previo a ella).
    fingerprint = build_run_fingerprint(cfg, data_root, specs, args.dataset, args.experiment)
    if args.resume:
        try:
            verify_run_fingerprint(run_root, fingerprint)
        except RuntimeError as exc:
            logger.error(str(exc))
            art.write_status(run_root, art.STATUS_FAILED, {"error": str(exc)})
            print(str(exc), file=sys.stderr)
            return 1
        logger.info("run_fingerprint verificado: datos, configuracion y alcance coinciden con la ejecucion original")

        removed_staging = art.finalize_resume(run_root)
        if removed_staging:
            logger.info(f"staging abandonado eliminado: {[str(p) for p in removed_staging]}")

    art.write_environment(run_root)
    art.write_resolved_config(run_root, cfg, vars(args))
    if not args.resume:
        write_run_fingerprint(run_root, fingerprint)

    folds_by_population: dict[tuple[str, bool], sp.PatientFolds] = {}
    condition_results: list[dict] = []

    for spec in specs:
        logger.info(f"preparando {spec.dataset}/{spec.condition} (rama {spec.branch})")
        try:
            condition_data = dmod.build_condition_data(
                data_root, cache_root, spec, cfg, force_features=args.force_features,
            )

            pop_key = (spec.dataset, spec.dn_reliable_only)
            new_folds = sp.build_patient_folds(
                condition_data.segments,
                n_splits=int(cfg["splits"]["n_splits"]),
                random_state=int(cfg["splits"]["random_state"]),
            )
            if pop_key in folds_by_population:
                if not sp.folds_are_identical(folds_by_population[pop_key], new_folds):
                    raise RuntimeError(
                        f"{spec.dataset}: los folds de '{spec.condition}' no coinciden con los "
                        "de otra condicion que deberia compartir poblacion (no_dn vs dn)"
                    )
            else:
                folds_by_population[pop_key] = new_folds

            negative_label_name = cfg["datasets"][spec.dataset]["negative_label_name"]
            result = run_condition(
                run_root, condition_data, folds_by_population[pop_key], cfg,
                negative_label_name, args.n_jobs, args.smoke_test, logger,
            )
        except Exception:
            logger.exception(f"{spec.dataset}/{spec.condition}: fallo no controlado durante la preparacion")
            result = {"spec": spec, "fold_status": {}, "ok": False, "summary": None}
        condition_results.append(result)

    # Comparacion emparejada no_dn vs dn, cuando ambos lados de la pareja
    # del dataset corrieron y terminaron OK en esta ejecucion.
    for dataset in datasets:
        pair = ABLATION_PAIRS.get(dataset)
        if pair is None:
            continue
        left = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == pair[0] and r["ok"]), None)
        right = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == pair[1] and r["ok"]), None)
        if left and right:
            prefix = run_root / "figures" / f"{dataset}__{pair[0]}_vs_{pair[1]}__denoising_comparison"
            art.plot_denoising_comparison(left["summary"]["oof_patients"], right["summary"]["oof_patients"], prefix, cfg)
            logger.info(f"{dataset}: figura de comparacion no_dn/dn generada ({pair[0]} vs {pair[1]})")

    ok_results = [r for r in condition_results if r["ok"] and r["summary"] is not None]
    if ok_results:
        def _tag(df: pd.DataFrame, r: dict) -> pd.DataFrame:
            return df.assign(dataset=r["spec"].dataset, condition=r["spec"].condition)

        pd.concat([_tag(r["summary"]["metrics_by_fold"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "metrics_by_fold.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["hyperparameter_search"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "hyperparameter_search.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame([r["summary"]["summary_row"] for r in ok_results]).to_csv(
            run_root / "metrics_summary.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["oof_patients"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "oof_patient_predictions.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["baseline_metrics"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "baseline_metrics.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["classification_report"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "classification_report.csv", index=False, lineterminator="\n"
        )
        confusion_frames = []
        for r in ok_results:
            cm = r["summary"]["confusion_matrix"].reset_index().rename(columns={"index": "real"})
            cm["dataset"], cm["condition"] = r["spec"].dataset, r["spec"].condition
            confusion_frames.append(cm)
        pd.concat(confusion_frames, ignore_index=True).to_csv(
            run_root / "confusion_matrix.csv", index=False, lineterminator="\n"
        )

    if folds_by_population:
        fold_tables = []
        for (dataset, dn_reliable_only), folds in folds_by_population.items():
            table = folds.fold_table.copy()
            table["dataset"] = dataset
            table["dn_reliable_only"] = dn_reliable_only
            fold_tables.append(table)
        pd.concat(fold_tables, ignore_index=True).to_csv(run_root / "folds.csv", index=False, lineterminator="\n")

    all_ok = bool(condition_results) and all(r["ok"] for r in condition_results)
    any_ok = any(r["ok"] for r in condition_results)
    if all_ok:
        final_status = art.STATUS_COMPLETED
    elif any_ok:
        final_status = art.STATUS_PARTIAL
    else:
        final_status = art.STATUS_FAILED

    art.write_status(run_root, final_status, {
        "conditions": [
            {
                "dataset": r["spec"].dataset, "condition": r["spec"].condition,
                "ok": r["ok"], "fold_status": r["fold_status"],
            }
            for r in condition_results
        ],
    })
    logger.info(f"ejecucion {run_id} finalizada con estado {final_status}")
    print(f"run_id={run_id} estado={final_status} -> {run_root}")
    return 0 if final_status == art.STATUS_COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
