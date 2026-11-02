"""SVM-RBF bajo el protocolo oof_v1 (flujo por defecto de ``run_experiment``)."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd

from ... import artifacts as art
from ... import data as dmod
from ... import evaluation as ev
from ... import splits as sp
from ...models import svm_rbf as svm_model
from ...runner.cli import plan_conditions, resolve_datasets, resolve_experiments
from ...runner.dry_run import dry_run_check
from ...runner.fingerprint import build_run_fingerprint, open_run, write_run_fingerprint
from ...runner.fold_markers import _clear_fold_failure_marker, _write_fold_failure
from ..svm_common import _load_fold_outputs, _smoke_test_config

# Pareja (lado no_dn, lado dn) de cada dataset en la ablacion, para la figura
# de comparacion emparejada. En Fraiwan el lado no_dn es la misma condicion
# "main_no_dn" que ya corre en el experimento principal (todas sus
# grabaciones son dn_reliable=True, asi que no hace falta una condicion
# "no_dn_reliable" separada).
ABLATION_PAIRS = {
    "ICBHI": ("no_dn_reliable", "dn_reliable"),
    "FRAIWAN_Extended": ("main_no_dn", "dn"),
    "COMBINED": ("no_dn_reliable", "dn_reliable"),
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

    oof_patients = ev.attach_source_dataset(oof_patients, condition_data.segments)

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

    by_source = None
    if bool(cfg.get("evaluation", {}).get("report_by_source", False)):
        by_source = ev.by_source_report(
            oof_patients, "source_dataset", negative_label_name, threshold, ci_specs, bootstrap_cfg,
        )

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
    if by_source is not None:
        for name, table in by_source.items():
            table.to_csv(cdir / f"{name}.csv", index=False, lineterminator="\n")

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
        "by_source": by_source,
    }


# ---------------------------------------------------------------------------
# Ejecucion completa de la SVM bajo oof_v1
# ---------------------------------------------------------------------------

def run_oof_svm(args: argparse.Namespace, cfg: dict) -> int:
    """Punto de entrada del protocolo oof_v1 para la SVM. ``run_experiment.main``
    despacha aqui cuando el TOML no es holdout-v3 ni fold-aware."""
    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = resolve_datasets(cfg, args.dataset)
    experiments = resolve_experiments(args.experiment, cfg)
    specs = plan_conditions(cfg, datasets, experiments)

    if args.dry_run:
        report = dry_run_check(data_root, cfg, specs)
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(report["checks"].to_string(index=False))
        print(f"\nveredicto: {'OK' if report['ok'] else 'REVISAR'}")
        return 0 if report["ok"] else 1

    # --resume: la huella se verifica ANTES de abrir run.log o escribir
    # cualquier archivo de la ejecucion existente (status.json,
    # environment.txt, resolved_config.toml, staging). Si no coincide, se sale
    # con error y la ejecucion queda intacta (ver open_run).
    fingerprint = build_run_fingerprint(cfg, data_root, specs, args.dataset, args.experiment)
    try:
        run_root, run_id = open_run(runs_root, args.model, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} dataset={args.dataset} experiment={args.experiment} "
        f"n_jobs={args.n_jobs} smoke_test={args.smoke_test} resume={bool(args.resume)}"
    )
    if args.resume:
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
                stratify_by_dataset=bool(cfg["splits"].get("stratify_by_dataset", False)),
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

        by_source_keys = (
            "metrics_by_source", "classification_report_by_source",
            "confusion_matrix_by_source", "bootstrap_by_source",
        )
        for key in by_source_keys:
            frames = [
                _tag(r["summary"]["by_source"][key], r) for r in ok_results
                if r["summary"].get("by_source")
            ]
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    run_root / f"{key}.csv", index=False, lineterminator="\n"
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
