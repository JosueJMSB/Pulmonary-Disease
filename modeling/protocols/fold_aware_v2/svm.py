"""SVM-RBF bajo el protocolo fold-aware (v2)."""

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
from ...runner.dry_run import dry_run_check_folded
from ...runner.fingerprint import build_folded_run_fingerprint, open_run, write_run_fingerprint
from ...runner.fold_markers import _clear_fold_failure_marker, _write_fold_failure
from ..svm_common import _load_fold_outputs, _smoke_test_config
from .config import _patient_folds_paths, _validate_final_model_disabled


# ---------------------------------------------------------------------------
# Protocolo fold-aware (v2): un fold y el resumen de una condicion,
# reconstruyendo la cache de cada fold por separado en vez de repartir una
# unica cache por dataset entre los 5 folds. ``run_fold``/``write_fold_artifacts``/
# ``publish_fold`` se reutilizan sin ningun cambio.
# ---------------------------------------------------------------------------

def run_condition_folded(
    run_root: Path,
    data_root: Path,
    cache_root: Path,
    spec: dmod.ConditionSpec,
    folds: sp.PatientFolds,
    cfg: dict,
    negative_label_name: str,
    n_jobs: int,
    smoke_test: bool,
    force_features: bool,
    logger: logging.Logger,
) -> dict:
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
            condition_data = dmod.build_condition_data(
                data_root, cache_root, spec, cfg_used, force_features=force_features, fold_id=fold_id,
            )
            result = svm_model.run_fold(
                fold_id, condition_data.segments, condition_data.X, train_p, val_p, test_p,
                cfg_used, negative_label_name, n_jobs=n_jobs,
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
        # source_dataset: la misma poblacion de pacientes vive en todos los
        # folds de un dataset (solo cambia como se denoisea su audio), asi
        # que el segments.csv de cualquier fold completado basta para mapear
        # patient_uid -> source_dataset.
        source_segments = dmod.load_task_segments(data_root, spec.dataset, fold_id=completed_ids[0])
        summary = summarize_condition_folded(
            run_root, spec, source_segments, fold_outputs, cfg, negative_label_name, logger,
        )

    return {"spec": spec, "fold_status": fold_status, "ok": condition_ok, "summary": summary}


def summarize_condition_folded(
    run_root: Path,
    spec: dmod.ConditionSpec,
    source_segments: pd.DataFrame,
    fold_outputs: list[dict],
    cfg: dict,
    negative_label_name: str,
    logger: logging.Logger,
) -> dict:
    """Como ``summarize_condition``, salvo que no ajusta ningun modelo final
    ni curva de aprendizaje: el protocolo fold-aware no tiene una unica cache
    de todos los pacientes sobre la que hacerlo (cada fold recalibro su
    propio denoising), y ese entrenamiento queda fuera de este protocolo de
    todos modos (ver ``_validate_final_model_disabled``). Todo lo demas
    -busqueda de hiperparametros, su heatmap, metricas por fold, OOF,
    bootstrap, por-fuente- se conserva igual.
    """
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    target_names = (negative_label_name, "COPD")
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))
    neg_key = negative_label_name.strip().lower()

    oof_patients = pd.concat([fo["patient_predictions"] for fo in fold_outputs], ignore_index=True)
    metrics_by_fold = pd.DataFrame([fo["test_metrics"] for fo in fold_outputs]).sort_values("fold").reset_index(drop=True)
    grid_tables = [fo["grid_search"] for fo in fold_outputs]
    baseline_metrics = pd.concat([fo["baseline_metrics"] for fo in fold_outputs], ignore_index=True)

    oof_patients = ev.attach_source_dataset(oof_patients, source_segments)

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

    _final_C, _final_gamma, aggregated_grid = svm_model.select_final_hyperparameters(grid_tables, cfg)
    art.plot_hyperparameter_heatmap(aggregated_grid, "balanced_accuracy",
                                     prefix.with_name(prefix.name + "__hyperparam_heatmap"), cfg)

    logger.info(
        f"{spec.dataset}/{spec.condition}: OOF balanced_accuracy={oof_metrics['balanced_accuracy']:.3f} "
        f"auroc={oof_metrics['auroc']:.3f}"
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


def _write_folded_run_tables(
    run_root: Path, condition_results: list[dict], folds_by_scope: dict[str, sp.PatientFolds],
) -> None:
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

    if folds_by_scope:
        fold_tables = []
        for dataset, folds in folds_by_scope.items():
            table = folds.fold_table.copy()
            table["dataset"] = dataset
            fold_tables.append(table)
        pd.concat(fold_tables, ignore_index=True).to_csv(run_root / "folds.csv", index=False, lineterminator="\n")


def run_folded_svm(args: argparse.Namespace, cfg: dict) -> int:
    """Punto de entrada del protocolo fold-aware (v2) para la SVM. ``main()``
    despacha aqui cuando el TOML declara ``[folds] patient_folds_csv``."""
    _validate_final_model_disabled(cfg)
    patient_folds_csv, patient_folds_manifest = _patient_folds_paths(cfg)

    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = resolve_datasets(cfg, args.dataset)
    experiments = resolve_experiments(args.experiment, cfg)
    specs = plan_conditions(cfg, datasets, experiments)

    if args.dry_run:
        report = dry_run_check_folded(data_root, cfg, specs, patient_folds_csv, patient_folds_manifest)
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(report["checks"].to_string(index=False))
        print(f"\nveredicto: {'OK' if report['ok'] else 'REVISAR'}")
        return 0 if report["ok"] else 1

    try:
        folds_by_scope = {
            dataset: sp.load_patient_folds(patient_folds_csv, patient_folds_manifest, dataset)
            for dataset in sorted({s.dataset for s in specs})
        }
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # --resume: la huella se verifica ANTES de abrir run.log o escribir
    # cualquier archivo de la ejecucion existente (ver open_run).
    fingerprint = build_folded_run_fingerprint(
        cfg, data_root, specs, args.dataset, args.experiment, folds_by_scope, smoke_test=bool(args.smoke_test),
    )
    try:
        run_root, run_id = open_run(runs_root, args.model, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} protocolo=fold-aware dataset={args.dataset} experiment={args.experiment} "
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

    condition_results: list[dict] = []
    for spec in specs:
        logger.info(f"preparando {spec.dataset}/{spec.condition} (rama {spec.branch}, protocolo fold-aware)")
        try:
            negative_label_name = cfg["datasets"][spec.dataset]["negative_label_name"]
            result = run_condition_folded(
                run_root, data_root, cache_root, spec, folds_by_scope[spec.dataset], cfg,
                negative_label_name, args.n_jobs, args.smoke_test, args.force_features, logger,
            )
        except Exception:
            logger.exception(f"{spec.dataset}/{spec.condition}: fallo no controlado durante la preparacion")
            result = {"spec": spec, "fold_status": {}, "ok": False, "summary": None}
        condition_results.append(result)

    # Comparacion emparejada no_dn vs dn (denoising_ablation), cuando ambos
    # lados corrieron y terminaron OK en esta ejecucion.
    for dataset in datasets:
        left = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == "no_dn" and r["ok"]), None)
        right = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == "dn" and r["ok"]), None)
        if left and right:
            prefix = run_root / "figures" / f"{dataset}__no_dn_vs_dn__denoising_comparison"
            art.plot_denoising_comparison(left["summary"]["oof_patients"], right["summary"]["oof_patients"], prefix, cfg)
            logger.info(f"{dataset}: figura de comparacion no_dn/dn generada")

    _write_folded_run_tables(run_root, condition_results, folds_by_scope)

    for r in condition_results:
        if r["ok"]:
            final_dir = art.condition_dir(run_root, r["spec"].dataset, r["spec"].condition) / "final"
            if final_dir.exists():
                raise RuntimeError(
                    f"{r['spec'].dataset}/{r['spec'].condition}: existe {final_dir}, pero el protocolo "
                    "fold-aware (v2) no debe generar ningun modelo final"
                )

    all_ok = bool(condition_results) and all(r["ok"] for r in condition_results)
    any_ok = any(r["ok"] for r in condition_results)
    final_status = art.STATUS_COMPLETED if all_ok else (art.STATUS_PARTIAL if any_ok else art.STATUS_FAILED)
    art.write_status(run_root, final_status, {
        "conditions": [
            {"dataset": r["spec"].dataset, "condition": r["spec"].condition, "ok": r["ok"], "fold_status": r["fold_status"]}
            for r in condition_results
        ],
    })
    logger.info(f"ejecucion {run_id} finalizada con estado {final_status}")
    print(f"run_id={run_id} estado={final_status} -> {run_root}")
    return 0 if final_status == art.STATUS_COMPLETED else 1
