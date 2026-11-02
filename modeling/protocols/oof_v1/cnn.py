"""CNN/CRNN bajo el protocolo oof_v1."""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch

from ... import artifacts as art
from ... import data as dmod
from ... import evaluation as ev
from ... import splits as sp
from ...models import cnn as cnn_model
from ...runner.cli import plan_conditions, resolve_datasets, resolve_experiments
from ...runner.dry_run import dry_run_check
from ...runner.fingerprint import build_run_fingerprint, open_run, write_run_fingerprint
from ...runner.fold_markers import _clear_fold_failure_marker, _write_fold_failure
from ..cnn_common import (
    _append_torch_environment,
    _discard_staging,
    _final_selection,
    _find_ok,
    _fold_configurations,
    _load_cnn_fold_outputs,
    _write_csv,
    _write_json,
    config_consistency_checks,
    fingerprint_sections,
    order_specs,
    smoke_test_config,
    write_cnn_fold_artifacts,
)

# (condicion izquierda, condicion derecha, nombre de la comparacion) por dataset.
PAIRED_COMPARISONS = {
    "ICBHI": (
        ("no_dn_reliable", "dn_reliable", "denoising"),
        ("main_no_dn", "main_no_dn_aug", "augmentation"),
    ),
    "FRAIWAN_Extended": (
        ("main_no_dn", "dn", "denoising"),
        ("main_no_dn", "main_no_dn_aug", "augmentation"),
    ),
    "COMBINED": (
        ("no_dn_reliable", "dn_reliable", "denoising"),
        ("main_no_dn", "main_no_dn_aug", "augmentation"),
    ),
}


def _condition_input_hashes(fingerprint: dict, cache_root: Path, spec: dmod.ConditionSpec) -> dict[str, str]:
    wanted = (f"{spec.dataset}/segments.csv", f"{spec.dataset}/segments_{spec.branch}.npy")
    hashes = {k: fingerprint["input_hashes"][k] for k in wanted if k in fingerprint["input_hashes"]}
    manifest_path = dmod.logmel_cache_paths(dmod.logmel_cache_dir(cache_root, spec.dataset, spec.branch))["manifest"]
    if manifest_path.is_file():
        output_hash = json.loads(manifest_path.read_text(encoding="utf-8")).get("output_sha256")
        if output_hash:
            hashes[f"{spec.dataset}/{spec.branch}/logmel.npy"] = output_hash
    return hashes


def run_cnn_condition(
    run_root: Path,
    condition: dmod.ConditionLogmel,
    folds: sp.PatientFolds,
    cfg: dict,
    cfg_used: dict,
    negative_label_name: str,
    device: torch.device,
    num_workers: int,
    smoke_test: bool,
    input_hashes: dict[str, str],
    logger,
) -> dict:
    """Todos los folds de una condicion (solo el primero en --smoke-test).

    Un fold fallido queda marcado FAILED con su causa y no impide los demas.
    Un error de memoria de GPU no reduce el batch: cambiaria el experimento.
    La condicion solo se resume si todos sus folds requeridos terminaron.
    """
    spec = condition.spec
    tag = f"{spec.dataset}/{spec.condition}"
    fold_ids = [0] if smoke_test else list(range(folds.n_splits))
    fold_status: dict[int, str] = {}

    for fold_id in fold_ids:
        if art.is_fold_complete(run_root, spec.dataset, spec.condition, fold_id):
            fold_status[fold_id] = "COMPLETED"
            logger.info(f"{tag} fold {fold_id + 1}: ya publicado, se reutiliza")
            continue

        staging = art.fold_staging_dir(run_root, spec.dataset, spec.condition, fold_id)
        train_p, val_p, test_p = folds.get_split(fold_id)
        try:
            configurations, source = _fold_configurations(run_root, spec, fold_id, cfg_used)
            logger.info(
                f"{tag} fold {fold_id + 1}: {len(train_p)}/{len(val_p)}/{len(test_p)} pacientes "
                f"train/validation/test, {len(configurations)} configuracion(es) ({source})"
            )
            result = cnn_model.run_cnn_fold(
                fold_id, condition, train_p, val_p, test_p, cfg_used, configurations,
                negative_label_name, device, num_workers, hyperparameter_source=source,
                log=lambda message, f=fold_id: logger.info(f"{tag} fold {f + 1}: {message}"),
            )
            write_cnn_fold_artifacts(staging, result, spec, cfg_used, negative_label_name, input_hashes)
            art.publish_fold(staging)
            _clear_fold_failure_marker(run_root, spec.dataset, spec.condition, fold_id)
            fold_status[fold_id] = "COMPLETED"
            for message in result.warnings:
                logger.warning(f"{tag} fold {fold_id + 1}: {message}")
            logger.info(
                f"{tag} fold {fold_id + 1}: OK lr={result.selected['lr']:g} "
                f"dropout={result.selected['dropout']:g} epocas={result.selected['best_epoch']} "
                f"test_BA={result.test_metrics['balanced_accuracy']:.3f} ({result.seconds / 60:.1f} min)"
            )
        except torch.cuda.OutOfMemoryError as exc:
            fold_status[fold_id] = "FAILED"
            error = RuntimeError(
                "memoria de GPU insuficiente. No se reduce batch_size automaticamente porque "
                "cambiaria el experimento: libere memoria o use otra GPU (--device cuda:N) y "
                f"continue con --resume. Detalle: {exc}"
            )
            _write_fold_failure(run_root, spec.dataset, spec.condition, fold_id, error)
            logger.error(f"{tag} fold {fold_id + 1}: {error}")
            _discard_staging(staging, device)
        except Exception as exc:  # noqa: BLE001 - se registra, nunca se omite en silencio
            fold_status[fold_id] = "FAILED"
            _write_fold_failure(run_root, spec.dataset, spec.condition, fold_id, exc)
            logger.exception(f"{tag} fold {fold_id + 1}: FALLO")
            _discard_staging(staging, device)

    condition_ok = all(fold_status.get(f) == "COMPLETED" for f in fold_ids)
    summary = None
    if condition_ok:
        try:
            fold_outputs = [_load_cnn_fold_outputs(run_root, spec.dataset, spec.condition, f) for f in fold_ids]
            summary = summarize_cnn_condition(
                run_root, condition, cfg, cfg_used, negative_label_name, fold_ids, fold_outputs,
                device, num_workers, input_hashes, logger,
            )
        except torch.cuda.OutOfMemoryError:
            condition_ok = False
            logger.exception(f"{tag}: memoria de GPU insuficiente en el resumen/modelo final; continue con --resume")
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            condition_ok = False
            logger.exception(f"{tag}: fallo al resumir la condicion o entrenar el modelo final")

    return {"spec": spec, "fold_status": fold_status, "ok": condition_ok, "summary": summary}


# ---------------------------------------------------------------------------
# Resumen de una condicion y modelo final
# ---------------------------------------------------------------------------

def _ensure_final_model(
    run_root: Path,
    condition: dmod.ConditionLogmel,
    cfg: dict,
    cfg_used: dict,
    negative_label_name: str,
    config: cnn_model.SearchConfiguration,
    epochs: int,
    selection_source: str,
    aggregated: pd.DataFrame,
    oof_metrics: dict,
    baseline_mean: dict,
    device: torch.device,
    num_workers: int,
    input_hashes: dict[str, str],
    logger,
) -> dict:
    spec = condition.spec
    tag = f"{spec.dataset}/{spec.condition}"
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    final_dir = cdir / "final"
    hyperparameters = {"config_index": config.index, "lr": config.lr, "dropout": config.dropout, "epochs": epochs}

    if (final_dir / "_SUCCESS").is_file():
        metadata = json.loads((final_dir / "model_metadata.json").read_text(encoding="utf-8"))
        stored = metadata.get("hyperparameters", {})
        if stored.get("config_index") == config.index and stored.get("epochs") == epochs:
            logger.info(f"{tag}: modelo final ya publicado, se reutiliza")
            return metadata
        raise RuntimeError(
            f"{tag}: final/ ya existe con hiperparametros {stored}, distintos de los recalculados "
            f"{hyperparameters}; no se sobrescribe"
        )

    staging = cdir / "final_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    segments = condition.segments
    logger.info(
        f"{tag}: entrenando modelo final (config {config.index}, lr={config.lr:g}, dropout={config.dropout:g}, "
        f"{epochs} epocas, {segments['patient_uid'].nunique()} pacientes)"
    )
    run = cnn_model.train_final_model(
        condition, cfg_used, config, epochs, device, num_workers,
        log=lambda message: logger.info(f"{tag} final: {message}"),
    )
    state = {k: v.detach().cpu().clone() for k, v in run.model.state_dict().items()}
    del run.model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    payload = cnn_model.checkpoint_payload(
        state, run.normalization, cfg, spec, negative_label_name, hyperparameters,
        extra={"input_hashes": dict(input_hashes)},
    )
    model_path = staging / "final_model.pt"
    cnn_model.save_checkpoint(model_path, payload)
    model_hash = art.sha256_file(model_path)
    (staging / "model_sha256.txt").write_text(model_hash + "\n", encoding="utf-8")
    _write_csv(run.history, staging / "final_history.csv")
    _write_json(run.normalization.to_dict(), staging / "normalization.json")

    metadata = {
        "model": dmod.model_architecture(cfg),
        "dataset": spec.dataset,
        "condition": spec.condition,
        "branch": spec.branch,
        "augment": spec.augment,
        "architecture": payload["architecture"],
        "hyperparameters": hyperparameters,
        "selection_source": selection_source,
        "aggregated_validation": aggregated.to_dict(orient="records"),
        "n_patients": int(segments["patient_uid"].nunique()),
        "n_recordings": int(segments["audio_id"].nunique()),
        "n_segments": int(len(segments)),
        "class_names": payload["class_names"],
        "label_mapping": payload["label_mapping"],
        "decision_threshold": payload["decision_threshold"],
        "normalization_weighted": run.normalization.weighted,
        "oof_metrics": oof_metrics,
        "baseline_mean": baseline_mean,
        "input_hashes": input_hashes,
        "model_sha256": model_hash,
        "training_seconds": run.seconds,
        "environment": cnn_model.torch_environment(device),
        "git": art.git_state(),
        "created_at_utc": art._now_iso(),
        "nota": "entrenado con todos los pacientes elegibles; sin metricas de test propias (las oficiales son OOF)",
    }
    _write_json(metadata, staging / "model_metadata.json")
    art.publish_fold(staging)
    logger.info(f"{tag}: modelo final publicado ({model_hash[:12]}...)")
    return metadata


def summarize_cnn_condition(
    run_root: Path,
    condition: dmod.ConditionLogmel,
    cfg: dict,
    cfg_used: dict,
    negative_label_name: str,
    fold_ids: list[int],
    fold_outputs: list[dict],
    device: torch.device,
    num_workers: int,
    input_hashes: dict[str, str],
    logger,
) -> dict:
    spec = condition.spec
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    threshold = float(cfg["evaluation"]["decision_threshold"])
    neg_key = negative_label_name.strip().lower()
    target_names = (negative_label_name, "COPD")

    oof_patients = pd.concat([fo["patient_predictions"] for fo in fold_outputs], ignore_index=True)
    metrics_by_fold = pd.DataFrame([fo["test_metrics"] for fo in fold_outputs]).sort_values("fold").reset_index(drop=True)
    search_summaries = pd.concat([fo["search_summary"] for fo in fold_outputs], ignore_index=True)
    baseline_metrics = pd.concat([fo["baseline_metrics"] for fo in fold_outputs], ignore_index=True)

    oof_patients = ev.attach_source_dataset(oof_patients, condition.segments)

    y_true = oof_patients["target_label"].to_numpy()
    y_score = oof_patients["score"].to_numpy()
    oof_metrics = ev.compute_patient_metrics(y_true, y_score, negative_label_name, threshold)

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
            y_true, y_score, fn,
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

    config, epochs, aggregated, selection_source = _final_selection(run_root, spec, fold_ids, search_summaries)

    summary_row = {
        "dataset": spec.dataset, "condition": spec.condition, "branch": spec.branch,
        "augment": spec.augment, "n_folds": len(fold_outputs),
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
        "final_config_index": config.index, "final_lr": config.lr,
        "final_dropout": config.dropout, "final_epochs": epochs,
    }

    classification_report = ev.classification_report_df(y_true, y_score, target_names, threshold)
    confusion_matrix = ev.confusion_matrix_df(y_true, y_score, target_names, threshold)

    cdir.mkdir(parents=True, exist_ok=True)
    _write_csv(oof_patients, cdir / "oof_patient_predictions.csv")
    _write_csv(metrics_by_fold, cdir / "metrics_by_fold.csv")
    _write_csv(baseline_metrics, cdir / "baseline_metrics.csv")
    _write_csv(search_summaries, cdir / "hyperparameter_search.csv")
    _write_csv(classification_report, cdir / "classification_report.csv")
    confusion_matrix.to_csv(cdir / "confusion_matrix.csv")
    _write_json(summary_row, cdir / "metrics_summary.json")
    if by_source is not None:
        for name, table in by_source.items():
            _write_csv(table, cdir / f"{name}.csv")

    prefix = run_root / "figures" / f"{spec.dataset}__{spec.condition}"
    art.plot_confusion_matrix(y_true, y_score, target_names, prefix.with_name(prefix.name + "__confusion_matrix"), cfg, threshold)
    art.plot_roc_curve(y_true, y_score, prefix.with_name(prefix.name + "__roc"), cfg)
    art.plot_pr_curve(y_true, y_score, prefix.with_name(prefix.name + "__pr"), cfg)
    art.plot_metrics_by_fold(metrics_by_fold, negative_label_name, prefix.with_name(prefix.name + "__metrics_by_fold"), cfg)
    if len(aggregated) > 1 and spec.hyperparameters_from is None:
        art.plot_parameter_heatmap(
            aggregated, "lr", "dropout", "val_balanced_accuracy",
            prefix.with_name(prefix.name + "__hyperparam_heatmap"), cfg,
            index_label="Learning rate", columns_label="Dropout",
        )

    _ensure_final_model(
        run_root, condition, cfg, cfg_used, negative_label_name, config, epochs, selection_source,
        aggregated, oof_metrics, baseline_mean, device, num_workers, input_hashes, logger,
    )
    logger.info(
        f"{spec.dataset}/{spec.condition}: OOF balanced_accuracy={oof_metrics['balanced_accuracy']:.3f} "
        f"auroc={oof_metrics['auroc']:.3f}"
    )

    return {
        "summary_row": summary_row,
        "oof_patients": oof_patients,
        "metrics_by_fold": metrics_by_fold,
        "baseline_metrics": baseline_metrics,
        "hyperparameter_search": search_summaries,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "by_source": by_source,
    }


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def dry_run(data_root: Path, cache_root: Path, cfg: dict, specs: list[dmod.ConditionSpec], device_arg: str) -> int:
    """Hashes, formas y folds (los mismos chequeos que la SVM) mas: Log-Mel
    identico al de la SVM, estado de la cache, clases por condicion, conteo de
    parametros y disponibilidad del dispositivo. No entrena ni extrae nada."""
    report = dry_run_check(data_root, cfg, specs)
    rows = report["checks"].to_dict(orient="records")
    verdict = {"ok": bool(report["ok"])}

    def add(dataset: str, check: str, passed: bool, detail: str) -> None:
        verdict["ok"] = verdict["ok"] and bool(passed)
        rows.append({"dataset": dataset, "check": check, "ok": bool(passed), "detail": detail})

    for check in config_consistency_checks(cfg):
        add("-", check["check"], check["ok"], check["detail"])

    for dataset, branch in sorted({(s.dataset, s.branch) for s in specs}):
        try:
            status, detail = dmod.logmel_cache_status(data_root, cache_root, dataset, branch, cfg)
            add(dataset, f"cache_logmel_{branch}", True, f"{status}: {detail}")
        except Exception as exc:  # noqa: BLE001
            add(dataset, f"cache_logmel_{branch}", False, str(exc))

    for spec in specs:
        try:
            segments = dmod.select_condition_segments(dmod.load_task_segments(data_root, spec.dataset), spec)
            table = sp.build_patient_table(segments)
            negative = cfg["datasets"][spec.dataset]["negative_label_name"]
            parts = []
            for label, name in ((1, "COPD"), (0, negative)):
                group = table.loc[table["target_label"] == label]
                parts.append(f"{name} {len(group)} ({int((~group['calibration_patient']).sum())} evaluables)")
            add(spec.dataset, f"clases[{spec.condition}]", table["target_label"].nunique() == 2, ", ".join(parts))
        except Exception as exc:  # noqa: BLE001
            add(spec.dataset, f"clases[{spec.condition}]", False, str(exc))

    parameters_check = f"parametros_{dmod.model_architecture(cfg)}"
    try:
        n_parameters = cnn_model.architecture_description(cfg)["n_parameters"]
        expected = cnn_model.expected_parameters(cfg)
        add("-", parameters_check, n_parameters == expected, f"{n_parameters} (esperado {expected})")
    except Exception as exc:  # noqa: BLE001
        add("-", parameters_check, False, str(exc))

    try:
        device = cnn_model.resolve_device(device_arg)
        add("-", "dispositivo", True, f"--device {device_arg} -> {cnn_model.device_description(device)}")
    except Exception as exc:  # noqa: BLE001
        add("-", "dispositivo", False, f"--device {device_arg}: {exc}")

    with pd.option_context("display.max_colwidth", 120, "display.width", 180):
        print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nveredicto: {'OK' if verdict['ok'] else 'REVISAR'}")
    return 0 if verdict["ok"] else 1


def _write_run_tables(run_root: Path, condition_results: list[dict], folds_by_population: dict) -> None:
    ok_results = [r for r in condition_results if r["ok"] and r["summary"] is not None]
    if ok_results:
        def tagged(key: str) -> pd.DataFrame:
            return pd.concat(
                [r["summary"][key].assign(dataset=r["spec"].dataset, condition=r["spec"].condition) for r in ok_results],
                ignore_index=True,
            )

        _write_csv(tagged("metrics_by_fold"), run_root / "metrics_by_fold.csv")
        _write_csv(tagged("hyperparameter_search"), run_root / "hyperparameter_search.csv")
        _write_csv(pd.DataFrame([r["summary"]["summary_row"] for r in ok_results]), run_root / "metrics_summary.csv")
        _write_csv(tagged("oof_patients"), run_root / "oof_patient_predictions.csv")
        _write_csv(tagged("baseline_metrics"), run_root / "baseline_metrics.csv")
        _write_csv(tagged("classification_report"), run_root / "classification_report.csv")
        confusion_frames = []
        for r in ok_results:
            cm = r["summary"]["confusion_matrix"].reset_index().rename(columns={"index": "real"})
            cm["dataset"], cm["condition"] = r["spec"].dataset, r["spec"].condition
            confusion_frames.append(cm)
        _write_csv(pd.concat(confusion_frames, ignore_index=True), run_root / "confusion_matrix.csv")

        by_source_keys = (
            "metrics_by_source", "classification_report_by_source",
            "confusion_matrix_by_source", "bootstrap_by_source",
        )
        for key in by_source_keys:
            frames = [
                r["summary"]["by_source"][key].assign(dataset=r["spec"].dataset, condition=r["spec"].condition)
                for r in ok_results
                if r["summary"].get("by_source")
            ]
            if frames:
                _write_csv(pd.concat(frames, ignore_index=True), run_root / f"{key}.csv")

    if folds_by_population:
        tables = []
        for (dataset, dn_reliable_only), folds in folds_by_population.items():
            table = folds.fold_table.copy()
            table["dataset"] = dataset
            table["dn_reliable_only"] = dn_reliable_only
            tables.append(table)
        _write_csv(pd.concat(tables, ignore_index=True), run_root / "folds.csv")


# ---------------------------------------------------------------------------
# Ejecucion completa de CNN/CRNN bajo oof_v1
# ---------------------------------------------------------------------------

def run_oof_cnn(args, cfg: dict) -> int:
    """Punto de entrada del protocolo oof_v1 para CNN/CRNN. ``cnn_experiment.run_cnn``
    despacha aqui cuando el TOML no declara ``[folds] patient_folds_csv``."""
    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = resolve_datasets(cfg, args.dataset)
    experiments = resolve_experiments(args.experiment, cfg)
    try:
        specs = order_specs(plan_conditions(cfg, datasets, experiments))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run(data_root, cache_root, cfg, specs, args.device)

    # "cnn" o "crnn": decide el directorio <runs-root>/<modelo>/, la huella y
    # la metadata. La CRNN reutiliza todo este flujo (ver crnn_experiment.py).
    model_name = dmod.model_architecture(cfg)
    blocking = [c for c in config_consistency_checks(cfg) if c["blocking"] and not c["ok"]]
    if blocking:
        for check in blocking:
            print(f"{check['check']}: {check['detail']}", file=sys.stderr)
        return 1

    cnn_model.configure_determinism(cfg)
    try:
        device = cnn_model.resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"--device {args.device}: {exc}", file=sys.stderr)
        return 1
    if device.type == "cuda":
        torch.cuda.set_device(device)

    architecture = cnn_model.architecture_description(cfg)
    expected_parameters = cnn_model.expected_parameters(cfg)
    if architecture["n_parameters"] != expected_parameters:
        print(
            f"la arquitectura tiene {architecture['n_parameters']} parametros; "
            f"{model_name}.toml espera {expected_parameters}",
            file=sys.stderr,
        )
        return 1

    # Con --resume, la huella se verifica ANTES de abrir run.log o escribir
    # cualquier archivo de la ejecucion existente. Si no coincide, se sale con
    # error y la ejecucion queda intacta (ver run_experiment.open_run).
    fingerprint = build_run_fingerprint(
        cfg, data_root, specs, args.dataset, args.experiment,
        sections=fingerprint_sections(cfg),
        extra={"model": model_name, "architecture": architecture, "smoke_test": bool(args.smoke_test)},
    )
    try:
        run_root, run_id = open_run(runs_root, model_name, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} model={model_name} dataset={args.dataset} experiment={args.experiment} "
        f"device={cnn_model.device_description(device)} num_workers={args.num_workers} "
        f"smoke_test={args.smoke_test} resume={bool(args.resume)} parametros={architecture['n_parameters']}"
    )
    if args.resume:
        logger.info("run_fingerprint verificado: datos, configuracion, arquitectura y modo coinciden")
        removed_staging = art.finalize_resume(run_root)
        if removed_staging:
            logger.info(f"staging abandonado eliminado: {[str(p) for p in removed_staging]}")

    art.write_environment(run_root)
    _append_torch_environment(run_root, device)
    art.write_resolved_config(run_root, cfg, vars(args))
    if not args.resume:
        write_run_fingerprint(run_root, fingerprint)

    cfg_used = smoke_test_config(cfg) if args.smoke_test else cfg
    logmel_cache: dict[tuple[str, str], tuple] = {}
    folds_by_population: dict[tuple[str, bool], sp.PatientFolds] = {}
    condition_results: list[dict] = []

    for spec in specs:
        tag = f"{spec.dataset}/{spec.condition}"
        logger.info(f"preparando {tag} (rama {spec.branch}, augment={spec.augment})")
        try:
            key = (spec.dataset, spec.branch)
            if key not in logmel_cache:
                started = time.perf_counter()
                logmel_cache[key] = dmod.extract_or_load_logmel(
                    data_root, cache_root, spec.dataset, spec.branch, cfg, force=args.force_features,
                    progress=lambda done, total, k=key: logger.info(f"log-mel {k[0]}/{k[1]}: {done}/{total}"),
                )
                logger.info(
                    f"log-mel {key[0]}/{key[1]} listo en {time.perf_counter() - started:.1f} s, "
                    f"forma {logmel_cache[key][0].shape}"
                )
            logmel, rows = logmel_cache[key]
            condition = dmod.build_condition_logmel(data_root, spec, logmel, rows)

            pop_key = (spec.dataset, spec.dn_reliable_only)
            new_folds = sp.build_patient_folds(
                condition.segments,
                n_splits=int(cfg["splits"]["n_splits"]),
                random_state=int(cfg["splits"]["random_state"]),
                stratify_by_dataset=bool(cfg["splits"].get("stratify_by_dataset", False)),
            )
            if pop_key in folds_by_population:
                if not sp.folds_are_identical(folds_by_population[pop_key], new_folds):
                    raise RuntimeError(f"{tag}: los folds no coinciden con los de otra condicion de la misma poblacion")
            else:
                folds_by_population[pop_key] = new_folds

            result = run_cnn_condition(
                run_root, condition, folds_by_population[pop_key], cfg, cfg_used,
                cfg["datasets"][spec.dataset]["negative_label_name"], device, args.num_workers,
                args.smoke_test, _condition_input_hashes(fingerprint, cache_root, spec), logger,
            )
        except Exception:  # noqa: BLE001
            logger.exception(f"{tag}: fallo no controlado durante la preparacion")
            result = {"spec": spec, "fold_status": {}, "ok": False, "summary": None}
        condition_results.append(result)

    threshold = float(cfg["evaluation"]["decision_threshold"])
    for dataset in datasets:
        for left, right, label in PAIRED_COMPARISONS.get(dataset, ()):
            left_result = _find_ok(condition_results, dataset, left)
            right_result = _find_ok(condition_results, dataset, right)
            if left_result and right_result:
                art.plot_denoising_comparison(
                    left_result["summary"]["oof_patients"], right_result["summary"]["oof_patients"],
                    run_root / "figures" / f"{dataset}__{left}_vs_{right}__{label}_comparison", cfg,
                    left_label=left, right_label=right,
                    title=f"Comparacion emparejada: {left} frente a {right}",
                    score_label="Probabilidad de COPD por paciente",
                    threshold=threshold, score_limits=(0.0, 1.0),
                )
                logger.info(f"{dataset}: comparacion emparejada {left} vs {right} generada")

    _write_run_tables(run_root, condition_results, folds_by_population)

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
