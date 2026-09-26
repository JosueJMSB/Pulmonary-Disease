"""SVM-RBF en el protocolo holdout + validacion cruzada v3
(``PLAN-EXPERIMENTO FINAL.md``).

``run_experiment.main`` despacha aqui cuando el TOML declara
``protocol = "holdout_cv_v3"``. Cada una de las 30 combinaciones (C, gamma) se
evalua en los 5 folds internos: escalado y SVM se ajustan SOLO con los cuatro
folds de train (con los mismos pesos por muestra de siempre), y la decision
``decision_function >= 0`` se agrega segmento -> grabacion -> paciente en el
fold de validation. Se elige UNA combinacion global (``holdout_cv``); no se
guarda ningun modelo ni se toca la prueba externa.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import artifacts as art
from . import data as dmod
from . import holdout_cv as hcv
from . import run_experiment as rexp
from .models import svm_rbf as svm_model

MODEL_NAME = "svm_rbf"


def _fit_and_score(train_X, train_y, train_weights, val_X, C, gamma, fixed_params) -> dict:
    """Una combinacion (C, gamma): ajuste en train y puntajes de validation.
    Funcion de nivel de modulo para que ``joblib`` la ejecute en procesos
    separados con ``--n-jobs``; un fallo se devuelve como dato, no se lanza,
    para que solo esa configuracion quede incompleta."""
    started = time.perf_counter()
    try:
        scaler, svm = svm_model.fit_scaler_and_svm(train_X, train_y, train_weights, C, gamma, fixed_params)
        scores = svm_model.decision_scores(scaler, svm, val_X)
        return {"ok": True, "scores": scores, "seconds": time.perf_counter() - started}
    except Exception as exc:  # noqa: BLE001 - se registra como fallo de la unidad
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__,
                "seconds": time.perf_counter() - started}


@dataclass
class SvmFoldContext:
    fold_id: int
    train_X: np.ndarray
    train_y: np.ndarray
    train_weights: np.ndarray
    val_X: np.ndarray
    val_seg: pd.DataFrame


def make_prepare_fold(data_root, cache_root, spec, cfg, split, force_features):
    """Carga las caracteristicas del fold ``cv/fold_XX`` de la condicion y
    separa train/validation por pacientes, verificando antes que el fold
    contiene exactamente a los pacientes del split y a ninguno de la prueba
    externa."""
    def prepare_fold(fold_id: int, train_patients: list[str], val_patients: list[str]) -> SvmFoldContext:
        fold_ref = hcv.cv_fold_ref(fold_id)
        # Antes de extraer caracteristicas o escribir cache: si el fold contiene a
        # alguien de la prueba externa (o no coincide con el split), se aborta.
        hcv.verify_fold_segments_against_split(
            dmod.load_task_segments(data_root, spec.dataset, fold_ref), split, fold_id,
        )
        condition = dmod.build_condition_data(
            data_root, cache_root, spec, cfg, force_features=force_features, fold_id=fold_ref,
        )
        train_seg, train_X = dmod.select_rows_by_patients(condition.segments, condition.X, train_patients)
        val_seg, val_X = dmod.select_rows_by_patients(condition.segments, condition.X, val_patients)
        for name, part in (("train", train_seg), ("validation", val_seg)):
            if part.empty:
                raise ValueError(f"fold {fold_id}: el conjunto {name} quedo vacio")
        return SvmFoldContext(
            fold_id=fold_id, train_X=train_X, train_y=train_seg["target_label"].to_numpy(),
            train_weights=dmod.compute_sample_weights(train_seg), val_X=val_X, val_seg=val_seg,
        )

    return prepare_fold


def make_evaluate_fold(cfg: dict, n_jobs: int):
    fixed_params = svm_model.svm_fixed_params(cfg)

    def evaluate_fold(context: SvmFoldContext, pending: list[dict]):
        # Con n_jobs <= 1 corre en este proceso; con mas, reparte las
        # configuraciones pendientes del fold entre procesos.
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_fit_and_score)(
                context.train_X, context.train_y, context.train_weights, context.val_X,
                config["C"], config["gamma"], fixed_params,
            )
            for config in pending
        )
        for config, result in zip(pending, results):
            if not result["ok"]:
                yield config, hcv.UnitFailure(error=result["error"], error_type=result["error_type"])
                continue
            try:
                output = hcv.UnitOutput(
                    val_scores=hcv.build_val_scores(context.val_seg, result["scores"]), seconds=result["seconds"],
                )
            except Exception as exc:  # noqa: BLE001
                yield config, hcv.UnitFailure(error=str(exc), error_type=type(exc).__name__)
                continue
            yield config, output

    return evaluate_fold


def run_holdout_svm(args, cfg: dict) -> int:
    """Punto de entrada de la SVM en holdout_cv_v3."""
    hcv.validate_holdout_config(cfg)  # RuntimeError explicito antes de tocar datos
    split_csv, split_manifest = hcv.holdout_split_paths(cfg)

    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = rexp.resolve_datasets(cfg, args.dataset)
    experiments = rexp.resolve_experiments(args.experiment, cfg)
    specs = rexp.plan_conditions(cfg, datasets, experiments)
    for spec in specs:
        if spec.augment or spec.hyperparameters_from:
            print(f"{spec.dataset}/{spec.condition}: la SVM no admite augmentation ni hiperparametros reutilizados",
                  file=sys.stderr)
            return 2

    candidates = hcv.svm_candidates(cfg)

    if args.dry_run:
        report = hcv.dry_run_check_holdout(data_root, cfg, specs, split_csv, split_manifest)
        rows = report["checks"].to_dict(orient="records")
        rows.append({
            "dataset": "-", "check": "candidatos_svm", "ok": len(candidates) > 0,
            "detail": f"{len(candidates)} configuraciones (C x gamma), {hcv.N_INNER_FOLDS} folds cada una",
        })
        ok = bool(report["ok"]) and len(candidates) > 0
        with pd.option_context("display.max_colwidth", 120, "display.width", 180):
            print(pd.DataFrame(rows).to_string(index=False))
        print(f"\nveredicto: {'OK' if ok else 'REVISAR'}")
        return 0 if ok else 1

    try:
        splits = hcv.load_splits(cfg, {s.dataset for s in specs})
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    smoke = bool(args.smoke_test)
    cfg_used = rexp._smoke_test_config(cfg) if smoke else cfg
    configs = hcv.svm_candidates(cfg_used)
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))

    try:
        fingerprint = hcv.build_holdout_fingerprint(
            cfg, data_root, specs, args.dataset, args.experiment, splits, hcv.SVM_FINGERPRINT_SECTIONS,
            extra={"model": MODEL_NAME, "smoke_test": smoke}, smoke_test=smoke,
        )
        run_root, run_id = rexp.open_run(runs_root, MODEL_NAME, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} protocolo={hcv.PROTOCOL_NAME} model={MODEL_NAME} dataset={args.dataset} "
        f"experiment={args.experiment} n_jobs={args.n_jobs} smoke_test={smoke} resume={bool(args.resume)}"
    )
    if args.resume:
        logger.info("run_fingerprint verificado: datos, split, configuracion y modo coinciden con la ejecucion original")
        removed = art.finalize_resume(run_root) + hcv.cleanup_abandoned_unit_staging(run_root)
        if removed:
            logger.info(f"staging abandonado eliminado: {[str(p) for p in removed]}")

    art.write_environment(run_root)
    art.write_resolved_config(run_root, cfg, vars(args))
    if not args.resume:
        rexp.write_run_fingerprint(run_root, fingerprint)

    results: list[dict] = []
    for spec in specs:
        tag = f"{spec.dataset}/{spec.condition}"
        logger.info(f"preparando {tag} (rama {spec.branch}, {len(configs)} configuraciones x folds internos)")
        try:
            split = splits[spec.dataset]
            fold_ids = [0] if smoke else list(range(split.n_splits))
            negative_label_name = cfg["datasets"][spec.dataset]["negative_label_name"]
            hcv.run_condition_cv(
                run_root=run_root, spec=spec, model_name=MODEL_NAME, split=split, configs=configs,
                hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, negative_label_name=negative_label_name,
                threshold=threshold, fold_ids=fold_ids,
                prepare_fold=make_prepare_fold(data_root, cache_root, spec, cfg_used, split, args.force_features),
                evaluate_fold=make_evaluate_fold(cfg_used, args.n_jobs), release_fold=None, logger=logger,
            )
            result = hcv.summarize_condition(
                run_root=run_root, spec=spec, cfg_used=cfg_used, model_name=MODEL_NAME, configs=configs,
                hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, fold_ids=fold_ids,
                negative_label_name=negative_label_name, threshold=threshold,
                hyperparameter_source="search", reused_config_index=None,
                provenance=hcv.condition_provenance(
                    cfg, fingerprint, spec, run_id, smoke, bool(args.resume), split, fold_ids,
                ),
                logger=logger,
            )
        except Exception:  # noqa: BLE001 - se registra, nunca se omite en silencio
            logger.exception(f"{tag}: fallo no controlado")
            result = {"spec": spec, "status": "FAILED", "ok": False, "summary_row": None}
        results.append(result)

    hcv.write_run_tables(run_root, results)
    hcv.assert_no_final_model(run_root, results)

    final_status = hcv.final_run_status(results)
    art.write_status(run_root, final_status, {
        "protocol": hcv.PROTOCOL_NAME,
        "conditions": [
            {"dataset": r["spec"].dataset, "condition": r["spec"].condition, "status": r["status"]}
            for r in results
        ],
    })
    logger.info(f"ejecucion {run_id} finalizada con estado {final_status}")
    print(f"run_id={run_id} estado={final_status} -> {run_root}")
    return 0 if final_status == art.STATUS_COMPLETED else 1
