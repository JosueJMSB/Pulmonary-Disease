"""CNN y CRNN en el protocolo holdout + validacion cruzada v3
(``PLAN-EXPERIMENTO FINAL.md``).

``run_experiment.main`` despacha aqui cuando el TOML declara
``protocol = "holdout_cv_v3"`` y ``--model`` es ``cnn`` o ``crnn``. Ambas redes
comparten este modulo (como en los protocolos anteriores): solo cambia la
arquitectura (``models/cnn.py::build_model``) y el directorio de resultados.

Cada una de las 20 configuraciones (muestreo determinista de
learning rate x dropout x weight decay x batch size, ver
``holdout_cv.nn_candidates``) se entrena DESDE CERO en cuatro folds y se valida
en el quinto, en los cinco folds internos; el optimizador (AdamW), el
scheduler, el recorte de gradiente y AMP son fijos. Por unidad se guarda la
mejor epoca segun validation (criterio de ``models/cnn.py::selection_key``) y
sus predicciones. Se elige UNA configuracion global por pipeline y se registra
la mediana de sus mejores epocas, que usara despues el reentrenamiento del
modelo definitivo -que NO ocurre en esta entrega-.

``no_dn_aug`` no repite la busqueda: reutiliza la configuracion global
elegida para ``no_dn`` (mismo modelo y dataset), vuelve a correr los cinco
folds con SpecAugment -solo sobre los batches de entrenamiento- y guarda sus
propios resultados y mejores epocas.

Nota metodologica: como en los protocolos anteriores, la mejor epoca de cada
unidad se elige con el mismo fold de validation sobre el que se reporta, asi
que la metrica de validation de las redes es ligeramente optimista frente a la
de la SVM (que no tiene epocas). Es el procedimiento que fija el plan; conviene
declararlo al comparar modelos.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
import time
from dataclasses import dataclass

import pandas as pd
import torch

from . import artifacts as art
from . import cnn_experiment as cexp
from . import data as dmod
from . import holdout_cv as hcv
from . import run_experiment as rexp
from . import splits as sp
from .models import cnn as cnn_model

# Secciones que un TOML de CRNN v3 debe copiar sin cambios de su CNN de
# referencia, ademas de las del protocolo comun (cnn_experiment).
HOLDOUT_EXTRA_SHARED_SECTIONS = ("protocol", "holdout", "outer_test", "final_model")


@dataclass(frozen=True)
class HoldoutConfiguration(cnn_model.SearchConfiguration):
    """``SearchConfiguration`` (index, lr, dropout) + los dos hiperparametros
    que en v3 tambien se buscan. ``train_with_validation`` solo lee
    ``index``/``lr``/``dropout``; el resto llega por ``TrainingSettings``."""

    weight_decay: float
    batch_size: int


def to_configuration(candidate: dict) -> HoldoutConfiguration:
    return HoldoutConfiguration(
        index=int(candidate["config_index"]), lr=float(candidate["lr"]), dropout=float(candidate["dropout"]),
        weight_decay=float(candidate["weight_decay"]), batch_size=int(candidate["batch_size"]),
    )


def settings_for(base: cnn_model.TrainingSettings, config: HoldoutConfiguration) -> cnn_model.TrainingSettings:
    """Ajustes de entrenamiento de UNA configuracion: los fijos del TOML con su
    batch size y su weight decay."""
    return dataclasses.replace(base, batch_size=config.batch_size, weight_decay=config.weight_decay)


def smoke_test_config(cfg: dict) -> dict:
    """Dos epocas como maximo; el fold (0) y la unica configuracion los decide
    el lanzador."""
    smoke = copy.deepcopy(cfg)
    smoke["training"]["max_epochs"] = min(2, int(cfg["training"]["max_epochs"]))
    smoke["training"]["min_epochs"] = min(int(cfg["training"]["min_epochs"]), smoke["training"]["max_epochs"])
    return smoke


def holdout_consistency_checks(cfg: dict) -> list[dict]:
    """Los chequeos de ``cnn_experiment.config_consistency_checks`` mas las
    secciones propias del protocolo v3 (una CRNN no arranca si difiere de su
    CNN de referencia en alguna de ellas)."""
    checks = cexp.config_consistency_checks(cfg)
    if dmod.model_architecture(cfg) == "crnn":
        reference_path = dmod.reference_cnn_config_path(cfg)
        cnn_cfg = dmod.load_config(reference_path)
        for section in HOLDOUT_EXTRA_SHARED_SECTIONS:
            same = cfg.get(section) == cnn_cfg.get(section)
            checks.append({
                "check": f"{section}_igual_cnn", "ok": same, "blocking": True,
                "detail": f"identico a {reference_path.name}" if same
                else f"difiere de {reference_path.name}: se pierde la comparabilidad con la CNN",
            })
    return checks


# ---------------------------------------------------------------------------
# Un fold: datos, y una configuracion tras otra
# ---------------------------------------------------------------------------

@dataclass
class NnFoldContext:
    fold_id: int
    condition: dmod.ConditionLogmel
    logmel: object
    train_seg: pd.DataFrame
    val_seg: pd.DataFrame


def make_prepare_fold(data_root, cache_root, spec, cfg, split, force_features, logger):
    tag = f"{spec.dataset}/{spec.condition}"

    def prepare_fold(fold_id: int, train_patients: list[str], val_patients: list[str]) -> NnFoldContext:
        fold_ref = hcv.cv_fold_ref(fold_id)
        # Antes de calcular el Log-Mel o escribir cache: si el fold contiene a
        # alguien de la prueba externa (o no coincide con el split), se aborta.
        hcv.verify_fold_segments_against_split(
            dmod.load_task_segments(data_root, spec.dataset, fold_ref), split, fold_id,
        )
        started = time.perf_counter()
        logmel, rows = dmod.extract_or_load_logmel(
            data_root, cache_root, spec.dataset, spec.branch, cfg, force=force_features, fold_id=fold_ref,
            progress=lambda done, total: logger.info(f"log-mel {spec.dataset}/{fold_ref}/{spec.branch}: {done}/{total}"),
        )
        logger.info(
            f"{tag} fold {fold_id}: log-mel {spec.branch} listo en {time.perf_counter() - started:.1f} s, "
            f"forma {logmel.shape}"
        )
        condition = dmod.build_condition_logmel(data_root, spec, logmel, rows, fold_id=fold_ref)

        train_seg = sp.filter_segments_by_patients(condition.segments, train_patients)
        val_seg = sp.filter_segments_by_patients(condition.segments, val_patients)
        for name, part in (("train", train_seg), ("validation", val_seg)):
            if part.empty:
                raise ValueError(f"fold {fold_id}: el conjunto {name} quedo vacio")
        if set(train_seg["patient_uid"]) & set(val_seg["patient_uid"]):
            raise RuntimeError(f"fold {fold_id}: hay pacientes compartidos entre train y validation")
        return NnFoldContext(fold_id=fold_id, condition=condition, logmel=logmel, train_seg=train_seg, val_seg=val_seg)

    return prepare_fold


def make_evaluate_fold(spec, cfg, base_settings, device, negative_label_name, logger):
    tag = f"{spec.dataset}/{spec.condition}"
    augment_cfg = cfg["augmentation"] if spec.augment else None

    def evaluate_fold(context: NnFoldContext, pending: list[dict]):
        for candidate in pending:
            config = to_configuration(candidate)
            settings = settings_for(base_settings, config)
            outcome = None
            try:
                run = cnn_model.train_with_validation(
                    context.logmel, context.train_seg, context.val_seg, cfg, config, settings, device,
                    negative_label_name,
                    # La condicion NO entra en la semilla: no_dn, dn y no_dn_aug parten de la misma
                    # inicializacion en cada (fold, configuracion).
                    seed_parts=(spec.dataset, context.fold_id, config.index, "cv"),
                    augment_cfg=augment_cfg,
                    log=lambda message, f=context.fold_id: logger.info(f"{tag} fold {f}: {message}"),
                )
                if run.best_logits is None:
                    raise RuntimeError("el entrenamiento no produjo logits de validation")
                outcome = hcv.UnitOutput(
                    val_scores=hcv.build_val_scores(context.val_seg, cnn_model.sigmoid(run.best_logits)),
                    best_epoch=int(run.best_epoch), epochs_run=int(run.epochs_run),
                    stopped_early=bool(run.stopped_early),
                    # history.csv autosuficiente: los cuatro hiperparametros buscados (lr y dropout ya vienen).
                    history=run.history.assign(
                        fold=context.fold_id, weight_decay=config.weight_decay, batch_size=config.batch_size,
                    ),
                    seconds=float(run.seconds),
                )
            except torch.cuda.OutOfMemoryError as exc:
                outcome = hcv.UnitFailure(
                    error="memoria de GPU insuficiente. No se reduce batch_size automaticamente porque cambiaria "
                          f"el experimento: libere memoria o use otra GPU (--device cuda:N) y continue con --resume. {exc}",
                    error_type="OutOfMemoryError",
                )
            except Exception as exc:  # noqa: BLE001 - se registra como fallo de la unidad
                outcome = hcv.UnitFailure(error=str(exc), error_type=type(exc).__name__)
            if isinstance(outcome, hcv.UnitFailure):
                cnn_model._release(device)
            yield candidate, outcome

    return evaluate_fold


def make_release_fold(device):
    def release_fold(context: NnFoldContext) -> None:
        context.logmel = None
        context.condition = None
        cnn_model._release(device)

    return release_fold


def load_reused_candidate(run_root, spec: dmod.ConditionSpec, candidates: list[dict]) -> dict:
    """La configuracion global ya elegida para la condicion base (``no_dn``),
    tomada de SU ``best_hyperparameters.json``. Se niega si la base no termino
    completa: reutilizar una configuracion elegida con folds faltantes
    mezclaria un criterio parcial."""
    source_dir = art.condition_dir(run_root, spec.dataset, spec.hyperparameters_from)
    best_path = source_dir / "best_hyperparameters.json"
    status = art.read_status(source_dir).get("status")
    if not best_path.is_file() or status != "COMPLETED":
        raise RuntimeError(
            f"{spec.condition} reutiliza la configuracion global de {spec.hyperparameters_from}, pero esa "
            f"condicion no esta COMPLETED (estado={status!r}, best_hyperparameters.json existe={best_path.is_file()})"
        )
    best = json.loads(best_path.read_text(encoding="utf-8"))
    index = int(best["config_index"])
    candidate = next((c for c in candidates if int(c["config_index"]) == index), None)
    if candidate is None:
        raise RuntimeError(f"la configuracion {index} de {spec.hyperparameters_from} no esta en la lista de candidatos")
    for key in hcv.NN_HYPERPARAMETER_KEYS:
        if float(best["hyperparameters"][key]) != float(candidate[key]):
            raise RuntimeError(
                f"los hiperparametros guardados de {spec.hyperparameters_from} no coinciden con el candidato "
                f"{index} ({key}: {best['hyperparameters'][key]} vs {candidate[key]})"
            )
    return candidate


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def dry_run_holdout(
    data_root, cache_root, cfg: dict, specs: list[dmod.ConditionSpec], device_arg: str, split_csv, split_manifest,
) -> int:
    report = hcv.dry_run_check_holdout(data_root, cfg, specs, split_csv, split_manifest)
    rows = report["checks"].to_dict(orient="records")
    verdict = {"ok": bool(report["ok"])}

    def add(dataset: str, check: str, passed: bool, detail: str) -> None:
        verdict["ok"] = verdict["ok"] and bool(passed)
        rows.append({"dataset": dataset, "check": check, "ok": bool(passed), "detail": detail})

    for check in holdout_consistency_checks(cfg):
        add("-", check["check"], check["ok"], check["detail"])

    try:
        candidates = hcv.nn_candidates(cfg)
        combinations = {tuple(c[k] for k in hcv.NN_HYPERPARAMETER_KEYS) for c in candidates}
        add("-", "candidatos_unicos", len(combinations) == len(candidates) == int(cfg["search"]["n_configurations"]),
            f"{len(combinations)} combinaciones unicas de {int(cfg['search']['n_configurations'])} pedidas")
    except Exception as exc:  # noqa: BLE001
        add("-", "candidatos_unicos", False, str(exc))

    splits = {}
    for dataset in sorted({s.dataset for s in specs}):
        try:
            splits[dataset] = sp.load_holdout_split(split_csv, split_manifest, dataset)
        except Exception:  # noqa: BLE001 - ya reportado por dry_run_check_holdout
            continue

    for dataset, branch in sorted({(s.dataset, s.branch) for s in specs}):
        split = splits.get(dataset)
        if split is None:
            continue
        for fold_id in range(split.n_splits):
            tag = f"cache_logmel_{branch}[{hcv.cv_fold_ref(fold_id)}]"
            try:
                status, detail = dmod.logmel_cache_status(
                    data_root, cache_root, dataset, branch, cfg, fold_id=hcv.cv_fold_ref(fold_id),
                )
                add(dataset, tag, True, f"{status}: {detail}")
            except Exception as exc:  # noqa: BLE001
                add(dataset, tag, False, str(exc))

    for spec in specs:
        split = splits.get(spec.dataset)
        if split is None:
            continue
        for fold_id in range(split.n_splits):
            tag = f"clases[{spec.condition}][{hcv.cv_fold_ref(fold_id)}]"
            try:
                segments = dmod.select_condition_segments(
                    dmod.load_task_segments(data_root, spec.dataset, hcv.cv_fold_ref(fold_id)), spec,
                )
                negative = cfg["datasets"][spec.dataset]["negative_label_name"]
                parts = [
                    f"{name} {int((segments['target_label'] == label).sum())} segmentos"
                    for label, name in ((1, "COPD"), (0, negative))
                ]
                add(spec.dataset, tag, segments["target_label"].nunique() == 2, ", ".join(parts))
            except Exception as exc:  # noqa: BLE001
                add(spec.dataset, tag, False, str(exc))

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


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run_holdout_cnn(args, cfg: dict) -> int:
    """CNN o CRNN en holdout_cv_v3 (segun ``[model] architecture``)."""
    hcv.validate_holdout_config(cfg)  # RuntimeError explicito antes de tocar datos
    model_name = dmod.model_architecture(cfg)
    if model_name == "crnn" and args.force_features:
        print(
            "--force-features no se admite con --model crnn: la cache Log-Mel es compartida con la CNN. "
            "Si hace falta regenerarla, hagalo con --model cnn.",
            file=sys.stderr,
        )
        return 2
    split_csv, split_manifest = hcv.holdout_split_paths(cfg)

    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = rexp.resolve_datasets(cfg, args.dataset)
    experiments = rexp.resolve_experiments(args.experiment, cfg)
    try:
        specs = cexp.order_specs(rexp.plan_conditions(cfg, datasets, experiments))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        return dry_run_holdout(data_root, cache_root, cfg, specs, args.device, split_csv, split_manifest)

    blocking = [c for c in holdout_consistency_checks(cfg) if c["blocking"] and not c["ok"]]
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

    try:
        splits = hcv.load_splits(cfg, {s.dataset for s in specs})
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    smoke = bool(args.smoke_test)
    cfg_used = smoke_test_config(cfg) if smoke else cfg
    all_candidates = hcv.nn_candidates(cfg)
    search_candidates = all_candidates[:1] if smoke else all_candidates
    threshold = float(cfg["evaluation"]["decision_threshold"])
    base_settings = cnn_model.TrainingSettings.from_config(cfg_used, args.num_workers)

    sections = hcv.NN_FINGERPRINT_SECTIONS + (("cnn",) if model_name == "cnn" else ("model", "crnn"))
    try:
        fingerprint = hcv.build_holdout_fingerprint(
            cfg, data_root, specs, args.dataset, args.experiment, splits, sections,
            extra={"model": model_name, "architecture": architecture, "smoke_test": smoke}, smoke_test=smoke,
        )
        run_root, run_id = rexp.open_run(runs_root, model_name, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} protocolo={hcv.PROTOCOL_NAME} model={model_name} dataset={args.dataset} "
        f"experiment={args.experiment} device={cnn_model.device_description(device)} num_workers={args.num_workers} "
        f"smoke_test={smoke} resume={bool(args.resume)} parametros={architecture['n_parameters']}"
    )
    if args.resume:
        logger.info("run_fingerprint verificado: datos, split, configuracion, arquitectura y modo coinciden")
        removed = art.finalize_resume(run_root) + hcv.cleanup_abandoned_unit_staging(run_root)
        if removed:
            logger.info(f"staging abandonado eliminado: {[str(p) for p in removed]}")

    art.write_environment(run_root)
    cexp._append_torch_environment(run_root, device)
    art.write_resolved_config(run_root, cfg, vars(args))
    if not args.resume:
        rexp.write_run_fingerprint(run_root, fingerprint)

    results: list[dict] = []
    for spec in specs:
        tag = f"{spec.dataset}/{spec.condition}"
        logger.info(f"preparando {tag} (rama {spec.branch}, augment={spec.augment}, protocolo {hcv.PROTOCOL_NAME})")
        try:
            split = splits[spec.dataset]
            fold_ids = [0] if smoke else list(range(split.n_splits))
            negative_label_name = cfg["datasets"][spec.dataset]["negative_label_name"]

            if spec.hyperparameters_from is None:
                configs, source, reused_index = search_candidates, "search", None
            else:
                reused = load_reused_candidate(run_root, spec, all_candidates)
                configs, source, reused_index = [reused], f"reused_from:{spec.hyperparameters_from}", int(reused["config_index"])

            hcv.run_condition_cv(
                run_root=run_root, spec=spec, model_name=model_name, split=split, configs=configs,
                hyperparameter_keys=hcv.NN_HYPERPARAMETER_KEYS, negative_label_name=negative_label_name,
                threshold=threshold, fold_ids=fold_ids,
                prepare_fold=make_prepare_fold(data_root, cache_root, spec, cfg_used, split, args.force_features, logger),
                evaluate_fold=make_evaluate_fold(spec, cfg_used, base_settings, device, negative_label_name, logger),
                release_fold=make_release_fold(device), logger=logger,
            )
            result = hcv.summarize_condition(
                run_root=run_root, spec=spec, cfg_used=cfg_used, model_name=model_name, configs=configs,
                hyperparameter_keys=hcv.NN_HYPERPARAMETER_KEYS, fold_ids=fold_ids,
                negative_label_name=negative_label_name, threshold=threshold,
                hyperparameter_source=source, reused_config_index=reused_index,
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
