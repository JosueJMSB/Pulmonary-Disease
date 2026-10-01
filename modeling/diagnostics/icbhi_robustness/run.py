"""Ejecuta los cuatro cofinalistas de ICBHI en folds internos repetidos."""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from ... import artifacts as art
from ... import cnn_experiment as cexp
from ... import data as dmod
from ... import evaluation as ev
from ... import holdout_cnn as hcnn
from ... import holdout_cv as hcv
from ... import splits as sp
from ...models import cnn as cnn_model

from .baselines import (
    apply_permuted_labels,
    permutation_map,
    run_device_baseline,
)
from .common import (
    DATASET,
    HYPERPARAMETER_KEYS,
    load_architecture_config,
    load_protocol,
    resolve_runtime_path,
    seed_tag,
    validate_source_selection,
    write_json,
)
from .reporting import (
    paired_comparisons,
    robustness_summary,
    write_pipeline_outputs,
)

RUNS_DIRNAME = "icbhi_robustness"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Valida configuraciones congeladas de cuatro cofinalistas ICBHI "
            "sobre repartos repetidos de development. El test externo permanece bloqueado."
        )
    )
    parser.add_argument("--protocol-config", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--source-runs-root",
        type=Path,
        default=None,
        help="Raiz que contiene cnn/<run_id> y crnn/<run_id> del holdout-v3.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", nargs="?", const="latest", default=None)
    parser.add_argument(
        "--only-seed",
        type=int,
        default=None,
        help="Ejecuta una sola repeticion preespecificada (util para smoke operativo del servidor).",
    )
    parser.add_argument("--skip-permutation-control", action="store_true")
    return parser.parse_args(argv)


def split_paths(workspace_root: Path, seed: int) -> tuple[Path, Path]:
    root = workspace_root / "splits" / seed_tag(seed)
    return root / "holdout_splits.csv", root / "holdout_splits_manifest.json"


def selected_seeds(protocol, only_seed: int | None) -> tuple[int, ...]:
    if only_seed is None:
        return protocol.seeds
    if int(only_seed) not in protocol.seeds:
        raise ValueError(
            f"--only-seed {only_seed} no esta preespecificada en {list(protocol.seeds)}"
        )
    return (int(only_seed),)


def validate_fold_data(data_root: Path, split: sp.HoldoutSplit, fold_id: int) -> dict:
    fold_ref = hcv.cv_fold_ref(fold_id)
    segments = dmod.load_task_segments(data_root, DATASET, fold_ref)
    hcv.verify_fold_segments_against_split(segments, split, fold_id)
    manifest_path = dmod.dataset_root(data_root, DATASET, fold_ref) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("verdict") != "PASS":
        raise RuntimeError(f"{manifest_path}: verdict no es PASS")
    if manifest.get("outer_test", {}).get("excluded") is not True:
        raise RuntimeError(f"{manifest_path}: no demuestra exclusion del test externo")
    for name in ("segments_no_dn.npy", "segments_dn.npy"):
        path = dmod.dataset_root(data_root, DATASET, fold_ref) / name
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "fold": fold_id,
        "n_patients": int(segments["patient_uid"].nunique()),
        "n_segments": int(len(segments)),
        "manifest": str(manifest_path),
        "manifest_sha256": art.sha256_file(manifest_path),
    }


def dry_run_checks(
    protocol,
    seeds: tuple[int, ...],
    workspace_root: Path,
    data_root: Path,
    source_runs_root: Path,
    device_arg: str,
) -> tuple[pd.DataFrame, dict]:
    rows = []
    provenance = {}

    def add(scope: str, check: str, ok: bool, detail: str) -> None:
        rows.append({"scope": scope, "check": check, "ok": bool(ok), "detail": detail})

    for pipeline in protocol.pipelines:
        try:
            provenance[pipeline.id] = validate_source_selection(
                pipeline, source_runs_root
            )
            cfg = load_architecture_config(protocol, pipeline)
            config_path = protocol.architecture_config_path(pipeline.architecture)
            provenance[pipeline.id].update(
                {
                    "architecture_config_path": str(config_path),
                    "architecture_config_file_sha256": art.sha256_file(config_path),
                    "architecture_config_fingerprint": pipeline.source_config_sha256,
                }
            )
            architecture = cnn_model.architecture_description(cfg)
            expected = cnn_model.expected_parameters(cfg)
            if architecture["n_parameters"] != expected:
                raise RuntimeError(
                    f"{architecture['n_parameters']} parametros, se esperaban {expected}"
                )
            add(
                pipeline.id,
                "seleccion_congelada",
                True,
                (
                    f"config={pipeline.source_config_index}, "
                    f"run_id={pipeline.source_run_id}, parametros={expected}"
                ),
            )
        except Exception as exc:  # noqa: BLE001
            add(pipeline.id, "seleccion_congelada", False, str(exc))

    for seed in seeds:
        tag = seed_tag(seed)
        try:
            csv_path, manifest_path = split_paths(workspace_root, seed)
            split = sp.load_holdout_split(csv_path, manifest_path, DATASET)
            original_csv, original_manifest = protocol.original_split_paths
            original = sp.load_holdout_split(original_csv, original_manifest, DATASET)
            if split.blocked_test_patients() != original.blocked_test_patients():
                raise RuntimeError("los pacientes del test externo no coinciden con el split original")
            add(
                tag,
                "split",
                True,
                f"{len(split.development_patients())} development, {len(split.blocked_test_patients())} test bloqueados",
            )
            seed_data_root = data_root / tag
            for fold_id in range(protocol.n_splits):
                detail = validate_fold_data(seed_data_root, split, fold_id)
                add(
                    tag,
                    f"fold_{fold_id:02d}",
                    True,
                    f"{detail['n_patients']} pacientes, {detail['n_segments']} segmentos",
                )
        except Exception as exc:  # noqa: BLE001
            add(tag, "split_o_datos", False, str(exc))

    try:
        device = cnn_model.resolve_device(device_arg)
        add("-", "dispositivo", True, cnn_model.device_description(device))
    except Exception as exc:  # noqa: BLE001
        add("-", "dispositivo", False, str(exc))
    report = pd.DataFrame(rows)
    return report, provenance


def build_fingerprint(
    protocol,
    seeds: tuple[int, ...],
    workspace_root: Path,
    data_root: Path,
    source_provenance: dict,
) -> dict:
    inputs = {
        "protocol.toml": art.sha256_file(protocol.path),
        **{
            f"source_selection/{pipeline_id}": row["sha256"]
            for pipeline_id, row in source_provenance.items()
        },
        **{
            f"architecture_config/{pipeline_id}": row[
                "architecture_config_file_sha256"
            ]
            for pipeline_id, row in source_provenance.items()
        },
    }
    for seed in seeds:
        csv_path, manifest_path = split_paths(workspace_root, seed)
        inputs[f"{seed_tag(seed)}/split_csv"] = art.sha256_file(csv_path)
        inputs[f"{seed_tag(seed)}/split_manifest"] = art.sha256_file(manifest_path)
        for fold_id in range(protocol.n_splits):
            manifest = (
                dmod.dataset_root(
                    data_root / seed_tag(seed), DATASET, hcv.cv_fold_ref(fold_id)
                )
                / "manifest.json"
            )
            inputs[f"{seed_tag(seed)}/fold_{fold_id:02d}/manifest"] = art.sha256_file(
                manifest
            )
    code = hcv.code_fingerprint()
    payload = json.dumps(
        {"inputs": inputs, "code_sha256": code["code_sha256"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    import hashlib

    return {
        "protocol": protocol.raw["protocol"],
        "dataset": DATASET,
        "seeds": list(seeds),
        "outer_test_accessed": False,
        "inputs": inputs,
        "code_sha256": code["code_sha256"],
        "fingerprint_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def open_run(runs_root: Path, resume: str | None, fingerprint: dict) -> tuple[Path, str]:
    parent = runs_root / RUNS_DIRNAME
    if resume:
        if resume == "latest":
            candidates = sorted(
                path for path in parent.iterdir()
                if path.is_dir() and (path / "fingerprint.json").is_file()
            ) if parent.is_dir() else []
            if not candidates:
                raise FileNotFoundError(f"no hay ejecuciones para reanudar en {parent}")
            run_root = candidates[-1]
        else:
            run_root = parent / resume
        stored_path = run_root / "fingerprint.json"
        if not stored_path.is_file():
            raise FileNotFoundError(f"no existe {stored_path}")
        stored = json.loads(stored_path.read_text(encoding="utf-8"))
        if stored.get("fingerprint_sha256") != fingerprint["fingerprint_sha256"]:
            raise RuntimeError(
                "--resume rechazado: cambiaron codigo, splits, datos o selecciones congeladas"
            )
        art.write_status(
            run_root,
            art.STATUS_RUNNING,
            {"protocol": "icbhi_robustness_v1", "resumed_at_utc": art._now_iso()},
        )
        return run_root, run_root.name

    run_id = art.new_run_id()
    run_root = parent / run_id
    if run_root.exists():
        raise FileExistsError(run_root)
    (run_root / "repetitions").mkdir(parents=True)
    (run_root / "figures").mkdir()
    write_json(run_root / "fingerprint.json", fingerprint)
    art.write_status(
        run_root,
        art.STATUS_RUNNING,
        {
            "protocol": "icbhi_robustness_v1",
            "created_at_utc": art._now_iso(),
            "outer_test_accessed": False,
        },
    )
    return run_root, run_id


def pipeline_context(
    *,
    protocol,
    pipeline,
    split,
    data_root: Path,
    cache_root: Path,
    device,
    num_workers: int,
    logger,
):
    cfg = load_architecture_config(protocol, pipeline)
    base_settings = cnn_model.TrainingSettings.from_config(cfg, num_workers)
    spec = pipeline.condition_spec()
    prepare = hcnn.make_prepare_fold(
        data_root, cache_root, spec, cfg, split, False, logger
    )
    evaluate = hcnn.make_evaluate_fold(
        spec, cfg, base_settings, device, "Healthy", logger
    )
    return cfg, spec, prepare, evaluate


def run_pipeline(
    *,
    protocol,
    pipeline,
    split,
    seed: int,
    data_root: Path,
    cache_root: Path,
    repetition_root: Path,
    device,
    num_workers: int,
    source_provenance: dict,
    logger,
) -> dict:
    pipeline_root = repetition_root / "pipelines" / pipeline.id
    pipeline_root.mkdir(parents=True, exist_ok=True)
    cfg, spec, prepare, evaluate = pipeline_context(
        protocol=protocol,
        pipeline=pipeline,
        split=split,
        data_root=data_root,
        cache_root=cache_root,
        device=device,
        num_workers=num_workers,
        logger=logger,
    )
    fold_ids = list(range(protocol.n_splits))
    hcv.run_condition_cv(
        run_root=pipeline_root,
        spec=spec,
        model_name=pipeline.architecture,
        split=split,
        configs=[pipeline.candidate()],
        hyperparameter_keys=HYPERPARAMETER_KEYS,
        negative_label_name="Healthy",
        threshold=float(cfg["evaluation"]["decision_threshold"]),
        fold_ids=fold_ids,
        prepare_fold=prepare,
        evaluate_fold=evaluate,
        release_fold=hcnn.make_release_fold(device),
        logger=logger,
    )
    return write_pipeline_outputs(
        pipeline_root=pipeline_root,
        spec=spec,
        pipeline=pipeline,
        cfg=cfg,
        fold_ids=fold_ids,
        seed=seed,
        source_provenance=source_provenance,
        logger=logger,
    )


def run_permutation_control(
    *,
    protocol,
    base_pipeline,
    split,
    seed: int,
    mapping: pd.DataFrame,
    data_root: Path,
    cache_root: Path,
    repetition_root: Path,
    device,
    num_workers: int,
    source_provenance: dict,
    logger,
) -> dict:
    control = dataclasses.replace(
        base_pipeline,
        id=f"{base_pipeline.id}_permuted",
        condition="permuted_labels",
        augment=False,
    )
    cfg = load_architecture_config(protocol, base_pipeline)
    spec = control.condition_spec()
    base_prepare = hcnn.make_prepare_fold(
        data_root, cache_root, spec, cfg, split, False, logger
    )

    def prepare(fold_id, train_patients, val_patients):
        context = base_prepare(fold_id, train_patients, val_patients)
        context.train_seg = apply_permuted_labels(context.train_seg, mapping)
        context.val_seg = apply_permuted_labels(context.val_seg, mapping)
        return context

    settings = cnn_model.TrainingSettings.from_config(cfg, num_workers)
    evaluate = hcnn.make_evaluate_fold(
        spec, cfg, settings, device, "Healthy", logger
    )
    pipeline_root = repetition_root / "controls" / control.id
    pipeline_root.mkdir(parents=True, exist_ok=True)
    fold_ids = list(range(protocol.n_splits))
    hcv.run_condition_cv(
        run_root=pipeline_root,
        spec=spec,
        model_name=control.architecture,
        split=split,
        configs=[control.candidate()],
        hyperparameter_keys=HYPERPARAMETER_KEYS,
        negative_label_name="Healthy",
        threshold=float(cfg["evaluation"]["decision_threshold"]),
        fold_ids=fold_ids,
        prepare_fold=prepare,
        evaluate_fold=evaluate,
        release_fold=hcnn.make_release_fold(device),
        logger=logger,
    )
    permutation_hash = str(mapping["permutation_sha256"].iloc[0])
    provenance = {
        **source_provenance,
        "label_permutation_sha256": permutation_hash,
    }
    return write_pipeline_outputs(
        pipeline_root=pipeline_root,
        spec=spec,
        pipeline=control,
        cfg=cfg,
        fold_ids=fold_ids,
        seed=seed,
        source_provenance=provenance,
        logger=logger,
        control=True,
    )


def write_device_baseline(
    repetition_root: Path,
    seed: int,
    split,
    fold_segments: dict[int, pd.DataFrame],
    settings: dict,
    threshold: float,
) -> dict:
    fold_metrics, predictions, pooled = run_device_baseline(
        split,
        fold_segments,
        settings,
        random_state=seed,
        negative_label_name="Healthy",
        threshold=threshold,
    )
    target = repetition_root / "device_baseline"
    target.mkdir(parents=True, exist_ok=True)
    fold_metrics.to_csv(
        target / "fold_metrics.csv", index=False, lineterminator="\n"
    )
    predictions.to_csv(
        target / "patient_predictions.csv", index=False, lineterminator="\n"
    )
    ev.confusion_matrix_df(
        predictions["target_label"].to_numpy(),
        predictions["score"].to_numpy(),
        ("Healthy", "COPD"),
        threshold,
    ).reset_index().rename(columns={"index": "real"}).to_csv(
        target / "confusion_matrix.csv", index=False, lineterminator="\n"
    )
    write_json(
        target / "manifest.json",
        {
            "protocol": "icbhi_robustness_v1",
            "baseline": "device_only_logistic_regression",
            "seed": seed,
            "settings": settings,
            "outer_test_accessed": False,
            "pooled_metrics": pooled,
        },
    )
    return {
        "seed": seed,
        "baseline": "device_only",
        **{f"pooled_{metric}": pooled[metric] for metric in hcv.SUMMARY_METRICS},
    }


def execute_repetition(
    *,
    protocol,
    seed: int,
    workspace_root: Path,
    data_root: Path,
    cache_root: Path,
    run_root: Path,
    device,
    num_workers: int,
    source_provenance: dict,
    logger,
    skip_permutation_control: bool,
) -> tuple[list[dict], dict]:
    tag = seed_tag(seed)
    csv_path, manifest_path = split_paths(workspace_root, seed)
    split = sp.load_holdout_split(csv_path, manifest_path, DATASET)
    seed_data_root = data_root / tag
    seed_cache_root = cache_root / tag
    repetition_root = run_root / "repetitions" / tag
    repetition_root.mkdir(parents=True, exist_ok=True)

    fold_segments = {}
    fold_manifests = []
    for fold_id in range(protocol.n_splits):
        detail = validate_fold_data(seed_data_root, split, fold_id)
        fold_manifests.append(detail)
        fold_segments[fold_id] = dmod.load_task_segments(
            seed_data_root, DATASET, hcv.cv_fold_ref(fold_id)
        )

    threshold = float(protocol.raw["evaluation"]["decision_threshold"])
    device_row = write_device_baseline(
        repetition_root,
        seed,
        split,
        fold_segments,
        protocol.raw["device_baseline"],
        threshold,
    )

    rows = []
    for pipeline in protocol.pipelines:
        logger.info(
            f"{tag}: preparando {pipeline.id} con configuracion congelada "
            f"{pipeline.source_config_index}"
        )
        rows.append(
            run_pipeline(
                protocol=protocol,
                pipeline=pipeline,
                split=split,
                seed=seed,
                data_root=seed_data_root,
                cache_root=seed_cache_root,
                repetition_root=repetition_root,
                device=device,
                num_workers=num_workers,
                source_provenance=source_provenance[pipeline.id],
                logger=logger,
            )
        )

    if (
        protocol.raw["permutation_control"]["enabled"]
        and not skip_permutation_control
    ):
        base_id = protocol.raw["permutation_control"]["pipeline_id"]
        base_pipeline = protocol.pipeline(base_id)
        mapping = permutation_map(
            split,
            random_state=seed,
            max_attempts=int(
                protocol.raw["permutation_control"]["max_shuffle_attempts"]
            ),
        )
        control_dir = repetition_root / "controls"
        control_dir.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(
            control_dir / "label_permutation.csv",
            index=False,
            lineterminator="\n",
        )
        rows.append(
            run_permutation_control(
                protocol=protocol,
                base_pipeline=base_pipeline,
                split=split,
                seed=seed,
                mapping=mapping,
                data_root=seed_data_root,
                cache_root=seed_cache_root,
                repetition_root=repetition_root,
                device=device,
                num_workers=num_workers,
                source_provenance=source_provenance[base_id],
                logger=logger,
            )
        )

    write_json(
        repetition_root / "manifest.json",
        {
            "protocol": protocol.raw["protocol"],
            "seed": seed,
            "split_csv_sha256": art.sha256_file(csv_path),
            "split_manifest_sha256": art.sha256_file(manifest_path),
            "fold_data": fold_manifests,
            "outer_test": {
                "n_blocked_patients": len(split.blocked_test_patients()),
                "accessed": False,
            },
            "pipelines": [
                {
                    "id": row["pipeline"],
                    "negative_control": row["negative_control"],
                    "status": "COMPLETED",
                }
                for row in rows
            ],
            "device_baseline": "COMPLETED",
        },
    )
    return rows, device_row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol_config)
    seeds = selected_seeds(protocol, args.only_seed)
    workspace_root = resolve_runtime_path(
        args.workspace_root,
        "PULMONARY_ICBHI_ROBUSTNESS_ROOT",
        "modeling/runtime/icbhi_robustness",
    )
    data_root = args.data_root.resolve()
    cache_root = args.cache_root.resolve()
    runs_root = args.runs_root.resolve()
    source_runs_root = (
        args.source_runs_root.resolve()
        if args.source_runs_root is not None
        else runs_root
    )

    report, source_provenance = dry_run_checks(
        protocol,
        seeds,
        workspace_root,
        data_root,
        source_runs_root,
        args.device,
    )
    with pd.option_context(
        "display.max_colwidth", 120, "display.width", 200
    ):
        print(report.to_string(index=False))
    if not bool(report["ok"].all()):
        print("\nveredicto: REVISAR", file=sys.stderr)
        return 1
    print("\nveredicto: OK")
    if args.dry_run:
        print("No se entreno nada (--dry-run).")
        return 0

    representative_cfg = load_architecture_config(
        protocol, protocol.pipeline("cnn_no_dn")
    )
    cnn_model.configure_determinism(representative_cfg)
    try:
        device = cnn_model.resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"--device {args.device}: {exc}", file=sys.stderr)
        return 1
    if device.type == "cuda":
        torch.cuda.set_device(device)

    fingerprint = build_fingerprint(
        protocol,
        seeds,
        workspace_root,
        data_root,
        source_provenance,
    )
    try:
        run_root, run_id = open_run(runs_root, args.resume, fingerprint)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1
    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} protocol={protocol.raw['protocol']} seeds={list(seeds)} "
        f"device={cnn_model.device_description(device)} num_workers={args.num_workers}"
    )
    art.write_environment(run_root)
    cexp._append_torch_environment(run_root, device)
    write_json(
        run_root / "resolved_protocol.json",
        {
            "protocol_path": str(protocol.path),
            "protocol": protocol.raw,
            "source_selections": source_provenance,
            "cli": vars(args),
            "outer_test_accessed": False,
        },
    )

    all_rows = []
    device_rows = []
    try:
        for seed in seeds:
            rows, device_row = execute_repetition(
                protocol=protocol,
                seed=seed,
                workspace_root=workspace_root,
                data_root=data_root,
                cache_root=cache_root,
                run_root=run_root,
                device=device,
                num_workers=args.num_workers,
                source_provenance=source_provenance,
                logger=logger,
                skip_permutation_control=args.skip_permutation_control,
            )
            all_rows.extend(rows)
            device_rows.append(device_row)

        rows_frame = pd.DataFrame(all_rows)
        rows_frame.to_csv(
            run_root / "robustness_by_repetition.csv",
            index=False,
            lineterminator="\n",
        )
        robustness_summary(rows_frame).to_csv(
            run_root / "robustness_summary.csv",
            index=False,
            lineterminator="\n",
        )
        paired_comparisons(rows_frame).to_csv(
            run_root / "paired_comparisons.csv",
            index=False,
            lineterminator="\n",
        )
        pd.DataFrame(device_rows).to_csv(
            run_root / "device_baseline_by_repetition.csv",
            index=False,
            lineterminator="\n",
        )
        control_rows = rows_frame.loc[rows_frame["negative_control"]]
        if not control_rows.empty:
            control_rows.to_csv(
                run_root / "permutation_control_by_repetition.csv",
                index=False,
                lineterminator="\n",
            )
        write_json(
            run_root / "run_manifest.json",
            {
                "protocol": protocol.raw["protocol"],
                "run_id": run_id,
                "status": "COMPLETED",
                "seeds": list(seeds),
                "pipelines": [pipeline.id for pipeline in protocol.pipelines],
                "outer_test": {"enabled": False, "accessed": False},
                "final_model": {"enabled": False},
                "interpretation": (
                    "Diagnostico de robustez; no selecciona automaticamente "
                    "un modelo ni autoriza abrir el test externo."
                ),
            },
        )
        art.write_status(
            run_root,
            art.STATUS_COMPLETED,
            {
                "protocol": protocol.raw["protocol"],
                "outer_test_accessed": False,
                "n_repetitions": len(seeds),
            },
        )
    except KeyboardInterrupt:
        art.write_status(
            run_root,
            "INTERRUPTED",
            {
                "protocol": protocol.raw["protocol"],
                "outer_test_accessed": False,
            },
        )
        logger.exception("ejecucion interrumpida")
        return 130
    except Exception as exc:  # noqa: BLE001
        art.write_status(
            run_root,
            art.STATUS_FAILED,
            {
                "protocol": protocol.raw["protocol"],
                "outer_test_accessed": False,
                "error": str(exc),
            },
        )
        logger.exception("fallo en el diagnostico de robustez")
        return 1

    logger.info(f"ejecucion {run_id} COMPLETED")
    print(f"run_id={run_id} estado=COMPLETED -> {run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



