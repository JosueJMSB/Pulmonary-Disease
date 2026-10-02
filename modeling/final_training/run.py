"""CLI del protocolo holdout_final_v1: reentrenamiento definitivo (100% del
80% de desarrollo) y evaluacion externa UNICA de los pipelines congelados en
``modeling/configs/final_test/selected_pipelines.toml``.

    python -m modeling.final_training.run \\
        --selection-config modeling/configs/final_test/selected_pipelines.toml \\
        --dataset all --device cuda:0 --dry-run

    python -m modeling.final_training.run \\
        --selection-config modeling/configs/final_test/selected_pipelines.toml \\
        --dataset all --device cuda:0 --execute

``--dataset all`` recorre ICBHI, FRAIWAN_Extended y COMBINED en un solo
proceso (no hay busqueda de hiperparametros que paralelizar entre ellos, asi
que no hace falta un lanzador de subprocesos como ``run_holdout_sequence.py``).
Un dataset con seleccion pendiente (``status`` distinto de ``"approved"`` en
la seleccion congelada) queda ``BLOCKED_SELECTION`` y no detiene a los demas.
Un dataset ya ``COMPLETED`` no vuelve a abrir el test: ``--resume`` lo omite.

No hay rutas personales: ``--data-root``/``--cache-root``/``--runs-root``/
``--source-runs-root``, o ``PULMONARY_DATA_ROOT``/``PULMONARY_CACHE_ROOT``/
``PULMONARY_RUNS_ROOT``/``PULMONARY_SOURCE_RUNS_ROOT``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

from .. import artifacts as art
from .. import data as dmod
from ..models import cnn as cnn_model
from . import core
from . import selection as sel

DATASET_CHOICES = ("ICBHI", "FRAIWAN_Extended", "COMBINED", "all")
DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda|cuda:\d+)$")
RUN_MODEL_NAME = "final_training"  # <runs_root>/final_training/<run_id>/


def _device_arg(value: str) -> str:
    if not DEVICE_PATTERN.match(value):
        raise argparse.ArgumentTypeError("use auto, cpu, cuda o cuda:N")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reentrenamiento definitivo y evaluacion externa unica (protocolo holdout_final_v1)."
    )
    parser.add_argument("--selection-config", type=Path, default=None, help=f"Por defecto {sel.DEFAULT_SELECTION_CONFIG}.")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, required=True)
    parser.add_argument("--device", type=_device_arg, default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument(
        "--source-runs-root", type=Path, default=None,
        help="Raiz de las ejecuciones de holdout_cv_v3 contra las que se verifican los hiperparametros "
             "congelados (por defecto modeling/runs, el mismo PULMONARY_RUNS_ROOT de esas corridas).",
    )
    parser.add_argument(
        "--force-features", action="store_true",
        help="Regenera la cache Log-Mel de la etapa final aunque ya exista y sea valida.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Valida seleccion, split y datos; no entrena ni abre el test.")
    mode.add_argument("--execute", action="store_true", help="Entrena el modelo definitivo y evalua el test una sola vez.")
    parser.add_argument("--resume", nargs="?", const="latest", default=None, metavar="RUN_ID")
    return parser.parse_args(argv)


def resolve_datasets(dataset_arg: str) -> list[str]:
    return ["ICBHI", "FRAIWAN_Extended", "COMBINED"] if dataset_arg == "all" else [dataset_arg]


def _append_torch_environment(run_root: Path, device: torch.device) -> None:
    path = run_root / "environment.txt"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = [f"{key}: {value}" for key, value in cnn_model.torch_environment(device).items()]
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def _write_sequence_status(run_root: Path, status: str, extra: dict) -> None:
    art.write_status(run_root, status, extra)
    shutil.copyfile(run_root / "status.json", run_root / "sequence_status.json")


def _final_status(results: list[dict]) -> str:
    statuses = {r["status"] for r in results}
    if statuses == {core.STATUS_COMPLETED}:
        return art.STATUS_COMPLETED
    if statuses & {core.STATUS_COMPLETED}:
        return art.STATUS_PARTIAL
    if statuses & {core.STATUS_FAILED, core.STATUS_INTERRUPTED}:
        return art.STATUS_FAILED
    return art.STATUS_PARTIAL  # p. ej. todo BLOCKED_SELECTION: nada fallo, pero nada se completo


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        selection_cfg = sel.load_selection_config(args.selection_config)
    except sel.SelectionError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    datasets = resolve_datasets(args.dataset)
    data_root = dmod.resolve_path(args.data_root, "PULMONARY_DATA_ROOT", "preprocessing/data/holdout_calibrated")
    cache_root = dmod.resolve_path(args.cache_root, "PULMONARY_CACHE_ROOT", "modeling/cache/logmel_holdout")
    runs_root = dmod.resolve_path(args.runs_root, "PULMONARY_RUNS_ROOT", "modeling/runs")
    source_runs_root = dmod.resolve_path(args.source_runs_root, "PULMONARY_SOURCE_RUNS_ROOT", "modeling/runs")

    if args.dry_run:
        report = core.dry_run_check(selection_cfg, datasets, data_root, source_runs_root)
        with pd.option_context("display.max_colwidth", 120, "display.width", 180):
            print(report["checks"].to_string(index=False))
        print(f"\nveredicto: {'OK' if report['ok'] else 'REVISAR'}")
        return 0 if report["ok"] else 1

    try:
        device = cnn_model.resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        print(f"--device {args.device}: {exc}", file=sys.stderr)
        return 1
    if device.type == "cuda":
        torch.cuda.set_device(device)

    try:
        run_root, run_id = art.init_run(runs_root, RUN_MODEL_NAME, args.resume, bool(args.resume))
    except (FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root, name="final_training")
    if args.resume:
        removed = core.cleanup_abandoned_staging(run_root)
        art.write_status(run_root, art.STATUS_RUNNING, {"resumed_at_utc": art._now_iso()})
        if removed:
            logger.info(f"staging abandonado (sin predicciones) eliminado: {[str(p) for p in removed]}")

    logger.info(
        f"run_id={run_id} protocolo={sel.PROTOCOL_NAME} dataset={args.dataset} "
        f"device={cnn_model.device_description(device)} resume={bool(args.resume)}"
    )
    art.write_environment(run_root)
    _append_torch_environment(run_root, device)

    results: list[dict] = []
    for dataset in datasets:
        try:
            result = core.run_pipeline_for_dataset(
                run_root=run_root, dataset=dataset, selection_cfg=selection_cfg,
                data_root=data_root, cache_root=cache_root, source_runs_root=source_runs_root,
                device=device, num_workers=args.num_workers, force_features=args.force_features,
                dry_run=False, logger=logger,
            )
        except KeyboardInterrupt:
            logger.warning(f"{dataset}: interrumpido por el usuario; ningun artefacto parcial se publico")
            results.append({"dataset": dataset, "status": core.STATUS_INTERRUPTED})
            _write_sequence_status(run_root, core.STATUS_INTERRUPTED, {"datasets": results})
            print(f"run_id={run_id} estado={core.STATUS_INTERRUPTED} -> {run_root}")
            return 130
        except Exception as exc:  # noqa: BLE001 - se registra, nunca se omite en silencio
            logger.exception(f"{dataset}: fallo no controlado")
            results.append({"dataset": dataset, "status": core.STATUS_FAILED, "detail": str(exc)})
            continue
        results.append(result)

    summary_rows = [
        {
            "dataset": r["dataset"], "condition": r.get("condition"), "status": r["status"],
            **{m: r.get("metrics", {}).get(m) for m in core.SUMMARY_METRICS},
        }
        for r in results
    ]
    pd.DataFrame(summary_rows).to_csv(run_root / "final_test_summary.csv", index=False, lineterminator="\n")

    final_status = _final_status(results)
    _write_sequence_status(run_root, final_status, {
        "protocol": sel.PROTOCOL_NAME,
        "datasets": [{"dataset": r["dataset"], "status": r["status"]} for r in results],
    })
    logger.info(f"ejecucion {run_id} finalizada con estado {final_status}")
    print(f"run_id={run_id} estado={final_status} -> {run_root}")
    return 0 if final_status == art.STATUS_COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
