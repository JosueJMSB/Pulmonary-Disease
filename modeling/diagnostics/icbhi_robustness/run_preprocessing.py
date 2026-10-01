"""Orquesta el preprocesamiento fold-aware de las repeticiones diagnosticas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ... import artifacts as art

from .common import DATASET, REPO_ROOT, load_protocol, resolve_runtime_path, seed_tag, write_json


@dataclass(frozen=True)
class Job:
    seed: int
    fold: int
    command: tuple[str, ...]
    log_path: Path
    manifest_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepara no_dn y dn para cada nuevo fold de ICBHI. "
            "Sin --execute solo imprime los comandos y no procesa audio."
        )
    )
    parser.add_argument("--protocol-config", type=Path, default=None)
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenera folds que ya tengan un manifest PASS. Por defecto se omiten.",
    )
    return parser.parse_args(argv)


def _manifest_passes(path: Path, split_csv: Path, split_manifest: Path) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("verdict") == "PASS"
        and manifest.get("protocol") == "holdout-v3"
        and manifest.get("stage") == "cv"
        and manifest.get("dataset_scope") == DATASET
        and manifest.get("holdout_splits_csv_sha256") == art.sha256_file(split_csv)
        and manifest.get("holdout_splits_manifest_sha256") == art.sha256_file(split_manifest)
        and manifest.get("outer_test", {}).get("excluded") is True
    )


def build_jobs(protocol, workspace_root: Path, data_root: Path, overwrite: bool) -> list[Job]:
    jobs = []
    for seed in protocol.seeds:
        split_dir = workspace_root / "splits" / seed_tag(seed)
        split_csv = split_dir / "holdout_splits.csv"
        split_manifest = split_dir / "holdout_splits_manifest.json"
        if not split_csv.is_file() or not split_manifest.is_file():
            raise FileNotFoundError(
                f"faltan splits para {seed_tag(seed)}; ejecute primero build_splits"
            )
        output_root = data_root / seed_tag(seed)
        for fold in range(protocol.n_splits):
            manifest_path = (
                output_root / DATASET / "cv" / f"fold_{fold:02d}" / "manifest.json"
            )
            if not overwrite and _manifest_passes(
                manifest_path, split_csv, split_manifest
            ):
                continue
            log_path = (
                workspace_root
                / "logs"
                / "preprocessing"
                / seed_tag(seed)
                / f"fold_{fold:02d}.log"
            )
            command = (
                sys.executable,
                "-u",
                "preprocessing/fold_denoising.py",
                "--protocol",
                "holdout-v3",
                "--stage",
                "cv",
                "--dataset-scope",
                DATASET,
                "--fold-id",
                str(fold),
                "--split-csv",
                str(split_csv),
                "--split-manifest",
                str(split_manifest),
                "--output-root",
                str(output_root),
            )
            jobs.append(
                Job(
                    seed=seed,
                    fold=fold,
                    command=command,
                    log_path=log_path,
                    manifest_path=manifest_path,
                )
            )
    return jobs


def run_job(job: Job) -> dict:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with open(job.log_path, "w", encoding="utf-8") as log:
        process = subprocess.run(
            list(job.command),
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status = "COMPLETED" if process.returncode == 0 else "FAILED"
    if status == "COMPLETED" and not job.manifest_path.is_file():
        status = "FAILED"
    return {
        "seed": job.seed,
        "fold": job.fold,
        "status": status,
        "returncode": int(process.returncode),
        "command": list(job.command),
        "log": str(job.log_path),
        "manifest": str(job.manifest_path),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_parallel < 1:
        raise ValueError("--max-parallel debe ser >= 1")
    protocol = load_protocol(args.protocol_config)
    workspace_root = resolve_runtime_path(
        args.workspace_root,
        "PULMONARY_ICBHI_ROBUSTNESS_ROOT",
        "modeling/runtime/icbhi_robustness",
    )
    data_root = args.data_root.resolve()
    jobs = build_jobs(protocol, workspace_root, data_root, args.overwrite)

    if not jobs:
        print("Los 25 folds ya tienen manifest PASS con los splits esperados.")
        return 0
    if not args.execute:
        for job in jobs:
            print(" ".join(job.command))
        print(
            f"Plan: {len(jobs)} fold(s), paralelismo={args.max_parallel}. "
            "No se ejecuto nada; agregue --execute en el servidor."
        )
        return 0

    results = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as pool:
        future_to_job = {pool.submit(run_job, job): job for job in jobs}
        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)
            print(
                f"{seed_tag(result['seed'])} fold_{result['fold']:02d}: "
                f"{result['status']} -> {result['log']}",
                flush=True,
            )
    results.sort(key=lambda row: (row["seed"], row["fold"]))
    status = "COMPLETED" if all(row["status"] == "COMPLETED" for row in results) else "FAILED"
    write_json(
        workspace_root / "preprocessing_status.json",
        {
            "protocol": protocol.raw["protocol"],
            "status": status,
            "outer_test_accessed": False,
            "jobs": results,
        },
    )
    print(f"Preprocesamiento diagnostico: {status}")
    return 0 if status == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

