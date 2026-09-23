"""Lanzador secuencial del protocolo fold-aware (v2): SVM-RBF -> CNN -> CRNN,
cada uno primero sobre ICBHI+Fraiwan y luego sobre COMBINED.

    python -m modeling.run_folded_sequence --dry-run-only
    python -m modeling.run_folded_sequence --execute --device cuda:0

Script independiente de ``run_combined_sequence.py`` (sin compartir ningun
import mas alla de utilidades genericas ya existentes como
``modeling.data.resolve_path``): ese lanzador queda intacto. La mecanica es
la misma, ya validada alli -streaming en vivo linea por linea via
``subprocess.Popen``, ``run_id`` capturado apenas aparece (y conservado si el
entrenamiento se interrumpe), ``INTERRUPTED`` distinto de ``FAILED``,
``--runs-root`` > ``PULMONARY_RUNS_ROOT`` > valor por defecto,
``--num-workers 0`` por defecto, ``--resume-sequence``-, solo que apunta a
los seis TOML del protocolo fold-aware, en este orden exacto (para el
``--dry-run`` de validacion y, con ``--execute``, para el entrenamiento):

    1. svm_rbf  con svm_rbf_v2.toml          (ICBHI + FRAIWAN_Extended)
    2. svm_rbf  con svm_rbf_combined_v2.toml (COMBINED)
    3. cnn      con cnn_v2.toml
    4. cnn      con cnn_combined_v2.toml
    5. crnn     con crnn_v2.toml
    6. crnn     con crnn_combined_v2.toml

Se detiene ante el primer error (dry-run o entrenamiento) sin avanzar al
paso siguiente. No fija ninguna GPU fisica: ``CUDA_VISIBLE_DEVICES`` se
establece, si hace falta, al invocar este script en el servidor, no dentro
de el.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import data as dmod

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# (modelo, nombre del TOML) en el orden exacto de ejecucion: cada modelo,
# primero individual (ICBHI+Fraiwan) y luego COMBINED, antes de pasar al
# siguiente modelo.
STEPS = (
    ("svm_rbf", "svm_rbf_v2.toml"),
    ("svm_rbf", "svm_rbf_combined_v2.toml"),
    ("cnn", "cnn_v2.toml"),
    ("cnn", "cnn_combined_v2.toml"),
    ("crnn", "crnn_v2.toml"),
    ("crnn", "crnn_combined_v2.toml"),
)
RUN_ID_PATTERN = re.compile(r"run_id=(\S+)")

STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_DRY_RUN_OK = "DRY_RUN_OK"
STATUS_SKIPPED = "SKIPPED"

EXIT_INTERRUPTED = 130


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida (y, con --execute, entrena) SVM-RBF, CNN y CRNN en secuencia, "
                    "cada uno primero sobre ICBHI+Fraiwan y luego sobre COMBINED (protocolo fold-aware v2)."
    )
    parser.add_argument(
        "--dry-run-only", action="store_true",
        help="Solo valida los seis pasos con --dry-run; nunca entrena, aunque se pase --execute.",
    )
    parser.add_argument(
        "--execute", action="store_true",
        help="Entrena tras validar. Sin esta bandera, el comportamiento es el de --dry-run-only.",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Dispositivo para CNN/CRNN (auto, cpu, cuda o cuda:N). La SVM siempre corre en CPU.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="--n-jobs para la SVM.")
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="--num-workers para CNN/CRNN (por defecto 0).",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path, default=None,
        help="Por defecto, PULMONARY_RUNS_ROOT si esta definida, si no modeling/runs dentro "
             "del repositorio. Se usa tanto para folded_sequences/ como para cada subproceso.",
    )
    parser.add_argument(
        "--resume-sequence", nargs="?", const="latest", default=None, metavar="SEQUENCE_ID",
        help="Continua una secuencia existente (SEQUENCE_ID o, sin valor, la mas reciente): "
             "omite los pasos ya COMPLETED y reanuda (--resume <run_id>) el que quedo a medias.",
    )
    return parser.parse_args(argv)


def resolve_runs_root(cli_value: Path | str | None) -> Path:
    return dmod.resolve_path(cli_value, "PULMONARY_RUNS_ROOT", "modeling/runs")


@dataclass
class StepResult:
    status: str
    returncode: int | None
    seconds: float
    command: list[str]
    run_id: str | None


def build_command(
    model: str, config_name: str, args: argparse.Namespace, dry_run: bool,
    resume_run_id: str | None, runs_root: Path,
) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "modeling.run_experiment",
        "--model", model, "--dataset", "all", "--experiment", "all",
        "--config", str(CONFIGS_DIR / config_name),
        "--runs-root", str(runs_root),
    ]
    if args.data_root is not None:
        cmd += ["--data-root", str(args.data_root)]
    if args.cache_root is not None:
        cmd += ["--cache-root", str(args.cache_root)]
    if model in ("cnn", "crnn"):
        cmd += ["--device", args.device, "--num-workers", str(args.num_workers)]
    else:
        cmd += ["--n-jobs", str(args.n_jobs)]
    if dry_run:
        cmd += ["--dry-run"]
    elif resume_run_id is not None:
        cmd += ["--resume", resume_run_id]
    return cmd


def run_step(
    model: str,
    config_name: str,
    args: argparse.Namespace,
    dry_run: bool,
    resume_run_id: str | None,
    runs_root: Path,
    log_fh,
    on_run_id=None,
) -> StepResult:
    """Lanza el subproceso y procesa su salida linea por linea, en vivo (ver
    run_combined_sequence.run_step, misma mecanica, script independiente)."""
    command = build_command(model, config_name, args, dry_run, resume_run_id, runs_root)
    log_fh.write(f"[{_now_iso()}] {model} ({config_name}): {' '.join(command)}\n")
    log_fh.flush()

    started = time.perf_counter()
    run_id: str | None = None
    process = subprocess.Popen(
        command, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_fh.write(line)
            log_fh.flush()
            if run_id is None:
                match = RUN_ID_PATTERN.search(line)
                if match:
                    run_id = match.group(1)
                    if on_run_id is not None:
                        on_run_id(run_id)
        returncode = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        seconds = time.perf_counter() - started
        log_fh.write(
            f"[{_now_iso()}] {model} ({config_name}): interrumpido por el usuario, subproceso detenido "
            f"({seconds:.1f} s, run_id={run_id})\n"
        )
        log_fh.flush()
        return StepResult(
            status=STATUS_INTERRUPTED, returncode=process.returncode, seconds=seconds,
            command=command, run_id=run_id,
        )

    seconds = time.perf_counter() - started
    log_fh.write(f"[{_now_iso()}] {model} ({config_name}): codigo de salida {returncode} ({seconds:.1f} s)\n")
    log_fh.flush()
    status = STATUS_COMPLETED if returncode == 0 else STATUS_FAILED
    return StepResult(status=status, returncode=returncode, seconds=seconds, command=command, run_id=run_id)


def _find_latest_sequence(sequences_root: Path) -> Path | None:
    if not sequences_root.is_dir():
        return None
    candidates = [
        p for p in sequences_root.iterdir()
        if p.is_dir() and (p / "sequence_status.json").is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _write_status(status_path: Path, sequence_id: str, steps: list[dict], final_status: str) -> None:
    payload = {"sequence_id": sequence_id, "final_status": final_status, "updated_at_utc": _now_iso(), "steps": steps}
    status_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _step_key(model: str, config_name: str) -> str:
    return f"{model}:{config_name}"


def _last_train_run_id(steps: list[dict], step_key: str) -> str | None:
    for step in reversed(steps):
        if step.get("phase") == "train" and step.get("step_key") == step_key and step.get("run_id"):
            return step["run_id"]
    return None


def _run_and_record(
    model: str,
    config_name: str,
    args: argparse.Namespace,
    dry_run: bool,
    resume_run_id: str | None,
    phase: str,
    runs_root: Path,
    steps: list[dict],
    status_path: Path,
    sequence_id: str,
    log_fh,
) -> StepResult:
    step_record = {
        "phase": phase, "model": model, "config": config_name, "step_key": _step_key(model, config_name),
        "status": STATUS_RUNNING, "returncode": None, "seconds": None, "command": [], "run_id": None,
        "at_utc": _now_iso(),
    }
    steps.append(step_record)
    _write_status(status_path, sequence_id, steps, STATUS_RUNNING)

    def on_run_id(found_run_id: str) -> None:
        step_record["run_id"] = found_run_id
        _write_status(status_path, sequence_id, steps, STATUS_RUNNING)

    result = run_step(model, config_name, args, dry_run, resume_run_id, runs_root, log_fh, on_run_id=on_run_id)
    step_record.update({
        "status": result.status,
        "returncode": result.returncode,
        "seconds": result.seconds,
        "command": result.command,
        "run_id": result.run_id or step_record["run_id"],
        "at_utc": _now_iso(),
    })
    _write_status(status_path, sequence_id, steps, STATUS_RUNNING)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs_root = resolve_runs_root(args.runs_root)
    sequences_root = runs_root / "folded_sequences"

    steps: list[dict] = []
    if args.resume_sequence:
        if args.resume_sequence == "latest":
            sequence_dir = _find_latest_sequence(sequences_root)
            if sequence_dir is None:
                print("--resume-sequence: no hay ninguna secuencia previa que reanudar", file=sys.stderr)
                return 2
        else:
            sequence_dir = sequences_root / args.resume_sequence
            if not (sequence_dir / "sequence_status.json").is_file():
                print(f"--resume-sequence {args.resume_sequence}: no existe {sequence_dir}", file=sys.stderr)
                return 2
        sequence_id = sequence_dir.name
        stored = json.loads((sequence_dir / "sequence_status.json").read_text(encoding="utf-8"))
        steps = list(stored.get("steps", []))
    else:
        sequence_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        sequence_dir = sequences_root / sequence_id
        sequence_dir.mkdir(parents=True, exist_ok=False)

    log_path = sequence_dir / "sequence.log"
    status_path = sequence_dir / "sequence_status.json"
    completed_train_steps = {
        step["step_key"] for step in steps if step.get("phase") == "train" and step.get("status") == STATUS_COMPLETED
    }
    execute = bool(args.execute and not args.dry_run_only)

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(
            f"[{_now_iso()}] secuencia {sequence_id}: validando los seis pasos con --dry-run "
            "(SVM->CNN->CRNN, cada uno ICBHI+Fraiwan y luego COMBINED)\n"
        )
        log_fh.flush()

        for model, config_name in STEPS:
            result = _run_and_record(
                model, config_name, args, True, None, "dry_run", runs_root, steps, status_path, sequence_id, log_fh,
            )
            if result.status == STATUS_INTERRUPTED:
                log_fh.write(f"[{_now_iso()}] {model} ({config_name}): --dry-run interrumpido por el usuario\n")
                _write_status(status_path, sequence_id, steps, STATUS_INTERRUPTED)
                print(f"secuencia interrumpida durante el dry-run de {model} ({config_name}); vea {log_path}", file=sys.stderr)
                return EXIT_INTERRUPTED
            if result.status != STATUS_COMPLETED:
                log_fh.write(f"[{_now_iso()}] {model} ({config_name}): --dry-run fallo; secuencia detenida, no se entrena nada\n")
                _write_status(status_path, sequence_id, steps, STATUS_FAILED)
                print(f"dry-run de {model} ({config_name}) fallo (codigo {result.returncode}); vea {log_path}", file=sys.stderr)
                return 1

        log_fh.write(f"[{_now_iso()}] los seis --dry-run terminaron OK\n")
        log_fh.flush()

        if not execute:
            _write_status(status_path, sequence_id, steps, STATUS_DRY_RUN_OK)
            print(f"dry-run-only: los seis pasos validaron OK -> {sequence_dir}")
            return 0

        for model, config_name in STEPS:
            step_key = _step_key(model, config_name)
            if step_key in completed_train_steps:
                log_fh.write(f"[{_now_iso()}] {model} ({config_name}): ya estaba COMPLETED en esta secuencia, se omite\n")
                steps.append({
                    "phase": "train", "model": model, "config": config_name, "step_key": step_key,
                    "status": STATUS_SKIPPED, "returncode": 0, "seconds": 0.0, "command": [], "run_id": None,
                    "at_utc": _now_iso(),
                })
                _write_status(status_path, sequence_id, steps, STATUS_RUNNING)
                continue

            resume_run_id = _last_train_run_id(steps, step_key)
            if resume_run_id:
                log_fh.write(f"[{_now_iso()}] {model} ({config_name}): reanudando run_id={resume_run_id}\n")
                log_fh.flush()

            result = _run_and_record(
                model, config_name, args, False, resume_run_id, "train", runs_root, steps, status_path, sequence_id, log_fh,
            )
            if result.status == STATUS_INTERRUPTED:
                log_fh.write(
                    f"[{_now_iso()}] {model} ({config_name}): entrenamiento interrumpido por el usuario "
                    f"(run_id={result.run_id})\n"
                )
                _write_status(status_path, sequence_id, steps, STATUS_INTERRUPTED)
                print(
                    f"secuencia interrumpida durante el entrenamiento de {model} ({config_name}) "
                    f"(run_id={result.run_id}); vea {log_path}. Reanude con --resume-sequence {sequence_id}",
                    file=sys.stderr,
                )
                return EXIT_INTERRUPTED
            if result.status != STATUS_COMPLETED:
                log_fh.write(f"[{_now_iso()}] {model} ({config_name}): entrenamiento fallo; secuencia detenida\n")
                _write_status(status_path, sequence_id, steps, STATUS_FAILED)
                print(
                    f"{model} ({config_name}) fallo (codigo {result.returncode}); vea {log_path}. "
                    f"Corrija la causa y reanude con --resume-sequence {sequence_id}",
                    file=sys.stderr,
                )
                return 1

    _write_status(status_path, sequence_id, steps, STATUS_COMPLETED)
    print(f"secuencia {sequence_id} completada: los seis pasos entrenados -> {sequence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
