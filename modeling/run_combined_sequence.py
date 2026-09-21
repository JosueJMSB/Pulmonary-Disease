"""Lanzador secuencial SVM-RBF -> CNN -> CRNN para el dataset COMBINED.

    python -m modeling.run_combined_sequence --dry-run-only
    python -m modeling.run_combined_sequence --execute --device cuda:0

Primero ejecuta el ``--dry-run`` de los tres modelos, siempre en ese orden;
si alguno falla, la secuencia se detiene sin entrenar nada. Con ``--execute``
entrena secuencialmente SVM -> CNN -> CRNN: CPU para la SVM (no usa GPU) y
``--device`` para CNN/CRNN. Si un modelo falla (o termina PARTIAL), la
secuencia se detiene sin lanzar el siguiente.

Cada paso es una invocacion de ``python -m modeling.run_experiment`` en un
subproceso, leida linea por linea en tiempo real (``subprocess.Popen``, no
``subprocess.run``): cada linea se refleja de inmediato en esta misma
consola (visible en la sesion tmux) y se escribe y sincroniza a
``sequence.log`` sin esperar a que el subproceso termine. En cuanto aparece
``run_id=`` -run_experiment lo registra con ``logger.info`` justo despues de
abrir la ejecucion, mucho antes de terminar- se guarda de inmediato en
``sequence_status.json``. Esto es lo que hace seguro a
``--resume-sequence``: si el entrenamiento se interrumpe (Ctrl+C, o la sesion
se cae) a mitad de un fold, el run_id de esa ejecucion ya quedo persistido y
``--resume-sequence`` puede reanudarla con ``--resume <run_id>`` en vez de
empezar desde cero.

    <runs-root>/combined_sequences/<sequence_id>/
    |-- sequence.log
    `-- sequence_status.json

``--resume-sequence`` (con un SEQUENCE_ID o, sin valor, la mas reciente)
reanuda: omite los modelos que ya terminaron COMPLETED y, para el modelo que
quedo a medias (FAILED, INTERRUPTED o incluso RUNNING si el proceso murio sin
que este script llegara a registrar el desenlace), pasa ``--resume <run_id>``
a ``run_experiment`` con el run_id exacto registrado en el intento anterior
(no ``--resume`` a secas: evita reanudar por error la ejecucion mas reciente
de ese modelo si en el medio se corrio otro experimento del mismo --model
sobre otro dataset).

Un Ctrl+C durante un paso termina el subproceso (con margen para que cierre
en orden) y marca la secuencia como INTERRUPTED -distinto de FAILED, que es
para cuando el propio modelo termina con un error-, preservando siempre el
run_id ya capturado.

Ruta de resultados: sin ``--runs-root``, se respeta ``PULMONARY_RUNS_ROOT``
antes de caer en el valor por defecto dentro del repositorio (mismo orden de
prioridad que ``modeling.data.resolve_path``): ``--runs-root`` >
``PULMONARY_RUNS_ROOT`` > ``modeling/runs``. La ruta resuelta una sola vez se
usa tanto para ``combined_sequences/`` como para el ``--runs-root`` que se le
pasa a cada subproceso, de modo que los modelos y el estado de la secuencia
siempre terminan bajo la misma raiz.

No fija ninguna GPU fisica: ``CUDA_VISIBLE_DEVICES`` se establece, si hace
falta, al invocar este script en el servidor, no dentro de el.
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

MODELS = ("svm_rbf", "cnn", "crnn")
CONFIG_NAMES = {
    "svm_rbf": "svm_rbf_combined.toml",
    "cnn": "cnn_combined.toml",
    "crnn": "crnn_combined.toml",
}
RUN_ID_PATTERN = re.compile(r"run_id=(\S+)")

STATUS_RUNNING = "RUNNING"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_DRY_RUN_OK = "DRY_RUN_OK"
STATUS_SKIPPED = "SKIPPED"

# Codigo de salida convencional para una interrupcion (128 + SIGINT).
EXIT_INTERRUPTED = 130


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida (y, con --execute, entrena) SVM-RBF, CNN y CRNN en secuencia "
                    "sobre el dataset COMBINED."
    )
    parser.add_argument(
        "--dry-run-only", action="store_true",
        help="Solo valida los tres modelos con --dry-run; nunca entrena, aunque se pase --execute.",
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
        help="--num-workers para CNN/CRNN (por defecto 0: el servidor mostro advertencias de "
             "fork() con mas workers, y los entrenamientos anteriores corrieron bien con 0).",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--runs-root", type=Path, default=None,
        help="Por defecto, PULMONARY_RUNS_ROOT si esta definida, si no modeling/runs dentro "
             "del repositorio. Se usa tanto para combined_sequences/ como para cada subproceso.",
    )
    parser.add_argument(
        "--resume-sequence", nargs="?", const="latest", default=None, metavar="SEQUENCE_ID",
        help="Continua una secuencia existente (SEQUENCE_ID o, sin valor, la mas reciente): "
             "omite los modelos ya COMPLETED y reanuda (--resume <run_id>) el que quedo a medias.",
    )
    return parser.parse_args(argv)


def resolve_runs_root(cli_value: Path | str | None) -> Path:
    """``--runs-root`` > ``PULMONARY_RUNS_ROOT`` > ``modeling/runs`` del repo.

    Mismo orden de prioridad que ``modeling.data.resolve_path``/
    ``resolve_runs_root``, reutilizado aqui para que este lanzador y cada
    subproceso ``run_experiment`` (que resuelve su propio --runs-root con la
    misma funcion) siempre terminen de acuerdo.
    """
    return dmod.resolve_path(cli_value, "PULMONARY_RUNS_ROOT", "modeling/runs")


@dataclass
class StepResult:
    status: str
    returncode: int | None
    seconds: float
    command: list[str]
    run_id: str | None


def build_command(
    model: str, args: argparse.Namespace, dry_run: bool, resume_run_id: str | None, runs_root: Path,
) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "modeling.run_experiment",
        "--model", model, "--dataset", "COMBINED", "--experiment", "all",
        "--config", str(CONFIGS_DIR / CONFIG_NAMES[model]),
        "--runs-root", str(runs_root),
    ]
    if args.data_root is not None:
        cmd += ["--data-root", str(args.data_root)]
    if model in ("cnn", "crnn"):
        if args.cache_root is not None:
            cmd += ["--cache-root", str(args.cache_root)]
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
    args: argparse.Namespace,
    dry_run: bool,
    resume_run_id: str | None,
    runs_root: Path,
    log_fh,
    on_run_id=None,
) -> StepResult:
    """Lanza el subproceso y procesa su salida linea por linea, en vivo.

    Cada linea se imprime de inmediato (progreso visible en la sesion donde
    corre este script) y se escribe y sincroniza a ``sequence.log`` sin
    esperar a que el subproceso termine. En cuanto una linea trae
    ``run_id=...`` (run_experiment lo registra apenas abre la ejecucion, no
    solo al terminar), se llama a ``on_run_id`` para que quede persistido de
    inmediato. Un Ctrl+C durante la espera termina el subproceso (con margen
    para un cierre en orden) y se reporta como INTERRUPTED, conservando el
    run_id ya capturado.
    """
    command = build_command(model, args, dry_run, resume_run_id, runs_root)
    log_fh.write(f"[{_now_iso()}] {model}: {' '.join(command)}\n")
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
            f"[{_now_iso()}] {model}: interrumpido por el usuario, subproceso detenido "
            f"({seconds:.1f} s, run_id={run_id})\n"
        )
        log_fh.flush()
        return StepResult(
            status=STATUS_INTERRUPTED, returncode=process.returncode, seconds=seconds,
            command=command, run_id=run_id,
        )

    seconds = time.perf_counter() - started
    log_fh.write(f"[{_now_iso()}] {model}: codigo de salida {returncode} ({seconds:.1f} s)\n")
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


def _last_train_run_id(steps: list[dict], model: str) -> str | None:
    """El run_id mas reciente registrado para ``model`` en la fase de entrenamiento."""
    for step in reversed(steps):
        if step.get("phase") == "train" and step.get("model") == model and step.get("run_id"):
            return step["run_id"]
    return None


def _run_and_record(
    model: str,
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
    """Ejecuta un paso y mantiene ``sequence_status.json`` al dia: al empezar,
    en cuanto aparece el run_id (aunque el proceso muera justo despues) y al
    terminar (COMPLETED, FAILED o INTERRUPTED)."""
    step_record = {
        "phase": phase, "model": model, "status": STATUS_RUNNING,
        "returncode": None, "seconds": None, "command": [], "run_id": None,
        "at_utc": _now_iso(),
    }
    steps.append(step_record)
    _write_status(status_path, sequence_id, steps, STATUS_RUNNING)

    def on_run_id(found_run_id: str) -> None:
        step_record["run_id"] = found_run_id
        _write_status(status_path, sequence_id, steps, STATUS_RUNNING)

    result = run_step(model, args, dry_run, resume_run_id, runs_root, log_fh, on_run_id=on_run_id)
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
    sequences_root = runs_root / "combined_sequences"

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
    completed_train_models = {
        step["model"] for step in steps if step.get("phase") == "train" and step.get("status") == STATUS_COMPLETED
    }
    execute = bool(args.execute and not args.dry_run_only)

    with open(log_path, "a", encoding="utf-8") as log_fh:
        log_fh.write(f"[{_now_iso()}] secuencia {sequence_id}: validando SVM -> CNN -> CRNN con --dry-run\n")
        log_fh.flush()

        for model in MODELS:
            result = _run_and_record(
                model, args, True, None, "dry_run", runs_root, steps, status_path, sequence_id, log_fh,
            )
            if result.status == STATUS_INTERRUPTED:
                log_fh.write(f"[{_now_iso()}] {model}: --dry-run interrumpido por el usuario\n")
                _write_status(status_path, sequence_id, steps, STATUS_INTERRUPTED)
                print(f"secuencia interrumpida durante el dry-run de {model}; vea {log_path}", file=sys.stderr)
                return EXIT_INTERRUPTED
            if result.status != STATUS_COMPLETED:
                log_fh.write(f"[{_now_iso()}] {model}: --dry-run fallo; secuencia detenida, no se entrena nada\n")
                _write_status(status_path, sequence_id, steps, STATUS_FAILED)
                print(f"dry-run de {model} fallo (codigo {result.returncode}); vea {log_path}", file=sys.stderr)
                return 1

        log_fh.write(f"[{_now_iso()}] los tres --dry-run terminaron OK\n")
        log_fh.flush()

        if not execute:
            _write_status(status_path, sequence_id, steps, STATUS_DRY_RUN_OK)
            print(f"dry-run-only: SVM, CNN y CRNN validaron OK -> {sequence_dir}")
            return 0

        for model in MODELS:
            if model in completed_train_models:
                log_fh.write(f"[{_now_iso()}] {model}: ya estaba COMPLETED en esta secuencia, se omite\n")
                steps.append({
                    "phase": "train", "model": model, "status": STATUS_SKIPPED,
                    "returncode": 0, "seconds": 0.0, "command": [], "run_id": None, "at_utc": _now_iso(),
                })
                _write_status(status_path, sequence_id, steps, STATUS_RUNNING)
                continue

            resume_run_id = _last_train_run_id(steps, model)
            if resume_run_id:
                log_fh.write(f"[{_now_iso()}] {model}: reanudando run_id={resume_run_id}\n")
                log_fh.flush()

            result = _run_and_record(
                model, args, False, resume_run_id, "train", runs_root, steps, status_path, sequence_id, log_fh,
            )
            if result.status == STATUS_INTERRUPTED:
                log_fh.write(
                    f"[{_now_iso()}] {model}: entrenamiento interrumpido por el usuario (run_id={result.run_id})\n"
                )
                _write_status(status_path, sequence_id, steps, STATUS_INTERRUPTED)
                print(
                    f"secuencia interrumpida durante el entrenamiento de {model} (run_id={result.run_id}); "
                    f"vea {log_path}. Reanude con --resume-sequence {sequence_id}",
                    file=sys.stderr,
                )
                return EXIT_INTERRUPTED
            if result.status != STATUS_COMPLETED:
                log_fh.write(f"[{_now_iso()}] {model}: entrenamiento fallo; secuencia detenida\n")
                _write_status(status_path, sequence_id, steps, STATUS_FAILED)
                print(
                    f"{model} fallo (codigo {result.returncode}); vea {log_path}. "
                    f"Corrija la causa y reanude con --resume-sequence {sequence_id}",
                    file=sys.stderr,
                )
                return 1

    _write_status(status_path, sequence_id, steps, STATUS_COMPLETED)
    print(f"secuencia {sequence_id} completada: SVM -> CNN -> CRNN -> {sequence_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
