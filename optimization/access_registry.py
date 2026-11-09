"""Registro de la evaluacion INT8 sobre el test externo.

El test externo YA se uso para evaluar los modelos ``.pt`` en la corrida
final. Este registro no lo describe como nunca consultado: solo impide
REPETIR la evaluacion INT8 de cada dataset, en cualquier corrida de
``optimization``. Vive en ``<output_root>/_test_access/<dataset>_int8.json``,
fuera de cada corrida, para que una corrida nueva no pueda saltarselo.

Estados: STARTED -> PREDICTIONS_WRITTEN -> COMPLETED.

- Sin registro: se crea en STARTED y la evaluacion comienza.
- Misma huella y misma corrida, en STARTED, sin archivo de predicciones: se
  REANUDA y se vuelve a inferir (la inferencia es determinista; las
  predicciones se escriben de forma atomica, asi que no hay archivos a medias).
- Misma huella y misma corrida, en STARTED, CON archivo de predicciones (la
  interrupcion ocurrio entre escribirlo y actualizar el registro): se
  RECUPERA ese archivo. ``evaluate_test`` valida su contenido contra los
  segmentos de test antes de usarlo y, si no concuerda, exige revision manual.
- Misma huella y misma corrida, en PREDICTIONS_WRITTEN: se REANUDA
  reutilizando las predicciones guardadas, previa comprobacion de su hash.
- COMPLETED, otra huella u otra corrida: se BLOQUEA.
- Estados inconsistentes (PREDICTIONS_WRITTEN sin archivo o con otro hash):
  se detiene y exige revision manual.

Bloqueo por dataset: ``exclusive_lock`` crea ``<dataset>_int8.lock`` con
creacion exclusiva y lo borra al terminar (tambien si hay una excepcion o
Ctrl+C). Si ya existe, otra evaluacion de ese dataset esta en curso, o una
anterior murio sin liberarlo (corte de luz, ``kill -9``): se bloquea y el
mensaje indica el proceso que lo creo. Solo se borra a mano tras confirmar
que ese proceso ya no existe.
"""

from __future__ import annotations

import json
import os
import socket
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .bundle import sha256_file

STARTED = "STARTED"
PREDICTIONS_WRITTEN = "PREDICTIONS_WRITTEN"
COMPLETED = "COMPLETED"

RESUME_NEW = "new"
RESUME_REINFER = "resume_reinfer"
RESUME_REUSE_PREDICTIONS = "resume_reuse_predictions"
RESUME_RECOVER_PREDICTIONS = "resume_recover_predictions"


class TestAccessBlocked(RuntimeError):
    """La evaluacion pedida es nueva, distinta o ya se completo."""


class TestAccessInconsistent(RuntimeError):
    """El registro y los archivos no concuerdan: requiere revision manual."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def registry_path(access_dir: Path, dataset: str) -> Path:
    return Path(access_dir) / f"{dataset}_int8.json"


def lock_path(access_dir: Path, dataset: str) -> Path:
    return Path(access_dir) / f"{dataset}_int8.lock"


@contextmanager
def exclusive_lock(access_dir: Path, dataset: str):
    """Impide dos evaluaciones simultaneas del mismo dataset."""
    path = lock_path(access_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    info = {"dataset": dataset, "pid": os.getpid(), "host": socket.gethostname(), "created_at_utc": _now()}
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = path.read_text(encoding="utf-8").strip()
        except OSError:
            holder = "?"
        raise TestAccessBlocked(
            f"{dataset}: ya hay una evaluacion INT8 en curso o una anterior termino sin liberar el bloqueo "
            f"({path}: {holder}). Si ese proceso ya no existe, borre el archivo y repita el mismo comando"
        ) from None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(info, ensure_ascii=False) + "\n")
        yield path
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def read_record(access_dir: Path, dataset: str) -> dict | None:
    path = registry_path(access_dir, dataset)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write(access_dir: Path, dataset: str, record: dict) -> None:
    path = registry_path(access_dir, dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def begin(access_dir: Path, dataset: str, fingerprint: dict, run_dir: Path, predictions_path: Path) -> str:
    """Abre o reanuda la evaluacion. Devuelve ``RESUME_NEW``,
    ``RESUME_REINFER``, ``RESUME_RECOVER_PREDICTIONS`` o
    ``RESUME_REUSE_PREDICTIONS``. Debe llamarse dentro de ``exclusive_lock``."""
    run_dir = str(Path(run_dir).resolve())
    record = read_record(access_dir, dataset)
    if record is None:
        _write(access_dir, dataset, {
            "dataset": dataset, "status": STARTED, "fingerprint": fingerprint, "run_dir": run_dir,
            "predictions_file": str(Path(predictions_path).resolve()), "predictions_sha256": None,
            "created_at_utc": _now(), "updated_at_utc": _now(),
            "note": ("El test externo ya se uso para evaluar el modelo .pt en la corrida final; "
                     "este registro solo impide repetir la evaluacion INT8."),
        })
        return RESUME_NEW

    status = record.get("status")
    if status == COMPLETED:
        raise TestAccessBlocked(
            f"{dataset}: la evaluacion INT8 del test ya se completo ({registry_path(access_dir, dataset)}); "
            "no se repite"
        )
    if record.get("fingerprint") != fingerprint:
        differing = sorted(k for k in set(fingerprint) | set(record.get("fingerprint", {}))
                           if fingerprint.get(k) != record.get("fingerprint", {}).get(k))
        raise TestAccessBlocked(
            f"{dataset}: ya existe una evaluacion INT8 iniciada con otra huella (difiere en {differing}); "
            "solo se puede reanudar esa misma evaluacion"
        )
    if record.get("run_dir") != run_dir:
        raise TestAccessBlocked(
            f"{dataset}: la evaluacion INT8 se inicio en otra corrida ({record.get('run_dir')}); "
            "reanudela desde esa corrida"
        )
    if status == STARTED:
        if Path(record["predictions_file"]).is_file():
            # Se escribio el archivo pero no se llego a actualizar el registro.
            return RESUME_RECOVER_PREDICTIONS
        return RESUME_REINFER
    if status == PREDICTIONS_WRITTEN:
        path = Path(record["predictions_file"])
        if not path.is_file():
            raise TestAccessInconsistent(
                f"{dataset}: el registro indica PREDICTIONS_WRITTEN pero no existe {path}; requiere revision manual"
            )
        if sha256_file(path) != record.get("predictions_sha256"):
            raise TestAccessInconsistent(
                f"{dataset}: {path} no coincide con el hash registrado; requiere revision manual"
            )
        return RESUME_REUSE_PREDICTIONS
    raise TestAccessInconsistent(f"{dataset}: estado de registro desconocido {status!r}")


def mark_predictions_written(access_dir: Path, dataset: str, predictions_path: Path) -> None:
    record = read_record(access_dir, dataset)
    if record is None or record.get("status") not in (STARTED, PREDICTIONS_WRITTEN):
        raise TestAccessInconsistent(f"{dataset}: no hay una evaluacion INT8 abierta que actualizar")
    record["status"] = PREDICTIONS_WRITTEN
    record["predictions_file"] = str(Path(predictions_path).resolve())
    record["predictions_sha256"] = sha256_file(predictions_path)
    record["updated_at_utc"] = _now()
    _write(access_dir, dataset, record)


def mark_completed(access_dir: Path, dataset: str, outputs: dict) -> None:
    record = read_record(access_dir, dataset)
    if record is None or record.get("status") != PREDICTIONS_WRITTEN:
        raise TestAccessInconsistent(f"{dataset}: solo se completa una evaluacion con predicciones escritas")
    record["status"] = COMPLETED
    record["outputs"] = outputs
    record["completed_at_utc"] = _now()
    record["updated_at_utc"] = _now()
    _write(access_dir, dataset, record)
