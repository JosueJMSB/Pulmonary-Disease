"""Configuracion y rutas del Objetivo 3.

Orden de resolucion de cada ruta: argumento de la CLI, luego variable de
entorno y por ultimo el valor por defecto de ``config.toml`` (relativo a la
raiz del repositorio). No hay rutas personales en el codigo.

Disposicion de las salidas::

    <output_root>/
    |-- _test_access/<dataset>_int8.json      registro comun a todas las corridas
    `-- <opt_run_id>/
        |-- run_config.json
        |-- <dataset>/                        model_fp32.onnx, model_int8.onnx,
        |                                     model_bundle.json, informes...
        |-- benchmark/<dataset>/
        `-- report/
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

ENV_DATA_ROOT = "PULMONARY_DATA_ROOT"
ENV_RUNS_ROOT = "PULMONARY_RUNS_ROOT"
ENV_OPT_ROOT = "PULMONARY_OPT_ROOT"

FINAL_TRAINING_SUBDIR = "final_training"


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["_config_path"] = str(config_path.resolve())
    return cfg


def canonical_sha256(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def acceptance_sha256(cfg: dict) -> str:
    """Hash del criterio de aceptacion en desarrollo (seccion ``[acceptance]``)."""
    return canonical_sha256(cfg["acceptance"])


def _resolve(cli_value, env_var: str, default_relative: str) -> Path:
    if cli_value is not None:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (REPO_ROOT / default_relative).resolve()


def resolve_data_root(cli_value, cfg: dict) -> Path:
    return _resolve(cli_value, ENV_DATA_ROOT, cfg["paths"]["default_data_root"])


def resolve_runs_root(cli_value, cfg: dict) -> Path:
    return _resolve(cli_value, ENV_RUNS_ROOT, cfg["paths"]["default_runs_root"])


def resolve_output_root(cli_value, cfg: dict, runs_root: Path | None = None) -> Path:
    if cli_value is not None:
        return Path(cli_value).expanduser().resolve()
    env_value = os.environ.get(ENV_OPT_ROOT)
    if env_value:
        return Path(env_value).expanduser().resolve()
    base = runs_root if runs_root is not None else resolve_runs_root(None, cfg)
    return (Path(base) / cfg["paths"]["default_output_subdir"]).resolve()


def resolve_final_run_dir(cli_value, cfg: dict, runs_root: Path) -> Path:
    if cli_value is not None:
        path = Path(cli_value).expanduser().resolve()
    else:
        path = Path(runs_root) / FINAL_TRAINING_SUBDIR / cfg["final_run"]["run_id"]
    if not path.is_dir():
        raise FileNotFoundError(f"no existe la corrida final: {path}")
    return path


def test_access_dir(output_root: Path, cfg: dict) -> Path:
    return Path(output_root) / cfg["paths"]["test_access_subdir"]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + secrets.token_hex(3)


def resolve_run_dir(run_dir, run_id, output_root: Path, create: bool) -> Path:
    """``--run-dir`` explicito, o ``<output_root>/<run_id>``. Con ``create`` y
    sin ninguno de los dos, crea una corrida nueva."""
    if run_dir is not None:
        path = Path(run_dir).expanduser().resolve()
    elif run_id is not None:
        path = Path(output_root) / run_id
    elif create:
        path = Path(output_root) / new_run_id()
    else:
        raise ValueError("indique --run-dir o --run-id de una corrida de optimization existente")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise FileNotFoundError(f"no existe la corrida de optimization: {path}")
    return path


def dataset_entries(cfg: dict, selected: str = "all") -> list[dict]:
    entries = list(cfg["datasets"])
    if selected == "all":
        return entries
    chosen = [e for e in entries if e["name"] == selected]
    if not chosen:
        names = [e["name"] for e in entries]
        raise ValueError(f"dataset desconocido: {selected!r} (disponibles: {names})")
    return chosen


def dataset_choices(cfg: dict) -> list[str]:
    return ["all"] + [e["name"] for e in cfg["datasets"]]


def dataset_dir(run_dir: Path, dataset: str) -> Path:
    return Path(run_dir) / dataset


def write_json_atomic(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv_atomic(frame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    frame.to_csv(tmp, index=False, lineterminator="\n")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_run_config(run_dir: Path, cfg: dict, extra: dict) -> None:
    """Guarda (o completa) ``run_config.json`` de la corrida."""
    path = Path(run_dir) / "run_config.json"
    current = read_json(path) if path.is_file() else {"created_at_utc": now_iso()}
    clean_cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    current["config"] = clean_cfg
    current["config_path"] = cfg.get("_config_path")
    current["config_sha256"] = canonical_sha256(clean_cfg)
    current["acceptance_sha256"] = acceptance_sha256(cfg)
    current.setdefault("invocations", []).append({"at_utc": now_iso(), **extra})
    write_json_atomic(current, path)
