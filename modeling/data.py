"""Configuracion, rutas, condiciones por experimento, cache de caracteristicas
y pesos por muestra.

Nada de este modulo entrena nada: prepara las matrices ``(n_segmentos, 188)``
y los metadatos que ``models/svm_rbf.py`` y ``run_experiment.py`` consumen.
"""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import features as feat

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "svm_rbf.toml"
BRANCHES = ("no_dn", "dn")


def load_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "rb") as fh:
        return tomllib.load(fh)


# ---------------------------------------------------------------------------
# Resolucion de rutas: CLI > variable de entorno > valor por defecto del TOML
# ---------------------------------------------------------------------------

def resolve_path(cli_value: str | Path | None, env_var: str, default_relative: str) -> Path:
    """CLI, luego variable de entorno, luego el valor por defecto del TOML.

    Ninguna ruta personal vive en el codigo: el valor por defecto es relativo
    a la raiz del repositorio.
    """
    if cli_value is not None:
        return Path(cli_value).resolve()
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value).resolve()
    return (REPO_ROOT / default_relative).resolve()


def resolve_data_root(cli_value: str | Path | None, cfg: dict) -> Path:
    return resolve_path(cli_value, "PULMONARY_DATA_ROOT", cfg["paths"]["default_data_root"])


def resolve_runs_root(cli_value: str | Path | None, cfg: dict) -> Path:
    return resolve_path(cli_value, "PULMONARY_RUNS_ROOT", cfg["paths"]["default_runs_root"])


def resolve_cache_root(cli_value: str | Path | None, cfg: dict) -> Path:
    return resolve_path(cli_value, "PULMONARY_CACHE_ROOT", cfg["paths"]["default_cache_root"])


# ---------------------------------------------------------------------------
# Condiciones por experimento (configs/svm_rbf.toml -> [experiments.*])
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConditionSpec:
    dataset: str
    condition: str
    branch: str
    dn_reliable_only: bool
    reused_from_main: bool = False
    # Solo CNN. ``augment`` activa SpecAugment en entrenamiento;
    # ``hyperparameters_from`` indica la condicion cuyo lr/dropout se reutiliza
    # fold a fold en vez de buscarlos. Con los valores por defecto, la SVM no
    # cambia en nada.
    augment: bool = False
    hyperparameters_from: str | None = None


def experiment_condition_specs(cfg: dict, experiment: str, dataset: str) -> list[ConditionSpec]:
    """Condiciones de un experimento para un dataset. La entrada del TOML puede
    ser una tabla unica (``main``) o una lista de tablas (las ablaciones);
    ambas formas se normalizan aqui a una lista."""
    entry = cfg["experiments"][experiment][dataset]
    entries = entry if isinstance(entry, list) else [entry]
    return [
        ConditionSpec(
            dataset=dataset,
            condition=e["condition"],
            branch=e["branch"],
            dn_reliable_only=bool(e["dn_reliable_only"]),
            reused_from_main=bool(e.get("reused_from_main", False)),
            augment=bool(e.get("augment", False)),
            hyperparameters_from=e.get("hyperparameters_from"),
        )
        for e in entries
    ]


def experiment_names(cfg: dict) -> list[str]:
    """Experimentos definidos en el TOML, en su orden de aparicion."""
    return list(cfg["experiments"].keys())


def all_condition_specs(cfg: dict, dataset: str) -> list[ConditionSpec]:
    """Union de las condiciones de todos los experimentos del TOML para un
    dataset, deduplicadas por (condicion, rama)."""
    seen: dict[tuple[str, str], ConditionSpec] = {}
    for experiment in experiment_names(cfg):
        if dataset not in cfg["experiments"][experiment]:
            continue
        for spec in experiment_condition_specs(cfg, experiment, dataset):
            seen[(spec.condition, spec.branch)] = spec
    return list(seen.values())


def dataset_names(cfg: dict) -> list[str]:
    return list(cfg["datasets"].keys())


# ---------------------------------------------------------------------------
# Carga de la salida de prepare_task_data.py
# ---------------------------------------------------------------------------

REQUIRED_SEGMENT_COLUMNS = {
    "task_array_index", "source_array_index", "segment_id", "audio_id",
    "patient_uid", "diagnosis", "target_label", "target_name",
    "calibration_patient", "dn_reliable",
}


def load_task_segments(data_root: Path, dataset: str) -> pd.DataFrame:
    path = Path(data_root) / dataset / "segments.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"no existe {path}. Ejecute primero modeling/prepare_task_data.py"
        )
    df = pd.read_csv(
        path, dtype={"audio_id": str, "patient_uid": str, "segment_id": str}
    )
    missing = REQUIRED_SEGMENT_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {sorted(missing)}")
    df["calibration_patient"] = df["calibration_patient"].astype(bool)
    df["dn_reliable"] = df["dn_reliable"].astype(bool)
    df["target_label"] = df["target_label"].astype(np.int64)
    return df


def load_task_manifest(data_root: Path, dataset: str) -> dict:
    path = Path(data_root) / dataset / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no existe {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_branch_array(data_root: Path, dataset: str, branch: str) -> np.ndarray:
    path = Path(data_root) / dataset / f"segments_{branch}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"no existe {path}")
    return np.load(path, mmap_mode="r")


def select_condition_segments(segments: pd.DataFrame, spec: ConditionSpec) -> pd.DataFrame:
    """Filtra por fiabilidad del denoising cuando la condicion lo exige."""
    df = segments
    if spec.dn_reliable_only:
        df = df.loc[df["dn_reliable"]]
    df = df.reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{spec.dataset}/{spec.condition}: la seleccion quedo vacia")
    return df


# ---------------------------------------------------------------------------
# Cache de caracteristicas: por dataset y RAMA, no por condicion. main y la
# ablacion de ICBHI comparten la rama no_dn; cachear por rama evita repetir
# la extraccion entre condiciones que solo difieren en que filas seleccionan.
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(cfg: dict) -> str:
    relevant = {k: cfg[k] for k in ("acoustic", "logmel", "mfcc", "summary") if k in cfg}
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_cache_dir(cache_root: Path, dataset: str, branch: str) -> Path:
    return Path(cache_root) / dataset / branch


def _feature_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "X": cache_dir / "X.npy",
        "rows": cache_dir / "feature_rows.csv",
        "schema": cache_dir / "feature_schema.json",
        "manifest": cache_dir / "feature_manifest.json",
    }


FEATURE_ROW_COLUMNS = [
    "task_array_index", "segment_id", "audio_id", "patient_uid",
    "diagnosis", "target_label", "target_name",
]


def extract_or_load_features(
    data_root: Path,
    cache_root: Path,
    dataset: str,
    branch: str,
    cfg: dict,
    force: bool = False,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Caracteristicas (n, 188) de TODOS los segmentos de un dataset/rama.

    Reutiliza la cache si los hashes de ``segments.csv`` y del ``.npy``, y la
    huella de la configuracion acustica, no cambiaron desde que se escribio;
    ``force=True`` la regenera sin comprobar nada.
    """
    segments = load_task_segments(data_root, dataset)
    csv_path = Path(data_root) / dataset / "segments.csv"
    npy_path = Path(data_root) / dataset / f"segments_{branch}.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(f"no existe {npy_path}")

    manifest = {
        "dataset": dataset,
        "branch": branch,
        "segments_csv_sha256": sha256_file(csv_path),
        f"segments_{branch}_npy_sha256": sha256_file(npy_path),
        "config_fingerprint": _config_fingerprint(cfg),
        "n_features": feat.N_FEATURES,
        "feature_names": list(feat.FEATURE_NAMES),
    }
    compare_keys = {
        "segments_csv_sha256", f"segments_{branch}_npy_sha256",
        "config_fingerprint", "n_features",
    }

    cache_dir = feature_cache_dir(cache_root, dataset, branch)
    paths = _feature_cache_paths(cache_dir)

    if not force and all(p.is_file() for p in paths.values()):
        stored = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if all(stored.get(k) == manifest[k] for k in compare_keys):
            schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
            rows = pd.read_csv(
                paths["rows"],
                dtype={"audio_id": str, "patient_uid": str, "segment_id": str},
            )
            X = np.load(paths["X"])
            valid = (
                X.shape == (len(rows), feat.N_FEATURES)
                and schema.get("feature_names") == list(feat.FEATURE_NAMES)
            )
            if valid:
                return X, rows, list(feat.FEATURE_NAMES)

    array = load_branch_array(data_root, dataset, branch)
    if array.shape[0] != len(segments):
        raise ValueError(
            f"{dataset}/{branch}: {array.shape[0]} filas en el .npy, "
            f"{len(segments)} en segments.csv"
        )

    ordered = segments.sort_values("task_array_index").reset_index(drop=True)
    if not np.array_equal(ordered["task_array_index"].to_numpy(), np.arange(len(ordered))):
        raise ValueError(f"{dataset}: task_array_index no es 0..n-1 consecutivo")

    n = len(ordered)
    X = np.empty((n, feat.N_FEATURES), dtype=np.float32)
    for i in range(n):
        segment = np.asarray(array[i], dtype=np.float64)
        X[i] = feat.extract_segment_features(segment, cfg)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_X = paths["X"].with_name(paths["X"].name + ".part")
    # np.save() le anade ".npy" al nombre si no termina ya en eso: pasado un
    # Path que termina en ".part" (no ".npy"), escribiria "X.npy.part.npy" y
    # el os.replace() de abajo fallaria buscando "X.npy.part". Abrir el
    # archivo en modo binario y pasar el file object evita esa logica: numpy
    # nunca toca el nombre de un file object ya abierto.
    with open(tmp_X, "wb") as fh:
        np.save(fh, X)
    os.replace(tmp_X, paths["X"])

    ordered[FEATURE_ROW_COLUMNS].to_csv(paths["rows"], index=False, lineterminator="\n")

    schema = {"feature_names": list(feat.FEATURE_NAMES), "n_features": feat.N_FEATURES}
    paths["schema"].write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest["n_segments"] = n
    manifest["output_X_sha256"] = sha256_file(paths["X"])
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return X, ordered[FEATURE_ROW_COLUMNS], list(feat.FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Datos de una condicion: segmentos filtrados + su matriz de caracteristicas
# ---------------------------------------------------------------------------

@dataclass
class ConditionData:
    spec: ConditionSpec
    segments: pd.DataFrame
    X: np.ndarray
    feature_names: list[str]


def build_condition_data(
    data_root: Path,
    cache_root: Path,
    spec: ConditionSpec,
    cfg: dict,
    force_features: bool = False,
) -> ConditionData:
    full_X, feature_rows, feature_names = extract_or_load_features(
        data_root, cache_root, spec.dataset, spec.branch, cfg, force=force_features,
    )
    segments = load_task_segments(data_root, spec.dataset)
    segments = segments.sort_values("task_array_index").reset_index(drop=True)

    if not segments["segment_id"].equals(feature_rows["segment_id"]):
        raise RuntimeError(
            f"{spec.dataset}/{spec.branch}: segments.csv y la cache de "
            "caracteristicas no estan alineados fila a fila"
        )

    mask = (
        segments["dn_reliable"].to_numpy()
        if spec.dn_reliable_only
        else np.ones(len(segments), dtype=bool)
    )
    filtered_segments = segments.loc[mask].reset_index(drop=True)
    filtered_X = full_X[mask]

    if filtered_segments.empty:
        raise ValueError(f"{spec.dataset}/{spec.condition}: la seleccion quedo vacia")

    return ConditionData(spec=spec, segments=filtered_segments, X=filtered_X, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Pesos por muestra: w_s = 1 / (N_c * R_p * S_r), normalizados a media 1.
# ---------------------------------------------------------------------------

def select_rows_by_patients(
    segments: pd.DataFrame, X: np.ndarray, patient_ids: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subconjunto de ``segments``/``X`` (alineados fila a fila) cuyos
    pacientes estan en ``patient_ids``."""
    if len(segments) != len(X):
        raise ValueError(f"segments ({len(segments)} filas) y X ({len(X)} filas) no coinciden")
    mask = segments["patient_uid"].isin(set(patient_ids)).to_numpy()
    return segments.loc[mask].reset_index(drop=True), X[mask]


def compute_sample_weights(segments: pd.DataFrame) -> np.ndarray:
    """Pondera cada segmento para que cada clase y cada paciente contribuyan
    lo mismo al ajuste, sin importar cuantos pacientes, grabaciones o
    segmentos aporte cada uno.

    N_c, R_p y S_r se cuentan dentro de ``segments`` -el conjunto que se va a
    pesar (train de un fold, o train+validation al reajustar, o toda la
    condicion al entrenar el modelo final)-, nunca sobre el corpus completo:
    el peso es siempre relativo al conjunto que efectivamente se ajusta.
    """
    required = {"patient_uid", "audio_id", "target_label"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"faltan columnas: {sorted(missing)}")
    if segments.empty:
        raise ValueError("no se pueden calcular pesos sobre un conjunto vacio")

    n_c = segments.drop_duplicates("patient_uid").groupby("target_label")["patient_uid"].size()
    r_p = segments.drop_duplicates("audio_id").groupby("patient_uid")["audio_id"].size()
    s_r = segments.groupby("audio_id")["segment_id"].transform("size")

    class_count = segments["target_label"].map(n_c).to_numpy(dtype=np.float64)
    patient_recordings = segments["patient_uid"].map(r_p).to_numpy(dtype=np.float64)
    recording_segments = s_r.to_numpy(dtype=np.float64)

    weights = 1.0 / (class_count * patient_recordings * recording_segments)
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("los pesos calculados no son validos (media no finita o <= 0)")
    return weights / mean_weight


# ---------------------------------------------------------------------------
# Cache Log-Mel para la CNN: (N, 1, n_mels, n_frames) float32 por dataset y
# RAMA, alineada fila a fila con segments.csv (fila i = task_array_index i).
# Reutiliza exactamente compute_stft_magnitude + compute_logmel_db de
# features/logmel.py: la CNN ve la misma representacion intermedia que la SVM.
# No importa torch: el entorno de la SVM sigue funcionando sin PyTorch.
# ---------------------------------------------------------------------------

CNN_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "cnn.toml"
CRNN_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "crnn.toml"
LOGMEL_CONFIG_SECTIONS = ("acoustic", "logmel")
NETWORK_ARCHITECTURES = ("cnn", "crnn")


def model_architecture(cfg: dict) -> str:
    """Red que describe un TOML de redes: ``[model] architecture``.

    cnn.toml no tiene esa seccion y se interpreta como ``"cnn"``, de modo que
    su configuracion (y su huella de ejecucion) no cambia. Sin dependencia de
    torch, para que run_experiment pueda validarlo antes de importar PyTorch.
    """
    name = str(cfg.get("model", {}).get("architecture", "cnn"))
    if name not in NETWORK_ARCHITECTURES:
        raise ValueError(f"[model] architecture desconocida: {name!r} (use {NETWORK_ARCHITECTURES})")
    return name
LOGMEL_COMPARE_KEYS = ("segments_csv_sha256", "segments_npy_sha256", "config_fingerprint", "shape", "dtype")


def logmel_config_fingerprint(cfg: dict) -> str:
    relevant = {k: cfg[k] for k in LOGMEL_CONFIG_SECTIONS}
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logmel_cache_dir(cache_root: Path, dataset: str, branch: str) -> Path:
    return Path(cache_root) / dataset / branch


def logmel_cache_paths(cache_dir: Path) -> dict[str, Path]:
    return {
        "logmel": cache_dir / "logmel.npy",
        "rows": cache_dir / "logmel_rows.csv",
        "schema": cache_dir / "logmel_schema.json",
        "manifest": cache_dir / "logmel_manifest.json",
    }


def logmel_shape(cfg: dict, n_segments: int) -> tuple[int, int, int, int]:
    return (int(n_segments), 1, int(cfg["logmel"]["n_mels"]), int(cfg["acoustic"]["expected_frames"]))


def _logmel_expected_manifest(data_root: Path, dataset: str, branch: str, cfg: dict, n_segments: int) -> dict:
    csv_path = Path(data_root) / dataset / "segments.csv"
    npy_path = Path(data_root) / dataset / f"segments_{branch}.npy"
    if not npy_path.is_file():
        raise FileNotFoundError(f"no existe {npy_path}")
    return {
        "dataset": dataset,
        "branch": branch,
        "segments_csv_sha256": sha256_file(csv_path),
        "segments_npy_sha256": sha256_file(npy_path),
        "config_fingerprint": logmel_config_fingerprint(cfg),
        "shape": list(logmel_shape(cfg, n_segments)),
        "dtype": "float32",
    }


def _ordered_segments(data_root: Path, dataset: str) -> pd.DataFrame:
    segments = load_task_segments(data_root, dataset)
    ordered = segments.sort_values("task_array_index").reset_index(drop=True)
    if not np.array_equal(ordered["task_array_index"].to_numpy(), np.arange(len(ordered))):
        raise ValueError(f"{dataset}: task_array_index no es 0..n-1 consecutivo")
    return ordered


def logmel_cache_status(data_root: Path, cache_root: Path, dataset: str, branch: str, cfg: dict) -> tuple[str, str]:
    """``("valida" | "ausente" | "desactualizada", detalle)`` sin extraer nada.

    Una cache ausente o desactualizada no es un error: la ejecucion la
    regenera. --dry-run solo informa en que estado esta.
    """
    ordered = _ordered_segments(data_root, dataset)
    paths = logmel_cache_paths(logmel_cache_dir(cache_root, dataset, branch))
    if not all(p.is_file() for p in paths.values()):
        return "ausente", "se generara al ejecutar"
    expected = _logmel_expected_manifest(data_root, dataset, branch, cfg, len(ordered))
    stored = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    changed = [k for k in LOGMEL_COMPARE_KEYS if stored.get(k) != expected[k]]
    if changed:
        return "desactualizada", "cambio: " + ", ".join(changed)
    if stored.get("output_sha256") != sha256_file(paths["logmel"]):
        return "desactualizada", "logmel.npy no coincide con el hash registrado"
    return "valida", f"forma {tuple(expected['shape'])}"


def extract_or_load_logmel(
    data_root: Path,
    cache_root: Path,
    dataset: str,
    branch: str,
    cfg: dict,
    force: bool = False,
    progress=None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Log-Mel ``(N, 1, n_mels, n_frames)`` de TODOS los segmentos de un
    dataset/rama, cacheado en disco.

    Se reutiliza si coinciden los hashes de ``segments.csv`` y del ``.npy``,
    la huella de [acoustic]/[logmel], la forma y el hash del propio
    ``logmel.npy``. ``progress(hechos, total)`` se llama cada 500 segmentos.
    La escritura va a ``logmel.npy.part`` via ``open_memmap`` (que, a
    diferencia de ``np.save``, no altera el nombre) y se publica con
    ``os.replace`` solo al terminar.
    """
    import gc

    ordered = _ordered_segments(data_root, dataset)
    expected = _logmel_expected_manifest(data_root, dataset, branch, cfg, len(ordered))
    cache_dir = logmel_cache_dir(cache_root, dataset, branch)
    paths = logmel_cache_paths(cache_dir)

    if not force and all(p.is_file() for p in paths.values()):
        stored = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if (all(stored.get(k) == expected[k] for k in LOGMEL_COMPARE_KEYS)
                and stored.get("output_sha256") == sha256_file(paths["logmel"])):
            rows = pd.read_csv(paths["rows"], dtype={"audio_id": str, "patient_uid": str, "segment_id": str})
            logmel = np.load(paths["logmel"])
            if (logmel.shape == tuple(expected["shape"]) and logmel.dtype == np.float32
                    and rows["segment_id"].equals(ordered["segment_id"])):
                return logmel, rows

    array = load_branch_array(data_root, dataset, branch)
    segment_length = int(cfg["acoustic"]["segment_length"])
    if array.shape != (len(ordered), segment_length):
        raise ValueError(
            f"{dataset}/{branch}: .npy de forma {array.shape}, se esperaba ({len(ordered)}, {segment_length})"
        )

    shape = logmel_shape(cfg, len(ordered))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = paths["logmel"].with_name(paths["logmel"].name + ".part")
    out = np.lib.format.open_memmap(str(tmp), mode="w+", dtype=np.float32, shape=shape)
    try:
        for i in range(shape[0]):
            magnitude, _ = feat.logmel.compute_stft_magnitude(np.asarray(array[i], dtype=np.float64), cfg["acoustic"])
            out[i, 0] = feat.logmel.compute_logmel_db(magnitude, cfg).astype(np.float32)
            if progress is not None and ((i + 1) % 500 == 0 or i + 1 == shape[0]):
                progress(i + 1, shape[0])
        out.flush()
    finally:
        # El memmap debe cerrarse antes del os.replace (obligatorio en Windows).
        del out
        gc.collect()
    os.replace(tmp, paths["logmel"])

    ordered[FEATURE_ROW_COLUMNS].to_csv(paths["rows"], index=False, lineterminator="\n")
    schema = {
        "shape": list(shape),
        "dtype": "float32",
        "axes": ["segment", "channel", "mel_band", "frame"],
        "row_order": "task_array_index",
        "sample_rate": int(cfg["acoustic"]["sample_rate"]),
        "hop_length": int(cfg["acoustic"]["hop_length"]),
        "n_mels": int(cfg["logmel"]["n_mels"]),
        "fmin": cfg["logmel"]["fmin"],
        "fmax": cfg["logmel"]["fmax"],
        "units": "dB (power_to_db, ref=1.0)",
    }
    paths["schema"].write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = {**expected, "n_segments": int(shape[0]), "output_sha256": sha256_file(paths["logmel"])}
    paths["manifest"].write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return np.load(paths["logmel"]), ordered[FEATURE_ROW_COLUMNS]


@dataclass
class ConditionLogmel:
    """Segmentos de una condicion + el Log-Mel completo de su dataset/rama.

    No se copia el subconjunto: ``segments["cache_row"]`` indica la fila de
    cada segmento en ``logmel``, de modo que varias condiciones de la misma
    rama comparten un unico array en memoria.
    """

    spec: ConditionSpec
    segments: pd.DataFrame
    logmel: np.ndarray


def build_condition_logmel(
    data_root: Path, spec: ConditionSpec, logmel: np.ndarray, rows: pd.DataFrame,
) -> ConditionLogmel:
    ordered = _ordered_segments(data_root, spec.dataset)
    if not ordered["segment_id"].equals(rows["segment_id"]):
        raise RuntimeError(
            f"{spec.dataset}/{spec.branch}: segments.csv y la cache Log-Mel no estan alineados fila a fila"
        )
    if logmel.shape[0] != len(ordered):
        raise RuntimeError(f"{spec.dataset}/{spec.branch}: {logmel.shape[0]} filas Log-Mel para {len(ordered)} segmentos")
    ordered["cache_row"] = np.arange(len(ordered), dtype=np.int64)
    selected = select_condition_segments(ordered, spec)
    return ConditionLogmel(spec=spec, segments=selected, logmel=logmel)
