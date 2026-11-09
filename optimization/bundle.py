"""Paquete de inferencia independiente (``model_bundle.json``).

Contiene TODO lo necesario para ir de un WAV a una decision sin PyTorch ni el
checkpoint ``.pt``: normalizacion por banda mel, configuracion acustica y
log-mel, parametros de preprocesamiento de la etapa final (incluido
``peak_ceiling``), clases, umbral y regla de agregacion. Los modelos ONNX FP32
e INT8 se describen en bloques SEPARADOS (``models.fp32`` y ``models.int8``),
cada uno con su propio hash, tamano, versiones y metadatos.

Este modulo no importa torch ni onnxruntime.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

BUNDLE_FILENAME = "model_bundle.json"
BUNDLE_FORMAT_VERSION = 1
FP32_FILENAME = "model_fp32.onnx"
INT8_FILENAME = "model_int8.onnx"
VERSIONS = ("fp32", "int8")


class BundleError(ValueError):
    """El paquete esta incompleto, es inconsistente o sus archivos cambiaron."""


# Rutas obligatorias (claves anidadas) y su tipo esperado.
_REQUIRED = {
    ("format_version",): int,
    ("dataset",): str,
    ("condition",): str,
    ("branch",): str,
    ("input", "name"): str,
    ("input", "shape"): list,
    ("output", "name"): str,
    ("normalization", "mean"): list,
    ("normalization", "std"): list,
    ("acoustic_config",): dict,
    ("logmel_config",): dict,
    ("preprocessing", "target_sr"): int,
    ("preprocessing", "resample_ratios"): dict,
    ("preprocessing", "antialias", "passband_hz"): (int, float),
    ("preprocessing", "antialias", "stopband_hz"): (int, float),
    ("preprocessing", "antialias", "cutoff_hz"): (int, float),
    ("preprocessing", "antialias", "ripple_db"): (int, float),
    ("preprocessing", "antialias", "padtype"): str,
    ("preprocessing", "intermediate_float32"): bool,
    ("preprocessing", "bandpass", "low_hz"): (int, float),
    ("preprocessing", "bandpass", "high_hz"): (int, float),
    ("preprocessing", "bandpass", "order"): int,
    ("preprocessing", "denoising", "enabled"): bool,
    ("preprocessing", "denoising", "window"): str,
    ("preprocessing", "denoising", "nperseg"): int,
    ("preprocessing", "denoising", "noverlap"): int,
    ("preprocessing", "denoising", "noise_pct"): (int, float),
    ("preprocessing", "denoising", "oversubtraction"): (int, float),
    ("preprocessing", "denoising", "spectral_floor"): (int, float),
    ("preprocessing", "amplitude", "target_rms"): (int, float),
    ("preprocessing", "amplitude", "max_gain"): (int, float),
    ("preprocessing", "amplitude", "peak_ceiling"): (int, float),
    ("preprocessing", "segmentation", "window_samples"): int,
    ("preprocessing", "segmentation", "hop_samples"): int,
    ("classes", "names"): list,
    ("classes", "negative_label_name"): str,
    ("classes", "positive_label_name"): str,
    ("decision", "threshold"): (int, float),
    ("decision", "rule"): str,
    ("aggregation", "segment_to_recording"): str,
    ("aggregation", "recording_to_patient"): str,
    ("source", "final_run_id"): str,
    ("source", "checkpoint_sha256"): str,
    ("models", "fp32", "file"): str,
    ("models", "fp32", "sha256"): str,
    ("models", "fp32", "size_bytes"): int,
}

_REQUIRED_INT8 = {
    ("models", "int8", "file"): str,
    ("models", "int8", "sha256"): str,
    ("models", "int8", "size_bytes"): int,
    ("models", "int8", "quantization"): dict,
    ("models", "int8", "calibration", "inventory_sha256"): str,
    ("models", "int8", "calibration", "roles"): list,
}


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _get(obj: dict, keys: tuple):
    current = obj
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def validate_bundle(bundle: dict, require_int8: bool = False) -> None:
    """Rechaza un paquete al que le falte cualquier parametro de inferencia."""
    required = dict(_REQUIRED)
    if require_int8:
        required.update(_REQUIRED_INT8)
    problems = []
    for keys, expected in required.items():
        try:
            value = _get(bundle, keys)
        except KeyError:
            problems.append(f"falta {'.'.join(keys)}")
            continue
        if expected is bool:
            ok = isinstance(value, bool)
        elif expected is int:
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif expected == (int, float):
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        else:
            ok = isinstance(value, expected)
        if not ok:
            problems.append(f"{'.'.join(keys)} tiene tipo {type(value).__name__}")
    if bundle.get("format_version") not in (None, BUNDLE_FORMAT_VERSION) and not problems:
        problems.append(f"format_version {bundle.get('format_version')} no soportada")

    if not problems:
        n_mels = int(bundle["logmel_config"].get("n_mels", -1))
        if len(bundle["normalization"]["mean"]) != n_mels or len(bundle["normalization"]["std"]) != n_mels:
            problems.append("normalization.mean/std no tienen n_mels valores")
        if any(float(s) <= 0 for s in bundle["normalization"]["std"]):
            problems.append("normalization.std contiene valores <= 0")
        threshold = float(bundle["decision"]["threshold"])
        if not 0.0 < threshold < 1.0:
            problems.append("decision.threshold fuera de (0, 1)")
        if bundle["classes"]["positive_label_name"] != "COPD":
            problems.append("classes.positive_label_name debe ser COPD")
        peak = float(bundle["preprocessing"]["amplitude"]["peak_ceiling"])
        if not 0.0 < peak <= 1.0:
            problems.append("preprocessing.amplitude.peak_ceiling fuera de (0, 1]")
        if bundle["branch"] == "dn" and not bundle["preprocessing"]["denoising"]["enabled"]:
            problems.append("branch dn con denoising.enabled = false")
        if bundle["branch"] == "no_dn" and bundle["preprocessing"]["denoising"]["enabled"]:
            problems.append("branch no_dn con denoising.enabled = true")
        if require_int8 and bundle["models"]["int8"]["calibration"]["roles"] != ["train"]:
            problems.append("la calibracion INT8 solo puede usar el rol 'train' (desarrollo)")
        if (bundle.get("models", {}).get("int8") or {}).get("sha256") == bundle["models"]["fp32"]["sha256"]:
            problems.append("models.int8.sha256 es igual al de models.fp32")
    if problems:
        raise BundleError("paquete de inferencia invalido: " + "; ".join(problems))


def load_bundle(bundle_dir: Path, require_int8: bool = False) -> dict:
    path = Path(bundle_dir) / BUNDLE_FILENAME
    if not path.is_file():
        raise BundleError(f"no existe {path}")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    validate_bundle(bundle, require_int8=require_int8)
    return bundle


def write_bundle(bundle: dict, bundle_dir: Path, require_int8: bool = False) -> Path:
    validate_bundle(bundle, require_int8=require_int8)
    path = Path(bundle_dir) / BUNDLE_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def model_entry(bundle: dict, version: str) -> dict:
    if version not in VERSIONS:
        raise BundleError(f"version desconocida {version!r}; use una de {VERSIONS}")
    entry = (bundle.get("models") or {}).get(version)
    if not entry:
        raise BundleError(f"el paquete no tiene el bloque models.{version}")
    return entry


def model_path(bundle_dir: Path, bundle: dict, version: str) -> Path:
    return Path(bundle_dir) / model_entry(bundle, version)["file"]


def verify_model_file(bundle_dir: Path, bundle: dict, version: str) -> Path:
    """El archivo ONNX existe y su hash coincide con el declarado en su bloque."""
    entry = model_entry(bundle, version)
    path = Path(bundle_dir) / entry["file"]
    if not path.is_file():
        raise BundleError(f"no existe el modelo {version}: {path}")
    actual = sha256_file(path)
    if actual != entry["sha256"]:
        raise BundleError(
            f"el hash de {path.name} ({actual[:12]}...) no coincide con models.{version}.sha256 "
            f"({entry['sha256'][:12]}...)"
        )
    if path.stat().st_size != int(entry["size_bytes"]):
        raise BundleError(f"el tamano de {path.name} no coincide con models.{version}.size_bytes")
    return path


def bundle_file_sha256(bundle_dir: Path) -> str:
    return sha256_file(Path(bundle_dir) / BUNDLE_FILENAME)
