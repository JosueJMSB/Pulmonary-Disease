"""Flujo WAV -> entrada del modelo, sin PyTorch.

Reproduce, para UNA grabacion, el mismo preprocesamiento de la etapa final
del protocolo holdout (``preprocessing/fold_denoising.py::run_final_v1`` y
``process_fold_recordings``), reutilizando sus funciones puras:

1. Remuestreo a 4 kHz con el FIR antialiasing de la fase 2
   (``phase2_standardization.get_filter`` + ``resample_poly``). La fase 2
   guardaba el audio remuestreado en WAV de coma flotante de 32 bits; aqui se
   reproduce ese redondeo (``intermediate_float32``).
2. Pasa-banda Butterworth de fase cero (``phase3_cleaning.apply_bandpass``).
3. Reduccion de ruido por sustraccion espectral, solo en la rama ``dn``
   (``phase3_cleaning.spectral_subtract``).
4. Normalizacion de amplitud con ``target_rms``, ``max_gain`` y
   ``peak_ceiling`` (``phase3_cleaning.normalize``).
5. Segmentos de 5 s con 50 % de solape (rejilla de ``phase4_temporal``).
6. Log-mel de cada segmento (``modeling/features/logmel.py``) y
   normalizacion por banda mel ``(x - mean) / std`` en float32, igual que
   ``modeling.models.cnn.LogmelDataset``.

Todos los parametros salen del paquete ``model_bundle.json``. Como las
funciones de ``preprocessing/`` leen algunas constantes de su ``config.py``,
``verify_runtime_matches`` comprueba que esas constantes coinciden con el
paquete antes de procesar nada.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSING_DIR = REPO_ROOT / "preprocessing"

PREPROCESSING_CODE_FILES = (
    "preprocessing/config.py",
    "preprocessing/utils.py",
    "preprocessing/phase2_standardization.py",
    "preprocessing/phase3_cleaning.py",
    "preprocessing/phase4_temporal.py",
    "modeling/features/logmel.py",
)

REQUIRED_PREPROCESSING_PARAMS = (
    "nperseg", "noverlap", "noise_pct", "oversubtraction", "spectral_floor",
    "target_rms", "max_gain", "peak_ceiling",
    "bandpass_low_hz", "bandpass_high_hz", "bandpass_order",
)


class PipelineConfigError(RuntimeError):
    """El paquete y el codigo de preprocessing/ no describen el mismo flujo."""


class RecordingTooShortError(ValueError):
    """La grabacion no alcanza para un solo segmento de 5 s."""


_MODULES: SimpleNamespace | None = None


def preprocessing_modules() -> SimpleNamespace:
    """Importa los modulos de ``preprocessing/`` con el mismo ``sys.path`` que
    usan sus propios scripts (``fold_denoising.py``)."""
    global _MODULES
    if _MODULES is None:
        if str(PREPROCESSING_DIR) not in sys.path:
            sys.path.insert(0, str(PREPROCESSING_DIR))
        import config as pcfg  # noqa: E402  (modulo de preprocessing/)
        import phase2_standardization as p2
        import phase3_cleaning as p3
        import phase4_temporal as p4
        import utils as pu

        _MODULES = SimpleNamespace(cfg=pcfg, p2=p2, p3=p3, p4=p4, u=pu)
    return _MODULES


def preprocessing_code_sha256() -> str:
    digest = hashlib.sha256()
    for rel in PREPROCESSING_CODE_FILES:
        path = REPO_ROOT / rel
        digest.update(rel.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Bloque "preprocessing" del paquete
# ---------------------------------------------------------------------------

def build_preprocessing_block(params: dict, branch: str) -> dict:
    """Construye ``bundle["preprocessing"]`` a partir del
    ``preprocessing_params.json`` de la etapa final y de las constantes de
    ``preprocessing/config.py``."""
    m = preprocessing_modules()
    cfg = m.cfg
    missing = [k for k in REQUIRED_PREPROCESSING_PARAMS if k not in params]
    if missing:
        raise PipelineConfigError(f"preprocessing_params.json no contiene: {missing}")
    mismatches = []
    if int(params["bandpass_low_hz"]) != int(cfg.BANDPASS_LOW):
        mismatches.append("bandpass_low_hz")
    if int(params["bandpass_high_hz"]) != int(cfg.BANDPASS_HIGH):
        mismatches.append("bandpass_high_hz")
    if int(params["bandpass_order"]) != int(cfg.BANDPASS_ORDER):
        mismatches.append("bandpass_order")
    if mismatches:
        raise PipelineConfigError(
            f"preprocessing_params.json y preprocessing/config.py difieren en {mismatches}: "
            "apply_bandpass usa las constantes de config.py"
        )
    if branch not in ("dn", "no_dn"):
        raise PipelineConfigError(f"rama desconocida: {branch!r}")

    return {
        "target_sr": int(cfg.TARGET_SR),
        "resample_ratios": {str(k): [int(v[0]), int(v[1])] for k, v in cfg.RESAMPLE_RATIOS.items()},
        "antialias": {
            "passband_hz": float(cfg.ANTIALIAS_PASSBAND_HZ),
            "stopband_hz": float(cfg.ANTIALIAS_STOPBAND_HZ),
            "cutoff_hz": float(cfg.ANTIALIAS_CUTOFF_HZ),
            "ripple_db": float(cfg.ANTIALIAS_RIPPLE_DB),
            "padtype": str(cfg.ANTIALIAS_PADTYPE),
        },
        "intermediate_float32": True,
        "bandpass": {
            "low_hz": float(params["bandpass_low_hz"]),
            "high_hz": float(params["bandpass_high_hz"]),
            "order": int(params["bandpass_order"]),
            "zero_phase": True,
        },
        "denoising": {
            "enabled": branch == "dn",
            "window": str(cfg.STFT_WINDOW),
            "nperseg": int(params["nperseg"]),
            "noverlap": int(params["noverlap"]),
            "noise_pct": float(params["noise_pct"]),
            "oversubtraction": float(params["oversubtraction"]),
            "spectral_floor": float(params["spectral_floor"]),
        },
        "amplitude": {
            "target_rms": float(params["target_rms"]),
            "max_gain": float(params["max_gain"]),
            "peak_ceiling": float(params["peak_ceiling"]),
        },
        "segmentation": {
            "segment_seconds": float(cfg.SEGMENT_SECONDS),
            "overlap": float(cfg.SEGMENT_OVERLAP),
            "window_samples": int(m.p4.WINDOW_SAMPLES),
            "hop_samples": int(m.p4.HOP_SAMPLES),
        },
        "calibration_counts": {
            "denoising_fit_patients": len(params.get("denoising_fit_patient_ids", [])),
            "target_rms_fit_patients": len(params.get("target_rms_fit_patient_ids", [])),
        },
        "code_sha256": preprocessing_code_sha256(),
    }


def verify_runtime_matches(block: dict) -> list[str]:
    """Lanza ``PipelineConfigError`` si las constantes de ``preprocessing/``
    que usan las funciones reutilizadas no coinciden con el paquete. Devuelve
    avisos no bloqueantes (por ejemplo, otro hash de codigo)."""
    m = preprocessing_modules()
    cfg = m.cfg
    problems = []
    if int(block["target_sr"]) != int(cfg.TARGET_SR):
        problems.append("target_sr")
    expected_ratios = {str(k): [int(v[0]), int(v[1])] for k, v in cfg.RESAMPLE_RATIOS.items()}
    if block["resample_ratios"] != expected_ratios:
        problems.append("resample_ratios")
    aa = block["antialias"]
    for key, const in (("passband_hz", cfg.ANTIALIAS_PASSBAND_HZ), ("stopband_hz", cfg.ANTIALIAS_STOPBAND_HZ),
                       ("cutoff_hz", cfg.ANTIALIAS_CUTOFF_HZ), ("ripple_db", cfg.ANTIALIAS_RIPPLE_DB)):
        if float(aa[key]) != float(const):
            problems.append(f"antialias.{key}")
    if aa["padtype"] != cfg.ANTIALIAS_PADTYPE:
        problems.append("antialias.padtype")
    bp = block["bandpass"]
    if (float(bp["low_hz"]), float(bp["high_hz"]), int(bp["order"])) != (
            float(cfg.BANDPASS_LOW), float(cfg.BANDPASS_HIGH), int(cfg.BANDPASS_ORDER)):
        problems.append("bandpass")
    if block["denoising"]["window"] != cfg.STFT_WINDOW:
        problems.append("denoising.window")
    seg = block["segmentation"]
    if (int(seg["window_samples"]), int(seg["hop_samples"])) != (int(m.p4.WINDOW_SAMPLES), int(m.p4.HOP_SAMPLES)):
        problems.append("segmentation")
    if problems:
        raise PipelineConfigError(f"el paquete no coincide con preprocessing/config.py en: {problems}")
    warnings = []
    if block.get("code_sha256") and block["code_sha256"] != preprocessing_code_sha256():
        warnings.append(
            "el codigo de preprocessing/ o de modeling/features/logmel.py cambio desde que se creo el paquete"
        )
    return warnings


# ---------------------------------------------------------------------------
# Procesamiento
# ---------------------------------------------------------------------------

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    return preprocessing_modules().u.read_audio(path)


def process_waveform(x: np.ndarray, sr: int, block: dict) -> tuple[np.ndarray, dict]:
    """Senal original -> segmentos ``(n, window_samples)`` en float32."""
    m = preprocessing_modules()
    x = np.asarray(x, dtype=np.float64)
    target_sr = int(block["target_sr"])
    if int(sr) == target_sr:
        y = x
    else:
        if str(int(sr)) not in block["resample_ratios"]:
            raise PipelineConfigError(
                f"frecuencia de origen no contemplada: {sr} Hz "
                f"(admitidas: {sorted(block['resample_ratios'])} y {target_sr})"
            )
        filt = m.p2.get_filter(int(sr))
        y = m.p2.resample_poly(x, up=filt["up"], down=filt["down"], window=filt["coefficients"],
                               padtype=block["antialias"]["padtype"])
    if block["intermediate_float32"]:
        y = y.astype(np.float32).astype(np.float64)

    x_bp = m.p3.apply_bandpass(y)
    dn = block["denoising"]
    if dn["enabled"]:
        x_proc = m.p3.spectral_subtract(
            x_bp, int(dn["nperseg"]), int(dn["noverlap"]),
            float(dn["noise_pct"]), float(dn["oversubtraction"]), float(dn["spectral_floor"]),
        )
    else:
        x_proc = x_bp
    amp = block["amplitude"]
    y_norm, gain, limiter = m.p3.normalize(
        x_proc, float(amp["target_rms"]), float(amp["peak_ceiling"]), float(amp["max_gain"]),
    )
    if not np.isfinite(y_norm).all():
        raise FloatingPointError("NaN/Inf tras normalizar la grabacion")
    y32 = np.asarray(y_norm, dtype=np.float32)

    window = int(block["segmentation"]["window_samples"])
    hop = int(block["segmentation"]["hop_samples"])
    n = int(m.p4.segments_per_recording(y32.size, window, hop))
    if n == 0:
        raise RecordingTooShortError(
            f"la grabacion dura {y32.size / target_sr:.2f} s; se necesitan al menos {window / target_sr:.2f} s"
        )
    segments = np.stack([y32[k * hop:k * hop + window] for k in range(n)]).astype(np.float32)
    info = {
        "source_sr": int(sr), "n_samples_4k": int(y32.size), "duration_s": float(y32.size / target_sr),
        "n_segments": n, "gain": float(gain), "limiter": str(limiter),
    }
    return segments, info


def logmel_from_segments(segments: np.ndarray, acoustic_cfg: dict, logmel_cfg: dict) -> np.ndarray:
    """Log-mel ``(n, 1, n_mels, frames)`` en float32, igual que
    ``modeling.data.extract_or_load_logmel``."""
    from modeling.features import logmel as fl

    cfg = {"acoustic": acoustic_cfg, "logmel": logmel_cfg}
    n_mels = int(logmel_cfg["n_mels"])
    frames = int(acoustic_cfg["expected_frames"])
    out = np.empty((len(segments), 1, n_mels, frames), dtype=np.float32)
    for i in range(len(segments)):
        magnitude, _ = fl.compute_stft_magnitude(np.asarray(segments[i], dtype=np.float64), acoustic_cfg)
        out[i, 0] = fl.compute_logmel_db(magnitude, cfg).astype(np.float32)
    return out


def normalize_logmel(logmel: np.ndarray, mean, std) -> np.ndarray:
    """``(x - mean) / std`` por banda mel, en float32 (como ``LogmelDataset``)."""
    mean32 = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1, 1)
    std32 = np.asarray(std, dtype=np.float32).reshape(1, 1, -1, 1)
    return ((np.asarray(logmel, dtype=np.float32) - mean32) / std32).astype(np.float32)


def wav_to_model_input(path: Path, bundle: dict) -> tuple[np.ndarray, dict]:
    """WAV -> entrada normalizada del modelo ``(n, 1, n_mels, frames)``."""
    x, sr = read_wav(path)
    segments, info = process_waveform(x, sr, bundle["preprocessing"])
    logmel = logmel_from_segments(segments, bundle["acoustic_config"], bundle["logmel_config"])
    norm = bundle["normalization"]
    return normalize_logmel(logmel, norm["mean"], norm["std"]), info
