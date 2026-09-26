"""Extraccion de las 188 caracteristicas acusticas para la SVM-RBF.

El pipeline calcula una unica STFT por segmento (``logmel.compute_stft_magnitude``)
y la reutiliza tanto para el log-mel/MFCC (``logmel.py``) como para los ocho
descriptores espectrales y temporales (``acoustic.py``). Ninguno de los dos
modulos vuelve a llamar a ``librosa.stft``.

``extract_segment_features`` combina ambos y resume las 39 series MFCC/delta y
los 8 descriptores con media, desviacion poblacional, percentil 10 y percentil
90: 47 x 4 = 188 columnas, en un orden determinista (``FEATURE_NAMES``).
"""

from __future__ import annotations

import numpy as np

from . import acoustic, logmel

STAT_SUFFIXES = ("mean", "std", "p10", "p90")

MFCC_SERIES_NAMES = tuple(
    [f"mfcc_{i:02d}" for i in range(logmel.N_MFCC)]
    + [f"mfcc_delta_{i:02d}" for i in range(logmel.N_MFCC)]
    + [f"mfcc_delta2_{i:02d}" for i in range(logmel.N_MFCC)]
)

BASE_SERIES_NAMES = MFCC_SERIES_NAMES + acoustic.DESCRIPTOR_NAMES

FEATURE_NAMES = tuple(
    f"{name}_{stat}" for name in BASE_SERIES_NAMES for stat in STAT_SUFFIXES
)

N_FEATURES = len(FEATURE_NAMES)
assert N_FEATURES == 188, f"se esperaban 188 columnas, hay {N_FEATURES}"


def _summarize(series: np.ndarray) -> np.ndarray:
    """Media, desviacion poblacional, percentil 10 y percentil 90 de una serie.

    ``ddof=0`` (poblacional, no muestral) y percentiles con interpolacion
    lineal (comportamiento por defecto de ``numpy.percentile``), fijados aqui
    para que no dependan de valores por defecto que pudieran cambiar entre
    versiones de numpy.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.size == 0:
        return np.zeros(4, dtype=np.float64)
    return np.array(
        [
            float(np.mean(series)),
            float(np.std(series, ddof=0)),
            float(np.percentile(series, 10, method="linear")),
            float(np.percentile(series, 90, method="linear")),
        ],
        dtype=np.float64,
    )


def extract_segment_features(segment: np.ndarray, cfg: dict) -> np.ndarray:
    """Vector de 188 caracteristicas para un solo segmento de 20000 muestras.

    ``cfg`` es el diccionario ``acoustic``/``logmel``/``mfcc`` ya resuelto de
    ``configs/oof_v1/svm_rbf.toml`` (ver ``modeling.data.load_config``).
    """
    segment = np.asarray(segment, dtype=np.float64)
    expected = int(cfg["acoustic"]["segment_length"])
    if segment.shape != (expected,):
        raise ValueError(
            f"segmento de forma {segment.shape}, se esperaba ({expected},)"
        )

    magnitude, freqs = logmel.compute_stft_magnitude(segment, cfg["acoustic"])
    mfcc_stack = logmel.compute_mfcc_stack(magnitude, cfg)  # (39, n_frames)
    descriptors = acoustic.compute_descriptors(segment, magnitude, freqs, cfg)

    series_list = list(mfcc_stack) + descriptors
    if len(series_list) != len(BASE_SERIES_NAMES):
        raise RuntimeError(
            f"{len(series_list)} series calculadas, se esperaban "
            f"{len(BASE_SERIES_NAMES)}"
        )

    summaries = np.concatenate([_summarize(s) for s in series_list])
    if summaries.shape != (N_FEATURES,):
        raise RuntimeError(f"vector de forma {summaries.shape}, esperado ({N_FEATURES},)")
    if not np.isfinite(summaries).all():
        raise FloatingPointError("el vector de caracteristicas contiene NaN o Inf")
    return summaries.astype(np.float32)


def extract_batch_features(segments: np.ndarray, cfg: dict) -> np.ndarray:
    """Aplica ``extract_segment_features`` fila a fila sobre un array (n, L)."""
    segments = np.asarray(segments)
    out = np.empty((segments.shape[0], N_FEATURES), dtype=np.float32)
    for i in range(segments.shape[0]):
        out[i] = extract_segment_features(segments[i], cfg)
    return out


__all__ = [
    "FEATURE_NAMES",
    "BASE_SERIES_NAMES",
    "STAT_SUFFIXES",
    "N_FEATURES",
    "extract_segment_features",
    "extract_batch_features",
    "acoustic",
    "logmel",
]
