"""Ocho descriptores acusticos por trama, reutilizando la STFT compartida.

Los seis descriptores espectrales (centroide, roll-off, ancho de banda, flujo,
planitud, curtosis) se calculan sobre la magnitud de la STFT restringida a
50-1800 Hz, tal como exige el plan. La tasa de cruces por cero y el RMS son
descriptores de forma de onda -no espectrales- y se calculan sobre la senal
completa en el dominio del tiempo, con el mismo ``frame_length``/``hop_length``
que la STFT para que las 309 tramas queden alineadas con las demas series.

Todas las funciones usan un piso ``_EPS`` en los denominadores: una trama de
silencio, un tono puro o una senal constante no deben producir NaN/Inf.
"""

from __future__ import annotations

import numpy as np
import librosa

DESCRIPTOR_NAMES = (
    "spectral_centroid_hz",
    "spectral_rolloff85_hz",
    "spectral_bandwidth_hz",
    "spectral_flux",
    "spectral_flatness",
    "zero_crossing_rate",
    "spectral_kurtosis",
    "rms",
)

_EPS = 1e-12
_ROLLOFF_PERCENT = 0.85


def _band_mask(freqs: np.ndarray, low_hz: float, high_hz: float) -> np.ndarray:
    return (freqs >= low_hz) & (freqs <= high_hz)


def spectral_centroid(mag_band: np.ndarray, freqs_band: np.ndarray) -> np.ndarray:
    total = mag_band.sum(axis=0)
    weighted = (freqs_band[:, None] * mag_band).sum(axis=0)
    return weighted / np.maximum(total, _EPS)


def spectral_rolloff85(mag_band: np.ndarray, freqs_band: np.ndarray) -> np.ndarray:
    """Frecuencia bajo la que cae el 85 % de la energia de la banda util.

    Vectorizado sobre tramas: ``reached`` marca, por bin, si la energia
    acumulada ya alcanzo el umbral; ``argmax`` sobre un array booleano da el
    primer ``True`` de cada columna sin recorrer las tramas en Python.
    """
    total = mag_band.sum(axis=0)
    cumulative = np.cumsum(mag_band, axis=0)
    threshold = _ROLLOFF_PERCENT * total
    reached = cumulative >= threshold[None, :]
    idx = np.argmax(reached, axis=0)
    out = freqs_band[idx]
    fallback = (total <= _EPS) | (~reached.any(axis=0))
    out = np.where(fallback, freqs_band[0], out)
    return out.astype(np.float64)


def spectral_bandwidth(mag_band: np.ndarray, freqs_band: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    total = mag_band.sum(axis=0)
    deviation_sq = (freqs_band[:, None] - centroid[None, :]) ** 2
    weighted = (deviation_sq * mag_band).sum(axis=0)
    return np.sqrt(weighted / np.maximum(total, _EPS))


def spectral_flux(mag_band: np.ndarray) -> np.ndarray:
    """Distancia L2 entre espectros consecutivos normalizados por L1.

    Cada trama se normaliza primero a que su magnitud sume 1 (norma L1), y
    despues se mide la distancia euclidiana entre tramas consecutivas. La
    primera trama no tiene una anterior con la que compararse y se omite: la
    serie resultante tiene 308 valores, no 309.
    """
    l1_norm = mag_band.sum(axis=0, keepdims=True)
    normalized = mag_band / np.maximum(l1_norm, _EPS)
    diff = normalized[:, 1:] - normalized[:, :-1]
    return np.sqrt((diff ** 2).sum(axis=0))


def spectral_flatness(mag_band: np.ndarray) -> np.ndarray:
    """Media geometrica sobre media aritmetica de la magnitud, por trama."""
    mag_safe = mag_band + _EPS
    geometric_mean = np.exp(np.mean(np.log(mag_safe), axis=0))
    arithmetic_mean = np.mean(mag_band, axis=0) + _EPS
    return geometric_mean / arithmetic_mean


def spectral_kurtosis(mag_band: np.ndarray) -> np.ndarray:
    """Curtosis de Pearson (no en exceso) de la distribucion de magnitud.

    ``m2**2`` en el denominador lleva un piso ``_EPS``: en una trama de
    magnitud constante (incluida la nula) el numerador y el denominador son
    ambos cero y el cociente debe quedar en 0, no en NaN.
    """
    mean = mag_band.mean(axis=0, keepdims=True)
    deviation = mag_band - mean
    m2 = np.mean(deviation ** 2, axis=0)
    m4 = np.mean(deviation ** 4, axis=0)
    return m4 / (m2 ** 2 + _EPS)


def waveform_descriptors(segment: np.ndarray, acoustic_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Tasa de cruces por cero y RMS, alineados en tramas con la STFT."""
    frame_length = int(acoustic_cfg["n_fft"])
    hop_length = int(acoustic_cfg["hop_length"])
    y = segment.astype(np.float32)

    zcr = librosa.feature.zero_crossing_rate(
        y, frame_length=frame_length, hop_length=hop_length, center=False,
    )[0].astype(np.float64)
    rms = librosa.feature.rms(
        y=y, frame_length=frame_length, hop_length=hop_length, center=False,
    )[0].astype(np.float64)
    return zcr, rms


def compute_descriptors(
    segment: np.ndarray, magnitude: np.ndarray, freqs: np.ndarray, cfg: dict
) -> list[np.ndarray]:
    """Los 8 descriptores, en el orden fijo de ``DESCRIPTOR_NAMES``."""
    acoustic_cfg = cfg["acoustic"]
    low_hz = float(acoustic_cfg["band_low_hz"])
    high_hz = float(acoustic_cfg["band_high_hz"])
    mask = _band_mask(freqs, low_hz, high_hz)
    if not mask.any():
        raise ValueError(f"la banda {low_hz}-{high_hz} Hz no contiene ningun bin de la STFT")

    mag_band = magnitude[mask, :]
    freqs_band = freqs[mask]

    centroid = spectral_centroid(mag_band, freqs_band)
    rolloff = spectral_rolloff85(mag_band, freqs_band)
    bandwidth = spectral_bandwidth(mag_band, freqs_band, centroid)
    flux = spectral_flux(mag_band)
    flatness = spectral_flatness(mag_band)
    zcr, rms = waveform_descriptors(segment, acoustic_cfg)
    kurtosis = spectral_kurtosis(mag_band)

    descriptors = [centroid, rolloff, bandwidth, flux, flatness, zcr, kurtosis, rms]
    for name, series in zip(DESCRIPTOR_NAMES, descriptors):
        if not np.isfinite(series).all():
            raise FloatingPointError(f"el descriptor {name} contiene NaN o Inf")
    return descriptors
