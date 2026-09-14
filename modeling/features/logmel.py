"""STFT compartida, log-mel y MFCC/delta/delta-delta.

La STFT se calcula una sola vez por segmento (``compute_stft_magnitude``) y la
reutilizan tanto el log-mel de este modulo como los descriptores espectrales
de ``acoustic.py``: ninguno de los dos vuelve a llamar a ``librosa.stft``.
"""

from __future__ import annotations

import numpy as np
import librosa

N_MFCC = 13


def compute_stft_magnitude(segment: np.ndarray, acoustic_cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Magnitud de la STFT y sus frecuencias, con los parametros fijados.

    ``center=False`` para que las 309 tramas queden alineadas exactamente con
    ``librosa.feature.rms``/``zero_crossing_rate`` calculados con el mismo
    ``frame_length``/``hop_length`` en ``acoustic.py``: todas las series
    comparten el mismo eje temporal de trama.
    """
    n_fft = int(acoustic_cfg["n_fft"])
    hop_length = int(acoustic_cfg["hop_length"])
    win_length = int(acoustic_cfg["win_length"])
    sr = int(acoustic_cfg["sample_rate"])

    stft_complex = librosa.stft(
        segment.astype(np.float32),
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=acoustic_cfg.get("window", "hann"),
        center=bool(acoustic_cfg.get("center", False)),
        pad_mode="constant",
    )
    magnitude = np.abs(stft_complex).astype(np.float64)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    expected_frames = int(acoustic_cfg["expected_frames"])
    if magnitude.shape[1] != expected_frames:
        raise ValueError(
            f"la STFT produjo {magnitude.shape[1]} tramas, se esperaban {expected_frames}"
        )
    return magnitude, freqs


def compute_logmel_db(magnitude: np.ndarray, cfg: dict) -> np.ndarray:
    """Log-mel en dB a partir de la magnitud de la STFT ya calculada.

    Se pasa el espectrograma de potencia por ``S=`` para que
    ``librosa.feature.melspectrogram`` no recalcule la STFT internamente.
    """
    logmel_cfg = cfg["logmel"]
    sr = int(cfg["acoustic"]["sample_rate"])
    n_fft = int(cfg["acoustic"]["n_fft"])

    power_spec = magnitude ** int(logmel_cfg.get("power", 2))
    mel_power = librosa.feature.melspectrogram(
        S=power_spec,
        sr=sr,
        n_fft=n_fft,
        n_mels=int(logmel_cfg["n_mels"]),
        fmin=float(logmel_cfg["fmin"]),
        fmax=float(logmel_cfg["fmax"]),
        htk=bool(logmel_cfg.get("htk", False)),
        norm=logmel_cfg.get("norm", "slaney"),
        power=1.0,  # mel_power ya esta en potencia; no volver a elevar.
    )
    logmel_db = librosa.power_to_db(
        mel_power,
        ref=float(logmel_cfg.get("ref", 1.0)),
        amin=float(logmel_cfg.get("amin", 1e-10)),
        top_db=float(logmel_cfg.get("top_db", 80.0)),
    )
    expected_shape = (int(logmel_cfg["n_mels"]), magnitude.shape[1])
    if logmel_db.shape != expected_shape:
        raise ValueError(f"log-mel de forma {logmel_db.shape}, esperado {expected_shape}")
    if not np.isfinite(logmel_db).all():
        raise FloatingPointError("el log-mel contiene NaN o Inf")
    return logmel_db


def compute_mfcc_stack(magnitude: np.ndarray, cfg: dict) -> np.ndarray:
    """13 MFCC (incluido el 0) + 13 delta + 13 delta-delta: array (39, n_frames).

    DCT-II ortonormal (``norm="ortho"``, el comportamiento de
    ``librosa.feature.mfcc`` con ``dct_type=2``). Los delta usan
    ``width=9`` sobre el eje temporal, con el relleno de borde por defecto de
    ``librosa.feature.delta`` (repite el valor del borde).
    """
    mfcc_cfg = cfg["mfcc"]
    logmel_db = compute_logmel_db(magnitude, cfg)

    n_mfcc = int(mfcc_cfg.get("n_mfcc", N_MFCC))
    mfcc = librosa.feature.mfcc(S=logmel_db, n_mfcc=n_mfcc, dct_type=2, norm="ortho")

    width = int(mfcc_cfg.get("delta_width", 9))
    delta1 = librosa.feature.delta(mfcc, width=width, order=int(mfcc_cfg.get("delta_order", 1)))
    delta2 = librosa.feature.delta(mfcc, width=width, order=int(mfcc_cfg.get("delta2_order", 2)))

    stack = np.concatenate([mfcc, delta1, delta2], axis=0)
    expected_rows = 3 * n_mfcc
    if stack.shape != (expected_rows, magnitude.shape[1]):
        raise ValueError(
            f"pila MFCC de forma {stack.shape}, esperado ({expected_rows}, {magnitude.shape[1]})"
        )
    if not np.isfinite(stack).all():
        raise FloatingPointError("MFCC/delta/delta-delta contienen NaN o Inf")
    return stack
