"""Datos sinteticos compartidos por las pruebas (ningun dato real)."""

from __future__ import annotations

import copy
import hashlib

N_MELS = 4


def fake_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


BASE_BUNDLE = {
    "format_version": 1,
    "dataset": "SINTETICO",
    "condition": "dn",
    "branch": "dn",
    "input": {"name": "logmel", "shape": ["batch", 1, N_MELS, 9]},
    "output": {"name": "logit"},
    "normalization": {"mean": [0.0] * N_MELS, "std": [1.0] * N_MELS},
    "acoustic_config": {"sample_rate": 4000, "expected_frames": 9},
    "logmel_config": {"n_mels": N_MELS},
    "preprocessing": {
        "target_sr": 4000,
        "resample_ratios": {"44100": [40, 441], "10000": [2, 5]},
        "antialias": {"passband_hz": 1800.0, "stopband_hz": 2000.0, "cutoff_hz": 1900.0,
                      "ripple_db": 65.0, "padtype": "line"},
        "intermediate_float32": True,
        "bandpass": {"low_hz": 50.0, "high_hz": 1800.0, "order": 4},
        "denoising": {"enabled": True, "window": "hann", "nperseg": 256, "noverlap": 192,
                      "noise_pct": 15.0, "oversubtraction": 4.0, "spectral_floor": 0.05},
        "amplitude": {"target_rms": 0.03, "max_gain": 20.0, "peak_ceiling": 0.95},
        "segmentation": {"window_samples": 20000, "hop_samples": 10000},
    },
    "classes": {"names": ["Normal", "COPD"], "negative_label_name": "Normal", "positive_label_name": "COPD"},
    "decision": {"threshold": 0.5, "rule": "score >= threshold -> COPD"},
    "aggregation": {"segment_to_recording": "mean", "recording_to_patient": "mean"},
    "source": {"final_run_id": "RUN", "checkpoint_sha256": fake_sha("ckpt")},
    "models": {
        "fp32": {"file": "model_fp32.onnx", "sha256": fake_sha("fp32"), "size_bytes": 10},
        "int8": None,
    },
}

INT8_BLOCK = {
    "file": "model_int8.onnx",
    "sha256": fake_sha("int8"),
    "size_bytes": 4,
    "quantization": {"format": "QDQ"},
    "calibration": {"roles": ["train"], "inventory_sha256": fake_sha("inventario")},
    "frozen": False,
}


def make_bundle(with_int8: bool = False) -> dict:
    bundle = copy.deepcopy(BASE_BUNDLE)
    if with_int8:
        bundle["models"]["int8"] = copy.deepcopy(INT8_BLOCK)
    return bundle
