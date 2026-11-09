"""Flujo de audio con senales sinteticas (ningun WAV real).

No lee ningun archivo de ``modeling/configs``: la configuracion acustica y de
log-mel se define aqui, con los mismos valores que ``[acoustic]`` y
``[logmel]`` de la CNN final, para que la prueba funcione tambien en la laptop.
"""

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("soundfile")
pytest.importorskip("librosa")

from optimization import audio_pipeline as ap  # noqa: E402

ACOUSTIC = {
    "sample_rate": 4000, "segment_length": 20000, "n_fft": 256, "win_length": 256, "hop_length": 64,
    "center": False, "window": "hann", "band_low_hz": 50, "band_high_hz": 1800, "expected_frames": 309,
}
LOGMEL = {
    "n_mels": 64, "power": 2, "fmin": 50, "fmax": 1800, "htk": False, "norm": "slaney",
    "ref": 1.0, "amin": 1e-10, "top_db": 80,
}


def _params():
    cfg = ap.preprocessing_modules().cfg
    return {
        "nperseg": cfg.STFT_NPERSEG, "noverlap": cfg.STFT_NOVERLAP, "noise_pct": cfg.NOISE_PCT,
        "oversubtraction": cfg.OVERSUBTRACTION, "spectral_floor": cfg.SPECTRAL_FLOOR,
        "target_rms": cfg.TARGET_RMS, "max_gain": cfg.MAX_GAIN, "peak_ceiling": cfg.PEAK_CEILING,
        "bandpass_low_hz": cfg.BANDPASS_LOW, "bandpass_high_hz": cfg.BANDPASS_HIGH,
        "bandpass_order": cfg.BANDPASS_ORDER,
    }


def _signal(seconds, sr, seed=0):
    rng = np.random.RandomState(seed)
    return 0.05 * rng.randn(int(seconds * sr))


def test_block_contains_peak_ceiling_and_matches_runtime():
    block = ap.build_preprocessing_block(_params(), "dn")
    assert block["amplitude"]["peak_ceiling"] == pytest.approx(ap.preprocessing_modules().cfg.PEAK_CEILING)
    assert block["denoising"]["enabled"] is True
    assert ap.verify_runtime_matches(block) == []


def test_runtime_mismatch_is_rejected():
    block = ap.build_preprocessing_block(_params(), "no_dn")
    block["target_sr"] = 8000
    with pytest.raises(ap.PipelineConfigError):
        ap.verify_runtime_matches(block)


def test_missing_parameter_is_rejected():
    params = _params()
    del params["peak_ceiling"]
    with pytest.raises(ap.PipelineConfigError, match="peak_ceiling"):
        ap.build_preprocessing_block(params, "dn")


@pytest.mark.parametrize("branch", ["no_dn", "dn"])
def test_segment_grid_at_4khz(branch):
    block = ap.build_preprocessing_block(_params(), branch)
    segments, info = ap.process_waveform(_signal(12.0, 4000), 4000, block)
    # (48000 - 20000) // 10000 + 1 = 3 segmentos de 5 s con 50 % de solape.
    assert segments.shape == (3, 20000)
    assert segments.dtype == np.float32
    assert info["n_segments"] == 3
    assert np.max(np.abs(segments)) <= block["amplitude"]["peak_ceiling"] + 1e-6


def test_resampling_from_44100():
    block = ap.build_preprocessing_block(_params(), "no_dn")
    segments, info = ap.process_waveform(_signal(10.0, 44100), 44100, block)
    assert info["n_samples_4k"] == int(np.ceil(10.0 * 44100 * 40 / 441))
    assert segments.shape[1] == 20000


def test_unsupported_sample_rate_is_rejected():
    block = ap.build_preprocessing_block(_params(), "no_dn")
    with pytest.raises(ap.PipelineConfigError, match="no contemplada"):
        ap.process_waveform(_signal(6.0, 48000), 48000, block)


def test_too_short_recording_is_rejected():
    block = ap.build_preprocessing_block(_params(), "no_dn")
    with pytest.raises(ap.RecordingTooShortError):
        ap.process_waveform(_signal(4.0, 4000), 4000, block)


def test_logmel_shape_and_normalization_formula():
    block = ap.build_preprocessing_block(_params(), "no_dn")
    segments, _ = ap.process_waveform(_signal(6.0, 4000), 4000, block)
    logmel = ap.logmel_from_segments(segments, ACOUSTIC, LOGMEL)
    assert logmel.shape == (1, 1, 64, 309) and logmel.dtype == np.float32
    mean = np.linspace(-50, -40, 64)
    std = np.linspace(5, 6, 64)
    out = ap.normalize_logmel(logmel, mean, std)
    expected = (logmel - mean.astype(np.float32).reshape(1, 1, -1, 1)) / std.astype(np.float32).reshape(1, 1, -1, 1)
    np.testing.assert_array_equal(out, expected.astype(np.float32))
