"""Los pesos por muestra dan igual contribucion total por clase y por
paciente; ``select_rows_by_patients`` mantiene segments y X alineados."""

import numpy as np
import pandas as pd
import pytest

from .. import data as dmod


def _sample_segments() -> pd.DataFrame:
    return pd.DataFrame([
        {"patient_uid": "P1", "audio_id": "rec1", "segment_id": "s1", "target_label": 0},
        {"patient_uid": "P1", "audio_id": "rec1", "segment_id": "s2", "target_label": 0},
        {"patient_uid": "P1", "audio_id": "rec2", "segment_id": "s3", "target_label": 0},
        {"patient_uid": "P2", "audio_id": "rec3", "segment_id": "s4", "target_label": 0},
        {"patient_uid": "P2", "audio_id": "rec3", "segment_id": "s5", "target_label": 0},
        {"patient_uid": "P2", "audio_id": "rec3", "segment_id": "s6", "target_label": 0},
        {"patient_uid": "P3", "audio_id": "rec4", "segment_id": "s7", "target_label": 1},
        {"patient_uid": "P3", "audio_id": "rec4", "segment_id": "s8", "target_label": 1},
    ])


def test_weights_normalized_to_mean_one():
    segments = _sample_segments()
    weights = dmod.compute_sample_weights(segments)
    assert weights.mean() == pytest.approx(1.0)
    assert (weights > 0).all()


def test_weights_equal_contribution_per_class():
    segments = _sample_segments()
    segments = segments.assign(weight=dmod.compute_sample_weights(segments))
    per_class = segments.groupby("target_label")["weight"].sum()
    assert per_class.loc[0] == pytest.approx(per_class.loc[1])


def test_weights_equal_contribution_per_patient_within_a_class():
    segments = _sample_segments()
    segments = segments.assign(weight=dmod.compute_sample_weights(segments))
    class0 = segments.loc[segments["target_label"] == 0]
    per_patient = class0.groupby("patient_uid")["weight"].sum()
    # P1 (2 grabaciones, 3 segmentos) y P2 (1 grabacion, 3 segmentos) deben
    # aportar lo mismo a la clase 0, sin importar cuantas grabaciones o
    # segmentos tenga cada uno.
    assert per_patient.loc["P1"] == pytest.approx(per_patient.loc["P2"])


def test_weights_equal_contribution_per_recording_within_a_patient():
    segments = _sample_segments()
    segments = segments.assign(weight=dmod.compute_sample_weights(segments))
    p1 = segments.loc[segments["patient_uid"] == "P1"]
    per_recording = p1.groupby("audio_id")["weight"].sum()
    # rec1 tiene 2 segmentos, rec2 tiene 1: deben aportar lo mismo al paciente.
    assert per_recording.loc["rec1"] == pytest.approx(per_recording.loc["rec2"])


def test_weights_reject_empty_input():
    with pytest.raises(ValueError):
        dmod.compute_sample_weights(pd.DataFrame(columns=["patient_uid", "audio_id", "segment_id", "target_label"]))


def test_select_rows_by_patients_keeps_segments_and_x_aligned():
    segments = pd.DataFrame({"patient_uid": ["A", "A", "B", "C"]})
    X = np.arange(8).reshape(4, 2)
    sub_seg, sub_X = dmod.select_rows_by_patients(segments, X, ["A", "C"])
    assert sub_seg["patient_uid"].tolist() == ["A", "A", "C"]
    assert np.array_equal(sub_X, X[[0, 1, 3]])


def test_select_rows_by_patients_rejects_mismatched_lengths():
    segments = pd.DataFrame({"patient_uid": ["A", "B"]})
    X = np.zeros((3, 2))
    with pytest.raises(ValueError):
        dmod.select_rows_by_patients(segments, X, ["A"])
