"""Los pesos por muestra dan igual contribucion total por clase y por
paciente; ``select_rows_by_patients`` mantiene segments y X alineados."""

import json
from pathlib import Path

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


# ---------------------------------------------------------------------------
# fold_id (protocolo fold-aware v2): fold_id=None reproduce las rutas
# actuales byte a byte; fold_id=N añade fold_00..fold_04.
# ---------------------------------------------------------------------------

def test_dataset_root_without_fold_id_matches_current_layout(tmp_path):
    assert dmod.dataset_root(tmp_path, "ICBHI") == tmp_path / "ICBHI"
    assert dmod.dataset_root(tmp_path, "ICBHI", fold_id=None) == tmp_path / "ICBHI"


def test_dataset_root_with_fold_id_appends_zero_padded_segment(tmp_path):
    assert dmod.dataset_root(tmp_path, "ICBHI", fold_id=0) == tmp_path / "ICBHI" / "fold_00"
    assert dmod.dataset_root(tmp_path, "ICBHI", fold_id=4) == tmp_path / "ICBHI" / "fold_04"


def test_feature_and_logmel_cache_dir_fold_id_none_matches_current_layout(tmp_path):
    assert dmod.feature_cache_dir(tmp_path, "ICBHI", "no_dn") == tmp_path / "ICBHI" / "no_dn"
    assert dmod.logmel_cache_dir(tmp_path, "ICBHI", "no_dn") == tmp_path / "ICBHI" / "no_dn"


def test_feature_and_logmel_cache_dir_with_fold_id(tmp_path):
    assert dmod.feature_cache_dir(tmp_path, "ICBHI", "no_dn", fold_id=2) == tmp_path / "ICBHI" / "fold_02" / "no_dn"
    assert dmod.logmel_cache_dir(tmp_path, "ICBHI", "no_dn", fold_id=2) == tmp_path / "ICBHI" / "fold_02" / "no_dn"


def _write_toy_task_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "task_array_index": 0, "source_array_index": 0, "segment_id": "S0", "audio_id": "A0",
        "patient_uid": "P0", "diagnosis": "COPD", "target_label": 1, "target_name": "COPD",
        "calibration_patient": False, "dn_reliable": True,
    }]).to_csv(root / "segments.csv", index=False)
    (root / "manifest.json").write_text(json.dumps({"verdict": "PASS"}) + "\n", encoding="utf-8")
    np.save(root / "segments_no_dn.npy", np.zeros((1, 4), dtype=np.float32))


def test_load_task_segments_and_manifest_read_from_fold_subdirectory(tmp_path):
    data_root = tmp_path / "data"
    _write_toy_task_dataset(data_root / "ICBHI" / "fold_03")

    segments = dmod.load_task_segments(data_root, "ICBHI", fold_id=3)
    assert segments.loc[0, "patient_uid"] == "P0"
    manifest = dmod.load_task_manifest(data_root, "ICBHI", fold_id=3)
    assert manifest["verdict"] == "PASS"
    array = dmod.load_branch_array(data_root, "ICBHI", "no_dn", fold_id=3)
    assert array.shape == (1, 4)


def test_load_task_segments_without_fold_id_ignores_fold_subdirectories(tmp_path):
    data_root = tmp_path / "data"
    _write_toy_task_dataset(data_root / "ICBHI")
    # Una carpeta fold_00 al lado no debe interferir con la ruta sin fold_id.
    _write_toy_task_dataset(data_root / "ICBHI" / "fold_00")

    segments = dmod.load_task_segments(data_root, "ICBHI")
    assert len(segments) == 1
