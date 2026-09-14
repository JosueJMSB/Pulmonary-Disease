"""Sin fuga entre train/validation/test, calibracion siempre en train, y
cada paciente evaluable cae en test exactamente una vez."""

import pandas as pd
import pytest

from .. import splits as sp


def _make_segments(n_pos=20, n_neg=20, n_calib_pos=2, n_calib_neg=2, segments_per_patient=3) -> pd.DataFrame:
    rows = []
    for label, n, n_calib in ((1, n_pos, n_calib_pos), (0, n_neg, n_calib_neg)):
        prefix = "P" if label == 1 else "N"
        for i in range(n):
            patient_uid = f"{prefix}{i:03d}"
            is_calib = i < n_calib
            for seg_idx in range(segments_per_patient):
                rows.append({
                    "patient_uid": patient_uid,
                    "audio_id": f"{patient_uid}_rec0",
                    "segment_id": f"{patient_uid}_rec0_{seg_idx:02d}",
                    "target_label": label,
                    "calibration_patient": is_calib,
                })
    return pd.DataFrame(rows)


def test_build_patient_folds_has_no_leakage():
    segments = _make_segments()
    folds = sp.build_patient_folds(segments, n_splits=5, random_state=20260914)
    # build_patient_folds ya corre verify_folds internamente; si llega aqui, paso.
    assert folds.n_splits == 5
    assert len(folds.patient_table) == 40


def test_calibration_patients_always_in_train():
    segments = _make_segments()
    folds = sp.build_patient_folds(segments)
    calib_ids = set(
        folds.patient_table.loc[folds.patient_table["calibration_patient"], "patient_uid"]
    )
    for fold_id in range(folds.n_splits):
        train, val, test = folds.get_split(fold_id)
        assert calib_ids <= set(train)
        assert not (calib_ids & set(val))
        assert not (calib_ids & set(test))


def test_each_evaluable_patient_in_test_exactly_once():
    segments = _make_segments()
    folds = sp.build_patient_folds(segments)
    evaluable = set(
        folds.patient_table.loc[~folds.patient_table["calibration_patient"], "patient_uid"]
    )
    coverage = {pid: 0 for pid in evaluable}
    for fold_id in range(folds.n_splits):
        _, _, test = folds.get_split(fold_id)
        for pid in test:
            coverage[pid] += 1
    assert all(count == 1 for count in coverage.values())


def test_both_classes_in_validation_and_test_every_fold():
    segments = _make_segments()
    folds = sp.build_patient_folds(segments)
    label_by_patient = dict(zip(folds.patient_table["patient_uid"], folds.patient_table["target_label"]))
    for fold_id in range(folds.n_splits):
        _, val, test = folds.get_split(fold_id)
        assert {label_by_patient[p] for p in val} == {0, 1}
        assert {label_by_patient[p] for p in test} == {0, 1}


def test_reproducible_with_same_seed():
    segments = _make_segments()
    a = sp.build_patient_folds(segments, random_state=42)
    b = sp.build_patient_folds(segments, random_state=42)
    assert sp.folds_are_identical(a, b)


def test_different_seed_can_differ():
    segments = _make_segments()
    a = sp.build_patient_folds(segments, random_state=1)
    b = sp.build_patient_folds(segments, random_state=2)
    # No es una garantia matematica, pero con 40 pacientes es extremadamente
    # improbable que dos semillas distintas produzcan la misma particion.
    assert not sp.folds_are_identical(a, b)


def test_patient_with_two_labels_raises():
    segments = _make_segments()
    segments.loc[segments["patient_uid"] == "P000", "target_label"] = 0
    segments.loc[segments.index[0], "target_label"] = 1  # misma fila -> inconsistente
    with pytest.raises(ValueError):
        sp.build_patient_table(segments)


def test_insufficient_minority_class_raises():
    segments = _make_segments(n_pos=3, n_neg=20, n_calib_pos=0, n_calib_neg=2)
    with pytest.raises(ValueError):
        sp.build_patient_folds(segments, n_splits=5)
