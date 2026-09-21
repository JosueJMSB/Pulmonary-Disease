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


# ---------------------------------------------------------------------------
# source_dataset / stratify_by_dataset (dataset COMBINED: ICBHI + Fraiwan)
# ---------------------------------------------------------------------------

def _make_combined_segments(counts: dict, segments_per_patient: int = 2, calibration: dict | None = None) -> pd.DataFrame:
    """``counts`` mapea ``(source_dataset, target_label) -> n_pacientes``."""
    calibration = calibration or {}
    rows = []
    for (source, label), n in counts.items():
        n_calib = calibration.get((source, label), 0)
        for i in range(n):
            patient_uid = f"{source}{label}_{i:03d}"
            is_calib = i < n_calib
            for seg_idx in range(segments_per_patient):
                rows.append({
                    "patient_uid": patient_uid,
                    "audio_id": f"{patient_uid}_rec0",
                    "segment_id": f"{patient_uid}_rec0_{seg_idx:02d}",
                    "target_label": label,
                    "calibration_patient": is_calib,
                    "dataset": source,
                })
    return pd.DataFrame(rows)


def test_source_dataset_added_when_dataset_column_present():
    segments = _make_combined_segments({("ICBHI", 1): 5, ("ICBHI", 0): 5, ("FRAIWAN", 1): 5, ("FRAIWAN", 0): 5})
    table = sp.build_patient_table(segments)
    assert "source_dataset" in table.columns
    assert set(table["source_dataset"].unique()) == {"ICBHI", "FRAIWAN"}


def test_source_dataset_absent_without_dataset_column():
    # Comportamiento anterior sin cambios: sin columna "dataset" no se añade
    # source_dataset ni se valida nada relacionado con fuentes.
    segments = _make_segments()
    table = sp.build_patient_table(segments)
    assert "source_dataset" not in table.columns


def test_build_patient_table_rejects_patient_with_two_sources():
    segments = _make_combined_segments({("ICBHI", 1): 5, ("ICBHI", 0): 5})
    segments.loc[segments.index[0], "dataset"] = "FRAIWAN"
    with pytest.raises(ValueError, match="mas de una fuente"):
        sp.build_patient_table(segments)


def test_stratify_by_dataset_requires_source_dataset_column():
    segments = _make_segments()
    table = sp.build_patient_table(segments)
    with pytest.raises(ValueError, match="stratify_by_dataset"):
        sp.assign_fold_groups(table, n_splits=5, stratify_by_dataset=True)


def test_stratify_by_dataset_keeps_four_strata_in_every_fold():
    segments = _make_combined_segments({
        ("ICBHI", 1): 25, ("ICBHI", 0): 15,
        ("FRAIWAN", 1): 6, ("FRAIWAN", 0): 12,
    })
    folds = sp.build_patient_folds(segments, n_splits=5, stratify_by_dataset=True)
    assert folds.stratify_by_dataset is True

    table = folds.patient_table
    source_by_patient = dict(zip(table["patient_uid"], table["source_dataset"]))
    label_by_patient = dict(zip(table["patient_uid"], table["target_label"]))
    expected = {("ICBHI", 1), ("ICBHI", 0), ("FRAIWAN", 1), ("FRAIWAN", 0)}

    for fold_id in range(folds.n_splits):
        _, val, test = folds.get_split(fold_id)
        for group_name, ids in (("validation", val), ("test", test)):
            strata = {(source_by_patient[p], label_by_patient[p]) for p in ids}
            assert strata == expected, f"fold {fold_id} {group_name}: {strata}"


def test_stratify_by_dataset_insufficient_stratum_raises():
    segments = _make_combined_segments({
        ("ICBHI", 1): 20, ("ICBHI", 0): 20, ("FRAIWAN", 1): 3, ("FRAIWAN", 0): 20,
    })
    with pytest.raises(ValueError, match="estrato"):
        sp.build_patient_folds(segments, n_splits=5, stratify_by_dataset=True)


def test_stratify_by_dataset_defaults_to_false():
    segments = _make_segments()
    folds = sp.build_patient_folds(segments)
    assert folds.stratify_by_dataset is False


def test_stratify_by_dataset_false_keeps_previous_behavior_with_dataset_column():
    # Con stratify_by_dataset=False, tener la columna "dataset" no cambia
    # nada: solo se exigen las dos clases, no los cuatro estratos.
    segments = _make_combined_segments({
        ("ICBHI", 1): 20, ("ICBHI", 0): 20, ("FRAIWAN", 1): 20, ("FRAIWAN", 0): 20,
    })
    folds = sp.build_patient_folds(segments, n_splits=5, stratify_by_dataset=False)
    assert folds.stratify_by_dataset is False
    label_by_patient = dict(zip(folds.patient_table["patient_uid"], folds.patient_table["target_label"]))
    for fold_id in range(folds.n_splits):
        _, val, test = folds.get_split(fold_id)
        assert {label_by_patient[p] for p in val} == {0, 1}
        assert {label_by_patient[p] for p in test} == {0, 1}
