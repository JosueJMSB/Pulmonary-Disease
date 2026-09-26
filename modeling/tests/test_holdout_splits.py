"""Separacion holdout-v3 (PLAN-EXPERIMENTO FINAL.md, seccion 2): 80 % desarrollo /
20 % prueba externa bloqueada por paciente, 5 folds internos solo sobre el 80 %,
COMBINED heredando la asignacion de su fuente, manifiesto con hashes y conteos,
y ausencia total de fuga. Todo sobre segments.csv sinteticos -no toca datos
reales-.
"""

import json

import pandas as pd
import pytest

from .. import build_master_folds as bmf
from .. import splits as sp

N_SPLITS = 5


def _make_segments(source: str, n_pos: int, n_neg: int, segments_per_patient: int = 2) -> pd.DataFrame:
    rows = []
    for label, n in ((1, n_pos), (0, n_neg)):
        for i in range(n):
            patient_uid = f"{source}_{label}_{i:03d}"
            for seg_idx in range(segments_per_patient):
                rows.append({
                    "patient_uid": patient_uid,
                    "audio_id": f"{patient_uid}_rec0",
                    "segment_id": f"{patient_uid}_rec0_{seg_idx:02d}",
                    "target_label": label,
                    "calibration_patient": i < 2,
                    "dataset": source,
                })
    return pd.DataFrame(rows)


def _write_segments_csv(path, source: str, n_pos: int, n_neg: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _make_segments(source, n_pos, n_neg).to_csv(path, index=False)


def _split(source="ICBHI", n_pos=40, n_neg=20):
    table = sp.build_patient_table(_make_segments(source, n_pos, n_neg))
    return sp.assign_holdout_split(table, n_splits=N_SPLITS, random_state=20260914)


def _both_scopes():
    icbhi = _split("ICBHI", n_pos=40, n_neg=20)
    fraiwan = _split("FRAIWAN", n_pos=8, n_neg=30)
    return icbhi, fraiwan, sp.combine_holdout_splits({"ICBHI": icbhi, "FRAIWAN_Extended": fraiwan}, N_SPLITS)


# ---------------------------------------------------------------------------
# Separacion 80/20 por paciente
# ---------------------------------------------------------------------------

def test_split_is_80_20_by_patient_without_overlap():
    table = _split()
    development = table.loc[table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT]
    test = table.loc[table["outer_role"] == sp.OUTER_ROLE_TEST]

    assert len(development) + len(test) == len(table) == 60
    assert len(test) == 12 and len(development) == 48          # 20 % / 80 %
    assert not set(development["patient_uid"]) & set(test["patient_uid"])
    assert table["patient_uid"].is_unique


def test_split_is_stratified_by_class():
    table = _split(n_pos=40, n_neg=20)
    test = table.loc[table["outer_role"] == sp.OUTER_ROLE_TEST]
    development = table.loc[table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT]
    # 40:20 -> proporcion 2:1 en ambos conjuntos.
    assert test["target_label"].value_counts().to_dict() == {1: 8, 0: 4}
    assert development["target_label"].value_counts().to_dict() == {1: 32, 0: 16}


def test_test_patients_have_inner_fold_group_minus_one_and_development_0_to_4():
    table = _split()
    test = table.loc[table["outer_role"] == sp.OUTER_ROLE_TEST]
    development = table.loc[table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT]
    assert (test["inner_fold_group"] == sp.OUTER_TEST_FOLD_GROUP).all()
    assert set(development["inner_fold_group"]) == set(range(N_SPLITS))


def test_inner_folds_are_built_only_from_development_and_keep_both_classes():
    table = _split()
    development = table.loc[table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT]
    for fold_id in range(N_SPLITS):
        validation = development.loc[development["inner_fold_group"] == fold_id]
        assert set(validation["target_label"]) == {0, 1}
    # Cada paciente de desarrollo cae en exactamente un fold de validacion.
    assert development["patient_uid"].is_unique


def test_split_is_deterministic_and_independent_of_input_order():
    segments = _make_segments("ICBHI", 40, 20)
    table = sp.build_patient_table(segments)
    first = sp.assign_holdout_split(table, n_splits=N_SPLITS, random_state=20260914)
    shuffled = sp.assign_holdout_split(table.sample(frac=1.0, random_state=1), n_splits=N_SPLITS, random_state=20260914)
    columns = ["patient_uid", "outer_role", "inner_fold_group"]
    assert first[columns].reset_index(drop=True).equals(shuffled[columns].reset_index(drop=True))


def test_split_rejects_a_minority_class_too_small_for_the_inner_folds():
    # 5 COPD en total -> la prueba se lleva 1 y desarrollo queda con 4 < 5 folds.
    table = sp.build_patient_table(_make_segments("ICBHI", n_pos=5, n_neg=40))
    with pytest.raises(ValueError, match="insuficiente"):
        sp.assign_holdout_split(table, n_splits=N_SPLITS, random_state=20260914)


# ---------------------------------------------------------------------------
# verify_holdout_split: la fuga o una asignacion incoherente se rechazan
# ---------------------------------------------------------------------------

def test_verify_rejects_a_test_patient_with_a_fold_group():
    table = _split().copy()
    test_idx = table.index[table["outer_role"] == sp.OUTER_ROLE_TEST][0]
    table.loc[test_idx, "inner_fold_group"] = 0
    with pytest.raises(RuntimeError, match="prueba"):
        sp.verify_holdout_split(table, N_SPLITS)


def test_verify_rejects_a_development_patient_without_fold():
    table = _split().copy()
    dev_idx = table.index[table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT][0]
    table.loc[dev_idx, "inner_fold_group"] = sp.OUTER_TEST_FOLD_GROUP
    with pytest.raises(RuntimeError, match="fuera de los grupos"):
        sp.verify_holdout_split(table, N_SPLITS)


def test_verify_rejects_duplicated_patients():
    table = _split()
    duplicated = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(RuntimeError, match="repetidos"):
        sp.verify_holdout_split(duplicated, N_SPLITS)


def test_verify_rejects_a_validation_fold_with_a_single_class():
    table = _split().copy()
    development = table.loc[table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT]
    in_fold_zero = development.index[development["inner_fold_group"] == 0]
    table.loc[in_fold_zero, "target_label"] = 1  # el fold 0 solo con COPD
    with pytest.raises(RuntimeError, match="ambas clases"):
        sp.verify_holdout_split(table, N_SPLITS)


# ---------------------------------------------------------------------------
# COMBINED hereda exactamente la asignacion de cada fuente
# ---------------------------------------------------------------------------

def test_combined_inherits_exact_assignment_of_each_source():
    icbhi, fraiwan, combined = _both_scopes()
    columns = ["outer_role", "inner_fold_group"]
    by_patient = combined.set_index("patient_uid")[columns]
    for source_table in (icbhi, fraiwan):
        for row in source_table.itertuples():
            assert by_patient.loc[row.patient_uid, "outer_role"] == row.outer_role
            assert by_patient.loc[row.patient_uid, "inner_fold_group"] == row.inner_fold_group
    assert len(combined) == len(icbhi) + len(fraiwan)


def test_combined_keeps_every_source_and_class_stratum_in_test_and_in_each_validation_fold():
    _, _, combined = _both_scopes()
    strata = set(zip(combined["source_dataset"], combined["target_label"]))
    assert len(strata) == 4
    test = combined.loc[combined["outer_role"] == sp.OUTER_ROLE_TEST]
    assert set(zip(test["source_dataset"], test["target_label"])) == strata
    development = combined.loc[combined["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT]
    for fold_id in range(N_SPLITS):
        fold = development.loc[development["inner_fold_group"] == fold_id]
        assert set(zip(fold["source_dataset"], fold["target_label"])) == strata


def test_combine_rejects_patients_shared_between_sources():
    icbhi = _split("ICBHI", 40, 20)
    with pytest.raises(ValueError, match="compartidos"):
        sp.combine_holdout_splits({"A": icbhi, "B": icbhi}, N_SPLITS)


# ---------------------------------------------------------------------------
# CSV + manifiesto: ida y vuelta, verificacion de hash y protocolo
# ---------------------------------------------------------------------------

def _write_split(tmp_path):
    icbhi, fraiwan, combined = _both_scopes()
    frame = pd.concat([
        sp.holdout_split_to_frame(icbhi, "ICBHI"),
        sp.holdout_split_to_frame(fraiwan, "FRAIWAN_Extended"),
        sp.holdout_split_to_frame(combined, "COMBINED"),
    ], ignore_index=True)
    csv_path, manifest_path = tmp_path / "holdout_splits.csv", tmp_path / "holdout_splits_manifest.json"
    sp.write_holdout_split_csv(frame, csv_path, manifest_path, {
        "seed": 20260914, "n_splits": N_SPLITS,
        "counts": {"ICBHI": {"n_patients": len(icbhi)}, "COMBINED": {"n_patients": len(combined)}},
    })
    return csv_path, manifest_path


def test_write_and_load_holdout_split_roundtrip(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    for scope in ("ICBHI", "FRAIWAN_Extended", "COMBINED"):
        split = sp.load_holdout_split(csv_path, manifest_path, scope)
        assert split.n_splits == N_SPLITS
        assert len(split.development_patients()) + len(split.blocked_test_patients()) == len(split.patient_table)


def test_load_rejects_a_csv_that_changed_after_the_manifest(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(ValueError, match="sha256"):
        sp.load_holdout_split(csv_path, manifest_path, "ICBHI")


def test_load_rejects_a_manifest_of_another_protocol(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "fold-aware-v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        sp.load_holdout_split(csv_path, manifest_path, "ICBHI")


def test_load_rejects_a_patient_count_that_disagrees_with_the_manifest(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["counts"]["ICBHI"]["n_patients"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifiesto"):
        sp.load_holdout_split(csv_path, manifest_path, "ICBHI")


def test_cv_split_partitions_development_and_never_returns_test_patients(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    split = sp.load_holdout_split(csv_path, manifest_path, "COMBINED")
    development = set(split.development_patients())
    blocked = split.blocked_test_patients()
    validated = []
    for fold_id in range(split.n_splits):
        train, val = split.cv_split(fold_id)
        assert not set(train) & set(val)
        assert set(train) | set(val) == development
        assert not (set(train) | set(val)) & blocked
        validated += val
    # Todos los pacientes de desarrollo se validan exactamente una vez.
    assert sorted(validated) == sorted(development)
    assert set(split.cv_fold_table()["role"]) == {sp.ROLE_TRAIN, sp.ROLE_VALIDATION}


def test_assert_no_blocked_patients_raises_when_a_test_patient_shows_up(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    split = sp.load_holdout_split(csv_path, manifest_path, "ICBHI")
    leaked = next(iter(split.blocked_test_patients()))
    with pytest.raises(RuntimeError, match="prueba externa"):
        split.assert_no_blocked_patients({leaked}, "prueba")
    split.assert_no_blocked_patients(set(split.development_patients()), "prueba")  # no lanza


# ---------------------------------------------------------------------------
# build_master_folds --protocol holdout-v3
# ---------------------------------------------------------------------------

def _run_builder(tmp_path, *extra):
    return bmf.main([
        "--protocol", "holdout-v3",
        "--icbhi-segments", str(tmp_path / "ICBHI" / "segments.csv"),
        "--fraiwan-segments", str(tmp_path / "FRAIWAN_Extended" / "segments.csv"),
        "--output-csv", str(tmp_path / "holdout_splits.csv"),
        "--output-manifest", str(tmp_path / "holdout_splits_manifest.json"),
        *extra,
    ])


def test_build_master_folds_holdout_v3_writes_split_and_manifest(tmp_path):
    _write_segments_csv(tmp_path / "ICBHI" / "segments.csv", "ICBHI", n_pos=40, n_neg=20)
    _write_segments_csv(tmp_path / "FRAIWAN_Extended" / "segments.csv", "FRAIWAN", n_pos=8, n_neg=30)

    assert _run_builder(tmp_path) == 0

    frame = pd.read_csv(tmp_path / "holdout_splits.csv", dtype={"patient_uid": str})
    assert list(frame.columns) == list(sp.HOLDOUT_SPLIT_COLUMNS)
    assert set(frame["dataset_scope"]) == {"ICBHI", "FRAIWAN_Extended", "COMBINED"}
    assert (frame["dataset_scope"] == "COMBINED").sum() == 60 + 38

    manifest = json.loads((tmp_path / "holdout_splits_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "holdout-v3"
    assert manifest["seed"] == 20260914 and manifest["n_splits"] == N_SPLITS
    assert set(manifest["source_segments_sha256"]) == {"ICBHI", "FRAIWAN_Extended"}
    icbhi_counts = manifest["counts"]["ICBHI"]
    assert icbhi_counts["by_outer_role"] == {"development": 48, "test": 12}
    assert icbhi_counts["by_outer_role_and_class"]["test|1"] == 8
    assert manifest["counts"]["COMBINED"]["n_patients"] == 98
    # Los conteos por fuente, clase, rol y fold interno estan registrados.
    for key in ("by_source_dataset", "by_target_label", "by_source_role_and_class", "by_inner_fold_group_and_class"):
        assert key in manifest["counts"]["COMBINED"]

    for scope in ("ICBHI", "FRAIWAN_Extended", "COMBINED"):
        sp.load_holdout_split(tmp_path / "holdout_splits.csv", tmp_path / "holdout_splits_manifest.json", scope)


def test_build_master_folds_holdout_v3_is_a_single_split(tmp_path):
    _write_segments_csv(tmp_path / "ICBHI" / "segments.csv", "ICBHI", n_pos=40, n_neg=20)
    _write_segments_csv(tmp_path / "FRAIWAN_Extended" / "segments.csv", "FRAIWAN", n_pos=8, n_neg=30)
    assert _run_builder(tmp_path) == 0
    before = (tmp_path / "holdout_splits.csv").read_bytes()

    # Repetirlo con los mismos datos es idempotente.
    assert _run_builder(tmp_path) == 0
    assert (tmp_path / "holdout_splits.csv").read_bytes() == before

    # Con datos distintos NO reemplaza la separacion sin --overwrite.
    _write_segments_csv(tmp_path / "ICBHI" / "segments.csv", "ICBHI", n_pos=45, n_neg=25)
    assert _run_builder(tmp_path) == 2
    assert (tmp_path / "holdout_splits.csv").read_bytes() == before
    assert _run_builder(tmp_path, "--overwrite") == 0
    assert (tmp_path / "holdout_splits.csv").read_bytes() != before


def test_build_master_folds_default_protocol_is_still_fold_aware_v2(tmp_path):
    _write_segments_csv(tmp_path / "ICBHI" / "segments.csv", "ICBHI", n_pos=20, n_neg=15)
    _write_segments_csv(tmp_path / "FRAIWAN_Extended" / "segments.csv", "FRAIWAN", n_pos=6, n_neg=12)
    rc = bmf.main([
        "--icbhi-segments", str(tmp_path / "ICBHI" / "segments.csv"),
        "--fraiwan-segments", str(tmp_path / "FRAIWAN_Extended" / "segments.csv"),
        "--output-csv", str(tmp_path / "patient_folds.csv"),
        "--output-manifest", str(tmp_path / "patient_folds_manifest.json"),
    ])
    assert rc == 0
    assert "fold_group" in pd.read_csv(tmp_path / "patient_folds.csv").columns
    assert not (tmp_path / "holdout_splits.csv").exists()
