"""build_master_folds.py: ICBHI y Fraiwan se estratifican por separado y
COMBINED hereda su fold_group, sin sorteo propio. Todo sobre segments.csv
sinteticos -no toca datos reales-."""

import json

import pandas as pd

from .. import build_master_folds as bmf
from .. import splits as sp


def _make_segments_csv(path, source: str, n_pos: int, n_neg: int, segments_per_patient: int = 2) -> None:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_master_folds_writes_csv_and_manifest(tmp_path):
    icbhi_path = tmp_path / "ICBHI" / "segments.csv"
    fraiwan_path = tmp_path / "FRAIWAN_Extended" / "segments.csv"
    _make_segments_csv(icbhi_path, "ICBHI", n_pos=20, n_neg=15)
    _make_segments_csv(fraiwan_path, "FRAIWAN", n_pos=6, n_neg=12)

    output_csv = tmp_path / "patient_folds.csv"
    output_manifest = tmp_path / "patient_folds_manifest.json"
    rc = bmf.main([
        "--icbhi-segments", str(icbhi_path), "--fraiwan-segments", str(fraiwan_path),
        "--output-csv", str(output_csv), "--output-manifest", str(output_manifest),
    ])

    assert rc == 0
    assert output_csv.is_file()
    assert output_manifest.is_file()

    frame = pd.read_csv(output_csv, dtype={"patient_uid": str})
    assert set(frame["dataset_scope"].unique()) == {"ICBHI", "FRAIWAN_Extended", "COMBINED"}
    assert (frame["dataset_scope"] == "ICBHI").sum() == 35
    assert (frame["dataset_scope"] == "FRAIWAN_Extended").sum() == 18
    assert (frame["dataset_scope"] == "COMBINED").sum() == 35 + 18
    # Ningun grupo especial -1: calibration_patient ya no reserva nada.
    assert (frame["fold_group"] == sp.CALIBRATION_GROUP).sum() == 0

    manifest = json.loads(output_manifest.read_text(encoding="utf-8"))
    assert manifest["counts"]["ICBHI"]["n_patients"] == 35
    assert manifest["counts"]["COMBINED"]["n_patients"] == 35 + 18


def test_build_master_folds_combined_matches_source_fold_groups(tmp_path):
    icbhi_path = tmp_path / "ICBHI" / "segments.csv"
    fraiwan_path = tmp_path / "FRAIWAN_Extended" / "segments.csv"
    _make_segments_csv(icbhi_path, "ICBHI", n_pos=20, n_neg=15)
    _make_segments_csv(fraiwan_path, "FRAIWAN", n_pos=6, n_neg=12)

    output_csv = tmp_path / "patient_folds.csv"
    output_manifest = tmp_path / "patient_folds_manifest.json"
    bmf.main([
        "--icbhi-segments", str(icbhi_path), "--fraiwan-segments", str(fraiwan_path),
        "--output-csv", str(output_csv), "--output-manifest", str(output_manifest),
    ])

    frame = pd.read_csv(output_csv, dtype={"patient_uid": str})
    by_scope = {
        scope: dict(zip(group["patient_uid"], group["fold_group"]))
        for scope, group in frame.groupby("dataset_scope")
    }
    for patient_uid, fold_group in by_scope["COMBINED"].items():
        source_scope = "ICBHI" if patient_uid.startswith("ICBHI_") else "FRAIWAN_Extended"
        assert by_scope[source_scope][patient_uid] == fold_group


def test_loaded_master_folds_pass_verification(tmp_path):
    icbhi_path = tmp_path / "ICBHI" / "segments.csv"
    fraiwan_path = tmp_path / "FRAIWAN_Extended" / "segments.csv"
    _make_segments_csv(icbhi_path, "ICBHI", n_pos=20, n_neg=15)
    _make_segments_csv(fraiwan_path, "FRAIWAN", n_pos=6, n_neg=12)

    output_csv = tmp_path / "patient_folds.csv"
    output_manifest = tmp_path / "patient_folds_manifest.json"
    bmf.main([
        "--icbhi-segments", str(icbhi_path), "--fraiwan-segments", str(fraiwan_path),
        "--output-csv", str(output_csv), "--output-manifest", str(output_manifest),
    ])

    for scope in ("ICBHI", "FRAIWAN_Extended", "COMBINED"):
        folds = sp.load_patient_folds(output_csv, output_manifest, scope)
        assert folds.n_splits == 5
        for fold_id in range(folds.n_splits):
            train, val, test = folds.get_split(fold_id)
            assert not (set(train) & set(val) & set(test))
