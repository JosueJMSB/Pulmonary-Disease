"""Protocolo fold-aware (v2), de punta a punta con la SVM y datos sinteticos:
cada fold lee su propia carpeta fold_00..fold_04, no se genera ningun
modelo final, y arrancar sin [final_model] enabled=false falla antes de
tocar datos. No necesita PyTorch (por eso solo cubre la SVM aqui; CNN/CRNN
comparten la misma orquestacion via cnn_experiment.run_folded_cnn, ya
cubierta por pruebas unitarias en test_folded_protocol_configs.py).
"""

import copy
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from .. import data as dmod
from .. import run_experiment as rexp
from .. import splits as sp

N_PER_CLASS = 6
N_SPLITS = 5


def _toy_patient_folds(tmp_path):
    """5 folds sobre 12 pacientes (6 COPD + 6 Control), respect_calibration_patient=False."""
    rows = []
    for label, prefix in ((1, "P"), (0, "N")):
        for i in range(N_PER_CLASS):
            patient_uid = f"{prefix}{i:03d}"
            for seg_idx in range(2):
                rows.append({
                    "patient_uid": patient_uid,
                    "audio_id": f"{patient_uid}_rec0",
                    "segment_id": f"{patient_uid}_rec0_{seg_idx:02d}",
                    "target_label": label,
                    "calibration_patient": i == 0,  # historico, no debe excluir a nadie del sorteo
                    "dataset": "TOY",
                })
    segments = pd.DataFrame(rows)
    folds = sp.build_patient_folds(
        segments, n_splits=N_SPLITS, random_state=20260914, respect_calibration_patient=False,
    )
    csv_path = tmp_path / "modeling_data" / "patient_folds.csv"
    manifest_path = tmp_path / "modeling_data" / "patient_folds_manifest.json"
    frame = sp.patient_folds_to_frame(folds, "TOY")
    sp.write_patient_folds_csv(frame, csv_path, manifest_path, {
        "counts": {"TOY": {"n_patients": len(folds.patient_table)}},
    })
    return folds, csv_path, manifest_path


def _write_fold_data(data_root, fold_id, folds: sp.PatientFolds, patient_folds_csv):
    """segments.csv + ambas ramas .npy para un fold, con las 12 filas de
    patient_folds.csv (misma poblacion en todos los folds; solo el audio
    -aqui, sintetico- cambiaria fold a fold en la version real)."""
    fold_dir = data_root / "TOY" / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, patient in enumerate(folds.patient_table.itertuples()):
        rows.append({
            "task_array_index": i, "source_array_index": i,
            "segment_id": f"{patient.patient_uid}_S", "audio_id": f"{patient.patient_uid}_rec0",
            "patient_uid": patient.patient_uid, "diagnosis": "COPD" if patient.target_label else "Control",
            "target_label": int(patient.target_label),
            "target_name": "COPD" if patient.target_label else "Control",
            "calibration_patient": bool(patient.calibration_patient), "dn_reliable": True,
        })
    segments = pd.DataFrame(rows)
    segments.to_csv(fold_dir / "segments.csv", index=False)

    rng = np.random.default_rng(fold_id)
    n = len(segments)
    # Senal separable por clase para que el grid search de la SVM converja.
    offsets = np.where(segments["target_label"].to_numpy() == 1, 2.0, -2.0)
    array_no_dn = (rng.standard_normal((n, 20000)) * 0.05 + offsets[:, None]).astype(np.float32)
    array_dn = array_no_dn * 0.9
    np.save(fold_dir / "segments_no_dn.npy", array_no_dn)
    np.save(fold_dir / "segments_dn.npy", array_dn)

    output_hashes = {
        name: dmod.sha256_file(fold_dir / name)
        for name in ("segments.csv", "segments_no_dn.npy", "segments_dn.npy")
    }
    (fold_dir / "manifest.json").write_text(
        json.dumps({
            "output_hashes": output_hashes,
            "patient_folds_csv_sha256": dmod.sha256_file(patient_folds_csv),
        }), encoding="utf-8",
    )


def _folded_svm_cfg(tmp_path, data_root):
    cfg = copy.deepcopy(dmod.load_config())
    cfg["svm"]["c_grid"] = [1.0]
    cfg["svm"]["gamma_grid"] = ["scale"]
    cfg["paths"] = {
        "default_data_root": str(data_root),
        "default_runs_root": str(tmp_path / "runs"),
        "default_cache_root": str(tmp_path / "cache"),
    }
    cfg["folds"] = {
        "patient_folds_csv": str(tmp_path / "modeling_data" / "patient_folds.csv"),
        "patient_folds_manifest": str(tmp_path / "modeling_data" / "patient_folds_manifest.json"),
    }
    cfg["final_model"] = {"enabled": False}
    cfg["datasets"] = {"TOY": {"positive_diagnosis": "COPD", "negative_diagnosis": "Control", "negative_label_name": "Control"}}
    cfg["experiments"] = {
        "main": {"TOY": {"condition": "no_dn", "branch": "no_dn", "dn_reliable_only": False}},
    }
    return cfg


def _make_args(**overrides):
    base = dict(
        model="svm_rbf", dataset="TOY", experiment="main", n_jobs=1, device="auto", num_workers=0,
        data_root=None, runs_root=None, cache_root=None, config=None, dry_run=False,
        smoke_test=True, force_features=False, resume=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_run_folded_svm_end_to_end_smoke(tmp_path):
    data_root = tmp_path / "fold_calibrated"
    folds, csv_path, manifest_path = _toy_patient_folds(tmp_path)
    _write_fold_data(data_root, 0, folds, csv_path)  # --smoke-test solo toca el fold 0

    cfg = _folded_svm_cfg(tmp_path, data_root)
    assert rexp.is_folded_protocol(cfg)

    rc = rexp.run_folded_svm(_make_args(), cfg)

    assert rc == 0
    run_dirs = list((tmp_path / "runs").glob("svm_rbf/*"))
    assert len(run_dirs) == 1
    run_root = run_dirs[0]

    status = (run_root / "status.json").read_text(encoding="utf-8")
    assert '"COMPLETED"' in status

    cdir = run_root / "datasets" / "TOY" / "no_dn"
    assert not (cdir / "final").exists()
    oof = pd.read_csv(cdir / "oof_patient_predictions.csv")
    # smoke-test: solo el fold 0 -> solo su grupo de test (tamano exacto segun
    # como StratifiedKFold reparte 6+6 pacientes en 5 folds; no siempre es igual).
    train, val, test = folds.get_split(0)
    assert len(oof) == len(test)
    assert set(oof["patient_uid"]) == set(test)


def test_run_folded_svm_dry_run_checks_all_five_folds_and_fails_on_missing(tmp_path):
    data_root = tmp_path / "fold_calibrated"
    folds, csv_path, manifest_path = _toy_patient_folds(tmp_path)
    _write_fold_data(data_root, 0, folds, csv_path)  # faltan los folds 1..4 a proposito

    cfg = _folded_svm_cfg(tmp_path, data_root)
    rc = rexp.run_folded_svm(_make_args(dry_run=True), cfg)

    assert rc == 1  # veredicto: REVISAR (faltan fold_01..fold_04)


def test_run_folded_svm_dry_run_passes_when_all_folds_present(tmp_path):
    data_root = tmp_path / "fold_calibrated"
    folds, csv_path, manifest_path = _toy_patient_folds(tmp_path)
    for fold_id in range(N_SPLITS):
        _write_fold_data(data_root, fold_id, folds, csv_path)

    cfg = _folded_svm_cfg(tmp_path, data_root)
    rc = rexp.run_folded_svm(_make_args(dry_run=True), cfg)

    assert rc == 0


def test_run_folded_svm_dry_run_fails_when_patient_folds_csv_changed_after_preprocessing(tmp_path):
    data_root = tmp_path / "fold_calibrated"
    folds, csv_path, manifest_path = _toy_patient_folds(tmp_path)
    for fold_id in range(N_SPLITS):
        _write_fold_data(data_root, fold_id, folds, csv_path)

    # patient_folds.csv se regenera (p.ej. nuevo sorteo) DESPUES de haber
    # preprocesado los folds: el hash grabado en cada manifest.json ya no
    # coincide con el patient_folds.csv actual, aunque este siga siendo
    # autoconsistente con su propio manifiesto (sp.load_patient_folds no
    # detecta nada raro por si solo; el enlace fold<->csv es lo que se rompe).
    frame = pd.read_csv(csv_path, dtype={"patient_uid": str})
    frame.loc[0, "calibration_patient"] = not bool(frame.loc[0, "calibration_patient"])
    sp.write_patient_folds_csv(frame, csv_path, manifest_path, {
        "counts": {"TOY": {"n_patients": len(folds.patient_table)}},
    })

    cfg = _folded_svm_cfg(tmp_path, data_root)
    rc = rexp.run_folded_svm(_make_args(dry_run=True), cfg)

    assert rc == 1  # veredicto: REVISAR (folds preprocesados con un patient_folds.csv distinto al actual)


def test_run_folded_svm_rejects_final_model_enabled_true(tmp_path):
    data_root = tmp_path / "fold_calibrated"
    folds, csv_path, manifest_path = _toy_patient_folds(tmp_path)
    _write_fold_data(data_root, 0, folds, csv_path)

    cfg = _folded_svm_cfg(tmp_path, data_root)
    cfg["final_model"]["enabled"] = True

    with pytest.raises(RuntimeError, match="final_model"):
        rexp.run_folded_svm(_make_args(), cfg)

    # No debe haberse creado ninguna corrida: el chequeo es lo primero que pasa.
    assert not (tmp_path / "runs").exists()
