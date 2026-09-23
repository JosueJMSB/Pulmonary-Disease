"""fold_denoising.py: recalibracion de denoising/TARGET_RMS restringida al
train de cada fold, Fraiwan nunca re-barre la resolucion/agresividad,
ausencia de fuga, segmentacion y contrato de columnas. Todo con audio
sintetico (senos + ruido) -nunca toca el corpus real ni preprocessing/data/-.

``preprocessing/`` no es un paquete Python (los scripts de fase usan
``sys.path.insert`` + imports planos, ver phase3_cleaning.py/phase4_temporal.py):
este archivo de pruebas sigue la misma convencion.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

import fold_denoising as fd
import phase3_cleaning as p3
import phase4_temporal as p4
import utils as u

SR = 4000


def _sine_plus_noise(seconds: float, freq: float = 200.0, seed: int = 0) -> np.ndarray:
    n = int(round(seconds * SR))
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    return (0.2 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(n)).astype(np.float64)


def _write_patient_folds(tmp_path, dataset_scope: str, roles_by_patient: dict, source_dataset: str = "ICBHI"):
    """``roles_by_patient``: patient_uid -> (fold_group, target_label)."""
    rows = []
    for patient_uid, (fold_group, target_label) in roles_by_patient.items():
        rows.append({
            "dataset_scope": dataset_scope, "patient_uid": patient_uid, "target_label": target_label,
            "source_dataset": source_dataset, "calibration_patient": False, "fold_group": fold_group,
        })
    frame = pd.DataFrame(rows)
    csv_path = tmp_path / "patient_folds.csv"
    manifest_path = tmp_path / "patient_folds_manifest.json"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    manifest = {"patient_folds_csv_sha256": u.file_sha256(csv_path)}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return csv_path, manifest_path


# ---------------------------------------------------------------------------
# load_patient_roles
# ---------------------------------------------------------------------------

def test_load_patient_roles_assigns_test_validation_train(tmp_path):
    roles_by_patient = {f"P{i}": (i % 5, i % 2) for i in range(15)}
    csv_path, manifest_path = _write_patient_folds(tmp_path, "ICBHI", roles_by_patient)

    roles = fd.load_patient_roles(csv_path, manifest_path, "ICBHI", fold_id=0)

    test_patients = set(roles.loc[roles["role"] == "test", "patient_uid"])
    val_patients = set(roles.loc[roles["role"] == "validation", "patient_uid"])
    assert test_patients == {p for p, (g, _) in roles_by_patient.items() if g == 0}
    assert val_patients == {p for p, (g, _) in roles_by_patient.items() if g == 1}
    assert not (test_patients & val_patients)


def test_load_patient_roles_rejects_hash_mismatch(tmp_path):
    csv_path, manifest_path = _write_patient_folds(tmp_path, "ICBHI", {"P0": (0, 1), "P1": (1, 0)})
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")  # csv cambia despues de escribir el manifiesto

    with pytest.raises(ValueError, match="sha256"):
        fd.load_patient_roles(csv_path, manifest_path, "ICBHI", fold_id=0)


def test_load_patient_roles_rejects_stray_calibration_group(tmp_path):
    csv_path, manifest_path = _write_patient_folds(tmp_path, "ICBHI", {"P0": (-1, 1), "P1": (0, 0)})
    with pytest.raises(ValueError, match="fold_group=-1"):
        fd.load_patient_roles(csv_path, manifest_path, "ICBHI", fold_id=0)


# ---------------------------------------------------------------------------
# Grabaciones autorizadas: audio_id del segments.csv de la tarea, no solo
# patient_uid (Fraiwan puede tener Bell/Diaphragm ademas de su Extended,
# todas bajo el mismo paciente).
# ---------------------------------------------------------------------------

def test_fraiwan_authorized_audio_ids_only_admit_extended_not_bell_or_diaphragm(tmp_path, monkeypatch):
    task_segments_csv = tmp_path / "fraiwan_extended_segments.csv"
    pd.DataFrame({"audio_id": ["F1_Extended", "F2_Extended"]}).to_csv(task_segments_csv, index=False)
    monkeypatch.setitem(fd.AUTHORIZED_SEGMENTS_CSV, "FRAIWAN_Extended", task_segments_csv)

    authorized = fd.load_authorized_audio_ids("FRAIWAN_Extended")
    assert authorized == {"F1_Extended", "F2_Extended"}

    # F1 tiene, ademas de su Extended, grabaciones Bell/Diaphragm admitidas
    # en fase 2 bajo el MISMO patient_uid: filtrar solo por paciente las
    # admitiria por error.
    admitted = pd.DataFrame([
        _admitted_row("F1_Extended", "F1", "FRAIWAN", "f1_extended.wav"),
        _admitted_row("F1_Bell", "F1", "FRAIWAN", "f1_bell.wav"),
        _admitted_row("F1_Diaphragm", "F1", "FRAIWAN", "f1_diaphragm.wav"),
        _admitted_row("F2_Extended", "F2", "FRAIWAN", "f2_extended.wav"),
    ])
    roles = pd.DataFrame([
        {"patient_uid": "F1", "role": "train"}, {"patient_uid": "F2", "role": "test"},
    ])

    scope_meta = fd.select_scope_recordings(admitted, authorized, roles)

    assert set(scope_meta["audio_id"]) == {"F1_Extended", "F2_Extended"}


def test_select_scope_recordings_raises_when_selection_is_empty():
    admitted = pd.DataFrame([_admitted_row("F1_Bell", "F1", "FRAIWAN", "f1_bell.wav")])
    roles = pd.DataFrame([{"patient_uid": "F1", "role": "train"}])
    with pytest.raises(ValueError, match="ningun audio_id autorizado"):
        fd.select_scope_recordings(admitted, {"F1_Extended"}, roles)


# ---------------------------------------------------------------------------
# verify_phase2_audio_hashes: el audio remuestreado no debe haber cambiado
# desde que la fase 2 registro su output_sha256.
# ---------------------------------------------------------------------------

def test_verify_phase2_audio_hashes_rejects_mismatched_sha(tmp_path):
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"contenido de prueba")
    scope_meta = pd.DataFrame([{
        "audio_id": "A_rec", "output_path": str(audio_path), "output_sha256": "0" * 64,
    }])
    with pytest.raises(ValueError, match="sha256"):
        fd.verify_phase2_audio_hashes(scope_meta)


def test_verify_phase2_audio_hashes_passes_when_sha_matches(tmp_path):
    audio_path = tmp_path / "a.wav"
    audio_path.write_bytes(b"contenido de prueba")
    scope_meta = pd.DataFrame([{
        "audio_id": "A_rec", "output_path": str(audio_path), "output_sha256": u.file_sha256(audio_path),
    }])
    fd.verify_phase2_audio_hashes(scope_meta)  # no debe lanzar


# ---------------------------------------------------------------------------
# recalibrate_denoising: Fraiwan fijo, ICBHI/COMBINED recalibran con train.
# ---------------------------------------------------------------------------

def _admitted_row(audio_id, patient_uid, dataset, output_path):
    return {
        "audio_id": audio_id, "dataset": dataset, "patient_uid": patient_uid,
        "diagnosis": "COPD", "device": "Meditron", "filter": "", "zone": "AL",
        "quality_status": "PASS", "quality_reasons": "", "output_path": output_path,
        "samples_out_actual": int(SR * 5.0), "status": "OK",
    }


def test_fraiwan_never_recalibrates_denoising_params(tmp_path, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("sweep_denoising no debe llamarse para Fraiwan")
    monkeypatch.setattr(p3, "sweep_denoising", _boom)

    admitted = pd.DataFrame([
        _admitted_row("F1_rec", "F1", "FRAIWAN", "f1.wav"),
        _admitted_row("F2_rec", "F2", "FRAIWAN", "f2.wav"),
    ])
    roles = pd.DataFrame([
        {"patient_uid": "F1", "role": "train"}, {"patient_uid": "F2", "role": "test"},
    ])
    monkeypatch.setattr(p3, "compute_rms_distribution", lambda meta: pd.DataFrame({"rms_post_bandpass": [0.03]}))

    params = fd.recalibrate_denoising("FRAIWAN_Extended", admitted, roles)

    assert params["nperseg"] == int(fd.cfg.STFT_NPERSEG)
    assert params["noverlap"] == int(fd.cfg.STFT_NOVERLAP)
    assert params["noise_pct"] == float(fd.cfg.NOISE_PCT)
    assert params["oversubtraction"] == float(fd.cfg.OVERSUBTRACTION)
    assert params["spectral_floor"] == float(fd.cfg.SPECTRAL_FLOOR)
    assert params["denoising_fit_patient_ids"] == []


def test_icbhi_recalibrates_denoising_params_using_only_train(tmp_path, monkeypatch):
    calls = {}

    def _fake_sweep(calib_meta, nperseg, noverlap):
        calls["calib_meta_patients"] = sorted(calib_meta["patient_uid"].unique())
        calls["nperseg"] = nperseg
        calls["noverlap"] = noverlap
        return pd.DataFrame([{
            "noise_pct": 10, "oversubtraction": 3.0, "spectral_floor": 0.02,
            "cycle_gap_power_ratio_db": 2.0, "cycle_correlation": 0.99, "cycle_correlation_p10": 0.98,
            "crackle_correlation": 0.98, "crackle_correlation_p10": 0.98,
            "wheeze_correlation": 0.98, "wheeze_correlation_p10": 0.98,
            "musical_noise_ratio": 1.0, "musical_noise_ratio_p90": 1.1,
            "spectral_distortion_db": 3.0, "spectral_distortion_db_p90": 4.0,
        }])

    def _fake_select(table):
        row = table.iloc[0]
        return row, 1, 1, float(row["cycle_gap_power_ratio_db"])

    monkeypatch.setattr(p3, "annotation_gap_recordings", lambda: {"P1_rec"})
    monkeypatch.setattr(p3, "sweep_denoising", _fake_sweep)
    monkeypatch.setattr(p3, "select_robust_candidate", _fake_select)
    monkeypatch.setattr(p3, "compute_rms_distribution", lambda meta: pd.DataFrame({"rms_post_bandpass": [0.025, 0.035]}))

    admitted = pd.DataFrame([
        _admitted_row("P1_rec", "P1", "ICBHI", "p1.wav"),
        _admitted_row("P2_rec", "P2", "ICBHI", "p2.wav"),  # test: no debe entrar al calib_meta
    ])
    roles = pd.DataFrame([
        {"patient_uid": "P1", "role": "train"}, {"patient_uid": "P2", "role": "test"},
    ])

    params = fd.recalibrate_denoising("ICBHI", admitted, roles)

    assert calls["calib_meta_patients"] == ["P1"]  # solo train, nunca P2 (test)
    assert calls["nperseg"] == fd.cfg.STFT_NPERSEG
    assert calls["noverlap"] == fd.cfg.STFT_NOVERLAP
    assert params["noise_pct"] == 10
    assert params["oversubtraction"] == 3.0
    assert params["spectral_floor"] == 0.02
    assert params["denoising_fit_patient_ids"] == ["P1"]
    assert params["target_rms"] == pytest.approx(0.03)  # mediana de [0.025, 0.035]


def test_icbhi_raises_when_no_train_patient_has_annotation_gap(monkeypatch):
    monkeypatch.setattr(p3, "annotation_gap_recordings", lambda: set())  # ningun audio_id califica
    admitted = pd.DataFrame([_admitted_row("P1_rec", "P1", "ICBHI", "p1.wav")])
    roles = pd.DataFrame([{"patient_uid": "P1", "role": "train"}])

    with pytest.raises(ValueError, match="hueco anotado"):
        fd.recalibrate_denoising("ICBHI", admitted, roles)


def test_verify_no_leakage_raises_when_test_patient_fed_calibration():
    roles = pd.DataFrame([
        {"patient_uid": "P1", "role": "train"}, {"patient_uid": "P2", "role": "test"},
    ])
    params = {"denoising_fit_patient_ids": ["P1", "P2"], "target_rms_fit_patient_ids": ["P1"]}
    with pytest.raises(RuntimeError, match="fuga de calibracion"):
        fd.verify_no_leakage_into_calibration(roles, params)


def test_verify_no_leakage_passes_when_only_train_used():
    roles = pd.DataFrame([
        {"patient_uid": "P1", "role": "train"}, {"patient_uid": "P2", "role": "test"},
    ])
    params = {"denoising_fit_patient_ids": ["P1"], "target_rms_fit_patient_ids": ["P1"]}
    fd.verify_no_leakage_into_calibration(roles, params)  # no debe lanzar


# ---------------------------------------------------------------------------
# process_fold_recordings + build_fold_segment_inventory + fill_fold_arrays:
# audio sintetico real (senos + ruido), sin mockear el DSP.
# ---------------------------------------------------------------------------

def test_process_and_segment_synthetic_recordings(tmp_path, monkeypatch):
    signals = {
        "a.wav": _sine_plus_noise(5.0, freq=150.0, seed=1),   # 1 segmento exacto
        "b.wav": _sine_plus_noise(7.5, freq=300.0, seed=2),   # 2 segmentos (50% solape)
    }

    def _fake_read_audio(path):
        return signals[Path(path).name], SR
    monkeypatch.setattr(fd.u, "read_audio", _fake_read_audio)

    admitted = pd.DataFrame([
        _admitted_row("A_rec", "PA", "ICBHI", "a.wav"),
        _admitted_row("B_rec", "PB", "ICBHI", "b.wav"),
    ])
    admitted.loc[admitted["audio_id"] == "B_rec", "samples_out_actual"] = signals["b.wav"].size
    roles = pd.DataFrame([
        {"patient_uid": "PA", "role": "train", "target_label": 1, "source_dataset": "ICBHI", "calibration_patient": False},
        {"patient_uid": "PB", "role": "test", "target_label": 0, "source_dataset": "ICBHI", "calibration_patient": False},
    ])
    params = {
        "nperseg": 256, "noverlap": 192, "noise_pct": 15, "oversubtraction": 4.0,
        "spectral_floor": 0.05, "target_rms": 0.03, "max_gain": 20.0,
    }

    processed = fd.process_fold_recordings(admitted, params)
    assert set(processed["no_dn_signals"]) == {"A_rec", "B_rec"}
    assert set(processed["dn_signals"]) == {"A_rec", "B_rec"}
    assert all(np.isfinite(sig).all() for sig in processed["no_dn_signals"].values())
    assert all(np.isfinite(sig).all() for sig in processed["dn_signals"].values())

    inventory = fd.build_fold_segment_inventory(
        "ICBHI", 2, processed["scope_meta"], roles,
        processed["dn_reliable_map"], processed["dn_reason_map"], cycles_by_audio={},
    )
    assert (inventory.loc[inventory["audio_id"] == "A_rec"].shape[0]) == 1
    assert (inventory.loc[inventory["audio_id"] == "B_rec"].shape[0]) == 2
    assert set(inventory["fold_id"]) == {2}
    assert set(inventory.loc[inventory["patient_uid"] == "PA", "role"]) == {"train"}
    assert set(inventory.loc[inventory["patient_uid"] == "PB", "role"]) == {"test"}
    assert inventory["task_array_index"].tolist() == list(range(len(inventory)))
    assert (inventory["source_array_index"] == inventory["task_array_index"]).all()
    for col in fd.REQUIRED_SEGMENT_COLUMNS:
        assert col in inventory.columns, col

    no_dn_array, dn_array = fd.fill_fold_arrays(inventory, processed["no_dn_signals"], processed["dn_signals"])
    assert no_dn_array.shape == (len(inventory), p4.WINDOW_SAMPLES)
    assert dn_array.shape == no_dn_array.shape
    assert np.isfinite(no_dn_array).all() and np.isfinite(dn_array).all()

    fd.validate_fold_output(inventory, no_dn_array, dn_array)  # no debe lanzar


def test_validate_fold_output_rejects_mismatched_branch_shapes():
    inventory = pd.DataFrame({col: [0] for col in fd.REQUIRED_SEGMENT_COLUMNS})
    inventory["dn_reliable"] = True
    no_dn = np.zeros((1, 20000), dtype=np.float32)
    dn = np.zeros((1, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="formas distintas"):
        fd.validate_fold_output(inventory, no_dn, dn)


# ---------------------------------------------------------------------------
# run_fold: orquestacion completa, con phase3/phase4/utils mockeados a nivel
# de sus funciones de entrada (admitted_recordings, _load_cycle_bounds,
# read_audio) para nunca tocar preprocessing/data/ ni el corpus real.
# ---------------------------------------------------------------------------

def test_run_fold_end_to_end_publishes_atomically(tmp_path, monkeypatch):
    # 5 pacientes, uno por fold_group (0..4), todos con grabacion admitida
    # real: para fold_id=0, train = grupos {2,3,4} = F2,F3,F4 (con audio).
    patient_ids = [f"F{i}" for i in range(5)]
    signals = {f"{p}.wav": _sine_plus_noise(5.0, seed=10 + i) for i, p in enumerate(patient_ids)}
    monkeypatch.setattr(fd.u, "read_audio", lambda path: (signals[Path(path).name], SR))
    monkeypatch.setattr(fd.p3, "admitted_recordings", lambda: pd.DataFrame([
        _admitted_row(f"{p}_rec", p, "FRAIWAN", f"{p}.wav") for p in patient_ids
    ]))
    monkeypatch.setattr(fd.p4, "_load_cycle_bounds", lambda: {})
    # Lista autorizada sintetica (no el segments.csv real de modeling/) y
    # verificacion de sha256 de fase 2 desactivada: esta prueba cubre la
    # orquestacion/publicacion atomica, no esas dos verificaciones -cada una
    # tiene su propia prueba dedicada arriba.
    monkeypatch.setattr(fd, "load_authorized_audio_ids", lambda scope: {f"{p}_rec" for p in patient_ids})
    monkeypatch.setattr(fd, "verify_phase2_audio_hashes", lambda scope_meta: None)

    roles_by_patient = {p: (i, i % 2) for i, p in enumerate(patient_ids)}
    csv_path, manifest_path = _write_patient_folds(tmp_path, "FRAIWAN_Extended", roles_by_patient, source_dataset="FRAIWAN")

    output_root = tmp_path / "fold_calibrated"
    manifest = fd.run_fold("FRAIWAN_Extended", 0, csv_path, manifest_path, output_root)

    target_dir = output_root / "FRAIWAN_Extended" / "fold_00"
    assert manifest["verdict"] == "PASS"
    assert (target_dir / "segments.csv").is_file()
    assert (target_dir / "segments_no_dn.npy").is_file()
    assert (target_dir / "segments_dn.npy").is_file()
    assert (target_dir / "preprocessing_params.json").is_file()
    assert (target_dir / "manifest.json").is_file()
    assert not (output_root / "FRAIWAN_Extended" / "fold_00_staging").exists()

    on_disk_manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk_manifest["verdict"] == "PASS"
    assert on_disk_manifest["patient_folds_csv_sha256"] == u.file_sha256(csv_path)
    params_on_disk = json.loads((target_dir / "preprocessing_params.json").read_text(encoding="utf-8"))
    assert params_on_disk["noise_pct"] == float(fd.cfg.NOISE_PCT)  # Fraiwan: fijo

    # Re-ejecutar debe volver a publicar limpio (sin dejar staging huerfano).
    manifest2 = fd.run_fold("FRAIWAN_Extended", 0, csv_path, manifest_path, output_root)
    assert manifest2["verdict"] == "PASS"
    assert not (output_root / "FRAIWAN_Extended" / "fold_00_staging").exists()
    assert not (output_root / "FRAIWAN_Extended" / "fold_00_previous_swap").exists()
