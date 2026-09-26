"""fold_denoising.py --protocol holdout-v3 --stage cv (PLAN-EXPERIMENTO FINAL.md):
folds internos SOLO sobre el 80 % de desarrollo, con roles train/validation, sin
ninguna huella de la prueba externa (ni en el CSV, ni en los NPY, ni en la
calibracion ni en las estadisticas), calibracion de denoising y TARGET_RMS solo
con train, verificacion de hashes de fase 2, correspondencia exacta entre CSV y
arreglos, y publicacion atomica. Audio sintetico (senos + ruido) escrito en
tmp_path: nunca toca el corpus real ni preprocessing/data/.

``preprocessing/`` no es un paquete Python: este archivo sigue la convencion de
test_fold_denoising.py (``sys.path.insert`` + imports planos).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

import fold_denoising as fd
import phase3_cleaning as p3
import phase4_temporal as p4
import utils as u

SR = 4000
SCOPE = "FRAIWAN_Extended"


def _signal(seed: int, seconds: float = 5.0, freq: float = 200.0) -> np.ndarray:
    n = int(round(seconds * SR))
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    return (0.2 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(n)).astype(np.float64)


# ---------------------------------------------------------------------------
# Split sintetico: 10 pacientes de desarrollo (grupos 0..4, dos por grupo) y 2
# de prueba externa (inner_fold_group = -1).
# ---------------------------------------------------------------------------

DEV = [f"F{i:02d}" for i in range(10)]
BLOCKED = ["T00", "T01"]


def _patients() -> dict:
    patients = {uid: ("development", i % 5, i % 2) for i, uid in enumerate(DEV)}
    patients.update({uid: ("test", -1, i % 2) for i, uid in enumerate(BLOCKED)})
    return patients


def _write_split(tmp_path, scope=SCOPE, source="FRAIWAN", patients=None):
    """``patients``: patient_uid -> (outer_role, inner_fold_group, target_label)."""
    rows = [
        {
            "dataset_scope": scope, "patient_uid": uid, "source_dataset": source, "target_label": label,
            "outer_role": role, "inner_fold_group": group, "calibration_patient": False,
        }
        for uid, (role, group, label) in (patients or _patients()).items()
    ]
    csv_path = tmp_path / "holdout_splits.csv"
    manifest_path = tmp_path / "holdout_splits_manifest.json"
    pd.DataFrame(rows).to_csv(csv_path, index=False, lineterminator="\n")
    manifest_path.write_text(json.dumps({
        "protocol": "holdout-v3", "holdout_splits_csv_sha256": u.file_sha256(csv_path),
    }), encoding="utf-8")
    return csv_path, manifest_path


# ---------------------------------------------------------------------------
# load_holdout_cv_roles
# ---------------------------------------------------------------------------

def test_roles_have_only_train_and_validation_and_never_include_the_external_test(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)

    roles, blocked = fd.load_holdout_cv_roles(csv_path, manifest_path, SCOPE, fold_id=1)

    assert set(roles["role"]) == {"train", "validation"}
    assert set(roles.loc[roles["role"] == "validation", "patient_uid"]) == {"F01", "F06"}    # grupo 1
    assert set(roles.loc[roles["role"] == "train", "patient_uid"]) == set(DEV) - {"F01", "F06"}
    assert set(roles["patient_uid"]) == set(DEV)                     # 10 de desarrollo, ninguno de prueba
    assert blocked == frozenset(BLOCKED)
    assert not blocked & set(roles["patient_uid"])


def test_every_development_patient_is_validation_in_exactly_one_fold(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    validated = []
    for fold_id in range(5):
        roles, _ = fd.load_holdout_cv_roles(csv_path, manifest_path, SCOPE, fold_id)
        validated += roles.loc[roles["role"] == "validation", "patient_uid"].tolist()
    assert sorted(validated) == sorted(DEV)


def test_roles_reject_a_csv_that_changed_after_its_manifest(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(ValueError, match="sha256"):
        fd.load_holdout_cv_roles(csv_path, manifest_path, SCOPE, 0)


def test_roles_reject_a_manifest_of_another_protocol(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "fold-aware-v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        fd.load_holdout_cv_roles(csv_path, manifest_path, SCOPE, 0)


def test_roles_reject_a_test_patient_with_a_fold_group_and_out_of_range_folds(tmp_path):
    patients = _patients()
    patients["T00"] = ("test", 2, 0)
    csv_path, manifest_path = _write_split(tmp_path, patients=patients)
    with pytest.raises(ValueError, match="prueba"):
        fd.load_holdout_cv_roles(csv_path, manifest_path, SCOPE, 0)

    valid = tmp_path / "valid"
    valid.mkdir()
    csv_path, manifest_path = _write_split(valid)
    with pytest.raises(ValueError, match="fuera de rango"):
        fd.load_holdout_cv_roles(csv_path, manifest_path, SCOPE, 5)


# ---------------------------------------------------------------------------
# Ausencia de la prueba externa: verificaciones
# ---------------------------------------------------------------------------

def _inventory(patients, roles=None):
    return pd.DataFrame({"patient_uid": patients, "role": roles or ["train"] * len(patients)})


def _params(fit_denoising=(), fit_rms=()):
    return {"denoising_fit_patient_ids": list(fit_denoising), "target_rms_fit_patient_ids": list(fit_rms)}


def test_verify_outer_test_excluded_passes_when_nobody_leaked():
    scope_meta = pd.DataFrame({"patient_uid": ["F00", "F01"]})
    fd.verify_outer_test_excluded(
        frozenset(BLOCKED), scope_meta=scope_meta, inventory=_inventory(["F00", "F01"]),
        params=_params(["F00"], ["F00", "F01"]),
    )


@pytest.mark.parametrize("where", ("grabaciones procesadas", "segments.csv", "calibracion"))
def test_verify_outer_test_excluded_detects_a_leak_anywhere(where):
    clean = ["F00", "F01"]
    scope = pd.DataFrame({"patient_uid": clean + (["T00"] if where == "grabaciones procesadas" else [])})
    inventory = _inventory(clean + (["T01"] if where == "segments.csv" else []))
    params = _params(fit_rms=clean + (["T00"] if where == "calibracion" else []))
    with pytest.raises(RuntimeError, match="prueba externa"):
        fd.verify_outer_test_excluded(frozenset(BLOCKED), scope_meta=scope, inventory=inventory, params=params)


def test_verify_inventory_matches_roles_requires_exactly_the_split_patients():
    roles = pd.DataFrame({"patient_uid": ["A", "B", "C"], "role": ["train", "train", "validation"]})
    inventory = _inventory(["A", "B", "C"], ["train", "train", "validation"])
    fd.verify_inventory_matches_roles(inventory, roles)                                  # no lanza

    with pytest.raises(RuntimeError, match="sin segmentos"):                              # falta B
        fd.verify_inventory_matches_roles(_inventory(["A", "C"], ["train", "validation"]), roles)
    with pytest.raises(RuntimeError, match="roles train y validation"):                   # aparece un rol test
        fd.verify_inventory_matches_roles(_inventory(["A", "C"], ["train", "test"]), roles)
    with pytest.raises(RuntimeError, match="sobrantes"):                                  # sobra D
        fd.verify_inventory_matches_roles(
            _inventory(["A", "B", "C", "D"], ["train", "train", "validation", "train"]), roles)


# ---------------------------------------------------------------------------
# La calibracion usa SOLO los pacientes de train del fold interno
# ---------------------------------------------------------------------------

def _admitted_row(audio_id, patient_uid, dataset, output_path="x.wav", sha=""):
    return {
        "audio_id": audio_id, "dataset": dataset, "patient_uid": patient_uid,
        "diagnosis": "COPD", "device": "Meditron", "filter": "", "zone": "AL",
        "quality_status": "PASS", "quality_reasons": "", "output_path": str(output_path),
        "output_sha256": sha, "samples_out_actual": int(SR * 5.0), "status": "OK",
    }


def test_denoising_and_rms_calibration_use_only_train_patients_of_the_fold(tmp_path, monkeypatch):
    patients = {"P0": ("development", 0, 1), "P1": ("development", 1, 0), "P2": ("development", 2, 1),
                "P3": ("development", 3, 0), "P4": ("development", 4, 1), "T0": ("test", -1, 0)}
    csv_path, manifest_path = _write_split(tmp_path, scope="ICBHI", source="ICBHI", patients=patients)
    roles, blocked = fd.load_holdout_cv_roles(csv_path, manifest_path, "ICBHI", fold_id=0)   # validation = P0

    admitted = pd.DataFrame([_admitted_row(f"{p}_rec", p, "ICBHI") for p in patients])
    scope_meta = fd.select_scope_recordings(admitted, {f"{p}_rec" for p in patients}, roles)
    assert "T0" not in set(scope_meta["patient_uid"])                # la prueba externa no entra

    seen = {}

    def fake_sweep(calib_meta, nperseg, noverlap):
        seen["sweep_patients"] = sorted(calib_meta["patient_uid"].unique())
        return pd.DataFrame([{
            "noise_pct": 10, "oversubtraction": 3.0, "spectral_floor": 0.02,
            "cycle_gap_power_ratio_db": 2.0, "cycle_correlation": 0.99, "cycle_correlation_p10": 0.98,
            "crackle_correlation": 0.98, "crackle_correlation_p10": 0.98,
            "wheeze_correlation": 0.98, "wheeze_correlation_p10": 0.98,
            "musical_noise_ratio": 1.0, "musical_noise_ratio_p90": 1.1,
            "spectral_distortion_db": 3.0, "spectral_distortion_db_p90": 4.0,
        }])

    def fake_rms(meta):
        seen["rms_patients"] = sorted(meta["patient_uid"].unique())
        return pd.DataFrame({"rms_post_bandpass": [0.025, 0.035]})

    monkeypatch.setattr(p3, "annotation_gap_recordings", lambda: {f"{p}_rec" for p in patients})
    monkeypatch.setattr(p3, "sweep_denoising", fake_sweep)
    monkeypatch.setattr(p3, "select_robust_candidate", lambda table: (table.iloc[0], 1, 1, 2.0))
    monkeypatch.setattr(p3, "compute_rms_distribution", fake_rms)

    params = fd.recalibrate_denoising("ICBHI", scope_meta, roles)

    train = ["P1", "P2", "P3", "P4"]
    assert seen["sweep_patients"] == train and seen["rms_patients"] == train   # ni P0 (validation) ni T0 (prueba)
    assert params["denoising_fit_patient_ids"] == train and params["target_rms_fit_patient_ids"] == train
    fd.verify_no_leakage_into_calibration(roles, params)
    fd.verify_outer_test_excluded(blocked, scope_meta=scope_meta, inventory=_inventory(train), params=params)


# ---------------------------------------------------------------------------
# Correspondencia exacta entre segments.csv, segments_no_dn.npy y segments_dn.npy
# ---------------------------------------------------------------------------

def test_csv_and_both_npy_branches_correspond_row_by_row(monkeypatch):
    signals = {"a.wav": _signal(1, 5.0, 150.0), "b.wav": _signal(2, 7.5, 300.0)}
    monkeypatch.setattr(fd.u, "read_audio", lambda path: (signals[Path(path).name], SR))
    scope_meta = pd.DataFrame([
        _admitted_row("A_rec", "PA", "ICBHI", "a.wav"), _admitted_row("B_rec", "PB", "ICBHI", "b.wav"),
    ])
    scope_meta.loc[scope_meta["audio_id"] == "B_rec", "samples_out_actual"] = signals["b.wav"].size
    roles = pd.DataFrame([
        {"patient_uid": "PA", "role": "train", "target_label": 1, "source_dataset": "ICBHI", "calibration_patient": False},
        {"patient_uid": "PB", "role": "validation", "target_label": 0, "source_dataset": "ICBHI", "calibration_patient": False},
    ])
    params = {"nperseg": 256, "noverlap": 192, "noise_pct": 15, "oversubtraction": 4.0,
              "spectral_floor": 0.05, "target_rms": 0.03, "max_gain": 20.0}

    processed = fd.process_fold_recordings(scope_meta, params)
    inventory = fd.build_fold_segment_inventory(
        "ICBHI", 0, processed["scope_meta"], roles, processed["dn_reliable_map"], processed["dn_reason_map"], {},
    )
    no_dn, dn = fd.fill_fold_arrays(inventory, processed["no_dn_signals"], processed["dn_signals"])

    assert no_dn.shape == dn.shape == (len(inventory), p4.WINDOW_SAMPLES) and len(inventory) == 3
    assert inventory["task_array_index"].tolist() == list(range(len(inventory)))
    for i, seg in enumerate(inventory.itertuples()):
        assert seg.end_sample - seg.start_sample == p4.WINDOW_SAMPLES
        assert np.array_equal(no_dn[i], processed["no_dn_signals"][seg.audio_id][seg.start_sample:seg.end_sample])
        assert np.array_equal(dn[i], processed["dn_signals"][seg.audio_id][seg.start_sample:seg.end_sample])
    assert set(inventory["role"]) == {"train", "validation"}
    fd.validate_fold_output(inventory, no_dn, dn)


# ---------------------------------------------------------------------------
# run_fold_holdout_v3 de punta a punta, con WAV sinteticos reales
# ---------------------------------------------------------------------------

def _prepare_world(tmp_path, monkeypatch):
    """WAV reales (uno de 5 s por paciente, de desarrollo Y de prueba), lista
    autorizada y reporte de fase 2 sinteticos; el resto del pipeline es real."""
    patients = _patients()
    rows = []
    for i, uid in enumerate(patients):
        path = tmp_path / "audio" / f"{uid}.wav"
        path.parent.mkdir(exist_ok=True)
        sf.write(str(path), _signal(100 + i), SR)
        rows.append(_admitted_row(f"{uid}_ext", uid, "FRAIWAN", path, u.file_sha256(path)))
    admitted = pd.DataFrame(rows)

    authorized = tmp_path / "fraiwan_extended_segments.csv"      # incluye TAMBIEN a los de prueba: el
    pd.DataFrame({"audio_id": admitted["audio_id"]}).to_csv(authorized, index=False)   # split los excluye
    report = tmp_path / "2b_resampling.csv"
    admitted.to_csv(report, index=False)

    monkeypatch.setattr(fd.p3, "admitted_recordings", lambda: admitted)
    monkeypatch.setattr(fd.p4, "_load_cycle_bounds", lambda: {})
    monkeypatch.setitem(fd.AUTHORIZED_SEGMENTS_CSV, SCOPE, authorized)
    monkeypatch.setattr(fd.cfg, "R2_RESAMPLING", report)
    csv_path, manifest_path = _write_split(tmp_path)
    return admitted, csv_path, manifest_path, authorized, report


def test_run_fold_publishes_only_development_patients_with_train_and_validation_roles(tmp_path, monkeypatch):
    admitted, csv_path, manifest_path, authorized, report = _prepare_world(tmp_path, monkeypatch)
    calibrated = []
    original_rms = p3.compute_rms_distribution
    monkeypatch.setattr(fd.p3, "compute_rms_distribution",
                        lambda meta: calibrated.append(set(meta["patient_uid"])) or original_rms(meta))
    output_root = tmp_path / "holdout_calibrated"

    manifest = fd.run_fold_holdout_v3(SCOPE, 1, csv_path, manifest_path, output_root, stage="cv")

    target = output_root / SCOPE / "cv" / "fold_01"
    for name in ("segments.csv", "segments_no_dn.npy", "segments_dn.npy", "preprocessing_params.json", "manifest.json"):
        assert (target / name).is_file(), name
    assert not (output_root / SCOPE / "cv" / "fold_01_staging").exists()

    segments = pd.read_csv(target / "segments.csv", dtype={"patient_uid": str})
    assert set(segments["patient_uid"]) == set(DEV)                       # exactamente los 10 de desarrollo
    assert not set(segments["patient_uid"]) & set(BLOCKED)                # ninguno de prueba
    assert set(segments["role"]) == {"train", "validation"}
    assert set(segments.loc[segments["role"] == "validation", "patient_uid"]) == {"F01", "F06"}
    assert (segments["fold_id"] == 1).all()

    no_dn, dn = np.load(target / "segments_no_dn.npy"), np.load(target / "segments_dn.npy")
    assert no_dn.shape == dn.shape == (len(segments), p4.WINDOW_SAMPLES) == (10, 20000)
    assert np.isfinite(no_dn).all() and np.isfinite(dn).all()

    # Calibracion solo con train (8 pacientes): ni validation ni prueba.
    train = set(DEV) - {"F01", "F06"}
    assert calibrated == [train]
    params = json.loads((target / "preprocessing_params.json").read_text(encoding="utf-8"))
    assert set(params["target_rms_fit_patient_ids"]) == train
    assert params["denoising_fit_patient_ids"] == []                       # Fraiwan: 5 parametros fijos
    assert params["noise_pct"] == float(fd.cfg.NOISE_PCT) and params["nperseg"] == int(fd.cfg.STFT_NPERSEG)
    assert not set(params["target_rms_fit_patient_ids"]) & (set(BLOCKED) | {"F01", "F06"})

    # El manifiesto liga el fold con el split, los datos de entrada y los parametros calculados.
    on_disk = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(manifest))
    assert on_disk["protocol"] == "holdout-v3" and on_disk["stage"] == "cv" and on_disk["fold_id"] == 1
    assert on_disk["holdout_splits_csv_sha256"] == u.file_sha256(csv_path)
    assert on_disk["holdout_splits_manifest_sha256"] == u.file_sha256(manifest_path)
    assert on_disk["counts_by_role"] == {"train": 8, "validation": 2}
    assert on_disk["outer_test"] == {"excluded": True, "n_blocked_patients": 2}
    assert on_disk["input_hashes"]["authorized_segments_csv_sha256"] == u.file_sha256(authorized)
    assert on_disk["input_hashes"]["phase2_resampling_report_sha256"] == u.file_sha256(report)
    assert on_disk["input_hashes"]["n_phase2_audios_verified"] == 10
    assert on_disk["calibrated_parameters"]["target_rms"] == pytest.approx(params["target_rms"])
    for name, digest in on_disk["output_hashes"].items():
        assert digest == u.file_sha256(target / name), name
    assert set(on_disk["output_hashes"]) == {
        "segments.csv", "segments_no_dn.npy", "segments_dn.npy", "preprocessing_params.json"}


def test_rerunning_a_fold_republishes_atomically_and_a_failure_keeps_the_previous_fold(tmp_path, monkeypatch):
    _, csv_path, manifest_path, *_ = _prepare_world(tmp_path, monkeypatch)
    output_root = tmp_path / "holdout_calibrated"
    fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, output_root)
    target = output_root / SCOPE / "cv" / "fold_00"
    before = (target / "manifest.json").read_text(encoding="utf-8")

    fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, output_root)          # se puede repetir
    assert not (output_root / SCOPE / "cv" / "fold_00_staging").exists()
    assert not (output_root / SCOPE / "cv" / "fold_00_previous_swap").exists()
    previous = (target / "manifest.json").read_text(encoding="utf-8")

    def boom():
        raise RuntimeError("fallo simulado al armar el manifiesto")

    monkeypatch.setattr(fd, "_git_state", boom)
    with pytest.raises(RuntimeError, match="simulado"):
        fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, output_root)
    assert not (output_root / SCOPE / "cv" / "fold_00_staging").exists()           # staging limpiado
    assert (target / "manifest.json").read_text(encoding="utf-8") == previous       # destino intacto
    assert json.loads(before)["verdict"] == "PASS"


def test_a_phase2_audio_with_a_wrong_sha_stops_the_fold_before_writing_anything(tmp_path, monkeypatch):
    admitted, csv_path, manifest_path, *_ = _prepare_world(tmp_path, monkeypatch)
    victim = Path(admitted.loc[admitted["patient_uid"] == "F03", "output_path"].iloc[0])
    sf.write(str(victim), _signal(999), SR)                        # el archivo cambio despues de la fase 2
    output_root = tmp_path / "holdout_calibrated"

    with pytest.raises(ValueError, match="sha256"):
        fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, output_root)
    assert not output_root.exists()


def test_a_fold_of_a_split_that_changed_after_the_manifest_is_rejected(tmp_path, monkeypatch):
    _, csv_path, manifest_path, *_ = _prepare_world(tmp_path, monkeypatch)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(ValueError, match="sha256"):
        fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, tmp_path / "holdout_calibrated")


def test_run_fold_rejects_an_unknown_stage(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    with pytest.raises(ValueError, match="stage"):
        fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, tmp_path / "out", stage="final")


# ---------------------------------------------------------------------------
# CLI y compatibilidad con fold-aware-v2
# ---------------------------------------------------------------------------

def test_cli_holdout_v3_runs_and_reports_the_blocked_test_patients(tmp_path, monkeypatch, capsys):
    _, csv_path, manifest_path, *_ = _prepare_world(tmp_path, monkeypatch)
    output_root = tmp_path / "holdout_calibrated"

    rc = fd.main([
        "--protocol", "holdout-v3", "--stage", "cv", "--dataset-scope", SCOPE, "--fold-id", "2",
        "--split-csv", str(csv_path), "--split-manifest", str(manifest_path), "--output-root", str(output_root),
    ])

    out = capsys.readouterr().out
    assert rc == 0 and "Veredicto       : PASS" in out and "2 pacientes bloqueados" in out
    assert (output_root / SCOPE / "cv" / "fold_02" / "manifest.json").is_file()


def test_cli_holdout_v3_reports_failures_with_exit_code_1(tmp_path, capsys):
    csv_path, manifest_path = _write_split(tmp_path)
    rc = fd.main([
        "--protocol", "holdout-v3", "--dataset-scope", SCOPE, "--fold-id", "9",
        "--split-csv", str(csv_path), "--split-manifest", str(manifest_path), "--output-root", str(tmp_path / "o"),
    ])
    assert rc == 1 and "FALLO" in capsys.readouterr().err


def test_parse_args_defaults_and_protocol_specific_flags():
    v3 = fd.parse_args(["--protocol", "holdout-v3", "--dataset-scope", "ICBHI", "--fold-id", "0"])
    assert v3.stage == "cv" and v3.output_root is None and v3.split_csv is None
    assert fd.HOLDOUT_OUTPUT_ROOT.name == "holdout_calibrated"
    assert fd.DEFAULT_HOLDOUT_SPLIT_CSV.name == "holdout_splits.csv"

    v2 = fd.parse_args(["--dataset-scope", "ICBHI", "--fold-id", "0"])
    assert v2.protocol == "fold-aware-v2" and v2.stage is None and v2.output_root is None
    assert fd.OUTPUT_ROOT.name == "fold_calibrated"

    with pytest.raises(SystemExit):                                # --stage solo aplica a holdout-v3
        fd.parse_args(["--stage", "cv", "--dataset-scope", "ICBHI", "--fold-id", "0"])


def test_fold_aware_v2_entry_points_are_untouched():
    for name in ("load_patient_roles", "run_fold", "recalibrate_denoising", "process_fold_recordings"):
        assert callable(getattr(fd, name))
    assert fd.OUTPUT_ROOT.parts[-2:] == ("data", "fold_calibrated")
