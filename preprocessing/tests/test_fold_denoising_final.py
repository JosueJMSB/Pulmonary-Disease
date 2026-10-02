"""fold_denoising.py --protocol holdout-v3 --stage final (PLAN-ENTRENAMIENTO-FINAL.md,
CORRECIONES.md): TODO el 80 % de desarrollo (sin fold interno) MAS, por primera
vez, la prueba externa -calibrando el denoising y TARGET_RMS EXCLUSIVAMENTE con
desarrollo, pero aplicando esos parametros tambien a la prueba-. Esta etapa ya
abre el test: la autorizacion COMPLETA (protocolo, estado 'approved', hashes
del split, hash y contenido del artefacto de procedencia y, para ICBHI,
revision por dispositivo) se valida ANTES de leer cualquier audio; cambiar
UNICAMENTE 'status' nunca basta por si solo. Si la salida final ya existe y
coincide exactamente con el split/seleccion/codigo actuales, se reutiliza sin
reprocesar audio; si existe pero algo cambio, se rechaza en vez de
reemplazarla en silencio. Audio sintetico (senos + ruido) escrito en
tmp_path: nunca toca el corpus real ni preprocessing/data/.

Sigue la misma convencion que test_fold_denoising_holdout.py (``sys.path.insert``
+ imports planos; ``preprocessing/`` no es un paquete Python).
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

DEV = [f"F{i:02d}" for i in range(10)]
BLOCKED = ["T00", "T01"]
DEFAULT_HP = {"lr": 0.001, "dropout": 0.2, "weight_decay": 0.0001, "batch_size": 4}


def _signal(seed: int, seconds: float = 5.0, freq: float = 200.0) -> np.ndarray:
    n = int(round(seconds * SR))
    t = np.arange(n) / SR
    rng = np.random.default_rng(seed)
    return (0.2 * np.sin(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(n)).astype(np.float64)


def _patients() -> dict:
    patients = {uid: ("development", i % 5, i % 2) for i, uid in enumerate(DEV)}
    patients.update({uid: ("test", -1, i % 2) for i, uid in enumerate(BLOCKED)})
    return patients


def _write_split(tmp_path, scope=SCOPE, source="FRAIWAN", patients=None):
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


def _write_best_hp(source_runs_root, *, dataset=SCOPE, condition="dn", branch="dn", run_id="toy_run"):
    """Artefacto de procedencia real: ``verify_final_access_authorized`` valida
    su contenido, no solo su hash. Devuelve ``(source_artifact_relativo, sha256)``."""
    best_dir = source_runs_root / "cnn" / run_id / "datasets" / dataset / condition
    best_dir.mkdir(parents=True, exist_ok=True)
    path = best_dir / "best_hyperparameters.json"
    path.write_text(json.dumps({
        "protocol": "holdout_cv_v3", "model": "cnn", "dataset": dataset, "condition": condition, "branch": branch,
        "config_index": 0, "hyperparameters": DEFAULT_HP, "median_best_epoch": 1,
    }), encoding="utf-8")
    relative = f"cnn/{run_id}/datasets/{dataset}/{condition}/best_hyperparameters.json"
    return relative, u.file_sha256(path)


def _write_selection(
    tmp_path, *, dataset=SCOPE, status="approved", split_csv=None, split_manifest=None,
    source_artifact=None, source_artifact_sha256=None, device_review_path="", device_review_sha256="",
    architecture="cnn", source_config_index=0, epochs=1, hyperparameters=None,
):
    """``verify_final_access_authorized`` valida tambien ``architecture``,
    ``source_config_index``, ``epochs`` y ``[pipelines.<dataset>.hyperparameters]``
    contra el artefacto de procedencia; los valores por omision coinciden con los
    que ``_write_best_hp`` escribe por omision para que el "mundo consistente" siga
    autorizandose."""
    path = tmp_path / "selected_pipelines.toml"
    csv_sha = u.file_sha256(split_csv) if split_csv else "0" * 64
    manifest_sha = u.file_sha256(split_manifest) if split_manifest else "0" * 64
    hp = hyperparameters if hyperparameters is not None else DEFAULT_HP
    path.write_text(f"""
protocol = "holdout_final_v1"

[split]
csv_sha256 = "{csv_sha}"
manifest_sha256 = "{manifest_sha}"

[pipelines.{dataset}]
status = "{status}"
architecture = "{architecture}"
condition = "dn"
branch = "dn"
source_config_index = {source_config_index}
epochs = {epochs}
source_artifact = "{source_artifact or 'missing/best_hyperparameters.json'}"
source_artifact_sha256 = "{source_artifact_sha256 or '0' * 64}"
device_review_path = "{device_review_path}"
device_review_sha256 = "{device_review_sha256}"

[pipelines.{dataset}.hyperparameters]
lr = {hp["lr"]}
dropout = {hp["dropout"]}
weight_decay = {hp["weight_decay"]}
batch_size = {hp["batch_size"]}
""", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_holdout_final_roles: a diferencia de load_holdout_cv_roles, la prueba
# externa SI aparece (role="test"), sin fold interno (todo el desarrollo es
# role="train").
# ---------------------------------------------------------------------------

def test_roles_cover_all_development_as_train_and_all_test_as_test(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)

    roles = fd.load_holdout_final_roles(csv_path, manifest_path, SCOPE)

    assert set(roles["role"]) == {"train", "test"}
    assert set(roles.loc[roles["role"] == "train", "patient_uid"]) == set(DEV)
    assert set(roles.loc[roles["role"] == "test", "patient_uid"]) == set(BLOCKED)
    assert set(roles["patient_uid"]) == set(DEV) | set(BLOCKED)   # a diferencia de la etapa cv: TODOS aparecen


def test_roles_reject_a_csv_that_changed_after_its_manifest(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(ValueError, match="sha256"):
        fd.load_holdout_final_roles(csv_path, manifest_path, SCOPE)


def test_roles_reject_a_manifest_of_another_protocol(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol"] = "fold-aware-v2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        fd.load_holdout_final_roles(csv_path, manifest_path, SCOPE)


def test_roles_reject_a_test_patient_with_a_fold_group(tmp_path):
    patients = _patients()
    patients["T00"] = ("test", 2, 0)
    csv_path, manifest_path = _write_split(tmp_path, patients=patients)
    with pytest.raises(ValueError, match="prueba"):
        fd.load_holdout_final_roles(csv_path, manifest_path, SCOPE)


# ---------------------------------------------------------------------------
# Autorizacion COMPLETA (CORRECIONES.md seccion 2): protocolo, estado,
# hashes del split, artefacto de procedencia (hash + contenido) y, para
# ICBHI, revision por dispositivo. Cambiar SOLO "status" nunca basta.
# ---------------------------------------------------------------------------

def test_load_final_selection_status_reads_the_declared_status(tmp_path):
    path = _write_selection(tmp_path, status="pending_device_review")
    assert fd.load_final_selection_status(path, SCOPE) == "pending_device_review"


def test_load_final_selection_status_rejects_an_unknown_dataset(tmp_path):
    path = _write_selection(tmp_path, dataset="ICBHI")
    with pytest.raises(ValueError, match=SCOPE):
        fd.load_final_selection_status(path, SCOPE)


def _authorized_world(tmp_path):
    """Split + artefacto de procedencia reales y mutuamente consistentes:
    el caso que SI debe autorizarse."""
    csv_path, manifest_path = _write_split(tmp_path)
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root)
    selection_path = _write_selection(
        tmp_path, status="approved", split_csv=csv_path, split_manifest=manifest_path,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    return csv_path, manifest_path, selection_path, source_runs_root


def test_verify_final_access_authorized_passes_with_a_consistent_world(tmp_path):
    csv_path, manifest_path, selection_path, source_runs_root = _authorized_world(tmp_path)
    pipeline = fd.verify_final_access_authorized(selection_path, SCOPE, csv_path, manifest_path, source_runs_root)
    assert pipeline["status"] == "approved"


def test_verify_final_access_authorized_rejects_pending_status(tmp_path):
    csv_path, manifest_path, _selection_path, source_runs_root = _authorized_world(tmp_path)
    relative, artifact_sha = _write_best_hp(source_runs_root)
    pending = _write_selection(
        tmp_path, status="pending_device_review", split_csv=csv_path, split_manifest=manifest_path,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    with pytest.raises(RuntimeError, match="pending_device_review"):
        fd.verify_final_access_authorized(pending, SCOPE, csv_path, manifest_path, source_runs_root)


def test_verify_final_access_authorized_rejects_a_split_hash_mismatch_even_if_approved(tmp_path):
    csv_path, manifest_path, selection_path, source_runs_root = _authorized_world(tmp_path)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")  # el split cambio despues de congelar la seleccion
    with pytest.raises(RuntimeError, match="sha256"):
        fd.verify_final_access_authorized(selection_path, SCOPE, csv_path, manifest_path, source_runs_root)


def test_verify_final_access_authorized_rejects_a_source_artifact_hash_mismatch(tmp_path):
    csv_path, manifest_path, selection_path, source_runs_root = _authorized_world(tmp_path)
    _write_best_hp(source_runs_root, condition="dn")  # mismo archivo, OTRO contenido (rehace el hash)
    (source_runs_root / "cnn" / "toy_run" / "datasets" / SCOPE / "dn" / "best_hyperparameters.json").write_text(
        json.dumps({**{"protocol": "holdout_cv_v3", "model": "cnn", "dataset": SCOPE, "condition": "dn", "branch": "dn",
                        "config_index": 0, "median_best_epoch": 1},
                    "hyperparameters": {**DEFAULT_HP, "lr": 0.5}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="sha256"):
        fd.verify_final_access_authorized(selection_path, SCOPE, csv_path, manifest_path, source_runs_root)


def test_verify_final_access_authorized_rejects_an_artifact_describing_another_pipeline(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    source_runs_root = tmp_path / "source_runs"
    # condition="dn" coincide con la seleccion (para aislar el mismatch); solo branch difiere.
    relative, artifact_sha = _write_best_hp(source_runs_root, condition="dn", branch="no_dn")
    selection_path = _write_selection(
        tmp_path, status="approved", split_csv=csv_path, split_manifest=manifest_path,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    with pytest.raises(RuntimeError, match="branch"):
        fd.verify_final_access_authorized(selection_path, SCOPE, csv_path, manifest_path, source_runs_root)


def test_approving_icbhi_falsely_without_device_review_is_rejected(tmp_path):
    """Cambiar UNICAMENTE status a 'approved' nunca basta por si solo para
    ICBHI: sin device_review_path/sha256 registrados, la autorizacion COMPLETA
    debe seguir rechazando el acceso al test (CORRECIONES.md seccion 2)."""
    csv_path, manifest_path = _write_split(tmp_path, scope="ICBHI", source="ICBHI")
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root, dataset="ICBHI", condition="dn", branch="dn")
    selection_path = _write_selection(
        tmp_path, dataset="ICBHI", status="approved", split_csv=csv_path, split_manifest=manifest_path,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    with pytest.raises(RuntimeError, match="device_review"):
        fd.verify_final_access_authorized(selection_path, "ICBHI", csv_path, manifest_path, source_runs_root)


def test_icbhi_passes_once_device_review_is_registered_and_matches(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path, scope="ICBHI", source="ICBHI")
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root, dataset="ICBHI", condition="dn", branch="dn")
    review_path = tmp_path / "device_review.csv"
    review_path.write_text("device,balanced_accuracy\nMeditron,0.95\n", encoding="utf-8")
    selection_path = _write_selection(
        tmp_path, dataset="ICBHI", status="approved", split_csv=csv_path, split_manifest=manifest_path,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
        device_review_path=review_path.as_posix(), device_review_sha256=u.file_sha256(review_path),
    )
    pipeline = fd.verify_final_access_authorized(selection_path, "ICBHI", csv_path, manifest_path, source_runs_root)
    assert pipeline["status"] == "approved"


# ---------------------------------------------------------------------------
# build_final_segment_inventory / verify_final_inventory_matches_roles
# ---------------------------------------------------------------------------

def _admitted_row(audio_id, patient_uid, dataset, output_path="x.wav", sha=""):
    return {
        "audio_id": audio_id, "dataset": dataset, "patient_uid": patient_uid,
        "diagnosis": "COPD", "device": "Meditron", "filter": "", "zone": "AL",
        "quality_status": "PASS", "quality_reasons": "", "output_path": str(output_path),
        "output_sha256": sha, "samples_out_actual": int(SR * 5.0), "status": "OK",
    }


def test_final_inventory_has_fold_id_minus_one_and_train_test_roles(monkeypatch):
    signals = {"a.wav": _signal(1, 5.0, 150.0), "b.wav": _signal(2, 7.5, 300.0)}
    monkeypatch.setattr(fd.u, "read_audio", lambda path: (signals[Path(path).name], SR))
    scope_meta = pd.DataFrame([
        _admitted_row("A_rec", "PA", "ICBHI", "a.wav"), _admitted_row("B_rec", "PB", "ICBHI", "b.wav"),
    ])
    scope_meta.loc[scope_meta["audio_id"] == "B_rec", "samples_out_actual"] = signals["b.wav"].size
    roles = pd.DataFrame([
        {"patient_uid": "PA", "role": "train", "target_label": 1, "source_dataset": "ICBHI", "calibration_patient": False},
        {"patient_uid": "PB", "role": "test", "target_label": 0, "source_dataset": "ICBHI", "calibration_patient": False},
    ])
    params = {"nperseg": 256, "noverlap": 192, "noise_pct": 15, "oversubtraction": 4.0,
              "spectral_floor": 0.05, "target_rms": 0.03, "max_gain": 20.0}

    processed = fd.process_fold_recordings(scope_meta, params)
    inventory = fd.build_final_segment_inventory(
        "ICBHI", processed["scope_meta"], roles, processed["dn_reliable_map"], processed["dn_reason_map"], {},
    )
    no_dn, dn = fd.fill_fold_arrays(inventory, processed["no_dn_signals"], processed["dn_signals"])

    assert (inventory["fold_id"] == -1).all()
    assert set(inventory["role"]) == {"train", "test"}
    assert set(inventory.loc[inventory["role"] == "test", "patient_uid"]) == {"PB"}
    fd.validate_fold_output(inventory, no_dn, dn)                 # mismas comprobaciones genericas
    fd.verify_final_inventory_matches_roles(inventory, roles)     # no lanza


def test_verify_final_inventory_matches_roles_requires_exactly_train_and_test():
    roles = pd.DataFrame({"patient_uid": ["A", "B", "C"], "role": ["train", "train", "test"]})

    def _inventory(patients, roles_list):
        return pd.DataFrame({"patient_uid": patients, "role": roles_list})

    fd.verify_final_inventory_matches_roles(_inventory(["A", "B", "C"], ["train", "train", "test"]), roles)

    with pytest.raises(RuntimeError, match="roles train y test"):
        fd.verify_final_inventory_matches_roles(_inventory(["A", "B"], ["train", "validation"]), roles)
    with pytest.raises(RuntimeError, match="sin segmentos"):
        fd.verify_final_inventory_matches_roles(_inventory(["A", "C"], ["train", "test"]), roles)   # falta B
    with pytest.raises(RuntimeError, match="sobrantes"):
        fd.verify_final_inventory_matches_roles(
            _inventory(["A", "B", "C", "D"], ["train", "train", "test", "train"]), roles,
        )


def test_verify_final_counts_detects_a_mismatch():
    fd.verify_final_counts("ICBHI", {"train": 72, "test": 18})          # no lanza: coincide con el plan
    with pytest.raises(RuntimeError, match="ICBHI"):
        fd.verify_final_counts("ICBHI", {"train": 71, "test": 18})
    fd.verify_final_counts("UNKNOWN_SCOPE", {"train": 1, "test": 1})    # sin conteo esperado: no verifica nada


# ---------------------------------------------------------------------------
# run_final_v1 de punta a punta, con WAV sinteticos reales
# ---------------------------------------------------------------------------

def _prepare_world(tmp_path, monkeypatch, status="approved"):
    """WAV reales (uno de 5 s por paciente, de desarrollo Y de prueba), lista
    autorizada, reporte de fase 2, artefacto de procedencia y seleccion
    congelada sinteticos, todos mutuamente consistentes."""
    patients = _patients()
    rows = []
    for i, uid in enumerate(patients):
        path = tmp_path / "audio" / f"{uid}.wav"
        path.parent.mkdir(exist_ok=True)
        sf.write(str(path), _signal(100 + i), SR)
        rows.append(_admitted_row(f"{uid}_ext", uid, "FRAIWAN", path, u.file_sha256(path)))
    admitted = pd.DataFrame(rows)

    authorized = tmp_path / "fraiwan_extended_segments.csv"
    pd.DataFrame({"audio_id": admitted["audio_id"]}).to_csv(authorized, index=False)
    report = tmp_path / "2b_resampling.csv"
    admitted.to_csv(report, index=False)

    monkeypatch.setattr(fd.p3, "admitted_recordings", lambda: admitted)
    monkeypatch.setattr(fd.p4, "_load_cycle_bounds", lambda: {})
    monkeypatch.setitem(fd.AUTHORIZED_SEGMENTS_CSV, SCOPE, authorized)
    monkeypatch.setattr(fd.cfg, "R2_RESAMPLING", report)
    # EXPECTED_FINAL_COUNTS trae los conteos REALES de ICBHI/FRAIWAN_Extended/COMBINED
    # (72/18, 34/9, 106/27); aqui se usan 10 pacientes de desarrollo y 2 de prueba, asi
    # que se sustituye SOLO la entrada de este scope por el conteo sintetico esperado.
    monkeypatch.setitem(fd.EXPECTED_FINAL_COUNTS, SCOPE, {"train": len(DEV), "test": len(BLOCKED)})

    csv_path, manifest_path = _write_split(tmp_path)
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root)
    selection_path = _write_selection(
        tmp_path, status=status, split_csv=csv_path, split_manifest=manifest_path,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    return admitted, csv_path, manifest_path, authorized, report, selection_path, source_runs_root


def test_run_final_v1_processes_development_and_test_with_the_same_parameters(tmp_path, monkeypatch):
    admitted, csv_path, manifest_path, authorized, report, selection_path, source_runs_root = _prepare_world(tmp_path, monkeypatch)
    calibrated = []
    original_rms = p3.compute_rms_distribution
    monkeypatch.setattr(fd.p3, "compute_rms_distribution",
                        lambda meta: calibrated.append(set(meta["patient_uid"])) or original_rms(meta))
    output_root = tmp_path / "holdout_calibrated"

    manifest = fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, output_root)

    target = output_root / SCOPE / "final"
    for name in ("segments.csv", "segments_no_dn.npy", "segments_dn.npy", "preprocessing_params.json", "manifest.json"):
        assert (target / name).is_file(), name
    assert not (output_root / SCOPE / "final_staging").exists()

    segments = pd.read_csv(target / "segments.csv", dtype={"patient_uid": str})
    assert set(segments["patient_uid"]) == set(DEV) | set(BLOCKED)        # AHORA la prueba SI aparece
    assert set(segments["role"]) == {"train", "test"}
    assert set(segments.loc[segments["role"] == "test", "patient_uid"]) == set(BLOCKED)
    assert (segments["fold_id"] == -1).all()

    no_dn, dn = np.load(target / "segments_no_dn.npy"), np.load(target / "segments_dn.npy")
    assert no_dn.shape == dn.shape == (len(segments), p4.WINDOW_SAMPLES)
    assert np.isfinite(no_dn).all() and np.isfinite(dn).all()

    # Calibracion con TODO el desarrollo (10 pacientes) y NUNCA con la prueba.
    assert calibrated == [set(DEV)]
    params = json.loads((target / "preprocessing_params.json").read_text(encoding="utf-8"))
    assert set(params["target_rms_fit_patient_ids"]) == set(DEV)
    assert not set(params["target_rms_fit_patient_ids"]) & set(BLOCKED)
    assert params["denoising_fit_patient_ids"] == []                      # Fraiwan: 5 parametros fijos

    on_disk = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(manifest))
    assert on_disk["protocol"] == "holdout-v3" and on_disk["stage"] == "final"
    assert on_disk["counts_by_role"] == {"train": 10, "test": 2}
    assert on_disk["outer_test"] == {"included": True, "n_patients": 2}   # a diferencia de la etapa cv: incluida
    assert on_disk["selection_status_at_generation"] == "approved"
    assert on_disk["selection_pipeline_fingerprint"] and on_disk["code_sha256"]
    for name, digest in on_disk["output_hashes"].items():
        assert digest == u.file_sha256(target / name), name


def test_run_final_v1_refuses_when_the_selection_is_not_approved(tmp_path, monkeypatch):
    _, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(
        tmp_path, monkeypatch, status="pending_device_review",
    )
    output_root = tmp_path / "holdout_calibrated"

    with pytest.raises(RuntimeError, match="pending_device_review"):
        fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, output_root)
    assert not output_root.exists()        # nada se toco: ni fase 2, ni calibracion, ni publicacion


def test_run_final_v1_rejects_a_split_that_changed_after_the_manifest(tmp_path, monkeypatch):
    _, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(tmp_path, monkeypatch)
    with open(csv_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
    with pytest.raises(RuntimeError, match="sha256"):
        fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, tmp_path / "holdout_calibrated")


def test_run_final_v1_stops_before_writing_anything_on_a_phase2_hash_mismatch(tmp_path, monkeypatch):
    admitted, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(tmp_path, monkeypatch)
    victim = Path(admitted.loc[admitted["patient_uid"] == "F03", "output_path"].iloc[0])
    sf.write(str(victim), _signal(999), SR)                           # el archivo cambio despues de la fase 2
    output_root = tmp_path / "holdout_calibrated"

    with pytest.raises(ValueError, match="sha256"):
        fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, output_root)
    assert not output_root.exists()


# ---------------------------------------------------------------------------
# Reutilizacion idempotente: una segunda llamada con la MISMA entrada no debe
# volver a leer ningun audio; si la entrada cambio, se rechaza en vez de
# reemplazar la salida en silencio (CORRECIONES.md seccion 3).
# ---------------------------------------------------------------------------

def test_rerunning_with_the_same_inputs_reuses_without_reading_audio(tmp_path, monkeypatch):
    _, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(tmp_path, monkeypatch)
    output_root = tmp_path / "holdout_calibrated"
    first = fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, output_root)

    def _trap(*args, **kwargs):
        raise AssertionError("la reutilizacion idempotente no debe volver a leer audio")

    monkeypatch.setattr(fd.p3, "admitted_recordings", _trap)
    monkeypatch.setattr(fd.u, "read_audio", _trap)

    second = fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, output_root)
    assert second == first
    assert not (output_root / SCOPE / "final_staging").exists()


def test_rerunning_after_the_split_changed_is_rejected_not_silently_replaced(tmp_path, monkeypatch):
    _, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(tmp_path, monkeypatch)
    output_root = tmp_path / "holdout_calibrated"
    fd.run_final_v1(SCOPE, csv_path, manifest_path, selection_path, source_runs_root, output_root)
    target = output_root / SCOPE / "final"
    before = (target / "manifest.json").read_text(encoding="utf-8")

    # Split distinto, pero sincronizado consigo mismo (csv y manifiesto coinciden
    # entre si): verify_final_access_authorized lo acepta -el split "oficial"
    # simplemente cambio-, pero la salida final anterior ya no coincide con el.
    other_patients = _patients()
    other_patients["F00"] = ("development", 3, 1)  # mismo paciente, otro grupo interno (irrelevante aqui, pero cambia el csv)
    (tmp_path / "other").mkdir()
    other_csv, other_manifest = _write_split(tmp_path / "other", patients=other_patients)
    relative, artifact_sha = _write_best_hp(source_runs_root)  # mismo artefacto real que _prepare_world ya escribio
    other_selection = _write_selection(
        tmp_path / "other", status="approved", split_csv=other_csv, split_manifest=other_manifest,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
    )

    with pytest.raises(RuntimeError, match="no se reemplaza"):
        fd.run_final_v1(SCOPE, other_csv, other_manifest, other_selection, source_runs_root, output_root)

    assert (target / "manifest.json").read_text(encoding="utf-8") == before   # destino intacto


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_stage_final_runs_and_reports_the_processed_test_patients(tmp_path, monkeypatch, capsys):
    _, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(tmp_path, monkeypatch)
    output_root = tmp_path / "holdout_calibrated"

    rc = fd.main([
        "--protocol", "holdout-v3", "--stage", "final", "--dataset-scope", SCOPE,
        "--split-csv", str(csv_path), "--split-manifest", str(manifest_path),
        "--selection-config", str(selection_path), "--source-runs-root", str(source_runs_root),
        "--output-root", str(output_root),
    ])

    out = capsys.readouterr().out
    assert rc == 0 and "Veredicto       : PASS" in out and "2 pacientes PROCESADOS" in out
    assert (output_root / SCOPE / "final" / "manifest.json").is_file()


def test_cli_stage_final_rejects_fold_id(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    with pytest.raises(SystemExit):
        fd.parse_args([
            "--protocol", "holdout-v3", "--stage", "final", "--dataset-scope", SCOPE, "--fold-id", "0",
            "--split-csv", str(csv_path), "--split-manifest", str(manifest_path),
        ])


def test_cli_stage_final_reports_blocked_selection_with_exit_code_1(tmp_path, monkeypatch, capsys):
    _, csv_path, manifest_path, _, _, selection_path, source_runs_root = _prepare_world(
        tmp_path, monkeypatch, status="pending_device_review",
    )
    rc = fd.main([
        "--protocol", "holdout-v3", "--stage", "final", "--dataset-scope", SCOPE,
        "--split-csv", str(csv_path), "--split-manifest", str(manifest_path),
        "--selection-config", str(selection_path), "--source-runs-root", str(source_runs_root),
        "--output-root", str(tmp_path / "out"),
    ])
    assert rc == 1 and "FALLO" in capsys.readouterr().err


def test_parse_args_stage_cv_still_requires_fold_id():
    with pytest.raises(SystemExit):
        fd.parse_args(["--protocol", "holdout-v3", "--stage", "cv", "--dataset-scope", "ICBHI"])


def test_run_fold_holdout_v3_still_rejects_the_final_stage(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    with pytest.raises(ValueError, match="run_final_v1"):
        fd.run_fold_holdout_v3(SCOPE, 0, csv_path, manifest_path, tmp_path / "out", stage="final")
