"""SVM-RBF de punta a punta en el protocolo holdout-v3, con datos sinteticos:
cada configuracion se evalua en los cinco folds internos, se elige una sola
configuracion global, la prueba externa nunca aparece (ni en los folds ni en las
predicciones), no se crea ningun modelo definitivo y ``--resume``/``--dry-run``
verifican la huella y el enlace con el split. No necesita PyTorch.
"""

import copy
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from .. import data as dmod
from .. import holdout_cv as hcv
from .. import holdout_svm as hsvm
from .. import splits as sp

N_SPLITS = 5


def _toy_segments(n_pos=20, n_neg=15):
    rows = []
    for label, n in ((1, n_pos), (0, n_neg)):
        for i in range(n):
            uid = f"TOY_{label}_{i:03d}"
            rows.append({
                "patient_uid": uid, "audio_id": f"{uid}_rec0", "segment_id": f"{uid}_rec0_00",
                "target_label": label, "calibration_patient": False, "dataset": "TOY",
            })
    return pd.DataFrame(rows)


def _toy_split_files(tmp_path):
    table = sp.assign_holdout_split(
        sp.build_patient_table(_toy_segments()), n_splits=N_SPLITS, random_state=20260914,
    )
    csv_path = tmp_path / "modeling_data" / "holdout_splits.csv"
    manifest_path = tmp_path / "modeling_data" / "holdout_splits_manifest.json"
    sp.write_holdout_split_csv(
        sp.holdout_split_to_frame(table, "TOY"), csv_path, manifest_path,
        {"seed": 20260914, "n_splits": N_SPLITS, "counts": {"TOY": {"n_patients": len(table)}}},
    )
    return sp.load_holdout_split(csv_path, manifest_path, "TOY"), csv_path, manifest_path


def _write_cv_fold(data_root, fold_id, split, csv_path, manifest_path, include_blocked=False):
    """``<data_root>/TOY/cv/fold_XX/`` con train+validation de ese fold interno
    (misma estructura que produce preprocessing/fold_denoising.py)."""
    fold_dir = data_root / "TOY" / "cv" / f"fold_{fold_id:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    train, val = split.cv_split(fold_id)
    label_by = dict(zip(split.patient_table["patient_uid"], split.patient_table["target_label"]))
    entries = [(p, "train") for p in train] + [(p, "validation") for p in val]
    if include_blocked:
        entries.append((sorted(split.blocked_test_patients())[0], "train"))

    rows = []
    for i, (uid, role) in enumerate(entries):
        label = int(label_by[uid])
        rows.append({
            "task_array_index": i, "source_array_index": i, "segment_id": f"{uid}_S", "audio_id": f"{uid}_rec0",
            "patient_uid": uid, "diagnosis": "COPD" if label else "Healthy", "target_label": label,
            "target_name": "COPD" if label else "Healthy", "calibration_patient": False, "dn_reliable": True,
            "role": role, "fold_id": fold_id, "dataset": "TOY", "source_dataset": "TOY",
            "device": "Meditron" if i % 2 == 0 else "AKG",
        })
    segments = pd.DataFrame(rows)
    segments.to_csv(fold_dir / "segments.csv", index=False)

    rng = np.random.default_rng(fold_id)
    # Senal separable por clase para que la SVM converja.
    offsets = np.where(segments["target_label"].to_numpy() == 1, 2.0, -2.0)
    array_no_dn = (rng.standard_normal((len(segments), 20000)) * 0.05 + offsets[:, None]).astype(np.float32)
    np.save(fold_dir / "segments_no_dn.npy", array_no_dn)
    np.save(fold_dir / "segments_dn.npy", array_no_dn * 0.9)

    (fold_dir / "manifest.json").write_text(json.dumps({
        "protocol": "holdout-v3", "stage": "cv", "dataset_scope": "TOY", "fold_id": fold_id,
        "holdout_splits_csv_sha256": dmod.sha256_file(csv_path),
        "holdout_splits_manifest_sha256": dmod.sha256_file(manifest_path),
        "output_hashes": {
            name: dmod.sha256_file(fold_dir / name)
            for name in ("segments.csv", "segments_no_dn.npy", "segments_dn.npy")
        },
    }), encoding="utf-8")


def _svm_cfg(tmp_path, csv_path, manifest_path):
    cfg = copy.deepcopy(dmod.load_config(dmod.HOLDOUT_FINAL_CONFIGS_DIR / "svm_rbf_v3.toml"))
    cfg["svm"]["c_grid"] = [1.0, 10.0]
    cfg["svm"]["gamma_grid"] = ["scale", 0.1]          # 4 configuraciones
    cfg["holdout"]["split_csv"] = str(csv_path)
    cfg["holdout"]["split_manifest"] = str(manifest_path)
    cfg["figures"] = {"dpi": 50, "formats": ["png"]}       # figuras pequenas para las pruebas
    cfg["datasets"] = {"TOY": {"positive_diagnosis": "COPD", "negative_diagnosis": "Healthy", "negative_label_name": "Healthy"}}
    cfg["experiments"] = {
        "main": {"TOY": {"condition": "no_dn", "branch": "no_dn", "dn_reliable_only": False}},
    }
    return cfg


def _args(tmp_path, data_root, **overrides):
    base = dict(
        model="svm_rbf", dataset="TOY", experiment="main", n_jobs=1, device="auto", num_workers=0,
        data_root=data_root, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache", config=None,
        dry_run=False, smoke_test=False, force_features=False, resume=None,
    )
    base.update(overrides)
    return Namespace(**base)


def _setup(tmp_path, folds=range(N_SPLITS), include_blocked=False):
    split, csv_path, manifest_path = _toy_split_files(tmp_path)
    data_root = tmp_path / "holdout_calibrated"
    for fold_id in folds:
        _write_cv_fold(data_root, fold_id, split, csv_path, manifest_path, include_blocked=include_blocked)
    return split, data_root, _svm_cfg(tmp_path, csv_path, manifest_path), csv_path, manifest_path


def _only_run(tmp_path):
    runs = list((tmp_path / "runs" / "svm_rbf").glob("*"))
    assert len(runs) == 1
    return runs[0]


# ---------------------------------------------------------------------------

def test_full_run_evaluates_every_configuration_in_five_folds_and_picks_one_global_winner(tmp_path):
    split, data_root, cfg, *_ = _setup(tmp_path)

    rc = hsvm.run_holdout_svm(_args(tmp_path, data_root), cfg)

    assert rc == 0
    run_root = _only_run(tmp_path)
    assert json.loads((run_root / "status.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"
    cdir = run_root / "datasets" / "TOY" / "no_dn"

    search = pd.read_csv(cdir / "cv_search_results.csv")
    assert len(search) == 4 * N_SPLITS                                    # 4 configuraciones x 5 folds
    assert set(search["status"]) == {"COMPLETED"}
    assert sorted(search["fold"].unique()) == list(range(N_SPLITS))
    best = json.loads((cdir / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert best["model"] == "svm_rbf" and best["expected_folds"] == N_SPLITS
    assert best["median_best_epoch"] is None and best["best_epochs_by_fold"] is None   # la SVM no tiene epocas
    assert len(pd.read_csv(cdir / "cv_config_summary.csv")) == 4

    # Solo pacientes de desarrollo, uno por paciente; nada de la prueba externa.
    patients = pd.read_csv(cdir / "cv_patient_predictions.csv", dtype={"patient_uid": str})
    assert sorted(patients["patient_uid"]) == split.development_patients()
    assert not set(patients["patient_uid"]) & split.blocked_test_patients()

    assert not (cdir / "final").exists()                                  # sin modelo definitivo
    assert (cdir / "cv" / "config_000" / "fold_00" / "_SUCCESS").is_file()

    # Tablas y figuras automaticamente.
    for figure in ("confusion_matrix", "roc", "pr", "metrics_by_fold"):
        assert (cdir / "figures" / f"{figure}.png").is_file(), figure

    # Tabla global completa: todas las metricas con media/desviacion entre folds y valor agrupado.
    summary = pd.read_csv(run_root / "cv_run_summary.csv")
    for metric in hcv.SUMMARY_METRICS:
        for prefix in ("fold_mean_", "fold_std_", "pooled_"):
            assert f"{prefix}{metric}" in summary.columns, f"{prefix}{metric}"

    # La huella guarda la version del codigo (hash reproducible de modeling/**/*.py).
    fingerprint = json.loads((run_root / "run_fingerprint.json").read_text(encoding="utf-8"))
    assert len(fingerprint["code_sha256"]) == 64 and "holdout_cv.py" in fingerprint["code_files"]
    manifest = json.loads((cdir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["hashes"]["code_sha256"] == fingerprint["code_sha256"]


def test_smoke_test_touches_only_fold_zero_and_the_first_configuration(tmp_path):
    _, data_root, cfg, *_ = _setup(tmp_path, folds=[0])     # los folds 1..4 no existen

    rc = hsvm.run_holdout_svm(_args(tmp_path, data_root, smoke_test=True), cfg)

    assert rc == 0
    cdir = _only_run(tmp_path) / "datasets" / "TOY" / "no_dn"
    search = pd.read_csv(cdir / "cv_search_results.csv")
    assert len(search) == 1 and search.loc[0, "fold"] == 0 and search.loc[0, "config_index"] == 0


def test_run_refuses_to_start_when_outer_test_or_final_model_is_enabled(tmp_path):
    _, data_root, cfg, *_ = _setup(tmp_path)
    for section in ("outer_test", "final_model"):
        bad = copy.deepcopy(cfg)
        bad[section]["enabled"] = True
        with pytest.raises(RuntimeError, match=section):
            hsvm.run_holdout_svm(_args(tmp_path, data_root), bad)
    assert not (tmp_path / "runs").exists()                  # el chequeo ocurre antes de crear nada


def test_a_fold_that_contains_an_outer_test_patient_fails_instead_of_training(tmp_path):
    _, data_root, cfg, *_ = _setup(tmp_path, folds=[0], include_blocked=True)

    rc = hsvm.run_holdout_svm(_args(tmp_path, data_root, smoke_test=True), cfg)

    assert rc == 1
    cdir = _only_run(tmp_path) / "datasets" / "TOY" / "no_dn"
    search = pd.read_csv(cdir / "cv_search_results.csv")
    assert set(search["status"]) == {"FAILED"}
    assert "prueba externa" in search.loc[0, "error"]
    assert not (cdir / "best_hyperparameters.json").exists()      # nada se selecciono


def test_resume_reuses_published_units_and_rejects_changed_data(tmp_path, monkeypatch):
    _, data_root, cfg, *_ = _setup(tmp_path)
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root), cfg) == 0
    run_id = _only_run(tmp_path).name

    calls = []
    original = hsvm._fit_and_score
    monkeypatch.setattr(hsvm, "_fit_and_score", lambda *a, **k: calls.append(1) or original(*a, **k))

    # Con la misma huella: no se repite ninguna unidad.
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root, resume=run_id), cfg) == 0
    assert calls == []

    # Con datos modificados: --resume se rechaza y la ejecucion queda intacta.
    np.save(data_root / "TOY" / "cv" / "fold_02" / "segments_no_dn.npy", np.zeros((28, 20000), dtype=np.float32))
    before = (_only_run(tmp_path) / "status.json").read_text(encoding="utf-8")
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root, resume=run_id), cfg) == 1
    assert (_only_run(tmp_path) / "status.json").read_text(encoding="utf-8") == before


def test_resume_is_rejected_when_the_code_changes_and_the_run_stays_intact(tmp_path, monkeypatch, capsys):
    _, data_root, cfg, *_ = _setup(tmp_path)
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root), cfg) == 0
    run_root = _only_run(tmp_path)
    run_id = run_root.name
    before = {p.name: p.read_bytes() for p in (run_root / "status.json", run_root / "run_fingerprint.json")}

    real = hcv.code_fingerprint()
    edited = {**real["code_files"], "holdout_svm.py": "0" * 64}
    monkeypatch.setattr(hcv, "code_fingerprint", lambda root=None: {"code_sha256": "f" * 64, "code_files": edited})

    calls = []
    original = hsvm._fit_and_score
    monkeypatch.setattr(hsvm, "_fit_and_score", lambda *a, **k: calls.append(1) or original(*a, **k))

    assert hsvm.run_holdout_svm(_args(tmp_path, data_root, resume=run_id), cfg) == 1
    err = capsys.readouterr().err
    assert "codigo de modeling" in err and "holdout_svm.py" in err
    assert calls == []                                                    # no se reutilizo ni se entreno nada
    assert {p.name: p.read_bytes() for p in (run_root / "status.json", run_root / "run_fingerprint.json")} == before


def test_resume_is_rejected_when_the_split_changes(tmp_path):
    split, data_root, cfg, csv_path, manifest_path = _setup(tmp_path)
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root), cfg) == 0
    run_id = _only_run(tmp_path).name

    # Se regenera el split (aunque siga siendo autoconsistente con su manifiesto).
    frame = pd.read_csv(csv_path, dtype={"patient_uid": str})
    frame.loc[0, "calibration_patient"] = not bool(frame.loc[0, "calibration_patient"])
    sp.write_holdout_split_csv(frame, csv_path, manifest_path, {
        "seed": 20260914, "n_splits": N_SPLITS, "counts": {"TOY": {"n_patients": len(frame)}},
    })
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root, resume=run_id), cfg) == 1


# ---------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------

def test_dry_run_passes_when_all_five_cv_folds_are_linked_to_the_current_split(tmp_path, capsys):
    _, data_root, cfg, *_ = _setup(tmp_path)
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root, dry_run=True), cfg) == 0
    out = capsys.readouterr().out
    assert "veredicto: OK" in out and "candidatos_svm" in out
    assert not (tmp_path / "runs").exists()                  # el dry-run no escribe nada


def test_dry_run_fails_when_a_fold_is_missing(tmp_path):
    _, data_root, cfg, *_ = _setup(tmp_path, folds=[0, 1, 2, 3])
    assert hsvm.run_holdout_svm(_args(tmp_path, data_root, dry_run=True), cfg) == 1


def test_dry_run_fails_when_the_split_changed_after_preprocessing(tmp_path):
    split, data_root, cfg, csv_path, manifest_path = _setup(tmp_path)
    frame = pd.read_csv(csv_path, dtype={"patient_uid": str})
    frame.loc[0, "calibration_patient"] = not bool(frame.loc[0, "calibration_patient"])
    sp.write_holdout_split_csv(frame, csv_path, manifest_path, {
        "seed": 20260914, "n_splits": N_SPLITS, "counts": {"TOY": {"n_patients": len(frame)}},
    })
    report = hcv.dry_run_check_holdout(data_root, cfg, [_spec()], csv_path, manifest_path)
    checks = report["checks"]
    assert not report["ok"]
    failed = set(checks.loc[~checks["ok"], "check"])
    assert any(name.startswith("holdout_splits_csv_sha256") for name in failed)


def test_dry_run_fails_when_a_fold_contains_an_outer_test_patient(tmp_path):
    _, data_root, cfg, csv_path, manifest_path = _setup(tmp_path, folds=[0, 1, 2, 3, 4], include_blocked=True)
    report = hcv.dry_run_check_holdout(data_root, cfg, [_spec()], csv_path, manifest_path)
    assert not report["ok"]
    failed = report["checks"].loc[~report["checks"]["ok"]]
    assert set(failed["check"]) == {f"pacientes_vs_split[cv/fold_{k:02d}]" for k in range(N_SPLITS)}
    assert failed["detail"].str.contains("prueba externa").all()


def test_dry_run_fails_when_a_fold_manifest_is_of_another_protocol(tmp_path):
    _, data_root, cfg, csv_path, manifest_path = _setup(tmp_path)
    fold_manifest = data_root / "TOY" / "cv" / "fold_03" / "manifest.json"
    payload = json.loads(fold_manifest.read_text(encoding="utf-8"))
    payload["protocol"] = "fold-aware-v2"
    fold_manifest.write_text(json.dumps(payload), encoding="utf-8")
    report = hcv.dry_run_check_holdout(data_root, cfg, [_spec()], csv_path, manifest_path)
    assert not report["ok"]
    assert "manifest_protocol[cv/fold_03]" in set(report["checks"].loc[~report["checks"]["ok"], "check"])


def _spec():
    return dmod.ConditionSpec(dataset="TOY", condition="no_dn", branch="no_dn", dn_reliable_only=False)


def test_input_hashes_cover_the_split_and_every_fold_but_only_fold_zero_in_smoke_mode(tmp_path):
    split, data_root, cfg, csv_path, manifest_path = _setup(tmp_path)
    full = hcv.holdout_input_hashes(data_root, [_spec()], {"TOY": split}, csv_path, manifest_path)
    assert {"holdout_splits.csv", "holdout_splits_manifest.json"} <= set(full)
    assert len([k for k in full if k.endswith("/segments.csv")]) == N_SPLITS
    smoke = hcv.holdout_input_hashes(data_root, [_spec()], {"TOY": split}, csv_path, manifest_path, smoke_test=True)
    assert len([k for k in smoke if k.endswith("/segments.csv")]) == 1
