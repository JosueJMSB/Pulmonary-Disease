"""Protocolo holdout_final_v1 de punta a punta: entrena con el 100% del
"desarrollo" sintetico (arquitectura/acustica REALES de
configs/holdout_final/cnn_v3.toml, como los smoke-test de holdout_cnn.py),
publica ``final_model.pt`` ANTES de evaluar, evalua la prueba externa UNA sola
vez, y ``--resume`` reutiliza las predicciones si ya se habian calculado. Se
omite si no hay PyTorch (igual que los demas tests de CNN)."""

import dataclasses
import json
import logging

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from .. import core
from .. import selection as sel
from ... import artifacts as art
from ... import data as dmod
from ... import splits as sp

V3_DIR = dmod.HOLDOUT_FINAL_CONFIGS_DIR
LOGGER = logging.getLogger("test_final_training_run")
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Datos y seleccion sinteticos
# ---------------------------------------------------------------------------

def _toy_segments(n_pos, n_neg, prefix):
    """``sp.build_patient_table`` agrega incondicionalmente sobre ``audio_id``
    (``n_recordings``) y ``segment_id`` (``n_segments``): sin esas columnas,
    ``groupby(...).agg(...)`` lanza un KeyError. Un valor unico por fila basta."""
    rows = []
    for label, n in ((1, n_pos), (0, n_neg)):
        for i in range(n):
            uid = f"{prefix}_{label}_{i:03d}"
            rows.append({
                "patient_uid": uid, "target_label": label, "calibration_patient": False, "dataset": prefix,
                "audio_id": f"{uid}_rec0", "segment_id": f"{uid}_rec0_000",
            })
    return pd.DataFrame(rows)


def _write_split(tmp_path, dataset):
    table = sp.assign_holdout_split(sp.build_patient_table(_toy_segments(12, 12, dataset)), n_splits=5, random_state=20260914)
    csv_path = tmp_path / "holdout_splits.csv"
    manifest_path = tmp_path / "holdout_splits_manifest.json"
    sp.write_holdout_split_csv(
        sp.holdout_split_to_frame(table, dataset), csv_path, manifest_path,
        {"seed": 20260914, "n_splits": 5, "counts": {dataset: {"n_patients": len(table)}}},
    )
    return sp.load_holdout_split(csv_path, manifest_path, dataset), csv_path, manifest_path


def _write_final_stage(
    data_root, dataset, split, csv_path, manifest_path, sources=None, selection_status="approved",
    selection_pipeline_fingerprint=None,
):
    """``<data_root>/<dataset>/final/``: segments.csv + .npy con role=train
    (TODO el desarrollo) / role=test (la prueba externa), forma (n, 20000)
    -el segment_length real de cnn_v3.toml-. El dispositivo se asigna POR
    CLASE (AKG solo COPD, Meditron solo Control) a proposito: cada dispositivo
    queda de una sola clase, para ejercitar el NaN de compute_metrics_by_device.
    ``sources`` (opcional) asigna source_dataset por paciente, para probar el
    desglose por fuente (COMBINED)."""
    final_dir = data_root / dataset / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    dev_patients = split.development_patients()
    test_patients = sorted(split.blocked_test_patients())
    label_by = dict(zip(split.patient_table["patient_uid"], split.patient_table["target_label"]))
    source_by = sources or {p: dataset for p in dev_patients + test_patients}
    entries = [(p, "train") for p in dev_patients] + [(p, "test") for p in test_patients]

    rows = []
    for i, (uid, role) in enumerate(entries):
        label = int(label_by[uid])
        rows.append({
            "task_array_index": i, "source_array_index": i, "segment_id": f"{uid}_S", "audio_id": f"{uid}_rec0",
            "patient_uid": uid, "diagnosis": "COPD" if label else "Normal", "target_label": label,
            "target_name": "COPD" if label else "Normal", "calibration_patient": False, "dn_reliable": True,
            "role": role, "fold_id": -1, "dataset": dataset, "source_dataset": source_by[uid],
            "device": "AKG" if label == 1 else "Meditron",
        })
    segments = pd.DataFrame(rows)
    segments.to_csv(final_dir / "segments.csv", index=False)

    rng = np.random.default_rng(0)
    offsets = np.where(segments["target_label"].to_numpy() == 1, 2.0, -2.0)
    array = (rng.standard_normal((len(segments), 20000)) * 0.05 + offsets[:, None]).astype(np.float32)
    np.save(final_dir / "segments_no_dn.npy", array)
    np.save(final_dir / "segments_dn.npy", array * 0.9)

    (final_dir / "preprocessing_params.json").write_text(json.dumps({"target_rms": 0.03}), encoding="utf-8")
    (final_dir / "manifest.json").write_text(json.dumps({
        "verdict": "PASS", "protocol": "holdout-v3", "stage": "final", "dataset_scope": dataset,
        "n_segments": len(segments),
        "counts_by_role": {"train": len(dev_patients), "test": len(test_patients)},
        "holdout_splits_csv_sha256": dmod.sha256_file(csv_path),
        "holdout_splits_manifest_sha256": dmod.sha256_file(manifest_path),
        "selection_status_at_generation": selection_status,
        "selection_pipeline_fingerprint": selection_pipeline_fingerprint,
        "output_hashes": {
            name: dmod.sha256_file(final_dir / name)
            for name in ("segments.csv", "segments_no_dn.npy", "segments_dn.npy", "preprocessing_params.json")
        },
    }), encoding="utf-8")
    return final_dir


def _write_best_hp(
    source_runs_root, *, dataset, architecture="cnn", run_id="toy_run", condition="dn", branch=None,
    hp=None, config_index=0, epochs=1, protocol="holdout_cv_v3",
):
    """Artefacto de procedencia real (``validate_source_artifact`` valida su
    contenido, no solo su hash). Devuelve ``(source_artifact_relativo, sha256)``."""
    branch = branch or condition
    hp = dict(hp or {"lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4, "batch_size": 4})
    best_dir = source_runs_root / architecture / run_id / "datasets" / dataset / condition
    best_dir.mkdir(parents=True, exist_ok=True)
    path = best_dir / "best_hyperparameters.json"
    path.write_text(json.dumps({
        "protocol": protocol, "model": architecture, "dataset": dataset, "condition": condition,
        "branch": branch, "config_index": config_index, "hyperparameters": hp, "median_best_epoch": epochs,
    }), encoding="utf-8")
    relative = f"{architecture}/{run_id}/datasets/{dataset}/{condition}/best_hyperparameters.json"
    return relative, dmod.sha256_file(path)


def _make_pipeline(dataset, source_runs_root, **overrides):
    condition = overrides.get("condition", "dn")
    branch = overrides.get("branch", "dn")
    architecture = overrides.get("architecture", "cnn")
    hp = overrides.get("hyperparameters", {"lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4, "batch_size": 4})
    epochs = overrides.get("epochs", 1)
    relative, artifact_sha = _write_best_hp(
        source_runs_root, dataset=dataset, architecture=architecture, condition=condition, branch=branch,
        hp=hp, epochs=epochs,
    )
    defaults = dict(
        dataset=dataset, architecture=architecture, condition=condition, branch=branch,
        reference_config=V3_DIR / "cnn_v3.toml", hyperparameters=hp, epochs=epochs,
        status="approved", source_run_id="toy_run", source_config_index=0,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
        robustness_run_id="", device_review_path="", device_review_sha256="",
    )
    defaults.update(overrides)
    return sel.SelectedPipeline(**defaults)


def _write_selection_toml(
    path, pipeline: sel.SelectedPipeline, *, threshold, seed_base,
    split_csv_sha256, split_manifest_sha256, bootstrap_n_resamples, bootstrap_confidence, bootstrap_random_state,
):
    """TOML REAL (no un ``"placeholder"``): ``sel.final_pipeline_fingerprint``
    necesita parsear ``[pipelines.<dataset>]`` de verdad para que el dry-run
    pueda comparar la huella del preprocesamiento final contra la seleccion
    actual (ver _write_final_stage/selection_pipeline_fingerprint)."""
    hp = pipeline.hyperparameters
    path.write_text(f"""
protocol = "holdout_final_v1"

[selection]
threshold = {threshold}
seed_base = {seed_base}

[split]
csv = "ignored-in-tests.csv"
manifest = "ignored-in-tests-manifest.json"
csv_sha256 = "{split_csv_sha256}"
manifest_sha256 = "{split_manifest_sha256}"

[bootstrap]
n_resamples = {bootstrap_n_resamples}
confidence = {bootstrap_confidence}
random_state = {bootstrap_random_state}

[pipelines.{pipeline.dataset}]
architecture = "{pipeline.architecture}"
condition = "{pipeline.condition}"
branch = "{pipeline.branch}"
reference_config = "modeling/configs/holdout_final/cnn_v3.toml"
epochs = {pipeline.epochs}
status = "{pipeline.status}"
source_run_id = "{pipeline.source_run_id}"
source_config_index = {pipeline.source_config_index}
source_artifact = "{pipeline.source_artifact}"
source_artifact_sha256 = "{pipeline.source_artifact_sha256}"
robustness_run_id = "{pipeline.robustness_run_id}"
device_review_path = "{pipeline.device_review_path}"
device_review_sha256 = "{pipeline.device_review_sha256}"

[pipelines.{pipeline.dataset}.hyperparameters]
lr = {hp['lr']}
dropout = {hp['dropout']}
weight_decay = {hp['weight_decay']}
batch_size = {hp['batch_size']}
""", encoding="utf-8")
    return path


def _setup(tmp_path, dataset="FRAIWAN_Extended", sources=None, status="approved", **pipeline_overrides):
    split, csv_path, manifest_path = _write_split(tmp_path, dataset)
    data_root = tmp_path / "holdout_calibrated"
    csv_sha, manifest_sha = dmod.sha256_file(csv_path), dmod.sha256_file(manifest_path)

    source_runs_root = tmp_path / "source_runs"
    pipeline = _make_pipeline(dataset, source_runs_root, status=status, **pipeline_overrides)

    selection_path = tmp_path / "selected_pipelines.toml"
    _write_selection_toml(
        selection_path, pipeline, threshold=0.5, seed_base=20260914,
        split_csv_sha256=csv_sha, split_manifest_sha256=manifest_sha,
        bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
    )
    pipeline_fp = sel.final_pipeline_fingerprint(selection_path, dataset)
    _write_final_stage(
        data_root, dataset, split, csv_path, manifest_path, sources=sources, selection_status=status,
        selection_pipeline_fingerprint=pipeline_fp,
    )

    selection_cfg = sel.SelectionConfig(
        path=selection_path, threshold=0.5, seed_base=20260914,
        split_csv=csv_path, split_manifest=manifest_path,
        split_csv_sha256=csv_sha, split_manifest_sha256=manifest_sha,
        bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
        pipelines={dataset: pipeline},
    )
    return split, data_root, selection_cfg, source_runs_root


def _run(run_root, dataset, selection_cfg, data_root, source_runs_root, **overrides):
    kwargs = dict(
        run_root=run_root, dataset=dataset, selection_cfg=selection_cfg, data_root=data_root,
        cache_root=run_root.parent / "cache", source_runs_root=source_runs_root, device=DEVICE,
        num_workers=0, force_features=False, dry_run=False, logger=LOGGER,
    )
    kwargs.update(overrides)
    return core.run_pipeline_for_dataset(**kwargs)


# ---------------------------------------------------------------------------
# --dry-run: no toca senales ni etiquetas del test
# ---------------------------------------------------------------------------

def test_dry_run_check_approves_without_training(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path)
    report = core.dry_run_check(selection_cfg, ["FRAIWAN_Extended"], data_root, source_runs_root)
    assert report["ok"], report["checks"].to_string()
    assert not (tmp_path / "runs").exists()


def test_dry_run_check_detects_a_selection_drifted_after_the_final_stage_was_generated(tmp_path):
    """CORRECIONES.md seccion 5: el dry-run compara el fingerprint del
    preprocesamiento final con el pipeline seleccionado ACTUALMENTE. Si la
    seleccion cambio despues de generar esa etapa, debe reportarlo -nunca
    asumir en silencio que sigue vigente-."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path)
    # El PIPELINE cambia (otra epoca congelada) DESPUES de que se genero la
    # etapa final: la huella de [pipelines.<dataset>] ya no coincide.
    changed_pipeline = dataclasses.replace(selection_cfg.pipelines["FRAIWAN_Extended"], epochs=99)
    _write_selection_toml(
        selection_cfg.path, changed_pipeline, threshold=0.5, seed_base=20260914,
        split_csv_sha256=selection_cfg.split_csv_sha256, split_manifest_sha256=selection_cfg.split_manifest_sha256,
        bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
    )
    report = core.dry_run_check(selection_cfg, ["FRAIWAN_Extended"], data_root, source_runs_root)
    checks = report["checks"]
    row = checks.loc[checks["check"] == "selection_pipeline_fingerprint"].iloc[0]
    assert not row["ok"]


def test_dry_run_check_reports_blocked_selection_without_raising(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(
        tmp_path, dataset="ICBHI", status="pending_device_review",
    )
    report = core.dry_run_check(selection_cfg, ["ICBHI"], data_root, source_runs_root)
    assert not report["ok"]
    failing = report["checks"]
    assert not failing.loc[failing["check"] == "seleccion_aprobada", "ok"].iloc[0]


def test_run_pipeline_for_dataset_with_dry_run_true_does_not_create_run_root(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path)
    run_root = tmp_path / "runs" / "x"
    result = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root, dry_run=True)
    assert result["status"] == "DRY_RUN_OK"
    assert result["detail"] == "No se cargaron senales, etiquetas ni predicciones del test."
    assert not run_root.exists()


def test_dry_run_never_touches_segments_npy_or_logmel(tmp_path, monkeypatch):
    """CORRECIONES.md seccion 3: dry_run_check() no debe llamar a
    load_task_segments(), np.load() ni extraer Log-Mel. Se trampea cada una
    para que CUALQUIER llamada falle la prueba, y se confirma que el dry-run
    de todas formas aprueba (usa solo manifest.json + sha256 de archivo)."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path)

    def _trap(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"dry_run_check no debe llamar a {name}")
        return _raise

    monkeypatch.setattr(dmod, "load_task_segments", _trap("load_task_segments"))
    monkeypatch.setattr(dmod, "extract_or_load_logmel", _trap("extract_or_load_logmel"))
    monkeypatch.setattr(np, "load", _trap("np.load"))

    report = core.dry_run_check(selection_cfg, ["FRAIWAN_Extended"], data_root, source_runs_root)
    assert report["ok"], report["checks"].to_string()


# ---------------------------------------------------------------------------
# Seleccion pendiente: BLOCKED_SELECTION, sin tocar nada
# ---------------------------------------------------------------------------

def test_pending_device_review_blocks_without_touching_the_test(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(
        tmp_path, dataset="ICBHI", status="pending_device_review",
    )
    run_root = tmp_path / "runs" / "x"
    result = _run(run_root, "ICBHI", selection_cfg, data_root, source_runs_root)
    assert result["status"] == core.STATUS_BLOCKED_SELECTION
    assert not run_root.exists()


# ---------------------------------------------------------------------------
# Entrenamiento definitivo + evaluacion externa unica
# ---------------------------------------------------------------------------

def _prepare_run_root(tmp_path):
    run_root = tmp_path / "runs" / "20260101T000000Z_abcdef"
    run_root.mkdir(parents=True)
    (run_root / "datasets").mkdir()
    return run_root


def test_full_run_trains_with_all_development_patients_and_completes(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)

    result = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    assert result["status"] == core.STATUS_COMPLETED
    pdir = core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn")
    assert (pdir / "_SUCCESS").is_file()
    assert core.is_pipeline_complete(run_root, "FRAIWAN_Extended", "dn")

    for name in (
        "final_model.pt", "model_manifest.json", "selection_snapshot.json", "normalization.json",
        "training_history.csv", "test_access_log.json",
        "test_segment_predictions.csv", "test_recording_predictions.csv", "test_patient_predictions.csv",
        "test_metrics.json", "test_metrics_summary.csv", "test_bootstrap_ci.csv",
        "test_classification_report.csv", "test_confusion_matrix.csv",
        "test_metrics_by_device.csv", "test_confusion_matrix_by_device.csv",
    ):
        assert (pdir / name).is_file(), name
    for figure in ("confusion_matrix", "roc", "pr"):
        assert (pdir / "figures" / f"{figure}.png").is_file(), figure
    # plot_roc_curve/plot_pr_curve escriben su propio CSV (fpr/tpr y precision/recall
    # por umbral) junto a la figura: no hace falta un "roc_curve.csv"/"pr_curve.csv" aparte.
    for curve in ("roc", "pr"):
        assert (pdir / "figures" / f"{curve}.csv").is_file(), curve
    assert not (pdir / "test_metrics_by_source.csv").is_file()  # una sola fuente: no aplica

    manifest = json.loads((pdir / "model_manifest.json").read_text(encoding="utf-8"))
    assert manifest["epochs"] == 1
    assert manifest["n_train_patients"] == len(split.development_patients())
    assert manifest["n_test_patients"] == len(split.blocked_test_patients())
    assert any("FRAIWAN_Extended" in w for w in manifest["warnings"])  # advertencia de pocos positivos

    patients = pd.read_csv(pdir / "test_patient_predictions.csv", dtype={"patient_uid": str})
    assert set(patients["patient_uid"]) == split.blocked_test_patients()
    assert not (set(patients["patient_uid"]) & set(split.development_patients()))

    history = pd.read_csv(pdir / "training_history.csv")
    assert len(history) == 1  # exactamente pipeline.epochs; sin validation
    assert {"epoch", "train_loss", "weight_decay", "batch_size"} <= set(history.columns)

    by_device = pd.read_csv(pdir / "test_metrics_by_device.csv")
    # AKG = solo COPD, Meditron = solo Control (ver _write_final_stage): ambos de una sola clase,
    # asi que balanced_accuracy/macro_f1/auroc/auprc quedan en NaN, pero el recall de la unica
    # clase presente SI esta definido (ev.compute_metrics_by_device).
    assert by_device["balanced_accuracy"].isna().all()
    akg_recall_copd = by_device.set_index("device").loc["AKG", "recall_copd"]
    assert 0.0 <= akg_recall_copd <= 1.0


def test_threshold_and_epochs_come_from_the_selection_not_the_reference_toml(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=2)
    run_root = _prepare_run_root(tmp_path)
    _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    history = pd.read_csv(core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn") / "training_history.csv")
    assert len(history) == 2  # cnn_v3.toml pide 150 epocas de busqueda; aqui solo cuentan las del pipeline


def test_combined_like_dataset_writes_metrics_by_source(tmp_path):
    split, csv_path, manifest_path = _write_split(tmp_path, "FRAIWAN_Extended")
    data_root = tmp_path / "holdout_calibrated"
    csv_sha, manifest_sha = dmod.sha256_file(csv_path), dmod.sha256_file(manifest_path)
    half = len(split.development_patients()) // 2
    sources = {p: ("ICBHI" if i < half else "FRAIWAN_Extended") for i, p in enumerate(split.development_patients())}
    sources.update({p: ("ICBHI" if i % 2 == 0 else "FRAIWAN_Extended") for i, p in enumerate(sorted(split.blocked_test_patients()))})

    source_runs_root = tmp_path / "source_runs"
    pipeline = _make_pipeline("FRAIWAN_Extended", source_runs_root, epochs=1)
    selection_path = tmp_path / "selected_pipelines.toml"
    _write_selection_toml(
        selection_path, pipeline, threshold=0.5, seed_base=20260914,
        split_csv_sha256=csv_sha, split_manifest_sha256=manifest_sha,
        bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
    )
    pipeline_fp = sel.final_pipeline_fingerprint(selection_path, "FRAIWAN_Extended")
    _write_final_stage(
        data_root, "FRAIWAN_Extended", split, csv_path, manifest_path, sources=sources,
        selection_pipeline_fingerprint=pipeline_fp,
    )
    selection_cfg = sel.SelectionConfig(
        path=selection_path, threshold=0.5, seed_base=20260914, split_csv=csv_path, split_manifest=manifest_path,
        split_csv_sha256=csv_sha, split_manifest_sha256=manifest_sha,
        bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
        pipelines={"FRAIWAN_Extended": pipeline},
    )
    run_root = _prepare_run_root(tmp_path)

    _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    pdir = core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn")
    assert (pdir / "test_metrics_by_source.csv").is_file()
    by_source = pd.read_csv(pdir / "test_metrics_by_source.csv")
    assert set(by_source["source_dataset"]) == {"ICBHI", "FRAIWAN_Extended"}


# ---------------------------------------------------------------------------
# Un run COMPLETED no vuelve a abrir el test
# ---------------------------------------------------------------------------

def test_completed_pipeline_is_not_retrained_or_reevaluated(tmp_path, monkeypatch):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED

    def _never_call(*args, **kwargs):
        raise AssertionError("no deberia reentrenar: el pipeline ya esta COMPLETED")

    monkeypatch.setattr(core, "train_pipeline", _never_call)
    second = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert second["status"] == core.STATUS_COMPLETED
    assert second.get("detail") == "ya completado"


# ---------------------------------------------------------------------------
# --resume reutiliza predicciones ya calculadas, sin reentrenar
# ---------------------------------------------------------------------------

def _revert_publish_to_staging(run_root, dataset, condition):
    pdir = core.pipeline_dir(run_root, dataset, condition)
    staging = core.pipeline_staging_dir(run_root, dataset, condition)
    (pdir / "_SUCCESS").unlink()
    pdir.rename(staging)
    return staging


def test_resume_reuses_existing_predictions_without_retraining(tmp_path, monkeypatch):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED

    staging = _revert_publish_to_staging(run_root, "FRAIWAN_Extended", "dn")
    assert (staging / "test_segment_predictions.csv").is_file()

    calls = []
    original = core.train_pipeline

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "train_pipeline", spy)
    second = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    assert second["status"] == core.STATUS_COMPLETED
    assert calls == []  # no se volvio a entrenar: se reutilizaron las predicciones
    assert core.is_pipeline_complete(run_root, "FRAIWAN_Extended", "dn")


def test_resume_from_published_model_without_predictions_does_not_retrain(tmp_path, monkeypatch):
    """Si una ejecucion anterior publico ``final_model.pt`` pero se interrumpio
    antes de escribir predicciones, --resume debe reanudar la evaluacion
    cargando EXACTAMENTE ese checkpoint (``cnn_model.load_checkpoint``), sin
    llamar a ``train_pipeline`` (CORRECIONES.md seccion 4)."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED

    staging = _revert_publish_to_staging(run_root, "FRAIWAN_Extended", "dn")
    model_bytes_before = (staging / "final_model.pt").read_bytes()
    # Simular una interrupcion ANTES de escribir predicciones: se borran (el
    # modelo, normalization.json y training_history.csv, que son del
    # entrenamiento, se conservan).
    for name in ("test_segment_predictions.csv", "test_recording_predictions.csv",
                 "test_patient_predictions.csv", "model_manifest.json"):
        (staging / name).unlink(missing_ok=True)

    calls = []
    original = core.train_pipeline

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "train_pipeline", spy)
    second = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    assert second["status"] == core.STATUS_COMPLETED
    assert calls == []  # nunca se reentrena: se carga el checkpoint ya publicado
    published = core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn")
    assert (published / "final_model.pt").read_bytes() == model_bytes_before
    assert core.is_pipeline_complete(run_root, "FRAIWAN_Extended", "dn")


# ---------------------------------------------------------------------------
# Registro persistente de acceso al test (fuera del staging)
# ---------------------------------------------------------------------------

def test_test_access_registry_records_status_progression_to_completed(tmp_path):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    result = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert result["status"] == core.STATUS_COMPLETED

    record = core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")
    assert record is not None and record["status"] == core.TEST_ACCESS_COMPLETED
    model_path = core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn") / "final_model.pt"
    assert record["model_sha256"] == dmod.sha256_file(model_path)
    assert record["split_csv_sha256"] == selection_cfg.split_csv_sha256


def test_test_access_registry_never_overwrites_a_different_fingerprint_or_model(tmp_path):
    run_root = tmp_path / "runs" / "x"
    core.write_test_access_record(
        run_root, "FRAIWAN_Extended", "dn", pipeline_fingerprint="fp-a", model_sha256="model-a",
        status=core.TEST_ACCESS_STARTED, split_csv_sha256="x", split_manifest_sha256="y",
    )
    with pytest.raises(core.FinalTrainingError, match="otra huella"):
        core.write_test_access_record(
            run_root, "FRAIWAN_Extended", "dn", pipeline_fingerprint="fp-b", model_sha256="model-a",
            status=core.TEST_ACCESS_STARTED, split_csv_sha256="x", split_manifest_sha256="y",
        )
    with pytest.raises(core.FinalTrainingError, match="otra huella"):
        core.write_test_access_record(
            run_root, "FRAIWAN_Extended", "dn", pipeline_fingerprint="fp-a", model_sha256="model-b",
            status=core.TEST_ACCESS_STARTED, split_csv_sha256="x", split_manifest_sha256="y",
        )
    # Mismos valores: SI se permite avanzar el estado.
    core.write_test_access_record(
        run_root, "FRAIWAN_Extended", "dn", pipeline_fingerprint="fp-a", model_sha256="model-a",
        status=core.TEST_ACCESS_PREDICTIONS_WRITTEN, split_csv_sha256="x", split_manifest_sha256="y",
    )
    assert core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")["status"] == core.TEST_ACCESS_PREDICTIONS_WRITTEN


def test_orphaned_completed_registry_without_publication_is_rejected(tmp_path):
    """Inconsistencia grave: el registro externo dice COMPLETED pero no existe
    la publicacion correspondiente (p. ej. se borro 'datasets/' a mano).
    Nunca se reintenta solo: exige revision manual."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    core.write_test_access_record(
        run_root, "FRAIWAN_Extended", "dn", pipeline_fingerprint="fp", model_sha256="model",
        status=core.TEST_ACCESS_COMPLETED, split_csv_sha256="x", split_manifest_sha256="y",
    )
    with pytest.raises(core.FinalTrainingError, match="inconsistencia grave"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)


def test_resume_rejects_a_changed_codebase_and_leaves_staging_untouched(tmp_path, monkeypatch):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED
    staging = _revert_publish_to_staging(run_root, "FRAIWAN_Extended", "dn")
    before = (staging / "model_manifest.json").read_text(encoding="utf-8")

    monkeypatch.setattr(core, "code_fingerprint", lambda: {"code_sha256": "changed", "code_files": {}})
    with pytest.raises(core.FinalTrainingError, match="codigo de modeling"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    # Rechazado ANTES de tocar nada: el staging queda identico al de antes del intento de --resume.
    assert (staging / "model_manifest.json").read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Una interrupcion/fallo no publica artefactos parciales
# ---------------------------------------------------------------------------

def test_a_failure_during_training_does_not_publish_anything(tmp_path, monkeypatch):
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fallo simulado de entrenamiento")

    monkeypatch.setattr(core, "train_pipeline", boom)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    pdir = core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn")
    staging = core.pipeline_staging_dir(run_root, "FRAIWAN_Extended", "dn")
    assert not pdir.exists()
    assert not (staging / "test_segment_predictions.csv").is_file()
    assert not (staging / "final_model.pt").is_file()
    status = art.read_status(staging)
    assert status["status"] == core.STATUS_FAILED


def test_interruption_after_checkpoint_before_inference_preserves_the_model(tmp_path, monkeypatch):
    """Punto 2 de CORRECIONES.md seccion 4: el checkpoint ya se publico (con su
    normalizacion e historial), pero algo falla durante la evaluacion. El
    modelo no se pierde y nada se publica."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fallo simulado durante la evaluacion")

    monkeypatch.setattr(core, "evaluate_pipeline", boom)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    staging = core.pipeline_staging_dir(run_root, "FRAIWAN_Extended", "dn")
    assert (staging / "final_model.pt").is_file()
    assert (staging / "normalization.json").is_file() and (staging / "training_history.csv").is_file()
    assert core.is_model_ready(staging)
    assert not (staging / "test_segment_predictions.csv").is_file()
    assert not core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn").exists()
    record = core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")
    assert record is not None and record["status"] == core.TEST_ACCESS_STARTED  # se registro el acceso, no se completo


def test_interruption_after_predictions_before_reports_allows_a_clean_resume(tmp_path, monkeypatch):
    """Punto 4: la inferencia termino y las predicciones ya se escribieron,
    pero write_reports falla. Nada se publica, pero --resume reutiliza esas
    predicciones (igual que test_resume_reuses_existing_predictions...)."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fallo simulado de reportes")

    monkeypatch.setattr(core, "write_reports", boom)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    staging = core.pipeline_staging_dir(run_root, "FRAIWAN_Extended", "dn")
    assert (staging / "test_segment_predictions.csv").is_file()
    record = core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")
    assert record["status"] == core.TEST_ACCESS_PREDICTIONS_WRITTEN
    assert not core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn").exists()

    monkeypatch.undo()
    second = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert second["status"] == core.STATUS_COMPLETED


def test_resume_after_predictions_written_does_not_retrain_or_reinfer_and_completes_all_artifacts(tmp_path, monkeypatch):
    """Interrupcion justo despues de escribir las predicciones -con el orden
    corregido, model_manifest.json ya quedo escrito y el registro externo ya
    avanzo a PREDICTIONS_WRITTEN antes de esta interrupcion- pero antes de
    completar los demas artefactos (reportes, status COMPLETED, publicacion).
    --resume debe reutilizar esas predicciones sin reentrenar NI repetir la
    inferencia (Caso A: ``_reload_evaluation_from_predictions``, nunca
    ``evaluate_pipeline``), y terminar con TODOS los archivos requeridos
    publicados."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fallo simulado despues de las predicciones")

    monkeypatch.setattr(core, "write_reports", boom)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    staging = core.pipeline_staging_dir(run_root, "FRAIWAN_Extended", "dn")
    assert (staging / "test_segment_predictions.csv").is_file()
    assert (staging / "test_recording_predictions.csv").is_file()
    assert (staging / "test_patient_predictions.csv").is_file()
    assert (staging / "model_manifest.json").is_file()  # ya escrito ANTES del registro PREDICTIONS_WRITTEN
    record = core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")
    assert record["status"] == core.TEST_ACCESS_PREDICTIONS_WRITTEN
    assert not core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn").exists()

    monkeypatch.undo()
    train_calls, eval_calls = [], []
    original_train, original_eval = core.train_pipeline, core.evaluate_pipeline

    def train_spy(*args, **kwargs):
        train_calls.append(1)
        return original_train(*args, **kwargs)

    def eval_spy(*args, **kwargs):
        eval_calls.append(1)
        return original_eval(*args, **kwargs)

    monkeypatch.setattr(core, "train_pipeline", train_spy)
    monkeypatch.setattr(core, "evaluate_pipeline", eval_spy)
    second = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    assert second["status"] == core.STATUS_COMPLETED
    assert train_calls == []  # nunca se reentrena
    assert eval_calls == []   # nunca se repite la inferencia: se reutilizan las predicciones ya escritas

    assert core.is_pipeline_complete(run_root, "FRAIWAN_Extended", "dn")
    published = core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn")
    for name in core.REQUIRED_PUBLISHED_FILES:
        assert (published / name).is_file(), name
    assert core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")["status"] == core.TEST_ACCESS_COMPLETED


def test_interruption_during_publish_allows_a_safe_retry_without_retraining(tmp_path, monkeypatch):
    """Punto 5: write_reports y status.json ya marcaron COMPLETED, pero
    ``publish_pipeline`` falla antes del ``os.replace`` final. El registro
    externo -actualizado a COMPLETED solo DESPUES de publicar- se queda en
    PREDICTIONS_WRITTEN: nunca miente sobre una publicacion que no ocurrio.
    Un reintento reutiliza las predicciones ya calculadas (Caso A) y completa
    la publicacion sin reentrenar ni reevaluar."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)

    def boom(*args, **kwargs):
        raise RuntimeError("fallo simulado de publicacion")

    monkeypatch.setattr(core, "publish_pipeline", boom)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    assert not core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn").exists()
    staging = core.pipeline_staging_dir(run_root, "FRAIWAN_Extended", "dn")
    assert staging.exists() and (staging / "test_segment_predictions.csv").is_file()
    record = core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")
    assert record["status"] == core.TEST_ACCESS_PREDICTIONS_WRITTEN  # nunca llego a COMPLETED: no se publico

    monkeypatch.undo()
    calls = []
    original = core.train_pipeline

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(core, "train_pipeline", spy)
    second = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)

    assert second["status"] == core.STATUS_COMPLETED
    assert calls == []  # el reintento reutilizo las predicciones; nunca reentreno
    assert core.read_test_access_record(run_root, "FRAIWAN_Extended", "dn")["status"] == core.TEST_ACCESS_COMPLETED


def test_predictions_without_model_ready_is_rejected(tmp_path):
    """Si existen predicciones pero falta el modelo listo (_MODEL_READY),
    nunca se reentrena para 'resolverlo': se exige revision manual."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED

    staging = _revert_publish_to_staging(run_root, "FRAIWAN_Extended", "dn")
    (staging / core.MODEL_READY_MARKER).unlink()

    with pytest.raises(core.FinalTrainingError, match="falta el modelo listo"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)


def test_registry_started_without_a_staging_directory_is_rejected(tmp_path):
    """Si el registro externo indica STARTED/PREDICTIONS_WRITTEN y desaparecio
    el staging correspondiente, se perdio la evidencia de ese acceso: se
    detiene en vez de proceder como si nada hubiera pasado."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    core.write_test_access_record(
        run_root, "FRAIWAN_Extended", "dn", pipeline_fingerprint="fp", model_sha256="model",
        status=core.TEST_ACCESS_STARTED, split_csv_sha256="x", split_manifest_sha256="y",
    )
    with pytest.raises(core.FinalTrainingError, match="se perdio"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)


def test_completed_pipeline_rejects_reuse_if_the_selection_drifted(tmp_path):
    """Antes de reutilizar un directorio con _SUCCESS, su fingerprint.json
    debe seguir coincidiendo con la seleccion ACTUAL."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED

    drifted_cfg = dataclasses.replace(selection_cfg, threshold=0.6)
    with pytest.raises(core.FinalTrainingError, match="congelada de este dataset"):
        _run(run_root, "FRAIWAN_Extended", drifted_cfg, data_root, source_runs_root)


def test_completed_pipeline_rejects_reuse_if_a_required_file_is_missing(tmp_path):
    """Un ``_SUCCESS`` no basta: deben seguir presentes modelo, normalizacion,
    historial, manifiesto, predicciones y metricas."""
    split, data_root, selection_cfg, source_runs_root = _setup(tmp_path, epochs=1)
    run_root = _prepare_run_root(tmp_path)
    first = _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
    assert first["status"] == core.STATUS_COMPLETED

    (core.pipeline_dir(run_root, "FRAIWAN_Extended", "dn") / "test_bootstrap_ci.csv").unlink()
    with pytest.raises(core.FinalTrainingError, match="faltan archivos requeridos"):
        _run(run_root, "FRAIWAN_Extended", selection_cfg, data_root, source_runs_root)
