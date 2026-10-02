"""Carga y validacion de ``selected_pipelines.toml`` (``selection.py``): forma
del TOML (incluidos los rangos de ``CORRECIONES.md`` seccion 2), estado de
aprobacion, split sin cambios, artefacto de procedencia (ruta EXACTA + hash +
contenido), revision por dispositivo para ICBHI, y la huella canonica POR
DATASET (que cambiar la aprobacion de uno no debe alterar la de otro). No
necesita PyTorch (no entrena nada)."""

import dataclasses
import json

import pandas as pd
import pytest

from .. import selection as sel
from ... import data as dmod
from ... import splits as sp

V3_DIR = dmod.HOLDOUT_FINAL_CONFIGS_DIR
DEFAULT_HP = {"lr": 0.001, "dropout": 0.2, "weight_decay": 0.0001, "batch_size": 4}


def _toy_segments(n_pos=10, n_neg=10, prefix="TOY"):
    """``sp.build_patient_table`` agrega incondicionalmente sobre ``audio_id``
    (``n_recordings``) y ``segment_id`` (``n_segments``), no solo sobre
    ``dataset``: sin esas dos columnas, ``groupby(...).agg(...)`` lanza un
    KeyError. Un ``audio_id``/``segment_id`` unico por fila basta aqui (cada
    paciente de prueba tiene una sola grabacion con un solo segmento)."""
    rows = []
    for label, n in ((1, n_pos), (0, n_neg)):
        for i in range(n):
            uid = f"{prefix}_{label}_{i:03d}"
            rows.append({
                "patient_uid": uid, "target_label": label, "calibration_patient": False, "dataset": prefix,
                "audio_id": f"{uid}_rec0", "segment_id": f"{uid}_rec0_000",
            })
    return pd.DataFrame(rows)


def _write_split(tmp_path, dataset="TOY"):
    table = sp.assign_holdout_split(sp.build_patient_table(_toy_segments(prefix=dataset)), n_splits=5, random_state=20260914)
    csv_path = tmp_path / "holdout_splits.csv"
    manifest_path = tmp_path / "holdout_splits_manifest.json"
    sp.write_holdout_split_csv(
        sp.holdout_split_to_frame(table, dataset), csv_path, manifest_path,
        {"seed": 20260914, "n_splits": 5, "counts": {dataset: {"n_patients": len(table)}}},
    )
    return csv_path, manifest_path


def _write_best_hp(
    source_runs_root, *, dataset, architecture="cnn", run_id="toy_run", condition="dn", branch=None,
    hp=None, config_index=0, epochs=1, protocol="holdout_cv_v3",
):
    """Artefacto de procedencia real en disco (``best_hyperparameters.json``),
    bajo ``source_runs_root``. Devuelve ``(source_artifact_relativo, sha256)``,
    listos para ``[pipelines.<dataset>] source_artifact``/``source_artifact_sha256``."""
    branch = branch or condition
    hp = dict(hp or DEFAULT_HP)
    best_dir = source_runs_root / architecture / run_id / "datasets" / dataset / condition
    best_dir.mkdir(parents=True, exist_ok=True)
    path = best_dir / "best_hyperparameters.json"
    path.write_text(json.dumps({
        "protocol": protocol, "model": architecture, "dataset": dataset, "condition": condition,
        "branch": branch, "config_index": config_index, "hyperparameters": hp, "median_best_epoch": epochs,
    }), encoding="utf-8")
    relative = f"{architecture}/{run_id}/datasets/{dataset}/{condition}/best_hyperparameters.json"
    return relative, dmod.sha256_file(path)


def _write_toml(path, *, csv_path, manifest_path, status="approved", dataset="TOY",
                 source_artifact="toy/best_hyperparameters.json", source_artifact_sha256="0" * 64,
                 threshold=0.5, epochs=1, architecture="cnn", condition="dn", batch_size=4,
                 device_review_path="", device_review_sha256=""):
    # [split] csv/manifest: rutas ABSOLUTAS (en POSIX, que Windows tambien acepta). load_selection_config
    # las resuelve como ``dmod.REPO_ROOT / valor``; con un valor ya absoluto, pathlib descarta REPO_ROOT y
    # deja la ruta absoluta tal cual -asi el TOML de prueba puede apuntar a tmp_path sin tocar el repo real-.
    path.write_text(f"""
protocol = "holdout_final_v1"

[selection]
threshold = {threshold}
seed_base = 20260914

[split]
csv = "{csv_path.as_posix()}"
manifest = "{manifest_path.as_posix()}"
csv_sha256 = "{dmod.sha256_file(csv_path)}"
manifest_sha256 = "{dmod.sha256_file(manifest_path)}"

[bootstrap]
n_resamples = 20
confidence = 0.95
random_state = 0

[pipelines.{dataset}]
architecture = "{architecture}"
condition = "{condition}"
branch = "{condition}"
reference_config = "modeling/configs/holdout_final/cnn_v3.toml"
epochs = {epochs}
status = "{status}"
source_run_id = "toy_run"
source_config_index = 0
source_artifact = "{source_artifact}"
source_artifact_sha256 = "{source_artifact_sha256}"
robustness_run_id = ""
device_review_path = "{device_review_path}"
device_review_sha256 = "{device_review_sha256}"

[pipelines.{dataset}.hyperparameters]
lr = 0.001
dropout = 0.2
weight_decay = 0.0001
batch_size = {batch_size}
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Forma del TOML, incluidos los rangos (CORRECIONES.md seccion 2)
# ---------------------------------------------------------------------------

def test_load_selection_config_reads_every_field(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "selected_pipelines.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path)
    cfg = sel.load_selection_config(path)
    assert cfg.threshold == 0.5 and cfg.seed_base == 20260914
    assert cfg.bootstrap_n_resamples == 20 and cfg.bootstrap_confidence == 0.95
    pipeline = cfg.pipeline_for("TOY")
    assert pipeline.condition == "dn" and pipeline.branch == "dn" and pipeline.epochs == 1
    assert pipeline.hyperparameters == DEFAULT_HP
    assert pipeline.source_artifact_sha256 == "0" * 64
    assert pipeline.approved


def test_load_selection_config_rejects_wrong_protocol(tmp_path):
    path = tmp_path / "selected_pipelines.toml"
    path.write_text('protocol = "holdout_cv_v3"\n', encoding="utf-8")
    with pytest.raises(sel.SelectionError, match="protocol"):
        sel.load_selection_config(path)


def test_pipeline_for_unknown_dataset_raises(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "selected_pipelines.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path)
    cfg = sel.load_selection_config(path)
    with pytest.raises(sel.SelectionError, match="COMBINED"):
        cfg.pipeline_for("COMBINED")


def test_unknown_status_is_rejected_at_load_time(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "selected_pipelines.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path, status="maybe")
    with pytest.raises(sel.SelectionError, match="status"):
        sel.load_selection_config(path)


@pytest.mark.parametrize("threshold", (0.0, 1.0, 1.5, -0.1))
def test_threshold_out_of_range_is_rejected(tmp_path, threshold):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "selected_pipelines.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path, threshold=threshold)
    with pytest.raises(sel.SelectionError, match="threshold"):
        sel.load_selection_config(path)


@pytest.mark.parametrize("epochs", (0, -1))
def test_non_positive_epochs_is_rejected(tmp_path, epochs):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "selected_pipelines.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path, epochs=epochs)
    with pytest.raises(sel.SelectionError, match="epochs"):
        sel.load_selection_config(path)


def test_non_positive_batch_size_is_rejected(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "selected_pipelines.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path, batch_size=0)
    with pytest.raises(sel.SelectionError, match="batch_size"):
        sel.load_selection_config(path)


def test_unknown_architecture_or_condition_is_rejected(tmp_path):
    csv_path, manifest_path = _write_split(tmp_path)
    path = tmp_path / "a.toml"
    _write_toml(path, csv_path=csv_path, manifest_path=manifest_path, architecture="resnet")
    with pytest.raises(sel.SelectionError, match="architecture"):
        sel.load_selection_config(path)

    path2 = tmp_path / "b.toml"
    _write_toml(path2, csv_path=csv_path, manifest_path=manifest_path, condition="weird")
    with pytest.raises(sel.SelectionError, match="condition"):
        sel.load_selection_config(path2)


# ---------------------------------------------------------------------------
# Validaciones de acceso al test
# ---------------------------------------------------------------------------

def _cfg_with_pipeline(tmp_path, *, source_runs_root=None, write_source_artifact=True, **pipeline_overrides):
    csv_path, manifest_path = _write_split(tmp_path)
    dataset = pipeline_overrides.get("dataset", "TOY")
    source_runs_root = source_runs_root or (tmp_path / "source_runs")
    if write_source_artifact:
        relative, artifact_sha = _write_best_hp(
            source_runs_root, dataset=dataset,
            architecture=pipeline_overrides.get("architecture", "cnn"),
            condition=pipeline_overrides.get("condition", "dn"),
            branch=pipeline_overrides.get("branch", "dn"),
        )
    else:
        relative, artifact_sha = "missing/best_hyperparameters.json", "0" * 64
    defaults = dict(
        dataset=dataset, architecture="cnn", condition="dn", branch="dn",
        reference_config=V3_DIR / "cnn_v3.toml",
        hyperparameters=dict(DEFAULT_HP),
        epochs=1, status="approved", source_run_id="toy_run", source_config_index=0,
        source_artifact=relative, source_artifact_sha256=artifact_sha,
        robustness_run_id="", device_review_path="", device_review_sha256="",
    )
    defaults.update(pipeline_overrides)
    pipeline = sel.SelectedPipeline(**defaults)
    selection_path = tmp_path / "selected_pipelines.toml"
    if not selection_path.is_file():
        selection_path.write_text("placeholder", encoding="utf-8")
    cfg = sel.SelectionConfig(
        path=selection_path, threshold=0.5, seed_base=20260914,
        split_csv=csv_path, split_manifest=manifest_path,
        split_csv_sha256=dmod.sha256_file(csv_path), split_manifest_sha256=dmod.sha256_file(manifest_path),
        bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
        pipelines={dataset: pipeline},
    )
    return cfg, pipeline


def test_validate_pipeline_approved_rejects_pending_or_blocked(tmp_path):
    for status in (sel.STATUS_PENDING_DEVICE_REVIEW, sel.STATUS_BLOCKED):
        _, pipeline = _cfg_with_pipeline(tmp_path, status=status)
        with pytest.raises(sel.SelectionError, match=status):
            sel.validate_pipeline_approved(pipeline)


def test_validate_split_unchanged_detects_a_modified_csv(tmp_path):
    cfg, _ = _cfg_with_pipeline(tmp_path)
    sel.validate_split_unchanged(cfg)  # no lanza: coincide

    cfg.split_csv.write_text(cfg.split_csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(sel.SelectionError, match="holdout_splits.csv"):
        sel.validate_split_unchanged(cfg)


def test_validate_source_artifact_resolves_the_declared_path_exactly(tmp_path):
    """source_runs_root / source_artifact, NUNCA reconstruido desde
    architecture/source_run_id/dataset/condition."""
    cfg, pipeline = _cfg_with_pipeline(tmp_path)
    source_runs_root = tmp_path / "source_runs"
    best = sel.validate_source_artifact(pipeline, source_runs_root)  # no lanza
    assert best["dataset"] == "TOY" and best["hyperparameters"] == DEFAULT_HP

    # Moviendo el artefacto a una ruta DISTINTA de la reconstruida por convencion
    # (misma raiz, otro nombre de archivo) y actualizando source_artifact para
    # apuntar ahi, la validacion debe seguir funcionando -prueba de que de verdad
    # usa el campo declarado, no la convencion architecture/run_id/dataset/condition-.
    moved = source_runs_root / "en_otro_lado.json"
    (source_runs_root / pipeline.source_artifact).replace(moved)
    relocated = dataclasses.replace(pipeline, source_artifact="en_otro_lado.json")
    sel.validate_source_artifact(relocated, source_runs_root)  # sigue sin lanzar


def test_validate_source_artifact_detects_a_hash_mismatch(tmp_path):
    cfg, pipeline = _cfg_with_pipeline(tmp_path)
    source_runs_root = tmp_path / "source_runs"
    sel.validate_source_artifact(pipeline, source_runs_root)  # coincide, no lanza

    # El archivo cambio (otro lr) sin actualizar source_artifact_sha256 en la seleccion congelada.
    _write_best_hp(source_runs_root, dataset="TOY", hp={**DEFAULT_HP, "lr": 0.01})
    with pytest.raises(sel.SelectionError, match="sha256"):
        sel.validate_source_artifact(pipeline, source_runs_root)


def test_validate_source_artifact_checks_content_matches_this_pipeline(tmp_path):
    """El hash puede coincidir (el archivo es real) pero describir OTRO
    dataset/condicion/rama/modelo: el contenido tambien se valida, campo a campo."""
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root, dataset="TOY", condition="dn", branch="no_dn")
    _, pipeline = _cfg_with_pipeline(
        tmp_path, write_source_artifact=False, source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    with pytest.raises(sel.SelectionError, match="branch"):
        sel.validate_source_artifact(pipeline, source_runs_root)


def test_validate_source_artifact_rejects_wrong_protocol(tmp_path):
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root, dataset="TOY", protocol="holdout_final_v1")
    _, pipeline = _cfg_with_pipeline(
        tmp_path, write_source_artifact=False, source_artifact=relative, source_artifact_sha256=artifact_sha,
    )
    with pytest.raises(sel.SelectionError, match="protocol"):
        sel.validate_source_artifact(pipeline, source_runs_root)


def test_validate_source_artifact_rejects_config_index_mismatch(tmp_path):
    source_runs_root = tmp_path / "source_runs"
    relative, artifact_sha = _write_best_hp(source_runs_root, dataset="TOY", config_index=5)
    _, pipeline = _cfg_with_pipeline(
        tmp_path, write_source_artifact=False, source_artifact=relative, source_artifact_sha256=artifact_sha,
        source_config_index=0,
    )
    with pytest.raises(sel.SelectionError, match="config_index"):
        sel.validate_source_artifact(pipeline, source_runs_root)


def test_validate_source_artifact_requires_the_declared_artifact(tmp_path):
    _, pipeline = _cfg_with_pipeline(tmp_path, write_source_artifact=False)
    with pytest.raises(sel.SelectionError, match="procedencia"):
        sel.validate_source_artifact(pipeline, tmp_path / "source_runs")


def test_device_review_is_only_required_for_icbhi_when_approved(tmp_path):
    _, fraiwan_pipeline = _cfg_with_pipeline(tmp_path, dataset="FRAIWAN_Extended")
    sel.validate_device_review(fraiwan_pipeline)  # no es ICBHI: nunca lo exige

    _, icbhi_pending = _cfg_with_pipeline(tmp_path, dataset="ICBHI", status="pending_device_review")
    sel.validate_device_review(icbhi_pending)  # no aprobado: tampoco lo exige todavia

    _, icbhi_approved = _cfg_with_pipeline(tmp_path, dataset="ICBHI", status="approved")
    with pytest.raises(sel.SelectionError, match="device_review"):
        sel.validate_device_review(icbhi_approved)


def test_approving_icbhi_falsely_without_device_review_is_rejected(tmp_path):
    """Cambiar UNICAMENTE status a 'approved' nunca basta por si solo para
    ICBHI: sin device_review_path/sha256 registrados, la validacion completa
    debe seguir rechazando el acceso al test."""
    cfg, _ = _cfg_with_pipeline(tmp_path, dataset="ICBHI", status="approved")
    source_runs_root = tmp_path / "source_runs"
    with pytest.raises(sel.SelectionError, match="device_review"):
        sel.validate_pipeline_for_execution(cfg, "ICBHI", source_runs_root)


def test_device_review_passes_once_the_file_and_hash_are_registered(tmp_path):
    review_path = tmp_path / "device_review.csv"
    review_path.write_text("device,balanced_accuracy\nMeditron,0.95\n", encoding="utf-8")

    # device_review_path ABSOLUTA (ver _write_toml): dmod.REPO_ROOT / absoluta = absoluta, sin tocar el repo real.
    _, icbhi_approved = _cfg_with_pipeline(
        tmp_path, dataset="ICBHI", status="approved",
        device_review_path=str(review_path), device_review_sha256=dmod.sha256_file(review_path),
    )
    sel.validate_device_review(icbhi_approved)  # no lanza


def test_validate_reference_config_checks_protocol_and_architecture(tmp_path):
    _, pipeline = _cfg_with_pipeline(tmp_path)
    cfg = sel.validate_reference_config(pipeline)
    assert dmod.is_holdout_protocol(cfg) and dmod.model_architecture(cfg) == "cnn"

    _, wrong = _cfg_with_pipeline(tmp_path, architecture="crnn")
    with pytest.raises(sel.SelectionError, match="arquitectura"):
        sel.validate_reference_config(wrong)


def test_validate_pipeline_for_execution_runs_every_check_in_order(tmp_path):
    cfg, _ = _cfg_with_pipeline(tmp_path, status="pending_device_review")
    source_runs_root = tmp_path / "source_runs"
    with pytest.raises(sel.SelectionError, match="pending_device_review"):
        sel.validate_pipeline_for_execution(cfg, "TOY", source_runs_root)

    cfg, _ = _cfg_with_pipeline(tmp_path, status="approved")
    pipeline = sel.validate_pipeline_for_execution(cfg, "TOY", tmp_path / "source_runs")
    assert pipeline.dataset == "TOY" and pipeline.approved


# ---------------------------------------------------------------------------
# Huella canonica POR DATASET: aprobar/cambiar un pipeline no debe alterar la
# huella de otro dentro del MISMO selected_pipelines.toml.
# ---------------------------------------------------------------------------

def _pipeline(dataset, condition, branch, source_artifact, source_artifact_sha256, status):
    return sel.SelectedPipeline(
        dataset=dataset, architecture="cnn", condition=condition, branch=branch,
        reference_config=V3_DIR / "cnn_v3.toml", hyperparameters=dict(DEFAULT_HP),
        epochs=1, status=status, source_run_id="toy_run", source_config_index=0,
        source_artifact=source_artifact, source_artifact_sha256=source_artifact_sha256,
        robustness_run_id="", device_review_path="", device_review_sha256="",
    )


def test_canonical_fingerprint_of_one_dataset_does_not_depend_on_another(tmp_path):
    source_runs_root = tmp_path / "source_runs"
    icbhi_artifact, icbhi_sha = _write_best_hp(source_runs_root, dataset="ICBHI", condition="no_dn", branch="no_dn")
    fraiwan_artifact, fraiwan_sha = _write_best_hp(source_runs_root, dataset="FRAIWAN_Extended", condition="dn", branch="dn")
    csv_path, manifest_path = _write_split(tmp_path, dataset="ICBHI")
    selection_path = tmp_path / "selected_pipelines.toml"
    selection_path.write_text("placeholder", encoding="utf-8")

    def _cfg(icbhi_status):
        icbhi = _pipeline("ICBHI", "no_dn", "no_dn", icbhi_artifact, icbhi_sha, icbhi_status)
        fraiwan = _pipeline("FRAIWAN_Extended", "dn", "dn", fraiwan_artifact, fraiwan_sha, "approved")
        cfg = sel.SelectionConfig(
            path=selection_path, threshold=0.5, seed_base=20260914, split_csv=csv_path, split_manifest=manifest_path,
            split_csv_sha256=dmod.sha256_file(csv_path), split_manifest_sha256=dmod.sha256_file(manifest_path),
            bootstrap_n_resamples=20, bootstrap_confidence=0.95, bootstrap_random_state=0,
            pipelines={"ICBHI": icbhi, "FRAIWAN_Extended": fraiwan},
        )
        return cfg, fraiwan

    cfg_pending, fraiwan_pending = _cfg("pending_device_review")
    cfg_approved, fraiwan_approved = _cfg("approved")
    reference_fp = "same-architecture-fingerprint"

    fraiwan_fp_pending = sel.canonical_pipeline_fingerprint(fraiwan_pending, cfg_pending, reference_fp)
    fraiwan_fp_approved = sel.canonical_pipeline_fingerprint(fraiwan_approved, cfg_approved, reference_fp)
    assert fraiwan_fp_pending == fraiwan_fp_approved  # cambiar SOLO el status de ICBHI no toca a FRAIWAN

    icbhi_fp_pending = sel.canonical_pipeline_fingerprint(cfg_pending.pipeline_for("ICBHI"), cfg_pending, reference_fp)
    icbhi_fp_approved = sel.canonical_pipeline_fingerprint(cfg_approved.pipeline_for("ICBHI"), cfg_approved, reference_fp)
    assert icbhi_fp_pending != icbhi_fp_approved  # pero la huella de ICBHI MISMO si cambia
