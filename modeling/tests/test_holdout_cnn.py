"""CNN/CRNN en el protocolo holdout-v3 (``holdout_cnn.py``): una configuracion
por unidad con AdamW/scheduler/AMP fijos, mejor epoca y predicciones de
validation por unidad, SpecAugment solo en los batches de entrenamiento,
``no_dn_aug`` reutilizando la configuracion global de ``no_dn``, y un extremo a
extremo diminuto en CPU. Se omite si no hay PyTorch (igual que test_cnn.py).
"""

import copy
import json
import logging
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from .. import artifacts as art  # noqa: E402
from .. import data as dmod  # noqa: E402
from .. import holdout_cnn as hcnn  # noqa: E402
from .. import holdout_cv as hcv  # noqa: E402
from .. import run_experiment as rexp  # noqa: E402
from ..models import cnn as cnn_model  # noqa: E402
from .test_holdout_svm_run import _setup  # noqa: E402  (mismo split y folds sinteticos)

V3_DIR = dmod.HOLDOUT_FINAL_CONFIGS_DIR
LOGGER = logging.getLogger("test_holdout_cnn")
AUGMENT = {
    "time_mask_max_frames": 2, "time_mask_probability": 1.0,
    "freq_mask_max_bands": 2, "freq_mask_probability": 1.0, "fill_value": 0.0,
}


def _tiny_cfg():
    return {
        "cnn": {"in_channels": 1, "channels": [4, 8], "hidden_units": 8},
        "training": {
            "batch_size": 4, "eval_batch_size": 8, "max_epochs": 2, "min_epochs": 2, "patience": 1,
            "weight_decay": 1e-4, "grad_clip_norm": 1.0, "t_max": 2, "eta_min": 1e-6, "amp": False,
        },
        "evaluation": {"decision_threshold": 0.5},
        "normalization": {"weighted": True, "std_floor": 1e-6},
        "seeds": {"base": 1},
    }


def _tiny_data():
    """16 segmentos de train (8 pacientes) y 8 de validation (4 pacientes), Log-Mel 1x16x16."""
    rng = np.random.default_rng(0)
    rows = []
    for patient in range(12):
        for k in range(2):
            rows.append({
                "cache_row": len(rows), "patient_uid": f"P{patient:02d}", "audio_id": f"P{patient:02d}_rec0",
                "segment_id": f"P{patient:02d}_s{k}", "target_label": patient % 2, "device": "Meditron",
                "dataset": "TOY",
            })
    segments = pd.DataFrame(rows)
    logmel = rng.standard_normal((len(segments), 1, 16, 16)).astype(np.float32)
    logmel += segments["target_label"].to_numpy()[:, None, None, None]
    train_ids = {f"P{p:02d}" for p in range(8)}
    train_seg = segments.loc[segments["patient_uid"].isin(train_ids)].reset_index(drop=True)
    val_seg = segments.loc[~segments["patient_uid"].isin(train_ids)].reset_index(drop=True)
    return logmel, train_seg, val_seg


CANDIDATE = {"config_index": 7, "lr": 1e-3, "dropout": 0.2, "weight_decay": 1e-4, "batch_size": 4}


def _run_tiny(augment_cfg=None):
    cfg = _tiny_cfg()
    logmel, train_seg, val_seg = _tiny_data()
    config = hcnn.to_configuration(CANDIDATE)
    settings = hcnn.settings_for(cnn_model.TrainingSettings.from_config(cfg, 0), config)
    run = cnn_model.train_with_validation(
        logmel, train_seg, val_seg, cfg, config, settings, torch.device("cpu"), "Healthy",
        seed_parts=("TOY", 0, config.index, "cv"), augment_cfg=augment_cfg,
    )
    return run, val_seg, settings


# ---------------------------------------------------------------------------
# Configuracion por unidad
# ---------------------------------------------------------------------------

def test_configuration_carries_the_four_searched_hyperparameters():
    config = hcnn.to_configuration(CANDIDATE)
    assert (config.index, config.lr, config.dropout) == (7, 1e-3, 0.2)
    assert (config.weight_decay, config.batch_size) == (1e-4, 4)


def test_settings_override_only_batch_size_and_weight_decay():
    cfg = dmod.load_config(V3_DIR / "cnn_v3.toml")
    base = cnn_model.TrainingSettings.from_config(cfg, num_workers=0)
    settings = hcnn.settings_for(base, hcnn.to_configuration({**CANDIDATE, "weight_decay": 1e-3, "batch_size": 64}))
    assert (settings.batch_size, settings.weight_decay) == (64, 1e-3)
    for field in ("max_epochs", "min_epochs", "patience", "grad_clip_norm", "t_max", "eta_min", "amp", "threshold"):
        assert getattr(settings, field) == getattr(base, field), field
    assert (base.max_epochs, base.min_epochs, base.patience, base.threshold) == (150, 20, 15, 0.5)


def test_smoke_config_keeps_the_protocol_but_trains_two_epochs():
    cfg = dmod.load_config(V3_DIR / "cnn_v3.toml")
    smoke = hcnn.smoke_test_config(cfg)
    assert smoke["training"]["max_epochs"] == 2 and smoke["training"]["min_epochs"] == 2
    assert cfg["training"]["max_epochs"] == 150                       # el original no se toca
    assert smoke["search"] == cfg["search"]


@pytest.mark.parametrize("name", ("crnn_v3.toml", "crnn_combined_v3.toml"))
def test_crnn_consistency_checks_pass_for_the_shipped_configs_and_fail_after_a_change(name):
    cfg = dmod.load_config(V3_DIR / name)
    blocking = [c for c in hcnn.holdout_consistency_checks(cfg) if c["blocking"]]
    assert blocking and all(c["ok"] for c in blocking)

    changed = copy.deepcopy(cfg)
    changed["search"]["n_configurations"] = 10
    failing = {c["check"] for c in hcnn.holdout_consistency_checks(changed) if c["blocking"] and not c["ok"]}
    assert "search_igual_cnn" in failing


def test_cnn_consistency_checks_are_not_blocking():
    cfg = dmod.load_config(V3_DIR / "cnn_v3.toml")
    assert not any(c["blocking"] for c in hcnn.holdout_consistency_checks(cfg))


# ---------------------------------------------------------------------------
# Entrenamiento de una unidad
# ---------------------------------------------------------------------------

def test_train_with_validation_keeps_the_best_epoch_logits_that_match_its_metrics():
    run, val_seg, settings = _run_tiny()
    assert run.best_logits is not None and run.best_logits.shape == (len(val_seg),)
    metrics, *_ = cnn_model.evaluate_patients(
        val_seg, cnn_model.sigmoid(run.best_logits), "Healthy", settings.threshold,
    )
    assert metrics["balanced_accuracy"] == pytest.approx(run.best_metrics["balanced_accuracy"])
    assert 1 <= run.best_epoch <= run.epochs_run == 2


def test_augmentation_is_applied_only_to_training_batches(monkeypatch):
    calls = []
    original = cnn_model.spec_augment

    def counting(x, aug_cfg, generator):
        calls.append(int(x.shape[0]))
        return original(x, aug_cfg, generator)

    monkeypatch.setattr(cnn_model, "spec_augment", counting)

    run, _, settings = _run_tiny(augment_cfg=AUGMENT)
    # 16 segmentos de train / batch 4 = 4 batches por epoca; nunca en validation.
    assert len(calls) == run.epochs_run * 4 and max(calls) <= settings.batch_size

    calls.clear()
    _run_tiny(augment_cfg=None)
    assert calls == []                                              # sin augmentation, ninguna llamada


def test_evaluate_fold_yields_outputs_and_keeps_going_after_a_failing_configuration():
    logmel, train_seg, val_seg = _tiny_data()
    context = hcnn.NnFoldContext(fold_id=0, condition=None, logmel=logmel, train_seg=train_seg, val_seg=val_seg)
    spec = dmod.ConditionSpec(dataset="TOY", condition="no_dn", branch="no_dn", dn_reliable_only=False)
    base = cnn_model.TrainingSettings.from_config(_tiny_cfg(), 0)
    evaluate = hcnn.make_evaluate_fold(spec, _tiny_cfg(), base, torch.device("cpu"), "Healthy", LOGGER)

    broken = {**CANDIDATE, "config_index": 1, "batch_size": 0}       # DataLoader invalido
    outcomes = list(evaluate(context, [broken, CANDIDATE]))

    assert [c["config_index"] for c, _ in outcomes] == [1, 7]
    assert isinstance(outcomes[0][1], hcv.UnitFailure)
    good = outcomes[1][1]
    assert isinstance(good, hcv.UnitOutput)
    assert list(good.val_scores.columns) == hcv.SCORE_COLUMNS and len(good.val_scores) == len(val_seg)
    assert good.best_epoch in (1, 2) and good.epochs_run == 2 and "fold" in good.history.columns
    assert good.val_scores["score"].between(0.0, 1.0).all()          # probabilidades

    # history.csv es autosuficiente: lleva los cuatro hiperparametros de la configuracion en cada fila.
    history = good.history
    assert set(history["config_index"]) == {CANDIDATE["config_index"]}
    assert set(history["lr"]) == {CANDIDATE["lr"]} and set(history["dropout"]) == {CANDIDATE["dropout"]}
    assert set(history["weight_decay"]) == {CANDIDATE["weight_decay"]}
    assert set(history["batch_size"]) == {CANDIDATE["batch_size"]}
    assert len(history) == good.epochs_run and history["epoch"].tolist() == [1, 2]


# ---------------------------------------------------------------------------
# no_dn_aug reutiliza la configuracion global de no_dn
# ---------------------------------------------------------------------------

def _write_base_condition(run_root, candidate, status="COMPLETED"):
    cdir = art.condition_dir(run_root, "TOY", "no_dn")
    cdir.mkdir(parents=True)
    (cdir / "best_hyperparameters.json").write_text(json.dumps({
        "config_index": candidate["config_index"],
        "hyperparameters": {k: candidate[k] for k in hcv.NN_HYPERPARAMETER_KEYS},
    }), encoding="utf-8")
    art.write_status(cdir, status, {})


AUG_SPEC = dmod.ConditionSpec(
    dataset="TOY", condition="no_dn_aug", branch="no_dn", dn_reliable_only=False,
    augment=True, hyperparameters_from="no_dn",
)


def test_load_reused_candidate_returns_the_global_configuration_of_no_dn(tmp_path):
    candidates = hcv.nn_candidates(dmod.load_config(V3_DIR / "cnn_v3.toml"))
    chosen = candidates[11]
    _write_base_condition(tmp_path, chosen)
    assert hcnn.load_reused_candidate(tmp_path, AUG_SPEC, candidates) == chosen


def test_load_reused_candidate_refuses_an_unfinished_or_inconsistent_base(tmp_path):
    candidates = hcv.nn_candidates(dmod.load_config(V3_DIR / "cnn_v3.toml"))
    with pytest.raises(RuntimeError, match="COMPLETED"):
        hcnn.load_reused_candidate(tmp_path / "missing", AUG_SPEC, candidates)

    partial = tmp_path / "partial"
    _write_base_condition(partial, candidates[3], status="PARTIAL")
    with pytest.raises(RuntimeError, match="COMPLETED"):
        hcnn.load_reused_candidate(partial, AUG_SPEC, candidates)

    inconsistent = tmp_path / "inconsistent"
    _write_base_condition(inconsistent, {**candidates[3], "lr": 0.5})
    with pytest.raises(RuntimeError, match="no coinciden"):
        hcnn.load_reused_candidate(inconsistent, AUG_SPEC, candidates)


# ---------------------------------------------------------------------------
# Puntos de entrada
# ---------------------------------------------------------------------------

def _nn_cfg(csv_path, manifest_path, experiments):
    cfg = copy.deepcopy(dmod.load_config(V3_DIR / "cnn_v3.toml"))
    cfg["holdout"]["split_csv"] = str(csv_path)
    cfg["holdout"]["split_manifest"] = str(manifest_path)
    cfg["figures"] = {"dpi": 50, "formats": ["png"]}       # figuras pequenas para las pruebas
    cfg["datasets"] = {"TOY": {"positive_diagnosis": "COPD", "negative_diagnosis": "Healthy", "negative_label_name": "Healthy"}}
    cfg["experiments"] = experiments
    return cfg


MAIN_EXPERIMENT = {
    "main": {"TOY": {"condition": "no_dn", "branch": "no_dn", "dn_reliable_only": False}},
}

AUGMENTATION_EXPERIMENT = {
    "augmentation_ablation": {"TOY": [
        {"condition": "no_dn", "branch": "no_dn", "dn_reliable_only": False, "reused_from_main": True},
        {"condition": "no_dn_aug", "branch": "no_dn", "dn_reliable_only": False, "augment": True,
         "hyperparameters_from": "no_dn"},
    ]},
}


def _args(tmp_path, data_root, **overrides):
    base = dict(
        model="cnn", dataset="TOY", experiment="augmentation_ablation", n_jobs=1, device="cpu", num_workers=0,
        data_root=data_root, runs_root=tmp_path / "runs", cache_root=tmp_path / "cache", config=None,
        dry_run=False, smoke_test=True, force_features=False, resume=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_run_refuses_to_start_when_outer_test_or_final_model_is_enabled(tmp_path):
    _, data_root, _, csv_path, manifest_path = _setup(tmp_path, folds=[0])
    for section in ("outer_test", "final_model"):
        cfg = _nn_cfg(csv_path, manifest_path, AUGMENTATION_EXPERIMENT)
        cfg[section]["enabled"] = True
        with pytest.raises(RuntimeError, match=section):
            hcnn.run_holdout_cnn(_args(tmp_path, data_root), cfg)
    assert not (tmp_path / "runs").exists()


def test_crnn_rejects_force_features_because_the_logmel_cache_is_shared(tmp_path):
    cfg = dmod.load_config(V3_DIR / "crnn_v3.toml")
    assert hcnn.run_holdout_cnn(_args(tmp_path, tmp_path, model="crnn", force_features=True), cfg) == 2


def test_dry_run_checks_cache_classes_parameters_candidates_and_device(tmp_path, capsys):
    _, data_root, _, csv_path, manifest_path = _setup(tmp_path)
    cfg = _nn_cfg(csv_path, manifest_path, MAIN_EXPERIMENT)
    specs = rexp.plan_conditions(cfg, ["TOY"], ["main"])

    rc = hcnn.dry_run_holdout(data_root, tmp_path / "cache", cfg, specs, "cpu", csv_path, manifest_path)

    out = capsys.readouterr().out
    assert rc == 0 and "veredicto: OK" in out
    for check in ("candidatos_unicos", "parametros_cnn", "dispositivo", "cache_logmel_no_dn[cv/fold_00]"):
        assert check in out
    assert not (tmp_path / "runs").exists()


def test_smoke_run_trains_no_dn_then_reuses_its_configuration_for_no_dn_aug(tmp_path):
    split, data_root, _, csv_path, manifest_path = _setup(tmp_path, folds=[0])
    cfg = _nn_cfg(csv_path, manifest_path, AUGMENTATION_EXPERIMENT)

    rc = hcnn.run_holdout_cnn(_args(tmp_path, data_root), cfg)

    assert rc == 0
    runs = list((tmp_path / "runs" / "cnn").glob("*"))
    assert len(runs) == 1
    run_root = runs[0]
    base_dir = run_root / "datasets" / "TOY" / "no_dn"
    aug_dir = run_root / "datasets" / "TOY" / "no_dn_aug"

    base = json.loads((base_dir / "best_hyperparameters.json").read_text(encoding="utf-8"))
    aug = json.loads((aug_dir / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert base["hyperparameters_source"] == "search" and base["augment"] is False
    assert aug["hyperparameters_source"] == "reused_from:no_dn" and aug["augment"] is True
    assert aug["config_index"] == base["config_index"] and aug["hyperparameters"] == base["hyperparameters"]
    assert aug["best_epochs_by_fold"] is not None and aug["median_best_epoch"] in (1, 2)   # propias de no_dn_aug

    for cdir in (base_dir, aug_dir):
        assert len(pd.read_csv(cdir / "cv_search_results.csv")) == 1
        patients = pd.read_csv(cdir / "cv_patient_predictions.csv", dtype={"patient_uid": str})
        assert not set(patients["patient_uid"]) & split.blocked_test_patients()
        assert not (cdir / "final").exists()
        for figure in ("confusion_matrix", "roc", "pr", "metrics_by_fold"):
            assert (cdir / "figures" / f"{figure}.png").is_file(), figure
        history = pd.read_csv(cdir / "cv" / "config_000" / "fold_00" / "history.csv")
        assert {"lr", "dropout", "weight_decay", "batch_size", "fold", "epoch"} <= set(history.columns)
    assert list(run_root.rglob("*.pt")) == []                       # ningun modelo guardado
    assert json.loads((run_root / "status.json").read_text(encoding="utf-8"))["status"] == "COMPLETED"
