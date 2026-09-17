"""CNN: arquitectura, normalizacion solo con train, perdida ponderada,
SpecAugment reproducible y exclusivo de entrenamiento, checkpoints,
agregacion por paciente, un fold sintetico completo, seleccion del modelo
final y rechazo de un --resume incompatible.

Todo corre en CPU con datos sinteticos; nunca toca el corpus real. Si PyTorch
no esta instalado (entorno solo-SVM) el modulo entero se omite.
"""

import copy

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from .. import data as dmod  # noqa: E402
from .. import run_experiment as rexp  # noqa: E402
from .. import splits as sp  # noqa: E402
from ..cnn_experiment import CNN_FINGERPRINT_SECTIONS  # noqa: E402
from ..models import cnn as cnn_model  # noqa: E402

AUG_ALWAYS = {
    "time_mask_max_frames": 20, "time_mask_probability": 1.0,
    "freq_mask_max_bands": 6, "freq_mask_probability": 1.0,
    "fill_value": 0.0,
}
TRAIN_PATIENTS = [f"P{i}" for i in range(4)] + [f"N{i}" for i in range(4)]
VAL_PATIENTS = ["P4", "N4"]
TEST_PATIENTS = ["P5", "N5"]


@pytest.fixture(scope="module")
def cfg():
    return dmod.load_config(dmod.CNN_CONFIG_PATH)


def _tiny_cfg(cfg):
    tiny = copy.deepcopy(cfg)
    tiny["search"]["configurations"] = cfg["search"]["configurations"][:1]
    tiny["training"].update(max_epochs=2, min_epochs=1, patience=1, batch_size=4, eval_batch_size=8, amp=False)
    return tiny


def _synthetic_condition(n_per_class=6, segments_per_patient=2, seed=0):
    """Dos clases con Log-Mel desplazado (+1.5 / -1.5); 1 grabacion por paciente."""
    rng = np.random.RandomState(seed)
    rows, arrays = [], []
    for label, prefix in ((1, "P"), (0, "N")):
        for i in range(n_per_class):
            patient = f"{prefix}{i}"
            for s in range(segments_per_patient):
                rows.append({
                    "patient_uid": patient, "audio_id": f"{patient}_r0", "segment_id": f"{patient}_r0_{s}",
                    "target_label": label, "calibration_patient": False, "cache_row": len(arrays),
                })
                arrays.append((rng.randn(1, 64, 309) + (1.5 if label else -1.5)).astype(np.float32))
    spec = dmod.ConditionSpec(dataset="TOY", condition="main_no_dn", branch="no_dn", dn_reliable_only=False)
    return dmod.ConditionLogmel(spec=spec, segments=pd.DataFrame(rows), logmel=np.stack(arrays))


def test_parameter_count_is_exact(cfg):
    assert cfg["cnn"]["expected_parameters"] == 1_205_921
    for dropout in (0.3, 0.5):
        assert cnn_model.count_parameters(cnn_model.build_model(cfg, dropout=dropout)) == 1_205_921
    assert cnn_model.architecture_description(cfg)["n_parameters"] == 1_205_921


def test_forward_returns_one_logit_per_segment(cfg):
    model = cnn_model.build_model(cfg, dropout=0.3).eval()
    with torch.no_grad():
        out = model(torch.randn(3, 1, 64, 309))
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()


def test_normalization_uses_only_the_given_rows():
    rng = np.random.RandomState(0)
    logmel = rng.randn(10, 1, 4, 7).astype(np.float32)
    stats = cnn_model.compute_normalization(logmel, [0, 1, 2])

    block = logmel[[0, 1, 2], 0].astype(np.float64)
    np.testing.assert_allclose(stats.mean, block.mean(axis=(0, 2)), atol=1e-6)
    np.testing.assert_allclose(stats.std, block.std(axis=(0, 2)), atol=1e-6)

    # Alterar filas que no son de entrenamiento no cambia nada.
    altered = logmel.copy()
    altered[3:] = 1000.0
    again = cnn_model.compute_normalization(altered, [0, 1, 2])
    np.testing.assert_array_equal(stats.mean, again.mean)
    np.testing.assert_array_equal(stats.std, again.std)


def test_weighted_normalization_follows_weights():
    logmel = np.zeros((2, 1, 1, 3), dtype=np.float32)
    logmel[0] = 1.0
    logmel[1] = 5.0
    stats = cnn_model.compute_normalization(logmel, [0, 1], weights=[3.0, 1.0])
    assert stats.mean[0] == pytest.approx(2.0)          # (3*1 + 1*5) / 4
    assert stats.std[0] == pytest.approx(np.sqrt(3.0))  # E[x^2] = 7, var = 7 - 4
    assert stats.weighted


def test_weighted_bce_loss_matches_manual_computation():
    logits = torch.tensor([2.0, -1.0, 0.5])
    targets = torch.tensor([1.0, 0.0, 0.0])
    weights = torch.tensor([0.5, 2.0, 0.5])
    manual = []
    for z, y, w in zip(logits.tolist(), targets.tolist(), weights.tolist()):
        p = 1.0 / (1.0 + np.exp(-z))
        manual.append(w * -(y * np.log(p) + (1 - y) * np.log(1 - p)))
    loss = cnn_model.weighted_bce_loss(logits, targets, weights)
    assert float(loss) == pytest.approx(float(np.mean(manual)), rel=1e-6)


def test_spec_augment_is_reproducible_bounded_and_not_in_place():
    x = torch.ones(8, 1, 64, 309)
    a = cnn_model.spec_augment(x, AUG_ALWAYS, torch.Generator().manual_seed(123))
    b = cnn_model.spec_augment(x, AUG_ALWAYS, torch.Generator().manual_seed(123))
    c = cnn_model.spec_augment(x, AUG_ALWAYS, torch.Generator().manual_seed(456))

    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.equal(x, torch.ones_like(x))
    assert set(torch.unique(a).tolist()) <= {0.0, 1.0}
    for sample in a[:, 0]:
        assert int((sample == 0).all(dim=0).sum()) <= 20  # tramas enmascaradas
        assert int((sample == 0).all(dim=1).sum()) <= 6   # bandas enmascaradas


def test_augmentation_is_applied_only_while_training(cfg, monkeypatch):
    condition = _synthetic_condition()
    tiny = _tiny_cfg(cfg)
    calls = []
    original = cnn_model.spec_augment

    def spy(x, aug_cfg, generator):
        calls.append(torch.is_grad_enabled())
        return original(x, aug_cfg, generator)

    monkeypatch.setattr(cnn_model, "spec_augment", spy)
    train = sp.filter_segments_by_patients(condition.segments, TRAIN_PATIENTS)
    val = sp.filter_segments_by_patients(condition.segments, VAL_PATIENTS)
    settings = cnn_model.TrainingSettings.from_config(tiny, num_workers=0)

    run = cnn_model.train_with_validation(
        condition.logmel, train, val, tiny, cnn_model.search_configurations(tiny)[0], settings,
        torch.device("cpu"), "Control", seed_parts=("TOY", 0, 0, "search"), augment_cfg=AUG_ALWAYS,
    )

    batches_per_epoch = int(np.ceil(len(train) / settings.batch_size))
    assert len(calls) == run.epochs_run * batches_per_epoch  # nunca en la inferencia de validation
    assert all(calls)


def test_checkpoint_roundtrip_reproduces_outputs(tmp_path, cfg):
    torch.manual_seed(0)
    model = cnn_model.build_model(cfg, dropout=0.3).eval()
    stats = cnn_model.NormalizationStats(mean=np.zeros(64), std=np.ones(64), n_segments=1, weighted=False)
    spec = dmod.ConditionSpec("TOY", "main_no_dn", "no_dn", False)
    payload = cnn_model.checkpoint_payload(
        {k: v.clone() for k, v in model.state_dict().items()}, stats, cfg, spec, "Healthy",
        {"config_index": 0, "lr": 1e-3, "dropout": 0.3, "epochs": 5},
    )
    path = tmp_path / "model.pt"

    cnn_model.save_checkpoint(path, payload)
    loaded = cnn_model.load_checkpoint(path)
    restored = cnn_model.model_from_checkpoint(loaded)

    for key, value in model.state_dict().items():
        assert torch.equal(value, loaded["state_dict"][key])
    x = torch.randn(2, 1, 64, 309)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))
    assert loaded["label_mapping"] == {"Healthy": 0, "COPD": 1}
    assert not path.with_name("model.pt.part").exists()


def test_patient_evaluation_uses_two_stage_mean_and_threshold():
    segments = pd.DataFrame([
        {"patient_uid": "A", "audio_id": "A1", "segment_id": "a", "target_label": 1},
        {"patient_uid": "A", "audio_id": "A1", "segment_id": "b", "target_label": 1},
        {"patient_uid": "A", "audio_id": "A2", "segment_id": "c", "target_label": 1},
        {"patient_uid": "B", "audio_id": "B1", "segment_id": "d", "target_label": 0},
    ])
    probabilities = np.array([0.2, 0.4, 0.9, 0.6])

    metrics, _, _, patients = cnn_model.evaluate_patients(segments, probabilities, "Healthy", 0.5)

    scores = dict(zip(patients["patient_uid"], patients["score"]))
    assert scores["A"] == pytest.approx((0.3 + 0.9) / 2)  # 0.6, no la media directa 0.5
    assert scores["B"] == pytest.approx(0.6)
    assert metrics["recall_copd"] == pytest.approx(1.0)
    assert metrics["recall_healthy"] == pytest.approx(0.0)
    assert metrics["min_class_recall"] == pytest.approx(0.0)


def test_run_cnn_fold_end_to_end_on_synthetic_data(cfg):
    condition = _synthetic_condition()
    tiny = _tiny_cfg(cfg)

    result = cnn_model.run_cnn_fold(
        0, condition, TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS, tiny,
        cnn_model.search_configurations(tiny), "Control", torch.device("cpu"), num_workers=0,
    )

    assert sorted(result.patient_predictions["patient_uid"]) == ["N5", "P5"]
    assert set(result.segment_predictions["patient_uid"]) == {"N5", "P5"}
    assert result.selected["source"] == "search"
    assert 1 <= result.selected["best_epoch"] <= 2
    assert len(result.refit_history) == result.selected["best_epoch"]
    assert result.normalization_search.n_segments == 16  # solo train
    assert result.normalization_refit.n_segments == 20   # train + validation
    assert sorted(result.baseline_table["strategy"]) == ["most_frequent", "stratified"]
    assert result.segment_predictions["score"].between(0.0, 1.0).all()
    model = cnn_model.build_model(tiny, dropout=result.selected["dropout"])
    model.load_state_dict(result.model_state)


def test_final_configuration_uses_mean_validation_and_median_epochs():
    rows = []
    for fold, (epoch0, ba0, epoch1, ba1) in enumerate([(10, 0.7, 5, 0.9), (30, 0.7, 7, 0.6), (20, 0.7, 40, 0.9)]):
        for index, lr, epoch, ba in ((0, 1e-3, epoch0, ba0), (1, 3e-4, epoch1, ba1)):
            rows.append({
                "fold": fold, "config_index": index, "lr": lr, "dropout": 0.3, "best_epoch": epoch,
                "val_balanced_accuracy": ba, "val_macro_f1": 0.5, "val_min_class_recall": 0.5, "val_loss": 0.6,
            })

    config, epochs, aggregated = cnn_model.select_final_configuration(pd.DataFrame(rows))

    assert config.index == 1          # media 0.8 frente a 0.7
    assert epochs == 7                # mediana de 5, 7, 40
    assert len(aggregated) == 2


def test_resume_rejects_incompatible_fingerprint(tmp_path, cfg):
    dataset_dir = tmp_path / "data" / "TOY"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "segments.csv").write_text("segment_id\nS0\n", encoding="utf-8")
    np.save(dataset_dir / "segments_no_dn.npy", np.zeros((1, 4), dtype=np.float32))
    specs = [dmod.ConditionSpec("TOY", "main_no_dn", "no_dn", False)]
    run_root = tmp_path / "run"
    run_root.mkdir()
    extra = {"model": "cnn", "architecture": cnn_model.architecture_description(cfg), "smoke_test": False}

    def fingerprint(config, extra_values):
        return rexp.build_run_fingerprint(
            config, tmp_path / "data", specs, "TOY", "main",
            sections=CNN_FINGERPRINT_SECTIONS, extra=extra_values,
        )

    rexp.write_run_fingerprint(run_root, fingerprint(cfg, extra))
    rexp.verify_run_fingerprint(run_root, fingerprint(cfg, copy.deepcopy(extra)))

    with pytest.raises(RuntimeError):
        rexp.verify_run_fingerprint(run_root, fingerprint(cfg, {**extra, "smoke_test": True}))

    for section, key, value in (("training", "batch_size", 16), ("augmentation", "time_mask_max_frames", 10)):
        changed = copy.deepcopy(cfg)
        changed[section][key] = value
        with pytest.raises(RuntimeError):
            rexp.verify_run_fingerprint(run_root, fingerprint(changed, extra))


def test_checkpoint_with_unknown_architecture_is_rejected(cfg):
    model = cnn_model.build_model(cfg, dropout=0.3)
    payload = {
        "architecture": {**cnn_model.architecture_description(cfg), "class": "OtraRed", "dropout": 0.3},
        "state_dict": model.state_dict(),
    }
    with pytest.raises(ValueError, match="OtraRed"):
        cnn_model.model_from_checkpoint(payload)


def test_run_cnn_incompatible_resume_leaves_status_and_log_intact(tmp_path, cfg, monkeypatch, capsys):
    from types import SimpleNamespace

    from ..cnn_experiment import run_cnn

    # Evita cambiar el modo determinista global de torch dentro de la suite.
    monkeypatch.setattr(cnn_model, "configure_determinism", lambda _cfg: None)

    data_root = tmp_path / "data"
    dataset_dir = data_root / "ICBHI"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "segments.csv").write_text("segment_id\nS0\n", encoding="utf-8")
    np.save(dataset_dir / "segments_no_dn.npy", np.zeros((1, 4), dtype=np.float32))

    runs_root = tmp_path / "runs"
    run_root = runs_root / "cnn" / "20260101T000000Z_abcdef"
    staging = run_root / "datasets" / "ICBHI" / "main_no_dn" / "fold_01_staging"
    staging.mkdir(parents=True)
    (staging / "parcial.csv").write_text("a\n1\n", encoding="utf-8")
    (run_root / "status.json").write_text('{"status": "PARTIAL"}\n', encoding="utf-8")
    (run_root / "run.log").write_text("linea original del log\n", encoding="utf-8")
    rexp.write_run_fingerprint(run_root, {
        "dataset_arg": "all", "experiment_arg": "all",
        "config_fingerprint": "otra-configuracion", "input_hashes": {},
    })

    def snapshot():
        return {
            p.relative_to(run_root).as_posix(): (p.read_bytes() if p.is_file() else None)
            for p in sorted(run_root.rglob("*"))
        }

    before = snapshot()
    args = SimpleNamespace(
        model="cnn", dataset="ICBHI", experiment="main", data_root=data_root, runs_root=runs_root,
        cache_root=tmp_path / "cache", dry_run=False, smoke_test=False, device="cpu", num_workers=0,
        force_features=False, resume="20260101T000000Z_abcdef",
    )

    assert run_cnn(args, cfg) == 1
    assert "--resume rechazado" in capsys.readouterr().err
    assert snapshot() == before
    assert (run_root / "status.json").read_text(encoding="utf-8") == '{"status": "PARTIAL"}\n'
    assert (run_root / "run.log").read_text(encoding="utf-8") == "linea original del log\n"
