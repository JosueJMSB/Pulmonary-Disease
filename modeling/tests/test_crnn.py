"""CRNN (CNN + BiGRU + atencion): configuracion igual a la CNN, conteo exacto
de parametros, formas intermedias, pesos de atencion, gradientes, integracion
con el protocolo compartido de models/cnn.py (sin fuga, normalizacion solo con
train, agregacion por paciente), checkpoints y rechazo de un --resume
incompatible.

Todo corre en CPU con datos sinteticos; nunca toca el corpus real. Si PyTorch
no esta instalado, el modulo se omite.
"""

import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from .. import data as dmod  # noqa: E402
from .. import run_experiment as rexp  # noqa: E402
from ..cnn_experiment import (  # noqa: E402
    CNN_FINGERPRINT_SECTIONS,
    CRNN_FINGERPRINT_SECTIONS,
    SHARED_PROTOCOL_SECTIONS,
    config_consistency_checks,
    fingerprint_sections,
)
from ..crnn_experiment import run_crnn  # noqa: E402
from ..models import cnn as cnn_model  # noqa: E402
from ..models import crnn as crnn_arch  # noqa: E402

TRAIN_PATIENTS = [f"P{i}" for i in range(4)] + [f"N{i}" for i in range(4)]
VAL_PATIENTS = ["P4", "N4"]
TEST_PATIENTS = ["P5", "N5"]
EXPECTED_PARAMETERS = 1_192_482


@pytest.fixture(scope="module")
def cfg():
    return dmod.load_config(dmod.CRNN_CONFIG_PATH)


@pytest.fixture(scope="module")
def cnn_cfg():
    return dmod.load_config(dmod.CNN_CONFIG_PATH)


def _tiny_cfg(cfg):
    tiny = copy.deepcopy(cfg)
    tiny["search"]["configurations"] = cfg["search"]["configurations"][:1]
    tiny["training"].update(max_epochs=2, min_epochs=1, patience=1, batch_size=4, eval_batch_size=8, amp=False)
    return tiny


def _synthetic_condition(n_per_class=6, segments_per_patient=2, seed=0):
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


def test_config_declares_crnn_and_copies_the_cnn_protocol(cfg, cnn_cfg):
    assert dmod.model_architecture(cfg) == "crnn"
    assert dmod.model_architecture(cnn_cfg) == "cnn"
    for section in SHARED_PROTOCOL_SECTIONS:
        assert cfg[section] == cnn_cfg[section], section
    assert all(check["ok"] for check in config_consistency_checks(cfg))


def test_protocol_difference_is_blocking(cfg):
    changed = copy.deepcopy(cfg)
    changed["training"]["batch_size"] = 16
    blocking = [c for c in config_consistency_checks(changed) if c["blocking"] and not c["ok"]]
    assert [c["check"] for c in blocking] == ["training_igual_cnn"]


def test_original_crnn_reference_defaults_to_cnn_toml(cfg):
    # crnn.toml no declara [model] reference_cnn_config: sigue comparandose
    # con cnn.toml, exactamente como antes de que existiera COMBINED.
    assert dmod.reference_cnn_config_path(cfg) == dmod.CNN_CONFIG_PATH


def test_combined_crnn_compares_against_combined_cnn_toml():
    combined_crnn_cfg = dmod.load_config(dmod.CNN_CONFIG_PATH.parent / "crnn_combined.toml")
    combined_cnn_cfg = dmod.load_config(dmod.CNN_CONFIG_PATH.parent / "cnn_combined.toml")

    assert dmod.reference_cnn_config_path(combined_crnn_cfg) == dmod.CNN_CONFIG_PATH.parent / "cnn_combined.toml"
    for section in SHARED_PROTOCOL_SECTIONS:
        assert combined_crnn_cfg[section] == combined_cnn_cfg[section], section
    assert all(check["ok"] for check in config_consistency_checks(combined_crnn_cfg))


def test_combined_crnn_flags_difference_from_combined_cnn_toml_not_original():
    combined_crnn_cfg = dmod.load_config(dmod.CNN_CONFIG_PATH.parent / "crnn_combined.toml")
    original_cnn_cfg = dmod.load_config(dmod.CNN_CONFIG_PATH)
    # cnn_combined.toml difiere de cnn.toml en [datasets]/[experiments]
    # (COMBINED en vez de ICBHI/FRAIWAN_Extended): si la CRNN combinada se
    # comparara por error contra cnn.toml, esto se marcaria como bloqueante.
    assert combined_crnn_cfg["datasets"] != original_cnn_cfg["datasets"]
    blocking = [c["check"] for c in config_consistency_checks(combined_crnn_cfg) if c["blocking"] and not c["ok"]]
    assert blocking == []


def test_parameter_count_is_exact(cfg):
    assert cfg["crnn"]["expected_parameters"] == EXPECTED_PARAMETERS
    for dropout in (0.3, 0.5):
        model = cnn_model.build_model(cfg, dropout)
        assert isinstance(model, crnn_arch.CopdCRNN)
        assert cnn_model.count_parameters(model) == EXPECTED_PARAMETERS
    assert cnn_model.expected_parameters(cfg) == EXPECTED_PARAMETERS
    assert cnn_model.architecture_description(cfg)["n_parameters"] == EXPECTED_PARAMETERS


def test_parameter_breakdown_matches_the_design(cfg):
    model = crnn_arch.build_crnn(cfg, dropout=0.3)

    def count(module):
        return sum(p.numel() for p in module.parameters())

    assert count(model.features) == 582_304
    assert count(model.projection) == 65_792
    assert count(model.gru) == 494_592
    assert count(model.attention) == 16_513
    assert count(model.head) == 33_281


def test_intermediate_shapes(cfg):
    model = crnn_arch.build_crnn(cfg, dropout=0.3)
    shapes = model.intermediate_shapes(torch.randn(2, 1, 64, 309))
    assert shapes == {
        "features": (2, 128, 4, 77),
        "sequence": (2, 77, 512),
        "projection": (2, 77, 128),
        "recurrent": (2, 77, 256),
        "attention": (2, 77),
        "pooled": (2, 256),
        "logits": (2, 1),
    }
    description = cnn_model.architecture_description(cfg)
    assert description["sequence_steps"] == 77
    assert description["step_ms"] == pytest.approx(64.0)


def test_forward_returns_logits_and_optional_attention_weights(cfg):
    model = crnn_arch.build_crnn(cfg, dropout=0.3).eval()
    x = torch.randn(3, 1, 64, 309)
    with torch.no_grad():
        logits = model(x)
        logits_again, weights = model(x, return_attention=True)

    assert logits.shape == (3, 1)
    assert torch.equal(logits, logits_again)
    assert weights.shape == (3, 77)
    assert torch.isfinite(weights).all()
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))


def test_gradients_reach_every_parameter(cfg):
    torch.manual_seed(0)
    model = crnn_arch.build_crnn(cfg, dropout=0.3).train()
    logits = model(torch.randn(4, 1, 64, 309)).squeeze(1)
    loss = cnn_model.weighted_bce_loss(logits, torch.tensor([1.0, 0.0, 1.0, 0.0]), torch.ones(4))
    loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert any(float(p.grad.abs().sum()) > 0 for p in model.gru.parameters())
    assert any(float(p.grad.abs().sum()) > 0 for p in model.attention.parameters())


def test_checkpoint_roundtrip_restores_the_crnn(tmp_path, cfg):
    torch.manual_seed(0)
    model = cnn_model.build_model(cfg, dropout=0.5).eval()
    stats = cnn_model.NormalizationStats(mean=np.zeros(64), std=np.ones(64), n_segments=1, weighted=False)
    spec = dmod.ConditionSpec("TOY", "main_no_dn", "no_dn", False)
    payload = cnn_model.checkpoint_payload(
        {k: v.clone() for k, v in model.state_dict().items()}, stats, cfg, spec, "Normal",
        {"config_index": 2, "lr": 1e-3, "dropout": 0.5, "epochs": 3},
    )
    path = tmp_path / "crnn.pt"

    cnn_model.save_checkpoint(path, payload)
    loaded = cnn_model.load_checkpoint(path)
    restored = cnn_model.model_from_checkpoint(loaded)

    assert isinstance(restored, crnn_arch.CopdCRNN)
    assert loaded["architecture"]["class"] == crnn_arch.ARCHITECTURE_NAME
    assert loaded["label_mapping"] == {"Normal": 0, "COPD": 1}
    x = torch.randn(2, 1, 64, 309)
    with torch.no_grad():
        logits, weights = model(x, return_attention=True)
        restored_logits, restored_weights = restored(x, return_attention=True)
    assert torch.equal(logits, restored_logits)
    assert torch.equal(weights, restored_weights)


def test_shared_fold_protocol_runs_end_to_end_with_the_crnn(cfg):
    condition = _synthetic_condition()
    tiny = _tiny_cfg(cfg)

    result = cnn_model.run_cnn_fold(
        0, condition, TRAIN_PATIENTS, VAL_PATIENTS, TEST_PATIENTS, tiny,
        cnn_model.search_configurations(tiny), "Control", torch.device("cpu"), num_workers=0,
    )

    assert sorted(result.patient_predictions["patient_uid"]) == ["N5", "P5"]   # solo test
    assert set(result.segment_predictions["patient_uid"]) == {"N5", "P5"}
    assert result.normalization_search.n_segments == 16                         # solo train
    assert result.normalization_refit.n_segments == 20                          # train + validation
    assert result.segment_predictions["score"].between(0.0, 1.0).all()
    assert len(result.refit_history) == result.selected["best_epoch"]
    restored = cnn_model.build_model(tiny, dropout=result.selected["dropout"])
    assert isinstance(restored, crnn_arch.CopdCRNN)
    restored.load_state_dict(result.model_state)


def test_fingerprint_sections_and_incompatible_resume(tmp_path, cfg, cnn_cfg):
    assert fingerprint_sections(cfg) == CRNN_FINGERPRINT_SECTIONS
    assert fingerprint_sections(cnn_cfg) == CNN_FINGERPRINT_SECTIONS
    assert "crnn" in CRNN_FINGERPRINT_SECTIONS and "cnn" not in CRNN_FINGERPRINT_SECTIONS

    dataset_dir = tmp_path / "data" / "TOY"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "segments.csv").write_text("segment_id\nS0\n", encoding="utf-8")
    np.save(dataset_dir / "segments_no_dn.npy", np.zeros((1, 4), dtype=np.float32))
    specs = [dmod.ConditionSpec("TOY", "main_no_dn", "no_dn", False)]
    run_root = tmp_path / "run"
    run_root.mkdir()
    extra = {"model": "crnn", "architecture": cnn_model.architecture_description(cfg), "smoke_test": False}

    def fingerprint(config, extra_values):
        return rexp.build_run_fingerprint(
            config, tmp_path / "data", specs, "TOY", "main",
            sections=fingerprint_sections(config), extra=extra_values,
        )

    rexp.write_run_fingerprint(run_root, fingerprint(cfg, extra))
    rexp.verify_run_fingerprint(run_root, fingerprint(cfg, copy.deepcopy(extra)))

    changed = copy.deepcopy(cfg)
    changed["crnn"]["gru_dropout"] = 0.1
    with pytest.raises(RuntimeError):
        rexp.verify_run_fingerprint(run_root, fingerprint(changed, extra))

    cnn_extra = {"model": "cnn", "architecture": cnn_model.architecture_description(cnn_cfg), "smoke_test": False}
    with pytest.raises(RuntimeError):
        rexp.verify_run_fingerprint(run_root, fingerprint(cfg, cnn_extra))


def test_run_crnn_refuses_before_touching_anything(cfg, cnn_cfg):
    assert run_crnn(SimpleNamespace(force_features=False), cnn_cfg) == 2  # TOML de otra arquitectura
    assert run_crnn(SimpleNamespace(force_features=True), cfg) == 2       # no sobrescribe la cache compartida
