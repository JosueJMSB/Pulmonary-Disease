"""Los 6 TOML del protocolo fold-aware (v2) cargan, declaran [folds]/
[final_model], y las secciones compartidas de CRNN coinciden con su CNN de
referencia. No entrena ni toca datos reales."""

import pytest

from .. import data as dmod
from .. import run_experiment as rexp

V1_CONFIGS_DIR = dmod.OOF_V1_CONFIGS_DIR
V2_CONFIGS_DIR = dmod.FOLD_AWARE_V2_CONFIGS_DIR

V2_CONFIGS = (
    "svm_rbf_v2.toml", "cnn_v2.toml", "crnn_v2.toml",
    "svm_rbf_combined_v2.toml", "cnn_combined_v2.toml", "crnn_combined_v2.toml",
)


@pytest.mark.parametrize("name", V2_CONFIGS)
def test_v2_config_loads_and_declares_folded_protocol(name):
    cfg = dmod.load_config(V2_CONFIGS_DIR / name)
    assert rexp.is_folded_protocol(cfg)
    assert cfg["final_model"]["enabled"] is False


@pytest.mark.parametrize("name", ("svm_rbf.toml", "cnn.toml", "crnn.toml", "svm_rbf_combined.toml"))
def test_original_configs_are_not_folded_protocol(name):
    cfg = dmod.load_config(V1_CONFIGS_DIR / name)
    assert not rexp.is_folded_protocol(cfg)


@pytest.mark.parametrize("name", V2_CONFIGS)
def test_v2_config_patient_folds_paths_resolve_under_repo_root(name):
    cfg = dmod.load_config(V2_CONFIGS_DIR / name)
    csv_path, manifest_path = rexp._patient_folds_paths(cfg)
    assert csv_path == dmod.REPO_ROOT / "modeling" / "data" / "patient_folds.csv"
    assert manifest_path == dmod.REPO_ROOT / "modeling" / "data" / "patient_folds_manifest.json"


def test_validate_final_model_disabled_accepts_v2_configs():
    for name in V2_CONFIGS:
        cfg = dmod.load_config(V2_CONFIGS_DIR / name)
        rexp._validate_final_model_disabled(cfg)  # no debe lanzar


def test_validate_final_model_disabled_rejects_missing_or_true():
    original = dmod.load_config(V1_CONFIGS_DIR / "svm_rbf.toml")
    with pytest.raises(RuntimeError, match="final_model"):
        rexp._validate_final_model_disabled(original)

    cfg = dmod.load_config(V2_CONFIGS_DIR / "svm_rbf_v2.toml")
    cfg["final_model"]["enabled"] = True
    with pytest.raises(RuntimeError, match="final_model"):
        rexp._validate_final_model_disabled(cfg)


@pytest.mark.parametrize("name", ("svm_rbf_v2.toml", "svm_rbf_combined_v2.toml"))
def test_svm_v2_configs_have_no_augmentation_ablation(name):
    cfg = dmod.load_config(V2_CONFIGS_DIR / name)
    assert set(dmod.experiment_names(cfg)) == {"main", "denoising_ablation"}


@pytest.mark.parametrize("name", ("cnn_v2.toml", "crnn_v2.toml", "cnn_combined_v2.toml", "crnn_combined_v2.toml"))
def test_cnn_and_crnn_v2_configs_have_all_three_experiments(name):
    cfg = dmod.load_config(V2_CONFIGS_DIR / name)
    assert set(dmod.experiment_names(cfg)) == {"main", "denoising_ablation", "augmentation_ablation"}


def test_v2_individual_configs_cover_icbhi_and_fraiwan():
    for name in ("svm_rbf_v2.toml", "cnn_v2.toml", "crnn_v2.toml"):
        cfg = dmod.load_config(V2_CONFIGS_DIR / name)
        assert set(dmod.dataset_names(cfg)) == {"ICBHI", "FRAIWAN_Extended"}


def test_v2_combined_configs_cover_only_combined():
    for name in ("svm_rbf_combined_v2.toml", "cnn_combined_v2.toml", "crnn_combined_v2.toml"):
        cfg = dmod.load_config(V2_CONFIGS_DIR / name)
        assert dmod.dataset_names(cfg) == ["COMBINED"]


def test_v2_configs_never_set_dn_reliable_only_true():
    for name in V2_CONFIGS:
        cfg = dmod.load_config(V2_CONFIGS_DIR / name)
        for experiment in cfg["experiments"].values():
            for dataset_entry in experiment.values():
                if not isinstance(dataset_entry, (list, dict)):
                    continue
                entries = dataset_entry if isinstance(dataset_entry, list) else [dataset_entry]
                for entry in entries:
                    assert entry.get("dn_reliable_only", False) is False


def test_crnn_v2_config_consistency_with_cnn_v2():
    torch = pytest.importorskip("torch")  # noqa: F841
    from ..cnn_experiment import config_consistency_checks, SHARED_PROTOCOL_SECTIONS

    crnn_cfg = dmod.load_config(V2_CONFIGS_DIR / "crnn_v2.toml")
    cnn_cfg = dmod.load_config(V2_CONFIGS_DIR / "cnn_v2.toml")
    for section in SHARED_PROTOCOL_SECTIONS:
        assert crnn_cfg.get(section) == cnn_cfg.get(section), section
    assert all(check["ok"] for check in config_consistency_checks(crnn_cfg))


def test_crnn_combined_v2_config_consistency_with_cnn_combined_v2():
    torch = pytest.importorskip("torch")  # noqa: F841
    from ..cnn_experiment import config_consistency_checks, SHARED_PROTOCOL_SECTIONS

    crnn_cfg = dmod.load_config(V2_CONFIGS_DIR / "crnn_combined_v2.toml")
    cnn_cfg = dmod.load_config(V2_CONFIGS_DIR / "cnn_combined_v2.toml")
    for section in SHARED_PROTOCOL_SECTIONS:
        assert crnn_cfg.get(section) == cnn_cfg.get(section), section
    assert all(check["ok"] for check in config_consistency_checks(crnn_cfg))
    assert dmod.reference_cnn_config_path(crnn_cfg) == V2_CONFIGS_DIR / "cnn_combined_v2.toml"
