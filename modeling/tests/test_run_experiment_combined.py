"""Compatibilidad de COMBINED con la CLI de run_experiment.py.

Cubre el rechazo con mensaje claro de un ``--dataset`` que no esta definido
en el TOML seleccionado (en cualquier direccion: COMBINED contra un TOML de
un solo dataset, o un dataset individual contra un TOML combinado), que
``--dataset all`` solo usa los datasets del TOML elegido, y las entradas de
COMBINED en las tablas de comparaciones emparejadas (denoising/augmentation).
No entrena ni toca datos reales: el rechazo ocurre antes de resolver
data-root o abrir ninguna ejecucion.
"""

import pytest

from .. import data as dmod
from .. import run_experiment as rexp

CONFIGS_DIR = dmod.OOF_V1_CONFIGS_DIR


def test_combined_rejected_on_single_dataset_svm_config(capsys):
    rc = rexp.main(["--model", "svm_rbf", "--dataset", "COMBINED", "--experiment", "main"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "COMBINED" in captured.err
    assert "no esta definido" in captured.err


def test_icbhi_rejected_on_combined_only_svm_config(capsys):
    rc = rexp.main([
        "--model", "svm_rbf", "--dataset", "ICBHI", "--experiment", "main",
        "--config", str(CONFIGS_DIR / "svm_rbf_combined.toml"),
    ])
    captured = capsys.readouterr()
    assert rc == 2
    assert "ICBHI" in captured.err
    assert "no esta definido" in captured.err


def test_dataset_names_and_all_resolve_only_to_combined():
    cfg = dmod.load_config(CONFIGS_DIR / "svm_rbf_combined.toml")
    assert dmod.dataset_names(cfg) == ["COMBINED"]
    assert rexp.resolve_datasets(cfg, "COMBINED") == ["COMBINED"]
    assert rexp.resolve_datasets(cfg, "all") == ["COMBINED"]


def test_ablation_pairs_include_combined_denoising_pair():
    assert rexp.ABLATION_PAIRS["COMBINED"] == ("no_dn_reliable", "dn_reliable")


def test_cnn_paired_comparisons_include_combined_denoising_and_augmentation():
    # cnn_experiment importa torch al nivel de modulo; se omite si no esta instalado.
    pytest.importorskip("torch")
    from ..cnn_experiment import PAIRED_COMPARISONS

    pairs = dict((label, (left, right)) for left, right, label in PAIRED_COMPARISONS["COMBINED"])
    assert pairs["denoising"] == ("no_dn_reliable", "dn_reliable")
    assert pairs["augmentation"] == ("main_no_dn", "main_no_dn_aug")


def test_combined_configs_declare_report_by_source_and_stratify_by_dataset():
    for name in ("svm_rbf_combined.toml", "cnn_combined.toml", "crnn_combined.toml"):
        cfg = dmod.load_config(CONFIGS_DIR / name)
        assert cfg["splits"]["stratify_by_dataset"] is True
        assert cfg["evaluation"]["report_by_source"] is True
        assert dmod.dataset_names(cfg) == ["COMBINED"]
        assert cfg["datasets"]["COMBINED"]["negative_label_name"] == "Control"
