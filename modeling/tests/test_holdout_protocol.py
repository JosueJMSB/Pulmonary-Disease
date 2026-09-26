"""Nucleo compartido del protocolo holdout-v3 (``holdout_cv.py``), sin PyTorch:
configuraciones, candidatos deterministas, seleccion global sin mezclar
ganadores por fold, unidades atomicas con reanudacion, artefactos, analisis por
fuente/dispositivo y bloqueo de la prueba externa. Datos sinteticos; no entrena
ningun modelo real.
"""

import copy
import dataclasses
import json
import logging

import numpy as np
import pandas as pd
import pytest

from .. import artifacts as art
from .. import data as dmod
from .. import evaluation as ev
from .. import holdout_cv as hcv
from .. import run_experiment as rexp
from .. import splits as sp

V3_DIR = dmod.HOLDOUT_FINAL_CONFIGS_DIR
V3_CONFIGS = (
    "svm_rbf_v3.toml", "svm_rbf_combined_v3.toml", "cnn_v3.toml",
    "cnn_combined_v3.toml", "crnn_v3.toml", "crnn_combined_v3.toml",
)
NN_CONFIGS = ("cnn_v3.toml", "cnn_combined_v3.toml", "crnn_v3.toml", "crnn_combined_v3.toml")
SVM_CONFIGS = ("svm_rbf_v3.toml", "svm_rbf_combined_v3.toml")

# Secciones que una CRNN debe copiar de su CNN de referencia (las de
# cnn_experiment.SHARED_PROTOCOL_SECTIONS mas las propias de v3; se listan aqui
# para no importar PyTorch).
SHARED_SECTIONS = (
    "protocol", "acoustic", "logmel", "weights", "normalization", "training", "search", "selection",
    "metrics", "evaluation", "augmentation", "seeds", "determinism", "datasets", "experiments",
    "holdout", "outer_test", "final_model", "paths",
)


def _load(name):
    return dmod.load_config(V3_DIR / name)


# ---------------------------------------------------------------------------
# Configuraciones
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", V3_CONFIGS)
def test_v3_config_declares_the_protocol_and_blocks_test_and_final_model(name):
    cfg = _load(name)
    assert dmod.is_holdout_protocol(cfg)
    assert not rexp.is_folded_protocol(cfg)          # no activa el protocolo v2
    assert cfg["outer_test"]["enabled"] is False
    assert cfg["final_model"]["enabled"] is False
    assert cfg["holdout"]["n_splits"] == 5
    hcv.validate_holdout_config(cfg)                 # no debe lanzar


@pytest.mark.parametrize("name", ("svm_rbf.toml", "cnn.toml", "crnn.toml", "svm_rbf_combined.toml"))
def test_original_configs_are_not_holdout_protocol(name):
    assert not dmod.is_holdout_protocol(dmod.load_config(dmod.OOF_V1_CONFIGS_DIR / name))


@pytest.mark.parametrize("name", ("svm_rbf_v2.toml", "cnn_v2.toml", "crnn_combined_v2.toml"))
def test_fold_aware_v2_configs_are_not_holdout_protocol(name):
    cfg = dmod.load_config(dmod.FOLD_AWARE_V2_CONFIGS_DIR / name)
    assert not dmod.is_holdout_protocol(cfg)
    assert rexp.is_folded_protocol(cfg)


def test_all_six_configs_use_the_same_split_files_and_seed():
    """Los mismos pacientes y folds para SVM, CNN y CRNN, y en los tres datasets."""
    holdouts = {name: _load(name)["holdout"] for name in V3_CONFIGS}
    assert len({json.dumps(h, sort_keys=True) for h in holdouts.values()}) == 1
    csv_path, manifest_path = hcv.holdout_split_paths(_load("svm_rbf_v3.toml"))
    assert csv_path == dmod.REPO_ROOT / "modeling" / "data" / "holdout_splits.csv"
    assert manifest_path == dmod.REPO_ROOT / "modeling" / "data" / "holdout_splits_manifest.json"


def test_svm_configs_have_no_augmentation_and_only_no_dn_and_dn():
    for name in SVM_CONFIGS:
        cfg = _load(name)
        assert set(dmod.experiment_names(cfg)) == {"main", "denoising_ablation"}
        conditions = {s.condition for d in dmod.dataset_names(cfg) for s in dmod.all_condition_specs(cfg, d)}
        assert conditions == {"no_dn", "dn"}


def test_network_configs_have_the_three_conditions_and_aug_reuses_no_dn():
    for name in NN_CONFIGS:
        cfg = _load(name)
        assert set(dmod.experiment_names(cfg)) == {"main", "denoising_ablation", "augmentation_ablation"}
        for dataset in dmod.dataset_names(cfg):
            by_condition = {s.condition: s for s in dmod.all_condition_specs(cfg, dataset)}
            assert set(by_condition) == {"no_dn", "dn", "no_dn_aug"}
            assert by_condition["no_dn_aug"].augment is True
            assert by_condition["no_dn_aug"].hyperparameters_from == "no_dn"
            assert by_condition["no_dn_aug"].branch == "no_dn"
            assert not any(s.dn_reliable_only for s in by_condition.values())
            assert by_condition["no_dn"].hyperparameters_from is None and not by_condition["no_dn"].augment


def test_datasets_of_each_config_family():
    for name in ("svm_rbf_v3.toml", "cnn_v3.toml", "crnn_v3.toml"):
        assert dmod.dataset_names(_load(name)) == ["ICBHI", "FRAIWAN_Extended"]
    for name in ("svm_rbf_combined_v3.toml", "cnn_combined_v3.toml", "crnn_combined_v3.toml"):
        assert dmod.dataset_names(_load(name)) == ["COMBINED"]


@pytest.mark.parametrize(
    "cnn_name, crnn_name",
    (("cnn_v3.toml", "crnn_v3.toml"), ("cnn_combined_v3.toml", "crnn_combined_v3.toml")),
)
def test_crnn_copies_the_shared_sections_of_its_cnn(cnn_name, crnn_name):
    cnn_cfg, crnn_cfg = _load(cnn_name), _load(crnn_name)
    for section in SHARED_SECTIONS:
        assert crnn_cfg.get(section) == cnn_cfg.get(section), section
    assert dmod.model_architecture(crnn_cfg) == "crnn"
    assert dmod.reference_cnn_config_path(crnn_cfg) == V3_DIR / cnn_name


def test_reference_cnn_config_resolves_inside_each_protocol_folder():
    assert dmod.reference_cnn_config_path(dmod.load_config(dmod.OOF_V1_CONFIGS_DIR / "crnn_combined.toml")) == (
        dmod.OOF_V1_CONFIGS_DIR / "cnn_combined.toml"
    )
    assert dmod.reference_cnn_config_path(dmod.load_config(dmod.FOLD_AWARE_V2_CONFIGS_DIR / "crnn_v2.toml")) == (
        dmod.FOLD_AWARE_V2_CONFIGS_DIR / "cnn_v2.toml"
    )
    assert dmod.reference_cnn_config_path(dmod.load_config(dmod.OOF_V1_CONFIGS_DIR / "crnn.toml")) == dmod.CNN_CONFIG_PATH


def test_validate_rejects_outer_test_or_final_model_enabled_or_missing():
    for section in ("outer_test", "final_model"):
        cfg = _load("svm_rbf_v3.toml")
        cfg[section]["enabled"] = True
        with pytest.raises(RuntimeError, match=section):
            hcv.validate_holdout_config(cfg)
        cfg = _load("svm_rbf_v3.toml")
        del cfg[section]
        with pytest.raises(RuntimeError, match=section):
            hcv.validate_holdout_config(cfg)


def test_validate_rejects_other_selection_criteria_folds_and_folds_section():
    cfg = _load("svm_rbf_v3.toml")
    cfg["selection"]["criteria"] = list(reversed(cfg["selection"]["criteria"]))
    with pytest.raises(RuntimeError, match="criteria"):
        hcv.validate_holdout_config(cfg)

    cfg = _load("svm_rbf_v3.toml")
    cfg["holdout"]["n_splits"] = 3
    with pytest.raises(RuntimeError, match="n_splits"):
        hcv.validate_holdout_config(cfg)

    cfg = _load("svm_rbf_v3.toml")
    cfg["folds"] = {"patient_folds_csv": "x.csv"}
    with pytest.raises(RuntimeError, match="folds"):
        hcv.validate_holdout_config(cfg)

    with pytest.raises(RuntimeError, match="protocol"):
        hcv.validate_holdout_config(dmod.load_config(dmod.FOLD_AWARE_V2_CONFIGS_DIR / "svm_rbf_v2.toml"))


def test_fold_reference_maps_to_the_cv_subdirectory_without_changing_older_layouts(tmp_path):
    assert dmod.dataset_root(tmp_path, "ICBHI") == tmp_path / "ICBHI"
    assert dmod.dataset_root(tmp_path, "ICBHI", 2) == tmp_path / "ICBHI" / "fold_02"
    assert dmod.dataset_root(tmp_path, "ICBHI", hcv.cv_fold_ref(2)) == tmp_path / "ICBHI" / "cv" / "fold_02"
    assert dmod.feature_cache_dir(tmp_path, "ICBHI", "no_dn", hcv.cv_fold_ref(4)) == (
        tmp_path / "ICBHI" / "cv" / "fold_04" / "no_dn"
    )


# ---------------------------------------------------------------------------
# Candidatos deterministas
# ---------------------------------------------------------------------------

def test_svm_candidates_are_the_30_combinations_in_a_stable_order():
    for name in SVM_CONFIGS:
        candidates = hcv.svm_candidates(_load(name))
        assert len(candidates) == 30
        assert [c["config_index"] for c in candidates] == list(range(30))
        assert len({(c["C"], str(c["gamma"])) for c in candidates}) == 30
        assert [c["C"] for c in candidates[:5]] == [0.01] * 5          # C mayor, luego gamma
        assert [c["gamma"] for c in candidates[:5]] == ["scale", 0.0001, 0.001, 0.01, 0.1]
        assert sorted({c["C"] for c in candidates}) == [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def test_nn_candidates_are_20_unique_deterministic_combinations_of_the_space():
    cfg = _load("cnn_v3.toml")
    first, second = hcv.nn_candidates(cfg), hcv.nn_candidates(copy.deepcopy(cfg))
    assert first == second                                            # determinista
    assert len(first) == 20 and [c["config_index"] for c in first] == list(range(20))
    combos = [tuple(c[k] for k in hcv.NN_HYPERPARAMETER_KEYS) for c in first]
    assert len(set(combos)) == 20                                     # sin reemplazo
    search = cfg["search"]
    for c in first:
        assert c["lr"] in search["learning_rate"] and c["dropout"] in search["dropout"]
        assert c["weight_decay"] in search["weight_decay"] and c["batch_size"] in search["batch_size"]


def test_nn_candidates_depend_on_the_seed():
    cfg = _load("cnn_v3.toml")
    other = copy.deepcopy(cfg)
    other["search"]["random_state"] = 1
    assert hcv.nn_candidates(cfg) != hcv.nn_candidates(other)


def test_cnn_and_crnn_receive_the_same_candidate_list_in_every_config_family():
    reference = hcv.nn_candidates(_load("cnn_v3.toml"))
    for name in NN_CONFIGS:
        assert hcv.nn_candidates(_load(name)) == reference


def test_nn_candidates_reject_an_oversized_budget():
    cfg = _load("cnn_v3.toml")
    cfg["search"]["n_configurations"] = 109
    with pytest.raises(ValueError, match="n_configurations"):
        hcv.nn_candidates(cfg)


def test_median_best_epoch_rounds_half_up_and_ignores_missing():
    assert hcv.median_best_epoch([10, 20, 30, 40, 50]) == 30
    assert hcv.median_best_epoch([10, 21]) == 16            # 15.5 -> 16
    assert hcv.median_best_epoch([None, None]) is None
    assert hcv.median_best_epoch([1]) == 1


# ---------------------------------------------------------------------------
# Resumen por configuracion y seleccion global
# ---------------------------------------------------------------------------

def _search(per_config: dict, expected_folds: int = 5) -> pd.DataFrame:
    """``per_config``: config_index -> lista (uno por fold) de dicts con las
    metricas de la unidad; ``None`` marca un fold FALLIDO."""
    rows = []
    for index, folds in per_config.items():
        for fold, metrics in enumerate(folds):
            row = {
                "model": "svm_rbf", "dataset": "TOY", "condition": "no_dn", "config_index": index, "fold": fold,
                "C": float(index + 1), "gamma": "scale", "best_epoch": np.nan,
            }
            if metrics is None:
                row.update({"status": hcv.STATUS_UNIT_FAILED, **{c: np.nan for c in hcv.UNIT_METRIC_COLUMNS}})
            else:
                base = {c: np.nan for c in hcv.UNIT_METRIC_COLUMNS}
                base.update({"macro_f1": 0.5, "min_class_recall": 0.5, "recall_copd": 0.5, "recall_negative": 0.5})
                base.update(metrics)
                row.update({"status": hcv.STATUS_UNIT_COMPLETED, **base})
            rows.append(row)
    return pd.DataFrame(rows)


def _ba(*values, **extra):
    return [{"balanced_accuracy": v, **extra} for v in values]


def _summary(per_config):
    return hcv.config_summary_frame(_search(per_config), 5, hcv.SVM_HYPERPARAMETER_KEYS)


def test_config_summary_has_mean_and_sample_std_across_folds():
    search = _search({0: _ba(0.6, 0.8)})
    summary = hcv.config_summary_frame(search, 2, hcv.SVM_HYPERPARAMETER_KEYS)
    row = summary.iloc[0]
    assert row["balanced_accuracy_mean"] == pytest.approx(0.7)
    assert row["balanced_accuracy_std"] == pytest.approx(np.sqrt(0.02))   # ddof=1
    assert bool(row["complete"]) and row["n_folds_completed"] == 2


def test_incomplete_or_failed_configuration_can_never_be_selected():
    # La 0 tiene la mejor media en 4 folds, pero falla el quinto.
    summary = _summary({0: _ba(0.99, 0.99, 0.99, 0.99) + [None], 1: _ba(0.70, 0.70, 0.70, 0.70, 0.70)})
    assert summary.set_index("config_index")["complete"].to_dict() == {0: False, 1: True}
    best, info = hcv.select_best_configuration(summary)
    assert int(best["config_index"]) == 1
    assert info["n_incomplete_candidates"] == 1 and info["n_complete_candidates"] == 1


def test_selection_is_global_and_never_mixes_per_fold_winners():
    a = _ba(0.90, 0.90, 0.90, 0.50, 0.50)     # gana 3 de 5 folds, media 0.74
    b = _ba(0.70, 0.70, 0.70, 0.95, 0.95)     # gana 2 de 5 folds, media 0.80
    best, _ = hcv.select_best_configuration(_summary({0: a, 1: b}))
    assert int(best["config_index"]) == 1


def test_selection_without_any_complete_configuration_raises():
    with pytest.raises(hcv.SelectionError):
        hcv.select_best_configuration(_summary({0: _ba(0.9, 0.9, 0.9, 0.9) + [None]}))


def test_tie_break_order_macro_f1_then_min_recall_then_std_then_config_order():
    flat = _ba(0.75, 0.75, 0.75, 0.75, 0.75)

    def with_extra(**extra):
        return [{**m, **extra} for m in flat]

    # 1) balanced accuracy media empatada -> gana el mayor macro F1.
    best, info = hcv.select_best_configuration(
        _summary({0: with_extra(macro_f1=0.60), 1: with_extra(macro_f1=0.70)}))
    assert int(best["config_index"]) == 1 and info["decided_by"] == "macro_f1_mean"
    assert info["criteria_applied"] == ["balanced_accuracy_mean", "macro_f1_mean"]

    # 2) ademas macro F1 empatado -> gana el mayor minimo recall entre clases.
    best, info = hcv.select_best_configuration(
        _summary({0: with_extra(min_class_recall=0.40), 1: with_extra(min_class_recall=0.60)}))
    assert int(best["config_index"]) == 1 and info["decided_by"] == "min_class_recall_mean"

    # 3) medias iguales (suma 3.5) -> gana la menor desviacion estandar de balanced accuracy.
    volatile = _ba(0.5, 1.0, 0.5, 1.0, 0.5)
    steady = _ba(0.75, 0.75, 0.5, 0.75, 0.75)
    best, info = hcv.select_best_configuration(_summary({0: volatile, 1: steady}))
    assert int(best["config_index"]) == 1 and info["decided_by"] == "balanced_accuracy_std"

    # 4) todo empatado -> el orden determinista de la configuracion (indice menor).
    best, info = hcv.select_best_configuration(_summary({3: flat, 1: flat, 2: flat}))
    assert int(best["config_index"]) == 1 and info["decided_by"] == "config_order"
    assert info["criteria_applied"] == list(hcv.SELECTION_CRITERIA)


def test_reused_configuration_requires_that_it_completed_its_folds():
    summary = _summary({0: _ba(0.9, 0.9, 0.9, 0.9, 0.9), 1: _ba(0.5, 0.5, 0.5, 0.5) + [None]})
    best, info = hcv.select_reused_configuration(summary, 0, "no_dn")
    assert int(best["config_index"]) == 0 and info["decided_by"] == "reused_from:no_dn"
    with pytest.raises(hcv.SelectionError):
        hcv.select_reused_configuration(summary, 1, "no_dn")


# ---------------------------------------------------------------------------
# Puntajes de validacion y comprobaciones de fuga contra el split
# ---------------------------------------------------------------------------

def _toy_segments(n_pos=20, n_neg=15):
    rows = []
    for label, n in ((1, n_pos), (0, n_neg)):
        for i in range(n):
            uid = f"TOY_{label}_{i:03d}"
            for k in range(2):
                rows.append({
                    "patient_uid": uid, "audio_id": f"{uid}_rec0", "segment_id": f"{uid}_rec0_{k:02d}",
                    "target_label": label, "calibration_patient": False, "dataset": "TOY",
                })
    return pd.DataFrame(rows)


def _toy_split(n_pos=20, n_neg=15):
    table = sp.assign_holdout_split(sp.build_patient_table(_toy_segments(n_pos, n_neg)), n_splits=5, random_state=20260914)
    return sp.HoldoutSplit(dataset_scope="TOY", patient_table=table, n_splits=5)


def _fold_segments(split, fold_id, include_blocked=False, drop_one=False, wrong_role=False):
    train, val = split.cv_split(fold_id)
    label_by = dict(zip(split.patient_table["patient_uid"], split.patient_table["target_label"]))
    rows = []
    patients = [(p, "train") for p in train] + [(p, "validation") for p in val]
    if include_blocked:
        patients.append((next(iter(split.blocked_test_patients())), "train"))
    if drop_one:
        patients = patients[1:]
    if wrong_role:
        patients[0] = (patients[0][0], "validation")
    for uid, role in patients:
        rows.append({
            "segment_id": f"{uid}_s0", "audio_id": f"{uid}_rec0", "patient_uid": uid,
            "target_label": label_by[uid], "role": role, "dataset": "TOY", "device": "Meditron",
        })
    return pd.DataFrame(rows)


def test_verify_fold_segments_accepts_exactly_the_split_patients():
    split = _toy_split()
    hcv.verify_fold_segments_against_split(_fold_segments(split, 0), split, 0)


def test_verify_fold_segments_rejects_a_blocked_test_patient():
    split = _toy_split()
    with pytest.raises(RuntimeError, match="prueba externa"):
        hcv.verify_fold_segments_against_split(_fold_segments(split, 0, include_blocked=True), split, 0)


def test_verify_fold_segments_rejects_missing_patients_and_wrong_roles():
    split = _toy_split()
    with pytest.raises(RuntimeError, match="no coinciden"):
        hcv.verify_fold_segments_against_split(_fold_segments(split, 1, drop_one=True), split, 1)
    with pytest.raises(RuntimeError, match="rol distinto"):
        hcv.verify_fold_segments_against_split(_fold_segments(split, 1, wrong_role=True), split, 1)


def test_verify_fold_segments_rejects_a_test_role():
    split = _toy_split()
    segments = _fold_segments(split, 2)
    segments.loc[0, "role"] = "test"
    with pytest.raises(RuntimeError, match="roles no permitidos"):
        hcv.verify_fold_segments_against_split(segments, split, 2)


def test_build_val_scores_validates_shape_and_finiteness():
    val = _fold_segments(_toy_split(), 0).query("role == 'validation'")
    scores = hcv.build_val_scores(val, np.linspace(0, 1, len(val)))
    assert list(scores.columns) == hcv.SCORE_COLUMNS
    assert scores["source_dataset"].eq("TOY").all() and scores["device"].eq("Meditron").all()
    with pytest.raises(ValueError):
        hcv.build_val_scores(val, np.zeros(len(val) + 1))
    with pytest.raises(FloatingPointError):
        hcv.build_val_scores(val, np.full(len(val), np.nan))


# ---------------------------------------------------------------------------
# Analisis por fuente y por dispositivo
# ---------------------------------------------------------------------------

def _predictions(devices_by_patient: dict, sources_by_patient: dict, labels: dict, scores: dict) -> pd.DataFrame:
    rows = []
    for uid, device in devices_by_patient.items():
        for k in range(2):
            rows.append({
                "segment_id": f"{uid}_{k}", "audio_id": f"{uid}_rec0", "patient_uid": uid,
                "target_label": labels[uid], "source_dataset": sources_by_patient[uid], "device": device,
                "score": scores[uid], "fold": 0,
            })
    return pd.DataFrame(rows)


def test_metrics_by_device_are_nan_when_a_device_has_a_single_class_but_report_counts_and_recalls():
    devices = {"P1": "Meditron", "P2": "Meditron", "P3": "Litt3200", "P4": "Litt3200", "N1": "Meditron", "N2": "AKG"}
    labels = {"P1": 1, "P2": 1, "P3": 1, "P4": 1, "N1": 0, "N2": 0}
    scores = {"P1": 0.9, "P2": 0.2, "P3": 0.8, "P4": 0.7, "N1": 0.1, "N2": 0.4}
    segments = _predictions(devices, {u: "X" for u in devices}, labels, scores)

    patient_device = ev.aggregate_patient_device(segments)
    metrics = ev.compute_metrics_by_device(patient_device, "Healthy", threshold=0.5).set_index("device")

    litt = metrics.loc["Litt3200"]                     # solo COPD
    assert litt["n_copd"] == 2 and litt["n_healthy"] == 0 and not litt["both_classes"]
    assert litt["recall_copd"] == 1.0 and np.isnan(litt["recall_healthy"])
    assert np.isnan(litt["auroc"]) and np.isnan(litt["auprc_copd"])
    assert np.isnan(litt["balanced_accuracy"]) and np.isnan(litt["macro_f1"])

    akg = metrics.loc["AKG"]                           # solo Healthy
    assert akg["n_copd"] == 0 and akg["recall_healthy"] == 1.0 and np.isnan(akg["recall_copd"])

    meditron = metrics.loc["Meditron"]                 # ambas clases
    assert meditron["both_classes"] and not np.isnan(meditron["auroc"])
    assert meditron["recall_copd"] == 0.5 and meditron["recall_healthy"] == 1.0
    assert meditron["balanced_accuracy"] == pytest.approx(0.75)


def test_patient_device_aggregation_counts_a_patient_once_per_device():
    rows = []
    for device, score in (("A", 0.2), ("B", 0.8)):     # un paciente con dos dispositivos
        rows.append({"segment_id": f"s_{device}", "audio_id": f"a_{device}", "patient_uid": "P1",
                     "target_label": 1, "device": device, "score": score})
    aggregated = ev.aggregate_patient_device(pd.DataFrame(rows))
    assert sorted(aggregated["device"]) == ["A", "B"] and (aggregated["patient_uid"] == "P1").all()


def test_group_artifacts_write_device_files_always_and_source_files_only_with_two_sources(tmp_path):
    labels = {"P1": 1, "P2": 1, "P3": 0, "P4": 0}
    scores = {"P1": 0.9, "P2": 0.6, "P3": 0.1, "P4": 0.3}
    devices = {u: "Meditron" for u in labels}

    two = _predictions(devices, {"P1": "ICBHI", "P2": "FRAIWAN", "P3": "ICBHI", "P4": "FRAIWAN"}, labels, scores)
    _, patients = hcv.aggregate_winner_predictions(two)
    out_two = tmp_path / "two"
    out_two.mkdir()
    hcv.write_group_artifacts(out_two, two, patients, "Control", 0.5)
    for name in ("cv_metrics_by_device.csv", "cv_counts_by_device.csv", "cv_confusion_matrix_by_device.csv",
                 "cv_metrics_by_source.csv", "cv_classification_report_by_source.csv",
                 "cv_confusion_matrix_by_source.csv"):
        assert (out_two / name).is_file(), name

    one = _predictions(devices, {u: "ICBHI" for u in labels}, labels, scores)
    _, patients_one = hcv.aggregate_winner_predictions(one)
    out_one = tmp_path / "one"
    out_one.mkdir()
    hcv.write_group_artifacts(out_one, one, patients_one, "Control", 0.5)
    assert (out_one / "cv_metrics_by_device.csv").is_file()
    assert not (out_one / "cv_metrics_by_source.csv").exists()


def test_aggregate_winner_predictions_gives_one_prediction_per_patient_and_rejects_two_folds():
    labels = {"P1": 1, "N1": 0}
    segments = _predictions({"P1": "A", "N1": "A"}, {"P1": "X", "N1": "X"}, labels, {"P1": 0.9, "N1": 0.1})
    recordings, patients = hcv.aggregate_winner_predictions(segments)
    assert sorted(patients["patient_uid"]) == ["N1", "P1"] and len(recordings) == 2

    twice = pd.concat([segments, segments.assign(fold=1)], ignore_index=True)
    with pytest.raises(RuntimeError, match="mas de un fold"):
        hcv.aggregate_winner_predictions(twice)


def test_curve_frames_have_matching_lengths():
    y_true = np.array([0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8, 0.9])
    roc, pr = hcv.curve_frames(y_true, y_score)
    assert list(roc.columns) == ["fpr", "tpr", "threshold"]
    assert list(pr.columns) == ["precision", "recall", "threshold"]
    assert roc["fpr"].between(0, 1).all() and pr["precision"].between(0, 1).all()


# ---------------------------------------------------------------------------
# Orquestacion: 5 folds x configuraciones, unidades atomicas y reanudacion
# ---------------------------------------------------------------------------

CONFIGS = [{"config_index": i, "C": float(i + 1), "gamma": "scale"} for i in range(5)]
SPEC = dmod.ConditionSpec(dataset="TOY", condition="no_dn", branch="no_dn", dn_reliable_only=False)
LOGGER = logging.getLogger("test_holdout_protocol")
MIN_CFG = {
    "protocol": "holdout_cv_v3", "svm": {"c_grid": [1, 10], "gamma_grid": ["scale", 0.1]},
    "figures": {"dpi": 50, "formats": ["png"]},          # figuras pequenas: las pruebas no necesitan 300 dpi
}
PROVENANCE = {"run_id": "RUN_X", "hashes": {"config_sha256": "abc"}}


def _val_frame(split, val_patients):
    label_by = dict(zip(split.patient_table["patient_uid"], split.patient_table["target_label"]))
    rows = []
    for i, uid in enumerate(sorted(val_patients)):
        for k in range(2):
            rows.append({
                "segment_id": f"{uid}_s{k}", "audio_id": f"{uid}_rec0", "patient_uid": uid,
                "target_label": label_by[uid], "dataset": "TOY", "device": "Meditron" if i % 2 == 0 else "AKG",
            })
    return pd.DataFrame(rows)


def _evaluator(split, calls, fail=None):
    """Configuracion i clasifica mal a |i - 2| pacientes de validation: la 2 es
    la mejor en los cinco folds."""
    def prepare_fold(fold_id, train_patients, val_patients):
        return {"fold": fold_id, "val": _val_frame(split, val_patients), "patients": sorted(val_patients)}

    def evaluate_fold(context, pending):
        for config in pending:
            calls.append((config["config_index"], context["fold"]))
            if fail is not None and fail(config, context["fold"]):
                yield config, hcv.UnitFailure(error="fallo simulado", error_type="RuntimeError")
                continue
            wrong = set(context["patients"][: abs(config["config_index"] - 2)])
            scores = []
            for row in context["val"].itertuples():
                good = 0.9 if row.target_label == 1 else 0.1
                scores.append(1.0 - good if row.patient_uid in wrong else good)
            yield config, hcv.UnitOutput(
                val_scores=hcv.build_val_scores(context["val"], scores),
                best_epoch=10 * (context["fold"] + 1), epochs_run=20, stopped_early=True, seconds=1.0,
            )

    return prepare_fold, evaluate_fold


def _run(tmp_path, split, calls, fail=None, fold_ids=(0, 1, 2, 3, 4)):
    prepare_fold, evaluate_fold = _evaluator(split, calls, fail)
    hcv.run_condition_cv(
        run_root=tmp_path, spec=SPEC, model_name="svm_rbf", split=split, configs=CONFIGS,
        hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, negative_label_name="Healthy", threshold=0.5,
        fold_ids=list(fold_ids), prepare_fold=prepare_fold, evaluate_fold=evaluate_fold,
        release_fold=None, logger=LOGGER,
    )


def _summarize(tmp_path, fold_ids=(0, 1, 2, 3, 4), reused=None):
    return hcv.summarize_condition(
        run_root=tmp_path, spec=SPEC, cfg_used=MIN_CFG, model_name="svm_rbf", configs=CONFIGS,
        hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, fold_ids=list(fold_ids), negative_label_name="Healthy",
        threshold=0.5, hyperparameter_source="search", reused_config_index=reused,
        provenance=PROVENANCE, logger=LOGGER,
    )


FIGURES = ("confusion_matrix", "roc", "pr", "metrics_by_fold")

ARTIFACTS = (
    "cv_search_results.csv", "cv_config_summary.csv", "best_hyperparameters.json", "cv_fold_metrics.csv",
    "cv_metrics_summary.csv", "cv_patient_predictions.csv", "cv_recording_predictions.csv",
    "cv_segment_predictions.csv", "cv_confusion_matrix.csv", "cv_classification_report.csv",
    "cv_roc_curve.csv", "cv_pr_curve.csv", "cv_metrics_by_device.csv", "cv_counts_by_device.csv",
    "cv_confusion_matrix_by_device.csv", "resolved_config.toml", "run_manifest.json", "status.json",
)


def test_every_configuration_is_evaluated_in_all_five_folds_and_one_global_winner_is_chosen(tmp_path):
    split = _toy_split()
    calls = []
    _run(tmp_path, split, calls)

    assert sorted(calls) == sorted((c, f) for c in range(5) for f in range(5))   # 25 unidades
    result = _summarize(tmp_path)
    cdir = art.condition_dir(tmp_path, "TOY", "no_dn")

    assert result["status"] == "COMPLETED" and result["ok"]
    for name in ARTIFACTS:
        assert (cdir / name).is_file(), name
    assert not (cdir / "final").exists()                                  # sin modelo definitivo
    assert not (cdir / "cv_metrics_by_source.csv").exists()               # una sola fuente
    for figure in FIGURES:                                                # tablas Y figuras automaticamente
        assert (cdir / "figures" / f"{figure}.png").is_file(), figure

    best = json.loads((cdir / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert best["config_index"] == 2 and best["hyperparameters"] == {"C": 3.0, "gamma": "scale"}
    assert best["hyperparameters_source"] == "search" and best["fold_ids"] == [0, 1, 2, 3, 4]
    assert best["best_epochs_by_fold"] == {"0": 10, "1": 20, "2": 30, "3": 40, "4": 50}
    assert best["median_best_epoch"] == 30
    assert best["selection"]["decided_by"] == "balanced_accuracy_mean"
    assert best["hashes"] == {"config_sha256": "abc"}
    assert best["metrics"]["fold_mean"]["balanced_accuracy"] == pytest.approx(1.0)

    search = pd.read_csv(cdir / "cv_search_results.csv")
    assert len(search) == 25 and set(search["status"]) == {"COMPLETED"}
    summary = pd.read_csv(cdir / "cv_config_summary.csv")
    assert summary["selected"].sum() == 1 and bool(summary.loc[summary["config_index"] == 2, "selected"].iloc[0])
    assert summary["complete"].all()


def test_only_development_patients_are_predicted_and_each_exactly_once(tmp_path):
    split = _toy_split()
    _run(tmp_path, split, [])
    _summarize(tmp_path)
    cdir = art.condition_dir(tmp_path, "TOY", "no_dn")

    patients = pd.read_csv(cdir / "cv_patient_predictions.csv", dtype={"patient_uid": str})
    assert sorted(patients["patient_uid"]) == split.development_patients()    # una por paciente de desarrollo
    assert not set(patients["patient_uid"]) & split.blocked_test_patients()   # ningun paciente de prueba
    segments = pd.read_csv(cdir / "cv_segment_predictions.csv", dtype={"patient_uid": str})
    assert not set(segments["patient_uid"]) & split.blocked_test_patients()
    summary = pd.read_csv(cdir / "cv_metrics_summary.csv")
    assert list(summary["statistic"]) == ["fold_mean", "fold_std", "pooled"]
    assert summary.loc[2, "n"] == len(split.development_patients())


def test_run_manifest_and_status_document_the_blocked_test_and_missing_final_model(tmp_path):
    split = _toy_split()
    _run(tmp_path, split, [])
    _summarize(tmp_path)
    cdir = art.condition_dir(tmp_path, "TOY", "no_dn")
    manifest = json.loads((cdir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"] == "holdout_cv_v3" and manifest["run_id"] == "RUN_X"
    assert manifest["outer_test"] == {"enabled": False, "accessed": False}
    assert manifest["final_model"] == {"enabled": False}
    assert art.read_status(cdir)["status"] == "COMPLETED"
    assert 'protocol = "holdout_cv_v3"' in (cdir / "resolved_config.toml").read_text(encoding="utf-8")


def test_resume_skips_completed_units_and_reruns_only_failed_ones(tmp_path):
    split = _toy_split()
    calls = []
    fail_once = lambda config, fold: config["config_index"] == 2 and fold == 3          # noqa: E731
    _run(tmp_path, split, calls, fail=fail_once)

    result = _summarize(tmp_path)
    cdir = art.condition_dir(tmp_path, "TOY", "no_dn")
    assert result["status"] == "PARTIAL" and not result["ok"]
    search = pd.read_csv(cdir / "cv_search_results.csv")
    failed = search.loc[search["status"] == "FAILED"]
    assert len(failed) == 1 and failed.iloc[0]["config_index"] == 2 and "fallo simulado" in failed.iloc[0]["error"]
    # La configuracion 2 (la mejor) quedo incompleta: no puede seleccionarse.
    best = json.loads((cdir / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert best["config_index"] != 2 and best["selection"]["n_incomplete_candidates"] == 1

    # Reanudar: solo se vuelve a ejecutar la unidad fallida.
    calls.clear()
    _run(tmp_path, split, calls)
    assert calls == [(2, 3)]
    result = _summarize(tmp_path)
    assert result["status"] == "COMPLETED"
    assert json.loads((cdir / "best_hyperparameters.json").read_text(encoding="utf-8"))["config_index"] == 2
    assert not hcv.unit_failure_marker(tmp_path, "TOY", "no_dn", 2, 3).exists()

    # Una tercera ejecucion no evalua nada.
    calls.clear()
    _run(tmp_path, split, calls)
    assert calls == []


def test_abandoned_unit_staging_is_removed_and_never_counted_as_a_result(tmp_path):
    split = _toy_split()
    _run(tmp_path, split, [], fold_ids=(0,))
    staging = hcv.unit_staging_dir(tmp_path, "TOY", "no_dn", 4, 1)
    staging.mkdir(parents=True)
    (staging / "unit_result.json").write_text("{}", encoding="utf-8")
    assert not hcv.is_unit_complete(tmp_path, "TOY", "no_dn", 4, 1)

    removed = hcv.cleanup_abandoned_unit_staging(tmp_path)
    assert removed == [staging] and not staging.exists()
    assert hcv.is_unit_complete(tmp_path, "TOY", "no_dn", 4, 0)               # lo publicado se conserva


def test_a_failure_while_preparing_a_fold_marks_every_pending_unit_failed(tmp_path):
    split = _toy_split()

    def broken_prepare(fold_id, train_patients, val_patients):
        raise FileNotFoundError("falta el .npy del fold")

    hcv.run_condition_cv(
        run_root=tmp_path, spec=SPEC, model_name="svm_rbf", split=split, configs=CONFIGS,
        hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, negative_label_name="Healthy", threshold=0.5,
        fold_ids=[0], prepare_fold=broken_prepare, evaluate_fold=lambda c, p: iter(()),
        release_fold=None, logger=LOGGER,
    )
    assert all(hcv.unit_failure_marker(tmp_path, "TOY", "no_dn", c["config_index"], 0).is_file() for c in CONFIGS)
    result = _summarize(tmp_path, fold_ids=(0,))
    assert result["status"] == "FAILED" and not result["ok"] and result["summary_row"] is None
    assert art.read_status(art.condition_dir(tmp_path, "TOY", "no_dn"))["status"] == "FAILED"


def test_reused_configuration_gets_its_own_results_and_epochs(tmp_path):
    """no_dn_aug: sin busqueda; reutiliza la configuracion de no_dn y guarda
    sus propios resultados y mejores epocas por fold."""
    split = _toy_split()
    aug_spec = dataclasses.replace(SPEC, condition="no_dn_aug", augment=True, hyperparameters_from="no_dn")
    only = [CONFIGS[4]]                                     # la configuracion que eligio no_dn
    prepare_fold, evaluate_fold = _evaluator(split, [])
    hcv.run_condition_cv(
        run_root=tmp_path, spec=aug_spec, model_name="svm_rbf", split=split, configs=only,
        hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, negative_label_name="Healthy", threshold=0.5,
        fold_ids=[0, 1, 2, 3, 4], prepare_fold=prepare_fold, evaluate_fold=evaluate_fold,
        release_fold=None, logger=LOGGER,
    )
    result = hcv.summarize_condition(
        run_root=tmp_path, spec=aug_spec, cfg_used=MIN_CFG, model_name="svm_rbf", configs=only,
        hyperparameter_keys=hcv.SVM_HYPERPARAMETER_KEYS, fold_ids=[0, 1, 2, 3, 4], negative_label_name="Healthy",
        threshold=0.5, hyperparameter_source="reused_from:no_dn", reused_config_index=4,
        provenance=PROVENANCE, logger=LOGGER,
    )

    cdir = art.condition_dir(tmp_path, "TOY", "no_dn_aug")
    best = json.loads((cdir / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert result["status"] == "COMPLETED"
    assert best["config_index"] == 4 and best["augment"] is True
    assert best["hyperparameters_source"] == "reused_from:no_dn"
    assert best["selection"]["decided_by"] == "reused_from:no_dn"
    assert best["best_epochs_by_fold"] == {"0": 10, "1": 20, "2": 30, "3": 40, "4": 50}   # propias
    search = pd.read_csv(cdir / "cv_search_results.csv")
    assert set(search["config_index"]) == {4} and len(search) == 5                      # sin busqueda


def test_final_model_directory_is_a_hard_error(tmp_path):
    split = _toy_split()
    _run(tmp_path, split, [])
    result = _summarize(tmp_path)
    hcv.assert_no_final_model(tmp_path, [result])                       # no lanza
    (art.condition_dir(tmp_path, "TOY", "no_dn") / "final").mkdir()
    with pytest.raises(RuntimeError, match="modelo definitivo"):
        hcv.assert_no_final_model(tmp_path, [result])


def test_final_run_status_summarises_conditions():
    ok, partial, failed = ({"status": s} for s in ("COMPLETED", "PARTIAL", "FAILED"))
    assert hcv.final_run_status([ok, ok]) == art.STATUS_COMPLETED
    assert hcv.final_run_status([ok, partial]) == art.STATUS_PARTIAL
    assert hcv.final_run_status([ok, failed]) == art.STATUS_PARTIAL
    assert hcv.final_run_status([failed, failed]) == art.STATUS_FAILED
    assert hcv.final_run_status([]) == art.STATUS_FAILED


def test_run_summary_table_has_every_metric_with_fold_mean_std_and_pooled_value(tmp_path):
    split = _toy_split()
    _run(tmp_path, split, [])
    result = _summarize(tmp_path)

    row = result["summary_row"]
    for metric in hcv.SUMMARY_METRICS:       # accuracy, BA, recall COPD/negativo, minimo recall, macro-F1, AUROC, AUPRC
        for prefix in ("fold_mean_", "fold_std_", "pooled_"):
            assert f"{prefix}{metric}" in row, f"{prefix}{metric}"
    assert row["fold_mean_balanced_accuracy"] == pytest.approx(1.0) and row["n_folds"] == 5

    hcv.write_run_tables(tmp_path, [result])
    table = pd.read_csv(tmp_path / "cv_run_summary.csv")
    expected = {f"{p}{m}" for p in ("fold_mean_", "fold_std_", "pooled_") for m in hcv.SUMMARY_METRICS}
    assert expected <= set(table.columns) and len(table) == 1


def test_figures_failures_are_logged_but_do_not_invalidate_the_condition(tmp_path, monkeypatch):
    split = _toy_split()
    _run(tmp_path, split, [])

    def broken(*args, **kwargs):
        raise RuntimeError("matplotlib no disponible")

    monkeypatch.setattr(art, "plot_roc_curve", broken)
    result = _summarize(tmp_path)

    cdir = art.condition_dir(tmp_path, "TOY", "no_dn")
    assert result["status"] == "COMPLETED"
    assert not (cdir / "figures" / "roc.png").exists()
    assert (cdir / "figures" / "pr.png").is_file() and (cdir / "cv_roc_curve.csv").is_file()   # el resto sigue


# ---------------------------------------------------------------------------
# Huella del codigo: --resume no reutiliza unidades entrenadas con otra implementacion
# ---------------------------------------------------------------------------

def _code_tree(root):
    (root / "models").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "__pycache__").mkdir()
    (root / "a.py").write_text("x = 1\n", encoding="utf-8")
    (root / "models" / "m.py").write_text("y = 2\n", encoding="utf-8")
    (root / "tests" / "test_a.py").write_text("assert True\n", encoding="utf-8")
    (root / "__pycache__" / "junk.py").write_text("z = 3\n", encoding="utf-8")
    (root / "run_holdout_sequence.py").write_text("launcher = 1\n", encoding="utf-8")
    return root


def test_code_fingerprint_is_reproducible_and_covers_only_relevant_files(tmp_path):
    root = _code_tree(tmp_path / "code")
    first, second = hcv.code_fingerprint(root), hcv.code_fingerprint(root)
    assert first == second                                          # reproducible
    assert set(first["code_files"]) == {"a.py", "models/m.py"}       # sin tests, __pycache__ ni lanzadores
    assert len(first["code_sha256"]) == 64


def test_code_fingerprint_changes_when_any_relevant_file_changes_or_appears(tmp_path):
    root = _code_tree(tmp_path / "code")
    base = hcv.code_fingerprint(root)

    (root / "models" / "m.py").write_text("y = 3\n", encoding="utf-8")
    edited = hcv.code_fingerprint(root)
    assert edited["code_sha256"] != base["code_sha256"]
    assert {k for k in edited["code_files"] if edited["code_files"][k] != base["code_files"][k]} == {"models/m.py"}

    (root / "new_module.py").write_text("w = 4\n", encoding="utf-8")
    assert hcv.code_fingerprint(root)["code_sha256"] != edited["code_sha256"]


def test_code_fingerprint_ignores_tests_launchers_and_line_endings(tmp_path):
    root = _code_tree(tmp_path / "code")
    base = hcv.code_fingerprint(root)

    (root / "tests" / "test_a.py").write_text("assert False\n", encoding="utf-8")            # pruebas
    (root / "run_holdout_sequence.py").write_text("launcher = 2\n", encoding="utf-8")        # lanzador
    (root / "__pycache__" / "junk.py").write_text("z = 9\n", encoding="utf-8")
    (root / "a.py").write_bytes(b"x = 1\r\n")                                               # CRLF, mismo contenido
    assert hcv.code_fingerprint(root) == base


def test_real_code_fingerprint_lists_the_training_modules_and_no_tests():
    files = set(hcv.code_fingerprint()["code_files"])
    assert {"holdout_cv.py", "holdout_svm.py", "holdout_cnn.py", "models/cnn.py", "models/svm_rbf.py",
            "data.py", "evaluation.py", "splits.py"} <= files
    assert not any(name.startswith("tests/") for name in files)
    assert "run_holdout_sequence.py" not in files


def _fingerprint(**overrides):
    fp = {"dataset_arg": "all", "experiment_arg": "all", "config_fingerprint": "cfg", "input_hashes": {"x": "1"},
          "code_sha256": "code-A", "code_files": {"a.py": "h1", "b.py": "h2"}}
    fp.update(overrides)
    return fp


def test_resume_verification_rejects_changed_code_and_names_the_changed_files(tmp_path):
    rexp.write_run_fingerprint(tmp_path, _fingerprint())
    rexp.verify_run_fingerprint(tmp_path, _fingerprint())                                   # igual: no lanza

    changed = _fingerprint(code_sha256="code-B", code_files={"a.py": "h1", "b.py": "OTRO"})
    with pytest.raises(RuntimeError, match="codigo de modeling") as info:
        rexp.verify_run_fingerprint(tmp_path, changed)
    assert "b.py" in str(info.value) and "a.py" not in str(info.value).split("archivos:")[1]


def test_resume_verification_is_unchanged_for_protocols_without_a_code_hash(tmp_path):
    legacy = {k: v for k, v in _fingerprint().items() if not k.startswith("code_")}
    rexp.write_run_fingerprint(tmp_path, legacy)
    rexp.verify_run_fingerprint(tmp_path, legacy)                                           # v1/v2: ambos sin huella

    with pytest.raises(RuntimeError, match="codigo de modeling"):                           # una v3 nunca reanuda una corrida sin huella
        rexp.verify_run_fingerprint(tmp_path, _fingerprint())
