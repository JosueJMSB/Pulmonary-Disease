"""Prueba sin internet (comparacion) e indicadores del informe, con datos sinteticos."""

import pandas as pd
import pytest

pytest.importorskip("sklearn")

from optimization import offline_check, paths, report  # noqa: E402

SHA = {"fp32": "f" * 64, "int8": "e" * 64, "bundle": "b" * 64}


def _write_run(base, mode, connected, datasets=("ICBHI",), scores=None, status="OK", int8_sha=SHA["int8"]):
    out = base / mode
    out.mkdir(parents=True, exist_ok=True)
    meta = {"mode": mode, "probe_connected": connected, "datasets": {}}
    for name in datasets:
        versions = {}
        for version in ("fp32", "int8"):
            sha = SHA["fp32"] if version == "fp32" else int8_sha
            frame = pd.DataFrame({
                "audio_id": ["a1", "a2"], "model_sha256": [sha, sha],
                "recording_score": scores or ["0.25", "0.75"], "decision": [0, 1], "status": [status, status],
            })
            fname = f"{name}_{version}.csv"
            paths.write_csv_atomic(frame, out / fname)
            versions[version] = {"file": fname, "model_sha256": sha, "n_ok": 2, "n_recordings": 2}
        meta["datasets"][name] = {"bundle_sha256": SHA["bundle"], "versions": versions}
    return meta


def test_identical_predictions_with_connectivity_change_pass(tmp_path):
    online = _write_run(tmp_path, "online", True)
    offline = _write_run(tmp_path, "offline", False)
    result = offline_check.compare_runs(tmp_path, online, offline)
    assert result["status"] == "PASS"
    item = result["datasets"]["ICBHI"]
    assert item["status"] == "PASS"
    assert item["fp32_sha256"] == SHA["fp32"] and item["int8_sha256"] == SHA["int8"]
    assert item["bundle_sha256"] == SHA["bundle"]


def test_online_run_without_connection_is_not_verified(tmp_path):
    online = _write_run(tmp_path, "online", False)
    offline = _write_run(tmp_path, "offline", False)
    result = offline_check.compare_runs(tmp_path, online, offline)
    assert result["status"] == "NO VERIFICADO"
    assert result["datasets"]["ICBHI"]["status"] == "NO VERIFICADO"


def test_offline_run_with_connection_is_not_verified(tmp_path):
    online = _write_run(tmp_path, "online", True)
    offline = _write_run(tmp_path, "offline", True)
    assert offline_check.compare_runs(tmp_path, online, offline)["status"] == "NO VERIFICADO"


def test_different_predictions_fail(tmp_path):
    online = _write_run(tmp_path, "online", True)
    offline = _write_run(tmp_path, "offline", False, scores=["0.25", "0.76"])
    result = offline_check.compare_runs(tmp_path, online, offline)
    assert result["status"] == "FAIL" and result["datasets"]["ICBHI"]["status"] == "FAIL"


def test_dataset_missing_in_one_run_fails(tmp_path):
    online = _write_run(tmp_path, "online", True, datasets=("ICBHI", "COMBINED"))
    offline = _write_run(tmp_path, "offline", False, datasets=("ICBHI",))
    result = offline_check.compare_runs(tmp_path, online, offline)
    assert result["status"] == "FAIL"
    assert result["datasets"]["ICBHI"]["status"] == "PASS"
    assert result["datasets"]["COMBINED"]["status"] == "FAIL"


def _bundle():
    return {"models": {"fp32": {"sha256": SHA["fp32"]}, "int8": {"sha256": SHA["int8"]}}}


def _offline_result(datasets):
    return {"status": "PASS", "datasets": {
        name: {"status": "PASS", "detail": "ok", "fp32_sha256": SHA["fp32"], "int8_sha256": SHA["int8"],
               "bundle_sha256": SHA["bundle"]} for name in datasets}}


def test_report_only_marks_tested_datasets():
    offline = _offline_result(["ICBHI"])
    assert report.offline_status("ICBHI", _bundle(), SHA["bundle"], offline)["iov_status"] == "CUMPLE"
    # Probar solo ICBHI no aprueba los demas datasets.
    assert report.offline_status("COMBINED", _bundle(), SHA["bundle"], offline)["iov_status"] == "NO VERIFICADO"


def test_report_rejects_offline_result_for_other_models():
    offline = _offline_result(["ICBHI"])
    other_bundle = {"models": {"fp32": {"sha256": SHA["fp32"]}, "int8": {"sha256": "0" * 64}}}
    row = report.offline_status("ICBHI", other_bundle, SHA["bundle"], offline)
    assert row["iov_status"] == "NO VERIFICADO" and "int8_sha256" in row["value"]
    row = report.offline_status("ICBHI", _bundle(), "1" * 64, offline)
    assert row["iov_status"] == "NO VERIFICADO" and "bundle_sha256" in row["value"]


def test_report_without_offline_result_is_not_verified():
    assert report.offline_status("ICBHI", _bundle(), SHA["bundle"], None)["iov_status"] == "NO VERIFICADO"


IOV = {"max_performance_drop_points": 20.0,
       "performance_metrics": ["balanced_accuracy", "recall_copd", "recall_negative"]}
GOALS = {"max_ba_drop_points": 5.0}


def _test_summary(int8):
    base = {"balanced_accuracy": 0.90, "recall_copd": 1.00, "recall_negative": 0.80}
    return {"metrics_base_pt_publicado": base, "metrics_int8": {**base, **int8}, "small_test_warning": "test pequeno"}


def test_performance_fails_if_any_committed_metric_drops_too_much():
    # BA cae 2 puntos (cumple), pero el recall de EPOC cae 30 puntos.
    rows = report.performance_status("ICBHI", _test_summary({"balanced_accuracy": 0.88, "recall_copd": 0.70}),
                                     IOV, GOALS)
    by_indicator = {r["indicator"].split(" ")[0]: r for r in rows}
    assert by_indicator["caida_desempeno"]["iov_status"] == "NO CUMPLE"
    assert by_indicator["caida_balanced_accuracy"]["iov_status"] == "CUMPLE"
    assert by_indicator["caida_recall_copd"]["iov_status"] == "NO CUMPLE"
    assert by_indicator["caida_balanced_accuracy"]["goal_status"] == "CUMPLE"
    assert by_indicator["caida_recall_copd"]["goal_status"] == "-"


def test_performance_passes_only_if_all_metrics_pass():
    rows = report.performance_status("ICBHI", _test_summary({}), IOV, GOALS)
    assert all(r["iov_status"] == "CUMPLE" for r in rows)
    assert len(rows) == 1 + len(IOV["performance_metrics"])


def test_performance_without_test_is_not_verified():
    rows = report.performance_status("ICBHI", None, IOV, GOALS)
    assert all(r["iov_status"] == "NO VERIFICADO" for r in rows)
