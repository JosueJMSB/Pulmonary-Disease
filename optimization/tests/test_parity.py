import numpy as np
import pytest

from optimization import parity


def test_fp32_parity_passes_within_strict_tolerance():
    logits = np.array([-3.0, 0.5, 2.0])
    probs = 1.0 / (1.0 + np.exp(-logits))
    report, changes = parity.fp32_parity(logits, logits + 1e-6, probs, probs + 1e-7, ["a", "b", "c"], 0.5,
                                         max_abs_logit=1e-4, max_abs_prob=1e-5)
    assert report["status"] == "PASS"
    assert report["n_decision_changes"] == 0 and changes.empty


def test_fp32_parity_fails_outside_tolerance():
    logits = np.array([-3.0, 0.5, 2.0])
    probs = 1.0 / (1.0 + np.exp(-logits))
    report, _ = parity.fp32_parity(logits, logits + 1e-3, probs, probs, ["a", "b", "c"], 0.5,
                                   max_abs_logit=1e-4, max_abs_prob=1e-5)
    assert report["status"] == "FAIL"


def test_decision_changes_are_reported_separately():
    report, changes = parity.fp32_parity(np.zeros(2), np.zeros(2), np.array([0.5, 0.2]),
                                         np.array([0.4999999, 0.2]), ["x", "y"], 0.5,
                                         max_abs_logit=1e-4, max_abs_prob=1e-5)
    assert report["status"] == "PASS"
    assert report["n_decision_changes"] == 1
    assert changes.loc[0, "id"] == "x"


def test_int8_report_has_no_pass_fail_and_counts_changes():
    fp32 = np.array([0.9, 0.55, 0.2, 0.45])
    int8 = np.array([0.85, 0.45, 0.25, 0.55])
    report, changes = parity.int8_report(fp32, int8, None, None, ["a", "b", "c", "d"], 0.5)
    assert "status" not in report
    assert report["n_decision_changes"] == 2
    assert report["concordance"] == pytest.approx(0.5)
    assert set(changes["id"]) == {"b", "d"}
    assert report["abs_prob_diff"]["max"] == pytest.approx(0.10)


def test_acceptance_check():
    acceptance = {"max_drop_balanced_accuracy": 0.05, "max_drop_recall_copd": 0.05,
                  "max_drop_recall_negative": 0.05}
    fp32 = {"balanced_accuracy": 0.90, "recall_copd": 0.95, "recall_negative": 0.85}
    ok = parity.acceptance_check(fp32, {"balanced_accuracy": 0.88, "recall_copd": 0.95,
                                        "recall_negative": 0.81}, acceptance)
    assert ok["accepted"]
    bad = parity.acceptance_check(fp32, {"balanced_accuracy": 0.88, "recall_copd": 0.85,
                                         "recall_negative": 0.85}, acceptance)
    assert not bad["accepted"]
    assert [c["metric"] for c in bad["checks"] if not c["passed"]] == ["recall_copd"]
