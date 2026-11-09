"""Congelacion y precondiciones del acceso al test (sin datos reales)."""

import copy

import pytest

pytest.importorskip("sklearn")

from optimization import bundle as bmod  # noqa: E402
from optimization import compare_dev, evaluate_test, paths  # noqa: E402
from optimization.tests.helpers import make_bundle  # noqa: E402


def _frozen_dataset(tmp_path, cfg):
    ds_dir = tmp_path / "SINTETICO"
    ds_dir.mkdir()
    bundle = make_bundle(with_int8=True)
    for version, payload in (("fp32", b"fp32-model"), ("int8", b"int8")):
        path = ds_dir / bundle["models"][version]["file"]
        path.write_bytes(payload)
        bundle["models"][version]["sha256"] = bmod.sha256_file(path)
        bundle["models"][version]["size_bytes"] = len(payload)
    bundle["models"]["int8"]["frozen"] = True
    bmod.write_bundle(bundle, ds_dir, require_int8=True)
    check = {"status": "PASS", "fp32_sha256": bundle["models"]["fp32"]["sha256"],
             "int8_sha256": bundle["models"]["int8"]["sha256"]}
    paths.write_json_atomic(check, ds_dir / "pipeline_check.json")
    freeze = {
        "fp32_sha256": bundle["models"]["fp32"]["sha256"],
        "int8_sha256": bundle["models"]["int8"]["sha256"],
        "bundle_sha256": bmod.bundle_file_sha256(ds_dir),
        "acceptance_sha256": paths.acceptance_sha256(cfg),
        "pipeline_check_sha256": bmod.sha256_file(ds_dir / "pipeline_check.json"),
    }
    paths.write_json_atomic(freeze, ds_dir / "freeze.json")
    return ds_dir, bundle


def test_preconditions_pass_for_a_consistent_frozen_int8(tmp_path):
    cfg = paths.load_config()
    ds_dir, _ = _frozen_dataset(tmp_path, cfg)
    evaluate_test.check_preconditions(ds_dir, cfg)


def test_missing_freeze_blocks_test_access(tmp_path):
    cfg = paths.load_config()
    ds_dir, _ = _frozen_dataset(tmp_path, cfg)
    (ds_dir / "freeze.json").unlink()
    with pytest.raises(evaluate_test.PreconditionError, match="congelado"):
        evaluate_test.check_preconditions(ds_dir, cfg)


def test_changed_acceptance_criterion_blocks_test_access(tmp_path):
    cfg = paths.load_config()
    ds_dir, _ = _frozen_dataset(tmp_path, cfg)
    changed = copy.deepcopy(cfg)
    changed["acceptance"]["max_drop_balanced_accuracy"] = 0.20
    with pytest.raises(evaluate_test.PreconditionError, match="acceptance"):
        evaluate_test.check_preconditions(ds_dir, changed)


def test_failed_pipeline_check_blocks_test_access(tmp_path):
    cfg = paths.load_config()
    ds_dir, bundle = _frozen_dataset(tmp_path, cfg)
    paths.write_json_atomic({"status": "FAIL", "fp32_sha256": bundle["models"]["fp32"]["sha256"],
                             "int8_sha256": bundle["models"]["int8"]["sha256"]}, ds_dir / "pipeline_check.json")
    with pytest.raises(evaluate_test.PreconditionError, match="check_pipeline"):
        evaluate_test.check_preconditions(ds_dir, cfg)


def test_modified_int8_blocks_test_access(tmp_path):
    cfg = paths.load_config()
    ds_dir, _ = _frozen_dataset(tmp_path, cfg)
    (ds_dir / "model_int8.onnx").write_bytes(b"INT8")
    with pytest.raises(bmod.BundleError):
        evaluate_test.check_preconditions(ds_dir, cfg)


def test_freeze_requires_pipeline_check_of_the_same_models(tmp_path):
    cfg = paths.load_config()
    ds_dir, bundle = _frozen_dataset(tmp_path, cfg)
    ok, _ = compare_dev._pipeline_check_ok(ds_dir, bundle)
    assert ok
    other = copy.deepcopy(bundle)
    other["models"]["int8"]["sha256"] = "f" * 64
    ok, detail = compare_dev._pipeline_check_ok(ds_dir, other)
    assert not ok and "otros modelos" in detail
    (ds_dir / "pipeline_check.json").unlink()
    ok, detail = compare_dev._pipeline_check_ok(ds_dir, bundle)
    assert not ok and "pipeline_check" in detail
