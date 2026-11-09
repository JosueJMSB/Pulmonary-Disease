import pytest

from optimization import bundle as bmod
from optimization.tests.helpers import fake_sha, make_bundle


def test_valid_bundle_passes():
    bmod.validate_bundle(make_bundle())
    bmod.validate_bundle(make_bundle(with_int8=True), require_int8=True)


def test_missing_peak_ceiling_is_rejected():
    bundle = make_bundle()
    del bundle["preprocessing"]["amplitude"]["peak_ceiling"]
    with pytest.raises(bmod.BundleError, match="peak_ceiling"):
        bmod.validate_bundle(bundle)


@pytest.mark.parametrize("keys", [
    ("preprocessing", "amplitude", "target_rms"),
    ("preprocessing", "amplitude", "max_gain"),
    ("preprocessing", "denoising", "noise_pct"),
    ("normalization", "std"),
    ("decision", "threshold"),
])
def test_any_missing_inference_parameter_is_rejected(keys):
    bundle = make_bundle()
    node = bundle
    for key in keys[:-1]:
        node = node[key]
    del node[keys[-1]]
    with pytest.raises(bmod.BundleError):
        bmod.validate_bundle(bundle)


def test_int8_block_is_required_only_when_asked():
    bundle = make_bundle()
    bmod.validate_bundle(bundle)
    with pytest.raises(bmod.BundleError, match="models.int8"):
        bmod.validate_bundle(bundle, require_int8=True)


def test_fp32_and_int8_hashes_must_be_separate():
    bundle = make_bundle(with_int8=True)
    assert bundle["models"]["fp32"]["sha256"] != bundle["models"]["int8"]["sha256"]
    bundle["models"]["int8"]["sha256"] = bundle["models"]["fp32"]["sha256"]
    with pytest.raises(bmod.BundleError, match="igual"):
        bmod.validate_bundle(bundle, require_int8=True)


def test_calibration_must_use_only_development_role():
    bundle = make_bundle(with_int8=True)
    bundle["models"]["int8"]["calibration"]["roles"] = ["train", "test"]
    with pytest.raises(bmod.BundleError, match="train"):
        bmod.validate_bundle(bundle, require_int8=True)


def test_branch_and_denoising_must_agree():
    bundle = make_bundle()
    bundle["branch"] = "no_dn"
    with pytest.raises(bmod.BundleError, match="no_dn"):
        bmod.validate_bundle(bundle)


def test_verify_model_file_detects_altered_file(tmp_path):
    bundle = make_bundle()
    model = tmp_path / "model_fp32.onnx"
    model.write_bytes(b"0123456789")
    bundle["models"]["fp32"]["sha256"] = bmod.sha256_file(model)
    bundle["models"]["fp32"]["size_bytes"] = 10
    assert bmod.verify_model_file(tmp_path, bundle, "fp32") == model
    model.write_bytes(b"0123456780")
    with pytest.raises(bmod.BundleError, match="hash"):
        bmod.verify_model_file(tmp_path, bundle, "fp32")


def test_write_and_load_roundtrip(tmp_path):
    bundle = make_bundle(with_int8=True)
    bmod.write_bundle(bundle, tmp_path, require_int8=True)
    loaded = bmod.load_bundle(tmp_path, require_int8=True)
    assert loaded == bundle
    assert loaded["models"]["int8"]["sha256"] == fake_sha("int8")
