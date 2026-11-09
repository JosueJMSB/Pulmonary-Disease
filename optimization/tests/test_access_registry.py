import pytest

from optimization import access_registry as ar

FINGERPRINT = {"dataset": "X", "int8_sha256": "a" * 64, "bundle_sha256": "b" * 64}


def _paths(tmp_path):
    access = tmp_path / "_test_access"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    predictions = run_dir / "preds.csv"
    return access, run_dir, predictions


def test_new_evaluation_then_resume_after_started_reinfers(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    assert ar.begin(access, "X", FINGERPRINT, run_dir, preds) == ar.RESUME_NEW
    # Interrupcion tras STARTED: misma huella y misma corrida -> se reanuda.
    assert ar.begin(access, "X", FINGERPRINT, run_dir, preds) == ar.RESUME_REINFER


def test_resume_reuses_written_predictions(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    preds.write_text("segment_id,logit_int8\ns1,0.1\n", encoding="utf-8")
    ar.mark_predictions_written(access, "X", preds)
    assert ar.begin(access, "X", FINGERPRINT, run_dir, preds) == ar.RESUME_REUSE_PREDICTIONS


def test_altered_predictions_require_manual_review(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    preds.write_text("segment_id,logit_int8\ns1,0.1\n", encoding="utf-8")
    ar.mark_predictions_written(access, "X", preds)
    preds.write_text("segment_id,logit_int8\ns1,0.9\n", encoding="utf-8")
    with pytest.raises(ar.TestAccessInconsistent):
        ar.begin(access, "X", FINGERPRINT, run_dir, preds)


def test_missing_predictions_after_written_require_manual_review(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    preds.write_text("x\n", encoding="utf-8")
    ar.mark_predictions_written(access, "X", preds)
    preds.unlink()
    with pytest.raises(ar.TestAccessInconsistent):
        ar.begin(access, "X", FINGERPRINT, run_dir, preds)


def test_different_fingerprint_is_blocked(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    other = {**FINGERPRINT, "int8_sha256": "c" * 64}
    with pytest.raises(ar.TestAccessBlocked, match="int8_sha256"):
        ar.begin(access, "X", other, run_dir, preds)


def test_other_run_dir_is_blocked(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    other_run = tmp_path / "otra_corrida"
    other_run.mkdir()
    with pytest.raises(ar.TestAccessBlocked):
        ar.begin(access, "X", FINGERPRINT, other_run, other_run / "preds.csv")


def test_completed_evaluation_is_never_repeated(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    preds.write_text("x\n", encoding="utf-8")
    ar.mark_predictions_written(access, "X", preds)
    ar.mark_completed(access, "X", {"summary": "ok"})
    with pytest.raises(ar.TestAccessBlocked, match="completo"):
        ar.begin(access, "X", FINGERPRINT, run_dir, preds)


def test_datasets_are_independent(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    assert ar.begin(access, "Y", {**FINGERPRINT, "dataset": "Y"}, run_dir, run_dir / "y.csv") == ar.RESUME_NEW


def test_predictions_written_but_registry_still_started_are_recovered(tmp_path):
    # Interrupcion entre escribir el archivo y actualizar el registro.
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    preds.write_text("segment_id,logit_int8\ns1,0.1\n", encoding="utf-8")
    assert ar.read_record(access, "X")["status"] == ar.STARTED
    assert ar.begin(access, "X", FINGERPRINT, run_dir, preds) == ar.RESUME_RECOVER_PREDICTIONS
    ar.mark_predictions_written(access, "X", preds)
    assert ar.begin(access, "X", FINGERPRINT, run_dir, preds) == ar.RESUME_REUSE_PREDICTIONS


def test_lock_blocks_a_second_concurrent_evaluation(tmp_path):
    access, _, _ = _paths(tmp_path)
    with ar.exclusive_lock(access, "X"):
        assert ar.lock_path(access, "X").is_file()
        with pytest.raises(ar.TestAccessBlocked, match="en curso"):
            with ar.exclusive_lock(access, "X"):
                pass
        # Otro dataset no queda bloqueado.
        with ar.exclusive_lock(access, "Y"):
            pass
    assert not ar.lock_path(access, "X").exists()


def test_lock_is_released_after_an_exception(tmp_path):
    access, _, _ = _paths(tmp_path)
    with pytest.raises(RuntimeError):
        with ar.exclusive_lock(access, "X"):
            raise RuntimeError("interrupcion")
    assert not ar.lock_path(access, "X").exists()
    with ar.exclusive_lock(access, "X"):
        pass


def test_stale_lock_is_not_removed_automatically(tmp_path):
    access, _, _ = _paths(tmp_path)
    access.mkdir(parents=True)
    ar.lock_path(access, "X").write_text('{"pid": 999999}\n', encoding="utf-8")
    with pytest.raises(ar.TestAccessBlocked, match="999999"):
        with ar.exclusive_lock(access, "X"):
            pass
    assert ar.lock_path(access, "X").is_file()


def test_record_does_not_claim_test_was_never_used(tmp_path):
    access, run_dir, preds = _paths(tmp_path)
    ar.begin(access, "X", FINGERPRINT, run_dir, preds)
    note = ar.read_record(access, "X")["note"]
    assert "ya se uso" in note and "INT8" in note
