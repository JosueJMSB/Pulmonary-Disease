"""Lanzador secuencial del protocolo fold-aware (v2): orden de los 6 pasos
(cada modelo primero individual, luego COMBINED), deteccion de fallos,
persistencia del run_id ante una interrupcion y --resume-sequence.

Mismo patron de doble de prueba que test_run_combined_sequence.py, pero
para este script independiente. ``subprocess.Popen`` se reemplaza por un
doble: esto nunca invoca modeling.run_experiment de verdad ni toca datos
reales.
"""

import json

from .. import run_folded_sequence as rfs


def _config_basename(path_str: str) -> str:
    return path_str.replace("\\", "/").rsplit("/", 1)[-1]


class FakePopen:
    def __init__(self, command, script):
        self.command = command
        model = command[command.index("--model") + 1]
        config_name = _config_basename(command[command.index("--config") + 1])
        is_dry_run = "--dry-run" in command
        phase = "dry_run" if is_dry_run else "train"
        outcome = script.get((phase, model, config_name), {})
        self._returncode = outcome.get("returncode", 0)
        self._run_id = outcome.get("run_id", f"RUN_{model}_{config_name}")
        self._interrupt = outcome.get("interrupt_after_run_id", False)
        self._is_dry_run = is_dry_run
        self.returncode = None
        self.stdout = self._make_stdout()

    def _make_stdout(self):
        if self._is_dry_run:
            yield "veredicto: OK\n" if self._returncode == 0 else "veredicto: REVISAR\n"
            return
        yield f"2026-09-21 00:00:00,000 INFO run_id={self._run_id} dataset=all experiment=all\n"
        if self._interrupt:
            raise KeyboardInterrupt()
        estado = "COMPLETED" if self._returncode == 0 else "FAILED"
        yield f"run_id={self._run_id} estado={estado} -> /fake/runs/{self.command[1]}\n"

    def wait(self, timeout=None):
        self.returncode = self._returncode
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


def _fake_popen_factory(script, calls):
    def _factory(command, cwd=None, stdout=None, stderr=None, text=None, bufsize=None):
        calls.append(command)
        return FakePopen(command, script)
    return _factory


def _models_and_configs(calls):
    out = []
    for cmd in calls:
        model = cmd[cmd.index("--model") + 1]
        config_name = _config_basename(cmd[cmd.index("--config") + 1])
        out.append((model, config_name))
    return out


def _read_status(tmp_path) -> dict:
    status_path = next((tmp_path / "folded_sequences").glob("*/sequence_status.json"))
    return json.loads(status_path.read_text(encoding="utf-8"))


EXPECTED_ORDER = list(rfs.STEPS)


def test_dry_run_only_visits_all_six_steps_in_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rfs.main(["--dry-run-only", "--runs-root", str(tmp_path)])

    assert rc == 0
    assert _models_and_configs(calls) == EXPECTED_ORDER
    assert _read_status(tmp_path)["final_status"] == "DRY_RUN_OK"


def test_dry_run_failure_stops_before_later_steps(tmp_path, monkeypatch):
    script = {("dry_run", "cnn", "cnn_v2.toml"): {"returncode": 1}}
    calls = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rfs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 1
    # Se detiene justo al fallar cnn_v2.toml (paso 3 de 6).
    assert _models_and_configs(calls) == EXPECTED_ORDER[:3]
    status = _read_status(tmp_path)
    assert status["final_status"] == "FAILED"
    assert all(step["phase"] == "dry_run" for step in status["steps"])


def test_execute_trains_all_six_steps_in_order_after_dry_run_passes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rfs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 0
    assert _models_and_configs(calls) == EXPECTED_ORDER + EXPECTED_ORDER
    status = _read_status(tmp_path)
    assert status["final_status"] == "COMPLETED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [s["status"] for s in train_steps] == ["COMPLETED"] * 6


def test_training_failure_stops_sequence_and_records_run_id(tmp_path, monkeypatch):
    script = {("train", "cnn", "cnn_combined_v2.toml"): {"returncode": 1, "run_id": "RUN_CNN_COMBINED_1"}}
    calls = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rfs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 1
    # 6 dry-run + svm_v2 + svm_combined_v2 + cnn_v2 + cnn_combined_v2 (falla): nunca llega a crnn.
    assert _models_and_configs(calls) == EXPECTED_ORDER + EXPECTED_ORDER[:4]
    status = _read_status(tmp_path)
    assert status["final_status"] == "FAILED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [s["status"] for s in train_steps] == ["COMPLETED", "COMPLETED", "COMPLETED", "FAILED"]
    assert train_steps[-1]["run_id"] == "RUN_CNN_COMBINED_1"


def test_resume_sequence_skips_completed_and_resumes_failed_step(tmp_path, monkeypatch):
    script1 = {("train", "cnn", "cnn_combined_v2.toml"): {"returncode": 1, "run_id": "RUN_CNN_COMBINED_1"}}
    calls1 = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory(script1, calls1))
    rc1 = rfs.main(["--execute", "--runs-root", str(tmp_path)])
    assert rc1 == 1
    sequence_id = next((tmp_path / "folded_sequences").iterdir()).name

    calls2 = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory({}, calls2))
    rc2 = rfs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence", sequence_id])

    assert rc2 == 0
    # Dry-run siempre se repite; en el entrenamiento, los 3 primeros pasos se
    # omiten (ya COMPLETED) y se reanuda cnn_combined_v2 antes de seguir con crnn.
    assert _models_and_configs(calls2) == EXPECTED_ORDER + EXPECTED_ORDER[3:]

    cnn_combined_resume_call = calls2[len(EXPECTED_ORDER)]
    assert "--resume" in cnn_combined_resume_call
    assert cnn_combined_resume_call[cnn_combined_resume_call.index("--resume") + 1] == "RUN_CNN_COMBINED_1"

    status = _read_status(tmp_path)
    assert status["final_status"] == "COMPLETED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    statuses = [(s["model"], s["config"], s["status"]) for s in train_steps]
    assert statuses == [
        ("svm_rbf", "svm_rbf_v2.toml", "COMPLETED"),
        ("svm_rbf", "svm_rbf_combined_v2.toml", "COMPLETED"),
        ("cnn", "cnn_v2.toml", "COMPLETED"),
        ("cnn", "cnn_combined_v2.toml", "FAILED"),
        ("svm_rbf", "svm_rbf_v2.toml", "SKIPPED"),
        ("svm_rbf", "svm_rbf_combined_v2.toml", "SKIPPED"),
        ("cnn", "cnn_v2.toml", "SKIPPED"),
        ("cnn", "cnn_combined_v2.toml", "COMPLETED"),
        ("crnn", "crnn_v2.toml", "COMPLETED"),
        ("crnn", "crnn_combined_v2.toml", "COMPLETED"),
    ]


def test_interrupted_training_preserves_run_id_and_marks_sequence_interrupted(tmp_path, monkeypatch):
    script = {("train", "crnn", "crnn_v2.toml"): {"run_id": "RUN_CRNN_X", "interrupt_after_run_id": True}}
    calls = []
    monkeypatch.setattr(rfs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rfs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == rfs.EXIT_INTERRUPTED
    assert _models_and_configs(calls) == EXPECTED_ORDER + EXPECTED_ORDER[:5]
    status = _read_status(tmp_path)
    assert status["final_status"] == "INTERRUPTED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert train_steps[-1]["status"] == "INTERRUPTED"
    assert train_steps[-1]["run_id"] == "RUN_CRNN_X"


def test_resolve_runs_root_prefers_cli_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULMONARY_RUNS_ROOT", str(tmp_path / "from_env"))
    resolved = rfs.resolve_runs_root(tmp_path / "from_cli")
    assert resolved == (tmp_path / "from_cli").resolve()


def test_num_workers_default_is_zero():
    args = rfs.parse_args([])
    assert args.num_workers == 0


def test_build_command_passes_cache_root_to_svm_too(tmp_path):
    cache_root = tmp_path / "cache"
    args = rfs.parse_args(["--cache-root", str(cache_root)])
    runs_root = tmp_path / "runs"

    svm_cmd = rfs.build_command(
        "svm_rbf", "svm_rbf_v2.toml", args, dry_run=False, resume_run_id=None, runs_root=runs_root,
    )
    cnn_cmd = rfs.build_command(
        "cnn", "cnn_v2.toml", args, dry_run=False, resume_run_id=None, runs_root=runs_root,
    )

    # La SVM tambien usa cache de caracteristicas: --cache-root no es
    # exclusivo de CNN/CRNN (antes solo se pasaba en la rama cnn/crnn).
    assert "--cache-root" in svm_cmd
    assert svm_cmd[svm_cmd.index("--cache-root") + 1] == str(cache_root)
    assert "--cache-root" in cnn_cmd
    assert cnn_cmd[cnn_cmd.index("--cache-root") + 1] == str(cache_root)


def test_resume_sequence_without_any_previous_sequence_fails_clearly(tmp_path):
    rc = rfs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence"])
    assert rc == 2
