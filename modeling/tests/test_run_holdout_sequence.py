"""Lanzador secuencial del protocolo holdout-v3: orden de los 6 pasos (cada
modelo primero individual, luego COMBINED), deteccion de fallos, persistencia
del run_id ante una interrupcion, --resume-sequence y el alcance del lanzador
(no construye splits ni preprocesamiento, no elige arquitectura, no reentrena).

``subprocess.Popen`` se reemplaza por un doble: esto nunca invoca
modeling.run_experiment de verdad ni toca datos reales.
"""

import json

from .. import data as dmod
from .. import run_holdout_sequence as rhs


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
        yield f"2026-09-26 00:00:00,000 INFO run_id={self._run_id} protocolo=holdout_cv_v3\n"
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
    return [
        (cmd[cmd.index("--model") + 1], _config_basename(cmd[cmd.index("--config") + 1])) for cmd in calls
    ]


def _read_status(tmp_path) -> dict:
    status_path = next((tmp_path / "holdout_sequences").glob("*/sequence_status.json"))
    return json.loads(status_path.read_text(encoding="utf-8"))


EXPECTED_ORDER = list(rhs.STEPS)


def test_steps_are_the_six_v3_configs_in_the_required_order():
    assert EXPECTED_ORDER == [
        ("svm_rbf", "svm_rbf_v3.toml"), ("svm_rbf", "svm_rbf_combined_v3.toml"),
        ("cnn", "cnn_v3.toml"), ("cnn", "cnn_combined_v3.toml"),
        ("crnn", "crnn_v3.toml"), ("crnn", "crnn_combined_v3.toml"),
    ]
    assert rhs.CONFIGS_DIR == dmod.HOLDOUT_FINAL_CONFIGS_DIR
    for _, config_name in EXPECTED_ORDER:
        assert (rhs.CONFIGS_DIR / config_name).is_file()


def test_dry_run_only_visits_all_six_steps_in_order_and_never_trains(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rhs.main(["--dry-run-only", "--execute", "--runs-root", str(tmp_path)])

    assert rc == 0
    assert _models_and_configs(calls) == EXPECTED_ORDER
    assert all("--dry-run" in c for c in calls)
    assert _read_status(tmp_path)["final_status"] == "DRY_RUN_OK"


def test_without_execute_the_sequence_only_validates(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory({}, calls))
    assert rhs.main(["--runs-root", str(tmp_path)]) == 0
    assert all("--dry-run" in c for c in calls) and len(calls) == 6


def test_dry_run_failure_stops_before_later_steps(tmp_path, monkeypatch):
    script = {("dry_run", "cnn", "cnn_v3.toml"): {"returncode": 1}}
    calls = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rhs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 1
    assert _models_and_configs(calls) == EXPECTED_ORDER[:3]          # se detiene en cnn_v3 (paso 3 de 6)
    status = _read_status(tmp_path)
    assert status["final_status"] == "FAILED"
    assert all(step["phase"] == "dry_run" for step in status["steps"])   # nunca se entreno


def test_execute_trains_all_six_steps_in_order_after_dry_run_passes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rhs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 0
    assert _models_and_configs(calls) == EXPECTED_ORDER + EXPECTED_ORDER
    status = _read_status(tmp_path)
    assert status["final_status"] == "COMPLETED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [s["status"] for s in train_steps] == ["COMPLETED"] * 6
    assert all(s["run_id"] for s in train_steps)                     # los run_id quedan registrados


def test_training_failure_stops_sequence_and_records_run_id(tmp_path, monkeypatch):
    script = {("train", "cnn", "cnn_combined_v3.toml"): {"returncode": 1, "run_id": "RUN_CNN_COMBINED_1"}}
    calls = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rhs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 1
    assert _models_and_configs(calls) == EXPECTED_ORDER + EXPECTED_ORDER[:4]   # nunca llega a la CRNN
    status = _read_status(tmp_path)
    assert status["final_status"] == "FAILED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [s["status"] for s in train_steps] == ["COMPLETED", "COMPLETED", "COMPLETED", "FAILED"]
    assert train_steps[-1]["run_id"] == "RUN_CNN_COMBINED_1"


def test_resume_sequence_skips_completed_and_resumes_failed_step(tmp_path, monkeypatch):
    script1 = {("train", "cnn", "cnn_combined_v3.toml"): {"returncode": 1, "run_id": "RUN_CNN_COMBINED_1"}}
    calls1 = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory(script1, calls1))
    assert rhs.main(["--execute", "--runs-root", str(tmp_path)]) == 1
    sequence_id = next((tmp_path / "holdout_sequences").iterdir()).name

    calls2 = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory({}, calls2))
    rc2 = rhs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence", sequence_id])

    assert rc2 == 0
    # El dry-run se repite siempre; en el entrenamiento se omiten los tres primeros pasos
    # (ya COMPLETED) y se reanuda cnn_combined_v3 antes de seguir con la CRNN.
    assert _models_and_configs(calls2) == EXPECTED_ORDER + EXPECTED_ORDER[3:]
    resume_call = calls2[len(EXPECTED_ORDER)]
    assert resume_call[resume_call.index("--resume") + 1] == "RUN_CNN_COMBINED_1"

    statuses = [(s["model"], s["config"], s["status"]) for s in _read_status(tmp_path)["steps"] if s["phase"] == "train"]
    assert statuses == [
        ("svm_rbf", "svm_rbf_v3.toml", "COMPLETED"),
        ("svm_rbf", "svm_rbf_combined_v3.toml", "COMPLETED"),
        ("cnn", "cnn_v3.toml", "COMPLETED"),
        ("cnn", "cnn_combined_v3.toml", "FAILED"),
        ("svm_rbf", "svm_rbf_v3.toml", "SKIPPED"),
        ("svm_rbf", "svm_rbf_combined_v3.toml", "SKIPPED"),
        ("cnn", "cnn_v3.toml", "SKIPPED"),
        ("cnn", "cnn_combined_v3.toml", "COMPLETED"),
        ("crnn", "crnn_v3.toml", "COMPLETED"),
        ("crnn", "crnn_combined_v3.toml", "COMPLETED"),
    ]


def test_interrupted_training_preserves_run_id_and_marks_sequence_interrupted(tmp_path, monkeypatch):
    script = {("train", "crnn", "crnn_v3.toml"): {"run_id": "RUN_CRNN_X", "interrupt_after_run_id": True}}
    calls = []
    monkeypatch.setattr(rhs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rhs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == rhs.EXIT_INTERRUPTED
    assert _models_and_configs(calls) == EXPECTED_ORDER + EXPECTED_ORDER[:5]
    status = _read_status(tmp_path)
    assert status["final_status"] == "INTERRUPTED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert train_steps[-1]["status"] == "INTERRUPTED" and train_steps[-1]["run_id"] == "RUN_CRNN_X"


def test_resume_sequence_without_any_previous_sequence_fails_clearly(tmp_path):
    assert rhs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence"]) == 2


def test_resolve_runs_root_prefers_cli_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULMONARY_RUNS_ROOT", str(tmp_path / "from_env"))
    assert rhs.resolve_runs_root(tmp_path / "from_cli") == (tmp_path / "from_cli").resolve()
    assert rhs.resolve_runs_root(None) == (tmp_path / "from_env").resolve()


def test_num_workers_default_is_zero():
    assert rhs.parse_args([]).num_workers == 0


def test_build_command_passes_cache_root_to_all_three_models_and_device_only_to_networks(tmp_path):
    args = rhs.parse_args(["--cache-root", str(tmp_path / "cache"), "--data-root", str(tmp_path / "data"),
                           "--device", "cuda:1", "--n-jobs", "4"])
    commands = {
        model: rhs.build_command(model, f"{model}_v3.toml", args, dry_run=False, resume_run_id=None,
                                 runs_root=tmp_path / "runs")
        for model in ("svm_rbf", "cnn", "crnn")
    }
    for model, cmd in commands.items():
        assert cmd[cmd.index("--cache-root") + 1] == str(tmp_path / "cache"), model     # tambien la SVM
        assert cmd[cmd.index("--data-root") + 1] == str(tmp_path / "data"), model
        assert cmd[cmd.index("--runs-root") + 1] == str(tmp_path / "runs"), model
    assert "--device" not in commands["svm_rbf"] and commands["svm_rbf"][commands["svm_rbf"].index("--n-jobs") + 1] == "4"
    assert commands["cnn"][commands["cnn"].index("--device") + 1] == "cuda:1"
    assert "--n-jobs" not in commands["crnn"]


def test_commands_only_run_experiment_never_splits_preprocessing_or_final_stages(tmp_path):
    args = rhs.parse_args([])
    for model, config_name in rhs.STEPS:
        for dry_run in (True, False):
            cmd = rhs.build_command(model, config_name, args, dry_run, None, tmp_path)
            text = " ".join(cmd)
            assert "modeling.run_experiment" in text
            for forbidden in ("build_master_folds", "fold_denoising", "--smoke-test", "--final", "--outer"):
                assert forbidden not in text, (forbidden, text)
            assert ("--dry-run" in cmd) is dry_run
            assert "holdout_final" in cmd[cmd.index("--config") + 1].replace("\\", "/")


def test_resume_command_carries_the_run_id_only_when_not_a_dry_run(tmp_path):
    args = rhs.parse_args([])
    resumed = rhs.build_command("cnn", "cnn_v3.toml", args, False, "RUN_A", tmp_path)
    assert resumed[resumed.index("--resume") + 1] == "RUN_A"
    dry = rhs.build_command("cnn", "cnn_v3.toml", args, True, "RUN_A", tmp_path)
    assert "--resume" not in dry
