"""Lanzador secuencial SVM -> CNN -> CRNN: salida en vivo linea por linea,
persistencia inmediata del run_id (incluso si el entrenamiento se
interrumpe), resolucion de --runs-root/PULMONARY_RUNS_ROOT y el default de
--num-workers.

``subprocess.Popen`` se reemplaza por un doble de prueba: esto nunca invoca
modeling.run_experiment de verdad ni toca datos reales.
"""

import json

import pytest

from .. import run_combined_sequence as rcs


class FakePopen:
    """Simula subprocess.Popen para run_combined_sequence.run_step.

    ``script`` mapea ``(phase, model) -> {"returncode", "run_id",
    "interrupt_after_run_id"}``; una entrada ausente equivale a exito.
    Para una llamada de entrenamiento, ``stdout`` primero produce la linea
    de log que run_experiment.main imprime justo al abrir la ejecucion
    (``... INFO run_id=... dataset=...``), igual que en la vida real, mucho
    antes de la linea final ``run_id=... estado=...``. Si
    ``interrupt_after_run_id`` es True, la iteracion levanta
    ``KeyboardInterrupt`` justo despues de esa primera linea, simulando una
    interrupcion a mitad de entrenamiento.
    """

    def __init__(self, command, script):
        self.command = command
        model = command[command.index("--model") + 1]
        is_dry_run = "--dry-run" in command
        phase = "dry_run" if is_dry_run else "train"
        outcome = script.get((phase, model), {})
        self._returncode = outcome.get("returncode", 0)
        self._run_id = outcome.get("run_id", f"RUN_{model}")
        self._interrupt = outcome.get("interrupt_after_run_id", False)
        self._is_dry_run = is_dry_run
        self.returncode = None
        self.stdout = self._make_stdout()

    def _make_stdout(self):
        if self._is_dry_run:
            yield "veredicto: OK\n" if self._returncode == 0 else "veredicto: REVISAR\n"
            return
        yield f"2026-09-21 00:00:00,000 INFO run_id={self._run_id} dataset=COMBINED experiment=all\n"
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


def _models_called(calls: list[list[str]]) -> list[str]:
    return [cmd[cmd.index("--model") + 1] for cmd in calls]


def _read_status(tmp_path) -> dict:
    status_path = next((tmp_path / "combined_sequences").glob("*/sequence_status.json"))
    return json.loads(status_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Orden, deteccion de fallos y --resume-sequence (comportamiento ya existente)
# ---------------------------------------------------------------------------

def test_dry_run_only_visits_all_three_models_in_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rcs.main(["--dry-run-only", "--runs-root", str(tmp_path)])

    assert rc == 0
    assert _models_called(calls) == ["svm_rbf", "cnn", "crnn"]
    assert _read_status(tmp_path)["final_status"] == "DRY_RUN_OK"


def test_dry_run_failure_stops_before_later_models_and_never_trains(tmp_path, monkeypatch):
    calls = []
    script = {("dry_run", "cnn"): {"returncode": 1}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rcs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 1
    assert _models_called(calls) == ["svm_rbf", "cnn"]
    status = _read_status(tmp_path)
    assert status["final_status"] == "FAILED"
    assert all(step["phase"] == "dry_run" for step in status["steps"])


def test_execute_trains_in_order_after_dry_run_passes(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rcs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 0
    assert _models_called(calls) == ["svm_rbf", "cnn", "crnn", "svm_rbf", "cnn", "crnn"]
    status = _read_status(tmp_path)
    assert status["final_status"] == "COMPLETED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [s["status"] for s in train_steps] == ["COMPLETED", "COMPLETED", "COMPLETED"]


def test_training_failure_stops_sequence_and_records_run_id(tmp_path, monkeypatch):
    calls = []
    script = {("train", "cnn"): {"returncode": 1, "run_id": "RUN_CNN_1"}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rcs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == 1
    assert _models_called(calls) == ["svm_rbf", "cnn", "crnn", "svm_rbf", "cnn"]
    status = _read_status(tmp_path)
    assert status["final_status"] == "FAILED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [(s["model"], s["status"]) for s in train_steps] == [
        ("svm_rbf", "COMPLETED"), ("cnn", "FAILED"),
    ]
    assert train_steps[1]["run_id"] == "RUN_CNN_1"


def test_resume_sequence_skips_completed_and_resumes_failed_model(tmp_path, monkeypatch):
    calls1 = []
    script1 = {("train", "cnn"): {"returncode": 1, "run_id": "RUN_CNN_1"}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script1, calls1))
    rc1 = rcs.main(["--execute", "--runs-root", str(tmp_path)])
    assert rc1 == 1
    sequence_id = next((tmp_path / "combined_sequences").iterdir()).name

    calls2 = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls2))
    rc2 = rcs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence", sequence_id])

    assert rc2 == 0
    assert _models_called(calls2) == ["svm_rbf", "cnn", "crnn", "cnn", "crnn"]

    cnn_resume_call = calls2[3]
    assert "--resume" in cnn_resume_call
    assert cnn_resume_call[cnn_resume_call.index("--resume") + 1] == "RUN_CNN_1"
    svm_train_call = calls1[3]
    assert "--resume" not in svm_train_call

    status = _read_status(tmp_path)
    assert status["final_status"] == "COMPLETED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [(s["model"], s["status"]) for s in train_steps] == [
        ("svm_rbf", "COMPLETED"), ("cnn", "FAILED"),
        ("svm_rbf", "SKIPPED"), ("cnn", "COMPLETED"), ("crnn", "COMPLETED"),
    ]


def test_resume_sequence_without_id_uses_the_latest_one(tmp_path, monkeypatch):
    script = {("train", "crnn"): {"returncode": 1, "run_id": "RUN_CRNN_1"}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script, []))
    rc1 = rcs.main(["--execute", "--runs-root", str(tmp_path)])
    assert rc1 == 1

    calls2 = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls2))
    rc2 = rcs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence"])

    assert rc2 == 0
    assert _models_called(calls2) == ["svm_rbf", "cnn", "crnn", "crnn"]


def test_resume_sequence_without_any_previous_sequence_fails_clearly(tmp_path, capsys):
    rc = rcs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "resume-sequence" in captured.err


def test_dry_run_only_flag_overrides_execute(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))
    rc = rcs.main(["--execute", "--dry-run-only", "--runs-root", str(tmp_path)])
    assert rc == 0
    assert _models_called(calls) == ["svm_rbf", "cnn", "crnn"]


# ---------------------------------------------------------------------------
# 1. Salida y log en tiempo real + run_id conservado ante una interrupcion
# ---------------------------------------------------------------------------

def test_output_lines_are_mirrored_live_and_written_to_log(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rcs.main(["--dry-run-only", "--runs-root", str(tmp_path)])

    log_path = next((tmp_path / "combined_sequences").glob("*/sequence.log"))
    log_text = log_path.read_text(encoding="utf-8")
    assert log_text.count("veredicto: OK") == 3  # una por cada dry-run (svm, cnn, crnn)
    captured = capsys.readouterr()
    assert "veredicto: OK" in captured.out  # mismo progreso reflejado en la consola


def test_training_run_id_is_captured_from_the_early_log_line(tmp_path, monkeypatch):
    # run_experiment registra run_id= con logger.info justo al abrir la
    # ejecucion, mucho antes de la linea final "run_id=... estado=...": el
    # lanzador debe quedarse con ese primer valor.
    calls = []
    script = {("train", "svm_rbf"): {"run_id": "RUN_SVM_EARLY"}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rcs.main(["--execute", "--runs-root", str(tmp_path)])

    status = _read_status(tmp_path)
    svm_train_step = next(s for s in status["steps"] if s["phase"] == "train" and s["model"] == "svm_rbf")
    assert svm_train_step["run_id"] == "RUN_SVM_EARLY"


def test_interrupted_training_preserves_run_id_and_marks_sequence_interrupted(tmp_path, monkeypatch):
    calls = []
    script = {("train", "cnn"): {"run_id": "RUN_CNN_INTERRUPTED", "interrupt_after_run_id": True}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script, calls))

    rc = rcs.main(["--execute", "--runs-root", str(tmp_path)])

    assert rc == rcs.EXIT_INTERRUPTED
    # Nunca llega a crnn: la secuencia se detiene en cuanto cnn se interrumpe.
    assert _models_called(calls) == ["svm_rbf", "cnn", "crnn", "svm_rbf", "cnn"]
    status = _read_status(tmp_path)
    assert status["final_status"] == "INTERRUPTED"
    train_steps = [s for s in status["steps"] if s["phase"] == "train"]
    assert [(s["model"], s["status"]) for s in train_steps] == [
        ("svm_rbf", "COMPLETED"), ("cnn", "INTERRUPTED"),
    ]
    assert train_steps[1]["run_id"] == "RUN_CNN_INTERRUPTED"


def test_resume_after_interrupted_training_reuses_the_captured_run_id(tmp_path, monkeypatch):
    calls1 = []
    script1 = {("train", "cnn"): {"run_id": "RUN_CNN_X", "interrupt_after_run_id": True}}
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory(script1, calls1))
    rc1 = rcs.main(["--execute", "--runs-root", str(tmp_path)])
    assert rc1 == rcs.EXIT_INTERRUPTED
    sequence_id = next((tmp_path / "combined_sequences").iterdir()).name

    calls2 = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls2))
    rc2 = rcs.main(["--execute", "--runs-root", str(tmp_path), "--resume-sequence", sequence_id])

    assert rc2 == 0
    cnn_train_calls = [
        c for c in calls2 if c[c.index("--model") + 1] == "cnn" and "--dry-run" not in c
    ]
    assert len(cnn_train_calls) == 1
    assert "--resume" in cnn_train_calls[0]
    assert cnn_train_calls[0][cnn_train_calls[0].index("--resume") + 1] == "RUN_CNN_X"

    status = _read_status(tmp_path)
    assert status["final_status"] == "COMPLETED"


# ---------------------------------------------------------------------------
# 2. --runs-root > PULMONARY_RUNS_ROOT > valor por defecto
# ---------------------------------------------------------------------------

def test_resolve_runs_root_prefers_cli_over_env(tmp_path, monkeypatch):
    monkeypatch.setenv("PULMONARY_RUNS_ROOT", str(tmp_path / "from_env"))
    resolved = rcs.resolve_runs_root(tmp_path / "from_cli")
    assert resolved == (tmp_path / "from_cli").resolve()


def test_resolve_runs_root_uses_env_var_when_cli_not_given(tmp_path, monkeypatch):
    monkeypatch.setenv("PULMONARY_RUNS_ROOT", str(tmp_path / "from_env"))
    resolved = rcs.resolve_runs_root(None)
    assert resolved == (tmp_path / "from_env").resolve()


def test_resolve_runs_root_falls_back_to_repo_default(monkeypatch):
    monkeypatch.delenv("PULMONARY_RUNS_ROOT", raising=False)
    resolved = rcs.resolve_runs_root(None)
    assert resolved == (rcs.REPO_ROOT / "modeling" / "runs").resolve()


def test_main_uses_env_runs_root_when_cli_flag_is_absent(tmp_path, monkeypatch):
    env_root = tmp_path / "from_env"
    monkeypatch.setenv("PULMONARY_RUNS_ROOT", str(env_root))
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rcs.main(["--dry-run-only"])

    assert rc == 0
    assert (env_root / "combined_sequences").is_dir()
    for call in calls:
        assert "--runs-root" in call
        assert call[call.index("--runs-root") + 1] == str(env_root.resolve())


def test_main_cli_runs_root_overrides_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("PULMONARY_RUNS_ROOT", str(tmp_path / "from_env"))
    cli_root = tmp_path / "from_cli"
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rc = rcs.main(["--dry-run-only", "--runs-root", str(cli_root)])

    assert rc == 0
    assert (cli_root / "combined_sequences").is_dir()
    assert not (tmp_path / "from_env").exists()
    for call in calls:
        assert call[call.index("--runs-root") + 1] == str(cli_root.resolve())


# ---------------------------------------------------------------------------
# 3. --num-workers por defecto en 0
# ---------------------------------------------------------------------------

def test_num_workers_default_is_zero():
    args = rcs.parse_args([])
    assert args.num_workers == 0


def test_num_workers_zero_passed_to_cnn_and_crnn_but_not_svm(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rcs.main(["--dry-run-only", "--runs-root", str(tmp_path)])

    for call in calls:
        model = call[call.index("--model") + 1]
        if model in ("cnn", "crnn"):
            assert "--num-workers" in call
            assert call[call.index("--num-workers") + 1] == "0"
        else:
            assert "--num-workers" not in call


def test_explicit_num_workers_is_respected(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rcs.subprocess, "Popen", _fake_popen_factory({}, calls))

    rcs.main(["--dry-run-only", "--runs-root", str(tmp_path), "--num-workers", "4"])

    cnn_call = next(c for c in calls if c[c.index("--model") + 1] == "cnn")
    assert cnn_call[cnn_call.index("--num-workers") + 1] == "4"
