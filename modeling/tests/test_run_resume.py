"""--resume incompatible: la ejecucion existente queda intacta.

Prueba el helper compartido ``run_experiment.open_run`` (SVM, CNN y CRNN).
No necesita PyTorch. La prueba de extremo a extremo de la CNN esta en
test_cnn.py.
"""

import json

import numpy as np
import pytest

from .. import data as dmod
from .. import run_experiment as rexp


def _snapshot(root):
    """Todas las rutas (archivos y directorios) y el contenido de cada archivo."""
    return {
        p.relative_to(root).as_posix(): (p.read_bytes() if p.is_file() else None)
        for p in sorted(root.rglob("*"))
    }


def _toy_inputs(tmp_path):
    dataset_dir = tmp_path / "data" / "TOY"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "segments.csv").write_text("segment_id\nS0\n", encoding="utf-8")
    np.save(dataset_dir / "segments_no_dn.npy", np.zeros((1, 4), dtype=np.float32))
    return tmp_path / "data", [dmod.ConditionSpec("TOY", "main_no_dn", "no_dn", False)]


def _existing_run(runs_root, model, run_id, stored_fingerprint):
    run_root = runs_root / model / run_id
    staging = run_root / "datasets" / "TOY" / "main_no_dn" / "fold_01_staging"
    staging.mkdir(parents=True)
    (staging / "parcial.csv").write_text("a\n1\n", encoding="utf-8")
    (run_root / "status.json").write_text(json.dumps({"status": "PARTIAL"}) + "\n", encoding="utf-8")
    (run_root / "run.log").write_text("linea original del log\n", encoding="utf-8")
    rexp.write_run_fingerprint(run_root, stored_fingerprint)
    return run_root


def test_incompatible_resume_leaves_run_untouched(tmp_path):
    cfg = dmod.load_config()
    data_root, specs = _toy_inputs(tmp_path)
    stored = rexp.build_run_fingerprint(cfg, data_root, specs, "TOY", "main")
    run_root = _existing_run(tmp_path / "runs", "svm_rbf", "RUN1", stored)
    before = _snapshot(run_root)

    incompatible = rexp.build_run_fingerprint(cfg, data_root, specs, "TOY", "denoising_ablation")
    with pytest.raises(RuntimeError, match="--resume rechazado"):
        rexp.open_run(tmp_path / "runs", "svm_rbf", "RUN1", incompatible)

    assert _snapshot(run_root) == before


def test_compatible_resume_is_located_without_writing(tmp_path):
    cfg = dmod.load_config()
    data_root, specs = _toy_inputs(tmp_path)
    stored = rexp.build_run_fingerprint(cfg, data_root, specs, "TOY", "main")
    run_root = _existing_run(tmp_path / "runs", "svm_rbf", "RUN1", stored)
    before = _snapshot(run_root)

    located, run_id = rexp.open_run(tmp_path / "runs", "svm_rbf", "RUN1", stored)

    assert (located, run_id) == (run_root, "RUN1")
    # open_run no escribe: la limpieza de staging y el nuevo estado los hace
    # quien llama (art.finalize_resume), y solo despues de este punto.
    assert _snapshot(run_root) == before
