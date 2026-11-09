"""La inferencia no debe necesitar PyTorch.

Se importa el flujo de inferencia en un proceso nuevo en el que cualquier
importacion de torch falla. Si algun modulo lo importara, la prueba falla.
"""

import importlib.util
import subprocess
import sys

import pytest

from optimization import paths

REQUIRED = ("numpy", "pandas", "sklearn")
AUDIO = ("scipy", "soundfile", "librosa")

BLOCKER = """
import sys
class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch bloqueado en esta prueba")
        return None
sys.meta_path.insert(0, _BlockTorch())
"""


def _available(names):
    return all(importlib.util.find_spec(n) is not None for n in names)


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", BLOCKER + code], cwd=str(paths.REPO_ROOT),
                          capture_output=True, text=True)


@pytest.mark.skipif(not _available(REQUIRED), reason="faltan numpy, pandas o scikit-learn")
def test_inference_modules_import_without_torch():
    result = _run(
        "import optimization.bundle, optimization.parity, optimization.scoring, optimization.access_registry\n"
        "import optimization.audio_pipeline, optimization.ort_session, optimization.infer\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


@pytest.mark.skipif(not _available(REQUIRED + AUDIO), reason="faltan las librerias del flujo de audio")
def test_audio_flow_imports_without_torch():
    result = _run(
        "import optimization.audio_pipeline as ap\n"
        "ap.preprocessing_modules()\n"
        "import modeling.features.logmel, modeling.evaluation\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")
