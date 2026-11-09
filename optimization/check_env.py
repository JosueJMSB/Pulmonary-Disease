"""Verifica el entorno antes de ejecutar nada.

    python -m optimization.check_env --role server
    python -m optimization.check_env --role laptop

- Importa cada dependencia del rol e informa su version.
- Comprueba, en un proceso aparte, que importar ``optimization.infer`` (y el
  flujo de audio que usa) NO carga PyTorch.
- En el rol ``server`` exige ademas torch y onnx, que solo usan la
  exportacion y la cuantizacion.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

LAPTOP_MODULES = ("numpy", "scipy", "pandas", "soundfile", "librosa", "sklearn", "onnxruntime", "psutil")
SERVER_EXTRA = ("torch", "onnx")
REPO_PATHS = (
    "optimization/infer.py",
    "preprocessing/config.py",
    "preprocessing/utils.py",
    "preprocessing/phase2_standardization.py",
    "preprocessing/phase3_cleaning.py",
    "preprocessing/phase4_temporal.py",
    "modeling/__init__.py",
    "modeling/features/__init__.py",
    "modeling/features/logmel.py",
    "modeling/features/acoustic.py",
    "modeling/evaluation.py",
)


def check_modules(names) -> dict:
    out = {}
    for name in names:
        try:
            module = importlib.import_module(name)
            out[name] = {"ok": True, "version": getattr(module, "__version__", "?")}
        except Exception as exc:  # noqa: BLE001
            out[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return out


def infer_loads_torch() -> dict:
    code = (
        "import sys, json\n"
        "import optimization.infer, optimization.audio_pipeline as ap\n"
        "ap.preprocessing_modules()\n"
        "import modeling.features.logmel\n"
        "print(json.dumps({'torch_loaded': 'torch' in sys.modules}))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        return {"ok": False, "error": result.stderr.strip()[-2000:]}
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    return {"ok": not payload["torch_loaded"], **payload}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=("laptop", "server"), required=True)
    args = parser.parse_args(argv)
    names = LAPTOP_MODULES + (SERVER_EXTRA if args.role == "server" else ())
    report = {
        "role": args.role,
        "python": sys.version.split()[0],
        "modules": check_modules(names),
        "repo_files": {p: (REPO_ROOT / p).is_file() for p in REPO_PATHS},
    }
    # Importar torch en este mismo proceso (rol server) no afecta a la prueba:
    # la comprobacion de infer corre en un proceso nuevo.
    report["infer_without_torch"] = infer_loads_torch()
    ok = (all(m["ok"] for m in report["modules"].values())
          and all(report["repo_files"].values())
          and report["infer_without_torch"]["ok"])
    report["status"] = "OK" if ok else "FALLA"
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
