"""Orquestacion de ``--model crnn``.

La CRNN sigue exactamente el protocolo de la CNN -folds, pesos, normalizacion,
busqueda, seleccion por validation, reajuste, test unico, OOF, bootstrap,
linea base Dummy, staging atomico, huella y --resume-, asi que reutiliza
``cnn_experiment.run_cnn``. Lo especifico de la CRNN esta en:

- ``models/crnn.py``: la red;
- ``configs/oof_v1/crnn.toml``: ``[model] architecture = "crnn"`` y ``[crnn]``;
- ``cnn_experiment.config_consistency_checks``: rechaza ejecutar si crnn.toml
  difiere de cnn.toml en alguna seccion del protocolo;
- ``cnn_experiment.fingerprint_sections``: la huella incluye [model]/[crnn].

Los resultados van a ``<runs-root>/crnn/<run_id>/``; nunca se tocan
``runs/svm_rbf`` ni ``runs/cnn``.
"""

from __future__ import annotations

import sys

from . import cnn_experiment
from . import data as dmod


def run_crnn(args, cfg: dict) -> int:
    architecture = dmod.model_architecture(cfg)
    if architecture != "crnn":
        print(
            f'--model crnn necesita un TOML con [model] architecture = "crnn" (se recibio {architecture!r})',
            file=sys.stderr,
        )
        return 2
    if args.force_features:
        # La cache Log-Mel es la misma que usa la CNN: si es valida, se reutiliza
        # tal cual, y si falta o esta desactualizada se regenera sola. Forzar su
        # regeneracion desde la CRNN sobrescribiria la cache de otro modelo.
        print(
            "--force-features no se admite con --model crnn: la cache Log-Mel es compartida "
            "con la CNN. Si hace falta regenerarla, hagalo con --model cnn.",
            file=sys.stderr,
        )
        return 2
    return cnn_experiment.run_cnn(args, cfg)
