"""Argumentos de la CLI de ``run_experiment`` y planificacion de condiciones."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from .. import data as dmod

MODEL_CHOICES = ("svm_rbf", "cnn", "crnn")
DATASET_CHOICES = ("ICBHI", "FRAIWAN_Extended", "COMBINED", "all")
EXPERIMENT_CHOICES = ("main", "denoising_ablation", "augmentation_ablation", "all")
CONFIG_PATHS = {
    "svm_rbf": dmod.DEFAULT_CONFIG_PATH,
    "cnn": dmod.CNN_CONFIG_PATH,
    "crnn": dmod.CRNN_CONFIG_PATH,
}
DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda|cuda:\d+)$")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _device_arg(value: str) -> str:
    if not DEVICE_PATTERN.match(value):
        raise argparse.ArgumentTypeError("use auto, cpu, cuda o cuda:N")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y evalua modelos COPD vs Control (SVM-RBF, CNN o CRNN), dataset por dataset."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, required=True,
        help="COMBINED requiere un --config con [datasets.COMBINED] (ver configs/<protocolo>/*_combined.toml).",
    )
    parser.add_argument(
        "--experiment", choices=EXPERIMENT_CHOICES, required=True,
        help="augmentation_ablation solo existe para la CNN y la CRNN.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Procesos para la busqueda de la SVM (por defecto 1). CNN y CRNN no lo usan.",
    )
    parser.add_argument(
        "--device", type=_device_arg, default="auto",
        help="Solo CNN/CRNN: auto, cpu, cuda o cuda:N (por defecto auto).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Solo CNN/CRNN: workers del DataLoader (por defecto 2).",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument(
        "--cache-root", type=Path, default=None,
        help="Cache de caracteristicas (SVM) o Log-Mel compartida (CNN/CRNN); por defecto "
             "PULMONARY_CACHE_ROOT o la ruta de [paths] del TOML del modelo.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Ruta alternativa al TOML (por defecto configs/oof_v1/svm_rbf.toml, "
             "configs/oof_v1/cnn.toml o configs/oof_v1/crnn.toml segun --model).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Valida formas, hashes, conteos y folds; no entrena.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="SVM: fold 1 y una combinacion (C, gamma). CNN/CRNN: fold 1, una configuracion y dos epocas.",
    )
    parser.add_argument(
        "--force-features", action="store_true",
        help="Regenera la cache de caracteristicas (SVM) o de Log-Mel (solo CNN; la CRNN lo rechaza "
             "para no sobrescribir la cache compartida).",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
        help="Continua una ejecucion existente (RUN_ID o, sin valor, la mas reciente).",
    )
    return parser.parse_args(argv)


def resolve_datasets(cfg: dict, dataset_arg: str) -> list[str]:
    return dmod.dataset_names(cfg) if dataset_arg == "all" else [dataset_arg]


def resolve_experiments(experiment_arg: str, cfg: dict) -> list[str]:
    """``all`` son todos los experimentos del TOML del modelo, en su orden:
    main y denoising_ablation para la SVM; ademas augmentation_ablation para
    la CNN."""
    return dmod.experiment_names(cfg) if experiment_arg == "all" else [experiment_arg]


def plan_conditions(cfg: dict, datasets: list[str], experiments: list[str]) -> list[dmod.ConditionSpec]:
    """Condiciones (dataset, condicion) a ejecutar, sin duplicar cuando una
    misma condicion aparece en mas de un experimento (el caso de Fraiwan
    main_no_dn, reutilizada en la ablacion)."""
    seen: dict[tuple[str, str], dmod.ConditionSpec] = {}
    for dataset in datasets:
        for experiment in experiments:
            for spec in dmod.experiment_condition_specs(cfg, experiment, dataset):
                seen[(spec.dataset, spec.condition)] = spec
    return list(seen.values())
