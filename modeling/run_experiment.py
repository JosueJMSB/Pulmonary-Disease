"""CLI unico de los experimentos de modelado (SVM-RBF y CNN).

    python -u -m modeling.run_experiment \
        --model svm_rbf --dataset all --experiment all --n-jobs 4 \
        --data-root <ruta> --runs-root <ruta>

    python -u -m modeling.run_experiment \
        --model cnn --dataset all --experiment all --device cuda:0 \
        --data-root <ruta> --runs-root <ruta> --cache-root <ruta>

``--model cnn`` carga ``configs/oof_v1/cnn.toml`` y despacha a
``modeling.cnn_experiment``; ``--model svm_rbf`` sigue el flujo de
``protocols/oof_v1/svm.py``. Un TOML con ``protocol = "holdout_cv_v3"`` (carpeta
``configs/holdout_final/``) despacha, para los tres modelos, a
``modeling.holdout_svm`` / ``modeling.holdout_cnn``; los TOML de ``oof_v1`` y
``fold_aware_v2`` no se ven afectados. No hay rutas personales: ``--data-root``/``--runs-root``/
``--cache-root``, o las variables ``PULMONARY_DATA_ROOT`` /
``PULMONARY_RUNS_ROOT`` / ``PULMONARY_CACHE_ROOT``, resuelven donde viven los
datos y los resultados en cada maquina (ver ``modeling.data.resolve_path``).

Este modulo solo conserva ``main()``: la CLI, la huella de ``--resume``, el
``--dry-run`` y las marcas de fallo viven en ``modeling/runner/``, y cada
protocolo en ``modeling/protocols/``. Todos los nombres que antes se definian
aqui se reexportan, para que ``rexp.<nombre>`` siga funcionando.
"""

from __future__ import annotations

import sys

from . import data as dmod
from .protocols.fold_aware_v2.config import (  # noqa: F401 - reexportados
    _patient_folds_paths,
    _validate_final_model_disabled,
    is_folded_protocol,
)
from .protocols.fold_aware_v2.svm import (  # noqa: F401 - reexportados
    _write_folded_run_tables,
    run_condition_folded,
    run_folded_svm,
    summarize_condition_folded,
)
from .protocols.oof_v1.svm import (  # noqa: F401 - reexportados
    ABLATION_PAIRS,
    run_condition,
    run_oof_svm,
    summarize_condition,
)
from .protocols.svm_common import _coerce_gamma_column, _load_fold_outputs, _smoke_test_config  # noqa: F401
from .runner.cli import (  # noqa: F401 - reexportados
    CONFIG_PATHS,
    DATASET_CHOICES,
    DEVICE_PATTERN,
    EXPERIMENT_CHOICES,
    MODEL_CHOICES,
    _device_arg,
    parse_args,
    plan_conditions,
    resolve_datasets,
    resolve_experiments,
)
from .runner.dry_run import dry_run_check, dry_run_check_folded  # noqa: F401
from .runner.fingerprint import (  # noqa: F401 - reexportados
    FOLDED_RUN_FINGERPRINT_SECTIONS,
    RUN_FINGERPRINT_CONFIG_SECTIONS,
    _config_run_fingerprint,
    _folded_input_hashes_for_specs,
    _input_hashes_for_specs,
    _run_fingerprint_path,
    build_folded_run_fingerprint,
    build_run_fingerprint,
    open_run,
    verify_run_fingerprint,
    write_run_fingerprint,
)
from .runner.fold_markers import (  # noqa: F401 - reexportados
    _clear_fold_failure_marker,
    _fold_failure_marker,
    _write_fold_failure,
)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = dmod.load_config(args.config or CONFIG_PATHS[args.model])

    if args.experiment != "all" and args.experiment not in cfg["experiments"]:
        print(f"--experiment {args.experiment} no esta definido para --model {args.model}", file=sys.stderr)
        return 2

    if args.dataset != "all" and args.dataset not in dmod.dataset_names(cfg):
        print(
            f"--dataset {args.dataset} no esta definido en {args.config or CONFIG_PATHS[args.model]} "
            f"(datasets disponibles: {dmod.dataset_names(cfg)})",
            file=sys.stderr,
        )
        return 2

    if dmod.is_holdout_protocol(cfg):
        # Protocolo holdout + validacion cruzada v3 (PLAN-EXPERIMENTO FINAL.md):
        # se activa con ``protocol = "holdout_cv_v3"`` en el TOML. Imports
        # diferidos: el entorno de la SVM no necesita PyTorch.
        if args.model == "svm_rbf":
            from .holdout_svm import run_holdout_svm

            return run_holdout_svm(args, cfg)
        architecture = dmod.model_architecture(cfg)
        if architecture != args.model:
            print(f"--model {args.model} con un TOML de arquitectura {architecture!r}", file=sys.stderr)
            return 2
        from .holdout_cnn import run_holdout_cnn

        return run_holdout_cnn(args, cfg)

    if args.model in ("cnn", "crnn"):
        # El TOML debe describir la misma red que --model: evita, por ejemplo,
        # entrenar una CRNN y guardarla bajo runs/cnn con --config cnn.toml.
        architecture = dmod.model_architecture(cfg)
        if architecture != args.model:
            print(f"--model {args.model} con un TOML de arquitectura {architecture!r}", file=sys.stderr)
            return 2
        # Imports diferidos: el entorno de la SVM no necesita PyTorch instalado.
        if args.model == "crnn":
            from .crnn_experiment import run_crnn

            return run_crnn(args, cfg)
        from .cnn_experiment import run_cnn

        return run_cnn(args, cfg)

    if is_folded_protocol(cfg):
        return run_folded_svm(args, cfg)

    return run_oof_svm(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
