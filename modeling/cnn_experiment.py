"""Orquestacion de ``--model cnn`` (PLAN-CNN.md).

Mismo protocolo que la SVM -mismos pacientes, folds, condiciones, agregacion,
metricas, bootstrap, linea base Dummy, staging atomico, huella y --resume-,
con lo especifico de la red en ``models/cnn.py``. Se invoca desde
``run_experiment.main`` cuando ``--model cnn``; los resultados van a
``<runs-root>/cnn/<run_id>/`` y nunca tocan las ejecuciones de la SVM.

Los folds y las condiciones se procesan en secuencia sobre una sola GPU, para
respetar el servidor compartido.

Este modulo solo conserva el despachador ``run_cnn``: lo comun a CNN/CRNN vive
en ``protocols/cnn_common.py`` y cada protocolo en ``protocols/oof_v1/cnn.py`` y
``protocols/fold_aware_v2/cnn.py``. Todos los nombres que antes se definian
aqui se reexportan, para que ``cexp.<nombre>`` siga funcionando.
"""

from __future__ import annotations

from .protocols.cnn_common import (  # noqa: F401 - reexportados
    CNN_FINGERPRINT_SECTIONS,
    CRNN_FINGERPRINT_SECTIONS,
    MODEL_NAME,
    SHARED_PROTOCOL_SECTIONS,
    _append_torch_environment,
    _discard_staging,
    _final_selection,
    _find_ok,
    _fold_configurations,
    _load_cnn_fold_outputs,
    _write_csv,
    _write_json,
    config_consistency_checks,
    fingerprint_sections,
    order_specs,
    smoke_test_config,
    write_cnn_fold_artifacts,
)
from .protocols.fold_aware_v2.cnn import (  # noqa: F401 - reexportados
    FOLDED_PAIRED_COMPARISONS,
    _folded_condition_input_hashes,
    _write_run_tables_folded,
    dry_run_folded,
    run_cnn_condition_folded,
    run_folded_cnn,
    summarize_cnn_condition_folded,
)
from .protocols.fold_aware_v2.config import is_folded_protocol
from .protocols.oof_v1.cnn import (  # noqa: F401 - reexportados
    PAIRED_COMPARISONS,
    _condition_input_hashes,
    _ensure_final_model,
    _write_run_tables,
    dry_run,
    run_cnn_condition,
    run_oof_cnn,
    summarize_cnn_condition,
)


def run_cnn(args, cfg: dict) -> int:
    if is_folded_protocol(cfg):
        return run_folded_cnn(args, cfg)
    return run_oof_cnn(args, cfg)
