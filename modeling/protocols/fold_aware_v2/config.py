"""Deteccion y requisitos de configuracion del protocolo fold-aware (v2)."""

from __future__ import annotations

from pathlib import Path

from ... import data as dmod


# ---------------------------------------------------------------------------
# Protocolo fold-aware (v2): el denoising se recalibra por fold (ver
# preprocessing/fold_denoising.py) en vez de una sola vez de forma global, asi
# que cada fold reconstruye su propia cache de features/Log-Mel y ya no hace
# falta mantener a calibration_patient siempre en train. Un TOML activa este
# protocolo declarando [folds] patient_folds_csv; sin esa seccion, el
# comportamiento de run_experiment.py/cnn_experiment.py es exactamente el de
# antes.
# ---------------------------------------------------------------------------

def is_folded_protocol(cfg: dict) -> bool:
    return "patient_folds_csv" in cfg.get("folds", {})


def _patient_folds_paths(cfg: dict) -> tuple[Path, Path]:
    folds_cfg = cfg.get("folds", {})
    if "patient_folds_csv" not in folds_cfg:
        raise RuntimeError("el protocolo fold-aware (v2) requiere [folds] patient_folds_csv en el TOML")
    csv_path = dmod.REPO_ROOT / folds_cfg["patient_folds_csv"]
    manifest_path = dmod.REPO_ROOT / folds_cfg.get(
        "patient_folds_manifest", "modeling/data/patient_folds_manifest.json"
    )
    return csv_path, manifest_path


def _validate_final_model_disabled(cfg: dict) -> None:
    """El protocolo fold-aware nunca entrena un modelo definitivo: no hay una
    unica cache de todos los pacientes sobre la que hacerlo (cada fold tiene
    su propio denoising recalibrado), y esta etapa queda para un benchmarking
    posterior. ``[final_model] enabled = false`` debe estar explicito en el
    TOML -no es un valor por defecto- para que quede documentado y forme
    parte de la huella de la ejecucion (ver FOLDED_RUN_FINGERPRINT_SECTIONS).
    """
    enabled = cfg.get("final_model", {}).get("enabled")
    if enabled is not False:
        raise RuntimeError(
            "el protocolo fold-aware (v2) exige '[final_model]\\nenabled = false' explicito "
            f"en el TOML (valor actual: {enabled!r}); este protocolo no entrena modelo definitivo"
        )
