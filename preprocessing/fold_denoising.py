"""Fase 3-4 fold-aware: denoising y TARGET_RMS recalibrados por fold.

    python fold_denoising.py --dataset-scope ICBHI --fold-id 0
    python fold_denoising.py --dataset-scope COMBINED --fold-id 3

Hoy (``phase3_cleaning.py``) los cinco parametros del denoising y el
objetivo de normalizacion se calibran UNA SOLA VEZ, de forma global, sobre un
20 % fijo de pacientes, antes de que exista ninguna particion train/test.
Este script hace lo mismo que esas dos fases, pero recalibrando por
``(dataset_scope, fold)``, usando SOLO los pacientes de train de ESE fold:

- ICBHI y COMBINED: ``nperseg``/``noverlap`` quedan fijos (256/192, nunca se
  re-barre la resolucion STFT). Se recalibran ``noise_pct``, ``oversubtraction``
  y ``spectral_floor`` con ``phase3_cleaning.sweep_denoising`` restringido a
  las grabaciones ICBHI de train de ese fold con hueco anotado suficiente.
- Fraiwan Extended: los cinco parametros quedan fijos en los valores
  globales de ``config.py`` (no tiene ciclos anotados para calibrar nada).
- Los tres: ``TARGET_RMS`` se recalibra siempre, con la mediana de RMS
  post-pasabanda sobre las grabaciones de train de ese fold (en COMBINED,
  ICBHI+Fraiwan juntos).

No modifica ``phase3_cleaning.py`` ni ``phase4_temporal.py``, ni sus salidas
(``data/interim/clean/``, ``data/final/``): reutiliza sus funciones puras
(``apply_bandpass``, ``spectral_subtract``, ``normalize``, ``sweep_denoising``,
``select_robust_candidate``, ``compute_rms_distribution``,
``recording_denoising_metrics``, ``dn_reliability``, ``segments_per_recording``,
``cycle_classification``) componiendolas aqui, sin pasar por
``process_recording_both_branches`` (que escribe WAV por rutas globales de
fase 3) ni por ``data/interim/clean/``: va directo del audio remuestreado de
fase 2 (``reports/phase2/2b_resampling.csv``) a los arreglos finales de este
fold.

La asignacion de quien es train/validation/test viene de
``modeling/data/patient_folds.csv`` (generado por
``modeling.build_master_folds``, fuera de este script): aqui no se decide
ningun fold ni ninguna clase, solo se consume esa asignacion, verificando su
hash contra ``patient_folds_manifest.json`` antes de usarla. Ese mismo hash
se guarda en el manifiesto de cada fold (``patient_folds_csv_sha256``), para
que el dry-run de entrenamiento pueda verificar que preprocesamiento y
entrenamiento usaron exactamente la misma division.

La lista de grabaciones autorizadas para cada ``dataset_scope`` es el
conjunto de ``audio_id`` unicos del ``segments.csv`` vigente del lado
``modeling/`` para esa tarea (``AUTHORIZED_SEGMENTS_CSV``), NUNCA solo el
``patient_uid``: un mismo paciente de Fraiwan puede tener grabaciones Bell/
Diaphragm ademas de su Extended, todas bajo el mismo ``patient_uid`` -
filtrar solo por paciente admitiria audio que esa tarea nunca uso. La senal
se sigue leyendo desde el audio remuestreado de fase 2, pero solo para esos
``audio_id``. Antes de procesar cada uno, se compara el archivo en disco con
``output_sha256`` de ``2b_resampling.csv``; si no coincide, se detiene el
proceso (ver ``verify_phase2_audio_hashes``).

Salida, publicada atomicamente (staging + rename, ver ``utils.prepare_staging``/
``swap_staging_into_place``):

    preprocessing/data/fold_calibrated/<dataset_scope>/fold_<00..04>/
    |-- segments.csv
    |-- segments_no_dn.npy
    |-- segments_dn.npy
    |-- preprocessing_params.json
    `-- manifest.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy

import config as cfg
import phase3_cleaning as p3
import phase4_temporal as p4
import utils as u

DATASET_SCOPES = ("ICBHI", "FRAIWAN_Extended", "COMBINED")
N_SPLITS = 5
OUTPUT_ROOT = cfg.PREPROC / "data" / "fold_calibrated"
DEFAULT_PATIENT_FOLDS_CSV = cfg.ROOT / "modeling" / "data" / "patient_folds.csv"
DEFAULT_PATIENT_FOLDS_MANIFEST = cfg.ROOT / "modeling" / "data" / "patient_folds_manifest.json"

# Lista autorizada de audio_id por dataset_scope: el segments.csv vigente del
# lado modeling/ para esa tarea, no el corpus admitido completo de fase 2 ni
# solo el patient_uid (ver docstring del modulo).
AUTHORIZED_SEGMENTS_CSV = {
    "ICBHI": cfg.ROOT / "modeling" / "data" / "copd_vs_control" / "ICBHI" / "segments.csv",
    "FRAIWAN_Extended": cfg.ROOT / "modeling" / "data" / "copd_vs_control" / "FRAIWAN_Extended" / "segments.csv",
    "COMBINED": cfg.ROOT / "modeling" / "data" / "copd_vs_control_combined" / "COMBINED" / "segments.csv",
}

REQUIRED_SEGMENT_COLUMNS = (
    "task_array_index", "source_array_index", "segment_id", "audio_id", "dataset",
    "patient_uid", "diagnosis", "target_label", "target_name", "source_dataset",
    "calibration_patient", "dn_reliable", "dn_flag_reason", "fold_id", "role",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recalibra denoising/TARGET_RMS por fold y segmenta (protocolo fold-aware v2)."
    )
    parser.add_argument("--dataset-scope", choices=DATASET_SCOPES, required=True)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--patient-folds-csv", type=Path, default=DEFAULT_PATIENT_FOLDS_CSV)
    parser.add_argument("--patient-folds-manifest", type=Path, default=DEFAULT_PATIENT_FOLDS_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Asignacion de roles: leida de patient_folds.csv, no recalculada aqui.
# ---------------------------------------------------------------------------

def load_patient_roles(csv_path: Path, manifest_path: Path, dataset_scope: str, fold_id: int) -> pd.DataFrame:
    """``patient_uid`` -> rol (train/validation/test) para ``(dataset_scope,
    fold_id)``, leyendo ``patient_folds.csv`` directamente. Mismo contrato de
    verificacion que ``modeling.splits.load_patient_folds``: si el csv no
    coincide con el hash de su manifiesto, se rechaza -ningun consumidor
    puede usar una version desincronizada sin que se detecte.
    """
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_hash = u.file_sha256(csv_path)
    expected_hash = manifest.get("patient_folds_csv_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            f"{csv_path}: sha256 {actual_hash[:12]}... no coincide con el manifiesto "
            f"{manifest_path} ({str(expected_hash)[:12]}...); regenere con "
            "'python -m modeling.build_master_folds'"
        )

    full = pd.read_csv(csv_path, dtype={"patient_uid": str})
    required = {"dataset_scope", "patient_uid", "target_label", "source_dataset", "calibration_patient", "fold_group"}
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"{csv_path}: faltan columnas {sorted(missing)}")

    table = full.loc[full["dataset_scope"] == dataset_scope].drop(columns="dataset_scope").reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{csv_path}: no hay filas para dataset_scope={dataset_scope!r}")
    table["calibration_patient"] = table["calibration_patient"].astype(bool)

    evaluable = table.loc[table["fold_group"] != -1]
    if evaluable.empty:
        raise ValueError(f"{dataset_scope}: ningun paciente con fold_group valido")
    n_splits = int(evaluable["fold_group"].max()) + 1
    if not (0 <= fold_id < n_splits):
        raise ValueError(f"fold_id={fold_id} fuera de rango para {dataset_scope} (0..{n_splits - 1})")
    if len(evaluable) != len(table):
        raise ValueError(
            f"{dataset_scope}: {len(table) - len(evaluable)} paciente(s) con fold_group=-1; "
            "el protocolo fold-aware (v2) no admite un grupo de calibracion especial"
        )

    val_group = (fold_id + 1) % n_splits
    role = np.where(
        table["fold_group"].to_numpy() == fold_id, "test",
        np.where(table["fold_group"].to_numpy() == val_group, "validation", "train"),
    )
    table = table.assign(role=role)
    return table


def _train_patient_uids(roles: pd.DataFrame) -> set:
    return set(roles.loc[roles["role"] == "train", "patient_uid"])


# ---------------------------------------------------------------------------
# Grabaciones autorizadas: audio_id del segments.csv de la tarea (modeling/),
# no solo patient_uid -ver docstring del modulo.
# ---------------------------------------------------------------------------

def load_authorized_audio_ids(dataset_scope: str) -> set:
    """``audio_id`` unicos del ``segments.csv`` vigente de ``modeling/`` para
    ``dataset_scope``: la lista autorizada de grabaciones de esta tarea."""
    csv_path = AUTHORIZED_SEGMENTS_CSV[dataset_scope]
    frame = pd.read_csv(csv_path, dtype={"audio_id": str}, usecols=["audio_id"])
    audio_ids = set(frame["audio_id"].unique())
    if not audio_ids:
        raise ValueError(f"{csv_path}: no contiene ningun audio_id")
    return audio_ids


def select_scope_recordings(admitted: pd.DataFrame, authorized_audio_ids: set, roles: pd.DataFrame) -> pd.DataFrame:
    """Grabaciones admitidas de fase 2 restringidas a la interseccion de (a)
    ``audio_id`` autorizado para esta tarea y (b) ``patient_uid`` con rol
    asignado en este fold. Se exigen ambas condiciones: (a) excluye
    grabaciones de otro filtro/dispositivo bajo el mismo paciente -p.ej.
    Bell/Diaphragm de un paciente Fraiwan cuya Extended es la unica
    autorizada-; (b) excluye pacientes fuera de este ``dataset_scope``.
    """
    scope_ids = set(roles["patient_uid"])
    scope_meta = admitted.loc[
        admitted["audio_id"].isin(authorized_audio_ids) & admitted["patient_uid"].isin(scope_ids)
    ].reset_index(drop=True)
    if scope_meta.empty:
        raise ValueError(
            "ningun audio_id autorizado con paciente de este dataset_scope tiene grabacion admitida"
        )
    return scope_meta


def verify_phase2_audio_hashes(scope_meta: pd.DataFrame) -> None:
    """Cada audio de ``scope_meta`` debe seguir siendo, byte a byte, el mismo
    que registro la fase 2 (``output_sha256`` de ``2b_resampling.csv``): si
    el archivo remuestreado cambio sin volver a correr esa fase, se detiene
    el proceso antes de calibrar o segmentar nada con datos desactualizados.
    """
    for row in scope_meta.itertuples():
        audio_path = Path(cfg.ROOT) / row.output_path
        actual_hash = u.file_sha256(audio_path)
        expected_hash = row.output_sha256
        if actual_hash != expected_hash:
            raise ValueError(
                f"{row.audio_id}: sha256 {actual_hash[:12]}... no coincide con "
                f"output_sha256 {str(expected_hash)[:12]}... de 2b_resampling.csv "
                f"({audio_path}); vuelva a ejecutar phase2_standardization.py"
            )


# ---------------------------------------------------------------------------
# Recalibracion de denoising y TARGET_RMS, solo con train de este fold.
# ---------------------------------------------------------------------------

def recalibrate_denoising(dataset_scope: str, scope_meta: pd.DataFrame, roles: pd.DataFrame) -> dict:
    """Los cinco parametros de denoising y ``TARGET_RMS`` para este fold, mas
    los ``patient_uid`` exactos usados para calibrar cada uno (para dejar
    constancia y poder verificar despues que ninguno es de validation/test).
    ``scope_meta`` ya viene filtrado por audio_id autorizado (ver
    ``select_scope_recordings``); aqui solo se restringe ademas a train.
    """
    train_ids = _train_patient_uids(roles)
    train_meta = scope_meta.loc[scope_meta["patient_uid"].isin(train_ids)].reset_index(drop=True)
    if train_meta.empty:
        raise ValueError(f"{dataset_scope}: ningun paciente de train con grabaciones admitidas")

    if dataset_scope == "FRAIWAN_Extended":
        # Fraiwan no tiene ciclos anotados: nunca alimenta el barrido, asi que
        # conserva los 5 valores fijos de config.py en todos los folds.
        denoising_params = {
            "nperseg": int(cfg.STFT_NPERSEG), "noverlap": int(cfg.STFT_NOVERLAP),
            "noise_pct": float(cfg.NOISE_PCT), "oversubtraction": float(cfg.OVERSUBTRACTION),
            "spectral_floor": float(cfg.SPECTRAL_FLOOR),
        }
        denoising_fit_patient_ids: list[str] = []
    else:
        gap_ids = p3.annotation_gap_recordings()
        icbhi_train_gap = train_meta.loc[
            (train_meta["dataset"] == "ICBHI") & train_meta["audio_id"].isin(gap_ids)
        ]
        if icbhi_train_gap.empty:
            raise ValueError(
                f"{dataset_scope}: ningun paciente ICBHI de train con hueco anotado suficiente "
                "para recalibrar el denoising en este fold"
            )
        # nperseg/noverlap fijos: no se vuelve a barrer la resolucion STFT por fold.
        sweep_df = p3.sweep_denoising(train_meta, int(cfg.STFT_NPERSEG), int(cfg.STFT_NOVERLAP))
        sweep_df["cumple_restricciones"] = sweep_df.apply(p3.satisfies_constraints, axis=1)
        best, _n_valid, _n_near, _best_objective = p3.select_robust_candidate(sweep_df)
        denoising_params = {
            "nperseg": int(cfg.STFT_NPERSEG), "noverlap": int(cfg.STFT_NOVERLAP),
            "noise_pct": float(best["noise_pct"]), "oversubtraction": float(best["oversubtraction"]),
            "spectral_floor": float(best["spectral_floor"]),
        }
        denoising_fit_patient_ids = sorted(icbhi_train_gap["patient_uid"].unique().tolist())

    rms_df = p3.compute_rms_distribution(train_meta)
    target_rms = float(rms_df["rms_post_bandpass"].median())
    target_rms_fit_patient_ids = sorted(train_meta["patient_uid"].unique().tolist())

    return {
        **denoising_params,
        "target_rms": target_rms,
        "max_gain": float(cfg.MAX_GAIN),
        "denoising_fit_patient_ids": denoising_fit_patient_ids,
        "target_rms_fit_patient_ids": target_rms_fit_patient_ids,
    }


def verify_no_leakage_into_calibration(roles: pd.DataFrame, params: dict) -> None:
    """Ningun paciente de validation/test de este fold puede haber sido usado
    para calibrar el denoising o TARGET_RMS de ese mismo fold."""
    non_train = set(roles.loc[roles["role"] != "train", "patient_uid"])
    fit_ids = set(params["denoising_fit_patient_ids"]) | set(params["target_rms_fit_patient_ids"])
    leaked = non_train & fit_ids
    if leaked:
        raise RuntimeError(
            f"fuga de calibracion: paciente(s) de validation/test usados para calibrar "
            f"este fold: {sorted(leaked)}"
        )


# ---------------------------------------------------------------------------
# Aplicar band-pass + (denoising fijo o recalibrado) + normalizacion a TODAS
# las grabaciones del fold (train+validation+test), componiendo las
# funciones puras directamente -sin pasar por process_recording_both_branches
# ni por data/interim/clean/-.
# ---------------------------------------------------------------------------

def process_fold_recordings(scope_meta: pd.DataFrame, params: dict) -> dict:
    """Aplica banda pasante + denoising + normalizacion a TODAS las
    grabaciones de ``scope_meta`` (train+validation+test de este fold; ya
    filtrado por audio_id autorizado y patient_uid del scope, ver
    ``select_scope_recordings``)."""
    if scope_meta.empty:
        raise ValueError("ningun paciente de este dataset_scope tiene grabaciones admitidas")

    no_dn_signals: dict[str, np.ndarray] = {}
    dn_signals: dict[str, np.ndarray] = {}
    dn_reliable_map: dict[str, bool] = {}
    dn_reason_map: dict[str, str] = {}

    for _, row in scope_meta.iterrows():
        audio_id = row["audio_id"]
        x, sr = u.read_audio(Path(cfg.ROOT) / row["output_path"])
        if sr != cfg.TARGET_SR:
            raise ValueError(f"{audio_id}: frecuencia {sr} Hz, se esperaba {cfg.TARGET_SR}")

        x_bp = p3.apply_bandpass(x)
        x_dn_raw = p3.spectral_subtract(
            x_bp, params["nperseg"], params["noverlap"],
            params["noise_pct"], params["oversubtraction"], params["spectral_floor"],
        )

        metrics = p3.recording_denoising_metrics(audio_id, x_bp, x_dn_raw, params["nperseg"], params["noverlap"], sr)
        dn_reliable, dn_reason = p3.dn_reliability(metrics)
        dn_reliable_map[audio_id] = dn_reliable
        dn_reason_map[audio_id] = dn_reason

        y_no_dn, _gain, _limiter = p3.normalize(x_bp, params["target_rms"], cfg.PEAK_CEILING, params["max_gain"])
        y_dn, _gain, _limiter = p3.normalize(x_dn_raw, params["target_rms"], cfg.PEAK_CEILING, params["max_gain"])
        if y_no_dn.size != x.size or y_dn.size != x.size:
            raise ValueError(f"{audio_id}: el tamano cambio tras el procesamiento")
        if not np.isfinite(y_no_dn).all() or not np.isfinite(y_dn).all():
            raise ValueError(f"{audio_id}: NaN/Inf tras normalizar")

        no_dn_signals[audio_id] = y_no_dn.astype(np.float32)
        dn_signals[audio_id] = y_dn.astype(np.float32)

    return {
        "scope_meta": scope_meta, "no_dn_signals": no_dn_signals, "dn_signals": dn_signals,
        "dn_reliable_map": dn_reliable_map, "dn_reason_map": dn_reason_map,
    }


# ---------------------------------------------------------------------------
# Segmentacion: misma rejilla (5 s, 50 % de solape) que phase4_temporal.py,
# reutilizando sus funciones puras. Ambas ramas comparten los mismos limites
# de muestra por construccion (misma grabacion, misma rejilla).
# ---------------------------------------------------------------------------

def build_fold_segment_inventory(
    dataset_scope: str, fold_id: int, scope_meta: pd.DataFrame, roles: pd.DataFrame,
    dn_reliable_map: dict, dn_reason_map: dict, cycles_by_audio: dict,
) -> pd.DataFrame:
    role_by_patient = dict(zip(roles["patient_uid"], roles["role"]))
    label_by_patient = dict(zip(roles["patient_uid"], roles["target_label"]))
    calib_by_patient = dict(zip(roles["patient_uid"], roles["calibration_patient"]))
    source_by_patient = dict(zip(roles["patient_uid"], roles["source_dataset"]))

    rows = []
    for _, r in scope_meta.iterrows():
        audio_id, dataset, patient_uid = r["audio_id"], r["dataset"], r["patient_uid"]
        n_samples = int(r["samples_out_actual"])
        n = p4.segments_per_recording(n_samples, p4.WINDOW_SAMPLES, p4.HOP_SAMPLES)
        tail = n_samples - ((n - 1) * p4.HOP_SAMPLES + p4.WINDOW_SAMPLES) if n > 0 else n_samples
        label = int(label_by_patient[patient_uid])

        for k in range(n):
            start_sample = k * p4.HOP_SAMPLES
            end_sample = start_sample + p4.WINDOW_SAMPLES
            row = {
                "segment_id": f"{audio_id}_{k:03d}", "audio_id": audio_id, "dataset": dataset,
                "patient_uid": patient_uid, "diagnosis": r["diagnosis"], "device": r["device"],
                "zone": r["zone"], "filter": r["filter"],
                "start_sample": start_sample, "end_sample": end_sample,
                "start_s": round(start_sample / cfg.TARGET_SR, 4), "end_s": round(end_sample / cfg.TARGET_SR, 4),
                "tail_samples": int(tail), "segment_idx": k, "n_segments_in_recording": n,
                "quality_status": r["quality_status"], "quality_reasons": r["quality_reasons"],
                "calibration_patient": bool(calib_by_patient[patient_uid]),
                "dn_reliable": dn_reliable_map[audio_id], "dn_flag_reason": dn_reason_map.get(audio_id, ""),
                "fold_id": fold_id, "role": role_by_patient[patient_uid],
                "target_label": label, "target_name": "COPD" if label == 1 else "Control",
                "source_dataset": source_by_patient[patient_uid],
            }
            row.update(p4.cycle_classification(audio_id, cycles_by_audio, start_sample, end_sample, dataset))
            rows.append(row)

    inventory = pd.DataFrame(rows)
    if inventory.empty:
        raise ValueError(f"{dataset_scope}/fold_{fold_id:02d}: la seleccion quedo vacia (0 segmentos)")
    if inventory["segment_id"].duplicated().any():
        raise ValueError(f"{dataset_scope}/fold_{fold_id:02d}: segment_id duplicados")

    inventory.insert(0, "task_array_index", np.arange(len(inventory), dtype=np.int64))
    # No hay un array global previo del que copiar filas (a diferencia de
    # prepare_task_data.py): cada fold regenera su propio audio desde cero,
    # con su propio denoising. source_array_index queda identico a
    # task_array_index, documentando que esta es la unica fuente de este fold.
    inventory.insert(1, "source_array_index", inventory["task_array_index"])
    return inventory


def fill_fold_arrays(inventory: pd.DataFrame, no_dn_signals: dict, dn_signals: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(inventory)
    window = p4.WINDOW_SAMPLES
    no_dn_array = np.empty((n, window), dtype=np.float32)
    dn_array = np.empty((n, window), dtype=np.float32)
    for i, seg in enumerate(inventory.itertuples()):
        no_dn_array[i] = no_dn_signals[seg.audio_id][seg.start_sample:seg.end_sample]
        dn_array[i] = dn_signals[seg.audio_id][seg.start_sample:seg.end_sample]
    return no_dn_array, dn_array


def validate_fold_output(inventory: pd.DataFrame, no_dn_array: np.ndarray, dn_array: np.ndarray) -> None:
    if no_dn_array.shape != dn_array.shape:
        raise ValueError(f"las dos ramas tienen formas distintas: {no_dn_array.shape} vs {dn_array.shape}")
    if no_dn_array.shape[0] != len(inventory):
        raise ValueError(f"{no_dn_array.shape[0]} filas en los arreglos, {len(inventory)} en el inventario")
    if not np.isfinite(no_dn_array).all() or not np.isfinite(dn_array).all():
        raise ValueError("NaN/Inf en los arreglos generados")
    missing = set(REQUIRED_SEGMENT_COLUMNS) - set(inventory.columns)
    if missing:
        raise ValueError(f"faltan columnas en segments.csv: {sorted(missing)}")
    n_unreliable_before = int(inventory.loc[~inventory["dn_reliable"], "audio_id"].nunique())
    # dn_reliable=False nunca filtra: si esto cambiara el conteo de filas
    # respecto de incluir esas grabaciones, algo las estaria excluyendo.
    if n_unreliable_before and inventory.loc[inventory["dn_reliable"] == False].empty:  # noqa: E712
        raise RuntimeError("dn_reliable=False parece haber excluido segmentos; no deberia filtrar nada")


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cfg.ROOT, check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=cfg.ROOT, check=True, capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def run_fold(
    dataset_scope: str,
    fold_id: int,
    patient_folds_csv: Path,
    patient_folds_manifest: Path,
    output_root: Path,
) -> dict:
    """Orquesta un ``(dataset_scope, fold_id)`` completo: recalibra, procesa
    audio, segmenta y publica atomicamente. Devuelve el manifiesto escrito.
    """
    admitted = p3.admitted_recordings()
    roles = load_patient_roles(patient_folds_csv, patient_folds_manifest, dataset_scope, fold_id)
    authorized_audio_ids = load_authorized_audio_ids(dataset_scope)
    scope_meta = select_scope_recordings(admitted, authorized_audio_ids, roles)
    verify_phase2_audio_hashes(scope_meta)

    params = recalibrate_denoising(dataset_scope, scope_meta, roles)
    verify_no_leakage_into_calibration(roles, params)

    processed = process_fold_recordings(scope_meta, params)
    cycles_by_audio = p4._load_cycle_bounds()
    inventory = build_fold_segment_inventory(
        dataset_scope, fold_id, processed["scope_meta"], roles,
        processed["dn_reliable_map"], processed["dn_reason_map"], cycles_by_audio,
    )
    no_dn_array, dn_array = fill_fold_arrays(inventory, processed["no_dn_signals"], processed["dn_signals"])
    validate_fold_output(inventory, no_dn_array, dn_array)

    target_dir = Path(output_root) / dataset_scope / f"fold_{fold_id:02d}"
    staging = u.prepare_staging(target_dir)
    try:
        np.save(staging / "segments_no_dn.npy", no_dn_array)
        np.save(staging / "segments_dn.npy", dn_array)
        inventory.to_csv(staging / "segments.csv", index=False, lineterminator="\n")

        preprocessing_params = {
            "dataset_scope": dataset_scope, "fold_id": fold_id,
            **params,
            "peak_ceiling": float(cfg.PEAK_CEILING),
            "bandpass_low_hz": int(cfg.BANDPASS_LOW), "bandpass_high_hz": int(cfg.BANDPASS_HIGH),
            "bandpass_order": int(cfg.BANDPASS_ORDER),
        }
        (staging / "preprocessing_params.json").write_text(
            json.dumps(preprocessing_params, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        output_hashes = {
            "segments.csv": u.file_sha256(staging / "segments.csv"),
            "segments_no_dn.npy": u.file_sha256(staging / "segments_no_dn.npy"),
            "segments_dn.npy": u.file_sha256(staging / "segments_dn.npy"),
        }
        manifest = {
            "verdict": "PASS",
            "dataset_scope": dataset_scope, "fold_id": fold_id,
            "patient_folds_csv_sha256": u.file_sha256(patient_folds_csv),
            "n_segments": int(len(inventory)),
            "n_patients": int(inventory["patient_uid"].nunique()),
            "shape": list(no_dn_array.shape), "dtype": str(no_dn_array.dtype),
            "counts_by_role": {
                role: int(inventory.loc[inventory["role"] == role, "patient_uid"].nunique())
                for role in ("train", "validation", "test")
            },
            "output_hashes": output_hashes,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "git": _git_state(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except Exception:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise

    u.swap_staging_into_place(target_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    u.section(f"FOLD-DENOISING: {args.dataset_scope} / fold_{args.fold_id:02d}")
    try:
        manifest = run_fold(
            args.dataset_scope, args.fold_id, args.patient_folds_csv,
            args.patient_folds_manifest, args.output_root,
        )
    except Exception as exc:  # noqa: BLE001 - se reporta con claridad, nunca se oculta
        print(f"FALLO: {exc}", file=sys.stderr)
        return 1

    print(f"  Segmentos       : {manifest['n_segments']}")
    print(f"  Pacientes       : {manifest['n_patients']} ({manifest['counts_by_role']})")
    print(f"  Forma           : {manifest['shape']} ({manifest['dtype']})")
    print(f"  Salida          : {Path(args.output_root) / args.dataset_scope / f'fold_{args.fold_id:02d}'}")
    print("  Veredicto       : PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
