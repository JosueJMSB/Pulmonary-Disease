"""Particion por paciente: calibracion siempre en train, K-fold anidado.

Todo el modulo trabaja a nivel de paciente. Los segmentos y grabaciones nunca
se reparten por su cuenta: heredan la particion de su ``patient_uid``, de modo
que no puede haber fuga de grabaciones ni de segmentos entre conjuntos si no
la hay de pacientes.

Esquema por fold k (0-indexado):

    test       = grupo k
    validation = grupo (k + 1) % n_splits
    train      = pacientes de calibracion + los demas grupos

Los pacientes de calibracion (``calibration_patient=True`` en el manifiesto de
la fase 3) quedan fuera del ``StratifiedKFold``: por contrato, deben permanecer
siempre en entrenamiento, en todos los folds y en todas las condiciones.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CALIBRATION_GROUP = -1
ROLE_TRAIN = "train"
ROLE_VALIDATION = "validation"
ROLE_TEST = "test"


def build_patient_table(segments: pd.DataFrame) -> pd.DataFrame:
    """Una fila por ``patient_uid`` con su etiqueta y si es de calibracion.

    Falla de forma explicita si un paciente aparece con mas de una etiqueta o
    con valores mixtos de ``calibration_patient``: ambos son invariantes del
    manifiesto de origen y una violacion indica un problema de datos, no algo
    que deba promediarse o ignorarse en silencio.

    Si ``segments`` trae una columna ``dataset`` (siempre presente en la
    salida real de ``prepare_task_data``/``prepare_combined_task_data``, pero
    opcional aqui para no romper pruebas que arman segmentos sinteticos sin
    ella), se añade ``source_dataset`` con la misma validacion de unicidad por
    paciente que ``target_label``/``calibration_patient``.
    """
    required = {"patient_uid", "target_label", "calibration_patient"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"faltan columnas en segments: {sorted(missing)}")

    grouped = segments.groupby("patient_uid")
    n_labels = grouped["target_label"].nunique()
    bad_label = n_labels[n_labels != 1]
    if len(bad_label):
        raise ValueError(f"pacientes con mas de una etiqueta: {bad_label.index.tolist()}")

    n_calib = grouped["calibration_patient"].nunique()
    bad_calib = n_calib[n_calib != 1]
    if len(bad_calib):
        raise ValueError(
            f"pacientes con calibration_patient inconsistente: {bad_calib.index.tolist()}"
        )

    agg_kwargs = dict(
        target_label=("target_label", "first"),
        calibration_patient=("calibration_patient", "first"),
        n_recordings=("audio_id", "nunique"),
        n_segments=("segment_id", "size"),
    )
    if "dataset" in segments.columns:
        n_source = grouped["dataset"].nunique()
        bad_source = n_source[n_source != 1]
        if len(bad_source):
            raise ValueError(f"pacientes con mas de una fuente (dataset): {bad_source.index.tolist()}")
        agg_kwargs["source_dataset"] = ("dataset", "first")

    table = grouped.agg(**agg_kwargs).reset_index()
    return table


def assign_fold_groups(
    patient_table: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 20260914,
    stratify_by_dataset: bool = False,
    respect_calibration_patient: bool = True,
) -> pd.DataFrame:
    """Anade ``fold_group``: -1 para calibracion, 0..n_splits-1 en el resto.

    ``StratifiedKFold`` con semilla fija corre solo sobre los pacientes que no
    son de calibracion. Por defecto estratifica solo por ``target_label``; con
    ``stratify_by_dataset=True`` estratifica por la combinacion
    ``source_dataset`` + ``target_label`` (requiere que ``patient_table``
    tenga ``source_dataset``, ver ``build_patient_table``), de modo que cada
    grupo conserve tambien la proporcion de fuentes del conjunto evaluable.

    ``respect_calibration_patient=False`` (protocolo fold-aware: el denoising
    ya no se calibra una sola vez de forma global, sino por fold, as que no
    hace falta mantener a esos pacientes siempre en train) hace que TODOS los
    pacientes entren al ``StratifiedKFold`` -incluidos los historicamente
    marcados como ``calibration_patient``-, sin reservar ningun grupo -1. La
    columna ``calibration_patient`` se conserva intacta en la tabla, solo como
    trazabilidad; deja de leerse para decidir quien participa.
    """
    table = patient_table.copy()
    table["fold_group"] = CALIBRATION_GROUP

    if respect_calibration_patient:
        evaluable = table.loc[~table["calibration_patient"]].sort_values("patient_uid")
    else:
        evaluable = table.sort_values("patient_uid")
    if evaluable.empty:
        raise ValueError("no hay pacientes evaluables (todos son de calibracion)")

    label_counts = evaluable["target_label"].value_counts()
    if len(label_counts) < 2:
        raise ValueError(f"la poblacion evaluable tiene una sola clase: {label_counts.to_dict()}")

    if stratify_by_dataset:
        if "source_dataset" not in evaluable.columns:
            raise ValueError(
                "stratify_by_dataset=True requiere source_dataset en patient_table "
                "(segments debe traer una columna 'dataset')"
            )
        strata = evaluable["source_dataset"].astype(str) + "__" + evaluable["target_label"].astype(str)
        minority_phrase = "el estrato (fuente+clase) minoritario"
    else:
        strata = evaluable["target_label"]
        minority_phrase = "la clase minoritaria"

    counts = strata.value_counts()
    if counts.min() < n_splits:
        raise ValueError(
            f"{minority_phrase} tiene {counts.min()} pacientes, "
            f"insuficiente para {n_splits} folds estratificados"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = np.empty(len(evaluable), dtype=np.int64)
    for fold_id, (_, test_idx) in enumerate(skf.split(evaluable["patient_uid"], strata)):
        groups[test_idx] = fold_id

    table.loc[evaluable.index, "fold_group"] = groups
    return table


def build_fold_table(patient_table: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Tabla larga ``(fold, patient_uid, role)`` para los ``n_splits`` folds."""
    if "fold_group" not in patient_table.columns:
        raise ValueError("patient_table no tiene fold_group; llame a assign_fold_groups primero")

    rows = []
    calibration_ids = patient_table.loc[patient_table["fold_group"] == CALIBRATION_GROUP, "patient_uid"]
    for fold_id in range(n_splits):
        val_group = (fold_id + 1) % n_splits
        test_mask = patient_table["fold_group"] == fold_id
        val_mask = patient_table["fold_group"] == val_group
        train_mask = ~test_mask & ~val_mask & (patient_table["fold_group"] != CALIBRATION_GROUP)

        for patient_uid in patient_table.loc[test_mask, "patient_uid"]:
            rows.append({"fold": fold_id, "patient_uid": patient_uid, "role": ROLE_TEST})
        for patient_uid in patient_table.loc[val_mask, "patient_uid"]:
            rows.append({"fold": fold_id, "patient_uid": patient_uid, "role": ROLE_VALIDATION})
        for patient_uid in patient_table.loc[train_mask, "patient_uid"]:
            rows.append({"fold": fold_id, "patient_uid": patient_uid, "role": ROLE_TRAIN})
        for patient_uid in calibration_ids:
            rows.append({"fold": fold_id, "patient_uid": patient_uid, "role": ROLE_TRAIN})

    fold_table = pd.DataFrame(rows, columns=["fold", "patient_uid", "role"])
    duplicated = fold_table.duplicated(["fold", "patient_uid"])
    if duplicated.any():
        raise RuntimeError("un paciente aparece dos veces en el mismo fold")
    return fold_table


@dataclass(frozen=True)
class PatientFolds:
    """Particion completa: tabla de pacientes + asignacion larga por fold."""

    patient_table: pd.DataFrame
    fold_table: pd.DataFrame
    n_splits: int
    stratify_by_dataset: bool = False
    respect_calibration_patient: bool = True

    def get_split(self, fold_id: int) -> tuple[list[str], list[str], list[str]]:
        """``(train_patients, validation_patients, test_patients)`` del fold."""
        sub = self.fold_table.loc[self.fold_table["fold"] == fold_id]
        train = sub.loc[sub["role"] == ROLE_TRAIN, "patient_uid"].tolist()
        val = sub.loc[sub["role"] == ROLE_VALIDATION, "patient_uid"].tolist()
        test = sub.loc[sub["role"] == ROLE_TEST, "patient_uid"].tolist()
        return train, val, test


def build_patient_folds(
    segments: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 20260914,
    stratify_by_dataset: bool = False,
    respect_calibration_patient: bool = True,
) -> PatientFolds:
    """Construye y verifica la particion completa a partir de ``segments``."""
    patient_table = build_patient_table(segments)
    patient_table = assign_fold_groups(
        patient_table, n_splits=n_splits, random_state=random_state,
        stratify_by_dataset=stratify_by_dataset,
        respect_calibration_patient=respect_calibration_patient,
    )
    fold_table = build_fold_table(patient_table, n_splits=n_splits)
    folds = PatientFolds(
        patient_table=patient_table, fold_table=fold_table, n_splits=n_splits,
        stratify_by_dataset=stratify_by_dataset,
        respect_calibration_patient=respect_calibration_patient,
    )
    verify_folds(folds)
    return folds


def verify_folds(folds: PatientFolds) -> None:
    """Todas las comprobaciones de fuga exigidas por el plan. Lanza si falla.

    Con ``stratify_by_dataset=True`` (y ``source_dataset`` disponible en
    ``patient_table``), ademas de las dos clases exige que validation y test
    de cada fold contengan los mismos estratos fuente+clase presentes en la
    poblacion evaluable (los "cuatro estratos" del plan combinado). Sin eso,
    el comportamiento es identico al anterior: solo exige ambas clases.

    Con ``respect_calibration_patient=False`` (protocolo fold-aware),
    ``calibration_ids`` es explicitamente el conjunto vacio y ``evaluable_ids``
    son TODOS los pacientes de ``patient_table``, sin leer la columna
    ``calibration_patient`` para nada (queda solo como trazabilidad historica).
    Ademas se exige que ningun paciente haya quedado en el grupo especial -1.
    """
    patient_table = folds.patient_table
    label_by_patient = dict(zip(patient_table["patient_uid"], patient_table["target_label"]))

    if folds.respect_calibration_patient:
        calibration_ids = set(
            patient_table.loc[patient_table["calibration_patient"], "patient_uid"]
        )
        evaluable_ids = set(patient_table["patient_uid"]) - calibration_ids
    else:
        calibration_ids = set()
        evaluable_ids = set(patient_table["patient_uid"])
        stray = int((patient_table["fold_group"] == CALIBRATION_GROUP).sum())
        if stray:
            raise RuntimeError(
                f"respect_calibration_patient=False pero {stray} paciente(s) quedaron "
                "en el grupo de calibracion (fold_group=-1)"
            )

    check_strata = folds.stratify_by_dataset and "source_dataset" in patient_table.columns
    if check_strata:
        source_by_patient = dict(zip(patient_table["patient_uid"], patient_table["source_dataset"]))
        expected_strata = {(source_by_patient[pid], label_by_patient[pid]) for pid in evaluable_ids}

    test_coverage: dict[str, int] = {pid: 0 for pid in evaluable_ids}

    for fold_id in range(folds.n_splits):
        train, val, test = folds.get_split(fold_id)
        train_set, val_set, test_set = set(train), set(val), set(test)

        if train_set & val_set:
            raise RuntimeError(f"fold {fold_id}: train y validation se solapan")
        if train_set & test_set:
            raise RuntimeError(f"fold {fold_id}: train y test se solapan")
        if val_set & test_set:
            raise RuntimeError(f"fold {fold_id}: validation y test se solapan")

        if not calibration_ids <= train_set:
            raise RuntimeError(f"fold {fold_id}: no todos los pacientes de calibracion estan en train")
        if calibration_ids & (val_set | test_set):
            raise RuntimeError(f"fold {fold_id}: un paciente de calibracion cayo en val/test")

        if check_strata:
            for role_name, ids in (("validation", val_set), ("test", test_set)):
                strata_here = {(source_by_patient[pid], label_by_patient[pid]) for pid in ids}
                if strata_here != expected_strata:
                    raise RuntimeError(
                        f"fold {fold_id}: {role_name} no contiene los estratos fuente+clase "
                        f"esperados {sorted(expected_strata)}; se encontraron {sorted(strata_here)}"
                    )
        else:
            for role_name, ids in (("validation", val_set), ("test", test_set)):
                labels_here = {label_by_patient[pid] for pid in ids}
                if labels_here != {0, 1}:
                    raise RuntimeError(
                        f"fold {fold_id}: {role_name} no contiene ambas clases ({labels_here})"
                    )

        for pid in test_set:
            test_coverage[pid] = test_coverage.get(pid, 0) + 1

    missing = [pid for pid, n in test_coverage.items() if n == 0]
    if missing:
        raise RuntimeError(f"pacientes que nunca caen en test: {missing}")
    repeated = [pid for pid, n in test_coverage.items() if n > 1]
    if repeated:
        raise RuntimeError(f"pacientes que caen en test mas de una vez: {repeated}")


def folds_are_identical(a: PatientFolds, b: PatientFolds) -> bool:
    """True si dos particiones asignan exactamente los mismos roles.

    Se usa para verificar que ``no_dn`` y ``dn`` de una misma condicion
    comparten fold a fold: al construirse desde la misma poblacion de
    pacientes y la misma semilla, deben coincidir por construccion, pero la
    comprobacion explicita deja constancia en vez de asumirlo.
    """
    left = a.fold_table.sort_values(["fold", "patient_uid"]).reset_index(drop=True)
    right = b.fold_table.sort_values(["fold", "patient_uid"]).reset_index(drop=True)
    return left.equals(right)


def filter_segments_by_patients(segments: pd.DataFrame, patient_ids: list[str]) -> pd.DataFrame:
    """Subconjunto de segmentos cuyos pacientes estan en ``patient_ids``."""
    return segments.loc[segments["patient_uid"].isin(set(patient_ids))].reset_index(drop=True)


# ---------------------------------------------------------------------------
# patient_folds.csv: asignacion maestra persistida (protocolo fold-aware).
#
# Se calcula una sola vez (build_master_folds.py) y se reutiliza tal cual en
# preprocessing/fold_denoising.py y en las corridas de SVM/CNN/CRNN: ningun
# consumidor vuelve a correr StratifiedKFold, todos leen el mismo csv.
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_bytes)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


PATIENT_FOLDS_COLUMNS = (
    "dataset_scope", "patient_uid", "target_label", "source_dataset",
    "calibration_patient", "fold_group",
)


def patient_folds_to_frame(folds: PatientFolds, dataset_scope: str) -> pd.DataFrame:
    """Una fila por paciente, en el formato de ``patient_folds.csv``."""
    table = folds.patient_table
    source_dataset = table["source_dataset"] if "source_dataset" in table.columns else pd.NA
    return pd.DataFrame({
        "dataset_scope": dataset_scope,
        "patient_uid": table["patient_uid"],
        "target_label": table["target_label"],
        "source_dataset": source_dataset,
        "calibration_patient": table["calibration_patient"],
        "fold_group": table["fold_group"],
    })


def combine_patient_folds(parts: dict[str, PatientFolds], n_splits: int = 5) -> PatientFolds:
    """Une varias particiones de una sola fuente (p. ej. ICBHI y Fraiwan, cada
    una ya construida con ``respect_calibration_patient=False``) copiando el
    ``fold_group`` que cada paciente recibio en su particion de origen -no
    vuelve a correr ``StratifiedKFold``-. Pensado para construir COMBINED
    reutilizando los folds de ICBHI y de Fraiwan tal cual, no con un sorteo
    conjunto nuevo.
    """
    tables = []
    for source_name, folds in parts.items():
        if "source_dataset" not in folds.patient_table.columns:
            raise ValueError(f"{source_name}: patient_table no tiene source_dataset")
        tables.append(folds.patient_table)

    combined_table = pd.concat(tables, ignore_index=True)
    duplicated = combined_table["patient_uid"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"patient_uid compartidos entre fuentes: "
            f"{sorted(combined_table.loc[duplicated, 'patient_uid'])}"
        )

    fold_table = build_fold_table(combined_table, n_splits=n_splits)
    combined_folds = PatientFolds(
        patient_table=combined_table, fold_table=fold_table, n_splits=n_splits,
        stratify_by_dataset=True, respect_calibration_patient=False,
    )
    verify_folds(combined_folds)
    return combined_folds


def write_patient_folds_csv(
    frame: pd.DataFrame, csv_path: Path, manifest_path: Path, manifest_extra: dict,
) -> dict:
    """Escribe ``patient_folds.csv`` (ya concatenado entre dataset_scope) y su
    manifiesto (hash del csv + lo que el llamador quiera registrar: hashes de
    los segments.csv fuente, semillas, conteos, git). Devuelve el manifiesto."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, lineterminator="\n")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_folds_csv_sha256": sha256_file(csv_path),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **manifest_extra,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def load_patient_folds(csv_path: Path, manifest_path: Path, dataset_scope: str) -> PatientFolds:
    """Lee ``patient_folds.csv`` para un ``dataset_scope`` y verifica que
    coincide exactamente con lo que registro su manifiesto (hash del csv y
    numero de pacientes) antes de reconstruir la particion. Ningun consumidor
    -``preprocessing/fold_denoising.py`` ni las corridas de modelo- puede usar
    una version desincronizada sin que se detecte.
    """
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_hash = sha256_file(csv_path)
    expected_hash = manifest.get("patient_folds_csv_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            f"{csv_path}: sha256 {actual_hash[:12]}... no coincide con el manifiesto "
            f"{manifest_path} ({str(expected_hash)[:12]}...); regenere con build_master_folds"
        )

    full = pd.read_csv(csv_path, dtype={"patient_uid": str})
    missing = set(PATIENT_FOLDS_COLUMNS) - set(full.columns)
    if missing:
        raise ValueError(f"{csv_path}: faltan columnas {sorted(missing)}")
    full["calibration_patient"] = full["calibration_patient"].astype(bool)

    table = full.loc[full["dataset_scope"] == dataset_scope].drop(columns="dataset_scope").reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{csv_path}: no hay filas para dataset_scope={dataset_scope!r}")

    expected_counts = manifest.get("counts", {}).get(dataset_scope, {})
    expected_n_patients = expected_counts.get("n_patients")
    if expected_n_patients is not None and int(expected_n_patients) != len(table):
        raise ValueError(
            f"{dataset_scope}: {len(table)} pacientes en el csv, "
            f"{expected_n_patients} en el manifiesto"
        )

    evaluable = table.loc[table["fold_group"] != CALIBRATION_GROUP]
    if evaluable.empty:
        raise ValueError(f"{dataset_scope}: ningun paciente con fold_group valido")
    n_splits = int(evaluable["fold_group"].max()) + 1
    stratify_by_dataset = "source_dataset" in table.columns and table["source_dataset"].nunique() > 1

    fold_table = build_fold_table(table, n_splits=n_splits)
    folds = PatientFolds(
        patient_table=table, fold_table=fold_table, n_splits=n_splits,
        stratify_by_dataset=stratify_by_dataset, respect_calibration_patient=False,
    )
    verify_folds(folds)
    return folds
