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

from dataclasses import dataclass

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

    table = grouped.agg(
        target_label=("target_label", "first"),
        calibration_patient=("calibration_patient", "first"),
        n_recordings=("audio_id", "nunique"),
        n_segments=("segment_id", "size"),
    ).reset_index()
    return table


def assign_fold_groups(
    patient_table: pd.DataFrame, n_splits: int = 5, random_state: int = 20260914
) -> pd.DataFrame:
    """Anade ``fold_group``: -1 para calibracion, 0..n_splits-1 en el resto.

    ``StratifiedKFold`` con semilla fija corre solo sobre los pacientes que no
    son de calibracion, estratificando por ``target_label`` para que cada
    grupo conserve la proporcion de clases del conjunto evaluable.
    """
    table = patient_table.copy()
    table["fold_group"] = CALIBRATION_GROUP

    evaluable = table.loc[~table["calibration_patient"]].sort_values("patient_uid")
    if evaluable.empty:
        raise ValueError("no hay pacientes evaluables (todos son de calibracion)")

    counts = evaluable["target_label"].value_counts()
    if len(counts) < 2:
        raise ValueError(f"la poblacion evaluable tiene una sola clase: {counts.to_dict()}")
    if counts.min() < n_splits:
        raise ValueError(
            f"la clase minoritaria tiene {counts.min()} pacientes, "
            f"insuficiente para {n_splits} folds estratificados"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = np.empty(len(evaluable), dtype=np.int64)
    for fold_id, (_, test_idx) in enumerate(
        skf.split(evaluable["patient_uid"], evaluable["target_label"])
    ):
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

    def get_split(self, fold_id: int) -> tuple[list[str], list[str], list[str]]:
        """``(train_patients, validation_patients, test_patients)`` del fold."""
        sub = self.fold_table.loc[self.fold_table["fold"] == fold_id]
        train = sub.loc[sub["role"] == ROLE_TRAIN, "patient_uid"].tolist()
        val = sub.loc[sub["role"] == ROLE_VALIDATION, "patient_uid"].tolist()
        test = sub.loc[sub["role"] == ROLE_TEST, "patient_uid"].tolist()
        return train, val, test


def build_patient_folds(
    segments: pd.DataFrame, n_splits: int = 5, random_state: int = 20260914
) -> PatientFolds:
    """Construye y verifica la particion completa a partir de ``segments``."""
    patient_table = build_patient_table(segments)
    patient_table = assign_fold_groups(patient_table, n_splits=n_splits, random_state=random_state)
    fold_table = build_fold_table(patient_table, n_splits=n_splits)
    folds = PatientFolds(patient_table=patient_table, fold_table=fold_table, n_splits=n_splits)
    verify_folds(folds)
    return folds


def verify_folds(folds: PatientFolds) -> None:
    """Todas las comprobaciones de fuga exigidas por el plan. Lanza si falla."""
    patient_table = folds.patient_table
    label_by_patient = dict(zip(patient_table["patient_uid"], patient_table["target_label"]))
    calibration_ids = set(
        patient_table.loc[patient_table["calibration_patient"], "patient_uid"]
    )
    evaluable_ids = set(patient_table["patient_uid"]) - calibration_ids

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
