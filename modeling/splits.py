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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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


# ---------------------------------------------------------------------------
# Protocolo holdout-v3 (PLAN-EXPERIMENTO FINAL.md): una unica separacion
# 80 % desarrollo / 20 % prueba externa bloqueada, por paciente, y 5 folds
# internos SOLO sobre el 80 %. Cada paciente de desarrollo es validado
# exactamente una vez; los de prueba llevan ``inner_fold_group = -1`` y ningun
# codigo de preprocesamiento, cache o entrenamiento de esta etapa puede
# leerlos (``HoldoutSplit.blocked_test_patients`` existe solo para VERIFICAR
# que no aparecen en ninguna parte).
#
# A diferencia de ``patient_folds.csv`` (protocolo fold-aware v2), aqui no hay
# rol "test" dentro de los folds: cada fold interno k tiene solo ``train``
# (los otros cuatro grupos) y ``validation`` (el grupo k).
# ---------------------------------------------------------------------------

HOLDOUT_PROTOCOL_NAME = "holdout-v3"
OUTER_ROLE_DEVELOPMENT = "development"
OUTER_ROLE_TEST = "test"
OUTER_TEST_FOLD_GROUP = -1
HOLDOUT_TEST_FRACTION = 0.2

HOLDOUT_SPLIT_COLUMNS = (
    "dataset_scope", "patient_uid", "source_dataset", "target_label",
    "outer_role", "inner_fold_group", "calibration_patient",
)


def assign_holdout_split(
    patient_table: pd.DataFrame,
    test_fraction: float = HOLDOUT_TEST_FRACTION,
    n_splits: int = 5,
    random_state: int = 20260914,
) -> pd.DataFrame:
    """Anade ``outer_role`` (development/test) e ``inner_fold_group``.

    1. ``StratifiedShuffleSplit`` (una sola particion, semilla fija) separa
       ~``test_fraction`` de los pacientes como prueba externa, estratificando
       por ``target_label``.
    2. ``StratifiedKFold`` (misma semilla) reparte SOLO los pacientes de
       desarrollo en ``n_splits`` grupos, tambien estratificado por clase.

    Los pacientes de prueba reciben ``inner_fold_group = -1``. El orden de
    entrada no influye: se ordena por ``patient_uid`` antes de sortear.
    ``calibration_patient`` no participa en nada (queda solo como
    trazabilidad historica, igual que en fold-aware v2).
    """
    required = {"patient_uid", "target_label"}
    missing = required - set(patient_table.columns)
    if missing:
        raise ValueError(f"faltan columnas en patient_table: {sorted(missing)}")

    table = patient_table.sort_values("patient_uid").reset_index(drop=True)
    labels = table["target_label"].to_numpy()
    counts = table["target_label"].value_counts()
    if len(counts) < 2:
        raise ValueError(f"la poblacion tiene una sola clase: {counts.to_dict()}")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=random_state)
    dev_idx, _test_idx = next(splitter.split(table["patient_uid"], labels))
    dev_idx = np.sort(dev_idx)

    table["outer_role"] = OUTER_ROLE_TEST
    table.loc[dev_idx, "outer_role"] = OUTER_ROLE_DEVELOPMENT
    table["inner_fold_group"] = OUTER_TEST_FOLD_GROUP

    development = table.iloc[dev_idx]
    dev_counts = development["target_label"].value_counts()
    if len(dev_counts) < 2 or dev_counts.min() < n_splits:
        raise ValueError(
            f"la clase minoritaria de desarrollo tiene {int(dev_counts.min())} pacientes, "
            f"insuficiente para {n_splits} folds internos estratificados"
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    groups = np.empty(len(development), dtype=np.int64)
    for fold_id, (_, val_idx) in enumerate(skf.split(development["patient_uid"], development["target_label"])):
        groups[val_idx] = fold_id
    table.loc[development.index, "inner_fold_group"] = groups
    return table


def verify_holdout_split(table: pd.DataFrame, n_splits: int = 5, context: str = "") -> None:
    """Comprobaciones de fuga y de cobertura de la separacion holdout-v3.
    Lanza si alguna falla; nunca corrige nada en silencio."""
    where = f"{context}: " if context else ""
    required = {"patient_uid", "target_label", "outer_role", "inner_fold_group"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{where}faltan columnas {sorted(missing)}")

    if table["patient_uid"].duplicated().any():
        raise RuntimeError(
            f"{where}patient_uid repetidos: {sorted(table.loc[table['patient_uid'].duplicated(), 'patient_uid'])}"
        )
    unknown_roles = set(table["outer_role"]) - {OUTER_ROLE_DEVELOPMENT, OUTER_ROLE_TEST}
    if unknown_roles:
        raise ValueError(f"{where}outer_role desconocido: {sorted(unknown_roles)}")

    development = table.loc[table["outer_role"] == OUTER_ROLE_DEVELOPMENT]
    test = table.loc[table["outer_role"] == OUTER_ROLE_TEST]
    if development.empty or test.empty:
        raise RuntimeError(f"{where}desarrollo y prueba deben tener pacientes ({len(development)}/{len(test)})")

    if not (test["inner_fold_group"] == OUTER_TEST_FOLD_GROUP).all():
        raise RuntimeError(f"{where}hay pacientes de prueba con inner_fold_group distinto de {OUTER_TEST_FOLD_GROUP}")
    groups = development["inner_fold_group"]
    if not groups.between(0, n_splits - 1).all():
        raise RuntimeError(f"{where}hay pacientes de desarrollo fuera de los grupos 0..{n_splits - 1}")
    if set(groups.unique()) != set(range(n_splits)):
        raise RuntimeError(f"{where}no todos los folds internos 0..{n_splits - 1} tienen pacientes")

    for name, part in (("desarrollo", development), ("prueba", test)):
        if set(part["target_label"]) != {0, 1}:
            raise RuntimeError(f"{where}{name} no contiene ambas clases")
    for fold_id in range(n_splits):
        val = development.loc[groups == fold_id]
        train = development.loc[groups != fold_id]
        for name, part in (("validation", val), ("train", train)):
            if set(part["target_label"]) != {0, 1}:
                raise RuntimeError(f"{where}fold interno {fold_id}: {name} no contiene ambas clases")

    if "source_dataset" in table.columns and table["source_dataset"].nunique() > 1:
        strata = set(zip(table["source_dataset"], table["target_label"]))
        checks = [("desarrollo", development), ("prueba", test)]
        checks += [(f"validation del fold interno {k}", development.loc[groups == k]) for k in range(n_splits)]
        for name, part in checks:
            here = set(zip(part["source_dataset"], part["target_label"]))
            if here != strata:
                raise RuntimeError(
                    f"{where}{name} no contiene todos los estratos fuente+clase "
                    f"{sorted(strata)}; se encontraron {sorted(here)}"
                )


def combine_holdout_splits(parts: dict[str, pd.DataFrame], n_splits: int = 5) -> pd.DataFrame:
    """COMBINED: copia EXACTAMENTE la asignacion (outer_role, inner_fold_group)
    que cada paciente recibio en su dataset de origen. No vuelve a sortear."""
    tables = []
    for source_name, table in parts.items():
        if "source_dataset" not in table.columns or table["source_dataset"].isna().any():
            raise ValueError(f"{source_name}: la tabla necesita source_dataset en todos los pacientes")
        tables.append(table)
    combined = pd.concat(tables, ignore_index=True)
    duplicated = combined["patient_uid"].duplicated()
    if duplicated.any():
        raise ValueError(
            f"patient_uid compartidos entre fuentes: {sorted(combined.loc[duplicated, 'patient_uid'])}"
        )
    verify_holdout_split(combined, n_splits, context="COMBINED")
    return combined


def holdout_split_to_frame(table: pd.DataFrame, dataset_scope: str) -> pd.DataFrame:
    """Una fila por paciente, en el formato de ``holdout_splits.csv``."""
    source_dataset = table["source_dataset"] if "source_dataset" in table.columns else pd.NA
    calibration = table["calibration_patient"] if "calibration_patient" in table.columns else False
    frame = pd.DataFrame({
        "dataset_scope": dataset_scope,
        "patient_uid": table["patient_uid"],
        "source_dataset": source_dataset,
        "target_label": table["target_label"],
        "outer_role": table["outer_role"],
        "inner_fold_group": table["inner_fold_group"],
        "calibration_patient": calibration,
    })
    return frame.loc[:, list(HOLDOUT_SPLIT_COLUMNS)].reset_index(drop=True)


def write_holdout_split_csv(
    frame: pd.DataFrame, csv_path: Path, manifest_path: Path, manifest_extra: dict,
) -> dict:
    """Escribe ``holdout_splits.csv`` y su manifiesto (hash del csv, semilla,
    conteos y lo que el llamador quiera registrar). Devuelve el manifiesto."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, lineterminator="\n")

    manifest = {
        "protocol": HOLDOUT_PROTOCOL_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_splits_csv_sha256": sha256_file(csv_path),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        **manifest_extra,
    }
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


@dataclass(frozen=True)
class HoldoutSplit:
    """Separacion holdout-v3 de UN dataset_scope. ``patient_table`` incluye
    los pacientes de prueba solo para poder verificar su ausencia."""

    dataset_scope: str
    patient_table: pd.DataFrame
    n_splits: int

    def development_table(self) -> pd.DataFrame:
        table = self.patient_table
        return table.loc[table["outer_role"] == OUTER_ROLE_DEVELOPMENT].reset_index(drop=True)

    def development_patients(self) -> list[str]:
        return sorted(self.development_table()["patient_uid"].tolist())

    def blocked_test_patients(self) -> frozenset:
        """Pacientes de la prueba externa. SOLO para comprobar que no aparecen
        en datos, caches, calibracion ni entrenamiento; ninguna etapa de
        desarrollo debe usarlos para otra cosa."""
        table = self.patient_table
        return frozenset(table.loc[table["outer_role"] == OUTER_ROLE_TEST, "patient_uid"])

    def cv_split(self, fold_id: int) -> tuple[list[str], list[str]]:
        """``(train_patients, validation_patients)`` del fold interno ``fold_id``:
        validation = grupo ``fold_id``; train = los otros cuatro grupos."""
        if not 0 <= int(fold_id) < self.n_splits:
            raise ValueError(f"fold_id={fold_id} fuera de rango (0..{self.n_splits - 1})")
        development = self.development_table()
        in_fold = development["inner_fold_group"] == int(fold_id)
        return (
            development.loc[~in_fold, "patient_uid"].tolist(),
            development.loc[in_fold, "patient_uid"].tolist(),
        )

    def cv_fold_table(self) -> pd.DataFrame:
        """Tabla larga ``(fold, patient_uid, role)`` de los folds internos."""
        rows = []
        for fold_id in range(self.n_splits):
            train, val = self.cv_split(fold_id)
            rows += [{"fold": fold_id, "patient_uid": p, "role": ROLE_TRAIN} for p in train]
            rows += [{"fold": fold_id, "patient_uid": p, "role": ROLE_VALIDATION} for p in val]
        return pd.DataFrame(rows, columns=["fold", "patient_uid", "role"])

    def assert_no_blocked_patients(self, patient_ids, context: str = "") -> None:
        leaked = self.blocked_test_patients() & set(patient_ids)
        if leaked:
            where = f"{context}: " if context else ""
            raise RuntimeError(
                f"{where}{len(leaked)} paciente(s) de la prueba externa bloqueada aparecen en "
                f"datos de desarrollo: {sorted(leaked)[:5]}"
            )


def load_holdout_split(csv_path: Path, manifest_path: Path, dataset_scope: str) -> HoldoutSplit:
    """Lee ``holdout_splits.csv`` para un ``dataset_scope`` y verifica que
    coincide con su manifiesto (hash del csv, protocolo, numero de pacientes y
    de folds) antes de reconstruir la separacion."""
    csv_path, manifest_path = Path(csv_path), Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    actual_hash = sha256_file(csv_path)
    expected_hash = manifest.get("holdout_splits_csv_sha256")
    if actual_hash != expected_hash:
        raise ValueError(
            f"{csv_path}: sha256 {actual_hash[:12]}... no coincide con el manifiesto "
            f"{manifest_path} ({str(expected_hash)[:12]}...); regenere con "
            "'python -m modeling.build_master_folds --protocol holdout-v3'"
        )
    if manifest.get("protocol") != HOLDOUT_PROTOCOL_NAME:
        raise ValueError(
            f"{manifest_path}: protocol={manifest.get('protocol')!r}, se esperaba {HOLDOUT_PROTOCOL_NAME!r}"
        )

    full = pd.read_csv(csv_path, dtype={"patient_uid": str, "source_dataset": str})
    missing = set(HOLDOUT_SPLIT_COLUMNS) - set(full.columns)
    if missing:
        raise ValueError(f"{csv_path}: faltan columnas {sorted(missing)}")
    full["calibration_patient"] = full["calibration_patient"].astype(bool)

    table = full.loc[full["dataset_scope"] == dataset_scope].drop(columns="dataset_scope").reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{csv_path}: no hay filas para dataset_scope={dataset_scope!r}")

    expected_n = manifest.get("counts", {}).get(dataset_scope, {}).get("n_patients")
    if expected_n is not None and int(expected_n) != len(table):
        raise ValueError(f"{dataset_scope}: {len(table)} pacientes en el csv, {expected_n} en el manifiesto")

    development = table.loc[table["outer_role"] == OUTER_ROLE_DEVELOPMENT]
    if development.empty:
        raise ValueError(f"{dataset_scope}: ningun paciente de desarrollo")
    n_splits = int(development["inner_fold_group"].max()) + 1
    if manifest.get("n_splits") is not None and int(manifest["n_splits"]) != n_splits:
        raise ValueError(f"{dataset_scope}: {n_splits} folds en el csv, {manifest['n_splits']} en el manifiesto")

    verify_holdout_split(table, n_splits, context=dataset_scope)
    return HoldoutSplit(dataset_scope=dataset_scope, patient_table=table, n_splits=n_splits)
