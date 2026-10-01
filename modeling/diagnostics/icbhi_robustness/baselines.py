"""Controles negativos y baseline de dispositivo para ICBHI."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ... import evaluation as ev
from ... import splits as sp


def patient_metadata(segments: pd.DataFrame) -> pd.DataFrame:
    required = {"patient_uid", "target_label", "device"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"faltan columnas para el baseline de dispositivo: {sorted(missing)}")
    labels = segments.groupby("patient_uid")["target_label"].nunique()
    if (labels != 1).any():
        raise ValueError("hay pacientes con etiquetas mixtas")
    rows = []
    for patient_uid, group in segments.groupby("patient_uid", sort=True):
        devices = tuple(sorted(set(group["device"].fillna("unknown").astype(str))))
        rows.append(
            {
                "patient_uid": str(patient_uid),
                "target_label": int(group["target_label"].iloc[0]),
                "devices": devices,
                "device_signature": "|".join(devices),
            }
        )
    return pd.DataFrame(rows)


def device_matrix(
    table: pd.DataFrame, patient_ids: list[str], categories: tuple[str, ...]
) -> tuple[pd.DataFrame, np.ndarray]:
    part = (
        table.set_index("patient_uid")
        .loc[list(patient_ids)]
        .reset_index()
    )
    matrix = np.zeros((len(part), len(categories)), dtype=np.float64)
    category_index = {name: index for index, name in enumerate(categories)}
    for row_index, devices in enumerate(part["devices"]):
        for device in devices:
            if device in category_index:
                matrix[row_index, category_index[device]] = 1.0
    return part, matrix


def run_device_baseline(
    split: sp.HoldoutSplit,
    fold_segments: dict[int, pd.DataFrame],
    settings: dict,
    random_state: int,
    negative_label_name: str,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Regresion logistica que recibe unicamente el/los dispositivos."""
    fold_rows = []
    prediction_frames = []
    for fold_id in range(split.n_splits):
        segments = fold_segments[fold_id]
        split.assert_no_blocked_patients(segments["patient_uid"], f"device_baseline/fold_{fold_id:02d}")
        metadata = patient_metadata(segments)
        train_ids, val_ids = split.cv_split(fold_id)
        train_devices = metadata.loc[
            metadata["patient_uid"].isin(train_ids), "devices"
        ]
        categories = tuple(sorted({device for devices in train_devices for device in devices}))
        if not categories:
            raise ValueError(f"fold {fold_id}: train no contiene dispositivos")
        train, x_train = device_matrix(metadata, train_ids, categories)
        validation, x_validation = device_matrix(metadata, val_ids, categories)
        model = LogisticRegression(
            C=float(settings["C"]),
            class_weight=settings["class_weight"],
            solver=str(settings["solver"]),
            random_state=int(random_state),
            max_iter=1000,
        )
        model.fit(x_train, train["target_label"].to_numpy(dtype=np.int64))
        scores = model.predict_proba(x_validation)[:, 1]
        predictions = validation[
            ["patient_uid", "target_label", "device_signature"]
        ].assign(score=scores, fold=fold_id)
        prediction_frames.append(predictions)
        metrics = ev.compute_patient_metrics(
            predictions["target_label"].to_numpy(),
            predictions["score"].to_numpy(),
            negative_label_name,
            threshold,
        )
        metrics["recall_negative"] = metrics[
            f"recall_{negative_label_name.lower()}"
        ]
        metrics["min_class_recall"] = float(
            min(metrics["recall_copd"], metrics["recall_negative"])
        )
        fold_rows.append({"fold": fold_id, **metrics, "categories": "|".join(categories)})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    counts = predictions["patient_uid"].value_counts()
    if set(counts.index) != set(split.development_patients()) or not (counts == 1).all():
        raise RuntimeError("el baseline no produjo exactamente una prediccion por paciente de development")
    pooled = ev.compute_patient_metrics(
        predictions["target_label"].to_numpy(),
        predictions["score"].to_numpy(),
        negative_label_name,
        threshold,
    )
    pooled["recall_negative"] = pooled[f"recall_{negative_label_name.lower()}"]
    pooled["min_class_recall"] = float(
        min(pooled["recall_copd"], pooled["recall_negative"])
    )
    return pd.DataFrame(fold_rows), predictions, pooled


def permutation_map(
    split: sp.HoldoutSplit,
    random_state: int,
    max_attempts: int,
) -> pd.DataFrame:
    """Permuta etiquetas por paciente conservando conteos y ambas clases/fold."""
    development = (
        split.development_table()[["patient_uid", "target_label"]]
        .sort_values("patient_uid")
        .reset_index(drop=True)
    )
    original = development["target_label"].to_numpy(dtype=np.int64)
    rng = np.random.RandomState(int(random_state))
    patient_to_fold = development["patient_uid"].map(
        {
            patient: fold
            for fold in range(split.n_splits)
            for patient in split.cv_split(fold)[1]
        }
    )
    if patient_to_fold.isna().any():
        raise RuntimeError("hay pacientes de development sin fold interno")
    folds = patient_to_fold.to_numpy(dtype=np.int64)
    permuted = None
    for _attempt in range(int(max_attempts)):
        candidate = rng.permutation(original)
        if np.array_equal(candidate, original):
            continue
        valid = True
        for fold in range(split.n_splits):
            validation_labels = candidate[folds == fold]
            train_labels = candidate[folds != fold]
            if set(validation_labels) != {0, 1} or set(train_labels) != {0, 1}:
                valid = False
                break
        if valid:
            permuted = candidate
            break
    if permuted is None:
        raise RuntimeError("no se encontro una permutacion valida con ambas clases en todos los folds")
    output = development.rename(columns={"target_label": "original_target_label"})
    output["permuted_target_label"] = permuted
    output["fold"] = folds
    payload = "\n".join(
        f"{row.patient_uid}:{row.permuted_target_label}"
        for row in output.itertuples(index=False)
    )
    output["permutation_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return output


def apply_permuted_labels(
    frame: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    labels = mapping.set_index("patient_uid")["permuted_target_label"]
    result = frame.copy()
    result["target_label"] = result["patient_uid"].map(labels)
    if result["target_label"].isna().any():
        missing = sorted(
            result.loc[result["target_label"].isna(), "patient_uid"].unique()
        )
        raise RuntimeError(f"faltan etiquetas permutadas para {missing[:5]}")
    result["target_label"] = result["target_label"].astype(np.int64)
    return result

