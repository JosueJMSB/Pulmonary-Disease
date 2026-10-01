"""Genera folds internos repetidos conservando intacto el holdout externo."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ... import artifacts as art
from ... import data as dmod
from ... import splits as sp

from .common import DATASET, load_protocol, resolve_runtime_path, seed_tag, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea repartos nuevos del development de ICBHI; "
            "nunca cambia ni procesa el test externo."
        )
    )
    parser.add_argument("--protocol-config", type=Path, default=None)
    parser.add_argument(
        "--reference-data-root",
        type=Path,
        required=True,
        help=(
            "Raiz holdout_calibrated v3 original. Solo se lee "
            "ICBHI/cv/fold_00/segments.csv para obtener dispositivos de development."
        ),
    )
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def patient_device_table(reference_data_root: Path, split: sp.HoldoutSplit) -> pd.DataFrame:
    segments = dmod.load_task_segments(Path(reference_data_root), DATASET, "cv/fold_00")
    split.assert_no_blocked_patients(segments["patient_uid"], "device_signature")
    expected = set(split.development_patients())
    present = set(segments["patient_uid"])
    if present != expected:
        raise RuntimeError(
            "el fold de referencia no contiene exactamente development: "
            f"sobran={sorted(present - expected)[:5]}, faltan={sorted(expected - present)[:5]}"
        )
    if "device" not in segments.columns or segments["device"].isna().any():
        raise ValueError("segments.csv necesita device completo para estratificar")
    labels = segments.groupby("patient_uid")["target_label"].nunique()
    if (labels != 1).any():
        raise ValueError("hay pacientes con etiquetas mixtas")

    rows = []
    for patient_uid, group in segments.groupby("patient_uid", sort=True):
        signature = "|".join(sorted(set(group["device"].astype(str))))
        rows.append(
            {
                "patient_uid": str(patient_uid),
                "target_label": int(group["target_label"].iloc[0]),
                "device_signature": signature,
            }
        )
    return pd.DataFrame(rows)


def repeated_split_frame(
    original: sp.HoldoutSplit, devices: pd.DataFrame, seed: int
) -> pd.DataFrame:
    table = original.patient_table.copy()
    development = table.loc[
        table["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT
    ].copy()
    development = development.merge(
        devices,
        on=["patient_uid", "target_label"],
        how="left",
        validate="one_to_one",
    )
    if development["device_signature"].isna().any():
        raise RuntimeError("faltan dispositivos en pacientes de development")
    development["stratum"] = (
        development["target_label"].astype(str)
        + ":"
        + development["device_signature"]
    )
    counts = development["stratum"].value_counts()
    if int(counts.min()) < original.n_splits:
        raise ValueError(
            "no se puede estratificar clase+dispositivo en 5 folds; "
            f"estratos insuficientes: {counts[counts < original.n_splits].to_dict()}"
        )

    development = development.sort_values("patient_uid").reset_index()
    splitter = StratifiedKFold(
        n_splits=original.n_splits, shuffle=True, random_state=int(seed)
    )
    fold_group = pd.Series(index=development.index, dtype="int64")
    for fold_id, (_train, validation) in enumerate(
        splitter.split(development["patient_uid"], development["stratum"])
    ):
        fold_group.iloc[validation] = int(fold_id)
    table.loc[development["index"], "inner_fold_group"] = fold_group.to_numpy(
        dtype="int64"
    )

    columns = ["patient_uid", "outer_role", "inner_fold_group", "target_label"]
    original_test = (
        original.patient_table.loc[
            original.patient_table["outer_role"] == sp.OUTER_ROLE_TEST, columns
        ]
        .sort_values("patient_uid")
        .reset_index(drop=True)
    )
    repeated_test = (
        table.loc[table["outer_role"] == sp.OUTER_ROLE_TEST, columns]
        .sort_values("patient_uid")
        .reset_index(drop=True)
    )
    if not original_test.equals(repeated_test):
        raise RuntimeError("el holdout externo cambio al construir los folds diagnosticos")
    sp.verify_holdout_split(
        table, original.n_splits, context=f"{DATASET}/{seed_tag(seed)}"
    )
    return sp.holdout_split_to_frame(table, DATASET)


def counts_manifest(frame: pd.DataFrame) -> dict:
    by_role = frame["outer_role"].value_counts().sort_index().to_dict()
    by_role_class = (
        frame.groupby(["outer_role", "target_label"])
        .size()
        .rename_axis(["outer_role", "target_label"])
        .to_dict()
    )
    return {
        "n_patients": int(frame["patient_uid"].nunique()),
        "by_role": {str(key): int(value) for key, value in by_role.items()},
        "by_role_class": {
            f"{role}|{label}": int(value)
            for (role, label), value in by_role_class.items()
        },
    }


def assignment_signature(frame: pd.DataFrame) -> tuple[tuple[str, int], ...]:
    development = frame.loc[
        frame["outer_role"] == sp.OUTER_ROLE_DEVELOPMENT,
        ["patient_uid", "inner_fold_group"],
    ]
    return tuple(
        (str(row.patient_uid), int(row.inner_fold_group))
        for row in development.sort_values("patient_uid").itertuples(index=False)
    )


def build_all(
    protocol,
    reference_data_root: Path,
    workspace_root: Path,
    overwrite: bool,
) -> list[dict]:
    original_csv, original_manifest = protocol.original_split_paths
    original = sp.load_holdout_split(original_csv, original_manifest, DATASET)
    devices = patient_device_table(reference_data_root, original)
    workspace_root.mkdir(parents=True, exist_ok=True)
    devices.to_csv(
        workspace_root / "development_patient_devices.csv",
        index=False,
        lineterminator="\n",
    )
    original_manifest_hash = art.sha256_file(original_manifest)
    original_csv_hash = art.sha256_file(original_csv)
    original_frame = sp.holdout_split_to_frame(original.patient_table, DATASET)
    seen_assignments = {assignment_signature(original_frame): "split_original"}
    rows = []
    for seed in protocol.seeds:
        target = workspace_root / "splits" / seed_tag(seed)
        csv_path = target / "holdout_splits.csv"
        manifest_path = target / "holdout_splits_manifest.json"
        if (csv_path.exists() or manifest_path.exists()) and not overwrite:
            raise FileExistsError(
                f"ya existe {target}; use --overwrite solo si pretende regenerarlo"
            )
        frame = repeated_split_frame(original, devices, seed)
        signature = assignment_signature(frame)
        if signature in seen_assignments:
            raise RuntimeError(
                f"{seed_tag(seed)} repite exactamente la asignacion de "
                f"{seen_assignments[signature]}; cambie esa semilla en protocol.toml"
            )
        seen_assignments[signature] = seed_tag(seed)
        manifest = sp.write_holdout_split_csv(
            frame,
            csv_path,
            manifest_path,
            manifest_extra={
                "diagnostic_protocol": protocol.raw["protocol"],
                "dataset_scope": DATASET,
                "n_splits": protocol.n_splits,
                "random_state": int(seed),
                "stratification": protocol.raw["stratification"],
                "outer_holdout_preserved": True,
                "outer_test_accessed": False,
                "source_split_csv_sha256": original_csv_hash,
                "source_split_manifest_sha256": original_manifest_hash,
                "counts": {DATASET: counts_manifest(frame)},
            },
        )
        rows.append(
            {
                "seed": seed,
                "split_csv": str(csv_path),
                "split_manifest": str(manifest_path),
                "split_csv_sha256": manifest["holdout_splits_csv_sha256"],
            }
        )
    write_json(
        workspace_root / "splits_manifest.json",
        {
            "protocol": protocol.raw["protocol"],
            "outer_test_accessed": False,
            "source_split_csv_sha256": original_csv_hash,
            "repetitions": rows,
        },
    )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    protocol = load_protocol(args.protocol_config)
    workspace_root = resolve_runtime_path(
        args.workspace_root,
        "PULMONARY_ICBHI_ROBUSTNESS_ROOT",
        "modeling/runtime/icbhi_robustness",
    )
    rows = build_all(
        protocol,
        args.reference_data_root.resolve(),
        workspace_root,
        args.overwrite,
    )
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"Splits diagnosticos escritos en {workspace_root / 'splits'}; "
        "test externo intacto y bloqueado"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

