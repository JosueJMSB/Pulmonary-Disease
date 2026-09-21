"""Construye el dataset logico combinado COPD vs control.

Combina, en una unica coleccion ``COMBINED``, la cohorte ICBHI (COPD y
Healthy, sin filtrar por dispositivo) con la cohorte Fraiwan Extended
(COPD y Normal, unicamente ``filter == "Extended"``). Reutiliza las
funciones seguras de ``prepare_task_data`` (validacion de fuentes, copia
por bloques con verificacion, hashes SHA-256 y publicacion atomica) para
no volver a procesar audio ni modificar los arreglos originales.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import prepare_task_data as ptd

DEFAULT_SOURCE_ROOT = ptd.DEFAULT_SOURCE_ROOT
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "copd_vs_control_combined"
BRANCHES = ptd.BRANCHES
OUTPUT_NAME = "COMBINED"

# (dataset, diagnosticos admitidos, filtro requerido o None)
COHORTS: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("ICBHI", ("COPD", "Healthy"), None),
    ("FRAIWAN", ("COPD", "Normal"), "Extended"),
)

LABEL_MAPPING = {"COPD": 1, "Healthy": 0, "Normal": 0}

EXPECTED_GROUPS: dict[tuple[str, str], dict[str, int]] = {
    ("ICBHI", "COPD"): {"patients": 64, "recordings": 793, "segments": 6055},
    ("ICBHI", "Healthy"): {"patients": 26, "recordings": 35, "segments": 240},
    ("FRAIWAN", "COPD"): {"patients": 9, "recordings": 9, "segments": 51},
    ("FRAIWAN", "Normal"): {"patients": 34, "recordings": 34, "segments": 190},
}
EXPECTED_TOTAL: dict[str, int] = {"patients": 133, "recordings": 871, "segments": 6536}
EXPECTED_HARMONIZED: dict[str, dict[str, int]] = {
    "COPD": {"patients": 73, "recordings": 802, "segments": 6106},
    "Control": {"patients": 60, "recordings": 69, "segments": 430},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera el dataset combinado COPD vs control (ICBHI + Fraiwan Extended)."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_combined(metadata: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y armoniza las cohortes definidas en ``COHORTS``.

    No depende de conteos absolutos: solo aplica los filtros clinicos, la
    armonizacion de etiquetas, el orden estable y las verificaciones
    estructurales (duplicados, una etiqueta por paciente, pacientes no
    compartidos entre fuentes y filtros exclusivos). La comparacion contra
    los conteos esperados del proyecto se hace por separado en
    ``verify_expected_counts``.
    """

    combined_mask = None
    for dataset, diagnoses, required_filter in COHORTS:
        mask = (metadata["dataset"] == dataset) & metadata["diagnosis"].isin(diagnoses)
        if required_filter is not None:
            mask &= metadata["filter"] == required_filter
        combined_mask = mask if combined_mask is None else (combined_mask | mask)

    selected = metadata.loc[combined_mask].copy()
    selected = selected.sort_values("array_index", kind="stable").reset_index(drop=True)
    selected = selected.rename(columns={"array_index": "source_array_index"})
    selected.insert(0, "task_array_index", np.arange(len(selected), dtype=np.int64))

    is_copd = selected["diagnosis"] == "COPD"
    selected.insert(
        selected.columns.get_loc("diagnosis") + 1,
        "target_label",
        is_copd.astype(np.int8),
    )
    selected.insert(
        selected.columns.get_loc("target_label") + 1,
        "target_name",
        np.where(is_copd, "COPD", "Control"),
    )

    if selected.empty:
        raise ValueError("La seleccion combinada quedo vacia.")

    for dataset, diagnoses, required_filter in COHORTS:
        group = selected[selected["dataset"] == dataset]
        found_diagnoses = set(group["diagnosis"].unique())
        if found_diagnoses != set(diagnoses):
            raise ValueError(
                f"{dataset}: se esperaban los diagnosticos {diagnoses}, "
                f"se obtuvieron {sorted(found_diagnoses)}."
            )
        if required_filter is not None:
            found_filters = set(group["filter"].dropna().unique())
            if found_filters != {required_filter}:
                raise ValueError(
                    f"{dataset}: filtros {sorted(found_filters)}, "
                    f"esperado unicamente {required_filter!r}."
                )

    if selected["segment_id"].duplicated().any():
        dupes = selected.loc[selected["segment_id"].duplicated(), "segment_id"].tolist()
        raise ValueError(f"segment_id duplicados en la seleccion combinada: {dupes}")

    per_patient_labels = selected.groupby("patient_uid")["target_label"].nunique()
    if (per_patient_labels != 1).any():
        bad = per_patient_labels[per_patient_labels != 1].index.tolist()
        raise ValueError(f"Pacientes con mas de una etiqueta: {bad}")

    datasets = [cohort[0] for cohort in COHORTS]
    for i, dataset_a in enumerate(datasets):
        patients_a = set(selected.loc[selected["dataset"] == dataset_a, "patient_uid"])
        for dataset_b in datasets[i + 1 :]:
            patients_b = set(selected.loc[selected["dataset"] == dataset_b, "patient_uid"])
            overlap = patients_a & patients_b
            if overlap:
                raise ValueError(
                    f"patient_uid compartidos entre {dataset_a} y {dataset_b}: {sorted(overlap)}"
                )

    if not np.array_equal(
        selected["task_array_index"].to_numpy(dtype=np.int64),
        np.arange(len(selected), dtype=np.int64),
    ):
        raise ValueError("task_array_index no es consecutivo o no esta alineado.")

    return selected


def _counts(group: pd.DataFrame) -> dict[str, int]:
    return {
        "patients": int(group["patient_uid"].nunique()),
        "recordings": int(group["audio_id"].nunique()),
        "segments": int(len(group)),
    }


def verify_expected_counts(
    selected: pd.DataFrame,
    expected_groups: dict[tuple[str, str], dict[str, int]] | None = None,
    expected_total: dict[str, int] | None = None,
    expected_harmonized: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, dict[str, int]], dict[str, int], dict[str, dict[str, int]]]:
    """Compara la seleccion contra los conteos esperados del proyecto.

    Los parametros ``expected_*`` son inyectables para poder probar la
    logica de comparacion con datos sinteticos; en produccion se usan los
    conteos reales definidos como constantes del modulo.
    """

    expected_groups = EXPECTED_GROUPS if expected_groups is None else expected_groups
    expected_total = EXPECTED_TOTAL if expected_total is None else expected_total
    expected_harmonized = EXPECTED_HARMONIZED if expected_harmonized is None else expected_harmonized

    observed_groups: dict[str, dict[str, int]] = {}
    for (dataset, diagnosis), expected in expected_groups.items():
        group = selected[(selected["dataset"] == dataset) & (selected["diagnosis"] == diagnosis)]
        observed = _counts(group)
        if observed != expected:
            raise ValueError(f"{dataset}/{diagnosis}: conteos {observed}, esperados {expected}.")
        observed_groups[f"{dataset}/{diagnosis}"] = observed

    observed_total = _counts(selected)
    if observed_total != expected_total:
        raise ValueError(f"Total combinado: conteos {observed_total}, esperados {expected_total}.")

    observed_harmonized: dict[str, dict[str, int]] = {}
    for target_name, expected in expected_harmonized.items():
        group = selected[selected["target_name"] == target_name]
        observed = _counts(group)
        if observed != expected:
            raise ValueError(
                f"{target_name}: conteos armonizados {observed}, esperados {expected}."
            )
        observed_harmonized[target_name] = observed

    return observed_groups, observed_total, observed_harmonized


def build_combined(
    staging_root: Path,
    source_root: Path,
    source_arrays: dict[str, np.ndarray],
    selected: pd.DataFrame,
    source_hashes: dict[str, str],
    chunk_size: int,
    group_counts: dict[str, dict[str, int]],
    total_counts: dict[str, int],
    harmonized_counts: dict[str, dict[str, int]],
) -> dict[str, Any]:
    task_dir = staging_root / OUTPUT_NAME
    task_dir.mkdir(parents=True, exist_ok=False)
    source_indices = selected["source_array_index"].to_numpy(dtype=np.int64)

    output_hashes: dict[str, str] = {}
    for branch in BRANCHES:
        output_path = task_dir / f"segments_{branch}.npy"
        ptd.copy_and_verify(source_arrays[branch], source_indices, output_path, chunk_size)
        output_hashes[output_path.name] = ptd.sha256_file(output_path)

    inventory_path = task_dir / "segments.csv"
    selected.to_csv(inventory_path, index=False, lineterminator="\n")
    output_hashes[inventory_path.name] = ptd.sha256_file(inventory_path)

    no_dn = np.load(task_dir / "segments_no_dn.npy", mmap_mode="r", allow_pickle=False)
    dn = np.load(task_dir / "segments_dn.npy", mmap_mode="r", allow_pickle=False)
    if no_dn.shape != dn.shape:
        raise ValueError("COMBINED: las ramas no_dn y dn tienen formas diferentes.")

    reread = pd.read_csv(inventory_path)
    if len(reread) != no_dn.shape[0]:
        raise ValueError(
            "COMBINED: segments.csv y los arreglos .npy no tienen el mismo numero de filas."
        )
    if not np.array_equal(
        reread["task_array_index"].to_numpy(dtype=np.int64),
        np.arange(len(reread), dtype=np.int64),
    ):
        raise ValueError("COMBINED: task_array_index desalineado tras escribir segments.csv.")

    source_shape = [int(source_arrays["no_dn"].shape[0]), int(source_arrays["no_dn"].shape[1])]

    manifest: dict[str, Any] = {
        "verdict": "PASS",
        "task": "COPD_vs_Control_Combined",
        "output_name": OUTPUT_NAME,
        "cohorts": [
            {"dataset": dataset, "diagnoses": list(diagnoses), "required_filter": required_filter}
            for dataset, diagnoses, required_filter in COHORTS
        ],
        "label_mapping": LABEL_MAPPING,
        "ordering": (
            "ordenado por array_index original de forma estable "
            "(renombrado a source_array_index); task_array_index = 0..N-1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": ptd.git_state(),
        "source_root": str(source_root.resolve()),
        "source_shape": source_shape,
        "output_shape": [int(no_dn.shape[0]), int(no_dn.shape[1])],
        "dtype": str(no_dn.dtype),
        "counts_by_dataset_diagnosis": group_counts,
        "counts_total": total_counts,
        "counts_harmonized": harmonized_counts,
        "source_hashes": source_hashes,
        "output_hashes": output_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (task_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("--chunk-size debe ser mayor que cero.")
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"La salida ya existe: {output_root}. Use --overwrite para regenerarla."
        )
    output_root.parent.mkdir(parents=True, exist_ok=True)

    metadata, source_arrays = ptd.validate_sources(source_root)
    source_hashes = {
        "segments.csv": ptd.sha256_file(source_root / "segments.csv"),
        **{
            f"segments_{branch}.npy": ptd.sha256_file(source_root / f"segments_{branch}.npy")
            for branch in BRANCHES
        },
    }

    selected = select_combined(metadata)
    group_counts, total_counts, harmonized_counts = verify_expected_counts(selected)
    print(f"COMBINED: copiando {len(selected)} segmentos por rama...", flush=True)

    staging_root = Path(
        tempfile.mkdtemp(prefix=output_root.name + "_staging_", dir=output_root.parent)
    )
    try:
        manifest = build_combined(
            staging_root, source_root, source_arrays, selected,
            source_hashes, args.chunk_size, group_counts, total_counts, harmonized_counts,
        )
        ptd.publish(staging_root, output_root, args.overwrite)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(f"COMBINED: PASS {manifest['output_shape']}", flush=True)
    print(f"Salida publicada en: {output_root / OUTPUT_NAME}")
    print("Veredicto global: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
