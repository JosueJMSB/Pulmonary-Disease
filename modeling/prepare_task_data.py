"""Construye subconjuntos COPD vs control desde la salida de la fase 4.

No modifica los arreglos originales. Copia las filas elegibles de ``no_dn``
y ``dn`` y conserva el indice de origen para trazabilidad.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOT = REPO_ROOT / "preprocessing" / "data" / "final"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data" / "copd_vs_control"
BRANCHES = ("no_dn", "dn")
EXPECTED_SOURCE_SHAPE = (8747, 20000)
EXPECTED_SOURCE_DTYPE = np.dtype("float32")


@dataclass(frozen=True)
class TaskSpec:
    """Seleccion clinica y conteos que deben reproducirse."""

    output_name: str
    dataset: str
    diagnoses: tuple[str, ...]
    required_filter: str | None
    expected: dict[str, dict[str, int]]


TASKS = (
    TaskSpec(
        "ICBHI", "ICBHI", ("COPD", "Healthy"), None,
        {
            "COPD": {"patients": 64, "recordings": 793, "segments": 6055},
            "Healthy": {"patients": 26, "recordings": 35, "segments": 240},
        },
    ),
    TaskSpec(
        "FRAIWAN_Extended", "FRAIWAN", ("COPD", "Normal"), "Extended",
        {
            "COPD": {"patients": 9, "recordings": 9, "segments": 51},
            "Normal": {"patients": 34, "recordings": 34, "segments": 190},
        },
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera arreglos para COPD vs control.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def git_state() -> dict[str, str | bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        return {"commit": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def validate_sources(source_root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    csv_path = source_root / "segments.csv"
    paths = {branch: source_root / f"segments_{branch}.npy" for branch in BRANCHES}
    missing = [str(path) for path in (csv_path, *paths.values()) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Faltan entradas de fase 4: " + ", ".join(missing))

    metadata = pd.read_csv(csv_path)
    required = {
        "array_index", "segment_id", "audio_id", "dataset", "patient_uid",
        "diagnosis", "filter", "calibration_patient", "dn_reliable",
    }
    absent = sorted(required.difference(metadata.columns))
    if absent:
        raise ValueError(f"Faltan columnas requeridas: {absent}")
    if len(metadata) != EXPECTED_SOURCE_SHAPE[0]:
        raise ValueError(f"Se esperaban 8747 filas; se encontraron {len(metadata)}.")
    indices = metadata["array_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(len(metadata), dtype=np.int64)):
        raise ValueError("array_index no coincide con el orden de segments.csv.")
    if metadata["segment_id"].duplicated().any():
        raise ValueError("segments.csv contiene segment_id duplicados.")

    arrays: dict[str, np.ndarray] = {}
    for branch, path in paths.items():
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != EXPECTED_SOURCE_SHAPE:
            raise ValueError(f"{path.name}: forma {array.shape} incorrecta.")
        if array.dtype != EXPECTED_SOURCE_DTYPE:
            raise ValueError(f"{path.name}: dtype {array.dtype}, esperado float32.")
        arrays[branch] = array
    return metadata, arrays


def select_task(metadata: pd.DataFrame, spec: TaskSpec) -> pd.DataFrame:
    mask = (metadata["dataset"] == spec.dataset) & metadata["diagnosis"].isin(spec.diagnoses)
    if spec.required_filter is not None:
        mask &= metadata["filter"] == spec.required_filter

    selected = metadata.loc[mask].copy()
    selected = selected.sort_values("array_index", kind="stable").reset_index(drop=True)
    selected = selected.rename(columns={"array_index": "source_array_index"})
    selected.insert(0, "task_array_index", np.arange(len(selected), dtype=np.int64))
    selected.insert(
        selected.columns.get_loc("diagnosis") + 1,
        "target_label",
        (selected["diagnosis"] == "COPD").astype(np.int8),
    )
    selected.insert(
        selected.columns.get_loc("target_label") + 1,
        "target_name",
        np.where(selected["diagnosis"] == "COPD", "COPD", "Control"),
    )

    if selected.empty:
        raise ValueError(f"La seleccion {spec.output_name} quedo vacia.")
    if set(selected["diagnosis"].unique()) != set(spec.diagnoses):
        raise ValueError(f"{spec.output_name}: faltan clases esperadas.")
    if selected["segment_id"].duplicated().any():
        raise ValueError(f"{spec.output_name}: hay segmentos duplicados.")
    per_patient_labels = selected.groupby("patient_uid")["target_label"].nunique()
    if (per_patient_labels != 1).any():
        bad = per_patient_labels[per_patient_labels != 1].index.tolist()
        raise ValueError(f"{spec.output_name}: pacientes con mas de una etiqueta: {bad}")
    if spec.required_filter is not None:
        found_filters = set(selected["filter"].dropna().unique())
        if found_filters != {spec.required_filter}:
            raise ValueError(
                f"{spec.output_name}: filtros {found_filters}, esperado {spec.required_filter}."
            )

    for diagnosis, expected in spec.expected.items():
        group = selected[selected["diagnosis"] == diagnosis]
        observed = {
            "patients": int(group["patient_uid"].nunique()),
            "recordings": int(group["audio_id"].nunique()),
            "segments": int(len(group)),
        }
        if observed != expected:
            raise ValueError(
                f"{spec.output_name}/{diagnosis}: conteos {observed}, esperados {expected}."
            )
    return selected


def copy_and_verify(
    source: np.ndarray,
    source_indices: np.ndarray,
    destination: Path,
    chunk_size: int,
) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    output = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=EXPECTED_SOURCE_DTYPE,
        shape=(len(source_indices), source.shape[1]),
    )
    try:
        for start in range(0, len(source_indices), chunk_size):
            stop = min(start + chunk_size, len(source_indices))
            block = np.asarray(source[source_indices[start:stop]], dtype=EXPECTED_SOURCE_DTYPE)
            if not np.isfinite(block).all():
                raise ValueError(f"NaN o Inf en filas {start}:{stop} de {destination.name}.")
            output[start:stop] = block
        output.flush()
    finally:
        del output
    os.replace(temporary, destination)

    written = np.load(destination, mmap_mode="r", allow_pickle=False)
    expected_shape = (len(source_indices), source.shape[1])
    if written.shape != expected_shape or written.dtype != EXPECTED_SOURCE_DTYPE:
        raise ValueError(f"Salida invalida: {destination}")
    for start in range(0, len(source_indices), chunk_size):
        stop = min(start + chunk_size, len(source_indices))
        if not np.array_equal(written[start:stop], source[source_indices[start:stop]]):
            raise ValueError(
                f"Copia distinta del origen en {destination.name}, filas {start}:{stop}."
            )


def class_counts(metadata: pd.DataFrame) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for diagnosis, group in metadata.groupby("diagnosis", sort=True):
        calibration = group["calibration_patient"].astype(str).str.lower().eq("true")
        reliable = group["dn_reliable"].astype(str).str.lower().eq("true")
        result[str(diagnosis)] = {
            "patients": int(group["patient_uid"].nunique()),
            "recordings": int(group["audio_id"].nunique()),
            "segments": int(len(group)),
            "calibration_patients": int(group.loc[calibration, "patient_uid"].nunique()),
            "dn_unreliable_recordings": int(group.loc[~reliable, "audio_id"].nunique()),
        }
    return result


def build_task(
    staging_root: Path,
    source_root: Path,
    source_arrays: dict[str, np.ndarray],
    selected: pd.DataFrame,
    spec: TaskSpec,
    source_hashes: dict[str, str],
    chunk_size: int,
) -> dict[str, Any]:
    task_dir = staging_root / spec.output_name
    task_dir.mkdir(parents=True, exist_ok=False)
    source_indices = selected["source_array_index"].to_numpy(dtype=np.int64)

    output_hashes: dict[str, str] = {}
    for branch in BRANCHES:
        output_path = task_dir / f"segments_{branch}.npy"
        copy_and_verify(source_arrays[branch], source_indices, output_path, chunk_size)
        output_hashes[output_path.name] = sha256_file(output_path)

    inventory_path = task_dir / "segments.csv"
    selected.to_csv(inventory_path, index=False, lineterminator="\n")
    output_hashes[inventory_path.name] = sha256_file(inventory_path)

    no_dn = np.load(task_dir / "segments_no_dn.npy", mmap_mode="r", allow_pickle=False)
    dn = np.load(task_dir / "segments_dn.npy", mmap_mode="r", allow_pickle=False)
    if no_dn.shape != dn.shape:
        raise ValueError(f"{spec.output_name}: ramas con formas diferentes.")

    manifest: dict[str, Any] = {
        "verdict": "PASS",
        "task": "COPD_vs_Control",
        "output_name": spec.output_name,
        "dataset": spec.dataset,
        "diagnoses": list(spec.diagnoses),
        "label_mapping": {"COPD": 1, "Healthy": 0, "Normal": 0},
        "required_filter": spec.required_filter,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(),
        "source_root": str(source_root.resolve()),
        "source_shape": list(EXPECTED_SOURCE_SHAPE),
        "output_shape": [int(no_dn.shape[0]), int(no_dn.shape[1])],
        "dtype": str(no_dn.dtype),
        "class_counts": class_counts(selected),
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


def publish(staging_root: Path, output_root: Path, overwrite: bool) -> None:
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"La salida ya existe: {output_root}. Use --overwrite para regenerarla."
        )
    backup = output_root.with_name(output_root.name + "_previous")
    if backup.exists():
        shutil.rmtree(backup)
    if output_root.exists():
        os.replace(output_root, backup)
    try:
        os.replace(staging_root, output_root)
    except Exception:
        if backup.exists() and not output_root.exists():
            os.replace(backup, output_root)
        raise
    if backup.exists():
        shutil.rmtree(backup)


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

    metadata, source_arrays = validate_sources(source_root)
    source_hashes = {
        "segments.csv": sha256_file(source_root / "segments.csv"),
        **{
            f"segments_{branch}.npy": sha256_file(source_root / f"segments_{branch}.npy")
            for branch in BRANCHES
        },
    }

    staging_root = Path(
        tempfile.mkdtemp(prefix=output_root.name + "_staging_", dir=output_root.parent)
    )
    try:
        manifests = []
        for spec in TASKS:
            selected = select_task(metadata, spec)
            print(f"{spec.output_name}: copiando {len(selected)} segmentos por rama...", flush=True)
            manifest = build_task(
                staging_root, source_root, source_arrays, selected, spec,
                source_hashes, args.chunk_size,
            )
            manifests.append(manifest)
            print(f"{spec.output_name}: PASS {manifest['output_shape']}", flush=True)

        root_manifest = {
            "verdict": "PASS",
            "task": "COPD_vs_Control",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "datasets": [manifest["output_name"] for manifest in manifests],
            "total_segments_per_branch": int(
                sum(manifest["output_shape"][0] for manifest in manifests)
            ),
            "git": git_state(),
            "source_hashes": source_hashes,
        }
        (staging_root / "manifest.json").write_text(
            json.dumps(root_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        publish(staging_root, output_root, args.overwrite)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise

    print(f"Salida publicada en: {output_root}")
    print("Veredicto global: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
