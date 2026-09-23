"""Asignacion maestra de folds para el protocolo fold-aware (v2).

    python -m modeling.build_master_folds

Construye, de una sola vez, la particion de 5 folds por paciente que
``preprocessing/fold_denoising.py`` y las corridas de SVM/CNN/CRNN (v2)
deben reutilizar exactamente igual -ningun consumidor vuelve a correr
``StratifiedKFold``. A diferencia del protocolo actual, los pacientes
historicamente marcados como ``calibration_patient`` ya NO quedan fuera de
la rotacion (``respect_calibration_patient=False``): la columna se conserva
en el csv solo como trazabilidad.

ICBHI y Fraiwan Extended se estratifican por separado, cada uno solo por
clase (ambos superan el minimo de 5 pacientes por clase). COMBINED no
sortea sus propios folds: cada paciente hereda el ``fold_group`` que ya
tenia en la particion de su fuente (ICBHI o Fraiwan), de modo que los tres
protocolos (ICBHI, Fraiwan, COMBINED) son consistentes entre si por
construccion, no por una estratificacion conjunta aparte.

No toca audio en absoluto: parte de los ``segments.csv`` que
``prepare_task_data.py``/``prepare_combined_task_data.py`` ya generaron
(dataset, patient_uid, target_label, calibration_patient), intactos.

Escribe, siempre juntos y versionados:

    modeling/data/patient_folds.csv
    modeling/data/patient_folds_manifest.json
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd

from . import splits as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ICBHI_SEGMENTS = REPO_ROOT / "modeling" / "data" / "copd_vs_control" / "ICBHI" / "segments.csv"
DEFAULT_FRAIWAN_SEGMENTS = REPO_ROOT / "modeling" / "data" / "copd_vs_control" / "FRAIWAN_Extended" / "segments.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "modeling" / "data" / "patient_folds.csv"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "modeling" / "data" / "patient_folds_manifest.json"

N_SPLITS = 5
RANDOM_STATE = 20260914


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera la asignacion maestra de folds (protocolo fold-aware v2)."
    )
    parser.add_argument("--icbhi-segments", type=Path, default=DEFAULT_ICBHI_SEGMENTS)
    parser.add_argument("--fraiwan-segments", type=Path, default=DEFAULT_FRAIWAN_SEGMENTS)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT_MANIFEST)
    return parser.parse_args(argv)


def _git_state() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _counts_for_scope(frame: pd.DataFrame) -> dict:
    scoped = frame
    by_fold_group = {
        str(int(k)): int(v) for k, v in scoped["fold_group"].value_counts().sort_index().items()
    }
    by_target_label = {
        str(int(k)): int(v) for k, v in scoped["target_label"].value_counts().sort_index().items()
    }
    return {
        "n_patients": int(len(scoped)),
        "by_fold_group": by_fold_group,
        "by_target_label": by_target_label,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    icbhi_segments = pd.read_csv(args.icbhi_segments, dtype={"patient_uid": str})
    fraiwan_segments = pd.read_csv(args.fraiwan_segments, dtype={"patient_uid": str})

    icbhi_folds = sp.build_patient_folds(
        icbhi_segments, n_splits=N_SPLITS, random_state=RANDOM_STATE,
        stratify_by_dataset=False, respect_calibration_patient=False,
    )
    fraiwan_folds = sp.build_patient_folds(
        fraiwan_segments, n_splits=N_SPLITS, random_state=RANDOM_STATE,
        stratify_by_dataset=False, respect_calibration_patient=False,
    )
    combined_folds = sp.combine_patient_folds(
        {"ICBHI": icbhi_folds, "FRAIWAN_Extended": fraiwan_folds}, n_splits=N_SPLITS,
    )

    scopes = {
        "ICBHI": icbhi_folds,
        "FRAIWAN_Extended": fraiwan_folds,
        "COMBINED": combined_folds,
    }
    frame = pd.concat(
        [sp.patient_folds_to_frame(folds, scope) for scope, folds in scopes.items()],
        ignore_index=True,
    )

    manifest_extra = {
        "git": _git_state(),
        "source_segments_sha256": {
            "ICBHI": sp.sha256_file(args.icbhi_segments),
            "FRAIWAN_Extended": sp.sha256_file(args.fraiwan_segments),
        },
        "seeds": {
            "ICBHI": {"n_splits": N_SPLITS, "random_state": RANDOM_STATE},
            "FRAIWAN_Extended": {"n_splits": N_SPLITS, "random_state": RANDOM_STATE},
            "COMBINED": "hereda fold_group de ICBHI y FRAIWAN_Extended; sin sorteo propio",
        },
        "counts": {
            scope: _counts_for_scope(frame.loc[frame["dataset_scope"] == scope])
            for scope in scopes
        },
    }

    manifest = sp.write_patient_folds_csv(frame, args.output_csv, args.output_manifest, manifest_extra)

    print(f"patient_folds.csv -> {args.output_csv}")
    print(f"patient_folds_manifest.json -> {args.output_manifest}")
    for scope, counts in manifest["counts"].items():
        print(
            f"  {scope}: {counts['n_patients']} pacientes, "
            f"por fold_group={counts['by_fold_group']}, por clase={counts['by_target_label']}"
        )
    print("Veredicto: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
