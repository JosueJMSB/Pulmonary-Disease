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

Con ``--protocol holdout-v3`` (PLAN-EXPERIMENTO FINAL.md) genera en cambio la
separacion 80 % desarrollo / 20 % prueba externa bloqueada, y 5 folds internos
solo sobre el 80 %:

    python -m modeling.build_master_folds --protocol holdout-v3

    modeling/data/holdout_splits.csv
    modeling/data/holdout_splits_manifest.json

El resto de este docstring describe el protocolo fold-aware v2 (por defecto);
la separacion holdout-v3 se describe en ``main_holdout_v3``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import splits as sp

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ICBHI_SEGMENTS = REPO_ROOT / "modeling" / "data" / "copd_vs_control" / "ICBHI" / "segments.csv"
DEFAULT_FRAIWAN_SEGMENTS = REPO_ROOT / "modeling" / "data" / "copd_vs_control" / "FRAIWAN_Extended" / "segments.csv"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "modeling" / "data" / "patient_folds.csv"
DEFAULT_OUTPUT_MANIFEST = REPO_ROOT / "modeling" / "data" / "patient_folds_manifest.json"
DEFAULT_HOLDOUT_CSV = REPO_ROOT / "modeling" / "data" / "holdout_splits.csv"
DEFAULT_HOLDOUT_MANIFEST = REPO_ROOT / "modeling" / "data" / "holdout_splits_manifest.json"

PROTOCOL_FOLD_AWARE_V2 = "fold-aware-v2"
PROTOCOL_HOLDOUT_V3 = "holdout-v3"

N_SPLITS = 5
RANDOM_STATE = 20260914


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera la asignacion maestra de pacientes (fold-aware v2 o holdout-v3)."
    )
    parser.add_argument(
        "--protocol", choices=(PROTOCOL_FOLD_AWARE_V2, PROTOCOL_HOLDOUT_V3), default=PROTOCOL_FOLD_AWARE_V2,
        help="fold-aware-v2 (por defecto): patient_folds.csv; holdout-v3: holdout_splits.csv.",
    )
    parser.add_argument("--icbhi-segments", type=Path, default=DEFAULT_ICBHI_SEGMENTS)
    parser.add_argument("--fraiwan-segments", type=Path, default=DEFAULT_FRAIWAN_SEGMENTS)
    parser.add_argument(
        "--output-csv", type=Path, default=None,
        help=f"Por defecto {DEFAULT_OUTPUT_CSV} (fold-aware-v2) o {DEFAULT_HOLDOUT_CSV} (holdout-v3).",
    )
    parser.add_argument(
        "--output-manifest", type=Path, default=None,
        help=f"Por defecto {DEFAULT_OUTPUT_MANIFEST} (fold-aware-v2) o {DEFAULT_HOLDOUT_MANIFEST} (holdout-v3).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Solo holdout-v3: permite reemplazar un holdout_splits.csv existente cuyo contenido difiere. "
             "Sin esta bandera la separacion 80/20 se conserva: es una unica separacion.",
    )
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


def _holdout_counts(frame: pd.DataFrame) -> dict:
    """Conteos por rol externo, fuente, clase y fold interno (una sola escala:
    numero de pacientes) de UN dataset_scope, con claves JSON-nativas."""
    def by(columns: list[str]) -> dict:
        grouped = frame.groupby(columns, dropna=False).size()
        return {"|".join(str(v) for v in (key if isinstance(key, tuple) else (key,))): int(n)
                for key, n in grouped.items()}

    return {
        "n_patients": int(len(frame)),
        "by_outer_role": by(["outer_role"]),
        "by_source_dataset": by(["source_dataset"]),
        "by_target_label": by(["target_label"]),
        "by_outer_role_and_class": by(["outer_role", "target_label"]),
        "by_source_role_and_class": by(["source_dataset", "outer_role", "target_label"]),
        "by_inner_fold_group_and_class": by(["inner_fold_group", "target_label"]),
    }


def main_holdout_v3(args: argparse.Namespace) -> int:
    """Separacion holdout-v3 (PLAN-EXPERIMENTO FINAL.md, seccion 2).

    ICBHI y Fraiwan Extended se separan cada uno por su cuenta -estratificado
    por clase, semilla ``RANDOM_STATE``-: ~80 % desarrollo (que se reparte en
    ``N_SPLITS`` folds internos, tambien estratificado) y ~20 % prueba externa
    bloqueada (``inner_fold_group = -1``). COMBINED hereda exactamente la
    asignacion de cada paciente de su dataset de origen; no se vuelve a sortear.

    Es una unica separacion: si ``holdout_splits.csv`` ya existe con otro
    contenido, se niega a reemplazarlo salvo con ``--overwrite``; si coincide
    byte a byte con lo que se generaria, lo deja como esta (idempotente).
    """
    output_csv = args.output_csv or DEFAULT_HOLDOUT_CSV
    output_manifest = args.output_manifest or DEFAULT_HOLDOUT_MANIFEST

    sources = {"ICBHI": args.icbhi_segments, "FRAIWAN_Extended": args.fraiwan_segments}
    parts: dict[str, pd.DataFrame] = {}
    for scope, segments_path in sources.items():
        segments = pd.read_csv(segments_path, dtype={"patient_uid": str})
        table = sp.assign_holdout_split(
            sp.build_patient_table(segments), test_fraction=sp.HOLDOUT_TEST_FRACTION,
            n_splits=N_SPLITS, random_state=RANDOM_STATE,
        )
        sp.verify_holdout_split(table, N_SPLITS, context=scope)
        parts[scope] = table
    combined = sp.combine_holdout_splits(parts, n_splits=N_SPLITS)

    scopes = {"ICBHI": parts["ICBHI"], "FRAIWAN_Extended": parts["FRAIWAN_Extended"], "COMBINED": combined}
    frame = pd.concat(
        [sp.holdout_split_to_frame(table, scope) for scope, table in scopes.items()], ignore_index=True,
    )

    new_text = frame.to_csv(index=False, lineterminator="\n")
    if Path(output_csv).is_file():
        existing_text = Path(output_csv).read_text(encoding="utf-8")
        if existing_text == new_text and Path(output_manifest).is_file():
            print(f"{output_csv}: ya existe y coincide con la separacion que se generaria; no se modifica.")
            print("Veredicto: PASS")
            return 0
        if not args.overwrite:
            print(
                f"{output_csv} ya existe con otro contenido. La separacion 80/20 es unica: "
                "no se reemplaza sin --overwrite.",
                file=sys.stderr,
            )
            return 2

    manifest_extra = {
        "git": _git_state(),
        "seed": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "test_fraction": sp.HOLDOUT_TEST_FRACTION,
        "source_segments_sha256": {
            scope: sp.sha256_file(path) for scope, path in sources.items()
        },
        "method": {
            "outer": "StratifiedShuffleSplit por clase, independiente en ICBHI y FRAIWAN_Extended",
            "inner": "StratifiedKFold por clase, solo sobre desarrollo",
            "COMBINED": "hereda outer_role e inner_fold_group de su dataset de origen; sin sorteo propio",
        },
        "counts": {scope: _holdout_counts(frame.loc[frame["dataset_scope"] == scope]) for scope in scopes},
    }
    manifest = sp.write_holdout_split_csv(frame, output_csv, output_manifest, manifest_extra)

    print(f"holdout_splits.csv -> {output_csv}")
    print(f"holdout_splits_manifest.json -> {output_manifest}")
    for scope, counts in manifest["counts"].items():
        print(f"  {scope}: {counts['n_patients']} pacientes, por rol={counts['by_outer_role']}, "
              f"rol|clase={counts['by_outer_role_and_class']}")
    print("Veredicto: PASS")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.protocol == PROTOCOL_HOLDOUT_V3:
        return main_holdout_v3(args)

    output_csv = args.output_csv or DEFAULT_OUTPUT_CSV
    output_manifest = args.output_manifest or DEFAULT_OUTPUT_MANIFEST

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

    manifest = sp.write_patient_folds_csv(frame, output_csv, output_manifest, manifest_extra)

    print(f"patient_folds.csv -> {output_csv}")
    print(f"patient_folds_manifest.json -> {output_manifest}")
    for scope, counts in manifest["counts"].items():
        print(
            f"  {scope}: {counts['n_patients']} pacientes, "
            f"por fold_group={counts['by_fold_group']}, por clase={counts['by_target_label']}"
        )
    print("Veredicto: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
