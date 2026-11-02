"""Comprobaciones de ``--dry-run`` compartidas por SVM, CNN y CRNN."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import data as dmod
from .. import splits as sp


# ---------------------------------------------------------------------------
# --dry-run: valida formas, hashes, conteos y folds sin entrenar nada
# ---------------------------------------------------------------------------

def dry_run_check(data_root: Path, cfg: dict, specs: list[dmod.ConditionSpec]) -> dict:
    rows: list[dict] = []
    ok = True

    for dataset in sorted({s.dataset for s in specs}):
        try:
            segments = dmod.load_task_segments(data_root, dataset)
            manifest = dmod.load_task_manifest(data_root, dataset)
        except (FileNotFoundError, ValueError) as exc:
            rows.append({"dataset": dataset, "check": "carga_segments", "ok": False, "detail": str(exc)})
            ok = False
            continue
        rows.append({
            "dataset": dataset, "check": "carga_segments", "ok": True,
            "detail": f"{len(segments)} segmentos, {segments['patient_uid'].nunique()} pacientes",
        })

        manifest_hashes = manifest.get("output_hashes", {})
        csv_path = Path(data_root) / dataset / "segments.csv"
        csv_hash = dmod.sha256_file(csv_path)
        expected_csv_hash = manifest_hashes.get("segments.csv")
        csv_hash_ok = expected_csv_hash is not None and csv_hash == expected_csv_hash
        ok = ok and csv_hash_ok
        rows.append({
            "dataset": dataset, "check": "sha256_segments_csv", "ok": csv_hash_ok,
            "detail": (
                "coincide con manifest.json"
                if csv_hash_ok
                else f"{csv_hash[:12]}... vs manifest {str(expected_csv_hash)[:12]}..."
            ),
        })

        for branch in dmod.BRANCHES:
            try:
                array = dmod.load_branch_array(data_root, dataset, branch)
            except FileNotFoundError as exc:
                rows.append({"dataset": dataset, "check": f"npy_{branch}", "ok": False, "detail": str(exc)})
                ok = False
                continue
            shape_ok = array.shape[0] == len(segments) and array.shape[1] == int(cfg["acoustic"]["segment_length"])
            ok = ok and shape_ok
            rows.append({
                "dataset": dataset, "check": f"forma_{branch}", "ok": shape_ok,
                "detail": f"{array.shape} vs ({len(segments)}, {cfg['acoustic']['segment_length']})",
            })

            npy_path = Path(data_root) / dataset / f"segments_{branch}.npy"
            npy_hash = dmod.sha256_file(npy_path)
            expected_npy_hash = manifest_hashes.get(f"segments_{branch}.npy")
            npy_hash_ok = expected_npy_hash is not None and npy_hash == expected_npy_hash
            ok = ok and npy_hash_ok
            rows.append({
                "dataset": dataset, "check": f"sha256_{branch}", "ok": npy_hash_ok,
                "detail": (
                    "coincide con manifest.json"
                    if npy_hash_ok
                    else f"{npy_hash[:12]}... vs manifest {str(expected_npy_hash)[:12]}..."
                ),
            })

    fold_cache: dict[tuple[str, bool], sp.PatientFolds] = {}
    for spec in specs:
        try:
            segments = dmod.load_task_segments(data_root, spec.dataset)
            selected = dmod.select_condition_segments(segments, spec)
            key = (spec.dataset, spec.dn_reliable_only)
            if key not in fold_cache:
                fold_cache[key] = sp.build_patient_folds(
                    selected, n_splits=int(cfg["splits"]["n_splits"]),
                    random_state=int(cfg["splits"]["random_state"]),
                    stratify_by_dataset=bool(cfg["splits"].get("stratify_by_dataset", False)),
                )
            n_patients = len(fold_cache[key].patient_table)
            rows.append({
                "dataset": spec.dataset, "check": f"folds[{spec.condition}]", "ok": True,
                "detail": f"{n_patients} pacientes, {fold_cache[key].n_splits} folds, verificado sin fuga",
            })
        except Exception as exc:  # noqa: BLE001 - se reporta, nunca se oculta
            ok = False
            rows.append({"dataset": spec.dataset, "check": f"folds[{spec.condition}]", "ok": False, "detail": str(exc)})

    return {"ok": ok, "checks": pd.DataFrame(rows)}


def dry_run_check_folded(
    data_root: Path, cfg: dict, specs: list[dmod.ConditionSpec],
    patient_folds_csv: Path, patient_folds_manifest: Path,
) -> dict:
    """Como ``dry_run_check``, pero validando la asignacion maestra de folds
    y, por cada dataset, sus 5 carpetas ``fold_00``..``fold_04`` (cada una con
    su propio ``segments.csv``/``segments_<rama>.npy`` y su propio
    ``manifest.json``), en vez de una unica carpeta compartida."""
    rows: list[dict] = []
    ok = True
    patient_folds_csv_hash = dmod.sha256_file(Path(patient_folds_csv))

    for dataset in sorted({s.dataset for s in specs}):
        try:
            folds = sp.load_patient_folds(patient_folds_csv, patient_folds_manifest, dataset)
        except Exception as exc:  # noqa: BLE001 - se reporta, nunca se oculta
            ok = False
            rows.append({"dataset": dataset, "check": "patient_folds", "ok": False, "detail": str(exc)})
            continue
        rows.append({
            "dataset": dataset, "check": "patient_folds", "ok": True,
            "detail": f"{len(folds.patient_table)} pacientes, {folds.n_splits} folds, verificado sin fuga",
        })

        for fold_id in range(folds.n_splits):
            tag = f"fold_{fold_id:02d}"
            try:
                segments = dmod.load_task_segments(data_root, dataset, fold_id)
                manifest = dmod.load_task_manifest(data_root, dataset, fold_id)
            except (FileNotFoundError, ValueError) as exc:
                rows.append({"dataset": dataset, "check": f"carga_segments[{tag}]", "ok": False, "detail": str(exc)})
                ok = False
                continue

            # El preprocesamiento de este fold debe haberse hecho contra el
            # patient_folds.csv actual -si cambio desde entonces (nuevo sorteo,
            # nuevos pacientes), el fold preprocesado queda invalidado: no hay
            # garantia de que su train/validation/test siga siendo el mismo.
            fold_csv_hash = manifest.get("patient_folds_csv_sha256")
            csv_link_ok = fold_csv_hash is not None and fold_csv_hash == patient_folds_csv_hash
            ok = ok and csv_link_ok
            rows.append({
                "dataset": dataset, "check": f"patient_folds_csv_sha256[{tag}]", "ok": csv_link_ok,
                "detail": (
                    "coincide con patient_folds.csv actual" if csv_link_ok else
                    f"{tag}/manifest.json quedo desincronizado de patient_folds.csv; "
                    "regenere este fold con preprocessing/fold_denoising.py"
                ),
            })

            root = dmod.dataset_root(data_root, dataset, fold_id)
            manifest_hashes = manifest.get("output_hashes", {})
            csv_hash = dmod.sha256_file(root / "segments.csv")
            expected_csv_hash = manifest_hashes.get("segments.csv")
            csv_hash_ok = expected_csv_hash is not None and csv_hash == expected_csv_hash
            ok = ok and csv_hash_ok
            rows.append({
                "dataset": dataset, "check": f"sha256_segments_csv[{tag}]", "ok": csv_hash_ok,
                "detail": "coincide con manifest.json" if csv_hash_ok else "no coincide con manifest.json",
            })

            for branch in dmod.BRANCHES:
                try:
                    array = dmod.load_branch_array(data_root, dataset, branch, fold_id)
                except FileNotFoundError as exc:
                    rows.append({"dataset": dataset, "check": f"npy_{branch}[{tag}]", "ok": False, "detail": str(exc)})
                    ok = False
                    continue
                shape_ok = array.shape[0] == len(segments) and array.shape[1] == int(cfg["acoustic"]["segment_length"])
                ok = ok and shape_ok
                rows.append({
                    "dataset": dataset, "check": f"forma_{branch}[{tag}]", "ok": shape_ok,
                    "detail": f"{array.shape} vs ({len(segments)}, {cfg['acoustic']['segment_length']})",
                })

                npy_hash = dmod.sha256_file(root / f"segments_{branch}.npy")
                expected_npy_hash = manifest_hashes.get(f"segments_{branch}.npy")
                npy_hash_ok = expected_npy_hash is not None and npy_hash == expected_npy_hash
                ok = ok and npy_hash_ok
                rows.append({
                    "dataset": dataset, "check": f"sha256_{branch}[{tag}]", "ok": npy_hash_ok,
                    "detail": "coincide con manifest.json" if npy_hash_ok else "no coincide con manifest.json",
                })

    return {"ok": ok, "checks": pd.DataFrame(rows)}
