"""CLI unico de los experimentos de modelado (SVM-RBF y CNN).

    python -u -m modeling.run_experiment \
        --model svm_rbf --dataset all --experiment all --n-jobs 4 \
        --data-root <ruta> --runs-root <ruta>

    python -u -m modeling.run_experiment \
        --model cnn --dataset all --experiment all --device cuda:0 \
        --data-root <ruta> --runs-root <ruta> --cache-root <ruta>

``--model cnn`` carga ``configs/cnn.toml`` y despacha a
``modeling.cnn_experiment``; ``--model svm_rbf`` sigue exactamente el flujo de
este archivo. No hay rutas personales: ``--data-root``/``--runs-root``/
``--cache-root``, o las variables ``PULMONARY_DATA_ROOT`` /
``PULMONARY_RUNS_ROOT`` / ``PULMONARY_CACHE_ROOT``, resuelven donde viven los
datos y los resultados en cada maquina (ver ``modeling.data.resolve_path``).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import re
import shutil
import sys
from pathlib import Path

import pandas as pd

from . import artifacts as art
from . import data as dmod
from . import evaluation as ev
from . import splits as sp
from .models import svm_rbf as svm_model

MODEL_CHOICES = ("svm_rbf", "cnn", "crnn")
DATASET_CHOICES = ("ICBHI", "FRAIWAN_Extended", "COMBINED", "all")
EXPERIMENT_CHOICES = ("main", "denoising_ablation", "augmentation_ablation", "all")
CONFIG_PATHS = {
    "svm_rbf": dmod.DEFAULT_CONFIG_PATH,
    "cnn": dmod.CNN_CONFIG_PATH,
    "crnn": dmod.CRNN_CONFIG_PATH,
}
DEVICE_PATTERN = re.compile(r"^(auto|cpu|cuda|cuda:\d+)$")

# Pareja (lado no_dn, lado dn) de cada dataset en la ablacion, para la figura
# de comparacion emparejada. En Fraiwan el lado no_dn es la misma condicion
# "main_no_dn" que ya corre en el experimento principal (todas sus
# grabaciones son dn_reliable=True, asi que no hace falta una condicion
# "no_dn_reliable" separada).
ABLATION_PAIRS = {
    "ICBHI": ("no_dn_reliable", "dn_reliable"),
    "FRAIWAN_Extended": ("main_no_dn", "dn"),
    "COMBINED": ("no_dn_reliable", "dn_reliable"),
}


# ---------------------------------------------------------------------------
# Protocolo fold-aware (v2): el denoising se recalibra por fold (ver
# preprocessing/fold_denoising.py) en vez de una sola vez de forma global, asi
# que cada fold reconstruye su propia cache de features/Log-Mel y ya no hace
# falta mantener a calibration_patient siempre en train. Un TOML activa este
# protocolo declarando [folds] patient_folds_csv; sin esa seccion, el
# comportamiento de run_experiment.py/cnn_experiment.py es exactamente el de
# antes.
# ---------------------------------------------------------------------------

def is_folded_protocol(cfg: dict) -> bool:
    return "patient_folds_csv" in cfg.get("folds", {})


def _patient_folds_paths(cfg: dict) -> tuple[Path, Path]:
    folds_cfg = cfg.get("folds", {})
    if "patient_folds_csv" not in folds_cfg:
        raise RuntimeError("el protocolo fold-aware (v2) requiere [folds] patient_folds_csv en el TOML")
    csv_path = dmod.REPO_ROOT / folds_cfg["patient_folds_csv"]
    manifest_path = dmod.REPO_ROOT / folds_cfg.get(
        "patient_folds_manifest", "modeling/data/patient_folds_manifest.json"
    )
    return csv_path, manifest_path


def _validate_final_model_disabled(cfg: dict) -> None:
    """El protocolo fold-aware nunca entrena un modelo definitivo: no hay una
    unica cache de todos los pacientes sobre la que hacerlo (cada fold tiene
    su propio denoising recalibrado), y esta etapa queda para un benchmarking
    posterior. ``[final_model] enabled = false`` debe estar explicito en el
    TOML -no es un valor por defecto- para que quede documentado y forme
    parte de la huella de la ejecucion (ver FOLDED_RUN_FINGERPRINT_SECTIONS).
    """
    enabled = cfg.get("final_model", {}).get("enabled")
    if enabled is not False:
        raise RuntimeError(
            "el protocolo fold-aware (v2) exige '[final_model]\\nenabled = false' explicito "
            f"en el TOML (valor actual: {enabled!r}); este protocolo no entrena modelo definitivo"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _device_arg(value: str) -> str:
    if not DEVICE_PATTERN.match(value):
        raise argparse.ArgumentTypeError("use auto, cpu, cuda o cuda:N")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena y evalua modelos COPD vs Control (SVM-RBF, CNN o CRNN), dataset por dataset."
    )
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument(
        "--dataset", choices=DATASET_CHOICES, required=True,
        help="COMBINED requiere un --config con [datasets.COMBINED] (ver configs/*_combined.toml).",
    )
    parser.add_argument(
        "--experiment", choices=EXPERIMENT_CHOICES, required=True,
        help="augmentation_ablation solo existe para la CNN y la CRNN.",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Procesos para la busqueda de la SVM (por defecto 1). CNN y CRNN no lo usan.",
    )
    parser.add_argument(
        "--device", type=_device_arg, default="auto",
        help="Solo CNN/CRNN: auto, cpu, cuda o cuda:N (por defecto auto).",
    )
    parser.add_argument(
        "--num-workers", type=int, default=2,
        help="Solo CNN/CRNN: workers del DataLoader (por defecto 2).",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument(
        "--cache-root", type=Path, default=None,
        help="Cache de caracteristicas (SVM) o Log-Mel compartida (CNN/CRNN); por defecto "
             "PULMONARY_CACHE_ROOT o la ruta de [paths] del TOML del modelo.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Ruta alternativa al TOML (por defecto configs/svm_rbf.toml, configs/cnn.toml "
             "o configs/crnn.toml segun --model).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Valida formas, hashes, conteos y folds; no entrena.")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="SVM: fold 1 y una combinacion (C, gamma). CNN/CRNN: fold 1, una configuracion y dos epocas.",
    )
    parser.add_argument(
        "--force-features", action="store_true",
        help="Regenera la cache de caracteristicas (SVM) o de Log-Mel (solo CNN; la CRNN lo rechaza "
             "para no sobrescribir la cache compartida).",
    )
    parser.add_argument(
        "--resume", nargs="?", const="latest", default=None, metavar="RUN_ID",
        help="Continua una ejecucion existente (RUN_ID o, sin valor, la mas reciente).",
    )
    return parser.parse_args(argv)


def resolve_datasets(cfg: dict, dataset_arg: str) -> list[str]:
    return dmod.dataset_names(cfg) if dataset_arg == "all" else [dataset_arg]


def resolve_experiments(experiment_arg: str, cfg: dict) -> list[str]:
    """``all`` son todos los experimentos del TOML del modelo, en su orden:
    main y denoising_ablation para la SVM; ademas augmentation_ablation para
    la CNN."""
    return dmod.experiment_names(cfg) if experiment_arg == "all" else [experiment_arg]


def plan_conditions(cfg: dict, datasets: list[str], experiments: list[str]) -> list[dmod.ConditionSpec]:
    """Condiciones (dataset, condicion) a ejecutar, sin duplicar cuando una
    misma condicion aparece en mas de un experimento (el caso de Fraiwan
    main_no_dn, reutilizada en la ablacion)."""
    seen: dict[tuple[str, str], dmod.ConditionSpec] = {}
    for dataset in datasets:
        for experiment in experiments:
            for spec in dmod.experiment_condition_specs(cfg, experiment, dataset):
                seen[(spec.dataset, spec.condition)] = spec
    return list(seen.values())


def _smoke_test_config(cfg: dict) -> dict:
    """Copia de ``cfg`` con la rejilla recortada a un solo (C, gamma)."""
    cfg2 = copy.deepcopy(cfg)
    cfg2["svm"]["c_grid"] = cfg["svm"]["c_grid"][:1]
    cfg2["svm"]["gamma_grid"] = cfg["svm"]["gamma_grid"][:1]
    return cfg2


# ---------------------------------------------------------------------------
# Huella de la ejecucion: lo que --resume debe verificar antes de reutilizar
# nada. Sin esto, un --resume podria continuar folds calculados sobre datos
# o configuracion distintos sin ningun aviso.
# ---------------------------------------------------------------------------

RUN_FINGERPRINT_CONFIG_SECTIONS = (
    "acoustic", "logmel", "mfcc", "summary", "splits", "weights",
    "svm", "selection", "bootstrap", "datasets", "experiments", "evaluation",
)


def _config_run_fingerprint(cfg: dict, sections: tuple[str, ...] = RUN_FINGERPRINT_CONFIG_SECTIONS) -> str:
    relevant = {k: cfg[k] for k in sections if k in cfg}
    payload = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _input_hashes_for_specs(data_root: Path, specs: list[dmod.ConditionSpec]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for dataset in sorted({s.dataset for s in specs}):
        csv_path = Path(data_root) / dataset / "segments.csv"
        hashes[f"{dataset}/segments.csv"] = dmod.sha256_file(csv_path)
    for spec in specs:
        npy_path = Path(data_root) / spec.dataset / f"segments_{spec.branch}.npy"
        hashes[f"{spec.dataset}/segments_{spec.branch}.npy"] = dmod.sha256_file(npy_path)
    return hashes


def build_run_fingerprint(
    cfg: dict,
    data_root: Path,
    specs: list[dmod.ConditionSpec],
    dataset_arg: str,
    experiment_arg: str,
    sections: tuple[str, ...] = RUN_FINGERPRINT_CONFIG_SECTIONS,
    extra: dict | None = None,
) -> dict:
    """Todo lo que define si una ejecucion es 'la misma' para --resume.

    Los folds no se listan aparte: son una funcion determinista de la
    poblacion (que depende de estos mismos hashes de entrada) y de
    ``splits.random_state``/``splits.n_splits`` (dentro de la configuracion
    fingerprint-ada), asi que verificar datos + configuracion basta para
    garantizar que los folds serian identicos si se reconstruyeran.

    ``sections`` elige que secciones del TOML entran en la huella (la CNN usa
    las suyas) y ``extra`` anade datos JSON-nativos que no viven en el TOML
    (arquitectura, modo --smoke-test). Con los valores por defecto la huella
    de la SVM es identica a la de versiones anteriores.
    """
    fingerprint = {
        "dataset_arg": dataset_arg,
        "experiment_arg": experiment_arg,
        "config_fingerprint": _config_run_fingerprint(cfg, sections),
        "input_hashes": _input_hashes_for_specs(data_root, specs),
    }
    if extra is not None:
        fingerprint["extra"] = extra
    return fingerprint


def _run_fingerprint_path(run_root: Path) -> Path:
    return run_root / "run_fingerprint.json"


def write_run_fingerprint(run_root: Path, fingerprint: dict) -> None:
    _run_fingerprint_path(run_root).write_text(
        json.dumps(fingerprint, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify_run_fingerprint(run_root: Path, fingerprint: dict) -> None:
    """Se niega a reanudar si los datos, la configuracion o el alcance
    (--dataset/--experiment) de la ejecucion original cambiaron: reutilizar
    folds, features cacheadas o modelos ya calculados sobre una entrada
    distinta mezclaria resultados incoherentes sin ningun aviso.
    """
    path = _run_fingerprint_path(run_root)
    if not path.is_file():
        raise RuntimeError(
            f"no se puede reanudar {run_root}: falta run_fingerprint.json "
            "(ejecucion de una version del codigo anterior a esta comprobacion; "
            "empiece una ejecucion nueva)"
        )
    stored = json.loads(path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    if stored.get("dataset_arg") != fingerprint["dataset_arg"]:
        mismatches.append(f"--dataset cambio: {stored.get('dataset_arg')} -> {fingerprint['dataset_arg']}")
    if stored.get("experiment_arg") != fingerprint["experiment_arg"]:
        mismatches.append(f"--experiment cambio: {stored.get('experiment_arg')} -> {fingerprint['experiment_arg']}")
    if stored.get("config_fingerprint") != fingerprint["config_fingerprint"]:
        mismatches.append("la configuracion del TOML cambio en alguna seccion incluida en la huella")
    if stored.get("extra") != fingerprint.get("extra"):
        mismatches.append(
            f"cambio la arquitectura, el modelo o el modo --smoke-test: "
            f"{stored.get('extra')} -> {fingerprint.get('extra')}"
        )
    stored_hashes = stored.get("input_hashes", {})
    for key, value in fingerprint["input_hashes"].items():
        if stored_hashes.get(key) != value:
            mismatches.append(f"entrada modificada desde la ejecucion original: {key}")
    for key in stored_hashes:
        if key not in fingerprint["input_hashes"]:
            mismatches.append(f"la ejecucion original dependia de una entrada que ya no aplica: {key}")
    if mismatches:
        raise RuntimeError(
            f"--resume rechazado para {run_root}: no coincide con la ejecucion original.\n  - "
            + "\n  - ".join(mismatches)
        )


FOLDED_RUN_FINGERPRINT_SECTIONS = RUN_FINGERPRINT_CONFIG_SECTIONS + ("folds", "final_model")


def _folded_input_hashes_for_specs(
    data_root: Path,
    specs: list[dmod.ConditionSpec],
    folds_by_scope: dict[str, sp.PatientFolds],
    smoke_test: bool = False,
) -> dict[str, str]:
    """Como ``_input_hashes_for_specs``, pero un hash por ``(dataset, fold)``:
    cada fold tiene su propio ``segments.csv``/``segments_<rama>.npy``.

    ``smoke_test=True`` restringe el hash al fold 0 -el unico que
    ``run_condition_folded``/``run_cnn_condition_folded`` van a tocar-, para
    que un smoke-test no exija tener ya en disco los otros 4 folds.
    """
    hashes: dict[str, str] = {}
    for spec in specs:
        n_splits = folds_by_scope[spec.dataset].n_splits
        fold_ids = [0] if smoke_test else range(n_splits)
        for fold_id in fold_ids:
            root = dmod.dataset_root(data_root, spec.dataset, fold_id)
            hashes[f"{spec.dataset}/fold_{fold_id:02d}/segments.csv"] = dmod.sha256_file(root / "segments.csv")
            hashes[f"{spec.dataset}/fold_{fold_id:02d}/segments_{spec.branch}.npy"] = dmod.sha256_file(
                root / f"segments_{spec.branch}.npy"
            )
    return hashes


def build_folded_run_fingerprint(
    cfg: dict,
    data_root: Path,
    specs: list[dmod.ConditionSpec],
    dataset_arg: str,
    experiment_arg: str,
    folds_by_scope: dict[str, sp.PatientFolds],
    sections: tuple[str, ...] = FOLDED_RUN_FINGERPRINT_SECTIONS,
    extra: dict | None = None,
    smoke_test: bool = False,
) -> dict:
    fingerprint = {
        "dataset_arg": dataset_arg,
        "experiment_arg": experiment_arg,
        "config_fingerprint": _config_run_fingerprint(cfg, sections),
        "input_hashes": _folded_input_hashes_for_specs(data_root, specs, folds_by_scope, smoke_test=smoke_test),
    }
    if extra is not None:
        fingerprint["extra"] = extra
    return fingerprint


def open_run(runs_root: Path, model: str, resume: str | None, fingerprint: dict) -> tuple[Path, str]:
    """Crea una ejecucion nueva o localiza la que se quiere reanudar.

    Con --resume, ``art.init_run`` solo localiza el directorio y la huella se
    verifica ANTES de cualquier escritura: no se abre run.log, no cambia
    status.json y no se tocan environment.txt, resolved_config.toml ni el
    staging abandonado. Si la huella no coincide se lanza RuntimeError y la
    ejecucion existente queda byte a byte como estaba. Lo usan la SVM, la CNN
    y la CRNN.
    """
    run_root, run_id = art.init_run(runs_root, model, resume, bool(resume))
    if resume:
        verify_run_fingerprint(run_root, fingerprint)
    return run_root, run_id


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


# ---------------------------------------------------------------------------
# Un fold, con reintentos y marcas de fallo persistentes
# ---------------------------------------------------------------------------

def _fold_failure_marker(run_root: Path, dataset: str, condition: str, fold_id: int) -> Path:
    return art.condition_dir(run_root, dataset, condition) / f"fold_{fold_id + 1:02d}_FAILED.json"


def _write_fold_failure(run_root: Path, dataset: str, condition: str, fold_id: int, exc: Exception) -> None:
    marker = _fold_failure_marker(run_root, dataset, condition, fold_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({
            "fold": fold_id, "error": str(exc), "type": type(exc).__name__,
            "failed_at_utc": art._now_iso(),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _clear_fold_failure_marker(run_root: Path, dataset: str, condition: str, fold_id: int) -> None:
    marker = _fold_failure_marker(run_root, dataset, condition, fold_id)
    if marker.is_file():
        marker.unlink()


def _coerce_gamma_column(series: pd.Series) -> pd.Series:
    """Recupera el tipo real de la columna ``gamma`` tras un viaje por CSV.

    La columna mezcla el texto "scale"/"auto" con floats (0.001, 0.01, 0.1).
    Un CSV no distingue tipos por celda: si UNA fila no es numerica, pandas
    guarda la columna ENTERA como texto, y "0.001" vuelve como el string
    ``"0.001"`` en vez del float ``0.001``. Pasado tal cual a ``SVC(gamma=...)``
    eso revienta con ``InvalidParameterError``. Aqui se convierte a float todo
    lo que no sea literalmente "scale" o "auto".
    """
    def _coerce(value):
        if isinstance(value, str) and value not in ("scale", "auto"):
            return float(value)
        return value

    return series.map(_coerce)


def _load_fold_outputs(run_root: Path, dataset: str, condition: str, fold_id: int) -> dict:
    fdir = art.fold_dir(run_root, dataset, condition, fold_id)
    grid_search = pd.read_csv(fdir / "grid_search.csv")
    grid_search["gamma"] = _coerce_gamma_column(grid_search["gamma"])
    return {
        "fold": fold_id,
        "grid_search": grid_search,
        "test_metrics": json.loads((fdir / "test_metrics.json").read_text(encoding="utf-8")),
        "segment_predictions": pd.read_csv(
            fdir / "test_segment_predictions.csv",
            dtype={"audio_id": str, "patient_uid": str, "segment_id": str},
        ),
        "recording_predictions": pd.read_csv(
            fdir / "test_recording_predictions.csv", dtype={"audio_id": str, "patient_uid": str}
        ),
        "patient_predictions": pd.read_csv(
            fdir / "test_patient_predictions.csv", dtype={"patient_uid": str}
        ),
        "baseline_metrics": pd.read_csv(fdir / "baseline_metrics.csv"),
        "selected": json.loads((fdir / "selected_hyperparameters.json").read_text(encoding="utf-8")),
    }


def run_condition(
    run_root: Path,
    condition_data: dmod.ConditionData,
    folds: sp.PatientFolds,
    cfg: dict,
    negative_label_name: str,
    n_jobs: int,
    smoke_test: bool,
    logger: logging.Logger,
) -> dict:
    """Todos los folds de una condicion (o solo el primero en --smoke-test).

    Cada fold es independiente de los demas: si uno falla, se marca FAILED,
    se registra en un archivo que sobrevive a la limpieza de staging, y se
    continua con el siguiente fold de la misma condicion. La condicion solo
    se resume (OOF, figuras, modelo final) si TODOS sus folds requeridos
    terminaron completos.
    """
    spec = condition_data.spec
    segments, X = condition_data.segments, condition_data.X
    fold_ids = [0] if smoke_test else list(range(folds.n_splits))
    cfg_used = _smoke_test_config(cfg) if smoke_test else cfg

    fold_status: dict[int, str] = {}
    for fold_id in fold_ids:
        if art.is_fold_complete(run_root, spec.dataset, spec.condition, fold_id):
            fold_status[fold_id] = "COMPLETED"
            logger.info(f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: ya publicado, se reutiliza")
            continue

        staging = art.fold_staging_dir(run_root, spec.dataset, spec.condition, fold_id)
        train_p, val_p, test_p = folds.get_split(fold_id)
        try:
            result = svm_model.run_fold(
                fold_id, segments, X, train_p, val_p, test_p, cfg_used, negative_label_name, n_jobs=n_jobs,
            )
            art.write_fold_artifacts(staging, result)
            art.publish_fold(staging)
            _clear_fold_failure_marker(run_root, spec.dataset, spec.condition, fold_id)
            fold_status[fold_id] = "COMPLETED"
            logger.info(
                f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: OK "
                f"C={result.selected_C} gamma={result.selected_gamma} "
                f"balanced_accuracy={result.test_metrics['balanced_accuracy']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - se registra, nunca se omite en silencio
            fold_status[fold_id] = "FAILED"
            _write_fold_failure(run_root, spec.dataset, spec.condition, fold_id, exc)
            logger.exception(f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: FALLO")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    completed_ids = [fid for fid, status in fold_status.items() if status == "COMPLETED"]
    condition_ok = len(completed_ids) == len(fold_ids)
    fold_outputs = [_load_fold_outputs(run_root, spec.dataset, spec.condition, fid) for fid in completed_ids]

    summary = None
    if condition_ok and fold_outputs:
        summary = summarize_condition(run_root, condition_data, folds, fold_outputs, cfg, negative_label_name, logger)

    return {"spec": spec, "fold_status": fold_status, "ok": condition_ok, "summary": summary}


# ---------------------------------------------------------------------------
# Resumen de una condicion: OOF, figuras, modelo final
# ---------------------------------------------------------------------------

def summarize_condition(
    run_root: Path,
    condition_data: dmod.ConditionData,
    folds: sp.PatientFolds,
    fold_outputs: list[dict],
    cfg: dict,
    negative_label_name: str,
    logger: logging.Logger,
) -> dict:
    spec = condition_data.spec
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    target_names = (negative_label_name, "COPD")
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))
    neg_key = negative_label_name.strip().lower()

    oof_patients = pd.concat([fo["patient_predictions"] for fo in fold_outputs], ignore_index=True)
    metrics_by_fold = pd.DataFrame([fo["test_metrics"] for fo in fold_outputs]).sort_values("fold").reset_index(drop=True)
    grid_tables = [fo["grid_search"] for fo in fold_outputs]
    baseline_metrics = pd.concat([fo["baseline_metrics"] for fo in fold_outputs], ignore_index=True)

    oof_patients = ev.attach_source_dataset(oof_patients, condition_data.segments)

    oof_y_true = oof_patients["target_label"].to_numpy()
    oof_y_score = oof_patients["score"].to_numpy()
    oof_metrics = ev.compute_patient_metrics(oof_y_true, oof_y_score, negative_label_name, threshold)

    bootstrap_cfg = cfg["bootstrap"]
    ci_specs = {
        "balanced_accuracy": ev.metric_fn_balanced_accuracy(threshold),
        "auroc": ev.metric_fn_auroc(),
        "auprc_copd": ev.metric_fn_auprc(),
        "recall_copd": ev.metric_fn_recall_copd(threshold),
        f"recall_{neg_key}": ev.metric_fn_recall_negative(threshold),
    }
    bootstrap_results = {
        name: ev.bootstrap_confidence_interval(
            oof_y_true, oof_y_score, fn,
            n_resamples=int(bootstrap_cfg["n_resamples"]),
            confidence=float(bootstrap_cfg["confidence"]),
            random_state=int(bootstrap_cfg["random_state"]),
        )
        for name, fn in ci_specs.items()
    }

    by_source = None
    if bool(cfg.get("evaluation", {}).get("report_by_source", False)):
        by_source = ev.by_source_report(
            oof_patients, "source_dataset", negative_label_name, threshold, ci_specs, bootstrap_cfg,
        )

    numeric_cols = ["accuracy", "balanced_accuracy", "recall_copd", f"recall_{neg_key}", "macro_f1", "auroc", "auprc_copd"]
    fold_mean = metrics_by_fold[numeric_cols].mean().to_dict()
    fold_std = metrics_by_fold[numeric_cols].std(ddof=1).to_dict()

    # Linea base de cordura (DummyClassifier), promediada entre folds por
    # estrategia: si la SVM no le saca ventaja clara en balanced_accuracy,
    # el resultado no es defendible aunque su cifra aislada luzca bien.
    baseline_mean = (
        baseline_metrics.groupby("strategy")[["balanced_accuracy", "recall_copd", f"recall_{neg_key}"]]
        .mean()
        .to_dict(orient="index")
    )

    summary_row = {
        "dataset": spec.dataset, "condition": spec.condition, "branch": spec.branch,
        "n_folds": len(fold_outputs),
        **{f"oof_{k}": v for k, v in oof_metrics.items()},
        **{f"fold_mean_{k}": v for k, v in fold_mean.items()},
        **{f"fold_std_{k}": v for k, v in fold_std.items()},
        **{
            f"ci95_{name}_{bound}": bootstrap_results[name][bound]
            for name in bootstrap_results for bound in ("point", "lower", "upper")
        },
        **{
            f"baseline_{strategy}_{metric}": value
            for strategy, metrics in baseline_mean.items()
            for metric, value in metrics.items()
        },
    }

    classification_report = ev.classification_report_df(oof_y_true, oof_y_score, target_names, threshold)
    confusion_matrix = ev.confusion_matrix_df(oof_y_true, oof_y_score, target_names, threshold)

    cdir.mkdir(parents=True, exist_ok=True)
    oof_patients.to_csv(cdir / "oof_patient_predictions.csv", index=False, lineterminator="\n")
    metrics_by_fold.to_csv(cdir / "metrics_by_fold.csv", index=False, lineterminator="\n")
    baseline_metrics.to_csv(cdir / "baseline_metrics.csv", index=False, lineterminator="\n")
    pd.concat(grid_tables, ignore_index=True).to_csv(cdir / "hyperparameter_search.csv", index=False, lineterminator="\n")
    classification_report.to_csv(cdir / "classification_report.csv", index=False, lineterminator="\n")
    confusion_matrix.to_csv(cdir / "confusion_matrix.csv")
    (cdir / "metrics_summary.json").write_text(
        json.dumps(art._json_safe(summary_row), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if by_source is not None:
        for name, table in by_source.items():
            table.to_csv(cdir / f"{name}.csv", index=False, lineterminator="\n")

    prefix = run_root / "figures" / f"{spec.dataset}__{spec.condition}"
    art.plot_confusion_matrix(oof_y_true, oof_y_score, target_names,
                               prefix.with_name(prefix.name + "__confusion_matrix"), cfg, threshold)
    art.plot_roc_curve(oof_y_true, oof_y_score, prefix.with_name(prefix.name + "__roc"), cfg)
    art.plot_pr_curve(oof_y_true, oof_y_score, prefix.with_name(prefix.name + "__pr"), cfg)
    art.plot_metrics_by_fold(metrics_by_fold, negative_label_name,
                              prefix.with_name(prefix.name + "__metrics_by_fold"), cfg)

    final_C, final_gamma, aggregated_grid = svm_model.select_final_hyperparameters(grid_tables, cfg)
    art.plot_hyperparameter_heatmap(aggregated_grid, "balanced_accuracy",
                                     prefix.with_name(prefix.name + "__hyperparam_heatmap"), cfg)

    learning_curve_df = pd.DataFrame()
    fold0 = next((fo for fo in fold_outputs if fo["fold"] == 0), None)
    if fold0 is not None:
        train_p, val_p, _test_p = folds.get_split(0)
        pool_seg, pool_X = dmod.select_rows_by_patients(condition_data.segments, condition_data.X, train_p + val_p)
        try:
            learning_curve_df = svm_model.compute_learning_curve(
                pool_seg, pool_X, float(fold0["selected"]["C"]), fold0["selected"]["gamma"], cfg,
            )
        except ValueError as exc:
            logger.warning(f"{spec.dataset}/{spec.condition}: curva de aprendizaje omitida ({exc})")
    art.plot_learning_curve(learning_curve_df, prefix.with_name(prefix.name + "__learning_curve"), cfg)

    scaler, svm, _weights = svm_model.fit_final_model(
        condition_data.segments, condition_data.X, final_C, final_gamma, cfg
    )
    art.save_final_model(
        run_root, spec.dataset, spec.condition, scaler, svm, condition_data.feature_names, cfg,
        final_C, final_gamma, negative_label_name,
        metadata_extra={
            "n_patients": int(condition_data.segments["patient_uid"].nunique()),
            "n_recordings": int(condition_data.segments["audio_id"].nunique()),
            "n_segments": int(len(condition_data.segments)),
            "oof_metrics": oof_metrics,
            "baseline_mean": baseline_mean,
            "aggregated_validation_grid": aggregated_grid.to_dict(orient="records"),
        },
    )

    logger.info(
        f"{spec.dataset}/{spec.condition}: OOF balanced_accuracy={oof_metrics['balanced_accuracy']:.3f} "
        f"auroc={oof_metrics['auroc']:.3f}  modelo final C={final_C} gamma={final_gamma}"
    )

    return {
        "summary_row": summary_row,
        "oof_patients": oof_patients,
        "metrics_by_fold": metrics_by_fold,
        "baseline_metrics": baseline_metrics,
        "hyperparameter_search": pd.concat(grid_tables, ignore_index=True),
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "by_source": by_source,
    }


# ---------------------------------------------------------------------------
# Protocolo fold-aware (v2): un fold y el resumen de una condicion,
# reconstruyendo la cache de cada fold por separado en vez de repartir una
# unica cache por dataset entre los 5 folds. ``run_fold``/``write_fold_artifacts``/
# ``publish_fold`` se reutilizan sin ningun cambio.
# ---------------------------------------------------------------------------

def run_condition_folded(
    run_root: Path,
    data_root: Path,
    cache_root: Path,
    spec: dmod.ConditionSpec,
    folds: sp.PatientFolds,
    cfg: dict,
    negative_label_name: str,
    n_jobs: int,
    smoke_test: bool,
    force_features: bool,
    logger: logging.Logger,
) -> dict:
    fold_ids = [0] if smoke_test else list(range(folds.n_splits))
    cfg_used = _smoke_test_config(cfg) if smoke_test else cfg

    fold_status: dict[int, str] = {}
    for fold_id in fold_ids:
        if art.is_fold_complete(run_root, spec.dataset, spec.condition, fold_id):
            fold_status[fold_id] = "COMPLETED"
            logger.info(f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: ya publicado, se reutiliza")
            continue

        staging = art.fold_staging_dir(run_root, spec.dataset, spec.condition, fold_id)
        train_p, val_p, test_p = folds.get_split(fold_id)
        try:
            condition_data = dmod.build_condition_data(
                data_root, cache_root, spec, cfg_used, force_features=force_features, fold_id=fold_id,
            )
            result = svm_model.run_fold(
                fold_id, condition_data.segments, condition_data.X, train_p, val_p, test_p,
                cfg_used, negative_label_name, n_jobs=n_jobs,
            )
            art.write_fold_artifacts(staging, result)
            art.publish_fold(staging)
            _clear_fold_failure_marker(run_root, spec.dataset, spec.condition, fold_id)
            fold_status[fold_id] = "COMPLETED"
            logger.info(
                f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: OK "
                f"C={result.selected_C} gamma={result.selected_gamma} "
                f"balanced_accuracy={result.test_metrics['balanced_accuracy']:.3f}"
            )
        except Exception as exc:  # noqa: BLE001 - se registra, nunca se omite en silencio
            fold_status[fold_id] = "FAILED"
            _write_fold_failure(run_root, spec.dataset, spec.condition, fold_id, exc)
            logger.exception(f"{spec.dataset}/{spec.condition} fold {fold_id + 1}: FALLO")
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

    completed_ids = [fid for fid, status in fold_status.items() if status == "COMPLETED"]
    condition_ok = len(completed_ids) == len(fold_ids)
    fold_outputs = [_load_fold_outputs(run_root, spec.dataset, spec.condition, fid) for fid in completed_ids]

    summary = None
    if condition_ok and fold_outputs:
        # source_dataset: la misma poblacion de pacientes vive en todos los
        # folds de un dataset (solo cambia como se denoisea su audio), asi
        # que el segments.csv de cualquier fold completado basta para mapear
        # patient_uid -> source_dataset.
        source_segments = dmod.load_task_segments(data_root, spec.dataset, fold_id=completed_ids[0])
        summary = summarize_condition_folded(
            run_root, spec, source_segments, fold_outputs, cfg, negative_label_name, logger,
        )

    return {"spec": spec, "fold_status": fold_status, "ok": condition_ok, "summary": summary}


def summarize_condition_folded(
    run_root: Path,
    spec: dmod.ConditionSpec,
    source_segments: pd.DataFrame,
    fold_outputs: list[dict],
    cfg: dict,
    negative_label_name: str,
    logger: logging.Logger,
) -> dict:
    """Como ``summarize_condition``, salvo que no ajusta ningun modelo final
    ni curva de aprendizaje: el protocolo fold-aware no tiene una unica cache
    de todos los pacientes sobre la que hacerlo (cada fold recalibro su
    propio denoising), y ese entrenamiento queda fuera de este protocolo de
    todos modos (ver ``_validate_final_model_disabled``). Todo lo demas
    -busqueda de hiperparametros, su heatmap, metricas por fold, OOF,
    bootstrap, por-fuente- se conserva igual.
    """
    cdir = art.condition_dir(run_root, spec.dataset, spec.condition)
    target_names = (negative_label_name, "COPD")
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))
    neg_key = negative_label_name.strip().lower()

    oof_patients = pd.concat([fo["patient_predictions"] for fo in fold_outputs], ignore_index=True)
    metrics_by_fold = pd.DataFrame([fo["test_metrics"] for fo in fold_outputs]).sort_values("fold").reset_index(drop=True)
    grid_tables = [fo["grid_search"] for fo in fold_outputs]
    baseline_metrics = pd.concat([fo["baseline_metrics"] for fo in fold_outputs], ignore_index=True)

    oof_patients = ev.attach_source_dataset(oof_patients, source_segments)

    oof_y_true = oof_patients["target_label"].to_numpy()
    oof_y_score = oof_patients["score"].to_numpy()
    oof_metrics = ev.compute_patient_metrics(oof_y_true, oof_y_score, negative_label_name, threshold)

    bootstrap_cfg = cfg["bootstrap"]
    ci_specs = {
        "balanced_accuracy": ev.metric_fn_balanced_accuracy(threshold),
        "auroc": ev.metric_fn_auroc(),
        "auprc_copd": ev.metric_fn_auprc(),
        "recall_copd": ev.metric_fn_recall_copd(threshold),
        f"recall_{neg_key}": ev.metric_fn_recall_negative(threshold),
    }
    bootstrap_results = {
        name: ev.bootstrap_confidence_interval(
            oof_y_true, oof_y_score, fn,
            n_resamples=int(bootstrap_cfg["n_resamples"]),
            confidence=float(bootstrap_cfg["confidence"]),
            random_state=int(bootstrap_cfg["random_state"]),
        )
        for name, fn in ci_specs.items()
    }

    by_source = None
    if bool(cfg.get("evaluation", {}).get("report_by_source", False)):
        by_source = ev.by_source_report(
            oof_patients, "source_dataset", negative_label_name, threshold, ci_specs, bootstrap_cfg,
        )

    numeric_cols = ["accuracy", "balanced_accuracy", "recall_copd", f"recall_{neg_key}", "macro_f1", "auroc", "auprc_copd"]
    fold_mean = metrics_by_fold[numeric_cols].mean().to_dict()
    fold_std = metrics_by_fold[numeric_cols].std(ddof=1).to_dict()
    baseline_mean = (
        baseline_metrics.groupby("strategy")[["balanced_accuracy", "recall_copd", f"recall_{neg_key}"]]
        .mean()
        .to_dict(orient="index")
    )

    summary_row = {
        "dataset": spec.dataset, "condition": spec.condition, "branch": spec.branch,
        "n_folds": len(fold_outputs),
        **{f"oof_{k}": v for k, v in oof_metrics.items()},
        **{f"fold_mean_{k}": v for k, v in fold_mean.items()},
        **{f"fold_std_{k}": v for k, v in fold_std.items()},
        **{
            f"ci95_{name}_{bound}": bootstrap_results[name][bound]
            for name in bootstrap_results for bound in ("point", "lower", "upper")
        },
        **{
            f"baseline_{strategy}_{metric}": value
            for strategy, metrics in baseline_mean.items()
            for metric, value in metrics.items()
        },
    }

    classification_report = ev.classification_report_df(oof_y_true, oof_y_score, target_names, threshold)
    confusion_matrix = ev.confusion_matrix_df(oof_y_true, oof_y_score, target_names, threshold)

    cdir.mkdir(parents=True, exist_ok=True)
    oof_patients.to_csv(cdir / "oof_patient_predictions.csv", index=False, lineterminator="\n")
    metrics_by_fold.to_csv(cdir / "metrics_by_fold.csv", index=False, lineterminator="\n")
    baseline_metrics.to_csv(cdir / "baseline_metrics.csv", index=False, lineterminator="\n")
    pd.concat(grid_tables, ignore_index=True).to_csv(cdir / "hyperparameter_search.csv", index=False, lineterminator="\n")
    classification_report.to_csv(cdir / "classification_report.csv", index=False, lineterminator="\n")
    confusion_matrix.to_csv(cdir / "confusion_matrix.csv")
    (cdir / "metrics_summary.json").write_text(
        json.dumps(art._json_safe(summary_row), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if by_source is not None:
        for name, table in by_source.items():
            table.to_csv(cdir / f"{name}.csv", index=False, lineterminator="\n")

    prefix = run_root / "figures" / f"{spec.dataset}__{spec.condition}"
    art.plot_confusion_matrix(oof_y_true, oof_y_score, target_names,
                               prefix.with_name(prefix.name + "__confusion_matrix"), cfg, threshold)
    art.plot_roc_curve(oof_y_true, oof_y_score, prefix.with_name(prefix.name + "__roc"), cfg)
    art.plot_pr_curve(oof_y_true, oof_y_score, prefix.with_name(prefix.name + "__pr"), cfg)
    art.plot_metrics_by_fold(metrics_by_fold, negative_label_name,
                              prefix.with_name(prefix.name + "__metrics_by_fold"), cfg)

    _final_C, _final_gamma, aggregated_grid = svm_model.select_final_hyperparameters(grid_tables, cfg)
    art.plot_hyperparameter_heatmap(aggregated_grid, "balanced_accuracy",
                                     prefix.with_name(prefix.name + "__hyperparam_heatmap"), cfg)

    logger.info(
        f"{spec.dataset}/{spec.condition}: OOF balanced_accuracy={oof_metrics['balanced_accuracy']:.3f} "
        f"auroc={oof_metrics['auroc']:.3f}"
    )

    return {
        "summary_row": summary_row,
        "oof_patients": oof_patients,
        "metrics_by_fold": metrics_by_fold,
        "baseline_metrics": baseline_metrics,
        "hyperparameter_search": pd.concat(grid_tables, ignore_index=True),
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "by_source": by_source,
    }


def _write_folded_run_tables(
    run_root: Path, condition_results: list[dict], folds_by_scope: dict[str, sp.PatientFolds],
) -> None:
    ok_results = [r for r in condition_results if r["ok"] and r["summary"] is not None]
    if ok_results:
        def _tag(df: pd.DataFrame, r: dict) -> pd.DataFrame:
            return df.assign(dataset=r["spec"].dataset, condition=r["spec"].condition)

        pd.concat([_tag(r["summary"]["metrics_by_fold"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "metrics_by_fold.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["hyperparameter_search"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "hyperparameter_search.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame([r["summary"]["summary_row"] for r in ok_results]).to_csv(
            run_root / "metrics_summary.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["oof_patients"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "oof_patient_predictions.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["baseline_metrics"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "baseline_metrics.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["classification_report"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "classification_report.csv", index=False, lineterminator="\n"
        )
        confusion_frames = []
        for r in ok_results:
            cm = r["summary"]["confusion_matrix"].reset_index().rename(columns={"index": "real"})
            cm["dataset"], cm["condition"] = r["spec"].dataset, r["spec"].condition
            confusion_frames.append(cm)
        pd.concat(confusion_frames, ignore_index=True).to_csv(
            run_root / "confusion_matrix.csv", index=False, lineterminator="\n"
        )

        by_source_keys = (
            "metrics_by_source", "classification_report_by_source",
            "confusion_matrix_by_source", "bootstrap_by_source",
        )
        for key in by_source_keys:
            frames = [
                _tag(r["summary"]["by_source"][key], r) for r in ok_results
                if r["summary"].get("by_source")
            ]
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    run_root / f"{key}.csv", index=False, lineterminator="\n"
                )

    if folds_by_scope:
        fold_tables = []
        for dataset, folds in folds_by_scope.items():
            table = folds.fold_table.copy()
            table["dataset"] = dataset
            fold_tables.append(table)
        pd.concat(fold_tables, ignore_index=True).to_csv(run_root / "folds.csv", index=False, lineterminator="\n")


def run_folded_svm(args: argparse.Namespace, cfg: dict) -> int:
    """Punto de entrada del protocolo fold-aware (v2) para la SVM. ``main()``
    despacha aqui cuando el TOML declara ``[folds] patient_folds_csv``."""
    _validate_final_model_disabled(cfg)
    patient_folds_csv, patient_folds_manifest = _patient_folds_paths(cfg)

    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = resolve_datasets(cfg, args.dataset)
    experiments = resolve_experiments(args.experiment, cfg)
    specs = plan_conditions(cfg, datasets, experiments)

    if args.dry_run:
        report = dry_run_check_folded(data_root, cfg, specs, patient_folds_csv, patient_folds_manifest)
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(report["checks"].to_string(index=False))
        print(f"\nveredicto: {'OK' if report['ok'] else 'REVISAR'}")
        return 0 if report["ok"] else 1

    try:
        folds_by_scope = {
            dataset: sp.load_patient_folds(patient_folds_csv, patient_folds_manifest, dataset)
            for dataset in sorted({s.dataset for s in specs})
        }
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    # --resume: la huella se verifica ANTES de abrir run.log o escribir
    # cualquier archivo de la ejecucion existente (ver open_run).
    fingerprint = build_folded_run_fingerprint(
        cfg, data_root, specs, args.dataset, args.experiment, folds_by_scope, smoke_test=bool(args.smoke_test),
    )
    try:
        run_root, run_id = open_run(runs_root, args.model, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} protocolo=fold-aware dataset={args.dataset} experiment={args.experiment} "
        f"n_jobs={args.n_jobs} smoke_test={args.smoke_test} resume={bool(args.resume)}"
    )
    if args.resume:
        logger.info("run_fingerprint verificado: datos, configuracion y alcance coinciden con la ejecucion original")
        removed_staging = art.finalize_resume(run_root)
        if removed_staging:
            logger.info(f"staging abandonado eliminado: {[str(p) for p in removed_staging]}")

    art.write_environment(run_root)
    art.write_resolved_config(run_root, cfg, vars(args))
    if not args.resume:
        write_run_fingerprint(run_root, fingerprint)

    condition_results: list[dict] = []
    for spec in specs:
        logger.info(f"preparando {spec.dataset}/{spec.condition} (rama {spec.branch}, protocolo fold-aware)")
        try:
            negative_label_name = cfg["datasets"][spec.dataset]["negative_label_name"]
            result = run_condition_folded(
                run_root, data_root, cache_root, spec, folds_by_scope[spec.dataset], cfg,
                negative_label_name, args.n_jobs, args.smoke_test, args.force_features, logger,
            )
        except Exception:
            logger.exception(f"{spec.dataset}/{spec.condition}: fallo no controlado durante la preparacion")
            result = {"spec": spec, "fold_status": {}, "ok": False, "summary": None}
        condition_results.append(result)

    # Comparacion emparejada no_dn vs dn (denoising_ablation), cuando ambos
    # lados corrieron y terminaron OK en esta ejecucion.
    for dataset in datasets:
        left = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == "no_dn" and r["ok"]), None)
        right = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == "dn" and r["ok"]), None)
        if left and right:
            prefix = run_root / "figures" / f"{dataset}__no_dn_vs_dn__denoising_comparison"
            art.plot_denoising_comparison(left["summary"]["oof_patients"], right["summary"]["oof_patients"], prefix, cfg)
            logger.info(f"{dataset}: figura de comparacion no_dn/dn generada")

    _write_folded_run_tables(run_root, condition_results, folds_by_scope)

    for r in condition_results:
        if r["ok"]:
            final_dir = art.condition_dir(run_root, r["spec"].dataset, r["spec"].condition) / "final"
            if final_dir.exists():
                raise RuntimeError(
                    f"{r['spec'].dataset}/{r['spec'].condition}: existe {final_dir}, pero el protocolo "
                    "fold-aware (v2) no debe generar ningun modelo final"
                )

    all_ok = bool(condition_results) and all(r["ok"] for r in condition_results)
    any_ok = any(r["ok"] for r in condition_results)
    final_status = art.STATUS_COMPLETED if all_ok else (art.STATUS_PARTIAL if any_ok else art.STATUS_FAILED)
    art.write_status(run_root, final_status, {
        "conditions": [
            {"dataset": r["spec"].dataset, "condition": r["spec"].condition, "ok": r["ok"], "fold_status": r["fold_status"]}
            for r in condition_results
        ],
    })
    logger.info(f"ejecucion {run_id} finalizada con estado {final_status}")
    print(f"run_id={run_id} estado={final_status} -> {run_root}")
    return 0 if final_status == art.STATUS_COMPLETED else 1


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = dmod.load_config(args.config or CONFIG_PATHS[args.model])

    if args.experiment != "all" and args.experiment not in cfg["experiments"]:
        print(f"--experiment {args.experiment} no esta definido para --model {args.model}", file=sys.stderr)
        return 2

    if args.dataset != "all" and args.dataset not in dmod.dataset_names(cfg):
        print(
            f"--dataset {args.dataset} no esta definido en {args.config or CONFIG_PATHS[args.model]} "
            f"(datasets disponibles: {dmod.dataset_names(cfg)})",
            file=sys.stderr,
        )
        return 2

    if args.model in ("cnn", "crnn"):
        # El TOML debe describir la misma red que --model: evita, por ejemplo,
        # entrenar una CRNN y guardarla bajo runs/cnn con --config cnn.toml.
        architecture = dmod.model_architecture(cfg)
        if architecture != args.model:
            print(f"--model {args.model} con un TOML de arquitectura {architecture!r}", file=sys.stderr)
            return 2
        # Imports diferidos: el entorno de la SVM no necesita PyTorch instalado.
        if args.model == "crnn":
            from .crnn_experiment import run_crnn

            return run_crnn(args, cfg)
        from .cnn_experiment import run_cnn

        return run_cnn(args, cfg)

    if is_folded_protocol(cfg):
        return run_folded_svm(args, cfg)

    data_root = dmod.resolve_data_root(args.data_root, cfg)
    cache_root = dmod.resolve_cache_root(args.cache_root, cfg)
    runs_root = dmod.resolve_runs_root(args.runs_root, cfg)

    datasets = resolve_datasets(cfg, args.dataset)
    experiments = resolve_experiments(args.experiment, cfg)
    specs = plan_conditions(cfg, datasets, experiments)

    if args.dry_run:
        report = dry_run_check(data_root, cfg, specs)
        with pd.option_context("display.max_colwidth", 120, "display.width", 160):
            print(report["checks"].to_string(index=False))
        print(f"\nveredicto: {'OK' if report['ok'] else 'REVISAR'}")
        return 0 if report["ok"] else 1

    # --resume: la huella se verifica ANTES de abrir run.log o escribir
    # cualquier archivo de la ejecucion existente (status.json,
    # environment.txt, resolved_config.toml, staging). Si no coincide, se sale
    # con error y la ejecucion queda intacta (ver open_run).
    fingerprint = build_run_fingerprint(cfg, data_root, specs, args.dataset, args.experiment)
    try:
        run_root, run_id = open_run(runs_root, args.model, args.resume, fingerprint)
    except (RuntimeError, FileNotFoundError, FileExistsError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    logger = art.setup_run_logger(run_root)
    logger.info(
        f"run_id={run_id} dataset={args.dataset} experiment={args.experiment} "
        f"n_jobs={args.n_jobs} smoke_test={args.smoke_test} resume={bool(args.resume)}"
    )
    if args.resume:
        logger.info("run_fingerprint verificado: datos, configuracion y alcance coinciden con la ejecucion original")

        removed_staging = art.finalize_resume(run_root)
        if removed_staging:
            logger.info(f"staging abandonado eliminado: {[str(p) for p in removed_staging]}")

    art.write_environment(run_root)
    art.write_resolved_config(run_root, cfg, vars(args))
    if not args.resume:
        write_run_fingerprint(run_root, fingerprint)

    folds_by_population: dict[tuple[str, bool], sp.PatientFolds] = {}
    condition_results: list[dict] = []

    for spec in specs:
        logger.info(f"preparando {spec.dataset}/{spec.condition} (rama {spec.branch})")
        try:
            condition_data = dmod.build_condition_data(
                data_root, cache_root, spec, cfg, force_features=args.force_features,
            )

            pop_key = (spec.dataset, spec.dn_reliable_only)
            new_folds = sp.build_patient_folds(
                condition_data.segments,
                n_splits=int(cfg["splits"]["n_splits"]),
                random_state=int(cfg["splits"]["random_state"]),
                stratify_by_dataset=bool(cfg["splits"].get("stratify_by_dataset", False)),
            )
            if pop_key in folds_by_population:
                if not sp.folds_are_identical(folds_by_population[pop_key], new_folds):
                    raise RuntimeError(
                        f"{spec.dataset}: los folds de '{spec.condition}' no coinciden con los "
                        "de otra condicion que deberia compartir poblacion (no_dn vs dn)"
                    )
            else:
                folds_by_population[pop_key] = new_folds

            negative_label_name = cfg["datasets"][spec.dataset]["negative_label_name"]
            result = run_condition(
                run_root, condition_data, folds_by_population[pop_key], cfg,
                negative_label_name, args.n_jobs, args.smoke_test, logger,
            )
        except Exception:
            logger.exception(f"{spec.dataset}/{spec.condition}: fallo no controlado durante la preparacion")
            result = {"spec": spec, "fold_status": {}, "ok": False, "summary": None}
        condition_results.append(result)

    # Comparacion emparejada no_dn vs dn, cuando ambos lados de la pareja
    # del dataset corrieron y terminaron OK en esta ejecucion.
    for dataset in datasets:
        pair = ABLATION_PAIRS.get(dataset)
        if pair is None:
            continue
        left = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == pair[0] and r["ok"]), None)
        right = next((r for r in condition_results if r["spec"].dataset == dataset and r["spec"].condition == pair[1] and r["ok"]), None)
        if left and right:
            prefix = run_root / "figures" / f"{dataset}__{pair[0]}_vs_{pair[1]}__denoising_comparison"
            art.plot_denoising_comparison(left["summary"]["oof_patients"], right["summary"]["oof_patients"], prefix, cfg)
            logger.info(f"{dataset}: figura de comparacion no_dn/dn generada ({pair[0]} vs {pair[1]})")

    ok_results = [r for r in condition_results if r["ok"] and r["summary"] is not None]
    if ok_results:
        def _tag(df: pd.DataFrame, r: dict) -> pd.DataFrame:
            return df.assign(dataset=r["spec"].dataset, condition=r["spec"].condition)

        pd.concat([_tag(r["summary"]["metrics_by_fold"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "metrics_by_fold.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["hyperparameter_search"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "hyperparameter_search.csv", index=False, lineterminator="\n"
        )
        pd.DataFrame([r["summary"]["summary_row"] for r in ok_results]).to_csv(
            run_root / "metrics_summary.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["oof_patients"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "oof_patient_predictions.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["baseline_metrics"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "baseline_metrics.csv", index=False, lineterminator="\n"
        )
        pd.concat([_tag(r["summary"]["classification_report"], r) for r in ok_results], ignore_index=True).to_csv(
            run_root / "classification_report.csv", index=False, lineterminator="\n"
        )
        confusion_frames = []
        for r in ok_results:
            cm = r["summary"]["confusion_matrix"].reset_index().rename(columns={"index": "real"})
            cm["dataset"], cm["condition"] = r["spec"].dataset, r["spec"].condition
            confusion_frames.append(cm)
        pd.concat(confusion_frames, ignore_index=True).to_csv(
            run_root / "confusion_matrix.csv", index=False, lineterminator="\n"
        )

        by_source_keys = (
            "metrics_by_source", "classification_report_by_source",
            "confusion_matrix_by_source", "bootstrap_by_source",
        )
        for key in by_source_keys:
            frames = [
                _tag(r["summary"]["by_source"][key], r) for r in ok_results
                if r["summary"].get("by_source")
            ]
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(
                    run_root / f"{key}.csv", index=False, lineterminator="\n"
                )

    if folds_by_population:
        fold_tables = []
        for (dataset, dn_reliable_only), folds in folds_by_population.items():
            table = folds.fold_table.copy()
            table["dataset"] = dataset
            table["dn_reliable_only"] = dn_reliable_only
            fold_tables.append(table)
        pd.concat(fold_tables, ignore_index=True).to_csv(run_root / "folds.csv", index=False, lineterminator="\n")

    all_ok = bool(condition_results) and all(r["ok"] for r in condition_results)
    any_ok = any(r["ok"] for r in condition_results)
    if all_ok:
        final_status = art.STATUS_COMPLETED
    elif any_ok:
        final_status = art.STATUS_PARTIAL
    else:
        final_status = art.STATUS_FAILED

    art.write_status(run_root, final_status, {
        "conditions": [
            {
                "dataset": r["spec"].dataset, "condition": r["spec"].condition,
                "ok": r["ok"], "fold_status": r["fold_status"],
            }
            for r in condition_results
        ],
    })
    logger.info(f"ejecucion {run_id} finalizada con estado {final_status}")
    print(f"run_id={run_id} estado={final_status} -> {run_root}")
    return 0 if final_status == art.STATUS_COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
