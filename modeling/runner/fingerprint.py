"""Huella de una ejecucion y apertura/verificacion de ``--resume``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .. import artifacts as art
from .. import data as dmod
from .. import splits as sp


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
    # Solo el protocolo holdout-v3 guarda la version del codigo; en los demas
    # ambos lados son None y esta comprobacion no cambia nada.
    if stored.get("code_sha256") != fingerprint.get("code_sha256"):
        old_files, new_files = stored.get("code_files", {}), fingerprint.get("code_files", {})
        changed = sorted(k for k in set(old_files) | set(new_files) if old_files.get(k) != new_files.get(k))
        detail = f" (archivos: {', '.join(changed[:10])}{' ...' if len(changed) > 10 else ''})" if changed else ""
        mismatches.append(
            "el codigo de modeling/ cambio desde la ejecucion original: reutilizar unidades entrenadas "
            f"con otra implementacion mezclaria resultados incoherentes{detail}"
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
