"""Fase 4 (phantom): evaluacion INT8 en el test externo, UNA vez por dataset.

    python -m optimization.evaluate_test --run-dir <corrida> --dataset ICBHI --confirm-test-access

El test externo ya se uso para evaluar los modelos ``.pt`` en la corrida
final; sus metricas publicadas se reutilizan y el ``.pt`` no se vuelve a
evaluar. Este script evalua solo el ONNX INT8 congelado.

Antes de leer ninguna fila de test exige:

- ``--confirm-test-access``;
- ``freeze.json`` (INT8 congelado tras cumplir el criterio de aceptacion);
- que el hash del INT8 y del paquete coincidan con los congelados;
- que ``acceptance_sha256`` del ``config.toml`` actual sea el mismo que al
  congelar (el test no sirve para ajustar el criterio);
- ``pipeline_check.json`` con estado PASS para esos mismos modelos;
- que los datos de la etapa final coincidan con ``model_manifest.json``.

El registro ``access_registry`` permite REANUDAR la misma evaluacion tras una
interrupcion (misma huella, misma corrida) y bloquea una evaluacion nueva,
distinta o ya completada. Si la interrupcion ocurrio despues de escribir las
predicciones y antes de actualizar el registro, el archivo se recupera solo
si su contenido concuerda con los segmentos de test. Un bloqueo por dataset
impide dos evaluaciones simultaneas.

ONNX FP32 no se evalua en test: su paridad con ``.pt`` se verifico en
desarrollo. Las tablas lo indican asi y no le atribuyen metricas de test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import access_registry as reg
from . import audio_pipeline as ap
from . import bundle as bmod
from . import dev_data as dd
from . import envinfo
from . import ort_session
from . import paths
from .parity import decision_changes, difference_stats
from .scoring import SUMMARY_METRICS, aggregate, bootstrap_table, confusion_table, patient_metrics, sigmoid

PREDICTIONS_FILE = "test_segment_predictions_int8.csv"
FP32_TEST_NOTE = "no evaluado en test; paridad con .pt verificada en desarrollo"
PREDICTION_COLUMNS = ("segment_id", "audio_id", "patient_uid", "target_label", "logit_int8", "score")


class PreconditionError(RuntimeError):
    pass


def load_validated_predictions(path: Path, test: pd.DataFrame, dataset: str) -> np.ndarray:
    """Lee las predicciones INT8 guardadas y comprueba que corresponden, fila a
    fila, a los segmentos de test actuales. Devuelve los logits.

    Lanza ``TestAccessInconsistent`` si algo no concuerda: no se reintenta ni
    se sobrescribe en silencio."""
    stored = pd.read_csv(path, dtype={"segment_id": str, "audio_id": str, "patient_uid": str})
    problems = []
    missing = sorted(set(PREDICTION_COLUMNS) - set(stored.columns))
    if missing:
        problems.append(f"faltan columnas {missing}")
    elif len(stored) != len(test):
        problems.append(f"{len(stored)} filas frente a {len(test)} segmentos de test")
    else:
        for col in ("segment_id", "audio_id", "patient_uid"):
            if not np.array_equal(stored[col].to_numpy(dtype=str), test[col].to_numpy(dtype=str)):
                problems.append(f"la columna {col} no coincide con los segmentos de test")
        if not np.array_equal(stored["target_label"].to_numpy(dtype=np.int64),
                              test["target_label"].to_numpy(dtype=np.int64)):
            problems.append("las etiquetas no coinciden con los segmentos de test")
        logits = stored["logit_int8"].to_numpy(dtype=np.float64)
        if not np.isfinite(logits).all():
            problems.append("hay logits no finitos")
        elif not np.allclose(stored["score"].to_numpy(dtype=np.float64), sigmoid(logits), rtol=0.0, atol=1e-12):
            problems.append("score no es sigmoid(logit_int8)")
    if problems:
        raise reg.TestAccessInconsistent(
            f"{dataset}: {path} no es valido ({'; '.join(problems)}); requiere revision manual"
        )
    return logits


def parse_args(argv, cfg):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=[e["name"] for e in cfg["datasets"]], required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--final-run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--confirm-test-access", action="store_true",
                        help="Obligatorio: confirma que se evaluara el INT8 en el test externo.")
    return parser.parse_args(argv)


def check_preconditions(ds_dir: Path, cfg: dict) -> tuple[dict, dict]:
    freeze_path = ds_dir / "freeze.json"
    if not freeze_path.is_file():
        raise PreconditionError("el INT8 no esta congelado (ejecute compare_dev --freeze)")
    freeze = paths.read_json(freeze_path)
    bundle = bmod.load_bundle(ds_dir, require_int8=True)
    if not bundle["models"]["int8"].get("frozen"):
        raise PreconditionError("el paquete no marca el INT8 como congelado")
    if bmod.bundle_file_sha256(ds_dir) != freeze["bundle_sha256"]:
        raise PreconditionError("model_bundle.json cambio despues de congelar")
    bmod.verify_model_file(ds_dir, bundle, "int8")
    if bundle["models"]["int8"]["sha256"] != freeze["int8_sha256"]:
        raise PreconditionError("el INT8 no es el congelado")
    if paths.acceptance_sha256(cfg) != freeze["acceptance_sha256"]:
        raise PreconditionError(
            "el criterio [acceptance] de config.toml cambio despues de congelar; el test no puede usarse para ajustarlo"
        )
    check_path = ds_dir / "pipeline_check.json"
    if not check_path.is_file():
        raise PreconditionError("falta pipeline_check.json (ejecute check_pipeline)")
    check = paths.read_json(check_path)
    if check.get("status") != "PASS":
        raise PreconditionError(f"check_pipeline no paso (estado {check.get('status')!r})")
    if check.get("int8_sha256") != freeze["int8_sha256"] or check.get("fp32_sha256") != freeze["fp32_sha256"]:
        raise PreconditionError("check_pipeline se hizo con otros modelos ONNX")
    if bmod.sha256_file(check_path) != freeze.get("pipeline_check_sha256"):
        raise PreconditionError("pipeline_check.json cambio despues de congelar")
    return bundle, freeze


def evaluate(entry: dict, cfg: dict, data_root: Path, final_run_dir: Path, run_dir: Path, output_root: Path) -> None:
    name = entry["name"]
    ds_dir = paths.dataset_dir(run_dir, name)
    bundle, freeze = check_preconditions(ds_dir, cfg)
    final = dd.load_final_model(final_run_dir, entry)
    data_hashes = dd.verify_final_inputs(data_root, final)
    prefix = f"{name}/{dd.STAGE}"

    test_dir = ds_dir / "test_int8"
    test_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = test_dir / PREDICTIONS_FILE
    fingerprint = {
        "dataset": name,
        "condition": entry["condition"],
        "int8_sha256": freeze["int8_sha256"],
        "bundle_sha256": freeze["bundle_sha256"],
        "acceptance_sha256": freeze["acceptance_sha256"],
        "segments_csv_sha256": data_hashes[f"{prefix}/segments.csv"],
        "segments_npy_sha256": data_hashes[f"{prefix}/segments_{entry['branch']}.npy"],
        "onnxruntime_version": ort_session.onnxruntime_version(),
    }
    access_dir = paths.test_access_dir(output_root, cfg)
    # Todo lo que lee o escribe el registro ocurre con el bloqueo del dataset.
    with reg.exclusive_lock(access_dir, name):
        _evaluate_locked(name, entry, cfg, data_root, ds_dir, run_dir, bundle, final, fingerprint, access_dir,
                         test_dir, predictions_path)


def _evaluate_locked(name, entry, cfg, data_root, ds_dir, run_dir, bundle, final, fingerprint, access_dir,
                     test_dir, predictions_path) -> None:
    mode = reg.begin(access_dir, name, fingerprint, run_dir, predictions_path)
    print(f"{name}: evaluacion INT8 del test -> modo {mode}")

    segments = dd.load_segments(data_root, name)
    test = dd.select_role(segments, dd.ROLE_TEST)
    if mode == reg.RESUME_REUSE_PREDICTIONS:
        logits = load_validated_predictions(predictions_path, test, name)
    elif mode == reg.RESUME_RECOVER_PREDICTIONS:
        # La interrupcion ocurrio entre escribir el archivo y actualizar el
        # registro: se valida y se registra, sin volver a inferir.
        logits = load_validated_predictions(predictions_path, test, name)
        reg.mark_predictions_written(access_dir, name, predictions_path)
        print(f"{name}: predicciones recuperadas de {predictions_path}")
    else:
        logmel = dd.logmel_for_rows(data_root, name, entry["branch"], test,
                                    bundle["acoustic_config"], bundle["logmel_config"])
        norm = bundle["normalization"]
        inputs = ap.normalize_logmel(logmel, norm["mean"], norm["std"])
        session = ort_session.make_session(ds_dir / bundle["models"]["int8"]["file"], int(cfg["inference"]["threads"]))
        logits = ort_session.run_logits(session, inputs, int(cfg["inference"]["batch_size"]))
        columns = ["segment_id", "audio_id", "patient_uid", "target_label"] + [
            c for c in ("source_dataset", "device") if c in test.columns]
        frame = test[columns].assign(logit_int8=logits, score=sigmoid(logits))
        paths.write_csv_atomic(frame, predictions_path)
        reg.mark_predictions_written(access_dir, name, predictions_path)

    threshold = float(bundle["decision"]["threshold"])
    negative = bundle["classes"]["negative_label_name"]
    probs = sigmoid(logits)
    scored, recordings, patients = aggregate(test, probs)
    if "source_dataset" in test.columns:
        by_patient = test.drop_duplicates("patient_uid").set_index("patient_uid")["source_dataset"]
        patients["source_dataset"] = patients["patient_uid"].map(by_patient)
    metrics = patient_metrics(patients, negative, threshold)
    boot = final.selection_snapshot["bootstrap"]
    ci = bootstrap_table(patients, threshold, int(boot["n_resamples"]), float(boot["confidence"]),
                         int(boot["random_state"]))

    # Referencia publicada del modelo base .pt (no se reevalua).
    base_metrics = paths.read_json(final.model_dir / "test_metrics.json")
    base_patients = pd.read_csv(final.model_dir / "test_patient_predictions.csv", dtype={"patient_uid": str})
    base_segments = pd.read_csv(final.model_dir / "test_segment_predictions.csv", dtype={"segment_id": str})
    merged_seg = scored[["segment_id", "score"]].merge(base_segments[["segment_id", "score"]], on="segment_id",
                                                       suffixes=("_int8", "_base"), how="inner")
    if len(merged_seg) != len(scored):
        raise RuntimeError(f"{name}: los segmentos de test no coinciden con las predicciones publicadas del .pt")
    merged_pat = patients[["patient_uid", "score"]].merge(base_patients[["patient_uid", "score"]], on="patient_uid",
                                                          suffixes=("_int8", "_base"), how="inner")
    seg_changes = decision_changes(merged_seg["segment_id"], merged_seg["score_base"], merged_seg["score_int8"],
                                   threshold, "base_pt", "int8")
    pat_changes = decision_changes(merged_pat["patient_uid"], merged_pat["score_base"], merged_pat["score_int8"],
                                   threshold, "base_pt", "int8")

    # Las ocho metricas del resultado base, con sus IC (los del .pt, publicados).
    base_ci_path = final.model_dir / "test_bootstrap_ci.csv"
    base_ci = pd.read_csv(base_ci_path).set_index("metric") if base_ci_path.is_file() else None
    int8_ci = ci.set_index("metric")
    rows = []
    for m in SUMMARY_METRICS:
        row = {
            "metric": m,
            "base_pt_publicado": float(base_metrics[m]),
            "onnx_int8": float(metrics[m]),
            "onnx_fp32": FP32_TEST_NOTE,
            "drop_points": 100.0 * (float(base_metrics[m]) - float(metrics[m])),
            "int8_ci_lower": float(int8_ci.loc[m, "lower"]) if m in int8_ci.index else float("nan"),
            "int8_ci_upper": float(int8_ci.loc[m, "upper"]) if m in int8_ci.index else float("nan"),
        }
        if base_ci is not None and m in base_ci.index:
            row["base_ci_lower"] = float(base_ci.loc[m, "lower"])
            row["base_ci_upper"] = float(base_ci.loc[m, "upper"])
        rows.append(row)
    comparison = pd.DataFrame(rows)

    paths.write_json_atomic(metrics, test_dir / "test_metrics_int8.json")
    paths.write_csv_atomic(ci, test_dir / "test_bootstrap_ci_int8.csv")
    paths.write_csv_atomic(confusion_table(patients, negative, threshold).reset_index(),
                           test_dir / "test_confusion_matrix_int8.csv")
    paths.write_csv_atomic(recordings, test_dir / "test_recording_predictions_int8.csv")
    paths.write_csv_atomic(patients, test_dir / "test_patient_predictions_int8.csv")
    paths.write_csv_atomic(comparison, test_dir / "test_comparison_base_vs_int8.csv")
    paths.write_csv_atomic(seg_changes, test_dir / "test_decision_changes_segment.csv")
    paths.write_csv_atomic(pat_changes, test_dir / "test_decision_changes_patient.csv")
    summary = {
        "dataset": name,
        "labels": {"base_pt_publicado": "modelo base .pt, metricas publicadas en la corrida final",
                   "onnx_int8": "ONNX INT8 congelado, evaluado una vez en esta corrida",
                   "onnx_fp32": FP32_TEST_NOTE},
        "n_test_patients": int(len(patients)),
        "n_test_copd": int((patients["target_label"] == 1).sum()),
        "n_test_segments": int(len(test)),
        "metrics_int8": metrics,
        "metrics_base_pt_publicado": base_metrics,
        "segment_concordance_with_base": float(1.0 - len(seg_changes) / len(merged_seg)),
        "patient_decision_changes": int(len(pat_changes)),
        "abs_segment_score_diff_vs_base": difference_stats(merged_seg["score_base"], merged_seg["score_int8"]),
        "small_test_warning": (
            f"El test de {name} tiene {len(patients)} pacientes ({int((patients['target_label'] == 1).sum())} con EPOC): "
            "un solo paciente cambia mucho las metricas y los intervalos son amplios."
        ),
        "fingerprint": fingerprint,
        "environment": envinfo.collect(),
        "created_at_utc": paths.now_iso(),
    }
    paths.write_json_atomic(summary, test_dir / "test_summary_int8.json")
    reg.mark_completed(access_dir, name, {"test_dir": str(test_dir), "summary_sha256": bmod.sha256_file(
        test_dir / "test_summary_int8.json")})
    print(f"{name}: BA INT8 = {metrics['balanced_accuracy']:.4f} frente a {base_metrics['balanced_accuracy']:.4f} "
          f"del .pt publicado; recall EPOC {metrics['recall_copd']:.4f} -> {test_dir}")


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = paths.load_config(known.config)
    args = parse_args(argv, cfg)
    if not args.confirm_test_access:
        print("falta --confirm-test-access: no se accede al test", file=sys.stderr)
        return 2
    data_root = paths.resolve_data_root(args.data_root, cfg)
    runs_root = paths.resolve_runs_root(args.runs_root, cfg)
    output_root = paths.resolve_output_root(args.output_root, cfg, runs_root)
    final_run_dir = paths.resolve_final_run_dir(args.final_run_dir, cfg, runs_root)
    run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=False)
    entry = paths.dataset_entries(cfg, args.dataset)[0]
    try:
        check_preconditions(paths.dataset_dir(run_dir, entry["name"]), cfg)
    except PreconditionError as exc:
        print(f"{entry['name']}: no se accede al test: {exc}", file=sys.stderr)
        return 2
    paths.record_run_config(run_dir, cfg, {"phase": "evaluate_test", "dataset": entry["name"],
                                           "argv": sys.argv[1:] if argv is None else list(argv)})
    try:
        evaluate(entry, cfg, data_root, final_run_dir, run_dir, output_root)
    except (reg.TestAccessBlocked, reg.TestAccessInconsistent) as exc:
        print(f"{entry['name']}: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
