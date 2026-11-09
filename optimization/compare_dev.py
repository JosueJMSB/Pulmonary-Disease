"""Fase 3 (phantom): INT8 frente a FP32 en DESARROLLO y congelacion.

    python -m optimization.compare_dev --run-dir <corrida> --dataset all
    python -m optimization.compare_dev --run-dir <corrida> --dataset all --freeze

1. Informe sin tolerancia de aprobacion (``parity.int8_report``): diferencias
   de puntuacion y cambios de decision por segmento, grabacion y paciente.
2. Criterio de aceptacion fijado de antemano en ``config.toml [acceptance]``:
   perdida maxima de balanced accuracy, recall de EPOC y recall de la clase
   negativa del INT8 frente al FP32, por paciente de desarrollo.
3. ``--freeze`` congela el INT8 SOLO si el criterio se cumple y si
   ``check_pipeline`` paso para estos mismos modelos. Si no, termina con
   error y deja el intento registrado en ``attempts.jsonl``. Al congelar se
   guarda ``acceptance_sha256``; ``evaluate_test.py`` lo vuelve a comprobar.

Las metricas de desarrollo solo comparan versiones: esos pacientes
entrenaron el modelo final, asi que no estiman el desempeno.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import audio_pipeline as ap
from . import bundle as bmod
from . import dev_data as dd
from . import ort_session
from . import parity
from . import paths
from .quantize import append_attempt
from .scoring import aggregate, patient_metrics, sigmoid

EXIT_NOT_ACCEPTED = 3


def parse_args(argv, cfg):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--final-run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--freeze", action="store_true",
                        help="Congela el INT8 si cumple el criterio de aceptacion y check_pipeline paso.")
    return parser.parse_args(argv)


def _pipeline_check_ok(ds_dir: Path, bundle: dict) -> tuple[bool, str]:
    path = ds_dir / "pipeline_check.json"
    if not path.is_file():
        return False, "falta pipeline_check.json (ejecute check_pipeline antes de congelar)"
    report = paths.read_json(path)
    if report.get("status") != "PASS":
        return False, f"check_pipeline termino con estado {report.get('status')!r}"
    if report.get("fp32_sha256") != bundle["models"]["fp32"]["sha256"] or \
            report.get("int8_sha256") != bundle["models"]["int8"]["sha256"]:
        return False, "check_pipeline se hizo con otros modelos ONNX"
    return True, "PASS"


def compare_dataset(entry: dict, cfg: dict, data_root: Path, final_run_dir: Path, run_dir: Path,
                    freeze: bool) -> bool:
    name = entry["name"]
    ds_dir = paths.dataset_dir(run_dir, name)
    bundle = bmod.load_bundle(ds_dir, require_int8=True)
    if (ds_dir / "freeze.json").is_file():
        # Una vez congelado, los informes de desarrollo no se reescriben.
        frozen = paths.read_json(ds_dir / "freeze.json")
        print(f"{name}: el INT8 ya esta congelado ({frozen['int8_sha256'][:12]}...); no se repite la comparacion")
        return True
    fp32_path = bmod.verify_model_file(ds_dir, bundle, "fp32")
    int8_path = bmod.verify_model_file(ds_dir, bundle, "int8")
    final = dd.load_final_model(final_run_dir, entry)
    dd.verify_final_inputs(data_root, final)

    segments = dd.load_segments(data_root, name)
    dev = dd.select_role(segments, dd.ROLE_DEVELOPMENT)
    logmel = dd.logmel_for_rows(data_root, name, entry["branch"], dev,
                                bundle["acoustic_config"], bundle["logmel_config"])
    norm = bundle["normalization"]
    inputs = ap.normalize_logmel(logmel, norm["mean"], norm["std"])
    threads = int(cfg["inference"]["threads"])
    batch = int(cfg["inference"]["batch_size"])
    logits_fp32 = ort_session.run_logits(ort_session.make_session(fp32_path, threads), inputs, batch)
    logits_int8 = ort_session.run_logits(ort_session.make_session(int8_path, threads), inputs, batch)
    probs_fp32, probs_int8 = sigmoid(logits_fp32), sigmoid(logits_int8)
    threshold = float(bundle["decision"]["threshold"])
    negative = bundle["classes"]["negative_label_name"]

    seg_report, seg_changes = parity.int8_report(probs_fp32, probs_int8, logits_fp32, logits_int8,
                                                 dev["segment_id"].to_numpy(), threshold)
    _, rec_fp32, pat_fp32 = aggregate(dev, probs_fp32)
    _, rec_int8, pat_int8 = aggregate(dev, probs_int8)
    rec_int8 = rec_int8.set_index("audio_id").loc[rec_fp32["audio_id"]].reset_index()
    pat_int8 = pat_int8.set_index("patient_uid").loc[pat_fp32["patient_uid"]].reset_index()
    rec_report, rec_changes = parity.int8_report(rec_fp32["score"], rec_int8["score"], None, None,
                                                 rec_fp32["audio_id"].to_numpy(), threshold)
    pat_report, pat_changes = parity.int8_report(pat_fp32["score"], pat_int8["score"], None, None,
                                                 pat_fp32["patient_uid"].to_numpy(), threshold)
    metrics_fp32 = patient_metrics(pat_fp32, negative, threshold)
    metrics_int8 = patient_metrics(pat_int8, negative, threshold)
    acceptance = parity.acceptance_check(metrics_fp32, metrics_int8, cfg["acceptance"])

    report = {
        "dataset": name,
        "role": dd.ROLE_DEVELOPMENT,
        "note": acceptance["note"],
        "fp32_sha256": bundle["models"]["fp32"]["sha256"],
        "int8_sha256": bundle["models"]["int8"]["sha256"],
        "n_segments": int(len(dev)),
        "n_recordings": int(dev["audio_id"].nunique()),
        "n_patients": int(dev["patient_uid"].nunique()),
        "segment_level": seg_report,
        "recording_level": rec_report,
        "patient_level": pat_report,
        "dev_patient_metrics": {"onnx_fp32": metrics_fp32, "onnx_int8": metrics_int8},
        "acceptance": acceptance,
        "acceptance_config": cfg["acceptance"],
        "acceptance_sha256": paths.acceptance_sha256(cfg),
        "created_at_utc": paths.now_iso(),
    }
    paths.write_json_atomic(report, ds_dir / "dev_comparison.json")
    paths.write_csv_atomic(pd.DataFrame({
        "segment_id": dev["segment_id"], "audio_id": dev["audio_id"], "patient_uid": dev["patient_uid"],
        "target_label": dev["target_label"], "logit_fp32": logits_fp32, "logit_int8": logits_int8,
        "prob_fp32": probs_fp32, "prob_int8": probs_int8,
    }), ds_dir / "dev_comparison_segments.csv")
    paths.write_csv_atomic(seg_changes, ds_dir / "dev_decision_changes_segment.csv")
    paths.write_csv_atomic(rec_changes, ds_dir / "dev_decision_changes_recording.csv")
    paths.write_csv_atomic(pat_changes, ds_dir / "dev_decision_changes_patient.csv")
    append_attempt(ds_dir, {
        "at_utc": paths.now_iso(), "event": "dev_comparison", "int8_sha256": report["int8_sha256"],
        "accepted": acceptance["accepted"], "checks": acceptance["checks"],
        "acceptance_sha256": report["acceptance_sha256"],
    })
    summary = ", ".join(f"{c['metric']}: perdida {c['drop']:+.4f} (max {c['max_drop']})" for c in acceptance["checks"])
    print(f"{name}: concordancia por segmento {seg_report['concordance']:.4f}; {summary}; "
          f"{'ACEPTADO' if acceptance['accepted'] else 'NO ACEPTADO'}")

    if not freeze:
        return acceptance["accepted"]
    if not acceptance["accepted"]:
        append_attempt(ds_dir, {"at_utc": paths.now_iso(), "event": "freeze_refused",
                                "reason": "criterio de aceptacion no cumplido", "int8_sha256": report["int8_sha256"]})
        print(f"{name}: congelacion BLOQUEADA: el INT8 no cumple el criterio de aceptacion en desarrollo",
              file=sys.stderr)
        return False
    ok, detail = _pipeline_check_ok(ds_dir, bundle)
    if not ok:
        append_attempt(ds_dir, {"at_utc": paths.now_iso(), "event": "freeze_refused", "reason": detail,
                                "int8_sha256": report["int8_sha256"]})
        print(f"{name}: congelacion BLOQUEADA: {detail}", file=sys.stderr)
        return False

    frozen_at = paths.now_iso()
    bundle["models"]["int8"]["frozen"] = True
    bundle["models"]["int8"]["frozen_at_utc"] = frozen_at
    bmod.write_bundle(bundle, ds_dir, require_int8=True)
    freeze_record = {
        "dataset": name,
        "frozen_at_utc": frozen_at,
        "fp32_sha256": bundle["models"]["fp32"]["sha256"],
        "int8_sha256": bundle["models"]["int8"]["sha256"],
        "bundle_sha256": bmod.bundle_file_sha256(ds_dir),
        "acceptance": cfg["acceptance"],
        "acceptance_sha256": paths.acceptance_sha256(cfg),
        "acceptance_result": acceptance,
        "dev_comparison_sha256": bmod.sha256_file(ds_dir / "dev_comparison.json"),
        "pipeline_check_sha256": bmod.sha256_file(ds_dir / "pipeline_check.json"),
    }
    paths.write_json_atomic(freeze_record, ds_dir / "freeze.json")
    append_attempt(ds_dir, {"at_utc": frozen_at, "event": "frozen", "int8_sha256": freeze_record["int8_sha256"]})
    print(f"{name}: INT8 CONGELADO ({freeze_record['int8_sha256'][:12]}...)")
    return True


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = paths.load_config(known.config)
    args = parse_args(argv, cfg)
    data_root = paths.resolve_data_root(args.data_root, cfg)
    runs_root = paths.resolve_runs_root(args.runs_root, cfg)
    output_root = paths.resolve_output_root(args.output_root, cfg, runs_root)
    final_run_dir = paths.resolve_final_run_dir(args.final_run_dir, cfg, runs_root)
    run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=False)
    paths.record_run_config(run_dir, cfg, {"phase": "compare_dev", "freeze": args.freeze,
                                           "argv": sys.argv[1:] if argv is None else list(argv)})
    results = {e["name"]: compare_dataset(e, cfg, data_root, final_run_dir, run_dir, args.freeze)
               for e in paths.dataset_entries(cfg, args.dataset)}
    if args.freeze and not all(results.values()):
        return EXIT_NOT_ACCEPTED
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
