"""Comprobacion previa al test (phantom): flujo WAV -> decision sin PyTorch.

    python -m optimization.check_pipeline --run-dir <corrida> --dataset all

Para un conjunto fijo de grabaciones de DESARROLLO (semilla de
``config.toml [pipeline_check]``):

1. Lee el WAV original (cuyo hash se compara con el registrado por la fase 2)
   y lo procesa con ``audio_pipeline`` usando SOLO el paquete.
2. Compara los segmentos y el log-mel obtenidos con los almacenados en la
   etapa final (``segments_<rama>.npy``), con tolerancia.
3. Compara las probabilidades ONNX FP32 del flujo completo con las de los
   segmentos almacenados (cuya paridad con PyTorch ya se verifico en la
   exportacion), con tolerancia estricta. Para INT8 solo se registran las
   diferencias.
4. Ejecuta ``InferenceEngine.predict_wav`` (el mismo codigo de ``infer.py``)
   y comprueba en un proceso aparte que ``optimization.infer`` no carga torch.

Escribe ``pipeline_check.json`` con estado PASS o FAIL. ``compare_dev
--freeze`` y ``evaluate_test`` exigen PASS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio_pipeline as ap
from . import bundle as bmod
from . import dev_data as dd
from . import ort_session
from . import paths
from .infer import InferenceEngine
from .parity import difference_stats
from .scoring import sigmoid


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
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Raiz de las rutas source_path del informe de la fase 2 (por defecto, el repositorio).")
    return parser.parse_args(argv)


def infer_imports_torch() -> bool:
    code = "import sys, optimization.infer; print(int('torch' in sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], cwd=str(paths.REPO_ROOT),
                            capture_output=True, text=True, check=True)
    return result.stdout.strip().endswith("1")


def load_phase2_sources() -> pd.DataFrame:
    cfg = ap.preprocessing_modules().cfg
    report = pd.read_csv(cfg.R2_RESAMPLING, dtype={"audio_id": str})
    return report.loc[report["status"] == "OK"].set_index("audio_id")


def check_dataset(entry: dict, cfg: dict, data_root: Path, final_run_dir: Path, run_dir: Path,
                  audio_root: Path, torch_loaded: bool) -> str:
    name = entry["name"]
    ds_dir = paths.dataset_dir(run_dir, name)
    bundle = bmod.load_bundle(ds_dir, require_int8=True)
    fp32_path = bmod.verify_model_file(ds_dir, bundle, "fp32")
    int8_path = bmod.verify_model_file(ds_dir, bundle, "int8")
    final = dd.load_final_model(final_run_dir, entry)
    dd.verify_final_inputs(data_root, final)
    warnings = ap.verify_runtime_matches(bundle["preprocessing"])

    pc = cfg["pipeline_check"]
    segments = dd.load_segments(data_root, name)
    dev = dd.select_role(segments, dd.ROLE_DEVELOPMENT)
    audio_ids = dd.sample_recordings(dev, int(pc["n_recordings"]), int(pc["seed"]))
    sources = load_phase2_sources()
    threads = int(cfg["inference"]["threads"])
    sess_fp32 = ort_session.make_session(fp32_path, threads)
    sess_int8 = ort_session.make_session(int8_path, threads)
    engine = InferenceEngine(ds_dir, "fp32", threads)
    norm = bundle["normalization"]

    rows = []
    for audio_id in audio_ids:
        row = {"audio_id": audio_id, "status": "OK", "detail": ""}
        try:
            if audio_id not in sources.index:
                raise FileNotFoundError(f"{audio_id} no figura en el informe de la fase 2")
            src = sources.loc[audio_id]
            wav = Path(audio_root) / src["source_path"]
            if bmod.sha256_file(wav) != src["source_sha256"]:
                raise ValueError(f"el WAV {wav} no coincide con el hash registrado en la fase 2")
            stored_rows = dev.loc[dev["audio_id"] == audio_id].sort_values("start_sample")
            stored = dd.stored_segments_for_rows(data_root, name, entry["branch"], stored_rows)
            x, sr = ap.read_wav(wav)
            new, info = ap.process_waveform(x, sr, bundle["preprocessing"])
            row.update({"source_sr": int(sr), "n_segments_stored": int(len(stored)), "n_segments_new": int(len(new))})
            if len(new) != len(stored):
                raise ValueError(f"{len(new)} segmentos nuevos frente a {len(stored)} almacenados")
            row["max_abs_waveform"] = float(np.max(np.abs(new.astype(np.float64) - stored.astype(np.float64))))
            lm_new = ap.logmel_from_segments(new, bundle["acoustic_config"], bundle["logmel_config"])
            lm_stored = ap.logmel_from_segments(stored, bundle["acoustic_config"], bundle["logmel_config"])
            row["max_abs_logmel_db"] = float(np.max(np.abs(lm_new.astype(np.float64) - lm_stored.astype(np.float64))))
            in_new = ap.normalize_logmel(lm_new, norm["mean"], norm["std"])
            in_stored = ap.normalize_logmel(lm_stored, norm["mean"], norm["std"])
            p32_new = sigmoid(ort_session.run_logits(sess_fp32, in_new))
            p32_stored = sigmoid(ort_session.run_logits(sess_fp32, in_stored))
            p8_new = sigmoid(ort_session.run_logits(sess_int8, in_new))
            p8_stored = sigmoid(ort_session.run_logits(sess_int8, in_stored))
            row["max_abs_prob_fp32"] = difference_stats(p32_new, p32_stored)["max"]
            row["max_abs_prob_int8"] = difference_stats(p8_new, p8_stored)["max"]
            engine_result = engine.predict_wav(wav)
            row["recording_score_engine_fp32"] = engine_result["recording_score"]
            row["recording_score_stored_fp32"] = float(p32_stored.mean())
            row["abs_recording_diff_fp32"] = abs(engine_result["recording_score"] - float(p32_stored.mean()))
            row["decision_change_fp32"] = bool(
                (engine_result["recording_score"] >= engine.threshold) != (float(p32_stored.mean()) >= engine.threshold))
            failures = []
            if row["max_abs_waveform"] > float(pc["max_abs_waveform"]):
                failures.append("forma de onda")
            if row["max_abs_logmel_db"] > float(pc["max_abs_logmel_db"]):
                failures.append("log-mel")
            if row["max_abs_prob_fp32"] > float(pc["max_abs_prob_fp32"]) or \
                    row["abs_recording_diff_fp32"] > float(pc["max_abs_prob_fp32"]):
                failures.append("probabilidad FP32")
            if failures:
                row["status"] = "FAIL"
                row["detail"] = "fuera de tolerancia: " + ", ".join(failures)
        except Exception as exc:  # noqa: BLE001 - cada grabacion se registra; el estado global sera FAIL
            row["status"] = "FAIL"
            row["detail"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    table = pd.DataFrame(rows)
    paths.write_csv_atomic(table, ds_dir / "pipeline_check_recordings.csv")
    all_ok = bool((table["status"] == "OK").all()) and not torch_loaded
    report = {
        "dataset": name,
        "status": "PASS" if all_ok else "FAIL",
        "role": dd.ROLE_DEVELOPMENT,
        "fp32_sha256": bundle["models"]["fp32"]["sha256"],
        "int8_sha256": bundle["models"]["int8"]["sha256"],
        "n_recordings": int(len(table)),
        "n_failed": int((table["status"] != "OK").sum()),
        "tolerances": dict(pc),
        "infer_imports_torch": bool(torch_loaded),
        "int8_note": "Las diferencias INT8 (flujo frente a almacenado) solo se registran; no deciden el estado.",
        "max_abs_waveform": float(table.get("max_abs_waveform", pd.Series(dtype=float)).max()),
        "max_abs_logmel_db": float(table.get("max_abs_logmel_db", pd.Series(dtype=float)).max()),
        "max_abs_prob_fp32": float(table.get("max_abs_prob_fp32", pd.Series(dtype=float)).max()),
        "max_abs_prob_int8": float(table.get("max_abs_prob_int8", pd.Series(dtype=float)).max()),
        "warnings": warnings,
        "created_at_utc": paths.now_iso(),
    }
    paths.write_json_atomic(report, ds_dir / "pipeline_check.json")
    print(f"{name}: check_pipeline {report['status']} ({report['n_failed']} de {report['n_recordings']} "
          f"grabaciones fallidas; torch cargado por infer: {torch_loaded})")
    return report["status"]


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
    audio_root = Path(args.audio_root).resolve() if args.audio_root else Path(ap.preprocessing_modules().cfg.ROOT)
    paths.record_run_config(run_dir, cfg, {"phase": "check_pipeline", "audio_root": str(audio_root),
                                           "argv": sys.argv[1:] if argv is None else list(argv)})
    torch_loaded = infer_imports_torch()
    statuses = [check_dataset(e, cfg, data_root, final_run_dir, run_dir, audio_root, torch_loaded)
                for e in paths.dataset_entries(cfg, args.dataset)]
    return 0 if all(s == "PASS" for s in statuses) else 2


if __name__ == "__main__":
    raise SystemExit(main())
