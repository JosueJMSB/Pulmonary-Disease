"""Fase 1 (phantom, con PyTorch y onnx): checkpoint .pt -> ONNX FP32.

    python -m optimization.export_onnx --dataset all \
        --data-root <ruta> --runs-root <ruta> --output-root <ruta>

Para cada CNN final:

1. Verifica el hash del checkpoint y de los datos de la etapa final contra
   ``model_manifest.json``.
2. Exporta la red a ONNX FP32 (lote dinamico) y valida el grafo con
   ``onnx.checker``.
3. Comprueba la paridad PyTorch -> ONNX FP32 en segmentos de DESARROLLO con
   tolerancia estricta (``config.toml [parity_fp32]``). Los cambios de
   decision se reportan aparte.
4. Escribe el paquete independiente ``model_bundle.json`` con todos los
   parametros de inferencia y el bloque ``models.fp32``.

Nunca toca el test externo ni modifica la corrida final.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio_pipeline as ap
from . import bundle as bmod
from . import dev_data as dd
from . import envinfo
from . import ort_session
from . import parity
from . import paths
from .scoring import sigmoid


def parse_args(argv=None, cfg=None):
    cfg = cfg or paths.load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--final-run-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None, help="Corrida de optimization (se crea si no existe).")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Reemplaza un ONNX FP32 ya exportado (nunca si el INT8 ya esta congelado).")
    return parser.parse_args(argv)


def _export(model, input_shape, out_path: Path, cfg: dict) -> str:
    import torch

    exp = cfg["export"]
    dummy = torch.zeros((2, *input_shape), dtype=torch.float32)
    kwargs = dict(
        input_names=[exp["input_name"]],
        output_names=[exp["output_name"]],
        dynamic_axes={exp["input_name"]: {0: "batch"}, exp["output_name"]: {0: "batch"}},
        opset_version=int(exp["opset"]),
        do_constant_folding=True,
    )
    tmp = out_path.with_name(out_path.name + ".part")
    try:
        torch.onnx.export(model, (dummy,), str(tmp), dynamo=bool(exp["dynamo"]), **kwargs)
        exporter = "dynamo" if exp["dynamo"] else "torchscript"
    except TypeError:
        # Versiones de PyTorch sin el argumento ``dynamo``.
        torch.onnx.export(model, (dummy,), str(tmp), **kwargs)
        exporter = "torchscript (sin argumento dynamo)"
    os.replace(tmp, out_path)
    return exporter


def _torch_logits(model, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    import torch

    out = np.empty(len(inputs), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch_size], dtype=np.float32))
            out[start:start + len(batch)] = model(batch).reshape(-1).double().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("logits no finitos en PyTorch")
    return out


def export_dataset(entry: dict, cfg: dict, data_root: Path, final_run_dir: Path, run_dir: Path,
                   overwrite: bool) -> str:
    import onnx
    import torch

    from modeling.models import cnn as cnn_model

    name = entry["name"]
    ds_dir = paths.dataset_dir(run_dir, name)
    ds_dir.mkdir(parents=True, exist_ok=True)
    if (ds_dir / "freeze.json").is_file():
        raise RuntimeError(f"{name}: el INT8 ya esta congelado en {ds_dir}; no se vuelve a exportar")
    fp32_path = ds_dir / bmod.FP32_FILENAME
    if fp32_path.exists() and not overwrite:
        raise FileExistsError(f"{name}: ya existe {fp32_path}; use --overwrite para reemplazarlo")

    final = dd.load_final_model(final_run_dir, entry)
    checkpoint_sha = dd.verify_checkpoint(final)
    data_hashes = dd.verify_final_inputs(data_root, final)
    params = dd.load_preprocessing_params(data_root, name)
    preprocessing_block = ap.build_preprocessing_block(params, entry["branch"])
    warnings = ap.verify_runtime_matches(preprocessing_block)

    payload = cnn_model.load_checkpoint(final.checkpoint_path)
    model = cnn_model.model_from_checkpoint(payload).to("cpu").float().eval()
    n_params = int(sum(p.numel() for p in model.parameters()))
    expected = int(cfg["model"]["expected_parameters"])
    if n_params != expected or int(payload["architecture"]["n_parameters"]) != expected:
        raise RuntimeError(f"{name}: {n_params} parametros, se esperaban {expected}")
    threshold = float(payload["decision_threshold"])
    if threshold != final.threshold:
        raise RuntimeError(f"{name}: umbral del checkpoint {threshold} distinto del manifiesto {final.threshold}")
    if payload.get("branch") != entry["branch"]:
        raise RuntimeError(f"{name}: el checkpoint es de la rama {payload.get('branch')!r}")

    input_shape = [int(v) for v in payload["input_shape"]]
    exporter = _export(model, input_shape, fp32_path, cfg)
    onnx_model = onnx.load(str(fp32_path))
    onnx.checker.check_model(onnx_model)

    norm = payload["normalization"]
    mean = [float(v) for v in norm["mean"].tolist()]
    std = [float(v) for v in norm["std"].tolist()]
    acoustic_cfg = dict(payload["acoustic_config"])
    logmel_cfg = dict(payload["logmel_config"])

    # Paridad estricta en desarrollo.
    segments = dd.load_segments(data_root, name)
    dev = dd.select_role(segments, dd.ROLE_DEVELOPMENT)
    max_segments = int(cfg["parity_fp32"]["max_segments"])
    if max_segments > 0 and len(dev) > max_segments:
        dev = dd.sample_patients_stratified(dev, max_segments, int(cfg["quantization"]["seed"]))
    logmel = dd.logmel_for_rows(data_root, name, entry["branch"], dev, acoustic_cfg, logmel_cfg)
    inputs = ap.normalize_logmel(logmel, mean, std)
    batch = int(cfg["export"]["batch_size"])
    torch.set_num_threads(int(cfg["inference"]["threads"]))
    logits_torch = _torch_logits(model, inputs, batch)
    session = ort_session.make_session(fp32_path, int(cfg["inference"]["threads"]))
    logits_onnx = ort_session.run_logits(session, inputs, batch)
    probs_torch, probs_onnx = sigmoid(logits_torch), sigmoid(logits_onnx)
    report, changes = parity.fp32_parity(
        logits_torch, logits_onnx, probs_torch, probs_onnx, dev["segment_id"].to_numpy(), threshold,
        float(cfg["parity_fp32"]["max_abs_logit"]), float(cfg["parity_fp32"]["max_abs_prob"]),
    )
    report.update({"dataset": name, "role": dd.ROLE_DEVELOPMENT, "n_segments": int(len(dev)),
                   "n_patients": int(dev["patient_uid"].nunique()), "warnings": warnings})
    paths.write_json_atomic(report, ds_dir / "export_parity.json")
    paths.write_csv_atomic(pd.DataFrame({
        "segment_id": dev["segment_id"], "audio_id": dev["audio_id"], "patient_uid": dev["patient_uid"],
        "logit_torch": logits_torch, "logit_onnx_fp32": logits_onnx,
        "prob_torch": probs_torch, "prob_onnx_fp32": probs_onnx,
    }), ds_dir / "export_parity_segments.csv")
    paths.write_csv_atomic(changes, ds_dir / "export_parity_decision_changes.csv")

    versions = envinfo.package_versions()
    bundle = {
        "format_version": bmod.BUNDLE_FORMAT_VERSION,
        "dataset": name,
        "condition": entry["condition"],
        "branch": entry["branch"],
        "architecture": payload["architecture"],
        "input": {"name": cfg["export"]["input_name"], "shape": ["batch", *input_shape], "dtype": "float32",
                  "axes": ["segment", "channel", "mel_band", "frame"]},
        "output": {"name": cfg["export"]["output_name"], "meaning": "logit; sigmoid(logit) = P(COPD) por segmento"},
        "normalization": {"axis": "mel_band", "mean": mean, "std": std,
                          "n_segments": int(norm["n_segments"]), "weighted": bool(norm["weighted"]),
                          "formula": "(logmel - mean) / std, en float32"},
        "acoustic_config": acoustic_cfg,
        "logmel_config": logmel_cfg,
        "preprocessing": preprocessing_block,
        "classes": {"names": list(payload["class_names"]), "label_mapping": dict(payload["label_mapping"]),
                    "negative_label_name": str(payload["class_names"][0]), "positive_label_name": "COPD"},
        "decision": {"score": "sigmoid(logit)", "threshold": threshold, "rule": "score >= threshold -> COPD"},
        "aggregation": {"segment_to_recording": "mean", "recording_to_patient": "mean",
                        "description": payload.get("aggregation")},
        "source": {
            "final_run_id": cfg["final_run"]["run_id"],
            "final_model_dir": f"final_training/{cfg['final_run']['run_id']}/datasets/{name}/{entry['condition']}",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_size_bytes": int(final.checkpoint_path.stat().st_size),
            "model_manifest_sha256": bmod.sha256_file(final.manifest_path),
            "pipeline_fingerprint": final.manifest["hashes"].get("pipeline_fingerprint"),
            "split_csv_sha256": final.manifest["hashes"].get("split_csv_sha256"),
            "split_manifest_sha256": final.manifest["hashes"].get("split_manifest_sha256"),
            "final_data_hashes": data_hashes,
        },
        "models": {
            "fp32": {
                "file": bmod.FP32_FILENAME,
                "sha256": bmod.sha256_file(fp32_path),
                "size_bytes": int(fp32_path.stat().st_size),
                "precision": "float32",
                "opset": int(cfg["export"]["opset"]),
                "exporter": exporter,
                "torch_version": torch.__version__,
                "onnx_version": versions.get("onnx"),
                "onnxruntime_version": ort_session.onnxruntime_version(),
                "created_at_utc": paths.now_iso(),
                "parity": {
                    "status": report["status"],
                    "max_abs_logit": report["abs_logit_diff"]["max"],
                    "max_abs_prob": report["abs_prob_diff"]["max"],
                    "n_decision_changes": report["n_decision_changes"],
                    "n_segments": report["n_segments"],
                    "report": "export_parity.json",
                    "report_sha256": bmod.sha256_file(ds_dir / "export_parity.json"),
                },
            },
            "int8": None,
        },
        "created_at_utc": paths.now_iso(),
    }
    bmod.write_bundle(bundle, ds_dir)
    paths.write_json_atomic({"phase": "export_onnx", **envinfo.collect(), "load": envinfo.system_load()},
                            ds_dir / "environment_export.json")
    print(f"{name}: ONNX FP32 exportado; paridad {report['status']} "
          f"(max |dlogit| = {report['abs_logit_diff']['max']:.3g}, "
          f"max |dprob| = {report['abs_prob_diff']['max']:.3g}, "
          f"cambios de decision = {report['n_decision_changes']}) -> {ds_dir}")
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
    run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=True)
    paths.record_run_config(run_dir, cfg, {
        "phase": "export_onnx", "argv": sys.argv[1:] if argv is None else list(argv),
        "data_root": str(data_root), "final_run_dir": str(final_run_dir), "output_root": str(output_root),
    })
    print(f"corrida de optimization: {run_dir}")
    statuses = {}
    for entry in paths.dataset_entries(cfg, args.dataset):
        statuses[entry["name"]] = export_dataset(entry, cfg, data_root, final_run_dir, run_dir, args.overwrite)
    failed = [k for k, v in statuses.items() if v != "PASS"]
    if failed:
        print(f"PARIDAD FP32 FALLIDA en {failed}: no continue con la cuantizacion", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
