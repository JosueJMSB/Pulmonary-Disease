"""Fase 2 (phantom, onnx + onnxruntime): ONNX FP32 -> ONNX INT8.

    python -m optimization.quantize --run-dir <corrida> --dataset all

- Cuantizacion estatica con ONNX Runtime (formato, tipos, canal y metodo de
  calibracion en ``config.toml [quantization]``).
- Calibracion SOLO con segmentos de pacientes de desarrollo
  (``role == "train"``), elegidos de forma reproducible y repartidos entre
  pacientes. Nunca con pacientes de test.
- Guarda el inventario de calibracion, los parametros completos y el
  analisis de que operadores quedaron en INT8 y cuales siguieron en FP32.
- Escribe el bloque ``models.int8`` del paquete (hash y metadatos propios,
  separados de ``models.fp32``) y registra el intento en ``attempts.jsonl``.
- Se niega a actuar si la paridad FP32 no paso o si el INT8 ya esta congelado.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import audio_pipeline as ap
from . import bundle as bmod
from . import dev_data as dd
from . import envinfo
from . import ort_session
from . import paths

QDQ_OPS = ("QuantizeLinear", "DequantizeLinear")


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
    parser.add_argument("--overwrite", action="store_true",
                        help="Nuevo intento de cuantizacion (solo antes de congelar).")
    return parser.parse_args(argv)


def analyze_operators(model) -> dict:
    """Analisis estatico del grafo QDQ: un nodo de calculo cuenta como
    cuantizado si todas sus entradas no constantes vienen de un
    ``DequantizeLinear``. ONNX Runtime fusiona despues esos patrones en
    operadores enteros; este analisis describe el grafo guardado."""
    nodes = list(model.graph.node)
    initializers = {i.name for i in model.graph.initializer}
    dq_outputs = {o for n in nodes if n.op_type == "DequantizeLinear" for o in n.output}
    graph_inputs = {i.name for i in model.graph.input}
    per_op = {}
    for node in nodes:
        if node.op_type in QDQ_OPS:
            continue
        inputs = [i for i in node.input if i]
        variable = [i for i in inputs if i not in initializers]
        quantized = bool(variable) and all(i in dq_outputs for i in variable)
        stats = per_op.setdefault(node.op_type, {"quantized": 0, "fp32": 0})
        stats["quantized" if quantized else "fp32"] += 1
        if not quantized and any(i in graph_inputs for i in variable):
            stats["reads_graph_input"] = True
    counts = Counter(n.op_type for n in nodes)
    return {
        "method": "analisis estatico del grafo QDQ guardado",
        "node_counts": dict(sorted(counts.items())),
        "n_quantize_linear": int(counts.get("QuantizeLinear", 0)),
        "n_dequantize_linear": int(counts.get("DequantizeLinear", 0)),
        "per_op_type": per_op,
        "n_compute_nodes": int(sum(v["quantized"] + v["fp32"] for v in per_op.values())),
        "n_quantized": int(sum(v["quantized"] for v in per_op.values())),
        "n_fp32": int(sum(v["fp32"] for v in per_op.values())),
        "fp32_op_types": sorted(k for k, v in per_op.items() if v["fp32"] > 0),
    }


def _calibration_reader(inputs: np.ndarray, input_name: str, batch_size: int):
    from onnxruntime.quantization import CalibrationDataReader

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._batches = [inputs[i:i + batch_size] for i in range(0, len(inputs), batch_size)]
            self._index = 0

        def get_next(self):
            if self._index >= len(self._batches):
                return None
            batch = self._batches[self._index]
            self._index += 1
            return {input_name: np.ascontiguousarray(batch, dtype=np.float32)}

        def rewind(self):
            self._index = 0

    return _Reader()


def append_attempt(ds_dir: Path, record: dict) -> None:
    path = ds_dir / "attempts.jsonl"
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def quantize_dataset(entry: dict, cfg: dict, data_root: Path, final_run_dir: Path, run_dir: Path,
                     overwrite: bool) -> None:
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    name = entry["name"]
    ds_dir = paths.dataset_dir(run_dir, name)
    if (ds_dir / "freeze.json").is_file():
        raise RuntimeError(f"{name}: el INT8 ya esta congelado; no se vuelve a cuantizar")
    bundle = bmod.load_bundle(ds_dir)
    if bundle["models"]["fp32"]["parity"]["status"] != "PASS":
        raise RuntimeError(f"{name}: la paridad PyTorch -> ONNX FP32 no paso; no se cuantiza")
    fp32_path = bmod.verify_model_file(ds_dir, bundle, "fp32")
    int8_path = ds_dir / bmod.INT8_FILENAME
    if int8_path.exists() and not overwrite:
        raise FileExistsError(f"{name}: ya existe {int8_path}; use --overwrite para un nuevo intento")

    final = dd.load_final_model(final_run_dir, entry)
    dd.verify_final_inputs(data_root, final)

    qcfg = cfg["quantization"]
    segments = dd.load_segments(data_root, name)
    dev = dd.select_role(segments, dd.ROLE_DEVELOPMENT)
    calibration = dd.sample_patients_stratified(dev, int(qcfg["calibration_segments"]), int(qcfg["seed"]))
    if set(calibration["role"]) != {dd.ROLE_DEVELOPMENT}:
        raise RuntimeError(f"{name}: la calibracion contiene filas fuera de desarrollo")
    inventory = calibration[["segment_id", "audio_id", "patient_uid", "target_label", "role", "task_array_index"]]
    paths.write_csv_atomic(inventory, ds_dir / "calibration_inventory.csv")

    logmel = dd.logmel_for_rows(data_root, name, entry["branch"], calibration,
                                bundle["acoustic_config"], bundle["logmel_config"])
    norm = bundle["normalization"]
    inputs = ap.normalize_logmel(logmel, norm["mean"], norm["std"])

    source_path = fp32_path
    preprocess = {"requested": bool(qcfg["run_preprocess"]), "ran": False, "error": None}
    if qcfg["run_preprocess"]:
        pre_path = ds_dir / "model_fp32_preprocessed.onnx"
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process

            quant_pre_process(str(fp32_path), str(pre_path))
            source_path = pre_path
            preprocess["ran"] = True
            preprocess["file"] = pre_path.name
            preprocess["sha256"] = bmod.sha256_file(pre_path)
        except Exception as exc:  # noqa: BLE001 - se registra y se cuantiza el FP32 sin preprocesar
            preprocess["error"] = f"{type(exc).__name__}: {exc}"

    input_name = bundle["input"]["name"]
    reader = _calibration_reader(inputs, input_name, int(qcfg["calibration_batch_size"]))
    op_types = list(qcfg["op_types_to_quantize"]) or None
    tmp = int8_path.with_name(int8_path.name + ".part")
    quantize_static(
        model_input=str(source_path),
        model_output=str(tmp),
        calibration_data_reader=reader,
        quant_format=getattr(QuantFormat, qcfg["format"]),
        activation_type=getattr(QuantType, qcfg["activation_type"]),
        weight_type=getattr(QuantType, qcfg["weight_type"]),
        per_channel=bool(qcfg["per_channel"]),
        reduce_range=bool(qcfg["reduce_range"]),
        calibrate_method=getattr(CalibrationMethod, qcfg["calibration_method"]),
        op_types_to_quantize=op_types,
    )
    os.replace(tmp, int8_path)
    int8_model = onnx.load(str(int8_path))
    onnx.checker.check_model(int8_model)
    operators = analyze_operators(int8_model)
    paths.write_json_atomic(operators, ds_dir / "quantization_ops.json")

    versions = envinfo.package_versions()
    quantization = {
        "method": "static post-training quantization (onnxruntime.quantization.quantize_static)",
        "format": qcfg["format"],
        "activation_type": qcfg["activation_type"],
        "weight_type": qcfg["weight_type"],
        "per_channel": bool(qcfg["per_channel"]),
        "reduce_range": bool(qcfg["reduce_range"]),
        "calibration_method": qcfg["calibration_method"],
        "op_types_to_quantize": op_types or "todos los admitidos",
        "preprocess": preprocess,
        "source_model_sha256": bmod.sha256_file(source_path),
    }
    bundle["models"]["int8"] = {
        "file": bmod.INT8_FILENAME,
        "sha256": bmod.sha256_file(int8_path),
        "size_bytes": int(int8_path.stat().st_size),
        "precision": "int8 (QDQ)",
        "derived_from_fp32_sha256": bundle["models"]["fp32"]["sha256"],
        "onnx_version": versions.get("onnx"),
        "onnxruntime_version": ort_session.onnxruntime_version(),
        "created_at_utc": paths.now_iso(),
        "quantization": quantization,
        "calibration": {
            "roles": [dd.ROLE_DEVELOPMENT],
            "n_segments": int(len(calibration)),
            "n_patients": int(calibration["patient_uid"].nunique()),
            "seed": int(qcfg["seed"]),
            "batch_size": int(qcfg["calibration_batch_size"]),
            "inventory": "calibration_inventory.csv",
            "inventory_sha256": bmod.sha256_file(ds_dir / "calibration_inventory.csv"),
        },
        "operators": {
            "report": "quantization_ops.json",
            "n_compute_nodes": operators["n_compute_nodes"],
            "n_quantized": operators["n_quantized"],
            "n_fp32": operators["n_fp32"],
            "fp32_op_types": operators["fp32_op_types"],
        },
        "frozen": False,
        "frozen_at_utc": None,
    }
    bmod.write_bundle(bundle, ds_dir, require_int8=True)
    append_attempt(ds_dir, {
        "at_utc": paths.now_iso(), "event": "quantized", "int8_sha256": bundle["models"]["int8"]["sha256"],
        "quantization": quantization, "calibration_inventory_sha256": bundle["models"]["int8"]["calibration"]["inventory_sha256"],
    })
    paths.write_json_atomic({"phase": "quantize", **envinfo.collect(), "load": envinfo.system_load()},
                            ds_dir / "environment_quantize.json")
    print(f"{name}: INT8 creado ({operators['n_quantized']} de {operators['n_compute_nodes']} nodos de calculo "
          f"en INT8; en FP32: {operators['fp32_op_types']}) -> {int8_path}")


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
    paths.record_run_config(run_dir, cfg, {"phase": "quantize", "argv": sys.argv[1:] if argv is None else list(argv)})
    for entry in paths.dataset_entries(cfg, args.dataset):
        quantize_dataset(entry, cfg, data_root, final_run_dir, run_dir, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
