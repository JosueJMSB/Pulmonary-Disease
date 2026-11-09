"""Sesion de ONNX Runtime con opciones fijas.

FP32 e INT8 se ejecutan SIEMPRE con las mismas opciones: proveedor de CPU,
``intra_op_num_threads`` indicado, ``inter_op_num_threads = 1``, ejecucion
secuencial y el mismo nivel de optimizacion del grafo. Asi, una diferencia de
tiempo se debe al modelo y no a la configuracion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def session_options_description(threads: int) -> dict:
    return {
        "provider": "CPUExecutionProvider",
        "intra_op_num_threads": int(threads),
        "inter_op_num_threads": 1,
        "execution_mode": "ORT_SEQUENTIAL",
        "graph_optimization_level": "ORT_ENABLE_ALL",
    }


def make_session(model_path: Path, threads: int):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])


def run_logits(session, inputs: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Logits ``(n,)`` en float64 para entradas ``(n, 1, n_mels, frames)``."""
    input_name = session.get_inputs()[0].name
    inputs = np.asarray(inputs, dtype=np.float32)
    out = np.empty(len(inputs), dtype=np.float64)
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        result = session.run(None, {input_name: batch})[0]
        out[start:start + len(batch)] = np.asarray(result, dtype=np.float64).reshape(-1)
    if not np.isfinite(out).all():
        raise FloatingPointError("logits no finitos en ONNX Runtime")
    return out


def onnxruntime_version() -> str:
    import onnxruntime as ort

    return ort.__version__
