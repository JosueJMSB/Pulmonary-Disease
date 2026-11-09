"""Exportacion y cuantizacion de una CNN DIMINUTA con datos aleatorios.

Solo comprueba que el encadenamiento PyTorch -> ONNX FP32 -> ONNX INT8
funciona con las versiones instaladas; no usa los checkpoints reales.
Se omite si faltan torch, onnx u onnxruntime (por ejemplo, en la laptop).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from modeling.models.cnn import CopdCNN  # noqa: E402
from optimization import export_onnx, ort_session, parity, paths, quantize  # noqa: E402
from optimization.scoring import sigmoid  # noqa: E402


def test_tiny_cnn_export_parity_and_int8(tmp_path):
    torch.manual_seed(0)
    model = CopdCNN(channels=(4, 8), hidden_units=8, dropout=0.0).eval()
    cfg = paths.load_config()
    fp32_path = tmp_path / "tiny_fp32.onnx"
    export_onnx._export(model, [1, 16, 40], fp32_path, cfg)
    onnx.checker.check_model(onnx.load(str(fp32_path)))

    rng = np.random.RandomState(0)
    inputs = rng.randn(32, 1, 16, 40).astype(np.float32)
    logits_torch = export_onnx._torch_logits(model, inputs, 8)
    logits_onnx = ort_session.run_logits(ort_session.make_session(fp32_path, 1), inputs, 8)
    report, _ = parity.fp32_parity(logits_torch, logits_onnx, sigmoid(logits_torch), sigmoid(logits_onnx),
                                   np.arange(32), 0.5, cfg["parity_fp32"]["max_abs_logit"],
                                   cfg["parity_fp32"]["max_abs_prob"])
    assert report["status"] == "PASS", report

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    int8_path = tmp_path / "tiny_int8.onnx"
    reader = quantize._calibration_reader(inputs, cfg["export"]["input_name"], 8)
    quantize_static(str(fp32_path), str(int8_path), reader, quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True,
                    calibrate_method=CalibrationMethod.MinMax)
    ops = quantize.analyze_operators(onnx.load(str(int8_path)))
    assert ops["n_dequantize_linear"] > 0
    assert ops["n_compute_nodes"] == ops["n_quantized"] + ops["n_fp32"]
    logits_int8 = ort_session.run_logits(ort_session.make_session(int8_path, 1), inputs, 8)
    int8_report, _ = parity.int8_report(sigmoid(logits_onnx), sigmoid(logits_int8), logits_onnx, logits_int8,
                                        np.arange(32), 0.5)
    assert int8_report["n_items"] == 32
