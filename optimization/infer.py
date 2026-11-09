"""Inferencia WAV -> decision con un paquete ONNX, SIN PyTorch.

    python -m optimization.infer --bundle-dir <dir del paquete> --version int8 \
        --wav grabacion1.wav grabacion2.wav --out predicciones.csv --block-network

    python -m optimization.infer --bundle-dir <dir> --version fp32 \
        --list grabaciones.csv --audio-root <dir de los WAV> --out predicciones.csv

``--list`` es un CSV con ``audio_id`` y ``wav_path`` (absoluta o relativa a
``--audio-root``) y, opcionalmente, ``patient_uid`` y ``target_label``. Con
``patient_uid`` tambien se escribe la decision por paciente
(``<out>_patients.csv``): media de las grabaciones del paciente.

Todo ocurre en el equipo local: modelo ONNX, ``model_bundle.json`` y el
codigo de ``preprocessing/`` y ``modeling/features/``. ``--block-network``
impide abrir conexiones de red durante la ejecucion: cualquier intento
termina en error.

Este modulo no importa torch (lo comprueban ``check_env.py`` y
``tests/test_no_torch.py``).
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import audio_pipeline as ap
from . import bundle as bmod
from . import ort_session


class NetworkBlockedError(RuntimeError):
    pass


def block_network() -> None:
    """Hace fallar cualquier intento de conexion de red en este proceso."""

    def _refuse(*_args, **_kwargs):
        raise NetworkBlockedError("acceso a la red bloqueado por --block-network")

    socket.socket.connect = _refuse
    socket.socket.connect_ex = _refuse
    socket.create_connection = _refuse
    socket.getaddrinfo = _refuse


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


class InferenceEngine:
    """Paquete + modelo ONNX verificado + sesion de ONNX Runtime."""

    def __init__(self, bundle_dir: Path, version: str, threads: int = 1, batch_size: int = 64):
        self.bundle_dir = Path(bundle_dir)
        self.version = version
        self.bundle = bmod.load_bundle(self.bundle_dir, require_int8=(version == "int8"))
        self.model_path = bmod.verify_model_file(self.bundle_dir, self.bundle, version)
        self.model_sha256 = self.bundle["models"][version]["sha256"]
        self.warnings = ap.verify_runtime_matches(self.bundle["preprocessing"])
        self.threads = int(threads)
        self.batch_size = int(batch_size)
        self.session = ort_session.make_session(self.model_path, self.threads)
        self.threshold = float(self.bundle["decision"]["threshold"])

    def logits(self, inputs: np.ndarray) -> np.ndarray:
        return ort_session.run_logits(self.session, inputs, self.batch_size)

    def predict_wav(self, wav_path: Path) -> dict:
        start = time.perf_counter()
        inputs, info = ap.wav_to_model_input(Path(wav_path), self.bundle)
        probs = _sigmoid(self.logits(inputs))
        score = float(probs.mean())
        elapsed = time.perf_counter() - start
        return {
            "recording_score": score,
            "decision": int(score >= self.threshold),
            "segment_scores": probs,
            "seconds": elapsed,
            **info,
        }


def _resolve_wav(path_value: str, audio_root: Path | None, list_dir: Path | None) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    if audio_root is not None:
        return audio_root / path
    if list_dir is not None:
        return list_dir / path
    return path.resolve()


def predict_table(engine: InferenceEngine, recordings: pd.DataFrame, audio_root: Path | None = None,
                  list_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    rows = []
    for rec in recordings.itertuples(index=False):
        wav = _resolve_wav(getattr(rec, "wav_path"), audio_root, list_dir)
        row = {"audio_id": getattr(rec, "audio_id"), "wav_path": str(wav)}
        for optional in ("patient_uid", "target_label"):
            if hasattr(rec, optional):
                row[optional] = getattr(rec, optional)
        try:
            result = engine.predict_wav(wav)
            row.update({
                "recording_score": repr(result["recording_score"]),
                "decision": result["decision"],
                "decision_name": "COPD" if result["decision"] else engine.bundle["classes"]["negative_label_name"],
                "n_segments": result["n_segments"], "duration_s": result["duration_s"],
                "source_sr": result["source_sr"], "seconds": result["seconds"], "status": "OK", "error": "",
            })
        except (ap.RecordingTooShortError, ap.PipelineConfigError, OSError, ValueError) as exc:
            row.update({"recording_score": "", "decision": "", "decision_name": "", "status": "ERROR",
                        "error": f"{type(exc).__name__}: {exc}"})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.insert(0, "version", engine.version)
    table.insert(1, "model_sha256", engine.model_sha256)

    patients = None
    ok = table.loc[table["status"] == "OK"]
    if "patient_uid" in table.columns and not ok.empty:
        scores = ok.assign(score=ok["recording_score"].astype(float))
        agg = {"score": ("score", "mean"), "n_recordings": ("audio_id", "size")}
        if "target_label" in scores.columns:
            agg["target_label"] = ("target_label", "first")
        patients = scores.groupby("patient_uid", as_index=False).agg(**agg)
        patients["decision"] = (patients["score"] >= engine.threshold).astype(int)
        patients["score"] = patients["score"].map(repr)
        patients.insert(0, "version", engine.version)
    return table, patients


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--version", choices=bmod.VERSIONS, default="int8")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=Path, nargs="+")
    group.add_argument("--list", type=Path)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--block-network", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.block_network:
        block_network()
    engine = InferenceEngine(args.bundle_dir, args.version, args.threads)
    for warning in engine.warnings:
        print(f"AVISO: {warning}", file=sys.stderr)
    if args.wav:
        recordings = pd.DataFrame({"audio_id": [p.stem for p in args.wav], "wav_path": [str(p) for p in args.wav]})
        list_dir = None
    else:
        recordings = pd.read_csv(args.list, dtype={"audio_id": str, "patient_uid": str})
        list_dir = args.list.resolve().parent
    table, patients = predict_table(engine, recordings, args.audio_root, list_dir)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False, lineterminator="\n")
        if patients is not None:
            patients.to_csv(args.out.with_name(args.out.stem + "_patients.csv"), index=False, lineterminator="\n")
    else:
        print(table.drop(columns=["model_sha256"]).to_string(index=False))
    return 0 if (table["status"] == "OK").all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
