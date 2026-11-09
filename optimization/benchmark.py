"""Fase 5: mediciones de recursos de ONNX FP32 frente a ONNX INT8 (solo CPU).

    # phantom: entradas fijas de DESARROLLO (una sola vez por corrida)
    python -m optimization.benchmark prepare --run-dir <corrida>

    # phantom (o laptop, con la carpeta de transferencia)
    python -m optimization.benchmark run --run-dir <corrida> --label phantom

Ambas versiones se miden en la misma CPU y con las mismas condiciones:
mismas entradas, lote de 1 segmento, mismos hilos (``[benchmark] threads``),
mismas opciones de ONNX Runtime, mismo calentamiento y mismas repeticiones.
Solo se usan audios y segmentos de desarrollo: el test no se reabre.

Medidas:

- tamano de los archivos ONNX FP32 e INT8 (el ``.pt`` solo como dato aparte);
- latencia del modelo por segmento: mediana y p95;
- tiempo total WAV -> decision por grabacion: mediana y p95;
- pico de RAM del proceso en cuatro modos, cada uno en un proceso nuevo:
  ``baseline_model`` (onnxruntime sin modelo), ``model`` (sesion + inferencia
  sobre log-mel ya calculado), ``baseline_pipeline`` (todas las librerias del
  flujo, sin modelo) y ``pipeline`` (WAV -> decision). El IOV de memoria se
  verifica con ``pipeline``; los demas modos son explicativos.
- especificaciones de la CPU y carga del sistema antes y despues.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

MEMORY_MODES = ("baseline_model", "model", "baseline_pipeline", "pipeline")


# ---------------------------------------------------------------------------
# Utilidades (sin dependencias pesadas)
# ---------------------------------------------------------------------------

def latency_stats(values) -> dict:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "median": float("nan"), "p95": float("nan"), "mean": float("nan"),
                "min": float("nan"), "max": float("nan")}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def relative_change_pct(reference: float, candidate: float) -> float:
    """Cambio relativo en % (positivo = el candidato es mayor)."""
    if reference == 0 or not np.isfinite(reference) or not np.isfinite(candidate):
        return float("nan")
    return 100.0 * (candidate - reference) / reference


def peak_rss_bytes() -> int | None:
    """Pico de memoria residente del proceso actual."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        if hasattr(info, "peak_wset"):  # Windows
            return int(info.peak_wset)
    except ImportError:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        return None


def current_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        with open("/proc/self/statm", encoding="utf-8") as fh:
            import os

            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Tareas hijas: cada medicion corre en un proceso nuevo
# ---------------------------------------------------------------------------

def _child_latency(args) -> dict:
    from . import ort_session

    inputs = np.load(args.inputs)
    session = ort_session.make_session(args.model, args.threads)
    name = session.get_inputs()[0].name
    n = len(inputs)
    for i in range(args.warmup):
        session.run(None, {name: inputs[i % n:i % n + 1]})
    times = []
    for i in range(args.reps):
        batch = inputs[i % n:i % n + 1]
        start = time.perf_counter_ns()
        session.run(None, {name: batch})
        times.append((time.perf_counter_ns() - start) / 1e6)
    return {"times_ms": times}


def _child_e2e(args) -> dict:
    from .infer import InferenceEngine

    recordings = pd.read_csv(args.recordings, dtype={"audio_id": str})
    engine = InferenceEngine(args.bundle_dir, args.version, args.threads)
    audio_root = Path(args.audio_root)
    first = audio_root / recordings.iloc[0]["wav_path"]
    for _ in range(max(1, args.warmup // 10)):
        engine.predict_wav(first)
    rows = []
    for rec in recordings.itertuples(index=False):
        wav = audio_root / rec.wav_path
        for rep in range(args.reps):
            start = time.perf_counter()
            result = engine.predict_wav(wav)
            rows.append({"audio_id": rec.audio_id, "rep": rep, "seconds": time.perf_counter() - start,
                         "duration_s": result["duration_s"], "n_segments": result["n_segments"]})
    return {"rows": rows}


def _child_memory(args) -> dict:
    mode = args.mode
    if mode in ("baseline_model", "model"):
        import onnxruntime  # noqa: F401

        from . import ort_session

        rss_after_imports = current_rss_bytes()
        if mode == "model":
            inputs = np.load(args.inputs)
            session = ort_session.make_session(args.model, args.threads)
            name = session.get_inputs()[0].name
            for i in range(len(inputs)):
                session.run(None, {name: inputs[i:i + 1]})
    else:
        from . import audio_pipeline as ap
        from . import infer

        ap.preprocessing_modules()
        from modeling.features import logmel  # noqa: F401

        rss_after_imports = current_rss_bytes()
        if mode == "pipeline":
            recordings = pd.read_csv(args.recordings, dtype={"audio_id": str})
            engine = infer.InferenceEngine(args.bundle_dir, args.version, args.threads)
            for rec in recordings.itertuples(index=False):
                engine.predict_wav(Path(args.audio_root) / rec.wav_path)
    return {"mode": mode, "peak_rss_bytes": peak_rss_bytes(), "rss_after_imports_bytes": rss_after_imports,
            "torch_loaded": "torch" in sys.modules}


def _run_child(task: str, out_json: Path, **kwargs) -> dict:
    cmd = [sys.executable, "-m", "optimization.benchmark", "_child", "--task", task, "--out", str(out_json)]
    for key, value in kwargs.items():
        if value is not None:
            cmd += [f"--{key.replace('_', '-')}", str(value)]
    subprocess.run(cmd, cwd=str(paths.REPO_ROOT), check=True)
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# prepare: entradas fijas de desarrollo
# ---------------------------------------------------------------------------

def prepare(args, cfg) -> int:
    from . import audio_pipeline as ap
    from . import bundle as bmod
    from . import dev_data as dd
    from .check_pipeline import load_phase2_sources

    data_root = paths.resolve_data_root(args.data_root, cfg)
    runs_root = paths.resolve_runs_root(args.runs_root, cfg)
    output_root = paths.resolve_output_root(args.output_root, cfg, runs_root)
    final_run_dir = paths.resolve_final_run_dir(args.final_run_dir, cfg, runs_root)
    run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=False)
    bcfg = cfg["benchmark"]
    sources = load_phase2_sources()
    for entry in paths.dataset_entries(cfg, args.dataset):
        name = entry["name"]
        ds_dir = paths.dataset_dir(run_dir, name)
        bundle = bmod.load_bundle(ds_dir, require_int8=True)
        final = dd.load_final_model(final_run_dir, entry)
        dd.verify_final_inputs(data_root, final)
        dev = dd.select_role(dd.load_segments(data_root, name), dd.ROLE_DEVELOPMENT)
        out_dir = run_dir / "benchmark" / "inputs" / name
        out_dir.mkdir(parents=True, exist_ok=True)

        lat_rows = dd.sample_patients_stratified(dev, int(bcfg["latency_segments"]), int(bcfg["seed"]))
        logmel = dd.logmel_for_rows(data_root, name, entry["branch"], lat_rows,
                                    bundle["acoustic_config"], bundle["logmel_config"])
        inputs = ap.normalize_logmel(logmel, bundle["normalization"]["mean"], bundle["normalization"]["std"])
        tmp = out_dir / "latency_inputs.part.npy"
        np.save(tmp, inputs)
        tmp.replace(out_dir / "latency_inputs.npy")
        paths.write_csv_atomic(lat_rows[["segment_id", "audio_id", "patient_uid", "role"]],
                               out_dir / "latency_inputs_rows.csv")

        audio_ids = dd.sample_recordings(dev, int(bcfg["e2e_recordings"]), int(bcfg["seed"]))
        meta = dev.drop_duplicates("audio_id").set_index("audio_id")
        rows = []
        for audio_id in audio_ids:
            src = sources.loc[audio_id]
            rows.append({"audio_id": audio_id, "wav_path": src["source_path"], "source_sha256": src["source_sha256"],
                         "patient_uid": meta.loc[audio_id, "patient_uid"],
                         "target_label": int(meta.loc[audio_id, "target_label"]), "role": dd.ROLE_DEVELOPMENT})
        paths.write_csv_atomic(pd.DataFrame(rows), out_dir / "e2e_recordings.csv")
        print(f"{name}: {len(inputs)} segmentos y {len(rows)} grabaciones de desarrollo preparados -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# run: mediciones
# ---------------------------------------------------------------------------

def run(args, cfg) -> int:
    from . import bundle as bmod
    from . import envinfo

    bcfg = cfg["benchmark"]
    if args.bundle_root is not None:
        bundle_root = Path(args.bundle_root).resolve()
        inputs_root = Path(args.inputs_root).resolve() if args.inputs_root else bundle_root
        out_root = Path(args.out_dir).resolve() if args.out_dir else bundle_root / "benchmark_results" / args.label
    else:
        runs_root = paths.resolve_runs_root(args.runs_root, cfg)
        output_root = paths.resolve_output_root(args.output_root, cfg, runs_root)
        run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=False)
        bundle_root = run_dir
        inputs_root = Path(args.inputs_root).resolve() if args.inputs_root else run_dir / "benchmark" / "inputs"
        out_root = Path(args.out_dir).resolve() if args.out_dir else run_dir / "benchmark" / args.label
    if args.audio_root is not None:
        audio_root = Path(args.audio_root).resolve()
    else:
        from . import audio_pipeline as ap

        audio_root = Path(ap.preprocessing_modules().cfg.ROOT)
    threads_list = [int(t) for t in (args.threads or bcfg["threads"])]
    warmup, reps = int(bcfg["warmup"]), int(bcfg["repetitions"])
    e2e_reps, mem_threads = int(bcfg["e2e_repetitions"]), int(bcfg["memory_threads"])

    for entry in paths.dataset_entries(cfg, args.dataset):
        name = entry["name"]
        bundle_dir = bundle_root / name
        bundle = bmod.load_bundle(bundle_dir, require_int8=True)
        model_paths = {v: bmod.verify_model_file(bundle_dir, bundle, v) for v in ("fp32", "int8")}
        inputs = inputs_root / name / "latency_inputs.npy"
        recordings = inputs_root / name / "e2e_recordings.csv"
        out_dir = out_root / name
        tmp_dir = out_dir / "_child"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        environment = {"label": args.label, "before": envinfo.system_load(), **envinfo.collect()}

        sizes = {
            "onnx_fp32_bytes": int(model_paths["fp32"].stat().st_size),
            "onnx_int8_bytes": int(model_paths["int8"].stat().st_size),
            "checkpoint_pt_bytes_informative": bundle["source"].get("checkpoint_size_bytes"),
            "note": "La reduccion de tamano se calcula solo entre ONNX FP32 e ONNX INT8.",
        }
        sizes["reduction_pct"] = -relative_change_pct(sizes["onnx_fp32_bytes"], sizes["onnx_int8_bytes"])

        lat_rows, e2e_rows, mem_rows = [], [], []
        for threads in threads_list:
            for version in ("fp32", "int8"):
                res = _run_child("latency", tmp_dir / f"lat_{version}_{threads}.json", model=model_paths[version],
                                 inputs=inputs, threads=threads, warmup=warmup, reps=reps)
                lat_rows += [{"version": version, "threads": threads, "rep": i, "ms": t}
                             for i, t in enumerate(res["times_ms"])]
                res = _run_child("e2e", tmp_dir / f"e2e_{version}_{threads}.json", bundle_dir=bundle_dir,
                                 version=version, threads=threads, recordings=recordings, audio_root=audio_root,
                                 warmup=warmup, reps=e2e_reps)
                e2e_rows += [{"version": version, "threads": threads, **r} for r in res["rows"]]
        for version in ("fp32", "int8"):
            for mode in MEMORY_MODES:
                if mode.startswith("baseline") and version == "int8":
                    continue  # la base no depende del modelo
                res = _run_child("memory", tmp_dir / f"mem_{mode}_{version}.json", mode=mode,
                                 model=model_paths[version], bundle_dir=bundle_dir, version=version,
                                 threads=mem_threads, inputs=inputs, recordings=recordings, audio_root=audio_root)
                mem_rows.append({"version": "-" if mode.startswith("baseline") else version, "threads": mem_threads,
                                 **res})

        lat = pd.DataFrame(lat_rows)
        e2e = pd.DataFrame(e2e_rows)
        mem = pd.DataFrame(mem_rows)
        lat_summary = pd.DataFrame([{"version": v, "threads": t, **latency_stats(g["ms"])}
                                    for (v, t), g in lat.groupby(["version", "threads"])])
        e2e_summary = pd.DataFrame([{"version": v, "threads": t, **latency_stats(g["seconds"]),
                                     "share_faster_than_audio": float((g["seconds"] < g["duration_s"]).mean())}
                                    for (v, t), g in e2e.groupby(["version", "threads"])])
        environment["after"] = envinfo.system_load()
        environment["conditions"] = {
            "batch_size": 1, "threads": threads_list, "warmup": warmup, "repetitions": reps,
            "e2e_repetitions": e2e_reps, "memory_threads": mem_threads,
            "inputs": "segmentos y grabaciones de desarrollo (role == 'train'); el test no se usa",
            "session_options": {"intra_op_num_threads": "segun threads", "inter_op_num_threads": 1,
                                "execution_mode": "ORT_SEQUENTIAL", "graph_optimization_level": "ORT_ENABLE_ALL"},
            "memory_iov_mode": "pipeline",
        }
        paths.write_json_atomic(sizes, out_dir / "sizes.json")
        paths.write_csv_atomic(lat, out_dir / "latency_raw.csv")
        paths.write_csv_atomic(lat_summary, out_dir / "latency_summary.csv")
        paths.write_csv_atomic(e2e, out_dir / "e2e_raw.csv")
        paths.write_csv_atomic(e2e_summary, out_dir / "e2e_summary.csv")
        paths.write_csv_atomic(mem, out_dir / "memory.csv")
        paths.write_json_atomic(environment, out_dir / "environment.json")
        print(f"{name}: benchmark '{args.label}' terminado -> {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv, cfg):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--config", type=Path, default=None)
        p.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
        p.add_argument("--runs-root", type=Path, default=None)
        p.add_argument("--output-root", type=Path, default=None)
        p.add_argument("--run-dir", type=Path, default=None)
        p.add_argument("--run-id", default=None)

    p_prep = sub.add_parser("prepare", help="Entradas fijas de desarrollo (phantom).")
    common(p_prep)
    p_prep.add_argument("--data-root", type=Path, default=None)
    p_prep.add_argument("--final-run-dir", type=Path, default=None)

    p_run = sub.add_parser("run", help="Mediciones de FP32 e INT8.")
    common(p_run)
    p_run.add_argument("--label", required=True, help="Nombre de la plataforma, p. ej. phantom o laptop.")
    p_run.add_argument("--bundle-root", type=Path, default=None,
                       help="Carpeta con <dataset>/model_bundle.json (p. ej. la carpeta de transferencia en la laptop).")
    p_run.add_argument("--inputs-root", type=Path, default=None)
    p_run.add_argument("--audio-root", type=Path, default=None)
    p_run.add_argument("--out-dir", type=Path, default=None)
    p_run.add_argument("--threads", type=int, nargs="+", default=None)

    p_child = sub.add_parser("_child", help=argparse.SUPPRESS)
    p_child.add_argument("--task", choices=("latency", "e2e", "memory"), required=True)
    p_child.add_argument("--out", type=Path, required=True)
    p_child.add_argument("--model", type=Path, default=None)
    p_child.add_argument("--bundle-dir", type=Path, default=None)
    p_child.add_argument("--version", default=None)
    p_child.add_argument("--threads", type=int, default=1)
    p_child.add_argument("--inputs", type=Path, default=None)
    p_child.add_argument("--recordings", type=Path, default=None)
    p_child.add_argument("--audio-root", type=Path, default=None)
    p_child.add_argument("--warmup", type=int, default=0)
    p_child.add_argument("--reps", type=int, default=1)
    p_child.add_argument("--mode", choices=MEMORY_MODES, default=None)
    p_child.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = paths.load_config(known.config)
    args = parse_args(argv, cfg)
    if args.command == "_child":
        handlers = {"latency": _child_latency, "e2e": _child_e2e, "memory": _child_memory}
        result = handlers[args.task](args)
        paths.write_json_atomic(result, args.out)
        return 0
    if args.command == "prepare":
        return prepare(args, cfg)
    return run(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
