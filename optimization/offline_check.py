"""Fase 6: prueba de funcionamiento sin internet (laptop).

    # phantom: carpeta de transferencia (despues de "benchmark prepare")
    python -m optimization.offline_check prepare --run-dir <corrida> [--copy-audio]

    # laptop, con conexion
    python -m optimization.offline_check run --transfer-dir <transfer> --mode online --out-dir <salida>
    # laptop, con el wifi y la red APAGADOS
    python -m optimization.offline_check run --transfer-dir <transfer> --mode offline --out-dir <salida>
    # laptop
    python -m optimization.offline_check compare --out-dir <salida>

Se ejecuta el flujo completo WAV -> preprocesamiento -> ONNX Runtime ->
decision, con todos los archivos y dependencias instalados localmente, para
FP32 e INT8. Antes de inferir se registra si habia conexion (intentos de
conexion TCP a varios servidores publicos); despues la red se bloquea dentro
del proceso. Los archivos de la carpeta de transferencia se verifican contra
``TRANSFER_MANIFEST.json`` antes de usarlos.

``compare`` da un resultado POR DATASET, con los hashes de FP32, INT8 y del
paquete probados. Un dataset es PASS solo si:

- la corrida ``online`` SI detecto conexion y la ``offline`` NO la detecto;
- figura en ambas corridas, con FP32 e INT8, y con los mismos hashes;
- sus predicciones son IDENTICAS y todas las grabaciones terminaron en OK.

Si falla la condicion de conectividad, el resultado es NO VERIFICADO.
``report.py`` solo marca CUMPLE un dataset que figure aqui con estado PASS y
con los mismos hashes que su corrida. No se desconecta phantom.
"""

from __future__ import annotations

import argparse
import shutil
import socket
import sys
from pathlib import Path

import pandas as pd

from . import bundle as bmod
from . import paths

TRANSFER_FILES = (bmod.BUNDLE_FILENAME, bmod.FP32_FILENAME, bmod.INT8_FILENAME)
INPUT_FILES = ("latency_inputs.npy", "latency_inputs_rows.csv", "e2e_recordings.csv")
VERSIONS = ("fp32", "int8")
# Varios destinos y puertos: un cortafuegos que bloquee solo uno no basta
# para declarar "sin conexion".
PROBE_TARGETS = (("1.1.1.1", 53), ("8.8.8.8", 53), ("1.1.1.1", 443), ("9.9.9.9", 443))
COMPARED_COLUMNS = ["audio_id", "model_sha256", "recording_score", "decision", "status"]


def probe_connectivity(timeout: float = 3.0) -> dict:
    """``connected`` es True si ALGUN destino responde."""
    results = {}
    for host, port in PROBE_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                results[f"{host}:{port}"] = True
        except OSError:
            results[f"{host}:{port}"] = False
    return {"connected": any(results.values()), "targets": results}


def verify_transfer(transfer: Path, datasets: list[str]) -> None:
    """Comprueba los archivos de la transferencia contra su manifiesto."""
    manifest = paths.read_json(transfer / "TRANSFER_MANIFEST.json")
    files = manifest.get("files", {})
    for name in datasets:
        if not any(k.startswith(f"{name}/") for k in files):
            raise FileNotFoundError(f"{name} no figura en {transfer / 'TRANSFER_MANIFEST.json'}")
    for rel, sha in files.items():
        if rel.startswith("audio/") or rel.split("/", 1)[0] in datasets:
            if bmod.sha256_file(transfer / rel) != sha:
                raise ValueError(f"{transfer / rel} no coincide con TRANSFER_MANIFEST.json")


def prepare(args, cfg) -> int:
    runs_root = paths.resolve_runs_root(args.runs_root, cfg)
    output_root = paths.resolve_output_root(args.output_root, cfg, runs_root)
    run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=False)
    transfer = run_dir / "transfer"
    manifest = {"created_at_utc": paths.now_iso(), "run_dir": str(run_dir), "files": {}}
    audio_root = None
    if args.copy_audio:
        from . import audio_pipeline as ap

        audio_root = Path(args.audio_root).resolve() if args.audio_root else Path(ap.preprocessing_modules().cfg.ROOT)
    for entry in paths.dataset_entries(cfg, args.dataset):
        name = entry["name"]
        src_dir = paths.dataset_dir(run_dir, name)
        bundle = bmod.load_bundle(src_dir, require_int8=True)
        for version in ("fp32", "int8"):
            bmod.verify_model_file(src_dir, bundle, version)
        dst = transfer / name
        dst.mkdir(parents=True, exist_ok=True)
        for fname in TRANSFER_FILES:
            shutil.copy2(src_dir / fname, dst / fname)
        inputs_dir = run_dir / "benchmark" / "inputs" / name
        for fname in INPUT_FILES:
            if not (inputs_dir / fname).is_file():
                raise FileNotFoundError(f"falta {inputs_dir / fname}: ejecute antes 'benchmark prepare'")
            shutil.copy2(inputs_dir / fname, dst / fname)
        if args.copy_audio:
            recordings = pd.read_csv(dst / "e2e_recordings.csv", dtype={"audio_id": str})
            audio_dst = transfer / "audio"
            audio_dst.mkdir(parents=True, exist_ok=True)
            new_paths = []
            for rec in recordings.itertuples(index=False):
                src = audio_root / rec.wav_path
                if bmod.sha256_file(src) != rec.source_sha256:
                    raise ValueError(f"{src} no coincide con el hash registrado")
                target = audio_dst / f"{rec.audio_id}{Path(rec.wav_path).suffix}"
                shutil.copy2(src, target)
                new_paths.append(f"audio/{target.name}")
                manifest["files"][f"audio/{target.name}"] = rec.source_sha256
            recordings["wav_path"] = new_paths
            paths.write_csv_atomic(recordings, dst / "e2e_recordings.csv")
        for f in sorted(dst.iterdir()):
            manifest["files"][f"{name}/{f.name}"] = bmod.sha256_file(f)
    manifest["audio_copied"] = bool(args.copy_audio)
    paths.write_json_atomic(manifest, transfer / "TRANSFER_MANIFEST.json")
    print(f"carpeta de transferencia lista: {transfer}")
    return 0


def run(args, cfg) -> int:
    from . import envinfo
    from .infer import InferenceEngine, block_network, predict_table

    transfer = Path(args.transfer_dir).resolve()
    out_dir = Path(args.out_dir).resolve() / args.mode
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [entry["name"] for entry in paths.dataset_entries(cfg, args.dataset)]
    verify_transfer(transfer, names)
    probe = probe_connectivity()
    connected = probe["connected"]
    block_network()
    if args.audio_root is not None:
        audio_root = Path(args.audio_root).resolve()
    elif (transfer / "audio").is_dir():
        audio_root = transfer
    else:
        from . import audio_pipeline as ap

        audio_root = Path(ap.preprocessing_modules().cfg.ROOT)
    meta = {"mode": args.mode, "probe_connected": connected, "probe_targets": probe["targets"],
            "network_blocked_in_process": True, "audio_root": str(audio_root), "started_at_utc": paths.now_iso(),
            "transfer_manifest_sha256": bmod.sha256_file(transfer / "TRANSFER_MANIFEST.json"),
            "datasets": {}, "environment": envinfo.collect()}
    for name in names:
        recordings = pd.read_csv(transfer / name / "e2e_recordings.csv", dtype={"audio_id": str, "patient_uid": str})
        ds_meta = {"bundle_sha256": bmod.bundle_file_sha256(transfer / name), "versions": {}}
        for version in VERSIONS:
            engine = InferenceEngine(transfer / name, version, threads=1)
            table, _ = predict_table(engine, recordings, audio_root=audio_root)
            out = out_dir / f"{name}_{version}.csv"
            # El tiempo de ejecucion varia entre corridas: no forma parte de la comparacion.
            paths.write_csv_atomic(table.drop(columns=["seconds"], errors="ignore"), out)
            ds_meta["versions"][version] = {"file": out.name, "model_sha256": engine.model_sha256,
                                            "n_ok": int((table["status"] == "OK").sum()),
                                            "n_recordings": int(len(table))}
        meta["datasets"][name] = ds_meta
    meta["finished_at_utc"] = paths.now_iso()
    paths.write_json_atomic(meta, out_dir / "run_meta.json")
    print(f"corrida '{args.mode}' terminada (conexion detectada al inicio: {connected}) -> {out_dir}")
    if args.mode == "offline" and connected:
        print("AVISO: habia conexion durante la corrida 'offline'; repitala con la red apagada", file=sys.stderr)
    if args.mode == "online" and not connected:
        print("AVISO: no se detecto conexion durante la corrida 'online'; repitala con la red encendida",
              file=sys.stderr)
    return 0


def compare_dataset(name: str, base: Path, online: dict | None, offline: dict | None) -> dict:
    """Resultado de un dataset, sin mirar la conectividad (la aplica ``compare_runs``)."""
    if online is None or offline is None:
        missing = "online" if online is None else "offline"
        return {"status": "FAIL", "detail": f"falta en la corrida {missing}"}
    out = {"bundle_sha256": online.get("bundle_sha256"), "versions": {}}
    problems = []
    if online.get("bundle_sha256") != offline.get("bundle_sha256"):
        problems.append("el paquete difiere entre corridas")
    for version in VERSIONS:
        a_info = online.get("versions", {}).get(version)
        b_info = offline.get("versions", {}).get(version)
        if a_info is None or b_info is None:
            problems.append(f"falta {version} en alguna corrida")
            continue
        out[f"{version}_sha256"] = a_info["model_sha256"]
        if a_info["model_sha256"] != b_info["model_sha256"]:
            problems.append(f"{version}: el modelo difiere entre corridas")
        a = pd.read_csv(base / "online" / a_info["file"], dtype=str, keep_default_na=False)
        b = pd.read_csv(base / "offline" / b_info["file"], dtype=str, keep_default_na=False)
        same = len(a) > 0 and a[COMPARED_COLUMNS].equals(b[COMPARED_COLUMNS])
        all_ok = bool((a["status"] == "OK").all() and (b["status"] == "OK").all())
        out["versions"][version] = {"identical": bool(same), "all_ok": all_ok, "n_recordings": int(len(a))}
        if not same:
            problems.append(f"{version}: las predicciones difieren")
        if not all_ok:
            problems.append(f"{version}: alguna grabacion no termino en OK")
    out["status"] = "FAIL" if problems else "PASS"
    out["detail"] = "; ".join(problems) if problems else "predicciones identicas con y sin conexion (FP32 e INT8)"
    return out


def compare_runs(base: Path, online_meta: dict, offline_meta: dict) -> dict:
    online_connected = online_meta.get("probe_connected")
    offline_connected = offline_meta.get("probe_connected")
    connectivity_ok = online_connected is True and offline_connected is False
    names = sorted(set(online_meta.get("datasets", {})) | set(offline_meta.get("datasets", {})))
    datasets = {}
    for name in names:
        item = compare_dataset(name, base, online_meta.get("datasets", {}).get(name),
                               offline_meta.get("datasets", {}).get(name))
        if not connectivity_ok and item["status"] == "PASS":
            item["status"] = "NO VERIFICADO"
            item["detail"] = ("predicciones identicas, pero no se demostro el cambio de conectividad "
                              f"(online conectada: {online_connected}; offline conectada: {offline_connected})")
        datasets[name] = item
    if not names:
        status, detail = "FAIL", "ninguna corrida tiene datasets"
    elif not connectivity_ok:
        status = "NO VERIFICADO"
        detail = (f"la corrida online debe tener conexion y la offline no "
                  f"(online: {online_connected}; offline: {offline_connected})")
    elif all(d["status"] == "PASS" for d in datasets.values()):
        status, detail = "PASS", "predicciones identicas con y sin conexion en todos los datasets probados"
    else:
        status, detail = "FAIL", "algun dataset fallo; ver 'datasets'"
    return {"status": status, "detail": detail, "datasets": datasets,
            "online_probe_connected": online_connected, "offline_probe_connected": offline_connected,
            "online_transfer_manifest_sha256": online_meta.get("transfer_manifest_sha256"),
            "offline_transfer_manifest_sha256": offline_meta.get("transfer_manifest_sha256"),
            "created_at_utc": paths.now_iso()}


def compare(args, cfg) -> int:
    base = Path(args.out_dir).resolve()
    online_meta = paths.read_json(base / "online" / "run_meta.json")
    offline_meta = paths.read_json(base / "offline" / "run_meta.json")
    report = compare_runs(base, online_meta, offline_meta)
    paths.write_json_atomic(report, base / "offline_comparison.json")
    print(f"prueba sin internet: {report['status']} ({report['detail']})")
    for name, item in report["datasets"].items():
        print(f"  {name}: {item['status']} ({item['detail']})")
    return 0 if report["status"] == "PASS" else 2


def parse_args(argv, cfg):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("--config", type=Path, default=None)
    p_prep.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
    p_prep.add_argument("--runs-root", type=Path, default=None)
    p_prep.add_argument("--output-root", type=Path, default=None)
    p_prep.add_argument("--run-dir", type=Path, default=None)
    p_prep.add_argument("--run-id", default=None)
    p_prep.add_argument("--copy-audio", action="store_true")
    p_prep.add_argument("--audio-root", type=Path, default=None)
    p_run = sub.add_parser("run")
    p_run.add_argument("--config", type=Path, default=None)
    p_run.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
    p_run.add_argument("--transfer-dir", type=Path, required=True)
    p_run.add_argument("--mode", choices=("online", "offline"), required=True)
    p_run.add_argument("--out-dir", type=Path, required=True)
    p_run.add_argument("--audio-root", type=Path, default=None)
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--config", type=Path, default=None)
    p_cmp.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = paths.load_config(known.config)
    args = parse_args(argv, cfg)
    return {"prepare": prepare, "run": run, "compare": compare}[args.command](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
