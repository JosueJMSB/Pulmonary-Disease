"""Fase 0 (phantom): comprueba que existen los WAV ORIGINALES antes de empezar.

    python -m optimization.check_audio --dataset all [--audio-root <raiz>] [--no-hash]

``check_pipeline``, ``benchmark prepare/run`` (tiempo total WAV -> decision) y
``offline_check prepare --copy-audio`` leen los WAV originales indicados en
``source_path`` del informe de la fase 2
(``preprocessing/reports/phase2/2b_resampling.csv``). No bastan los ``.npy``
de segmentos ni los WAV ya remuestreados a 4 kHz.

Para cada dataset se revisan TODAS las grabaciones de desarrollo
(``role == "train"``); las de test no se usan en estas fases y no se revisan.
Por defecto se compara tambien el SHA-256 de cada WAV con ``source_sha256``
del informe (``--no-hash`` solo comprueba que el archivo exista).

Termina con codigo 0 si todo esta disponible y 1 si falta algo. No escribe
nada salvo con ``--out``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import audio_pipeline as ap
from . import bundle as bmod
from . import dev_data as dd
from . import paths


def check_dataset(name: str, data_root: Path, audio_root: Path, sources: pd.DataFrame, verify_hash: bool) -> dict:
    dev = dd.select_role(dd.load_segments(data_root, name), dd.ROLE_DEVELOPMENT)
    audio_ids = sorted(dev["audio_id"].unique())
    missing_report, missing_file, bad_hash = [], [], []
    for audio_id in audio_ids:
        if audio_id not in sources.index:
            missing_report.append(audio_id)
            continue
        src = sources.loc[audio_id]
        wav = audio_root / src["source_path"]
        if not wav.is_file():
            missing_file.append(str(wav))
        elif verify_hash and bmod.sha256_file(wav) != src["source_sha256"]:
            bad_hash.append(str(wav))
    ok = not (missing_report or missing_file or bad_hash)
    return {"status": "OK" if ok else "FALTA", "n_development_recordings": len(audio_ids),
            "missing_in_phase2_report": missing_report, "missing_files": missing_file,
            "hash_mismatch": bad_hash, "hash_verified": bool(verify_hash)}


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = paths.load_config(known.config)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Raiz de las rutas source_path (por defecto, la raiz del repositorio).")
    parser.add_argument("--no-hash", action="store_true", help="Solo comprobar que los archivos existan.")
    parser.add_argument("--out", type=Path, default=None, help="Guardar el resultado en este JSON.")
    args = parser.parse_args(argv)

    data_root = paths.resolve_data_root(args.data_root, cfg)
    pcfg = ap.preprocessing_modules().cfg
    audio_root = Path(args.audio_root).resolve() if args.audio_root else Path(pcfg.ROOT)
    report_path = Path(pcfg.R2_RESAMPLING)
    if not report_path.is_file():
        print(f"no existe el informe de la fase 2: {report_path}", file=sys.stderr)
        return 1
    phase2 = pd.read_csv(report_path, dtype={"audio_id": str})
    sources = phase2.loc[phase2["status"] == "OK"].set_index("audio_id")

    result = {"audio_root": str(audio_root), "phase2_report": str(report_path),
              "phase2_report_sha256": bmod.sha256_file(report_path), "datasets": {}}
    for entry in paths.dataset_entries(cfg, args.dataset):
        result["datasets"][entry["name"]] = check_dataset(entry["name"], data_root, audio_root, sources,
                                                          verify_hash=not args.no_hash)
    ok = all(d["status"] == "OK" for d in result["datasets"].values())
    result["status"] = "OK" if ok else "FALTA"
    if args.out is not None:
        paths.write_json_atomic(result, args.out)
    for name, d in result["datasets"].items():
        print(f"{name}: {d['status']} ({d['n_development_recordings']} grabaciones de desarrollo; "
              f"sin informe {len(d['missing_in_phase2_report'])}, sin archivo {len(d['missing_files'])}, "
              f"hash distinto {len(d['hash_mismatch'])})")
        for path in (d["missing_files"] + d["hash_mismatch"])[:5]:
            print(f"    {path}")
    print(json.dumps({"status": result["status"], "audio_root": result["audio_root"]}, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
