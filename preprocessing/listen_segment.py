"""Exporta y, opcionalmente, reproduce segmentos generados por la fase 4.

Ejemplos
--------
Listar segmentos disponibles::

    python preprocessing/listen_segment.py --list 10

Exportar el mismo segmento en las dos ramas::

    python preprocessing/listen_segment.py --index 0 --branch both

Exportar y abrir un segmento con el reproductor predeterminado de Windows::

    python preprocessing/listen_segment.py --index 0 --branch no_dn --play

Los WAV creados son copias para escucha en PCM de 16 bits. Los arrays float32
de la fase 4 no se modifican y siguen siendo la entrada destinada al modelo.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg
import utils as u


DEFAULT_OUTPUT = cfg.DATA / "listening"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Escucha o exporta un segmento de la fase 4."
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--index", type=int, help="Valor de array_index en segments.csv (por defecto: 0)."
    )
    selection.add_argument(
        "--segment-id", help="Identificador exacto de la columna segment_id."
    )
    parser.add_argument(
        "--branch", choices=("no_dn", "dn", "both"), default="both",
        help="Rama que se exportará (por defecto: both).",
    )
    parser.add_argument(
        "--list", dest="list_count", type=int, metavar="N",
        help="Muestra los primeros N segmentos y termina sin exportar.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT,
        help=f"Carpeta de salida (por defecto: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Abre el WAV exportado con el reproductor predeterminado de Windows.",
    )
    return parser.parse_args()


def select_row(metadata, array_index, segment_id):
    if segment_id is not None:
        matches = metadata.loc[metadata["segment_id"] == segment_id]
        if matches.empty:
            raise ValueError(f"No existe segment_id={segment_id!r}.")
        return matches.iloc[0]

    requested = 0 if array_index is None else array_index
    matches = metadata.loc[metadata["array_index"] == requested]
    if matches.empty:
        minimum = int(metadata["array_index"].min())
        maximum = int(metadata["array_index"].max())
        raise ValueError(
            f"No existe array_index={requested}. El intervalo disponible es {minimum}-{maximum}."
        )
    return matches.iloc[0]


def main():
    args = parse_args()
    metadata_path = cfg.FINAL / "segments.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No existe {metadata_path}. Ejecute primero phase4_temporal.py."
        )

    metadata = pd.read_csv(metadata_path, dtype={"audio_id": str, "patient_uid": str})

    if args.list_count is not None:
        if args.list_count < 1:
            raise ValueError("--list debe ser mayor que cero.")
        columns = [
            "array_index", "segment_id", "dataset", "patient_uid",
            "diagnosis", "device", "zone", "start_s", "end_s",
        ]
        print(metadata[columns].head(args.list_count).to_string(index=False))
        return

    row = select_row(metadata, args.index, args.segment_id)
    array_index = int(row["array_index"])
    branches = ("no_dn", "dn") if args.branch == "both" else (args.branch,)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("SEGMENTO SELECCIONADO")
    print(f"  array_index : {array_index}")
    print(f"  segment_id  : {row['segment_id']}")
    print(f"  dataset     : {row['dataset']}")
    print(f"  paciente    : {row['patient_uid']}")
    print(f"  diagnóstico : {row['diagnosis']}")
    print(f"  dispositivo : {row['device']}")
    print(f"  zona        : {row['zone']}")
    print(f"  intervalo   : {row['start_s']:.2f}-{row['end_s']:.2f} s")

    exported = []
    for branch in branches:
        array_path = cfg.FINAL / f"segments_{branch}.npy"
        if not array_path.exists():
            raise FileNotFoundError(f"No existe {array_path}.")
        segments = np.load(array_path, mmap_mode="r")
        if array_index >= segments.shape[0]:
            raise IndexError(
                f"array_index={array_index} excede las {segments.shape[0]} filas de {array_path.name}."
            )

        audio = np.asarray(segments[array_index], dtype=np.float32)
        output = args.output_dir / f"{row['segment_id']}_{branch}.wav"
        u.write_atomic(output, audio, cfg.TARGET_SR, subtype="PCM_16")
        exported.append(output.resolve())
        print(f"  WAV {branch:<5}: {output.resolve()}")

    if args.play:
        if os.name != "nt":
            raise RuntimeError("--play está implementado para Windows.")
        for output in exported:
            os.startfile(output)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
