"""
Fase 1 - Verificacion de datos.

Antes de aplicar cualquier transformacion sobre las senales se comprueba que los
datos mantengan una relacion correcta con la metadata y que las grabaciones
cumplan condiciones minimas de calidad.

  1a  Integridad de los datos
  1b  Extraccion de anotaciones de los ciclos respiratorios
  1c  Calidad de senal

Esta fase no modifica ningun audio: solo produce informes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as cfg
import utils as u


SUBTYPE_BITS = {"PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32}
DURATION_TOLERANCE_S = 0.02


# ---------------------------------------------------------------------------
# 1a - Integridad de los datos
# ---------------------------------------------------------------------------

def step_1a_integrity(meta):
    """Contrasta la metadata con las propiedades reales de cada archivo.

    Verifica que no existan registros huerfanos en ninguna de las dos
    direcciones y que la duracion, la frecuencia de muestreo, el numero de
    canales y la profundidad de bits declarados coincidan con los del archivo.
    """
    u.section("FASE 1a - INTEGRIDAD DE LOS DATOS")

    rows = []
    progress = u.Progress(len(meta), "cabeceras", every=100)

    for _, r in meta.iterrows():
        path = Path(r["abs_path"])
        entry = {
            "dataset": r["dataset"],
            "audio_id": r["audio_id"],
            "audio_path": r["audio_path"],
            "file_exists": path.exists(),
            "sr_ok": False,
            "channels_ok": False,
            "duration_ok": False,
            "bit_depth_ok": False,
            "detail": "",
        }

        if path.exists():
            try:
                info = u.audio_info(path)
                problems = []

                entry["sr_ok"] = info.samplerate == int(r["sample_rate_hz"])
                if not entry["sr_ok"]:
                    problems.append(f"sr {info.samplerate} vs {r['sample_rate_hz']}")

                entry["channels_ok"] = info.channels == int(r["channels"])
                if not entry["channels_ok"]:
                    problems.append(f"canales {info.channels} vs {r['channels']}")

                delta = abs(info.duration - float(r["duration_seconds"]))
                entry["duration_ok"] = delta <= DURATION_TOLERANCE_S
                if not entry["duration_ok"]:
                    problems.append(f"duracion {info.duration:.3f} vs {r['duration_seconds']}")

                bits = SUBTYPE_BITS.get(info.subtype)
                entry["bit_depth_ok"] = bits == int(r["bit_depth"])
                if not entry["bit_depth_ok"]:
                    problems.append(f"bits {info.subtype} vs {r['bit_depth']}")

                entry["detail"] = "; ".join(problems)
            except Exception as exc:
                entry["detail"] = f"no legible: {exc}"
        else:
            entry["detail"] = "archivo ausente"

        rows.append(entry)
        progress.step()

    report = pd.DataFrame(rows)
    report["passes"] = (
        report["file_exists"] & report["sr_ok"] & report["channels_ok"]
        & report["duration_ok"] & report["bit_depth_ok"]
    )

    # Huerfanos: audio presente en disco que la metadata no declara
    orphans = []
    for name, root, _ in cfg.DATASETS:
        declared = {Path(p).resolve() for p in meta.loc[meta["dataset"] == name, "abs_path"]}
        on_disk = {p.resolve() for p in (root / "audio").rglob("*.wav")}
        for extra in sorted(on_disk - declared):
            orphans.append({"dataset": name, "audio_path": str(extra), "issue": "sin declarar"})

    print(f"\n  Archivos contrastados      : {len(report)}")
    print(f"  Ausentes en disco          : {(~report['file_exists']).sum()}")
    print(f"  Discrepancias con metadata : {(~report['passes']).sum()}")
    print(f"  Huerfanos sin declarar     : {len(orphans)}")

    failures = report.loc[~report["passes"]]
    if not failures.empty:
        print("\n  Primeras discrepancias:")
        for _, f in failures.head(10).iterrows():
            print(f"    {f['audio_id']}: {f['detail']}")

    out = cfg.REPORTS / "integrity.csv"
    report.to_csv(out, index=False)
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")

    return report


# ---------------------------------------------------------------------------
# 1b - Extraccion de anotaciones de los ciclos respiratorios
# ---------------------------------------------------------------------------

def parse_annotation(path):
    """Lee un archivo de anotacion y devuelve sus ciclos.

    Cada linea contiene inicio, fin, presencia de crepitantes y presencia de
    sibilancias, separados por tabulador y sin encabezado.
    """
    cycles = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 4:
                raise ValueError(f"{path.name}: se esperaban 4 columnas, hay {len(parts)}")
            start, end = float(parts[0]), float(parts[1])
            crackles, wheezes = int(parts[2]), int(parts[3])
            cycles.append((start, end, crackles, wheezes))
    return cycles


def event_label(crackles, wheezes):
    """Categoria del ciclo a partir de sus dos banderas."""
    if crackles and wheezes:
        return "both"
    if crackles:
        return "crackles"
    if wheezes:
        return "wheezes"
    return "normal"


def step_1b_annotations(meta):
    """Reconstruye la tabla de ciclos respiratorios desde los archivos de anotacion.

    Solo ICBHI dispone de anotaciones temporales. La tabla se regenera para que
    el pipeline sea reproducible desde cero, y se contrasta con la version ya
    presente en el repositorio sin sobrescribirla.
    """
    u.section("FASE 1b - EXTRACCION DE ANOTACIONES")

    icbhi = meta.loc[meta["dataset"] == "ICBHI"]
    rows = []
    anomalies = []
    progress = u.Progress(len(icbhi), "anotaciones", every=100)

    for _, r in icbhi.iterrows():
        path = u.annotation_path(r)
        if path is None or not path.exists():
            anomalies.append(f"{r['audio_id']}: anotacion ausente")
            progress.step()
            continue

        try:
            cycles = parse_annotation(path)
        except ValueError as exc:
            anomalies.append(str(exc))
            progress.step()
            continue

        if not cycles:
            anomalies.append(f"{r['audio_id']}: sin ciclos")

        duration = float(r["duration_seconds"])
        previous_end = None
        for idx, (start, end, crackles, wheezes) in enumerate(cycles, start=1):
            if end <= start:
                anomalies.append(f"{r['audio_id']} ciclo {idx}: duracion no positiva")
            if end > duration + 0.05:
                anomalies.append(f"{r['audio_id']} ciclo {idx}: excede la duracion del audio")
            if previous_end is not None and start < previous_end - 1e-6:
                anomalies.append(f"{r['audio_id']} ciclo {idx}: solapa con el anterior")
            previous_end = end

            rows.append({
                "dataset": "ICBHI",
                "audio_id": r["audio_id"],
                "patient_uid": r["patient_uid"],
                "diagnosis": r["diagnosis"],
                "device": r["device"],
                "zone": r["zone"],
                "cycle_idx": idx,
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(end - start, 3),
                "crackles": crackles,
                "wheezes": wheezes,
                "label": event_label(crackles, wheezes),
            })
        progress.step()

    cycles_df = pd.DataFrame(rows)
    counts = cycles_df["label"].value_counts()

    print(f"\n  Grabaciones con anotacion : {cycles_df['audio_id'].nunique()}")
    print(f"  Ciclos respiratorios      : {len(cycles_df)}")
    for label in ("normal", "crackles", "wheezes", "both"):
        print(f"    {label:<10} {counts.get(label, 0):>6}")
    print(f"  Anomalias detectadas      : {len(anomalies)}")
    for a in anomalies[:10]:
        print(f"    {a}")

    # Contraste con la tabla ya presente en el repositorio
    if cfg.ICBHI_CYCLES.exists():
        committed = pd.read_csv(cfg.ICBHI_CYCLES)
        same_rows = len(committed) == len(cycles_df)
        same_labels = (
            committed["label"].value_counts().to_dict()
            == cycles_df["label"].value_counts().to_dict()
        )
        verdict = "coincide" if (same_rows and same_labels) else "DIFIERE"
        print(f"\n  Contraste con icbhi_respiratory_cycles.csv: {verdict}"
              f"  ({len(committed)} filas en el repositorio)")

    return cycles_df, anomalies


# ---------------------------------------------------------------------------
# 1c - Calidad de senal
# ---------------------------------------------------------------------------

def step_1c_signal_quality(meta):
    """Evalua si cada archivo contiene una senal respiratoria utilizable.

    Calcula saturacion, RMS y varianza, y la relacion senal-ruido estimada por
    percentiles. No excluye nada mientras los umbrales de config no esten
    fijados: la primera ejecucion sirve para observar las distribuciones.
    """
    u.section("FASE 1c - CALIDAD DE SENAL")

    rows = []
    progress = u.Progress(len(meta), "audios", every=50)

    for _, r in meta.iterrows():
        path = Path(r["abs_path"])
        entry = {
            "dataset": r["dataset"],
            "audio_id": r["audio_id"],
            "patient_uid": r["patient_uid"],
            "diagnosis": r["diagnosis"],
            "device": r["device"],
            "zone": r["zone"],
            "filter": r["filter"],
            "sample_rate_hz": int(r["sample_rate_hz"]),
            "bit_depth": int(r["bit_depth"]),
            "duration_s": float(r["duration_seconds"]),
        }

        try:
            x, sr = u.read_audio(path)
            sat_pct, sat_runs = u.saturation_stats(x)
            entry.update({
                "rms": u.rms(x),
                "variance": float(np.var(x)),
                "dc_offset": u.dc_offset(x),
                "peak": float(np.max(np.abs(x))) if x.size else 0.0,
                "saturation_pct": sat_pct,
                "saturation_runs": sat_runs,
                "snr_db": u.estimate_snr_db(x, sr),
                "n_frames": int(u.frame_energy(x, sr).size),
                "readable": True,
            })
        except Exception as exc:
            entry.update({
                "rms": np.nan, "variance": np.nan, "dc_offset": np.nan,
                "peak": np.nan, "saturation_pct": np.nan, "saturation_runs": 0,
                "snr_db": np.nan, "n_frames": 0, "readable": False,
                "error": str(exc),
            })

        rows.append(entry)
        progress.step()

    quality = pd.DataFrame(rows)

    print("\n  Distribuciones sobre el conjunto completo:\n")
    u.describe(quality["saturation_pct"], "saturacion", " %", "{:8.4f}")
    u.describe(quality["rms"], "RMS", "", "{:8.5f}")
    u.describe(quality["peak"], "pico", "", "{:8.5f}")
    u.describe(quality["dc_offset"], "componente continua", "", "{:8.5f}")
    u.describe(quality["snr_db"], "SNR", " dB", "{:8.2f}")

    # El mismo modelo de estetoscopio aparece en ambos corpus, de modo que la
    # agrupacion debe distinguir tambien el conjunto de origen: son campanas de
    # adquisicion distintas aunque el instrumento coincida.
    quality["group"] = quality["dataset"] + " / " + quality["device"]

    print("\n  Por conjunto y dispositivo:\n")
    print(f"  {'grupo':<24}{'n':>5}{'satur.med':>11}{'satur.max':>11}"
          f"{'RMS med':>10}{'SNR med':>9}{'SNR p10':>9}")
    for group_name, group in quality.groupby("group"):
        snr = group["snr_db"].replace([np.inf, -np.inf], np.nan).dropna()
        print(f"  {group_name:<24}{len(group):>5}"
              f"{group['saturation_pct'].mean():>11.4f}"
              f"{group['saturation_pct'].max():>11.4f}"
              f"{group['rms'].median():>10.5f}"
              f"{snr.median() if not snr.empty else float('nan'):>9.2f}"
              f"{snr.quantile(0.10) if not snr.empty else float('nan'):>9.2f}")

    report_saturation_by_diagnosis(quality)
    report_threshold_candidates(quality)

    # Candidatos segun los umbrales vigentes
    flags = evaluate_thresholds(quality)
    quality = quality.join(flags)

    out = cfg.REPORTS / "signal_quality.csv"
    quality.to_csv(out, index=False)
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")

    return quality


def report_saturation_by_diagnosis(quality):
    """Cruce entre saturacion y diagnostico, restringido a ICBHI.

    La saturacion no es un defecto neutro: al recortar los picos genera
    armonicos que no existian en la senal original. Si se concentra en una
    clase, esos armonicos se convierten en un rasgo espurio asociado a ella.
    """
    icbhi = quality.loc[quality["dataset"] == "ICBHI"]
    if icbhi.empty:
        return

    print("\n  Saturacion por diagnostico (ICBHI):\n")
    print(f"  {'diagnostico':<18}{'n':>5}{'media':>10}{'max':>10}{'>1%':>7}{'>5%':>7}")
    for dx, group in icbhi.groupby("diagnosis"):
        print(f"  {dx:<18}{len(group):>5}{group['saturation_pct'].mean():>10.3f}"
              f"{group['saturation_pct'].max():>10.3f}"
              f"{(group['saturation_pct'] > 1).sum():>7}"
              f"{(group['saturation_pct'] > 5).sum():>7}")

    affected = icbhi.loc[icbhi["saturation_pct"] > 1]
    if not affected.empty:
        classes = affected["diagnosis"].unique()
        print(f"\n  De las {len(affected)} grabaciones con mas del 1 % de saturacion,"
              f" {len(classes)} clase(s) distinta(s): {', '.join(sorted(classes))}")


def report_threshold_candidates(quality):
    """Efecto de distintos umbrales, para fijarlos observando la distribucion."""
    print("\n  Efecto de umbrales candidatos de saturacion:\n")
    print(f"  {'umbral':>9}{'excluidos':>11}{'% corpus':>10}   reparto por clase (ICBHI)")
    icbhi = quality.loc[quality["dataset"] == "ICBHI"]
    for threshold in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        excluded = quality.loc[quality["saturation_pct"] > threshold]
        by_class = icbhi.loc[icbhi["saturation_pct"] > threshold, "diagnosis"].value_counts()
        detail = ", ".join(f"{k}:{v}" for k, v in by_class.items()) or "-"
        print(f"  {threshold:>8.1f}%{len(excluded):>11}"
              f"{100 * len(excluded) / len(quality):>9.1f}%   {detail}")

    print("\n  Efecto de umbrales candidatos de SNR:\n")
    print(f"  {'umbral':>9}{'excluidos':>11}{'% corpus':>10}   reparto por grupo")
    for threshold in (3.0, 4.0, 5.0, 6.0, 7.0):
        excluded = quality.loc[quality["snr_db"] < threshold]
        by_group = excluded["group"].value_counts()
        detail = ", ".join(f"{k.split(' / ')[1]}:{v}" for k, v in by_group.items()) or "-"
        print(f"  {threshold:>8.1f} dB{len(excluded):>10}"
              f"{100 * len(excluded) / len(quality):>9.1f}%   {detail}")


def evaluate_thresholds(quality):
    """Marca cada grabacion segun los umbrales vigentes en config.

    Un umbral en None significa que aun no se ha fijado, y en ese caso el
    criterio correspondiente no excluye nada.
    """
    n = len(quality)
    flags = pd.DataFrame(index=quality.index)

    flags["fail_saturation"] = (
        quality["saturation_pct"] > cfg.MAX_SATURATION_PCT
        if cfg.MAX_SATURATION_PCT is not None else pd.Series(False, index=quality.index)
    )
    flags["fail_rms"] = (
        quality["rms"] < cfg.MIN_RMS
        if cfg.MIN_RMS is not None else pd.Series(False, index=quality.index)
    )
    flags["fail_variance"] = quality["variance"] <= cfg.MIN_VARIANCE
    flags["fail_snr"] = (
        quality["snr_db"] < cfg.MIN_SNR_DB
        if cfg.MIN_SNR_DB is not None else pd.Series(False, index=quality.index)
    )
    flags["fail_unreadable"] = ~quality["readable"]
    flags["excluded"] = flags.any(axis=1)
    return flags


def write_exclusions(quality):
    """Registro de las grabaciones excluidas, con el criterio incumplido."""
    criteria = {
        "fail_unreadable": "archivo no legible",
        "fail_variance": "varianza nula (archivo plano)",
        "fail_saturation": "saturacion por encima del umbral",
        "fail_rms": "RMS por debajo del minimo",
        "fail_snr": "SNR por debajo del minimo",
    }

    rows = []
    for _, r in quality.loc[quality["excluded"]].iterrows():
        reasons = [text for flag, text in criteria.items() if r.get(flag)]
        rows.append({
            "dataset": r["dataset"],
            "audio_id": r["audio_id"],
            "device": r["device"],
            "criterio": "; ".join(reasons),
            "saturation_pct": r["saturation_pct"],
            "rms": r["rms"],
            "variance": r["variance"],
            "snr_db": r["snr_db"],
        })

    columns = ["dataset", "audio_id", "device", "criterio",
               "saturation_pct", "rms", "variance", "snr_db"]
    exclusions = pd.DataFrame(rows, columns=columns)

    out = cfg.REPORTS / "exclusions.csv"
    exclusions.to_csv(out, index=False)

    admitted = len(quality) - len(exclusions)
    print(f"\n  Admitidos : {admitted}")
    print(f"  Excluidos : {len(exclusions)}")
    if not exclusions.empty:
        print("\n  Por criterio:")
        for criterio, group in exclusions.groupby("criterio"):
            print(f"    {criterio:<44} {len(group):>4}")
        print("\n  Por dispositivo:")
        for device, group in exclusions.groupby("device"):
            print(f"    {device:<44} {len(group):>4}")
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")

    return exclusions


# ---------------------------------------------------------------------------

def main():
    cfg.ensure_dirs()

    u.section("FASE 1 - VERIFICACION DE DATOS")
    meta = u.load_metadata()
    print(f"  Registros en la metadata: {len(meta)}")
    for name, group in meta.groupby("dataset"):
        print(f"    {name:<10} {len(group):>5} audios"
              f"  ·  {group['patient_uid'].nunique():>4} pacientes")

    pending = cfg.pending_parameters()
    if pending:
        print(f"\n  Parametros sin fijar ({len(pending)}): {', '.join(pending)}")
        print("  Los umbrales de admision en None no excluyen ninguna grabacion.")

    integrity = step_1a_integrity(meta)
    cycles, anomalies = step_1b_annotations(meta)
    quality = step_1c_signal_quality(meta)
    exclusions = write_exclusions(quality)

    u.section("RESUMEN DE LA FASE 1")
    print(f"  Integridad    : {int(integrity['passes'].sum())}/{len(integrity)} sin discrepancias")
    print(f"  Anotaciones   : {len(cycles)} ciclos, {len(anomalies)} anomalias")
    print(f"  Calidad       : {int(quality['readable'].sum())}/{len(quality)} legibles")
    print(f"  Admitidos     : {len(quality) - len(exclusions)}")
    print(f"  Excluidos     : {len(exclusions)}")

    return {"integrity": integrity, "cycles": cycles,
            "quality": quality, "exclusions": exclusions}


if __name__ == "__main__":
    main()
