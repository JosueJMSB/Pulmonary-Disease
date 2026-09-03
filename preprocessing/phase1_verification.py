"""
Fase 1 - Verificacion de datos.

Antes de aplicar cualquier transformacion sobre las senales se comprueba que los
datos mantengan una relacion correcta con la metadata y que las grabaciones
cumplan condiciones minimas de calidad.

  1a  Integridad de los datos
  1b  Validacion y regeneracion de anotaciones respiratorias
  1c  Calidad objetiva de senal
  1d  Duplicados binarios
  1e  Validacion contra las fuentes de metadata

El resultado contractual es phase1_manifest.csv. Distingue la calidad
acustica (PASS/REVIEW/EXCLUDE) de la elegibilidad para modelado, para no
confundir una incidencia de procedencia o duplicacion con una senal defectuosa.

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
ANNOTATION_TOLERANCE_S = 0.05


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
            "row_type": "metadata",
            "file_exists": path.is_file(),
            "readable_header": False,
            "actual_sr": np.nan,
            "declared_sr": int(r["sample_rate_hz"]),
            "actual_channels": np.nan,
            "declared_channels": int(r["channels"]),
            "actual_duration_s": np.nan,
            "declared_duration_s": float(r["duration_seconds"]),
            "duration_delta_s": np.nan,
            "actual_subtype": "",
            "declared_bit_depth": int(r["bit_depth"]),
            "frames": 0,
            "file_size_bytes": path.stat().st_size if path.is_file() else 0,
            "sr_ok": False,
            "channels_ok": False,
            "duration_ok": False,
            "bit_depth_ok": False,
            "detail": "",
        }

        if path.is_file():
            try:
                info = u.audio_info(path)
                problems = []
                entry.update({
                    "readable_header": True,
                    "actual_sr": info.samplerate,
                    "actual_channels": info.channels,
                    "actual_duration_s": info.duration,
                    "actual_subtype": info.subtype,
                    "frames": info.frames,
                })

                entry["sr_ok"] = info.samplerate == int(r["sample_rate_hz"])
                if not entry["sr_ok"]:
                    problems.append(f"sr {info.samplerate} vs {r['sample_rate_hz']}")

                entry["channels_ok"] = info.channels == int(r["channels"])
                if not entry["channels_ok"]:
                    problems.append(f"canales {info.channels} vs {r['channels']}")

                delta = abs(info.duration - float(r["duration_seconds"]))
                entry["duration_delta_s"] = delta
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
        report["file_exists"] & report["readable_header"]
        & report["sr_ok"] & report["channels_ok"]
        & report["duration_ok"] & report["bit_depth_ok"]
    )

    # Huerfanos: audio presente en disco que la metadata no declara
    orphans = []
    for name, root, _ in cfg.DATASETS:
        declared = {Path(p).resolve() for p in meta.loc[meta["dataset"] == name, "abs_path"]}
        on_disk = {p.resolve() for p in (root / "audio").rglob("*.wav")}
        for extra in sorted(on_disk - declared):
            orphans.append({
                "dataset": name,
                "audio_id": "",
                "audio_path": str(extra.relative_to(root)),
                "row_type": "orphan_audio",
                "file_exists": True,
                "readable_header": False,
                "actual_sr": np.nan,
                "declared_sr": np.nan,
                "actual_channels": np.nan,
                "declared_channels": np.nan,
                "actual_duration_s": np.nan,
                "declared_duration_s": np.nan,
                "duration_delta_s": np.nan,
                "actual_subtype": "",
                "declared_bit_depth": np.nan,
                "frames": 0,
                "file_size_bytes": extra.stat().st_size,
                "sr_ok": False,
                "channels_ok": False,
                "duration_ok": False,
                "bit_depth_ok": False,
                "detail": "audio presente en disco y ausente de la metadata",
                "passes": False,
            })

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
    complete_report = pd.concat(
        [report, pd.DataFrame(orphans, columns=report.columns)], ignore_index=True
    )
    complete_report.to_csv(out, index=False)
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")

    return complete_report


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
    """Valida las anotaciones ICBHI y regenera sus dos tablas derivadas.

    Las fuentes .txt y los CSV originales nunca se sobrescriben. Los resultados
    regenerados se guardan en reports/ y se comparan por contenido, no solo por
    numero de filas o distribucion de etiquetas.
    """
    u.section("FASE 1b - EXTRACCION DE ANOTACIONES")

    icbhi = meta.loc[meta["dataset"] == "ICBHI"]
    rows = []
    summaries = []
    issues = []
    valid_source_cycles = {}
    progress = u.Progress(len(icbhi), "anotaciones", every=100)

    for _, r in icbhi.iterrows():
        path = u.annotation_path(r)
        if path is None or not path.exists():
            issues.append({
                "audio_id": r["audio_id"], "cycle_idx": "",
                "source": "annotation_txt", "issue": "MISSING_ANNOTATION",
                "detail": str(path),
            })
            progress.step()
            continue

        try:
            cycles = parse_annotation(path)
        except (OSError, ValueError) as exc:
            issues.append({
                "audio_id": r["audio_id"], "cycle_idx": "",
                "source": "annotation_txt", "issue": "UNREADABLE_ANNOTATION",
                "detail": str(exc),
            })
            progress.step()
            continue

        if not cycles:
            issues.append({
                "audio_id": r["audio_id"], "cycle_idx": "",
                "source": "annotation_txt", "issue": "EMPTY_ANNOTATION",
                "detail": str(path),
            })

        duration = float(r["duration_seconds"])
        previous_end = None
        valid_cycles = []
        for idx, (start, end, crackles, wheezes) in enumerate(cycles, start=1):
            cycle_issues = []
            if not np.isfinite([start, end]).all():
                cycle_issues.append("NON_FINITE_TIME")
            if start < 0:
                cycle_issues.append("NEGATIVE_START")
            if end <= start:
                cycle_issues.append("NON_POSITIVE_DURATION")
            if end > duration + ANNOTATION_TOLERANCE_S:
                cycle_issues.append("END_AFTER_AUDIO")
            if previous_end is not None and start < previous_end - 1e-6:
                cycle_issues.append("OVERLAP_WITH_PREVIOUS")
            if crackles not in (0, 1) or wheezes not in (0, 1):
                cycle_issues.append("INVALID_EVENT_FLAG")
            previous_end = end

            for issue in cycle_issues:
                issues.append({
                    "audio_id": r["audio_id"], "cycle_idx": idx,
                    "source": "annotation_txt", "issue": issue,
                    "detail": f"start={start}; end={end}; crackles={crackles}; wheezes={wheezes}",
                })
            if cycle_issues:
                continue
            valid_cycles.append((start, end, crackles, wheezes))

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
        valid_source_cycles[r["audio_id"]] = valid_cycles
        progress.step()

    cycle_columns = [
        "dataset", "audio_id", "patient_uid", "diagnosis", "device", "zone",
        "cycle_idx", "start_s", "end_s", "duration_s", "crackles", "wheezes", "label",
    ]
    cycles_df = pd.DataFrame(rows, columns=cycle_columns)

    for _, r in icbhi.iterrows():
        group = cycles_df.loc[cycles_df["audio_id"] == r["audio_id"]]
        raw_cycles = valid_source_cycles.get(r["audio_id"], [])
        exact_durations = np.array([end - start for start, end, _, _ in raw_cycles])
        counts = group["label"].value_counts()
        n_cycles = len(group)
        annotated_s = float(exact_durations.sum()) if n_cycles else 0.0
        summaries.append({
            "dataset": "ICBHI",
            "audio_id": r["audio_id"],
            "patient_uid": r["patient_uid"],
            "diagnosis": r["diagnosis"],
            "device": r["device"],
            "zone": r["zone"],
            "duration_seconds": round(float(r["duration_seconds"]), 3),
            "n_cycles": n_cycles,
            "n_normal": int(counts.get("normal", 0)),
            "n_crackles_only": int(counts.get("crackles", 0)),
            "n_wheezes_only": int(counts.get("wheezes", 0)),
            "n_both": int(counts.get("both", 0)),
            "n_with_crackles": int(group["crackles"].sum()) if n_cycles else 0,
            "n_with_wheezes": int(group["wheezes"].sum()) if n_cycles else 0,
            "pct_with_crackles": round(100 * group["crackles"].sum() / n_cycles, 2) if n_cycles else 0.0,
            "pct_with_wheezes": round(100 * group["wheezes"].sum() / n_cycles, 2) if n_cycles else 0.0,
            "mean_cycle_s": round(float(exact_durations.mean()), 3) if n_cycles else np.nan,
            "annotated_s": round(annotated_s, 3),
            "coverage_pct": round(100 * annotated_s / float(r["duration_seconds"]), 2),
            "annotation_path": path_relative(u.annotation_path(r), cfg.ICBHI_DIR),
        })

    summary_df = pd.DataFrame(summaries)
    counts = cycles_df["label"].value_counts()

    print(f"\n  Grabaciones con anotacion : {cycles_df['audio_id'].nunique()}")
    print(f"  Ciclos respiratorios      : {len(cycles_df)}")
    for label in ("normal", "crackles", "wheezes", "both"):
        print(f"    {label:<10} {counts.get(label, 0):>6}")
    source_issue_count = len(issues)

    # Contraste completo con la tabla presente en el repositorio.
    if cfg.ICBHI_CYCLES.exists():
        committed = pd.read_csv(cfg.ICBHI_CYCLES)
        compare_cycle_tables(cycles_df, committed, issues)
    else:
        issues.append({
            "audio_id": "", "cycle_idx": "", "source": "committed_cycles",
            "issue": "MISSING_COMMITTED_TABLE", "detail": str(cfg.ICBHI_CYCLES),
        })
    if cfg.ICBHI_CYCLE_SUMMARY.exists():
        committed_summary = pd.read_csv(cfg.ICBHI_CYCLE_SUMMARY)
        compare_summary_tables(summary_df, committed_summary, issues)
    else:
        issues.append({
            "audio_id": "", "cycle_idx": "", "source": "committed_summary",
            "issue": "MISSING_COMMITTED_TABLE", "detail": str(cfg.ICBHI_CYCLE_SUMMARY),
        })

    out_cycles = cfg.REPORTS / "icbhi_respiratory_cycles_regenerated.csv"
    out_summary = cfg.REPORTS / "icbhi_cycle_summary_regenerated.csv"
    out_validation = cfg.REPORTS / "annotation_validation.csv"
    cycles_df.to_csv(out_cycles, index=False)
    summary_df.to_csv(out_summary, index=False)
    validation = pd.DataFrame(
        issues, columns=["audio_id", "cycle_idx", "source", "issue", "detail"]
    )
    validation.to_csv(out_validation, index=False)

    print(f"  Anomalias en fuentes      : {source_issue_count}")
    print(f"  Incidencias totales       : {len(validation)}")
    for _, issue in validation.head(10).iterrows():
        print(f"    {issue['audio_id']} {issue['cycle_idx']}: {issue['issue']}")
    verdict = "coincide" if validation.empty else "REVISAR"
    print(f"\n  Contraste con icbhi_respiratory_cycles.csv: {verdict}")
    print(f"  -> {out_cycles.relative_to(cfg.ROOT)}")
    print(f"  -> {out_summary.relative_to(cfg.ROOT)}")
    print(f"  -> {out_validation.relative_to(cfg.ROOT)}")

    return cycles_df, summary_df, validation


def path_relative(path, root):
    """Ruta POSIX relativa a root, estable entre sistemas operativos."""
    return Path(path).relative_to(root).as_posix()


def compare_cycle_tables(regenerated, committed, issues):
    """Registra diferencias de contenido entre ciclos regenerados y guardados."""
    required = set(regenerated.columns)
    missing = sorted(required - set(committed.columns))
    if missing:
        issues.append({
            "audio_id": "", "cycle_idx": "", "source": "committed_cycles",
            "issue": "MISSING_COLUMNS", "detail": ", ".join(missing),
        })
        return

    keys = ["audio_id", "cycle_idx"]
    for table_name, table in (("regenerated", regenerated), ("committed", committed)):
        duplicates = table.loc[table.duplicated(keys, keep=False), keys]
        for _, duplicate in duplicates.drop_duplicates().iterrows():
            issues.append({
                "audio_id": duplicate["audio_id"], "cycle_idx": duplicate["cycle_idx"],
                "source": table_name, "issue": "DUPLICATE_CYCLE_KEY", "detail": "",
            })

    merged = regenerated.merge(
        committed[list(regenerated.columns)], on=keys, how="outer",
        suffixes=("_new", "_stored"), indicator=True,
    )
    for _, row in merged.loc[merged["_merge"] != "both"].iterrows():
        issues.append({
            "audio_id": row["audio_id"], "cycle_idx": row["cycle_idx"],
            "source": "committed_cycles", "issue": "ROW_" + row["_merge"].upper(),
            "detail": "left_only=solo regenerado; right_only=solo almacenado",
        })

    both = merged.loc[merged["_merge"] == "both"]
    numeric = {"start_s", "end_s", "duration_s"}
    for column in (required - set(keys)):
        left = both[f"{column}_new"]
        right = both[f"{column}_stored"]
        if column in numeric:
            different = ~np.isclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                atol=0.001, rtol=0, equal_nan=True,
            )
        else:
            different = left.astype(str) != right.astype(str)
        for idx in both.index[different]:
            row = both.loc[idx]
            issues.append({
                "audio_id": row["audio_id"], "cycle_idx": row["cycle_idx"],
                "source": "committed_cycles", "issue": "VALUE_MISMATCH",
                "detail": f"{column}: {row[f'{column}_new']} vs {row[f'{column}_stored']}",
            })


def compare_summary_tables(regenerated, committed, issues):
    """Registra diferencias de contenido en el resumen por grabacion."""
    required = list(regenerated.columns)
    missing = sorted(set(required) - set(committed.columns))
    if missing:
        issues.append({
            "audio_id": "", "cycle_idx": "", "source": "committed_summary",
            "issue": "MISSING_COLUMNS", "detail": ", ".join(missing),
        })
        return
    merged = regenerated.merge(
        committed[required], on="audio_id", how="outer",
        suffixes=("_new", "_stored"), indicator=True,
    )
    for _, row in merged.loc[merged["_merge"] != "both"].iterrows():
        issues.append({
            "audio_id": row["audio_id"], "cycle_idx": "",
            "source": "committed_summary", "issue": "ROW_" + row["_merge"].upper(),
            "detail": "left_only=solo regenerado; right_only=solo almacenado",
        })
    both = merged.loc[merged["_merge"] == "both"]
    numeric_columns = set(regenerated.select_dtypes(include=np.number).columns) - {"audio_id"}
    count_columns = {
        "n_cycles", "n_normal", "n_crackles_only", "n_wheezes_only",
        "n_both", "n_with_crackles", "n_with_wheezes",
    }
    percentage_columns = {"pct_with_crackles", "pct_with_wheezes", "coverage_pct"}
    for column in (set(required) - {"audio_id"}):
        left = both[f"{column}_new"]
        right = both[f"{column}_stored"]
        if column in numeric_columns:
            if column in count_columns:
                tolerance = 0.0
            elif column in percentage_columns:
                tolerance = 0.011
            else:
                tolerance = 0.0011
            different = ~np.isclose(
                pd.to_numeric(left, errors="coerce"),
                pd.to_numeric(right, errors="coerce"),
                atol=tolerance, rtol=0, equal_nan=True,
            )
        else:
            different = left.astype(str) != right.astype(str)
        for idx in both.index[different]:
            row = both.loc[idx]
            issues.append({
                "audio_id": row["audio_id"], "cycle_idx": "",
                "source": "committed_summary", "issue": "VALUE_MISMATCH",
                "detail": f"{column}: {row[f'{column}_new']} vs {row[f'{column}_stored']}",
            })


# ---------------------------------------------------------------------------
# 1c - Calidad de senal
# ---------------------------------------------------------------------------

def step_1c_signal_quality(meta):
    """Evalua si cada archivo contiene una senal respiratoria utilizable.

    Calcula indicadores objetivos sin alterar las muestras. El proxy de SNR se
    mide sobre una copia temporal limitada a 50-1800 Hz para que sea comparable
    entre las frecuencias de muestreo originales.
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
            "file_sha256": "",
            "error": "",
        }

        try:
            entry["file_sha256"] = u.file_sha256(path)
            x, sr = u.read_audio(path)
            sat_pct, sat_runs = u.saturation_stats(x, sr)
            silence_pct, silence_longest = u.digital_silence_stats(x)
            snr_value, snr_status = u.snr_proxy_db(x, sr)
            finite = bool(np.isfinite(x).all())
            entry.update({
                "rms": u.rms(x),
                "rms_without_dc": u.rms_without_dc(x),
                "variance": float(np.var(x)),
                "dc_offset": u.dc_offset(x),
                "peak": float(np.max(np.abs(x))) if x.size else 0.0,
                "saturation_pct": sat_pct,
                "saturation_runs": sat_runs,
                "digital_silence_pct": silence_pct,
                "digital_silence_longest_samples": silence_longest,
                "digital_silence_longest_s": silence_longest / sr,
                "snr_proxy_db": snr_value,
                "snr_status": snr_status,
                "n_frames": int(u.frame_energy(x, sr).size),
                "n_samples": int(x.size),
                "has_non_finite": not finite,
                "readable": True,
            })
        except Exception as exc:
            entry.update({
                "rms": np.nan, "rms_without_dc": np.nan,
                "variance": np.nan, "dc_offset": np.nan,
                "peak": np.nan, "saturation_pct": np.nan, "saturation_runs": 0,
                "digital_silence_pct": np.nan,
                "digital_silence_longest_samples": 0,
                "digital_silence_longest_s": np.nan,
                "snr_proxy_db": np.nan, "snr_status": "READ_ERROR",
                "n_frames": 0, "n_samples": 0, "has_non_finite": False,
                "readable": False,
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
    u.describe(quality["digital_silence_pct"], "silencio digital", " %", "{:8.3f}")
    u.describe(quality["snr_proxy_db"], "SNR proxy 50-1800 Hz", " dB", "{:8.2f}")

    # El mismo modelo de estetoscopio aparece en ambos corpus, de modo que la
    # agrupacion debe distinguir tambien el conjunto de origen: son campanas de
    # adquisicion distintas aunque el instrumento coincida.
    filter_name = quality["filter"].replace("", "sin filtro")
    quality["group"] = (
        quality["dataset"] + " / " + quality["device"] + " / " + filter_name
    )

    print("\n  Por conjunto y dispositivo:\n")
    print(f"  {'grupo':<24}{'n':>5}{'satur.med':>11}{'satur.max':>11}"
          f"{'RMS med':>10}{'SNR med':>9}{'SNR p10':>9}")
    for group_name, group in quality.groupby("group"):
        snr = group["snr_proxy_db"].dropna()
        print(f"  {group_name:<24}{len(group):>5}"
              f"{group['saturation_pct'].mean():>11.4f}"
              f"{group['saturation_pct'].max():>11.4f}"
              f"{group['rms'].median():>10.5f}"
              f"{snr.median() if not snr.empty else float('nan'):>9.2f}"
              f"{snr.quantile(0.10) if not snr.empty else float('nan'):>9.2f}")

    report_saturation_by_diagnosis(quality)
    report_threshold_candidates(quality)

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
    """Efecto de distintos umbrales como candidatos de revision manual."""
    print("\n  Efecto de umbrales candidatos de saturacion:\n")
    print(f"  {'umbral':>9}{'marcados':>11}{'% corpus':>10}   reparto por clase (ICBHI)")
    icbhi = quality.loc[quality["dataset"] == "ICBHI"]
    for threshold in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
        selected = quality.loc[quality["saturation_pct"] > threshold]
        by_class = icbhi.loc[icbhi["saturation_pct"] > threshold, "diagnosis"].value_counts()
        detail = ", ".join(f"{k}:{v}" for k, v in by_class.items()) or "-"
        print(f"  {threshold:>8.1f}%{len(selected):>11}"
              f"{100 * len(selected) / len(quality):>9.1f}%   {detail}")

    print("\n  Efecto de umbrales candidatos de SNR:\n")
    print(f"  {'umbral':>9}{'marcados':>11}{'% corpus':>10}   reparto por grupo")
    for threshold in (3.0, 4.0, 5.0, 6.0, 7.0):
        selected = quality.loc[quality["snr_proxy_db"] < threshold]
        by_group = selected["group"].value_counts()
        detail = ", ".join(f"{k}:{v}" for k, v in by_group.items()) or "-"
        print(f"  {threshold:>8.1f} dB{len(selected):>10}"
              f"{100 * len(selected) / len(quality):>9.1f}%   {detail}")


def step_1d_duplicates(quality):
    """Detecta copias binarias y decide su elegibilidad para modelado."""
    u.section("FASE 1d - DUPLICADOS BINARIOS")
    rows = []
    modeling = {}
    duplicate_groups = quality.groupby("file_sha256", dropna=False)
    duplicate_hashes = sorted(
        hash_value for hash_value, group in duplicate_groups
        if hash_value and len(group) > 1
    )

    for number, hash_value in enumerate(duplicate_hashes, start=1):
        group = quality.loc[quality["file_sha256"] == hash_value].sort_values("audio_id")
        group_id = f"DUP_{number:03d}"
        conflicting = group["patient_uid"].nunique() > 1 or group["diagnosis"].nunique() > 1
        duplicate_type = "LABEL_CONFLICT" if conflicting else "REDUNDANT_COPY"

        for position, (_, item) in enumerate(group.iterrows()):
            if conflicting:
                action = "EXCLUDE_FROM_MODELING"
                status = "EXCLUDE_LABEL_CONFLICT"
                reason = "mismo contenido binario asociado a paciente o diagnostico distinto"
            elif position == 0:
                action = "KEEP_REFERENCE"
                status = "ELIGIBLE"
                reason = "copia de referencia conservada"
            else:
                action = "EXCLUDE_REDUNDANT_COPY"
                status = "EXCLUDE_REDUNDANT"
                reason = "copia binaria redundante dentro del mismo paciente y diagnostico"

            modeling[item["audio_id"]] = {
                "duplicate_group_id": group_id,
                "duplicate_type": duplicate_type,
                "modeling_status": status,
                "modeling_reason": reason,
            }
            rows.append({
                "duplicate_group_id": group_id,
                "file_sha256": hash_value,
                "dataset": item["dataset"],
                "audio_id": item["audio_id"],
                "patient_uid": item["patient_uid"],
                "diagnosis": item["diagnosis"],
                "device": item["device"],
                "filter": item["filter"],
                "duplicate_type": duplicate_type,
                "action": action,
                "modeling_status": status,
                "reason": reason,
            })

    columns = [
        "duplicate_group_id", "file_sha256", "dataset", "audio_id",
        "patient_uid", "diagnosis", "device", "filter", "duplicate_type",
        "action", "modeling_status", "reason",
    ]
    report = pd.DataFrame(rows, columns=columns)
    out = cfg.REPORTS / "duplicate_audio_report.csv"
    report.to_csv(out, index=False)
    print(f"  Grupos duplicados         : {len(duplicate_hashes)}")
    print(f"  Archivos implicados       : {len(report)}")
    print(f"  Exclusiones para modelado : {(report['modeling_status'] != 'ELIGIBLE').sum() if len(report) else 0}")
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")
    return report, modeling


def _normal_text(value):
    return " ".join(str(value).strip().split())


def _normal_age(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return _normal_text(value)


def step_1e_metadata_sources(meta):
    """Contrasta los CSV derivados con las fuentes clinicas disponibles."""
    u.section("FASE 1e - VALIDACION DE METADATA FUENTE")
    issues = []

    def add_issue(row, field, current, source, reference, issue="VALUE_MISMATCH"):
        issues.append({
            "dataset": row["dataset"], "audio_id": row["audio_id"],
            "patient_uid": row["patient_uid"], "field": field,
            "metadata_value": current, "source_value": source,
            "source_reference": reference, "issue": issue,
        })

    diagnoses = pd.read_csv(
        cfg.ICBHI_DIAGNOSES, header=None, names=["patient_id", "diagnosis"],
        dtype=str, keep_default_na=False,
    )
    diagnoses = dict(zip(diagnoses["patient_id"].str.strip(), diagnoses["diagnosis"].str.strip()))

    for _, row in meta.loc[meta["dataset"] == "ICBHI"].iterrows():
        patient_id = row["patient_id"].strip()
        source_diagnosis = diagnoses.get(patient_id)
        if source_diagnosis is None:
            add_issue(row, "patient_id", patient_id, "", str(cfg.ICBHI_DIAGNOSES), "PATIENT_NOT_IN_SOURCE")
        elif row["diagnosis"] != source_diagnosis:
            add_issue(row, "diagnosis", row["diagnosis"], source_diagnosis, str(cfg.ICBHI_DIAGNOSES))
        validate_filename(row, issues)

    diagnosis_aliases = {
        "n": "Normal", "normal": "Normal", "asthma": "Asthma",
        "heart failure": "HeartFailure", "copd": "COPD",
        "pneumonia": "Pneumonia", "bron": "Bronchitis",
        "bronchitis": "Bronchitis", "lung fibrosis": "LungFibrosis",
        "heart failure + copd": "HeartFailure-COPD",
        "plueral effusion": "PleuralEffusion",
        "pleural effusion": "PleuralEffusion",
        "heart failure + lung fibrosis": "HeartFailure-LungFibrosis",
        "asthma and lung fibrosis": "Asthma-LungFibrosis",
    }
    source = pd.read_excel(cfg.FRAIWAN_SOURCE_XLSX, usecols="A:E", dtype=object)
    source.columns = ["age", "gender", "zone", "sound_type", "diagnosis"]
    source["patient_id"] = [f"F{i:03d}" for i in range(1, len(source) + 1)]
    source = source.set_index("patient_id")

    for _, row in meta.loc[meta["dataset"] == "FRAIWAN"].iterrows():
        patient_id = row["patient_id"].strip()
        if patient_id not in source.index:
            add_issue(row, "patient_id", patient_id, "", str(cfg.FRAIWAN_SOURCE_XLSX), "PATIENT_NOT_IN_SOURCE")
            continue
        raw = source.loc[patient_id]
        raw_diagnosis_key = _normal_text(raw["diagnosis"]).casefold()
        source_diagnosis = diagnosis_aliases.get(raw_diagnosis_key)
        if source_diagnosis is None:
            add_issue(row, "diagnosis", row["diagnosis"], raw["diagnosis"], str(cfg.FRAIWAN_SOURCE_XLSX), "UNRECOGNIZED_SOURCE_VALUE")
        comparisons = {
            "diagnosis": (row["diagnosis"], source_diagnosis),
            "age": (_normal_age(row["age"]), _normal_age(raw["age"])),
            "gender": (_normal_text(row["gender"]).upper(), _normal_text(raw["gender"]).upper()),
            "zone": (_normal_text(row["zone"]).replace(" ", "").upper(), _normal_text(raw["zone"]).replace(" ", "").upper()),
            "sound_type": (_normal_text(row["sound_type"]).upper(), _normal_text(raw["sound_type"]).upper()),
        }
        for field, (current, original) in comparisons.items():
            if original is not None and current != original:
                add_issue(row, field, current, original, str(cfg.FRAIWAN_SOURCE_XLSX))
        validate_filename(row, issues)

    columns = [
        "dataset", "audio_id", "patient_uid", "field", "metadata_value",
        "source_value", "source_reference", "issue",
    ]
    report = pd.DataFrame(issues, columns=columns)
    out = cfg.REPORTS / "metadata_validation.csv"
    report.to_csv(out, index=False)
    print(f"  Registros contrastados    : {len(meta)}")
    print(f"  Incidencias               : {len(report)}")
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")
    return report


def validate_filename(row, issues):
    """Comprueba que nombre, ruta y campos estructurados sean coherentes."""
    actual_name = Path(row["audio_path"]).name
    if actual_name != row["filename"]:
        issues.append({
            "dataset": row["dataset"], "audio_id": row["audio_id"],
            "patient_uid": row["patient_uid"], "field": "filename",
            "metadata_value": row["filename"], "source_value": actual_name,
            "source_reference": row["audio_path"], "issue": "PATH_FILENAME_MISMATCH",
        })
    parts = Path(row["filename"]).stem.split("_")
    expected = [
        row["patient_id"], row["diagnosis"], row["zone"], row["device"],
        row["recording_id"] if row["dataset"] == "ICBHI" else row["filter"],
    ]
    if len(parts) != 5:
        issues.append({
            "dataset": row["dataset"], "audio_id": row["audio_id"],
            "patient_uid": row["patient_uid"], "field": "filename",
            "metadata_value": row["filename"], "source_value": "_".join(expected),
            "source_reference": row["audio_path"], "issue": "INVALID_FILENAME_STRUCTURE",
        })
        return
    field_names = ["patient_id", "diagnosis", "zone", "device", "recording_or_filter"]
    for field, current, wanted in zip(field_names, parts, expected):
        if current != wanted:
            issues.append({
                "dataset": row["dataset"], "audio_id": row["audio_id"],
                "patient_uid": row["patient_uid"], "field": f"filename.{field}",
                "metadata_value": current, "source_value": wanted,
                "source_reference": row["audio_path"], "issue": "FILENAME_FIELD_MISMATCH",
            })


def build_manifest(meta, integrity, quality, annotation_validation, metadata_validation, modeling):
    """Construye una fila contractual por audio para las fases posteriores."""
    u.section("MANIFIESTO DE ADMISION DE LA FASE 1")
    integrity_rows = integrity.loc[integrity["row_type"] == "metadata"].set_index("audio_id")
    quality_rows = quality.set_index("audio_id")
    annotation_bad = set(annotation_validation["audio_id"].dropna()) - {""}
    annotation_global = bool((annotation_validation["audio_id"].fillna("") == "").any())
    metadata_bad = set(metadata_validation["audio_id"].dropna()) - {""}
    metadata_global = bool((metadata_validation["audio_id"].fillna("") == "").any())

    rows = []
    for _, item in meta.iterrows():
        audio_id = item["audio_id"]
        integ = integrity_rows.loc[audio_id]
        signal = quality_rows.loc[audio_id]
        hard = []
        review = []

        if not bool(integ["passes"]):
            hard.append("INTEGRITY_FAILURE")
        if not bool(signal["readable"]):
            hard.append("UNREADABLE_AUDIO")
        if int(signal["n_samples"]) == 0:
            hard.append("EMPTY_AUDIO")
        if bool(signal["has_non_finite"]):
            hard.append("NON_FINITE_SAMPLES")
        if pd.notna(signal["variance"]) and float(signal["variance"]) <= cfg.MIN_VARIANCE:
            hard.append("FLAT_SIGNAL")
        if pd.notna(signal["rms"]) and float(signal["rms"]) < cfg.MIN_RMS:
            hard.append("RMS_BELOW_MINIMUM")

        if pd.notna(signal["saturation_pct"]) and float(signal["saturation_pct"]) > cfg.MAX_SATURATION_PCT:
            review.append("HIGH_SATURATION")
        if signal["snr_status"] != "OK":
            review.append(f"SNR_{signal['snr_status']}")
        elif pd.notna(signal["snr_proxy_db"]) and float(signal["snr_proxy_db"]) < cfg.MIN_SNR_DB:
            review.append("LOW_SNR_PROXY")
        if pd.notna(signal["digital_silence_pct"]) and float(signal["digital_silence_pct"]) >= cfg.MAX_DIGITAL_SILENCE_PCT:
            review.append("HIGH_DIGITAL_SILENCE")
        annotation_status = "NOT_AVAILABLE"
        if item["dataset"] == "ICBHI":
            annotation_status = "INVALID" if (audio_id in annotation_bad or annotation_global) else "VALID"
            if annotation_status == "INVALID":
                review.append("ANNOTATION_INCIDENT")
        metadata_ok = not (audio_id in metadata_bad or metadata_global)
        if not metadata_ok:
            review.append("METADATA_INCIDENT")

        quality_status = "EXCLUDE" if hard else ("REVIEW" if review else "PASS")
        duplicate = modeling.get(audio_id, {})
        modeling_status = duplicate.get("modeling_status", "ELIGIBLE")
        modeling_reason = duplicate.get("modeling_reason", "")
        eligible = quality_status != "EXCLUDE" and modeling_status == "ELIGIBLE"
        all_reasons = hard + review + ([modeling_reason] if modeling_reason else [])

        rows.append({
            "dataset": item["dataset"], "audio_id": audio_id,
            "patient_uid": item["patient_uid"], "diagnosis": item["diagnosis"],
            "device": item["device"], "filter": item["filter"], "zone": item["zone"],
            "audio_path": item["audio_path"], "file_sha256": signal["file_sha256"],
            "integrity_ok": bool(integ["passes"]), "metadata_ok": metadata_ok,
            "annotation_status": annotation_status,
            "quality_status": quality_status,
            "quality_reasons": ";".join(hard + review),
            "modeling_status": modeling_status,
            "modeling_reason": modeling_reason,
            "pipeline_eligible": eligible,
            "duplicate_group_id": duplicate.get("duplicate_group_id", ""),
            "duplicate_type": duplicate.get("duplicate_type", ""),
            "saturation_pct": signal["saturation_pct"],
            "rms": signal["rms"], "rms_without_dc": signal["rms_without_dc"],
            "variance": signal["variance"], "snr_proxy_db": signal["snr_proxy_db"],
            "snr_status": signal["snr_status"],
            "digital_silence_pct": signal["digital_silence_pct"],
            "digital_silence_longest_samples": signal["digital_silence_longest_samples"],
            "digital_silence_longest_s": signal["digital_silence_longest_s"],
            "reasons": ";".join(all_reasons),
        })

    manifest = pd.DataFrame(rows)
    if len(manifest) != len(meta) or manifest["audio_id"].nunique() != len(meta):
        raise RuntimeError("El manifiesto no contiene exactamente una fila por audio_id")
    manifest.to_csv(cfg.PHASE1_MANIFEST, index=False)
    print("  Calidad:")
    for status, count in manifest["quality_status"].value_counts().items():
        print(f"    {status:<10} {count:>5}")
    print("  Elegibilidad para modelado:")
    for status, count in manifest["modeling_status"].value_counts().items():
        print(f"    {status:<28} {count:>5}")
    print(f"  Admitidos por el pipeline : {int(manifest['pipeline_eligible'].sum())}")
    print(f"\n  -> {cfg.PHASE1_MANIFEST.relative_to(cfg.ROOT)}")
    return manifest


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
        print("  Corresponden a fases posteriores y no afectan la verificacion actual.")

    integrity = step_1a_integrity(meta)
    cycles, cycle_summary, annotation_validation = step_1b_annotations(meta)
    quality = step_1c_signal_quality(meta)
    duplicates, modeling = step_1d_duplicates(quality)
    metadata_validation = step_1e_metadata_sources(meta)
    manifest = build_manifest(
        meta, integrity, quality, annotation_validation, metadata_validation, modeling
    )

    u.section("RESUMEN DE LA FASE 1")
    checked = integrity.loc[integrity["row_type"] == "metadata"]
    print(f"  Integridad    : {int(checked['passes'].sum())}/{len(checked)} sin discrepancias")
    print(f"  Anotaciones   : {len(cycles)} ciclos, {len(annotation_validation)} incidencias")
    print(f"  Calidad       : {int(quality['readable'].sum())}/{len(quality)} legibles")
    print(f"  Duplicados    : {duplicates['duplicate_group_id'].nunique()} grupos, {len(duplicates)} archivos")
    print(f"  Metadata      : {len(metadata_validation)} incidencias")
    print(f"  PASS          : {(manifest['quality_status'] == 'PASS').sum()}")
    print(f"  REVIEW        : {(manifest['quality_status'] == 'REVIEW').sum()}")
    print(f"  EXCLUDE       : {(manifest['quality_status'] == 'EXCLUDE').sum()}")
    print(f"  Elegibles     : {int(manifest['pipeline_eligible'].sum())}")

    return {
        "integrity": integrity, "cycles": cycles, "cycle_summary": cycle_summary,
        "annotation_validation": annotation_validation, "quality": quality,
        "duplicates": duplicates, "metadata_validation": metadata_validation,
        "manifest": manifest,
    }


if __name__ == "__main__":
    main()
