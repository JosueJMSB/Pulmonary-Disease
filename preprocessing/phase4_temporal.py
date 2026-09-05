"""
Fase 4 - Estandarizacion temporal.

  4b  Determinacion de la longitud de ventana  (corre primero: 4a la necesita)
  4a  Segmentacion con solape del 50 %

Convierte las dos ramas de la fase 3 (clean/no_dn, clean/dn), de duracion
desigual (5.00 a 86.20 s), en dos arrays de ventanas de tamano fijo. Ambas
ramas comparten exactamente la misma rejilla de segmentos -mismo origen,
mismos limites de muestra-, de modo que el inventario tiene una fila por
segmento y no por segmento y rama: la fila `array_index` describe
simultaneamente esa misma fila en los dos .npy.

Antes de segmentar se valida por completo la salida de la fase 3: veredicto
PASS, 1249 audio_id por rama, un par no_dn/dn exacto por audio_id con
identidad clinica y duracion identicas entre ramas, y el SHA-256 de cada uno
de los 2498 WAV contra el que registro el manifiesto. La fase se niega a
continuar si algo no coincide.

Los limites de cada segmento se calculan y se cortan en INDICES DE MUESTRA
enteros (start_sample:end_sample), nunca en segundos redondeados: a 4 kHz
con esta ventana y este solape el salto ya es un entero exacto (10000
muestras), pero trabajar en muestras evita que un cambio futuro de ventana
o solape introduzca un error de redondeo silencioso.

Las columnas de ciclos anotados distinguen contencion completa de simple
solape: ICBHI no anota en que punto del ciclo ocurre un crepitante o una
sibilancia, de modo que un segmento que solo roza el borde de un ciclo con
ese evento NO garantiza contener el sonido. Fraiwan no tiene anotacion de
ciclos: esas columnas quedan en NA (no en False ni en 0), porque la ausencia
de anotacion no es lo mismo que la ausencia del evento.

Nada se excluye aqui. `dn_reliable=False` marca 23 grabaciones donde la rama
`dn` quedo daniada; sus segmentos se generan igual, en las dos ramas, con la
marca propagada. Si mas adelante se decide excluirlas de una comparacion
dn/no_dn, deben excluirse de AMBAS ramas a la vez -nunca solo de una-, o la
comparacion dejaria de evaluarse sobre el mismo conjunto de grabaciones.
"""

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as cfg
import utils as u


WINDOW_SAMPLES = int(round(cfg.SEGMENT_SECONDS * cfg.TARGET_SR)) if cfg.SEGMENT_SECONDS else None
HOP_SAMPLES = (
    int(round(cfg.SEGMENT_SECONDS * (1 - cfg.SEGMENT_OVERLAP) * cfg.TARGET_SR))
    if cfg.SEGMENT_SECONDS else None
)

BRANCHES = ("no_dn", "dn")
CLINICAL_COLUMNS = (
    "dataset", "patient_uid", "diagnosis", "device", "filter", "zone",
    "quality_status", "quality_reasons", "calibration_patient",
)
VALIDATION_SAMPLE_SIZE = 60
CYCLE_EVENT_COLUMNS = (
    "n_complete_cycles", "n_partial_cycles",
    "has_complete_crackle_cycle", "has_complete_wheeze_cycle",
    "overlaps_crackle_cycle", "overlaps_wheeze_cycle",
)


def segments_per_recording(n_samples, window=None, hop=None):
    """Numero de ventanas que caben en una grabacion, sin relleno ni cola extra."""
    window = WINDOW_SAMPLES if window is None else window
    hop = HOP_SAMPLES if hop is None else hop
    if n_samples < window:
        return 0
    return (n_samples - window) // hop + 1


# ---------------------------------------------------------------------------
# Validacion completa de la entrada (salida de la fase 3)
# ---------------------------------------------------------------------------

def validate_phase3_input():
    """Comprueba la salida de la fase 3 antes de segmentar. Se niega a
    continuar si algo no coincide: segmentar sobre una salida incompleta o
    alterada produciria ventanas de una senal que ya no es la que se valido.
    """
    u.section("VALIDACION DE LA ENTRADA (SALIDA DE LA FASE 3)")
    checks = []

    def record(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"  {name:<24}: {detail}  [{'OK' if ok else 'REVISAR'}]")

    if not cfg.R3_SUMMARY.exists():
        raise FileNotFoundError(f"No existe {cfg.R3_SUMMARY}. Ejecute phase3_cleaning.py.")
    summary = pd.read_csv(cfg.R3_SUMMARY)
    verdict = str(summary["verdict"].iloc[0])
    record("veredicto_fase3", verdict == "PASS", verdict)

    if not cfg.PHASE3_MANIFEST.exists():
        raise FileNotFoundError(f"No existe {cfg.PHASE3_MANIFEST}. Ejecute phase3_cleaning.py.")
    manifest = pd.read_csv(cfg.PHASE3_MANIFEST, dtype={"audio_id": str})

    n_audio = manifest["audio_id"].nunique()
    record("audios_unicos", n_audio == 1249, f"{n_audio} (esperado 1249)")

    branch_sets = manifest.groupby("audio_id")["branch"].apply(lambda s: frozenset(s))
    expected_pair = frozenset(BRANCHES)
    n_bad_pairs = int((branch_sets != expected_pair).sum())
    record("pares_no_dn_dn", n_bad_pairs == 0, f"{n_bad_pairs} audio_id sin el par exacto")

    structure_ok = n_audio == 1249 and n_bad_pairs == 0
    if not structure_ok:
        checks_df = pd.DataFrame(checks)
        checks_df.to_csv(cfg.R4_INPUT_VALIDATION, index=False)
        raise RuntimeError(
            "La estructura del manifiesto de fase 3 no es la esperada: "
            "no se puede continuar sin 1249 audios con un par no_dn/dn exacto cada uno."
        )

    # fillna antes de comparar: pandas conserva NaN como flotante incluso
    # tras astype(str) (no lo convierte al literal "nan"), y NaN != NaN en
    # IEEE 754 marcaria como discrepancia dos ausencias identicas -por
    # ejemplo "filter" en ICBHI, o "quality_reasons" cuando no hay ninguna-.
    wide = manifest.pivot(index="audio_id", columns="branch")
    mismatches = 0
    for col in CLINICAL_COLUMNS:
        left = wide[(col, "no_dn")].fillna("__NA__").astype(str)
        right = wide[(col, "dn")].fillna("__NA__").astype(str)
        mismatches += int((left != right).sum())
    record("identidad_clinica", mismatches == 0, f"{mismatches} discrepancias entre ramas")

    dur_mismatch = int((wide[("samples", "no_dn")] != wide[("samples", "dn")]).sum())
    record("duracion_identica", dur_mismatch == 0, f"{dur_mismatch} audio_id con distinta duracion")

    print(f"  Verificando SHA-256 de {len(manifest)} archivos contra el manifiesto...")
    hash_bad = []
    progress = u.Progress(len(manifest), "hash", every=250)
    for _, r in manifest.iterrows():
        path = Path(cfg.ROOT) / r["output_path"]
        actual = u.file_sha256(path) if path.exists() else "AUSENTE"
        if actual != r["output_sha256"]:
            hash_bad.append(f"{r['audio_id']}/{r['branch']}")
        progress.step()
    record("sha256_wav", not hash_bad, f"{len(hash_bad)} de {len(manifest)} no coinciden")

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(cfg.R4_INPUT_VALIDATION, index=False)
    print(f"  -> {cfg.R4_INPUT_VALIDATION.relative_to(cfg.ROOT)}")

    if not checks_df["ok"].all():
        bad = checks_df.loc[~checks_df["ok"], "check"].tolist()
        raise RuntimeError(
            "La entrada de la fase 3 no pasa la validacion completa: " + ", ".join(bad)
        )

    return manifest


# ---------------------------------------------------------------------------
# 4b - Determinacion de la longitud de ventana
# ---------------------------------------------------------------------------

def _load_cycle_bounds():
    """Ciclos de ICBHI en muestras, agrupados por audio_id."""
    cycles = pd.read_csv(cfg.R1_CYCLES, dtype={"audio_id": str})
    result = {}
    for audio_id, g in cycles.groupby("audio_id"):
        bounds = np.round(g[["start_s", "end_s"]].to_numpy() * cfg.TARGET_SR).astype(np.int64)
        result[audio_id] = {
            "bounds": bounds,
            "crackles": g["crackles"].to_numpy(),
            "wheezes": g["wheezes"].to_numpy(),
        }
    return result


def compute_window_length_report(no_dn_meta, cycles_by_audio):
    """Mide cada candidata de SEGMENT_CANDIDATES sobre el corpus real.

    Pasada solo de metadatos: no lee audio, solo las duraciones en muestras
    del manifiesto de fase 3 y los ciclos anotados de ICBHI.
    """
    samples_map = dict(zip(no_dn_meta["audio_id"], no_dn_meta["samples"]))
    dataset_map = dict(zip(no_dn_meta["audio_id"], no_dn_meta["dataset"]))
    patient_map = dict(zip(no_dn_meta["audio_id"], no_dn_meta["patient_uid"]))
    n_cycles_total = sum(len(v["bounds"]) for v in cycles_by_audio.values())

    rows = []
    for seconds in cfg.SEGMENT_CANDIDATES:
        window = int(round(seconds * cfg.TARGET_SR))
        hop = int(round(seconds * (1 - cfg.SEGMENT_OVERLAP) * cfg.TARGET_SR))

        n_segments = 0
        recordings_lost = 0
        tail_list = []
        per_patient = {}
        complete_w = partial_w = empty_w = 0
        complete_counts = []
        cycles_contained = 0

        for audio_id, n_samples in samples_map.items():
            n = segments_per_recording(n_samples, window, hop)
            if n == 0:
                recordings_lost += 1
            n_segments += n
            pid = patient_map[audio_id]
            per_patient[pid] = per_patient.get(pid, 0) + n
            covered = (n - 1) * hop + window if n > 0 else 0
            tail_list.append(n_samples - covered)

            if dataset_map[audio_id] != "ICBHI" or n == 0:
                continue
            cyc = cycles_by_audio.get(audio_id)
            if cyc is None or not len(cyc["bounds"]):
                continue
            bounds = cyc["bounds"]
            contained_any = np.zeros(len(bounds), dtype=bool)
            for k in range(n):
                s, e = k * hop, k * hop + window
                complete = (bounds[:, 0] >= s) & (bounds[:, 1] <= e)
                overlap = (bounds[:, 0] < e) & (bounds[:, 1] > s)
                complete_counts.append(int(complete.sum()))
                if complete.any():
                    complete_w += 1
                elif overlap.any():
                    partial_w += 1
                else:
                    empty_w += 1
                contained_any |= complete
            cycles_contained += int(contained_any.sum())

        total_w = complete_w + partial_w + empty_w
        patient_counts = np.array(list(per_patient.values())) if per_patient else np.zeros(0)
        tail = np.array(tail_list, dtype=float)

        rows.append({
            "segment_seconds": seconds,
            "hop_seconds": round(seconds * (1 - cfg.SEGMENT_OVERLAP), 4),
            "window_samples": window, "hop_samples": hop,
            "n_segments": n_segments,
            "recordings_lost": recordings_lost,
            "patients_lost": int((patient_counts == 0).sum()) if len(patient_counts) else 0,
            "pct_windows_with_complete_cycle": round(100 * complete_w / total_w, 2) if total_w else float("nan"),
            "pct_windows_with_partial_cycle": round(100 * partial_w / total_w, 2) if total_w else float("nan"),
            "pct_windows_empty": round(100 * empty_w / total_w, 2) if total_w else float("nan"),
            "mean_complete_cycles_per_window": round(float(np.mean(complete_counts)), 3) if complete_counts else float("nan"),
            "pct_cycles_contained": round(100 * cycles_contained / n_cycles_total, 2) if n_cycles_total else float("nan"),
            "median_segments_per_patient": float(np.median(patient_counts)) if len(patient_counts) else float("nan"),
            "patients_under_5_segments": int((patient_counts < 5).sum()) if len(patient_counts) else 0,
            "mean_uncovered_tail_s": round(float(tail.mean() / cfg.TARGET_SR), 4) if len(tail) else float("nan"),
            "max_uncovered_tail_s": round(float(tail.max() / cfg.TARGET_SR), 4) if len(tail) else float("nan"),
            "size_mb_float32_per_branch": round(n_segments * window * 4 / 1024 ** 2, 1),
            "selected": bool(abs(seconds - cfg.SEGMENT_SECONDS) < 1e-9),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4a - Segmentacion
# ---------------------------------------------------------------------------

def cycle_classification(audio_id, cycles_by_audio, start_sample, end_sample, dataset):
    """Contencion completa frente a simple solape, para un segmento.

    Un ciclo "completo" cae integro dentro del segmento; uno "parcial" lo
    toca sin caber entero. Solo el completo garantiza que un evento
    adventicio anotado en ese ciclo esta realmente contenido: ICBHI no anota
    en que punto del ciclo ocurre el crepitante o la sibilancia, de modo que
    un segmento que solo roza el borde de un ciclo con ese evento no
    garantiza contener el sonido.

    Fraiwan no tiene ciclos anotados: se devuelve NA (no False ni 0), porque
    la ausencia de anotacion no es lo mismo que la ausencia del evento.
    """
    if dataset != "ICBHI":
        return {col: pd.NA for col in CYCLE_EVENT_COLUMNS}

    cyc = cycles_by_audio.get(audio_id)
    if cyc is None or not len(cyc["bounds"]):
        return {
            "n_complete_cycles": 0, "n_partial_cycles": 0,
            "has_complete_crackle_cycle": False, "has_complete_wheeze_cycle": False,
            "overlaps_crackle_cycle": False, "overlaps_wheeze_cycle": False,
        }

    bounds, crackles, wheezes = cyc["bounds"], cyc["crackles"], cyc["wheezes"]
    complete = (bounds[:, 0] >= start_sample) & (bounds[:, 1] <= end_sample)
    overlap = (bounds[:, 0] < end_sample) & (bounds[:, 1] > start_sample)
    partial = overlap & ~complete
    return {
        "n_complete_cycles": int(complete.sum()),
        "n_partial_cycles": int(partial.sum()),
        "has_complete_crackle_cycle": bool((complete & (crackles == 1)).any()),
        "has_complete_wheeze_cycle": bool((complete & (wheezes == 1)).any()),
        "overlaps_crackle_cycle": bool((overlap & (crackles == 1)).any()),
        "overlaps_wheeze_cycle": bool((overlap & (wheezes == 1)).any()),
    }


def build_segment_inventory(no_dn_meta, cycles_by_audio, dn_reliable_map, dn_reason_map):
    """Rejilla de segmentos, identica para las dos ramas: mismo origen,
    mismos indices de muestra. Un ciclo `for` por grabacion basta porque los
    limites no dependen del audio, solo de su longitud en muestras."""
    rows = []
    for _, r in no_dn_meta.iterrows():
        audio_id = r["audio_id"]
        dataset = r["dataset"]
        n_samples = int(r["samples"])
        n = segments_per_recording(n_samples)
        tail = n_samples - ((n - 1) * HOP_SAMPLES + WINDOW_SAMPLES) if n > 0 else n_samples

        for k in range(n):
            start_sample = k * HOP_SAMPLES
            end_sample = start_sample + WINDOW_SAMPLES
            row = {
                "segment_id": f"{audio_id}_{k:03d}",
                "audio_id": audio_id, "dataset": dataset,
                "patient_uid": r["patient_uid"], "diagnosis": r["diagnosis"],
                "device": r["device"], "zone": r["zone"], "filter": r["filter"],
                "start_sample": start_sample, "end_sample": end_sample,
                "start_s": round(start_sample / cfg.TARGET_SR, 4),
                "end_s": round(end_sample / cfg.TARGET_SR, 4),
                "tail_samples": int(tail),
                "segment_idx": k, "n_segments_in_recording": n,
                "quality_status": r["quality_status"], "quality_reasons": r["quality_reasons"],
                "calibration_patient": bool(r["calibration_patient"]),
                "dn_reliable": dn_reliable_map[audio_id],
                "dn_flag_reason": dn_reason_map.get(audio_id, ""),
            }
            row.update(cycle_classification(audio_id, cycles_by_audio, start_sample, end_sample, dataset))
            rows.append(row)

    inventory = pd.DataFrame(rows)
    inventory.insert(0, "array_index", np.arange(len(inventory), dtype=np.int64))
    return inventory


# ---------------------------------------------------------------------------
# Escritura de los .npy por memoria mapeada
# ---------------------------------------------------------------------------

def fill_branch_array(branch, inventory, staging_dir, path_map, samples_map):
    """Escribe el array de una rama directamente en disco por memmap.

    No reserva el array completo en RAM (serian ~667 MB por rama): abre el
    .npy de destino como memoria mapeada y escribe fila a fila conforme se
    lee cada grabacion, una sola vez por grabacion.
    """
    dest = staging_dir / f"segments_{branch}.npy"
    tmp = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)

    array = np.lib.format.open_memmap(
        str(tmp), mode="w+", dtype=np.dtype(cfg.SEGMENT_DTYPE),
        shape=(len(inventory), WINDOW_SAMPLES),
    )

    by_recording = inventory.groupby("audio_id", sort=False)
    progress = u.Progress(inventory["audio_id"].nunique(), f"segmentos {branch}", every=100)
    for audio_id, group in by_recording:
        wav_path = Path(cfg.ROOT) / path_map[audio_id]
        x, sr = u.read_audio(wav_path)
        if sr != cfg.TARGET_SR:
            raise ValueError(f"{audio_id}/{branch}: frecuencia {sr} Hz, se esperaba {cfg.TARGET_SR}")
        expected_samples = int(samples_map[audio_id])
        if x.size != expected_samples:
            raise ValueError(
                f"{audio_id}/{branch}: {x.size} muestras en disco, "
                f"{expected_samples} en el manifiesto"
            )
        for _, seg in group.iterrows():
            array[int(seg["array_index"]), :] = x[seg["start_sample"]:seg["end_sample"]]
        progress.step()

    array.flush()
    del array
    gc.collect()
    os.replace(tmp, dest)
    return dest


# ---------------------------------------------------------------------------
# Validacion de la salida
# ---------------------------------------------------------------------------

def validate_output(inventory, staging_dir, path_maps):
    lines = []
    ok = True

    arrays = {b: np.load(staging_dir / f"segments_{b}.npy", mmap_mode="r") for b in BRANCHES}

    rows_match = all(a.shape[0] == len(inventory) for a in arrays.values())
    ok = ok and rows_match
    lines.append(f"1. Filas del inventario = filas de cada .npy: inventario={len(inventory)}"
                 f" no_dn={arrays['no_dn'].shape[0]} dn={arrays['dn'].shape[0]}"
                 f"  [{'OK' if rows_match else 'REVISAR'}]")

    same_shape = arrays["no_dn"].shape == arrays["dn"].shape
    ok = ok and same_shape
    lines.append(f"2. Forma identica entre ramas: {arrays['no_dn'].shape} vs {arrays['dn'].shape}"
                 f"  [{'OK' if same_shape else 'REVISAR'}]")

    width_ok = arrays["no_dn"].shape[1] == WINDOW_SAMPLES
    ok = ok and width_ok
    lines.append(f"3. Ancho = {WINDOW_SAMPLES} muestras ({cfg.SEGMENT_SECONDS} s x {cfg.TARGET_SR} Hz)"
                 f"  [{'OK' if width_ok else 'REVISAR'}]")

    nan_bad = {}
    for branch, arr in arrays.items():
        bad = 0
        for start in range(0, arr.shape[0], 500):
            chunk = np.asarray(arr[start:start + 500])
            bad += int((~np.isfinite(chunk)).any(axis=1).sum())
        nan_bad[branch] = bad
    nan_ok = all(v == 0 for v in nan_bad.values())
    ok = ok and nan_ok
    lines.append(f"4. NaN/Inf: no_dn={nan_bad['no_dn']} dn={nan_bad['dn']}"
                 f"  [{'OK' if nan_ok else 'REVISAR'}]")

    rng = np.random.RandomState(0)
    sample_n = min(VALIDATION_SAMPLE_SIZE, len(inventory))
    sample_idx = rng.choice(len(inventory), size=sample_n, replace=False) if sample_n else []
    identity_mismatches = 0
    wav_cache = {}
    for i in sample_idx:
        seg = inventory.iloc[int(i)]
        for branch in BRANCHES:
            cache_key = (branch, seg["audio_id"])
            if cache_key not in wav_cache:
                wav_cache[cache_key] = u.read_audio(
                    Path(cfg.ROOT) / path_maps[branch][seg["audio_id"]]
                )[0]
            x = wav_cache[cache_key]
            expected = x[seg["start_sample"]:seg["end_sample"]].astype(cfg.SEGMENT_DTYPE)
            got = np.asarray(arrays[branch][int(seg["array_index"])])
            if not np.array_equal(expected, got):
                identity_mismatches += 1
    identity_ok = identity_mismatches == 0
    ok = ok and identity_ok
    lines.append(f"5. Muestras identicas al origen ({sample_n} segmentos x 2 ramas)"
                 f": {identity_mismatches} discrepancias  [{'OK' if identity_ok else 'REVISAR'}]")

    n_no_segments = int((inventory.groupby("audio_id")["n_segments_in_recording"].first() == 0).sum())
    coverage_ok = n_no_segments == 0
    ok = ok and coverage_ok
    lines.append(f"6. Grabaciones admitidas sin segmentos: {n_no_segments}"
                 f"  [{'OK' if coverage_ok else 'REVISAR'}]")

    fraiwan_base = (
        inventory.loc[inventory["dataset"] == "FRAIWAN", "audio_id"]
        .str.replace(r"_(Bell|Diaphragm|Extended)$", "", regex=True)
    )
    fraiwan_patients = inventory.loc[inventory["dataset"] == "FRAIWAN"].assign(base=fraiwan_base)
    split_bases = fraiwan_patients.groupby("base")["patient_uid"].nunique()
    filters_ok = bool((split_bases <= 1).all())
    ok = ok and filters_ok
    lines.append(f"7. Modos de filtrado de Fraiwan con un unico paciente: "
                 f"{int((split_bases > 1).sum())} excepciones  [{'OK' if filters_ok else 'REVISAR'}]")

    n_calib_patients = inventory.loc[inventory["calibration_patient"], "patient_uid"].nunique()
    calib_table = pd.read_csv(cfg.R3_CALIBRATION, dtype={"patient_uid": str})
    expected_calib = int(calib_table["is_calibration"].sum())
    calib_ok = n_calib_patients == expected_calib
    ok = ok and calib_ok
    lines.append(f"8. Pacientes de calibracion presentes y marcados: {n_calib_patients}"
                 f" (esperado {expected_calib})  [{'OK' if calib_ok else 'REVISAR'}]")

    unreliable_audio = set(inventory.loc[~inventory["dn_reliable"].astype(bool), "audio_id"])
    propagation_bad = 0
    for audio_id in unreliable_audio:
        group = inventory.loc[inventory["audio_id"] == audio_id, "dn_reliable"]
        if not (~group.astype(bool)).all():
            propagation_bad += 1
    propagation_ok = propagation_bad == 0
    ok = ok and propagation_ok
    lines.append(f"9. dn_reliable=False propagado a todos los segmentos de la grabacion: "
                 f"{propagation_bad} grabaciones inconsistentes  [{'OK' if propagation_ok else 'REVISAR'}]"
                 f"  ({len(unreliable_audio)} grabaciones marcadas)")

    segment_ok = cfg.SEGMENT_SECONDS is not None
    ok = ok and segment_ok
    lines.append(f"10. SEGMENT_SECONDS fijado: {cfg.SEGMENT_SECONDS}  [{'OK' if segment_ok else 'REVISAR'}]")

    lines.append("11. Determinismo: no se comprueba dentro de una sola ejecucion; "
                  "reejecutar debe reproducir segments.csv y ambos .npy byte a byte "
                  "(misma entrada, mismo codigo -> misma salida).")

    return {"ok": ok, "report_lines": lines, "nan_bad": nan_bad,
            "identity_mismatches": identity_mismatches, "identity_checked": sample_n * len(BRANCHES)}


# ---------------------------------------------------------------------------

def build_segment_summary(inventory):
    """Estadisticas que van directas al informe: desbalance, clases
    minoritarias, segmentos por paciente, audio descartado en las colas."""
    by_diag = inventory.groupby("diagnosis").size().rename("n_segments")
    by_patient = inventory.groupby("patient_uid").size()
    tails = inventory.drop_duplicates("audio_id")["tail_samples"] / cfg.TARGET_SR

    summary = by_diag.reset_index()
    summary["pct_segments"] = round(100 * summary["n_segments"] / len(inventory), 2)
    summary = summary.sort_values("n_segments", ascending=False)

    print("\n  Segmentos por diagnostico:")
    print(summary.to_string(index=False))
    print(f"\n  Segmentos por paciente   : mediana={by_patient.median():.0f}"
          f"  minimo={by_patient.min()}  bajo 5: {(by_patient < 5).sum()}")
    print(f"  Cola descartada por grabacion (s): media={tails.mean():.3f}"
          f"  max={tails.max():.3f}  total={tails.sum():.1f} s")

    return summary


# ---------------------------------------------------------------------------

def main():
    cfg.ensure_dirs()

    if cfg.SEGMENT_SECONDS is None:
        u.section("FASE 4 - PARAMETRO SIN FIJAR")
        print("  SEGMENT_SECONDS no esta fijado en config.py.")
        return {"verdict": "PENDING"}

    u.section("FASE 4 - ESTANDARIZACION TEMPORAL")
    try:
        manifest = validate_phase3_input()
    except RuntimeError as exc:
        u.section("FASE 4 - VALIDACION DE ENTRADA FALLIDA, NO SE PROCESA NADA")
        print(f"  {exc}")
        pd.DataFrame([{"verdict": "FAIL", "stage": "input_validation", "detail": str(exc)}]
                     ).to_csv(cfg.R4_SUMMARY, index=False)
        return {"verdict": "FAIL"}

    no_dn_meta = manifest.loc[manifest["branch"] == "no_dn"].reset_index(drop=True)
    dn_rows = manifest.loc[manifest["branch"] == "dn"].set_index("audio_id")
    dn_reliable_map = dn_rows["dn_reliable"].astype(bool).to_dict()
    dn_reason_map = dn_rows["dn_flag_reason"].fillna("").to_dict()
    path_maps = {
        b: dict(zip(manifest.loc[manifest["branch"] == b, "audio_id"],
                     manifest.loc[manifest["branch"] == b, "output_path"]))
        for b in BRANCHES
    }
    samples_maps = {
        b: dict(zip(manifest.loc[manifest["branch"] == b, "audio_id"],
                     manifest.loc[manifest["branch"] == b, "samples"]))
        for b in BRANCHES
    }

    u.section("FASE 4b - DETERMINACION DE LA LONGITUD DE VENTANA")
    cycles_by_audio = _load_cycle_bounds()
    window_df = compute_window_length_report(no_dn_meta, cycles_by_audio)
    window_df.to_csv(cfg.R4_WINDOW_LENGTH, index=False)
    print(window_df.round(3).to_string(index=False))
    chosen = window_df.loc[window_df["selected"]].iloc[0]
    print(f"\n  Longitud fijada: {cfg.SEGMENT_SECONDS} s  (ventana={WINDOW_SAMPLES} muestras,"
          f" salto={HOP_SAMPLES} muestras, {int(chosen['n_segments'])} segmentos por rama)")
    print(f"  -> {cfg.R4_WINDOW_LENGTH.relative_to(cfg.ROOT)}")

    u.section("FASE 4a - SEGMENTACION")
    inventory = build_segment_inventory(no_dn_meta, cycles_by_audio, dn_reliable_map, dn_reason_map)
    print(f"  Segmentos totales (por rama): {len(inventory)}")
    print(f"  Grabaciones de calibracion   : {inventory.loc[inventory['calibration_patient'],'patient_uid'].nunique()}")
    print(f"  Grabaciones dn_reliable=False: {(~inventory.drop_duplicates('audio_id')['dn_reliable']).sum()}")

    u.prepare_staging(cfg.FINAL)
    staging = u.staging_dir_for(cfg.FINAL)

    for branch in BRANCHES:
        fill_branch_array(branch, inventory, staging, path_maps[branch], samples_maps[branch])
    inventory.to_csv(staging / "segments.csv", index=False)

    u.section("VALIDACION DE LA FASE 4")
    checks = validate_output(inventory, staging, path_maps)
    for line in checks["report_lines"]:
        print(f"  {line}")

    u.section("RESUMEN DE SEGMENTOS (informativo, para la memoria)")
    print("  Nota: las estadisticas de ciclos, crepitantes y sibilancias se calculan"
          " unicamente sobre ICBHI; Fraiwan no tiene ciclos anotados.")
    segment_summary = build_segment_summary(inventory)
    segment_summary.to_csv(cfg.R4_SEGMENT_SUMMARY, index=False)
    print(f"\n  -> {cfg.R4_SEGMENT_SUMMARY.relative_to(cfg.ROOT)}")

    hashes = {
        "segments_no_dn_sha256": u.file_sha256(staging / "segments_no_dn.npy"),
        "segments_dn_sha256": u.file_sha256(staging / "segments_dn.npy"),
        "segments_csv_sha256": u.file_sha256(staging / "segments.csv"),
    }
    arr_no_dn = np.load(staging / "segments_no_dn.npy", mmap_mode="r")
    summary_row = {
        "verdict": "PASS" if checks["ok"] else "FAIL",
        "n_segments": len(inventory),
        "segment_seconds": cfg.SEGMENT_SECONDS, "segment_overlap": cfg.SEGMENT_OVERLAP,
        "window_samples": WINDOW_SAMPLES, "hop_samples": HOP_SAMPLES,
        "dtype": str(arr_no_dn.dtype), "shape_rows": arr_no_dn.shape[0], "shape_cols": arr_no_dn.shape[1],
        "n_recordings": no_dn_meta["audio_id"].nunique(),
        "n_calibration_patients": int(inventory.loc[inventory["calibration_patient"], "patient_uid"].nunique()),
        "n_dn_unreliable_recordings": int((~inventory.drop_duplicates("audio_id")["dn_reliable"]).sum()),
        "identity_check_segments": checks["identity_checked"],
        "identity_check_mismatches": checks["identity_mismatches"],
        "cycles_computed_on": "ICBHI_only",
        **hashes,
    }
    del arr_no_dn

    if not checks["ok"]:
        pd.DataFrame([summary_row]).to_csv(cfg.R4_SUMMARY, index=False)
        inventory.to_csv(cfg.R4_FAILED_ATTEMPT, index=False)
        u.section("FASE 4 - VALIDACION FALLIDA, NO SE REEMPLAZA LA SALIDA")
        exists = cfg.FINAL.exists()
        print(f"  {'data/final/ permanece intacto.' if exists else 'data/final/ no existia y sigue sin existir.'}")
        print(f"  Intento fallido en {cfg.R4_FAILED_ATTEMPT.relative_to(cfg.ROOT)}")
        return {"verdict": "FAIL"}

    u.swap_staging_into_place(cfg.FINAL)
    pd.DataFrame([summary_row]).to_csv(cfg.R4_SUMMARY, index=False)
    if cfg.R4_FAILED_ATTEMPT.exists():
        cfg.R4_FAILED_ATTEMPT.unlink()

    u.section("RESUMEN DE LA FASE 4")
    print(f"  Segmentos       : {len(inventory)} por rama")
    print(f"  Forma           : {summary_row['shape_rows']} x {summary_row['shape_cols']}"
          f"  ({summary_row['dtype']})")
    total_mb = sum(p.stat().st_size for p in cfg.FINAL.glob("*.npy")) / 1024 ** 2
    print(f"  Tamano en disco : {total_mb:.1f} MB (dos ramas)")
    print(f"  Salida          : {cfg.FINAL.relative_to(cfg.ROOT)}")
    print(f"  Inventario      : {(cfg.FINAL / 'segments.csv').relative_to(cfg.ROOT)}")
    print(f"  Veredicto       : PASS")

    return {"verdict": "PASS"}


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["verdict"] == "PASS" else 1)
