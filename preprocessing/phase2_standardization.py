"""
Fase 2 - Estandarizacion de la senal.

Los audios proceden de campanas con condiciones tecnicas distintas y presentan
frecuencias de muestreo heterogeneas. Esta fase las lleva a una frecuencia comun
de 4 kHz.

  2a  Filtro anti-aliasing
  2b  Estandarizacion de frecuencia

El filtro anti-aliasing se disena explicitamente en vez de usar el que
resample_poly aplica por defecto (ventana Kaiser con beta fijo en 5.0, cuyo
corte cae exactamente en el nuevo Nyquist y solo ofrece 6 dB de atenuacion
ahi). El diseno propio fija una banda de transicion de 1800 a 2000 Hz con al
menos 60 dB de atenuacion medidos, tanto en los coeficientes como con tonos
puros y con audio real.

La salida se escribe primero en un directorio de staging y solo reemplaza a
la version anterior si supera toda la validacion. Si algo falla, la salida
existente permanece intacta y el error queda documentado.
"""

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import scipy
import soundfile as sf
from scipy.signal import firwin, freqz, kaiserord, resample_poly

import config as cfg
import utils as u


# Las etapas intermedias se almacenan en coma flotante de 32 bits. Evita la
# perdida por cuantizacion al encadenar transformaciones y elimina el riesgo de
# recorte cuando el filtrado produce un ligero rebase por encima de 1.0.
INTERIM_SUBTYPE = "FLOAT"

STAGING = cfg.INTERIM / "resampled_staging"
PREVIOUS_SWAP = cfg.INTERIM / "resampled_previous_swap"

SPECTRAL_CHECK_SECONDS = 5

TONE_PASSBAND_HZ = (50, 100, 500, 1000, 1600, 1800)
TONE_STOPBAND_HZ = (2050, 2200, 2500)
TONE_DURATION_S = 2.0


# ---------------------------------------------------------------------------
# Manifiesto de admision
# ---------------------------------------------------------------------------

def admitted_recordings():
    """Metadata admitida por el manifiesto contractual de la fase 1.

    PASS y REVIEW pueden continuar; EXCLUDE y las copias no elegibles para
    modelado quedan fuera. La ausencia o inconsistencia del manifiesto detiene
    la fase para impedir que se procese accidentalmente el corpus completo.
    """
    meta = u.load_metadata()
    manifest_path = cfg.PHASE1_MANIFEST

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No existe {manifest_path}. Ejecute primero phase1_verification.py."
        )

    manifest = pd.read_csv(manifest_path, dtype={"audio_id": str})
    required = {"audio_id", "quality_status", "modeling_status", "pipeline_eligible",
                "file_sha256"}
    missing_columns = sorted(required - set(manifest.columns))
    if missing_columns:
        raise ValueError(f"Manifiesto incompleto: faltan {', '.join(missing_columns)}")
    if manifest["audio_id"].duplicated().any():
        raise ValueError("El manifiesto contiene audio_id duplicados")

    meta_ids = set(meta["audio_id"])
    manifest_ids = set(manifest["audio_id"])
    if meta_ids != manifest_ids:
        missing = sorted(meta_ids - manifest_ids)
        extra = sorted(manifest_ids - meta_ids)
        raise ValueError(
            f"El manifiesto no coincide con la metadata: faltan={missing[:5]}, sobran={extra[:5]}"
        )

    eligible = (
        manifest["quality_status"].isin(["PASS", "REVIEW"])
        & manifest["modeling_status"].eq("ELIGIBLE")
    )
    declared_eligible = manifest["pipeline_eligible"].astype(str).str.lower().map(
        {"true": True, "false": False}
    )
    if declared_eligible.isna().any() or not declared_eligible.equals(eligible):
        raise ValueError("pipeline_eligible contradice los estados del manifiesto")

    admitted_ids = set(manifest.loc[eligible, "audio_id"])
    admitted = meta.loc[meta["audio_id"].isin(admitted_ids)].reset_index(drop=True)
    n_not_eligible = int((~eligible).sum())
    n_review = int((eligible & manifest["quality_status"].eq("REVIEW")).sum())
    return admitted, manifest, n_not_eligible, n_review


# ---------------------------------------------------------------------------
# Filtro anti-aliasing explicito
# ---------------------------------------------------------------------------

def design_antialias_filter(source_sr, up):
    """Disena el FIR anti-aliasing para una relacion de remuestreo.

    Se disena en el dominio interpolado que usa resample_poly (fs_origen * up),
    porque ahi es donde el filtro actua antes de decimar por down. La banda de
    transicion es fija en Hz (1800-2000); el numero de coeficientes resultante
    depende de cuan estrecha sea esa banda en relacion al Nyquist del dominio
    interpolado, que crece con el factor de interpolacion.
    """
    fs_design = source_sr * up
    width = (cfg.ANTIALIAS_STOPBAND_HZ - cfg.ANTIALIAS_PASSBAND_HZ) / (fs_design / 2)
    numtaps, beta = kaiserord(cfg.ANTIALIAS_RIPPLE_DB, width)
    if numtaps % 2 == 0:
        numtaps += 1
    coefficients = firwin(numtaps, cutoff=cfg.ANTIALIAS_CUTOFF_HZ,
                           window=("kaiser", beta), fs=fs_design, scale=True)
    return coefficients, fs_design, numtaps, float(beta)


_FILTER_CACHE = {}


def get_filter(source_sr):
    """FIR anti-aliasing para una frecuencia de origen, disenado una sola vez."""
    if source_sr not in _FILTER_CACHE:
        up, down = cfg.RESAMPLE_RATIOS[source_sr]
        coefficients, fs_design, numtaps, beta = design_antialias_filter(source_sr, up)
        _FILTER_CACHE[source_sr] = {
            "coefficients": coefficients, "up": up, "down": down,
            "fs_design": fs_design, "numtaps": numtaps, "beta": round(beta, 4),
        }
    return _FILTER_CACHE[source_sr]


def verify_filter_design(source_sr):
    """Ondulacion en banda pasante y atenuacion en banda eliminada, con freqz.

    Evalua los coeficientes directamente, sin pasar audio por ellos: es la
    comprobacion de que el diseno matematico cumple su especificacion.
    """
    filt = get_filter(source_sr)
    orig_nyquist = source_sr / 2

    passband_freqs = np.linspace(20, cfg.ANTIALIAS_PASSBAND_HZ, 400)
    stopband_freqs = np.linspace(cfg.ANTIALIAS_STOPBAND_HZ, orig_nyquist, 2000)

    _, h_pass = freqz(filt["coefficients"], worN=passband_freqs, fs=filt["fs_design"])
    _, h_stop = freqz(filt["coefficients"], worN=stopband_freqs, fs=filt["fs_design"])
    passband_db = 20 * np.log10(np.abs(h_pass) + 1e-300)
    stopband_db = 20 * np.log10(np.abs(h_stop) + 1e-300)

    ripple_db = float(passband_db.max() - passband_db.min())
    worst_attenuation_db = float(-stopband_db.max())

    return {
        "source_sr": source_sr, "up": filt["up"], "down": filt["down"],
        "fs_design_hz": filt["fs_design"], "numtaps": filt["numtaps"], "beta": filt["beta"],
        "passband_ripple_db": round(ripple_db, 5),
        "stopband_min_attenuation_db": round(worst_attenuation_db, 3),
        "passband_ok": ripple_db <= cfg.ANTIALIAS_PASSBAND_TOL_DB,
        "stopband_ok": worst_attenuation_db >= cfg.ANTIALIAS_MIN_ATTEN_DB,
    }


def tone_gain_db(source_sr, freq, duration=TONE_DURATION_S):
    """Ganancia medida haciendo pasar un tono puro por el remuestreo real.

    A diferencia de freqz, esto ejercita la cadena completa -interpolacion,
    filtrado y decimacion- tal como se aplica a cada archivo, con el mismo
    padtype. Se descarta el primer y ultimo decimo de la salida porque el
    arranque y cierre del filtro distorsionan esos tramos por construccion.
    """
    filt = get_filter(source_sr)
    n = int(round(duration * source_sr))
    t = np.arange(n) / source_sr
    x = np.sin(2 * np.pi * freq * t)
    y = resample_poly(x, up=filt["up"], down=filt["down"],
                       window=filt["coefficients"], padtype=cfg.ANTIALIAS_PADTYPE)
    edge = max(1, y.size // 10)
    core = y[edge:-edge] if y.size > 2 * edge else y
    rms_in = np.sqrt(np.mean(x ** 2))
    rms_out = np.sqrt(np.mean(core ** 2)) if core.size else 0.0
    if rms_out <= 0:
        return float("-inf")
    return float(20 * np.log10(rms_out / rms_in))


def verify_tones(source_sr):
    """Prueba con tonos puros: uno por frecuencia de banda pasante y eliminada."""
    rows = []
    for freq in TONE_PASSBAND_HZ:
        gain_db = tone_gain_db(source_sr, freq)
        rows.append({
            "source_sr": source_sr, "freq_hz": freq, "band": "passband",
            "gain_db": round(gain_db, 4), "ok": abs(gain_db) <= cfg.ANTIALIAS_PASSBAND_TOL_DB,
        })
    for freq in TONE_STOPBAND_HZ:
        gain_db = tone_gain_db(source_sr, freq)
        rows.append({
            "source_sr": source_sr, "freq_hz": freq, "band": "stopband",
            "gain_db": round(gain_db, 4), "ok": gain_db <= -cfg.ANTIALIAS_MIN_ATTEN_DB,
        })
    return rows


# ---------------------------------------------------------------------------
# Rutas de salida y comparacion espectral sobre audio real
# ---------------------------------------------------------------------------

def relative_output(dataset, audio_path):
    """Ruta relativa (bajo el conjunto) que identifica un archivo de salida."""
    relative = audio_path.split("audio/", 1)[-1]
    return Path(dataset) / relative


def staging_path(dataset, audio_path):
    return STAGING / relative_output(dataset, audio_path)


def final_path(dataset, audio_path):
    return cfg.RESAMPLED / relative_output(dataset, audio_path)


def high_band_energy(x, sr, cutoff):
    """Fraccion de energia por encima de una frecuencia dada."""
    if x.size < 2:
        return float("nan")
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    total = spectrum.sum()
    if total <= 0:
        return float("nan")
    return float(spectrum[freqs > cutoff].sum() / total)


def band_profile(x, sr, edges):
    """Potencia media por banda, comparable entre frecuencias de muestreo.

    La magnitud de la transformada escala con el numero de muestras, de modo
    que |X|^2 escala con su cuadrado. Dividir por N^2 devuelve potencia media
    por banda, que si es comparable entre una senal a 44.1 kHz y la misma
    remuestreada a 4 kHz pese a tener once veces menos muestras.
    """
    if x.size < 2:
        return np.full(len(edges) - 1, np.nan)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2 / (x.size ** 2)
    freqs = np.fft.rfftfreq(x.size, 1.0 / sr)
    return np.array([
        spectrum[(freqs >= lo) & (freqs < hi)].sum()
        for lo, hi in zip(edges[:-1], edges[1:])
    ])


# Fraccion de la frecuencia de Nyquist hasta la que el filtro anti-aliasing
# mantiene su banda plana. Por encima comienza la transicion, donde la
# atenuacion es el comportamiento correcto y no un defecto.
PASSBAND_FRACTION = 0.80


def spectral_comparison(x, source_sr, y, target_sr, band_hz=100):
    """Compara banda a banda el espectro antes y despues del remuestreo.

    No tiene sentido buscar contenido por encima del nuevo Nyquist: el muestreo
    lo impide por construccion. Lo que se comprueba es el estado de la banda que
    si sobrevive, y se distinguen dos fenomenos opuestos.

      - Un exceso de energia (ratio por encima de 1) indica aliasing: contenido
        que se replego desde frecuencias altas y contamino la banda util.
      - Un defecto de energia (ratio por debajo de 1) en las bandas proximas a
        Nyquist es la atenuacion del propio filtro en su banda de transicion,
        que es su funcion.

    Devuelve la desviacion dentro de la banda plana y el maximo exceso sobre el
    espectro original en cualquier banda.
    """
    nyquist = target_sr / 2
    edges = np.arange(0, nyquist + band_hz, band_hz)

    reference = band_profile(x, source_sr, edges)
    obtained = band_profile(y, target_sr, edges)

    significant = reference > reference.max() * 1e-6
    if not significant.any():
        return float("nan"), float("nan")

    ratio = np.where(significant, obtained / np.where(reference > 0, reference, 1), np.nan)

    flat = significant & (edges[:-1] < nyquist * PASSBAND_FRACTION)
    passband_deviation = float(np.max(np.abs(ratio[flat] - 1.0))) if flat.any() else float("nan")
    max_excess = float(np.nanmax(ratio[significant] - 1.0))

    return passband_deviation, max_excess


def stratified_real_audio_sample(meta):
    """Seleccion determinista y estratificada para el diagnostico espectral.

    Los seis audios de 10 kHz (todos los que hay) y, por cada dispositivo a
    44.1 kHz, los primeros doce por audio_id. Sustituye una seleccion sin
    estratificar, que dejaba fuera dispositivos y frecuencias de origen
    enteras del diagnostico.
    """
    resampled = meta.loc[meta["sample_rate_hz"].astype(int) != cfg.TARGET_SR]
    parts = [resampled.loc[resampled["sample_rate_hz"].astype(int) == 10000]]
    forty_four = resampled.loc[resampled["sample_rate_hz"].astype(int) == 44100]
    for _, group in forty_four.groupby("device"):
        parts.append(group.sort_values("audio_id").head(12))
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else resampled.head(0)


# ---------------------------------------------------------------------------
# Staging: escritura segura y reemplazo atomico
# ---------------------------------------------------------------------------

def prepare_staging():
    """Directorio de staging limpio, con salvaguarda ante una ruta inesperada."""
    if STAGING.exists():
        resolved = STAGING.resolve()
        expected = (cfg.INTERIM / "resampled_staging").resolve()
        if resolved != expected:
            raise RuntimeError(
                f"Ruta de staging inesperada, se aborta por seguridad: {resolved}"
            )
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)


def write_atomic(path, y, sr):
    """Escribe primero un archivo temporal y lo renombra al completar.

    Si el proceso se interrumpe a mitad de la escritura, queda un .part
    huerfano y nunca un .wav truncado bajo el nombre final. El sufijo ".part"
    (no ".wav") evita que un intento fallido se cuente como salida valida en
    el recuento de archivos por glob "*.wav". El formato se declara de forma
    explicita porque el nombre temporal ya no permite inferirlo de la
    extension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    sf.write(str(tmp), y.astype(np.float32), sr, subtype=INTERIM_SUBTYPE, format="WAV")
    os.replace(tmp, path)


def swap_staging_into_place():
    """Reemplazo atomico de resampled/ por resampled_staging/.

    Renombrar es la unica operacion atomica disponible a nivel de sistema de
    archivos; copiar no lo es. Si el reemplazo falla a medio camino, se
    restaura la carpeta anterior automaticamente.
    """
    had_previous = cfg.RESAMPLED.exists()
    try:
        if had_previous:
            if PREVIOUS_SWAP.exists():
                shutil.rmtree(PREVIOUS_SWAP)
            os.replace(cfg.RESAMPLED, PREVIOUS_SWAP)
        os.replace(STAGING, cfg.RESAMPLED)
    except OSError:
        if had_previous and PREVIOUS_SWAP.exists() and not cfg.RESAMPLED.exists():
            os.replace(PREVIOUS_SWAP, cfg.RESAMPLED)
        raise
    if had_previous and PREVIOUS_SWAP.exists():
        shutil.rmtree(PREVIOUS_SWAP)


# ---------------------------------------------------------------------------
# Procesamiento por archivo
# ---------------------------------------------------------------------------

def process_recording(row, manifest_row):
    """Remuestrea (o copia) una grabacion y devuelve su fila de auditoria.

    Antes de procesar, contrasta el SHA-256 del archivo de origen contra el
    que registro la fase 1: si difieren, el audio cambio entre fases y no hay
    garantia de que siga siendo el que se valido.
    """
    audio_id = row["audio_id"]
    source_path = Path(row["abs_path"])
    destination = staging_path(row["dataset"], row["audio_path"])
    declared_sr = int(row["sample_rate_hz"])

    entry = {
        "audio_id": audio_id, "dataset": row["dataset"], "audio_path": row["audio_path"],
        "patient_uid": row["patient_uid"], "diagnosis": row["diagnosis"],
        "device": row["device"], "filter": row["filter"], "zone": row["zone"],
        "quality_status": manifest_row["quality_status"],
        "quality_reasons": manifest_row["quality_reasons"],
        "modeling_status": manifest_row["modeling_status"],
        "source_path": source_path.resolve().relative_to(cfg.ROOT).as_posix(),
        "output_path": final_path(row["dataset"], row["audio_path"]).relative_to(cfg.ROOT).as_posix(),
        "source_sha256": "", "output_sha256": "",
        "declared_sr": declared_sr, "actual_sr_in": None, "target_sr": cfg.TARGET_SR,
        "up": None, "down": None, "transformed": None,
        "scipy_version": scipy.__version__,
        "fir_numtaps": None, "fir_beta": None, "fir_window": "kaiser",
        "fir_passband_hz": cfg.ANTIALIAS_PASSBAND_HZ, "fir_stopband_hz": cfg.ANTIALIAS_STOPBAND_HZ,
        "fir_padtype": "", "samples_in": None,
        "samples_out_expected": None, "samples_out_actual": None,
        "duration_in_s": None, "duration_out_s": None,
        "time_error_s": None, "time_error_samples": None,
        "channels_out": None, "subtype_out": "",
        "peak_before": None, "peak_after": None, "exceeds_unity": None,
        "has_nan_or_inf": None, "status": "PENDING", "error": "",
    }

    try:
        source_sha = u.file_sha256(source_path)
        entry["source_sha256"] = source_sha
        expected_sha = str(manifest_row["file_sha256"])
        if expected_sha and expected_sha != "nan" and source_sha != expected_sha:
            raise ValueError(
                f"SHA-256 de origen no coincide con phase1_manifest.csv "
                f"({source_sha[:12]}... vs {expected_sha[:12]}...)"
            )

        x, actual_sr = u.read_audio(source_path)
        entry["actual_sr_in"] = actual_sr
        if actual_sr != declared_sr:
            raise ValueError(
                f"frecuencia real {actual_sr} Hz distinta de la declarada {declared_sr} Hz"
            )

        entry["samples_in"] = int(x.size)
        entry["duration_in_s"] = x.size / actual_sr
        entry["peak_before"] = float(np.max(np.abs(x))) if x.size else 0.0

        if actual_sr == cfg.TARGET_SR:
            y = x
            entry.update({"up": 1, "down": 1, "transformed": False,
                          "samples_out_expected": int(x.size)})
        else:
            if actual_sr not in cfg.RESAMPLE_RATIOS:
                raise ValueError(f"frecuencia de origen no contemplada: {actual_sr} Hz")
            filt = get_filter(actual_sr)
            y = resample_poly(x, up=filt["up"], down=filt["down"],
                               window=filt["coefficients"], padtype=cfg.ANTIALIAS_PADTYPE)
            expected = -(-x.size * filt["up"] // filt["down"])  # techo entero
            entry.update({
                "up": filt["up"], "down": filt["down"], "transformed": True,
                "fir_numtaps": filt["numtaps"], "fir_beta": filt["beta"],
                "fir_padtype": cfg.ANTIALIAS_PADTYPE, "samples_out_expected": expected,
            })

        finite = bool(np.isfinite(y).all())
        entry["has_nan_or_inf"] = not finite
        if not finite:
            raise ValueError("la senal resultante contiene NaN o infinitos")

        entry["samples_out_actual"] = int(y.size)
        entry["duration_out_s"] = y.size / cfg.TARGET_SR
        entry["peak_after"] = float(np.max(np.abs(y))) if y.size else 0.0
        entry["exceeds_unity"] = bool(entry["peak_after"] > 1.0)
        entry["channels_out"] = 1

        if entry["samples_out_actual"] != entry["samples_out_expected"]:
            raise ValueError(
                f"samples_out {entry['samples_out_actual']} distinto del esperado "
                f"{entry['samples_out_expected']}"
            )

        time_error_s = entry["duration_out_s"] - entry["duration_in_s"]
        time_error_samples = time_error_s * cfg.TARGET_SR
        entry["time_error_s"] = time_error_s
        entry["time_error_samples"] = time_error_samples
        if abs(time_error_samples) > cfg.ANTIALIAS_MAX_TIME_ERROR_SAMPLES + 1e-6:
            raise ValueError(
                f"error temporal {time_error_samples:.3f} muestras excede la tolerancia"
            )

        write_atomic(destination, y, cfg.TARGET_SR)
        entry["output_sha256"] = u.file_sha256(destination)
        entry["subtype_out"] = INTERIM_SUBTYPE
        entry["status"] = "OK"
    except Exception as exc:
        entry["status"] = "FAIL"
        entry["error"] = str(exc)

    return entry


# ---------------------------------------------------------------------------
# Orquestacion
# ---------------------------------------------------------------------------

def run(meta, manifest):
    """Remuestrea cada grabacion admitida hacia el directorio de staging."""
    u.section("FASE 2a/2b - FILTRO ANTI-ALIASING Y REMUESTREO")

    by_rate = meta["sample_rate_hz"].astype(int).value_counts().sort_index()
    print("  Frecuencias de origen:")
    design_checks = []
    for rate, count in by_rate.items():
        if rate == cfg.TARGET_SR:
            print(f"    {rate:>6} Hz  {count:>5} archivos   sin transformar")
            continue
        if rate not in cfg.RESAMPLE_RATIOS:
            print(f"    {rate:>6} Hz  {count:>5} archivos   SIN REGLA DEFINIDA")
            continue
        filt = get_filter(rate)
        check = verify_filter_design(rate)
        design_checks.append(check)
        print(f"    {rate:>6} Hz  {count:>5} archivos   remuestreo {filt['up']}/{filt['down']}"
              f"  FIR {filt['numtaps']} coef., beta={filt['beta']:.3f}")

    print("\n  Diseno del filtro (banda de transicion "
          f"{cfg.ANTIALIAS_PASSBAND_HZ}-{cfg.ANTIALIAS_STOPBAND_HZ} Hz):\n")
    print(f"  {'origen':>9}{'ondulacion':>14}{'atenuacion':>14}   veredicto")
    for c in design_checks:
        ok = c["passband_ok"] and c["stopband_ok"]
        print(f"  {c['source_sr']:>7} Hz{c['passband_ripple_db']:>12.4f} dB"
              f"{c['stopband_min_attenuation_db']:>12.2f} dB   [{'OK' if ok else 'REVISAR'}]")

    prepare_staging()

    manifest_by_id = manifest.set_index("audio_id")
    rows = []
    progress = u.Progress(len(meta), "remuestreo", every=50)
    for _, r in meta.iterrows():
        entry = process_recording(r, manifest_by_id.loc[r["audio_id"]])
        rows.append(entry)
        progress.step()

    report = pd.DataFrame(rows)
    n_fail = int((report["status"] == "FAIL").sum())
    print(f"\n  Procesados : {len(report)}")
    print(f"  OK         : {int((report['status'] == 'OK').sum())}")
    print(f"  FAIL       : {n_fail}")
    if n_fail:
        print("\n  Fallos:")
        for _, f in report.loc[report["status"] == "FAIL"].iterrows():
            print(f"    {f['audio_id']}: {f['error']}")

    return report, design_checks


def validate_structural(meta, report):
    """Comprobaciones estructurales sobre el staging, antes del reemplazo."""
    ok = report.loc[report["status"] == "OK"]

    on_disk = list(STAGING.rglob("*.wav"))
    header_bad = []
    for _, r in ok.iterrows():
        path = staging_path(r["dataset"], r["audio_path"])
        info = sf.info(str(path))
        if info.samplerate != cfg.TARGET_SR or info.channels != 1 or info.subtype != INTERIM_SUBTYPE:
            header_bad.append(r["audio_id"])

    untouched = ok.loc[~ok["transformed"]]
    identity_mismatches = []
    for _, r in untouched.iterrows():
        src_row = meta.loc[meta["audio_id"] == r["audio_id"]].iloc[0]
        x, _ = u.read_audio(src_row["abs_path"])
        y, _ = u.read_audio(staging_path(r["dataset"], r["audio_path"]))
        if not np.array_equal(x, y):
            identity_mismatches.append(r["audio_id"])

    return {
        "n_expected": len(meta), "n_processed": len(report),
        "n_ok": len(ok), "n_fail": int((report["status"] == "FAIL").sum()),
        "count_matches": len(meta) == len(report),
        "files_on_disk": len(on_disk), "files_match_ok_rows": len(on_disk) == len(ok),
        "has_nan_or_inf_any": bool(ok["has_nan_or_inf"].any()) if len(ok) else False,
        "exceeds_unity_count": int(ok["exceeds_unity"].sum()) if len(ok) else 0,
        "header_mismatches": header_bad,
        "untouched_total": len(untouched), "untouched_mismatches": identity_mismatches,
    }


def validate_spectral(meta):
    """Verificacion espectral: tonos sinteticos y muestra real estratificada."""
    tone_rows = []
    for rate in sorted(cfg.RESAMPLE_RATIOS):
        if (meta["sample_rate_hz"].astype(int) == rate).any():
            tone_rows.extend(verify_tones(rate))
    tones_df = pd.DataFrame(tone_rows)

    sample = stratified_real_audio_sample(meta)
    real_rows = []
    for _, r in sample.iterrows():
        source_sr = int(r["sample_rate_hz"])
        x, sr = u.read_audio(r["abs_path"])
        y, _ = u.read_audio(staging_path(r["dataset"], r["audio_path"]))

        n_in = min(x.size, sr * SPECTRAL_CHECK_SECONDS)
        n_out = min(y.size, cfg.TARGET_SR * SPECTRAL_CHECK_SECONDS)
        foldable = high_band_energy(x[:n_in], sr, cfg.TARGET_SR // 2)
        deviation, excess = spectral_comparison(x[:n_in], sr, y[:n_out], cfg.TARGET_SR)
        real_rows.append({
            "audio_id": r["audio_id"], "device": r["device"], "source_sr": source_sr,
            "foldable_energy_pct": 100 * foldable if not np.isnan(foldable) else np.nan,
            "passband_deviation_pct": 100 * deviation if not np.isnan(deviation) else np.nan,
            "max_excess_pct": 100 * excess if not np.isnan(excess) else np.nan,
        })
    real_df = pd.DataFrame(real_rows)

    return tones_df, real_df


# ---------------------------------------------------------------------------

def main():
    cfg.ensure_dirs()

    meta, manifest, n_not_eligible, n_review = admitted_recordings()

    u.section("FASE 2 - ESTANDARIZACION DE LA SENAL")
    print(f"  Grabaciones a procesar : {len(meta)}")
    if n_not_eligible:
        print(f"  No elegibles en fase 1 : {n_not_eligible}")
    if n_review:
        print(f"  Admitidas con REVIEW   : {n_review}")
    print(f"  Frecuencia de destino  : {cfg.TARGET_SR} Hz")
    print(f"  Formato intermedio     : coma flotante de 32 bits")

    report, design_checks = run(meta, manifest)

    u.section("VALIDACION DE LA FASE 2")
    struct = validate_structural(meta, report)
    tones_df, real_df = validate_spectral(meta)

    design_ok = all(c["passband_ok"] and c["stopband_ok"] for c in design_checks)
    tones_ok = bool(tones_df.empty or tones_df["ok"].all())
    real_ok = bool(real_df.empty or (real_df["max_excess_pct"].dropna() < 5.0).all())
    struct_ok = (
        struct["count_matches"] and struct["n_fail"] == 0
        and not struct["has_nan_or_inf_any"] and struct["files_match_ok_rows"]
        and not struct["header_mismatches"] and not struct["untouched_mismatches"]
    )
    verdict_ok = design_ok and tones_ok and real_ok and struct_ok

    print(f"  1. Diseno del filtro (freqz)      : [{'OK' if design_ok else 'REVISAR'}]")
    print(f"  2. Tonos puros ({len(tones_df)} pruebas)        : [{'OK' if tones_ok else 'REVISAR'}]")
    print(f"  3. Audio real estratificado ({len(real_df)})   : [{'OK' if real_ok else 'REVISAR'}]")
    print(f"  4. Estructura ({struct['n_processed']} archivos)        : [{'OK' if struct_ok else 'REVISAR'}]")
    print(f"     - conteo esperado/procesado  : {struct['n_expected']}/{struct['n_processed']}")
    print(f"     - fallos                     : {struct['n_fail']}")
    print(f"     - NaN/Inf                    : {'si' if struct['has_nan_or_inf_any'] else 'no'}")
    print(f"     - cabeceras incorrectas      : {len(struct['header_mismatches'])}")
    print(f"     - identidad ({struct['untouched_total']} sin transformar) : "
          f"{len(struct['untouched_mismatches'])} con diferencias")
    print(f"     - picos > 1.0 (advertencia, no fallo): {struct['exceeds_unity_count']}")

    # Informes de validacion: se escriben siempre, pasen o no.
    design_df = pd.DataFrame(design_checks)
    design_df.to_csv(cfg.REPORTS / "phase2_filter_design.csv", index=False)
    tones_df.to_csv(cfg.REPORTS / "phase2_tone_response.csv", index=False)
    real_df.to_csv(cfg.REPORTS / "phase2_spectral_check.csv", index=False)

    summary = pd.DataFrame([{
        "verdict": "PASS" if verdict_ok else "FAIL",
        "n_expected": struct["n_expected"], "n_processed": struct["n_processed"],
        "n_ok": struct["n_ok"], "n_fail": struct["n_fail"],
        "design_ok": design_ok, "tones_ok": tones_ok, "real_audio_ok": real_ok,
        "structural_ok": struct_ok,
        "untouched_total": struct["untouched_total"],
        "untouched_mismatches": len(struct["untouched_mismatches"]),
        "header_mismatches": len(struct["header_mismatches"]),
        "target_sr": cfg.TARGET_SR,
        "antialias_passband_hz": cfg.ANTIALIAS_PASSBAND_HZ,
        "antialias_stopband_hz": cfg.ANTIALIAS_STOPBAND_HZ,
        "antialias_min_attenuation_db": cfg.ANTIALIAS_MIN_ATTEN_DB,
        "antialias_passband_tol_db": cfg.ANTIALIAS_PASSBAND_TOL_DB,
        "scipy_version": scipy.__version__,
    }])
    summary.to_csv(cfg.REPORTS / "phase2_validation_summary.csv", index=False)

    if not verdict_ok:
        report.to_csv(cfg.REPORTS / "resampling_attempt_failed.csv", index=False)
        u.section("FASE 2 - VALIDACION FALLIDA, NO SE REEMPLAZA LA SALIDA")
        print(f"  La salida anterior en {cfg.RESAMPLED.relative_to(cfg.ROOT)} permanece intacta.")
        print(f"  Intento fallido registrado en reports/resampling_attempt_failed.csv")
        print(f"  Detalle de la causa en reports/phase2_validation_summary.csv")
        return {"report": report, "verdict": "FAIL", "summary": summary}

    swap_staging_into_place()
    report.to_csv(cfg.REPORTS / "resampling.csv", index=False)
    stale = cfg.REPORTS / "resampling_attempt_failed.csv"
    if stale.exists():
        stale.unlink()

    u.section("RESUMEN DE LA FASE 2")
    print(f"  Procesadas            : {len(report)}")
    print(f"  Remuestreadas         : {int(report['transformed'].sum())}")
    print(f"  Sin transformar       : {int((~report['transformed']).sum())}")
    print(f"  Ondulacion en banda   : {design_df['passband_ripple_db'].max():.4f} dB"
          f"  (tolerancia {cfg.ANTIALIAS_PASSBAND_TOL_DB} dB)")
    print(f"  Atenuacion minima     : {design_df['stopband_min_attenuation_db'].min():.2f} dB"
          f"  (minimo exigido {cfg.ANTIALIAS_MIN_ATTEN_DB} dB)")
    print(f"  Muestras alteradas    : {len(struct['untouched_mismatches'])}"
          f" de {struct['untouched_total']} sin transformar")
    print(f"  Salida                : {cfg.RESAMPLED.relative_to(cfg.ROOT)}")

    total_mb = sum(p.stat().st_size for p in cfg.RESAMPLED.rglob("*.wav")) / 1024 ** 2
    print(f"  Tamano en disco       : {total_mb:.1f} MB")
    print(f"  Veredicto             : PASS")

    return {"report": report, "verdict": "PASS", "summary": summary}


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["verdict"] == "PASS" else 1)
