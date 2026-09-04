"""
Fase 3 - Limpieza de senal.

  3a  Filtro pasa-banda
  3b  Denoising (sustraccion espectral)
  3c  Normalizacion de amplitud

Produce dos ramas comparables a partir de las 1249 grabaciones admitidas por
la fase 2:

  clean/no_dn/   pasa-banda + normalizacion (sin denoising)
  clean/dn/      pasa-banda + denoising + normalizacion

El documento exige comparar la senal con reduccion de ruido contra la senal
sin ella: esta fase no decide si el denoising se adopta, produce ambas ramas
y la evidencia para decidirlo despues, comparando el rendimiento de los
modelos entrenados sobre cada una.

Los cinco parametros del denoising y el objetivo de normalizacion se eligen
observando solo un subconjunto de calibracion (20 % de los pacientes,
separado por paciente y persistido como contrato en
reports/phase3/calibration_patients.csv), nunca el corpus completo: en esta
fase la particion train/test todavia no existe, de modo que la unica forma de
evitar que esos parametros se ajusten sobre datos que despues seran de
prueba es fijar ahora esa lista y que la particion posterior la respete.

Modo de uso, en dos pasadas:

    python phase3_cleaning.py --calibrar   # mide y recomienda; no escribe audio
    python phase3_cleaning.py              # ejecuta el corpus completo

La segunda pasada se niega a correr si algun parametro de config.py sigue en
None.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, istft, sosfiltfilt, sosfreqz, stft

import config as cfg
import utils as u


INTERIM_SUBTYPE = "FLOAT"
STAGING = u.staging_dir_for(cfg.CLEAN)

# Recordings usadas para la prueba nula del emparejamiento STFT/ISTFT.
NULL_TEST_SAMPLES = 8
NULL_TEST_TOLERANCE = 1e-9

# Valores centrales usados mientras se explora la resolucion de la STFT, antes
# de que la agresividad tenga su propio barrido.
CENTRAL_NOISE_PCT = 10
CENTRAL_OVERSUBTRACTION = 1.5
CENTRAL_SPECTRAL_FLOOR = 0.01

# Una grabacion de ICBHI sirve como referencia de ruido real si al menos este
# margen de su duracion queda fuera de los ciclos anotados, con al menos un
# segundo a cada lado para que la medicion tenga sustento.
ANNOTATION_GAP_FRACTION = 0.05
MIN_SECONDS_PER_SIDE = 1.0
MIN_EVENT_SAMPLES = 10      # muestras minimas para medir un evento adventicio


# ---------------------------------------------------------------------------
# Entrada: salida admitida de la fase 2
# ---------------------------------------------------------------------------

def admitted_recordings():
    """Grabaciones de la fase 2 en estado OK.

    Se parte del informe de la fase 2 y no del manifiesto de la fase 1,
    porque es el que describe el audio que realmente existe en resampled/:
    ya incorpora el filtro anti-aliasing y esta a 4 kHz.
    """
    path = cfg.R2_RESAMPLING
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecute primero phase2_standardization.py."
        )
    report = pd.read_csv(path, dtype={"audio_id": str})
    meta = report.loc[report["status"] == "OK"].reset_index(drop=True)
    n_bad = len(report) - len(meta)
    if n_bad:
        print(f"  Aviso: {n_bad} grabaciones de la fase 2 no estan en estado OK; se omiten.")
    return meta


# ---------------------------------------------------------------------------
# Subconjunto de calibracion
# ---------------------------------------------------------------------------

def build_calibration_set(meta):
    """Selecciona el 20 % de los pacientes, estratificado por dataset y
    diagnostico, con semilla fija, y lo persiste como contrato.

    Determinista: el mismo corpus admitido y la misma semilla producen
    siempre la misma lista. No hace falta distinguir entre generarla de
    nuevo o leerla de un archivo previo.
    """
    patients = (
        meta.drop_duplicates("patient_uid")[["patient_uid", "dataset", "diagnosis"]]
        .sort_values("patient_uid")
        .reset_index(drop=True)
    )
    counts = meta["patient_uid"].value_counts()

    rng = np.random.RandomState(cfg.CALIBRATION_SEED)
    chosen = []
    for (_, _), group in patients.groupby(["dataset", "diagnosis"], sort=True):
        ids = sorted(group["patient_uid"])
        n = int(round(len(ids) * cfg.CALIBRATION_FRACTION))
        if n:
            chosen.extend(rng.choice(ids, size=n, replace=False).tolist())

    chosen_set = set(chosen)
    table = patients.copy()
    table["is_calibration"] = table["patient_uid"].isin(chosen_set)
    table["n_recordings"] = table["patient_uid"].map(counts).fillna(0).astype(int)
    table["calibration_fraction"] = cfg.CALIBRATION_FRACTION
    table["calibration_seed"] = cfg.CALIBRATION_SEED
    table = table.sort_values(
        ["is_calibration", "dataset", "diagnosis", "patient_uid"],
        ascending=[False, True, True, True],
    )
    table.to_csv(cfg.R3_CALIBRATION, index=False)

    calib = table.loc[table["is_calibration"]]
    print(f"  Pacientes totales          : {len(patients)}")
    print(f"  Pacientes de calibracion   : {len(calib)}"
          f"  ({100 * len(calib) / len(patients):.1f} %)")
    print(f"  Grabaciones de calibracion : {int(calib['n_recordings'].sum())}"
          f"  de {len(meta)}")
    print(f"  -> {cfg.R3_CALIBRATION.relative_to(cfg.ROOT)}")

    return chosen_set


def load_calibration_set():
    """Lee el contrato de calibracion ya persistido."""
    if not cfg.R3_CALIBRATION.exists():
        raise FileNotFoundError(
            f"No existe {cfg.R3_CALIBRATION}. Ejecute primero "
            "'python phase3_cleaning.py --calibrar'."
        )
    table = pd.read_csv(cfg.R3_CALIBRATION, dtype={"patient_uid": str})
    return set(table.loc[table["is_calibration"], "patient_uid"])


# ---------------------------------------------------------------------------
# Ciclos respiratorios anotados (referencia de ruido real, solo ICBHI)
# ---------------------------------------------------------------------------

_CYCLES = None
_CYCLE_SUMMARY = None


def _load_cycles():
    global _CYCLES, _CYCLE_SUMMARY
    if _CYCLES is None:
        _CYCLES = pd.read_csv(cfg.R1_CYCLES, dtype={"audio_id": str})
        _CYCLE_SUMMARY = pd.read_csv(cfg.R1_CYCLE_SUMMARY, dtype={"audio_id": str})
    return _CYCLES, _CYCLE_SUMMARY


def _cycle_intervals(audio_id, only=None):
    """Intervalos [inicio, fin) en muestras de los ciclos anotados.

    `only` restringe a los ciclos que llevan un evento adventicio concreto:
    "crackles" o "wheezes". Un ciclo con ambos aparece en las dos listas.
    """
    cycles, _ = _load_cycles()
    rows = cycles.loc[cycles["audio_id"] == audio_id]
    if only is not None:
        rows = rows.loc[rows[only] == 1]
    sr = cfg.TARGET_SR
    return [(int(round(c["start_s"] * sr)), int(round(c["end_s"] * sr)))
            for _, c in rows.iterrows()]


def _mask_from_intervals(intervals, n_samples, pad=0):
    mask = np.zeros(n_samples, dtype=bool)
    for a, b in intervals:
        lo = max(0, a - pad)
        hi = min(n_samples, b + pad)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def cycle_mask(audio_id, n_samples):
    """True donde hay un ciclo respiratorio anotado."""
    return _mask_from_intervals(_cycle_intervals(audio_id), n_samples)


def crackles_mask(audio_id, n_samples):
    """True donde el ciclo anotado tiene crepitantes."""
    return _mask_from_intervals(_cycle_intervals(audio_id, "crackles"), n_samples)


def wheezes_mask(audio_id, n_samples):
    """True donde el ciclo anotado tiene sibilancias.

    Se evalua por separado de los crepitantes porque los dos eventos no son
    igual de fragiles: las sibilancias son tonales y concentran energia en
    bandas estrechas, mientras que los crepitantes son transitorios de banda
    ancha y de 5-20 ms, mucho mas sensibles a cualquier procesado espectral.
    """
    return _mask_from_intervals(_cycle_intervals(audio_id, "wheezes"), n_samples)


def noise_mask(audio_id, n_samples):
    """True en el hueco entre ciclos, excluyendo un colchon a cada lado.

    Los limites anotados son manuales y el sonido respiratorio no empieza ni
    termina de golpe: las muestras contiguas a un ciclo contienen ataque o
    caida del propio sonido. Incluirlas en la referencia de ruido la
    contamina con senal.
    """
    guard = int(round(cfg.CYCLE_GUARD_MS * cfg.TARGET_SR / 1000.0))
    return ~_mask_from_intervals(_cycle_intervals(audio_id), n_samples, pad=guard)


def annotation_gap_recordings():
    """audio_id de ICBHI con hueco suficiente entre ciclos anotados."""
    _, summary = _load_cycles()
    threshold = 100 * (1 - ANNOTATION_GAP_FRACTION)
    return set(summary.loc[summary["coverage_pct"] < threshold, "audio_id"])


def cycle_gap_snr_proxy_db(x, signal_mask, gap_mask):
    """Cociente de energia entre los ciclos anotados y el hueco entre ellos.

        cycle_gap_snr_proxy_db = 10 * log10( <x^2>_ciclo / <x^2>_hueco )

    donde <>_ciclo promedia sobre las muestras dentro de un ciclo anotado y
    <>_hueco sobre las que estan a mas de CYCLE_GUARD_MS de cualquier limite
    de ciclo.

    **No es una SNR real y no debe llamarse asi.** El hueco entre ciclos no
    garantiza ruido puro: puede contener sonido cardiaco, movimiento o
    respiracion no anotada. Verificado en el corpus: algunas grabaciones dan
    valores negativos con correlacion intacta (0.98), lo que indica que el
    hueco tenia mas energia que el propio ciclo, no que el denoising fallara.

    Su valor esta en que el denominador procede de una region distinta de la
    senal y no de la propia distribucion que se evalua, de modo que -a
    diferencia de la SNR por percentiles- no crece mecanicamente al aumentar
    la agresividad de la sustraccion. Por eso sirve para elegir, aunque su
    magnitud absoluta no sea interpretable como relacion senal-ruido.
    """
    min_samples = int(MIN_SECONDS_PER_SIDE * cfg.TARGET_SR)
    if signal_mask.sum() < min_samples or gap_mask.sum() < min_samples:
        return float("nan")
    signal_energy = float(np.mean(x[signal_mask] ** 2))
    gap_energy = float(np.mean(x[gap_mask] ** 2))
    if signal_energy <= 0 or gap_energy <= 0:
        return float("nan")
    return float(10 * np.log10(signal_energy / gap_energy))


def correlation_in_mask(x_before, x_after, mask):
    """Correlacion entre la senal antes y despues, restringida a una mascara."""
    if mask.sum() < MIN_EVENT_SAMPLES:
        return float("nan")
    a, b = x_before[mask], x_after[mask]
    if np.std(a) <= 0 or np.std(b) <= 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ---------------------------------------------------------------------------
# 3a - Pasa-banda
# ---------------------------------------------------------------------------

_BANDPASS_SOS = None


def bandpass_sos():
    global _BANDPASS_SOS
    if _BANDPASS_SOS is None:
        _BANDPASS_SOS = butter(cfg.BANDPASS_ORDER, [cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH],
                                btype="band", fs=cfg.TARGET_SR, output="sos")
    return _BANDPASS_SOS


def apply_bandpass(x):
    """Butterworth de orden 4 entre 50 y 1800 Hz, en fase cero.

    sosfiltfilt no desplaza la senal en el tiempo: es un requisito, porque un
    desfase invalidaria las marcas de tiempo de los 6898 ciclos anotados.
    """
    if x.size == 0:
        return x
    return sosfiltfilt(bandpass_sos(), x)


def verify_bandpass_design():
    """Respuesta del pasa-banda de una pasada y de las dos que aplica
    sosfiltfilt, medida con sosfreqz y no asumida.

    sosfiltfilt filtra en ambos sentidos: la respuesta efectiva es |H|^2, de
    modo que la atenuacion real en los bordes nominales (50 y 1800 Hz) es de
    6 dB y no de 3. Los puntos de -3 dB efectivos caen dentro de la banda
    nominal.
    """
    sos = bandpass_sos()
    freqs = np.linspace(1, cfg.TARGET_SR / 2 - 1, 4000)
    w, h = sosfreqz(sos, worN=freqs, fs=cfg.TARGET_SR)
    db_one = 20 * np.log10(np.abs(h) + 1e-300)
    db_two = 2 * db_one

    def at_two(f):
        return float(db_two[np.searchsorted(w, f)])

    center = (cfg.BANDPASS_LOW + cfg.BANDPASS_HIGH) / 2

    def crossing(low_side):
        side = (w < center) if low_side else (w > center)
        idx = np.where(side & (db_two <= -3.0))[0]
        if not idx.size:
            return float("nan")
        return float(w[idx.max()]) if low_side else float(w[idx.min()])

    ripple_band = (w >= 100) & (w <= 1500)
    ripple_db = float(db_two[ripple_band].max() - db_two[ripple_band].min())

    return {
        "order": cfg.BANDPASS_ORDER, "low_hz": cfg.BANDPASS_LOW, "high_hz": cfg.BANDPASS_HIGH,
        "gain_one_pass_50hz_db": round(at_two(cfg.BANDPASS_LOW) / 2, 3),
        "gain_one_pass_1800hz_db": round(at_two(cfg.BANDPASS_HIGH) / 2, 3),
        "gain_two_pass_50hz_db": round(at_two(cfg.BANDPASS_LOW), 3),
        "gain_two_pass_1800hz_db": round(at_two(cfg.BANDPASS_HIGH), 3),
        "effective_minus3db_low_hz": round(crossing(True), 2),
        "effective_minus3db_high_hz": round(crossing(False), 2),
        "ripple_100_1500hz_db": round(ripple_db, 4),
        "ripple_ok": ripple_db <= cfg.BANDPASS_RIPPLE_TOL_DB,
        "attenuation_20hz_db": round(at_two(20), 2),
        "attenuation_1950hz_db": round(at_two(1950), 2),
    }


def verify_bandpass_zero_phase():
    """Retardo medido, no asumido: correlacion cruzada entre ruido blanco y
    su version filtrada. Debe ser exactamente 0 muestras."""
    rng = np.random.RandomState(0)
    x = rng.randn(cfg.TARGET_SR)
    y = apply_bandpass(x)
    margin = 200
    xc, yc = x[margin:-margin], y[margin:-margin]
    corr = np.correlate(yc - yc.mean(), xc - xc.mean(), mode="full")
    return int(np.argmax(corr) - (len(xc) - 1))


def band_energy_fractions(x):
    """Fraccion de energia bajo 50 Hz, en banda (50-1800) y sobre 1800 Hz."""
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / cfg.TARGET_SR)
    total = spectrum.sum()
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    below = float(spectrum[freqs < cfg.BANDPASS_LOW].sum() / total)
    inband = float(spectrum[(freqs >= cfg.BANDPASS_LOW) & (freqs <= cfg.BANDPASS_HIGH)].sum() / total)
    above = float(spectrum[freqs > cfg.BANDPASS_HIGH].sum() / total)
    return below, inband, above


# ---------------------------------------------------------------------------
# 3b - Denoising (sustraccion espectral)
# ---------------------------------------------------------------------------

def spectral_subtract(x, nperseg, noverlap, noise_pct, alpha, beta):
    """Sustraccion espectral con suelo proporcional.

    El ruido se estima por banda como un percentil bajo de su potencia a lo
    largo de toda la grabacion. El suelo es proporcional (beta*|X|^2) y no
    absoluto: escala con la energia local en vez de imponer un piso plano,
    que es la formulacion que menos ruido musical genera. La salida se
    recorta a la longitud de entrada: el analisis y la sintesis de scipy
    pueden anadir relleno al final.
    """
    _, _, X = stft(x, fs=cfg.TARGET_SR, window=cfg.STFT_WINDOW,
                    nperseg=nperseg, noverlap=noverlap, boundary="zeros", padded=True)
    power = np.abs(X) ** 2
    noise = np.percentile(power, noise_pct, axis=1, keepdims=True)
    subtracted = np.maximum(power - alpha * noise, beta * power)
    Y = np.sqrt(subtracted) * np.exp(1j * np.angle(X))
    _, y = istft(Y, fs=cfg.TARGET_SR, window=cfg.STFT_WINDOW,
                 nperseg=nperseg, noverlap=noverlap, boundary=True)
    return y[: x.size]


def compute_stft(x, nperseg, noverlap):
    """STFT con los parametros de la fase, devolviendo tambien las frecuencias.

    Se calcula una sola vez por senal y se reparte entre las metricas que la
    necesitan, en vez de que cada una la recalcule por su cuenta.
    """
    freqs, _, X = stft(x, fs=cfg.TARGET_SR, window=cfg.STFT_WINDOW,
                        nperseg=nperseg, noverlap=noverlap,
                        boundary="zeros", padded=True)
    return freqs, X


def musical_noise_ratio(X_before, X_after):
    """Razon de curtosis del espectro de magnitud, despues frente a antes.

    La sustraccion espectral agresiva deja picos aislados en el
    espectrograma (ruido musical), que se manifiestan como un exceso de
    curtosis en la distribucion de magnitudes por trama. Un valor > 1
    indica su aparicion.
    """
    def kurtosis(X):
        mag = np.abs(X).ravel()
        mag = mag[mag > 0]
        if mag.size < 4:
            return float("nan")
        std = mag.std()
        if std <= 0:
            return float("nan")
        return float(np.mean(((mag - mag.mean()) / std) ** 4))

    before, after = kurtosis(X_before), kurtosis(X_after)
    if not (np.isfinite(before) and np.isfinite(after)) or before <= 0:
        return float("nan")
    return after / before


def spectral_distortion(freqs, X_before, X_after, band=None):
    """Distancia log-espectral RMS, restringida a la banda util.

    Se limita a 50-1800 Hz y no se calcula sobre el espectro completo: fuera
    de esa banda la senal ya paso por el pasa-banda y sus bins estan
    practicamente vacios, de modo que aportan diferencias nulas que diluyen
    la media y hacen parecer menor la distorsion donde de verdad importa.
    """
    lo, hi = (cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH) if band is None else band
    in_band = (freqs >= lo) & (freqs <= hi)
    if not in_band.any():
        return float("nan")
    n = min(X_before.shape[1], X_after.shape[1])
    p0 = np.abs(X_before[in_band, :n]) ** 2
    p1 = np.abs(X_after[in_band, :n]) ** 2
    floor = 1e-12 * max(float(p0.max()), 1e-12)
    diff_db = 10 * np.log10((p1 + floor) / (p0 + floor))
    return float(np.sqrt(np.mean(diff_db ** 2)))


def verify_null_denoising(sample_signals, nperseg, noverlap):
    """Con alpha=0 y beta=1 la cadena STFT->sustraccion->ISTFT debe devolver
    la entrada con error de precision de maquina.

    Si no lo hace, el emparejamiento ventana/salto/reconstruccion esta roto
    y todo lo que se mida despues es un artefacto, no el efecto real del
    denoising. Debe correr antes que cualquier barrido.
    """
    worst = 0.0
    for x in sample_signals:
        y = spectral_subtract(x, nperseg, noverlap, noise_pct=10, alpha=0.0, beta=1.0)
        worst = max(worst, float(np.max(np.abs(x - y))))
    return worst


def recording_denoising_metrics(audio_id, x_bp, x_dn, nperseg, noverlap, sr=None):
    """Metricas de una sola grabacion, antes frente a despues del denoising.

    Las que dependen de anotaciones quedan en NaN para Fraiwan, que no las
    tiene. Se usa tanto en el barrido de calibracion -agregando entre
    grabaciones- como en la ejecucion completa, donde se guarda una fila por
    grabacion.
    """
    sr = cfg.TARGET_SR if sr is None else sr
    freqs, X_bp = compute_stft(x_bp, nperseg, noverlap)
    _, X_dn = compute_stft(x_dn, nperseg, noverlap)

    signal = cycle_mask(audio_id, x_bp.size)
    gap = noise_mask(audio_id, x_bp.size)
    crackles = crackles_mask(audio_id, x_bp.size)
    wheezes = wheezes_mask(audio_id, x_bp.size)

    proxy_before, _ = u.snr_proxy_db(x_bp, sr)
    proxy_after, _ = u.snr_proxy_db(x_dn, sr)
    proxy_delta = (proxy_after - proxy_before
                    if np.isfinite(proxy_before) and np.isfinite(proxy_after)
                    else float("nan"))

    return {
        "cycle_gap_snr_proxy_db": cycle_gap_snr_proxy_db(x_dn, signal, gap),
        "cycle_correlation": correlation_in_mask(x_bp, x_dn, signal),
        "crackle_correlation": (correlation_in_mask(x_bp, x_dn, crackles)
                                 if crackles.sum() >= MIN_EVENT_SAMPLES else float("nan")),
        "wheeze_correlation": (correlation_in_mask(x_bp, x_dn, wheezes)
                                if wheezes.sum() >= MIN_EVENT_SAMPLES else float("nan")),
        "musical_noise_ratio": musical_noise_ratio(X_bp, X_dn),
        "spectral_distortion_db": spectral_distortion(freqs, X_bp, X_dn),
        "snr_proxy_delta_db": proxy_delta,
    }


def _aggregate(values, low_pct=10, high_pct=90):
    """Media, percentiles y extremos de una metrica entre grabaciones."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not arr.size:
        nan = float("nan")
        return nan, nan, nan, nan, nan, 0
    return (float(arr.mean()), float(np.percentile(arr, low_pct)),
            float(np.percentile(arr, high_pct)), float(arr.min()),
            float(arr.max()), int(arr.size))


def _evaluate_denoising(gap_meta, nperseg, noverlap, noise_pct, alpha, beta):
    """Barrido sobre las grabaciones con hueco anotado del subconjunto de
    calibracion, agregando cada metrica por media y por percentiles.

    El peor caso se reporta aparte de la media: verificado en el corpus, una
    correlacion media de 0.973 puede convivir con grabaciones concretas en
    0.84, y una regla que solo mire la media no lo ve.
    """
    collected = {k: [] for k in (
        "cycle_gap_snr_proxy_db", "cycle_correlation", "crackle_correlation",
        "wheeze_correlation", "musical_noise_ratio", "spectral_distortion_db",
        "snr_proxy_delta_db",
    )}

    for _, r in gap_meta.iterrows():
        x, sr = u.read_audio(Path(cfg.ROOT) / r["output_path"])
        x_bp = apply_bandpass(x)
        x_dn = spectral_subtract(x_bp, nperseg, noverlap, noise_pct, alpha, beta)
        metrics = recording_denoising_metrics(r["audio_id"], x_bp, x_dn,
                                               nperseg, noverlap, sr)
        for key, value in metrics.items():
            collected[key].append(value)

    snr_mean, _, _, snr_min, _, snr_n = _aggregate(collected["cycle_gap_snr_proxy_db"])
    cyc_mean, cyc_p10, _, cyc_min, _, _ = _aggregate(collected["cycle_correlation"])
    cra_mean, cra_p10, _, cra_min, _, cra_n = _aggregate(collected["crackle_correlation"])
    whe_mean, whe_p10, _, whe_min, _, whe_n = _aggregate(collected["wheeze_correlation"])
    mus_mean, _, mus_p90, _, mus_max, _ = _aggregate(collected["musical_noise_ratio"])
    dis_mean, _, dis_p90, _, _, _ = _aggregate(collected["spectral_distortion_db"])
    pro_mean, _, _, _, _, _ = _aggregate(collected["snr_proxy_delta_db"])

    return {
        "n_recordings": len(gap_meta), "n_snr": snr_n,
        "n_crackle_recordings": cra_n, "n_wheeze_recordings": whe_n,
        "cycle_gap_snr_proxy_db": snr_mean, "cycle_gap_snr_proxy_db_min": snr_min,
        "cycle_correlation": cyc_mean, "cycle_correlation_p10": cyc_p10,
        "cycle_correlation_min": cyc_min,
        "crackle_correlation": cra_mean, "crackle_correlation_p10": cra_p10,
        "crackle_correlation_min": cra_min,
        "wheeze_correlation": whe_mean, "wheeze_correlation_p10": whe_p10,
        "wheeze_correlation_min": whe_min,
        "musical_noise_ratio": mus_mean, "musical_noise_ratio_p90": mus_p90,
        "musical_noise_ratio_max": mus_max,
        "spectral_distortion_db": dis_mean, "spectral_distortion_db_p90": dis_p90,
        "snr_proxy_delta_db": pro_mean,
    }


def satisfies_constraints(row):
    """Restricciones sobre la media y sobre el percentil, no solo la media."""
    return (row["cycle_correlation"] >= cfg.MIN_CYCLE_CORRELATION
            and row["cycle_correlation_p10"] >= cfg.MIN_CYCLE_CORRELATION_P10
            and row["musical_noise_ratio"] <= cfg.MAX_MUSICAL_NOISE_RATIO
            and row["musical_noise_ratio_p90"] <= cfg.MAX_MUSICAL_NOISE_RATIO_P90)


def dn_reliability(metrics):
    """Marca si la rama con denoising es fiable para una grabacion concreta.

    Reutiliza los mismos umbrales que la regla de seleccion, pero aplicados a
    la grabacion individual en vez de al agregado: una configuracion puede
    cumplir de sobra en promedio y aun asi destrozar casos concretos. Medido
    sobre el corpus, la estimacion de ruido por percentil bajo se degrada
    cuando el sonido respiratorio es casi continuo, porque entonces ese
    percentil ya no es ruido sino senal, y sustraerlo multiplicado por alpha
    retira contenido real.

    No excluye nada: la rama no_dn conserva esas grabaciones intactas y la
    decision de que hacer con ellas corresponde a la etapa de modelado.
    """
    reasons = []
    correlation = metrics.get("cycle_correlation", float("nan"))
    musical = metrics.get("musical_noise_ratio", float("nan"))
    if np.isfinite(correlation) and correlation < cfg.MIN_CYCLE_CORRELATION:
        reasons.append("LOW_CYCLE_CORRELATION")
    if np.isfinite(musical) and musical > cfg.MAX_MUSICAL_NOISE_RATIO_P90:
        reasons.append("HIGH_MUSICAL_NOISE")
    return (not reasons), ";".join(reasons)


def sweep_stft_resolution(calib_meta):
    """Barrido de ventana y salto, con agresividad fija en valores centrales."""
    gap_ids = annotation_gap_recordings()
    gap_meta = calib_meta.loc[
        (calib_meta["dataset"] == "ICBHI") & calib_meta["audio_id"].isin(gap_ids)
    ]
    print(f"  Grabaciones con hueco anotado (calibracion): {len(gap_meta)}")

    rows = []
    for nperseg in cfg.STFT_CANDIDATES:
        for frac in cfg.OVERLAP_FRACTION_CANDIDATES:
            noverlap = int(round(nperseg * frac))
            metrics = _evaluate_denoising(
                gap_meta, nperseg, noverlap,
                CENTRAL_NOISE_PCT, CENTRAL_OVERSUBTRACTION, CENTRAL_SPECTRAL_FLOOR,
            )
            rows.append({
                "nperseg": nperseg, "noverlap": noverlap, "overlap_fraction": frac,
                "window_ms": round(1000 * nperseg / cfg.TARGET_SR, 1),
                **metrics,
            })
    return pd.DataFrame(rows)


def sweep_denoising(calib_meta, nperseg, noverlap):
    """Barrido en dos tramos: agresividad primero, suelo despues, con la
    resolucion ya elegida."""
    gap_ids = annotation_gap_recordings()
    gap_meta = calib_meta.loc[
        (calib_meta["dataset"] == "ICBHI") & calib_meta["audio_id"].isin(gap_ids)
    ]
    print(f"  Grabaciones con hueco anotado (calibracion): {len(gap_meta)}")

    rows = []
    print("  Tramo 1: percentil de ruido x sobre-sustraccion (beta fijo)")
    for pct in cfg.NOISE_PCT_CANDIDATES:
        for alpha in cfg.OVERSUBTRACTION_CANDIDATES:
            metrics = _evaluate_denoising(gap_meta, nperseg, noverlap, pct, alpha,
                                           CENTRAL_SPECTRAL_FLOOR)
            rows.append({"stage": "agresividad", "noise_pct": pct, "oversubtraction": alpha,
                         "spectral_floor": CENTRAL_SPECTRAL_FLOOR, **metrics})

    stage1 = pd.DataFrame(rows)
    valid = stage1.loc[stage1.apply(satisfies_constraints, axis=1)]
    ranked = valid if len(valid) else stage1
    top = ranked.sort_values("cycle_gap_snr_proxy_db", ascending=False).iloc[0]
    pct_chosen, alpha_chosen = top["noise_pct"], top["oversubtraction"]
    print(f"  -> percentil={pct_chosen}  alpha={alpha_chosen}"
          f"  {'(cumple las restricciones)' if len(valid) else '(NINGUNA config. las cumple)'}")

    print("  Tramo 2: suelo espectral, con percentil y alpha ya elegidos")
    for beta in cfg.SPECTRAL_FLOOR_CANDIDATES:
        metrics = _evaluate_denoising(gap_meta, nperseg, noverlap, pct_chosen, alpha_chosen, beta)
        rows.append({"stage": "suelo", "noise_pct": pct_chosen, "oversubtraction": alpha_chosen,
                     "spectral_floor": beta, **metrics})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3c - Normalizacion de amplitud
# ---------------------------------------------------------------------------

def compute_rms_distribution(calib_meta):
    """RMS y pico tras el pasa-banda, sobre el subconjunto de calibracion.

    El objetivo de normalizacion se fija sobre esta distribucion y no sobre
    la de la senal sin filtrar: el pasa-banda reduce el RMS por un factor
    mediano de ~0.25, de modo que el nivel actual no es representativo del
    que tendra la senal que realmente se normaliza.
    """
    rows = []
    for _, r in calib_meta.iterrows():
        x, _ = u.read_audio(Path(cfg.ROOT) / r["output_path"])
        x_bp = apply_bandpass(x)
        rows.append({
            "audio_id": r["audio_id"], "dataset": r["dataset"], "device": r["device"],
            "rms_post_bandpass": u.rms(x_bp),
            "peak_post_bandpass": float(np.max(np.abs(x_bp))) if x_bp.size else 0.0,
        })
    return pd.DataFrame(rows)


def normalize(x, target_rms, peak_ceiling, max_gain):
    """Normalizacion RMS con dos topes: el pico no puede rebasar
    peak_ceiling y la ganancia no puede rebasar max_gain.

    El pico por si solo no basta: grabaciones con RMS y pico bajos a la vez
    (verificado: la mayoria del RMS post-filtrado, con crest factor medio
    alto) reciben del techo de pico permiso para amplificarse mucho mas de
    lo que el objetivo de RMS pide, lo que amplificaria sobre todo su ruido.
    """
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    cur_rms = u.rms(x)
    if cur_rms <= 0:
        return x, 0.0, "zero_signal"

    candidates = [
        ("target_rms", target_rms / cur_rms),
        ("peak_ceiling", peak_ceiling / peak if peak > 0 else float("inf")),
        ("max_gain", float(max_gain)),
    ]
    limiter, gain = min(candidates, key=lambda kv: kv[1])
    return x * gain, float(gain), limiter


# ---------------------------------------------------------------------------
# Rutas de salida
# ---------------------------------------------------------------------------

def relative_output(dataset, audio_path):
    relative = audio_path.split("audio/", 1)[-1]
    return Path(dataset) / relative


def branch_staging_path(branch, dataset, audio_path):
    return STAGING / branch / relative_output(dataset, audio_path)


def branch_final_path(branch, dataset, audio_path):
    base = cfg.CLEAN_NO_DN if branch == "no_dn" else cfg.CLEAN_DN
    return base / relative_output(dataset, audio_path)


# ---------------------------------------------------------------------------
# Procesamiento por grabacion (ambas ramas)
# ---------------------------------------------------------------------------

def process_recording_both_branches(row, calibration_ids, params):
    """Aplica el pasa-banda, bifurca, aplica denoising en una rama, normaliza
    ambas y escribe al staging.

    Devuelve las dos filas de auditoria del manifiesto, la fila de energia
    por banda y la fila de metricas del denoising. Estas ultimas se calculan
    sobre el corpus completo y no solo sobre la calibracion: es lo que
    permite localizar despues las grabaciones concretas que el denoising
    altera mas, en vez de conocer solo el promedio.
    """
    audio_id = row["audio_id"]
    dataset, audio_path = row["dataset"], row["audio_path"]
    source_path = Path(cfg.ROOT) / row["output_path"]
    is_calibration = row["patient_uid"] in calibration_ids

    base = {
        "audio_id": audio_id, "dataset": dataset, "audio_path": audio_path,
        "patient_uid": row["patient_uid"],
        "diagnosis": row["diagnosis"], "device": row["device"], "filter": row["filter"],
        "zone": row["zone"], "quality_status": row["quality_status"],
        "quality_reasons": row["quality_reasons"], "calibration_patient": is_calibration,
    }
    energy_row = dict(base)
    metrics_row = dict(base)
    branch_rows = []

    try:
        source_sha = u.file_sha256(source_path)
        expected_sha = str(row["output_sha256"])
        if source_sha != expected_sha:
            raise ValueError(
                f"SHA-256 de la salida de fase 2 no coincide "
                f"({source_sha[:12]}... vs {expected_sha[:12]}...)"
            )

        x, sr = u.read_audio(source_path)
        if sr != cfg.TARGET_SR:
            raise ValueError(f"frecuencia inesperada {sr} Hz, se esperaba {cfg.TARGET_SR}")

        x_bp = apply_bandpass(x)
        below0, inband0, above0 = band_energy_fractions(x)
        below1, inband1, above1 = band_energy_fractions(x_bp)
        # El umbral se evalua sobre inband0 (fraccion de la energia ORIGINAL
        # que sobrevive al filtro), no sobre inband1: la del filtrado es casi
        # 100 % por construccion, ya que el pasa-banda ya retiro lo demas.
        low_energy = (cfg.MIN_INBAND_ENERGY_PCT is not None
                       and 100 * inband0 < cfg.MIN_INBAND_ENERGY_PCT)
        energy_row.update({
            "samples": int(x.size), "duration_s": x.size / cfg.TARGET_SR,
            "energy_below50_pct_before": 100 * below0, "energy_inband_pct_before": 100 * inband0,
            "energy_above1800_pct_before": 100 * above0,
            "energy_below50_pct_after": 100 * below1, "energy_inband_pct_after": 100 * inband1,
            "energy_above1800_pct_after": 100 * above1,
            "low_inband_energy": bool(low_energy), "status": "OK", "error": "",
        })

        x_dn = spectral_subtract(x_bp, params["nperseg"], params["noverlap"],
                                  params["noise_pct"], params["oversubtraction"],
                                  params["spectral_floor"])

        metrics = recording_denoising_metrics(
            audio_id, x_bp, x_dn, params["nperseg"], params["noverlap"], sr)
        dn_reliable, dn_reason = dn_reliability(metrics)
        metrics_row.update(metrics)
        metrics_row.update({"dn_reliable": dn_reliable, "dn_flag_reason": dn_reason,
                             "status": "OK", "error": ""})

        for branch, signal in (("no_dn", x_bp), ("dn", x_dn)):
            if signal.size != x.size:
                raise ValueError(
                    f"rama {branch}: {signal.size} muestras, se esperaban {x.size}"
                )
            y, gain, limiter = normalize(signal, params["target_rms"],
                                          cfg.PEAK_CEILING, params["max_gain"])
            dest = branch_staging_path(branch, dataset, audio_path)
            u.write_atomic(dest, y, cfg.TARGET_SR, INTERIM_SUBTYPE)
            out_sha = u.file_sha256(dest)
            branch_rows.append({
                **base, "branch": branch,
                "output_path": branch_final_path(branch, dataset, audio_path)
                    .relative_to(cfg.ROOT).as_posix(),
                "output_sha256": out_sha,
                "samples": int(y.size), "duration_s": y.size / cfg.TARGET_SR,
                "rms_before_norm": u.rms(signal),
                "peak_before_norm": float(np.max(np.abs(signal))) if signal.size else 0.0,
                "gain": gain, "limiter": limiter,
                "rms_final": u.rms(y), "peak_final": float(np.max(np.abs(y))) if y.size else 0.0,
                "target_rms": params["target_rms"],
                "nperseg": params["nperseg"] if branch == "dn" else "",
                "noverlap": params["noverlap"] if branch == "dn" else "",
                "noise_pct": params["noise_pct"] if branch == "dn" else "",
                "oversubtraction": params["oversubtraction"] if branch == "dn" else "",
                "spectral_floor": params["spectral_floor"] if branch == "dn" else "",
                # Solo la rama con denoising puede ser poco fiable; la otra no
                # pasa por la sustraccion espectral.
                "dn_reliable": dn_reliable if branch == "dn" else True,
                "dn_flag_reason": dn_reason if branch == "dn" else "",
                "has_nan_or_inf": bool(not np.isfinite(y).all()),
                "status": "OK", "error": "",
            })
    except Exception as exc:
        energy_row.update({"status": "FAIL", "error": str(exc)})
        metrics_row.update({"status": "FAIL", "error": str(exc)})
        for branch in ("no_dn", "dn"):
            branch_rows.append({**base, "branch": branch, "status": "FAIL", "error": str(exc)})

    return branch_rows, energy_row, metrics_row


# ---------------------------------------------------------------------------
# Modo calibracion
# ---------------------------------------------------------------------------

def run_calibration():
    u.section("FASE 3 - CALIBRACION (recomienda; no procesa el corpus completo)")
    meta = admitted_recordings()
    print(f"  Grabaciones elegibles de la fase 2: {len(meta)}")

    u.section("SUBCONJUNTO DE CALIBRACION")
    calibration_ids = build_calibration_set(meta)
    calib_meta = meta.loc[meta["patient_uid"].isin(calibration_ids)].reset_index(drop=True)

    u.section("FASE 3a - DISENO DEL PASA-BANDA")
    design = verify_bandpass_design()
    delay = verify_bandpass_zero_phase()
    design["measured_delay_samples"] = delay
    design["zero_phase_ok"] = (delay == 0)
    pd.DataFrame([design]).to_csv(cfg.R3_BANDPASS_DESIGN, index=False)
    print(f"  Ondulacion 100-1500 Hz : {design['ripple_100_1500hz_db']} dB"
          f"  [{'OK' if design['ripple_ok'] else 'REVISAR'}]")
    print(f"  -3 dB efectivos        : {design['effective_minus3db_low_hz']}"
          f" - {design['effective_minus3db_high_hz']} Hz"
          f"  (nominal: {cfg.BANDPASS_LOW}-{cfg.BANDPASS_HIGH} Hz)")
    print(f"  Atenuacion en bordes   : {design['gain_two_pass_50hz_db']} dB en 50 Hz,"
          f" {design['gain_two_pass_1800hz_db']} dB en 1800 Hz (dos pasadas)")
    print(f"  Retardo medido         : {delay} muestras  [{'OK' if delay == 0 else 'REVISAR'}]")
    print(f"  -> {cfg.R3_BANDPASS_DESIGN.relative_to(cfg.ROOT)}")

    u.section("FASE 3b - PRUEBA NULA (alpha=0, beta=1)")
    sample_rows = meta.sample(n=min(NULL_TEST_SAMPLES, len(meta)), random_state=0)
    sample_signals = [
        apply_bandpass(u.read_audio(Path(cfg.ROOT) / p)[0])
        for p in sample_rows["output_path"]
    ]
    worst_null = verify_null_denoising(sample_signals, nperseg=256, noverlap=192)
    null_ok = worst_null < NULL_TEST_TOLERANCE
    print(f"  Error maximo sobre {len(sample_signals)} grabaciones: {worst_null:.2e}"
          f"  [{'OK' if null_ok else 'REVISAR'}]")
    if not null_ok:
        raise RuntimeError(
            "La prueba nula de STFT/ISTFT ha fallado: el emparejamiento "
            "ventana/salto/reconstruccion no reproduce la entrada. Revisar "
            "antes de continuar; todo lo demas mediria un artefacto."
        )

    u.section("FASE 3b - BARRIDO DE RESOLUCION (STFT)")
    resolution_df = sweep_stft_resolution(calib_meta)
    resolution_df.to_csv(cfg.R3_STFT_RESOLUTION, index=False)
    columns = ["nperseg", "noverlap", "window_ms", "cycle_gap_snr_proxy_db",
               "cycle_correlation", "crackle_correlation", "wheeze_correlation",
               "musical_noise_ratio", "spectral_distortion_db"]
    print(resolution_df[columns].round(4).to_string(index=False))
    best_res = resolution_df.sort_values("cycle_gap_snr_proxy_db", ascending=False).iloc[0]
    print(f"  Recomendado: nperseg={int(best_res['nperseg'])}"
          f" noverlap={int(best_res['noverlap'])}"
          f" ({best_res['window_ms']} ms, {best_res['overlap_fraction']:.0%} solape)")
    print(f"  -> {cfg.R3_STFT_RESOLUTION.relative_to(cfg.ROOT)}")

    u.section("FASE 3b - BARRIDO DE AGRESIVIDAD Y SUELO ESPECTRAL")
    nperseg_chosen = int(best_res["nperseg"])
    noverlap_chosen = int(best_res["noverlap"])
    sweep_df = sweep_denoising(calib_meta, nperseg_chosen, noverlap_chosen)
    sweep_df["cumple_restricciones"] = sweep_df.apply(satisfies_constraints, axis=1)
    sweep_df.to_csv(cfg.R3_DENOISING_SWEEP, index=False)
    columns = ["stage", "noise_pct", "oversubtraction", "spectral_floor",
               "cycle_gap_snr_proxy_db", "cycle_correlation", "cycle_correlation_p10",
               "cycle_correlation_min", "crackle_correlation_p10", "wheeze_correlation_p10",
               "musical_noise_ratio", "musical_noise_ratio_p90",
               "spectral_distortion_db", "cumple_restricciones"]
    print(sweep_df[columns].round(4).to_string(index=False))
    print(f"  -> {cfg.R3_DENOISING_SWEEP.relative_to(cfg.ROOT)}")

    valid = sweep_df.loc[sweep_df["cumple_restricciones"]]
    ranked = valid if len(valid) else sweep_df
    best = ranked.sort_values("cycle_gap_snr_proxy_db", ascending=False).iloc[0]
    print(f"\n  Regla de seleccion: maximizar cycle_gap_snr_proxy_db sujeta a")
    print(f"    correlacion en ciclo   media >= {cfg.MIN_CYCLE_CORRELATION}"
          f"  y p10 >= {cfg.MIN_CYCLE_CORRELATION_P10}")
    print(f"    ruido musical          media <= {cfg.MAX_MUSICAL_NOISE_RATIO}"
          f"  y p90 <= {cfg.MAX_MUSICAL_NOISE_RATIO_P90}")
    print(f"  {'Cumple las restricciones' if len(valid) else 'NINGUNA configuracion las cumple'}"
          f" -> percentil={best['noise_pct']} alpha={best['oversubtraction']}"
          f" beta={best['spectral_floor']}")
    print(f"    cycle_gap_snr_proxy = {best['cycle_gap_snr_proxy_db']:.3f} dB"
          f"  (min {best['cycle_gap_snr_proxy_db_min']:.2f})")
    print(f"    correlacion ciclo    = {best['cycle_correlation']:.4f}"
          f"  (p10 {best['cycle_correlation_p10']:.4f}, min {best['cycle_correlation_min']:.4f})")
    print(f"    crepitantes / sibil. = {best['crackle_correlation']:.4f}"
          f" / {best['wheeze_correlation']:.4f}  (p10)")
    print(f"    ruido musical        = {best['musical_noise_ratio']:.3f}"
          f"  (p90 {best['musical_noise_ratio_p90']:.3f})")
    print("\n  Nota: snr_proxy_delta_db se reporta por continuidad con la fase 1,"
          " pero NO es apta")
    print("  para elegir: crece con alpha sin optimo interior."
          " cycle_gap_snr_proxy_db tampoco es")
    print("  una SNR real -el hueco entre ciclos no garantiza ruido puro-,"
          " pero su denominador")
    print("  procede de otra region de la senal y por eso no crece de forma mecanica.")

    u.section("FASE 3c - DISTRIBUCION DE RMS POST-FILTRADO")
    rms_df = compute_rms_distribution(calib_meta)
    rms_df.to_csv(cfg.R3_RMS_DISTRIBUTION, index=False)
    u.describe(rms_df["rms_post_bandpass"], "RMS post-filtrado", "", "{:.5f}")
    target_candidate = float(rms_df["rms_post_bandpass"].median())
    gain_needed = target_candidate / rms_df["rms_post_bandpass"].replace(0, np.nan)
    print(f"  TARGET_RMS candidato (mediana)     : {target_candidate:.5f}")
    print(f"  Ganancia que pediria ese objetivo  : "
          f"p50={gain_needed.median():.2f}x p90={gain_needed.quantile(.9):.2f}x"
          f" p99={gain_needed.quantile(.99):.2f}x max={gain_needed.max():.2f}x")
    print(f"  -> {cfg.R3_RMS_DISTRIBUTION.relative_to(cfg.ROOT)}")

    u.section("FASE 3 - CALIBRACION COMPLETA")
    print("  Revise los informes de reports/phase3/ y fije en config.py:")
    for name, source in cfg.pending_phase3_parameters():
        print(f"    {name:<16} -> {source}")
    print("  Luego ejecute 'python phase3_cleaning.py' (sin --calibrar) para el corpus completo.")


# ---------------------------------------------------------------------------
# Modo de ejecucion completa
# ---------------------------------------------------------------------------

def validate_full_run(meta, manifest):
    lines = []
    ok = True

    n_fail = int((manifest["status"] == "FAIL").sum())
    ok = ok and n_fail == 0
    lines.append(f"1. Sin fallos: {n_fail} filas FAIL  [{'OK' if n_fail == 0 else 'REVISAR'}]")

    okrows = manifest.loc[manifest["status"] == "OK"]
    counts = okrows.groupby("branch").size()
    counts_ok = counts.get("no_dn", 0) == len(meta) and counts.get("dn", 0) == len(meta)
    ok = ok and counts_ok
    lines.append(f"2. Conteo por rama: no_dn={counts.get('no_dn', 0)} dn={counts.get('dn', 0)}"
                 f" (esperado {len(meta)} cada una)  [{'OK' if counts_ok else 'REVISAR'}]")

    nan_any = bool(okrows["has_nan_or_inf"].any()) if len(okrows) else False
    ok = ok and not nan_any
    lines.append(f"3. NaN/Inf: {'si' if nan_any else 'no'}  [{'OK' if not nan_any else 'REVISAR'}]")

    dur_map = meta.set_index("audio_id")["samples_out_actual"]
    dur_bad = sum(
        1 for _, r in okrows.iterrows()
        if int(r["samples"]) != int(dur_map.get(r["audio_id"], -1))
    )
    ok = ok and dur_bad == 0
    lines.append(f"4. Duracion/muestras identicas a fase 2: {dur_bad} discrepancias"
                 f"  [{'OK' if dur_bad == 0 else 'REVISAR'}]")

    peak_bad = int((okrows["peak_final"] > cfg.PEAK_CEILING + 1e-6).sum()) if len(okrows) else 0
    ok = ok and peak_bad == 0
    lines.append(f"5. Pico <= {cfg.PEAK_CEILING}: {peak_bad} exceden"
                 f"  [{'OK' if peak_bad == 0 else 'REVISAR'}]")

    gain_bad = int((okrows["gain"] > cfg.MAX_GAIN + 1e-6).sum()) if len(okrows) else 0
    ok = ok and gain_bad == 0
    lines.append(f"6. Ganancia <= MAX_GAIN ({cfg.MAX_GAIN}): {gain_bad} exceden"
                 f"  [{'OK' if gain_bad == 0 else 'REVISAR'}]")

    delay = verify_bandpass_zero_phase()
    ok = ok and delay == 0
    lines.append(f"7. Fase cero del pasa-banda: retardo={delay} muestras"
                 f"  [{'OK' if delay == 0 else 'REVISAR'}]")

    # Se lee de STAGING y no de la columna output_path del manifiesto: esta
    # validacion corre antes del reemplazo atomico, cuando el audio todavia
    # esta en clean_staging/ y la ruta final registrada aun no existe.
    identical, checked = 0, 0
    for audio_id, group in okrows.groupby("audio_id"):
        if set(group["branch"]) != {"no_dn", "dn"}:
            continue
        dataset = group["dataset"].iloc[0]
        audio_path = group["audio_path"].iloc[0]
        p0 = branch_staging_path("no_dn", dataset, audio_path)
        p1 = branch_staging_path("dn", dataset, audio_path)
        y0, _ = u.read_audio(p0)
        y1, _ = u.read_audio(p1)
        checked += 1
        if y0.size == y1.size and np.array_equal(y0, y1):
            identical += 1
    branches_differ = identical == 0
    ok = ok and branches_differ
    lines.append(f"8. Ramas distintas (denoising tuvo efecto): {identical}/{checked} identicas"
                 f"  [{'OK' if branches_differ else 'REVISAR'}]")

    manifest_rows_ok = len(manifest) == 2 * len(meta)
    ok = ok and manifest_rows_ok
    lines.append(f"9. Filas del manifiesto: {len(manifest)} (esperado {2 * len(meta)})"
                 f"  [{'OK' if manifest_rows_ok else 'REVISAR'}]")

    return {"ok": ok, "report_lines": lines, "peak_bad": peak_bad, "gain_bad": gain_bad,
            "identical": identical, "checked": checked, "n_fail": n_fail}


def run_full():
    pending = cfg.pending_phase3_parameters()
    if pending:
        u.section("FASE 3 - PARAMETROS SIN FIJAR")
        for name, source in pending:
            print(f"  {name:<16} -> vea {source}")
        print("\n  Ejecute primero 'python phase3_cleaning.py --calibrar', revise los")
        print("  informes de reports/phase3/ y fije los valores en config.py.")
        return {"verdict": "PENDING"}

    u.section("FASE 3 - LIMPIEZA DE SENAL")
    meta = admitted_recordings()
    calibration_ids = load_calibration_set()
    print(f"  Grabaciones a procesar  : {len(meta)}  (x2 ramas = {2 * len(meta)} archivos)")
    print(f"  Pacientes de calibracion: {len(calibration_ids)}")

    params = {
        "nperseg": cfg.STFT_NPERSEG, "noverlap": cfg.STFT_NOVERLAP,
        "noise_pct": cfg.NOISE_PCT, "oversubtraction": cfg.OVERSUBTRACTION,
        "spectral_floor": cfg.SPECTRAL_FLOOR, "target_rms": cfg.TARGET_RMS,
        "max_gain": cfg.MAX_GAIN,
    }
    print(f"  STFT {params['nperseg']}/{params['noverlap']}"
          f"  percentil={params['noise_pct']}  alpha={params['oversubtraction']}"
          f"  beta={params['spectral_floor']}")
    print(f"  TARGET_RMS={params['target_rms']:.5f}  PEAK_CEILING={cfg.PEAK_CEILING}"
          f"  MAX_GAIN={params['max_gain']}")

    u.prepare_staging(cfg.CLEAN)

    branch_rows, energy_rows, metrics_rows = [], [], []
    progress = u.Progress(len(meta), "limpieza", every=50)
    for _, r in meta.iterrows():
        rows, energy, metrics = process_recording_both_branches(r, calibration_ids, params)
        branch_rows.extend(rows)
        energy_rows.append(energy)
        metrics_rows.append(metrics)
        progress.step()

    manifest = pd.DataFrame(branch_rows)
    pd.DataFrame(energy_rows).to_csv(cfg.R3_BAND_ENERGY, index=False)
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(cfg.R3_DENOISING_METRICS, index=False)

    print(f"\n  Filas de manifiesto : {len(manifest)}  (2 x {len(meta)})")
    print(f"  OK                  : {int((manifest['status'] == 'OK').sum())}")
    print(f"  FAIL                : {int((manifest['status'] == 'FAIL').sum())}")

    u.section("VALIDACION DE LA FASE 3")
    checks = validate_full_run(meta, manifest)
    for line in checks["report_lines"]:
        print(f"  {line}")

    # Efecto del denoising sobre el corpus completo, no solo sobre la
    # calibracion: es lo que permite senalar despues las grabaciones
    # concretas que mas se alteran.
    annotated = metrics_df.loc[metrics_df["cycle_correlation"].notna()]
    u.section("EFECTO DEL DENOISING SOBRE EL CORPUS COMPLETO")
    print(f"  Grabaciones con anotacion : {len(annotated)} de {len(metrics_df)}"
          f"  (solo ICBHI tiene ciclos anotados)")
    for column, label in (("cycle_correlation", "correlacion en ciclo"),
                           ("crackle_correlation", "correlacion crepitantes"),
                           ("wheeze_correlation", "correlacion sibilancias")):
        values = annotated[column].dropna()
        if len(values):
            print(f"  {label:<26} media={values.mean():.4f}"
                  f"  p10={values.quantile(0.10):.4f}  min={values.min():.4f}"
                  f"  (n={len(values)})")
    musical = metrics_df["musical_noise_ratio"].dropna()
    distortion = metrics_df["spectral_distortion_db"].dropna()
    print(f"  {'ruido musical':<26} media={musical.mean():.3f}"
          f"  p90={musical.quantile(0.90):.3f}  max={musical.max():.3f}")
    print(f"  {'distorsion 50-1800 Hz':<26} media={distortion.mean():.2f} dB"
          f"  p90={distortion.quantile(0.90):.2f} dB")

    flagged = metrics_df.loc[~metrics_df["dn_reliable"].fillna(True).astype(bool)]
    print(f"\n  Rama dn marcada como poco fiable : {len(flagged)} de {len(metrics_df)}"
          f"  ({100 * len(flagged) / max(len(metrics_df), 1):.1f} %)")
    worst = annotated.nsmallest(5, "cycle_correlation")
    if len(worst):
        print("  Grabaciones mas alteradas por el denoising:")
        for _, w in worst.iterrows():
            print(f"    {w['audio_id']:<22} correlacion={w['cycle_correlation']:.4f}"
                  f"  ruido_musical={w['musical_noise_ratio']:.3f}")
    print("  No se excluye ninguna: la rama no_dn las conserva intactas y la")
    print("  columna dn_reliable del manifiesto permite decidirlo en el modelado.")
    print(f"\n  -> {cfg.R3_DENOISING_METRICS.relative_to(cfg.ROOT)}")

    corr_values = annotated["cycle_correlation"].dropna()
    summary = pd.DataFrame([{
        "verdict": "PASS" if checks["ok"] else "FAIL",
        "n_recordings": len(meta), "n_manifest_rows": len(manifest),
        "n_fail": checks["n_fail"], **params,
        "peak_ceiling": cfg.PEAK_CEILING,
        "bandpass_low_hz": cfg.BANDPASS_LOW, "bandpass_high_hz": cfg.BANDPASS_HIGH,
        "bandpass_order": cfg.BANDPASS_ORDER,
        "cycle_guard_ms": cfg.CYCLE_GUARD_MS,
        "calibration_fraction": cfg.CALIBRATION_FRACTION,
        "calibration_seed": cfg.CALIBRATION_SEED,
        "min_cycle_correlation": cfg.MIN_CYCLE_CORRELATION,
        "min_cycle_correlation_p10": cfg.MIN_CYCLE_CORRELATION_P10,
        "max_musical_noise_ratio": cfg.MAX_MUSICAL_NOISE_RATIO,
        "max_musical_noise_ratio_p90": cfg.MAX_MUSICAL_NOISE_RATIO_P90,
        "corpus_cycle_correlation_mean": float(corr_values.mean()) if len(corr_values) else float("nan"),
        "corpus_cycle_correlation_p10": float(corr_values.quantile(0.10)) if len(corr_values) else float("nan"),
        "corpus_cycle_correlation_min": float(corr_values.min()) if len(corr_values) else float("nan"),
        "corpus_musical_noise_mean": float(musical.mean()) if len(musical) else float("nan"),
        "dn_flagged_recordings": int(len(flagged)),
        "branches_identical": checks["identical"], "branches_checked": checks["checked"],
        "peak_violations": checks["peak_bad"], "gain_violations": checks["gain_bad"],
    }])
    summary.to_csv(cfg.R3_SUMMARY, index=False)

    if not checks["ok"]:
        manifest.to_csv(cfg.R3_FAILED_ATTEMPT, index=False)
        u.section("FASE 3 - VALIDACION FALLIDA, NO SE REEMPLAZA LA SALIDA")
        exists = cfg.CLEAN.exists()
        print(f"  {'clean/ permanece intacto.' if exists else 'clean/ no existia y sigue sin existir.'}")
        print(f"  Intento fallido en {cfg.R3_FAILED_ATTEMPT.relative_to(cfg.ROOT)}")
        print(f"  Detalle en {cfg.R3_SUMMARY.relative_to(cfg.ROOT)}")
        return {"verdict": "FAIL"}

    u.swap_staging_into_place(cfg.CLEAN)
    manifest.to_csv(cfg.PHASE3_MANIFEST, index=False)
    if cfg.R3_FAILED_ATTEMPT.exists():
        cfg.R3_FAILED_ATTEMPT.unlink()

    u.section("RESUMEN DE LA FASE 3")
    print(f"  no_dn : {int((manifest['branch'] == 'no_dn').sum())} archivos")
    print(f"  dn    : {int((manifest['branch'] == 'dn').sum())} archivos")
    total_mb = sum(p.stat().st_size for p in cfg.CLEAN.rglob("*.wav")) / 1024 ** 2
    print(f"  Tamano en disco : {total_mb:.1f} MB")
    print(f"  Salida          : {cfg.CLEAN.relative_to(cfg.ROOT)}")
    print(f"  Manifiesto      : {cfg.PHASE3_MANIFEST.relative_to(cfg.ROOT)}")
    print(f"  Veredicto       : PASS")

    return {"verdict": "PASS"}


# ---------------------------------------------------------------------------

def main():
    cfg.ensure_dirs()
    parser = argparse.ArgumentParser(description="Fase 3 - limpieza de senal")
    parser.add_argument("--calibrar", action="store_true",
                         help="Solo mide y recomienda parametros; no procesa el corpus completo")
    args = parser.parse_args()

    if args.calibrar:
        run_calibration()
        return {"verdict": "CALIBRATED"}
    return run_full()


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result["verdict"] in ("PASS", "CALIBRATED") else 1)
