"""
Fase 2 - Estandarizacion de la senal.

Los audios proceden de campanas con condiciones tecnicas distintas y presentan
frecuencias de muestreo heterogeneas. Esta fase las lleva a una frecuencia comun
de 4 kHz.

  2a  Filtro anti-aliasing
  2b  Estandarizacion de frecuencia

Ambas etapas se ejecutan como una sola operacion polifasica, en el orden que
exige el documento: primero el filtrado, despues la decimacion. Las grabaciones
que ya se encuentran a la frecuencia de destino no se transforman, para que no
reciban un filtrado que las demas no reciben.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

import config as cfg
import utils as u


# Las etapas intermedias se almacenan en coma flotante de 32 bits. Evita la
# perdida por cuantizacion al encadenar transformaciones y elimina el riesgo de
# recorte cuando el filtrado produce un ligero rebase por encima de 1.0.
INTERIM_SUBTYPE = "FLOAT"

SPECTRAL_CHECK_SAMPLES = 12
SPECTRAL_CHECK_SECONDS = 5


# ---------------------------------------------------------------------------

def admitted_recordings():
    """Metadata de las grabaciones que superaron la fase 1.

    Si aun no existe el informe de calidad, se procesa el corpus completo y se
    deja constancia de ello.
    """
    meta = u.load_metadata()
    quality_path = cfg.REPORTS / "signal_quality.csv"

    if not quality_path.exists():
        print("  Aviso: no se encontro signal_quality.csv; se procesa el corpus completo.")
        return meta, 0

    quality = pd.read_csv(quality_path)
    excluded = set(quality.loc[quality["excluded"], "audio_id"])
    admitted = meta.loc[~meta["audio_id"].isin(excluded)].reset_index(drop=True)
    return admitted, len(excluded)


def output_path(row):
    """Ruta de salida, replicando la estructura de origen bajo el conjunto."""
    relative = row["audio_path"].split("audio/", 1)[-1]
    return cfg.RESAMPLED / row["dataset"] / relative


def standardize(x, source_sr):
    """Aplica anti-aliasing y decimacion a la frecuencia de destino.

    Devuelve la senal remuestreada y si hubo transformacion. La relacion entre
    frecuencias no es entera, por lo que se emplea remuestreo racional: la senal
    se interpola por el numerador, se filtra con el pasa-bajos anti-aliasing y
    se decima por el denominador. La implementacion polifasica integra las tres
    operaciones y calcula unicamente las muestras que subsisten.
    """
    if source_sr == cfg.TARGET_SR:
        return x, False

    if source_sr not in cfg.RESAMPLE_RATIOS:
        raise ValueError(f"frecuencia de origen no contemplada: {source_sr} Hz")

    up, down = cfg.RESAMPLE_RATIOS[source_sr]
    return resample_poly(x, up=up, down=down), True


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


# ---------------------------------------------------------------------------

def run(meta):
    """Remuestrea cada grabacion y deja constancia de lo aplicado."""
    u.section("FASE 2 - ANTI-ALIASING Y ESTANDARIZACION DE FRECUENCIA")

    by_rate = meta["sample_rate_hz"].astype(int).value_counts().sort_index()
    print("  Frecuencias de origen:")
    for rate, count in by_rate.items():
        if rate == cfg.TARGET_SR:
            action = "sin transformar"
        elif rate in cfg.RESAMPLE_RATIOS:
            up, down = cfg.RESAMPLE_RATIOS[rate]
            action = f"remuestreo {up}/{down}"
        else:
            action = "SIN REGLA DEFINIDA"
        print(f"    {rate:>6} Hz  {count:>5} archivos   {action}")

    rows = []
    progress = u.Progress(len(meta), "remuestreo", every=50)

    for _, r in meta.iterrows():
        source_sr = int(r["sample_rate_hz"])
        destination = output_path(r)
        destination.parent.mkdir(parents=True, exist_ok=True)

        x, sr = u.read_audio(r["abs_path"])
        y, transformed = standardize(x, sr)

        # El filtrado puede producir un rebase leve por encima del fondo de
        # escala. Se registra, pero no se recorta: el formato de coma flotante
        # lo admite y la normalizacion de la fase 3 lo resolvera.
        overshoot = float(np.max(np.abs(y))) if y.size else 0.0

        sf.write(str(destination), y.astype(np.float32), cfg.TARGET_SR,
                 subtype=INTERIM_SUBTYPE)

        rows.append({
            "dataset": r["dataset"],
            "audio_id": r["audio_id"],
            "device": r["device"],
            "source_sr": source_sr,
            "target_sr": cfg.TARGET_SR,
            "transformed": transformed,
            "samples_in": int(x.size),
            "samples_out": int(y.size),
            "duration_in_s": round(x.size / sr, 4),
            "duration_out_s": round(y.size / cfg.TARGET_SR, 4),
            "peak_out": overshoot,
            "clipped_if_int16": overshoot > 1.0,
            "output_path": str(destination.relative_to(cfg.ROOT)),
        })
        progress.step()

    report = pd.DataFrame(rows)

    print(f"\n  Transformadas         : {int(report['transformed'].sum())}")
    print(f"  Copiadas sin cambio   : {int((~report['transformed']).sum())}")

    drift = (report["duration_out_s"] - report["duration_in_s"]).abs()
    print(f"  Desviacion de duracion: max {drift.max():.5f} s")

    rebase = report.loc[report["clipped_if_int16"]]
    print(f"  Rebase sobre 1.0      : {len(rebase)} grabaciones"
          f"  (max {report['peak_out'].max():.4f})")
    if len(rebase):
        print("    Se conservan en coma flotante sin recortar; la normalizacion")
        print("    de la fase 3c las llevara al nivel comun.")

    out = cfg.REPORTS / "resampling.csv"
    report.to_csv(out, index=False)
    print(f"\n  -> {out.relative_to(cfg.ROOT)}")

    return report


# ---------------------------------------------------------------------------

def verify(meta, report):
    """Comprobaciones que exige el documento para esta fase."""
    u.section("VERIFICACION DE LA FASE 2")

    # 1. La banda conservada no debe haber sido contaminada por aliasing.
    #    Buscar contenido sobre el nuevo Nyquist no serviria: el muestreo lo
    #    impide por construccion. Lo que se comprueba es que el espectro de la
    #    banda que si sobrevive coincida con el del original.
    resampled = meta.loc[meta["sample_rate_hz"].astype(int) != cfg.TARGET_SR]
    sample = resampled.head(SPECTRAL_CHECK_SAMPLES) if len(resampled) else resampled

    flat_top = int(cfg.TARGET_SR / 2 * PASSBAND_FRACTION)
    print(f"  1. Fidelidad espectral de la banda conservada  ({len(sample)} grabaciones)\n")
    print(f"     {'audio_id':<26}{'origen':>9}{'replegable':>12}"
          f"{'banda plana':>13}{'exceso':>10}")

    worst_deviation = 0.0
    worst_excess = -1.0
    for _, r in sample.iterrows():
        source_sr = int(r["sample_rate_hz"])
        x, sr = u.read_audio(r["abs_path"])
        y, _ = u.read_audio(output_path(r))

        n_in = min(x.size, sr * SPECTRAL_CHECK_SECONDS)
        n_out = min(y.size, cfg.TARGET_SR * SPECTRAL_CHECK_SECONDS)

        foldable = high_band_energy(x[:n_in], sr, cfg.TARGET_SR // 2)
        deviation, excess = spectral_comparison(x[:n_in], sr, y[:n_out], cfg.TARGET_SR)
        if not np.isnan(deviation):
            worst_deviation = max(worst_deviation, deviation)
        if not np.isnan(excess):
            worst_excess = max(worst_excess, excess)

        print(f"     {r['audio_id']:<26}{source_sr:>8}Hz"
              f"{100 * foldable:>11.4f}%{100 * deviation:>12.3f}%{100 * excess:>9.2f}%")

    fidelity_ok = worst_deviation < 0.02
    no_aliasing = worst_excess < 0.05

    print(f"\n     'replegable'  energia sobre {cfg.TARGET_SR // 2} Hz en el original,"
          f" la que aliasaria sin filtro")
    print(f"     'banda plana' desviacion maxima por debajo de {flat_top} Hz")
    print(f"     'exceso'      energia anadida en cualquier banda; positiva indicaria aliasing")
    print(f"\n     Fidelidad en banda plana : {100 * worst_deviation:6.3f} %"
          f"   [{'OK' if fidelity_ok else 'REVISAR'}]")
    print(f"     Exceso maximo            : {100 * worst_excess:6.2f} %"
          f"   [{'OK' if no_aliasing else 'REVISAR'}]")
    if worst_excess < 0:
        print(f"     Ninguna banda gana energia: el filtro solo atenua, como debe.")

    # 2. Las grabaciones ya a 4 kHz deben conservar sus muestras intactas
    untouched = meta.loc[meta["sample_rate_hz"].astype(int) == cfg.TARGET_SR]
    print(f"\n  2. Integridad de las {len(untouched)} grabaciones no transformadas\n")

    mismatches = 0
    checked = 0
    for _, r in untouched.iterrows():
        x, _ = u.read_audio(r["abs_path"])
        y, _ = u.read_audio(output_path(r))
        if x.size != y.size or not np.allclose(x, y, rtol=0, atol=1e-9):
            mismatches += 1
        checked += 1

    print(f"     Contrastadas          : {checked}")
    print(f"     Con muestras alteradas: {mismatches}"
          f"   [{'OK' if mismatches == 0 else 'REVISAR'}]")

    # 3. Las duraciones deben conservarse
    drift = (report["duration_out_s"] - report["duration_in_s"]).abs()
    print(f"\n  3. Conservacion de la duracion\n")
    print(f"     Desviacion maxima     : {drift.max():.6f} s"
          f"   [{'OK' if drift.max() < 0.01 else 'REVISAR'}]")

    return {"passband_deviation": worst_deviation,
            "max_excess": worst_excess,
            "identity_mismatches": mismatches,
            "max_duration_drift": float(drift.max())}


# ---------------------------------------------------------------------------

def main():
    cfg.ensure_dirs()

    meta, n_excluded = admitted_recordings()

    u.section("FASE 2 - ESTANDARIZACION DE LA SENAL")
    print(f"  Grabaciones a procesar : {len(meta)}")
    if n_excluded:
        print(f"  Excluidas en la fase 1 : {n_excluded}")
    print(f"  Frecuencia de destino  : {cfg.TARGET_SR} Hz")
    print(f"  Formato intermedio     : coma flotante de 32 bits")

    report = run(meta)
    checks = verify(meta, report)

    u.section("RESUMEN DE LA FASE 2")
    print(f"  Procesadas            : {len(report)}")
    print(f"  Remuestreadas         : {int(report['transformed'].sum())}")
    print(f"  Sin transformar       : {int((~report['transformed']).sum())}")
    print(f"  Fidelidad banda plana : {100 * checks['passband_deviation']:.3f} %")
    print(f"  Exceso maximo         : {100 * checks['max_excess']:.2f} %")
    print(f"  Muestras alteradas    : {checks['identity_mismatches']}")
    print(f"  Salida                : {cfg.RESAMPLED.relative_to(cfg.ROOT)}")

    total_mb = sum(p.stat().st_size for p in cfg.RESAMPLED.rglob("*.wav")) / 1024 ** 2
    print(f"  Tamano en disco       : {total_mb:.1f} MB")

    return report


if __name__ == "__main__":
    main()
