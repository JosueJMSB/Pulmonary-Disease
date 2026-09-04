"""
Utilidades compartidas por las cuatro fases del pipeline.

Agrupa la lectura de audio, los indicadores de calidad de senal y la carga de la
metadata unificada de ambos corpus.
"""

import hashlib
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import config as cfg


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def load_metadata():
    """Carga y concatena la metadata de ambos corpus.

    Todas las columnas se leen como texto para preservar identificadores como
    "101" o "F001"; las conversiones numericas se hacen donde se necesitan.
    Anade dos columnas: la raiz del dataset y la ruta absoluta del audio.
    """
    required = {
        "dataset", "audio_id", "patient_uid", "diagnosis", "device",
        "zone", "audio_path", "sample_rate_hz", "channels",
        "duration_seconds", "bit_depth",
    }
    frames = []
    for name, root, meta_path in cfg.DATASETS:
        df = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(
                f"Metadata incompleta en {meta_path}: faltan {', '.join(missing)}"
            )
        if not (df["dataset"] == name).all():
            wrong = sorted(df.loc[df["dataset"] != name, "dataset"].unique())
            raise ValueError(
                f"La columna dataset de {meta_path} contiene valores inesperados: {wrong}"
            )
        df["dataset_root"] = str(root)
        df["abs_path"] = [str(root / Path(p)) for p in df["audio_path"]]
        frames.append(df)
    metadata = pd.concat(frames, ignore_index=True)
    duplicated_ids = metadata.loc[
        metadata["audio_id"].duplicated(keep=False), "audio_id"
    ].unique()
    if len(duplicated_ids):
        raise ValueError(f"audio_id duplicados en la metadata: {duplicated_ids[:10].tolist()}")
    duplicated_paths = metadata.loc[
        metadata["abs_path"].duplicated(keep=False), "abs_path"
    ].unique()
    if len(duplicated_paths):
        raise ValueError(f"Rutas duplicadas en la metadata: {duplicated_paths[:10].tolist()}")
    return metadata


def annotation_path(row):
    """Ruta del .txt de anotacion correspondiente a una grabacion de ICBHI."""
    if row["dataset"] != "ICBHI":
        return None
    audio_rel = Path(row["audio_path"])
    try:
        inside_audio = audio_rel.relative_to("audio")
    except ValueError as exc:
        raise ValueError(f"Ruta ICBHI fuera de audio/: {audio_rel}") from exc
    return cfg.ICBHI_ANNOTATIONS / inside_audio.with_suffix(".txt")


# ---------------------------------------------------------------------------
# Lectura de audio
# ---------------------------------------------------------------------------

def read_audio(path):
    """Lee un WAV mono y devuelve (senal float64 en [-1, 1], frecuencia).

    soundfile normaliza cualquier profundidad de bits al mismo rango, de modo
    que los archivos de 16 y 24 bits se tratan de forma homogenea.
    """
    x, sr = sf.read(str(path), dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def audio_info(path):
    """Propiedades de la cabecera sin leer las muestras."""
    return sf.info(str(path))


# ---------------------------------------------------------------------------
# Indicadores de calidad de senal (fase 1c)
# ---------------------------------------------------------------------------

def rms(x):
    """Valor eficaz de la senal completa."""
    return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0


def rms_without_dc(x):
    """Valor eficaz tras retirar la componente continua."""
    if not x.size:
        return 0.0
    centered = x - np.mean(x)
    return float(np.sqrt(np.mean(centered ** 2)))


def _runs(mask):
    """Longitudes de las rachas verdaderas de una mascara booleana."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.size or not mask.any():
        return np.empty(0, dtype=np.int64)
    edges = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.size]))
    return ends - starts


def saturation_stats(x, sr, level=None, run_ms=None):
    """Porcentaje de muestras en rachas de saturacion y numero de rachas.

    Una muestra aislada en fondo de escala puede ser casualidad; la saturacion
    se manifiesta como muestras consecutivas pegadas al tope, que son las que
    forman la meseta plana. Solo se contabilizan las rachas que alcanzan la
    longitud minima.
    """
    level = cfg.SATURATION_LEVEL if level is None else level
    run_ms = cfg.SATURATION_RUN_MS if run_ms is None else run_ms
    run_min = max(1, int(round(sr * run_ms / 1000.0)))

    if x.size == 0:
        return 0.0, 0

    lengths = _runs(np.abs(x) >= level)
    long_enough = lengths >= run_min
    n_saturated = int(lengths[long_enough].sum())
    return 100.0 * n_saturated / x.size, int(long_enough.sum())


def digital_silence_stats(x):
    """Porcentaje de ceros exactos y longitud de la racha mas larga."""
    if not x.size:
        return 0.0, 0
    lengths = _runs(x == 0.0)
    longest = int(lengths.max()) if lengths.size else 0
    return 100.0 * float(np.count_nonzero(x == 0.0)) / x.size, longest


def file_sha256(path, chunk_size=1024 * 1024):
    """SHA-256 del archivo para detectar copias binarias exactas."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def frame_energy(x, sr, frame_ms=None):
    """Energia media de cada trama consecutiva de duracion fija."""
    frame_ms = cfg.QUALITY_FRAME_MS if frame_ms is None else frame_ms
    n = int(round(sr * frame_ms / 1000.0))
    if n < 1 or x.size < n:
        return np.empty(0)
    n_frames = x.size // n
    frames = x[: n_frames * n].reshape(n_frames, n)
    return (frames ** 2).mean(axis=1)


def estimate_snr_db(x, sr, frame_ms=None, low_pct=None, high_pct=None):
    """Relacion senal-ruido estimada por percentiles, en decibelios.

    Las auscultaciones no contienen un tramo de ruido puro delimitado, por lo
    que la medicion directa no es aplicable. Se divide la senal en tramas, se
    calcula la energia de cada una, y se toma el cociente entre un percentil
    alto de esa distribucion --que representa el nivel de senal-- y uno bajo
    --que representa el suelo de ruido--. Se evitan el maximo y el minimo
    absolutos por su sensibilidad a tramas anomalas.
    """
    low_pct = cfg.SNR_LOW_PCT if low_pct is None else low_pct
    high_pct = cfg.SNR_HIGH_PCT if high_pct is None else high_pct

    energies = frame_energy(x, sr, frame_ms)
    if energies.size == 0:
        return float("nan"), "NO_FRAMES"
    if not np.isfinite(energies).all():
        return float("nan"), "NON_FINITE"

    low = float(np.percentile(energies, low_pct))
    high = float(np.percentile(energies, high_pct))
    # Un tramo de ceros puede adquirir residuos del orden de 1e-33 por el
    # filtrado en fase cero. Tratarlo como ruido real produciria SNR absurdas
    # de cientos de dB; por debajo de la precision numerica se declara que el
    # suelo de ruido no es estimable.
    numerical_floor = np.finfo(np.float64).eps * max(high, 1.0)
    if low <= numerical_floor:
        return float("nan"), "UNESTIMABLE"
    if high <= 0:
        return float("nan"), "UNESTIMABLE"
    return float(10.0 * np.log10(high / low)), "OK"


def snr_proxy_db(x, sr, band=None):
    """Proxy de SNR calculado en la banda comun, sin modificar el audio.

    Se retira DC, se aplica un Butterworth de cuarto orden en fase cero y se
    estima la relacion entre percentiles de energia. El estado explica por que
    un valor no pudo medirse; nunca se devuelve infinito.
    """
    band = cfg.SNR_BAND if band is None else band
    if not x.size:
        return float("nan"), "EMPTY_SIGNAL"
    if not np.isfinite(x).all():
        return float("nan"), "NON_FINITE"
    low, high = band
    nyquist = sr / 2.0
    if low <= 0 or high >= nyquist or low >= high:
        return float("nan"), "INVALID_BAND"
    centered = x - np.mean(x)
    try:
        sos = butter(4, [low, high], btype="bandpass", fs=sr, output="sos")
        filtered = sosfiltfilt(sos, centered)
    except (ValueError, FloatingPointError):
        return float("nan"), "FILTER_FAILED"
    return estimate_snr_db(filtered, sr)


def dc_offset(x):
    """Desplazamiento de linea base, como fraccion del fondo de escala."""
    return float(np.mean(x)) if x.size else 0.0


# ---------------------------------------------------------------------------
# Staging: escritura segura y reemplazo atomico
# ---------------------------------------------------------------------------
#
# Toda fase que regenera un directorio de salida completo sigue el mismo
# patron: escribir en un directorio "<destino>_staging" homonimo, validar por
# completo, y solo entonces reemplazar el destino con un renombrado atomico.
# Si algo falla a mitad de camino, el destino anterior permanece intacto. La
# fase 2 lo introdujo para una sola carpeta (resampled/); estas versiones
# genericas, parametrizadas por directorio, permiten que la fase 3 la reuse
# para clean/ -que contiene ambas ramas bajo un unico padre- sin duplicar la
# logica ni inventar un intercambio transaccional de multiples carpetas.

def staging_dir_for(target_dir):
    """Directorio de staging homonimo de un directorio de destino."""
    return target_dir.with_name(target_dir.name + "_staging")


def previous_swap_dir_for(target_dir):
    """Directorio temporal donde se aparca el destino anterior durante el swap."""
    return target_dir.with_name(target_dir.name + "_previous_swap")


def prepare_staging(target_dir):
    """Directorio de staging limpio, con salvaguarda ante una ruta inesperada.

    Antes de borrar un directorio existente se comprueba que su ruta resuelta
    coincide exactamente con la esperada: protege contra el caso en que
    "<destino>_staging" hubiera sido reemplazado por un enlace simbolico a
    otro lugar.
    """
    staging = staging_dir_for(target_dir)
    if staging.exists():
        resolved = staging.resolve()
        expected = staging_dir_for(target_dir).resolve()
        if resolved != expected:
            raise RuntimeError(
                f"Ruta de staging inesperada, se aborta por seguridad: {resolved}"
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    return staging


def write_atomic(path, y, sr, subtype="FLOAT"):
    """Escribe primero un archivo temporal y lo renombra al completar.

    Si el proceso se interrumpe a mitad de la escritura, queda un .part
    huerfano y nunca un .wav truncado bajo el nombre final. El sufijo ".part"
    (no ".wav") evita que un intento fallido se cuente como salida valida en
    un recuento de archivos por glob "*.wav". El formato se declara de forma
    explicita porque el nombre temporal ya no permite inferirlo de la
    extension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    sf.write(str(tmp), y.astype(np.float32), sr, subtype=subtype, format="WAV")
    os.replace(tmp, path)


def swap_staging_into_place(target_dir):
    """Reemplazo atomico del directorio de destino por su staging homonimo.

    Renombrar es la unica operacion atomica disponible a nivel de sistema de
    archivos; copiar no lo es. Si el reemplazo falla a medio camino, se
    restaura el directorio anterior automaticamente.
    """
    staging = staging_dir_for(target_dir)
    previous = previous_swap_dir_for(target_dir)
    had_previous = target_dir.exists()
    try:
        if had_previous:
            if previous.exists():
                shutil.rmtree(previous)
            os.replace(target_dir, previous)
        os.replace(staging, target_dir)
    except OSError:
        if had_previous and previous.exists() and not target_dir.exists():
            os.replace(previous, target_dir)
        raise
    if had_previous and previous.exists():
        shutil.rmtree(previous)


# ---------------------------------------------------------------------------
# Presentacion
# ---------------------------------------------------------------------------

class Progress:
    """Indicador de avance para bucles largos sobre archivos."""

    def __init__(self, total, label="", every=50):
        self.total = total
        self.label = label
        self.every = every
        self.start = time.time()
        self.n = 0

    def step(self, k=1):
        self.n += k
        if self.n % self.every and self.n != self.total:
            return
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed else 0
        remaining = (self.total - self.n) / rate if rate else 0
        sys.stdout.write(
            f"\r  {self.label} {self.n}/{self.total}"
            f"  ({100 * self.n / self.total:5.1f}%)"
            f"  {elapsed:5.1f}s transcurridos"
            f"  ~{remaining:5.1f}s restantes   "
        )
        sys.stdout.flush()
        if self.n == self.total:
            sys.stdout.write("\n")


def section(title):
    """Encabezado de seccion en la salida por consola."""
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def describe(series, label, unit="", fmt="{:.4f}"):
    """Resumen de una distribucion: minimo, cuartiles, mediana y maximo."""
    s = pd.Series(series).dropna()
    s = s[np.isfinite(s)]
    if s.empty:
        print(f"  {label:<26} sin datos")
        return
    q = s.quantile([0.01, 0.25, 0.50, 0.75, 0.99])
    print(
        f"  {label:<26}"
        f" min={fmt.format(s.min())}"
        f"  p1={fmt.format(q[0.01])}"
        f"  p25={fmt.format(q[0.25])}"
        f"  med={fmt.format(q[0.50])}"
        f"  p75={fmt.format(q[0.75])}"
        f"  p99={fmt.format(q[0.99])}"
        f"  max={fmt.format(s.max())}{unit}"
    )
