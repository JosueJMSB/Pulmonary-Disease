"""
Utilidades compartidas por las cuatro fases del pipeline.

Agrupa la lectura de audio, los indicadores de calidad de senal y la carga de la
metadata unificada de ambos corpus.
"""

import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf

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
    frames = []
    for name, root, meta_path in cfg.DATASETS:
        df = pd.read_csv(meta_path, dtype=str, keep_default_na=False)
        df["dataset_root"] = str(root)
        df["abs_path"] = [str(root / p) for p in df["audio_path"]]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def annotation_path(row):
    """Ruta del .txt de anotacion correspondiente a una grabacion de ICBHI."""
    if row["dataset"] != "ICBHI":
        return None
    rel = row["audio_path"].replace("audio/", "annotations/", 1)
    return cfg.ICBHI_DIR / (rel[:-4] + ".txt")


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


def saturation_stats(x, level=None, run_min=None):
    """Porcentaje de muestras en rachas de saturacion y numero de rachas.

    Una muestra aislada en fondo de escala puede ser casualidad; la saturacion
    se manifiesta como muestras consecutivas pegadas al tope, que son las que
    forman la meseta plana. Solo se contabilizan las rachas que alcanzan la
    longitud minima.
    """
    level = cfg.SATURATION_LEVEL if level is None else level
    run_min = cfg.SATURATION_RUN_MIN if run_min is None else run_min

    if x.size == 0:
        return 0.0, 0

    mask = np.abs(x) >= level
    if not mask.any():
        return 0.0, 0

    edges = np.diff(mask.astype(np.int8))
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [mask.size]))

    lengths = ends - starts
    long_enough = lengths >= run_min
    n_saturated = int(lengths[long_enough].sum())
    return 100.0 * n_saturated / x.size, int(long_enough.sum())


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
        return float("nan")

    low = float(np.percentile(energies, low_pct))
    high = float(np.percentile(energies, high_pct))
    if low <= 0:
        return float("inf") if high > 0 else float("nan")
    return 10.0 * np.log10(high / low)


def dc_offset(x):
    """Desplazamiento de linea base, como fraccion del fondo de escala."""
    return float(np.mean(x)) if x.size else 0.0


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
