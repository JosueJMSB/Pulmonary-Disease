"""
Parametros del pipeline de preprocesamiento.

Todos los valores numericos que gobiernan el pipeline residen en este archivo.
Los que aparecen como None son los que el documento declara "a determinar tras
observar la distribucion real"; permanecen sin fijar hasta que la fase 1 los
mida, y mientras tanto ninguna grabacion es excluida.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "Audios_Respiratorios"

ICBHI_DIR = RAW / "Dataset1-ICBHI"
FRAIWAN_DIR = RAW / "Dataset2_Mendeley"

ICBHI_META = ICBHI_DIR / "metadata" / "icbhi_audio_metadata.csv"
FRAIWAN_META = FRAIWAN_DIR / "metadata" / "fraiwan_audio_metadata.csv"
ICBHI_CYCLES = ICBHI_DIR / "metadata" / "icbhi_respiratory_cycles.csv"
ICBHI_ANNOTATIONS = ICBHI_DIR / "annotations"

PREPROC = ROOT / "preprocessing"
REPORTS = PREPROC / "reports"
FIGURES = REPORTS / "figures"

DATA = PREPROC / "data"
INTERIM = DATA / "interim"
RESAMPLED = INTERIM / "resampled"
CLEAN_NO_DN = INTERIM / "clean_no_dn"
CLEAN_DN = INTERIM / "clean_dn"
FINAL = DATA / "final"

DATASETS = (
    ("ICBHI", ICBHI_DIR, ICBHI_META),
    ("FRAIWAN", FRAIWAN_DIR, FRAIWAN_META),
)

# ---------------------------------------------------------------------------
# Fase 1c - Calidad de senal
# ---------------------------------------------------------------------------

# Una muestra suelta en fondo de escala puede ser casualidad; la saturacion se
# caracteriza por muestras consecutivas, que son las que forman la meseta plana.
SATURATION_LEVEL = 0.9999      # fraccion del fondo de escala que cuenta como tope
SATURATION_RUN_MIN = 3         # muestras consecutivas para considerarlo saturacion

# Estimacion percentilica de la relacion senal-ruido. Las auscultaciones no
# contienen un tramo de ruido puro delimitado, de modo que el suelo de ruido se
# estima con las tramas mas silenciosas y el nivel de senal con las mas intensas.
QUALITY_FRAME_MS = 64          # duracion de trama para la energia de corto plazo
SNR_LOW_PCT = 10               # percentil que estima el suelo de ruido
SNR_HIGH_PCT = 90              # percentil que estima el nivel de senal

# Umbrales de admision. Se fijan tras observar las distribuciones que produce la
# fase 1 sobre el conjunto completo. Mientras sean None no se excluye nada.
#
# Observado en la primera ejecucion sobre las 1256 grabaciones:
#
#   Saturacion  distribucion continua, sin discontinuidad natural. Mediana 0 %,
#               percentil 75 en 0.17 % y maximo en 48.10 %. Por encima del 5 %
#               una de cada veinte muestras esta destruida y el espectro se
#               corrompe de forma apreciable: ese es el criterio del corte.
#               Afecta a 53 grabaciones, el 4.2 % del corpus.
#
#   RMS         minimo 0.00244, sin discontinuidad y sin ningun archivo por
#               debajo de 0.001. No hay capturas fallidas en el corpus, de modo
#               que el umbral actua unicamente como salvaguarda.
#
#   Varianza    ningun archivo constante: el minimo esta muy por encima de cero.
#
#   SNR         mediana 10.15 dB, minimo 2.57 dB, distribucion continua. El
#               corte en 5 dB retira 11 grabaciones sin concentrar las
#               exclusiones de forma desproporcionada en un solo dispositivo,
#               lo que evitaria introducir sesgo instrumental.

MAX_SATURATION_PCT = 5.0       # % maximo de muestras en rachas de saturacion
MIN_RMS = 0.001                # por debajo se considera captura fallida
MIN_VARIANCE = 0.0             # varianza nula identifica el archivo constante
MIN_SNR_DB = 5.0               # relacion senal-ruido minima admisible

# ---------------------------------------------------------------------------
# Fase 2 - Estandarizacion de frecuencia
# ---------------------------------------------------------------------------

TARGET_SR = 4000

# Fracciones del remuestreo racional. Las grabaciones que ya estan a TARGET_SR
# no se transforman, para que no reciban un filtrado que las demas no reciben.
RESAMPLE_RATIOS = {
    44100: (40, 441),
    10000: (2, 5),
}

# ---------------------------------------------------------------------------
# Fase 3 - Limpieza de senal
# ---------------------------------------------------------------------------

BANDPASS_LOW = 50
BANDPASS_HIGH = 1800
BANDPASS_ORDER = 4

STFT_WINDOW = "hann"
STFT_NPERSEG = None            # longitud de ventana, a determinar en 3b
STFT_NOVERLAP = None           # salto entre tramas, a determinar en 3b
NOISE_PCT = None               # percentil bajo para estimar el ruido por banda
OVERSUBTRACTION = None         # factor de sobre-sustraccion
SPECTRAL_FLOOR = None          # suelo espectral

TARGET_RMS = None              # objetivo de la normalizacion, a determinar en 3c

# ---------------------------------------------------------------------------
# Fase 4 - Estandarizacion temporal
# ---------------------------------------------------------------------------

SEGMENT_OVERLAP = 0.50
SEGMENT_SECONDS = None         # longitud de ventana, a determinar en 4b

# Duraciones evaluadas en la etapa 4b
SEGMENT_CANDIDATES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def ensure_dirs():
    """Crea los directorios de salida si no existen."""
    for d in (REPORTS, FIGURES, RESAMPLED, CLEAN_NO_DN, CLEAN_DN, FINAL):
        d.mkdir(parents=True, exist_ok=True)


def pending_parameters():
    """Devuelve los parametros que siguen sin fijar."""
    here = globals()
    names = [
        "MAX_SATURATION_PCT", "MIN_RMS", "MIN_SNR_DB",
        "STFT_NPERSEG", "STFT_NOVERLAP", "NOISE_PCT",
        "OVERSUBTRACTION", "SPECTRAL_FLOOR", "TARGET_RMS", "SEGMENT_SECONDS",
    ]
    return [n for n in names if here[n] is None]
