"""Parametros versionados del pipeline de preprocesamiento.

La fase 1 distingue entre fallos objetivos (EXCLUDE) e indicadores acusticos
que requieren revision (REVIEW). Ninguna de estas decisiones borra o modifica
los audios originales.
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
ICBHI_CYCLE_SUMMARY = ICBHI_DIR / "metadata" / "icbhi_cycle_summary.csv"
ICBHI_DIAGNOSES = ICBHI_DIR / "metadata" / "patient_diagnosis.csv"
ICBHI_ANNOTATIONS = ICBHI_DIR / "annotations"
FRAIWAN_SOURCE_XLSX = FRAIWAN_DIR / "metadata" / "Data annotation.xlsx"

PREPROC = ROOT / "preprocessing"

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
# Informes
# ---------------------------------------------------------------------------
#
# Se agrupan por fase, y dentro de cada carpeta el prefijo numerico indica la
# etapa que los produce, de modo que el listado ordenado del directorio refleja
# el orden de ejecucion. Los informes sin prefijo no pertenecen a una sola
# etapa: resumen la fase completa.
#
# Todas las rutas viven aqui y no como literales dispersos en los modulos de
# fase: renombrar un informe debe ser un cambio en un solo lugar.

REPORTS = PREPROC / "reports"
REPORTS_P1 = REPORTS / "phase1"
REPORTS_P2 = REPORTS / "phase2"
REPORTS_P3 = REPORTS / "phase3"
FIGURES = REPORTS / "figures"

# Fase 1 - Verificacion de datos
R1_INTEGRITY = REPORTS_P1 / "1a_integrity.csv"
R1_ANNOTATION_VALIDATION = REPORTS_P1 / "1b_annotation_validation.csv"
R1_CYCLES = REPORTS_P1 / "1b_cycles_regenerated.csv"
R1_CYCLE_SUMMARY = REPORTS_P1 / "1b_cycle_summary_regenerated.csv"
R1_SIGNAL_QUALITY = REPORTS_P1 / "1c_signal_quality.csv"
R1_DUPLICATES = REPORTS_P1 / "1d_duplicates.csv"
R1_METADATA_VALIDATION = REPORTS_P1 / "1e_metadata_validation.csv"
PHASE1_MANIFEST = REPORTS_P1 / "manifest.csv"

# Fase 2 - Estandarizacion de la senal
R2_FILTER_DESIGN = REPORTS_P2 / "2a_filter_design.csv"
R2_TONE_RESPONSE = REPORTS_P2 / "2a_tone_response.csv"
R2_SPECTRAL_CHECK = REPORTS_P2 / "2a_spectral_check.csv"
R2_RESAMPLING = REPORTS_P2 / "2b_resampling.csv"
R2_FAILED_ATTEMPT = REPORTS_P2 / "2b_resampling_attempt_failed.csv"
R2_SUMMARY = REPORTS_P2 / "validation_summary.csv"

# ---------------------------------------------------------------------------
# Fase 1c - Calidad de senal
# ---------------------------------------------------------------------------

# Una muestra suelta en fondo de escala puede ser casualidad. La racha minima se
# expresa en tiempo para aplicar el mismo criterio a 4, 10 y 44.1 kHz.
SATURATION_LEVEL = 0.9999      # fraccion del fondo de escala que cuenta como tope
SATURATION_RUN_MS = 0.5        # duracion minima de una meseta de fondo de escala

# Estimacion percentilica de la relacion senal-ruido. Las auscultaciones no
# contienen un tramo de ruido puro delimitado, de modo que el suelo de ruido se
# estima con las tramas mas silenciosas y el nivel de senal con las mas intensas.
QUALITY_FRAME_MS = 64          # duracion de trama para la energia de corto plazo
SNR_LOW_PCT = 10               # percentil que estima el suelo de ruido
SNR_HIGH_PCT = 90              # percentil que estima el nivel de senal

# Banda comun usada solo para medir el proxy de SNR. La copia filtrada no se
# escribe en disco y no sustituye el filtrado definitivo de la fase 3.
COMMON_BAND_LOW = 50
COMMON_BAND_HIGH = 1800
SNR_BAND = (COMMON_BAND_LOW, COMMON_BAND_HIGH)

# Porcentaje de muestras exactamente cero que requiere revision. Este criterio
# detecta relleno digital; no pretende identificar pausas respiratorias normales.
MAX_DIGITAL_SILENCE_PCT = 10.0

# Umbrales de calidad. Saturacion y SNR generan REVIEW; los fallos objetivos
# (archivo ilegible, muestras no finitas, senal vacia o plana) generan EXCLUDE.
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
#   SNR         mediana 12.63 dB en la banda comun de 50-1800 Hz y minimo
#               medible de 0.49 dB. El corte en 5 dB marca 20 grabaciones para
#               revision (16 AKG C417L y 4 Meditron); no las elimina, porque esa
#               concentracion por dispositivo podria sesgar el experimento.

MAX_SATURATION_PCT = 5.0       # por encima: REVIEW
MIN_RMS = 0.001                # por debajo: EXCLUDE (captura fallida)
MIN_VARIANCE = 0.0             # varianza nula: EXCLUDE (archivo constante)
MIN_SNR_DB = 5.0               # proxy por debajo: REVIEW

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

# Filtro anti-aliasing explicito. El filtro implicito de resample_poly (ventana
# Kaiser con beta=5.0 fijo, sin parametro propio) situa su corte exactamente en
# el nuevo Nyquist, sin margen de guarda: verificado con freqz, a 2000 Hz solo
# ofrece 6 dB de atenuacion y no alcanza 60 dB hasta pasados los 2500 Hz. En
# este corpus la energia real por encima de 2000 Hz es inferior al 0.02 % en
# todos los casos verificados, de modo que el defecto no corrompia audio de
# forma medible, pero tampoco era una especificacion citable. Este filtro fija
# una banda de transicion de 200 Hz (1800-2000 Hz) con al menos 60 dB de
# atenuacion comprobados, tanto con freqz como con tonos puros.
ANTIALIAS_PASSBAND_HZ = COMMON_BAND_HIGH   # 1800 Hz: limite superior de la banda util
ANTIALIAS_STOPBAND_HZ = 2000               # nuevo Nyquist
ANTIALIAS_CUTOFF_HZ = 1900                 # corte del FIR, centro de la transicion
ANTIALIAS_RIPPLE_DB = 65                   # atenuacion solicitada a kaiserord
ANTIALIAS_MIN_ATTEN_DB = 60                # atenuacion minima aceptada en verificacion
ANTIALIAS_PASSBAND_TOL_DB = 0.1            # ondulacion maxima aceptada hasta 1800 Hz
ANTIALIAS_PADTYPE = "line"                 # continuacion lineal, no ceros, en los bordes
ANTIALIAS_MAX_TIME_ERROR_SAMPLES = 1.0     # error temporal maximo, en muestras de salida

# ---------------------------------------------------------------------------
# Fase 3 - Limpieza de senal
# ---------------------------------------------------------------------------

BANDPASS_LOW = COMMON_BAND_LOW
BANDPASS_HIGH = COMMON_BAND_HIGH
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
    for d in (REPORTS, REPORTS_P1, REPORTS_P2, REPORTS_P3, FIGURES,
              RESAMPLED, CLEAN_NO_DN, CLEAN_DN, FINAL):
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
