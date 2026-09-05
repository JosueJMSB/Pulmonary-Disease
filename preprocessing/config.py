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

# Las dos ramas de la fase 3 comparten un unico padre. Eso convierte su
# reemplazo atomico en un solo renombrado (CLEAN <-> CLEAN_staging) en vez de
# un intercambio transaccional de dos directorios hermanos independientes.
CLEAN = INTERIM / "clean"
CLEAN_NO_DN = CLEAN / "no_dn"
CLEAN_DN = CLEAN / "dn"

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

# Fase 3 - Limpieza de senal
R3_CALIBRATION = REPORTS_P3 / "calibration_patients.csv"
R3_BANDPASS_DESIGN = REPORTS_P3 / "3a_bandpass_design.csv"
R3_BAND_ENERGY = REPORTS_P3 / "3a_band_energy.csv"
R3_STFT_RESOLUTION = REPORTS_P3 / "3b_stft_resolution.csv"
R3_DENOISING_SWEEP = REPORTS_P3 / "3b_denoising_sweep.csv"
R3_DENOISING_METRICS = REPORTS_P3 / "3b_denoising_metrics.csv"
R3_CALIBRATION_PROVENANCE = REPORTS_P3 / "calibration_provenance.csv"
R3_RMS_DISTRIBUTION = REPORTS_P3 / "3c_rms_distribution.csv"
R3_NORMALIZATION = REPORTS_P3 / "3c_normalization.csv"
PHASE3_MANIFEST = REPORTS_P3 / "manifest.csv"
R3_FAILED_ATTEMPT = REPORTS_P3 / "manifest_attempt_failed.csv"
R3_SUMMARY = REPORTS_P3 / "validation_summary.csv"

# Fase 4 - Estandarizacion temporal
REPORTS_P4 = REPORTS / "phase4"
R4_INPUT_VALIDATION = REPORTS_P4 / "4_input_validation.csv"
R4_WINDOW_LENGTH = REPORTS_P4 / "4b_window_length.csv"
R4_SEGMENT_SUMMARY = REPORTS_P4 / "4a_segment_summary.csv"
R4_FAILED_ATTEMPT = REPORTS_P4 / "validation_attempt_failed.csv"
R4_SUMMARY = REPORTS_P4 / "validation_summary.csv"

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

# Subconjunto de calibracion: se eligen los parametros de 3b y 3c observando
# solo estas grabaciones, nunca el corpus completo. Es una fraccion de
# PACIENTES, no de grabaciones, para que los tres modos de filtrado de un
# mismo paciente de Fraiwan queden siempre del mismo lado. La lista elegida se
# persiste en reports/phase3/calibration_patients.csv y la particion train/test
# que se defina mas adelante queda obligada a colocar a estos pacientes en el
# lado de entrenamiento: en la fase 3 esa particion aun no existe, de modo que
# "evitar los datos de prueba" solo puede cumplirse fijando ahora el contrato
# y respetandolo despues.
CALIBRATION_FRACTION = 0.20
CALIBRATION_SEED = 20250903

# --- 3a Pasa-banda ---
BANDPASS_LOW = COMMON_BAND_LOW
BANDPASS_HIGH = COMMON_BAND_HIGH
BANDPASS_ORDER = 4

# sosfiltfilt filtra en ambos sentidos: la respuesta efectiva es |H|^2, de modo
# que la atenuacion real en los bordes nominales es de 6 dB y no de 3. Se mide
# y se reporta en 3a_bandpass_design.csv; no se corrige, porque el filtrado de
# fase cero es un requisito y no una preferencia.
BANDPASS_RIPPLE_TOL_DB = 0.1       # ondulacion maxima admitida entre 100 y 1500 Hz

# Umbral de marcado (no de exclusion) por bajo contenido en la banda util.
# Medido antes de esta fase sobre una muestra estratificada: la fraccion de
# energia que sobrevive al pasa-banda varia entre 0.095 % y 77.8 % segun la
# grabacion. 1 % separa el puñado de casos extremos -practicamente sin
# contenido en banda- del resto de la distribucion; se revisa con
# reports/phase3/3a_band_energy.csv, que cubre las 1249 grabaciones.
MIN_INBAND_ENERGY_PCT = 1.0

# --- 3b Denoising (sustraccion espectral) ---
STFT_WINDOW = "hann"

# Rejillas evaluadas en el modo --calibrar. La rejilla de sobre-sustraccion
# llega hasta 5.0 para que contenga el punto realmente elegido y el cruce de
# la restriccion de ruido musical: un valor fijado fuera de lo barrido no es
# trazable desde el informe.
STFT_CANDIDATES = (128, 256, 512)              # 32, 64 y 128 ms a 4 kHz
OVERLAP_FRACTION_CANDIDATES = (0.50, 0.75)
NOISE_PCT_CANDIDATES = (5, 10, 15, 20)
OVERSUBTRACTION_CANDIDATES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0)
SPECTRAL_FLOOR_CANDIDATES = (0.002, 0.01, 0.05)

# Colchon excluido de la referencia de ruido a cada lado de un ciclo anotado.
# Los limites son manuales y el sonido respiratorio no empieza ni termina de
# golpe: las muestras contiguas a un ciclo contienen ataque o caida del propio
# sonido, y contarlas como ruido contamina la referencia con senal.
CYCLE_GUARD_MS = 100

# Restricciones de la regla de seleccion. La SNR proxy por percentiles NO se
# usa para elegir (crece con alpha sin optimo interior: mide su propia
# agresividad). Se maximiza el cociente de potencia ciclo-hueco, sujeto a
# conservar tanto los ciclos completos como sus eventos adventicios y a
# limitar el ruido musical y la distorsion espectral en la banda util.
#
# Se restringe sobre la media Y sobre el percentil 10, no solo sobre la media:
# medido en la calibracion, la correlacion media (0.973 con alpha=4.0) oculta
# grabaciones concretas que bajan a 0.84. El percentil 10 captura la
# degradacion sistematica sin que una sola grabacion atipica vete toda la
# rejilla; el minimo se reporta para trazabilidad pero no restringe, porque es
# practicamente el mismo con alpha=3.0 que con 5.0 y por tanto no discrimina
# entre configuraciones.
MIN_CYCLE_CORRELATION = 0.95        # media, dentro de los ciclos anotados
MIN_CYCLE_CORRELATION_P10 = 0.95    # percentil 10 entre grabaciones
MIN_CRACKLE_CORRELATION = 0.95      # media en ciclos con crepitantes
MIN_CRACKLE_CORRELATION_P10 = 0.95  # percentil 10 en ciclos con crepitantes
MIN_WHEEZE_CORRELATION = 0.95       # media en ciclos con sibilancias
MIN_WHEEZE_CORRELATION_P10 = 0.95   # percentil 10 en ciclos con sibilancias
MAX_MUSICAL_NOISE_RATIO = 1.5       # media de la razon de curtosis (despues/antes)
MAX_MUSICAL_NOISE_RATIO_P90 = 2.0   # percentil 90 entre grabaciones
MAX_SPECTRAL_DISTORTION_DB = 9.0    # media en 50-1800 Hz
MAX_SPECTRAL_DISTORTION_DB_P90 = 10.0

# Si varias configuraciones quedan a menos de este margen del mejor cociente
# ciclo-hueco, gana la que mejor preserve el peor p10 entre ciclos, crepitantes
# y sibilancias. Evita escoger el borde mas agresivo por una mejora marginal.
POWER_RATIO_TOLERANCE_DB = 0.20

# Umbrales por grabacion. No excluyen archivos: marcan dn_reliable=False para
# que una comparacion posterior conserve el mismo subconjunto en ambas ramas.
MIN_RECORDING_CYCLE_CORRELATION = 0.90
MIN_RECORDING_EVENT_CORRELATION = 0.90
MAX_RECORDING_MUSICAL_NOISE_RATIO = 2.0
MAX_RECORDING_SPECTRAL_DISTORTION_DB = 10.0

# Fijados tras la calibracion sobre 46 pacientes (117 grabaciones de ICBHI con
# hueco anotado suficiente). Todos los numeros citados salen de
# reports/phase3/3b_stft_resolution.csv y 3b_denoising_sweep.csv.
#
#   Resolucion   las seis combinaciones dan metricas casi identicas con
#                alpha=1.5 (cociente 1.96-2.01 dB). Se elige 256/192
#                (64 ms, 75 % de solape) en vez del maximo nominal 512/256,
#                que solo gana 0.027 dB,
#                porque 512/256 es peor en las cuatro metricas de conservacion:
#                correlacion 0.9970 vs 0.9980, crepitantes 0.9977 vs 0.9984,
#                sibilancias 0.9957 vs 0.9970 y distorsion 3.53 vs 2.49 dB. El
#                solape del 75 % gana al del 50 % en las tres ventanas por
#                igual, y una ventana mas corta preserva mejor los crepitantes,
#                que son transitorios de 5-20 ms.
#
#   Agresividad  el maximo admisible es percentil=15 con alpha=5.0. La regla
#                robusta admite hasta 0.20 dB de diferencia en el objetivo y
#                elige el mejor peor-p10 entre ciclos, crepitantes y
#                sibilancias. Por eso selecciona percentil=15 con alpha=4.0:
#
#                             p15/a4.0      p15/a5.0    limite
#                  objetivo    2.424 dB      2.536 dB     margen 0.20
#                  corr p10    0.9700        0.9582       0.95
#                  crepit. p10 0.9711        0.9601       0.95
#                  sibilan.p10 0.9697        0.9588       0.95
#                  musical     1.3217        1.4001       1.50
#                  musical p90 1.6328        1.8015       2.00
#                  distorsion  7.58 dB       8.83 dB      9.00
#
#   Suelo        con p15/a4.0, beta=0.05 pierde solo 0.050 dB frente a 0.01,
#                pero mejora el peor p10 (0.9756 vs 0.9697), reduce la
#                distorsion (6.04 vs 7.58 dB) y baja el ruido musical.
STFT_NPERSEG = 256
STFT_NOVERLAP = 192
NOISE_PCT = 15
OVERSUBTRACTION = 4.0
SPECTRAL_FLOOR = 0.05

# --- 3c Normalizacion de amplitud ---
PEAK_CEILING = 0.95             # el pico final no puede rebasarlo

# TARGET_RMS: mediana del RMS post-pasabanda sobre el subconjunto de
# calibracion (281 grabaciones). MAX_GAIN: la ganancia que ese objetivo pide
# en la rama con denoising (que arranca de un RMS mas bajo, al haberse
# retirado energia real) tiene p99=18.5x y maximo 37.7x, con solo 3 patrones
# de un puñado de pacientes por encima de 20x. Un tope de 20x afecta al 1.1 %
# de las filas (6 de 562, ambas ramas) y son precisamente las grabaciones que
# conviene marcar para revision, no amplificar sin limite.
TARGET_RMS = 0.03012
MAX_GAIN = 20.0

# ---------------------------------------------------------------------------
# Fase 4 - Estandarizacion temporal
# ---------------------------------------------------------------------------

SEGMENT_OVERLAP = 0.50

# Duraciones evaluadas en la etapa 4b. Medido sobre el corpus real (fase 3,
# 1249 grabaciones): 8 s deja a 58 pacientes con menos de 5 segmentos y a uno
# con cero, lo que rompe la agregacion a nivel de paciente prevista para el
# etiquetado debil; 6 s produce una cola sin cubrir de 2.00 s en las
# grabaciones de 20 s -la mitad del corpus- porque el salto de 3 s no encaja
# en esa duracion, y su contencion de ciclos (74.1 %) cae por debajo de la de
# 5 s (81.1 %): la contencion no es monotona con la longitud, depende de como
# encaje la rejilla en la duracion real. Entre las dos que quedan viables,
# 4 s (11294 segmentos, 44.4 % de ventanas con ciclo completo, mediana de 20
# segmentos por paciente) y 5 s (8747 segmentos, 65.3 %, mediana de 15), se
# elige 5 s: gana 21 puntos de contencion de ciclos a cambio de un 23 % menos
# de segmentos, cubre exactamente las grabaciones de 20 s sin cola, y ninguna
# grabacion se pierde -la mas corta del corpus dura justo 5.00 s-.
SEGMENT_CANDIDATES = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
SEGMENT_SECONDS = 5.0

# Los segmentos son la entrada del modelo, no una etapa intermedia. Se
# guardan en coma flotante de 32 bits para que la senal llegue exactamente
# como la dejo la fase 3, sin recuantizar: unos 667 MB por rama sobre 269 GB
# libres. int16 habria ahorrado la mitad y habria sido seguro -el techo de
# pico en 0.95 impide el recorte-, pero no hay razon para pagar una perdida
# que no hace falta.
SEGMENT_DTYPE = "float32"


def ensure_dirs():
    """Crea los directorios de salida si no existen."""
    for d in (REPORTS, REPORTS_P1, REPORTS_P2, REPORTS_P3, REPORTS_P4, FIGURES,
              RESAMPLED, CLEAN_NO_DN, CLEAN_DN, FINAL):
        d.mkdir(parents=True, exist_ok=True)


# Parametros que la fase 3 debe tener fijados antes de una ejecucion completa.
# MIN_INBAND_ENERGY_PCT queda fuera deliberadamente: es un umbral de marcado
# para el informe, no algo de lo que dependa la correccion del procesamiento.
PHASE3_REQUIRED_PARAMETERS = (
    "STFT_NPERSEG", "STFT_NOVERLAP", "NOISE_PCT",
    "OVERSUBTRACTION", "SPECTRAL_FLOOR", "TARGET_RMS", "MAX_GAIN",
)


def pending_parameters():
    """Devuelve los parametros que siguen sin fijar."""
    here = globals()
    names = [
        "MAX_SATURATION_PCT", "MIN_RMS", "MIN_SNR_DB",
        *PHASE3_REQUIRED_PARAMETERS,
        "SEGMENT_SECONDS",
    ]
    return [n for n in names if here[n] is None]


def pending_phase3_parameters():
    """Parametros de la fase 3 que faltan por fijar, con el informe que los determina."""
    here = globals()
    sources = {
        "STFT_NPERSEG": "reports/phase3/3b_stft_resolution.csv",
        "STFT_NOVERLAP": "reports/phase3/3b_stft_resolution.csv",
        "NOISE_PCT": "reports/phase3/3b_denoising_sweep.csv",
        "OVERSUBTRACTION": "reports/phase3/3b_denoising_sweep.csv",
        "SPECTRAL_FLOOR": "reports/phase3/3b_denoising_sweep.csv",
        "TARGET_RMS": "reports/phase3/3c_rms_distribution.csv",
        "MAX_GAIN": "reports/phase3/3c_rms_distribution.csv",
    }
    return [(n, sources[n]) for n in PHASE3_REQUIRED_PARAMETERS if here[n] is None]
