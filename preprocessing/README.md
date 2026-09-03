# Pipeline de preprocesamiento

Implementación del **Resultado esperado 2** del proyecto: un procedimiento reproducible que
asegura la consistencia de las señales de audio procedentes de dos corpus y cuatro
dispositivos de captura distintos, antes de la extracción de características.

El procedimiento está descrito y validado por un especialista en Inteligencia Artificial en
el documento *Pipeline de preprocesamiento de datos*, versión 2. Este README documenta su
traducción a código.

---

## Las cuatro fases

```
FASE 1 · Verificación de datos
   1a  Integridad de los datos
   1b  Validación y regeneración de anotaciones respiratorias
   1c  Calidad de señal
   1d  Duplicados binarios
   1e  Validación contra las fuentes de metadata

FASE 2 · Estandarización de la señal
   2a  Filtro anti-aliasing
   2b  Estandarización de frecuencia a 4 kHz

FASE 3 · Limpieza de señal
   3a  Filtrado pasa-banda 50–1800 Hz
   3b  Denoising
   3c  Normalización de amplitud

FASE 4 · Estandarización temporal
   4a  Segmentación
   4b  Estandarización de duración
```

---

## Correspondencia con el documento

| Etapa del documento | Archivo | Salida |
|---|---|---|
| 1a · Integridad de los datos | `phase1_verification.py` | `reports/integrity.csv` |
| 1b · Validación de anotaciones | `phase1_verification.py` | `reports/icbhi_respiratory_cycles_regenerated.csv`, `reports/icbhi_cycle_summary_regenerated.csv`, `reports/annotation_validation.csv` |
| 1c · Calidad de señal | `phase1_verification.py` | `reports/signal_quality.csv` |
| 1d · Duplicados binarios | `phase1_verification.py` | `reports/duplicate_audio_report.csv` |
| 1e · Validación de metadata | `phase1_verification.py` | `reports/metadata_validation.csv`, `reports/phase1_manifest.csv` |
| 2a · Filtro anti-aliasing | `phase2_standardization.py` | `reports/phase2_filter_design.csv`, `reports/phase2_tone_response.csv`, `reports/phase2_spectral_check.csv` |
| 2b · Estandarización de frecuencia | `phase2_standardization.py` | `data/interim/resampled/`, `reports/resampling.csv`, `reports/phase2_validation_summary.csv` |
| 3a · Filtrado pasa-banda | `phase3_cleaning.py` | — |
| 3b · Denoising | `phase3_cleaning.py` | `reports/denoising_metrics.csv` |
| 3c · Normalización de amplitud | `phase3_cleaning.py` | `data/interim/clean_no_dn/`, `clean_dn/` |
| 4a · Segmentación | `phase4_temporal.py` | `data/final/segments_*.npy`, `segments.csv` |
| 4b · Estandarización de duración | `phase4_temporal.py` | `reports/window_length.csv` |

---

## Estructura

```
preprocessing/
├── config.py                    Todos los parámetros del pipeline
├── utils.py                     Lectura de audio, energía por tramas, SNR, RMS
├── phase1_verification.py
├── phase2_standardization.py
├── phase3_cleaning.py           Pendiente de implementación
├── phase4_temporal.py           Pendiente de implementación
├── run_pipeline.py              Orquestador pendiente
│
├── reports/                     VERSIONADO
│   ├── integrity.csv
│   ├── signal_quality.csv
│   ├── annotation_validation.csv
│   ├── icbhi_respiratory_cycles_regenerated.csv
│   ├── icbhi_cycle_summary_regenerated.csv
│   ├── duplicate_audio_report.csv
│   ├── metadata_validation.csv
│   ├── phase1_manifest.csv
│   ├── resampling.csv
│   ├── phase2_filter_design.csv
│   ├── phase2_tone_response.csv
│   ├── phase2_spectral_check.csv
│   ├── phase2_validation_summary.csv
│   ├── rms_distribution.csv
│   ├── denoising_metrics.csv
│   ├── window_length.csv
│   └── figures/
│
└── data/                        NO VERSIONADO (salvo segments.csv)
    ├── interim/
    │   ├── resampled/           salida de la fase 2
    │   ├── clean_no_dn/         fase 3 · rama sin denoising
    │   └── clean_dn/            fase 3 · rama con denoising
    └── final/
        ├── segments_no_dn.npy
        ├── segments_dn.npy
        └── segments.csv         VERSIONADO
```

Durante la ejecución de la fase 2 aparecen brevemente `data/interim/resampled_staging/` y,
si ya existía una salida previa, `resampled_previous_swap/`. Son transitorios: la fase los
consume al reemplazar `resampled/` de forma atómica y no quedan en disco al terminar, salvo
que la ejecución se interrumpa a mitad de camino. Caen dentro del `.gitignore` igual que el
resto de `data/`.

El audio procesado no se versiona porque es derivable: se regenera ejecutando el pipeline
sobre los datos originales. Ocupa unos 1.3 GB en total.

`segments.csv` sí se versiona: es el registro de trazabilidad de lo que el pipeline produjo.

`phase1_manifest.csv` contiene una fila por audio. `quality_status` usa `PASS`, `REVIEW`
o `EXCLUDE`: una revisión no equivale a eliminar el archivo. `modeling_status` separa las
copias redundantes o con etiquetas contradictorias de los defectos acústicos. La fase 2
solo admite filas con calidad `PASS`/`REVIEW` y `modeling_status=ELIGIBLE`.

---

## La bifurcación del denoising

El documento establece que la señal con reducción de ruido debe compararse con la señal sin
ella. Eso implica que **dos versiones del audio limpio coexisten** a partir de la etapa 3b,
y que la fase 4 se ejecuta sobre ambas, produciendo dos arrays de segmentos.

```
resampled/ ──► pasa-banda ──┬──► normalización ──► clean_no_dn/ ──► segments_no_dn.npy
                            │
                            └──► denoising ──► normalización ──► clean_dn/ ──► segments_dn.npy
```

El denoising se adopta de forma definitiva solo si la mejora de la relación señal-ruido es
sustancial y el rendimiento del modelo no se degrada.

---

## Parámetros

### Fijados por el documento

| Parámetro | Valor | Etapa |
|---|---|---|
| Frecuencia de muestreo objetivo | 4000 Hz | 2b |
| Pasa-banda | 50 – 1800 Hz | 3a |
| Ventana de la STFT | Hann | 3b |
| Solapamiento entre segmentos | 50 % | 4a |
| Métrica de evaluación del denoising | Relación señal-ruido | 3b |

### A determinar en la primera ejecución

El documento establece que estos parámetros se fijan tras observar las distribuciones
reales del corpus. Cada uno tiene una pasada de determinación que produce su informe.

| Parámetro | Etapa | Se determina observando |
|---|---|---|
| Umbral de saturación | 1c | Distribución del porcentaje de muestras en fondo de escala |
| RMS mínimo | 1c | Distribución de RMS, separando fallos de captura de grabaciones tenues |
| Duración de trama, percentiles y SNR mínima | 1c | Distribución de SNR sobre los 1256 audios |
| Longitud y salto de la STFT | 3b | Equilibrio entre resolución espectral y temporal |
| Percentil de estimación del ruido | 3b | Efecto sobre un subconjunto de validación |
| Sobre-sustracción y suelo espectral | 3b | Aparición de ruido musical en el subconjunto |
| RMS objetivo | 3c | `reports/rms_distribution.csv` |
| Longitud de ventana | 4b | `reports/window_length.csv` |

Todos residen en `config.py`, en un único lugar, de modo que las decisiones numéricas del
pipeline sean auditables y el documento y el código no se desincronicen.

---

## Detalles de implementación

### Fase 2 · Remuestreo

La relación entre frecuencias no es entera, por lo que se emplea remuestreo racional
(`scipy.signal.resample_poly`), que integra el filtrado y la decimación en una sola
operación polifásica.

| Origen | Fracción | Archivos |
|---|---|---:|
| 44 100 Hz | 441 / 40 | 824 |
| 10 000 Hz | 5 / 2 | 6 |
| 4 000 Hz | sin transformación | 419 elegibles (426 en el corpus original) |

Los archivos elegibles que ya están a 4 kHz **no se procesan**, para que no reciban un filtrado
adicional que los demás no experimentan y que introduciría una diferencia sistemática entre
subgrupos.

**Filtro anti-aliasing explícito.** `resample_poly` sin argumentos aplica por defecto una
ventana Kaiser con β=5.0 fijo, cuyo corte cae exactamente en el nuevo Nyquist (2000 Hz): ahí
solo ofrece 6 dB de atenuación, y no alcanza 60 dB hasta pasados los 2500 Hz. Sobre este
corpus la energía real por encima de 2000 Hz resultó inferior al 0.02 % en todos los casos
verificados, de modo que el defecto no llegó a corromper audio de forma medible — pero
tampoco era una especificación citable en una tesis. Por eso se diseña un FIR propio con
`scipy.signal.kaiserord` y `firwin`, con banda de transición fija en 1800–2000 Hz:

| Origen | Coeficientes | β | Ondulación en banda pasante | Atenuación desde 2000 Hz |
|---|---:|---:|---:|---:|
| 44 100 Hz | 35 049 | 6.204 | 0.009 dB | 64.97 dB |
| 10 000 Hz | 399 | 6.204 | 0.009 dB | 64.63 dB |

El número de coeficientes depende del factor de interpolación: el FIR se diseña en el
dominio ya interpolado (`fs_origen × up`), que para 44.1 kHz es 1.764 MHz, de modo que una
banda de transición de 200 Hz resulta muy estrecha en términos relativos. El coste
computacional no escala igual: aplicar el filtro de 35 049 coeficientes toma ~0.13 s por
grabación de 20 s frente a ~0.04 s del filtro por defecto, gracias a la implementación
polifásica de `resample_poly`.

La verificación combina tres métodos independientes, todos con reporte propio:

1. **Diseño** (`phase2_filter_design.csv`) — respuesta en frecuencia de los coeficientes con
   `freqz`, sin pasar audio.
2. **Tonos puros** (`phase2_tone_response.csv`) — 18 pruebas (50, 100, 500, 1000, 1600,
   1800, 2050, 2200 y 2500 Hz, por cada frecuencia de origen) que ejercitan la cadena
   completa de `resample_poly`, no solo los coeficientes.
3. **Audio real estratificado** (`phase2_spectral_check.csv`) — los 6 audios de 10 kHz y 12
   por cada dispositivo a 44.1 kHz (36 en total), comparados banda a banda contra el
   original.

**Reemplazo atómico.** La fase escribe en `data/interim/resampled_staging/` y solo
reemplaza `resampled/` si las tres verificaciones y los chequeos estructurales —conteo,
ausencia de `NaN`/`Inf`, identidad exacta de los 419 archivos no transformados, cabeceras
correctas— pasan todos. Si algo falla, la salida anterior permanece intacta, se escribe
`reports/resampling_attempt_failed.csv` con el intento fallido, y el proceso termina con
código de salida distinto de cero. Esto evita el escenario de una reejecución interrumpida
o con umbrales cambiados de la fase 1 dejando archivos huérfanos en `resampled/` que ya no
corresponden a ninguna fila del manifiesto.

Antes de procesar cada grabación se contrasta su SHA-256 contra el que registró la fase 1 en
`phase1_manifest.csv`: si no coincide, el audio cambió entre fases y la grabación falla de
forma explícita en vez de procesarse en silencio.

### Fase 3 · Filtrado de fase cero

El pasa-banda se aplica con filtrado bidireccional, que no introduce desplazamiento
temporal. Es un requisito y no una preferencia: un desfase invalidaría las marcas de tiempo
de las anotaciones de ciclo respiratorio.

### Fase 4 · Orden de ejecución

La etapa 4b se ejecuta **antes** que la 4a, porque la segmentación necesita conocer la
longitud de ventana. El documento las presenta en orden inverso por lógica expositiva.

---

## Trazabilidad de los segmentos

`data/final/segments.csv` conecta cada fila del array `.npy` con su origen:

| Columna | Descripción |
|---|---|
| `idx` | Índice de la fila correspondiente en el `.npy` |
| `dataset` | `ICBHI` o `FRAIWAN` |
| `patient_uid` | Identificador de paciente |
| `audio_id` | Grabación de procedencia |
| `device` | Dispositivo de captura |
| `zone` | Zona de auscultación |
| `filter` | Modo de filtrado (solo Fraiwan) |
| `start_s`, `end_s` | Posición del segmento dentro de la grabación |
| `label` | Diagnóstico heredado del paciente |
| `branch` | `no_dn` o `dn`, según la rama de denoising |

`patient_uid` es la columna que permite particionar por paciente. Todas las ventanas de un
mismo paciente, y los tres modos de filtrado de una misma grabación de Fraiwan, deben
permanecer en la misma partición.

---

## Ejecución

```bash
# fase 1: verificación y manifiesto
python preprocessing/phase1_verification.py

# fase 2: ejecutar solo después de revisar el manifiesto
python preprocessing/phase2_standardization.py
```

**Requisitos:** Python 3.11 o superior. Las versiones empleadas se encuentran fijadas en
`requirements.txt`; `openpyxl` permite contrastar la metadata de Fraiwan con su Excel fuente.

Tiempos de ejecución medidos: fase 1 entre 3 y 5 minutos, fase 2 unos 5 minutos (incluida
la verificación del filtro FIR explícito). Fase 3 se estima entre 8 y 15 minutos y fase 4
en torno a 5; ninguna de las dos está implementada todavía.

---

## Verificación

| Fase | Comprobación |
|---|---|
| 1 | `phase1_manifest.csv` tiene 1256 `audio_id` únicos; toda incidencia lleva estado y razón; los ciclos regenerados se comparan por contenido con los del repositorio; no existen valores infinitos de SNR |
| 2 | El FIR anti-aliasing cumple ≤0.1 dB de ondulación hasta 1800 Hz y ≥60 dB de atenuación desde 2000 Hz, verificado con `freqz`, con tonos puros y con audio real estratificado; los 419 archivos ya a 4 kHz conservan sus muestras exactas (`np.array_equal`); `resampling.csv` tiene 1249 `audio_id` únicos y ninguna fila `FAIL` |
| 3 | `denoising_metrics.csv` contiene la SNR antes y después por grabación; el pasa-banda no desplaza temporalmente la señal |
| 4 | Las filas de `segments.csv` coinciden con la primera dimensión de cada `.npy`; los tres modos de filtrado de un paciente de Fraiwan quedan siempre juntos |
| Global | Una ejecución desde cero sobre el repositorio limpio reproduce `segments.csv` byte a byte |

Las distribuciones de la fase 1 se examinan además desagregadas por dataset, dispositivo y
filtro. Los indicadores de saturación y SNR producen `REVIEW`, no una exclusión automática,
para que una decisión de limpieza no introduzca el mismo sesgo instrumental que el pipeline
busca estudiar.
