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
   1b  Extracción de anotaciones de los ciclos respiratorios
   1c  Calidad de señal

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
| 1b · Extracción de anotaciones | `phase1_verification.py` | `icbhi_respiratory_cycles.csv`, `icbhi_cycle_summary.csv` |
| 1c · Calidad de señal | `phase1_verification.py` | `reports/signal_quality.csv`, `reports/exclusions.csv` |
| 2a · Filtro anti-aliasing | `phase2_standardization.py` | — |
| 2b · Estandarización de frecuencia | `phase2_standardization.py` | `data/interim/resampled/` |
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
├── phase3_cleaning.py
├── phase4_temporal.py
├── run_pipeline.py              Orquestador
│
├── reports/                     VERSIONADO
│   ├── integrity.csv
│   ├── signal_quality.csv
│   ├── exclusions.csv
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

El audio procesado no se versiona porque es derivable: se regenera ejecutando el pipeline
sobre los datos originales. Ocupa unos 1.3 GB en total.

`segments.csv` sí se versiona: es el registro de trazabilidad de lo que el pipeline produjo.

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

La relación entre frecuencias no es entera, por lo que se emplea remuestreo racional. El
filtro anti-aliasing y la decimación se aplican en una sola operación polifásica, en el
orden que exige el documento.

| Origen | Fracción | Archivos |
|---|---|---:|
| 44 100 Hz | 441 / 40 | 824 |
| 10 000 Hz | 5 / 2 | 6 |
| 4 000 Hz | sin transformación | 426 |

Los archivos que ya están a 4 kHz **no se procesan**, para que no reciban un filtrado
adicional que los demás no experimentan y que introduciría una diferencia sistemática entre
subgrupos.

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
# pipeline completo
python preprocessing/run_pipeline.py

# una fase concreta, para reanudar sin repetir lo anterior
python preprocessing/run_pipeline.py --phase 2
```

**Requisitos:** Python 3.11 o superior, con `numpy`, `scipy`, `pandas` y `soundfile`.

Tiempos de ejecución aproximados: fase 1 entre 3 y 5 minutos, fase 2 entre 10 y 20, fase 3
entre 8 y 15, fase 4 unos 5. El pipeline completo se ejecuta en menos de tres cuartos de
hora.

---

## Verificación

| Fase | Comprobación |
|---|---|
| 1 | Admitidos más excluidos suman 1256; cada baja de `exclusions.csv` lleva criterio y valor; los 6898 ciclos regenerados coinciden con los del repositorio |
| 2 | El espectro de una grabación de 44.1 kHz no conserva energía apreciable sobre 2000 Hz tras el remuestreo; los 426 archivos ya a 4 kHz conservan su hash MD5 |
| 3 | `denoising_metrics.csv` contiene la SNR antes y después por grabación; el pasa-banda no desplaza temporalmente la señal |
| 4 | Las filas de `segments.csv` coinciden con la primera dimensión de cada `.npy`; los tres modos de filtrado de un paciente de Fraiwan quedan siempre juntos |
| Global | Una ejecución desde cero sobre el repositorio limpio reproduce `segments.csv` byte a byte |

Las distribuciones de la fase 1 se examinan además desagregadas por dispositivo, para
comprobar que las exclusiones no se concentren en un instrumento concreto e introduzcan el
mismo sesgo instrumental que el pipeline busca evitar.
