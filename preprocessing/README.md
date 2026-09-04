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
| 1a · Integridad de los datos | `phase1_verification.py` | `reports/phase1/1a_integrity.csv` |
| 1b · Validación de anotaciones | `phase1_verification.py` | `reports/phase1/1b_cycles_regenerated.csv`, `1b_cycle_summary_regenerated.csv`, `1b_annotation_validation.csv` |
| 1c · Calidad de señal | `phase1_verification.py` | `reports/phase1/1c_signal_quality.csv` |
| 1d · Duplicados binarios | `phase1_verification.py` | `reports/phase1/1d_duplicates.csv` |
| 1e · Validación de metadata | `phase1_verification.py` | `reports/phase1/1e_metadata_validation.csv`, `reports/phase1/manifest.csv` |
| 2a · Filtro anti-aliasing | `phase2_standardization.py` | `reports/phase2/2a_filter_design.csv`, `2a_tone_response.csv`, `2a_spectral_check.csv` |
| 2b · Estandarización de frecuencia | `phase2_standardization.py` | `data/interim/resampled/`, `reports/phase2/2b_resampling.csv`, `validation_summary.csv` |
| 3a · Filtrado pasa-banda | `phase3_cleaning.py` | `reports/phase3/3a_bandpass_design.csv`, `3a_band_energy.csv` |
| 3b · Denoising | `phase3_cleaning.py` | `reports/phase3/3b_stft_resolution.csv`, `3b_denoising_sweep.csv` |
| 3c · Normalización de amplitud | `phase3_cleaning.py` | `data/interim/clean/no_dn/`, `clean/dn/`, `reports/phase3/manifest.csv` |
| 4a · Segmentación | `phase4_temporal.py` | `data/final/segments_*.npy`, `segments.csv` |
| 4b · Estandarización de duración | `phase4_temporal.py` | `reports/phase4/4b_window_length.csv` |

---

## Estructura

```
preprocessing/
├── config.py                    Todos los parámetros del pipeline
├── utils.py                     Lectura de audio, energía por tramas, SNR, RMS
├── phase1_verification.py
├── phase2_standardization.py
├── phase3_cleaning.py
├── phase4_temporal.py           Pendiente de implementación
├── run_pipeline.py              Orquestador pendiente
│
├── reports/                     VERSIONADO · ver reports/README.md
│   ├── README.md                Índice de todos los informes
│   ├── phase1/
│   │   ├── 1a_integrity.csv
│   │   ├── 1b_annotation_validation.csv
│   │   ├── 1b_cycles_regenerated.csv
│   │   ├── 1b_cycle_summary_regenerated.csv
│   │   ├── 1c_signal_quality.csv
│   │   ├── 1d_duplicates.csv
│   │   ├── 1e_metadata_validation.csv
│   │   └── manifest.csv         Resultado contractual de la fase
│   ├── phase2/
│   │   ├── 2a_filter_design.csv
│   │   ├── 2a_tone_response.csv
│   │   ├── 2a_spectral_check.csv
│   │   ├── 2b_resampling.csv
│   │   └── validation_summary.csv
│   ├── phase3/
│   │   ├── calibration_patients.csv
│   │   ├── 3a_bandpass_design.csv
│   │   ├── 3a_band_energy.csv
│   │   ├── 3b_stft_resolution.csv
│   │   ├── 3b_denoising_sweep.csv
│   │   ├── 3b_denoising_metrics.csv
│   │   ├── 3c_rms_distribution.csv
│   │   ├── manifest.csv         Resultado contractual de la fase (2 filas por audio)
│   │   └── validation_summary.csv
│   └── figures/
│
└── data/                        NO VERSIONADO (salvo segments.csv)
    ├── interim/
    │   ├── resampled/           salida de la fase 2
    │   └── clean/               salida de la fase 3
    │       ├── no_dn/           pasa-banda + normalización
    │       └── dn/              + denoising
    └── final/
        ├── segments_no_dn.npy
        ├── segments_dn.npy
        └── segments.csv         VERSIONADO
```

Durante la ejecución de una fase con reemplazo atómico aparece brevemente un directorio
`<destino>_staging` (y, si ya existía una salida previa, `<destino>_previous_swap`). Son
transitorios: la fase los consume al reemplazar el destino y no quedan en disco al terminar,
salvo que la ejecución se interrumpa a mitad de camino. Caen dentro del `.gitignore` igual que
el resto de `data/`. La fase 3 anida sus dos ramas bajo un único padre (`clean/`) precisamente
para que ese reemplazo sea un solo renombrado en vez de un intercambio transaccional de dos
carpetas hermanas.

El audio procesado no se versiona porque es derivable: se regenera ejecutando el pipeline
sobre los datos originales. Ocupa unos 1.2 GB en total (390 MB en `resampled/`, 779 MB en
`clean/`, entre las dos ramas).

`segments.csv` sí se versiona: es el registro de trazabilidad de lo que el pipeline produjo.

`reports/phase1/manifest.csv` contiene una fila por audio. `quality_status` usa `PASS`, `REVIEW`
o `EXCLUDE`: una revisión no equivale a eliminar el archivo. `modeling_status` separa las
copias redundantes o con etiquetas contradictorias de los defectos acústicos. La fase 2
solo admite filas con calidad `PASS`/`REVIEW` y `modeling_status=ELIGIBLE`.

---

## La bifurcación del denoising

El documento establece que la señal con reducción de ruido debe compararse con la señal sin
ella. Eso implica que **dos versiones del audio limpio coexisten** a partir de la etapa 3b,
y que la fase 4 se ejecuta sobre ambas, produciendo dos arrays de segmentos.

```
resampled/ ──► pasa-banda ──┬──► normalización ──► clean/no_dn/ ──► segments_no_dn.npy
                            │
                            └──► denoising ──► normalización ──► clean/dn/ ──► segments_dn.npy
```

**`no_dn` es la rama de referencia y `dn` la experimental**, hasta que la comparación de
modelos diga otra cosa. La fase 3 no toma esa decisión: produce ambas ramas, verificadas como
distintas entre sí, y la evidencia para decidir. `no_dn` solo lleva pasa-banda y normalización
—operaciones cuyo efecto está acotado y medido—, mientras que `dn` añade una sustracción
espectral cuyos parámetros se eligieron optimizando una métrica que, aun siendo la mejor
disponible, no es una SNR real. Mientras esa comparación no exista, lo defendible es tratar
`no_dn` como la línea base.

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

### Determinados por calibración

Elegidos con `python phase3_cleaning.py --calibrar`, observando solo el subconjunto de
calibración (20 % de los pacientes, nunca el corpus completo) y con la regla de selección
documentada en `reports/README.md`. El razonamiento completo de cada valor está en los
comentarios de `config.py`.

| Parámetro | Etapa | Valor | Determinado con |
|---|---|---:|---|
| Ventana / salto de la STFT | 3b | 256 / 192 muestras (64 ms, 75 % solape) | `reports/phase3/3b_stft_resolution.csv` |
| Percentil de estimación del ruido | 3b | 15 | `reports/phase3/3b_denoising_sweep.csv` |
| Sobre-sustracción (α) | 3b | 4.0 | ídem |
| Suelo espectral (β) | 3b | 0.01 | ídem |
| RMS objetivo | 3c | 0.03012 | `reports/phase3/3c_rms_distribution.csv` |
| Tope de ganancia | 3c | 20× | ídem |

### Pendientes (fase 4)

| Parámetro | Etapa | Se determina observando |
|---|---|---|
| Longitud de ventana | 4b | `reports/phase4/4b_window_length.csv` |

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

1. **Diseño** (`phase2/2a_filter_design.csv`) — respuesta en frecuencia de los coeficientes con
   `freqz`, sin pasar audio.
2. **Tonos puros** (`phase2/2a_tone_response.csv`) — 18 pruebas (50, 100, 500, 1000, 1600,
   1800, 2050, 2200 y 2500 Hz, por cada frecuencia de origen) que ejercitan la cadena
   completa de `resample_poly`, no solo los coeficientes.
3. **Audio real estratificado** (`phase2/2a_spectral_check.csv`) — los 6 audios de 10 kHz y 12
   por cada dispositivo a 44.1 kHz (36 en total), comparados banda a banda contra el
   original.

**Reemplazo atómico.** La fase escribe en `data/interim/resampled_staging/` y solo
reemplaza `resampled/` si las tres verificaciones y los chequeos estructurales —conteo,
ausencia de `NaN`/`Inf`, identidad exacta de los 419 archivos no transformados, cabeceras
correctas— pasan todos. Si algo falla, la salida anterior permanece intacta, se escribe
`reports/phase2/2b_resampling_attempt_failed.csv` con el intento fallido, y el proceso termina con
código de salida distinto de cero. Esto evita el escenario de una reejecución interrumpida
o con umbrales cambiados de la fase 1 dejando archivos huérfanos en `resampled/` que ya no
corresponden a ninguna fila del manifiesto.

Antes de procesar cada grabación se contrasta su SHA-256 contra el que registró la fase 1 en
`reports/phase1/manifest.csv`: si no coincide, el audio cambió entre fases y la grabación falla de
forma explícita en vez de procesarse en silencio.

### Fase 3 · Limpieza de señal

**3a — Filtrado de fase cero.** Butterworth de orden 4 aplicado con `sosfiltfilt`, que filtra
en ambos sentidos y no introduce desplazamiento temporal: verificado con ruido blanco y
correlación cruzada, el retardo medido es de 0 muestras exactas. Es un requisito y no una
preferencia, porque un desfase invalidaría las marcas de tiempo de las anotaciones de ciclo
respiratorio. Al filtrar en ambos sentidos, la atenuación *efectiva* en los bordes nominales
(50 y 1800 Hz) es de 6 dB y no de 3: los puntos de −3 dB reales caen dentro de la banda
nominal (medido: 55.5–1777.7 Hz), un detalle que casi ninguna implementación documenta.

**3b — Denoising con un objetivo que no se auto-cumple.** La sustracción espectral con suelo
proporcional (`|S|² = max(|X|² − α·ruido, β·|X|²)`) se valida primero con una **prueba nula**:
con α=0 y β=1, la cadena STFT→sustracción→ISTFT debe reproducir la entrada con error de
precisión de máquina (verificado: 4.4·10⁻¹⁶). Si no lo hiciera, el emparejamiento
ventana/salto/reconstrucción estaría roto y toda medición posterior sería un artefacto.

Los parámetros no se eligen maximizando la SNR por percentiles: esa métrica crece con la
agresividad sin límite, porque el propio método reduce el suelo de ruido con el que se calcula.
Se maximiza en su lugar `cycle_gap_power_ratio_db` —potencia dentro de los ciclos anotados
frente a la de los huecos entre ellos, con un colchón de 100 ms a cada lado—. La configuración
solo es elegible si conserva los ciclos, crepitancias y sibilancias con media y percentil 10
≥ 0.95, mantiene el ruido musical dentro de sus límites y restringe la distorsión espectral
en 50–1800 Hz. Los extremos también se reportan y los casos degradados se marcan individualmente.

Esa métrica **no es una SNR** y el código y los informes la nombran en consecuencia: el
hueco entre ciclos no garantiza ruido puro. Su valor está en que el denominador procede de una
región distinta de la señal, y por eso no crece de forma mecánica con α. La justificación
completa está en `reports/README.md`.

**Cuándo el denoising no es de fiar.** El manifiesto marca con `dn_reliable` las 23 grabaciones
(1.8 %) en que la rama `dn` requiere revisión. El mecanismo está medido: el ruido se estima como un
percentil bajo a lo largo del tiempo, así que se degrada cuando el sonido respiratorio es casi
continuo —ahí ese percentil ya no es ruido sino señal—. Contra la intuición, las más dañadas no
son las de poca energía en banda sino las de mucha.

**3c — Normalización con tres límites.** `ganancia = min(RMS_objetivo / RMS_actual,
techo_pico / pico_actual, ganancia_máxima)`. El techo de pico por sí solo no basta: se
verificó que grabaciones con RMS y pico bajos a la vez reciben del techo permiso para
amplificarse mucho más de lo que pide el objetivo de RMS (hasta 85× cuando el RMS pedía 18×),
lo que amplificaría sobre todo su ruido. El tope de ganancia (20×) afecta solo al 1.1 % de las
filas del manifiesto, precisamente los casos que conviene marcar y no silenciar.

**Calibración por paciente.** Los cinco parámetros de 3b y el objetivo de 3c se eligen
observando solo el 20 % de los pacientes (`reports/phase3/calibration_patients.csv`), nunca
el corpus completo: en esta fase la partición train/test todavía no existe, de modo que la
única forma de que esos parámetros no queden ajustados sobre datos que después sean de prueba
es fijar ahora esa lista como contrato y que la partición posterior la respete.

La ejecución completa verifica además la huella SHA-256 del código, la entrada de fase 2,
la selección de pacientes y los informes de calibración. Si cualquiera cambió desde
`--calibrar`, se detiene antes de reemplazar los audios.

**Reemplazo atómico.** Igual que la fase 2, escribe en `data/interim/clean_staging/` (con las
dos ramas dentro) y solo reemplaza `clean/` si las nueve comprobaciones pasan, incluida que
las dos ramas resulten distintas entre sí —evidencia de que el denoising tuvo efecto, no una
suposición—.

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

# fase 3, en dos pasadas:
python preprocessing/phase3_cleaning.py --calibrar   # mide y recomienda; no escribe audio
python preprocessing/phase3_cleaning.py               # con los parametros ya fijados en config.py
```

La segunda pasada de la fase 3 se niega a correr si algún parámetro de `config.py` sigue en
`None`, e indica qué informe de `--calibrar` lo determina.

**Requisitos:** Python 3.11 o superior. Las versiones empleadas se encuentran fijadas en
`requirements.txt`; `openpyxl` permite contrastar la metadata de Fraiwan con su Excel fuente.

Tiempos de ejecución medidos: fase 1 entre 3 y 5 minutos, fase 2 unos 5 minutos, fase 3
alrededor de 5 minutos por pasada (calibración y ejecución completa). Fase 4 no está
implementada todavía; se estima en torno a 5 minutos.

---

## Verificación

| Fase | Comprobación |
|---|---|
| 1 | `reports/phase1/manifest.csv` tiene 1256 `audio_id` únicos; toda incidencia lleva estado y razón; los ciclos regenerados se comparan por contenido con los del repositorio; no existen valores infinitos de SNR |
| 2 | El FIR anti-aliasing cumple ≤0.1 dB de ondulación hasta 1800 Hz y ≥60 dB de atenuación desde 2000 Hz, verificado con `freqz`, con tonos puros y con audio real estratificado; los 419 archivos ya a 4 kHz conservan sus muestras exactas (`np.array_equal`); `resampling.csv` tiene 1249 `audio_id` únicos y ninguna fila `FAIL` |
| 3 | Prueba nula de STFT/ISTFT con error < 1e-9; retardo del pasa-banda = 0 muestras; `clean/no_dn/` y `clean/dn/` tienen 1249 archivos cada una, sin `NaN`/`Inf`, con la misma duración que la fase 2; ningún pico supera 0.95 ni ninguna ganancia supera el tope; las dos ramas resultan distintas entre sí; `reports/phase3/manifest.csv` tiene 2498 filas (1249 × 2 ramas) |
| 4 | Las filas de `segments.csv` coinciden con la primera dimensión de cada `.npy`; los tres modos de filtrado de un paciente de Fraiwan quedan siempre juntos |
| Global | Una ejecución desde cero sobre el repositorio limpio reproduce `segments.csv` byte a byte |

Las distribuciones de la fase 1 se examinan además desagregadas por dataset, dispositivo y
filtro. Los indicadores de saturación y SNR producen `REVIEW`, no una exclusión automática,
para que una decisión de limpieza no introduzca el mismo sesgo instrumental que el pipeline
busca estudiar.
