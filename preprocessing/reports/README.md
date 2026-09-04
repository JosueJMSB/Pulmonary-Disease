# Informes del preprocesamiento

Cada fase del pipeline escribe sus informes en su propia carpeta. Dentro de cada una, el
prefijo numérico indica la etapa que los produce, de modo que el listado ordenado del
directorio refleja el orden de ejecución. Los archivos **sin prefijo** no pertenecen a una
sola etapa: resumen la fase completa.

```
reports/
├── phase1/   Verificación de datos      · 8 informes
├── phase2/   Estandarización de señal   · 5 informes
├── phase3/   Limpieza de señal          · 9 informes
└── figures/  Gráficos, compartidos
```

Todas las rutas están declaradas en [`config.py`](../config.py); ningún módulo construye un
nombre de archivo por su cuenta.

---

## Fase 1 · Verificación de datos

Producidos por [`phase1_verification.py`](../phase1_verification.py). Esta fase **no modifica
ningún audio**: solo mide, clasifica y deja constancia.

| Informe | Etapa | Contenido |
|---|---|---|
| `1a_integrity.csv` | 1a | Una fila por audio declarado, más una por huérfano encontrado en disco. Contrasta frecuencia, canales, duración y profundidad de bits declarados contra los reales. |
| `1b_annotation_validation.csv` | 1b | Una fila por incidencia detectada en las anotaciones de ciclo respiratorio. **Vacío significa que no hubo ninguna.** |
| `1b_cycles_regenerated.csv` | 1b | Los 6898 ciclos reconstruidos desde los `.txt` originales, con tiempos, banderas de evento y `event_label`. Se compara fila por fila contra la tabla versionada del corpus. |
| `1b_cycle_summary_regenerated.csv` | 1b | Resumen por grabación: número de ciclos, reparto de eventos, duración media y cobertura anotada. |
| `1c_signal_quality.csv` | 1c | Una fila por audio con todas las mediciones de calidad: saturación, RMS, varianza, componente continua, pico, silencio digital, SNR proxy y su estado. **No contiene ninguna columna de admisión**: la selección vive solo en el manifiesto. |
| `1d_duplicates.csv` | 1d | Los grupos de archivos idénticos bit a bit según SHA-256, con su naturaleza (`LABEL_CONFLICT` o `REDUNDANT_COPY`) y la decisión tomada sobre cada miembro. |
| `1e_metadata_validation.csv` | 1e | Una fila por discrepancia entre la metadata unificada y las fuentes clínicas originales, o entre la metadata y el nombre de archivo. **Vacío significa concordancia total.** |
| **`manifest.csv`** | — | **El resultado contractual de la fase.** Una fila por audio con dos ejes independientes: `quality_status` (`PASS` / `REVIEW` / `EXCLUDE`) para la calidad acústica, y `modeling_status` para la elegibilidad. La columna `pipeline_eligible` resume ambos. Es el único archivo que la fase 2 consulta para decidir qué procesar. |

### Por qué dos ejes en el manifiesto

Un archivo duplicado no tiene mala señal: tiene un problema de procedencia. Mezclar ambos
motivos en una sola columna habría ocultado después la causa real de cada baja. `quality_status`
responde «¿esta señal sirve?» y `modeling_status` responde «¿debe entrar al modelo?».

---

## Fase 2 · Estandarización de la señal

Producidos por [`phase2_standardization.py`](../phase2_standardization.py).

| Informe | Etapa | Contenido |
|---|---|---|
| `2a_filter_design.csv` | 2a | Una fila por frecuencia de origen. Especificación medida del FIR anti-aliasing: coeficientes, β, ondulación en banda pasante y atenuación mínima en banda eliminada, con su veredicto. Verificado con `freqz` sobre los coeficientes. |
| `2a_tone_response.csv` | 2a | Ganancia medida al hacer pasar tonos puros por la cadena completa de remuestreo. Nueve frecuencias por cada origen: seis en banda pasante y tres en banda eliminada. |
| `2a_spectral_check.csv` | 2a | Comparación banda a banda del espectro antes y después del remuestreo, sobre una muestra determinista y estratificada de audio real. Un `max_excess_pct` positivo indicaría aliasing. |
| `2b_resampling.csv` | 2b | Una fila por grabación procesada, con 42 columnas de auditoría: hashes de origen y salida, muestras esperadas y obtenidas, error temporal, parámetros del FIR aplicado, picos antes y después, y estado. |
| `2b_resampling_attempt_failed.csv` | 2b | **Solo aparece si una ejecución falla la validación.** Registra el intento descartado. Su ausencia indica que la última ejecución fue correcta. |
| **`validation_summary.csv`** | — | **El veredicto de la fase.** Una fila con el resultado global (`PASS` / `FAIL`), los conteos, el resultado de cada bloque de verificación y la configuración empleada, incluida la versión de SciPy. |

### Cómo leer el veredicto

La fase 2 solo reemplaza su salida anterior si `validation_summary.csv` dice `PASS`. Si dice
`FAIL`, el audio de `data/interim/resampled/` sigue siendo el de la ejecución anterior y la
causa está en las columnas `design_ok`, `tones_ok`, `real_audio_ok` y `structural_ok`.

### Una advertencia sobre `output_sha256` entre ejecuciones

`soundfile`/`libsndfile` escribe un bloque `PEAK` con una marca de tiempo en los WAV de tipo
`FLOAT`. Eso hace que **el archivo cambie de bytes, y por tanto de SHA-256, cada vez que la
fase 2 se reejecuta**, aunque las muestras de audio decodificadas sean idénticas —verificado:
`resample_poly` con los mismos coeficientes produce el mismo array hasta el último bit en
llamadas independientes—. No es un problema de reproducibilidad de la señal, solo de
identidad de archivo entre ejecuciones distintas: la comprobación de la fase 3 sigue siendo
válida porque siempre contrasta contra el `2b_resampling.csv` **de la ejecución de fase 2 más
reciente**, nunca contra uno de una ejecución anterior.

---

## Fase 3 · Limpieza de señal

Producidos por [`phase3_cleaning.py`](../phase3_cleaning.py). Produce dos ramas comparables
—`clean/no_dn/` (pasa-banda + normalización) y `clean/dn/` (además, denoising)— a partir de
las 1249 grabaciones admitidas por la fase 2. Se ejecuta en dos pasadas:
`--calibrar` mide y recomienda sin tocar el corpus completo; sin esa bandera, procesa las
1249 grabaciones con los parámetros ya fijados en `config.py`.

| Informe | Etapa | Contenido |
|---|---|---|
| `calibration_patients.csv` | — | El 20 % de los pacientes (238 → 46), estratificado por dataset y diagnóstico, elegido para calibrar los parámetros de 3b y 3c. Es un contrato: la partición train/test que se defina después debe respetar que estos pacientes queden del lado de entrenamiento. |
| `3a_bandpass_design.csv` | 3a | Respuesta del pasa-banda de una pasada y de las dos que aplica `sosfiltfilt`, medida con `sosfreqz`. Incluye los puntos de −3 dB *efectivos* (no los nominales) y el retardo medido por correlación cruzada. |
| `3a_band_energy.csv` | 3a | Una fila por grabación: fracción de energía bajo 50 Hz, en banda (50–1800 Hz) y sobre 1800 Hz, antes y después del filtro. `energy_inband_pct_before` es la que importa para saber cuánta energía original sobrevive; `_after` es casi siempre ≈100 % por construcción y no debe leerse como lo mismo. |
| `3b_stft_resolution.csv` | 3b | Barrido de ventana y salto de la STFT, medido solo sobre el subconjunto de calibración. |
| `3b_denoising_sweep.csv` | 3b | Barrido en dos tramos —agresividad y suelo espectral—, con cada métrica agregada por media, percentil y extremo. La columna `cumple_restricciones` marca qué configuraciones son elegibles. `snr_proxy_delta_db` se reporta por continuidad con la fase 1 pero **no se usa para elegir**: crece con la agresividad sin darse la vuelta. |
| `3b_denoising_metrics.csv` | 3b | Una fila por grabación del corpus completo (no solo la calibración) con el efecto real del denoising, y la columna `dn_reliable` que marca dónde no es de fiar. |
| `calibration_provenance.csv` | — | Huellas SHA-256 del código, la entrada y los informes usados en la calibración. La ejecución completa se detiene si alguna no coincide. |
| `3c_rms_distribution.csv` | 3c | RMS y pico tras el pasa-banda, sobre el subconjunto de calibración. De aquí sale `TARGET_RMS`. |
| **`manifest.csv`** | — | **El resultado contractual de la fase.** Una fila por audio *y por rama* (2498 filas): ruta de salida, hash, ganancia aplicada, qué límite mandó (`target_rms` / `peak_ceiling` / `max_gain`), los parámetros de denoising cuando la rama es `dn`, y `dn_reliable`. |
| `validation_summary.csv` | — | El veredicto de la fase (`PASS`/`FAIL`), los parámetros empleados y los conteos de las nueve comprobaciones. |

### Qué mide `cycle_gap_power_ratio_db`, y qué no

    cycle_gap_power_ratio_db = 10 · log10( ⟨x²⟩_ciclo / ⟨x²⟩_hueco )

donde el numerador promedia sobre las muestras dentro de un ciclo anotado y el denominador
sobre las que están a más de 100 ms de cualquier límite de ciclo.

**No es una relación señal-ruido y no debe llamarse así.** El hueco entre ciclos no garantiza
ruido puro: puede contener sonido cardiaco, movimiento o respiración sin anotar. Verificado en
el corpus, hay grabaciones que dan valores negativos con la correlación intacta en 0.98 —el
hueco tenía más energía que el propio ciclo, lo que dice algo de la anotación, no del
denoising—.

Su utilidad está en otra cosa: el denominador procede de una región distinta de la señal y no
de la propia distribución que se evalúa, de modo que —a diferencia de la SNR por percentiles—
no crece de forma mecánica al aumentar la agresividad. Sirve para ordenar configuraciones,
no para afirmar cuánto ruido tiene una grabación.

El colchón de 100 ms existe porque los límites anotados son manuales y el sonido respiratorio
no empieza ni termina de golpe: las muestras contiguas a un ciclo contienen ataque o caída del
propio sonido, y contarlas como ruido contamina la referencia con señal.

### La regla de selección de los parámetros de denoising

Se maximiza `cycle_gap_power_ratio_db` sujeta a restricciones sobre la media **y sobre
el percentil**:

| Restricción | Media | Percentil |
|---|---|---|
| Correlación dentro de los ciclos | ≥ 0.95 | p10 ≥ 0.95 |
| Correlación en crepitancias | ≥ 0.95 | p10 ≥ 0.95 |
| Correlación en sibilancias | ≥ 0.95 | p10 ≥ 0.95 |
| Ruido musical (razón de curtosis) | ≤ 1.5 | p90 ≤ 2.0 |
| Distorsión espectral en 50–1800 Hz | ≤ 9 dB | p90 ≤ 10 dB |

No se toma automáticamente el mayor cociente. Entre las configuraciones situadas a no más de
0.20 dB del máximo válido se elige la que maximiza el menor p10 de conservación entre ciclos,
crepitancias y sibilancias; la distorsión, el ruido musical y una menor sobresustracción
resuelven empates. Esta regla evita operar en el borde agresivo por una mejora marginal.

La restricción sobre el percentil no es redundante: medido en la calibración, una correlación
media de 0.973 convive con grabaciones concretas en 0.84, y una regla que solo mire la media
no lo ve. El mínimo se reporta para trazabilidad pero no restringe, porque es prácticamente el
mismo con α=3.0 que con α=5.0 y por tanto no discrimina entre configuraciones.

La conservación se mide por separado para crepitantes y sibilancias, porque no son igual de
frágiles: las sibilancias son tonales y concentran energía en bandas estrechas, mientras que
los crepitantes son transitorios de 5–20 ms y de banda ancha, mucho más sensibles a cualquier
procesado espectral.

### Cuándo el denoising no es de fiar

`dn_reliable` marca las grabaciones donde la rama `dn` requiere revisión: 23 de 1249 (1.8 %).
A nivel individual se marca correlación de ciclo o evento menor de 0.90, ruido musical mayor
de 2.0 o distorsión espectral mayor de 10 dB. El mecanismo
está medido: la estimación de ruido es un percentil bajo **a lo largo del tiempo**, de modo que
se degrada cuando el sonido respiratorio es casi continuo —entonces ese percentil ya no es
ruido sino señal, y sustraerlo multiplicado por α retira contenido real—. Contra la intuición,
las grabaciones más dañadas no son las de poca energía en banda sino las de mucha.

Ninguna se excluye: `no_dn` las conserva intactas y la decisión corresponde al modelado.

### Cómo leer el veredicto

Igual que en la fase 2: `clean/` solo se reemplaza si `validation_summary.csv` dice `PASS`. Si
dice `FAIL`, la salida anterior (o su ausencia, en una primera ejecución) permanece intacta y
el intento queda en `manifest_attempt_failed.csv`.

---

## Convenciones comunes

- **Un informe vacío es un resultado, no un error.** Los informes de validación
  (`1b_annotation_validation.csv`, `1e_metadata_validation.csv`) listan incidencias: que no
  tengan filas significa que no se encontró ninguna.
- **Los informes son versionados.** A diferencia del audio procesado, que se regenera
  ejecutando el pipeline, los informes se conservan en el repositorio porque son la evidencia
  de lo que se midió y cuándo.
- **Ninguna cifra de la memoria de tesis debería escribirse a mano.** Toda afirmación
  cuantitativa sobre el corpus debería poder rastrearse hasta una columna de alguno de estos
  archivos.
