# Informes del preprocesamiento

Cada fase del pipeline escribe sus informes en su propia carpeta. Dentro de cada una, el
prefijo numérico indica la etapa que los produce, de modo que el listado ordenado del
directorio refleja el orden de ejecución. Los archivos **sin prefijo** no pertenecen a una
sola etapa: resumen la fase completa.

```
reports/
├── phase1/   Verificación de datos      · 8 informes
├── phase2/   Estandarización de señal   · 5 informes
├── phase3/   Limpieza de señal          · pendiente
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

---

## Fase 3 · Limpieza de señal

Pendiente de implementación. La carpeta existe para que la estructura sea visible desde ahora.

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
