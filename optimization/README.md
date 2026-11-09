# optimization/ — Versión ligera de la CNN (Objetivo 3)

Este código convierte las tres CNN finales a **ONNX FP32** (referencia) y crea su versión optimizada **ONNX INT8**. Luego mide clasificación, tamaño, memoria y tiempos de ambas versiones. El plan completo está en [PLAN-OPTIMIZACION.md](PLAN-OPTIMIZACION.md).

> **Estado:** el código está escrito, pero **ninguna fase se ha ejecutado**. Tampoco se corrió pytest, ni se exportó o cuantizó ningún modelo. Los comandos de abajo los ejecutará el usuario, en phantom y en la laptop.

## Qué no cambia

- Se usan las tres CNN finales de la corrida `20261002T151359Z_3edb43`: `ICBHI/no_dn`, `FRAIWAN_Extended/dn` y `COMBINED/dn`. No hay reentrenamiento, folds nuevos ni validación cruzada.
- Se mantienen el split 80/20, las grabaciones, el preprocesamiento final, la normalización, el umbral 0,5 y la agregación segmento → grabación → paciente.
- La calibración INT8 usa **solo pacientes de desarrollo** (`role == "train"`).
- No se modifica `preprocessing/`, `modeling/` ni la corrida final en `runs/`.

## Archivos

| Archivo | Fase | Plataforma | Qué hace |
|---|---|---|---|
| `config.toml` | — | ambas | Corrida final, datasets, tolerancias, criterio de aceptación, benchmark, IOV y metas |
| `paths.py` | — | ambas | Rutas: CLI → variable de entorno → `config.toml` |
| `bundle.py` | — | ambas | Paquete `model_bundle.json`: todos los parámetros de inferencia y bloques separados `models.fp32` / `models.int8` |
| `audio_pipeline.py` | — | ambas | WAV → remuestreo → pasa-banda → reducción de ruido (dn) → normalización → segmentos → log-mel |
| `scoring.py`, `parity.py` | — | ambas | Agregación, métricas por paciente, criterios FP32 e INT8, criterio de aceptación |
| `ort_session.py` | — | ambas | Sesión de ONNX Runtime con opciones fijas |
| `dev_data.py` | — | phantom | Carga verificada (por hash) de la corrida final y de los segmentos por rol |
| `access_registry.py` | 4 | phantom | Registro de la evaluación INT8 del test: reanudable, bloquea repeticiones |
| `envinfo.py` | — | ambas | Versiones, CPU, RAM y carga del sistema |
| `check_env.py` | 0 | ambas | Verifica dependencias y que `infer.py` no cargue PyTorch |
| `check_audio.py` | 0 | phantom | Verifica que existan los WAV originales de desarrollo (y su hash) |
| `export_onnx.py` | 1 | phantom | `.pt` → ONNX FP32, paridad estricta y paquete |
| `quantize.py` | 2 | phantom | ONNX FP32 → ONNX INT8 (cuantización estática, calibración con desarrollo) |
| `check_pipeline.py` | 3 | phantom | Paridad del flujo WAV → decisión frente a los segmentos almacenados |
| `compare_dev.py` | 3 | phantom | INT8 frente a FP32 en desarrollo y congelación condicionada |
| `evaluate_test.py` | 4 | phantom | Evaluación INT8 única en test, comparada con el `.pt` publicado |
| `benchmark.py` | 5 | phantom (laptop opcional) | Tamaño, latencia, tiempo total y pico de RAM |
| `infer.py` | 6 | ambas | Inferencia WAV → decisión **sin PyTorch** |
| `offline_check.py` | 6 | laptop | Prueba con y sin internet |
| `report.py` | 7 | phantom | Tabla de benchmarking y estado de los IOV |
| `expert_review_template.md` | 7 | — | Plantilla del acta del experto en IA |
| `tests/` | — | ambas | Pruebas con datos sintéticos (no ejecutadas todavía) |

## Rutas

Ninguna ruta personal está en el código. Cada ruta se resuelve con el argumento de la CLI, después con la variable de entorno y, por último, con el valor de `config.toml`:

| Argumento | Variable | Contenido |
|---|---|---|
| `--data-root` | `PULMONARY_DATA_ROOT` | `<data_root>/<dataset>/final/segments.csv`, `segments_<rama>.npy`, `preprocessing_params.json` |
| `--runs-root` | `PULMONARY_RUNS_ROOT` | `<runs_root>/final_training/20261002T151359Z_3edb43/` (o `--final-run-dir`) |
| `--output-root` | `PULMONARY_OPT_ROOT` | Salidas de optimization (por defecto `<runs_root>/optimization`) |
| `--run-dir` / `--run-id` | — | Una corrida concreta de optimization |

En el servidor, los datos finales están en `/home/jsivincha/pulmonary_workspace/data/holdout_calibrated/`. Conviene que las salidas queden **fuera del repositorio**:

```bash
export PULMONARY_DATA_ROOT=/home/jsivincha/pulmonary_workspace/data/holdout_calibrated
export PULMONARY_RUNS_ROOT=/home/jsivincha/pulmonary_workspace/runs
export PULMONARY_OPT_ROOT=/home/jsivincha/pulmonary_workspace/runs/optimization
```

### WAV originales (requisito previo)

`check_pipeline`, `benchmark run` y `offline_check prepare --copy-audio` leen los **WAV originales** que figuran en la columna `source_path` de `preprocessing/reports/phase2/2b_resampling.csv`. **No bastan** los `.npy` de segmentos ni los WAV remuestreados que se transfirieron antes.

- Las rutas `source_path` son relativas (por ejemplo, `Audios_Respiratorios/Dataset1-ICBHI/...`). Por defecto se buscan desde la raíz del repositorio. Si los audios están en otra carpeta, se indica con `--audio-root <raíz>` en `check_audio`, `check_pipeline`, `benchmark run` y `offline_check prepare`.
- Antes del paso 1 hay que ejecutar `check_audio` (paso 0b). Comprueba que existan todas las grabaciones de desarrollo y que su SHA-256 coincida con el informe de la fase 2. Si devuelve `FALTA`, primero hay que copiar los audios.

## Comandos en phantom (servidor)

Se ejecutan desde la raíz del repositorio, con el entorno de PyTorch del proyecto activado. **El orden importa**: la evaluación INT8 del test es única, así que las pruebas, `check_pipeline` y la congelación deben pasar antes.

```bash
# 0. Dependencias (las instala el usuario) y entorno
pip install -r optimization/requirements-server.txt
python -m optimization.check_env --role server

# 0b. WAV originales de desarrollo disponibles y con el hash de la fase 2
python -m optimization.check_audio --dataset all
#     Si los audios no estan bajo la raiz del repositorio, por ejemplo:
#     python -m optimization.check_audio --dataset all --audio-root /ruta/que/contiene/Audios_Respiratorios

# 1. Pruebas automatizadas, antes de tocar datos reales
python -m pytest optimization/tests -q

# 2. Exportación a ONNX FP32 + paridad estricta (crea la corrida opt_v1)
python -m optimization.export_onnx --dataset all --run-id opt_v1

# 3. Cuantización INT8 (calibración solo con desarrollo)
python -m optimization.quantize --dataset all --run-id opt_v1

# 4. Flujo WAV -> decisión frente a los segmentos almacenados
python -m optimization.check_pipeline --dataset all --run-id opt_v1

# 5. Comparación en desarrollo y congelación (solo si cumple el criterio)
python -m optimization.compare_dev --dataset all --run-id opt_v1 --freeze

# 6. Evaluación INT8 en test: una vez por dataset.
#    Si se interrumpe, se repite el MISMO comando y se reanuda.
python -m optimization.evaluate_test --dataset ICBHI --run-id opt_v1 --confirm-test-access
python -m optimization.evaluate_test --dataset FRAIWAN_Extended --run-id opt_v1 --confirm-test-access
python -m optimization.evaluate_test --dataset COMBINED --run-id opt_v1 --confirm-test-access

# 7. Benchmark de recursos (solo entradas de desarrollo)
python -m optimization.benchmark prepare --run-id opt_v1
python -m optimization.benchmark run --run-id opt_v1 --label phantom

# 8. Carpeta de transferencia para la laptop
python -m optimization.offline_check prepare --run-id opt_v1 --copy-audio

# 9. Informe (después de traer offline_comparison.json desde la laptop)
#    Reemplace la ruta por la del archivo traido desde la laptop.
python -m optimization.report --run-id opt_v1 --label phantom \
    --offline-comparison /ruta/a/offline_comparison.json
#    Cuando el acta del experto este firmada, se agrega su ruta:
#    python -m optimization.report --run-id opt_v1 --label phantom \
#        --offline-comparison /ruta/a/offline_comparison.json --expert-review /ruta/a/acta_firmada.pdf
```

Si los WAV no están bajo la raíz del repositorio, se añade el mismo `--audio-root <raíz>` a los pasos 4, 7 (`benchmark run`) y 8.

Si `compare_dev --freeze` devuelve el código 3, el INT8 no cumplió el criterio de aceptación. Solo se puede intentar de nuevo ajustando la cuantización con datos de desarrollo (`quantize --overwrite`). Cada intento queda en `attempts.jsonl`.

**Evaluación del test (paso 6):**
- Solo puede haber una evaluación en curso por dataset. Al empezar se crea `<output_root>/_test_access/<dataset>_int8.lock`, que se borra al terminar, también tras un error o Ctrl+C.
- Si el proceso muere sin borrarlo (corte de luz, `kill -9`), el comando siguiente se detiene y muestra el PID y el equipo que lo creó. Borre el `.lock` a mano solo después de confirmar que ese proceso ya no existe; luego repita el mismo comando.
- Si la interrupción ocurrió después de escribir las predicciones y antes de actualizar el registro, el archivo se **recupera** sin volver a inferir. Antes se comprueba fila por fila que corresponde a los segmentos de test. Si no corresponde, se detiene y pide revisión manual.

## Comandos en la laptop (Windows, sin PyTorch)

1. Copiar a la laptop la carpeta `<corrida>/transfer/` generada en el paso 8. La laptop debe tener también el código del repositorio:
   - `optimization/`
   - `preprocessing/` (solo el código)
   - `modeling/__init__.py`
   - `modeling/features/`
   - `modeling/evaluation.py`
2. Instalar las dependencias reales del flujo de audio en un entorno sin PyTorch: `onnxruntime`, `numpy`, `scipy`, `pandas`, `soundfile`, `librosa`, `scikit-learn` y `psutil`.

```powershell
python -m venv .venv-opt
.venv-opt\Scripts\activate
pip install -r optimization\requirements-laptop.txt
python -m optimization.check_env --role laptop        # confirma que torch no se carga
python -m pytest optimization\tests -q                 # las pruebas de torch/onnx se omiten

# Prueba sin internet: primero con conexión (debe detectarla)...
python -m optimization.offline_check run --transfer-dir <transfer> --mode online --out-dir <salida>
# ...luego con el wifi y el cable de red APAGADOS...
python -m optimization.offline_check run --transfer-dir <transfer> --mode offline --out-dir <salida>
# ...y la comparación (puede ser con la red encendida)
python -m optimization.offline_check compare --out-dir <salida>

# Opcional: repetir las mediciones de recursos en la laptop
python -m optimization.benchmark run --bundle-root <transfer> --audio-root <transfer> --label laptop --out-dir <salida>\benchmark
```

Las pruebas de `optimization/tests` no leen ningún archivo de `modeling/configs`, así que en la laptop basta con las carpetas indicadas en el punto 1.

`compare` da un resultado **por dataset**, junto con los hashes de FP32, INT8 y el paquete probados. El resultado es `PASS` solo si se cumplen tres condiciones:
- la corrida `online` detectó conexión y la `offline` no;
- el dataset aparece en ambas corridas, con los mismos modelos;
- las predicciones de FP32 e INT8 son idénticas y todas las grabaciones terminaron en OK.

Si la conectividad no cambió como se esperaba, el resultado es `NO VERIFICADO`.

Después se copia `<salida>/offline_comparison.json` a phantom para el paso 9.

## Criterios

- **PyTorch → ONNX FP32:** tolerancia estricta y bloqueante (`[parity_fp32]`). Los cambios de decisión se reportan aparte.
- **ONNX FP32 → INT8:** sin tolerancia de aprobación. Se registran las diferencias de puntuación y los cambios de decisión por segmento, grabación y paciente.
- **Aceptación del INT8:** `[acceptance]` fija la pérdida máxima permitida en balanced accuracy, recall de EPOC y recall de la clase negativa, por paciente de desarrollo. Se fija antes de ejecutar nada. `evaluate_test` se niega a acceder al test si ese criterio cambió después de congelar.
- **Test externo:** ya se usó para evaluar los modelos `.pt`; sus métricas publicadas se reutilizan. El registro solo impide repetir la evaluación INT8 de cada dataset.
- **Métricas en test:** las mismas ocho del resultado base, cada una con su IC por bootstrap: accuracy, balanced accuracy, recall de EPOC, recall de la clase negativa, macro-F1, mínimo recall por clase, AUROC y AUPRC de EPOC. La comparación también incluye los IC publicados del `.pt`.
- **Caída de desempeño (IOV):** se verifica **por separado** en cada métrica de `[iov] performance_metrics` (BA, recall de EPOC y recall de la clase negativa). El IOV global solo se marca CUMPLE si las tres cumplen. La meta más estricta (`max_ba_drop_points`) se aplica solo a BA y así se rotula.
- **Sin internet (IOV):** se marca por dataset. Un dataset solo cumple si figura en `offline_comparison.json` con estado PASS y con los mismos hashes de FP32, INT8 y paquete que su corrida. Probar un solo dataset no aprueba los otros.
- **Tamaño:** se compara ONNX FP32 con ONNX INT8. El `.pt` se informa aparte.
- **Memoria:** el IOV se verifica con el pico de RAM del **flujo completo** (modo `pipeline`). Los modos `model` y de base son explicativos.
- **Tiempos:** se dan mediana y p95 del tiempo por segmento y del tiempo total WAV → decisión, en la misma CPU y condiciones para ambas versiones.

## Salidas

```
<output_root>/
├── _test_access/<dataset>_int8.json      registro de la evaluación INT8 del test
└── opt_v1/
    ├── run_config.json
    ├── <dataset>/
    │   ├── model_fp32.onnx, model_int8.onnx, model_bundle.json
    │   ├── export_parity.json (+ CSV)          paridad estricta FP32
    │   ├── calibration_inventory.csv, quantization_ops.json, attempts.jsonl
    │   ├── pipeline_check.json (+ CSV)
    │   ├── dev_comparison.json (+ CSV), freeze.json
    │   └── test_int8/                          métricas, IC, predicciones, comparación con el .pt
    ├── benchmark/inputs/<dataset>/             entradas fijas de desarrollo
    ├── benchmark/<plataforma>/<dataset>/       tamaños, latencias, tiempos, memoria, entorno
    ├── transfer/                               carpeta para la laptop
    └── report/                                 benchmark_table.{csv,md}, iov_status.{csv,md}
```

## Lista de comprobación después de ejecutar

- [ ] `check_env` en estado OK en phantom y en la laptop.
- [ ] `check_audio` en estado OK en phantom (WAV originales de desarrollo con el hash de la fase 2).
- [ ] `pytest optimization/tests` sin fallos en phantom.
- [ ] `export_parity.json` con estado PASS en los tres datasets.
- [ ] `quantization_ops.json` revisado: qué operadores quedaron en FP32.
- [ ] `pipeline_check.json` con estado PASS.
- [ ] `freeze.json` en los tres datasets (o razones documentadas en `attempts.jsonl`).
- [ ] Evaluación INT8 completada por dataset (`_test_access/*.json` en COMPLETED).
- [ ] Benchmark con condiciones y carga del servidor registradas.
- [ ] `offline_comparison.json` con estado PASS en los tres datasets.
- [ ] `report/iov_status.md` revisado y acta del experto firmada.

## Limitaciones

Las mediciones en phantom y en la laptop permiten comparar FP32 con INT8. **No prueban el rendimiento ni la compatibilidad en un celular de bajo costo**: la validación en un dispositivo físico queda pendiente.
