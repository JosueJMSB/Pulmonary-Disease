# Plan de optimización: versión ligera de la CNN (Objetivo 3)

Estado: **código escrito, sin ejecutar**. Este documento fija qué se hará, con qué datos, cómo se medirá y qué evidencias quedarán. Los comandos están en [README.md](README.md).

---

## 1. Objetivo, resultados y alcance

**Objetivo 3.** Obtener una versión ligera de las CNN ya seleccionadas y estimar si reduce el uso de recursos computacionales sin comprometer la detección de EPOC.

| Resultado | Qué se hará | Prioridad |
|---|---|---|
| 1. Implementación de la versión ligera | Exportar cada CNN a ONNX FP32 (referencia) y crear su versión ONNX INT8 | Principal |
| 2. Evaluación del modelo optimizado | Medir clasificación, tamaño, memoria y tiempos de ambas versiones en las mismas condiciones | Principal |
| 3. Benchmarking | Informe que reúne y compara las mediciones del resultado 2. No es un experimento nuevo | Se deriva del 2 |

**Alcance.**
- Se trabaja con las tres CNN finales ya seleccionadas. No se entrena ningún modelo nuevo.
- Las mediciones se harán en el servidor phantom y en una laptop. **No prueban el rendimiento ni la compatibilidad en un celular de bajo costo**, porque no se dispone de uno. La validación en un dispositivo físico queda pendiente.
- La herramienta es de tamizaje: no reemplaza la espirometría.

---

## 2. Qué entendemos por dispositivo de bajo costo

**Definición operativa.** Un dispositivo de bajo costo es un equipo de cómputo de uso general, como un celular de gama de entrada o una computadora básica, que podría ejecutar todo el análisis del audio por sí solo. Para eso debe funcionar usando solo su procesador, con RAM y almacenamiento limitados y sin conexión a internet. El costo es bajo no solo por el precio del equipo, sino por lo que el sistema no exige: comprar equipos especializados, pagar servidores o internet, contar con una tarjeta gráfica o con personal especializado. Esta tesis **no fija un precio** ni elige un celular concreto, así que no afirmará que verificó el costo de ningún equipo.

**Perfil de referencia.** No es un requisito validado: sirve solo para interpretar las mediciones.

| Recurso | Referencia | Base |
|---|---|---|
| RAM | 1 a 2 GB | Memoria mínima que Google exige a los celulares de entrada con Android (Go edition): 1 GB en Android 11 y 12, 2 GB en Android 13 (Android Developers, *Android (Go edition)*, consultado el 1 de noviembre de 2026: https://developer.android.com/guide/topics/androidgo) |
| Procesador | Pocos núcleos de bajo consumo, sin tarjeta gráfica | Se simula limitando los hilos (1, 2 y 4) |
| Conexión | Sin internet durante el análisis | Requisito de diseño |

**Metas propuestas.** Son metas, no pruebas de compatibilidad móvil:
- pico de RAM del flujo completo de **100 MB o menos** (10 % de 1 GB);
- archivo INT8 de **10 MB o menos**;
- tiempo total por grabación **menor que la duración de la grabación**.

**Conectividad limitada.** Es un escenario sin internet o con internet inestable. Todo el flujo, desde el WAV hasta la decisión, debe ejecutarse en el equipo, con los modelos, parámetros y dependencias instalados localmente.

**Qué no se afirmará:**
- que el modelo funciona en un celular;
- que su rendimiento está "garantizado";
- que un equipo cumple un precio;
- que la conectividad está verificada sin haber hecho la prueba de la fase 6.

---

## 3. Punto de partida

| Dataset | Condición | Checkpoint (corrida `20261002T151359Z_3edb43`) |
|---|---|---|
| ICBHI | no_dn | `datasets/ICBHI/no_dn/final_model.pt` |
| FRAIWAN_Extended | dn | `datasets/FRAIWAN_Extended/dn/final_model.pt` |
| COMBINED | dn | `datasets/COMBINED/dn/final_model.pt` |

- Son tres conjuntos de pesos de la **misma arquitectura** (1 205 921 parámetros).
- Los MAC estimados describen la arquitectura. La cuantización cambia la representación y el costo de las operaciones, no necesariamente su número, así que no se presentarán como reducción lograda por INT8.

**Lo que no cambia:**
- no hay folds nuevos ni validación cruzada;
- se mantienen el split oficial 80/20 y las mismas grabaciones;
- se mantiene el preprocesamiento final, calibrado solo con desarrollo (`<data_root>/<dataset>/final/`);
- se mantienen la normalización del checkpoint, el umbral 0,5 y la agregación segmento → grabación → paciente;
- no se modifican `preprocessing/`, `modeling/`, `runs/` ni sus configuraciones.

**Test externo.** El test externo **ya se usó** para evaluar los modelos `.pt` en la corrida final. Sus métricas publicadas se reutilizan y el `.pt` no se vuelve a evaluar. La única evaluación nueva en test será la del INT8, una vez por dataset.

---

## 4. Versiones que se comparan

| Versión | Formato | Para qué sirve |
|---|---|---|
| Original | `.pt` | Origen; se conserva sin cambios. Sus métricas de test publicadas son la referencia de clasificación |
| Referencia | ONNX FP32 | Base de comparación de tamaño, memoria y tiempo. No se evalúa en test |
| Optimizada | ONNX INT8 | Versión ligera, por cuantización estática |

ONNX es el **formato** y ONNX Runtime el **motor** que lo ejecuta. PyTorch solo se usa para cargar y exportar los checkpoints. Se eligió ONNX Runtime porque phantom tiene PyTorch 2.11, versión en la que PyTorch está trasladando sus herramientas de cuantización a torchao.

---

## 5. Correcciones incorporadas

1. **Paridad diferenciada.**
   - PyTorch → ONNX FP32: tolerancia estricta y bloqueante (|Δlogit| ≤ 1e-4 y |Δprob| ≤ 1e-5). Los cambios de decisión se reportan aparte.
   - ONNX FP32 → INT8: sin tolerancia de aprobación. Se registran las diferencias de puntuación (media, mediana, p95 y máximo) y los cambios de decisión por segmento, grabación y paciente.
2. **Paquete completo (`model_bundle.json`).**
   - Contiene la normalización, la configuración acústica y log-mel, todos los parámetros del preprocesamiento final (incluido `peak_ceiling`), las clases, el umbral y la regla de agregación.
   - Tiene bloques **separados** `models.fp32` y `models.int8`, cada uno con su hash, tamaño, versiones y metadatos. Si falta un campo, el paquete se rechaza.
3. **Rutas configurables** (CLI → entorno → `config.toml`).
   - En el servidor, los datos están en `/home/jsivincha/pulmonary_workspace/data/holdout_calibrated/`.
   - Las salidas van fuera del repositorio, en `PULMONARY_OPT_ROOT`.
4. **Inferencia sin PyTorch.**
   - `infer.py` usa solo el ONNX, el paquete, onnxruntime y los módulos de `preprocessing/` y `modeling/features/`, que no importan torch.
   - `check_env.py` y `tests/test_no_torch.py` lo comprueban.
5. **Etiquetas correctas.**
   - Clasificación en test: "base `.pt` (publicado)" frente a "ONNX INT8". A ONNX FP32 no se le atribuyen métricas de test.
   - Recursos: FP32 frente a INT8, siempre con entradas de desarrollo.
6. **IOV de memoria.** Se verifica con el pico de RAM del flujo completo. La memoria del modelo aislado es explicativa. Si solo bajan los pesos y no el pico del flujo completo, no se afirmará que bajó la memoria necesaria para usar el sistema.
7. **Congelación con criterio previo.**
   - `[acceptance]` fija, antes de ejecutar, la pérdida máxima (0,05) en balanced accuracy, recall de EPOC y recall de la clase negativa, comparando INT8 con FP32 por paciente de desarrollo.
   - Si no se cumple, la congelación se bloquea y el intento queda en `attempts.jsonl`.
   - Al congelar se guarda `acceptance_sha256`, y `evaluate_test` comprueba que no haya cambiado.
   - Esas métricas de desarrollo solo comparan versiones: no estiman el desempeño.
8. **Redacción del test.** El registro solo impide **repetir la evaluación INT8** de cada dataset; nunca describe el test como no consultado.
9. **Dependencias reales de la laptop:** `onnxruntime`, `numpy`, `scipy`, `pandas`, `soundfile`, `librosa`, `scikit-learn` y `psutil`, más el código indicado en el README.
10. **Orden seguro.** pytest → exportación → cuantización → `check_pipeline` → `compare_dev --freeze` → evaluación en test. `evaluate_test` exige `pipeline_check.json` en PASS y `freeze.json`.
11. **Reanudación.** El registro guarda una huella de la evaluación: dataset, hashes del INT8 y del paquete, criterio, datos y versión de onnxruntime.
    - Con la misma huella y la misma corrida, se reanuda según el punto donde se cortó:
      - tras STARTED, sin archivo de predicciones, se vuelve a inferir;
      - tras STARTED, con el archivo ya escrito, se recupera ese archivo, validado fila por fila contra los segmentos de test;
      - tras PREDICTIONS_WRITTEN, se reutilizan las predicciones, previa comprobación de su hash.
    - Se bloquea una evaluación nueva, distinta o ya completada.
    - Un bloqueo por dataset (`<dataset>_int8.lock`) impide dos evaluaciones simultáneas.
    - Las inconsistencias exigen revisión manual.
12. **Este documento** se crea o actualiza según exista.
13. **Revisión del código (segunda ronda).**
    - La prueba sin internet exige que la corrida online sí detecte conexión. El resultado se da por dataset, con sus hashes, y el informe solo marca CUMPLE los datasets probados con esos mismos modelos.
    - La caída de desempeño se verifica por separado en BA, recall de EPOC y recall de la clase negativa.
    - En test se informan las ocho métricas del resultado base, con IC.
    - `check_audio` comprueba antes de empezar que existan los WAV originales de desarrollo.

---

## 6. Procedimiento

| Fase | Script | Plataforma | Datos |
|---|---|---|---|
| 0 | `check_env`, `check_audio`, `pytest` | ambas (`check_audio` en phantom) | Sintéticos; `check_audio` solo comprueba que existan los WAV de desarrollo |
| 1 | `export_onnx` | phantom | Desarrollo (paridad FP32) |
| 2 | `quantize` | phantom | Desarrollo (calibración) |
| 3 | `check_pipeline`, `compare_dev --freeze` | phantom | Desarrollo |
| 4 | `evaluate_test` | phantom | Test, una vez por dataset y solo para INT8 |
| 5 | `benchmark prepare/run` | phantom (laptop opcional) | Desarrollo |
| 6 | `offline_check` | laptop | Desarrollo |
| 7 | `report` | phantom | Salidas anteriores |

**Medición de recursos.**
- FP32 e INT8 se ejecutan en la misma CPU, con las mismas entradas, lote de 1 segmento, mismos hilos (1, 2 y 4), mismas opciones de ONNX Runtime, mismo calentamiento y mismas repeticiones.
- Se reportan mediana y p95 del tiempo por segmento y del tiempo total WAV → decisión.
- El pico de RAM se mide en un proceso nuevo para cada modo.
- Se registran la CPU, la memoria y la carga del servidor antes y después.
- No se presupone que INT8 será más rápido.

---

## 7. Indicadores

Tamaño, memoria y tiempo se expresan en porcentaje relativo a ONNX FP32.

| Resultado | IOV original (compromiso) | Meta más estricta (hipótesis) | Medición |
|---|---|---|---|
| 1 | Tamaño reducido ≥ 5 % | ≥ 50 % y archivo ≤ 10 MB | ONNX FP32 frente a INT8 |
| 2 | Caída de hasta 20 puntos en el desempeño, verificada por separado en BA, recall de EPOC y recall de la clase negativa | ≤ 5 puntos en balanced accuracy | Por paciente en test: INT8 frente al `.pt` publicado |
| 2 | Tiempo que no empeore más de 30 % | ≤ 0 % y tiempo total menor que la duración de la grabación | Mediana WAV → decisión |
| 2 | Memoria reducida ≥ 5 % | ≥ 50 % y ≤ 100 MB | Pico de RAM del flujo completo |
| 2 | Validación con un experto en IA | Se mantiene, documentada | Acta (`expert_review_template.md`) |
| 3 | Benchmarking con ≥ 5 parámetros, con énfasis en recursos | Se mantiene | `report/benchmark_table.*` |

El límite original de 20 puntos se conserva, pero es demasiado permisivo para afirmar que no se compromete la detección. Esa conclusión se apoyará en la meta de 5 puntos y en el recall de EPOC.

**Métricas clínicas, por paciente:** las ocho del resultado base (accuracy, balanced accuracy, recall de EPOC, recall de la clase negativa, F1-macro, mínimo recall por clase, AUROC y AUPRC de EPOC), la matriz de confusión e intervalos por bootstrap (misma configuración que la evaluación final). La concordancia por segmento es solo una comprobación técnica.

**FRAIWAN_Extended:** su test tiene 9 pacientes, 2 con EPOC. Sus conclusiones serán prudentes.

---

## 8. Evidencias que quedarán

- Modelos ONNX FP32 e INT8 y paquete con hashes de origen y de salida.
- Versiones de librerías y entorno de cada fase.
- Resultado de la paridad FP32.
- Parámetros de cuantización, inventario de calibración y operadores en INT8 y en FP32.
- Comparación en desarrollo, intentos y congelación.
- Comprobación del flujo WAV → decisión.
- Registro de la evaluación INT8 del test, con predicciones y métricas por paciente.
- Latencias crudas, picos de RAM y condiciones de medición.
- Prueba sin internet.
- Tabla final y estado de cada IOV.

---

## 9. Riesgos y limitaciones

- No hay un dispositivo de bajo costo físico.
- INT8 podría no ser más rápido en la CPU de phantom, o algunos operadores podrían quedar en FP32. Se reportará lo que ocurra.
- Parte del pico de RAM corresponde a ONNX Runtime y a las librerías de audio. Por eso se informan también los modos de base.
- El test de FRAIWAN_Extended es muy pequeño.
- Los audios provienen de estetoscopios electrónicos y micrófonos de investigación.
- El exportador de PyTorch 2.11 podría exigir `dynamo = true` en `config.toml`.
- `audio_pipeline` admite grabaciones de origen a 4, 10 y 44,1 kHz, que son las frecuencias que contempla el preprocesamiento. Otras se rechazan de forma explícita.
