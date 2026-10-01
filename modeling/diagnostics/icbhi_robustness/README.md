# Diagnóstico de robustez de ICBHI

Este paquete es independiente de holdout_final. No modifica sus splits,
configuraciones ni resultados, no crea un modelo definitivo y nunca lee audio
del holdout externo de 18 pacientes.

## Pregunta que responde

Comprueba si el empate perfecto de validación de los cuatro cofinalistas se
mantiene al cambiar la composición de los folds de los 72 pacientes de
desarrollo: CNN no_dn, CNN dn, CRNN dn y CRNN no_dn_aug.

Los hiperparámetros están congelados en protocol.toml y se contrastan campo
por campo con los best_hyperparameters.json originales. No se vuelven a
muestrear las 20 configuraciones.

## Diseño

- 5 semillas nuevas x 5 folds internos.
- División siempre por paciente.
- Estratificación conjunta por clase y firma de dispositivo. Las firmas con
  menos de cinco pacientes se agrupan dentro de su misma clase para que los
  cinco folds sean factibles, sin mezclar clases ni excluir pacientes.
- El mismo fold para los cuatro cofinalistas.
- Normalización, pesos y denoising calculados únicamente con train.
- Métricas agregadas segmento -> grabación -> paciente.
- Baseline de regresión logística que usa únicamente dispositivo.
- Control negativo CNN no_dn con etiquetas permutadas por paciente.
- Resultados pareados por semilla; el código no proclama un ganador.
- La aleatoriedad del entrenamiento se mantiene fija entre repeticiones para
  aislar el efecto de cambiar la composición de los folds. La comparación no
  pretende medir variabilidad debida a distintas inicializaciones.

Se realizan 100 ajustes de cofinalistas (5 x 5 x 4) y, salvo que se desactive
explícitamente, 25 ajustes del control negativo.

## Orden de ejecución en el servidor

~~~bash
export ROBUST_ROOT=/home/jsivincha/pulmonary_workspace/icbhi_robustness
export ROBUST_DATA=$ROBUST_ROOT/data
export ROBUST_CACHE=/home/jsivincha/pulmonary_workspace/cache/icbhi_robustness
export RUNS_ROOT=/home/jsivincha/pulmonary_workspace/runs
~~~

Las pruebas, que no se ejecutaron al crear esta entrega:

~~~bash
python -m pytest modeling/diagnostics/icbhi_robustness/tests -q
~~~

Crear los cinco repartos:

~~~bash
python -m modeling.diagnostics.icbhi_robustness.build_splits \
  --reference-data-root /home/jsivincha/pulmonary_workspace/data/holdout_calibrated \
  --workspace-root "$ROBUST_ROOT"
~~~

Ver primero el plan de preprocesamiento, sin ejecutarlo:

~~~bash
python -m modeling.diagnostics.icbhi_robustness.run_preprocessing \
  --workspace-root "$ROBUST_ROOT" \
  --data-root "$ROBUST_DATA" \
  --max-parallel 2
~~~

Ejecutar el preprocesamiento:

~~~bash
python -m modeling.diagnostics.icbhi_robustness.run_preprocessing \
  --workspace-root "$ROBUST_ROOT" \
  --data-root "$ROBUST_DATA" \
  --max-parallel 2 \
  --execute
~~~

Validar hashes, datos, selecciones y GPU sin entrenar:

~~~bash
python -u -m modeling.diagnostics.icbhi_robustness.run \
  --workspace-root "$ROBUST_ROOT" \
  --data-root "$ROBUST_DATA" \
  --cache-root "$ROBUST_CACHE" \
  --runs-root "$RUNS_ROOT" \
  --source-runs-root "$RUNS_ROOT" \
  --device cuda:0 \
  --num-workers 0 \
  --dry-run
~~~

Entrenar, seleccionando una única GPU física desde el shell:

~~~bash
CUDA_VISIBLE_DEVICES=2 python -u -m modeling.diagnostics.icbhi_robustness.run \
  --workspace-root "$ROBUST_ROOT" \
  --data-root "$ROBUST_DATA" \
  --cache-root "$ROBUST_CACHE" \
  --runs-root "$RUNS_ROOT" \
  --source-runs-root "$RUNS_ROOT" \
  --device cuda:0 \
  --num-workers 0
~~~

Para reanudar se repite el comando anterior y se agrega --resume.

## Salidas principales

Se escriben bajo runs/icbhi_robustness/run_id:

- robustness_by_repetition.csv;
- robustness_summary.csv;
- paired_comparisons.csv;
- device_baseline_by_repetition.csv;
- permutation_control_by_repetition.csv;
- predicciones, historiales, métricas por fold, dispositivo y figuras dentro
  de cada repetición.

Todos los manifiestos declaran outer_test.accessed=false. Esta prueba no
autoriza abrir el test ni sustituye la posterior decisión metodológica.

