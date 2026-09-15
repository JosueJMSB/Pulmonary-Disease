# Modelado

## Preparacion de COPD frente a control

La seleccion clinica es una salida derivada de la fase 4 del preprocesamiento.
No modifica los arreglos originales ni realiza particion, escalado, balanceo
o aumento.

```bash
python -m modeling.prepare_task_data
```

Produce dos directorios bajo `modeling/data/copd_vs_control/`:

- `ICBHI`: COPD frente a Healthy, todas las grabaciones elegibles.
- `FRAIWAN_Extended`: COPD frente a Normal, solo el filtro Extended.

Cada directorio contiene `segments_no_dn.npy` y `segments_dn.npy`
perfectamente alineados, un inventario con `task_array_index` y
`source_array_index`, y un manifiesto con conteos y hashes. Los archivos
`.npy` son derivados locales y no se versionan.

Para regenerar deliberadamente una salida existente:

```bash
python -m modeling.prepare_task_data --overwrite
```

---

## SVM-RBF: COPD frente a Control

Implementacion completa segun `PLAN.md`.

**Requisitos:** Python 3.12 o superior -las versiones fijadas de NumPy
(2.5.2) y SciPy (1.18.1) en `requirements.txt` lo exigen-. Instalar las
dependencias de modelado (ademas de las del pipeline de preprocesamiento):

```bash
pip install -r requirements-modeling.txt
```

### Estructura

```text
modeling/
├── configs/svm_rbf.toml     Todos los parametros del experimento
├── features/
│   ├── logmel.py            STFT compartida, log-mel, MFCC/delta/delta2
│   └── acoustic.py          8 descriptores acusticos por trama
├── models/
│   └── svm_rbf.py           Busqueda de (C, gamma), fold, modelo final
├── data.py                  Rutas, condiciones, cache de caracteristicas, pesos
├── splits.py                Particion por paciente (calibracion + K-fold anidado)
├── evaluation.py            Agregacion, metricas, bootstrap (independiente del modelo)
├── artifacts.py             Runs, staging atomico, figuras, serializacion de modelos
├── run_experiment.py        CLI unico
└── prepare_task_data.py
```

`data.py`, `splits.py`, `evaluation.py` y `artifacts.py` no conocen la SVM:
los reutilizaran la CNN y la CRNN cuando existan.

### Ejecucion

```bash
python -u -m modeling.run_experiment \
  --model svm_rbf \
  --dataset all \
  --experiment all \
  --n-jobs 4 \
  --data-root /ruta/a/copd_vs_control \
  --runs-root /ruta/a/runs
```

Sin `--data-root`/`--runs-root`, se usan `PULMONARY_DATA_ROOT`/
`PULMONARY_RUNS_ROOT`/`PULMONARY_CACHE_ROOT` y, si tampoco existen,
`modeling/data/copd_vs_control`, `modeling/runs/svm` y
`modeling/cache/features` bajo la raiz del repositorio. Ningun valor por
defecto es una ruta personal.

Flags utiles:

- `--dry-run`: valida formas, hashes, conteos y folds sin entrenar nada.
- `--smoke-test`: un solo fold y una sola combinacion (C, gamma); confirma
  que la ejecucion completa (features, fold, figuras, modelo) funciona antes
  de lanzar la busqueda completa.
- `--force-features`: ignora la cache de caracteristicas y la regenera.
- `--resume [RUN_ID]`: continua la ultima ejecucion (o una concreta),
  reutilizando folds ya publicados y reintentando solo los que fallaron o
  quedaron a medias.

### Experimentos y condiciones

| Experimento | Dataset | Condicion | Rama | Poblacion |
|---|---|---|---|---|
| `main` | ICBHI | `main_no_dn` | no_dn | Todas las grabaciones elegibles |
| `main` | FRAIWAN_Extended | `main_no_dn` | no_dn | Todas las grabaciones Extended |
| `denoising_ablation` | ICBHI | `no_dn_reliable` | no_dn | Solo `dn_reliable=True` |
| `denoising_ablation` | ICBHI | `dn_reliable` | dn | Solo `dn_reliable=True` |
| `denoising_ablation` | FRAIWAN_Extended | `main_no_dn` (reutilizada) | no_dn | Todas (ya son 100% fiables) |
| `denoising_ablation` | FRAIWAN_Extended | `dn` | dn | Todas |

Ver el razonamiento completo (por que ICBHI_120 se excluye de la ablacion, por
que Fraiwan reutiliza `main_no_dn`) en los comentarios de `configs/svm_rbf.toml`.

### Salidas de una ejecucion

```text
<runs-root>/svm_rbf/<run_id>/
├── run.log, status.json, run_fingerprint.json
├── resolved_config.toml, environment.txt
├── folds.csv, hyperparameter_search.csv, metrics_by_fold.csv
├── metrics_summary.csv, oof_patient_predictions.csv, baseline_metrics.csv
├── classification_report.csv, confusion_matrix.csv
├── figures/                     PNG (300 DPI) + PDF + CSV de cada figura
└── datasets/<dataset>/<condicion>/
    ├── fold_01 .. fold_05/
    │   ├── grid_search.csv, selected_hyperparameters.json
    │   ├── test_{segment,recording,patient}_predictions.csv
    │   ├── baseline_metrics.csv     DummyClassifier, mismo test que la SVM
    │   ├── test_metrics.json
    │   └── model_fold.joblib        scaler + SVM de ESE fold (reproducible)
    ├── oof_patient_predictions.csv, metrics_by_fold.csv, baseline_metrics.csv, ...
    └── final/
        ├── final_model.joblib       scaler + SVM + metadata de inferencia
        ├── model_metadata.json
        └── model_sha256.txt
```

`run_fingerprint.json` es lo que verifica `--resume`: hash de los datos de
entrada (`segments.csv`/`segments_<rama>.npy` de cada dataset usado), huella
de la configuracion (acustica, splits, SVM, seleccion, bootstrap) y el
`--dataset`/`--experiment` pedidos. Si algo de eso cambio desde la ejecucion
original, `--resume` se niega en vez de mezclar resultados incompatibles.

`baseline_metrics.csv` compara la SVM con un `DummyClassifier` (estrategias
`most_frequent` y `stratified`), entrenado y evaluado exactamente igual que
la SVM de ese fold: es la comprobacion de cordura minima antes de creer
cualquier metrica de la SVM en aislamiento.

`final_model.joblib` guarda `label_mapping` con el nombre real de la clase
negativa del dataset (`Healthy` en ICBHI, `Normal` en Fraiwan), no una
etiqueta generica.

Los CSV/JSON/figuras del run son texto y se pueden versionar si se desea
conservar la evidencia de una ejecucion concreta; `.joblib`, `runs/` y
`cache/` completos quedan fuera de git por tamano (ver `.gitignore`).

### Pruebas

```bash
pytest modeling/tests
```

Cubren extraccion de caracteristicas (formas, ausencia de NaN/Inf en
silencio/tono/constante), particion por paciente (sin fuga, calibracion
siempre en train), agregacion y metricas (contra calculos manuales) y un
fold completo de la SVM sobre datos sinteticos separables. Nunca tocan
`modeling/data/copd_vs_control/`: son verificaciones del codigo, no una
ejecucion del experimento.

---

## CNN 2D propia: COPD frente a Control

Implementacion segun `PLAN-CNN.md`: CNN de 4 bloques convolucionales en
PyTorch, **1 205 921 parametros**, sobre el Log-Mel `1 x 64 x 309` (misma STFT
y mismo Log-Mel que la SVM). Reutiliza pacientes, folds, condiciones,
agregacion, metricas, bootstrap y linea base Dummy de la SVM, asi que la
comparacion entre ambos modelos es directa.

```bash
pip install -r requirements-cnn.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

Elegir la rueda CUDA segun el driver del servidor (`nvidia-smi`) y fijar
despues la version exacta instalada en `requirements-cnn.txt`.

### Archivos

```text
modeling/
├── configs/cnn.toml         Arquitectura, entrenamiento, busqueda, augmentation, condiciones
├── models/cnn.py            Red, normalizacion, SpecAugment, entrenamiento, fold, checkpoints
├── cnn_experiment.py        Orquestacion de --model cnn (runs, resumen, modelo final)
└── data.py                  + cache Log-Mel por dataset/rama (sin dependencia de torch)
```

### Ejecucion

```bash
# 1. Validar sin entrenar: hashes, formas, folds, clases, cache, parametros y GPU
python -u -m modeling.run_experiment --model cnn --dataset all --experiment all \
  --device cuda:0 --data-root /ruta/a/copd_vs_control \
  --runs-root /ruta/a/runs --cache-root /ruta/a/cache/logmel --dry-run

# 2. Prueba corta: fold 1, una configuracion, dos epocas
python -u -m modeling.run_experiment --model cnn --dataset ICBHI --experiment main \
  --device cuda:0 --data-root /ruta/a/copd_vs_control \
  --runs-root /ruta/a/runs --cache-root /ruta/a/cache/logmel --smoke-test

# 3. Ejecucion completa
python -u -m modeling.run_experiment --model cnn --dataset all --experiment all \
  --device cuda:0 --num-workers 2 --data-root /ruta/a/copd_vs_control \
  --runs-root /ruta/a/runs --cache-root /ruta/a/cache/logmel
```

- `--device auto|cpu|cuda|cuda:N`: `cuda:N` falla con un error explicito si
  esa GPU no existe; nunca cae a CPU en silencio.
- `--num-workers` (por defecto 2): workers del DataLoader. El resultado no
  depende de este valor (el orden y SpecAugment usan generadores propios).
- Con `--dry-run` la cache Log-Mel ausente o desactualizada no es un error:
  se informa y se genera al ejecutar. La primera extraccion recorre los 6536
  segmentos por rama.
- `--resume` rechaza continuar si cambian los datos, cualquier seccion de
  `cnn.toml` que afecte al entrenamiento, la arquitectura o el modo
  `--smoke-test`. `--device` y `--num-workers` si pueden cambiar entre
  intentos: por ejemplo, para pasar a otra GPU tras un error de memoria.
- Un error de memoria de GPU marca el fold como FAILED y **no** reduce el
  batch (cambiaria el experimento).

### Condiciones

Las cinco de la SVM mas la ablacion de augmentation:

| Experimento | Dataset | Condicion | Hiperparametros |
|---|---|---|---|
| `main` | ICBHI / FRAIWAN_Extended | `main_no_dn` | Busqueda de 4 configuraciones por fold |
| `denoising_ablation` | ICBHI | `no_dn_reliable`, `dn_reliable` | Busqueda propia |
| `denoising_ablation` | FRAIWAN_Extended | `main_no_dn` (reutilizada), `dn` | Busqueda propia |
| `augmentation_ablation` | ICBHI / FRAIWAN_Extended | `main_no_dn` (reutilizada), `main_no_dn_aug` | lr y dropout de `main_no_dn` en ese fold; mejor epoca propia |

### Protocolo por fold

1. Normalizacion por banda Mel calculada solo con train (ponderada con los
   mismos pesos `w_s` de la perdida).
2. Cada configuracion (lr, dropout) se entrena en train con `BCEWithLogitsLoss`
   ponderada, AdamW, clipping 1.0, coseno y AMP en CUDA; se evalua validation
   por paciente cada epoca; parada tras 15 epocas sin mejorar (minimo 20).
3. Se elige configuracion y epoca por balanced accuracy, macro-F1, minimo
   recall entre clases y perdida de validation.
4. Se reinicializa y se entrena con train + validation exactamente esa epoca.
5. Test se evalua una sola vez: probabilidad por segmento -> media por
   grabacion -> media por paciente -> umbral 0.5.

El modelo final usa la configuracion con mejor validation promedio entre folds
y la mediana de sus mejores epocas, entrenado con todos los pacientes elegibles.

### Salidas

```text
<runs-root>/cnn/<run_id>/
├── run.log, status.json, run_fingerprint.json, resolved_config.toml
├── environment.txt              incluye torch, CUDA, cuDNN, GPU y flags de determinismo
├── folds.csv, hyperparameter_search.csv, metrics_by_fold.csv, metrics_summary.csv
├── oof_patient_predictions.csv, baseline_metrics.csv, classification_report.csv, confusion_matrix.csv
├── figures/                     matrices, ROC/PR, metricas por fold, mapa lr x dropout,
│                                comparaciones no_dn/dn y sin/con augmentation
└── datasets/<dataset>/<condicion>/
    ├── fold_01 .. fold_05/
    │   ├── history.csv, search_summary.csv, refit_history.csv
    │   ├── selected_hyperparameters.json, normalization.json, warnings.json
    │   ├── test_{segment,recording,patient}_predictions.csv, test_metrics.json
    │   ├── baseline_metrics.csv
    │   ├── training_curves.{png,pdf,csv}
    │   └── model_fold.pt
    └── final/
        ├── final_model.pt           state_dict + arquitectura + normalizacion + clases + umbral
        ├── model_metadata.json, model_sha256.txt, final_history.csv, normalization.json
```

Los checkpoints se cargan con `torch.load(ruta, weights_only=True)` y
`models.cnn.model_from_checkpoint(...)`. `warnings.json` registra cualquier
operacion sin implementacion determinista detectada durante el fold.

### Pruebas

```bash
pytest modeling/tests
```

`test_logmel_cache.py` no necesita PyTorch. `test_cnn.py` se omite si PyTorch
no esta instalado; con PyTorch corre en CPU y con datos sinteticos.
