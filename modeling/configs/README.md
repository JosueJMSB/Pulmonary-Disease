# Configuraciones de modelado

Las configuraciones se agrupan por protocolo experimental para evitar mezclar
archivos con particiones y objetivos de evaluación diferentes.

| Carpeta | Protocolo | Qué evalúa |
|---|---|---|
| `oof_v1/` | **Histórico.** Predicciones out-of-fold con un 20 % fijo de pacientes de calibración del preprocesamiento. | Ningún modelo definitivo; experimento exploratorio original. |
| `fold_aware_v2/` | **Evaluación OOF anterior.** Preprocesamiento (denoising y `TARGET_RMS`) recalibrado por fold; cada fold hace de prueba una vez; sin modelo final. | El procedimiento de entrenamiento y optimización de cada modelo. |
| `holdout_final/` | **Protocolo definitivo con prueba externa reservada** (`holdout_cv_v3`). | Ver abajo. |

Cada carpeta mantiene juntos los TOML de SVM-RBF, CNN y CRNN, tanto para los
datasets individuales (ICBHI y FRAIWAN_Extended) como para COMBINED. Los
lanzadores resuelven estas rutas desde `modeling.data`; no se deben mezclar TOML
pertenecientes a protocolos distintos dentro de una misma ejecución. Los
protocolos `oof_v1` y `fold_aware_v2` siguen siendo ejecutables tal cual: el
protocolo nuevo es aditivo y solo se activa con `protocol = "holdout_cv_v3"`.

## `holdout_final/` (holdout + validación cruzada v3)

1. Los pacientes de cada dataset se separan **una única vez** en 80 %
   desarrollo y 20 % prueba externa **bloqueada**
   (`modeling/data/holdout_splits.csv`, semilla `20260914`, estratificado por
   clase, independiente en ICBHI y Fraiwan; COMBINED hereda exactamente la
   asignación de cada paciente).
2. Solo el 80 % de desarrollo se reparte en 5 folds internos: cuatro para
   entrenar y uno para validar, rotando, de modo que cada paciente de desarrollo
   se valida una vez. El preprocesamiento (denoising y `TARGET_RMS`) se
   recalibra por fold con los pacientes de train de ese fold.
3. **Cada configuración exacta** se evalúa en los cinco folds y se elige **una
   sola configuración global** por pipeline (modelo × dataset × condición), con
   este orden de criterios: mayor balanced accuracy media, mayor macro F1
   medio, mayor media del menor recall entre clases, menor desviación estándar
   de balanced accuracy y, como desempate final, el orden determinista de la
   configuración. Una configuración incompleta o fallida en algún fold no puede
   seleccionarse.
4. No se usa la prueba externa (`[outer_test] enabled = false`) ni se guarda un
   modelo definitivo (`[final_model] enabled = false`); ambos deben estar
   declarados de forma explícita o el ejecutor se niega a arrancar. La
   selección entre SVM, CNN y CRNN pertenece al benchmarking posterior.

| Modelo | Condiciones | Búsqueda |
|---|---|---|
| SVM-RBF (`svm_rbf_v3.toml`, `svm_rbf_combined_v3.toml`) | `no_dn`, `dn` | 6 × 5 = 30 combinaciones `(C, gamma)`, todas en los 5 folds |
| CNN (`cnn_v3.toml`, `cnn_combined_v3.toml`) | `no_dn`, `dn`, `no_dn_aug` | 20 de las 108 combinaciones de learning rate × dropout × weight decay × batch size (muestreo determinista sin reemplazo, semilla `20260914`) |
| CRNN (`crnn_v3.toml`, `crnn_combined_v3.toml`) | `no_dn`, `dn`, `no_dn_aug` | La misma lista de 20 configuraciones y el mismo presupuesto que la CNN |

- CNN y CRNN usan AdamW, scheduler coseno, recorte de gradiente y AMP (en CUDA)
  **fijos**: ni la arquitectura ni el optimizador se buscan. Entrenan hasta 150
  épocas (mínimo 20, paciencia 15); umbral de decisión 0.5.
- `no_dn_aug` **no repite la búsqueda**: reutiliza la configuración global de
  `no_dn` (mismo modelo y dataset), vuelve a correr los cinco folds con
  SpecAugment solo sobre los batches de entrenamiento y guarda sus propios
  resultados y mejores épocas.
- Por pipeline se registra la **mediana de las mejores épocas** de la
  configuración ganadora (`best_hyperparameters.json`), que se usará para
  reentrenar el modelo definitivo en una etapa posterior.
- Una CRNN se niega a arrancar si alguna sección compartida difiere de su CNN
  de referencia (`[model] reference_cnn_config`).
- **Limitación a declarar al comparar modelos:** en las redes, la mejor época de
  cada unidad se elige con el mismo fold de validación sobre el que se reporta,
  de modo que su métrica de validación es ligeramente optimista frente a la de
  la SVM (que no tiene épocas).

### Artefactos por pipeline

En `<runs-root>/<modelo>/<run_id>/datasets/<dataset>/<condición>/`:
`cv_search_results.csv` (todas las configuraciones × folds),
`cv_config_summary.csv` (media y desviación por configuración),
`best_hyperparameters.json`, `cv_fold_metrics.csv`, `cv_metrics_summary.csv`
(media/desviación entre folds y métricas agrupadas),
`cv_patient_predictions.csv`, `cv_recording_predictions.csv`,
`cv_segment_predictions.csv`, `cv_confusion_matrix.csv`,
`cv_classification_report.csv`, `cv_roc_curve.csv`, `cv_pr_curve.csv`,
`cv_metrics_by_device.csv`, `cv_counts_by_device.csv`,
`cv_confusion_matrix_by_device.csv` (análisis descriptivo, nunca usado para
seleccionar; un dispositivo con una sola clase deja AUROC/AUPRC en `NaN`),
`resolved_config.toml`, `run_manifest.json` y `status.json`. En COMBINED, además,
`cv_metrics_by_source.csv`, `cv_classification_report_by_source.csv` y
`cv_confusion_matrix_by_source.csv`. Además, `figures/` con la matriz de
confusión, las curvas ROC y precisión-recall (pacientes agrupados) y las
métricas por fold, en los formatos y dpi de `[figures]`. En la raíz de la
ejecución, `cv_run_summary.csv` consolida un renglón por pipeline con todas las
métricas (accuracy, balanced accuracy, recall COPD, recall de la clase negativa,
mínimo recall entre clases, macro F1, AUROC y AUPRC COPD), cada una con su media
y desviación estándar entre folds y su valor agrupado. Las unidades intermedias
(`cv/config_NNN/fold_NN/`, con `history.csv` autosuficiente en las redes)
permiten reanudar sin repetir lo ya completado.

### Reanudación segura

`--resume` reutiliza unidades solo si coinciden la configuración, los datos, el
split, la arquitectura y el modo, **y la versión del código**: la huella guarda
un hash reproducible de `modeling/**/*.py` (sin pruebas ni lanzadores
secuenciales, y con saltos de línea normalizados). Si el código cambió después
de una ejecución incompleta, la reanudación se rechaza y nombra los archivos
modificados; hay que empezar una ejecución nueva. El hash también queda en
`run_manifest.json` y `best_hyperparameters.json`.

### Orden de ejecución

`build_master_folds --protocol holdout-v3` → `preprocessing/fold_denoising.py
--protocol holdout-v3 --stage cv` (15 folds) → `run_holdout_sequence
--dry-run-only` → `run_holdout_sequence --execute`. Ver `modeling/README.md`.
