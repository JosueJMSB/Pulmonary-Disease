# Revisión por dispositivo — ICBHI / CNN `no_dn`

## Evidencia

`icbhi_no_dn_cv_metrics_by_device.csv` (sha256 `1454de4c782da3c4f3497d72fe5291c786a2eb73f4e6c38f022d2268bd52c118`)
es una copia inmutable, byte a byte, de:

```
modeling/runs/cnn/20260926T193325Z_141939/datasets/ICBHI/no_dn/cv_metrics_by_device.csv
```

Generado por el protocolo `holdout_cv_v3` (validación cruzada de 5 folds internos
sobre el 80 % de desarrollo de ICBHI). **No utiliza ningún paciente de la
prueba externa bloqueada**: `holdout_cv_v3` nunca accede a esos 18 pacientes
(`outer_test.enabled = false` en todo ese protocolo).

## Hallazgo registrado

De los cuatro dispositivos de ICBHI, **Meditron es el único con ambas clases**:

| device | n_patient_devices | n_copd | n_healthy | both_classes | recall_copd | recall_healthy | balanced_accuracy |
|---|---:|---:|---:|---|---:|---:|---:|
| AKGC417L | 24 | 24 | 0 | False | 1.0 | — | — |
| Litt3200 | 9 | 9 | 0 | False | 1.0 | — | — |
| LittC2SE | 15 | 15 | 0 | False | 1.0 | — | — |
| **Meditron** | **26** | **5** | **21** | **True** | **1.0** | **1.0** | **1.0** |

Meditron reúne 26 pacientes (5 COPD, 21 Healthy) y, dentro de ese subgrupo
-el único donde `balanced_accuracy` tiene sentido, porque los otros tres solo
tienen COPD-, CNN `no_dn` alcanza `balanced_accuracy = 1.0` (`recall_copd =
1.0`, `recall_healthy = 1.0`, matriz de confusión 21/0/0/5 sin errores).

## Decisión

Con el subgrupo de ambas clases evaluado y sin discrepancia frente al
resultado agrupado, se registra esta revisión como satisfecha y el pipeline
ICBHI `no_dn` pasa de `pending_device_review` a `approved` en
`selected_pipelines.toml` (`device_review_path`/`device_review_sha256`
apuntan al CSV de arriba).
