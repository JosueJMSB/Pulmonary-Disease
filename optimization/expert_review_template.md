# Acta de revisión del benchmarking (Objetivo 3)

Medio de verificación del resultado 2: revisión documentada por un experto en Inteligencia Artificial.

| Campo | Valor |
|---|---|
| Revisor | |
| Formación o cargo | |
| Fecha | |
| Corrida de optimization revisada | |
| Hash de `report/benchmark_table.csv` | |
| Hash de `report/iov_status.csv` | |

## Qué se revisó

- [ ] Paridad PyTorch → ONNX FP32 (`export_parity.json` de cada dataset).
- [ ] Parámetros de cuantización e inventario de calibración (solo pacientes de desarrollo).
- [ ] Operadores que quedaron en INT8 y en FP32 (`quantization_ops.json`).
- [ ] Criterio de aceptación en desarrollo y congelación (`dev_comparison.json`, `freeze.json`).
- [ ] Comprobación del flujo WAV → decisión (`pipeline_check.json`).
- [ ] Evaluación única del INT8 en test y comparación con el `.pt` publicado (`test_int8/`).
- [ ] Mediciones de tamaño, memoria y tiempo, y sus condiciones (`benchmark/<plataforma>/`).
- [ ] Prueba sin internet en la laptop (`offline_comparison.json`).
- [ ] Redacción de las conclusiones: no afirman compatibilidad con un celular de bajo costo.

## Observaciones

## Conclusión del revisor

## Firma
