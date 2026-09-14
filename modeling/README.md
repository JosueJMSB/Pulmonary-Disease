# Modelado

## Preparacion de COPD frente a control

La seleccion clinica es una salida derivada de la fase 4. No modifica los
arreglos originales ni realiza particion, escalado, balanceo o aumento.

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
