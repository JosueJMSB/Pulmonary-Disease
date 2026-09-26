# Configuraciones de modelado

Las configuraciones se agrupan por protocolo experimental para evitar mezclar
archivos con particiones y objetivos de evaluación diferentes.

- `oof_v1/`: protocolo exploratorio original con predicciones out-of-fold.
- `fold_aware_v2/`: protocolo con preprocesamiento recalibrado por fold y sin
  modelo final desplegable.
- `holdout_final/`: reservado para el protocolo definitivo con 80 % de
  desarrollo, validación cruzada interna y 20 % de prueba bloqueado.

Cada carpeta mantiene juntos los TOML de SVM-RBF, CNN y CRNN, tanto para los
datasets individuales como para COMBINED. Los lanzadores resuelven estas rutas
desde `modeling.data`; no se deben mezclar TOML pertenecientes a protocolos
distintos dentro de una misma ejecución.
