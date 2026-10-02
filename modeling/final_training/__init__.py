"""Protocolo holdout_final_v1: reentrenamiento definitivo (100% del 80% de
desarrollo) y evaluacion externa unica (20% de prueba bloqueada), para los
pipelines congelados en ``modeling/configs/final_test/selected_pipelines.toml``.

Independiente de ``oof_v1``, ``fold_aware_v2``, ``holdout_cv_v3`` e
``icbhi_robustness``: no modifica ni reutiliza sus directorios de resultados.
"""
