"""SVM-RBF: busqueda manual de (C, gamma) por fold y ajuste del modelo final.

La busqueda es manual -no ``GridSearchCV``- porque la seleccion ocurre
DESPUES de agregar segmento -> grabacion -> paciente, algo que scikit-learn
no expresa de forma nativa: cada combinacion (C, gamma) se entrena sobre
segmentos, pero se evalua sobre pacientes agregados.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .. import data as dmod
from .. import evaluation as ev
from .. import splits as sp


def svm_fixed_params(cfg: dict) -> dict:
    """Los parametros de la SVM que el plan fija y NO se buscan.

    ``probability`` no se pasa: el valor que exige el plan (False) ya es el
    default de ``SVC``, y pasarlo explicitamente produce una advertencia de
    obsolescencia en scikit-learn 1.9. Ademas la clase nunca usa
    ``predict_proba``: solo ``decision_function``, que no depende de este
    parametro.
    """
    svm_cfg = cfg["svm"]
    raw_class_weight = svm_cfg.get("class_weight", "none")
    class_weight = None if raw_class_weight in (None, "none", "None") else raw_class_weight
    return dict(
        kernel=svm_cfg.get("kernel", "rbf"),
        shrinking=bool(svm_cfg.get("shrinking", True)),
        tol=float(svm_cfg.get("tol", 1e-3)),
        cache_size=float(svm_cfg.get("cache_size_mb", 1024)),
        class_weight=class_weight,
    )


def hyperparameter_grid(cfg: dict) -> list[tuple[float, object]]:
    """Las 16 combinaciones (C, gamma) del espacio de busqueda, en orden
    C-mayor luego gamma, para que la tabla de resultados sea legible."""
    svm_cfg = cfg["svm"]
    c_grid = list(svm_cfg["c_grid"])
    gamma_grid = list(svm_cfg["gamma_grid"])
    return [(float(c), gamma) for c in c_grid for gamma in gamma_grid]


def fit_scaler_and_svm(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    C: float,
    gamma,
    fixed_params: dict,
) -> tuple[StandardScaler, SVC]:
    """Ajusta ``StandardScaler`` y ``SVC`` con los mismos pesos por muestra."""
    scaler = StandardScaler()
    scaler.fit(X, sample_weight=sample_weight)
    X_scaled = scaler.transform(X)
    svm = SVC(C=C, gamma=gamma, **fixed_params)
    svm.fit(X_scaled, y, sample_weight=sample_weight)
    return scaler, svm


def decision_scores(scaler: StandardScaler, svm: SVC, X: np.ndarray) -> np.ndarray:
    return svm.decision_function(scaler.transform(X))


# ---------------------------------------------------------------------------
# Seleccion de hiperparametros: balanced accuracy > macro-F1 > min(recall) >
# menor C > orden de gamma (scale, 0.001, 0.01, 0.1).
# ---------------------------------------------------------------------------

def select_hyperparameters(grid_table: pd.DataFrame, criteria: list[str], gamma_order: list) -> pd.Series:
    """Aplica, en orden, los criterios de desempate del plan sobre una tabla
    con columnas ``C``, ``gamma`` y las metricas de ``criteria``."""
    table = grid_table.copy()
    table["neg_C"] = -table["C"].astype(float)
    rank = {g: i for i, g in enumerate(gamma_order)}
    table["gamma_order"] = table["gamma"].map(lambda g: rank.get(g, len(gamma_order)))

    sort_columns: list[str] = []
    ascending: list[bool] = []
    for name in criteria:
        if name in ("balanced_accuracy", "macro_f1", "min_class_recall"):
            sort_columns.append(name)
            ascending.append(False)  # mayor es mejor
        elif name == "neg_C":
            sort_columns.append("neg_C")
            ascending.append(False)  # mayor neg_C == menor C
        elif name == "gamma_order":
            sort_columns.append("gamma_order")
            ascending.append(True)  # menor indice == antes en gamma_grid
        else:
            raise ValueError(f"criterio de seleccion desconocido: {name}")

    ranked = table.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    return ranked.iloc[0]


# ---------------------------------------------------------------------------
# Un fold completo: 16 combinaciones en validacion, reajuste, evaluacion.
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    fold: int
    selected_C: float
    selected_gamma: object
    grid_table: pd.DataFrame
    segment_predictions: pd.DataFrame
    recording_predictions: pd.DataFrame
    patient_predictions: pd.DataFrame
    test_metrics: dict
    n_support: dict = field(default_factory=dict)
    scaler: StandardScaler | None = None
    svm: SVC | None = None
    baseline_table: pd.DataFrame = field(default_factory=pd.DataFrame)


# Comprobacion de cordura: si la SVM no supera esto con holgura, el
# resultado no es defendible sin importar cuan bien se vea aislado.
#   most_frequent -> siempre predice la clase mayoritaria de train+val.
#   stratified    -> respeta la prevalencia de train+val, al azar.
DUMMY_STRATEGIES = ("most_frequent", "stratified")
DUMMY_SCORE_THRESHOLD = 0.5  # sobre predict_proba, no sobre decision_function


def compute_dummy_baseline(
    train_seg: pd.DataFrame,
    test_seg: pd.DataFrame,
    negative_label_name: str,
    fold_id: int,
    random_state: int = 20260914,
) -> pd.DataFrame:
    """``DummyClassifier`` como linea base de cordura, evaluado exactamente
    igual que la SVM: mismo train (aqui, train+validation del fold), mismo
    test, misma agregacion segmento -> grabacion -> paciente y las mismas
    metricas por paciente. La unica diferencia es el umbral: se usa 0.5 sobre
    ``predict_proba`` porque un modelo dummy no tiene un margen que umbralizar
    en 0 como la SVM.
    """
    train_y = train_seg["target_label"].to_numpy()
    dummy_X_train = np.zeros((len(train_seg), 1))
    dummy_X_test = np.zeros((len(test_seg), 1))

    rows = []
    for strategy in DUMMY_STRATEGIES:
        clf = DummyClassifier(strategy=strategy, random_state=random_state)
        clf.fit(dummy_X_train, train_y)
        proba = clf.predict_proba(dummy_X_test)
        if ev.POSITIVE_LABEL in clf.classes_:
            positive_idx = list(clf.classes_).index(ev.POSITIVE_LABEL)
            scores = proba[:, positive_idx]
        else:
            scores = np.zeros(len(test_seg))  # train+val no tenia ningun COPD (no deberia pasar)

        test_seg_scores = test_seg.assign(score=scores)
        patient_scores = ev.aggregate_segment_to_patient(test_seg_scores)
        metrics = ev.compute_patient_metrics(
            patient_scores["target_label"].to_numpy(), patient_scores["score"].to_numpy(),
            negative_label_name=negative_label_name, threshold=DUMMY_SCORE_THRESHOLD,
        )
        rows.append({"fold": fold_id, "strategy": strategy, **metrics})
    return pd.DataFrame(rows)


def _evaluate_combo(
    train_X: np.ndarray,
    train_y: np.ndarray,
    train_weights: np.ndarray,
    val_X: np.ndarray,
    val_seg: pd.DataFrame,
    C: float,
    gamma,
    fixed_params: dict,
    threshold: float,
    fold_id: int,
) -> dict:
    """Una combinacion (C, gamma): ajuste en train, agregacion por paciente
    en validation. Funcion de nivel de modulo para que ``joblib`` pueda
    ejecutarla en procesos separados con ``--n-jobs``."""
    scaler, svm = fit_scaler_and_svm(train_X, train_y, train_weights, C, gamma, fixed_params)
    val_scores = decision_scores(scaler, svm, val_X)
    val_seg_scores = val_seg.assign(score=val_scores)
    val_patient_scores = ev.aggregate_segment_to_patient(val_seg_scores)
    y_pred = ev.scores_to_predictions(val_patient_scores["score"].to_numpy(), threshold)
    selection = ev.selection_scores(val_patient_scores["target_label"].to_numpy(), y_pred)
    return {
        "fold": fold_id, "C": C, "gamma": gamma,
        "n_val_patients": int(len(val_patient_scores)),
        **selection,
    }


def run_fold(
    fold_id: int,
    segments: pd.DataFrame,
    X: np.ndarray,
    train_patients: list[str],
    val_patients: list[str],
    test_patients: list[str],
    cfg: dict,
    negative_label_name: str,
    n_jobs: int = 1,
) -> FoldResult:
    """Un fold completo, tal como lo describe el plan:

    1. Ajustar scaler con train.
    2. Entrenar las 16 combinaciones.
    3. Predecir margenes sobre validation.
    4. Agregar segmento -> grabacion -> paciente.
    5. Seleccionar hiperparametros.
    6. Reajustar scaler y SVM con train + validation.
    7. Evaluar una sola vez sobre test.
    """
    train_seg, train_X = dmod.select_rows_by_patients(segments, X, train_patients)
    val_seg, val_X = dmod.select_rows_by_patients(segments, X, val_patients)
    test_seg, test_X = dmod.select_rows_by_patients(segments, X, test_patients)

    for name, seg in (("train", train_seg), ("validation", val_seg), ("test", test_seg)):
        if seg.empty:
            raise ValueError(f"fold {fold_id}: el conjunto {name} quedo vacio")

    fixed_params = svm_fixed_params(cfg)
    grid = hyperparameter_grid(cfg)
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))
    criteria = list(cfg["selection"]["criteria"])
    gamma_order = list(cfg["svm"]["gamma_grid"])

    train_weights = dmod.compute_sample_weights(train_seg)
    train_y = train_seg["target_label"].to_numpy()

    # n_jobs<=1 corre en el mismo proceso (sin overhead de arranque); valores
    # mayores reparten las 16 combinaciones entre procesos, cada uno con su
    # propia copia de train/validation (independientes entre si).
    grid_rows = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_evaluate_combo)(
            train_X, train_y, train_weights, val_X, val_seg, C, gamma, fixed_params, threshold, fold_id,
        )
        for C, gamma in grid
    )
    grid_table = pd.DataFrame(grid_rows)

    selected = select_hyperparameters(grid_table, criteria, gamma_order)
    selected_C = float(selected["C"])
    selected_gamma = selected["gamma"]

    refit_seg = pd.concat([train_seg, val_seg], ignore_index=True)
    refit_X = np.concatenate([train_X, val_X], axis=0)
    refit_weights = dmod.compute_sample_weights(refit_seg)
    scaler, svm = fit_scaler_and_svm(
        refit_X, refit_seg["target_label"].to_numpy(), refit_weights,
        selected_C, selected_gamma, fixed_params,
    )

    test_scores = decision_scores(scaler, svm, test_X)
    test_seg_scores = test_seg.assign(score=test_scores)
    test_recording_scores = ev.aggregate_segment_to_recording(test_seg_scores)
    test_patient_scores = ev.aggregate_recording_to_patient(test_recording_scores)

    test_metrics = ev.compute_patient_metrics(
        test_patient_scores["target_label"].to_numpy(),
        test_patient_scores["score"].to_numpy(),
        negative_label_name=negative_label_name,
        threshold=threshold,
    )
    test_metrics.update({"fold": fold_id, "C": selected_C, "gamma": selected_gamma})

    # Linea base de cordura: entrenada sobre el mismo train+validation que el
    # reajuste de la SVM, evaluada sobre el mismo test.
    baseline_table = compute_dummy_baseline(refit_seg, test_seg, negative_label_name, fold_id)

    n_support = {}
    if hasattr(svm, "classes_") and hasattr(svm, "n_support_"):
        n_support = {int(c): int(n) for c, n in zip(svm.classes_, svm.n_support_)}

    return FoldResult(
        fold=fold_id,
        selected_C=selected_C,
        selected_gamma=selected_gamma,
        grid_table=grid_table,
        segment_predictions=test_seg_scores,
        recording_predictions=test_recording_scores,
        patient_predictions=test_patient_scores,
        test_metrics=test_metrics,
        n_support=n_support,
        scaler=scaler,
        svm=svm,
        baseline_table=baseline_table,
    )


# ---------------------------------------------------------------------------
# Seleccion global de (C, gamma) para el modelo final, solo con validacion.
# ---------------------------------------------------------------------------

def select_final_hyperparameters(
    fold_grid_tables: list[pd.DataFrame], cfg: dict
) -> tuple[float, object, pd.DataFrame]:
    """Promedia las metricas de validacion de cada combinacion a traves de
    los folds y aplica los mismos criterios de desempate.

    Usa unicamente resultados de validacion (nunca de test), como exige el
    plan para elegir el (C, gamma) del modelo final.
    """
    combined = pd.concat(fold_grid_tables, ignore_index=True)
    aggregated = combined.groupby(["C", "gamma"], as_index=False).agg(
        balanced_accuracy=("balanced_accuracy", "mean"),
        macro_f1=("macro_f1", "mean"),
        min_class_recall=("min_class_recall", "mean"),
        n_val_patients=("n_val_patients", "sum"),
    )
    criteria = list(cfg["selection"]["criteria"])
    gamma_order = list(cfg["svm"]["gamma_grid"])
    selected = select_hyperparameters(aggregated, criteria, gamma_order)
    return float(selected["C"]), selected["gamma"], aggregated


def fit_final_model(
    segments: pd.DataFrame, X: np.ndarray, C: float, gamma, cfg: dict
) -> tuple[StandardScaler, SVC, np.ndarray]:
    """Reajuste final sobre TODOS los pacientes disponibles de la condicion.

    No produce metricas de prueba nuevas: (C, gamma) deben venir ya elegidos
    por ``select_final_hyperparameters`` a partir de las validaciones de los
    folds *out-of-fold*.
    """
    weights = dmod.compute_sample_weights(segments)
    fixed_params = svm_fixed_params(cfg)
    scaler, svm = fit_scaler_and_svm(
        X, segments["target_label"].to_numpy(), weights, C, gamma, fixed_params
    )
    return scaler, svm, weights


# ---------------------------------------------------------------------------
# Curva de aprendizaje calculada SIN el conjunto de prueba del fold.
# ---------------------------------------------------------------------------

DEFAULT_LEARNING_CURVE_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def compute_learning_curve(
    pool_segments: pd.DataFrame,
    pool_X: np.ndarray,
    C: float,
    gamma,
    cfg: dict,
    fractions: tuple[float, ...] = DEFAULT_LEARNING_CURVE_FRACTIONS,
    n_internal_splits: int = 3,
    random_state: int = 20260914,
) -> pd.DataFrame:
    """Curva de aprendizaje sin tocar el test del fold.

    ``pool_segments``/``pool_X`` deben ser el train+validation de UN fold (el
    mismo conjunto sobre el que ese fold reajusta antes de evaluar test): una
    validacion cruzada interna de ``n_internal_splits`` particiona esos
    pacientes, y para cada fraccion creciente del train interno se ajusta el
    modelo con el (C, gamma) ya seleccionado por ese fold y se mide balanced
    accuracy / macro-F1 por paciente sobre el resto del pool. El test real
    del fold no aparece en ningun punto de este calculo.
    """
    from sklearn.model_selection import StratifiedKFold

    patients = sp.build_patient_table(pool_segments)
    if patients["target_label"].nunique() < 2:
        raise ValueError("el pool de la curva de aprendizaje tiene una sola clase")

    fixed_params = svm_fixed_params(cfg)
    threshold = float(cfg["svm"].get("decision_threshold", 0.0))
    patient_ids = patients["patient_uid"].to_numpy()
    labels = patients["target_label"].to_numpy()

    n_internal_splits = min(n_internal_splits, int(np.min(np.bincount(labels))))
    if n_internal_splits < 2:
        raise ValueError("no hay suficientes pacientes por clase para la validacion interna")

    skf = StratifiedKFold(n_splits=n_internal_splits, shuffle=True, random_state=random_state)
    rows = []

    for internal_fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, labels)):
        internal_train_ids = patient_ids[train_idx]
        internal_test_ids = patient_ids[test_idx]
        rng = np.random.RandomState(random_state + internal_fold)
        shuffled = internal_train_ids.copy()
        rng.shuffle(shuffled)

        eval_seg, eval_X = dmod.select_rows_by_patients(pool_segments, pool_X, internal_test_ids.tolist())
        if eval_seg["target_label"].nunique() < 2:
            continue

        for fraction in fractions:
            n_take = max(2, int(round(len(shuffled) * fraction)))
            subset_ids = shuffled[:n_take]
            sub_seg, sub_X = dmod.select_rows_by_patients(pool_segments, pool_X, subset_ids.tolist())
            if sub_seg["target_label"].nunique() < 2:
                continue

            weights = dmod.compute_sample_weights(sub_seg)
            scaler, svm = fit_scaler_and_svm(sub_X, sub_seg["target_label"].to_numpy(), weights, C, gamma, fixed_params)

            scores = decision_scores(scaler, svm, eval_X)
            eval_seg_scores = eval_seg.assign(score=scores)
            eval_patient_scores = ev.aggregate_segment_to_patient(eval_seg_scores)
            metrics = ev.selection_scores(
                eval_patient_scores["target_label"].to_numpy(),
                ev.scores_to_predictions(eval_patient_scores["score"].to_numpy(), threshold),
            )
            rows.append({
                "internal_fold": internal_fold,
                "fraction": fraction,
                "n_train_patients": int(len(subset_ids)),
                "n_eval_patients": int(len(eval_patient_scores)),
                **metrics,
            })

    curve = pd.DataFrame(rows)
    if curve.empty:
        return curve
    return curve.groupby("fraction", as_index=False).agg(
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        n_train_patients_mean=("n_train_patients", "mean"),
    )
