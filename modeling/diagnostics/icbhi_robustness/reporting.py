"""Artefactos por repeticion y resumen pareado entre semillas."""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ... import artifacts as art
from ... import evaluation as ev
from ... import holdout_cv as hcv

from .common import HYPERPARAMETER_KEYS, Pipeline, write_json


def write_pipeline_outputs(
    *,
    pipeline_root: Path,
    spec,
    pipeline: Pipeline,
    cfg: dict,
    fold_ids: list[int],
    seed: int,
    source_provenance: dict,
    logger,
    control: bool = False,
) -> dict:
    records = hcv.load_unit_records(
        pipeline_root, spec, [pipeline.candidate()], fold_ids
    )
    search = hcv.search_results_frame(
        records, spec, pipeline.architecture, "Healthy", HYPERPARAMETER_KEYS
    )
    condition_dir = art.condition_dir(
        pipeline_root, spec.dataset, spec.condition
    )
    condition_dir.mkdir(parents=True, exist_ok=True)
    search.to_csv(
        condition_dir / "cv_fixed_configuration_results.csv",
        index=False,
        lineterminator="\n",
    )
    completed = [record for record in records if record["status"] == hcv.STATUS_UNIT_COMPLETED]
    if len(completed) != len(fold_ids):
        art.write_status(
            pipeline_root,
            art.STATUS_FAILED,
            {
                "protocol": "icbhi_robustness_v1",
                "seed": seed,
                "pipeline": pipeline.id,
                "completed_folds": len(completed),
                "expected_folds": len(fold_ids),
            },
        )
        raise RuntimeError(
            f"{pipeline.id}/{seed}: solo {len(completed)}/{len(fold_ids)} folds completos"
        )

    fold_metrics = hcv.fold_metrics_frame(completed, HYPERPARAMETER_KEYS)
    segments = hcv.winner_segment_predictions(
        pipeline_root, spec, pipeline.source_config_index, fold_ids
    )
    recordings, patients = hcv.aggregate_winner_predictions(segments)
    y_true = patients["target_label"].to_numpy(dtype=np.int64)
    y_score = patients["score"].to_numpy(dtype=np.float64)
    threshold = float(cfg["evaluation"]["decision_threshold"])
    pooled = ev.compute_patient_metrics(y_true, y_score, "Healthy", threshold)
    pooled["recall_negative"] = pooled["recall_healthy"]
    pooled["min_class_recall"] = float(
        min(pooled["recall_copd"], pooled["recall_negative"])
    )
    metrics_summary = hcv.metrics_summary_frame(fold_metrics, pooled)
    roc, pr = hcv.curve_frames(y_true, y_score)

    fold_metrics.to_csv(
        condition_dir / "cv_fold_metrics.csv", index=False, lineterminator="\n"
    )
    metrics_summary.to_csv(
        condition_dir / "cv_metrics_summary.csv", index=False, lineterminator="\n"
    )
    segments.to_csv(
        condition_dir / "cv_segment_predictions.csv", index=False, lineterminator="\n"
    )
    recordings.to_csv(
        condition_dir / "cv_recording_predictions.csv", index=False, lineterminator="\n"
    )
    patients.to_csv(
        condition_dir / "cv_patient_predictions.csv", index=False, lineterminator="\n"
    )
    ev.confusion_matrix_df(
        y_true, y_score, ("Healthy", "COPD"), threshold
    ).reset_index().rename(columns={"index": "real"}).to_csv(
        condition_dir / "cv_confusion_matrix.csv", index=False, lineterminator="\n"
    )
    ev.classification_report_df(
        y_true, y_score, ("Healthy", "COPD"), threshold
    ).to_csv(
        condition_dir / "cv_classification_report.csv",
        index=False,
        lineterminator="\n",
    )
    roc.to_csv(condition_dir / "cv_roc_curve.csv", index=False, lineterminator="\n")
    pr.to_csv(condition_dir / "cv_pr_curve.csv", index=False, lineterminator="\n")
    hcv.write_group_artifacts(condition_dir, segments, patients, "Healthy", threshold)
    hcv.write_figures(
        condition_dir,
        cfg,
        y_true,
        y_score,
        ("Healthy", "COPD"),
        threshold,
        fold_metrics,
        "Healthy",
        logger,
        f"{pipeline.id}/{seed}",
    )

    best_epochs = {
        str(int(record["fold"])): record.get("best_epoch")
        for record in sorted(completed, key=lambda row: row["fold"])
    }
    manifest = {
        "protocol": "icbhi_robustness_v1",
        "diagnostic_only": True,
        "negative_control": bool(control),
        "outer_test": {"enabled": False, "accessed": False},
        "final_model": {"enabled": False},
        "seed": int(seed),
        "pipeline": pipeline.id,
        "architecture": pipeline.architecture,
        "condition": spec.condition,
        "branch": pipeline.branch,
        "augment": pipeline.augment,
        "frozen_hyperparameters": pipeline.candidate(),
        "source_selection": source_provenance,
        "best_epochs_by_fold": best_epochs,
        "pooled_metrics": pooled,
        "written_at_utc": art._now_iso(),
    }
    write_json(condition_dir / "robustness_manifest.json", manifest)
    art.write_status(
        pipeline_root,
        art.STATUS_COMPLETED,
        {
            "protocol": "icbhi_robustness_v1",
            "seed": seed,
            "pipeline": pipeline.id,
            "outer_test_accessed": False,
        },
    )
    return {
        "seed": int(seed),
        "pipeline": pipeline.id,
        "architecture": pipeline.architecture,
        "condition": spec.condition,
        "negative_control": bool(control),
        **{f"pooled_{metric}": pooled[metric] for metric in hcv.SUMMARY_METRICS},
        **{
            f"fold_mean_{metric}": float(fold_metrics[metric].mean())
            for metric in hcv.SUMMARY_METRICS
        },
        **{
            f"fold_std_{metric}": float(fold_metrics[metric].std(ddof=1))
            for metric in hcv.SUMMARY_METRICS
        },
    }


def robustness_summary(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [f"pooled_{metric}" for metric in hcv.SUMMARY_METRICS]
    output = []
    for pipeline, group in rows.loc[~rows["negative_control"]].groupby(
        "pipeline", sort=False
    ):
        row = {
            "pipeline": pipeline,
            "architecture": group["architecture"].iloc[0],
            "condition": group["condition"].iloc[0],
            "n_repetitions": int(len(group)),
            "n_perfect_balanced_accuracy": int(
                np.isclose(group["pooled_balanced_accuracy"], 1.0).sum()
            ),
        }
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())
        output.append(row)
    return pd.DataFrame(output)


def paired_comparisons(rows: pd.DataFrame) -> pd.DataFrame:
    clean = rows.loc[~rows["negative_control"]]
    pipelines = list(dict.fromkeys(clean["pipeline"]))
    metrics = (
        "pooled_balanced_accuracy",
        "pooled_macro_f1",
        "pooled_min_class_recall",
        "pooled_auroc",
    )
    output = []
    for left, right in itertools.combinations(pipelines, 2):
        a = clean.loc[clean["pipeline"] == left].set_index("seed")
        b = clean.loc[clean["pipeline"] == right].set_index("seed")
        common = sorted(set(a.index) & set(b.index))
        for metric in metrics:
            differences = a.loc[common, metric] - b.loc[common, metric]
            tied = np.isclose(differences, 0.0)
            output.append(
                {
                    "pipeline_a": left,
                    "pipeline_b": right,
                    "metric": metric,
                    "n_paired_repetitions": len(common),
                    "mean_difference_a_minus_b": float(differences.mean()),
                    "std_difference": float(differences.std(ddof=1)),
                    "min_difference": float(differences.min()),
                    "max_difference": float(differences.max()),
                    "wins_a": int(((differences > 0) & ~tied).sum()),
                    "ties": int(tied.sum()),
                    "wins_b": int(((differences < 0) & ~tied).sum()),
                }
            )
    return pd.DataFrame(output)

