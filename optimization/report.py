"""Fase 7 (phantom): tabla de benchmarking y estado de los indicadores.

    python -m optimization.report --run-dir <corrida> --label phantom \
        [--offline-comparison <salida laptop>/offline_comparison.json] \
        [--expert-review <acta firmada>]

La tabla separa dos bloques con su origen explicito:

- Clasificacion en test (por paciente): "base .pt (publicado)" frente a
  "ONNX INT8 (evaluado una vez)". ONNX FP32 no se evaluo en test y la tabla
  lo dice asi; no se le atribuyen metricas de test.
- Recursos: ONNX FP32 frente a ONNX INT8, en la misma CPU y condiciones.

Cada indicador queda como CUMPLE, NO CUMPLE o NO VERIFICADO (si falta la
medicion). Las metas mas estrictas se reportan como hipotesis comprobadas o
no; nunca como promesas.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import bundle as bmod
from . import paths
from .benchmark import relative_change_pct
from .scoring import SUMMARY_METRICS

FP32_TEST_NOTE = "no evaluado en test; paridad con .pt verificada en desarrollo"
MB = 1024 * 1024


def _status(ok) -> str:
    if ok is None:
        return "NO VERIFICADO"
    return "CUMPLE" if ok else "NO CUMPLE"


def performance_status(name: str, test: dict | None, iov: dict, goals: dict) -> list[dict]:
    """IOV de caida de desempeno: una fila por metrica comprometida, mas una
    fila global que solo CUMPLE si todas cumplen. La meta (mas estricta) se
    aplica solo a balanced accuracy y asi se rotula."""
    metrics = list(iov["performance_metrics"])
    limit = float(iov["max_performance_drop_points"])
    rows, oks = [], []
    for metric in metrics:
        if test:
            drop = 100.0 * (float(test["metrics_base_pt_publicado"][metric]) - float(test["metrics_int8"][metric]))
            ok = drop <= limit
            value = f"{drop:+.2f} puntos"
        else:
            ok, value = None, "sin evaluacion INT8 en test"
        oks.append(ok)
        is_ba = metric == "balanced_accuracy"
        goal_ok = (drop <= float(goals["max_ba_drop_points"]) if test else None) if is_ba else None
        rows.append({"dataset": name, "result": 2,
                     "indicator": f"caida_{metric} (.pt publicado -> INT8, por paciente)",
                     "value": value, "iov": f"<= {limit} puntos", "iov_status": _status(ok),
                     "goal": f"<= {goals['max_ba_drop_points']} puntos" if is_ba else "-",
                     "goal_status": _status(goal_ok) if is_ba else "-"})
    overall = None if any(o is None for o in oks) else all(oks)
    detail = f"{len(metrics)} metricas evaluadas por separado: {', '.join(metrics)}"
    if test:
        detail += f" ({test['small_test_warning']})"
    rows.insert(0, {"dataset": name, "result": 2,
                    "indicator": "caida_desempeno (todas las metricas comprometidas)",
                    "value": detail, "iov": f"cada metrica <= {limit} puntos", "iov_status": _status(overall),
                    "goal": "-", "goal_status": "-"})
    return rows


def offline_status(name: str, bundle: dict, bundle_sha256: str, offline: dict | None) -> dict:
    """Estado de la prueba sin internet PARA ESTE DATASET. Exige que el
    dataset figure en la comparacion y que los hashes de FP32, INT8 y del
    paquete probados en la laptop sean los de esta corrida."""
    row = {"dataset": name, "result": 2, "indicator": "funcionamiento_sin_internet (laptop)",
           "iov": "predicciones identicas con y sin conexion (FP32 e INT8)", "goal": "-", "goal_status": "-"}
    if not offline:
        return {**row, "value": "prueba no ejecutada", "iov_status": _status(None)}
    entry = (offline.get("datasets") or {}).get(name)
    if entry is None:
        return {**row, "value": "este dataset no se incluyo en la prueba", "iov_status": _status(None)}
    expected = {"fp32_sha256": bundle["models"]["fp32"]["sha256"], "int8_sha256": bundle["models"]["int8"]["sha256"],
                "bundle_sha256": bundle_sha256}
    mismatched = [k for k, v in expected.items() if entry.get(k) != v]
    if mismatched:
        return {**row, "value": f"se probaron otros modelos (difiere {', '.join(mismatched)})",
                "iov_status": _status(None)}
    ok = {"PASS": True, "FAIL": False}.get(entry.get("status"))
    return {**row, "value": entry.get("detail", ""), "iov_status": _status(ok)}


def _read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.is_file() else None


def dataset_report(name: str, cfg: dict, run_dir: Path, label: str, offline: dict | None,
                   expert_review: Path | None) -> tuple[list, list]:
    ds_dir = paths.dataset_dir(run_dir, name)
    bundle = bmod.load_bundle(ds_dir, require_int8=True)
    bench = run_dir / "benchmark" / label / name
    lat = _read_csv(bench / "latency_summary.csv")
    e2e = _read_csv(bench / "e2e_summary.csv")
    mem = _read_csv(bench / "memory.csv")
    test_path = ds_dir / "test_int8" / "test_summary_int8.json"
    test = paths.read_json(test_path) if test_path.is_file() else None
    iov, goals = cfg["iov"], cfg["goals"]

    table, status = [], []
    # --- Clasificacion en test (las ocho metricas del resultado base) ---
    for metric in SUMMARY_METRICS:
        row = {"dataset": name, "block": "clasificacion_test_por_paciente", "criterion": metric,
               "base_pt_publicado": None, "onnx_int8": None, "onnx_fp32": FP32_TEST_NOTE}
        if test:
            row["base_pt_publicado"] = test["metrics_base_pt_publicado"].get(metric)
            row["onnx_int8"] = test["metrics_int8"].get(metric)
        table.append(row)

    # --- Recursos ---
    fp32_bytes = int(bundle["models"]["fp32"]["size_bytes"])
    int8_bytes = int(bundle["models"]["int8"]["size_bytes"])
    table.append({"dataset": name, "block": "recursos", "criterion": "tamano_archivo_mb",
                  "onnx_fp32": fp32_bytes / MB, "onnx_int8": int8_bytes / MB,
                  "base_pt_publicado": (bundle["source"].get("checkpoint_size_bytes") or 0) / MB or None})
    peaks = {}
    if mem is not None:
        for mode in ("pipeline", "model", "baseline_pipeline", "baseline_model"):
            sub = mem.loc[mem["mode"] == mode]
            values = {r["version"]: r["peak_rss_bytes"] for _, r in sub.iterrows()}
            peaks[mode] = values
            table.append({"dataset": name, "block": "recursos", "criterion": f"pico_ram_{mode}_mb",
                          "onnx_fp32": values.get("fp32", values.get("-", float("nan"))) / MB,
                          "onnx_int8": values.get("int8", values.get("-", float("nan"))) / MB})
    for summary, unit, crit in ((lat, "ms", "latencia_modelo_segmento"), (e2e, "s", "tiempo_total_grabacion")):
        if summary is None:
            continue
        for threads, group in summary.groupby("threads"):
            by_version = group.set_index("version")
            for stat in ("median", "p95"):
                table.append({"dataset": name, "block": "recursos",
                              "criterion": f"{crit}_{stat}_{unit}_{int(threads)}hilos",
                              "onnx_fp32": by_version.loc["fp32", stat] if "fp32" in by_version.index else None,
                              "onnx_int8": by_version.loc["int8", stat] if "int8" in by_version.index else None})

    # --- Indicadores ---
    size_red = -relative_change_pct(fp32_bytes, int8_bytes)
    status.append({"dataset": name, "result": 1, "indicator": "reduccion_tamano (ONNX FP32 -> INT8)",
                   "value": f"{size_red:.2f} %", "iov": f">= {iov['min_size_reduction_pct']} %",
                   "iov_status": _status(size_red >= iov["min_size_reduction_pct"]),
                   "goal": f">= {goals['min_size_reduction_pct']} % y <= {goals['max_model_file_mb']} MB",
                   "goal_status": _status(size_red >= goals["min_size_reduction_pct"]
                                          and int8_bytes / MB <= goals["max_model_file_mb"])})

    pipe = peaks.get("pipeline", {})
    if "fp32" in pipe and "int8" in pipe:
        mem_red = -relative_change_pct(pipe["fp32"], pipe["int8"])
        mem_ok = mem_red >= iov["min_memory_reduction_pct"]
        goal_ok = mem_red >= goals["min_memory_reduction_pct"] and pipe["int8"] / MB <= goals["max_pipeline_peak_mb"]
        value = f"{mem_red:.2f} % (pico flujo completo {pipe['fp32'] / MB:.1f} -> {pipe['int8'] / MB:.1f} MB)"
    else:
        mem_ok = goal_ok = None
        value = "sin medicion"
    status.append({"dataset": name, "result": 2, "indicator": "reduccion_memoria (pico RAM del flujo completo)",
                   "value": value, "iov": f">= {iov['min_memory_reduction_pct']} %", "iov_status": _status(mem_ok),
                   "goal": f">= {goals['min_memory_reduction_pct']} % y <= {goals['max_pipeline_peak_mb']} MB",
                   "goal_status": _status(goal_ok)})

    if e2e is not None and {"fp32", "int8"} <= set(e2e["version"]):
        increases, faster_than_audio = [], []
        for threads, group in e2e.groupby("threads"):
            g = group.set_index("version")
            increases.append((int(threads), relative_change_pct(g.loc["fp32", "median"], g.loc["int8", "median"])))
            faster_than_audio.append(float(g.loc["int8", "share_faster_than_audio"]))
        time_ok = all(v <= iov["max_time_increase_pct"] for _, v in increases)
        time_goal = all(v <= 0 for _, v in increases) and all(s == 1.0 for s in faster_than_audio)
        value = "; ".join(f"{t} hilos: {v:+.2f} %" for t, v in increases)
    else:
        time_ok = time_goal = None
        value = "sin medicion"
    status.append({"dataset": name, "result": 2, "indicator": "cambio_tiempo_total (mediana WAV -> decision)",
                   "value": value, "iov": f"<= +{iov['max_time_increase_pct']} %", "iov_status": _status(time_ok),
                   "goal": "<= 0 % y tiempo total menor que la duracion de cada grabacion",
                   "goal_status": _status(time_goal)})

    status += performance_status(name, test, iov, goals)
    status.append(offline_status(name, bundle, bmod.bundle_file_sha256(ds_dir), offline))
    status.append({"dataset": name, "result": 2, "indicator": "validacion_experto_IA (medio de verificacion)",
                   "value": str(expert_review) if expert_review else "acta pendiente",
                   "iov": "revision documentada", "iov_status": "DOCUMENTADO" if expert_review else "PENDIENTE",
                   "goal": "-", "goal_status": "-"})
    return table, status


def _to_markdown(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(f"{v:.4f}" if isinstance(v, float) else ("" if v is None else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = paths.load_config(known.config)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=paths.dataset_choices(cfg), default="all")
    parser.add_argument("--runs-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--label", default="phantom")
    parser.add_argument("--offline-comparison", type=Path, default=None)
    parser.add_argument("--expert-review", type=Path, default=None)
    args = parser.parse_args(argv)
    runs_root = paths.resolve_runs_root(args.runs_root, cfg)
    output_root = paths.resolve_output_root(args.output_root, cfg, runs_root)
    run_dir = paths.resolve_run_dir(args.run_dir, args.run_id, output_root, create=False)
    offline = paths.read_json(args.offline_comparison) if args.offline_comparison else None
    if args.expert_review is not None and not args.expert_review.is_file():
        print(f"no existe el acta {args.expert_review}", file=sys.stderr)
        return 2

    tables, statuses = [], []
    for entry in paths.dataset_entries(cfg, args.dataset):
        t, s = dataset_report(entry["name"], cfg, run_dir, args.label, offline, args.expert_review)
        tables += t
        statuses += s
    out = run_dir / "report"
    table = pd.DataFrame(tables)[["dataset", "block", "criterion", "base_pt_publicado", "onnx_fp32", "onnx_int8"]]
    status = pd.DataFrame(statuses)
    paths.write_csv_atomic(table, out / "benchmark_table.csv")
    paths.write_csv_atomic(status, out / "iov_status.csv")
    (out / "benchmark_table.md").write_text(
        "# Benchmarking ONNX FP32 frente a ONNX INT8\n\n"
        "Clasificacion: metricas por paciente en test. El modelo base `.pt` se evaluo en la corrida final y sus "
        "metricas publicadas se reutilizan; ONNX FP32 no se evaluo en test.\n\n"
        "Recursos: misma CPU y mismas condiciones para FP32 e INT8, con entradas de desarrollo.\n\n"
        + _to_markdown(table), encoding="utf-8")
    (out / "iov_status.md").write_text(
        "# Estado de los indicadores\n\nIOV = compromiso original; meta = hipotesis mas estricta.\n\n"
        + _to_markdown(status), encoding="utf-8")
    print(f"informe escrito en {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
