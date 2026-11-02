"""Marcas persistentes de fallo por fold (sobreviven a la limpieza de staging)."""

from __future__ import annotations

import json
from pathlib import Path

from .. import artifacts as art


# ---------------------------------------------------------------------------
# Un fold, con reintentos y marcas de fallo persistentes
# ---------------------------------------------------------------------------

def _fold_failure_marker(run_root: Path, dataset: str, condition: str, fold_id: int) -> Path:
    return art.condition_dir(run_root, dataset, condition) / f"fold_{fold_id + 1:02d}_FAILED.json"


def _write_fold_failure(run_root: Path, dataset: str, condition: str, fold_id: int, exc: Exception) -> None:
    marker = _fold_failure_marker(run_root, dataset, condition, fold_id)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({
            "fold": fold_id, "error": str(exc), "type": type(exc).__name__,
            "failed_at_utc": art._now_iso(),
        }, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _clear_fold_failure_marker(run_root: Path, dataset: str, condition: str, fold_id: int) -> None:
    marker = _fold_failure_marker(run_root, dataset, condition, fold_id)
    if marker.is_file():
        marker.unlink()
