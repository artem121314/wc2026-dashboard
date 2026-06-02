"""Check whether real historical data can enable the supervised model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from model import check_historical_model_readiness


def _status(value: bool) -> str:
    return "available" if value else "missing"


def main() -> int:
    readiness = check_historical_model_readiness(attempt_training=True)

    print("Historical expected-impact model readiness")
    print("=" * 52)
    print(f"Training file: {readiness['training_path']}")
    print(f"Training file status: {_status(bool(readiness['training_file_exists']))}")
    print(f"Template file: {readiness['template_path']}")
    print(f"Template status: {_status(bool(readiness['template_exists']))}")
    print(f"Rows: {readiness['row_count']}")
    print(f"Required columns present: {readiness['required_columns_present']}")
    missing = readiness.get("missing_required_columns", [])
    if missing:
        print("Missing required columns: " + ", ".join(str(col) for col in missing))
    print("Tournament years available: " + (", ".join(str(year) for year in readiness["tournament_years_available"]) or "none"))
    missing_years = readiness.get("missing_expected_tournament_years", [])
    if missing_years:
        print("Expected tournament years not yet present: " + ", ".join(str(year) for year in missing_years))
    print(f"Target rows available: {readiness['target_available_rows']}")
    print(f"Target range valid: {readiness['target_valid_range']}")
    if readiness.get("score_range_issues"):
        print("Score range issues: " + ", ".join(str(col) for col in readiness["score_range_issues"]))
    print(f"Can train supervised model: {readiness['can_train']}")
    print(f"Supervised model available: {readiness['model_available']}")
    print(f"Model status: {readiness['model_status']}")
    if readiness.get("disabled_reason"):
        print(f"Disabled reason: {readiness['disabled_reason']}")

    model_summary = readiness.get("model_summary")
    if isinstance(model_summary, dict) and model_summary.get("model_available"):
        print(f"Selected model: {model_summary.get('selected_model_name')}")
        metrics = model_summary.get("metrics")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            print("Model metrics:")
            print(metrics.to_string(index=False))

    warnings = list(readiness.get("warnings", []))
    errors = list(readiness.get("errors", []))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")
    print("=" * 52)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
