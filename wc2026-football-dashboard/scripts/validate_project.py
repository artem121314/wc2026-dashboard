"""Validate the real-data-only WC 2026 dashboard project."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_pipeline import check_data_availability, load_processed_data
from data_sources import DATA_SOURCE_SPECS


VALID_OUTCOMES = {"Pending", "Overperformed", "Met expectations", "Underperformed"}
SCORE_COLUMNS = [
    "pre_tournament_expected_impact_score",
    "baseline_expected_impact_score",
    "actual_tournament_impact_score",
    "value_efficiency_score",
    "value_opportunity_score",
    "availability_score",
    "risk_score",
]


def validate_source_files() -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    status = check_data_availability()
    for key, item in status.items():
        spec = DATA_SOURCE_SPECS[key]
        if not item["exists"] or item["rows"] == 0:
            message = f"{spec.name} missing or empty: {spec.path}"
            if spec.required_for_refresh:
                warnings.append(message)
            else:
                warnings.append(message)
            continue
        if item["missing_columns"]:
            errors.append(f"{spec.name} missing columns: {', '.join(item['missing_columns'])}")
    return warnings, errors


def validate_processed_data() -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    df = load_processed_data()
    if df.empty:
        warnings.append("Processed dashboard data is missing or empty.")
        return warnings, errors

    for col in SCORE_COLUMNS:
        if col not in df.columns:
            warnings.append(f"Score column not present: {col}")
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if not values.between(0, 100).all():
            errors.append(f"Score column outside 0-100 range: {col}")

    if "performance_outcome" not in df.columns:
        errors.append("performance_outcome column is missing.")
    else:
        invalid = set(df["performance_outcome"].dropna().unique()) - VALID_OUTCOMES
        if invalid:
            errors.append(f"Invalid performance_outcome values: {', '.join(sorted(invalid))}")

    return warnings, errors


def validate_imports() -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    errors: list[str] = []
    modules = [
        "data_loader",
        "data_pipeline",
        "data_sources",
        "feature_engineering",
        "model",
        "preprocessing",
        "tournament_data",
        "tournament_impact",
        "utils",
        "visualisations",
    ]
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:  # pragma: no cover - validation script
            errors.append(f"Import failed for {module}: {exc}")
    return warnings, errors


def main() -> int:
    all_warnings: list[str] = []
    all_errors: list[str] = []
    for validator in [validate_source_files, validate_processed_data, validate_imports]:
        warnings, errors = validator()
        all_warnings.extend(warnings)
        all_errors.extend(errors)

    print("Validation warnings:")
    if all_warnings:
        for warning in all_warnings:
            print(f"  - {warning}")
    else:
        print("  none")

    print("Validation errors:")
    if all_errors:
        for error in all_errors:
            print(f"  - {error}")
        return 1
    print("  none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
