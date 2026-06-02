"""Check whether real historical data can enable the supervised model."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from model import PREDICTOR_COVERAGE_WARNING
from model import check_historical_model_readiness
from model import historical_feature_availability
from model import load_historical_training_data


def _status(value: bool) -> str:
    return "available" if value else "missing"


def _feature_availability(data: pd.DataFrame) -> pd.DataFrame:
    return historical_feature_availability(data)


def _yes_no(value: object) -> str:
    return "yes" if bool(value) else "no"


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
    print(f"Historical target rows available: {readiness['target_available_rows']}")
    print(f"Usable target rows: {readiness.get('usable_target_rows', 0)}")
    print(f"Target range valid: {readiness['target_valid_range']}")
    if readiness.get("score_range_issues"):
        print("Score range issues: " + ", ".join(str(col) for col in readiness["score_range_issues"]))
    print(f"Minimum training rows: {readiness.get('minimum_training_rows')}")
    print(f"Minimum required predictor coverage: {readiness.get('minimum_required_predictor_coverage')}")
    print(f"Activation predictor values complete: {_yes_no(readiness['required_feature_values_present'])}")
    print(f"Predictor coverage sufficient: {_yes_no(readiness.get('predictor_coverage_sufficient'))}")
    print(f"Activation status: {readiness.get('activation_status')}")
    missing_groups = readiness.get("missing_predictor_groups", [])
    if missing_groups:
        print("Missing predictor groups: " + ", ".join(str(group) for group in missing_groups))
    usable_recruitment = readiness.get("usable_recruitment_predictors", [])
    print(
        "Usable recruitment predictors: "
        + (", ".join(str(feature) for feature in usable_recruitment) or "none")
        + f" ({readiness.get('usable_recruitment_predictor_count', 0)}/"
        + f"{readiness.get('minimum_recruitment_predictors_required', 0)} required)"
    )
    missing_features = readiness.get("missing_required_feature_values", [])
    if missing_features:
        print("Unavailable activation predictor values: " + ", ".join(str(col) for col in missing_features))
    below_coverage = readiness.get("required_features_below_minimum_coverage", [])
    if below_coverage:
        print("Required predictors below coverage threshold: " + ", ".join(str(col) for col in below_coverage))
    print(f"Can train supervised model: {readiness['can_train']}")
    print(f"Supervised model available: {readiness['model_available']}")
    print(f"Model status: {readiness['model_status']}")
    if readiness.get("disabled_reason"):
        print(f"Disabled reason: {readiness['disabled_reason']}")
    if readiness.get("readiness_message"):
        print(f"Readiness message: {readiness['readiness_message']}")

    model_summary = readiness.get("model_summary")
    if isinstance(model_summary, dict) and model_summary.get("model_available"):
        print(f"Selected model: {model_summary.get('selected_model_name')}")
        metrics = model_summary.get("metrics")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            print("Model metrics:")
            print(metrics.to_string(index=False))

    historical_data = load_historical_training_data()
    if not historical_data.empty:
        print("Feature availability:")
        feature_table = _feature_availability(historical_data).copy()
        display_columns = [
            "feature",
            "feature_group",
            "requirement",
            "non_null",
            "pct_non_null",
            "minimum_pct_required",
            "distinct_values",
            "usable_for_modelling",
        ]
        print(feature_table[display_columns].to_string(index=False))

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
    print("Conclusion:")
    if readiness.get("model_available"):
        selected = None
        if isinstance(model_summary, dict):
            selected = model_summary.get("selected_model_name")
        print(f"  MODEL ENABLED: enough real predictor coverage exists. Selected model: {selected or 'available'}.")
    elif readiness.get("target_available_rows", 0) == 0:
        print("  MODEL DISABLED: historical player-tournament target rows are missing.")
    elif readiness.get("disabled_reason") == PREDICTOR_COVERAGE_WARNING or readiness.get("missing_predictor_groups"):
        print("  MODEL DISABLED: target rows exist, but market/club-season predictor coverage is insufficient.")
    else:
        print(f"  MODEL DISABLED: {readiness.get('disabled_reason', 'historical data is not ready')}")
    print("=" * 52)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
