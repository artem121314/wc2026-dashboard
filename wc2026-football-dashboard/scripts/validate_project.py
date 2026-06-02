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


ReportRow = tuple[str, str, str]


def add(rows: list[ReportRow], level: str, section: str, message: str) -> None:
    rows.append((level, section, message))


def validate_project_structure(rows: list[ReportRow]) -> None:
    required_paths = [
        "app/streamlit_app.py",
        "src/data_loader.py",
        "src/data_pipeline.py",
        "src/data_sources.py",
        "src/feature_engineering.py",
        "src/model.py",
        "src/tournament_data.py",
        "src/tournament_impact.py",
        "src/visualisations.py",
        "data/raw",
        "data/historical",
        "data/processed",
        "README.md",
        "requirements.txt",
    ]
    missing = [path for path in required_paths if not (PROJECT_ROOT / path).exists()]
    if missing:
        add(rows, "FAIL", "Project structure", "Missing required project paths: " + ", ".join(missing))
    else:
        add(rows, "PASS", "Project structure", "Core files and directories are present.")


def validate_source_files(rows: list[ReportRow]) -> None:
    status = check_data_availability()
    labels = {
        "current_player_pool": "data/raw/current_player_pool.csv",
        "market_values": "data/raw/market_values.csv",
        "national_team_context": "data/raw/national_team_context.csv",
        "player_performance_inputs": "data/raw/player_performance_inputs.csv",
        "tournament_match_data": "data/raw/tournament_match_data.csv",
        "historical_training": "data/historical/world_cup_player_training_data.csv",
        "processed_dashboard": "data/processed/player_dashboard_data.csv",
    }
    for key, label in labels.items():
        item = status[key]
        spec = DATA_SOURCE_SPECS[key]
        if not item["exists"]:
            level = "WARN"
            if key == "tournament_match_data":
                message = f"{label} is missing. Actual tournament impact remains pending."
            elif key == "historical_training":
                message = f"{label} is missing. The supervised model remains disabled."
            elif key == "processed_dashboard":
                message = f"{label} is missing. Run Refresh data after adding real raw inputs."
            else:
                message = f"{label} is missing. Add a compliant real CSV before relying on refresh."
            add(rows, level, "Data files", message)
            continue
        if item["missing_columns"]:
            add(rows, "FAIL", "Data files", f"{label} is missing columns: {', '.join(item['missing_columns'])}")
            continue
        if item["rows"] == 0:
            if key == "tournament_match_data":
                add(rows, "WARN", "Data files", f"{label} exists with zero rows. Actual impact is pending.")
            elif key == "historical_training":
                add(rows, "WARN", "Data files", f"{label} exists with zero rows. The supervised model remains disabled.")
            else:
                add(rows, "WARN", "Data files", f"{label} exists but is empty.")
            continue
        add(rows, "PASS", "Data files", f"{label} loads with {int(item['rows']):,} rows.")


def validate_processed_data(rows: list[ReportRow]) -> None:
    df = load_processed_data()
    if df.empty:
        add(rows, "WARN", "Processed data", "Processed dashboard data is missing or empty.")
        return

    add(rows, "PASS", "Processed data", f"Processed dashboard data loads with {len(df):,} rows.")
    required = DATA_SOURCE_SPECS["processed_dashboard"].required_columns
    missing_required = [col for col in required if col not in df.columns]
    if missing_required:
        add(rows, "FAIL", "Processed data", "Missing processed columns: " + ", ".join(missing_required))
    else:
        add(rows, "PASS", "Processed data", "Required processed dashboard columns are present.")

    for col in SCORE_COLUMNS:
        if col not in df.columns:
            add(rows, "WARN", "Score ranges", f"Score column not present: {col}")
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            add(rows, "WARN", "Score ranges", f"{col} has no available real values yet.")
        elif values.between(0, 100).all():
            add(rows, "PASS", "Score ranges", f"{col} values are within 0-100.")
        else:
            add(rows, "FAIL", "Score ranges", f"{col} contains values outside 0-100.")

    if "performance_outcome" not in df.columns:
        add(rows, "FAIL", "Outcomes", "performance_outcome column is missing.")
    else:
        invalid = set(df["performance_outcome"].dropna().unique()) - VALID_OUTCOMES
        if invalid:
            add(rows, "FAIL", "Outcomes", "Invalid performance_outcome values: " + ", ".join(sorted(invalid)))
        else:
            add(rows, "PASS", "Outcomes", "performance_outcome values are valid.")


def validate_imports(rows: list[ReportRow]) -> None:
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
    failed: list[str] = []
    for module in modules:
        try:
            __import__(module)
        except Exception as exc:  # pragma: no cover - validation script
            failed.append(f"{module}: {exc}")
    if failed:
        add(rows, "FAIL", "Imports", "Import failures: " + " | ".join(failed))
    else:
        add(rows, "PASS", "Imports", "Application modules import successfully.")


def print_report(rows: list[ReportRow]) -> None:
    print("WC 2026 Value Opportunity Dashboard validation")
    print("=" * 52)
    for level, section, message in rows:
        print(f"[{level}] {section}: {message}")
    print("=" * 52)
    counts = {level: sum(1 for row in rows if row[0] == level) for level in ["PASS", "WARN", "FAIL"]}
    print(f"Summary: {counts['PASS']} PASS, {counts['WARN']} WARN, {counts['FAIL']} FAIL")


def main() -> int:
    rows: list[ReportRow] = []
    validate_project_structure(rows)
    validate_source_files(rows)
    validate_processed_data(rows)
    validate_imports(rows)
    print_report(rows)
    return 1 if any(level == "FAIL" for level, _, _ in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
