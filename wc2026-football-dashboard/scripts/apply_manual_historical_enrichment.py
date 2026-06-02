"""Apply manually curated real historical predictor enrichment.

The script updates only fields supplied in a compliant manual CSV. It does not
guess missing values, fuzzy-match players, or overwrite values with blanks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_sources import HISTORICAL_DATA_DIR, HISTORICAL_TRAINING_PATH


DEFAULT_INPUT_PATH = HISTORICAL_DATA_DIR / "manual_enrichment.csv"
TEMPLATE_PATH = HISTORICAL_DATA_DIR / "manual_enrichment_template.csv"

KEY_COLUMNS = ["tournament_year", "player_name"]
OPTIONAL_CONTEXT_COLUMNS = ["country", "position"]
ENRICHMENT_COLUMNS = [
    "club",
    "league",
    "market_value_before_tournament",
    "club_minutes_previous_season",
    "goals_previous_season",
    "assists_previous_season",
    "senior_national_team_caps",
    "injury_availability_score",
    "recent_form_score",
    "role_fit_score",
]
PROVENANCE_COLUMNS = {
    "data_source": "manual_enrichment_data_source",
    "source_url": "manual_enrichment_source_url",
    "notes": "manual_enrichment_notes",
}
REQUIRED_INPUT_COLUMNS = [*KEY_COLUMNS, *OPTIONAL_CONTEXT_COLUMNS, *ENRICHMENT_COLUMNS, *PROVENANCE_COLUMNS.keys()]


def _normalise_key(df: pd.DataFrame) -> pd.DataFrame:
    normalised = df.copy()
    normalised["tournament_year"] = pd.to_numeric(normalised["tournament_year"], errors="coerce").astype("Int64")
    normalised["player_name"] = normalised["player_name"].astype(str).str.strip()
    return normalised


def _validate_input(enrichment: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in enrichment.columns]
    if missing:
        issues.append("Manual enrichment file is missing columns: " + ", ".join(missing))
        return issues

    if enrichment.empty:
        issues.append("Manual enrichment file has zero rows.")

    duplicate_keys = enrichment.duplicated(KEY_COLUMNS, keep=False)
    if duplicate_keys.any():
        duplicate_preview = enrichment.loc[duplicate_keys, KEY_COLUMNS].head(10).to_dict("records")
        issues.append(f"Manual enrichment file contains duplicate player-year keys: {duplicate_preview}")

    no_source = enrichment["data_source"].isna() | enrichment["data_source"].astype(str).str.strip().eq("")
    value_supplied = enrichment[ENRICHMENT_COLUMNS].notna().any(axis=1)
    if (value_supplied & no_source).any():
        issues.append("Rows with enrichment values must include data_source provenance.")

    return issues


def apply_manual_enrichment(
    input_path: Path = DEFAULT_INPUT_PATH,
    historical_path: Path = HISTORICAL_TRAINING_PATH,
    output_path: Path = HISTORICAL_TRAINING_PATH,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Manual enrichment file not found: {input_path}. Copy {TEMPLATE_PATH} to {DEFAULT_INPUT_PATH} and fill it with real sourced values."
        )
    if not historical_path.exists():
        raise FileNotFoundError(f"Historical training file not found: {historical_path}")

    historical = _normalise_key(pd.read_csv(historical_path))
    enrichment = pd.read_csv(input_path)
    issues = _validate_input(enrichment)
    if issues:
        raise ValueError(" | ".join(issues))
    enrichment = _normalise_key(enrichment)

    historical_keys = historical[KEY_COLUMNS].apply(tuple, axis=1)
    enrichment_keys = enrichment[KEY_COLUMNS].apply(tuple, axis=1)
    unmatched = sorted(set(enrichment_keys) - set(historical_keys))
    if unmatched:
        preview = unmatched[:10]
        raise ValueError(f"Manual enrichment contains keys not found in historical training data: {preview}")

    historical = historical.set_index(KEY_COLUMNS)
    enrichment = enrichment.set_index(KEY_COLUMNS)
    update_counts: dict[str, int] = {}

    for col in ENRICHMENT_COLUMNS:
        if col not in historical.columns:
            historical[col] = pd.NA
        values = enrichment[col]
        mask = values.notna()
        historical.loc[mask.index[mask], col] = values.loc[mask]
        update_counts[col] = int(mask.sum())

    for source_col, target_col in PROVENANCE_COLUMNS.items():
        if target_col not in historical.columns:
            historical[target_col] = pd.NA
        values = enrichment[source_col]
        mask = enrichment[ENRICHMENT_COLUMNS].notna().any(axis=1) & values.notna()
        historical.loc[mask.index[mask], target_col] = values.loc[mask]
        update_counts[target_col] = int(mask.sum())

    enriched = historical.reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    return enriched, update_counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply manual historical predictor enrichment from a real sourced CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Manual enrichment CSV path.")
    parser.add_argument("--historical", type=Path, default=HISTORICAL_TRAINING_PATH, help="Historical training CSV path.")
    parser.add_argument("--output", type=Path, default=HISTORICAL_TRAINING_PATH, help="Output historical CSV path.")
    args = parser.parse_args()

    try:
        enriched, counts = apply_manual_enrichment(args.input, args.historical, args.output)
    except Exception as exc:
        print("Manual historical enrichment was not applied.")
        print(f"Reason: {exc}")
        print("No synthetic fallback will be generated.")
        return 1

    print("Manual historical enrichment applied")
    print("=" * 52)
    print(f"Rows in historical file: {len(enriched):,}")
    print(f"Output file: {args.output}")
    print("Updated non-null counts:")
    for col, count in counts.items():
        if count:
            print(f"  - {col}: {count:,}")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
