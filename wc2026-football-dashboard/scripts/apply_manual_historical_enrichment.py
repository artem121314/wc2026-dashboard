"""Apply manually curated real historical predictor enrichment.

The script updates only fields supplied in a compliant manual CSV. It does not
guess missing values, fuzzy-match players, or overwrite values with blanks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_sources import HISTORICAL_DATA_DIR, HISTORICAL_TRAINING_PATH


DEFAULT_INPUT_PATH = HISTORICAL_DATA_DIR / "manual_enrichment_template.csv"
TEMPLATE_PATH = HISTORICAL_DATA_DIR / "manual_enrichment_template.csv"
AUDIT_LOG_PATH = HISTORICAL_DATA_DIR / "source_audit_log.csv"

KEY_COLUMNS = ["tournament_year", "player_name"]
OPTIONAL_CONTEXT_COLUMNS = ["country", "position"]
ENRICHMENT_COLUMNS = [
    "club",
    "league",
    "market_value_before_tournament",
    "club_level_score",
    "league_strength_score",
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
AUDIT_COLUMNS = [
    "source_name",
    "source_url_or_location",
    "source_type",
    "license_or_usage_note",
    "access_method",
    "date_accessed",
    "fields_used",
    "allowed_for_project",
    "notes",
]


def _normalise_key(df: pd.DataFrame) -> pd.DataFrame:
    normalised = df.copy()
    normalised["tournament_year"] = pd.to_numeric(normalised["tournament_year"], errors="coerce").astype("Int64")
    normalised["player_name"] = normalised["player_name"].astype(str).str.strip()
    if "country" in normalised.columns:
        normalised["country"] = normalised["country"].astype(str).str.strip()
        normalised.loc[normalised["country"].str.lower().isin(["", "nan", "none", "<na>"]), "country"] = pd.NA
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


def _value_supplied(df: pd.DataFrame) -> pd.Series:
    return df[ENRICHMENT_COLUMNS].notna().any(axis=1)


def _merge_key_columns(historical: pd.DataFrame, enrichment: pd.DataFrame) -> list[str]:
    if (
        "country" in historical.columns
        and "country" in enrichment.columns
        and enrichment.loc[_value_supplied(enrichment), "country"].notna().any()
    ):
        return [*KEY_COLUMNS, "country"]
    return KEY_COLUMNS


def _save_backup(historical_path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup_path = historical_path.with_name(f"{historical_path.stem}.backup_{timestamp}{historical_path.suffix}")
    backup_path.write_bytes(historical_path.read_bytes())
    return backup_path


def _update_audit_log(input_path: Path, enrichment: pd.DataFrame, update_counts: dict[str, int]) -> None:
    supplied = enrichment.loc[_value_supplied(enrichment)].copy()
    if supplied.empty:
        return

    audit_rows = []
    accessed = datetime.now(timezone.utc).date().isoformat()
    fields_used = ", ".join(col for col, count in update_counts.items() if count and col in ENRICHMENT_COLUMNS)
    for data_source, source_rows in supplied.groupby("data_source", dropna=True):
        urls = sorted(
            {
                str(url).strip()
                for url in source_rows.get("source_url", pd.Series(dtype=object)).dropna().tolist()
                if str(url).strip()
            }
        )
        notes = sorted(
            {
                str(note).strip()
                for note in source_rows.get("notes", pd.Series(dtype=object)).dropna().tolist()
                if str(note).strip()
            }
        )
        audit_rows.append(
            {
                "source_name": str(data_source),
                "source_url_or_location": "; ".join(urls) or str(input_path),
                "source_type": "manual/user-provided CSV",
                "license_or_usage_note": "Manual enrichment supplied by the user from compliant real sources.",
                "access_method": "local CSV merge",
                "date_accessed": accessed,
                "fields_used": fields_used,
                "allowed_for_project": True,
                "notes": "; ".join(notes) or "Applied through scripts/apply_manual_historical_enrichment.py.",
            }
        )

    existing = pd.DataFrame(columns=AUDIT_COLUMNS)
    if AUDIT_LOG_PATH.exists():
        try:
            existing = pd.read_csv(AUDIT_LOG_PATH)
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame(columns=AUDIT_COLUMNS)
    for col in AUDIT_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
    updated = pd.concat([existing[AUDIT_COLUMNS], pd.DataFrame(audit_rows)[AUDIT_COLUMNS]], ignore_index=True)
    updated.to_csv(AUDIT_LOG_PATH, index=False)


def apply_manual_enrichment(
    input_path: Path = DEFAULT_INPUT_PATH,
    historical_path: Path = HISTORICAL_TRAINING_PATH,
    output_path: Path = HISTORICAL_TRAINING_PATH,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Manual enrichment file not found: {input_path}. Fill {TEMPLATE_PATH} or pass --input with another real sourced CSV."
        )
    if not historical_path.exists():
        raise FileNotFoundError(f"Historical training file not found: {historical_path}")

    historical = _normalise_key(pd.read_csv(historical_path))
    enrichment = pd.read_csv(input_path)
    issues = _validate_input(enrichment)
    if issues:
        raise ValueError(" | ".join(issues))
    enrichment = _normalise_key(enrichment)

    key_columns = _merge_key_columns(historical, enrichment)
    duplicate_historical = historical.duplicated(key_columns, keep=False)
    if duplicate_historical.any():
        duplicate_preview = historical.loc[duplicate_historical, key_columns].head(10).to_dict("records")
        raise ValueError(
            "Historical training data has duplicate merge keys for the selected join columns. "
            "Provide country in the enrichment CSV so the merge is unambiguous. Preview: "
            f"{duplicate_preview}"
        )
    supplied_enrichment = enrichment.loc[_value_supplied(enrichment)].copy()
    historical_keys = set(historical[key_columns].apply(tuple, axis=1))
    enrichment_keys = supplied_enrichment[key_columns].apply(tuple, axis=1) if not supplied_enrichment.empty else pd.Series(dtype=object)
    unmatched = sorted(set(enrichment_keys) - historical_keys)
    matched_enrichment = supplied_enrichment.loc[~enrichment_keys.isin(unmatched)].copy() if not supplied_enrichment.empty else supplied_enrichment

    backup_path = _save_backup(historical_path) if output_path == historical_path else None
    historical = historical.set_index(key_columns)
    matched_enrichment = matched_enrichment.set_index(key_columns)
    update_counts: dict[str, int] = {}

    for col in ENRICHMENT_COLUMNS:
        if col not in historical.columns:
            historical[col] = pd.NA
        values = matched_enrichment[col] if col in matched_enrichment.columns else pd.Series(dtype=object)
        if values.empty:
            update_counts[col] = 0
            continue
        supplied_mask = values.notna()
        if overwrite:
            write_mask = supplied_mask
        else:
            current_values = historical.loc[values.index, col]
            write_mask = supplied_mask & current_values.isna()
        historical.loc[write_mask.index[write_mask], col] = values.loc[write_mask]
        update_counts[col] = int(write_mask.sum())

    for source_col, target_col in PROVENANCE_COLUMNS.items():
        if target_col not in historical.columns:
            historical[target_col] = pd.NA
        values = matched_enrichment[source_col] if source_col in matched_enrichment.columns else pd.Series(dtype=object)
        if values.empty:
            update_counts[target_col] = 0
            continue
        supplied_mask = matched_enrichment[ENRICHMENT_COLUMNS].notna().any(axis=1) & values.notna()
        if overwrite:
            write_mask = supplied_mask
        else:
            current_values = historical.loc[values.index, target_col]
            write_mask = supplied_mask & current_values.isna()
        historical.loc[write_mask.index[write_mask], target_col] = values.loc[write_mask]
        update_counts[target_col] = int(write_mask.sum())

    enriched = historical.reset_index()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output_path, index=False)
    _update_audit_log(input_path, matched_enrichment.reset_index(), update_counts)
    report = {
        "update_counts": update_counts,
        "unmatched_rows": len(unmatched),
        "unmatched_preview": unmatched[:20],
        "matched_rows": len(matched_enrichment),
        "key_columns": key_columns,
        "backup_path": str(backup_path) if backup_path else "",
    }
    return enriched, report


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply manual historical predictor enrichment from a real sourced CSV.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Manual enrichment CSV path.")
    parser.add_argument("--historical", type=Path, default=HISTORICAL_TRAINING_PATH, help="Historical training CSV path.")
    parser.add_argument("--output", type=Path, default=HISTORICAL_TRAINING_PATH, help="Output historical CSV path.")
    parser.add_argument("--overwrite", choices=["true", "false"], default="false", help="Overwrite existing non-null values.")
    args = parser.parse_args()

    try:
        enriched, report = apply_manual_enrichment(
            args.input,
            args.historical,
            args.output,
            overwrite=args.overwrite.lower() == "true",
        )
    except Exception as exc:
        print("Manual historical enrichment was not applied.")
        print(f"Reason: {exc}")
        print("No synthetic fallback will be generated.")
        return 1

    print("Manual historical enrichment applied")
    print("=" * 52)
    print(f"Rows in historical file: {len(enriched):,}")
    print(f"Output file: {args.output}")
    if report.get("backup_path"):
        print(f"Backup saved: {report['backup_path']}")
    print("Join keys: " + ", ".join(str(col) for col in report["key_columns"]))
    print(f"Matched enrichment rows: {report['matched_rows']:,}")
    print(f"Unmatched enrichment rows: {report['unmatched_rows']:,}")
    if report["unmatched_preview"]:
        print("Unmatched preview:")
        for row in report["unmatched_preview"]:
            print(f"  - {row}")
    print("Updated value counts:")
    for col, count in report["update_counts"].items():
        if count:
            print(f"  - {col}: {count:,}")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
