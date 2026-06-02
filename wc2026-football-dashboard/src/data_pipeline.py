"""Real-data-only dashboard refresh pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_sources import (
    DATA_SOURCE_SPECS,
    PROCESSED_DASHBOARD_PATH as DEFAULT_PROCESSED_DASHBOARD_PATH,
    PROCESSED_DATA_DIR,
)
from tournament_data import load_aggregated_tournament_data


PROCESSED_DASHBOARD_PATH = DEFAULT_PROCESSED_DASHBOARD_PATH
MANIFEST_PATH = PROCESSED_DATA_DIR / "dashboard_refresh_manifest.json"


def _read_csv_if_available(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _source_status(spec_key: str) -> dict[str, Any]:
    spec = DATA_SOURCE_SPECS[spec_key]
    exists = spec.path.exists()
    rows = 0
    missing_columns: list[str] = []
    if exists:
        data = _read_csv_if_available(spec.path)
        rows = len(data)
        missing_columns = [col for col in spec.required_columns if col not in data.columns]
    available = exists and rows > 0 and not missing_columns
    if available:
        status = "available"
    elif exists and spec_key == "tournament_match_data":
        status = "pending"
    elif exists and rows == 0:
        status = "empty"
    else:
        status = "missing"
    return {
        "name": spec.name,
        "path": str(spec.path),
        "exists": exists,
        "rows": rows,
        "missing_columns": missing_columns,
        "available": available,
        "status": status,
        "required_for_refresh": spec.required_for_refresh,
        "description": spec.description,
    }


def check_data_availability() -> dict[str, dict[str, Any]]:
    """Report availability of real raw, historical and processed files."""

    return {key: _source_status(key) for key in DATA_SOURCE_SPECS}


def load_raw_real_data() -> dict[str, pd.DataFrame]:
    """Load local real CSV files. Missing files return empty frames."""

    return {
        key: _read_csv_if_available(spec.path)
        for key, spec in DATA_SOURCE_SPECS.items()
        if key not in {"processed_dashboard"}
    }


def validate_raw_data(raw_data: dict[str, pd.DataFrame] | None = None) -> dict[str, list[str]]:
    """Validate available real CSV inputs without requiring optional files."""

    raw_data = load_raw_real_data() if raw_data is None else raw_data
    warnings: list[str] = []
    errors: list[str] = []
    for key, spec in DATA_SOURCE_SPECS.items():
        if key == "processed_dashboard":
            continue
        data = raw_data.get(key, pd.DataFrame())
        if data.empty:
            if spec.required_for_refresh:
                errors.append(f"{spec.name} is missing or empty: {spec.path}")
            else:
                warnings.append(f"{spec.name} is missing or empty: {spec.path}")
            continue
        missing = [col for col in spec.required_columns if col not in data.columns]
        if missing:
            message = f"{spec.name} is missing required columns: {', '.join(missing)}"
            if spec.required_for_refresh:
                errors.append(message)
            else:
                warnings.append(message)
    current = raw_data.get("current_player_pool", pd.DataFrame())
    market = raw_data.get("market_values", pd.DataFrame())
    national = raw_data.get("national_team_context", pd.DataFrame())
    if not current.empty:
        if "market_value_eur" not in current.columns and (
            market.empty or not {"player_name", "market_value_eur"}.issubset(market.columns)
        ):
            warnings.append(
                "Market value data is unavailable; value efficiency and value opportunity scores will be disabled."
            )
        if "national_team_strength" not in current.columns and (
            national.empty or not {"country", "national_team_strength"}.issubset(national.columns)
        ):
            warnings.append(
                "National team strength is unavailable; baseline expected impact may be disabled."
            )
    return {"warnings": warnings, "errors": errors}


def _merge_optional_sources(player_pool: pd.DataFrame, raw_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = player_pool.copy()

    performance_inputs = raw_data.get("player_performance_inputs", pd.DataFrame())
    if not performance_inputs.empty:
        merge_cols = ["player_name"]
        if "country" in performance_inputs.columns and "country" in df.columns:
            merge_cols.append("country")
        performance_cols = [col for col in performance_inputs.columns if col not in merge_cols]
        df = df.drop(columns=[col for col in performance_cols if col in df.columns], errors="ignore")
        df = df.merge(performance_inputs[[*merge_cols, *performance_cols]], on=merge_cols, how="left")

    market_values = raw_data.get("market_values", pd.DataFrame())
    if not market_values.empty:
        merge_cols = ["player_name"]
        if "country" in market_values.columns and "country" in df.columns:
            merge_cols.append("country")
        value_cols = [col for col in market_values.columns if col not in merge_cols]
        df = df.drop(columns=[col for col in value_cols if col in df.columns], errors="ignore")
        df = df.merge(market_values[[*merge_cols, *value_cols]], on=merge_cols, how="left")

    national_context = raw_data.get("national_team_context", pd.DataFrame())
    if not national_context.empty and "country" in national_context.columns:
        context_cols = [
            col
            for col in [
                "national_team_strength",
                "group_difficulty_score",
                "draw_context_score",
                "group",
                "group_opponents",
            ]
            if col in national_context.columns
        ]
        df = df.drop(columns=[col for col in context_cols if col in df.columns], errors="ignore")
        df = df.merge(national_context[["country", *context_cols]], on="country", how="left")

    return df


def build_processed_player_dataset(raw_data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Build the dashboard dataset from real local CSV inputs only."""

    raw_data = load_raw_real_data() if raw_data is None else raw_data
    validation = validate_raw_data(raw_data)
    if validation["errors"]:
        raise ValueError(" ; ".join(validation["errors"]))

    from preprocessing import prepare_player_data

    player_pool = _merge_optional_sources(raw_data["current_player_pool"], raw_data)
    tournament = load_aggregated_tournament_data()
    if not tournament.empty:
        player_pool = player_pool.merge(tournament, on="player_name", how="left")
    return prepare_player_data(player_pool)


def save_processed_data(df: pd.DataFrame, path: Path | str = PROCESSED_DASHBOARD_PATH) -> Path:
    """Save dashboard-ready processed data and refresh manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    manifest = {
        "last_refresh_utc": datetime.now(timezone.utc).isoformat(),
        "processed_path": str(path),
        "rows": int(len(df)),
        "missing_market_values": int(pd.to_numeric(df.get("market_value_eur"), errors="coerce").isna().sum())
        if "market_value_eur" in df.columns
        else None,
        "expected_impact_available": int(df["pre_tournament_expected_impact_score"].notna().sum())
        if "pre_tournament_expected_impact_score" in df.columns
        else 0,
        "breakout_candidate_score_available": int(df["breakout_candidate_score"].notna().sum())
        if "breakout_candidate_score" in df.columns
        else 0,
        "world_cup_debutant_values_available": int(df["is_world_cup_debutant"].notna().sum())
        if "is_world_cup_debutant" in df.columns
        else 0,
        "previous_world_cup_minutes_available": int(df["previous_world_cup_minutes"].notna().sum())
        if "previous_world_cup_minutes" in df.columns
        else 0,
        "actual_tournament_impact_available": int(df["actual_tournament_impact_score"].notna().sum())
        if "actual_tournament_impact_score" in df.columns
        else 0,
        "refresh_source": "raw files",
        "data_status": check_data_availability(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def refresh_dashboard_data() -> dict[str, Any]:
    """Refresh processed data from local real CSV inputs."""

    raw_data = load_raw_real_data()
    validation = validate_raw_data(raw_data)
    if validation["errors"]:
        return {
            "success": False,
            "message": "Refresh failed because required real data is missing or invalid.",
            "warnings": validation["warnings"],
            "errors": validation["errors"],
            "data_status": check_data_availability(),
        }
    processed = build_processed_player_dataset(raw_data)
    path = save_processed_data(processed)
    return {
        "success": True,
        "message": f"Refreshed {len(processed):,} players from real CSV inputs.",
        "warnings": validation["warnings"],
        "errors": [],
        "path": str(path),
        "data_status": check_data_availability(),
    }


def load_processed_data(path: Path | str = PROCESSED_DASHBOARD_PATH) -> pd.DataFrame:
    """Load processed dashboard data if available."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def get_last_refresh_timestamp() -> str:
    """Return the latest refresh timestamp from the manifest if available."""

    manifest = get_refresh_manifest()
    return str(manifest.get("last_refresh_utc", "")) if manifest else ""


def get_refresh_manifest() -> dict[str, Any]:
    """Return the latest refresh manifest if available."""

    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
