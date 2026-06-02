"""Cleaning and preprocessing pipeline for player data."""

from __future__ import annotations

import pandas as pd

from feature_engineering import (
    calculate_all_features,
    calculate_expected_impact_scores,
    calculate_value_opportunity_score,
)
from player_profiles import calculate_profile_scores
from tournament_impact import calculate_actual_tournament_impact_score


REQUIRED_COLUMNS = [
    "player_name",
    "country",
    "club",
    "league",
    "age",
    "position",
    "minutes",
    "goals",
    "assists",
    "shots",
    "key_passes",
    "progressive_passes",
    "progressive_carries",
    "successful_dribbles",
    "tackles",
    "interceptions",
    "aerial_duels_won",
    "pass_completion_pct",
    "turnovers",
    "club_level_score",
    "recent_form_score",
    "expected_minutes_score",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "market_value_eur",
    "national_team_strength",
    "draw_context_score",
    "group_difficulty_score",
    "injury_availability_score",
    "tactical_fit_score",
    "final_squad_selection_score",
    "caps",
    "international_goals",
    "previous_world_cup_minutes",
    "previous_world_cup_matches",
    "previous_world_cup_impact_score",
    "senior_national_team_caps",
    "major_tournament_experience",
]

OPTIONAL_BOOLEAN_COLUMNS = [
    "is_world_cup_debutant",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if required columns are absent."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _coerce_optional_boolean(series: pd.Series) -> pd.Series:
    """Coerce common real CSV boolean representations while preserving missing values."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")
    normalised = series.astype(str).str.strip().str.lower()
    mapped = normalised.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
    )
    mapped[series.isna()] = pd.NA
    return mapped.astype("boolean")


def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types and clip ranges while preserving missing real values."""

    validate_columns(df)
    df = df.copy()
    for col in OPTIONAL_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for col in OPTIONAL_BOOLEAN_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    text_cols = [
        "player_name",
        "country",
        "club",
        "league",
        "position",
        "group",
        "group_opponents",
        "injury_status",
        "injury_source_url",
        "market_value_source",
        "model_expected_impact_source",
        "expected_impact_source",
        "model_warning",
        "performance_outcome",
        "recommendation",
        "recommendation_reason",
        "risk_band",
        "risk_reason",
        "latest_market_value_date",
        "salimt_latest_market_value_date",
        "latest_value_competition_id",
        "tm_current_club_name",
        "tm_current_club_domestic_competition_id",
        "tm_sub_position",
        "tm_foot",
        "tm_national_team_name",
        "tm_confederation",
        "tm_url",
        "tm_image_url",
        "data_source",
        "data_refresh_date",
    ]
    text_cols = [col for col in text_cols if col in df.columns]
    for col in text_cols:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    for col in OPTIONAL_BOOLEAN_COLUMNS:
        df[col] = _coerce_optional_boolean(df[col])

    numeric_cols = [col for col in df.columns if col not in text_cols and col not in OPTIONAL_BOOLEAN_COLUMNS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "caps" in df.columns:
        senior_caps = pd.to_numeric(df["senior_national_team_caps"], errors="coerce")
        caps = pd.to_numeric(df["caps"], errors="coerce")
        df["senior_national_team_caps"] = senior_caps.where(senior_caps.notna(), caps)

    df["age"] = df["age"].clip(16, 45).round()
    df["minutes"] = df["minutes"].clip(0, 5000).round()
    df["market_value_eur"] = df["market_value_eur"].clip(lower=0).round()
    df["pass_completion_pct"] = df["pass_completion_pct"].clip(0, 100)

    score_cols = [
        "national_team_strength",
        "club_level_score",
        "recent_form_score",
        "expected_minutes_score",
    ]
    for col in score_cols:
        df[col] = df[col].clip(0, 100)
    df["previous_world_cup_impact_score"] = df["previous_world_cup_impact_score"].clip(0, 100)

    return df


def apply_expected_impact_model(df: pd.DataFrame) -> pd.DataFrame:
    """Apply historical expected-impact model when available, otherwise baseline."""

    from model import predict_expected_impact, train_expected_impact_model

    df = df.copy()
    model_summary = train_expected_impact_model()
    df["pre_tournament_expected_impact_score"] = predict_expected_impact(df, model_summary)
    if model_summary.get("model_available"):
        df["model_expected_impact_source"] = "historical_model"
        df["expected_impact_source"] = "supervised_historical_model"
        df["model_warning"] = ""
    else:
        baseline_available = df["baseline_expected_impact_score"].notna()
        df["model_expected_impact_source"] = "disabled"
        df["expected_impact_source"] = "unavailable_missing_features"
        df.loc[baseline_available, "expected_impact_source"] = "transparent_baseline_fallback"
        df["model_warning"] = str(model_summary.get("warning", ""))
    df["historical_model_available"] = bool(model_summary.get("model_available", False))
    df["model_status"] = str(model_summary.get("model_status", "unknown"))
    return df


def finalise_player_outputs(
    df: pd.DataFrame,
    strategy: str = "Balanced club",
    custom_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Calculate expected impact, actual impact validation, and opportunity fields."""

    df = calculate_expected_impact_scores(df)
    df = apply_expected_impact_model(df)
    df = calculate_actual_tournament_impact_score(df)
    df = calculate_value_opportunity_score(df, strategy=strategy, custom_weights=custom_weights)
    return df.sort_values("value_opportunity_score", ascending=False).reset_index(drop=True)


def prepare_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, engineer features, assign profiles, and calculate model outputs."""

    df = clean_player_data(df)
    df = calculate_all_features(df)
    df = calculate_profile_scores(df)
    return finalise_player_outputs(df)
