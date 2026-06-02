"""Cleaning and preprocessing pipeline for player data."""

from __future__ import annotations

import pandas as pd

from feature_engineering import calculate_all_features, calculate_success_score
from player_profiles import calculate_profile_scores


REQUIRED_COLUMNS = [
    "player_name",
    "country",
    "club",
    "league",
    "age",
    "position",
    "market_value_eur",
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
    "national_team_strength",
    "club_level_score",
    "recent_form_score",
    "expected_minutes_score",
]


def validate_columns(df: pd.DataFrame) -> None:
    """Raise a clear error if required columns are absent."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def clean_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic type coercion, missing value handling, and range clipping."""

    validate_columns(df)
    df = df.copy()
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

    numeric_cols = [col for col in df.columns if col not in text_cols]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if not df[col].isna().all() else 0)

    df["age"] = df["age"].clip(16, 45).round().astype(int)
    df["minutes"] = df["minutes"].clip(0, 5000).round().astype(int)
    df["market_value_eur"] = df["market_value_eur"].clip(lower=0).round().astype(int)
    df["pass_completion_pct"] = df["pass_completion_pct"].clip(0, 100)

    score_cols = [
        "national_team_strength",
        "club_level_score",
        "recent_form_score",
        "expected_minutes_score",
    ]
    for col in score_cols:
        df[col] = df[col].clip(0, 100)

    return df


def prepare_player_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data, engineer scores, assign profiles, and calculate success score."""

    df = clean_player_data(df)
    df = calculate_all_features(df)
    df = calculate_profile_scores(df)
    df = calculate_success_score(df)
    return df.sort_values("success_probability", ascending=False).reset_index(drop=True)
