"""Actual World Cup tournament impact scoring.

The functions in this module are intentionally data-source agnostic. They use
box-score style tournament fields when available and leave outcome fields
pending when the tournament feed has not arrived yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


TOURNAMENT_FIELD_ALIASES = {
    "tournament_minutes": ["tournament_minutes", "actual_wc_minutes"],
    "tournament_goals": ["tournament_goals", "actual_wc_goals"],
    "tournament_assists": ["tournament_assists", "actual_wc_assists"],
}

TOURNAMENT_NUMERIC_FIELDS = [
    "tournament_minutes",
    "tournament_starts",
    "tournament_goals",
    "tournament_assists",
    "tournament_shots",
    "tournament_key_passes",
    "tournament_tackles",
    "tournament_interceptions",
    "tournament_saves",
    "tournament_clean_sheets",
    "tournament_match_rating",
]


def _coalesce_tournament_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Create canonical tournament columns from known source aliases."""

    df = df.copy()
    original_columns = set(df.columns)
    for target, aliases in TOURNAMENT_FIELD_ALIASES.items():
        if target not in df.columns:
            df[target] = pd.Series(np.nan, index=df.index, dtype=float)
        for alias in aliases:
            if alias in original_columns:
                current = pd.to_numeric(df[target], errors="coerce")
                source = pd.to_numeric(df[alias], errors="coerce")
                df[target] = current.where(current.notna(), source)

    for col in TOURNAMENT_NUMERIC_FIELDS:
        if col not in df.columns:
            df[col] = np.nan if col == "tournament_match_rating" else 0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col != "tournament_match_rating":
            df[col] = df[col].fillna(0)

    return df


def calculate_raw_tournament_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate raw tournament impact from available box-score fields."""

    df = _coalesce_tournament_fields(df)
    minutes = df["tournament_minutes"].clip(lower=0)
    per90 = (minutes / 90).replace(0, np.nan)
    starts = df["tournament_starts"].fillna(0)
    match_rating_score = ((df["tournament_match_rating"] - 6.0) / 2.5 * 100).clip(0, 100)

    rates = {
        "goals": (df["tournament_goals"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "assists": (df["tournament_assists"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "shots": (df["tournament_shots"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "key_passes": (df["tournament_key_passes"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "tackles": (df["tournament_tackles"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "interceptions": (df["tournament_interceptions"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "saves": (df["tournament_saves"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
        "clean_sheets": (df["tournament_clean_sheets"] / per90).replace([np.inf, -np.inf], np.nan).fillna(0),
    }
    minutes_score = (minutes / 450 * 100).clip(0, 100)
    starts_score = (starts / 5 * 100).clip(0, 100)

    raw = pd.Series(0.0, index=df.index)
    position = df.get("position", pd.Series("", index=df.index)).fillna("")
    goalkeeper = position.eq("Goalkeeper")
    defender = position.isin(["Centre-back", "Full-back"])
    midfielder = position.isin(["Defensive midfielder", "Central midfielder", "Attacking midfielder"])
    forward = position.isin(["Winger", "Forward"])

    raw.loc[goalkeeper] = (
        minutes_score.loc[goalkeeper] * 0.25
        + starts_score.loc[goalkeeper] * 0.15
        + rates["saves"].rank(pct=True).mul(100).loc[goalkeeper] * 0.25
        + rates["clean_sheets"].rank(pct=True).mul(100).loc[goalkeeper] * 0.20
        + match_rating_score.fillna(50).loc[goalkeeper] * 0.15
    )
    raw.loc[defender] = (
        minutes_score.loc[defender] * 0.25
        + starts_score.loc[defender] * 0.15
        + rates["tackles"].rank(pct=True).mul(100).loc[defender] * 0.18
        + rates["interceptions"].rank(pct=True).mul(100).loc[defender] * 0.18
        + rates["clean_sheets"].rank(pct=True).mul(100).loc[defender] * 0.12
        + match_rating_score.fillna(50).loc[defender] * 0.12
    )
    raw.loc[midfielder] = (
        minutes_score.loc[midfielder] * 0.22
        + starts_score.loc[midfielder] * 0.12
        + rates["assists"].rank(pct=True).mul(100).loc[midfielder] * 0.16
        + rates["key_passes"].rank(pct=True).mul(100).loc[midfielder] * 0.18
        + rates["tackles"].rank(pct=True).mul(100).loc[midfielder] * 0.12
        + rates["interceptions"].rank(pct=True).mul(100).loc[midfielder] * 0.08
        + match_rating_score.fillna(50).loc[midfielder] * 0.12
    )
    raw.loc[forward] = (
        minutes_score.loc[forward] * 0.20
        + starts_score.loc[forward] * 0.10
        + rates["goals"].rank(pct=True).mul(100).loc[forward] * 0.28
        + rates["assists"].rank(pct=True).mul(100).loc[forward] * 0.16
        + rates["shots"].rank(pct=True).mul(100).loc[forward] * 0.14
        + rates["key_passes"].rank(pct=True).mul(100).loc[forward] * 0.06
        + match_rating_score.fillna(50).loc[forward] * 0.06
    )

    fallback = ~(goalkeeper | defender | midfielder | forward)
    raw.loc[fallback] = (
        minutes_score.loc[fallback] * 0.35
        + starts_score.loc[fallback] * 0.20
        + rates["goals"].rank(pct=True).mul(100).loc[fallback] * 0.15
        + rates["assists"].rank(pct=True).mul(100).loc[fallback] * 0.15
        + match_rating_score.fillna(50).loc[fallback] * 0.15
    )

    has_tournament_data = (
        (minutes > 0)
        | (starts > 0)
        | (df["tournament_goals"] > 0)
        | (df["tournament_assists"] > 0)
        | df["tournament_match_rating"].notna()
    )
    df["has_tournament_match_data"] = has_tournament_data
    df["raw_tournament_impact_score"] = raw.clip(0, 100).where(has_tournament_data).round(1)
    return df


def calculate_position_adjusted_impact(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw impact into a within-position 0-100 score."""

    df = df.copy()
    if "raw_tournament_impact_score" not in df.columns:
        df = calculate_raw_tournament_impact(df)

    adjusted = pd.Series(np.nan, index=df.index, dtype=float)
    valid = df["raw_tournament_impact_score"].notna()
    if valid.any():
        adjusted.loc[valid] = (
            df.loc[valid]
            .groupby("position")["raw_tournament_impact_score"]
            .rank(method="average", pct=True)
            .mul(100)
            .round(1)
        )
    df["position_adjusted_tournament_impact_score"] = adjusted
    return df


def calculate_actual_tournament_impact_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add actual impact, performance delta and validation outcome fields."""

    df = calculate_position_adjusted_impact(calculate_raw_tournament_impact(df))
    df["actual_tournament_impact_score"] = df["position_adjusted_tournament_impact_score"]
    if "pre_tournament_expected_impact_score" in df.columns:
        expected = pd.to_numeric(df["pre_tournament_expected_impact_score"], errors="coerce")
    else:
        expected = pd.Series(np.nan, index=df.index)
    actual = pd.to_numeric(df["actual_tournament_impact_score"], errors="coerce")
    df["performance_delta"] = (actual - expected).round(1)
    df["performance_outcome"] = np.select(
        [
            actual.isna(),
            df["performance_delta"] >= 10,
            df["performance_delta"] <= -10,
        ],
        ["Pending", "Overperformed", "Underperformed"],
        default="Met expectations",
    )
    return df
