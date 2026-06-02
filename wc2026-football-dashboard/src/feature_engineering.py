"""Feature engineering for football scouting analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


VOLUME_METRICS = [
    "goals",
    "non_penalty_goals",
    "assists",
    "shots",
    "key_passes",
    "progressive_passes",
    "progressive_carries",
    "passes_into_final_third",
    "carries_into_final_third",
    "successful_dribbles",
    "tackles",
    "interceptions",
    "aerial_duels_won",
    "turnovers",
    "through_balls",
    "crosses_into_box",
    "shot_creating_actions_proxy",
    "shots_in_box",
    "touches_in_box",
    "pressures",
    "recoveries",
    "long_passes",
    "saves",
    "clean_sheets",
    "aerial_threat",
]


CORE_PERCENTILE_METRICS = [
    *VOLUME_METRICS,
    "pass_completion_pct",
    "xg_proxy",
    "xa_proxy",
]


def clip_score(values: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Keep score-like values inside a 0-100 range."""

    return np.clip(values, 0, 100)


def add_rate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-90 rates for volume metrics."""

    df = df.copy()
    minutes = df["minutes"].replace(0, np.nan)
    per90_factor = minutes / 90
    for metric in VOLUME_METRICS:
        if metric in df.columns:
            df[f"{metric}_per90"] = (df[metric] / per90_factor).replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def add_proxy_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Create simple xG/xA-style proxy metrics from box threat and creation signals."""

    df = df.copy()
    df["xg_proxy"] = (
        df.get("shots_in_box", 0) * 0.13
        + (df.get("shots", 0) - df.get("shots_in_box", 0)).clip(lower=0) * 0.035
        + df.get("touches_in_box", 0) * 0.012
    ).round(2)
    df["xa_proxy"] = (
        df.get("key_passes", 0) * 0.055
        + df.get("through_balls", 0) * 0.09
        + df.get("crosses_into_box", 0) * 0.035
        + df.get("shot_creating_actions_proxy", 0) * 0.025
    ).round(2)
    return df


def calculate_percentiles(
    df: pd.DataFrame,
    metric_columns: list[str],
    group_col: str = "position",
) -> pd.DataFrame:
    """Calculate percentile ranks within position groups.

    Volume metrics are ranked on per-90 values when available; percentage and
    score columns are ranked directly. This keeps forwards, defenders,
    midfielders, and goalkeepers on fairer positional baselines.
    """

    df = df.copy()
    for metric in metric_columns:
        if metric not in df.columns and f"{metric}_per90" not in df.columns:
            continue
        source_col = f"{metric}_per90" if f"{metric}_per90" in df.columns else metric
        percentile_col = f"{metric}_percentile"
        df[percentile_col] = (
            df.groupby(group_col)[source_col]
            .rank(method="average", pct=True, ascending=True)
            .mul(100)
            .round(1)
        )
    return df


def _weighted_score(df: pd.DataFrame, weights: dict[str, float], default: float = 50) -> pd.Series:
    """Calculate a weighted score from percentile columns."""

    score = pd.Series(0.0, index=df.index)
    total_weight = 0.0
    for metric, weight in weights.items():
        col = f"{metric}_percentile"
        if col in df.columns:
            values = df[col].fillna(default)
        elif metric in df.columns:
            values = df[metric].fillna(default)
        else:
            values = pd.Series(default, index=df.index)
        score = score + values * weight
        total_weight += abs(weight)
    if total_weight == 0:
        return pd.Series(default, index=df.index)
    return clip_score(score / total_weight)


def calculate_attacking_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["attacking_score"] = _weighted_score(
        df,
        {
            "non_penalty_goals": 0.30,
            "xg_proxy": 0.25,
            "shots_in_box": 0.20,
            "touches_in_box": 0.10,
            "shots": 0.15,
        },
    ).round(1)
    return df


def calculate_creativity_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["creativity_score"] = _weighted_score(
        df,
        {
            "assists": 0.20,
            "key_passes": 0.25,
            "xa_proxy": 0.25,
            "through_balls": 0.15,
            "shot_creating_actions_proxy": 0.15,
        },
    ).round(1)
    return df


def calculate_progression_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["progression_score"] = _weighted_score(
        df,
        {
            "progressive_passes": 0.30,
            "progressive_carries": 0.25,
            "passes_into_final_third": 0.20,
            "carries_into_final_third": 0.15,
            "successful_dribbles": 0.10,
        },
    ).round(1)
    return df


def calculate_defensive_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["defensive_score"] = _weighted_score(
        df,
        {
            "tackles": 0.25,
            "interceptions": 0.25,
            "recoveries": 0.20,
            "aerial_duels_won": 0.15,
            "pressures": 0.15,
        },
    ).round(1)
    return df


def calculate_possession_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    turnover_col = "turnovers_percentile"
    turnover_control = 100 - df[turnover_col] if turnover_col in df.columns else 50
    df["turnover_control_score"] = clip_score(turnover_control)
    df["possession_score"] = (
        df.get("pass_completion_pct_percentile", 50) * 0.35
        + df.get("progressive_passes_percentile", 50) * 0.15
        + df.get("long_passes_percentile", 50) * 0.10
        + df["turnover_control_score"] * 0.25
        + df.get("successful_dribbles_percentile", 50) * 0.15
    ).round(1)
    return df


def calculate_overall_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate a role-sensitive overall profile score."""

    df = df.copy()
    weights_by_position = {
        "Goalkeeper": {
            "attacking_score": 0.02,
            "creativity_score": 0.06,
            "progression_score": 0.17,
            "defensive_score": 0.25,
            "possession_score": 0.50,
        },
        "Centre-back": {
            "attacking_score": 0.05,
            "creativity_score": 0.05,
            "progression_score": 0.22,
            "defensive_score": 0.42,
            "possession_score": 0.26,
        },
        "Full-back": {
            "attacking_score": 0.10,
            "creativity_score": 0.18,
            "progression_score": 0.25,
            "defensive_score": 0.27,
            "possession_score": 0.20,
        },
        "Defensive midfielder": {
            "attacking_score": 0.06,
            "creativity_score": 0.12,
            "progression_score": 0.25,
            "defensive_score": 0.34,
            "possession_score": 0.23,
        },
        "Central midfielder": {
            "attacking_score": 0.10,
            "creativity_score": 0.22,
            "progression_score": 0.28,
            "defensive_score": 0.18,
            "possession_score": 0.22,
        },
        "Attacking midfielder": {
            "attacking_score": 0.24,
            "creativity_score": 0.32,
            "progression_score": 0.20,
            "defensive_score": 0.08,
            "possession_score": 0.16,
        },
        "Winger": {
            "attacking_score": 0.27,
            "creativity_score": 0.24,
            "progression_score": 0.24,
            "defensive_score": 0.07,
            "possession_score": 0.18,
        },
        "Forward": {
            "attacking_score": 0.48,
            "creativity_score": 0.14,
            "progression_score": 0.13,
            "defensive_score": 0.10,
            "possession_score": 0.15,
        },
    }

    overall = pd.Series(0.0, index=df.index)
    for position, weights in weights_by_position.items():
        mask = df["position"] == position
        if not mask.any():
            continue
        position_score = pd.Series(0.0, index=df.index)
        for score_col, weight in weights.items():
            position_score = position_score + df[score_col] * weight
        overall.loc[mask] = position_score.loc[mask]
    df["overall_score"] = clip_score(overall).round(1)
    return df


def calculate_age_curve_score(df: pd.DataFrame) -> pd.DataFrame:
    """Score players by broad age curve fit for their position."""

    df = df.copy()
    peak_age = {
        "Goalkeeper": 30,
        "Centre-back": 28,
        "Full-back": 25,
        "Defensive midfielder": 27,
        "Central midfielder": 26,
        "Attacking midfielder": 25,
        "Winger": 24,
        "Forward": 25,
    }
    peaks = df["position"].map(peak_age).fillna(26)
    age_gap = (df["age"] - peaks).abs()
    df["age_curve_score"] = clip_score(100 - age_gap * 9).round(1)
    return df


def calculate_success_score(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate heuristic success score and probability-like output.

    This is a scouting-support heuristic for the MVP, not a betting model or
    claim of true tournament outcome probability.
    """

    df = df.copy()
    for col in [
        "overall_score",
        "national_team_strength",
        "expected_minutes_score",
        "best_profile_score",
        "age_curve_score",
        "club_level_score",
        "recent_form_score",
    ]:
        if col not in df.columns:
            df[col] = 50

    df["current_performance_score"] = df["overall_score"]
    df["role_fit_score"] = df["best_profile_score"]
    df["success_score"] = (
        0.25 * df["current_performance_score"]
        + 0.20 * df["national_team_strength"]
        + 0.15 * df["expected_minutes_score"]
        + 0.15 * df["role_fit_score"]
        + 0.10 * df["age_curve_score"]
        + 0.10 * df["club_level_score"]
        + 0.05 * df["recent_form_score"]
    ).round(1)

    probability = 100 / (1 + np.exp(-(df["success_score"] - 58) / 10))
    df["success_probability"] = clip_score(probability).round(1)
    return df


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the complete feature engineering pipeline."""

    df = add_proxy_metrics(df)
    df = add_rate_metrics(df)
    df = calculate_percentiles(df, CORE_PERCENTILE_METRICS)
    df = calculate_attacking_score(df)
    df = calculate_creativity_score(df)
    df = calculate_progression_score(df)
    df = calculate_defensive_score(df)
    df = calculate_possession_score(df)
    df = calculate_overall_score(df)
    df = calculate_age_curve_score(df)
    return df

