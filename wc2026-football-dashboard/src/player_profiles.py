"""Football role/profile scoring definitions."""

from __future__ import annotations

import pandas as pd


PLAYER_PROFILES = {
    "Ball-progressing midfielder": {
        "progressive_passes_percentile": 0.30,
        "progressive_carries_percentile": 0.25,
        "passes_into_final_third_percentile": 0.20,
        "successful_dribbles_percentile": 0.10,
        "pass_completion_pct_percentile": 0.10,
        "turnovers_percentile": -0.05,
    },
    "Chance creator": {
        "key_passes_percentile": 0.30,
        "xa_proxy_percentile": 0.25,
        "shot_creating_actions_proxy_percentile": 0.20,
        "through_balls_percentile": 0.15,
        "crosses_into_box_percentile": 0.10,
    },
    "Box threat forward": {
        "non_penalty_goals_percentile": 0.30,
        "xg_proxy_percentile": 0.25,
        "shots_in_box_percentile": 0.20,
        "touches_in_box_percentile": 0.15,
        "aerial_threat_percentile": 0.10,
    },
    "High-volume winger": {
        "successful_dribbles_percentile": 0.28,
        "progressive_carries_percentile": 0.24,
        "crosses_into_box_percentile": 0.18,
        "key_passes_percentile": 0.14,
        "shots_percentile": 0.11,
        "turnovers_percentile": -0.05,
    },
    "Defensive midfielder": {
        "tackles_percentile": 0.25,
        "interceptions_percentile": 0.25,
        "recoveries_percentile": 0.18,
        "progressive_passes_percentile": 0.14,
        "pass_completion_pct_percentile": 0.13,
        "turnovers_percentile": -0.05,
    },
    "Modern full-back": {
        "progressive_carries_percentile": 0.22,
        "crosses_into_box_percentile": 0.20,
        "key_passes_percentile": 0.17,
        "tackles_percentile": 0.16,
        "interceptions_percentile": 0.13,
        "pass_completion_pct_percentile": 0.12,
    },
    "Ball-playing centre-back": {
        "progressive_passes_percentile": 0.28,
        "long_passes_percentile": 0.20,
        "pass_completion_pct_percentile": 0.18,
        "aerial_duels_won_percentile": 0.14,
        "interceptions_percentile": 0.12,
        "turnovers_percentile": -0.08,
    },
    "High-pressing forward": {
        "pressures_percentile": 0.30,
        "tackles_percentile": 0.16,
        "non_penalty_goals_percentile": 0.18,
        "shots_percentile": 0.14,
        "touches_in_box_percentile": 0.12,
        "recoveries_percentile": 0.10,
    },
    "Goalkeeper distributor": {
        "long_passes_percentile": 0.28,
        "pass_completion_pct_percentile": 0.22,
        "progressive_passes_percentile": 0.18,
        "clean_sheets_percentile": 0.14,
        "saves_percentile": 0.10,
        "turnovers_percentile": -0.08,
    },
}


PROFILE_ELIGIBILITY = {
    "Ball-progressing midfielder": ["Defensive midfielder", "Central midfielder", "Attacking midfielder"],
    "Chance creator": ["Central midfielder", "Attacking midfielder", "Winger", "Full-back"],
    "Box threat forward": ["Forward", "Winger", "Attacking midfielder"],
    "High-volume winger": ["Winger", "Attacking midfielder", "Full-back"],
    "Defensive midfielder": ["Defensive midfielder", "Central midfielder"],
    "Modern full-back": ["Full-back"],
    "Ball-playing centre-back": ["Centre-back"],
    "High-pressing forward": ["Forward", "Winger", "Attacking midfielder"],
    "Goalkeeper distributor": ["Goalkeeper"],
}


def _profile_score_from_row(player_row: pd.Series, profile_weights: dict[str, float]) -> float:
    score = 0.0
    total_weight = 0.0
    for metric, weight in profile_weights.items():
        value = float(player_row.get(metric, 50))
        if weight < 0:
            value = 100 - value
        score += value * abs(weight)
        total_weight += abs(weight)
    if total_weight == 0:
        return 50.0
    return round(max(0, min(100, score / total_weight)), 1)


def assign_best_profile(player_row: pd.Series) -> str:
    """Return the highest scoring predefined football profile for a player."""

    player_position = player_row.get("position")
    scores = {
        profile: _profile_score_from_row(player_row, weights)
        for profile, weights in PLAYER_PROFILES.items()
        if player_position in PROFILE_ELIGIBILITY.get(profile, [])
    }
    if not scores:
        scores = {
            profile: _profile_score_from_row(player_row, weights)
            for profile, weights in PLAYER_PROFILES.items()
        }
    return max(scores, key=scores.get)


def calculate_profile_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add profile score columns plus best profile labels."""

    df = df.copy()
    for profile, weights in PLAYER_PROFILES.items():
        safe_name = profile.lower().replace("-", "").replace(" ", "_")
        df[f"{safe_name}_score"] = df.apply(lambda row: _profile_score_from_row(row, weights), axis=1)

    profile_score_cols = {
        profile: profile.lower().replace("-", "").replace(" ", "_") + "_score"
        for profile in PLAYER_PROFILES
    }
    df["best_profile"] = df.apply(assign_best_profile, axis=1)
    df["best_profile_score"] = df.apply(
        lambda row: row[profile_score_cols[row["best_profile"]]],
        axis=1,
    ).round(1)
    return df


def get_profile_score_column(profile: str) -> str:
    """Return the dataframe column name for a profile score."""

    return profile.lower().replace("-", "").replace(" ", "_") + "_score"


def get_eligible_positions(profile: str) -> list[str]:
    """Return positions eligible for a profile."""

    return PROFILE_ELIGIBILITY.get(profile, [])
