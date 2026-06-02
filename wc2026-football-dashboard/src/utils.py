"""Utility helpers for display and scouting summaries."""

from __future__ import annotations

import pandas as pd


STRENGTH_METRICS = {
    "attacking_score": "attacking output",
    "creativity_score": "chance creation",
    "progression_score": "ball progression",
    "defensive_score": "defensive activity",
    "possession_score": "possession security",
    "pass_completion_pct_percentile": "passing reliability",
    "progressive_passes_percentile": "progressive passing",
    "successful_dribbles_percentile": "one-v-one carrying",
    "key_passes_percentile": "final-third passing",
    "xg_proxy_percentile": "shot quality",
}


def format_currency(value: float) -> str:
    """Format market value in a compact EUR style."""

    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}k"
    return f"EUR {value:.0f}"


def score_badge(score: float) -> str:
    if score >= 80:
        return "Elite"
    if score >= 68:
        return "Strong"
    if score >= 55:
        return "Solid"
    return "Watchlist"


def get_strengths(player: pd.Series, limit: int = 3) -> list[str]:
    """Return a short list of strongest areas."""

    values = []
    for metric, label in STRENGTH_METRICS.items():
        if metric in player:
            values.append((label, float(player[metric])))
    values.sort(key=lambda item: item[1], reverse=True)
    return [label for label, _ in values[:limit]]


def get_risks(player: pd.Series, limit: int = 3) -> list[str]:
    """Return a short list of weaker/risk areas."""

    candidates = [
        ("playing-time security", float(player.get("expected_minutes_score", 50))),
        ("age-curve fit", float(player.get("age_curve_score", 50))),
        ("recent form", float(player.get("recent_form_score", 50))),
        ("possession risk", float(player.get("turnover_control_score", 50))),
        ("club-level context", float(player.get("club_level_score", 50))),
        ("role-fit certainty", float(player.get("role_fit_score", 50))),
    ]
    candidates.sort(key=lambda item: item[1])
    return [label for label, score in candidates[:limit] if score < 62] or ["no major red flags in the sample inputs"]


def scouting_summary(player: pd.Series) -> str:
    """Generate a short scouting-style player summary."""

    strengths = get_strengths(player)
    strengths_text = ", ".join(strengths[:-1]) + f" and {strengths[-1]}" if len(strengths) > 1 else strengths[0]
    return (
        f"{player['player_name']} profiles as a {player['best_profile']}. "
        f"His strongest areas are {strengths_text}. Based on current indicators, age, playing time "
        f"and recent performance, the model estimates a {player['success_probability']:.0f}% probability "
        "of a successful World Cup."
    )


def similar_players(df: pd.DataFrame, player: pd.Series, n: int = 5) -> pd.DataFrame:
    """Find nearby players by position, profile, and core score similarity."""

    pool = df[
        (df["position"] == player["position"])
        & (df["player_name"] != player["player_name"])
    ].copy()
    if pool.empty:
        pool = df[df["player_name"] != player["player_name"]].copy()

    score_cols = [
        "attacking_score",
        "creativity_score",
        "progression_score",
        "defensive_score",
        "possession_score",
        "success_probability",
    ]
    distance = 0
    for col in score_cols:
        distance = distance + (pool[col] - float(player[col])).abs()
    profile_penalty = (pool["best_profile"] != player["best_profile"]).astype(int) * 18
    pool["similarity_distance"] = distance + profile_penalty
    return pool.sort_values("similarity_distance").head(n)

