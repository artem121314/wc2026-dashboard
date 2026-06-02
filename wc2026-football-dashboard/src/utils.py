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

    value = pd.to_numeric(value, errors="coerce")
    if pd.isna(value):
        return "Unavailable"
    value = float(value)
    if value >= 1_000_000:
        return f"EUR {value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"EUR {value / 1_000:.0f}k"
    return f"EUR {value:.0f}"


def score_badge(score: float) -> str:
    if pd.isna(score):
        return "Unavailable"
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
            value = pd.to_numeric(player[metric], errors="coerce")
            if pd.notna(value):
                values.append((label, float(value)))
    values.sort(key=lambda item: item[1], reverse=True)
    if not values:
        return ["available real-data inputs"]
    return [label for label, _ in values[:limit]]


def get_risks(player: pd.Series, limit: int = 3) -> list[str]:
    """Return a short list of weaker/risk areas."""

    def numeric(value: object, default: float) -> float:
        parsed = pd.to_numeric(value, errors="coerce")
        return default if pd.isna(parsed) else float(parsed)

    candidates = [
        ("playing-time security", numeric(player.get("expected_minutes_score", 50), 50)),
        ("age-curve fit", numeric(player.get("age_curve_score", 50), 50)),
        ("recent form", numeric(player.get("recent_form_score", 50), 50)),
        ("possession risk", numeric(player.get("turnover_control_score", 50), 50)),
        ("club-level context", numeric(player.get("club_level_score", 50), 50)),
        ("role-fit certainty", numeric(player.get("role_fit_score", 50), 50)),
        ("injury availability", numeric(player.get("availability_score", player.get("injury_availability_score", 100)), 100)),
        ("market-value efficiency", numeric(player.get("value_efficiency_score", 50), 50)),
    ]
    candidates.sort(key=lambda item: item[1])
    return [label for label, score in candidates[:limit] if score < 62] or ["no major red flags in the current inputs"]


def scouting_summary(player: pd.Series) -> str:
    """Generate a short scouting-style player summary."""

    strengths = get_strengths(player)
    strengths_text = ", ".join(strengths[:-1]) + f" and {strengths[-1]}" if len(strengths) > 1 else strengths[0]
    expected = player.get("pre_tournament_expected_impact_score", pd.NA)
    opportunity = player.get("value_opportunity_score", pd.NA)
    expected_text = "unavailable" if pd.isna(expected) else f"{expected:.0f}/100"
    opportunity_text = "unavailable" if pd.isna(opportunity) else f"{opportunity:.0f}/100"
    return (
        f"{player['player_name']} profiles as a {player['best_profile']}. "
        f"His strongest areas are {strengths_text}. The dashboard estimates a "
        f"{expected_text} pre-tournament World Cup impact score "
        f"and a {opportunity_text} value opportunity score."
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
        "pre_tournament_expected_impact_score",
        "value_opportunity_score",
    ]
    distance = 0
    for col in score_cols:
        distance = distance + (pool[col] - float(player[col])).abs()
    profile_penalty = (pool["best_profile"] != player["best_profile"]).astype(int) * 18
    pool["similarity_distance"] = distance + profile_penalty
    return pool.sort_values("similarity_distance").head(n)
