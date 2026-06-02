"""Real World Cup match-data ingestion from local CSV exports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_sources import TOURNAMENT_MATCH_DATA_PATH


TOURNAMENT_MATCH_COLUMNS = [
    "player_name",
    "match_id",
    "date",
    "opponent",
    "stage",
    "minutes",
    "goals",
    "assists",
    "shots",
    "key_passes",
    "tackles",
    "interceptions",
    "saves",
    "clean_sheets",
    "match_rating",
]


def load_tournament_match_data(path: Path | str = TOURNAMENT_MATCH_DATA_PATH) -> pd.DataFrame:
    """Load real match-level World Cup data if provided locally."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=TOURNAMENT_MATCH_COLUMNS)
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=TOURNAMENT_MATCH_COLUMNS)
    if data.empty:
        return pd.DataFrame(columns=TOURNAMENT_MATCH_COLUMNS)
    return data


def aggregate_tournament_match_data(match_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate real match rows to player-level tournament fields."""

    if match_data.empty:
        return pd.DataFrame()

    df = match_data.copy()
    for col in [
        "minutes",
        "goals",
        "assists",
        "shots",
        "key_passes",
        "tackles",
        "interceptions",
        "saves",
        "clean_sheets",
        "match_rating",
    ]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce").fillna(0)

    df["tournament_start"] = (df["minutes"] >= 60).astype(int)
    grouped = (
        df.groupby("player_name", as_index=False)
        .agg(
            tournament_matches=("match_id", "nunique"),
            tournament_minutes=("minutes", "sum"),
            tournament_starts=("tournament_start", "sum"),
            tournament_goals=("goals", "sum"),
            tournament_assists=("assists", "sum"),
            tournament_shots=("shots", "sum"),
            tournament_key_passes=("key_passes", "sum"),
            tournament_tackles=("tackles", "sum"),
            tournament_interceptions=("interceptions", "sum"),
            tournament_saves=("saves", "sum"),
            tournament_clean_sheets=("clean_sheets", "sum"),
            tournament_match_rating=("match_rating", "mean"),
        )
    )
    grouped["tournament_match_rating"] = grouped["tournament_match_rating"].replace(0, pd.NA)
    return grouped


def load_aggregated_tournament_data(path: Path | str = TOURNAMENT_MATCH_DATA_PATH) -> pd.DataFrame:
    """Load and aggregate real tournament data from the local CSV path."""

    return aggregate_tournament_match_data(load_tournament_match_data(path))
