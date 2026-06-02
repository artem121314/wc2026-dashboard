"""Compliant real-data source definitions.

Place manually curated, licensed, or otherwise compliant data exports in the
paths below. The app intentionally does not scrape restricted football data
providers such as Opta, WhoScored, Wyscout, or similar commercial sources.
Public GitHub raw CSVs or Kaggle files can be used only after the data has been
downloaded or configured explicitly by the user.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
HISTORICAL_DATA_DIR = DATA_ROOT / "historical"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

CURRENT_PLAYER_POOL_PATH = RAW_DATA_DIR / "current_player_pool.csv"
MARKET_VALUES_PATH = RAW_DATA_DIR / "market_values.csv"
NATIONAL_TEAM_CONTEXT_PATH = RAW_DATA_DIR / "national_team_context.csv"
TOURNAMENT_MATCH_DATA_PATH = RAW_DATA_DIR / "tournament_match_data.csv"
HISTORICAL_TRAINING_PATH = HISTORICAL_DATA_DIR / "world_cup_player_training_data.csv"
PROCESSED_DASHBOARD_PATH = PROCESSED_DATA_DIR / "player_dashboard_data.csv"


@dataclass(frozen=True)
class DataSourceSpec:
    name: str
    path: Path
    required_columns: tuple[str, ...]
    required_for_refresh: bool
    description: str


DATA_SOURCE_SPECS = {
    "current_player_pool": DataSourceSpec(
        name="Current player pool",
        path=CURRENT_PLAYER_POOL_PATH,
        required_columns=(
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
        ),
        required_for_refresh=True,
        description="Real current player pool and recent senior performance export.",
    ),
    "market_values": DataSourceSpec(
        name="Market values",
        path=MARKET_VALUES_PATH,
        required_columns=("player_name", "market_value_eur"),
        required_for_refresh=False,
        description="Real market value export keyed by player_name, optionally country.",
    ),
    "national_team_context": DataSourceSpec(
        name="National team context",
        path=NATIONAL_TEAM_CONTEXT_PATH,
        required_columns=("country", "national_team_strength"),
        required_for_refresh=False,
        description="Real national-team strength and tournament context export.",
    ),
    "tournament_match_data": DataSourceSpec(
        name="Tournament match data",
        path=TOURNAMENT_MATCH_DATA_PATH,
        required_columns=(
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
        ),
        required_for_refresh=False,
        description="Real World Cup match-level box-score rows. Add rows after each match and refresh.",
    ),
    "historical_training": DataSourceSpec(
        name="Historical training data",
        path=HISTORICAL_TRAINING_PATH,
        required_columns=(
            "tournament_year",
            "player_name",
            "age",
            "position",
            "market_value_before_tournament",
            "club_level_score",
            "league_strength_score",
            "club_minutes_previous_season",
            "goals_previous_season",
            "assists_previous_season",
            "national_team_caps",
            "expected_starter_score",
            "national_team_strength",
            "group_difficulty_score",
            "injury_availability_score",
            "recent_form_score",
            "role_fit_score",
            "actual_tournament_impact_score",
        ),
        required_for_refresh=False,
        description="Real curated 2014/2018/2022 player-level training data.",
    ),
    "processed_dashboard": DataSourceSpec(
        name="Processed dashboard data",
        path=PROCESSED_DASHBOARD_PATH,
        required_columns=(
            "player_name",
            "pre_tournament_expected_impact_score",
            "value_opportunity_score",
            "performance_outcome",
        ),
        required_for_refresh=False,
        description="Dashboard-ready processed data generated from real inputs.",
    ),
}
