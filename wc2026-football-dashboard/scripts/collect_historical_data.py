"""Collect real historical World Cup player-tournament training rows.

This collector uses compliant public CSV files from the Fjelstul World Cup
Database as republished by DataHub under CC-BY-SA 4.0. It does not scrape
restricted football data providers and it does not fabricate unavailable
fields.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_sources import HISTORICAL_DATA_DIR, HISTORICAL_TRAINING_PATH


DATAHUB_BASE_URL = "https://datahub.io/football/worldcup/_r/-"
FJELSTUL_REPOSITORY_URL = "https://github.com/jfjelstul/worldcup"
DATAHUB_DATASET_URL = "https://datahub.io/football/worldcup"
TARGET_YEARS = {2014, 2018, 2022}

SOURCE_TABLES = {
    "tournaments": f"{DATAHUB_BASE_URL}/tournaments.csv",
    "squads": f"{DATAHUB_BASE_URL}/squads.csv",
    "players": f"{DATAHUB_BASE_URL}/players.csv",
    "matches": f"{DATAHUB_BASE_URL}/matches.csv",
    "player_appearances": f"{DATAHUB_BASE_URL}/player_appearances.csv",
    "goals": f"{DATAHUB_BASE_URL}/goals.csv",
    "substitutions": f"{DATAHUB_BASE_URL}/substitutions.csv",
    "bookings": f"{DATAHUB_BASE_URL}/bookings.csv",
    "tournament_standings": f"{DATAHUB_BASE_URL}/tournament_standings.csv",
    "group_standings": f"{DATAHUB_BASE_URL}/group_standings.csv",
}

AUDIT_LOG_PATH = HISTORICAL_DATA_DIR / "source_audit_log.csv"
AUDIT_COLUMNS = [
    "source_name",
    "source_url_or_location",
    "source_type",
    "license_or_usage_note",
    "access_method",
    "date_accessed",
    "fields_used",
    "allowed_for_project",
    "notes",
]

MODEL_AND_TARGET_COLUMNS = [
    "tournament_year",
    "player_name",
    "country",
    "position",
    "age",
    "club",
    "league",
    "market_value_before_tournament",
    "club_level_score",
    "league_strength_score",
    "national_team_strength",
    "group_difficulty_score",
    "club_minutes_previous_season",
    "goals_previous_season",
    "assists_previous_season",
    "recent_form_score",
    "role_fit_score",
    "expected_starter_score",
    "national_team_caps",
    "senior_national_team_caps",
    "major_tournament_experience",
    "is_world_cup_debutant",
    "previous_world_cup_minutes",
    "previous_world_cup_matches",
    "previous_world_cup_impact_score",
    "age_group",
    "injury_availability_score",
    "minutes_at_tournament",
    "starts_at_tournament",
    "appearances_at_tournament",
    "substitute_appearances_at_tournament",
    "goals_at_tournament",
    "assists_at_tournament",
    "clean_sheets_at_tournament",
    "yellow_cards_at_tournament",
    "red_cards_at_tournament",
    "actual_tournament_impact_score",
    "data_source",
    "data_source_url",
]


def _read_public_csv(url: str) -> pd.DataFrame:
    """Read one public CSV table from the configured source."""

    return pd.read_csv(url)


def _player_name(given_name: object, family_name: object) -> str:
    parts = [
        str(part).strip()
        for part in [given_name, family_name]
        if pd.notna(part) and str(part).strip()
    ]
    return " ".join(parts)


def _normalise_position(position_code: object, position_name: object) -> str:
    code = str(position_code).strip().upper() if pd.notna(position_code) else ""
    name = str(position_name).strip().lower() if pd.notna(position_name) else ""
    if code == "GK" or "keeper" in name:
        return "Goalkeeper"
    if code == "DF" or "defender" in name:
        return "Defender"
    if code == "MF" or "midfielder" in name:
        return "Midfielder"
    if code == "FW" or "forward" in name:
        return "Forward"
    return "Unknown"


def _age_group(age: object) -> object:
    if pd.isna(age):
        return pd.NA
    age_value = float(age)
    if age_value <= 21:
        return "U21"
    if age_value <= 23:
        return "U23"
    if age_value <= 25:
        return "U25"
    if age_value <= 29:
        return "Prime"
    return "Veteran"


def _event_minute(df: pd.DataFrame) -> pd.Series:
    regulation = pd.to_numeric(df.get("minute_regulation"), errors="coerce").fillna(0)
    stoppage = pd.to_numeric(df.get("minute_stoppage"), errors="coerce").fillna(0)
    return regulation + stoppage


def _position_percentile(values: pd.Series, groups: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0)
    return numeric.groupby(groups).rank(method="average", pct=True).mul(100)


def _calculate_actual_impact(df: pd.DataFrame) -> pd.Series:
    """Create a transparent MVP target from real tournament output fields."""

    groups = df["position"].fillna("Unknown")
    minutes_pct = _position_percentile(df["minutes_at_tournament"], groups)
    starts_pct = _position_percentile(df["starts_at_tournament"], groups)
    goals_pct = _position_percentile(df["goals_at_tournament"], groups)
    clean_sheets_pct = _position_percentile(df["clean_sheets_at_tournament"], groups)

    raw = pd.Series(0.0, index=df.index)
    is_goalkeeper = df["position"].eq("Goalkeeper")
    raw.loc[is_goalkeeper] = (
        0.45 * minutes_pct.loc[is_goalkeeper]
        + 0.25 * starts_pct.loc[is_goalkeeper]
        + 0.30 * clean_sheets_pct.loc[is_goalkeeper]
    )
    raw.loc[~is_goalkeeper] = (
        0.45 * minutes_pct.loc[~is_goalkeeper]
        + 0.20 * starts_pct.loc[~is_goalkeeper]
        + 0.35 * goals_pct.loc[~is_goalkeeper]
    )
    impact = raw.groupby(groups).rank(method="average", pct=True).mul(100).clip(0, 100)
    impact.loc[pd.to_numeric(df["minutes_at_tournament"], errors="coerce").fillna(0).le(0)] = 0
    return impact.round(1)


def _build_player_tournament_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build real player-tournament rows for all available men's World Cups."""

    tournaments = tables["tournaments"].copy()
    tournaments = tournaments[
        tournaments["tournament_name"].astype(str).str.contains("Men's World Cup", case=False, na=False)
    ].copy()
    tournament_ids = set(tournaments["tournament_id"])

    if not tournament_ids:
        raise RuntimeError("No men's World Cup tournaments were found in the public source tables.")

    squads = tables["squads"].copy()
    squads = squads[squads["tournament_id"].isin(tournament_ids)].copy()
    players = tables["players"].copy()
    matches = tables["matches"].copy()
    matches = matches[matches["tournament_id"].isin(tournament_ids)].copy()
    appearances = tables["player_appearances"].copy()
    appearances = appearances[appearances["tournament_id"].isin(tournament_ids)].copy()
    goals = tables["goals"].copy()
    goals = goals[goals["tournament_id"].isin(tournament_ids)].copy()
    substitutions = tables["substitutions"].copy()
    substitutions = substitutions[substitutions["tournament_id"].isin(tournament_ids)].copy()
    bookings = tables["bookings"].copy()
    bookings = bookings[bookings["tournament_id"].isin(tournament_ids)].copy()

    matches["base_match_length"] = np.where(pd.to_numeric(matches["extra_time"], errors="coerce").fillna(0).eq(1), 120, 90)
    substitutions["event_minute"] = _event_minute(substitutions)
    max_sub_minute = (
        substitutions.groupby("match_id")["event_minute"].max().rename("max_substitution_minute").reset_index()
    )
    matches = matches.merge(max_sub_minute, on="match_id", how="left")
    matches["match_length"] = matches[["base_match_length", "max_substitution_minute"]].max(axis=1).fillna(matches["base_match_length"])

    sub_off = (
        substitutions[substitutions["going_off"].eq(1)]
        .groupby(["tournament_id", "match_id", "team_id", "player_id"])["event_minute"]
        .min()
        .rename("sub_off_minute")
        .reset_index()
    )
    sub_on = (
        substitutions[substitutions["coming_on"].eq(1)]
        .groupby(["tournament_id", "match_id", "team_id", "player_id"])["event_minute"]
        .min()
        .rename("sub_on_minute")
        .reset_index()
    )

    app = appearances.merge(
        matches[
            [
                "tournament_id",
                "match_id",
                "home_team_id",
                "away_team_id",
                "home_team_score",
                "away_team_score",
                "match_length",
            ]
        ],
        on=["tournament_id", "match_id"],
        how="left",
    )
    app = app.merge(sub_off, on=["tournament_id", "match_id", "team_id", "player_id"], how="left")
    app = app.merge(sub_on, on=["tournament_id", "match_id", "team_id", "player_id"], how="left")

    starter = pd.to_numeric(app["starter"], errors="coerce").fillna(0).eq(1)
    substitute = pd.to_numeric(app["substitute"], errors="coerce").fillna(0).eq(1)
    app["minutes"] = pd.NA
    app.loc[starter, "minutes"] = app.loc[starter, "sub_off_minute"].combine_first(app.loc[starter, "match_length"])
    app.loc[substitute, "minutes"] = app.loc[substitute, "match_length"] - app.loc[substitute, "sub_on_minute"]
    app["minutes"] = pd.to_numeric(app["minutes"], errors="coerce").clip(lower=0)

    is_gk = app["position_code"].astype(str).str.upper().eq("GK")
    home_clean_sheet = app["team_id"].eq(app["home_team_id"]) & pd.to_numeric(app["away_team_score"], errors="coerce").eq(0)
    away_clean_sheet = app["team_id"].eq(app["away_team_id"]) & pd.to_numeric(app["home_team_score"], errors="coerce").eq(0)
    app["clean_sheet"] = (is_gk & (home_clean_sheet | away_clean_sheet) & app["minutes"].fillna(0).gt(0)).astype(int)

    appearance_agg = (
        app.groupby(["tournament_id", "team_id", "player_id"], as_index=False)
        .agg(
            minutes_at_tournament=("minutes", "sum"),
            starts_at_tournament=("starter", "sum"),
            appearances_at_tournament=("match_id", "nunique"),
            substitute_appearances_at_tournament=("substitute", "sum"),
            clean_sheets_at_tournament=("clean_sheet", "sum"),
        )
    )

    goals = goals[pd.to_numeric(goals["own_goal"], errors="coerce").fillna(0).eq(0)].copy()
    goals_agg = (
        goals.groupby(["tournament_id", "player_team_id", "player_id"], as_index=False)
        .size()
        .rename(columns={"player_team_id": "team_id", "size": "goals_at_tournament"})
    )

    cards_agg = (
        bookings.groupby(["tournament_id", "team_id", "player_id"], as_index=False)
        .agg(
            yellow_cards_at_tournament=("yellow_card", "sum"),
            red_cards_at_tournament=("sending_off", "sum"),
        )
    )

    squad_rows = squads.merge(
        tournaments[["tournament_id", "year", "start_date"]],
        on="tournament_id",
        how="left",
    ).merge(
        players[["player_id", "birth_date"]],
        on="player_id",
        how="left",
    )

    start_date = pd.to_datetime(squad_rows["start_date"], errors="coerce")
    birth_date = pd.to_datetime(squad_rows["birth_date"], errors="coerce")
    squad_rows["age"] = ((start_date - birth_date).dt.days / 365.25).apply(np.floor)
    squad_rows.loc[birth_date.isna() | start_date.isna(), "age"] = pd.NA

    output = squad_rows.merge(
        appearance_agg,
        on=["tournament_id", "team_id", "player_id"],
        how="left",
    ).merge(
        goals_agg,
        on=["tournament_id", "team_id", "player_id"],
        how="left",
    ).merge(
        cards_agg,
        on=["tournament_id", "team_id", "player_id"],
        how="left",
    )

    real_zero_columns = [
        "minutes_at_tournament",
        "starts_at_tournament",
        "appearances_at_tournament",
        "substitute_appearances_at_tournament",
        "goals_at_tournament",
        "clean_sheets_at_tournament",
        "yellow_cards_at_tournament",
        "red_cards_at_tournament",
    ]
    for col in real_zero_columns:
        output[col] = pd.to_numeric(output[col], errors="coerce").fillna(0)

    output["tournament_year"] = pd.to_numeric(output["year"], errors="coerce").astype("Int64")
    output["player_name"] = output.apply(lambda row: _player_name(row.get("given_name"), row.get("family_name")), axis=1)
    output["country"] = output["team_name"]
    output["position"] = output.apply(lambda row: _normalise_position(row.get("position_code"), row.get("position_name")), axis=1)
    output["age_group"] = output["age"].apply(_age_group)
    output["assists_at_tournament"] = pd.NA
    output["actual_tournament_impact_score"] = _calculate_actual_impact(output)
    output["data_source"] = "Fjelstul World Cup Database via DataHub"
    output["data_source_url"] = DATAHUB_DATASET_URL

    unavailable_columns = [
        "club",
        "league",
        "market_value_before_tournament",
        "club_level_score",
        "league_strength_score",
        "club_minutes_previous_season",
        "goals_previous_season",
        "assists_previous_season",
        "recent_form_score",
        "role_fit_score",
        "expected_starter_score",
        "national_team_caps",
        "senior_national_team_caps",
        "major_tournament_experience",
        "injury_availability_score",
    ]
    for col in unavailable_columns:
        output[col] = pd.NA

    output = output.sort_values(
        ["tournament_year", "country", "player_name"],
        kind="stable",
    )
    return output.reset_index(drop=True)


def _build_team_strength_scores(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Use previous World Cup final standing as a real-derived team-strength proxy."""

    tournaments = tables["tournaments"].copy()
    tournaments = tournaments[
        tournaments["tournament_name"].astype(str).str.contains("Men's World Cup", case=False, na=False)
    ][["tournament_id", "year"]].copy()
    standings = tables["tournament_standings"].copy()
    standings = standings.merge(tournaments, on="tournament_id", how="inner")
    standings["position"] = pd.to_numeric(standings["position"], errors="coerce")
    team_count = standings.groupby("tournament_id")["team_id"].transform("nunique")
    denominator = (team_count - 1).replace(0, pd.NA)
    standings["world_cup_finish_strength_score"] = ((team_count - standings["position"]) / denominator * 100).clip(0, 100)
    return standings[
        ["tournament_id", "year", "team_id", "team_name", "world_cup_finish_strength_score"]
    ].dropna(subset=["world_cup_finish_strength_score"])


def _add_previous_team_strength(output: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add national-team strength from the team's most recent previous World Cup finish."""

    output = output.copy()
    strength_scores = _build_team_strength_scores(tables)
    scores_by_team = {
        team_id: group.sort_values("year")[["year", "world_cup_finish_strength_score"]].to_records(index=False).tolist()
        for team_id, group in strength_scores.groupby("team_id")
    }

    def previous_strength(row: pd.Series) -> object:
        team_scores = scores_by_team.get(row["team_id"], [])
        prior_scores = [score for year, score in team_scores if year < row["tournament_year"]]
        return prior_scores[-1] if prior_scores else pd.NA

    output["national_team_strength"] = output.apply(previous_strength, axis=1)
    return output


def _add_group_difficulty(output: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Add average opponent previous-strength proxy for the current group."""

    output = output.copy()
    group_standings = tables["group_standings"].copy()
    group_standings = group_standings[
        group_standings["stage_name"].astype(str).str.lower().eq("group stage")
    ][["tournament_id", "group_name", "team_id"]].drop_duplicates()

    team_strength = output[
        ["tournament_id", "team_id", "national_team_strength"]
    ].drop_duplicates(subset=["tournament_id", "team_id"])
    groups = group_standings.merge(team_strength, on=["tournament_id", "team_id"], how="left")

    difficulty: dict[tuple[object, object], object] = {}
    for (tournament_id, group_name), group in groups.groupby(["tournament_id", "group_name"]):
        for _, row in group.iterrows():
            opponents = group[group["team_id"] != row["team_id"]]
            values = pd.to_numeric(opponents["national_team_strength"], errors="coerce").dropna()
            difficulty[(tournament_id, row["team_id"])] = round(float(values.mean()), 1) if not values.empty else pd.NA

    output["group_difficulty_score"] = [
        difficulty.get((row.tournament_id, row.team_id), pd.NA) for row in output.itertuples(index=False)
    ]
    return output


def _add_previous_world_cup_experience(output: pd.DataFrame) -> pd.DataFrame:
    """Add real-derived previous World Cup experience by player ID."""

    output = output.sort_values(["player_id", "tournament_year"], kind="stable").copy()
    output["previous_world_cup_minutes"] = (
        output.groupby("player_id")["minutes_at_tournament"].cumsum() - output["minutes_at_tournament"]
    )
    output["previous_world_cup_matches"] = (
        output.groupby("player_id")["appearances_at_tournament"].cumsum() - output["appearances_at_tournament"]
    )
    output["previous_world_cup_impact_score"] = output.groupby("player_id")["actual_tournament_impact_score"].shift(1)
    output["is_world_cup_debutant"] = output["previous_world_cup_matches"].fillna(0).eq(0)
    return output


def _build_training_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    all_rows = _build_player_tournament_rows(tables)
    all_rows = _add_previous_team_strength(all_rows, tables)
    all_rows = _add_group_difficulty(all_rows, tables)
    all_rows = _add_previous_world_cup_experience(all_rows)

    output = all_rows[all_rows["tournament_year"].isin(TARGET_YEARS)].copy()
    output = output[MODEL_AND_TARGET_COLUMNS].sort_values(
        ["tournament_year", "country", "player_name"],
        kind="stable",
    )
    return output.reset_index(drop=True)


def _build_source_audit_log() -> pd.DataFrame:
    accessed = date.today().isoformat()
    rows = [
        {
            "source_name": "Fjelstul World Cup Database via DataHub",
            "source_url_or_location": DATAHUB_DATASET_URL,
            "source_type": "public/open CSV dataset",
            "license_or_usage_note": "CC-BY-SA 4.0; attribution required to Joshua C. Fjelstul, Ph.D. and share-alike applies.",
            "access_method": "Direct CSV download from DataHub stable URLs",
            "date_accessed": accessed,
            "fields_used": (
                "squads, players, tournaments, matches, player_appearances, goals, substitutions, bookings, "
                "tournament_standings, group_standings"
            ),
            "allowed_for_project": True,
            "notes": (
                "Used to create real player-tournament rows for 2014, 2018 and 2022. "
                "Previous World Cup experience is derived from earlier real World Cup rows. "
                "National-team strength is a previous World Cup final-standing proxy, not FIFA ranking. "
                "Club, assists, market values, senior caps and club-season pre-tournament features are not provided and remain null."
            ),
        },
        {
            "source_name": "Original Fjelstul World Cup Database repository",
            "source_url_or_location": FJELSTUL_REPOSITORY_URL,
            "source_type": "upstream public/open dataset repository",
            "license_or_usage_note": "CC-BY-SA 4.0; attribution and share-alike requirements documented in repository README.",
            "access_method": "Referenced for provenance and licensing",
            "date_accessed": accessed,
            "fields_used": "Provenance and license metadata",
            "allowed_for_project": True,
            "notes": "DataHub states the CSV dataset is sourced from this database.",
        },
        {
            "source_name": "StatsBomb Open Data",
            "source_url_or_location": "https://github.com/statsbomb/open-data",
            "source_type": "public/open event-data repository",
            "license_or_usage_note": "Public open-data terms require attribution; only included competitions may be used.",
            "access_method": "Candidate source inspected, not used in current generated CSV",
            "date_accessed": accessed,
            "fields_used": "",
            "allowed_for_project": True,
            "notes": "Not used for this MVP because the public competition index covers 2018 and 2022 but not 2014.",
        },
        {
            "source_name": "Opta, WhoScored, Wyscout direct website scraping",
            "source_url_or_location": "Restricted/commercial provider websites",
            "source_type": "restricted/commercial data providers",
            "license_or_usage_note": "Use only licensed exports when the user has rights. Do not scrape provider websites.",
            "access_method": "Rejected",
            "date_accessed": accessed,
            "fields_used": "",
            "allowed_for_project": False,
            "notes": "Rejected to preserve the project's compliant real-data-only workflow.",
        },
        {
            "source_name": "Transfermarkt direct page scraping",
            "source_url_or_location": "Transfermarkt website pages",
            "source_type": "market-value website",
            "license_or_usage_note": "Manual, licensed or clearly permitted exports only. Direct scraping was not implemented.",
            "access_method": "Rejected",
            "date_accessed": accessed,
            "fields_used": "",
            "allowed_for_project": False,
            "notes": "Rejected for direct scraping; market values remain null until a compliant export is provided.",
        },
    ]
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def collect_historical_data() -> pd.DataFrame:
    """Download compliant public source tables and write the historical training CSV."""

    HISTORICAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    tables = {name: _read_public_csv(url) for name, url in SOURCE_TABLES.items()}
    training_rows = _build_training_rows(tables)
    training_rows.to_csv(HISTORICAL_TRAINING_PATH, index=False)
    _build_source_audit_log().to_csv(AUDIT_LOG_PATH, index=False)
    return training_rows


def main() -> int:
    try:
        training_rows = collect_historical_data()
    except Exception as exc:
        print("Historical data collection failed.")
        print(f"Reason: {exc}")
        print("No synthetic fallback will be generated. Check network access or provide compliant local CSVs.")
        return 1

    years = ", ".join(str(year) for year in sorted(training_rows["tournament_year"].dropna().unique()))
    print("Historical World Cup training data collected")
    print("=" * 52)
    print(f"Rows written: {len(training_rows):,}")
    print(f"Tournament years: {years}")
    print(f"Training file: {HISTORICAL_TRAINING_PATH}")
    print(f"Source audit log: {AUDIT_LOG_PATH}")
    print("Unavailable source fields were left null; no values were fabricated.")
    print("=" * 52)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
