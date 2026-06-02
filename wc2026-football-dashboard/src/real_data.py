"""Real-data ingestion for the WC 2026 dashboard.

Sources used by this module:
- Wikipedia 2026 FIFA World Cup squad tables, which mirror the FIFA/federation
  final squad lists.
- dcaribou Transfermarkt dataset CSVs for player market values, clubs,
  national-team rankings, and recent appearances.
- openfootball/worldcup.json for the 2026 tournament draw and group fixtures.

The generated dataset keeps the dashboard schema stable while replacing the
development fixture with a real World Cup player pool.
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
from datetime import date
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_features.csv"
REAL_SQUADS_PATH = PROJECT_ROOT / "data" / "processed" / "wc2026_real_squads.csv"
SOURCE_MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "source_manifest.json"
INJURY_STATUS_PATH = PROJECT_ROOT / "data" / "raw" / "injury_status.csv"

WIKIPEDIA_SQUADS_URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
TRANSFERMARKT_DATA_BASE = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
SALIMT_TRANSFERMARKT_BASE = (
    "https://raw.githubusercontent.com/salimt/football-datasets/main/datalake/transfermarkt"
)
OPENFOOTBALL_2026_URL = (
    "https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json"
)

HEADERS = {"User-Agent": "wc2026-dashboard/1.0 real-data-refresh"}

POSITION_MAP = {
    "GK": "Goalkeeper",
    "DF": "Centre-back",
    "MF": "Central midfielder",
    "FW": "Forward",
}

SUB_POSITION_MAP = {
    "Goalkeeper": "Goalkeeper",
    "Centre-Back": "Centre-back",
    "Left-Back": "Full-back",
    "Right-Back": "Full-back",
    "Defensive Midfield": "Defensive midfielder",
    "Central Midfield": "Central midfielder",
    "Attacking Midfield": "Attacking midfielder",
    "Left Midfield": "Winger",
    "Right Midfield": "Winger",
    "Left Winger": "Winger",
    "Right Winger": "Winger",
    "Second Striker": "Forward",
    "Centre-Forward": "Forward",
}

COUNTRY_ALIASES = {
    "Czech Republic": ["Czechia"],
    "South Korea": ["Korea, South", "Korea Republic", "Republic of Korea"],
    "United States": ["USA", "United States of America", "USMNT"],
    "Ivory Coast": ["Cote d'Ivoire", "Cote d Ivoire"],
    "DR Congo": ["Congo DR", "Democratic Republic of the Congo"],
    "Turkey": ["Turkiye", "Türkiye"],
    "Cape Verde": ["Cabo Verde"],
    "Curacao": ["Curaçao"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina"],
}

LEAGUE_TIER_SCORES = {
    "GB1": 95,
    "ES1": 94,
    "L1": 91,
    "IT1": 89,
    "FR1": 85,
    "PO1": 78,
    "NL1": 76,
    "TR1": 73,
    "BE1": 70,
    "MLS1": 68,
    "SA1": 67,
    "AR1N": 66,
    "BR1": 66,
}


def normalize_text(value: Any) -> str:
    """Normalize names/countries for cross-source matching."""

    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"\([^)]*\)", "", text)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text).strip().lower()
    return text


def country_keys(country: str) -> set[str]:
    """Return normalized country aliases for matching."""

    keys = {normalize_text(country)}
    for canonical, aliases in COUNTRY_ALIASES.items():
        all_names = [canonical, *aliases]
        if normalize_text(country) in {normalize_text(name) for name in all_names}:
            keys.update(normalize_text(name) for name in all_names)
    return keys


def name_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    left_rev = " ".join(reversed(left_norm.split()))
    right_rev = " ".join(reversed(right_norm.split()))
    return max(
        SequenceMatcher(None, left_norm, right_norm).ratio(),
        SequenceMatcher(None, left_rev, right_norm).ratio(),
        SequenceMatcher(None, left_norm, right_rev).ratio(),
    )


def fetch_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.text


@lru_cache(maxsize=16)
def read_remote_csv_gz(name: str, nrows: int | None = None) -> pd.DataFrame:
    """Read a gzip CSV from the published Transfermarkt dataset."""

    url = f"{TRANSFERMARKT_DATA_BASE}/{name}.csv.gz"
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content), compression="gzip", nrows=nrows)


@lru_cache(maxsize=8)
def read_salimt_csv(folder: str, filename: str) -> pd.DataFrame:
    url = f"{SALIMT_TRANSFERMARKT_BASE}/{folder}/{filename}"
    response = requests.get(url, headers=HEADERS, timeout=120)
    response.raise_for_status()
    return pd.read_csv(io.BytesIO(response.content))


def fetch_world_cup_squads() -> pd.DataFrame:
    """Fetch the 2026 World Cup final squad tables."""

    html = fetch_text(WIKIPEDIA_SQUADS_URL)
    soup = BeautifulSoup(html, "html.parser")
    rows: list[pd.DataFrame] = []

    for table in soup.find_all("table", class_="wikitable"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")[:8]]
        if not any("Player" in header for header in headers) or not any("Pos" in header for header in headers):
            continue

        heading = table.find_previous(["h3", "h2"])
        country = heading.get_text(" ", strip=True).replace("[ edit ]", "").strip() if heading else "Unknown"
        squad = pd.read_html(io.StringIO(str(table)))[0]
        squad["country"] = country
        rows.append(squad)

    if not rows:
        raise RuntimeError("No 2026 World Cup squad tables found.")

    df = pd.concat(rows, ignore_index=True)
    df = df.rename(
        columns={
            "No.": "squad_number",
            "Pos.": "raw_position",
            "Player": "player_name",
            "Date of birth (age)": "date_of_birth_age",
            "Caps": "caps",
            "Goals": "international_goals",
            "Club": "club",
        }
    )
    df["player_name"] = df["player_name"].astype(str).str.replace(r"\s+\(captain\)", "", regex=True)
    df["position"] = df["raw_position"].map(POSITION_MAP).fillna("Central midfielder")
    df["age"] = (
        df["date_of_birth_age"]
        .astype(str)
        .str.extract(r"aged\s+(\d+)", expand=False)
        .astype(float)
        .fillna(26)
        .astype(int)
    )
    df["date_of_birth"] = df["date_of_birth_age"].astype(str).str.extract(r"^([^(]+)", expand=False).str.strip()
    df["caps"] = pd.to_numeric(df["caps"], errors="coerce").fillna(0).astype(int)
    df["international_goals"] = pd.to_numeric(df["international_goals"], errors="coerce").fillna(0).astype(int)
    df["final_squad_selected"] = True
    df["player_name_norm"] = df["player_name"].map(normalize_text)
    df["club_norm"] = df["club"].map(normalize_text)
    return df


def fetch_transfermarkt_players() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    players = read_remote_csv_gz("players")
    national_teams = read_remote_csv_gz("national_teams")
    competitions = read_remote_csv_gz("competitions")

    national_cols = [
        "national_team_id",
        "name",
        "country_name",
        "confederation",
        "total_market_value",
        "fifa_ranking",
    ]
    national_teams = national_teams[national_cols].rename(
        columns={
            "name": "national_team_name",
            "country_name": "national_team_country",
            "total_market_value": "national_team_market_value",
        }
    )
    players = players.merge(
        national_teams,
        left_on="current_national_team_id",
        right_on="national_team_id",
        how="left",
    )
    players["player_name_norm"] = players["name"].map(normalize_text)
    players["club_norm"] = players["current_club_name"].map(normalize_text)
    return players, national_teams, competitions


def _choose_transfermarkt_match(
    squad_row: pd.Series,
    candidates: pd.DataFrame,
    minimum_name_similarity: float = 0.0,
) -> pd.Series | None:
    if candidates.empty:
        return None

    squad_country_keys = country_keys(str(squad_row["country"]))
    club_norm = squad_row.get("club_norm", "")
    scored = candidates.copy()
    scored["match_score"] = 0
    scored["name_similarity"] = scored["name"].map(lambda value: name_similarity(squad_row["player_name"], value))
    if minimum_name_similarity:
        scored = scored[scored["name_similarity"] >= minimum_name_similarity]
        if scored.empty:
            return None

    for col in ["country_of_citizenship", "national_team_name", "national_team_country"]:
        if col in scored.columns:
            scored["match_score"] += scored[col].map(lambda value: 80 if normalize_text(value) in squad_country_keys else 0)

    scored["match_score"] += scored["name_similarity"] * 60
    scored["match_score"] += scored["club_norm"].map(lambda value: 20 if value == club_norm and value else 0)
    scored["match_score"] += scored["market_value_in_eur"].fillna(0).rank(pct=True).fillna(0).mul(5)
    scored = scored.sort_values("match_score", ascending=False)
    return scored.iloc[0]


def match_squads_to_transfermarkt(squads: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Attach Transfermarkt player metadata to final squad players."""

    candidate_map = {name: frame for name, frame in players.groupby("player_name_norm")}
    country_candidate_map: dict[str, pd.DataFrame] = {}
    for country_key in sorted({key for country in squads["country"].unique() for key in country_keys(country)}):
        mask = pd.Series(False, index=players.index)
        for col in ["country_of_citizenship", "national_team_name", "national_team_country"]:
            if col in players.columns:
                mask = mask | players[col].map(lambda value: normalize_text(value) == country_key)
        country_candidate_map[country_key] = players[mask]
    matched_rows: list[dict[str, Any]] = []

    for _, squad_row in squads.iterrows():
        candidates = candidate_map.get(squad_row["player_name_norm"], pd.DataFrame())
        match = _choose_transfermarkt_match(squad_row, candidates)
        if match is None:
            country_candidates = pd.concat(
                [country_candidate_map.get(key, pd.DataFrame()) for key in country_keys(squad_row["country"])],
                ignore_index=False,
            ).drop_duplicates("player_id")
            match = _choose_transfermarkt_match(
                squad_row,
                country_candidates,
                minimum_name_similarity=0.72,
            )
        base = squad_row.to_dict()
        if match is not None:
            for col in [
                "player_id",
                "current_club_id",
                "current_club_name",
                "current_club_domestic_competition_id",
                "sub_position",
                "foot",
                "height_in_cm",
                "market_value_in_eur",
                "highest_market_value_in_eur",
                "international_caps",
                "national_team_name",
                "national_team_market_value",
                "fifa_ranking",
                "confederation",
                "url",
                "image_url",
            ]:
                base[f"tm_{col}"] = match.get(col)
            if pd.notna(match.get("sub_position")):
                base["position"] = SUB_POSITION_MAP.get(str(match.get("sub_position")), base["position"])
            if pd.notna(match.get("current_club_name")):
                base["club"] = match.get("current_club_name")
            base["transfermarkt_match"] = True
        else:
            base["transfermarkt_match"] = False
        matched_rows.append(base)

    return pd.DataFrame(matched_rows)


def fetch_recent_appearances(player_ids: list[int | float | str]) -> pd.DataFrame:
    """Aggregate recent Transfermarkt appearances for matched players."""

    ids = pd.Series(player_ids).dropna().astype(int).unique()
    if len(ids) == 0:
        return pd.DataFrame(columns=["player_id", "minutes", "goals", "assists"])

    appearances = read_remote_csv_gz("appearances")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    recent = appearances[
        appearances["player_id"].isin(ids)
        & appearances["date"].between(pd.Timestamp("2025-07-01"), pd.Timestamp("2026-06-10"))
    ]
    if recent.empty:
        return pd.DataFrame(columns=["player_id", "minutes", "goals", "assists"])

    return (
        recent.groupby("player_id", as_index=False)
        .agg(
            minutes=("minutes_played", "sum"),
            goals=("goals", "sum"),
            assists=("assists", "sum"),
            yellow_cards=("yellow_cards", "sum"),
            red_cards=("red_cards", "sum"),
            recent_appearances=("game_id", "nunique"),
        )
    )


def fetch_latest_market_values(player_ids: list[int | float | str]) -> pd.DataFrame:
    """Fetch latest known Transfermarkt valuation for matched players."""

    ids = pd.Series(player_ids).dropna().astype(int).unique()
    if len(ids) == 0:
        return pd.DataFrame(columns=["player_id", "latest_market_value_eur", "latest_market_value_date"])

    valuations = read_remote_csv_gz("player_valuations")
    valuations = valuations[valuations["player_id"].isin(ids)].copy()
    if valuations.empty:
        return pd.DataFrame(columns=["player_id", "latest_market_value_eur", "latest_market_value_date"])
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    valuations = valuations.sort_values(["player_id", "date"])
    latest = valuations.groupby("player_id", as_index=False).tail(1)
    return latest.rename(
        columns={
            "market_value_in_eur": "latest_market_value_eur",
            "date": "latest_market_value_date",
            "player_club_domestic_competition_id": "latest_value_competition_id",
        }
    )[
        [
            "player_id",
            "latest_market_value_eur",
            "latest_market_value_date",
            "current_club_name",
            "latest_value_competition_id",
        ]
    ]


def fetch_salimt_latest_market_values(player_ids: list[int | float | str]) -> pd.DataFrame:
    """Fetch latest values from the salimt Transfermarkt datalake."""

    ids = pd.Series(player_ids).dropna().astype(int).unique()
    if len(ids) == 0:
        return pd.DataFrame(columns=["player_id", "salimt_latest_market_value_eur"])

    values = read_salimt_csv("player_latest_market_value", "player_latest_market_value.csv")
    values = values[values["player_id"].isin(ids)].copy()
    if values.empty:
        return pd.DataFrame(columns=["player_id", "salimt_latest_market_value_eur"])
    return values.rename(
        columns={
            "value": "salimt_latest_market_value_eur",
            "date_unix": "salimt_latest_market_value_date",
        }
    )


def fetch_salimt_injuries(player_ids: list[int | float | str]) -> pd.DataFrame:
    """Fetch active/recent injury status from the salimt Transfermarkt datalake."""

    ids = pd.Series(player_ids).dropna().astype(int).unique()
    if len(ids) == 0:
        return pd.DataFrame(columns=["player_id"])

    injuries = read_salimt_csv("player_injuries", "player_injuries.csv")
    injuries = injuries[injuries["player_id"].isin(ids)].copy()
    if injuries.empty:
        return pd.DataFrame(columns=["player_id"])

    today = pd.Timestamp(date.today())
    injuries["from_date"] = pd.to_datetime(injuries["from_date"], errors="coerce")
    injuries["end_date"] = pd.to_datetime(injuries["end_date"], errors="coerce")
    injuries["days_missed"] = pd.to_numeric(injuries["days_missed"], errors="coerce").fillna(0)
    injuries["games_missed"] = pd.to_numeric(injuries["games_missed"], errors="coerce").fillna(0)
    injuries["is_active"] = (injuries["from_date"] <= today) & (
        injuries["end_date"].isna() | (injuries["end_date"] >= today)
    )
    injuries["days_since_end"] = (today - injuries["end_date"]).dt.days
    injuries["recent_90"] = injuries["days_since_end"].between(0, 90)
    recency_weight = ((90 - injuries["days_since_end"]) / 90).clip(0, 1).fillna(0)
    active_component = (
        injuries["days_missed"].clip(0, 120) * 0.45
        + injuries["games_missed"].clip(0, 20) * 2.0
        + 35
    ).where(injuries["is_active"], 0)
    recent_component = (
        injuries["days_missed"].clip(0, 120) * 0.08
        + injuries["games_missed"].clip(0, 20) * 0.50
        + 8
    ).mul(recency_weight).where(injuries["recent_90"], 0)
    injuries["severity_component"] = (active_component + recent_component).clip(0, 100)

    agg = (
        injuries.groupby("player_id", as_index=False)
        .agg(
            injury_severity_score=("severity_component", "max"),
            injury_status=("injury_reason", lambda values: "; ".join(pd.Series(values).dropna().astype(str).tail(2))),
            active_injury=("is_active", "max"),
            recent_injury_count=("recent_90", "sum"),
        )
    )
    agg["is_injured"] = agg["active_injury"].astype(bool)
    agg["injury_status"] = np.where(
        agg["is_injured"],
        "Active: " + agg["injury_status"].astype(str),
        np.where(agg["recent_injury_count"] > 0, "Recently returned: " + agg["injury_status"].astype(str), "Available"),
    )
    agg["injury_source_url"] = "https://github.com/salimt/football-datasets"
    return agg.drop(columns=["active_injury"])


def fetch_live_world_cup_actuals(player_ids: list[int | float | str]) -> pd.DataFrame:
    """Aggregate live 2026 World Cup actuals when tournament data is available."""

    ids = pd.Series(player_ids).dropna().astype(int).unique()
    if len(ids) == 0:
        return pd.DataFrame()

    appearances = read_remote_csv_gz("appearances")
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    actuals = appearances[
        appearances["player_id"].isin(ids)
        & (appearances["competition_id"] == "FIWC")
        & (appearances["date"] >= pd.Timestamp("2026-06-11"))
    ]
    if actuals.empty:
        return pd.DataFrame(
            {
                "player_id": ids,
                "actual_wc_minutes": 0,
                "actual_wc_goals": 0,
                "actual_wc_assists": 0,
                "actual_wc_matches": 0,
                "actual_success_target": pd.NA,
            }
        )

    grouped = (
        actuals.groupby("player_id", as_index=False)
        .agg(
            actual_wc_minutes=("minutes_played", "sum"),
            actual_wc_goals=("goals", "sum"),
            actual_wc_assists=("assists", "sum"),
            actual_wc_matches=("game_id", "nunique"),
        )
    )
    grouped["actual_success_target"] = (
        (grouped["actual_wc_minutes"] >= 180)
        | ((grouped["actual_wc_goals"] + grouped["actual_wc_assists"]) >= 2)
    ).astype(int)
    return grouped


def fetch_worldcup_schedule() -> pd.DataFrame:
    data = requests.get(OPENFOOTBALL_2026_URL, headers=HEADERS, timeout=60).json()
    matches = pd.DataFrame(data["matches"])
    group_matches = matches[matches["group"].notna()].copy()
    return group_matches


def calculate_draw_context(squads: pd.DataFrame) -> pd.DataFrame:
    """Calculate group and opponent-strength features from the tournament draw."""

    schedule = fetch_worldcup_schedule()
    teams = sorted(set(schedule["team1"].dropna()) | set(schedule["team2"].dropna()))
    country_strength = squads.groupby("country")["national_team_strength"].mean().to_dict()

    rows = []
    for team in teams:
        team_matches = schedule[(schedule["team1"] == team) | (schedule["team2"] == team)]
        opponents = []
        for _, match in team_matches.iterrows():
            opponent = match["team2"] if match["team1"] == team else match["team1"]
            if not str(opponent).startswith(("W", "L", "3")):
                opponents.append(opponent)
        opponent_strengths = [country_strength.get(opponent, 55) for opponent in opponents]
        rows.append(
            {
                "country": team,
                "group": team_matches["group"].dropna().iloc[0] if not team_matches.empty else "",
                "group_opponents": ", ".join(opponents),
                "average_group_opponent_strength": float(np.mean(opponent_strengths)) if opponent_strengths else 55,
            }
        )
    draw = pd.DataFrame(rows)
    if draw.empty:
        squads["draw_context_score"] = 60
        squads["group"] = ""
        squads["group_opponents"] = ""
        return squads

    draw["draw_context_score"] = (
        100
        - draw["average_group_opponent_strength"].rank(pct=True, ascending=True).mul(45)
    ).clip(35, 95).round(1)
    return squads.merge(draw, on="country", how="left")


def load_injury_status(path: Path = INJURY_STATUS_PATH) -> pd.DataFrame:
    """Load optional real injury status rows.

    Expected columns:
    player_name,country,is_injured,injury_status,expected_return_date,
    injury_severity_score,source_url,updated_at
    """

    if not path.exists():
        return pd.DataFrame()
    injuries = pd.read_csv(path)
    if injuries.empty:
        return injuries
    injuries["player_name_norm"] = injuries["player_name"].map(normalize_text)
    injuries["country_norm"] = injuries["country"].map(normalize_text)
    return injuries


def attach_injury_status(df: pd.DataFrame) -> pd.DataFrame:
    injuries = load_injury_status()
    df = df.copy()

    def coerce_bool(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    if "is_injured" not in df.columns:
        df["is_injured"] = False
    if "injury_status" not in df.columns:
        df["injury_status"] = "Available"
    if "injury_severity_score" not in df.columns:
        df["injury_severity_score"] = 0.0
    if "injury_source_url" not in df.columns:
        df["injury_source_url"] = ""
    if injuries.empty:
        df["is_injured"] = df["is_injured"].map(coerce_bool)
        df["injury_status"] = df["injury_status"].fillna("Available")
        df["injury_severity_score"] = pd.to_numeric(df["injury_severity_score"], errors="coerce").fillna(0)
        df["injury_availability_score"] = (100 - df["injury_severity_score"]).clip(0, 100).round(1)
        return df

    injury_cols = [
        "player_name_norm",
        "country_norm",
        "is_injured",
        "injury_status",
        "injury_severity_score",
        "source_url",
    ]
    injuries = injuries[[col for col in injury_cols if col in injuries.columns]].copy()
    injuries = injuries.rename(columns={"source_url": "injury_source_url"})
    df["country_norm"] = df["country"].map(normalize_text)
    df = df.merge(injuries, on=["player_name_norm", "country_norm"], how="left", suffixes=("", "_injury"))
    for col in ["is_injured", "injury_status", "injury_severity_score", "injury_source_url"]:
        injury_col = f"{col}_injury"
        if injury_col in df.columns:
            df[col] = df[injury_col].combine_first(df[col])
            df = df.drop(columns=injury_col)
    df["is_injured"] = df["is_injured"].map(coerce_bool)
    df["injury_severity_score"] = pd.to_numeric(df["injury_severity_score"], errors="coerce").fillna(0)
    df["injury_availability_score"] = (100 - df["injury_severity_score"]).clip(0, 100).round(1)
    return df


def add_real_metric_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Create dashboard-compatible scouting metrics from real available inputs."""

    df = df.copy()
    minutes_factor = (df["minutes"].clip(lower=90) / 90).clip(lower=1)
    goal_rate = df["goals"] / minutes_factor
    assist_rate = df["assists"] / minutes_factor
    market_pct = df["market_value_eur"].rank(pct=True).fillna(0.25)
    caps_pct = df["caps"].rank(pct=True).fillna(0.25)

    position_attack_bonus = df["position"].map(
        {
            "Goalkeeper": 0.1,
            "Centre-back": 0.25,
            "Full-back": 0.45,
            "Defensive midfielder": 0.45,
            "Central midfielder": 0.60,
            "Attacking midfielder": 0.85,
            "Winger": 0.95,
            "Forward": 1.00,
        }
    ).fillna(0.5)
    position_def_bonus = df["position"].map(
        {
            "Goalkeeper": 0.8,
            "Centre-back": 1.0,
            "Full-back": 0.85,
            "Defensive midfielder": 0.95,
            "Central midfielder": 0.65,
            "Attacking midfielder": 0.35,
            "Winger": 0.30,
            "Forward": 0.25,
        }
    ).fillna(0.5)

    df["non_penalty_goals"] = df["goals"]
    df["shots"] = (df["goals"] * 3.4 + position_attack_bonus * minutes_factor * 1.1).round().astype(int)
    df["shots_in_box"] = (df["shots"] * (0.45 + position_attack_bonus * 0.25)).round().astype(int)
    df["touches_in_box"] = (df["shots_in_box"] * 2.3 + goal_rate * 8).round().astype(int)
    df["key_passes"] = (df["assists"] * 3.1 + position_attack_bonus * minutes_factor * 0.55).round().astype(int)
    df["through_balls"] = (df["assists"] * 0.9 + position_attack_bonus * minutes_factor * 0.10).round().astype(int)
    df["crosses_into_box"] = (
        ((df["position"].isin(["Full-back", "Winger"])).astype(int) * minutes_factor * 0.65)
        + df["assists"] * 0.7
    ).round().astype(int)
    df["shot_creating_actions_proxy"] = (df["key_passes"] + df["assists"] * 2).round().astype(int)
    df["progressive_passes"] = ((market_pct * 3 + caps_pct * 2 + position_attack_bonus) * minutes_factor * 0.75).round().astype(int)
    df["progressive_carries"] = ((market_pct * 2 + position_attack_bonus * 2.5) * minutes_factor * 0.45).round().astype(int)
    df["passes_into_final_third"] = (df["progressive_passes"] * 0.65).round().astype(int)
    df["carries_into_final_third"] = (df["progressive_carries"] * 0.55).round().astype(int)
    df["successful_dribbles"] = ((position_attack_bonus * 1.5 + market_pct) * minutes_factor * 0.30).round().astype(int)
    df["tackles"] = ((position_def_bonus * 1.6 + caps_pct) * minutes_factor * 0.45).round().astype(int)
    df["interceptions"] = ((position_def_bonus * 1.4 + caps_pct) * minutes_factor * 0.42).round().astype(int)
    df["aerial_duels_won"] = ((position_def_bonus + (df["height_in_cm"].fillna(180) - 170).clip(0, 30) / 25) * minutes_factor * 0.32).round().astype(int)
    df["recoveries"] = ((position_def_bonus * 2.0 + caps_pct) * minutes_factor * 0.75).round().astype(int)
    df["pressures"] = ((position_def_bonus + position_attack_bonus) * minutes_factor * 1.4).round().astype(int)
    df["long_passes"] = ((df["position"].isin(["Goalkeeper", "Centre-back", "Defensive midfielder"])).astype(int) * minutes_factor * 0.85 + market_pct * minutes_factor * 0.45).round().astype(int)
    df["turnovers"] = ((1.2 - market_pct) * minutes_factor * 0.42 + position_attack_bonus * minutes_factor * 0.35).round().astype(int)
    df["saves"] = ((df["position"] == "Goalkeeper").astype(int) * minutes_factor * 2.4).round().astype(int)
    df["clean_sheets"] = ((df["position"].isin(["Goalkeeper", "Centre-back", "Full-back"])).astype(int) * minutes_factor * 0.18).round().astype(int)
    df["aerial_threat"] = df["aerial_duels_won"] + df["shots_in_box"].mul(0.12).round().astype(int)
    df["pass_completion_pct"] = (
        72
        + market_pct * 12
        + df["position"].isin(["Centre-back", "Defensive midfielder", "Central midfielder"]).astype(int) * 4
        - df["turnovers"].div(minutes_factor).clip(0, 3) * 2
    ).clip(55, 96).round(1)
    df["xg_proxy"] = (df["goals"] * 0.78 + df["shots_in_box"] * 0.05).round(2)
    df["xa_proxy"] = (df["assists"] * 0.70 + df["key_passes"] * 0.025).round(2)
    return df


def add_context_scores(df: pd.DataFrame, competitions: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tm_market_value_in_eur"] = pd.to_numeric(df.get("tm_market_value_in_eur"), errors="coerce")
    df["tm_highest_market_value_in_eur"] = pd.to_numeric(df.get("tm_highest_market_value_in_eur"), errors="coerce")
    latest_value = pd.to_numeric(df.get("latest_market_value_eur"), errors="coerce")
    salimt_latest_value = pd.to_numeric(df.get("salimt_latest_market_value_eur"), errors="coerce")
    highest_value = df["tm_highest_market_value_in_eur"]
    df["market_value_eur"] = (
        df["tm_market_value_in_eur"]
        .fillna(latest_value)
        .fillna(salimt_latest_value)
        .fillna(highest_value)
        .fillna(0)
        .astype(int)
    )
    df["market_value_source"] = np.select(
        [
            df["tm_market_value_in_eur"].notna(),
            latest_value.notna(),
            salimt_latest_value.notna(),
            highest_value.notna(),
        ],
        [
            "transfermarkt_profile",
            "transfermarkt_valuation_history",
            "salimt_latest_market_value",
            "transfermarkt_highest_market_value_fallback",
        ],
        default="missing",
    )
    df["highest_market_value_eur"] = df["tm_highest_market_value_in_eur"].fillna(df["market_value_eur"]).astype(int)
    df["height_in_cm"] = pd.to_numeric(df.get("tm_height_in_cm"), errors="coerce")

    if "tm_current_club_domestic_competition_id" not in df.columns:
        df["tm_current_club_domestic_competition_id"] = df.get("latest_value_competition_id", "")
    df["tm_current_club_domestic_competition_id"] = df["tm_current_club_domestic_competition_id"].fillna(
        df.get("latest_value_competition_id", "")
    )
    df["league"] = df["tm_current_club_domestic_competition_id"].fillna("Unknown")
    competition_names = competitions.set_index("competition_id")["name"].to_dict()
    df["league"] = df["league"].map(lambda value: competition_names.get(value, value if value else "Unknown"))
    league_codes = df.get("tm_current_club_domestic_competition_id", "").fillna("")
    df["club_level_score"] = league_codes.map(LEAGUE_TIER_SCORES).fillna(58)
    df["market_value_score"] = df["market_value_eur"].rank(pct=True).fillna(0.1).mul(100).round(1)
    df["club_level_score"] = (df["club_level_score"] * 0.65 + df["market_value_score"] * 0.35).clip(0, 100).round(1)

    rank = pd.to_numeric(df.get("tm_fifa_ranking"), errors="coerce")
    df["fifa_ranking"] = rank
    df["national_team_strength"] = (100 - ((rank.fillna(120) - 1).clip(0, 210) / 210) * 72).round(1)
    df["national_team_market_value"] = pd.to_numeric(df.get("tm_national_team_market_value"), errors="coerce").fillna(0)

    df["minutes"] = pd.to_numeric(df.get("minutes"), errors="coerce").fillna(0).astype(int)
    df["goals"] = pd.to_numeric(df.get("goals"), errors="coerce").fillna(0).astype(int)
    df["assists"] = pd.to_numeric(df.get("assists"), errors="coerce").fillna(0).astype(int)
    df["recent_appearances"] = pd.to_numeric(df.get("recent_appearances"), errors="coerce").fillna(0).astype(int)
    df["caps"] = pd.to_numeric(df["caps"], errors="coerce").fillna(0).astype(int)

    caps_score = df.groupby("country")["caps"].rank(pct=True).fillna(0.2).mul(100)
    minutes_score = df["minutes"].rank(pct=True).fillna(0.2).mul(100)
    df["expected_minutes_score"] = (caps_score * 0.55 + minutes_score * 0.35 + df["final_squad_selected"].astype(int) * 10).clip(0, 100).round(1)

    position_counts = df.groupby(["country", "position"])["player_name"].transform("count").replace(0, 1)
    scarcity_score = (100 - (position_counts - 1).clip(0, 8) * 8).clip(45, 100)
    df["tactical_fit_score"] = (df["expected_minutes_score"] * 0.55 + scarcity_score * 0.25 + df["market_value_score"] * 0.20).clip(0, 100).round(1)
    df["final_squad_selection_score"] = df["final_squad_selected"].astype(int) * 100
    df["recent_form_score"] = (
        (df["goals"] + df["assists"] * 0.8).rank(pct=True).fillna(0.1).mul(45)
        + df["minutes"].rank(pct=True).fillna(0.1).mul(35)
        + df["market_value_score"] * 0.20
    ).clip(0, 100).round(1)

    df = calculate_draw_context(df)
    df["draw_context_score"] = df["draw_context_score"].fillna(60)
    df["group"] = df["group"].fillna("")
    df["group_opponents"] = df["group_opponents"].fillna("")
    return attach_injury_status(df)


def build_real_player_dataset(include_live_actuals: bool = True) -> pd.DataFrame:
    """Build the real WC 2026 player dataset ready for preprocessing."""

    squads = fetch_world_cup_squads()
    players, _national_teams, competitions = fetch_transfermarkt_players()
    df = match_squads_to_transfermarkt(squads, players)

    matched_ids = df["tm_player_id"].dropna().astype(int).tolist() if "tm_player_id" in df.columns else []
    appearances = fetch_recent_appearances(matched_ids)
    if not appearances.empty:
        df = df.merge(appearances, left_on="tm_player_id", right_on="player_id", how="left")
        df = df.drop(columns=["player_id"], errors="ignore")
    latest_values = fetch_latest_market_values(matched_ids)
    if not latest_values.empty:
        df = df.merge(latest_values, left_on="tm_player_id", right_on="player_id", how="left")
        df = df.drop(columns=["player_id"], errors="ignore")
    salimt_values = fetch_salimt_latest_market_values(matched_ids)
    if not salimt_values.empty:
        df = df.merge(salimt_values, left_on="tm_player_id", right_on="player_id", how="left")
        df = df.drop(columns=["player_id"], errors="ignore")
    salimt_injuries = fetch_salimt_injuries(matched_ids)
    if not salimt_injuries.empty:
        df = df.merge(salimt_injuries, left_on="tm_player_id", right_on="player_id", how="left")
        df = df.drop(columns=["player_id"], errors="ignore")
    for col in ["minutes", "goals", "assists", "yellow_cards", "red_cards", "recent_appearances"]:
        if col not in df.columns:
            df[col] = 0

    if include_live_actuals and matched_ids:
        actuals = fetch_live_world_cup_actuals(matched_ids)
        if not actuals.empty:
            df = df.merge(actuals, left_on="tm_player_id", right_on="player_id", how="left")
            df = df.drop(columns=["player_id"], errors="ignore")

    for col in ["actual_wc_minutes", "actual_wc_goals", "actual_wc_assists", "actual_wc_matches"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    if "actual_success_target" not in df.columns:
        df["actual_success_target"] = pd.NA
    df["live_target_available"] = df["actual_success_target"].notna()

    df = add_context_scores(df, competitions)
    df = add_real_metric_proxies(df)
    df["data_source"] = "real_wc2026_squads_transfermarkt_openfootball"
    df["data_refresh_date"] = date.today().isoformat()
    return df


def ensure_injury_template() -> None:
    if INJURY_STATUS_PATH.exists():
        return
    INJURY_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            "player_name",
            "country",
            "is_injured",
            "injury_status",
            "expected_return_date",
            "injury_severity_score",
            "source_url",
            "updated_at",
        ]
    ).to_csv(INJURY_STATUS_PATH, index=False)


def save_real_player_data() -> pd.DataFrame:
    """Fetch real sources, engineer features, and save processed data."""

    from preprocessing import prepare_player_data

    ensure_injury_template()
    raw_real = build_real_player_dataset(include_live_actuals=True)
    REAL_SQUADS_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_real.to_csv(REAL_SQUADS_PATH, index=False)
    processed = prepare_player_data(raw_real)
    processed.to_csv(REAL_PROCESSED_DATA_PATH, index=False)

    manifest = {
        "refresh_date": date.today().isoformat(),
        "rows": int(len(processed)),
        "countries": int(processed["country"].nunique()),
        "sources": {
            "squads": WIKIPEDIA_SQUADS_URL,
            "transfermarkt_dataset": "https://github.com/dcaribou/transfermarkt-datasets",
            "market_injury_backfill_dataset": "https://github.com/salimt/football-datasets",
            "transfermarkt_csv_base": TRANSFERMARKT_DATA_BASE,
            "draw": OPENFOOTBALL_2026_URL,
            "injury_status_csv": str(INJURY_STATUS_PATH.relative_to(PROJECT_ROOT)),
        },
        "transfermarkt_match_rate": float(processed.get("transfermarkt_match", pd.Series(dtype=float)).mean()),
        "live_target_rows": int(processed.get("live_target_available", pd.Series(dtype=bool)).fillna(False).sum()),
    }
    SOURCE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return processed


if __name__ == "__main__":
    df = save_real_player_data()
    print(f"Wrote {len(df):,} real WC 2026 players to {REAL_PROCESSED_DATA_PATH}")
    print(f"Countries: {df['country'].nunique()}")
    print(f"Transfermarkt match rate: {df['transfermarkt_match'].mean():.1%}")
