"""Data loading and synthetic fallback generation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "sample" / "wc2026_sample_players.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "player_features.csv"


COUNTRY_STRENGTH = {
    "Argentina": 92,
    "Brazil": 91,
    "France": 94,
    "England": 90,
    "Spain": 89,
    "Germany": 88,
    "Portugal": 87,
    "Netherlands": 86,
    "Uruguay": 82,
    "Croatia": 81,
    "Morocco": 79,
    "Colombia": 78,
    "Switzerland": 78,
    "Denmark": 77,
    "Senegal": 77,
    "USA": 76,
    "Japan": 75,
    "Nigeria": 74,
    "Mexico": 74,
    "Norway": 74,
    "South Korea": 73,
    "Ghana": 70,
    "Canada": 70,
    "Ecuador": 72,
    "Australia": 69,
    "Serbia": 76,
    "Poland": 74,
    "Austria": 75,
    "Cameroon": 70,
    "Egypt": 72,
    "Turkey": 74,
}


CLUBS_BY_LEAGUE = {
    "Premier League": [
        ("Manchester City", 96),
        ("Arsenal", 92),
        ("Liverpool", 91),
        ("Chelsea", 84),
        ("Tottenham", 84),
        ("Manchester United", 83),
        ("Aston Villa", 80),
        ("Newcastle United", 79),
    ],
    "LaLiga": [
        ("Real Madrid", 97),
        ("Barcelona", 92),
        ("Atletico Madrid", 86),
        ("Real Sociedad", 78),
        ("Villarreal", 76),
        ("Athletic Club", 75),
    ],
    "Bundesliga": [
        ("Bayern Munich", 95),
        ("Bayer Leverkusen", 89),
        ("Borussia Dortmund", 86),
        ("RB Leipzig", 84),
        ("Eintracht Frankfurt", 75),
    ],
    "Serie A": [
        ("Inter Milan", 89),
        ("AC Milan", 84),
        ("Juventus", 83),
        ("Napoli", 82),
        ("Atalanta", 79),
        ("Roma", 76),
    ],
    "Ligue 1": [
        ("Paris Saint-Germain", 91),
        ("Monaco", 78),
        ("Lyon", 73),
        ("Marseille", 75),
        ("Lille", 74),
    ],
    "Primeira Liga": [
        ("Benfica", 80),
        ("Porto", 78),
        ("Sporting CP", 79),
        ("Braga", 69),
    ],
    "Eredivisie": [
        ("Ajax", 75),
        ("PSV", 76),
        ("Feyenoord", 74),
        ("AZ Alkmaar", 67),
    ],
    "MLS": [
        ("Inter Miami", 66),
        ("LAFC", 63),
        ("Seattle Sounders", 61),
        ("Atlanta United", 60),
    ],
    "Brasileirao": [
        ("Flamengo", 70),
        ("Palmeiras", 71),
        ("Fluminense", 66),
        ("Sao Paulo", 64),
    ],
    "Saudi Pro League": [
        ("Al Hilal", 72),
        ("Al Nassr", 72),
        ("Al Ittihad", 68),
        ("Al Ahli", 68),
    ],
}


POSITION_WEIGHTS = {
    "Goalkeeper": 0.09,
    "Centre-back": 0.16,
    "Full-back": 0.14,
    "Defensive midfielder": 0.12,
    "Central midfielder": 0.15,
    "Attacking midfielder": 0.12,
    "Winger": 0.13,
    "Forward": 0.09,
}


POSITION_RATES = {
    "Goalkeeper": {
        "goals": 0.00,
        "assists": 0.01,
        "shots": 0.02,
        "key_passes": 0.03,
        "progressive_passes": 4.4,
        "progressive_carries": 0.15,
        "passes_into_final_third": 1.1,
        "carries_into_final_third": 0.02,
        "successful_dribbles": 0.02,
        "tackles": 0.05,
        "interceptions": 0.10,
        "aerial_duels_won": 0.20,
        "turnovers": 0.28,
        "through_balls": 0.02,
        "crosses_into_box": 0.00,
        "shot_creating_actions_proxy": 0.10,
        "shots_in_box": 0.00,
        "touches_in_box": 0.05,
        "pressures": 0.4,
        "recoveries": 3.2,
        "long_passes": 8.2,
        "saves": 3.0,
        "clean_sheets": 0.34,
    },
    "Centre-back": {
        "goals": 0.04,
        "assists": 0.03,
        "shots": 0.55,
        "key_passes": 0.14,
        "progressive_passes": 4.5,
        "progressive_carries": 0.75,
        "passes_into_final_third": 3.4,
        "carries_into_final_third": 0.35,
        "successful_dribbles": 0.08,
        "tackles": 1.35,
        "interceptions": 1.65,
        "aerial_duels_won": 3.15,
        "turnovers": 0.50,
        "through_balls": 0.05,
        "crosses_into_box": 0.02,
        "shot_creating_actions_proxy": 0.25,
        "shots_in_box": 0.35,
        "touches_in_box": 0.95,
        "pressures": 5.5,
        "recoveries": 6.2,
        "long_passes": 5.8,
        "saves": 0.0,
        "clean_sheets": 0.30,
    },
    "Full-back": {
        "goals": 0.04,
        "assists": 0.14,
        "shots": 0.75,
        "key_passes": 0.90,
        "progressive_passes": 3.7,
        "progressive_carries": 2.7,
        "passes_into_final_third": 2.9,
        "carries_into_final_third": 1.4,
        "successful_dribbles": 0.65,
        "tackles": 2.05,
        "interceptions": 1.25,
        "aerial_duels_won": 1.05,
        "turnovers": 1.05,
        "through_balls": 0.16,
        "crosses_into_box": 2.15,
        "shot_creating_actions_proxy": 1.15,
        "shots_in_box": 0.28,
        "touches_in_box": 1.45,
        "pressures": 12.5,
        "recoveries": 6.8,
        "long_passes": 2.9,
        "saves": 0.0,
        "clean_sheets": 0.29,
    },
    "Defensive midfielder": {
        "goals": 0.05,
        "assists": 0.08,
        "shots": 0.85,
        "key_passes": 0.65,
        "progressive_passes": 4.8,
        "progressive_carries": 1.45,
        "passes_into_final_third": 3.5,
        "carries_into_final_third": 0.85,
        "successful_dribbles": 0.45,
        "tackles": 2.45,
        "interceptions": 2.05,
        "aerial_duels_won": 1.35,
        "turnovers": 0.85,
        "through_balls": 0.15,
        "crosses_into_box": 0.20,
        "shot_creating_actions_proxy": 0.95,
        "shots_in_box": 0.22,
        "touches_in_box": 0.85,
        "pressures": 13.5,
        "recoveries": 7.7,
        "long_passes": 4.2,
        "saves": 0.0,
        "clean_sheets": 0.28,
    },
    "Central midfielder": {
        "goals": 0.09,
        "assists": 0.13,
        "shots": 1.25,
        "key_passes": 1.00,
        "progressive_passes": 4.2,
        "progressive_carries": 2.05,
        "passes_into_final_third": 3.2,
        "carries_into_final_third": 1.2,
        "successful_dribbles": 0.85,
        "tackles": 1.75,
        "interceptions": 1.35,
        "aerial_duels_won": 0.80,
        "turnovers": 1.05,
        "through_balls": 0.25,
        "crosses_into_box": 0.35,
        "shot_creating_actions_proxy": 1.35,
        "shots_in_box": 0.45,
        "touches_in_box": 1.25,
        "pressures": 13.0,
        "recoveries": 6.6,
        "long_passes": 3.4,
        "saves": 0.0,
        "clean_sheets": 0.24,
    },
    "Attacking midfielder": {
        "goals": 0.18,
        "assists": 0.23,
        "shots": 2.15,
        "key_passes": 1.85,
        "progressive_passes": 2.6,
        "progressive_carries": 3.05,
        "passes_into_final_third": 2.15,
        "carries_into_final_third": 1.65,
        "successful_dribbles": 1.65,
        "tackles": 1.10,
        "interceptions": 0.55,
        "aerial_duels_won": 0.45,
        "turnovers": 1.75,
        "through_balls": 0.48,
        "crosses_into_box": 0.75,
        "shot_creating_actions_proxy": 2.45,
        "shots_in_box": 0.95,
        "touches_in_box": 2.85,
        "pressures": 12.8,
        "recoveries": 4.5,
        "long_passes": 1.9,
        "saves": 0.0,
        "clean_sheets": 0.18,
    },
    "Winger": {
        "goals": 0.23,
        "assists": 0.20,
        "shots": 2.55,
        "key_passes": 1.45,
        "progressive_passes": 1.95,
        "progressive_carries": 4.5,
        "passes_into_final_third": 1.65,
        "carries_into_final_third": 2.25,
        "successful_dribbles": 2.65,
        "tackles": 0.90,
        "interceptions": 0.45,
        "aerial_duels_won": 0.55,
        "turnovers": 2.20,
        "through_balls": 0.24,
        "crosses_into_box": 1.95,
        "shot_creating_actions_proxy": 2.05,
        "shots_in_box": 1.15,
        "touches_in_box": 3.95,
        "pressures": 13.8,
        "recoveries": 4.2,
        "long_passes": 1.25,
        "saves": 0.0,
        "clean_sheets": 0.17,
    },
    "Forward": {
        "goals": 0.43,
        "assists": 0.12,
        "shots": 3.25,
        "key_passes": 0.78,
        "progressive_passes": 1.05,
        "progressive_carries": 1.65,
        "passes_into_final_third": 0.75,
        "carries_into_final_third": 0.95,
        "successful_dribbles": 0.95,
        "tackles": 0.55,
        "interceptions": 0.22,
        "aerial_duels_won": 1.75,
        "turnovers": 1.65,
        "through_balls": 0.12,
        "crosses_into_box": 0.18,
        "shot_creating_actions_proxy": 1.10,
        "shots_in_box": 2.25,
        "touches_in_box": 5.2,
        "pressures": 14.5,
        "recoveries": 3.2,
        "long_passes": 0.75,
        "saves": 0.0,
        "clean_sheets": 0.12,
    },
}


FIRST_NAMES = [
    "Lucas",
    "Mateo",
    "Julian",
    "Gabriel",
    "Rafael",
    "Thiago",
    "Enzo",
    "Nico",
    "Leo",
    "Bruno",
    "Joao",
    "Hugo",
    "Theo",
    "Ousmane",
    "Amadou",
    "Youssef",
    "Sofiane",
    "Kylian",
    "Jude",
    "Declan",
    "Mason",
    "Cole",
    "Lamine",
    "Pedri",
    "Jamal",
    "Florian",
    "Kai",
    "Xavi",
    "Santiago",
    "Federico",
    "Tyler",
    "Weston",
    "Alphonso",
    "Daichi",
    "Take",
    "Min-jae",
    "Victor",
    "Mohamed",
]


LAST_NAMES = [
    "Alvarez",
    "Silva",
    "Santos",
    "Pereira",
    "Martinez",
    "Garcia",
    "Rodriguez",
    "Fernandez",
    "Diaz",
    "Torres",
    "Costa",
    "Ramos",
    "Kone",
    "Diop",
    "Sarr",
    "Hakimi",
    "Benali",
    "Mbaye",
    "Smith",
    "Walker",
    "Rice",
    "Palmer",
    "Williams",
    "Musiala",
    "Wirtz",
    "Muller",
    "De Jong",
    "Van Dijk",
    "Valverde",
    "Pulisic",
    "Davies",
    "Tanaka",
    "Kim",
    "Osimhen",
    "Salah",
]


def _weighted_choice(items: Iterable[str], weights: Iterable[float], rng: np.random.Generator) -> str:
    values = list(items)
    probs = np.array(list(weights), dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(values, p=probs))


def _unique_player_names(n_players: int, rng: np.random.Generator) -> list[str]:
    seen: dict[str, int] = {}
    names: list[str] = []
    for _ in range(n_players):
        base = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        names.append(base if count == 0 else f"{base} {count + 1}")
    return names


def _sample_metric(rate: float, minutes: int, talent: float, rng: np.random.Generator) -> float:
    exposure = max(minutes / 90, 0.1)
    mean = max(rate * exposure * talent, 0.0)
    if mean <= 0.02:
        return 0.0
    return float(rng.poisson(mean))


def generate_sample_players(n_players: int = 260, random_state: int = 26) -> pd.DataFrame:
    """Create a realistic, reproducible fallback fixture."""

    rng = np.random.default_rng(random_state)
    player_names = _unique_player_names(n_players, rng)
    countries = list(COUNTRY_STRENGTH)
    country_weights = np.array(list(COUNTRY_STRENGTH.values()), dtype=float) ** 1.4

    records: list[dict[str, object]] = []
    positions = list(POSITION_WEIGHTS)
    position_weights = list(POSITION_WEIGHTS.values())
    leagues = list(CLUBS_BY_LEAGUE)
    league_weights = np.array([1.25, 1.05, 0.95, 0.9, 0.75, 0.5, 0.45, 0.35, 0.35, 0.25])
    league_weights = league_weights / league_weights.sum()

    for player_name in player_names:
        country = _weighted_choice(countries, country_weights, rng)
        national_strength = COUNTRY_STRENGTH[country] + rng.normal(0, 3)
        position = _weighted_choice(positions, position_weights, rng)
        league = str(rng.choice(leagues, p=league_weights))
        club, club_level = CLUBS_BY_LEAGUE[league][rng.integers(0, len(CLUBS_BY_LEAGUE[league]))]

        age = int(np.clip(round(rng.normal(25.5, 4.2)), 17, 37))
        minutes = int(np.clip(round(rng.normal(2050, 680)), 240, 3800))
        if rng.random() < 0.11:
            minutes = int(np.clip(round(rng.normal(850, 270)), 180, 1500))

        base_talent = np.clip(rng.normal(1.0, 0.22), 0.55, 1.65)
        environment = (club_level / 75) * (national_strength / 78)
        talent = float(np.clip(base_talent * (0.86 + 0.18 * environment), 0.50, 1.85))
        rates = POSITION_RATES[position]

        row: dict[str, object] = {
            "player_name": player_name,
            "country": country,
            "club": club,
            "league": league,
            "age": age,
            "position": position,
            "minutes": minutes,
            "national_team_strength": round(float(np.clip(national_strength, 45, 98)), 1),
            "club_level_score": round(float(np.clip(club_level + rng.normal(0, 3), 45, 98)), 1),
            "recent_form_score": round(float(np.clip(58 + talent * 22 + rng.normal(0, 9), 30, 98)), 1),
            "expected_minutes_score": round(
                float(np.clip((minutes / 3200) * 82 + (national_strength - 70) * 0.35 + rng.normal(0, 7), 18, 98)),
                1,
            ),
        }

        for metric, rate in rates.items():
            if metric in {"clean_sheets"}:
                row[metric] = int(np.clip(_sample_metric(rate, minutes, talent, rng), 0, max(minutes // 90, 1)))
            elif metric == "saves":
                row[metric] = int(_sample_metric(rate, minutes, 2.0 - min(talent, 1.6) * 0.25, rng))
            else:
                row[metric] = int(_sample_metric(rate, minutes, talent, rng))

        if position != "Goalkeeper":
            pass_base = {
                "Centre-back": 88,
                "Defensive midfielder": 86,
                "Central midfielder": 84,
                "Full-back": 81,
                "Attacking midfielder": 80,
                "Winger": 77,
                "Forward": 76,
            }[position]
        else:
            pass_base = 76

        row["pass_completion_pct"] = round(
            float(np.clip(rng.normal(pass_base + talent * 3 - row["turnovers"] / max(minutes / 90, 1) * 1.5, 3.4), 58, 96)),
            1,
        )

        goals = int(row["goals"])
        row["non_penalty_goals"] = max(0, goals - int(rng.random() < 0.18 and goals > 4))
        row["aerial_threat"] = int(row["aerial_duels_won"]) + int(row["shots_in_box"] * rng.uniform(0.15, 0.35))

        age_value = np.interp(age, [17, 21, 25, 29, 33, 37], [0.75, 1.35, 1.45, 1.12, 0.72, 0.35])
        position_value = {
            "Goalkeeper": 9,
            "Centre-back": 13,
            "Full-back": 12,
            "Defensive midfielder": 14,
            "Central midfielder": 16,
            "Attacking midfielder": 18,
            "Winger": 20,
            "Forward": 22,
        }[position]
        minutes_factor = np.clip(minutes / 2200, 0.32, 1.18)
        value_m = position_value * (talent**2.25) * (club_level / 76) * age_value * minutes_factor
        value_m *= float(rng.lognormal(mean=0.0, sigma=0.22))
        row["market_value_eur"] = int(np.clip(value_m, 0.7, 185) * 1_000_000)

        records.append(row)

    return pd.DataFrame(records)


def save_sample_data(path: Path = SAMPLE_DATA_PATH, n_players: int = 260) -> pd.DataFrame:
    """Generate and write the sample dataset."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_players(n_players=n_players)
    df.to_csv(path, index=False)
    return df


def load_raw_player_data(path: Path | str = SAMPLE_DATA_PATH) -> pd.DataFrame:
    """Load sample data, generating it if missing."""

    path = Path(path)
    if not path.exists():
        return save_sample_data(path)
    return pd.read_csv(path)


def load_player_data(
    sample_path: Path | str = SAMPLE_DATA_PATH,
    processed_path: Path | str = PROCESSED_DATA_PATH,
    refresh: bool = False,
    use_real_sources: bool = False,
) -> pd.DataFrame:
    """Load processed data or build it from real sources/sample fallback."""

    processed_path = Path(processed_path)
    if processed_path.exists() and not refresh:
        return pd.read_csv(processed_path)

    if use_real_sources:
        from real_data import save_real_player_data

        return save_real_player_data()

    from preprocessing import prepare_player_data

    raw_df = load_raw_player_data(sample_path)
    processed_df = prepare_player_data(raw_df)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(processed_path, index=False)
    return processed_df


if __name__ == "__main__":
    load_player_data(refresh=True, use_real_sources=True)
    print(f"Wrote real processed data to {PROCESSED_DATA_PATH}")
