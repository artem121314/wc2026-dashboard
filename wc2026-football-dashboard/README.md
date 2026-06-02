# WC 2026 Player Scouting & Performance Dashboard

An interactive football analytics dashboard built as a portfolio project for recruitment, scouting, and performance analysis ahead of the 2026 FIFA World Cup.

The app helps analysts explore player profiles, filter a scouting pool, compare football-specific metrics, and estimate whether a player is likely to have a strong World Cup performance. The first version ships with a clean synthetic sample dataset so the full workflow runs immediately and can later be replaced with real player data.

## Why This Project Exists

Recruitment and national-team scouting workflows often need a fast way to move from a large player pool to a role-specific shortlist. This project demonstrates how to combine football domain logic, feature engineering, transparent modelling, and an interactive dashboard into a tool that feels useful for a club data analyst or recruitment analyst.

## Dashboard Screenshots

Screenshots can be added here after running the Streamlit app locally.

- Overview page
- Player search page
- Player profile page
- Profile ranking page
- Prediction explanation page

## Key Features

- Explore a sample pool of 260 World Cup-relevant player profiles.
- Filter players by position, profile, country, club, league, age, market value, minutes, and predicted success probability.
- Compare players using percentile-adjusted football metrics.
- Rank players by role-specific profiles such as chance creator, modern full-back, high-pressing forward, and goalkeeper distributor.
- Estimate World Cup success probability using a transparent scouting-support heuristic.
- View prediction explanations with positive factors, negative factors, similar players, and confidence level.
- Generate short scouting-style player summaries.

## Data Sources

The MVP uses a reproducible synthetic dataset stored in:

```text
data/sample/wc2026_sample_players.csv
```

The project is designed so this file can be replaced with real data later. The expected columns include player identity fields, market value, playing time, event-style metrics, national-team strength, club-level score, recent form, and expected minutes.

Future real-data sources could include:

- StatsBomb Open Data
- FBref / StatBomb-style standard and shooting tables
- Transfermarkt-style market value data
- Club or national-team scouting databases
- Historical FIFA World Cup player and match data

## Methodology

Volume metrics are converted to per-90 rates before percentile ranking. Percentiles are calculated within position groups so defenders, midfielders, forwards, and goalkeepers are compared against relevant peers.

The dashboard calculates:

- `attacking_score`
- `creativity_score`
- `progression_score`
- `defensive_score`
- `possession_score`
- `overall_score`
- `best_profile`
- `best_profile_score`
- `success_score`
- `success_probability`

The first success model is intentionally transparent. Since real 2026 tournament outcomes do not exist yet, the MVP uses a heuristic score:

```text
success_score =
0.25 * current_performance_score
+ 0.20 * national_team_strength_score
+ 0.15 * expected_minutes_score
+ 0.15 * role_fit_score
+ 0.10 * age_curve_score
+ 0.10 * club_level_score
+ 0.05 * recent_form_score
```

The score is mapped to a probability-like 0-100 output for scouting triage. It is not a betting model.

## Player Profiles

The project includes weighted scoring systems for:

- Ball-progressing midfielder
- Chance creator
- Box threat forward
- High-volume winger
- Defensive midfielder
- Modern full-back
- Ball-playing centre-back
- High-pressing forward
- Goalkeeper distributor

Profile definitions live in:

```text
src/player_profiles.py
```

## How To Run Locally

```bash
cd wc2026-football-dashboard
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The first run loads the sample data and processed feature file. To regenerate them:

```bash
python src/data_loader.py
```

## Project Structure

```text
wc2026-football-dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_dashboard_prototype.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── player_profiles.py
│   ├── model.py
│   ├── visualisations.py
│   └── utils.py
├── outputs/
│   ├── charts/
│   ├── model/
│   └── reports/
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations

- The current dataset is synthetic and intended for MVP development.
- The success target is heuristic rather than learned from real World Cup outcomes.
- The model does not account for injuries, final squad selection, tactical fit, opponent strength, or tournament draw.
- Market values and team-strength inputs are estimates for demonstration.

## Future Improvements

- Replace the sample data with real player-season and event data.
- Train against historical World Cup and continental tournament outcomes.
- Add injury history, squad competition, and minutes projection models.
- Add uncertainty intervals around success predictions.
- Add downloadable scouting reports.
- Add pitch-style visuals for ball progression, shot maps, and defensive actions.

