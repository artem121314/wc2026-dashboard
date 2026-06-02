# WC 2026 Player Scouting & Performance Dashboard

An interactive football analytics dashboard built as a portfolio project for recruitment, scouting, and performance analysis ahead of the 2026 FIFA World Cup.

The app helps analysts explore the 2026 World Cup player pool, filter by squad context and market information, compare role-specific metrics, and estimate whether a player is likely to have a strong World Cup performance. The default processed dataset is built from real public squad, market, ranking, draw and appearance sources.

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

- Explore the 2026 World Cup final-squad player pool.
- Filter players by position, profile, country, group, club, league, age, market value, minutes, injury status, and predicted success probability.
- Compare players using percentile-adjusted football metrics.
- Rank players by role-specific profiles such as chance creator, modern full-back, high-pressing forward, and goalkeeper distributor.
- Estimate World Cup success probability using real final-squad selection, current market value, FIFA ranking context, tactical fit, injury availability, draw difficulty and recent player output.
- View prediction explanations with positive factors, negative factors, similar players, and confidence level.
- Generate short scouting-style player summaries.
- Refresh the dataset as the tournament begins so live World Cup appearances can become the model target.

## Data Sources

The default processed dataset is stored in:

```text
data/processed/player_features.csv
```

It is generated from:

- 2026 FIFA World Cup squad tables: `https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads`
- Transfermarkt dataset by dcaribou: `https://github.com/dcaribou/transfermarkt-datasets`
- Transfermarkt CSV data host: `https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data`
- openfootball 2026 World Cup schedule: `https://github.com/openfootball/worldcup.json`
- Optional real injury feed: `data/raw/injury_status.csv`

The source manifest for the last refresh is written to:

```text
data/processed/source_manifest.json
```

## Methodology

Recent minutes, goals, assists, international caps, market values, FIFA ranking context, final squad selection, tactical fit, injury availability, group draw and opponent strength are transformed into scouting features. Percentiles are calculated within position groups so defenders, midfielders, forwards, and goalkeepers are compared against relevant peers.

The dashboard calculates:

- `attacking_score`
- `creativity_score`
- `progression_score`
- `defensive_score`
- `possession_score`
- `overall_score`
- `best_profile`
- `best_profile_score`
- `tactical_fit_score`
- `injury_availability_score`
- `draw_context_score`
- `final_squad_selection_score`
- `market_value_score`
- `success_score`
- `success_probability`

The pre-tournament model is intentionally transparent:

```text
success_score =
0.18 * current_performance_score
+ 0.15 * expected_minutes_score
+ 0.14 * tactical_fit_score
+ 0.13 * role_fit_score
+ 0.12 * injury_availability_score
+ 0.10 * national_team_strength_score
+ 0.08 * draw_context_score
+ 0.05 * age_curve_score
+ 0.04 * club_level_score
+ 0.04 * recent_form_score
+ 0.03 * final_squad_selection_score
+ 0.01 * market_value_score
```

When live World Cup appearance rows are available from the Transfermarkt dataset, the model can train against an actual tournament target based on World Cup minutes and direct goal contribution. Before enough live target rows exist, the transparent pre-tournament target keeps the dashboard usable.

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

The first run loads the processed real-data feature file. To regenerate it:

```bash
python src/data_loader.py
```

To refresh the real World Cup dataset directly:

```bash
python src/real_data.py
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
│   ├── real_data.py
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

## Future Improvements

- Train against historical World Cup and continental tournament outcomes.
- Add uncertainty intervals around success predictions.
- Add downloadable scouting reports.
- Add pitch-style visuals for ball progression, shot maps, and defensive actions.
