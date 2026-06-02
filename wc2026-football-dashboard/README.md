# WC 2026 Player Scouting & Performance Dashboard

An interactive football analytics dashboard built as a portfolio project for recruitment, scouting, and performance analysis ahead of the 2026 FIFA World Cup.

The app helps analysts explore the 2026 World Cup player pool, filter by squad context and market information, compare role-specific metrics, predict expected tournament impact, and identify value opportunities before the tournament. Once match data is available, the dashboard compares predicted impact against actual World Cup performance.

## Why This Project Exists

Recruitment teams often want to find players before a major tournament changes the market. This project is framed as a pre-tournament analytics workflow for an English club: predict expected World Cup impact, rank undervalued players by recruitment opportunity, then validate whether the prediction was confirmed by actual tournament output.

## Dashboard Screenshots

Screenshots can be added here after running the Streamlit app locally.

- Overview page
- Recruitment shortlist page
- Player profile page
- Profile ranking page
- Expected impact model page

## Key Features

- Explore the 2026 World Cup final-squad player pool.
- Filter players by position, profile, country, group, club, league, age, market value, expected impact, value opportunity, and performance outcome.
- Compare players using percentile-adjusted football metrics.
- Rank players by role-specific profiles such as chance creator, modern full-back, high-pressing forward, and goalkeeper distributor.
- Predict `pre_tournament_expected_impact_score` using a historical supervised model when training data is available.
- Use `baseline_expected_impact_score` only as a transparent fallback.
- Rank players by `value_opportunity_score` with strategy presets for value-focused, balanced and impact-focused clubs.
- Compare predicted impact with `actual_tournament_impact_score` once live World Cup match data is available.
- Classify players as Pending, Overperformed, Met expectations or Underperformed.
- View explanation factors, similar players, feature importance and model evaluation metrics.
- Generate short scouting-style player summaries.
- Refresh the dataset as the tournament begins so actual World Cup impact can be calculated.

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

Historical model training data should be collected in:

```text
data/historical/world_cup_player_training_data.csv
```

The placeholder file is committed with headers only. It is not fake historical data.

## Methodology

Recent minutes, goals, assists, international caps, market values, FIFA ranking context, final squad selection, tactical fit, injury availability, group draw and opponent strength are transformed into pre-tournament features. Percentiles are calculated within position groups so defenders, midfielders, forwards, and goalkeepers are compared against relevant peers.

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
- `baseline_expected_impact_score`
- `pre_tournament_expected_impact_score`
- `actual_tournament_impact_score`
- `performance_delta`
- `performance_outcome`
- `value_efficiency_score`
- `value_opportunity_score`

## Historical Model Design

The intended supervised model trains on previous World Cups:

- World Cup 2014
- World Cup 2018
- World Cup 2022

Each row represents one player before a tournament. Example features include age, position, market value before the tournament, club and league strength, previous-season minutes, goals, assists, national-team caps, expected starter score, national-team strength, group difficulty, injury availability, recent form and role fit.

The target is:

```text
actual_tournament_impact_score
```

The app trains Ridge Regression, Random Forest and Gradient Boosting regressors with scikit-learn and evaluates MAE, RMSE and R-squared. The default selected model is Gradient Boosting when historical training data is available.

If the historical training file has no rows, the app does not fail. It shows:

```text
Historical training data is not available yet. The dashboard is currently using a transparent baseline expected-impact score.
```

## Baseline Fallback

The manual weighted score is a transparent fallback only:

```text
baseline_expected_impact_score =
0.25 * current_performance_score
+ 0.20 * expected_minutes_score
+ 0.15 * role_fit_score
+ 0.15 * national_team_context_score
+ 0.10 * tournament_draw_score
+ 0.10 * availability_score
+ 0.05 * age_upside_score
```

## Actual Tournament Impact

`actual_tournament_impact_score` is a 0-100 position-adjusted score calculated from World Cup match data when available. The framework supports tournament minutes, starts, goals, assists, shots, key passes, tackles, interceptions, saves, clean sheets and match ratings.

If match data is not available yet, `actual_tournament_impact_score` is null and `performance_outcome` is Pending.

```text
performance_delta = actual_tournament_impact_score - pre_tournament_expected_impact_score
```

Outcomes:

- Pending
- Overperformed
- Met expectations
- Underperformed

## Value Opportunity

`value_opportunity_score` is a business ranking metric, not a pure prediction. It combines expected impact, market-value efficiency, resale age profile, role fit and availability.

```text
value_opportunity_score =
0.40 * pre_tournament_expected_impact_score
+ 0.25 * value_efficiency_score
+ 0.15 * age_resale_score
+ 0.10 * role_fit_score
+ 0.10 * availability_score
```

The Streamlit app includes strategy presets:

- Value-focused club
- Balanced club
- Impact-focused club

## Why This Is Not Just Arbitrary Weighting

Expected impact is intended to be model-based. Historical World Cup data is used to learn relationships between pre-tournament features and tournament impact. Manual scores are only used as a fallback until historical data is available. The model is validated by comparing `pre_tournament_expected_impact_score` against `actual_tournament_impact_score` once the World Cup is in progress.

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
│   ├── historical/
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
│   ├── tournament_impact.py
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
