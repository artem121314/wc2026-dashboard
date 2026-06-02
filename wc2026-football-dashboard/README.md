# WC 2026 Value Opportunity Dashboard

Recruitment decision-support dashboard for an English football club preparing for the 2026 FIFA World Cup.

No synthetic data is used. If real historical or tournament data is missing, the relevant model or validation section is disabled or marked as pending.

## Business Aim

Help recruitment analysts identify players to scout or buy before the 2026 World Cup because they are undervalued relative to their expected tournament impact.

## Key Decision Question

Which players offer the best value opportunity when expected World Cup impact is compared with market value, age profile, role fit, availability, risk and recent senior minutes?

## Dashboard Workflow

1. Before the tournament: use real pre-tournament player data to estimate expected World Cup impact.
2. Recruitment ranking: rank players by value opportunity.
3. During or after the tournament: ingest real World Cup match data from `data/raw/tournament_match_data.csv`.
4. Validation: compare expected impact with actual impact and classify players as Overperformed, Met expectations, Underperformed or Pending.

## Data Refresh From The Dashboard

The Streamlit sidebar includes:

- `Refresh data`: reads real CSVs from `data/raw/`, processes the dashboard dataset, updates tournament impact if match rows exist, and saves `data/processed/player_dashboard_data.csv`.
- `Clear cache and reload`: clears Streamlit caches and reloads the latest processed data.

To update actual tournament impact, add new match rows to `data/raw/tournament_match_data.csv` and click Refresh data.

## How To Refresh The Dashboard Data

1. Place real CSV files in `data/raw/`.
2. Optional: place real curated historical training data in `data/historical/`.
3. Run the app with `streamlit run app/streamlit_app.py`.
4. Click `Refresh data` in the Streamlit sidebar.
5. The app rebuilds `data/processed/player_dashboard_data.csv`.
6. If `data/raw/tournament_match_data.csv` is missing or empty, actual impact remains `Pending`.

For local development, the same refresh logic is available from the command line:

```bash
python scripts/refresh_data.py
```

No synthetic data is used. Missing real data disables only the affected model, score or validation section.

## Required Real Data Files

Expected local real-data inputs:

```text
data/raw/current_player_pool.csv
data/raw/player_performance_inputs.csv
data/raw/market_values.csv
data/raw/national_team_context.csv
data/raw/tournament_match_data.csv
data/historical/world_cup_player_training_data.csv
data/processed/player_dashboard_data.csv
```

The app runs when `data/processed/player_dashboard_data.csv` exists. If it is missing, the dashboard shows a clear warning and asks the user to add real CSV inputs and refresh.

Column-level schemas are documented in `docs/data_schema.md`. Additional source columns can be passed through for auditability, but the pipeline should not fabricate missing values.

## Historical Training Data Requirements

The supervised model trains only if `data/historical/world_cup_player_training_data.csv` contains real curated historical World Cup data with these columns:

```text
tournament_year
player_name
age
position
market_value_before_tournament
club_level_score
league_strength_score
club_minutes_previous_season
goals_previous_season
assists_previous_season
national_team_caps
expected_starter_score
national_team_strength
group_difficulty_score
injury_availability_score
recent_form_score
role_fit_score
actual_tournament_impact_score
```

Recommended historical tournaments:

- World Cup 2014
- World Cup 2018
- World Cup 2022

## Actual Tournament Data Requirements

`data/raw/tournament_match_data.csv` should contain real match-level rows:

```text
player_name
match_id
date
opponent
stage
minutes
goals
assists
shots
key_passes
tackles
interceptions
saves
clean_sheets
match_rating
```

If this file is missing or empty, `actual_tournament_impact_score` is null, `performance_delta` is null, and `performance_outcome` is Pending.

## Supervised Model

When real historical training data exists, the app trains:

- Ridge Regression
- RandomForestRegressor
- GradientBoostingRegressor

It evaluates MAE, RMSE and R-squared, selects the model with the lowest MAE, and uses it to produce:

```text
pre_tournament_expected_impact_score
expected_impact_source = supervised_historical_model
```

## If Historical Data Is Missing

The app does not train a model and does not create training rows. It returns:

```text
model_available = False
model_status = disabled_missing_real_historical_data
```

Dashboard warning:

```text
Historical World Cup training data is not available. The supervised expected-impact model is disabled until real curated historical data is added.
```

If the current player dataset has the required baseline fields, expected impact falls back to:

```text
expected_impact_source = transparent_baseline_fallback
```

If baseline fields are also missing:

```text
expected_impact_source = unavailable_missing_features
```

## Why Baseline Fallback Is Not A True ML Model

`baseline_expected_impact_score` is a transparent weighted score used only when the supervised historical model is disabled. It is useful for triage, but it has not learned relationships from historical outcomes.

The supervised expected-impact model is the actual modelling workflow. It becomes available only after real curated historical rows are added.

## Value Opportunity Score

`value_opportunity_score` is a recruitment business metric, not the supervised model. It combines:

- expected impact
- value efficiency
- age resale
- role fit
- availability
- playing-time confidence

Strategy presets:

- Value-focused club
- Balanced club
- Impact-focused club
- Development-focused club
- Custom Strategy

## Data Sources And Compliance

The dashboard is designed to ingest compliant/licensed data exports, but does not scrape restricted football data providers.

Do not scrape Opta, WhoScored, Wyscout or other commercial/restricted providers. Use manually curated CSVs, licensed exports, public compliant CSVs, or locally downloaded files the user is permitted to use.

## How To Run Locally

```bash
cd wc2026-football-dashboard
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Run project validation:

```bash
python scripts/validate_project.py
```

## Project Structure

```text
wc2026-football-dashboard/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── raw/
│   ├── historical/
│   └── processed/
├── docs/
│   └── data_schema.md
├── notebooks/
├── scripts/
│   ├── refresh_data.py
│   └── validate_project.py
├── src/
│   ├── data_loader.py
│   ├── data_pipeline.py
│   ├── data_sources.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── player_profiles.py
│   ├── preprocessing.py
│   ├── tournament_data.py
│   ├── tournament_impact.py
│   ├── utils.py
│   └── visualisations.py
├── outputs/
├── README.md
├── requirements.txt
└── .gitignore
```

## Limitations

- The supervised expected-impact model is disabled until real curated historical World Cup player data is provided.
- Actual impact validation remains pending until real 2026 World Cup match rows are added.
- Baseline expected impact is transparent triage logic, not a trained model.
- Market values and availability signals depend on the quality and refresh cadence of the real CSV inputs.

## Future Improvements

- Add a curated 2014/2018/2022 historical training dataset.
- Add uncertainty intervals around expected impact.
- Add downloadable scouting reports.
- Add club-specific positional needs and squad-depth constraints.
- Add licensed event-data support when compliant exports are available.
