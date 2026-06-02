# World Cup 2026 Recruitment Intelligence Dashboard

Recruitment intelligence for identifying high-upside World Cup targets before the tournament and validating outcomes during the competition.

No synthetic data is used. If real historical or tournament data is missing, the relevant model or validation section is disabled or marked as not yet available.

Product naming options considered:

- World Cup 2026 Recruitment Intelligence Dashboard
- WC 2026 Scouting Intelligence Hub
- World Cup 2026 Player Opportunity Intelligence
- WC 2026 Recruitment Opportunity Hub

The implemented default is **World Cup 2026 Recruitment Intelligence Dashboard** because it best matches the business-facing recruitment workflow.

## Business Aim

Help recruitment analysts identify players to scout or buy before the 2026 World Cup because they are undervalued relative to their expected tournament impact.

## Key Decision Question

Which players offer the best value opportunity when expected World Cup impact is compared with market value, age profile, role fit, availability, risk and recent senior minutes?

## Dashboard Workflow

1. Before the tournament: use real pre-tournament player data to estimate expected World Cup impact.
2. Recruitment ranking: rank players by value opportunity.
3. Breakout analysis: identify young value-opportunity players who could use the tournament as a breakout platform.
4. During or after the tournament: ingest real World Cup match data from `data/raw/tournament_match_data.csv`.
5. Validation: compare expected impact with actual impact and classify players as Overperformed, Met expectations, Underperformed or Pending.

The first app view is an executive dashboard for recruitment decisions. Detailed source availability, refresh diagnostics, model readiness, training metrics and feature coverage live in the **Data & Model Ops** tab.

## Expected WC Impact

Expected WC Impact estimates how strongly a player is expected to influence the 2026 World Cup based on the active historical model. It is a tournament-impact score, not an overall player-quality rating.

Interpretation bands:

- 0-30: Low
- 30-50: Moderate
- 50-70: Strong
- 70-85: High
- 85+: Elite

The current supervised MVP is the limited historical model. It is trained on real World Cup-derived historical player-tournament rows and actual tournament impact outcomes. It does not yet include rich club-form, market, scouting or injury inputs, so the dashboard labels that limitation clearly.

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
data/historical/world_cup_player_training_data_template.csv
data/historical/world_cup_player_training_data.csv
data/processed/player_dashboard_data.csv
```

The app runs when `data/processed/player_dashboard_data.csv` exists. If it is missing, the dashboard shows a clear warning and asks the user to add real CSV inputs and refresh.

Column-level schemas are documented in `docs/data_schema.md`. Additional source columns can be passed through for auditability, but the pipeline should not fabricate missing values.

## Historical Training Data Requirements

The supervised model trains only if `data/historical/world_cup_player_training_data.csv` contains real curated historical World Cup data with these columns:

Each historical row should be one player-tournament observation:

```text
player_name + tournament_year
```

Examples:

- Player A before World Cup 2014, then actual impact at World Cup 2014.
- Player B before World Cup 2018, then actual impact at World Cup 2018.
- Player C before World Cup 2022, then actual impact at World Cup 2022.

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

Optional historical modelling features:

```text
is_world_cup_debutant
previous_world_cup_minutes
previous_world_cup_matches
previous_world_cup_impact_score
senior_national_team_caps
major_tournament_experience
age_group
```

These optional fields can improve the model when real curated values exist, but they are not required for model training.

See `docs/historical_data_collection_plan.md` for a practical collection workflow and compliant source guidance.

Recommended historical tournaments:

- World Cup 2014
- World Cup 2018
- World Cup 2022

## Supervised Expected-Impact Model Modes

The project now supports three honest model modes:

1. `limited_historical_context_model`

   MVP supervised model trained on real World Cup-derived historical features only:

   - age
   - position
   - previous World Cup minutes
   - previous World Cup matches
   - previous World Cup impact score when available
   - debutant status
   - group difficulty proxy when available
   - tournament year

   This is real supervised learning on historical player-tournament rows and `actual_tournament_impact_score`, but it is deliberately labelled as limited because rich recruitment predictors are not yet available historically.

2. `full_recruitment_model`

   Preferred future model once enriched real pre-tournament predictors are available:

   - market values
   - club-season minutes, goals and assists
   - club and league strength
   - senior caps
   - availability, recent form and role fit

3. `baseline_fallback`

   Transparent weighted baseline used only if neither supervised model can train.

The model uses sklearn pipelines with `SimpleImputer`, `OneHotEncoder`, and model-specific preprocessing. Imputation happens only inside the model pipeline for missing real values; it does not write fabricated values into source CSVs.

## How To Enable The Full Recruitment Model

The current project can train the limited historical context model from collected real World Cup data. To enable the full recruitment model:

To enable the supervised model:

1. Fill real predictor values in `data/historical/manual_enrichment_template.csv`, or use another compliant local enrichment CSV.
2. Apply the values without fabricating missing data:

```bash
python scripts/apply_manual_historical_enrichment.py --input data/historical/manual_enrichment_template.csv
```

3. Keep one row per `player_name + tournament_year` in `data/historical/world_cup_player_training_data.csv`.
4. Leave unavailable fields blank.
5. Run:

```bash
python scripts/check_model_readiness.py
python scripts/validate_project.py
```

6. If readiness passes, refresh the dashboard data:

```bash
python scripts/refresh_data.py
```

7. Run the app. The Data & Model Ops tab will show whether the limited or full supervised model is active.

No synthetic rows are used. If target rows are missing, the model is disabled because historical target data is missing. If rich predictor coverage is too thin, the app uses the limited supervised historical model where possible and clearly labels it as an MVP model.

## Historical Data Collection

The project includes a real-data collection script for the first historical target table:

```bash
python scripts/collect_historical_data.py
```

Current implemented source:

- [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup), accessed through the [DataHub World Cup CSV dataset](https://datahub.io/football/worldcup)
- License: CC-BY-SA 4.0, with attribution and share-alike requirements
- Fields collected: squads, player birth dates, player appearances, starts, goals, substitutions, bookings, matches, clean-sheet context, tournament standings and group standings
- Years collected: World Cup 2014, 2018 and 2022

The collector writes:

```text
data/historical/world_cup_player_training_data.csv
data/historical/source_audit_log.csv
```

The first `actual_tournament_impact_score` is an MVP target built only from real tournament output. Outfield players are scored from tournament minutes, starts and goals. Goalkeepers are scored from tournament minutes, starts and clean sheets. Scores are percentile-adjusted within broad position groups and scaled 0-100.

The collector also derives previous World Cup experience from real earlier World Cup records:

- `previous_world_cup_minutes`: cumulative prior World Cup minutes by player ID.
- `previous_world_cup_matches`: cumulative prior World Cup appearances by player ID.
- `previous_world_cup_impact_score`: most recent prior World Cup impact score where a prior row exists.
- `is_world_cup_debutant`: `True` only when previous World Cup appearances equal zero.

Debutant status is not guessed from age or reputation.

The historical file includes a simple real-data-derived `national_team_strength` proxy based on the team's most recent previous World Cup final standing. Higher scores mean stronger previous World Cup finish. This is not FIFA ranking. `group_difficulty_score` is calculated as the average of the other group teams' previous World Cup strength proxies when those opponent scores are available.

Fields still missing from the open source include club, league, assists, pre-tournament market value, club-season minutes, club-season goals/assists, senior national-team caps, injury availability and tactical role fit. These remain null and are not fabricated.

Because rich recruitment predictors are still not populated, the full recruitment model is not available yet. The limited historical context model can still train on the real World Cup-derived features. To enrich toward the full model, join or manually curate compliant real pre-tournament feature exports into `data/historical/world_cup_player_training_data.csv`, then run:

```bash
python scripts/check_model_readiness.py
python scripts/validate_project.py
python scripts/refresh_data.py
```

The source audit log records every dataset used and rejected. Restricted providers such as Opta, WhoScored, Wyscout and direct Transfermarkt page scraping are not used.

## Manual Historical Predictor Enrichment

No reliable compliant public source has been joined automatically for historical club, league and pre-tournament market context yet. The project therefore includes a manual enrichment workflow:

```text
data/historical/manual_enrichment_template.csv
docs/manual_historical_enrichment.md
scripts/apply_manual_historical_enrichment.py
```

Copy the template to `data/historical/manual_enrichment.csv`, fill only values supported by compliant real sources, then run:

```bash
python scripts/apply_manual_historical_enrichment.py
python scripts/check_model_readiness.py
python scripts/validate_project.py
```

The enrichment script uses exact `tournament_year + player_name` matching only. It rejects duplicate keys, requires provenance for supplied values, updates only non-null fields, and does not fuzzy-match or fabricate missing values.

`scripts/check_model_readiness.py` reports which mode can train, which features are used, and which features are excluded due to missingness, low coverage or no variation.

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

The model is trained on player-tournament observations, not on repeated appearances by the same players. This lets it generalise from historical player profiles to new 2026 players, including players who have never appeared at a World Cup.

If the active mode is `limited_historical_context_model`, the dashboard shows:

```text
Expected impact is generated by a limited supervised historical model trained on real World Cup-derived features. Rich recruitment predictors such as market value and club-season stats are not yet available for historical training, so this should be treated as an MVP model.
```

## Handling World Cup Debutants

Many young value targets may be playing their first World Cup. This is expected, not a data problem.

Previous World Cup experience is optional. If `is_world_cup_debutant`, `previous_world_cup_minutes`, or related experience fields are missing, the dashboard does not invent them. It shows `Not provided` and disables only the debutant-specific filters.

The Breakout Candidates tab is designed for young first-time or low-experience players. It combines expected impact, value efficiency, age/resale profile, role fit and playing-time confidence into `breakout_candidate_score`.

## Historical Model Status

The current model status should be interpreted by mode.

Possible statuses:

```text
Historical World Cup target data is missing. The supervised model is disabled until real player-tournament rows are added.
```

```text
Supervised model active: limited historical context model.
```

```text
Supervised model active: full recruitment model.
```

```text
Supervised model disabled: using transparent baseline fallback.
```

Minimum supervised training requirements:

```text
minimum_training_rows = 300
minimum_required_predictor_coverage = 0.60
minimum usable real predictor features = 3
```

The full model requires enough real recruitment/pre-tournament predictors. The limited model does not require unavailable market values, club-season stats, caps or injury data.

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

Check historical model readiness:

```bash
python scripts/check_model_readiness.py
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
│   ├── data_schema.md
│   ├── historical_data_collection_plan.md
│   └── manual_historical_enrichment.md
├── notebooks/
├── scripts/
│   ├── apply_manual_historical_enrichment.py
│   ├── collect_historical_data.py
│   ├── check_model_readiness.py
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

- The active supervised model is currently the limited historical context model; the full recruitment model still needs richer real historical pre-tournament predictors.
- Actual impact validation remains pending until real 2026 World Cup match rows are added.
- Baseline expected impact is transparent triage logic used only if supervised modelling cannot train.
- Market values and availability signals depend on the quality and refresh cadence of the real CSV inputs.

## Future Improvements

- Enrich the curated 2014/2018/2022 historical training dataset with real market, club-season, caps, availability, form and role-fit predictors.
- Add uncertainty intervals around expected impact.
- Add downloadable scouting reports.
- Add club-specific positional needs and squad-depth constraints.
- Add licensed event-data support when compliant exports are available.
