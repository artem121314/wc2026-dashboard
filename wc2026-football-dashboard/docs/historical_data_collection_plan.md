# Historical Data Collection Plan

This project does not include fabricated historical training rows. The supervised expected-impact model becomes available only after `data/historical/world_cup_player_training_data.csv` is populated with real, curated player-tournament observations.

## Required Grain

Each row must represent one player before one World Cup and that player's actual impact in the same tournament.

```text
player_name + tournament_year
```

Examples:

- Player A before World Cup 2014, then actual impact at World Cup 2014.
- Player B before World Cup 2018, then actual impact at World Cup 2018.
- Player C before World Cup 2022, then actual impact at World Cup 2022.

This is not repeated-player history. A 2026 target player does not need to have played in a previous World Cup. The model learns from historical player profiles and applies those patterns to current players, including debutants.

## Target File

Build:

```text
data/historical/world_cup_player_training_data.csv
```

Use the schema-only template:

```text
data/historical/world_cup_player_training_data_template.csv
```

Copy the template headers, then populate rows only from real, compliant sources.

## Tournaments

Recommended historical tournaments:

- World Cup 2014
- World Cup 2018
- World Cup 2022

The readiness script checks whether these tournament years are present. Missing years are reported so the analyst knows which coverage gaps remain.

## Feature Groups

### A. Player Identity And Context

| Column | Notes |
| --- | --- |
| `player_name` | Player name before the tournament. |
| `tournament_year` | One of the historical World Cup years. |
| `country` | National team. |
| `position` | Normalised position used by the dashboard. |
| `age` | Player age before the tournament. |
| `club` | Club before the tournament. |
| `league` | League before the tournament. |

### B. Pre-Tournament Market And Context

| Column | Notes |
| --- | --- |
| `market_value_before_tournament` | Real pre-tournament market value in euros. |
| `club_level_score` | Real or documented club-strength score scaled 0-100. |
| `league_strength_score` | Real or documented league-strength score scaled 0-100. |
| `national_team_strength` | Real or documented national-team strength score scaled 0-100. The current collector uses most recent previous World Cup final standing as a proxy, not FIFA ranking. |
| `group_difficulty_score` | Group/tournament draw difficulty score scaled 0-100. The current collector averages opponents' previous World Cup strength proxies where available. |

### C. Pre-Tournament Performance

| Column | Notes |
| --- | --- |
| `club_minutes_previous_season` | Club minutes in the season before the tournament. |
| `goals_previous_season` | Club goals in the season before the tournament. |
| `assists_previous_season` | Club assists in the season before the tournament. |
| `recent_form_score` | Real or documented form score scaled 0-100. |
| `role_fit_score` | Real or documented tactical/role fit score scaled 0-100. |
| `expected_starter_score` | Expected national-team starter/minutes likelihood score scaled 0-100. |

### D. National-Team Experience

These columns are optional model signals. Missing previous World Cup experience must not be treated as an error, especially for younger players.

| Column | Notes |
| --- | --- |
| `senior_national_team_caps` | Caps before the tournament. |
| `major_tournament_experience` | Count or documented score for prior major senior tournaments. |
| `is_world_cup_debutant` | True when real previous World Cup appearances equal zero; false when prior appearances exist. Do not infer from age. |
| `previous_world_cup_minutes` | Cumulative real World Cup minutes before this tournament. |
| `previous_world_cup_matches` | Cumulative real World Cup appearances before this tournament. |
| `previous_world_cup_impact_score` | Most recent prior World Cup impact score if a prior player-tournament row exists. |

### E. Target

| Column | Notes |
| --- | --- |
| `actual_tournament_impact_score` | Real position-adjusted impact score for that player in that tournament, scaled 0-100. |

The target should be calculated from real tournament match data. If match data is incomplete, leave the target blank until it can be supported.

## Initial Real Collection Layer

The repository includes a compliant collection script:

```bash
python scripts/collect_historical_data.py
```

The first implemented source is the Fjelstul World Cup Database as republished by DataHub in CSV form. It is used because it provides real squad, appearance, match, goal, substitution, booking and player birth-date tables for the men's World Cup, including 2014, 2018 and 2022.

The collector writes:

```text
data/historical/world_cup_player_training_data.csv
data/historical/source_audit_log.csv
```

For this MVP collection, the available real fields support:

- `player_name`
- `tournament_year`
- `country`
- `position`
- `age`
- `minutes_at_tournament`
- `starts_at_tournament`
- `appearances_at_tournament`
- `substitute_appearances_at_tournament`
- `goals_at_tournament`
- `clean_sheets_at_tournament`
- `yellow_cards_at_tournament`
- `red_cards_at_tournament`
- `actual_tournament_impact_score`
- `previous_world_cup_minutes`
- `previous_world_cup_matches`
- `previous_world_cup_impact_score`
- `is_world_cup_debutant`
- `national_team_strength`
- `group_difficulty_score`

The same source does not provide club, league, assists, market value, club-season performance, senior national-team caps, injury availability or tactical fit inputs. Those columns remain blank/null until a compliant source or manually curated export is added.

Previous World Cup experience is calculated from real prior World Cup player records using source player IDs. The training rows remain 2014, 2018 and 2022, but earlier World Cups are read internally so the collector can calculate whether a player had prior World Cup appearances.

`national_team_strength` is currently a previous World Cup finish proxy: the winner of the team's most recent prior World Cup appearance receives 100, the lowest-ranked team receives 0, and other teams are scaled by final standing. If a team has no prior World Cup standing in the source, the value remains null.

`group_difficulty_score` averages the available `national_team_strength` values for the other teams in the same group. If opponent strength proxies are unavailable, the group difficulty remains null.

## Manual Predictor Enrichment

Historical club, league, market-value, club-season, caps, availability, form and role-fit predictors require separate compliant sources. The project does not currently auto-join these fields because public market-value datasets often have unclear licensing or unreliable player-name matching.

Use:

```text
data/historical/manual_enrichment_template.csv
docs/manual_historical_enrichment.md
scripts/apply_manual_historical_enrichment.py
```

The manual workflow uses exact `tournament_year + player_name` keys, requires provenance for supplied values, and updates only non-null real values. It does not fuzzy-match players or fabricate missing predictors.

## MVP Target Construction

The initial `actual_tournament_impact_score` is a transparent 0-100 target built from real tournament output:

- Outfield players: tournament minutes, starts and goals.
- Goalkeepers: tournament minutes, starts and clean sheets.

Each component is percentile-ranked within broad position groups, then converted to a final position-adjusted percentile score. Assists are not included in the first target because the selected public source does not provide them. This is a practical MVP target, not a final definition of player impact.

The current collection is sufficient for the `limited_historical_context_model`, which is the MVP supervised model. It trains on real World Cup-derived context and experience features rather than fabricated recruitment inputs.

The richer `full_recruitment_model` remains unavailable until compliant real pre-tournament recruitment features are added. The collected target rows are still the foundation for that richer model because future market, club-season, caps, availability and role-fit exports can be joined onto the same player-tournament rows.

## Model Modes

| Mode | When Used | Features |
| --- | --- | --- |
| `limited_historical_context_model` | Real World Cup target rows and at least three usable real historical context predictors exist. | Age, position, prior World Cup minutes/matches/impact, debutant status, group difficulty, tournament year and other World Cup-derived context features where usable. |
| `full_recruitment_model` | Rich real pre-tournament recruitment predictors have enough coverage. | Market value, club-season performance, club/league strength, caps, availability, recent form and role fit. |
| `baseline_fallback` | No supervised mode can train. | Transparent manually weighted current-player baseline only. |

The limited model is not perfect, but it is real supervised learning trained on real historical player-tournament rows and actual tournament impact outcomes.

## Compliant Source Options

The project should ingest CSV exports. Do not implement direct scraping from restricted football data providers.

| Source type | Example use | Status | Notes |
| --- | --- | --- | --- |
| Public FIFA match reports and squad pages | Squads, minutes, match participation, basic tournament context | Public/open | Use manually curated or exported data where permitted. |
| Kaggle or public datasets | Historical squads, appearances, goals, match-level summaries | Public/open if license permits | Check license and attribution requirements before use. |
| StatsBomb Open Data | Event/match data where relevant tournaments are available | Public/open for included competitions | Use only tournaments actually covered by the open-data license. |
| FBref-style exported CSVs | Club-season performance inputs | Permitted only if export/use is allowed | Do not scrape in the app. Store compliant local CSV exports. |
| Transfermarkt-style market values | Pre-tournament market values | Manually curated/licensed/permitted only | Do not scrape restricted pages from the app. |
| Licensed Opta/Wyscout/StatsBomb provider exports | Event, tracking, availability, tactical, or role-fit data | Licensed/commercial export | Use only if the user has the rights to use and store the export. |
| Restricted provider websites | Direct scraping from Opta, WhoScored, Wyscout, or similar | Not allowed to scrape | The dashboard is an ingestion layer, not a scraper. |

## Suggested Workflow

1. Copy `data/historical/world_cup_player_training_data_template.csv`.
2. Populate rows for World Cup 2014, 2018 and 2022 from real permitted sources.
3. Keep one row per player per tournament.
4. Leave unavailable optional fields blank.
5. Keep all score fields on a 0-100 scale with documented formulas.
6. Run:

```bash
python scripts/check_model_readiness.py
python scripts/validate_project.py
```

7. If readiness passes and enough real rows exist, run:

```bash
python scripts/refresh_data.py
```

8. Open the dashboard. The Data & Model Ops tab will show the supervised model as available once training succeeds.

## Modelling Readiness Rules

The readiness script checks:

- historical training file exists
- template file exists
- required columns are present
- row count is high enough for a train/test split
- tournament years are present
- target values are available
- score columns are inside 0-100
- the supervised model can train with scikit-learn

If historical rows are missing, the model remains disabled and the dashboard uses the transparent baseline fallback where current-player baseline fields exist.
