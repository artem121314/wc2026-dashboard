# Data Inputs

This project uses real data only. Do not add synthetic, fake, generated, demo, or fabricated rows.

The dashboard is designed to ingest compliant local CSV exports. It does not scrape Opta, WhoScored, Wyscout, or other restricted football data providers. Licensed provider exports can be used only when the user has the right to use them.

## Refresh Workflow

1. Place real CSV files in `data/raw/`.
2. Optionally place real curated historical World Cup training data in `data/historical/`.
3. Run the Streamlit app and click `Refresh data`, or run `python scripts/refresh_data.py`.
4. The refresh pipeline validates and merges available real inputs.
5. The pipeline writes `data/processed/player_dashboard_data.csv`.
6. If `data/raw/tournament_match_data.csv` is missing or empty, actual tournament impact remains pending.

## Raw Files

| File | Required for refresh | Purpose |
| --- | --- | --- |
| `data/raw/current_player_pool.csv` | Yes | Current player identity, country, club, league, age and position. |
| `data/raw/player_performance_inputs.csv` | Yes | Recent senior performance inputs used for profiles, baseline impact and value ranking. |
| `data/raw/market_values.csv` | Optional if market value is already in current player pool | Real market values keyed by player. |
| `data/raw/national_team_context.csv` | Optional if national team strength is already in current player pool | Real country strength, draw and group context. |
| `data/raw/tournament_match_data.csv` | Optional until matches are played | Real 2026 World Cup match-level rows for actual impact validation. |

The current repository includes raw CSVs seeded from the existing real project dataset. Missing values were not fabricated. If a future source does not contain a field, leave it blank and let the pipeline disable the affected score or section.

Optional debutant and experience fields can be supplied in the raw player pool or performance input files when a real source provides them:

- `is_world_cup_debutant`
- `previous_world_cup_minutes`
- `previous_world_cup_matches`
- `previous_world_cup_impact_score`
- `senior_national_team_caps`
- `major_tournament_experience`

These fields are not required. Missing previous World Cup experience is normal for young targets and does not disable expected-impact scoring. The dashboard shows missing debutant or previous World Cup fields as `Not provided` and disables only the related filters.

## Historical File

`data/historical/world_cup_player_training_data.csv` is optional until real curated historical data is available.

Use `data/historical/world_cup_player_training_data_template.csv` as the schema-only starting point. It contains headers only and no data.

When this file exists and has valid real rows, the supervised expected-impact model trains from previous World Cups. When it is missing, empty, or invalid, the model is disabled and the dashboard clearly labels any expected impact as a transparent baseline fallback when baseline inputs are available.

The historical training file should use player-tournament observations. Each row should represent one player before one World Cup and their actual impact in that tournament, for example:

- Player A before World Cup 2014, then actual impact at World Cup 2014.
- Player B before World Cup 2018, then actual impact at World Cup 2018.
- Player C before World Cup 2022, then actual impact at World Cup 2022.

The model does not require a 2026 player to have appeared at a previous World Cup. Previous World Cup experience fields are optional model signals only. If real curated historical data includes them, the supervised model can use them; if not, the model trains from the available player-profile, role, team-context and performance features.

The first real historical target collection can be run with:

```bash
python scripts/collect_historical_data.py
```

This writes real player-tournament rows from the public Fjelstul/DataHub World Cup CSV tables and creates `data/historical/source_audit_log.csv`. It does not fabricate unavailable fields. The current source supports tournament output fields such as minutes, starts, goals, goalkeeper clean sheets and cards.

The collector derives previous World Cup experience from real earlier World Cup records:

- `previous_world_cup_minutes`
- `previous_world_cup_matches`
- `previous_world_cup_impact_score`
- `is_world_cup_debutant`

Debutant status is based on previous World Cup appearances, not age. The collector also derives `national_team_strength` from the team's most recent previous World Cup final standing and `group_difficulty_score` from group opponents' previous-strength proxies. These are transparent proxies, not FIFA rankings.

The current source does not provide club, market values, assists, senior national-team caps, injury availability, tactical fit or club-season pre-tournament inputs. Those fields remain blank until a compliant export is added.

The existing historical file can still train the `limited_historical_context_model` because that mode uses only real World Cup-derived context and experience features. The richer `full_recruitment_model` remains a future enrichment target until compliant market, club-season, caps, form, availability and role-fit exports are added. If no supervised mode can train, the app uses `baseline_fallback`.

## Manual Historical Enrichment

Use `data/historical/manual_enrichment_template.csv` when a compliant source or licensed/manual export provides historical predictor values that are not available from the World Cup source.

Recommended workflow:

1. Fill `data/historical/manual_enrichment_template.csv`, or prepare another compliant CSV with the same columns.
2. Fill only real sourced values.
3. Include `data_source` for every row where a value is supplied.
4. Run:

```bash
python scripts/apply_manual_historical_enrichment.py --input data/historical/manual_enrichment_template.csv
```

The script matches exact `tournament_year + player_name` keys, also uses `country` when provided, rejects duplicate keys, saves a backup, updates only non-null supplied values, and does not overwrite existing non-null values unless explicitly run with `--overwrite true`. It does not fuzzy-match names, infer values, or fabricate missing fields.

See `docs/manual_historical_enrichment.md` for details.

## Tournament Updates

To update live tournament validation:

1. Add real match rows to `data/raw/tournament_match_data.csv`.
2. Keep one row per player per match.
3. Click `Refresh data` in Streamlit or run `python scripts/refresh_data.py`.
4. The pipeline aggregates match rows by player and recalculates actual tournament impact.

If this file has zero rows, `actual_tournament_impact_score` and `performance_delta` remain null and `performance_outcome` remains `Pending`.

## Processed File

`data/processed/player_dashboard_data.csv` is generated by the refresh pipeline. The app can load this file directly, but the preferred workflow is to refresh it from real raw CSV inputs whenever the source data changes.

See `docs/data_schema.md` for column-level schemas.
