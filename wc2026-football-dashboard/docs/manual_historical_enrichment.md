# Manual Historical Predictor Enrichment

The current historical World Cup dataset contains real target rows and real-derived World Cup experience features. It does not yet contain enough real pre-tournament club, league, market, caps, availability, form or role-fit predictor values to train the supervised expected-impact model.

This workflow exists for fields that cannot be safely joined from a public source without risking unreliable matching or non-compliant scraping.

## Why Manual Enrichment Is Needed

The project inspected the existing local project data and public-source options for historical player club, league and market-value context around World Cups 2014, 2018 and 2022.

Current result:

- The Fjelstul/DataHub World Cup source provides squads, appearances, goals, substitutions, bookings, standings, groups and player birth dates.
- It does not provide club, league, pre-tournament market value, club-season performance, senior national-team caps, injury availability, recent form or role fit.
- Public web sources for historical market values often rely on Transfermarkt-derived data or unclear licensing.
- Direct scraping from Transfermarkt, WhoScored, Opta, Wyscout or similar providers is not allowed in this project.
- Name-only matching across independent public datasets is risky for historical football data and should not be forced.

## Template

Use:

```text
data/historical/manual_enrichment_template.csv
```

Copy it to:

```text
data/historical/manual_enrichment.csv
```

Then fill only values supported by real, compliant sources.

## Template Columns

| Column | Required | Notes |
| --- | --- | --- |
| `tournament_year` | Yes | Must match an existing historical training row. |
| `player_name` | Yes | Must match `data/historical/world_cup_player_training_data.csv` exactly. |
| `country` | Recommended | Context for manual QA. |
| `position` | Recommended | Context for manual QA. |
| `club` | Optional | Club before the tournament from a compliant source. |
| `league` | Optional | League before the tournament from a compliant source. |
| `market_value_before_tournament` | Optional | Real pre-tournament market value in euros. |
| `club_level_score` | Optional | Real or documented club-strength score scaled 0-100. Leave blank if no defensible source exists. |
| `league_strength_score` | Optional | Real or documented league-strength score scaled 0-100. Leave blank if no defensible source exists. |
| `club_minutes_previous_season` | Optional | Club minutes in the season immediately before the tournament. |
| `goals_previous_season` | Optional | Club goals in the season immediately before the tournament. |
| `assists_previous_season` | Optional | Club assists in the season immediately before the tournament. |
| `senior_national_team_caps` | Optional | Senior caps before the tournament. Do not approximate this from World Cup appearances. |
| `injury_availability_score` | Optional | Real or documented score scaled 0-100. Leave blank if no defensible source exists. |
| `recent_form_score` | Optional | Real or documented score scaled 0-100. Leave blank if no defensible source exists. |
| `role_fit_score` | Optional | Real or documented tactical/role-fit score scaled 0-100. Leave blank if no defensible source exists. |
| `data_source` | Required when values are supplied | Dataset/export name or manual curation source. |
| `source_url` | Recommended | Public URL, local export path, or license reference. |
| `notes` | Recommended | Include caveats, matching notes, license notes or curator comments. |

## Acceptable Sources

Use only sources you are allowed to use:

- Public/open CSV datasets with clear licenses.
- Official FIFA squad/match report PDFs or pages, manually curated where permitted.
- Licensed provider exports if you have the rights to store and use them.
- Local CSV exports supplied by the user.
- Kaggle datasets only after checking license and attribution requirements.

Do not use:

- Direct scraping from Transfermarkt pages.
- Direct scraping from WhoScored, Opta, Wyscout or other restricted providers.
- Values copied from unclear-license datasets.
- Median, random, guessed or age-inferred values.

## Applying Enrichment

After filling `data/historical/manual_enrichment_template.csv`, run:

```bash
python scripts/apply_manual_historical_enrichment.py --input data/historical/manual_enrichment_template.csv
```

The script:

- uses exact `tournament_year + player_name` matches, with `country` also used when provided
- rejects duplicate keys
- rejects rows with values but no `data_source`
- updates only non-null supplied values
- does not overwrite existing non-null historical values unless run with `--overwrite true`
- saves a timestamped backup before modifying the historical file
- appends manual source provenance to `data/historical/source_audit_log.csv`
- leaves blank fields unchanged
- does not fuzzy-match players
- does not generate fallback values

Then run:

```bash
python scripts/check_model_readiness.py
python scripts/validate_project.py
```

## Model Readiness Threshold

The supervised model remains disabled until:

- the historical target has enough real rows
- target values are valid 0-100 scores
- required predictor columns are present
- each required predictor has at least 60% real non-null coverage
- each required predictor has enough variation to be useful

`scripts/check_model_readiness.py` prints feature availability by column so the next missing input is visible.

## How To Make The Supervised Model Turn On

The warning disappears only when real predictor coverage is sufficient. Do not add synthetic or guessed values to force activation.

1. Fill real predictor values in `data/historical/manual_enrichment_template.csv`.
2. Apply the enrichment:

```bash
python scripts/apply_manual_historical_enrichment.py --input data/historical/manual_enrichment_template.csv
```

3. Check readiness:

```bash
python scripts/check_model_readiness.py
```

4. If feature coverage reaches the threshold, the supervised model becomes enabled.
5. Refresh the dashboard dataset:

```bash
python scripts/refresh_data.py
```

6. Open the dashboard and click Refresh data if you want Streamlit to re-read the same updated inputs.

Current activation rules:

- at least 300 usable historical target rows
- at least 60% real non-null coverage for required player, tournament-context and World Cup-experience predictors
- at least three usable real recruitment/pre-tournament predictors
- enough target variation and predictor variation for the model to learn
