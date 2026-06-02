# Data Schema

This project uses real CSV inputs only. Blank cells are acceptable when a real source does not provide a value. Do not add synthetic rows or fabricated values to satisfy a schema.

Types are expected CSV types after reading with pandas.

## `data/raw/current_player_pool.csv`

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `player_name` | Player name from the source export. | Yes | string | source player name | modelling, filtering, display |
| `country` | National team or eligible country. | Yes | string | country name | modelling, filtering, display |
| `club` | Current club. | Yes | string | club name | filtering, display |
| `league` | Current league or domestic competition. | Yes | string | league name | filtering, display |
| `age` | Player age at refresh time or tournament reference date. | Yes | number | 23 | modelling, filtering |
| `position` | Normalised football position. | Yes | string | Winger | modelling, filtering, profiles |
| `squad_number` | Shirt number if supplied by the source. | Optional | number/string | 7 | display |
| `raw_position` | Original position text before normalisation. | Optional | string | Right Winger | display |
| `date_of_birth` | Date of birth if available. | Optional | date/string | 2001-01-01 | display, audit |
| `caps` | Senior national-team caps if available. | Optional | number | 24 | modelling |
| `international_goals` | Senior international goals if available. | Optional | number | 6 | display, modelling |
| `final_squad_selected` | Real squad-selection flag if officially known. | Optional | boolean | true | filtering, display |
| `tm_player_id` | Transfermarkt player identifier if present in a compliant export. | Optional | number/string | 123456 | audit |
| `tm_current_club_name` | Club name from the market-value source export. | Optional | string | club name | audit |
| `tm_current_club_domestic_competition_id` | Domestic competition identifier from the source export. | Optional | string | GB1 | audit |
| `tm_sub_position` | Detailed position from the source export. | Optional | string | Centre-Forward | display |
| `tm_foot` | Preferred foot if available. | Optional | string | right | display |
| `tm_height_in_cm` | Height in centimetres if available. | Optional | number | 180 | display |
| `tm_national_team_name` | National team name from source export. | Optional | string | country name | audit |
| `tm_confederation` | Confederation if available. | Optional | string | UEFA | filtering, audit |
| `tm_url` | Source profile URL if included in a compliant export. | Optional | string | https://... | audit |
| `tm_image_url` | Source image URL if included in a compliant export. | Optional | string | https://... | display |
| `data_source` | Human-readable source label. | Optional | string | licensed_export | audit |
| `data_refresh_date` | Date the source export was refreshed. | Optional | date/string | 2026-06-02 | audit |

## `data/raw/market_values.csv`

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `player_name` | Player name matching the current player pool. | Yes | string | source player name | merge key |
| `country` | Country used as an optional merge disambiguator. | Optional | string | country name | merge key |
| `market_value_eur` | Current real market value in euros. | Yes | number | 25000000 | modelling, filtering, value ranking |
| `market_value_source` | Source label for the selected market value. | Optional | string | licensed_export | audit |
| `highest_market_value_eur` | Highest recorded market value in euros if available. | Optional | number | 45000000 | display, audit |
| `latest_market_value_eur` | Latest market value from a source feed if distinct. | Optional | number | 25000000 | audit |
| `latest_market_value_date` | Date attached to the latest market value. | Optional | date/string | 2026-06-02 | audit |
| `salimt_latest_market_value_eur` | Public GitHub market-value export value if used compliantly. | Optional | number | 25000000 | audit |
| `salimt_latest_market_value_date` | Date for the public GitHub market-value export. | Optional | date/string | 2026-06-02 | audit |
| `tm_market_value_in_eur` | Market value from a compliant Transfermarkt-derived export. | Optional | number | 25000000 | audit |
| `tm_highest_market_value_in_eur` | Highest value from a compliant Transfermarkt-derived export. | Optional | number | 45000000 | audit |
| `tm_url` | Source profile URL if included in the export. | Optional | string | https://... | audit |
| `data_source` | Human-readable source label. | Optional | string | licensed_export | audit |
| `data_refresh_date` | Date the source export was refreshed. | Optional | date/string | 2026-06-02 | audit |

## `data/raw/national_team_context.csv`

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `country` | National team name. | Yes | string | country name | merge key, filtering |
| `national_team_strength` | Real or curated country strength score scaled 0-100. | Yes | number | 72.5 | modelling, baseline |
| `draw_context_score` | Tournament draw context score scaled 0-100 if available. | Optional | number | 58.0 | baseline, value ranking |
| `group_difficulty_score` | Group difficulty score scaled 0-100 if available. | Optional | number | 58.0 | modelling |
| `group` | World Cup group if known. | Optional | string | Group A | display, filtering |
| `group_opponents` | Group opponents if known. | Optional | string | team list | display |
| `average_group_opponent_strength` | Average group opponent strength if curated. | Optional | number | 64.0 | modelling, audit |
| `fifa_ranking` | Real FIFA ranking if supplied. | Optional | number | 12 | modelling, audit |
| `national_team_market_value` | Squad market value if supplied. | Optional | number | 600000000 | modelling, audit |
| `tm_national_team_market_value` | National-team value from source export if supplied. | Optional | number | 600000000 | audit |
| `tm_fifa_ranking` | FIFA ranking from source export if supplied. | Optional | number | 12 | audit |
| `tm_confederation` | Confederation. | Optional | string | UEFA | filtering |
| `data_source` | Human-readable source label. | Optional | string | licensed_export | audit |
| `data_refresh_date` | Date the source export was refreshed. | Optional | date/string | 2026-06-02 | audit |

## `data/raw/player_performance_inputs.csv`

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `player_name` | Player name matching the current player pool. | Yes | string | source player name | merge key |
| `country` | Country used as an optional merge disambiguator. | Optional | string | country name | merge key |
| `minutes` | Recent senior minutes from the real source window. | Yes | number | 2400 | modelling, filtering, confidence |
| `goals` | Recent senior goals. | Yes | number | 10 | profiles, modelling |
| `assists` | Recent senior assists. | Yes | number | 7 | profiles, modelling |
| `shots` | Recent shots. | Yes | number | 60 | profiles |
| `key_passes` | Recent key passes. | Yes | number | 40 | profiles |
| `progressive_passes` | Recent progressive passes. | Yes | number | 120 | profiles |
| `progressive_carries` | Recent progressive carries. | Yes | number | 80 | profiles |
| `successful_dribbles` | Recent successful dribbles. | Yes | number | 35 | profiles |
| `tackles` | Recent tackles. | Yes | number | 55 | profiles |
| `interceptions` | Recent interceptions. | Yes | number | 35 | profiles |
| `aerial_duels_won` | Recent aerial duels won. | Yes | number | 45 | profiles |
| `pass_completion_pct` | Pass completion percentage. | Yes | number | 84.5 | profiles |
| `turnovers` | Recent turnovers or possession losses. | Yes | number | 30 | profiles |
| `club_level_score` | Real or curated club strength score scaled 0-100. | Yes | number | 76.0 | modelling, baseline |
| `recent_form_score` | Real or curated recent form score scaled 0-100. | Yes | number | 68.0 | modelling, baseline |
| `expected_minutes_score` | Real or curated expected tournament minutes score scaled 0-100. | Yes | number | 70.0 | modelling, baseline |
| `non_penalty_goals` | Recent non-penalty goals. | Optional | number | 8 | profiles |
| `shots_in_box` | Recent shots in the box. | Optional | number | 35 | proxy metrics |
| `touches_in_box` | Recent touches in the box. | Optional | number | 95 | proxy metrics |
| `through_balls` | Recent through balls. | Optional | number | 12 | profiles |
| `crosses_into_box` | Recent crosses into the box. | Optional | number | 20 | profiles |
| `shot_creating_actions_proxy` | Source or curated shot-creation proxy. | Optional | number | 80 | profiles |
| `passes_into_final_third` | Recent passes into final third. | Optional | number | 90 | profiles |
| `carries_into_final_third` | Recent carries into final third. | Optional | number | 45 | profiles |
| `recoveries` | Recent ball recoveries. | Optional | number | 120 | profiles |
| `pressures` | Recent pressing actions if supplied. | Optional | number | 300 | profiles |
| `long_passes` | Recent long passes. | Optional | number | 150 | profiles |
| `saves` | Recent saves for goalkeepers. | Optional | number | 80 | profiles |
| `clean_sheets` | Recent clean sheets for goalkeepers/defenders. | Optional | number | 10 | profiles |
| `aerial_threat` | Aerial threat score or count if supplied. | Optional | number | 70 | profiles |
| `xg_proxy` | Existing xG proxy from a source pipeline if supplied. | Optional | number | 8.5 | display, audit |
| `xa_proxy` | Existing xA proxy from a source pipeline if supplied. | Optional | number | 6.2 | display, audit |
| `yellow_cards` | Recent yellow cards. | Optional | number | 5 | risk, display |
| `red_cards` | Recent red cards. | Optional | number | 1 | risk, display |
| `recent_appearances` | Recent appearances. | Optional | number | 32 | confidence, display |
| `height_in_cm` | Height in centimetres. | Optional | number | 180 | display |
| `market_value_score` | Existing market-value score if supplied by source pipeline. | Optional | number | 66.0 | audit |
| `tactical_fit_score` | Real or curated tactical fit score scaled 0-100. | Optional | number | 72.0 | modelling, ranking |
| `final_squad_selection_score` | Real or curated selection confidence score scaled 0-100. | Optional | number | 80.0 | modelling, ranking |
| `injury_availability_score` | Availability score from real injury/availability inputs. | Optional | number | 95.0 | baseline, risk |
| `injury_severity_score` | Injury severity score if available. | Optional | number | 10.0 | risk |
| `injury_status` | Current injury status if available. | Optional | string | available | display, risk |
| `recent_injury_count` | Count of recent injuries if available. | Optional | number | 1 | risk |
| `is_injured` | Current injury flag if available. | Optional | boolean | false | risk |
| `injury_source_url` | Source URL for injury status if supplied. | Optional | string | https://... | audit |
| `caps` | Senior national-team caps if supplied here. | Optional | number | 24 | modelling |
| `international_goals` | Senior international goals if supplied here. | Optional | number | 6 | display |
| `data_source` | Human-readable source label. | Optional | string | licensed_export | audit |
| `data_refresh_date` | Date the source export was refreshed. | Optional | date/string | 2026-06-02 | audit |

## `data/raw/tournament_match_data.csv`

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `player_name` | Player name matching the dashboard data. | Yes | string | source player name | merge key |
| `match_id` | Match identifier. | Yes | string | 2026-group-a-01 | actual impact |
| `date` | Match date. | Yes | date/string | 2026-06-11 | actual impact |
| `opponent` | Opponent country. | Yes | string | opponent country | display, audit |
| `stage` | Tournament stage. | Yes | string | Group | display, audit |
| `minutes` | Minutes played in the match. | Yes | number | 90 | actual impact |
| `goals` | Goals in the match. | Yes | number | 1 | actual impact |
| `assists` | Assists in the match. | Yes | number | 0 | actual impact |
| `shots` | Shots in the match. | Yes | number | 3 | actual impact |
| `key_passes` | Key passes in the match. | Yes | number | 2 | actual impact |
| `tackles` | Tackles in the match. | Yes | number | 4 | actual impact |
| `interceptions` | Interceptions in the match. | Yes | number | 2 | actual impact |
| `saves` | Saves in the match. | Yes | number | 5 | actual impact |
| `clean_sheets` | Clean sheet flag/count for the match. | Yes | number | 1 | actual impact |
| `match_rating` | Real match rating if supplied by a compliant source. | Yes | number | 7.4 | actual impact |

## `data/historical/world_cup_player_training_data.csv`

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `tournament_year` | Historical World Cup year. | Yes | number | 2022 | modelling |
| `player_name` | Historical player name. | Yes | string | source player name | audit |
| `age` | Player age before that tournament. | Yes | number | 25 | modelling |
| `position` | Normalised position before that tournament. | Yes | string | Central midfielder | modelling |
| `market_value_before_tournament` | Real pre-tournament market value in euros. | Yes | number | 35000000 | modelling |
| `club_level_score` | Real or curated club strength score scaled 0-100. | Yes | number | 78.0 | modelling |
| `league_strength_score` | Real or curated league strength score scaled 0-100. | Yes | number | 82.0 | modelling |
| `club_minutes_previous_season` | Club minutes in the previous season. | Yes | number | 2700 | modelling |
| `goals_previous_season` | Club goals in the previous season. | Yes | number | 8 | modelling |
| `assists_previous_season` | Club assists in the previous season. | Yes | number | 7 | modelling |
| `national_team_caps` | Caps before the tournament. | Yes | number | 34 | modelling |
| `expected_starter_score` | Real or curated starter likelihood score scaled 0-100. | Yes | number | 75.0 | modelling |
| `national_team_strength` | Country strength score scaled 0-100. | Yes | number | 80.0 | modelling |
| `group_difficulty_score` | Group difficulty score scaled 0-100. | Yes | number | 62.0 | modelling |
| `injury_availability_score` | Pre-tournament availability score scaled 0-100. | Yes | number | 95.0 | modelling |
| `recent_form_score` | Pre-tournament form score scaled 0-100. | Yes | number | 70.0 | modelling |
| `role_fit_score` | Tactical/role fit score scaled 0-100. | Yes | number | 74.0 | modelling |
| `actual_tournament_impact_score` | Real position-adjusted historical tournament impact target scaled 0-100. | Yes | number | 68.0 | modelling target |

## `data/processed/player_dashboard_data.csv`

The processed file is generated by the refresh pipeline and may include passthrough source columns for auditability. The dashboard expects the following core fields.

| Column | Description | Required | Type | Example | Used for |
| --- | --- | --- | --- | --- | --- |
| `player_name` | Player name. | Yes | string | source player name | display, filtering |
| `country` | Country. | Yes | string | country name | display, filtering |
| `club` | Club. | Yes | string | club name | display, filtering |
| `league` | League. | Yes | string | league name | display, filtering |
| `age` | Age. | Yes | number | 23 | filtering, ranking |
| `position` | Normalised position. | Yes | string | Winger | filtering, profiles |
| `best_profile` | Best matching player profile. | Yes | string | High-volume winger | filtering, display |
| `best_profile_score` | Best profile score scaled 0-100. | Yes | number | 72.0 | profiles, ranking |
| `market_value_eur` | Real market value in euros. | Yes | number | 25000000 | filtering, ranking |
| `recent_senior_minutes` | Recent senior minutes used as confidence proxy. | Yes | number | 2400 | filtering, ranking |
| `attacking_score` | Position-adjusted attacking score. | Optional | number | 67.0 | comparison |
| `creativity_score` | Position-adjusted creativity score. | Optional | number | 71.0 | comparison |
| `progression_score` | Position-adjusted progression score. | Optional | number | 64.0 | comparison |
| `defensive_score` | Position-adjusted defensive score. | Optional | number | 55.0 | comparison |
| `possession_score` | Possession/security score. | Optional | number | 70.0 | comparison |
| `overall_score` | Role-sensitive current performance score. | Optional | number | 68.0 | baseline |
| `baseline_expected_impact_score` | Transparent baseline expected-impact score. | Optional | number | 66.0 | fallback prediction |
| `pre_tournament_expected_impact_score` | Main expected World Cup impact score. | Yes | number | 66.0 | modelling, ranking |
| `expected_impact_source` | Source of the expected-impact score. | Yes | string | transparent_baseline_fallback | audit, display |
| `value_efficiency_score` | Expected impact relative to market value. | Optional | number | 75.0 | value ranking |
| `value_opportunity_score` | Recruitment business ranking score. | Yes | number | 74.0 | shortlist ranking |
| `recommendation` | Shortlist recommendation category. | Yes | string | Watchlist | shortlist |
| `recommendation_reason` | Explanation for recommendation. | Yes | string | reason text | display |
| `risk_band` | Risk category. | Yes | string | Medium | filtering, display |
| `risk_reason` | Explanation for risk band. | Yes | string | reason text | display |
| `actual_tournament_impact_score` | Real position-adjusted 2026 tournament impact when match data exists. | Optional | number | 71.0 | validation |
| `performance_delta` | Actual impact minus pre-tournament expected impact. | Optional | number | 5.0 | validation |
| `performance_outcome` | Validation label. | Yes | string | Pending | validation |
| `historical_model_available` | Whether the supervised model was available at refresh time. | Optional | boolean | false | audit |
| `model_status` | Model status string. | Optional | string | disabled_missing_real_historical_data | audit |
| `model_warning` | Model warning shown when disabled. | Optional | string | warning text | audit |

Valid `performance_outcome` values are `Pending`, `Overperformed`, `Met expectations`, and `Underperformed`.
