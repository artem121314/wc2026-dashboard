from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_player_data, refresh_player_data
from data_pipeline import check_data_availability, get_last_refresh_timestamp, get_refresh_manifest
from feature_engineering import VALUE_STRATEGIES, calculate_value_opportunity_score
from model import FALLBACK_WARNING, check_historical_model_readiness, train_expected_impact_model
from player_profiles import PLAYER_PROFILES, PROFILE_ELIGIBILITY, get_eligible_positions, get_profile_score_column
from utils import format_currency, scouting_summary
from visualisations import (
    create_age_vs_value_opportunity_chart,
    create_breakout_candidates_chart,
    create_expected_impact_vs_value_efficiency_chart,
    create_market_value_vs_value_opportunity_chart,
    create_metric_percentile_chart,
    create_performance_delta_distribution_chart,
    create_percentile_bar_chart,
    create_predicted_vs_actual_impact_chart,
    create_price_to_impact_quadrant_chart,
    create_radar_chart,
    create_recommendation_breakdown_chart,
    create_risk_vs_opportunity_matrix,
    create_top_players_chart,
    create_top_value_opportunities_chart,
)

# Naming options considered:
# - World Cup 2026 Recruitment Intelligence Dashboard
# - WC 2026 Scouting Intelligence Hub
# - World Cup 2026 Player Opportunity Intelligence
# - WC 2026 Recruitment Opportunity Hub
APP_NAME = "World Cup 2026 Recruitment Intelligence Dashboard"
APP_SUBTITLE = (
    "Recruitment intelligence for identifying high-upside World Cup targets before the tournament "
    "and validating outcomes during the competition."
)

st.set_page_config(
    page_title=APP_NAME,
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
    }
    [data-testid="stMetricLabel"] { color: #475569; }
    .scout-note {
        border-left: 4px solid #0f766e;
        background: #f8fafc;
        padding: 1rem 1.1rem;
        border-radius: 0 8px 8px 0;
        color: #0f172a;
    }
    .executive-hero {
        border: 1px solid #dbe3ea;
        background: linear-gradient(135deg, #f8fafc 0%, #eef6f4 100%);
        border-radius: 8px;
        padding: 1.25rem 1.35rem;
        margin-bottom: 1rem;
    }
    .section-note {
        color: #475569;
        font-size: 0.95rem;
        line-height: 1.45;
    }
    .status-pill {
        display: inline-block;
        border: 1px solid #99f6e4;
        background: #ccfbf1;
        color: #134e4a;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.82rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    .soft-card {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_player_data()


@st.cache_resource(show_spinner=False)
def get_model_summary() -> dict[str, object]:
    return train_expected_impact_model()


@st.cache_resource(show_spinner=False)
def get_model_readiness() -> dict[str, object]:
    return check_historical_model_readiness(attempt_training=False)


def clear_app_cache() -> None:
    st.cache_data.clear()
    st.cache_resource.clear()


def format_timestamp(value: object) -> str:
    if not value:
        return "Not yet available"
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return parsed.strftime("%d %b %Y, %H:%M UTC")


def friendly_model_mode(mode: object) -> str:
    labels = {
        "limited_historical_context_model": "Limited historical model",
        "full_recruitment_model": "Full recruitment model",
        "baseline_fallback": "Transparent baseline fallback",
    }
    return labels.get(str(mode), str(mode).replace("_", " ").title())


def friendly_source(source: object) -> str:
    labels = {
        "supervised_historical_model": "Supervised historical model",
        "transparent_baseline_fallback": "Transparent baseline fallback",
        "unavailable_missing_features": "Not yet available",
    }
    return labels.get(str(source), str(source).replace("_", " ").title())


def business_status(value: bool, active_text: str = "Updated", inactive_text: str = "Not yet available") -> str:
    return active_text if value else inactive_text


def impact_band(score: object) -> str:
    parsed = pd.to_numeric(score, errors="coerce")
    if pd.isna(parsed):
        return "Not yet available"
    score = float(parsed)
    if score >= 85:
        return "Elite"
    if score >= 70:
        return "High"
    if score >= 50:
        return "Strong"
    if score >= 30:
        return "Moderate"
    return "Low"


def impact_percentile(df: pd.DataFrame, player: pd.Series) -> float | None:
    values = pd.to_numeric(df["pre_tournament_expected_impact_score"], errors="coerce")
    score = pd.to_numeric(player.get("pre_tournament_expected_impact_score"), errors="coerce")
    if pd.isna(score) or values.dropna().empty:
        return None
    return round(float((values <= score).mean() * 100), 1)


def positional_rank(df: pd.DataFrame, player: pd.Series) -> tuple[int | None, int]:
    position_df = df[df["position"] == player["position"]].copy()
    scores = pd.to_numeric(position_df["pre_tournament_expected_impact_score"], errors="coerce")
    if scores.dropna().empty:
        return None, len(position_df)
    position_df["_impact_rank"] = scores.rank(ascending=False, method="min")
    rank = position_df.loc[position_df["player_name"] == player["player_name"], "_impact_rank"]
    if rank.empty or pd.isna(rank.iloc[0]):
        return None, len(position_df)
    return int(rank.iloc[0]), int(scores.notna().sum())


def expected_impact_drivers(player: pd.Series) -> list[str]:
    drivers: list[str] = []
    previous_impact = pd.to_numeric(player.get("previous_world_cup_impact_score"), errors="coerce")
    previous_minutes = pd.to_numeric(player.get("previous_world_cup_minutes"), errors="coerce")
    previous_matches = pd.to_numeric(player.get("previous_world_cup_matches"), errors="coerce")
    age = pd.to_numeric(player.get("age"), errors="coerce")
    group_difficulty = pd.to_numeric(player.get("group_difficulty_score"), errors="coerce")
    debutant = str(player.get("is_world_cup_debutant", "")).strip().lower()

    if pd.notna(previous_impact) and previous_impact >= 60:
        drivers.append("Strong previous World Cup impact")
    if pd.notna(previous_minutes) and previous_minutes >= 360:
        drivers.append("Experienced tournament player")
    elif pd.notna(previous_matches) and previous_matches > 0:
        drivers.append("Some prior World Cup experience")
    if debutant in {"true", "1", "yes"}:
        drivers.append("World Cup debutant profile")
    if pd.notna(age) and 22 <= age <= 29:
        drivers.append("Favourable age profile")
    elif pd.notna(age) and age <= 21:
        drivers.append("Young upside profile")
    if pd.notna(group_difficulty) and group_difficulty >= 75:
        drivers.append("Challenging group context")
    elif pd.notna(group_difficulty) and group_difficulty <= 45:
        drivers.append("More favourable group context")
    if not drivers:
        drivers.append("Limited historical feature signal available")
    return drivers[:4]


def model_limitation_note(model_summary: dict[str, object]) -> str:
    if str(model_summary.get("model_mode")) == "limited_historical_context_model":
        return (
            "The active model uses real World Cup-derived historical features. It does not yet include rich club-form, "
            "market, scouting or injury inputs, so treat it as an MVP supervised model."
        )
    return "The active model uses the best available real historical predictor set."


def status_label(status: dict[str, object], pending_when_missing: bool = False) -> str:
    if status.get("available"):
        return "available"
    return "pending" if pending_when_missing else "missing"


def render_refresh_controls() -> None:
    st.sidebar.header("Data refresh")
    if st.sidebar.button("Refresh data", type="primary"):
        result = refresh_player_data()
        clear_app_cache()
        if result.get("success"):
            st.sidebar.success(str(result.get("message")))
        else:
            st.sidebar.error(str(result.get("message")))
            for error in result.get("errors", []):
                st.sidebar.warning(error)
    if st.sidebar.button("Clear cache and reload"):
        clear_app_cache()
        st.rerun()
    st.sidebar.caption(
        "To update actual tournament impact, add new match rows to "
        "`data/raw/tournament_match_data.csv` and click Refresh data."
    )


def select_strategy() -> tuple[str, dict[str, float]]:
    st.sidebar.header("Recruitment strategy")
    strategy_options = [*VALUE_STRATEGIES.keys(), "Custom Strategy"]
    strategy = st.sidebar.radio("Strategy preset", strategy_options, index=1)
    if strategy != "Custom Strategy":
        weights = VALUE_STRATEGIES[strategy]
    else:
        st.sidebar.caption("Custom weights are normalised to sum to 1.00.")
        raw_weights = {
            "pre_tournament_expected_impact_score": st.sidebar.slider("Expected impact weight", 0.0, 1.0, 0.35, 0.05),
            "value_efficiency_score": st.sidebar.slider("Value efficiency weight", 0.0, 1.0, 0.25, 0.05),
            "age_resale_score": st.sidebar.slider("Age resale weight", 0.0, 1.0, 0.15, 0.05),
            "role_fit_score": st.sidebar.slider("Role fit weight", 0.0, 1.0, 0.10, 0.05),
            "availability_score": st.sidebar.slider("Availability weight", 0.0, 1.0, 0.10, 0.05),
            "playing_time_confidence_score": st.sidebar.slider("Minutes confidence weight", 0.0, 1.0, 0.10, 0.05),
        }
        total = sum(raw_weights.values()) or 1.0
        weights = {key: value / total for key, value in raw_weights.items()}
    st.sidebar.caption("Active value opportunity weights")
    st.sidebar.dataframe(
        pd.DataFrame(
            [{"Component": key.replace("_", " ").title(), "Weight": round(value, 3)} for key, value in weights.items()]
        ),
        hide_index=True,
        width="stretch",
    )
    return strategy, weights


def available_profiles_for_positions(positions: list[str]) -> list[str]:
    return sorted(
        [
            profile
            for profile, eligible_positions in PROFILE_ELIGIBILITY.items()
            if any(position in eligible_positions for position in positions)
        ]
    )


def optional_has_data(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def truthy_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})


def format_not_provided(value: object) -> str:
    return "Not provided" if pd.isna(value) or value == "" else str(value)


def format_optional_bool(value: object) -> str:
    if pd.isna(value):
        return "Not provided"
    parsed = str(value).strip().lower()
    if parsed in {"true", "1", "yes"}:
        return "Yes"
    if parsed in {"false", "0", "no"}:
        return "No"
    return str(value)


def format_optional_table_columns(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "World Cup debutant" in table.columns:
        table["World Cup debutant"] = table["World Cup debutant"].map(format_optional_bool)
    for col in ["Previous WC minutes", "Senior NT caps", "Major tournament experience", "Age group"]:
        if col in table.columns:
            table[col] = table[col].map(format_not_provided)
    return table


def format_score(value: object, digits: int = 1) -> str:
    parsed = pd.to_numeric(value, errors="coerce")
    return "Not yet available" if pd.isna(parsed) else f"{float(parsed):.{digits}f}"


def selected_player_from_event(event: object) -> str | None:
    try:
        points = event.selection.points  # type: ignore[attr-defined]
    except AttributeError:
        try:
            points = event["selection"]["points"]  # type: ignore[index]
        except Exception:
            return None
    if not points:
        return None
    point = points[0]
    customdata = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if customdata:
        return str(customdata[0])
    label = point.get("y") if isinstance(point, dict) else getattr(point, "y", None)
    return str(label) if label else None


def render_selectable_plot(fig, key: str) -> str | None:
    event = st.plotly_chart(
        fig,
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": False},
    )
    return selected_player_from_event(event)


def player_brief_card(df: pd.DataFrame, player_name: str | None, model_summary: dict[str, object]) -> None:
    if not player_name or player_name not in set(df["player_name"]):
        st.markdown(
            "<div class='section-note'>Select a player in the chart to inspect the recruitment case.</div>",
            unsafe_allow_html=True,
        )
        return
    player = df[df["player_name"] == player_name].iloc[0]
    percentile = impact_percentile(df, player)
    rank, rank_total = positional_rank(df, player)
    rank_text = "Not yet available" if rank is None else f"#{rank} of {rank_total} {player['position']} players"
    percentile_text = "Not yet available" if percentile is None else f"{percentile:.1f}th percentile"
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown(f"#### {player['player_name']}")
    st.caption(f"{player['country']} · {player['club']} · {player['position']} · {player['best_profile']}")
    cols = st.columns(4)
    cols[0].metric("Market value", format_currency(player.get("market_value_eur")))
    cols[1].metric("Expected WC Impact", format_score(player.get("pre_tournament_expected_impact_score")))
    cols[2].metric("Impact band", impact_band(player.get("pre_tournament_expected_impact_score")))
    cols[3].metric("Value opportunity", format_score(player.get("value_opportunity_score")))
    st.write(f"Pool context: **{percentile_text}** · Positional rank: **{rank_text}**")
    st.write(f"Recommendation: **{player['recommendation']}** · Risk: **{player['risk_band']}**")
    st.write("Why this score: " + "; ".join(expected_impact_drivers(player)) + ".")
    st.caption(model_limitation_note(model_summary))
    st.markdown("</div>", unsafe_allow_html=True)


def style_business_table(table: pd.DataFrame):
    score_columns = [
        col
        for col in [
            "Expected WC Impact",
            "Value opportunity",
            "Value opportunity score",
            "Breakout score",
            "Breakout candidate score",
        ]
        if col in table.columns
    ]

    def score_style(value: object) -> str:
        parsed = pd.to_numeric(value, errors="coerce")
        if pd.isna(parsed):
            return ""
        if parsed >= 75:
            return "background-color: #dcfce7; color: #14532d; font-weight: 700;"
        if parsed >= 55:
            return "background-color: #ecfeff; color: #164e63;"
        if parsed < 35:
            return "background-color: #fef2f2; color: #7f1d1d;"
        return ""

    styler = table.style
    if score_columns:
        return styler.map(score_style, subset=score_columns)
    return styler


def format_business_table_values(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "Market value" in table.columns:
        table["Market value"] = table["Market value"].map(format_currency)
    if "Expected impact source" in table.columns:
        table["Expected impact source"] = table["Expected impact source"].map(friendly_source)
    if "Model mode" in table.columns:
        table["Model mode"] = table["Model mode"].map(friendly_model_mode)
    if "Outcome" in table.columns:
        table["Outcome"] = table["Outcome"].replace({"Pending": "Not yet available"})
    return table


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    all_positions = sorted(df["position"].dropna().unique())
    selected_positions = st.sidebar.multiselect("Position", all_positions, default=all_positions)
    profile_options = available_profiles_for_positions(selected_positions)
    st.sidebar.caption("Available profiles depend on selected positions.")
    selected_profiles = st.sidebar.multiselect("Player role/profile", profile_options, default=profile_options)

    selected_countries = st.sidebar.multiselect("Country", sorted(df["country"].dropna().unique()))
    selected_clubs = st.sidebar.multiselect("Club", sorted(df["club"].dropna().unique()))
    selected_recommendations = st.sidebar.multiselect("Recommendation", sorted(df["recommendation"].dropna().unique()))
    selected_risk = st.sidebar.multiselect("Risk band", sorted(df["risk_band"].dropna().unique()))
    selected_outcomes = st.sidebar.multiselect("Performance outcome", sorted(df["performance_outcome"].dropna().unique()))

    if optional_has_data(df, "age_group"):
        age_group_options = [group for group in ["U21", "U23", "U25", "Prime", "Veteran"] if group in set(df["age_group"].dropna())]
        selected_age_groups = st.sidebar.multiselect("Age group", age_group_options, default=age_group_options)
    else:
        selected_age_groups = []
        st.sidebar.caption("Age-group filtering is unavailable until age values are present.")

    breakout_values = (
        pd.to_numeric(df["breakout_candidate_score"], errors="coerce").dropna()
        if "breakout_candidate_score" in df.columns
        else pd.Series(dtype=float)
    )
    breakout_range: tuple[float, float] | None = None
    if not breakout_values.empty:
        breakout_range = st.sidebar.slider(
            "Breakout candidate score range",
            0.0,
            100.0,
            (float(breakout_values.min()), float(breakout_values.max())),
            step=1.0,
        )
    else:
        st.sidebar.caption("Breakout score filtering is unavailable until the required real-derived scores exist.")

    show_breakout_only = False
    if "breakout_candidate_flag" in df.columns and truthy_series(df["breakout_candidate_flag"]).any():
        show_breakout_only = st.sidebar.checkbox("Show only breakout candidates")

    debutant_available = optional_has_data(df, "is_world_cup_debutant")
    show_debutants_only = False
    if debutant_available:
        show_debutants_only = st.sidebar.checkbox("Show only World Cup debutants")
    else:
        st.sidebar.caption("World Cup debutant filtering is disabled until debutant data is provided.")

    previous_minutes_values = (
        pd.to_numeric(df["previous_world_cup_minutes"], errors="coerce").dropna()
        if "previous_world_cup_minutes" in df.columns
        else pd.Series(dtype=float)
    )
    max_previous_wc_minutes: float | None = None
    if not previous_minutes_values.empty:
        max_previous_wc_minutes = st.sidebar.number_input(
            "Max previous World Cup minutes",
            min_value=0.0,
            value=float(previous_minutes_values.max()),
            step=90.0,
        )

    minutes_min = int(pd.to_numeric(df["recent_senior_minutes"], errors="coerce").fillna(0).min())
    minutes_max = int(pd.to_numeric(df["recent_senior_minutes"], errors="coerce").fillna(0).max())
    minutes_range = st.sidebar.slider(
        "Recent senior minutes range",
        minutes_min,
        minutes_max,
        (minutes_min, minutes_max),
        step=50,
        help=(
            "Recent senior minutes are used as a confidence and data-volume proxy. In a production version this "
            "should be standardised to club + national-team minutes over the last 12 months."
        ),
    )

    filtered = df.copy()
    filtered = filtered[filtered["position"].isin(selected_positions)]
    filtered = filtered[filtered["best_profile"].isin(selected_profiles)]
    if selected_countries:
        filtered = filtered[filtered["country"].isin(selected_countries)]
    if selected_clubs:
        filtered = filtered[filtered["club"].isin(selected_clubs)]
    if selected_recommendations:
        filtered = filtered[filtered["recommendation"].isin(selected_recommendations)]
    if selected_risk:
        filtered = filtered[filtered["risk_band"].isin(selected_risk)]
    if selected_outcomes:
        filtered = filtered[filtered["performance_outcome"].isin(selected_outcomes)]
    if selected_age_groups:
        filtered = filtered[filtered["age_group"].isin(selected_age_groups)]
    if breakout_range is not None:
        filtered = filtered[pd.to_numeric(filtered["breakout_candidate_score"], errors="coerce").between(*breakout_range)]
    if show_breakout_only:
        filtered = filtered[truthy_series(filtered["breakout_candidate_flag"])]
    if show_debutants_only:
        filtered = filtered[truthy_series(filtered["is_world_cup_debutant"])]
    if max_previous_wc_minutes is not None:
        filtered = filtered[pd.to_numeric(filtered["previous_world_cup_minutes"], errors="coerce") <= max_previous_wc_minutes]
    filtered = filtered[filtered["recent_senior_minutes"].between(*minutes_range)]
    return filtered


def render_data_status_panel(model_summary: dict[str, object], df: pd.DataFrame | None = None) -> None:
    status = check_data_availability()
    manifest = get_refresh_manifest()
    last_refresh = format_timestamp(get_last_refresh_timestamp())

    def file_status(key: str, pending_when_unavailable: bool = False) -> str:
        item = status.get(key, {})
        state = str(item.get("status", "missing"))
        if pending_when_unavailable and state in {"missing", "empty", "pending"}:
            return "pending"
        return state

    def row_count(key: str) -> int:
        return int(status.get(key, {}).get("rows", 0) or 0)

    st.markdown("#### Data files")
    data_rows = [
        ("current_player_pool.csv", file_status("current_player_pool"), row_count("current_player_pool")),
        ("market_values.csv", file_status("market_values"), row_count("market_values")),
        ("national_team_context.csv", file_status("national_team_context"), row_count("national_team_context")),
        ("player_performance_inputs.csv", file_status("player_performance_inputs"), row_count("player_performance_inputs")),
        ("tournament_match_data.csv", file_status("tournament_match_data", pending_when_unavailable=True), row_count("tournament_match_data")),
        ("world_cup_player_training_data.csv", file_status("historical_training"), row_count("historical_training")),
        ("player_dashboard_data.csv", file_status("processed_dashboard"), row_count("processed_dashboard")),
    ]
    st.dataframe(pd.DataFrame(data_rows, columns=["File", "Status", "Rows"]), hide_index=True, width="stretch")

    if df is None:
        df = pd.DataFrame()

    expected_count = (
        int(df["pre_tournament_expected_impact_score"].notna().sum())
        if "pre_tournament_expected_impact_score" in df.columns
        else int(manifest.get("expected_impact_available", 0) or 0)
    )
    actual_count = (
        int(df["actual_tournament_impact_score"].notna().sum())
        if "actual_tournament_impact_score" in df.columns
        else int(manifest.get("actual_tournament_impact_available", 0) or 0)
    )
    missing_market_values = (
        int(pd.to_numeric(df["market_value_eur"], errors="coerce").isna().sum())
        if "market_value_eur" in df.columns
        else manifest.get("missing_market_values", "unknown")
    )
    breakout_count = (
        int(df["breakout_candidate_score"].notna().sum())
        if "breakout_candidate_score" in df.columns
        else 0
    )
    debutant_status = "available" if optional_has_data(df, "is_world_cup_debutant") else "not provided"
    previous_wc_status = "available" if optional_has_data(df, "previous_world_cup_minutes") else "not provided"
    refresh_source = str(manifest.get("refresh_source", "raw files" if status["current_player_pool"].get("available") else "processed fallback"))
    pipeline_rows = [
        ("Last refresh timestamp", last_refresh),
        ("Refresh source", refresh_source),
        ("Players loaded", f"{len(df):,}" if not df.empty else f"{int(manifest.get('rows', 0) or 0):,}"),
        ("Missing market values", missing_market_values),
        ("Expected impact available", f"{expected_count:,}"),
        ("Breakout score available", f"{breakout_count:,}"),
        ("World Cup debutant data", debutant_status),
        ("Previous WC minutes data", previous_wc_status),
        ("Actual tournament impact available", f"{actual_count:,}"),
    ]
    st.markdown("#### Pipeline status")
    st.dataframe(pd.DataFrame(pipeline_rows, columns=["Metric", "Value"]), hide_index=True, width="stretch")

    baseline_available = "baseline_expected_impact_score" in df.columns and df["baseline_expected_impact_score"].notna().any()
    actual_available = actual_count > 0
    if "expected_impact_source" in df.columns:
        sources = sorted(str(source) for source in df["expected_impact_source"].dropna().unique())
        expected_source = ", ".join(sources) if sources else "unavailable"
    else:
        expected_source = "unavailable"
    model_rows = [
        (
            "Supervised model",
            f"active: {model_summary.get('model_mode')}"
            if model_summary.get("model_available")
            else "disabled: using transparent baseline fallback",
        ),
        ("Baseline fallback", "available" if baseline_available else "disabled"),
        ("Expected impact source used", expected_source),
        ("Actual impact validation", "available" if actual_available else "pending"),
    ]
    st.markdown("#### Model status")
    st.dataframe(pd.DataFrame(model_rows, columns=["Component", "Status"]), hide_index=True, width="stretch")


def render_business_status(model_summary: dict[str, object], df: pd.DataFrame) -> None:
    actual_available = "actual_tournament_impact_score" in df.columns and df["actual_tournament_impact_score"].notna().any()
    status_rows = [
        ("Player pool", "Updated" if not df.empty else "Not yet available"),
        ("Historical model", "Active" if model_summary.get("model_available") else "Baseline fallback"),
        ("Model mode", friendly_model_mode(model_summary.get("model_mode", "baseline_fallback"))),
        ("Tournament validation", business_status(actual_available, "Available", "Not yet available")),
        ("Last data refresh", format_timestamp(get_last_refresh_timestamp())),
    ]
    st.dataframe(pd.DataFrame(status_rows, columns=["Signal", "Status"]), hide_index=True, width="stretch")


def shortlist_table(df: pd.DataFrame) -> None:
    cols = [
        "player_name",
        "country",
        "club",
        "age",
        "age_group",
        "position",
        "best_profile",
        "market_value_eur",
        "recent_senior_minutes",
        "pre_tournament_expected_impact_score",
        "value_efficiency_score",
        "value_opportunity_score",
        "breakout_candidate_score",
        "recommendation",
        "recommendation_reason",
        "risk_band",
        "actual_tournament_impact_score",
        "performance_outcome",
    ]
    table = df[[col for col in cols if col in df.columns]].rename(
        columns={
            "player_name": "Player",
            "country": "Country",
            "club": "Club",
            "age": "Age",
            "age_group": "Age group",
            "position": "Position",
            "best_profile": "Profile",
            "market_value_eur": "Market value",
            "recent_senior_minutes": "Recent senior minutes",
            "pre_tournament_expected_impact_score": "Expected WC Impact",
            "value_efficiency_score": "Value efficiency",
            "value_opportunity_score": "Value opportunity",
            "breakout_candidate_score": "Breakout score",
            "recommendation": "Recommendation",
            "recommendation_reason": "Recommendation reason",
            "risk_band": "Risk band",
            "actual_tournament_impact_score": "Actual WC Impact",
            "performance_outcome": "Outcome",
        }
    )
    table["Impact band"] = table["Expected WC Impact"].map(impact_band)
    table = format_optional_table_columns(table)
    table = format_business_table_values(table)
    st.dataframe(
        style_business_table(table),
        width="stretch",
        hide_index=True,
        column_config={
            "Expected WC Impact": st.column_config.NumberColumn(format="%.1f"),
            "Value opportunity": st.column_config.NumberColumn(format="%.1f"),
            "Breakout score": st.column_config.NumberColumn(format="%.1f"),
            "Value efficiency": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def overview_page(df: pd.DataFrame, filtered_df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.markdown(
        """
<div class='executive-hero'>
  <div class='status-pill'>Recruitment Intelligence</div>
  <h2 style='margin:0 0 0.35rem 0;'>Which players should we scout or buy before the World Cup?</h2>
  <div class='section-note'>
    Identify high-upside tournament targets by combining Expected WC Impact, market value, age profile,
    role fit, availability and risk into a practical recruitment shortlist.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if model_summary.get("model_available"):
        model_mode = str(model_summary.get("model_mode", "unknown"))
        if model_mode == "limited_historical_context_model":
            st.info("Supervised model active: Limited historical model.")
        elif model_mode == "full_recruitment_model":
            st.success("Supervised model active: Full recruitment model.")
        else:
            st.info(f"Supervised model active: {friendly_model_mode(model_mode)}.")
    else:
        st.warning(str(model_summary.get("warning") or model_summary.get("readiness_message") or FALLBACK_WARNING))

    st.markdown("#### Business status")
    render_business_status(model_summary, df)

    kpis = st.columns(6)
    kpis[0].metric("Players screened", f"{len(df):,}")
    kpis[1].metric("Priority targets", f"{(df['recommendation'] == 'Priority target').sum():,}")
    kpis[2].metric("Breakout candidates", f"{truthy_series(df['breakout_candidate_flag']).sum():,}")
    kpis[3].metric("Avg Expected WC Impact", f"{df['pre_tournament_expected_impact_score'].mean():.1f}")
    kpis[4].metric("Avg Value Opportunity", f"{df['value_opportunity_score'].mean():.1f}")
    kpis[5].metric("Model mode", friendly_model_mode(model_summary.get("model_mode", "baseline_fallback")))

    st.markdown("#### Top recruitment outputs")
    top_left, top_right = st.columns(2)
    top_cols = [
        "player_name",
        "country",
        "club",
        "position",
        "market_value_eur",
        "pre_tournament_expected_impact_score",
        "value_opportunity_score",
        "recommendation",
        "risk_band",
    ]
    with top_left:
        st.markdown("##### Top value opportunities")
        top_value = filtered_df.nlargest(8, "value_opportunity_score")[[col for col in top_cols if col in filtered_df.columns]].rename(
            columns={
                "player_name": "Player",
                "country": "Country",
                "club": "Club",
                "position": "Position",
                "market_value_eur": "Market value",
                "pre_tournament_expected_impact_score": "Expected WC Impact",
                "value_opportunity_score": "Value opportunity",
                "recommendation": "Recommendation",
                "risk_band": "Risk",
            }
        )
        st.dataframe(style_business_table(format_business_table_values(top_value)), hide_index=True, width="stretch")
    with top_right:
        st.markdown("##### Top breakout candidates")
        top_breakout = filtered_df.nlargest(8, "breakout_candidate_score")[
            [col for col in [*top_cols, "breakout_candidate_score", "age"] if col in filtered_df.columns]
        ].rename(
            columns={
                "player_name": "Player",
                "country": "Country",
                "club": "Club",
                "position": "Position",
                "age": "Age",
                "market_value_eur": "Market value",
                "pre_tournament_expected_impact_score": "Expected WC Impact",
                "value_opportunity_score": "Value opportunity",
                "breakout_candidate_score": "Breakout score",
                "recommendation": "Recommendation",
                "risk_band": "Risk",
            }
        )
        st.dataframe(style_business_table(format_business_table_values(top_breakout)), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Price-to-impact quadrant")
        selected = render_selectable_plot(create_price_to_impact_quadrant_chart(filtered_df), "overview_price_impact")
    with right:
        st.markdown("#### Selected player")
        player_brief_card(filtered_df, selected, model_summary)

    visual_left, visual_right = st.columns(2)
    with visual_left:
        st.markdown("#### Top opportunities")
        selected_bar = render_selectable_plot(create_top_value_opportunities_chart(filtered_df), "overview_top_opps")
    with visual_right:
        st.markdown("#### Recommendation mix")
        st.plotly_chart(create_recommendation_breakdown_chart(filtered_df), width="stretch", config={"displayModeBar": False})
    if selected_bar:
        player_brief_card(filtered_df, selected_bar, model_summary)

    st.markdown("#### What Expected WC Impact means")
    st.markdown(
        """
Expected WC Impact estimates how strongly a player is expected to influence the 2026 World Cup based on the active
historical model. It is a tournament-impact score, not an overall player-quality rating.

Interpretation bands: 0-30 Low, 30-50 Moderate, 50-70 Strong, 70-85 High, 85+ Elite.

Next step: use the shortlist to prioritise live scouting, then add real 2026 match rows during the tournament to validate
whether targets overperform, meet expectations or underperform.
        """
    )
    st.caption(model_limitation_note(model_summary))


def recruitment_shortlist_page(df: pd.DataFrame) -> None:
    st.subheader("Recruitment Shortlist")
    controls = st.columns(6)
    max_age = controls[0].number_input("Max age", min_value=16, max_value=45, value=24)
    max_value_m = controls[1].number_input("Max market value EUR m", min_value=0.0, value=30.0, step=1.0)
    min_impact = controls[2].number_input("Min expected impact", min_value=0.0, max_value=100.0, value=0.0)
    min_opportunity = controls[3].number_input("Min value opportunity", min_value=0.0, max_value=100.0, value=0.0)
    min_minutes = controls[4].number_input("Min recent minutes", min_value=0, value=0, step=100)
    rec_filter = controls[5].multiselect("Recommendation", sorted(df["recommendation"].unique()))

    shortlist = df[
        (df["age"] <= max_age)
        & (df["market_value_eur"] <= max_value_m * 1_000_000)
        & (df["pre_tournament_expected_impact_score"].fillna(0) >= min_impact)
        & (df["value_opportunity_score"].fillna(0) >= min_opportunity)
        & (df["recent_senior_minutes"].fillna(0) >= min_minutes)
    ].copy()
    if rec_filter:
        shortlist = shortlist[shortlist["recommendation"].isin(rec_filter)]
    shortlist = shortlist.sort_values("value_opportunity_score", ascending=False)
    shortlist_table(shortlist)


def breakout_candidates_page(df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Breakout Candidates")
    st.write(
        "Identify young value-opportunity players who may use the World Cup as a breakout platform. "
        "Debutant status and previous World Cup experience are optional context fields; missing values are not inferred."
    )

    df = df.copy()
    for col in [
        "breakout_candidate_score",
        "value_opportunity_score",
        "pre_tournament_expected_impact_score",
        "age_group",
        "is_world_cup_debutant",
        "previous_world_cup_minutes",
    ]:
        if col not in df.columns:
            df[col] = pd.NA

    debutant_available = optional_has_data(df, "is_world_cup_debutant")
    previous_minutes_available = optional_has_data(df, "previous_world_cup_minutes")
    if not debutant_available or not previous_minutes_available:
        st.warning(
            "Debutant-specific filters are disabled until real debutant or previous World Cup experience fields are provided."
        )

    first = st.columns(5)
    max_age = first[0].number_input("Max age", min_value=16, max_value=45, value=23, key="breakout_max_age")
    max_value_m = first[1].number_input(
        "Max market value EUR m",
        min_value=0.0,
        value=30.0,
        step=1.0,
        key="breakout_max_value",
    )
    min_impact = first[2].number_input("Min expected impact", 0.0, 100.0, 0.0, key="breakout_min_impact")
    min_opportunity = first[3].number_input("Min value opportunity", 0.0, 100.0, 0.0, key="breakout_min_opp")
    min_breakout = first[4].number_input("Min breakout score", 0.0, 100.0, 0.0, key="breakout_min_score")

    second = st.columns(5)
    positions = sorted(df["position"].dropna().unique())
    selected_positions = second[0].multiselect("Position", positions, default=positions, key="breakout_positions")
    profile_options = available_profiles_for_positions(selected_positions)
    selected_profiles = second[1].multiselect("Profile", profile_options, default=profile_options, key="breakout_profiles")
    age_group_options = (
        [group for group in ["U21", "U23", "U25", "Prime", "Veteran"] if group in set(df["age_group"].dropna())]
        if "age_group" in df.columns
        else []
    )
    selected_age_groups = second[2].multiselect(
        "Age group",
        age_group_options,
        default=age_group_options,
        key="breakout_age_groups",
    )
    show_debutants_only = (
        second[3].checkbox("Only WC debutants", key="breakout_debutants")
        if debutant_available
        else False
    )
    max_previous_minutes = (
        second[4].number_input(
            "Max previous WC minutes",
            min_value=0.0,
            value=float(pd.to_numeric(df["previous_world_cup_minutes"], errors="coerce").max()),
            step=90.0,
            key="breakout_previous_minutes",
        )
        if previous_minutes_available
        else None
    )

    breakout = df.copy()
    breakout = breakout[pd.to_numeric(breakout["age"], errors="coerce") <= max_age]
    if "market_value_eur" in breakout.columns:
        breakout = breakout[pd.to_numeric(breakout["market_value_eur"], errors="coerce") <= max_value_m * 1_000_000]
    breakout = breakout[pd.to_numeric(breakout["pre_tournament_expected_impact_score"], errors="coerce").fillna(-1) >= min_impact]
    breakout = breakout[pd.to_numeric(breakout["value_opportunity_score"], errors="coerce").fillna(-1) >= min_opportunity]
    breakout = breakout[pd.to_numeric(breakout["breakout_candidate_score"], errors="coerce").fillna(-1) >= min_breakout]
    if selected_positions:
        breakout = breakout[breakout["position"].isin(selected_positions)]
    if selected_profiles:
        breakout = breakout[breakout["best_profile"].isin(selected_profiles)]
    if selected_age_groups:
        breakout = breakout[breakout["age_group"].isin(selected_age_groups)]
    if show_debutants_only:
        breakout = breakout[truthy_series(breakout["is_world_cup_debutant"])]
    if max_previous_minutes is not None:
        breakout = breakout[pd.to_numeric(breakout["previous_world_cup_minutes"], errors="coerce") <= max_previous_minutes]

    breakout = breakout.sort_values("breakout_candidate_score", ascending=False, na_position="last")
    st.markdown("#### Breakout candidates: age vs expected impact")
    selected = render_selectable_plot(create_breakout_candidates_chart(breakout), "breakout_age_impact")
    if selected:
        player_brief_card(breakout, selected, model_summary)

    cols = [
        "player_name",
        "country",
        "club",
        "league",
        "age",
        "age_group",
        "position",
        "best_profile",
        "market_value_eur",
        "pre_tournament_expected_impact_score",
        "value_opportunity_score",
        "breakout_candidate_score",
        "is_world_cup_debutant",
        "previous_world_cup_minutes",
        "senior_national_team_caps",
        "major_tournament_experience",
        "recommendation",
        "risk_band",
    ]
    table = breakout[[col for col in cols if col in breakout.columns]].rename(
        columns={
            "player_name": "Player",
            "country": "Country",
            "club": "Club",
            "league": "League",
            "age": "Age",
            "age_group": "Age group",
            "position": "Position",
            "best_profile": "Profile",
            "market_value_eur": "Market value",
            "pre_tournament_expected_impact_score": "Expected WC Impact",
            "value_opportunity_score": "Value opportunity",
            "breakout_candidate_score": "Breakout score",
            "is_world_cup_debutant": "World Cup debutant",
            "previous_world_cup_minutes": "Previous WC minutes",
            "senior_national_team_caps": "Senior NT caps",
            "major_tournament_experience": "Major tournament experience",
            "recommendation": "Recommendation",
            "risk_band": "Risk band",
        }
    )
    table = format_optional_table_columns(table)
    table = format_business_table_values(table)
    st.dataframe(
        style_business_table(table),
        hide_index=True,
        width="stretch",
        column_config={
            "Expected WC Impact": st.column_config.NumberColumn(format="%.1f"),
            "Value opportunity": st.column_config.NumberColumn(format="%.1f"),
            "Breakout score": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def charts_page(filtered_df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Market & Opportunity Intelligence")
    st.caption("Use these visuals to challenge the shortlist, compare price against projected tournament impact and isolate risk-adjusted value.")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("#### Market value vs value opportunity")
        selected_value = render_selectable_plot(create_market_value_vs_value_opportunity_chart(filtered_df), "chart_value_market")
    with chart_b:
        st.markdown("#### Age vs value opportunity")
        selected_age = render_selectable_plot(create_age_vs_value_opportunity_chart(filtered_df), "chart_age_value")
    selected = selected_value or selected_age
    if selected:
        player_brief_card(filtered_df, selected, model_summary)

    chart_c, chart_d = st.columns(2)
    with chart_c:
        st.markdown("#### Expected impact vs value efficiency")
        selected_efficiency = render_selectable_plot(create_expected_impact_vs_value_efficiency_chart(filtered_df), "chart_impact_efficiency")
    with chart_d:
        st.markdown("#### Recommendation breakdown")
        st.plotly_chart(create_recommendation_breakdown_chart(filtered_df), width="stretch", config={"displayModeBar": False})
    if selected_efficiency:
        player_brief_card(filtered_df, selected_efficiency, model_summary)

    chart_e, chart_f = st.columns(2)
    with chart_e:
        st.markdown("#### Predicted vs actual impact")
        st.plotly_chart(create_predicted_vs_actual_impact_chart(filtered_df), width="stretch", config={"displayModeBar": False})
    with chart_f:
        st.markdown("#### Performance delta")
        st.plotly_chart(create_performance_delta_distribution_chart(filtered_df), width="stretch", config={"displayModeBar": False})

    st.markdown("#### Risk vs opportunity matrix")
    selected_risk = render_selectable_plot(create_risk_vs_opportunity_matrix(filtered_df), "chart_risk_opportunity")
    if selected_risk:
        player_brief_card(filtered_df, selected_risk, model_summary)


def profile_rankings_page(filtered_df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Role Rankings")
    st.caption("Rank players by recruitment role profile. Profile weights are transparent business scouting heuristics, separate from the supervised Expected WC Impact model.")
    selected_profile = st.selectbox("Select profile", list(PLAYER_PROFILES.keys()))
    score_col = get_profile_score_column(selected_profile)
    eligible_positions = get_eligible_positions(selected_profile)
    ranking_df = filtered_df[filtered_df["position"].isin(eligible_positions)] if eligible_positions else filtered_df
    selected = render_selectable_plot(create_top_players_chart(ranking_df, selected_profile), "profile_rankings")
    if selected:
        player_brief_card(ranking_df, selected, model_summary)


def player_profile_page(df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Player Profile")
    selected_player = st.selectbox("Select player", sorted(df["player_name"].unique()))
    player = df[df["player_name"] == selected_player].iloc[0]

    percentile = impact_percentile(df, player)
    rank, rank_total = positional_rank(df, player)
    percentile_text = "Not yet available" if percentile is None else f"{percentile:.1f}th percentile"
    rank_text = "Not yet available" if rank is None else f"#{rank} of {rank_total}"

    st.markdown(
        f"""
<div class='executive-hero'>
  <div class='status-pill'>{player['position']} · {player['best_profile']}</div>
  <h2 style='margin:0 0 0.25rem 0;'>{player['player_name']}</h2>
  <div class='section-note'>{player['country']} · {player['club']} · World Cup debutant: {format_optional_bool(player.get('is_world_cup_debutant'))}</div>
</div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(6)
    cols[0].metric("Market value", format_currency(player.get("market_value_eur")))
    cols[1].metric("Expected WC Impact", format_score(player.get("pre_tournament_expected_impact_score")))
    cols[2].metric("Impact band", impact_band(player.get("pre_tournament_expected_impact_score")))
    cols[3].metric("Pool percentile", percentile_text)
    cols[4].metric("Position rank", rank_text)
    cols[5].metric("Value opportunity", format_score(player.get("value_opportunity_score")))

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Recruitment read")
        st.markdown(f"<div class='scout-note'>{scouting_summary(player)}</div>", unsafe_allow_html=True)
        st.write(f"Recommendation: **{player['recommendation']}**")
        st.write(player["recommendation_reason"])
        st.write(f"Risk: **{player['risk_band']}** · {player['risk_reason']}")
    with right:
        st.markdown("#### Why this Expected WC Impact?")
        st.markdown(
            "\n".join(f"- {driver}" for driver in expected_impact_drivers(player))
            + f"\n- Model mode: {friendly_model_mode(model_summary.get('model_mode', 'baseline_fallback'))}"
        )
        st.caption(
            "Expected WC Impact estimates tournament influence, not overall player quality. "
            + model_limitation_note(model_summary)
        )

    score_metrics = [
        metric
        for metric in [
            "attacking_score",
            "creativity_score",
            "progression_score",
            "defensive_score",
            "possession_score",
            "overall_score",
        ]
        if metric in df.columns
    ]
    if score_metrics:
        chart_left, chart_right = st.columns(2)
        with chart_left:
            st.markdown("#### Player profile radar")
            st.plotly_chart(create_radar_chart(player, df, score_metrics), width="stretch", config={"displayModeBar": False})
        with chart_right:
            st.markdown("#### Versus positional average")
            st.plotly_chart(create_percentile_bar_chart(player, df, score_metrics), width="stretch", config={"displayModeBar": False})

    percentile_metrics = [
        metric
        for metric in ["goals", "assists", "key_passes", "progressive_passes", "progressive_carries", "tackles", "interceptions"]
        if f"{metric}_percentile" in player.index
    ]
    if percentile_metrics:
        st.markdown("#### Metric percentiles")
        st.plotly_chart(create_metric_percentile_chart(player, percentile_metrics), width="stretch", config={"displayModeBar": False})


def model_page(model_summary: dict[str, object], readiness: dict[str, object]) -> None:
    st.subheader("Expected Impact Model")
    st.markdown("#### Historical model readiness")
    readiness_rows = [
        ("Model mode", friendly_model_mode(model_summary.get("model_mode", readiness.get("selected_model_mode", "baseline_fallback")))),
        ("Full recruitment model", "available" if readiness.get("full_recruitment_model_available") else "not yet available"),
        ("Limited historical model", "available" if readiness.get("limited_historical_context_model_available") else "not available"),
        ("Historical training file", "available" if readiness.get("training_file_exists") else "missing"),
        ("Template file", "available" if readiness.get("template_exists") else "missing"),
        ("Training rows", readiness.get("row_count", 0)),
        ("Target rows", readiness.get("target_available_rows", 0)),
        ("Usable target rows", readiness.get("usable_target_rows", 0)),
        ("Tournament years", ", ".join(str(year) for year in readiness.get("tournament_years_available", [])) or "none"),
        ("Required columns", "available" if readiness.get("required_columns_present") else "missing"),
        ("Predictor coverage sufficient", "yes" if readiness.get("predictor_coverage_sufficient") else "no"),
        ("Missing predictor groups", ", ".join(str(group) for group in readiness.get("missing_predictor_groups", [])) or "none"),
        ("Activation status", readiness.get("activation_status", "unknown")),
        ("Model status", model_summary.get("model_status", readiness.get("model_status", "unknown"))),
        ("Can train", "yes" if readiness.get("can_train") else "no"),
        ("Supervised model", "available" if model_summary.get("model_available") else "disabled"),
    ]
    st.dataframe(pd.DataFrame(readiness_rows, columns=["Check", "Status"]), hide_index=True, width="stretch")
    if model_summary.get("model_available") and model_summary.get("model_mode") == "limited_historical_context_model":
        st.info(
            "The supervised model is active, but currently limited to real World Cup-derived historical context "
            "features. Add richer real pre-tournament predictors to enable the full recruitment model."
        )
    elif not readiness.get("can_train"):
        st.warning(str(readiness.get("readiness_message") or readiness.get("disabled_reason") or "Historical training data is not ready."))
        if readiness.get("activation_status") == "disabled_insufficient_predictor_coverage":
            st.info(
                "Next action: enrich historical rows with real pre-tournament predictor values such as market value, "
                "club-season minutes, goals, assists, senior caps, role fit or availability."
            )
    if readiness.get("missing_required_columns"):
        st.write("Missing required columns: " + ", ".join(str(col) for col in readiness["missing_required_columns"]))
    feature_availability = readiness.get("feature_availability")
    if isinstance(feature_availability, pd.DataFrame) and not feature_availability.empty:
        display_columns = [
            "feature",
            "feature_group",
            "requirement",
            "non_null",
            "pct_non_null",
            "minimum_pct_required",
            "usable_for_modelling",
        ]
        st.markdown("#### Feature coverage")
        st.dataframe(feature_availability[display_columns], width="stretch", hide_index=True)

    st.markdown("#### Training result")
    if model_summary.get("model_available"):
        st.success(f"Supervised model available: {model_summary.get('selected_model_name')}")
        st.write(f"Model mode: **{friendly_model_mode(model_summary.get('model_mode'))}**")
        st.write(f"Training rows: {model_summary.get('training_row_count')}")
        metrics = model_summary.get("metrics")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            st.dataframe(metrics, width="stretch", hide_index=True)
        excluded = model_summary.get("features_excluded")
        if isinstance(excluded, pd.DataFrame) and not excluded.empty:
            st.markdown("#### Features excluded")
            st.dataframe(excluded, width="stretch", hide_index=True)
        importance = model_summary.get("feature_importance")
        if isinstance(importance, pd.DataFrame) and not importance.empty:
            st.markdown("#### Feature importance")
            st.dataframe(importance.head(20), width="stretch", hide_index=True)
    else:
        st.warning(model_summary.get("warning", FALLBACK_WARNING))
        st.write(f"Model status: `{model_summary.get('model_status')}`")
        st.write(f"Expected historical path: `{model_summary.get('data_source_path')}`")
    features = model_summary.get("model_features")
    if isinstance(features, list) and features:
        st.markdown("#### Current supervised feature plan")
        st.write(", ".join(str(feature) for feature in features))


def data_model_ops_page(df: pd.DataFrame, model_summary: dict[str, object], readiness: dict[str, object]) -> None:
    st.subheader("Data & Model Ops")
    st.caption("Technical transparency for refresh status, source availability, model readiness and validation artifacts.")
    render_data_status_panel(model_summary, df)
    st.divider()
    model_page(model_summary, readiness)


def methodology_page() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
The dashboard uses real data only. No synthetic data is used. If real historical or tournament data is missing, the relevant model or validation section is disabled or marked as not yet available.

Expected WC Impact estimates how strongly a player is expected to influence the 2026 World Cup based on the active historical model. It is a tournament-impact score, not an overall player-quality rating.

The active MVP model is a limited supervised historical model trained on real player-tournament rows from past World Cups. It uses real World Cup-derived context and experience features. Rich recruitment predictors such as club-season form, market context, role fit and availability can be added later through compliant data exports.

The expected-impact model is designed to handle World Cup debutants. It does not require a player to have appeared at previous World Cups. Previous World Cup experience is included only as an optional signal.

The model is trained on player-tournament observations, not on repeated appearances by the same players. This allows the model to generalise from historical player profiles to new 2026 players.

`value_opportunity_score` is a recruitment business metric. It combines expected impact, value efficiency, age resale, role fit, availability and playing-time confidence. Strategy presets and custom weights change this ranking without changing the expected-impact model.

Actual tournament impact is updated only from `data/raw/tournament_match_data.csv`. Add real match rows after each World Cup match and click Refresh data.
        """
    )


def main() -> None:
    render_refresh_controls()
    strategy, weights = select_strategy()
    model_summary = get_model_summary()
    model_readiness = get_model_readiness()
    raw_df = get_data()

    st.title(APP_NAME)
    st.caption(APP_SUBTITLE)

    if raw_df.empty:
        st.warning("Processed dashboard data is missing. Click Refresh data after adding real CSV inputs, or run the refresh pipeline.")
        render_data_status_panel(model_summary, raw_df)
        st.stop()

    df = calculate_value_opportunity_score(raw_df, strategy=strategy, custom_weights=weights if strategy == "Custom Strategy" else None)
    filtered_df = apply_sidebar_filters(df)
    if filtered_df.empty:
        st.warning("No players match the current filters. Showing the full dataset instead.")
        filtered_df = df

    tabs = st.tabs(
        [
            "Overview",
            "Recruitment Shortlist",
            "Breakout Candidates",
            "Market & Opportunity",
            "Player Profile",
            "Role Rankings",
            "Data & Model Ops",
            "Methodology",
        ]
    )
    with tabs[0]:
        overview_page(df, filtered_df, model_summary)
    with tabs[1]:
        recruitment_shortlist_page(filtered_df)
    with tabs[2]:
        breakout_candidates_page(filtered_df, model_summary)
    with tabs[3]:
        charts_page(filtered_df, model_summary)
    with tabs[4]:
        player_profile_page(filtered_df, model_summary)
    with tabs[5]:
        profile_rankings_page(filtered_df, model_summary)
    with tabs[6]:
        data_model_ops_page(df, model_summary, model_readiness)
    with tabs[7]:
        methodology_page()


if __name__ == "__main__":
    main()
