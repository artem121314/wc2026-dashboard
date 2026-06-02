from __future__ import annotations

import sys
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
    create_performance_delta_distribution_chart,
    create_predicted_vs_actual_impact_chart,
    create_price_to_impact_quadrant_chart,
    create_recommendation_breakdown_chart,
    create_risk_vs_opportunity_matrix,
    create_top_players_chart,
    create_top_value_opportunities_chart,
)


st.set_page_config(
    page_title="WC 2026 Value Opportunity Dashboard",
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
    last_refresh = get_last_refresh_timestamp() or "No refresh manifest found"

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
        ("Supervised model", "available" if model_summary.get("model_available") else "disabled"),
        ("Baseline fallback", "available" if baseline_available else "disabled"),
        ("Expected impact source used", expected_source),
        ("Actual impact validation", "available" if actual_available else "pending"),
    ]
    st.markdown("#### Model status")
    st.dataframe(pd.DataFrame(model_rows, columns=["Component", "Status"]), hide_index=True, width="stretch")


def shortlist_table(df: pd.DataFrame) -> None:
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
        "recent_senior_minutes",
        "pre_tournament_expected_impact_score",
        "expected_impact_source",
        "value_efficiency_score",
        "value_opportunity_score",
        "breakout_candidate_score",
        "is_world_cup_debutant",
        "previous_world_cup_minutes",
        "recommendation",
        "recommendation_reason",
        "risk_band",
        "risk_reason",
        "actual_tournament_impact_score",
        "performance_delta",
        "performance_outcome",
    ]
    table = df[[col for col in cols if col in df.columns]].rename(
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
            "recent_senior_minutes": "Recent senior minutes",
            "pre_tournament_expected_impact_score": "Expected WC Impact",
            "expected_impact_source": "Expected impact source",
            "value_efficiency_score": "Value efficiency",
            "value_opportunity_score": "Value opportunity score",
            "breakout_candidate_score": "Breakout candidate score",
            "is_world_cup_debutant": "World Cup debutant",
            "previous_world_cup_minutes": "Previous WC minutes",
            "recommendation": "Recommendation",
            "recommendation_reason": "Recommendation reason",
            "risk_band": "Risk band",
            "risk_reason": "Risk reason",
            "actual_tournament_impact_score": "Actual WC Impact",
            "performance_delta": "Performance delta",
            "performance_outcome": "Outcome",
        }
    )
    table = format_optional_table_columns(table)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Market value": st.column_config.NumberColumn(format="EUR %.0f"),
            "Expected WC Impact": st.column_config.NumberColumn(format="%.1f"),
            "Value opportunity score": st.column_config.NumberColumn(format="%.1f"),
            "Breakout candidate score": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def overview_page(df: pd.DataFrame, filtered_df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Recruitment Decision Support")
    st.write(
        "Which players should a club scout or buy before the 2026 World Cup because they are undervalued relative "
        "to their expected tournament impact?"
    )
    if not model_summary.get("model_available"):
        st.warning(FALLBACK_WARNING)
    render_data_status_panel(model_summary, df)

    kpis = st.columns(5)
    kpis[0].metric("Players", f"{len(df):,}")
    kpis[1].metric("Avg expected impact", f"{df['pre_tournament_expected_impact_score'].mean():.1f}")
    kpis[2].metric("Priority targets", f"{(df['recommendation'] == 'Priority target').sum():,}")
    kpis[3].metric("Pending outcomes", f"{(df['performance_outcome'] == 'Pending').sum():,}")
    kpis[4].metric("Filtered players", f"{len(filtered_df):,}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Price-to-impact quadrant")
        st.plotly_chart(create_price_to_impact_quadrant_chart(filtered_df), width="stretch")
    with right:
        st.markdown("#### Top value opportunities")
        st.plotly_chart(create_top_value_opportunities_chart(filtered_df), width="stretch")


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


def breakout_candidates_page(df: pd.DataFrame) -> None:
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
    st.plotly_chart(create_breakout_candidates_chart(breakout), width="stretch")

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
            "value_opportunity_score": "Value opportunity score",
            "breakout_candidate_score": "Breakout candidate score",
            "is_world_cup_debutant": "World Cup debutant",
            "previous_world_cup_minutes": "Previous WC minutes",
            "senior_national_team_caps": "Senior NT caps",
            "major_tournament_experience": "Major tournament experience",
            "recommendation": "Recommendation",
            "risk_band": "Risk band",
        }
    )
    table = format_optional_table_columns(table)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Market value": st.column_config.NumberColumn(format="EUR %.0f"),
            "Expected WC Impact": st.column_config.NumberColumn(format="%.1f"),
            "Value opportunity score": st.column_config.NumberColumn(format="%.1f"),
            "Breakout candidate score": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def charts_page(filtered_df: pd.DataFrame) -> None:
    st.subheader("Recruitment Charts")
    chart_a, chart_b = st.columns(2)
    with chart_a:
        st.markdown("#### Market value vs value opportunity")
        st.plotly_chart(create_market_value_vs_value_opportunity_chart(filtered_df), width="stretch")
    with chart_b:
        st.markdown("#### Age vs value opportunity")
        st.plotly_chart(create_age_vs_value_opportunity_chart(filtered_df), width="stretch")

    chart_c, chart_d = st.columns(2)
    with chart_c:
        st.markdown("#### Expected impact vs value efficiency")
        st.plotly_chart(create_expected_impact_vs_value_efficiency_chart(filtered_df), width="stretch")
    with chart_d:
        st.markdown("#### Recommendation breakdown")
        st.plotly_chart(create_recommendation_breakdown_chart(filtered_df), width="stretch")

    chart_e, chart_f = st.columns(2)
    with chart_e:
        st.markdown("#### Predicted vs actual impact")
        st.plotly_chart(create_predicted_vs_actual_impact_chart(filtered_df), width="stretch")
    with chart_f:
        st.markdown("#### Performance delta")
        st.plotly_chart(create_performance_delta_distribution_chart(filtered_df), width="stretch")

    st.markdown("#### Risk vs opportunity matrix")
    st.plotly_chart(create_risk_vs_opportunity_matrix(filtered_df), width="stretch")


def profile_rankings_page(filtered_df: pd.DataFrame) -> None:
    st.subheader("Top Players By Profile")
    selected_profile = st.selectbox("Select profile", list(PLAYER_PROFILES.keys()))
    score_col = get_profile_score_column(selected_profile)
    eligible_positions = get_eligible_positions(selected_profile)
    ranking_df = filtered_df[filtered_df["position"].isin(eligible_positions)] if eligible_positions else filtered_df
    st.plotly_chart(create_top_players_chart(ranking_df, selected_profile), width="stretch")


def player_profile_page(df: pd.DataFrame) -> None:
    st.subheader("Player Profile")
    selected_player = st.selectbox("Select player", sorted(df["player_name"].unique()))
    player = df[df["player_name"] == selected_player].iloc[0]
    def score_text(value: object) -> str:
        parsed = pd.to_numeric(value, errors="coerce")
        return "Unavailable" if pd.isna(parsed) else f"{float(parsed):.1f}"

    cols = st.columns(5)
    cols[0].metric("Player", player["player_name"])
    cols[1].metric("Market value", format_currency(player["market_value_eur"]))
    cols[2].metric("Expected impact", score_text(player["pre_tournament_expected_impact_score"]))
    cols[3].metric("Value opportunity", score_text(player["value_opportunity_score"]))
    cols[4].metric("Risk", player["risk_band"])
    st.markdown(f"<div class='scout-note'>{scouting_summary(player)}</div>", unsafe_allow_html=True)
    st.write(f"Recommendation: **{player['recommendation']}**")
    st.write(player["recommendation_reason"])
    st.write(f"Risk reason: {player['risk_reason']}")


def model_page(model_summary: dict[str, object], readiness: dict[str, object]) -> None:
    st.subheader("Expected Impact Model")
    st.markdown("#### Historical model readiness")
    readiness_rows = [
        ("Historical training file", "available" if readiness.get("training_file_exists") else "missing"),
        ("Template file", "available" if readiness.get("template_exists") else "missing"),
        ("Training rows", readiness.get("row_count", 0)),
        ("Target rows", readiness.get("target_available_rows", 0)),
        ("Tournament years", ", ".join(str(year) for year in readiness.get("tournament_years_available", [])) or "none"),
        ("Required columns", "available" if readiness.get("required_columns_present") else "missing"),
        ("Can train", "yes" if readiness.get("can_train") else "no"),
        ("Supervised model", "available" if model_summary.get("model_available") else "disabled"),
    ]
    st.dataframe(pd.DataFrame(readiness_rows, columns=["Check", "Status"]), hide_index=True, width="stretch")
    if not readiness.get("can_train"):
        st.warning(str(readiness.get("disabled_reason", "Historical training data is not ready.")))
    if readiness.get("missing_required_columns"):
        st.write("Missing required columns: " + ", ".join(str(col) for col in readiness["missing_required_columns"]))

    st.markdown("#### Training result")
    if model_summary.get("model_available"):
        st.success(f"Supervised model available: {model_summary.get('selected_model_name')}")
        st.write(f"Training rows: {model_summary.get('training_row_count')}")
        metrics = model_summary.get("metrics")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            st.dataframe(metrics, width="stretch", hide_index=True)
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


def methodology_page() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
The dashboard uses real data only. No synthetic data is used. If real historical or tournament data is missing, the relevant model or validation section is disabled or marked as pending.

The supervised expected-impact model trains only from `data/historical/world_cup_player_training_data.csv` when real curated historical rows are available. If that file is missing, empty or invalid, `pre_tournament_expected_impact_score` falls back to `baseline_expected_impact_score` only when the required current-player features exist.

The expected-impact model is designed to handle World Cup debutants. It does not require a player to have appeared at previous World Cups. Previous World Cup experience is included only as an optional signal. For young players, the model relies more on club performance, expected national-team role, recent minutes, age/upside, market value and team context.

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

    st.title("WC 2026 Value Opportunity Dashboard")
    st.caption("Recruitment decision support for expected World Cup impact, value opportunity and validation.")

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
            "Charts",
            "Player Profile",
            "Profile Rankings",
            "Model",
            "Methodology",
        ]
    )
    with tabs[0]:
        overview_page(df, filtered_df, model_summary)
    with tabs[1]:
        recruitment_shortlist_page(filtered_df)
    with tabs[2]:
        breakout_candidates_page(filtered_df)
    with tabs[3]:
        charts_page(filtered_df)
    with tabs[4]:
        player_profile_page(filtered_df)
    with tabs[5]:
        profile_rankings_page(filtered_df)
    with tabs[6]:
        model_page(model_summary, model_readiness)
    with tabs[7]:
        methodology_page()


if __name__ == "__main__":
    main()
