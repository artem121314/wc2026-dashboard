from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from data_loader import load_player_data
from feature_engineering import VALUE_STRATEGIES, calculate_value_opportunity_score
from model import FALLBACK_WARNING, confidence_level, explain_prediction, train_expected_impact_model
from player_profiles import PLAYER_PROFILES, get_eligible_positions, get_profile_score_column
from utils import format_currency, get_risks, get_strengths, scouting_summary, similar_players
from visualisations import (
    create_age_vs_value_opportunity_chart,
    create_expected_impact_distribution_chart,
    create_market_value_vs_expected_impact_chart,
    create_market_value_vs_value_opportunity_chart,
    create_metric_percentile_chart,
    create_percentile_bar_chart,
    create_performance_delta_distribution_chart,
    create_predicted_vs_actual_impact_chart,
    create_radar_chart,
    create_top_players_chart,
    create_top_value_opportunities_chart,
)


st.set_page_config(
    page_title="WC 2026 Recruitment Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
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
    .small-muted { color: #64748b; font-size: 0.9rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_player_data()


@st.cache_resource(show_spinner=False)
def get_model_summary() -> dict[str, object]:
    return train_expected_impact_model()


def select_strategy() -> str:
    st.sidebar.header("Recruitment strategy")
    return st.sidebar.radio(
        "Strategy preset",
        list(VALUE_STRATEGIES.keys()),
        index=0,
        help="Updates the business ranking score without changing the expected-impact model.",
    )


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    selected_positions = st.sidebar.multiselect(
        "Position",
        sorted(df["position"].unique()),
        default=sorted(df["position"].unique()),
    )
    selected_profiles = st.sidebar.multiselect(
        "Player role/profile",
        sorted(df["best_profile"].unique()),
        default=sorted(df["best_profile"].unique()),
    )
    selected_countries = st.sidebar.multiselect("Country", sorted(df["country"].unique()))
    selected_clubs = st.sidebar.multiselect("Club", sorted(df["club"].unique()))
    selected_leagues = st.sidebar.multiselect("League", sorted(df["league"].unique()))
    selected_groups = st.sidebar.multiselect("World Cup group", sorted(df.get("group", pd.Series()).dropna().unique()))
    selected_outcomes = st.sidebar.multiselect(
        "Performance outcome",
        sorted(df["performance_outcome"].dropna().unique()),
    )

    age_range = st.sidebar.slider(
        "Age range",
        int(df["age"].min()),
        int(df["age"].max()),
        (int(df["age"].min()), int(df["age"].max())),
    )
    value_range = st.sidebar.slider(
        "Market value range (EUR m)",
        float(df["market_value_eur"].min() / 1_000_000),
        float(df["market_value_eur"].max() / 1_000_000),
        (float(df["market_value_eur"].min() / 1_000_000), float(df["market_value_eur"].max() / 1_000_000)),
        step=0.5,
    )
    impact_range = st.sidebar.slider(
        "Pre-tournament expected impact",
        0,
        100,
        (0, 100),
        step=1,
    )
    opportunity_range = st.sidebar.slider(
        "Value opportunity score",
        0,
        100,
        (0, 100),
        step=1,
    )

    filtered = df.copy()
    filtered = filtered[filtered["position"].isin(selected_positions)]
    filtered = filtered[filtered["best_profile"].isin(selected_profiles)]
    if selected_countries:
        filtered = filtered[filtered["country"].isin(selected_countries)]
    if selected_clubs:
        filtered = filtered[filtered["club"].isin(selected_clubs)]
    if selected_leagues:
        filtered = filtered[filtered["league"].isin(selected_leagues)]
    if selected_groups:
        filtered = filtered[filtered["group"].isin(selected_groups)]
    if selected_outcomes:
        filtered = filtered[filtered["performance_outcome"].isin(selected_outcomes)]

    filtered = filtered[
        filtered["age"].between(*age_range)
        & filtered["market_value_eur"].between(value_range[0] * 1_000_000, value_range[1] * 1_000_000)
        & filtered["pre_tournament_expected_impact_score"].between(*impact_range)
        & filtered["value_opportunity_score"].between(*opportunity_range)
    ]
    return filtered


def display_shortlist_table(df: pd.DataFrame) -> None:
    st.markdown("#### Pre-tournament recruitment shortlist")
    table_cols = [
        "player_name",
        "country",
        "club",
        "league",
        "age",
        "position",
        "best_profile",
        "market_value_eur",
        "pre_tournament_expected_impact_score",
        "baseline_expected_impact_score",
        "value_efficiency_score",
        "value_opportunity_score",
        "recommendation",
        "actual_tournament_impact_score",
        "performance_delta",
        "performance_outcome",
    ]
    table = df[[col for col in table_cols if col in df.columns]].rename(
        columns={
            "player_name": "Player",
            "country": "Country",
            "club": "Club",
            "league": "League",
            "age": "Age",
            "position": "Position",
            "best_profile": "Profile",
            "market_value_eur": "Market value",
            "pre_tournament_expected_impact_score": "Pre-tournament expected impact",
            "baseline_expected_impact_score": "Baseline expected impact",
            "value_efficiency_score": "Value efficiency",
            "value_opportunity_score": "Value opportunity score",
            "recommendation": "Recommendation",
            "actual_tournament_impact_score": "Actual tournament impact",
            "performance_delta": "Performance delta",
            "performance_outcome": "Outcome",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Market value": st.column_config.NumberColumn(format="EUR %.0f"),
            "Pre-tournament expected impact": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "Baseline expected impact": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "Value opportunity score": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
        },
    )


def overview_page(df: pd.DataFrame, filtered_df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Pre-Tournament Recruitment Analytics")
    st.write(
        "This dashboard is built for an English club recruitment workflow before the 2026 FIFA World Cup. "
        "It ranks players by expected tournament impact and value opportunity, then keeps the structure ready "
        "to validate those predictions against actual World Cup performance when match data becomes available."
    )
    if not model_summary.get("model_available"):
        st.warning(FALLBACK_WARNING)

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Players", f"{len(df):,}")
    kpi_cols[1].metric("Countries", f"{df['country'].nunique():,}")
    kpi_cols[2].metric("Avg market value", format_currency(df["market_value_eur"].mean()))
    kpi_cols[3].metric("Avg expected impact", f"{df['pre_tournament_expected_impact_score'].mean():.1f}")
    kpi_cols[4].metric("Priority targets", f"{(df['recommendation'] == 'Priority target').sum():,}")
    kpi_cols[5].metric("Actual outcomes", f"{df['actual_tournament_impact_score'].notna().sum():,}")
    if "data_refresh_date" in df.columns:
        st.caption(f"Dataset refresh: {df['data_refresh_date'].dropna().iloc[0]}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Market value vs expected impact")
        st.plotly_chart(create_market_value_vs_expected_impact_chart(filtered_df), width="stretch")
    with right:
        st.markdown("#### Top value opportunities")
        st.plotly_chart(create_top_value_opportunities_chart(filtered_df), width="stretch")

    lower_left, lower_right = st.columns(2)
    with lower_left:
        st.markdown("#### Market value vs value opportunity")
        st.plotly_chart(create_market_value_vs_value_opportunity_chart(filtered_df), width="stretch")
    with lower_right:
        st.markdown("#### Expected impact distribution")
        st.plotly_chart(create_expected_impact_distribution_chart(filtered_df), width="stretch")


def shortlist_page(filtered_df: pd.DataFrame) -> None:
    st.subheader("Recruitment Shortlist")
    sort_col = st.selectbox(
        "Sort players by",
        [
            "value_opportunity_score",
            "pre_tournament_expected_impact_score",
            "value_efficiency_score",
            "baseline_expected_impact_score",
            "market_value_eur",
            "age",
        ],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    sorted_df = filtered_df.sort_values(sort_col, ascending=False)
    display_shortlist_table(sorted_df)

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Age vs value opportunity")
        st.plotly_chart(create_age_vs_value_opportunity_chart(sorted_df), width="stretch")
    with chart_right:
        st.markdown("#### Market value vs value opportunity")
        st.plotly_chart(create_market_value_vs_value_opportunity_chart(sorted_df), width="stretch")


def player_profile_page(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("Player Profile")
    player_pool = filtered_df if not filtered_df.empty else df
    selected_player = st.selectbox(
        "Select player",
        player_pool["player_name"].sort_values().tolist(),
    )
    player = df[df["player_name"] == selected_player].iloc[0]

    top = st.columns([1.2, 1, 1, 1, 1])
    top[0].metric("Player", player["player_name"])
    top[1].metric("Market value", format_currency(player["market_value_eur"]))
    top[2].metric("Expected impact", f"{player['pre_tournament_expected_impact_score']:.1f}/100")
    top[3].metric("Value opportunity", f"{player['value_opportunity_score']:.1f}/100")
    top[4].metric("Outcome", player["performance_outcome"])
    if "market_value_source" in player.index:
        st.caption(f"Market value source: {player['market_value_source']}")

    info_cols = st.columns(5)
    info_cols[0].metric("Country", player["country"])
    info_cols[1].metric("Club", player["club"])
    info_cols[2].metric("Position", player["position"])
    info_cols[3].metric("Profile", player["best_profile"])
    info_cols[4].metric("Confidence", confidence_level(player))

    context_cols = st.columns(5)
    context_cols[0].metric("Baseline impact", f"{player['baseline_expected_impact_score']:.1f}/100")
    context_cols[1].metric("Value efficiency", f"{player['value_efficiency_score']:.1f}/100")
    context_cols[2].metric("Tactical fit", f"{player.get('role_fit_score', 0):.1f}/100")
    context_cols[3].metric("Availability", f"{player.get('availability_score', 0):.1f}/100")
    context_cols[4].metric("Draw context", f"{player.get('tournament_draw_score', 0):.1f}/100")

    st.markdown(f"<div class='scout-note'>{scouting_summary(player)}</div>", unsafe_allow_html=True)

    strengths_col, risks_col = st.columns(2)
    with strengths_col:
        st.markdown("#### Key strengths")
        for item in get_strengths(player):
            st.write(f"- {item}")
    with risks_col:
        st.markdown("#### Weaknesses / risks")
        for item in get_risks(player):
            st.write(f"- {item}")

    score_metrics = [
        "attacking_score",
        "creativity_score",
        "progression_score",
        "defensive_score",
        "possession_score",
        "overall_score",
    ]
    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        st.markdown("#### Role radar")
        st.plotly_chart(create_radar_chart(player, df, score_metrics), width="stretch")
    with chart_col_2:
        st.markdown("#### Versus positional average")
        st.plotly_chart(create_percentile_bar_chart(player, df, score_metrics), width="stretch")

    percentile_metrics = [
        "non_penalty_goals",
        "xg_proxy",
        "xa_proxy",
        "key_passes",
        "progressive_passes",
        "progressive_carries",
        "successful_dribbles",
        "tackles",
        "interceptions",
        "pass_completion_pct",
    ]
    st.markdown("#### Metric percentiles")
    st.plotly_chart(create_metric_percentile_chart(player, percentile_metrics), width="stretch")


def profile_rankings_page(filtered_df: pd.DataFrame) -> None:
    st.subheader("Top Players By Profile")
    selected_profile = st.selectbox("Select profile", list(PLAYER_PROFILES.keys()))
    score_col = get_profile_score_column(selected_profile)
    eligible_positions = get_eligible_positions(selected_profile)
    ranking_df = filtered_df[filtered_df["position"].isin(eligible_positions)] if eligible_positions else filtered_df
    top_players = ranking_df.nlargest(20, score_col)

    st.plotly_chart(create_top_players_chart(ranking_df, selected_profile), width="stretch")
    display_cols = [
        "player_name",
        "country",
        "club",
        "league",
        "age",
        "position",
        score_col,
        "pre_tournament_expected_impact_score",
        "value_opportunity_score",
    ]
    st.dataframe(
        top_players[display_cols].rename(
            columns={
                "player_name": "Player",
                "country": "Country",
                "club": "Club",
                "league": "League",
                "age": "Age",
                "position": "Position",
                score_col: "Profile score",
                "pre_tournament_expected_impact_score": "Expected impact",
                "value_opportunity_score": "Value opportunity",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def expected_impact_page(df: pd.DataFrame, filtered_df: pd.DataFrame, model_summary: dict[str, object]) -> None:
    st.subheader("Expected Impact Model")
    if model_summary.get("model_available"):
        st.success("Historical training data found. Expected impact is using the supervised model output.")
        metrics = model_summary.get("metrics")
        if isinstance(metrics, pd.DataFrame) and not metrics.empty:
            st.markdown("#### Model evaluation")
            st.dataframe(metrics, width="stretch", hide_index=True)
    else:
        st.warning(FALLBACK_WARNING)

    player_pool = filtered_df if not filtered_df.empty else df
    selected_player = st.selectbox(
        "Select player for model explanation",
        player_pool.sort_values("value_opportunity_score", ascending=False)["player_name"].tolist(),
        key="expected_impact_player",
    )
    player = df[df["player_name"] == selected_player].iloc[0]
    explanation = explain_prediction(player)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Expected impact", f"{player['pre_tournament_expected_impact_score']:.1f}/100")
    metric_cols[1].metric("Baseline impact", f"{player['baseline_expected_impact_score']:.1f}/100")
    metric_cols[2].metric("Value opportunity", f"{player['value_opportunity_score']:.1f}/100")
    metric_cols[3].metric("Actual impact", "Pending" if pd.isna(player["actual_tournament_impact_score"]) else f"{player['actual_tournament_impact_score']:.1f}/100")
    metric_cols[4].metric("Delta", "Pending" if pd.isna(player["performance_delta"]) else f"{player['performance_delta']:.1f}")
    metric_cols[5].metric("Outcome", player["performance_outcome"])

    st.markdown("#### Baseline explanation factors")
    factors = pd.DataFrame(explanation["all_factors"]).rename(
        columns={"factor": "Factor", "score": "Score", "weight": "Weight", "contribution": "Weighted contribution"}
    )
    st.dataframe(factors, width="stretch", hide_index=True)

    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("#### Top positive factors")
        for factor in explanation["top_positive_factors"]:
            st.write(f"- {factor['factor']}: {factor['score']:.1f}/100")
    with neg_col:
        st.markdown("#### Top negative factors")
        negative = explanation["top_negative_factors"]
        if negative:
            for factor in negative:
                st.write(f"- {factor['factor']}: {factor['score']:.1f}/100")
        else:
            st.write("No major negative factors from the current model inputs.")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.markdown("#### Predicted vs actual impact")
        st.plotly_chart(create_predicted_vs_actual_impact_chart(filtered_df), width="stretch")
    with chart_right:
        st.markdown("#### Performance delta distribution")
        st.plotly_chart(create_performance_delta_distribution_chart(filtered_df), width="stretch")

    st.markdown("#### Similar players")
    similar = similar_players(df, player)
    st.dataframe(
        similar[
            [
                "player_name",
                "country",
                "club",
                "position",
                "best_profile",
                "pre_tournament_expected_impact_score",
                "value_opportunity_score",
            ]
        ].rename(
            columns={
                "player_name": "Player",
                "country": "Country",
                "club": "Club",
                "position": "Position",
                "best_profile": "Profile",
                "pre_tournament_expected_impact_score": "Expected impact",
                "value_opportunity_score": "Value opportunity",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    importance = model_summary.get("feature_importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        st.markdown("#### Feature importance")
        st.dataframe(importance.head(15), width="stretch", hide_index=True)


def methodology_page() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
**Main modelling objective**

The main prediction is designed to come from a supervised model trained on historical World Cup player data from 2014, 2018 and 2022. Each historical row should represent one player before a tournament, with pre-tournament features mapped to the actual tournament impact they produced.

**Historical training data**

The app checks `data/historical/world_cup_player_training_data.csv`. If that file contains historical rows and an `actual_tournament_impact_score` target, the dashboard trains Ridge Regression, Random Forest and Gradient Boosting regressors, evaluates MAE, RMSE and R-squared, and uses the selected model to produce `pre_tournament_expected_impact_score`.

**Baseline fallback**

The manual weighted score is only a transparent fallback named `baseline_expected_impact_score`:

`0.25 * current_performance_score + 0.20 * expected_minutes_score + 0.15 * role_fit_score + 0.15 * national_team_context_score + 0.10 * tournament_draw_score + 0.10 * availability_score + 0.05 * age_upside_score`

No true model is possible until historical training data is collected. When the historical file is unavailable, the dashboard shows a warning and uses the baseline score for expected impact.

**Actual tournament impact**

`actual_tournament_impact_score` is a position-adjusted 0-100 outcome score from World Cup match data. It uses tournament minutes, starts, goals, assists, shots, key passes, tackles, interceptions, saves, clean sheets and match ratings when available. Players are ranked within position groups where possible.

**Prediction validation**

`performance_delta = actual_tournament_impact_score - pre_tournament_expected_impact_score`. Pending players have no live match data yet. Once actual impact exists, players are classified as Overperformed, Met expectations or Underperformed.

**Value opportunity**

`value_opportunity_score` is a business ranking metric for recruitment strategy, not a pure prediction. It combines expected impact, value efficiency, age resale, role fit and availability. The strategy preset changes these business weights while leaving the model prediction untouched.
        """
    )


def main() -> None:
    model_summary = get_model_summary()
    strategy = select_strategy()
    df = calculate_value_opportunity_score(get_data(), strategy=strategy)
    filtered_df = apply_filters(df)

    st.title("WC 2026 Player Scouting & Performance Dashboard")
    st.caption("Pre-tournament recruitment analytics for expected impact, value opportunity and prediction validation.")

    if filtered_df.empty:
        st.warning("No players match the current filters. Widen the sidebar filters to continue.")
        filtered_df = df

    tabs = st.tabs(
        [
            "Overview",
            "Recruitment Shortlist",
            "Player Profile",
            "Profile Rankings",
            "Expected Impact Model",
            "Methodology",
        ]
    )
    with tabs[0]:
        overview_page(df, filtered_df, model_summary)
    with tabs[1]:
        shortlist_page(filtered_df)
    with tabs[2]:
        player_profile_page(df, filtered_df)
    with tabs[3]:
        profile_rankings_page(filtered_df)
    with tabs[4]:
        expected_impact_page(df, filtered_df, model_summary)
    with tabs[5]:
        methodology_page()


if __name__ == "__main__":
    main()
