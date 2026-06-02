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
from model import confidence_level, explain_prediction, train_success_model
from player_profiles import PLAYER_PROFILES, get_eligible_positions, get_profile_score_column
from utils import format_currency, get_risks, get_strengths, scouting_summary, similar_players
from visualisations import (
    create_market_value_vs_score_chart,
    create_metric_percentile_chart,
    create_percentile_bar_chart,
    create_radar_chart,
    create_success_probability_chart,
    create_top_players_chart,
)


st.set_page_config(
    page_title="WC 2026 Player Scouting Dashboard",
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
def get_model_summary(df: pd.DataFrame) -> dict[str, object]:
    return train_success_model(df, model_type="logistic_regression")


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
    minutes_range = st.sidebar.slider(
        "Minutes played range",
        int(df["minutes"].min()),
        int(df["minutes"].max()),
        (int(df["minutes"].min()), int(df["minutes"].max())),
        step=50,
    )
    probability_range = st.sidebar.slider(
        "Predicted World Cup success probability",
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

    filtered = filtered[
        filtered["age"].between(*age_range)
        & filtered["market_value_eur"].between(value_range[0] * 1_000_000, value_range[1] * 1_000_000)
        & filtered["minutes"].between(*minutes_range)
        & filtered["success_probability"].between(*probability_range)
    ]
    return filtered


def display_player_table(df: pd.DataFrame) -> None:
    table = df[
        [
            "player_name",
            "country",
            "club",
            "league",
            "age",
            "position",
            "best_profile",
            "market_value_eur",
            "minutes",
            "goals",
            "assists",
            "xg_proxy",
            "xa_proxy",
            "progression_score",
            "creativity_score",
            "defensive_score",
            "overall_score",
            "success_probability",
        ]
    ].rename(
        columns={
            "player_name": "Player",
            "country": "Country",
            "club": "Club",
            "league": "League",
            "age": "Age",
            "position": "Position",
            "best_profile": "Profile",
            "market_value_eur": "Market value",
            "minutes": "Minutes",
            "goals": "Goals",
            "assists": "Assists",
            "xg_proxy": "xG proxy",
            "xa_proxy": "xA proxy",
            "progression_score": "Progression score",
            "creativity_score": "Creativity score",
            "defensive_score": "Defensive score",
            "overall_score": "Overall profile score",
            "success_probability": "Predicted success probability",
        }
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Market value": st.column_config.NumberColumn(format="EUR %.0f"),
            "xG proxy": st.column_config.NumberColumn(format="%.2f"),
            "xA proxy": st.column_config.NumberColumn(format="%.2f"),
            "Predicted success probability": st.column_config.ProgressColumn(
                min_value=0,
                max_value=100,
                format="%.1f%%",
            ),
        },
    )


def overview_page(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("WC 2026 Player Scouting & Performance Dashboard")
    st.write(
        "This dashboard is a football analytics tool built to support player scouting and performance analysis "
        "ahead of the 2026 FIFA World Cup. It allows users to explore player profiles, compare players, filter "
        "by market value and position, and estimate whether a player is likely to have a strong World Cup performance."
    )

    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Players", f"{len(df):,}")
    kpi_cols[1].metric("Countries", f"{df['country'].nunique():,}")
    kpi_cols[2].metric("Avg market value", format_currency(df["market_value_eur"].mean()))
    kpi_cols[3].metric("Avg age", f"{df['age'].mean():.1f}")
    kpi_cols[4].metric("Profiles", f"{len(PLAYER_PROFILES)}")
    kpi_cols[5].metric("Filtered players", f"{len(filtered_df):,}")

    st.divider()
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("#### Market value vs performance")
        st.plotly_chart(create_market_value_vs_score_chart(filtered_df), width="stretch")
    with right:
        st.markdown("#### Top predicted performers")
        top_cols = ["player_name", "country", "position", "club", "best_profile", "success_probability"]
        st.dataframe(
            df.nlargest(10, "success_probability")[top_cols].rename(
                columns={
                    "player_name": "Player",
                    "country": "Country",
                    "position": "Position",
                    "club": "Club",
                    "best_profile": "Profile",
                    "success_probability": "Success probability",
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Success probability distribution")
    st.plotly_chart(create_success_probability_chart(filtered_df), width="stretch")


def player_search_page(filtered_df: pd.DataFrame) -> None:
    st.subheader("Player Search")
    st.caption("Use the sidebar filters to narrow the player pool by role, market, playing time, and prediction bands.")
    sort_col = st.selectbox(
        "Sort players by",
        ["success_probability", "overall_score", "best_profile_score", "market_value_eur", "minutes", "age"],
        format_func=lambda value: value.replace("_", " ").title(),
    )
    sorted_df = filtered_df.sort_values(sort_col, ascending=False)
    display_player_table(sorted_df)


def player_profile_page(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("Player Profile")
    player_pool = filtered_df if not filtered_df.empty else df
    selected_player = st.selectbox(
        "Select player",
        player_pool["player_name"].sort_values().tolist(),
    )
    player = df[df["player_name"] == selected_player].iloc[0]

    top = st.columns([1.25, 1, 1, 1])
    top[0].metric("Player", player["player_name"])
    top[1].metric("Market value", format_currency(player["market_value_eur"]))
    top[2].metric("Age", f"{int(player['age'])}")
    top[3].metric("Success probability", f"{player['success_probability']:.1f}%")

    info_cols = st.columns(4)
    info_cols[0].metric("Country", player["country"])
    info_cols[1].metric("Club", player["club"])
    info_cols[2].metric("Position", player["position"])
    info_cols[3].metric("Profile", player["best_profile"])

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

    st.markdown("#### Metric percentiles")
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
        "success_probability",
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
                "success_probability": "Success probability",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def prediction_page(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("World Cup Success Prediction")
    st.info(
        "This is a scouting-support model, not a betting model. The first version uses a transparent heuristic label "
        "because real 2026 World Cup outcomes do not exist yet."
    )

    model_summary = get_model_summary(df)
    auc = model_summary.get("auc")
    if auc:
        st.caption(f"Logistic regression trained against the heuristic success label. Validation ROC AUC: {auc:.2f}.")

    player_pool = filtered_df if not filtered_df.empty else df
    selected_player = st.selectbox(
        "Select player for prediction explanation",
        player_pool.sort_values("success_probability", ascending=False)["player_name"].tolist(),
        key="prediction_player",
    )
    player = df[df["player_name"] == selected_player].iloc[0]
    explanation = explain_prediction(player)

    metric_cols = st.columns(4)
    metric_cols[0].metric("Predicted success probability", f"{player['success_probability']:.1f}%")
    metric_cols[1].metric("Success score", f"{player['success_score']:.1f}/100")
    metric_cols[2].metric("Confidence", confidence_level(player))
    metric_cols[3].metric("Profile", player["best_profile"])

    factors = pd.DataFrame(explanation["all_factors"]).rename(
        columns={"factor": "Factor", "score": "Score", "weight": "Weight", "contribution": "Weighted contribution"}
    )
    st.markdown("#### Explanation factors")
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
            st.write("No major negative factors from the heuristic inputs.")

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
                "overall_score",
                "success_probability",
            ]
        ].rename(
            columns={
                "player_name": "Player",
                "country": "Country",
                "club": "Club",
                "position": "Position",
                "best_profile": "Profile",
                "overall_score": "Overall score",
                "success_probability": "Success probability",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    importance = model_summary.get("feature_importance")
    if isinstance(importance, pd.DataFrame) and not importance.empty:
        st.markdown("#### Model feature importance")
        st.dataframe(importance.head(10), width="stretch", hide_index=True)


def methodology_page() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
**Data sources**

The MVP uses a reproducible synthetic sample of 260 player seasons. It is designed so real data can later replace `data/sample/wc2026_sample_players.csv` without changing the dashboard logic.

**Metric definitions**

Volume actions are converted to per-90 rates before percentile ranking. Percentiles are calculated within position groups, so defenders, midfielders, forwards, and goalkeepers are judged against relevant peers.

**Profiles**

Each football profile uses a weighted score across relevant percentile metrics. Negative turnover weights are interpreted as lower-turnover players scoring better.

**Success probability**

The first version uses this transparent success score:

`0.25 * current_performance_score + 0.20 * national_team_strength_score + 0.15 * expected_minutes_score + 0.15 * role_fit_score + 0.10 * age_curve_score + 0.10 * club_level_score + 0.05 * recent_form_score`

The score is mapped to a 0-100 probability-like output for scouting triage. It is not a betting model.

**Limitations**

The current data is synthetic, the target label is heuristic, and the model does not include injuries, tactical usage, squad selection uncertainty, opponent strength, or real tournament outcomes.

**Future improvements**

Replace the sample data with real event and scouting data, add historical World Cup training labels, add uncertainty intervals, include squad selection logic, and validate profile weights with domain experts.
        """
    )


def main() -> None:
    df = get_data()
    filtered_df = apply_filters(df)

    st.title("WC 2026 Player Scouting & Performance Dashboard")
    st.caption("A portfolio-ready football analytics MVP for scouting, role profiling, and World Cup performance triage.")

    if filtered_df.empty:
        st.warning("No players match the current filters. Widen the sidebar filters to continue.")
        filtered_df = df

    tabs = st.tabs(
        [
            "Overview",
            "Player Search",
            "Player Profile",
            "Profile Rankings",
            "Success Prediction",
            "Methodology",
        ]
    )
    with tabs[0]:
        overview_page(df, filtered_df)
    with tabs[1]:
        player_search_page(filtered_df)
    with tabs[2]:
        player_profile_page(df, filtered_df)
    with tabs[3]:
        profile_rankings_page(filtered_df)
    with tabs[4]:
        prediction_page(df, filtered_df)
    with tabs[5]:
        methodology_page()


if __name__ == "__main__":
    main()
