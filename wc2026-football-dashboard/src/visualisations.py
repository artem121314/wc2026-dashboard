"""Plotly visualisations for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from player_profiles import get_eligible_positions, get_profile_score_column


METRIC_LABELS = {
    "attacking_score": "Attacking",
    "creativity_score": "Creativity",
    "progression_score": "Progression",
    "defensive_score": "Defensive",
    "possession_score": "Possession",
    "overall_score": "Overall",
    "success_probability": "Success probability",
    "xg_proxy": "xG proxy",
    "xa_proxy": "xA proxy",
}


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def create_radar_chart(player: pd.Series, df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Create a radar chart for a player's key score metrics."""

    labels = [_label(metric) for metric in metrics]
    values = [float(player.get(metric, 0)) for metric in metrics]
    positional_average = [
        float(df.loc[df["position"] == player["position"], metric].mean()) if metric in df.columns else 0
        for metric in metrics
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + values[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=str(player["player_name"]),
            line_color="#0f766e",
            fillcolor="rgba(15,118,110,0.20)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=positional_average + positional_average[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=f"{player['position']} average",
            line_color="#64748b",
            fillcolor="rgba(100,116,139,0.12)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10))),
        showlegend=True,
        margin=dict(l=30, r=30, t=30, b=30),
        height=420,
    )
    return fig


def create_percentile_bar_chart(player: pd.Series, df: pd.DataFrame, metrics: list[str]) -> go.Figure:
    """Compare a player with their positional average."""

    rows = []
    position_df = df[df["position"] == player["position"]]
    for metric in metrics:
        rows.append({"Metric": _label(metric), "Value": float(player.get(metric, 0)), "Type": player["player_name"]})
        rows.append({"Metric": _label(metric), "Value": float(position_df[metric].mean()), "Type": "Positional average"})

    chart_df = pd.DataFrame(rows)
    fig = px.bar(
        chart_df,
        y="Metric",
        x="Value",
        color="Type",
        barmode="group",
        orientation="h",
        color_discrete_sequence=["#0f766e", "#94a3b8"],
        text_auto=".1f",
    )
    fig.update_layout(
        xaxis_range=[0, 100],
        height=360,
        margin=dict(l=10, r=20, t=20, b=20),
        legend_title_text="",
    )
    return fig


def create_metric_percentile_chart(player: pd.Series, metrics: list[str]) -> go.Figure:
    """Show percentile standing in specific football actions."""

    rows = []
    for metric in metrics:
        percentile_col = f"{metric}_percentile"
        rows.append({"Metric": _label(metric), "Percentile": float(player.get(percentile_col, 0))})
    chart_df = pd.DataFrame(rows).sort_values("Percentile")
    fig = px.bar(
        chart_df,
        x="Percentile",
        y="Metric",
        orientation="h",
        range_x=[0, 100],
        color="Percentile",
        color_continuous_scale=["#ef4444", "#f59e0b", "#0f766e"],
        text="Percentile",
    )
    fig.update_traces(texttemplate="%{text:.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=360, margin=dict(l=10, r=35, t=20, b=20), coloraxis_showscale=False)
    return fig


def create_top_players_chart(df: pd.DataFrame, profile: str) -> go.Figure:
    """Create top player chart for a predefined profile."""

    score_col = get_profile_score_column(profile)
    eligible_positions = get_eligible_positions(profile)
    chart_df = df[df["position"].isin(eligible_positions)] if eligible_positions else df
    chart_df = chart_df.nlargest(20, score_col).sort_values(score_col)
    fig = px.bar(
        chart_df,
        y="player_name",
        x=score_col,
        color="position",
        orientation="h",
        hover_data=["country", "club", "league", "age", "market_value_eur", "success_probability"],
        labels={score_col: "Profile score", "player_name": "Player"},
    )
    fig.update_layout(height=620, margin=dict(l=10, r=20, t=20, b=20), legend_title_text="Position")
    return fig


def create_market_value_vs_score_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of market value against overall profile score."""

    fig = px.scatter(
        df,
        x="market_value_eur",
        y="overall_score",
        color="position",
        size="minutes",
        hover_name="player_name",
        hover_data=["country", "club", "best_profile", "success_probability"],
        labels={"market_value_eur": "Market value (EUR)", "overall_score": "Overall profile score"},
    )
    fig.update_layout(height=470, margin=dict(l=10, r=20, t=20, b=20), xaxis_tickprefix="EUR ")
    return fig


def create_success_probability_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of success probability by position."""

    fig = px.histogram(
        df,
        x="success_probability",
        color="position",
        nbins=24,
        opacity=0.78,
        labels={"success_probability": "Predicted success probability"},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=20, t=20, b=20), barmode="overlay")
    return fig
