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
    "pre_tournament_expected_impact_score": "Expected impact",
    "baseline_expected_impact_score": "Baseline expected impact",
    "value_opportunity_score": "Value opportunity",
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
        hover_data=["country", "club", "league", "age", "market_value_eur", "pre_tournament_expected_impact_score"],
        labels={score_col: "Profile score", "player_name": "Player"},
    )
    fig.update_layout(height=620, margin=dict(l=10, r=20, t=20, b=20), legend_title_text="Position")
    return fig


def create_market_value_vs_expected_impact_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of market value against expected World Cup impact."""

    fig = px.scatter(
        df,
        x="market_value_eur",
        y="pre_tournament_expected_impact_score",
        color="position",
        size="minutes",
        hover_name="player_name",
        hover_data=["country", "club", "best_profile", "value_opportunity_score"],
        labels={
            "market_value_eur": "Market value (EUR)",
            "pre_tournament_expected_impact_score": "Pre-tournament expected impact",
        },
    )
    fig.update_layout(height=470, margin=dict(l=10, r=20, t=20, b=20), xaxis_tickprefix="EUR ")
    return fig


def create_price_to_impact_quadrant_chart(df: pd.DataFrame) -> go.Figure:
    """Price-to-impact recruitment quadrant."""

    chart_df = df[df["pre_tournament_expected_impact_score"].notna()].copy()
    if chart_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Expected-impact data is unavailable.", showarrow=False, x=0.5, y=0.5)
        fig.update_layout(height=470, xaxis_visible=False, yaxis_visible=False)
        return fig

    value_threshold = chart_df["market_value_eur"].median()
    impact_threshold = chart_df["pre_tournament_expected_impact_score"].median()
    fig = px.scatter(
        chart_df,
        x="market_value_eur",
        y="pre_tournament_expected_impact_score",
        color="recommendation",
        size="value_opportunity_score",
        hover_name="player_name",
        hover_data=["country", "club", "position", "age", "value_efficiency_score"],
        labels={
            "market_value_eur": "Market value (EUR)",
            "pre_tournament_expected_impact_score": "Expected WC impact",
        },
    )
    fig.add_vline(x=value_threshold, line_dash="dash", line_color="#94a3b8")
    fig.add_hline(y=impact_threshold, line_dash="dash", line_color="#94a3b8")
    annotations = [
        ("Undervalued targets", value_threshold * 0.45, min(100, impact_threshold + 15)),
        ("Premium targets", value_threshold * 1.45, min(100, impact_threshold + 15)),
        ("Low-cost depth", value_threshold * 0.45, max(0, impact_threshold - 15)),
        ("Overpriced / low impact", value_threshold * 1.45, max(0, impact_threshold - 15)),
    ]
    for text, x, y in annotations:
        fig.add_annotation(text=text, x=x, y=y, showarrow=False, font=dict(size=12, color="#334155"))
    fig.update_layout(height=500, margin=dict(l=10, r=20, t=20, b=20), xaxis_tickprefix="EUR ")
    return fig


def create_market_value_vs_value_opportunity_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of market value against recruitment opportunity."""

    fig = px.scatter(
        df,
        x="market_value_eur",
        y="value_opportunity_score",
        color="recommendation",
        size="pre_tournament_expected_impact_score",
        hover_name="player_name",
        hover_data=["country", "club", "position", "age", "market_value_source"],
        labels={"market_value_eur": "Market value (EUR)", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_layout(height=470, margin=dict(l=10, r=20, t=20, b=20), xaxis_tickprefix="EUR ")
    return fig


def create_age_vs_value_opportunity_chart(df: pd.DataFrame) -> go.Figure:
    """Show age profile versus value opportunity."""

    fig = px.scatter(
        df,
        x="age",
        y="value_opportunity_score",
        color="position",
        size="pre_tournament_expected_impact_score",
        hover_name="player_name",
        hover_data=["country", "club", "market_value_eur", "recommendation"],
        labels={"age": "Age", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=20, t=20, b=20))
    return fig


def create_predicted_vs_actual_impact_chart(df: pd.DataFrame) -> go.Figure:
    """Compare pre-tournament expected impact with actual tournament impact."""

    chart_df = df[df["actual_tournament_impact_score"].notna()].copy()
    if chart_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Actual tournament data is pending. Add rows to data/raw/tournament_match_data.csv and refresh.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=430, xaxis_visible=False, yaxis_visible=False)
        return fig
    fig = px.scatter(
        chart_df,
        x="pre_tournament_expected_impact_score",
        y="actual_tournament_impact_score",
        color="performance_outcome",
        hover_name="player_name",
        hover_data=[col for col in ["country", "club", "position", "performance_delta"] if col in chart_df.columns],
        labels={
            "pre_tournament_expected_impact_score": "Pre-tournament expected impact",
            "actual_tournament_impact_score": "Actual tournament impact",
        },
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Expectation line",
            line=dict(color="#94a3b8", dash="dash"),
        )
    )
    fig.update_layout(height=430, margin=dict(l=10, r=20, t=20, b=20))
    return fig


def create_performance_delta_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of actual minus expected impact."""

    chart_df = df[df["performance_delta"].notna()]
    if chart_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Performance deltas are pending until real tournament match data is added.",
            showarrow=False,
            x=0.5,
            y=0.5,
        )
        fig.update_layout(height=390, xaxis_visible=False, yaxis_visible=False)
        return fig
    fig = px.histogram(
        chart_df,
        x="performance_delta",
        color="performance_outcome",
        nbins=24,
        labels={"performance_delta": "Actual impact minus expected impact"},
    )
    fig.update_layout(height=390, margin=dict(l=10, r=20, t=20, b=20), barmode="overlay")
    return fig


def create_top_value_opportunities_chart(df: pd.DataFrame) -> go.Figure:
    """Rank the top value opportunities."""

    chart_df = df.nlargest(20, "value_opportunity_score").sort_values("value_opportunity_score")
    fig = px.bar(
        chart_df,
        y="player_name",
        x="value_opportunity_score",
        color="position",
        orientation="h",
        hover_data=["country", "club", "market_value_eur", "pre_tournament_expected_impact_score", "recommendation"],
        labels={"player_name": "Player", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_layout(height=620, margin=dict(l=10, r=20, t=20, b=20), legend_title_text="Position")
    return fig


def create_expected_impact_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Histogram of expected impact by position."""

    fig = px.histogram(
        df,
        x="pre_tournament_expected_impact_score",
        color="position",
        nbins=24,
        opacity=0.78,
        labels={"pre_tournament_expected_impact_score": "Pre-tournament expected impact"},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=20, t=20, b=20), barmode="overlay")
    return fig


def create_expected_impact_vs_value_efficiency_chart(df: pd.DataFrame) -> go.Figure:
    """Compare expected impact with market-value efficiency."""

    fig = px.scatter(
        df,
        x="value_efficiency_score",
        y="pre_tournament_expected_impact_score",
        color="risk_band",
        size="value_opportunity_score",
        hover_name="player_name",
        hover_data=["country", "club", "position", "market_value_eur", "recommendation"],
        labels={
            "value_efficiency_score": "Value efficiency",
            "pre_tournament_expected_impact_score": "Expected WC impact",
        },
    )
    fig.update_layout(height=430, margin=dict(l=10, r=20, t=20, b=20))
    return fig


def create_recommendation_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of recommendation categories."""

    counts = df["recommendation"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["Recommendation", "Players"]
    fig = px.bar(counts, x="Recommendation", y="Players", color="Recommendation", text="Players")
    fig.update_layout(height=360, margin=dict(l=10, r=20, t=20, b=20), showlegend=False)
    return fig


def create_risk_vs_opportunity_matrix(df: pd.DataFrame) -> go.Figure:
    """Show risk score against value opportunity."""

    fig = px.scatter(
        df,
        x="risk_score",
        y="value_opportunity_score",
        color="recommendation",
        size="pre_tournament_expected_impact_score",
        hover_name="player_name",
        hover_data=["country", "club", "age", "market_value_eur", "risk_reason"],
        labels={"risk_score": "Risk score", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_layout(height=430, margin=dict(l=10, r=20, t=20, b=20))
    return fig


def create_market_value_vs_score_chart(df: pd.DataFrame) -> go.Figure:
    """Backward-compatible alias for market value versus expected impact."""

    return create_market_value_vs_expected_impact_chart(df)
