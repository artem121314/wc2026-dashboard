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

CHART_FONT = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
CHART_PALETTE = ["#0f766e", "#2563eb", "#d97706", "#7c3aed", "#dc2626", "#0891b2", "#65a30d"]
BUSINESS_HOVER_FIELDS = [
    "player_name",
    "country",
    "club",
    "position",
    "market_value_label",
    "expected_impact_label",
    "value_opportunity_label",
    "recommendation",
    "risk_band",
]
BUSINESS_HOVER_TEMPLATE = (
    "<b>%{customdata[0]}</b><br>"
    "%{customdata[1]} · %{customdata[3]}<br>"
    "%{customdata[2]}<br>"
    "Market value: %{customdata[4]}<br>"
    "Expected WC Impact: %{customdata[5]}<br>"
    "Value Opportunity: %{customdata[6]}<br>"
    "Recommendation: %{customdata[7]}<br>"
    "Risk: %{customdata[8]}"
    "<extra></extra>"
)


def _label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _empty_message(text: str, height: int = 430) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, x=0.5, y=0.5)
    fig.update_layout(height=height, xaxis_visible=False, yaxis_visible=False)
    return fig


def _format_eur(value: object) -> str:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return "Unavailable"
    parsed = float(parsed)
    if parsed >= 1_000_000:
        return f"€{parsed / 1_000_000:.1f}M"
    if parsed >= 1_000:
        return f"€{parsed / 1_000:.0f}K"
    return f"€{parsed:.0f}"


def _score_label(value: object) -> str:
    parsed = pd.to_numeric(value, errors="coerce")
    return "Unavailable" if pd.isna(parsed) else f"{float(parsed):.1f}"


def _business_chart_df(df: pd.DataFrame) -> pd.DataFrame:
    chart_df = df.copy()
    chart_df["market_value_label"] = chart_df.get("market_value_eur", pd.Series(index=chart_df.index, dtype=float)).map(_format_eur)
    chart_df["expected_impact_label"] = chart_df.get(
        "pre_tournament_expected_impact_score", pd.Series(index=chart_df.index, dtype=float)
    ).map(_score_label)
    chart_df["value_opportunity_label"] = chart_df.get(
        "value_opportunity_score", pd.Series(index=chart_df.index, dtype=float)
    ).map(_score_label)
    for col in ["player_name", "country", "club", "position", "recommendation", "risk_band"]:
        if col not in chart_df.columns:
            chart_df[col] = "Unavailable"
    return chart_df


def _marker_size_column(chart_df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col not in chart_df.columns:
            continue
        values = pd.to_numeric(chart_df[col], errors="coerce")
        if values.dropna().empty:
            continue
        chart_df["_marker_size"] = values.fillna(1.0).clip(lower=1.0)
        return "_marker_size"
    return None


def _apply_business_layout(fig: go.Figure, height: int = 470) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=24, t=28, b=34),
        font=dict(family=CHART_FONT, size=12, color="#0f172a"),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        legend_title_text="",
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#0f172a", font=dict(color="#f8fafc", size=12)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, title_font=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, title_font=dict(size=12))
    return fig


def _apply_business_hover(fig: go.Figure) -> go.Figure:
    for trace in fig.data:
        trace.update(hovertemplate=BUSINESS_HOVER_TEMPLATE)
    return fig


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
    if score_col not in df.columns or df[score_col].dropna().empty:
        return _empty_message("Profile score data is not yet available.", height=620)
    eligible_positions = get_eligible_positions(profile)
    chart_df = df[df["position"].isin(eligible_positions)] if eligible_positions else df
    chart_df = _business_chart_df(chart_df.nlargest(20, score_col).sort_values(score_col))
    fig = px.bar(
        chart_df,
        y="player_name",
        x=score_col,
        color="position",
        orientation="h",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={score_col: "Profile score", "player_name": "Player"},
        text=score_col,
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 105], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=620))


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

    required = ["market_value_eur", "pre_tournament_expected_impact_score"]
    if any(col not in df.columns for col in required):
        return _empty_message("Market value or expected-impact data is unavailable.", height=470)
    chart_df = df[df["pre_tournament_expected_impact_score"].notna() & df["market_value_eur"].notna()].copy()
    if chart_df.empty:
        return _empty_message("Market value or expected-impact data is unavailable.", height=470)
    chart_df = _business_chart_df(chart_df)

    value_threshold = chart_df["market_value_eur"].median()
    impact_threshold = chart_df["pre_tournament_expected_impact_score"].median()
    size_col = _marker_size_column(chart_df, ["value_opportunity_score"])
    fig = px.scatter(
        chart_df,
        x="market_value_eur",
        y="pre_tournament_expected_impact_score",
        color="recommendation",
        size=size_col,
        hover_name="player_name",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={
            "market_value_eur": "Market value",
            "pre_tournament_expected_impact_score": "Expected WC impact",
        },
        size_max=26,
    )
    fig.add_vline(x=value_threshold, line_dash="dash", line_color="#64748b", line_width=1.2)
    fig.add_hline(y=impact_threshold, line_dash="dash", line_color="#64748b", line_width=1.2)
    annotations = [
        ("Undervalued<br>targets", value_threshold * 0.45, min(98, impact_threshold + 18)),
        ("Premium<br>targets", value_threshold * 1.45, min(98, impact_threshold + 18)),
        ("Low-cost<br>depth", value_threshold * 0.45, max(5, impact_threshold - 18)),
        ("Low-impact /<br>overpriced", value_threshold * 1.45, max(5, impact_threshold - 18)),
    ]
    for text, x, y in annotations:
        fig.add_annotation(
            text=text,
            x=x,
            y=y,
            showarrow=False,
            align="center",
            font=dict(size=12, color="#334155"),
            bgcolor="rgba(248,250,252,0.92)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            borderpad=6,
        )
    fig.update_xaxes(tickprefix="€", tickformat="~s")
    fig.update_yaxes(range=[0, 100], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=520))


def create_market_value_vs_value_opportunity_chart(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of market value against recruitment opportunity."""

    required = ["market_value_eur", "value_opportunity_score"]
    if any(col not in df.columns for col in required):
        return _empty_message("Market value or value-opportunity data is unavailable.", height=470)
    chart_df = df[df["market_value_eur"].notna() & df["value_opportunity_score"].notna()].copy()
    if chart_df.empty:
        return _empty_message("Market value or value-opportunity data is unavailable.", height=470)
    chart_df = _business_chart_df(chart_df)
    size_col = _marker_size_column(chart_df, ["pre_tournament_expected_impact_score"])
    fig = px.scatter(
        chart_df,
        x="market_value_eur",
        y="value_opportunity_score",
        color="recommendation",
        size=size_col,
        hover_name="player_name",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={"market_value_eur": "Market value (EUR)", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_xaxes(tickprefix="€", tickformat="~s")
    fig.update_yaxes(range=[0, 100], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=470))


def create_age_vs_value_opportunity_chart(df: pd.DataFrame) -> go.Figure:
    """Show age profile versus value opportunity."""

    required = ["age", "value_opportunity_score"]
    if any(col not in df.columns for col in required):
        return _empty_message("Age or value-opportunity data is unavailable.")
    chart_df = df[df["age"].notna() & df["value_opportunity_score"].notna()].copy()
    if chart_df.empty:
        return _empty_message("Age or value-opportunity data is unavailable.")
    chart_df = _business_chart_df(chart_df)
    size_col = _marker_size_column(chart_df, ["pre_tournament_expected_impact_score"])
    fig = px.scatter(
        chart_df,
        x="age",
        y="value_opportunity_score",
        color="position",
        size=size_col,
        hover_name="player_name",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={"age": "Age", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_yaxes(range=[0, 100], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=430))


def create_breakout_candidates_chart(df: pd.DataFrame) -> go.Figure:
    """Show young breakout candidates by age and expected World Cup impact."""

    required = ["age", "pre_tournament_expected_impact_score"]
    if any(col not in df.columns for col in required):
        return _empty_message("Age or expected-impact data is unavailable.")
    chart_df = df[df["age"].notna() & df["pre_tournament_expected_impact_score"].notna()].copy()
    if chart_df.empty:
        return _empty_message("Age or expected-impact data is unavailable.")
    chart_df = _business_chart_df(chart_df)

    color_col = (
        "value_opportunity_score"
        if "value_opportunity_score" in chart_df.columns and chart_df["value_opportunity_score"].notna().any()
        else "recommendation"
    )
    size_col = _marker_size_column(chart_df, ["market_value_eur", "recent_senior_minutes"])

    fig = px.scatter(
        chart_df,
        x="age",
        y="pre_tournament_expected_impact_score",
        color=color_col,
        size=size_col,
        hover_name="player_name",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_continuous_scale=["#dbeafe", "#2563eb", "#0f766e"] if color_col == "value_opportunity_score" else None,
        color_discrete_sequence=CHART_PALETTE if color_col != "value_opportunity_score" else None,
        labels={
            "age": "Age",
            "pre_tournament_expected_impact_score": "Expected WC impact",
            "value_opportunity_score": "Value opportunity score",
        },
        size_max=28,
    )
    fig.update_yaxes(range=[0, 100], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=480))


def create_predicted_vs_actual_impact_chart(df: pd.DataFrame) -> go.Figure:
    """Compare pre-tournament expected impact with actual tournament impact."""

    chart_df = df[df["actual_tournament_impact_score"].notna()].copy()
    if chart_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Actual tournament data is not yet available. Add real 2026 match rows and refresh.",
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
            text="Performance deltas are not yet available until real tournament match data is added.",
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

    if "value_opportunity_score" not in df.columns or df["value_opportunity_score"].dropna().empty:
        return _empty_message("Value-opportunity data is unavailable.", height=620)
    chart_df = _business_chart_df(df.nlargest(20, "value_opportunity_score").sort_values("value_opportunity_score"))
    fig = px.bar(
        chart_df,
        y="player_name",
        x="value_opportunity_score",
        color="recommendation",
        orientation="h",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={"player_name": "Player", "value_opportunity_score": "Value opportunity score"},
        text="value_opportunity_score",
    )
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 105], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=620))


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

    required = ["value_efficiency_score", "pre_tournament_expected_impact_score"]
    if any(col not in df.columns for col in required):
        return _empty_message("Expected impact or value-efficiency data is unavailable.")
    chart_df = df[df["value_efficiency_score"].notna() & df["pre_tournament_expected_impact_score"].notna()].copy()
    if chart_df.empty:
        return _empty_message("Expected impact or value-efficiency data is unavailable.")
    chart_df = _business_chart_df(chart_df)
    size_col = _marker_size_column(chart_df, ["value_opportunity_score"])
    fig = px.scatter(
        chart_df,
        x="value_efficiency_score",
        y="pre_tournament_expected_impact_score",
        color="risk_band",
        size=size_col,
        hover_name="player_name",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={
            "value_efficiency_score": "Value efficiency",
            "pre_tournament_expected_impact_score": "Expected WC impact",
        },
    )
    fig.update_xaxes(range=[0, 100], tickformat=".0f")
    fig.update_yaxes(range=[0, 100], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=430))


def create_recommendation_breakdown_chart(df: pd.DataFrame) -> go.Figure:
    """Bar chart of recommendation categories."""

    counts = df["recommendation"].fillna("Unknown").value_counts().reset_index()
    counts.columns = ["Recommendation", "Players"]
    fig = px.bar(
        counts,
        x="Recommendation",
        y="Players",
        color="Recommendation",
        text="Players",
        color_discrete_sequence=CHART_PALETTE,
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    return _apply_business_layout(fig, height=360)


def create_risk_vs_opportunity_matrix(df: pd.DataFrame) -> go.Figure:
    """Show risk score against value opportunity."""

    required = ["risk_score", "value_opportunity_score"]
    if any(col not in df.columns for col in required):
        return _empty_message("Risk or value-opportunity data is unavailable.")
    chart_df = df[df["risk_score"].notna() & df["value_opportunity_score"].notna()].copy()
    if chart_df.empty:
        return _empty_message("Risk or value-opportunity data is unavailable.")
    chart_df = _business_chart_df(chart_df)
    size_col = _marker_size_column(chart_df, ["pre_tournament_expected_impact_score"])
    fig = px.scatter(
        chart_df,
        x="risk_score",
        y="value_opportunity_score",
        color="recommendation",
        size=size_col,
        hover_name="player_name",
        custom_data=BUSINESS_HOVER_FIELDS,
        color_discrete_sequence=CHART_PALETTE,
        labels={"risk_score": "Risk score", "value_opportunity_score": "Value opportunity score"},
    )
    fig.update_xaxes(range=[0, 100], tickformat=".0f")
    fig.update_yaxes(range=[0, 100], tickformat=".0f")
    return _apply_business_hover(_apply_business_layout(fig, height=430))


def create_market_value_vs_score_chart(df: pd.DataFrame) -> go.Figure:
    """Backward-compatible alias for market value versus expected impact."""

    return create_market_value_vs_expected_impact_chart(df)
