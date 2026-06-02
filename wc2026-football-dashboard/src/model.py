"""Model-based expected World Cup impact utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_sources import HISTORICAL_TRAINING_PATH


FALLBACK_WARNING = (
    "Historical World Cup training data is not available. The supervised expected-impact model is disabled until "
    "real curated historical data is added."
)
MODEL_STATUS_AVAILABLE = "available"
MODEL_STATUS_DISABLED = "disabled_missing_real_historical_data"

EXPECTED_IMPACT_FEATURES = [
    "age",
    "position",
    "market_value_before_tournament",
    "club_level_score",
    "league_strength_score",
    "club_minutes_previous_season",
    "goals_previous_season",
    "assists_previous_season",
    "national_team_caps",
    "expected_starter_score",
    "national_team_strength",
    "group_difficulty_score",
    "injury_availability_score",
    "recent_form_score",
    "role_fit_score",
]

REQUIRED_HISTORICAL_COLUMNS = [
    "tournament_year",
    "player_name",
    *EXPECTED_IMPACT_FEATURES,
    "actual_tournament_impact_score",
]

NUMERIC_FEATURES = [feature for feature in EXPECTED_IMPACT_FEATURES if feature != "position"]
CATEGORICAL_FEATURES = ["position"]

BASELINE_COMPONENTS = {
    "Current performance": ("current_performance_score", 0.25),
    "Expected minutes": ("expected_minutes_score", 0.20),
    "Role fit": ("role_fit_score", 0.15),
    "National team context": ("national_team_context_score", 0.15),
    "Tournament draw": ("tournament_draw_score", 0.10),
    "Availability": ("availability_score", 0.10),
    "Age upside": ("age_upside_score", 0.05),
}


def load_historical_training_data(path: Path | str = HISTORICAL_TRAINING_PATH) -> pd.DataFrame:
    """Load historical World Cup player training rows if the file is available."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if data.empty:
        return pd.DataFrame()
    return data


def _coalesce_feature(df: pd.DataFrame, target: str, aliases: list[str], default: Any) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index)
    for alias in [target, *aliases]:
        if alias in df.columns:
            candidate = df[alias]
            if target == "position":
                values = candidate.combine_first(values)
            else:
                values = pd.to_numeric(values, errors="coerce").combine_first(pd.to_numeric(candidate, errors="coerce"))
    return values.fillna(default)


def prepare_expected_impact_features(df: pd.DataFrame) -> pd.DataFrame:
    """Map current or historical data into the model feature schema."""

    mapped = pd.DataFrame(index=df.index)
    mapped["age"] = _coalesce_feature(df, "age", [], 26)
    mapped["position"] = _coalesce_feature(df, "position", [], "Unknown").fillna("Unknown").astype(str)
    mapped["market_value_before_tournament"] = _coalesce_feature(
        df,
        "market_value_before_tournament",
        ["market_value_eur"],
        0,
    )
    mapped["club_level_score"] = _coalesce_feature(df, "club_level_score", [], 55)
    mapped["league_strength_score"] = _coalesce_feature(
        df,
        "league_strength_score",
        ["club_level_score"],
        55,
    )
    mapped["club_minutes_previous_season"] = _coalesce_feature(
        df,
        "club_minutes_previous_season",
        ["minutes"],
        0,
    )
    mapped["goals_previous_season"] = _coalesce_feature(df, "goals_previous_season", ["goals"], 0)
    mapped["assists_previous_season"] = _coalesce_feature(df, "assists_previous_season", ["assists"], 0)
    mapped["national_team_caps"] = _coalesce_feature(df, "national_team_caps", ["caps"], 0)
    mapped["expected_starter_score"] = _coalesce_feature(
        df,
        "expected_starter_score",
        ["expected_minutes_score"],
        50,
    )
    mapped["national_team_strength"] = _coalesce_feature(df, "national_team_strength", [], 55)
    mapped["group_difficulty_score"] = _coalesce_feature(
        df,
        "group_difficulty_score",
        ["tournament_draw_score", "draw_context_score"],
        55,
    )
    mapped["injury_availability_score"] = _coalesce_feature(
        df,
        "injury_availability_score",
        ["availability_score"],
        100,
    )
    mapped["recent_form_score"] = _coalesce_feature(df, "recent_form_score", [], 50)
    mapped["role_fit_score"] = _coalesce_feature(df, "role_fit_score", ["best_profile_score"], 50)

    mapped["market_value_before_tournament"] = np.log1p(
        pd.to_numeric(mapped["market_value_before_tournament"], errors="coerce").fillna(0).clip(lower=0)
    )
    for feature in NUMERIC_FEATURES:
        mapped[feature] = pd.to_numeric(mapped[feature], errors="coerce")
    return mapped[EXPECTED_IMPACT_FEATURES]


def _build_model(model_type: str, random_state: int) -> Pipeline:
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "ridge":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )
    if model_type == "ridge":
        regressor = Ridge(alpha=1.0)
    elif model_type == "random_forest":
        regressor = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=random_state)
    else:
        regressor = GradientBoostingRegressor(random_state=random_state)
    return Pipeline([("features", preprocessor), ("regressor", regressor)])


def evaluate_expected_impact_model(y_true: pd.Series, y_pred: np.ndarray | pd.Series) -> dict[str, float]:
    """Evaluate expected-impact predictions with regression metrics."""

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _empty_training_result(model_type: str = "gradient_boosting") -> dict[str, object]:
    return {
        "model": None,
        "selected_model_name": None,
        "model_type": model_type,
        "model_available": False,
        "model_status": MODEL_STATUS_DISABLED,
        "target_source": "disabled",
        "warning": FALLBACK_WARNING,
        "training_row_count": 0,
        "data_source_path": str(HISTORICAL_TRAINING_PATH),
        "metrics": pd.DataFrame(columns=["model", "mae", "rmse", "r2"]),
        "feature_importance": pd.DataFrame(columns=["feature", "importance"]),
    }


def train_expected_impact_model(
    historical_data: pd.DataFrame | None = None,
    model_type: str = "gradient_boosting",
    random_state: int = 26,
) -> dict[str, object]:
    """Train supervised expected-impact regressors on historical World Cup data."""

    training_data = load_historical_training_data() if historical_data is None else historical_data.copy()
    if training_data.empty:
        return _empty_training_result(model_type)

    missing_columns = [col for col in REQUIRED_HISTORICAL_COLUMNS if col not in training_data.columns]
    if missing_columns:
        result = _empty_training_result(model_type)
        result["model_status"] = "disabled_invalid_real_historical_data"
        result["warning"] = (
            "Historical World Cup training data is present but missing required columns: "
            + ", ".join(missing_columns)
        )
        result["training_row_count"] = int(len(training_data))
        return result

    y = pd.to_numeric(training_data["actual_tournament_impact_score"], errors="coerce")
    valid = y.notna()
    training_data = training_data.loc[valid].copy()
    y = y.loc[valid].clip(0, 100)
    if len(training_data) < 30 or y.nunique() < 2:
        result = _empty_training_result(model_type)
        result["model_status"] = "disabled_insufficient_real_historical_rows"
        result["warning"] = (
            "Historical World Cup training data is present but does not contain enough valid target rows to train "
            "a supervised model."
        )
        result["training_row_count"] = int(len(training_data))
        return result

    X = prepare_expected_impact_features(training_data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
    )

    model_names = ["ridge", "random_forest", "gradient_boosting"]
    trained_models: dict[str, Pipeline] = {}
    metrics = []
    for name in model_names:
        candidate = _build_model(name, random_state)
        candidate.fit(X_train, y_train)
        preds = np.clip(candidate.predict(X_test), 0, 100)
        trained_models[name] = candidate
        metrics.append({"model": name, **evaluate_expected_impact_model(y_test, preds)})

    metrics_df = pd.DataFrame(metrics).sort_values("mae", ascending=True).reset_index(drop=True)
    selected_name = str(metrics_df.iloc[0]["model"])
    selected_model = trained_models[selected_name]

    return {
        "model": selected_model,
        "all_models": trained_models,
        "selected_model_name": selected_name,
        "model_type": selected_name,
        "model_available": True,
        "model_status": MODEL_STATUS_AVAILABLE,
        "target_source": "historical World Cup actual_tournament_impact_score",
        "warning": "",
        "training_row_count": int(len(training_data)),
        "data_source_path": str(HISTORICAL_TRAINING_PATH),
        "metrics": metrics_df,
        "feature_importance": get_feature_importance(selected_model, selected_name),
    }


def get_feature_importance(model: Pipeline | None, model_type: str) -> pd.DataFrame:
    """Extract regressor importances or coefficients."""

    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = model.named_steps["features"].get_feature_names_out()
    regressor = model.named_steps["regressor"]
    if model_type == "ridge":
        values = getattr(regressor, "coef_", np.zeros(len(feature_names)))
    else:
        values = getattr(regressor, "feature_importances_", np.zeros(len(feature_names)))

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance["abs_importance"] = importance["importance"].abs()
    return importance.sort_values("abs_importance", ascending=False).drop(columns="abs_importance")


def predict_expected_impact(df: pd.DataFrame, trained: dict[str, object]) -> pd.Series:
    """Predict expected World Cup impact, falling back to the baseline score."""

    fallback = (
        pd.to_numeric(df["baseline_expected_impact_score"], errors="coerce")
        if "baseline_expected_impact_score" in df.columns
        else pd.Series(pd.NA, index=df.index)
    )
    model = trained.get("model")
    if model is None:
        return fallback.clip(0, 100).round(1)

    X = prepare_expected_impact_features(df)
    return pd.Series(np.clip(model.predict(X), 0, 100), index=df.index).round(1)


def explain_prediction(player_row: pd.Series) -> dict[str, object]:
    """Explain one expected-impact estimate through the transparent baseline."""

    factors = []
    for label, (col, weight) in BASELINE_COMPONENTS.items():
        value = float(player_row.get(col, 50))
        contribution = value * weight
        factors.append(
            {
                "factor": label,
                "score": round(value, 1),
                "weight": weight,
                "contribution": round(contribution, 1),
                "direction": "positive" if value >= 65 else "negative" if value < 50 else "neutral",
            }
        )

    factor_df = pd.DataFrame(factors).sort_values("contribution", ascending=False)
    positive = factor_df[factor_df["direction"].isin(["positive", "neutral"])].head(4)
    negative = factor_df[factor_df["direction"] == "negative"].sort_values("score").head(3)
    return {
        "top_positive_factors": positive.to_dict("records"),
        "top_negative_factors": negative.to_dict("records"),
        "all_factors": factor_df.to_dict("records"),
    }


def confidence_level(player_row: pd.Series) -> str:
    """Simple confidence band based on data availability and model source."""

    if str(player_row.get("model_expected_impact_source", "")).startswith("historical"):
        return "Model-based"
    minutes = float(player_row.get("minutes", 0))
    expected_minutes = float(player_row.get("expected_minutes_score", 50))
    recent_form = float(player_row.get("recent_form_score", 50))
    confidence_score = 0.55 * min(minutes / 2400, 1) * 100 + 0.25 * expected_minutes + 0.20 * recent_form
    if confidence_score >= 76:
        return "High baseline"
    if confidence_score >= 58:
        return "Medium baseline"
    return "Low baseline"
