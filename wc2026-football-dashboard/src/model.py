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
from data_sources import HISTORICAL_TRAINING_TEMPLATE_PATH


FALLBACK_WARNING = (
    "Historical World Cup training data is not available. The supervised expected-impact model is disabled until "
    "real curated historical data is added."
)
MODEL_STATUS_AVAILABLE = "available"
MODEL_STATUS_DISABLED = "disabled_missing_real_historical_data"

BASE_EXPECTED_IMPACT_FEATURES = [
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

OPTIONAL_EXPERIENCE_FEATURES = [
    "is_world_cup_debutant",
    "previous_world_cup_minutes",
    "previous_world_cup_matches",
    "previous_world_cup_impact_score",
    "senior_national_team_caps",
    "major_tournament_experience",
    "age_group",
]

EXPECTED_IMPACT_FEATURES = [*BASE_EXPECTED_IMPACT_FEATURES, *OPTIONAL_EXPERIENCE_FEATURES]

REQUIRED_HISTORICAL_COLUMNS = [
    "tournament_year",
    "player_name",
    "country",
    "club",
    "league",
    *BASE_EXPECTED_IMPACT_FEATURES,
    "actual_tournament_impact_score",
]

EXPECTED_HISTORICAL_TOURNAMENT_YEARS = {2014, 2018, 2022}
MIN_HISTORICAL_TRAINING_ROWS = 30
HISTORICAL_SCORE_COLUMNS = [
    "club_level_score",
    "league_strength_score",
    "expected_starter_score",
    "national_team_strength",
    "group_difficulty_score",
    "injury_availability_score",
    "recent_form_score",
    "role_fit_score",
    "previous_world_cup_impact_score",
    "actual_tournament_impact_score",
]

CATEGORICAL_FEATURES = ["position", "age_group"]

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


def _read_historical_file_with_columns(path: Path | str) -> pd.DataFrame:
    """Read a historical CSV while preserving zero-row template columns."""

    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def check_historical_model_readiness(
    path: Path | str = HISTORICAL_TRAINING_PATH,
    template_path: Path | str = HISTORICAL_TRAINING_TEMPLATE_PATH,
    attempt_training: bool = True,
) -> dict[str, object]:
    """Check whether real historical player-tournament data can train the model."""

    path = Path(path)
    template_path = Path(template_path)
    readiness: dict[str, object] = {
        "training_path": str(path),
        "template_path": str(template_path),
        "training_file_exists": path.exists(),
        "template_exists": template_path.exists(),
        "row_count": 0,
        "required_columns_present": False,
        "missing_required_columns": [],
        "tournament_years_available": [],
        "missing_expected_tournament_years": sorted(EXPECTED_HISTORICAL_TOURNAMENT_YEARS),
        "target_available_rows": 0,
        "target_valid_range": False,
        "score_range_issues": [],
        "can_train": False,
        "model_available": False,
        "model_status": MODEL_STATUS_DISABLED,
        "disabled_reason": "Historical World Cup training data is missing.",
        "warnings": [],
        "errors": [],
        "model_summary": None,
    }

    if not template_path.exists():
        readiness["warnings"].append("Historical training template is missing.")

    if not path.exists():
        readiness["warnings"].append("Historical training file is missing.")
        return readiness

    data = _read_historical_file_with_columns(path)
    readiness["row_count"] = int(len(data))
    missing_required = [col for col in REQUIRED_HISTORICAL_COLUMNS if col not in data.columns]
    readiness["missing_required_columns"] = missing_required
    readiness["required_columns_present"] = not missing_required

    if len(data) == 0:
        readiness["disabled_reason"] = "Historical training file exists but has zero real player-tournament rows."
        readiness["warnings"].append(readiness["disabled_reason"])
        if missing_required:
            readiness["warnings"].append("Zero-row historical file is missing required schema columns: " + ", ".join(missing_required))
        return readiness

    if missing_required:
        readiness["disabled_reason"] = "Historical training file is missing required columns."
        readiness["errors"].append(readiness["disabled_reason"] + " Missing: " + ", ".join(missing_required))
        return readiness

    years = sorted(pd.to_numeric(data["tournament_year"], errors="coerce").dropna().astype(int).unique().tolist())
    readiness["tournament_years_available"] = years
    readiness["missing_expected_tournament_years"] = sorted(EXPECTED_HISTORICAL_TOURNAMENT_YEARS - set(years))
    if readiness["missing_expected_tournament_years"]:
        readiness["warnings"].append(
            "Expected historical tournaments not yet present: "
            + ", ".join(str(year) for year in readiness["missing_expected_tournament_years"])
        )

    target = pd.to_numeric(data["actual_tournament_impact_score"], errors="coerce")
    readiness["target_available_rows"] = int(target.notna().sum())
    target_values = target.dropna()
    readiness["target_valid_range"] = bool(not target_values.empty and target_values.between(0, 100).all())
    if target_values.empty:
        readiness["errors"].append("actual_tournament_impact_score has no real target values.")
    elif not target_values.between(0, 100).all():
        readiness["errors"].append("actual_tournament_impact_score contains values outside 0-100.")

    score_range_issues: list[str] = []
    for col in HISTORICAL_SCORE_COLUMNS:
        if col not in data.columns:
            continue
        values = pd.to_numeric(data[col], errors="coerce").dropna()
        if not values.empty and not values.between(0, 100).all():
            score_range_issues.append(col)
    readiness["score_range_issues"] = score_range_issues
    if score_range_issues:
        readiness["errors"].append("Score columns outside 0-100: " + ", ".join(score_range_issues))

    if len(data) < MIN_HISTORICAL_TRAINING_ROWS:
        readiness["warnings"].append(
            f"At least {MIN_HISTORICAL_TRAINING_ROWS} valid real rows are recommended before training; found {len(data)}."
        )
    if target.nunique(dropna=True) < 2:
        readiness["warnings"].append("Target has fewer than two distinct values, so the model cannot learn a relationship.")

    no_errors = not readiness["errors"]
    enough_rows = len(data) >= MIN_HISTORICAL_TRAINING_ROWS
    enough_targets = readiness["target_available_rows"] >= MIN_HISTORICAL_TRAINING_ROWS and target.nunique(dropna=True) >= 2
    readiness["can_train"] = bool(no_errors and enough_rows and enough_targets)

    if readiness["can_train"] and attempt_training:
        model_summary = train_expected_impact_model(data)
        readiness["model_summary"] = model_summary
        readiness["model_available"] = bool(model_summary.get("model_available", False))
        readiness["model_status"] = str(model_summary.get("model_status", "unknown"))
        if readiness["model_available"]:
            readiness["disabled_reason"] = ""
        else:
            readiness["disabled_reason"] = str(model_summary.get("warning", "Model training failed."))
            readiness["warnings"].append(readiness["disabled_reason"])
    elif readiness["can_train"]:
        readiness["disabled_reason"] = "Historical training data appears ready; training was not attempted."
    elif not readiness["disabled_reason"] or readiness["disabled_reason"] == "Historical World Cup training data is missing.":
        readiness["disabled_reason"] = "Historical training data is present but not yet trainable."

    return readiness


def _coalesce_feature(df: pd.DataFrame, target: str, aliases: list[str], default: Any = pd.NA) -> pd.Series:
    values = pd.Series(pd.NA, index=df.index)
    for alias in [target, *aliases]:
        if alias in df.columns:
            candidate = df[alias]
            if target in CATEGORICAL_FEATURES:
                values = candidate.combine_first(values)
            else:
                values = pd.to_numeric(values, errors="coerce").combine_first(pd.to_numeric(candidate, errors="coerce"))
    if default is not pd.NA:
        values = values.fillna(default)
    return values


def _coalesce_boolean_feature(df: pd.DataFrame, target: str) -> pd.Series:
    if target not in df.columns:
        return pd.Series(pd.NA, index=df.index)
    series = df[target]
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").astype("Int64").astype(float)
    normalised = series.astype(str).str.strip().str.lower()
    mapped = normalised.map(
        {
            "true": 1.0,
            "1": 1.0,
            "yes": 1.0,
            "y": 1.0,
            "false": 0.0,
            "0": 0.0,
            "no": 0.0,
            "n": 0.0,
        }
    )
    mapped[series.isna()] = pd.NA
    return pd.to_numeric(mapped, errors="coerce")


def _derive_age_group(age: pd.Series) -> pd.Series:
    age = pd.to_numeric(age, errors="coerce")
    group = pd.Series(pd.NA, index=age.index, dtype="object")
    group.loc[age <= 21] = "U21"
    group.loc[age.between(22, 23)] = "U23"
    group.loc[age.between(24, 25)] = "U25"
    group.loc[age.between(26, 29)] = "Prime"
    group.loc[age >= 30] = "Veteran"
    return group


def select_expected_impact_features(training_data: pd.DataFrame) -> list[str]:
    """Use optional debutant/experience fields only when real historical values exist."""

    selected = list(BASE_EXPECTED_IMPACT_FEATURES)
    for feature in OPTIONAL_EXPERIENCE_FEATURES:
        if feature in training_data.columns and training_data[feature].notna().any():
            selected.append(feature)
    return selected


def prepare_expected_impact_features(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Map current or historical data into the model feature schema."""

    feature_columns = feature_columns or BASE_EXPECTED_IMPACT_FEATURES
    mapped = pd.DataFrame(index=df.index)
    mapped["age"] = _coalesce_feature(df, "age", [])
    mapped["position"] = _coalesce_feature(df, "position", [], "Unknown").fillna("Unknown").astype(str)
    mapped["market_value_before_tournament"] = _coalesce_feature(
        df,
        "market_value_before_tournament",
        ["market_value_eur"],
    )
    mapped["club_level_score"] = _coalesce_feature(df, "club_level_score", [])
    mapped["league_strength_score"] = _coalesce_feature(
        df,
        "league_strength_score",
        ["club_level_score"],
    )
    mapped["club_minutes_previous_season"] = _coalesce_feature(
        df,
        "club_minutes_previous_season",
        ["minutes"],
    )
    mapped["goals_previous_season"] = _coalesce_feature(df, "goals_previous_season", ["goals"])
    mapped["assists_previous_season"] = _coalesce_feature(df, "assists_previous_season", ["assists"])
    mapped["national_team_caps"] = _coalesce_feature(
        df,
        "national_team_caps",
        ["senior_national_team_caps", "caps"],
    )
    mapped["expected_starter_score"] = _coalesce_feature(
        df,
        "expected_starter_score",
        ["expected_minutes_score"],
    )
    mapped["national_team_strength"] = _coalesce_feature(df, "national_team_strength", [])
    mapped["group_difficulty_score"] = _coalesce_feature(
        df,
        "group_difficulty_score",
        ["tournament_draw_score", "draw_context_score"],
    )
    mapped["injury_availability_score"] = _coalesce_feature(
        df,
        "injury_availability_score",
        ["availability_score"],
    )
    mapped["recent_form_score"] = _coalesce_feature(df, "recent_form_score", [])
    mapped["role_fit_score"] = _coalesce_feature(df, "role_fit_score", ["best_profile_score"])

    if "is_world_cup_debutant" in feature_columns:
        mapped["is_world_cup_debutant"] = _coalesce_boolean_feature(df, "is_world_cup_debutant")
    if "previous_world_cup_minutes" in feature_columns:
        mapped["previous_world_cup_minutes"] = _coalesce_feature(df, "previous_world_cup_minutes", [])
    if "previous_world_cup_matches" in feature_columns:
        mapped["previous_world_cup_matches"] = _coalesce_feature(df, "previous_world_cup_matches", [])
    if "previous_world_cup_impact_score" in feature_columns:
        mapped["previous_world_cup_impact_score"] = _coalesce_feature(df, "previous_world_cup_impact_score", [])
    if "senior_national_team_caps" in feature_columns:
        mapped["senior_national_team_caps"] = _coalesce_feature(
            df,
            "senior_national_team_caps",
            ["national_team_caps", "caps"],
        )
    if "major_tournament_experience" in feature_columns:
        mapped["major_tournament_experience"] = _coalesce_feature(df, "major_tournament_experience", [])
    if "age_group" in feature_columns:
        existing_age_group = df["age_group"] if "age_group" in df.columns else pd.Series(pd.NA, index=df.index)
        mapped["age_group"] = existing_age_group.combine_first(_derive_age_group(mapped["age"])).astype(str)
        mapped.loc[existing_age_group.isna() & _derive_age_group(mapped["age"]).isna(), "age_group"] = "Unknown"

    mapped["market_value_before_tournament"] = np.log1p(
        pd.to_numeric(mapped["market_value_before_tournament"], errors="coerce").clip(lower=0)
    )
    numeric_features = [feature for feature in feature_columns if feature not in CATEGORICAL_FEATURES]
    for feature in numeric_features:
        mapped[feature] = pd.to_numeric(mapped[feature], errors="coerce")
    return mapped[feature_columns]


def _build_model(model_type: str, random_state: int, feature_columns: list[str]) -> Pipeline:
    numeric_features = [feature for feature in feature_columns if feature not in CATEGORICAL_FEATURES]
    categorical_features = [feature for feature in feature_columns if feature in CATEGORICAL_FEATURES]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "ridge":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
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
        "model_features": BASE_EXPECTED_IMPACT_FEATURES,
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

    feature_columns = select_expected_impact_features(training_data)
    X = prepare_expected_impact_features(training_data, feature_columns)
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
        candidate = _build_model(name, random_state, feature_columns)
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
        "model_features": feature_columns,
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

    feature_columns = list(trained.get("model_features", BASE_EXPECTED_IMPACT_FEATURES))
    X = prepare_expected_impact_features(df, feature_columns)
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
