"""Transparent success prediction utilities for the MVP."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_FEATURES = [
    "overall_score",
    "national_team_strength",
    "expected_minutes_score",
    "best_profile_score",
    "age_curve_score",
    "club_level_score",
    "recent_form_score",
    "market_value_eur",
    "minutes",
    "attacking_score",
    "creativity_score",
    "progression_score",
    "defensive_score",
    "possession_score",
]


HEURISTIC_COMPONENTS = {
    "Current performance": ("current_performance_score", 0.25),
    "National team strength": ("national_team_strength", 0.20),
    "Expected minutes": ("expected_minutes_score", 0.15),
    "Role fit": ("role_fit_score", 0.15),
    "Age curve": ("age_curve_score", 0.10),
    "Club level": ("club_level_score", 0.10),
    "Recent form": ("recent_form_score", 0.05),
}


def create_heuristic_target(df: pd.DataFrame, threshold: float = 72) -> pd.Series:
    """Create a synthetic label for the first model version."""

    return (df["success_score"] >= threshold).astype(int)


def train_success_model(
    df: pd.DataFrame,
    model_type: str = "logistic_regression",
    random_state: int = 26,
) -> dict[str, object]:
    """Train a transparent first-pass classifier against the heuristic label."""

    available_features = [col for col in MODEL_FEATURES if col in df.columns]
    model_df = df[available_features].copy()
    model_df["market_value_eur"] = np.log1p(model_df["market_value_eur"])
    y = create_heuristic_target(df)

    if y.nunique() < 2 or len(df) < 30:
        return {
            "model": None,
            "features": available_features,
            "auc": None,
            "feature_importance": pd.DataFrame(),
        }

    X_train, X_test, y_train, y_test = train_test_split(
        model_df,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=250, max_depth=5, random_state=random_state)
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
            ]
        )

    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds) if y_test.nunique() == 2 else None
    importance = get_feature_importance(model, available_features, model_type)

    return {
        "model": model,
        "features": available_features,
        "auc": auc,
        "feature_importance": importance,
    }


def get_feature_importance(model: object, features: list[str], model_type: str) -> pd.DataFrame:
    """Extract model importances or coefficients."""

    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    if model_type == "logistic_regression":
        classifier = model.named_steps["classifier"]
        values = classifier.coef_[0]
    else:
        values = getattr(model, "feature_importances_", np.zeros(len(features)))

    importance = pd.DataFrame({"feature": features, "importance": values})
    importance["abs_importance"] = importance["importance"].abs()
    return importance.sort_values("abs_importance", ascending=False).drop(columns="abs_importance")


def predict_success_probabilities(df: pd.DataFrame, trained: dict[str, object]) -> pd.Series:
    """Return model probabilities, falling back to heuristic probabilities."""

    model = trained.get("model")
    features = trained.get("features", [])
    if model is None or not features:
        return df["success_probability"]
    X = df[features].copy()
    if "market_value_eur" in X.columns:
        X["market_value_eur"] = np.log1p(X["market_value_eur"])
    return pd.Series(model.predict_proba(X)[:, 1] * 100, index=df.index).round(1)


def explain_prediction(player_row: pd.Series) -> dict[str, object]:
    """Explain one prediction using the transparent heuristic components."""

    factors = []
    for label, (col, weight) in HEURISTIC_COMPONENTS.items():
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
    """Simple confidence band based on sample size and input consistency."""

    minutes = float(player_row.get("minutes", 0))
    expected_minutes = float(player_row.get("expected_minutes_score", 50))
    recent_form = float(player_row.get("recent_form_score", 50))
    confidence_score = 0.55 * min(minutes / 2400, 1) * 100 + 0.25 * expected_minutes + 0.20 * recent_form
    if confidence_score >= 76:
        return "High"
    if confidence_score >= 58:
        return "Medium"
    return "Low"

