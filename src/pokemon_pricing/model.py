from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.model_selection import train_test_split

from pokemon_pricing.config import MODEL_DIR, PROCESSED_DIR
from pokemon_pricing.features import feature_columns, model_frame


def train_model(frame: pd.DataFrame, model_path: Path = MODEL_DIR / "pricing_model.joblib") -> dict:
    data = model_frame(frame)
    features = feature_columns()

    x = data[features]
    y = data["target_log_price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = HistGradientBoostingRegressor(
        learning_rate=0.06,
        max_iter=350,
        l2_regularization=0.05,
        random_state=42,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = {
        "training_rows": int(len(data)),
        "mae_log_price": float(mean_absolute_error(y_test, predictions)),
        "median_ae_log_price": float(median_absolute_error(y_test, predictions)),
    }

    importance = permutation_importance(model, x_test, y_test, n_repeats=5, random_state=42)
    importance_frame = pd.DataFrame(
        {
            "feature": features,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "metrics": metrics}, model_path)
    importance_frame.to_csv(PROCESSED_DIR / "feature_importance.csv", index=False)

    return metrics


def score_cards(
    frame: pd.DataFrame,
    model_path: Path = MODEL_DIR / "pricing_model.joblib",
    under_threshold: float = 0.85,
    over_threshold: float = 1.15,
) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    scored = frame.copy()
    valid = scored["observed_price"].notna() & (scored["observed_price"] > 0)
    fair_log_price = model.predict(scored.loc[valid, features])
    fair_price = np.expm1(fair_log_price)

    scored.loc[valid, "modeled_fair_price"] = fair_price
    scored.loc[valid, "pricing_ratio"] = scored.loc[valid, "observed_price"] / fair_price
    scored.loc[valid, "pricing_label"] = "well_priced"
    scored.loc[valid & (scored["pricing_ratio"] < under_threshold), "pricing_label"] = "under_priced"
    scored.loc[valid & (scored["pricing_ratio"] > over_threshold), "pricing_label"] = "over_priced"

    return scored
