"""Data loading, model comparison and the saved predictor.

The data is the cleaned Boston housing sample used in the Udacity machine
learning course: 489 neighbourhoods and three features.

* ``RM``      - average number of rooms per home
* ``LSTAT``   - percentage of residents with lower socio-economic status
* ``PTRATIO`` - pupils per teacher in local schools
* ``MEDV``    - median home value in USD (target, scaled for inflation)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "housing.csv"
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"

FEATURES = ["RM", "LSTAT", "PTRATIO"]
TARGET = "MEDV"

# Valid input ranges for the app, taken from the data (with a small margin).
LIMITS = {"RM": (3.0, 9.0), "LSTAT": (1.0, 40.0), "PTRATIO": (12.0, 23.0)}


def load(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(FEATURES + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)


def candidate_models(random_state: int = 42) -> dict:
    """Models compared in ``scripts/train.py``, each with a small hyper-parameter grid."""
    return {
        "Mean (baseline)": (DummyRegressor(), {}),
        "Linear regression": (Pipeline([("scale", StandardScaler()), ("m", LinearRegression())]), {}),
        "Polynomial regression (degree 2)": (
            Pipeline([("poly", PolynomialFeatures(2, include_bias=False)), ("scale", StandardScaler()), ("m", LinearRegression())]),
            {},
        ),
        "Decision tree": (DecisionTreeRegressor(random_state=random_state), {"max_depth": list(range(2, 11))}),
        "Random forest": (
            RandomForestRegressor(n_estimators=400, random_state=random_state),
            {"min_samples_leaf": [1, 3, 5, 10], "max_features": [1, 2, 3]},
        ),
        "Gradient boosting": (
            GradientBoostingRegressor(random_state=random_state),
            {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
        ),
    }


def tune(model, grid: dict, X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Grid search with 5-fold CV (R²). Models without a grid are returned unchanged."""
    if not grid:
        return model
    search = GridSearchCV(model, grid, cv=KFold(5, shuffle=True, random_state=random_state), scoring="r2", n_jobs=-1)
    search.fit(X, y)
    return search.best_estimator_


@dataclass
class Prediction:
    price: float
    low: float
    high: float


class HousePricePredictor:
    """A fitted model plus a residual-based 80% prediction interval.

    The interval uses the 10th and 90th percentiles of the cross-validated
    residuals, so it reflects how wrong the model has actually been on unseen
    neighbourhoods.
    """

    def __init__(self, model, residual_q10: float, residual_q90: float, model_name: str, cv_r2: float) -> None:
        self.model = model
        self.q10 = residual_q10
        self.q90 = residual_q90
        self.model_name = model_name
        self.cv_r2 = cv_r2

    def predict(self, rm: float, lstat: float, ptratio: float) -> Prediction:
        validate_inputs(rm, lstat, ptratio)
        x = pd.DataFrame([[rm, lstat, ptratio]], columns=FEATURES)
        p = float(self.model.predict(x)[0])
        return Prediction(price=p, low=max(p + self.q10, 0.0), high=p + self.q90)

    def save(self, path: Path = MODEL_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path = MODEL_PATH) -> "HousePricePredictor":
        return joblib.load(path)


def validate_inputs(rm: float, lstat: float, ptratio: float) -> None:
    """Reject values outside the range the model was trained on."""
    for name, value in zip(FEATURES, (rm, lstat, ptratio)):
        lo, hi = LIMITS[name]
        if not isinstance(value, (int, float)) or np.isnan(value) or not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
