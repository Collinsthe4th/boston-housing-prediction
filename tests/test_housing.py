import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from housing.app import create_app, parse
from housing.model import FEATURES, TARGET, HousePricePredictor, load, validate_inputs


@pytest.fixture(scope="module")
def predictor():
    df = load()
    model = RandomForestRegressor(n_estimators=50, min_samples_leaf=3, random_state=0).fit(df[FEATURES], df[TARGET])
    return HousePricePredictor(model, -80000, 75000, "Random forest", 0.83)


def test_data_loads_with_expected_shape():
    df = load()
    assert list(df.columns[:4]) == ["RM", "LSTAT", "PTRATIO", "MEDV"]
    assert len(df) == 489 and df.isna().sum().sum() == 0


def test_prediction_moves_in_the_expected_direction(predictor):
    small = predictor.predict(5.0, 25.0, 21.0).price
    large = predictor.predict(7.5, 4.0, 15.0).price
    assert large > small
    p = predictor.predict(6.2, 12.0, 18.5)
    assert p.low < p.price < p.high


def test_validate_inputs_rejects_out_of_range():
    with pytest.raises(ValueError):
        validate_inputs(12, 10, 18)
    with pytest.raises(ValueError):
        validate_inputs(6, float("nan"), 18)


def test_parse_collects_errors():
    values, errors = parse({"RM": "6", "LSTAT": "abc", "PTRATIO": "40"})
    assert values["RM"] == 6.0
    assert len(errors) == 2


def test_save_and_load_roundtrip(predictor, tmp_path):
    path = tmp_path / "m.joblib"
    predictor.save(path)
    again = HousePricePredictor.load(path)
    assert again.predict(6, 10, 18).price == pytest.approx(predictor.predict(6, 10, 18).price)


def test_flask_routes(predictor):
    client = create_app(predictor).test_client()
    assert client.get("/").status_code == 200
    page = client.post("/", data={"RM": "6.5", "LSTAT": "8", "PTRATIO": "17"})
    assert page.status_code == 200 and b"Estimated median value" in page.data
    bad = client.post("/", data={"RM": "99", "LSTAT": "8", "PTRATIO": "17"})
    assert bad.status_code == 400 and b"RM must be between" in bad.data
    api = client.post("/api/predict", json={"rm": 6.5, "lstat": 8, "ptratio": 17}).get_json()
    assert api["interval_80pct"][0] < api["price_usd"] < api["interval_80pct"][1]
    assert client.post("/api/predict", json={}).status_code == 400
    assert client.get("/health").get_json()["model"] == "Random forest"
