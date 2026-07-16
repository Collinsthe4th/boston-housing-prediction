"""House Price Predictor: Flask web app and JSON API.

Run locally (after ``python scripts/train.py``):
    flask --app housing.app run
"""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from .model import LIMITS, MODEL_PATH, HousePricePredictor

DEFAULTS = {"RM": 6.2, "LSTAT": 12.0, "PTRATIO": 18.5}


def parse(source) -> tuple[dict, list[str]]:
    """Read RM, LSTAT and PTRATIO from a form or JSON body and collect errors."""
    values, errors = {}, []
    for name, (lo, hi) in LIMITS.items():
        raw = source.get(name, source.get(name.lower()))
        try:
            v = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name} must be a number")
            continue
        if not lo <= v <= hi:
            errors.append(f"{name} must be between {lo} and {hi}")
        values[name] = v
    return values, errors


def create_app(predictor: HousePricePredictor | None = None) -> Flask:
    app = Flask(__name__)
    state = {"predictor": predictor}

    def get() -> HousePricePredictor:
        if state["predictor"] is None:
            if not MODEL_PATH.exists():
                raise RuntimeError("No trained model found. Run: python scripts/train.py")
            state["predictor"] = HousePricePredictor.load()
        return state["predictor"]

    @app.route("/", methods=["GET", "POST"])
    def home():
        pred = get()
        ctx = {"values": DEFAULTS.copy(), "limits": LIMITS, "result": None, "errors": [],
               "model_name": pred.model_name, "cv_r2": pred.cv_r2}
        if request.method == "POST":
            values, errors = parse(request.form)
            ctx["values"].update(values)
            ctx["errors"] = errors
            if not errors:
                ctx["result"] = pred.predict(values["RM"], values["LSTAT"], values["PTRATIO"])
        return render_template("index.html", **ctx), (400 if ctx["errors"] else 200)

    @app.post("/api/predict")
    def api_predict():
        body = request.get_json(silent=True) or {}
        values, errors = parse(body)
        if errors:
            return jsonify(errors=errors), 400
        p = get().predict(values["RM"], values["LSTAT"], values["PTRATIO"])
        return jsonify(inputs=values, price_usd=round(p.price, 0), interval_80pct=[round(p.low, 0), round(p.high, 0)])

    @app.get("/health")
    def health():
        p = get()
        return jsonify(status="ok", model=p.model_name, cv_r2=p.cv_r2)

    return app
