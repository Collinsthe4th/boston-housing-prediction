"""House Price Predictor in Streamlit.

The model is rebuilt at start-up with the settings chosen by nested
cross-validation in scripts/train.py (stored in results/metrics.json), so the
app needs no saved model file.

Run locally:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from housing.model import FEATURES, LIMITS, TARGET, HousePricePredictor, candidate_models, load  # noqa: E402

st.set_page_config(page_title="House Price Predictor", layout="wide")

HEADER = "https://images.unsplash.com/photo-1565402170291-8491f14678db?auto=format&fit=crop&w=1600&h=340&q=80"
st.image(HEADER, use_container_width=True)


@st.cache_resource
def build() -> tuple[HousePricePredictor, pd.DataFrame, dict]:
    metrics = json.loads((ROOT / "results" / "metrics.json").read_text())
    df = load()
    model, _ = candidate_models()[metrics["best_model"]]
    model.set_params(**metrics["best_params"])
    model.fit(df[FEATURES], df[TARGET])
    iv = metrics["interval_80pct"]
    predictor = HousePricePredictor(model, iv["residual_q10"], iv["residual_q90"], metrics["best_model"], metrics["models"][metrics["best_model"]]["r2"])
    return predictor, df, metrics


predictor, df, metrics = build()

st.title("House Price Predictor")
st.caption("This app estimates the median home value of a Boston neighbourhood from three facts about it. The random forest behind it was trained on 489 neighbourhoods.")

left, right = st.columns([1, 1.2])
with left:
    rm = st.slider("Average rooms per home", LIMITS["RM"][0], LIMITS["RM"][1], 6.2, step=0.05)
    lstat = st.slider("Lower-status residents (%)", LIMITS["LSTAT"][0], LIMITS["LSTAT"][1], 12.0, step=0.1)
    ptratio = st.slider("Pupils per teacher", LIMITS["PTRATIO"][0], LIMITS["PTRATIO"][1], 18.5, step=0.1)

with right:
    p = predictor.predict(rm, lstat, ptratio)
    st.metric("Estimated median value", f"${p.price:,.0f}")
    st.write(f"80% of similar neighbourhoods fell between **${p.low:,.0f}** and **${p.high:,.0f}**.")
    st.caption(
        "The range comes from the model's own errors on neighbourhoods it had not seen during cross-validation "
        "(10th and 90th percentiles)."
    )
    pct = float((df[TARGET] < p.price).mean())
    st.progress(pct, text=f"More expensive than {pct:.0%} of the 489 neighbourhoods")

st.subheader("How the models compared (nested 5-fold cross-validation)")
table = pd.DataFrame(metrics["models"]).T.rename(columns={"r2": "R²", "rmse": "RMSE (USD)", "mae": "MAE (USD)"})
st.dataframe(table.style.format({"R²": "{:.3f}", "RMSE (USD)": "{:,.0f}", "MAE (USD)": "{:,.0f}"}), use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    st.image(str(ROOT / "results" / "figures" / "predicted_vs_actual.png"))
with c2:
    feature = st.selectbox("Price against", FEATURES, format_func={"RM": "Rooms per home", "LSTAT": "Lower-status residents (%)", "PTRATIO": "Pupils per teacher"}.get)
    st.scatter_chart(df, x=feature, y=TARGET, size=10)

st.markdown("[Source code and README](https://github.com/Collinsthe4th/boston-housing-prediction) · Header photo from Unsplash.")
