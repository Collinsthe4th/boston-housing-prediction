"""Compare models with nested cross-validation, then fit and save the best one.

Nested CV: the outer 5 folds measure performance, and the grid search inside
each training part picks the hyper-parameters. This keeps the tuning from
leaking into the score.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from housing.model import FEATURES, TARGET, HousePricePredictor, candidate_models, load, tune  # noqa: E402

RESULTS = ROOT / "results"


def nested_cv(model, grid, X, y, seed=42):
    outer = KFold(5, shuffle=True, random_state=seed)
    preds = np.zeros(len(y))
    for tr, te in outer.split(X):
        best = tune(clone(model), grid, X.iloc[tr], y.iloc[tr], random_state=seed)
        best.fit(X.iloc[tr], y.iloc[tr])
        preds[te] = best.predict(X.iloc[te])
    return preds


def main() -> None:
    df = load()
    X, y = df[FEATURES], df[TARGET]

    scores, oof = {}, {}
    for name, (model, grid) in candidate_models().items():
        p = nested_cv(model, grid, X, y)
        oof[name] = p
        scores[name] = {
            "r2": round(float(r2_score(y, p)), 4),
            "rmse": round(float(root_mean_squared_error(y, p)), 0),
            "mae": round(float(mean_absolute_error(y, p)), 0),
        }
        print(f"{name:34s} R² {scores[name]['r2']:.4f}  RMSE {scores[name]['rmse']:,.0f}")

    best = max((n for n in scores if "baseline" not in n), key=lambda n: scores[n]["r2"])
    residuals = y.to_numpy() - oof[best]
    q10, q90 = np.percentile(residuals, [10, 90])

    model, grid = candidate_models()[best]
    final = tune(model, grid, X, y)
    final.fit(X, y)
    params = {k: v for k, v in final.get_params().items() if k in (grid or {})}
    HousePricePredictor(final, float(q10), float(q90), best, scores[best]["r2"]).save()

    metrics = {
        "rows": int(len(df)),
        "cv": "nested 5-fold (grid search inside each outer training fold)",
        "models": scores,
        "best_model": best,
        "best_params": params,
        "interval_80pct": {"residual_q10": round(float(q10), 0), "residual_q90": round(float(q90), 0)},
        "target_mean": round(float(y.mean()), 0),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y / 1000, oof[best] / 1000, s=10, alpha=0.6, color="#2b6cb0")
    lim = [0, max(y.max(), oof[best].max()) / 1000 * 1.05]
    ax.plot(lim, lim, color="black", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("Actual median value (USD thousands)")
    ax.set_ylabel("Predicted (USD thousands)")
    ax.set_title(f"{best}: out-of-fold predictions\nR² {scores[best]['r2']:.3f}, RMSE {scores[best]['rmse']:,.0f}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "predicted_vs_actual.png", dpi=150)
    plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ("best_model", "best_params", "interval_80pct")}, indent=2))


if __name__ == "__main__":
    main()
