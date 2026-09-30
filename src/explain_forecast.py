"""
Explain the XGBoost forecast with SHAP, in plain English.
Owner: Jessica Miyumo

This is FR4. The point is not the SHAP plot, it is the sentences underneath it.
Jill can read a beeswarm plot. Lucas cannot, and he is the one who has to justify a
decision to a faculty committee, so the output he needs is a sentence like "the
forecast is 40 MW above average, mostly because demand was high at this hour last
week". The plot is here as the evidence behind those sentences, not as the answer.

Run the training script first so there is a saved model to explain:
    python src/train_xgboost.py
    python src/explain_forecast.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from features import load_data, make_features, chronological_split, HORIZON

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "xgboost.json"
CHART_PATH = REPO_ROOT / "results" / "shap_drivers.png"
SENTENCES_PATH = REPO_ROOT / "results" / "shap_explanations.json"

# Rows to run SHAP over, and the random seed to make the sample reproducible. The dashboard does not need to see every row, and SHAP is slow on large datasets.
SAMPLE_SIZE = 1000
SEED = 42

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BAR = "#2a78d6"

# Simplified version of the feature groups in plot_results.py, because this script does not need to know about every single lag and rolling mean. The dashboard does not show them.
FEATURE_GROUPS = {
    "lag_24": "demand at this hour yesterday",
    "lag_48": "demand at this hour two days ago",
    "lag_168": "demand at this hour last week",
    "lag_336": "demand at this hour two weeks ago",
    "roll_mean_24": "average demand over the previous day",
    "roll_mean_168": "average demand over the previous week",
    "hour": "the time of day",
    "hour_sin": "the time of day",
    "hour_cos": "the time of day",
    "dayofweek": "the day of the week",
    "day_sin": "the day of the week",
    "day_cos": "the day of the week",
    "is_weekend": "it being a weekend",
    "month": "the time of year",
    "holiday": "it being a public holiday",
    "school": "schools being in session",
}

# Four measures at three stations. Lucas does not care which station.
WEATHER_GROUPS = {
    "t2m": "the temperature",
    "qv2m": "the humidity",
    "tql": "the cloud and rain water in the air",
    "w2m": "the wind speed",
}


def group_of(feature):
    """Map a column name to the plain English thing it actually represents."""
    if feature in FEATURE_GROUPS:
        return FEATURE_GROUPS[feature]
    prefix = feature.split("_")[0]
    if prefix in WEATHER_GROUPS:
        return WEATHER_GROUPS[prefix]
    return feature  


def grouped_shap(shap_values, feature_cols):
    """Add up the SHAP values of every column that belongs to the same plain English group."""
    frame = pd.DataFrame(shap_values, columns=feature_cols)
    # Sending the colmn names through group_of() gives a Series of the same length as the number of columns, which is what groupby() needs. The sum() then adds up all the columns that have the same group name.
    return frame.T.groupby(frame.columns.map(group_of)).sum().T


def plot_global_drivers(grouped):
    """Horizontal bar of what drives the forecast on average across the test set."""
    # Absolute zeros are not interesting, so drop them. The rest are sorted by the average absolute effect on the forecast.
    importance = grouped.abs().mean().sort_values()

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    ax.barh(importance.index, importance.values, color=BAR, height=0.68, zorder=3)
    for name, value in importance.items():
        ax.annotate(f"{value:.1f}", xy=(value, name), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    fontsize=8.5, color=INK)

    ax.set_title("What drives the day-ahead forecast", fontsize=13, color=INK,
                 loc="left", pad=16)
    ax.set_xlabel("Average effect on the forecast (MW)", fontsize=9, color=MUTED)
    ax.tick_params(axis="y", labelsize=9.5, colors=INK, length=0)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.set_xlim(0, importance.max() * 1.15)

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved drivers chart to {CHART_PATH}")
    return importance


def explain_one(timestamp, prediction, actual, base_value, contributions, top_n=3):
    """
    Turn one row of grouped SHAP values into sentences.

    The function the dashboard calls. Returns a dict rather than printing, so the
    dashboard decides the layout, not me from inside the model code.
    """
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    drivers = []
    for name, value in ranked.head(top_n).items():
        direction = "pushed it up" if value > 0 else "pulled it down"
        drivers.append({
            "driver": name,
            "effect_mw": round(float(value), 1),
            "sentence": f"{name.capitalize()} {direction} by {abs(value):.0f} MW.",
        })

    difference = prediction - base_value
    above_or_below = "above" if difference > 0 else "below"
    day_label = f"{timestamp:%A} {timestamp.day} {timestamp:%B}"
    hour_label = timestamp.strftime("%I%p").lstrip("0").lower()
    headline = (
        f"The forecast for {day_label} at {hour_label} is {prediction:,.0f} MW, "
        f"which is {abs(difference):.0f} MW {above_or_below} the average level "
        "the model has learned from the data."
    )

    return {
        "timestamp": timestamp.isoformat(),
        "forecast_mw": round(float(prediction), 1),
        "actual_mw": None if actual is None else round(float(actual), 1),
        "headline": headline,
        "drivers": drivers,
    }


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"No saved model at {MODEL_PATH}. Run python src/train_xgboost.py first."
        )

    df = load_data()
    data, feature_cols = make_features(df)
    _, test = chronological_split(data)

    model = xgb.XGBRegressor()
    model.load_model(MODEL_PATH)

    sample = test.sample(min(SAMPLE_SIZE, len(test)), random_state=SEED).sort_index()
    X = sample[feature_cols]

    # TreeExplainer for gradient boosting: exact, not the sampling approximation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    grouped = grouped_shap(shap_values, feature_cols)

    importance = plot_global_drivers(grouped)

    print("\n=== Average effect on the forecast (MW) ===")
    print(importance.sort_values(ascending=False).round(2).to_string())

    # Highest, lowest and median rather than three random rows, so the report shows the
    # sentences holding up at the extremes and not just on a cherry-picked hour.
    predictions = pd.Series(model.predict(X), index=sample.index)
    picks = {
        "highest forecast": predictions.idxmax(),
        "lowest forecast": predictions.idxmin(),
        "typical forecast": (predictions - predictions.median()).abs().idxmin(),
    }

    explanations = {}
    print("\n=== Worked examples ===")
    for label, idx in picks.items():
        position = sample.index.get_loc(idx)
        item = explain_one(
            timestamp=sample.loc[idx, "datetime"],
            prediction=predictions.loc[idx],
            actual=sample.loc[idx, "nat_demand"],
            base_value=float(explainer.expected_value),
            contributions=grouped.iloc[position],
        )
        explanations[label] = item
        print(f"\n[{label}]  actual was {item['actual_mw']:,.0f} MW")
        print(item["headline"])
        for driver in item["drivers"]:
            print("  - " + driver["sentence"])

    SENTENCES_PATH.write_text(json.dumps({
        "forecast_horizon_hours": HORIZON,
        "average_effect_mw": importance.sort_values(ascending=False).round(3).to_dict(),
        "examples": explanations,
    }, indent=2))
    print(f"\nSaved explanations to {SENTENCES_PATH}")


if __name__ == "__main__":
    main()