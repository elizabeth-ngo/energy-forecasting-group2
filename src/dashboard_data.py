"""
Data layer for the energy demand forecasting dashboard.

Everything in this file is plain pandas/numpy (no Streamlit), so it can be
checked on its own before launching the dashboard:

    python src/dashboard_data.py

It reads the files the modelling leads save in results/:

    <model>_predictions.csv   datetime, actual_demand, predicted_demand, model_name
    <model>_results.json      scores, run settings, (some models) top features
    shap_explanations.json    plain-English explanations of XGBoost forecasts

Scores in the comparison table are recomputed from the predictions with the
shared functions in src/metrics.py, so the numbers on screen always match the
curves on screen. The JSON files are used for the seasonal-naive baseline
(it has no prediction file), training time and run settings. If a JSON score
disagrees with the recomputed one, a warning is raised instead of hiding it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import mae, mape, rmse

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

# Display name -> file prefix inside results/
MODEL_PREFIXES = {
    "XGBoost": "xgboost",
    "CatBoost": "catboost",
    "LSTM": "lstm",
    "GRU": "gru",
}
BASELINE = "Seasonal-naive baseline"
ACTUAL = "Actual demand"
SHAP_FILE = "shap_explanations.json"
PREDICTION_COLUMNS = ["datetime", "actual_demand", "predicted_demand", "model_name"]

# Same colours as results/model_comparison.png so the dashboard and the report match
COLORS = {
    ACTUAL: "#222222",
    BASELINE: "#2b78d6",
    "XGBoost": "#eb6a34",
    "CatBoost": "#2bbf8b",
    "LSTM": "#eda100",
    "GRU": "#e87aa4",
}

CLOSE_TOLERANCE = 0.05              # MAE within 5% of the best model counts as "level"
SCORE_MATCH_TOLERANCE = 0.01        # MW allowed between JSON score and recomputed score
EXPLANATION_MATCH_TOLERANCE = 0.15  # MW; the explanation file rounds to 0.1


@dataclass
class DashboardData:
    wide: pd.DataFrame          # datetime, actual_demand, one column per model
    models: list                # models that loaded, in display order
    results: dict               # model -> parsed <model>_results.json
    baseline: dict | None       # {"MAE":..., "RMSE":..., "MAPE (%)":...}
    shap: dict | None           # parsed shap_explanations.json
    examples: list              # parsed explanation examples
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- loading

def results_signature(results_dir: Path = RESULTS_DIR) -> tuple:
    """Fingerprint of the results folder. Changes when any file is updated,
    which lets the dashboard reload automatically after a model is re-run."""
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return ()
    return tuple(
        sorted(
            (p.name, p.stat().st_mtime_ns)
            for p in results_dir.iterdir()
            if p.suffix in {".csv", ".json"}
        )
    )


def load_predictions(model: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    path = Path(results_dir) / f"{MODEL_PREFIXES[model]}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found")
    df = pd.read_csv(path)
    missing = [c for c in PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns {missing}")
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    for col in ("actual_demand", "predicted_demand"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["datetime", "actual_demand", "predicted_demand"]].isnull().any().any():
        raise ValueError(f"{path.name} has empty or unreadable values")
    if df["datetime"].duplicated().any():
        raise ValueError(f"{path.name} has repeated timestamps")
    return df.sort_values("datetime").reset_index(drop=True)


def load_results_json(model: str, results_dir: Path = RESULTS_DIR) -> dict:
    path = Path(results_dir) / f"{MODEL_PREFIXES[model]}_results.json"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if model not in data.get("results", {}):
        raise ValueError(f"{path.name} has no scores for {model}")
    return data


def _combine(frames: dict, warnings: list) -> pd.DataFrame:
    """Join every model's predictions on datetime into one wide table."""
    base = None
    for model, df in frames.items():
        part = df[["datetime", "actual_demand", "predicted_demand"]].rename(
            columns={"predicted_demand": model}
        )
        if base is None:
            base = part
            continue
        merged = base.merge(part, on="datetime", how="inner", suffixes=("", "__other"))
        if merged.empty:
            raise ValueError(f"{model} shares no hours with the other models")
        if len(merged) != len(base) or len(merged) != len(part):
            warnings.append(
                f"{model} covers different hours from the models before it. "
                f"Only the {len(merged):,} hours they share are shown."
            )
        if not np.allclose(merged["actual_demand"], merged["actual_demand__other"], atol=1e-3):
            warnings.append(
                f"{model} reports different actual demand from the other models "
                "for the same hours, so the models may not use the same data."
            )
        base = merged.drop(columns=["actual_demand__other"])
    return base.reset_index(drop=True)


def _parse_examples(shap: dict, warnings: list) -> list:
    raw = shap.get("examples", {})
    if not isinstance(raw, dict):
        warnings.append("shap_explanations.json: 'examples' is not in the expected format and was ignored.")
        return []
    out = []
    for label, ex in raw.items():
        try:
            out.append(
                {
                    "label": label[:1].upper() + label[1:],
                    "timestamp": pd.Timestamp(ex["timestamp"]),
                    "forecast_mw": float(ex["forecast_mw"]),
                    "actual_mw": float(ex["actual_mw"]),
                    "headline": str(ex["headline"]),
                    "drivers": [(str(d["sentence"]), float(d["effect_mw"])) for d in ex["drivers"]],
                }
            )
        except (KeyError, TypeError, ValueError) as err:
            warnings.append(f"Explanation '{label}' was skipped because it is incomplete ({err}).")
    return out


def _load_shap(results_dir: Path, warnings: list):
    path = Path(results_dir) / SHAP_FILE
    if not path.exists():
        warnings.append(f"{SHAP_FILE} not found, so the explanation views are empty.")
        return None, []
    try:
        with open(path, encoding="utf-8") as f:
            shap = json.load(f)
    except ValueError as err:
        warnings.append(f"{SHAP_FILE} could not be read ({err}).")
        return None, []
    return shap, _parse_examples(shap, warnings)


def load_all(results_dir: Path = RESULTS_DIR) -> DashboardData:
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results folder not found: {results_dir}")

    warnings: list = []
    frames: dict = {}
    for model in MODEL_PREFIXES:
        try:
            frames[model] = load_predictions(model, results_dir)
        except (FileNotFoundError, ValueError) as err:
            warnings.append(f"{model} is not shown: {err}.")
    if not frames:
        raise ValueError(f"No usable prediction files found in {results_dir}. " + " ".join(warnings))

    wide = _combine(frames, warnings)
    models = list(frames)

    results: dict = {}
    for model in models:
        try:
            results[model] = load_results_json(model, results_dir)
        except (FileNotFoundError, ValueError) as err:
            warnings.append(f"{model}: {err}. Training time and run settings are unavailable.")

    baseline = None
    for model, res in results.items():
        candidate = res["results"].get(BASELINE)
        if not candidate:
            continue
        if baseline is None:
            baseline = candidate
        elif abs(candidate["MAE"] - baseline["MAE"]) > SCORE_MATCH_TOLERANCE:
            warnings.append(
                f"The baseline score in {model}'s results file differs from the other files, "
                "so the models may not have been tested on the same data."
            )

    for model, res in results.items():
        reported = res["results"].get(model)
        if reported and abs(reported["MAE"] - mae(wide["actual_demand"], wide[model])) > SCORE_MATCH_TOLERANCE:
            warnings.append(
                f"The MAE in {MODEL_PREFIXES[model]}_results.json does not match "
                f"{MODEL_PREFIXES[model]}_predictions.csv. The dashboard uses the predictions file; "
                "the two files may come from different runs."
            )

    shap, examples = _load_shap(results_dir, warnings)
    if "XGBoost" in wide.columns and examples:
        lookup = wide.set_index("datetime")["XGBoost"]
        for ex in examples:
            ts = ex["timestamp"]
            if ts not in lookup.index:
                warnings.append(f"Explanation '{ex['label']}' is for {ts}, which is outside the prediction range.")
            elif abs(lookup[ts] - ex["forecast_mw"]) > EXPLANATION_MATCH_TOLERANCE:
                warnings.append(
                    f"Explanation '{ex['label']}' says the forecast was {ex['forecast_mw']:,.1f} MW "
                    f"but xgboost_predictions.csv has {lookup[ts]:,.1f} MW for that hour."
                )

    return DashboardData(
        wide=wide,
        models=models,
        results=results,
        baseline=baseline,
        shap=shap,
        examples=examples,
        warnings=warnings,
    )


# ---------------------------------------------------------------- scores

def scores_table(data: DashboardData) -> pd.DataFrame:
    """One row per model (plus the baseline) over the whole test period."""
    rows = []
    if data.baseline:
        rows.append(
            {
                "Model": BASELINE,
                "MAE (MW)": data.baseline["MAE"],
                "RMSE (MW)": data.baseline["RMSE"],
                "MAPE (%)": data.baseline["MAPE (%)"],
                "Training time (s)": np.nan,
            }
        )
    for model in data.models:
        actual, pred = data.wide["actual_demand"], data.wide[model]
        reported = data.results.get(model, {}).get("results", {}).get(model, {})
        rows.append(
            {
                "Model": model,
                "MAE (MW)": mae(actual, pred),
                "RMSE (MW)": rmse(actual, pred),
                "MAPE (%)": mape(actual, pred),
                "Training time (s)": reported.get("TrainingSeconds", np.nan),
            }
        )
    df = pd.DataFrame(rows)
    if data.baseline:
        base = data.baseline["MAE"]
        df["MAE vs baseline (%)"] = (df["MAE (MW)"] - base) / base * 100
        df.loc[df["Model"] == BASELINE, "MAE vs baseline (%)"] = np.nan
    return df


def window_scores(window: pd.DataFrame, models: list) -> pd.DataFrame:
    """Scores for the hours currently selected on the dashboard."""
    rows = []
    for model in models:
        rows.append(
            {
                "Model": model,
                "MAE (MW)": mae(window["actual_demand"], window[model]),
                "RMSE (MW)": rmse(window["actual_demand"], window[model]),
                "MAPE (%)": mape(window["actual_demand"], window[model]),
            }
        )
    return pd.DataFrame(rows)


def format_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a numeric scores table into display strings (avoids NaN showing as 'None')."""
    out = pd.DataFrame({"Model": df["Model"]})
    out["MAE (MW)"] = df["MAE (MW)"].map(lambda v: f"{v:,.1f}")
    out["RMSE (MW)"] = df["RMSE (MW)"].map(lambda v: f"{v:,.1f}")
    out["MAPE (%)"] = df["MAPE (%)"].map(lambda v: f"{v:.2f}")
    if "MAE vs baseline (%)" in df.columns:
        out["Error vs baseline"] = df["MAE vs baseline (%)"].map(
            lambda v: "-" if pd.isna(v) else f"{v:+.0f}%"
        )
    if "Training time (s)" in df.columns:
        out["Training time (s)"] = df["Training time (s)"].map(
            lambda v: "-" if pd.isna(v) else f"{v:,.1f}"
        )
    return out


def comparison_summary(scores: pd.DataFrame) -> list:
    """Plain-English sentences built from the scores, so they stay true if models are re-run."""
    models = scores[scores["Model"] != BASELINE].sort_values("MAE (MW)")
    if models.empty:
        return []
    lines = []
    best = models.iloc[0]
    lines.append(f"{best['Model']} has the lowest average error ({best['MAE (MW)']:.1f} MW).")
    if len(models) > 1:
        second = models.iloc[1]
        gap = second["MAE (MW)"] - best["MAE (MW)"]
        if gap / best["MAE (MW)"] <= CLOSE_TOLERANCE:
            lines.append(
                f"{second['Model']} is only {gap:.1f} MW behind, so the two are effectively "
                "level on this test set."
            )
    base_rows = scores[scores["Model"] == BASELINE]
    if not base_rows.empty:
        base_mae = base_rows.iloc[0]["MAE (MW)"]
        better = models[models["MAE (MW)"] < base_mae]
        worse = models[models["MAE (MW)"] >= base_mae]
        if not better.empty:
            parts = [
                f"{r['Model']} ({(base_mae - r['MAE (MW)']) / base_mae * 100:.0f}% lower error)"
                for _, r in better.iterrows()
            ]
            lines.append("Better than the baseline: " + ", ".join(parts) + ".")
        if not worse.empty:
            parts = [
                f"{r['Model']} ({(r['MAE (MW)'] - base_mae) / base_mae * 100:.0f}% higher error)"
                for _, r in worse.iterrows()
            ]
            lines.append("Worse than the baseline: " + ", ".join(parts) + ".")
    return lines


# ---------------------------------------------------------------- forecast views

def filter_window(wide: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows from the start of `start` to the end of `end` (both inclusive, by date)."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)
    return wide[(wide["datetime"] >= start_ts) & (wide["datetime"] < end_ts)]


def chart_frame(window: pd.DataFrame, models: list) -> pd.DataFrame:
    """Long format (datetime, series, demand_mw) for plotting actual demand and forecasts."""
    long = window.melt(
        id_vars="datetime",
        value_vars=["actual_demand"] + list(models),
        var_name="series",
        value_name="demand_mw",
    )
    long["series"] = long["series"].replace({"actual_demand": ACTUAL})
    return long


def error_band(wide: pd.DataFrame, model: str, low: float = 0.10, high: float = 0.90):
    """Offsets (MW) to add to a forecast so that actual demand fell inside the
    resulting range in (high - low) of the test hours. Based on the model's own
    errors over the whole test period, so it describes past performance only."""
    residual = wide["actual_demand"] - wide[model]
    lo, hi = residual.quantile([low, high])
    return float(lo), float(hi)


def band_frame(window: pd.DataFrame, model: str, offsets: tuple) -> pd.DataFrame:
    lo, hi = offsets
    return pd.DataFrame(
        {
            "datetime": window["datetime"],
            "lower": window[model] + lo,
            "upper": window[model] + hi,
        }
    )


def forecast_row(wide: pd.DataFrame, ts: pd.Timestamp):
    rows = wide[wide["datetime"] == ts]
    return None if rows.empty else rows.iloc[0]


def find_example(examples: list, ts: pd.Timestamp):
    return next((e for e in examples if e["timestamp"] == ts), None)


# ---------------------------------------------------------------- explanations and model details

def driver_table(shap: dict | None) -> pd.DataFrame:
    avg = (shap or {}).get("average_effect_mw", {})
    df = pd.DataFrame(
        {"driver": list(avg.keys()), "effect_mw": [float(v) for v in avg.values()]}
    )
    return df.sort_values("effect_mw", ascending=False).reset_index(drop=True)


_DAILY_LAGS = {24: "yesterday", 48: "two days ago", 168: "last week", 336: "two weeks ago"}
_WEATHER = {"t2m": "Temperature", "qv2m": "Humidity", "tql": "Cloud and rain water", "w2m": "Wind speed"}
_FIXED = {
    "holiday": "Public holiday",
    "is_weekend": "Weekend",
    "dayofweek": "Day of the week",
    "month": "Month",
    "hour": "Hour of the day",
    "hour_sin": "Time of day (cyclic, sine)",
    "hour_cos": "Time of day (cyclic, cosine)",
}


def friendly_feature(name: str) -> str:
    """Readable label for a model input, keeping the raw name in brackets.
    Names that are not recognised are shown unchanged rather than guessed."""
    m = re.fullmatch(r"lag_(\d+)", name)
    if m:
        hours = int(m.group(1))
        if hours in _DAILY_LAGS:
            return f"Demand at this hour {_DAILY_LAGS[hours]} ({name})"
        if hours % 24 == 0:
            return f"Demand at this hour {hours // 24} days ago ({name})"
        return f"Demand {hours} hours earlier ({name})"
    m = re.fullmatch(r"(t2m|qv2m|tql|w2m)_(\w+)", name)
    if m:
        return f"{_WEATHER[m.group(1)]} at station {m.group(2)} ({name})"
    if name in _FIXED:
        return f"{_FIXED[name]} ({name})"
    return name


def top_features_table(result: dict | None) -> pd.DataFrame:
    feats = (result or {}).get("top_features")
    if not feats:
        return pd.DataFrame(columns=["feature", "importance"])
    df = pd.DataFrame(
        {"feature": [friendly_feature(k) for k in feats], "importance": [float(v) for v in feats.values()]}
    )
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


RUN_CONFIG_FIELDS = [
    ("forecast_horizon_hours", "Forecast horizon (hours)"),
    ("test_year", "Test year"),
    ("n_train_rows", "Training rows"),
    ("n_test_rows", "Test rows"),
    ("n_features", "Features"),
    ("lookback_hours", "Look-back (hours)"),
    ("seed", "Random seed"),
]


def run_config_table(results: dict) -> pd.DataFrame:
    rows = []
    for model, res in results.items():
        cfg = res.get("run_config", {})
        row = {"Model": model}
        for key, label in RUN_CONFIG_FIELDS:
            value = cfg.get(key)
            if value is None:
                row[label] = "-"
            elif isinstance(value, int) and key.startswith("n_"):
                row[label] = f"{value:,}"
            else:
                row[label] = str(value)
        rows.append(row)
    return pd.DataFrame(rows)


def forecast_horizon_hours(results: dict):
    """The forecast horizon if every model agrees on one, otherwise None."""
    hours = {res.get("run_config", {}).get("forecast_horizon_hours") for res in results.values()}
    hours.discard(None)
    return hours.pop() if len(hours) == 1 else None


if __name__ == "__main__":
    # Quick check that the results folder loads: python src/dashboard_data.py
    data = load_all()
    print(f"Results folder : {RESULTS_DIR}")
    print(f"Models loaded  : {', '.join(data.models)}")
    print(
        f"Hours          : {len(data.wide):,} "
        f"({data.wide['datetime'].min()} to {data.wide['datetime'].max()})"
    )
    print("\nScores over the whole test period:")
    scores = scores_table(data)
    print(format_scores(scores).to_string(index=False))
    print("\nSummary:")
    for line in comparison_summary(scores):
        print(" -", line)
    print(f"\nExplanation examples loaded: {len(data.examples)}")
    if data.warnings:
        print("\nWarnings:")
        for w in data.warnings:
            print(" !", w)
    else:
        print("\nNo warnings: all files line up.")
