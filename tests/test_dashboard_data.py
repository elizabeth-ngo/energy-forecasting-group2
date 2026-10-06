"""
Checks for src/dashboard_data.py (35 checks).

Run from the repository root:

    python tests/test_dashboard_data.py

It reads the real files in results/. Whenever a check needs to break something
(delete a file, damage a value), it does so on a temporary copy, so nothing in
the repository is ever changed. Prints OK or FAIL for each check and exits with
a non-zero code if any check fails.
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import dashboard_data as dd  # noqa: E402

RESULTS = REPO_ROOT / "results"
passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"OK   {name}")
    else:
        failed += 1
        print(f"FAIL {name}")


def results_copy():
    """A throwaway copy of results/ that a check is free to damage."""
    folder = Path(tempfile.mkdtemp())
    for f in RESULTS.iterdir():
        if f.suffix in {".csv", ".json"}:
            shutil.copy(f, folder / f.name)
    return folder


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


if not RESULTS.is_dir():
    sys.exit(f"Results folder not found: {RESULTS}. Run this from the repository root.")

try:
    data = dd.load_all(RESULTS)
except (FileNotFoundError, ValueError) as err:
    sys.exit(f"The real results could not be loaded, so the checks cannot start: {err}")

# ---- 1-4: the typical-error band holds about 80% of hours, for every model
for model in data.models:
    low, high = dd.error_band(data.wide, model)
    residual = data.wide["actual_demand"] - data.wide[model]
    share = ((residual >= low) & (residual <= high)).mean()
    check(f"error band for {model} covers about 80% of hours (got {share:.3f})", abs(share - 0.80) < 0.01)

# ---- 5-7: date window and window scores
day = pd.Timestamp("2020-03-05").date()
window = dd.filter_window(data.wide, day, day)
check(f"a one-day window has 24 rows (got {len(window)})", len(window) == 24)
check(
    "the window starts at 00:00 and ends at 23:00",
    window["datetime"].min().hour == 0 and window["datetime"].max().hour == 23,
)
first_model = data.models[0]
manual_mae = (window["actual_demand"] - window[first_model]).abs().mean()
check(
    "window MAE matches a hand calculation",
    abs(dd.window_scores(window, [first_model]).iloc[0]["MAE (MW)"] - manual_mae) < 1e-9,
)

# ---- 8-9: a missing file is reported, the rest still loads
folder = results_copy()
(folder / "gru_predictions.csv").unlink()
loaded = dd.load_all(folder)
check(
    "a missing GRU predictions file skips GRU and shows a warning",
    "GRU" not in loaded.models and any("GRU" in w for w in loaded.warnings),
)

folder = results_copy()
(folder / "lstm_results.json").unlink()
loaded = dd.load_all(folder)
check(
    "a missing LSTM results file keeps LSTM and shows a warning",
    "LSTM" in loaded.models and any("LSTM" in w for w in loaded.warnings),
)
# ---- 10: the scores table still builds without that file
scores_without_json = dd.scores_table(loaded)
dd.format_scores(scores_without_json)
check("the scores table still builds without the LSTM results file", len(scores_without_json) == len(loaded.models) + 1)

# ---- 11-15: damaged prediction files are skipped, not shown wrongly
folder = results_copy()
frame = pd.read_csv(folder / "catboost_predictions.csv").drop(columns=["actual_demand"])
frame.to_csv(folder / "catboost_predictions.csv", index=False)
loaded = dd.load_all(folder)
check(
    "a predictions file with a missing column skips that model with a warning",
    "CatBoost" not in loaded.models and any("missing columns" in w for w in loaded.warnings),
)

folder = results_copy()
frame = pd.read_csv(folder / "xgboost_predictions.csv")
frame.loc[5, "predicted_demand"] = np.nan
frame.to_csv(folder / "xgboost_predictions.csv", index=False)
loaded = dd.load_all(folder)
check(
    "empty values in a predictions file skip that model with a warning",
    "XGBoost" not in loaded.models and any("empty or unreadable" in w for w in loaded.warnings),
)

folder = results_copy()
frame = pd.read_csv(folder / "lstm_predictions.csv")
pd.concat([frame, frame.iloc[[0]]]).to_csv(folder / "lstm_predictions.csv", index=False)
loaded = dd.load_all(folder)
check(
    "repeated timestamps skip that model with a warning",
    "LSTM" not in loaded.models and any("repeated" in w for w in loaded.warnings),
)

folder = results_copy()
frame = pd.read_csv(folder / "gru_predictions.csv")
frame.iloc[:100].to_csv(folder / "gru_predictions.csv", index=False)
loaded = dd.load_all(folder)
check(
    "a model covering fewer hours limits the data to the shared hours and warns",
    len(loaded.wide) == 100 and any("different hours" in w for w in loaded.warnings),
)

folder = results_copy()
frame = pd.read_csv(folder / "gru_predictions.csv")
frame["actual_demand"] += 5
frame.to_csv(folder / "gru_predictions.csv", index=False)
loaded = dd.load_all(folder)
check(
    "a model reporting different actual demand triggers a warning",
    any("different actual demand" in w for w in loaded.warnings),
)

# ---- 16-17: scores and explanations that disagree with the predictions
folder = results_copy()
payload = read_json(folder / "xgboost_results.json")
payload["results"]["XGBoost"]["MAE"] = 10.0
write_json(folder / "xgboost_results.json", payload)
loaded = dd.load_all(folder)
check("a results-file score that disagrees with the predictions triggers a warning", any("does not match" in w for w in loaded.warnings))

folder = results_copy()
payload = read_json(folder / "shap_explanations.json")
examples = payload.get("examples", {})
first_key = next(iter(examples), None)
if first_key is not None:
    examples[first_key]["forecast_mw"] = 1000.0
    write_json(folder / "shap_explanations.json", payload)
    loaded = dd.load_all(folder)
    check("an explanation whose forecast disagrees with the predictions triggers a warning", any("says the forecast was" in w for w in loaded.warnings))
else:
    check("an explanation whose forecast disagrees with the predictions triggers a warning (no examples found to alter)", False)

# ---- 18-20: explanation file problems
folder = results_copy()
payload = read_json(folder / "shap_explanations.json")
examples = payload.get("examples", {})
before = len(dd.load_all(folder).examples)
first_key = next(iter(examples), None)
if first_key is not None:
    del examples[first_key]["headline"]
    write_json(folder / "shap_explanations.json", payload)
    loaded = dd.load_all(folder)
    check(
        "an incomplete explanation is skipped with a warning and the others stay",
        len(loaded.examples) == before - 1 and any("incomplete" in w for w in loaded.warnings),
    )
else:
    check("an incomplete explanation is skipped with a warning and the others stay (no examples found to alter)", False)

folder = results_copy()
(folder / "shap_explanations.json").unlink()
loaded = dd.load_all(folder)
check(
    "a missing explanations file gives empty explanation views and a warning",
    loaded.shap is None and loaded.examples == [] and dd.driver_table(loaded.shap).empty and bool(loaded.warnings),
)

folder = results_copy()
(folder / "shap_explanations.json").write_text("{not valid json", encoding="utf-8")
loaded = dd.load_all(folder)
check("an unreadable explanations file does not crash and shows a warning", loaded.shap is None and any("could not be read" in w for w in loaded.warnings))

# ---- 21-22: no usable results at all gives a clear error
try:
    dd.load_all(Path(tempfile.mkdtemp()))
    check("an empty results folder raises a clear error", False)
except ValueError as err:
    check("an empty results folder raises a clear error", "No usable prediction files" in str(err))

try:
    dd.load_all(Path(tempfile.mkdtemp()) / "does_not_exist")
    check("a missing results folder raises a clear error", False)
except FileNotFoundError:
    check("a missing results folder raises a clear error", True)

# ---- 23: the page reloads when a result file changes
folder = results_copy()
before = dd.results_signature(folder)
time.sleep(0.05)
target = folder / "gru_predictions.csv"
target.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
check("the results signature changes after a file is rewritten", before != dd.results_signature(folder))

# ---- 24-31: readable names for model inputs
names = {
    "lag_168": "Demand at this hour last week (lag_168)",
    "lag_24": "Demand at this hour yesterday (lag_24)",
    "lag_72": "Demand at this hour 3 days ago (lag_72)",
    "lag_5": "Demand 5 hours earlier (lag_5)",
    "t2m_toc": "Temperature at station toc (t2m_toc)",
    "holiday": "Public holiday (holiday)",
    "day_sin": "day_sin",
    "roll_mean_24": "roll_mean_24",
}
for raw, expected in names.items():
    check(f"readable name for {raw}", dd.friendly_feature(raw) == expected)

# ---- 32-33: feature-importance tables
check("a results file with no feature list gives an empty importance table", dd.top_features_table({}).empty)
for model, result in data.results.items():
    table = dd.top_features_table(result)
    if not table.empty:
        values = list(table["importance"])
        check(f"{model} importance table is sorted from largest to smallest", values == sorted(values, reverse=True))
        break
else:
    check("an importance table exists for at least one model", False)

# ---- 34: run-settings table is safe to display
config = dd.run_config_table(data.results)
check("the run-settings table contains only text cells", all(config[c].map(type).eq(str).all() for c in config.columns))

# ---- 35: forecast horizon comes from the results files
horizons = {r.get("run_config", {}).get("forecast_horizon_hours") for r in data.results.values()}
horizons.discard(None)
expected = horizons.pop() if len(horizons) == 1 else None
check("the forecast horizon matches the results files", dd.forecast_horizon_hours(data.results) == expected)

print(f"\n{passed + failed} checks: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
