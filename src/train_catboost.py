"""
Train and evaluate the CatBoost model for energy load forecasting.
Owner: Jessica Miyumo

CatBoost is the second gradient boosting model in my baseline stream. It is here
so the tree side of the comparison is not just one model, because if XGBoost beats
the LSTM I want to be able to say that boosting does well on this data in general,
not that one library happened to get lucky on one split.

Steps:
1. Read features and split by time (from features.py, same as XGBoost)
2. Compare against the same seasonal-naive baseline (demand = same hour last week)
3. Train CatBoost on the exact same feature matrix XGBoost gets
4. Evaluate using the common metrics from metrics.py (MAE, RMSE and MAPE)
5. Save the scores and the run config so the results can be reproduced later
"""

import json
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor

from features import load_data, make_features, chronological_split, HORIZON
# Elizabeth's shared metrics, same as train_xgboost.py. Nothing gets redefined here.
from metrics import mae, rmse, mape, evaluate_all

# Anchored to the repo root so it saves to the same place wherever I run it from.
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "catboost_results.json"
MODEL_PATH = REPO_ROOT / "models" / "catboost.cbm"

# Have to match train_xgboost.py or the two models are not being tested on the same thing.
SEED = 42
TEST_YEAR = 2020


def main():
    # 1. Load, feature engineer, and split the data
    # Same three calls as the XGBoost script, so both models see identical data
    df = load_data()
    data, feature_cols = make_features(df)
    train, test = chronological_split(data, test_year=TEST_YEAR)

    X_train, y_train = train[feature_cols], train["nat_demand"]
    X_test, y_test = test[feature_cols], test["nat_demand"]

    results = {}

    # 2. Baseline: seasonal-naive
    # Rerun here so the script works on its own. Should match the XGBoost run exactly.
    baseline_pred = test["lag_168"].to_numpy()
    results["Seasonal-naive baseline"] = evaluate_all(y_test, baseline_pred)

    # 3. Train CatBoost
    # Close to the XGBoost settings on purpose. Neither is tuned yet, because tuning one
    # and not the other measures my time rather than the models.
    # verbose=0 stops the per-iteration printing.
    # allow_writing_files=False stops CatBoost dumping a catboost_info folder in the repo.
    params = {
        "iterations": 400,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }

    start = time.time()
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    train_secs = time.time() - start

    cat_pred = model.predict(X_test)
    results["CatBoost"] = evaluate_all(y_test, cat_pred)
    results["CatBoost"]["TrainingSeconds"] = round(train_secs, 2)

    # 4. Results summary
    print("\n=== CatBoost results (2020 test set) ===")
    for name, scores in results.items():
        line = " | ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                          for k, v in scores.items())
        print(f"{name:26s} {line}")

    # 5. Feature importance, same as the XGBoost script, for the dashboard drivers
    importance = (pd.Series(model.get_feature_importance(), index=feature_cols)
                  .sort_values(ascending=False))
    print("\n=== Top 10 features driving the forecast ===")
    print(importance.head(10).round(3).to_string())

    # 6. Save the scores and the settings that produced them (FR3)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "model": "CatBoost",
        "run_config": {
            "seed": SEED,
            "test_year": TEST_YEAR,
            "forecast_horizon_hours": HORIZON,
            "n_train_rows": len(train),
            "n_test_rows": len(test),
            "n_features": len(feature_cols),
            "hyperparameters": params,
        },
        "results": results,
        "top_features": importance.head(10).round(4).to_dict(),
    }, indent=2))
    print(f"\nSaved results and run config to {RESULTS_PATH}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Saved fitted model to {MODEL_PATH}")


if __name__ == "__main__":
    main()