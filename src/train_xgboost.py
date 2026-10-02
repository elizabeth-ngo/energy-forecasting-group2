"""
Train and evaluate the XGBoost baseline model for energy load forecasting.
Owner: Jessica Miyumo
Steps:
1. Read features and split by time (from features.py)
2. Outperform a seasonal-naive baseline (demand = same hour last week)
3. Teaching XGBoost to use the built-in features
4. Evaluate both using the common metrics (MAE, RMSE and MAPE)
5. Store results and add features important for the dashboard.
"""

import json
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from features import load_data, make_features, chronological_split, HORIZON
# Elizabeth's shared metrics, so every model is scored by the same functions.
from metrics import mae, rmse, mape, evaluate_all

# Anchored to the repo root so it saves to the same place wherever I run it from.
REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "xgboost_results.json"
# Saved so explain_forecast.py explains this exact model, not a retrained one.
MODEL_PATH = REPO_ROOT / "models" / "xgboost.json"
# Written with the same columns as every other model, so the dashboard reads them the same way.
PREDICTIONS_PATH = REPO_ROOT / "results" / "xgboost_predictions.csv"

SEED = 42
TEST_YEAR = 2020


def main():
    # 1. Load , feature engineer, and split the data
    df = load_data()
    data, feature_cols = make_features(df)
    train, test = chronological_split(data, test_year=TEST_YEAR)

    X_train, y_train = train[feature_cols], train["nat_demand"]
    X_test, y_test = test[feature_cols], test["nat_demand"]

    results = {}

    # 2. Baseline: seasonal-naive
    # such as lag_168 is already a feature, we can just use that column for the baseline prediction
    baseline_pred = test["lag_168"].to_numpy()
    results["Seasonal-naive baseline"] = evaluate_all(y_test, baseline_pred)

    # 3. Train XGBoost
    # In a dict so the same values go into the model and into the saved run config.
    params = {
        "n_estimators": 400,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": SEED,
    }

    start = time.time()
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    train_secs = time.time() - start

    xgb_pred = model.predict(X_test)
    results["XGBoost"] = evaluate_all(y_test, xgb_pred)
    results["XGBoost"]["TrainingSeconds"] = round(train_secs, 2)

    # 4. Results summary
    print("\n=== Model comparison (2020 test set) ===")
    for name, scores in results.items():
        line = " | ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                           for k, v in scores.items())
        print(f"{name:26s} {line}")

    # 5. Feature importance for the dashboard
    importance = (pd.Series(model.feature_importances_, index=feature_cols)
                  .sort_values(ascending=False))
    print("\n=== Top 10 features driving the forecast ===")
    print(importance.head(10).round(3).to_string())

    # 6. Save the scores and the settings that produced them (FR3).
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        "model": "XGBoost",
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
    model.save_model(MODEL_PATH)
    print(f"Saved fitted model to {MODEL_PATH}")

    # 7. Save the predictions for the dashboard. actual_demand is in there too because
    # the forecast versus actuals chart needs both, not just the predicted value.
    pd.DataFrame({
        "datetime": test["datetime"].values,
        "actual_demand": y_test.values,
        "predicted_demand": xgb_pred,
        "model_name": "XGBoost",
    }).to_csv(PREDICTIONS_PATH, index=False)
    print(f"Saved predictions to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()