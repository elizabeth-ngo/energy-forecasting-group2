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
import numpy as np
import pandas as pd
import xgboost as xgb

from features import load_data, make_features, chronological_split

# using the same metrics as the dashboard (MAE, RMSE, MAPE)
def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)

def evaluate_all(y_true, y_pred):
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred), "MAPE (%)": mape(y_true, y_pred)}


def main():
    # 1. Load , feature engineer, and split the data
    df = load_data()
    data, feature_cols = make_features(df)
    train, test = chronological_split(data)

    X_train, y_train = train[feature_cols], train["nat_demand"]
    X_test, y_test = test[feature_cols], test["nat_demand"]

    results = {}

    # 2. Baseline: seasonal-naive 
    # such as lag_168 is already a feature, we can just use that column for the baseline prediction
    baseline_pred = test["lag_168"].to_numpy()
    results["Seasonal-naive baseline"] = evaluate_all(y_test, baseline_pred)

    # 3. Train XGBoost
    start = time.time()
    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
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


if __name__ == "__main__":
    main()