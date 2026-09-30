"""
Feature engineering for the energy load forecasting XGBoost model.
Owner :Jessica  Miyumo
Using Elizabeth's cleaned dataset, adds the time-series features XGBoost needs, because tree models do not understand time order on their own.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Worked out from __file__ so it finds the data from the repo root or from inside src/.
# i used to test the code on my local machine and it worked fine but when i tried to run it on google collab it gave me an error that the file was not found so i had to change the path to the correct one
# Pasting into a Colab cell has no __file__, so pass the path in: load_data(path=...)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = REPO_ROOT / "data" / "processed" / "cleaned_dataset.csv"

def load_data(path=DATA_PATH):
    """Load the cleaned dataset and add calendar/time features."""
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Add calendar/time features for the model to learn from
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Cyclical encoding so the model knows hour 23 is next to hour 0
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    return df


# How far ahead we forecast, in hours. 24 = day ahead, which is what FR5 asks for. Every lag below is built around this number, so it lives up here where it is visible.
HORIZON = 24


def make_features(df, horizon=HORIZON, lags=(24, 48, 168, 336)):
    """
    Add lag and rolling-mean features on the target (nat_demand).

    Nothing here is allowed to reach inside the last 24 hours, because a day-ahead
    forecast of hour t only knows demand up to t minus 24.

    My first version used lag_1, which made it a one-hour-ahead forecast wearing a
    day-ahead label (MAE 21 MW, but predicting 3pm tomorrow needs 2pm tomorrow).
    Restricted to what we would really have, MAE is about 53 vs a 71.65 baseline.

    horizon=1 gives back the old one-hour-ahead behaviour if we want to compare.
    """
    data = df.copy()

    if any(lag < horizon for lag in lags):
        raise ValueError(
            f"Lags {[l for l in lags if l < horizon]} are shorter than the {horizon}h "
            "forecast horizon, so that data would not exist yet at forecast time."
        )

    # Lag features: yesterday, the day before, last week, the week before that
    for lag in lags:
        data[f"lag_{lag}"] = data["nat_demand"].shift(lag)

    # Rolling averages shifted by the horizon, not by 1, so nothing leaks in from the gap
    shifted = data["nat_demand"].shift(horizon)
    data[f"roll_mean_24"] = shifted.rolling(24).mean()
    data[f"roll_mean_168"] = shifted.rolling(168).mean()

    # Weather columns already in Elizabeth's  data .These are observed weather, not forecast, so our numbers are a bit optimistic.
    weather_cols = [c for c in data.columns
                    if any(c.startswith(p) for p in ("t2m", "qv2m", "tql", "w2m"))]

    feature_cols = [
        "hour", "dayofweek", "month", "is_weekend",
        "hour_sin", "hour_cos", "day_sin", "day_cos",
        "holiday", "school",
        *[f"lag_{lag}" for lag in lags],
        "roll_mean_24", "roll_mean_168",
        *weather_cols,
    ]

    # Drops rows with NaNs (the first max(lags) + 168 or so, until the lags fill up)
    data = data.dropna(subset=feature_cols + ["nat_demand"]).reset_index(drop=True)
    return data, feature_cols


def chronological_split(df, test_year=2020):
    """Train on everything before test_year, test on test_year. No leakage."""
    train = df[df["datetime"].dt.year < test_year].reset_index(drop=True)
    test = df[df["datetime"].dt.year >= test_year].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    df = load_data()
    print("Loaded:", df.shape)
    data, feature_cols = make_features(df)
    print("After features:", data.shape)
    print("Number of features:", len(feature_cols))
    train, test = chronological_split(data)
    print("Train rows:", len(train), "| Test rows:", len(test))
    print("Train covers:", train['datetime'].min(), "->", train['datetime'].max())
    print("Test covers: ", test['datetime'].min(), "->", test['datetime'].max())