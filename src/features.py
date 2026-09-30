"""
Feature engineering for the energy load forecasting XGBoost model.
Owner :Jessica  Miyumo
Using Elizabeth's cleaned dataset, adds the time-series features XGBoost needs, because tree models do not understand time order on their own.
"""

import numpy as np
import pandas as pd

# When transfering to GOOGle collab ...#remember to change to ensure teh path is correct
DATA_PATH = "data/processed/cleaned_dataset.csv"
# i used to test the code on my local machine and it worked fine but when i tried to run it on google collab it gave me an error that the file was not found so i had to change the path to the correct one

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


def make_features(df, lags=(24, 168)):
    """Add lag and rolling-mean features on the target (nat_demand)."""
    data = df.copy()

    # Lag features
    for lag in lags:
        data[f"lag_{lag}"] = data["nat_demand"].shift(lag)

    # Rolling short-term and long-term averages of the target
    shifted = data["nat_demand"].shift(1)
    data["roll_mean_24"] = shifted.rolling(24).mean()
    data["roll_mean_168"] = shifted.rolling(168).mean()

    # Weather columns already in Elizabeth's  data 
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

    # Removes any rows with NaN values in the features or target (first 168 rows due to lags)
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