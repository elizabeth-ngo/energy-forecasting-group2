"""
Train and evaluate the LSTM and GRU sequence models for energy load forecasting.

Owner: Jessica
 The model architecture lives in one function
(build_model) so it can be tuned or replaced without touching the data handling or the
scoring, both of which have to stay identical to the tree models for the comparison to
be fair.

    python src/train_sequence.py --model lstm
    python src/train_sequence.py --model gru
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from features import load_data, HORIZON
from metrics import evaluate_all

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
MODEL_DIR = REPO_ROOT / "models"

SEED = 42
TEST_YEAR = 2020
LOOKBACK = 192
EPOCHS = 20
BATCH_SIZE = 64

# Target first, so its column index is predictable when the scalers are fitted
TARGET = "nat_demand"
CALENDAR_COLS = ["hour_sin", "hour_cos", "day_sin", "day_cos",
                 "is_weekend", "holiday", "school"]


def make_sequences(df, feature_cols, lookback, horizon=HORIZON):
    """
   Generate the 3D arrays necessary for the recurrent layers: (samples, lookback, features).

    The element that failed to be conveyed from the traffic project. 
    That variant anticipated the following reading from the window just prior to it. 
    In this scenario, the window is required to conclude a full horizon prematurely, 
    meaning for a target at row i, it spans from i - horizon - lookback + 1 to i - horizon,
    purposefully maintaining a 24-hour separation.
    """
    values = df[feature_cols].to_numpy(dtype="float32")
    target = df[[TARGET]].to_numpy(dtype="float32")

    starts = np.arange(lookback + horizon - 1, len(df))
    windows = np.stack([values[i - horizon - lookback + 1: i - horizon + 1]
                        for i in starts])
    targets = target[starts]
    meta = df.iloc[starts][["datetime", TARGET]].reset_index(drop=True)
    return windows, targets, meta


def build_model(model_type, input_shape):
    '''
    Build a Keras model with the specified recurrent layer type (LSTM or GRU).'''

    try:
        from tensorflow.keras.callbacks import EarlyStopping  # type: ignore[import-not-found]
        from tensorflow.keras.layers import Dense, Dropout, GRU, Input, LSTM  # type: ignore[import-not-found]
        from tensorflow.keras.models import Sequential  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ModuleNotFoundError(
            "TensorFlow is required to train the sequence models. "
            "Install the project dependencies and rerun this script."
        ) from exc

    recurrent_layer = LSTM if model_type == "lstm" else GRU

    model = Sequential([
        Input(shape=input_shape),
        recurrent_layer(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=5,
                               restore_best_weights=True)
    return model, early_stop


def main(model_type, epochs, lookback):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)

    df = load_data()
    weather_cols = [c for c in df.columns
                    if any(c.startswith(p) for p in ("t2m", "qv2m", "tql", "w2m"))]
    # The target is an input too. Normal for a sequence model, and safe because of the gap.
    feature_cols = [TARGET] + CALENDAR_COLS + weather_cols

    X, y, meta = make_sequences(df, feature_cols, lookback)

    # For the test/train split, we use the same year-based split as the tree models. The
    # sequence models are not allowed to peek into the future, so we cannot use a random split. The test set is the same as the tree models, so the scores are comparable.
    is_test = meta["datetime"].dt.year >= TEST_YEAR
    X_train, X_test = X[~is_test], X[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    
    feature_scaler, target_scaler = MinMaxScaler(), MinMaxScaler()
    n_features = X_train.shape[-1]
    X_train_s = feature_scaler.fit_transform(
        X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_s = feature_scaler.transform(
        X_test.reshape(-1, n_features)).reshape(X_test.shape)
    y_train_s = target_scaler.fit_transform(y_train)
    del X_train, X_test   

    model, early_stop = build_model(model_type, X_train_s.shape[1:])

    start = time.time()
    history = model.fit(
        X_train_s, y_train_s,
        validation_split=0.15,   
        epochs=epochs, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=2, shuffle=False,
    )
    train_secs = time.time() - start

    # The model outputs scaled values, so we have to invert the scaling to get back to MW.
    predictions = target_scaler.inverse_transform(
        model.predict(X_test_s, verbose=0)).reshape(-1)
    actual = y_test.reshape(-1)

    results = {}
    test_meta = meta[is_test].reset_index(drop=True)
   
    baseline = df.set_index("datetime")[TARGET].reindex(
        test_meta["datetime"] - pd.Timedelta(hours=168)).to_numpy()
    valid = ~np.isnan(baseline)
    results["Seasonal-naive baseline"] = evaluate_all(actual[valid], baseline[valid])

    name = model_type.upper()
    results[name] = evaluate_all(actual, predictions)
    results[name]["TrainingSeconds"] = round(train_secs, 2)
    results[name]["EpochsRun"] = len(history.history["loss"])

    print(f"\n=== {name} results ({TEST_YEAR} test set) ===")
    for label, scores in results.items():
        line = " | ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
                          for k, v in scores.items())
        print(f"{label:26s} {line}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{model_type}_results.json").write_text(json.dumps({
        "model": name,
        "run_config": {
            "seed": SEED,
            "test_year": TEST_YEAR,
            "forecast_horizon_hours": HORIZON,
            "lookback_hours": lookback,
            "n_train_rows": int(len(X_train_s)),
            "n_test_rows": int(len(X_test_s)),
            "n_features": n_features,
            "hyperparameters": {
                "recurrent_units": 64, "dropout": 0.2, "dense_units": 32,
                "optimizer": "adam", "loss": "mse",
                "batch_size": BATCH_SIZE, "max_epochs": epochs,
                "early_stopping_patience": 5,
            },
            # The one model here that will not reproduce exactly on a GPU
            "determinism_note": "Seeded with tf.keras.utils.set_random_seed. Exact on "
                                "CPU; GPU training is not bit-reproducible.",
        },
        "results": results,
    }, indent=2))
    model.save(MODEL_DIR / f"{model_type}_model.keras")

    # Same columns as the tree models write, so the dashboard reads them the same way.
    pd.DataFrame({
        "datetime": test_meta["datetime"],
        "actual_demand": actual,
        "predicted_demand": predictions,
        "model_name": name,
    }).to_csv(RESULTS_DIR / f"{model_type}_predictions.csv", index=False)

    print(f"\nSaved results to {RESULTS_DIR / f'{model_type}_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LSTM or GRU baseline")
    parser.add_argument("--model", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    args = parser.parse_args()
    main(args.model, args.epochs, args.lookback)
