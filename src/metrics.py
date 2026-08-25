# =============================================================
# Shared Evaluation Metrics
# Owner: Elizabeth (Data & Platform Engineer)
#
# Common metric functions so Jessica (ML) and Joud (DL) evaluate their
# models the same way and results are directly comparable.
#
# Usage:
#   from metrics import mae, rmse, mape, evaluate_all
#   scores = evaluate_all(y_true, y_pred)
#   print(scores)
# =============================================================

import numpy as np


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    Average absolute difference between predicted and actual demand (MW).
    Lower is better. Same unit as the target (MW).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    Like MAE, but penalises large errors more heavily (squares the
    difference before averaging). Lower is better. Same unit as the
    target (MW).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon=1e-8):
    """
    Mean Absolute Percentage Error.
    Average error as a percentage of the actual value — easier to
    interpret across different scales than MAE/RMSE. Lower is better.

    epsilon avoids division by zero if y_true contains 0 (shouldn't
    happen for nat_demand, but kept as a safety net).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)


def evaluate_all(y_true, y_pred):
    """
    Convenience function: returns all three metrics in one dict.
    Use this for consistent reporting across models (e.g. in worklogs,
    comparison tables, dashboard).
    """
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE (%)": mape(y_true, y_pred),
    }


if __name__ == "__main__":
    # Quick self-test with dummy numbers — run `python src/metrics.py`
    # to confirm the functions work on your machine.
    y_true = [100, 200, 300, 400]
    y_pred = [110, 190, 310, 420]
    print("Self-test with dummy data:")
    print(evaluate_all(y_true, y_pred))
