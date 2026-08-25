# =============================================================
# Data Cleaning — Primary Dataset (Electricity Load Forecasting)
# Role: Data & Platform Engineer
# Input : data/primary-dataset/continuous_dataset.csv
#         data/primary-dataset/weekly_pre-dispatch_forecast.csv
# Output: data/processed/cleaned_dataset.csv
# =============================================================

import pandas as pd
import numpy as np

# -----------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------
main_path = "data/primary-dataset/continuous dataset.csv"
forecast_path = "data/primary-dataset/weekly pre-dispatch forecast.csv"

df = pd.read_csv(main_path)
forecast = pd.read_csv(forecast_path, encoding="utf-8-sig")  # file has a BOM

print("Raw shape:", df.shape)
print("Forecast shape:", forecast.shape)

# -----------------------------------------------------------
# 2. Parse datetime columns
# -----------------------------------------------------------
df["datetime"] = pd.to_datetime(df["datetime"])
forecast["datetime"] = pd.to_datetime(forecast["datetime"])

# -----------------------------------------------------------
# 3. Check & handle duplicates
# -----------------------------------------------------------
dup_count = df["datetime"].duplicated().sum()
print("Duplicate timestamps in main data:", dup_count)
if dup_count > 0:
    df = df.drop_duplicates(subset="datetime", keep="first")

# -----------------------------------------------------------
# 4. Check for missing hourly timestamps (gaps in the series)
# -----------------------------------------------------------
full_range = pd.date_range(df["datetime"].min(), df["datetime"].max(), freq="h")
missing_ts = full_range.difference(df["datetime"])
print("Missing hourly timestamps:", len(missing_ts))

if len(missing_ts) > 0:
    # Reindex to the full hourly range so gaps become visible NaN rows,
    # then interpolate numeric columns linearly (safe for short gaps).
    df = df.set_index("datetime").reindex(full_range)
    df.index.name = "datetime"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=6)
    df = df.reset_index()

# -----------------------------------------------------------
# 5. Check remaining nulls
# -----------------------------------------------------------
null_counts = df.isnull().sum()
print("Nulls per column after interpolation:\n", null_counts[null_counts > 0])
# Drop any row still null after interpolation (e.g. gap longer than limit=6h)
df = df.dropna()

# -----------------------------------------------------------
# 6. Outlier check on target (nat_demand) — IQR method, flag only
# -----------------------------------------------------------
q1, q3 = df["nat_demand"].quantile([0.25, 0.75])
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
df["nat_demand_outlier_flag"] = ((df["nat_demand"] < lower) | (df["nat_demand"] > upper)).astype(int)
print("nat_demand outliers flagged:", df["nat_demand_outlier_flag"].sum(),
      f"(bounds: {lower:.1f} - {upper:.1f})")
# NOTE: outliers are flagged, not removed — Jessica/Joud can decide whether
# to exclude them when training, since some may be genuine demand spikes
# (e.g. holidays) rather than data errors.

# -----------------------------------------------------------
# 7. Standardise column names (lowercase, no spaces)
# -----------------------------------------------------------
df.columns = [c.strip().lower() for c in df.columns]

# -----------------------------------------------------------
# 8. Sort chronologically and reset index
# -----------------------------------------------------------
df = df.sort_values("datetime").reset_index(drop=True)

# -----------------------------------------------------------
# 9. Basic sanity checks before saving
# -----------------------------------------------------------
assert df["datetime"].is_monotonic_increasing, "Datetime not sorted!"
assert df["datetime"].duplicated().sum() == 0, "Duplicates remain!"
assert df.isnull().sum().sum() == 0, "Nulls remain!"

print("\nFinal cleaned shape:", df.shape)
print("Date range:", df["datetime"].min(), "->", df["datetime"].max())

# -----------------------------------------------------------
# 10. Save cleaned output
# -----------------------------------------------------------
import os
os.makedirs("data/processed", exist_ok=True)
out_path = "data/processed/cleaned_dataset.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved cleaned dataset to {out_path}")