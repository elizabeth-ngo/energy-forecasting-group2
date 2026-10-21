# Data Pipeline — README

Owner: Elizabeth (Data & Platform Engineer)

This document explains how to reproduce the data cleaning pipeline for the
primary dataset (Electricity Load Forecasting). Anyone on the team should be
able to follow these steps and get the exact same output.

## 1. Requirements

- Python 3.10+
- Packages: `pandas`, `numpy`

Install packages (only needs to be done once per machine):

```powershell
pip install pandas numpy
```

## 2. Folder structure expected

This script assumes you are standing at the **repo root** when you run it,
and that the raw files are already in place:

```
energy-forecasting-group2/
├── data/
│   ├── primary-dataset/
│   │   ├── continuous dataset.csv
│   │   └── weekly pre-dispatch forecast.csv
│   └── processed/              <- output goes here (auto-created)
├── src/
│   ├── 01_data_cleaning.py
│   ├── metrics.py
│   └── data_dictionary.md
```

## 3. How to run

From the repo root, in the VS Code terminal (PowerShell):

```powershell
cd C:\Users\Admin\energy-forecasting-group2
python src/01_data_cleaning.py
```

## 4. What the script does

1. Loads the two raw primary-dataset CSVs.
2. Parses the `datetime` column.
3. Checks and removes duplicate timestamps (if any).
4. Checks for missing hourly timestamps (gaps) and interpolates gaps up to
   6 hours long. Rows with longer gaps are dropped.
5. Flags outliers in `nat_demand` using the IQR method — **outliers are
   flagged, not removed**, since some may be genuine demand spikes (e.g.
   public holidays) rather than data errors. Modeling leads can decide
   whether to exclude them.
6. Standardises column names (lowercase, no extra spaces).
7. Sorts rows chronologically and runs sanity checks (no duplicates, no
   nulls, sorted order) before saving.
8. Saves the result to `data/processed/cleaned_dataset.csv`.

## 5. Output

- `data/processed/cleaned_dataset.csv` — cleaned primary dataset, same
  columns as the raw file plus `nat_demand_outlier_flag` (0 = normal,
  1 = flagged outlier).

## 6. Reproducibility notes

- The script only uses relative paths (`data/...`), so it works on any
  machine as long as you run it from the repo root.
- No manual/GUI steps are involved — everything is scripted.
- Running the script twice on the same input produces the same output.

## 7. Related files

- `src/data_dictionary.md` — explains what each column in the cleaned
  dataset means.
- `src/metrics.py` — shared evaluation metric functions (MAE, RMSE, MAPE)
  for use by the modeling leads (Jessica, Joud).
