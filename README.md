# energy-forecasting-group2
## Running the dashboard

The dashboard shows the forecasts from all four models (XGBoost, CatBoost, LSTM, GRU) against the actual demand, compares their scores, and explains forecasts in plain English. It reads the files in `results/`, so those files must be present.

Run every command from the repository root (the `energy-forecasting-group2` folder).

1. Get the latest files:

```
   git pull
```

2. Install the libraries (needed once):

```
   pip install streamlit altair pandas numpy
```

3. Check that the results files line up (optional, takes a few seconds). It should end with `No warnings: all files line up.`:

```
   python src/dashboard_data.py
```

4. Start the dashboard:

```
   python -m streamlit run src/dashboard.py
```

   The page opens at http://localhost:8501. If Streamlit asks for an email, press Enter to skip. Press Ctrl+C in the terminal to stop it.

   On Windows, use `python -m streamlit` rather than `streamlit`, because the Scripts folder is usually not on PATH.

### Running the checks

```
python tests/test_dashboard_data.py
```

This runs 35 checks on the dashboard's data layer (src/dashboard_data.py). The last line should read `35 checks: 35 passed, 0 failed`. The checks only read `results/`; anything they damage on purpose is done on a temporary copy.

### If something looks wrong

- A yellow warning at the top of the dashboard means a file in `results/` is missing or does not match the others. The message says which file.
- `ModuleNotFoundError`: run the pip install command in step 2.
- After new files are pushed to `results/`, run `git pull` and refresh the browser page.