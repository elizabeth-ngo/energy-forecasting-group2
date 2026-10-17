# energy-forecasting-group2
## Running the dashboard

The dashboard shows the forecasts from all four models (XGBoost, CatBoost, LSTM, GRU) against the actual demand, compares their scores, and explains forecasts in plain English. It reads the files in `results/`, so those files must be present.

You need Python 3 (the project was tested with 3.9 and 3.11). Run every command from the repository root.

1. Get the code. First time only:

```
   git clone https://github.com/elizabeth-ngo/energy-forecasting-group2.git
   cd energy-forecasting-group2
```

   Already have it? Run `git pull` instead.

2. Install the libraries (needed once).

   Windows:

```
      pip install -r requirements.txt
```

   macOS and Linux:

```
      python3 -m pip install -r requirements.txt
```

3. Start the dashboard.

   Windows:

```
   python -m streamlit run src/dashboard.py
```

   macOS and Linux:

```
   python3 -m streamlit run src/dashboard.py
```

   The page opens at http://localhost:8501. If Streamlit asks for an email, press Enter to skip. Press Ctrl+C in the terminal to stop it.

### Optional checks

Use `python3` instead of `python` on macOS and Linux.

```
python src/dashboard_data.py
python tests/test_dashboard_data.py
```

The first should end with `No warnings: all files line up.` and the second with `35 checks: 35 passed, 0 failed`. The checks only read `results/`; anything they damage on purpose is done on a temporary copy.

### If something looks wrong

- `command not found` for `python` or `pip` (common on macOS): use `python3` and `python3 -m pip`.
- `ModuleNotFoundError`: run the install command in step 2 again.
- A yellow warning at the top of the dashboard means a file in `results/` is missing or does not match the others. The message names the file.
- After new files are pushed to `results/`, run `git pull` and refresh the browser page.