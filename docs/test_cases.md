# Test cases

Group 2, ICT30016 ICT Innovation Project. Test basis: Week 8 work log, Activity 1 (feature table) and Activity 2 (test plan).

Each member writes the test cases for their own features in their own section below. Pull before editing this file and push straight after, so edits do not collide.

## F2 Results loading (FR1, FR2, FR3)

Owner: Elizabeth. Automated run on a second machine: Jessica.

**Expectation:** all data supplied in `results/` is shown as intended. A missing or damaged file shows a warning and the page still loads.

**Quality metric and target:** checks passed out of checks run, and warnings on the current files. All checks pass and there are 0 warnings.

**Testing type:** automated testing, plus a manual look in the browser.

**Code under test:** `src/dashboard_data.py` (data layer) and `src/dashboard.py` (page).

### Before you start

Run every command from the repository root (the `energy-forecasting-group2` folder).

```
git pull
pip install streamlit altair pandas numpy
```

The folder `results/` must hold the four `*_predictions.csv` files, the four `*_results.json` files and `shap_explanations.json`.

### Test cases on the current files

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F2-01 | The real files load and agree with each other | Run `python src/dashboard_data.py` | The last line reads `No warnings: all files line up.` |
| F2-02 | The 35 automated checks | Run `python tests/test_dashboard_data.py` | Every line starts with `OK`. The last line reads `35 checks: 35 passed, 0 failed`. The exit code is 0 (`echo $LASTEXITCODE` in PowerShell prints 0). |
| F2-03 | The page loads with no warning | Run `python -m streamlit run src/dashboard.py`, open http://localhost:8501 and look at the top of the page | The four tabs show. No yellow warning box appears under the title. |
| F2-04 | Every model is on the page | In the browser, open the Model comparison tab | The table has 5 rows: the seasonal-naive baseline plus XGBoost, CatBoost, LSTM and GRU. Each row has a MAE, RMSE and MAPE with its unit. |
| F2-05 | A score on the page equals the score in the file | Open `results/xgboost_results.json` and find the MAE. Compare with the XGBoost MAE in the Model comparison tab. Repeat for one other model. | The two numbers differ by no more than 0.01 MW (the target in the Week 8 feature table). |
| F2-06 | The forecast chart shows the right hours | In the sidebar, set Date range to 5 March 2020 as both start and end, then open the Forecast vs actual tab | The chart covers 00:00 to 23:00 (24 hourly points). Actual demand and each chosen model are drawn. |

### Test cases for a missing or damaged file

These change a file in `results/` on purpose. Do them on your own copy of the repo and put the file back afterwards with `git restore results/<file name>`. Never push the damaged state.

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F2-07 | A missing predictions file | Rename `results/gru_predictions.csv` to `gru_predictions.old`. Reload the page. | The page loads. A warning names GRU. GRU is missing from the charts and the table. The other three models are unchanged. |
| F2-08 | A missing results file | Rename `results/lstm_results.json` to `lstm_results.old`. Reload the page. | The page loads. A warning names LSTM. LSTM is still shown. Its run settings and top features are not available. |
| F2-09 | A damaged predictions file | Open `results/catboost_predictions.csv`, delete the `actual_demand` header and save. Reload the page. | The page loads. A warning says columns are missing. CatBoost is not shown. |
| F2-10 | A model covering fewer hours | Keep only the first 100 data rows of `results/gru_predictions.csv`. Reload the page. | The page loads and a warning says the models cover different hours. The charts only use the hours all models share. |
| F2-11 | An unreadable explanations file | Replace the contents of `results/shap_explanations.json` with `{not valid json`. Reload the page. | The page loads. A warning says the file could not be read. The Why this forecast tab shows no examples instead of an error. |
| F2-12 | No results at all | Rename the `results/` folder to `results_old` and start the dashboard. | The page does not crash with a traceback. A red box reads `Could not load the results:` followed by the reason (no usable prediction files), and a line below says which folder was expected. |
| F2-13 | A score that disagrees with the predictions | In `results/xgboost_results.json` change the XGBoost MAE to 10.0. Reload the page. | A warning says the score does not match the predictions. |

### Which automated checks cover which case

`tests/test_dashboard_data.py` runs F2-07 to F2-13 on temporary copies of `results/`, so nothing in the repo is changed. Run it first. The manual versions above are for seeing the warning as a user would.

| Case | Checks in `test_dashboard_data.py` |
|---|---|
| F2-01, F2-04 | loading all models, scores table builds (5 rows) |
| F2-06 | one-day window has 24 rows, window starts at 00:00 and ends at 23:00, window MAE matches a hand calculation |
| F2-07 | missing GRU predictions file skips GRU and warns |
| F2-08 | missing LSTM results file keeps LSTM and warns, scores table still builds |
| F2-09 | missing column, empty values and repeated timestamps each skip the model with a warning |
| F2-10 | fewer hours limits data to the shared hours and warns, different actual demand warns |
| F2-11 | missing and unreadable explanations file, incomplete explanation skipped |
| F2-12 | empty results folder and missing results folder give a clear error |
| F2-13 | score that disagrees with the predictions warns, explanation that disagrees warns |
| Other | 4 error-band coverage checks, results signature changes on rewrite, 8 readable-name checks, 2 importance-table checks, run-settings text cells, forecast horizon |

### Pass rule for F2

F2 passes when F2-01 to F2-06 pass on the current files, the automated run in F2-02 passes on a second machine (Jessica), and no F2 defect with a high rating is open in Jira.

### Results

Fill in a row each time the cases are run. Leave the row empty until then.

| Date | Tester | Machine | Cases run | Passed | Failed | Jira ticket for each failure |
|---|---|---|---|---|---|---|
| | | | | | | |
| | | | | | | |
