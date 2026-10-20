# Test cases

Group 2, ICT30016 ICT Innovation Project. Test basis: Week 8 work log, Activity 1 (feature table) and Activity 2 (test plan).

Each member writes the test cases for their own features in their own section below. Pull before editing this file and push straight after, so edits do not collide.

## Requirements check (FR1 to FR6)

Owner: Alisha. Open the dashboard as described in the README, look at the place in the third column, and write Met, Partly met or Not met in the Result column. Add one line of evidence, such as what you saw or a screenshot name.

| FR | What to check | Where to look | Result | Evidence or note | Tester and date |
|---|---|---|---|---|---|
| FR1 | One table with MAE, RMSE and MAPE for the baseline and all four models | Dashboard, Model comparison tab |Met | All 5 rows has results for MAE, RMSE & MAPE| Alisha 10/10/26 |
| FR2 | One pipeline for cleaning and features that any dataset can pass through without rewriting dataset-specific code | The scripts in src/, ask Jessica and Joud |Partly met |Used only for the primary dataset (code adjustments) |Alisha 10/10/26|
| FR3 | Seed, data split and hyperparameters shown for every model run | Dashboard, About the models tab |Met| Each model displays seed, data & hyperparameters| Alisha 10/10/26|
| FR4 | The main factors behind a forecast written in plain English | Dashboard, Why this forecast tab |Met | Features the reviewer sessions with 5 reviewers| Alisha 11/10/26|
| FR5 | Day-ahead forecast against recent actual demand, with an error band | Dashboard, Forecast vs actual tab (pick one model in the sidebar) | Met |chosen model in sidebar displays forecast vs actual demand | Alisha 11/10/26
| FR6 | Results load without the user running code or installing software | README steps, and the facilitator's answer on local run or hosted |Not met | Requires local installation (tracked via KAN-19)  |Alisha 11/10/26|

## F1 Accuracy and repeatability of the tree model (FR1, FR3)

Jessica owns it. Run again on a different machine: Elizabeth.

**Expectation:** XGBoost and CatBoost make a real day-ahead forecast that is better than the seasonal-naive baseline, and any saved run can be used again from the way it was saved.

**Quality metric and target:** compare the MAE to the seasonal-naive baseline, and see if the recorded score can be reproduced on a second run. Both tree models have an MAE at least 20% lower than the baseline, and a second run on the same computer gives exactly the same score, down to three decimal places.

Types of testing: regression testing and unit testing.

`src/features.py`, `src/train_xgboost.py`, `src/train_catboost.py`, and `src/metrics.py` are the files that are being tested.

### Before you begin

Use the root of the repository to run all of these commands:

```
git pull
pip install -r requirements-training.txt
```

Files in `results/` are changed by training. If you don't want to keep the output, do this on your own copy and then use `git restore results/` to get it back.

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F1-01 | Both tree models beat the seasonal-naive baseline | Run `python src/train_xgboost.py` and `python src/train_catboost.py`. Read the comparison table that's been printed out. | The baseline MAE is 71.651 MW. CatBoost and XGBoost are both at least 20% below it. Last seen, XGBoost had 53.524 MW and CatBoost had 52.748 MW, which is about 25% and 26% lower. |
| F1-02 | The same run happens the same way on the same machine | Run `python src/train_xgboost.py` twice without making any changes to it. Compare the MAE, RMSE and MAPE in `results/xgboost_results.json` between the two runs. | All three of them are the same to three decimal places. You can ignore `TrainingSeconds` because it is expected to be different. |
| F1-03 | Scores on a second machine | Elizabeth runs `python src/train_xgboost.py` on her machine and sends in the MAE. | The MAE is within 0.1 MW of 53.524. An exact match is **not** expected. Before the requirements were pinned, two machines gave 53.524 and 52.918 on xgboost 2.1.4 against 3.2.0. With the versions pinned the gap narrowed to 0.059 MW (53.524 against 53.465) but did not close, most likely because the machines still differ on Python version and operating system. This is why KPI 1 cannot claim exact reproduction across machines. CatBoost has not been run on a second device yet |
| F1-04 | The run configuration is recorded | Open `results/xgboost_results.json` and `results/catboost_results.json` and look for `run_config`. | Each one shows `seed`, `test_year`, `forecast_horizon_hours`, `n_train_rows`, `n_test_rows`, `n_features`, and a `hyperparameters` block with a list of all the values that were given to the model. |
| F1-05 | The forecast is really for the next day | Open `src/features.py`. Make sure to look at `HORIZON` and the `lags` default in `make_features`. Then run `python -c "import sys; sys.path.insert(0,'src'); from features import make_features, load_data; make_features(load_data(), lags=(1, 24))"` | `HORIZON` is 24 and every lag is at least 24. The command throws a `ValueError` saying the lag is shorter than the horizon, so adding a feature that falls inside the forecast gap by accident is not possible. |
| F1-06 | There is only one piece of code that scores each model | Look in `src/train_xgboost.py`, `src/train_catboost.py` and `src/train_sequence.py` for `def mae` or `def rmse`. | No training script sets its own metric functions. All three import from `src/metrics.py`. |
| F1-07 | The saved score matches the saved predictions | Run `python -c "import pandas as pd, json; d=pd.read_csv('results/xgboost_predictions.csv'); print(round((d.actual_demand-d.predicted_demand).abs().mean(),3), round(json.load(open('results/xgboost_results.json'))['results']['XGBoost']['MAE'],3))"` | The two numbers printed are the same to three decimal places. Do it again for CatBoost. |

### F1 pass rule

F1 passes when F1-01, F1-02 and F1-04 to F1-07 pass on my machine, and F1-03 falls within the 0.1 MW tolerance on Elizabeth's. Each failure is raised in Jira with both scores and the machine each one came from.

### F1 results

| Date | Tester | Machine | Cases run | Passed | Failed | Notes |
|---|---|---|---|---|---|---|
| 19/10/2026 | Jessica | MacBook, Python 3.9, local run | F1-01, F1-02, F1-04 to F1-07 | 6 | 0 | F1-01: baseline MAE 71.651, XGBoost 53.524, CatBoost 52.748, 25% and 26% below the baseline. F1-02: two runs had the same MAE to 14 decimals, only difference was `TrainingSeconds` (1.29 then 1.44). F1-04: both results files show all seven `run_config` keys. F1-05: `make_features(lags=(1, 24))` raised `ValueError: Lags [1] are shorter than the 24h forecast horizon`. F1-06: no training script defines its own metrics, all three import from `metrics.py`. F1-07: predictions and results file agree to 3 dp for both models. |
| 20/10/2026 | Elizabeth | Windows laptop, Python 3.11, fresh clone and clean virtual environment | F1-03 | 1 | 0 | XGBoost MAE 53.465, RMSE 75.235, MAPE 4.628%, baseline 71.651 / 100.096 / 6.111. Run twice with identical results, so her machine is self-consistent. The gap to Jessica's 53.524 is 0.059 MW, inside the 0.1 MW tolerance. Both machines ran xgboost 2.1.4, so the remaining difference most likely comes from Python 3.9 on macOS against 3.11 on Windows (the operating system also differs) rather than the library version. Nothing from this run was pushed, so `results/` on main is still Jessica's. |

## F4 reviewer sessions (plain-English explanation)

Owner: Alisha. Target: 4 of 5 reviewers can state the forecast and its main driver without help.

| Reviewer | Date | Could state the forecast? | Could name the main driver? | Where they hesitated |
|---|---|---|---|---|
| 1 |10/10/26 |Yes|Yes| Brief pause on reading SHAP importance scale|
| 2 |10/10/26 |Yes|Yes|None|
| 3 | 11/10/26|Yes|Yes| Took a little bit of time to understand (lag) names |
| 4 |11/10/26 |Yes|Yes| A little bit confused on the terminology|
| 5 |12/10/26 |Yes|Yes|none|

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

## F3 Forecast chart and controls (FR5), LSTM and GRU

Owner: Joud. Repeat run on a second laptop: Elizabeth.

**Expectation:** the chart loads and redraws quickly when the model or date range changes.

**Quality metric and target:** seconds to load and seconds to redraw. Under 5 s to load and under 2 s to redraw on a team laptop.

**Testing type:** performance testing.

### Before you start

Run every command from the repository root. Close other heavy apps. Use a stopwatch (phone is fine). Time three runs for each case and record the slowest.

```
git pull
python -m streamlit run src/dashboard.py
```

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F3-01 | Load time with LSTM | Stop the app and start it again. Start the stopwatch when http://localhost:8501 is opened. Stop it when the Forecast vs actual tab shows the chart with LSTM selected. Repeat 3 times. | Slowest of 3 runs is under 5 s. |
| F3-02 | Load time with GRU | Same as F3-01 with only GRU selected. | Slowest of 3 runs is under 5 s. |
| F3-03 | Redraw when switching to LSTM | With another model shown, select LSTM in the sidebar. Time from the click until the chart has finished redrawing. Repeat 3 times. | Slowest of 3 runs is under 2 s. |
| F3-04 | Redraw when switching to GRU | Same as F3-03 for GRU. | Slowest of 3 runs is under 2 s. |
| F3-05 | Redraw when the date range changes | With LSTM and GRU both selected, change the date range (for example 5 March 2020 to 5 April 2020). Time until the chart has finished redrawing. Repeat 3 times. | Slowest of 3 runs is under 2 s. |

### Pass rule for F3

F3 passes when F3-01 to F3-05 meet their targets on my laptop and Elizabeth's repeat run also meets them. The slowest time is the one recorded.

### F3 repeat run (Elizabeth)

Run on 11 Oct 2026 on my own Windows laptop, local run, with the current results files (including the full shap_explanations.json). Times are in seconds.

| Case | Runs | Slowest | Target | Result |
|---|---|---|---|---|
| F3-01 Load time with LSTM | 1.91, 1.86, 1.75, 1.75, 1.91 | 1.91 | under 5 | Pass |
| F3-02 Load time with GRU | Same measurement as F3-01: the page always opens with all four models selected | 1.91 | under 5 | Pass |
| F3-03 Redraw when switching to LSTM | 0.63, 0.56, 0.43, 0.44, 0.51 | 0.63 | under 2 | Pass |
| F3-04 Redraw when switching to GRU | 0.60, 0.60, 0.46, 0.64, 0.61 | 0.64 | under 2 | Pass |
| F3-05 Redraw when the date range changes (5 Mar to 5 Apr 2020) | 0.73, 0.75, 0.83 | 0.83 | under 2 | Pass |

F3 passes on both laptops. The slowest times recorded are Joud's: 2.04 s to load and 0.93 s to redraw.

Re-run on 20 Oct 2026 after the page started opening on the full test period (1 Jan to 27 Jun 2020). The first start after the laptop had been idle took 12.96 s to load, and the five starts after it took 3.96, 2.20, 2.23, 2.61 and 1.24 s. Redraw, slowest of 3: LSTM 1.51 s, GRU 1.60 s, date range 1.36 s. All redraws are under 2 s, but about twice the 11 Oct times (0.43 to 0.83 s) because the chart now draws six months of hours by default.

## F5 Setup from a fresh clone (FR6), LSTM and GRU

Owner: Joud. Observer: Elizabeth.

**Expectation:** someone outside the team can get the dashboard running from the README alone.

**Quality metric and target:** commands needed and errors met. 3 commands or fewer and no errors.

**Testing type:** system testing.

**Open point:** FR6 says no installation. Alisha is asking the facilitator whether a local run is accepted or the dashboard must be hosted. If hosting is required, F5-01 and F5-02 change to open the hosted link.

### Before you start

Use a different machine or a new empty folder, with no earlier copy of the repo. Open only the README. Do not use any other notes.

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F5-01 | A fresh clone runs from the README | Clone the repo into a new folder. Follow the README steps in order, typing only what it says. Count each command. | The dashboard opens in the browser using 3 commands or fewer. |
| F5-02 | No errors met | During F5-01, note every error message or step that did not work. | 0 errors. Any error is written down with the step where it happened. |
| F5-03 | LSTM and GRU are on the page | Open the Model comparison tab. | LSTM and GRU rows are shown with MAE, RMSE and MAPE. |
| F5-04 | The README warnings are accurate | Follow the Troubleshooting lines in the README (yellow warning, `ModuleNotFoundError`, `git pull`). | Each line describes what actually happens. |

### Pass rule for F5

F5 passes when F5-01 to F5-03 pass on a machine that has not run the project before, with Elizabeth watching. Any step that fails or takes more than 3 commands is sent to Elizabeth so she can fix the README.

## F6 End-to-end pipeline (FR1, FR2, FR3), LSTM and GRU

Owner: Joud for the LSTM and GRU stages. Tree models: Jessica. Dashboard reload: Elizabeth.

**Expectation:** each stage uses the last stage's output with no manual edits: cleaning, features, four models, results files, dashboard.

**Quality metric and target:** stages that run in order from a fresh clone, and models that record seed, split and hyperparameters. 5 of 5 stages and 4 of 4 models.

**Testing type:** integration testing.

### Before you start

Training writes to `results/`. Do this on your own copy and put the files back afterwards with `git restore results/`. Do not push changed results unless the team agrees.

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F6-01 | The LSTM stage runs from the features stage | From a fresh clone, run `python3 src/train_sequence.py --model lstm`. Training is slow: the last recorded run took about 2,040 s (34 minutes). | It finishes with no error and rewrites `results/lstm_predictions.csv` and `results/lstm_results.json`. No file was edited by hand. The MAE should match the previous run (seed 42, exact on CPU). |
| F6-02 | The GRU stage runs the same way | Run `python3 src/train_sequence.py --model gru`. | It finishes with no error and rewrites `results/gru_predictions.csv` and `results/gru_results.json`. No file was edited by hand. |
| F6-03 | LSTM and GRU record seed, split and hyperparameters | Open `results/lstm_results.json` and `results/gru_results.json` and look under `run_config`. | Each file shows `seed`, `test_year`, `lookback_hours`, `n_train_rows`, `n_test_rows` and a `hyperparameters` block. |
| F6-04 | The predictions line up with the other models | Open `results/lstm_predictions.csv` and `results/gru_predictions.csv`. Compare with `results/xgboost_predictions.csv`. | Same columns (`datetime`, `actual_demand`, `predicted_demand`, `model_name`) and the same hours. |
| F6-05 | The dashboard uses the new output | Reload the dashboard after F6-01 and F6-02. | The LSTM and GRU numbers and the Run settings table match the new results files. No warning appears. |

### Pass rule for F6

F6 passes when F6-01 to F6-05 pass for LSTM and GRU, so the team reaches 4 of 4 models recording seed, split and hyperparameters together with Jessica's tree models.

### Tree model stages: XGBoost and CatBoost

Jessica is the owner for the XGBoost, CatBoost and explanation stages. Joud covers the LSTM and GRU stages in F6-01 to F6-05 above. These follow on from Joud's numbers so the F6 pass rule covers all four models.

| ID | What is checked | Steps | Expected result |
|---|---|---|---|
| F6-06 | The XGBoost stage runs after the features stage | Run `python src/train_xgboost.py` from a fresh copy. It only takes seconds to train. | It finishes without any errors and writes `results/xgboost_predictions.csv`, `results/xgboost_results.json` and `models/xgboost.json`. No files were changed by hand. |
| F6-07 | The CatBoost stage runs the same way | Run `python src/train_catboost.py`. | It finishes without any errors and writes `results/catboost_predictions.csv`, `results/catboost_results.json` and `models/catboost.cbm`. |
| F6-08 | Both tree models record seed, split and hyperparameters | Open both results files and look under `run_config`. | Each one shows `seed`, `test_year`, `forecast_horizon_hours`, `n_train_rows`, `n_test_rows`, `n_features` and a `hyperparameters` block. Put together with F6-03, this makes 4 of 4 models. |
| F6-09 | The predictions line up with the other models | Compare `results/xgboost_predictions.csv` and `results/catboost_predictions.csv` with the LSTM and GRU files. | Same columns (`datetime`, `actual_demand`, `predicted_demand`, `model_name`), 4,273 rows and the same hours in all four. |
| F6-10 | The explanation stage starts from the saved model | After F6-06, run `python src/explain_forecast.py`. | It finishes without any errors and rewrites `results/shap_explanations.json` and `results/xgboost_drivers.png`. There is a record for every test hour, not just the three named examples. Check with `python -c "import json; print(len(json.load(open('results/shap_explanations.json'))['examples']))"`, which prints 4276, being three named examples plus 4,273 hours. |
| F6-11 | The explanations match the predictions | Run `python src/dashboard_data.py`. | The last line says `No warnings: all files line up.` A mismatch means the explanations and predictions came from different training runs, so both scripts need running again in the right order. |
| F6-12 | The dashboard uses the new output | Reload the dashboard after F6-06, F6-07 and F6-10. | The XGBoost and CatBoost numbers and the Run settings table match the new results files. The Why this forecast tab can explain any hour in the date range, not just three. No warning appears. |
| F6-13 | The CatBoost driver chart starts from the saved model | After F6-07, run `python src/catboost_drivers.py`. Then move `models/catboost.cbm` somewhere else and run it again. | It finishes without any errors and rewrites `results/catboost_drivers.png`. It prints `CatBoost: average effect on the forecast (MW)` followed by 16 grouped drivers, the largest being demand at this hour last week (38.6 MW). It does not change `results/catboost_predictions.csv`, `results/catboost_results.json` or `results/shap_explanations.json`. Without `models/catboost.cbm` it stops with the message `No saved model` and tells you to run `python src/train_catboost.py` first. |

#### Pass rule for the tree model stages

F6-06 to F6-13 pass on a fresh copy with no changes made by hand to any file in between, and the dashboard reload in F6-12 shows no warning.

#### Note on the order of the stages

`explain_forecast.py` and `catboost_drivers.py` read the saved models in `models/`, which git ignores, so they are never in a new copy. Each one has to run after its own training script, and the explanation has to come from the same training run as the model it explains. Running the explanation stage against an older model gives explanations whose forecast values don't match `xgboost_predictions.csv`, which is what F6-11 is there to catch.

#### Results for the tree model stages

| Date | Tester | Machine | Cases run | Passed | Failed | Notes |
|---|---|---|---|---|---|---|
| 19/10/2026 | Jessica | MacBook, Python 3.9, local run | F6-06 to F6-12 | 7 | 0 | F6-06 and F6-07 both trained with no errors and wrote their predictions, results and model files. F6-08: both `run_config` blocks hold all seven keys, which with F6-03 makes 4 of 4 models. F6-09: all four prediction files have the same columns, 4,273 rows and identical hours. F6-10: `explain_forecast.py` wrote 4,276 entries, three named examples plus every test hour. F6-11: `dashboard_data.py` said `No warnings: all files line up.` F6-12: dashboard reloaded clean and the Why this forecast tab resolved hours outside the three named examples. |
| 20/10/2026 | Elizabeth | Windows laptop, Python 3.11, local run after git pull | F6-11 and F6-12 | 2 | 0 | F6-11: `dashboard_data.py` said `No warnings: all files line up.` and the 35 checks gave `35 checks: 35 passed, 0 failed`. F6-12: the dashboard opened with no warning, the Model comparison scores matched the results files (XGBoost 53.52, CatBoost 52.75, LSTM 117.09, GRU 120.12), the Run settings table showed all four models, and Look up any hour explained an hour outside the three named examples. |
| 20/10/2026 | Jessica | MacBook, Python 3.9, local run | F6-13 | 1 | 0 | `catboost_drivers.py` ran with no errors and rewrote `results/catboost_drivers.png`. It printed `CatBoost: average effect on the forecast (MW)` followed by 16 grouped drivers, largest demand at this hour last week at 38.6 MW. Checked by MD5 before and after that it left `catboost_predictions.csv`, `catboost_results.json` and `shap_explanations.json` unchanged. With `models/catboost.cbm` moved aside it stopped with `No saved model at .../models/catboost.cbm. Run python src/train_catboost.py first.` and wrote nothing. |
### Results for F3, F5 and F6

Fill in a row each time the cases are run. Leave the rows empty until then.

| Date | Tester | Machine | Cases run | Slowest time or result | Passed | Failed | Jira ticket for each failure |
|---|---|---|---|---|---|---|---|
| 11/10/2026 | Joud | MacBook Air, local run | F3-01 to F3-05 | Load 2.04 s, redraw 0.93 s (slowest of 3) | 5 | 0 | none |
| 11/10/2026 | Joud | MacBook Air, fresh clone on Desktop | F5-01 to F5-03 | 8 commands, 3 errors (pip and python not found; python3 and pip3 needed on Mac). F5-03 passed. | 1 | 2 | KAN-19 |
| 12/10/2026 | Joud | MacBook Air, fresh clone on Desktop (re-run after README update) | F5-01 to F5-03 | README followed exactly: 3 commands (clone, install, run), 0 errors. LSTM and GRU shown on Model comparison tab. | 3 | 0 | KAN-19 |
| 20/10/2026 | Elizabeth | Windows laptop, local run, page opening on the full test period | F3-01 to F3-05 | Load: 12.96 s on the first start after the laptop had been idle, then 3.96, 2.20, 2.23, 2.61 and 1.24 s. Redraw, slowest of 3: LSTM 1.51 s, GRU 1.60 s, date range 1.36 s. | 5 | 0 | none |
| 20/10/2026 | Elizabeth | Windows laptop, local run | F5-04 | Line 1 (`command not found` on macOS): not testable on Windows, matches Joud's run on 11/10 where python and pip were not found on the Mac. Line 2 (`ModuleNotFoundError`): in an empty Python environment `dashboard_data.py` printed `ModuleNotFoundError: No module named 'numpy'`, while `python -m streamlit run` printed `No module named streamlit`; running the install command again fixed it. README line reworded to cover both messages. Line 3 (yellow warning): with `gru_predictions.csv` renamed the page showed `GRU is not shown: gru_predictions.csv not found.` and GRU left the table. Line 4 (`git pull`): confirmed on 11/10, when Jessica's updated results appeared after `git pull` and a page refresh. | 1 | 0 | none |

## F2 Results

Fill in a row each time the cases are run. Leave the row empty until then.

| Date | Tester | Machine | Cases run | Passed | Failed | Jira ticket for each failure |
|---|---|---|---|---|---|---|
| 6/10/2026 | Elizabeth | Windows laptop, local run | F2-01 to F2-06 | 5 | 1 | F2-05: scores were shown to 1 decimal, so the gap to the results file could reach 0.05 MW. Fixed by showing 2 decimals. |
| 6/10/2026 | Elizabeth | Windows laptop, local run | F2-02 and F2-05, re-run after the fix | 2 | 0 | none |
| 11/10/2026 | Jessica | MacBook, local run | F2-02 | 1 | 0 | none |
| 11/10/2026 | Elizabeth | Windows laptop, local run | F2-01 and F2-02, re-run after Jessica's update to results/ | 2 | 0 | none |
| 18/10/2026 | Elizabeth | Windows laptop | F2-05 re-run after default date range change | 1 | 0 |(found by Jessica: default view showed 54.24, fixed to 53.52) |