"""
Print the top SHAP drivers for CatBoost.

It only works if you have already trained the CatBoost model, because it needs the saved model file to load.
It oly writes its own chart file, not the predictions or the scores, because those are already written by the training script.
Run the training script first :
    python src/train_catboost.py
    python src/catboost_drivers.py
"""

import sys
from pathlib import Path
 
import shap
from catboost import CatBoostRegressor
 
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import load_data, make_features, chronological_split
from explain_forecast import grouped_shap, plot_global_drivers
 
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "models" / "catboost.cbm"
CHART_PATH = REPO_ROOT / "results" / "catboost_drivers.png"
 
if not MODEL_PATH.exists():
    raise SystemExit(f"No saved model at {MODEL_PATH}. Run python src/train_catboost.py first.")
 
df = load_data()
data, feature_cols = make_features(df)
_, test = chronological_split(data)
 
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
 
# not sampled from the test set, because the dashboard needs to be able to look up any hour. SHAP is fast enough on this dataset.
shap_values = shap.TreeExplainer(model).shap_values(test[feature_cols])
grouped = grouped_shap(shap_values, feature_cols)
 
importance = plot_global_drivers(grouped, CHART_PATH, "CatBoost")
 
print("\nCatBoost: average effect on the forecast (MW)")
print(importance.sort_values(ascending=False).round(1).to_string())
