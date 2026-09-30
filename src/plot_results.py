"""
Plot the model comparison chart from the saved results.
Owner: Jessica Miyumo

This is FR5. It does not retrain anything, it just reads the JSON files the training scripts write into results/ and turns them into one chart I can put in the work log and the report. The chart always matches whatever the last saved run was, so I cannot accidentally publish a chart of numbers I can no longer reproduce.
Run the training scripts first, then:
    python src/plot_results.py  


Run the training scripts first, then:
    python src/plot_results.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # until a gui is added, so it does not need a display to run
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
OUTPUT_PATH = RESULTS_DIR / "model_comparison.png"

#made the colours match the dashboard, so the same model is always the same colour across all outputs
COLOURS = {
    "Seasonal-naive baseline": "#2a78d6",# blue
    "XGBoost": "#eb6834",# orange
    "CatBoost": "#2bbe8a", # aquamarine
    "LSTM": "#eda100",# yellow
    "GRU": "#e87ba4",# pink
}
FALLBACK_COLOUR = "#898781"

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def load_results():
    """Read every *_results.json in results/ and pull out the scores."""
    scores = {}
    baselines = {}

    for path in sorted(RESULTS_DIR.glob("*_results.json")):
        payload = json.loads(path.read_text())
        for name, metrics in payload["results"].items():
            if name == "Seasonal-naive baseline":
                # Kept apart and checked below, not silently overwritten
                baselines[payload["model"]] = metrics
            else:
                scores[name] = metrics

    if not scores:
        raise SystemExit(
            "No model results found in results/. Run the training scripts first."
        )

    # Calculating the baseline in two different scripts should give the same result, or the comparison is not fair. Check that here rather than silently using one and ignoring the other.
    unique = {round(m["MAE"], 6) for m in baselines.values()}
    if len(unique) > 1:
        raise SystemExit(
            f"The seasonal-naive baseline does not match across scripts: {baselines}. "
            "Check that both are using the same test year and the same dataset."
        )

    # Dashboard shows the baseline first, then the models in a preferred order, then any other models that happen to be in results/. The preferred order is the same as the dashboard.
    preferred = ["XGBoost", "CatBoost", "LSTM", "GRU"]
    ordered = {"Seasonal-naive baseline": next(iter(baselines.values()))}
    for name in preferred:
        if name in scores:
            ordered[name] = scores.pop(name)
    ordered.update(scores)
    return ordered


def draw_group(ax, models, metrics, results, title, ylabel, value_fmt):
    """Create one of the two panels in the model comparison chart, with a group of bars for each metric."""
    x = np.arange(len(metrics))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
    
        offset = (i - (len(models) - 1) / 2) * width
        values = [results[model][m] for m in metrics]
        bars = ax.bar(x + offset, values, width * 0.92, label=model,
                      color=COLOURS.get(model, FALLBACK_COLOUR), zorder=3)

        # Values on every bar, so the numbers are readable and it survives black and white
        for bar, value in zip(bars, values):
            ax.annotate(value_fmt.format(value),
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left")
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.set_xticks(x, metrics, fontsize=9.5, color=INK)
    ax.tick_params(axis="y", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0)

    # Recessive grid. The bars are the point.
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")

  
    ax.set_ylim(0, max(results[m][k] for m in models for k in metrics) * 1.18)


def main():
    results = load_results()
    models = list(results.keys())

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(10, 4.4), gridspec_kw={"width_ratios": [2, 1]}
    )
    fig.patch.set_facecolor("#fcfcfb")
    for ax in (ax_left, ax_right):
        ax.set_facecolor("#fcfcfb")

    #MAE and RMSE are in MW, MAPE is a percentage. The dashboard shows the same two panels, so the labels match.
    draw_group(ax_left, models, ["MAE", "RMSE"], results,
               "Forecast error (MW, lower is better)", "MW", "{:.1f}")
    draw_group(ax_right, models, ["MAPE (%)"], results,
               "Error as a percentage", "%", "{:.2f}%")

    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models),
               frameon=False, fontsize=9.5, labelcolor=INK,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Model comparison on the 2020 test set", fontsize=13,
                 color=INK, x=0.011, ha="left", y=0.98)
    fig.text(0.011, 0.9,
             "All models scored with the shared metrics in src/metrics.py, "
             "on the same chronological train/test split.",
             fontsize=8.5, color=MUTED, ha="left")

    fig.tight_layout(rect=[0, 0.06, 1, 0.88])
    fig.savefig(OUTPUT_PATH, dpi=200, facecolor=fig.get_facecolor())
    print(f"Saved chart to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()