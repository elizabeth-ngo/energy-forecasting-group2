"""
Energy demand forecasting dashboard (Streamlit).

Run from the repository root:

    pip install streamlit
    python -m streamlit run src/dashboard.py

Optional check that the result files load, before launching:

    python src/dashboard_data.py

The page reads everything from results/ (predictions, scores, explanations).
When a model is re-run and its files change, reload the page to see the update.
"""

import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import altair as alt
import pandas as pd
import streamlit as st

import dashboard_data as dd

st.set_page_config(page_title="Energy demand forecast", layout="wide")
alt.data_transformers.disable_max_rows()


@st.cache_data(show_spinner=False)
def get_data(signature):
    # `signature` is only here so the cache refreshes when result files change.
    return dd.load_all()


# ---------------------------------------------------------------- charts

def color_scale(names):
    return alt.Scale(domain=list(names), range=[dd.COLORS.get(n, "#888888") for n in names])


def forecast_chart(window, models, band=None, band_model=None):
    long = dd.chart_frame(window, models)
    names = [dd.ACTUAL] + list(models)
    lines = (
        alt.Chart(long)
        .mark_line(strokeWidth=1.6)
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("demand_mw:Q", title="Demand (MW)", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "series:N", scale=color_scale(names), legend=alt.Legend(title=None, orient="bottom")
            ),
            tooltip=[
                alt.Tooltip("datetime:T", title="Time", format="%a %d %b, %H:%M"),
                alt.Tooltip("series:N", title="Series"),
                alt.Tooltip("demand_mw:Q", title="MW", format=",.0f"),
            ],
        )
        .interactive()
    )
    if band is None:
        return lines.properties(height=380, width="container")
    area = (
        alt.Chart(band)
        .mark_area(opacity=0.18, color=dd.COLORS.get(band_model, "#888888"))
        .encode(
            x=alt.X("datetime:T", title=None),
            y=alt.Y("lower:Q", title="Demand (MW)", scale=alt.Scale(zero=False)),
            y2=alt.Y2("upper:Q"),
        )
    )
    return alt.layer(area, lines).properties(height=380, width="container")


def score_chart(scores, metric):
    df = scores[["Model", metric]].rename(columns={metric: "value"})
    names = list(df["Model"])
    fmt = ",.2f" if "%" in metric else ",.1f"
    base = alt.Chart(df).encode(
        x=alt.X("Model:N", sort=names, title=None, axis=alt.Axis(labelAngle=0, labelLimit=260)),
        y=alt.Y("value:Q", title=metric),
    )
    bars = base.mark_bar().encode(
        color=alt.Color("Model:N", scale=color_scale(names), legend=None),
        tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("value:Q", title=metric, format=",.2f")],
    )
    labels = base.mark_text(dy=-8).encode(text=alt.Text("value:Q", format=fmt))
    return alt.layer(bars, labels).properties(height=340, width="container")


def horizontal_bars(df, label_col, value_col, x_title):
    return (
        alt.Chart(df)
        .mark_bar(color="#2b78d6")
        .encode(
            x=alt.X(f"{value_col}:Q", title=x_title),
            y=alt.Y(f"{label_col}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=340, labelOverlap=False)),
            tooltip=[
                alt.Tooltip(f"{label_col}:N", title="Factor"),
                alt.Tooltip(f"{value_col}:Q", title=x_title, format=",.2f"),
            ],
        )
        .properties(height=max(200, 30 * len(df)), width="container")
    )


# ---------------------------------------------------------------- tabs

def render_explanation(ex):
    st.markdown(ex["headline"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast (XGBoost)", f"{ex['forecast_mw']:,.0f} MW")
    c2.metric("Actual", f"{ex['actual_mw']:,.0f} MW")
    c3.metric("Forecast minus actual", f"{ex['forecast_mw'] - ex['actual_mw']:+,.0f} MW")
    st.markdown("What pushed the forecast up or down:")
    for sentence, effect in ex["drivers"]:
        arrow = "▲" if effect > 0 else "▼"
        st.markdown(f"- {arrow} {sentence}")
    st.caption(
        f"The {len(ex['drivers'])} largest factors are shown. An explanation says why the model "
        "produced this number, not whether the number was right."
    )


def tab_forecast(data, selected, start, end):
    window = dd.filter_window(data.wide, start, end)
    if window.empty:
        st.info("There is no data in this date range. Try a different range in the sidebar.")
        return

    band = None
    band_model = None
    if len(selected) == 1:
        band_model = selected[0]
        if st.checkbox("Show typical error range (80% of hours)", value=True, key="show_band"):
            band = dd.band_frame(window, band_model, dd.error_band(data.wide, band_model))
    else:
        st.caption("Select a single model in the sidebar to see its typical error range.")

    st.altair_chart(forecast_chart(window, selected, band, band_model))
    st.caption("Hover over a line for exact values. Drag to pan and scroll to zoom.")
    if band is not None:
        st.caption(
            "Shaded band: in 80% of test hours the actual demand landed inside this range around "
            "the forecast. It is based on the errors this model made over the whole test period, "
            "so it describes past performance, not a guarantee."
        )

    st.subheader("Accuracy over the selected period")
    st.dataframe(dd.format_scores(dd.window_scores(window, selected)), hide_index=True)
    st.caption(
        f"Based on {len(window):,} {'hour' if len(window) == 1 else 'hours'}. Lower is better. "
        "MAE is the average gap between the forecast and actual demand, in MW."
    )
    if len(window) < 48:
        st.caption(
            "This is a short period, so scores can swing a lot. Scores for the full test period "
            "are in the Model comparison tab."
        )


def tab_compare(data):
    scores = dd.scores_table(data)
    st.subheader("How the models compare")
    st.caption("Scores cover every hour of the test period. Lower is better for all three measures.")
    for line in dd.comparison_summary(scores):
        st.markdown(f"- {line}")
    st.dataframe(dd.format_scores(scores), hide_index=True)
    st.caption(
        "Error vs baseline: a minus sign means lower error than the baseline, a plus sign means higher. "
        "The baseline simply repeats demand from the same hour one week earlier."
    )
    metric = st.radio("Show chart for", ["MAE (MW)", "RMSE (MW)", "MAPE (%)"], horizontal=True, key="metric")
    st.altair_chart(score_chart(scores, metric))
    with st.expander("What the scores mean"):
        st.markdown(
            "- MAE: the average gap between forecast and actual demand, in MW.\n"
            "- RMSE: like MAE, but large misses count for more, so it is higher when a model has a few big errors.\n"
            "- MAPE: the average gap as a percentage of actual demand."
        )


def tab_why(data):
    st.subheader("Why the model forecasts what it does")
    st.caption(
        "These explanations come from the XGBoost model. They show which factors pushed a forecast "
        "up or down. They describe the model's reasoning, not what causes demand in the real world."
    )
    drivers = dd.driver_table(data.shap)
    if drivers.empty:
        st.info("No driver information found in results/shap_explanations.json.")
    else:
        st.markdown("What drives the forecast on average")
        st.altair_chart(horizontal_bars(drivers, "driver", "effect_mw", "Average effect on the forecast (MW)"))
        st.caption("A longer bar means that factor moves the forecast by more MW on average.")

    st.markdown("Explained forecasts")
    if not data.examples:
        st.info("No explained forecasts found yet.")
    else:
        labels = [e["label"] for e in data.examples]
        choice = st.selectbox("Example hour", labels, key="example_choice")
        render_explanation(next(e for e in data.examples if e["label"] == choice))

    with st.expander("Look up any hour"):
        first = data.wide["datetime"].min().date()
        last = data.wide["datetime"].max().date()
        default_day = data.examples[0]["timestamp"].date() if data.examples else first
        day = st.date_input("Day", value=default_day, min_value=first, max_value=last, key="lookup_day")
        hour = st.selectbox("Hour", list(range(24)), format_func=lambda h: f"{h:02d}:00", key="lookup_hour")
        ts = pd.Timestamp(day) + pd.Timedelta(hours=int(hour))
        row = dd.forecast_row(data.wide, ts)
        if row is None:
            st.info(f"There is no forecast for {ts:%d %b %Y, %H:%M}. The data ends at {data.wide['datetime'].max():%d %b %Y, %H:%M}.")
        elif "XGBoost" not in data.wide.columns:
            st.info("XGBoost results are not loaded, so there is nothing to look up.")
        else:
            example = dd.find_example(data.examples, ts)
            if example is not None:
                render_explanation(example)
            else:
                forecast, actual = float(row["XGBoost"]), float(row["actual_demand"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Forecast (XGBoost)", f"{forecast:,.0f} MW")
                c2.metric("Actual", f"{actual:,.0f} MW")
                c3.metric("Forecast minus actual", f"{forecast - actual:+,.0f} MW")
                st.caption(
                    "A written explanation is not available for this hour yet. Explanations currently "
                    f"cover {len(data.examples)} example hours."
                )


def tab_about(data):
    st.subheader("Run settings")
    if data.results:
        st.dataframe(dd.run_config_table(data.results), hide_index=True)
    else:
        st.info("No results files found, so run settings are unavailable.")
    with st.expander("Model hyperparameters"):
        for model, res in data.results.items():
            params = res.get("run_config", {}).get("hyperparameters")
            if params:
                st.markdown(model)
                st.json(params)

    with_features = [(m, dd.top_features_table(r)) for m, r in data.results.items()]
    with_features = [(m, df) for m, df in with_features if not df.empty]
    if with_features:
        st.subheader("Most important inputs")
        st.caption(
            "Each model reports importance on its own scale, so compare the order within a model, "
            "not the numbers across models."
        )
        cols = st.columns(len(with_features))
        for col, (model, df) in zip(cols, with_features):
            with col:
                st.markdown(model)
                st.altair_chart(horizontal_bars(df, "feature", "importance", "Importance (model's own scale)"))

    st.subheader("Where the data comes from")
    st.markdown(
        "Hourly national electricity demand and weather readings from the team's cleaned primary dataset "
        "(prepared in src/01_data_cleaning.py). Models were trained on earlier data and tested on later "
        "data in time order. Scores and forecasts here come from the files in the results folder."
    )


# ---------------------------------------------------------------- page

st.title("Day-ahead electricity demand forecast")
st.caption("Group 2 · ICT30016 ICT Innovation Project · Client: ARC Research Hub for Integrated Energy Storage Solutions")

try:
    data = get_data(dd.results_signature())
except (FileNotFoundError, ValueError) as err:
    st.error(f"Could not load the results: {err}")
    st.caption(f"Expected files in: {dd.RESULTS_DIR}")
    st.stop()

for message in data.warnings:
    st.warning(message)

first_ts, last_ts = data.wide["datetime"].min(), data.wide["datetime"].max()
horizon = dd.forecast_horizon_hours(data.results)
ahead = f"{horizon} hours ahead" if horizon else "ahead of time"
st.markdown(
    f"Each forecast hour was predicted {ahead} and is compared here with what demand actually turned out "
    f"to be. All models were tested on the same hours ({first_ts:%d %b %Y} to {last_ts:%d %b %Y}). "
    "Accuracy on past data is not a guarantee for future days."
)

st.sidebar.header("Controls")
selected = st.sidebar.multiselect("Models to show", list(data.models), default=list(data.models))
if not selected:
    st.info("Choose at least one model in the sidebar.")
    st.stop()

min_day, max_day = first_ts.date(), last_ts.date()
default_start = min_day
picked = st.sidebar.date_input(
    "Date range", value=(default_start, max_day), min_value=min_day, max_value=max_day
)
if not (isinstance(picked, (tuple, list)) and len(picked) == 2):
    st.info("Pick an end date in the sidebar to show the chart.")
    st.stop()
start_day, end_day = picked

tabs = st.tabs(["Forecast vs actual", "Model comparison", "Why this forecast", "About the models"])
with tabs[0]:
    tab_forecast(data, selected, start_day, end_day)
with tabs[1]:
    tab_compare(data)
with tabs[2]:
    tab_why(data)
with tabs[3]:
    tab_about(data)

st.sidebar.caption("Data is read from the results folder. Reload the page after re-running a model.")
