# Data Dictionary — Primary Dataset (Electricity Load Forecasting)

Owner: Elizabeth (Data & Platform Engineer)
Source file: `data/processed/cleaned_dataset.csv`
Original raw sources: `continuous dataset.csv` (main) + `weekly pre-dispatch
forecast.csv` (benchmark forecast, not yet merged into the cleaned file).

Frequency: hourly. Coverage: 2015-01-03 01:00 → 2020-06-27 00:00 (48,048 rows,
no gaps, no duplicates).

Weather variables are recorded at three stations, indicated by the column
suffix: `_toc`, `_san`, `_dav`. Each station has the same 4 weather measures.

| Column | Type | Unit | Description |
|---|---|---|---|
| `datetime` | datetime | — | Timestamp of the observation (hourly). |
| `nat_demand` | float | MW | **Target variable.** National electricity demand at this hour. Corresponds to what AEMO's terminology calls "Operational/National Demand" — the amount of power the system needs to supply. |
| `t2m_toc` | float | °C | Air temperature at 2 metres, station "toc". |
| `qv2m_toc` | float | kg/kg | Specific humidity at 2 metres, station "toc". |
| `tql_toc` | float | — | Total precipitable liquid water (cloud/rain water content), station "toc". |
| `w2m_toc` | float | m/s | Wind speed at 2 metres, station "toc". |
| `t2m_san` | float | °C | Air temperature at 2 metres, station "san". |
| `qv2m_san` | float | kg/kg | Specific humidity at 2 metres, station "san". |
| `tql_san` | float | — | Total precipitable liquid water, station "san". |
| `w2m_san` | float | m/s | Wind speed at 2 metres, station "san". |
| `t2m_dav` | float | °C | Air temperature at 2 metres, station "dav". |
| `qv2m_dav` | float | kg/kg | Specific humidity at 2 metres, station "dav". |
| `tql_dav` | float | — | Total precipitable liquid water, station "dav". |
| `w2m_dav` | float | m/s | Wind speed at 2 metres, station "dav". |
| `holiday_id` | int | — | Numeric code identifying which specific holiday, if any (0 = not a holiday). Values 1–22 map to different named holidays; exact mapping not yet confirmed with the raw source documentation. |
| `holiday` | int (0/1) | — | Whether the day is a public holiday (1) or not (0). |
| `school` | int (0/1) | — | Whether schools are in session (1) or on break (0) — a common driver of daytime demand patterns. |
| `nat_demand_outlier_flag` | int (0/1) | — | Added during cleaning (`01_data_cleaning.py`). 1 = this hour's `nat_demand` value falls outside the IQR bounds (558.8–1788.8 MW) and may be worth reviewing before training. Not removed from the dataset — kept for the modeling leads to decide. |

## Notes for the modeling leads (Jessica, Joud)

- Station names `toc`/`san`/`dav` are location codes from the original
  source; we have not yet confirmed which real cities/regions they
  correspond to. Flag if this matters for your model.
- `Holiday_ID` values are not yet individually documented (which number =
  which holiday). Let Elizabeth know if you need this broken out.
- The secondary forecast file (`weekly pre-dispatch forecast.csv`,
  columns: `datetime`, `load_forecast`) is a naive/benchmark forecast — not
  yet merged into the cleaned dataset. Useful as a baseline to compare
  model performance against.

## Reference

For background on demand terminology (native/operational/scheduled
demand), see AEMO's *Demand Terms in EMMS Data Model* (July 2025) —
provided as reference documentation, not a training source.
