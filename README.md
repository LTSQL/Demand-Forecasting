# Retail Demand Forecasting — Framed as a Stocking Decision

**Business question:** Given daily demand history for a SKU, how many units should we order each day — not "what will accuracy be," but *what decision minimises expected cost?*

## Why this project

Most forecasting projects stop at reporting an accuracy metric (RMSE, MAPE) and call it done. That's not useful to a business on its own — a forecast only matters once it's turned into a decision.

This project explicitly converts the forecast into an order quantity recommendation using a cost-benefit (newsvendor) model, and quantifies the £ impact of using it instead of a naive point forecast.

## Approach

This is a flat project — every script, data file, and output image sits in the same folder. No subfolders needed.

- **`generate_data.py`** — simulates 2 years of daily demand for one SKU with trend, weekly seasonality, a promo effect, and noise (stands in for a sales database extract).
- **`forecast_and_decide.py`**:
  - Fits a Holt-Winters exponential smoothing model (additive trend + weekly seasonality), validated on a held-out final 30 days.
  - States two business cost assumptions explicitly: profit lost per stockout unit, and holding/markdown cost per overstocked unit.
  - Uses the classic newsvendor formula to convert the point forecast + residual uncertainty into a cost-optimised order quantity (adds safety stock, doesn't just use the raw forecast).
  - Simulates and compares the total cost over the test period of (a) ordering the naive point forecast vs (b) ordering the cost-optimised quantity.

## How to run

```bash
pip install pandas numpy matplotlib statsmodels scikit-learn scipy
python generate_data.py
python forecast_and_decide.py
```

## Key findings (from this run)

- Model MAE: 6.6 units/day, MAPE: 6.9% on the 30-day held-out test period.
- Given the stated cost assumptions (stockout costs more than overstock), the optimal service level was 77%, meaning it's worth carrying visible extra stock rather than ordering to the point forecast.
- Simply ordering to the point forecast (ignoring uncertainty) cost 16% more over the test period than the cost-optimised order quantity — a concrete, defensible number to bring to a stakeholder conversation.

## What I'd do with more time

- Cost assumptions were illustrative; a real version would pull these from finance/ops data.
- Add per-SKU forecasting at scale (this is single-SKU; real retail forecasting is often thousands of SKUs — worth showing awareness of that scaling problem).
- Compare Holt-Winters against a gradient-boosted model (e.g. LightGBM with lag features) to show the trade-off between interpretability and raw accuracy.

