"""
Forecasts demand for the next 30 days, then converts the forecast into a
CONCRETE STOCKING DECISION using a cost-benefit model — this is the part
most student projects skip. Stopping at "RMSE = X" tells a business nothing;
this script answers "how many units should we order, and what's the
expected cost of getting it wrong?"
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# All files (data, charts) are saved right next to this script, so there's
# no folder structure to worry about — this works no matter what the repo
# folder is named or where it's cloned/extracted to.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "daily_demand.csv")
OUT_DIR = SCRIPT_DIR

df = pd.read_csv(DATA_PATH, parse_dates=["date"], encoding="utf-8").set_index("date")
y = df["units_sold"]

# --- Train/test split: last 30 days held out ---
train, test = y[:-30], y[-30:]

model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7).fit()
forecast = model.forecast(30)

mae = mean_absolute_error(test, forecast)
mape = mean_absolute_percentage_error(test, forecast)
resid_std = (test - forecast).std()

print(f"Holt-Winters (additive trend + weekly seasonality)")
print(f"MAE:  {mae:.1f} units/day")
print(f"MAPE: {mape:.1%}")
print(f"Residual std dev: {resid_std:.1f} units/day  <- used for the safety stock calc below")

# --- Business framing: newsvendor-style stocking decision ---
# Underestimating demand (stockout) loses a sale + goodwill.
# Overestimating demand (overstock) costs holding/markdown.
# These costs are illustrative assumptions, stated explicitly rather than
# hidden — a real analyst would source these from finance/ops.
unit_margin_lost_on_stockout = 12.00   # £ profit lost per unit of unmet demand
unit_cost_of_overstock = 3.50          # £ holding/markdown cost per excess unit

# Critical ratio (newsvendor formula): order up to the quantile that balances
# stockout cost against overstock cost.
critical_ratio = unit_margin_lost_on_stockout / (unit_margin_lost_on_stockout + unit_cost_of_overstock)
from scipy.stats import norm
z = norm.ppf(critical_ratio)
safety_stock_per_day = z * resid_std

recommended_daily_order = forecast + safety_stock_per_day
recommended_daily_order = recommended_daily_order.clip(lower=0).round()

print(f"\nCritical ratio (target service level): {critical_ratio:.1%}")
print(f"Recommended safety stock: {safety_stock_per_day:.1f} extra units/day on top of point forecast")
print(f"\nRecommended order quantity for next 7 days:")
print(recommended_daily_order.head(7).round(0).to_string())

# --- Expected cost comparison: naive point forecast vs cost-optimised order ---
def expected_cost(order_qty, actual):
    stockout = np.maximum(actual - order_qty, 0)
    overstock = np.maximum(order_qty - actual, 0)
    return (stockout * unit_margin_lost_on_stockout + overstock * unit_cost_of_overstock).sum()

cost_naive = expected_cost(forecast.round(), test)
cost_optimised = expected_cost(recommended_daily_order, test)

print(f"\nSimulated cost over the 30 held-out test days:")
print(f"  Ordering exactly the point forecast: £{cost_naive:,.0f}")
print(f"  Ordering with cost-optimised safety stock: £{cost_optimised:,.0f}")
print(f"  Estimated saving: £{cost_naive - cost_optimised:,.0f} ({(1 - cost_optimised/cost_naive):.1%} lower)")

# --- Chart ---
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(train.index[-60:], train.values[-60:], label="Historical (train)", color="gray")
ax.plot(test.index, test.values, label="Actual", color="black", linewidth=1.5)
ax.plot(test.index, forecast.values, label="Point forecast", color="#3b6ea5")
ax.plot(test.index, recommended_daily_order.values, label="Recommended order (with safety stock)",
        color="#d9822b", linestyle="--")
ax.legend()
ax.set_title("Demand Forecast vs Cost-Optimised Order Quantity")
ax.set_ylabel("Units")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/forecast_vs_order.png", dpi=150)
print(f"\nSaved chart -> {OUT_DIR}/forecast_vs_order.png")

# --- Save recommendation table ---
result = pd.DataFrame({
    "date": test.index,
    "actual": test.values,
    "point_forecast": forecast.round(1).values,
    "recommended_order": recommended_daily_order.values,
})
result.to_csv(f"{OUT_DIR}/forecast_recommendations.csv", index=False, encoding="utf-8")
print(f"Saved recommendation table -> {OUT_DIR}/forecast_recommendations.csv")
