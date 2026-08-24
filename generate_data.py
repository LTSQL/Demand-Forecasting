"""
Simulates 2 years of daily demand for a retail SKU, with trend, weekly
seasonality, a promo effect, and noise — representative of what a
retail/e-commerce analyst would pull from a sales database.
"""
import os
import numpy as np
import pandas as pd

# All files (data, charts) are saved right next to this script, so there's
# no folder structure to worry about — this works no matter what the repo
# folder is named or where it's cloned/extracted to.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(7)

dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
n = len(dates)

trend = np.linspace(50, 90, n)  # slow growth over 2 years
weekly = 15 * np.sin(2 * np.pi * (dates.dayofweek) / 7 + 1)  # weekend peaks
yearly = 10 * np.sin(2 * np.pi * dates.dayofyear / 365 + 2)  # mild seasonality
noise = np.random.normal(0, 6, n)

# promo days: random ~4% of days, boosts demand ~40%
promo = np.random.rand(n) < 0.04
promo_effect = promo * (trend * 0.4)

demand = trend + weekly + yearly + noise + promo_effect
demand = np.clip(demand, 0, None).round().astype(int)

df = pd.DataFrame({
    "date": dates,
    "units_sold": demand,
    "promo_flag": promo.astype(int),
})
out_path = os.path.join(SCRIPT_DIR, "daily_demand.csv")
df.to_csv(out_path, index=False, encoding="utf-8")
print(f"Generated {n} days of demand data -> {out_path}")
