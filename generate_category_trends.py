"""
Generates 12 months of category-level platform performance data:
GMV, NMV, orders, returns, COD/prepaid split, RTO orders.

This is platform-wide data (not seller-specific) -- it represents what
the platform as a whole is seeing in each category over the last year.
It lets us derive a REAL trend signal per category (based on recent order
growth, RTO rate, return rate, COD share) instead of a guessed/manual value.

Run:  python generate_category_trends.py
Output: category_trends.csv  (6 categories x 12 months = 72 rows)
"""

import numpy as np
import pandas as pd

np.random.seed(11)

CATEGORIES = ["Kurtis", "Footwear", "Jewellery", "Home Decor", "Kids Wear", "Electronics Accessories"]
MONTHS = list(range(1, 13))  # 1 = 12 months ago ... 12 = last month

# Each category has its own growth trajectory over the year:
# base_orders = starting monthly order volume, growth_per_month = trend direction/strength,
# seasonality = how much it swings up/down across the year (festival categories swing more)
CATEGORY_PROFILE = {
    "Kurtis":                  {"base_orders": 3000, "growth_per_month": 40,  "seasonality": 0.15},
    "Footwear":                {"base_orders": 4200, "growth_per_month": -10, "seasonality": 0.05},
    "Jewellery":               {"base_orders": 1800, "growth_per_month": 25,  "seasonality": 0.30},
    "Home Decor":              {"base_orders": 2200, "growth_per_month": -25, "seasonality": 0.05},
    "Kids Wear":               {"base_orders": 2600, "growth_per_month": 15,  "seasonality": 0.10},
    "Electronics Accessories": {"base_orders": 5000, "growth_per_month": 60,  "seasonality": 0.05},
}

AVG_ORDER_VALUE = {
    "Kurtis": 550, "Footwear": 750, "Jewellery": 450,
    "Home Decor": 650, "Kids Wear": 400, "Electronics Accessories": 500,
}

rows = []
for category, profile in CATEGORY_PROFILE.items():
    for month in MONTHS:
        seasonal_boost = 1 + profile["seasonality"] * np.sin(month / 12 * 2 * np.pi)
        orders = max(200, int(
            (profile["base_orders"] + profile["growth_per_month"] * month) * seasonal_boost
            + np.random.normal(0, profile["base_orders"] * 0.04)
        ))
        aov = AVG_ORDER_VALUE[category] * np.random.uniform(0.95, 1.05)
        gmv = orders * aov

        return_rate = np.clip(np.random.normal(0.12, 0.02), 0.03, 0.30)
        returns = int(orders * return_rate)
        nmv = gmv - returns * aov

        cod_share = np.clip(np.random.normal(0.55, 0.08), 0.2, 0.85)
        cod_orders = int(orders * cod_share)
        prepaid_orders = orders - cod_orders

        # RTO tends to be higher in categories/months with higher COD share
        rto_rate = np.clip(np.random.normal(0.10, 0.03) + (cod_share - 0.5) * 0.08, 0.02, 0.35)
        rto_orders = int(cod_orders * rto_rate)

        rows.append({
            "category": category,
            "month": month,
            "orders": orders,
            "gmv": round(gmv, 2),
            "nmv": round(nmv, 2),
            "returns": returns,
            "return_rate": round(return_rate, 3),
            "cod_orders": cod_orders,
            "prepaid_orders": prepaid_orders,
            "cod_share": round(cod_share, 3),
            "rto_orders": rto_orders,
            "rto_rate": round(rto_rate, 3),
        })

df = pd.DataFrame(rows)
df.to_csv("category_trends.csv", index=False)
print(f"Generated {len(df)} rows across {len(CATEGORY_PROFILE)} categories x 12 months.")
print(df.head(12))
