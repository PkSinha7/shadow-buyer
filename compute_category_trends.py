"""
Computes a per-category trend summary from category_trends.csv:
  - momentum: % change in orders, last 3 months vs the 3 months before that
  - trend_score: momentum normalized to 0-1 across categories --
    this REPLACES the old hardcoded/manual trend value entirely
  - current cod_share, rto_rate, return_rate (most recent month)

This is what Shadow Buyer and Ripple use automatically once a seller picks
a category -- no manual guessing required.

Run:  python compute_category_trends.py
Output: category_trend_summary.csv
"""

import pandas as pd

df = pd.read_csv("category_trends.csv")

summary_rows = []
for category, g in df.groupby("category"):
    g = g.sort_values("month")
    last3 = g.tail(3)["orders"].mean()
    prior3 = g.iloc[-6:-3]["orders"].mean()
    momentum = (last3 - prior3) / prior3 if prior3 > 0 else 0.0

    latest = g.iloc[-1]
    prior = g.iloc[-2] if len(g) >= 2 else latest
    drr = latest["orders"] / 30.0
    prior_drr = prior["orders"] / 30.0
    drr_change_pct = ((drr - prior_drr) / prior_drr * 100) if prior_drr > 0 else 0.0

    summary_rows.append({
        "category": category,
        "momentum": round(momentum, 4),
        "latest_orders": int(latest["orders"]),
        "latest_gmv": latest["gmv"],
        "latest_nmv": latest["nmv"],
        "cod_share": latest["cod_share"],
        "rto_rate": latest["rto_rate"],
        "return_rate": latest["return_rate"],
        "drr": round(drr, 1),
        "drr_change_pct": round(drr_change_pct, 1),
    })

summary = pd.DataFrame(summary_rows)

# Normalize momentum to a 0-1 trend_score across categories, so the model
# always sees a consistent 0-1 scale regardless of raw growth-rate magnitude
min_m, max_m = summary["momentum"].min(), summary["momentum"].max()
if max_m > min_m:
    summary["trend_score"] = (summary["momentum"] - min_m) / (max_m - min_m)
else:
    summary["trend_score"] = 0.5

summary.to_csv("category_trend_summary.csv", index=False)
print(summary.to_string(index=False))
