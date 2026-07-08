"""
Generates a synthetic dataset of sellers and their products.

~40 sellers, ~40-70 products each (~2000+ rows). Columns:
  seller_id, product_id, product_title, category, price, discount_pct,
  rating, num_reviews, orders_last_30d, views_last_30d,
  trend_score, cod_share, rto_rate, return_rate, succeeded

IMPORTANT: trend_score, cod_share, rto_rate, return_rate are no longer
guessed per-product. They're pulled from category_trend_summary.csv,
which is computed from a full year of category-level platform data
(GMV, NMV, orders, returns, COD/RTO) in generate_category_trends.py +
compute_category_trends.py. Run those two scripts BEFORE this one.

succeeded = 1 if the product became a top performer in its category (label
for the Shadow Buyer classifier). orders_last_30d is the regression target
for the Ripple Simulator.
"""

import numpy as np
import pandas as pd

np.random.seed(7)

try:
    trend_summary = pd.read_csv("category_trend_summary.csv").set_index("category")
except FileNotFoundError:
    raise SystemExit(
        "category_trend_summary.csv not found. Run these first:\n"
        "  python generate_category_trends.py\n"
        "  python compute_category_trends.py"
    )

SELLERS = [f"S{100+i}" for i in range(40)]
CATEGORIES = list(trend_summary.index)

PRODUCT_NAMES = {
    "Kurtis": ["Floral Anarkali Kurti", "Cotton Straight Kurti", "Printed A-line Kurti", "Embroidered Kurti Set"],
    "Footwear": ["Men's Running Shoes", "Women's Flat Sandals", "Kids Casual Sneakers", "Ethnic Juttis"],
    "Jewellery": ["Oxidised Silver Jhumkas", "Kundan Choker Set", "Pearl Drop Earrings", "Beaded Bracelet"],
    "Home Decor": ["Wall Hanging Clock", "LED Fairy Lights", "Cotton Cushion Cover", "Ceramic Planter"],
    "Kids Wear": ["Kids Printed Frock", "Boys Cotton Shirt", "Girls Party Dress", "Kids Pajama Set"],
    "Electronics Accessories": ["Bluetooth Neckband", "Phone Ring Holder", "USB Cable 3-pack", "Wireless Mouse"],
}

rows = []
product_counter = 1

for seller in SELLERS:
    # Each seller has a hidden "skill" level that biases their success rate
    seller_skill = np.random.normal(0, 1)
    n_products = np.random.randint(40, 71)

    for _ in range(n_products):
        category = np.random.choice(CATEGORIES)
        cat_trend = trend_summary.loc[category]

        title = np.random.choice(PRODUCT_NAMES[category])
        price = float(np.random.gamma(2.5, 250) + 150).__round__(0)
        discount_pct = np.random.choice([0, 5, 10, 15, 20, 25], p=[0.35, 0.15, 0.15, 0.15, 0.1, 0.1])

        # Real platform-level category signals -- same for every product in this
        # category right now, small noise added since individual listings vary slightly
        trend_score = float(np.clip(cat_trend["trend_score"] + np.random.normal(0, 0.03), 0, 1))
        cod_share = float(np.clip(cat_trend["cod_share"] + np.random.normal(0, 0.02), 0, 1))
        rto_rate = float(np.clip(cat_trend["rto_rate"] + np.random.normal(0, 0.01), 0, 1))
        return_rate = float(np.clip(cat_trend["return_rate"] + np.random.normal(0, 0.01), 0, 1))

        rating = np.clip(np.random.normal(3.9 + seller_skill * 0.15, 0.4), 2.5, 5.0)
        num_reviews = max(0, int(np.random.poisson(40) + seller_skill * 10))
        views_last_30d = max(10, int(np.random.gamma(2, 200) * (0.6 + trend_score)))

        # Orders driven by price attractiveness, real trend momentum, rating,
        # seller skill, discount -- and now penalized by high RTO/return rate,
        # since a category prone to failed deliveries or returns dampens net demand
        price_category_avg = 550
        price_position = (price - price_category_avg) / price_category_avg

        expected_orders = (
            18
            + 30 * trend_score
            + 8 * discount_pct / 100 * 10
            + 6 * (rating - 3.5)
            + 5 * seller_skill
            - 20 * price_position
            - 15 * rto_rate
            - 10 * return_rate
            + np.random.normal(0, 4)
        )
        orders_last_30d = max(0, int(expected_orders))

        succeeded_score = expected_orders + np.random.normal(0, 3)
        rows.append({
            "seller_id": seller,
            "product_id": f"P{product_counter:04d}",
            "product_title": title,
            "category": category,
            "price": price,
            "discount_pct": discount_pct,
            "rating": round(rating, 2),
            "num_reviews": num_reviews,
            "orders_last_30d": orders_last_30d,
            "views_last_30d": views_last_30d,
            "trend_score": round(trend_score, 3),
            "cod_share": round(cod_share, 3),
            "rto_rate": round(rto_rate, 3),
            "return_rate": round(return_rate, 3),
            "_succeeded_score": succeeded_score,
        })
        product_counter += 1

df = pd.DataFrame(rows)

threshold = df["_succeeded_score"].quantile(0.65)
df["succeeded"] = (df["_succeeded_score"] >= threshold).astype(int)
df = df.drop(columns=["_succeeded_score"])

df.to_csv("products_dataset.csv", index=False)
print(f"Generated {len(df)} products across {df['seller_id'].nunique()} sellers.")
print(f"Success rate: {df['succeeded'].mean():.2%}")
print(df.head())
