"""
Shadow Buyer + Ripple Simulator API

  GET  /                        -> frontend
  GET  /sellers                 -> list of seller IDs
  GET  /categories              -> list of categories
  GET  /category_trends/{cat}   -> real platform trend signal for a category
  GET  /products/{seller_id}    -> this seller's products (for Ripple tab dropdown)
  POST /predict_desirability    -> Shadow Buyer: will a new product succeed?
  POST /predict_ripple          -> Ripple Simulator: effect of a price change

Run locally:  uvicorn app:app --reload --port 8000
"""

import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

app = FastAPI(title="Shadow Buyer + Ripple API")

df = pd.read_csv("products_dataset.csv")
sb_bundle = joblib.load("shadow_buyer_model.joblib")
rp_bundle = joblib.load("ripple_model.joblib")

seller_stats = sb_bundle["seller_stats"].set_index("seller_id")
trend_summary = sb_bundle["trend_summary"]  # indexed by category

# ---------- Optional: TabPFN (tabular foundation model) as an upgrade over Random Forest ----------
# Never required. If tabpfn isn't installed, or the training script couldn't fetch it
# (needs a Hugging Face account + accepted terms + login -- see train_shadow_buyer.py),
# this silently stays off and every prediction just uses Random Forest as before.
TABPFN_READY = False
tabpfn_model = None
try:
    if sb_bundle.get("tabpfn_train_X") is not None:
        from tabpfn import TabPFNClassifier
        tabpfn_model = TabPFNClassifier(device="cpu")
        tabpfn_model.fit(sb_bundle["tabpfn_train_X"], sb_bundle["tabpfn_train_y"])
        TABPFN_READY = True
        print("TabPFN loaded -- Shadow Buyer will use it as the primary model, with Random Forest as fallback.")
    else:
        print("TabPFN context not found in bundle (wasn't available at training time) -- using Random Forest.")
except Exception as e:
    print(f"TabPFN failed to load ({type(e).__name__}) -- using Random Forest instead. This is safe and expected if TabPFN isn't set up.")
    TABPFN_READY = False

# Dummy seasonal/event presets. Each event boosts different categories by
# different amounts (e.g. Diwali helps Jewellery/Kurtis a lot, barely helps
# Electronics). These are illustrative estimates, not derived from real data --
# clearly flagged as such in the UI. A seller can select more than one.
EVENT_PRESETS = {
    "diwali": {
        "label": "Diwali sale",
        "boost": {"Kurtis": 0.35, "Jewellery": 0.45, "Home Decor": 0.30,
                  "Electronics Accessories": 0.25, "Footwear": 0.15, "Kids Wear": 0.20},
    },
    "holi": {
        "label": "Holi season",
        "boost": {"Kurtis": 0.20, "Jewellery": 0.10, "Home Decor": 0.15,
                  "Electronics Accessories": 0.05, "Footwear": 0.10, "Kids Wear": 0.15},
    },
    "wedding_season": {
        "label": "Wedding season",
        "boost": {"Jewellery": 0.40, "Kurtis": 0.30, "Footwear": 0.15,
                  "Home Decor": 0.20, "Electronics Accessories": 0.05, "Kids Wear": 0.10},
    },
    "valentines_day": {
        "label": "Valentine's Day",
        "boost": {"Jewellery": 0.30, "Kurtis": 0.10, "Footwear": 0.05,
                  "Home Decor": 0.05, "Electronics Accessories": 0.05, "Kids Wear": 0.05},
    },
    "blockbuster_sale": {
        "label": "Blockbuster / Big sale days",
        "boost": {"Electronics Accessories": 0.45, "Footwear": 0.30, "Kurtis": 0.25,
                  "Jewellery": 0.15, "Home Decor": 0.20, "Kids Wear": 0.25},
    },
    "back_to_school": {
        "label": "Back to school season",
        "boost": {"Kids Wear": 0.35, "Electronics Accessories": 0.20, "Footwear": 0.15,
                  "Kurtis": 0.05, "Home Decor": 0.05, "Jewellery": 0.05},
    },
    "monsoon_sale": {
        "label": "Monsoon sale",
        "boost": {"Footwear": 0.15, "Home Decor": 0.10, "Electronics Accessories": 0.10,
                  "Kurtis": 0.05, "Jewellery": 0.05, "Kids Wear": 0.05},
    },
}


@app.get("/events")
def get_events():
    return [{"id": k, "label": v["label"]} for k, v in EVENT_PRESETS.items()]


# ---------- Shadow Buyer ----------

class DesirabilityRequest(BaseModel):
    seller_id: str
    category: str
    price: float = Field(..., gt=0)
    discount_pct: float = Field(0, ge=0, le=100)
    selected_events: list[str] = Field(
        default_factory=list,
        description="IDs of seasonal/event presets the seller believes apply, e.g. ['diwali', 'wedding_season']"
    )


def _predict_prob(X: pd.DataFrame) -> tuple[float, str]:
    """Returns (success_probability, model_used). Tries TabPFN first if ready, falls back to Random Forest."""
    if TABPFN_READY:
        try:
            return float(tabpfn_model.predict_proba(X)[0, 1]), "tabpfn"
        except Exception as e:
            print(f"TabPFN prediction failed for this request ({type(e).__name__}) -- using Random Forest instead.")
    return float(sb_bundle["model"].predict_proba(X)[0, 1]), "random_forest"


@app.post("/predict_desirability")
def predict_desirability(req: DesirabilityRequest):
    if req.seller_id not in seller_stats.index:
        raise HTTPException(status_code=404, detail=f"Unknown seller_id '{req.seller_id}'. Try one of: {list(seller_stats.index)}")

    if req.category not in sb_bundle["categories"]:
        raise HTTPException(status_code=400, detail=f"Unknown category. Choose one of: {sb_bundle['categories']}")

    unknown_events = [e for e in req.selected_events if e not in EVENT_PRESETS]
    if unknown_events:
        raise HTTPException(status_code=400, detail=f"Unknown event id(s): {unknown_events}. Choose from: {list(EVENT_PRESETS.keys())}")

    s = seller_stats.loc[req.seller_id]
    cat_trend = trend_summary.loc[req.category]

    base_trend = float(cat_trend["trend_score"])
    event_boost = sum(EVENT_PRESETS[e]["boost"].get(req.category, 0.0) for e in req.selected_events)
    trend_score = min(1.0, base_trend + event_boost)

    def build_row(price: float, discount_pct: float) -> pd.DataFrame:
        row = {
            "price": price,
            "discount_pct": discount_pct,
            "trend_score": trend_score,
            "cod_share": float(cat_trend["cod_share"]),
            "rto_rate": float(cat_trend["rto_rate"]),
            "return_rate": float(cat_trend["return_rate"]),
            "seller_success_rate": s["seller_success_rate"],
            "seller_avg_rating": s["seller_avg_rating"],
        }
        for c in sb_bundle["categories"]:
            row[f"cat_{c}"] = 1 if c == req.category else 0
        return pd.DataFrame([row])[sb_bundle["feature_cols"]]

    X = build_row(req.price, req.discount_pct)
    prob_success, model_used = _predict_prob(X)

    if prob_success >= 0.65:
        verdict = "likely to succeed"
    elif prob_success >= 0.4:
        verdict = "uncertain — worth adjusting"
    else:
        verdict = "unlikely to succeed as-is"

    importances = dict(zip(sb_bundle["feature_cols"], sb_bundle["model"].feature_importances_))
    tips = []
    if req.price > df["price"].median() and importances.get("price", 0) > 0.15:
        tips.append("Price is above the typical range for similar products — consider testing a lower price point.")
    if req.discount_pct == 0:
        tips.append("No discount applied — even a small launch discount often helps early traction.")
    if trend_score < 0.35:
        tips.append(f"{req.category} isn't trending strongly on the platform right now — timing may hurt visibility.")
    if req.selected_events:
        event_labels = ", ".join(EVENT_PRESETS[e]["label"] for e in req.selected_events)
        tips.append(f"Boosted by: {event_labels} — good timing if the listing is ready before the sale starts.")
    if cat_trend["rto_rate"] > 0.12:
        tips.append(f"{req.category} has an above-average RTO rate right now — consider limiting COD or confirming orders before shipping.")
    if cat_trend["return_rate"] > 0.14:
        tips.append(f"{req.category} has a higher-than-usual return rate lately — double check sizing/description accuracy.")
    if s["seller_success_rate"] < 0.3:
        tips.append("Your account's historical success rate is on the lower side — consider strengthening product photos/description.")
    if not tips:
        tips.append("No major red flags — this looks like a reasonably solid listing.")

    # ---- Price benchmark: how this price compares to the category ----
    cat_products = df[df["category"] == req.category]
    successful_products = cat_products[cat_products["succeeded"] == 1]
    category_median_price = float(cat_products["price"].median())
    successful_median_price = float(successful_products["price"].median()) if len(successful_products) else category_median_price
    price_percentile = float((cat_products["price"] < req.price).mean() * 100)

    # ---- Similar real products in this category, closest by price ----
    cat_products = cat_products.copy()
    cat_products["price_diff"] = (cat_products["price"] - req.price).abs()
    similar = cat_products.sort_values("price_diff").head(4)
    similar_products = [
        {
            "title": r["product_title"],
            "price": round(float(r["price"]), 2),
            "rating": round(float(r["rating"]), 2),
            "orders_last_30d": int(r["orders_last_30d"]),
            "succeeded": bool(r["succeeded"]),
        }
        for _, r in similar.iterrows()
    ]

    # ---- Sensitivity: how nearby price/discount choices shift the outcome ----
    sensitivity = []
    variants = [
        ("20% lower price", req.price * 0.8, req.discount_pct),
        ("10% lower price", req.price * 0.9, req.discount_pct),
        ("Your price", req.price, req.discount_pct),
        ("10% higher price", req.price * 1.1, req.discount_pct),
        ("+10% extra discount", req.price, min(100.0, req.discount_pct + 10)),
    ]
    for label, p, d in variants:
        Xv = build_row(round(p, 2), d)
        prob_v, _ = _predict_prob(Xv)
        sensitivity.append({
            "label": label,
            "price": round(p, 2),
            "discount_pct": d,
            "success_probability": round(prob_v, 3),
        })

    # ---- Seller vs category benchmark ----
    category_avg_success_rate = float(df[df["category"] == req.category]["succeeded"].mean())

    return {
        "success_probability": round(prob_success, 3),
        "model_used": model_used,
        "verdict": verdict,
        "tips": tips,
        "category_signal": {
            "base_trend_score": round(base_trend, 3),
            "trend_score": round(trend_score, 3),
            "event_boost": round(event_boost, 3),
            "cod_share": round(float(cat_trend["cod_share"]), 3),
            "rto_rate": round(float(cat_trend["rto_rate"]), 3),
            "return_rate": round(float(cat_trend["return_rate"]), 3),
            "momentum_pct": round(float(cat_trend["momentum"]) * 100, 1),
        },
        "price_benchmark": {
            "category_median_price": round(category_median_price, 2),
            "successful_median_price": round(successful_median_price, 2),
            "price_percentile": round(price_percentile, 1),
        },
        "similar_products": similar_products,
        "sensitivity": sensitivity,
        "seller_benchmark": {
            "seller_success_rate": round(float(s["seller_success_rate"]), 3),
            "category_avg_success_rate": round(category_avg_success_rate, 3),
        },
    }


@app.get("/category_trends/{category}")
def get_category_trend(category: str):
    if category not in trend_summary.index:
        raise HTTPException(status_code=404, detail=f"Unknown category '{category}'")
    c = trend_summary.loc[category]
    return {
        "category": category,
        "trend_score": round(float(c["trend_score"]), 3),
        "momentum_pct": round(float(c["momentum"]) * 100, 1),
        "cod_share": round(float(c["cod_share"]), 3),
        "rto_rate": round(float(c["rto_rate"]), 3),
        "return_rate": round(float(c["return_rate"]), 3),
        "drr": round(float(c["drr"]), 1),
        "drr_change_pct": round(float(c["drr_change_pct"]), 1),
        "latest_gmv": float(c["latest_gmv"]),
        "latest_orders": int(c["latest_orders"]),
        "explainers": {
            "drr": "Daily Run Rate — average orders per day for this category across the whole platform right now.",
            "momentum": "How much orders have grown (or shrunk) over the last 3 months compared to the 3 months before that. Positive means the category is heating up.",
            "cod_share": "What share of orders in this category are Cash on Delivery vs prepaid. Higher COD share usually means more price-sensitive buyers.",
            "rto_rate": "Return to Origin rate — of the COD orders shipped, what percentage come back undelivered (buyer refused, wrong address, unreachable). This is a pure cost to the seller: shipping paid twice, nothing sold.",
            "return_rate": "Of orders that WERE delivered, what percentage the buyer sent back afterward (wrong size, not as described, changed mind).",
        },
    }


# ---------- Ripple Simulator ----------

class RippleRequest(BaseModel):
    product_id: str
    price_change_pct: float = Field(..., description="Negative for a price drop, e.g. -15 for -15%")
    selected_events: list[str] = Field(
        default_factory=list,
        description="IDs of seasonal/event presets to simulate alongside the price change, e.g. ['diwali']"
    )


def _predict_orders(row: dict) -> float:
    X_row = {k: row[k] for k in ["price", "discount_pct", "rating", "trend_score", "rto_rate", "return_rate"]}
    for c in rp_bundle["categories"]:
        X_row[f"cat_{c}"] = 1 if row["category"] == c else 0
    X = pd.DataFrame([X_row])[rp_bundle["feature_cols"]]
    return float(rp_bundle["model"].predict(X)[0])


@app.post("/predict_ripple")
def predict_ripple(req: RippleRequest):
    match = df[df["product_id"] == req.product_id]
    if match.empty:
        raise HTTPException(status_code=404, detail=f"Unknown product_id '{req.product_id}'")

    unknown_events = [e for e in req.selected_events if e not in EVENT_PRESETS]
    if unknown_events:
        raise HTTPException(status_code=400, detail=f"Unknown event id(s): {unknown_events}. Choose from: {list(EVENT_PRESETS.keys())}")

    product = match.iloc[0].to_dict()
    category = product["category"]
    cat_trend = trend_summary.loc[category]

    # Use the LIVE category signal (not the frozen value from when the dataset
    # was generated) plus any selected event boosts, so the simulation reflects
    # what's happening right now -- same logic as the "Will it sell?" tab.
    base_trend = float(cat_trend["trend_score"])
    event_boost = sum(EVENT_PRESETS[e]["boost"].get(category, 0.0) for e in req.selected_events)
    trend_score = min(1.0, base_trend + event_boost)

    product["trend_score"] = trend_score
    product["rto_rate"] = float(cat_trend["rto_rate"])
    product["return_rate"] = float(cat_trend["return_rate"])

    current_orders = _predict_orders(product)

    new_product = dict(product)
    new_product["price"] = product["price"] * (1 + req.price_change_pct / 100)
    new_orders = _predict_orders(new_product)

    current_revenue = current_orders * product["price"]
    new_revenue = new_orders * new_product["price"]
    revenue_change_pct = ((new_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0

    return {
        "product_id": req.product_id,
        "product_title": product["product_title"],
        "category": category,
        "current_price": round(product["price"], 2),
        "new_price": round(new_product["price"], 2),
        "current_predicted_orders": round(current_orders, 1),
        "new_predicted_orders": round(new_orders, 1),
        "current_predicted_revenue": round(current_revenue, 2),
        "new_predicted_revenue": round(new_revenue, 2),
        "revenue_change_pct": round(revenue_change_pct, 2),
        "recommendation": "Worth trying" if revenue_change_pct > 0 else "Likely to reduce revenue — reconsider",
        "category_signal": {
            "base_trend_score": round(base_trend, 3),
            "trend_score": round(trend_score, 3),
            "event_boost": round(event_boost, 3),
            "rto_rate": round(float(cat_trend["rto_rate"]), 3),
            "return_rate": round(float(cat_trend["return_rate"]), 3),
            "momentum_pct": round(float(cat_trend["momentum"]) * 100, 1),
        },
    }


@app.get("/products/{seller_id}")
def get_products(seller_id: str):
    rows = df[df["seller_id"] == seller_id][["product_id", "product_title", "price", "category"]]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No products found for seller_id '{seller_id}'")
    return rows.to_dict(orient="records")


@app.get("/sellers")
def get_sellers():
    return sorted(df["seller_id"].unique().tolist())


@app.get("/categories")
def get_categories():
    return sb_bundle["categories"]


app.mount("/", StaticFiles(directory="static", html=True), name="static")
