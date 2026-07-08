"""
Shadow Buyer model: predicts whether a NEW product (not yet listed) will succeed.

Deliberately only uses features a seller would know BEFORE listing —
no rating/reviews/orders, since those don't exist yet for an unlisted product:
  - price
  - discount_pct
  - category (one-hot encoded)
  - trend_score (real category momentum, derived from platform order growth)
  - cod_share (category's current COD order share -- platform-level)
  - rto_rate (category's current return-to-origin rate -- platform-level)
  - return_rate (category's current post-delivery return rate -- platform-level)
  - seller_success_rate (this seller's historical track record)
  - seller_avg_rating (this seller's historical average rating, as a trust proxy)

trend_score/cod_share/rto_rate/return_rate come from category_trend_summary.csv,
computed from a year of real platform category data -- not guessed per-listing.

Model: Random Forest Classifier (an ensemble of decision trees) -- chosen because
it captures COMBINATIONS of factors (e.g. "high price + low-trend category" is much
riskier than either factor alone), which a plain logistic regression can't do as well.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

df = pd.read_csv("products_dataset.csv")
trend_summary = pd.read_csv("category_trend_summary.csv")

# Seller-level historical stats (their track record going into a new listing)
seller_stats = df.groupby("seller_id").agg(
    seller_success_rate=("succeeded", "mean"),
    seller_avg_rating=("rating", "mean"),
).reset_index()

df = df.merge(seller_stats, on="seller_id", how="left")

CATEGORIES = sorted(df["category"].unique())
cat_dummies = pd.get_dummies(df["category"], prefix="cat")

BASE_FEATURES = [
    "price", "discount_pct", "trend_score", "cod_share", "rto_rate", "return_rate",
    "seller_success_rate", "seller_avg_rating",
]
FEATURE_COLS = BASE_FEATURES + list(cat_dummies.columns)

X = pd.concat([df[BASE_FEATURES], cat_dummies], axis=1)
y = df["succeeded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    min_samples_leaf=4,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]

print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
print(f"ROC AUC:  {roc_auc_score(y_test, probs):.3f}")
print(classification_report(y_test, preds))

importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances:")
for f, imp in importances:
    print(f"  {f:25s} {imp:.3f}")

joblib.dump({
    "model": model,
    "feature_cols": FEATURE_COLS,
    "categories": CATEGORIES,
    "seller_stats": seller_stats,
    "trend_summary": trend_summary.set_index("category"),
}, "shadow_buyer_model.joblib")

print("\nSaved shadow_buyer_model.joblib")
