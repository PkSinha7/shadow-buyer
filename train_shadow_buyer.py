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
print("\nFeature importances (Random Forest):")
for f, imp in importances:
    print(f"  {f:25s} {imp:.3f}")

# ---------- Optional: compare against TabPFN, a tabular foundation model ----------
# This is entirely optional and never breaks training if unavailable. TabPFN needs:
#   1. pip install tabpfn
#   2. A free Hugging Face account, having accepted the model's terms at
#      https://huggingface.co/Prior-Labs/tabpfn_3, and being logged in via
#      `hf auth login` (or an HF_TOKEN environment variable) on THIS machine.
# If any of that isn't set up, this block just prints why and moves on --
# Random Forest above is already saved and remains the app's guaranteed model.
TABPFN_MAX_CONTEXT_ROWS = 800  # keeps CPU inference fast; TabPFN docs note CPU is
                                # only fast up to ~1000 rows of context

tabpfn_train_X, tabpfn_train_y = None, None
try:
    from tabpfn import TabPFNClassifier

    if len(X_train) > TABPFN_MAX_CONTEXT_ROWS:
        tabpfn_train_X, _, tabpfn_train_y, _ = train_test_split(
            X_train, y_train,
            train_size=TABPFN_MAX_CONTEXT_ROWS,
            random_state=42,
            stratify=y_train,
        )
    else:
        tabpfn_train_X, tabpfn_train_y = X_train, y_train

    tabpfn_model = TabPFNClassifier(device="cpu")
    tabpfn_model.fit(tabpfn_train_X, tabpfn_train_y)
    tabpfn_probs = tabpfn_model.predict_proba(X_test)[:, 1]
    tabpfn_preds = (tabpfn_probs >= 0.5).astype(int)

    print(f"\n--- TabPFN comparison (context: {len(tabpfn_train_X)} rows) ---")
    print(f"Accuracy: {accuracy_score(y_test, tabpfn_preds):.3f}")
    print(f"ROC AUC:  {roc_auc_score(y_test, tabpfn_probs):.3f}")
    print("(Random Forest numbers are above for direct comparison)")

except ImportError:
    print("\nTabPFN not installed -- skipping comparison. (Run 'pip install tabpfn' to try it; app works fine without it.)")
except Exception as e:
    print(f"\nTabPFN unavailable -- skipping comparison: {type(e).__name__}: {str(e)[:200]}")
    print("(Likely needs Hugging Face login/terms acceptance -- see comment above. App works fine without it.)")
    tabpfn_train_X, tabpfn_train_y = None, None

joblib.dump({
    "model": model,
    "feature_cols": FEATURE_COLS,
    "categories": CATEGORIES,
    "seller_stats": seller_stats,
    "trend_summary": trend_summary.set_index("category"),
    "tabpfn_train_X": tabpfn_train_X,   # small context sample for optional TabPFN use in app.py
    "tabpfn_train_y": tabpfn_train_y,   # None if TabPFN wasn't available at training time
}, "shadow_buyer_model.joblib")

print("\nSaved shadow_buyer_model.joblib")
