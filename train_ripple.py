"""
Ripple Simulator model: predicts orders_last_30d for an EXISTING product,
so we can compare "orders at current price" vs "orders at a hypothetical new price"
and estimate the revenue impact of a price change.

Model: Gradient Boosting Regressor -- a step up from linear regression because
it can capture non-linear effects (e.g. small discounts barely move orders,
but past a certain discount threshold orders jump sharply).
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("products_dataset.csv")

CATEGORIES = sorted(df["category"].unique())
cat_dummies = pd.get_dummies(df["category"], prefix="cat")

FEATURE_COLS = ["price", "discount_pct", "rating", "trend_score", "rto_rate", "return_rate"] + list(cat_dummies.columns)

X = pd.concat([df[["price", "discount_pct", "rating", "trend_score", "rto_rate", "return_rate"]], cat_dummies], axis=1)
y = df["orders_last_30d"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=150,
    max_depth=3,
    learning_rate=0.08,
    random_state=42,
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, preds):.2f} orders")
print(f"R^2:  {r2_score(y_test, preds):.3f}")

importances = sorted(zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances:")
for f, imp in importances:
    print(f"  {f:25s} {imp:.3f}")

joblib.dump({
    "model": model,
    "feature_cols": FEATURE_COLS,
    "categories": CATEGORIES,
}, "ripple_model.joblib")

print("\nSaved ripple_model.joblib")
