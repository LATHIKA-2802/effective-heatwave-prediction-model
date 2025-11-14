import numpy as np
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb

# Load models
rf = joblib.load("heatwave_model_effective.pkl")
xgb_model = joblib.load("xgboost_heatwave_final.pkl")

# Features used
features = ['t2m', 'rh2m', 'ws10m', 'heat_index', 'consec_hot_days', 't2m_3day_avg']

# ------------------------- RANDOM FOREST FEATURE IMPORTANCE -------------------------
rf_importance = rf.feature_importances_

plt.figure(figsize=(8,5))
plt.barh(features, rf_importance)
plt.xlabel("Importance Score")
plt.title("Random Forest Feature Importance")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ------------------------- XGBOOST FEATURE IMPORTANCE (Corrected) -------------------------
# Extract importance from booster
booster = xgb_model.get_booster()
importance_dict = booster.get_score(importance_type='gain')

# Convert dict to list aligned with feature order
xgb_importance = [importance_dict.get(f, 0) for f in booster.feature_names]

plt.figure(figsize=(8,5))
plt.barh(booster.feature_names, xgb_importance, color='orange')
plt.xlabel("Importance Score (Gain)")
plt.title("XGBoost Feature Importance")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

