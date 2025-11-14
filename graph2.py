import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import joblib
import xgboost as xgb

# Load models
rf = joblib.load("heatwave_model_effective.pkl")
xgb_model = joblib.load("xgboost_heatwave_final.pkl")

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Predict probabilities
rf_prob = rf.predict_proba(X_test)[:,1]
xgb_prob = xgb_model.predict_proba(X_test)[:,1]

# ROC values
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)

roc_auc_rf = auc(fpr_rf, tpr_rf)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot
plt.figure(figsize=(8,6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC = {roc_auc_xgb:.3f})")

plt.plot([0,1], [0,1], 'k--')  # diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid(True)
plt.show()
