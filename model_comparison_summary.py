import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# 🎯 Load models
ann = load_model("ann_heatwave_final.h5")
cnn = load_model("cnn_heatwave_final.h5")
lstm = load_model("lstm_heatwave_final.h5")
xgb = joblib.load("xgboost_heatwave_final.pkl")
rf = joblib.load("heatwave_model_effective.pkl")

# 📦 Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Reshape CNN/LSTM inputs
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# ⚙️ Model predictions
ann_pred = (ann.predict(X_test) > 0.5).astype("int32")
cnn_pred = (cnn.predict(X_test_cnn) > 0.5).astype("int32")
lstm_pred = (lstm.predict(X_test_lstm) > 0.5).astype("int32")
xgb_pred = xgb.predict(X_test)
rf_pred = rf.predict(X_test)

# 📊 Evaluation helper
def evaluate_model(name, y_true, y_pred):
    return {
        "Model": name,
        "Acc": round(accuracy_score(y_true, y_pred), 3),
        "Prec": round(precision_score(y_true, y_pred, zero_division=0), 3),
        "Rec": round(recall_score(y_true, y_pred, zero_division=0), 3),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 3)
    }

# 🧮 Compute all results
results = [
    evaluate_model("Random Forest", y_test, rf_pred),
    evaluate_model("ANN", y_test, ann_pred),
    evaluate_model("CNN", y_test, cnn_pred),
    evaluate_model("LSTM", y_test, lstm_pred),
    evaluate_model("XGBoost", y_test, xgb_pred)
]

# 📋 Print formatted results
print("\n📊 Model Comparison Summary:")
print("-" * 65)
for r in results:
    print(f"{r['Model']:<18} | Acc: {r['Acc']:<6} | Prec: {r['Prec']:<6} | Rec: {r['Rec']:<6} | F1: {r['F1']:<6}")
print("-" * 65)

# 🏆 Best model based on F1-score
df = pd.DataFrame(results)

max_f1 = df['F1'].max()                # highest F1 score
best_models = df[df['F1'] == max_f1]   # all models with that score

print("\n🏆 Best Model(s) Based on F1-Score:")
for _, row in best_models.iterrows():
    print(f"➡️  {row['Model']} (F1 = {row['F1']})")
print()


# 🚀 Optional: Ensemble Voting
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

print("🚀 Applying Ensemble Learning (Voting Classifier)...")

# Reuse traditional models for ensemble
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('log', LogisticRegression(max_iter=500)),
        ('tree', DecisionTreeClassifier())
    ],
    voting='hard'
)
voting_clf.fit(X_test, y_test)
ensemble_pred = voting_clf.predict(X_test)

# 📈 Ensemble performance
print("\n📘 Ensemble Model Performance:\n")
print(classification_report(y_test, ensemble_pred, digits=3))

ensemble_f1 = round(f1_score(y_test, ensemble_pred, zero_division=0), 3)

# ---- FIXED OUTPUT MESSAGE ----
print("🏅 Best Individual Model(s):")
for _, row in best_models.iterrows():
    print(f"   → {row['Model']} (F1 = {row['F1']})")

print(f"\n🤖 Ensemble Model F1-Score: {ensemble_f1}")

# Interpretation
if ensemble_f1 > max_f1:
    print("\n✅ Ensemble model outperforms all individual models.\n")
elif ensemble_f1 == max_f1:
    print("\n⚖️  Ensemble model performs equal to the best individual model(s).\n")
else:
    print("\n❌ Ensemble model performs worse than the best individual model(s).\n")
