import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------- Config ----------
METRICS_CSV = "model_comparison_summary.csv"      # must exist
PREDICTIONS_CSV = "multi_model_predictions.csv"   # must exist
Y_TEST_NPY = "y_test.npy"                          # optional (only for Confusion Matrix)
OUTDIR = "graphs"                                  # where images will be saved

os.makedirs(OUTDIR, exist_ok=True)

# ---------- Helpers ----------
def savefig(name):
    path = os.path.join(OUTDIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {path}")

# =========================================================
# Graph 1: Model Comparison Bar Charts (Accuracy/Precision/Recall/F1)
# =========================================================
metrics_df = pd.read_csv(METRICS_CSV)
# Accept either wide or already-shaped columns
# Expect columns: Model, Accuracy, Precision, Recall, F1 or F1 Score
metrics_df.columns = [c.strip().replace(" ", "_") for c in metrics_df.columns]
if "F1_Score" in metrics_df.columns and "F1" not in metrics_df.columns:
    metrics_df["F1"] = metrics_df["F1_Score"]

needed_cols = ["Model", "Accuracy", "Precision", "Recall", "F1"]
missing = [c for c in needed_cols if c not in metrics_df.columns]
if missing:
    raise ValueError(f"Columns missing in {METRICS_CSV}: {missing}")

# Plot each metric as a separate bar chart (clean look)
for metric in ["Accuracy", "Precision", "Recall", "F1"]:
    plt.figure(figsize=(8, 4.5))
    plt.bar(metrics_df["Model"], metrics_df[metric])
    plt.title(f"Model Comparison — {metric}")
    plt.xlabel("Model")
    plt.ylabel(metric)
    plt.xticks(rotation=15)
    savefig(f"model_comparison_{metric.lower()}.png")

# Also a grouped bar chart with all four metrics
plt.figure(figsize=(10, 5))
x = np.arange(len(metrics_df))
w = 0.18
plt.bar(x - 1.5*w, metrics_df["Accuracy"], width=w, label="Accuracy")
plt.bar(x - 0.5*w, metrics_df["Precision"], width=w, label="Precision")
plt.bar(x + 0.5*w, metrics_df["Recall"],    width=w, label="Recall")
plt.bar(x + 1.5*w, metrics_df["F1"],        width=w, label="F1")
plt.xticks(x, metrics_df["Model"], rotation=15)
plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Model Comparison — All Metrics")
plt.legend()
savefig("model_comparison_all_metrics.png")

# =========================================================
# Graph 2: Prediction Trend Across Days (0/1) per Model
# =========================================================
pred_df = pd.read_csv(PREDICTIONS_CSV)
pred_df.columns = [c.strip() for c in pred_df.columns]

# Try to detect prediction columns by suffix "_Pred"
pred_cols = [c for c in pred_df.columns if c.endswith("_Pred")]
if not pred_cols:
    # fallback: look for typical names
    candidates = ["RF_Pred","ANN_Pred","CNN_Pred","LSTM_Pred","XGB_Pred"]
    pred_cols = [c for c in candidates if c in pred_df.columns]

if not pred_cols:
    raise ValueError("No prediction columns found in multi_model_predictions.csv (e.g., *_Pred).")

# Trend per model (each as own line)
plt.figure(figsize=(12, 4))
for col in pred_cols:
    plt.plot(pred_df.index, pred_df[col], marker='', linewidth=1, label=col.replace("_Pred",""))
plt.yticks([0, 1], ["No Heatwave (0)", "Heatwave (1)"])
plt.xlabel("Day Index")
plt.ylabel("Prediction")
plt.title("Heatwave Predictions over Days (All Models)")
plt.legend(ncol=min(len(pred_cols), 5))
savefig("prediction_trend_all_models.png")

# =========================================================
# Graph 3: Count of Heatwave Predictions per Model (bar)
# =========================================================
counts = {col.replace("_Pred",""): int(pred_df[col].sum()) for col in pred_cols}
plt.figure(figsize=(7, 4))
plt.bar(list(counts.keys()), list(counts.values()))
plt.xlabel("Model")
plt.ylabel("Count of Predicted Heatwaves (1s)")
plt.title("How Many Heatwaves Did Each Model Predict?")
plt.xticks(rotation=15)
savefig("heatwave_counts_by_model.png")

# =========================================================
# Graph 4: (Optional) Confusion Matrix for Best Model
#          Only if y_test.npy exists AND we have a column with true labels.
# =========================================================
if os.path.exists(Y_TEST_NPY):
    y_true = np.load(Y_TEST_NPY)
    # Try to align: if predictions file also has a 'True' column, use that instead
    true_col = None
    for name in ["true","label","y_true","actual","Heatwave"]:
        if name in pred_df.columns:
            true_col = name
            break
    if true_col is not None:
        y_true = pred_df[true_col].values

    # Pick "best" model from metrics (max F1)
    best_row = metrics_df.iloc[metrics_df["F1"].idxmax()]
    best_name = str(best_row["Model"])
    best_pred_col = None
    # Map model names to prediction column names
    name_to_col = {
        "Random Forest": "RF_Pred",
        "RF": "RF_Pred",
        "ANN": "ANN_Pred",
        "CNN": "CNN_Pred",
        "LSTM": "LSTM_Pred",
        "XGBoost": "XGB_Pred",
        "XGB": "XGB_Pred"
    }
    # Try exact and flexible match
    for k,v in name_to_col.items():
        if k.lower() in best_name.lower() and v in pred_df.columns:
            best_pred_col = v
            break
    # Fallback: if not found, just pick the highest F1 among those present in pred_df
    if best_pred_col is None:
        present_map = {k:v for k,v in name_to_col.items() if v in pred_df.columns}
        # choose first present
        if present_map:
            best_pred_col = list(present_map.values())[0]
            best_name = [k for k,v in present_map.items() if v==best_pred_col][0]
    if best_pred_col is not None:
        y_pred = pred_df[best_pred_col].values
        # Ensure same length
        m = min(len(y_true), len(y_pred))
        y_true, y_pred = y_true[:m], y_pred[:m]

        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Heatwave (0)","Heatwave (1)"])
        plt.figure(figsize=(5, 4))
        disp.plot(values_format="d")
        plt.title(f"Confusion Matrix — {best_name}")
        savefig("confusion_matrix_best_model.png")
    else:
        print("ℹ️ Skipping confusion matrix: couldn't match best model to a *_Pred column.")
else:
    print("ℹ️ Skipping confusion matrix: y_test.npy not found.")

print("\n🎉 All graphs generated in the 'graphs' folder.")
