import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import joblib
from tensorflow.keras.models import load_model

# Load models
rf = joblib.load("heatwave_model_effective.pkl")
xgb = joblib.load("xgboost_heatwave_final.pkl")
ann = load_model("ann_heatwave_final.h5")
cnn = load_model("cnn_heatwave_final.h5")
lstm = load_model("lstm_heatwave_final.h5")

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Reshape for CNN/LSTM
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Predictions
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
ann_pred = (ann.predict(X_test) > 0.5).astype("int32")
cnn_pred = (cnn.predict(X_test_cnn) > 0.5).astype("int32")
lstm_pred = (lstm.predict(X_test_lstm) > 0.5).astype("int32")

# Listing models
model_names = ["Random Forest", "ANN", "CNN", "LSTM", "XGBoost"]
predictions = [rf_pred, ann_pred, cnn_pred, lstm_pred, xgb_pred]


# Function to plot confusion matrix
def plot_conf_matrix(cm, model_name):
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.colorbar()

    classes = ["No Heatwave", "Heatwave"]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # print values inside squares
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='black', fontsize=12)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()


# Generate confusion matrix for each model
for name, pred in zip(model_names, predictions):
    cm = confusion_matrix(y_test, pred)
    plot_conf_matrix(cm, name)
