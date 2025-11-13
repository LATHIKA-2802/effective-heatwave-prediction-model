import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
import xgboost as xgb
import joblib
import os

print("✅ Starting Unified Model Training for Effective Heatwave Prediction...")

# ---------------------------------------------------------
# Step 1: Load Dataset
# ---------------------------------------------------------
data_path = "C:/Users/lathika/Desktop/data mining project/combined_heatwave_data.csv"
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print(df.head())

# ---------------------------------------------------------
# Step 2: Create Target Column Consistently
# ---------------------------------------------------------
temp_threshold = df['t2m'].quantile(0.95)
humidity_threshold = 70  # adjustable threshold
df['heatwave'] = ((df['t2m'] >= temp_threshold) & (df['rh2m'] >= humidity_threshold)).astype(int)

print("\nHeatwave value counts:\n", df['heatwave'].value_counts())
if df['heatwave'].nunique() < 2:
    raise ValueError("❌ Only one class found! Adjust thresholds to detect more heatwaves.")

# ---------------------------------------------------------
# Step 3: Feature Engineering (Same as RF)
# ---------------------------------------------------------
df['consec_hot_days'] = (df['heatwave'] == 1).astype(int)
df['consec_hot_days'] = df['consec_hot_days'].groupby(
    (df['consec_hot_days'] != df['consec_hot_days'].shift()).cumsum()
).cumsum()

df['heat_index'] = df['t2m'] + 0.33 * df['rh2m'] - 0.70 * df['ws10m'] - 4.00
df['t2m_3day_avg'] = df['t2m'].rolling(3, min_periods=1).mean()

features = ['t2m', 'rh2m', 'ws10m', 'heat_index', 'consec_hot_days', 't2m_3day_avg']
X = df[features]
y = df['heatwave']

print(f"\n✅ Features used: {features}")

# ---------------------------------------------------------
# Step 4: Train-Test Split + SMOTE
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\n✅ Applied SMOTE successfully!")
print("Class distribution after SMOTE:\n", pd.Series(y_train_res).value_counts())

# ---------------------------------------------------------
# Step 5: Scale Data
# ---------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "scaler_final.pkl")

# ---------------------------------------------------------
# 🔹 ANN Model
# ---------------------------------------------------------
print("\n🚀 Training ANN model...")
ann_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_scaled, y_train_res, epochs=20, batch_size=32, verbose=1)
y_pred_ann = (ann_model.predict(X_test_scaled) > 0.5).astype(int)
print("\n📊 ANN Evaluation:")
print(confusion_matrix(y_test, y_pred_ann))
print(classification_report(y_test, y_pred_ann))
ann_model.save("ann_heatwave_final.h5")

# ---------------------------------------------------------
# 🔹 CNN Model
# ---------------------------------------------------------
print("\n🚀 Training CNN model...")
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

cnn_model = Sequential([
    Conv1D(64, 2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
    MaxPooling1D(2),
    Dropout(0.3),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train_res, epochs=15, batch_size=32, verbose=1)
y_pred_cnn = (cnn_model.predict(X_test_cnn) > 0.5).astype(int)
print("\n📊 CNN Evaluation:")
print(confusion_matrix(y_test, y_pred_cnn))
print(classification_report(y_test, y_pred_cnn))
cnn_model.save("cnn_heatwave_final.h5")

# ---------------------------------------------------------
# 🔹 LSTM Model
# ---------------------------------------------------------
print("\n🚀 Training LSTM model...")
X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train_scaled.shape[1]), return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_res, epochs=20, batch_size=32, verbose=1)
y_pred_lstm = (lstm_model.predict(X_test_lstm) > 0.5).astype(int)
print("\n📊 LSTM Evaluation:")
print(confusion_matrix(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm))
lstm_model.save("lstm_heatwave_final.h5")

# ---------------------------------------------------------
# 🔹 XGBoost Model
# ---------------------------------------------------------
print("\n🚀 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)
y_pred_xgb = xgb_model.predict(X_test)
print("\n📊 XGBoost Evaluation:")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
joblib.dump(xgb_model, "xgboost_heatwave_final.pkl")

print("\n🎉 All models trained and saved successfully!")

# ---------------------------------------------------------
# Step 9: Save Test Data for Model Comparison
# ---------------------------------------------------------
import numpy as np

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("✅ Saved test data as X_test.npy and y_test.npy for comparison!")

