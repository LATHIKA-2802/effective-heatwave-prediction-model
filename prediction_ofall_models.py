import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ------------------------------
# Step 1: Load Trained Models
# ------------------------------
print("📦 Loading models...")
ann = load_model("ann_heatwave_final.h5")
cnn = load_model("cnn_heatwave_final.h5")
lstm = load_model("lstm_heatwave_final.h5")
xgb = joblib.load("xgboost_heatwave_final.pkl")
rf = joblib.load("heatwave_model_effective.pkl")
scaler = joblib.load("scaler_final.pkl")
print("✅ All models loaded successfully!")

# ------------------------------
# Step 2: Load New Data
# ------------------------------
new_df = pd.read_csv("gpt_heatwave.csv")
new_df.columns = new_df.columns.str.lower()
new_df = new_df.sort_values(by=['lat','lon','mo','dy']).reset_index(drop=True)

# ------------------------------
# Step 3: Feature Engineering
# ------------------------------
new_df['heat_index'] = new_df['t2m'] + 0.33*new_df['rh2m'] - 0.70*new_df['ws10m'] - 4.00
new_df['consec_hot_days'] = (new_df['t2m'] >= new_df['t2m'].quantile(0.90)).astype(int)
new_df['consec_hot_days'] = new_df['consec_hot_days'].groupby((new_df['consec_hot_days'] != new_df['consec_hot_days'].shift()).cumsum()).cumsum()
new_df['t2m_3day_avg'] = new_df['t2m'].rolling(3, min_periods=1).mean()

features = ['t2m','rh2m','ws10m','heat_index','consec_hot_days','t2m_3day_avg']
X_new = new_df[features]
X_scaled = scaler.transform(X_new)

# CNN & LSTM reshaping
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

# ------------------------------
# Step 4: Predictions
# ------------------------------
new_df['RF_Pred'] = rf.predict(X_scaled)
new_df['ANN_Pred'] = (ann.predict(X_scaled) > 0.5).astype(int)
new_df['CNN_Pred'] = (cnn.predict(X_cnn) > 0.5).astype(int)
new_df['LSTM_Pred'] = (lstm.predict(X_lstm) > 0.5).astype(int)
new_df['XGB_Pred'] = xgb.predict(X_scaled)

# ------------------------------
# Step 5: Save Output
# ------------------------------
save_path = "multi_model_predictions.csv"
new_df.to_csv(save_path, index=False)

print("\n✅ Predictions Completed!")
print(f"📁 Saved results to: {save_path}")

print("\n🔍 Preview of Predictions:")
print(new_df[['t2m','rh2m','ws10m','RF_Pred','ANN_Pred','CNN_Pred','LSTM_Pred','XGB_Pred']].head(20))
