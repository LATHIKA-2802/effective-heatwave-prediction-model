import pandas as pd
import joblib

# ------------------------------
# Step 1: Load trained model
# ------------------------------
model = joblib.load("C:/Users/lathika/Desktop/data mining project/heatwave_model_effective.pkl")
print("Model loaded successfully!")

# ------------------------------
# Step 2: Load new dataset
# ------------------------------
new_df = pd.read_csv("C:/Users/lathika/Desktop/data mining project/gpt_heatwave.csv")
print("New dataset loaded!")

# Make column names lowercase
new_df.columns = new_df.columns.str.lower()

# ------------------------------
# Step 3: Sort by location and date
# ------------------------------
new_df = new_df.sort_values(by=['lat','lon','mo','dy']).reset_index(drop=True)

# ------------------------------
# Step 4: Feature engineering
# ------------------------------
new_df['heat_index'] = new_df['t2m'] + 0.33*new_df['rh2m'] - 0.70*new_df['ws10m'] - 4.00

# Dynamic thresholds based on new dataset
temp_threshold = new_df['t2m'].quantile(0.95)
humidity_threshold = new_df['rh2m'].quantile(0.95)
new_df['hot_flag'] = ((new_df['t2m'] >= temp_threshold) & (new_df['rh2m'] >= humidity_threshold)).astype(int)

# Consecutive hot days
new_df['consec_hot_days'] = new_df['hot_flag'].groupby(
    (new_df['hot_flag'] != new_df['hot_flag'].shift()).cumsum()
).cumsum()

# 3-day rolling average temperature
new_df['t2m_3day_avg'] = new_df['t2m'].rolling(3, min_periods=1).mean()

# ------------------------------
# Step 5: Prepare features
# ------------------------------
features = ['t2m','rh2m','ws10m','heat_index','consec_hot_days','t2m_3day_avg']
X_new = new_df[features]

# ------------------------------
# Step 6: Predict heatwaves
# ------------------------------
new_df['heatwave_prediction'] = model.predict(X_new)
new_df['heatwave_prob'] = model.predict_proba(X_new)[:,1]

# ------------------------------
# Step 7: Display & save results
# ------------------------------
print(new_df[['lat','lon','mo','dy','t2m','rh2m','heatwave_prediction','heatwave_prob']].head(20))
output_path = "C:/Users/lathika/Desktop/data mining project/new_data_predictions_effective.csv"
new_df.to_csv(output_path, index=False)
print("Predictions saved successfully!")
