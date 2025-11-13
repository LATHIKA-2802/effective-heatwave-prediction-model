import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ------------------------------
# Step 1: Load dataset
# ------------------------------
df = pd.read_csv("C:/Users/lathika/Desktop/data mining project/combined_heatwave_data.csv")
print("Dataset head:\n", df.head())
print("\nMissing values:\n", df.isnull().sum())

# ------------------------------
# Step 2: Define heatwave dynamically
# ------------------------------
temp_threshold = df['t2m'].quantile(0.95)  # top 5% hottest days
humidity_threshold = 70  # adjust if needed
df['heatwave'] = ((df['t2m'] >= temp_threshold) & (df['rh2m'] >= humidity_threshold)).astype(int)

print("\nHeatwave class distribution:\n", df['heatwave'].value_counts())
if df['heatwave'].nunique() < 2:
    raise ValueError("Not enough heatwave days detected. Lower thresholds!")

# ------------------------------
# Step 3: Feature engineering
# ------------------------------
df['consec_hot_days'] = (df['heatwave'] == 1).astype(int)
df['consec_hot_days'] = df['consec_hot_days'].groupby(
    (df['consec_hot_days'] != df['consec_hot_days'].shift()).cumsum()
).cumsum()

df['heat_index'] = df['t2m'] + 0.33*df['rh2m'] - 0.70*df['ws10m'] - 4.00
df['t2m_3day_avg'] = df['t2m'].rolling(3, min_periods=1).mean()

# ------------------------------
# Step 4: Prepare features & target
# ------------------------------
features = ['t2m','rh2m','ws10m','heat_index','consec_hot_days','t2m_3day_avg']
X = df[features]
y = df['heatwave']

# Time-based split
train = df[df['YEAR'] <= df['YEAR'].max()-1]
test = df[df['YEAR'] > df['YEAR'].max()-1]

X_train = train[features]
y_train = train['heatwave']
X_test = test[features]
y_test = test['heatwave']

# ------------------------------
# Step 5: Handle class imbalance with SMOTE
# ------------------------------
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nBefore SMOTE:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", pd.Series(y_train_res).value_counts())

# ------------------------------
# Step 6: Train Random Forest
# ------------------------------
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train_res, y_train_res)

# Predict test set
y_pred = model.predict(X_test)

# ------------------------------
# Step 7: Evaluate model
# ------------------------------
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ------------------------------
# Step 8: Save model
# ------------------------------
joblib.dump(model, "heatwave_model_effective.pkl")
print("\nModel saved as heatwave_model_effective.pkl")
