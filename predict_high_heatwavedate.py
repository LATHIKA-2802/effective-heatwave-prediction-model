import pandas as pd

# Load dataset
df = pd.read_csv("combined_heatwave_data.csv")

# Convert column names to lowercase
df.columns = df.columns.str.lower()

# ------------------------------
# STEP 1: DEFINE HEATWAVE (same logic you used in model training)
# ------------------------------
temp_threshold = df['t2m'].quantile(0.95)   # Top 5% hottest temperature
humidity_threshold = 70                     # Fixed humidity threshold
df['heatwave'] = ((df['t2m'] >= temp_threshold) & (df['rh2m'] >= humidity_threshold)).astype(int)

# ------------------------------
# STEP 2: Extract only heatwave days
# ------------------------------
heatwave_days = df[df['heatwave'] == 1][['year','mo','dy','t2m','rh2m','ws10m']]

print("\n🔥 Heatwave Days Identified:\n")
print(heatwave_days)

# ------------------------------
# STEP 3: Count heatwave occurrences per month-Year
# ------------------------------
heatwave_count = heatwave_days.groupby(['year','mo']).size().reset_index(name='heatwave_occurences')

print("\n📊 Heatwave Frequency Per Month:\n")
print(heatwave_count)

# ------------------------------
# STEP 4: Find Month & Year with Maximum Heatwaves
# ------------------------------
if len(heatwave_count) > 0:
    peak = heatwave_count.loc[heatwave_count['heatwave_occurences'].idxmax()]
    print(f"\n🏆 Most Severe Heatwave Period:")
    print(f"   Year: {int(peak.year)}, Month: {int(peak.mo)}, Heatwave occurences: {peak.heatwave_occurences}")
else:
    print("\n⚠️ No heatwaves detected with the selected threshold. Try lowering thresholds.\n")
