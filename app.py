import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heatwave_model_effective.pkl")

st.set_page_config(page_title="Heatwave Predictor", layout="centered")

# Title and description
st.markdown("""
# 🌞 Heatwave Prediction App
Predict whether a day is likely to be a heatwave based on temperature, humidity, and wind speed.
""")

# Input form inside a card
with st.container():
    st.subheader("Enter Weather Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lat = st.number_input("Latitude", value=8.5)
        lon = st.number_input("Longitude", value=76.25)
        mo = st.number_input("Month (1-12)", 1, 12, 1)
    
    with col2:
        dy = st.number_input("Day (1-31)", 1, 31, 1)
        t2m = st.number_input("Temperature (°C)", value=30.0)
        rh2m = st.number_input("Humidity (%)", value=80.0)
    
    with col3:
        ws10m = st.number_input("Wind Speed (m/s)", value=3.0)
    
    st.markdown("---")
    
    if st.button("Predict Heatwave"):
        # Feature engineering
        heat_index = t2m + 0.33*rh2m - 0.70*ws10m - 4.0
        hot_flag = 1 if t2m >= 35 else 0
        consec_hot_days = hot_flag
        t2m_3day_avg = t2m
        
        X_new = pd.DataFrame([[t2m, rh2m, ws10m, heat_index, consec_hot_days, t2m_3day_avg]],
                             columns=['t2m','rh2m','ws10m','heat_index','consec_hot_days','t2m_3day_avg'])
        
        prediction = model.predict(X_new)[0]
        prob = model.predict_proba(X_new)[0,1]
        
        # Result display
        st.markdown("### 🔥 Prediction Result")
        col1, col2 = st.columns(2)
        col1.metric("Heatwave (0=No, 1=Yes)", prediction)
        col2.metric("Probability", f"{prob:.2f}")
        
        # Custom color-coded messages
        if prediction == 1:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#FF8C00; color:white; font-size:18px;">
                Heatwave likely! Take precautions.
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div style="padding:15px; border-radius:10px; background-color:#32cd32 ; color:white; font-size:18px;">
                 No heatwave predicted for this day.
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Show input summary as a card
        st.markdown("### 🌤️ Entered Weather Data")
        st.dataframe(pd.DataFrame({
            "Latitude": [lat],
            "Longitude": [lon],
            "Month": [mo],
            "Day": [dy],
            "Temperature": [t2m],
            "Humidity": [rh2m],
            "Wind Speed": [ws10m]
        }))
