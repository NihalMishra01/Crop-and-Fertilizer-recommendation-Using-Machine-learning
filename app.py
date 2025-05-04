import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie

# CSS for larger text, bold black labels, and white input text for N, P, K
st.markdown(
    """
    <style>
    /* Background and app container */
    html, body, .stApp, .block-container {
        background-color: #f4fff4 !important;
    }

    /* Centered title AgriSmart */
    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }

    .title-container h1 {
        font-size: 48px !important;
        font-family: 'Trebuchet MS', sans-serif;
        color: #2e7d32 !important;
        font-weight: 900 !important;
        padding: 10px 20px;
        border-radius: 12px;
        display: inline-block;
        background: linear-gradient(90deg, #e8f5e9, #c8e6c9);
        box-shadow: 0 0 10px rgba(76, 175, 80, 0.2);
    }

    .subtitle {
        font-size: 24px !important;
        text-align: center;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        color: #1b5e20 !important;
        margin-bottom: 40px;
    }

    /* Headings and labels */
    h1, h2, h3, h4, h5, h6, label, .stTextInput>label, .stNumberInput>label, .stSlider>label, .stSelectbox>label, .stMarkdown, .stButton>button {
        color: #000 !important;
        font-weight: 900 !important;
    }

    /* Input labels (e.g., 🌡️ Temperature) */
    .stSlider > label, .stNumberInput > label, .stSelectbox > label, .stTextInput > label {
        font-size: 28px !important;
        font-weight: 900 !important;
        color: #1b5e20 !important;
    }

    /* Input field values and slider min-max */
    input[type="number"], input[type="text"], .stSlider span {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #000000 !important;
    }

    .stSlider > div[data-baseweb="slider"] > div > div {
        font-size: 20px !important;
        font-weight: 700 !important;
        color: #4caf50 !important;
    }

    /* Slider/input height & size */
    .stSlider > div[data-baseweb="slider"] {
        min-height: 70px !important;
    }
    .stNumberInput, .stSelectbox, .stTextInput {
        min-height: 60px !important;
        font-size: 1.8rem !important;
    }

    /* Get Recommendation button */
    .stButton>button {
        background-color: #2e7d32 !important;
        color: #fff !important;
        font-weight: 900 !important;
        border-radius: 16px !important;
        padding: 22px 0 !important;
        min-height: 64px !important;
        font-size: 1.3rem !important;
    }

    .stButton>button:hover {
        background-color: #1b5e20 !important;
        color: #fff !important;
        min-height: 64px !important;
    }

    /* Metric output styling */
    .stMetricLabel, .stMetricValue, div[data-testid="stMetric"] > div > label, div[data-testid="stMetric"] > div > div {
        color: #000 !important;
        font-weight: 900 !important;
    }

    /* Output box */
    .output-box {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(67,160,71,0.10);
        padding: 40px 32px;
        margin: 32px 0;
        color: #000 !important;
        font-weight: 900 !important;
        border: 1.5px solid #e0e0e0;
        text-align: center;
        min-height: 80px;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 36px;
        background-color: #e8f5e9;
        border-radius: 20px;
        margin-top: 50px;
        border: 1.5px solid #c8e6c9;
        color: #000 !important;
        font-weight: 900 !important;
        width: 100%;
    }

    /* White input text for N, P, K number inputs */
    input[aria-label="🌱 Nitrogen (N)"] {
        color: #fff !important;
        background: #388e3c !important;
        font-weight: 900 !important;
    }
    input[aria-label="🪨 Phosphorous (P)"] {
        color: #fff !important;
        background: #388e3c !important;
        font-weight: 900 !important;
    }
    input[aria-label="🥔 Potassium (K)"] {
        color: #fff !important;
        background: #388e3c !important;
        font-weight: 900 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title HTML (place this where you want title to appear)
st.markdown("""
<div class="title-container">
    <h1>AgriSmart</h1>
</div>
<div class="subtitle">🌿 Smart Crop & Fertilizer Recommendation System 🌾</div>
""", unsafe_allow_html=True)


# Lottie animation loader
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_plant = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_2glqweqs.json")
lottie_leaf = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_4kx2q32n.json")

# Load models and preprocessing objects
try:
    crop_model = joblib.load('crop_recommendation_model.pkl')
    crop_scaler = joblib.load('crop_feature_scaler.pkl')
    crop_label_encoder = joblib.load('crop_label_encoder.pkl')
    crop_features = joblib.load('crop_features.pkl')
    fertilizer_model = joblib.load('fertilizer_recommendation_model.pkl')
    fertilizer_scaler = joblib.load('fertilizer_feature_scaler.pkl')
    fertilizer_encoders = joblib.load('fertilizer_label_encoders.pkl')
    fertilizer_features = joblib.load('fertilizer_features.pkl')
except Exception as e:
    st.error(f"❌ Error loading models: {str(e)}")
    st.stop()

def predict_crop_and_fertilizer():
    try:
        crop_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        crop_input_df = pd.DataFrame(crop_input, columns=crop_features)
        scaled_crop_input = crop_scaler.transform(crop_input_df)
        crop_encoded = crop_model.predict(scaled_crop_input)[0]
        crop_name = crop_label_encoder.inverse_transform([crop_encoded])[0]
        soil_type_map = {'Sandy': 0, 'Loamy': 1, 'Black': 2, 'Red': 3, 'Clayey': 4}
        soil_type_num = soil_type_map.get(soil_type, 0)
        fert_input = np.array([[temperature, humidity, 40, soil_type_num, 0, N, K, P]])
        fert_input_df = pd.DataFrame(fert_input, columns=fertilizer_features)
        scaled_fert_input = fertilizer_scaler.transform(fert_input_df)
        fert_encoded = fertilizer_model.predict(scaled_fert_input)[0]
        fertilizer_map = {
            0: 'Urea', 1: 'DAP', 2: '14-35-14', 3: '28-28', 4: '17-17-17', 5: '20-20', 6: '10-26-26'
        }
        fertilizer_name = fertilizer_map.get(fert_encoded, 'Unknown')
        fertilizer_descriptions = {
            'Urea': 'Nitrogen-rich fertilizer (46% N)',
            'DAP': 'Diammonium Phosphate (18% N, 46% P2O5)',
            '14-35-14': 'NPK fertilizer (14% N, 35% P2O5, 14% K2O)',
            '28-28': 'NP fertilizer (28% N, 28% P2O5)',
            '17-17-17': 'Balanced NPK fertilizer (17% N, 17% P2O5, 17% K2O)',
            '20-20': 'NP fertilizer (20% N, 20% P2O5)',
            '10-26-26': 'NPK fertilizer (10% N, 26% P2O5, 26% K2O)'
        }
        return crop_name, fertilizer_name, fertilizer_descriptions.get(fertilizer_name, '')
    except Exception as e:
        st.error(f"❌ Error in prediction: {str(e)}")
        return None, None, None

# Input Section
st.header("🌡️ Environmental & Soil Inputs")
col1, col2 = st.columns(2)
with col1:
    temperature = st.slider("🌡️ Temperature (°C)", 10.0, 50.0, 20.0)
    rainfall = st.slider("💧 Rainfall (mm)", 20.0, 300.0, 202.0)
    humidity = st.slider("💦 Humidity (%)", 10.0, 100.0, 82.0)
    ph = st.slider("🧪 Soil pH", 3.5, 9.5, 6.5)
with col2:
    N = st.number_input("🌱 Nitrogen (N)", min_value=0, max_value=140, value=90)
    P = st.number_input("🪨 Phosphorous (P)", min_value=5, max_value=145, value=42)
    K = st.number_input("🥔 Potassium (K)", min_value=5, max_value=205, value=43)
    soil_type = st.selectbox("🧑‍🌾 Soil Type", ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])

# Soil Analysis Section
st.header("📊 Current Soil Analysis")
st_lottie(lottie_leaf, height=80, key="soil-leaf")
col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Nitrogen", f"{N} kg/ha")
with col4:
    st.metric("Phosphorous", f"{P} kg/ha")
with col5:
    st.metric("Potassium", f"{K} kg/ha")

# Recommendation Section
st.header("🌿 Recommendations")
if st.button("Get Recommendations", key="recommend_button"):
    if not (0 <= N <= 140 and 5 <= P <= 145 and 5 <= K <= 205 and 10 <= temperature <= 50 and 10 <= humidity <= 100 and 3.5 <= ph <= 9.5 and 20 <= rainfall <= 300):
        st.error("❌ Please ensure all input values are within their specified ranges")
        st.stop()
    crop_name, fertilizer_name, fertilizer_desc = predict_crop_and_fertilizer()
    if crop_name and fertilizer_name:
        st_lottie(lottie_plant, height=80, key="rec-plant")
        st.markdown(f"""
        <div class='output-box'>🌱 Recommended Crop: <b>{crop_name.capitalize()}</b></div>
        """, unsafe_allow_html=True)
        st_lottie(lottie_leaf, height=60, key="rec-leaf")
        st.markdown(f"""
        <div class='output-box'>💊 Recommended Fertilizer: <b>{fertilizer_name}</b><br>{fertilizer_desc}</div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>AgriSmart - Smart Agriculture Solutions</p>
    <p>© 2024 All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)