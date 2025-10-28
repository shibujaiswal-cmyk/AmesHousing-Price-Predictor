# ==========================================================
# Ames Housing Price Predictor - Final Version (Local)
# ==========================================================
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

# --- App Title ---
st.set_page_config(page_title="Ames Housing Price Predictor", layout="centered")
st.title("🏡 Ames Housing Price Predictor (Local)")
st.caption("Place your trained pipeline (Ames_RF_Tuned.pkl) in the outputs/ folder and press Predict.")

# --- Load Model ---
model_path = Path("outputs/Ames_RF_Tuned.pkl")

if model_path.exists():
    try:
        pipe = joblib.load(model_path)
        st.success(f"✅ Loaded pipeline: {model_path.name}")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        pipe = None
else:
    st.error("❌ Model file not found. Please place Ames_RF_Tuned.pkl in the outputs/ folder.")
    pipe = None

# --- Define UI for Inputs ---
st.markdown("### 🧮 Enter House Features")

LotArea = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=20000, value=8500)
OverallQual = st.slider("Overall Quality (1–10)", 1, 10, 7)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2023, value=2003)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=4000, value=1800)
GarageCars = st.slider("Garage Cars", 0, 4, 2)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=900)
FullBath = st.slider("Full Bathrooms", 0, 3, 2)
HalfBath = st.slider("Half Bathrooms", 0, 2, 1)

# --- Combine Inputs into DataFrame ---
input_data = pd.DataFrame({
    "LotArea": [LotArea],
    "OverallQual": [OverallQual],
    "YearBuilt": [YearBuilt],
    "GrLivArea": [GrLivArea],
    "GarageCars": [GarageCars],
    "TotalBsmtSF": [TotalBsmtSF],
    "FullBath": [FullBath],
    "HalfBath": [HalfBath]
})

# --- Predict Button ---
if st.button("Predict"):
    if pipe is not None:
        try:
            prediction = pipe.predict(input_data)[0]
            st.success(f"🏠 **Predicted Sale Price:** ${prediction:,.0f}")
        except Exception as e:
            st.error(f"❌ Prediction error: {e}")
    else:
        st.warning("⚠️ Model not loaded. Please check outputs/Ames_RF_Tuned.pkl")

# --- Footer ---
st.markdown("---")
st.markdown("👨‍💻 Developed by **Shibu Jaiswal** | Data Science Portfolio Project (Ames Housing)")
