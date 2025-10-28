# app.py
import streamlit as st
import numpy as np
import joblib
from pathlib import Path

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="🏠 Ames Housing Price Predictor",
    page_icon="🏡",
    layout="centered",
)

st.title("🏡 Ames Housing Price Predictor")
st.markdown(
    """
    ### Predict house prices in Ames, Iowa  
    Enter key property details below — this model uses 8 key features from the Ames Housing dataset.
    """
)

# -------------------------
# Load Model
# -------------------------
model_path = Path("outputs/Ames_RF_Tuned.pkl")

if not model_path.exists():
    st.error("❌ Model not found in `outputs/`. Please make sure `Ames_RF_Tuned.pkl` exists there.")
    st.stop()

try:
    model = joblib.load(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Could not load model: {e}")
    st.stop()

# -------------------------
# Input Form
# -------------------------
st.header("📋 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    lot_area = st.number_input("Lot Area (sq ft)", 500, 100000, 10000, step=500)
    overall_qual = st.slider("Overall Quality (1-10)", 1, 10, 5)
    year_built = st.number_input("Year Built", 1900, 2025, 2000, step=1)
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", 400, 5000, 1500, step=100)

with col2:
    full_bath = st.number_input("Full Bathrooms", 0, 5, 2)
    bedroom_abv_gr = st.number_input("Bedrooms Above Ground", 1, 10, 3)
    fireplaces = st.number_input("Fireplaces", 0, 5, 1)
    garage_cars = st.number_input("Garage Capacity (Cars)", 0, 6, 2)

# Collect input data
input_data = np.array([[lot_area, overall_qual, year_built, gr_liv_area,
                        full_bath, bedroom_abv_gr, fireplaces, garage_cars]])

# -------------------------
# Prediction Button
# -------------------------
st.markdown("---")
if st.button("🔮 Predict Sale Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🏡 Estimated Sale Price: **${prediction:,.0f}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with ❤️ by Shibu Jaiswal | Powered by Streamlit & Scikit-Learn")
