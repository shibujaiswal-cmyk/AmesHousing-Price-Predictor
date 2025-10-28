# Ames Housing Price Predictor

End-to-end ML project predicting house prices using the Ames Housing dataset.
Includes a Streamlit demo app (local).

Quick start:
1. Put your trained pipeline (Ames_RF_Tuned.pkl) into `outputs/`.
2. Create venv and install requirements: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
# Ames Housing Price Predictor

**End-to-end ML project** — data cleaning, feature engineering, model training (Random Forest), and a local Streamlit demo.  
Author: **Shibu Jaiswal** • MSc Data Science — MAHE

---

## 🚀 Project Summary
This project predicts sale prices of houses in Ames, Iowa using a Random Forest regression model.  
The repo contains:
- Data exploration and preprocessing notebooks (`notebooks/`)  
- Final trained pipeline: `outputs/Ames_RF_Tuned.pkl` (8-feature model)  
- App interface code for local demo: `app.py`  
- Training script: `retrain_8features.py`  
- Requirements: `requirements.txt`

> **Note:** The Streamlit app is runnable locally. If you want a hosted demo link, contact me and I can deploy on Streamlit Cloud or provide a hosted instance.

---

## 🔍 Demo (local)
To run locally (recommended for reviewers and clients):

1. Clone the repository:
```bash
git clone https://github.com/ShibuJaiswal/AmesHousing-Price-Predictor.git
cd AmesHousing-Price-Predictor
