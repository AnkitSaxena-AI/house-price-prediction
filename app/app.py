"""
🏡 King County House Price Predictor — Streamlit demo
Loads the trained XGBoost model and predicts a home's sale price from its attributes.
Run locally:  streamlit run app.py
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL = joblib.load(HERE / "house_price_model.joblib")
ZIPS = pd.read_csv(HERE / "zipcode_lookup.csv")
META = json.load(open(HERE / "model_meta.json"))
FEATURES = META["features"]
ZIP_LOOKUP = ZIPS.set_index("zipcode").to_dict("index")
GLOBAL_ZIP = META.get("global_zip_level", 13.0)
SALE_YEAR = 2015  # dataset covers sales through May 2015

st.set_page_config(page_title="King County House Price Predictor", page_icon="🏡", layout="centered")

st.title("🏡 King County House Price Predictor")
st.caption(
    f"XGBoost model · test R² = {META['r2']:.3f} · typical error ≈ ${META['rmse_dollar']:,.0f}. "
    "Enter a home's details to estimate its sale price (Seattle / King County, 2014–15 market)."
)


def predict_price(inputs: dict) -> float:
    """Build the model's feature vector from user inputs and return the predicted price ($)."""
    z = ZIP_LOOKUP.get(inputs["zipcode"], None)
    lat = z["lat"] if z else 47.5
    lon = z["long"] if z else -122.2
    living15 = z["sqft_living15"] if z else inputs["sqft_living"]
    lot15 = z["sqft_lot15"] if z else inputs["sqft_lot"]
    zip_level = z["zip_price_level"] if z else GLOBAL_ZIP

    sqft_above = max(inputs["sqft_living"] - inputs["sqft_basement"], 0)
    house_age = max(SALE_YEAR - inputs["yr_built"], 0)

    row = {
        "sqft_living": inputs["sqft_living"], "sqft_lot": inputs["sqft_lot"],
        "bedrooms": inputs["bedrooms"], "bathrooms": inputs["bathrooms"],
        "floors": inputs["floors"], "waterfront": inputs["waterfront"],
        "view": inputs["view"], "condition": inputs["condition"], "grade": inputs["grade"],
        "sqft_above": sqft_above, "sqft_basement": inputs["sqft_basement"],
        "house_age": house_age, "renovated": inputs["renovated"],
        "lat": lat, "long": lon, "sqft_living15": living15, "sqft_lot15": lot15,
        "zip_price_level": zip_level,
    }
    X = pd.DataFrame([[row[f] for f in FEATURES]], columns=FEATURES)
    return float(np.expm1(MODEL.predict(X)[0]))


# ---- input form ----
st.subheader("House details")
c1, c2, c3 = st.columns(3)
with c1:
    sqft_living = st.number_input("Living area (sqft)", 300, 14000, 1900, step=50)
    bedrooms = st.number_input("Bedrooms", 0, 12, 3)
    bathrooms = st.number_input("Bathrooms", 0.0, 9.0, 2.0, step=0.25)
    floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], index=0)
with c2:
    sqft_lot = st.number_input("Lot size (sqft)", 400, 200000, 7500, step=100)
    sqft_basement = st.number_input("Basement (sqft)", 0, 5000, 0, step=50)
    grade = st.slider("Construction grade (1–13)", 1, 13, 7)
    condition = st.slider("Condition (1–5)", 1, 5, 3)
with c3:
    zipcode = st.selectbox("Zipcode", sorted(ZIP_LOOKUP.keys()), index=sorted(ZIP_LOOKUP.keys()).index(98103) if 98103 in ZIP_LOOKUP else 0)
    yr_built = st.number_input("Year built", 1900, 2015, 1975)
    view = st.selectbox("View score (0 = none, 4 = excellent)", [0, 1, 2, 3, 4], index=0)
    colw1, colw2 = st.columns(2)
    waterfront = 1 if colw1.checkbox("Waterfront") else 0
    renovated = 1 if colw2.checkbox("Renovated") else 0

if st.button("💰 Estimate price", type="primary", use_container_width=True):
    inp = dict(sqft_living=sqft_living, sqft_lot=sqft_lot, bedrooms=bedrooms, bathrooms=bathrooms,
               floors=floors, waterfront=waterfront, view=view, condition=condition, grade=grade,
               sqft_basement=sqft_basement, yr_built=yr_built, renovated=renovated, zipcode=zipcode)
    price = predict_price(inp)
    lo, hi = price - META["rmse_dollar"], price + META["rmse_dollar"]
    st.markdown(f"## Estimated sale price: **${price:,.0f}**")
    st.caption(f"Typical range (± model RMSE): ${max(lo,0):,.0f} – ${hi:,.0f}")
    st.progress(min(price / 2_000_000, 1.0))

with st.expander("ℹ️ About this model"):
    st.write(
        "Trained on 21,613 King County home sales (May 2014 – May 2015). The model is an XGBoost regressor on "
        "log-price using 18 engineered features. Location features (latitude/longitude, neighbourhood sizes and a "
        "target-encoded zipcode price level) are inferred from the selected zipcode. "
        f"Hold-out performance: R² = {META['r2']:.3f}, RMSE ≈ ${META['rmse_dollar']:,.0f}."
    )
