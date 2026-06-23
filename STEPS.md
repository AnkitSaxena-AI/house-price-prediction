# 🚀 Deploy the House Price Predictor (Streamlit Community Cloud)

The trained model and demo live in `app/`. You can run the demo locally in seconds, or deploy it free on
Streamlit Community Cloud so anyone can try it from a public URL.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app loads `app/house_price_model.joblib` + `app/zipcode_lookup.csv` and predicts a sale price from the inputs.

## Deploy free on Streamlit Community Cloud

1. Push this repository to GitHub (already done if you're reading this on GitHub).
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `AnkitSaxena-AI/house-price-prediction`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
5. Click **Deploy**. The first build installs the dependencies (a couple of minutes) and your app goes live at a
   `…streamlit.app` URL.
6. Copy that URL into the README's **Live Demo** badge.

### Notes
- Streamlit Cloud installs from the repo-root `requirements.txt`. It includes `tensorflow-cpu` (used only by the
  notebook's neural-network section). If you want a leaner/faster build for the app alone, you can deploy a branch
  whose `requirements.txt` lists only: `streamlit`, `xgboost`, `scikit-learn`, `joblib`, `pandas`, `numpy`.
- The model file is ~1.5 MB, well within Streamlit Cloud limits — no Git LFS needed.
- To update the deployed app, just push to `main`; Streamlit redeploys automatically.
