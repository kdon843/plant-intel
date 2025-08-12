# app.py — Plant Intel (simplified)

import os
import streamlit as st

# Point recommender to your local Windows file BEFORE importing it
os.environ["LENIENT_PARQ"] = r"C:\Users\kerim\Downloads\ucanr_nlp_dataset_lenient.parquet"
# Optional, if you have these locally too:
# os.environ["PASS_PARQ"]    = r"C:\Users\kerim\Downloads\ucanr_nlp_passages.parquet"
# os.environ["DETAILS_PARQ"] = r"C:\Users\kerim\Downloads\ucanr_details.parquet"

import recommender as reco
from PIL import Image  # used in image section

st.set_page_config(page_title="Plant Intel", page_icon="🌿", layout="centered")

@st.cache_resource
def _init_reco():
    # Loads CSV/Parquet, builds vocab & TF-IDF lazily
    return reco.load()

# Initialize recommender once
try:
    _init_reco()
except Exception as e:
    st.error(f"Failed to load recommender data: {e}")

st.title("Plant Intel")

# -------------------------
# Text-based recommendation
# -------------------------
st.subheader("Text-based recommendation")
col1, col2 = st.columns(2)
with col1:
    host = st.text_input("Host", value="tomato")
with col2:
    disease = st.text_input("Disease", value="late blight")

if st.button("Get recommendations"):
    try:
        recs = reco.recommend_for_disease(disease, host_hint=host, k=3)
        if not recs:
            st.warning("No recommendations found.")
        else:
            for r in recs:
                st.markdown(f"**{r.get('host','?')} — {r.get('disease','?')}**")
                stage = r.get("stage")
                source = r.get("source")
                if stage or source:
                    st.caption(" • ".join([v for v in [stage, source] if v]))
                if r.get("detail_url"):
                    st.write(r["detail_url"])
                st.write(r.get("management_snippet", "(no text)"))
                st.divider()
    except Exception as e:
        st.error(f"Recommender error: {e}")
        st.exception(e)

# -------------------------
# Image-based diagnosis
# -------------------------
st.subheader("Image-based diagnosis")
uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
    except Exception:
        st.error("Could not read image. Please upload a valid JPG/PNG.")
        img = None

    if img is not None and st.button("Predict from image"):
        try:
            # Your original app referenced models_service.predict_from_image
            # Keep as-is if you already have that module wired up.
            from models_service import predict_from_image
            label, score, rec = predict_from_image(img)
            st.success(f"Prediction: **{label}** (confidence: {score*100:.1f}%)")
        except Exception as e:
            st.error(f"Image prediction error: {e}")
            st.exception(e)
