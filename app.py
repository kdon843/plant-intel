import streamlit as st
from recommender import recommend_for_disease  # or from models_service import predict_from_image
# app.py
import os
os.environ["LENIENT_PARQ"] = r"C:\Users\kerim\Downloads\ucanr_nlp_dataset_lenient.parquet"
# Optional, if you have them:
# os.environ["PASS_PARQ"]    = r"C:\Users\kerim\Downloads\ucanr_nlp_passages.parquet"
# os.environ["DETAILS_PARQ"] = r"C:\Users\kerim\Downloads\ucanr_details.parquet"

import streamlit as st
import recommender as reco

@st.cache_resource
def _init_reco():
    return reco.load()

_init_reco()

st.title("Plant Intel")

st.subheader("Text-based recommendation")
host = st.text_input("Host", value="tomato")
disease = st.text_input("Disease", value="late blight")

if st.button("Get recommendations"):
    try:
        recs = recommend_for_disease(disease, host_hint=host, k=3)
        if not recs:
            st.warning("No recommendations found.")
        else:
            for r in recs:
                st.markdown(f"**{r.get('host','?')} — {r.get('disease','?')}**")
                if r.get("detail_url"):
                    st.write(r["detail_url"])
                st.write(r.get("management_snippet","(no text)"))
                st.divider()
    except Exception as e:
        st.error(f"Recommender error: {e}")

st.subheader("Image-based diagnosis")
uploaded = st.file_uploader("Upload a leaf photo", type=["jpg","jpeg","png"])

if uploaded is not None:
    from PIL import Image
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image")

    if st.button("Predict from image"):
        try:
            from models_service import predict_from_image
            label, score, rec = predict_from_image(img)
            st.success(f"Prediction: **{label}** (confidence: {score*100:.1f}%)")
        except Exception as e:
            st.error(f"Image prediction error: {e}")
