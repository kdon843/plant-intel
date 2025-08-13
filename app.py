# app.py — Plant Intel (Streamlit)
import os
# Quiet TensorFlow logs and force PyTorch path for transformers (optional but nice)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

# ---- Point recommender to data (local paths OR s3:// URIs). Examples: ----
# Local (Windows):
# os.environ["LENIENT_PARQ"] = r"C:\Users\kerim\Downloads\ucanr_nlp_dataset_lenient.parquet"
# os.environ["PASS_PARQ"]    = r"C:\Users\kerim\Downloads\ucanr_nlp_passages.parquet"
# os.environ["DETAILS_PARQ"] = r"C:\Users\kerim\Downloads\ucanr_details.parquet"
#
# Or S3:
# os.environ["LENIENT_PARQ"] = "s3://capstone-plant-data/Capstone_Data/datasets/nlp/ucanr_nlp_dataset_lenient.parquet"
# os.environ["PASS_PARQ"]    = "s3://capstone-plant-data/Capstone_Data/datasets/nlp/ucanr_nlp_passages.parquet"
# os.environ["DETAILS_PARQ"] = "s3://capstone-plant-data/Capstone_Data/parsed/ucanr_details.parquet"
#
# Optional NLP classifier (SageMaker endpoint) and label map:
# os.environ["NLP_ENDPOINT"]     = "your-nlp-endpoint"
# os.environ["NLP_LABELS_JSON"]  = "s3://capstone-plant-data/Capstone_Data/models/nlp/labels.json"

import streamlit as st
from PIL import Image


import os
# ---- Point recommender to data 
os.environ["LENIENT_PARQ"] = "s3://capstone-plant-data/Capstone_Data/nlp/recs_model/ucanr_nlp_dataset.parquet"
os.environ["PASS_PARQ"]    = "s3://capstone-plant-data/Capstone_Data/datasets/nlp/ucanr_nlp_passages.parquet"
os.environ["DETAILS_PARQ"] = "s3://capstone-plant-data/Capstone_Data/reference/ucanr/ucanr_details.parquet"


import recommender as reco
# handy names
humanize = reco.humanize
recommend_for_disease = reco.recommend_for_disease
recommend_from_text = reco.recommend_from_text
predict_from_text = reco.predict_from_text

st.set_page_config(page_title="Plant Intel", page_icon="🌿", layout="centered")
st.title("Plant Intel")

@st.cache_resource
def _init_reco():
    # Loads CSV/Parquet, builds vocab & (lazy) TF-IDF
    return reco.load()

# Initialize recommender once
try:
    _init_reco()
except Exception as e:
    st.error(f"Failed to load recommender data: {e}")

# ------------- helpers -------------
def render_recs(rows):
    if not rows:
        st.warning("No recommendations found.")
        return
    for r in rows:
        dis = humanize(r.get("disease"))
        host = humanize(r.get("host"))
        st.markdown(f"**{dis or '?'}** on **{host or '?'}**")
        meta = " • ".join(v for v in [r.get("stage"), r.get("source")] if v)
        if meta:
            st.caption(meta)
        if r.get("detail_url"):
            st.write(r["detail_url"])
        st.write(r.get("management_snippet", "(no text)"))
        st.divider()

# ------------- UI -------------
tab_text, tab_disease, tab_image = st.tabs(
    ["Text → Recommend", "Disease → Recommend", "Image Diagnosis"]
)

# -------------------------
# Text → Recommend (NLP endpoint if set; fuzzy fallback otherwise)
# -------------------------
with tab_text:
    st.subheader("Describe the problem")
    text = st.text_area("Example: 'leaf has black spots on rose leaves'", value="")
    col1, col2 = st.columns(2)
    with col1:
        host_hint = st.text_input("Host (optional)", value="")
    with col2:
        k = st.slider("How many suggestions?", 1, 5, 1)

    if st.button("Analyze & Recommend", type="primary"):
        try:
            preds = predict_from_text(text, top_k=max(1, k))
            if preds:
                st.info("Top disease guesses: " + ", ".join(f"{humanize(lbl)} ({score:.0%})" for lbl, score in preds))
            rows = recommend_from_text(text, host_hint or None, k=k)
            render_recs(rows)
        except Exception as e:
            st.error(f"Text analysis error: {e}")
            st.exception(e)

# -------------------------
# Disease → Recommend (direct)
# -------------------------
with tab_disease:
    st.subheader("Recommend by disease")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        host = st.text_input("Host", value="tomato")
    with col2:
        disease = st.text_input("Disease", value="late blight")
    with col3:
        k2 = st.slider("Suggestions", 1, 5, 3, key="k2")

    if st.button("Get recommendations", key="by_disease"):
        try:
            rows = recommend_for_disease(disease, host_hint=host, k=k2)
            render_recs(rows)
        except Exception as e:
            st.error(f"Recommender error: {e}")
            st.exception(e)

# -------------------------
# Image Diagnosis (keeps your models_service flow)
# -------------------------
with tab_image:
    st.subheader("Image-based diagnosis")
    uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
    img = None
    if uploaded is not None:
        try:
            img = Image.open(uploaded)
            st.image(img, caption="Uploaded image", use_container_width=True)
        except Exception:
            st.error("Could not read image. Please upload a valid JPG/PNG.")

    if img is not None and st.button("Predict from image", key="predict_image"):
        try:
            # Your original app used models_service.predict_from_image — keep it if configured
            from models_service import predict_from_image
            label, score, rec = predict_from_image(img)
            st.success(f"Prediction: **{humanize(label)}** (confidence: {score*100:.1f}%)")
            # If your image service returns a disease string, show text-based recs too:
            try:
                rows = recommend_for_disease(label, host_hint=None, k=1)
                if rows:
                    st.caption("Suggested management:")
                    render_recs(rows)
            except Exception:
                pass
        except Exception as e:
            st.error(f"Image prediction error: {e}")
            st.exception(e)

# Optional: small footer with env hints (collapsed)
with st.expander("Runtime info", expanded=False):
    st.write({
        "LENIENT_PARQ": os.getenv("LENIENT_PARQ"),
        "PASS_PARQ": os.getenv("PASS_PARQ"),
        "DETAILS_PARQ": os.getenv("DETAILS_PARQ"),
        "NLP_ENDPOINT": os.getenv("NLP_ENDPOINT"),
        "NLP_LABELS_JSON": os.getenv("NLP_LABELS_JSON"),
    })
