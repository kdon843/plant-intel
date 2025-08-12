# app.py — Plant Intel (with data checks & upload fallback)

import os
import shutil
import streamlit as st
from PIL import Image

# --------- Configure where the recommender looks for files ----------
# If you are running LOCALLY and the file is truly on your C: drive, keep this:
os.environ["LENIENT_PARQ"]  = r"C:\Users\kerim\Downloads\ucanr_nlp_dataset_lenient.parquet"
# Optional (set these if you have them locally too)
# os.environ["PASS_PARQ"]     = r"C:\Users\kerim\Downloads\ucanr_nlp_passages.parquet"
# os.environ["DETAILS_PARQ"]  = r"C:\Users\kerim\Downloads\ucanr_details.parquet"

# Local fallback locations inside the project (these match recommender defaults)
LOCAL_LENIENT  = "data/ucanr/parsed/nlp/ucanr_nlp_dataset_lenient.parquet"
LOCAL_PASSAGES = "data/ucanr/parsed/nlp/ucanr_nlp_passages.parquet"
LOCAL_DETAILS  = "data/ucanr/parsed/ucanr_details.parquet"
# Ensure local folders exist for all datasets
os.makedirs(os.path.dirname(p), exist_ok=True


# Helper to read the *effective* paths recommender will use
def _effective_path(env_key: str, local_default: str):
    p = os.getenv(env_key, "").strip()
    return p if p else local_default

E_LENIENT  = _effective_path("LENIENT_PARQ",  LOCAL_LENIENT)
E_PASSAGES = _effective_path("PASS_PARQ",     LOCAL_PASSAGES)
E_DETAILS  = _effective_path("DETAILS_PARQ",  LOCAL_DETAILS)

st.set_page_config(page_title="Plant Intel", page_icon="🌿", layout="centered")
st.title("Plant Intel")

# ------------- Data status & upload fallback (works on Cloud too) -------------
st.subheader("Data status")
c1, c2 = st.columns(2)

with c1:
    st.write("**Expected paths**")
    st.code(f"LENIENT_PARQ : {E_LENIENT}")
    st.code(f"PASS_PARQ    : {E_PASSAGES}")
    st.code(f"DETAILS_PARQ : {E_DETAILS}")

with c2:
    st.write("**File exists?**")
    st.write(f"Lenient : {'✅' if os.path.exists(E_LENIENT) else '❌'}")
    st.write(f"Passages: {'✅' if os.path.exists(E_PASSAGES) else '❌'}")
    st.write(f"Details : {'✅' if os.path.exists(E_DETAILS) else '⚠️ (optional)'}")

# If you’re on Streamlit Cloud or the C: path isn’t reachable, upload here:
with st.expander("🛠️ If missing, upload your lenient dataset (.parquet)"):
    up_len = st.file_uploader("Upload uc anr_nlp_dataset_lenient.parquet", type=["parquet"], key="upl_len")
    dest_choice = st.radio(
        "Save uploaded file to:",
        options=[E_LENIENT, LOCAL_LENIENT],
        index=0 if not E_LENIENT.startswith("C:") else 1,  # default to project path if C: looks remote
    )
    if up_len and st.button("Save file"):
        os.makedirs(os.path.dirname(dest_choice), exist_ok=True)
        with open(dest_choice, "wb") as f:
            f.write(up_len.read())
        st.success(f"Saved file to: {dest_choice}. Click 'Rerun' in the menu to reload.")

# ------------- Initialize recommender only when we have at least one dataset -------------
ready = os.path.exists(E_LENIENT) or os.path.exists(E_PASSAGES) or os.path.exists(E_DETAILS)
if not ready:
    st.warning(
        "No recommender data found yet. "
        "Either make sure the paths above exist on this machine, or upload the lenient dataset."
    )
    st.stop()

# Import and init recommender AFTER the files are in place
import recommender as reco

@st.cache_resource
def _init_reco():
    return reco.load()

try:
    _init_reco()
except Exception as e:
    st.error(f"Failed to load recommender data: {e}")
    st.stop()

# ------------------------- Text-based recommendation -------------------------
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
                meta = " • ".join([v for v in [r.get("stage"), r.get("source")] if v])
                if meta:
                    st.caption(meta)
                if r.get("detail_url"):
                    st.write(r["detail_url"])
                st.write(r.get("management_snippet", "(no text)"))
                st.divider()
    except Exception as e:
        st.error(f"Recommender error: {e}")
        st.exception(e)

# ------------------------- Image-based diagnosis (optional) -------------------------
st.subheader("Image-based diagnosis")
uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"], key="upl_img")

if uploaded is not None:
    try:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
    except Exception:
        st.error("Could not read image. Please upload a valid JPG/PNG.")
        img = None

    if img is not None and st.button("Predict from image"):
        try:
            from models_service import predict_from_image
            label, score, rec = predict_from_image(img)
            st.success(f"Prediction: **{label}** (confidence: {score*100:.1f}%)")
        except Exception as e:
            st.error(f"Image prediction error: {e}")
            st.exception(e)
