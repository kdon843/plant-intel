# app.py — Plant Intel (Streamlit)
# ------------------------------------------------------------
# Features
# - Text recommender (local, no endpoint required)
# - Optional image diagnosis via SageMaker endpoint (uses st.secrets)
# - QA expander to run sanity tests on host+disease pairs
# ------------------------------------------------------------

import os
import io
import json
import traceback
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
from PIL import Image

# Local module (use the full recommender.py I sent)
import recommender as reco

# =============== App Config ===============
st.set_page_config(page_title="Plant Intel", page_icon="🌿", layout="wide")

# =============== Secrets / Config ===============
# Prefer Streamlit Secrets in cloud; fallback to env vars for local dev
AWS_REGION   = st.secrets.get("AWS_REGION", os.getenv("AWS_REGION", "us-east-1"))
IMG_ENDPOINT = st.secrets.get("IMG_ENDPOINT", os.getenv("IMG_ENDPOINT", ""))  # e.g. "plant-resnet50-prod"
USE_IMAGE_API = bool(IMG_ENDPOINT)

# =============== Helpers ===============
@st.cache_resource
def _init_reco():
    # Loads CSV/Parquet, builds vocab & TF-IDF lazily
    return reco.load()

def _fmt_result_card(r: Dict):
    title = f"**{r.get('host','?')} — {r.get('disease','?')}**"
    meta  = " • ".join([x for x in [r.get("stage",""), r.get("source","")] if x])
    if meta:
        st.caption(meta)
    st.markdown(title)
    if r.get("detail_url"):
        st.write(r["detail_url"])
    st.write(r.get("management_snippet","(no text)"))
    st.divider()

def _get_boto3_runtime(region: str):
    """
    Returns a boto3 sagemaker-runtime client if credentials are configured,
    otherwise raises a helpful error.
    """
    try:
        import boto3
        # If running in Streamlit Cloud, put keys in st.secrets
        if "aws_access_key_id" in st.secrets and "aws_secret_access_key" in st.secrets:
            sess = boto3.session.Session(
                aws_access_key_id=st.secrets["aws_access_key_id"],
                aws_secret_access_key=st.secrets["aws_secret_access_key"],
                region_name=region,
            )
            return sess.client("sagemaker-runtime")
        # Else try default credential chain (EC2/ECS/IAM role or local ~/.aws)
        return boto3.client("sagemaker-runtime", region_name=region)
    except Exception as e:
        raise RuntimeError(
            "AWS credentials not found. In Streamlit Cloud, add them in Settings → Secrets "
            "(aws_access_key_id / aws_secret_access_key)."
        ) from e

def predict_image_with_endpoint(pil_img: Image.Image, endpoint: str, region: str) -> Dict:
    """
    Sends a JPEG to your SageMaker endpoint and returns parsed JSON.
    Your model handler should return something like:
      {"label": "class_36", "score": 0.937, "topk": [...]}
    """
    runtime = _get_boto3_runtime(region)
    buf = io.BytesIO()
    # JPEG recommended unless your handler expects PNG
    pil_img.convert("RGB").save(buf, format="JPEG", quality=90)
    payload = buf.getvalue()

    resp = runtime.invoke_endpoint(
        EndpointName=endpoint,
        ContentType="image/jpeg",
        Body=payload
    )
    body = resp["Body"].read()
    try:
        return json.loads(body)
    except Exception:
        # Fallback: return raw string if model returns plain text
        return {"raw": body.decode("utf-8", errors="ignore")}

# =============== UI ===============
st.title("🌿 Plant Intel")
st.write("Diagnose plant diseases by **text** or **image**, and get management recommendations.")

# Show environment status in sidebar
with st.sidebar:
    st.header("Status")
    _ok = _init_reco()
    st.success("Recommender loaded") if _ok else st.warning("Recommender not loaded")
    if USE_IMAGE_API:
        st.info(f"Image endpoint: **{IMG_ENDPOINT}** ({AWS_REGION})")
    else:
        st.warning("Image endpoint not configured")

tab_text, tab_image, tab_qa = st.tabs(["🔤 Diagnose by Text", "🖼️ Diagnose by Image", "🧪 QA / Admin"])

# =============== Tab: Text ===============
with tab_text:
    st.subheader("Text-based recommendation")
    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input("Host (e.g., tomato, bell pepper, grape)", value="tomato")
    with col2:
        disease = st.text_input("Disease (e.g., late blight, black spot)", value="late blight")

    k = st.slider("How many suggestions?", min_value=1, max_value=5, value=3, step=1)
    allow_other_hosts = st.toggle("Allow other hosts if strict match fails", value=True)

    go = st.button("Get recommendations", type="primary")
    if go:
        try:
            results = reco.recommend_for_disease(
                disease=disease,
                host_hint=host,
                k=k,
                allow_other_hosts=allow_other_hosts,
            )
            if not results:
                st.warning("No matches found. Try a simpler disease name or enable other-host fallback.")
            else:
                for r in results:
                    _fmt_result_card(r)
        except Exception as e:
            st.error(f"Recommender error: {e}")
            st.exception(e)

# =============== Tab: Image ===============
with tab_image:
    st.subheader("Image-based diagnosis (optional SageMaker endpoint)")
    st.caption("Configure `IMG_ENDPOINT` and AWS credentials via Streamlit Secrets to enable this feature.")
    uploaded = st.file_uploader("Upload a leaf photo (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        tr
