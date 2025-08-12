import io
from PIL import Image
import streamlit as st
from models_service import predict_from_image

st.set_page_config(page_title="Plant Intel", page_icon="🌿")
st.title("Plant Intel")
st.caption("Upload a plant photo **or** describe symptoms in text. Not both.")

mode = st.radio("Choose input type:", ["Image", "Text"], horizontal=True)

if mode == "Image":
    file = st.file_uploader("Upload a leaf/plant photo", type=["jpg","jpeg","png","webp"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Preview", use_container_width=True)
        if st.button("Diagnose image"):
            with st.spinner("Running image model…"):
                label, score, rec = predict_from_image(img)
            st.success(f"Prediction: **{label}**  (confidence: {score:.1%})")
            st.markdown(f"**Recommendation:** {rec}")

elif mode == "Text":
    symptoms = st.text_area("Describe symptoms (e.g., spots, leaf curl, mold, etc.)")
    if st.button("Analyze text", disabled=not symptoms.strip()):
        with st.spinner("Running NLP model…"):
            label, score, rec = predict_from_text(symptoms.strip())
        st.success(f"Predicted disease: **{label}**  (confidence: {score:.1%})")
        st.markdown(f"**Recommendation:** {rec}")

# app.py or models_service.py
import json, pathlib

def load_labels():
    p = pathlib.Path(__file__).with_name("labels.json")
    return json.loads(p.read_text(encoding="utf-8"))

LABELS = load_labels()

def idx_to_name(idx):
    return LABELS[idx] if 0 <= idx < len(LABELS) else f"class_{idx}"

def humanize(label_str):
    if isinstance(label_str, str) and label_str.startswith("class_"):
        try:
            return idx_to_name(int(label_str.split("_")[1]))
        except Exception:
            pass
    return label_str
label, score, rec = predict_from_image(img)
label = humanize(label)
st.success(f"Prediction: **{label}** (confidence: {score*100:.1f}%)")

st.divider()
st.caption("© Plant Intel • Educational use. Always follow local agricultural guidance.")
