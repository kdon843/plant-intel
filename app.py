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

# app.py (snippet)

   
