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
import streamlit as st
import recommender  # the module above

st.title("Plant Intel")

@st.cache_resource
def _init_recommender():
    recommender.load()
    return True

_init_recommender()

st.subheader("Text-based recommendation")
col1, col2 = st.columns(2)
with col1:
    host = st.text_input("Host (e.g., tomato, bell pepper)", value="tomato")
with col2:
    disease = st.text_input("Disease (e.g., late blight)", value="late blight")

k = st.slider("How many suggestions?", 1, 5, 3)

if st.button("Get recommendations"):
    try:
        results = recommender.recommend(disease=disease, host=host, k=k)
        if not results:
            st.warning("No matches found. Try a simpler disease name.")
        for r in results:
            st.markdown(f"**{r.get('host','?')} — {r.get('disease','?')}**")
            if r.get("detail_url"):
                st.write(r["detail_url"])
            st.caption(f"Source: {r.get('source','')}")
            st.write(r.get("management_snippet","(no text)"))
            st.divider()
    except Exception as e:
        st.error(f"Recommender error: {e}")

label, score, rec = predict_from_image(img)
label = humanize(label)
st.success(f"Prediction: **{label}** (confidence: {score*100:.1f}%)")

st.divider()
st.caption("© Plant Intel • Educational use. Always follow local agricultural guidance.")
