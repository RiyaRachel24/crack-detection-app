import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Crack Detection App", layout="centered")

st.title("🛣️ Crack Detection & Severity Analysis")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(image)
    probs = results[0].probs

    label = results[0].names[probs.top1]
    confidence = float(probs.top1conf)

    # Simple, stable severity logic
    if confidence >= 0.8:
        severity = "High"
    elif confidence >= 0.5:
        severity = "Medium"
    else:
        severity = "Low"

    st.success(f"🧠 Prediction: **{label}**")
    st.info(f"📊 Confidence: **{confidence:.2f}**")
    st.warning(f"⚠️ Severity: **{severity}**")

