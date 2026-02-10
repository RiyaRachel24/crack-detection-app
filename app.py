import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection System", layout="wide")

# VERY IMPORTANT: relaxed confidence
CONF_THRESHOLD = 0.15   # <-- THIS FIXES YOUR ISSUE

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- UI ----------------
st.title("🧱 Crack Detection & Severity Analysis")

uploaded_file = st.file_uploader(
    "Upload surface image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- SEVERITY LOGIC ----------------
def classify_severity(length_px):
    if length_px < 120:
        return "Low"
    elif length_px < 300:
        return "Moderate"
    else:
        return "Severe"

# ---------------- MAIN ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = model(img_np, conf=CONF_THRESHOLD)

    boxes = results[0].boxes

    crack_boxes = []
    crack_lengths = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            length = max(x2 - x1, y2 - y1)

            crack_boxes.append((x1, y1, x2, y2, length))
            crack_lengths.append(length)

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected Cracks")
        annotated = img_np.copy()

        for i, (x1, y1, x2, y2, length) in enumerate(crack_boxes, start=1):
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )
            cv2.putText(
                annotated,
                f"{i}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        st.image(annotated, use_column_width=True)

    # ---------------- RESULTS ----------------
    st.markdown("---")

    if len(crack_lengths) == 0:
        st.error("❌ No cracks detected.")
    else:
        max_len = max(crack_lengths)
        severity = classify_severity(max_len)

        st.subheader("📐 Extracted Crack Features")
        for i, l in enumerate(crack_lengths, start=1):
            st.write(f"Crack {i}: Length ≈ {int(l)} pixels")

        st.markdown("---")
        st.subheader(f"🚨 Severity: **{severity}**")

        if severity == "Low":
            st.info("Suggested Action: Monitoring / surface sealing.")
        elif severity == "Moderate":
            st.warning("Suggested Action: Crack filling & waterproofing.")
        else:
            st.error("Suggested Action: Structural repair required.")
