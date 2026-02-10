import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crack Detection System", layout="wide")

CONF_THRESHOLD = 0.35   # confidence filter (important)
MIN_BOX_AREA = 800      # removes tiny false boxes

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # make sure best.pt is in same folder

model = load_model()

# ---------------- UI ----------------
st.title("Crack Detection & Severity Analysis")

uploaded_file = st.file_uploader(
    "Upload surface image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- FUNCTIONS ----------------
def classify_severity(max_length):
    """
    Severity logic (panel-safe, explainable):
    - Length < 150 px  → Low
    - 150–350 px       → Moderate
    - > 350 px         → Severe
    """
    if max_length < 150:
        return "Low"
    elif max_length < 350:
        return "Moderate"
    else:
        return "Severe"


# ---------------- MAIN PIPELINE ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # YOLO inference
    results = model(img_np, conf=CONF_THRESHOLD)

    boxes = results[0].boxes

    valid_boxes = []
    crack_lengths = []

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)

            if area < MIN_BOX_AREA:
                continue

            length = max(x2 - x1, y2 - y1)
            crack_lengths.append(length)
            valid_boxes.append((x1, y1, x2, y2, length))

    # ---------------- DISPLAY ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected Cracks")
        annotated = img_np.copy()

        for i, (x1, y1, x2, y2, length) in enumerate(valid_boxes, start=1):
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
        st.error("❌ No cracks detected in the image.")
    else:
        max_len = max(crack_lengths)
        severity = classify_severity(max_len)

        st.subheader("📐 Extracted Crack Features")
        for i, l in enumerate(crack_lengths, start=1):
            st.write(f"Crack {i}: Length ≈ {int(l)} pixels")

        st.markdown("---")
        st.subheader(f"🚨 Severity: **{severity}**")

        if severity == "Low":
            st.info("Suggested Action: Monitor / minor surface sealing.")
        elif severity == "Moderate":
            st.warning("Suggested Action: Crack filling & water-proofing.")
        else:
            st.error("Suggested Action: Structural repair required.")

