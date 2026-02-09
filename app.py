import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("🛣️ Crack Detection & Severity Analysis")

# -------------------------------
# Load YOLO Model (Classification)
# -------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your trained model

model = load_model()

# -------------------------------
# Crack Detection using OpenCV
# -------------------------------
def detect_crack_boxes(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    lengths = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        # IMPORTANT FILTER (THIS FIXES FALSE CRACKS)
        if area > 300 and h > 25:
            boxes.append((x, y, w, h))
            lengths.append(max(w, h))

    return boxes, lengths

# -------------------------------
# Severity Logic
# -------------------------------
def calculate_severity(total_length):
    if total_length < 150:
        return "Low", ["Monitor periodically"]
    elif total_length < 400:
        return "Moderate", [
            "Crack filling",
            "Prevent water ingress"
        ]
    else:
        return "High", [
            "Immediate repair required",
            "Structural inspection recommended"
        ]

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # YOLO Prediction (Crack / No Crack)
    # -------------------------------
    yolo_result = model(image)[0]
    probs = yolo_result.probs

    crack_present = False
    if probs is not None:
        crack_present = probs.top1 == 0  # assumes class 0 = crack

    if not crack_present:
        st.warning("No cracks detected by the model.")
    else:
        # -------------------------------
        # OpenCV Crack Detection
        # -------------------------------
        boxes, lengths = detect_crack_boxes(gray)

        if len(boxes) == 0:
            st.warning("Crack present, but no significant crack regions extracted.")
        else:
            vis = img_np.copy()
            total_length = sum(lengths)

            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(
                    vis, (x, y), (x + w, y + h),
                    (255, 255, 0), 2
                )
                cv2.putText(
                    vis,
                    f"{i+1}",
                    (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            st.image(
                vis,
                caption="Detected Crack Regions (Bounding Boxes)",
                use_column_width=True
            )

            # -------------------------------
            # Features Display
            # -------------------------------
            st.subheader("📏 Extracted Crack Features")
            for i, length in enumerate(lengths):
                st.write(f"Crack {i+1} → Length: {length} pixels")

            # -------------------------------
            # Severity
            # -------------------------------
            severity, actions = calculate_severity(total_length)

            st.subheader(f"🚨 Severity: {severity}")
            st.write(f"Total Crack Length: {total_length} pixels")

            st.subheader("🛠 Suggested Action")
            for act in actions:
                st.write(f"- {act}")
