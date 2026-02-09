import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("Crack Detection & Severity Analysis")

MODEL_PATH = "best.pt"

MIN_AREA = 1200          # filters texture noise
MIN_ASPECT_RATIO = 3.0   # crack must be elongated
MAX_CRACKS = 2           # avoid over-counting

# ----------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ----------------------------------------
def calculate_severity(boxes):
    if not boxes:
        return "No Crack", "No action required"

    total_length = sum(max(w, h) for (_, _, w, h) in boxes)
    avg_width = np.mean([min(w, h) for (_, _, w, h) in boxes])

    if total_length > 350 and avg_width > 18:
        return "Severe", "Immediate structural repair required"
    elif total_length > 180:
        return "Moderate", "Crack filling and sealing recommended"
    else:
        return "Low", "Monitor periodically"

# ----------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    h, w, _ = img_np.shape

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------- YOLO DETECTION ----------
    results = model(image, conf=0.4)

    if len(results[0].boxes) == 0:
        st.warning("No crack detected in the image.")
        st.stop()

    # ---------- PREPROCESS ----------
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-5)
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue

        boxes.append((x, y, bw, bh))

    if not boxes:
        st.warning("Crack present, but no significant crack regions extracted.")
        st.stop()

    # ---------- SORT & LIMIT ----------
    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    boxes = boxes[:MAX_CRACKS]

    # ---------- DRAW BOXES ----------
    draw_img = img_np.copy()
    features = []

    for i, (x, y, bw, bh) in enumerate(boxes):
        cv2.rectangle(draw_img, (x, y), (x + bw, y + bh), (0, 255, 255), 3)
        cv2.putText(
            draw_img,
            f"Crack {i+1}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        features.append(max(bw, bh))

    st.image(draw_img, caption="Detected Crack Regions", use_column_width=True)

    # ---------- FEATURES ----------
    st.subheader("📏 Extracted Crack Features")
    for i, length in enumerate(features):
        st.write(f"Crack {i+1}: Length ≈ {int(length)} pixels")

    # ---------- SEVERITY ----------
    severity, action = calculate_severity(boxes)

    st.subheader(f"⚠️ Severity: {severity}")
    st.write("**Suggested Action:**")
    st.write(f"- {action}")
