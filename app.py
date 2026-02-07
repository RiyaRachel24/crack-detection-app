import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🧱 Crack Detection & Severity Analysis")

MODEL_PATH = "best.pt"

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ---------------- SEVERITY LOGIC ----------------
def get_severity(length_px):
    if length_px < 150:
        return "Low"
    elif length_px < 400:
        return "Moderate"
    else:
        return "High"

def get_suggestions(severity):
    if severity == "Low":
        return ["Monitor periodically", "Seal surface if required"]
    elif severity == "Moderate":
        return ["Crack filling", "Prevent water ingress"]
    else:
        return ["Structural inspection", "Immediate repair required"]

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload a crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)

    # ---------------- YOLO INFERENCE ----------------
    results = model(img_np)[0]

    draw = ImageDraw.Draw(image)
    crack_count = 0
    lengths = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        # Class 0 assumed = crack
        if cls == 0 and conf > 0.4:
            crack_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            length_px = x2 - x1
            lengths.append(length_px)

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline="yellow", width=3)
            draw.text((x1, y1 - 10), f"{crack_count}", fill="yellow")

    st.image(image, caption="Detected Cracks", use_column_width=True)

    # ---------------- RESULTS ----------------
    if crack_count == 0:
        st.warning("No cracks detected.")
    else:
        max_length = max(lengths)
        severity = get_severity(max_length)

        st.subheader("📏 Extracted Crack Features")
        for i, l in enumerate(lengths):
            st.write(f"Crack {i+1}: Length = {l} pixels")

        st.subheader(f"🚨 Severity: {severity}")

        st.subheader("🛠 Suggested Action")
        for s in get_suggestions(severity):
            st.write(f"- {s}")
