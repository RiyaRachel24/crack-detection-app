import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("Crack Detection & Severity Analysis")

@st.cache_resource
def load_model():
    return YOLO("best.pt")   # detection model ONLY

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    draw = img.copy()

    results = model(img)[0]

    crack_lengths = []
    crack_id = 0

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            conf = float(box.conf[0])

            if conf < 0.25:   # LOWERED, SAFE
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            roi = img[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 40, 120)

            length = np.count_nonzero(edges)
            crack_lengths.append(length)
            crack_id += 1

            cv2.rectangle(draw, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                draw,
                f"Crack {crack_id}",
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        st.image(draw, caption="Detected Crack Regions", use_column_width=True)

        st.subheader("Extracted Crack Features")
        total_length = sum(crack_lengths)

        for i, l in enumerate(crack_lengths, 1):
            st.write(f"Crack {i}: Length ≈ {l} pixels")

        # Severity (DEFENSIBLE RULE-BASED)
        if total_length < 300:
            severity = "Low"
            actions = ["Monitor periodically"]
        elif total_length < 800:
            severity = "Moderate"
            actions = ["Crack filling", "Seal surface to prevent water ingress"]
        else:
            severity = "High"
            actions = ["Structural inspection", "Immediate repair required"]

        st.subheader(f"Severity: {severity}")
        st.subheader("Suggested Action")
        for a in actions:
            st.write(f"• {a}")

    else:
        st.info("No cracks detected in the image.")
