import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("Crack Detection & Severity Analysis")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # MUST be detection model

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_draw = img_np.copy()

    results = model(img_np)[0]

    crack_lengths = []
    crack_count = 0

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Only crack class, confidence filter
            if conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            roi = img_np[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            length = np.count_nonzero(edges)
            if length < 80:
                continue  # removes noise

            crack_count += 1
            crack_lengths.append(length)

            # Draw box
            cv2.rectangle(
                img_draw,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )
            cv2.putText(
                img_draw,
                f"Crack {crack_count}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        st.image(img_draw, caption="Detected Crack Regions", use_column_width=True)

        if crack_count > 0:
            st.subheader("Extracted Crack Features")
            total_length = sum(crack_lengths)

            for i, l in enumerate(crack_lengths, 1):
                st.write(f"Crack {i}: Length ≈ {l} pixels")

            # Severity logic (SIMPLE + DEFENSIBLE)
            if total_length < 300:
                severity = "Low"
                action = ["Monitor periodically"]
            elif total_length < 800:
                severity = "Moderate"
                action = ["Crack filling", "Seal to prevent water ingress"]
            else:
                severity = "High"
                action = ["Structural inspection", "Immediate repair"]

            st.subheader(f"Severity: {severity}")
            st.subheader("Suggested Action")
            for a in action:
                st.write(f"• {a}")

        else:
            st.warning("Crack detected, but no significant regions extracted.")

    else:
        st.info("No cracks detected in the image.")
