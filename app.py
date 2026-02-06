import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🟡 Crack Detection & Severity Analysis")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

def calculate_severity(crack_lengths):
    if not crack_lengths:
        return "No Crack"

    max_len = max(crack_lengths)

    if max_len < 150:
        return "Low"
    elif max_len < 300:
        return "Moderate"
    else:
        return "High"

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort contours left to right
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    crack_lengths = []
    output = img.copy()
    crack_id = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:   # IMPORTANT: removes tiny noise
            continue

        length = cv2.arcLength(cnt, False)
        crack_lengths.append(length)

        # Draw contour (cleaner)
        cv2.drawContours(output, [cnt], -1, (255, 255, 0), 3)

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(
            output,
            str(crack_id),
            (x + w//2, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        crack_id += 1

    severity = calculate_severity(crack_lengths)

    st.image(output, caption="Detected Cracks", use_column_width=True)

    st.subheader("📏 Crack-wise Lengths")
    for i, l in enumerate(crack_lengths):
        st.write(f"Crack {i+1}: {l:.2f} pixels")

    st.subheader("🚦 Severity")
    st.success(severity)
