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
    total_length = sum(crack_lengths)

    if total_length < 100:
        return "Low"
    elif total_length < 300:
        return "Moderate"
    else:
        return "High"

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    # --- GRAYSCALE ---
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # --- THRESHOLDING ---
    _, thresh = cv2.threshold(
        gray, 150, 255, cv2.THRESH_BINARY_INV
    )

    # --- FIND CONTOURS ---
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crack_lengths = []
    crack_id = 1

    output_img = img_np.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue  # remove noise

        # Draw YELLOW contour
        cv2.drawContours(
            output_img, [cnt], -1, (255, 255, 0), 2
        )

        # Crack length
        length = cv2.arcLength(cnt, True)
        crack_lengths.append(length)

        # Label crack
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(
            output_img,
            f"Crack {crack_id}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2
        )

        crack_id += 1

    severity = calculate_severity(crack_lengths)

    st.image(
        output_img,
        caption="Highlighted Cracks (Yellow)",
        use_column_width=True
    )

    st.subheader("🧮 Crack Measurements")
    for i, l in enumerate(crack_lengths):
        st.write(f"Crack {i+1}: Length = {l:.2f} pixels")

    st.subheader("🚦 Severity Level")
    st.success(severity)
