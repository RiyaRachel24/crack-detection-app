import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("Crack Detection & Highlighting")

st.write("This step highlights and numbers detected cracks.")

uploaded_file = st.file_uploader(
    "Upload a crack image",
    type=["jpg", "jpeg", "png"]
)

def detect_and_draw_cracks(image):
    # Convert PIL to OpenCV format
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blur, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crack_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Ignore very small regions (noise)
        if area < 100:
            continue

        crack_count += 1

        x, y, w, h = cv2.boundingRect(cnt)

        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Label crack
        cv2.putText(
            img,
            f"Crack {crack_count}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1
        )

    return img, crack_count


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    processed_img, num_cracks = detect_and_draw_cracks(image)

    st.image(processed_img, caption="Highlighted Cracks", use_column_width=True)
    st.success(f"Total cracks detected: {num_cracks}")
