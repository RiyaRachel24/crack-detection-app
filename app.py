import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("Crack Detection & Severity Analysis")

# -------------------------------
# Severity logic (FINAL)
# -------------------------------
def calculate_severity(length_px, width_px):
    score = (0.6 * length_px) + (0.4 * width_px)

    if score < 300:
        return "Low", score
    elif score < 700:
        return "Moderate", score
    else:
        return "Severe", score


# -------------------------------
# Image uploader
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload crack / non-crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Preprocessing
    # -------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # -------------------------------
    # Contour detection
    # -------------------------------
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crack_count = 0
    total_length = 0
    total_width = 0

    output_img = img.copy()

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        # ❗ Filter noise
        if area < 100 or h < 20:
            continue

        crack_count += 1
        length_px = max(w, h)
        width_px = min(w, h)

        total_length += length_px
        total_width += width_px

        # Draw BOX (clean, panel-safe)
        cv2.rectangle(
            output_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            2
        )

        cv2.putText(
            output_img,
            f"{crack_count}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    # -------------------------------
    # Results
    # -------------------------------
    st.subheader("Detected Crack Regions")
    st.image(output_img, use_column_width=True)

    if crack_count == 0:
        st.success("No cracks detected in the image.")
    else:
        avg_width = total_width / crack_count
        severity, score = calculate_severity(total_length, avg_width)

        st.subheader("📏 Extracted Crack Features")
        st.markdown(f"- **Number of cracks:** {crack_count}")
        st.markdown(f"- **Total crack length:** `{total_length}` pixels")
        st.markdown(f"- **Average crack width:** `{round(avg_width, 2)}` pixels")

        st.subheader("🔥 Severity Assessment")
        st.markdown(f"### **Severity: {severity}**")
        st.markdown(f"- Severity score: `{round(score, 2)}`")

        st.subheader("🛠 Suggested Action")
        if severity == "Low":
            st.info("Monitor periodically. Cosmetic repair if required.")
        elif severity == "Moderate":
            st.warning("Crack filling and sealing recommended to prevent water ingress.")
        else:
            st.error("Immediate structural inspection and professional repair required.")
