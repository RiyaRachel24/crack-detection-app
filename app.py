import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if file is None:
    st.stop()

# ---------------- IMAGE PROCESS ----------------
image = Image.open(file).convert("RGB")
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CRACK DETECTION ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

H, W = gray.shape

valid_cracks = []
widths = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    area = cv2.contourArea(cnt)
    length = max(w, h)
    width = min(w, h)

    # Reject small noisy blobs
    if area < 100:
        continue

    # Crack must be long
    if length < 40:
        continue

    # Crack must be thin
    if width > 40:
        continue

    valid_cracks.append((x, y, w, h))
    widths.append(width)

# ---------------- NO CRACK ----------------
if len(valid_cracks) == 0:
    st.error("No cracks detected.")
    st.image(image, use_column_width=True)
    st.stop()

# ---------------- DRAW RESULTS ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Detected Cracks")
    annotated = img.copy()

    for (x, y, w, h) in valid_cracks:
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,255), 2)

    st.image(annotated, use_column_width=True)

# ---------------- WIDTH CALCULATION ----------------
max_width_pixels = max(widths)

# ⚠️ pixel → mm conversion (adjust if needed)
PIXEL_TO_MM = 0.01
max_width_mm = max_width_pixels * PIXEL_TO_MM

# ---------------- SEVERITY INDEX ----------------
SI = max_width_mm / 0.30

if SI <= 0.33:
    severity = "🟢 Minor"
elif SI <= 1.0:
    severity = "🟡 Moderate"
else:
    severity = "🔴 Severe"

# ---------------- DISPLAY METRICS ----------------
st.markdown("---")
st.subheader("Crack Measurements")

st.write(f"Max crack width (pixels): {max_width_pixels:.2f}")
st.write(f"Estimated crack width (mm): {max_width_mm:.3f}")
st.write(f"Severity Index (SI): {SI:.2f}")

st.markdown("---")
st.subheader(f"Severity: {severity}")

# ---------------- ACTION ----------------
st.subheader("Suggested Action")

if SI <= 0.33:
    st.write("• Monitor periodically\n• Surface sealing")
elif SI <= 1.0:
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection required\n• Professional repair recommended")


