import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection", layout="wide")
st.title("Crack Detection & Severity Analysis")

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if file is None:
    st.stop()

image = Image.open(file).convert("RGB")
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CRACK DETECTION ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Sensitive edge detection (captures thin cracks)
edges = cv2.Canny(blur, 30, 100)

# Connect broken crack segments
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=2)

contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cracks = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    length = max(w, h)
    width = min(w, h)

    # remove tiny noise
    if length < 30:
        continue

    # remove blobs (not crack-like)
    if width > 0.5 * length:
        continue

    cracks.append((x, y, w, h, length, width))

# ---------------- NO CRACK ----------------
if len(cracks) == 0:
    st.error("No cracks detected.")
    st.image(image, use_container_width=True)
    st.stop()

# ---------------- DRAW ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

with col2:
    st.subheader("Detected Cracks")

    annotated = img.copy()

    for i, (x, y, w, h, length, width) in enumerate(cracks, start=1):
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,255), 3)
        cv2.putText(
            annotated,
            f"{i}",
            (x, y-8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,255),
            2
        )

    st.image(annotated, use_container_width=True)

# ---------------- FEATURE EXTRACTION ----------------
st.subheader("Extracted Crack Features")

lengths = []
widths = []

for i, (_, _, _, _, L, W) in enumerate(cracks, start=1):
    lengths.append(L)
    widths.append(W)
    st.write(f"Crack {i} → Length ≈ {int(L)} px | Width ≈ {int(W)} px")

# ---------------- WIDTH CONVERSION ----------------
st.markdown("---")
st.subheader("Width Conversion")

pixel_to_mm = st.number_input(
    "Pixel to mm scale (mm per pixel)",
    value=0.01,
    step=0.001,
    help="Camera calibration factor"
)

max_width_px = max(widths)
max_width_mm = max_width_px * pixel_to_mm

st.write(f"Max crack width = {max_width_mm:.3f} mm")

# ---------------- SEVERITY INDEX ----------------
st.markdown("---")
st.subheader("Severity Analysis")

SI = max_width_mm / 0.30

if max_width_mm <= 0.10:
    severity = "Minor"
elif max_width_mm <= 0.30:
    severity = "Moderate"
else:
    severity = "Severe"

st.write(f"Severity Index (SI) = {SI:.2f}")
st.success(f"Severity = {severity}")

# ---------------- ACTION ----------------
st.markdown("---")
st.subheader("Suggested Action")

if severity == "Minor":
    st.write("• Surface sealing\n• Monitoring")
elif severity == "Moderate":
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection\n• Immediate repair required")
