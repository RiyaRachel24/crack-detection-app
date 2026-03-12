import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])
if file is None:
    st.stop()

# ---------------- LOAD IMAGE ----------------
image = Image.open(file).convert("RGB")
img = np.array(image)
h_img, w_img = img.shape[:2]
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CONTRAST ENHANCEMENT ----------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# ---------------- DENOISE ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------- OTSU THRESHOLD ----------------
_, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ---------------- MORPHOLOGY: connect crack segments ----------------
# Use a thin horizontal+vertical kernel to link crack pixels
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)

# Remove small speckles
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel_open, iterations=1)

# ---------------- FIND CONTOURS ----------------
contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_cracks = []
widths = []
lengths_px = []

# ---------------- FILTER: keep only true crack shapes ----------------
for cnt in contours:
    area = cv2.contourArea(cnt)

    # Minimum area — filters tiny noise
    if area < 300:
        continue

    # Use minAreaRect for accurate width/length regardless of orientation
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (rw, rh), angle = rect

    long_side  = max(rw, rh)
    short_side = min(rw, rh)

    # Skip tiny detections
    if long_side < 60:
        continue

    # Cracks are elongated — aspect ratio must be high
    aspect_ratio = long_side / (short_side + 1)
    if aspect_ratio < 3.0:
        continue

    # Skip contours hugging the image border (usually artifacts)
    x, y, bw, bh = cv2.boundingRect(cnt)
    margin = 8
    if x <= margin or y <= margin or x + bw >= w_img - margin or y + bh >= h_img - margin:
        continue

    box = cv2.boxPoints(rect)
    box = np.int32(box)
    valid_cracks.append(box)
    widths.append(short_side)
    lengths_px.append(long_side)

# ---------------- NO CRACK ----------------
if len(valid_cracks) == 0:
    st.error("No cracks detected")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Threshold")
        st.image(thresh, use_column_width=True)
    st.stop()

# ---------------- DRAW RESULTS ----------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)
with col2:
    annotated = img.copy()
    for box in valid_cracks:
        cv2.drawContours(annotated, [box], 0, (0, 255, 255), 2)
    st.subheader("Detected Crack")
    st.image(annotated, use_column_width=True)

# ---------------- MEASUREMENTS ----------------
PIXEL_TO_MM = 0.02

max_width_px  = float(max(widths))
max_length_px = float(max(lengths_px))

max_width_mm  = max_width_px  * PIXEL_TO_MM
max_length_mm = max_length_px * PIXEL_TO_MM

# ---------------- SEVERITY (IS 456:2000) ----------------
# Threshold: 0.3 mm hair crack | 1.0 mm moderate | >1.0 mm severe
if max_width_mm <= 0.3:
    severity = "🟢 Minor"
    actions = ["Monitor periodically", "Surface sealing if needed"]
elif max_width_mm <= 1.0:
    severity = "🟡 Moderate"
    actions = ["Crack filling", "Apply waterproof coating"]
else:
    severity = "🔴 Severe"
    actions = ["Structural inspection required", "Professional repair recommended"]

# ---------------- DISPLAY ----------------
st.markdown("---")
st.subheader("Crack Measurements")

c1, c2, c3 = st.columns(3)
c1.metric("Width (px)",  f"{int(max_width_px)} px")
c2.metric("Width (mm)",  f"{max_width_mm:.3f} mm")
c3.metric("Length (mm)", f"{max_length_mm:.1f} mm")

st.markdown("---")
st.subheader(f"Severity: {severity}")

st.subheader("Suggested Action")
for action in actions:
    st.write(f"• {action}")
