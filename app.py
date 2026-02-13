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
cracks = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    length = max(w, h)
    width = min(w, h)

    # basic filtering
    if length < 80:
        continue

    cracks.append((x, y, w, h, length, width))

# ---------------- NO CRACK ----------------
if len(cracks) == 0:
    st.error("No cracks detected.")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.stop()

# ---------------- DISPLAY ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Uploaded Image")
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Detected Cracks")
    annotated = img.copy()

    lengths = []
    widths = []

    for i, (x, y, w, h, length, width) in enumerate(cracks, start=1):
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0,255,255), 3)
        cv2.putText(annotated, str(i), (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        lengths.append(length)
        widths.append(width)

    st.image(annotated, use_column_width=True)

# ---------------- FEATURES ----------------
st.subheader("Extracted Crack Features")

for i, (l, w) in enumerate(zip(lengths, widths), start=1):
    st.write(f"Crack {i}: Length ≈ {int(l)} px | Width ≈ {int(w)} px")

# ---------------- SEVERITY USING WIDTH ----------------
max_width_px = max(widths)

# pixel to mm conversion (assumed calibration)
pixel_to_mm = 0.05
max_width_mm = max_width_px * pixel_to_mm

# Severity Index formula (Eurocode limit 0.30 mm)
SI = max_width_mm / 0.30

if SI <= 0.33:
    severity = "Minor"
elif SI <= 1.0:
    severity = "Moderate"
else:
    severity = "Severe"

st.markdown("---")
st.subheader(f"Severity: {severity}")
st.write(f"Max width ≈ {max_width_mm:.3f} mm")
st.write(f"Severity Index = {SI:.2f}")

# ---------------- ACTION ----------------
st.subheader("Suggested Action")

if severity == "Minor":
    st.write("• Surface sealing\n• Monitor crack")
elif severity == "Moderate":
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection required\n• Immediate repair")
