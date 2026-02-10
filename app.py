import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection & Severity Analysis", layout="wide")
st.title("🧱 Crack Detection & Severity Analysis")

st.info("📱 Tip: Open this app on your phone to capture a live image using the camera.")

# ---------------- IMAGE INPUT (BROWSE + CAMERA) ----------------
uploaded_file = st.file_uploader(
    "Upload or capture crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.stop()

# ---------------- LOAD IMAGE ----------------
image = Image.open(uploaded_file).convert("RGB")
img = np.array(image)

# ---------------- PREPROCESS ----------------
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

edges = cv2.Canny(clean, 50, 150)

# ---------------- FIND CONTOURS ----------------
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

H, W = gray.shape
cracks = []

# Pixel → mm conversion (dataset-based approximation)
PIXEL_TO_MM = 0.05

# ---------------- FILTER & FEATURE EXTRACTION ----------------
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    length_px = max(w, h)
    width_px = min(w, h)

    # Noise removal
    if length_px < 80:
        continue
    if width_px < 2:
        continue
    if w > 0.9 * W:
        continue

    width_mm = width_px * PIXEL_TO_MM
    cracks.append((x, y, w, h, length_px, width_px, width_mm))

# ---------------- DISPLAY ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Detected Cracks")
    annotated = img.copy()

    for i, (x, y, w, h, _, _, _) in enumerate(cracks, start=1):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 255, 255),
            3
        )
        cv2.putText(
            annotated,
            f"{i}",
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    st.image(annotated, use_column_width=True)

# ---------------- NO CRACK CASE ----------------
if len(cracks) == 0:
    st.error("❌ No cracks detected.")
    st.stop()

# ---------------- FEATURES ----------------
st.markdown("---")
st.subheader("📏 Extracted Crack Features")

widths_mm = []

for i, (_, _, _, _, l_px, w_px, w_mm) in enumerate(cracks, start=1):
    widths_mm.append(w_mm)
    st.write(
        f"• Crack {i}: Length ≈ **{l_px} px**, Width ≈ **{round(w_mm, 3)} mm**"
    )

# ---------------- SEVERITY INDEX (YOUR FORMULA) ----------------
max_width_mm = max(widths_mm)

# Crack Severity Index
SI = max_width_mm / 0.30

# ---------------- CLASSIFICATION ----------------
if SI <= 0.33:
    severity = "🟢 Minor"
elif SI <= 1.00:
    severity = "🟡 Moderate"
else:
    severity = "🔴 Severe"

# ---------------- RESULT ----------------
st.markdown("---")
st.subheader("🚦 Crack Severity Result")

st.write(f"**Maximum Crack Width (mm):** {round(max_width_mm, 3)}")
st.write(f"**Severity Index (SI = w / 0.30):** {round(SI, 2)}")
st.subheader(f"**Severity Class:** {severity}")

# ---------------- ACTION ----------------
st.subheader("🛠 Suggested Action")

if "Minor" in severity:
    st.write("• Surface sealing\n• Periodic monitoring")
elif "Moderate" in severity:
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection\n• Professional repair required")
