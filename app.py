import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- INPUT MODE ----------------
mode = st.radio(
    "Select image input method:",
    ("Upload Image", "Use Camera")
)

image = None

if mode == "Upload Image":
    file = st.file_uploader(
        "Upload concrete beam image",
        type=["jpg", "jpeg", "png"]
    )
    if file:
        image = Image.open(file).convert("RGB")

else:
    camera_image = st.camera_input("Take a photo of the crack")
    if camera_image:
        image = Image.open(camera_image).convert("RGB")

if image is None:
    st.stop()

# ---------------- IMAGE PREP ----------------
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    st.image(image, use_column_width=True)

# ---------------- CRACK DETECTION ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(
    dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

H, W = gray.shape
valid_cracks = []
widths_px = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    length = max(w, h)
    width = min(w, h)

    # Strict filters (prevents false cracks)
    if length < 80:
        continue
    if width < 2:
        continue
    if w > 0.7 * W:
        continue

    valid_cracks.append((x, y, x + w, y + h))
    widths_px.append(width)

if len(valid_cracks) == 0:
    st.error("❌ No significant cracks detected.")
    st.stop()

# ---------------- DRAW BOXES ----------------
annotated = img.copy()

for i, (x1, y1, x2, y2) in enumerate(valid_cracks, start=1):
    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        3
    )
    cv2.putText(
        annotated,
        str(i),
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

with col2:
    st.subheader("Detected Cracks")
    st.image(annotated, use_column_width=True)

# ---------------- WIDTH → MM CONVERSION ----------------
# Example calibration (mention this in viva)
PIXEL_TO_MM = 0.05  # assumed scale
widths_mm = [w * PIXEL_TO_MM for w in widths_px]
max_width = max(widths_mm)

# ---------------- SEVERITY FORMULA ----------------
SI = max_width / 0.30  # Eurocode / IS limit

if SI <= 0.33:
    severity = "🟢 Minor"
elif SI <= 1.0:
    severity = "🟡 Moderate"
else:
    severity = "🔴 Severe"

# ---------------- RESULTS ----------------
st.markdown("---")
st.subheader("📏 Extracted Crack Widths")

for i, w in enumerate(widths_mm, start=1):
    st.write(f"• Crack {i}: Width ≈ **{w:.2f} mm**")

st.markdown(f"### 🚦 Severity: **{severity}**")
st.write(f"**Severity Index (SI):** `{SI:.2f}`")

# ---------------- ACTION ----------------
st.subheader("🛠 Suggested Action")

if severity == "🟢 Minor":
    st.write("• Surface sealing\n• Periodic monitoring")
elif severity == "🟡 Moderate":
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection\n• Professional repair required")
