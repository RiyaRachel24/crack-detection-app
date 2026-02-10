import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="wide"
)

st.title("Crack Detection & Severity Analysis")

st.markdown(
"""
📌 **Input Options**
- Upload an image **OR**
- Capture image directly using device camera
"""
)

# ---------------- INPUT METHOD ----------------
col_input1, col_input2 = st.columns(2)

image = None

with col_input1:
    uploaded_file = st.file_uploader(
        "📁 Upload concrete image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

with col_input2:
    camera_photo = st.camera_input("📷 Capture image")
    if camera_photo:
        image = Image.open(camera_photo).convert("RGB")

if image is None:
    st.stop()

# ---------------- IMAGE PREP ----------------
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CRACK DETECTION ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 60, 160)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(
    dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

H, W = gray.shape
cracks = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    length = max(w, h)
    width_px = min(w, h)

    # STRICT FILTERING
    if length < 80:
        continue
    if width_px > 0.25 * W:
        continue

    cracks.append((x, y, w, h, length, width_px))

# ---------------- NO CRACK CASE ----------------
if len(cracks) == 0:
    st.error("❌ No significant cracks detected.")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.stop()

# ---------------- DISPLAY ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Detected Cracks")
    annotated = img.copy()

    widths_mm = []
    lengths_px = []

    for i, (x, y, w, h, length, width_px) in enumerate(cracks, start=1):
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
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        pixel_to_mm = 0.02
        widths_mm.append(width_px * pixel_to_mm)
        lengths_px.append(length)

    st.image(annotated, use_column_width=True)

# ---------------- FEATURES ----------------
st.subheader("📏 Extracted Crack Features")

for i, (l, w) in enumerate(zip(lengths_px, widths_mm), start=1):
    st.write(
        f"• Crack {i}: Length ≈ **{int(l)} px**, Width ≈ **{w:.2f} mm**"
    )

# ---------------- SEVERITY ----------------
max_width = max(widths_mm)
SI = max_width / 0.30

if SI <= 0.33:
    severity = "🟢 Minor"
elif SI <= 1.00:
    severity = "🟡 Moderate"
else:
    severity = "🔴 Severe"

st.markdown("---")
st.subheader(f"🚦 Severity: **{severity}**")
st.write(f"Severity Index (SI) = **{SI:.2f}**")

# ---------------- ACTION ----------------
st.subheader("🛠 Suggested Action")

if severity.startswith("🟢"):
    st.write("• Surface sealing\n• Periodic monitoring")
elif severity.startswith("🟡"):
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection\n• Professional repair required")
