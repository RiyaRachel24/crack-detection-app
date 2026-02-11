import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection", layout="centered")
st.title("Crack Detection & Severity Analysis")
st.caption("Upload concrete surface images only")

# ---------------- UPLOAD ----------------
file = st.file_uploader(
    "Upload concrete beam image",
    type=["jpg", "jpeg", "png"]
)

if file is None:
    st.stop()

image = Image.open(file).convert("RGB")
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- SURFACE VALIDATION ----------------
def is_concrete_surface(gray_img):
    std_dev = np.std(gray_img)
    mean_val = np.mean(gray_img)

    # Concrete typically has moderate texture variance
    if std_dev < 18:
        return False
    if mean_val < 40 or mean_val > 220:
        return False

    return True

if not is_concrete_surface(gray):
    st.warning("⚠ Uploaded image does not appear to be a concrete surface.")
    st.stop()

# ---------------- CRACK DETECTION ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

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
    width_pixels = min(w, h)

    aspect_ratio = length / (width_pixels + 1e-5)

    # STRICT FILTERS
    if length < 80:
        continue

    if aspect_ratio < 3:   # must be long & thin
        continue

    if w > 0.8 * W:   # remove large texture blobs
        continue

    cracks.append((x, y, w, h, length, width_pixels))

# ---------------- NO CRACK ----------------
if len(cracks) == 0:
    st.error("❌ No significant cracks detected.")
    st.stop()

# ---------------- DRAW LAYOUT ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Detected Cracks")
    annotated = img.copy()

    for i, (x, y, w, h, length, width_pixels) in enumerate(cracks, start=1):
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

# ---------------- FEATURE EXTRACTION ----------------
st.subheader("Extracted Crack Features")

width_mm_list = []

for i, (x, y, w, h, length, width_pixels) in enumerate(cracks, start=1):

    # Pixel-to-mm assumption (example scaling)
    pixel_to_mm = 0.02
    width_mm = width_pixels * pixel_to_mm
    width_mm_list.append(width_mm)

    st.write(
        f"• Crack {i}: Length ≈ {int(length)} px | "
        f"Width ≈ {width_mm:.3f} mm"
    )

# ---------------- SEVERITY CALCULATION ----------------
max_width = max(width_mm_list)

# Severity Index
SI = max_width / 0.30

# Classification
if SI <= 0.33:
    severity = "Minor"
elif SI <= 1.00:
    severity = "Moderate"
else:
    severity = "Severe"

st.markdown("---")
st.subheader(f"Severity: {severity}")
st.write(f"Severity Index (SI) = {SI:.2f}")

# ---------------- SUGGESTED ACTION ----------------
st.subheader("Suggested Action")

if severity == "Minor":
    st.write("• Surface sealing")
    st.write("• Routine monitoring")

elif severity == "Moderate":
    st.write("• Crack filling")
    st.write("• Waterproof coating")

else:
    st.write("• Structural inspection required")
    st.write("• Professional repair recommended")
