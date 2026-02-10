import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection", layout="wide")
st.title("Crack Detection & Severity Analysis")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload concrete surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.stop()

# ---------------- LOAD IMAGE ----------------
image = Image.open(uploaded_file).convert("RGB")
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- PREPROCESS ----------------
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 60, 160)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
dilated = cv2.dilate(edges, kernel, iterations=1)

contours, _ = cv2.findContours(
    dilated,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

H, W = gray.shape
cracks = []
lengths = []

# ---------------- FILTER CONTOURS ----------------
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    length = max(w, h)
    area = cv2.contourArea(cnt)

    # IMPORTANT FILTERS (review-safe)
    if length < 80:        # remove tiny noise
        continue
    if area < 100:         # remove texture
        continue
    if w > 0.8 * W:        # remove full-width edges
        continue

    cracks.append((x, y, w, h))
    lengths.append(length)

# ---------------- DISPLAY ----------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image, use_column_width=True)

with col2:
    st.subheader("Detected Cracks")
    annotated = img.copy()

    for i, (x, y, w, h) in enumerate(cracks, start=1):
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
st.markdown("---")

if len(cracks) == 0:
    st.error("❌ No cracks detected in the image.")
    st.stop()

# ---------------- FEATURES ----------------
st.subheader("📏 Extracted Crack Features")
for i, l in enumerate(lengths, start=1):
    st.write(f"• Crack {i}: Length ≈ **{int(l)} pixels**")

max_len = max(lengths)
count = len(lengths)

# ---------------- SEVERITY LOGIC ----------------
if max_len < 150 and count == 1:
    severity = "Low"
elif max_len < 350 and count <= 2:
    severity = "Moderate"
else:
    severity = "Severe"

# ---------------- RESULT ----------------
st.markdown("---")
st.subheader(f"🚦 Severity: **{severity}**")

# ---------------- ACTION ----------------
st.subheader("🛠 Suggested Action")

if severity == "Low":
    st.write("• Surface sealing\n• Periodic monitoring")
elif severity == "Moderate":
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection\n• Professional repair required")

