import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- UPLOAD ----------------
file = st.file_uploader("Upload concrete surface image", type=["jpg", "jpeg", "png"])
if file is None:
    st.stop()

image = Image.open(file).convert("RGB")
img = np.array(image)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

st.image(image, caption="Uploaded Image", use_column_width=True)

# ---------------- CRACK DETECTION (OPENCV) ----------------
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

    # STRICT FILTERS (THIS IS KEY)
    if length < 80:
        continue
    if w > 0.7 * W:
        continue

    cracks.append((x, y, w, h, length))

# ---------------- NO CRACK CASE ----------------
if len(cracks) == 0:
    st.error("❌ No significant cracks detected.")
    st.stop()

# ---------------- DRAW BOXES ----------------
col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)

    with col2:
        st.subheader("Detected Cracks")
        annotated = img_np.copy()

        for i, (x1, y1, x2, y2, length) in enumerate(valid_boxes, start=1):
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )
            cv2.putText(
                annotated,
                f"{i}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        st.image(annotated, use_column_width=True)

# ---------------- FEATURES ----------------
st.subheader("📏 Extracted Crack Features")
for i, l in enumerate(lengths, start=1):
    st.write(f"• Crack {i}: Length ≈ **{int(l)} pixels**")

max_len = max(lengths)
count = len(lengths)

# ---------------- SEVERITY (FIXED & LOGICAL) ----------------
if max_len < 150 and count == 1:
    severity = "Low"
elif max_len < 350 and count <= 2:
    severity = "Moderate"
else:
    severity = "Severe"

st.markdown("---")
st.subheader(f"🚦 Severity: **{severity}**")

# ---------------- ACTION ----------------
st.subheader("🛠 Suggested Action")
if severity == "Low":
    st.write("• Surface sealing\n• Monitoring")
elif severity == "Moderate":
    st.write("• Crack filling\n• Waterproof coating")
else:
    st.write("• Structural inspection\n• Professional repair required")
