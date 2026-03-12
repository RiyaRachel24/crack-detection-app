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

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CONTRAST ENHANCEMENT ----------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# ---------------- BLUR ----------------
blur = cv2.GaussianBlur(gray,(5,5),0)

# ---------------- EDGE DETECTION ----------------
edges = cv2.Canny(blur,30,100)

# ---------------- MORPHOLOGY ----------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
connected = cv2.morphologyEx(connected, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

# ---------------- FIND CONTOURS ----------------
contours,_ = cv2.findContours(connected,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

valid_cracks=[]
widths=[]

h_img, w_img = connected.shape

# ---------------- FILTER CONTOURS ----------------
for cnt in contours:

    x,y,w,h = cv2.boundingRect(cnt)

    # Ignore contours touching the border
    if x <= 5 or y <= 5 or x+w >= w_img-5 or y+h >= h_img-5:
        continue

    area = cv2.contourArea(cnt)

    if area < 40:
        continue

    rect = cv2.minAreaRect(cnt)
    (cx,cy),(rw,rh),angle = rect

    length = max(rw,rh)
    width = min(rw,rh)

    if length < 30:
        continue

    aspect_ratio = length/(width+1)

    if aspect_ratio < 3:
        continue

    box = cv2.boxPoints(rect)
    box = np.int32(box)

    valid_cracks.append(box)
    widths.append(width)

# ---------------- NO CRACK ----------------
if len(valid_cracks)==0:

    st.error("No cracks detected")

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image,use_column_width=True)

    with col2:
        st.subheader("Edge Detection")
        st.image(edges,use_column_width=True)

    st.stop()

# ---------------- DRAW RESULTS ----------------

col1,col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image,use_column_width=True)

with col2:
    annotated = img.copy()

    for box in valid_cracks:
        cv2.drawContours(annotated,[box],0,(0,255,255),3)

    st.subheader("Detected Cracks")
    st.image(annotated,use_column_width=True)

# ---------------- WIDTH CALCULATION ----------------

max_width_pixels = int(max(widths))

# Approximate pixel to mm conversion
PIXEL_TO_MM = 0.02
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

st.write(f"Max crack width (pixels): {max_width_pixels}")
st.write(f"Estimated crack width (mm): {max_width_mm:.3f}")
st.write(f"Severity Index (SI): {SI:.2f}")

st.markdown("---")
st.subheader(f"Severity: {severity}")

# ---------------- ACTION ----------------

st.subheader("Suggested Action")

if SI <= 0.33:
    st.write("• Monitor periodically")
    st.write("• Surface sealing")

elif SI <= 1.0:
    st.write("• Crack filling")
    st.write("• Waterproof coating")

else:
    st.write("• Structural inspection required")
    st.write("• Professional repair recommended")
