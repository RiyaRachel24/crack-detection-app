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

# ---------------- IMAGE ----------------
image = Image.open(file).convert("RGB")
img = np.array(image)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CONTRAST ENHANCEMENT ----------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)

# ---------------- BLUR ----------------
blur = cv2.GaussianBlur(gray,(5,5),0)

# ---------------- ADAPTIVE THRESHOLD ----------------
thresh = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11,
    2
)

# ---------------- MORPHOLOGY ----------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# connect crack fragments
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# remove noise
morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

# ---------------- CONTOURS ----------------
contours,_ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

valid_cracks=[]
widths=[]

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area < 50:
        continue

    x,y,w,h = cv2.boundingRect(cnt)

    length = max(w,h)
    width = min(w,h)

    if length < 20:
        continue

    aspect_ratio = length/(width+1)

    if aspect_ratio < 2:
        continue

    valid_cracks.append((x,y,w,h))
    widths.append(width)

# ---------------- NO CRACK ----------------
if len(valid_cracks)==0:

    st.error("No cracks detected")

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(image,use_column_width=True)

    with col2:
        st.subheader("Processed")
        st.image(morph,use_column_width=True)

    st.stop()

# ---------------- DRAW RESULTS ----------------

col1,col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image,use_column_width=True)

with col2:

    annotated = img.copy()

    for (x,y,w,h) in valid_cracks:
        cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,255),2)

    st.subheader("Detected Cracks")
    st.image(annotated,use_column_width=True)

# ---------------- WIDTH ----------------

max_width_pixels = int(max(widths))

PIXEL_TO_MM = 0.02
max_width_mm = max_width_pixels * PIXEL_TO_MM

# ---------------- SEVERITY ----------------

SI = max_width_mm / 0.30

if SI <= 0.33:
    severity = "🟢 Minor"
elif SI <= 1.0:
    severity = "🟡 Moderate"
else:
    severity = "🔴 Severe"

# ---------------- METRICS ----------------

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
