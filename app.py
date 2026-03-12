import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- FILE UPLOAD ----------------
file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if file is None:
    st.stop()

# ---------------- LOAD IMAGE ----------------
image = Image.open(file).convert("RGB")
img = np.array(image)

gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ---------------- CONTRAST ----------------
clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
gray = clahe.apply(gray)

# ---------------- SMOOTH ----------------
blur = cv2.GaussianBlur(gray,(5,5),0)

# ---------------- EXTRACT DARK CRACKS ----------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,25))
blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, kernel)

# ---------------- THRESHOLD ----------------
_,thresh = cv2.threshold(blackhat,20,255,cv2.THRESH_BINARY)

# ---------------- CONNECT CRACK ----------------
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
dilate = cv2.dilate(thresh,kernel,iterations=2)

# ---------------- REMOVE NOISE ----------------
clean = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

# ---------------- FIND CONTOURS ----------------
contours,_ = cv2.findContours(clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

largest=None
largest_area=0

for cnt in contours:

    area = cv2.contourArea(cnt)

    if area > largest_area:
        largest_area = area
        largest = cnt

# ---------------- NO CRACK ----------------
if largest is None:

    st.error("No cracks detected")

    col1,col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image,use_column_width=True)

    with col2:
        st.subheader("Processed")
        st.image(clean,use_column_width=True)

    st.stop()

# ---------------- BOUNDING BOX ----------------
rect = cv2.minAreaRect(largest)
box = cv2.boxPoints(rect)
box = np.int32(box)

annotated = img.copy()
cv2.drawContours(annotated,[box],0,(0,255,255),3)

# ---------------- DISPLAY ----------------
col1,col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    st.image(image,use_column_width=True)

with col2:
    st.subheader("Detected Crack")
    st.image(annotated,use_column_width=True)

# ---------------- WIDTH ----------------
width_pixels = int(min(rect[1]))

PIXEL_TO_MM = 0.02
width_mm = width_pixels * PIXEL_TO_MM

SI = width_mm / 0.30

if SI <= 0.33:
    severity = "🟢 Minor"
elif SI <= 1.0:
    severity = "🟡 Moderate"
else:
    severity = "🔴 Severe"

# ---------------- RESULTS ----------------
st.markdown("---")
st.subheader("Crack Measurements")

st.write(f"Crack width (pixels): {width_pixels}")
st.write(f"Estimated crack width (mm): {width_mm:.3f}")
st.write(f"Severity Index: {SI:.2f}")

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
