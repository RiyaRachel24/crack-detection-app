import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("🧱 Crack Detection & Severity Analysis")

# --------------------------------------------------
# LOAD YOLO CLASSIFICATION MODEL (NO CACHE DRAMA)
# --------------------------------------------------
model = YOLO("best.pt")  # classification model only

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a crack image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def detect_crack_with_yolo(image):
    """YOLO classification: crack / no crack"""
    results = model(image, verbose=False)[0]
    probs = results.probs
    cls = int(probs.top1)
    conf = float(probs.top1conf)
    return cls, conf

def extract_cracks_opencv(gray):
    """Extract cracks using OpenCV + skeletonization"""

    # Enhance contrast
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)

    # Morphological cleaning
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Binary
    _, binary = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY)

    # Skeletonization
    skeleton = skeletonize(binary // 255)

    return skeleton.astype(np.uint8)

def analyze_cracks(skeleton):
    """Label cracks and measure length"""
    labeled = label(skeleton)
    regions = regionprops(labeled)

    crack_data = []
    for i, region in enumerate(regions):
        length = region.area  # pixel length
        if length > 40:  # remove noise
            crack_data.append({
                "id": i + 1,
                "coords": region.coords,
                "length": length
            })
    return crack_data

def severity_from_length(max_length):
    """Rule-based severity"""
    if max_length < 150:
        return "Low", "Surface sealing"
    elif max_length < 350:
        return "Moderate", "Crack filling + water protection"
    else:
        return "High", "Structural inspection & repair"

# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    cls, conf = detect_crack_with_yolo(image)

    # Class 0 = crack, Class 1 = no crack (adjust if needed)
    if cls != 0:
        st.warning("No cracks detected in the image.")
        st.stop()

    st.success(f"Crack detected (confidence: {conf:.2f})")

    # Convert to grayscale
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    skeleton = extract_cracks_opencv(gray)
    crack_info = analyze_cracks(skeleton)

    if len(crack_info) == 0:
        st.warning("Crack detected, but unable to extract geometry clearly.")
        st.stop()

    # Draw cracks
    vis = img_np.copy()
    for crack in crack_info:
        for (r, c) in crack["coords"]:
            cv2.circle(vis, (c, r), 1, (255, 255, 0), -1)

        # Label
        y, x = crack["coords"][0]
        cv2.putText(
            vis,
            f"{crack['id']}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    st.image(vis, caption="Detected Cracks (Highlighted)", use_column_width=True)

    # Feature extraction
    st.subheader("📏 Extracted Crack Features")
    max_len = 0
    for crack in crack_info:
        st.write(f"Crack {crack['id']} → Length: {crack['length']} pixels")
        max_len = max(max_len, crack["length"])

    # Severity
    severity, action = severity_from_length(max_len)

    st.subheader("⚠️ Severity Assessment")
    st.markdown(f"**Severity:** `{severity}`")
    st.markdown(f"**Suggested Action:** {action}")
