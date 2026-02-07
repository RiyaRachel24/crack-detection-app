import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Crack Detection Web App",
    layout="centered"
)

st.title("🧱 Crack Detection and Feature Extraction")

st.write(
    "This app detects whether a crack is present using a deep learning model. "
    "If a crack is detected, OpenCV-based image processing is used to localize "
    "and extract crack features."
)

# --------------------------------------------------
# LOAD ML MODEL (CRACK / NO-CRACK CLASSIFIER)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained classifier

model = load_model()

# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# OPENCV + SKELETON CRACK EXTRACTION
# --------------------------------------------------
def extract_and_highlight_cracks(pil_image):
    img = np.array(pil_image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Adaptive thresholding (cracks are dark)
    binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        5
    )

    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Skeletonization → single-pixel crack line
    skeleton = skeletonize(cleaned // 255)

    # Separate individual cracks
    labeled = label(skeleton)
    regions = regionprops(labeled)

    overlay = img.copy()
    crack_features = []

    crack_id = 1
    for region in regions:
        # Remove tiny noise segments
        if region.area < 80:
            continue

        coords = region.coords
        length_px = len(coords)

        # Draw skeleton pixels (YELLOW)
        for y, x in coords:
            overlay[y, x] = [255, 255, 0]

        # Label crack with a clean number
        y0, x0 = coords[0]
        cv2.putText(
            overlay,
            str(crack_id),
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        crack_features.append({
            "crack_id": crack_id,
            "length_px": length_px
        })

        crack_id += 1

    return overlay, crack_features

# --------------------------------------------------
# MAIN APP LOGIC
# --------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_column_width=True)

    # -------- ML PREDICTION FIRST --------
    results = model(image)
    probs = results[0].probs
    pred_class = probs.top1
    class_name = model.names[pred_class]

    st.markdown(f"### Prediction: **{class_name}**")

    # -------- IMPORTANT FIX --------
    # Run OpenCV ONLY if model says CRACK
    if class_name.lower() == "crack":

        highlighted_img, cracks = extract_and_highlight_cracks(image)

        st.image(
            highlighted_img,
            caption="Detected Cracks (Skeleton-based)",
            use_column_width=True
        )

        st.subheader("📏 Extracted Crack Features")

        if len(cracks) == 0:
            st.warning(
                "Crack detected by model, but no clear crack segments "
                "were extracted after noise removal."
            )
        else:
            for c in cracks:
                st.write(
                    f"Crack {c['crack_id']} → Length: {c['length_px']} pixels"
                )

    else:
        # NO OpenCV processing here
        st.success("No crack detected in the image.")
