import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("Crack Detection & Severity Analysis")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained crack / non-crack model

model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ---- Read image ----
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- Model prediction ----
    results = model(image)
    probs = results[0].probs

    pred_class = probs.top1
    class_name = model.names[pred_class]

    st.markdown(f"### Prediction: **{class_name}**")

    # ---------------- ONLY IF CRACK ----------------
    if class_name.lower() == "crack":

        # ---- GRAYSCALE ----
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # ---- THRESHOLDING ----
        thresh_val = threshold_otsu(gray)
        binary = gray < thresh_val     # cracks are darker

        # ---- SKELETONIZATION (THIS IS THE KEY STEP) ----
        skeleton = skeletonize(binary)

        # ---- CRACK LENGTH ----
        crack_length_pixels = np.sum(skeleton)

        # ---- SEVERITY LOGIC (FIXED) ----
        if crack_length_pixels < 500:
            severity = "Low"
        elif crack_length_pixels < 1500:
            severity = "Moderate"
        else:
            severity = "High"

        # ---- OVERLAY SKELETON ON IMAGE ----
        overlay = img_np.copy()
        overlay[skeleton] = [255, 255, 0]   # YELLOW SINGLE LINE

        st.image(
            overlay,
            caption="Detected Crack (Skeletonized)",
            use_column_width=True
        )

        st.markdown(f"### Severity: **{severity}**")
        st.markdown(f"Crack Length (pixels): `{crack_length_pixels}`")

        # ---- MANUAL SUGGESTIONS ----
        st.subheader("Suggested Action")
        if severity == "Low":
            st.write("- Surface sealing\n- Monitor periodically")
        elif severity == "Moderate":
            st.write("- Crack filling\n- Prevent water ingress")
        else:
            st.write("- Structural inspection required\n- Immediate repair recommended")

    else:
        st.success("No crack detected in the image.")
