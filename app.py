import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ----------------------------
# Load trained model
# ----------------------------
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

st.set_page_config(page_title="Crack Detection", layout="centered")

st.title("🧱 Crack Detection on Beams")
st.write("Upload a beam image to detect crack, extract features, and estimate severity.")

# ----------------------------
# Feature extraction
# ----------------------------
def extract_crack_features(img_gray):
    # Edge detection
    edges = cv2.Canny(img_gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0, 0, 0

    largest = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest)
    x, y, w, h = cv2.boundingRect(largest)

    length_ratio = max(w, h) / max(img_gray.shape)
    width_ratio = min(w, h) / max(img_gray.shape)

    return length_ratio, width_ratio, area


# ----------------------------
# Severity classification (FIXED)
# ----------------------------
def classify_severity(length_ratio, width_ratio, area, img_shape):
    # Tuned thresholds for civil cracks

    if length_ratio > 0.35 or width_ratio > 0.025:
        return "HIGH"

    elif length_ratio > 0.20 or width_ratio > 0.015:
        return "MEDIUM"

    else:
        return "LOW"

# ----------------------------
# Upload image
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload Beam Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    # Convert image
    img_np = np.array(pil_img)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # ----------------------------
    # Crack / No Crack Prediction
    # ----------------------------
    results = model(pil_img)
    probs = results[0].probs.data.cpu().numpy()
    class_names = results[0].names

    pred_idx = int(np.argmax(probs))
    confidence = probs[pred_idx] * 100
    prediction = class_names[pred_idx]

    st.subheader("🧠 Prediction Result")

    if prediction.lower() == "crack":
        st.success("CRACK detected")
    else:
        st.info("NO CRACK detected")

    st.write(f"**Confidence:** {confidence:.2f}%")

    # ----------------------------
    # Feature Extraction + Severity
    # ----------------------------
    if prediction.lower() == "crack":
        length_ratio, width_ratio, area = extract_crack_features(img_gray)
        severity = classify_severity(
            length_ratio, width_ratio, area, img_gray.shape
        )

        st.subheader("📊 Crack Feature Analysis")
        st.write(f"**Length Ratio:** {length_ratio:.3f}")
        st.write(f"**Width Ratio:** {width_ratio:.3f}")
        st.write(f"**Crack Area (px):** {int(area)}")

        st.subheader("🚦 Severity Level")

        if severity == "LOW":
            st.success("LOW severity crack")
        elif severity == "MEDIUM":
            st.warning("MEDIUM severity crack")
        else:
            st.error("HIGH severity crack")

