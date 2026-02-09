import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# -------------------- PAGE SETUP --------------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("Crack Detection & Severity Analysis")

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your trained model

model = load_model()

# -------------------- SEVERITY LOGIC --------------------
def get_severity(total_length):
    if total_length < 150:
        return "Low", ["Monitor periodically"]
    elif total_length < 400:
        return "Moderate", ["Crack filling", "Prevent water ingress"]
    else:
        return "High", ["Structural inspection", "Immediate repair recommended"]

# -------------------- CRACK DETECTION (BOX-BASED) --------------------
def detect_crack_boxes(gray):
    """
    Uses OpenCV to extract crack-like regions and returns bounding boxes
    based on LENGTH (not area) to avoid missing thin cracks.
    """

    # Enhance contrast
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Strengthen cracks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    lengths = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        crack_length = max(w, h)

        # 🔑 KEY FIX: LENGTH-BASED FILTERING
        if crack_length > 80:   # tuned for visible cracks
            boxes.append((x, y, w, h))
            lengths.append(crack_length)

    return boxes, lengths

# -------------------- FILE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload a crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---------------- YOLO PREDICTION ----------------
    results = model(image)

    pred_label = "No crack"
    if results[0].probs is not None:
        cls_id = int(results[0].probs.top1)
        conf = float(results[0].probs.top1conf)

        if cls_id == 0 and conf > 0.4:
            pred_label = "Crack"

    # ---------------- CRACK PRESENT ----------------
    if pred_label == "Crack":
        boxes, lengths = detect_crack_boxes(gray)
        annotated = img_np.copy()

        if boxes:
            total_length = int(sum(lengths))

            # Draw boxes
            for i, (x, y, w, h) in enumerate(boxes):
                cv2.rectangle(
                    annotated,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    2
                )
                cv2.putText(
                    annotated,
                    str(i + 1),
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2
                )

            st.image(
                annotated,
                caption="Detected Crack Regions (Box-based)",
                use_column_width=True
            )

            # ---------------- FEATURES ----------------
            st.subheader("📏 Extracted Crack Features")
            for i, l in enumerate(lengths):
                st.write(f"Crack {i+1}: Length = {int(l)} pixels")

            severity, actions = get_severity(total_length)

            # ---------------- SEVERITY ----------------
            st.subheader(f"🚦 Severity: {severity}")
            st.write(f"Total crack length: **{total_length} pixels**")

            # ---------------- ACTIONS ----------------
            st.subheader("🛠️ Suggested Action")
            for a in actions:
                st.write(f"- {a}")

        else:
            st.warning(
                "Crack detected by model, but regions are too thin or fragmented to extract reliably."
            )

    # ---------------- NO CRACK ----------------
    else:
        st.success("No cracks detected in the image.")
