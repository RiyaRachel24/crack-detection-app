import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🧱 Crack Detection & Severity Analysis")

# --------------------------------------------------
# LOAD YOLO CLASSIFICATION MODEL
# --------------------------------------------------
model = YOLO("best.pt")  # crack / no-crack classifier

# --------------------------------------------------
# FUNCTIONS
# --------------------------------------------------
def yolo_classify(image):
    res = model(image, verbose=False)[0]
    cls = int(res.probs.top1)
    conf = float(res.probs.top1conf)
    return cls, conf

def detect_crack_boxes(gray):
    """
    Returns merged bounding boxes for cracks
    """
    # Strong thresholding
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove noise aggressively
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > 800:  # 🔑 filter small junk
            boxes.append([x, y, x + w, y + h])

    return boxes

def merge_boxes(boxes, gap=25):
    """
    Merge close boxes into ONE big box
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda b: b[0])
    merged = [boxes[0]]

    for box in boxes[1:]:
        last = merged[-1]

        if box[0] - last[2] < gap:
            last[2] = max(last[2], box[2])
            last[3] = max(last[3], box[3])
        else:
            merged.append(box)

    return merged

def severity_from_length(length):
    if length < 150:
        return "Low", ["Monitor periodically"]
    elif length < 350:
        return "Moderate", ["Crack filling", "Prevent water ingress"]
    else:
        return "High", ["Structural inspection", "Immediate repair"]

# --------------------------------------------------
# UI
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- YOLO CLASSIFICATION ----
    cls, conf = yolo_classify(image)

    # adjust if your labels differ
    if cls != 0:
        st.success("No crack detected.")
        st.stop()

    st.warning(f"Crack detected (confidence {conf:.2f})")

    # ---- OPENCV BOX DETECTION ----
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    raw_boxes = detect_crack_boxes(gray)
    boxes = merge_boxes(raw_boxes)

    if not boxes:
        st.warning("Crack present, but geometry could not be extracted clearly.")
        st.stop()

    # ---- DRAW BOXES ----
    vis = img_np.copy()
    lengths = []

    for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 3)
        cv2.putText(
            vis, f"Crack {i}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 0), 2
        )

        length = max(x2 - x1, y2 - y1)
        lengths.append(length)

    st.image(vis, caption="Detected Crack Regions", use_column_width=True)

    # ---- FEATURES ----
    st.subheader("📏 Extracted Crack Features")
    max_len = 0
    for i, l in enumerate(lengths, start=1):
        st.write(f"Crack {i} → Length: {l} pixels")
        max_len = max(max_len, l)

    # ---- SEVERITY ----
    severity, actions = severity_from_length(max_len)
    st.subheader(f"🚦 Severity: {severity}")

    st.subheader("🛠 Suggested Action")
    for a in actions:
        st.write(f"• {a}")
