import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.4        # confidence threshold
MIN_BOX_AREA = 800          # ignore tiny false cracks
IOU_MERGE_THRESH = 0.3      # merge overlapping boxes

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🧱 Crack Detection & Severity Analysis")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ---------------- HELPERS ----------------
def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter

    return inter / union if union > 0 else 0

def merge_boxes(boxes):
    merged = []
    for b in boxes:
        added = False
        for i in range(len(merged)):
            if iou(b, merged[i]) > IOU_MERGE_THRESH:
                merged[i] = [
                    min(b[0], merged[i][0]),
                    min(b[1], merged[i][1]),
                    max(b[2], merged[i][2]),
                    max(b[3], merged[i][3]),
                ]
                added = True
                break
        if not added:
            merged.append(b)
    return merged

def severity_from_length(length):
    if length < 150:
        return "Low", ["Monitor periodically"]
    elif length < 350:
        return "Moderate", ["Crack filling", "Prevent water ingress"]
    else:
        return "High", ["Structural inspection", "Immediate repair"]

# ---------------- UI ----------------
uploaded = st.file_uploader("Upload crack image", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    # ---------------- YOLO INFERENCE ----------------
    results = model(img_np)[0]

    raw_boxes = []
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if area > MIN_BOX_AREA:
            raw_boxes.append([x1, y1, x2, y2])

    if not raw_boxes:
        st.warning("No cracks detected.")
        st.stop()

    # ---------------- MERGE BOXES ----------------
    boxes = merge_boxes(raw_boxes)

    # ---------------- DRAW ----------------
    drawn = img_np.copy()
    crack_lengths = []

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(drawn, (x1, y1), (x2, y2), (255, 255, 0), 2)

        length = max(x2 - x1, y2 - y1)
        crack_lengths.append(length)

        cv2.putText(
            drawn,
            f"Crack {i+1}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    st.image(drawn, caption="Detected Cracks (Box-based)", use_column_width=True)

    # ---------------- FEATURES ----------------
    st.subheader("📏 Extracted Crack Features")
    for i, l in enumerate(crack_lengths):
        st.write(f"Crack {i+1}: Length ≈ **{int(l)} pixels**")

    # ---------------- SEVERITY ----------------
    max_len = max(crack_lengths)
    sev, actions = severity_from_length(max_len)

    st.subheader(f"🚦 Severity: **{sev}**")
    st.write(f"Maximum crack length used: **{int(max_len)} pixels**")

    st.subheader("🛠 Suggested Actions")
    for a in actions:
        st.write(f"- {a}")
