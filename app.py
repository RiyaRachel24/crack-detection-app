import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🧱 Crack Detection & Severity Analysis")

# Load classification model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # YOLOv8 classification model

model = load_model()

# ---------------- HELPERS ----------------
def classify_crack(img):
    results = model(img)
    probs = results[0].probs.data.cpu().numpy()
    cls = np.argmax(probs)
    conf = probs[cls]
    return cls, conf  # 0 = crack, 1 = no crack (depends on training)

def detect_cracks_opencv(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crack_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 300:  # remove noise
            continue

        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = max(w,h) / (min(w,h) + 1e-5)

        if aspect_ratio > 3:  # crack-like shape
            crack_boxes.append((x,y,w,h))

    return crack_boxes

def merge_boxes(boxes, threshold=20):
    merged = []
    for box in boxes:
        x,y,w,h = box
        merged_flag = False
        for i,(mx,my,mw,mh) in enumerate(merged):
            if abs(x-mx) < threshold and abs(y-my) < threshold:
                nx = min(x,mx)
                ny = min(y,my)
                nw = max(x+w, mx+mw) - nx
                nh = max(y+h, my+mh) - ny
                merged[i] = (nx,ny,nw,nh)
                merged_flag = True
                break
        if not merged_flag:
            merged.append(box)
    return merged

def severity_from_length(length):
    if length < 150:
        return "Low", ["Monitor periodically"]
    elif length < 400:
        return "Moderate", ["Crack filling", "Prevent water ingress"]
    else:
        return "High", ["Structural inspection", "Immediate repair"]

# ---------------- APP ----------------
uploaded = st.file_uploader("Upload crack image", type=["jpg","png","jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    cls, conf = classify_crack(img_np)

    if conf < 0.6:
        st.warning("Low confidence prediction. Try a clearer image.")
        st.stop()

    if cls != 0:
        st.success("✅ No crack detected in the image.")
        st.stop()

    # OpenCV crack detection
    boxes = detect_cracks_opencv(gray)
    boxes = merge_boxes(boxes)

    if len(boxes) == 0:
        st.warning("Crack present, but no significant crack regions extracted.")
        st.stop()

    annotated = img_np.copy()
    lengths = []

    for i,(x,y,w,h) in enumerate(boxes,1):
        cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,255,0),2)
        length = max(w,h)
        lengths.append(length)
        cv2.putText(
            annotated,
            f"{i}",
            (x, y-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,0),
            2
        )

    st.image(annotated, caption="Detected Crack Regions", use_column_width=True)

    st.subheader("📏 Extracted Crack Features")
    for i,l in enumerate(lengths,1):
        st.write(f"Crack {i}: Length ≈ {l} pixels")

    max_len = max(lengths)
    severity, actions = severity_from_length(max_len)

    st.subheader(f"🚦 Severity: {severity}")
    st.subheader("🛠 Suggested Action")
    for act in actions:
        st.write(f"- {act}")
