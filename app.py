import streamlit as st
import cv2
import numpy as np
from PIL import Image

# -------------------------------
# CONFIG
# -------------------------------
MIN_WIDTH = 12        # filter tiny texture
MIN_HEIGHT = 30
MERGE_DISTANCE = 40   # pixels
LOW_SEVERITY_LEN = 150
MOD_SEVERITY_LEN = 350
SEVERE_WIDTH = 20

# -------------------------------
# HELPERS
# -------------------------------
def preprocess(img_gray):
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    return closed


def merge_boxes(boxes):
    merged = []

    for box in boxes:
        x, y, w, h = box
        merged_flag = False

        for i, (mx, my, mw, mh) in enumerate(merged):
            if abs(x - mx) < MERGE_DISTANCE and abs(y - my) < MERGE_DISTANCE:
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged[i] = (nx, ny, nw, nh)
                merged_flag = True
                break

        if not merged_flag:
            merged.append(box)

    return merged


def calculate_severity(boxes):
    if not boxes:
        return "No Crack", "No action required"

    total_length = sum(max(w, h) for (_, _, w, h) in boxes)
    max_width = max(min(w, h) for (_, _, w, h) in boxes)

    if total_length > MOD_SEVERITY_LEN or max_width > SEVERE_WIDTH:
        return "Severe", "Structural repair required"
    elif total_length > LOW_SEVERITY_LEN:
        return "Moderate", "Crack filling and sealing recommended"
    else:
        return "Low", "Surface monitoring suggested"


# -------------------------------
# STREAMLIT UI
# -------------------------------
st.set_page_config(page_title="Crack Detection App", layout="wide")
st.title("🧱 Crack Detection & Severity Analysis")

uploaded_file = st.file_uploader("Upload crack image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img = np.array(image)
    output = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    processed = preprocess(img)

    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # HARD FILTER → removes fake cracks
        if w < MIN_WIDTH and h < MIN_HEIGHT:
            continue

        boxes.append((x, y, w, h))

    # MERGE fragmented boxes
    boxes = merge_boxes(boxes)

    # DRAW
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(output, f"Crack {i+1}", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    severity, action = calculate_severity(boxes)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.image(output, caption="Detected Cracks", use_column_width=True)

    st.markdown("### 📏 Extracted Crack Features")
    if boxes:
        for i, (_, _, w, h) in enumerate(boxes):
            st.write(f"Crack {i+1} → Length ≈ {max(w, h)} px")
    else:
        st.info("No cracks detected.")

    st.markdown(f"## 🔥 Severity: **{severity}**")
    st.markdown(f"### 🛠 Suggested Action: {action}")
