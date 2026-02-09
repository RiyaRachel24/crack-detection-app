import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- HELPER FUNCTIONS ----------------

def merge_boxes(boxes, distance_thresh=30):
    merged = []
    for (x, y, w, h) in boxes:
        found = False
        for i, (mx, my, mw, mh) in enumerate(merged):
            if abs(x - mx) < distance_thresh and abs(y - my) < distance_thresh:
                nx = min(x, mx)
                ny = min(y, my)
                nw = max(x + w, mx + mw) - nx
                nh = max(y + h, my + mh) - ny
                merged[i] = (nx, ny, nw, nh)
                found = True
                break
        if not found:
            merged.append((x, y, w, h))
    return merged


def detect_cracks(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    lengths = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 150:  # remove noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if max(w, h) < 40:  # ignore very small junk
            continue

        boxes.append((x, y, w, h))
        lengths.append(max(w, h))

    boxes = merge_boxes(boxes)
    return boxes, lengths


def severity_from_length(total_length):
    if total_length < 150:
        return "Low"
    elif total_length < 400:
        return "Moderate"
    else:
        return "High"


def suggested_action(severity):
    if severity == "Low":
        return [
            "Monitor periodically",
            "Seal minor surface cracks"
        ]
    elif severity == "Moderate":
        return [
            "Crack filling",
            "Prevent water ingress"
        ]
    else:
        return [
            "Structural inspection required",
            "Immediate repair recommended"
        ]


# ---------------- UI ----------------

uploaded_file = st.file_uploader(
    "Upload a crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    boxes, lengths = detect_cracks(gray)

    output = img_np.copy()
    total_length = sum(lengths)

    if boxes:
        for idx, (x, y, w, h) in enumerate(boxes, start=1):
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(
                output,
                f"{idx}",
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        st.image(output, caption="Detected Crack Regions", use_column_width=True)

        severity = severity_from_length(total_length)
        st.markdown(f"### 🚨 Severity: **{severity}**")
        st.markdown(f"**Total Crack Length (pixels):** `{total_length}`")

        st.markdown("### 🔧 Suggested Action")
        for act in suggested_action(severity):
            st.markdown(f"- {act}")

        st.markdown("### 📏 Extracted Crack Features")
        for i, l in enumerate(lengths, start=1):
            st.markdown(f"- Crack {i}: Length ≈ `{l}` pixels")

    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.warning("No cracks detected in the image.")
