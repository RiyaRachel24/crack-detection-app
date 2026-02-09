import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("Crack Detection & Severity Analysis")

# ------------------ IMAGE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"]
)

# ------------------ FUNCTIONS ------------------

def preprocess(gray):
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return edges


def detect_cracks(gray, original):
    edges = preprocess(gray)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crack_boxes = []
    crack_lengths = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        length = max(w, h)

        # --------- IMPORTANT FILTERS ----------
        if area < 40:
            continue

        if length < 60:
            continue

        aspect_ratio = length / (min(w, h) + 1)
        if aspect_ratio < 2.0:
            continue

        crack_boxes.append((x, y, w, h))
        crack_lengths.append(length)

    output = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)

    for i, (x, y, w, h) in enumerate(crack_boxes):
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            output,
            f"Crack {i+1}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    return output, crack_boxes, crack_lengths


def calculate_severity(total_length):
    if total_length < 150:
        return "Low", ["Monitor periodically"]
    elif total_length < 400:
        return "Moderate", [
            "Crack filling",
            "Prevent water ingress"
        ]
    else:
        return "High", [
            "Structural inspection required",
            "Immediate repair recommended"
        ]


# ------------------ MAIN LOGIC ------------------

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    annotated, boxes, lengths = detect_cracks(img_np, img_np)

    if len(boxes) == 0:
        st.warning("No structural cracks detected.")
    else:
        st.image(
            annotated,
            caption="Detected Crack Regions (Bounding Boxes)",
            use_column_width=True
        )

        st.subheader("📏 Extracted Crack Features")

        total_length = 0
        for i, length in enumerate(lengths):
            st.write(f"• Crack {i+1}: Length ≈ {length} pixels")
            total_length += length

        severity, actions = calculate_severity(total_length)

        st.subheader(f"🚦 Severity: {severity}")
        st.write(f"**Total Crack Length:** {total_length} pixels")

        st.subheader("🛠️ Suggested Actions")
        for act in actions:
            st.write(f"- {act}")
