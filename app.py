import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("Crack Detection & Severity Analysis")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # your trained model

model = load_model()

# ---------------- UPLOAD IMAGE ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- SEVERITY LOGIC ----------------
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

# ---------------- MAIN LOGIC ----------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    results = model(img_np)[0]

    draw = ImageDraw.Draw(image)

    crack_count = 0
    crack_lengths = []

    # ---------- SAFE BOX HANDLING ----------
    if results.boxes is not None and len(results.boxes) > 0:
        for i in range(len(results.boxes)):
            box = results.boxes[i]

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # class 0 = crack (adjust if your dataset differs)
    if cls == 0 and conf > 0.2:
                crack_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                length_px = x2 - x1
                crack_lengths.append(length_px)

                # Draw clean yellow box
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="yellow",
                    width=3
                )

                # Number label only (professional)
                draw.text(
                    (x1, max(y1 - 15, 0)),
                    f"{crack_count} ({conf:.2f})",
                    fill="yellow"
                )

    # ---------------- DISPLAY OUTPUT IMAGE ----------------
    st.subheader("Detected Cracks")
    st.image(image, use_column_width=True)

    # ---------------- FEATURES ----------------
    st.subheader("📏 Extracted Crack Features")

    if crack_count == 0:
        st.info("No cracks detected.")
    else:
        for i, length in enumerate(crack_lengths):
            st.write(f"Crack {i+1} → Length: **{length} pixels**")

        total_length = sum(crack_lengths)

        severity, actions = calculate_severity(total_length)

        # ---------------- SEVERITY ----------------
        st.subheader(f"⚠️ Severity: **{severity}**")
        st.write(f"Total Crack Length: **{total_length} pixels**")

        # ---------------- SUGGESTIONS ----------------
        st.subheader("🛠 Suggested Action")
        for action in actions:
            st.write(f"• {action}")


