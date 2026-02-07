import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crack Detection App",
    layout="centered"
)

st.title("🛠 Crack Detection & Severity Analysis")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # make sure best.pt is in same folder

model = load_model()

# ---------------- IMAGE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)

    # ---------------- YOLO INFERENCE ----------------
    results = model(img_np)[0]

    draw = ImageDraw.Draw(image)
    crack_lengths = []
    crack_id = 0

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Class 0 = crack (as trained)
            if cls == 0 and conf > 0.25:
                crack_id += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                length = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                crack_lengths.append(length)

                # Draw bounding box
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline="yellow",
                    width=3
                )

                # Label
                draw.text(
                    (x1, max(0, y1 - 15)),
                    f"{crack_id}",
                    fill="yellow"
                )

        st.image(image, caption="Detected Cracks", use_column_width=True)

        # ---------------- FEATURES ----------------
        st.subheader("📏 Extracted Crack Features")
        for i, length in enumerate(crack_lengths, start=1):
            st.write(f"Crack {i} → Length: **{length} pixels**")

        # ---------------- SEVERITY ----------------
        max_length = max(crack_lengths)

        if max_length < 150:
            severity = "Low"
            action = [
                "Monitor periodically",
                "No immediate repair required"
            ]
        elif max_length < 400:
            severity = "Moderate"
            action = [
                "Crack filling",
                "Prevent water ingress"
            ]
        else:
            severity = "High"
            action = [
                "Structural inspection required",
                "Immediate repair recommended"
            ]

        st.subheader("⚠ Severity Assessment")
        st.write(f"**Severity:** {severity}")
        st.write(f"**Maximum Crack Length:** {max_length} pixels")

        st.subheader("🧰 Suggested Action")
        for a in action:
            st.write(f"• {a}")

    else:
        st.warning("No cracks detected in the image.")
