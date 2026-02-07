import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# -------------------- CONFIG --------------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("🛣️ Crack Detection & Severity Analysis")

MODEL_PATH = "best.pt"

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------- IMAGE UPLOAD --------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_np = np.array(image)

    # -------------------- YOLO INFERENCE --------------------
    results = model(img_np, conf=0.25)[0]

    draw = ImageDraw.Draw(image)
    crack_lengths = []

    crack_id = 1

    # -------------------- DRAW BOXES --------------------
    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Assuming class 0 = crack
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="yellow",
                    width=3
                )

                # Label crack number
                draw.text(
                    (x1, y1 - 15),
                    f"Crack {crack_id}",
                    fill="yellow"
                )

                # Approx crack length (box diagonal)
                length = int(((x2 - x1)**2 + (y2 - y1)**2) ** 0.5)
                crack_lengths.append((crack_id, length))

                crack_id += 1

        st.image(image, caption="Detected Cracks", use_column_width=True)

        # -------------------- FEATURES --------------------
        st.subheader("📏 Extracted Crack Features")

        total_length = 0
        for cid, length in crack_lengths:
            st.write(f"• Crack {cid} → Length: **{length} pixels**")
            total_length += length

        # -------------------- SEVERITY --------------------
        if total_length < 300:
            severity = "Low"
            action = ["Monitor periodically"]
        elif total_length < 800:
            severity = "Moderate"
            action = ["Crack filling", "Prevent water ingress"]
        else:
            severity = "High"
            action = ["Structural inspection", "Immediate repair"]

        st.markdown("---")
        st.subheader(f"🚦 Severity: **{severity}**")
        st.write(f"Total Crack Length: **{total_length} pixels**")

        st.subheader("🛠️ Suggested Action")
        for a in action:
            st.write(f"• {a}")

    else:
        st.warning("No cracks detected in the image.")
