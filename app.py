import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crack Detection App", layout="centered")
st.title("Crack Detection & Severity Analysis")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")   # your trained YOLO crack model

model = load_model()

# ---------------- UPLOAD IMAGE ----------------
uploaded_file = st.file_uploader(
    "Upload a concrete surface image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.stop()

# ---------------- IMAGE PREP ----------------
image = Image.open(uploaded_file).convert("RGB")
img_np = np.array(image)
st.image(image, caption="Uploaded Image", use_column_width=True)

# ---------------- YOLO INFERENCE ----------------
results = model(img_np, conf=0.35, iou=0.45)
res = results[0]

# SAFETY CHECK (THIS IS THE BUG FIX)
if res.boxes is None or res.boxes.xyxy is None or len(res.boxes.xyxy) == 0:
    st.warning("❌ No cracks detected in the image.")
    st.stop()

boxes = res.boxes.xyxy.cpu().numpy()

# ---------------- FILTER BOXES (REMOVE FALSE POSITIVES) ----------------
filtered_boxes = []
H, W = img_np.shape[:2]

for box in boxes:
    x1, y1, x2, y2 = map(int, box)
    w = x2 - x1
    h = y2 - y1

    # HARD FILTERS (VERY IMPORTANT)
    if w < 15 or h < 40:        # remove texture / noise
        continue
    if w > 0.6 * W:             # remove full-width shadows
        continue

    filtered_boxes.append((x1, y1, x2, y2))

if len(filtered_boxes) == 0:
    st.warning("⚠️ Crack present, but not structurally significant.")
    st.stop()

# ---------------- DRAW BOXES ----------------
annotated = img_np.copy()
crack_lengths = []

for i, (x1, y1, x2, y2) in enumerate(filtered_boxes, start=1):
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(
        annotated,
        f"{i}",
        (x1, y1 - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )
    crack_lengths.append(max(x2 - x1, y2 - y1))

st.subheader("Detected Crack Regions")
st.image(annotated, use_column_width=True)

# ---------------- FEATURE EXTRACTION ----------------
st.subheader("📏 Extracted Crack Features")

for i, length in enumerate(crack_lengths, start=1):
    st.write(f"• Crack {i}: Length ≈ **{int(length)} pixels**")

max_length = max(crack_lengths)
num_cracks = len(crack_lengths)

# ---------------- SEVERITY LOGIC (FINAL) ----------------
# PANEL-SAFE, ENGINEERING LOGIC
if max_length < 120 and num_cracks == 1:
    severity = "Low"
elif max_length < 300 and num_cracks <= 2:
    severity = "Moderate"
else:
    severity = "Severe"

st.markdown("---")
st.subheader(f"🚦 Severity Level: **{severity}**")

# ---------------- SUGGESTED ACTION ----------------
st.subheader("🛠 Suggested Action")

if severity == "Low":
    st.markdown("""
- Surface sealing  
- Periodic monitoring  
- Prevent moisture ingress
""")
elif severity == "Moderate":
    st.markdown("""
- Crack filling / epoxy injection  
- Waterproof coating  
- Prevent further propagation
""")
else:
    st.markdown("""
- Structural inspection required  
- Professional repair recommended  
- Load assessment & reinforcement
""")
