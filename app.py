import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Crack Detection & Severity Analysis",
    layout="centered"
)

st.title("🛣️ Crack Detection & Severity Analysis")

# ---------------- HELPERS ----------------
def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray, blur

def detect_cracks(gray):
    # Adaptive thresholding (best for concrete textures)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3
    )
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return clean

def skeletonize_image(binary):
    binary_bool = binary > 0
    skeleton = skeletonize(binary_bool)
    return skeleton

def analyze_cracks(skeleton):
    labeled = label(skeleton)
    cracks = []

    for region in regionprops(labeled):
        if region.area < 30:
            continue  # remove noise
        cracks.append(region.area)  # length in pixels

    return cracks, labeled

def severity_from_length(total_length):
    if total_length < 300:
        return "Low"
    elif total_length < 700:
        return "Moderate"
    else:
        return "High"

def suggested_action(severity):
    if severity == "Low":
        return [
            "Monitor crack growth",
            "Seal surface if required"
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

def draw_cracks(image, labeled):
    output = image.copy()
    h, w, _ = output.shape

    for region in regionprops(labeled):
        if region.area < 30:
            continue

        coords = region.coords
        for y, x in coords:
            output[y, x] = [255, 255, 0]  # YELLOW skeleton

        cy, cx = region.centroid
        cv2.putText(
            output,
            f"{int(region.label)}",
            (int(cx), int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

    return output

# ---------------- UI ----------------
uploaded_file = st.file_uploader(
    "Upload crack image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.image(img_np, caption="Uploaded Image", use_column_width=True)

    gray, blur = preprocess(img_np)
    binary = detect_cracks(gray)
    skeleton = skeletonize_image(binary)

    crack_lengths, labeled = analyze_cracks(skeleton)

    if len(crack_lengths) == 0:
        st.warning("No cracks detected in the image.")
    else:
        overlay = draw_cracks(img_np, labeled)
        st.image(
            overlay,
            caption="Detected Cracks (Skeleton-based)",
            use_column_width=True
        )

        st.subheader("📏 Extracted Crack Features")
        total_length = 0
        for i, length in enumerate(crack_lengths, start=1):
            st.write(f"Crack {i} → Length: {length} pixels")
            total_length += length

        severity = severity_from_length(total_length)

        st.subheader(f"🚦 Severity: {severity}")
        st.write(f"Total Crack Length: {total_length} pixels")

        st.subheader("🛠 Suggested Action")
        for act in suggested_action(severity):
            st.write(f"• {act}")
