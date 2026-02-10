import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av

# ---------------- PAGE ----------------
st.set_page_config(page_title="Crack Detection (Live)", layout="wide")
st.title("Crack Detection & Severity Analysis – Live Camera")

st.info("Live camera stream using WebRTC. Capture frame to analyze cracks.")

# ---------------- VIDEO PROCESSOR ----------------
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

# ---------------- START CAMERA ----------------
ctx = webrtc_streamer(
    key="live-cam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# ---------------- CAPTURE FRAME ----------------
if ctx.video_processor and ctx.video_processor.frame is not None:
    if st.button("📸 Capture & Analyze Frame"):
        img = ctx.video_processor.frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---------------- CRACK DETECTION ----------------
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(
            dilated,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        H, W = gray.shape
        cracks = []
        lengths = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            length = max(w, h)

            # STRICT FILTERS (THIS FIXES FALSE CRACKS)
            if length < 80:
                continue
            if w > 0.7 * W:
                continue

            cracks.append((x, y, w, h, length))
            lengths.append(length)

        if len(cracks) == 0:
            st.error("❌ No significant cracks detected.")
            st.stop()

        # ---------------- DRAW BOXES ----------------
        annotated = img.copy()
        for i, (x, y, w, h, l) in enumerate(cracks, start=1):
            cv2.rectangle(
                annotated,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                3
            )
            cv2.putText(
                annotated,
                str(i),
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        # ---------------- DISPLAY ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Captured Frame")
            st.image(img, channels="BGR", use_column_width=True)

        with col2:
            st.subheader("Detected Cracks")
            st.image(annotated, channels="BGR", use_column_width=True)

        # ---------------- FEATURES ----------------
        st.subheader("📏 Extracted Crack Features")
        for i, l in enumerate(lengths, start=1):
            st.write(f"• Crack {i}: Length ≈ **{int(l)} pixels**")

        # ---------------- SEVERITY ----------------
        max_len = max(lengths)
        count = len(lengths)

        if max_len < 150 and count == 1:
            severity = "Low"
        elif max_len < 350 and count <= 2:
            severity = "Moderate"
        else:
            severity = "Severe"

        st.markdown("---")
        st.subheader(f"🚦 Severity: **{severity}**")

        # ---------------- ACTION ----------------
        st.subheader("🛠 Suggested Action")
        if severity == "Low":
            st.write("• Surface sealing\n• Periodic monitoring")
        elif severity == "Moderate":
            st.write("• Crack filling\n• Waterproof coating")
        else:
            st.write("• Structural inspection\n• Professional repair required")
