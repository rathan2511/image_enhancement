import streamlit as st
import numpy as np
import cv2
import os
from io import BytesIO
from PIL import Image

# Define the base directory
BASE_DIR = r"C:\tbptp"

# Paths to model files
PROTOTXT = os.path.join(BASE_DIR, "model", "colorization_deploy_v2.prototxt")
POINTS = os.path.join(BASE_DIR, "model", "pts_in_hull.npy")
MODEL = os.path.join(BASE_DIR, "model", "colorization_release_v2.caffemodel")

# Load the pre-trained model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Adjust model layers for proper colorization
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Streamlit UI
st.title("🎨 AI-Powered Image Colorization")
st.write("Upload a black-and-white image, and the AI will colorize it for you!")

uploaded_file = st.file_uploader("Upload a black & white image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    h, w = image.shape[:2]  # Get original dimensions

    # Convert to LAB color space
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize While Maintaining Aspect Ratio
    target_size = 224  # Model input size
    scale = min(target_size / h, target_size / w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(lab, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Extract L channel
    L_resized = cv2.split(resized)[0]
    L_resized -= 50  # Normalize

    # Perform colorization
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Warp back to original size
    ab = cv2.resize(ab, (w, h), interpolation=cv2.INTER_CUBIC)

    # Merge with original L channel and convert back to BGR
    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Convert to PIL format for Streamlit
    original_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    colorized_pil = Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))

    # Show images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_pil, caption="📷 Original Image", use_column_width=True)
    with col2:
        st.image(colorized_pil, caption="🎨 Colorized Image", use_column_width=True)

    # Download Button
    img_bytes = BytesIO()
    colorized_pil.save(img_bytes, format="PNG")
    st.download_button("⬇️ Download Colorized Image", img_bytes.getvalue(), "colorized.png", "image/png")
