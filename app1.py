import streamlit as st
import cv2
import numpy as np
import os
import subprocess
import uuid
from PIL import Image
import tempfile
import glob
import time

# ------------ ENHANCEMENT FUNCTION FIRST ------------
def enhance_image(input_path, esrgan_dir):
    test_script = os.path.join(esrgan_dir, "test.py")
    if not os.path.exists(test_script):
        raise FileNotFoundError("test.py not found in ESRGAN directory")

    input_image = cv2.imread(input_path)
    if input_image is None:
        raise ValueError(f"Could not load image for enhancement: {input_path}")

    resized_path = None
    h_c, w_c = input_image.shape[:2]
    rf = min(1.0, 400 / max(h_c, w_c))
    if rf < 1.0:
        resized_path = os.path.join(os.path.dirname(input_path), f"resized_input_{uuid.uuid4().hex}.png")
        resized_image = cv2.resize(input_image, (int(w_c * rf), int(h_c * rf)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(resized_path, resized_image)
        input_path = resized_path

    # Clear previous results
    results_dir = os.path.join(esrgan_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    for file in glob.glob(os.path.join(results_dir, "*.png")):
        os.remove(file)

    # Run ESRGAN
    subprocess.run([
        "python", "test.py",
        "--input", input_path,
        "--output_folder", results_dir,
        "--model_path", os.path.join(esrgan_dir, "models", "RRDB_ESRGAN_x4.pth")
    ], cwd=esrgan_dir, check=True)

    time.sleep(1)

    list_of_files = glob.glob(os.path.join(results_dir, "*.png"))
    if not list_of_files:
        raise FileNotFoundError("No output file generated in ESRGAN results folder.")

    latest_file = max(list_of_files, key=os.path.getctime)
    final_path = os.path.join(results_dir, f"enhanced_{uuid.uuid4().hex}.png")
    os.rename(latest_file, final_path)

    return final_path, resized_path

# ------------ COLORIZATION FUNCTION SECOND ------------
def colorize_image(image_path, model_dir, resize_factor=0.5):
    PROTOTXT = os.path.join(model_dir, "colorization_deploy_v2.prototxt")
    CAFFEMODEL = os.path.join(model_dir, "colorization_release_v2.caffemodel")
    POINTS = os.path.join(model_dir, "pts_in_hull.npy")

    net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFEMODEL)
    pts = np.load(POINTS)
    pts = pts.transpose().reshape(2, 313, 1, 1)

    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Invalid image file.")

    if 0 < resize_factor < 1.0:
        h, w = image.shape[:2]
        image = cv2.resize(image, (int(w * resize_factor), int(h * resize_factor)), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    target_size = 224
    scale = min(target_size / h, target_size / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(lab, (new_w, new_h))
    L = cv2.split(resized)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (w, h))
    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    output_path = os.path.join(os.path.dirname(image_path), f"colorized_{uuid.uuid4().hex}.png")
    cv2.imwrite(output_path, colorized)
    return output_path

# ------------ STREAMLIT APP UI ------------
st.title("Image Enhancement and Colorization Using Deep Learning")

uploaded_file = st.file_uploader("📤 Upload a black and white image", type=["png", "jpg", "jpeg"])
resize_factor = st.slider("📏 Resize factor before colorization", 0.1, 1.0, 0.5, 0.1)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_input:
        temp_input.write(uploaded_file.read())
        temp_input_path = temp_input.name

    st.image(temp_input_path, caption="📂 Original B&W Image", use_container_width=True)

    if st.button("✨ Process Image"):
        with st.spinner("🔧 Enhancing with ESRGAN..."):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(base_dir, "model")
            esrgan_dir = os.path.join(base_dir, "..", "ESRGAN")  # Adjust if needed

            try:
                enhanced_path, resized_path = enhance_image(temp_input_path, esrgan_dir)

                if resized_path:
                    st.image(resized_path, caption="📐 Resized Image (Before Enhancement)", use_container_width=True)

                st.image(enhanced_path, caption="🔧 Enhanced Image", use_container_width=True)

                with st.spinner("🎨 Colorizing..."):
                    colorized_path = colorize_image(enhanced_path, model_dir, resize_factor)

                if os.path.exists(colorized_path):
                    st.image(colorized_path, caption="🌈 Final Colorized Image", use_container_width=True)
                    with open(colorized_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Colorized Image",
                            data=file,
                            file_name="colorized_output.png",
                            mime="image/png"
                        )
                else:
                    st.error("❌ Colorization failed. Check model output.")

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
