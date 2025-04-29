import gdown
import os

# Create folders if they don't exist
os.makedirs("ESRGAN/models", exist_ok=True)
os.makedirs("colorization/models", exist_ok=True)

# Download ESRGAN model
esrgan_url = "https://drive.google.com/uc?id=1mkSj291Iy08l8oq-5dCLK-1j5YiRqf8c"
esrgan_output = "ESRGAN/models/RRDB_ESRGAN_x4.pth"
gdown.download(esrgan_url, esrgan_output, quiet=False)

# Download colorization caffemodel
color_model_url = "https://drive.google.com/uc?id=1v9ExZ4JTstWSJJS3qrM7YZf5EMKADAWL"
color_model_output = "colorization/models/colorization_release_v2.caffemodel"
gdown.download(color_model_url, color_model_output, quiet=False)

print("✅ Models downloaded successfully.")
