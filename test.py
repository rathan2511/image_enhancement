import os
import os.path as osp
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch  # Make sure this file is in the same folder
import argparse

# Argument parsing for input image and model path
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Path to the input image')
parser.add_argument('--model_path', type=str, default='models/RRDB_ESRGAN_x4.pth', help='Path to the ESRGAN model') # Added model path as argument
parser.add_argument('--output_folder', type=str, default='results', help='Path to the output folder') # Added output folder as argument
args = parser.parse_args()

# Define paths - now using arguments
model_path = args.model_path
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
model.eval()
model = model.to(device)

print(f"Model loaded from {model_path}\nTesting...")

# Output folder
output_folder = args.output_folder
os.makedirs(output_folder, exist_ok=True)

# Input image path from argument
input_path = args.input
base = osp.splitext(osp.basename(input_path))[0]

# Load image
img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
if img is None:
    print(f"Error: Could not load image from {input_path}.")
    exit()

print(f"\nProcessing: {base}")
print(f"Original shape: {img.shape}")

# Handle grayscale images
if len(img.shape) == 2:  # Single channel grayscale
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
elif img.shape[2] == 1:  # (H, W, 1) grayscale with channel
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # Convert to RGB as ESRGAN expects 3 channels

# Normalize to 0-1
img = img.astype(np.float32) / 255.0
img_tensor = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).unsqueeze(0).to(device)

with torch.no_grad():
    try:
        output = model(img_tensor).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    except Exception as e:
        print(f"Error during processing: {e}")
        exit()

# Post-process and save
if np.isnan(output).any():
    print(f"Error: Output contains NaN values.")
    exit()

output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # CHW to HWC and RGB channel order
output = (output * 255.0).astype(np.uint8)

output_path = osp.join(output_folder, f"{base}_rlt.png")
success = cv2.imwrite(output_path, output)

if success:
    print(f"🚀 Enhanced image saved to: {output_path}")
else:
    print(f"❌ Failed to save enhanced image to: {output_path}")

print("\nEnhancement process finished.")