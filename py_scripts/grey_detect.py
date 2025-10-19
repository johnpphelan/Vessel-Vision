#%%
import os
import random
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
IMAGE_DIR = "../images/CCSS/"
OUTPUT_CSV = "./grey_image_tags.csv"
BRIGHTNESS_THRESH = 100      # max mean brightness to consider "dark"
COLOR_STD_THRESH = 20        # max RGB std to consider "grey"
RESIZE_MAX = 1024            # downscale large images for faster processing
N_SAMPLES = 50               # number of images to randomly select for testing

# ----------------------------
# Helper function
# ----------------------------
def is_grey_image(img_path, brightness_thresh=100, color_std_thresh=20, resize_max=1024):
    """Return True if image is mostly grey/dark."""
    try:
        img = Image.open(img_path).convert("RGB")
        img.thumbnail((resize_max, resize_max))
        arr = np.array(img)
        mean_brightness = arr.mean()
        std_rgb = arr.std(axis=(0,1))
        return mean_brightness < brightness_thresh and std_rgb.mean() < color_std_thresh
    except:
        return False

# ----------------------------
# Load existing CSV if it exists
# ----------------------------
if os.path.exists(OUTPUT_CSV):
    print(f"CSV already exists. Loading {OUTPUT_CSV}...")
    df = pd.read_csv(OUTPUT_CSV)
else:
    # Scan all images
    all_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(".jpg")]

    grey_flags = []
    print("Identifying mostly grey images...")
    for img_path in tqdm(all_images):
        grey_flags.append(is_grey_image(img_path, BRIGHTNESS_THRESH, COLOR_STD_THRESH, RESIZE_MAX))

    # Save results to CSV
    df = pd.DataFrame({
        "image": [os.path.basename(p) for p in all_images],
        "is_grey": grey_flags
    })
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved grey image tags to {OUTPUT_CSV}")

# ----------------------------
# Randomly select 50 grey images for testing
# ----------------------------
grey_images = df[df["is_grey"]]["image"].tolist()
random.seed(42)
grey_subset = random.sample(grey_images, min(N_SAMPLES, len(grey_images)))
print(f"Random 50 grey images: {grey_subset}")