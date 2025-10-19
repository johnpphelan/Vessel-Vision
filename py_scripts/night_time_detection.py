import os
import random
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# ----------------------------
# Configuration
# ----------------------------
IMAGE_DIR = "./images/CCSS/"
CSV_FILE = "image_segmentation.csv"
OUTPUT_DIR = "./cropped_night_images"
TOP_CROP_RATIO = 0.2  # remove top 20%
N_SAMPLES = 50

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# Load CSV and filter to night
# ----------------------------
df = pd.read_csv(CSV_FILE)
night_df = df[df["segment"] == "night"]

# Get all images
all_images = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)
              if f.lower().endswith(".jpg")]

# Keep only images that appear in the CSV night list
night_images = [img for img in all_images if os.path.basename(img) in night_df["image"].values]

# Randomly sample 50 images (or fewer if not enough)
random.seed(42)
night_subset = random.sample(night_images, min(N_SAMPLES, len(night_images)))

print(f"Selected {len(night_subset)} night images for cropping.")

# ----------------------------
# Crop top portion of image
# ----------------------------
def crop_top(img_path, top_ratio=0.2, save_path=None):
    img = Image.open(img_path)
    width, height = img.size
    top = int(height * top_ratio)
    cropped = img.crop((0, top, width, height))  # (left, upper, right, lower)
    if save_path:
        cropped.save(save_path)
    return cropped

# ----------------------------
# Crop and save all
# ----------------------------
cropped_images = {}

for img_path in night_subset:
    out_path = os.path.join(OUTPUT_DIR, os.path.basename(img_path))
    cropped = crop_top(img_path, top_ratio=TOP_CROP_RATIO, save_path=out_path)
    cropped_images[os.path.basename(img_path)] = cropped

print(f"Cropped {len(cropped_images)} images saved to: {OUTPUT_DIR}")

# ----------------------------
# Display first cropped image
# ----------------------------
first_name = list(cropped_images.keys())[0]
plt.imshow(cropped_images[first_name])
plt.title(f"Cropped Example: {first_name}")
plt.axis("off")
plt.show()
