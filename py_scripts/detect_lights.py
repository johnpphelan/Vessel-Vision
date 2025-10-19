#%%
import cv2
import os
from tqdm import tqdm
import pandas as pd

# Parameters
MIN_AREA = 5
MAX_AREA = 500
THRESHOLD = 150
OUTPUT_BLOBS_CSV = "./grey_images_blobs_spherical.csv"
IMAGE_DIR = "../images/CCSS/"

# Load CSV
df = pd.read_csv("./image_segmentation_grey.csv")
# Assuming you have a 'grey' column; adjust if needed
grey_images = df[df["grey"]]["image"].tolist()

# Function to find roughly circular blobs
def find_spherical_blobs(img_path, threshold=THRESHOLD, min_area=MIN_AREA, max_area=MAX_AREA, circularity_thresh=0.7):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return []

    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * 3.14159 * (area / (perimeter ** 2))
        # circularity = 1.0 for a perfect circle
        if circularity >= circularity_thresh:
            blobs.append(cnt)
    return blobs

# Process all grey images
blob_counts = []

for img_name in tqdm(grey_images):
    img_path = os.path.join(IMAGE_DIR, img_name)
    blobs = find_spherical_blobs(img_path)
    blob_counts.append(len(blobs))

# Save results
df_blobs = pd.DataFrame({
    "image": grey_images,
    "blob_count": blob_counts
})
df_blobs.to_csv(OUTPUT_BLOBS_CSV, index=False)
print(f"Saved blob counts to {OUTPUT_BLOBS_CSV}")
# %%