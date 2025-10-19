# -----------------------------
# Libraries
# -----------------------------
library(imager)
library(keras)
library(purrr)
library(magick)
library(dplyr)
library(officer)
library(magrittr)

# -----------------------------
# Step 1: Crop bottom portion (remove sky)
# -----------------------------
crop_bottom <- function(img_path, save_path = NULL, bottom_fraction = 0.8) {
  img <- image_read(img_path)
  width <- image_info(img)$width
  height <- image_info(img)$height
  top_crop <- round(height * (1 - bottom_fraction))
  
  # Crop bottom_fraction of the image
  cropped_img <- image_crop(img, geometry = geometry_area(width = width, height = round(height*bottom_fraction), x = 0, y = top_crop))
  
  if(!is.null(save_path)) {
    image_write(cropped_img, path = save_path)
  }
  return(cropped_img)
}

# -----------------------------
# Step 2: Process 50 images
# -----------------------------
image_files <- list.files("./images/CCSS/", pattern="\\.jpg$", full.names=TRUE)
image_subset <- image_files[1:50]

cropped_folder <- "cropped_boats"
dir.create(cropped_folder, showWarnings = FALSE)

cropped_subset <- c()
for(img_path in image_subset) {
  save_path <- file.path(cropped_folder, basename(img_path))
  crop_bottom(img_path, save_path)
  cropped_subset <- c(cropped_subset, save_path)
}

# -----------------------------
# Step 3: Feature extraction from central-lower region
# -----------------------------
extract_center_patches <- function(img_path, patch_size = 224, bottom_fraction = 0.5) {
  img <- image_read(img_path)
  
  # Resize image to manageable size
  img <- image_scale(img, "512x512")  # adjust if needed
  
  info <- image_info(img)
  w <- info$width
  h <- info$height
  
  # Define central-lower crop
  x_start <- round(w * 0.25)
  y_start <- round(h * (1 - bottom_fraction))
  
  cropped <- image_crop(img, geometry = geometry_area(width = patch_size, height = patch_size, x = x_start, y = y_start))
  
  # Convert to array
  arr <- as.numeric(image_data(cropped))
  x_array <- array(arr, dim = c(1, patch_size, patch_size, 3))
  x_array <- imagenet_preprocess_input(x_array)
  
  # Extract CNN features
  feat <- model %>% predict(x_array)
  return(as.numeric(feat))
}

# Load pre-trained VGG16
model <- application_vgg16(weights = "imagenet", include_top = FALSE, input_shape = c(224,224,3))

# Extract features from cropped central-lower patches
feature_list <- map(cropped_subset, extract_center_patches)
feature_matrix <- do.call(rbind, feature_list)

# -----------------------------
# Step 4: K-means clustering
# -----------------------------
set.seed(42)
n_unique <- nrow(unique(feature_matrix))
k <- min(2, n_unique)  # prevent error if identical vectors

km <- kmeans(feature_matrix, centers = k)
cluster_assignments <- km$cluster
table(cluster_assignments)

# -----------------------------
# Step 5: PowerPoint slideshow
# -----------------------------
boat_clusters <- c(1)  # adjust based on table
boat_images <- cropped_subset[cluster_assignments %in% boat_clusters]

ppt <- read_pptx()
for (img_path in boat_images) {
  ppt <- ppt %>% 
    add_slide(layout = "Title and Content", master = "Office Theme") %>%
    ph_with(external_img(img_path, width = 8, height = 6), location = ph_location_type(type = "body"))
}

print(ppt, target = "boat_slideshow_center_lower_50.pptx")
