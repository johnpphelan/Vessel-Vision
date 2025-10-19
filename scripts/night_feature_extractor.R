library(imager)
library(keras)
library(purrr)
library(magick)
library(dplyr)
library(officer)
library(magrittr)
library(reticulate)
library(data.table)


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

csv_file <- "image_segmentation.csv"
df <- fread(csv_file)
night_df <- df[segment == "night"]
images<-list.files("./images/CCSS/", pattern="\\.jpg$", full.names=TRUE)
#now read in only those images with night
night_images <- images[basename(images) %in% night_df$image]

set.seed(42)  # for reproducibility
night_images_subset <- sample(night_images, 50)


library(magick)

# night_images_subset: character vector of 50 night image paths
cropped_images <- list()

# Parameters
top_crop_ratio <- 0.2 # fraction of the image height to remove from top

for (img_path in night_images_subset) {
  # Read image
  img <- image_read(img_path)
  
  # Get dimensions
  info <- image_info(img)
  width <- info$width
  height <- info$height
  
  # Calculate cropping box: remove top 30%
  crop_height <- height * (1 - top_crop_ratio)
  crop_width <- width
  crop_geometry <- geometry_area(width = crop_width, height = crop_height, x = 0, y = height * top_crop_ratio)
  
  # Crop image
  img_cropped <- image_crop(img, crop_geometry)
  
  # Store cropped image in list
  cropped_images[[basename(img_path)]] <- img_cropped
}

# Example: display first cropped image
print(cropped_images[[1]])

library(reticulate)
py_install(c("opencv-python", "numpy"))

cv2 <- import("cv2")
np <- import("numpy")

segment_water_py <- function(img_path) {
  cv2 <- import("cv2")
  np <- import("numpy")
  
  # Read file as raw bytes and decode — this forces OpenCV to handle any bit depth
  f <- reticulate::py_eval("open", convert = TRUE)(img_path, "rb")
  buf <- f$read()
  f$close()
  
  nparr <- np$frombuffer(buf, dtype = "uint8")
  img <- cv2$imdecode(nparr, cv2$IMREAD_COLOR)
  
  if (is.null(img)) {
    warning(paste("Cannot read:", img_path))
    return(NULL)
  }
  
  # Convert to HSV
  hsv <- cv2$cvtColor(img, cv2$COLOR_BGR2HSV)
  
  # Define rough HSV bounds for water
  lower_blue <- np$array(c(85, 20, 20), dtype = "uint8")
  upper_blue <- np$array(c(140, 255, 255), dtype = "uint8")
  
  mask <- cv2$inRange(hsv, lower_blue, upper_blue)
  
  # Morphological smoothing
  kernel <- cv2$getStructuringElement(cv2$MORPH_ELLIPSE, c(7L, 7L))
  mask <- cv2$morphologyEx(mask, cv2$MORPH_CLOSE, kernel)
  
  # Apply mask
  result <- cv2$bitwise_and(img, img, mask = mask)
  
  return(result)
}

tmp_dir <- "./tmp_seg"
dir.create(tmp_dir, showWarnings = FALSE)

tmp_dir <- "./tmp_seg"
dir.create(tmp_dir, showWarnings = FALSE)

segmented_images <- list()

for (name in names(cropped_images)) {
  tmp_path <- file.path(tmp_dir, paste0("tmp_", tools::file_path_sans_ext(name), ".png"))
  
  # Remove alpha channel, fill with white background
  img_noalpha <- image_background(cropped_images[[name]], "white")
  image_write(img_noalpha, tmp_path, format = "png", depth = 8)
  Sys.sleep(0.1)
  
  # Normalize path for Python
  py_path <- normalizePath(tmp_path, winslash = "/")
  
  seg <- segment_water_py(py_path)
  if (is.null(seg)) warning(paste("Segmentation failed for", name))
  
  segmented_images[[name]] <- seg
}

cv2$imwrite("seg_test.jpg", segmented_images[[1]])
image_read("seg_test.jpg")