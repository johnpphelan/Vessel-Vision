#install.packages(c("imager", "keras", "stats", "purrr"))
library(imager)
library(keras)
library(stats)
library(purrr)

#install_keras()


# Load VGG16 pretrained model
model <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,      # drop classification layers
  input_shape = c(224,224,3)
)
process_image <- function(img_path){
  # Load and resize image
  img <- image_load(img_path, target_size = c(224,224))
  x <- image_to_array(img)
  x <- array_reshape(x, c(1,224,224,3))
  
  # Preprocess for VGG16
  x <- imagenet_preprocess_input(x)
  
  # Extract features
  features <- model %>% predict(x)
  as.numeric(features)  # flatten to vector
}

image_files <- list.files("./images/CCSS/", pattern="\\.jpg$", full.names=TRUE)

image_files_subset <- image_files[1:10]
#images <- lapply(image_files_subset, load.image)
features_list <- map(image_files_subset, process_image)
feature_matrix <- do.call(rbind, features_list)

set.seed(42)
k <- 2  # rough: 1=boat, 2=non-boat
km <- kmeans(feature_matrix, centers = k)

# Add cluster info
cluster_assignments <- km$cluster
test = data.frame(image=image_files_subset, cluster=cluster_assignments)

boat_images <- image_files_subset[cluster_assignments == 1] # cluster 1 = boats
# Number of images to display
boat_images_abs <- normalizePath(boat_images)

# Open each image in default viewer
for (img_path in boat_images_abs) {
  shell.exec(img_path)
}

not_boats <- image_files_subset[cluster_assignments == 2] # cluster 2 = non-boats
not_boat_images_abs <- normalizePath(not_boats)
# Open each image in default viewer
for (img_path in not_boat_images_abs) {
  shell.exec(img_path)
}
# 
# count_objects <- function(img_path){
#   # Load at smaller size
#   img <- load.image(img_path)
#   
#   # Resize to something manageable, e.g., 512x512
#   img <- imresize(img, 512, 512)
#   
#   # Convert to grayscale
#   gray_img <- grayscale(img)
#   
#   # Threshold and label
#   bin <- threshold(gray_img, "75%")
#   lab <- suppressWarnings(suppressMessages(label(bin)))
#   
#   # Count objects
#   max(lab)
# }
# library(purrr)
# # boat_counts <- map_int(boat_images, count_objects)
# # Apply to boat-like images
# boat_images <- image_files_subset[cluster_assignments == 1] # cluster 1 = boats
# boat_counts <- map_int(boat_images, count_objects)
# 
# data.frame(image=boat_images, boat_count=boat_counts)
