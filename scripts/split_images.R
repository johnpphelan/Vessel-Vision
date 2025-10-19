# Paths
source_dir <- "./images/CCSS"
dest_base <- "./split_images"

# Create destination folders if they don't exist
dir.create(dest_base, showWarnings = FALSE)
for (i in 1:4) {
  dir.create(file.path(dest_base, as.character(i)), showWarnings = FALSE)
}

# List all images
all_images <- list.files(source_dir, pattern = "\\.jpg$", full.names = TRUE)

# Sample 1000 random images
set.seed(42)
sample_images <- sample(all_images, 1000)

# Split into 4 groups of 250
split_images <- split(sample_images, ceiling(seq_along(sample_images)/250))

# Copy images to folders
for (i in seq_along(split_images)) {
  dest_folder <- file.path(dest_base, as.character(i))
  file.copy(split_images[[i]], dest_folder)
}

cat("Done! Images split into 4 folders under", dest_base, "\n")
