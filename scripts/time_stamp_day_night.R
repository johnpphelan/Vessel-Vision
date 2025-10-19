# -----------------------------
# Libraries
# -----------------------------
library(data.table)
library(lubridate)
library(stringr)
library(suncalc)
library(dplyr)

# -----------------------------
# Configuration
# -----------------------------
image_dir <- "./images/CCSS/"       # folder with images
output_csv <- "image_segmentation.csv"

lux_csv_file <- "./sensor_data.csv"  # optional Lux CSV
lux_threshold <- 50                  # Lux threshold
lux_time_tolerance <- 300            # seconds

location <- "campbell_river"         # 'vancouver', 'campbell_river', 'china_creek', 'average'

locations <- list(
  vancouver = list(name="Vancouver", lat=49.2827, lon=-123.1207),
  campbell_river = list(name="Campbell River", lat=50.0163, lon=-125.2447),
  china_creek = list(name="China Creek", lat=49.2394, lon=-124.8053)
)

timestamp_pattern <- "_(\\d{8}T\\d{6}\\.\\d{3}Z)\\."  # e.g., image_20230901T210530.000Z.jpg

# -----------------------------
# 1️⃣ Load image filenames & parse timestamps
# -----------------------------
image_files <- list.files(image_dir, pattern="\\.(jpg|jpeg|png)$", full.names = TRUE)
filenames <- basename(image_files)

timestamps_str <- str_match(filenames, timestamp_pattern)[,2]
timestamps <- ymd_hms(timestamps_str, tz="UTC")

image_dt <- data.table(
  image = filenames,
  filepath = image_files,
  timestamp = timestamps
)

# Remove images without valid timestamp
image_dt <- image_dt[!is.na(timestamp)]

# -----------------------------
# 2️⃣ Load Lux sensor data (if exists)
# -----------------------------
lux_dt <- data.table()
if (file.exists(lux_csv_file)) {
  lux_temp <- fread(lux_csv_file)
  ts_col <- names(lux_temp)[tolower(names(lux_temp)) %in% c("timestamp","time","datetime","date_time")][1]
  lux_col <- names(lux_temp)[tolower(names(lux_temp)) %in% c("lux","light","light_level","illuminance","brightness")][1]
  
  if (!is.na(ts_col) & !is.na(lux_col)) {
    lux_temp[, timestamp := ymd_hms(get(ts_col), tz="UTC")]
    lux_temp[, lux := as.numeric(get(lux_col))]
    lux_dt <- lux_temp[, .(timestamp, lux)]
    setkey(lux_dt, timestamp)
  }
}

# -----------------------------
# 3️⃣ Precompute sunrise/sunset per unique date
# -----------------------------
unique_dates <- unique(as.Date(image_dt$timestamp))
sun_table <- data.table(date = unique_dates)

if (location == "average") {
  sun_table[, c("sunrise","sunset") := {
    sunrise_list <- c(); sunset_list <- c()
    for (loc_key in names(locations)) {
      s <- getSunlightTimes(date=date, lat=locations[[loc_key]]$lat, lon=locations[[loc_key]]$lon, tz="UTC")
      sunrise_list <- c(sunrise_list, s$sunrise)
      sunset_list <- c(sunset_list, s$sunset)
    }
    list(as_datetime(mean(as.numeric(sunrise_list)), tz="UTC"),
         as_datetime(mean(as.numeric(sunset_list)), tz="UTC"))
  }, by=date]
} else {
  loc <- locations[[location]]
  sun_table[, c("sunrise","sunset") := {
    s <- getSunlightTimes(date=date, lat=loc$lat, lon=loc$lon, tz="UTC")
    list(s$sunrise, s$sunset)
  }, by=date]
}

# Merge sunrise/sunset to images
image_dt[, date := as.Date(timestamp)]
image_dt <- merge(image_dt, sun_table, by="date", all.x=TRUE)

# -----------------------------
# 4️⃣ Match Lux readings safely (vectorized)
# -----------------------------
if (nrow(lux_dt) > 0) {
  if (!"lux" %in% colnames(lux_dt)) lux_dt[, lux := NA_real_]
  image_dt[, lux := lux_dt$lux[findInterval(timestamp, lux_dt$timestamp)]]
} else {
  image_dt[, lux := NA_real_]
}

# -----------------------------
# 5️⃣ Determine day/night segment
# -----------------------------
image_dt[, segment := fifelse(!is.na(lux) & lux >= lux_threshold, "day",
                              fifelse(!is.na(lux) & lux < lux_threshold, "night",
                                      fifelse(timestamp >= sunrise & timestamp < sunset, "day","night")))]

image_dt[, method := fifelse(!is.na(lux), "lux_sensor", "time_based")]

# -----------------------------
# 6️⃣ Save CSV
# -----------------------------
fwrite(image_dt[, .(image, timestamp, segment, method)], output_csv)
cat("Saved segmentation CSV to", output_csv, "\n")
