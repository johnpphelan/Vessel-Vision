#%%
import os
import re
from datetime import datetime, timezone
import csv
from pathlib import Path
from astral import LocationInfo
from astral.sun import sun
from tqdm import tqdm

# ----------------------------
# Configuration
# ----------------------------
IMAGE_DIR = "../images/CCSS"
OUTPUT_CSV = "./image_segmentation.csv"

LUX_DATA_FILE = "./sensor_data.csv"  # optional, can be empty
LUX_THRESHOLD = 50
LUX_TIME_TOLERANCE = 300

LOCATION = "campbell_river"  # 'vancouver', 'campbell_river', 'china_creek', 'average'

# Filename pattern
TIMESTAMP_PATTERN = r'_(\d{8}T\d{6}\.\d{3}Z)\.'
#%%
# ----------------------------
# Astral and lux functions
# ----------------------------
def get_location_info(loc_key):
    loc = {
        'vancouver': {'name': 'Vancouver','region': 'BC, Canada','latitude': 49.2827,'longitude': -123.1207,'timezone': 'America/Vancouver'},
        'campbell_river': {'name': 'Campbell River','region': 'BC, Canada','latitude': 50.0163,'longitude': -125.2447,'timezone': 'America/Vancouver'},
        'china_creek': {'name': 'China Creek (Port Alberni)','region': 'BC, Canada','latitude': 49.2394,'longitude': -124.8053,'timezone': 'America/Vancouver'}
    }[loc_key]
    return LocationInfo(loc['name'], loc['region'], loc['timezone'], loc['latitude'], loc['longitude'])

def get_sunrise_sunset(dt, location_key):
    from datetime import timedelta
    location = get_location_info(location_key)
    s = sun(location.observer, date=dt.date())
    sunrise, sunset = s['sunrise'], s['sunset']
    if sunset < sunrise:
        sunset += timedelta(days=1)
    return sunrise, sunset

def get_average_sunrise_sunset(dt):
    from datetime import timedelta
    all_sunrise, all_sunset = [], []
    for loc_key in ['vancouver', 'campbell_river', 'china_creek']:
        sunrise, sunset = get_sunrise_sunset(dt, loc_key)
        all_sunrise.append(sunrise)
        all_sunset.append(sunset)
    avg_sunrise_ts = sum(s.timestamp() for s in all_sunrise)/3
    avg_sunset_ts = sum(s.timestamp() for s in all_sunset)/3
    from datetime import datetime, timezone
    avg_sunrise = datetime.fromtimestamp(avg_sunrise_ts, tz=timezone.utc)
    avg_sunset = datetime.fromtimestamp(avg_sunset_ts, tz=timezone.utc)
    return avg_sunrise, avg_sunset

def load_lux_data_csv(filepath):
    import csv
    from datetime import datetime, timezone
    lux_data = {}
    if not os.path.exists(filepath):
        print(f"Warning: Lux data file '{filepath}' not found. Using time-based calculation only.")
        return lux_data
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp_str, lux_value = None, None
            for key in ['timestamp', 'time', 'datetime', 'date_time']:
                if key in row: timestamp_str = row[key]; break
            for key in ['lux', 'light', 'light_level', 'illuminance', 'brightness']:
                if key in row: lux_value = float(row[key]); break
            if timestamp_str and lux_value is not None:
                for fmt in ['%Y%m%dT%H%M%S.%fZ','%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S.%fZ','%Y-%m-%dT%H:%M:%SZ']:
                    try:
                        dt = datetime.strptime(timestamp_str, fmt)
                        if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                        lux_data[dt] = lux_value
                        break
                    except: continue
    print(f"Loaded {len(lux_data)} lux readings from {filepath}")
    return lux_data

def parse_timestamp_from_filename(filename):
    import re
    from datetime import datetime, timezone
    match = re.search(TIMESTAMP_PATTERN, filename)
    if not match: return None
    dt = datetime.strptime(match.group(1), '%Y%m%dT%H%M%S.%fZ')
    dt = dt.replace(tzinfo=timezone.utc)
    return dt

def get_lux_reading(dt, lux_data, tolerance_seconds=300):
    if not lux_data: return None, None
    closest_time, closest_lux, min_diff = None, None, float('inf')
    for sensor_time, lux_value in lux_data.items():
        diff = abs((dt - sensor_time).total_seconds())
        if diff < min_diff and diff <= tolerance_seconds:
            min_diff = diff
            closest_time, closest_lux = sensor_time, lux_value
    return closest_lux, min_diff if closest_lux is not None else None

def determine_segment(dt, location, lux_data, lux_threshold):
    lux_value, _ = get_lux_reading(dt, lux_data, LUX_TIME_TOLERANCE)
    if lux_value is not None:
        return ("day" if lux_value >= lux_threshold else "night", "lux_sensor")
    if location=='average': sunrise,sunset = get_average_sunrise_sunset(dt)
    else: sunrise,sunset = get_sunrise_sunset(dt, location)
    return ("day" if sunrise <= dt < sunset else "night", "time_based")

# ----------------------------
# Process images
# ----------------------------
def process_images(image_dir, location, lux_data, lux_threshold):
    results=[]
    image_dir = Path(image_dir)
    if not image_dir.exists(): 
        print(f"Error: Directory '{image_dir}' does not exist"); return results
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))
    print(f"Found {len(image_files)} image files")
    for image_path in tqdm(image_files, desc="Processing images"):
        dt = parse_timestamp_from_filename(image_path.name)
        if not dt:
            print(f"Warning: Could not parse timestamp from filename '{image_path.name}'")
            continue
        segment, method = determine_segment(dt, location, lux_data, lux_threshold)
        results.append({'filename':image_path.name, 'timestamp_utc':dt.isoformat(), 'segment':segment, 'method':method})
    return results

# ----------------------------
# Save CSV
# ----------------------------
def save_results_to_csv(results, output_file):
    import csv
    with open(output_file,'w',newline='') as csvfile:
        fieldnames=['filename','timestamp_utc','segment','method']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results: writer.writerow(row)
    print(f"Results saved to {output_file}")


# %%
def main():
    print("IMAGE_DIR:", IMAGE_DIR)
    print("OUTPUT_CSV:", OUTPUT_CSV)
    print("Exists?", os.path.exists(IMAGE_DIR))
    
    lux_data = load_lux_data_csv(LUX_DATA_FILE)
    results = process_images(IMAGE_DIR, LOCATION, lux_data, LUX_THRESHOLD)
    
    print("Number of images processed:", len(results))
    
    if results:
        save_results_to_csv(results, OUTPUT_CSV)
    else:
        print("No images processed, CSV will not be created.")

#%%

IMAGE_DIR = "../images/CCSS/"
image_dir = Path(IMAGE_DIR)
print("Exists:", image_dir.exists())
print("Files:", list(image_dir.glob("*")))

image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + \
              list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.JPEG")) + \
              list(image_dir.glob("*.png")) + list(image_dir.glob("*.PNG"))

for image_path in image_files:
    print("Checking:", image_path.name)
    dt = parse_timestamp_from_filename(image_path.name)
    if not dt:
        print("Could not parse timestamp:", image_path.name)
        continue
# %%
