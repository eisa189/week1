# generate_csv.py
# Create all_data.csv mapping image filenames to their class labels

import os
import csv

# Base directory where raw_data is stored
raw_data_dir = "./data/raw_data"
output_csv = "./data/all_data.csv"

# The 6 expected class labels
classes = [
    "non_damage",
    "damaged_infrastructure",
    "damaged_nature",
    "fires",
    "flood",
    "human_damage",
]

rows = []

for cls in classes:
    img_dir = os.path.join(raw_data_dir, cls, "images")
    if not os.path.exists(img_dir):
        print(f"⚠️ No images found for {cls}, skipping.")
        continue

    for fname in os.listdir(img_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            rows.append([fname, cls])

# Write CSV
with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image", "label"])
    writer.writerows(rows)

print(f"✅ CSV created at {output_csv} with {len(rows)} entries.")


