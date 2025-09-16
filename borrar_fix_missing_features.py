# fix_missing_features.py
# Create zero-vector .npy files (dim 768) for any images in all_data.csv
# that don't already have a feature file in ./data/feature/

import os
import csv
import numpy as np

csv_path = "./data/all_data.csv"
feature_dir = "./data/feature"
os.makedirs(feature_dir, exist_ok=True)

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_name = os.path.splitext(row["image"])[0]
        npy_path = os.path.join(feature_dir, base_name + ".npy")
        if not os.path.exists(npy_path):
            np.save(npy_path, np.zeros(768, dtype=np.float32))
            print(f"✅ Created missing feature: {npy_path}")

print("Done! All missing features fixed.")
