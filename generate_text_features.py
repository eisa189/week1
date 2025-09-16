# generate_text_features.py
# This script generates 768-dim BERT embeddings for each text file in the dataset
# and saves them as .npy files in ./data/feature/
# Requirements: transformers, torch, numpy

import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel

# Load pre-trained BERT (base-uncased, 768 hidden size)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

# Base paths
raw_data_dir = "./data/raw_data"
feature_dir = "./data/feature"

os.makedirs(feature_dir, exist_ok=True)

# Iterate over each class folder
classes = [
    "non_damage",
    "damaged_infrastructure",
    "damaged_nature",
    "fires",
    "flood",
    "human_damage",
]

for cls in classes:
    text_dir = os.path.join(raw_data_dir, cls, "text")
    if not os.path.exists(text_dir):
        print(f"⚠️ No text folder found for class {cls}, skipping.")
        continue

    for fname in os.listdir(text_dir):
        if not fname.endswith(".txt"):
            continue

        # The base filename (without extension) must match the image filename
        base_name = os.path.splitext(fname)[0]

        # Read text
        with open(os.path.join(text_dir, fname), "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            # If text is empty, store a zero vector
            embedding = np.zeros(768, dtype=np.float32)
        else:
            # Tokenize and run through BERT
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                # Take the [CLS] token embedding (first token) as sentence representation
                cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                embedding = cls_embedding.astype(np.float32)

        # Save to ./data/feature/<base_name>.npy
        out_path = os.path.join(feature_dir, base_name + ".npy")
        np.save(out_path, embedding)

        print(f"✅ Saved feature: {out_path}")
