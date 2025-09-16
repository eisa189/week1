## Usage

```python
python training.py
```

# Multi-modal Deep Learning for Disaster Detection

This repository contains the Week 1 deliverable for the course project on incident detection in social-media images, based on the paper:
The dataset is about (1 Gb) in size and could not be uploaded to GitHub.
Paper: **"Multi-modal deep learning framework for damage detection in social media posts"**  
PeerJ Computer Science, 2024  
DOI: [10.7717/peerj-cs.2262](https://doi.org/10.7717/peerj-cs.2262)

---

Before running you must download the data set and also install these dependencies

torch, torchvision, timm, einops
numpy, pandas, scikit-learn
pillow, tqdm
transformers

Es necesario ejecutar este código antes del entrenamiento.

python generate_text_features.py   # generates .npy embeddings from text
python generate_csv.py             # builds all_data.csv
python fix_missing_features.py     # fills in any missing .npy with zeros



