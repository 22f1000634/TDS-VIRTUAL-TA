import json
import numpy as np
import faiss
import os

# Path to the combined embeddings file
input_path = "data/embeddings/combined_embeddings.json"
output_index_path = "data/faiss/tds_faiss.index"
output_meta_path = "data/faiss/tds_faiss_meta.json"

# Load combined embeddings
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract embeddings and metadata
embeddings = []
metadata = []
for i, item in enumerate(data):
    emb = item["embedding"]
    embeddings.append(emb)
    # Store all other fields as metadata (except embedding)
    meta = {k: v for k, v in item.items() if k != "embedding"}
    meta["faiss_id"] = i  # Keep track of FAISS index
    metadata.append(meta)

embeddings = np.array(embeddings).astype("float32")
d = embeddings.shape[1]  # dimension

# Create FAISS index (L2 distance)
index = faiss.IndexFlatL2(d)
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} vectors of dimension {d}")

# Ensure output directory exists
os.makedirs(os.path.dirname(output_index_path), exist_ok=True)

# Save FAISS index
faiss.write_index(index, output_index_path)
print(f"Saved FAISS index to {output_index_path}")

# Save metadata for lookup
with open(output_meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)
print(f"Saved FAISS metadata to {output_meta_path}")
