import os
import json
from sentence_transformers import SentenceTransformer

# Choose your model from the MTEB leaderboard
MODEL_NAME = "all-MiniLM-L6-v2"  # or "bge-large-en-v1.5", "gte-large", etc.

model = SentenceTransformer(MODEL_NAME)

def embed_chunks(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedded_chunks = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        if not text.strip():
            print(f"Skipped empty chunk {i+1}")
            continue
        embedding = model.encode(text)
        chunk_with_embedding = dict(chunk)
        chunk_with_embedding["embedding"] = embedding.tolist()
        embedded_chunks.append(chunk_with_embedding)
        print(f"Embedded chunk {i+1}/{len(chunks)}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    embed_chunks(
        "data/chunk_data/tds_course_chunks.json",
        "data/embedding_data/tds_course_embeddings.json"
    )
    embed_chunks(
        "data/chunk_data/discourse_chunks.json",
        "data/embedding_data/discourse_embeddings.json"
    )
    print("Embeddings generated and saved successfully.")