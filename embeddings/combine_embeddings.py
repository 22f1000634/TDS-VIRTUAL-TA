import json
import glob
import os

# List your embedding files here
embedding_files = [
    "data/embedding_data/tds_course_embeddings.json",
    "data/embedding_data/discourse_embeddings.json"
]

combined_embeddings = []

for file_path in embedding_files:
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # If the file contains a list, extend; if dict, handle accordingly
        if isinstance(data, list):
            combined_embeddings.extend(data)
        else:
            print(f"Warning: {file_path} does not contain a list.")

# Save the combined list to a new file
output_path = "data/embeddings/combined_embeddings.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_embeddings, f, ensure_ascii=False, indent=2)

print(f"Combined {len(combined_embeddings)} embeddings into {output_path}")
