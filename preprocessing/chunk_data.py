import json
import os
import math

CHUNK_SIZE = 200  # words per chunk

def chunk_text(text, chunk_size=CHUNK_SIZE):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def process_file(input_path, output_path, source_type):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunked_data = []
    for idx, item in enumerate(data):
        # For discourse, include author and created_at metadata
        base_chunk = {
            "source": item.get("source", source_type),
            "filename": item.get("filename"),
            "url": item.get("url"),
        }
        if source_type == "discourse":
            base_chunk["author"] = item.get("author")
            base_chunk["created_at"] = item.get("created_at")
            base_chunk["topic_title"] = item.get("topic_title")

        text = item.get("text", "")
        chunks = chunk_text(text)
        for chunk_id, chunk_text_str in enumerate(chunks):
            chunk_entry = base_chunk.copy()
            chunk_entry["chunk_id"] = chunk_id
            chunk_entry["text"] = chunk_text_str
            chunked_data.append(chunk_entry)

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunked_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Process course data
    process_file(
        input_path="data/cleaned/tds_course_cleaned.json",
        output_path="data/chunk_data/tds_course_chunks.json",
        source_type="course"
    )
    # Process discourse data
    process_file(
        input_path="data/cleaned/tds_discourse_cleaned.json",
        output_path="data/chunk_data/discourse_chunks.json",
        source_type="discourse"
    )
