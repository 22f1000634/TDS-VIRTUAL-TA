import json
import re
import os

# Paths
RAW_PATH = "data/raw/course_scraped.json"
CLEANED_PATH = "data/cleaned/tds_course_cleaned.json"

# Ensure output directory exists
os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)

# Load raw data
with open(RAW_PATH, "r", encoding="utf-8") as f:
    course_data = json.load(f)

cleaned_chunks = []
for filename, item in course_data.items():
    url = item.get("url")
    content = item.get("content", "").strip()
    if not content or len(content) < 10:
        continue  # Skip empty or trivial content

    # Remove Markdown artifacts
    content = re.sub(r'!\[.*?\]\(.*?\)', '', content)  # Remove images
    content = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', content)  # Remove links, keep text
    content = re.sub(r'^#+\s*', '', content, flags=re.MULTILINE)  # Remove headers
    content = re.sub(r'^\s*[-*+]\s+', '', content, flags=re.MULTILINE)  # Remove bullet prefixes
    content = re.sub(r'\n\s*\n', '\n\n', content)  # Normalize blank lines
    content = content.strip()

    # Split into paragraphs (by double newline or bullet points)
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if len(p.strip()) > 20]

    # Create cleaned chunks
    for para in paragraphs:
        chunk = {
            "source": "course",
            "filename": filename,
            "url": url,
            "text": para
        }
        cleaned_chunks.append(chunk)

# Save cleaned data
with open(CLEANED_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned_chunks, f, indent=2, ensure_ascii=False)

print(f"Cleaned course chunks saved: {len(cleaned_chunks)} to {CLEANED_PATH}")