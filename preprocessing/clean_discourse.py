import json
import re
import os

# Paths
RAW_PATH = "data/raw/discourse_posts.json"
CLEANED_PATH = "data/cleaned/tds_discourse_cleaned.json"

# Ensure output directory exists
os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)

def clean_html(raw_html):
    """Remove HTML tags and normalize text."""
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr, '', raw_html)
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.strip()

# Load raw data
with open(RAW_PATH, "r", encoding="utf-8") as f:
    forum_data = json.load(f)

cleaned_posts = []
for post in forum_data:
    content = post.get("content", "").strip()
    if not content or len(content) < 10:
        continue  # Skip empty or trivial posts

    # Clean content
    content = clean_html(content)

    # Create cleaned post
    cleaned_post = {
        "source": "discourse",
        "author": post.get("author", "unknown"),
        "created_at": post.get("created_at"),
        "topic_title": post.get("topic_title", ""),
        "url": post.get("url"),
        "text": content
    }
    cleaned_posts.append(cleaned_post)

# Save cleaned data
with open(CLEANED_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned_posts, f, indent=2, ensure_ascii=False)

print(f"Cleaned forum posts saved: {len(cleaned_posts)} to {CLEANED_PATH}")