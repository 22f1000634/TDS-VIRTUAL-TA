import requests
import re
import json
from urllib.parse import urljoin, urlparse

BASE_URL = "https://tds.s-anand.net/#/2025-01/"
README_URL = urljoin(BASE_URL, "README.md")
SIDEBAR_URL = urljoin(BASE_URL, "_sidebar.md")

def fetch_markdown(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        print(f"[+] Fetched: {url}")
        return response.text
    except Exception as e:
        print(f"[!] Failed to fetch {url}: {e}")
        return ""

def extract_md_links(markdown_text):
    return re.findall(r"\[.*?\]\((.*?\.md)\)", markdown_text)

def scrape_docsify():
    readme_text = fetch_markdown(README_URL)
    sidebar_text = fetch_markdown(SIDEBAR_URL)
    links = set(extract_md_links(readme_text) + extract_md_links(sidebar_text))
    data = {}
    for link in links:
        raw_url = urljoin(BASE_URL, link)
        # Always use /../ before the filename in the hash URL
        filename = link.rsplit('/', 1)[-1]
        if filename.endswith('.md'):
            filename = filename[:-3]
        hash_url = f"https://tds.s-anand.net/#/{filename}"
        content = fetch_markdown(raw_url)
        data[link] = {
            "url": hash_url,
            "content": content
        }
    with open("data/raw/course_scraped.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("\n[+] Full content saved to 'data/raw/course_scraped.json'.")

if __name__ == "__main__":
    scrape_docsify()