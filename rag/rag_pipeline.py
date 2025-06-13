import os
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# FAISS and model configuration
INDEX_PATH = "data/faiss/tds_faiss.index"
META_PATH = "data/faiss/tds_faiss_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Load model, FAISS index, and metadata
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

def embed_query(query):
    """Convert query to embedding vector"""
    vec = model.encode([query])
    return np.array(vec).astype("float32")

def retrieve_top_k(query, k=10, required_link=None):
    """Retrieve top-k most relevant chunks for the query, optionally forcing inclusion of a required link"""
    try:
        query_vec = embed_query(query)
        D, I = index.search(query_vec, k)
        results = []
        seen_ids = set()

        for idx in I[0]:
            if idx < len(metadata):
                chunk = metadata[idx]
                results.append(chunk)
                seen_ids.add(chunk.get("faiss_id"))

        # Force include a required chunk by link
        if required_link:
            for chunk in metadata:
                if required_link in chunk.get("url", "") and chunk.get("faiss_id") not in seen_ids:
                    results.insert(0, chunk)
                    break

        return [{
            "url": c.get("url", ""),
            "text": c.get("text", ""),
            "source": c.get("source", ""),
            "filename": c.get("filename", "")
        } for c in results]
    except Exception as e:
        print(f"Error in retrieval: {e}")
        return []

def build_context(chunks):
    """Build context string from retrieved chunks"""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        if chunk["text"]:
            context_parts.append(f"[{i}] {chunk['text']}")
    return "\n\n".join(context_parts)

def get_llm_answer(query, context, model="openai/gpt-4o-mini", temperature=0.0):
    """Generate answer using LLM via AI Proxy"""
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    system_prompt = (
        "You are a helpful teaching assistant for the TDS (Tools in Data Science) course. "
        "Use the provided context to answer the user's question accurately and helpfully. "
        "If the answer is not in the context, say so clearly."
    )
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        "temperature": temperature,
        "max_tokens": 512
    }

    try:
        response = requests.post(AIPROXY_URL, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return None

def main():
    """Main RAG pipeline function"""
    print("TDS Virtual Teaching Assistant")
    print("=" * 40)

    while True:
        user_query = input("\nEnter your question (or 'quit' to exit): ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break

        link = input("Optional: Enter discourse link (if any): ").strip() or None

        print("\nRetrieving relevant information...")
        top_chunks = retrieve_top_k(user_query, k=10, required_link=link)

        if not top_chunks:
            print("No relevant information found.")
            continue

        context = build_context(top_chunks)

        print("Generating answer...")
        answer = get_llm_answer(user_query, context)

        if answer:
            print(f"\n{'='*50}")
            print("ANSWER:")
            print(f"{'='*50}")
            print(answer)

            print(f"\n{'='*50}")
            print("SOURCES:")
            print(f"{'='*50}")
            for i, chunk in enumerate(top_chunks, 1):
                url = chunk.get("url", "")
                source = chunk.get("source", "")
                if url:
                    print(f"[{i}] {url} ({source})")
        else:
            print("Sorry, I couldn't generate an answer at this time.")

if __name__ == "__main__":
    main()
