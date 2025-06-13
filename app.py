from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import json
import numpy as np
import faiss
import aiohttp
import asyncio
import logging
import traceback
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import re
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
INDEX_PATH = "data/faiss/tds_faiss.index"
META_PATH = "data/faiss/tds_faiss_meta.json"
MODEL_NAME = "all-MiniLM-L6-v2"

# Initialize FastAPI app
app = FastAPI(title="TDS Virtual Teaching Assistant", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
index = None
metadata = None
discourse_indices = None
discourse_index = None

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    image: Optional[str] = None
    link: Optional[str] = None

class LinkResponse(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkResponse]

@app.on_event("startup")
async def startup_event():
    global model, index, metadata, discourse_indices, discourse_index
    
    try:
        if not AIPROXY_TOKEN:
            logger.error("AIPROXY_TOKEN environment variable is not set")
            raise ValueError("AIPROXY_TOKEN environment variable is required")
        
        logger.info("Loading embedding model...")
        model = SentenceTransformer(MODEL_NAME)
        
        logger.info("Loading FAISS index...")
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}")
        index = faiss.read_index(INDEX_PATH)
        
        logger.info("Loading metadata...")
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(f"Metadata file not found at {META_PATH}")
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Identify Discourse chunks
        discourse_indices = []
        for i, chunk in enumerate(metadata):
            url = chunk.get("url", "")
            if "discourse.onlinedegree.iitm.ac.in" in url:
                discourse_indices.append(i)
        
        logger.info(f"Loaded {len(metadata)} chunks")
        logger.info(f" - Discourse posts: {len(discourse_indices)}")
        
        # Create Discourse index if needed
        if discourse_indices:
            logger.info("Creating Discourse index...")
            d = index.d
            discourse_index = faiss.IndexFlatIP(d)
            discourse_vectors = np.empty((len(discourse_indices), d), dtype=np.float32)
            for i, idx in enumerate(discourse_indices):
                discourse_vectors[i] = index.reconstruct(idx)
            discourse_index.add(discourse_vectors)
            logger.info(f"Created Discourse index with {len(discourse_indices)} vectors")
        
        logger.info("Startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        logger.error(traceback.format_exc())
        raise

def normalize_url(url):
    """Normalize URL for consistent comparison"""
    if not url:
        return ""
    # Remove query parameters and trailing slash
    url = re.sub(r'\?.*$', '', url).rstrip('/')
    return url.lower()

def get_base_topic_url(url):
    """Convert Discourse post URL to base topic URL"""
    if "discourse.onlinedegree.iitm.ac.in" in url:
        match = re.search(r'(https://discourse\.onlinedegree\.iitm\.ac\.in/t/[^/]+/\d+)', url)
        if match:
            return match.group(1)
    return url

def find_matching_chunk_by_topic(required_link):
    """Find chunk matching the required topic"""
    if not required_link or not metadata:
        return None
    
    # Get base topic URL for matching
    base_topic = get_base_topic_url(required_link)
    if not base_topic:
        return None
    
    # Find the first chunk with matching base URL
    for chunk in metadata:
        chunk_url = chunk.get("url", "")
        if base_topic in get_base_topic_url(chunk_url):
            return chunk
    
    return None

def embed_query(query):
    """Embed a query using the sentence transformer model."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        vec = model.encode([query])
        return np.array(vec).astype("float32")
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding error: {str(e)}")
    
import re

def extract_topic_id(url):
    """Extracts topic ID from a Discourse link (e.g., 165959 from .../165959/22)"""
    match = re.search(r'/t/[^/]+/(\d+)', url)
    return match.group(1) if match else None

def retrieve_top_k(query, k=10, required_link=None):
    """Retrieve top-k relevant chunks, optionally force-including a chunk by link or topic ID"""
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

        # Force include required chunk based on exact link or topic ID
        if required_link:
            topic_id = extract_topic_id(required_link)

            for chunk in metadata:
                url = chunk.get("url", "")
                faiss_id = chunk.get("faiss_id")

                if faiss_id in seen_ids:
                    continue

                if required_link in url or (topic_id and topic_id in url):
                    results.insert(0, chunk)
                    seen_ids.add(faiss_id)
                    break  # Only insert the first matching chunk

        return [{
            "url": c.get("url", ""),
            "text": c.get("text", ""),
            "source": c.get("source", ""),
            "filename": c.get("filename", "")
        } for c in results]

    except Exception as e:
        print(f"Error in retrieval: {e}")
        return []

def clean_link_text(text):
    """Clean text for link display"""
    if not text:
        return ""
    
    # Remove markdown and special characters
    text = re.sub(r'``````', '', text, flags=re.DOTALL)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[^\w\s.,?!-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Extract first meaningful sentence
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        if len(sentence) > 20:
            return (sentence[:120] + '...') if len(sentence) > 120 else sentence
    
    return (text[:120] + '...') if len(text) > 120 else text

def build_context(chunks):
    """Build context string from chunks"""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        text = chunk.get('text', '').strip()
        if text:
            url = chunk.get('url', '')
            source_info = f" (Source: {url})" if url else ""
            context_parts.append(f"[{i}] {text}{source_info}")
    
    return "\n\n".join(context_parts)

async def get_llm_answer(query, context, model_name="gpt-4o-mini", max_retries=3):
    """Get answer from LLM with retry logic"""
    if not AIPROXY_TOKEN:
        raise HTTPException(status_code=500, detail="AIPROXY_TOKEN not configured")
    
    retries = 0
    while retries < max_retries:
        try:
            logger.info(f"Generating answer for query: '{query[:50]}...' (attempt {retries+1})")
            
            system_prompt = (
                "You are a helpful teaching assistant for the TDS (Tools in Data Science) course. "
                "Use the provided context to answer the user's question accurately and helpfully. "
                "If the answer is not in the context, say so clearly. "
                "Always include sources in your response with exact URLs when available."
            )
            
            headers = {
                "Authorization": f"Bearer {AIPROXY_TOKEN}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                "temperature": 0.2,
                "max_tokens": 512
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(AIPROXY_URL, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully received answer from LLM")
                        return result["choices"][0]["message"]["content"].strip()
                    elif response.status == 429:  # Rate limit error
                        error_text = await response.text()
                        logger.warning(f"Rate limit reached, retrying after delay (retry {retries+1}): {error_text}")
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        error_text = await response.text()
                        error_msg = f"Error generating answer (status {response.status}): {error_text}"
                        logger.error(error_msg)
                        if retries >= max_retries - 1:
                            raise HTTPException(status_code=response.status, detail=error_msg)
                        retries += 1
                        await asyncio.sleep(2)
                        
        except asyncio.TimeoutError:
            logger.error(f"LLM API timeout (attempt {retries+1}/{max_retries})")
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail="LLM API timeout")
            await asyncio.sleep(3 * retries)
        except Exception as e:
            error_msg = f"Exception generating answer (attempt {retries+1}/{max_retries}): {e}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            retries += 1
            if retries >= max_retries:
                raise HTTPException(status_code=500, detail=error_msg)
            await asyncio.sleep(2)

@app.post("/answer_question")
async def answer_question(request: QuestionRequest):
    try:
        # Embed the question
        query_vec = embed_query(request.question)
        D, I = index.search(query_vec, 10)

        # Collect top chunks
        top_chunks = []
        seen_ids = set()
        for idx in I[0]:
            if idx < len(metadata):
                chunk = metadata[idx]
                top_chunks.append(chunk)
                seen_ids.add(chunk.get("faiss_id"))

        # Force include required link if provided
        if request.link:
            for chunk in metadata:
                if request.link in chunk.get("url", "") and chunk.get("faiss_id") not in seen_ids:
                    top_chunks.insert(0, chunk)
                    break

        # Build context from chunks
        context_parts = []
        links = []
        for i, chunk in enumerate(top_chunks, 1):
            text = chunk.get("text", "")
            original_url = chunk.get("url", "")
            if text:
                context_parts.append(f"[{i}] {text}")
                links.append({
                    "url": original_url,
                    "text": text[:100] + "..." if len(text) > 100 else text
                })

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = get_llm_answer(request.question, context)

        return {"answer": answer, "links": links}

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    


@app.get("/health")
async def health_check():
    try:
        health_status = {
            "status": "healthy",
            "model_loaded": model is not None,
            "index_loaded": index is not None,
            "discourse_index_loaded": discourse_index is not None,
            "chunks_count": len(metadata) if metadata else 0,
            "discourse_posts": len(discourse_indices) if discourse_indices else 0,
            "aiproxy_configured": bool(AIPROXY_TOKEN)
        }
        
        # Test FAISS index if loaded
        if index is not None:
            try:
                # Test with a simple query
                test_vec = np.random.random((1, index.d)).astype('float32')
                index.search(test_vec, 1)
                health_status["index_functional"] = True
            except Exception as e:
                health_status["index_functional"] = False
                health_status["index_error"] = str(e)
        
        logger.info("Health check completed successfully")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy", 
                "error": str(e),
                "aiproxy_configured": bool(AIPROXY_TOKEN)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
