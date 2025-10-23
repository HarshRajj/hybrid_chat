# pinecone_upload_scaled.py
import json
import time
import os
import hashlib
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential
from src import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "D://hybrid_chat_test/data/vietnam_travel_dataset.json"
BATCH_SIZE = 32
MAX_WORKERS = 5   # number of parallel threads
INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 1536 for text-embedding-3-small
CACHE_FILE = "embeddings_cache.json"  # Local cache for embeddings

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(api_key=config.OPENAI_API_KEY)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east1")
    )
else:
    print(f"Index {INDEX_NAME} already exists.")

index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_embeddings(texts, model="text-embedding-3-small"):
    """Generate embeddings using OpenAI API with retries."""
    resp = client.embeddings.create(model=model, input=texts)
    return [data.embedding for data in resp.data]

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_batch(batch):
    """Upload one batch to Pinecone."""
    ids = [item[0] for item in batch]
    texts = [item[1] for item in batch]
    metas = [item[2] for item in batch]
    embeddings = get_embeddings(texts)
    vectors = [{"id": _id, "values": emb, "metadata": meta}
               for _id, emb, meta in zip(ids, embeddings, metas)]
    index.upsert(vectors)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_batch_from_vectors(batch):
    """Upload pre-computed vectors to Pinecone."""
    index.upsert(batch)

# -----------------------------
# Caching functions
# -----------------------------
def compute_file_hash(filepath):
    """Compute SHA256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def load_embeddings_cache():
    """Load embeddings from cache if valid."""
    if not os.path.exists(CACHE_FILE):
        return None
    
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        # Check if dataset has changed
        current_hash = compute_file_hash(DATA_FILE)
        if cache_data.get("dataset_hash") != current_hash:
            print("📄 Dataset changed, cache invalid.")
            return None
        
        print("✅ Loading embeddings from cache...")
        return cache_data["embeddings"]
    except (json.JSONDecodeError, KeyError):
        print("⚠️  Cache file corrupted, regenerating...")
        return None

def save_embeddings_cache(embeddings, items):
    """Save embeddings to cache."""
    cache_data = {
        "dataset_hash": compute_file_hash(DATA_FILE),
        "timestamp": time.time(),
        "embeddings": embeddings,
        "items": items  # Store the processed items too
    }
    
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Saved {len(embeddings)} embeddings to cache.")

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"📊 Processing {len(items)} items...")

    # Check cache first
    cached_embeddings = load_embeddings_cache()
    
    if cached_embeddings and len(cached_embeddings) == len(items):
        print("🎯 Using cached embeddings!")
        embeddings = cached_embeddings
    else:
        print("🔄 Generating new embeddings...")
        # Generate embeddings in batches
        embeddings = []
        texts = [item[1] for item in items]
        
        for batch_texts in tqdm(list(chunked(texts, BATCH_SIZE)), desc="Generating embeddings"):
            batch_embeddings = get_embeddings(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Save to cache
        save_embeddings_cache(embeddings, items)

    print(f"📤 Preparing to upsert {len(items)} items to Pinecone...")

    # Prepare vectors for upload
    vectors = []
    for (item_id, _, meta), embedding in zip(items, embeddings):
        vectors.append({
            "id": item_id,
            "values": embedding,
            "metadata": meta
        })

    # Upload in batches
    batches = list(chunked(vectors, BATCH_SIZE))
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(upload_batch_from_vectors, batch): batch for batch in batches}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Uploading batches"):
            pass

    print("✅ All items uploaded successfully.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def upload_batch_from_vectors(batch):
    """Upload pre-computed vectors to Pinecone."""
    index.upsert(batch)

# -----------------------------
if __name__ == "__main__":
    main()
