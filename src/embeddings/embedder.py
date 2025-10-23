import os
import json
import hashlib
from typing import List, Dict
from openai import OpenAI
from src import config

class Embedder:
    """Handles text embeddings with simple file-based caching."""
    
    def __init__(self, model: str = None, cache_path: str = "data/embeddings_cache.json"):
        self.model = model or config.EMBED_MODEL or "text-embedding-3-small"
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        self._cache = self._load_cache()
    
    def _load_cache(self) -> Dict[str, List[float]]:
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def _save_cache(self):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f)
        except Exception:
            pass
    
    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model}:{text}".encode("utf-8")).hexdigest()
    
    def embed(self, text: str) -> List[float]:
        """Get embedding for a single text (uses cache)."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]
        
        resp = self.client.embeddings.create(model=self.model, input=[text])
        emb = resp.data[0].embedding
        self._cache[key] = emb
        self._save_cache()
        return emb
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts (batched API call)."""
        # Check cache first
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if key in self._cache:
                results.append(self._cache[key])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Fetch uncached embeddings
        if uncached_texts:
            resp = self.client.embeddings.create(model=self.model, input=uncached_texts)
            for idx, data in zip(uncached_indices, resp.data):
                emb = data.embedding
                results[idx] = emb
                key = self._cache_key(texts[idx])
                self._cache[key] = emb
            self._save_cache()
        
        return results