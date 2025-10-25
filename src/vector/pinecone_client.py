import time
import hashlib
import json
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential
from src import config

class PineconeClient:
    """Thin wrapper around Pinecone for vector operations with error handling and caching."""
    
    def __init__(self, index_name: str = None, dimension: int = None, enable_cache: bool = True):
        self.index_name = index_name or config.PINECONE_INDEX_NAME
        self.dimension = dimension or config.PINECONE_VECTOR_DIM
        self.enable_cache = enable_cache
        
        # Query result cache (in-memory)
        self.query_cache: Dict[str, List[Dict]] = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        
        try:
            self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
            print(f"✓ Connected to Pinecone")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Pinecone: {e}")
        
        self._ensure_index()
        
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"✓ Connected to index: {self.index_name}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to index '{self.index_name}': {e}")
    
    def _ensure_index(self):
        """Create index if it doesn't exist."""
        try:
            existing = self.pc.list_indexes().names()
            
            if self.index_name not in existing:
                print(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                # Wait for index to be ready
                print("⏳ Waiting for index to be ready...")
                time.sleep(10)
                print(f"✓ Index '{self.index_name}' created successfully")
            else:
                print(f"✓ Index '{self.index_name}' already exists")
                
        except Exception as e:
            raise RuntimeError(f"Failed to ensure index exists: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def upsert_batch(self, vectors: List[Dict[str, Any]]) -> Dict[str, int]:
        
        if not vectors:
            raise ValueError("Cannot upsert empty vector list")
        
        # Validate vector format
        for i, vec in enumerate(vectors):
            if 'id' not in vec or 'values' not in vec:
                raise ValueError(f"Vector at index {i} missing 'id' or 'values' field")
            if len(vec['values']) != self.dimension:
                raise ValueError(
                    f"Vector at index {i} has dimension {len(vec['values'])}, "
                    f"expected {self.dimension}"
                )
        
        try:
            result = self.index.upsert(vectors=vectors)
            return {
                "upserted_count": result.get("upserted_count", len(vectors)),
                "batch_size": len(vectors)
            }
        except Exception as e:
            raise RuntimeError(f"Failed to upsert batch of {len(vectors)} vectors: {e}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=5))
    def query(
        self, 
        vector: List[float], 
        top_k: int = 5, 
        include_metadata: bool = True,
        filter: Optional[Dict] = None
    ) -> List[Dict]:
        
        if len(vector) != self.dimension:
            raise ValueError(
                f"Query vector has dimension {len(vector)}, expected {self.dimension}"
            )
        
        # Generate cache key from vector + params
        if self.enable_cache:
            cache_key = self._get_cache_key(vector, top_k, filter)
            if cache_key in self.query_cache:
                self.cache_stats["hits"] += 1
                return self.query_cache[cache_key]
            self.cache_stats["misses"] += 1
        
        try:
            query_params = {
                "vector": vector,
                "top_k": top_k,
                "include_metadata": include_metadata,
                "include_values": False
            }
            
            if filter:
                query_params["filter"] = filter
            
            res = self.index.query(**query_params)
            matches = res.get("matches", [])
            
            # Ensure metadata exists for each match
            for match in matches:
                if "metadata" not in match:
                    match["metadata"] = {}
            
            # Cache the results
            if self.enable_cache:
                self.query_cache[cache_key] = matches
            
            return matches
            
        except Exception as e:
            raise RuntimeError(f"Failed to query index: {e}")
    
    def _get_cache_key(self, vector: List[float], top_k: int, filter: Optional[Dict]) -> str:
        """Generate a cache key from query parameters."""
        # Round vector values to reduce cache misses from floating point precision
        rounded_vector = [round(v, 6) for v in vector]
        key_data = {
            "vector": rounded_vector[:10],  # Use first 10 dims for key (representative)
            "top_k": top_k,
            "filter": filter
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        self.cache_stats = {"hits": 0, "misses": 0}
        print("✓ Pinecone query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        
        try:
            stats = self.index.describe_index_stats()
            
            # Add cache stats
            total_queries = self.cache_stats["hits"] + self.cache_stats["misses"]
            cache_hit_rate = (self.cache_stats["hits"] / total_queries * 100) if total_queries > 0 else 0
            
            return {
                "total_vector_count": stats.get("total_vector_count", 0),
                "dimension": stats.get("dimension", self.dimension),
                "index_fullness": stats.get("index_fullness", 0.0),
                "namespaces": stats.get("namespaces", {}),
                "cache": {
                    "enabled": self.enable_cache,
                    "size": len(self.query_cache),
                    "hits": self.cache_stats["hits"],
                    "misses": self.cache_stats["misses"],
                    "hit_rate": f"{cache_hit_rate:.1f}%"
                }
            }
        except Exception as e:
            print(f"⚠ Warning: Failed to get index stats: {e}")
            return {"error": str(e)}
    
    # def delete_by_ids(self, ids: List[str]) -> Dict[str, int]:
    #     """
    #     Delete vectors by their IDs.
        
    #     Args:
    #         ids: List of vector IDs to delete
            
    #     Returns:
    #         Dict with deletion count
    #     """
    #     if not ids:
    #         raise ValueError("Cannot delete with empty ID list")
        
    #     try:
    #         self.index.delete(ids=ids)
    #         return {"deleted_count": len(ids)}
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to delete {len(ids)} vectors: {e}")
    
    # def delete_all(self):
    #     """Delete all vectors (use with caution)."""
    #     try:
    #         print("⚠ WARNING: Deleting all vectors from index...")
    #         self.index.delete(delete_all=True)
    #         print("✓ All vectors deleted")
    #     except Exception as e:
    #         raise RuntimeError(f"Failed to delete all vectors: {e}")
    
    def health_check(self) -> bool:
        """
        Check if the client and index are healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            stats = self.get_stats()
            return "error" not in stats
        except Exception:
            return False