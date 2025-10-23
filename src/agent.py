from typing import List, Dict
from openai import OpenAI
from src import config
from src.embeddings.embedder import Embedder
from src.vector.pinecone_client import PineconeClient
from src.graph.neo4j_client import Neo4jClient
from src.prompt import build_prompt

class HybridRunner:
    """Orchestrates hybrid search: vector + graph + LLM."""
    
    def __init__(self, top_k: int = 5):
        self.embedder = Embedder()
        self.vector_store = PineconeClient()
        self.graph = Neo4jClient()
        self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.top_k = top_k
        self.chat_model = config.CHAT_MODEL or "gpt-4o-mini"
    
    def query(self, user_query: str) -> str:
        """Process a user query and return LLM response."""
        # 1. Embed query
        query_vec = self.embedder.embed(user_query)
        
        # 2. Vector search
        matches = self.vector_store.query(query_vec, top_k=self.top_k)
        
        # 3. Graph context
        node_ids = [m["id"] for m in matches]
        graph_facts = self.graph.search_by_topic_batch(node_ids, limit=10, max_workers=4)
        
        # 4. Build prompt
        prompt = build_prompt(user_query, matches, graph_facts)
        
        # 5. Call LLM
        resp = self.llm_client.chat.completions.create(
            model=self.chat_model,
            messages=prompt,
            max_tokens=600,
            temperature=0.2
        )
        return resp.choices[0].message.content
    
    def close(self):
        self.graph.close()