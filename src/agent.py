from typing import Dict, List, Optional
import time
import hashlib
import asyncio
from openai import AsyncOpenAI
from src import config
from src.embeddings.embedder import Embedder
from src.vector.pinecone_client import PineconeClient
from src.graph.neo4j_client import Neo4jClient
from src.router import HybridRouter
from src.prompt import build_prompt, build_direct_prompt, build_understanding_prompt


class HybridRunner:
    """
    Async query orchestrator with intelligent routing and multi-level caching.
    Routes queries to hybrid/graph/direct strategies with parallel vector+graph retrieval.
    """
    
    def __init__(self, top_k: int = 5, enable_streaming: bool = False):
        # Core components
        self.embedder = Embedder()
        self.vector_store = PineconeClient()
        self.graph = Neo4jClient()
        self.router = HybridRouter()
        self.llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        # Config
        self.top_k = top_k
        self.chat_model = config.CHAT_MODEL or "gpt-4o-mini"
        self.enable_streaming = enable_streaming
        
        # State - unified cache for simplicity
        self.conversation_history: List[Dict] = []
        self._cache: Dict[str, any] = {
            "understanding": {},  # Phase 1 understanding cache
            "responses": {},      # LLM response cache
            "timings": []         # Response time metrics
        }
    
    async def query(self, user_query: str, explain: bool = False) -> str:
        """
        Process user query asynchronously.
        Phase 1: Quick checks (greetings, out-of-scope)
        Phase 2: Route and execute strategy (hybrid/graph/direct)
        """
        # Start timing
        start_time = time.time()
        timings = {
            "query": user_query,
            "phase1": 0,
            "routing": 0,
            "embedding": 0,
            "vector_search": 0,
            "graph_search": 0,
            "llm_generation": 0,
            "total": 0,
            "strategy": "",
            "used_cache": False
        }
        
        # Add user query to history
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # PHASE 1: Quick checks for greetings and out-of-scope
        phase1_start = time.time()
        # Clean the query - remove quotes, extra spaces, punctuation
        query_lower = user_query.strip().strip("'\".,!?/\\").lower().strip()
        
        # Handle greetings instantly (no database needed!)
        greetings = ["hi", "hii", "hello", "hey", "good morning", "good afternoon", "good evening", "hola", "sup", "yo", "he", "helo", "hela", "heyy", "hey there"]
        if query_lower in greetings or len(query_lower) <= 3:  # Very short queries are likely greetings
            response = "Hello! I'm here to help you plan your Vietnam trip. What would you like to know about traveling in Vietnam?"
            self.conversation_history.append({"role": "assistant", "content": response})
            timings["phase1"] = time.time() - phase1_start
            timings["total"] = time.time() - start_time
            timings["strategy"] = "greeting"
            self._cache["timings"].append(timings)
            return response
        
        # Check for obvious out-of-scope keywords
        out_of_scope_keywords = ["london", "paris", "thailand", "japan", "korea", "singapore", "malaysia", "bali", "china", "europe"]
        if any(kw in query_lower for kw in out_of_scope_keywords):
            # Use LLM to confirm it's out of scope
            understanding = await self._understand_query(user_query, explain=explain)
            if understanding["is_out_of_scope"]:
                response = self._handle_out_of_scope(user_query)
                self.conversation_history.append({"role": "assistant", "content": response})
                timings["phase1"] = time.time() - phase1_start
                timings["total"] = time.time() - start_time
                timings["strategy"] = "out_of_scope"
                self._cache["timings"].append(timings)
                if explain:
                    print("🚫 Out of scope detected")
                return response
        
        timings["phase1"] = time.time() - phase1_start
        
        # PHASE 2: Route and execute query
        if explain:
            print(f"✅ Phase 1 passed ({timings['phase1']:.3f}s) → Retrieving data...")
        
        # Route query
        routing_start = time.time()
        route = self.router.route(user_query)
        strategy = route["strategy"]
        filters = route.get("filters", {})
        timings["routing"] = time.time() - routing_start
        timings["strategy"] = strategy
        
        if explain:
            print(f"🎯 Strategy: {strategy} (confidence: {route['confidence']:.2f}) - {route['reasoning']}")
            if filters:
                print(f"   Filters: {filters}")
        
        # Execute strategy asynchronously
        if strategy == "direct":
            response = await self._direct_llm(user_query, timings, explain=explain)
        elif strategy == "graph":
            response = await self._graph_focused(user_query, filters, timings, explain=explain)
        else:  # hybrid or vector (default)
            response = await self._hybrid_search(user_query, filters, timings, explain=explain)
        
        # Record total time
        timings["total"] = time.time() - start_time
        self._cache["timings"].append(timings)
        
        if explain:
            self._print_timing_breakdown(timings)
        
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    async def _understand_query(self, query: str, explain: bool = False) -> Dict:
        """Check if query is out-of-scope using LLM with conversation context."""
        # Check cache first
        if query in self._cache["understanding"]:
            if explain:
                print("💾 Using cached understanding")
            return self._cache["understanding"][query]
        
        # Get recent conversation history for context
        recent_history = self.conversation_history[-4:] if len(self.conversation_history) > 0 else None
        messages = build_understanding_prompt(query, history=recent_history)

        try:
            resp = await self.llm_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.2,
                max_tokens=200,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(resp.choices[0].message.content)
            
            # Cache the result
            self._cache["understanding"][query] = result
            
            if explain:
                print(f"📊 Understanding: {result.get('intent', 'analyzing...')}")
            
            return result
            
        except Exception as e:
            if explain:
                print(f"⚠️ Understanding phase error: {e}")
            return {
                "is_out_of_scope": False,
                "needs_clarification": False,
                "intent": "proceeding with query",
                "confidence": 0.5
            }
    
    def _get_max_tokens(self, query: str, strategy: str = "hybrid") -> int:
        """Get appropriate token limit based on strategy."""
        if strategy == "direct":
            return 400  # Direct LLM responses
        elif strategy == "graph":
            return 500  # Graph-focused responses
        else:
            return 600  # Hybrid responses (default)
    
    def _generate_cache_key(self, query: str, matches: List, graph_facts: List) -> str:
        """Generate cache key from query + retrieval results."""
        match_ids = [m.get('id', '') for m in matches[:3]]
        cache_str = f"{query.lower().strip()}|{'-'.join(match_ids)}|{len(graph_facts)}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _check_response_cache(self, cache_key: str, timings: Dict, explain: bool) -> Optional[str]:
        """Check if response is cached. Returns cached response or None."""
        if cache_key in self._cache["responses"]:
            if explain:
                print("💾 Cache hit!")
            timings["llm_generation"] = 0.001
            timings["used_cache"] = True
            return self._cache["responses"][cache_key]
        return None
    
    async def _hybrid_search(self, query: str, filters: Dict, timings: Dict, explain: bool = False) -> str:
        """Vector + graph search with async parallel execution."""
        
        # 1. Embed query
        embed_start = time.time()
        query_vec = self.embedder.embed(query)
        timings["embedding"] = time.time() - embed_start
        
        # 2. Fetch vector and graph data in parallel
        async def fetch_vector():
            start = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.vector_store.query, query_vec, self.top_k)
            return result, time.time() - start
        
        async def fetch_graph(node_ids):
            start = time.time()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.graph.search_by_topic_batch, node_ids, 10, 4)
            return result, time.time() - start
        
        # Execute in parallel
        matches, vector_time = await fetch_vector()
        timings["vector_search"] = vector_time
        
        node_ids = [m["id"] for m in matches]
        graph_facts, graph_time = await fetch_graph(node_ids)
        timings["graph_search"] = graph_time
        
        if explain:
            print(f"📊 Retrieved {len(matches)} vectors + {len(graph_facts)} graph facts")
        
        # 3. Check cache and generate response if needed
        cache_key = self._generate_cache_key(query, matches, graph_facts)
        if cached := self._check_response_cache(cache_key, timings, explain):
            return cached
        
        # 4. Build prompt and generate response
        llm_start = time.time()
        history_context = self.conversation_history[-4:] if len(self.conversation_history) > 1 else []
        prompt = build_prompt(query, matches, graph_facts, filters, history=history_context)
        max_tokens = self._get_max_tokens(query, "hybrid")
        
        response = await self._generate_llm_response(prompt, max_tokens, explain)
        
        timings["llm_generation"] = time.time() - llm_start
        self._cache["responses"][cache_key] = response
        
        return response
    
    async def _generate_llm_response(self, prompt: List[Dict], max_tokens: int, explain: bool = False) -> str:
        """Common LLM generation logic with streaming support."""
        if self.enable_streaming:
            return await self._stream_llm_response(prompt, max_tokens, explain)
        else:
            resp = await self.llm_client.chat.completions.create(
                model=self.chat_model,
                messages=prompt,
                max_tokens=max_tokens,
                temperature=0.2
            )
            return resp.choices[0].message.content
    
    def _handle_out_of_scope(self, query: str) -> str:
        """Handle queries outside Vietnam travel domain."""
        return (
            "I apologize, but I'm specifically designed for Vietnam travel queries. "
            "I can help with destinations, hotels, restaurants, attractions, itineraries, and general Vietnam travel info. "
            "Would you like to ask about traveling in Vietnam instead?"
        )
    
    async def _graph_focused(self, query: str, filters: Dict, timings: Dict, explain: bool = False) -> str:
        """Graph-focused search for 'nearby' queries."""
        # Get seed nodes via vector search
        embed_start = time.time()
        query_vec = self.embedder.embed(query)
        timings["embedding"] = time.time() - embed_start
        
        loop = asyncio.get_event_loop()
        vector_start = time.time()
        matches = await loop.run_in_executor(None, self.vector_store.query, query_vec, 3)
        timings["vector_search"] = time.time() - vector_start
        
        # Expand via graph (more neighbors for graph queries)
        graph_start = time.time()
        node_ids = [m["id"] for m in matches]
        graph_facts = await loop.run_in_executor(None, self.graph.search_by_topic_batch, node_ids, 20, 4)
        timings["graph_search"] = time.time() - graph_start
        
        if explain:
            print(f"🕸️  Graph focus: {len(matches)} seeds → {len(graph_facts)} relationships")
        
        # Check cache and generate response if needed
        cache_key = self._generate_cache_key(query, matches, graph_facts)
        if cached := self._check_response_cache(cache_key, timings, explain):
            return cached
        
        # Generate response
        llm_start = time.time()
        history_context = self.conversation_history[-4:] if len(self.conversation_history) > 1 else []
        prompt = build_prompt(query, matches, graph_facts, filters, history=history_context)
        max_tokens = self._get_max_tokens(query, "graph")
        
        response = await self._generate_llm_response(prompt, max_tokens, explain)
        
        timings["llm_generation"] = time.time() - llm_start
        self._cache["responses"][cache_key] = response
        
        return response
    
    async def _direct_llm(self, query: str, timings: Dict, explain: bool = False) -> str:
        """Direct LLM response without retrieval."""
        llm_start = time.time()
        history_context = self.conversation_history[-4:] if len(self.conversation_history) > 1 else []
        prompt = build_direct_prompt(query, history=history_context)
        max_tokens = self._get_max_tokens(query, "direct")
        
        response = await self._generate_llm_response(prompt, max_tokens, explain)
        
        timings["llm_generation"] = time.time() - llm_start
        return response
    
    async def _stream_llm_response(self, prompt: List[Dict], max_tokens: int, explain: bool = False) -> str:
        """Stream LLM response in real-time with async operations."""
        if explain:
            print("🤖 Assistant (streaming): ", end='', flush=True)
        else:
            print("\n" + "="*60)
            print("🤖 Assistant:")
            print("="*60)
        
        stream = await self.llm_client.chat.completions.create(
            model=self.chat_model,
            messages=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            stream=True
        )
        
        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                print(content, end='', flush=True)
                full_response += content
        
        print()  # New line after response
        if not explain:
            print("="*60)
        
        return full_response
    
    def _print_timing_breakdown(self, timings: Dict):
        """Print formatted timing breakdown."""
        print(f"\n⏱️  Timing: Phase1={timings['phase1']:.3f}s | Routing={timings['routing']:.3f}s", end='')
        if timings['embedding'] > 0:
            print(f" | Embed={timings['embedding']:.3f}s", end='')
        if timings['vector_search'] > 0:
            print(f" | Vector={timings['vector_search']:.3f}s", end='')
        if timings['graph_search'] > 0:
            print(f" | Graph={timings['graph_search']:.3f}s", end='')
        print(f" | LLM={timings['llm_generation']:.3f}s | Total={timings['total']:.3f}s", end='')
        if timings.get('used_cache'):
            print(" 💾", end='')
        print()
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    @property
    def stats(self) -> Dict:
        """Get comprehensive statistics (timing + cache)."""
        timings = self._cache["timings"]
        if not timings:
            return {"total_queries": 0, "cache": self.get_cache_stats()}
        
        num_queries = len(timings)
        return {
            "total_queries": num_queries,
            "avg_total": sum(t["total"] for t in timings) / num_queries,
            "avg_phase1": sum(t["phase1"] for t in timings) / num_queries,
            "avg_embedding": sum(t["embedding"] for t in timings) / num_queries,
            "avg_vector": sum(t["vector_search"] for t in timings) / num_queries,
            "avg_graph": sum(t["graph_search"] for t in timings) / num_queries,
            "avg_llm": sum(t["llm_generation"] for t in timings) / num_queries,
            "recent_queries": timings[-10:],
            "cache": self.get_cache_stats()
        }
    
    def get_timing_stats(self) -> Dict:
        """Get timing statistics (for backward compatibility)."""
        return self.stats
    
    def print_timing_summary(self):
        """Print formatted summary of timing statistics."""
        stats = self.stats
        if stats["total_queries"] == 0:
            print("No queries processed yet.")
            return
        
        print("\n" + "="*60)
        print(f"⏱️  SUMMARY: {stats['total_queries']} queries")
        print("="*60)
        print(f"Avg Total:  {stats['avg_total']:.3f}s")
        print(f"Avg Phase1: {stats['avg_phase1']:.3f}s | Embed: {stats['avg_embedding']:.3f}s")
        print(f"Avg Vector: {stats['avg_vector']:.3f}s | Graph: {stats['avg_graph']:.3f}s")
        print(f"Avg LLM:    {stats['avg_llm']:.3f}s")
        print("="*60)
    
    def clear_response_cache(self):
        """Clear LLM response cache."""
        self._cache["responses"].clear()
        print("✓ Response cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "responses": len(self._cache["responses"]),
            "understanding": len(self._cache["understanding"]),
            "embeddings": self.embedder.get_stats(),
            "vector": self.vector_store.get_stats().get("cache", {}),
            "graph": self.graph.get_cache_stats()
        }
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        return {
            **self.stats,
            "routing": self.router.get_stats()
        }
    
    def close(self):
        """Clean up resources."""
        self.graph.close()