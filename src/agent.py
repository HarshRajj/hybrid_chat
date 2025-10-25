from typing import Dict, List, Optional
import time
import hashlib
import asyncio
import logging
from openai import AsyncOpenAI
from src import config
from src.embeddings.embedder import Embedder
from src.vector.pinecone_client import PineconeClient
from src.graph.neo4j_client import Neo4jClient
from src.router import HybridRouter
from src.prompt import build_prompt, build_direct_prompt, build_understanding_prompt

logger = logging.getLogger(__name__)

# --- Helper Functions for IO (To remove boilerplate in search methods) ---

async def _run_sync_io(func, *args):
    """Executes a synchronous I/O function in a thread pool executor."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, *args)

# --- Main Orchestrator Class ---

class HybridRunner:
    """
    Async query orchestrator with intelligent routing and multi-level caching.
    Routes queries to hybrid/graph/direct strategies with parallel vector+graph retrieval.
    """
    
    def __init__(self, top_k: int = 5, enable_streaming: bool = False):
        self.embedder = Embedder()
        self.vector_store = PineconeClient()
        self.graph = Neo4jClient()
        self.router = HybridRouter()
        self.llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
        
        self.top_k = top_k
        self.chat_model = config.CHAT_MODEL or "gpt-4o-mini"
        self.enable_streaming = enable_streaming
        
        self.conversation_history: List[Dict] = []
        self._cache: Dict[str, any] = {
            "understanding": {}, 
            "responses": {},     
            "timings": []        
        }
        self.GREETINGS = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hola", "sup", "yo"]
    
    # === PUBLIC ENTRY POINT ===
    async def query(self, user_query: str, explain: bool = False) -> str:
        start_time = time.time()
        timings = self._initialize_timings(user_query)
        self.conversation_history.append({"role": "user", "content": user_query})
        
        # PHASE 1: Quick checks
        phase1_start = time.time()
        query_lower = user_query.strip().lower()
        
        # --- FIX 1: TIGHTEN GREETING LOGIC ---
        is_pure_greeting = query_lower in self.GREETINGS or (len(query_lower) <= 3 and any(g.startswith(query_lower) for g in self.GREETINGS))
        
        if is_pure_greeting:
            response = "Hello! I'm your Vietnam Travel Assistant. How can I help you plan your trip today?"
            self._finalize_query(response, timings, start_time, phase1_start, strategy="greeting")
            return response
        
        # PHASE 2: LLM Understanding (Semantic Intent & Scope Check)
        understanding_start = time.time()
        understanding = await self._understand_query(user_query, explain=explain)
        timings["understanding"] = time.time() - understanding_start
        
        if understanding.get("is_out_of_scope", False) or understanding.get("needs_clarification", False):
            response = self._handle_out_of_scope(user_query) if understanding["is_out_of_scope"] else understanding.get("clarification_message", "Could you please clarify your question?")
            strategy = "out_of_scope" if understanding["is_out_of_scope"] else "clarification"
            self._finalize_query(response, timings, start_time, phase1_start, strategy=strategy)
            return response
        
        # PHASE 3: Route and execute query
        timings["phase1"] = time.time() - phase1_start
        routing_start = time.time()
        
        llm_intent = understanding.get("intent", "")
        route = self.router.route(user_query, llm_intent=llm_intent)
        timings["routing"] = time.time() - routing_start
        timings["strategy"] = route["strategy"]
        
        if explain:
            logger.info(f"✅ Phase 1 passed ({timings['phase1']:.3f}s) → Retrieving data...")
            logger.info(f"🎯 Strategy: {route['strategy']} (confidence: {route['confidence']:.2f}) - {route['reasoning']}")
        
        # Execute strategy
        strategy_map = {
            "direct": self._direct_llm,
            "graph": self._rag_search, # Merged logic
            "hybrid": self._rag_search  # Merged logic
        }
        response = await strategy_map.get(route["strategy"], self._rag_search)(user_query, route.get("filters", {}), timings, explain, is_graph_focused=(route["strategy"] == "graph"))
        
        self._finalize_query(response, timings, start_time, phase1_start, strategy=route["strategy"], explain=explain)
        return response
    
    # === CORE RAG METHODS (CONSOLIDATED) ===
    
    async def _rag_search(self, query: str, filters: Dict, timings: Dict, explain: bool = False, is_graph_focused: bool = False) -> str:
        """Handles both Hybrid and Graph-Focused RAG execution."""
        
        # 1. Embed query (Sync operation, kept outside parallel block)
        embed_start = time.time()
        query_vec = self.embedder.embed(query)
        timings["embedding"] = time.time() - embed_start
        
        # 2. Fetch data (Sync operations run in executor)
        top_k_vec = 3 if is_graph_focused else self.top_k
        graph_neighbors = 20 if is_graph_focused else 10 # More neighbors for graph
        
        # Vector search (needed for node IDs)
        vector_start = time.time()
        matches = await _run_sync_io(self.vector_store.query, query_vec, top_k_vec)
        timings["vector_search"] = time.time() - vector_start
        
        # Graph search (dependent on vector results)
        node_ids = [m["id"] for m in matches]
        graph_start = time.time()
        graph_facts = await _run_sync_io(self.graph.search_by_topic_batch, node_ids, graph_neighbors, 4)
        timings["graph_search"] = time.time() - graph_start
        
        if explain:
            logger.info(f"📊 Retrieved {len(matches)} vectors + {len(graph_facts)} graph facts (Graph Focus: {is_graph_focused})")

        # 3. Cache Check
        cache_key = self._generate_cache_key(query, matches, graph_facts)
        if cached := self._check_response_cache(cache_key, timings, explain):
            return cached
        
        # 4. Generate Response
        llm_start = time.time()
        history_context = self.conversation_history[-5:-1] if len(self.conversation_history) > 1 else []
        prompt = build_prompt(query, matches, graph_facts, filters, history=history_context)
        max_tokens = self._get_max_tokens(query, "graph" if is_graph_focused else "hybrid")
        
        response = await self._generate_llm_response(prompt, max_tokens, explain)
        
        timings["llm_generation"] = time.time() - llm_start
        self._cache["responses"][cache_key] = response
        
        return response
    
    async def _direct_llm(self, query: str, timings: Dict, explain: bool = False) -> str:
        """Direct LLM response without retrieval."""
        llm_start = time.time()
        history_context = self.conversation_history[-5:-1] if len(self.conversation_history) > 1 else []
        prompt = build_direct_prompt(query, history=history_context)
        max_tokens = self._get_max_tokens(query, "direct")
        
        response = await self._generate_llm_response(prompt, max_tokens, explain)
        
        timings["llm_generation"] = time.time() - llm_start
        return response

    # === PRIVATE HELPERS: LLM GENERATION ===

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

    # === PRIVATE HELPERS: I/O & CACHE (SIMPLIFIED) ===

    async def _understand_query(self, query: str, explain: bool = False) -> Dict:
        """Check if query is out-of-scope using LLM with conversation context."""
        if query in self._cache["understanding"]:
            if explain: logger.debug("💾 Using cached understanding")
            return self._cache["understanding"][query]
        
        recent_history = self.conversation_history[-5:-1] if len(self.conversation_history) > 1 else None
        messages = build_understanding_prompt(query, history=recent_history)

        try:
            resp = await self.llm_client.chat.completions.create(
                model=self.chat_model, messages=messages, temperature=0.2, max_tokens=200, response_format={"type": "json_object"}
            )
            import json
            result = json.loads(resp.choices[0].message.content)
            self._cache["understanding"][query] = result
            if explain: logger.debug(f"📊 Understanding: {result.get('intent', 'analyzing...')}")
            return result
        except Exception as e:
            if explain: logger.warning(f"⚠️ Understanding phase error: {e}")
            return {"is_out_of_scope": False, "needs_clarification": False, "intent": "proceeding with query", "confidence": 0.5}

    def _generate_cache_key(self, query: str, matches: List, graph_facts: List) -> str:
        """Generate cache key from query + retrieval results only (not history for better cache hits)."""
        match_ids = [m.get('id', '') for m in matches[:3]]
        cache_str = f"{query.lower().strip()}|{'-'.join(match_ids)}|{len(graph_facts)}"
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _check_response_cache(self, cache_key: str, timings: Dict, explain: bool) -> Optional[str]:
        if cache_key in self._cache["responses"]:
            if explain: logger.debug("💾 Cache hit!")
            timings["llm_generation"] = 0.001
            timings["used_cache"] = True
            return self._cache["responses"][cache_key]
        return None
    
    def _get_max_tokens(self, query: str, strategy: str = "hybrid") -> int:
        return {"direct": 400, "graph": 500, "hybrid": 600}.get(strategy, 600)
    
    def _handle_out_of_scope(self, query: str) -> str:
        return (
            "I apologize, but I'm specifically designed for Vietnam travel queries. "
            "I can help with destinations, hotels, restaurants, attractions, itineraries, and general Vietnam travel info. "
            "Would you like to ask about traveling in Vietnam instead?"
        )
    
    # === PRIVATE HELPERS: TIMING & STATS (CONSOLIDATED) ===

    def _initialize_timings(self, query: str) -> Dict:
        return {
            "query": query,
            "phase1": 0, "understanding": 0, "routing": 0, "embedding": 0,
            "vector_search": 0, "graph_search": 0, "llm_generation": 0,
            "total": 0, "strategy": "", "used_cache": False
        }
        
    def _finalize_query(self, response: str, timings: Dict, start_time: float, phase1_start: float, strategy: str, explain: bool = False):
        timings["total"] = time.time() - start_time
        timings["phase1"] = time.time() - phase1_start
        timings["strategy"] = strategy
        self._cache["timings"].append(timings)
        self.conversation_history.append({"role": "assistant", "content": response})
        if explain:
            self._print_timing_breakdown(timings)

    def _print_timing_breakdown(self, timings: Dict):
        parts = [f"⏱️  Phase1={timings['phase1']:.3f}s"]
        if timings.get('understanding', 0) > 0: parts.append(f"Understanding={timings['understanding']:.3f}s")
        parts.append(f"Routing={timings['routing']:.3f}s")
        if timings['embedding'] > 0: parts.append(f"Embed={timings['embedding']:.3f}s")
        if timings['vector_search'] > 0: parts.append(f"Vector={timings['vector_search']:.3f}s")
        if timings['graph_search'] > 0: parts.append(f"Graph={timings['graph_search']:.3f}s")
        parts.append(f"LLM={timings['llm_generation']:.3f}s")
        parts.append(f"Total={timings['total']:.3f}s")
        if timings.get('used_cache'): parts.append("💾")
        logger.info(" | ".join(parts))

    # === PUBLIC STATS & MAINTENANCE METHODS ===

    @property
    def stats(self) -> Dict:
        # Implementation remains the same
        timings = self._cache["timings"]
        if not timings:
            return {"total_queries": 0, "cache": self.get_cache_stats()}
        
        num_queries = len(timings)
        # Calculate averages safely
        avg_func = lambda key: sum(t.get(key, 0) for t in timings) / num_queries
        
        return {
            "total_queries": num_queries,
            "avg_total": avg_func("total"),
            "avg_phase1": avg_func("phase1"),
            "avg_understanding": avg_func("understanding"),
            "avg_embedding": avg_func("embedding"),
            "avg_vector": avg_func("vector_search"),
            "avg_graph": avg_func("graph_search"),
            "avg_llm": avg_func("llm_generation"),
            "recent_queries": timings[-10:],
            "cache": self.get_cache_stats()
        }
    
    def get_timing_stats(self) -> Dict: return self.stats
    
    def print_timing_summary(self):
        stats = self.stats
        if stats["total_queries"] == 0:
            print("No queries processed yet.")
            return
        
        print("\n" + "="*60)
        print(f"⏱️  SUMMARY: {stats['total_queries']} queries")
        print("="*60)
        print(f"Avg Total:  {stats['avg_total']:.3f}s")
        print(f"Avg Phase1: {stats['avg_phase1']:.3f}s | Understanding: {stats['avg_understanding']:.3f}s")
        print(f"Avg Embed:  {stats['avg_embedding']:.3f}s")
        print(f"Avg Vector: {stats['avg_vector']:.3f}s | Graph: {stats['avg_graph']:.3f}s")
        print(f"Avg LLM:    {stats['avg_llm']:.3f}s")
        print("="*60)
    
    def clear_response_cache(self):
        self._cache["responses"].clear()
        logger.info("✓ Response cache cleared")
    
    def get_cache_stats(self) -> Dict:
        return {
            "responses": len(self._cache["responses"]),
            "understanding": len(self._cache["understanding"]),
            "embeddings": self.embedder.get_stats(),
            "vector": self.vector_store.get_stats().get("cache", {}),
            "graph": self.graph.get_cache_stats()
        }
    
    def get_performance_stats(self) -> Dict:
        return {**self.stats, "routing": self.router.get_stats()}
    
    def reset_conversation(self):
        self.conversation_history = []
    
    def close(self):
        self.graph.close()