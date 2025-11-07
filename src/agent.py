from typing import Dict, List, Optional, Tuple
import time, hashlib, asyncio, logging
from collections import deque
from openai import AsyncOpenAI
from src import config
from src.embeddings.embedder import Embedder
from src.vector.pinecone_client import PineconeClient
from src.graph.neo4j_client import Neo4jClient
from src.router import HybridRouter
from src.prompt import build_prompt, build_direct_prompt, build_understanding_prompt

logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY = 10
MAX_CACHE = 100
CACHE_TTL = 3600
TIMEOUT = 10.0

class TTLCache(dict):
    """Dict with TTL and size limit."""
    def __init__(self, max_size=MAX_CACHE, ttl=CACHE_TTL):
        super().__init__()
        self.max_size, self.ttl = max_size, ttl
        self._timestamps = {}
    
    def get(self, key, default=None):
        if key in self and time.time() - self._timestamps.get(key, 0) <= self.ttl:
            return super().get(key)
        self.pop(key, None)
        return default
    
    def __setitem__(self, key, value):
        if len(self) >= self.max_size:
            oldest = min(self._timestamps.items(), key=lambda x: x[1])[0]
            self.pop(oldest, None)
        super().__setitem__(key, value)
        self._timestamps[key] = time.time()

class GreetingHandler:
    """Centralized greeting detection and response."""
    GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hola", "sup", "yo"}
    INTROS = ["i'm ", "i m ", "i am ", "my name is ", "this is ", "call me "]
    TRAVEL = {"hotel", "restaurant", "visit", "trip", "travel", "itinerary", "tour", "vietnam", "hanoi"}
    
    @classmethod
    def is_greeting(cls, query: str) -> bool:
        q = query.lower().strip()
        has_greeting = any(q.startswith(g) for g in cls.GREETINGS)
        has_intro = any(p in q for p in cls.INTROS)
        has_travel = any(t in q for t in cls.TRAVEL)
        return (has_greeting or has_intro) and not has_travel
    
    @classmethod
    def extract_name(cls, query: str) -> Optional[str]:
        q = query.lower()
        for pattern in cls.INTROS:
            if pattern in q:
                parts = q.split(pattern, 1)
                if len(parts) > 1 and parts[1].split():
                    return parts[1].split()[0].strip(",.'\"!?").title()
        return None
    
    @classmethod
    def respond(cls, name: Optional[str] = None) -> str:
        greeting = f"Hello {name}! Nice to meet you." if name else "Hello!"
        return f"{greeting} I'm your Vietnam Travel Assistant. How can I help you plan your trip today?"

class HybridRunner:
    """Async RAG orchestrator with intelligent routing."""
    
    def __init__(self, top_k: int = 5, enable_streaming: bool = False):
        # Components
        self.embedder = Embedder()
        self.vector_store = PineconeClient()
        self.graph = Neo4jClient()
        self.router = HybridRouter()
        
        # LLM client
        if config.USE_CEREBRAS and config.CEREBRAS_API_KEY:
            self.llm_client = AsyncOpenAI(api_key=config.CEREBRAS_API_KEY, base_url="https://api.cerebras.ai/v1")
            self.chat_model = config.CEREBRAS_MODEL
        else:
            self.llm_client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            self.chat_model = config.CHAT_MODEL or "gpt-4o-mini"
        
        # Config
        self.top_k = top_k
        self.streaming = enable_streaming
        
        # State
        self.history = deque(maxlen=MAX_HISTORY * 2)
        self.response_cache = TTLCache()
        self.understanding_cache = TTLCache()
        self.timings = deque(maxlen=100)
    
    async def query(self, user_query: str, explain: bool = False) -> str:
        """Main query entry point."""
        start = time.time()
        timings = {"query": user_query, "phase1": 0, "embedding": 0, "vector": 0, "graph": 0, "llm": 0}
        self.history.append({"role": "user", "content": user_query})
        
        try:
            # Quick greeting check
            if GreetingHandler.is_greeting(user_query):
                response = GreetingHandler.respond(GreetingHandler.extract_name(user_query))
                return self._finalize(response, timings, start, "greeting", explain)
            
            # Parallel understanding + routing
            understanding, route = await self._understand_and_route(user_query, timings, explain)
            
            # Handle special cases
            if understanding.get("is_out_of_scope") or understanding.get("needs_clarification"):
                response = self._handle_special(understanding)
                return self._finalize(response, timings, start, "special", explain)
            
            # Execute strategy
            response = await self._execute(user_query, route, timings, explain)
            return self._finalize(response, timings, start, route["strategy"], explain)
            
        except asyncio.TimeoutError:
            return "I apologize, but the request timed out. Please try again."
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return "I apologize, but I encountered an error."
    
    async def _understand_and_route(self, query: str, timings: Dict, explain: bool) -> Tuple[Dict, Dict]:
        """Parallel understanding + routing."""
        t = time.time()
        
        # Run concurrently
        understanding_task = self._understand(query, explain)
        routing_task = asyncio.to_thread(self.router.route, query, None)
        understanding, route = await asyncio.gather(understanding_task, routing_task, return_exceptions=True)
        
        # Handle errors
        if isinstance(understanding, Exception):
            understanding = {"is_out_of_scope": False, "intent": "general", "confidence": 0.5}
        if isinstance(route, Exception):
            route = {"strategy": "hybrid", "confidence": 0.5, "reasoning": "fallback"}
        
        # Re-route with intent
        if intent := understanding.get("intent"):
            if intent not in ["greeting", "out_of_scope"]:
                route = await asyncio.to_thread(self.router.route, query, intent)
        
        timings["phase1"] = time.time() - t
        if explain:
            logger.info(f"🎯 {route['strategy']} ({route['confidence']:.2f})")
        return understanding, route
    
    async def _execute(self, query: str, route: Dict, timings: Dict, explain: bool) -> str:
        """Execute routing strategy."""
        strategy = route["strategy"]
        if strategy == "direct":
            return await self._direct(query, timings, explain)
        return await self._rag(query, route.get("filters", {}), timings, explain, strategy == "graph")
    
    async def _rag(self, query: str, filters: Dict, timings: Dict, explain: bool, graph_focus: bool) -> str:
        """RAG with vector + graph retrieval."""
        # Embed
        t = time.time()
        query_vec = await self.embedder.embed_async(query)
        timings["embedding"] = time.time() - t
        
        # Vector search
        t = time.time()
        top_k = 3 if graph_focus else self.top_k
        matches = await asyncio.to_thread(self.vector_store.query, query_vec, top_k)
        timings["vector"] = time.time() - t
        
        # Graph search
        t = time.time()
        names = [m.get("metadata", {}).get("name", "") for m in matches if m.get("metadata", {}).get("name")]
        neighbors = 20 if graph_focus else 10
        facts = await asyncio.to_thread(self.graph.search_by_topic_batch, names, neighbors, 4)
        timings["graph"] = time.time() - t
        
        if explain:
            logger.info(f"📊 {len(matches)} vectors + {len(facts)} facts")
        
        # Cache check
        cache_key = hashlib.md5(f"{query}|{'-'.join([m.get('id','')[:8] for m in matches[:3]])}|{len(facts)}".encode()).hexdigest()
        if cached := self.response_cache.get(cache_key):
            timings["llm"], timings["cached"] = 0.001, True
            if explain: logger.info("💾 Cache hit")
            return cached
        
        # Generate
        t = time.time()
        history = list(self.history)[-7:-1] if len(self.history) > 1 else []
        prompt = build_prompt(query, matches, facts, filters, history)
        max_tokens = 500 if graph_focus else 600
        response = await self._generate(prompt, max_tokens, explain)
        timings["llm"] = time.time() - t
        
        self.response_cache[cache_key] = response
        return response
    
    async def _direct(self, query: str, timings: Dict, explain: bool) -> str:
        """Direct LLM without retrieval."""
        t = time.time()
        history = list(self.history)[-7:-1] if len(self.history) > 1 else []
        prompt = build_direct_prompt(query, history)
        response = await self._generate(prompt, 400, explain)
        timings["llm"] = time.time() - t
        return response
    
    async def _generate(self, prompt: List[Dict], max_tokens: int, explain: bool) -> str:
        """Generate LLM response."""
        if self.streaming:
            return await self._stream(prompt, max_tokens, explain)
        
        resp = await asyncio.wait_for(
            self.llm_client.chat.completions.create(
                model=self.chat_model, messages=prompt, max_tokens=max_tokens, temperature=0.2
            ), timeout=TIMEOUT
        )
        return resp.choices[0].message.content
    
    async def _stream(self, prompt: List[Dict], max_tokens: int, explain: bool) -> str:
        """Stream LLM response."""
        if not explain:
            print("\n" + "="*60 + "\n🤖 Assistant:\n" + "="*60)
        
        stream = await asyncio.wait_for(
            self.llm_client.chat.completions.create(
                model=self.chat_model, messages=prompt, max_tokens=max_tokens, 
                temperature=0.2, stream=True
            ), timeout=TIMEOUT
        )
        
        full = ""
        async for chunk in stream:
            if content := chunk.choices[0].delta.content:
                print(content, end='', flush=True)
                full += content
        
        print("\n" + ("="*60 if not explain else ""))
        return full
    
    async def _understand(self, query: str, explain: bool) -> Dict:
        """Understand query intent."""
        if cached := self.understanding_cache.get(query):
            return cached
        
        history = list(self.history)[-7:-1] if len(self.history) > 1 else []
        messages = build_understanding_prompt(query, history)
        
        try:
            resp = await asyncio.wait_for(
                self.llm_client.chat.completions.create(
                    model=self.chat_model, messages=messages, temperature=0.2,
                    max_tokens=150, response_format={"type": "json_object"}
                ), timeout=8.0
            )
            import json
            result = json.loads(resp.choices[0].message.content)
            self.understanding_cache[query] = result
            return result
        except Exception as e:
            logger.warning(f"Understanding error: {e}")
            return {"is_out_of_scope": False, "intent": "general", "confidence": 0.5}
    
    def _handle_special(self, understanding: Dict) -> str:
        """Handle out-of-scope or clarification."""
        if understanding.get("is_out_of_scope"):
            return (
                "I apologize, but I'm specifically designed to assist with Vietnam travel planning. "
                "I can help with hotels, restaurants, attractions, itineraries, culture, and transportation.\n\n"
                "Is there anything about traveling to Vietnam I can help you with?"
            )
        return understanding.get("clarification_message", GreetingHandler.respond())
    
    def _finalize(self, response: str, timings: Dict, start: float, strategy: str, explain: bool = False) -> str:
        """Finalize query."""
        timings["total"] = time.time() - start
        timings["strategy"] = strategy
        self.timings.append(timings)
        self.history.append({"role": "assistant", "content": response})
        
        if explain:
            parts = [f"⏱️ {strategy}:"]
            if timings.get("phase1"): parts.append(f"Phase1={timings['phase1']:.3f}s")
            if timings.get("embedding"): parts.append(f"Embed={timings['embedding']:.3f}s")
            if timings.get("vector"): parts.append(f"Vec={timings['vector']:.3f}s")
            if timings.get("graph"): parts.append(f"Graph={timings['graph']:.3f}s")
            parts.append(f"LLM={timings['llm']:.3f}s Total={timings['total']:.3f}s")
            if timings.get("cached"): parts.append("💾")
            logger.info(" | ".join(parts))
        
        return response
    
    @property
    def stats(self) -> Dict:
        """Get statistics."""
        if not self.timings:
            return {"total_queries": 0}
        n = len(self.timings)
        avg = lambda k: sum(t.get(k, 0) for t in self.timings) / n
        return {
            "total_queries": n,
            "avg_total": avg("total"),
            "avg_phase1": avg("phase1"),
            "avg_embedding": avg("embedding"),
            "avg_vector": avg("vector"),
            "avg_graph": avg("graph"),
            "avg_llm": avg("llm")
        }
    
    def print_timing_summary(self):
        """Print timing summary."""
        s = self.stats
        if s["total_queries"] == 0:
            print("No queries processed yet.")
            return
        print(f"\n{'='*60}\n⏱️ SUMMARY: {s['total_queries']} queries\n{'='*60}")
        print(f"Avg Total:  {s['avg_total']:.3f}s")
        print(f"Avg Phase1: {s['avg_phase1']:.3f}s")
        print(f"Avg Embed:  {s['avg_embedding']:.3f}s")
        print(f"Avg Vector: {s['avg_vector']:.3f}s | Graph: {s['avg_graph']:.3f}s")
        print(f"Avg LLM:    {s['avg_llm']:.3f}s\n{'='*60}")
    
    def get_cache_stats(self) -> Dict:
        return {
            "responses": len(self.response_cache),
            "understanding": len(self.understanding_cache),
            "embeddings": self.embedder.get_stats(),
            "vector": self.vector_store.get_stats().get("cache", {}),
            "graph": self.graph.get_cache_stats()
        }
    
    def clear_cache(self):
        self.response_cache.clear()
        self.understanding_cache.clear()
    
    def reset_conversation(self):
        self.history.clear()
    
    def close(self):
        self.graph.close()