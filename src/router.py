import re, json
from typing import Dict, Optional, Set
from openai import OpenAI
from src import config
from src.prompt import build_routing_prompt

class HybridRouter:
    """Intelligent 4-tier query router with caching."""
    
    # Strategy constants
    GRAPH, VECTOR, HYBRID, DIRECT = "graph", "vector", "hybrid", "direct"
    
    # Keywords
    GRAPH_KW: Set[str] = {"near", "nearby", "close to", "around", "connected", "list of", "find me", "what are", "show me"}
    DIRECT_KW: Set[str] = {"weather", "climate", "visa", "culture", "history", "how to", "when to", "tell me about", "explain", "why"}
    
    # Destinations
    VIETNAM: Set[str] = {"vietnam", "hanoi", "ho chi minh", "hcmc", "saigon", "danang", "da nang", "hue", "nha trang", "dalat", "phu quoc", "sapa", "halong", "hoi an", "mekong"}
    OUT_SCOPE: Set[str] = {"russia", "china", "thailand", "cambodia", "laos", "myanmar", "singapore", "malaysia", "indonesia", "japan", "korea", "india", "europe", "america", "paris", "london", "tokyo", "bangkok"}
    
    def __init__(self, enable_llm_fallback: bool = False):
        self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.chat_model = config.CHAT_MODEL or "gpt-4o-mini"
        self.enable_llm = enable_llm_fallback
        
        self.stats = {"rule": 0, "intent": 0, "llm": 0, "total": 0}
        self._cache = {}
        self._patterns = {dest: re.compile(rf'\b{re.escape(dest)}\b', re.I) for dest in self.OUT_SCOPE}
    
    def route(self, query: str, llm_intent: Optional[str] = None) -> Dict:
        """Route query through cascade."""
        self.stats["total"] += 1
        
        # Cache check
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        q = cache_key
        words = set(q.split())
        
        # Tier 1: LLM intent override
        if llm_intent and (result := self._route_intent(llm_intent, q)):
            self.stats["intent"] += 1
            return self._cache_result(cache_key, result)
        
        # Tier 2: Rule-based
        # Graph keywords
        if words & self.GRAPH_KW or self._has_phrase(q, self.GRAPH_KW):
            result = self._result(self.GRAPH, "Rule: Relational query", 0.90, self._filters(q))
            self.stats["rule"] += 1
            return self._cache_result(cache_key, result)
        
        # Direct keywords
        if words & self.DIRECT_KW or self._has_phrase(q, self.DIRECT_KW):
            result = self._result(self.DIRECT, "Rule: General knowledge", 0.90)
            self.stats["rule"] += 1
            return self._cache_result(cache_key, result)
        
        # Tier 3: Scope validation
        if self._out_of_scope(q):
            result = self._result("out_of_scope", "Query outside Vietnam", 0.95)
            self.stats["rule"] += 1
            return self._cache_result(cache_key, result)
        
        # Tier 4: LLM fallback (optional)
        if self.enable_llm:
            result = self._llm_classify(query)
            self.stats["llm"] += 1
            return self._cache_result(cache_key, result)
        
        # Default: Hybrid
        result = self._result(self.HYBRID, "Default: Hybrid", 0.70, self._filters(q))
        self.stats["rule"] += 1
        return self._cache_result(cache_key, result)
    
    def _route_intent(self, intent: str, query: str) -> Optional[Dict]:
        """Route from LLM intent."""
        i = intent.lower()
        
        if any(k in i for k in ["nearby", "near", "location", "find places", "relationships"]):
            return self._result(self.GRAPH, f"Intent: {intent} → Graph", 0.95, self._filters(query))
        
        if any(k in i for k in ["follow-up", "clarification", "direct", "general", "conversational"]):
            return self._result(self.DIRECT, f"Intent: {intent} → Direct", 0.95)
        
        if any(k in i for k in ["recommendation", "best", "find", "search", "suggest"]):
            return self._result(self.HYBRID, f"Intent: {intent} → Hybrid", 0.90, self._filters(query))
        
        return None
    
    def _has_phrase(self, text: str, phrases: Set[str]) -> bool:
        """Check multi-word phrases."""
        return any(p in text for p in phrases if ' ' in p)
    
    def _out_of_scope(self, query: str) -> bool:
        """Check if out of scope."""
        # Check explicit out-of-scope destinations
        for dest, pattern in self._patterns.items():
            if pattern.search(query):
                return True
        
        # Trip without Vietnam destination
        trip_kw = {"trip", "itinerary", "visit", "tour", "travel to", "go to"}
        if any(k in query for k in trip_kw):
            if not any(d in query for d in self.VIETNAM):
                return True
        
        return False
    
    def _filters(self, query: str) -> Dict:
        """Extract filters."""
        filters = {}
        
        # Cities
        cities = [("hanoi", "Hanoi"), ("ho chi minh", "Ho Chi Minh"), ("hcmc", "Ho Chi Minh"), 
                  ("saigon", "Ho Chi Minh"), ("danang", "Da Nang"), ("da nang", "Da Nang"),
                  ("hue", "Hue"), ("nha trang", "Nha Trang"), ("dalat", "Da Lat"), 
                  ("sapa", "Sapa"), ("halong", "Ha Long")]
        
        for pattern, name in cities:
            if pattern in query:
                filters["city"] = name
                break
        
        # Types
        types = [("hotel", "Hotel"), ("restaurant", "Restaurant"), ("attraction", "Attraction"),
                 ("beach", "Beach"), ("temple", "Temple"), ("museum", "Museum"), ("cafe", "Cafe")]
        
        for pattern, name in types:
            if pattern in query:
                filters["type"] = name
                break
        
        return filters
    
    def _llm_classify(self, query: str) -> Dict:
        """LLM classification fallback."""
        strategies = {'graph': self.GRAPH, 'vector': self.VECTOR, 'hybrid': self.HYBRID, 'direct': self.DIRECT}
        messages = build_routing_prompt(query, strategies)
        
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.chat_model, messages=messages, temperature=0.1,
                max_tokens=80, response_format={"type": "json_object"}
            )
            result = json.loads(resp.choices[0].message.content)
            return self._result(
                result.get("strategy", self.HYBRID),
                f"LLM: {result.get('reasoning', 'Classified')[:50]}",
                result.get("confidence", 0.75)
            )
        except Exception as e:
            return self._result(self.HYBRID, f"LLM failed: {str(e)[:40]}", 0.50)
    
    def _result(self, strategy: str, reasoning: str, confidence: float = 1.0, filters: Optional[Dict] = None) -> Dict:
        """Build result."""
        return {"strategy": strategy, "reasoning": reasoning, "confidence": confidence, "filters": filters or {}}
    
    def _cache_result(self, key: str, result: Dict) -> Dict:
        """Cache with size limit."""
        if len(self._cache) >= 100:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = result
        return result
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        total = self.stats["total"]
        if total == 0:
            return {**self.stats}
        
        return {
            "total_queries": total,
            "rule_count": self.stats["rule"],
            "intent_count": self.stats["intent"],
            "llm_count": self.stats["llm"],
            "rule_pct": f"{(self.stats['rule'] / total * 100):.1f}%",
            "intent_pct": f"{(self.stats['intent'] / total * 100):.1f}%",
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        self._cache.clear()
    
    def explain_route(self, query: str) -> str:
        """Explain routing decision."""
        r = self.route(query)
        lines = [f"🔍 Query: '{query}'", f"🎯 Strategy: {r['strategy']}", 
                 f"💭 Reasoning: {r['reasoning']}", f"📊 Confidence: {r['confidence']:.2f}"]
        if r.get('filters'):
            lines.append(f"🔎 Filters: {r['filters']}")
        return "\n".join(lines)

def route_query(query: str) -> Dict:
    """Convenience function."""
    return HybridRouter().route(query)