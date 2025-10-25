import re
import json
from typing import Dict, Optional
from openai import OpenAI
from src import config
from src.prompt import build_routing_prompt


class HybridRouter:
    """
    4-Tier Intelligent Query Router with cascade priority:
    1. Scope validation (out-of-scope destinations) → REFUSE
    2. Rule-based keywords → GRAPH/DIRECT strategy  
    3. Semantic (optional) → Vector similarity on intent prototypes
    4. LLM classification → Expensive fallback (disabled by default)
    """
    
    # Strategy constants
    STRATEGY_GRAPH = "graph"        # Knowledge graph: relational, proximity queries
    STRATEGY_VECTOR = "vector"      # Vector DB: descriptive, semantic queries
    STRATEGY_HYBRID = "hybrid"      # Combined vector + graph (default)
    STRATEGY_DIRECT = "direct"      # Direct LLM: general knowledge, no retrieval
    
    # Keyword patterns by strategy (priority order)
    KEYWORDS = {
        STRATEGY_GRAPH: ["near", "nearby", "close to", "around", "connected", "list of", "find me", "what are"],
        STRATEGY_DIRECT: ["weather", "climate", "visa", "culture", "history", "how to", "when to", "tell me about", "explain"],
    }
    
    # Valid destinations (in-scope for Vietnam travel dataset)
    VALID_DESTINATIONS = [
        "vietnam", "hanoi", "ho chi minh", "hcmc", "saigon", "danang", "da nang",
        "hue", "nha trang", "dalat", "da lat", "phu quoc", "sapa", "sa pa",
        "halong", "ha long", "halong bay", "hoi an", "mekong"
    ]
    
    # Common out-of-scope destinations
    OUT_OF_SCOPE_DESTINATIONS = [
        "russia", "china", "thailand", "cambodia", "laos", "myanmar", "singapore",
        "malaysia", "indonesia", "japan", "korea", "india", "europe", "america",
        "africa", "australia", "paris", "london", "tokyo", "beijing", "moscow"
    ]
    
    def __init__(self, enable_semantic: bool = False):
       
        self.llm_client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.chat_model = config.CHAT_MODEL or "gpt-4o-mini"
        self.enable_semantic = enable_semantic
        
        # Statistics tracking
        self.stats = {
            "rule": 0,      # Rule-based routing
            "semantic": 0,  # Semantic vector routing
            "llm": 0,       # LLM fallback
            "total": 0
        }
        
        # Confidence threshold for semantic routing (before LLM fallback)
        self.SEMANTIC_CONFIDENCE_THRESHOLD = 0.75
    
    def route(self, query: str, llm_intent: str = None) -> Dict:
        """
        Route query through 4-tier cascade with optional LLM intent override.
        Args:
            query: User query string
            llm_intent: Optional semantic intent from LLM understanding phase
        Returns: {strategy, method, filters, confidence, reasoning}
        """
        self.stats["total"] += 1
        query_lower = query.lower()
        
        # --- TIER 0: LLM Intent Override (Highest Priority) ---
        # If we have semantic intent from understanding phase, use it to inform routing
        if llm_intent:
            intent_lower = llm_intent.lower()
            
            # Check for location/relationship queries
            if any(kw in intent_lower for kw in ["nearby", "near", "close to", "around", "location-based", "find places", "relationships"]):
                self.stats["rule"] += 1
                return self._build_result(
                    self.STRATEGY_GRAPH,
                    f"LLM Intent: {llm_intent} → Graph strategy",
                    method="llm_intent",
                    confidence=0.95,
                    filters=self._extract_filters(query)
                )
            
            # Check for direct/conversational queries
            if any(kw in intent_lower for kw in ["follow-up", "clarification", "direct answer", "general knowledge", "conversational"]):
                self.stats["rule"] += 1
                return self._build_result(
                    self.STRATEGY_DIRECT,
                    f"LLM Intent: {llm_intent} → Direct strategy",
                    method="llm_intent",
                    confidence=0.95
                )
        
        # --- TIER 1: Scope Validation (Detect out-of-scope queries) ---
        if self._is_out_of_scope(query_lower):
            self.stats["rule"] += 1
            return self._build_result(
                "out_of_scope",
                "Query is outside Vietnam travel domain - no retrieval needed",
                method="rule",
                confidence=0.95
            )
        
        # --- TIER 2: Rule-Based Routing (High Priority, 0 tokens) ---
        
        # Check for graph keywords
        if any(kw in query_lower for kw in self.KEYWORDS[self.STRATEGY_GRAPH]):
            self.stats["rule"] += 1
            return self._build_result(
                self.STRATEGY_GRAPH,
                "Rule: Detected relational/proximity keyword",
                method="rule",
                confidence=0.90,
                filters=self._extract_filters(query)
            )
        
        # Check for direct/general knowledge keywords
        if any(kw in query_lower for kw in self.KEYWORDS[self.STRATEGY_DIRECT]):
            self.stats["rule"] += 1
            return self._build_result(
                self.STRATEGY_DIRECT,
                "Rule: Detected general knowledge keyword",
                method="rule",
                confidence=0.90
            )
        
        # --- TIER 3: Semantic Routing (Optional, Low cost: embedding only) ---
        if self.enable_semantic:
            semantic_result = self._semantic_route(query)
            if semantic_result and semantic_result.get("confidence", 0) >= self.SEMANTIC_CONFIDENCE_THRESHOLD:
                self.stats["semantic"] += 1
                return semantic_result
        
        # --- TIER 4: LLM Classification (Expensive Fallback) ---
        # Only runs if rules and semantic are inconclusive
        # Disabled by default to save tokens - just default to hybrid
        # Uncomment to enable LLM fallback:
        # self.stats["llm"] += 1
        # return self._llm_classify(query)
        
        # Default to hybrid (safe fallback)
        self.stats["rule"] += 1
        return self._build_result(
            self.STRATEGY_HYBRID,
            "Default: Hybrid strategy (no specific patterns matched)",
            method="rule",
            confidence=0.70,
            filters=self._extract_filters(query)
        )
    
    def _build_result(self, strategy: str, reasoning: str, method: str = "rule", 
                     confidence: float = 1.0, filters: Optional[Dict] = None) -> Dict:
        """Helper to build routing result dictionary."""
        return {
            "strategy": strategy,
            "method": method,
            "reasoning": reasoning,
            "confidence": confidence,
            "filters": filters or {}
        }
    
    def _is_out_of_scope(self, query_lower: str) -> bool:
        """
        Check if query is outside Vietnam travel domain.
        Returns True if query mentions non-Vietnam destinations.
        """
        # Check for explicit out-of-scope destinations
        for destination in self.OUT_OF_SCOPE_DESTINATIONS:
            if destination in query_lower:
                # Make sure it's not a false positive (e.g., "russia" in "Prussian")
                # Check word boundaries
                if re.search(rf'\b{destination}\b', query_lower):
                    return True
        
        # Check if query has trip/itinerary keywords but NO Vietnam destinations
        trip_keywords = ["trip", "itinerary", "visit", "tour", "travel to", "go to", "plan"]
        has_trip_keyword = any(kw in query_lower for kw in trip_keywords)
        
        if has_trip_keyword:
            # Check if ANY Vietnam destination is mentioned
            has_vietnam_dest = any(dest in query_lower for dest in self.VALID_DESTINATIONS)
            
            if not has_vietnam_dest:
                # Trip query without Vietnam destination = likely out of scope
                return True
        
        return False
    
    def _extract_filters(self, query: str) -> Dict:
        """Extract simple filters: city and type."""
        filters = {}
        query_lower = query.lower()
        
        # Extract city
        cities = ["hanoi", "ho chi minh", "hcmc", "saigon", "danang", "hue", "nha trang", "dalat", "sapa", "halong"]
        for city in cities:
            if city in query_lower:
                filters["city"] = city.title() if city != "hcmc" else "Ho Chi Minh"
                break
        
        # Extract type
        types = ["hotel", "restaurant", "attraction", "beach", "temple", "museum", "cafe"]
        for typ in types:
            if typ in query_lower:
                filters["type"] = typ.title()
                break
        
        return filters
    
    def _semantic_route(self, query: str) -> Optional[Dict]:
        """
        Optional semantic routing using vector DB for intent classification.
        """
        return None  # Not implemented yet
        
    
    def _llm_classify(self, query: str) -> Dict:
        """
        Use LLM to classify query intent (expensive fallback).
        Only called when rules and semantic routing are inconclusive.
        """
        # Use centralized prompt builder
        strategies = {
            'graph': self.STRATEGY_GRAPH,
            'vector': self.STRATEGY_VECTOR,
            'hybrid': self.STRATEGY_HYBRID,
            'direct': self.STRATEGY_DIRECT
        }
        messages = build_routing_prompt(query, strategies)

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,
                max_tokens=100,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(resp.choices[0].message.content)
            
            return self._build_result(
                result.get("strategy", self.STRATEGY_HYBRID),
                f"LLM: {result.get('reasoning', 'Classified by LLM')}",
                method="llm",
                confidence=result.get("confidence", 0.75)
            )
            
        except Exception as e:
            print(f"⚠ LLM classification failed: {e}")
            return self._build_result(
                self.STRATEGY_HYBRID,
                f"LLM failed, defaulting to hybrid: {e}",
                method="llm",
                confidence=0.50
            )
    
    def get_stats(self) -> Dict:
        """Get routing statistics with percentages."""
        total = self.stats["total"]
        if total == 0:
            return {**self.stats, "rule_pct": "0%", "semantic_pct": "0%", "llm_pct": "0%"}
        
        return {
            "total_queries": total,
            "rule_count": self.stats["rule"],
            "semantic_count": self.stats["semantic"],
            "llm_count": self.stats["llm"],
            "rule_pct": f"{(self.stats['rule'] / total * 100):.1f}%",
            "semantic_pct": f"{(self.stats['semantic'] / total * 100):.1f}%",
            "llm_pct": f"{(self.stats['llm'] / total * 100):.1f}%"
        }
    
    def explain_route(self, query: str) -> str:
        """Get human-readable explanation of routing decision."""
        result = self.route(query)
        
        lines = [
            f"📍 Query: '{query}'",
            f"🎯 Strategy: {result['strategy']}",
            f"🔧 Method: {result['method']} (confidence: {result['confidence']:.2f})",
            f"💭 Reasoning: {result['reasoning']}"
        ]
        
        if result.get('filters'):
            lines.append(f"🔍 Filters: {result['filters']}")
        
        return "\n".join(lines)


# Convenience function for one-off routing
def route_query(query: str) -> Dict:
    """Quick routing function without instantiating router."""
    router = HybridRouter()
    return router.route(query)
