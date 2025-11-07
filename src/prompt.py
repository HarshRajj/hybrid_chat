from typing import List, Dict, Optional

# System prompts (cached)
UNDERSTAND_SYS = """You are a query analyzer for a Vietnam travel assistant.

**TASK:** Classify query and output JSON.

**GREETINGS:** Pure greetings/intros with NO travel intent → needs_clarification=true
**OUT-OF-SCOPE:** Non-Vietnam destinations, non-travel topics → is_out_of_scope=true
**IN-SCOPE:** Vietnam travel, hotels, restaurants, culture, itineraries

**JSON OUTPUT:**
{
    "is_out_of_scope": boolean,
    "needs_clarification": boolean,
    "clarification_message": "string",
    "intent": "string (user goal or 'greeting'/'out_of_scope')",
    "confidence": 0.0-1.0
}"""

RAG_SYS = """You are a Vietnam travel assistant. Answer concisely based on search results.

RULES:
- ONLY Vietnam travel questions
- CANNOT book/reserve or access real-time pricing
- Cite place names
- Plain text only (no markdown)
- Numbered lists if needed"""

DIRECT_SYS = """You are a Vietnam travel assistant for general questions.

RULES:
- ONLY Vietnam travel (weather, culture, visas, transportation)
- For non-travel/other countries → "I can only assist with Vietnam travel."
- CANNOT book/reserve
- Plain text only, concise"""

ROUTING_SYS = """Classify query into ONE strategy:
- {graph}: Relational/proximity (near, connected, list)
- {vector}: Descriptive/semantic (best places, recommendations)
- {hybrid}: Complex queries needing both
- {direct}: General knowledge (weather, culture, visas)

For bookings: use {hybrid}/{direct} and decline politely.

JSON: {{"strategy": "...", "reasoning": "...", "confidence": 0.0-1.0}}"""

def build_understanding_prompt(query: str, history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build understanding prompt."""
    msgs = [{"role": "system", "content": UNDERSTAND_SYS}]
    if history:
        msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": f"Query: {query}"})
    return msgs

def build_routing_prompt(query: str, strategies: Dict[str, str]) -> List[Dict]:
    """Build routing prompt."""
    sys = ROUTING_SYS.format(**strategies)
    return [{"role": "system", "content": sys}, {"role": "user", "content": f"Query: {query}"}]

def build_prompt(user_query: str, matches: List[Dict], graph_facts: List[Dict], 
                 filters: Optional[Dict] = None, history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build RAG prompt."""
    parts = []
    
    # Vector results
    if matches:
        vec = [f"- {m.get('metadata',{}).get('name','Unknown')}: {m.get('metadata',{}).get('description','')[:150]}" 
               for m in matches[:5]]
        parts.append("Relevant places:\n" + "\n".join(vec))
    
    # Graph facts
    if graph_facts:
        graph = [f"- {f['target_name']} ({f['target_type']}) is {f['rel']} {f['source']}" 
                 for f in graph_facts[:15]]
        parts.append("Nearby/connected:\n" + "\n".join(graph))
    
    content = f"Query: {user_query}\n\n" + "\n\n".join(parts)
    content += "\n\nAnswer the question. For itineraries, suggest 2-3 places."
    
    msgs = [{"role": "system", "content": RAG_SYS}]
    if history:
        msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": content})
    return msgs

def build_direct_prompt(user_query: str, history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build direct prompt."""
    msgs = [{"role": "system", "content": DIRECT_SYS}]
    if history:
        msgs.extend(history[-6:])
    msgs.append({"role": "user", "content": user_query})
    return msgs