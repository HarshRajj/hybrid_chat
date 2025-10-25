from typing import List, Dict, Optional
from src import config

def build_understanding_prompt(query: str, history: Optional[List[Dict]] = None) -> List[Dict]:
    system = """You are a highly reliable and strict query analyzer for a Vietnam travel assistant.

            **TASK:** Analyze the user's latest query, using conversation history for context, and output a JSON object to classify 
            its actionability and scope. If the user only greets with his name, consider the query NOT actionable, just respond with 
            a greeting and not retrieve any data.

            **CORE DIRECTIVE: MAXIMAL LENIENCY.** ALWAYS default to proceeding. Only block or ask for clarification under 
            the most extreme circumstances defined below.

            **RULES & CONTEXT:**
            1.  **Use History:** Always check conversation history to understand partial or ambiguous queries 
                (e.g., "4 days," "best," "yes," "itinerary").
            2.  **OUT OF SCOPE (is_out_of_scope=true):** Only if the query explicitly names a **non-Vietnam destination**
                (e.g., "Paris," "Thailand," "London").
            3.  **CLARIFICATION (needs_clarification=true):** Only if the query is **completely unintelligible** 
                (e.g., "h", "ys", single random letters) AND there is **NO useful conversation history**. 
                Greetings ("hi") are actionable if history provides context, otherwise they need clarification.
            4.  **PROCEED (Default):** Proceed with all other queries. If actionable intent exists, capture it in `intent`.

            **JSON Output Format:**
            {
                "is_out_of_scope": boolean,         // true ONLY if explicitly non-Vietnam
                "needs_clarification": boolean,     // true ONLY if completely unintelligible AND no history
                "clarification_message": "string",  // (If needs_clarification=true) Request for user input.
                "intent": "string",                 // The derived user goal (e.g., "find hotels in Hanoi", "plan 4 day trip", "confirm previous intent"). Set to "out_of_scope" if Rule 2 applies.
                "confidence": 0.0-1.0
            }"""
    
    messages = [{"role": "system", "content": system}]
    
    # Add conversation history for context (if provided)
    if history:
        messages.extend(history)
    
    # Add current query
    messages.append({"role": "user", "content": f"Query: {query}"})
    
    return messages

def build_routing_prompt(query: str, strategies: Dict[str, str]) -> List[Dict]:
    """Build prompt for LLM-based routing (fallback)."""
    system = f"""You are a query intent classifier for a Vietnam travel system.
Classify the user query into ONE of these strategies:
- {strategies['graph']}: Relational/proximity queries (near, connected, list of)
- {strategies['vector']}: Descriptive/semantic queries (best places, recommendations)
- {strategies['hybrid']}: Complex queries needing both vector + graph
- {strategies['direct']}: General knowledge (weather, culture, visas)

Note: For booking/reservation requests, use {strategies['hybrid']} or {strategies['direct']} 
and the system will politely decline while still providing helpful recommendations.

Respond in JSON: {{"strategy": "...", "reasoning": "...", "confidence": 0.0-1.0}}"""
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Query: {query}"}
    ]

def build_prompt(user_query: str, matches: List[Dict], graph_facts: List[Dict], 
                 filters: Optional[Dict] = None, history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build prompt for hybrid strategy (vector + graph) with optional conversation history."""
    system = (
        "You are a helpful Vietnam travel assistant. Answer concisely based on the "
        "search results and connections provided. Cite place names when helpful.\n\n"
        "Important: I cannot make bookings, reservations, or access real-time pricing/availability. "
        "If asked to book or reserve, politely decline but still provide helpful recommendations "
        "and information about the places they might want to visit.\n\n"
        "Format: Use plain text only. Do NOT use markdown formatting (no **, __, ##, etc.). "
        "Use simple punctuation and numbered lists (1. 2. 3.) if needed."
    )
    
    # Vector results
    vec_lines = []
    for m in matches[:5]:
        meta = m.get("metadata", {}) or {}
        vec_lines.append(
            f"- {meta.get('name', 'Unknown')}: {meta.get('description', '')[:150]}"
        )
    
    # Graph connections
    graph_lines = []
    for f in graph_facts[:15]:
        graph_lines.append(
            f"- {f['target_name']} ({f['target_type']}) is {f['rel']} {f['source']}"
        )
    
    user_content = (
        f"Query: {user_query}\n\n"
        f"Relevant places:\n" + "\n".join(vec_lines) + "\n\n"
        f"Nearby/connected:\n" + "\n".join(graph_lines) + "\n\n"
        "Answer the user's question. For itineraries, suggest 2-3 specific places."
    )
    
    messages = [{"role": "system", "content": system}]
    
    # Add conversation history if provided (for context)
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_content})
    
    return messages

def build_direct_prompt(user_query: str, history: Optional[List[Dict]] = None) -> List[Dict]:
    """Build prompt for direct LLM (no retrieval) with optional conversation history."""
    system = (
        "You are a helpful Vietnam travel assistant. Answer general questions about "
        "Vietnam travel, weather, culture, visas, etc. Be concise and practical.\n\n"
        "Important: I cannot make bookings, reservations, or access real-time pricing/availability. "
        "If asked to book or reserve, politely decline but still provide helpful information "
        "and recommendations for planning their trip.\n\n"
        "Format: Use plain text only. Do NOT use markdown formatting (no **, __, ##, etc.). "
        "Use simple punctuation and numbered lists (1. 2. 3.) if needed."
    )
    
    messages = [{"role": "system", "content": system}]
    
    # Add conversation history if provided (for context)
    if history:
        messages.extend(history)
    
    messages.append({"role": "user", "content": user_query})
    
    return messages