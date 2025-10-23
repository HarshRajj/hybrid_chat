from typing import List, Dict
from openai import OpenAI
from src import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

def search_summary(matches: List[Dict], model: str = None) -> str:
    """Generate a concise 2-3 line summary of top matches."""
    if not matches:
        return "No semantic matches found."
    
    lines = []
    for m in matches[:6]:
        meta = m.get("metadata", {}) or {}
        lines.append(f"{meta.get('name', '(unknown)')} — {meta.get('type', '')} — {meta.get('city', '')}")
    
    model = model or config.CHAT_MODEL or "gpt-4o-mini"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a concise summarizer. Produce a 2-3 line summary."},
            {"role": "user", "content": "Items:\n" + "\n".join(lines)}
        ],
        max_tokens=100,
        temperature=0.2
    )
    return resp.choices[0].message.content.strip()

def build_prompt(user_query: str, matches: List[Dict], graph_facts: List[Dict]) -> List[Dict]:
    """Build chat prompt combining vector matches and graph context."""
    system = (
        "You are a helpful travel assistant. Use the provided semantic search results "
        "and graph facts to answer the user's query briefly and concisely. "
        "Cite node IDs when referencing specific places."
    )
    
    # Vector context
    vec_lines = []
    for m in matches[:8]:
        meta = m.get("metadata", {}) or {}
        vec_lines.append(
            f"- id: {m.get('id')}, name: {meta.get('name', '')}, "
            f"type: {meta.get('type', '')}, city: {meta.get('city', '')}, "
            f"score: {m.get('score', 0):.3f}"
        )
    
    # Graph context
    graph_lines = [
        f"- ({f['source']}) -[{f['rel']}]-> ({f['target_id']}) {f['target_name']}: {f['target_desc']}"
        for f in graph_facts[:20]
    ]
    
    summary = search_summary(matches)
    
    user_content = (
        f"User query: {user_query}\n\n"
        f"Semantic summary:\n{summary}\n\n"
        f"Top matches (detailed):\n" + "\n".join(vec_lines) + "\n\n"
        f"Graph facts:\n" + "\n".join(graph_lines) + "\n\n"
        "Answer the user's question. Suggest 2-3 concrete itinerary steps and cite node IDs."
    )
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content}
    ]