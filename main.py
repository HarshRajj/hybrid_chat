from src.agent import HybridRunner
from src import config
import time
import sys
import asyncio

async def interactive_chat(streaming: bool = False):
    """Interactive CLI for hybrid travel assistant with intelligent routing."""
    print("\n" + "="*60)
    print("🌍 Vietnam Travel Assistant")
    print("="*60)
    print("Ask me anything about traveling in Vietnam!")
    print("I can help with hotels, restaurants, attractions, itineraries, and more.")
    print("\nCommands:")
    print("  - Type your question to get travel advice")
    print("  - Type 'stats' to see response time statistics")
    print("  - Type 'cache' to see cache statistics")
    print("  - Type 'exit' or 'quit' to end the session")
    
    if streaming:
        print("\n⚡ Streaming mode enabled - responses will appear in real-time!")
    
    print()
    
    runner = HybridRunner(top_k=5, enable_streaming=streaming)
    
    try:
        while True:
            query = input("\n📍 You: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye! Happy travels!")
                # Show final statistics
                runner.print_timing_summary()
                break
            
            if query.lower() == "stats":
                runner.print_timing_summary()
                continue
            
            if query.lower() == "cache":
                print("\n" + "="*60)
                print("💾 CACHE STATISTICS")
                print("="*60)
                stats = runner.get_performance_stats()
                
                print("\n🔹 Response Cache:")
                resp_cache = stats.get("response_cache", {})
                print(f"   Cached responses: {resp_cache.get('response_cache_size', 0)}")
                print(f"   Understanding cache: {resp_cache.get('understanding_cache_size', 0)}")
                
                print("\n🔹 Embedding Cache:")
                emb_stats = stats.get("embedding_cache", {})
                print(f"   Hits: {emb_stats.get('cache_hits', 0)}")
                print(f"   Misses: {emb_stats.get('cache_misses', 0)}")
                print(f"   Hit Rate: {emb_stats.get('hit_rate', '0%')}")
                
                print("\n🔹 Vector DB Cache:")
                vec_stats = stats.get("vector_cache", {})
                print(f"   Size: {vec_stats.get('size', 0)}")
                print(f"   Hits: {vec_stats.get('hits', 0)}")
                print(f"   Misses: {vec_stats.get('misses', 0)}")
                print(f"   Hit Rate: {vec_stats.get('hit_rate', '0%')}")
                
                print("\n🔹 Graph DB Cache:")
                graph_stats = stats.get("graph_cache", {})
                print(f"   Size: {graph_stats.get('size', 0)}")
                print(f"   Hits: {graph_stats.get('hits', 0)}")
                print(f"   Misses: {graph_stats.get('misses', 0)}")
                print(f"   Hit Rate: {graph_stats.get('hit_rate', '0%')}")
                
                print("="*60)
                continue
            
            print()
            try:
                # Query with explain mode to show routing decisions (now async)
                answer = await runner.query(query, explain=True)
                
                # Only print response if not streaming (streaming prints inline)
                if not streaming:
                    print("\n" + "="*60)
                    print("🤖 Assistant:")
                    print("="*60)
                    print(answer)
                    print("="*60)
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    finally:
        runner.close()

if __name__ == "__main__":
    config.validate_config()
    
    # Check for streaming flag
    streaming = "--stream" in sys.argv or "-s" in sys.argv
    
    if streaming:
        print("Starting in streaming mode...")
    
    asyncio.run(interactive_chat(streaming=streaming))