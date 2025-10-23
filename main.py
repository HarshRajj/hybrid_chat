from src.agent import HybridRunner
from src import config

def interactive_chat():
    """Interactive CLI for hybrid travel assistant."""
    print("\n" + "="*60)
    print("🌍 Hybrid Travel Assistant")
    print("="*60)
    print("Type your travel question or 'exit' to quit.\n")
    
    runner = HybridRunner(top_k=5)
    
    try:
        while True:
            query = input("\n📍 Enter your travel question: ").strip()
            
            if not query or query.lower() in ("exit", "quit", "q"):
                print("\n👋 Goodbye!")
                break
            
            print("\n⏳ Thinking...\n")
            try:
                answer = runner.query(query)
                print("="*60)
                print("🤖 Assistant Answer:")
                print("="*60)
                print(answer)
                print("="*60)
            except Exception as e:
                print(f"❌ Error: {e}")
    
    finally:
        runner.close()

if __name__ == "__main__":
    config.validate_config()
    interactive_chat()