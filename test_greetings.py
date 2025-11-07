"""
Test script to verify greeting detection works correctly.
"""

import asyncio
from src.agent import HybridRunner

async def test_greetings():
    print("="*70)
    print("Testing Greeting Detection")
    print("="*70)
    
    runner = HybridRunner()
    
    # Test cases that should be treated as greetings (NO retrieval)
    greeting_queries = [
        "hi",
        "hello",
        "hey there",
        "hi i'm harsh",
        "hello my name is john",
        "hi i m harsh raj and i am fine",
        "good morning, i'm sarah",
        "hey, call me mike"
    ]
    
    # Test cases that should proceed (travel intent)
    travel_queries = [
        "hi, i want to visit hanoi",
        "hello, can you recommend hotels in vietnam",
        "hey, planning a trip to sapa",
        "good morning, what are the best places to visit",
        "hi i'm john and i need help planning my vietnam trip"
    ]
    
    print("\n👋 Testing GREETING queries (should respond without retrieval):")
    print("-"*70)
    for query in greeting_queries:
        print(f"\nQuery: '{query}'")
        response = await runner.query(query, explain=False)
        
        # Check if it's a simple greeting response (no travel suggestions)
        has_suggestions = any(word in response.lower() for word in [
            "ha long", "hoi an", "sapa", "hanoi", "saigon", "hotel", "restaurant",
            "destination", "visit", "recommend", "itinerary", "day trip"
        ])
        
        status = "✅ GREETING" if not has_suggestions else "❌ FAILED (should not suggest places)"
        print(f"{status}")
        print(f"Response: {response}")
    
    print("\n" + "="*70)
    print("\n✈️  Testing TRAVEL queries (should retrieve and suggest):")
    print("-"*70)
    for query in travel_queries:
        print(f"\nQuery: '{query}'")
        response = await runner.query(query, explain=False)
        
        # Check if it provides travel suggestions
        has_suggestions = any(word in response.lower() for word in [
            "ha long", "hoi an", "sapa", "hanoi", "saigon", "hotel", "restaurant",
            "destination", "visit", "recommend", "place"
        ])
        
        status = "✅ TRAVEL RESPONSE" if has_suggestions else "❌ FAILED (should suggest places)"
        print(f"{status}")
        print(f"Response: {response[:200]}...")
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_greetings())
