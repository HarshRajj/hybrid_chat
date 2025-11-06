"""
Test script to verify out-of-scope detection works correctly.
Run this to test that non-travel queries are properly rejected.
"""

import asyncio
from src.agent import HybridRunner

async def test_out_of_scope():
    print("="*70)
    print("Testing Out-of-Scope Query Detection")
    print("="*70)
    
    runner = HybridRunner()
    
    # Test cases that should be marked out-of-scope
    out_of_scope_queries = [
        "how to make tea",
        "give me a recipe for pasta",
        "what is 2 + 2",
        "tell me about Paris",
        "how to code in Python",
        "movie recommendations",
        "what's the weather in London",
        "solve this math problem: x^2 + 5x + 6 = 0"
    ]
    
    # Test cases that should proceed (in-scope)
    in_scope_queries = [
        "best hotels in Hanoi",
        "4 day itinerary for Vietnam",
        "what's the weather like in Ho Chi Minh City",
        "Vietnamese tea ceremony",
        "traditional Vietnamese food recipes",
        "how to get around Hanoi"
    ]
    
    print("\n🚫 Testing OUT-OF-SCOPE queries (should be rejected):")
    print("-"*70)
    for query in out_of_scope_queries:
        print(f"\nQuery: '{query}'")
        response = await runner.query(query, explain=False)
        is_rejected = "specifically designed" in response or "Vietnam travel" in response
        status = "✅ REJECTED" if is_rejected else "❌ FAILED (should reject)"
        print(f"{status}")
        print(f"Response: {response[:100]}...")
    
    print("\n" + "="*70)
    print("\n✅ Testing IN-SCOPE queries (should proceed):")
    print("-"*70)
    for query in in_scope_queries:
        print(f"\nQuery: '{query}'")
        response = await runner.query(query, explain=False)
        is_rejected = "specifically designed" in response or "Vietnam travel" in response
        status = "✅ PROCEEDED" if not is_rejected else "❌ FAILED (should proceed)"
        print(f"{status}")
        print(f"Response: {response[:100]}...")
    
    print("\n" + "="*70)
    print("\n🔄 Testing PERSISTENT REJECTION (repeat same off-topic query 3 times):")
    print("-"*70)
    
    # Test persistent rejection
    test_query = "how to make tea"
    for attempt in range(1, 4):
        print(f"\nAttempt {attempt}: '{test_query}'")
        response = await runner.query(test_query, explain=False)
        is_rejected = "specifically designed" in response or "Vietnam travel" in response or "only assist" in response
        status = "✅ REJECTED" if is_rejected else f"❌ FAILED (should still reject on attempt {attempt})"
        print(f"{status}")
        print(f"Response: {response[:150]}...")
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_out_of_scope())
