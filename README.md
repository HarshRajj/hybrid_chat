# 🌍 Vietnam Travel Assistant - Hybrid RAG System

An intelligent travel assistant for Vietnam using **Hybrid RAG** (Retrieval-Augmented Generation) that combines vector search (Pinecone) and graph traversal (Neo4j) with smart query understanding.

## ⚡ Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Up Environment
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=sk-your-key-here
PINECONE_API_KEY=your-pinecone-key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

### 3. Load Data
```bash
# Load travel data to Neo4j
python scripts/load_to_neo4j.py

# Upload vectors to Pinecone
python scripts/pinecone_upload_scaled.py
```

### 4. Run the Assistant
```bash
python main.py
```

## 🎯 What It Does

- **Understands queries intelligently** - Only asks clarification when truly needed
- **Searches semantically** - Finds relevant hotels, attractions, restaurants using Pinecone
## 🎯 What It Does

- **Understands queries intelligently** - Only asks clarification when truly needed
- **Searches semantically** - Finds relevant hotels, attractions, restaurants using Pinecone
- **Explores relationships** - Discovers nearby places and connections using Neo4j
- **Generates smart responses** - Combines all context with GPT-4 for helpful answers
- **Lightning fast** - Multi-level caching + streaming responses

## ⚡ Performance Features

- **Async/Await Architecture**: True concurrent I/O for maximum speed (40-60% faster!)
- **Response Caching**: Instant answers for repeat queries (95% faster!)
- **Dynamic Token Sizing**: Optimizes LLM cost based on query complexity
- **Streaming Output**: Real-time response generation for better UX
- **Multi-level Caching**: Embeddings, vector search, graph queries all cached
- **Parallel Processing**: Concurrent vector + graph fetching with asyncio

See [ASYNC_ARCHITECTURE.md](ASYNC_ARCHITECTURE.md) for detailed async implementation.

## 📊 Example Queries

```
✅ "4 day romantic itinerary for Vietnam"
✅ "best beach hotels in Danang"
✅ "hotels in Vietnam"
✅ "restaurants near Hoan Kiem Lake"
✅ "what to see in Hue"
```

## 🏗️ Architecture

```
User Query
    ↓
Phase 1: Understanding (LLM analyzes query)
    ↓
Phase 2: Routing (graph/vector/hybrid/direct)
    ↓
Retrieval (Pinecone + Neo4j)
    ↓
Response Generation (GPT-4)
```

## 📁 Project Structure

```
hybrid_chat/
├── src/
│   ├── agent.py           # Main orchestrator (two-phase system)
│   ├── router.py          # Intelligent query routing
│   ├── prompt.py          # System prompts
│   ├── config.py          # Environment configuration
│   ├── embeddings/        # OpenAI embeddings with caching
│   ├── vector/            # Pinecone client
│   └── graph/             # Neo4j client
├── data/
│   └── vietnam_travel_dataset.json
├── scripts/
│   ├── load_to_neo4j.py
│   └── pinecone_upload_scaled.py
├── main.py                # Interactive CLI
├── requirements.txt
└── .env                   # Your API keys (create this)
```

## � Usage Modes

### Normal Mode
```bash
python main.py
```

### Streaming Mode (Real-time Output)
```bash
python main.py --stream
```

### Interactive Commands
- Type your question to get answers
- `stats` - View response time statistics
- `cache` - View cache performance
- `exit` - Exit (shows summary)

## �🔧 Technologies

- **OpenAI**: text-embedding-3-small (embeddings) + gpt-4o-mini (chat)
- **Pinecone**: Serverless vector database with caching
- **Neo4j**: Graph database for relationships with parallel queries
- **Python 3.x**: tenacity, tqdm, asyncio, concurrent.futures

## 📚 Documentation

- **IMPROVEMENTS.md** - Complete list of enhancements from original `hybrid_chat.py`
- **LLM_OPTIMIZATION.md** - Details on response caching, dynamic tokens, and streaming
- **TIMING_TRACKING.md** - Response time tracking and performance monitoring
- **PHASE1_SIMPLIFICATION.md** - Query understanding approach
- **hybrid_chat.py** - Original version for reference

## 🧪 Testing

```bash
# Test LLM optimizations (caching, tokens, streaming)
python test_llm_optimization.py

# Test timing tracking
python test_timing.py

# Test simplified Phase 1
python test_simplified_phase1.py
```

## 🧹 Cleanup (Optional)

To remove test files and redundant documentation:
```bash
# Run the cleanup script
.\cleanup.ps1
```

This keeps only essential files: README, IMPROVEMENTS, source code, and data.

## 🤝 Support

This is an internship assignment project demonstrating hybrid RAG architecture with intelligent query understanding.
