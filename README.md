## Vietnam Travel Assistant (Hybrid RAG)

A smart travel assistant for planning trips to Vietnam. Combines semantic search (Pinecone), graph traversal (Neo4j), and GPT-4 for fast, helpful answers.

### Key Features

- **Hybrid RAG Architecture**: Combines vector search (Pinecone) and graph traversal (Neo4j) for comprehensive answers
- **Intelligent Query Routing**: Automatically selects the best strategy (hybrid/graph/direct) for each query
- **Async/Await Performance**: Parallel I/O operations for 40-60% faster responses
- **Multi-Level Caching**: Instant responses for repeat queries (95% faster)
- **Conversational Context**: Maintains conversation history for follow-up questions
- **Streaming Output**: Real-time word-by-word responses
- **Smart Query Understanding**: LLM-powered intent detection handles greetings, out-of-scope queries, and clarifications

---

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**
   - Create a `.env` file in the project root:
     ```env
     # Required
     OPENAI_API_KEY=sk-your-key-here
     PINECONE_API_KEY=your-pinecone-key
     NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
     NEO4J_USER=neo4j
     NEO4J_PASSWORD=your-password
     
     # Optional: Use Cerebras for 10x faster inference
     USE_CEREBRAS=true
     CEREBRAS_API_KEY=csk-your-cerebras-key
     CEREBRAS_MODEL=llama3.1-8b
     ```
   - **For Neo4j Aura**: Use `neo4j+s://` URI from your Aura instance connection details
   - **For Cerebras**: Get free API key at https://cloud.cerebras.ai/ (1,800 tokens/s!)



3. **Load data (first-time setup only)**
   - You must run these scripts once to load the travel data into Neo4j and Pinecone:
     ```bash
     python scripts/load_to_neo4j.py
     python scripts/pinecone_upload_scaled.py
     ```

4. **Run the assistant**
   ```bash
   python main.py
   # or for streaming output:
   python main.py --stream
   ```

---

## Example Queries

- "best hotels in Hanoi"
- "4 day itinerary for Vietnam"
- "restaurants near Hoan Kiem Lake"
- "hi" (instant greeting)

---

## Project Structure

```
hybrid_chat/
├── src/
│   ├── agent.py           # Main orchestrator
│   ├── router.py          # Query routing
│   ├── prompt.py          # System prompts
│   ├── config.py          # Environment config
│   ├── embeddings/        # Embedding logic
│   ├── vector/            # Pinecone client
│   └── graph/             # Neo4j client
├── data/
│   └── vietnam_travel_dataset.json
├── scripts/
│   ├── load_to_neo4j.py
│   └── pinecone_upload_scaled.py
├── main.py                # CLI interface
├── requirements.txt
├── .env                   # Your API keys (create this)
├── README.md
└── IMPROVEMENTS.md
```

---

## Troubleshooting: Neo4j Connection Errors

If you have trouble connecting to Neo4j, try the following:

**For Neo4j Aura (Cloud):**
- Use the connection URI from your Aura console (format: `neo4j+s://xxxxx.databases.neo4j.io`)
- Use the password you set when creating the instance
- Make sure your IP is allowlisted in Aura settings (or allow all IPs for testing)
- Verify your instance is running in the Aura console

**For Local Neo4j:**
- Make sure Neo4j is running (use Neo4j Desktop or `neo4j start` for server installs)
- Default URI is usually `bolt://localhost:7687` and user is `neo4j`
- If you changed the default password, update it in `.env`
- Ensure your firewall allows connections to port 7687
- If using Docker, make sure the container exposes port 7687 and is running

**General:**
- Check that your `NEO4J_URI`, `NEO4J_USER`, and `NEO4J_PASSWORD` in `.env` are correct
- Try connecting with the Neo4j Browser to verify credentials

If you still have issues, see the [Neo4j documentation](https://neo4j.com/docs/) or check your logs for error details.

---

**Built by [HarshRajj](https://github.com/HarshRajj)**
