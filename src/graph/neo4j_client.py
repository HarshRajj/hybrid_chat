import time
from typing import List, Dict, Any, Optional, TypedDict
from neo4j import GraphDatabase, Driver
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from src import config # Assuming config.py holds NEO4J_URI, etc.

# --- Type Alias for Clarity ---
class GraphFact(TypedDict):
    source: str
    rel: str
    target: str
    description: str
# ------------------------------

class Neo4jClient:
    """
    Robust wrapper around Neo4j for graph operations, utilizing automatic retries
    for transient network or database errors.
    """
    
    def __init__(self):
        """Initializes the connection and verifies connectivity."""
        try:
            self.driver: Driver = GraphDatabase.driver(
                config.NEO4J_URI,
                auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
            )
            self._check_connection()
            print("✓ Neo4j Driver initialized and connection verified.")
        except Exception as e:
            # Catch potential authentication or connection errors
            raise ConnectionError(f"Failed to initialize Neo4j connection: {e}")
    
    def _check_connection(self):
        """Simple query to verify connection is alive."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1").single()
        except Exception as e:
            self.driver.close()
            raise ConnectionError(f"Neo4j connection health check failed: {e}")

    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            print("✓ Neo4j connection closed.")
    
    # --- Core Execution with Retry Logic ---
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _execute_read_query(self, query: str, parameters: Dict[str, Any]) -> List[Dict]:
        """Internal method to execute a read query with retry logic."""
        with self.driver.session() as session:
            result = session.run(query, parameters)
            # Fetch all records and return them as a list of dictionaries
            return [record.data() for record in result]
    
    # --- RAG Specific Query Method ---

    def search_by_topic(self, topic: str, limit: int = 10) -> List[GraphFact]:
        """
        Searches the graph for facts related to a given topic (e.g., a place or itinerary name).
        This is the primary method used in a RAG system.
        """
        # Use a case-insensitive search (toLower) on relevant node properties (like 'name').
        # This is more robust than requiring an exact ID match.
        query = (
            "MATCH (n)-[r]-(m) "
            "WHERE toLower(n.name) CONTAINS toLower($topic) "
            "   OR toLower(m.name) CONTAINS toLower($topic) "
            "RETURN "
            "   n.name AS source_name, "
            "   type(r) AS relationship, "
            "   m.name AS target_name, "
            "   m.description AS target_description "
            "LIMIT $limit"
        )
        params = {"topic": topic, "limit": limit}

        records: List[GraphFact] = []
        try:
            raw_records = self._execute_read_query(query, params)
            for record in raw_records:
                records.append({
                    "source": record.get("source_name", "Unknown"),
                    "rel": record.get("relationship", "RELATES_TO"),
                    "target": record.get("target_name", "Unknown"),
                    "target_type": record.get("target_type", "Unknown"),
                    "description": record.get("target_description", "")
                })
        except RetryError as e:
            # If all retries fail, log a warning and return an empty list.
            print(f"⚠️ Warning: Neo4j read query failed after multiple retries for topic '{topic}'. Skipping.")
        except Exception as e:
            print(f"❌ Error executing search_by_topic for '{topic}': {e}")
            
        return records

    # --- Batch Fact Fetching (Parallelism) ---

    def search_by_topic_batch(self, topics: List[str], limit: int = 5, max_workers: int = 4) -> List[GraphFact]:
        """
        Fetches facts for multiple topics in parallel using a ThreadPoolExecutor.
        Each individual search benefits from the internal retry logic.
        """
        all_facts: List[GraphFact] = []
        
        # Use ThreadPoolExecutor for I/O bound parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit search_by_topic for each topic
            futures = {executor.submit(self.search_by_topic, topic, limit): topic for topic in topics}
            
            for future in as_completed(futures):
                topic = futures[future]
                try:
                    # Collect the results, which are already in the GraphFact format
                    all_facts.extend(future.result())
                except Exception as e:
                    # Log the failure but don't stop the overall batch process
                    print(f"⚠️ Batch search failed for topic '{topic}'. Error: {e}")
                    continue
                    
        return all_facts
