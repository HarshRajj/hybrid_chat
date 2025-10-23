import json
from neo4j import GraphDatabase
from tqdm import tqdm
from ..src import config
DATA_FILE = "vietnam_travel_dataset.json"

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def check_existing_data(tx):
    """Check if data already exists in the database."""
    result = tx.run("MATCH (n:Entity) RETURN count(n) AS count")
    record = result.single()
    return record["count"] if record else 0

def create_constraints(tx):
    # generic uniqueness constraint on id for node label Entity (we also add label specific types)
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")

def upsert_node(tx, node):
    # use label from node['type'] and always add :Entity label
    labels = [node.get("type","Unknown"), "Entity"]
    label_cypher = ":" + ":".join(labels)
    # keep a subset of properties to store (avoid storing huge nested objects)
    props = {k:v for k,v in node.items() if k not in ("connections",)}
    # set properties using parameters
    tx.run(
        f"MERGE (n{label_cypher} {{id: $id}}) "
        "SET n += $props",
        id=node["id"], props=props
    )

def create_relationship(tx, source_id, rel):
    # rel is like {"relation": "Located_In", "target": "city_hanoi"}
    rel_type = rel.get("relation", "RELATED_TO")
    target_id = rel.get("target")
    if not target_id:
        return
    # Create relationship if both nodes exist
    cypher = (
        "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "RETURN r"
    )
    tx.run(cypher, source_id=source_id, target_id=target_id)

def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    with driver.session() as session:
        # Check if data already exists
        existing_count = session.execute_read(check_existing_data)
        
        if existing_count > 0:
            print(f"\n⚠️  Database already contains {existing_count} nodes.")
            response = input("Do you want to:\n  [1] Skip upload (keep existing data)\n  [2] Add/Update new data (merge)\n  [3] Clear and reload all data\nChoice (1/2/3): ").strip()
            
            if response == "1":
                print("✓ Skipping upload. Existing data preserved.")
                return
            elif response == "3":
                print("🗑️  Clearing existing data...")
                session.execute_write(lambda tx: tx.run("MATCH (n) DETACH DELETE n"))
                print("✓ Database cleared.")
            else:
                print("📝 Merging with existing data...")
        
        # Create constraints
        session.execute_write(create_constraints)
        
        # Upsert all nodes
        for node in tqdm(nodes, desc="Creating nodes"):
            session.execute_write(upsert_node, node)

        # Create relationships
        for node in tqdm(nodes, desc="Creating relationships"):
            conns = node.get("connections", [])
            for rel in conns:
                session.execute_write(create_relationship, node["id"], rel)

    print("\n✓ Done loading into Neo4j.")

if __name__ == "__main__":
    main()
