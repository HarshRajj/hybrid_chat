# ...existing code...
from neo4j import GraphDatabase
from pyvis.network import Network
import networkx as nx
from src import config
import sys
import logging
from typing import List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NEO_BATCH = 500  # number of relationships to fetch / visualize

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def sanitize_labels(labels: Any) -> List[str]:
    if not labels:
        return ["Entity"]
    try:
        return [str(l) for l in labels]
    except Exception:
        return ["Entity"]

def short(text: Any, n: int = 80) -> str:
    s = "" if text is None else str(text)
    return s if len(s) <= n else s[: n - 3] + "..."

def test_connection() -> bool:
    try:
        with driver.session() as session:
            session.run("RETURN 1").single()
        logger.info("✓ Neo4j connection OK")
        return True
    except Exception as e:
        logger.error("Neo4j connection failed: %s", e)
        return False

def fetch_subgraph(tx, limit: int = 500):
    q = (
        "MATCH (a:Entity)-[r]->(b:Entity) "
        "RETURN a.id AS a_id, labels(a) AS a_labels, a.name AS a_name, "
        "b.id AS b_id, labels(b) AS b_labels, b.name AS b_name, type(r) AS rel "
        "LIMIT $limit"
    )
    return list(tx.run(q, limit=limit))

def build_pyvis(rows, output_html="neo4j_viz.html"):
    net = Network(height="900px", width="100%",notebook=False, directed=True)
    for rec in rows:
        a_id = str(rec["a_id"])
        b_id = str(rec["b_id"])
        a_name = short(rec.get("a_name") or a_id)
        b_name = short(rec.get("b_name") or b_id)
        a_labels = sanitize_labels(rec.get("a_labels"))
        b_labels = sanitize_labels(rec.get("b_labels"))
        rel = rec.get("rel", "RELATED_TO")

        a_label = f"{a_name}\n({', '.join(a_labels)})"
        b_label = f"{b_name}\n({', '.join(b_labels)})"

        net.add_node(a_id, label=a_label, title=a_name)
        net.add_node(b_id, label=b_label, title=b_name)
        net.add_edge(a_id, b_id, title=str(rel))

    net.show(output_html)
    logger.info("Saved visualization to %s", output_html)

def main(limit: int = NEO_BATCH, output: str = "neo4j_viz.html"):
    if not test_connection():
        logger.error("Aborting - cannot connect to Neo4j. Check config and network.")
        sys.exit(1)

    try:
        with driver.session() as session:
            rows = session.execute_read(fetch_subgraph, limit=limit)
        build_pyvis(rows, output_html=output)
    except Exception as e:
        logger.exception("Failed to build visualization: %s", e)
    finally:
        try:
            driver.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
