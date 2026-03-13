"""Neo4j knowledge graph for Crystal Mancer.

Stores the crystal ↔ synthesis ↔ performance knowledge graph with full
Cypher query support.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Neo4j connection defaults
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "crystalmancer")


class KnowledgeGraph:
    """Neo4j-backed knowledge graph for catalysis data."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
    ):
        from neo4j import GraphDatabase
        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Connected to Neo4j at %s", uri)

    def close(self) -> None:
        self._driver.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ── Schema Setup ──────────────────────────────────────────────────────

    def create_schema(self) -> None:
        """Create indexes and constraints for the knowledge graph."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Crystal) REQUIRE c.cif_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.doi IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Element) REQUIRE e.symbol IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sg:SpaceGroup) REQUIRE sg.number IS UNIQUE",
        ]
        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (c:Crystal) ON (c.composition)",
            "CREATE INDEX IF NOT EXISTS FOR (sm:SynthesisMethod) ON (sm.name)",
            "CREATE INDEX IF NOT EXISTS FOR (a:Application) ON (a.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Paper) ON (p.title)",
        ]
        with self._driver.session() as session:
            for stmt in constraints + indexes:
                session.run(stmt)
        logger.info("Knowledge graph schema created.")

    # ── Node Creation ─────────────────────────────────────────────────────

    def upsert_crystal(self, record: dict[str, Any]) -> None:
        """Insert or update a Crystal node from a Phase 1 JSON record."""
        cypher = """
        MERGE (c:Crystal {cif_id: $cif_id})
        SET c.composition = $composition,
            c.spacegroup = $spacegroup,
            c.spacegroup_number = $spacegroup_number,
            c.cif_path = $cif_path
        WITH c

        // Space group node
        MERGE (sg:SpaceGroup {number: $spacegroup_number})
        SET sg.name = $spacegroup
        MERGE (c)-[:HAS_SPACEGROUP]->(sg)

        // Element nodes
        WITH c
        UNWIND $elements AS el
        MERGE (e:Element {symbol: el})
        MERGE (c)-[:CONTAINS_ELEMENT]->(e)
        """
        # Extract element symbols from composition
        import re
        composition = record.get("composition", "")
        elements = re.findall(r"[A-Z][a-z]?", composition)

        with self._driver.session() as session:
            session.run(cypher, {
                "cif_id": record["cif_id"],
                "composition": composition,
                "spacegroup": record.get("spacegroup", ""),
                "spacegroup_number": record.get("spacegroup_number", 0),
                "cif_path": record.get("cif_path", ""),
                "elements": list(set(elements)),
            })

    def upsert_paper(self, cif_id: str, paper: dict[str, Any]) -> None:
        """Insert a Paper node and link it to a Crystal."""
        cypher = """
        MATCH (c:Crystal {cif_id: $cif_id})

        // Paper node
        MERGE (p:Paper {doi: $doi})
        SET p.title = $title,
            p.year = $year
        MERGE (c)-[:REPORTED_IN]->(p)

        // Synthesis method
        WITH c, p
        MERGE (sm:SynthesisMethod {name: $synthesis_method})
        MERGE (c)-[:SYNTHESIZED_BY]->(sm)
        MERGE (p)-[:DESCRIBES_SYNTHESIS]->(sm)

        // Application
        WITH c, p
        MERGE (a:Application {name: $application})
        MERGE (c)-[:USED_FOR]->(a)
        MERGE (p)-[:DESCRIBES_APPLICATION]->(a)
        """
        doi = paper.get("doi")
        if not doi:
            return  # Can't create Paper node without DOI

        with self._driver.session() as session:
            session.run(cypher, {
                "cif_id": cif_id,
                "doi": doi,
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "synthesis_method": paper.get("synthesis_method", "other"),
                "application": paper.get("application", "other"),
            })

        # Performance metrics as properties on ACHIEVES relationship
        perf = paper.get("performance", {})
        perf_data = {k: v for k, v in perf.items() if v is not None}
        if perf_data:
            perf_cypher = """
            MATCH (c:Crystal {cif_id: $cif_id})
            MATCH (a:Application {name: $application})
            MERGE (c)-[r:ACHIEVES]->(a)
            SET r += $perf_data
            """
            with self._driver.session() as session:
                session.run(perf_cypher, {
                    "cif_id": cif_id,
                    "application": paper.get("application", "other"),
                    "perf_data": perf_data,
                })

    # ── Bulk Import ───────────────────────────────────────────────────────

    def import_records(self, records: list[dict[str, Any]]) -> int:
        """Bulk import Phase 1 JSON records into the knowledge graph.

        Returns the number of records imported.
        """
        count = 0
        for record in records:
            self.upsert_crystal(record)
            for paper in record.get("papers", []):
                self.upsert_paper(record["cif_id"], paper)
            count += 1
        logger.info("Imported %d records into knowledge graph.", count)
        return count

    # ── Basic Queries ─────────────────────────────────────────────────────

    def get_crystal(self, cif_id: str) -> dict | None:
        """Retrieve a crystal node by CIF ID."""
        cypher = "MATCH (c:Crystal {cif_id: $cif_id}) RETURN c"
        with self._driver.session() as session:
            result = session.run(cypher, {"cif_id": cif_id})
            record = result.single()
            return dict(record["c"]) if record else None

    def count_nodes(self) -> dict[str, int]:
        """Count nodes by label."""
        cypher = """
        CALL db.labels() YIELD label
        CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) AS count', {}) YIELD value
        RETURN label, value.count AS count
        """
        # Simpler version without APOC:
        simple_cypher = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        ORDER BY count DESC
        """
        with self._driver.session() as session:
            result = session.run(simple_cypher)
            return {record["label"]: record["count"] for record in result}

    def run_cypher(self, query: str, params: dict | None = None) -> list[dict]:
        """Execute an arbitrary Cypher query and return results."""
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [dict(record) for record in result]
