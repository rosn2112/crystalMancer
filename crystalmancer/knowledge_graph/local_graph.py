"""Lightweight graph database using NetworkX + SQLite.

A FREE, zero-dependency-setup alternative to Neo4j. No Docker, no server,
no paid tier — just pure Python. Uses NetworkX for graph traversal and
SQLite for persistence. Drop-in compatible with the KnowledgeGraph API.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import networkx as nx

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


class LocalKnowledgeGraph:
    """NetworkX + SQLite knowledge graph — fully local, zero setup.

    This replaces Neo4j when you don't want to run Docker.
    It stores the graph in memory (NetworkX) and persists to SQLite.
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = db_path or DEFAULT_OUTPUT_DIR / "knowledge_graph.db"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.G = nx.MultiDiGraph()
        self._init_db()
        self._load_from_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS edges (
                src TEXT NOT NULL,
                dst TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (src) REFERENCES nodes(id),
                FOREIGN KEY (dst) REFERENCES nodes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type);
            CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type);
            CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
            CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        for row in cursor.execute("SELECT id, node_type, properties FROM nodes"):
            node_id, node_type, props_str = row
            props = json.loads(props_str)
            self.G.add_node(node_id, node_type=node_type, **props)

        for row in cursor.execute("SELECT src, dst, edge_type, properties FROM edges"):
            src, dst, edge_type, props_str = row
            props = json.loads(props_str)
            self.G.add_edge(src, dst, edge_type=edge_type, **props)

        conn.close()
        logger.info("Loaded graph: %d nodes, %d edges.", self.G.number_of_nodes(), self.G.number_of_edges())

    def save(self) -> None:
        """Persist the in-memory graph to SQLite."""
        conn = sqlite3.connect(str(self._db_path))
        cursor = conn.cursor()

        # Upsert nodes
        for node_id, attrs in self.G.nodes(data=True):
            node_type = attrs.pop("node_type", "Unknown")
            cursor.execute(
                "INSERT OR REPLACE INTO nodes (id, node_type, properties) VALUES (?, ?, ?)",
                (node_id, node_type, json.dumps(attrs, default=str)),
            )
            attrs["node_type"] = node_type

        # Re-insert edges (clear and rebuild for simplicity)
        cursor.execute("DELETE FROM edges")
        for src, dst, attrs in self.G.edges(data=True):
            edge_type = attrs.get("edge_type", "RELATED")
            props = {k: v for k, v in attrs.items() if k != "edge_type"}
            cursor.execute(
                "INSERT INTO edges (src, dst, edge_type, properties) VALUES (?, ?, ?, ?)",
                (src, dst, edge_type, json.dumps(props, default=str)),
            )

        conn.commit()
        conn.close()
        logger.info("Saved graph: %d nodes, %d edges.", self.G.number_of_nodes(), self.G.number_of_edges())

    # ── Node Creation ─────────────────────────────────────────────────────

    def upsert_crystal(self, record: dict[str, Any]) -> None:
        """Import a Crystal node from a Phase 1 JSON record."""
        cif_id = record["cif_id"]
        composition = record.get("composition", "")

        # Crystal node
        self.G.add_node(f"crystal:{cif_id}", node_type="Crystal",
                        cif_id=cif_id, composition=composition,
                        spacegroup=record.get("spacegroup", ""),
                        spacegroup_number=record.get("spacegroup_number", 0))

        # Element nodes
        elements = set(re.findall(r"[A-Z][a-z]?", composition))
        for el in elements:
            self.G.add_node(f"element:{el}", node_type="Element", symbol=el)
            self.G.add_edge(f"crystal:{cif_id}", f"element:{el}", edge_type="CONTAINS_ELEMENT")

        # Space group node
        sg = record.get("spacegroup", "")
        sg_num = record.get("spacegroup_number", 0)
        if sg_num:
            self.G.add_node(f"sg:{sg_num}", node_type="SpaceGroup", number=sg_num, name=sg)
            self.G.add_edge(f"crystal:{cif_id}", f"sg:{sg_num}", edge_type="HAS_SPACEGROUP")

    def upsert_paper(self, cif_id: str, paper: dict[str, Any]) -> None:
        """Import a Paper + extraction results linked to a Crystal."""
        doi = paper.get("doi")
        if not doi:
            return

        # Paper node
        self.G.add_node(f"paper:{doi}", node_type="Paper",
                        doi=doi, title=paper.get("title", ""),
                        year=paper.get("year"), abstract=paper.get("abstract", ""))
        self.G.add_edge(f"crystal:{cif_id}", f"paper:{doi}", edge_type="REPORTED_IN")

        # Synthesis method
        synth = paper.get("synthesis_method", "other")
        self.G.add_node(f"synth:{synth}", node_type="SynthesisMethod", name=synth)
        self.G.add_edge(f"crystal:{cif_id}", f"synth:{synth}", edge_type="SYNTHESIZED_BY")

        # Application
        app = paper.get("application", "other")
        self.G.add_node(f"app:{app}", node_type="Application", name=app)
        self.G.add_edge(f"crystal:{cif_id}", f"app:{app}", edge_type="USED_FOR")

        # Performance as edge properties
        perf = paper.get("performance", {})
        perf_data = {k: v for k, v in perf.items() if v is not None}
        if perf_data:
            self.G.add_edge(f"crystal:{cif_id}", f"app:{app}",
                            edge_type="ACHIEVES", **perf_data)

    def import_records(self, records: list[dict[str, Any]]) -> int:
        """Bulk import Phase 1 JSON records."""
        count = 0
        for record in records:
            self.upsert_crystal(record)
            for paper in record.get("papers", []):
                self.upsert_paper(record["cif_id"], paper)
            count += 1
        self.save()
        logger.info("Imported %d records.", count)
        return count

    # ── Graph Queries (Python-based, same results as Cypher) ──────────────

    def find_crystals_for_application(
        self,
        application: str,
        max_overpotential_mV: float | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Find all crystals targeting a specific catalytic application."""
        app_node = f"app:{application}"
        if app_node not in self.G:
            return []

        results = []
        for src, _, data in self.G.in_edges(app_node, data=True):
            if data.get("edge_type") not in ("ACHIEVES", "USED_FOR"):
                continue
            if not src.startswith("crystal:"):
                continue

            crystal = dict(self.G.nodes[src])
            crystal["cif_id"] = crystal.get("cif_id", src.replace("crystal:", ""))

            # Get performance from ACHIEVES edges
            for _, _, edata in self.G.edges(src, data=True):
                if edata.get("edge_type") == "ACHIEVES":
                    crystal.update({k: v for k, v in edata.items() if k != "edge_type"})

            if max_overpotential_mV and crystal.get("overpotential_mV"):
                if crystal["overpotential_mV"] > max_overpotential_mV:
                    continue

            results.append(crystal)

        return results[:limit]

    def find_synthesis_routes(self, composition: str) -> list[dict[str, Any]]:
        """Find all synthesis methods reported for a composition."""
        results = []
        for node, attrs in self.G.nodes(data=True):
            if attrs.get("node_type") != "Crystal" or attrs.get("composition") != composition:
                continue
            for _, dst, data in self.G.edges(node, data=True):
                if data.get("edge_type") == "SYNTHESIZED_BY":
                    synth_attrs = dict(self.G.nodes[dst])
                    results.append({"synthesis_method": synth_attrs.get("name", dst)})
        return results

    def find_similar_materials(self, cif_id: str, limit: int = 20) -> list[dict[str, Any]]:
        """Find materials sharing elements or space groups."""
        source = f"crystal:{cif_id}"
        if source not in self.G:
            return []

        # Get source's elements and space group
        source_neighbors = set()
        for _, dst, data in self.G.edges(source, data=True):
            if data.get("edge_type") in ("CONTAINS_ELEMENT", "HAS_SPACEGROUP"):
                source_neighbors.add(dst)

        # Find other crystals sharing those nodes
        similarity: dict[str, int] = {}
        for shared_node in source_neighbors:
            for src, _, data in self.G.in_edges(shared_node, data=True):
                if src == source or not src.startswith("crystal:"):
                    continue
                similarity[src] = similarity.get(src, 0) + 1

        # Sort by shared count and return
        results = []
        for crystal_id, shared_count in sorted(similarity.items(), key=lambda x: -x[1])[:limit]:
            attrs = dict(self.G.nodes[crystal_id])
            attrs["shared_count"] = shared_count
            results.append(attrs)

        return results

    def get_stats(self) -> dict[str, int]:
        """Get node/edge counts by type."""
        node_counts: dict[str, int] = {}
        for _, attrs in self.G.nodes(data=True):
            nt = attrs.get("node_type", "Unknown")
            node_counts[nt] = node_counts.get(nt, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, attrs in self.G.edges(data=True):
            et = attrs.get("edge_type", "Unknown")
            edge_counts[et] = edge_counts.get(et, 0) + 1

        return {"nodes": node_counts, "edges": edge_counts,
                "total_nodes": self.G.number_of_nodes(),
                "total_edges": self.G.number_of_edges()}

    def close(self) -> None:
        self.save()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
