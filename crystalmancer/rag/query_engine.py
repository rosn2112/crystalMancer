"""Natural language → Cypher query engine.

Translates user questions about the knowledge graph into Cypher
queries using an LLM, executes them, and returns formatted results.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from crystalmancer.extraction.llm_client import chat_completion, extract_json

logger = logging.getLogger(__name__)

_NL2CYPHER_SYSTEM = """You are a Neo4j Cypher query expert for a materials science knowledge graph.

The graph has these node types:
- Crystal (cif_id, composition, spacegroup, spacegroup_number, cif_path)
- Paper (doi, title, year)
- Element (symbol)
- SpaceGroup (number, name)
- SynthesisMethod (name): hydrothermal, sol-gel, solid-state, sputtering, ALD, coprecipitation, etc.
- Application (name): OER, HER, CO2RR, ORR, NRR, photocatalysis, thermochemical, water-splitting

Relationships:
- (Crystal)-[:CONTAINS_ELEMENT]->(Element)
- (Crystal)-[:HAS_SPACEGROUP]->(SpaceGroup)
- (Crystal)-[:REPORTED_IN]->(Paper)
- (Crystal)-[:SYNTHESIZED_BY]->(SynthesisMethod)
- (Crystal)-[:USED_FOR]->(Application)
- (Crystal)-[:ACHIEVES {overpotential_mV, faradaic_efficiency_pct, tafel_slope_mV_dec, current_density_mA_cm2, stability_h}]->(Application)
- (Paper)-[:DESCRIBES_SYNTHESIS]->(SynthesisMethod)
- (Paper)-[:DESCRIBES_APPLICATION]->(Application)

Given a user question, return ONLY a JSON object:
{
  "cypher": "the Cypher query string",
  "explanation": "brief explanation of what the query does"
}

Rules:
- Always use LIMIT to prevent excessive results (default 25)
- Use OPTIONAL MATCH for properties that may not exist
- Return human-readable column names with AS aliases
- Handle case-insensitive matching where appropriate"""

_NL2CYPHER_USER = """Translate this question to a Cypher query:

"{question}"

Return only the JSON object with "cypher" and "explanation" keys."""


def question_to_cypher(
    question: str,
    model: str | None = None,
) -> dict[str, str]:
    """Convert a natural language question to a Cypher query.

    Returns
    -------
    dict
        {"cypher": "...", "explanation": "..."}
    """
    prompt = _NL2CYPHER_USER.format(question=question)
    result = extract_json(prompt, system_prompt=_NL2CYPHER_SYSTEM, model=model)
    return {
        "cypher": result.get("cypher", ""),
        "explanation": result.get("explanation", ""),
    }


def query_knowledge_graph(
    question: str,
    graph,  # KnowledgeGraph instance
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a question using the knowledge graph.

    Pipeline:
    1. Translate question → Cypher query (LLM)
    2. Execute Cypher on Neo4j
    3. Return structured results

    Parameters
    ----------
    question : str
        Natural language question.
    graph : KnowledgeGraph
        Connected Neo4j knowledge graph.
    model : str | None
        Override LLM model.

    Returns
    -------
    dict
        {
            "question": original question,
            "cypher": generated query,
            "explanation": what the query does,
            "results": list of result dicts,
            "num_results": count
        }
    """
    # Step 1: NL → Cypher
    query_info = question_to_cypher(question, model=model)
    cypher = query_info["cypher"]
    explanation = query_info["explanation"]

    logger.info("Generated Cypher: %s", cypher)
    logger.info("Explanation: %s", explanation)

    # Step 2: Execute
    try:
        results = graph.run_cypher(cypher)
    except Exception as exc:
        logger.error("Cypher execution failed: %s", exc)
        results = []

    return {
        "question": question,
        "cypher": cypher,
        "explanation": explanation,
        "results": results,
        "num_results": len(results),
    }
