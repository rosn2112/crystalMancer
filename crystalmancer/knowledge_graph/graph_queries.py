"""Pre-built Cypher queries for the Crystal Mancer knowledge graph.

Each function returns a (cypher_query, params) tuple that can be
executed via KnowledgeGraph.run_cypher().
"""

from __future__ import annotations


def find_crystals_for_application(
    application: str,
    max_overpotential_mV: float | None = None,
    min_faradaic_efficiency_pct: float | None = None,
    limit: int = 50,
) -> tuple[str, dict]:
    """Find crystals used for a specific catalytic application.

    Optionally filter by performance thresholds.
    """
    where_clauses = []
    params: dict = {"application": application, "limit": limit}

    if max_overpotential_mV is not None:
        where_clauses.append("r.overpotential_mV <= $max_op")
        params["max_op"] = max_overpotential_mV

    if min_faradaic_efficiency_pct is not None:
        where_clauses.append("r.faradaic_efficiency_pct >= $min_fe")
        params["min_fe"] = min_faradaic_efficiency_pct

    where = " AND " + " AND ".join(where_clauses) if where_clauses else ""

    cypher = f"""
    MATCH (c:Crystal)-[r:ACHIEVES]->(a:Application {{name: $application}})
    WHERE true{where}
    OPTIONAL MATCH (c)-[:SYNTHESIZED_BY]->(sm:SynthesisMethod)
    RETURN c.cif_id AS cif_id,
           c.composition AS composition,
           c.spacegroup AS spacegroup,
           sm.name AS synthesis_method,
           r.overpotential_mV AS overpotential,
           r.faradaic_efficiency_pct AS faradaic_efficiency,
           r.tafel_slope_mV_dec AS tafel_slope,
           r.current_density_mA_cm2 AS current_density,
           r.stability_h AS stability
    ORDER BY r.overpotential_mV ASC
    LIMIT $limit
    """
    return cypher, params


def find_synthesis_routes(composition: str) -> tuple[str, dict]:
    """Find all synthesis routes reported for a given composition."""
    cypher = """
    MATCH (c:Crystal {composition: $composition})-[:SYNTHESIZED_BY]->(sm:SynthesisMethod)
    OPTIONAL MATCH (c)-[:REPORTED_IN]->(p:Paper)-[:DESCRIBES_SYNTHESIS]->(sm)
    RETURN sm.name AS synthesis_method,
           collect(DISTINCT p.doi) AS paper_dois,
           collect(DISTINCT p.title) AS paper_titles,
           count(DISTINCT p) AS paper_count
    ORDER BY paper_count DESC
    """
    return cypher, {"composition": composition}


def find_similar_materials(
    cif_id: str,
    hops: int = 2,
    limit: int = 20,
) -> tuple[str, dict]:
    """Find materials similar to a given crystal via shared elements/space groups.

    Traverses up to *hops* relationships through Element and SpaceGroup nodes.
    """
    cypher = """
    MATCH (source:Crystal {cif_id: $cif_id})
    MATCH path = (source)-[:CONTAINS_ELEMENT|HAS_SPACEGROUP*1..""" + str(hops * 2) + """]->(shared)<-[:CONTAINS_ELEMENT|HAS_SPACEGROUP*1..""" + str(hops * 2) + """]-(target:Crystal)
    WHERE source <> target
    WITH target, count(DISTINCT shared) AS shared_count
    OPTIONAL MATCH (target)-[:USED_FOR]->(a:Application)
    OPTIONAL MATCH (target)-[:SYNTHESIZED_BY]->(sm:SynthesisMethod)
    RETURN target.cif_id AS cif_id,
           target.composition AS composition,
           target.spacegroup AS spacegroup,
           shared_count,
           collect(DISTINCT a.name) AS applications,
           collect(DISTINCT sm.name) AS synthesis_methods
    ORDER BY shared_count DESC
    LIMIT $limit
    """
    return cypher, {"cif_id": cif_id, "limit": limit}


def find_top_performers(
    metric: str = "overpotential_mV",
    application: str | None = None,
    ascending: bool = True,
    limit: int = 20,
) -> tuple[str, dict]:
    """Find top-performing crystals ranked by a specific metric."""
    order = "ASC" if ascending else "DESC"
    app_filter = "AND a.name = $application" if application else ""
    params: dict = {"metric": metric, "limit": limit}
    if application:
        params["application"] = application

    cypher = f"""
    MATCH (c:Crystal)-[r:ACHIEVES]->(a:Application)
    WHERE r['{metric}'] IS NOT NULL {app_filter}
    OPTIONAL MATCH (c)-[:SYNTHESIZED_BY]->(sm:SynthesisMethod)
    RETURN c.cif_id AS cif_id,
           c.composition AS composition,
           a.name AS application,
           r['{metric}'] AS metric_value,
           sm.name AS synthesis_method
    ORDER BY r['{metric}'] {order}
    LIMIT $limit
    """
    return cypher, params


def element_substitution_analysis(
    composition: str,
    element_from: str,
    element_to: str,
) -> tuple[str, dict]:
    """Find materials where element_from is substituted with element_to.

    Example: "What happens when Co→Fe in LaCoO₃?"
    """
    cypher = """
    // Find source materials with the original element
    MATCH (source:Crystal {composition: $composition})-[:CONTAINS_ELEMENT]->(e_from:Element {symbol: $el_from})
    MATCH (source)-[:CONTAINS_ELEMENT]->(shared:Element)
    WHERE shared.symbol <> $el_from

    // Find targets that have the substitute element + same shared elements
    MATCH (target:Crystal)-[:CONTAINS_ELEMENT]->(e_to:Element {symbol: $el_to})
    MATCH (target)-[:CONTAINS_ELEMENT]->(shared)
    WHERE target <> source

    WITH source, target, collect(DISTINCT shared.symbol) AS shared_elements

    // Get performance comparison
    OPTIONAL MATCH (source)-[r1:ACHIEVES]->(a:Application)
    OPTIONAL MATCH (target)-[r2:ACHIEVES]->(a)

    RETURN source.composition AS source_composition,
           target.composition AS target_composition,
           target.cif_id AS target_cif_id,
           shared_elements,
           a.name AS application,
           r1.overpotential_mV AS source_overpotential,
           r2.overpotential_mV AS target_overpotential,
           r1.tafel_slope_mV_dec AS source_tafel,
           r2.tafel_slope_mV_dec AS target_tafel
    """
    return cypher, {
        "composition": composition,
        "el_from": element_from,
        "el_to": element_to,
    }


def application_overview() -> tuple[str, dict]:
    """Get an overview of crystals per application with average performance."""
    cypher = """
    MATCH (c:Crystal)-[r:ACHIEVES]->(a:Application)
    RETURN a.name AS application,
           count(DISTINCT c) AS crystal_count,
           avg(r.overpotential_mV) AS avg_overpotential,
           avg(r.tafel_slope_mV_dec) AS avg_tafel_slope,
           avg(r.faradaic_efficiency_pct) AS avg_faradaic_efficiency
    ORDER BY crystal_count DESC
    """
    return cypher, {}
