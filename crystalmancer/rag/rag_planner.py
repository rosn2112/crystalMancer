"""RAG-assisted synthesis planner.

For each generated/selected crystal structure, retrieves similar
synthesized materials from the knowledge base and generates an
adapted synthesis plan using LLM.
"""

from __future__ import annotations

import logging
from typing import Any

from crystalmancer.extraction.llm_client import chat_completion

logger = logging.getLogger(__name__)

_SYNTHESIS_PLAN_SYSTEM = """You are a materials synthesis expert specializing in inorganic oxide catalysts, 
particularly perovskite-family materials for electrocatalysis and photocatalysis.

Given information about a target crystal structure and similar materials from the literature,
generate a practical synthesis plan. Include:

1. **Recommended Synthesis Route** — step-by-step procedure
2. **Precursors** — specific chemicals with CAS numbers where possible
3. **Equipment Required** — furnaces, autoclaves, etc.
4. **Key Parameters** — temperatures, times, atmospheres, heating rates
5. **Characterization Checklist** — what measurements to perform and why
6. **Potential Pitfalls** — common failure modes from literature
7. **Safety Notes** — any hazardous materials or conditions

Format as clean markdown. Be specific and practical — this should be usable as a lab protocol."""

_SYNTHESIS_PLAN_USER = """Generate a synthesis plan for the following target material:

**Target Composition:** {composition}
**Space Group:** {spacegroup}
**Target Application:** {application}

**Similar Materials from Literature:**
{similar_materials}

**Performance Targets:**
{performance_targets}

Generate a detailed, practical synthesis plan."""


def format_similar_materials(results: list[dict]) -> str:
    """Format similar materials search results for the LLM prompt."""
    if not results:
        return "No similar materials found in the database."

    lines = []
    for i, r in enumerate(results[:5], 1):
        lines.append(f"{i}. **{r.get('composition', 'Unknown')}** (CIF: {r.get('cif_id', 'N/A')})")
        if r.get("synthesis_method"):
            lines.append(f"   - Synthesis: {r['synthesis_method']}")
        if r.get("application"):
            lines.append(f"   - Application: {r['application']}")
        if r.get("abstract"):
            abstract = r["abstract"][:300] + "..." if len(r.get("abstract", "")) > 300 else r.get("abstract", "")
            lines.append(f"   - Abstract: {abstract}")
        if r.get("doi"):
            lines.append(f"   - DOI: {r['doi']}")
        lines.append("")

    return "\n".join(lines)


def format_performance_targets(targets: dict[str, float | None] | None) -> str:
    """Format performance targets for the prompt."""
    if not targets:
        return "No specific performance targets."

    lines = []
    labels = {
        "overpotential_mV": "Overpotential",
        "faradaic_efficiency_pct": "Faradaic Efficiency",
        "tafel_slope_mV_dec": "Tafel Slope",
        "current_density_mA_cm2": "Current Density",
        "stability_h": "Stability",
    }
    units = {
        "overpotential_mV": "mV",
        "faradaic_efficiency_pct": "%",
        "tafel_slope_mV_dec": "mV/dec",
        "current_density_mA_cm2": "mA/cm²",
        "stability_h": "h",
    }

    for key, label in labels.items():
        val = targets.get(key)
        if val is not None:
            lines.append(f"- {label}: {val} {units.get(key, '')}")

    return "\n".join(lines) if lines else "No specific performance targets."


def generate_synthesis_plan(
    composition: str,
    spacegroup: str = "unknown",
    application: str = "OER",
    similar_materials: list[dict] | None = None,
    performance_targets: dict[str, float | None] | None = None,
    model: str | None = None,
) -> str:
    """Generate a synthesis plan for a target material using RAG + LLM.

    Parameters
    ----------
    composition : str
        Target material composition (e.g., "LaCoO3").
    spacegroup : str
        Space group of target structure.
    application : str
        Target catalytic application.
    similar_materials : list[dict] | None
        Similar materials from FAISS search / knowledge graph.
    performance_targets : dict | None
        Target performance metrics.
    model : str | None
        Override OpenRouter model ID.

    Returns
    -------
    str
        Markdown-formatted synthesis plan.
    """
    prompt = _SYNTHESIS_PLAN_USER.format(
        composition=composition,
        spacegroup=spacegroup,
        application=application,
        similar_materials=format_similar_materials(similar_materials or []),
        performance_targets=format_performance_targets(performance_targets),
    )

    response = chat_completion(
        messages=[
            {"role": "system", "content": _SYNTHESIS_PLAN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.3,
        max_tokens=4096,
    )

    return response


def generate_characterization_checklist(
    composition: str,
    application: str = "OER",
    model: str | None = None,
) -> str:
    """Generate a characterization checklist for a synthesized material."""
    prompt = f"""For a newly synthesized {composition} catalyst targeting {application}:

Generate a prioritized characterization checklist with:
1. **Structural characterization** (XRD, Raman, FTIR, etc.)
2. **Morphological analysis** (SEM, TEM, BET, etc.)
3. **Compositional analysis** (XPS, EDS, ICP, etc.)
4. **Electrochemical testing** (LSV, CV, EIS, chronoamperometry, etc.)
5. **Expected results** for each measurement

Format as a checklist with brief explanations of what to look for."""

    return chat_completion(
        messages=[
            {"role": "system", "content": _SYNTHESIS_PLAN_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.2,
        max_tokens=2048,
    )
