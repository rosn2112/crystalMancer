#!/usr/bin/env python
"""
Crystal Mancer — Literature Mining via DOI + APIs
==================================================

Uses the 88M Sci-Hub DOI dataset + multiple APIs to find and extract
catalytic performance data from real scientific papers.

Sources:
  1. Semantic Scholar — paper metadata + abstracts
  2. CrossRef — DOI resolution + references
  3. Europe PMC — full-text OA articles
  4. OpenRouter LLM — extract structured data from abstracts

Run: conda run -n aienv python scripts/mine_literature.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mine_literature")

# ── API endpoints ────────────────────────────────────────────────
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"
CROSSREF_API = "https://api.crossref.org/works"
EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# ── Catalysis search queries ─────────────────────────────────────
CATALYSIS_QUERIES = [
    # OER/HER — electrocatalysis
    "perovskite oxide OER overpotential electrocatalyst",
    "transition metal oxide oxygen evolution reaction",
    "water splitting photocatalyst band gap",
    "SrTiO3 photocatalysis hydrogen evolution",
    "LaCoO3 perovskite electrocatalyst OER",
    "IrO2 RuO2 OER benchmark overpotential",
    "NiFe layered double hydroxide OER",
    "spinel oxide electrocatalyst oxygen reduction",
    # CO2RR
    "CO2 reduction electrocatalyst Faradaic efficiency",
    "copper oxide CO2 reduction selectivity",
    "perovskite CO2 reduction overpotential",
    # Photocatalysis
    "TiO2 photocatalyst hydrogen production",
    "BiVO4 photoelectrochemical water oxidation",
    "g-C3N4 photocatalyst visible light",
    # General
    "DFT calculation oxide catalyst formation energy",
    "machine learning crystal structure prediction catalyst",
    "high-throughput screening electrocatalyst descriptor",
    "Tafel slope exchange current density oxide",
]


@dataclass
class PaperRecord:
    """Structured record of a catalysis paper."""
    doi: str = ""
    title: str = ""
    abstract: str = ""
    year: int = 0
    authors: list[str] = field(default_factory=list)
    journal: str = ""
    source_api: str = ""

    # Extracted catalytic data (from LLM or regex)
    materials: list[str] = field(default_factory=list)
    overpotential_mV: float | None = None
    tafel_slope: float | None = None
    faradaic_efficiency: float | None = None
    current_density: float | None = None
    band_gap_eV: float | None = None
    reaction_type: str = ""  # OER, HER, CO2RR...
    on_scihub: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}


# ── Semantic Scholar search ──────────────────────────────────────
def search_semantic_scholar(query: str, limit: int = 100) -> list[PaperRecord]:
    """Search Semantic Scholar for catalysis papers."""
    papers = []
    offset = 0

    while offset < limit:
        batch = min(100, limit - offset)
        try:
            resp = requests.get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params={
                    "query": query,
                    "limit": batch,
                    "offset": offset,
                    "fields": "title,abstract,year,authors,externalIds,journal",
                },
                timeout=15,
            )
            if resp.status_code == 429:
                logger.warning("S2 rate limited, waiting 60s …")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("data", []):
                doi = (item.get("externalIds") or {}).get("DOI", "")
                rec = PaperRecord(
                    doi=doi,
                    title=item.get("title", ""),
                    abstract=item.get("abstract", ""),
                    year=item.get("year", 0) or 0,
                    authors=[a.get("name", "") for a in (item.get("authors") or [])[:5]],
                    journal=(item.get("journal") or {}).get("name", ""),
                    source_api="semantic_scholar",
                )
                if rec.doi and rec.abstract:
                    papers.append(rec)

            total = data.get("total", 0)
            offset += batch
            if offset >= total:
                break

            time.sleep(1)  # Rate limit: 1 req/s for unauthenticated

        except Exception as exc:
            logger.warning("S2 search failed for '%s': %s", query[:40], exc)
            break

    return papers


# ── CrossRef search ──────────────────────────────────────────────
def search_crossref(query: str, limit: int = 50) -> list[PaperRecord]:
    """Search CrossRef for catalysis papers with DOIs."""
    papers = []
    try:
        resp = requests.get(
            CROSSREF_API,
            params={
                "query": query,
                "rows": limit,
                "select": "DOI,title,abstract,published-print,author,container-title",
                "filter": "has-abstract:true",
            },
            headers={"User-Agent": "CrystalMancer/1.0 (mailto:research@example.com)"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("message", {}).get("items", []):
            doi = item.get("DOI", "")
            titles = item.get("title", [""])
            abstracts = item.get("abstract", "")
            # CrossRef abstracts have JATS XML tags, strip them
            import re
            if abstracts:
                abstracts = re.sub(r"<[^>]+>", "", abstracts)

            year = 0
            pub = item.get("published-print", {}).get("date-parts", [[0]])
            if pub and pub[0]:
                year = pub[0][0]

            rec = PaperRecord(
                doi=doi,
                title=titles[0] if titles else "",
                abstract=abstracts,
                year=year,
                authors=[f"{a.get('given','')} {a.get('family','')}".strip()
                         for a in (item.get("author") or [])[:5]],
                journal=(item.get("container-title") or [""])[0],
                source_api="crossref",
            )
            if rec.doi and rec.abstract:
                papers.append(rec)

    except Exception as exc:
        logger.warning("CrossRef search failed: %s", exc)

    return papers


# ── Europe PMC search (free full text) ───────────────────────────
def search_europe_pmc(query: str, limit: int = 50) -> list[PaperRecord]:
    """Search Europe PMC for open access catalysis papers."""
    papers = []
    try:
        resp = requests.get(
            EUROPE_PMC_API,
            params={
                "query": query,
                "resultType": "core",
                "pageSize": min(limit, 100),
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("resultList", {}).get("result", []):
            doi = item.get("doi", "")
            rec = PaperRecord(
                doi=doi,
                title=item.get("title", ""),
                abstract=item.get("abstractText", ""),
                year=int(item.get("pubYear", 0) or 0),
                authors=[f"{a.get('firstName', '')} {a.get('lastName', '')}".strip()
                         for a in (item.get("authorList", {}).get("author") or [])[:5]],
                journal=item.get("journalTitle", ""),
                source_api="europe_pmc",
            )
            if rec.doi and rec.abstract:
                papers.append(rec)

    except Exception as exc:
        logger.warning("Europe PMC search failed: %s", exc)

    return papers


# ── Extract catalytic data from abstracts ────────────────────────
def extract_catalytic_data_regex(paper: PaperRecord) -> PaperRecord:
    """Extract catalytic performance data using regex patterns.

    Faster than LLM extraction, catches most common reporting formats.
    """
    import re

    text = (paper.abstract or "") + " " + (paper.title or "")
    text_lower = text.lower()

    # Overpotential (mV)
    match = re.search(r"overpotential\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mV", text, re.IGNORECASE)
    if match:
        paper.overpotential_mV = float(match.group(1))

    # Tafel slope (mV/dec)
    match = re.search(r"Tafel\s+slope\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*mV\s*/?\s*dec", text, re.IGNORECASE)
    if match:
        paper.tafel_slope = float(match.group(1))

    # Faradaic efficiency (%)
    match = re.search(r"[Ff]aradaic\s+efficiency\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%", text)
    if match:
        paper.faradaic_efficiency = float(match.group(1))

    # Current density (mA/cm²)
    match = re.search(r"(\d+(?:\.\d+)?)\s*mA\s*/?\s*cm", text)
    if match:
        paper.current_density = float(match.group(1))

    # Band gap (eV)
    match = re.search(r"band\s*gap\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*eV", text, re.IGNORECASE)
    if match:
        paper.band_gap_eV = float(match.group(1))

    # Reaction type
    if any(w in text_lower for w in ["oer", "oxygen evolution"]):
        paper.reaction_type = "OER"
    elif any(w in text_lower for w in ["her", "hydrogen evolution"]):
        paper.reaction_type = "HER"
    elif any(w in text_lower for w in ["co2rr", "co₂ reduction", "co2 reduction"]):
        paper.reaction_type = "CO2RR"
    elif any(w in text_lower for w in ["orr", "oxygen reduction"]):
        paper.reaction_type = "ORR"
    elif any(w in text_lower for w in ["photocatal", "water splitting"]):
        paper.reaction_type = "photocatalysis"

    # Material names (common patterns)
    material_patterns = [
        r"([A-Z][a-z]?\d*(?:O\d+)?(?:\.\d+)?)",  # Simple formulas
        r"((?:Sr|Ba|La|Ca|Y|Sc|Ti|Mn|Fe|Co|Ni|Cu|Ru|Ir|Pt|Pd)(?:[A-Z][a-z]?\d*)*O\d+)",
    ]
    for pat in material_patterns:
        matches = re.findall(pat, text)
        paper.materials.extend([m for m in matches if len(m) > 2 and "O" in m])

    # Deduplicate materials
    paper.materials = list(set(paper.materials))[:10]

    return paper


# ── Main pipeline ────────────────────────────────────────────────
def main():
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Crystal Mancer — Literature Mining Pipeline             ║")
    logger.info("╚" + "═" * 58 + "╝")

    output_dir = DEFAULT_OUTPUT_DIR / "literature"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "catalysis_papers.jsonl"

    # Load DOI matcher
    try:
        from crystalmancer.literature.doi_matcher import SciHubDOIMatcher
        logger.info("Loading Sci-Hub DOI matcher (88M DOIs) …")
        doi_matcher = SciHubDOIMatcher(lazy=True)
    except Exception:
        doi_matcher = None

    # Collect papers from all sources
    all_papers: dict[str, PaperRecord] = {}  # doi -> paper (dedup)

    # Load existing papers
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("doi"):
                        all_papers[rec["doi"]] = PaperRecord(**{k: v for k, v in rec.items()
                                                                 if k in PaperRecord.__dataclass_fields__})
                except:
                    pass
        logger.info("Loaded %d existing paper records.", len(all_papers))

    for i, query in enumerate(CATALYSIS_QUERIES):
        logger.info("─" * 60)
        logger.info("Query %d/%d: %s", i + 1, len(CATALYSIS_QUERIES), query)

        # Search all APIs in parallel-ish fashion
        s2_papers = search_semantic_scholar(query, limit=200)
        cr_papers = search_crossref(query, limit=100)
        pmc_papers = search_europe_pmc(query, limit=100)

        new_count = 0
        for paper in s2_papers + cr_papers + pmc_papers:
            if paper.doi and paper.doi not in all_papers:
                # Extract catalytic data from abstract
                paper = extract_catalytic_data_regex(paper)

                # Check Sci-Hub availability
                if doi_matcher:
                    paper.on_scihub = doi_matcher.is_available(paper.doi)

                all_papers[paper.doi] = paper
                new_count += 1

        logger.info("  S2: %d, CrossRef: %d, PMC: %d → %d new papers",
                     len(s2_papers), len(cr_papers), len(pmc_papers), new_count)

        # Rate limiting between queries
        time.sleep(2)

    # Save all papers
    with open(output_file, "w", encoding="utf-8") as f:
        for paper in all_papers.values():
            f.write(json.dumps(paper.to_dict(), default=str) + "\n")

    # Statistics
    n_with_performance = sum(1 for p in all_papers.values()
                            if p.overpotential_mV or p.tafel_slope or p.faradaic_efficiency)
    n_on_scihub = sum(1 for p in all_papers.values() if p.on_scihub)
    n_with_materials = sum(1 for p in all_papers.values() if p.materials)

    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  LITERATURE MINING COMPLETE                              ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║  Total unique papers:    %-6d                           ║", len(all_papers))
    logger.info("║  With performance data:  %-6d                           ║", n_with_performance)
    logger.info("║  With material names:    %-6d                           ║", n_with_materials)
    logger.info("║  Available on Sci-Hub:   %-6d                           ║", n_on_scihub)
    logger.info("║  Saved to: %s", str(output_file)[:47])
    logger.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
