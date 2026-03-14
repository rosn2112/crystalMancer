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

Run:
  conda run -n aienv python scripts/mine_literature.py          # Full mode
  conda run -n aienv python scripts/mine_literature.py --quick  # Test mode (3 queries)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import random
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
    # ORR
    "perovskite oxygen reduction reaction ORR",
    "manganese oxide ORR electrocatalyst alkaline",
    # NRR — nitrogen reduction
    "nitrogen reduction reaction electrocatalyst ammonia",
    "metal oxide NRR Faradaic efficiency",
    # Photocatalysis
    "TiO2 photocatalyst hydrogen production",
    "BiVO4 photoelectrochemical water oxidation",
    "g-C3N4 photocatalyst visible light",
    "halide perovskite photocatalysis CO2",
    # Thermochemical water splitting
    "ceria redox thermochemical water splitting",
    "perovskite thermochemical solar fuel",
    # Batteries & energy storage
    "lithium cobalt oxide cathode capacity",
    "spinel LiMn2O4 battery cycling stability",
    "solid state electrolyte garnet oxide ionic conductivity",
    # General / computational
    "DFT calculation oxide catalyst formation energy",
    "machine learning crystal structure prediction catalyst",
    "high-throughput screening electrocatalyst descriptor",
    "Tafel slope exchange current density oxide",
]

# Quick mode uses only the first 3 queries for fast testing
QUICK_QUERIES = CATALYSIS_QUERIES[:3]


# ── LLM extraction prompt ───────────────────────────────────────
LLM_SYSTEM_PROMPT = """You are a materials science data extractor. Given a paper abstract,
extract catalytic performance data as JSON. Return ONLY valid JSON with these fields
(use null for missing values):

{
  "materials": ["SrTiO3", "LaCoO3"],
  "reaction_type": "OER",
  "overpotential_mV": 320,
  "tafel_slope_mV_dec": 65,
  "faradaic_efficiency_pct": 95.2,
  "current_density_mA_cm2": 10,
  "band_gap_eV": 3.2,
  "stability_hours": 24,
  "synthesis_method": "sol-gel",
  "electrolyte": "1M KOH",
  "temperature_C": 25
}

Rules:
- Extract ALL materials mentioned (chemical formulas only, e.g. "LaCoO3" not "lanthanum cobaltite")
- reaction_type must be one of: OER, HER, CO2RR, ORR, NRR, photocatalysis, thermochemical, battery
- Only extract numbers explicitly stated in the text
- Do NOT guess or infer values"""


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
    stability_hours: float | None = None
    synthesis_method: str = ""
    electrolyte: str = ""
    reaction_type: str = ""  # OER, HER, CO2RR...
    on_scihub: bool = False
    extraction_method: str = "regex"  # "regex", "llm", "regex+llm"

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v}

    def has_performance_data(self) -> bool:
        """Return True if any catalytic performance data was extracted."""
        return bool(
            self.overpotential_mV or self.tafel_slope or
            self.faradaic_efficiency or self.current_density or
            self.band_gap_eV or self.stability_hours
        )


# ── Robust HTTP GET with Exponential Backoff ─────────────────────
def robust_get(url: str, params: dict, headers: dict | None = None,
               max_retries: int = 5) -> requests.Response | None:
    """Perform a GET request with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code in [429, 502, 503, 504]:
                wait_time = (2 ** attempt) * 10 + random.randint(1, 10)
                logger.warning("HTTP %d for %s... waiting %ds (Attempt %d/%d)",
                               resp.status_code, url[:30], wait_time, attempt+1, max_retries)
                time.sleep(wait_time)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            wait_time = (2 ** attempt) * 5 + random.randint(1, 5)
            logger.warning("Request exception for %s...: %s, waiting %ds (Attempt %d/%d)",
                           url[:30], e, wait_time, attempt+1, max_retries)
            time.sleep(wait_time)

    logger.error("Max retries reached for %s", url[:50])
    return None


# ── Semantic Scholar search ──────────────────────────────────────
def search_semantic_scholar(query: str, limit: int = 100) -> list[PaperRecord]:
    """Search Semantic Scholar for catalysis papers."""
    papers = []
    offset = 0

    while offset < limit:
        batch = min(100, limit - offset)
        try:
            resp = robust_get(
                f"{SEMANTIC_SCHOLAR_API}/paper/search",
                params={
                    "query": query,
                    "limit": batch,
                    "offset": offset,
                    "fields": "title,abstract,year,authors,externalIds,journal",
                }
            )
            if not resp:
                break

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

            time.sleep(2)  # Base rate limit

        except Exception as exc:
            logger.warning("S2 search failed for '%s': %s", query[:40], exc)
            break

    return papers


# ── CrossRef search ──────────────────────────────────────────────
def search_crossref(query: str, limit: int = 50) -> list[PaperRecord]:
    """Search CrossRef for catalysis papers with DOIs."""
    papers = []
    try:
        resp = robust_get(
            CROSSREF_API,
            params={
                "query": query,
                "rows": limit,
                "select": "DOI,title,abstract,published-print,author,container-title",
                "filter": "has-abstract:true",
            },
            headers={"User-Agent": "CrystalMancer/1.0 (mailto:research@example.com)"}
        )
        if not resp:
            return papers

        data = resp.json()

        for item in data.get("message", {}).get("items", []):
            doi = item.get("DOI", "")
            titles = item.get("title", [""])
            abstracts = item.get("abstract", "")
            # CrossRef abstracts have JATS XML tags, strip them
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
        resp = robust_get(
            EUROPE_PMC_API,
            params={
                "query": query,
                "resultType": "core",
                "pageSize": min(limit, 100),
                "format": "json",
            }
        )
        if not resp:
            return papers

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


# ── Extract catalytic data — Regex ───────────────────────────────
def extract_catalytic_data_regex(paper: PaperRecord) -> PaperRecord:
    """Extract catalytic performance data using regex patterns.

    Faster than LLM extraction, catches most common reporting formats.
    """
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

    # Stability (hours)
    match = re.search(r"stability\s*(?:of|for)?\s*(\d+(?:\.\d+)?)\s*h(?:ours?)?", text, re.IGNORECASE)
    if match:
        paper.stability_hours = float(match.group(1))

    # Reaction type
    if any(w in text_lower for w in ["oer", "oxygen evolution"]):
        paper.reaction_type = "OER"
    elif any(w in text_lower for w in ["her", "hydrogen evolution"]):
        paper.reaction_type = "HER"
    elif any(w in text_lower for w in ["co2rr", "co₂ reduction", "co2 reduction"]):
        paper.reaction_type = "CO2RR"
    elif any(w in text_lower for w in ["orr", "oxygen reduction"]):
        paper.reaction_type = "ORR"
    elif any(w in text_lower for w in ["nrr", "nitrogen reduction", "ammonia synthesis"]):
        paper.reaction_type = "NRR"
    elif any(w in text_lower for w in ["photocatal", "water splitting"]):
        paper.reaction_type = "photocatalysis"
    elif any(w in text_lower for w in ["thermochemical", "solar fuel", "redox cycle"]):
        paper.reaction_type = "thermochemical"
    elif any(w in text_lower for w in ["battery", "cathode", "anode", "li-ion", "lithium"]):
        paper.reaction_type = "battery"

    # Material formulas (improved patterns)
    material_patterns = [
        # Perovskite-type: ABO3 with optional doping
        r"((?:Sr|Ba|La|Ca|Y|Sc|Bi|Pb|Pr|Nd|Sm|Gd|Ce|Na|K|Li)"
        r"(?:\d*\.?\d*)"
        r"(?:[A-Z][a-z]?\d*\.?\d*){1,3}"
        r"O\d*(?:\.\d+)?(?:[+-]?\w*)?)",
        # Generic MxOy patterns
        r"((?:Ti|Fe|Co|Mn|Ni|Cu|Zn|V|Cr|Mo|W|Ru|Ir|Nb|Ta|Zr|Hf|"
        r"Al|Si|Ga|In|Sn|Bi|Sb|Pt|Pd|Au|Ag|Cd|Ce)"
        r"(?:\d*\.?\d*)"
        r"(?:[A-Z][a-z]?\d*\.?\d*)*"
        r"O\d+(?:\.\d+)?)",
    ]
    for pat in material_patterns:
        matches = re.findall(pat, text)
        paper.materials.extend([m for m in matches if len(m) > 2])

    # Deduplicate materials
    paper.materials = list(set(paper.materials))[:15]
    paper.extraction_method = "regex"

    return paper


# ── Extract catalytic data — LLM ────────────────────────────────
def _has_openrouter_key() -> bool:
    return bool(os.environ.get("OPENROUTER_API_KEY", ""))


def extract_catalytic_data_llm(paper: PaperRecord) -> dict:
    """Extract catalytic data from abstract using OpenRouter LLM.

    Returns a dict of extracted fields, or empty dict on failure.
    """
    try:
        from crystalmancer.extraction.llm_client import extract_json
        result = extract_json(
            prompt=f"Abstract:\n{paper.abstract}\n\nTitle: {paper.title}",
            system_prompt=LLM_SYSTEM_PROMPT,
        )
        return result if isinstance(result, dict) else {}
    except Exception as exc:
        logger.debug("LLM extraction failed for %s: %s", paper.doi[:30], exc)
        return {}


def _merge_llm_results(paper: PaperRecord, llm_data: dict) -> None:
    """Merge LLM extraction results into paper record (LLM fills gaps)."""
    if not llm_data:
        return

    # Materials — merge and deduplicate
    llm_materials = llm_data.get("materials", [])
    if isinstance(llm_materials, list):
        all_mats = set(paper.materials) | set(str(m) for m in llm_materials if m)
        paper.materials = list(all_mats)[:15]

    # Numeric fields — LLM fills gaps only
    if paper.overpotential_mV is None and llm_data.get("overpotential_mV"):
        paper.overpotential_mV = float(llm_data["overpotential_mV"])
    if paper.tafel_slope is None and llm_data.get("tafel_slope_mV_dec"):
        paper.tafel_slope = float(llm_data["tafel_slope_mV_dec"])
    if paper.faradaic_efficiency is None and llm_data.get("faradaic_efficiency_pct"):
        paper.faradaic_efficiency = float(llm_data["faradaic_efficiency_pct"])
    if paper.current_density is None and llm_data.get("current_density_mA_cm2"):
        paper.current_density = float(llm_data["current_density_mA_cm2"])
    if paper.band_gap_eV is None and llm_data.get("band_gap_eV"):
        paper.band_gap_eV = float(llm_data["band_gap_eV"])
    if paper.stability_hours is None and llm_data.get("stability_hours"):
        paper.stability_hours = float(llm_data["stability_hours"])

    # String fields
    if not paper.reaction_type and llm_data.get("reaction_type"):
        paper.reaction_type = str(llm_data["reaction_type"])
    if not paper.synthesis_method and llm_data.get("synthesis_method"):
        paper.synthesis_method = str(llm_data["synthesis_method"])
    if not paper.electrolyte and llm_data.get("electrolyte"):
        paper.electrolyte = str(llm_data["electrolyte"])

    paper.extraction_method = "regex+llm"


def extract_catalytic_data(paper: PaperRecord, use_llm: bool = True) -> PaperRecord:
    """Extract catalytic data: regex first, then LLM fills gaps.

    Parameters
    ----------
    paper : PaperRecord
        Paper with abstract to extract from.
    use_llm : bool
        If True and OPENROUTER_API_KEY is set, use LLM after regex.
    """
    # Step 1: Always do regex (fast, no API required)
    paper = extract_catalytic_data_regex(paper)

    # Step 2: Optionally use LLM to fill gaps
    if use_llm and paper.abstract and _has_openrouter_key():
        try:
            llm_data = extract_catalytic_data_llm(paper)
            _merge_llm_results(paper, llm_data)
        except Exception:
            pass  # regex results are fine

    return paper


# ── Main pipeline ────────────────────────────────────────────────
def main(output_override: Path | None = None, quick: bool = False,
         use_llm: bool = False) -> dict:
    """Run the literature mining pipeline.

    Parameters
    ----------
    output_override : Path | None
        Override output directory path.
    quick : bool
        Quick mode: use 3 queries, 10 papers each (for testing).
    use_llm : bool
        Use LLM extraction when OPENROUTER_API_KEY is set.

    Returns
    -------
    dict
        Statistics about the mining run.
    """
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Crystal Mancer — Literature Mining Pipeline             ║")
    logger.info("╚" + "═" * 58 + "╝")

    if quick:
        logger.info("⚡ QUICK MODE: 3 queries, 10 papers each")
    if use_llm and _has_openrouter_key():
        logger.info("🤖 LLM extraction ENABLED (OpenRouter)")
    elif use_llm:
        logger.info("⚠️  LLM requested but OPENROUTER_API_KEY not set. Regex only.")
        use_llm = False

    output_dir = output_override or (DEFAULT_OUTPUT_DIR / "literature")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "catalysis_papers.jsonl"

    # Load DOI matcher
    doi_matcher = None
    try:
        from crystalmancer.literature.doi_matcher import SciHubDOIMatcher
        logger.info("Loading Sci-Hub DOI matcher (88M DOIs) …")
        doi_matcher = SciHubDOIMatcher(lazy=True)
    except Exception:
        pass

    # Collect papers from all sources
    all_papers: dict[str, PaperRecord] = {}  # doi -> paper (dedup)

    # Load existing papers
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    if rec.get("doi"):
                        all_papers[rec["doi"]] = PaperRecord(**{
                            k: v for k, v in rec.items()
                            if k in PaperRecord.__dataclass_fields__
                        })
                except Exception:
                    pass
        logger.info("Loaded %d existing paper records.", len(all_papers))

    # Select queries
    queries = QUICK_QUERIES if quick else CATALYSIS_QUERIES
    s2_limit = 10 if quick else 200
    cr_limit = 10 if quick else 100
    pmc_limit = 10 if quick else 100

    # Progress tracking
    total_new = 0
    total_llm_enriched = 0
    total_with_perf = 0

    for i, query in enumerate(queries):
        logger.info("─" * 60)
        logger.info("Query %d/%d: %s", i + 1, len(queries), query)

        # Search all APIs
        s2_papers = search_semantic_scholar(query, limit=s2_limit)
        cr_papers = search_crossref(query, limit=cr_limit)
        pmc_papers = search_europe_pmc(query, limit=pmc_limit)

        new_count = 0
        for paper in s2_papers + cr_papers + pmc_papers:
            if paper.doi and paper.doi not in all_papers:
                # Extract catalytic data (regex + optional LLM)
                paper = extract_catalytic_data(paper, use_llm=use_llm)

                # Check Sci-Hub availability
                if doi_matcher:
                    paper.on_scihub = doi_matcher.is_available(paper.doi)

                all_papers[paper.doi] = paper
                new_count += 1
                total_new += 1

                if paper.has_performance_data():
                    total_with_perf += 1
                if paper.extraction_method == "regex+llm":
                    total_llm_enriched += 1

        logger.info("  S2: %d, CrossRef: %d, PMC: %d → %d new papers",
                     len(s2_papers), len(cr_papers), len(pmc_papers), new_count)

        # Save incrementally (resume-safe)
        with open(output_file, "w", encoding="utf-8") as f:
            for p in all_papers.values():
                f.write(json.dumps(p.to_dict(), default=str) + "\n")

        # Rate limiting between queries
        time.sleep(3)

    # Final statistics
    n_total = len(all_papers)
    n_with_performance = sum(1 for p in all_papers.values() if p.has_performance_data())
    n_on_scihub = sum(1 for p in all_papers.values() if p.on_scihub)
    n_with_materials = sum(1 for p in all_papers.values() if p.materials)

    # Reaction type breakdown
    reaction_counts: dict[str, int] = {}
    for p in all_papers.values():
        if p.reaction_type:
            reaction_counts[p.reaction_type] = reaction_counts.get(p.reaction_type, 0) + 1

    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  LITERATURE MINING COMPLETE                              ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║  Total unique papers:    %-6d                           ║", n_total)
    logger.info("║  New this run:           %-6d                           ║", total_new)
    logger.info("║  With performance data:  %-6d                           ║", n_with_performance)
    logger.info("║  With material names:    %-6d                           ║", n_with_materials)
    logger.info("║  LLM-enriched:           %-6d                           ║", total_llm_enriched)
    logger.info("║  Available on Sci-Hub:   %-6d                           ║", n_on_scihub)
    if reaction_counts:
        logger.info("╠" + "═" * 58 + "╣")
        for rtype, count in sorted(reaction_counts.items(), key=lambda x: -x[1]):
            logger.info("║    %-22s %-6d                           ║", rtype, count)
    logger.info("║  Saved to: %s", str(output_file)[:47])
    logger.info("╚" + "═" * 58 + "╝")

    return {
        "total_papers": n_total,
        "new_papers": total_new,
        "with_performance": n_with_performance,
        "with_materials": n_with_materials,
        "llm_enriched": total_llm_enriched,
        "on_scihub": n_on_scihub,
        "reaction_types": reaction_counts,
        "output_file": str(output_file),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crystal Mancer — Literature Mining")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 queries, 10 papers each")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM extraction (requires OPENROUTER_API_KEY)")
    args = parser.parse_args()
    main(quick=args.quick, use_llm=args.use_llm)
