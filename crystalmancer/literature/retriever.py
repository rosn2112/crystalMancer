"""Orchestrate per-CIF paper retrieval from multiple sources.

Sources: Semantic Scholar, CrossRef, Europe PMC, CORE, PubMed.
All free, no API keys required.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from crystalmancer.config import DEFAULT_PAPER_CACHE_DIR
from crystalmancer.literature import semantic_scholar, crossref
from crystalmancer.literature.open_access import (
    search_europepmc,
    search_core,
    search_pubmed,
)

logger = logging.getLogger(__name__)


def _cache_key(composition: str, keywords: tuple[str, ...]) -> str:
    raw = f"{composition}|{'|'.join(keywords)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _load_cache(cache_dir: Path, key: str) -> list[dict] | None:
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _save_cache(cache_dir: Path, key: str, papers: list[dict]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{key}.json"
    cache_file.write_text(json.dumps(papers, ensure_ascii=False, indent=2), encoding="utf-8")


def _deduplicate(papers: list[dict]) -> list[dict]:
    """Deduplicate papers by DOI (keep first occurrence)."""
    seen_dois: set[str] = set()
    seen_titles: set[str] = set()
    unique: list[dict] = []
    for p in papers:
        doi = p.get("doi")
        title = (p.get("title") or "").lower().strip()
        if doi and doi in seen_dois:
            continue
        if not doi and title in seen_titles:
            continue
        if doi:
            seen_dois.add(doi)
        if title:
            seen_titles.add(title)
        unique.append(p)
    return unique


def retrieve_papers(
    composition: str,
    max_papers: int = 5,
    application_keywords: tuple[str, ...] = ("catalysis", "catalyst", "OER", "HER", "CO2RR"),
    cache_dir: Path = DEFAULT_PAPER_CACHE_DIR,
    use_open_access: bool = True,
) -> list[dict[str, Any]]:
    """Retrieve papers related to a material composition from 5 sources.

    Sources queried: Semantic Scholar, CrossRef, Europe PMC, CORE, PubMed.
    Results are cached per (composition, keywords) to avoid redundant API calls.

    Parameters
    ----------
    composition : str
        Reduced formula, e.g. "LaCoO3".
    max_papers : int
        Maximum number of papers to return.
    application_keywords : tuple[str, ...]
        Keywords combined with composition for the search query.
    cache_dir : Path
        Directory for caching API responses.
    use_open_access : bool
        If True, also query Europe PMC, CORE, and PubMed (default True).

    Returns
    -------
    list[dict]
        Papers with keys: doi, title, abstract, year, source.
    """
    key = _cache_key(composition, application_keywords)
    cached = _load_cache(cache_dir, key)
    if cached is not None:
        logger.debug("Cache hit for %s (%d papers).", composition, len(cached))
        return cached[:max_papers]

    # Build a query like "LaCoO3 catalysis catalyst OER HER"
    query = f"{composition} {' '.join(application_keywords)}"
    logger.info("Retrieving papers for %s from 5 sources …", composition)

    # Query all sources (per_source = max_papers to maximize coverage before dedup)
    per_source = max(max_papers, 3)

    s2_results = semantic_scholar.search_papers(query, limit=per_source)
    cr_results = crossref.search_papers(query, limit=per_source)

    oa_results: list[dict] = []
    if use_open_access:
        try:
            oa_results += search_europepmc(query, max_results=per_source)
        except Exception as exc:
            logger.debug("Europe PMC failed: %s", exc)
        try:
            oa_results += search_core(query, max_results=per_source)
        except Exception as exc:
            logger.debug("CORE failed: %s", exc)
        try:
            oa_results += search_pubmed(query, max_results=per_source)
        except Exception as exc:
            logger.debug("PubMed failed: %s", exc)

    # Merge all sources and deduplicate
    all_papers = s2_results + cr_results + oa_results
    merged = _deduplicate(all_papers)[:max_papers]

    # Cache
    _save_cache(cache_dir, key, merged)
    logger.info("Retrieved %d papers for %s (from %d raw results).",
                len(merged), composition, len(all_papers))
    return merged

