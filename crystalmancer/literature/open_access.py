"""Europe PMC, CORE, and PubMed API clients for open-access literature.

These supplement Semantic Scholar + CrossRef to maximize paper coverage,
especially for catalysis and materials science papers.
"""

from __future__ import annotations

import logging
import random
import re
import time
from typing import Any

import requests

from crystalmancer.config import BACKOFF_BASE, BACKOFF_FACTOR, BACKOFF_MAX, JITTER_MAX, MAX_RETRIES

logger = logging.getLogger(__name__)


def _backoff(attempt: int) -> None:
    delay = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
    delay += random.uniform(0, JITTER_MAX)
    time.sleep(delay)


# ═══════════════════════════════════════════════════════════════════════════════
#  Europe PMC — open-access biomedical & chemistry literature
# ═══════════════════════════════════════════════════════════════════════════════

EUROPEPMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search_europepmc(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search Europe PMC for papers matching a query.

    Europe PMC indexes PubMed, PMC, and preprint servers. Has excellent
    coverage of chemistry / materials science papers.

    Parameters
    ----------
    query : str
        Search query (e.g., "LaCoO3 OER catalyst").
    max_results : int
        Maximum papers to return.

    Returns
    -------
    list[dict]
        Papers with keys: title, abstract, doi, year, source, authors.
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": min(max_results, 25),
        "resultType": "core",  # includes abstract
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(EUROPEPMC_API, params=params, timeout=30)

            if resp.status_code == 429:
                _backoff(attempt)
                continue

            resp.raise_for_status()
            data = resp.json()

            papers = []
            for item in data.get("resultList", {}).get("result", []):
                abstract = item.get("abstractText", "")
                if not abstract:
                    continue

                papers.append({
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "doi": item.get("doi"),
                    "year": item.get("pubYear"),
                    "source": "europe_pmc",
                    "authors": item.get("authorString", ""),
                    "pmid": item.get("pmid"),
                    "pmcid": item.get("pmcid"),
                    "journal": item.get("journalTitle", ""),
                })

            logger.debug("Europe PMC returned %d papers for '%s'.", len(papers), query[:50])
            return papers[:max_results]

        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                logger.warning("Europe PMC search failed: %s", exc)
                return []
            _backoff(attempt)

    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE — world's largest open-access research aggregator
# ═══════════════════════════════════════════════════════════════════════════════

CORE_API = "https://api.core.ac.uk/v3/search/works"


def search_core(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search CORE for open-access papers.

    CORE aggregates 300M+ documents from 10K+ repositories worldwide.
    Free API with no key required for basic search.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int
        Maximum papers to return.

    Returns
    -------
    list[dict]
        Papers with keys: title, abstract, doi, year, source, fulltext_url.
    """
    params = {
        "q": query,
        "limit": min(max_results, 20),
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(CORE_API, params=params, timeout=30)

            if resp.status_code == 429:
                _backoff(attempt)
                continue

            if resp.status_code != 200:
                logger.debug("CORE returned %d.", resp.status_code)
                return []

            data = resp.json()

            papers = []
            for item in data.get("results", []):
                abstract = item.get("abstract", "")
                if not abstract:
                    continue

                papers.append({
                    "title": item.get("title", ""),
                    "abstract": abstract,
                    "doi": item.get("doi"),
                    "year": item.get("yearPublished"),
                    "source": "core",
                    "authors": ", ".join(
                        a.get("name", "") for a in item.get("authors", [])
                    ),
                    "fulltext_url": item.get("downloadUrl"),
                })

            logger.debug("CORE returned %d papers for '%s'.", len(papers), query[:50])
            return papers[:max_results]

        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                logger.warning("CORE search failed: %s", exc)
                return []
            _backoff(attempt)

    return []


# ═══════════════════════════════════════════════════════════════════════════════
#  PubMed — NIH biomedical literature database
# ═══════════════════════════════════════════════════════════════════════════════

PUBMED_SEARCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search PubMed for papers and retrieve their abstracts.

    Relevant for bio-inorganic catalysis, enzyme mimics, and
    catalysis-related biomedical applications.

    Parameters
    ----------
    query : str
        Search query.
    max_results : int
        Maximum papers to return.

    Returns
    -------
    list[dict]
        Papers with keys: title, abstract, doi, year, source, pmid.
    """
    # Step 1: Search for PMIDs
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(PUBMED_SEARCH_API, params=search_params, timeout=30)
            resp.raise_for_status()
            search_data = resp.json()
            pmids = search_data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return []

            # Step 2: Fetch abstracts for PMIDs
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
            }

            # Rate limit: NCBI allows 3 req/sec without key
            time.sleep(0.4)

            resp2 = requests.get(PUBMED_FETCH_API, params=fetch_params, timeout=30)
            resp2.raise_for_status()
            xml_text = resp2.text

            return _parse_pubmed_xml(xml_text)

        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                logger.warning("PubMed search failed: %s", exc)
                return []
            _backoff(attempt)

    return []


def _parse_pubmed_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse PubMed XML response into paper dicts (simple regex-based)."""
    papers = []

    # Split into articles
    articles = re.split(r"<PubmedArticle>", xml_text)[1:]

    for article in articles:
        title_match = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", article, re.DOTALL)
        abstract_match = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
        doi_match = re.search(r'<ArticleId IdType="doi">(.*?)</ArticleId>', article)
        pmid_match = re.search(r'<ArticleId IdType="pubmed">(.*?)</ArticleId>', article)
        year_match = re.search(r"<PubDate>.*?<Year>(.*?)</Year>", article, re.DOTALL)

        if not abstract_match:
            continue

        # Strip XML tags from abstract
        abstract = re.sub(r"<[^>]+>", "", abstract_match.group(1)).strip()
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""

        papers.append({
            "title": title,
            "abstract": abstract,
            "doi": doi_match.group(1) if doi_match else None,
            "year": year_match.group(1) if year_match else None,
            "source": "pubmed",
            "pmid": pmid_match.group(1) if pmid_match else None,
        })

    return papers
