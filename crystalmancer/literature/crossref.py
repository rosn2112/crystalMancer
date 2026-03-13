"""CrossRef API client for DOI-linked paper metadata."""

from __future__ import annotations

import logging
import random
import time
from typing import Any

import requests

from crystalmancer.config import (
    BACKOFF_BASE,
    BACKOFF_FACTOR,
    BACKOFF_MAX,
    CROSSREF_API_BASE,
    CROSSREF_MAILTO,
    JITTER_MAX,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def _backoff_sleep(attempt: int) -> None:
    delay = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
    delay += random.uniform(0, JITTER_MAX)
    time.sleep(delay)


def search_papers(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search CrossRef for papers matching *query*.

    Uses the polite pool (``mailto`` header) for higher rate limits.

    Returns
    -------
    list[dict]
        Each dict has keys: doi, title, abstract, year.
    """
    headers = {
        "User-Agent": f"CrystalMancer/0.1 (mailto:{CROSSREF_MAILTO})",
    }
    params = {
        "query": query,
        "rows": min(limit, 20),
        "select": "DOI,title,abstract,published-print,published-online",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                CROSSREF_API_BASE,
                params=params,
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                logger.warning("CrossRef rate limited (429). Backing off …")
                _backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                logger.error("CrossRef search failed after %d retries: %s", MAX_RETRIES, exc)
                return []
            logger.warning("CrossRef request error (%s). Retry %d …", exc, attempt + 1)
            _backoff_sleep(attempt)
    else:
        return []

    results: list[dict[str, Any]] = []
    for item in data.get("message", {}).get("items", []):
        title_parts = item.get("title", [])
        title = title_parts[0] if title_parts else ""

        # Extract abstract (CrossRef uses JATS XML fragments)
        abstract_raw = item.get("abstract", "")
        # Strip JATS tags
        import re
        abstract = re.sub(r"<[^>]+>", "", abstract_raw).strip()

        # Year from published-print or published-online
        year = None
        for date_field in ("published-print", "published-online"):
            date_parts = item.get(date_field, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break

        results.append({
            "doi": item.get("DOI"),
            "title": title,
            "abstract": abstract,
            "year": year,
        })

    return results
