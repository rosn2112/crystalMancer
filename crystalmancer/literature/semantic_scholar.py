"""Semantic Scholar API client with rate limiting and exponential backoff."""

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
    JITTER_MAX,
    MAX_RETRIES,
    S2_API_BASE,
)

logger = logging.getLogger(__name__)

# Module-level rate tracking
_request_timestamps: list[float] = []
_RATE_WINDOW = 300  # 5 minutes
_RATE_LIMIT = 95    # stay under 100


def _enforce_rate_limit() -> None:
    """Block until we're under the rate limit."""
    now = time.time()
    # Prune old timestamps
    _request_timestamps[:] = [t for t in _request_timestamps if now - t < _RATE_WINDOW]
    if len(_request_timestamps) >= _RATE_LIMIT:
        sleep_time = _RATE_WINDOW - (now - _request_timestamps[0]) + 1
        logger.info("Rate limit approaching — sleeping %.1fs", sleep_time)
        time.sleep(max(sleep_time, 1))


def _backoff_sleep(attempt: int) -> None:
    delay = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
    delay += random.uniform(0, JITTER_MAX)
    time.sleep(delay)


def search_papers(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search Semantic Scholar for papers matching *query*.

    Returns
    -------
    list[dict]
        Each dict has keys: doi, title, abstract, year.
        `doi` may be None if no DOI is registered.
    """
    _enforce_rate_limit()

    url = f"{S2_API_BASE}/paper/search"
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": "title,abstract,externalIds,year",
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=30)
            _request_timestamps.append(time.time())

            if resp.status_code == 429:
                logger.warning("S2 rate limited (429). Backing off …")
                _backoff_sleep(attempt)
                continue

            if resp.status_code == 404:
                # No results
                return []

            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                logger.error("S2 search failed after %d retries: %s", MAX_RETRIES, exc)
                return []
            logger.warning("S2 request error (%s). Retry %d …", exc, attempt + 1)
            _backoff_sleep(attempt)
    else:
        return []

    results: list[dict[str, Any]] = []
    for paper in data.get("data", []):
        ext_ids = paper.get("externalIds") or {}
        results.append({
            "doi": ext_ids.get("DOI"),
            "title": paper.get("title", ""),
            "abstract": paper.get("abstract") or "",
            "year": paper.get("year"),
        })

    return results
