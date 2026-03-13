"""Sci-Hub full-text paper retrieval and PDF-to-text extraction."""

from __future__ import annotations

import hashlib
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import requests

from crystalmancer.config import (
    BACKOFF_BASE,
    BACKOFF_FACTOR,
    BACKOFF_MAX,
    DEFAULT_PAPER_CACHE_DIR,
    JITTER_MAX,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)

SCIHUB_URLS = [
    "https://sci-hub.se",
    "https://sci-hub.st",
    "https://sci-hub.ru",
]


def _backoff_sleep(attempt: int) -> None:
    delay = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
    delay += random.uniform(0, JITTER_MAX)
    time.sleep(delay)


def _find_pdf_url(html: str, base_url: str) -> str | None:
    """Extract PDF download URL from Sci-Hub page HTML."""
    # Pattern 1: iframe/embed src
    patterns = [
        r'<iframe[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'<embed[^>]+src=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'<a[^>]+href=["\']([^"\']+\.pdf[^"\']*)["\']',
        r'(https?://[^\s"\'<>]+\.pdf)',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, re.IGNORECASE)
        if match:
            url = match.group(1)
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = base_url + url
            return url
    return None


def download_pdf(
    doi: str,
    cache_dir: Path | None = None,
) -> Path | None:
    """Download a paper PDF from Sci-Hub by DOI.

    Parameters
    ----------
    doi : str
        The paper DOI (e.g., "10.1039/c9ta01234").
    cache_dir : Path | None
        Directory to cache PDFs. Defaults to data/papers/pdfs/.

    Returns
    -------
    Path | None
        Path to the downloaded PDF, or None on failure.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_PAPER_CACHE_DIR / "pdfs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use DOI hash as filename to avoid path issues
    pdf_hash = hashlib.md5(doi.encode()).hexdigest()
    pdf_path = cache_dir / f"{pdf_hash}.pdf"

    if pdf_path.exists():
        logger.debug("PDF cache hit for DOI %s", doi)
        return pdf_path

    for base_url in SCIHUB_URLS:
        url = f"{base_url}/{doi}"
        for attempt in range(MAX_RETRIES):
            try:
                resp = requests.get(url, timeout=30, allow_redirects=True)
                if resp.status_code == 429:
                    _backoff_sleep(attempt)
                    continue
                if resp.status_code != 200:
                    break  # try next mirror

                # Check if we got a PDF directly
                content_type = resp.headers.get("content-type", "")
                if "pdf" in content_type.lower():
                    pdf_path.write_bytes(resp.content)
                    logger.info("Downloaded PDF for %s from %s", doi, base_url)
                    return pdf_path

                # Otherwise parse HTML for PDF link
                pdf_url = _find_pdf_url(resp.text, base_url)
                if pdf_url:
                    pdf_resp = requests.get(pdf_url, timeout=60)
                    if pdf_resp.status_code == 200:
                        pdf_path.write_bytes(pdf_resp.content)
                        logger.info("Downloaded PDF for %s from %s", doi, base_url)
                        return pdf_path

                break  # no PDF found on this mirror

            except requests.exceptions.RequestException as exc:
                if attempt == MAX_RETRIES - 1:
                    logger.debug("Mirror %s failed for %s: %s", base_url, doi, exc)
                    break
                _backoff_sleep(attempt)

    logger.warning("Could not download PDF for DOI %s from any Sci-Hub mirror.", doi)
    return None


def pdf_to_text(pdf_path: Path) -> str:
    """Extract text from a PDF file.

    Uses pdfplumber for pure-Python extraction (no system dependencies).
    Falls back to basic PyPDF2 if pdfplumber is unavailable.
    """
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return "\n\n".join(texts)
    except ImportError:
        pass

    # Fallback: PyPDF2
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(pdf_path))
        texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                texts.append(text)
        return "\n\n".join(texts)
    except ImportError:
        pass

    logger.error(
        "No PDF parser available. Install pdfplumber: pip install pdfplumber"
    )
    return ""


def fetch_fulltext(
    doi: str,
    cache_dir: Path | None = None,
) -> str | None:
    """Download and extract full text for a paper DOI.

    Returns
    -------
    str | None
        Full text of the paper, or None if unavailable.
    """
    # Check text cache first
    if cache_dir is None:
        cache_dir = DEFAULT_PAPER_CACHE_DIR / "pdfs"
    cache_dir.mkdir(parents=True, exist_ok=True)

    text_hash = hashlib.md5(doi.encode()).hexdigest()
    text_cache = cache_dir / f"{text_hash}.txt"

    if text_cache.exists():
        return text_cache.read_text(encoding="utf-8")

    pdf_path = download_pdf(doi, cache_dir)
    if pdf_path is None:
        return None

    text = pdf_to_text(pdf_path)
    if text:
        text_cache.write_text(text, encoding="utf-8")
    return text if text else None
