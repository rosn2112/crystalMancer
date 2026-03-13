"""COD CIF downloader with exponential backoff and idempotent resumption."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path

import requests

from crystalmancer.config import (
    BACKOFF_BASE,
    BACKOFF_FACTOR,
    BACKOFF_MAX,
    COD_CIF_URL_TEMPLATE,
    COD_SEARCH_URL,
    DEFAULT_CIF_DIR,
    JITTER_MAX,
    MAX_RETRIES,
)

logger = logging.getLogger(__name__)


def _backoff_sleep(attempt: int) -> None:
    """Sleep with exponential backoff + jitter."""
    delay = min(BACKOFF_BASE * (BACKOFF_FACTOR ** attempt), BACKOFF_MAX)
    delay += random.uniform(0, JITTER_MAX)
    time.sleep(delay)


def _request_with_retry(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response:
    """GET request with retries and exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            if resp.status_code == 429:
                logger.warning("Rate limited (429). Backing off (attempt %d).", attempt + 1)
                _backoff_sleep(attempt)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            logger.warning("Request failed (%s). Retrying (attempt %d).", exc, attempt + 1)
            _backoff_sleep(attempt)
    raise RuntimeError("Unreachable")  # pragma: no cover


def _parse_cod_response(resp: requests.Response) -> list[int]:
    """Parse COD API response into a list of integer IDs."""
    try:
        data = resp.json()
    except ValueError:
        lines = resp.text.strip().splitlines()
        return [int(line.strip()) for line in lines if line.strip().isdigit()]

    cod_ids: list[int] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                file_id = item.get("file")
                if file_id is not None:
                    try:
                        cod_ids.append(int(file_id))
                    except (ValueError, TypeError):
                        continue
            else:
                try:
                    cod_ids.append(int(item))
                except (ValueError, TypeError):
                    continue
    elif isinstance(data, dict):
        for key in data.keys():
            try:
                cod_ids.append(int(key))
            except (ValueError, TypeError):
                continue
    return cod_ids


# Targeted perovskite A-B element pairs (high-yield combinations)
_PEROVSKITE_AB_PAIRS = [
    ("Sr", "Ti"),  # SrTiO3
    ("Ba", "Ti"),  # BaTiO3
    ("La", "Co"),  # LaCoO3
    ("La", "Mn"),  # LaMnO3
    ("La", "Fe"),  # LaFeO3
    ("La", "Ni"),  # LaNiO3
    ("La", "Cr"),  # LaCrO3
    ("Ca", "Ti"),  # CaTiO3
    ("Sr", "Fe"),  # SrFeO3
    ("Ba", "Zr"),  # BaZrO3
    ("Bi", "Fe"),  # BiFeO3
    ("Nd", "Fe"),  # NdFeO3
    ("Pr", "Co"),  # PrCoO3
    ("Y", "Mn"),   # YMnO3
    ("Gd", "Fe"),  # GdFeO3
    ("Sr", "Co"),  # SrCoO3
    ("La", "Al"),  # LaAlO3
    ("Ca", "Mn"),  # CaMnO3
    ("Ba", "Co"),  # BaCoO3
    ("Sr", "Mn"),  # SrMnO3
]


def search_cod_oxide_ids(limit: int | None = None) -> list[int]:
    """Search COD for perovskite-like oxide materials.

    Uses targeted A-site + B-site + O queries to maximize perovskite hit rate.
    If *limit* is given, return at most that many IDs.
    """
    all_ids: list[int] = []
    seen: set[int] = set()
    per_query = max(10, (limit or 200) // len(_PEROVSKITE_AB_PAIRS) + 1)

    logger.info("Querying COD for perovskite oxides (%d A-B pairs) …", len(_PEROVSKITE_AB_PAIRS))

    for a_el, b_el in _PEROVSKITE_AB_PAIRS:
        if limit and len(all_ids) >= limit:
            break

        params = {
            "format": "json",
            "el1": a_el,
            "el2": b_el,
            "el3": "O",
            "nel1": "C",    # exclude organics
            "nel2": "H",    # exclude hydrates
        }

        try:
            resp = _request_with_retry(COD_SEARCH_URL, params=params, timeout=60)
            ids = _parse_cod_response(resp)

            new_ids = [cid for cid in ids if cid not in seen]
            seen.update(new_ids)
            all_ids.extend(new_ids[:per_query])

            logger.debug("  %s-%s-O → %d entries (%d new).", a_el, b_el, len(ids), len(new_ids))

        except Exception as exc:
            logger.debug("  %s-%s-O query failed: %s", a_el, b_el, exc)
            continue

    if limit is not None:
        all_ids = all_ids[:limit]

    logger.info("Found %d perovskite-targeted oxide entries in COD.", len(all_ids))
    return all_ids


def download_cif(cod_id: int, output_dir: Path = DEFAULT_CIF_DIR) -> Path | None:
    """Download a single CIF by COD ID.  Skips if already exists (idempotent)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_path = output_dir / f"{cod_id}.cif"
    if cif_path.exists():
        logger.debug("CIF %d already on disk — skipped.", cod_id)
        return cif_path
    url = COD_CIF_URL_TEMPLATE.format(cod_id)
    try:
        resp = _request_with_retry(url, timeout=30)
        cif_path.write_text(resp.text, encoding="utf-8")
        return cif_path
    except Exception as exc:
        logger.warning("Failed to download CIF %d: %s", cod_id, exc)
        return None


def download_cod_cifs(
    output_dir: Path = DEFAULT_CIF_DIR,
    limit: int | None = None,
    progress_callback=None,
) -> list[Path]:
    """Download oxide CIFs from COD.

    Parameters
    ----------
    output_dir : Path
        Directory to save CIF files into.
    limit : int | None
        Maximum number of CIFs to download. ``None`` for all.
    progress_callback : callable | None
        Called with ``(current_index, total)`` after each download.

    Returns
    -------
    list[Path]
        Paths to successfully downloaded CIF files.
    """
    cod_ids = search_cod_oxide_ids(limit=limit)
    downloaded: list[Path] = []
    for idx, cod_id in enumerate(cod_ids):
        path = download_cif(cod_id, output_dir)
        if path is not None:
            downloaded.append(path)
        if progress_callback:
            progress_callback(idx + 1, len(cod_ids))
    logger.info("Downloaded %d / %d CIFs.", len(downloaded), len(cod_ids))
    return downloaded
