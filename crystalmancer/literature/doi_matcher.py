"""Sci-Hub DOI matcher using Kaggle dataset.

Uses the 'Complete list of Sci-Hub DOIs' dataset from Kaggle
(https://www.kaggle.com/datasets/danttis/complete-list-of-sci-hub-dois)
to check which papers are available on Sci-Hub before attempting download.

The dataset contains ~85M DOIs. We load it into a set for O(1) lookups.
"""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Any

from crystalmancer.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Expected location of the Kaggle dataset
DEFAULT_DOI_FILE = PROJECT_ROOT / "data" / "scihub_dois" / "scihub_dois.csv"
GZIP_DOI_FILE = PROJECT_ROOT / "data" / "scihub_dois" / "scihub_dois.csv.gz"


class SciHubDOIMatcher:
    """Check DOI availability on Sci-Hub using the Kaggle DOI dataset.

    Usage
    -----
    1. Download the dataset from Kaggle:
       https://www.kaggle.com/datasets/danttis/complete-list-of-sci-hub-dois
    2. Place the CSV in data/scihub_dois/scihub_dois.csv[.gz]
    3. Use this class to check DOIs before attempting Sci-Hub download.

    Example
    -------
    >>> matcher = SciHubDOIMatcher()
    >>> matcher.is_available("10.1038/nature12373")
    True
    >>> matcher.filter_available(papers)
    [...]  # only papers with DOIs on Sci-Hub
    """

    def __init__(self, doi_file: Path | None = None, lazy: bool = True):
        """Initialize the matcher.

        Parameters
        ----------
        doi_file : Path | None
            Path to the DOI file (.csv, .csv.gz, or .txt with one DOI per line).
            Defaults to auto-detect in data/scihub_dois/.
        lazy : bool
            If True, don't load until first query. If False, load immediately.
        """
        self._doi_file = doi_file or DEFAULT_DOI_FILE
        self._gzip_file = GZIP_DOI_FILE
        self._doi_dir = DEFAULT_DOI_FILE.parent
        self._dois: set[str] | None = None
        self._loaded = False

        if not lazy:
            self._load()

    def _find_doi_file(self) -> Path | None:
        """Auto-detect the DOI file in the scihub_dois directory."""
        # Check explicit paths first
        if self._gzip_file.exists():
            return self._gzip_file
        if self._doi_file.exists():
            return self._doi_file

        # Auto-detect any DOI-related file
        if self._doi_dir.exists():
            for pattern in ["*doi*.txt", "*doi*.csv", "*scihub*.txt", "*scihub*.csv"]:
                matches = list(self._doi_dir.glob(pattern))
                if matches:
                    return max(matches, key=lambda p: p.stat().st_size)  # largest file

        return None

    def _load(self) -> None:
        """Load DOIs into memory. Supports .txt, .csv, and .csv.gz formats."""
        if self._loaded:
            return

        path = self._find_doi_file()
        if path is None:
            logger.warning(
                "Sci-Hub DOI dataset not found. "
                "Download from: https://www.kaggle.com/datasets/danttis/complete-list-of-sci-hub-dois "
                "and place in %s",
                self._doi_dir,
            )
            self._dois = set()
            self._loaded = True
            return

        logger.info("Loading Sci-Hub DOI dataset from %s (this takes ~30s for 88M DOIs) …", path.name)

        self._dois = set()
        opener = gzip.open if path.suffix == '.gz' else open

        try:
            with opener(path, 'rt', encoding='utf-8', errors='ignore') as f:
                if path.suffix == '.txt':
                    # Plain text format: one DOI per line
                    for i, line in enumerate(f):
                        doi = line.strip().lower()
                        if doi.startswith('10.'):
                            self._dois.add(doi)
                        if (i + 1) % 10_000_000 == 0:
                            logger.info("  Loaded %dM DOIs …", (i + 1) // 1_000_000)
                else:
                    # CSV format: DOI in first column
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for i, row in enumerate(reader):
                        if row:
                            doi = row[0].strip().lower()
                            if doi.startswith('10.'):
                                self._dois.add(doi)
                        if (i + 1) % 5_000_000 == 0:
                            logger.info("  Loaded %dM DOIs …", (i + 1) // 1_000_000)

        except Exception as exc:
            logger.error("Failed to load DOI dataset: %s", exc)
            self._dois = set()

        self._loaded = True
        logger.info("Loaded %s DOIs from Sci-Hub dataset.", f"{len(self._dois):,}")

    def is_available(self, doi: str) -> bool:
        """Check if a DOI is available on Sci-Hub."""
        if not self._loaded:
            self._load()
        if not self._dois:
            return True  # assume available if dataset not loaded
        return doi.strip().lower() in self._dois

    def filter_available(
        self,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Filter a list of papers to only those available on Sci-Hub.

        Parameters
        ----------
        papers : list[dict]
            Papers with 'doi' key.

        Returns
        -------
        list[dict]
            Papers whose DOIs are in the Sci-Hub dataset.
        """
        if not self._loaded:
            self._load()

        if not self._dois:
            # Dataset not loaded — return all papers unchanged
            return papers

        available = []
        for paper in papers:
            doi = paper.get("doi")
            if not doi:
                continue
            if self.is_available(doi):
                available.append(paper)

        logger.debug(
            "Sci-Hub DOI filter: %d/%d papers available.",
            len(available), len(papers),
        )
        return available

    @property
    def count(self) -> int:
        """Number of DOIs in the dataset."""
        if not self._loaded:
            self._load()
        return len(self._dois) if self._dois else 0


def download_kaggle_dataset(output_dir: Path | None = None) -> Path:
    """Provide instructions for downloading the Kaggle dataset.

    Returns the expected output path.
    """
    out = output_dir or DEFAULT_DOI_FILE.parent
    out.mkdir(parents=True, exist_ok=True)

    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  Sci-Hub DOI Dataset Setup                                    ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  1. Go to:                                                    ║
║     https://www.kaggle.com/datasets/danttis/                  ║
║     complete-list-of-sci-hub-dois                             ║
║                                                               ║
║  2. Download the CSV file                                     ║
║                                                               ║
║  3. Place it in:                                              ║
║     {str(out):<55s}                                           ║
║                                                               ║
║  4. (Optional) gzip it to save space:                         ║
║     gzip scihub_dois.csv                                      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""")
    return out
