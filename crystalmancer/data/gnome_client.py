"""GNoME (Graph Networks for Materials Exploration) dataset loader.

Google DeepMind's GNoME discovered 2.2M new crystals (380K stable).
The dataset is publicly available at gs://gdm_materials_discovery.

This module downloads and processes GNoME crystal structures.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

DEFAULT_GNOME_DIR = DEFAULT_OUTPUT_DIR / "gnome"

# Direct download URLs for GNoME data
GNOME_STABLE_URL = (
    "https://storage.googleapis.com/gdm_materials_discovery/"
    "gnome_data/stable_materials_summary.csv"
)
GNOME_CIF_BASE = (
    "https://storage.googleapis.com/gdm_materials_discovery/"
    "gnome_data/by_id/{material_id}.cif"
)
# Batch download using gsutil (much faster)
GNOME_BUCKET = "gs://gdm_materials_discovery/"


def download_gnome_summary(
    output_dir: Path = DEFAULT_GNOME_DIR,
) -> Path | None:
    """Download the GNoME stable materials summary CSV.

    Contains ~380,000 stable materials with:
    - Material ID
    - Chemical formula
    - Space group
    - Formation energy (eV/atom)
    - Energy above hull (eV/atom)
    - Number of sites

    Returns path to downloaded CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stable_materials_summary.csv"

    if csv_path.exists():
        logger.info("GNoME summary already downloaded (%s).", csv_path)
        return csv_path

    logger.info("Downloading GNoME stable materials summary (~50 MB) …")

    try:
        resp = requests.get(GNOME_STABLE_URL, timeout=300, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(csv_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total and downloaded % (5 * 1024 * 1024) == 0:
                    pct = downloaded / total * 100
                    logger.info("  %.0f%% downloaded …", pct)

        logger.info("✅ GNoME summary saved to %s", csv_path)
        return csv_path

    except Exception as exc:
        logger.error("Failed to download GNoME summary: %s", exc)
        csv_path.unlink(missing_ok=True)
        return None


def load_gnome_summary(
    csv_path: Path | None = None,
    output_dir: Path = DEFAULT_GNOME_DIR,
    filter_oxides: bool = True,
    max_energy_above_hull: float = 0.05,  # Very stable only
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and filter GNoME materials from summary CSV.

    Parameters
    ----------
    csv_path : Path | None
        Path to summary CSV. Downloads if not provided.
    filter_oxides : bool
        If True, only keep materials containing oxygen.
    max_energy_above_hull : float
        Maximum energy above hull (eV/atom). 0.05 = very stable.
    limit : int | None
        Max materials to return.

    Returns
    -------
    list[dict]
        Filtered materials with metadata.
    """
    if csv_path is None:
        csv_path = download_gnome_summary(output_dir)
    if csv_path is None or not csv_path.exists():
        return []

    materials = []
    logger.info("Loading GNoME summary from %s …", csv_path.name)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                formula = row.get("Reduced Formula", "")
                # GNoME uses "Decomposition Energy Per Atom" ≈ energy above hull
                e_hull = float(row.get("Decomposition Energy Per Atom", 999) or 999)

                # Apply filters
                if max_energy_above_hull and e_hull > max_energy_above_hull:
                    continue
                if filter_oxides and "O" not in formula:
                    continue

                material = {
                    "source": "gnome",
                    "material_id": row.get("MaterialId", ""),
                    "composition": formula,
                    "spacegroup": row.get("Space Group", ""),
                    "spacegroup_number": int(row.get("Space Group Number", 0) or 0),
                    "formation_energy_per_atom": float(row.get("Formation Energy Per Atom", 0) or 0),
                    "energy_above_hull": e_hull,
                    "nsites": int(row.get("NSites", 0) or 0),
                    "band_gap": float(row.get("Bandgap", 0) or 0),
                }
                materials.append(material)

                if limit and len(materials) >= limit:
                    break

            except (ValueError, KeyError) as exc:
                continue

    logger.info("Loaded %d GNoME materials (filtered).", len(materials))
    return materials


def download_gnome_cifs(
    materials: list[dict[str, Any]],
    output_dir: Path = DEFAULT_GNOME_DIR,
    batch_size: int = 100,
) -> list[Path]:
    """Download CIF files for selected GNoME materials.

    Parameters
    ----------
    materials : list[dict]
        Materials from load_gnome_summary().
    output_dir : Path
        Where to save CIF files.
    batch_size : int
        Download in batches to avoid overwhelming the server.

    Returns
    -------
    list[Path]
        Paths to downloaded CIF files.
    """
    cif_dir = output_dir / "cifs"
    cif_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    for i, mat in enumerate(materials):
        mat_id = mat["material_id"]
        cif_path = cif_dir / f"{mat_id}.cif"

        if cif_path.exists():
            downloaded.append(cif_path)
            continue

        url = GNOME_CIF_BASE.format(material_id=mat_id)
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                cif_path.write_text(resp.text, encoding="utf-8")
                downloaded.append(cif_path)
            else:
                logger.debug("CIF not found for %s (status %d)", mat_id, resp.status_code)
        except Exception as exc:
            logger.debug("Failed to download CIF for %s: %s", mat_id, exc)

        if (i + 1) % batch_size == 0:
            logger.info("  Downloaded %d/%d CIFs …", i + 1, len(materials))
            time.sleep(1)  # Rate limiting

    logger.info("✅ Downloaded %d/%d GNoME CIF files.", len(downloaded), len(materials))
    return downloaded


def download_gnome_bulk_gsutil(
    output_dir: Path = DEFAULT_GNOME_DIR,
) -> bool:
    """Download the entire GNoME dataset using gsutil (fastest method).

    Requires Google Cloud SDK: https://cloud.google.com/sdk/docs/install
    No authentication needed — the bucket is public.

    Returns True if successful.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Attempting bulk download via gsutil …")
    logger.info("Source: %s", GNOME_BUCKET)
    logger.info("Destination: %s", output_dir)

    try:
        result = subprocess.run(
            ["gsutil", "-m", "cp", "-r", GNOME_BUCKET, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )
        if result.returncode == 0:
            logger.info("✅ GNoME bulk download complete!")
            return True
        else:
            logger.warning("gsutil failed: %s", result.stderr[:500])
            return False
    except FileNotFoundError:
        logger.info(
            "gsutil not found. Install Google Cloud SDK:\n"
            "  brew install google-cloud-sdk\n"
            "Or download individual CIFs instead."
        )
        return False
    except subprocess.TimeoutExpired:
        logger.warning("Download timed out after 2 hours.")
        return False
