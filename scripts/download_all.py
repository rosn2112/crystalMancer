#!/usr/bin/env python
"""
Crystal Mancer — Overnight Bulk Data Download Script
=====================================================

Run this overnight to collect the full training dataset:
  conda run -n aienv python scripts/download_all.py

Downloads:
  1. Materials Project: ALL oxides with DFT data (~49K structures)
  2. GNoME: ~79K stable oxide structures + CIFs for top 10K
  3. COD: Perovskite-targeted CIFs
  4. Enriches with Sci-Hub DOI matching

Heavy data goes to Google Drive (symlinked) if available.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("download_all")

# ── Google Drive setup ──────────────────────────────────────────────
GDRIVE_CANDIDATES = [
    Path("/Users/roshan/Google Drive/Other computers/My PC - 1/Desktop/Code/CrystalMancerData"),
]

def find_or_create_gdrive_dir() -> Path | None:
    """Attempt to find Google Drive mount and create data dir."""
    for candidate in GDRIVE_CANDIDATES:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            logger.info("Using Google Drive: %s", candidate)
            return candidate
        except Exception as e:
            logger.warning("Could not create %s: %s", candidate, e)

    logger.info("Google Drive not found. Using local storage.")
    return None


def setup_storage() -> Path:
    """Set up data storage, preferring Google Drive for heavy files."""
    output_dir = DEFAULT_OUTPUT_DIR

    gdrive = find_or_create_gdrive_dir()
    if gdrive:
        # Ensure the local output directory exists before symlinking inside it
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Symlink heavy data dirs to Google Drive
        for subdir in ["materials_project", "gnome", "cifs", "literature"]:
            gdrive_sub = gdrive / subdir
            gdrive_sub.mkdir(parents=True, exist_ok=True)
            local_sub = output_dir / subdir
            if local_sub.exists() and not local_sub.is_symlink():
                # Move existing data to gdrive
                import shutil
                for f in local_sub.iterdir():
                    dest = gdrive_sub / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                local_sub.rmdir()
            if not local_sub.exists():
                local_sub.symlink_to(gdrive_sub)
                logger.info("  Symlinked %s → %s", local_sub.name, gdrive_sub)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# ── Materials Project: ALL oxides ────────────────────────────────
def download_all_mp_oxides(output_dir: Path, api_key: str) -> int:
    """Download ALL oxide structures from Materials Project.

    Strategy: query for all compounds containing O, with 2-5 elements,
    that are near the convex hull (stable). This gives ~49K structures.
    Each comes with DFT-computed formation energy, band gap, etc.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        logger.error("mp-api not installed. Run: conda run -n aienv pip install mp-api")
        return 0

    mp_dir = output_dir / "materials_project"
    mp_dir.mkdir(parents=True, exist_ok=True)
    jsonl_file = mp_dir / "mp_all_oxides.jsonl"
    cif_dir = mp_dir / "cifs"
    cif_dir.mkdir(exist_ok=True)

    # Check how many we already have
    existing_ids = set()
    if jsonl_file.exists():
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    rec = json.loads(line.strip())
                    existing_ids.add(rec.get("material_id"))
                except:
                    pass
        logger.info("Found %d existing MP records. Resuming.", len(existing_ids))

    logger.info("═" * 60)
    logger.info("  DOWNLOADING ALL OXIDES FROM MATERIALS PROJECT")
    logger.info("═" * 60)

    total_new = 0

    with MPRester(api_key) as mpr:
        # Query ALL oxides in one shot — MP handles pagination internally
        logger.info("Querying MP for all stable oxides (energy_above_hull < 0.1 eV/atom) …")
        logger.info("This query returns ~49K structures. Be patient.")

        try:
            docs = mpr.materials.summary.search(
                elements=["O"],
                num_elements=(2, 6),
                energy_above_hull=(None, 0.1),
                fields=[
                    "material_id", "formula_pretty", "structure", "symmetry",
                    "formation_energy_per_atom", "energy_above_hull",
                    "band_gap", "is_metal", "density", "volume", "nsites",
                    "is_stable", "ordering", "total_magnetization",
                ],
            )
            logger.info("MP returned %d total oxide structures.", len(docs))
        except Exception as exc:
            logger.error("MP query failed: %s", exc)
            # Try smaller batches by element system
            logger.info("Falling back to per-element queries …")
            docs = []
            b_elements = [
                "Ti", "Fe", "Co", "Mn", "Ni", "Cu", "Zn", "V", "Cr",
                "Mo", "W", "Ru", "Ir", "Nb", "Ta", "Zr", "Hf",
                "Al", "Si", "Ga", "In", "Sn", "Bi", "Sb",
                "La", "Ce", "Pr", "Nd", "Sm", "Gd", "Y", "Sc",
                "Sr", "Ba", "Ca", "Mg", "Li", "Na", "K",
                "Pb", "Cd", "Pt", "Pd", "Au", "Ag",
            ]
            for el in b_elements:
                try:
                    batch = mpr.materials.summary.search(
                        elements=[el, "O"],
                        num_elements=(2, 5),
                        energy_above_hull=(None, 0.1),
                        fields=[
                            "material_id", "formula_pretty", "structure", "symmetry",
                            "formation_energy_per_atom", "energy_above_hull",
                            "band_gap", "is_metal", "density", "volume", "nsites",
                        ],
                    )
                    docs.extend(batch)
                    logger.info("  %s-O: %d structures (total so far: %d)", el, len(batch), len(docs))
                    time.sleep(0.5)  # Rate limit
                except Exception as e2:
                    logger.warning("  %s-O query failed: %s", el, e2)

            # Deduplicate
            seen = set()
            unique_docs = []
            for d in docs:
                mid = d.material_id
                if mid not in seen:
                    seen.add(mid)
                    unique_docs.append(d)
            docs = unique_docs
            logger.info("Deduplicated to %d unique structures.", len(docs))

        # Write records
        with open(jsonl_file, "a", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                mid = doc.material_id
                if mid in existing_ids:
                    continue

                try:
                    struct = doc.structure
                    record = {
                        "source": "materials_project",
                        "material_id": mid,
                        "composition": doc.formula_pretty,
                        "spacegroup": doc.symmetry.symbol if doc.symmetry else "unknown",
                        "spacegroup_number": doc.symmetry.number if doc.symmetry else 0,
                        "formation_energy_per_atom": doc.formation_energy_per_atom,
                        "energy_above_hull": doc.energy_above_hull,
                        "band_gap": doc.band_gap,
                        "is_metal": doc.is_metal,
                        "density": doc.density,
                        "volume": doc.volume,
                        "nsites": doc.nsites,
                        "cif_string": struct.to(fmt="cif"),
                        "lattice": {
                            "a": struct.lattice.a, "b": struct.lattice.b,
                            "c": struct.lattice.c, "alpha": struct.lattice.alpha,
                            "beta": struct.lattice.beta, "gamma": struct.lattice.gamma,
                        },
                        "elements": [str(el) for el in struct.composition.elements],
                    }

                    f.write(json.dumps(record, default=str) + "\n")
                    existing_ids.add(mid)
                    total_new += 1

                    # Save CIF
                    cif_path = cif_dir / f"{mid}.cif"
                    if not cif_path.exists():
                        cif_path.write_text(record["cif_string"], encoding="utf-8")

                except Exception as exc:
                    logger.debug("Failed to process %s: %s", mid, exc)

                if (i + 1) % 2000 == 0:
                    logger.info("  Processed %d/%d (new: %d)", i + 1, len(docs), total_new)

    logger.info("✅ MP download complete: %d new + %d existing = %d total",
                total_new, len(existing_ids) - total_new, len(existing_ids))
    return len(existing_ids)


# ── GNoME: Download CIFs for selected structures ────────────────
def download_gnome_data(output_dir: Path, max_cifs: int = 10000) -> int:
    """Download GNoME structures with CIF files.

    The summary CSV is already downloaded (79K oxides).
    Here we download CIF files for the top candidates.
    """
    from crystalmancer.data.gnome_client import (
        load_gnome_summary, download_gnome_cifs, download_gnome_summary,
    )

    logger.info("═" * 60)
    logger.info("  DOWNLOADING GNoME CIF FILES")
    logger.info("═" * 60)

    # Ensure summary is downloaded
    gnome_dir = output_dir / "gnome"
    download_gnome_summary(gnome_dir)

    # Load all stable oxides
    materials = load_gnome_summary(
        output_dir=gnome_dir,
        filter_oxides=True,
        max_energy_above_hull=0.05,  # Very stable only for CIF download
        limit=max_cifs,
    )

    if not materials:
        logger.warning("No GNoME materials to download.")
        return 0

    logger.info("Downloading CIFs for %d GNoME materials …", len(materials))
    downloaded = download_gnome_cifs(materials, output_dir=gnome_dir, batch_size=100)

    logger.info("✅ GNoME: %d CIF files downloaded.", len(downloaded))
    return len(downloaded)


# ── Unified dataset builder ─────────────────────────────────────
def build_unified_dataset(output_dir: Path) -> int:
    """Merge all sources, deduplicate, and create unified dataset."""
    from crystalmancer.data.dataset_builder import DatasetBuilder

    logger.info("═" * 60)
    logger.info("  BUILDING UNIFIED DATASET")
    logger.info("═" * 60)

    builder = DatasetBuilder(output_dir=output_dir)
    n_cod = builder.add_cod_records()
    n_mp = builder.add_mp_records()
    n_gnome = builder.add_gnome_records()

    logger.info("Before dedup: COD=%d, MP=%d, GNoME=%d, Total=%d",
                n_cod, n_mp, n_gnome, len(builder.records))

    builder.build()

    stats = builder.stats
    logger.info("═" * 60)
    logger.info("  UNIFIED DATASET READY")
    logger.info("  Total unique: %d", stats["total"])
    logger.info("  With DFT energy: %d", stats["has_energy"])
    logger.info("  With catalysis data: %d", stats["has_catalysis"])
    logger.info("═" * 60)

    return stats["total"]


# ── Main ─────────────────────────────────────────────────────────
def main():
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Crystal Mancer — Overnight Bulk Download                ║")
    logger.info("╚" + "═" * 58 + "╝")

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        logger.error("Set MP_API_KEY: export MP_API_KEY='your_key'")
        sys.exit(1)

    # Setup storage (Google Drive if available)
    output_dir = setup_storage()

    t0 = time.time()

    # Step 1: Materials Project — ALL oxides
    n_mp = download_all_mp_oxides(output_dir, api_key)

    # Step 2: GNoME — Top 10K stable oxide CIFs
    n_gnome = download_gnome_data(output_dir, max_cifs=10000)

    # Step 3: Extract paper metadata (with DOIs!)
    logger.info("═" * 60)
    logger.info("  MINING LITERATURE FOR EXPERIMENTAL DATA")
    logger.info("═" * 60)
    try:
        from scripts.mine_literature import main as mine_main
        mine_main()
    except Exception as e:
        logger.error("Literature mining encountered an error: %s", e)

    # Step 4: Build unified dataset
    n_total = build_unified_dataset(output_dir)

    elapsed = time.time() - t0
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  DOWNLOAD COMPLETE                                       ║")
    logger.info("╠" + "═" * 58 + "╣")
    logger.info("║  Materials Project: %-6d structures                     ║", n_mp)
    logger.info("║  GNoME CIFs:       %-6d files                          ║", n_gnome)
    logger.info("║  Unified dataset:  %-6d total                          ║", n_total)
    logger.info("║  Time elapsed:     %-6.1f hours                         ║", elapsed / 3600)
    logger.info("╚" + "═" * 58 + "╝")


if __name__ == "__main__":
    main()
