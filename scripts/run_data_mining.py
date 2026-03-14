#!/usr/bin/env python
"""
Crystal Mancer — Data Mining Orchestrator
==========================================

Single entry point that runs the complete data mining + enrichment pipeline:

  1. Mine literature (Semantic Scholar + CrossRef + Europe PMC)
     → Extract catalytic performance (regex + optional LLM)
     → Output: catalysis_papers.jsonl

  2. Build unified dataset from all structural data sources
     → Canonical composition deduplication
     → Keep best polymorph per formula
     → Output: unified_dataset.jsonl

  3. Enrich unified dataset with literature
     → Match papers to structures by composition
     → Attach performance data to structural records

  4. Generate statistics report

Usage:
  # Quick mode (for testing — 3 queries, regex only)
  conda run -n aienv python scripts/run_data_mining.py --quick

  # Full mode (all queries, regex only)
  conda run -n aienv python scripts/run_data_mining.py --full

  # Full mode with LLM extraction (requires OPENROUTER_API_KEY)
  conda run -n aienv python scripts/run_data_mining.py --full --use-llm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("data_mining")


def detect_output_dir() -> Path:
    """Auto-detect the best output directory (same logic as download_all.py)."""
    # 1. Google Colab
    colab_path = Path("/content/drive/MyDrive/CrystalMancerData")
    if colab_path.parent.exists():
        colab_path.mkdir(parents=True, exist_ok=True)
        return colab_path

    # 2. macOS Google Drive
    cloud_storage = Path.home() / "Library" / "CloudStorage"
    if cloud_storage.exists():
        for gdir in sorted(cloud_storage.iterdir()):
            if gdir.name.startswith("GoogleDrive"):
                my_drive = gdir / "My Drive" / "CrystalMancerData"
                try:
                    my_drive.mkdir(parents=True, exist_ok=True)
                    return my_drive
                except Exception:
                    pass

    # 3. Fallback: local
    from crystalmancer.config import DEFAULT_OUTPUT_DIR
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Crystal Mancer — Data Mining Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_data_mining.py --quick           # Test run (3 queries)
  python scripts/run_data_mining.py --full             # Full mining
  python scripts/run_data_mining.py --full --use-llm   # Full + LLM extraction
  python scripts/run_data_mining.py --enrich-only      # Only enrich existing data
        """,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true",
                      help="Quick mode: 3 queries, 10 papers each (for testing)")
    mode.add_argument("--full", action="store_true",
                      help="Full mode: all queries, 200+ papers each")
    mode.add_argument("--enrich-only", action="store_true",
                      help="Skip mining, only enrich existing dataset")

    parser.add_argument("--use-llm", action="store_true",
                        help="Use LLM extraction (requires OPENROUTER_API_KEY)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Override output directory")

    args = parser.parse_args()

    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  Crystal Mancer — Data Mining Pipeline                   ║")
    logger.info("╚" + "═" * 58 + "╝")

    output_dir = args.output_dir or detect_output_dir()
    logger.info("Output directory: %s", output_dir)

    t0 = time.time()
    stats: dict = {}

    # ── Step 1: Literature Mining ────────────────────────────────
    if not args.enrich_only:
        logger.info("")
        logger.info("═" * 60)
        logger.info("  STEP 1: LITERATURE MINING")
        logger.info("═" * 60)

        from scripts.mine_literature import main as mine_main
        lit_stats = mine_main(
            output_override=output_dir / "literature",
            quick=args.quick,
            use_llm=args.use_llm,
        )
        stats["literature"] = lit_stats
    else:
        logger.info("Skipping literature mining (--enrich-only)")

    # ── Step 2: Build Unified Dataset ────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info("  STEP 2: BUILDING UNIFIED DATASET")
    logger.info("═" * 60)

    from crystalmancer.data.dataset_builder import DatasetBuilder

    builder = DatasetBuilder(output_dir=output_dir)
    n_cod = builder.add_cod_records()
    n_mp = builder.add_mp_records()
    n_gnome = builder.add_gnome_records()

    logger.info("Raw records: COD=%d, MP=%d, GNoME=%d, Total=%d",
                n_cod, n_mp, n_gnome, len(builder.records))

    # ── Step 3: Enrich with Literature Data ──────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info("  STEP 3: ENRICHING WITH LITERATURE DATA")
    logger.info("═" * 60)

    n_enriched = builder.enrich_with_literature(output_dir / "literature")

    # ── Step 4: Deduplicate & Save ───────────────────────────────
    logger.info("")
    logger.info("═" * 60)
    logger.info("  STEP 4: DEDUPLICATING & SAVING")
    logger.info("═" * 60)

    output_file = builder.build()
    ds_stats = builder.stats

    # ── Final Report ─────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("╔" + "═" * 58 + "╗")
    logger.info("║  DATA MINING PIPELINE COMPLETE                           ║")
    logger.info("╠" + "═" * 58 + "╣")
    if "literature" in stats:
        lit = stats["literature"]
        logger.info("║  Papers mined:         %-6d                             ║", lit.get("total_papers", 0))
        logger.info("║  With performance:     %-6d                             ║", lit.get("with_performance", 0))
    logger.info("║  ─────────────────────────────                             ║")
    logger.info("║  Dataset records:      %-6d                             ║", ds_stats["total"])
    for src, count in sorted(ds_stats["by_source"].items()):
        logger.info("║    %-20s %-6d                             ║", src, count)
    logger.info("║  With DFT energy:      %-6d                             ║", ds_stats["has_energy"])
    logger.info("║  With catalysis data:  %-6d                             ║", ds_stats["has_catalysis"])
    logger.info("║  Literature-enriched:  %-6d                             ║", n_enriched)
    logger.info("║  Time elapsed:         %.1f min                             ║", elapsed / 60)
    logger.info("╚" + "═" * 58 + "╝")

    # Save stats as JSON
    stats_file = output_dir / "mining_stats.json"
    stats["dataset"] = ds_stats
    stats["enriched_records"] = n_enriched
    stats["elapsed_seconds"] = elapsed
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info("Stats saved to: %s", stats_file)


if __name__ == "__main__":
    main()
