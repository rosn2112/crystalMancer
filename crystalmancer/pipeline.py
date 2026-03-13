"""End-to-end Phase 1 pipeline: download → filter → retrieve papers → extract → store → report."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from crystalmancer.cif.downloader import download_cod_cifs
from crystalmancer.cif.filter import filter_cifs
from crystalmancer.config import DEFAULT_CIF_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_PAPER_CACHE_DIR
from crystalmancer.extraction.extractor import extract_all
from crystalmancer.literature.retriever import retrieve_papers
from crystalmancer.reporting.summary import generate_report
from crystalmancer.storage.json_store import record_exists, save_record

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the Phase 1 pipeline run."""

    cif_dir: Path = DEFAULT_CIF_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    paper_cache_dir: Path = DEFAULT_PAPER_CACHE_DIR
    max_cifs: int | None = None          # None → no limit
    max_papers: int = 5                  # papers per CIF
    skip_download: bool = False          # use existing CIFs on disk
    dry_run: bool = False                # skip API calls, use sample data


def _make_sample_papers() -> list[dict]:
    """Return fake paper data for --dry-run mode."""
    return [
        {
            "doi": "10.1234/sample.001",
            "title": "Sample paper on perovskite OER catalysis",
            "abstract": (
                "LaCoO3 perovskite was synthesized via sol-gel method using citric acid. "
                "The catalyst achieved an overpotential of 350 mV at 10 mA cm⁻² "
                "with a Tafel slope of 62 mV/dec for the oxygen evolution reaction. "
                "Stability was maintained for 24 h of continuous operation."
            ),
            "year": 2024,
        }
    ]


def run_pipeline(config: PipelineConfig | None = None) -> None:
    """Execute the full Phase 1 pipeline.

    Steps:
    1. Download oxide CIFs from COD (or skip if ``--skip-download``)
    2. Filter for perovskite structures (space group + composition)
    3. For each perovskite CIF:
       a. Retrieve related papers (Semantic Scholar + CrossRef)
       b. Extract synthesis method, application, performance metrics
       c. Store structured JSON record
    4. Generate summary report
    """
    if config is None:
        config = PipelineConfig()

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not config.skip_download and not config.dry_run:
        print("\n📥  Step 1/4 — Downloading oxide CIFs from COD …\n")
        with tqdm(desc="Downloading CIFs", unit="file") as pbar:
            def _progress(current: int, total: int) -> None:
                pbar.total = total
                pbar.n = current
                pbar.refresh()

            download_cod_cifs(
                output_dir=config.cif_dir,
                limit=config.max_cifs,
                progress_callback=_progress,
            )
    else:
        if config.dry_run:
            print("\n🔬  Dry-run mode — skipping CIF download.\n")
        else:
            print("\n⏭️   Step 1/4 — Skipping download (using existing CIFs).\n")

    # ── Step 2: Filter ────────────────────────────────────────────────────
    print("🔍  Step 2/4 — Filtering perovskite structures …\n")

    if config.dry_run:
        # In dry-run mode, create a fake entry
        perovskites = [("dry_run_001", None, "LaCoO3")]
    else:
        perovskites = list(tqdm(
            filter_cifs(config.cif_dir),
            desc="Filtering CIFs",
            unit="struct",
        ))

    if config.max_cifs is not None:
        perovskites = perovskites[:config.max_cifs]

    print(f"\n   ✅ Found {len(perovskites)} perovskite structures.\n")

    if not perovskites:
        logger.warning("No perovskite structures found. Pipeline exiting.")
        return

    # ── Step 3: Retrieve papers + extract entities ────────────────────────
    print("📚  Step 3/4 — Retrieving literature & extracting entities …\n")
    records: list[dict] = []

    for cif_id, structure, composition in tqdm(perovskites, desc="Processing", unit="CIF"):
        # Resume support: skip if already processed
        if record_exists(cif_id, config.output_dir):
            logger.debug("Record for %s already exists — skipping.", cif_id)
            continue

        # Retrieve papers
        if config.dry_run:
            raw_papers = _make_sample_papers()
        else:
            raw_papers = retrieve_papers(
                composition=composition,
                max_papers=config.max_papers,
                cache_dir=config.paper_cache_dir,
            )

        # Extract entities from each paper
        processed_papers = []
        for paper in raw_papers:
            abstract = paper.get("abstract", "")
            if not abstract:
                continue

            extracted = extract_all(abstract)
            processed_papers.append({
                "doi": paper.get("doi"),
                "title": paper.get("title", ""),
                "abstract": abstract,
                "year": paper.get("year"),
                "synthesis_method": extracted["synthesis_method"],
                "synthesis_confidence": extracted["synthesis_confidence"],
                "application": extracted["application"],
                "application_confidence": extracted["application_confidence"],
                "performance": extracted["performance"],
            })

        # Build record
        record = {
            "cif_id": cif_id,
            "composition": composition,
            "spacegroup": (
                structure.get_space_group_info()[0]
                if structure is not None
                else "unknown"
            ),
            "spacegroup_number": (
                structure.get_space_group_info()[1]
                if structure is not None
                else None
            ),
            "cif_path": str(config.cif_dir / f"{cif_id}.cif"),
            "papers": processed_papers,
        }

        save_record(record, config.output_dir)
        records.append(record)

    print(f"\n   ✅ Processed {len(records)} new CIF records.\n")

    # ── Step 4: Report ────────────────────────────────────────────────────
    print("📊  Step 4/4 — Generating summary report …\n")
    generate_report(output_dir=config.output_dir)

    print("🏁  Pipeline complete!\n")
