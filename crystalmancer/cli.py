"""Crystal Mancer CLI entry point."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from crystalmancer.pipeline import PipelineConfig, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crystalmancer",
        description=(
            "Crystal Mancer Phase 1 — Knowledge Extraction & Dataset Construction.\n"
            "Build the CIF ↔ synthesis ↔ performance triplet database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cif-dir",
        type=Path,
        default=None,
        help="Directory to store/read CIF files (default: data/cifs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for output JSON records (default: data/output)",
    )
    parser.add_argument(
        "--max-cifs",
        type=int,
        default=None,
        help="Maximum number of CIFs to process (default: all)",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=5,
        help="Maximum papers to retrieve per CIF (default: 5)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip CIF download — use existing files in --cif-dir",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline with sample data (no API calls)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ── Logging ───────────────────────────────────────────────────────────
    log_level = logging.WARNING
    if args.verbose >= 2:
        log_level = logging.DEBUG
    elif args.verbose >= 1:
        log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── Config ────────────────────────────────────────────────────────────
    config = PipelineConfig(
        max_cifs=args.max_cifs,
        max_papers=args.max_papers,
        skip_download=args.skip_download,
        dry_run=args.dry_run,
    )
    if args.cif_dir is not None:
        config.cif_dir = args.cif_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir

    # ── Banner ────────────────────────────────────────────────────────────
    print(r"""
   ╔═══════════════════════════════════════════════════════════╗
   ║              🔮  C R Y S T A L   M A N C E R             ║
   ║         AI-Driven Catalyst Discovery Engine  v0.1        ║
   ╚═══════════════════════════════════════════════════════════╝
    """)

    # ── Run ───────────────────────────────────────────────────────────────
    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n\n⚡ Interrupted. Partial results are saved — rerun to resume.\n")
        sys.exit(130)
    except Exception as exc:
        logger = logging.getLogger(__name__)
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
