"""Summary reporting for Crystal Mancer pipeline output."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from crystalmancer.config import DEFAULT_OUTPUT_DIR
from crystalmancer.storage.json_store import load_all_records

logger = logging.getLogger(__name__)


def _flatten_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten nested records into one row per (CIF, paper) pair."""
    rows = []
    for rec in records:
        cif_id = rec.get("cif_id", "")
        composition = rec.get("composition", "")
        spacegroup = rec.get("spacegroup", "")

        papers = rec.get("papers", [])
        if not papers:
            rows.append({
                "cif_id": cif_id,
                "composition": composition,
                "spacegroup": spacegroup,
                "doi": None,
                "synthesis_method": None,
                "application": None,
                "overpotential_mV": None,
                "faradaic_efficiency_pct": None,
                "tafel_slope_mV_dec": None,
                "current_density_mA_cm2": None,
                "stability_h": None,
            })
        else:
            for paper in papers:
                perf = paper.get("performance", {})
                rows.append({
                    "cif_id": cif_id,
                    "composition": composition,
                    "spacegroup": spacegroup,
                    "doi": paper.get("doi"),
                    "synthesis_method": paper.get("synthesis_method"),
                    "application": paper.get("application"),
                    "overpotential_mV": perf.get("overpotential_mV"),
                    "faradaic_efficiency_pct": perf.get("faradaic_efficiency_pct"),
                    "tafel_slope_mV_dec": perf.get("tafel_slope_mV_dec"),
                    "current_density_mA_cm2": perf.get("current_density_mA_cm2"),
                    "stability_h": perf.get("stability_h"),
                })
    return rows


def generate_report(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    records: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    """Generate and print a summary report of the dataset.

    Parameters
    ----------
    output_dir : Path
        Where to find records and save the CSV summary.
    records : list[dict] | None
        If provided, use these records instead of loading from disk.

    Returns
    -------
    pd.DataFrame
        The flattened DataFrame.
    """
    if records is None:
        records = load_all_records(output_dir)

    if not records:
        logger.warning("No records found — nothing to report.")
        return pd.DataFrame()

    rows = _flatten_records(records)
    df = pd.DataFrame(rows)

    # ── Print Summary ─────────────────────────────────────────────────────
    total_cifs = df["cif_id"].nunique()
    total_papers = df["doi"].dropna().nunique()

    print("\n" + "=" * 60)
    print("  Crystal Mancer — Phase 1 Dataset Summary")
    print("=" * 60)
    print(f"\n  Total CIF structures:  {total_cifs}")
    print(f"  Total linked papers:   {total_papers}")

    # Application distribution
    print("\n  ── Application Distribution ──")
    app_counts = df["application"].value_counts(dropna=False)
    for app, count in app_counts.items():
        label = str(app) if app else "(none)"
        print(f"    {label:<20s} {int(count):>5d}")

    # Synthesis method distribution
    print("\n  ── Synthesis Method Distribution ──")
    synth_counts = df["synthesis_method"].value_counts(dropna=False)
    for method, count in synth_counts.items():
        label = str(method) if method else "(none)"
        print(f"    {label:<20s} {int(count):>5d}")

    # Space group distribution (top 10)
    print("\n  ── Top Space Groups ──")
    sg_counts = df["spacegroup"].value_counts().head(10)
    for sg, count in sg_counts.items():
        print(f"    {str(sg):<20s} {int(count):>5d}")

    # Performance coverage
    perf_cols = [
        "overpotential_mV",
        "faradaic_efficiency_pct",
        "tafel_slope_mV_dec",
        "current_density_mA_cm2",
        "stability_h",
    ]
    print("\n  ── Performance Metric Coverage ──")
    for col in perf_cols:
        if col in df.columns:
            non_null = df[col].notna().sum()
            print(f"    {col:<30s} {non_null:>5d} / {len(df)}")

    print("\n" + "=" * 60 + "\n")

    # Save CSV
    csv_path = output_dir / "summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Summary CSV saved to %s", csv_path)

    return df
