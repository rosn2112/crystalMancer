"""Structured JSON storage for CIF–synthesis–performance triplets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# ── JSON Schema Version ───────────────────────────────────────────────────────
SCHEMA_VERSION = "1.0.0"


def save_record(record: dict[str, Any], output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
    """Write a single CIF record as a JSON file.

    Parameters
    ----------
    record : dict
        Must contain at minimum ``cif_id`` key.
    output_dir : Path
        Directory to write into.

    Returns
    -------
    Path
        Path to the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cif_id = record["cif_id"]
    record["schema_version"] = SCHEMA_VERSION

    out_path = output_dir / f"{cif_id}.json"
    out_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    logger.debug("Saved record for %s → %s", cif_id, out_path)
    return out_path


def load_record(path: Path) -> dict[str, Any]:
    """Load a single JSON record."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_records(output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[dict[str, Any]]:
    """Load all JSON records from the output directory."""
    records: list[dict[str, Any]] = []
    if not output_dir.exists():
        return records
    for p in sorted(output_dir.glob("*.json")):
        try:
            records.append(load_record(p))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed record %s: %s", p.name, exc)
    return records


def record_exists(cif_id: str, output_dir: Path = DEFAULT_OUTPUT_DIR) -> bool:
    """Check if a record for *cif_id* already exists (for pipeline resume)."""
    return (output_dir / f"{cif_id}.json").exists()
