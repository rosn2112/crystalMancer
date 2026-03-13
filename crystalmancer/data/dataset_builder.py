"""Unified dataset builder — merges all data sources into one training set.

Sources:
  - COD (our perovskite-filtered CIFs + literature papers)
  - Materials Project (150K+ with DFT properties)
  - GNoME (380K stable crystals from DeepMind)
  - NOMAD (additional DFT data)

Outputs a single JSONL file + CIF directory ready for graph building + training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


class DatasetBuilder:
    """Merge all data sources into a unified training dataset.

    The unified format includes:
    - material_id (unique identifier)
    - source (cod/materials_project/gnome/nomad)
    - composition (reduced formula)
    - spacegroup + spacegroup_number
    - cif_path (path to CIF file)
    - formation_energy_per_atom (eV/atom, from DFT)
    - energy_above_hull (eV/atom)
    - band_gap (eV)
    - is_metal (bool)
    - catalysis_data (dict from literature, if available)
    """

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.records: list[dict[str, Any]] = []
        self._seen_compositions: dict[str, str] = {}  # composition → material_id

    def add_cod_records(self, records_dir: Path | None = None) -> int:
        """Import Phase 1 COD records (with literature data)."""
        rdir = records_dir or self.output_dir / "records"
        if not rdir.exists():
            logger.debug("No COD records found at %s", rdir)
            return 0

        count = 0
        for json_file in sorted(rdir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                record = {
                    "material_id": f"cod_{data.get('cif_id', json_file.stem)}",
                    "source": "cod",
                    "composition": data.get("composition", ""),
                    "spacegroup": data.get("spacegroup", ""),
                    "spacegroup_number": data.get("spacegroup_number", 0),
                    "cif_path": str(self.output_dir / "cifs" / f"{data.get('cif_id', '')}.cif"),
                    # COD doesn't have DFT data, but has literature data
                    "formation_energy_per_atom": None,
                    "energy_above_hull": None,
                    "band_gap": None,
                    "is_metal": None,
                    "catalysis_data": {
                        "papers": data.get("papers", []),
                    },
                }
                self.records.append(record)
                count += 1
            except Exception as exc:
                logger.debug("Failed to load %s: %s", json_file, exc)

        logger.info("Added %d COD records.", count)
        return count

    def add_mp_records(self, mp_dir: Path | None = None) -> int:
        """Import Materials Project records."""
        mdir = mp_dir or self.output_dir / "materials_project"
        jsonl_file = mdir / "mp_structures.jsonl"

        if not jsonl_file.exists():
            logger.debug("No MP data found at %s", jsonl_file)
            return 0

        count = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    record = {
                        "material_id": data["material_id"],
                        "source": "materials_project",
                        "composition": data.get("composition", ""),
                        "spacegroup": data.get("spacegroup", ""),
                        "spacegroup_number": data.get("spacegroup_number", 0),
                        "cif_path": str(mdir / "cifs" / f"{data['material_id']}.cif"),
                        "formation_energy_per_atom": data.get("formation_energy_per_atom"),
                        "energy_above_hull": data.get("energy_above_hull"),
                        "band_gap": data.get("band_gap"),
                        "is_metal": data.get("is_metal"),
                        "catalysis_data": None,
                    }
                    self.records.append(record)
                    count += 1
                except Exception:
                    continue

        logger.info("Added %d Materials Project records.", count)
        return count

    def add_gnome_records(self, gnome_dir: Path | None = None) -> int:
        """Import GNoME records."""
        gdir = gnome_dir or self.output_dir / "gnome"

        try:
            from crystalmancer.data.gnome_client import load_gnome_summary
            materials = load_gnome_summary(output_dir=gdir)
        except Exception as exc:
            logger.debug("Failed to load GNoME data: %s", exc)
            return 0

        count = 0
        for mat in materials:
            record = {
                "material_id": f"gnome_{mat['material_id']}",
                "source": "gnome",
                "composition": mat.get("composition", ""),
                "spacegroup": mat.get("spacegroup", ""),
                "spacegroup_number": mat.get("spacegroup_number", 0),
                "cif_path": str(gdir / "cifs" / f"{mat['material_id']}.cif"),
                "formation_energy_per_atom": mat.get("formation_energy_per_atom"),
                "energy_above_hull": mat.get("energy_above_hull"),
                "band_gap": mat.get("band_gap"),
                "is_metal": None,
                "catalysis_data": None,
            }
            self.records.append(record)
            count += 1

        logger.info("Added %d GNoME records.", count)
        return count

    def deduplicate(self) -> int:
        """Remove duplicate compositions, keeping the record with most data."""
        by_comp: dict[str, list[dict]] = {}
        for rec in self.records:
            comp = rec["composition"]
            by_comp.setdefault(comp, []).append(rec)

        deduped = []
        removed = 0
        for comp, recs in by_comp.items():
            if len(recs) == 1:
                deduped.append(recs[0])
            else:
                # Prefer: has DFT energy > has catalysis data > first
                def score(r):
                    s = 0
                    if r.get("formation_energy_per_atom") is not None:
                        s += 10
                    if r.get("catalysis_data"):
                        s += 5
                    if r.get("band_gap") is not None:
                        s += 2
                    return s

                best = max(recs, key=score)
                # Merge catalysis data from other records
                for r in recs:
                    if r is not best and r.get("catalysis_data"):
                        if best.get("catalysis_data") is None:
                            best["catalysis_data"] = r["catalysis_data"]
                deduped.append(best)
                removed += len(recs) - 1

        self.records = deduped
        logger.info("Deduplicated: removed %d duplicates, %d unique records.", removed, len(self.records))
        return removed

    def build(self) -> Path:
        """Build the unified dataset and save to JSONL."""
        output_file = self.output_dir / "unified_dataset.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate
        self.deduplicate()

        # Write
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, default=str) + "\n")

        # Stats
        sources = {}
        has_energy = 0
        has_catalysis = 0
        for rec in self.records:
            src = rec["source"]
            sources[src] = sources.get(src, 0) + 1
            if rec.get("formation_energy_per_atom") is not None:
                has_energy += 1
            if rec.get("catalysis_data"):
                has_catalysis += 1

        logger.info("=" * 60)
        logger.info("  UNIFIED DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info("  Total records: %d", len(self.records))
        for src, count in sorted(sources.items()):
            logger.info("    %s: %d", src, count)
        logger.info("  With DFT energy: %d", has_energy)
        logger.info("  With catalysis data: %d", has_catalysis)
        logger.info("  Saved to: %s", output_file)
        logger.info("=" * 60)

        return output_file

    @property
    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        sources = {}
        for rec in self.records:
            src = rec["source"]
            sources[src] = sources.get(src, 0) + 1
        return {
            "total": len(self.records),
            "by_source": sources,
            "has_energy": sum(1 for r in self.records if r.get("formation_energy_per_atom") is not None),
            "has_catalysis": sum(1 for r in self.records if r.get("catalysis_data")),
        }
