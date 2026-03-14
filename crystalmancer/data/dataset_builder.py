"""Unified dataset builder — merges all data sources into one training set.

Sources:
  - COD (our perovskite-filtered CIFs + literature papers)
  - Materials Project (52K+ with DFT properties)
  - GNoME (stable crystals from DeepMind)

Deduplication strategy:
  - Normalize compositions to canonical reduced formula via pymatgen
  - Keep the BEST polymorph per canonical formula (most data, lowest energy)
  - Merge catalysis/literature data from duplicates into the best record

Outputs a single JSONL file ready for graph building + training.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)


def _canonical_formula(composition: str) -> str:
    """Normalize a composition string to canonical reduced formula.

    Uses pymatgen's Composition class for robust normalization:
      '  LaCoO3 ' → 'CoLaO3'
      'La0.5Sr0.5CoO3' → 'Co2La1Sr1O6' (reduced integer formula)
      'TiO2' → 'O2Ti'

    Falls back to stripped+lowered string if pymatgen fails.
    """
    if not composition or not composition.strip():
        return ""
    try:
        from pymatgen.core import Composition
        comp = Composition(composition.strip())
        return comp.reduced_formula
    except Exception:
        # Fallback: just strip whitespace and use as-is
        return composition.strip()


class DatasetBuilder:
    """Merge all data sources into a unified training dataset.

    The unified format includes:
    - material_id (unique identifier)
    - source (cod/materials_project/gnome)
    - composition (raw formula string)
    - canonical_formula (normalized via pymatgen for dedup)
    - spacegroup + spacegroup_number
    - cif_path (path to CIF file)
    - cif_string (inline CIF for portability)
    - formation_energy_per_atom (eV/atom, from DFT)
    - energy_above_hull (eV/atom)
    - band_gap (eV)
    - is_metal (bool)
    - lattice (dict with a, b, c, alpha, beta, gamma)
    - elements (list of element symbols)
    - nsites (number of atoms)
    - catalysis_data (dict from literature enrichment)
    """

    def __init__(self, output_dir: Path = DEFAULT_OUTPUT_DIR):
        self.output_dir = output_dir
        self.records: list[dict[str, Any]] = []
        self._seen_ids: set[str] = set()

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
                mid = f"cod_{data.get('cif_id', json_file.stem)}"
                if mid in self._seen_ids:
                    continue
                self._seen_ids.add(mid)
                record = {
                    "material_id": mid,
                    "source": "cod",
                    "composition": data.get("composition", ""),
                    "canonical_formula": _canonical_formula(data.get("composition", "")),
                    "spacegroup": data.get("spacegroup", ""),
                    "spacegroup_number": data.get("spacegroup_number", 0),
                    "cif_path": str(self.output_dir / "cifs" / f"{data.get('cif_id', '')}.cif"),
                    "cif_string": data.get("cif_string", ""),
                    "formation_energy_per_atom": None,
                    "energy_above_hull": None,
                    "band_gap": None,
                    "is_metal": None,
                    "lattice": data.get("lattice"),
                    "elements": data.get("elements", []),
                    "nsites": data.get("nsites", 0),
                    "catalysis_data": {
                        "papers": data.get("papers", []),
                    } if data.get("papers") else None,
                }
                self.records.append(record)
                count += 1
            except Exception as exc:
                logger.debug("Failed to load %s: %s", json_file, exc)

        logger.info("Added %d COD records.", count)
        return count

    def add_mp_records(self, mp_dir: Path | None = None) -> int:
        """Import Materials Project records.

        Checks both `mp_all_oxides.jsonl` (new name from download_all.py)
        and `mp_structures.jsonl` (legacy name) for compatibility.
        """
        mdir = mp_dir or self.output_dir / "materials_project"

        # Try new name first, fallback to legacy
        jsonl_file = mdir / "mp_all_oxides.jsonl"
        if not jsonl_file.exists():
            jsonl_file = mdir / "mp_structures.jsonl"
        if not jsonl_file.exists():
            logger.debug("No MP data found at %s", mdir)
            return 0

        count = 0
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    mid = data["material_id"]
                    if mid in self._seen_ids:
                        continue
                    self._seen_ids.add(mid)
                    record = {
                        "material_id": mid,
                        "source": "materials_project",
                        "composition": data.get("composition", ""),
                        "canonical_formula": _canonical_formula(data.get("composition", "")),
                        "spacegroup": data.get("spacegroup", ""),
                        "spacegroup_number": data.get("spacegroup_number", 0),
                        "cif_path": str(mdir / "cifs" / f"{mid}.cif"),
                        "cif_string": data.get("cif_string", ""),
                        "formation_energy_per_atom": data.get("formation_energy_per_atom"),
                        "energy_above_hull": data.get("energy_above_hull"),
                        "band_gap": data.get("band_gap"),
                        "is_metal": data.get("is_metal"),
                        "lattice": data.get("lattice"),
                        "elements": data.get("elements", []),
                        "nsites": data.get("nsites", 0),
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
            mid = f"gnome_{mat['material_id']}"
            if mid in self._seen_ids:
                continue
            self._seen_ids.add(mid)
            record = {
                "material_id": mid,
                "source": "gnome",
                "composition": mat.get("composition", ""),
                "canonical_formula": _canonical_formula(mat.get("composition", "")),
                "spacegroup": mat.get("spacegroup", ""),
                "spacegroup_number": mat.get("spacegroup_number", 0),
                "cif_path": str(gdir / "cifs" / f"{mat['material_id']}.cif"),
                "cif_string": "",
                "formation_energy_per_atom": mat.get("formation_energy_per_atom"),
                "energy_above_hull": mat.get("energy_above_hull"),
                "band_gap": mat.get("band_gap"),
                "is_metal": None,
                "lattice": mat.get("lattice"),
                "elements": mat.get("elements", []),
                "nsites": mat.get("nsites", 0),
                "catalysis_data": None,
            }
            self.records.append(record)
            count += 1

        logger.info("Added %d GNoME records.", count)
        return count

    def enrich_with_literature(self, literature_dir: Path | None = None) -> int:
        """Enrich structural records with catalytic data from mined papers.

        Matches papers to structures by canonical composition formula.
        A paper mentioning "LaCoO3" gets linked to all records with that
        canonical formula.

        Returns the number of records enriched.
        """
        lit_dir = literature_dir or self.output_dir / "literature"
        papers_file = lit_dir / "catalysis_papers.jsonl"

        if not papers_file.exists():
            logger.info("No literature data found at %s", papers_file)
            return 0

        # Load papers and index by canonical material formula
        paper_by_material: dict[str, list[dict]] = {}
        total_papers = 0

        with open(papers_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    total_papers += 1
                    for mat_name in paper.get("materials", []):
                        canonical = _canonical_formula(mat_name)
                        if canonical:
                            paper_by_material.setdefault(canonical, []).append({
                                "doi": paper.get("doi", ""),
                                "title": paper.get("title", ""),
                                "year": paper.get("year", 0),
                                "reaction_type": paper.get("reaction_type", ""),
                                "overpotential_mV": paper.get("overpotential_mV"),
                                "tafel_slope": paper.get("tafel_slope"),
                                "faradaic_efficiency": paper.get("faradaic_efficiency"),
                                "current_density": paper.get("current_density"),
                                "band_gap_eV": paper.get("band_gap_eV"),
                                "stability_hours": paper.get("stability_hours"),
                                "synthesis_method": paper.get("synthesis_method", ""),
                                "electrolyte": paper.get("electrolyte", ""),
                            })
                except Exception:
                    continue

        logger.info("Loaded %d papers covering %d unique materials.",
                    total_papers, len(paper_by_material))

        # Match papers to structures
        enriched = 0
        for rec in self.records:
            canonical = rec.get("canonical_formula", "")
            if not canonical:
                continue

            matching_papers = paper_by_material.get(canonical, [])
            if matching_papers:
                existing = rec.get("catalysis_data") or {}
                existing_papers = existing.get("papers", [])
                existing_dois = {p.get("doi") for p in existing_papers if p.get("doi")}

                new_papers = [p for p in matching_papers if p.get("doi") not in existing_dois]
                if new_papers:
                    if rec.get("catalysis_data") is None:
                        rec["catalysis_data"] = {"papers": []}
                    rec["catalysis_data"]["papers"].extend(new_papers)

                    # Also set top-level performance from best paper
                    best = max(new_papers,
                               key=lambda p: sum(1 for v in [
                                   p.get("overpotential_mV"),
                                   p.get("tafel_slope"),
                                   p.get("faradaic_efficiency"),
                               ] if v is not None))

                    if best.get("reaction_type"):
                        rec["catalysis_data"]["reaction_type"] = best["reaction_type"]
                    if best.get("overpotential_mV") is not None:
                        rec["catalysis_data"]["overpotential_mV"] = best["overpotential_mV"]

                    enriched += 1

        logger.info("Enriched %d structural records with literature data.", enriched)
        return enriched

    def deduplicate(self) -> int:
        """Smart deduplication: keep best polymorph per canonical formula.

        Strategy:
          1. Group records by canonical_formula
          2. For each group, score every record on data richness
          3. Keep the BEST one, merge catalysis data from others
          4. If two records for the same formula come from different sources
             (e.g., MP + GNoME), prefer the one with DFT energy data

        This eliminates redundant polymorphs while preserving the most
        data-rich representative for each unique composition.
        """
        by_formula: dict[str, list[dict]] = {}
        no_formula: list[dict] = []

        for rec in self.records:
            canon = rec.get("canonical_formula", "")
            if canon:
                by_formula.setdefault(canon, []).append(rec)
            else:
                no_formula.append(rec)

        deduped = []
        removed = 0
        for canon, recs in by_formula.items():
            if len(recs) == 1:
                deduped.append(recs[0])
            else:
                # Score: DFT energy > catalysis data > band gap > CIF string
                def score(r: dict) -> float:
                    s = 0.0
                    if r.get("formation_energy_per_atom") is not None:
                        s += 100
                    if r.get("energy_above_hull") is not None:
                        # Bonus for being very stable
                        ehull = r["energy_above_hull"]
                        if isinstance(ehull, (int, float)) and ehull < 0.01:
                            s += 50
                        s += 20
                    if r.get("catalysis_data"):
                        n_papers = len(r.get("catalysis_data", {}).get("papers", []))
                        s += 30 + min(n_papers * 5, 20)
                    if r.get("band_gap") is not None:
                        s += 10
                    if r.get("cif_string"):
                        s += 5
                    if r.get("lattice"):
                        s += 5
                    if r.get("nsites", 0) > 0:
                        s += 2
                    return s

                best = max(recs, key=score)

                # Merge catalysis data from all duplicates
                for r in recs:
                    if r is not best and r.get("catalysis_data"):
                        if best.get("catalysis_data") is None:
                            best["catalysis_data"] = {"papers": []}
                        existing_dois = {p.get("doi") for p in
                                         best["catalysis_data"].get("papers", [])
                                         if p.get("doi")}
                        for p in r.get("catalysis_data", {}).get("papers", []):
                            if p.get("doi") and p["doi"] not in existing_dois:
                                best["catalysis_data"]["papers"].append(p)
                                existing_dois.add(p["doi"])

                    # Fill missing DFT data from other sources
                    if r is not best:
                        if best.get("formation_energy_per_atom") is None and r.get("formation_energy_per_atom") is not None:
                            best["formation_energy_per_atom"] = r["formation_energy_per_atom"]
                        if best.get("band_gap") is None and r.get("band_gap") is not None:
                            best["band_gap"] = r["band_gap"]

                deduped.append(best)
                removed += len(recs) - 1

        deduped.extend(no_formula)
        self.records = deduped
        logger.info("Deduplicated: removed %d duplicates, %d unique records remain.",
                    removed, len(self.records))
        return removed

    def build(self) -> Path:
        """Build the unified dataset and save to JSONL."""
        output_file = self.output_dir / "unified_dataset.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Deduplicate
        n_removed = self.deduplicate()

        # Write
        with open(output_file, "w", encoding="utf-8") as f:
            for rec in self.records:
                f.write(json.dumps(rec, default=str) + "\n")

        # Stats
        sources: dict[str, int] = {}
        has_energy = 0
        has_catalysis = 0
        has_bandgap = 0
        has_cif = 0
        for rec in self.records:
            src = rec["source"]
            sources[src] = sources.get(src, 0) + 1
            if rec.get("formation_energy_per_atom") is not None:
                has_energy += 1
            if rec.get("catalysis_data"):
                has_catalysis += 1
            if rec.get("band_gap") is not None:
                has_bandgap += 1
            if rec.get("cif_string") or rec.get("cif_path"):
                has_cif += 1

        logger.info("=" * 60)
        logger.info("  UNIFIED DATASET SUMMARY")
        logger.info("=" * 60)
        logger.info("  Total unique records:  %d", len(self.records))
        logger.info("  Duplicates removed:    %d", n_removed)
        for src, count in sorted(sources.items()):
            logger.info("    %s: %d", src, count)
        logger.info("  With DFT energy:       %d (%.1f%%)",
                    has_energy, 100 * has_energy / max(len(self.records), 1))
        logger.info("  With band gap:         %d (%.1f%%)",
                    has_bandgap, 100 * has_bandgap / max(len(self.records), 1))
        logger.info("  With catalysis data:   %d (%.1f%%)",
                    has_catalysis, 100 * has_catalysis / max(len(self.records), 1))
        logger.info("  With CIF:              %d (%.1f%%)",
                    has_cif, 100 * has_cif / max(len(self.records), 1))
        logger.info("  Saved to: %s", output_file)
        logger.info("=" * 60)

        return output_file

    @property
    def stats(self) -> dict[str, Any]:
        """Return dataset statistics."""
        sources: dict[str, int] = {}
        for rec in self.records:
            src = rec["source"]
            sources[src] = sources.get(src, 0) + 1
        return {
            "total": len(self.records),
            "by_source": sources,
            "has_energy": sum(1 for r in self.records if r.get("formation_energy_per_atom") is not None),
            "has_catalysis": sum(1 for r in self.records if r.get("catalysis_data")),
            "has_bandgap": sum(1 for r in self.records if r.get("band_gap") is not None),
        }
