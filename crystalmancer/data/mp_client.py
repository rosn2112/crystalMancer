"""Materials Project bulk data client.

Downloads crystal structures WITH computed properties from Materials Project.
This gives us DFT-computed formation energies, band gaps, and more — 
exactly what we need to pretrain the GNN on real physics.

Requires: pip install mp-api
API key: free from https://materialsproject.org/api
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

DEFAULT_MP_DIR = DEFAULT_OUTPUT_DIR / "materials_project"


def download_mp_structures(
    output_dir: Path = DEFAULT_MP_DIR,
    elements: list[str] | None = None,
    num_elements: tuple[int, int] = (2, 5),
    energy_above_hull_max: float = 0.1,  # eV/atom — thermodynamically stable
    limit: int | None = None,
    api_key: str | None = None,
    include_oxides_only: bool = True,
) -> list[dict[str, Any]]:
    """Bulk download structures + properties from Materials Project.

    This is the GOLD STANDARD for training data because every structure
    comes with DFT-computed properties (energy, band gap, etc.).

    Parameters
    ----------
    output_dir : Path
        Where to save the downloaded data.
    elements : list[str] | None
        Filter to specific elements. None = all.
    num_elements : tuple
        (min, max) number of elements in the formula.
    energy_above_hull_max : float
        Maximum energy above convex hull (eV/atom). 0.1 = near-stable.
    limit : int | None
        Maximum structures to download. None = all matching.
    api_key : str | None
        Materials Project API key. Falls back to MP_API_KEY env var.
    include_oxides_only : bool
        If True, only download oxides (contains O).

    Returns
    -------
    list[dict]
        Structures with properties.
    """
    try:
        from mp_api.client import MPRester
    except ImportError:
        logger.error(
            "mp-api not installed. Run: pip install mp-api\n"
            "Then get a free API key: https://materialsproject.org/api"
        )
        return []

    import os
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        logger.error(
            "Materials Project API key required.\n"
            "1. Sign up free: https://materialsproject.org\n"
            "2. Get key from dashboard\n"
            "3. export MP_API_KEY='your_key_here'"
        )
        return []

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Connecting to Materials Project API …")

    records = []
    batch_file = output_dir / "mp_structures.jsonl"

    with MPRester(api_key) as mpr:
        # Build query
        query_kwargs: dict[str, Any] = {
            "num_elements": num_elements,
            "energy_above_hull": (None, energy_above_hull_max),
            "fields": [
                "material_id",
                "formula_pretty",
                "structure",
                "symmetry",
                "formation_energy_per_atom",
                "energy_above_hull",
                "band_gap",
                "is_metal",
                "density",
                "volume",
                "nsites",
            ],
        }

        if elements:
            query_kwargs["elements"] = elements
        if include_oxides_only:
            query_kwargs["elements"] = query_kwargs.get("elements", []) or []
            if "O" not in query_kwargs["elements"]:
                # Ensure oxygen is included
                query_kwargs["chemsys"] = None  # clear
                # Use formula filter instead
                pass

        logger.info("Querying Materials Project (this may take a minute) …")
        try:
            docs = mpr.materials.summary.search(**query_kwargs)
        except Exception as exc:
            logger.error("MP API query failed: %s", exc)
            return []

        logger.info("Retrieved %d structures from Materials Project.", len(docs))

        with open(batch_file, "w", encoding="utf-8") as f:
            for i, doc in enumerate(docs):
                if limit and i >= limit:
                    break

                try:
                    struct = doc.structure
                    record = {
                        "source": "materials_project",
                        "material_id": doc.material_id,
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
                        # Save structure as CIF
                        "cif_string": struct.to(fmt="cif"),
                        "lattice": {
                            "a": struct.lattice.a,
                            "b": struct.lattice.b,
                            "c": struct.lattice.c,
                            "alpha": struct.lattice.alpha,
                            "beta": struct.lattice.beta,
                            "gamma": struct.lattice.gamma,
                        },
                        "elements": [str(el) for el in struct.composition.elements],
                    }
                    records.append(record)
                    f.write(json.dumps(record, default=str) + "\n")
                except Exception as exc:
                    logger.debug("Failed to process %s: %s", getattr(doc, 'material_id', '?'), exc)
                    continue

                if (i + 1) % 1000 == 0:
                    logger.info("  Processed %d/%d structures", i + 1, len(docs))

    # Also save CIF files for the graph builder
    cif_dir = output_dir / "cifs"
    cif_dir.mkdir(exist_ok=True)
    for rec in records:
        cif_path = cif_dir / f"{rec['material_id']}.cif"
        if not cif_path.exists():
            cif_path.write_text(rec["cif_string"], encoding="utf-8")

    logger.info(
        "✅ Saved %d structures to %s (%d CIF files)",
        len(records), batch_file, len(records),
    )

    return records


def download_mp_oxide_catalysts(
    output_dir: Path = DEFAULT_MP_DIR,
    api_key: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Download specifically oxide materials relevant to catalysis.

    Targets: transition metal oxides, perovskites, spinels.
    """
    # Common catalyst elements (A-site + B-site + O)
    catalyst_elements = [
        ["Sr", "Ti", "O"],
        ["La", "Co", "O"],
        ["La", "Mn", "O"],
        ["Ba", "Ti", "O"],
        ["La", "Fe", "O"],
        ["La", "Ni", "O"],
        ["Sr", "Fe", "O"],
        ["Bi", "Fe", "O"],
        ["Ca", "Ti", "O"],
        ["Sr", "Co", "O"],
        # Spinels
        ["Co", "O"],
        ["Mn", "O"],
        ["Fe", "O"],
        ["Ni", "O"],
        ["Ti", "O"],
        ["Ir", "O"],
        ["Ru", "O"],
    ]

    all_records = []
    seen_ids = set()

    for elem_set in catalyst_elements:
        logger.info("Querying MP for %s …", "-".join(elem_set))
        recs = download_mp_structures(
            output_dir=output_dir,
            elements=elem_set,
            limit=limit,
            api_key=api_key,
            include_oxides_only=True,
        )
        for r in recs:
            if r["material_id"] not in seen_ids:
                seen_ids.add(r["material_id"])
                all_records.append(r)

    logger.info("Total unique MP catalyst structures: %d", len(all_records))
    return all_records
