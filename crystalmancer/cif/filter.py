"""Perovskite CIF filter using pymatgen structure analysis."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Generator

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crystalmancer.config import (
    A_SITE_ELEMENTS,
    B_SITE_ELEMENTS,
    PEROVSKITE_SPACE_GROUPS,
)

logger = logging.getLogger(__name__)


def is_perovskite_spacegroup(structure: Structure, symprec: float = 0.1) -> bool:
    """Return True if the structure's space group number is in the perovskite set."""
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        sg_number = sga.get_space_group_number()
        return sg_number in PEROVSKITE_SPACE_GROUPS
    except Exception as exc:
        logger.debug("SpacegroupAnalyzer failed: %s", exc)
        return False


def is_perovskite_composition(structure: Structure) -> bool:
    """Check if the composition plausibly matches ABO₃ (or common variants).

    Validates:
    - Contains oxygen
    - Contains at least one A-site and one B-site element
    - A:B:O ratio roughly consistent with perovskite stoichiometry
      (1:1:3 for ABO₃, 2:2:7 for A₂B₂O₇, etc.)
    """
    comp = structure.composition
    elements = {str(el) for el in comp.elements}

    if "O" not in elements:
        return False

    a_present = elements & A_SITE_ELEMENTS
    b_present = elements & B_SITE_ELEMENTS

    if not a_present or not b_present:
        return False

    # Reduce formula to integer ratios
    reduced = comp.get_el_amt_dict()
    o_count = reduced.get("O", 0)
    if o_count == 0:
        return False

    # Sum A-site and B-site amounts
    a_total = sum(reduced.get(el, 0) for el in a_present)
    b_total = sum(reduced.get(el, 0) for el in b_present)

    if a_total == 0 or b_total == 0:
        return False

    # Normalize to oxygen count = 3
    norm = 3.0 / o_count
    a_norm = a_total * norm
    b_norm = b_total * norm

    # ABO₃:  A≈1, B≈1
    # A₂BO₄ (RP-1):  A≈1.5, B≈0.75  (looser)
    # Accept if A and B are each in [0.5, 2.5] when O is normalized to 3
    if 0.5 <= a_norm <= 2.5 and 0.5 <= b_norm <= 2.5:
        return True

    return False


def parse_cif(cif_path: Path) -> Structure | None:
    """Safely parse a CIF file into a pymatgen Structure."""
    try:
        structure = Structure.from_file(str(cif_path))
        return structure
    except Exception as exc:
        logger.debug("Could not parse %s: %s", cif_path.name, exc)
        return None


def filter_cifs(
    cif_dir: Path,
    require_composition: bool = True,
) -> Generator[tuple[str, Structure, str], None, None]:
    """Yield ``(cif_id, Structure, composition_string)`` for perovskite CIFs.

    Parameters
    ----------
    cif_dir : Path
        Directory containing CIF files (named ``{cod_id}.cif``).
    require_composition : bool
        If True (default), apply both space-group AND composition filters.
        If False, use space-group filter only.
    """
    cif_files = sorted(cif_dir.glob("*.cif"))
    logger.info("Scanning %d CIF files for perovskites …", len(cif_files))

    for cif_path in cif_files:
        structure = parse_cif(cif_path)
        if structure is None:
            continue

        if not is_perovskite_spacegroup(structure):
            continue

        if require_composition and not is_perovskite_composition(structure):
            continue

        cif_id = cif_path.stem
        composition = structure.composition.reduced_formula
        yield cif_id, structure, composition
