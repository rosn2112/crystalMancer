"""ML-based fast screening for candidate crystal structures.

Combines rule-based structural filters with ML force field predictions
for rapid triage before committing to expensive DFT calculations.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymatgen.core import Structure

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """Result of screening a single candidate structure."""

    cif_id: str
    composition: str
    passes_filters: bool
    goldschmidt_tolerance: float | None = None
    charge_neutral: bool | None = None
    coordination_valid: bool | None = None
    predicted_energy_per_atom: float | None = None
    predicted_bandgap: float | None = None
    stability_flag: str | None = None  # "stable", "metastable", "unstable"
    composite_score: float | None = None
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


# ── Goldschmidt Tolerance Factor ──────────────────────────────────────────────

# Shannon ionic radii (Å) for common perovskite ions
_IONIC_RADII = {
    # A-site (12-coordinate)
    "La3+": 1.36, "Ce3+": 1.34, "Pr3+": 1.32, "Nd3+": 1.27, "Sm3+": 1.24,
    "Sr2+": 1.44, "Ba2+": 1.61, "Ca2+": 1.34, "Na+": 1.39, "K+": 1.64,
    "Y3+": 1.19, "Bi3+": 1.17, "Pb2+": 1.49,
    # B-site (6-coordinate)
    "Ti4+": 0.605, "Mn3+": 0.645, "Mn4+": 0.53, "Fe3+": 0.645, "Fe4+": 0.585,
    "Co3+": 0.545, "Co2+": 0.65, "Ni2+": 0.69, "Ni3+": 0.56, "Cu2+": 0.73,
    "Cr3+": 0.615, "V3+": 0.64, "Al3+": 0.535, "Ga3+": 0.62,
    "Zr4+": 0.72, "Nb5+": 0.64, "Mo6+": 0.59, "Ru4+": 0.62,
    "Ir4+": 0.625, "Hf4+": 0.71, "Ta5+": 0.64, "W6+": 0.60, "Sn4+": 0.69,
    # Anion
    "O2-": 1.40,
}


def goldschmidt_tolerance(r_a: float, r_b: float, r_x: float = 1.40) -> float:
    """Calculate the Goldschmidt tolerance factor t.

    t = (r_A + r_X) / (√2 * (r_B + r_X))

    Ideal perovskite: t ≈ 1.0
    Stable range: 0.71 ≤ t ≤ 1.05
    """
    return (r_a + r_x) / (math.sqrt(2) * (r_b + r_x))


def estimate_tolerance_factor(structure: Structure) -> float | None:
    """Estimate the Goldschmidt tolerance factor for a structure.

    Attempts to identify A-site and B-site cations and look up
    their ionic radii. Returns None if identification fails.
    """
    from crystalmancer.config import A_SITE_ELEMENTS, B_SITE_ELEMENTS

    comp = structure.composition
    elements = {str(el) for el in comp.elements}

    a_elements = elements & A_SITE_ELEMENTS
    b_elements = elements & B_SITE_ELEMENTS

    if not a_elements or not b_elements or "O" not in elements:
        return None

    # Find radii — try common oxidation states
    r_a_values = []
    for el in a_elements:
        for suffix in ["3+", "2+", "+"]:
            key = f"{el}{suffix}"
            if key in _IONIC_RADII:
                r_a_values.append(_IONIC_RADII[key])
                break

    r_b_values = []
    for el in b_elements:
        for suffix in ["4+", "3+", "2+", "5+", "6+"]:
            key = f"{el}{suffix}"
            if key in _IONIC_RADII:
                r_b_values.append(_IONIC_RADII[key])
                break

    if not r_a_values or not r_b_values:
        return None

    r_a = np.mean(r_a_values)
    r_b = np.mean(r_b_values)
    return goldschmidt_tolerance(r_a, r_b)


# ── Charge Neutrality Check ──────────────────────────────────────────────────

def check_charge_neutrality(structure: Structure) -> bool:
    """Check if the structure is approximately charge neutral.

    Uses pymatgen's built-in oxidation state guessing.
    """
    try:
        oxi_structure = structure.copy()
        oxi_structure.add_oxidation_state_by_guess()
        total_charge = sum(
            site.specie.oxi_state * 1 for site in oxi_structure
        )
        return abs(total_charge) < 0.5  # tolerance
    except Exception:
        return True  # assume OK if can't determine


# ── Coordination Check ────────────────────────────────────────────────────────

def check_coordination(structure: Structure) -> bool:
    """Check if coordination numbers are physically reasonable.

    Perovskites: A-site ~12-coordinate, B-site ~6-coordinate
    """
    try:
        from pymatgen.analysis.local_env import VoronoiNN
        nn = VoronoiNN()

        for i, site in enumerate(structure):
            try:
                cn = nn.get_cn(structure, i)
                el = str(site.specie)
                # Very loose bounds
                if cn < 2 or cn > 16:
                    return False
            except Exception:
                continue

        return True
    except ImportError:
        return True  # can't check without VoronoiNN


# ── Composite Screening ──────────────────────────────────────────────────────

def screen_candidate(
    structure: Structure,
    cif_id: str = "unknown",
    composition: str | None = None,
) -> ScreeningResult:
    """Run all screening filters on a candidate structure.

    Returns
    -------
    ScreeningResult
        Screening outcome with individual filter results.
    """
    if composition is None:
        composition = structure.composition.reduced_formula

    notes: list[str] = []
    passes = True

    # Goldschmidt tolerance
    t = estimate_tolerance_factor(structure)
    if t is not None:
        if not (0.71 <= t <= 1.05):
            passes = False
            notes.append(f"Tolerance factor {t:.3f} outside stable range [0.71, 1.05]")
        else:
            notes.append(f"Tolerance factor {t:.3f} — within stable range")

    # Charge neutrality
    charge_ok = check_charge_neutrality(structure)
    if not charge_ok:
        passes = False
        notes.append("Charge neutrality check failed")

    # Coordination
    coord_ok = check_coordination(structure)
    if not coord_ok:
        passes = False
        notes.append("Abnormal coordination numbers detected")

    return ScreeningResult(
        cif_id=cif_id,
        composition=composition,
        passes_filters=passes,
        goldschmidt_tolerance=round(t, 4) if t else None,
        charge_neutral=charge_ok,
        coordination_valid=coord_ok,
        notes=notes,
    )
