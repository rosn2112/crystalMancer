"""Atom and bond featurizers for crystal graph construction.

Converts pymatgen Structures into numeric feature vectors suitable
for GNN training via PyTorch Geometric.
"""

from __future__ import annotations

import numpy as np
from pymatgen.core import Structure, Element

# ── Element Property Lookup Tables ────────────────────────────────────────────

# Pauling electronegativity (0 for noble gases / missing)
_ELECTRONEG: dict[str, float] = {
    "H": 2.20, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Sc": 1.36,
    "Ti": 1.54, "V": 1.63, "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88,
    "Ni": 1.91, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "Ge": 2.01, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Rb": 0.82, "Sr": 0.95, "Y": 1.22, "Zr": 1.33,
    "Nb": 1.60, "Mo": 2.16, "Ru": 2.20, "Rh": 2.28, "Pd": 2.20, "Ag": 1.93,
    "Sn": 1.96, "Sb": 2.05, "Te": 2.10, "I": 2.66, "Cs": 0.79, "Ba": 0.89,
    "La": 1.10, "Ce": 1.12, "Pr": 1.13, "Nd": 1.14, "Sm": 1.17, "Eu": 1.20,
    "Gd": 1.20, "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Yb": 1.25, "Lu": 1.27,
    "Hf": 1.30, "Ta": 1.50, "W": 2.36, "Re": 1.90, "Ir": 2.20, "Pt": 2.28,
    "Au": 2.54, "Pb": 2.33, "Bi": 2.02,
}

# Ionic radii in Å (Shannon, for common oxidation states)
_IONIC_RADIUS: dict[str, float] = {
    "Li": 0.76, "Na": 1.02, "K": 1.38, "Rb": 1.52, "Cs": 1.67,
    "Ca": 1.00, "Sr": 1.18, "Ba": 1.35,
    "La": 1.03, "Ce": 0.87, "Pr": 0.99, "Nd": 0.98, "Sm": 0.96, "Eu": 0.95,
    "Gd": 0.94, "Dy": 0.91, "Y": 0.90,
    "Ti": 0.61, "V": 0.54, "Cr": 0.62, "Mn": 0.53, "Fe": 0.55, "Co": 0.55,
    "Ni": 0.56, "Cu": 0.57, "Zn": 0.74, "Al": 0.54, "Ga": 0.62, "Sn": 0.69,
    "Zr": 0.72, "Nb": 0.64, "Mo": 0.59, "Ru": 0.62, "Rh": 0.60, "Ir": 0.63,
    "Hf": 0.71, "Ta": 0.64, "W": 0.60, "Sb": 0.60, "Bi": 1.03, "Pb": 1.19,
    "O": 1.40,
}

MAX_ATOMIC_NUM = 94  # up to Pu


def atom_features(element_str: str) -> np.ndarray:
    """Compute feature vector for a single atom.

    Features (total ~107-dim):
    - One-hot element embedding (94-dim)
    - Atomic number (1-dim, normalized)
    - Period (1-dim, normalized)
    - Group (1-dim, normalized)
    - Electronegativity (1-dim, normalized)
    - Ionic radius (1-dim, normalized)
    - Atomic mass (1-dim, normalized)
    - Number of valence electrons (1-dim, normalized)
    - Block encoding (4-dim: s, p, d, f)
    - Is transition metal (1-dim)
    - Is rare earth (1-dim)
    - Is oxygen (1-dim)
    """
    el = Element(element_str)

    # One-hot element
    one_hot = np.zeros(MAX_ATOMIC_NUM, dtype=np.float32)
    if el.Z <= MAX_ATOMIC_NUM:
        one_hot[el.Z - 1] = 1.0

    # Scalar features
    atomic_num = el.Z / MAX_ATOMIC_NUM
    period = (el.row - 1) / 6.0  # rows 1-7 → 0-1
    group = (el.group - 1) / 17.0 if el.group else 0.0  # groups 1-18

    electroneg = _ELECTRONEG.get(element_str, 0.0) / 4.0  # max ~4 Pauling
    ionic_r = _IONIC_RADIUS.get(element_str, 0.0) / 2.0  # max ~2 Å

    mass = el.atomic_mass / 250.0  # rough normalization
    n_valence = _count_valence(el) / 8.0

    # Block encoding
    block = np.zeros(4, dtype=np.float32)
    block_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    if el.block in block_map:
        block[block_map[el.block]] = 1.0

    is_tm = 1.0 if el.is_transition_metal else 0.0
    is_re = 1.0 if el.is_rare_earth_metal else 0.0
    is_o = 1.0 if element_str == "O" else 0.0

    return np.concatenate([
        one_hot,           # 94
        [atomic_num],      # 1
        [period],          # 1
        [group],           # 1
        [electroneg],      # 1
        [ionic_r],         # 1
        [mass],            # 1
        [n_valence],       # 1
        block,             # 4
        [is_tm],           # 1
        [is_re],           # 1
        [is_o],            # 1
    ]).astype(np.float32)  # total: 108


def _count_valence(el: Element) -> int:
    """Estimate valence electrons (simplified)."""
    if el.block == "s":
        return el.group if el.group else 0
    elif el.block == "p":
        return el.group - 10 if el.group else 0
    elif el.block == "d":
        return min(el.group, 8) if el.group else 0
    elif el.block == "f":
        return min(el.Z - 56, 14) if el.Z > 56 else min(el.Z - 88, 14)
    return 0


class GaussianExpansion:
    """Expand scalar distances into Gaussian basis features.

    This is the standard radial basis expansion used in SchNet, DimeNet, etc.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 8.0,
        num_gaussians: int = 40,
    ):
        self.centers = np.linspace(start, stop, num_gaussians, dtype=np.float32)
        self.width = (stop - start) / num_gaussians  # σ
        self._gamma = -0.5 / (self.width ** 2)

    def expand(self, distances: np.ndarray) -> np.ndarray:
        """Expand distances (N,) → Gaussian features (N, num_gaussians)."""
        distances = np.asarray(distances, dtype=np.float32)
        if distances.ndim == 0:
            distances = distances.reshape(1)
        # (N, 1) - (1, G) → (N, G)
        diff = distances[:, None] - self.centers[None, :]
        return np.exp(self._gamma * diff ** 2).astype(np.float32)


def bond_features(distance: float, gaussian_expansion: GaussianExpansion) -> np.ndarray:
    """Compute feature vector for a bond/edge.

    Features:
    - Raw distance (1-dim, normalized)
    - Gaussian distance expansion (num_gaussians-dim)
    """
    dist_norm = np.array([distance / 8.0], dtype=np.float32)  # normalize to ~[0,1]
    gauss = gaussian_expansion.expand(np.array([distance]))[0]  # (G,)
    return np.concatenate([dist_norm, gauss]).astype(np.float32)
