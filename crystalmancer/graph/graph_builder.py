"""Convert CIF / pymatgen Structure → PyTorch Geometric Data objects.

This module bridges pymatgen crystal structures and the GNN training
pipeline by constructing fully-featurized graph representations.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from crystalmancer.graph.featurizer import (
    atom_features,
    bond_features,
    GaussianExpansion,
)

logger = logging.getLogger(__name__)

# Default parameters
DEFAULT_RADIUS = 8.0       # Å — neighbor cutoff
DEFAULT_MAX_NEIGHBORS = 12  # max edges per atom
DEFAULT_NUM_GAUSSIANS = 40


def cif_to_graph(
    structure: Structure,
    radius: float = DEFAULT_RADIUS,
    max_neighbors: int = DEFAULT_MAX_NEIGHBORS,
    num_gaussians: int = DEFAULT_NUM_GAUSSIANS,
    performance_labels: dict[str, float | None] | None = None,
    metadata: dict[str, Any] | None = None,
) -> "Data":
    """Convert a pymatgen Structure to a PyTorch Geometric Data object.

    Parameters
    ----------
    structure : Structure
        The crystal structure (from CIF or pymatgen).
    radius : float
        Neighbor cutoff distance in Å.
    max_neighbors : int
        Maximum neighbors per atom (prevents very dense graphs).
    num_gaussians : int
        Number of Gaussian basis functions for distance expansion.
    performance_labels : dict | None
        Catalytic performance labels to attach as graph-level targets.
    metadata : dict | None
        Extra metadata (cif_id, composition, etc.) stored in Data object.

    Returns
    -------
    torch_geometric.data.Data
        Fully featurized crystal graph.
    """
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch and torch_geometric required. "
            "Install: pip install torch torch_geometric"
        )

    gauss_exp = GaussianExpansion(start=0.0, stop=radius, num_gaussians=num_gaussians)

    # ── Node features ─────────────────────────────────────────────────────
    node_feat_list = []
    for site in structure:
        el_str = str(site.specie)
        feat = atom_features(el_str)
        node_feat_list.append(feat)

    x = torch.tensor(np.array(node_feat_list), dtype=torch.float32)

    # ── Edge construction (neighbor list) ─────────────────────────────────
    all_neighbors = structure.get_all_neighbors(radius, include_index=True)
    src_list, dst_list, edge_feat_list = [], [], []

    for atom_idx, neighbors in enumerate(all_neighbors):
        # Sort by distance and keep max_neighbors closest
        neighbors = sorted(neighbors, key=lambda n: n[1])[:max_neighbors]
        for neighbor in neighbors:
            _, distance, neighbor_idx, _ = neighbor
            src_list.append(atom_idx)
            dst_list.append(neighbor_idx)
            feat = bond_features(distance, gauss_exp)
            edge_feat_list.append(feat)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_feat_list), dtype=torch.float32)
    else:
        # Isolated structure (shouldn't happen for crystals)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1 + num_gaussians), dtype=torch.float32)

    # ── Atomic positions (fractional coordinates) ─────────────────────────
    frac_coords = torch.tensor(
        structure.frac_coords, dtype=torch.float32
    )
    cart_coords = torch.tensor(
        structure.cart_coords, dtype=torch.float32
    )

    # ── Lattice parameters ────────────────────────────────────────────────
    lattice = structure.lattice
    lattice_params = torch.tensor([
        lattice.a / 20.0,       # normalize (typical range 3-15 Å)
        lattice.b / 20.0,
        lattice.c / 20.0,
        lattice.alpha / 180.0,  # normalize angles to [0, 1]
        lattice.beta / 180.0,
        lattice.gamma / 180.0,
    ], dtype=torch.float32)

    # Lattice matrix (for reconstructing structure)
    lattice_matrix = torch.tensor(
        lattice.matrix, dtype=torch.float32
    )

    # ── Space group features ──────────────────────────────────────────────
    try:
        sga = SpacegroupAnalyzer(structure)
        sg_number = sga.get_space_group_number()
    except Exception:
        sg_number = 0

    sg_onehot = torch.zeros(231, dtype=torch.float32)  # 1-230 + unknown (0)
    sg_onehot[sg_number] = 1.0

    # ── Global features ───────────────────────────────────────────────────
    vol_per_atom = structure.volume / len(structure)

    global_features = torch.cat([
        lattice_params,                         # 6
        sg_onehot,                              # 231
        torch.tensor([vol_per_atom / 50.0]),    # 1 (normalized)
        torch.tensor([len(structure) / 100.0]), # 1 (num atoms, normalized)
    ])  # total: 239

    # ── Performance labels (targets) ──────────────────────────────────────
    y_dict = {}
    if performance_labels:
        for key, value in performance_labels.items():
            if value is not None:
                y_dict[key] = torch.tensor([value], dtype=torch.float32)

    # ── Assemble Data object ──────────────────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=cart_coords,
        frac_coords=frac_coords,
        lattice_params=lattice_params,
        lattice_matrix=lattice_matrix,
        sg_onehot=sg_onehot,
        global_features=global_features,
        num_atoms=torch.tensor([len(structure)]),
    )

    # Attach performance targets
    for key, val in y_dict.items():
        setattr(data, key, val)

    # Attach metadata
    if metadata:
        for key, val in metadata.items():
            setattr(data, f"meta_{key}", val)

    return data


def structure_from_file(cif_path: str) -> Structure:
    """Load a pymatgen Structure from a CIF file."""
    return Structure.from_file(cif_path)
