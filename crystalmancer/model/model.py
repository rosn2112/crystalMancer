"""Crystal Mancer GNN model architecture.

SchNet-inspired message-passing network with performance conditioning
for catalytic property prediction + crystal structure generation.

⚠️ Note: This is the public scaffold. The full conditioning mechanism
and score network internals are withheld.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GaussianSmearing(nn.Module):
    """Learnable Gaussian distance expansion (SchNet-style)."""

    def __init__(self, start: float = 0.0, stop: float = 8.0, num_gaussians: int = 40):
        super().__init__()
        centers = torch.linspace(start, stop, num_gaussians)
        self.register_buffer("centers", centers)
        self.width = (stop - start) / num_gaussians

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        diff = distances.unsqueeze(-1) - self.centers
        return torch.exp(-0.5 * (diff / self.width) ** 2)


class InteractionBlock(MessagePassing):
    """Single message-passing layer with continuous-filter convolution."""

    def __init__(self, hidden_dim: int, edge_dim: int):
        super().__init__(aggr="add")
        self.mlp_edge = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mlp_node = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.mlp_node(out)
        return self.layernorm(x + out)  # residual connection

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        W = self.mlp_edge(edge_attr)
        return x_j * W


class ConditioningAdapter(nn.Module):
    """Inject performance target conditioning into the GNN.

    Encodes performance targets as a conditioning vector and adds it
    to the global graph representation at each layer.

    ⚠️ Simplified version — full architecture uses attention-based
    cross-conditioning at multiple levels.
    """

    def __init__(self, num_targets: int = 5, hidden_dim: int = 128):
        super().__init__()
        # Encode each target metric + a mask indicating which targets are provided
        self.target_encoder = nn.Sequential(
            nn.Linear(num_targets * 2, hidden_dim),  # values + masks
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        targets : (B, num_targets) — target performance values (0 for missing)
        masks : (B, num_targets) — 1 if target provided, 0 if missing

        Returns
        -------
        (B, hidden_dim) conditioning vector
        """
        combined = torch.cat([targets * masks, masks], dim=-1)
        return self.target_encoder(combined)


class CrystalMancerGNN(nn.Module):
    """Crystal Mancer GNN for property prediction.

    Architecture:
    1. Atom embedding → hidden dim
    2. N interaction (message-passing) blocks
    3. Global pooling → graph-level representation
    4. Performance conditioning injection (optional)
    5. Property prediction heads

    Parameters
    ----------
    atom_feature_dim : int
        Input atom feature dimension (108 from featurizer).
    edge_feature_dim : int
        Input edge feature dimension (41 from featurizer).
    hidden_dim : int
        Hidden dimension throughout the network.
    num_layers : int
        Number of message-passing layers.
    num_targets : int
        Number of performance prediction targets.
    global_feature_dim : int
        Dimension of global crystal features (239 from graph_builder).
    use_conditioning : bool
        If True, enable performance conditioning adapter.
    """

    def __init__(
        self,
        atom_feature_dim: int = 108,
        edge_feature_dim: int = 41,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_targets: int = 5,
        global_feature_dim: int = 239,
        use_conditioning: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Atom embedding
        self.atom_embed = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Interaction blocks
        self.interactions = nn.ModuleList([
            InteractionBlock(hidden_dim, edge_feature_dim)
            for _ in range(num_layers)
        ])

        # Global feature projection
        self.global_proj = nn.Sequential(
            nn.Linear(global_feature_dim, hidden_dim),
            nn.SiLU(),
        )

        # Conditioning adapter (optional)
        self.use_conditioning = use_conditioning
        if use_conditioning:
            self.conditioning = ConditioningAdapter(num_targets, hidden_dim)

        # Output heads — predict each performance metric
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # graph + global concat
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        data: "Data",
        condition_targets: Optional[torch.Tensor] = None,
        condition_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        data : PyG Data object
            Must contain: x, edge_index, edge_attr, global_features, batch.
        condition_targets : (B, num_targets) | None
            Performance targets for conditioning (generation mode).
        condition_masks : (B, num_targets) | None
            Masks for condition_targets.

        Returns
        -------
        (B, num_targets) predicted performance values
        """
        x = self.atom_embed(data.x)

        # Message passing
        for interaction in self.interactions:
            x = interaction(x, data.edge_index, data.edge_attr)

        # Global pooling
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_repr = global_mean_pool(x, batch)  # (B, hidden_dim)

        # Global features
        if hasattr(data, "global_features"):
            gf = data.global_features
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
            global_proj = self.global_proj(gf)  # (B, hidden_dim)
        else:
            global_proj = torch.zeros_like(graph_repr)

        # Conditioning injection
        if self.use_conditioning and condition_targets is not None:
            cond = self.conditioning(condition_targets, condition_masks)
            graph_repr = graph_repr + cond

        # Concat graph + global and predict
        combined = torch.cat([graph_repr, global_proj], dim=-1)  # (B, 2 * hidden_dim)
        predictions = self.output_head(combined)  # (B, num_targets)

        return predictions
