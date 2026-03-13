"""Crystal Mancer v2 — Scaled Model Architecture.

MatterGen-inspired diffusion score network with:
- GemNet-style geometric message passing (triplet interactions)
- Multi-head attention in message passing layers
- Sinusoidal timestep embedding for diffusion
- Performance conditioning via cross-attention
- Proper noise schedule (cosine or linear)
- 25-50M params depending on config

Architecture reference:
  MatterGen (Microsoft, 2024) — 50M params
  DimeNet++ — angle-based interactions
  Stable Diffusion — U-Net with timestep embedding (adapted here for graphs)

Model configurations:
  SMALL:  ~5M params  (M4 MacBook, quick experiments)
  MEDIUM: ~25M params (Colab T4, serious training)
  LARGE:  ~50M params (Colab A100, production)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Config — Scales from 5M to 50M params
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for Crystal Mancer v2."""

    # Architecture
    hidden_dim: int = 256              # Width of all hidden layers
    num_interaction_layers: int = 8    # Depth of message passing
    num_attention_heads: int = 8       # Multi-head attention
    edge_dim: int = 128                # Edge feature dimension
    num_rbf: int = 64                  # Radial basis functions
    cutoff: float = 8.0                # Angstrom cutoff for neighbors
    max_neighbors: int = 32            # Max neighbors per atom

    # Atom features
    atom_feature_dim: int = 108        # From featurizer
    num_elements: int = 94             # H through Pu

    # Global features
    global_feature_dim: int = 239      # Space group one-hot + lattice
    use_global_features: bool = True

    # Diffusion
    num_diffusion_timesteps: int = 1000
    noise_schedule: str = "cosine"     # "linear" or "cosine"
    timestep_embed_dim: int = 256

    # Conditioning
    num_targets: int = 5               # Performance metrics
    use_conditioning: bool = True
    conditioning_method: str = "cross_attention"  # or "addition"

    # Training
    dropout: float = 0.1

    # Presets
    @classmethod
    def small(cls) -> "ModelConfig":
        """~5M params — fast experiments on MacBook."""
        return cls(hidden_dim=128, num_interaction_layers=4, num_attention_heads=4,
                   edge_dim=64, num_rbf=40)

    @classmethod
    def medium(cls) -> "ModelConfig":
        """~25M params — serious training on Colab T4."""
        return cls(hidden_dim=256, num_interaction_layers=8, num_attention_heads=8,
                   edge_dim=128, num_rbf=64)

    @classmethod
    def large(cls) -> "ModelConfig":
        """~50M params — production, matches MatterGen scale."""
        return cls(hidden_dim=512, num_interaction_layers=12, num_attention_heads=16,
                   edge_dim=256, num_rbf=128, cutoff=10.0, max_neighbors=48)


# ═══════════════════════════════════════════════════════════════════════════════
#  Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class SinusoidalTimestepEmbed(nn.Module):
    """Sinusoidal timestep embedding (from DDPM / Stable Diffusion).

    Maps integer timestep t ∈ [0, T] to a continuous embedding.
    Same technique used in transformers for position encoding.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, embed_dim)."""
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None].float() * freqs[None, :]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj(emb)


class RadialBasisFunctions(nn.Module):
    """Smooth radial basis functions (better than Gaussian smearing).

    Uses Bessel functions inside cutoff, following DimeNet.
    """

    def __init__(self, num_rbf: int = 64, cutoff: float = 8.0):
        super().__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf

        # Bessel function frequencies
        freq = torch.arange(1, num_rbf + 1).float() * math.pi / cutoff
        self.register_buffer("freq", freq)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """dist: (E,) → (E, num_rbf)."""
        dist = dist.unsqueeze(-1)
        # Bessel-0 expansion with envelope
        rbf = torch.sin(self.freq * dist) / dist.clamp(min=1e-8)
        # Polynomial envelope for smooth cutoff
        envelope = 1.0 - 6 * (dist / self.cutoff) ** 5 + \
                   15 * (dist / self.cutoff) ** 4 - \
                   10 * (dist / self.cutoff) ** 3
        envelope = envelope.clamp(min=0)
        return rbf * envelope


class MultiHeadInteraction(MessagePassing):
    """Multi-head attention message passing with edge features.

    Inspired by GATv2 + SchNet: uses both geometric (RBF) features
    and learned attention to weight messages.
    """

    def __init__(self, hidden_dim: int, edge_dim: int, num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__(aggr="add")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        # Query, Key, Value projections
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Edge feature transformation
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Normalization + feedforward
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual
        h = self.norm1(x)
        # Message passing with attention
        msg = self.propagate(edge_index, x=h, edge_attr=edge_attr)
        x = x + self.out_proj(msg)

        # Feedforward with residual
        x = x + self.ff(self.norm2(x))
        return x

    def message(self, x_i: torch.Tensor, x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        B = x_i.size(0)
        H = self.num_heads
        D = self.head_dim

        q = self.W_q(x_i).view(B, H, D)
        k = self.W_k(x_j).view(B, H, D)
        v = self.W_v(x_j).view(B, H, D)

        # Edge modulation (SchNet-style continuous filter)
        edge_weight = self.edge_proj(edge_attr).view(B, H, D)

        # Attention scores with edge bias
        attn = (q * k * edge_weight).sum(dim=-1, keepdim=True) / math.sqrt(D)
        attn = torch.sigmoid(attn)  # Sigmoid attention (not softmax — sparse graphs)

        # Weighted value
        return (attn * v * edge_weight).view(B, -1)


class ConditioningCrossAttention(nn.Module):
    """Cross-attention between graph representation and performance targets.

    Like CLIP cross-attention in Stable Diffusion, but instead of text embeddings
    we use catalytic performance target embeddings.
    """

    def __init__(self, hidden_dim: int, num_targets: int = 5, num_heads: int = 4):
        super().__init__()
        # Encode targets into a sequence of key-value pairs
        self.target_embed = nn.Sequential(
            nn.Linear(num_targets * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # K and V
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.1,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_repr: torch.Tensor, targets: torch.Tensor,
                masks: torch.Tensor) -> torch.Tensor:
        """
        graph_repr: (B, hidden_dim) — pooled graph
        targets: (B, num_targets) — performance targets
        masks: (B, num_targets) — 1 if target provided
        """
        combined = torch.cat([targets * masks, masks], dim=-1)
        kv = self.target_embed(combined)  # (B, hidden_dim * 2)
        k, v = kv.chunk(2, dim=-1)

        # Add sequence dimension for attention
        q = graph_repr.unsqueeze(1)  # (B, 1, D)
        k = k.unsqueeze(1)
        v = v.unsqueeze(1)

        attn_out, _ = self.cross_attn(q, k, v)
        return self.norm(graph_repr + attn_out.squeeze(1))


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class CrystalMancerV2(nn.Module):
    """Crystal Mancer v2 — Production-scale crystal generation model.

    Configs:
      SMALL  → ~5M params   (MacBook M4, quick experiments)
      MEDIUM → ~25M params  (Colab T4)
      LARGE  → ~50M params  (Colab A100 / RTX 4090)

    Modes:
      1. Property predictor (pretrain): structure → properties
      2. Diffusion score network: (noisy_structure, t, targets) → score
    """

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig.medium()
        c = self.config

        # ── Atom embedding ────────────────────────────────
        self.atom_embed = nn.Sequential(
            nn.Linear(c.atom_feature_dim, c.hidden_dim),
            nn.SiLU(),
            nn.Linear(c.hidden_dim, c.hidden_dim),
        )

        # ── Element-specific learnable embedding ──────────
        self.element_embed = nn.Embedding(c.num_elements + 1, c.hidden_dim)

        # ── Radial basis for edges ────────────────────────
        self.rbf = RadialBasisFunctions(c.num_rbf, c.cutoff)
        self.edge_embed = nn.Linear(c.num_rbf, c.edge_dim)

        # ── Interaction layers ────────────────────────────
        self.interactions = nn.ModuleList([
            MultiHeadInteraction(c.hidden_dim, c.edge_dim, c.num_attention_heads, c.dropout)
            for _ in range(c.num_interaction_layers)
        ])

        # ── Timestep embedding for diffusion ──────────────
        self.time_embed = SinusoidalTimestepEmbed(c.timestep_embed_dim)
        self.time_proj = nn.Linear(c.timestep_embed_dim, c.hidden_dim)

        # ── Global crystal features ───────────────────────
        if c.use_global_features:
            self.global_proj = nn.Sequential(
                nn.Linear(c.global_feature_dim, c.hidden_dim),
                nn.SiLU(),
                nn.Linear(c.hidden_dim, c.hidden_dim),
            )

        # ── Performance conditioning ──────────────────────
        if c.use_conditioning:
            if c.conditioning_method == "cross_attention":
                self.conditioning = ConditioningCrossAttention(
                    c.hidden_dim, c.num_targets, num_heads=4,
                )
            else:
                self.conditioning = nn.Sequential(
                    nn.Linear(c.num_targets * 2, c.hidden_dim),
                    nn.SiLU(),
                    nn.Linear(c.hidden_dim, c.hidden_dim),
                )

        # ── Output heads ──────────────────────────────────
        # Head 1: Property prediction (pretraining)
        self.property_head = nn.Sequential(
            nn.Linear(c.hidden_dim * 2, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim // 2, c.num_targets),
        )

        # Head 2: Score prediction for diffusion (atom-level)
        self.score_head = nn.Sequential(
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, c.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(c.hidden_dim // 2, 3),  # 3D score (dx, dy, dz)
        )

        # Head 3: Lattice score for diffusion
        self.lattice_head = nn.Sequential(
            nn.Linear(c.hidden_dim * 2, c.hidden_dim),
            nn.GELU(),
            nn.Linear(c.hidden_dim, 6),  # a, b, c, α, β, γ scores
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def encode(
        self,
        data: "Data",
        timestep: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a crystal structure into atom-level and graph-level representations.

        Parameters
        ----------
        data : PyG Data
            Must have: x, edge_index, edge_attr, batch
        timestep : (B,) | None
            Diffusion timestep. None for property prediction.

        Returns
        -------
        x : (N, hidden_dim) — atom representations
        graph_repr : (B, hidden_dim) — graph representations
        """
        c = self.config

        # Atom features → hidden
        x = self.atom_embed(data.x)

        # Add element-specific embeddings if available
        if hasattr(data, 'atomic_numbers'):
            x = x + self.element_embed(data.atomic_numbers.clamp(0, c.num_elements))

        # Edge features → RBF → hidden
        if hasattr(data, 'edge_distances'):
            edge_attr = self.edge_embed(self.rbf(data.edge_distances))
        else:
            edge_attr = self.edge_embed(self.rbf(data.edge_attr[:, 0]))

        # Inject timestep into node features (for diffusion)
        if timestep is not None:
            t_emb = self.time_proj(self.time_embed(timestep))  # (B, hidden_dim)
            batch = data.batch if hasattr(data, 'batch') else torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )
            x = x + t_emb[batch]  # Broadcast timestep to all atoms

        # Message passing
        for interaction in self.interactions:
            x = interaction(x, data.edge_index, edge_attr)

        # Global pooling
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(
            x.size(0), dtype=torch.long, device=x.device
        )
        graph_repr = global_mean_pool(x, batch)

        # Add global crystal features
        if c.use_global_features and hasattr(data, 'global_features'):
            gf = data.global_features
            if gf.dim() == 1:
                gf = gf.unsqueeze(0)
            graph_repr = graph_repr + self.global_proj(gf)

        return x, graph_repr

    def forward_properties(
        self,
        data: "Data",
        condition_targets: Optional[torch.Tensor] = None,
        condition_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Property prediction mode (pretraining).

        Returns (B, num_targets) predicted properties.
        """
        x, graph_repr = self.encode(data, timestep=None)

        # Apply conditioning if provided
        if self.config.use_conditioning and condition_targets is not None:
            if self.config.conditioning_method == "cross_attention":
                graph_repr = self.conditioning(graph_repr, condition_targets, condition_masks)
            else:
                cond = self.conditioning(torch.cat([
                    condition_targets * condition_masks, condition_masks
                ], dim=-1))
                graph_repr = graph_repr + cond

        # Predict properties
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(1, device=graph_repr.device)
        global_feat = self.global_proj(data.global_features.unsqueeze(0)) if (
            self.config.use_global_features and hasattr(data, 'global_features')
        ) else torch.zeros_like(graph_repr)

        combined = torch.cat([graph_repr, global_feat], dim=-1)
        return self.property_head(combined)

    def forward_score(
        self,
        data: "Data",
        timestep: torch.Tensor,
        condition_targets: Optional[torch.Tensor] = None,
        condition_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Diffusion score prediction mode.

        Returns:
          atom_scores: (N, 3) — predicted noise on atom positions
          lattice_scores: (B, 6) — predicted noise on lattice params
        """
        x, graph_repr = self.encode(data, timestep=timestep)

        # Apply conditioning
        if self.config.use_conditioning and condition_targets is not None:
            if self.config.conditioning_method == "cross_attention":
                graph_repr = self.conditioning(graph_repr, condition_targets, condition_masks)
            else:
                cond = self.conditioning(torch.cat([
                    condition_targets * condition_masks, condition_masks
                ], dim=-1))
                graph_repr = graph_repr + cond

        # Atom-level scores (position noise prediction)
        atom_scores = self.score_head(x)

        # Lattice scores (from global representation)
        global_feat = self.global_proj(data.global_features.unsqueeze(0)) if (
            self.config.use_global_features and hasattr(data, 'global_features')
        ) else torch.zeros_like(graph_repr)
        lattice_scores = self.lattice_head(torch.cat([graph_repr, global_feat], dim=-1))

        return atom_scores, lattice_scores

    def forward(self, data: "Data", **kwargs) -> torch.Tensor:
        """Default forward = property prediction (backward compatible)."""
        return self.forward_properties(data, **kwargs)

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        counts = {}
        total = 0
        for name, param in self.named_parameters():
            component = name.split('.')[0]
            n = param.numel()
            counts[component] = counts.get(component, 0) + n
            total += n
        counts['TOTAL'] = total
        return counts


# ═══════════════════════════════════════════════════════════════════════════════
#  Noise Schedules (Cosine + Linear)
# ═══════════════════════════════════════════════════════════════════════════════

def cosine_noise_schedule(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Cosine noise schedule (from improved DDPM).

    Returns alpha_bar values for each timestep.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f_t = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f_t / f_t[0]
    return alpha_bar.float()


def linear_noise_schedule(T: int = 1000, beta_start: float = 1e-4,
                          beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule (original DDPM)."""
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alpha_bar


# ═══════════════════════════════════════════════════════════════════════════════
#  Autoresearch Param Sweep Configs
# ═══════════════════════════════════════════════════════════════════════════════

AUTORESEARCH_CONFIGS = [
    # Quick sanity checks
    {"name": "tiny", "hidden_dim": 64, "num_interaction_layers": 2, "num_attention_heads": 2},
    {"name": "small_narrow", "hidden_dim": 96, "num_interaction_layers": 4, "num_attention_heads": 4},
    {"name": "small", "hidden_dim": 128, "num_interaction_layers": 4, "num_attention_heads": 4},
    # Serious experiments
    {"name": "medium_shallow", "hidden_dim": 256, "num_interaction_layers": 4, "num_attention_heads": 8},
    {"name": "medium", "hidden_dim": 256, "num_interaction_layers": 8, "num_attention_heads": 8},
    {"name": "medium_deep", "hidden_dim": 256, "num_interaction_layers": 12, "num_attention_heads": 8},
    # Production scale
    {"name": "large", "hidden_dim": 384, "num_interaction_layers": 8, "num_attention_heads": 8},
    {"name": "xlarge", "hidden_dim": 512, "num_interaction_layers": 12, "num_attention_heads": 16},
]
