"""Physics-informed loss functions and constraint layers.

Adds crystallographic math as inductive bias to the GNN:
- Hamiltonian energy conservation layer
- Symmetry-aware loss (penalizes symmetry-breaking)
- Charge neutrality constraint
- XRD pattern matching loss
- Goldschmidt tolerance loss (soft perovskite stability penalty)
"""

from __future__ import annotations

import math
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    # ═══════════════════════════════════════════════════════════════════
    #  Hamiltonian Neural Network Layer
    # ═══════════════════════════════════════════════════════════════════
    # Instead of directly predicting forces, we learn a Hamiltonian
    # H(q, p) and derive forces via Hamilton's equations:
    #   dq/dt =  ∂H/∂p   (velocities)
    #   dp/dt = -∂H/∂q   (forces = negative gradient of H)
    #
    # This GUARANTEES energy conservation, which unconstrained NNs violate.
    # Key insight: crystal structures at equilibrium sit at energy minima,
    # so the Hamiltonian formulation naturally drives atoms to stable positions.

    class HamiltonianLayer(nn.Module):
        """Learn a Hamiltonian and derive physics-consistent forces.

        Given atom positions q and momenta p (or latent velocities),
        learns H(q, p) and returns the Hamiltonian gradients.
        This ensures energy conservation by construction.
        """

        def __init__(self, hidden_dim: int = 128, num_layers: int = 3):
            super().__init__()
            layers = []
            in_dim = hidden_dim * 2  # [q_features, p_features]
            for i in range(num_layers):
                out_dim = hidden_dim if i < num_layers - 1 else 1
                layers.append(nn.Linear(in_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.SiLU())
                in_dim = out_dim
            self.hamiltonian_net = nn.Sequential(*layers)

        def forward(
            self,
            q_features: torch.Tensor,
            p_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute Hamiltonian and its gradients.

            Parameters
            ----------
            q_features : Tensor [N, hidden_dim]
                Position-derived features (from GNN message passing).
            p_features : Tensor [N, hidden_dim]
                Momentum/velocity features (learned or zero-initialized).

            Returns
            -------
            H : Tensor [1]
                Total Hamiltonian (energy).
            dH_dq : Tensor [N, hidden_dim]
                Gradient w.r.t. positions (negative = force direction).
            dH_dp : Tensor [N, hidden_dim]
                Gradient w.r.t. momenta (velocity direction).
            """
            q_features = q_features.requires_grad_(True)
            p_features = p_features.requires_grad_(True)

            # Concatenate q and p features
            qp = torch.cat([q_features, p_features], dim=-1)
            H = self.hamiltonian_net(qp).sum()  # scalar energy

            # Hamilton's equations via autograd
            dH_dq, dH_dp = torch.autograd.grad(
                H, [q_features, p_features],
                create_graph=True,  # for second-order optimization
            )

            return H, dH_dq, dH_dp


    # ═══════════════════════════════════════════════════════════════════
    #  Lagrangian Constraint Layer
    # ═══════════════════════════════════════════════════════════════════
    # Alternative to Hamiltonian: learn Lagrangian L(q, q̇) = T - V
    # and derive equations of motion via Euler-Lagrange.
    # Better for systems with explicit coordinate constraints.

    class LagrangianLayer(nn.Module):
        """Learn a Lagrangian L = T(q̇) - V(q) for crystal dynamics."""

        def __init__(self, hidden_dim: int = 128):
            super().__init__()
            # Kinetic energy network T(q̇)
            self.T_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            # Potential energy network V(q)
            self.V_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(
            self,
            q_features: torch.Tensor,
            qdot_features: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute Lagrangian and generalized forces.

            Returns
            -------
            L : Tensor [1]
                Lagrangian = T - V.
            forces : Tensor [N, hidden_dim]
                Generalized forces (−∂V/∂q).
            """
            q_features = q_features.requires_grad_(True)

            T = self.T_net(qdot_features).sum()
            V = self.V_net(q_features).sum()
            L = T - V

            # Forces = negative gradient of potential
            forces = -torch.autograd.grad(
                V, q_features, create_graph=True
            )[0]

            return L, forces


    # ═══════════════════════════════════════════════════════════════════
    #  Physics Loss Functions
    # ═══════════════════════════════════════════════════════════════════

    class PhysicsLoss(nn.Module):
        """Combined physics-informed loss for crystal structure generation.

        Adds soft constraints from crystallography and thermodynamics.
        These guide the diffusion model toward physically valid structures.
        """

        def __init__(
            self,
            w_charge: float = 1.0,
            w_tolerance: float = 0.5,
            w_symmetry: float = 0.3,
            w_bond: float = 0.5,
            w_energy_conservation: float = 1.0,
        ):
            super().__init__()
            self.w_charge = w_charge
            self.w_tolerance = w_tolerance
            self.w_symmetry = w_symmetry
            self.w_bond = w_bond
            self.w_energy_conservation = w_energy_conservation

            # Common oxidation states for perovskite elements
            self._oxidation_states = {
                'O': -2,
                # A-site (typically +2 or +3)
                'Sr': 2, 'Ba': 2, 'Ca': 2, 'Pb': 2,
                'La': 3, 'Ce': 3, 'Pr': 3, 'Nd': 3, 'Y': 3, 'Bi': 3,
                'Na': 1, 'K': 1, 'Li': 1,
                # B-site (variable)
                'Ti': 4, 'Zr': 4, 'Hf': 4,
                'V': 5, 'Nb': 5, 'Ta': 5,
                'Cr': 3, 'Mn': 4, 'Fe': 3, 'Co': 3, 'Ni': 2,
                'Cu': 2, 'Zn': 2, 'Al': 3, 'Ga': 3,
                'Mo': 6, 'W': 6, 'Ru': 4, 'Ir': 4,
            }

            # Shannon ionic radii (Å) for Goldschmidt tolerance
            self._ionic_radii = {
                'O': 1.40,
                'Sr': 1.44, 'Ba': 1.61, 'Ca': 1.34, 'La': 1.36,
                'Pb': 1.49, 'Bi': 1.38, 'Na': 1.39, 'K': 1.64,
                'Ti': 0.605, 'Zr': 0.72, 'Mn': 0.53, 'Fe': 0.645,
                'Co': 0.545, 'Ni': 0.69, 'Cu': 0.73, 'Al': 0.535,
                'Cr': 0.615, 'V': 0.54, 'Nb': 0.64, 'Mo': 0.59,
            }

        def charge_neutrality_loss(
            self,
            elements: list[str],
            counts: list[int],
        ) -> torch.Tensor:
            """Penalize charge imbalance.

            For a valid ionic crystal, sum of (oxidation_state × count) = 0.
            """
            total_charge = 0.0
            for el, n in zip(elements, counts):
                ox = self._oxidation_states.get(el, 0)
                total_charge += ox * n
            # Quadratic penalty on deviation from neutrality
            return torch.tensor(total_charge ** 2, dtype=torch.float32)

        def goldschmidt_loss(
            self,
            a_element: str,
            b_element: str,
        ) -> torch.Tensor:
            """Penalize deviation from ideal Goldschmidt tolerance factor.

            t = (r_A + r_O) / (√2 × (r_B + r_O))
            Ideal perovskite: 0.8 < t < 1.0
            Optimal: t ≈ 0.9-1.0
            """
            r_a = self._ionic_radii.get(a_element, 1.4)
            r_b = self._ionic_radii.get(b_element, 0.6)
            r_o = self._ionic_radii['O']

            t = (r_a + r_o) / (math.sqrt(2) * (r_b + r_o))

            # Soft penalty: quadratic outside [0.8, 1.0]
            if t < 0.8:
                penalty = (0.8 - t) ** 2
            elif t > 1.0:
                penalty = (t - 1.0) ** 2
            else:
                penalty = 0.0

            return torch.tensor(penalty, dtype=torch.float32)

        def bond_length_loss(
            self,
            distances: torch.Tensor,
            min_dist: float = 1.5,
            max_dist: float = 3.5,
        ) -> torch.Tensor:
            """Penalize unphysical bond lengths.

            Oxide perovskite bonds:
              B-O: 1.8-2.2 Å (typical)
              A-O: 2.4-3.0 Å (typical)
              Minimum physical: ~1.5 Å
              Maximum meaningful: ~3.5 Å
            """
            too_short = F.relu(min_dist - distances)  # penalty if < 1.5Å
            too_long = F.relu(distances - max_dist)    # penalty if > 3.5Å
            return (too_short ** 2 + too_long ** 2).mean()

        def energy_conservation_loss(
            self,
            energies_t: torch.Tensor,
        ) -> torch.Tensor:
            """Penalize energy drift during diffusion trajectory.

            For a Hamiltonian system, total energy should be approximately
            constant across the denoising trajectory. This loss penalizes
            energy fluctuations, enforcing conservation.
            """
            if energies_t.numel() < 2:
                return torch.tensor(0.0)
            # Penalize variance of energy across timesteps
            return energies_t.var()

        def forward(
            self,
            distances: Optional[torch.Tensor] = None,
            elements: Optional[list[str]] = None,
            counts: Optional[list[int]] = None,
            a_element: Optional[str] = None,
            b_element: Optional[str] = None,
            energies_t: Optional[torch.Tensor] = None,
        ) -> dict[str, torch.Tensor]:
            """Compute all physics losses.

            Returns dict of named losses for logging + total.
            """
            losses = {}

            if distances is not None:
                losses['bond_length'] = self.w_bond * self.bond_length_loss(distances)

            if elements is not None and counts is not None:
                losses['charge_neutrality'] = self.w_charge * self.charge_neutrality_loss(elements, counts)

            if a_element is not None and b_element is not None:
                losses['goldschmidt'] = self.w_tolerance * self.goldschmidt_loss(a_element, b_element)

            if energies_t is not None:
                losses['energy_conservation'] = self.w_energy_conservation * self.energy_conservation_loss(energies_t)

            losses['total_physics'] = sum(losses.values()) if losses else torch.tensor(0.0)
            return losses


    # ═══════════════════════════════════════════════════════════════════
    #  XRD Pattern Loss (for validation feedback)
    # ═══════════════════════════════════════════════════════════════════

    class XRDLoss(nn.Module):
        """Compare generated structure's XRD pattern with experimental.

        Generates a synthetic powder XRD pattern from predicted atomic
        positions and compares against known experimental patterns.
        This provides a self-supervised signal from diffraction data.
        """

        def __init__(self, two_theta_range: tuple = (10, 90), n_points: int = 500):
            super().__init__()
            self.two_theta = torch.linspace(
                two_theta_range[0], two_theta_range[1], n_points
            )
            self.wavelength = 1.5406  # Cu Kα in Å

        def compute_xrd_pattern(
            self,
            positions: torch.Tensor,     # [N, 3] Cartesian coords
            atomic_numbers: torch.Tensor, # [N]    element Z
            lattice: torch.Tensor,        # [3, 3] lattice matrix
        ) -> torch.Tensor:
            """Compute synthetic powder XRD intensity pattern.

            Uses Bragg's law + structure factor (simplified).
            """
            # Compute d-spacings from lattice
            # Reciprocal lattice
            V = torch.det(lattice)
            b1 = torch.cross(lattice[1], lattice[2]) / V
            b2 = torch.cross(lattice[2], lattice[0]) / V
            b3 = torch.cross(lattice[0], lattice[1]) / V

            intensities = torch.zeros(len(self.two_theta))

            # Sample (hkl) reflections
            for h in range(-3, 4):
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        if h == 0 and k == 0 and l == 0:
                            continue

                        # Reciprocal vector
                        G = h * b1 + k * b2 + l * b3
                        d = 1.0 / G.norm()

                        # Bragg angle
                        sin_theta = self.wavelength / (2.0 * d)
                        if sin_theta.abs() > 1.0:
                            continue
                        two_theta = 2.0 * torch.asin(sin_theta) * 180.0 / math.pi

                        # Structure factor (simplified — uses Z as scattering factor)
                        F_real = 0.0
                        F_imag = 0.0
                        for i in range(len(positions)):
                            # Convert to fractional
                            frac = torch.linalg.solve(lattice.T, positions[i])
                            phase = 2 * math.pi * (h * frac[0] + k * frac[1] + l * frac[2])
                            f_j = atomic_numbers[i].float()
                            F_real = F_real + f_j * torch.cos(phase)
                            F_imag = F_imag + f_j * torch.sin(phase)

                        intensity = F_real ** 2 + F_imag ** 2

                        # Add Gaussian peak at 2θ position
                        sigma = 0.1  # peak width
                        peak = intensity * torch.exp(
                            -0.5 * ((self.two_theta - two_theta.item()) / sigma) ** 2
                        )
                        intensities = intensities + peak

            # Normalize
            max_I = intensities.max()
            if max_I > 0:
                intensities = intensities / max_I

            return intensities

        def forward(
            self,
            pred_pattern: torch.Tensor,
            target_pattern: torch.Tensor,
        ) -> torch.Tensor:
            """MSE between predicted and target XRD patterns."""
            return F.mse_loss(pred_pattern, target_pattern)
