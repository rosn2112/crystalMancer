"""Training loop for Crystal Mancer GNN.

Supports:
- Mixed precision (AMP) for GPU efficiency on Colab
- Gradient accumulation for large effective batch sizes
- Checkpoint save/resume
- Early stopping on validation loss
- Configurable via TrainConfig dataclass
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np

try:
    from torch_geometric.loader import DataLoader
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

from crystalmancer.model.model import CrystalMancerGNN

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    # Model
    atom_feature_dim: int = 108
    edge_feature_dim: int = 41
    hidden_dim: int = 128
    num_layers: int = 4
    num_targets: int = 5
    global_feature_dim: int = 239
    use_conditioning: bool = False
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    num_epochs: int = 100
    grad_accumulation_steps: int = 1
    use_amp: bool = True  # automatic mixed precision
    max_grad_norm: float = 1.0

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_every: int = 5  # save checkpoint every N epochs
    resume_from: str | None = None

    # Logging
    log_every: int = 10  # log every N batches

    # Time budget (for autoresearch)
    max_training_time_seconds: float | None = None  # None = no limit

    # Target keys in order
    target_keys: list[str] = field(default_factory=lambda: [
        "overpotential_mV",
        "faradaic_efficiency_pct",
        "tafel_slope_mV_dec",
        "current_density_mA_cm2",
        "stability_h",
    ])


class Trainer:
    """Handles the full training loop."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self._build_model()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.num_epochs,
        )
        self.scaler = torch.amp.GradScaler(enabled=config.use_amp and self.device.type == "cuda")
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.start_epoch = 0
        self.history: list[dict[str, float]] = []

        if config.resume_from:
            self._load_checkpoint(config.resume_from)

    def _build_model(self) -> CrystalMancerGNN:
        cfg = self.config
        model = CrystalMancerGNN(
            atom_feature_dim=cfg.atom_feature_dim,
            edge_feature_dim=cfg.edge_feature_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_targets=cfg.num_targets,
            global_feature_dim=cfg.global_feature_dim,
            use_conditioning=cfg.use_conditioning,
            dropout=cfg.dropout,
        )
        return model.to(self.device)

    def _compute_loss(self, data, predictions: torch.Tensor) -> torch.Tensor:
        """Masked MSE loss — only compute loss for targets that exist."""
        losses = []
        for i, key in enumerate(self.config.target_keys):
            if hasattr(data, key):
                target = getattr(data, key).to(self.device)
                pred = predictions[:, i:i+1]
                # Only include non-NaN targets
                mask = ~torch.isnan(target)
                if mask.any():
                    losses.append(F.mse_loss(pred[mask], target[mask]))

        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def train_epoch(self, train_loader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.use_amp and self.device.type == "cuda",
            ):
                predictions = self.model(data)
                loss = self._compute_loss(data, predictions)
                loss = loss / self.config.grad_accumulation_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.grad_accumulation_steps
            n_batches += 1

            if batch_idx % self.config.log_every == 0 and batch_idx > 0:
                logger.info("  Batch %d | Loss: %.6f", batch_idx, loss.item())

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for data in val_loader:
            data = data.to(self.device)
            predictions = self.model(data)
            loss = self._compute_loss(data, predictions)
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_loader, val_loader) -> dict[str, Any]:
        """Run the full training loop.

        Returns
        -------
        dict
            Training results with best_val_loss, epochs_trained, history.
        """
        logger.info("Training on %s | Model params: %d",
                     self.device, sum(p.numel() for p in self.model.parameters()))

        start_time = time.time()
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.start_epoch, self.config.num_epochs):
            # Time budget check
            if self.config.max_training_time_seconds:
                elapsed = time.time() - start_time
                if elapsed > self.config.max_training_time_seconds:
                    logger.info("Time budget reached (%.0fs). Stopping.", elapsed)
                    break

            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            self.scheduler.step()

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            self.history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
                "time_s": epoch_time,
            })

            logger.info(
                "Epoch %3d | Train: %.6f | Val: %.6f | LR: %.2e | Time: %.1fs",
                epoch, train_loss, val_loss, lr, epoch_time,
            )

            # Early stopping check
            if val_loss < self.best_val_loss - self.config.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    logger.info("Early stopping at epoch %d.", epoch)
                    break

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(epoch, is_best=False)

        total_time = time.time() - start_time
        result = {
            "best_val_loss": self.best_val_loss,
            "epochs_trained": len(self.history),
            "total_time_s": total_time,
            "device": str(self.device),
            "history": self.history,
        }

        # Save training results
        results_path = self.config.checkpoint_dir / "training_results.json"
        results_path.write_text(json.dumps(result, indent=2, default=str))
        logger.info("Training complete. Best val loss: %.6f", self.best_val_loss)
        return result

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.__dict__,
            "history": self.history,
        }
        path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info("Saved best model (val_loss=%.6f)", self.best_val_loss)

    def _load_checkpoint(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.start_epoch = checkpoint.get("epoch", 0) + 1
        self.history = checkpoint.get("history", [])
        logger.info("Resumed from epoch %d (val_loss=%.6f)", self.start_epoch, self.best_val_loss)
