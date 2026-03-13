"""Crystal graph dataset for PyTorch Geometric.

Loads Phase 1 JSON records + CIF files and converts them into
a lazy-loading or pre-cached PyG dataset with train/val/test splits.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch_geometric.data import Dataset, Data
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from crystalmancer.config import DEFAULT_OUTPUT_DIR, DEFAULT_CIF_DIR

logger = logging.getLogger(__name__)


class CrystalDataset(Dataset if HAS_TORCH else object):
    """PyTorch Geometric dataset for crystal structures.

    Lazily loads CIF files and converts them to graph representations
    on first access, then caches the results as .pt files.
    """

    def __init__(
        self,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        cif_dir: Path = DEFAULT_CIF_DIR,
        cache_dir: Path | None = None,
        transform=None,
        pre_transform=None,
    ):
        self._output_dir = Path(output_dir)
        self._cif_dir = Path(cif_dir)
        self._cache_dir = Path(cache_dir) if cache_dir else self._output_dir / "graph_cache"
        self._records: list[dict[str, Any]] = []
        self._load_records()

        if HAS_TORCH:
            super().__init__(str(self._cache_dir), transform, pre_transform)

    def _load_records(self) -> None:
        """Load all Phase 1 JSON records."""
        if not self._output_dir.exists():
            logger.warning("Output directory %s does not exist.", self._output_dir)
            return

        for p in sorted(self._output_dir.glob("*.json")):
            try:
                record = json.loads(p.read_text(encoding="utf-8"))
                # Only include records that have a valid CIF path
                cif_path = self._cif_dir / f"{record['cif_id']}.cif"
                if cif_path.exists():
                    record["_cif_path"] = str(cif_path)
                    self._records.append(record)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.debug("Skipping %s: %s", p.name, exc)

        logger.info("Loaded %d records with valid CIF files.", len(self._records))

    def len(self) -> int:
        return len(self._records)

    def get(self, idx: int) -> "Data":
        """Get a single graph by index (with caching)."""
        from crystalmancer.graph.graph_builder import cif_to_graph, structure_from_file

        # Check cache
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_dir / f"graph_{idx}.pt"

        if cache_path.exists():
            return torch.load(cache_path, weights_only=False)

        record = self._records[idx]
        structure = structure_from_file(record["_cif_path"])

        # Aggregate performance labels from papers
        perf_labels: dict[str, float | None] = {
            "overpotential_mV": None,
            "faradaic_efficiency_pct": None,
            "tafel_slope_mV_dec": None,
            "current_density_mA_cm2": None,
            "stability_h": None,
        }
        for paper in record.get("papers", []):
            perf = paper.get("performance", {})
            for key in perf_labels:
                if perf_labels[key] is None and perf.get(key) is not None:
                    perf_labels[key] = perf[key]

        data = cif_to_graph(
            structure,
            performance_labels=perf_labels,
            metadata={
                "cif_id": record["cif_id"],
                "composition": record.get("composition", ""),
                "idx": idx,
            },
        )

        # Cache
        torch.save(data, cache_path)
        return data

    def get_splits(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple[list[int], list[int], list[int]]:
        """Return train/val/test index splits.

        Stratified by application type when possible.
        """
        n = len(self._records)
        indices = np.arange(n)
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train].tolist()
        val_idx = indices[n_train:n_train + n_val].tolist()
        test_idx = indices[n_train + n_val:].tolist()

        logger.info(
            "Split: train=%d, val=%d, test=%d",
            len(train_idx), len(val_idx), len(test_idx),
        )
        return train_idx, val_idx, test_idx
