"""Sentence-transformer embeddings + FAISS index for semantic search."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from crystalmancer.config import DEFAULT_OUTPUT_DIR

logger = logging.getLogger(__name__)

# Default embedding model (384-dim, fast, good quality)
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbeddingIndex:
    """FAISS-backed semantic search over paper abstracts."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self._model_name = model_name
        self._model = None
        self._index = None
        self._documents: list[dict[str, Any]] = []  # metadata for each vector

    def _load_model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.info("Loaded embedding model: %s", self._model_name)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into embeddings."""
        self._load_model()
        return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def build_from_records(self, records: list[dict[str, Any]]) -> int:
        """Build the FAISS index from Phase 1 JSON records.

        Indexes each (CIF, paper abstract) pair.

        Returns
        -------
        int
            Number of documents indexed.
        """
        import faiss

        texts: list[str] = []
        docs: list[dict] = []

        for record in records:
            cif_id = record.get("cif_id", "")
            composition = record.get("composition", "")

            for paper in record.get("papers", []):
                abstract = paper.get("abstract", "")
                if not abstract:
                    continue

                # Combine composition + abstract for richer embedding
                text = f"{composition}: {abstract}"
                texts.append(text)
                docs.append({
                    "cif_id": cif_id,
                    "composition": composition,
                    "doi": paper.get("doi"),
                    "title": paper.get("title", ""),
                    "synthesis_method": paper.get("synthesis_method"),
                    "application": paper.get("application"),
                    "abstract": abstract,
                })

        if not texts:
            logger.warning("No abstracts to index.")
            return 0

        # Encode all texts
        embeddings = self._encode(texts)
        dim = embeddings.shape[1]

        # Build FAISS index (inner product for normalized vectors = cosine similarity)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._documents = docs

        logger.info("Built FAISS index with %d documents (dim=%d).", len(docs), dim)
        return len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Search for the most similar documents to a query.

        Returns
        -------
        list[dict]
            Top-k results, each with keys: score, cif_id, composition,
            doi, title, synthesis_method, application, abstract.
        """
        if self._index is None or not self._documents:
            logger.warning("Index is empty. Call build_from_records() first.")
            return []

        query_vec = self._encode([query]).astype(np.float32)
        scores, indices = self._index.search(query_vec, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx].copy()
            doc["score"] = float(score)
            results.append(doc)

        return results

    def save(self, path: Path) -> None:
        """Save the index and metadata to disk."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._index is not None:
            faiss.write_index(self._index, str(path / "faiss.index"))

        meta_path = path / "documents.json"
        meta_path.write_text(
            json.dumps(self._documents, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Saved embedding index to %s", path)

    def load(self, path: Path) -> None:
        """Load the index and metadata from disk."""
        import faiss

        path = Path(path)
        index_path = path / "faiss.index"
        meta_path = path / "documents.json"

        if index_path.exists():
            self._index = faiss.read_index(str(index_path))

        if meta_path.exists():
            self._documents = json.loads(meta_path.read_text(encoding="utf-8"))

        logger.info(
            "Loaded embedding index: %d documents.",
            len(self._documents),
        )
