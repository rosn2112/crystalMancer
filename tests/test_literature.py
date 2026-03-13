"""Unit tests for literature API clients (mocked HTTP)."""

import json
import pytest
from unittest.mock import patch, MagicMock

from crystalmancer.literature.semantic_scholar import search_papers as s2_search
from crystalmancer.literature.crossref import search_papers as cr_search
from crystalmancer.literature.retriever import retrieve_papers, _deduplicate


# ── Semantic Scholar Tests ────────────────────────────────────────────────────

class TestSemanticScholar:
    @patch("crystalmancer.literature.semantic_scholar.requests.get")
    def test_basic_search(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "title": "LaCoO3 for OER",
                    "abstract": "We studied LaCoO3...",
                    "externalIds": {"DOI": "10.1234/test1"},
                    "year": 2023,
                },
                {
                    "title": "SrTiO3 photocatalysis",
                    "abstract": "SrTiO3 was tested...",
                    "externalIds": {"DOI": "10.1234/test2"},
                    "year": 2022,
                },
            ]
        }
        mock_get.return_value = mock_resp

        results = s2_search("LaCoO3 catalysis", limit=5)
        assert len(results) == 2
        assert results[0]["doi"] == "10.1234/test1"
        assert results[0]["title"] == "LaCoO3 for OER"

    @patch("crystalmancer.literature.semantic_scholar.requests.get")
    def test_empty_results(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        results = s2_search("nonexistent compound xyz123")
        assert results == []

    @patch("crystalmancer.literature.semantic_scholar.requests.get")
    def test_404_returns_empty(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        results = s2_search("anything")
        assert results == []


# ── CrossRef Tests ────────────────────────────────────────────────────────────

class TestCrossRef:
    @patch("crystalmancer.literature.crossref.requests.get")
    def test_basic_search(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "message": {
                "items": [
                    {
                        "DOI": "10.1234/cr1",
                        "title": ["CrossRef paper on LaCoO3"],
                        "abstract": "<p>Abstract text here</p>",
                        "published-print": {"date-parts": [[2023, 1]]},
                    }
                ]
            }
        }
        mock_get.return_value = mock_resp

        results = cr_search("LaCoO3", limit=5)
        assert len(results) == 1
        assert results[0]["doi"] == "10.1234/cr1"
        assert "<p>" not in results[0]["abstract"]  # JATS tags stripped


# ── Deduplication Tests ───────────────────────────────────────────────────────

class TestDeduplication:
    def test_doi_dedup(self):
        papers = [
            {"doi": "10.1234/a", "title": "Paper A", "abstract": "abc"},
            {"doi": "10.1234/a", "title": "Paper A duplicate", "abstract": "abc"},
            {"doi": "10.1234/b", "title": "Paper B", "abstract": "def"},
        ]
        result = _deduplicate(papers)
        assert len(result) == 2
        assert result[0]["title"] == "Paper A"
        assert result[1]["title"] == "Paper B"

    def test_title_dedup_no_doi(self):
        papers = [
            {"doi": None, "title": "Same Title", "abstract": "abc"},
            {"doi": None, "title": "Same Title", "abstract": "def"},
            {"doi": None, "title": "Different Title", "abstract": "ghi"},
        ]
        result = _deduplicate(papers)
        assert len(result) == 2


# ── Retriever Integration (mocked) ───────────────────────────────────────────

class TestRetriever:
    @patch("crystalmancer.literature.retriever.search_pubmed", return_value=[])
    @patch("crystalmancer.literature.retriever.search_core", return_value=[])
    @patch("crystalmancer.literature.retriever.search_europepmc", return_value=[])
    @patch("crystalmancer.literature.retriever.crossref.search_papers")
    @patch("crystalmancer.literature.retriever.semantic_scholar.search_papers")
    def test_retrieve_merges_sources(self, mock_s2, mock_cr, mock_epmc, mock_core, mock_pm):
        mock_s2.return_value = [
            {"doi": "10.1234/s2", "title": "S2 paper", "abstract": "...", "year": 2023}
        ]
        mock_cr.return_value = [
            {"doi": "10.1234/cr", "title": "CR paper", "abstract": "...", "year": 2022}
        ]
        # Use tmp cache dir to avoid side effects
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            results = retrieve_papers("LaCoO3", max_papers=5, cache_dir=Path(tmp))
        assert len(results) == 2
        dois = {r["doi"] for r in results}
        assert "10.1234/s2" in dois
        assert "10.1234/cr" in dois
