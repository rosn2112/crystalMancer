"""Tests for the data mining pipeline.

Tests:
  - Regex extraction of catalytic performance data
  - LLM extraction merge logic (mocked)
  - Canonical formula normalization
  - Dataset builder MP file path fallback
  - Literature enrichment matching
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ── Regex Extraction Tests ───────────────────────────────────────

class TestRegexExtraction:
    """Test extract_catalytic_data_regex from mine_literature.py."""

    def test_overpotential_extraction(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test",
            abstract="The overpotential of 320 mV was achieved at 10 mA/cm2 in 1M KOH.",
        )
        result = extract_catalytic_data_regex(paper)
        assert result.overpotential_mV == 320.0
        assert result.current_density == 10.0

    def test_tafel_slope(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test2",
            abstract="The Tafel slope of 65 mV/dec indicates favorable kinetics",
        )
        result = extract_catalytic_data_regex(paper)
        assert result.tafel_slope == 65.0

    def test_faradaic_efficiency(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test3",
            abstract="Faradaic efficiency of 92.5% for CO2 reduction to CO",
        )
        result = extract_catalytic_data_regex(paper)
        assert result.faradaic_efficiency == 92.5
        assert result.reaction_type == "CO2RR"

    def test_band_gap(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test4",
            abstract="SrTiO3 has a band gap of 3.2 eV suitable for water splitting",
        )
        result = extract_catalytic_data_regex(paper)
        assert result.band_gap_eV == 3.2
        assert result.reaction_type == "photocatalysis"

    def test_battery_reaction_type(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test5",
            abstract="LiCoO2 cathode material for lithium ion battery",
        )
        result = extract_catalytic_data_regex(paper)
        assert result.reaction_type == "battery"

    def test_nrr_reaction_type(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test6",
            abstract="Nitrogen reduction reaction to produce ammonia synthesis",
        )
        result = extract_catalytic_data_regex(paper)
        assert result.reaction_type == "NRR"

    def test_material_extraction(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test7",
            abstract="LaCoO3 perovskite and SrTiO3 were compared for OER activity",
        )
        result = extract_catalytic_data_regex(paper)
        assert len(result.materials) > 0
        assert result.reaction_type == "OER"

    def test_no_data_returns_empty(self):
        from scripts.mine_literature import PaperRecord, extract_catalytic_data_regex

        paper = PaperRecord(
            doi="10.1234/test8",
            abstract="This paper discusses general chemistry concepts.",
        )
        result = extract_catalytic_data_regex(paper)
        assert not result.has_performance_data()


# ── LLM Extraction Merge Tests ──────────────────────────────────

class TestLLMMerge:
    """Test that LLM results correctly fill gaps in regex results."""

    def test_merge_fills_gaps(self):
        from scripts.mine_literature import PaperRecord, _merge_llm_results

        paper = PaperRecord(
            doi="10.1234/merge1",
            overpotential_mV=320.0,  # regex found this
            materials=["LaCoO3"],
        )
        llm_data = {
            "materials": ["LaCoO3", "SrCoO3"],  # LLM found extra material
            "overpotential_mV": 315,  # LLM disagrees — should NOT override
            "tafel_slope_mV_dec": 65,  # LLM fills gap
            "reaction_type": "OER",
            "synthesis_method": "sol-gel",
        }
        _merge_llm_results(paper, llm_data)

        assert paper.overpotential_mV == 320.0  # Kept regex value
        assert paper.tafel_slope == 65.0  # LLM filled gap
        assert "SrCoO3" in paper.materials  # LLM added material
        assert paper.reaction_type == "OER"
        assert paper.synthesis_method == "sol-gel"
        assert paper.extraction_method == "regex+llm"

    def test_merge_empty_dict(self):
        from scripts.mine_literature import PaperRecord, _merge_llm_results

        paper = PaperRecord(doi="10.1234/merge2", overpotential_mV=300)
        _merge_llm_results(paper, {})
        assert paper.overpotential_mV == 300  # unchanged


# ── Canonical Formula Tests ──────────────────────────────────────

class TestCanonicalFormula:
    """Test _canonical_formula normalization."""

    def test_basic_formula(self):
        from crystalmancer.data.dataset_builder import _canonical_formula

        assert _canonical_formula("TiO2") == "TiO2"
        assert _canonical_formula("  TiO2  ") == "TiO2"

    def test_empty_string(self):
        from crystalmancer.data.dataset_builder import _canonical_formula

        assert _canonical_formula("") == ""
        assert _canonical_formula("  ") == ""

    def test_same_composition_different_format(self):
        from crystalmancer.data.dataset_builder import _canonical_formula

        # These should produce the same canonical formula
        f1 = _canonical_formula("SrTiO3")
        f2 = _canonical_formula("Sr1Ti1O3")
        assert f1 == f2


# ── Dataset Builder Tests ────────────────────────────────────────

class TestDatasetBuilder:
    """Test DatasetBuilder functionality."""

    def test_mp_file_path_fallback(self):
        """Builder should find MP data using either filename."""
        from crystalmancer.data.dataset_builder import DatasetBuilder

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mp_dir = tmp_path / "materials_project"
            mp_dir.mkdir()

            # Write with the NEW filename
            jsonl = mp_dir / "mp_all_oxides.jsonl"
            record = {
                "material_id": "mp-1234",
                "composition": "TiO2",
                "spacegroup": "P42/mnm",
                "spacegroup_number": 136,
                "formation_energy_per_atom": -3.5,
                "energy_above_hull": 0.0,
                "band_gap": 3.0,
                "is_metal": False,
            }
            jsonl.write_text(json.dumps(record) + "\n")

            builder = DatasetBuilder(output_dir=tmp_path)
            n = builder.add_mp_records()
            assert n == 1
            assert builder.records[0]["material_id"] == "mp-1234"
            assert builder.records[0]["canonical_formula"] == "TiO2"

    def test_mp_legacy_filename(self):
        """Builder should fallback to legacy mp_structures.jsonl."""
        from crystalmancer.data.dataset_builder import DatasetBuilder

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mp_dir = tmp_path / "materials_project"
            mp_dir.mkdir()

            # Write with the LEGACY filename
            jsonl = mp_dir / "mp_structures.jsonl"
            record = {
                "material_id": "mp-5678",
                "composition": "SrTiO3",
            }
            jsonl.write_text(json.dumps(record) + "\n")

            builder = DatasetBuilder(output_dir=tmp_path)
            n = builder.add_mp_records()
            assert n == 1

    def test_deduplication_keeps_best(self):
        """Dedup should keep the record with the most data."""
        from crystalmancer.data.dataset_builder import DatasetBuilder

        with tempfile.TemporaryDirectory() as tmp:
            builder = DatasetBuilder(output_dir=Path(tmp))
            
            # Two records for same composition — different polymorphs
            builder.records = [
                {
                    "material_id": "mp-1",
                    "source": "materials_project",
                    "composition": "TiO2",
                    "canonical_formula": "TiO2",
                    "formation_energy_per_atom": -3.5,
                    "energy_above_hull": 0.0,
                    "band_gap": 3.0,
                    "catalysis_data": None,
                },
                {
                    "material_id": "gnome-1",
                    "source": "gnome",
                    "composition": "TiO2",
                    "canonical_formula": "TiO2",
                    "formation_energy_per_atom": None,
                    "energy_above_hull": None,
                    "band_gap": None,
                    "catalysis_data": {"papers": [{"doi": "10.1234/paper1"}]},
                },
            ]

            n_removed = builder.deduplicate()
            assert n_removed == 1
            assert len(builder.records) == 1

            # Should keep MP record (has DFT energy) and merge catalysis
            best = builder.records[0]
            assert best["material_id"] == "mp-1"
            assert best["formation_energy_per_atom"] == -3.5
            # Catalysis data should be merged from duplicate
            assert best.get("catalysis_data") is not None
            assert len(best["catalysis_data"]["papers"]) == 1

    def test_literature_enrichment(self):
        """enrich_with_literature should match papers to structures."""
        from crystalmancer.data.dataset_builder import DatasetBuilder

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            lit_dir = tmp_path / "literature"
            lit_dir.mkdir()

            # Create fake paper about TiO2
            papers_file = lit_dir / "catalysis_papers.jsonl"
            paper = {
                "doi": "10.1234/tio2-oer",
                "title": "TiO2 for OER",
                "year": 2023,
                "materials": ["TiO2"],
                "reaction_type": "OER",
                "overpotential_mV": 350,
            }
            papers_file.write_text(json.dumps(paper) + "\n")

            # Create a structural record for TiO2
            builder = DatasetBuilder(output_dir=tmp_path)
            builder.records = [
                {
                    "material_id": "mp-1",
                    "source": "materials_project",
                    "composition": "TiO2",
                    "canonical_formula": "TiO2",
                    "formation_energy_per_atom": -3.5,
                    "catalysis_data": None,
                },
            ]

            n_enriched = builder.enrich_with_literature(lit_dir)
            assert n_enriched == 1
            assert builder.records[0]["catalysis_data"] is not None
            assert len(builder.records[0]["catalysis_data"]["papers"]) == 1
            assert builder.records[0]["catalysis_data"]["papers"][0]["doi"] == "10.1234/tio2-oer"


# ── PaperRecord Tests ────────────────────────────────────────────

class TestPaperRecord:
    def test_has_performance_data(self):
        from scripts.mine_literature import PaperRecord

        p1 = PaperRecord(doi="1", overpotential_mV=300)
        assert p1.has_performance_data() is True

        p2 = PaperRecord(doi="2")
        assert p2.has_performance_data() is False

    def test_to_dict_excludes_falsy(self):
        from scripts.mine_literature import PaperRecord

        p = PaperRecord(doi="10.1234/x", title="Test", overpotential_mV=300)
        d = p.to_dict()
        assert "doi" in d
        assert "title" in d
        assert "overpotential_mV" in d
        assert "abstract" not in d  # empty string → falsy → excluded
