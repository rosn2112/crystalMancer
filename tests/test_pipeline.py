"""Integration test — pipeline dry-run with sample data."""

import json
import tempfile
from pathlib import Path

from crystalmancer.pipeline import PipelineConfig, run_pipeline


class TestPipelineDryRun:
    def test_dry_run_produces_output(self):
        """Full pipeline with --dry-run should produce valid JSON output."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config = PipelineConfig(
                cif_dir=tmp_path / "cifs",
                output_dir=tmp_path / "output",
                paper_cache_dir=tmp_path / "cache",
                dry_run=True,
            )
            run_pipeline(config)

            # Check output was written
            output_files = list((tmp_path / "output").glob("*.json"))
            assert len(output_files) >= 1

            # Validate JSON schema
            record = json.loads(output_files[0].read_text())
            assert "cif_id" in record
            assert "composition" in record
            assert "papers" in record
            assert isinstance(record["papers"], list)

            if record["papers"]:
                paper = record["papers"][0]
                assert "synthesis_method" in paper
                assert "application" in paper
                assert "performance" in paper
                perf = paper["performance"]
                assert "overpotential_mV" in perf

            # Check summary CSV was generated
            assert (tmp_path / "output" / "summary.csv").exists()
