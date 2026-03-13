"""Unit tests for entity extraction (synthesis, application, performance)."""

import pytest

from crystalmancer.extraction.synthesis import classify_synthesis
from crystalmancer.extraction.application import classify_application
from crystalmancer.extraction.performance import extract_metrics
from crystalmancer.extraction.extractor import extract_all


# ── Synthesis Method Classification ───────────────────────────────────────────

class TestSynthesisClassifier:
    def test_sol_gel(self):
        text = "The catalyst was prepared by a sol-gel method using citric acid as chelating agent."
        method, conf = classify_synthesis(text)
        assert method == "sol-gel"
        assert conf > 0.5

    def test_hydrothermal(self):
        text = "Hydrothermal treatment was performed at 180°C in a Teflon-lined autoclave."
        method, conf = classify_synthesis(text)
        assert method == "hydrothermal"
        assert conf > 0.5

    def test_solid_state(self):
        text = "Samples were prepared by solid-state reaction followed by calcination at 900°C."
        method, conf = classify_synthesis(text)
        assert method == "solid-state"
        assert conf > 0.5

    def test_ald(self):
        text = "Thin films were deposited by atomic layer deposition (ALD) at 250°C."
        method, conf = classify_synthesis(text)
        assert method == "ALD"
        assert conf > 0.5

    def test_sputtering(self):
        text = "Films were grown using magnetron sputtering with RF power of 100W."
        method, conf = classify_synthesis(text)
        assert method == "sputtering"
        assert conf > 0.5

    def test_electrodeposition(self):
        text = "The film was prepared by electrodeposition from an aqueous solution."
        method, conf = classify_synthesis(text)
        assert method == "electrodeposition"
        assert conf > 0.5

    def test_coprecipitation(self):
        text = "Nanoparticles were synthesized via coprecipitation using NaOH."
        method, conf = classify_synthesis(text)
        assert method == "coprecipitation"
        assert conf > 0.5

    def test_combustion(self):
        text = "Oxide was obtained by solution combustion synthesis using glycine-nitrate precursors."
        method, conf = classify_synthesis(text)
        assert method == "combustion"
        assert conf > 0.5

    def test_no_match(self):
        text = "We studied the electronic properties of the material."
        method, conf = classify_synthesis(text)
        assert method == "other"
        assert conf == 0.0


# ── Application Type Classification ──────────────────────────────────────────

class TestApplicationClassifier:
    def test_oer(self):
        text = "was tested for the oxygen evolution reaction in alkaline media"
        app, conf = classify_application(text)
        assert app == "OER"
        assert conf > 0.5

    def test_her(self):
        text = "showed excellent hydrogen evolution reaction performance"
        app, conf = classify_application(text)
        assert app == "HER"
        assert conf > 0.5

    def test_co2rr(self):
        text = "electrocatalytic CO2 reduction to CO with high selectivity"
        app, conf = classify_application(text)
        assert app == "CO2RR"
        assert conf >= 0.5

    def test_photocatalysis(self):
        text = "photocatalytic degradation of methylene blue under visible light irradiation"
        app, conf = classify_application(text)
        assert app == "photocatalysis"
        assert conf > 0.5

    def test_thermochemical(self):
        text = "thermochemical water splitting via chemical looping with oxygen carrier"
        app, conf = classify_application(text)
        assert app == "thermochemical"
        assert conf > 0.5

    def test_no_match(self):
        text = "crystal structure determination by X-ray diffraction"
        app, conf = classify_application(text)
        assert app == "other"
        assert conf == 0.0


# ── Performance Metric Extraction ─────────────────────────────────────────────

class TestPerformanceExtractor:
    def test_overpotential(self):
        text = "The catalyst showed an overpotential of 320 mV at 10 mA cm⁻²."
        metrics = extract_metrics(text)
        assert metrics["overpotential_mV"] == 320

    def test_eta_notation(self):
        text = "η = 280 mV for the OER at a current density of 10 mA/cm²."
        metrics = extract_metrics(text)
        assert metrics["overpotential_mV"] == 280

    def test_faradaic_efficiency(self):
        text = "achieved a Faradaic efficiency of 95.2% for CO production"
        metrics = extract_metrics(text)
        assert metrics["faradaic_efficiency_pct"] == 95.2

    def test_tafel_slope(self):
        text = "with a Tafel slope of 62 mV/dec"
        metrics = extract_metrics(text)
        assert metrics["tafel_slope_mV_dec"] == 62

    def test_current_density(self):
        text = "at a current density of 10 mA cm⁻²"
        metrics = extract_metrics(text)
        assert metrics["current_density_mA_cm2"] == 10

    def test_stability(self):
        text = "stability for 24 h of continuous operation"
        metrics = extract_metrics(text)
        assert metrics["stability_h"] == 24

    def test_out_of_bounds_overpotential(self):
        """Values outside sanity bounds should be rejected."""
        text = "overpotential of 5000 mV"
        metrics = extract_metrics(text)
        assert metrics["overpotential_mV"] is None

    def test_no_metrics(self):
        text = "The structure was characterized by XRD and Raman spectroscopy."
        metrics = extract_metrics(text)
        assert all(v is None for v in metrics.values())

    def test_multiple_metrics(self):
        text = (
            "The LaCoO3 catalyst exhibited an overpotential of 310 mV "
            "at 10 mA cm⁻² with a Tafel slope of 55 mV/dec. "
            "Stability was maintained for 48 h."
        )
        metrics = extract_metrics(text)
        assert metrics["overpotential_mV"] == 310
        assert metrics["current_density_mA_cm2"] == 10
        assert metrics["tafel_slope_mV_dec"] == 55
        assert metrics["stability_h"] == 48


# ── Unified Extractor ────────────────────────────────────────────────────────

class TestExtractAll:
    def test_full_abstract(self):
        text = (
            "LaCoO3 perovskite was synthesized via sol-gel method using citric acid. "
            "The catalyst achieved an overpotential of 350 mV at 10 mA cm⁻² "
            "with a Tafel slope of 62 mV/dec for the oxygen evolution reaction. "
            "Stability was maintained for 24 h of continuous operation."
        )
        result = extract_all(text)
        assert result["synthesis_method"] == "sol-gel"
        assert result["application"] == "OER"
        assert result["performance"]["overpotential_mV"] == 350
        assert result["performance"]["tafel_slope_mV_dec"] == 62
        assert result["performance"]["stability_h"] == 24
