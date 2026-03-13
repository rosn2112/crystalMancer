"""Unified entity extraction pipeline.

Combines synthesis method, application type, and performance metric
extractors into a single function call per abstract.
"""

from __future__ import annotations

from typing import Any

from crystalmancer.extraction.synthesis import classify_synthesis
from crystalmancer.extraction.application import classify_application
from crystalmancer.extraction.performance import extract_metrics


def extract_all(text: str) -> dict[str, Any]:
    """Run all extractors on an abstract / text block.

    Returns
    -------
    dict
        {
            "synthesis_method": str,
            "synthesis_confidence": float,
            "application": str,
            "application_confidence": float,
            "performance": {
                "overpotential_mV": float | None,
                "faradaic_efficiency_pct": float | None,
                "tafel_slope_mV_dec": float | None,
                "current_density_mA_cm2": float | None,
                "stability_h": float | None,
            }
        }
    """
    synth_method, synth_conf = classify_synthesis(text)
    app_type, app_conf = classify_application(text)
    perf = extract_metrics(text)

    return {
        "synthesis_method": synth_method,
        "synthesis_confidence": synth_conf,
        "application": app_type,
        "application_confidence": app_conf,
        "performance": perf,
    }
