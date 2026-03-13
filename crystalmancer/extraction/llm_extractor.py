"""LLM-powered entity extraction from scientific abstracts.

Uses OpenRouter free models to extract structured catalysis data.
Falls back to rule-based extraction on failure.
"""

from __future__ import annotations

import logging
from typing import Any

from crystalmancer.config import METRIC_BOUNDS
from crystalmancer.extraction.llm_client import extract_json
from crystalmancer.extraction.extractor import extract_all as rule_based_extract

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a materials science expert specializing in electrocatalysis and photocatalysis.

Your task: Extract structured information from a scientific abstract about a catalytic material.

Return ONLY valid JSON with this exact schema:
{
  "synthesis_method": one of ["hydrothermal", "sol-gel", "solid-state", "sputtering", "ALD", "electrodeposition", "coprecipitation", "combustion", "spray-pyrolysis", "other"],
  "application": one of ["OER", "HER", "CO2RR", "ORR", "NRR", "photocatalysis", "thermochemical", "water-splitting", "other"],
  "performance": {
    "overpotential_mV": float or null (overpotential in millivolts),
    "faradaic_efficiency_pct": float or null (Faradaic efficiency as percentage),
    "tafel_slope_mV_dec": float or null (Tafel slope in mV/decade),
    "current_density_mA_cm2": float or null (current density in mA/cm²),
    "stability_h": float or null (stability duration in hours)
  },
  "conditions": string or null (e.g., "1M KOH, pH 14"),
  "support_material": string or null (e.g., "Ni foam", "glassy carbon"),
  "morphology": string or null (e.g., "nanoparticles", "thin film", "nanosheets"),
  "synthesis_temperature_C": float or null (synthesis temperature in °C),
  "synthesis_summary": string or null (brief synthesis description)
}

Rules:
- Use null for any field you cannot confidently extract
- Only extract numeric values explicitly stated in the text
- Do NOT infer or calculate values
- Overpotential should be in mV (convert from V if needed)
- Current density should be in mA/cm² (convert units if needed)
"""

_USER_PROMPT_TEMPLATE = """Extract structured catalysis data from this abstract:

---
{abstract}
---

Return only the JSON object, no other text."""


def _validate_metrics(performance: dict) -> dict[str, float | None]:
    """Apply sanity bounds to extracted metrics."""
    validated: dict[str, float | None] = {}
    for key in [
        "overpotential_mV",
        "faradaic_efficiency_pct",
        "tafel_slope_mV_dec",
        "current_density_mA_cm2",
        "stability_h",
    ]:
        value = performance.get(key)
        if value is not None:
            try:
                value = float(value)
                lo, hi = METRIC_BOUNDS.get(key, (float("-inf"), float("inf")))
                if not (lo <= value <= hi):
                    logger.debug("Metric %s=%.2f out of bounds [%.1f, %.1f].", key, value, lo, hi)
                    value = None
            except (TypeError, ValueError):
                value = None
        validated[key] = value
    return validated


def llm_extract(
    abstract: str,
    model: str | None = None,
    fallback: bool = True,
) -> dict[str, Any]:
    """Extract entities from an abstract using an LLM.

    Parameters
    ----------
    abstract : str
        The paper abstract text.
    model : str | None
        Override OpenRouter model ID.
    fallback : bool
        If True, fall back to rule-based extraction on LLM failure.

    Returns
    -------
    dict
        Extraction results with keys: synthesis_method, application,
        performance, conditions, support_material, morphology, etc.
    """
    try:
        prompt = _USER_PROMPT_TEMPLATE.format(abstract=abstract)
        raw = extract_json(prompt, system_prompt=_SYSTEM_PROMPT, model=model)

        # Validate and normalize
        result = {
            "synthesis_method": raw.get("synthesis_method", "other"),
            "application": raw.get("application", "other"),
            "performance": _validate_metrics(raw.get("performance", {})),
            "conditions": raw.get("conditions"),
            "support_material": raw.get("support_material"),
            "morphology": raw.get("morphology"),
            "synthesis_temperature_C": raw.get("synthesis_temperature_C"),
            "synthesis_summary": raw.get("synthesis_summary"),
            "extraction_method": "llm",
        }

        logger.debug(
            "LLM extracted: synth=%s, app=%s",
            result["synthesis_method"],
            result["application"],
        )
        return result

    except Exception as exc:
        logger.warning("LLM extraction failed: %s", exc)
        if fallback:
            logger.info("Falling back to rule-based extraction.")
            rb = rule_based_extract(abstract)
            rb["extraction_method"] = "rule-based-fallback"
            rb["conditions"] = None
            rb["support_material"] = None
            rb["morphology"] = None
            rb["synthesis_temperature_C"] = None
            rb["synthesis_summary"] = None
            return rb
        raise
