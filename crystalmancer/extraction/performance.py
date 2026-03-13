"""Rule-based extraction of catalytic performance metrics from abstract text.

Extracts numeric values for:
- overpotential (mV)
- Faradaic efficiency (%)
- Tafel slope (mV/dec)
- current density (mA/cm²)
- stability / durability (hours)

All results are validated against sanity bounds defined in config.
"""

from __future__ import annotations

import re
from typing import Any

from crystalmancer.config import METRIC_BOUNDS

# ── Regex patterns ────────────────────────────────────────────────────────────
# Each pattern captures a numeric value near a descriptive keyword.
# We use named groups for clarity.

_NUMBER = r"(?P<value>\d+\.?\d*)"
_APPROX = r"(?:(?:~|≈|approximately|about|ca\.?|∼)\s*)?"

_PATTERNS: dict[str, list[re.Pattern]] = {
    "overpotential_mV": [
        # "overpotential of 320 mV"
        re.compile(
            r"overpotential\s+(?:of\s+)?" + _APPROX + _NUMBER + r"\s*mv",
            re.IGNORECASE,
        ),
        # "η = 320 mV"
        re.compile(
            r"[ηη]\s*[=≈~:]\s*" + _APPROX + _NUMBER + r"\s*mv",
            re.IGNORECASE,
        ),
        # "320 mV overpotential"
        re.compile(
            _APPROX + _NUMBER + r"\s*mv\s+overpotential",
            re.IGNORECASE,
        ),
    ],
    "faradaic_efficiency_pct": [
        # "Faradaic efficiency of 95.2%"
        re.compile(
            r"faradaic\s+efficiency\s+(?:of\s+)?" + _APPROX + _NUMBER + r"\s*%",
            re.IGNORECASE,
        ),
        # "FE of 95%"
        re.compile(
            r"\bFE\s+(?:of\s+)?" + _APPROX + _NUMBER + r"\s*%",
        ),
        # "95% Faradaic efficiency"
        re.compile(
            _APPROX + _NUMBER + r"\s*%\s+faradaic\s+efficiency",
            re.IGNORECASE,
        ),
    ],
    "tafel_slope_mV_dec": [
        # "Tafel slope of 58 mV dec⁻¹" or "mV/dec"
        re.compile(
            r"tafel\s+slope\s+(?:of\s+)?" + _APPROX + _NUMBER + r"\s*mv[\s/·]*dec",
            re.IGNORECASE,
        ),
        # "58 mV dec⁻¹ Tafel slope"
        re.compile(
            _APPROX + _NUMBER + r"\s*mv[\s/·]*dec[^a-z]*tafel",
            re.IGNORECASE,
        ),
    ],
    "current_density_mA_cm2": [
        # "current density of 10 mA cm⁻²"
        re.compile(
            r"current\s+density\s+(?:of\s+)?" + _APPROX + _NUMBER + r"\s*ma[\s/·]*cm",
            re.IGNORECASE,
        ),
        # "at 10 mA cm⁻²"
        re.compile(
            r"at\s+" + _APPROX + _NUMBER + r"\s*ma[\s/·]*cm",
            re.IGNORECASE,
        ),
        # "10 mA/cm²"
        re.compile(
            _APPROX + _NUMBER + r"\s*ma[\s/·]*cm",
            re.IGNORECASE,
        ),
        # "j = 10 mA cm⁻²"
        re.compile(
            r"j\s*[=≈~:]\s*" + _APPROX + _NUMBER + r"\s*ma[\s/·]*cm",
            re.IGNORECASE,
        ),
    ],
    "stability_h": [
        # "stability for 24 h" / "stability of 24 hours"
        re.compile(
            r"stability\s+(?:of|for|over)\s+" + _APPROX + _NUMBER + r"\s*(?:h\b|hours?)",
            re.IGNORECASE,
        ),
        # "operated for 24 h"
        re.compile(
            r"(?:operated|maintained|sustained|ran|running)\s+(?:for\s+)?" + _APPROX + _NUMBER + r"\s*(?:h\b|hours?)",
            re.IGNORECASE,
        ),
        # "24 h of continuous operation"
        re.compile(
            _APPROX + _NUMBER + r"\s*(?:h\b|hours?)\s+(?:of\s+)?(?:continuous|stable|long-term)",
            re.IGNORECASE,
        ),
        # "durability test for 100 h"
        re.compile(
            r"durability\s+(?:test\s+)?(?:for\s+)?" + _APPROX + _NUMBER + r"\s*(?:h\b|hours?)",
            re.IGNORECASE,
        ),
    ],
}


def _within_bounds(metric: str, value: float) -> bool:
    """Return True if value is within configured sanity bounds."""
    lo, hi = METRIC_BOUNDS.get(metric, (float("-inf"), float("inf")))
    return lo <= value <= hi


def extract_metrics(text: str) -> dict[str, float | None]:
    """Extract catalytic performance metrics from *text*.

    Returns
    -------
    dict[str, float | None]
        Keys are metric names; values are floats or None if not found /
        out of sanity bounds.
    """
    results: dict[str, float | None] = {
        "overpotential_mV": None,
        "faradaic_efficiency_pct": None,
        "tafel_slope_mV_dec": None,
        "current_density_mA_cm2": None,
        "stability_h": None,
    }

    for metric, patterns in _PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                try:
                    value = float(match.group("value"))
                except (ValueError, IndexError):
                    continue
                if _within_bounds(metric, value):
                    results[metric] = value
                    break  # first valid match per metric

    return results
