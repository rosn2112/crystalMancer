"""Rule-based synthesis method classifier.

Classifies abstracts into synthesis categories using keyword dictionaries
with priority ordering.  No LLM required.
"""

from __future__ import annotations

import re
from typing import Literal

SynthesisMethod = Literal[
    "hydrothermal",
    "sol-gel",
    "solid-state",
    "sputtering",
    "ALD",
    "electrodeposition",
    "coprecipitation",
    "combustion",
    "spray-pyrolysis",
    "other",
]

# Ordered by specificity — first match wins in priority tie-breaks.
# Each entry: (method_name, keywords, weight).
# Weight determines contribution to scoring; keywords found nearer the
# beginning of the abstract are slightly boosted.

_KEYWORD_MAP: list[tuple[SynthesisMethod, list[str], float]] = [
    ("ALD", [
        "atomic layer deposition", "ald", "ald-grown",
    ], 3.0),
    ("sputtering", [
        "sputtering", "sputter", "magnetron sputtering", "rf sputtering",
        "pulsed laser deposition", "pld",
    ], 3.0),
    ("electrodeposition", [
        "electrodeposition", "electrodeposited", "electroplating",
        "cathodic deposition", "anodic deposition",
    ], 3.0),
    ("hydrothermal", [
        "hydrothermal", "solvothermal", "autoclave",
        "teflon-lined", "teflon lined",
    ], 2.5),
    ("sol-gel", [
        "sol-gel", "sol gel", "citrate", "citric acid",
        "pechini", "chelating agent", "polymeric precursor",
    ], 2.5),
    ("coprecipitation", [
        "coprecipitation", "co-precipitation", "co precipitation",
        "precipitated",
    ], 2.0),
    ("combustion", [
        "combustion synthesis", "solution combustion",
        "self-propagating", "glycine-nitrate", "glycine nitrate",
    ], 2.5),
    ("spray-pyrolysis", [
        "spray pyrolysis", "spray-pyrolysis", "ultrasonic spray",
    ], 2.5),
    ("solid-state", [
        "solid-state", "solid state", "calcination", "calcined",
        "sintering", "sintered", "ball milling", "ball-milling",
        "mechanical milling", "high-temperature synthesis",
    ], 1.5),
]


def classify_synthesis(text: str) -> tuple[SynthesisMethod, float]:
    """Classify the synthesis method described in *text*.

    Returns
    -------
    tuple[SynthesisMethod, float]
        (method_name, confidence_score).  Confidence is in [0, 1].
        ``("other", 0.0)`` if no keywords match.
    """
    text_lower = text.lower()
    scores: dict[SynthesisMethod, float] = {}

    for method, keywords, weight in _KEYWORD_MAP:
        method_score = 0.0
        for kw in keywords:
            # Count occurrences
            count = len(re.findall(re.escape(kw), text_lower))
            if count > 0:
                # Position boost: earlier mentions get a small bonus
                first_pos = text_lower.index(kw)
                pos_factor = 1.0 + 0.2 * max(0, 1.0 - first_pos / max(len(text_lower), 1))
                method_score += count * weight * pos_factor
        if method_score > 0:
            scores[method] = scores.get(method, 0.0) + method_score

    if not scores:
        return ("other", 0.0)

    best_method = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_method]

    # Normalize confidence to [0, 1] range
    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0

    return (best_method, round(confidence, 3))
