"""Rule-based application type classifier for catalysis abstracts."""

from __future__ import annotations

import re
from typing import Literal

ApplicationType = Literal[
    "OER", "HER", "CO2RR", "ORR", "NRR",
    "photocatalysis", "thermochemical",
    "water-splitting", "other",
]

# Each entry: (application_type, keyword patterns, weight)
_APPLICATION_KEYWORDS: list[tuple[ApplicationType, list[str], float]] = [
    ("OER", [
        "oxygen evolution reaction", "oxygen evolution",
        r"\boer\b", "anodic water oxidation", "water oxidation catalyst",
        "o2 evolution",
    ], 3.0),
    ("HER", [
        "hydrogen evolution reaction", "hydrogen evolution",
        r"\bher\b", "cathodic hydrogen", "h2 evolution",
    ], 3.0),
    ("CO2RR", [
        "co2 reduction", "co₂ reduction", "carbon dioxide reduction",
        r"\bco2rr\b", "co2 electroreduction", "co₂ electroreduction",
        "co2 conversion", "electrochemical co2",
    ], 3.0),
    ("ORR", [
        "oxygen reduction reaction", "oxygen reduction",
        r"\borr\b", "cathodic oxygen", "o2 reduction",
    ], 3.0),
    ("NRR", [
        "nitrogen reduction", "n2 reduction",
        r"\bnrr\b", "ammonia synthesis electro", "electrochemical nitrogen",
    ], 3.0),
    ("photocatalysis", [
        "photocatal", "photo-catal", "photodegradation",
        "visible light", "uv light", "solar-driven",
        "band gap", "bandgap", "photoanode", "photocathode",
        "photoelectrochemical", "pec ",
    ], 2.0),
    ("water-splitting", [
        "water splitting", "water-splitting", "overall water",
        "electrolysis", "electrolyzer", "electrolyser",
    ], 2.0),
    ("thermochemical", [
        "thermochemical", "chemical looping", "redox cycling",
        "oxygen carrier", "two-step", "solar thermochemical",
    ], 2.5),
]


def classify_application(text: str) -> tuple[ApplicationType, float]:
    """Classify the catalytic application described in *text*.

    Returns
    -------
    tuple[ApplicationType, float]
        (application_type, confidence_score).  Confidence is in [0, 1].
    """
    text_lower = text.lower()
    scores: dict[ApplicationType, float] = {}

    for app_type, patterns, weight in _APPLICATION_KEYWORDS:
        app_score = 0.0
        for pat in patterns:
            matches = re.findall(pat, text_lower)
            app_score += len(matches) * weight
        if app_score > 0:
            scores[app_type] = scores.get(app_type, 0.0) + app_score

    if not scores:
        return ("other", 0.0)

    best_app = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_app]
    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0

    return (best_app, round(confidence, 3))
